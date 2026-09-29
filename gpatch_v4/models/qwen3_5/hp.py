# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""FSDP2 + zigzag CP parallelization for Qwen3.5 dense VL.

Text decoder layers / embeddings are FSDP-sharded. The ViT stays a full
replica on every rank; image tokens are gathered after the encoder.
"""
from __future__ import annotations

import types
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from gpatch_v4.core.dynamic_cp import DynamicCPState, bind_dynamic_cp_model
from gpatch_v4.models.hp_module import HpModule

from .attn import Qwen3_5CPAttention
from .checkpoint import _is_tied_lm_head, load_checkpoint_hp, save_checkpoint_hp
from .gdn import Qwen3_5CPGatedDeltaNet
from .modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

__all__ = [
    "Qwen3_5HpForConditionalGeneration",
    "apply_hp",
    "bind_dynamic_cp",
    "swap_parallel_modules",
    "sync_visual_grads",
]


class Qwen3_5HpForConditionalGeneration(Qwen3_5ForConditionalGeneration, HpModule):
    """Select GCore's hybrid-parallel construction path for Qwen3.5 VL."""


def apply_hp(
    model: nn.Module,
    ep_2d_mesh: "torch.distributed.device_mesh.DeviceMesh",
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    cp_mesh: Optional["torch.distributed.device_mesh.DeviceMesh"] = None,
    attn_backend: str = "sdpa",
    linear_cp_mode: str = "headwise",
) -> nn.Module:
    """Apply FSDP2 to the text network and bind zigzag CP. ViT is not sharded."""
    world_size = ep_2d_mesh.size()
    if dist.is_initialized() and world_size != dist.get_world_size():
        raise ValueError(
            f"ep_2d_mesh spans {world_size} ranks but the world has "
            f"{dist.get_world_size()}"
        )
    ep_size = ep_2d_mesh["ep"].size()
    if ep_size != 1:
        raise ValueError(f"qwen3_5 dense requires ep_size=1, got {ep_size}")
    ep_fsdp_mesh = ep_2d_mesh["ep_fsdp"]

    cp_size = cp_mesh.size() if cp_mesh is not None else 1
    if world_size % cp_size != 0:
        raise ValueError(
            f"world size {world_size} must be divisible by context parallel size {cp_size}"
        )
    text_config = model.config.get_text_config()
    if cp_size > 1 and linear_cp_mode == "headwise":
        if text_config.linear_num_key_heads % cp_size != 0:
            raise ValueError(
                f"linear_num_key_heads={text_config.linear_num_key_heads} "
                f"must be divisible by CP size {cp_size}"
            )
        if text_config.linear_num_value_heads % cp_size != 0:
            raise ValueError(
                f"linear_num_value_heads={text_config.linear_num_value_heads} "
                f"must be divisible by CP size {cp_size}"
            )
    model.config._attn_implementation = attn_backend
    text_config._attn_implementation = attn_backend

    swap_parallel_modules(model)
    model._linear_cp_mode = linear_cp_mode
    for layer in model.model.language_model.layers:
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.linear_cp_comm_type = linear_cp_mode
    if cp_size > 1:
        _bind_cp(model, cp_mesh, linear_cp_mode=linear_cp_mode)
    language_model = model.model.language_model
    layers = list(language_model.layers)
    _assert_fp32_master(model)

    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )

    grad_divide_factor = world_size // cp_size
    fsdp_kwargs = {"mp_policy": mp_policy, "mesh": ep_fsdp_mesh}
    for layer in layers:
        fully_shard(layer, **fsdp_kwargs)
        layer.set_gradient_divide_factor(grad_divide_factor)

    # embed / lm_head / norm are called outside language_model.forward, so they
    # get their own fully_shard. Do not wrap language_model itself.
    # Tied embed/lm_head share one Parameter; FSDP2 requires them in one group.
    if _is_tied_lm_head(model):
        fully_shard([language_model.embed_tokens, model.lm_head], **fsdp_kwargs)
    else:
        fully_shard(language_model.embed_tokens, **fsdp_kwargs)
        fully_shard(model.lm_head, **fsdp_kwargs)
    language_model.embed_tokens.set_gradient_divide_factor(grad_divide_factor)
    model.lm_head.set_gradient_divide_factor(grad_divide_factor)
    fully_shard(language_model.norm, **fsdp_kwargs)
    language_model.norm.set_gradient_divide_factor(grad_divide_factor)
    # Each fully_shard'd module is an FSDP root unless nested under another
    # fully_shard. Root sets reshard_after_forward=False, so wrapping only
    # layers keeps the whole decoder unsharded after forward (~bf16 full
    # weights on every rank). Wrap the VL root and ignore the ViT replica.
    visual_params = {
        param
        for name, param in model.named_parameters() if name.startswith("model.visual.")
    }
    fully_shard(model, **fsdp_kwargs, ignored_params=visual_params or None)
    model.set_gradient_divide_factor(grad_divide_factor)

    model._ep_size = 1
    model._ep_rank = 0
    model._ep_group = None
    model._ep_fsdp_mesh = ep_fsdp_mesh
    model._ep_2d_mesh = ep_2d_mesh
    model._grad_divide_factor = grad_divide_factor
    model.sync_visual_grads = types.MethodType(sync_visual_grads, model)
    model.clip_grad_norm_ = types.MethodType(_clip_grad_norm_multi_mesh, model)
    model.load_checkpoint_hp = types.MethodType(load_checkpoint_hp, model)
    model.save_checkpoint_hp = types.MethodType(save_checkpoint_hp, model)
    return model


def swap_parallel_modules(model: nn.Module) -> nn.Module:
    """Replace attention and GDN leaves with the parallel versions."""
    config = model.config.get_text_config()
    layers = list(model.model.language_model.layers)
    if len(config.layer_types) != len(layers):
        raise ValueError(
            f"config.layer_types has {len(config.layer_types)} entries but the model has "
            f"{len(layers)} layers."
        )
    device = _module_device(model)

    for layer_idx, (layer, layer_type) in enumerate(zip(layers, config.layer_types)):
        if layer_type not in ("full_attention", "linear_attention"):
            raise ValueError(f"unexpected layer_type {layer_type!r} at layer {layer_idx}")

        if layer_type == "full_attention":
            with torch.device(device):
                replacement = Qwen3_5CPAttention(config, layer_idx)
            layer.self_attn = _transplant(layer.self_attn, replacement)
            with torch.device(device):
                layer.self_attn.bind_te()
        else:
            with torch.device(device):
                replacement = Qwen3_5CPGatedDeltaNet(config, layer_idx)
            layer.linear_attn = _transplant(layer.linear_attn, replacement)

    return model


def _bind_cp(
    model: nn.Module,
    cp_mesh: "torch.distributed.device_mesh.DeviceMesh",
    linear_cp_mode: str = "headwise",
) -> None:
    assert linear_cp_mode in ("headwise", "chunkwise"), (
        f"linear_cp_mode must be 'headwise' or 'chunkwise', got {linear_cp_mode!r}"
    )
    cp_group = cp_mesh.get_group()
    cp_size = cp_mesh.size()
    cp_rank = cp_mesh.get_local_rank()

    for module in (model, model.model):
        module._cp_group = cp_group
        module._cp_size = cp_size
        module._cp_rank = cp_rank
        module._cp_mesh = cp_mesh
    for layer in model.model.language_model.layers:
        layer.cp_size = cp_size
        if hasattr(layer, "self_attn"):
            layer.self_attn.bind_cp(cp_group, cp_size)
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.linear_cp_comm_type = linear_cp_mode


def bind_dynamic_cp(model: nn.Module, cp_group: dist.ProcessGroup):
    return bind_dynamic_cp_model(
        model,
        cp_group,
        _get_dynamic_cp_state,
        _set_dynamic_cp_state,
    )


def _get_dynamic_cp_state(model: nn.Module) -> DynamicCPState:
    return DynamicCPState(
        getattr(model, "_cp_group", None),
        getattr(model, "_cp_size", 1),
        getattr(model, "_cp_rank", 0),
    )


def _set_dynamic_cp_state(model: nn.Module, state: DynamicCPState) -> None:
    linear_cp_mode = getattr(model, "_linear_cp_mode", "headwise")
    for module in (model, model.model):
        module._cp_group = state.group
        module._cp_size = state.size
        module._cp_rank = state.rank
    for layer in model.model.language_model.layers:
        layer.cp_size = state.size
        if hasattr(layer, "self_attn"):
            layer.self_attn.bind_cp(state.group, state.size)
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.linear_cp_comm_type = linear_cp_mode


def _module_device(module: nn.Module) -> torch.device:
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    raise ValueError("cannot determine module device: no parameters or buffers")


def _transplant(old: nn.Module, new: nn.Module) -> nn.Module:
    old_state = old.state_dict()
    new_state = new.state_dict()
    if set(old_state) != set(new_state):
        raise ValueError(
            "module replacement changed the state dict keys: "
            f"missing={sorted(set(old_state) - set(new_state))}, "
            f"unexpected={sorted(set(new_state) - set(old_state))}"
        )
    if not any(tensor.is_meta for tensor in old_state.values()):
        new.load_state_dict(old_state)
    return new


def _assert_fp32_master(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.dtype.is_floating_point and param.dtype != torch.float32:
            raise ValueError(
                f"apply_hp requires fp32 master weights, but {name}.dtype={param.dtype}. "
                "Build the model under `torch.set_default_dtype(torch.float32)`."
            )


def sync_visual_grads(model: nn.Module) -> None:
    """All-reduce ViT grads across the world and divide by DP size.

    ViT is FSDP-ignored, so it has no ``set_gradient_divide_factor``. Ranks
    that did not encode images still participate with a zero grad; skipping
    the collective on those ranks deadlocks the rest.
    """
    dp_size = model._grad_divide_factor
    for name, param in model.named_parameters():
        if not name.startswith("model.visual."):
            continue
        if isinstance(param, DTensor):
            raise RuntimeError(f"visual parameter {name} is a DTensor; ViT must stay unsharded")
        if not param.requires_grad:
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(dp_size)


def _clip_grad_norm_multi_mesh(
    self: nn.Module,
    max_norm: float = 2.0,
) -> float:
    self.sync_visual_grads()

    groups: dict[str, list] = {}
    for param in self.parameters():
        if param.grad is None:
            continue
        key = str(param.device_mesh) if isinstance(param, DTensor) else "replicated"
        groups.setdefault(key, []).append(param)

    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    total_sq = torch.zeros((), device=device)
    for params in groups.values():
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))
        if isinstance(norm, DTensor):
            norm = norm.full_tensor()
        total_sq = total_sq + norm**2

    total_norm = total_sq.sqrt()
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for param in self.parameters():
            if param.grad is not None:
                param.grad.detach().mul_(clip_coef)
    return total_norm.item()
