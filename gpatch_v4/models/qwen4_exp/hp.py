# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""EP + FSDP2 parallelization for Qwen3.8-Flash-Next.

Public API: :func:`apply_hp` — bind CP, swap the QSA / EP / Engram leaves, shard experts,
wrap with FSDP2, and bind ``clip_grad_norm_`` / checkpoint methods.
"""
from __future__ import annotations

import types
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard

from gpatch_v4.core.dynamic_cp import DynamicCPState, bind_dynamic_cp_model

from .checkpoint import load_checkpoint_hp, save_checkpoint_hp
from .engram import (
    OwnerShardedNGramEmbedding,
    Qwen4ExpEngramEmbedding,
    materialize_engram_tables,
)
from .moe import Qwen4ExpEPExperts
from .qsa import QSA_ATTN_BACKENDS, Qwen4ExpQSAAttention

__all__ = [
    "apply_hp",
    "bind_dynamic_cp",
    "set_activation_checkpointing",
    "swap_parallel_modules",
]


def _new_engram_process_group() -> Optional[dist.ProcessGroup]:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return None
    return dist.new_group(ranks=list(range(dist.get_world_size())))


def apply_hp(
    model: nn.Module,
    ep_2d_mesh: "torch.distributed.device_mesh.DeviceMesh",
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    cp_mesh: Optional["torch.distributed.device_mesh.DeviceMesh"] = None,
    attn_backend: str = "flex",
    qsa_score_query_chunk_size: Optional[int] = None,
    qsa_compile_score: bool = False,
    ep_backend: str = "eager",
    deepep_num_sms: int = 24,
    deepep_async_finish: bool = False,
    ep_weight_before_combine: bool = False,
) -> nn.Module:
    """Shard experts for EP, apply FSDP2, and bind the HpModule methods.

    Parameters
    ----------
    model : nn.Module
        A meta-device :class:`Qwen4ExpHpForConditionalGeneration`, built under
        ``torch.set_default_dtype(torch.float32)`` (fp32 master).
    ep_2d_mesh : DeviceMesh
        2-D mesh with ``mesh_dim_names=("ep_fsdp", "ep")``, spanning the whole world.
    mp_policy : MixedPrecisionPolicy, optional
        Defaults to ``param_dtype=bf16, reduce_dtype=fp32`` — fp32 master, bf16 forward,
        fp32 gradient reduction.
    cp_mesh : DeviceMesh, optional
        Contiguous context-parallel mesh used by QSA, GatedDeltaNet, and PLE.
    attn_backend : str, default "flex"
        QSA backend, see :data:`.qsa.QSA_ATTN_BACKENDS`.
    qsa_score_query_chunk_size : int, optional
        QSA query rows scored together. ``None`` uses the full query length.
    qsa_compile_score : bool, default False
        Compile QSA score selection.
    ep_backend : str, default "eager"
        ``"eager"`` uses PyTorch uneven all-to-all; ``"deepep"`` uses DeepEP.
    deepep_num_sms : int, default 24
    deepep_async_finish : bool, default False
        Whether DeepEP dispatch/combine uses asynchronous communication completion.
    ep_weight_before_combine : bool, default False
        For eager EP, ``False`` preserves the original path, which multiplies
        routing weights after the reverse all-to-all combine. DeepEP always
        applies weights before its fused combine.

    Returns
    -------
    nn.Module
        The same model, wrapped.
    """
    if attn_backend not in QSA_ATTN_BACKENDS:
        raise ValueError(f"attn_backend must be one of {QSA_ATTN_BACKENDS}, got {attn_backend!r}.")
    if ep_backend not in ("eager", "deepep"):
        raise ValueError(f"unknown ep_backend: {ep_backend!r}")
    if ep_backend == "deepep":
        from .deepep_a2a import set_deepep_num_sms

        set_deepep_num_sms(deepep_num_sms)

    world_size = ep_2d_mesh.size()
    if dist.is_initialized() and world_size != dist.get_world_size():
        raise ValueError(
            f"ep_2d_mesh spans {world_size} ranks but the world has "
            f"{dist.get_world_size()}; the Engram table is owner-sharded over the whole "
            "world, so the mesh must cover it."
        )
    ep_mesh = ep_2d_mesh["ep"]
    ep_fsdp_mesh = ep_2d_mesh["ep_fsdp"]

    cp_size = cp_mesh.size() if cp_mesh is not None else 1
    if world_size % cp_size != 0:
        raise ValueError(
            f"world size {world_size} must be divisible by context parallel size {cp_size}"
        )
    model.config.attn_backend = attn_backend
    model.config.qsa_score_query_chunk_size = qsa_score_query_chunk_size
    model.config.qsa_compile_score = qsa_compile_score
    model.config.ep_backend = ep_backend
    model.config.deepep_num_sms = deepep_num_sms
    model.config.ep_weight_before_combine = ep_weight_before_combine

    # FSDP2 async all-gather on WORLD vs Engram blocking all-to-all: sharing WORLD can deadlock.
    engram_group = _new_engram_process_group()
    swap_parallel_modules(
        model,
        attn_backend=attn_backend,
        qsa_score_query_chunk_size=qsa_score_query_chunk_size,
        qsa_compile_score=qsa_compile_score,
        ep_backend=ep_backend,
        deepep_async_finish=deepep_async_finish,
        ep_weight_before_combine=ep_weight_before_combine,
        ep_group=ep_mesh.get_group(),
        engram_group=engram_group,
    )
    if cp_size > 1:
        _bind_cp(model, cp_mesh)
    text_model = model.model.language_model
    layers = list(text_model.layers)
    mtp_layers = [] if model.mtp is None else list(model.mtp.layers)
    _assert_fp32_master(model)

    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        )

    with torch.no_grad():
        for layer in [*layers, *mtp_layers]:
            shard_experts_for_ep(layer.mlp.experts, ep_mesh)

    # The Engram table is excluded from FSDP2 below, so FSDP never applies the
    # mixed-precision cast to it. Every downstream consumer *is* cast to
    # ``mp_policy.param_dtype``, so the table has to cast its own output or the first
    # projection hits a dtype mismatch. Casting the looked-up values (not the table)
    # keeps the fp32 master for the optimizer.
    for module in model.modules():
        if isinstance(module, Qwen4ExpEngramEmbedding):
            module.ngram_embedding.output_dtype = mp_policy.param_dtype

    # ignored_params stay on meta through fully_shard; allocate the local shard
    # here so load_checkpoint_hp can copy_ and forward does not .to(meta).
    if ep_2d_mesh.device_type == "cpu":
        engram_device = torch.device("cpu")
    else:
        engram_device = torch.device(ep_2d_mesh.device_type, torch.cuda.current_device())
    materialize_engram_tables(model, engram_device, torch.float32)

    engram_params = _engram_parameters(model)

    # Effective number of independent samples. CP ranks see slices of the same sample, so
    # the FSDP mean must be over world/cp rather than world.
    grad_divide_factor = world_size // cp_size

    for layer in layers:
        # Experts reduce-scatter over ``ep_fsdp`` (each expert lives on one ``ep`` index),
        # everything else over the global mesh.
        fully_shard(layer.mlp.experts, mesh=ep_fsdp_mesh, mp_policy=mp_policy)
        layer.mlp.experts.set_gradient_divide_factor(grad_divide_factor)
        # The Engram shard is already partitioned by row ownership — FSDP must not shard
        # it a second time.
        fully_shard(layer, mp_policy=mp_policy, ignored_params=_engram_parameters(layer) or None)
        layer.set_gradient_divide_factor(grad_divide_factor)
        layer.set_modules_to_forward_prefetch([layer.mlp.experts])

    for layer in mtp_layers:
        fully_shard(layer.mlp.experts, mesh=ep_fsdp_mesh, mp_policy=mp_policy)
        layer.mlp.experts.set_gradient_divide_factor(grad_divide_factor)
        fully_shard(layer, mp_policy=mp_policy)
        layer.set_gradient_divide_factor(grad_divide_factor)
    if model.mtp is not None:
        fully_shard(model.mtp, mp_policy=mp_policy)
        model.mtp.set_gradient_divide_factor(grad_divide_factor)

    ignored_params = engram_params
    vision_model = model.model.visual
    ignored_params.update(vision_model.parameters())
    # FSDP ignores the fp32 vision master weights, so cast its compute explicitly.
    vision_model.forward = torch.autocast(
        device_type=ep_2d_mesh.device_type,
        dtype=mp_policy.param_dtype,
        enabled=mp_policy.param_dtype in (torch.float16, torch.bfloat16),
    )(vision_model.forward)
    model._vision_grad_divide_factor = grad_divide_factor

    fully_shard(text_model.embed_tokens, mp_policy=mp_policy)
    text_model.embed_tokens.set_gradient_divide_factor(grad_divide_factor)
    fully_shard(model.lm_head, mp_policy=mp_policy)
    model.lm_head.set_gradient_divide_factor(grad_divide_factor)
    fully_shard(model, mp_policy=mp_policy, ignored_params=ignored_params)
    model.set_gradient_divide_factor(grad_divide_factor)

    # After `fully_shard`, so FSDP receives plain tensors in `ignored_params` (its
    # well-tested path) and only the optimizer built later sees the DTensor.
    _wrap_engram_as_dtensor(model, world_size, grad_divide_factor)

    model._ep_size = ep_mesh.size()
    model._ep_rank = ep_mesh.get_local_rank()
    model._ep_group = ep_mesh.get_group()
    model._ep_fsdp_mesh = ep_fsdp_mesh
    model._ep_2d_mesh = ep_2d_mesh
    model._engram_param_ids = {id(p) for p in _engram_parameters(model)}
    model._step0_diagnostics_dir = None
    model._step0_diagnostics_clip_index = 0
    model.clip_grad_norm_ = types.MethodType(_clip_grad_norm_multi_mesh, model)
    model.load_checkpoint_hp = types.MethodType(load_checkpoint_hp, model)
    model.save_checkpoint_hp = types.MethodType(save_checkpoint_hp, model)
    return model


def swap_parallel_modules(
    model: nn.Module,
    attn_backend: str = "flex",
    qsa_score_query_chunk_size: Optional[int] = None,
    qsa_compile_score: bool = False,
    ep_backend: str = "eager",
    ep_group: Optional[dist.ProcessGroup] = None,
    engram_group: Optional[dist.ProcessGroup] = None,
    deepep_async_finish: bool = False,
    ep_weight_before_combine: bool = False,
) -> nn.Module:
    """Replace the three parallelism-sensitive leaves with gcore's versions.

    ``self_attn`` -> :class:`.qsa.Qwen4ExpQSAAttention` (batched selector + FlexAttention),
    ``mlp.experts`` -> :class:`.moe.Qwen4ExpEPExperts` (all-to-all EP),
    ``ple.ple_embedding`` -> :class:`.engram.Qwen4ExpEngramEmbedding` (owner-sharded table).

    Parameter names are unchanged in every case, so checkpoint FQNs still apply. Real
    (non-meta) weights are carried over, which is what makes this testable on CPU;
    the Engram table is row-sliced to this rank's ownership range as it is copied.

    Replacements are constructed on the **same device as the model**, which for the real
    flow is ``meta``. Building them eagerly would allocate a full 51.2 B n-gram table plus
    ~6.7 GB of fp32 experts per layer before immediately discarding most of it.

    Layer roles come from ``config.layer_types`` and ``config.ple_layer_ids`` rather than
    ``hasattr`` probing, so a layout change fails loudly instead of silently skipping.
    """
    config = model.config.get_text_config()
    layers = list(model.model.language_model.layers)
    if len(config.layer_types) != len(layers):
        raise ValueError(
            f"config.layer_types has {len(config.layer_types)} entries but the model has "
            f"{len(layers)} layers."
        )
    # `ple_layer_ids` is one-indexed upstream.
    ple_indices = {layer_id - 1 for layer_id in config.ple_layer_ids}
    device = _module_device(model)

    for layer_idx, (layer, layer_type) in enumerate(zip(layers, config.layer_types)):
        if layer_type not in ("qwen_sparse_attention", "linear_attention"):
            raise ValueError(f"unexpected layer_type {layer_type!r} at layer {layer_idx}")

        if layer_type == "qwen_sparse_attention":
            with torch.device(device):
                replacement = Qwen4ExpQSAAttention(
                    config,
                    layer_idx,
                    attn_backend=attn_backend,
                    score_query_chunk_size=qsa_score_query_chunk_size,
                    compile_score=qsa_compile_score,
                )
            layer.self_attn = _transplant(layer.self_attn, replacement)

        with torch.device(device):
            experts = Qwen4ExpEPExperts(
                config,
                ep_backend=ep_backend,
                deepep_async_finish=deepep_async_finish,
                ep_weight_before_combine=ep_weight_before_combine,
            )
        experts._layer_idx = layer_idx
        layer.mlp.experts = _transplant(layer.mlp.experts, experts)
        layer.mlp.experts.configure_ep(ep_group)

        if layer_idx in ple_indices:
            old = layer.ple.ple_embedding
            layer.ple.ple_embedding = _replace_engram(config, old, engram_group, device)

    if model.mtp is not None:
        mtp_config = model.mtp.config
        for layer_idx, layer in enumerate(model.mtp.layers):
            with torch.device(device):
                replacement = Qwen4ExpQSAAttention(
                    mtp_config,
                    layer_idx,
                    attn_backend=attn_backend,
                    score_query_chunk_size=qsa_score_query_chunk_size,
                    compile_score=qsa_compile_score,
                )
            layer.self_attn = _transplant(layer.self_attn, replacement)
            with torch.device(device):
                experts = Qwen4ExpEPExperts(
                    mtp_config,
                    ep_backend=ep_backend,
                    deepep_async_finish=deepep_async_finish,
                    ep_weight_before_combine=ep_weight_before_combine,
                )
            experts._layer_idx = layer_idx
            layer.mlp.experts = _transplant(layer.mlp.experts, experts)
            layer.mlp.experts.configure_ep(ep_group)

    return model


def _bind_cp(
    model: nn.Module,
    cp_mesh: "torch.distributed.device_mesh.DeviceMesh",
) -> None:
    cp_group = cp_mesh.get_group()
    cp_size = cp_mesh.size()
    cp_rank = cp_mesh.get_local_rank()

    _set_dynamic_cp_state(model, DynamicCPState(cp_group, cp_size, cp_rank))
    model._cp_mesh = cp_mesh


def bind_dynamic_cp(
    model: nn.Module,
    cp_group: dist.ProcessGroup,
):
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
    model._cp_group = state.group
    model._cp_size = state.size
    model._cp_rank = state.rank
    for layer in model.model.language_model.layers:
        layer.cp_size = state.size


def _module_device(module: nn.Module) -> torch.device:
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    raise ValueError("cannot determine module device: no parameters or buffers")


def set_activation_checkpointing(
    model: nn.Module,
    enabled: bool = True,
    selective_recompute: bool = False,
    selective_recompute_max_layers: Optional[int] = None,
) -> None:
    """Toggle per-layer activation checkpointing, keeping the Engram layer eager.

    The Engram-owning layer must run eagerly: recomputation would replay its
    owner-sharded all-to-all collectives, which is not supported (NVIDIA hit the same
    wall and exempts the same single block, checkpointing 47 of 48).

    With ``selective_recompute=True``, GDN layers checkpoint the attention prefix and
    local expert GEMMs, while MoE dispatch/combine stay outside the checkpoint boundary
    so EP communication is not replayed during backward. QSA layers retain full-layer
    checkpointing because FlexAttention backward has a large transient workspace; this
    bounds the peak memory while still avoiding communication replay on the 36 GDN
    layers. ``selective_recompute_max_layers`` can bound the number of GDN layers using
    this path when retaining all DeepEP outputs exceeds the available memory.

    Call **after** ``model.gradient_checkpointing_enable(...)``, which sets the flag on
    every :class:`GradientCheckpointingLayer`.
    """
    if selective_recompute_max_layers is not None and selective_recompute_max_layers < 0:
        raise ValueError(
            "selective_recompute_max_layers must be non-negative or None, "
            f"got {selective_recompute_max_layers}"
        )
    config = model.config.get_text_config()
    ple_indices = {layer_id - 1 for layer_id in config.ple_layer_ids}
    selective_candidates = [
        layer_idx for layer_idx, layer in enumerate(model.model.language_model.layers)
        if enabled and selective_recompute and layer_idx not in ple_indices and
        layer.layer_type == "linear_attention"
    ]
    if selective_recompute_max_layers is None:
        selected_selective_layers = set(selective_candidates)
    else:
        selected_selective_layers = set(selective_candidates[:selective_recompute_max_layers])
    for layer_idx, layer in enumerate(model.model.language_model.layers):
        is_checkpointed_layer = enabled and layer_idx not in ple_indices
        is_qsa_layer = layer.layer_type != "linear_attention"
        is_selective_layer = layer_idx in selected_selective_layers
        is_ple_selective = (
            enabled and selective_recompute and layer_idx in ple_indices and not is_qsa_layer
        )
        is_full_checkpointed_layer = is_checkpointed_layer and not is_selective_layer
        layer.gradient_checkpointing = is_full_checkpointed_layer
        layer.selective_gradient_checkpointing = is_selective_layer
        mlp = layer.mlp
        # The shared expert has no communication and can be recomputed safely.
        # Keep this enabled on the Engram layer too: only the PLE path is excluded
        # from recomputation there.
        mlp.selective_gradient_checkpointing = is_selective_layer or is_ple_selective
        # The callable exists only after gradient_checkpointing_enable().
        if mlp.selective_gradient_checkpointing:
            mlp._gradient_checkpointing_func = layer._gradient_checkpointing_func

        experts = mlp.experts
        experts.selective_gradient_checkpointing = is_selective_layer or is_ple_selective
        if experts.selective_gradient_checkpointing:
            experts._gradient_checkpointing_func = layer._gradient_checkpointing_func


def _transplant(old: nn.Module, new: nn.Module) -> nn.Module:
    """Move ``old``'s tensors into ``new``, which must have an identical state dict.

    Meta tensors are left alone — the real values arrive later from
    ``load_checkpoint_hp``.
    """
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
    for name, param in old.named_parameters():
        replacement = new.get_parameter(name)
        replacement.requires_grad_(replacement.requires_grad and param.requires_grad)
    return new


def _replace_engram(
    config,
    old: nn.Module,
    engram_group: Optional[dist.ProcessGroup],
    device: torch.device,
) -> Qwen4ExpEngramEmbedding:
    """Swap the dense n-gram table for the owner-sharded one, slicing this rank's rows."""
    with torch.device(device):
        new = Qwen4ExpEngramEmbedding(
            config,
            config.ple_embed_dim,
            old.layer_idx,
            old.ple_layer_index,
            process_group=engram_group,
        )
    old_weight = old.ngram_embedding.weight
    new.ngram_embedding.weight.requires_grad_(old_weight.requires_grad)
    if not old_weight.is_meta:
        table = new.ngram_embedding
        start = table.global_row_start
        with torch.no_grad():
            table.weight.copy_(old_weight[start:start + table.local_rows])
    return new


def _wrap_engram_as_dtensor(
    model: nn.Module,
    world_size: int,
    gradient_divide_factor: int,
) -> None:
    """Re-wrap each Engram shard as a global ``DTensor(Shard(0))`` over all ranks.

    The table is excluded from FSDP2 (it is already partitioned by row ownership), which
    left it a plain ``nn.Parameter`` while every other parameter became a DTensor. AdamW
    defaults to ``foreach=True`` and batches same-device/dtype parameters into one
    ``torch._foreach_mul_`` call, which rejects a mix of DTensor and plain tensors:

        RuntimeError: aten._foreach_mul_.Scalar: got mixed torch.Tensor and DTensor

    Making it a DTensor also means ``clip_grad_norm_`` reduces its gradient norm through
    normal DTensor semantics instead of a hand-rolled all-reduce, and gives DCP a global
    FQN it can reshard later. Lookups go through
    :attr:`OwnerShardedNGramEmbedding.local_weight`.

    The lookup backward already routes and sums every requester's gradient on the row
    owner. Unlike FSDP parameters, this ignored parameter has no reduce-scatter that
    cancels the loss function's ``gradient_divide_factor`` multiplier, so its hook must
    apply that division explicitly.

    Skipped only when distributed is not initialized (single-process CPU tests). Note it
    still applies at ``world_size == 1``: ``fully_shard`` produces DTensors even on a
    one-rank mesh, so the type mismatch would occur there too.
    """
    if not dist.is_initialized():
        return

    tables = [
        module for module in model.modules() if isinstance(module, OwnerShardedNGramEmbedding)
    ]
    if not tables:
        return

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    engram_mesh = init_device_mesh(device_type, (world_size, ), mesh_dim_names=("engram", ))
    for table in tables:
        if isinstance(table.weight, DTensor):
            continue
        local = table.weight
        table.weight = nn.Parameter(
            DTensor.from_local(local.data, device_mesh=engram_mesh, placements=[Shard(0)]),
            requires_grad=local.requires_grad,
        )
        if table.weight.requires_grad and gradient_divide_factor != 1:

            def divide_owner_gradient(grad: torch.Tensor) -> torch.Tensor:
                return grad / gradient_divide_factor

            table.weight.register_hook(divide_owner_gradient)


def _engram_parameters(module: nn.Module) -> set:
    """The Engram table parameters under ``module``, which FSDP2 must leave alone."""
    return {
        submodule.ngram_embedding.weight
        for submodule in module.modules() if isinstance(submodule, Qwen4ExpEngramEmbedding)
    }


def _assert_fp32_master(model: nn.Module) -> None:
    """Fail fast unless every floating-point parameter is fp32.

    The default ``mp_policy(param_dtype=bf16, reduce_dtype=fp32)`` only behaves as
    intended with an fp32 master: the forward casts down to bf16 and gradients reduce in
    fp32. A bf16 master makes the forward cast a no-op. Build under
    ``torch.set_default_dtype(torch.float32)``.
    """
    for name, param in model.named_parameters():
        if param.dtype.is_floating_point and param.dtype != torch.float32:
            raise ValueError(
                f"apply_hp requires fp32 master weights, but {name}.dtype={param.dtype}. "
                "Build the model under `torch.set_default_dtype(torch.float32)`."
            )


def shard_experts_for_ep(
    experts: nn.Module,
    ep_mesh: "torch.distributed.device_mesh.DeviceMesh",
) -> None:
    """Slice stacked expert weights along dim 0 so each rank keeps its own experts.

    Handles meta tensors by allocating a meta placeholder of the local shape;
    ``load_checkpoint_hp`` fills in the real values.
    """
    ep_size = ep_mesh.size()
    if experts.num_experts % ep_size != 0:
        raise ValueError(f"num_experts={experts.num_experts} not divisible by ep_size={ep_size}")
    num_local = experts.num_experts // ep_size

    for name in ("gate_up_proj", "down_proj"):
        param = getattr(experts, name)
        if param.shape[0] == num_local:
            continue
        local_shape = (num_local, ) + tuple(param.shape[1:])
        if param.is_meta:
            local = torch.empty(local_shape, dtype=param.dtype, device="meta")
        else:
            replicated = DTensor.from_local(
                param.data, device_mesh=ep_mesh, placements=[Replicate()]
            )
            sharded = replicated.redistribute(device_mesh=ep_mesh, placements=[Shard(0)])
            # ``clone()`` cuts the view chain so the full tensor's storage is released.
            local = sharded.to_local().contiguous().clone()
        setattr(experts, name, nn.Parameter(local, requires_grad=param.requires_grad))

    experts.configure_ep(ep_mesh.get_group())


def _clip_grad_norm_multi_mesh(
    self: nn.Module,
    max_norm: float = 2.0,
) -> float:
    """``clip_grad_norm_`` across parameters living on different meshes.

    Parameters live on different meshes, and ``clip_grad_norm_`` cannot stack DTensors
    across meshes, so we take a sub-norm per mesh and combine:

    * expert params — sharded on the ``ep_fsdp`` sub-mesh. Each expert lives on a single
      ``ep`` index, so their squared norm additionally needs an all-reduce over
      ``ep_group`` to cover every expert.
    * replicated vision params — synchronized before clipping and counted once.
      This method must be called exactly once per optimizer step to avoid reducing
      these gradients twice.
    * other parameters (including the Engram table, a ``Shard(0)`` DTensor over the world
      mesh) — DTensor semantics already reduce within the mesh.

    Adapted from DSV4's version. The Engram needed a hand-rolled all-reduce back when it
    was a plain local tensor; now that ``apply_hp`` wraps it as a DTensor it just falls
    into its own mesh group, and adding a manual reduce on top would **double-count** it.
    """
    vision_params = [param for param in self.model.visual.parameters() if param.requires_grad]
    if vision_params:
        has_gradient = torch.tensor(
            any(param.grad is not None for param in vision_params),
            device=vision_params[0].device,
            dtype=torch.int32,
        )
        dist.all_reduce(has_gradient, op=dist.ReduceOp.MAX)
        # Keep globally unused parameters out of AdamW's momentum and weight decay updates.
        if has_gradient.item():
            for param in vision_params:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                dist.all_reduce(param.grad)
                param.grad.div_(self._vision_grad_divide_factor)

    ep_fsdp_key = str(self._ep_fsdp_mesh) if self._ep_fsdp_mesh is not None else None

    groups: dict[str, list] = {}
    for param in self.parameters():
        if param.grad is None:
            continue
        key = str(param.device_mesh) if isinstance(param, DTensor) else "default"
        groups.setdefault(key, []).append(param)

    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    non_expert_sq = torch.zeros((), device=device)
    expert_sq = torch.zeros((), device=device)
    group_norms = {}

    for key, params in groups.items():
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))
        if isinstance(norm, DTensor):
            norm = norm.full_tensor()
        group_norms[key] = norm.detach()
        if key == ep_fsdp_key:
            expert_sq = expert_sq + norm**2
        else:
            non_expert_sq = non_expert_sq + norm**2

    local_expert_sq = expert_sq.detach().clone()
    if self._ep_group is not None:
        dist.all_reduce(expert_sq, group=self._ep_group)

    total_norm = (non_expert_sq + expert_sq).sqrt()
    if (self._step0_diagnostics_dir is not None and self._step0_diagnostics_clip_index == 0):
        from .diagnostics import write_diagnostic_record

        write_diagnostic_record(
            self._step0_diagnostics_dir,
            "grad_norm",
            {
                "group_norms": {
                    key: value.item()
                    for key, value in group_norms.items()
                },
                "local_expert_norm_sq": local_expert_sq.item(),
                "global_expert_norm_sq": expert_sq.item(),
                "non_expert_norm_sq": non_expert_sq.item(),
                "total_norm": total_norm.item(),
            },
        )
    self._step0_diagnostics_clip_index += 1
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for param in self.parameters():
            if param.grad is not None:
                param.grad.detach().mul_(clip_coef)
    return total_norm.item()
