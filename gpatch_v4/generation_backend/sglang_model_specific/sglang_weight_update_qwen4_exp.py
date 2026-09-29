# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Packed Qwen4-Exp expert and pinned PLE reload for SGLang."""

from __future__ import annotations

import re
from typing import Any

import torch

QWEN4_EXP_PACKED_WEIGHT_LOADER = (
    "gpatch_v4.generation_backend.sglang_model_specific."
    "sglang_weight_update_qwen4_exp.load_qwen4_exp_packed_weights"
)
_PACKED_EXPERT_MAPPINGS = (
    (".mlp.experts.gate_up_proj", ".mlp.experts.w13_weight", "w13"),
    (".mlp.experts.down_proj", ".mlp.experts.w2_weight", "w2"),
)
_PLE_SHARD_RE = re.compile(r"^(.*\.ngram_embedding)\.shard_(\d+)\.weight$")


def install_packed_weight_loader(server_args: dict[str, Any]) -> None:
    """Append the packed loader, copying any caller-owned loader list.

    Parameters
    ----------
    server_args : dict
        ``custom_weight_loader`` may be absent, a path string, or a sequence of paths.
    """
    configured = server_args.get("custom_weight_loader")
    if configured is None:
        loaders: list[str] = []
    elif isinstance(configured, str):
        loaders = [configured]
    else:
        loaders = list(configured)
    if QWEN4_EXP_PACKED_WEIGHT_LOADER not in loaders:
        loaders.append(QWEN4_EXP_PACKED_WEIGHT_LOADER)
    server_args["custom_weight_loader"] = loaders


def _has_identity_expert_locations(metadata: Any, num_experts: int) -> bool:
    if metadata is None:
        return True
    if metadata.ep_size != 1:
        return False
    locations = metadata.physical_to_logical_map_cpu
    if locations.ndim != 2 or locations.shape[1] != num_experts:
        return False
    identity = torch.arange(num_experts, device=locations.device, dtype=locations.dtype)
    return bool(torch.all(locations == identity).item())


def _normalize_qwen4_exp_weight_name(name: str) -> str:
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    return name


def _packed_expert_target(name: str) -> tuple[str, str] | None:
    normalized = _normalize_qwen4_exp_weight_name(name)
    for source, target, shard_id in _PACKED_EXPERT_MAPPINGS:
        if source in normalized:
            return normalized.replace(source, target, 1), shard_id
    return None


def _load_packed_expert(
    model: torch.nn.Module,
    params: dict[str, torch.nn.Parameter],
    name: str,
    weight: torch.Tensor,
) -> bool:
    target = _packed_expert_target(name)
    if target is None:
        return False
    target_name, shard_id = target
    param = params.get(target_name)
    if param is None:
        return False

    experts = model.get_submodule(target_name.rsplit(".", 1)[0])
    weight_loader_fused = getattr(experts, "weight_loader_fused", None)
    if (
        weight_loader_fused is None or experts.quant_config is not None or
        experts.moe_ep_size != 1 or experts.num_local_experts != weight.shape[0] or
        experts.use_presharded_weights or not experts.moe_runner_config.is_gated or
        param.ndim != 3 or weight.ndim != 3 or weight.dtype != torch.bfloat16 or
        param.dtype != weight.dtype
    ):
        return False

    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
    from sglang.srt.layers.moe.utils import get_moe_runner_backend

    backend = get_moe_runner_backend()
    if (
        not _has_identity_expert_locations(
            get_global_expert_location_metadata(), experts.num_local_experts
        ) or not (backend.is_triton() or backend.is_triton_kernels()) or experts.use_padded_loading
    ):
        return False

    if shard_id == "w13":
        tp_size = experts.moe_tp_size
        tp_rank = experts.moe_tp_rank
        if tp_size <= 0 or weight.shape[1] % (2 * tp_size) != 0 or not 0 <= tp_rank < tp_size:
            raise ValueError(
                f"Invalid Qwen4-Exp gate/up TP partition: "
                f"{weight.shape=}, {tp_size=}, {tp_rank=}"
            )
        half = weight.shape[1] // 2
        local_size = half // tp_size
        start = tp_rank * local_size
        expected_shape = (
            (weight.shape[0], weight.shape[2], 2 * local_size) if experts.use_triton_kernels else
            (weight.shape[0], 2 * local_size, weight.shape[2])
        )
        if tuple(param.shape) != expected_shape:
            raise ValueError(
                f"Qwen4-Exp gate/up target shape mismatch: "
                f"{tuple(param.shape)=}, {expected_shape=}"
            )
        gate = weight[:, start:start + local_size]
        up = weight[:, half + start:half + start + local_size]
        if experts.use_triton_kernels:
            param[:, :, :local_size].copy_(gate.transpose(-2, -1))
            param[:, :, local_size:].copy_(up.transpose(-2, -1))
        else:
            param[:, :local_size].copy_(gate)
            param[:, local_size:].copy_(up)
        return True

    weight_loader_fused(param, weight, target_name, shard_id)
    return True


def _load_pinned_ple_shard(
    model: torch.nn.Module,
    params: dict[str, torch.nn.Parameter],
    name: str,
    weight: torch.Tensor,
) -> bool:
    match = _PLE_SHARD_RE.fullmatch(_normalize_qwen4_exp_weight_name(name))
    if match is None:
        return False
    prefix, shard_index = match.groups()
    param = params.get(f"{prefix}.weight")
    if (
        param is None or param.device.type != "cpu" or not param.is_pinned() or
        not weight.is_cuda or param.dtype != torch.bfloat16 or weight.dtype != param.dtype or
        param.ndim != 2 or weight.ndim != 2 or weight.shape[1] != param.shape[1]
    ):
        return False
    embedding = model.get_submodule(prefix)
    text_config = getattr(model.config, "text_config", model.config)
    parts = text_config.split_ngram_parts
    shard_index = int(shard_index)
    if parts <= 0 or not 0 <= shard_index < parts:
        return False
    rows_per_shard = (embedding.org_vocab_size + parts - 1) // parts
    shard_start = shard_index * rows_per_shard
    shard_end = min(shard_start + rows_per_shard, embedding.org_vocab_size)
    if weight.shape[0] != shard_end - shard_start:
        return False
    tp_start = embedding.shard_indices.org_vocab_start_index
    tp_end = embedding.shard_indices.org_vocab_end_index
    start, end = max(shard_start, tp_start), min(shard_end, tp_end)
    if start < end:
        target = param[start - tp_start:end - tp_start]
        source = weight[start - shard_start:end - shard_start]
        # Copy into existing pinned storage; avoid a pageable CPU temporary and second copy.
        target.copy_(source, non_blocking=True)
        torch.cuda.current_stream(weight.device).synchronize()
    return True


@torch.no_grad()
def load_qwen4_exp_packed_weights(
    model: torch.nn.Module,
    named_tensors: list[tuple[str, torch.Tensor]],
) -> Any:
    """Load BF16 experts and pinned PLE shards, delegating other layouts."""
    params = None
    fallback = []
    for name, weight in named_tensors:
        if ".ngram_embedding.shard_" in name:
            if params is None:
                params = dict(model.named_parameters(remove_duplicate=False))
            if _load_pinned_ple_shard(model, params, name, weight):
                continue
        if _packed_expert_target(name) is not None:
            if params is None:
                params = dict(model.named_parameters(remove_duplicate=False))
            if _load_packed_expert(model, params, name, weight):
                continue
        fallback.append((name, weight))
    if fallback:
        return model.load_weights(fallback)
    return None
