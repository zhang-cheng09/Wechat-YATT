# coding=utf-8
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.distributed.tensor import DTensor


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local().detach()
    return tensor.detach()


def tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    local = _local_tensor(tensor).contiguous()
    raw = local.view(torch.uint8).reshape(-1).cpu().numpy().tobytes()
    return {
        "dtype": str(local.dtype),
        "shape": list(local.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def compact_tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    local = _local_tensor(tensor).contiguous()
    raw = local.view(torch.uint8).reshape(-1)
    if raw.numel() <= 1024 * 1024:
        selected = raw
        hash_kind = "full"
    else:
        indices = torch.linspace(
            0,
            raw.numel() - 1,
            steps=64,
            dtype=torch.float64,
            device=raw.device,
        ).to(torch.long)
        selected = raw.index_select(0, indices)
        hash_kind = "sample"
    payload = selected.cpu().numpy().tobytes()
    record = {
        "dtype": str(local.dtype),
        "shape": list(local.shape),
        "hash_kind": hash_kind,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "sum_fp64": local.sum(dtype=torch.float64).item(),
    }
    if local.is_floating_point():
        record["norm_sq_fp64"] = torch.linalg.vector_norm(local,
                                                          dtype=torch.float64).square().item()
    return record


def rng_fingerprint() -> dict[str, Any]:
    record = {
        "initial_seed": torch.initial_seed(),
        "cpu": tensor_fingerprint(torch.get_rng_state()),
    }
    if torch.cuda.is_available():
        record["cuda"] = tensor_fingerprint(torch.cuda.get_rng_state())
    return record


def parameter_fingerprint(model: torch.nn.Module) -> dict[str, Any]:
    digest = hashlib.sha256()
    local_numel = 0
    sampled_bytes = 0
    parameter_count = 0
    first_parameter = next(model.parameters())
    first_local = _local_tensor(first_parameter)
    value_sum = torch.zeros((), dtype=torch.float64, device=first_local.device)
    value_norm_sq = torch.zeros_like(value_sum)

    for name, parameter in model.named_parameters():
        local = _local_tensor(parameter)
        parameter_count += 1
        local_numel += local.numel()
        digest.update(name.encode("utf-8"))
        digest.update(str(local.dtype).encode("utf-8"))
        digest.update(str(tuple(local.shape)).encode("utf-8"))
        if local.numel() == 0:
            continue

        raw = local.contiguous().view(torch.uint8).reshape(-1)
        count = min(raw.numel(), 64)
        indices = torch.linspace(
            0,
            raw.numel() - 1,
            steps=count,
            dtype=torch.float64,
            device=raw.device,
        ).to(torch.long)
        probe = raw.index_select(0, indices).cpu().numpy().tobytes()
        digest.update(probe)
        sampled_bytes += len(probe)

        value_sum += local.sum(dtype=torch.float64)
        value_norm_sq += torch.linalg.vector_norm(local, dtype=torch.float64).square()

    return {
        "parameter_count": parameter_count,
        "local_numel": local_numel,
        "sampled_bytes": sampled_bytes,
        "sample_sha256": digest.hexdigest(),
        "value_sum_fp64": value_sum.item(),
        "value_norm_sq_fp64": value_norm_sq.item(),
    }


def packed_microbatch_fingerprint(
    batch: Mapping[str, Any],
    forward_kwargs: Mapping[str, Any],
    microbatch_index: int,
) -> dict[str, Any]:
    tensors = {}
    for source_name, source in (("batch", batch), ("forward", forward_kwargs)):
        for name, value in source.items():
            if isinstance(value, torch.Tensor):
                tensors[f"{source_name}.{name}"] = tensor_fingerprint(value)

    full_packed = batch.get("full_packed_seq_params")
    if full_packed is not None:
        tensors["packed.cu_seqlens_q"] = tensor_fingerprint(full_packed.cu_seqlens_q)
        tensors["packed.cu_seqlens_q_padded"] = tensor_fingerprint(full_packed.cu_seqlens_q_padded)

    cp_context = forward_kwargs.get("_qwen4_exp_cp_context")
    cp_metadata = None
    if cp_context is not None:
        tensors["cp.global_input_ids"] = tensor_fingerprint(cp_context.global_input_ids)
        tensors["cp.global_padding_mask"] = tensor_fingerprint(cp_context.global_padding_mask)
        if cp_context.global_cu_seqlens is not None:
            tensors["cp.global_cu_seqlens"] = tensor_fingerprint(cp_context.global_cu_seqlens)
        tensors["cp.global_segment_boundaries"] = tensor_fingerprint(
            cp_context.global_segment_boundaries
        )
        cp_metadata = {
            "rank": cp_context.rank,
            "size": cp_context.size,
            "local_sequence_start": cp_context.local_sequence_start,
            "local_sequence_length": cp_context.local_sequence_length,
            "global_document_boundaries": list(cp_context.global_document_boundaries),
            "global_segment_boundary_values": list(cp_context.global_segment_boundary_values),
        }

    return {
        "microbatch_index": microbatch_index,
        "tensors": tensors,
        "cp_metadata": cp_metadata,
    }


def write_deepep_forward_probe(
    directory: str,
    layer_index: int,
    call_index: int,
    tensors: Mapping[str, torch.Tensor],
) -> None:
    write_diagnostic_record(
        directory,
        "deepep_forward",
        {
            "layer_index": layer_index,
            "call_index": call_index,
            "stages": {
                name: compact_tensor_fingerprint(tensor)
                for name, tensor in tensors.items()
            },
        },
    )


def write_diagnostic_record(
    directory: str,
    kind: str,
    payload: Mapping[str, Any],
) -> None:
    rank = torch.distributed.get_rank()
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"rank_{rank:05d}.jsonl"
    record = {"kind": kind, "rank": rank, **payload}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")
