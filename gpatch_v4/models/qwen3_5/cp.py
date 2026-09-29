# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Zigzag context parallelism for Qwen3.5 / Qwen3.6-35B-A3B.

Non-THD splits the sequence into ``2 * cp`` chunks and gives rank ``r``
chunks ``[r, 2*cp-r-1]``. THD applies the same rule per document via TE
``thd_get_partitioned_indices``. Each packed document is padded to a
multiple of ``2 * cp``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from megatron.core import mpu

__all__ = [
    "Qwen3_5CPContext",
    "Qwen3_5PackedSeqParams",
    "qwen3_5_all_gather_cp",
    "qwen3_5_cp_chunk_data",
    "qwen3_5_cp_local_index",
    "qwen3_5_pack_sequences",
    "qwen3_5_packed_padding_mask",
]


@dataclass(frozen=True)
class Qwen3_5PackedSeqParams:
    """Global layout metadata for one packed THD row."""

    cu_seqlens_q: torch.Tensor
    cu_seqlens_q_padded: torch.Tensor
    max_seqlen_q: int
    total_seqlen: int
    qkv_format: str = "thd"

    def __post_init__(self) -> None:
        if self.cu_seqlens_q.ndim != 1 or self.cu_seqlens_q.numel() < 2:
            raise ValueError("cu_seqlens_q must be a rank-1 document-boundary tensor")
        if self.cu_seqlens_q.shape != self.cu_seqlens_q_padded.shape:
            raise ValueError("real and padded document boundaries must have the same shape")
        if self.cu_seqlens_q.dtype != torch.long or self.cu_seqlens_q_padded.dtype != torch.long:
            raise ValueError("packed document boundaries must use torch.long")
        if self.cu_seqlens_q.device != self.cu_seqlens_q_padded.device:
            raise ValueError("real and padded document boundaries must be on the same device")
        if self.qkv_format != "thd":
            raise ValueError(f"qkv_format must be 'thd', got {self.qkv_format!r}")
        real_lengths = self.cu_seqlens_q.diff()
        padded_lengths = self.cu_seqlens_q_padded.diff()
        if (
            int(self.cu_seqlens_q[0]) != 0 or int(self.cu_seqlens_q_padded[0]) != 0 or
            bool((real_lengths <= 0).any()) or bool((padded_lengths <= 0).any()) or
            bool((real_lengths > padded_lengths).any()) or
            int(self.cu_seqlens_q[-1]) > self.total_seqlen or
            int(self.cu_seqlens_q_padded[-1]) != self.total_seqlen
        ):
            raise ValueError("invalid packed document boundaries")
        if self.max_seqlen_q != int(padded_lengths.max()):
            raise ValueError("max_seqlen_q does not match the padded document layout")


def qwen3_5_pack_sequences(
    input_ids: list[torch.Tensor],
    labels: list[torch.Tensor],
    *,
    cp_size: int,
    pad_multiple: int,
    pad_token_id: int,
    label_ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Qwen3_5PackedSeqParams]:
    """Pack shifted documents with per-document TE zigzag padding."""
    if not input_ids:
        raise ValueError("input_ids must be non-empty")
    if len(input_ids) != len(labels):
        raise ValueError("input_ids and labels must contain the same number of documents")
    if cp_size <= 0 or pad_multiple <= 0:
        raise ValueError(
            f"cp_size and pad_multiple must be positive, got {cp_size} and {pad_multiple}"
        )

    device = input_ids[0].device
    lengths = []
    for index, (document_ids, document_labels) in enumerate(zip(input_ids, labels)):
        if document_ids.ndim != 1 or document_labels.shape != document_ids.shape:
            raise ValueError(
                f"packed document {index} IDs/labels must be matching 1-D tensors, got "
                f"{tuple(document_ids.shape)} and {tuple(document_labels.shape)}"
            )
        if document_ids.numel() == 0:
            raise ValueError(f"packed document {index} is empty after next-token shifting")
        if document_ids.device != device or document_labels.device != device:
            raise ValueError("all packed IDs and labels must be on the same device")
        lengths.append(document_ids.numel())

    padded_lengths = [
        ((length + pad_multiple - 1) // pad_multiple) * pad_multiple for length in lengths
    ]
    total_seqlen = sum(padded_lengths)
    total_alignment = cp_size * pad_multiple
    total_seqlen = ((total_seqlen + total_alignment - 1) // total_alignment) * total_alignment

    packed_ids = torch.full((1, total_seqlen), pad_token_id, dtype=torch.long, device=device)
    packed_labels = torch.full(
        (1, total_seqlen), label_ignore_index, dtype=torch.long, device=device
    )
    position_ids = torch.zeros((1, total_seqlen), dtype=torch.long, device=device)
    real_boundaries_host = [0]
    padded_boundaries_host = [0]
    physical = 0
    for length, padded_length, document_ids, document_labels in zip(
        lengths, padded_lengths, input_ids, labels
    ):
        packed_ids[0, physical:physical + length] = document_ids.long()
        packed_labels[0, physical:physical + length] = document_labels.long()
        position_ids[0, physical:physical + length] = torch.arange(length, device=device)
        physical += padded_length
        real_boundaries_host.append(real_boundaries_host[-1] + length)
        padded_boundaries_host.append(physical)
    padded_boundaries_host[-1] = total_seqlen
    padded_boundaries = torch.tensor(padded_boundaries_host, dtype=torch.long, device=device)
    packed_seq_params = Qwen3_5PackedSeqParams(
        cu_seqlens_q=torch.tensor(real_boundaries_host, dtype=torch.long, device=device),
        cu_seqlens_q_padded=padded_boundaries,
        max_seqlen_q=int(padded_boundaries.diff().max()),
        total_seqlen=total_seqlen,
    )
    return packed_ids, position_ids, packed_labels, packed_seq_params


def qwen3_5_packed_padding_mask(packed_seq_params: Qwen3_5PackedSeqParams, ) -> torch.Tensor:
    """Bool ``[1, T]`` mask: True on per-document pads and the CP tail."""
    mask = torch.ones(
        (1, packed_seq_params.total_seqlen),
        dtype=torch.bool,
        device=packed_seq_params.cu_seqlens_q.device,
    )
    real_lengths = packed_seq_params.cu_seqlens_q.diff().tolist()
    starts = packed_seq_params.cu_seqlens_q_padded[:-1].tolist()
    for start, real_length in zip(starts, real_lengths, strict=True):
        mask[0, start:start + real_length] = False
    return mask


def qwen3_5_cp_local_index(
    cp_rank: int,
    cp_size: int,
    sequence_length: int,
    device: torch.device,
    cu_seqlens_padded: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Global token indices owned by this CP rank (zigzag or per-doc THD)."""
    if cp_size <= 0 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"invalid CP rank/size: rank={cp_rank}, size={cp_size}")
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if cp_size == 1:
        return torch.arange(sequence_length, device=device, dtype=torch.long)
    if sequence_length % cp_size != 0:
        raise ValueError(
            f"sequence length {sequence_length} must be divisible by CP size {cp_size}"
        )
    if cu_seqlens_padded is not None:
        from megatron.core.extensions.transformer_engine import (
            get_thd_partitioned_indices,
        )
        return get_thd_partitioned_indices(
            cu_seqlens_padded.to(device=device, dtype=torch.int32),
            sequence_length,
            cp_size,
            cp_rank,
        ).to(dtype=torch.long)
    if sequence_length % (2 * cp_size) != 0:
        raise ValueError(
            f"zigzag CP requires seq_len divisible by 2*cp_size, "
            f"got seq_len={sequence_length} cp_size={cp_size}"
        )
    chunk = sequence_length // (2 * cp_size)
    front = torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk, device=device, dtype=torch.long)
    back_start = (2 * cp_size - cp_rank - 1) * chunk
    back = torch.arange(back_start, back_start + chunk, device=device, dtype=torch.long)
    return torch.cat((front, back), dim=0)


@dataclass(frozen=True)
class Qwen3_5CPContext:
    """Replicated metadata for one zigzag / THD-zigzag context-parallel sequence."""

    group: Optional[dist.ProcessGroup]
    rank: int
    size: int
    global_padding_mask: torch.Tensor
    local_index: torch.Tensor
    global_cu_seqlens: Optional[torch.Tensor] = None
    global_cu_seqlens_padded: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.size <= 0 or not 0 <= self.rank < self.size:
            raise ValueError(f"invalid CP rank/size: rank={self.rank}, size={self.size}")
        if self.size > 1 and self.group is None:
            raise ValueError("CP size greater than one requires a process group")
        if self.global_padding_mask.ndim != 2 or self.global_padding_mask.dtype != torch.bool:
            raise ValueError(
                "global_padding_mask must be a bool [batch, sequence] tensor, "
                f"got shape={tuple(self.global_padding_mask.shape)}, "
                f"dtype={self.global_padding_mask.dtype}"
            )
        if self.local_index.ndim != 1 or self.local_index.dtype != torch.long:
            raise ValueError("local_index must be a 1-D torch.long tensor")
        if self.local_index.numel() <= 0:
            raise ValueError("local_index must be non-empty")
        global_length = self.global_padding_mask.shape[1]
        if self.local_index.numel() * self.size != global_length:
            raise ValueError(
                "local_index length * CP size must equal the global sequence, "
                f"got local={self.local_index.numel()}, size={self.size}, "
                f"global={global_length}"
            )
        if self.local_index.device.type == "cpu":
            if bool((self.local_index < 0).any()) or bool(
                (self.local_index >= global_length).any()
            ):
                raise ValueError("local_index is outside the global sequence")
        if self.global_cu_seqlens is not None:
            boundaries = self.global_cu_seqlens
            if boundaries.ndim != 1 or boundaries.numel() < 2:
                raise ValueError("global_cu_seqlens must be a rank-1 boundary tensor")
            if self.global_padding_mask.shape[0] != 1:
                raise ValueError("packed THD requires a batch of one row")
            if int(boundaries[0]) != 0 or bool((boundaries.diff() <= 0).any()):
                raise ValueError("packed real boundaries must start at zero and increase")
            if self.global_cu_seqlens_padded is None:
                raise ValueError("packed THD requires global_cu_seqlens_padded")
            if self.global_cu_seqlens_padded.shape != boundaries.shape:
                raise ValueError("real and padded packed boundaries must have the same shape")
            if int(self.global_cu_seqlens_padded[-1]) != global_length:
                raise ValueError(
                    "padded packed boundary must end at the global length, "
                    f"got {int(self.global_cu_seqlens_padded[-1])} and {global_length}"
                )
        elif self.global_cu_seqlens_padded is not None:
            raise ValueError("global_cu_seqlens_padded requires global_cu_seqlens")

    @property
    def global_sequence_length(self) -> int:
        return self.global_padding_mask.shape[1]

    @property
    def local_sequence_length(self) -> int:
        return int(self.local_index.numel())

    @property
    def local_attention_mask(self) -> torch.Tensor:
        return self.global_padding_mask.index_select(1, self.local_index).logical_not()

    @property
    def global_segment_boundaries(self) -> torch.Tensor:
        if self.global_cu_seqlens_padded is not None:
            return self.global_cu_seqlens_padded
        if self.global_cu_seqlens is None:
            return torch.tensor(
                [0, self.global_sequence_length],
                dtype=torch.long,
                device=self.global_padding_mask.device,
            )
        if int(self.global_cu_seqlens[-1]) == self.global_sequence_length:
            return self.global_cu_seqlens
        return torch.cat(
            (
                self.global_cu_seqlens,
                self.global_cu_seqlens.new_tensor([self.global_sequence_length]),
            )
        )

    def gather_local(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return tensor.index_select(dim, self.local_index).contiguous()


def qwen3_5_cp_chunk_data(
    cp_rank: int,
    cp_size: int,
    cp_group: Optional[dist.ProcessGroup],
    *,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    position_ids: torch.Tensor,
    global_padding_mask: torch.Tensor,
    global_cu_seqlens: Optional[torch.Tensor] = None,
    global_cu_seqlens_padded: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Qwen3_5CPContext,
]:
    """Zigzag-slice SFT tensors and retain the replicated CP metadata."""
    if tokens.ndim != 2:
        raise ValueError(f"CP tokens must be [batch, sequence], got {tuple(tokens.shape)}")
    for name, tensor in (
        ("labels", labels),
        ("loss_mask", loss_mask),
        ("position_ids", position_ids),
        ("global_padding_mask", global_padding_mask),
    ):
        if tensor.shape != tokens.shape:
            raise ValueError(
                f"CP {name} shape {tuple(tensor.shape)} does not match tokens "
                f"{tuple(tokens.shape)}"
            )
    if global_padding_mask.dtype != torch.bool:
        raise ValueError(f"global_padding_mask must be bool, got {global_padding_mask.dtype}")
    if global_cu_seqlens is None:
        valid = global_padding_mask.logical_not()
        lengths = valid.sum(dim=-1, dtype=torch.long)
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        if not bool(torch.equal(valid, positions < lengths.unsqueeze(1))):
            raise NotImplementedError("qwen3_5 non-THD CP supports right-tail padding only")
    local_index = qwen3_5_cp_local_index(
        cp_rank,
        cp_size,
        tokens.shape[1],
        tokens.device,
        cu_seqlens_padded=global_cu_seqlens_padded,
    )
    context = Qwen3_5CPContext(
        group=cp_group,
        rank=cp_rank,
        size=cp_size,
        global_padding_mask=global_padding_mask,
        local_index=local_index,
        global_cu_seqlens=global_cu_seqlens,
        global_cu_seqlens_padded=global_cu_seqlens_padded,
    )
    return (
        context.gather_local(tokens),
        context.gather_local(labels),
        context.gather_local(loss_mask),
        context.gather_local(position_ids),
        context,
    )


class _Qwen3_5ThdAllGather(torch.autograd.Function):
    """Inverse of per-document ``thd_get_partitioned_indices``."""
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, cu_seqlens_padded: torch.Tensor, gather_dim: int):
        if gather_dim not in (0, 1):
            raise ValueError(f"THD CP gather only supports dim 0 or 1, got {gather_dim}")
        from megatron.core.extensions.transformer_engine import (
            get_thd_partitioned_indices,
        )
        cp_size = mpu.get_context_parallel_world_size()
        cp_group = mpu.get_context_parallel_group()
        parts = [torch.empty_like(tensor) for _ in range(cp_size)]
        dist.all_gather(parts, tensor.contiguous(), group=cp_group)
        global_length = tensor.shape[gather_dim] * cp_size
        out_shape = list(tensor.shape)
        out_shape[gather_dim] = global_length
        output = tensor.new_empty(out_shape)
        cu = cu_seqlens_padded.to(device=tensor.device, dtype=torch.int32)
        local_index = None
        cp_rank = mpu.get_context_parallel_rank()
        for rank, part in enumerate(parts):
            index = get_thd_partitioned_indices(cu, global_length, cp_size,
                                                rank).to(dtype=torch.long)
            output.index_copy_(gather_dim, index, part)
            if rank == cp_rank:
                local_index = index
        assert local_index is not None
        ctx.save_for_backward(local_index)
        ctx.gather_dim = gather_dim
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        local_index, = ctx.saved_tensors
        return grad_output.index_select(ctx.gather_dim, local_index), None, None


def qwen3_5_all_gather_cp(
    tensor: torch.Tensor,
    *,
    cu_seqlens_padded: Optional[torch.Tensor],
    gather_dim: int = 1,
) -> torch.Tensor:
    """Restore the global sequence axis from this rank's zigzag / THD shard."""
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size <= 1:
        return tensor
    # cp thd 时候走，自定义的thd zigzag
    if cu_seqlens_padded is not None:
        return _Qwen3_5ThdAllGather.apply(tensor, cu_seqlens_padded, gather_dim)
    # cp non thd 时候走，其实跟默认一样
    from gpatch_v4.core.mappings import all_gather_from_context_parallel_region
    return all_gather_from_context_parallel_region(tensor, gather_dim)
