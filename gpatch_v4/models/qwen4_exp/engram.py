# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Owner-sharded Engram (n-gram) embedding table.

At the released config the table is ``[320_001_536, 160]`` = **51.2 B parameters**
(~102 GB in bf16), sitting on decoder ``layers.1`` alone. It cannot be replicated, and
FSDP2's usual "all-gather the parameter before use" is pointless here: a forward only
touches ``B * S * 16`` rows (65 536 at B=1, S=4096), i.e. ~0.02 % of the table.

So the table is **row-sharded and stays sharded**. Rank ``r`` owns the contiguous range
``[r * local_rows, (r+1) * local_rows)`` and a lookup routes ids to their owner:

1. group the requested row ids by owner and all-to-all the ids,
2. each rank looks up only its own shard,
3. all-to-all the values back (autograd-aware).

Backward reverses the value route, so a row's gradient is only ever accumulated by its
owner — no cross-rank reduction of a 51.2 B tensor.

Hashing is **not** reimplemented: :class:`Qwen4ExpTextNGramEmbedding` already derives the
16 hash heads (8 bigram + 8 trigram, distinct primes above ``ngram_vocab_size_base``,
with EOS resetting the n-gram context) in pure PyTorch. We subclass it and swap only the
``nn.Embedding`` for the sharded table, which also keeps the checkpoint FQN
``...ple.ple_embedding.ngram_embedding.weight`` unchanged.

Deviation from the reference: NVIDIA routes ids through a *fixed-capacity* all-to-all.
We use the uneven variant gcore already ships, which is exact — no capacity to overflow,
and no tokens silently dropped when the hash distribution is skewed.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from typing_extensions import override

from .a2a import all_to_all_uneven
from .modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

__all__ = [
    "OwnerShardedNGramEmbedding",
    "Qwen4ExpEngramEmbedding",
    "iter_engram_hf_shards",
]


class OwnerShardedNGramEmbedding(nn.Module):
    """Row-sharded embedding table whose lookups are routed to the owning rank.

    Parameters
    ----------
    num_embeddings : int
        **Global** row count. Must be divisible by the group size.
    embedding_dim : int
        Row width (160 at the released config).
    process_group : dist.ProcessGroup, optional
        Group across which rows are sharded. ``None`` uses the default group; a group
        of size 1 (or no initialized distributed runtime) degenerates to a plain local
        lookup, which is what the CPU tests exercise.

    Attributes
    ----------
    weight : nn.Parameter
        The **local** shard, ``[local_rows, embedding_dim]``. Named ``weight`` so the
        state-dict FQN matches the released checkpoint.
    output_dtype : torch.dtype or None
        Cast applied to the looked-up values on the way out. Set by ``apply_hp`` from
        ``mp_policy.param_dtype``, because this table is **excluded** from FSDP2
        (``ignored_params``) and therefore never gets FSDP's mixed-precision cast: the
        master stays fp32 while every consumer downstream has been cast to bf16. Without
        this the first projection fails with
        ``expected mat1 and mat2 to have the same dtype``.

        The cast is on the *values* (``[tokens, dim]``), not the table, so it costs
        nothing meaningful and keeps the fp32 master intact for the optimizer — the
        cast's backward converts gradients back to fp32 on the way in.
        ``None`` means no cast, which is the plain CPU / no-mixed-precision case.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__()
        self.process_group = process_group
        self.output_dtype: Optional[torch.dtype] = None
        self.world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
        self.rank = dist.get_rank(process_group) if dist.is_initialized() else 0

        if num_embeddings % self.world_size != 0:
            raise ValueError(
                f"Engram table rows ({num_embeddings}) must be divisible by the owner "
                f"group size ({self.world_size})."
            )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.local_rows = num_embeddings // self.world_size
        self.global_row_start = self.rank * self.local_rows
        self.weight = nn.Parameter(torch.empty(self.local_rows, embedding_dim))

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"local_rows={self.local_rows}, world_size={self.world_size}"
        )

    @property
    def local_weight(self) -> torch.Tensor:
        """This rank's rows as a plain tensor.

        ``apply_hp`` wraps :attr:`weight` in a global ``DTensor(Shard(0))`` so the
        optimizer sees a uniform parameter type — mixing DTensor and plain tensors in one
        AdamW param group makes ``torch._foreach_mul_`` raise. Lookups need the raw local
        rows, and ``to_local()`` is autograd-aware, so gradients still reach the DTensor.
        """
        weight = self.weight
        return weight.to_local() if isinstance(weight, DTensor) else weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up global row ids, routing each to its owner.

        Parameters
        ----------
        input_ids : torch.Tensor
            Integer tensor of **global** row ids, any shape.

        Returns
        -------
        torch.Tensor
            ``[*input_ids.shape, embedding_dim]``.
        """
        flat_ids = input_ids.reshape(-1)
        if self.world_size == 1:
            values = F.embedding(flat_ids, self.local_weight)
            return self._cast_output(values.view(*input_ids.shape, self.embedding_dim))

        owner = torch.div(flat_ids, self.local_rows, rounding_mode="floor")
        # Stable so the permutation is reproducible across ranks and runs.
        order = torch.argsort(owner, stable=True)
        send_counts = torch.bincount(owner, minlength=self.world_size)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.process_group)
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        # Route ids to owners. Integer ids carry no gradient, so a plain collective.
        sent_ids = flat_ids[order]
        recv_ids = torch.empty(sum(recv_splits), dtype=flat_ids.dtype, device=flat_ids.device)
        dist.all_to_all_single(
            recv_ids,
            sent_ids.contiguous(),
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.process_group,
        )

        local_ids = recv_ids - self.global_row_start
        if self.world_size > 1 and local_ids.numel() > 0:
            out_of_range = (local_ids < 0) | (local_ids >= self.local_rows)
            if bool(out_of_range.any()):
                raise RuntimeError(
                    "Engram routing delivered a row id outside this rank's shard "
                    f"[{self.global_row_start}, {self.global_row_start + self.local_rows}); "
                    "the id->owner mapping disagrees across ranks."
                )
        local_values = F.embedding(local_ids, self.local_weight)

        # Route values back. This leg is differentiable, and its transpose is exactly
        # what makes each row's gradient land on (only) its owner.
        gathered = all_to_all_uneven(local_values, recv_splits, send_splits, self.process_group)

        # Undo the owner-grouping permutation. A gather by the inverse permutation keeps
        # this differentiable, unlike an in-place scatter into a fresh buffer.
        inverse_order = torch.empty_like(order)
        inverse_order[order] = torch.arange(order.numel(), device=order.device)
        values = gathered[inverse_order]
        return self._cast_output(values.view(*input_ids.shape, self.embedding_dim))

    def _cast_output(self, values: torch.Tensor) -> torch.Tensor:
        if self.output_dtype is None:
            return values
        return values.to(self.output_dtype)


def materialize_engram_tables(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Allocate ignored Engram shards on ``device``. FSDP2 leaves them on meta."""
    for module in model.modules():
        if not isinstance(module, OwnerShardedNGramEmbedding):
            continue
        weight = module.weight
        if not weight.is_meta:
            continue
        module.weight = nn.Parameter(
            torch.empty(tuple(weight.shape), device=device, dtype=dtype),
            requires_grad=weight.requires_grad,
        )


def iter_engram_hf_shards(
    table: OwnerShardedNGramEmbedding,
    target_key: str,
    split_ngram_parts: int,
    *,
    out_device: torch.device | str | None = None,
    yield_on_all_ranks: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield Engram rows in HF / SGLang ``split_ngram_parts`` disk layout.

    Trainer Engram ownership is ``world_size`` equal row slices named by owner rank.
    Released checkpoints and SGLang ``load_qwen4_exp_ple_shard`` instead index
    ``...ngram_embedding.shard_{i}.weight`` with
    ``i in [0, split_ngram_parts)`` and row start ``i * ceil(N / split_ngram_parts)``.
    Emitting owner ranks as ``shard_{owner}`` therefore mis-places rows whenever
    ``world_size != split_ngram_parts`` (e.g. EP32 train vs HF 128).

    Streams one owner broadcast at a time and packs complete HF shards before yield so
    the full ``[N, dim]`` table is never materialized.

    Parameters
    ----------
    table : OwnerShardedNGramEmbedding
    target_key : str
        Model FQN ending in ``.weight`` (e.g. ``...ngram_embedding.weight``).
    split_ngram_parts : int
        HF / SGLang partition count (128 on Qwen3.8-Flash-Next).
    out_device : torch.device or str, optional
        Device for yielded tensors. ``None`` keeps the broadcast buffer's device.
    yield_on_all_ranks : bool
        If False, only rank 0 yields (other ranks still join broadcasts).

    Yields
    ------
    tuple[str, torch.Tensor]
        ``(prefix.shard_{i}.weight, bf16 rows)`` with
        ``rows.shape[0] == min(rows_per_shard, N - i * rows_per_shard)``.
    """
    if split_ngram_parts <= 0:
        raise ValueError(f"split_ngram_parts must be positive, got {split_ngram_parts}")
    if not target_key.endswith(".weight"):
        raise ValueError(f"target_key must end with .weight, got {target_key!r}")

    prefix = target_key[:-len(".weight")]
    rank = table.rank
    world_size = table.world_size
    n_rows = table.num_embeddings
    dim = table.embedding_dim
    local_rows = table.local_rows
    rows_per_shard = (n_rows + split_ngram_parts - 1) // split_ngram_parts

    local = table.local_weight.detach()
    if local.dtype != torch.bfloat16:
        local = local.to(torch.bfloat16)
    # Engram PG is NCCL; broadcast must stay on CUDA even when the table was CPU-offloaded.
    if not local.is_cuda:
        local = local.cuda()
    local = local.contiguous()

    pending_idx: Optional[int] = None
    pending: Optional[torch.Tensor] = None
    pending_pos = 0

    def _maybe_yield() -> Iterator[tuple[str, torch.Tensor]]:
        nonlocal pending_idx, pending, pending_pos
        if pending is None:
            return
        assert pending_idx is not None
        if pending_pos != pending.shape[0]:
            raise RuntimeError(
                f"Engram HF shard {pending_idx} filled {pending_pos}/{pending.shape[0]} rows"
            )
        out = pending
        if out_device is not None:
            out = out.to(out_device)
        key = f"{prefix}.shard_{pending_idx}.weight"
        pending_idx = None
        pending = None
        pending_pos = 0
        if yield_on_all_ranks or rank == 0:
            yield key, out.detach().contiguous()
        else:
            del out

    for owner in range(world_size):
        if dist.is_initialized() and world_size > 1:
            buffer = local if owner == rank else torch.empty_like(local)
            dist.broadcast(buffer, src=owner, group=table.process_group)
        else:
            buffer = local

        owner_start = owner * local_rows
        local_off = 0
        while local_off < local_rows:
            global_row = owner_start + local_off
            if global_row >= n_rows:
                break
            shard_idx = global_row // rows_per_shard
            shard_g0 = shard_idx * rows_per_shard
            shard_g1 = min(shard_g0 + rows_per_shard, n_rows)
            shard_rows = shard_g1 - shard_g0
            take = min(local_rows - local_off, shard_g1 - global_row)

            if pending is None:
                pending_idx = shard_idx
                pending = torch.empty(shard_rows, dim, device=buffer.device, dtype=torch.bfloat16)
                pending_pos = global_row - shard_g0
                if pending_pos != 0:
                    raise RuntimeError(
                        f"Engram HF shard {shard_idx} started at local offset "
                        f"{pending_pos}; owners must be streamed in rank order"
                    )
            elif pending_idx != shard_idx:
                raise RuntimeError(
                    f"Engram HF shard packing jumped from {pending_idx} to {shard_idx}"
                )

            assert pending is not None
            pending[pending_pos:pending_pos + take].copy_(buffer[local_off:local_off + take])
            pending_pos += take
            local_off += take
            if pending_pos == pending.shape[0]:
                yield from _maybe_yield()

        if owner != rank:
            del buffer

    if pending is not None:
        yield from _maybe_yield()


class Qwen4ExpEngramEmbedding(Qwen4ExpTextNGramEmbedding):
    """Upstream n-gram embedding with the table replaced by an owner-sharded one.

    The hashing is inherited untouched; only the lookup is swapped. The parent's
    ``forward`` calls ``self.ngram_embedding(...)``, so delegating to it after the swap
    reuses upstream's hash construction verbatim.
    """
    def __init__(
        self,
        config,
        embedding_dim: int,
        layer_idx: int,
        ple_layer_index: int = 0,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__(config, embedding_dim, layer_idx, ple_layer_index)
        # The parent has just built a dense `nn.Embedding` over the full padded vocab.
        # That is 51.2 B parameters at the released config, so this class is only ever
        # constructed on the meta device in the real flow (the HpModule path already
        # requires meta construction); the tensor is discarded here either way.
        num_embeddings, dim = self.ngram_embedding.weight.shape
        self.ngram_embedding = OwnerShardedNGramEmbedding(num_embeddings, dim, process_group)

    @override
    def forward(
        self, input_ids: torch.Tensor, past_key_values: Optional[object] = None
    ) -> torch.Tensor:
        if past_key_values is not None:
            raise NotImplementedError(
                "Engram training path does not support a KV cache. The n-gram context is "
                "cached as a third conv state upstream."
            )
        return super().forward(input_ids, None)

    def forward_global_slice(
        self,
        global_input_ids: torch.Tensor,
        *,
        sequence_start: int,
        sequence_end: int,
        segment_boundaries: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Hash globally while looking up only this CP rank's positions."""
        if (
            sequence_start < 0 or sequence_end < sequence_start or
            sequence_end > global_input_ids.shape[1]
        ):
            raise ValueError(
                f"invalid Engram global slice [{sequence_start}, {sequence_end}) for "
                f"sequence length {global_input_ids.shape[1]}"
            )
        input_ids = global_input_ids.long()
        if segment_boundaries is None:
            shifted_tokens = [
                self._shift_right_ignore_eos(input_ids, shift) for shift in range(self.ngram_size)
            ]
        else:
            boundaries = segment_boundaries.to(device=input_ids.device, dtype=torch.long)
            if (
                boundaries.ndim != 1 or boundaries.numel() < 2 or int(boundaries[0]) != 0 or
                int(boundaries[-1]) != input_ids.shape[1] or bool((boundaries.diff() <= 0).any())
            ):
                raise ValueError("segment_boundaries must partition the complete input row")
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            segment_indices = torch.searchsorted(boundaries[1:], positions, right=True)
            segment_starts = boundaries.index_select(0, segment_indices)
            shifted_tokens = []
            for shift in range(self.ngram_size):
                shifted = self._shift_right_ignore_eos(input_ids, shift)
                if shift:
                    valid = positions - shift >= segment_starts
                    shifted = torch.where(
                        valid.unsqueeze(0),
                        shifted,
                        input_ids.new_full((), self.eos_token_id),
                    )
                shifted_tokens.append(shifted)
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(
                    mixed_ids,
                    shifted_tokens[position] * self.layer_multipliers[position],
                )
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
            blocks.append(ngram_ids + head_offsets.view(1, 1, -1))

        local_ids = torch.cat(blocks, dim=-1)[:, sequence_start:sequence_end]
        table_device = self.ngram_embedding.weight.device
        embeddings = self.ngram_embedding(local_ids.to(table_device))
        return embeddings.to(local_ids.device).flatten(-2)
