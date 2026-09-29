# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch

from megatron.core import mpu

from gpatch_v4.core.constants import MODEL_ARCH

try:
    from gpatch_v4.models.deepseek_v4.router_replay import (
        get_topk_layer_indices as _dsv4_get_topk_layer_indices,
    )
    from gpatch_v4.models.deepseek_v4.router_replay import (
        router_replay_ctx as _dsv4_router_replay_ctx,
    )
except ImportError:
    _dsv4_get_topk_layer_indices = None
    _dsv4_router_replay_ctx = None

try:
    from gpatch_v4.models.qwen4_exp.router_replay import (
        get_topk_layer_indices as _qwen4_get_topk_layer_indices,
    )
    from gpatch_v4.models.qwen4_exp.router_replay import (
        router_replay_ctx as _qwen4_router_replay_ctx,
    )
except Exception:
    _qwen4_get_topk_layer_indices = None
    _qwen4_router_replay_ctx = None


class RouterReplayOps(NamedTuple):
    get_topk_layer_indices: Callable
    router_replay_ctx: Callable


_ROUTER_REPLAY_OPS = {
    MODEL_ARCH.QWEN4_EXP:
        RouterReplayOps(
            get_topk_layer_indices=_qwen4_get_topk_layer_indices,
            router_replay_ctx=_qwen4_router_replay_ctx,
        ),
    MODEL_ARCH.DEEPSEEK_V4:
        RouterReplayOps(
            get_topk_layer_indices=_dsv4_get_topk_layer_indices,
            router_replay_ctx=_dsv4_router_replay_ctx,
        ),
}


def get_router_replay_ops(model_arch: Optional[str]) -> RouterReplayOps:
    """Per-arch ``get_topk_layer_indices`` + ``router_replay_ctx``.

    Missing / unknown ``model_arch`` uses DeepSeek-V4.
    """
    if model_arch not in _ROUTER_REPLAY_OPS:
        raise ValueError(f"Unknown model arch: {model_arch}")
    return _ROUTER_REPLAY_OPS[model_arch]


class RouterReplayMixin:
    """Router Replay (R3) support for FSDP2 backend.

    Replays sampler routing decisions during the actor's forward pass so
    that TopKRouter selects the same experts as the inference engine —
    eliminating routing drift caused by FP8/FP4 weight quantization.
    """
    def _get_router_replay_ops(self) -> RouterReplayOps:
        return get_router_replay_ops(self.policy_config.model_arch)

    def _get_topk_layer_indices(self) -> list[int]:
        ops = self._get_router_replay_ops()
        return ops.get_topk_layer_indices(self.hf_config)

    def _prepare_replay_indices(
        self,
        batches: List[Dict[str, Any]],
        seq_length: int,
        packed_seq_params=None,
    ) -> Optional[list[torch.Tensor]]:
        """Build per-TopKRouter-layer replay indices from a micro-batch.

        Parameters
        ----------
        batches : list[dict]
            Micro-batch dicts; each must contain ``"routed_experts"``
            of shape ``(sample_seq_len, num_all_layers, topk)``.
        seq_length : int
            Padded sequence length for this step.

        Returns
        -------
        list[Tensor] or None
            One ``(B * seq_length, topk)`` int64 tensor per TopKRouter
            layer, ready for :func:`router_replay_ctx`. ``None`` if the
            batch has no ``routed_experts``.
        """
        has_routed_experts = [batch.get("routed_experts") is not None for batch in batches]
        if not any(has_routed_experts):
            return None
        if not all(has_routed_experts):
            raise RuntimeError(
                "moe_router_replay requires routed_experts for every sample in a microbatch"
            )

        topk_indices = self._get_topk_layer_indices()
        if len(batches) == 1 and "local_cp_size" in batches[0]:
            routed_experts = batches[0]["routed_experts"]
            assert routed_experts.ndim == 3
            return [
                routed_experts[:, layer_index, :].contiguous().long()
                for layer_index in topk_indices
            ]
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
        dp_rank = mpu.get_data_parallel_rank()

        per_layer_accum: list[list[torch.Tensor]] = [[] for _ in topk_indices]

        if packed_seq_params is not None:
            padded_boundaries = packed_seq_params.cu_seqlens_q_padded.detach().cpu().tolist()
            assert len(padded_boundaries) == len(batches) + 1
            for sample_idx, batch in enumerate(batches):
                re = batch["routed_experts"]
                seq_batch = re.shape[0] - 1
                padded_len = padded_boundaries[sample_idx + 1] - padded_boundaries[sample_idx]
                assert seq_batch > 0 and padded_len >= seq_batch
                full_index = torch.arange(padded_len, device=re.device)
                index = (full_index + (full_index // seq_batch) * 6 * dp_rank) % seq_batch
                padded = re[index]
                for out_i, layer_i in enumerate(topk_indices):
                    per_layer_accum[out_i].append(padded[:, layer_i, :])

            packed_per_layer = [
                torch.cat(chunks, dim=0).contiguous().long() for chunks in per_layer_accum
            ]
            total_tokens = packed_seq_params.total_seqlen
            assert all(rows.shape[0] == total_tokens for rows in packed_per_layer)
            if cp_size > 1:
                assert total_tokens % cp_size == 0
                local_tokens = total_tokens // cp_size
                start = cp_rank * local_tokens
                packed_per_layer = [
                    rows[start:start + local_tokens].contiguous() for rows in packed_per_layer
                ]
            return packed_per_layer

        for batch in batches:
            re = batch["routed_experts"]
            seq_batch = re.shape[0] - 1

            full_index = torch.arange(seq_length, device=re.device)
            index = (full_index + (full_index // seq_batch) * 6 * dp_rank) % seq_batch

            if cp_size > 1:
                s_local = seq_length // cp_size
                start = cp_rank * s_local
                index = index[start:start + s_local]

            padded = re[index]

            for out_i, layer_i in enumerate(topk_indices):
                per_layer_accum[out_i].append(padded[:, layer_i, :])

        return [torch.cat(chunks, dim=0).contiguous().long() for chunks in per_layer_accum]

    @contextmanager
    def _maybe_router_replay(
        self,
        model,
        batches: List[Dict[str, Any]],
        seq_length: int,
        packed_seq_params=None,
    ):
        """Scope router replay for one micro-batch forward **and** backward.

        ``router_replay_ctx`` tears down replay state on exit. When gradient
        checkpointing is enabled the backward pass re-runs forward (recompute)
        and still needs the pinned ``expert_idx`` from rollout — callers must
        keep this context open until ``loss.backward()`` returns.
        """
        if not self.config.training.moe_router_replay:
            yield
            return

        indices = self._prepare_replay_indices(
            batches, seq_length, packed_seq_params=packed_seq_params
        )
        if indices is None:
            raise RuntimeError(
                "moe_router_replay=True but the batch contains no routed_experts"
            )

        ops = self._get_router_replay_ops()
        with ops.router_replay_ctx(model, indices):
            yield
