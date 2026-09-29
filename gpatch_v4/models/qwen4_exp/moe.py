# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Expert-parallel MoE for Qwen3.8-Flash-Next.

512 routed experts × 48 layers is ~120.8 B of the 125 B backbone, so the experts must be
partitioned across ranks. Each rank owns a contiguous slice of
``num_experts / ep_size`` experts; a forward ships each (token, top-k slot) pair to the
rank owning its expert, computes locally, and ships the result back.

Structure follows :class:`gpatch_v4.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4Experts`
``_forward_eager`` / ``fwd_gmm``, which is the validated pattern in this repo and uses the
identical 3-D stacked weight layout (``gate_up_proj [E, 2I, H]``, ``down_proj [E, H, I]``).
Differences: no FP8/FP4 QAT paths, and no ``swiglu_limit`` clamp — upstream
``Qwen4ExpTextExperts`` applies a plain ``act_fn(gate) * up``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.integrations.moe import _grouped_linear
from typing_extensions import override

from .a2a import all_to_all_uneven
from .deepep_a2a import fused_combine, fused_dispatch
from .modeling_qwen4_exp import Qwen4ExpTextExperts, Qwen4ExpTextTopKRouter

__all__ = ["Qwen4ExpEPExperts", "freeze_router_weights"]


def freeze_router_weights(model: nn.Module) -> None:
    """Freeze every TopK router weight before optimizer construction."""
    for module in model.modules():
        if isinstance(module, Qwen4ExpTextTopKRouter):
            module.weight.requires_grad_(False)


class _EmptyExpertsWithGrad(torch.autograd.Function):
    """Zero-token stub that keeps expert parameters in the autograd graph.

    Grouped MM never touches the weights when there are no rows, which would leave the
    expert parameters out of the graph so FSDP2 may not produce zero grads for them.
    Ported from DSV4, where the same failure mode was hit under activation checkpointing.
    """
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        top_k_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ctx.gate_up_meta = (gate_up_proj.shape, gate_up_proj.dtype, gate_up_proj.device)
        ctx.down_meta = (down_proj.shape, down_proj.dtype, down_proj.device)
        ctx.weight_meta = None
        if top_k_weights is not None:
            ctx.weight_meta = (
                top_k_weights.shape,
                top_k_weights.dtype,
                top_k_weights.device,
            )
        return torch.zeros_like(hidden_states)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        def zeros(meta):
            shape, dtype, device = meta
            return torch.zeros(shape, dtype=dtype, device=device)

        weight_grad = None if ctx.weight_meta is None else zeros(ctx.weight_meta)
        return grad_output, zeros(ctx.gate_up_meta), zeros(ctx.down_meta), weight_grad


def _deterministic_slot_reduce(
    expanded_output: torch.Tensor,
    valid_flat_idx: torch.Tensor,
    recv_expert: torch.Tensor,
) -> torch.Tensor:
    if recv_expert.ndim != 2:
        raise ValueError(f"recv_expert must be rank 2, got {tuple(recv_expert.shape)}")
    hidden_dim = expanded_output.shape[-1]
    slot_output = expanded_output.new_zeros(recv_expert.numel(), hidden_dim)
    slot_output.index_copy_(0, valid_flat_idx, expanded_output)
    return slot_output.view(
        recv_expert.shape[0],
        recv_expert.shape[1],
        hidden_dim,
    ).sum(dim=1)


class Qwen4ExpEPExperts(Qwen4ExpTextExperts):
    """Expert-parallel replacement for upstream's dense expert loop.

    Parameter names, shapes and dtypes are inherited unchanged, so the checkpoint FQNs
    ``mlp.experts.gate_up_proj`` / ``mlp.experts.down_proj`` still apply — after
    :func:`configure_ep` the leading dimension holds only this rank's slice.

    With ``ep_size == 1`` the forward delegates to upstream's implementation; that is the
    correct behaviour when EP is off, not a fallback.
    """
    selective_gradient_checkpointing = False
    _gradient_checkpointing_func = None

    def __init__(
        self,
        config,
        ep_backend: str = "eager",
        deepep_async_finish: bool = False,
        ep_weight_before_combine: bool = False,
    ) -> None:
        super().__init__(config)
        self.ep_backend = ep_backend
        self.deepep_async_finish = deepep_async_finish
        self.ep_weight_before_combine = ep_weight_before_combine
        self.ep_size = 1
        self.ep_rank = 0
        self.ep_group: Optional[dist.ProcessGroup] = None
        self.num_local_experts = self.num_experts
        self._layer_idx = -1
        self._deepep_probe_dir: Optional[str] = None
        self._deepep_probe_call_index = 0
        self._deterministic_deepep_reduce = False

    def _forward_local_experts(
        self,
        hidden_states: torch.Tensor,
        local_expert_id: torch.Tensor,
        top_k_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run local expert GEMMs, optionally checkpointing only their compute.

        Dispatch/combine stay outside this checkpoint boundary.  Thus the backward
        replay recomputes local expert GEMMs but does not replay EP communication or
        consume a second DeepEP handle.
        """
        if (
            self.selective_gradient_checkpointing and self.training and
            self._gradient_checkpointing_func is not None
        ):
            if top_k_weights is None:
                return self._gradient_checkpointing_func(
                    self._local_experts,
                    hidden_states,
                    local_expert_id,
                    None,
                )
            return self._gradient_checkpointing_func(
                self._local_experts,
                hidden_states,
                local_expert_id,
                top_k_weights,
            )
        return self._local_experts(
            hidden_states,
            local_expert_id,
            top_k_weights,
        )

    def configure_ep(self, ep_group: Optional[dist.ProcessGroup]) -> None:
        """Record the expert-parallel group and this rank's local expert count.

        Does **not** slice the weights — the caller (``apply_hp``) owns that, because it
        must also distribute them onto the FSDP2 mesh as DTensors.

        Raises
        ------
        ValueError
            If ``num_experts`` is not divisible by the group size.
        """
        if ep_group is None:
            self.ep_size, self.ep_rank, self.ep_group = 1, 0, None
            self.num_local_experts = self.num_experts
            return

        ep_size = dist.get_world_size(ep_group)
        if self.num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by ep_size ({ep_size})."
            )
        self.ep_size = ep_size
        self.ep_rank = dist.get_rank(ep_group)
        self.ep_group = ep_group
        self.num_local_experts = self.num_experts // ep_size

    @override
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to expert owners, compute locally, combine back.

        Parameters
        ----------
        hidden_states : torch.Tensor
            ``[N, H]`` flat token activations (``N = B * S``).
        top_k_index : torch.Tensor
            ``[N, K]`` int64 global expert ids.
        top_k_weights : torch.Tensor
            ``[N, K]`` routing weights.

        Returns
        -------
        torch.Tensor
            ``[N, H]``, summed over the top-k slots.
        """
        if self.ep_size == 1:
            return super().forward(hidden_states, top_k_index, top_k_weights)
        if self.ep_backend == "eager":
            return self._forward_eager(hidden_states, top_k_index, top_k_weights)
        if self.ep_backend == "deepep":
            return self._forward_deepep(hidden_states, top_k_index, top_k_weights)
        raise ValueError(f"unknown ep_backend: {self.ep_backend!r}")

    def _forward_eager(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        assert self.ep_group is not None

        num_tokens, hidden_dim = hidden_states.shape
        top_k = top_k_index.shape[-1]

        # Each (token, slot) pair is an independent dispatch unit.
        flat_expert = top_k_index.reshape(-1)
        flat_weight = top_k_weights.reshape(-1)
        flat_hidden = hidden_states.repeat_interleave(top_k, dim=0)

        # Group by owning rank so each rank's payload is contiguous.
        target_rank = torch.div(flat_expert, self.num_local_experts, rounding_mode="floor")
        order = torch.argsort(target_rank, stable=True)
        send_counts = torch.bincount(target_rank, minlength=self.ep_size)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        recv_hidden = all_to_all_uneven(flat_hidden[order], send_splits, recv_splits, self.ep_group)
        recv_expert = all_to_all_uneven(flat_expert[order], send_splits, recv_splits, self.ep_group)
        local_expert_id = recv_expert - self.ep_rank * self.num_local_experts
        if self.ep_weight_before_combine:
            recv_weight = all_to_all_uneven(
                flat_weight[order], send_splits, recv_splits, self.ep_group
            )
            local_output = self._forward_local_experts(recv_hidden, local_expert_id, recv_weight)
        else:
            local_output = self._forward_local_experts(recv_hidden, local_expert_id)

        combined = all_to_all_uneven(local_output, recv_splits, send_splits, self.ep_group)
        inverse_order = torch.empty_like(order)
        inverse_order[order] = torch.arange(order.numel(), device=order.device)
        combined = combined[inverse_order]
        if not self.ep_weight_before_combine:
            combined = (combined * flat_weight.unsqueeze(-1)).to(hidden_states.dtype)
        return combined.view(num_tokens, top_k, hidden_dim).sum(dim=1)

    def _forward_deepep(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        assert self.ep_group is not None
        num_tokens, hidden_dim = hidden_states.shape
        recv_hidden, recv_expert, recv_weight, tokens_per_expert, handle = fused_dispatch(
            hidden_states,
            top_k_index,
            top_k_weights,
            self.num_experts,
            self.ep_group,
            async_finish=self.deepep_async_finish,
        )

        recv_expert_flat = recv_expert.reshape(-1)
        valid_flat_idx = (recv_expert_flat >= 0).nonzero(as_tuple=True)[0]
        row_idx = torch.div(
            valid_flat_idx,
            recv_expert.shape[-1],
            rounding_mode="floor",
        )
        local_expert_id = recv_expert_flat.index_select(0, valid_flat_idx)
        local_weight = recv_weight.reshape(-1).index_select(0, valid_flat_idx)
        expanded_hidden = recv_hidden.index_select(0, row_idx)
        expanded_output = self._forward_local_experts(
            expanded_hidden,
            local_expert_id,
            local_weight,
        )

        if self._deterministic_deepep_reduce:
            recv_output = _deterministic_slot_reduce(
                expanded_output,
                valid_flat_idx,
                recv_expert,
            )
        else:
            recv_output = recv_hidden.new_zeros(recv_hidden.shape[0], hidden_dim)
            recv_output.index_add_(0, row_idx, expanded_output)
        # fused_combine already returns ``[N, H]``; ``.view`` would make a FSDP2-unsafe alias.
        combined = fused_combine(
            recv_output,
            self.ep_group,
            handle,
            async_finish=self.deepep_async_finish,
        )
        if self._deepep_probe_dir is not None:
            from .diagnostics import write_deepep_forward_probe

            write_deepep_forward_probe(
                self._deepep_probe_dir,
                self._layer_idx,
                self._deepep_probe_call_index,
                tensors={
                    "input.hidden_states": hidden_states,
                    "input.top_k_index": top_k_index,
                    "input.top_k_weights": top_k_weights,
                    "dispatch.recv_hidden": recv_hidden,
                    "dispatch.recv_expert": recv_expert,
                    "dispatch.recv_weight": recv_weight,
                    "dispatch.tokens_per_expert": tokens_per_expert,
                    "expanded.row_idx": row_idx,
                    "expanded.local_expert_id": local_expert_id,
                    "expanded.local_weight": local_weight,
                    "expanded.hidden": expanded_hidden,
                    "expert.expanded_output": expanded_output,
                    "local_reduce.recv_output": recv_output,
                    "combine.output": combined,
                },
            )
            self._deepep_probe_call_index += 1
        return combined

    def _local_experts(
        self,
        hidden_states: torch.Tensor,
        local_expert_id: torch.Tensor,
        top_k_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Grouped SwiGLU over this rank's experts.

        Each row is already assigned 1:1 to one local expert, so unlike upstream's
        implementation there is no top-k expansion or sentinel masking to undo.

        Parameters
        ----------
        hidden_states : torch.Tensor
            ``[S, H]`` rows received by this rank.
        local_expert_id : torch.Tensor
            ``[S]`` int64 ids in ``[0, num_local_experts)``.
        top_k_weights : torch.Tensor, optional
            ``[S]`` routing weights. Applied before combine on both eager and DeepEP.

        Returns
        -------
        torch.Tensor
            ``[S, H]`` expert outputs, in the input row order.
        """
        if hidden_states.shape[0] == 0:
            return _EmptyExpertsWithGrad.apply(
                hidden_states,
                self.gate_up_proj,
                self.down_proj,
                top_k_weights,
            )

        # Grouped MM needs rows sorted by expert so each expert owns a contiguous span.
        sorted_ids, perm = torch.sort(local_expert_id)
        rows = hidden_states[perm]
        sorted_weights = None if top_k_weights is None else top_k_weights[perm]

        # `bincount` rather than DSV4's `histc`: histc has no integer CPU kernel
        # ("histogram_cpu not implemented for 'Int'"), which would confine these tests to
        # GPUs, and bincount is exact integer counting instead of float bucketing. An
        # out-of-range local id makes this longer than `num_local_experts`, which
        # `_grouped_linear` then rejects — a loud failure rather than silent corruption.
        tokens_per_expert = torch.bincount(
            sorted_ids,
            minlength=self.num_local_experts,
        )
        offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

        gate_up = _grouped_linear(
            rows.to(self.gate_up_proj.dtype),
            self.gate_up_proj,
            offsets,
            bias=None,
            is_transposed=False
        )
        gate, up = gate_up.chunk(2, dim=-1)
        activated = self.act_fn(gate) * up
        projected = _grouped_linear(
            activated.to(self.down_proj.dtype),
            self.down_proj,
            offsets,
            bias=None,
            is_transposed=False
        )
        if sorted_weights is not None:
            projected = projected * sorted_weights.unsqueeze(-1)

        inverse_perm = torch.empty_like(perm)
        inverse_perm[perm] = torch.arange(perm.shape[0], device=perm.device)
        return projected[inverse_perm].to(hidden_states.dtype)
