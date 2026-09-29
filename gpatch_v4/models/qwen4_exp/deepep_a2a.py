# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Autograd-aware DeepEP dispatch and combine for Qwen4-Exp."""

from __future__ import annotations

import torch
import torch.distributed as dist

try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle
    try:
        from deep_ep.utils import EventOverlap
    except ImportError:
        from deep_ep.utils.event import EventOverlap

    HAVE_DEEP_EP = True
except ImportError:
    Buffer = None
    EventHandle = None
    EventOverlap = None
    HAVE_DEEP_EP = False

_buffer = None


def _require_deepep() -> None:
    if not HAVE_DEEP_EP:
        raise ImportError("DeepEP backend requested, but `deep_ep` is not installed.")


def _get_hidden_bytes(x: torch.Tensor) -> int:
    return x.size(1) * max(x.element_size(), 2)


def _get_buffer(group: dist.ProcessGroup, hidden_bytes: int):
    _require_deepep()
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
            num_nvl_bytes,
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
            num_rdma_bytes,
        )

    if (
        _buffer is None or _buffer.group != group or _buffer.num_nvl_bytes < num_nvl_bytes or
        _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


class _FusedDispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        previous_event = EventOverlap(EventHandle()) if async_finish else None
        buffer = _get_buffer(group, _get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        if async_finish:
            after_event.current_stream_wait()

        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        tokens_per_expert = torch.tensor(
            num_recv_tokens_per_expert_list,
            dtype=torch.int32,
            device=recv_x.device,
        )
        return (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            tokens_per_expert,
            handle,
        )

    @staticmethod
    def backward(
        ctx,
        grad_output,
        grad_token_indices,
        grad_token_probs,
        grad_tokens_per_expert,
        grad_handle,
    ):
        buffer = _get_buffer(ctx.group, _get_hidden_bytes(grad_output))
        previous_event = EventOverlap(EventHandle()) if ctx.async_finish else None
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            ctx.handle,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None


class _FusedCombine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        group,
        handle,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        previous_event = EventOverlap(EventHandle()) if async_finish else None
        buffer = _get_buffer(group, _get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        if async_finish:
            after_event.current_stream_wait()

        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x

    @staticmethod
    def backward(ctx, grad_output):
        previous_event = EventOverlap(EventHandle()) if ctx.async_finish else None
        buffer = _get_buffer(ctx.group, _get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None


def fused_dispatch(
    x: torch.Tensor,
    token_indices: torch.Tensor,
    token_probs: torch.Tensor,
    num_experts: int,
    group: dist.ProcessGroup,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
):
    """Dispatch tokens and top-k metadata to their expert-owning ranks.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``[num_tokens, hidden_size]``.
    token_indices : torch.Tensor
        Shape ``[num_tokens, top_k]``; global expert ids encoded as int64.
    token_probs : torch.Tensor
        Shape ``[num_tokens, top_k]``; internally communicated as float32.
    num_experts : int
    group : dist.ProcessGroup
    async_finish : bool, default False
    allocate_on_comm_stream : bool, default False

    Returns
    -------
    tuple
        Received token rows, local expert slots, routing weights, per-expert
        counts, and the opaque layout handle required by :func:`fused_combine`.

    Raises
    ------
    ImportError
        If the ordinary DeepEP ``Buffer`` API is unavailable.
    TypeError
        If expert ids are not int64.
    ValueError
        If expert ids are not a two-dimensional top-k tensor.
    """
    _require_deepep()
    if token_indices.dtype != torch.int64:
        raise TypeError(
            f"DeepEP requires token_indices dtype torch.int64, got {token_indices.dtype}"
        )
    if token_indices.dim() != 2:
        raise ValueError(
            "DeepEP requires token_indices shape [num_tokens, top_k], "
            f"got {tuple(token_indices.shape)}"
        )
    return _FusedDispatch.apply(
        x.contiguous(),
        token_indices.contiguous(),
        token_probs.float().contiguous(),
        num_experts,
        group,
        async_finish,
        allocate_on_comm_stream,
    )


def fused_combine(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    handle,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
) -> torch.Tensor:
    """Return local outputs to source tokens and reduce their top-k slots.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``[num_received_tokens, hidden_size]`` in dispatch layout.
    group : dist.ProcessGroup
    handle : object
        Opaque layout handle returned by :func:`fused_dispatch`.
    async_finish : bool, default False
    allocate_on_comm_stream : bool, default False

    Returns
    -------
    torch.Tensor
        Shape ``[num_source_tokens, hidden_size]``.

    Raises
    ------
    ImportError
        If the ordinary DeepEP ``Buffer`` API is unavailable.
    """
    _require_deepep()
    return _FusedCombine.apply(
        x.contiguous(),
        group,
        handle,
        async_finish,
        allocate_on_comm_stream,
    )


def set_deepep_num_sms(num_sms: int) -> None:
    """Set the SM budget used by ordinary DeepEP kernels.

    Parameters
    ----------
    num_sms : int
    """
    _require_deepep()
    Buffer.set_num_sms(num_sms)
