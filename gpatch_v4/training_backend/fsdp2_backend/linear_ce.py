from __future__ import annotations

import contextvars
import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import nn

from gpatch_v4.kernel import linear_cross_entropy, set_linear_ce_backend
from gpatch_v4.utils import log

_ORIGINAL_FORWARD_ATTR = "_gpatch_v4_linear_ce_original_forward"


@dataclass(frozen=True)
class LinearCERequest:
    labels: torch.Tensor
    backend: str
    temperature: float
    return_entropy: bool
    loss_mask: torch.Tensor | None


_linear_ce_request: contextvars.ContextVar[LinearCERequest | None] = (
    contextvars.ContextVar("fsdp2_linear_ce_request", default=None)
)


def resolve_output_head(model: nn.Module) -> nn.Linear:
    head = model.get_output_embeddings()
    if head is None:
        head = model.get_submodule("lm_head")
    assert isinstance(head, nn.Linear
                     ), (f"FSDP2 linear CE requires an nn.Linear output head, got {type(head)}")
    assert head.bias is None, "FSDP2 linear CE does not support an output-head bias"
    return head


def _linear_ce_head_forward(
    head: nn.Linear,
    hidden_states: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    request = _linear_ce_request.get()
    if request is None:
        original_forward = getattr(head, _ORIGINAL_FORWARD_ATTR)
        return original_forward(hidden_states, *args, **kwargs)

    assert not args and not kwargs, "FSDP2 linear CE output head accepts only hidden states"
    assert hidden_states.shape[:-1] == request.labels.shape, (
        f"{hidden_states.shape=} does not match {request.labels.shape=}"
    )
    assert hidden_states.device == request.labels.device
    assert request.labels.dtype == torch.long

    set_linear_ce_backend(request.backend)
    safe_labels = request.labels.clamp_min(0)
    if request.loss_mask is not None:
        assert request.loss_mask.shape == request.labels.shape
        assert request.loss_mask.device == hidden_states.device
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        keep_indices = request.loss_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        num_valid_tokens = keep_indices.numel()
        if num_valid_tokens == 0:
            # Keep both gradient paths when a CP rank has no supervised tokens.
            zero = (flat_hidden[:0].sum() + head.weight[:0].sum()).float()
            zero = zero.expand_as(request.labels)
            return (-zero, zero) if request.return_entropy else -zero
        hidden_states = flat_hidden.index_select(0, keep_indices)
        safe_labels = safe_labels.reshape(-1).index_select(0, keep_indices)
        if num_valid_tokens == 1:
            # Triton specializes a single token to a scalar.
            hidden_states = torch.cat((hidden_states, hidden_states), dim=0)
            safe_labels = torch.cat((safe_labels, safe_labels))

    ce_out = linear_cross_entropy(
        hidden_states.contiguous(),
        head.weight,
        safe_labels.contiguous(),
        request.temperature,
        "none",
        return_entropy=request.return_entropy,
    )
    if request.return_entropy:
        nll, entropy = ce_out
        if request.loss_mask is not None:
            entropy = entropy.new_zeros(request.labels.numel()).scatter(
                0, keep_indices, entropy[:num_valid_tokens]
            ).view_as(request.labels)
    else:
        nll = ce_out
    if request.loss_mask is not None:
        nll = nll.new_zeros(request.labels.numel()).scatter(
            0, keep_indices, nll[:num_valid_tokens]
        ).view_as(request.labels)
    if request.return_entropy:
        return -nll, entropy
    return -nll


def install_linear_ce_head_bypass(model: nn.Module) -> None:
    head = resolve_output_head(model)
    if hasattr(head, _ORIGINAL_FORWARD_ATTR):
        return
    setattr(head, _ORIGINAL_FORWARD_ATTR, head.forward)
    head.forward = types.MethodType(_linear_ce_head_forward, head)
    log(f"Installed linear CE head bypass for lm_head", rank=0)


@contextmanager
def linear_ce_head_context(
    head: nn.Module,
    labels: torch.Tensor,
    backend: str,
    temperature: float = 1.0,
    return_entropy: bool = False,
    *,
    loss_mask: torch.Tensor | None = None,
) -> Iterator[None]:
    """Activate fused CE for ``head.forward``; nestable (MTP depth labels)."""
    assert hasattr(head, _ORIGINAL_FORWARD_ATTR), "Linear CE head bypass is not installed"
    request = LinearCERequest(
        labels=labels,
        backend=backend,
        temperature=temperature,
        return_entropy=return_entropy,
        loss_mask=loss_mask,
    )
    token = _linear_ce_request.set(request)
    try:
        yield
    finally:
        _linear_ce_request.reset(token)


@contextmanager
def linear_ce_forward_context(
    model: nn.Module,
    labels: torch.Tensor,
    backend: str,
    temperature: float = 1.0,
    return_entropy: bool = False,
    *,
    loss_mask: torch.Tensor | None = None,
) -> Iterator[None]:
    head = resolve_output_head(model)
    with linear_ce_head_context(
        head, labels, backend, temperature, return_entropy=return_entropy, loss_mask=loss_mask
    ):
        yield
