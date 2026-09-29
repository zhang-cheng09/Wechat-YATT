# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Shared MoE router-replay primitives.

Model-specific modules bind a router class via ``iter_fn`` and keep their
own ``get_topk_layer_indices``. Replay lives on each router as
``module.router_replay``; there is no process-global registry.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Generator, Iterable, Optional

import torch
import torch.nn as nn


class RouterReplay:
    """Per-layer replay state attached to a TopK router module.

    Replay is active iff ``target_topk_idx is not None``. Instances live on
    their owning module and are discovered via ``model.modules()``.
    """
    def __init__(self) -> None:
        self.target_topk_idx: Optional[torch.Tensor] = None

    def set_target_indices(self, idx: torch.Tensor) -> None:
        self.target_topk_idx = idx

    def get_replay_topk(
        self,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(values, indices)`` by gathering *scores* at the saved target.

        Caller MUST set ``target_topk_idx`` first; the no-target path is
        handled at the router ``forward`` call site.
        """
        assert self.target_topk_idx is not None, (
            "target_topk_idx must be set before get_replay_topk; "
            "caller is responsible for the no-replay path"
        )
        indices = self.target_topk_idx.to(scores.device)
        return scores.gather(1, indices), indices


def iter_routers(
    model: nn.Module,
    router_cls: type[nn.Module],
) -> Generator[nn.Module, None, None]:
    for module in model.modules():
        if isinstance(module, router_cls):
            yield module


def extract_topk_layers(
    routed_experts: torch.Tensor,
    topk_layer_indices: list[int],
) -> list[torch.Tensor]:
    """Slice ``routed_experts`` to the TopKRouter layers in order.

    Parameters
    ----------
    routed_experts : Tensor
        Shape ``(seq_len, num_all_layers, topk)``.
    topk_layer_indices : list[int]

    Returns
    -------
    list[Tensor]
        One ``(seq_len, topk)`` tensor per index in ``topk_layer_indices``.
    """
    return [routed_experts[:, i, :] for i in topk_layer_indices]


def enable_router_replay(
    model: nn.Module,
    *,
    iter_fn: Callable[[nn.Module], Iterable[nn.Module]],
    empty_message: str,
) -> list[RouterReplay]:
    instances: list[RouterReplay] = []
    for module in iter_fn(model):
        if module.router_replay is None:
            module.router_replay = RouterReplay()
        instances.append(module.router_replay)
    if not instances:
        raise RuntimeError(empty_message)
    return instances


def disable_router_replay(
    model: nn.Module,
    *,
    iter_fn: Callable[[nn.Module], Iterable[nn.Module]],
) -> None:
    for module in iter_fn(model):
        module.router_replay = None


@contextmanager
def router_replay_ctx(
    model: nn.Module,
    per_layer_indices: list[torch.Tensor],
    *,
    iter_fn: Callable[[nn.Module], Iterable[nn.Module]],
    empty_message: str,
) -> Generator[None, None, None]:
    routers = enable_router_replay(model, iter_fn=iter_fn, empty_message=empty_message)
    if len(routers) != len(per_layer_indices):
        disable_router_replay(model, iter_fn=iter_fn)
        raise ValueError(
            f"Expected {len(routers)} index tensors (one per layer), "
            f"got {len(per_layer_indices)}"
        )
    for replay, indices in zip(routers, per_layer_indices, strict=True):
        replay.set_target_indices(indices)
    try:
        yield
    finally:
        disable_router_replay(model, iter_fn=iter_fn)


@contextmanager
def capture_routing_decisions(
    model: nn.Module,
    *,
    router_class_names: tuple[str, ...],
) -> Generator[list[Optional[torch.Tensor]], None, None]:
    """Record per-layer routing indices via forward hooks.

    Matches gate modules by ``type(module).__name__``. Each captured
    ``out`` is ``(router_logits, router_scores, router_indices)``; the
    hook stores ``out[2]``.
    """
    recorded: list[Optional[torch.Tensor]] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for module in model.modules():
        if type(module).__name__ in router_class_names:
            slot = len(recorded)
            recorded.append(None)

            def _make_hook(i: int):
                def hook(mod, inp, out):
                    recorded[i] = out[2].detach().clone()

                return hook

            handles.append(module.register_forward_hook(_make_hook(slot)))

    try:
        yield recorded
    finally:
        for h in handles:
            h.remove()
