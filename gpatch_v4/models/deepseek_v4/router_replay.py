# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""DeepSeek-V4 router-replay facade.

Shared attach/capture lives in :mod:`gpatch_v4.models.comm.router_replay`.
This module binds :class:`DeepseekV4TopKRouter` and skips ``hash_moe``
layers (:class:`DeepseekV4HashRouter` is determined by ``input_ids``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from gpatch_v4.models.comm.router_replay import RouterReplay
from gpatch_v4.models.comm.router_replay import (
    capture_routing_decisions as _capture_routing_decisions,
)
from gpatch_v4.models.comm.router_replay import (
    disable_router_replay as _disable_router_replay,
)
from gpatch_v4.models.comm.router_replay import (
    enable_router_replay as _enable_router_replay,
)
from gpatch_v4.models.comm.router_replay import extract_topk_layers, iter_routers
from gpatch_v4.models.comm.router_replay import router_replay_ctx as _router_replay_ctx

if TYPE_CHECKING:
    from typing import Generator

_EMPTY_ROUTERS = (
    "No DeepseekV4TopKRouter found in model; "
    "use capture_routing_decisions for upstream transformers models"
)

__all__ = [
    "RouterReplay",
    "capture_routing_decisions",
    "disable_router_replay",
    "enable_router_replay",
    "extract_topk_layers",
    "get_topk_layer_indices",
    "router_replay_ctx",
]


def get_topk_layer_indices(config) -> list[int]:
    """Decoder-layer indices that use ``DeepseekV4TopKRouter``.

    Hash-MoE layers are excluded. Indices refer to the ``num_layers``
    axis of sampler ``routed_experts``.
    """
    mlp_layer_types = config.mlp_layer_types
    return [i for i, t in enumerate(mlp_layer_types) if t != "hash_moe"]


def _iter_routers(model: nn.Module):
    from gpatch_v4.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4TopKRouter

    yield from iter_routers(model, DeepseekV4TopKRouter)


def enable_router_replay(model: nn.Module) -> list[RouterReplay]:
    return _enable_router_replay(model, iter_fn=_iter_routers, empty_message=_EMPTY_ROUTERS)


def disable_router_replay(model: nn.Module) -> None:
    _disable_router_replay(model, iter_fn=_iter_routers)


def router_replay_ctx(
    model: nn.Module,
    per_layer_indices: list[torch.Tensor],
) -> Generator[None, None, None]:
    return _router_replay_ctx(
        model,
        per_layer_indices,
        iter_fn=_iter_routers,
        empty_message=_EMPTY_ROUTERS,
    )


def capture_routing_decisions(
    model: nn.Module,
    *,
    router_class_names: tuple[str, ...] = ("DeepseekV4TopKRouter", ),
) -> Generator[list[Optional[torch.Tensor]], None, None]:
    return _capture_routing_decisions(model, router_class_names=router_class_names)
