# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from gpatch_v4.models.comm.router_replay import RouterReplay
from gpatch_v4.models.comm.router_replay import (
    disable_router_replay as _disable_router_replay,
)
from gpatch_v4.models.comm.router_replay import (
    enable_router_replay as _enable_router_replay,
)
from gpatch_v4.models.comm.router_replay import iter_routers
from gpatch_v4.models.comm.router_replay import router_replay_ctx as _router_replay_ctx

if TYPE_CHECKING:
    from typing import Generator

_EMPTY_ROUTERS = "No Qwen4ExpTextTopKRouter found in model"


def get_topk_layer_indices(config) -> list[int]:
    return list(range(config.get_text_config().num_hidden_layers))


def _iter_routers(model: nn.Module):
    from gpatch_v4.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextTopKRouter

    backbone = model.model if hasattr(model, "model") else model
    yield from iter_routers(backbone, Qwen4ExpTextTopKRouter)


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
