# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from gpatch_v4.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextTopKRouter
from gpatch_v4.models.qwen4_exp.router_replay import (
    get_topk_layer_indices,
    router_replay_ctx,
)
from gpatch_v4.training_backend.fsdp2_backend.mixin import RouterReplayMixin


def _router_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_experts_per_tok=2,
        num_experts=4,
        norm_topk_prob=True,
        hidden_size=3,
    )


class _TwoLayerQwenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gates = nn.ModuleList(
            [Qwen4ExpTextTopKRouter(_router_config()) for _ in range(2)]
        )


def test_qwen_router_replay_pins_indices_and_keeps_gradients():
    model = _TwoLayerQwenModel()
    router = model.gates[0]
    hidden_states = torch.randn(3, 3)
    target = torch.tensor([[0, 2], [1, 3], [2, 0]], dtype=torch.long)

    with router_replay_ctx(model, [target, target]):
        logits, replay_weights, replay_indices = router(hidden_states)

    probs = torch.softmax(logits, dtype=torch.float, dim=-1)
    expected_weights = probs.gather(1, target)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    assert torch.equal(replay_indices, target)
    assert torch.allclose(replay_weights, expected_weights.to(logits.dtype))
    assert router.router_replay is None

    replay_weights[:, 0].sum().backward()
    assert router.weight.grad is not None
    assert router.weight.grad.abs().sum() > 0


def test_qwen_topk_layer_indices_include_every_decoder_layer():
    text_config = SimpleNamespace(num_hidden_layers=4)
    config = SimpleNamespace(text_config=text_config)
    assert get_topk_layer_indices(config) == [0, 1, 2, 3]


def test_fsdp2_qwen_replay_uses_packed_route_rows():
    class _Stub(RouterReplayMixin):
        pass

    mixin = _Stub()
    mixin.policy_config = SimpleNamespace(model_arch="qwen4_exp")
    mixin.hf_config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2))
    mixin.config = SimpleNamespace(
        training=SimpleNamespace(moe_router_replay=True)
    )

    model = _TwoLayerQwenModel()
    routed_experts = torch.tensor(
        [
            [[0, 1], [2, 3]],
            [[1, 2], [3, 0]],
            [[2, 3], [0, 1]],
            [[3, 0], [1, 2]],
            [[3, 0], [1, 2]],
        ],
        dtype=torch.int32,
    )
    batches = [{"routed_experts": routed_experts}]
    hidden_states = torch.randn(4, 3)

    with patch("gpatch_v4.training_backend.fsdp2_backend.mixin.router_replay.mpu") as mock_mpu:
        mock_mpu.get_context_parallel_world_size.return_value = 1
        mock_mpu.get_context_parallel_rank.return_value = 0
        mock_mpu.get_data_parallel_rank.return_value = 0
        with mixin._maybe_router_replay(model, batches, seq_length=4):
            _, _, layer0_indices = model.gates[0](hidden_states)
            _, _, layer1_indices = model.gates[1](hidden_states)

    assert torch.equal(layer0_indices, routed_experts[:4, 0].long())
    assert torch.equal(layer1_indices, routed_experts[:4, 1].long())
