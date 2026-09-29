"""本次 loss 重构新增：value loss 单测（纯 CPU）。"""
from types import SimpleNamespace

import torch

from gpatch_v4.training_backend.loss import get_loss_fn
from gpatch_v4.training_backend.loss.mcore_specific_loss import value_loss_func
from gpatch_v4.training_backend.loss_factory import get_policy_loss_fn


class TestValueLoss:
    def test_without_clip(self):
        values = torch.tensor([[1.0, 3.0]], requires_grad=True)
        returns = torch.tensor([[1.0, 1.0]])
        mask = torch.tensor([[1.0, 1.0]])
        old_values = torch.zeros_like(values)
        loss, metrics = value_loss_func(
            SimpleNamespace(ppo_value_clip=None),
            old_values,
            values,
            returns,
            mask,
        )
        assert torch.allclose(loss, torch.tensor(1.0))
        assert torch.allclose(metrics["value_loss"], torch.tensor([2.0, 2.0]))
        loss.backward()
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()

    def test_with_clip_uses_max_of_clipped_and_unclipped(self):
        values = torch.tensor([[3.0]], requires_grad=True)
        old_values = torch.tensor([[1.0]])
        returns = torch.tensor([[1.0]])
        mask = torch.tensor([[1.0]])
        loss, _ = value_loss_func(
            SimpleNamespace(ppo_value_clip=0.5),
            old_values,
            values,
            returns,
            mask,
        )
        # unclipped (3-1)^2=4 > clipped (1.5-1)^2=0.25，取 max 后再 *0.5
        assert torch.allclose(loss, torch.tensor(2.0))
        loss.backward()
        assert values.grad is not None

    def test_legacy_and_new_match_for_clip_none_and_0_2(self):
        legacy_fn = get_policy_loss_fn("ppo_value_loss")
        new_fn = get_loss_fn("mcore", "ppo_value_loss")
        cases = (
            (
                None,
                torch.tensor([[1.0, 3.0, 0.5]]),
                torch.zeros(1, 3),
                torch.tensor([[1.0, 1.0, 2.0]]),
                torch.tensor([[1.0, 1.0, 0.0]]),
            ),
            (
                0.2,
                torch.tensor([[2.0, -0.5]]),
                torch.tensor([[1.0, 0.5]]),
                torch.tensor([[0.0, 1.0]]),
                torch.tensor([[1.0, 1.0]]),
            ),
        )
        for ppo_value_clip, values, old_values, returns, mask in cases:
            ppo_config = SimpleNamespace(ppo_value_clip=ppo_value_clip)
            values_l = values.detach().clone().requires_grad_(True)
            values_n = values.detach().clone().requires_grad_(True)
            loss_l, metrics_l = legacy_fn(
                ppo_config, old_values, values_l, returns, mask
            )
            loss_n, metrics_n = new_fn(
                ppo_config, old_values, values_n, returns, mask
            )
            torch.testing.assert_close(loss_l, loss_n)
            torch.testing.assert_close(metrics_l["value_loss"], metrics_n["value_loss"])
            loss_l.backward()
            loss_n.backward()
            torch.testing.assert_close(values_l.grad, values_n.grad)
