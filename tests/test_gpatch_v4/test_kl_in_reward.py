"""Unit tests for verl-style KL-in-reward."""

from types import SimpleNamespace

import pytest
import torch

from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.core.kl_ctrl import (
    AdaptiveKLController,
    FixedKLController,
    apply_kl_in_reward,
    get_kl_in_reward_controller,
    update_kl_in_reward_controller,
)


def test_apply_kl_in_reward_subtracts_token_sum_k1():
    rewards = [torch.tensor(1.0), torch.tensor(2.0)]
    old_logprobs = [
        torch.tensor([0.0, -1.0, -2.0]),
        torch.tensor([-0.5, -0.5]),
    ]
    ref_logprobs = [
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0]),
    ]
    mask = [
        torch.tensor([1.0, 1.0, 0.0]),
        torch.tensor([1.0, 1.0]),
    ]
    kl_ctrl = FixedKLController(kl_coef=0.1)

    result = apply_kl_in_reward(rewards, old_logprobs, ref_logprobs, mask, kl_ctrl)

    # sample0: sum_t (old-ref)*mask = 0 + (-1) = -1 → reward = 1 - 0.1*(-1) = 1.1
    # sample1: sum = -0.5 + -0.5 = -1 → reward = 2 - 0.1*(-1) = 2.1
    assert torch.allclose(result.rewards[0], torch.tensor(1.1))
    assert torch.allclose(result.rewards[1], torch.tensor(2.1))
    assert torch.allclose(result.rewards_before[0], torch.tensor(1.0))
    assert torch.allclose(result.seq_mean_kls[0], torch.tensor(-0.5))
    assert torch.allclose(result.seq_mean_kls[1], torch.tensor(-0.5))
    assert torch.allclose(torch.stack(result.kl_sums), torch.tensor([-1.0, -1.0]))
    assert result.metrics["ppo-metrics/kl_in_reward_coef"] == pytest.approx(0.1)
    assert result.metrics["ppo-metrics/kl_in_reward"] == pytest.approx(-0.1)


def test_adaptive_controller_update_formula():
    ctrl = AdaptiveKLController(init_kl_coef=0.001, target_kl=0.1, horizon=10000)
    # current_kl / target - 1 = 0.2 / 0.1 - 1 = 1 → clamped to 0.2
    ctrl.update(current_kl=0.2, n_steps=8)
    expected = 0.001 * (1.0 + 0.2 * 8 / 10000)
    assert ctrl.value == pytest.approx(expected)


def test_get_controller_types():
    fixed = get_kl_in_reward_controller(
        SimpleNamespace(
            kl_in_reward_ctrl_type="fixed",
            kl_in_reward_coef=0.002,
            kl_in_reward_target_kl=0.1,
            kl_in_reward_horizon=10000,
        )
    )
    assert isinstance(fixed, FixedKLController)
    assert fixed.value == pytest.approx(0.002)

    adaptive = get_kl_in_reward_controller(
        SimpleNamespace(
            kl_in_reward_ctrl_type="adaptive",
            kl_in_reward_coef=0.001,
            kl_in_reward_target_kl=0.1,
            kl_in_reward_horizon=10000,
        )
    )
    assert isinstance(adaptive, AdaptiveKLController)


def test_update_controller_reports_coef_used_this_step(monkeypatch):
    ctrl = AdaptiveKLController(init_kl_coef=0.001, target_kl=0.1, horizon=10000)
    # mean KL = 0.2 → proportional error clamped to +0.2, so coef must increase.
    seq_means = [torch.tensor(0.2), torch.tensor(0.2)]
    coef_used = ctrl.value

    def fake_all_reduce(tensor, group=None):
        return tensor

    monkeypatch.setattr("gpatch_v4.core.kl_ctrl.dist.all_reduce", fake_all_reduce)
    metrics = update_kl_in_reward_controller(ctrl, seq_means, dp_group=None)
    expected_new = coef_used * (1.0 + 0.2 * 2 / 10000)
    assert metrics["ppo-metrics/kl_in_reward_mean"] == pytest.approx(0.2)
    assert metrics["ppo-metrics/kl_in_reward_coef"] == pytest.approx(coef_used)
    assert metrics["ppo-metrics/kl_in_reward_coef_new"] == pytest.approx(expected_new)
    assert ctrl.value == pytest.approx(expected_new)
    assert metrics["ppo-metrics/kl_in_reward_coef_new"] > metrics["ppo-metrics/kl_in_reward_coef"]


def test_ppo_config_rejects_kl_in_reward_with_ppo_advantage():
    with pytest.raises(AssertionError, match="advantage_type='ppo'"):
        PpoConfig(use_kl_in_reward=True, advantage_type="ppo")


def test_ppo_config_rejects_kl_in_reward_with_skip_prev_logps():
    with pytest.raises(AssertionError, match="skip_prev_logps"):
        PpoConfig(use_kl_in_reward=True, skip_prev_logps=True)


def test_ppo_config_rejects_rs_without_threshold():
    with pytest.raises(AssertionError, match="off_policy_correction_rs_threshold"):
        PpoConfig(off_policy_correction_rs="token_k1")


def test_ppo_config_rejects_unknown_rs_option():
    with pytest.raises(AssertionError, match="unsupported off_policy_correction_rs"):
        PpoConfig(off_policy_correction_rs="token_k9", off_policy_correction_rs_threshold="1.0")


def test_ppo_config_rejects_unknown_ctrl_type():
    with pytest.raises(AssertionError, match="kl_in_reward_ctrl_type"):
        PpoConfig(kl_in_reward_ctrl_type="unknown")
