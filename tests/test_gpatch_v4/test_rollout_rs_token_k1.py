"""Unit tests for verl-aligned token_k1 rejection sampling."""

from types import SimpleNamespace

import math
import torch

from gpatch_v4.core.correction_helper import (
    compute_off_policy_correction_weights,
    compute_rollout_rejection_mask,
)


def _make_config(
    rs="token_k1",
    rs_threshold="0.5_5.0",
    upper_bound=2.0,
    mode="truncate",
):
    return SimpleNamespace(
        ppo=SimpleNamespace(
            off_policy_correction_level="token",
            off_policy_correction_mode=mode,
            off_policy_correction_upper_bound=upper_bound,
            off_policy_correction_lower_bound=None,
            off_policy_correction_veto_threshold=None,
            off_policy_correction_rs=rs,
            off_policy_correction_rs_threshold=rs_threshold,
        )
    )


def test_token_k1_keeps_is_weight_in_0p2_to_2():
    # IS weight = exp(prev - rollout). With rollout=0: weight=exp(prev).
    prev = torch.tensor([[0.0, math.log(3.0), math.log(0.1), math.log(1.0)]])
    roll = torch.zeros_like(prev)
    mask = torch.ones_like(prev)
    rs_mask, _ = compute_rollout_rejection_mask(prev - roll, mask, "token_k1", "0.5_5.0")
    # weights: 1.0 (keep), 3.0 (reject >2), 0.1 (reject <0.2), 1.0 (keep)
    assert torch.equal(rs_mask, torch.tensor([[1.0, 0.0, 0.0, 1.0]]))


def test_token_k1_via_off_policy_correction_weights():
    prev = torch.tensor([[0.0, math.log(3.0), math.log(0.1), math.log(1.0)]])
    roll = torch.zeros_like(prev)
    mask = torch.ones_like(prev)
    weights, modified_mask, metrics = compute_off_policy_correction_weights(
        True,
        _make_config(),
        prev,
        roll,
        mask,
    )
    assert torch.equal(modified_mask, torch.tensor([[1.0, 0.0, 0.0, 1.0]]))
    assert torch.allclose(weights[0, 0], torch.tensor(1.0))
    assert torch.allclose(weights[0, 1], torch.tensor(2.0))  # truncated by IS upper=2
    assert "off_policy_correction/rollout_rs_token_k1_masked_fraction" in metrics
    assert metrics["off_policy_correction/rollout_rs_masked_fraction"].item() == 0.5


def test_rs_applies_when_importance_sampling_is_off():
    prev = torch.tensor([[0.0, math.log(3.0), math.log(0.1), math.log(1.0)]])
    roll = torch.zeros_like(prev)
    mask = torch.ones_like(prev)
    weights, modified_mask, metrics = compute_off_policy_correction_weights(
        False,
        _make_config(),
        prev,
        roll,
        mask,
    )
    assert weights is None
    assert torch.equal(modified_mask, torch.tensor([[1.0, 0.0, 0.0, 1.0]]))
    assert metrics["off_policy_correction/rollout_rs_masked_fraction"].item() == 0.5


def test_rs_without_rollout_logprobs_raises():
    prev = torch.zeros(1, 2)
    mask = torch.ones(1, 2)
    try:
        compute_off_policy_correction_weights(False, _make_config(), prev, None, mask)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "rollout_log_probs" in str(exc)


def test_rs_disabled_keeps_full_mask():
    prev = torch.tensor([[math.log(3.0), math.log(0.1)]])
    roll = torch.zeros_like(prev)
    mask = torch.ones_like(prev)
    _, modified_mask, metrics = compute_off_policy_correction_weights(
        True,
        _make_config(rs=None, rs_threshold=None),
        prev,
        roll,
        mask,
    )
    assert torch.equal(modified_mask, mask)
    assert not any(k.startswith("off_policy_correction/rollout_rs") for k in metrics)


def test_token_k2_rejects_above_upper():
    # token_k2 = 0.5 * log_ratio^2; log_ratio=log(3) → 0.5*(log 3)^2 ≈ 0.603
    prev = torch.tensor([[0.0, math.log(3.0)]])
    roll = torch.zeros_like(prev)
    mask = torch.ones_like(prev)
    rs_mask, metrics = compute_rollout_rejection_mask(prev - roll, mask, "token_k2", "0.3")
    assert torch.equal(rs_mask, torch.tensor([[1.0, 0.0]]))
    assert metrics["rollout_rs_token_k2_masked_fraction"].item() == 0.5
