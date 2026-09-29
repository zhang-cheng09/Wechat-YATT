"""KL controllers and reward-side KL penalty (verl-style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
import torch.distributed as dist


class AdaptiveKLController:
    """Adaptive KL controller from https://arxiv.org/pdf/1909.08593.pdf."""
    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int):
        self.value = float(init_kl_coef)
        self.target = float(target_kl)
        self.horizon = int(horizon)

    def update(self, current_kl: float, n_steps: int) -> None:
        """Update the KL coefficient from measured KL.

        Parameters
        ----------
        current_kl : float
            Mean per-token KL over the batch.
        n_steps : int
            Number of sequences in the batch (verl uses batch size).
        """
        proportional_error = max(min(current_kl / self.target - 1.0, 0.2), -0.2)
        mult = 1.0 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL coefficient controller."""
    def __init__(self, kl_coef: float):
        self.value = float(kl_coef)

    def update(self, current_kl: float, n_steps: int) -> None:
        """No-op for fixed controller."""


@dataclass
class KlInRewardResult:
    """Per-micro-batch output of :func:`apply_kl_in_reward`."""

    rewards: List[torch.Tensor]
    rewards_before: List[torch.Tensor]
    seq_mean_kls: List[torch.Tensor]
    kl_sums: List[torch.Tensor]
    metrics: Dict[str, float]


def get_kl_in_reward_controller(ppo_config: Any):
    """Build KL-in-reward controller from ``PpoConfig``.

    Parameters
    ----------
    ppo_config
        Object with ``kl_in_reward_ctrl_type``, ``kl_in_reward_coef``,
        ``kl_in_reward_target_kl``, ``kl_in_reward_horizon``.

    Returns
    -------
    AdaptiveKLController or FixedKLController
    """
    ctrl_type = ppo_config.kl_in_reward_ctrl_type
    if ctrl_type == "fixed":
        return FixedKLController(kl_coef=ppo_config.kl_in_reward_coef)
    if ctrl_type == "adaptive":
        horizon = int(ppo_config.kl_in_reward_horizon)
        assert horizon > 0, f"kl_in_reward_horizon must be > 0, got {horizon}"
        return AdaptiveKLController(
            init_kl_coef=ppo_config.kl_in_reward_coef,
            target_kl=float(ppo_config.kl_in_reward_target_kl),
            horizon=horizon,
        )
    raise NotImplementedError(f"Unknown kl_in_reward_ctrl_type={ctrl_type}")


def apply_kl_in_reward(
    rewards: Sequence[torch.Tensor],
    old_logprobs: Sequence[torch.Tensor],
    ref_logprobs: Sequence[torch.Tensor],
    mask: Sequence[torch.Tensor],
    kl_ctrl: AdaptiveKLController | FixedKLController,
) -> KlInRewardResult:
    """Apply verl-style KL-in-reward to scalar rewards.

    For each sample: ``reward -= kl_coef * sum_t (old_logp - ref_logp)`` (k1).
    Also returns per-sequence mean token KL for adaptive controller updates.

    Parameters
    ----------
    rewards : sequence of torch.Tensor
        Scalar rewards.
    old_logprobs, ref_logprobs, mask : sequence of torch.Tensor
        Response-token aligned tensors.
    kl_ctrl
        Controller providing the current KL coefficient via ``.value``.

    Returns
    -------
    KlInRewardResult
        Adjusted float32 rewards, pre-KL rewards, per-sequence mean KLs,
        per-sequence KL sums, and local metrics.
    """
    kl_coef = kl_ctrl.value
    rewards_before = [r.detach().float().clone() for r in rewards]
    adjusted_rewards = []
    seq_mean_kls = []
    kl_sums = []

    for reward, old_lp, ref_lp, m in zip(rewards, old_logprobs, ref_logprobs, mask, strict=True):
        assert old_lp.shape == ref_lp.shape == m.shape, (
            f"KL-in-reward length mismatch: "
            f"{tuple(old_lp.shape)} vs {tuple(ref_lp.shape)} vs {tuple(m.shape)}"
        )
        kld = (old_lp.float() - ref_lp.float()) * m.float()
        tok = m.float().sum().clamp_min(1.0)
        # verl apply_kl_penalty: masked_mean over tokens, then mean over batch.
        seq_mean_kls.append((kld.sum() / tok).detach())
        kl_sum = kld.sum().detach()
        kl_sums.append(kl_sum)
        adjusted = reward.float() - kl_coef * kl_sum
        # Keep float32 so small KL penalties are not rounded away in bf16.
        adjusted_rewards.append(adjusted.view_as(reward).to(dtype=torch.float32))

    metrics: Dict[str, float] = {"ppo-metrics/kl_in_reward_coef": kl_coef}
    if kl_sums:
        metrics["ppo-metrics/kl_in_reward"] = torch.stack(kl_sums).mean().item() * kl_coef
    return KlInRewardResult(
        rewards=adjusted_rewards,
        rewards_before=rewards_before,
        seq_mean_kls=seq_mean_kls,
        kl_sums=kl_sums,
        metrics=metrics,
    )


def update_kl_in_reward_controller(
    kl_ctrl: AdaptiveKLController | FixedKLController,
    seq_mean_kls: Sequence[torch.Tensor],
    dp_group,
) -> Dict[str, float]:
    """All-reduce mean per-token KL and update adaptive controller once.

    Matches verl ``apply_kl_penalty``: one update per generate step with
    ``n_steps`` = global sequence count. Reports the coefficient that was
    used this step; ``kl_in_reward_coef_new`` is the value after the update.

    Parameters
    ----------
    kl_ctrl
        KL controller to update.
    seq_mean_kls : sequence of torch.Tensor
        Per-sequence mean token KLs collected this step (local rank).
    dp_group
        Data-parallel process group for all-reduce.

    Returns
    -------
    dict
        Metrics with global mean KL and coefficients.
    """
    coef_used = kl_ctrl.value
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seq_mean_kls:
        local_sum = torch.stack(list(seq_mean_kls)).sum().to(device=device)
        local_n = torch.tensor(float(len(seq_mean_kls)), device=device, dtype=torch.float32)
    else:
        local_sum = torch.zeros((), device=device, dtype=torch.float32)
        local_n = torch.zeros((), device=device, dtype=torch.float32)
    dist.all_reduce(local_sum, group=dp_group)
    dist.all_reduce(local_n, group=dp_group)
    metrics: Dict[str, float] = {}
    if local_n.item() > 0:
        current_kl = (local_sum / local_n).item()
        n_steps = int(local_n.item())
        kl_ctrl.update(current_kl=current_kl, n_steps=n_steps)
        metrics["ppo-metrics/kl_in_reward_mean"] = current_kl
        metrics["ppo-metrics/kl_in_reward_coef"] = coef_used
        metrics["ppo-metrics/kl_in_reward_coef_new"] = kl_ctrl.value
    return metrics
