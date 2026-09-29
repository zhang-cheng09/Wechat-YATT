"""新路径 opd_loss：三元组返回、policy-ref 固定 low-var KL、top-K 3D ratio。"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gpatch_v4.configs.ppo_config import DistillConfig
from gpatch_v4.core.adaptive_entropy import ENTROPY_COEF_KEY, LAST_WORLD_ENTROPY_KEY
from gpatch_v4.core.ppo_feature_store import (
    get_ppo_feature_store,
    reset_ppo_feature_store_for_test,
)
from gpatch_v4.training_backend.loss.ppo_loss import PolicyLossInput, opd_loss
from gpatch_v4.training_backend.loss_factory import (
    PolicyLossInput as FactoryPolicyLossInput,
    opd_loss_func as factory_opd_loss,
)
from gpatch_v4.training_backend.loss.utils import agg
from gpatch_v4.utils.ppo_utils import calculate_kl_loss

_REDUCE = "gpatch_v4.training_backend.loss.ppo_loss.reduce_metrics_across_data_parallel_group"


def _cfg(**ppo_kw):
    d = dict(
        loss_func="opd",
        ppo_logps_ratio_clamp=None,
        ppo_ratio_eps=0.2,
        ppo_clip_ratio_low=None,
        ppo_clip_ratio_high=None,
        ppo_dual_clip_ratio_c=None,
        ppo_clamp_kl_val=None,
        ppo_entropy_bonus=0.0,
        grpo_kl_loss_beta=1.0,
        opd_teacher_kl_loss_beta=0.0,
        enable_off_policy_correction=False,
        ppo_entropy_regularization_type=None,
        skip_prev_logps=False,
    )
    d.update(ppo_kw)
    return SimpleNamespace(ppo=DistillConfig(**d))


def _loss_input(advantages, curr, prev, ref, teacher, mask, **extra):
    return PolicyLossInput(
        advantages=advantages,
        prev_log_probs=prev,
        ref_log_probs=ref,
        curr_log_probs=curr,
        response_mask=mask,
        scaled_entropy=torch.tensor(0.0),
        teacher_log_probs=teacher,
        per_token_entropy=torch.zeros_like(curr),
        **extra,
    )


@patch(_REDUCE)
def test_opd_new_path_returns_triple_and_uses_low_var_kl(_mock_reduce):
    curr = torch.tensor([[-1.0, -2.0, -0.5, -1.5], [-0.8, -1.2, -2.0, -0.3]])
    ref = torch.tensor([[-2.0, -0.5, -1.5, -0.2], [-1.8, -0.4, -0.9, -2.2]])
    mask = torch.ones_like(curr)
    loss_input = _loss_input(
        advantages=torch.zeros_like(curr),
        curr=curr,
        prev=curr.detach(),
        ref=ref,
        teacher=torch.zeros_like(curr),
        mask=mask,
    )

    result = opd_loss(_cfg(), loss_input)
    assert len(result) == 3
    bwd_loss, bwd_count, metrics = result
    assert bwd_loss.dim() == 0
    assert torch.isfinite(bwd_loss)
    assert bwd_count.item() == curr.shape[0]
    assert "policy_ref_kl_loss" in metrics
    assert "teacher_student_kl_loss" in metrics
    for key in (
        "policy_loss_token",
        "policy_loss_sample",
        "entropy_token",
        "entropy_sample",
        "policy_ref_kl_token",
        "policy_ref_kl_sample",
        "teacher_student_kl_token",
        "teacher_student_kl_sample",
    ):
        assert key in metrics
        assert metrics[key].shape == (2, )

    low_var = calculate_kl_loss(
        curr, ref, use_absolute_kl=False, use_low_var_kl=True, clamp_kl_loss=False
    )
    abs_kl = calculate_kl_loss(
        curr, ref, use_absolute_kl=True, use_low_var_kl=False, clamp_kl_loss=False
    )
    expected, _ = agg(low_var, mask)
    other, _ = agg(abs_kl, mask)
    assert torch.allclose(bwd_loss, expected)
    assert not torch.allclose(bwd_loss, other)


@patch(_REDUCE)
def test_opd_new_path_ref_none_keeps_dual_metric_counts(_mock_reduce):
    curr = torch.tensor([[-1.0, -2.0], [-0.8, -1.2]])
    mask = torch.ones_like(curr)
    loss_input = _loss_input(
        advantages=torch.zeros_like(curr),
        curr=curr,
        prev=curr.detach(),
        ref=None,
        teacher=torch.zeros_like(curr),
        mask=mask,
    )
    _, _, metrics = opd_loss(_cfg(grpo_kl_loss_beta=0.0), loss_input)
    assert torch.allclose(metrics["policy_ref_kl_token"][0], torch.zeros(()))
    assert torch.allclose(
        metrics["policy_ref_kl_token"][1], metrics["policy_loss_token"][1]
    )
    assert torch.allclose(metrics["policy_ref_kl_sample"][0], torch.zeros(()))
    assert torch.allclose(
        metrics["policy_ref_kl_sample"][1], metrics["policy_loss_sample"][1]
    )


@patch(_REDUCE)
def test_opd_new_path_topk_returns_scalar_and_backward(_mock_reduce):
    B, S, K = 2, 6, 8
    advantages = torch.randn(B, S, K)
    mask = torch.ones(B, S)
    mask[:, :2] = 0
    prev_topk = torch.randn(B, S, K) - 2.0
    curr_topk = (prev_topk + torch.randn(B, S, K) * 0.01).requires_grad_(True)
    loss_input = _loss_input(
        advantages=advantages,
        curr=torch.randn(B, S),
        prev=torch.randn(B, S),
        ref=torch.randn(B, S),
        teacher=torch.randn(B, S),
        mask=mask,
        prev_topk_logprobs=prev_topk,
        curr_topk_logprobs=curr_topk,
    )

    bwd_loss, bwd_count, metrics = opd_loss(
        _cfg(grpo_kl_loss_beta=0.0, opd_teacher_kl_loss_beta=0.0),
        loss_input,
    )
    assert bwd_loss.dim() == 0 and torch.isfinite(bwd_loss)
    assert bwd_count.item() == B
    for key in (
        "policy_loss_token",
        "policy_loss_sample",
        "entropy_token",
        "entropy_sample",
        "policy_ref_kl_token",
        "policy_ref_kl_sample",
        "teacher_student_kl_token",
        "teacher_student_kl_sample",
    ):
        assert key in metrics
    bwd_loss.backward()
    assert curr_topk.grad is not None
    assert torch.isfinite(curr_topk.grad).all()
    ratio = metrics["ppo_ratio"][0] / metrics["ppo_ratio"][1]
    assert 0.9 < ratio < 1.1


@pytest.mark.parametrize("topk", (False, True))
@pytest.mark.parametrize("calculate_per_token_loss", (False, True))
@pytest.mark.parametrize("last_entropy, expected_coef", ((None, 0.0), (2.0, 0.0), (0.5, 0.07)))
def test_opd_adaptive_entropy_matches_factory(
    topk: bool,
    calculate_per_token_loss: bool,
    last_entropy: float | None,
    expected_coef: float,
) -> None:
    # OPD 独立分支也要读取上一步的熵系数及门控。
    reset_ppo_feature_store_for_test()
    try:
        if last_entropy is not None:
            store = get_ppo_feature_store()
            store.set(LAST_WORLD_ENTROPY_KEY, last_entropy)
            store.set(ENTROPY_COEF_KEY, 0.07)
        config = _cfg(
            use_adaptive_entropy=True,
            feature_store_enable=True,
            ppo_entropy_bonus=0.2,
            entropy_target=1.0,
            grpo_kl_loss_beta=0.0,
            log_prob_top_k=2 if topk else 0,
        )
        curr = torch.full((2, 3), -1.0)
        mask = torch.tensor([[1., 0., 1.], [0., 1., 1.]])
        entropy = torch.tensor([[1., 5., 2.], [4., 3., 1.]], requires_grad=True)
        kwargs = dict(
            advantages=torch.zeros(2, 3, 2) if topk else torch.zeros_like(curr),
            prev_log_probs=curr,
            ref_log_probs=None,
            curr_log_probs=curr,
            response_mask=mask,
            scaled_entropy=torch.tensor(0.0),
            per_token_entropy=entropy,
            teacher_log_probs=curr,
            calculate_per_token_loss=calculate_per_token_loss,
        )
        if topk:
            kwargs["prev_topk_logprobs"] = curr.unsqueeze(-1).expand(2, 3, 2)
            kwargs["curr_topk_logprobs"] = kwargs["prev_topk_logprobs"]
        with (
            patch(_REDUCE),
            patch(
                "gpatch_v4.training_backend.loss_factory.reduce_metrics_across_data_parallel_group"
            ),
        ):
            new_loss, count, new_metrics = opd_loss(config, PolicyLossInput(**kwargs))
            old_loss, old_metrics = factory_opd_loss(config, FactoryPolicyLossInput(**kwargs))
        expected_sum = (entropy * mask).sum() if calculate_per_token_loss else (
            (entropy * mask).sum(dim=-1) / mask.sum(dim=-1)
        ).sum()
        torch.testing.assert_close(new_loss, -expected_coef * expected_sum)
        torch.testing.assert_close(new_loss, old_loss)
        torch.testing.assert_close(count, old_metrics["loss"][1])
        for name, value in old_metrics.items():
            torch.testing.assert_close(new_metrics[name], value)
        new_grad = torch.autograd.grad(new_loss, entropy, retain_graph=True)[0]
        old_grad = torch.autograd.grad(old_loss, entropy)[0]
        torch.testing.assert_close(new_grad, old_grad)
    finally:
        reset_ppo_feature_store_for_test()


@pytest.mark.parametrize("enable_is", (False, True))
@pytest.mark.parametrize("calculate_per_token_loss", (False, True))
@pytest.mark.parametrize("topk", (False, True))
def test_opd_rejection_precedes_entropy_regularization(
    enable_is: bool, calculate_per_token_loss: bool, topk: bool
) -> None:
    # 被 RS 拒绝的 token 不得参与协方差选点和 clip 指标。
    config = _cfg(
        enable_off_policy_correction=enable_is,
        off_policy_correction_rs="token_k1",
        off_policy_correction_rs_threshold="0.5_5.0",
        ppo_entropy_regularization_type=None if topk else "kl-cov",
        log_prob_top_k=2 if topk else 0,
        ppo_kl_cov_ratio=0.5,
        ppo_kl_cov_coef=0.7,
        grpo_kl_loss_beta=0.0,
    )
    prev = torch.tensor([[-1., -2., -3., -4.], [-2., -3., -1., -4.]])
    rollout = prev.clone()
    rollout[0, 1] -= torch.log(torch.tensor(3.0))
    rollout[1, 2] += torch.log(torch.tensor(10.0))
    curr = (prev + torch.tensor([[0.1, -0.3, 0.3, -0.1], [0.2, -0.2, 0.4, -0.3]])).requires_grad_()
    kwargs = dict(
        advantages=torch.tensor([[0.5, 20., 2., -1.], [3., -2., 40., 1.]]),
        prev_log_probs=prev,
        ref_log_probs=None,
        curr_log_probs=curr,
        response_mask=torch.ones_like(curr),
        scaled_entropy=torch.tensor(0.0),
        per_token_entropy=torch.zeros_like(curr),
        rollout_log_probs=rollout,
        teacher_log_probs=prev,
        calculate_per_token_loss=calculate_per_token_loss,
    )
    if topk:
        kwargs["advantages"] = kwargs["advantages"].unsqueeze(-1) * torch.tensor([1., -0.5])
        kwargs["prev_topk_logprobs"] = prev.unsqueeze(-1).expand(2, 4, 2)
        kwargs["curr_topk_logprobs"] = curr.unsqueeze(-1) + torch.tensor([0., 0.05])
    with (
        patch(_REDUCE),
        patch("gpatch_v4.training_backend.loss_factory.reduce_metrics_across_data_parallel_group"),
    ):
        new_loss, _, new_metrics = opd_loss(config, PolicyLossInput(**kwargs))
        old_loss, old_metrics = factory_opd_loss(config, FactoryPolicyLossInput(**kwargs))
    torch.testing.assert_close(new_loss, old_loss)
    for name, value in old_metrics.items():
        torch.testing.assert_close(new_metrics[name], value)
    new_grad = torch.autograd.grad(new_loss, curr, retain_graph=True)[0]
    old_grad = torch.autograd.grad(old_loss, curr)[0]
    torch.testing.assert_close(new_grad, old_grad)
    assert new_grad[0, 1] == 0
    assert new_grad[1, 2] == 0
