import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gpatch_v4.configs.config import RlConfig, _assert_fsdp2_dynamic_cp_support
from gpatch_v4.configs.debug_config import DebugConfig
from gpatch_v4.configs.policy_config import PolicyConfig
from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.configs.training_config import RLTrainingConfig
from gpatch_v4.training_backend.fsdp2_backend.fsdp2_engine_lm import Fsdp2EngineLm
from gpatch_v4.training_backend.fsdp2_backend.mixin import (
    ForwardStepMixin,
    response_pad_fsdp2_thd_loss_tensors,
)
from gpatch_v4.training_backend.loss import PolicyLossInput as PolicyLossInputV2
from gpatch_v4.training_backend.loss_factory import PolicyLossInput
from gpatch_v4.utils.cuda_peak_tracker import (
    finish_policy_cuda_memory_profile,
    start_policy_cuda_memory_profile,
)


_REDUCE_FACTORY = (
    "gpatch_v4.training_backend.loss_factory.reduce_metrics_across_data_parallel_group"
)
_REDUCE_NEW = (
    "gpatch_v4.training_backend.loss.ppo_loss.reduce_metrics_across_data_parallel_group"
)


def _config(**ppo_kw):
    defaults = dict(
        loss_func="grpo",
        ppo_entropy_bonus=0.0,
        grpo_kl_loss_beta=0.0,
        enable_off_policy_correction=False,
        ppo_ratio_eps=0.2,
    )
    defaults.update(ppo_kw)
    return SimpleNamespace(ppo=PpoConfig(**defaults), debug=DebugConfig())


def _loss_input(cls, curr, **kwargs):
    b, s = curr.shape
    fields = dict(
        advantages=torch.ones(b, s),
        prev_log_probs=torch.zeros(b, s),
        ref_log_probs=None,
        curr_log_probs=curr,
        response_mask=torch.ones(b, s),
        scaled_entropy=torch.tensor(0.0),
        rollout_log_probs=torch.zeros(b, s),
        per_token_entropy=torch.zeros_like(curr),
        cu_seqlens_padded=None,
        local_cp_size=1,
    )
    fields.update(kwargs)
    return cls(**fields)


def _resolve(loss_name: str, use_legacy: bool):
    return Fsdp2EngineLm._resolve_fsdp2_policy_loss(
        SimpleNamespace(
            training_config=SimpleNamespace(use_legacy_loss=use_legacy),
            ppo_config=SimpleNamespace(loss_func=loss_name),
        )
    )


def _call(loss_name: str, use_legacy: bool, config, loss_input):
    loss_fn, _ = _resolve(loss_name, use_legacy)
    with patch(_REDUCE_FACTORY), patch(_REDUCE_NEW):
        return Fsdp2EngineLm._call_fsdp2_policy_loss(
            loss_fn, config, loss_input, use_legacy
        )


def test_rl_train_actor_uses_new_loss_helpers():
    src = inspect.getsource(Fsdp2EngineLm.rl_train_actor)
    assert "_fsdp2_policy_loss_input_cls" in src
    assert "_call_fsdp2_policy_loss" in src
    assert "_scale_fsdp2_rl_bwd_loss" in src


def test_policy_cuda_memory_profile_reports_phase_local_global_peaks():
    assert DebugConfig().profile_policy_cuda_memory is False
    assert start_policy_cuda_memory_profile(False) is None
    assert finish_policy_cuda_memory_profile(None) == {}

    module = "gpatch_v4.utils.cuda_peak_tracker"
    gib = 1024**3
    with (
        patch(f"{module}.torch.cuda.synchronize") as synchronize,
        patch(f"{module}.torch.cuda.memory_allocated", return_value=10 * gib),
        patch(f"{module}.torch.cuda.memory_reserved", return_value=12 * gib),
        patch(f"{module}.torch.cuda.max_memory_allocated", return_value=14 * gib),
        patch(f"{module}.torch.cuda.max_memory_reserved", return_value=15 * gib),
        patch(f"{module}.torch.cuda.current_device", return_value="cpu"),
        patch(f"{module}.torch.cuda.reset_peak_memory_stats") as reset_peak,
        patch(f"{module}.torch.distributed.all_reduce") as all_reduce,
    ):
        start = start_policy_cuda_memory_profile(True)
        metrics = finish_policy_cuda_memory_profile(start)

    assert synchronize.call_count == 2
    reset_peak.assert_called_once_with()
    all_reduce.assert_called_once()
    assert metrics == {
        "policy/memory_start_allocated_gib": 10.0,
        "policy/memory_start_reserved_gib": 12.0,
        "policy/memory_peak_allocated_gib": 14.0,
        "policy/memory_peak_reserved_gib": 15.0,
        "policy/memory_peak_extra_allocated_gib": 4.0,
        "policy/memory_peak_extra_reserved_gib": 3.0,
    }


def test_input_cls_and_unpack_arity():
    assert Fsdp2EngineLm._fsdp2_policy_loss_input_cls(True) is PolicyLossInput
    assert Fsdp2EngineLm._fsdp2_policy_loss_input_cls(False) is PolicyLossInputV2

    curr = torch.zeros(2, 3, requires_grad=True)
    legacy_loss, legacy_count, legacy_metrics = _call(
        "grpo", True, _config(), _loss_input(PolicyLossInput, curr)
    )
    assert legacy_count is None
    assert torch.isfinite(legacy_loss)
    assert "ppo_ratio" in legacy_metrics or "loss" in legacy_metrics

    new_loss, new_count, new_metrics = _call(
        "grpo", False, _config(), _loss_input(PolicyLossInputV2, curr)
    )
    assert new_count is not None
    assert new_count.ndim == 0
    assert torch.isfinite(new_loss)
    assert torch.isfinite(new_count)
    assert "loss" in new_metrics


def test_new_loss_can_defer_metric_reduction_for_dynamic_cp():
    loss_fn, _ = _resolve("grpo", False)
    curr = torch.zeros(2, 3, requires_grad=True)

    with patch(_REDUCE_NEW) as reduce_metrics:
        Fsdp2EngineLm._call_fsdp2_policy_loss(
            loss_fn,
            _config(),
            _loss_input(
                PolicyLossInputV2,
                curr,
                defer_scalar_metric_reduction=True,
            ),
            False,
        )
    reduce_metrics.assert_not_called()

    with patch(_REDUCE_NEW) as reduce_metrics:
        Fsdp2EngineLm._call_fsdp2_policy_loss(
            loss_fn,
            _config(),
            _loss_input(PolicyLossInputV2, curr),
            False,
        )
    reduce_metrics.assert_called_once()


def test_three_backward_scale_formulas():
    dp_size, num_microbatches = 2, 4
    global_token_count = torch.tensor(12.0)

    # 新 loss + per-token：L * dp / G
    leaf = torch.tensor(6.0, requires_grad=True)
    scaled = Fsdp2EngineLm._scale_fsdp2_rl_bwd_loss(
        leaf,
        sum_return_loss=True,
        dp_size=dp_size,
        num_microbatches=num_microbatches,
        global_loss_count=global_token_count,
    )
    scaled.backward()
    torch.testing.assert_close(leaf.grad, torch.tensor(dp_size / 12.0))

    # 新 loss + per-seq：L * dp / effective_gbs
    global_sequence_count = torch.tensor(5.0)
    leaf = torch.tensor(6.0, requires_grad=True)
    scaled = Fsdp2EngineLm._scale_fsdp2_rl_bwd_loss(
        leaf,
        sum_return_loss=True,
        dp_size=dp_size,
        num_microbatches=num_microbatches,
        global_loss_count=global_sequence_count,
    )
    scaled.backward()
    torch.testing.assert_close(leaf.grad, torch.tensor(dp_size / 5.0))

    # legacy mean：L / num_microbatches
    leaf = torch.tensor(6.0, requires_grad=True)
    scaled = Fsdp2EngineLm._scale_fsdp2_rl_bwd_loss(
        leaf,
        sum_return_loss=False,
        dp_size=dp_size,
        num_microbatches=num_microbatches,
    )
    scaled.backward()
    torch.testing.assert_close(leaf.grad, torch.tensor(1.0 / num_microbatches))


def test_local_rl_loss_count_uses_sample_mask_for_sequence_mean():
    batch = [
        {
            "mask": torch.tensor([1.0, 1.0]),
            "sample_mask": torch.tensor(True),
        },
        {
            "mask": torch.zeros(3),
            "sample_mask": torch.tensor(False),
        },
    ]

    assert Fsdp2EngineLm._local_rl_loss_count(batch, True) == 2.0
    assert Fsdp2EngineLm._local_rl_loss_count(batch, False) == 1.0


def test_local_rl_loss_count_rejects_unmasked_invalid_sample():
    batch = [
        {
            "mask": torch.tensor([1.0]),
            "sample_mask": torch.tensor(False),
        },
    ]

    with pytest.raises(AssertionError, match="all-zero response mask"):
        Fsdp2EngineLm._local_rl_loss_count(batch, False)


def test_fsdp2_dynamic_cp_accepts_sequence_mean_rl():
    training = SimpleNamespace(
        training_backend="fsdp2",
        use_dynamic_mbs=False,
        enable_mtp=False,
        online_mtp_sft=False,
        enable_dspark=False,
        ppo_dump_metrics_interval=-1,
        moe_balance_loss_coef=0,
    )
    policy = SimpleNamespace(
        model_arch="qwen4_exp",
        attn_implementation="flex",
        indexer_backend="eager",
        ep_backend="eager",
        override_transformer_config={"calculate_per_token_loss": False},
        dist_config=SimpleNamespace(
            dynamic_context_parallel=True,
            dynamic_cp_scheduler_type="default",
            context_parallel_size=1,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        ),
    )

    _assert_fsdp2_dynamic_cp_support(training, policy, rl_loss_func="gspo")


def test_three_backward_scales_on_real_loss():
    # 三条 engine 缩放接到真实 GRPO/GSPO 上，梯度有限且非零。
    cases = (
        ("grpo", False, True, True, torch.tensor(6.0)),
        ("grpo", False, False, True, torch.tensor(2.0)),
        ("gspo", True, False, False, None),
    )
    for loss_name, use_legacy, per_token, sum_return, global_count in cases:
        curr = torch.zeros(2, 3, requires_grad=True)
        cls = Fsdp2EngineLm._fsdp2_policy_loss_input_cls(use_legacy)
        bwd_loss, _, _ = _call(
            loss_name,
            use_legacy,
            _config(loss_func=loss_name),
            _loss_input(cls, curr, calculate_per_token_loss=per_token),
        )
        scaled = Fsdp2EngineLm._scale_fsdp2_rl_bwd_loss(
            bwd_loss,
            sum_return_loss=sum_return,
            dp_size=2,
            num_microbatches=2,
            global_loss_count=global_count,
        )
        scaled.backward()
        assert torch.isfinite(scaled)
        assert curr.grad is not None
        assert torch.isfinite(curr.grad).all()
        assert curr.grad.abs().sum() > 0


def test_per_token_dp_avg_matches_global_sum():
    # FSDP AVG：mean_i(L_i * DP / G) == (L0+L1)/G
    dp_size = 2
    global_token_count = torch.tensor(6.0)
    rank_sums = [torch.tensor(2.0), torch.tensor(4.0)]
    scaled = [
        Fsdp2EngineLm._scale_fsdp2_rl_bwd_loss(
            loss,
            sum_return_loss=True,
            dp_size=dp_size,
            num_microbatches=2,
            global_loss_count=global_token_count,
        ) for loss in rank_sums
    ]
    avg = (scaled[0] + scaled[1]) / dp_size
    torch.testing.assert_close(avg, (rank_sums[0] + rank_sums[1]) / global_token_count)


def test_mask_correction_bwd_count_differs_from_engine_token_count():
    # engine 用原始 mask 做 global_token_count；新 loss 的 _bwd_count 用 correction 后 mask。
    mask = torch.ones(2, 4)
    prev = torch.zeros(2, 4)
    rollout = torch.zeros(2, 4)
    rollout[0, 0] = -4.0
    rollout[0, 1] = 4.0
    curr = torch.zeros(2, 4, requires_grad=True)
    config = _config(
        enable_off_policy_correction=True,
        off_policy_correction_mode="mask",
        off_policy_correction_upper_bound=2.0,
        off_policy_correction_lower_bound=0.5,
    )
    bwd_loss, bwd_count, _ = _call(
        "grpo",
        False,
        config,
        _loss_input(
            PolicyLossInputV2,
            curr,
            prev_log_probs=prev,
            rollout_log_probs=rollout,
            response_mask=mask,
            calculate_per_token_loss=True,
        ),
    )
    engine_token_count = float(mask.sum().item())
    assert bwd_count.item() == 6.0
    assert engine_token_count == 8.0
    assert bwd_count.item() != engine_token_count

    scaled_engine = Fsdp2EngineLm._scale_fsdp2_rl_bwd_loss(
        bwd_loss,
        sum_return_loss=True,
        dp_size=1,
        num_microbatches=1,
        global_loss_count=torch.tensor(engine_token_count),
    )
    scaled_if_used_count = Fsdp2EngineLm._scale_fsdp2_rl_bwd_loss(
        bwd_loss.detach(),
        sum_return_loss=True,
        dp_size=1,
        num_microbatches=1,
        global_loss_count=bwd_count.detach(),
    )
    assert not torch.allclose(scaled_engine.detach(), scaled_if_used_count)


def test_thd_restore_then_new_gspo_has_finite_update():
    cu_padded = torch.tensor([0, 4, 8], dtype=torch.int32)
    response_starts = torch.tensor([1, 0], dtype=torch.int32)
    response_lengths = torch.tensor([2, 2], dtype=torch.int32)
    packed_curr = torch.tensor(
        [[-1.0, -2.0, -3.0, 0.0, -4.0, -5.0, 0.0, 0.0]],
        requires_grad=True,
    )
    padded = response_pad_fsdp2_thd_loss_tensors(
        {
            "curr_log_probs": packed_curr,
            "per_token_entropy": torch.zeros_like(packed_curr),
            "mask": torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]]),
            "advantages": torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.4, 0.5, 0.0, 0.0]]),
            "prev_log_probs": packed_curr.detach() - 0.05,
            "ref_log_probs": packed_curr.detach() - 0.1,
            "rollout_log_probs": packed_curr.detach() - 0.15,
        },
        cu_padded,
        response_starts,
        response_lengths,
    )
    assert padded["curr_log_probs"].shape[0] == 2
    assert padded["curr_log_probs"].ndim == 2

    bwd_loss, bwd_count, _ = _call(
        "gspo",
        False,
        _config(loss_func="gspo"),
        PolicyLossInputV2(
            advantages=padded["advantages"],
            prev_log_probs=padded["prev_log_probs"],
            ref_log_probs=padded["ref_log_probs"],
            curr_log_probs=padded["curr_log_probs"],
            response_mask=padded["mask"],
            scaled_entropy=torch.tensor(0.0),
            rollout_log_probs=padded["rollout_log_probs"],
            per_token_entropy=padded["per_token_entropy"],
            cu_seqlens_padded=None,
            local_cp_size=1,
        ),
    )
    assert bwd_count is not None
    scaled = Fsdp2EngineLm._scale_fsdp2_rl_bwd_loss(
        bwd_loss,
        sum_return_loss=True,
        dp_size=1,
        num_microbatches=1,
        global_loss_count=bwd_count,
    )
    scaled.backward()
    assert torch.isfinite(scaled)
    assert packed_curr.grad is not None
    assert torch.isfinite(packed_curr.grad).all()
    assert packed_curr.grad.abs().sum() > 0


def test_one_step_new_grpo_param_update():
    curr = torch.zeros(2, 3, requires_grad=True)
    bwd_loss, bwd_count, _ = _call(
        "grpo", False, _config(), _loss_input(PolicyLossInputV2, curr)
    )
    scaled = Fsdp2EngineLm._scale_fsdp2_rl_bwd_loss(
        bwd_loss,
        sum_return_loss=True,
        dp_size=1,
        num_microbatches=1,
        global_loss_count=bwd_count,
    )
    scaled.backward()
    assert torch.isfinite(scaled)
    assert torch.isfinite(bwd_count)
    assert torch.isfinite(curr.grad).all()
    assert curr.grad.abs().sum() > 0
    before = curr.detach().clone()
    curr.data -= 0.01 * curr.grad
    assert torch.isfinite(curr).all()
    assert not torch.equal(curr.detach(), before)


def test_fsdp2_grpo_linear_ce_accepts_qwen_thd_packing():
    config = RlConfig(
        training=RLTrainingConfig(
            training_backend="fsdp2",
            use_linear_ce=True,
        ),
        policy=PolicyConfig(
            model_arch="qwen4_exp",
            ppo_pack_seq=True,
        ),
        ppo=PpoConfig(loss_func="grpo"),
    )

    assert config.training.use_linear_ce
    assert config.policy.ppo_pack_seq


def test_fsdp2_grpo_linear_ce_rejects_processed_logprobs():
    with pytest.raises(AssertionError, match="use_original_logprob=True"):
        RlConfig(
            training=RLTrainingConfig(
                training_backend="fsdp2",
                use_linear_ce=True,
            ),
            policy=PolicyConfig(model_arch="qwen4_exp"),
            ppo=PpoConfig(
                loss_func="grpo",
                use_original_logprob=False,
            ),
        )


def test_fsdp2_linear_ce_shifts_labels_before_cp_partition():
    seen = []

    def _select_local_chunk(labels):
        seen.append(labels.clone())
        return labels[:, 2:5]

    mixin = ForwardStepMixin.__new__(ForwardStepMixin)
    mixin.prepare_data = SimpleNamespace(
        rl_train_cp_chunk_single_data=_select_local_chunk,
    )
    targets = torch.tensor([[10, 11, 12, 13, 14, 15]])

    local_labels = mixin._prepare_rl_linear_ce_labels(targets, pre_shifted=False)

    expected_shifted = torch.tensor([[11, 12, 13, 14, 15, 10]])
    torch.testing.assert_close(seen[0], expected_shifted)
    torch.testing.assert_close(local_labels, expected_shifted[:, 2:5])


def test_fsdp2_linear_ce_preserves_pre_shifted_thd_labels():
    mixin = ForwardStepMixin.__new__(ForwardStepMixin)
    mixin.prepare_data = SimpleNamespace(
        rl_train_cp_chunk_single_data=lambda labels: labels[:, 1:4],
    )
    shifted_targets = torch.tensor([[11, 12, 13, 14, 15, -100]])

    local_labels = mixin._prepare_rl_linear_ce_labels(
        shifted_targets,
        pre_shifted=True,
    )

    torch.testing.assert_close(local_labels, shifted_targets[:, 1:4])
