"""dynamic_batch_train may have rollout_gbs < train dp_size."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gpatch_v4.utils.dynamic_batch_utils import assign_train_steps, split_train_steps_by_dp


def _make_actor(
    *,
    dynamic_batch_train: bool,
    dynamic_cp: bool = False,
    train_gbs: int = 256,
    train_mbs: int = 1,
):
    from gpatch_v4.actor.grpo_train_actor import GrpoTrainActor

    actor = GrpoTrainActor.__new__(GrpoTrainActor)
    actor.config = SimpleNamespace(
        policy=SimpleNamespace(
            rollout_gen_type="default",
            dist_config=SimpleNamespace(dynamic_context_parallel=dynamic_cp),
            without_ref=True,
        ),
        ppo=SimpleNamespace(
            advantage_type="grpo",
            loss_func="grpo",
            skip_prev_logps=False,
            post_compute_logprobs="none",
            custom_advantage_py_path=None,
            custom_advantage_py_name=None,
            custom_post_advantage_py_name=None,
            post_compute_logprobs_py_path=None,
            post_compute_logprobs_py_name=None,
        ),
        training=SimpleNamespace(
            dynamic_batch_train=dynamic_batch_train,
            single_controller=True,
            sampling_keeping_strategy="all",
            sampling_repeat_n=8,
            sampling_keep_n=8,
            rollout_gbs=32,
            rollout_mbs=1,
            train_gbs=train_gbs,
            train_mbs=train_mbs,
            num_train_epoches=1,
            eval_interval=0,
            auto_load_from_save_ckpt=False,
            ppo_filter_samplings_path=None,
            ppo_filter_samplings_name=None,
        ),
        data=SimpleNamespace(py_path="unused.py", fn_name="build_dataset"),
        placement_type="disaggregated",
        checkpoint=SimpleNamespace(
            convert_mcore_to_hf_offline=False,
            save_ckpt_path="/tmp",
        ),
    )
    actor.require_critic_model = MagicMock(return_value=False)
    actor._register_rl_policy_loss = MagicMock()
    actor._validate_skip_prev_logps = MagicMock()
    return actor


def _run_validated_config(actor):
    with patch("gpatch_v4.actor.grpo_train_actor.mpu") as mpu, patch(
        "gpatch_v4.actor.grpo_train_actor.is_partial_colocated",
        return_value=False,
    ):
        mpu.get_data_parallel_world_size.return_value = 64
        actor.validated_config()


def test_dynamic_batch_allows_rollout_gbs_lt_dp_size():
    _run_validated_config(_make_actor(dynamic_batch_train=True, dynamic_cp=True))


def test_fixed_batch_still_rejects_rollout_gbs_lt_dp_size():
    with pytest.raises(AssertionError):
        _run_validated_config(_make_actor(dynamic_batch_train=False))


def test_dynamic_batch_without_dyn_cp_requires_train_gbs_divisible():
    with pytest.raises(AssertionError, match="divisible by dp_size"):
        _run_validated_config(
            _make_actor(dynamic_batch_train=True, dynamic_cp=False, train_gbs=200)
        )


def test_dynamic_batch_requires_train_gbs_to_cover_train_mbs_on_every_dp_rank():
    with pytest.raises(AssertionError):
        _run_validated_config(
            _make_actor(
                dynamic_batch_train=True,
                dynamic_cp=True,
                train_gbs=64,
                train_mbs=2,
            )
        )


def test_single_controller_uses_global_sampler_topology_for_schedule():
    actor = _make_actor(dynamic_batch_train=True, dynamic_cp=True, train_gbs=128)
    actor.tokenizer = object()
    factory_args = {}

    def build_dataset(config, tokenizer, dp_rank, dp_size):
        factory_args.update(dp_rank=dp_rank, dp_size=dp_size)
        dataset = list(range(64))
        return {
            "train_dataset": dataset,
            "train_dataloader": dataset,
            "train_sampler": None,
            "eval_dataset": None,
            "eval_dataloader": None,
            "eval_sampler": None,
        }

    with patch(
        "gpatch_v4.actor.grpo_train_actor.import_fn_from_path",
        return_value=build_dataset,
    ), patch("gpatch_v4.actor.grpo_train_actor.mpu") as mpu:
        mpu.get_data_parallel_rank.return_value = 7
        mpu.get_data_parallel_world_size.return_value = 64
        actor.build_dataset_and_dataloader()

    assert factory_args == {"dp_rank": 0, "dp_size": 1}
    assert actor.config.training.ppo_step_per_epoch == 2
    assert actor.config.training.total_ppo_step == 2
    assert actor.config.training.gradient_accumulation_steps == 2


def test_expanded_samples_split_across_dp64_without_empty_rank():
    training_config = SimpleNamespace(
        rollout_gbs=32,
        sampling_keep_n=8,
        rb_multiplier=1,
        train_gbs=128,
    )
    samples = [
        {
            "prompt_idx": prompt_idx,
            "sequence_lengths": response_idx + 1,
        }
        for prompt_idx in range(training_config.rollout_gbs)
        for response_idx in range(training_config.sampling_keep_n)
    ]

    samples = assign_train_steps(samples, training_config)
    dp_steps = split_train_steps_by_dp(
        samples,
        dp_size=64,
        train_mbs=1,
        dynamic_context_parallel=True,
    )

    assert len(samples) == 256
    assert len(dp_steps) == 64
    assert all(len(rank_steps) == 2 for rank_steps in dp_steps)
    assert all(step_samples for rank_steps in dp_steps for step_samples in rank_steps)
    assert sum(
        len(step_samples)
        for rank_steps in dp_steps
        for step_samples in rank_steps
    ) == 256
