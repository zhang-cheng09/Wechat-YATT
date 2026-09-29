import asyncio
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from gpatch_v4.configs.debug_config import DebugConfig
from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.agentic.generate_func import SamplerGenerateFuncAgentic
from gpatch_v4.agentic.env_manager.traj_env_manager import TrajEnvManager
from gpatch_v4.rollout_generator.async_rollout.agent_loop_actor import (
    AgentLoopActor,
    AgentLoopEvalResult,
)
from gpatch_v4.rollout_generator.async_rollout.rollout_controller import RolloutController
from gpatch_v4.training_backend.loss import PolicyLossInput, get_loss_fn
from gpatch_v4.training_backend.loss.utils import agg
from tasks.retool.multi_segment.dup_traj_env_manager import DupTrajEnvManager
from tasks.retool.multi_segment.dynamic_batch_hooks import (
    compute_eval_metrics,
    compute_rollout_metrics,
    convert_samples_to_train_data,
)
from tasks.retool.multi_segment.external_reward import RetoolMultiSegmentReward
from tasks.retool.async_agent_loop.retool_env import RetoolEnv


def _config(calculate_per_token_loss: bool = False):
    return SimpleNamespace(
        training=SimpleNamespace(
            train_gbs=2,
            metrics_report=[],
        ),
        policy=SimpleNamespace(
            override_transformer_config={
                "calculate_per_token_loss": calculate_per_token_loss,
            },
        ),
    )


def _sample(
    traj_id: int,
    segment_id: int,
    reward: float,
    normalized_reward: float,
    response_tokens: int = 2,
):
    tokens = torch.arange(response_tokens + 1)
    return {
        "_train_step_id": 0,
        "group_id": 0,
        "traj_id": traj_id,
        "segment_id": segment_id,
        "tokens": tokens,
        "prompt_lengths": torch.tensor(1),
        "sequence_lengths": torch.tensor(tokens.numel()),
        "rewards": torch.tensor(reward),
        "normalized_rewards": torch.tensor(normalized_reward),
        "sample_mask": torch.tensor(True),
        "mask": torch.ones(response_tokens),
    }


def test_dup_traj_keeps_reward_only_on_last_segment():
    manager = DupTrajEnvManager.__new__(DupTrajEnvManager)
    manager.env_config = {"config": {"dup_num_segments": 3}}
    base = {
        "tokens": [torch.tensor([1, 2, 3])],
        "rewards": [torch.tensor(7.0)],
    }

    duplicated = manager._apply_dup(base)

    assert len(duplicated["tokens"]) == 3
    assert [reward.item() for reward in duplicated["rewards"]] == [0.0, 0.0, 7.0]
    assert all(
        duplicated["tokens"][idx].data_ptr() != duplicated["tokens"][0].data_ptr()
        for idx in (1, 2)
    )


def test_dup_traj_random_segment_count_is_seeded():
    manager = DupTrajEnvManager.__new__(DupTrajEnvManager)
    manager.env_config = {"config": {"dup_max_segments": 3}}
    manager.sampling_seed_offset = 17

    assert manager._resolve_dup_k() == random.Random(17).randint(1, 3)


def test_dynamic_multi_segment_hooks_use_trajectory_equal_weights_and_metrics():
    samples = [
        _sample(0, 0, 1.0, -1.0),
        _sample(1, 0, 0.0, 1.0),
        _sample(1, 1, 0.0, 1.0),
        _sample(1, 2, 3.0, 1.0),
    ]

    metrics = compute_rollout_metrics(_config(), samples)
    converted = convert_samples_to_train_data(_config(), samples)

    weights = [sample["token_weights"].item() for sample in converted]
    assert torch.allclose(
        torch.tensor(weights),
        torch.tensor([2.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]),
    )
    assert metrics["rollout-rewards/global_rewards"] == 2.0
    assert metrics["multi_segment/actual_segment_gbs_mean"] == 4.0
    assert metrics["multi_segment/segments_per_trajectory_mean"] == 2.0
    assert metrics["multi_segment/token_weight_mean"] == 1.0
    assert metrics["multi_segment/normalized_reward_mean"] == 0.0


def test_dynamic_multi_segment_hooks_normalize_over_valid_trajectories():
    invalid = _sample(1, 0, 0.0, 0.0)
    invalid["sample_mask"] = torch.tensor(False)
    invalid["mask"].zero_()
    samples = [
        _sample(0, 0, 1.0, 1.0),
        invalid,
    ]

    metrics = compute_rollout_metrics(_config(), samples)
    converted = convert_samples_to_train_data(_config(), samples)

    assert converted[0]["token_weights"].item() == 1.0
    assert converted[1]["token_weights"].item() == 0.0
    assert metrics["multi_segment/trajectory_gbs"] == 2.0
    assert metrics["multi_segment/actual_trajectory_gbs_mean"] == 1.0


def _eval_batch():
    return {
        "group_id": [0, 0, 0, 1],
        "traj_id": [0, 0, 0, 0],
        "segment_id": [0, 1, 2, 0],
        "rewards": [torch.tensor(0.0) for _ in range(4)],
        "mask": [torch.ones(2, dtype=torch.bool) for _ in range(3)] + [
            torch.zeros(2, dtype=torch.bool)
        ],
        "prompt_lengths": [torch.tensor(1) for _ in range(4)],
        "sequence_lengths": [torch.tensor(3) for _ in range(4)],
        "_external_reward_target": ["4", "4", "4", "4"],
        "_external_reward_valid": [True, True, True, False],
        "_external_reward_response": ["\\boxed{4}"] * 3 + [""],
        "_external_reward_num_turns": [2, 2, 2, 0],
        "_external_reward_success_eligible": [True, True, True, False],
    }


def test_retool_eval_defers_scoring_without_changing_train_reward():
    train_env = RetoolEnv()
    train_env._ground_truth = "4"
    train_env._all_actions = []
    train_env._msg_turn_count = 0
    train_env._defer_external_reward = False
    _, train_reward, _, _, train_info = train_env.step("\\boxed{4}")

    eval_env = RetoolEnv()
    eval_env._ground_truth = "4"
    eval_env._all_actions = []
    eval_env._msg_turn_count = 0
    eval_env._defer_external_reward = True
    with patch(
        "tasks.retool.async_agent_loop.retool_env.compute_retool_reward",
        side_effect=AssertionError("eval reward ran inside the environment"),
    ):
        _, eval_reward, _, _, eval_info = eval_env.step("\\boxed{4}")

    assert train_reward == 1.0
    assert train_info["metrics"]["retool_acc"] == 1.0
    assert eval_reward == 0.0
    assert eval_info["metrics"]["retool_acc"] == 0.0
    assert eval_info["_external_reward_response"] == "\\boxed{4}"
    assert eval_info["_external_reward_num_turns"] == 2


def test_multi_segment_eval_reward_and_metrics_use_one_vote_per_trajectory():
    batch = _eval_batch()
    update = RetoolMultiSegmentReward._compute_update(batch)
    batch.update(update)

    assert [value.item() for value in batch["rewards"]] == [0.0, 0.0, 1.0, 0.0]
    assert [value.item() for value in batch["retool_acc"]] == [1.0, 1.0, 1.0, 0.0]
    samples = [{name: values[idx] for name, values in batch.items()} for idx in range(4)]
    metrics = compute_eval_metrics(_config(), samples)

    assert metrics["rollout-rewards/global_rewards"] == 0.5
    assert metrics["rollout-metrics/global_retool_acc"] == 0.5
    assert metrics["multi_segment/eval_trajectories"] == 2.0
    assert metrics["multi_segment/eval_valid_trajectories"] == 1.0
    assert metrics["multi_segment/eval_segments_per_trajectory_mean"] == 2.0


def test_multi_segment_eval_reward_signals_before_delayed_result():
    async def run():
        reward = RetoolMultiSegmentReward(
            config=SimpleNamespace(external_reward=SimpleNamespace(eval_delay_s=60)),
        )
        started = asyncio.Event()
        release = asyncio.Event()

        async def wait_for_release(delay):
            assert delay == 60
            await release.wait()

        with patch(
            "tasks.retool.multi_segment.external_reward.asyncio.sleep",
            side_effect=wait_for_release,
        ):
            task = asyncio.create_task(reward.calc_external_reward(
                [_eval_batch()], ppo_step=4, is_eval=True, _started_event=started,
            ))
            await asyncio.wait_for(started.wait(), timeout=1)
            assert not task.done()
            release.set()
            updates = await task
        assert updates[0]["rewards"][2].item() == 1.0

    asyncio.run(run())


def test_train_external_reward_scores_once_without_eval_delay():
    async def run():
        reward = RetoolMultiSegmentReward(
            config=SimpleNamespace(external_reward=SimpleNamespace(eval_delay_s=300)),
        )
        actor = AgentLoopActor.__new__(AgentLoopActor)
        actor.worker_id = 1
        actor.training_config = SimpleNamespace(use_external_reward=True)
        actor.external_reward = reward
        actor._stream_external_reward = False
        batch = _eval_batch()
        actor.generate_batches = AsyncMock(return_value=[batch])
        actor.score_gen_rm_batches = AsyncMock(side_effect=lambda rbs, *args, **kwargs: rbs)
        actor._maybe_put_rollout_batches_to_tq = AsyncMock(side_effect=lambda rbs, *args: rbs)
        from tasks.retool.async_agent_loop.reward_utils import compute_retool_reward

        with patch(
            "tasks.retool.multi_segment.external_reward.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep, patch(
            "tasks.retool.multi_segment.external_reward.compute_retool_reward",
            wraps=compute_retool_reward,
        ) as score:
            result = await actor._batch_loop([{}], 0, [0], use_colocate=False)

        sleep.assert_not_awaited()
        score.assert_called_once_with("\\boxed{4}", "4", 2)
        assert [value.item() for value in result[0]["rewards"]] == [0.0, 0.0, 1.0, 0.0]
        assert [value.item() for value in result[0]["retool_acc"]] == [1.0, 1.0, 1.0, 0.0]

    asyncio.run(run())


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("eval_override", [False, True])
def test_controller_setup_eval_reward_respects_master_switch(enabled, eval_override):
    async def run():
        train_info = SimpleNamespace(reward_py_path="train.py", reward_cls_name="TrainReward")
        eval_info = SimpleNamespace(reward_py_path="eval.py", reward_cls_name="EvalReward")
        training = SimpleNamespace(
            use_fast_tokenizer=True,
            eval_interval=4,
            use_external_reward=enabled,
            ppo_filter_samplings_path=None,
            custom_convert_samples_to_train_data_path=None,
            custom_compute_rollout_metrics_path=None,
            custom_compute_eval_metrics_path=None,
            use_bt_rm_reward=False,
            rollout_gbs=1,
            rollout_mbs=1,
            eval_rollout_gbs=1,
            eval_rollout_mbs=1,
            rb_multiplier=1,
            num_agent_loop_workers=1,
        )
        controller = RolloutController.__new__(RolloutController)
        controller.config = SimpleNamespace(
            training=training,
            policy=SimpleNamespace(
                hf_tokenizer_path="unused",
                dist_config=SimpleNamespace(
                    nnodes=1, num_gpus_per_node=1,
                    tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                ),
            ),
            ppo=SimpleNamespace(custom_reward_normalize_py_path=None),
            external_reward=SimpleNamespace(
                reward_info=[train_info],
                eval_reward_info=[eval_info] if eval_override else None,
            ),
        )
        controller.training_config = training
        controller._tail_batching = False
        controller.use_colocate = False
        controller.eval_external_reward = None
        controller._setup_streaming_teachers = AsyncMock()
        controller.setup_agent_loop_actors = AsyncMock()
        source = SimpleNamespace(
            eval_dataloader_length=1, dataloader_length=1, train_sampler=None,
        )
        module = "gpatch_v4.rollout_generator.async_rollout.rollout_controller"
        reward_cls = MagicMock()
        with patch(f"{module}.AutoTokenizer.from_pretrained"), patch(
            f"{module}.get_data_source", return_value=source,
        ), patch(f"{module}.ApplySamplingRolloutAttrFactory.get_apply_sampling"), patch(
            f"{module}.import_fn_from_path", return_value=reward_cls,
        ) as import_reward:
            await controller.setup()

        if enabled:
            selected = eval_info if eval_override else train_info
            import_reward.assert_called_once_with(selected.reward_py_path, selected.reward_cls_name)
            assert controller.eval_external_reward is reward_cls.return_value
        else:
            import_reward.assert_not_called()
            assert controller.eval_external_reward is None

    asyncio.run(run())


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("is_eval", [False, True])
@pytest.mark.parametrize("async_eval", [False, True])
@pytest.mark.parametrize("eval_override", [False, True])
def test_dup_traj_external_reward_respects_master_switch(
    monkeypatch, enabled, is_eval, async_eval, eval_override,
):
    manager = DupTrajEnvManager.__new__(DupTrajEnvManager)
    manager.env_config = {"env_id": 0, "config": {"dup_num_segments": 2}}
    manager.rl_config = SimpleNamespace(
        training=SimpleNamespace(use_external_reward=enabled),
        external_reward=SimpleNamespace(
            eval_reward_info=[object()] if eval_override else None,
            async_eval=async_eval,
        ),
    )
    manager._align_traj_path = lambda seed: None
    cache = SimpleNamespace(
        terminated=True,
        history=[{
            "_external_reward_response": "\\boxed{4}",
            "_external_reward_num_turns": 2,
            "_external_reward_success_eligible": True,
        }],
    )
    seen_data = []

    def reset(seed, ppo_step, data):
        seen_data.append(data)
        return cache

    manager._reset = reset
    monkeypatch.setattr(
        TrajEnvManager,
        "_formulate_rollout_batch",
        lambda self, rollout_cache: {
            "tokens": [torch.tensor([1, 2])],
            "rewards": [torch.tensor(0.0)],
        },
    )

    result = manager.run(1, 0, {"target": ["4"]}, is_eval=is_eval)

    if enabled:
        assert seen_data[0]["_defer_external_reward"] is True
        assert result["_external_reward_target"] == ["4", "4"]
        assert result["_external_reward_valid"] == [True, True]
        assert result["_external_reward_response"] == ["\\boxed{4}", "\\boxed{4}"]
    else:
        assert "_defer_external_reward" not in seen_data[0]
        assert "_external_reward_target" not in result
    assert all(value.item() == 0.0 for value in result["rewards"])


@pytest.mark.parametrize("enabled", [False, True])
def test_agent_loop_eval_reward_respects_master_switch(enabled):
    async def run():
        actor = AgentLoopActor.__new__(AgentLoopActor)
        actor.config = SimpleNamespace(
            external_reward=SimpleNamespace(eval_reward_info=[object()]),
        )
        actor.training_config = SimpleNamespace(use_external_reward=enabled)
        actor.worker_id = 1
        actor.external_reward = object() if enabled else None
        actor._stream_external_reward = False

        async def generate_batches(*args, is_eval=False, **kwargs):
            assert is_eval
            return [{"unique_id": ["prompt"], "tokens": [torch.tensor([1, 2])]}]

        actor.generate_batches = generate_batches
        actor.score_gen_rm_batches = AsyncMock(side_effect=lambda rbs, *args, **kwargs: rbs)

        result = await actor._batch_loop([{}], 4, [0], use_colocate=False, is_eval=True)
        assert isinstance(result, AgentLoopEvalResult)
        assert result.deferred_external_reward_indices == ([0] if enabled else [])

    asyncio.run(run())


def test_agentic_eval_sampling_uses_eval_parameters_and_logs_effective_cap():
    async def run():
        params = SimpleNamespace(
            temperature=1.0, top_k=1, top_p=1.0, min_p=0.0,
            max_new_tokens=128,
        )
        engine = SimpleNamespace(
            get_sampling_params_from_config=MagicMock(return_value=params),
            copy_sampling_params_with_seed_offset=MagicMock(
                side_effect=lambda value, seed: SimpleNamespace(**vars(value))
            ),
            async_generate=MagicMock(return_value=object()),
            wait_and_get_async_generate_output=AsyncMock(return_value=[
                SimpleNamespace(outputs=[SimpleNamespace(
                    token_ids=[3], prompt_len=2, output_logprobs=None,
                    finish_reason="stop",
                )]),
            ]),
        )
        infer_cfg = SimpleNamespace(eval_generate_params=SimpleNamespace(
            temperature=1.0, top_k=1, top_p=1.0, min_p=0.0,
        ))
        config = SimpleNamespace(
            policy=SimpleNamespace(hf_config=SimpleNamespace()),
            sampler=SimpleNamespace(infer_engine_configs=[infer_cfg]),
            training=SimpleNamespace(eval_sampling_repeat_n=1, moe_router_replay=False),
        )
        with patch("gpatch_v4.agentic.generate_func.log") as log:
            await SamplerGenerateFuncAgentic()(
                config, engine, 0, SimpleNamespace(eos_token_id=2),
                {"prompt_token_ids": [torch.tensor([1, 2])], "max_new_tokens": [64]},
                sampling_repeat_n=1, is_eval=True,
            )
        assert engine.get_sampling_params_from_config.call_args.kwargs["is_eval"] is True
        effective = engine.async_generate.call_args.args[1]
        assert (effective.temperature, effective.top_k, effective.top_p, effective.min_p) == (
            1.0, 1, 1.0, 0.0
        )
        assert effective.max_new_tokens == 64
        assert "eval_sampling_repeat_n=1" in log.call_args.args[0]
        assert "top_k=1" in log.call_args.args[0]

    asyncio.run(run())


def test_controller_uses_eval_specific_metrics_hook():
    batch = _eval_batch()
    batch.update(RetoolMultiSegmentReward._compute_update(batch))
    controller = RolloutController.__new__(RolloutController)
    controller.config = _config()
    controller.training_config = controller.config.training
    controller.custom_compute_eval_metrics = compute_eval_metrics

    metrics = controller._compute_eval_metrics([batch])

    assert metrics["eval-rollout-rewards/global_rewards"] == 0.5
    assert metrics["eval-multi_segment/eval_trajectories"] == 2.0


def test_controller_eval_reward_can_report_after_training_continues():
    async def run():
        release = asyncio.Event()

        class DelayedReward:
            async def calc_external_reward(
                self, batches, ppo_step, is_eval=False, _started_event=None,
            ):
                assert ppo_step == 4 and is_eval
                _started_event.set()
                await release.wait()
                return [{"rewards": [torch.tensor(1.0)]} for _ in batches]

        controller = RolloutController.__new__(RolloutController)
        controller.config = SimpleNamespace(
            external_reward=SimpleNamespace(async_eval=True),
        )
        controller.training_config = SimpleNamespace(
            eval_interval=1,
            use_external_reward=True,
            use_bt_rm_reward=False,
        )
        controller.data_source = SimpleNamespace(reset_eval=MagicMock())
        controller.eval_apply_sampling_rollout_attr = SimpleNamespace(
            clear_data_cache=MagicMock(),
        )
        controller.eval_external_reward = DelayedReward()
        controller._pending_eval = None
        controller._total_eval_step = 1
        controller._num_eval_microbatches = 1
        controller._read_eval_batches = MagicMock(return_value=[{}])
        controller._dispatch_eval_step = AsyncMock(return_value=([
            {"rewards": [torch.tensor(0.0)]},
        ], [0]))
        controller._compute_eval_metrics = MagicMock(return_value={"eval-reward": 1.0})

        with patch(
            "gpatch_v4.rollout_generator.async_rollout.rollout_controller.check_rollout_batches",
            return_value=True,
        ):
            immediate = await controller.evaluate(4)
            assert immediate == []
            assert controller._pending_eval is not None
            assert not controller.pending_eval_ready()
            release.set()
            completed = await controller.poll_eval(force=True)

        assert completed.source_step == 4
        assert completed.metrics == {"eval-reward": 1.0}
        controller.eval_apply_sampling_rollout_attr.clear_data_cache.assert_called_once()

    asyncio.run(run())


def test_trajectory_weights_preserve_response_padded_seq_mean_loss():
    baseline_values = torch.tensor([[2.0, 2.0], [8.0, 8.0]])
    baseline_mask = torch.ones_like(baseline_values)
    baseline_sum, baseline_count = agg(
        baseline_values,
        baseline_mask,
        calculate_per_token_loss=False,
    )

    duplicated_values = torch.tensor(
        [[2.0, 2.0], [8.0, 8.0], [8.0, 8.0], [8.0, 8.0]]
    )
    duplicated_mask = torch.ones_like(duplicated_values)
    token_weights = torch.tensor([[2.0], [2.0 / 3.0], [2.0 / 3.0], [2.0 / 3.0]])
    duplicated_sum, duplicated_count = agg(
        duplicated_values,
        duplicated_mask,
        calculate_per_token_loss=False,
        token_weights=token_weights,
    )

    assert torch.allclose(baseline_sum / baseline_count, duplicated_sum / duplicated_count)


def test_fsdp2_seq_mean_scale_uses_actual_segment_count():
    from gpatch_v4.training_backend.fsdp2_backend.fsdp2_engine_lm import (
        Fsdp2EngineLm,
    )

    baseline_values = torch.tensor([[2.0, 2.0], [8.0, 8.0]])
    baseline_sum, baseline_count = agg(
        baseline_values,
        torch.ones_like(baseline_values),
        calculate_per_token_loss=False,
    )
    duplicated_values = torch.tensor(
        [[2.0, 2.0], [8.0, 8.0], [8.0, 8.0], [8.0, 8.0]]
    )
    duplicated_sum, duplicated_count = agg(
        duplicated_values,
        torch.ones_like(duplicated_values),
        calculate_per_token_loss=False,
        token_weights=torch.tensor([[2.0], [2.0 / 3.0], [2.0 / 3.0], [2.0 / 3.0]]),
    )

    baseline_scaled = Fsdp2EngineLm._scale_fsdp2_rl_bwd_loss(
        baseline_sum,
        sum_return_loss=True,
        dp_size=1,
        num_microbatches=1,
        global_loss_count=baseline_count,
    )
    duplicated_scaled = Fsdp2EngineLm._scale_fsdp2_rl_bwd_loss(
        duplicated_sum,
        sum_return_loss=True,
        dp_size=1,
        num_microbatches=1,
        global_loss_count=duplicated_count,
    )
    torch.testing.assert_close(baseline_scaled, duplicated_scaled)


def test_per_token_weights_preserve_trajectory_equal_loss_for_variable_lengths():
    samples = [
        _sample(0, 0, 1.0, -1.0, response_tokens=1),
        _sample(1, 0, 0.0, 1.0, response_tokens=3),
        _sample(1, 1, 0.0, 1.0, response_tokens=3),
        _sample(1, 2, 3.0, 1.0, response_tokens=3),
    ]
    converted = convert_samples_to_train_data(
        _config(calculate_per_token_loss=True),
        samples,
    )
    token_weights = torch.tensor(
        [sample["token_weights"].item() for sample in converted]
    ).unsqueeze(1)
    values = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [8.0, 8.0, 8.0],
            [8.0, 8.0, 8.0],
            [8.0, 8.0, 8.0],
        ]
    )
    mask = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    weighted_sum, token_count = agg(
        values,
        mask,
        calculate_per_token_loss=True,
        token_weights=token_weights,
    )

    assert torch.allclose(weighted_sum / token_count, torch.tensor(5.0))


@patch(
    "gpatch_v4.training_backend.loss.ppo_loss.reduce_metrics_across_data_parallel_group"
)
def test_grpo_loss_accepts_response_padded_trajectory_weights(_mock_reduce_metrics):
    config = SimpleNamespace(
        ppo=PpoConfig(
            loss_func="grpo",
            grpo_kl_loss_beta=0.0,
        ),
        debug=DebugConfig(),
    )
    baseline_advantages = torch.tensor([[2.0, 2.0], [8.0, 8.0]])
    baseline_log_probs = torch.zeros_like(baseline_advantages)
    baseline_input = PolicyLossInput(
        advantages=baseline_advantages,
        prev_log_probs=baseline_log_probs,
        ref_log_probs=None,
        curr_log_probs=baseline_log_probs,
        response_mask=torch.ones_like(baseline_advantages),
        scaled_entropy=torch.tensor(0.0),
        per_token_entropy=torch.zeros_like(baseline_advantages),
        calculate_per_token_loss=False,
    )
    baseline_sum, baseline_count, _ = get_loss_fn("fsdp2", "grpo")(
        config,
        baseline_input,
    )

    duplicated_advantages = torch.tensor(
        [[2.0, 2.0], [8.0, 8.0], [8.0, 8.0], [8.0, 8.0]]
    )
    duplicated_log_probs = torch.zeros_like(duplicated_advantages)
    duplicated_input = PolicyLossInput(
        advantages=duplicated_advantages,
        prev_log_probs=duplicated_log_probs,
        ref_log_probs=None,
        curr_log_probs=duplicated_log_probs,
        response_mask=torch.ones_like(duplicated_advantages),
        scaled_entropy=torch.tensor(0.0),
        per_token_entropy=torch.zeros_like(duplicated_advantages),
        token_weights=torch.tensor(
            [[2.0], [2.0 / 3.0], [2.0 / 3.0], [2.0 / 3.0]]
        ),
        calculate_per_token_loss=False,
    )
    duplicated_sum, duplicated_count, _ = get_loss_fn("fsdp2", "grpo")(
        config,
        duplicated_input,
    )

    assert torch.allclose(
        baseline_sum / baseline_count,
        duplicated_sum / duplicated_count,
    )


def test_dynamic_batch_train_allows_fsdp2():
    from gpatch_v4.configs.training_config import RLTrainingConfig

    cfg = RLTrainingConfig(
        single_controller=True,
        dynamic_batch_train=True,
        training_backend="fsdp2",
    )
    assert cfg.training_backend == "fsdp2"


def test_is_flat_sample_list_distinguishes_rollout_batches():
    from gpatch_v4.utils.training_utils import is_flat_sample_list

    assert is_flat_sample_list([{"tokens": torch.tensor([1, 2, 3])}])
    assert not is_flat_sample_list([{"tokens": [torch.tensor([1, 2, 3])]}])


def test_fsdp2_dynamic_cp_logprobs_keep_flat_sample_shape():
    from gpatch_v4.training_backend.fsdp2_backend.fsdp2_engine_lm import (
        Fsdp2EngineLm,
    )

    samples = [
        {"tokens": torch.tensor([1, 2, 3])},
        {"tokens": torch.tensor([4, 5])},
    ]
    expected = [torch.tensor([0.1, 0.2]), torch.tensor([0.3])]
    engine = Fsdp2EngineLm.__new__(Fsdp2EngineLm)
    engine.policy_config = SimpleNamespace(
        without_ref=True,
        dist_config=SimpleNamespace(max_seqlen_per_dp_cp_rank_fwd_only=8),
    )
    engine.config = SimpleNamespace(
        training=SimpleNamespace(
            moe_pad_with_random_tokens=False,
            moe_router_replay=False,
        ),
    )
    engine.tokenizer = SimpleNamespace(pad_token_id=0)
    engine.model = object()
    engine.prepare_data = SimpleNamespace(
        rl_reroute_data_for_dynamic_cp=MagicMock(
            return_value=([{}], 1, 0.0, 0.0, {"routing": True})
        ),
    )
    engine._get_vocab_size = MagicMock(return_value=16)
    engine.onload_model = MagicMock()
    engine.set_model_eval = MagicMock()
    engine._forward_dynamic_cp_logprobs = MagicMock(return_value={})
    engine._reverse_dynamic_cp_logprobs = MagicMock(return_value=expected)

    with patch(
        "gpatch_v4.training_backend.fsdp2_backend.fsdp2_engine_lm."
        "expand_rollout_batches"
    ) as expand:
        ref_logps, prev_logps = engine.compute_log_probs_dynamic_cp(samples)

    expand.assert_not_called()
    assert ref_logps is None
    assert prev_logps is expected


def test_fsdp2_dynamic_cp_response_pad_keeps_token_weights():
    from gpatch_v4.training_backend.fsdp2_backend.mixin.forward import (
        ForwardStepMixin,
    )

    packed = torch.zeros(1, 6)
    batch = {
        "cp_group": object(),
        "local_cp_size": torch.tensor(1),
        "cp_partition_mode": "zigzag",
        "cu_seqlens_padded": torch.tensor([0, 3, 6], dtype=torch.int32),
        "dyn_cp_response_start": torch.tensor([1, 0], dtype=torch.int32),
        "dyn_cp_response_length": torch.tensor([2, 2], dtype=torch.int32),
        "mask": torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 0.0]]),
        "advantages": packed,
        "prev_log_probs": packed,
        "sample_mask": torch.ones(1, 6),
        "token_weights": torch.tensor([[0.0, 2.0, 2.0, 0.5, 0.5, 0.0]]),
    }
    with patch(
        "gpatch_v4.training_backend.fsdp2_backend.mixin.forward."
        "reconstruct_dynamic_cp_packed_tensor",
        side_effect=lambda tensor, *_args, **_kwargs: tensor,
    ):
        padded = ForwardStepMixin()._dynamic_cp_response_pad(
            batch,
            curr_log_probs=packed,
            entropy=packed,
        )

    assert padded["token_weights"].tolist() == [[2.0, 2.0], [0.5, 0.5]]
    assert padded["sample_mask"].tolist() == [1.0, 1.0]


def test_qwen38_retool_recipe_points_at_fsdp2_qwen4_exp():
    import pathlib

    import yaml

    task_dir = pathlib.Path(__file__).resolve().parents[2] / "tasks" / "retool" / "multi_segment"
    recipe = yaml.safe_load(
        (task_dir / "yaml" / "dup_traj_dynamic_batch_retool_qwen38_w_tq.yaml").read_text()
    )
    script = (task_dir / "scripts" / "run_dup_traj_dynamic_batch_retool_qwen38_w_tq.sh").read_text()

    assert recipe["training"]["training_backend"] == "fsdp2"
    assert recipe["training"]["rollout_gbs"] == 64
    tool_config = recipe["training"]["agentic"]["env_cfg_template"]["env_tool_config"]
    assert tool_config["tool_call_parser"] == "sglang/qwen3_coder"
    assert recipe["policy"]["model_arch"] == "qwen4_exp"
    assert recipe["policy"]["dist_config"]["expert_model_parallel_size"] == 32
    assert recipe["policy"]["override_transformer_config"]["calculate_per_token_loss"] is False
    assert recipe["sampler"]["model_info"][0]["model_arch"] == "qwen4_exp"
    assert recipe["ppo"]["loss_func"] == "gspo"
    assert recipe["ppo"]["ppo_clip_ratio_low"] == 0.0003
    assert recipe["ppo"]["ppo_clip_ratio_high"] == 0.0004
    assert "dup_traj_dynamic_batch_retool_qwen38_w_tq" in script
    assert "train_retool_multi_segment.py" in script
    assert "NCCL_NET_PLUGIN=none" in script
    assert "local_sandbox_server.py" in script
