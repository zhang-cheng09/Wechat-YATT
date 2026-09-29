"""E2H GRPO e2e: dapo_dataset + gaussian sampler metrics.

Smoke: 2 ppo_steps. Train jsonl is DAPO (`prompt`/`label`) with synthetic
``level = i % 4``. Assert e2h schedule/frac metrics and
``total_ppo_step = steps_per_epoch * num_train_epoches``.

Also: load ckpt after step 0 and check ``set_start_index`` skips that
step's prompts (compare dumped prompt tokens).
"""
import glob
import json
import os
import re
import shutil
import unittest

import numpy as np
import pytest
import torch

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.trainer import GrpoTrainer
from gpatch_v4.utils.e2h_sampler import E2HTaskSampler
from gpatch_v4_test_helper import (
    kill_all_actors_and_shutdown_ray,
    load_config,
    requires_sglang,
)

_CKPT = "save_e2h_grpo_e2e"
_CKPT_RESUME = "save_e2h_grpo_e2e_resume"
_CKPT_RESUME_B = "save_e2h_grpo_e2e_resume_b"
_ROLLOUT_A = "debug-tmp-e2h-resume-a"
_ROLLOUT_B = "debug-tmp-e2h-resume-b"
_DATASET_DIR = "tests/test_gpatch_v4/e2h_dapo_small"
_DATASET_N_LINES = 32
_NUM_LEVELS = 4
_CONFIG_NAME = "test_e2h_grpo_e2e"


def _extract_gt_answer(text):
    match = re.search(r"####\s*(-?\d+(\.\d+)?)", text)
    if match is None:
        return None
    raw = match.group(1)
    return float(raw) if "." in raw else int(raw)


def _prepare_dapo_e2h(n_lines: int = _DATASET_N_LINES, num_levels: int = _NUM_LEVELS) -> np.ndarray:
    os.makedirs(_DATASET_DIR, exist_ok=True)
    src = "hf-hub/DaertML/gsm8k-jsonl/train.jsonl"
    dst = os.path.join(_DATASET_DIR, "train.jsonl")
    level_list = []
    with open(src, "r") as fin, open(dst, "w") as fout:
        for i, line in enumerate(fin):
            if i >= n_lines:
                break
            rec = json.loads(line)
            gt = _extract_gt_answer(str(rec["answer"]))
            label = str(gt) if gt is not None else str(rec["answer"])
            level = i % num_levels
            level_list.append(level)
            fout.write(
                json.dumps(
                    {
                        "prompt": [{"role": "user", "content": rec["question"]}],
                        "label": label,
                        "level": level,
                    },
                    ensure_ascii=False,
                ) + "\n"
            )
    levels = np.asarray(level_list, dtype=np.int64)
    if int(levels.shape[0]) != n_lines:
        raise AssertionError(f"expected {n_lines} train rows, wrote {levels.shape[0]}")
    return levels


def _expected_sampler(levels: np.ndarray, config: RlConfig) -> E2HTaskSampler:
    e2h = config.data.e2h
    return E2HTaskSampler(
        levels=levels,
        rollout_gbs=config.training.rollout_gbs,
        rollout_mbs=config.training.rollout_mbs,
        dp_rank=0,
        dp_size=1,
        seed=config.data.sampler_seed,
        num_epochs=config.training.num_train_epoches,
        curriculum_schedule=e2h.curriculum_schedule,
        mu_exp=e2h.mu_exp,
        mu_end=e2h.mu_end,
        hard_end_prob=e2h.hard_end_prob,
        sigma=e2h.sigma,
        min_prob=e2h.min_prob,
        epoch_from_hard_cover=e2h.set_total_ppo_step_from_hard_cover,
        total_ppo_step=config.training.total_ppo_step,
    )


def _first_rank_steps(metrics) -> list:
    assert metrics is not None and len(metrics) > 0
    steps = metrics[0]
    assert isinstance(steps, list) and len(steps) > 0
    return steps


def _force_latest_iteration(ckpt_root: str, step: int) -> None:
    path = os.path.join(ckpt_root, "latest_checkpointed_iteration.txt")
    assert os.path.isfile(path), f"missing {path}"
    with open(path, "w", encoding="utf-8") as fout:
        fout.write(str(step))


def _prompt_token_tuples(rollout_dir: str, ppo_step: int) -> list[tuple[int, ...]]:
    paths = sorted(glob.glob(os.path.join(rollout_dir, f"rollout_batches_{ppo_step}_*.pt")))
    assert paths, f"no rollout_batches_{ppo_step}_*.pt under {rollout_dir}"
    batches = torch.load(paths[0], map_location="cpu", weights_only=False)
    assert isinstance(batches, list) and len(batches) > 0
    prompts = []
    for rb in batches:
        tokens_list = rb["tokens"]
        prompt_lens = rb["prompt_lengths"]
        for toks, plen in zip(tokens_list, prompt_lens, strict=True):
            if hasattr(toks, "tolist"):
                toks = toks.tolist()
            plen_i = int(plen.item()) if hasattr(plen, "item") else int(plen)
            prompts.append(tuple(int(x) for x in toks[:plen_i]))
    assert prompts, f"empty prompts in {paths[0]}"
    return prompts


def _stash_debug_tmp(dst: str) -> None:
    shutil.rmtree(dst, ignore_errors=True)
    assert os.path.isdir("debug-tmp"), "save_every_rollout_data did not write debug-tmp"
    shutil.move("debug-tmp", dst)


class TestE2hGrpoE2e(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def _run_train(self, config: RlConfig):
        trainer = GrpoTrainer()
        return await trainer.launch_then_run_with_recovery(config)

    @requires_sglang
    async def test_e2h_metrics_and_hard_cover_horizon(self):
        levels = _prepare_dapo_e2h()
        shutil.rmtree(_CKPT, ignore_errors=True)

        config = load_config(_CONFIG_NAME, RlConfig)
        assert config.data.e2h.enable
        assert config.debug.trainer_return_ppo_step_metrics
        sampler = _expected_sampler(levels, config)
        expected_total = sampler.steps_per_epoch * sampler.num_epochs
        self.assertEqual(config.training.num_train_epoches, 2)
        self.assertEqual(expected_total, sampler.steps_per_epoch * 2)
        self.assertGreater(expected_total, config.training.exit_step)

        metrics = await self._run_train(config)
        kill_all_actors_and_shutdown_ray()

        steps = _first_rank_steps(metrics)
        assert len(steps) == 2, f"expected 2 ppo steps, got {len(steps)}"
        k = sampler.num_tasks
        for i, step in enumerate(steps):
            p_keys = [f"e2h/p_level_{j}" for j in range(k)]
            frac_keys = [f"e2h/frac_level_{j}" for j in range(k)]
            for key in p_keys + frac_keys + ["e2h/t_frac"]:
                assert key in step, f"step {i} missing {key}; keys={sorted(step)}"
            p_sum = sum(float(step[key]) for key in p_keys)
            frac_sum = sum(float(step[key]) for key in frac_keys)
            assert p_sum == pytest.approx(1.0, abs=1e-5), p_sum
            assert frac_sum == pytest.approx(1.0, abs=1e-5), frac_sum
            t_frac = float(step["e2h/t_frac"])
            assert 0.0 <= t_frac < 1.0, t_frac
        # gaussian t=0 质量在 easy
        assert float(steps[0]["e2h/p_level_0"]) > float(steps[0][f"e2h/p_level_{k - 1}"])
        shutil.rmtree(_CKPT, ignore_errors=True)

    @requires_sglang
    async def test_e2h_ckpt_resume_skips_consumed_prompts(self):
        # Run A 训 step 0/1；把 latest 钉到 1 再续训，第一步 prompt 必须等于 A 的 step 1。
        levels = _prepare_dapo_e2h()
        leftover = (_CKPT_RESUME, _CKPT_RESUME_B, _ROLLOUT_A, _ROLLOUT_B, "debug-tmp")
        for path in leftover:
            shutil.rmtree(path, ignore_errors=True)
        try:
            config_a = load_config(_CONFIG_NAME, RlConfig)
            config_a.training.exit_step = 2
            config_a.training.save_interval = 1
            config_a.debug.save_every_rollout_data = True
            config_a.checkpoint.load_ckpt_path = _CKPT_RESUME
            config_a.checkpoint.save_ckpt_path = _CKPT_RESUME
            sampler = _expected_sampler(levels, config_a)
            self.assertGreater(sampler.steps_per_epoch, 1)

            metrics_a = await self._run_train(config_a)
            kill_all_actors_and_shutdown_ray()
            steps_a = _first_rank_steps(metrics_a)
            assert len(steps_a) == 2, f"expected 2 ppo steps, got {len(steps_a)}"
            _stash_debug_tmp(_ROLLOUT_A)
            prompts_a0 = _prompt_token_tuples(_ROLLOUT_A, 0)
            prompts_a1 = _prompt_token_tuples(_ROLLOUT_A, 1)
            self.assertNotEqual(prompts_a0, prompts_a1)

            _force_latest_iteration(_CKPT_RESUME, 1)
            config_b = load_config(_CONFIG_NAME, RlConfig)
            config_b.training.exit_step = 2
            config_b.debug.save_every_rollout_data = True
            config_b.checkpoint.load_ckpt_path = _CKPT_RESUME
            # 续训结束仍会 save final；另写目录，避免撞上 A 已写的 hf/2
            config_b.checkpoint.save_ckpt_path = _CKPT_RESUME_B
            metrics_b = await self._run_train(config_b)
            kill_all_actors_and_shutdown_ray()
            steps_b = _first_rank_steps(metrics_b)
            assert len(steps_b) == 1, f"resume should train 1 step, got {len(steps_b)}"
            _stash_debug_tmp(_ROLLOUT_B)
            prompts_b1 = _prompt_token_tuples(_ROLLOUT_B, 1)
            self.assertEqual(prompts_b1, prompts_a1)
            self.assertNotEqual(prompts_b1, prompts_a0)
        finally:
            for path in leftover:
                shutil.rmtree(path, ignore_errors=True)
