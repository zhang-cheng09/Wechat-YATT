import unittest

import numpy as np
from torch.utils.data import DataLoader, Dataset

from gpatch_v4.configs.data_config import E2hConfig
from gpatch_v4.utils.e2h_sampler import (
    E2HTaskSampler,
    approx_cover_steps,
    classic_schedule,
    cosine_schedule,
    expected_cover_steps,
    frac_metrics_from_counts,
    frac_metrics_from_levels,
    gaussian_mu_end_from_hard_prob,
    gaussian_schedule,
    hard_cover_total_ppo_step,
    calc_level_counts,
    collect_e2h_rollout_metrics,
    levels_from_rollout_batches,
    probs_to_array,
    remap_levels,
    simulate_unique_cover,
)
from tasks.math_rl_v4.dapo_dataset import collate_fn


class _IndexDataset(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return {"idx": idx, "prompt_len": 1, "input_ids": [1], "gt_label": 0.0}


def _levels_block(n_per_level: int, k: int) -> np.ndarray:
    return np.repeat(np.arange(k, dtype=np.int64), n_per_level)


def _make_sampler(
    levels: np.ndarray,
    rollout_gbs: int,
    dp_rank: int = 0,
    dp_size: int = 1,
    seed: int = 42,
    num_epochs: int = 2,
    curriculum_schedule: str = "gaussian",
    rollout_mbs: int = 1,
    mu_exp: float = 0.5,
    mu_end: float | None = None,
    hard_end_prob: float | None = None,
    sigma: float = 0.5,
    min_prob: bool | float = True,
    epoch_from_hard_cover: bool = True,
    total_ppo_step: int | None = None,
) -> E2HTaskSampler:
    return E2HTaskSampler(
        levels=levels,
        rollout_gbs=rollout_gbs,
        rollout_mbs=rollout_mbs,
        dp_rank=dp_rank,
        dp_size=dp_size,
        seed=seed,
        num_epochs=num_epochs,
        curriculum_schedule=curriculum_schedule,
        mu_exp=mu_exp,
        mu_end=mu_end,
        hard_end_prob=hard_end_prob,
        sigma=sigma,
        min_prob=min_prob,
        epoch_from_hard_cover=epoch_from_hard_cover,
        total_ppo_step=total_ppo_step,
    )


def _epoch_indices(sampler: E2HTaskSampler, mbs: int = 1) -> list[list[int]]:
    dataset = _IndexDataset(len(sampler.levels))
    loader = DataLoader(dataset, sampler=sampler, batch_size=mbs, drop_last=True)
    return [[int(x) for x in batch["idx"]] for batch in loader]


def _chunk_rank_steps(indices: list[int], prompts_per_rank: int) -> list[list[int]]:
    if prompts_per_rank <= 0 or len(indices) % prompts_per_rank != 0:
        raise AssertionError(
            f"rank yield len {len(indices)} not divisible by {prompts_per_rank}"
        )
    return [
        indices[i:i + prompts_per_rank]
        for i in range(0, len(indices), prompts_per_rank)
    ]


def _rebuild_global_step(per_rank: list[list[int]], dp_size: int, gbs: int) -> list[int]:
    rebuilt = [None] * gbs
    for rank in range(dp_size):
        if len(per_rank[rank]) != gbs // dp_size:
            raise AssertionError(
                f"rank {rank} step size {len(per_rank[rank])} != {gbs // dp_size}"
            )
        for j, idx in enumerate(per_rank[rank]):
            rebuilt[rank + j * dp_size] = idx
    if any(x is None for x in rebuilt):
        raise AssertionError("DP stride left holes in the global batch")
    return rebuilt


def _assert_step_partition(per_rank: list[list[int]], dp_size: int, gbs: int) -> list[int]:
    rebuilt = _rebuild_global_step(per_rank, dp_size, gbs)
    for i in range(dp_size):
        for j in range(i + 1, dp_size):
            overlap = set(per_rank[i]).intersection(per_rank[j])
            if overlap:
                raise AssertionError(
                    f"dp rank {i} and {j} share indices in one PPO step: {sorted(overlap)}"
                )
    if len(set(rebuilt)) != gbs:
        raise AssertionError(
            f"global step has duplicate indices: {rebuilt}"
        )
    return rebuilt


class RemapLevelsTest(unittest.TestCase):
    def test_sorted_unique_to_contiguous(self):
        mapped, unique = remap_levels(np.array([2, 0, 2, 4]))
        np.testing.assert_array_equal(unique, np.array([0, 2, 4]))
        np.testing.assert_array_equal(mapped, np.array([1, 0, 1, 2]))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            remap_levels(np.array([], dtype=np.int64))

    def test_non_integer_float_raises(self):
        with self.assertRaises(ValueError):
            remap_levels(np.array([0.5, 1.0]))


class ScheduleTest(unittest.TestCase):
    def test_probs_sum_to_one(self):
        k, t_total = 4, 20
        for t in (0, t_total // 2, t_total - 1):
            for probs in (
                classic_schedule(t, t_total, k),
                cosine_schedule(t, t_total, k),
                gaussian_schedule(t, t_total, k, mu_exp=0.5, sigma=0.5, min_prob=True),
            ):
                arr = probs_to_array(probs, k)
                self.assertAlmostEqual(float(arr.sum()), 1.0, places=6)
                self.assertTrue(np.all(arr >= 0.0))

    def test_classic_is_one_hot(self):
        k, t_total = 4, 8
        for t in range(t_total):
            arr = probs_to_array(classic_schedule(t, t_total, k), k)
            self.assertEqual(int(np.count_nonzero(arr)), 1)

    def test_cosine_moves_easy_to_hard(self):
        k, t_total = 4, 16
        early = probs_to_array(cosine_schedule(0, t_total, k), k)
        late = probs_to_array(cosine_schedule(t_total - 1, t_total, k), k)
        self.assertGreater(early[0], early[k - 1])
        self.assertGreater(late[k - 1], late[0])

    def test_gaussian_mass_moves_right(self):
        k, t_total = 4, 16
        early = probs_to_array(
            gaussian_schedule(0, t_total, k, mu_exp=0.5, sigma=0.5, min_prob=True), k
        )
        late = probs_to_array(
            gaussian_schedule(t_total - 1, t_total, k, mu_exp=0.5, sigma=0.5, min_prob=True),
            k,
        )
        self.assertEqual(int(np.argmax(early)), 0)
        self.assertEqual(int(np.argmax(late)), k - 1)

    def test_two_level_gaussian_reaches_configured_endpoint_and_dose(self):
        total_steps = 39
        kwargs = {
            "num_tasks": 2,
            "mu_exp": 1.5,
            "mu_end": 0.345240197898,
            "sigma": 0.5,
            "min_prob": False,
        }
        hard_probs = [
            gaussian_schedule(t, total_steps, **kwargs)[1]
            for t in range(total_steps)
        ]

        self.assertAlmostEqual(hard_probs[0], 0.1192029220, places=9)
        self.assertAlmostEqual(hard_probs[9], 0.1369535278, places=9)
        self.assertAlmostEqual(hard_probs[19], 0.1806786066, places=9)
        self.assertAlmostEqual(hard_probs[28], 0.2448001889, places=9)
        self.assertAlmostEqual(hard_probs[38], 0.35, places=9)
        self.assertAlmostEqual(sum(hard_probs) * 128, 995.2098247, places=5)

    def test_hard_end_prob_derives_two_level_mu_end(self):
        mu_end = gaussian_mu_end_from_hard_prob(
            num_tasks=2,
            sigma=0.5,
            min_prob=False,
            hard_end_prob=0.35,
        )
        final = gaussian_schedule(
            38,
            39,
            2,
            mu_exp=1.5,
            mu_end=mu_end,
            sigma=0.5,
            min_prob=False,
        )

        self.assertAlmostEqual(mu_end, 0.345240197898, places=11)
        self.assertAlmostEqual(final[1], 0.35, places=12)

    def test_hard_end_prob_supports_multiple_levels(self):
        mu_end = gaussian_mu_end_from_hard_prob(
            num_tasks=4,
            sigma=0.5,
            min_prob=False,
            hard_end_prob=0.35,
        )
        final = gaussian_schedule(
            20,
            21,
            4,
            mu_exp=1.5,
            mu_end=mu_end,
            sigma=0.5,
            min_prob=False,
        )

        self.assertGreater(mu_end, 0.0)
        self.assertLess(mu_end, 1.0)
        self.assertAlmostEqual(final[3], 0.35, places=12)

    def test_hard_end_prob_rejects_unreachable_target(self):
        with self.assertRaisesRegex(ValueError, "attainable range"):
            gaussian_mu_end_from_hard_prob(
                num_tasks=2,
                sigma=0.5,
                min_prob=True,
                hard_end_prob=0.35,
            )


class E2hConfigTest(unittest.TestCase):
    def test_disabled_skips_validation(self):
        cfg = E2hConfig(enable=False, curriculum_schedule="not-a-schedule")
        self.assertFalse(cfg.enable)

    def test_enabled_rejects_unknown_schedule(self):
        with self.assertRaises(ValueError):
            E2hConfig(enable=True, curriculum_schedule="linear")

    def test_enabled_accepts_min_prob_false(self):
        cfg = E2hConfig(enable=True, min_prob=False)
        self.assertIs(cfg.min_prob, False)

    def test_enabled_rejects_mu_end_outside_level_axis(self):
        with self.assertRaises(ValueError):
            E2hConfig(enable=True, mu_end=1.01)

    def test_enabled_accepts_hard_end_prob(self):
        cfg = E2hConfig(enable=True, hard_end_prob=0.35)
        self.assertEqual(cfg.hard_end_prob, 0.35)
        self.assertIsNone(cfg.mu_end)

    def test_enabled_rejects_both_gaussian_endpoints(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            E2hConfig(enable=True, mu_end=0.5, hard_end_prob=0.35)


class E2HTaskSamplerTest(unittest.TestCase):
    def test_classic_step_zero_only_level_zero(self):
        # classic 在 t=0 只抽 level 0，整步内 P(k) 不变
        k, n_per = 4, 16
        levels = _levels_block(n_per, k)
        gbs = 8
        sampler = _make_sampler(levels, gbs, curriculum_schedule="classic", num_epochs=2)
        idxs = list(sampler)[: sampler.prompts_per_rank]
        self.assertTrue(np.all(levels[idxs] == 0))

    def test_dp_ranks_reconstruct_global_batch(self):
        k, n_per, gbs, dp = 4, 16, 8, 4
        levels = _levels_block(n_per, k)
        per_rank = []
        for rank in range(dp):
            sampler = _make_sampler(
                levels, gbs, dp_rank=rank, dp_size=dp, curriculum_schedule="gaussian"
            )
            per_rank.append(list(sampler)[: sampler.prompts_per_rank])
        rebuilt = _assert_step_partition(per_rank, dp, gbs)
        sampler0 = _make_sampler(
            levels, gbs, dp_rank=0, dp_size=1, curriculum_schedule="gaussian"
        )
        expected, _ = sampler0._draw_global_indices(0, {t: 0 for t in range(k)})
        self.assertEqual(rebuilt, expected)

    def test_n_not_divisible_by_rollout_gbs_drops_tail(self):
        # 余数 drop_last，不成短 last step；prompt GBS 是 rollout_gbs 不是 train_gbs。
        k, n_per, gbs, dp = 4, 9, 8, 4
        levels = _levels_block(n_per, k)
        n = int(levels.shape[0])
        self.assertNotEqual(n % gbs, 0)
        ranks = []
        for rank in range(dp):
            sampler = _make_sampler(
                levels, gbs, dp_rank=rank, dp_size=dp, curriculum_schedule="gaussian"
            )
            self.assertEqual(sampler.num_dropped_tail, n % gbs)
            self.assertEqual(sampler.slot_steps_per_epoch, n // gbs)
            idxs = list(sampler)
            self.assertEqual(len(idxs), sampler.steps_per_epoch * sampler.prompts_per_rank)
            ranks.append(_chunk_rank_steps(idxs, sampler.prompts_per_rank))
        n_steps = ranks[0]
        self.assertEqual(len(n_steps), sampler.steps_per_epoch)
        for rank in range(1, dp):
            self.assertEqual(len(ranks[rank]), len(n_steps))
        for t in range(len(n_steps)):
            per_rank = [ranks[rank][t] for rank in range(dp)]
            _assert_step_partition(per_rank, dp, gbs)

    def test_resume_matches_fresh_run(self):
        k, n_per, gbs, dp, mbs = 4, 16, 8, 2, 1
        levels = _levels_block(n_per, k)
        resume_ppo_step = 3
        gas = gbs // (dp * mbs)
        full = []
        for rank in range(dp):
            sampler = _make_sampler(
                levels, gbs, dp_rank=rank, dp_size=dp, rollout_mbs=mbs, num_epochs=2
            )
            full.append(_epoch_indices(sampler, mbs=mbs))
        resumed = []
        for rank in range(dp):
            sampler = _make_sampler(
                levels, gbs, dp_rank=rank, dp_size=dp, rollout_mbs=mbs, num_epochs=2
            )
            sampler.set_start_index(resume_ppo_step * gas, mbs)
            resumed.append(_epoch_indices(sampler, mbs=mbs))
        skip = resume_ppo_step * gas
        for rank in range(dp):
            self.assertEqual(resumed[rank], full[rank][skip:])

    def test_start_index_shrinks_len_not_steps_per_epoch(self):
        # DataLoader resume 用缩短的 __len__；epoch 长度必须仍是 hard-cover S。
        levels = _levels_block(16, 4)
        gbs, mbs = 8, 1
        sampler = _make_sampler(levels, gbs, rollout_mbs=mbs)
        full_len = len(sampler)
        epoch_steps = sampler.steps_per_epoch
        sampler.set_start_index(3 * gbs, mbs)
        self.assertEqual(sampler.steps_per_epoch, epoch_steps)
        self.assertEqual(len(sampler), full_len - 3 * sampler.prompts_per_rank)
        self.assertGreater(full_len, len(sampler))

    def test_cross_epoch_restarts_curriculum_with_new_seed(self):
        k, n_per, gbs = 4, 16, 8
        levels = _levels_block(n_per, k)
        sampler = _make_sampler(levels, gbs, num_epochs=2, curriculum_schedule="gaussian")
        epoch0 = _epoch_indices(sampler)
        sampler.set_epoch(1)
        sampler.start_index = 0
        epoch1 = _epoch_indices(sampler)

        restarted = _make_sampler(levels, gbs, num_epochs=2, curriculum_schedule="gaussian")
        restarted.set_epoch(1)
        restarted.start_index = 0
        self.assertEqual(epoch1[0], _epoch_indices(restarted)[0])
        self.assertNotEqual(epoch0[0], epoch1[0])
        np.testing.assert_allclose(sampler.probs_at(0), sampler.probs_at(sampler.steps_per_epoch))
        self.assertEqual(len(epoch0), sampler.steps_per_epoch * sampler.prompts_per_rank)
        self.assertEqual(len(epoch1), sampler.steps_per_epoch * sampler.prompts_per_rank)
        boundary = _make_sampler(levels, gbs, num_epochs=2, curriculum_schedule="gaussian")
        boundary.set_start_index(sampler.steps_per_epoch * gbs, 1)
        self.assertEqual(boundary.epoch, 1)
        self.assertEqual(boundary.start_index, 0)
        self.assertEqual(_epoch_indices(boundary)[0], epoch1[0])

    def test_step_metrics_are_schedule_only(self):
        levels = _levels_block(16, 4)
        sampler = _make_sampler(levels, rollout_gbs=8, curriculum_schedule="classic")
        it = iter(sampler)
        for _ in range(sampler.prompts_per_rank):
            next(it)
        metrics = sampler.step_metrics(0)
        self.assertAlmostEqual(metrics["e2h/p_level_0"], 1.0)
        self.assertNotIn("e2h/frac_level_0", metrics)

    def test_frac_metrics_from_batch_levels(self):
        metrics = frac_metrics_from_levels([0, 0, 1, 1, 1, 1], num_tasks=3)
        self.assertAlmostEqual(metrics["e2h/frac_level_0"], 2.0 / 6.0)
        self.assertAlmostEqual(metrics["e2h/frac_level_1"], 4.0 / 6.0)
        self.assertAlmostEqual(metrics["e2h/frac_level_2"], 0.0)
        with self.assertRaises(ValueError):
            frac_metrics_from_levels([], num_tasks=2)
        with self.assertRaises(ValueError):
            frac_metrics_from_levels([3], num_tasks=2)

    def test_frac_metrics_from_dp_counts_not_mean_of_local_fracs(self):
        # 两档 rank 条数不同时，应对 count 求和，不能平均各 rank 的 frac。
        c0 = calc_level_counts([2] * 10, num_tasks=3)
        c1 = calc_level_counts([0] * 90, num_tasks=3)
        np.testing.assert_array_equal(c0, np.array([0, 0, 10]))
        np.testing.assert_array_equal(c1, np.array([90, 0, 0]))
        global_m = frac_metrics_from_counts(c0 + c1)
        self.assertAlmostEqual(global_m["e2h/frac_level_2"], 0.1)
        self.assertAlmostEqual(global_m["e2h/frac_level_0"], 0.9)
        local0 = frac_metrics_from_counts(c0)
        local1 = frac_metrics_from_counts(c1)
        self.assertAlmostEqual(
            (local0["e2h/frac_level_2"] + local1["e2h/frac_level_2"]) / 2.0, 0.5
        )
        np.testing.assert_array_equal(calc_level_counts([], num_tasks=3), np.zeros(3, dtype=np.int64))
        with self.assertRaises(ValueError):
            frac_metrics_from_counts(calc_level_counts([], num_tasks=3))

    def test_classic_hard_cover_is_last_stage(self):
        k, n_per, gbs = 4, 16, 8
        levels = _levels_block(n_per, k)
        sampler = _make_sampler(
            levels, gbs, curriculum_schedule="classic", num_epochs=1
        )
        report = simulate_unique_cover(sampler)
        T = sampler.total_iterations
        t_hard = next(
            t for t in range(T)
            if min(int(t * k / T), k - 1) == k - 1
        )
        need = (n_per + gbs - 1) // gbs
        self.assertEqual(report["steps"], t_hard + need)
        self.assertEqual(report["steps"], expected_cover_steps(sampler, k - 1))

    def test_balanced_expected_hard_cover_is_one_epoch(self):
        k, n_per, gbs = 4, 16, 8
        levels = _levels_block(n_per, k)
        sampler = _make_sampler(
            levels, gbs, curriculum_schedule="balanced", num_epochs=1
        )
        self.assertEqual(expected_cover_steps(sampler, k - 1), sampler.steps_per_epoch)
        report = simulate_unique_cover(sampler)
        self.assertIsNotNone(report["steps"])
        self.assertLessEqual(report["steps"], sampler.total_iterations)
        approx = approx_cover_steps(
            np.bincount(levels), gbs, curriculum_schedule="balanced"
        )
        self.assertEqual(approx["steps"], sampler.steps_per_epoch)
        self.assertEqual(approx["total_iterations"], sampler.total_iterations)

    def test_hard_cover_total_ppo_step_is_deterministic(self):
        levels = _levels_block(16, 4)
        a = _make_sampler(levels, 8, curriculum_schedule="gaussian", num_epochs=2)
        b = _make_sampler(levels, 8, curriculum_schedule="gaussian", num_epochs=2, seed=99)
        c = _make_sampler(levels, 8, curriculum_schedule="gaussian", num_epochs=1)
        self.assertEqual(a.steps_per_epoch, b.steps_per_epoch)
        self.assertEqual(a.steps_per_epoch, c.steps_per_epoch)
        self.assertEqual(hard_cover_total_ppo_step(a), a.steps_per_epoch * a.num_epochs)
        self.assertEqual(hard_cover_total_ppo_step(a), hard_cover_total_ppo_step(b))
        self.assertEqual(hard_cover_total_ppo_step(c), c.steps_per_epoch)
        hard_first = expected_cover_steps(a, a.num_tasks - 1)
        self.assertIsNotNone(hard_first)
        self.assertLessEqual(hard_first, a.steps_per_epoch)

    def test_gaussian_hard_cover_uses_endpoint_inclusive_schedule(self):
        # Endpoint-inclusive Gaussian can cover hard before a full dataset slot pass.
        k, n_per, gbs = 4, 16, 8
        levels = _levels_block(n_per, k)
        cover = _make_sampler(
            levels, gbs, curriculum_schedule="gaussian", epoch_from_hard_cover=True
        )
        self.assertEqual(cover.slot_steps_per_epoch, int(levels.shape[0]) // gbs)
        report = approx_cover_steps(
            np.bincount(levels),
            gbs,
            curriculum_schedule="gaussian",
            target_levels=[k - 1],
        )
        self.assertEqual(cover.steps_per_epoch, report["steps"])
        self.assertEqual(cover.step_metrics(cover.steps_per_epoch - 1)["e2h/t_frac"], 1.0)

    def test_all_level_cover_is_smallest_t_for_every_k(self):
        # False：每个 level 期望抽满；T 是自洽最短 horizon，且 >= hard-only。
        k, n_per, gbs = 4, 16, 8
        levels = _levels_block(n_per, k)
        all_cover = _make_sampler(
            levels, gbs, curriculum_schedule="gaussian", epoch_from_hard_cover=False
        )
        hard = _make_sampler(
            levels, gbs, curriculum_schedule="gaussian", epoch_from_hard_cover=True
        )
        report = approx_cover_steps(
            np.bincount(levels),
            gbs,
            curriculum_schedule="gaussian",
            target_levels=list(range(k)),
        )
        self.assertEqual(all_cover.steps_per_epoch, report["steps"])
        self.assertGreaterEqual(all_cover.steps_per_epoch, hard.steps_per_epoch)
        for lv in range(k):
            first = expected_cover_steps(all_cover, lv)
            self.assertIsNotNone(first)
            self.assertLessEqual(first, all_cover.steps_per_epoch)

    def test_resume_mid_second_epoch_matches_fresh(self):
        # 续训落到 epoch 1 中间，应等于 set_epoch(1) 后再 skip 同样的 PPO 步。
        k, n_per, gbs, dp, mbs = 4, 16, 8, 2, 1
        levels = _levels_block(n_per, k)
        skip_ppo = 2
        probe = _make_sampler(
            levels, gbs, dp_rank=0, dp_size=dp, rollout_mbs=mbs, num_epochs=2
        )
        s = probe.steps_per_epoch
        self.assertGreaterEqual(s, skip_ppo + 1)
        gas = gbs // (dp * mbs)
        resume_ppo = s + skip_ppo
        skip_batches = skip_ppo * gas
        for rank in range(dp):
            fresh = _make_sampler(
                levels, gbs, dp_rank=rank, dp_size=dp, rollout_mbs=mbs, num_epochs=2
            )
            fresh.set_epoch(1)
            fresh.start_index = 0
            epoch1 = _epoch_indices(fresh, mbs=mbs)
            resumed = _make_sampler(
                levels, gbs, dp_rank=rank, dp_size=dp, rollout_mbs=mbs, num_epochs=2
            )
            resumed.set_start_index(resume_ppo * gas, mbs)
            self.assertEqual(resumed.epoch, 1)
            self.assertEqual(resumed.start_index, skip_ppo * resumed.prompts_per_rank)
            self.assertEqual(_epoch_indices(resumed, mbs=mbs), epoch1[skip_batches:])

    def test_explicit_total_ppo_step_overrides_cover_horizon(self):
        levels = np.concatenate(
            [
                np.zeros(4480, dtype=np.int64),
                np.ones(1024, dtype=np.int64),
            ]
        )
        sampler = _make_sampler(
            levels,
            rollout_gbs=128,
            num_epochs=1,
            mu_exp=1.5,
            hard_end_prob=0.35,
            sigma=0.5,
            min_prob=False,
            epoch_from_hard_cover=False,
            total_ppo_step=39,
        )

        self.assertEqual(sampler.steps_per_epoch, 39)
        self.assertEqual(sampler.total_iterations, 39)
        self.assertEqual(sampler.horizon_source, "fixed_total_ppo_step")
        self.assertEqual(len(sampler), 39 * 128)
        self.assertAlmostEqual(sampler.mu_end, 0.345240197898, places=11)
        self.assertAlmostEqual(sampler.probs_at(0)[1], 0.1192029220)
        self.assertAlmostEqual(sampler.probs_at(38)[1], 0.35)
        self.assertEqual(sampler.step_metrics(38)["e2h/t_frac"], 1.0)
        self.assertEqual(sampler.step_metrics(38)["e2h/fixed_horizon"], 1.0)

    def test_explicit_total_ppo_step_must_split_evenly_across_epochs(self):
        with self.assertRaisesRegex(ValueError, "divisible by num_epochs"):
            _make_sampler(
                _levels_block(16, 2),
                rollout_gbs=8,
                num_epochs=2,
                total_ppo_step=39,
            )

    def test_fixed_horizon_resume_matches_fresh_order(self):
        levels = _levels_block(128, 2)
        full = _make_sampler(
            levels,
            rollout_gbs=16,
            rollout_mbs=1,
            num_epochs=1,
            total_ppo_step=39,
        )
        resumed = _make_sampler(
            levels,
            rollout_gbs=16,
            rollout_mbs=1,
            num_epochs=1,
            total_ppo_step=39,
        )
        resumed.set_start_index(7 * 16, 1)

        self.assertEqual(list(resumed), list(full)[7 * 16:])


class CollateLevelTest(unittest.TestCase):
    def test_collate_keeps_level(self):
        examples = [
            {"prompt_len": 2, "input_ids": [1, 2, 9], "gt_label": 1.0, "level": 0},
            {"prompt_len": 1, "input_ids": [3, 9], "gt_label": 2.0, "level": 3},
        ]
        batched = collate_fn(examples)
        self.assertEqual(batched["level"], [0, 3])
        self.assertEqual(batched["cache_keys"], ["level"])
        self.assertNotIn("level", collate_fn([
            {"prompt_len": 1, "input_ids": [1], "gt_label": 0.0},
        ]))


class CollectE2HMetricsTest(unittest.TestCase):
    def test_unique_prompt_idx_drops_dup_segments(self):
        levels = _levels_block(8, 2)
        sampler = _make_sampler(levels, rollout_gbs=4, curriculum_schedule="classic")
        rbs = [
            {
                "level": [0, 0, 0],
                "prompt_idx": [10, 10, 11],
            }
        ]
        uniq = levels_from_rollout_batches(rbs, unique_by_prompt_idx=True)
        self.assertEqual(uniq, [0, 0])
        all_lv = levels_from_rollout_batches(rbs, unique_by_prompt_idx=False)
        self.assertEqual(all_lv, [0, 0, 0])
        metrics = collect_e2h_rollout_metrics(sampler, ppo_step=0, rollout_batches=rbs)
        self.assertAlmostEqual(metrics["e2h/p_level_0"], 1.0)
        self.assertAlmostEqual(metrics["e2h/frac_level_0"], 1.0)
        self.assertAlmostEqual(metrics["e2h/frac_level_1"], 0.0)

    def test_reduce_counts_applies_before_frac(self):
        # unique [0, 1] → 本地 1:1；reduce 把 level1 乘 3 后 frac 为 1:3
        levels = _levels_block(8, 2)
        sampler = _make_sampler(levels, rollout_gbs=4, curriculum_schedule="classic")
        rbs = [{"level": [0, 1], "prompt_idx": [10, 11]}]

        def triple_level1(counts):
            out = np.array(counts, dtype=np.int64, copy=True)
            out[1] *= 3
            return out

        metrics = collect_e2h_rollout_metrics(
            sampler, 0, rbs, reduce_counts=triple_level1
        )
        self.assertAlmostEqual(metrics["e2h/frac_level_0"], 0.25)
        self.assertAlmostEqual(metrics["e2h/frac_level_1"], 0.75)

    def test_missing_level_raises(self):
        levels = _levels_block(8, 2)
        sampler = _make_sampler(levels, rollout_gbs=4, curriculum_schedule="classic")
        with self.assertRaises(KeyError):
            collect_e2h_rollout_metrics(
                sampler,
                ppo_step=0,
                rollout_batches=[{"prompt_idx": [0]}],
            )
