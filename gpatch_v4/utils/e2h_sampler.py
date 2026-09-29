from functools import partial
from typing import Callable, Iterator

import numpy as np
from torch.utils.data import Sampler


def remap_levels(raw_levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map raw difficulty labels to contiguous ``0 .. K-1`` in sorted order.

    Parameters
    ----------
    raw_levels : np.ndarray
        Per-sample labels; must be finite integer-valued and non-empty.

    Returns
    -------
    mapped : np.ndarray
        Shape ``[N]`` int64.
    unique_raw : np.ndarray
        Sorted unique raw labels; ``unique_raw[k]`` is the raw value of
        remapped id ``k``.
    """
    if raw_levels.size == 0:
        raise ValueError("E2H levels is empty")
    raw = np.asarray(raw_levels)
    if np.issubdtype(raw.dtype, np.floating):
        if not np.all(raw == np.floor(raw)):
            raise ValueError(f"E2H levels must be integer-valued, got {raw[:8]}")
        raw = raw.astype(np.int64)
    elif not np.issubdtype(raw.dtype, np.integer):
        try:
            raw = raw.astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"E2H levels must be integers, dtype={raw_levels.dtype}") from exc
    unique_raw = np.unique(raw)
    mapping = {int(v): i for i, v in enumerate(unique_raw.tolist())}
    mapped = np.array([mapping[int(v)] for v in raw], dtype=np.int64)
    return mapped, unique_raw.astype(np.int64)


def classic_schedule(t: int, T: int, num_tasks: int) -> dict[int, float]:
    active_task = min(int(t * num_tasks / T), num_tasks - 1)
    return dict(enumerate(np.eye(num_tasks)[active_task].tolist()))


def balanced_schedule(t: int, T: int, num_tasks: int) -> dict[int, float]:
    del t, T
    return {i: 1.0 / num_tasks for i in range(num_tasks)}


def cosine_schedule(t: int, T: int, num_tasks: int) -> dict[int, float]:
    total = num_tasks * (num_tasks + 1) / 2.0
    early = {i: (num_tasks - i) / total for i in range(num_tasks)}
    late = {i: (i + 1) / total for i in range(num_tasks)}
    alpha = 0.5 * (1 + np.cos(np.pi * t / T))
    probs = {i: alpha * early[i] + (1 - alpha) * late[i] for i in range(num_tasks)}
    p_min = 2 / (num_tasks * (num_tasks + 1))
    for i in range(num_tasks):
        probs[i] = max(probs[i], p_min)
    norm = sum(probs.values())
    return {i: probs[i] / norm for i in probs}


def _gaussian_min_prob(num_tasks: int, min_prob: bool | float) -> float:
    if min_prob is True:
        p_min = 2 / (num_tasks * (num_tasks + 1))
    elif min_prob is False:
        p_min = 0.0
    elif isinstance(min_prob, float):
        p_min = min_prob
    else:
        raise ValueError("min_prob should be either a boolean or a float")
    if p_min < 0:
        raise ValueError("min_prob must be >= 0")
    if num_tasks * p_min > 1:
        raise ValueError("num_tasks * p_min must not exceed 1")
    return float(p_min)


def _gaussian_probs_at_mu(
    mu: float,
    num_tasks: int,
    sigma: float,
    min_prob: bool | float,
) -> np.ndarray:
    if num_tasks <= 0:
        raise ValueError(f"num_tasks must be > 0, got {num_tasks}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    p_min = _gaussian_min_prob(num_tasks, min_prob)
    level_ids = np.arange(num_tasks, dtype=np.float64)
    base = np.exp(-((level_ids - float(mu)) ** 2) / (2 * sigma**2))
    q = base / base.sum()
    return p_min + (1 - num_tasks * p_min) * q


def gaussian_mu_end_from_hard_prob(
    num_tasks: int,
    sigma: float,
    min_prob: bool | float,
    hard_end_prob: float,
) -> float:
    """Solve the normalized Gaussian endpoint for a final hardest-level probability.

    The Gaussian center is constrained to the level axis ``[0, K-1]``.
    A target outside the probabilities attainable on that interval is rejected.
    """
    if num_tasks <= 0:
        raise ValueError(f"num_tasks must be > 0, got {num_tasks}")
    if isinstance(hard_end_prob, bool) or not 0.0 < hard_end_prob <= 1.0:
        raise ValueError(
            f"hard_end_prob must be in (0, 1], got {hard_end_prob}"
        )
    if num_tasks == 1:
        if np.isclose(hard_end_prob, 1.0, rtol=0.0, atol=1e-12):
            return 1.0
        raise ValueError(
            "hard_end_prob must be 1.0 when the dataset has only one level"
        )

    axis_end = float(num_tasks - 1)
    min_hard_prob = float(
        _gaussian_probs_at_mu(0.0, num_tasks, sigma, min_prob)[-1]
    )
    max_hard_prob = float(
        _gaussian_probs_at_mu(axis_end, num_tasks, sigma, min_prob)[-1]
    )
    tolerance = 1e-12
    if (
        hard_end_prob < min_hard_prob - tolerance
        or hard_end_prob > max_hard_prob + tolerance
    ):
        raise ValueError(
            f"hard_end_prob={hard_end_prob} is unreachable for "
            f"num_tasks={num_tasks}, sigma={sigma}, min_prob={min_prob}; "
            f"attainable range is [{min_hard_prob:.12g}, {max_hard_prob:.12g}]"
        )
    if np.isclose(hard_end_prob, min_hard_prob, rtol=0.0, atol=tolerance):
        return 0.0
    if np.isclose(hard_end_prob, max_hard_prob, rtol=0.0, atol=tolerance):
        return 1.0

    lo, hi = 0.0, axis_end
    for _ in range(80):
        mid = (lo + hi) / 2.0
        mid_prob = float(
            _gaussian_probs_at_mu(mid, num_tasks, sigma, min_prob)[-1]
        )
        if mid_prob < hard_end_prob:
            lo = mid
        else:
            hi = mid
    return ((lo + hi) / 2.0) / axis_end


def gaussian_schedule(
    t: int,
    T: int,
    num_tasks: int,
    mu_exp: float,
    sigma: float,
    min_prob: bool | float,
    mu_end: float = 1.0,
    **kwargs,
) -> dict[int, float]:
    del kwargs
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if not 0.0 <= mu_end <= 1.0:
        raise ValueError(f"mu_end must be in [0, 1], got {mu_end}")
    # PPO steps are indexed 0..T-1. Endpoint-inclusive progress ensures that
    # the final scheduled step reaches exactly ``mu_end``.
    progress = 0.0 if T == 1 else min(max(float(t) / float(T - 1), 0.0), 1.0)
    mu = progress**mu_exp * mu_end * (num_tasks - 1)
    probs = _gaussian_probs_at_mu(mu, num_tasks, sigma, min_prob)
    return dict(enumerate(probs.tolist()))


SCHEDULE_FUNCS_FACTORY = {
    "classic": classic_schedule,
    "balanced": balanced_schedule,
    "cosine": cosine_schedule,
    "gaussian": gaussian_schedule,
}
E2H_SCHEDULES = tuple(SCHEDULE_FUNCS_FACTORY)


def _schedule_func(
    curriculum_schedule: str,
    mu_exp: float,
    sigma: float,
    min_prob: bool | float,
    mu_end: float = 1.0,
) -> Callable[..., dict[int, float]]:
    try:
        fn = SCHEDULE_FUNCS_FACTORY[curriculum_schedule]
    except KeyError as exc:
        raise ValueError(
            f"unknown E2H curriculum_schedule={curriculum_schedule!r}, "
            f"expected one of {E2H_SCHEDULES}"
        ) from exc
    if curriculum_schedule == "gaussian":
        return partial(
            fn,
            mu_exp=mu_exp,
            sigma=sigma,
            min_prob=min_prob,
            mu_end=mu_end,
        )
    return fn


def probs_to_array(probs: dict[int, float], num_tasks: int) -> np.ndarray:
    arr = np.array([probs[i] for i in range(num_tasks)], dtype=np.float64)
    s = arr.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"invalid E2H schedule probabilities: {probs}")
    return arr / s


def calc_level_counts(levels, num_tasks: int) -> np.ndarray:
    """Per-level histogram of remapped ids. Empty ``levels`` is all zeros.

    Parameters
    ----------
    levels : sequence of int
        One entry per remaining sample after generate (and optional post-filter).
        May be empty on a rank that dropped every row.
    num_tasks : int
        Remapped ``K``; output length.

    Returns
    -------
    np.ndarray
        Shape ``[K]`` int64.

    Raises
    ------
    ValueError
        ``num_tasks <= 0``, or any value outside ``[0, num_tasks)``.
    """
    if num_tasks <= 0:
        raise ValueError(f"num_tasks must be > 0, got {num_tasks}")
    chosen = np.asarray([int(x) for x in levels], dtype=np.int64)
    if chosen.size > 0 and (np.any(chosen < 0) or np.any(chosen >= num_tasks)):
        raise ValueError(f"level out of range [0, {num_tasks}): unique={np.unique(chosen)}")
    return np.bincount(chosen, minlength=num_tasks).astype(np.int64, copy=False)


def frac_metrics_from_counts(counts: np.ndarray) -> dict[str, float]:
    """Convert a per-level histogram into ``e2h/frac_level_k``.

    Parameters
    ----------
    counts : np.ndarray
        Shape ``[K]``; non-negative. Caller all-reduces rank-local
        ``calc_level_counts`` first so the shares are the global batch.

    Returns
    -------
    dict[str, float]
        ``e2h/frac_level_k`` in ``[0, 1]``.

    Raises
    ------
    ValueError
        Not 1-d, any negative entry, or total count is 0.
    """
    counts = np.asarray(counts)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError(f"counts must be 1-d and non-empty, got {counts.shape}")
    if np.any(counts < 0):
        raise ValueError(f"counts must be non-negative, got {counts}")
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("E2H frac metrics need a non-empty level list")
    return {f"e2h/frac_level_{k}": float(counts[k]) / total for k in range(counts.size)}


def frac_metrics_from_levels(levels, num_tasks: int) -> dict[str, float]:
    """Rank-local empirical share of each remapped level.

    For the logged global metric, all-reduce ``calc_level_counts`` across DP
    then call ``frac_metrics_from_counts``.

    Parameters
    ----------
    levels : sequence of int
        One entry per remaining sample after generate (and optional post-filter).
        Repeats of the same prompt share a level, so the mean equals unique-prompt
        frac unless rows were dropped unevenly.
    num_tasks : int
        Remapped ``K``; keys are ``e2h/frac_level_0 .. K-1``.

    Returns
    -------
    dict[str, float]
        ``e2h/frac_level_k`` in ``[0, 1]``.

    Raises
    ------
    ValueError
        Empty ``levels``, or any value outside ``[0, num_tasks)``.
    """
    return frac_metrics_from_counts(calc_level_counts(levels, num_tasks))


class E2HTaskSampler(Sampler[int]):
    """DP-synchronous easy-to-hard prompt sampler.

    Every rank holds the full level vector. In-epoch step ``t`` draws
    ``rollout_gbs`` levels from ``P_t``, pops the next index from each
    level bucket, then rank ``r`` keeps ``indices[r::dp_size]``. ``P_t``
    is constant inside a PPO step. Each epoch replays the curriculum
    with ``T = steps_per_epoch`` and a new bucket permutation
    (``SeedSequence([seed, epoch])``). Resume is ``set_epoch`` +
    ``set_start_index`` (same contract as ``ResumableDistributedSampler``).

    Parameters
    ----------
    levels : np.ndarray
        Remapped per-sample ids, values in ``0 .. K-1``, length ``N``.
    rollout_gbs : int
        Global unique-prompt count per PPO step. Must be divisible by
        ``dp_size``.
    rollout_mbs : int
    dp_rank : int
    dp_size : int
    seed : int
    num_epochs : int
        Times to replay one curriculum epoch. Does not stretch ``P_t``.
    curriculum_schedule : str
        ``classic`` / ``balanced`` / ``cosine`` / ``gaussian``.
    mu_exp : float
        Gaussian mean exponent (paper ``beta``). Ignored by other schedules.
    sigma : float
        Gaussian std. Ignored by other schedules.
    min_prob : bool or float
        Gaussian floor. ``True`` uses ``2 / (K (K+1))``.
    mu_end : float, optional
        Advanced Gaussian endpoint as a fraction of the level axis. Defaults
        to 1.0 when neither endpoint option is configured.
    hard_end_prob : float, optional
        Desired hardest-level probability on the final Gaussian step. The
        sampler derives ``mu_end`` from it.
    epoch_from_hard_cover : bool
        If True, ``steps_per_epoch`` is the smallest ``T`` whose expected
        hard unique-cover finishes inside that ``T``. If False, the
        smallest ``T`` such that every remapped level's expected draws
        reach that level's bucket.
    total_ppo_step : int, optional
        Explicit total run horizon. When set, it overrides cover-based
        horizon selection and must divide evenly across ``num_epochs``.
    """
    def __init__(
        self,
        levels: np.ndarray,
        rollout_gbs: int,
        rollout_mbs: int,
        dp_rank: int,
        dp_size: int,
        seed: int,
        num_epochs: int,
        curriculum_schedule: str,
        mu_exp: float,
        sigma: float,
        min_prob: bool | float,
        epoch_from_hard_cover: bool = True,
        mu_end: float | None = None,
        total_ppo_step: int | None = None,
        hard_end_prob: float | None = None,
    ):
        levels = np.asarray(levels, dtype=np.int64)
        if levels.ndim != 1:
            raise ValueError(f"levels must be 1-d, got shape {levels.shape}")
        if dp_size <= 0 or dp_rank < 0 or dp_rank >= dp_size:
            raise ValueError(f"invalid dp rank/size: {dp_rank=} {dp_size=}")
        if rollout_gbs <= 0 or rollout_mbs <= 0:
            raise ValueError(f"invalid rollout sizes: {rollout_gbs=} {rollout_mbs=}")
        if rollout_gbs % dp_size != 0:
            raise ValueError(f"rollout_gbs={rollout_gbs} must be divisible by dp_size={dp_size}")
        if num_epochs <= 0:
            raise ValueError(f"num_epochs must be > 0, got {num_epochs}")

        self.levels = levels
        self.num_tasks = int(levels.max()) + 1 if levels.size > 0 else 0
        if self.num_tasks <= 0:
            raise ValueError("E2H levels is empty")
        if int(levels.min()
              ) != 0 or not np.array_equal(np.unique(levels), np.arange(self.num_tasks)):
            raise ValueError(
                "E2H levels must be remapped to contiguous 0..K-1, "
                f"got unique={np.unique(levels)}"
            )

        self.rollout_gbs = int(rollout_gbs)
        self.rollout_mbs = int(rollout_mbs)
        self.rank = int(dp_rank)
        self.num_replicas = int(dp_size)
        self.seed = int(seed)
        self.num_epochs = int(num_epochs)
        self.curriculum_schedule = curriculum_schedule
        self.mu_exp = float(mu_exp)
        self.sigma = float(sigma)
        self.min_prob = min_prob
        if hard_end_prob is not None:
            if curriculum_schedule != "gaussian":
                raise ValueError(
                    "hard_end_prob is only valid for the gaussian schedule"
                )
            if mu_end is not None:
                raise ValueError(
                    "mu_end and hard_end_prob are mutually exclusive"
                )
            self.mu_end = gaussian_mu_end_from_hard_prob(
                self.num_tasks,
                self.sigma,
                self.min_prob,
                hard_end_prob,
            )
            self.hard_end_prob = float(hard_end_prob)
        else:
            self.mu_end = 1.0 if mu_end is None else float(mu_end)
            self.hard_end_prob = None
        self.epoch_from_hard_cover = bool(epoch_from_hard_cover)
        self.fixed_total_ppo_step = (
            None if total_ppo_step is None else int(total_ppo_step)
        )
        self.schedule_func = _schedule_func(
            curriculum_schedule,
            mu_exp,
            sigma,
            min_prob,
            self.mu_end,
        )

        n = int(levels.shape[0])
        if n < self.rollout_gbs:
            raise ValueError(
                f"dataset size {n} < rollout_gbs {self.rollout_gbs}; "
                "E2H cannot form one PPO step"
            )
        # drop_last: leftover unique prompts do not form a short last PPO step.
        self.num_dropped_tail = n % self.rollout_gbs
        self.slot_steps_per_epoch = n // self.rollout_gbs
        if self.slot_steps_per_epoch <= 0:
            raise ValueError("E2H slot_steps_per_epoch is 0")
        n_by_level = np.bincount(levels, minlength=self.num_tasks)
        if self.fixed_total_ppo_step is not None:
            if self.fixed_total_ppo_step <= 0:
                raise ValueError(
                    "total_ppo_step must be > 0 when explicitly configured, "
                    f"got {self.fixed_total_ppo_step}"
                )
            if self.fixed_total_ppo_step % self.num_epochs != 0:
                raise ValueError(
                    "total_ppo_step must be divisible by num_epochs for E2H: "
                    f"{self.fixed_total_ppo_step} % {self.num_epochs} != 0"
                )
            self.steps_per_epoch = self.fixed_total_ppo_step // self.num_epochs
            self.horizon_source = "fixed_total_ppo_step"
        else:
            if self.epoch_from_hard_cover:
                targets = [self.num_tasks - 1]
            else:
                targets = list(range(self.num_tasks))
            report = approx_cover_steps(
                n_by_level,
                self.rollout_gbs,
                curriculum_schedule,
                mu_exp,
                sigma,
                min_prob,
                mu_end=self.mu_end,
                target_levels=targets,
            )
            if report["steps"] is None:
                kind = "hard" if self.epoch_from_hard_cover else "all-level"
                raise ValueError(
                    f"E2H expected {kind} cover has no finite epoch length; "
                    "check curriculum_schedule / n_by_level vs min_prob"
                )
            self.steps_per_epoch = int(report["steps"])
            self.horizon_source = (
                "hard_cover"
                if self.epoch_from_hard_cover
                else "all_level_cover"
            )
        # One epoch = one curriculum pass; num_epochs only repeats it.
        self.total_iterations = self.steps_per_epoch
        self.prompts_per_rank = self.rollout_gbs // self.num_replicas
        if self.prompts_per_rank % self.rollout_mbs != 0:
            raise ValueError(
                f"prompts_per_rank={self.prompts_per_rank} must be divisible by "
                f"rollout_mbs={self.rollout_mbs}"
            )

        self.epoch = 0
        self.start_index = 0
        self.indices_by_task = {}
        self._permute_buckets()

    def _permute_buckets(self) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        for k in range(self.num_tasks):
            bucket = np.where(self.levels == k)[0]
            if bucket.size == 0:
                raise ValueError(f"E2H level {k} has no samples")
            self.indices_by_task[k] = rng.permutation(bucket)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError(f"epoch must be >= 0, got {epoch}")
        self.epoch = int(epoch)
        self._permute_buckets()

    def set_start_index(self, start_step: int, batch_size: int) -> None:
        """Skip consumed rank-local microbatches. ``start_step`` is in
        microbatch units (``resume_ppo_step * rollout_gas``).
        """
        if batch_size != self.rollout_mbs:
            raise ValueError(
                f"set_start_index batch_size={batch_size} != rollout_mbs={self.rollout_mbs}"
            )
        samples_per_epoch = self.steps_per_epoch * self.prompts_per_rank
        steps_per_epoch = samples_per_epoch // batch_size
        if steps_per_epoch <= 0:
            self.start_index = 0
            return
        completed_epochs = start_step // steps_per_epoch
        self.set_epoch(completed_epochs)
        self.start_index = (start_step % steps_per_epoch) * batch_size

    def __len__(self) -> int:
        return self.steps_per_epoch * self.prompts_per_rank - self.start_index

    def _t_in_epoch(self, t: int) -> int:
        t = int(t)
        if t < 0:
            raise ValueError(f"t must be >= 0, got {t}")
        return t % self.steps_per_epoch

    def probs_at(self, t: int) -> np.ndarray:
        t_local = self._t_in_epoch(t)
        probs = self.schedule_func(t_local, self.total_iterations, self.num_tasks)
        return probs_to_array(probs, self.num_tasks)

    def step_metrics(self, t: int) -> dict[str, float]:
        t_local = self._t_in_epoch(t)
        probs = self.probs_at(t_local)
        metrics = {f"e2h/p_level_{k}": float(probs[k]) for k in range(self.num_tasks)}
        if self.curriculum_schedule == "gaussian":
            metrics["e2h/t_frac"] = (
                0.0
                if self.total_iterations == 1
                else float(t_local) / float(self.total_iterations - 1)
            )
            metrics["e2h/mu"] = (
                metrics["e2h/t_frac"] ** self.mu_exp
                * self.mu_end
                * (self.num_tasks - 1)
            )
        else:
            metrics["e2h/t_frac"] = float(t_local) / float(self.total_iterations)
        metrics["e2h/fixed_horizon"] = float(
            self.fixed_total_ppo_step is not None
        )
        return metrics

    def _rng_for_step(self, t: int) -> np.random.Generator:
        ss = np.random.SeedSequence([self.seed, self.epoch, self._t_in_epoch(t)])
        return np.random.default_rng(ss)

    def _draw_global_indices(
        self,
        t: int,
        task_ptrs: dict[int, int],
    ) -> tuple[list[int], np.ndarray]:
        rng = self._rng_for_step(t)
        probs = self.probs_at(t)
        chosen = rng.choice(np.arange(self.num_tasks), size=self.rollout_gbs, p=probs, replace=True)
        batch_indices = []
        for task in chosen:
            k = int(task)
            bucket = self.indices_by_task[k]
            p = task_ptrs[k] % len(bucket)
            batch_indices.append(int(bucket[p]))
            task_ptrs[k] += 1
        return batch_indices, chosen

    def __iter__(self) -> Iterator[int]:
        if self.start_index % self.prompts_per_rank != 0:
            raise ValueError(
                f"start_index={self.start_index} is not aligned to "
                f"prompts_per_rank={self.prompts_per_rank}"
            )
        skip_steps = self.start_index // self.prompts_per_rank
        t_start = skip_steps
        t_end = self.steps_per_epoch
        if t_start > t_end:
            raise ValueError(f"E2H resume step {t_start} exceeds epoch end {t_end}")

        task_ptrs = {k: 0 for k in range(self.num_tasks)}
        for t in range(t_start):
            self._draw_global_indices(t, task_ptrs)

        for t in range(t_start, t_end):
            global_indices, _chosen = self._draw_global_indices(t, task_ptrs)
            rank_indices = global_indices[self.rank::self.num_replicas]
            if len(rank_indices) != self.prompts_per_rank:
                raise ValueError(
                    f"rank slice len {len(rank_indices)} != prompts_per_rank "
                    f"{self.prompts_per_rank}"
                )
            yield from rank_indices


def _cover_within_horizon(
    counts: np.ndarray,
    rollout_gbs: int,
    total_iterations: int,
    schedule_func,
    targets: list[int],
) -> dict[int, int | None]:
    num_tasks = int(counts.shape[0])
    acc = {k: 0.0 for k in targets}
    first: dict[int, int | None] = {k: None for k in targets}
    for t in range(total_iterations):
        probs = probs_to_array(schedule_func(t, total_iterations, num_tasks), num_tasks)
        for k in targets:
            if first[k] is None:
                acc[k] += rollout_gbs * float(probs[k])
                if acc[k] >= float(counts[k]):
                    first[k] = t + 1
        if all(v is not None for v in first.values()):
            break
    return first


def _smallest_cover_horizon(
    counts: np.ndarray,
    rollout_gbs: int,
    schedule_func,
    targets: list[int],
) -> int | None:
    n_need = max(int(counts[k]) for k in targets)
    lo = max(1, (n_need + rollout_gbs - 1) // rollout_gbs)
    slot = max(1, int(counts.sum()) // rollout_gbs)
    max_t = max(lo, slot) * 4096

    def _ok(horizon: int) -> bool:
        first = _cover_within_horizon(counts, rollout_gbs, horizon, schedule_func, targets)
        return all(v is not None for v in first.values())

    hi = lo
    while not _ok(hi):
        if hi >= max_t:
            return None
        doubled = hi * 2
        hi = max_t if doubled > max_t or doubled <= hi else doubled

    left, right = lo, hi
    while left < right:
        mid = (left + right) // 2
        if _ok(mid):
            right = mid
        else:
            left = mid + 1
    return left


def approx_cover_steps(
    n_by_level,
    rollout_gbs: int,
    curriculum_schedule: str,
    mu_exp: float = 0.5,
    sigma: float = 0.5,
    min_prob: bool | float = True,
    target_levels: list[int] | None = None,
    total_iterations: int | None = None,
    mu_end: float = 1.0,
) -> dict:
    """Approximate unique-cover PPO steps from schedule + per-level counts.

    Expected-draw estimate ``B * sum_t P_t(k; T) >= n_k``, not seed-exact.
    ``P_t`` uses horizon ``T``. If ``total_iterations`` is None, ``T`` is the
    smallest horizon whose cover finishes inside that horizon (one-epoch
    curriculum length).

    Parameters
    ----------
    n_by_level : array-like of int
        Unique-prompt counts for remapped levels ``0 .. K-1``.
    rollout_gbs : int
    curriculum_schedule : str
    mu_exp, sigma, min_prob, mu_end
        Gaussian knobs; ignored by other schedules.
    target_levels : list of int, optional
        Default the hardest id ``K-1``.
    total_iterations : int, optional
        Schedule horizon ``T``. ``None`` searches for the smallest such ``T``.

    Returns
    -------
    dict
        ``steps`` is the max expected cover-count across targets, or ``None``
        if some target is not reached inside ``T``.
    """
    counts = np.asarray(n_by_level, dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError(f"n_by_level must be 1-d and non-empty, got {counts.shape}")
    if np.any(counts <= 0):
        raise ValueError(f"every level must have samples, got {counts}")
    num_tasks = int(counts.shape[0])
    n_total = int(counts.sum())
    if rollout_gbs <= 0:
        raise ValueError(f"rollout_gbs must be > 0, got {rollout_gbs}")
    if n_total < rollout_gbs:
        raise ValueError(f"dataset size {n_total} < rollout_gbs {rollout_gbs}")
    if target_levels is None:
        targets = [num_tasks - 1]
    else:
        targets = [int(x) for x in target_levels]
    if not targets:
        raise ValueError("target_levels is empty")
    for k in targets:
        if k < 0 or k >= num_tasks:
            raise ValueError(f"target level {k} out of range [0, {num_tasks})")
    if total_iterations is not None and int(total_iterations) <= 0:
        raise ValueError(f"total_iterations must be > 0, got {total_iterations}")

    schedule_func = _schedule_func(
        curriculum_schedule,
        mu_exp,
        sigma,
        min_prob,
        mu_end,
    )
    slot_steps = n_total // rollout_gbs
    if total_iterations is None:
        horizon = _smallest_cover_horizon(counts, rollout_gbs, schedule_func, targets)
        if horizon is None:
            n_map = {k: int(counts[k]) for k in range(num_tasks)}
            return {
                "steps": None,
                "expected_steps": {
                    k: None
                    for k in targets
                },
                "n_by_level": n_map,
                "steps_per_epoch": slot_steps,
                "total_iterations": None,
                "curriculum_schedule": curriculum_schedule,
                "within_scheduled_T": False,
            }
        first = _cover_within_horizon(counts, rollout_gbs, horizon, schedule_func, targets)
        n_map = {k: int(counts[k]) for k in range(num_tasks)}
        return {
            "steps": horizon,
            "expected_steps": dict(first),
            "n_by_level": n_map,
            "steps_per_epoch": slot_steps,
            "total_iterations": horizon,
            "curriculum_schedule": curriculum_schedule,
            "within_scheduled_T": True,
        }

    first = _cover_within_horizon(
        counts, rollout_gbs, int(total_iterations), schedule_func, targets
    )
    finished = all(v is not None for v in first.values())
    n_map = {k: int(counts[k]) for k in range(num_tasks)}
    return {
        "steps": max(first.values()) if finished else None,
        "expected_steps": dict(first),
        "n_by_level": n_map,
        "steps_per_epoch": slot_steps,
        "total_iterations": int(total_iterations),
        "curriculum_schedule": curriculum_schedule,
        "within_scheduled_T": finished,
    }


def expected_cover_steps(sampler: E2HTaskSampler, level: int) -> int | None:
    """Smallest 1-based in-epoch PPO-step whose expected draws of ``level`` reach the bucket.

    Parameters
    ----------
    sampler : E2HTaskSampler
    level : int
        Remapped id in ``0 .. K-1``.

    Returns
    -------
    int or None
        ``None`` if the sum over one-epoch ``T`` is still below the bucket size.
    """
    n_by_level = [len(sampler.indices_by_task[k]) for k in range(sampler.num_tasks)]
    report = approx_cover_steps(
        n_by_level,
        sampler.rollout_gbs,
        sampler.curriculum_schedule,
        sampler.mu_exp,
        sampler.sigma,
        sampler.min_prob,
        target_levels=[level],
        total_iterations=sampler.total_iterations,
        mu_end=sampler.mu_end,
    )
    return report["expected_steps"][level]


def hard_cover_total_ppo_step(sampler: E2HTaskSampler) -> int:
    """Run length: one-epoch hard-cover steps times ``num_epochs``.

    Parameters
    ----------
    sampler : E2HTaskSampler

    Returns
    -------
    int
        ``steps_per_epoch * num_epochs``.
    """
    return int(sampler.steps_per_epoch) * int(sampler.num_epochs)


def simulate_unique_cover(
    sampler: E2HTaskSampler,
    target_levels: list[int] | None = None,
    max_steps: int | None = None,
) -> dict:
    """Replay the train RNG until each target level's bucket is consumed once.

    ``steps`` is 1-based: coverage completes at the end of 0-indexed step
    ``t`` → ``steps = t + 1``. This is exact for ``sampler.seed``. It is not
    an almost-sure maximum: if ``P_t(k) < 1``, some seeds never finish.

    Parameters
    ----------
    sampler : E2HTaskSampler
    target_levels : list of int, optional
        Default the hardest remapped id ``K-1``.
    max_steps : int, optional
        Replay cap; default ``total_iterations``.

    Returns
    -------
    dict
        ``steps`` : int or None
            Max cover-count across targets, or ``None`` if unfinished.
        ``first_cover_steps`` : dict[int, int | None]
        ``expected_steps`` : dict[int, int | None]
        ``n_by_level`` : dict[int, int]
        ``steps_per_epoch`` / ``total_iterations`` : int
    """
    if target_levels is None:
        targets = [sampler.num_tasks - 1]
    else:
        targets = [int(x) for x in target_levels]
    if not targets:
        raise ValueError("target_levels is empty")
    for k in targets:
        if k < 0 or k >= sampler.num_tasks:
            raise ValueError(f"target level {k} out of range [0, {sampler.num_tasks})")

    n_by_level = {k: int(len(sampler.indices_by_task[k])) for k in range(sampler.num_tasks)}
    horizon = sampler.total_iterations if max_steps is None else int(max_steps)
    if horizon < 0:
        raise ValueError(f"max_steps must be >= 0, got {horizon}")
    horizon = min(horizon, sampler.total_iterations)

    task_ptrs = {k: 0 for k in range(sampler.num_tasks)}
    first_cover_steps: dict[int, int | None] = {k: None for k in targets}
    for t in range(horizon):
        sampler._draw_global_indices(t, task_ptrs)
        for k in targets:
            if first_cover_steps[k] is None and task_ptrs[k] >= n_by_level[k]:
                first_cover_steps[k] = t + 1
        if all(v is not None for v in first_cover_steps.values()):
            break

    finished = all(v is not None for v in first_cover_steps.values())
    return {
        "steps": max(first_cover_steps.values()) if finished else None,
        "first_cover_steps": first_cover_steps,
        "expected_steps": {
            k: expected_cover_steps(sampler, k)
            for k in targets
        },
        "n_by_level": n_by_level,
        "steps_per_epoch": sampler.steps_per_epoch,
        "total_iterations": sampler.total_iterations,
        "seed": sampler.seed,
        "curriculum_schedule": sampler.curriculum_schedule,
    }


def levels_from_rollout_batches(
    rollout_batches: list[dict],
    unique_by_prompt_idx: bool = True,
) -> list[int]:
    """Collect remapped ``level`` values from field-major rollout batches.

    Parameters
    ----------
    rollout_batches : list of dict
        Field-major batches after cache restore. Each must contain ``level``.
    unique_by_prompt_idx : bool
        If True and ``prompt_idx`` is present, keep one level per prompt so
        DupTraj segments and ``sampling_repeat_n`` copies do not inflate
        ``frac_level_*``.

    Returns
    -------
    list of int
        Levels in batch order.

    Raises
    ------
    KeyError
        A batch is missing ``level``.
    ValueError
        No levels remain after unique-ing.
    """
    if not rollout_batches:
        raise ValueError("E2H frac metrics need a non-empty level list")
    collected: list[int] = []
    seen_prompt: set[int] = set()
    for rb in rollout_batches:
        if "level" not in rb:
            raise KeyError(
                "E2H sampler is enabled but rollout batch has no 'level'; "
                "collate must put level in cache_keys"
            )
        levels = rb["level"]
        if unique_by_prompt_idx and "prompt_idx" in rb:
            prompt_idx = rb["prompt_idx"]
            if len(prompt_idx) != len(levels):
                raise ValueError(f"prompt_idx len {len(prompt_idx)} != level len {len(levels)}")
            for pidx, lv in zip(prompt_idx, levels):
                pidx_i = int(pidx)
                if pidx_i in seen_prompt:
                    continue
                seen_prompt.add(pidx_i)
                collected.append(int(lv))
        else:
            collected.extend(int(lv) for lv in levels)
    if not collected:
        raise ValueError("E2H frac metrics need a non-empty level list")
    return collected


def collect_e2h_rollout_metrics(
    sampler: E2HTaskSampler,
    ppo_step: int,
    rollout_batches: list[dict],
    unique_by_prompt_idx: bool = True,
    reduce_counts: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, float]:
    """Build ``e2h/p_level_*``, ``e2h/t_frac``, and ``e2h/frac_level_*``.

    Parameters
    ----------
    sampler : E2HTaskSampler
    ppo_step : int
    rollout_batches : list of dict
        Field-major batches after cache restore.
    unique_by_prompt_idx : bool
        If True and ``prompt_idx`` is present, count each prompt once so
        DupTraj / ``sampling_repeat_n`` copies do not inflate ``frac_*``.
    reduce_counts : callable or None
        Optional map over the rank-local histogram before ``frac_level_*``.
        Multi-controller passes DP all-reduce; last-rank wandb would
        otherwise log a local mix.

    Returns
    -------
    dict of str to float
    """
    metrics = sampler.step_metrics(ppo_step)
    level_values = levels_from_rollout_batches(
        rollout_batches,
        unique_by_prompt_idx=unique_by_prompt_idx,
    )
    counts = calc_level_counts(level_values, sampler.num_tasks)
    if reduce_counts is not None:
        counts = reduce_counts(counts)
    metrics.update(frac_metrics_from_counts(counts))
    return metrics
