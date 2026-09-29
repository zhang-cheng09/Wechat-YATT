"""Regex-based per-parameter learning-rate helpers.

Config dataclasses stay in ``gpatch_v4.configs.optimizer_config``.
"""
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from megatron.core.optimizer import ParamKey, ParamWithNamePredicate
except ImportError:
    ParamKey = None
    ParamWithNamePredicate = None

from gpatch_v4.utils.common_utils import log

GCORE_LR_GROUP_KEY = "gcore_lr_group"
DEFAULT_LR_GROUP_NAME = "default"
# Megatron load ckpt 用 (wd_mult, lr_mult, is_expert_parallel,
# is_decoupled_lr) 匹配 param group。named 组不能和默认组共用
# lr_mult=1.0。scheduler 不乘 lr_mult，只用 max_lr/min_lr。
_NAMED_GROUP_LR_MULT_STEP = 1e-3


def strip_module_prefix(name: str) -> str:
    """Strip leading DDP ``module.`` prefixes from a parameter name."""
    while name.startswith("module."):
        name = name[len("module."):]
    return name


def param_name_matches_regex(param_name: str, pattern: str) -> bool:
    """Return True if ``pattern`` matches ``param_name`` or its unwrapped form.

    Matching uses ``re.search``. The DDP ``module.`` prefix is stripped so
    the same pattern works before and after wrapping.
    """
    compiled = re.compile(pattern)
    if compiled.search(param_name) is not None:
        return True
    stripped = strip_module_prefix(param_name)
    return stripped != param_name and compiled.search(stripped) is not None


def lr_group_override(
    lr: float,
    min_lr: Optional[float],
    group_name: str,
    lr_mult: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a Megatron ParamGroupOverride dict for one named LR group."""
    override = {
        "max_lr": float(lr),
        GCORE_LR_GROUP_KEY: group_name,
    }
    if min_lr is not None:
        override["min_lr"] = float(min_lr)
    if lr_mult is not None:
        override["lr_mult"] = float(lr_mult)
    return override


def _make_param_lr_predicate(entries: List, index: int):
    """Match ``entries[index]`` and skip parameters claimed by earlier entries."""

    def _fn(param, param_name, _entries=entries, _index=index):
        del param
        for earlier in _entries[:_index]:
            if param_name_matches_regex(param_name, earlier.pattern):
                return False
        return param_name_matches_regex(param_name, _entries[_index].pattern)

    return _fn


def build_param_lr_config_overrides(param_lr_overrides: Optional[Iterable[Any]]) -> Dict:
    """Build Megatron ``config_overrides`` from ``param_lr_overrides``."""
    if ParamKey is None or ParamWithNamePredicate is None:
        if param_lr_overrides:
            raise RuntimeError(
                "optimizer.param_lr_overrides requires megatron.core ParamKey / "
                "ParamWithNamePredicate (Megatron > 0.15)"
            )
        return {}

    if not param_lr_overrides:
        return {}
    # yaml 列表顺序：先匹配先生效；lr_mult 也按这个 idx 赋。
    entries = list(param_lr_overrides)
    config_overrides = {}
    for idx, entry in enumerate(entries):
        predicate = ParamWithNamePredicate(
            name=f"gcore_param_lr_{entry.name}",
            fn=_make_param_lr_predicate(entries, idx),
        )
        # lr_mult 只当 Megatron load ckpt 时 param group 的身份证
        # (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr)，
        # scheduler 不算它，只用 max_lr/min_lr。
        # 默认组已经是 1.0，named 组必须错开，否则 resume 会对到 default 组上。
        config_overrides[ParamKey(with_name_predicate=predicate)] = lr_group_override(
            entry.lr,
            entry.min_lr,
            entry.name,
            lr_mult=1.0 + (idx + 1) * _NAMED_GROUP_LR_MULT_STEP,
        )
    return config_overrides


def _wd_mult(group: dict) -> float:
    if "wd_mult" in group:
        return float(group["wd_mult"])
    if "pre_wd_mult" in group:
        return float(group["pre_wd_mult"])
    return 1.0


def _lrs_all_close(values: Iterable[float]) -> bool:
    iterator = iter(values)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(math.isclose(first, lr, rel_tol=1e-9, abs_tol=0.0) for lr in iterator)


def _param_group_base_name(group: dict, index: int) -> str:
    name = group.get(GCORE_LR_GROUP_KEY)
    if name:
        return name
    if group.get("default_config", True):
        return DEFAULT_LR_GROUP_NAME
    return f"group{index}"


def lr_group_name_for_param_name(param_name: str, param_lr_overrides) -> str:
    """Yaml group name for one parameter, first matching override wins."""
    if not param_lr_overrides:
        return DEFAULT_LR_GROUP_NAME
    for entry in param_lr_overrides:
        if param_name_matches_regex(param_name, entry.pattern):
            return entry.name
    return DEFAULT_LR_GROUP_NAME


def param_group_log_name(group: dict, index: int, used: Optional[Set[str]] = None) -> str:
    """Stable unique name for one optimizer param group.

    Megatron splits a named LR override by expert-parallel and weight-decay
    flags. Optimizer dumps keep those suffixes so two physical groups stay
    distinguishable. Wandb keys go through ``get_megatron_param_group_lrs``,
    which collapses splits that share the same scheduled ``lr``.

    Parameters
    ----------
    group : dict
        Optimizer param group.
    index : int
        Position in ``optimizer.param_groups``.
    used : set of str, optional
        Names already claimed. On collision, ``_{index}`` is appended.

    Returns
    -------
    str
        Log key, e.g. ``audio``, ``audio_ep``, ``default_nodecay``.
    """
    name = _param_group_base_name(group, index)
    suffixes = []
    if group.get("is_expert_parallel"):
        suffixes.append("ep")
    if _wd_mult(group) == 0.0:
        suffixes.append("nodecay")
    if suffixes:
        name = f"{name}_{'_'.join(suffixes)}"
    if used is not None and name in used:
        name = f"{name}_{index}"
    if used is not None:
        used.add(name)
    return name


def get_megatron_param_group_lrs(optimizer) -> Dict[str, float]:
    """Current ``lr`` keyed by named optimizer group.

    Megatron splits a yaml name by expert-parallel and weight-decay. Splits
    that share the same scheduled ``lr`` collapse to the yaml name so wandb
    does not grow a chart per split. A split with a different ``lr`` keeps
    ``_ep`` / ``_nodecay`` (or ``_{index}`` on collision).
    """
    by_base: Dict[str, List[Tuple[int, dict, float]]] = {}
    param_groups = getattr(optimizer, "param_groups", None) or []
    for i, group in enumerate(param_groups):
        lr = group.get("lr")
        if lr is None:
            continue
        base = _param_group_base_name(group, i)
        by_base.setdefault(base, []).append((i, group, float(lr)))

    lrs: Dict[str, float] = {}
    used: Set[str] = set()
    for items in by_base.values():
        # 相同 lr 的 group 合并到一起，只报一份
        if _lrs_all_close(lr for _, _, lr in items):
            i, group, lr = items[0]
            name = _param_group_base_name(group, i)
            if name in used:
                name = f"{name}_{i}"
            used.add(name)
            lrs[name] = lr
            continue
        for i, group, lr in items:
            lrs[param_group_log_name(group, i, used)] = lr
    return lrs


def canonical_lr_from_group_lrs(group_lrs: Dict[str, float]) -> Optional[float]:
    """Default-schedule lr for the historical ``{prefix}/lr`` scalar.

    Prefers the exact ``default`` key, then any ``default_*`` split, then
    the first reported group.
    """
    if not group_lrs:
        return None
    if DEFAULT_LR_GROUP_NAME in group_lrs:
        return group_lrs[DEFAULT_LR_GROUP_NAME]
    for name, lr in group_lrs.items():
        if name.startswith(DEFAULT_LR_GROUP_NAME + "_"):
            return lr
    return next(iter(group_lrs.values()))


def unmatched_param_lr_override_names(optimizer, param_lr_overrides) -> List[str]:
    """Return override names that did not produce an optimizer param group."""
    if not param_lr_overrides:
        return []
    present = set()
    for group in getattr(optimizer, "param_groups", None) or []:
        name = group.get(GCORE_LR_GROUP_KEY)
        if name:
            present.add(name)
    return [entry.name for entry in param_lr_overrides if entry.name not in present]


def warn_unmatched_param_lr_overrides(optimizer, param_lr_overrides) -> None:
    """Log a rank-0 warning for override patterns that matched nothing."""
    unmatched = unmatched_param_lr_override_names(optimizer, param_lr_overrides)
    if not unmatched:
        return
    name_to_pattern = {entry.name: entry.pattern for entry in param_lr_overrides}
    for name in unmatched:
        log(
            f"WARN: optimizer.param_lr_overrides name={name!r} "
            f"pattern={name_to_pattern[name]!r} matched no trainable parameters",
            rank=0,
        )


def append_param_group_lr_metrics(metrics: dict, optimizer, prefix: str) -> None:
    """Write ``{prefix}/lr/{group_name}`` for each collapsed optimizer group."""
    for name, lr in get_megatron_param_group_lrs(optimizer).items():
        metrics[f"{prefix}/lr/{name}"] = lr


def record_named_lrs(
    metrics: dict,
    prefix: str,
    lrs: Any,
    group_lrs: Optional[Dict[str, float]] = None,
) -> Any:
    """Write ``{prefix}/lr`` and ``{prefix}/lr/{name}`` into ``metrics``.

    Parameters
    ----------
    metrics : dict
        Destination metric dict.
    prefix : str
        Metric prefix, e.g. ``"policy"`` or ``"finetune"``.
    lrs : dict or float
        If dict, every group's current lr. ``{prefix}/lr`` is the default
        schedule. The canonical ``default`` key is omitted from
        ``{prefix}/lr/{name}`` because it already lives at ``{prefix}/lr``.
        If float (FSDP / mlite), written to ``{prefix}/lr`` and
        ``group_lrs`` supplies extra per-key series.
    group_lrs : dict, optional
        Extra per-key lrs when ``lrs`` is a scalar.

    Returns
    -------
    float or None
        Canonical learning rate stored at ``{prefix}/lr``.
    """
    if isinstance(lrs, dict):
        group_lrs = lrs
        canonical = canonical_lr_from_group_lrs(group_lrs)
    else:
        canonical = lrs
        group_lrs = group_lrs or {}
    if canonical is not None:
        metrics[f"{prefix}/lr"] = canonical
    if group_lrs:
        group_lrs = dict(group_lrs)
        group_lrs.pop(DEFAULT_LR_GROUP_NAME, None)
        for name, group_lr in group_lrs.items():
            metrics[f"{prefix}/lr/{name}"] = group_lr
    return canonical
