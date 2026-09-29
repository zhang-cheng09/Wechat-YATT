import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from gpatch_v4.configs.optimizer_config import OptimizerConfig, ParamLrOverride


def _load_param_lr():
    """Load ``utils/param_lr.py`` without executing ``gpatch_v4.utils`` init.

    ``gpatch_v4.utils.__init__`` imports ``communication_utils``, which requires
    ``megatron.core``. These unit tests should run without Megatron.
    """
    path = Path(__file__).resolve().parents[2] / "gpatch_v4" / "utils" / "param_lr.py"
    spec = importlib.util.spec_from_file_location("_gcore_param_lr", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_param_lr = _load_param_lr()
GCORE_LR_GROUP_KEY = _param_lr.GCORE_LR_GROUP_KEY
append_param_group_lr_metrics = _param_lr.append_param_group_lr_metrics
build_param_lr_config_overrides = _param_lr.build_param_lr_config_overrides
canonical_lr_from_group_lrs = _param_lr.canonical_lr_from_group_lrs
get_megatron_param_group_lrs = _param_lr.get_megatron_param_group_lrs
param_group_log_name = _param_lr.param_group_log_name
param_name_matches_regex = _param_lr.param_name_matches_regex
record_named_lrs = _param_lr.record_named_lrs



def test_param_name_matches_regex_strips_module_prefix():
    assert param_name_matches_regex("module.audio_model.proj.weight", "audio_model")
    assert param_name_matches_regex("module.module.audio_model.proj.weight", r"^audio_model")
    assert param_name_matches_regex("audio_model.proj.weight", "audio_model")
    assert not param_name_matches_regex("language_model.decoder.weight", "audio_model")


def test_param_lr_override_requires_name():
    entry = ParamLrOverride(name="audio", pattern="audio_model", lr=1e-5)
    assert entry.name == "audio"
    assert entry.lr == pytest.approx(1e-5)
    with pytest.raises(ValueError, match="missing name"):
        ParamLrOverride(pattern="audio_model", lr=1e-5)


def test_optimizer_config_param_lr_overrides_post_init():
    cfg = OptimizerConfig(
        param_lr_overrides=[
            ParamLrOverride(name="audio", pattern="audio_model", lr=1e-5),
            ParamLrOverride(
                name="vision",
                pattern="vision_model",
                lr=2e-6,
                min_lr=0.0,
            ),
        ]
    )
    assert cfg.param_lr_overrides[0].name == "audio"
    assert cfg.param_lr_overrides[0].lr == pytest.approx(1e-5)
    assert cfg.param_lr_overrides[1].name == "vision"
    assert cfg.param_lr_overrides[1].min_lr == pytest.approx(0.0)


def test_optimizer_config_keeps_param_lr_override_yaml_order():
    cfg = OptimizerConfig(
        param_lr_overrides=[
            ParamLrOverride(name="vision", pattern="vision_model", lr=2e-6),
            ParamLrOverride(name="audio", pattern="audio_model", lr=1e-5),
        ]
    )
    assert [entry.name for entry in cfg.param_lr_overrides] == ["vision", "audio"]


def test_optimizer_config_rejects_bad_param_lr_overrides():
    with pytest.raises(ValueError, match="missing pattern"):
        ParamLrOverride(name="audio", pattern="", lr=1e-5)
    with pytest.raises(ValueError, match="missing lr"):
        ParamLrOverride(name="audio", pattern="audio_model")
    with pytest.raises(ValueError, match="not valid regex"):
        ParamLrOverride(name="audio", pattern="(", lr=1e-5)
    with pytest.raises(ValueError, match="duplicate name"):
        OptimizerConfig(
            param_lr_overrides=[
                ParamLrOverride(name="audio", pattern="audio_model", lr=1e-5),
                ParamLrOverride(name="audio", pattern="whisper", lr=1e-6),
            ]
        )


def test_get_megatron_param_group_lrs_reports_every_group():
    class _Opt:
        param_groups = [
            {
                "params": [0],
                "lr": 1e-6,
                "default_config": True,
                "max_lr": 1e-6,
            },
            {
                "params": [1],
                "lr": 1e-5,
                "default_config": False,
                "max_lr": 1e-5,
                GCORE_LR_GROUP_KEY: "audio",
            },
            {
                "params": [2],
                "lr": 2e-5,
                "default_config": False,
                "max_lr": 2e-5,
                GCORE_LR_GROUP_KEY: "audio",
                "is_expert_parallel": True,
            },
            {
                "params": [3],
                "lr": 3e-5,
                "default_config": False,
                "max_lr": 3e-5,
                GCORE_LR_GROUP_KEY: "audio",
                "wd_mult": 0.0,
            },
        ]

    lrs = get_megatron_param_group_lrs(_Opt())
    assert lrs["default"] == pytest.approx(1e-6)
    assert lrs["audio"] == pytest.approx(1e-5)
    assert lrs["audio_ep"] == pytest.approx(2e-5)
    assert lrs["audio_nodecay"] == pytest.approx(3e-5)
    assert canonical_lr_from_group_lrs(lrs) == pytest.approx(1e-6)


def test_get_megatron_param_group_lrs_collapses_same_lr_splits():
    class _Opt:
        param_groups = [
            {
                "params": [0],
                "lr": 5e-7,
                "default_config": True,
            },
            {
                "params": [1],
                "lr": 5e-7,
                "default_config": True,
                "wd_mult": 0.0,
            },
            {
                "params": [2],
                "lr": 5e-7,
                GCORE_LR_GROUP_KEY: "vit",
            },
            {
                "params": [3],
                "lr": 5e-7,
                GCORE_LR_GROUP_KEY: "vit",
                "wd_mult": 0.0,
            },
            {
                "params": [4],
                "lr": 5e-7,
                GCORE_LR_GROUP_KEY: "vit_proj",
            },
            {
                "params": [5],
                "lr": 5e-7,
                GCORE_LR_GROUP_KEY: "vit_proj",
                "wd_mult": 0.0,
            },
        ]

    lrs = get_megatron_param_group_lrs(_Opt())
    assert set(lrs) == {"default", "vit", "vit_proj"}
    assert lrs["default"] == pytest.approx(5e-7)
    assert lrs["vit"] == pytest.approx(5e-7)
    assert lrs["vit_proj"] == pytest.approx(5e-7)

    metrics = {}
    record_named_lrs(metrics, "finetune", lrs)
    assert metrics["finetune/lr"] == pytest.approx(5e-7)
    assert metrics["finetune/lr/vit"] == pytest.approx(5e-7)
    assert metrics["finetune/lr/vit_proj"] == pytest.approx(5e-7)
    assert "finetune/lr/default" not in metrics


def test_record_named_lrs_keeps_groups_that_differ():
    class _Opt:
        param_groups = [
            {"params": [0], "lr": 5e-7, "default_config": True},
            {"params": [1], "lr": 5e-7, "default_config": True, "wd_mult": 0.0},
            {"params": [2], "lr": 1e-6, GCORE_LR_GROUP_KEY: "vit"},
            {"params": [3], "lr": 1e-6, GCORE_LR_GROUP_KEY: "vit", "wd_mult": 0.0},
        ]

    lrs = get_megatron_param_group_lrs(_Opt())
    assert set(lrs) == {"default", "vit"}
    metrics = {}
    record_named_lrs(metrics, "finetune", lrs)
    assert metrics["finetune/lr"] == pytest.approx(5e-7)
    assert metrics["finetune/lr/vit"] == pytest.approx(1e-6)
    assert "finetune/lr/default" not in metrics
    assert "finetune/lr/vit_nodecay" not in metrics


def test_param_group_log_name_unique_on_collision():
    used = set()
    g0 = {GCORE_LR_GROUP_KEY: "audio", "lr": 1e-5}
    g1 = {GCORE_LR_GROUP_KEY: "audio", "lr": 9e-6}
    assert param_group_log_name(g0, 0, used) == "audio"
    assert param_group_log_name(g1, 1, used) == "audio_1"


def test_append_and_record_lr_metrics():
    class _Opt:
        param_groups = [
            {"params": [0], "lr": 1e-6, "default_config": True},
            {"params": [1], "lr": 1e-5, GCORE_LR_GROUP_KEY: "audio"},
            {
                "params": [2],
                "lr": 2e-5,
                GCORE_LR_GROUP_KEY: "audio",
                "is_expert_parallel": True,
            },
        ]

    metrics = {}
    append_param_group_lr_metrics(metrics, _Opt(), "finetune")
    assert metrics["finetune/lr/default"] == pytest.approx(1e-6)
    assert metrics["finetune/lr/audio"] == pytest.approx(1e-5)
    assert metrics["finetune/lr/audio_ep"] == pytest.approx(2e-5)

    actor_metrics = {}
    group_lrs = {"default": 1e-6, "audio": 1e-5, "audio_ep": 2e-5}
    canonical = record_named_lrs(actor_metrics, "policy", group_lrs)
    assert canonical == pytest.approx(1e-6)
    assert actor_metrics["policy/lr"] == pytest.approx(1e-6)
    assert actor_metrics["policy/lr/audio"] == pytest.approx(1e-5)
    assert actor_metrics["policy/lr/audio_ep"] == pytest.approx(2e-5)
    assert "policy/lr/default" not in actor_metrics

    scalar_metrics = {}
    record_named_lrs(scalar_metrics, "policy", 3e-4, {"default": 3e-4})
    assert scalar_metrics["policy/lr"] == pytest.approx(3e-4)
    assert "policy/lr/default" not in scalar_metrics


class _ToyOmni(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Linear(4, 4, bias=False)
        self.audio_model = nn.Linear(4, 4, bias=False)


@pytest.mark.parametrize("wrap_module_prefix", [False, True])
def test_param_lr_overrides_set_max_lr_on_matching_params(monkeypatch, wrap_module_prefix):
    pytest.importorskip("megatron.core.optimizer")
    from megatron.core.optimizer import OptimizerConfig as McoreOptimizerConfig
    from megatron.core.optimizer import _get_param_groups

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output_list, obj: output_list.__setitem__(0, obj),
    )

    inner = _ToyOmni()
    if wrap_module_prefix:
        model = nn.Module()
        model.add_module("module", inner)
    else:
        model = inner

    overrides = build_param_lr_config_overrides(
        [ParamLrOverride(name="audio", pattern="audio_model", lr=1e-5, min_lr=0.0)]
    )
    opt_config = McoreOptimizerConfig(optimizer="adam", lr=1e-6, min_lr=0.0)
    param_groups = _get_param_groups([model], opt_config, overrides)

    named = dict(model.named_parameters())
    audio_ids = {id(p) for n, p in named.items() if "audio_model" in n}
    llm_ids = {id(p) for n, p in named.items() if "language_model" in n}
    assert audio_ids and llm_ids

    audio_groups = [
        g for g in param_groups if any(id(p) in audio_ids for p in g["params"])
    ]
    llm_groups = [g for g in param_groups if any(id(p) in llm_ids for p in g["params"])]
    assert audio_groups
    assert llm_groups
    for group in audio_groups:
        assert group["max_lr"] == pytest.approx(1e-5)
        assert group["min_lr"] == pytest.approx(0.0)
        assert group[GCORE_LR_GROUP_KEY] == "audio"
        assert group["default_config"] is False
    for group in llm_groups:
        assert group["max_lr"] == pytest.approx(1e-6)
        assert group.get(GCORE_LR_GROUP_KEY) in (None, "default")
        assert group["default_config"] is True


def test_first_matching_param_lr_override_wins(monkeypatch):
    pytest.importorskip("megatron.core.optimizer")
    from megatron.core.optimizer import OptimizerConfig as McoreOptimizerConfig
    from megatron.core.optimizer import _get_param_groups

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output_list, obj: output_list.__setitem__(0, obj),
    )
    model = _ToyOmni()
    # YAML list order: more specific pattern first, catch-all last.
    overrides = build_param_lr_config_overrides(
        [
            ParamLrOverride(name="audio", pattern="audio_model", lr=1e-5),
            ParamLrOverride(name="all", pattern=".*", lr=9e-4),
        ]
    )
    opt_config = McoreOptimizerConfig(optimizer="adam", lr=1e-6, min_lr=0.0)
    param_groups = _get_param_groups([model], opt_config, overrides)
    named = dict(model.named_parameters())
    audio_ids = {id(p) for n, p in named.items() if "audio_model" in n}
    llm_ids = {id(p) for n, p in named.items() if "language_model" in n}
    audio_groups = [
        g for g in param_groups if any(id(p) in audio_ids for p in g["params"])
    ]
    llm_groups = [g for g in param_groups if any(id(p) in llm_ids for p in g["params"])]
    assert audio_groups[0]["max_lr"] == pytest.approx(1e-5)
    assert llm_groups[0]["max_lr"] == pytest.approx(9e-4)
    assert llm_groups[0][GCORE_LR_GROUP_KEY] == "all"


def test_optimizer_config_default_param_lr_overrides_empty():
    cfg = OptimizerConfig()
    assert cfg.param_lr_overrides == []
