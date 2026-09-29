import json
import os
from types import SimpleNamespace

import numpy as np
from datasets import Dataset

from gpatch_v4.configs.data_config import E2hConfig
from gpatch_v4.trainer.grpo_single_ctrl_trainer import apply_e2h_train_schedule
from gpatch_v4.utils.e2h_sampler import E2HTaskSampler
from tasks.retool.add_retool_e2h_levels import add_levels_from_metadata
from tasks.retool.async_agent_loop.retool_dataset import (
    RetoolDataset,
    get_dataset_and_dataloader,
)

_FIXTURE_META = os.path.join(
    os.path.dirname(__file__),
    "data",
    "retool_e2h_tiny",
    "metadata.json",
)


def _config(
    metadata_file,
    enable=True,
    schedule="classic",
    gbs=4,
    mbs=1,
    epochs=2,
    hard_cover=True,
    num_workers=1,
    total_ppo_step=None,
):
    return SimpleNamespace(
        data=SimpleNamespace(
            data_pathes=[metadata_file],
            eval_data_pathes=None,
            e2h=E2hConfig(
                enable=enable,
                curriculum_schedule=schedule,
                set_total_ppo_step_from_hard_cover=hard_cover,
            ),
            sampler_seed=42,
            dataloader_pin_memory=False,
            dataloader_num_workers=num_workers,
            multiprocessing_method="fork",
        ),
        training=SimpleNamespace(
            rollout_gbs=gbs,
            rollout_mbs=mbs,
            eval_interval=0,
            eval_rollout_mbs=None,
            num_train_epoches=epochs,
            total_ppo_step=total_ppo_step,
            ppo_step_per_epoch=None,
        ),
    )


def test_retool_e2h_classic_step0_only_level0():
    # classic t=0 只抽 level 0
    cfg = _config(_FIXTURE_META, schedule="classic")
    ret = get_dataset_and_dataloader(cfg, dp_rank=0, dp_size=1)
    sampler = ret["train_sampler"]
    assert isinstance(sampler, E2HTaskSampler)
    batch = next(iter(ret["train_dataloader"]))
    assert batch["cache_keys"] == ["level"]
    assert set(batch["level"]) == {0}


def test_retool_e2h_dataloader_len_matches_cover_steps():
    cfg = _config(_FIXTURE_META, schedule="gaussian", hard_cover=False)
    ret = get_dataset_and_dataloader(cfg, dp_rank=0, dp_size=1)
    sampler = ret["train_sampler"]
    num_mb = cfg.training.rollout_gbs // cfg.training.rollout_mbs
    assert len(ret["train_dataloader"]) == sampler.steps_per_epoch * num_mb
    schedule = {
        "total_ppo_step": sampler.steps_per_epoch * cfg.training.num_train_epoches,
        "ppo_step_per_epoch": sampler.steps_per_epoch,
    }
    train_info = apply_e2h_train_schedule(
        cfg,
        {
            "total_ppo_step": 4096,
            "ppo_step_per_epoch": 4096,
            "prev_ppo_step": 0,
        },
        schedule,
    )
    assert train_info["ppo_step_per_epoch"] == sampler.steps_per_epoch
    assert train_info["total_ppo_step"] == (
        sampler.steps_per_epoch * cfg.training.num_train_epoches
    )
    assert cfg.training.total_ppo_step == train_info["total_ppo_step"]
    assert train_info["total_ppo_step"] != 4096


def test_retool_e2h_gaussian_shifts_to_harder_levels():
    cfg = _config(_FIXTURE_META, schedule="gaussian", hard_cover=False)
    ret = get_dataset_and_dataloader(cfg, dp_rank=0, dp_size=1)
    dataset = ret["train_dataset"]
    sampler = ret["train_sampler"]
    idxs = list(sampler)
    gbs = sampler.prompts_per_rank
    assert sampler.steps_per_epoch > 1
    early = dataset.level_ids[np.asarray(idxs[:gbs])]
    late = dataset.level_ids[np.asarray(idxs[-gbs:])]
    assert float(early.mean()) < float(late.mean())


def test_retool_e2h_explicit_total_ppo_step_sets_fixed_horizon():
    cfg = _config(
        _FIXTURE_META,
        schedule="gaussian",
        epochs=2,
        total_ppo_step=6,
    )
    ret = get_dataset_and_dataloader(cfg, dp_rank=0, dp_size=1)
    sampler = ret["train_sampler"]

    assert sampler.steps_per_epoch == 3
    assert sampler.horizon_source == "fixed_total_ppo_step"
    assert len(ret["train_dataloader"]) == (
        3 * cfg.training.rollout_gbs // cfg.training.rollout_mbs
    )


def test_retool_dataset_disabled_keeps_resumable_sampler():
    cfg = _config(_FIXTURE_META, enable=False)
    ret = get_dataset_and_dataloader(cfg, dp_rank=0, dp_size=1)
    assert not isinstance(ret["train_sampler"], E2HTaskSampler)
    batch = next(iter(ret["train_dataloader"]))
    assert "level" not in batch
    assert batch["cache_keys"] == []


def test_retool_eval_dataset_does_not_require_train_curriculum_levels(tmp_path):
    eval_jsonl = tmp_path / "eval.jsonl"
    eval_jsonl.write_text(
        ''.join(
            json.dumps({"question": f"eval{i}", "target": str(i)}) + "\n"
            for i in range(3)
        ),
        encoding="utf-8",
    )
    cfg = _config(_FIXTURE_META, num_workers=0)
    cfg.data.multiprocessing_method = None
    cfg.data.eval_data_pathes = [str(eval_jsonl)]
    cfg.training.eval_interval = 1
    cfg.training.eval_rollout_mbs = 1

    ret = get_dataset_and_dataloader(cfg, dp_rank=0, dp_size=1)

    assert isinstance(ret["train_sampler"], E2HTaskSampler)
    assert ret["eval_dataset"].level_ids is None
    assert [batch["question"][0] for batch in ret["eval_dataloader"]] == [
        "eval0", "eval1", "eval2"
    ]
    assert next(iter(ret["eval_dataloader"]))["cache_keys"] == []


def test_retool_eval_dataset_loads_aime_parquet(tmp_path):
    eval_parquet = tmp_path / "aime.parquet"
    Dataset.from_list([
        {"problem": "What is 2+2?", "answer": "4"},
        {"problem": "What is 3+3?", "answer": "6"},
    ]).to_parquet(str(eval_parquet))
    cfg = _config(_FIXTURE_META, num_workers=0)
    cfg.data.multiprocessing_method = None
    cfg.data.eval_data_pathes = [str(eval_parquet)]
    cfg.training.eval_interval = 1
    cfg.training.eval_rollout_mbs = 1

    ret = get_dataset_and_dataloader(cfg, dp_rank=0, dp_size=1)

    assert isinstance(ret["train_sampler"], E2HTaskSampler)
    assert ret["eval_dataset"].level_ids is None
    assert [(batch["question"][0], batch["target"][0]) for batch in ret["eval_dataloader"]] == [
        ("What is 2+2?", "4"),
        ("What is 3+3?", "6"),
    ]


def test_retool_eval_metadata_loads_aime_parquet(tmp_path):
    eval_parquet = tmp_path / "aime.parquet"
    Dataset.from_list([
        {"problem": "What is 2+2?", "answer": "4"},
        {"problem": "What is 3+3?", "answer": "6"},
    ]).to_parquet(str(eval_parquet))
    eval_metadata = tmp_path / "eval_metadata.json"
    eval_metadata.write_text(
        json.dumps({
            "name": "aime_2024",
            "data_files": [{"fp": str(eval_parquet)}],
            "data_file_num_lines": [2],
        }),
        encoding="utf-8",
    )
    cfg = _config(_FIXTURE_META, num_workers=0)
    cfg.data.multiprocessing_method = None
    cfg.data.eval_data_pathes = [str(eval_metadata)]
    cfg.training.eval_interval = 1
    cfg.training.eval_rollout_mbs = 1

    ret = get_dataset_and_dataloader(cfg, dp_rank=0, dp_size=1)

    assert isinstance(ret["train_sampler"], E2HTaskSampler)
    assert ret["eval_dataset"].level_ids is None
    assert [(batch["question"][0], batch["target"][0]) for batch in ret["eval_dataloader"]] == [
        ("What is 2+2?", "4"),
        ("What is 3+3?", "6"),
    ]


def test_retool_e2h_missing_level_raises(tmp_path):
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text('{"question": "q", "target": "1"}\n', encoding="utf-8")
    meta = tmp_path / "metadata.json"
    meta.write_text(
        json.dumps({"data_files": [{"fp": str(jsonl)}]}),
        encoding="utf-8",
    )
    cfg = _config(str(meta))
    try:
        RetoolDataset(cfg, str(meta))
    except ValueError as exc:
        assert "level" in str(exc)
    else:
        raise AssertionError("expected missing level to raise")


def test_add_retool_e2h_levels_roundtrip(tmp_path):
    src_jsonl = tmp_path / "src.jsonl"
    rows = [{"question": f"q{i}", "target": str(i)} for i in range(20)]
    src_jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    src_meta = tmp_path / "src_meta.json"
    src_meta.write_text(
        json.dumps({"data_files": [{"fp": str(src_jsonl)}]}),
        encoding="utf-8",
    )
    dst = tmp_path / "dst"
    hist = add_levels_from_metadata(str(src_meta), str(dst), seed=42)
    assert sum(hist.values()) == 20
    cfg = _config(str(dst / "metadata.json"), schedule="balanced")
    ret = get_dataset_and_dataloader(cfg, dp_rank=0, dp_size=1)
    assert isinstance(ret["train_sampler"], E2HTaskSampler)
    assert set(int(x) for x in ret["train_dataset"].level_ids) == set(hist)
