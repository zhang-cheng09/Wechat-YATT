import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from gpatch_v4.training_backend.fsdp2_backend import checkpoint
from gpatch_v4.training_backend.fsdp2_backend import fsdp2_engine_lm
from gpatch_v4.training_backend.fsdp2_backend.lr_scheduler import FSDPLRScheduler
from gpatch_v4.training_backend.fsdp2_backend.optimizer import setup_optimizer


EXPERT_NAME = "model.layers.0.mlp.experts.gate_up_proj"
DENSE_NAME = "model.layers.0.self_attn.q_proj.weight"


def _optimizer_dcp_worker(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.mlp = torch.nn.Module()
        model.model.mlp.experts = torch.nn.Module()
        model.model.mlp.experts.gate_up_proj = torch.nn.Parameter(torch.zeros(1))
        parameter = model.model.mlp.experts.gate_up_proj
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        optimizer.state[parameter] = {
            "step": torch.tensor(float(rank + 1)),
            "exp_avg": torch.full_like(parameter, float(rank + 1)),
            "exp_avg_sq": torch.full_like(parameter, float(rank + 11)),
        }
        optimizer_state = checkpoint.OptimizerState(model, optimizer, ep_rank=rank)

        dcp.save({"optimizer_state": optimizer_state}, checkpoint_id=checkpoint_dir)
        for value in optimizer.state[parameter].values():
            value.zero_()
        dcp.load({"optimizer_state": optimizer_state}, checkpoint_id=checkpoint_dir)

        restored = optimizer.state[parameter]
        assert restored["step"].item() == rank + 1
        torch.testing.assert_close(
            restored["exp_avg"], torch.full_like(parameter, float(rank + 1))
        )
        torch.testing.assert_close(
            restored["exp_avg_sq"], torch.full_like(parameter, float(rank + 11))
        )
    finally:
        dist.destroy_process_group()


def test_optimizer_state_round_trips_distinct_ep_rank_values(tmp_path: Path) -> None:
    mp.start_processes(
        _optimizer_dcp_worker,
        args=(2, str(tmp_path / "pg_init"), str(tmp_path / "checkpoint")),
        nprocs=2,
        join=True,
        start_method="spawn",
    )


def _vision_training_state(
    dense_mesh: DeviceMesh, expert_mesh: DeviceMesh,
) -> tuple[torch.nn.Module, torch.optim.AdamW, FSDPLRScheduler]:
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="qwen4_exp")
    engram_name = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.weight"
    weights = {
        "model.language_model.embed_tokens.weight": torch.ones(4, 2),
        "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.ones(2, 2),
        engram_name: torch.ones(4, 2),
        "model.visual.patch_embed.proj.weight": torch.ones(2, 2),
    }
    for name, tensor in weights.items():
        tensor = tensor.to(dense_mesh.device_type)
        module = model
        *prefix, leaf = name.split(".")
        for part in prefix:
            if part not in module._modules:
                module.add_module(part, torch.nn.Module())
            module = module._modules[part]
        if not name.startswith("model.visual."):
            mesh = expert_mesh if name.endswith(".mlp.experts.gate_up_proj") else dense_mesh
            tensor = distribute_tensor(tensor, mesh, [Shard(0)])
        module.register_parameter(leaf, torch.nn.Parameter(tensor))
    model._engram_param_ids = {id(model.get_parameter(engram_name))}
    optimizer = setup_optimizer(
        SimpleNamespace(optimizer=SimpleNamespace(
            optimizer_type="adamw", lr=0.01, adam_beta1=0.9, adam_beta2=0.999,
            weight_decay=0.01, adam_epsilon=1e-8,
        )),
        model,
    )
    scheduler = FSDPLRScheduler(
        optimizer, init_lr=0.01, max_lr=0.04, min_lr=0.001,
        lr_warmup_steps=4, lr_decay_steps=10, lr_decay_style="cosine",
    )
    return model, optimizer, scheduler


def _vision_training_state_worker(
    rank: int, root: str, train_vision: bool, device_type: str,
) -> None:
    torch.set_num_threads(1)
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        init_method=f"file://{root}/rendezvous", rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        dense_mesh = DeviceMesh(device_type, torch.arange(2))
        ep_mesh = DeviceMesh(
            device_type, torch.arange(2).view(1, 2), mesh_dim_names=("ep_fsdp", "ep"),
        )
        model, optimizer, scheduler = _vision_training_state(dense_mesh, ep_mesh["ep_fsdp"])

        def train_step(
            candidate: torch.nn.Module,
            optim: torch.optim.AdamW,
            lr_scheduler: FSDPLRScheduler,
            has_image: bool,
        ) -> None:
            optim.zero_grad(set_to_none=True)
            for index, (name, param) in enumerate(candidate.named_parameters()):
                if name.startswith("model.visual.") and not has_image:
                    continue
                gradient = torch.arange(
                    param.numel(), dtype=param.dtype, device=param.device,
                ).reshape(param.shape)
                gradient.add_(index + 1 + (rank if ".mlp.experts." in name else 0))
                if isinstance(param, DTensor):
                    gradient = distribute_tensor(gradient, param.device_mesh, param.placements)
                param.grad = gradient
            optim.step()
            lr_scheduler.step()

        for _ in range(2):
            train_step(model, optimizer, scheduler, train_vision)
            for parameter in model.parameters():
                if id(parameter) in model._engram_param_ids:
                    assert parameter.device.type == device_type
                    assert optimizer.state[parameter]["exp_avg"].device.type == device_type
                    assert optimizer.state[parameter]["exp_avg_sq"].device.type == device_type
        config = SimpleNamespace(checkpoint=SimpleNamespace(
            save_ckpt_path=root, load_ckpt_path=root, no_save_optim=False, no_load_optim=False,
        ))
        # 仅替换 CUDA 清理与 RNG 读取，实际执行混合 Tensor/DTensor 的 DCP 保存和恢复。
        with (
            patch.object(torch.cuda, "synchronize"),
            patch.object(torch.cuda, "empty_cache"),
            patch.object(torch.cuda, "get_rng_state_all", return_value=[]),
        ):
            checkpoint.save_checkpoint(
                config, model, optimizer, scheduler, global_step=2,
                save_dcp_model=False, ep_rank=rank,
            )
        assert (Path(root) / "latest_checkpointed_iteration.txt").read_text() == "2"
        assert not (Path(root) / "iter_0000002/model").exists()

        restored, restored_optimizer, restored_scheduler = _vision_training_state(
            dense_mesh, ep_mesh["ep_fsdp"],
        )
        restored.load_state_dict(model.state_dict())
        step = checkpoint.load_checkpoint(
            config, restored, restored_optimizer, restored_scheduler, global_step=2,
            load_dcp_model=False, ep_rank=rank,
        )
        assert step == 2
        assert restored_scheduler.state_dict() == scheduler.state_dict()
        assert len(restored_optimizer.param_groups) == 3
        for group, expected_group in zip(restored_optimizer.param_groups, optimizer.param_groups):
            assert {key: value for key, value in group.items() if key != "params"} == {
                key: value for key, value in expected_group.items() if key != "params"
            }
        for name, parameter in model.named_parameters():
            actual_state = restored_optimizer.state[restored.get_parameter(name)]
            expected_state = optimizer.state[parameter]
            assert actual_state.keys() == expected_state.keys()
            for key, expected in expected_state.items():
                torch.testing.assert_close(actual_state[key], expected, rtol=0, atol=0)

        for has_image in (False, True):
            train_step(model, optimizer, scheduler, has_image)
            train_step(restored, restored_optimizer, restored_scheduler, has_image)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(restored.get_parameter(name), parameter, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("train_vision", [False, True], ids=["text_only", "image"])
@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_vision_training_state_round_trip(
    tmp_path: Path, train_vision: bool, device_type: str,
) -> None:
    if device_type == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    mp.start_processes(
        _vision_training_state_worker, args=(str(tmp_path), train_vision, device_type), nprocs=2,
        join=True, start_method="spawn",
    )


def test_optimizer_state_namespaces_and_restores_ep_parameters() -> None:
    optimizer_state = {
        "state": {
            EXPERT_NAME: {"exp_avg": torch.tensor([1.0])},
            DENSE_NAME: {"exp_avg": torch.tensor([2.0])},
        },
        "param_groups": [{"params": [EXPERT_NAME, DENSE_NAME]}],
    }
    checkpoint._map_ep_optimizer_parameter_names(optimizer_state, 2, add_prefix=True)
    assert list(optimizer_state["state"]) == [f"__ep_2__.{EXPERT_NAME}", DENSE_NAME]
    assert optimizer_state["param_groups"][0]["params"] == [
        f"__ep_2__.{EXPERT_NAME}",
        DENSE_NAME,
    ]

    checkpoint._map_ep_optimizer_parameter_names(optimizer_state, 2, add_prefix=False)
    assert list(optimizer_state["state"]) == [EXPERT_NAME, DENSE_NAME]
    assert optimizer_state["param_groups"][0]["params"] == [EXPERT_NAME, DENSE_NAME]


@pytest.mark.parametrize(
    ("model_arch", "resume_model_from_hf"),
    [("deepseek_v4", True), ("qwen4_exp", True), ("qwen2", False)],
)
def test_resume_selects_model_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_arch: str,
    resume_model_from_hf: bool,
) -> None:
    engine = object.__new__(fsdp2_engine_lm.Fsdp2EngineLm)
    engine.config = SimpleNamespace(
        training=SimpleNamespace(
            auto_load_from_save_ckpt=True,
            recompute=False,
        ),
        debug=SimpleNamespace(debug_no_optim=False),
    )
    engine.policy_config = SimpleNamespace(
        model_arch=model_arch,
        hf_model_path="/base",
        without_ref=True,
        without_optim=False,
    )
    engine.training_config = SimpleNamespace(use_linear_ce=False)
    engine.checkpoint_config = SimpleNamespace(
        load_ckpt_path=str(tmp_path),
        export_hf_save_path=None,
        no_save_optim=False,
    )
    engine._get_init_weight_context_manager = lambda: None
    engine.get_swap_state = lambda: SimpleNamespace(ref_model=True)
    engine.post_init_model = lambda _model: None
    loaded = {}
    model = SimpleNamespace(config=object(), _ep_rank=2)

    def build_model(_init_context, hf_model_path):
        loaded["hf_model_path"] = hf_model_path
        return model

    def load_training_state(*args, **kwargs):
        loaded["checkpoint_args"] = args
        loaded["checkpoint_kwargs"] = kwargs
        return 7

    engine.get_fsdp2_model = build_model
    clear_memory_calls = []
    monkeypatch.setattr(fsdp2_engine_lm, "get_latest_checkpoint_folder", lambda _path: 7)
    monkeypatch.setattr(fsdp2_engine_lm, "setup_optimizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(fsdp2_engine_lm, "setup_lr_scheduler", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(fsdp2_engine_lm, "load_checkpoint", load_training_state)
    monkeypatch.setattr(fsdp2_engine_lm, "clear_memory", lambda: clear_memory_calls.append(None))

    resumed_step = engine.setup_model_and_get_optimizer()

    assert resumed_step == 7
    expected_path = str(tmp_path / "hf" / "7") if resume_model_from_hf else "/base"
    assert loaded["hf_model_path"] == expected_path
    assert loaded["checkpoint_args"][-1] == 7
    assert loaded["checkpoint_kwargs"] == {
        "load_dcp_model": not resume_model_from_hf,
        "ep_rank": model._ep_rank if resume_model_from_hf else None,
    }
    assert clear_memory_calls == ([None] if model_arch == "deepseek_v4" else [])


def test_resume_reads_export_hf_save_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """配了 export_hf_save_path 时，续训从该目录的 step 子目录读 HF 权重。"""
    engine = object.__new__(fsdp2_engine_lm.Fsdp2EngineLm)
    engine.config = SimpleNamespace(
        training=SimpleNamespace(auto_load_from_save_ckpt=True, recompute=False),
        debug=SimpleNamespace(debug_no_optim=False),
    )
    engine.policy_config = SimpleNamespace(
        model_arch="qwen4_exp",
        hf_model_path="/base",
        without_ref=True,
        without_optim=False,
    )
    engine.training_config = SimpleNamespace(use_linear_ce=False)
    export_root = tmp_path / "hf_export"
    engine.checkpoint_config = SimpleNamespace(
        load_ckpt_path=str(tmp_path / "train"),
        export_hf_save_path=str(export_root),
        no_save_optim=False,
    )
    engine._get_init_weight_context_manager = lambda: None
    engine.get_swap_state = lambda: SimpleNamespace(ref_model=True)
    engine.post_init_model = lambda _model: None
    loaded = {}
    model = SimpleNamespace(config=object(), _ep_rank=0)

    def build_model(_init_context, hf_model_path):
        loaded["hf_model_path"] = hf_model_path
        return model

    engine.get_fsdp2_model = build_model
    monkeypatch.setattr(fsdp2_engine_lm, "get_latest_checkpoint_folder", lambda _path: 4)
    monkeypatch.setattr(fsdp2_engine_lm, "setup_optimizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(fsdp2_engine_lm, "setup_lr_scheduler", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(fsdp2_engine_lm, "load_checkpoint", lambda *_args, **_kwargs: 4)
    monkeypatch.setattr(fsdp2_engine_lm, "clear_memory", lambda: None)

    assert engine.setup_model_and_get_optimizer() == 4
    assert loaded["hf_model_path"] == str(export_root / "4")


@pytest.mark.parametrize("model_arch", ["deepseek_v4", "qwen4_exp"])
def test_hp_auto_resume_requires_saved_optimizer(model_arch: str) -> None:
    engine = object.__new__(fsdp2_engine_lm.Fsdp2EngineLm)
    engine.config = SimpleNamespace(
        training=SimpleNamespace(auto_load_from_save_ckpt=True),
    )
    engine.policy_config = SimpleNamespace(model_arch=model_arch)
    engine.checkpoint_config = SimpleNamespace(no_save_optim=True)

    with pytest.raises(ValueError, match="checkpoint.no_save_optim=False"):
        engine.setup_model_and_get_optimizer()
