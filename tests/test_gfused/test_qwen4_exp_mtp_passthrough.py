import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors import safe_open
from torch.distributed.device_mesh import init_device_mesh

from gpatch_v4.configs.optimizer_config import OptimizerConfig
from gpatch_v4.extended_model.qwen4_exp import Qwen4ExpPostInitModel
from gpatch_v4.models.qwen4_exp import (
    Qwen4ExpConfig,
    Qwen4ExpHpForConditionalGeneration,
    Qwen4ExpTextConfig,
    apply_hp,
    swap_parallel_modules,
)
from gpatch_v4.training_backend.fsdp2_backend.hp_builder.base import meta_construct_fp32
from gpatch_v4.training_backend.fsdp2_backend.hp_builder.qwen4_exp import Qwen4ExpHpBuilder
from gpatch_v4.models.qwen4_exp.checkpoint import save_checkpoint_hp
from gpatch_v4.training_backend.fsdp2_backend.optimizer import setup_optimizer
from test_qwen4_exp_apply_hp_gpu import TINY_KWARGS


def _tiny_config(**text_overrides) -> Qwen4ExpConfig:
    text_kwargs = {**TINY_KWARGS, **text_overrides}
    return Qwen4ExpConfig(
        text_config=Qwen4ExpTextConfig(**text_kwargs),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=1,
            out_hidden_size=text_kwargs["hidden_size"],
            num_position_embeddings=16,
        ),
    )


def _reject_mtp_forward(module, args) -> None:
    raise AssertionError("load-only MTP must not execute forward")


def _train_save_worker(
    rank: int,
    root: str,
    enable_mtp: bool,
    ep_size: int,
    distributed_save: bool,
) -> None:
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{root / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=180),
    )
    try:
        engine = SimpleNamespace(
            training_config=SimpleNamespace(enable_mtp=enable_mtp),
            config=SimpleNamespace(debug=SimpleNamespace(debug_truncate_num_hidden_layers=None)),
        )
        cfg = Qwen4ExpHpBuilder().prepare_config(
            Qwen4ExpHpForConditionalGeneration,
            str(root / "source"),
            engine,
            False,
        )
        model = meta_construct_fp32(Qwen4ExpHpForConditionalGeneration, cfg)
        mesh = init_device_mesh(
            "cuda",
            (2 // ep_size, ep_size),
            mesh_dim_names=("ep_fsdp", "ep"),
        )
        model = apply_hp(model, mesh, attn_backend="flex", ep_backend="eager")
        model.load_checkpoint_hp(str(root / "source"))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        Qwen4ExpPostInitModel(
            SimpleNamespace(
                training=SimpleNamespace(
                    freeze_moe_router=False,
                    online_mtp_sft=False,
                    recompute=True,
                    selective_recompute=False,
                    selective_recompute_max_layers=None,
                )
            )
        )(model)
        model.train()
        assert (model.mtp is not None) is enable_mtp
        mtp_parameters = {
            name: param
            for name, param in model.named_parameters() if name.startswith("mtp.")
        }
        assert bool(mtp_parameters) is enable_mtp
        for param in mtp_parameters.values():
            assert not param.requires_grad
        before = {name: param.to_local().clone() for name, param in mtp_parameters.items()}
        hooks = []
        if enable_mtp:
            hooks = [
                module.register_forward_pre_hook(_reject_mtp_forward)
                for module in model.mtp.modules()
            ]
        optimizer = setup_optimizer(
            SimpleNamespace(optimizer=OptimizerConfig(lr=0.01, weight_decay=0.1)),
            model,
        )
        assert not (
            {id(param)
             for param in mtp_parameters.values()} &
            {id(param)
             for group in optimizer.param_groups
             for param in group["params"]}
        )
        head_before = model.lm_head.weight.to_local().clone()
        vocab_size = cfg.get_text_config().vocab_size
        torch.manual_seed(123 + rank)
        for _ in range(3):
            tokens = torch.randint(0, vocab_size, (1, 24), device="cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=tokens).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, vocab_size),
                tokens[:, 1:].reshape(-1),
            )
            assert torch.isfinite(loss)
            loss.backward()
            assert all(param.grad is None for param in mtp_parameters.values())
            model.clip_grad_norm_(1.0)
            optimizer.step()
        assert not torch.equal(head_before, model.lm_head.weight.to_local())
        for name, param in mtp_parameters.items():
            assert torch.equal(before[name], param.to_local()), name
            assert param not in optimizer.state
        for hook in hooks:
            hook.remove()
        model.save_checkpoint_hp(
            str(root / "saved"),
            orig_ckpt_dir=str(root / "source"),
            preserve_mtp=enable_mtp,
            save_distributed_ep_shard=distributed_save,
        )
        if rank == 0:
            source_map = json.loads((root / "source/model.safetensors.index.json").read_text()
                                   )["weight_map"]
            saved_map = json.loads((root / "saved/model.safetensors.index.json").read_text()
                                  )["weight_map"]
            source_keys = {key for key in source_map if key.startswith("mtp.")}
            saved_keys = {key for key in saved_map if key.startswith("mtp.")}
            assert len(source_keys) == 31
            assert saved_keys == (source_keys if enable_mtp else set())
            for key in saved_keys:
                with (
                    safe_open(root / "source" / source_map[key], framework="pt") as original,
                    safe_open(root / "saved" / saved_map[key], framework="pt") as saved,
                ):
                    expected, actual = original.get_tensor(key), saved.get_tensor(key)
                    assert expected.dtype == actual.dtype
                    assert torch.equal(expected, actual), key
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="需要至少 2 张 GPU")
@pytest.mark.parametrize(
    ("enable_mtp", "ep_size", "distributed_save"),
    [(False, 2, False), (True, 1, False), (True, 2, False), (True, 2, True)],
)
def test_load_only_mtp_survives_training_and_save(
    tmp_path: Path,
    enable_mtp: bool,
    ep_size: int,
    distributed_save: bool,
) -> None:
    torch.manual_seed(42)
    cfg = _tiny_config(
        mtp={
            "layer_types": ["full_attention"],
            "num_hidden_layers": 1
        },
        mtp_num_hidden_layers=1,
    )
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        model = Qwen4ExpHpForConditionalGeneration(cfg)
    finally:
        torch.set_default_dtype(prev)
    swap_parallel_modules(model, attn_backend="dense")
    model._ep_size, model._ep_rank = 1, 0
    origin = tmp_path / "origin"
    cfg.save_pretrained(origin)
    save_checkpoint_hp(
        model, str(tmp_path / "source"), orig_ckpt_dir=str(origin), preserve_mtp=True
    )
    del model
    mp.start_processes(
        _train_save_worker,
        args=(str(tmp_path), enable_mtp, ep_size, distributed_save),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
