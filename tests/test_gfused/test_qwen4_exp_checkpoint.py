"""流式 checkpoint I/O 的测试。

单进程覆盖：原生权重名、缺文件时响亮报错、save->load 往返一致、Engram 分片对齐契约、
meta buffer 被重新计算。
多 rank（gloo）覆盖：EP 切片 + Engram 按 owner 行区间加载后，各 rank 拼起来等于原权重。
"""
import datetime
import json
import os
import pathlib
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from gpatch_v4.configs.checkpoint_config import CheckpointConfig
from gpatch_v4.models.qwen4_exp import (
    Qwen4ExpConfig,
    Qwen4ExpHpForConditionalGeneration,
    Qwen4ExpTextConfig,
    swap_parallel_modules,
)
from gpatch_v4.models.hp_module import HpModule
from gpatch_v4.models.qwen4_exp.checkpoint import (
    _read_weight_map,
    load_checkpoint_hp,
    save_checkpoint_hp,
)
from gpatch_v4.training_backend.fsdp2_backend.mixin import CheckpointMixin

NUM_EXPERTS = 8

TINY_KWARGS = dict(
    vocab_size=128,
    hidden_size=32,
    num_hidden_layers=4,
    full_attention_interval=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    linear_num_key_heads=1,
    linear_num_value_heads=2,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=4,
    output_gate_type="sigmoid",
    num_experts=NUM_EXPERTS,
    num_experts_per_tok=2,
    moe_intermediate_size=8,
    shared_expert_intermediate_size=8,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=16,
    indexer_compress_ratio=4,
    hc_count=4,
    hc_lowrank=8,
    ple_layer_ids=[2],
    ple_embed_dim=32,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=256,
    make_ngram_vocab_size_divisible_by=128,
    split_ngram_parts=4,
    eos_token_id=1,
    tie_word_embeddings=False,
    use_cache=False,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
    },
)


def _config(**text_overrides) -> Qwen4ExpConfig:
    return Qwen4ExpConfig(
        text_config=Qwen4ExpTextConfig(**{**TINY_KWARGS, **text_overrides}),
        vision_config=dict(
            depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
            patch_size=2, spatial_merge_size=2, temporal_patch_size=1,
            out_hidden_size=TINY_KWARGS["hidden_size"], num_position_embeddings=16,
        ),
    )


def _build(seed: int = 0, **text_overrides) -> Qwen4ExpHpForConditionalGeneration:
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        torch.manual_seed(seed)
        model = Qwen4ExpHpForConditionalGeneration(_config(**text_overrides))
    finally:
        torch.set_default_dtype(prev)
    return model


def _fake_ep_state(model, ep_size: int = 1, ep_rank: int = 0):
    """load_checkpoint_hp 读 model._ep_size / _ep_rank，正常由 apply_hp 设置。"""
    model._ep_size = ep_size
    model._ep_rank = ep_rank
    return model


# ---------------------------------------------------------------------------
# 校验与失败路径
# ---------------------------------------------------------------------------


def test_missing_config_json_raises() -> None:
    """transformers 对缺 config.json 的目录会静默返回默认值，所以必须显式报错。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="config.json"):
            _read_weight_map(pathlib.Path(tmpdir))


def test_missing_index_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        (pathlib.Path(tmpdir) / "config.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="index"):
            _read_weight_map(pathlib.Path(tmpdir))


def test_save_requires_orig_ckpt_dir() -> None:
    model = _fake_ep_state(_build())
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="orig_ckpt_dir"):
            save_checkpoint_hp(model, tmpdir)


def test_save_requires_loaded_mtp_when_preserving() -> None:
    model = _fake_ep_state(_build())
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="loaded qwen4_exp MTP"):
            save_checkpoint_hp(model, tmpdir, orig_ckpt_dir=tmpdir, preserve_mtp=True)


@pytest.mark.parametrize("mismatch", ["missing", "unexpected"])
def test_load_rejects_mtp_key_mismatch(tmp_path, mismatch: str) -> None:
    model = _fake_ep_state(
        _build(
            mtp={"layer_types": ["full_attention"], "num_hidden_layers": 1},
            mtp_num_hidden_layers=1,
        )
    )
    weight_map = {
        key: "unopened.safetensors"
        for key in model.state_dict() if key.startswith("mtp.")
    }
    if mismatch == "missing":
        del weight_map["mtp.fc_hidden.weight"]
    else:
        weight_map["mtp.unexpected.weight"] = "unopened.safetensors"
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    with pytest.raises(RuntimeError, match=rf"{mismatch}=\['mtp\."):
        load_checkpoint_hp(model, str(tmp_path))


def test_save_preserves_loaded_mtp_keys() -> None:
    model = _fake_ep_state(
        _build(
            mtp={"layer_types": ["full_attention"], "num_hidden_layers": 1},
            mtp_num_hidden_layers=1,
        )
    )
    swap_parallel_modules(model, attn_backend="dense")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        origin = tmp / "origin"
        origin.mkdir()
        _write_source_dir(origin)
        out = tmp / "out"
        save_checkpoint_hp(model, str(out), orig_ckpt_dir=str(origin), preserve_mtp=True)
        weight_map = _read_weight_map(out)
    assert len([key for key in weight_map if key.startswith("mtp.")]) == 31


@pytest.mark.parametrize(
    ("model_arch", "has_mtp", "preserve_mtp"),
    [
        ("qwen4_exp", False, False),
        ("qwen4_exp", True, True),
        ("deepseek_v4", False, True),
    ],
)
@pytest.mark.parametrize("distributed_save", [False, True])
def test_checkpoint_mixin_saves_training_state(
    monkeypatch,
    model_arch: str,
    has_mtp: bool,
    preserve_mtp: bool,
    distributed_save: bool,
) -> None:
    class RecordingModel(HpModule):
        def save_checkpoint_hp(self, save_path: str, **kwargs) -> None:
            self.save_path = save_path
            self.kwargs = kwargs

    with tempfile.TemporaryDirectory() as tmpdir:
        dcp_calls = []
        monkeypatch.setattr(
            "gpatch_v4.training_backend.fsdp2_backend.mixin.checkpoint.save_checkpoint",
            lambda *_args, **kwargs: dcp_calls.append(kwargs),
        )
        engine = CheckpointMixin()
        engine.model = RecordingModel()
        engine.model.mtp = object() if has_mtp else None
        engine.model._ep_rank = 0
        engine.optimizer = None
        engine.lr_scheduler = None
        checkpoint_config = CheckpointConfig(save_ckpt_path=tmpdir)
        assert checkpoint_config.save_distributed_ep_shard is False
        checkpoint_config.save_distributed_ep_shard = distributed_save
        engine.config = SimpleNamespace(
            checkpoint=checkpoint_config,
            training=SimpleNamespace(enable_mtp=False, enable_dspark=False),
            policy=SimpleNamespace(model_arch=model_arch, hf_model_path="/source"),
        )
        engine.policy_config = engine.config.policy

        engine.save_checkpoint(2)

    assert engine.model.save_path == str(pathlib.Path(tmpdir) / "hf" / "2")
    assert engine.model.kwargs == {
        "orig_ckpt_dir": "/source",
        "preserve_mtp": preserve_mtp,
        "save_distributed_ep_shard": distributed_save,
    }
    assert dcp_calls == [{
        "global_step": 2,
        "save_dcp_model": False,
        "ep_rank": 0,
    }]


def test_checkpoint_mixin_uses_export_hf_save_path(monkeypatch) -> None:
    """配了 export_hf_save_path 时，HF 权重写到该目录下的 step 子目录。"""

    class RecordingModel(HpModule):
        def save_checkpoint_hp(self, save_path: str, **kwargs) -> None:
            self.save_path = save_path

    with tempfile.TemporaryDirectory() as tmpdir:
        root = pathlib.Path(tmpdir)
        monkeypatch.setattr(
            "gpatch_v4.training_backend.fsdp2_backend.mixin.checkpoint.save_checkpoint",
            lambda *_args, **_kwargs: None,
        )
        engine = CheckpointMixin()
        engine.model = RecordingModel()
        engine.model.mtp = None
        engine.model._ep_rank = 0
        engine.optimizer = None
        engine.lr_scheduler = None
        engine.config = SimpleNamespace(
            checkpoint=CheckpointConfig(
                save_ckpt_path=str(root / "train"),
                export_hf_save_path=str(root / "hf_export"),
            ),
            training=SimpleNamespace(enable_mtp=False, enable_dspark=False),
            policy=SimpleNamespace(model_arch="qwen4_exp", hf_model_path="/source"),
        )
        engine.policy_config = engine.config.policy

        engine.save_checkpoint(2)

    assert engine.model.save_path == str(root / "hf_export" / "2")


# ---------------------------------------------------------------------------
# 往返
# ---------------------------------------------------------------------------


def _write_source_dir(directory: pathlib.Path) -> None:
    _config().save_pretrained(directory)


def test_save_then_load_round_trip_preserves_weights() -> None:
    source = _fake_ep_state(_build(seed=1))
    swap_parallel_modules(source, attn_backend="dense")
    source.model.language_model.layers[1].ple.ple_embedding.ngram_embedding.weight.data.normal_()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        origin = tmp / "origin"
        origin.mkdir()
        _write_source_dir(origin)
        out = tmp / "out"

        save_checkpoint_hp(source, str(out), orig_ckpt_dir=str(origin))
        assert (out / "model.safetensors.index.json").is_file()
        weight_map = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
        assert "model.language_model.embed_tokens.weight" in weight_map
        assert "model.visual.patch_embed.proj.weight" in weight_map
        assert not any(key.startswith("model.layers.") for key in weight_map)
        assert (out / "config.json").is_file()
        shards = sorted(p.name for p in out.glob("*.safetensors"))
        assert shards and all("-of-" in s and "PENDING" not in s for s in shards)

        target = _fake_ep_state(_build(seed=2))
        swap_parallel_modules(target, attn_backend="dense")
        load_checkpoint_hp(target, str(out))

    source_state = source.state_dict()
    target_state = target.state_dict()
    assert set(source_state) == set(target_state)
    for key, expected in source_state.items():
        got = target_state[key]
        if expected.dtype.is_floating_point:
            # 落盘是 bf16，所以按 bf16 精度比较
            torch.testing.assert_close(
                got.to(torch.bfloat16).float().cpu(),
                expected.to(torch.bfloat16).float().cpu(),
                rtol=0, atol=0, msg=lambda m, k=key: f"{k}: {m}"
            )
        else:
            torch.testing.assert_close(got.cpu(), expected.cpu(), rtol=0, atol=0)


def test_load_skips_engram_when_debug_truncation_drops_its_layer() -> None:
    source = _fake_ep_state(_build(seed=1))
    swap_parallel_modules(source, attn_backend="dense")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = pathlib.Path(tmpdir)
        origin = root / "origin"
        origin.mkdir()
        _write_source_dir(origin)
        checkpoint = root / "checkpoint"
        save_checkpoint_hp(source, str(checkpoint), orig_ckpt_dir=str(origin))

        target = _fake_ep_state(_build(seed=2))
        target.model.language_model.layers[1].ple = None
        load_checkpoint_hp(target, str(checkpoint))

    assert not any("ngram_embedding" in key for key in target.state_dict())


def test_load_recomputes_non_persistent_rope_buffers() -> None:
    """rope 的 inv_freq 是 persistent=False，不在 checkpoint 里，必须重算而不是留 meta。"""
    source = _fake_ep_state(_build(seed=1))
    swap_parallel_modules(source, attn_backend="dense")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        origin = tmp / "origin"
        origin.mkdir()
        _write_source_dir(origin)
        out = tmp / "out"
        save_checkpoint_hp(source, str(out), orig_ckpt_dir=str(origin))

        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            with torch.device("meta"):
                target = Qwen4ExpHpForConditionalGeneration(_config())
        finally:
            torch.set_default_dtype(prev)
        swap_parallel_modules(target, attn_backend="dense")
        _fake_ep_state(target)
        # `to_empty` 把 buffer 也换成未初始化内存（不再是 meta）。loader 必须无条件重算
        # 非持久化 buffer，而不是只在 is_meta 时重算——否则这里会留下垃圾值。
        target.to_empty(device="cpu")
        load_checkpoint_hp(target, str(out))

    rotary = target.model.language_model.rotary_emb
    assert not rotary.inv_freq.is_meta
    assert not rotary.original_inv_freq.is_meta
    torch.testing.assert_close(rotary.inv_freq.cpu(), source.model.language_model.rotary_emb.inv_freq.cpu())
    torch.testing.assert_close(
        rotary.original_inv_freq.cpu(), source.model.language_model.rotary_emb.inv_freq.cpu()
    )


def test_unhandled_meta_buffer_raises() -> None:
    """新增了 persistent=False 的 buffer 却忘了处理时，必须报错而不是零初始化。"""
    from gpatch_v4.models.qwen4_exp.checkpoint import _materialize_meta_buffers

    model = _build()
    model.model.language_model.layers[0].register_buffer(
        "surprise", torch.empty(3, device="meta"), persistent=False
    )
    with pytest.raises(RuntimeError, match="unhandled meta buffers"):
        _materialize_meta_buffers(model)


# ---------------------------------------------------------------------------
# 多 rank：EP 切片 + Engram owner 行区间
# ---------------------------------------------------------------------------


def _sharded_load_worker(
    rank: int, world_size: int, init_file: str, ckpt_dir: str, truth_path: str
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        # Ground truth comes from a file the parent wrote. Rebuilding a reference model
        # *here* would be wrong: `dist` is initialized, so its Engram table would be
        # owner-sharded too and `full[start:start+local]` would slice an already-local
        # tensor.
        reference_state = torch.load(truth_path, map_location="cpu", weights_only=True)

        model = _build(seed=9)
        swap_parallel_modules(
            model, attn_backend="dense", engram_group=dist.group.WORLD
        )
        # 手工设置 EP 状态并切 expert 权重（正常由 apply_hp 做）
        _fake_ep_state(model, ep_size=world_size, ep_rank=rank)
        num_local = NUM_EXPERTS // world_size
        for layer in model.model.language_model.layers:
            for name in ("gate_up_proj", "down_proj"):
                param = getattr(layer.mlp.experts, name)
                setattr(
                    layer.mlp.experts,
                    name,
                    torch.nn.Parameter(param.data[:num_local].clone()),
                )
            layer.mlp.experts.configure_ep(dist.group.WORLD)

        load_checkpoint_hp(model, ckpt_dir)

        # expert：本 rank 只应拿到自己那一段
        for layer_idx, layer in enumerate(model.model.language_model.layers):
            for name in ("gate_up_proj", "down_proj"):
                got = getattr(layer.mlp.experts, name)
                expected = reference_state[f"model.language_model.layers.{layer_idx}.mlp.experts.{name}"]
                expected = expected[rank * num_local:(rank + 1) * num_local]
                torch.testing.assert_close(
                    got.to(torch.bfloat16).float().cpu(),
                    expected.to(torch.bfloat16).float().cpu(),
                    rtol=0, atol=0
                )

        # Engram：本 rank 只应拿到自己拥有的行
        table = model.model.language_model.layers[1].ple.ple_embedding.ngram_embedding
        full = reference_state["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight"]
        expected_rows = full[table.global_row_start:table.global_row_start + table.local_rows]
        torch.testing.assert_close(
            table.weight.to(torch.bfloat16).float().cpu(),
            expected_rows.to(torch.bfloat16).float().cpu(),
            rtol=0, atol=0
        )

        # 非切分参数应与参考完全一致
        torch.testing.assert_close(
            model.lm_head.weight.to(torch.bfloat16).float().cpu(),
            reference_state["lm_head.weight"].to(torch.bfloat16).float().cpu(),
            rtol=0, atol=0,
        )
    finally:
        dist.destroy_process_group()


def test_sharded_load_gives_each_rank_its_own_slice() -> None:
    world_size = 4
    source = _fake_ep_state(_build(seed=1))
    swap_parallel_modules(source, attn_backend="dense")
    source.model.language_model.layers[1].ple.ple_embedding.ngram_embedding.weight.data.normal_()

    previous_threads = torch.get_num_threads()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            origin = tmp / "origin"
            origin.mkdir()
            _write_source_dir(origin)
            ckpt = tmp / "ckpt"
            save_checkpoint_hp(source, str(ckpt), orig_ckpt_dir=str(origin))
            truth = tmp / "truth.pt"
            torch.save(source.state_dict(), truth)

            init_file = str(tmp / "pg_init")
            mp.start_processes(
                _sharded_load_worker,
                args=(world_size, init_file, str(ckpt), str(truth)),
                nprocs=world_size,
                join=True,
                start_method="spawn",
            )
    finally:
        torch.set_num_threads(previous_threads)


def _sharded_save_round_trip_worker(
    rank: int, world_size: int, init_file: str, origin_dir: str, ckpt_dir: str
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        mesh = init_device_mesh(
            "cpu", (world_size, 1), mesh_dim_names=("ep", "ep_fsdp")
        )
        ep_mesh = mesh["ep"]
        ep_fsdp_mesh = mesh["ep_fsdp"]
        num_local = NUM_EXPERTS // world_size

        source = _build(seed=1)
        swap_parallel_modules(
            source, attn_backend="dense", engram_group=dist.group.WORLD
        )
        _fake_ep_state(source, ep_size=world_size, ep_rank=rank)
        source._ep_group = ep_mesh.get_group()
        source._ep_fsdp_mesh = ep_fsdp_mesh
        for layer in source.model.language_model.layers:
            for name in ("gate_up_proj", "down_proj"):
                param = getattr(layer.mlp.experts, name)
                local = param.data[rank * num_local:(rank + 1) * num_local].clone()
                sharded = distribute_tensor(local, ep_fsdp_mesh, [Shard(0)])
                setattr(layer.mlp.experts, name, torch.nn.Parameter(sharded))
            layer.mlp.experts.configure_ep(ep_mesh.get_group())

        save_checkpoint_hp(source, ckpt_dir, orig_ckpt_dir=origin_dir)

        if rank == 0:
            from safetensors import safe_open

            weight_map = json.loads(
                (pathlib.Path(ckpt_dir) / "model.safetensors.index.json").read_text()
            )["weight_map"]
            for disk_key, shard in weight_map.items():
                if disk_key.endswith((".experts.gate_up_proj", ".experts.down_proj")):
                    with safe_open(
                        str(pathlib.Path(ckpt_dir) / shard), framework="pt", device="cpu"
                    ) as handle:
                        assert handle.get_slice(disk_key).get_shape()[0] == NUM_EXPERTS

        target = _build(seed=9)
        swap_parallel_modules(
            target, attn_backend="dense", engram_group=dist.group.WORLD
        )
        _fake_ep_state(target, ep_size=world_size, ep_rank=rank)
        target._ep_group = ep_mesh.get_group()
        target._ep_fsdp_mesh = ep_fsdp_mesh
        for layer in target.model.language_model.layers:
            for name in ("gate_up_proj", "down_proj"):
                param = getattr(layer.mlp.experts, name)
                local = param.data[:num_local].clone()
                sharded = distribute_tensor(local, ep_fsdp_mesh, [Shard(0)])
                setattr(layer.mlp.experts, name, torch.nn.Parameter(sharded))
            layer.mlp.experts.configure_ep(ep_mesh.get_group())

        load_checkpoint_hp(target, ckpt_dir)

        source_state = source.state_dict()
        target_state = target.state_dict()
        assert set(source_state) == set(target_state)
        for key, expected in source_state.items():
            got = target_state[key]
            if isinstance(got, DTensor):
                got = got.to_local()
                expected = expected.to_local()
            if expected.dtype.is_floating_point:
                got = got.to(torch.bfloat16).float()
                expected = expected.to(torch.bfloat16).float()
            torch.testing.assert_close(got.cpu(), expected.cpu(), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_sharded_save_then_load_preserves_all_experts() -> None:
    previous_threads = torch.get_num_threads()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            origin = tmp / "origin"
            origin.mkdir()
            _write_source_dir(origin)
            mp.start_processes(
                _sharded_save_round_trip_worker,
                args=(2, str(tmp / "pg_init"), str(origin), str(tmp / "ckpt")),
                nprocs=2,
                join=True,
                start_method="spawn",
            )
    finally:
        torch.set_num_threads(previous_threads)


def _partial_shard_worker(
    rank: int, world_size: int, init_file: str, ckpt_dir: str, truth_path: str
) -> None:
    """每个 rank 只拥有 checkpoint 里**某个分片的一部分**时，也必须正确加载。

    这是我们的 loader 比参考实现更宽松的地方：它在全局行空间里求交集，所以 owner 行区间
    和 checkpoint 分片边界不需要对齐。（NVIDIA 那边要求整分片对齐，因为它是把 view
    直写进自己的 DTensor 分片，而不是拷贝重叠区间——那条约束不适用于这里。）
    """
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        truth = torch.load(truth_path, map_location="cpu", weights_only=True)
        full = truth["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight"]

        model = _build(seed=9)
        swap_parallel_modules(model, attn_backend="dense", engram_group=dist.group.WORLD)
        _fake_ep_state(model, ep_size=1, ep_rank=0)
        load_checkpoint_hp(model, ckpt_dir)

        table = model.model.language_model.layers[1].ple.ple_embedding.ngram_embedding
        expected = full[table.global_row_start:table.global_row_start + table.local_rows]
        torch.testing.assert_close(
            table.weight.to(torch.bfloat16).float(), expected.to(torch.bfloat16).float(),
            rtol=0, atol=0
        )
    finally:
        dist.destroy_process_group()


def test_engram_loads_when_owner_range_straddles_checkpoint_shards() -> None:
    """单进程保存 -> Engram 只有 1 个分片；用 2 个 rank 读，每个 rank 只要半个分片。"""
    source = _fake_ep_state(_build(seed=1))
    swap_parallel_modules(source, attn_backend="dense")
    source.model.language_model.layers[1].ple.ple_embedding.ngram_embedding.weight.data.normal_()

    previous_threads = torch.get_num_threads()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            origin = tmp / "origin"
            origin.mkdir()
            _write_source_dir(origin)
            ckpt = tmp / "ckpt"
            save_checkpoint_hp(source, str(ckpt), orig_ckpt_dir=str(origin))
            index = json.loads((ckpt / "model.safetensors.index.json").read_text())
            shards = [
                k for k in index["weight_map"] if "ngram_embedding.shard_" in k
            ]
            assert len(shards) == 4, shards  # HF split_ngram_parts, not world_size
            truth = tmp / "truth.pt"
            torch.save(source.state_dict(), truth)

            mp.start_processes(
                _partial_shard_worker,
                args=(2, str(tmp / "pg_init"), str(ckpt), str(truth)),
                nprocs=2,
                join=True,
                start_method="spawn",
            )
    finally:
        torch.set_num_threads(previous_threads)
