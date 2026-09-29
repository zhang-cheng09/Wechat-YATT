import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors import safe_open
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor
from transformers import PretrainedConfig

from gpatch_v4.models.deepseek_v4 import checkpoint
from gpatch_v4.models.deepseek_v4.weight_export import iter_disk_checkpoint_tensors


def _expert_tensors() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    tensors = {}
    for prefix in ("layers.0", "layers.1", "mtp.layers.0"):
        for kind, rows in (("gate_up_proj", 256), ("down_proj", 128)):
            tensors[f"{prefix}.mlp.experts.{kind}"] = torch.randn(
                4, rows, 128, generator=generator
            )
    tensors["layers.0.input_layernorm.weight"] = torch.arange(128, dtype=torch.float32)
    return tensors


def _save_worker(rank: int, root: str, ep_size: int, dtype_format: str, expert_dtype: str, distributed_save: bool) -> None:
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group(
        "gloo", init_method=f"file://{root / 'rendezvous'}",
        rank=rank, world_size=4, timeout=datetime.timedelta(seconds=60),
    )
    try:
        # 打乱全局 rank，验证 writer 使用 mesh 坐标而非全局 rank 的整除关系。
        mesh = DeviceMesh(
            "cpu", torch.tensor([2, 0, 3, 1]).reshape(4 // ep_size, ep_size),
            mesh_dim_names=("ep_fsdp", "ep"),
        )
        dense_mesh = DeviceMesh("cpu", torch.arange(4))
        ep_rank = mesh["ep"].get_local_rank()
        state = {}
        for name, tensor in _expert_tensors().items():
            is_expert = ".experts." in name
            partition = tensor.chunk(ep_size)[ep_rank].contiguous() if is_expert else tensor
            state[f"model.{name}"] = distribute_tensor(
                partition, mesh["ep_fsdp"] if is_expert else dense_mesh, [Shard(0)]
            )
        model = SimpleNamespace(
            config=PretrainedConfig(expert_dtype=expert_dtype),
            _ep_size=ep_size, _ep_rank=ep_rank, _ep_fsdp_mesh=mesh["ep_fsdp"],
            _ep_group=mesh["ep"].get_group(),
            state_dict=lambda: state, can_generate=lambda: False,
        )
        tensor_factory = torch.tensor
        written_keys = []
        save_file = checkpoint.save_file

        def cpu_tensor(*args, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return tensor_factory(*args, **kwargs)

        def record_save(tensors, filename):
            written_keys.extend(tensors)
            save_file(tensors, filename)

        # 只替换 CUDA 控制标量/缓存操作；DTensor 通信、导出、写盘、索引均运行真实代码。
        with (
            patch.object(checkpoint.torch, "tensor", side_effect=cpu_tensor),
            patch.object(checkpoint.torch.cuda, "empty_cache"),
            patch.object(checkpoint.dist, "all_gather_into_tensor", side_effect=AssertionError("跨 EP gather")),
            patch.object(checkpoint.dist, "all_gather", wraps=dist.all_gather) as ep_gather,
            patch.object(checkpoint, "save_file", side_effect=record_save),
        ):
            options = {"save_distributed_ep_shard": True} if distributed_save else {}
            checkpoint._save_checkpoint_hp(
                model, str(root / "saved"), str(root / "source"),
                max_shard_size=64 * 1024, dtype_format=dtype_format, preserve_mtp=True,
                **options,
            )
            assert ep_gather.call_count == (6 if ep_size > 1 and not distributed_save else 0)
        # 跨 rank 重名会在总索引中被覆盖，因此同时核对实际 writer 的 key 清单。
        all_keys = [None] * 4
        dist.all_gather_object(all_keys, written_keys)
        if rank == 0:
            keys = [key for rank_keys in all_keys for key in rank_keys]
            assert len(keys) == len(set(keys))
            assert all(all_keys)
            index = json.loads((root / "saved/model.safetensors.index.json").read_text())
            expected = {}
            for name, tensor in _expert_tensors().items():
                expected.update(iter_disk_checkpoint_tensors(
                    name, tensor, dtype_format=dtype_format, expert_dtype=expert_dtype,
                ))
            assert set(keys) == set(index["weight_map"]) == set(expected)
            for key, filename in index["weight_map"].items():
                with safe_open(root / "saved" / filename, framework="pt") as reader:
                    actual = reader.get_tensor(key)
                # 比较原始字节，覆盖 FP4/FP8 权重及 scale，要求与完整专家导出完全一致。
                assert actual.dtype == expected[key].dtype
                assert actual.shape == expected[key].shape
                assert torch.equal(actual.view(torch.uint8), expected[key].contiguous().view(torch.uint8))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("ep_size", "dtype_format", "expert_dtype"),
    [(1, "bf16", "fp4"), (2, "bf16", "fp4"), (2, "quantized", "fp4"), (2, "quantized", "fp8")],
)
@pytest.mark.parametrize("distributed_save", [False, True])
def test_ep_partitions_save_complete_checkpoint(
    tmp_path: Path, ep_size: int, dtype_format: str, expert_dtype: str, distributed_save: bool,
) -> None:
    """多批 expert、dense、MTP 的独立写盘结果必须等于完整张量导出。"""
    source = tmp_path / "source"
    source.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (source / name).write_text("{}")
    torch.set_num_threads(1)
    mp.start_processes(
        _save_worker, args=(str(tmp_path), ep_size, dtype_format, expert_dtype, distributed_save),
        nprocs=4, join=True, start_method="fork",
    )
