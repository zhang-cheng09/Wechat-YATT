import datetime
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors import safe_open
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor

from gpatch_v4.models.qwen4_exp import checkpoint


def _checkpoint_tensors() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    tensors = {}
    text_prefix = "model.language_model"
    for layer in range(2):
        for kind, rows in (("gate_up_proj", 6), ("down_proj", 3)):
            tensors[f"{text_prefix}.layers.{layer}.mlp.experts.{kind}"] = torch.randn(
                8, rows, 3, generator=generator,
            )
    tensors[f"{text_prefix}.norm.weight"] = torch.arange(3, dtype=torch.float32)
    tensors[f"{text_prefix}.hash_indices"] = torch.arange(5)
    tensors["model.visual.patch_embed.proj.weight"] = torch.arange(6, dtype=torch.float32).view(2, 3)
    return tensors


def _save_worker(
    rank: int, root: str, ep_size: int, distributed_save: bool,
) -> None:
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group(
        "gloo", init_method=f"file://{root / 'rendezvous'}",
        rank=rank, world_size=4, timeout=datetime.timedelta(seconds=60),
    )
    try:
        # 打乱 rank，且 rank 0 不一定是本 EP 分区的 writer。
        mesh = DeviceMesh(
            "cpu", torch.tensor([2, 0, 3, 1]).reshape(4 // ep_size, ep_size),
            mesh_dim_names=("ep_fsdp", "ep"),
        )
        dense_mesh = DeviceMesh("cpu", torch.arange(4))
        ep_rank = mesh["ep"].get_local_rank()
        model = torch.nn.Module()
        model._ep_size = ep_size
        model._ep_rank = ep_rank
        model._ep_fsdp_mesh = mesh["ep_fsdp"]
        model._ep_group = mesh["ep"].get_group()
        tensors = _checkpoint_tensors()
        for name, tensor in tensors.items():
            module = model
            *prefix, leaf = name.split(".")
            for part in prefix:
                if part not in module._modules:
                    module.add_module(part, torch.nn.Module())
                module = module._modules[part]
            if tensor.is_floating_point():
                is_expert = name.endswith(checkpoint._EXPERT_SUFFIXES)
                partition = tensor.chunk(ep_size)[ep_rank].contiguous() if is_expert else tensor
                if name.startswith("model.visual."):
                    value = tensor.clone()
                else:
                    value = distribute_tensor(
                        partition, mesh["ep_fsdp"] if is_expert else dense_mesh, [Shard(0)],
                    )
                module.register_parameter(leaf, torch.nn.Parameter(value))
            else:
                module.register_buffer(leaf, tensor)

        opened_partitions = []
        path_open = Path.open

        def record_open(path, mode="r", *args, **kwargs):
            if mode == "r+b":
                assert model._ep_fsdp_mesh.get_local_rank() == 0
                opened_partitions.append(path)
            return path_open(path, mode, *args, **kwargs)

        # 只替换 CUDA 搬运，真实执行 Gloo FSDP gather、分区写盘及原 loader。
        with (
            patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor),
            patch.object(torch.cuda, "synchronize"),
        ):
            with (
                patch.object(checkpoint.dist, "all_gather", wraps=dist.all_gather) as ep_gather,
                patch.object(Path, "open", record_open),
            ):
                options = {"save_distributed_ep_shard": True} if distributed_save else {}
                checkpoint.save_checkpoint_hp(model, str(root / "saved"), str(root / "source"), **options)
                assert ep_gather.call_count == (4 if ep_size > 1 and not distributed_save else 0)
            # 在线权重更新仍需要完整专家，不能随磁盘保存一起改成局部分片。
            model._ep_group = mesh["ep"].get_group()
            text_prefix = "model.language_model"
            key = f"{text_prefix}.layers.0.mlp.experts.gate_up_proj"
            full = checkpoint._gather_expert_for_save(model, model.state_dict()[key])
            torch.testing.assert_close(full, tensors[key], rtol=0, atol=0)
        is_writer = distributed_save and ep_size > 1 and model._ep_fsdp_mesh.get_local_rank() == 0
        assert len(opened_partitions) == (4 if is_writer else 0)

        index = json.loads((root / "saved/model.safetensors.index.json").read_text())
        expected = {
            key: value.to(torch.bfloat16)
            if value.is_floating_point() else value
            for key, value in tensors.items()
        }
        if rank == 0:
            assert (root / "saved/preprocessor_config.json").read_text() == (
                root / "source/preprocessor_config.json"
            ).read_text()
            assert set(index["weight_map"]) == set(expected)
            assert index["metadata"]["total_size"] == sum(
                tensor.numel() * tensor.element_size() for tensor in expected.values()
            )
            for key, filename in index["weight_map"].items():
                assert "PENDING" not in filename
                with safe_open(root / "saved" / filename, framework="pt") as reader:
                    torch.testing.assert_close(reader.get_tensor(key), expected[key], rtol=0, atol=0)

        before = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        with patch.object(checkpoint, "_load_device", lambda: torch.device("cpu")):
            checkpoint.load_checkpoint_hp(model, str(root / "saved"))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value.to(torch.bfloat16), before[key].to(torch.bfloat16), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("ep_size", [1, 2, 4])
@pytest.mark.parametrize("distributed_save", [False, True])
def test_ep_local_save_preserves_hf_layout(
    tmp_path: Path, ep_size: int, distributed_save: bool,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "preprocessor_config.json").write_text('{"processor_class": "Qwen3VLProcessor"}')
    mp.start_processes(
        _save_worker, args=(str(tmp_path), ep_size, distributed_save), nprocs=4,
        join=True, start_method="fork",
    )
