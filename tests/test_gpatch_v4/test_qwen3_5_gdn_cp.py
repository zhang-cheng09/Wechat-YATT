"""THD + CP GDN vs Transformers eager GDN.

Each case runs ``Qwen3_5GatedDeltaNet`` (Transformers eager) on every packed
document independently, then reruns ``Qwen3_5CPGatedDeltaNet`` under zigzag
THD CP (``headwise`` / ``chunkwise``), gathers the shards, and compares.

Usage::

    cd /work/wepsdl/gcore/gcore_fsdp_opd_qwen35_4B
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    pytest -v -s --timeout=600 tests/test_gpatch_v4/test_qwen3_5_gdn_cp.py
"""

import importlib.util
import os
import socket
import unittest
from types import SimpleNamespace

import ray
import torch
import torch.nn.functional as F

from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

_FLA_AVAILABLE = importlib.util.find_spec("fla") is not None
requires_fla = unittest.skipUnless(_FLA_AVAILABLE, "flash-linear-attention (`fla`) not installed")

NUM_GPUS_NEEDED = 2
HIDDEN_SIZE = 256
LINEAR_KEY_HEAD_DIM = 64
LINEAR_VALUE_HEAD_DIM = 64
LINEAR_NUM_KEY_HEADS = 4
LINEAR_NUM_VALUE_HEADS = 8
THD_REAL_LENGTHS = [97, 181]
# Transformers eager GDN is the python chunk kernel; CP paths use FLA.
# bf16 max error sits around 5e-3..7e-3, so keep cosine as the layout check.
ATOL = 1e-2
RTOL = 1e-2


def _gdn_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        linear_num_value_heads=LINEAR_NUM_VALUE_HEADS,
        linear_num_key_heads=LINEAR_NUM_KEY_HEADS,
        linear_key_head_dim=LINEAR_KEY_HEAD_DIM,
        linear_value_head_dim=LINEAR_VALUE_HEAD_DIM,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"],
    )


def _copy_state(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    assert set(src_state) == set(dst_state), (
        f"GDN state keys changed: missing={sorted(set(src_state) - set(dst_state))} "
        f"unexpected={sorted(set(dst_state) - set(src_state))}"
    )
    dst.load_state_dict(src_state)


def _gather_thd(local: torch.Tensor, cu_seqlens_padded: torch.Tensor, group) -> torch.Tensor:
    from gpatch_v4.models.qwen3_5.cp import qwen3_5_cp_local_index

    cp_size = group.size()
    gathered = [torch.empty_like(local) for _ in range(cp_size)]
    torch.distributed.all_gather(gathered, local.contiguous(), group=group)
    global_length = local.shape[1] * cp_size
    output = local.new_empty(local.shape[0], global_length, *local.shape[2:])
    for rank, part in enumerate(gathered):
        index = qwen3_5_cp_local_index(
            rank,
            cp_size,
            global_length,
            local.device,
            cu_seqlens_padded=cu_seqlens_padded,
        )
        output.index_copy_(1, index, part)
    return output


@ray.remote(num_gpus=1)
def _gdn_thd_cp_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    linear_cp_comm_type: str,
    real_lengths: list[int],
):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    torch.cuda.set_device(0)

    from megatron.core.jit import disable_jit_fuser

    disable_jit_fuser()

    from gpatch_v4.models.qwen3_5.cp import (
        Qwen3_5CPContext,
        qwen3_5_cp_local_index,
        qwen3_5_pack_sequences,
        qwen3_5_packed_padding_mask,
    )
    from gpatch_v4.models.qwen3_5.gdn import Qwen3_5CPGatedDeltaNet
    from gpatch_v4.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    assert linear_cp_comm_type in ("headwise", "chunkwise")
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", torch.cuda.current_device())
    cp_group = torch.distributed.group.WORLD
    cp_size = world_size
    cp_rank = rank
    pad_multiple = 2 * cp_size

    dummy_ids = [torch.arange(length, device=device, dtype=torch.long) for length in real_lengths]
    packed_ids, _position_ids, packed_labels, packed = qwen3_5_pack_sequences(
        dummy_ids,
        dummy_ids,
        cp_size=cp_size,
        pad_multiple=pad_multiple,
        pad_token_id=0,
        label_ignore_index=-100,
    )
    del packed_ids, packed_labels
    global_padding_mask = qwen3_5_packed_padding_mask(packed)
    global_length = packed.total_seqlen
    valid = global_padding_mask.logical_not()

    config = _gdn_config()
    eager_gdn = Qwen3_5GatedDeltaNet(config, layer_idx=0).to(device=device, dtype=torch.bfloat16)
    eager_gdn.eval()
    for tensor in eager_gdn.parameters():
        torch.distributed.broadcast(tensor.data, src=0)

    full_hidden = torch.empty(
        (1, global_length, HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
    )
    if rank == 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(20260920)
        full_hidden.normal_(generator=generator)
        full_hidden.masked_fill_(global_padding_mask.unsqueeze(-1), 0)
    torch.distributed.broadcast(full_hidden, src=0)

    eager_output = torch.zeros_like(full_hidden)
    starts = packed.cu_seqlens_q_padded[:-1].tolist()
    real = packed.cu_seqlens_q.diff().tolist()
    with torch.no_grad():
        for start, length in zip(starts, real, strict=True):
            document = full_hidden[:, start:start + length]
            attention_mask = torch.ones(1, length, device=device, dtype=torch.bool)
            eager_output[:, start:start + length] = eager_gdn(
                document,
                cache_params=None,
                attention_mask=attention_mask,
            )

    cp_gdn = Qwen3_5CPGatedDeltaNet(config, layer_idx=0).to(device=device, dtype=torch.bfloat16)
    cp_gdn.eval()
    _copy_state(eager_gdn, cp_gdn)
    cp_gdn.linear_cp_comm_type = linear_cp_comm_type
    del eager_gdn

    local_index = qwen3_5_cp_local_index(
        cp_rank,
        cp_size,
        global_length,
        device,
        cu_seqlens_padded=packed.cu_seqlens_q_padded,
    )
    cp_context = Qwen3_5CPContext(
        group=cp_group,
        rank=cp_rank,
        size=cp_size,
        global_padding_mask=global_padding_mask,
        local_index=local_index,
        global_cu_seqlens=packed.cu_seqlens_q,
        global_cu_seqlens_padded=packed.cu_seqlens_q_padded,
    )
    local_hidden = cp_context.gather_local(full_hidden, dim=1)
    with torch.no_grad():
        local_output = cp_gdn(
            local_hidden,
            cache_params=None,
            attention_mask=cp_context.local_attention_mask,
            cp_context=cp_context,
        )
    gathered = _gather_thd(local_output, packed.cu_seqlens_q_padded, cp_group)

    eager_valid = eager_output.float().masked_select(valid.unsqueeze(-1)).reshape(-1, HIDDEN_SIZE)
    gathered_valid = gathered.float().masked_select(valid.unsqueeze(-1)).reshape(-1, HIDDEN_SIZE)
    difference = (eager_valid - gathered_valid).abs()
    max_abs = float(difference.max().item())
    mean_abs = float(difference.mean().item())
    cosine = float(
        F.cosine_similarity(eager_valid.flatten(), gathered_valid.flatten(), dim=0).item()
    )
    print(
        f"[rank {rank}] linear_cp_comm_type={linear_cp_comm_type} "
        f"cp={cp_size} global_len={global_length} local_len={local_hidden.shape[1]} "
        f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} cosine={cosine:.9f}"
    )
    torch.testing.assert_close(
        gathered_valid,
        eager_valid,
        atol=ATOL,
        rtol=RTOL,
        msg=(
            f"rank {rank} {linear_cp_comm_type} THD CP vs eager: "
            f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} cosine={cosine:.9f}"
        ),
    )

    result = {
        "rank": rank,
        "output_shape": list(local_output.shape),
        "output_dtype": str(local_output.dtype),
        "output_finite": bool(torch.isfinite(local_output).all()),
        "local_seq_len": local_hidden.shape[1],
        "global_seq_len": global_length,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "cosine": cosine,
    }
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return result


@requires_fla
@unittest.skipUnless(
    torch.cuda.device_count() >= 1,
    "needs a CUDA device on the pytest driver",
)
class TestQwen35GdnThdCp(unittest.TestCase):
    """Compare zigzag THD CP GDN to per-document Transformers eager GDN."""

    def setUp(self):
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < NUM_GPUS_NEEDED:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(
                f"need >= {NUM_GPUS_NEEDED} GPUs in Ray cluster, only {total_gpus}"
            )

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    def _run(self, linear_cp_comm_type: str, master_port: int) -> None:
        master_addr = socket.gethostbyname(ray.util.get_node_ip_address())
        futures = [
            _gdn_thd_cp_worker.remote(
                rank=rank,
                world_size=NUM_GPUS_NEEDED,
                master_addr=master_addr,
                master_port=master_port,
                linear_cp_comm_type=linear_cp_comm_type,
                real_lengths=THD_REAL_LENGTHS,
            ) for rank in range(NUM_GPUS_NEEDED)
        ]
        results = ray.get(futures)
        self.assertEqual(len(results), NUM_GPUS_NEEDED)
        for result in results:
            tag = f"rank {result['rank']} {linear_cp_comm_type}"
            self.assertEqual(result["output_shape"][0], 1, tag)
            self.assertEqual(result["output_shape"][1], result["local_seq_len"], tag)
            self.assertEqual(result["output_shape"][2], HIDDEN_SIZE, tag)
            self.assertEqual(result["output_dtype"], "torch.bfloat16", tag)
            self.assertTrue(result["output_finite"], f"{tag}: NaN/Inf")
            self.assertLessEqual(
                result["max_abs"],
                ATOL + RTOL * 10,
                f"{tag}: max_abs {result['max_abs']:.6e} vs eager",
            )
            self.assertGreater(result["cosine"], 0.99, tag)

    def test_thd_cp2_headwise(self):
        """CP=2 packed THD headwise vs per-document Transformers eager."""
        self._run("headwise", master_port=12461)

    def test_thd_cp2_chunkwise(self):
        """CP=2 packed THD chunkwise vs per-document Transformers eager."""
        self._run("chunkwise", master_port=12462)


if __name__ == "__main__":
    unittest.main()
