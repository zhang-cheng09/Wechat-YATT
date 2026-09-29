"""Fused dyn-CP reroute with chunked all-to-all (2 procs / 2 GPUs, real NCCL)."""
from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gpatch_v4.utils.dynamic_cp_utils import reroute_samples_to_dcp_ranks_by_keys_fused

WORLD_SIZE = 2
BASE_LEN = 4096


class _TpSizeOne:

    def size(self) -> int:
        return 1


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _sample(gid: int) -> dict:
    n = BASE_LEN + gid * 128
    base = gid * 1_000_000
    return {
        "tokens": torch.arange(base, base + n, dtype=torch.int64),
        "loss_mask": torch.arange(base, base + n, dtype=torch.float32) + 0.25 * gid,
        "vision_data": torch.arange(base, base + 3 * n, dtype=torch.bfloat16),
        "original_seq_len": torch.tensor([n - 1], dtype=torch.int32),
    }


def _worker(rank: int, world_size: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        # 输入放 CPU，与真实训练一致；rank0 持有 gid{0,1}，rank1 持有 gid{2,3}
        all_samples = {gid: _sample(gid) for gid in range(4)}
        local_gids = [0, 1] if rank == 0 else [2, 3]
        # 交叉路由：rank0 <- {1,2}，rank1 <- {0,3}
        sample_id_groups = [[[1, 2], [0, 3]]]
        seqlens = {
            key: [(gid, all_samples[gid][key].numel()) for gid in range(4)]
            for key in all_samples[0]
        }
        args = (
            [all_samples[gid] for gid in local_gids],
            torch.tensor(local_gids, dtype=torch.int64),
            seqlens,
            sample_id_groups,
            torch.tensor([0, 2, 4], dtype=torch.int32),
            dist.group.WORLD,
            _TpSizeOne(),
            dist.group.WORLD,
            world_size,
        )
        single = reroute_samples_to_dcp_ranks_by_keys_fused(*args)

        n_a2a = {"n": 0}
        real_a2a = dist.all_to_all_single

        def counted(*a, **kw):
            n_a2a["n"] += 1
            return real_a2a(*a, **kw)

        dist.all_to_all_single = counted
        try:
            # 每轮 ~16KB，强制切成多轮
            chunked = reroute_samples_to_dcp_ranks_by_keys_fused(
                *args, a2a_chunk_gb=16 / (1 << 20)
            )
        finally:
            dist.all_to_all_single = real_a2a
        assert n_a2a["n"] > 1, f"expected multiple a2a rounds, got {n_a2a['n']}"

        expected_gids = sample_id_groups[0][rank]
        assert sorted(chunked) == sorted(single) == sorted(expected_gids)
        for gid in expected_gids:
            for key, exp in all_samples[gid].items():
                assert chunked[gid][key].device.type == "cpu"
                torch.testing.assert_close(chunked[gid][key], exp.reshape(-1))
                torch.testing.assert_close(chunked[gid][key], single[gid][key].cpu())
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"need >= {WORLD_SIZE} CUDA devices for real NCCL spawn",
)
def test_fused_reroute_chunked_a2a_matches_single():
    mp.spawn(_worker, args=(WORLD_SIZE, _find_free_port()), nprocs=WORLD_SIZE, join=True)
