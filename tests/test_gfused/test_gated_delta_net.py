"""Module-level forward test for ``GatedDeltaNet`` (Ray-based).

Each case first runs a TP=1/CP=1 eager forward on the full input, then
reruns with the requested ``(tp, sp, cp, qkv_format, linear_cp_mode)``,
gathers the parallel output back to the global layout, and compares it
to the eager result.

THD packing follows ``Megatron-LM/local/test_gdn_fla_thd_cp.py``: zigzag
metadata, ``prebuild_thd_cp_partition_routes``, then a TP sequence-parallel
slice.

The matrix:

================  =====  =====  ==========  ============  =========
``tp``            ``sp``  ``cp``  GPUs needed  ``qkv_format``  ``linear_cp_mode``
================  =====  =====  ==========  ============  =========
1                 False  1      1           sbhd          chunkwise
2                 False  1      2           sbhd          chunkwise
2                 True   1      2           sbhd          chunkwise
1                 False  2      2           sbhd          chunkwise
2                 False  2      4           sbhd          chunkwise
2                 True   2      4           sbhd          chunkwise
2                 True   2      4           thd           chunkwise
2                 True   2      4           thd           headwise
================  =====  =====  ==========  ============  =========

Manual-run only -- not in CI by default. Requires:

- Ray cluster with at least 4 GPUs
- ``flash-linear-attention`` (``fla``) installed

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=600 \\
        tests/test_gfused/test_gated_delta_net.py
"""

import importlib.util
import os
import socket
import unittest
from itertools import accumulate

import ray
import torch
import torch.nn.functional as F

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

_FLA_AVAILABLE = importlib.util.find_spec("fla") is not None
requires_fla = unittest.skipUnless(_FLA_AVAILABLE, "flash-linear-attention (`fla`) not installed")

NUM_GPUS_NEEDED = 4

SEQ_LENGTH = 64
MICRO_BATCH_SIZE = 2
HIDDEN_SIZE = 256
THD_SEQUENCE_LENGTHS = [128, 192]
ATOL = 5e-3
RTOL = 1e-2


def shard_thd_zigzag(
    full: torch.Tensor, sequence_lengths: list[int], cp_rank: int, cp_size: int
) -> torch.Tensor:
    """Shard each packed sequence with Megatron's zigzag CP layout."""
    local_sequences = []
    offset = 0
    for sequence_length in sequence_lengths:
        assert sequence_length % (2 * cp_size) == 0
        chunk_length = sequence_length // (2 * cp_size)
        sequence = full[offset : offset + sequence_length]
        first_chunk = cp_rank
        second_chunk = 2 * cp_size - cp_rank - 1
        local_sequences.append(
            torch.cat(
                [
                    sequence[first_chunk * chunk_length : (first_chunk + 1) * chunk_length],
                    sequence[second_chunk * chunk_length : (second_chunk + 1) * chunk_length],
                ],
                dim=0,
            )
        )
        offset += sequence_length
    assert offset == full.size(0)
    return torch.cat(local_sequences, dim=0).contiguous()


def gather_thd_zigzag(
    local: torch.Tensor,
    sequence_lengths: list[int],
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Gather rank-local zigzag packed output into full THD order."""
    cp_size = cp_group.size()
    if cp_size == 1:
        return local
    gathered = [torch.empty_like(local) for _ in range(cp_size)]
    torch.distributed.all_gather(gathered, local.contiguous(), group=cp_group)

    full_sequences = []
    local_offset = 0
    for sequence_length in sequence_lengths:
        assert sequence_length % (2 * cp_size) == 0
        chunk_length = sequence_length // (2 * cp_size)
        local_sequence_length = 2 * chunk_length
        chunks = [None] * (2 * cp_size)
        for cp_rank, rank_output in enumerate(gathered):
            local_sequence = rank_output[local_offset : local_offset + local_sequence_length]
            chunks[cp_rank] = local_sequence[:chunk_length]
            chunks[2 * cp_size - cp_rank - 1] = local_sequence[chunk_length:]
        assert all(chunk is not None for chunk in chunks)
        full_sequences.append(torch.cat(chunks, dim=0))
        local_offset += local_sequence_length
    assert local_offset == local.size(0)
    return torch.cat(full_sequences, dim=0).contiguous()


def shard_sbhd_zigzag(full: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    """Shard an SBHD tensor ``[S, B, H]`` with zigzag CP along the sequence dim."""
    seq_length = full.size(0)
    assert seq_length % (2 * cp_size) == 0
    chunk_length = seq_length // (2 * cp_size)
    full_view = full.view(2 * cp_size, chunk_length, *full.shape[1:])
    first_chunk = cp_rank
    second_chunk = 2 * cp_size - cp_rank - 1
    return torch.cat([full_view[first_chunk], full_view[second_chunk]], dim=0).contiguous()


def gather_sbhd_zigzag(
    local: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Gather zigzag SBHD shards back to the global sequence order."""
    cp_size = cp_group.size()
    if cp_size == 1:
        return local
    gathered = [torch.empty_like(local) for _ in range(cp_size)]
    torch.distributed.all_gather(gathered, local.contiguous(), group=cp_group)
    assert local.size(0) % 2 == 0
    chunk_length = local.size(0) // 2
    chunks = [None] * (2 * cp_size)
    for cp_rank, rank_output in enumerate(gathered):
        chunks[cp_rank] = rank_output[:chunk_length]
        chunks[2 * cp_size - cp_rank - 1] = rank_output[chunk_length:]
    assert all(chunk is not None for chunk in chunks)
    return torch.cat(chunks, dim=0).contiguous()


def gather_sequence_parallel(
    local: torch.Tensor, tp_group: torch.distributed.ProcessGroup, sp: bool
) -> torch.Tensor:
    """Concatenate sequence-parallel shards along dim 0. No-op when SP is off."""
    if not sp or tp_group.size() == 1:
        return local
    gathered = [torch.empty_like(local) for _ in range(tp_group.size())]
    torch.distributed.all_gather(gathered, local.contiguous(), group=tp_group)
    return torch.cat(gathered, dim=0)


def _shard_fused_sections(
    src: torch.Tensor, sections: tuple[int, ...], tp_rank: int, tp_size: int
) -> torch.Tensor:
    """Split each fused section along TP, then concat. Matches GDN checkpoint layout."""
    total = sum(sections)
    dims = [i for i, size in enumerate(src.shape) if size == total]
    assert len(dims) == 1, f"fused dim {total} not unique in {tuple(src.shape)}"
    dim = dims[0]
    local_pieces = []
    for piece in src.split(list(sections), dim=dim):
        assert piece.size(dim) % tp_size == 0, (tuple(piece.shape), dim, tp_size)
        chunk = piece.size(dim) // tp_size
        local_pieces.append(piece.narrow(dim, tp_rank * chunk, chunk))
    return torch.cat(local_pieces, dim=dim)


def copy_tp_shards_from_full(
    full_params: dict[str, torch.Tensor],
    full_buffers: dict[str, torch.Tensor],
    model: torch.nn.Module,
    tp_rank: int,
    tp_size: int,
    in_proj_sections: tuple[int, ...],
    conv_sections: tuple[int, ...],
) -> None:
    """Copy TP=1 tensors into a TP-sharded GDN.

    ``in_proj.weight`` and ``conv1d.weight`` are fused concatenations, so each
    logical section is split independently. Other TP parameters are a single
    contiguous split along the mismatched dim.
    """
    def _copy_shard(name: str, src: torch.Tensor, dst: torch.Tensor) -> None:
        src = src.to(device=dst.device, dtype=dst.dtype)
        if src.shape == dst.shape:
            dst.data.copy_(src)
            return
        if name.endswith("in_proj.weight"):
            sharded = _shard_fused_sections(src, in_proj_sections, tp_rank, tp_size)
        elif name.endswith("conv1d.weight") or name.endswith("conv1d.bias"):
            sharded = _shard_fused_sections(src, conv_sections, tp_rank, tp_size)
        else:
            split_dims = [i for i, (a, b) in enumerate(zip(src.shape, dst.shape)) if a != b]
            assert len(split_dims) == 1, (
                f"{name}: cannot infer TP split {tuple(src.shape)} -> {tuple(dst.shape)}"
            )
            dim = split_dims[0]
            shard_size = dst.size(dim)
            assert src.size(dim) == shard_size * tp_size, (
                f"{name}: src dim {dim} {src.size(dim)} != {shard_size} * {tp_size}"
            )
            sharded = src.narrow(dim, tp_rank * shard_size, shard_size)
        assert sharded.shape == dst.shape, (
            f"{name}: sharded {tuple(sharded.shape)} != dst {tuple(dst.shape)}"
        )
        dst.data.copy_(sharded)

    for name, dst in model.named_parameters():
        _copy_shard(name, full_params[name], dst)
    for name, dst in model.named_buffers():
        if dst.numel() == 0:
            continue
        _copy_shard(name, full_buffers[name], dst)


@ray.remote(num_gpus=1)
def _gdn_compare_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    tp: int,
    sp: bool,
    cp: int,
    linear_cp_mode: str = "chunkwise",
    qkv_format: str = "sbhd",
    sequence_lengths: list[int] | None = None,
    seq_length: int = SEQ_LENGTH,
    micro_batch_size: int = MICRO_BATCH_SIZE,
    hidden_size: int = HIDDEN_SIZE,
):
    """Run TP=1/CP=1 eager GDN, then the parallel layout, and compare.

    Parameters
    ----------
    rank, world_size, master_addr, master_port
        ``torch.distributed`` rendezvous for this worker.
    tp, sp, cp
        Parallel layout under test. ``sp`` enables sequence parallel on TP.
    linear_cp_mode
        ``"chunkwise"`` or ``"headwise"``.
    qkv_format
        ``"sbhd"`` or packed ``"thd"``.
    sequence_lengths
        Global packed lengths for THD. Ignored for SBHD.
    seq_length, micro_batch_size, hidden_size
        Global SBHD sizes. THD always uses batch 1 after packing.

    Returns
    -------
    dict
        Local parallel shape/dtype plus eager-comparison metrics.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    torch.cuda.set_device(0)

    from megatron.core.jit import disable_jit_fuser

    disable_jit_fuser()

    import torch.nn.functional as F_inner
    from megatron.core import parallel_state as mpu
    from megatron.core.context_parallel_layout import prebuild_thd_cp_partition_routes
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_experimental_attention_variant_module_spec,
    )
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    torch.distributed.init_process_group(backend="nccl")
    is_thd = qkv_format == "thd"
    assert qkv_format in ("sbhd", "thd")
    assert linear_cp_mode in ("chunkwise", "headwise")
    if is_thd:
        assert sequence_lengths is not None
        micro_batch_size = 1
    device = torch.device("cuda", torch.cuda.current_device())

    def _build_gdn(tp_size: int, sequence_parallel: bool, cp_size: int) -> GatedDeltaNet:
        pg_collection = ProcessGroupCollection(
            tp=mpu.get_tensor_model_parallel_group(),
            cp=mpu.get_context_parallel_group(),
            tp_cp=mpu.get_tensor_and_context_parallel_group(),
        )
        transformer_config = TransformerConfig(
            hidden_size=hidden_size,
            linear_conv_kernel_dim=2,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            layernorm_zero_centered_gamma=True,
            num_attention_heads=8,
            activation_func=F_inner.silu,
            bf16=True,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=sequence_parallel,
            context_parallel_size=cp_size,
            cp_partition_mode="zigzag",
            experimental_attention_variant="gdn",
            linear_attention_freq=[1],
            linear_cp_mode=linear_cp_mode,
            transformer_impl="transformer_engine",
            deterministic_mode=False,
        )
        gdn_submodules = get_experimental_attention_variant_module_spec(
            config=transformer_config,
        ).submodules
        return (
            GatedDeltaNet(
                transformer_config,
                submodules=gdn_submodules,
                layer_number=1,
                bias=False,
                conv_bias=False,
                conv_init=1.0,
                use_qk_l2norm=True,
                A_init_range=(1, 16),
                pg_collection=pg_collection,
            )
            .cuda()
            .bfloat16()
            .eval()
        )

    def _make_packed_seq_params(cp_size: int) -> PackedSeqParams:
        cu_seqlens = torch.tensor(
            [0, *accumulate(sequence_lengths)], device=device, dtype=torch.int32
        )
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=max(sequence_lengths),
            max_seqlen_kv=max(sequence_lengths),
            cp_partition_mode="zigzag",
            local_cp_size=cp_size,
            cp_group=mpu.get_context_parallel_group(),
        )
        prebuild_thd_cp_partition_routes(
            packed_seq_params,
            cp_group=mpu.get_context_parallel_group(),
        )
        return packed_seq_params

    def _forward(gdn: GatedDeltaNet, hidden_states: torch.Tensor, cp_size: int) -> torch.Tensor:
        packed_seq_params = _make_packed_seq_params(cp_size) if is_thd else None
        with torch.no_grad():
            output, _bias = gdn(
                hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
            )
        return output

    # -- Eager TP=1 CP=1 on the full input --
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(123)
    eager_gdn = _build_gdn(tp_size=1, sequence_parallel=False, cp_size=1)
    for tensor in [*eager_gdn.parameters(), *eager_gdn.buffers()]:
        if tensor.is_cuda and tensor.numel() > 0:
            torch.distributed.broadcast(tensor, src=0)

    if is_thd:
        full_hidden = torch.empty(
            (sum(sequence_lengths), 1, hidden_size),
            device=device,
            dtype=torch.bfloat16,
        )
    else:
        full_hidden = torch.empty(
            (seq_length, micro_batch_size, hidden_size),
            device=device,
            dtype=torch.bfloat16,
        )
    if rank == 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(20260912)
        full_hidden.normal_(generator=generator)
    torch.distributed.broadcast(full_hidden, src=0)

    eager_output = _forward(eager_gdn, full_hidden, cp_size=1)
    full_params = {name: tensor.detach().clone() for name, tensor in eager_gdn.named_parameters()}
    full_buffers = {name: tensor.detach().clone() for name, tensor in eager_gdn.named_buffers()}
    del eager_gdn

    torch.distributed.barrier()
    mpu.destroy_model_parallel()
    torch.cuda.empty_cache()
    torch.distributed.barrier()

    # -- Parallel layout under test --
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp,
    )
    model_parallel_cuda_manual_seed(123)
    parallel_gdn = _build_gdn(tp_size=tp, sequence_parallel=sp, cp_size=cp)
    qk_dim = 64 * 4
    v_dim = 64 * 8
    copy_tp_shards_from_full(
        full_params,
        full_buffers,
        parallel_gdn,
        tp_rank=mpu.get_tensor_model_parallel_rank(),
        tp_size=tp,
        in_proj_sections=(qk_dim, qk_dim, v_dim, v_dim, 8, 8),
        conv_sections=(qk_dim, qk_dim, v_dim),
    )
    del full_params, full_buffers

    if is_thd:
        hidden_states = shard_thd_zigzag(
            full_hidden,
            sequence_lengths,
            mpu.get_context_parallel_rank(),
            cp,
        )
    else:
        hidden_states = shard_sbhd_zigzag(
            full_hidden, mpu.get_context_parallel_rank(), cp
        )
    if sp:
        tp_rank = mpu.get_tensor_model_parallel_rank()
        sp_seg = hidden_states.shape[0] // tp
        assert hidden_states.shape[0] % tp == 0
        hidden_states = hidden_states[tp_rank * sp_seg : (tp_rank + 1) * sp_seg]

    local_output = _forward(parallel_gdn, hidden_states, cp_size=cp)
    gathered = gather_sequence_parallel(
        local_output, mpu.get_tensor_model_parallel_group(), sp
    )
    if is_thd:
        gathered = gather_thd_zigzag(
            gathered, sequence_lengths, mpu.get_context_parallel_group()
        )
    else:
        gathered = gather_sbhd_zigzag(gathered, mpu.get_context_parallel_group())

    difference = (eager_output.float() - gathered.float()).abs()
    max_abs = float(difference.max().item())
    mean_abs = float(difference.mean().item())
    cosine = float(
        F.cosine_similarity(
            eager_output.float().flatten(), gathered.float().flatten(), dim=0
        ).item()
    )
    print(
        f"[rank {rank}] tp={tp} sp={sp} cp={cp} "
        f"linear_cp_mode={linear_cp_mode} qkv_format={qkv_format} "
        f"local.shape={list(local_output.shape)} gathered.shape={list(gathered.shape)} "
        f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} cosine={cosine:.9f}"
    )
    torch.testing.assert_close(
        gathered,
        eager_output,
        atol=ATOL,
        rtol=RTOL,
        msg=(
            f"rank {rank} tp={tp} sp={sp} cp={cp} linear_cp_mode={linear_cp_mode} "
            f"qkv_format={qkv_format}: max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} "
            f"cosine={cosine:.9f}"
        ),
    )

    result = {
        "rank": rank,
        "output_shape": list(local_output.shape),
        "output_dtype": str(local_output.dtype),
        "output_finite": bool(torch.isfinite(local_output).all()),
        "expected_seq_len": hidden_states.shape[0],
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "cosine": cosine,
    }
    torch.distributed.barrier()
    mpu.destroy_model_parallel()
    torch.distributed.destroy_process_group()
    return result


@requires_fla
@unittest.skipUnless(
    torch.cuda.device_count() >= NUM_GPUS_NEEDED,
    f"needs {NUM_GPUS_NEEDED} GPUs, have {torch.cuda.device_count()}",
)
class TestGatedDeltaNetSmoke(unittest.TestCase):
    """Compare TP/SP/CP GDN forwards against a TP=1/CP=1 eager baseline."""

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

    def _run(
        self,
        tp: int,
        sp: bool,
        cp: int,
        master_port: int,
        linear_cp_mode: str = "chunkwise",
        qkv_format: str = "sbhd",
        sequence_lengths: list[int] | None = None,
    ) -> None:
        """Fan out ``tp * cp`` Ray workers and compare gathered output to eager."""
        world_size = tp * cp
        master_addr = socket.gethostbyname(ray.util.get_node_ip_address())
        sp_size = tp if sp else 1
        if qkv_format == "thd":
            assert sequence_lengths is not None
            micro_batch_size = 1
            expected_seq = sum(seq // cp for seq in sequence_lengths) // sp_size
            expected_batch = 1
        else:
            micro_batch_size = (
                1 if linear_cp_mode == "chunkwise" and cp > 1 else MICRO_BATCH_SIZE
            )
            expected_seq = SEQ_LENGTH // sp_size // cp
            expected_batch = micro_batch_size

        futures = [
            _gdn_compare_worker.remote(
                rank=r,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
                tp=tp,
                sp=sp,
                cp=cp,
                linear_cp_mode=linear_cp_mode,
                qkv_format=qkv_format,
                sequence_lengths=sequence_lengths,
                micro_batch_size=micro_batch_size,
            )
            for r in range(world_size)
        ]
        results = ray.get(futures)

        self.assertEqual(
            len(results),
            world_size,
            f"expected {world_size} results, got {len(results)}",
        )
        for r in results:
            tag = (
                f"rank {r['rank']} (tp={tp}, sp={sp}, cp={cp}, "
                f"linear_cp_mode={linear_cp_mode}, qkv_format={qkv_format})"
            )
            self.assertEqual(
                r["output_shape"][0],
                expected_seq,
                f"{tag}: seq dim {r['output_shape'][0]} != {expected_seq}",
            )
            self.assertEqual(
                r["output_shape"][1],
                expected_batch,
                f"{tag}: batch dim {r['output_shape'][1]} != {expected_batch}",
            )
            self.assertEqual(
                r["output_shape"][2],
                HIDDEN_SIZE,
                f"{tag}: hidden dim {r['output_shape'][2]} != {HIDDEN_SIZE}",
            )
            self.assertEqual(
                r["output_dtype"],
                "torch.bfloat16",
                f"{tag}: output dtype {r['output_dtype']} != torch.bfloat16",
            )
            self.assertTrue(
                r["output_finite"],
                f"{tag}: output contains NaN/Inf",
            )
            self.assertLessEqual(
                r["max_abs"],
                ATOL + RTOL * 10,
                f"{tag}: max_abs {r['max_abs']:.6e} vs eager",
            )

    def test_tp1_cp1(self):
        """TP=1 CP=1 SBHD vs its own eager baseline (1 GPU)."""
        self._run(tp=1, sp=False, cp=1, master_port=12361)

    def test_tp2_cp1(self):
        """TP=2 SBHD vs TP=1 CP=1 eager (2 GPUs)."""
        self._run(tp=2, sp=False, cp=1, master_port=12362)

    def test_tp2_sp_cp1(self):
        """TP=2 + SP SBHD vs TP=1 CP=1 eager (2 GPUs)."""
        self._run(tp=2, sp=True, cp=1, master_port=12363)

    def test_tp1_cp2(self):
        """CP=2 SBHD chunkwise vs TP=1 CP=1 eager (2 GPUs)."""
        self._run(tp=1, sp=False, cp=2, master_port=12364)

    def test_tp2_cp2(self):
        """TP=2 + CP=2 SBHD chunkwise vs TP=1 CP=1 eager (4 GPUs)."""
        self._run(tp=2, sp=False, cp=2, master_port=12365)

    def test_tp2_sp_cp2(self):
        """TP=2 + SP + CP=2 SBHD chunkwise vs TP=1 CP=1 eager (4 GPUs)."""
        self._run(tp=2, sp=True, cp=2, master_port=12366)

    def test_tp2_sp_cp2_thd_chunkwise(self):
        """TP=2 + SP + CP=2 packed THD chunkwise vs TP=1 CP=1 eager (4 GPUs)."""
        self._run(
            tp=2,
            sp=True,
            cp=2,
            master_port=12367,
            linear_cp_mode="chunkwise",
            qkv_format="thd",
            sequence_lengths=THD_SEQUENCE_LENGTHS,
        )

    def test_tp2_sp_cp2_thd_headwise(self):
        """TP=2 + SP + CP=2 packed THD headwise vs TP=1 CP=1 eager (4 GPUs)."""
        self._run(
            tp=2,
            sp=True,
            cp=2,
            master_port=12368,
            linear_cp_mode="headwise",
            qkv_format="thd",
            sequence_lengths=THD_SEQUENCE_LENGTHS,
        )


class TestLinearCpCommTypeRewrite(unittest.TestCase):
    """``linear_cp_comm_type=fla`` rewrite follows the current Megatron version."""

    def test_fla_follows_megatron_version(self):
        from megatron.core.transformer.transformer_config import TransformerConfig
        from omegaconf import OmegaConf

        from gpatch_v4.configs.config import FinetuneConfig
        from gpatch_v4.configs.utils import merge_hydra_config

        config = merge_hydra_config(
            FinetuneConfig,
            OmegaConf.create({
                "policy": {
                    "model_arch": "qwen3_5_moe",
                    "override_transformer_config": {"linear_cp_comm_type": "fla"},
                },
            }),
        )
        otc = config.policy.override_transformer_config
        if hasattr(TransformerConfig, "linear_cp_mode"):
            self.assertEqual(otc.get("linear_cp_mode"), "chunkwise", otc)
            self.assertNotIn("linear_cp_comm_type", otc)
        else:
            self.assertEqual(otc.get("linear_cp_comm_type"), "fla", otc)
            self.assertNotIn("linear_cp_mode", otc)


if __name__ == "__main__":
    unittest.main()
