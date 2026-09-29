"""Real 2-GPU SFT loss parity for dynamic CP token/sequence normalization."""

import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


WORLD_SIZE = 2


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _worker(rank, master_port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=WORLD_SIZE)

    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices

    from gpatch_v4.training_backend.loss_factory import (
        FinetuneLossInput,
        _aggregate_packed_sft_ce,
        ce_loss,
    )

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        order="tp-cp-ep-dp-pp",
        dynamic_context_parallel=True,
        min_dynamic_context_parallel_size=1,
    )
    cp_group = parallel_state.get_dynamic_data_context_parallel_groups(
        group_size=WORLD_SIZE
    )

    generator = torch.Generator(device="cuda").manual_seed(20260904)
    vocab_size = 32
    full_logits_data = torch.randn(
        1,
        16,
        vocab_size,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    labels = torch.randint(
        0,
        vocab_size,
        (1, 16),
        generator=generator,
        device="cuda",
    )
    loss_mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]],
        device="cuda",
        dtype=torch.float32,
    )
    loss_weights = torch.linspace(0.5, 1.5, 16, device="cuda").view(1, 16)
    cu_seqlens = torch.tensor([0, 8, 16], device="cuda", dtype=torch.int32)
    config = SimpleNamespace(
        training=SimpleNamespace(
            cross_entropy_loss_fusion=False,
            cross_entropy_fusion_impl="native",
        )
    )

    baseline_logits = full_logits_data.clone().requires_grad_(True)
    baseline_token_losses = F.cross_entropy(
        baseline_logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
    ).view_as(loss_mask)
    weighted_baseline = baseline_token_losses * loss_weights
    baseline_sequence_sum = (
        (weighted_baseline[:, :8] * loss_mask[:, :8]).sum()
        / loss_mask[:, :8].sum().clamp(min=1)
        + (weighted_baseline[:, 8:] * loss_mask[:, 8:]).sum()
        / loss_mask[:, 8:].sum().clamp(min=1)
    )
    baseline_sequence_sum.backward()

    for partition_mode in ("zigzag", "contiguous"):
        if partition_mode == "zigzag":
            indices = get_thd_partitioned_indices(
                cu_seqlens,
                full_logits_data.shape[1],
                WORLD_SIZE,
                rank,
            )
        else:
            local_tokens = full_logits_data.shape[1] // WORLD_SIZE
            indices = torch.arange(
                rank * local_tokens,
                (rank + 1) * local_tokens,
                device="cuda",
            )

        local_logits = full_logits_data.index_select(1, indices).clone().requires_grad_(True)
        local_labels = labels.index_select(1, indices)
        local_mask = loss_mask.index_select(1, indices)
        local_weights = loss_weights.index_select(1, indices)
        local_losses = ce_loss(
            config,
            local_logits,
            local_labels,
            local_mask,
            return_src_loss=True,
        )
        loss_input = FinetuneLossInput(
            logits=local_logits,
            batch={"labels": local_labels, "loss_mask": local_mask},
            calculate_per_token_loss=False,
            cu_seqlens_padded=cu_seqlens,
            local_cp_size=WORLD_SIZE,
            cp_partition_mode=partition_mode,
            cp_group=cp_group,
        )
        sequence_sum, sample_count, token_sum, token_count = (
            _aggregate_packed_sft_ce(
                local_losses,
                local_mask,
                local_weights,
                loss_input,
            )
        )

        torch.testing.assert_close(sequence_sum, baseline_sequence_sum.detach())
        torch.testing.assert_close(sample_count, torch.tensor(2.0, device="cuda"))
        sequence_sum.backward()
        torch.testing.assert_close(
            local_logits.grad,
            baseline_logits.grad.index_select(1, indices),
            atol=1e-6,
            rtol=1e-5,
        )

        token_stats = torch.stack([token_sum, token_count])
        dist.all_reduce(token_stats, group=cp_group)
        torch.testing.assert_close(
            token_stats[0],
            (weighted_baseline * loss_mask).sum().detach(),
        )
        torch.testing.assert_close(token_stats[1], loss_mask.sum())

        token_logits = full_logits_data.index_select(1, indices).clone().requires_grad_(True)
        token_pair = ce_loss(
            config,
            token_logits,
            local_labels,
            local_mask,
            skip_cp_reduce=True,
            cp_group=cp_group,
            loss_weights=local_weights,
        )
        global_token_count = token_pair[1].clone()
        dist.all_reduce(global_token_count, group=cp_group)
        (token_pair[0] / global_token_count).backward()

        token_baseline_logits = full_logits_data.clone().requires_grad_(True)
        token_baseline_losses = F.cross_entropy(
            token_baseline_logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            reduction="none",
        ).view_as(loss_mask)
        token_baseline = (
            token_baseline_losses * loss_weights * loss_mask
        ).sum() / loss_mask.sum()
        token_baseline.backward()
        torch.testing.assert_close(
            token_logits.grad,
            token_baseline_logits.grad.index_select(1, indices),
            atol=1e-6,
            rtol=1e-5,
        )

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < WORLD_SIZE, reason="requires 2 GPUs")
def test_real_dynamic_cp_sft_both_normalizations():
    mp.spawn(
        _worker,
        args=(_find_free_port(),),
        nprocs=WORLD_SIZE,
        join=True,
    )
