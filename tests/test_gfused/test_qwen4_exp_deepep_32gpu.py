# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Qwen4-Exp 普通 DeepEP 的 32-GPU 原语与 eager-EP 数值对照。

原语测试没有黄金解，只检查 dispatch/combine 的 shape、finite 和梯度非零。
数值测试对照同一拓扑下的 eager EP：log-softmax / loss / per-param grad norm。
跨节点 DeepEP 的 raw logits max 会被词表 outlier 拉高，不作为硬阈值。

2 卡 dense-forward 对照仍在 ``test_qwen4_exp_apply_hp_gpu.py``。

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tasks/deepseek/V3/deepep_env.sh
    export PYTHONPATH="$RCDIR/transformers/src:$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    pytest -v -s --timeout=2400 tests/test_gfused/test_qwen4_exp_deepep_32gpu.py

跨节点 NVSHMEM 要求每台机器 hostname 唯一；本集群默认 ``TENCENT64.site`` 会触发 Fatal IPC。
"""

import importlib.util
import math
import os
import unittest

import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ray.util.placement_group import remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

from gpatch_v4.models.qwen4_exp import (
    Qwen4ExpConfig,
    Qwen4ExpHpForConditionalGeneration,
    apply_hp,
)
from gpatch_v4.models.qwen4_exp.deepep_a2a import (
    fused_combine,
    fused_dispatch,
    set_deepep_num_sms,
)
from gpatch_v4.orches.placement_group import _create_placement_group

MODEL_PATH = os.environ.get(
    "QWEN4_EXP_MODEL_PATH",
    "/mnt/ceph-hz1-csp/mm-base-plt2/user_edsonzhou/hf-hub/Qwen/Qwen3.8-Flash-Next",
)
NUM_GPUS = 32
EP_SIZE = 32
NUM_LAYERS = 4
SEQ_LEN = 256
RTOL = 0.06
GRAD_NORM_RTOL = 0.1
# log_softmax max is still outlier-sensitive across 32 ranks; 0.06 is for mean/p99.
LOGPROB_MAX_RTOL = 0.1

requires_deepep = unittest.skipUnless(
    importlib.util.find_spec("deep_ep") is not None,
    "deep_ep not installed",
)


@ray.remote(num_cpus=0, num_gpus=0)
def _get_node_ip() -> str:
    return ray.util.get_node_ip_address()


def _init_process_group(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


@ray.remote(num_gpus=1)
def _primitive_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> dict:
    _init_process_group(rank, world_size, master_addr, master_port)
    try:
        set_deepep_num_sms(24)
        # Internode kernel: hidden_int4 = hidden_bytes / sizeof(int4=16B), and
        # hidden_int4 % 32 == 0. For bf16 that is hidden_dim % 256 == 0.
        # hidden_dim=64 → hidden_int4=8, which aborted on 4-node EP32.
        hidden_dim = 2560
        num_tokens = 16
        num_experts = world_size * 2
        top_k = 2

        tokens = (
            torch.arange(
                num_tokens * hidden_dim,
                dtype=torch.bfloat16,
                device="cuda",
            )
            .reshape(num_tokens, hidden_dim)
            .add_(rank * 100)
        )
        tokens.requires_grad_(True)
        generator = torch.Generator(device="cuda").manual_seed(20260909 + rank)
        topk_idx = torch.randint(
            0,
            num_experts,
            (num_tokens, top_k),
            generator=generator,
            dtype=torch.int64,
            device="cuda",
        )
        topk_weights = torch.rand(
            num_tokens,
            top_k,
            generator=generator,
            dtype=torch.float32,
            device="cuda",
        )
        topk_weights = torch.softmax(topk_weights, dim=-1)
        topk_weights.requires_grad_(True)

        recv_tokens, recv_idx, recv_weights, tokens_per_expert, handle = (
            fused_dispatch(
                tokens,
                topk_idx,
                topk_weights,
                num_experts,
                dist.group.WORLD,
            )
        )
        row_idx, slot_idx = (recv_idx >= 0).nonzero(as_tuple=True)
        recv_output = recv_tokens.new_zeros(recv_tokens.shape)
        if row_idx.numel() > 0:
            expanded = recv_tokens.index_select(0, row_idx).float() * 2.0
            expanded = expanded * recv_weights[row_idx, slot_idx].unsqueeze(-1)
            recv_output.index_add_(0, row_idx, expanded.to(recv_output.dtype))

        combined = fused_combine(recv_output, dist.group.WORLD, handle)
        combined.float().sum().backward()
        return {
            "rank": rank,
            "combined_shape": list(combined.shape),
            "combined_finite": bool(torch.isfinite(combined).all()),
            "tokens_grad_finite": bool(torch.isfinite(tokens.grad).all()),
            "tokens_grad_norm": tokens.grad.float().norm().item(),
            "weights_grad_finite": bool(torch.isfinite(topk_weights.grad).all()),
            "weights_grad_norm": topk_weights.grad.float().norm().item(),
            "num_local_experts": tokens_per_expert.numel(),
        }
    finally:
        dist.destroy_process_group()


def _truncate_config(config):
    config.num_hidden_layers = NUM_LAYERS
    config.layer_types = list(config.layer_types[:NUM_LAYERS])
    config.ple_layer_ids = [
        layer_id for layer_id in config.ple_layer_ids
        if layer_id <= NUM_LAYERS
    ]
    config.use_cache = False
    return config


def _build_model(ep_backend: str) -> Qwen4ExpHpForConditionalGeneration:
    config = Qwen4ExpConfig.from_pretrained(MODEL_PATH)
    _truncate_config(config.text_config)
    mesh = init_device_mesh(
        "cuda",
        mesh_shape=(NUM_GPUS // EP_SIZE, EP_SIZE),
        mesh_dim_names=("ep_fsdp", "ep"),
    )

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model = Qwen4ExpHpForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(previous_dtype)

    model = apply_hp(
        model,
        mesh,
        attn_backend="dense",
        ep_backend=ep_backend,
    )
    model.load_checkpoint_hp(MODEL_PATH)
    return model


def _global_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    norms = {}
    ep_group = model._ep_group
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad
        if isinstance(grad, DTensor):
            grad = grad.to_local()
        norm_sq = grad.float().square().sum()
        if ".experts." in name:
            dist.all_reduce(norm_sq, group=ep_group)
        else:
            dist.all_reduce(norm_sq)
        norms[name] = norm_sq.sqrt().item()
    return norms


def _run_backend(
    ep_backend: str,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, float, float, dict[str, float]]:
    model = _build_model(ep_backend)
    model.train()
    output = model(input_ids=input_ids)
    logits = output.logits
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = input_ids[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    loss.backward()
    total_grad_norm = model.clip_grad_norm_(max_norm=2.0)
    if isinstance(total_grad_norm, torch.Tensor):
        total_grad_norm = total_grad_norm.item()
    grad_norms = _global_grad_norms(model)
    result = (
        logits.detach().cpu(),
        loss.item(),
        float(total_grad_norm),
        grad_norms,
    )
    del model, output, logits, loss
    torch.cuda.empty_cache()
    return result


def _logit_stats(eager: torch.Tensor, deepep: torch.Tensor) -> dict[str, float]:
    eager_f = eager.float()
    deepep_f = deepep.float()
    diff = (deepep_f - eager_f).abs()
    scale = max(eager_f.abs().max().item(), 1.0e-12)
    n = diff.numel()
    # CUDA quantile() rejects tensors with >= 2**24 elements (seq=256 * vocab).
    sample = diff.reshape(-1)
    quantile_limit = 1 << 24
    if sample.numel() >= quantile_limit:
        sample = sample[:: (sample.numel() + quantile_limit - 1) // quantile_limit]
    log_eager = torch.log_softmax(eager_f, dim=-1)
    log_deepep = torch.log_softmax(deepep_f, dim=-1)
    log_diff = (log_deepep - log_eager).abs()
    log_scale = max(log_eager.abs().max().item(), 1.0e-12)
    return {
        "logits_rel": diff.max().item() / scale,
        "logits_mean_rel": diff.mean().item() / scale,
        "logits_p99_rel": sample.quantile(0.99).item() / scale,
        "logits_p999_rel": sample.quantile(0.999).item() / scale,
        "logits_frac_gt_rtol": float((diff > RTOL * scale).sum().item()) / n,
        "logprob_rel": log_diff.max().item() / log_scale,
        "logprob_mean_rel": log_diff.mean().item() / log_scale,
        "eager_abs_max": eager_f.abs().max().item(),
        "diff_abs_max": diff.max().item(),
    }


@ray.remote(num_gpus=1)
def _parity_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> dict:
    _init_process_group(rank, world_size, master_addr, master_port)
    try:
        # EP=32 / DP=1：所有 rank 必须吃同一条序列，否则 FSDP 分片前向不自洽。
        config = Qwen4ExpConfig.from_pretrained(MODEL_PATH).get_text_config()
        generator = torch.Generator().manual_seed(42)
        input_ids = torch.randint(
            0,
            config.vocab_size,
            (1, SEQ_LEN),
            generator=generator,
        ).cuda()

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        eager_logits, eager_loss, eager_gnorm, eager_grad_norms = _run_backend(
            "eager",
            input_ids,
        )
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        deepep_logits, deepep_loss, deepep_gnorm, deepep_grad_norms = _run_backend(
            "deepep",
            input_ids,
        )

        stats = _logit_stats(eager_logits, deepep_logits)
        logits_rel = stats["logits_rel"]
        loss_rel = abs(deepep_loss - eager_loss) / max(abs(eager_loss), 1.0e-12)
        total_grad_rel = abs(deepep_gnorm - eager_gnorm) / max(eager_gnorm, 1.0e-12)

        missing = set(eager_grad_norms) ^ set(deepep_grad_norms)
        assert not missing, f"rank {rank}: grad key mismatch {sorted(missing)[:8]}"

        worst_grad_name = ""
        worst_grad_rel = 0.0
        per_param_rels = {}
        for name, eager_norm in eager_grad_norms.items():
            deepep_norm = deepep_grad_norms[name]
            if eager_norm <= 1.0e-8:
                assert deepep_norm <= 1.0e-7, (
                    f"rank {rank}: {name} eager grad is zero but DeepEP "
                    f"grad norm is {deepep_norm}"
                )
                per_param_rels[name] = 0.0
                continue
            grad_rel = abs(deepep_norm - eager_norm) / eager_norm
            per_param_rels[name] = grad_rel
            if grad_rel > worst_grad_rel:
                worst_grad_name = name
                worst_grad_rel = grad_rel

        return {
            "rank": rank,
            "logits_finite": bool(
                torch.isfinite(eager_logits).all()
                and torch.isfinite(deepep_logits).all()
            ),
            "eager_loss": eager_loss,
            "deepep_loss": deepep_loss,
            "eager_gnorm": eager_gnorm,
            "deepep_gnorm": deepep_gnorm,
            "logits_rel": logits_rel,
            "logits_mean_rel": stats["logits_mean_rel"],
            "logits_p99_rel": stats["logits_p99_rel"],
            "logits_p999_rel": stats["logits_p999_rel"],
            "logits_frac_gt_rtol": stats["logits_frac_gt_rtol"],
            "logprob_rel": stats["logprob_rel"],
            "logprob_mean_rel": stats["logprob_mean_rel"],
            "eager_abs_max": stats["eager_abs_max"],
            "diff_abs_max": stats["diff_abs_max"],
            "loss_rel": loss_rel,
            "total_grad_rel": total_grad_rel,
            "worst_grad_name": worst_grad_name,
            "worst_grad_rel": worst_grad_rel,
            "per_param_rels": per_param_rels,
        }
    finally:
        dist.destroy_process_group()


def _run_workers(worker, master_port: int) -> list[dict]:
    placement = _create_placement_group(NUM_GPUS)
    pg, bundle_indices = placement
    try:
        master_addr = ray.get(
            _get_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_indices[0],
                )
            ).remote()
        )
        futures = [
            worker.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_indices[rank],
                )
            ).remote(
                rank,
                NUM_GPUS,
                master_addr,
                master_port,
            )
            for rank in range(NUM_GPUS)
        ]
        return ray.get(futures)
    finally:
        remove_placement_group(pg)


@requires_deepep
class TestQwen4ExpDeepEP32Gpu(unittest.TestCase):
    def setUp(self):
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < NUM_GPUS:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(
                f"need >= {NUM_GPUS} GPUs in Ray cluster, got {total_gpus}"
            )

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    def test_dispatch_combine_autograd_ep32(self):
        results = _run_workers(_primitive_worker, master_port=12960)
        for result in results:
            self.assertEqual(result["combined_shape"], [16, 2560])
            self.assertTrue(result["combined_finite"])
            self.assertTrue(result["tokens_grad_finite"])
            self.assertGreater(result["tokens_grad_norm"], 0.0)
            self.assertTrue(result["weights_grad_finite"])
            self.assertGreater(result["weights_grad_norm"], 0.0)
            self.assertEqual(result["num_local_experts"], 2)

    def test_eager_ep32_matches_deepep_ep32(self):
        if not os.path.isdir(MODEL_PATH):
            self.skipTest(f"model path does not exist: {MODEL_PATH}")

        results = _run_workers(_parity_worker, master_port=12980)
        self.assertEqual(len(results), NUM_GPUS)
        rank0 = results[0]
        print(
            f"\n  eager loss              = {rank0['eager_loss']:.6f}\n"
            f"  deepep loss             = {rank0['deepep_loss']:.6f}\n"
            f"  loss rel_diff           = {rank0['loss_rel']:.6f}\n"
            f"  eager total_grad_norm   = {rank0['eager_gnorm']:.6f}\n"
            f"  deepep total_grad_norm  = {rank0['deepep_gnorm']:.6f}\n"
            f"  grad rel_diff           = {rank0['total_grad_rel']:.6f}\n"
            f"  logits max/p99/p999/mean rel = "
            f"{rank0['logits_rel']:.3e} / {rank0['logits_p99_rel']:.3e} / "
            f"{rank0['logits_p999_rel']:.3e} / {rank0['logits_mean_rel']:.3e}\n"
            f"  logits frac > {RTOL}     = {rank0['logits_frac_gt_rtol']:.3e}\n"
            f"  logprob max/mean rel    = "
            f"{rank0['logprob_rel']:.3e} / {rank0['logprob_mean_rel']:.3e}\n"
            f"  eager |logits| max      = {rank0['eager_abs_max']:.4f}\n"
            f"  |deepep-eager| max      = {rank0['diff_abs_max']:.4f}"
        )
        header = f"  {'parameter':<75s} {'rel_diff':>10s}"
        print("\n--- per-param grad norm rel_diff (rank 0) ---")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name in sorted(rank0["per_param_rels"]):
            print(f"  {name:<75s} {rank0['per_param_rels'][name]:10.4e}")

        for result in results:
            self.assertTrue(result["logits_finite"], f"rank {result['rank']}: NaN/Inf logits")
            self.assertLess(
                result["logprob_mean_rel"],
                RTOL,
                f"rank {result['rank']}: logprob mean rel_diff={result['logprob_mean_rel']:.2e}",
            )
            self.assertLess(
                result["logprob_rel"],
                LOGPROB_MAX_RTOL,
                f"rank {result['rank']}: logprob max rel_diff={result['logprob_rel']:.2e}",
            )
            self.assertLess(
                result["logits_p99_rel"],
                RTOL,
                f"rank {result['rank']}: logits p99 rel_diff={result['logits_p99_rel']:.2e}",
            )
            self.assertLess(
                result["loss_rel"],
                RTOL,
                f"rank {result['rank']}: loss rel_diff={result['loss_rel']:.2e}",
            )
            self.assertLess(
                result["total_grad_rel"],
                RTOL,
                f"rank {result['rank']}: total grad rel_diff={result['total_grad_rel']:.2e}",
            )
            self.assertLess(
                result["worst_grad_rel"],
                GRAD_NORM_RTOL,
                f"rank {result['rank']}: {result['worst_grad_name']} "
                f"grad rel_diff={result['worst_grad_rel']:.2e}",
            )
