#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Per-kernel accuracy and performance test suite for the 9 Triton kernels.

Architecture:
  ┌─────────────────────────────────────────────────────────────────────┐
  │                        CUDA (NVIDIA GPU)                            │
  │  Each of the 9 kernels is called directly with isolated inputs.    │
  │  Accuracy: compare against FP32 PyTorch reference.                 │
  │  Performance: median latency and throughput per kernel.            │
  │  → Use this to identify which kernel to optimize.                  │
  ├─────────────────────────────────────────────────────────────────────┤
  │                        MLU / NPU (Cambricon / Ascend)               │
  │  Triton kernels are NOT supported on these backends.               │
  │  End-to-end API (linear_cross_entropy) is tested against the       │
  │  pure-PyTorch reference implementation.                            │
  │  → Use this to verify correctness on non-CUDA accelerators.        │
  └─────────────────────────────────────────────────────────────────────┘

Usage::

    # CUDA: per-kernel precision + perf
    python test_kernels.py --mode cuda

    # MLU/NPU: end-to-end precision + perf
    python test_kernels.py --mode mlu

    # All modes (auto-detect)
    python test_kernels.py
"""

import argparse
from contextlib import nullcontext
import sys

import torch
import torch.nn.functional as F

try:
    import triton
except ImportError:
    triton = None

from gpatch_v4.core.device import get_device_backend_name, get_device_module, is_cuda
from gpatch_v4.kernel import linear_cross_entropy

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE_NAME = get_device_backend_name()
DEV = get_device_module()
IS_CUDA = is_cuda()

# MLU 用固定 core 数 launch；CUDA 走下面的 token/GEMM tiling grid
if IS_CUDA:
    TOTAL_CORE_NUM = None
    from gpatch_v4.kernel.triton import linear_ce_kernels as _kernels
    from gpatch_v4.kernel.triton.linear_ce_kernels import (
        BackwardEnum,
        EntropyReductionEnum,
        get_entropy_reduction_enum_number,
        efficient_entropy_forward,
        efficient_entropy_backward,
        set_backward_method,
    )

else:
    from triton.backends.mlu import driver
    _devprob = driver.BangUtils().get_device_properties(DEV.current_device())
    TOTAL_CORE_NUM = _devprob.get('cluster_num') * _devprob.get("core_num_per_cluster")
    from gpatch_v4.kernel.mlu_priv import linear_ce_kernels as _kernels
    from gpatch_v4.kernel.mlu_priv.linear_ce_kernels import (
        BackwardEnum,
        EntropyReductionEnum,
        get_entropy_reduction_enum_number,
        efficient_entropy_forward,
        efficient_entropy_backward,
        set_backward_method,
    )


def _token_grid(num_tokens):
    if IS_CUDA:
        def grid(meta):
            return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]), )
        return grid
    return (TOTAL_CORE_NUM, )


def _gemm_grid(num_tokens, n_dim):
    if IS_CUDA:
        def grid(meta):
            return (
                triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]) *
                triton.cdiv(n_dim, meta["BLOCK_SIZE_N"]),
            )
        return grid
    return (TOTAL_CORE_NUM, )


def _d_hidden_grid(num_tokens, hidden_size):
    if IS_CUDA:
        def grid(meta):
            return (
                triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]) *
                triton.cdiv(hidden_size, meta["BLOCK_SIZE_K"]),
            )
        return grid
    return (TOTAL_CORE_NUM, )


def _d_weight_grid(vocab_size, hidden_size):
    if IS_CUDA:
        def grid(meta):
            return (
                triton.cdiv(vocab_size, meta["BLOCK_SIZE_N"]) *
                triton.cdiv(hidden_size, meta["BLOCK_SIZE_K"]),
            )
        return grid
    return (TOTAL_CORE_NUM, )


def _epilogue_launch_kwargs(num_splits, tp=False):
    if IS_CUDA:
        return {}
    block_sizes = (
        _kernels.epilogue_tp_block_sizes
        if tp
        else _kernels.epilogue_block_sizes
    )
    block_size_m, block_size_n = block_sizes(num_splits)
    return {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "num_warps": 1,
        "num_stages": 1 if tp else 2,
    }


def _nvtx_range(msg: str):
    try:
        nvtx = torch.cuda.nvtx
    except (AttributeError, RuntimeError):
        return nullcontext()
    try:
        return nvtx.range(msg)
    except Exception:
        return nullcontext()


HAVE_TRITON = _kernels.HAVE_TRITON
SUPPORT_CUDA_TMA = _kernels.SUPPORT_CUDA_TMA

DTYPE = torch.bfloat16
TEMPERATURE = 1.0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOCAB_PER_SPLIT = 1024  # from kernels.py forward path

# Perf shapes from the latency table: (num_tokens, hidden_size, vocab_size).
# K6/K7: MLU launchers select mouter vs non-mouter from each shape.
PERF_SHAPES = {
    1: [(512, 2048, 128256), (4096, 2048, 128256)],
    2: [(512, 2048, 128256), (16384, 2048, 128256)],
    3: [(512, 2048, 128256), (16384, 2048, 128256)],
    4: [(512, 2048, 128256), (16384, 2048, 128256)],
    5: [(512, 2048, 128256), (4096, 2048, 128256), (16384, 2048, 128256)],
    6: [(64, 2048, 128256), (512, 2048, 128256), (4096, 2048, 128256)],
    7: [(64, 2048, 128256), (512, 2048, 128256), (4096, 2048, 128256)],
    8: [(512, 2048, 128256), (8192, 2048, 128256), (16384, 2048, 128256)],
    9: [(512, 2048, 128256), (16384, 2048, 128256)],
}

# ---------------------------------------------------------------------------
# PyTorch reference implementations (FP32, pure PyTorch)
# ---------------------------------------------------------------------------

def reference_forward(hidden, weight, labels, reduction, temperature):
    """Pure-PyTorch cross-entropy + token entropy forward.

    NOTE on sign convention: the kernels return ``logprobs = log_softmax[label]``
    (i.e. ``logit - logsumexp``, *positive* — the standard PPO ``log_prob``), NOT
    the negated cross-entropy ``-log_softmax``. PPO consumes it as
    ``ratio = exp(log_prob - old_log_prob)`` (see core_algos.py), so the sign must
    stay positive. The reference therefore returns ``gather`` directly instead of
    ``-gather``.
    """
    reduction_str = {0: "none", 1: "sum", 2: "mean"}[reduction]
    logits = F.linear(hidden, weight) / temperature
    vocab_size = logits.shape[1]
    valid_labels = labels.clamp(0, vocab_size - 1)
    lse = torch.logsumexp(logits, dim=-1)
    log_probs = logits - lse.unsqueeze(-1)
    gather = log_probs.gather(dim=-1, index=valid_labels.unsqueeze(-1)).squeeze(-1)

    if reduction_str == "none":
        logprobs = gather
    elif reduction_str == "sum":
        logprobs = gather.sum()
    else:
        logprobs = gather.mean()

    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return logprobs, entropy, lse


def reference_entropy_only(hidden, weight, labels, temperature):
    """Compute only entropy: H = sum(p * log(p)) per token."""
    logits = F.linear(hidden, weight) / temperature
    lse = torch.logsumexp(logits, dim=-1)
    log_probs = logits - lse.unsqueeze(-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy, lse


def reference_backward(dlogprobs, dentropy, hidden, weight, labels,
                       maximum, acc, entropy_b, reduction, temperature):
    """Analytic backward for cross-entropy + entropy loss.

    Matches the kernel sign convention: forward returns ``logprob = +log_softmax``
    (see ``reference_forward``). The backward kernel multiplies the upstream
    ``dlogprobs`` by ``-1`` at entry (kernels.py ``d_logprobs = -1 * d_logprobs``)
    so that the CE branch ``probs - one_hot`` ends up differentiating
    ``+log_softmax``. We mirror that here by negating ``dlogprobs`` before the
    ``probs * dlogprobs`` coupling term, so the reference and kernel agree.
    """
    num_tokens, vocab_size = labels.shape[0], weight.shape[0]
    logits = F.linear(hidden, weight) / temperature
    lse = torch.logsumexp(logits, dim=-1)
    probs = (logits - lse.unsqueeze(-1)).exp()
    valid_labels = labels.clamp(0, vocab_size - 1)

    reduction_str = {0: "none", 1: "sum", 2: "mean"}[reduction]
    one_hot = torch.zeros_like(probs).scatter_(-1, valid_labels.unsqueeze(-1), 1.0)

    # kernel negates the upstream grad at entry; mirror it so the +log_softmax
    # convention is consistent end-to-end.
    dlogprobs = -dlogprobs

    if reduction_str == "none":
        d_ce = probs - one_hot
        d_ce = d_ce + probs * dlogprobs.unsqueeze(-1)
    elif reduction_str == "sum":
        d_ce = probs - one_hot
        d_ce = d_ce + probs * dlogprobs
    else:
        d_ce = probs - one_hot
        d_ce = d_ce + probs * (dlogprobs / num_tokens)

    log_p = logits - lse.unsqueeze(-1)
    d_entropy = dentropy.unsqueeze(-1) * (1 + log_p) * probs
    d_logits = (d_ce + d_entropy) / temperature

    d_weight = torch.matmul(d_logits.t(), hidden) / temperature
    d_hidden = torch.matmul(d_logits, weight) / temperature
    return d_hidden, d_weight


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def to_cpu(t):
    return t.detach().float().cpu()


def mad(a, b):
    return (to_cpu(a) - to_cpu(b)).abs().max().item()


def rel_err(a, b, eps=1e-5):
    diff = (to_cpu(a) - to_cpu(b)).abs()
    denom = to_cpu(b).abs().clamp_min(eps)
    return (diff / denom).max().item()


def make_inputs(num_tokens, hidden_size, vocab_size, seed=0, with_grad_inputs=False):
    torch.manual_seed(seed)
    hidden = torch.randn(num_tokens, hidden_size, dtype=DTYPE, device=DEVICE_NAME)
    weight = torch.randn(vocab_size, hidden_size, dtype=DTYPE, device=DEVICE_NAME)
    labels = torch.randint(0, vocab_size, (num_tokens,), device=DEVICE_NAME)
    if with_grad_inputs:
        # Mark leaves as requiring grad so the autograd graph is built and
        # torch.autograd.grad((ent, lp), (hidden, weight), ...) can run.
        hidden = hidden.detach().requires_grad_(True)
        weight = weight.detach().requires_grad_(True)
        g_entropy = torch.empty((num_tokens,), dtype=DTYPE, device=DEVICE_NAME).uniform_(-0.5, 0.5)
        g_logprobs = torch.empty((num_tokens,), dtype=DTYPE, device=DEVICE_NAME).uniform_(-1, 1)
        return hidden, weight, labels, g_entropy, g_logprobs
    return hidden, weight, labels


# ---------------------------------------------------------------------------
# CUDA tests: per-kernel precision + performance
# ---------------------------------------------------------------------------

def test_kernel_1_general_mainloop(rtol=1e-2, atol=5e-3,
                                   num_tokens=512, hidden_size=2048, vocab_size=128256):
    """efficient_entropy_kernel_general_mainloop — Forward mainloop."""
    print(f"\n  [Kernel 1] efficient_entropy_kernel_general_mainloop  "
          f"shape=({num_tokens},{hidden_size},{vocab_size})")
    hidden, weight, labels = make_inputs(num_tokens, hidden_size, vocab_size)

    reduction = 2
    vocab_per_split = VOCAB_PER_SPLIT
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

    _max   = torch.empty((num_tokens, num_splits), dtype=torch.float32, device=DEVICE_NAME)
    _accu  = torch.empty((num_tokens, num_splits), dtype=torch.float32, device=DEVICE_NAME)
    _ent_b = torch.empty((num_tokens, num_splits), dtype=torch.float32, device=DEVICE_NAME)
    _logp  = torch.empty((num_tokens,), dtype=torch.float32, device=DEVICE_NAME)
    logp_out = torch.empty((num_tokens,), dtype=torch.float32, device=DEVICE_NAME)

    for _ in range(3):  # warmup
        _kernels.efficient_entropy_kernel_general_mainloop[
            (num_tokens // 128 * num_splits,)
        ](0, hidden, weight, labels, num_tokens, hidden_size, vocab_size,
          vocab_per_split, hidden.stride(0), hidden.stride(1),
          weight.stride(0), weight.stride(1), _max, _max.stride(0), _max.stride(1),
          _accu, _accu.stride(0), _accu.stride(1), _ent_b, _ent_b.stride(0), _ent_b.stride(1),
          _logp, _logp.stride(0), logp_out, 1.0 / TEMPERATURE,
          USE_TMA=SUPPORT_CUDA_TMA and hidden.stride(1) == 1 and weight.stride(1) == 1)
    DEV.synchronize()

    # profile
    iters = 100
    start = DEV.Event(enable_timing=True)
    end = DEV.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        _kernels.efficient_entropy_kernel_general_mainloop[
            (num_tokens // 128 * num_splits,)
        ](0, hidden, weight, labels, num_tokens, hidden_size, vocab_size,
          vocab_per_split, hidden.stride(0), hidden.stride(1),
          weight.stride(0), weight.stride(1), _max, _max.stride(0), _max.stride(1),
          _accu, _accu.stride(0), _accu.stride(1), _ent_b, _ent_b.stride(0), _ent_b.stride(1),
          _logp, _logp.stride(0), logp_out, 1.0 / TEMPERATURE,
          USE_TMA=SUPPORT_CUDA_TMA and hidden.stride(1) == 1 and weight.stride(1) == 1)
        end.record()
        DEV.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    median_ms = samples[iters // 2]
    tokens_per_sec = num_tokens / (median_ms * 1e-3)

    # accuracy: merge the per-split (_max, _accu, _ent_b) the way the epilogue
    # does (log-sum-exp across splits) and compare the resulting per-token lse
    # and entropy against the reference. The mainloop alone does NOT produce a
    # final per-token value — each split holds a partial max / sum-exp / weighted
    # entropy, so comparing a single split (or the flattened per-split tensor)
    # against ref_lse / ref_ent is meaningless and, for _ent_b, a size mismatch.
    ref_ent, ref_lse = reference_entropy_only(hidden.float(), weight.float(), labels, TEMPERATURE)

    _max_cpu = to_cpu(_max[:num_tokens, :num_splits])
    _accu_cpu = to_cpu(_accu[:num_tokens, :num_splits])
    _ent_b_cpu = to_cpu(_ent_b[:num_tokens, :num_splits])

    global_max = _max_cpu.max(dim=-1).values          # (num_tokens,)
    # rescale each split's partial sum-exp / weighted-entropy to the global max
    scale = (_max_cpu - global_max.unsqueeze(-1)).exp()
    accu = (scale * _accu_cpu).sum(dim=-1)            # Z = sum_v exp(logit - max)
    entropy_b = (scale * _ent_b_cpu).sum(dim=-1) / accu  # E[logit] under softmax
    merged_lse = global_max + accu.log()
    merged_ent = merged_lse - entropy_b               # H = lse - E[logit]

    m_max = mad(merged_lse, ref_lse)
    m_ent_b = mad(merged_ent, ref_ent)

    return {
        "kernel": "efficient_entropy_kernel_general_mainloop",
        "shape": (num_tokens, hidden_size, vocab_size),
        "latency_ms": round(median_ms, 3),
        "tokens_per_sec": round(tokens_per_sec, 0),
        "max_mad": m_max,
        "entropy_b_mad": m_ent_b,
        "max_pass": m_max < atol or m_max / ref_lse.abs().max().item() < rtol,
        "entropy_b_pass": m_ent_b < atol or m_ent_b / ref_ent.abs().max().item() < rtol,
    }


def test_kernel_2_epilogue(rtol=1e-2, atol=5e-3,
                           num_tokens=16384, hidden_size=2048, vocab_size=128256):
    """efficient_entropy_triton_kernel_epilogue — Forward epilogue."""
    print(f"\n  [Kernel 2] efficient_entropy_triton_kernel_epilogue  "
          f"shape=({num_tokens},{hidden_size},{vocab_size})")
    vocab_per_split = VOCAB_PER_SPLIT
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

    hidden, weight, labels = make_inputs(num_tokens, hidden_size, vocab_size)

    _max   = torch.empty((num_tokens, num_splits), dtype=torch.float32, device=DEVICE_NAME).abs()
    _accu  = torch.empty((num_tokens, num_splits), dtype=torch.float32, device=DEVICE_NAME)
    _ent_b = torch.empty((num_tokens, num_splits), dtype=torch.float32, device=DEVICE_NAME)
    _logp  = torch.empty((num_tokens,), dtype=torch.float32, device=DEVICE_NAME)
    logp_out = torch.zeros((num_tokens,), dtype=torch.float32, device=DEVICE_NAME)

    # Run mainloop first
    _kernels.efficient_entropy_kernel_general_mainloop[
        (num_tokens // 128 * num_splits,)
    ](0, hidden, weight, labels, num_tokens, hidden_size, vocab_size,
      vocab_per_split, hidden.stride(0), hidden.stride(1),
      weight.stride(0), weight.stride(1), _max, _max.stride(0), _max.stride(1),
      _accu, _accu.stride(0), _accu.stride(1), _ent_b, _ent_b.stride(0), _ent_b.stride(1),
      _logp, _logp.stride(0), logp_out, 1.0 / TEMPERATURE,
      USE_TMA=False)
    DEV.synchronize()

    maximum_out = torch.empty_like(_logp)
    accumulate_out = torch.empty_like(_logp)
    entropy_out = torch.empty_like(_logp)
    entropy_b_out = torch.empty_like(_logp)

    # In none-mode the epilogue finalizes logprobs IN PLACE into ``_logp``
    # (it does ``_logp = lse - _logp``). So running it twice ping-pongs ``_logp``
    # between raw-logit and finalized-logprob. Snapshot the raw state and reset
    # it every iteration so do_bench always measures the same work and the final
    # state is deterministically finalized.
    _logp_raw = _logp.clone()

    def run_epilogue():
        _kernels.efficient_entropy_triton_kernel_epilogue[
            _token_grid(num_tokens)
        ](_max, _max.stride(0), _max.stride(1), num_tokens, num_splits,
          maximum_out, maximum_out.stride(0),
          _accu, _accu.stride(0), _accu.stride(1),
          accumulate_out, accumulate_out.stride(0),
          _ent_b, _ent_b.stride(0), _ent_b.stride(1),
          entropy_b_out, entropy_b_out.stride(0),
          entropy_out, entropy_out.stride(0),
          _logp, _logp.stride(0), logp_out,
          EntropyReductionEnum._None,
          **_epilogue_launch_kwargs(num_splits))

    # warmup
    for _ in range(3):
        _logp.copy_(_logp_raw)
        run_epilogue()
    DEV.synchronize()

    def run_once():
        _logp.copy_(_logp_raw)
        run_epilogue()

    quantiles = [0.5, 0.2, 0.8]
    median_ms, p20_o, p80_o = triton.testing.do_bench(run_once, quantiles=quantiles)
    tokens_per_sec = num_tokens / (median_ms * 1e-3)

    # accuracy check (reduction=none → per-token logprobs)
    ref_lp, ref_ent, ref_max = reference_forward(
        hidden.float(), weight.float(), labels, 0, TEMPERATURE
    )

    # FP32 references for the three reduced buffers the epilogue produces.
    #   maximum     = max_v logit_v
    #   accumulate  = sum_v exp(logit_v - maximum) = exp(lse - maximum)
    #   entropy_b   = sum_v logit_v * softmax_v   (already normalized by the epilogue)
    logits_ref = F.linear(hidden.float(), weight.float()) / TEMPERATURE
    ref_maximum = logits_ref.max(dim=-1).values
    ref_lse = torch.logsumexp(logits_ref, dim=-1)
    ref_accumulate = (ref_lse - ref_maximum).exp()
    softmax_ref = (logits_ref - ref_lse.unsqueeze(-1)).exp()
    ref_entropy_b = (logits_ref * softmax_ref).sum(dim=-1)

    m_ent = mad(entropy_out, ref_ent)
    m_max = mad(maximum_out, ref_maximum)
    m_acc = mad(accumulate_out, ref_accumulate)
    m_eb  = mad(entropy_b_out, ref_entropy_b)

    # ``_logp`` now holds the finalized per-token logprob (= +log_softmax);
    # ``logp_out`` is the scalar buffer only touched by sum/mean (untouched here).
    _logp.copy_(_logp_raw)
    run_epilogue()
    DEV.synchronize()
    # DEBUG: inspect what _logp actually holds at compare time.
    _logp_cpu = _logp.detach().float().cpu()
    ref_lp_cpu = ref_lp.detach().float().cpu()
    print(f"    [debug] _logp raw  min/max/absmax = {_logp_raw.min().item():.3f}/{_logp_raw.max().item():.3f}/{_logp_raw.abs().max().item():.3f}")
    print(f"    [debug] _logp final min/max/absmax = {_logp_cpu.min().item():.3f}/{_logp_cpu.max().item():.3f}/{_logp_cpu.abs().max().item():.3f}")
    print(f"    [debug] ref_lp      min/max/absmax = {ref_lp_cpu.min().item():.3f}/{ref_lp_cpu.max().item():.3f}/{ref_lp_cpu.abs().max().item():.3f}")
    print(f"    [debug] first 5 _logp = {[round(x,3) for x in _logp_cpu[:5].tolist()]}")
    print(f"    [debug] first 5 ref_lp = {[round(x,3) for x in ref_lp_cpu[:5].tolist()]}")
    m_lp = mad(_logp, ref_lp)

    print(f"    [chk] maximum    mad={m_max:.4e},  pass={m_max < atol or m_max / ref_maximum.abs().max().item() < rtol}")
    print(f"    [chk] accumulate mad={m_acc:.4e}  pass={m_acc < atol or m_acc / ref_accumulate.abs().max().item() < rtol}")
    print(f"    [chk] entropy_b  mad={m_eb:.4e}  pass={m_eb < atol or m_eb / ref_entropy_b.abs().max().item() < rtol}")

    max_pass = m_max < atol or m_max / ref_maximum.abs().max().item() < rtol
    acc_pass = m_acc < atol or m_acc / ref_accumulate.abs().max().item() < rtol
    eb_pass = m_eb < atol or m_eb / ref_entropy_b.abs().max().item() < rtol

    return {
        "kernel": "efficient_entropy_triton_kernel_epilogue",
        "shape": (num_tokens, hidden_size, vocab_size),
        "latency_ms": round(median_ms, 3),
        "tokens_per_sec": round(tokens_per_sec, 0),
        "entropy_mad": m_ent,
        "logprobs_mad": m_lp,
        "maximum_mad": m_max,
        "accumulate_mad": m_acc,
        "entropy_b_mad": m_eb,
        "entropy_pass": m_ent < atol or m_ent / ref_ent.abs().max().item() < rtol,
        "logprobs_pass": m_lp < atol or m_lp / ref_lp.abs().max().item() < rtol,
        "maximum_pass": max_pass,
        "accumulate_pass": acc_pass,
        "entropy_b_pass": eb_pass,
    }

def test_kernel_3_epilogue_tp(rtol=1e-2, atol=5e-3,
                              num_tokens=512, hidden_size=2048, vocab_size=128256):
    """efficient_entropy_triton_kernel_epilogue_tp — Forward TP epilogue (standalone).

    This kernel does NOT need a real tensor-parallel process group: it is a pure
    per-token online-softmax *combine* that merges the per-split buffers of ONE
    rank, given the cross-rank-reduced global per-split max (``reduced_max``) and
    that rank's own unreduced per-split max (``original_max``). The full TP path
    wraps it with ``all_reduce(MAX)`` on ``_max`` (→ reduced_max, with
    ``_max_backup`` kept as original_max) before, and ``all_reduce(SUM)`` on the
    accumulate/entropy_b outputs after. We simulate R ranks here by generating R
    independent per-split buffer sets, taking the elementwise max across ranks to
    stand in for ``all_reduce(MAX)``, running the kernel once per rank (each with
    its own original_max), and summing the R outputs to stand in for
    ``all_reduce(SUM)``. The summed result must equal the single-rank (non-TP)
    epilogue's global_max / global_accu / global_entropy_b — i.e. the same values
    you'd get from reducing the full unsplit vocabulary in one shot.

    Equivalence (why summing R per-rank combines == the global reduction):
      accu_s^r = Σ_v exp(x^r_v - max_s^r), so
      exp(max_s^r - gmax) * accu_s^r = Σ_v exp(x^r_v - gmax).
      Summing over ranks r and splits s gives Σ_v exp(x_v - gmax) = global_accu.
      Same argument holds for entropy_b. So the per-rank combine + cross-rank SUM
      reconstructs the exact global online-softmax reduction.
    """
    print(f"\n  [Kernel 3] efficient_entropy_triton_kernel_epilogue_tp  "
          f"shape=({num_tokens},{hidden_size},{vocab_size})")
    vocab_per_split = VOCAB_PER_SPLIT
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split
    world_size = 4  # simulated TP ranks

    # Build R independent mainloop-style per-split buffers. Each "rank" owns a
    # disjoint slice of a full-vocab logit tensor so that the cross-rank sum
    # reconstructs the whole-vocab softmax. We partition vocab by rank (not by
    # split) so each rank produces `num_splits` genuine per-split (max,accu,ent_b).
    torch.manual_seed(0)
    full_logits = torch.randn(num_tokens, vocab_size, device=DEVICE_NAME, dtype=torch.float32)
    # disjoint vocab partition across ranks (remainder goes to the last rank)
    per_rank_vocab = [vocab_size // world_size + (r < vocab_size % world_size) for r in range(world_size)]

    rank_max = []    # rank_max[r]: [num_tokens, num_splits] local per-split max
    rank_accu = []   # rank_accu[r]: same
    rank_entb = []   # rank_entb[r]: same
    vocab_off = 0
    for r in range(world_size):
        vr = per_rank_vocab[r]
        rmax = torch.empty(num_tokens, num_splits, dtype=torch.float32, device=DEVICE_NAME)
        racc = torch.empty(num_tokens, num_splits, dtype=torch.float32, device=DEVICE_NAME)
        rent = torch.empty(num_tokens, num_splits, dtype=torch.float32, device=DEVICE_NAME)
        for s in range(num_splits):
            # Intersect the global split band [g_lo, g_hi) with this rank's
            # vocab range [vocab_off, vocab_off+vr). Mirror the mainloop: max
            # over the slice, accu/ent_b shifted by that max. A rank with no
            # vocab in a split yields a degenerate split (max=0, accu=0).
            g_lo = s * vocab_per_split
            g_hi = min((s + 1) * vocab_per_split, vocab_size)
            lo = max(g_lo, vocab_off)
            hi = min(g_hi, vocab_off + vr)
            if hi <= lo:
                # rank has no vocab in this split -> degenerate split (max=0, accu=0)
                rmax[:, s] = 0.0
                racc[:, s] = 0.0
                rent[:, s] = 0.0
                continue
            chunk = full_logits[:, lo:hi]
            m = chunk.max(dim=-1).values
            e = (chunk - m.unsqueeze(-1)).exp()
            rmax[:, s] = m
            racc[:, s] = e.sum(dim=-1)
            rent[:, s] = (chunk * e).sum(dim=-1)
        rank_max.append(rmax)
        rank_accu.append(racc)
        rank_entb.append(rent)
        vocab_off += vr

    # Simulate all_reduce(MAX) of _max across ranks -> reduced_max.
    # (original_max for rank r is its own rank_max[r].)
    reduced_max = torch.stack(rank_max, dim=0).max(dim=0).values  # [num_tokens, num_splits]

    # Per-rank kernel output, then cross-rank combine. Only `accumulate` and
    # `entropy_b` get all_reduce(SUM) in the real TP path — `maximum` is NOT
    # summed because every rank computes the identical global_max (they all
    # share the same reduced_max, and the kernel's global_max = max(reduced_max)
    # is deterministic). This matches kernels.py, which all_reduce(SUM)s only
    # accumulate_and_entropy_b (line ~811), not maximum. So take g_max from any
    # single rank and SUM only accu/entropy_b.
    g_max = torch.empty(num_tokens, dtype=torch.float32, device=DEVICE_NAME)
    g_accu = torch.zeros(num_tokens, dtype=torch.float32, device=DEVICE_NAME)
    g_entb = torch.zeros(num_tokens, dtype=torch.float32, device=DEVICE_NAME)

    grid = _token_grid(num_tokens)

    for r in range(world_size):
        out_max = torch.empty(num_tokens, dtype=torch.float32, device=DEVICE_NAME)
        out_accu = torch.empty(num_tokens, dtype=torch.float32, device=DEVICE_NAME)
        out_entb = torch.empty(num_tokens, dtype=torch.float32, device=DEVICE_NAME)
        _kernels.efficient_entropy_triton_kernel_epilogue_tp[grid](
            num_tokens, num_splits,
            reduced_max, reduced_max.stride(0), reduced_max.stride(1),
            rank_max[r], rank_max[r].stride(0), rank_max[r].stride(1),
            rank_accu[r], rank_accu[r].stride(0), rank_accu[r].stride(1),
            rank_entb[r], rank_entb[r].stride(0), rank_entb[r].stride(1),
            out_max, out_max.stride(0),
            out_accu, out_accu.stride(0),
            out_entb, out_entb.stride(0),
            **_epilogue_launch_kwargs(num_splits, tp=True),
        )
        g_max = out_max  # identical across ranks; keep one copy
        g_accu += out_accu
        g_entb += out_entb
    DEV.synchronize()

    # Reference 1: pure-PyTorch whole-vocab reduction (the ground truth the TP
    # path must reconstruct). global_max = max over ALL vocab; global_accu =
    # Σ exp(x - global_max); global_entb = Σ x·exp(x - global_max).
    ref_gmax = full_logits.max(dim=-1).values
    ref_gaccu = (full_logits - ref_gmax.unsqueeze(-1)).exp().sum(dim=-1)
    ref_gentb = (full_logits * (full_logits - ref_gmax.unsqueeze(-1)).exp()).sum(dim=-1)

    m_max = mad(g_max, ref_gmax)
    m_acc = mad(g_accu, ref_gaccu)
    m_eb = mad(g_entb, ref_gentb)

    print(f"    [chk] vs PyTorch whole-vocab: max_mad={m_max:.4e} acc_mad={m_acc:.4e} eb_mad={m_eb:.4e}")

    max_pass = m_max < atol or m_max / ref_gmax.abs().max().item() < rtol
    acc_pass = m_acc < atol or m_acc / ref_gaccu.abs().max().item() < rtol
    eb_pass = m_eb < atol or m_eb / ref_gentb.abs().max().item() < rtol

    # latency: a SINGLE kernel launch (one rank). This is the per-kernel cost,
    # matching how the other kernel tests are timed. The full TP path's
    # wall-clock (R ranks running in parallel + two all-reduces) is a
    # multi-card property that a single-device test cannot measure, so we do
    # not pretend to time it here.
    out_max = torch.empty(num_tokens, dtype=torch.float32, device=DEVICE_NAME)
    out_accu = torch.empty(num_tokens, dtype=torch.float32, device=DEVICE_NAME)
    out_entb = torch.empty(num_tokens, dtype=torch.float32, device=DEVICE_NAME)

    def run_once():
        _kernels.efficient_entropy_triton_kernel_epilogue_tp[grid](
            num_tokens, num_splits,
            reduced_max, reduced_max.stride(0), reduced_max.stride(1),
            rank_max[0], rank_max[0].stride(0), rank_max[0].stride(1),
            rank_accu[0], rank_accu[0].stride(0), rank_accu[0].stride(1),
            rank_entb[0], rank_entb[0].stride(0), rank_entb[0].stride(1),
            out_max, out_max.stride(0),
            out_accu, out_accu.stride(0),
            out_entb, out_entb.stride(0),
            **_epilogue_launch_kwargs(num_splits, tp=True),
        )

    quantiles = [0.5, 0.2, 0.8]
    for _ in range(3):
        run_once()
    DEV.synchronize()
    median_ms, p20_o, p80_o = triton.testing.do_bench(run_once, quantiles=quantiles)
    tokens_per_sec = num_tokens / (median_ms * 1e-3)

    return {
        "kernel": "efficient_entropy_triton_kernel_epilogue_tp",
        "shape": (num_tokens, hidden_size, vocab_size),
        "latency_ms": round(median_ms, 3),
        "tokens_per_sec": round(tokens_per_sec, 0),
        "maximum_mad": m_max,
        "accumulate_mad": m_acc,
        "entropy_b_mad": m_eb,
        "maximum_pass": max_pass,
        "accumulate_pass": acc_pass,
        "entropy_b_pass": eb_pass,
    }


def test_kernel_4_epilogue_tp_update(rtol=1e-2, atol=5e-3,
                                     num_tokens=512, hidden_size=2048, vocab_size=128256):
    """efficient_entropy_triton_epilogue_tp_update — Forward TP update (standalone).

    This kernel does NOT actually require a tensor-parallel process group: it is a
    pure per-token epilogue that normalizes ``entropy_b`` and finalizes ``entropy``
    and ``logprobs`` from the already-reduced ``maximum`` / ``accumulate`` /
    ``entropy_b`` / raw ``logprobs`` buffers. In the TP forward it merely runs
    *after* the all-reduces, so we can benchmark it standalone by producing the
    same inputs with the non-TP epilogue and timing only this kernel.
    """
    print(f"\n  [Kernel 4] efficient_entropy_triton_epilogue_tp_update  "
          f"shape=({num_tokens},{hidden_size},{vocab_size})")
    hidden, weight, labels = make_inputs(num_tokens, hidden_size, vocab_size)

    vocab_per_split = VOCAB_PER_SPLIT
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

    _max   = torch.empty((num_tokens, num_splits), dtype=torch.float32, device=DEVICE_NAME)
    _accu  = torch.empty((num_tokens, num_splits), dtype=torch.float32, device=DEVICE_NAME)
    _ent_b = torch.empty((num_tokens, num_splits), dtype=torch.float32, device=DEVICE_NAME)
    _logp  = torch.empty((num_tokens,), dtype=torch.float32, device=DEVICE_NAME)
    logp_out = torch.empty((num_tokens,), dtype=torch.float32, device=DEVICE_NAME)

    # 1) Run the mainloop to populate per-split buffers + raw logprobs.
    _kernels.efficient_entropy_kernel_general_mainloop[
        (num_tokens // 128 * num_splits,)
    ](0, hidden, weight, labels, num_tokens, hidden_size, vocab_size,
      vocab_per_split, hidden.stride(0), hidden.stride(1),
      weight.stride(0), weight.stride(1), _max, _max.stride(0), _max.stride(1),
      _accu, _accu.stride(0), _accu.stride(1), _ent_b, _ent_b.stride(0), _ent_b.stride(1),
      _logp, _logp.stride(0), logp_out, 1.0 / TEMPERATURE,
      USE_TMA=False)

    # 2) Reduce the per-split buffers down to per-token maximum / accumulate /
    #    entropy_b using the non-TP epilogue. Its outputs are exactly the inputs
    #    that tp_update consumes in the TP path.
    maximum_out = torch.empty_like(_logp)
    accumulate_out = torch.empty_like(_logp)
    entropy_out = torch.empty_like(_logp)
    entropy_b_out = torch.empty_like(_logp)

    # The non-TP epilogue finalizes ``_logp`` *in place*, but tp_update expects
    # the RAW per-token logprob (in the TP path epilogue_tp never touches
    # ``_logprobs``). Snapshot the raw values before the epilogue clobbers them.
    _logp_raw = _logp.clone()

    _kernels.efficient_entropy_triton_kernel_epilogue[
        (num_tokens // 16,)
    ](_max, _max.stride(0), _max.stride(1), num_tokens, num_splits,
      maximum_out, maximum_out.stride(0),
      _accu, _accu.stride(0), _accu.stride(1),
      accumulate_out, accumulate_out.stride(0),
      _ent_b, _ent_b.stride(0), _ent_b.stride(1),
      entropy_b_out, entropy_b_out.stride(0),
      entropy_out, entropy_out.stride(0),
      _logp, _logp.stride(0), logp_out,
      EntropyReductionEnum._Mean,
      **_epilogue_launch_kwargs(num_splits))
    DEV.synchronize()

    # The non-TP epilogue already normalizes ``entropy_b = entropy_b / accumulate``
    # (kernels.py epilogue line ``tl.fdiv(global_entropy_b, global_accu)``). But
    # ``tp_update`` expects the *un-normalized* (accumulated) form, because in the
    # real TP path ``epilogue_tp`` stores entropy_b BEFORE the division and
    # ``tp_update`` performs it (``entropy_b = entropy_b / accumulate``). Feeding
    # the already-divided value would divide twice and blow up entropy. Undo the
    # epilogue's division so tp_update sees exactly what the TP path feeds it.
    entropy_b_unnorm = entropy_b_out * accumulate_out

    # 3) Benchmark ONLY the tp_update kernel. Give it fresh scratch outputs so
    #    repeated iterations measure the real work, not idempotent re-stores.
    entropy_tp = torch.empty_like(entropy_out)
    logp_tp = torch.empty((), dtype=torch.float32, device=DEVICE_NAME)

    # tp_update writes the *normalized* entropy_b back to its input buffer in
    # place (kernels.py ``tl.store(entropy_b_ptr, entropy_b/accumulate)``). So
    # feeding the same buffer across iterations would corrupt it (divide-by-
    # accumulate applied N times). Use a scratch input we re-fill from the
    # pristine un-normalized values every iteration so the kernel always sees
    # the true TP-path input.
    entropy_b_in = torch.empty_like(entropy_b_unnorm)

    grid = _token_grid(num_tokens)
    # def grid(meta):
    #     return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]),)
    def run_once():
        logp_tp.zero_()
        entropy_b_in.copy_(entropy_b_unnorm)
        _kernels.efficient_entropy_triton_epilogue_tp_update[grid](
            num_tokens,
            _logp_raw, _logp_raw.stride(0),
            maximum_out, maximum_out.stride(0),
            accumulate_out, accumulate_out.stride(0),
            entropy_b_in, entropy_b_in.stride(0),
            entropy_tp, entropy_tp.stride(0),
            logp_tp,
            2,  # reduction = mean
        )
        # DEV.synchronize()
    for _ in range(3):  # warmup
        run_once()
    DEV.synchronize()

    # iters = 100
    # start = DEV.Event(enable_timing=True)
    # end = DEV.Event(enable_timing=True)
    # samples = []
    # for _ in range(iters):
    #     start.record()
    #     run_once()
    #     end.record()
    #     DEV.synchronize()
    #     samples.append(start.elapsed_time(end))
    # samples.sort()
    # median_ms = samples[iters // 2]
    # tokens_per_sec = num_tokens / (median_ms * 1e-3)
    quantiles = [0.5, 0.2, 0.8]
    median_ms, p20_o, p80_o = triton.testing.do_bench(run_once, quantiles=quantiles)
    tokens_per_sec = num_tokens / (median_ms * 1e-3)
    # accuracy: compare the tp_update outputs against the FP32 reference.
    ref_lp, ref_ent, _ = reference_forward(
        hidden.float(), weight.float(), labels, 2, TEMPERATURE
    )
    m_ent = mad(entropy_tp, ref_ent)
    # logp_tp is a scalar (reduction=mean); compare against the reference mean.
    m_lp_scalar = abs(float(logp_tp.cpu()) - float(ref_lp.cpu()))
    # print("core_num:======================================", float(logp_tp.cpu()) )
    return {
        "kernel": "efficient_entropy_triton_epilogue_tp_update",
        "shape": (num_tokens, hidden_size, vocab_size),
        "latency_ms": round(median_ms, 3),
        "tokens_per_sec": round(tokens_per_sec, 0),
        "entropy_mad": m_ent,
        "logprobs_mad": m_lp_scalar,
        "entropy_pass": m_ent < atol or m_ent / ref_ent.abs().max().item() < rtol,
        "logprobs_pass": m_lp_scalar < atol or m_lp_scalar / ref_lp.abs().max().item() < rtol,
    }


def test_kernel_5_backward_mainloop_mn(rtol=1e-2, atol=1e-2,
                                       num_tokens=512, hidden_size=2048, vocab_size=128256):
    """efficient_entropy_backward_kernel_general_mainloop_MN — Backward fused MN."""
    print(f"\n  [Kernel 5] efficient_entropy_backward_kernel_general_mainloop_MN  "
          f"shape=({num_tokens},{hidden_size},{vocab_size})")
    hidden, weight, labels = make_inputs(num_tokens, hidden_size, vocab_size)

    set_backward_method(BackwardEnum._Total_Fuse_MN)

    # Forward to get intermediates
    tri_out = efficient_entropy_forward(hidden, weight, labels, reduction=2, temperature=TEMPERATURE)
    logp_out, entropy_out, maximum_out, acc_out, ent_b_out = tri_out

    dlogprobs = torch.ones_like(logp_out)
    dentropy   = torch.ones_like(entropy_out)

    d_hidden = torch.zeros_like(hidden)
    d_weight = torch.zeros_like(weight)

    for _ in range(3):
        dh, dw = efficient_entropy_backward(
            dlogprobs, dentropy, hidden, weight, labels,
            maximum_out, acc_out, ent_b_out, reduction=2, temperature=TEMPERATURE,
        )
    DEV.synchronize()

    iters = 100
    start = DEV.Event(enable_timing=True)
    end = DEV.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        dh, dw = efficient_entropy_backward(
            dlogprobs, dentropy, hidden, weight, labels,
            maximum_out, acc_out, ent_b_out, reduction=2, temperature=TEMPERATURE,
        )
        end.record()
        DEV.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    median_ms = samples[iters // 2]

    dh_ref, dw_ref = reference_backward(
        dlogprobs.float(), dentropy.float(), hidden.float(), weight.float(), labels,
        maximum_out.float(), acc_out.float(), ent_b_out.float(), 2, TEMPERATURE,
    )
    m_dh = mad(dh_ref, dh)
    m_dw = mad(dw_ref, dw)

    set_backward_method(BackwardEnum._Split_Dlogits_N)
    return {
        "kernel": "efficient_entropy_backward_kernel_general_mainloop_MN",
        "shape": (num_tokens, hidden_size, vocab_size),
        "latency_ms": round(median_ms, 3),
        "tokens_per_sec": round(num_tokens / (median_ms * 1e-3), 0),
        "d_hidden_mad": m_dh,
        "d_weight_mad": m_dw,
        "d_hidden_pass": m_dh < atol or m_dh / dh_ref.abs().max().item() < rtol,
        "d_weight_pass": m_dw < atol or m_dw / dw_ref.abs().max().item() < rtol,
    }


def _d_hidden_variant(num_tokens, hidden_size):
    if IS_CUDA:
        return "cuda"
    use_mouter = getattr(_kernels, "_d_hidden_use_mouter", None)
    if use_mouter is None:
        return "unknown"
    return "mouter" if use_mouter(num_tokens, hidden_size) else "non-mouter"


def _d_weight_variant(num_tokens, hidden_size):
    if IS_CUDA:
        return "cuda"
    use_mouter = getattr(_kernels, "_d_weight_use_mouter", None)
    if use_mouter is None:
        return "unknown"
    return "mouter" if use_mouter(num_tokens, hidden_size) else "non-mouter"


def test_kernel_6_backward_d_hidden(rtol=1e-2, atol=1e-2,
                                    num_tokens=512, hidden_size=2048, vocab_size=128256):
    """efficient_entropy_backward_kernel_d_hidden — Backward d_hidden only."""
    variant = _d_hidden_variant(num_tokens, hidden_size)
    print(f"\n  [Kernel 6] efficient_entropy_backward_kernel_d_hidden  "
          f"shape=({num_tokens},{hidden_size},{vocab_size})  variant={variant}")
    hidden, weight, labels = make_inputs(num_tokens, hidden_size, vocab_size)

    set_backward_method(BackwardEnum._Total_Fuse_MN)
    tri_out = efficient_entropy_forward(hidden, weight, labels, reduction=2, temperature=TEMPERATURE)
    logp_out, entropy_out, maximum_out, acc_out, ent_b_out = tri_out

    dlogprobs = torch.ones_like(logp_out)
    dentropy   = torch.ones_like(entropy_out)

    d_hidden = torch.zeros_like(hidden)
    d_hidden_grid = _d_hidden_grid(num_tokens, hidden_size)
    launch_d_hidden = (
        None if IS_CUDA else
        getattr(_kernels, "launch_efficient_entropy_backward_kernel_d_hidden", None)
    )

    def run_d_hidden():
        if launch_d_hidden is not None:
            # MLU host launcher picks mouter vs non-mouter from (tokens, hidden).
            launch_d_hidden(
                num_tokens, hidden_size, vocab_size, 0,
                hidden, weight, labels,
                maximum_out, acc_out, dentropy, dlogprobs, 2, ent_b_out,
                d_hidden, 1.0 / TEMPERATURE,
            )
            return
        # CUDA linear_ce_kernels 是 @triton.jit，必须 [grid] 启动
        _kernels.efficient_entropy_backward_kernel_d_hidden[d_hidden_grid](
            num_tokens, hidden_size, vocab_size, 0,
            hidden, hidden.stride(0), hidden.stride(1),
            weight, weight.stride(0), weight.stride(1),
            labels, labels.stride(0),
            maximum_out, maximum_out.stride(0),
            acc_out, acc_out.stride(0),
            dentropy, dentropy.stride(0),
            dlogprobs, dlogprobs.stride(0) if dlogprobs.dim() > 0 else 0,
            2, ent_b_out, ent_b_out.stride(0),
            d_hidden, d_hidden.stride(0), d_hidden.stride(1),
            1.0 / TEMPERATURE,
        )

    for _ in range(3):
        run_d_hidden()
    DEV.synchronize()

    iters = 100
    start = DEV.Event(enable_timing=True)
    end = DEV.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        run_d_hidden()
        end.record()
        DEV.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    median_ms = samples[iters // 2]

    dh_ref, _ = reference_backward(
        dlogprobs.float(), dentropy.float(), hidden.float(), weight.float(), labels,
        maximum_out.float(), acc_out.float(), ent_b_out.float(), 2, TEMPERATURE,
    )
    m_dh = mad(dh_ref, d_hidden)

    set_backward_method(BackwardEnum._Split_Dlogits_N)
    return {
        "kernel": "efficient_entropy_backward_kernel_d_hidden",
        "shape": (num_tokens, hidden_size, vocab_size),
        "variant": variant,
        "latency_ms": round(median_ms, 3),
        "tokens_per_sec": round(num_tokens / (median_ms * 1e-3), 0),
        "d_hidden_mad": m_dh,
        "d_hidden_pass": m_dh < atol or m_dh / dh_ref.abs().max().item() < rtol,
    }


def test_kernel_7_backward_d_weight(rtol=1e-2, atol=1e-2,
                                    num_tokens=512, hidden_size=2048, vocab_size=128256):
    """efficient_entropy_backward_kernel_d_weight — Backward d_weight only."""
    variant = _d_weight_variant(num_tokens, hidden_size)
    print(f"\n  [Kernel 7] efficient_entropy_backward_kernel_d_weight  "
          f"shape=({num_tokens},{hidden_size},{vocab_size})  variant={variant}")
    hidden, weight, labels = make_inputs(num_tokens, hidden_size, vocab_size)

    set_backward_method(BackwardEnum._Total_Fuse_MN)
    tri_out = efficient_entropy_forward(hidden, weight, labels, reduction=2, temperature=TEMPERATURE)
    logp_out, entropy_out, maximum_out, acc_out, ent_b_out = tri_out

    dlogprobs = torch.ones_like(logp_out)
    dentropy   = torch.ones_like(entropy_out)
    d_weight = torch.zeros_like(weight)
    d_weight_grid = _d_weight_grid(vocab_size, hidden_size)
    launch_d_weight = (
        None if IS_CUDA else
        getattr(_kernels, "launch_efficient_entropy_backward_kernel_d_weight", None)
    )

    def run_d_weight():
        if launch_d_weight is not None:
            launch_d_weight(
                num_tokens, hidden_size, vocab_size, 0,
                hidden, weight, labels,
                maximum_out, acc_out, dentropy, dlogprobs, 2, ent_b_out,
                d_weight, 1.0 / TEMPERATURE,
            )
            return
        _kernels.efficient_entropy_backward_kernel_d_weight[d_weight_grid](
            num_tokens, hidden_size, vocab_size, 0,
            hidden, hidden.stride(0), hidden.stride(1),
            weight, weight.stride(0), weight.stride(1),
            labels, labels.stride(0),
            maximum_out, maximum_out.stride(0),
            acc_out, acc_out.stride(0),
            dentropy, dentropy.stride(0),
            dlogprobs, dlogprobs.stride(0) if dlogprobs.dim() > 0 else 0,
            2, ent_b_out, ent_b_out.stride(0),
            d_weight, d_weight.stride(0), d_weight.stride(1),
            1.0 / TEMPERATURE,
        )

    for _ in range(3):
        run_d_weight()
    DEV.synchronize()

    iters = 100
    start = DEV.Event(enable_timing=True)
    end = DEV.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        run_d_weight()
        end.record()
        DEV.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    median_ms = samples[iters // 2]

    _, dw_ref = reference_backward(
        dlogprobs.float(), dentropy.float(), hidden.float(), weight.float(), labels,
        maximum_out.float(), acc_out.float(), ent_b_out.float(), 2, TEMPERATURE,
    )
    m_dw = mad(dw_ref, d_weight)

    set_backward_method(BackwardEnum._Split_Dlogits_N)
    return {
        "kernel": "efficient_entropy_backward_kernel_d_weight",
        "shape": (num_tokens, hidden_size, vocab_size),
        "variant": variant,
        "latency_ms": round(median_ms, 3),
        "tokens_per_sec": round(num_tokens / (median_ms * 1e-3), 0),
        "d_weight_mad": m_dw,
        "d_weight_pass": m_dw < atol or m_dw / dw_ref.abs().max().item() < rtol,
    }


def _autograd_d_logits_reference(hidden, weight, labels, reduction, temperature):
    """Ground-truth backward via torch.autograd.

    Mirrors the kernel's forward convention (see ``reference_forward``):
    ``logprobs = +log_softmax[label]`` and per-token ``entropy =
    -sum(p log p)``, both reduced by ``reduction`` (0=none, 1=sum, 2=mean).
    Returns ``(d_hidden, d_weight)`` for the combined loss
    ``reduced(logprobs) + reduced(entropy)`` with upstream grads of ones.

    The hand-written ``reference_backward`` does NOT match autograd (it gets the
    entropy / reduction scaling wrong), so the d_logits kernel is validated
    against this autograd oracle instead.
    """
    hidden = hidden.detach().float().requires_grad_(True)
    weight = weight.detach().float().requires_grad_(True)
    logits = F.linear(hidden, weight) / temperature
    valid_labels = labels.clamp(0, logits.shape[1] - 1)
    lse = torch.logsumexp(logits, dim=-1)
    log_probs = logits - lse.unsqueeze(-1)
    gather = log_probs.gather(-1, valid_labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)

    if reduction == 0:  # none: per-token loss, upstream grads are ones
        loss = gather + entropy
        grads = (torch.ones_like(gather), torch.ones_like(entropy))
    elif reduction == 1:  # sum
        loss = gather.sum() + entropy.sum()
        grads = None
    else:  # mean
        loss = gather.mean() + entropy.mean()
        grads = None

    d_hidden, d_weight = torch.autograd.grad(loss, (hidden, weight), grad_outputs=grads)
    return d_hidden.detach(), d_weight.detach()


def test_kernel_8_backward_d_logits(rtol=1e-2, atol=1e-2,
                                    num_tokens=16384, hidden_size=2048, vocab_size=128256):
    """efficient_entropy_backward_kernel_general_d_logits — Backward d_logits (full vocab).

    This kernel only produces ``d_logits``; ``d_hidden = d_logits @ weight`` and
    ``d_weight = d_logits.T @ hidden`` are deferred to matmuls (matching the
    ``_Total_Separate`` host path in kernels.py). We therefore validate the
    kernel by comparing those two downstream gradients against the
    ``torch.autograd`` ground truth.

    NOTE: the previous version of this test passed ``num_tokens, vocab_size`` as
    the ``d_logits`` strides in its warmup launch, which wrote out of bounds on
    MLU and triggered a device coredump (CN_INVOKE_ERROR). The strides now come
    from ``d_logits_buf.stride()`` in both the warmup and the timed loop.
    """
    print(f"\n  [Kernel 8] efficient_entropy_backward_kernel_general_d_logits  "
          f"shape=({num_tokens},{hidden_size},{vocab_size})")
    reduction = 2  # mean
    hidden, weight, labels = make_inputs(num_tokens, hidden_size, vocab_size)

    # autograd ground truth (fp32 leaves) for d_hidden / d_weight of the combined
    # mean-reduced logprobs + entropy loss with upstream grads of ones.
    dh_ref, dw_ref = _autograd_d_logits_reference(
        hidden, weight, labels, reduction, TEMPERATURE
    )

    set_backward_method(BackwardEnum._Total_Separate)
    tri_out = efficient_entropy_forward(hidden, weight, labels, reduction=reduction, temperature=TEMPERATURE)
    logp_out, entropy_out, maximum_out, acc_out, ent_b_out = tri_out

    dlogprobs = torch.ones_like(logp_out)
    # The forward kernel returns ``entropy`` per-token even for mean/sum
    # reduction (only ``logprobs`` is reduced to a scalar), so the backward
    # kernel consumes ``dentropy`` per-token WITHOUT any 1/num_tokens scaling
    # (it re-applies that scaling only to the scalar ``dlogprobs``). The autograd
    # oracle uses ``entropy.mean()``, whose per-token upstream grad is 1/N, so we
    # must pass ``dentropy = ones / num_tokens`` to match — passing plain ones
    # inflates the entropy branch by N and blows up d_hidden/d_weight.
    dentropy = torch.ones_like(entropy_out) / num_tokens

    d_logits_buf = torch.empty((num_tokens, vocab_size), dtype=DTYPE, device=DEVICE_NAME)

    d_logits_grid = _gemm_grid(num_tokens, vocab_size)

    def run():
        _kernels.efficient_entropy_backward_kernel_general_d_logits[d_logits_grid](
            num_tokens, hidden_size, vocab_size, 0,
            hidden, hidden.stride(0), hidden.stride(1),
            weight, weight.stride(0), weight.stride(1),
            labels, labels.stride(0),
            maximum_out, maximum_out.stride(0),
            acc_out, acc_out.stride(0),
            dentropy, dentropy.stride(0),
            dlogprobs, 0, reduction,
            ent_b_out, ent_b_out.stride(0),
            d_logits_buf, d_logits_buf.stride(0), d_logits_buf.stride(1),
            1.0 / TEMPERATURE,
            USE_TMA=False,
        )

    for _ in range(3):  # warmup
        run()
    DEV.synchronize()

    # accuracy: d_logits -> d_hidden / d_weight, vs autograd oracle.
    d_hidden = torch.matmul(d_logits_buf.float(), weight.float())
    d_weight = torch.matmul(d_logits_buf.float().t(), hidden.float())
    m_dh = mad(dh_ref, d_hidden)
    m_dw = mad(dw_ref, d_weight)
    dh_pass = m_dh < atol or m_dh / dh_ref.abs().max().item() < rtol
    dw_pass = m_dw < atol or m_dw / dw_ref.abs().max().item() < rtol
    print(f"    d_hidden: mad={m_dh:.4e}  pass={dh_pass}  (ref|max={dh_ref.abs().max().item():.4e})")
    print(f"    d_weight: mad={m_dw:.4e}  pass={dw_pass}  (ref|max={dw_ref.abs().max().item():.4e})")

    # latency: triton.testing.do_bench handles its own warmup + repeated timing
    # (matching how kernels 2/3/4 are benchmarked). It returns the median latency
    # in ms plus the requested quantiles.
    quantiles = [0.5, 0.2, 0.8]
    median_ms, p20_o, p80_o = triton.testing.do_bench(run, quantiles=quantiles)

    set_backward_method(BackwardEnum._Split_Dlogits_N)
    return {
        "kernel": "efficient_entropy_backward_kernel_general_d_logits",
        "shape": (num_tokens, hidden_size, vocab_size),
        "latency_ms": round(median_ms, 3),
        "tokens_per_sec": round(num_tokens / (median_ms * 1e-3), 0),
        "d_hidden_mad": m_dh,
        "d_weight_mad": m_dw,
        "d_hidden_pass": dh_pass,
        "d_weight_pass": dw_pass,
    }


def test_kernel_9_backward_d_logits_split_n(rtol=1e-2, atol=1e-2,
                                            num_tokens=16384, hidden_size=2048, vocab_size=128256):
    """efficient_entropy_backward_kernel_general_d_logits_split_N — Backward d_logits (split vocab)."""
    print(f"\n  [Kernel 9] efficient_entropy_backward_kernel_general_d_logits_split_N  "
          f"shape=({num_tokens},{hidden_size},{vocab_size})")
    hidden, weight, labels = make_inputs(num_tokens, hidden_size, vocab_size)

    vocab_per_split = 9504
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

    set_backward_method(BackwardEnum._Split_Dlogits_N)
    tri_out = efficient_entropy_forward(hidden, weight, labels, reduction=2, temperature=TEMPERATURE)
    logp_out, entropy_out, maximum_out, acc_out, ent_b_out = tri_out

    dlogprobs = torch.ones_like(logp_out)
    dentropy   = torch.ones_like(entropy_out)

    d_logits_buf = torch.empty((num_tokens, vocab_per_split), dtype=DTYPE, device=DEVICE_NAME)

    # CUDA: GEMM tiling；非 CUDA: 固定 TOTAL_CORE_NUM
    d_logits_grid = _gemm_grid(num_tokens, vocab_per_split)

    for split_idx in range(num_splits):
        _kernels.efficient_entropy_backward_kernel_general_d_logits_split_N[
            d_logits_grid
        ](
            split_idx, num_tokens, hidden_size, vocab_size, vocab_per_split, 0,
            hidden, hidden.stride(0), hidden.stride(1),
            weight, weight.stride(0), weight.stride(1),
            labels, labels.stride(0),
            maximum_out, maximum_out.stride(0),
            acc_out, acc_out.stride(0),
            dentropy, dentropy.stride(0),
            dlogprobs, 0, 2,
            ent_b_out, ent_b_out.stride(0),
            d_logits_buf, d_logits_buf.stride(0), d_logits_buf.stride(1),
            1.0 / TEMPERATURE,
            USE_TMA=False,
        )
    DEV.synchronize()

    # Benchmark a single operator launch (one split), not the full split loop,
    # so the reported latency is the per-launch cost of this kernel in isolation.
    split_idx = 0

    def run():
        _kernels.efficient_entropy_backward_kernel_general_d_logits_split_N[d_logits_grid](
            split_idx, num_tokens, hidden_size, vocab_size, vocab_per_split, 0,
            hidden, hidden.stride(0), hidden.stride(1),
            weight, weight.stride(0), weight.stride(1),
            labels, labels.stride(0),
            maximum_out, maximum_out.stride(0),
            acc_out, acc_out.stride(0),
            dentropy, dentropy.stride(0),
            dlogprobs, 0, 2,
            ent_b_out, ent_b_out.stride(0),
            d_logits_buf, d_logits_buf.stride(0), d_logits_buf.stride(1),
            1.0 / TEMPERATURE,
            USE_TMA=False,
        )

    # latency: triton.testing.do_bench handles its own warmup + repeated timing
    # (matching how kernel 8 is benchmarked). It returns the median latency
    # in ms plus the requested quantiles.
    quantiles = [0.5, 0.2, 0.8]
    median_ms, p20_o, p80_o = triton.testing.do_bench(run, quantiles=quantiles)

    set_backward_method(BackwardEnum._Split_Dlogits_N)
    return {
        "kernel": "efficient_entropy_backward_kernel_general_d_logits_split_N",
        "shape": (num_tokens, hidden_size, vocab_size),
        "latency_ms": round(median_ms, 3),
        "tokens_per_sec": round(num_tokens / (median_ms * 1e-3), 0),
    }


# ---------------------------------------------------------------------------
# MLU tests: end-to-end API vs PyTorch reference
# ---------------------------------------------------------------------------

def test_mlu_e2e():
    """Test end-to-end linear_cross_entropy on non-CUDA backends."""
    print("\n  [MLU/NPU mode] End-to-end linear_cross_entropy vs PyTorch reference")
    num_tokens, hidden_size, vocab_size = 512, 2048, 128256

    for red in ["none", "sum", "mean"]:
        torch.manual_seed(42)
        hidden, weight, labels = make_inputs(num_tokens, hidden_size, vocab_size, seed=42)

        ref_lp, ref_ent, _ = reference_forward(
            hidden.float(), weight.float(), labels,
            {"none": 0, "sum": 1, "mean": 2}[red], TEMPERATURE
        )

        try:
            lp, ent = linear_cross_entropy(hidden, weight, labels, TEMPERATURE, red)
            m_lp  = mad(ref_lp, lp)
            m_ent = mad(ref_ent, ent)
            print(f"    reduction={red:4s}  kernel_ok=True  "
                  f"lp_mad={m_lp:.4e}  ent_mad={m_ent:.4e}  pass={m_lp < 5e-3 or m_lp/ref_lp.abs().max().item() < 1e-2}")
        except Exception as e:
            print(f"    reduction={red:4s}  kernel_ok=False  error={e}")


def benchmark_mlu_e2e(iters=100, warmup=10):
    """End-to-end performance on MLU/NPU."""
    print("\n  [MLU/NPU mode] End-to-end performance")
    test_cases = [
        (512,  2048, 128256),
        # (4096, 2048, 128256),
        # (8192, 2048, 128256),
    ]
    print(f"  {'tokens':>8}  {'hidden':>8}  {'vocab':>8}  {'latency_ms':>12}  {'tokens/s':>12}")
    print("  " + "-" * 58)
    for nt, hs, vs in test_cases:
        torch.manual_seed(0)
        hidden, weight, labels, g_entropy, g_logprobs = make_inputs(nt, hs, vs, with_grad_inputs=True)
        for _ in range(warmup):
            lp, ent = linear_cross_entropy(hidden, weight, labels, TEMPERATURE)
            torch.autograd.grad((ent, lp), (hidden, weight), (g_entropy, g_logprobs))
        DEV.synchronize()

        start = DEV.Event(enable_timing=True)
        end = DEV.Event(enable_timing=True)
        samples = []
        for _ in range(iters):
            lp, ent = linear_cross_entropy(hidden, weight, labels, TEMPERATURE)
            start.record()
            torch.autograd.grad((ent, lp), (hidden, weight), (g_entropy, g_logprobs))
            end.record()
            DEV.synchronize()
            samples.append(start.elapsed_time(end))
        samples.sort()
        median_ms = samples[iters // 2]
        print(f"  {nt:>8}  {hs:>8}  {vs:>8}  {median_ms:>12.3f}  {nt / (median_ms * 1e-3):>12,.0f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Force line-buffered stdout so progress prints show up immediately even when
    # output is piped/redirected (non-TTY) — otherwise the header sits in a 4-8KB
    # block buffer and "nothing prints" until a kernel fills it.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cuda", "mlu", "auto"], default="auto")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    print("=" * 70)
    print("Efficient Entropy Kernel Test Suite  (9 Triton kernels)")
    print("=" * 70)
    print(f"  Device : {DEVICE_NAME}")
    print(f"  dtype  : {DTYPE}")
    print(f"  Triton : {HAVE_TRITON}")
    print(f"  TMA    : {SUPPORT_CUDA_TMA}")
    print(f"  IS_CUDA: {IS_CUDA}")

    if DEVICE_NAME == "cpu" or not DEV.is_available():
        print("[CPU — nothing to run. Exiting.]")
        return

    mode = args.mode
    if mode == "auto":
        mode = "cuda" if IS_CUDA else "mlu"

    if mode == "cuda" and DEVICE_NAME != "cuda":
        # The per-kernel tests were written for NVIDIA GPUs (BLOCK_SIZE/num_warps
        # autotune configs). On MLU/NPU the kernels still run (Triton is
        # supported), but the configs may be far from optimal and the
        # 512x2048x128256 workload can take a long time to autotune — which is
        # why it looked like "nothing printed". Proceed only if explicitly asked.
        print(f"[WARNING] --mode cuda requested but device is {DEVICE_NAME!r}. "
              f"Per-kernel configs are tuned for NVIDIA GPUs; expect slow autotune.")

    # if mode == "mlu":
    #     print("\n[Mode: MLU/NPU — end-to-end API test]")
    #     print("NOTE: Per-kernel Triton tests are NOT available on MLU/NPU.")
    #     test_mlu_e2e()
    #     benchmark_mlu_e2e(iters=args.iters, warmup=args.warmup)
    #     return

    # CUDA mode: test each kernel individually
    print("\n[Mode: CUDA — per-kernel precision + performance tests]")
    print("\n" + "=" * 70)
    print("Section 1 — Per-Kernel Accuracy")
    print("=" * 70)

    results = []

    r = test_kernel_1_general_mainloop()
    r1_pass = r.get("max_pass", False) and r.get("entropy_b_pass", False)
    print(f"    max_pass={r.get('max_pass')}  entropy_b_pass={r.get('entropy_b_pass')}  "
          f"max_mad={r.get('max_mad', -1):.4e}  ent_b_mad={r.get('entropy_b_mad', -1):.4e}")
    results.append(("Kernel1_mainloop", r1_pass))

    print("\n" + "=" * 70)
    print("Section 2 — Per-Kernel Performance  (multi-shape table)")
    print("=" * 70)
    print(f"  {'Kernel':>60}  {'shape':>22}  {'variant':>12}  {'latency_ms':>12}  {'tokens/s':>12}")
    print("  " + "-" * 126)

    _PERF_FNS = {
        1: test_kernel_1_general_mainloop,
        2: test_kernel_2_epilogue,
        3: test_kernel_3_epilogue_tp,
        4: test_kernel_4_epilogue_tp_update,
        5: test_kernel_5_backward_mainloop_mn,
        6: test_kernel_6_backward_d_hidden,
        7: test_kernel_7_backward_d_weight,
        8: test_kernel_8_backward_d_logits,
        9: test_kernel_9_backward_d_logits_split_n,
    }
    perf_results = []
    for kid, fn in _PERF_FNS.items():
        for nt, hs, vs in PERF_SHAPES[kid]:
            perf_results.append(fn(num_tokens=nt, hidden_size=hs, vocab_size=vs))
    for pr in perf_results:
        if pr.get("skipped"):
            print(f"  {pr['kernel']:>60}  {'(skipped)':>22}  {'-':>12}  {'-':>12}  {'-':>12}")
        else:
            shape = pr.get("shape", ("?", "?", "?"))
            shape_s = f"{shape[0]},{shape[1]},{shape[2]}"
            variant = pr.get("variant", "-")
            lp = pr.get("latency_ms", -1)
            tps = pr.get("tokens_per_sec", -1)
            print(f"  {pr['kernel']:>60}  {shape_s:>22}  {variant:>12}  {lp:>12.3f}  {tps:>12,.0f}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    acc_pass = all(v for _, v in results if v is not None)
    print(f"  Accuracy : {'PASSED' if acc_pass else 'FAILED'}")
    for name, v in results:
        status = "PASS" if v else ("SKIP" if v is None else "FAIL")
        print(f"    {name:40s}  {status}")

"""
H20 gpatch_v4.kernel.triton.linear_ce_kernels 测试结果:
---
                     efficient_entropy_kernel_general_mainloop         512,2048,128256             -         2.501       204,685
                     efficient_entropy_kernel_general_mainloop        4096,2048,128256             -        17.944       228,268
                      efficient_entropy_triton_kernel_epilogue         512,2048,128256             -         0.010    49,382,717
                      efficient_entropy_triton_kernel_epilogue       16384,2048,128256             -         0.025   653,895,254
                   efficient_entropy_triton_kernel_epilogue_tp         512,2048,128256             -         0.010    50,793,649
                   efficient_entropy_triton_kernel_epilogue_tp       16384,2048,128256             -         0.028   591,224,007
                   efficient_entropy_triton_epilogue_tp_update         512,2048,128256             -         0.011    46,511,628
                   efficient_entropy_triton_epilogue_tp_update       16384,2048,128256             -         0.013  1,239,709,438
         efficient_entropy_backward_kernel_general_mainloop_MN         512,2048,128256             -       110.108         4,650
         efficient_entropy_backward_kernel_general_mainloop_MN        4096,2048,128256             -       793.115         5,164
         efficient_entropy_backward_kernel_general_mainloop_MN       16384,2048,128256             -      3169.737         5,169
                    efficient_entropy_backward_kernel_d_hidden          64,2048,128256          cuda       283.421           226
                    efficient_entropy_backward_kernel_d_hidden         512,2048,128256          cuda      1429.381           358
                    efficient_entropy_backward_kernel_d_hidden        4096,2048,128256          cuda      9619.457           426
                    efficient_entropy_backward_kernel_d_weight          64,2048,128256          cuda       233.940           274
                    efficient_entropy_backward_kernel_d_weight         512,2048,128256          cuda      1173.315           436
                    efficient_entropy_backward_kernel_d_weight        4096,2048,128256          cuda      8995.072           455
            efficient_entropy_backward_kernel_general_d_logits         512,2048,128256             -        15.166        33,759
            efficient_entropy_backward_kernel_general_d_logits        8192,2048,128256             -       240.942        34,000
            efficient_entropy_backward_kernel_general_d_logits       16384,2048,128256             -       482.977        33,923
    efficient_entropy_backward_kernel_general_d_logits_split_N         512,2048,128256             -         1.139       449,470
    efficient_entropy_backward_kernel_general_d_logits_split_N       16384,2048,128256             -        35.676       459,251
----


MLU590 gpatch_v4.kernel.triton.linear_ce_kernels 结果：
(这里是强制测试的，一些看着快很多的 OP 会直接报错，结果没什么意义)

MLU590 gpatch_v4.kernel.mlu_priv.linear_ce_kernels 结果：
---
                     efficient_entropy_kernel_general_mainloop         512,2048,128256             -         2.591       197,597
                     efficient_entropy_kernel_general_mainloop        4096,2048,128256             -        20.099       203,789
                      efficient_entropy_triton_kernel_epilogue         512,2048,128256             -         0.014    37,426,901
                      efficient_entropy_triton_kernel_epilogue       16384,2048,128256             -         0.052   314,351,496
                   efficient_entropy_triton_kernel_epilogue_tp         512,2048,128256             -         0.008    64,974,616
                   efficient_entropy_triton_kernel_epilogue_tp       16384,2048,128256             -         0.061   270,541,615
                   efficient_entropy_triton_epilogue_tp_update         512,2048,128256             -         0.009    59,813,084
                   efficient_entropy_triton_epilogue_tp_update       16384,2048,128256             -         0.009  1,845,045,315
         efficient_entropy_backward_kernel_general_mainloop_MN         512,2048,128256             -        42.479        12,053
         efficient_entropy_backward_kernel_general_mainloop_MN        4096,2048,128256             -       341.105        12,008
         efficient_entropy_backward_kernel_general_mainloop_MN       16384,2048,128256             -      1333.186        12,289
                    efficient_entropy_backward_kernel_d_hidden          64,2048,128256    non-mouter        37.561         1,704
                    efficient_entropy_backward_kernel_d_hidden         512,2048,128256        mouter        13.870        36,914
                    efficient_entropy_backward_kernel_d_hidden        4096,2048,128256        mouter       110.771        36,977
                    efficient_entropy_backward_kernel_d_weight          64,2048,128256    non-mouter        15.266         4,192
                    efficient_entropy_backward_kernel_d_weight         512,2048,128256    non-mouter        25.924        19,750
                    efficient_entropy_backward_kernel_d_weight        4096,2048,128256        mouter        86.979        47,092
            efficient_entropy_backward_kernel_general_d_logits         512,2048,128256             -         4.296       119,189
            efficient_entropy_backward_kernel_general_d_logits        8192,2048,128256             -        66.974       122,316
            efficient_entropy_backward_kernel_general_d_logits       16384,2048,128256             -       133.889       122,370
    efficient_entropy_backward_kernel_general_d_logits_split_N         512,2048,128256             -         0.412     1,243,926
    efficient_entropy_backward_kernel_general_d_logits_split_N       16384,2048,128256             -        10.347     1,583,509
----

"""

if __name__ == "__main__":
    main()

