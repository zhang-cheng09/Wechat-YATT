import math
import typing

import torch
import torch.distributed as dist

try:
    import tilelang
    import tilelang.language as T
except ImportError:
    from gfused.fake_tilelang_stub import language_stub as T
    from gfused.fake_tilelang_stub import tilelang_stub as tilelang

_BACKWARD_CHUNK_SIZES = (32768, 16384, 8192)
_BACKWARD_WORKSPACE_BYTES = 1024**3
_LOG2_E = 1.4426950408889634
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
def _linear_ce_local_stats(
    hidden_size: int,
    local_vocab_size: int,
    temperature: float,
    dtype: str,
    vocab_per_split: int = 2048,
    block_M: int = 64,
    block_N: int = 256,
    block_K: int = 32,
    threads: int = 128,
    num_stages: int = 3,
):
    assert hidden_size % block_K == 0
    assert vocab_per_split % block_N == 0

    num_tokens = T.dynamic("num_tokens")
    num_splits = (local_vocab_size + vocab_per_split - 1) // vocab_per_split
    inv_temperature = 1.0 / temperature
    accum_dtype = T.float32

    @T.prim_func
    def linear_ce_local_stats_kernel(
        Hidden: T.Tensor([num_tokens, hidden_size], dtype),
        Weight: T.Tensor([local_vocab_size, hidden_size], dtype),
        PartialStats: T.Tensor([3, num_splits, num_tokens], accum_dtype),
    ):
        with T.Kernel(
            T.ceildiv(num_tokens, block_M),
            num_splits,
            threads=threads,
        ) as (pid_m, pid_n):
            hidden_shared = T.alloc_shared([block_M, block_K], dtype)
            weight_shared = T.alloc_shared([block_N, block_K], dtype)

            logits = T.alloc_fragment([block_M, block_N], accum_dtype)
            exp_values = T.alloc_fragment([block_M, block_N], accum_dtype)

            running_max = T.alloc_fragment([block_M], accum_dtype)
            previous_max = T.alloc_fragment([block_M], accum_dtype)
            running_sum = T.alloc_fragment([block_M], accum_dtype)
            running_moment = T.alloc_fragment([block_M], accum_dtype)
            max_scale = T.alloc_fragment([block_M], accum_dtype)
            block_sum = T.alloc_fragment([block_M], accum_dtype)
            block_moment = T.alloc_fragment([block_M], accum_dtype)

            T.fill(running_max, -T.infinity(accum_dtype))
            T.fill(running_sum, 0)
            T.fill(running_moment, 0)

            for tile_n in T.serial(vocab_per_split // block_N):
                vocab_start = pid_n * vocab_per_split + tile_n * block_N
                T.clear(logits)
                for tile_k in T.Pipelined(
                    hidden_size // block_K,
                    num_stages=num_stages,
                ):
                    T.copy(
                        Hidden[
                            pid_m * block_M:(pid_m + 1) * block_M,
                            tile_k * block_K:(tile_k + 1) * block_K,
                        ],
                        hidden_shared,
                    )
                    T.copy(
                        Weight[
                            vocab_start:vocab_start + block_N,
                            tile_k * block_K:(tile_k + 1) * block_K,
                        ],
                        weight_shared,
                    )
                    T.gemm(
                        hidden_shared,
                        weight_shared,
                        logits,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for i, j in T.Parallel(block_M, block_N):
                    token_idx = pid_m * block_M + i
                    vocab_idx = vocab_start + j
                    valid = (
                        token_idx < num_tokens and vocab_idx < local_vocab_size and
                        vocab_idx < (pid_n + 1) * vocab_per_split
                    )
                    logits[i, j] = T.if_then_else(
                        valid,
                        logits[i, j] * inv_temperature,
                        -T.infinity(accum_dtype),
                    )

                T.copy(running_max, previous_max)
                T.fill(running_max, -T.infinity(accum_dtype))
                T.reduce_max(logits, running_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    running_max[i] = T.max(running_max[i], previous_max[i])
                    max_scale[i] = T.exp2((previous_max[i] - running_max[i]) * _LOG2_E)

                for i, j in T.Parallel(block_M, block_N):
                    token_idx = pid_m * block_M + i
                    vocab_idx = vocab_start + j
                    valid = (
                        token_idx < num_tokens and vocab_idx < local_vocab_size and
                        vocab_idx < (pid_n + 1) * vocab_per_split
                    )
                    exp_values[i, j] = T.if_then_else(
                        valid,
                        T.exp2((logits[i, j] - running_max[i]) * _LOG2_E),
                        0,
                    )

                T.reduce_sum(exp_values, block_sum, dim=1, clear=True)
                for i, j in T.Parallel(block_M, block_N):
                    token_idx = pid_m * block_M + i
                    vocab_idx = vocab_start + j
                    valid = (
                        token_idx < num_tokens and vocab_idx < local_vocab_size and
                        vocab_idx < (pid_n + 1) * vocab_per_split
                    )
                    exp_values[i, j] = T.if_then_else(
                        valid,
                        exp_values[i, j] * logits[i, j],
                        0,
                    )
                T.reduce_sum(
                    exp_values,
                    block_moment,
                    dim=1,
                    clear=True,
                )
                for i in T.Parallel(block_M):
                    running_sum[i] = (running_sum[i] * max_scale[i] + block_sum[i])
                    running_moment[i] = (running_moment[i] * max_scale[i] + block_moment[i])

            for i in T.Parallel(block_M):
                token_idx = pid_m * block_M + i
                if token_idx < num_tokens:
                    PartialStats[0, pid_n, token_idx] = running_max[i]
                    PartialStats[1, pid_n, token_idx] = running_sum[i]
                    PartialStats[2, pid_n, token_idx] = running_moment[i]

    return linear_ce_local_stats_kernel


@tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
def _linear_ce_target_logits(
    hidden_size: int,
    local_vocab_size: int,
    temperature: float,
    dtype: str,
    block_K: int = 128,
    threads: int = 128,
):
    assert hidden_size % block_K == 0

    num_tokens = T.dynamic("num_tokens")
    inv_temperature = 1.0 / temperature
    accum_dtype = T.float32

    @T.prim_func
    def linear_ce_target_logits_kernel(
        Hidden: T.Tensor([num_tokens, hidden_size], dtype),
        Weight: T.Tensor([local_vocab_size, hidden_size], dtype),
        Labels: T.Tensor([num_tokens], "int64"),
        VocabOffset: T.int64,
        TargetLogits: T.Tensor([num_tokens], accum_dtype),
    ):
        with T.Kernel(num_tokens, threads=threads) as pid:
            hidden_fragment = T.alloc_fragment([block_K], accum_dtype)
            weight_fragment = T.alloc_fragment([block_K], accum_dtype)
            product = T.alloc_fragment([block_K], accum_dtype)
            block_sum = T.alloc_fragment([1], accum_dtype)
            target_logit = T.alloc_fragment([1], accum_dtype)

            local_label = Labels[pid] - VocabOffset
            T.fill(target_logit, 0)
            if local_label >= 0 and local_label < local_vocab_size:
                for tile_k in T.serial(hidden_size // block_K):
                    for k in T.Parallel(block_K):
                        hidden_fragment[k] = Hidden[
                            pid,
                            tile_k * block_K + k,
                        ]
                        weight_fragment[k] = Weight[
                            local_label,
                            tile_k * block_K + k,
                        ]
                        product[k] = hidden_fragment[k] * weight_fragment[k]
                    T.reduce_sum(product, block_sum, dim=0, clear=True)
                    target_logit[0] += block_sum[0]
            TargetLogits[pid] = target_logit[0] * inv_temperature

    return linear_ce_target_logits_kernel


@tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
def _linear_ce_dlogits(
    hidden_size: int,
    temperature: float,
    dtype: str,
    block_M: int = 64,
    block_N: int = 256,
    block_K: int = 32,
    threads: int = 128,
    num_stages: int = 3,
):
    assert hidden_size % block_K == 0

    num_tokens = T.dynamic("num_tokens")
    vocab_size = T.dynamic("vocab_size")
    inv_temperature = 1.0 / temperature
    accum_dtype = T.float32

    @T.prim_func
    def linear_ce_dlogits_kernel(
        Hidden: T.Tensor([num_tokens, hidden_size], dtype),
        Weight: T.Tensor([vocab_size, hidden_size], dtype),
        Labels: T.Tensor([num_tokens], "int64"),
        Lse: T.Tensor([num_tokens], accum_dtype),
        MeanLogit: T.Tensor([num_tokens], accum_dtype),
        GradLogProbs: T.Tensor([num_tokens], accum_dtype),
        GradEntropy: T.Tensor([num_tokens], accum_dtype),
        VocabOffset: T.int64,
        DLogits: T.Tensor([num_tokens, vocab_size], dtype),
    ):
        with T.Kernel(
            T.ceildiv(num_tokens, block_M),
            T.ceildiv(vocab_size, block_N),
            threads=threads,
        ) as (pid_m, pid_n):
            hidden_shared = T.alloc_shared([block_M, block_K], dtype)
            weight_shared = T.alloc_shared([block_N, block_K], dtype)
            logits = T.alloc_fragment([block_M, block_N], accum_dtype)

            T.clear(logits)
            for tile_k in T.Pipelined(
                hidden_size // block_K,
                num_stages=num_stages,
            ):
                T.copy(
                    Hidden[
                        pid_m * block_M:(pid_m + 1) * block_M,
                        tile_k * block_K:(tile_k + 1) * block_K,
                    ],
                    hidden_shared,
                )
                T.copy(
                    Weight[
                        pid_n * block_N:(pid_n + 1) * block_N,
                        tile_k * block_K:(tile_k + 1) * block_K,
                    ],
                    weight_shared,
                )
                T.gemm(
                    hidden_shared,
                    weight_shared,
                    logits,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            for i, j in T.Parallel(block_M, block_N):
                token_idx = pid_m * block_M + i
                vocab_idx = pid_n * block_N + j
                if token_idx < num_tokens and vocab_idx < vocab_size:
                    scaled_logit = logits[i, j] * inv_temperature
                    probability = T.exp2((scaled_logit - Lse[token_idx]) * _LOG2_E)
                    is_label = T.if_then_else(
                        VocabOffset + vocab_idx == Labels[token_idx],
                        1.0,
                        0.0,
                    )
                    grad_scaled_logit = (
                        GradLogProbs[token_idx] * (is_label - probability) -
                        GradEntropy[token_idx] * probability *
                        (scaled_logit - MeanLogit[token_idx])
                    )
                    DLogits[token_idx, vocab_idx] = (grad_scaled_logit * inv_temperature)

    return linear_ce_dlogits_kernel


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise TypeError(
        "TileLang linear_cross_entropy supports bfloat16/float16 inputs, "
        f"got {dtype}"
    )


def _world_size(group: dist.ProcessGroup | None) -> int:
    return 1 if group is None else dist.get_world_size(group)


def _vocab_offset(
    local_vocab_size: int,
    group: dist.ProcessGroup | None,
) -> int:
    rank = 0 if group is None else dist.get_rank(group)
    return rank * local_vocab_size


def _backward_chunk_size(
    num_tokens: int,
    element_size: int,
    local_vocab_size: int,
) -> int:
    for chunk_size in _BACKWARD_CHUNK_SIZES:
        actual_chunk_size = min(chunk_size, local_vocab_size)
        workspace_bytes = num_tokens * actual_chunk_size * element_size
        if workspace_bytes <= _BACKWARD_WORKSPACE_BYTES:
            return actual_chunk_size
    return min(_BACKWARD_CHUNK_SIZES[-1], local_vocab_size)


def _forward_token_stats(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    group: dist.ProcessGroup | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    local_vocab_size = weight.shape[0]
    vocab_per_split = min(2048, local_vocab_size)
    vocab_per_split = max(128, vocab_per_split // 128 * 128)
    vocab_offset = _vocab_offset(local_vocab_size, group)

    stats_kernel = _linear_ce_local_stats(
        hidden.shape[1],
        local_vocab_size,
        temperature,
        _dtype_name(hidden.dtype),
        vocab_per_split=vocab_per_split,
    )
    target_block_k = math.gcd(hidden.shape[1], 256)
    target_kernel = _linear_ce_target_logits(
        hidden.shape[1],
        local_vocab_size,
        temperature,
        _dtype_name(hidden.dtype),
        block_K=target_block_k,
        threads=max(128, target_block_k),
    )
    partial_stats = stats_kernel(hidden, weight)
    target_logits = target_kernel(
        hidden,
        weight,
        labels,
        vocab_offset,
    )

    local_maximum = partial_stats[0].amax(dim=0)
    global_maximum = local_maximum.clone()
    if _world_size(group) > 1:
        dist.all_reduce(
            global_maximum,
            op=dist.ReduceOp.MAX,
            group=group,
        )

    scale = torch.exp(partial_stats[0] - global_maximum.unsqueeze(0))
    global_stats = torch.stack(
        (
            torch.sum(partial_stats[1] * scale, dim=0),
            torch.sum(partial_stats[2] * scale, dim=0),
            target_logits,
        )
    )
    if _world_size(group) > 1:
        dist.all_reduce(
            global_stats,
            op=dist.ReduceOp.SUM,
            group=group,
        )

    lse = global_maximum + torch.log(global_stats[0])
    mean_logit = global_stats[1] / global_stats[0]
    log_probs = global_stats[2] - lse
    entropy = lse - mean_logit
    return log_probs, entropy, lse, mean_logit


def _backward_projection(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    mean_logit: torch.Tensor,
    grad_log_probs: torch.Tensor,
    grad_entropy: torch.Tensor,
    temperature: float,
    group: dist.ProcessGroup | None,
    compute_hidden_grad: bool,
    compute_weight_grad: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    hidden_grad = (torch.empty_like(hidden) if compute_hidden_grad else None)
    weight_grad = (torch.empty_like(weight) if compute_weight_grad else None)
    dlogits_kernel = _linear_ce_dlogits(
        hidden.shape[1],
        temperature,
        _dtype_name(hidden.dtype),
    )
    vocab_offset = _vocab_offset(weight.shape[0], group)
    chunk_size = _backward_chunk_size(
        hidden.shape[0],
        hidden.element_size(),
        weight.shape[0],
    )

    for start in range(0, weight.shape[0], chunk_size):
        end = min(start + chunk_size, weight.shape[0])
        weight_chunk = weight[start:end]
        dlogits = dlogits_kernel(
            hidden,
            weight_chunk,
            labels,
            lse,
            mean_logit,
            grad_log_probs,
            grad_entropy,
            vocab_offset + start,
        )
        if compute_hidden_grad:
            assert hidden_grad is not None
            if start == 0:
                torch.mm(
                    dlogits,
                    weight_chunk,
                    out=hidden_grad,
                )
            else:
                hidden_grad.addmm_(dlogits, weight_chunk)
        if compute_weight_grad:
            assert weight_grad is not None
            torch.mm(
                dlogits.transpose(0, 1),
                hidden,
                out=weight_grad[start:end],
            )
        del dlogits

    return hidden_grad, weight_grad


class _LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        dist_process_group: dist.ProcessGroup | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_hidden_shape = hidden.shape
        hidden = hidden.reshape(-1, hidden.shape[-1]).contiguous()
        weight = weight.contiguous()
        labels = labels.reshape(-1).contiguous()
        log_probs, entropy, lse, mean_logit = _forward_token_stats(
            hidden,
            weight,
            labels,
            temperature,
            dist_process_group,
        )

        ctx.save_for_backward(
            hidden,
            weight,
            labels,
            lse,
            mean_logit,
        )
        ctx.original_hidden_shape = original_hidden_shape
        ctx.temperature = temperature
        ctx.dist_process_group = dist_process_group
        ctx.compute_hidden_grad = ctx.needs_input_grad[0]
        ctx.compute_weight_grad = ctx.needs_input_grad[1]
        return log_probs, entropy

    @staticmethod
    def backward(
        ctx,
        grad_log_probs: torch.Tensor | None,
        grad_entropy: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
    ]:
        hidden, weight, labels, lse, mean_logit = ctx.saved_tensors
        if grad_log_probs is None:
            grad_log_probs = torch.zeros_like(lse)
        else:
            grad_log_probs = grad_log_probs.reshape(-1).float().contiguous()
        if grad_entropy is None:
            grad_entropy = torch.zeros_like(lse)
        else:
            grad_entropy = grad_entropy.reshape(-1).float().contiguous()

        hidden_grad, weight_grad = _backward_projection(
            hidden,
            weight,
            labels,
            lse,
            mean_logit,
            grad_log_probs,
            grad_entropy,
            ctx.temperature,
            ctx.dist_process_group,
            ctx.compute_hidden_grad,
            ctx.compute_weight_grad,
        )
        if hidden_grad is not None:
            hidden_grad = hidden_grad.to(hidden.dtype).view(ctx.original_hidden_shape)
        if weight_grad is not None:
            weight_grad = weight_grad.to(weight.dtype)
        return hidden_grad, weight_grad, None, None, None


def _validate_inputs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    reduction: str,
    dist_process_group: dist.ProcessGroup | None,
) -> None:
    if (not isinstance(temperature, float) or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature must be a finite positive float, "
                         f"got {temperature!r}")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"Invalid reduction: {reduction!r}; choose from none, sum, mean")
    assert hidden.ndim >= 2 and weight.ndim == 2
    assert labels.shape == hidden.shape[:-1]
    assert hidden.shape[-1] == weight.shape[1]
    assert hidden.dtype == weight.dtype
    _dtype_name(hidden.dtype)
    assert hidden.shape[-1] % 32 == 0
    assert weight.shape[0] > 0
    assert labels.dtype == torch.long
    assert hidden.is_cuda and weight.is_cuda and labels.is_cuda
    assert hidden.device == weight.device == labels.device
    assert dist_process_group is None or dist.is_initialized()


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: typing.Optional[float] = 1.0,
    reduction: typing.Optional[str] = "none",
    dist_process_group: typing.Optional[dist.ProcessGroup] = None,
    return_entropy: bool = False,
):
    """Compute fused linear cross entropy with optional token entropy.

    Parameters
    ----------
    hidden : torch.Tensor
        Shape ``[..., H]`` with FP16 or BF16 elements.
    weight : torch.Tensor
        Shape ``[V, H]``. Under vocabulary parallelism, ``V`` is the
        equally sized rank-local vocabulary.
    labels : torch.Tensor
        Global token ids with shape ``hidden.shape[:-1]``.
    temperature : float
        Finite positive softmax temperature.
    reduction : str
        One of ``"none"``, ``"sum"``, or ``"mean"``.
    dist_process_group : torch.distributed.ProcessGroup or None
        Vocabulary-parallel group. ``None`` means `weight` contains the
        complete vocabulary.
    return_entropy : bool
        Return per-token entropy together with the loss.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        NLL loss and, when requested, FP32 per-token entropy.

    Raises
    ------
    ValueError
        If `temperature` or `reduction` is invalid.
    TypeError
        If projection tensors do not use FP16 or BF16.
    AssertionError
        If tensor shapes, layouts, devices, or TP state are invalid.

    Notes
    -----
    Hidden gradients are rank-local vocabulary contributions; the caller's
    sequence-parallel reduction must aggregate them. Backward selects 8K,
    16K, or 32K local-vocabulary chunks using a 1 GiB workspace budget and
    an 8K minimum chunk.
    """
    assert temperature is not None
    assert reduction is not None
    reduction = reduction.lower()
    _validate_inputs(
        hidden,
        weight,
        labels,
        temperature,
        reduction,
        dist_process_group,
    )
    if labels.numel() == 0:
        zero = (hidden.reshape(-1)[:0].sum().float() * 0 + weight.reshape(-1)[:1].sum().float() * 0)
        loss = hidden[..., 0].float() * 0 + zero
        entropy = loss
        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            loss = loss.mean()
        if return_entropy:
            return loss, entropy.view(labels.shape)
        return loss

    log_probs, entropy = _LinearCrossEntropy.apply(
        hidden,
        weight,
        labels,
        temperature,
        dist_process_group,
    )
    loss = -log_probs
    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()
    else:
        loss = loss.view(labels.shape)

    if return_entropy:
        return loss, entropy.view(labels.shape)
    return loss
