import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    import tilelang
    import tilelang.language as T
except ImportError:
    from gfused.fake_tilelang_stub import language_stub as T
    from gfused.fake_tilelang_stub import tilelang_stub as tilelang

_SPLIT_N_SIZE = 8192
_LOG2_E = 1.4426950408889634
_PASS_CONFIGS = {
    # 动态输出索引会触发误报。
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    # 当前双流水循环无法通过 warp-specialized 重写。
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(out_idx=[-2, -1], pass_configs=_PASS_CONFIGS)
def _linear_kl_local_stats(
    student_hidden_size: int,
    teacher_hidden_size: int,
    local_vocab_size: int,
    temperature: float,
    dtype: str,
    vocab_per_split: int = 2048,
    block_M: int = 32,
    block_N: int = 128,
    block_K: int = 32,
    threads: int = 256,
    num_stages: int = 2,
):
    assert student_hidden_size % block_K == 0
    assert teacher_hidden_size % block_K == 0
    assert vocab_per_split % block_N == 0

    num_tokens = T.dynamic("num_tokens")
    num_splits = (local_vocab_size + vocab_per_split - 1) // vocab_per_split
    inv_temperature = 1.0 / temperature
    accum_dtype = T.float32

    @T.prim_func
    def linear_kl_local_stats_kernel(
        StudentHidden: T.Tensor([num_tokens, student_hidden_size], dtype),
        TeacherHidden: T.Tensor([num_tokens, teacher_hidden_size], dtype),
        StudentWeight: T.Tensor([local_vocab_size, student_hidden_size], dtype),
        TeacherWeight: T.Tensor([local_vocab_size, teacher_hidden_size], dtype),
        PartialMaxima: T.Tensor([2, num_splits, num_tokens], accum_dtype),
        PartialMoments: T.Tensor([3, num_splits, num_tokens], accum_dtype),
    ):
        # block_M 是 token 维度
        # block_K 是 hidden_state 维度
        # block_N 是 vocab_size 维度
        # 每个 block 按 block_N 流式处理一块 [block_M, vocab_per_split] 区域。
        with T.Kernel(
            T.ceildiv(num_tokens, block_M),
            num_splits,
            threads=threads,
        ) as (pid_m, pid_n):  # bx, by
            hidden_shared = T.alloc_shared([block_M, block_K], dtype)
            weight_shared = T.alloc_shared([block_N, block_K], dtype)
            logits_cast = T.alloc_shared([block_M, block_N], dtype)

            student_logits = T.alloc_fragment([block_M, block_N], accum_dtype)
            teacher_logits = T.alloc_fragment([block_M, block_N], accum_dtype)
            exp_values = T.alloc_fragment([block_M, block_N], accum_dtype)

            student_max = T.alloc_fragment([block_M], accum_dtype)
            student_max_prev = T.alloc_fragment([block_M], accum_dtype)
            student_scale = T.alloc_fragment([block_M], accum_dtype)
            student_sum = T.alloc_fragment([block_M], accum_dtype)
            student_block_sum = T.alloc_fragment([block_M], accum_dtype)

            teacher_max = T.alloc_fragment([block_M], accum_dtype)
            teacher_max_prev = T.alloc_fragment([block_M], accum_dtype)
            teacher_scale = T.alloc_fragment([block_M], accum_dtype)
            teacher_sum = T.alloc_fragment([block_M], accum_dtype)
            teacher_block_sum = T.alloc_fragment([block_M], accum_dtype)

            cross = T.alloc_fragment([block_M], accum_dtype)
            cross_block = T.alloc_fragment([block_M], accum_dtype)

            T.fill(student_max, -T.infinity(accum_dtype))
            T.fill(teacher_max, -T.infinity(accum_dtype))
            T.fill(student_sum, 0)
            T.fill(teacher_sum, 0)
            T.fill(cross, 0)

            # 内部按 block_N 分块串行处理。
            for tile_n in T.serial(vocab_per_split // block_N):
                vocab_start = (pid_n * vocab_per_split + tile_n * block_N)

                T.clear(student_logits)
                # 计算 student [block_M, block_N] 的 logits
                for tile_k in T.Pipelined(
                    student_hidden_size // block_K,
                    num_stages=num_stages,
                ):
                    T.copy(
                        StudentHidden[
                            pid_m * block_M:(pid_m + 1) * block_M,
                            tile_k * block_K:(tile_k + 1) * block_K,
                        ],
                        hidden_shared,
                    )
                    T.copy(
                        StudentWeight[
                            vocab_start:vocab_start + block_N,
                            tile_k * block_K:(tile_k + 1) * block_K,
                        ],
                        weight_shared,
                    )
                    T.gemm(
                        hidden_shared,
                        weight_shared,
                        student_logits,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                # 先按线性层输出精度舍入，再用 FP32 累积 KL。
                T.copy(student_logits, logits_cast)
                for i, j in T.Parallel(block_M, block_N):
                    token_idx = pid_m * block_M + i
                    vocab_idx = vocab_start + j
                    valid = (
                        token_idx < num_tokens and vocab_idx < local_vocab_size and
                        vocab_idx < (pid_n + 1) * vocab_per_split
                    )
                    student_logits[i, j] = T.if_then_else(
                        valid,
                        logits_cast[i, j] * inv_temperature,
                        -T.infinity(accum_dtype),
                    )

                T.clear(teacher_logits)
                # 计算 teacher [block_M, block_N] 的 logits
                for tile_k in T.Pipelined(
                    teacher_hidden_size // block_K,
                    num_stages=num_stages,
                ):
                    T.copy(
                        TeacherHidden[
                            pid_m * block_M:(pid_m + 1) * block_M,
                            tile_k * block_K:(tile_k + 1) * block_K,
                        ],
                        hidden_shared,
                    )
                    T.copy(
                        TeacherWeight[
                            vocab_start:vocab_start + block_N,
                            tile_k * block_K:(tile_k + 1) * block_K,
                        ],
                        weight_shared,
                    )
                    T.gemm(
                        hidden_shared,
                        weight_shared,
                        teacher_logits,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                T.copy(teacher_logits, logits_cast)
                for i, j in T.Parallel(block_M, block_N):
                    token_idx = pid_m * block_M + i
                    vocab_idx = vocab_start + j
                    valid = (
                        token_idx < num_tokens and vocab_idx < local_vocab_size and
                        vocab_idx < (pid_n + 1) * vocab_per_split
                    )
                    teacher_logits[i, j] = T.if_then_else(
                        valid,
                        logits_cast[i, j] * inv_temperature,
                        -T.infinity(accum_dtype),
                    )

                # teacher online softmax
                T.copy(teacher_max, teacher_max_prev)
                T.fill(teacher_max, -T.infinity(accum_dtype))
                T.reduce_max(
                    teacher_logits,
                    teacher_max,
                    dim=1,
                    clear=False,
                )
                for i in T.Parallel(block_M):
                    teacher_max[i] = T.max(teacher_max[i], teacher_max_prev[i])
                    teacher_scale[i] = T.exp2((teacher_max_prev[i] - teacher_max[i]) * _LOG2_E)
                for i, j in T.Parallel(block_M, block_N):
                    token_idx = pid_m * block_M + i
                    vocab_idx = vocab_start + j
                    valid = (
                        token_idx < num_tokens and vocab_idx < local_vocab_size and
                        vocab_idx < (pid_n + 1) * vocab_per_split
                    )
                    exp_values[i, j] = T.if_then_else(
                        valid,
                        T.exp2((teacher_logits[i, j] - teacher_max[i]) * _LOG2_E),
                        0,
                    )
                T.reduce_sum(
                    exp_values,
                    teacher_block_sum,
                    dim=1,
                    clear=True,
                )
                # online 累加 sum(exp(teacher_logits - max))
                for i in T.Parallel(block_M):
                    teacher_sum[i] = (teacher_sum[i] * teacher_scale[i] + teacher_block_sum[i])

                # t_j/s_j 是缩放 logits，q_j/p_j 是教师/学生概率。
                # KL = sum(q_j * (t_j - s_j)) + logZ_s - logZ_t。
                for i, j in T.Parallel(block_M, block_N):
                    token_idx = pid_m * block_M + i
                    vocab_idx = vocab_start + j
                    valid = (
                        token_idx < num_tokens and vocab_idx < local_vocab_size and
                        vocab_idx < (pid_n + 1) * vocab_per_split
                    )
                    exp_values[i, j] = T.if_then_else(
                        valid,
                        exp_values[i, j] * (teacher_logits[i, j] - student_logits[i, j]),
                        0,
                    )
                T.reduce_sum(
                    exp_values,
                    cross_block,
                    dim=1,
                    clear=True,
                )
                # online 累加
                for i in T.Parallel(block_M):
                    cross[i] = (cross[i] * teacher_scale[i] + cross_block[i])

                T.copy(student_max, student_max_prev)
                T.fill(student_max, -T.infinity(accum_dtype))
                T.reduce_max(
                    student_logits,
                    student_max,
                    dim=1,
                    clear=False,
                )
                for i in T.Parallel(block_M):
                    student_max[i] = T.max(student_max[i], student_max_prev[i])
                    student_scale[i] = T.exp2((student_max_prev[i] - student_max[i]) * _LOG2_E)
                for i, j in T.Parallel(block_M, block_N):
                    token_idx = pid_m * block_M + i
                    vocab_idx = vocab_start + j
                    valid = (
                        token_idx < num_tokens and vocab_idx < local_vocab_size and
                        vocab_idx < (pid_n + 1) * vocab_per_split
                    )
                    exp_values[i, j] = T.if_then_else(
                        valid,
                        T.exp2((student_logits[i, j] - student_max[i]) * _LOG2_E),
                        0,
                    )
                T.reduce_sum(
                    exp_values,
                    student_block_sum,
                    dim=1,
                    clear=True,
                )
                # online 累加 sum(exp(student_logits - max))
                for i in T.Parallel(block_M):
                    student_sum[i] = (student_sum[i] * student_scale[i] + student_block_sum[i])

            # 跨 block 聚合
            for i in T.Parallel(block_M):
                token_idx = pid_m * block_M + i
                if token_idx < num_tokens:
                    # pid_n 是 by，num_splits 那一维
                    PartialMaxima[0, pid_n, token_idx] = student_max[i]
                    PartialMaxima[1, pid_n, token_idx] = teacher_max[i]
                    PartialMoments[0, pid_n, token_idx] = student_sum[i]
                    PartialMoments[1, pid_n, token_idx] = teacher_sum[i]
                    PartialMoments[2, pid_n, token_idx] = cross[i]

    return linear_kl_local_stats_kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _linear_kl_dlogits(
    temperature: float,
    dtype: str,
    compute_teacher_grad: bool,
    block_size: int = 256,
    threads: int = 256,
):
    num_tokens = T.dynamic("num_tokens")
    vocab_size = T.dynamic("vocab_size")
    inv_temperature = 1.0 / temperature
    accum_dtype = T.float32

    @T.prim_func
    def linear_kl_dlogits_kernel(
        StudentLogits: T.Tensor([num_tokens, vocab_size], dtype),
        TeacherLogits: T.Tensor([num_tokens, vocab_size], dtype),
        StudentLse: T.Tensor([num_tokens], accum_dtype),
        TeacherLse: T.Tensor([num_tokens], accum_dtype),
        TokenKl: T.Tensor([num_tokens], accum_dtype),
        GradKl: T.Tensor([num_tokens], accum_dtype),
    ):
        with T.Kernel(
            T.ceildiv(num_tokens * vocab_size, block_size),
            threads=threads,
        ) as pid:
            for i in T.Parallel(block_size):
                offset = pid * block_size + i
                if offset < num_tokens * vocab_size:
                    token_idx = offset // vocab_size
                    student_logit = (
                        StudentLogits[token_idx, offset % vocab_size] * inv_temperature
                    )
                    teacher_logit = (
                        TeacherLogits[token_idx, offset % vocab_size] * inv_temperature
                    )
                    student_prob = T.exp2((student_logit - StudentLse[token_idx]) * _LOG2_E)
                    teacher_prob = T.exp2((teacher_logit - TeacherLse[token_idx]) * _LOG2_E)
                    # GradKl 已在算子外包含掩码、alpha 和 T^2。
                    grad_scale = GradKl[token_idx] * inv_temperature
                    StudentLogits[token_idx, offset %
                                  vocab_size] = (grad_scale * (student_prob - teacher_prob))
                    if compute_teacher_grad:
                        TeacherLogits[token_idx, offset % vocab_size] = (
                            grad_scale * teacher_prob * (
                                teacher_logit - TeacherLse[token_idx] - student_logit +
                                StudentLse[token_idx] - TokenKl[token_idx]
                            )
                        )

    return linear_kl_dlogits_kernel


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise TypeError(f"linear_full_kl supports bfloat16/float16 inputs, got {dtype}")


def _world_size(group: dist.ProcessGroup | None) -> int:
    return 1 if group is None else dist.get_world_size(group)


def _forward_token_kl(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_weight: torch.Tensor,
    temperature: float,
    group: dist.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = student_hidden.shape[0]
    local_vocab_size = student_weight.shape[0]
    vocab_per_split = min(2048, local_vocab_size)
    vocab_per_split = max(128, vocab_per_split // 128 * 128)

    stats_kernel = _linear_kl_local_stats(
        student_hidden.shape[1],
        teacher_hidden.shape[1],
        local_vocab_size,
        temperature,
        _dtype_name(student_hidden.dtype),
        vocab_per_split=vocab_per_split,
    )
    partial_maxima, partial_moments = stats_kernel(
        student_hidden,
        teacher_hidden,
        student_weight,
        teacher_weight,
    )

    local_maxima = partial_maxima.amax(dim=1)
    global_maxima = local_maxima.clone()
    if _world_size(group) > 1:
        dist.all_reduce(
            global_maxima,
            op=dist.ReduceOp.MAX,
            group=group,
        )

    # 每个 split 直接对齐到全局最大值，避免重复重标定。
    student_scale = torch.exp(partial_maxima[0] - global_maxima[0].unsqueeze(0))
    teacher_scale = torch.exp(partial_maxima[1] - global_maxima[1].unsqueeze(0))
    local_moments = torch.stack(
        (
            torch.sum(partial_moments[0] * student_scale, dim=0),
            torch.sum(partial_moments[1] * teacher_scale, dim=0),
            torch.sum(partial_moments[2] * teacher_scale, dim=0),
        )
    )
    if _world_size(group) > 1:
        dist.all_reduce(
            local_moments,
            op=dist.ReduceOp.SUM,
            group=group,
        )

    student_lse = global_maxima[0] + torch.log(local_moments[0])
    teacher_lse = global_maxima[1] + torch.log(local_moments[1])
    # KL(q||p) = E_q[t-s] + logZ_s - logZ_t。
    token_kl = (local_moments[2] / local_moments[1] + student_lse - teacher_lse)
    assert token_kl.shape == (num_tokens, )
    return token_kl, student_lse, teacher_lse


def _dlogits_backward(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_weight: torch.Tensor,
    student_lse: torch.Tensor,
    teacher_lse: torch.Tensor,
    token_kl: torch.Tensor,
    grad_kl: torch.Tensor,
    temperature: float,
    vocab_chunk_size: int,
    compute_student_hidden_grad: bool,
    compute_teacher_hidden_grad: bool,
    compute_student_weight_grad: bool,
    compute_teacher_weight_grad: bool,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    local_vocab_size = student_weight.shape[0]
    student_hidden_grad = (
        torch.zeros_like(student_hidden, dtype=torch.float32)
        if compute_student_hidden_grad else None
    )
    student_weight_grad = (
        torch.empty_like(student_weight) if compute_student_weight_grad else None
    )
    teacher_hidden_grad = (
        torch.zeros_like(teacher_hidden, dtype=torch.float32)
        if compute_teacher_hidden_grad else None
    )
    teacher_weight_grad = (
        torch.empty_like(teacher_weight) if compute_teacher_weight_grad else None
    )
    compute_teacher_grad = (compute_teacher_hidden_grad or compute_teacher_weight_grad)

    dlogits_kernel = _linear_kl_dlogits(
        temperature,
        _dtype_name(student_hidden.dtype),
        compute_teacher_grad,
    )
    for start in range(0, local_vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, local_vocab_size)
        student_weight_chunk = student_weight[start:end]
        teacher_weight_chunk = teacher_weight[start:end]
        student_dlogits = F.linear(student_hidden, student_weight_chunk)
        teacher_dlogits = F.linear(teacher_hidden, teacher_weight_chunk)
        # 临时输出原地改写为梯度，避免再分配同尺寸缓冲区。
        dlogits_kernel(
            student_dlogits,
            teacher_dlogits,
            student_lse,
            teacher_lse,
            token_kl,
            grad_kl,
        )

        if compute_student_hidden_grad:
            assert student_hidden_grad is not None
            student_hidden_grad.add_(torch.mm(student_dlogits, student_weight_chunk).float())
        if compute_student_weight_grad:
            assert student_weight_grad is not None
            student_weight_grad[start:end] = torch.mm(
                student_dlogits.transpose(0, 1),
                student_hidden,
            )
        if compute_teacher_hidden_grad:
            assert teacher_hidden_grad is not None
            teacher_hidden_grad.add_(torch.mm(teacher_dlogits, teacher_weight_chunk).float())
        if compute_teacher_weight_grad:
            assert teacher_weight_grad is not None
            teacher_weight_grad[start:end] = torch.mm(
                teacher_dlogits.transpose(0, 1),
                teacher_hidden,
            )

    return (
        student_hidden_grad,
        student_weight_grad,
        teacher_hidden_grad,
        teacher_weight_grad,
    )


def _validate_projection_inputs(
    student_hidden_states: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    student_output_weight: torch.Tensor,
    teacher_output_weight: torch.Tensor,
    tp_group: dist.ProcessGroup | None,
) -> None:
    assert student_hidden_states.ndim == teacher_hidden_states.ndim == 3
    assert student_output_weight.ndim == teacher_output_weight.ndim == 2
    seq_len, batch, student_hidden_size = student_hidden_states.shape
    teacher_hidden_size = teacher_hidden_states.shape[2]
    local_vocab_size = student_output_weight.shape[0]
    assert teacher_hidden_states.shape[:2] == (batch, seq_len)
    assert student_output_weight.shape[1] == student_hidden_size
    assert teacher_output_weight.shape == (
        local_vocab_size,
        teacher_hidden_size,
    )
    assert (
        student_hidden_states.dtype == student_output_weight.dtype == teacher_output_weight.dtype
    )
    _dtype_name(student_hidden_states.dtype)
    assert student_hidden_size % 32 == teacher_hidden_size % 32 == 0
    assert local_vocab_size > 0
    tensors = (
        student_hidden_states,
        teacher_hidden_states,
        student_output_weight,
        teacher_output_weight,
    )
    assert all(
        tensor.is_cuda and tensor.device == student_hidden_states.device for tensor in tensors
    )
    assert tp_group is None or dist.is_initialized()


class _LinearFullKl(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        student_output_weight: torch.Tensor,
        teacher_output_weight: torch.Tensor,
        temperature: float,
        tp_group: dist.ProcessGroup | None,
    ) -> torch.Tensor:
        assert student_hidden.ndim == teacher_hidden.ndim == 2
        assert student_hidden.shape[0] == teacher_hidden.shape[0]

        student_input_dtype = student_hidden.dtype
        teacher_input_dtype = teacher_hidden.dtype
        teacher_hidden = (teacher_hidden.to(dtype=teacher_output_weight.dtype).contiguous())
        student_hidden = student_hidden.contiguous()
        student_weight = student_output_weight.contiguous()
        teacher_weight = teacher_output_weight.contiguous()
        token_kl, student_lse, teacher_lse = _forward_token_kl(
            student_hidden,
            teacher_hidden,
            student_weight,
            teacher_weight,
            temperature,
            tp_group,
        )

        ctx.save_for_backward(
            student_hidden,
            teacher_hidden,
            student_weight,
            teacher_weight,
            student_lse,
            teacher_lse,
            token_kl,
        )
        ctx.student_hidden_dtype = student_input_dtype
        ctx.teacher_hidden_dtype = teacher_input_dtype
        ctx.temperature = temperature
        ctx.compute_student_hidden_grad = ctx.needs_input_grad[0]
        ctx.compute_teacher_hidden_grad = ctx.needs_input_grad[1]
        ctx.compute_student_weight_grad = ctx.needs_input_grad[2]
        ctx.compute_teacher_weight_grad = ctx.needs_input_grad[3]
        return token_kl

    @staticmethod
    def backward(
        ctx,
        grad_token_kl: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
    ]:
        (
            student_hidden,
            teacher_hidden,
            student_weight,
            teacher_weight,
            student_lse,
            teacher_lse,
            token_kl,
        ) = ctx.saved_tensors
        grad_kl = grad_token_kl.contiguous().view(-1).float()
        (
            student_hidden_grad,
            student_weight_grad,
            teacher_hidden_grad,
            teacher_weight_grad,
        ) = _dlogits_backward(
            student_hidden,
            teacher_hidden,
            student_weight,
            teacher_weight,
            student_lse,
            teacher_lse,
            token_kl,
            grad_kl,
            ctx.temperature,
            _SPLIT_N_SIZE,
            ctx.compute_student_hidden_grad,
            ctx.compute_teacher_hidden_grad,
            ctx.compute_student_weight_grad,
            ctx.compute_teacher_weight_grad,
        )

        student_hidden_grad = (
            student_hidden_grad.to(ctx.student_hidden_dtype)
            if ctx.compute_student_hidden_grad else None
        )
        student_weight_grad = (
            student_weight_grad.to(student_weight.dtype)
            if ctx.compute_student_weight_grad else None
        )
        teacher_hidden_grad = (
            teacher_hidden_grad.to(ctx.teacher_hidden_dtype)
            if ctx.compute_teacher_hidden_grad else None
        )
        teacher_weight_grad = (
            teacher_weight_grad.to(teacher_weight.dtype)
            if ctx.compute_teacher_weight_grad else None
        )
        return (
            student_hidden_grad,
            teacher_hidden_grad,
            student_weight_grad,
            teacher_weight_grad,
            None,
            None,
        )


def linear_full_kl(
    student_hidden_states: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    student_output_weight: torch.Tensor,
    teacher_output_weight: torch.Tensor,
    loss_mask: torch.Tensor,
    temperature: float,
    tp_group: dist.ProcessGroup | None,
    kl_alpha_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute dense teacher-to-student KL without materializing full logits.

    Parameters
    ----------
    student_hidden_states : torch.Tensor
        Shape ``[S, B, Hs]`` with FP16 or BF16 elements.
    teacher_hidden_states : torch.Tensor
        Shape ``[B, S, Ht]``. Values are cast to the teacher weight dtype.
    student_output_weight : torch.Tensor
        TP-local shape ``[V, Hs]`` and the student hidden dtype.
    teacher_output_weight : torch.Tensor
        TP-local shape ``[V, Ht]`` with the student projection dtype.
    loss_mask : torch.Tensor
        Shape ``[B, S]``. Applied outside every TileLang kernel.
    temperature : float
        Finite positive softmax temperature. The KL is multiplied by
        ``temperature**2`` to preserve soft-target gradient scale.
    tp_group : torch.distributed.ProcessGroup or None
        Vocabulary-parallel group. ``None`` means the supplied vocabulary is
        complete and no collective is issued.
    kl_alpha_mask : torch.Tensor or None
        Optional shape ``[B]`` sample weights, also applied outside kernels.

    Returns
    -------
    torch.Tensor
        FP32 shape ``[2]`` containing loss sum and effective mask sum.

    Raises
    ------
    ValueError
        If temperature or mask shapes are invalid.
    TypeError
        If projection tensors use unsupported dtypes.
    AssertionError
        If projection layouts, devices, or hidden-size alignment are invalid.

    Notes
    -----
    TP ranks must use identical hidden states, masks, alpha weights, and
    upstream loss scaling. Weight gradients cover the local vocabulary shard.
    Hidden gradients are the local-vocabulary contribution and must be summed
    across `tp_group` by the caller, matching the reference function.
    Backward materializes at most 8192 local-vocabulary columns at once.
    """
    if (not isinstance(temperature, float) or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature must be a finite positive float, "
                         f"got {temperature!r}")
    _validate_projection_inputs(
        student_hidden_states,
        teacher_hidden_states,
        student_output_weight,
        teacher_output_weight,
        tp_group,
    )
    seq_len, batch, student_hidden_size = student_hidden_states.shape
    if loss_mask.shape != (batch, seq_len):
        raise ValueError(
            f"loss_mask must have shape {(batch, seq_len)}, "
            f"got {tuple(loss_mask.shape)}"
        )
    if kl_alpha_mask is not None and kl_alpha_mask.numel() != batch:
        raise ValueError(
            f"kl_alpha_mask must contain {batch} values, "
            f"got {kl_alpha_mask.numel()}"
        )
    if loss_mask.device != student_hidden_states.device:
        raise ValueError("loss_mask must be on the student device")
    if (kl_alpha_mask is not None and kl_alpha_mask.device != student_hidden_states.device):
        raise ValueError("kl_alpha_mask must be on the student device")

    # 在算子前压缩，前向和反向都只处理有效词元。
    if kl_alpha_mask is not None:
        kl_weight = kl_alpha_mask.view(-1, 1).float()
        count_mask = loss_mask * (kl_weight != 0)
        weighted_mask = loss_mask.float() * kl_weight
    else:
        count_mask = loss_mask
        weighted_mask = loss_mask.float()
    flat_count_mask = count_mask.reshape(-1).float()
    active_indices = torch.nonzero(flat_count_mask != 0, as_tuple=False).view(-1)

    if active_indices.numel() == 0:
        zero = weighted_mask.reshape(-1)[:0].sum()
        for tensor in (
            student_hidden_states,
            teacher_hidden_states,
            student_output_weight,
            teacher_output_weight,
        ):
            zero = zero + tensor.reshape(-1)[:0].float().sum()
        return torch.cat((zero.view(1), flat_count_mask.sum().view(1)))

    batch_indices = active_indices // seq_len
    sequence_indices = active_indices % seq_len
    student_indices = sequence_indices * batch + batch_indices
    student_hidden = student_hidden_states.reshape(seq_len * batch,
                                                   student_hidden_size).index_select(
                                                       0, student_indices
                                                   )
    teacher_hidden = teacher_hidden_states.reshape(
        batch * seq_len, teacher_hidden_states.shape[-1]
    ).index_select(0, active_indices)

    token_kl = _LinearFullKl.apply(
        student_hidden,
        teacher_hidden,
        student_output_weight,
        teacher_output_weight,
        temperature,
        tp_group,
    )
    active_weight = (weighted_mask.reshape(-1).index_select(0, active_indices) * (temperature**2))
    loss = torch.sum(token_kl * active_weight)
    return torch.cat((loss.view(1), flat_count_mask.sum().view(1)))
