from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import mpu, parallel_state, tensor_parallel

try:
    from megatron.core.extensions.transformer_engine import te_parallel_cross_entropy
except:
    te_parallel_cross_entropy = None
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

from gpatch_v4.core.mappings import (
    all_gather_from_context_parallel_region_no_zigzag,
    reduce_from_context_parallel_region,
)
from gpatch_v4.training_backend.loss.dpo_loss import compute_dpo_loss_core
from gpatch_v4.training_backend.loss.registry import register_loss
from gpatch_v4.training_backend.loss_factory import (
    _aggregate_packed_sft_ce,
    grad_cache_loss_replay_func,
)
from gpatch_v4.training_backend.vocab_parallel_entropy import vocab_parallel_entropy
from gpatch_v4.utils import (
    all_reduce_autograd,
    average_losses_across_data_parallel_group,
    from_parallel_logits_to_logprobs,
    masked_mean,
)
from gpatch_v4.utils.training_utils import from_parallel_logits_to_topk_logprobs

try:
    from gpatch_v4.kernel import (
        linear_cross_entropy,
        linear_full_kl,
        set_linear_ce_backend,
    )
except Exception:
    linear_cross_entropy = None
    linear_full_kl = None
    set_linear_ce_backend = None


@dataclass
class FinetuneLossInput:
    logits: torch.Tensor
    batch: Dict[str, torch.Tensor]
    unwrapped_model: Optional[torch.nn.Module] = None
    skip_cp_loss_reduce: bool = False
    """Dict from ``_postprocess`` when linear CE is enabled.
    Expected keys: ``hidden_states``, ``weight``, ``runtime_gather_output``.
    Used by ``linear_ce`` loss to compute fused linear+CE."""
    linear_ce_input: Optional[Dict[str, Any]] = None
    """Context-parallel process group for loss reduction.
    When not None, this group is used instead of ``mpu.get_context_parallel_group()``.
    Set by the dynamic-CP path to the per-microbatch dynamic CP subgroup."""
    cp_group: Optional[Any] = None
    cur_mbs_step: Optional[int] = None
    num_mbs: Optional[int] = None
    teacher_output_weight: Optional[torch.Tensor] = None
    calculate_per_token_loss: bool = False
    cu_seqlens_padded: Optional[torch.Tensor] = None
    local_cp_size: int = 1
    cp_partition_mode: str = "zigzag"


def _reduce_mcore_token_losses(
    loss: torch.Tensor,
    loss_mask: torch.Tensor,
    kl_alpha_mask: Optional[torch.Tensor] = None,
    return_src_loss: bool = False,
    skip_cp_reduce: bool = False,
    cp_group=None,
    loss_weights: Optional[torch.Tensor] = None,
):
    # loss from CE kernel is [S, B]; returned aggregate / src loss is [B, S].
    loss = loss.transpose(0, 1).contiguous()
    if return_src_loss:
        return loss

    if kl_alpha_mask is not None:
        ce_weight = (1 - kl_alpha_mask).view(-1, 1).float()
        loss = loss * ce_weight
        loss_mask = loss_mask * (ce_weight != 0)

    losses = loss.view(-1).float()
    loss_mask = loss_mask.view(-1).float()

    if loss_weights is not None:
        weighted_mask = loss_mask * loss_weights.view(-1).float()
    else:
        weighted_mask = loss_mask

    loss = torch.sum(losses * weighted_mask)
    total_tokens = loss_mask.sum()
    loss = torch.cat([loss.view(1), total_tokens.view(1)])

    # When calculate_per_token_loss=True the pipeline schedule + finalize_model_grads
    # handle CP gradient aggregation.  Skipping the AVG here keeps each CP rank's
    # local loss_sum/tokens so finalize_model_grads produces correctly scaled gradients.
    _cp_group = cp_group if cp_group is not None else mpu.get_context_parallel_group()
    if not skip_cp_reduce and dist.get_world_size(_cp_group) > 1:
        torch.distributed.all_reduce(loss, group=_cp_group, op=torch.distributed.ReduceOp.AVG)
    return loss


def ce_loss(
    config,
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    kl_alpha_mask: torch.Tensor = None,
    return_src_loss: bool = False,
    skip_cp_reduce: bool = False,
    cp_group=None,
    loss_weights: Optional[torch.Tensor] = None,
):
    labels = labels.transpose(0, 1).contiguous()
    logits = logits.transpose(0, 1).clone(memory_format=torch.contiguous_format)

    if config.training.cross_entropy_loss_fusion:
        if config.training.cross_entropy_fusion_impl == 'te':
            if te_parallel_cross_entropy is not None:
                labels = torch.as_strided(labels, labels.size(), (labels.size()[1], 1))
                loss = te_parallel_cross_entropy(
                    logits, labels, parallel_state.get_tensor_model_parallel_group()
                )
            else:
                raise RuntimeError("Trying to use a TE block when it's not present.")
        elif config.training.cross_entropy_fusion_impl == 'native':
            loss = fused_vocab_parallel_cross_entropy(
                logits, labels, parallel_state.get_tensor_model_parallel_group()
            )
    else:
        loss = tensor_parallel.vocab_parallel_cross_entropy(logits, labels)

    return _reduce_mcore_token_losses(
        loss,
        loss_mask,
        kl_alpha_mask=kl_alpha_mask,
        return_src_loss=return_src_loss,
        skip_cp_reduce=skip_cp_reduce,
        cp_group=cp_group,
        loss_weights=loss_weights,
    )


@register_loss(backends=("mcore", ), loss_name="cross_entropy", log_registration=True)
def cross_entropy_loss_func(
    config,
    loss_input: FinetuneLossInput,
):
    """LM cross-entropy loss.

    Args:
        labels (Tensor): ``[B, S]``.
        logits (Tensor): ``[B, S, V]`` from the output layer.
        loss_mask (Tensor): ``[B, S]``.
    Returns:
        Tensor: ``[B, S]``.
    """
    batch = loss_input.batch
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    loss_weights = batch.get("loss_weights", None)
    packed_sequence_mean = (
        loss_input.cu_seqlens_padded is not None and not loss_input.calculate_per_token_loss
    )

    if loss_input.linear_ce_input is not None:
        assert config.training.return_hidden_states_for_ce
        hidden_ce_loss = linear_ce_loss if config.training.use_linear_ce else compact_ce_loss
        loss = hidden_ce_loss(
            config,
            loss_input.linear_ce_input,
            labels,
            loss_mask,
            return_src_loss=packed_sequence_mean,
            skip_cp_reduce=loss_input.skip_cp_loss_reduce,
            cp_group=loss_input.cp_group,
            loss_weights=loss_weights,
        )
    else:
        logits = loss_input.logits.float()
        loss = ce_loss(
            config,
            logits,
            labels,
            loss_mask,
            return_src_loss=packed_sequence_mean,
            skip_cp_reduce=loss_input.skip_cp_loss_reduce,
            cp_group=loss_input.cp_group,
            loss_weights=loss_weights,
        )
    #TODO: check loss nan or not

    if packed_sequence_mean:
        bwd_sum, sample_count, token_sum, token_count = _aggregate_packed_sft_ce(
            loss,
            loss_mask,
            loss_weights,
            loss_input,
        )
        metrics = {"lm_loss": torch.stack([token_sum, token_count])}
        return (bwd_sum, sample_count.detach().to(torch.int), metrics)

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    # metrics: [sum, count] 对，由消费端统一做 DP all-reduce 再除
    metrics = {"lm_loss": loss.clone().detach()}

    # dump metrics
    if batch.get("should_dump_metrics", False):
        assert not config.training.return_hidden_states_for_ce, "暂不支持，因为没有 logits"
        metrics.update({"dump/mask": batch["full_loss_mask"].detach().bool().to(device="cpu")})

        if config.training.dump_metrics_logprobs_topk > 0:
            topk_logprobs, topk_token_ids = from_parallel_logits_to_topk_logprobs(
                vocab_parallel_logits=logits,
                topk=config.training.dump_metrics_logprobs_topk,
                temperature=1.0,
            )
            topk_logprobs = topk_logprobs.to(dtype=torch.bfloat16, device="cpu")
            topk_token_ids = topk_token_ids.to(dtype=torch.int32, device="cpu")
            metrics.update(
                {
                    "dump/topk_logprobs": topk_logprobs,
                    "dump/topk_token_ids": topk_token_ids,
                }
            )

        if config.training.ppo_dump_per_token_entropy:
            _, per_token_entropy_unmasked = vocab_parallel_entropy(logits.detach().clone())
            metrics.update(
                {
                    "dump/per_token_entropy":
                        per_token_entropy_unmasked.to(dtype=torch.bfloat16, device="cpu")
                }
            )

    return (loss[0].clone(), local_num_tokens, metrics)


def compact_ce_loss(
    config: Any,
    linear_ce_input: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    kl_alpha_mask: torch.Tensor = None,
    return_src_loss: bool = False,
    skip_cp_reduce: bool = False,
    cp_group: Optional[Any] = None,
    loss_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert config.training.ce_compaction
    assert not config.training.use_linear_ce
    labels = labels.transpose(0, 1).contiguous()

    hidden_states = linear_ce_input["hidden_states"]
    output_layer = linear_ce_input["output_layer"]
    tp_group = output_layer.tp_group
    if output_layer.sequence_parallel:
        hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
            hidden_states,
            tensor_parallel_output_grad=False,
        )
    weight = linear_ce_input["weight"]
    if weight is None:
        weight = output_layer.weight
    assert output_layer.bias is None, ("CE compaction requires an output projection without bias")
    assert not output_layer.gather_output
    assert not linear_ce_input["runtime_gather_output"]

    loss_mask_sb = loss_mask.transpose(0, 1).contiguous()
    assert hidden_states.shape[:-1] == labels.shape == loss_mask_sb.shape, (
        "compact CE shape mismatch: "
        f"{hidden_states.shape=}, {labels.shape=}, {loss_mask_sb.shape=}"
    )
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_labels = labels.reshape(-1)
    keep_indices = (loss_mask_sb.reshape(-1) != 0).nonzero(as_tuple=False).flatten()
    num_valid_tokens = keep_indices.numel()
    if num_valid_tokens == 0:
        zero_loss = (flat_hidden[:0].sum() + weight[:0].sum()).float()
        loss = zero_loss.expand(labels.numel()).view_as(labels)
    else:
        compact_hidden = flat_hidden.index_select(0, keep_indices).unsqueeze(1)
        compact_labels = flat_labels.index_select(0, keep_indices)
        assert compact_hidden.shape[-1] == weight.shape[-1]
        if tp_group is not None and dist.get_world_size(tp_group) > 1:
            # Enter TP region so vocab-parallel CE all-reduces d_hidden.
            compact_hidden = tensor_parallel.copy_to_tensor_model_parallel_region(
                compact_hidden,
                group=tp_group,
            )
        compact_logits = F.linear(compact_hidden, weight).float()
        compact_labels = compact_labels.unsqueeze(1)
        if config.training.cross_entropy_loss_fusion:
            if config.training.cross_entropy_fusion_impl == 'te':
                if te_parallel_cross_entropy is not None:
                    compact_labels = torch.as_strided(
                        compact_labels,
                        compact_labels.size(),
                        (compact_labels.size()[1], 1),
                    )
                    compact_loss = te_parallel_cross_entropy(
                        compact_logits,
                        compact_labels,
                        parallel_state.get_tensor_model_parallel_group(),
                    )
                else:
                    raise RuntimeError("Trying to use a TE block when it's not present.")
            elif config.training.cross_entropy_fusion_impl == 'native':
                compact_loss = fused_vocab_parallel_cross_entropy(
                    compact_logits,
                    compact_labels,
                    parallel_state.get_tensor_model_parallel_group(),
                )
        else:
            compact_loss = tensor_parallel.vocab_parallel_cross_entropy(
                compact_logits,
                compact_labels,
            )
        loss = compact_loss.new_zeros(labels.numel())
        loss = loss.scatter(0, keep_indices, compact_loss.reshape(-1))
        loss = loss.view_as(labels)

    return _reduce_mcore_token_losses(
        loss,
        loss_mask,
        kl_alpha_mask=kl_alpha_mask,
        return_src_loss=return_src_loss,
        skip_cp_reduce=skip_cp_reduce,
        cp_group=cp_group,
        loss_weights=loss_weights,
    )


def linear_ce_loss(
    config,
    linear_ce_input: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    kl_alpha_mask: torch.Tensor = None,
    return_src_loss: bool = False,
    skip_cp_reduce: bool = False,
    cp_group=None,
    loss_weights: Optional[torch.Tensor] = None,
    pre_gathered_hidden_states: Optional[torch.Tensor] = None,
):
    """Compute linear (fused) cross-entropy loss from hidden states.

    Returns:
        If ``return_src_loss``: per-token loss ``[b, s]``.
        Otherwise: ``torch.Tensor`` ``[loss_sum, total_tokens]``.
    """
    set_linear_ce_backend(config.training.linear_ce_backend)
    # [b s] => [s b]
    labels = labels.transpose(0, 1).contiguous()

    output_layer = linear_ce_input['output_layer']
    tp_group = output_layer.tp_group
    if pre_gathered_hidden_states is not None:
        assert output_layer.sequence_parallel
        hidden_states = pre_gathered_hidden_states
    elif output_layer.sequence_parallel:
        hidden_states = linear_ce_input["hidden_states"]
        hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
            hidden_states,
            tensor_parallel_output_grad=True,
        )
    else:
        hidden_states = linear_ce_input["hidden_states"]
        assert tp_group is None or dist.get_world_size(tp_group) <= 1, (
            "linear_cross_entropy backward does not all-reduce d_hidden across TP ranks. "
            "With TP > 1 and sequence_parallel=False there is no reduce-scatter to compensate, "
            "so d_hidden would be incorrect. Enable sequence_parallel or use vocab_parallel_cross_entropy."
        )
    weight = linear_ce_input["weight"]
    if weight is None:
        weight = output_layer.weight

    if not config.training.ce_compaction:
        loss = linear_cross_entropy(
            hidden_states,
            weight,
            labels,
            1.0,
            "none",
            tp_group,
        )
    else:
        loss_mask_sb = loss_mask.transpose(0, 1).contiguous()
        assert hidden_states.shape[:-1] == labels.shape == loss_mask_sb.shape, (
            "linear CE shape mismatch: "
            f"{hidden_states.shape=}, {labels.shape=}, {loss_mask_sb.shape=}"
        )

        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_labels = labels.reshape(-1)
        keep_indices = (loss_mask_sb.reshape(-1) != 0).nonzero(as_tuple=False).flatten()
        num_valid_tokens = keep_indices.numel()
        if num_valid_tokens == 0:
            # Preserve zero gradients for hidden states and output weights.
            zero_loss = flat_hidden[:0].sum() + weight[:0].sum()
            loss = zero_loss.expand(labels.numel()).view_as(labels)
        else:
            linear_ce_hidden = flat_hidden.index_select(0, keep_indices)
            linear_ce_labels = flat_labels.index_select(0, keep_indices)
            # Triton specializes one token to a scalar, so pad it to two rows.
            if num_valid_tokens == 1:
                linear_ce_hidden = torch.cat((linear_ce_hidden, linear_ce_hidden), dim=0)
                linear_ce_labels = torch.cat((linear_ce_labels, linear_ce_labels))
            loss = linear_cross_entropy(
                linear_ce_hidden,
                weight,
                linear_ce_labels,
                1.0,
                "none",
                tp_group,
            )
            dense_loss = loss.new_zeros(labels.numel())
            loss = dense_loss.scatter(0, keep_indices, loss[:num_valid_tokens])
            loss = loss.view_as(labels)

    return _reduce_mcore_token_losses(
        loss,
        loss_mask,
        kl_alpha_mask=kl_alpha_mask,
        return_src_loss=return_src_loss,
        skip_cp_reduce=skip_cp_reduce,
        cp_group=cp_group,
        loss_weights=loss_weights,
    )


def logits_kl_loss(
    teacher_logits, student_logits, loss_mask, temperature, kl_alpha_mask: torch.Tensor = None
):
    torch.cuda.synchronize()
    teacher_logits = teacher_logits.float()
    student_logits = student_logits.float()

    if mpu.get_tensor_model_parallel_world_size() > 1:
        teacher_logits = teacher_logits / temperature
        student_logits = student_logits / temperature
        teacher_logits_max, _ = torch.max(teacher_logits, dim=-1)
        torch.distributed.all_reduce(
            teacher_logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=mpu.get_tensor_model_parallel_group(),
        )
        teacher_logits = teacher_logits - teacher_logits_max.unsqueeze(dim=-1)

        denom_teacher = torch.sum(torch.exp(teacher_logits), dim=-1)
        # We can't use standard reduction function here since the computation
        # that follows it isn't identical across TP ranks.
        denom_teacher = all_reduce_autograd(
            denom_teacher, group=mpu.get_tensor_model_parallel_group()
        )
        # Maximum value along vocab dimension across all GPUs.
        student_logits_max, _ = torch.max(student_logits, dim=-1)
        torch.distributed.all_reduce(
            student_logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=mpu.get_tensor_model_parallel_group(),
        )
        student_logits = student_logits - student_logits_max.unsqueeze(dim=-1).detach()

        denom_student = torch.sum(torch.exp(student_logits), dim=-1)
        denom_student = all_reduce_autograd(
            denom_student, group=mpu.get_tensor_model_parallel_group()
        )

        bsz, slen, sharded_vocab_size = student_logits.shape
        student_log_prob = student_logits - torch.log(denom_student).view(bsz, slen, 1).expand(
            bsz, slen, sharded_vocab_size
        )
        teacher_log_prob = teacher_logits - torch.log(denom_teacher).view(bsz, slen, 1).expand(
            bsz, slen, sharded_vocab_size
        )
        kl_loss = torch.sum(
            F.kl_div(student_log_prob, teacher_log_prob, reduction="none", log_target=True),
            dim=-1,
        ).float()
        if kl_alpha_mask is not None:
            kl_weight = kl_alpha_mask.view(-1, 1).float()
            kl_loss = kl_loss * kl_weight
            loss_mask = loss_mask * (kl_weight != 0)

        kl_loss = kl_loss.view(-1)
        local_mask = loss_mask.view(-1).float()
        # T² 抵消软目标的高温梯度衰减。
        kl_loss = torch.sum(kl_loss * local_mask * (temperature**2))
        torch.distributed.all_reduce(kl_loss, group=mpu.get_tensor_model_parallel_group())
    else:
        kl_loss = torch.sum(
            F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="none",
            ),
            dim=-1,
        ).float()
        if kl_alpha_mask is not None:
            kl_weight = kl_alpha_mask.view(-1, 1).float()
            kl_loss = kl_loss * kl_weight
            loss_mask = loss_mask * (kl_weight != 0)

        kl_loss = kl_loss.view(-1)
        local_mask = loss_mask.view(-1).float()
        # T² 抵消软目标的高温梯度衰减。
        kl_loss = torch.sum(kl_loss * local_mask * (temperature**2))

    reduce_kl_tensor = torch.cat([kl_loss.view(1), local_mask.sum().view(1)])
    if dist.get_world_size(mpu.get_context_parallel_group()) > 1:
        torch.distributed.all_reduce(
            reduce_kl_tensor,
            group=mpu.get_context_parallel_group(),
            op=torch.distributed.ReduceOp.AVG,
        )
    return reduce_kl_tensor


def _offpd_megatron_loss_outputs(
    loss: torch.Tensor,
    num_tokens: torch.Tensor,
    metrics: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    # Return a token sum so the schedule does not multiply a CP-reduced mean by cp_size.
    num_tokens = num_tokens.detach().to(dtype=torch.int)
    token_scale = num_tokens.clamp(min=1).to(dtype=loss.dtype)
    return (loss * token_scale, num_tokens, metrics)


@register_loss(backends=("mcore", ), loss_name="ce_with_kl_eager")
def ce_with_kl_eager(
    config,
    loss_input: FinetuneLossInput,
):
    # [b s] => [s b]
    kl_loss_alpha = config.distill.offpd_loss_alpha
    temperature = config.distill.offpd_temperature

    batch = loss_input.batch
    parallel_logits = loss_input.logits

    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    loss_weights = batch.get("loss_weights", None)
    use_data_alpha = (
        config.training.enable_teacher_kl_loss and config.distill.enable_data_with_alpha
    )
    kl_alpha_mask = batch["kl_alpha_mask"] if use_data_alpha else None
    if kl_alpha_mask is not None:
        assert kl_alpha_mask.shape[0] == labels.shape[
            0], f"shape mismatch {kl_alpha_mask.shape} {labels.shape}"

    ce_kl_alpha_mask = (
        kl_alpha_mask if
        (config.training.enable_teacher_kl_loss and config.distill.enable_data_with_alpha) else None
    )
    if config.training.use_linear_ce:
        loss = linear_ce_loss(
            config,
            loss_input.linear_ce_input,
            labels,
            loss_mask,
            kl_alpha_mask=ce_kl_alpha_mask,
            skip_cp_reduce=loss_input.skip_cp_loss_reduce,
            cp_group=loss_input.cp_group,
            loss_weights=loss_weights,
        )

        linear_ce_input = loss_input.linear_ce_input
        student_hidden_states = linear_ce_input["hidden_states"]
        student_output_layer = linear_ce_input["output_layer"]
        assert not student_output_layer.config.use_mup, (
            "Teacher hidden-state transfer does not yet support MuP logit scaling"
        )
        student_weight = linear_ce_input["weight"]
        if student_weight is None:
            student_weight = student_output_layer.weight
        student_output = student_output_layer(
            student_hidden_states,
            weight=student_weight,
            runtime_gather_output=linear_ce_input["runtime_gather_output"],
        )
        if isinstance(student_output, tuple):
            student_output = student_output[0]
        parallel_logits = student_output.transpose(0, 1).contiguous()
    else:
        loss = ce_loss(
            config,
            parallel_logits,
            labels,
            loss_mask,
            kl_alpha_mask=ce_kl_alpha_mask,
            loss_weights=loss_weights,
        )
    ce_numel = loss[1].detach()  # token count for metrics
    if loss[1] != 0:
        lm_loss = loss[0] / loss[1]
    else:
        lm_loss = loss[0]

    if config.training.enable_teacher_kl_loss:
        teacher_hidden_states = batch["teacher_hidden_states"]
        teacher_output_weight = loss_input.teacher_output_weight
        assert teacher_output_weight is not None
        assert teacher_hidden_states.shape[-1] == teacher_output_weight.shape[-1], (
            f"Teacher hidden/head mismatch: {teacher_hidden_states.shape} vs "
            f"{teacher_output_weight.shape}"
        )
        with torch.no_grad():
            teacher_logits = F.linear(
                teacher_hidden_states.to(dtype=teacher_output_weight.dtype),
                teacher_output_weight,
            )
        assert teacher_logits.shape == parallel_logits.shape, (
            f"Teacher/Student logits mismatch: {teacher_logits.shape} vs "
            f"{parallel_logits.shape}"
        )
        # logits_kl_loss 函数中，TP > 1 时需要手动实现 all-reduce 来跨 TP ranks
        # 计算全局 softmax 和 KL 散度（因为每个 rank 只持有 vocab 的一部分 logits）。
        reduce_kl_loss = logits_kl_loss(
            teacher_logits,  # Teacher [B, S, V//TP]
            parallel_logits,  # Student [B, S, V//TP]
            loss_mask,
            temperature,
            kl_alpha_mask=kl_alpha_mask if config.distill.enable_data_with_alpha else None
        )
        kl_numel = reduce_kl_loss[1].detach()
        if reduce_kl_loss[1] != 0:
            kl_loss = reduce_kl_loss[0] / reduce_kl_loss[1]
        else:
            kl_loss = reduce_kl_loss[0]

        if config.distill.enable_data_with_alpha:
            loss = kl_loss + lm_loss
        else:
            loss = kl_loss_alpha * kl_loss + (1 - kl_loss_alpha) * lm_loss

        # 各 metric 必须使用各自分支的 token 计数加权：
        #   - lm_loss 仅在 (1-alpha)>0 的 token 上有贡献 -> ce_numel
        #   - kl_loss 仅在 alpha>0 的 token 上有贡献   -> kl_numel
        # 否则 alpha 接近 0 或 1 时 DP 聚合 sum/count 会得到 0（loss 仍生效）。
        # 总 loss 用原始 loss_mask 的 token 数加权，使其与 alpha 取值
        # （包括 (0,1) 的中间值）无关，避免重复计数。
        total_numel = loss_mask.sum().detach()
        if dist.get_world_size(mpu.get_context_parallel_group()) > 1:
            torch.distributed.all_reduce(total_numel, group=mpu.get_context_parallel_group())
        with torch.no_grad():
            metrics = {
                "loss": torch.stack([loss.detach() * total_numel, total_numel]),
                "lm_loss": torch.stack([lm_loss.detach() * ce_numel, ce_numel]),
                "kl_loss": torch.stack([kl_loss.detach() * kl_numel, kl_numel]),
            }
        return _offpd_megatron_loss_outputs(loss, total_numel, metrics)
    else:
        loss = lm_loss
        with torch.no_grad():
            metrics = {
                "loss": torch.stack([loss.detach() * ce_numel, ce_numel]),
                "lm_loss": torch.stack([lm_loss.detach() * ce_numel, ce_numel]),
            }
        return _offpd_megatron_loss_outputs(loss, ce_numel, metrics)


@register_loss(backends=("mcore", ), loss_name="ce_with_kl")
def ce_with_kl(
    config,
    loss_input: FinetuneLossInput,
):
    if not config.training.use_linear_ce:
        return ce_with_kl_eager(config, loss_input)

    kl_loss_alpha = config.distill.offpd_loss_alpha
    temperature = float(config.distill.offpd_temperature)
    batch = loss_input.batch
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    loss_weights = batch.get("loss_weights", None)
    use_data_alpha = (
        config.training.enable_teacher_kl_loss and config.distill.enable_data_with_alpha
    )
    kl_alpha_mask = batch["kl_alpha_mask"] if use_data_alpha else None
    if kl_alpha_mask is not None:
        assert kl_alpha_mask.shape[0] == labels.shape[0], (
            f"shape mismatch {kl_alpha_mask.shape} {labels.shape}"
        )

    linear_ce_input = loss_input.linear_ce_input
    student_hidden_states = linear_ce_input["hidden_states"]
    student_output_layer = linear_ce_input["output_layer"]
    pre_gathered_hidden_states = None
    if student_output_layer.sequence_parallel:
        student_hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
            student_hidden_states,
            tensor_parallel_output_grad=True,
        )
        pre_gathered_hidden_states = student_hidden_states
    loss = linear_ce_loss(
        config,
        linear_ce_input,
        labels,
        loss_mask,
        kl_alpha_mask=kl_alpha_mask,
        skip_cp_reduce=loss_input.skip_cp_loss_reduce,
        cp_group=loss_input.cp_group,
        loss_weights=loss_weights,
        pre_gathered_hidden_states=pre_gathered_hidden_states,
    )
    ce_numel = loss[1].detach()
    lm_loss = loss[0] / loss[1] if loss[1] != 0 else loss[0]

    if not config.training.enable_teacher_kl_loss:
        with torch.no_grad():
            metrics = {
                "loss": torch.stack([lm_loss.detach() * ce_numel, ce_numel]),
                "lm_loss": torch.stack([lm_loss.detach() * ce_numel, ce_numel]),
            }
        return _offpd_megatron_loss_outputs(lm_loss, ce_numel, metrics)

    assert not student_output_layer.config.use_mup, (
        "Teacher hidden-state transfer does not yet support MuP logit scaling"
    )
    student_weight = linear_ce_input["weight"]
    if student_weight is None:
        student_weight = student_output_layer.weight
    tp_group = student_output_layer.tp_group

    teacher_hidden_states = batch["teacher_hidden_states"]
    teacher_output_weight = loss_input.teacher_output_weight
    assert teacher_output_weight is not None
    assert teacher_hidden_states.shape[-1] == teacher_output_weight.shape[-1], (
        f"Teacher hidden/head mismatch: {teacher_hidden_states.shape} vs "
        f"{teacher_output_weight.shape}"
    )
    reduce_kl_loss = linear_full_kl(
        student_hidden_states,
        teacher_hidden_states.detach(),
        student_weight,
        teacher_output_weight.detach(),
        loss_mask,
        temperature,
        tp_group,
        kl_alpha_mask=kl_alpha_mask,
    )
    cp_group = (
        loss_input.cp_group if loss_input.cp_group is not None else mpu.get_context_parallel_group()
    )
    if (not loss_input.skip_cp_loss_reduce and dist.get_world_size(cp_group) > 1):
        torch.distributed.all_reduce(
            reduce_kl_loss,
            group=cp_group,
            op=torch.distributed.ReduceOp.AVG,
        )

    kl_numel = reduce_kl_loss[1].detach()
    kl_loss = (
        reduce_kl_loss[0] / reduce_kl_loss[1] if reduce_kl_loss[1] != 0 else reduce_kl_loss[0]
    )
    if config.distill.enable_data_with_alpha:
        loss = kl_loss + lm_loss
    else:
        loss = kl_loss_alpha * kl_loss + (1 - kl_loss_alpha) * lm_loss

    total_numel = loss_mask.sum().detach()
    if (not loss_input.skip_cp_loss_reduce and dist.get_world_size(cp_group) > 1):
        torch.distributed.all_reduce(
            total_numel,
            group=cp_group,
        )
    with torch.no_grad():
        metrics = {
            "loss": torch.stack([loss.detach() * total_numel, total_numel]),
            "lm_loss": torch.stack([lm_loss.detach() * ce_numel, ce_numel]),
            "kl_loss": torch.stack([kl_loss.detach() * kl_numel, kl_numel]),
        }
    return _offpd_megatron_loss_outputs(loss, total_numel, metrics)


def _dpo_loss_from_sequence_logprobs(
    config,
    policy_seq_logps: torch.Tensor,
    ref_seq_logps: torch.Tensor,
    chosen_token_counts: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Apply pairwise DPO and optional FTX to aligned sequence log-probs."""
    assert policy_seq_logps.numel() % 2 == 0, "DPO requires chosen/rejected pairs"
    rbs = policy_seq_logps.numel() // 2
    policy_chosen_logps, policy_rejected_logps = policy_seq_logps.split(rbs)
    ref_chosen_logps, ref_rejected_logps = ref_seq_logps.split(rbs)
    losses, chosen_rewards, rejected_rewards = compute_dpo_loss_core(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=config.training.dpo_beta,
        label_smoothing=config.training.dpo_label_smoothing,
        loss_type=config.training.dpo_loss_type,
    )
    if config.training.dpo_ftx_gamma > 1e-6:
        assert chosen_token_counts is not None
        losses -= (
            config.training.dpo_ftx_gamma * policy_chosen_logps /
            chosen_token_counts.clamp_min(1.0)
        )
    return (
        losses,
        chosen_rewards,
        rejected_rewards,
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )


def _contiguous_thd_pair_logits_mean(
    logits: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute BSHD-equivalent chosen/rejected logit means for contiguous THD."""
    assert cu_seqlens_padded.numel() % 2 == 1, "DPO requires chosen/rejected pairs"
    cp_group = mpu.get_context_parallel_group()
    local_rows = logits.shape[1]
    global_start = cp_group.rank() * local_rows
    global_rows = torch.arange(
        global_start,
        global_start + local_rows,
        device=logits.device,
        dtype=cu_seqlens_padded.dtype,
    )
    sequence_ids = torch.bucketize(global_rows, cu_seqlens_padded[1:], right=True)
    chosen_rows = sequence_ids < (cu_seqlens_padded.numel() - 1) // 2

    # Match BSHD's ``logits.mean()`` semantics while avoiding a per-sequence
    # slice/reduction and host synchronization for every packed boundary.
    row_sums = logits.detach().sum(dim=(0, 2))
    elements_per_row = logits.shape[0] * logits.shape[2]
    stats = torch.stack(
        (
            row_sums[chosen_rows].sum(),
            chosen_rows.sum().to(logits.dtype) * elements_per_row,
            row_sums[~chosen_rows].sum(),
            (~chosen_rows).sum().to(logits.dtype) * elements_per_row,
        )
    )

    dist.all_reduce(stats, group=cp_group, op=dist.ReduceOp.SUM)
    return stats[0] / stats[1].clamp_min(1), stats[2] / stats[3].clamp_min(1)


def contiguous_thd_dpo_loss(
    config,
    loss_input: FinetuneLossInput,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compute DPO over DSv4 contiguous-CP THD shards and return pair-level values.

    Log-probs are gathered back into physical packed-buffer order before
    sequence aggregation. ``cu_seqlens_q_padded`` is required because its
    boundaries include the inter-sequence CP alignment padding.
    """
    logits = loss_input.logits.float()
    batch = loss_input.batch
    assert "ref_logprobs" in batch, "THD DPO requires packed reference log-probs"
    packed_seq_params = batch["full_packed_seq_params"]
    assert packed_seq_params.cp_partition_mode == "contiguous", (
        "THD DPO currently supports only DSv4 contiguous CP partitioning"
    )
    local_logps = from_parallel_logits_to_logprobs(
        vocab_parallel_logits=logits,
        target=batch["labels"],
        inference_only=False,
        ignore_cp=True,
        pre_shifted=True,
    )
    local_mask = batch["loss_mask"].to(dtype=local_logps.dtype)
    local_ref_logps = batch["ref_logprobs"].to(dtype=local_logps.dtype)
    assert local_logps.shape == local_mask.shape == local_ref_logps.shape

    policy_logps = all_gather_from_context_parallel_region_no_zigzag(local_logps)
    loss_mask = all_gather_from_context_parallel_region_no_zigzag(local_mask)
    ref_logps = all_gather_from_context_parallel_region_no_zigzag(local_ref_logps)
    cu_seqlens = packed_seq_params.cu_seqlens_q_padded
    policy_chosen_logits_mean, policy_rejected_logits_mean = _contiguous_thd_pair_logits_mean(
        logits, cu_seqlens
    )
    policy_seq_logps = []
    ref_seq_logps = []
    sequence_token_counts = []
    for seq_start, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        seq_slice = slice(int(seq_start.item()), int(seq_end.item()))
        mask = loss_mask[0, seq_slice]
        policy_seq_logps.append((policy_logps[0, seq_slice] * mask).sum())
        ref_seq_logps.append((ref_logps[0, seq_slice] * mask).sum())
        sequence_token_counts.append(mask.sum())

    policy_seq_logps = torch.stack(policy_seq_logps)
    ref_seq_logps = torch.stack(ref_seq_logps)
    sequence_token_counts = torch.stack(sequence_token_counts)
    chosen_token_counts = sequence_token_counts[:sequence_token_counts.numel() // 2]
    (
        losses,
        chosen_rewards,
        rejected_rewards,
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    ) = _dpo_loss_from_sequence_logprobs(
        config,
        policy_seq_logps,
        ref_seq_logps,
        chosen_token_counts,
    )

    return (
        losses,
        chosen_rewards,
        rejected_rewards,
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        policy_chosen_logits_mean,
        policy_rejected_logits_mean,
    )


def bshd_dpo_loss(
    config,
    loss_input: FinetuneLossInput,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compute DPO for standard BSHD, including Megatron zigzag CP."""
    logits = loss_input.logits.float()
    batch = loss_input.batch
    labels = batch["labels"]
    loss_mask = batch["full_loss_mask"][:, :-1]
    target = batch["full_tokens"]

    assert logits.shape[0] % 2 == 0, "mbs must be 2*n"
    rbs = logits.shape[0] // 2
    policy_chosen_logits, policy_rejected_logits = logits.split(rbs, dim=0)
    logps = from_parallel_logits_to_logprobs(
        vocab_parallel_logits=logits, target=target, inference_only=False
    )
    logps = (logps * loss_mask).sum(-1)
    ref_logps = (batch["ref_logprobs"] * loss_mask).sum(-1)
    chosen_token_counts = None
    if config.training.dpo_ftx_gamma > 1e-6:
        chosen_labels = labels.split(rbs, dim=0)[0]
        chosen_token_counts = (chosen_labels != -100).sum(-1)
        if config.policy.dist_config.context_parallel_size > 1:
            chosen_token_counts = reduce_from_context_parallel_region(chosen_token_counts)
    (
        losses,
        chosen_rewards,
        rejected_rewards,
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    ) = _dpo_loss_from_sequence_logprobs(
        config,
        logps,
        ref_logps,
        chosen_token_counts,
    )

    return (
        losses,
        chosen_rewards,
        rejected_rewards,
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        policy_chosen_logits.detach().mean().float(),
        policy_rejected_logits.detach().mean().float(),
    )


@register_loss(backends=("mcore", ), loss_name="dpo")
def dpo_loss_func(
    config,
    loss_input: FinetuneLossInput,
):
    """Megatron-backend DPO loss (uses ``from_parallel_logits_to_logprobs``)."""
    # TODO: support ``logprobs_from_linear_ce`` so DPO can avoid materializing
    # vocab logits when ``training.use_linear_ce`` is enabled.
    batch = loss_input.batch
    is_thd = batch.get("full_packed_seq_params") is not None

    # Standard BSHD, including Megatron zigzag CP, uses the non-THD helper.
    # Only DSv4 contiguous THD CP is supported here. Generic THD + CP uses
    # different partition/reorder semantics and needs its own aggregation path.
    if is_thd:
        (
            losses,
            chosen_rewards,
            rejected_rewards,
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            policy_chosen_logits_mean,
            policy_rejected_logits_mean,
        ) = contiguous_thd_dpo_loss(config, loss_input)
    else:
        (
            losses,
            chosen_rewards,
            rejected_rewards,
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            policy_chosen_logits_mean,
            policy_rejected_logits_mean,
        ) = bshd_dpo_loss(config, loss_input)

    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    reported_loss = losses.mean()
    losses = reported_loss
    metrics = {
        "dpo-metrics/rewards-accuracies": reward_accuracies.mean().float(),
        "dpo-metrics/rewards-chosen": chosen_rewards.mean().float(),
        "dpo-metrics/rewards-rejected": rejected_rewards.mean().float(),
        "dpo-metrics/rewards-margins": (chosen_rewards - rejected_rewards).mean().float(),
        "dpo-metrics/logps-rejected": policy_rejected_logps.detach().mean().float(),
        "dpo-metrics/logps-chosen": policy_chosen_logps.detach().mean().float(),
        "dpo-metrics/ref-logps-rejected": ref_rejected_logps.detach().mean().float(),
        "dpo-metrics/ref-logps-chosen": ref_chosen_logps.detach().mean().float(),
        "dpo-metrics/logits-rejected": policy_rejected_logits_mean.detach().mean().float(),
        "dpo-metrics/logits-chosen": policy_chosen_logits_mean.detach().mean().float(),
        "dpo-metrics/loss": reported_loss.clone().detach().float(),
    }
    keys = sorted(list(metrics.keys()))
    avg_metrics = average_losses_across_data_parallel_group([metrics[k] for k in keys])
    metrics = {k: v for k, v in zip(keys, avg_metrics)}

    return (losses, metrics)


@register_loss(backends=("mcore", ), loss_name="square_averaging_cross_entropy")
def square_averaging_cross_entropy_loss_func(
    config,
    loss_input: FinetuneLossInput,
):
    """LM cross-entropy loss with square-averaging weighting.

    Each sample is weighted by ``1 / sqrt(num_answer_tokens)`` so that
    short-answer samples are not overwhelmed by long-answer ones.

    When ``calculate_per_token_loss=True`` (required by dynamic CP), this
    function returns 3 elements ``(loss_sum, effective_num_tokens, metrics)``
    so that ``forward_step_calc_loss`` takes the per-token-loss branch and
    avoids the ``output_tensor *= cp_group_size`` multiplication that would
    otherwise double the gradient under CP=2.

    To preserve the exact square-averaging normalisation while keeping
    ``num_tokens`` as an integer (required by
    ``torch.distributed.broadcast``), both ``loss_sum`` and
    ``effective_num_tokens`` are scaled by ``_SA_SCALE = 10_000``:

    - ``loss_sum = weighted_sum * _SA_SCALE``
    - ``effective_num_tokens = int(W / cp_size * _SA_SCALE)``

    where ``W = square_averaging_weights.sum()``.  After
    ``finalize_model_grads`` divides by ``total_num_tokens``, the
    ``_SA_SCALE`` factor cancels and the gradient equals the correct
    square-averaging gradient ``Σ(weighted_sum) / Σ(W)``.

    Args:
        labels (Tensor): ``[B, S]``.
        logits (Tensor): ``[B, S, V]`` from the output layer.
        loss_mask (Tensor): ``[B, S]``.
    Returns:
        When ``calculate_per_token_loss=True``:
            ``(loss_sum, effective_num_tokens, metrics)`` – 3-element tuple.
        Otherwise:
            ``(loss, metrics)`` – 2-element tuple (legacy behaviour).
    """
    batch = loss_input.batch
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    # shape = [b,] 都是一样的
    square_averaging_weights = batch["square_averaging_weights"]
    assert square_averaging_weights is not None
    assert square_averaging_weights.requires_grad is False

    if loss_input.linear_ce_input is not None:
        assert config.training.return_hidden_states_for_ce
        hidden_ce_loss = linear_ce_loss if config.training.use_linear_ce else compact_ce_loss
        losses = hidden_ce_loss(
            config,
            loss_input.linear_ce_input,
            labels,
            loss_mask,
            return_src_loss=True,
        )
    else:
        logits = loss_input.logits.float()
        losses = ce_loss(config, logits, labels, loss_mask, return_src_loss=True)
    loss_weight = loss_mask.sum(dim=-1).float()
    _cp_group = loss_input.cp_group if loss_input.cp_group is not None else mpu.get_context_parallel_group(
    )
    if dist.get_world_size(_cp_group) > 1:
        torch.distributed.all_reduce(loss_weight, group=_cp_group)
    loss_weight = 1 / torch.clamp_min(loss_weight.sqrt(), 1)
    loss_weight = torch.where(
        loss_mask == 1, loss_weight.unsqueeze(1),
        torch.tensor(0.0, dtype=loss_weight.dtype, device=loss_weight.device)
    )

    weighted_losses = losses.float() * loss_weight
    #TODO: check loss nan or not

    # ``skip_cp_loss_reduce`` is set to ``calc_per_token_loss`` by the
    # finetune backend, so it serves as a reliable proxy for whether
    # per-token-loss normalisation is active.
    calculate_per_token_loss = loss_input.skip_cp_loss_reduce

    if calculate_per_token_loss:
        # --- Per-token-loss path (required by dynamic CP) ---
        # Return (loss_sum, effective_num_tokens, metrics) so that
        # forward_step_calc_loss takes the 3-element branch and avoids the
        # ``output_tensor *= cp_group_size`` bug.
        #
        # To preserve the square-averaging normalisation semantics
        # (divide by Σ 1/√N_i instead of by total token count), we encode
        # the square-averaging weight sum into ``num_tokens``.
        #
        # Because ``num_tokens`` must be an integer (``forward_step_calc_loss``
        # initialises it as ``torch.tensor(0, dtype=torch.int)`` and
        # ``torch.distributed.broadcast`` requires matching dtypes), we scale
        # both ``loss_sum`` and ``effective_num_tokens`` by a constant
        # ``_SA_SCALE`` so that the weight sum can be represented as an
        # integer:
        #
        #   effective_num_tokens = int(W / cp_size * _SA_SCALE)   (int)
        #   loss_sum            = weighted_sum * _SA_SCALE        (float tensor)
        #
        # After ``finalize_model_grads`` all-reduces num_tokens across the
        # dp_cp_group and calls ``scale_gradients(1 / total_num_tokens)``,
        # the gradient becomes:
        #
        #   _SA_SCALE * Σ_dp grad(weighted_sum_dp)
        #   ─────────────────────────────────────── = Σ_dp grad(weighted_sum_dp) / Σ_dp W_dp
        #   _SA_SCALE * Σ_dp W_dp
        #
        # which is exactly the correct global square-averaging gradient.
        # The _SA_SCALE factor cancels out perfectly.
        #
        # Dividing by cp_size avoids double-counting because
        # square_averaging_weights is per-sample (identical across CP ranks)
        # whereas loss_mask is per-token (different across CP ranks).
        _SA_SCALE = 10_000

        cp_size = dist.get_world_size(_cp_group)

        loss_sum = weighted_losses.sum().view(1).clone() * _SA_SCALE

        effective_num_tokens_float = square_averaging_weights.sum() / cp_size * _SA_SCALE
        effective_num_tokens = torch.tensor(int(effective_num_tokens_float.item()), dtype=torch.int)

        # Do NOT do CP all-reduce on loss_sum/effective_num_tokens when
        # skip_cp_loss_reduce is True (the dynamic-CP case).  Each CP rank
        # keeps its local values and finalize_model_grads aggregates
        # num_tokens across the dp_cp_group.
        if not loss_input.skip_cp_loss_reduce and cp_size > 1:
            # Non-dynamic-CP fallback: CP AVG on [loss_sum, effective_num_tokens]
            loss_for_reduce = torch.stack([loss_sum.float(), effective_num_tokens.float()])
            torch.distributed.all_reduce(
                loss_for_reduce, group=_cp_group, op=torch.distributed.ReduceOp.AVG
            )
            loss_sum = loss_for_reduce[0].view(1)
            effective_num_tokens = loss_for_reduce[1].to(torch.int)

        # Reporting loss: weighted mean for display purposes (unscaled)
        reporting_loss = (weighted_losses.sum() / square_averaging_weights.sum()).clone().detach()
        reporting_loss = average_losses_across_data_parallel_group([reporting_loss])
        metrics = {"lm_loss": reporting_loss}

        return (loss_sum, effective_num_tokens, metrics)
    else:
        # --- Legacy path (calculate_per_token_loss=False) ---
        # losses.sum() 包括了 mbs 序列之和，计算 loss 时应该求 mean
        # 所以 square_averaging_weights 直接多个求 sum 就可以了
        loss = weighted_losses.sum().view(1).clone() / square_averaging_weights.sum().view(1)
        if (not loss_input.skip_cp_loss_reduce and dist.get_world_size(_cp_group) > 1):
            torch.distributed.all_reduce(loss, group=_cp_group)

        reporting_loss = loss.clone().detach()
        reporting_loss = average_losses_across_data_parallel_group([reporting_loss])
        return (loss, {"lm_loss": reporting_loss})


@register_loss(backends=("mcore", ), loss_name="ppo_value_loss")
def value_loss_func(ppo_config, old_values, values, returns, mask):
    if ppo_config.ppo_value_clip is not None:
        values_clipped = old_values + (values - old_values).clamp(
            -ppo_config.ppo_value_clip, ppo_config.ppo_value_clip
        )
        surr1 = (values_clipped - returns)**2
        surr2 = (values - returns)**2
        loss = torch.max(surr1, surr2)
    else:
        loss = (values - returns)**2

    with torch.no_grad():
        numel = mask.sum()
    loss = masked_mean(0.5 * loss, mask)
    metrics = {"value_loss": torch.stack([loss.detach() * numel, numel])}

    return loss, metrics


@register_loss(backends=("mcore", ), loss_name="rm_bt")
def rm_bt_loss_func(
    config,
    loss_input: FinetuneLossInput,
):
    """Bradley-Terry pairwise reward-model loss (sequence-level scalar reward).

    The model carries a scalar value head (``LinearForLastLayer`` -> 1), so
    ``logits`` here is the per-token scalar ``[B, S, 1]``. The sequence-level
    reward is read at each sequence's last valid token (index
    ``sequence_lengths - 1``). Within a micro-batch the first half are the
    chosen sequences and the second half the rejected ones.

    Parameters
    ----------
    loss_input : FinetuneLossInput
        ``logits`` shape ``[B, S, 1]``; ``batch["sequence_lengths"]`` shape
        ``[B]`` holds the valid-token count per sequence (last index + 1).
    """
    batch = loss_input.batch
    scores = loss_input.logits
    assert scores is not None, "rm_bt loss requires the scalar head output as logits"
    assert scores.dim() == 3 and scores.shape[-1] == 1, \
        f"rm_bt expects [B, S, 1] scalar head output, got {tuple(scores.shape)}"

    sequence_lengths = batch["sequence_lengths"]
    bs = scores.shape[0]
    assert bs % 2 == 0, f"reward model micro-batch must be 2*n (chosen|rejected), got {bs}"

    # last-token pooling: hidden_state at index (sequencxe_lengths - 1)
    last_idx = (sequence_lengths.long() - 1).clamp(min=0)
    rewards = scores[torch.arange(bs, device=scores.device), last_idx, 0].float()  # [B]

    r_chosen, r_rejected = rewards.split(bs // 2, dim=0)
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()

    with torch.no_grad():
        acc = (r_chosen > r_rejected).float().mean()
        margin = (r_chosen - r_rejected).mean()
        metrics = {
            "rm-metrics/loss": loss.detach().float(),
            "rm-metrics/acc": acc.float(),
            "rm-metrics/reward_margin": margin.float(),
            "rm-metrics/reward_chosen": r_chosen.mean().float(),
            "rm-metrics/reward_rejected": r_rejected.mean().float(),
        }
    keys = sorted(metrics.keys())
    avg_metrics = average_losses_across_data_parallel_group([metrics[k] for k in keys])
    metrics = {k: v for k, v in zip(keys, avg_metrics)}

    return (loss, metrics)


register_loss(backends=("mcore", ), loss_name="grad_cache_loss")(grad_cache_loss_replay_func)
