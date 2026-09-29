# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
from contextlib import nullcontext
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from tqdm import tqdm

from megatron.core import mpu
from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices
from megatron.core.utils import divide

from gpatch_v4.core.mappings import (
    all_gather_from_context_parallel_region,
    all_gather_from_context_parallel_region_no_zigzag,
)
from gpatch_v4.core.seqlen_balancing import convert_mbs_for_pack_seq
from gpatch_v4.extended_model import (
    DeepseekV4PrepareDataForwardLLM,
    Qwen4ExpPrepareDataForwardLLM,
)
from gpatch_v4.models.deepseek_v4.thd import count_cross_cp_segments
from gpatch_v4.models.hp_module import HpModule
from gpatch_v4.training_backend.fsdp2_backend.linear_ce import linear_ce_forward_context
from gpatch_v4.training_backend.fsdp2_backend.mixin.unpack_seq import (
    unpack_thd_log_probs,
)
from gpatch_v4.training_backend.fsdp2_backend.mtp_loss import mtp_per_depth_valid_count
from gpatch_v4.training_backend.loss import Fsdp2FinetuneLossInput, get_loss_fn
from gpatch_v4.training_backend.loss_factory import load_balancing_loss_func
from gpatch_v4.utils import (
    clear_memory,
    get_batches_max_seqlen,
    get_k_split_list,
    get_max_seqlen_within_dp,
    log,
)
from gpatch_v4.utils.dynamic_cp_utils import (
    as_sequence_sample_mask,
    packed_to_response_padded,
    reconstruct_dynamic_cp_packed_tensor,
)
from gpatch_v4.utils.training_utils import (
    build_grpo_compact_ce_mask,
    selective_log_softmax_raw,
)

try:
    from gpatch_v4.models.deepseek_v4.dspark import (
        prepare_dspark_batch,
        shard_dspark_batch_for_contiguous_cp,
    )
except ImportError:
    prepare_dspark_batch = None
    shard_dspark_batch_for_contiguous_cp = None

try:
    from gpatch_v4.training_backend.fsdp2_backend.dspark_loss import (
        accumulate_dspark_report_metrics,
        dspark_loss_denominator,
        reduce_dspark_metrics,
    )
except ImportError:
    accumulate_dspark_report_metrics = None
    dspark_loss_denominator = None
    reduce_dspark_metrics = None

_FSDP2_PACKED_RL_PREPARE_TYPES = tuple(
    cls for cls in (
        DeepseekV4PrepareDataForwardLLM,
        Qwen4ExpPrepareDataForwardLLM,
    ) if cls is not None
)

selective_log_softmax_compiled = torch.compile(dynamic=True)(selective_log_softmax_raw)


class ForwardStepMixin:
    """FSDP2 logprob / SFT forward.

    ``compute_logprobs`` calls ``_maybe_router_replay``. The engine MUST also
    mix in :class:`RouterReplayMixin`.
    """
    def _dynamic_cp_partition_mode(
        self,
        batch: Dict[str, Any],
        fwd_kwargs: Dict[str, Any],
    ) -> str:
        packed_seq_params = fwd_kwargs.get("packed_seq_params")
        if packed_seq_params is None:
            packed_seq_params = batch.get("full_packed_seq_params")
        cp_partition_mode = self.prepare_data.cp_partition_mode
        if packed_seq_params is not None:
            cp_partition_mode = (
                getattr(packed_seq_params, "cp_partition_mode", None) or cp_partition_mode
            )
        if cp_partition_mode not in ("contiguous", "zigzag"):
            raise ValueError(f"Unsupported dynamic-CP partition mode: {cp_partition_mode!r}")
        return cp_partition_mode

    def _dynamic_cp_local_tensor(
        self,
        batch: Dict[str, Any],
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        cp_group = batch["cp_group"]
        cp_size = dist.get_world_size(cp_group)
        if cp_size == 1:
            return tensor
        assert tensor.shape[0] == 1 and tensor.shape[1] % cp_size == 0
        local_tokens = tensor.shape[1] // cp_size
        cp_rank = dist.get_rank(cp_group)
        cp_partition_mode = batch["cp_partition_mode"]
        if cp_partition_mode == "contiguous":
            start = cp_rank * local_tokens
            return tensor[:, start:start + local_tokens].contiguous()
        if cp_partition_mode == "zigzag":
            index = get_thd_partitioned_indices(
                batch["cu_seqlens_padded"],
                tensor.shape[1],
                cp_size,
                cp_rank,
            ).to(device=tensor.device, dtype=torch.long)
            return tensor.index_select(1, index).contiguous()
        raise ValueError(f"Unsupported dynamic-CP partition mode: {cp_partition_mode!r}")

    def _dynamic_cp_response_pad(
        self,
        batch: Dict[str, Any],
        curr_log_probs: torch.Tensor,
        entropy: torch.Tensor,
    ) -> Dict[str, Any]:
        cp_group = batch["cp_group"]
        local_cp_size = batch["local_cp_size"]
        cu_seqlens_padded = batch["cu_seqlens_padded"]
        cp_partition_mode = batch["cp_partition_mode"]
        reconstructed = {
            "curr_log_probs":
                reconstruct_dynamic_cp_packed_tensor(
                    curr_log_probs,
                    cu_seqlens_padded,
                    local_cp_size,
                    cp_partition_mode=cp_partition_mode,
                    cp_group=cp_group,
                ),
            "per_token_entropy":
                reconstruct_dynamic_cp_packed_tensor(
                    entropy,
                    cu_seqlens_padded,
                    local_cp_size,
                    cp_partition_mode=cp_partition_mode,
                    cp_group=cp_group,
                ),
        }
        padded = packed_to_response_padded(
            {
                **reconstructed,
                "mask": batch["mask"],
                "advantages": batch["advantages"],
                "prev_log_probs": batch["prev_log_probs"],
                "ref_log_probs": batch.get("ref_log_probs"),
                "rollout_log_probs": batch.get("rollout_log_probs"),
                "teacher_log_probs": batch.get("teacher_log_probs"),
                "sample_mask": batch.get("sample_mask"),
                "token_weights": batch.get("token_weights"),
            },
            cu_seqlens_padded,
            batch["dyn_cp_response_start"],
            batch["dyn_cp_response_length"],
        )
        mask = padded["mask"]
        per_token_entropy = padded["per_token_entropy"]
        return {
            **padded,
            "sample_mask": as_sequence_sample_mask(padded["sample_mask"]),
            "scaled_entropy": (per_token_entropy * mask).sum() / mask.sum().clamp(min=1),
        }

    @torch.no_grad()
    def compute_logprobs(
        self,
        model,
        batches_list: List[Dict[str, Any]],
        batch_log_str: str,
        enable_r3: bool = False,
    ) -> torch.Tensor:
        total_samples = len(batches_list)
        pack_seq = self.policy_config.ppo_pack_seq
        if pack_seq:
            assert _FSDP2_PACKED_RL_PREPARE_TYPES and isinstance(
                self.prepare_data,
                _FSDP2_PACKED_RL_PREPARE_TYPES,
            ), ("FSDP2 packed RL forward currently supports deepseek_v4 and qwen4_exp")
            seq_length = self.training_config.seq_length
            batch_iter = convert_mbs_for_pack_seq(
                batches_list,
                max_token_len=seq_length,
                pad_each_doc_to_multi_of=self.prepare_data._pad_each_doc_to_multi_of,
                dp_group=mpu.get_data_parallel_group(),
                cp_size=mpu.get_context_parallel_world_size(),
            )
            sample_order = {id(sample): idx for idx, sample in enumerate(batches_list)}
            logprobs_list = [None] * total_samples
        else:
            seq_length = get_batches_max_seqlen(
                batches_list, self.training_config.pad_to_mulitiple_of
            )
            seq_length = get_max_seqlen_within_dp(seq_length)
            num_microbatches = divide(total_samples, self.forward_only_mbs)
            batch_iter = get_k_split_list(batches_list, num_microbatches)
            logprobs_list = []

        for batches in tqdm(batch_iter, desc=batch_log_str, disable=dist.get_rank() != 0):
            model_fwd_args = self.prepare_data.model_forward_only(
                batches,
                seq_length,
                self.tokenizer.pad_token_id,
                pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                vocab_size=self._get_vocab_size(),
            )
            target = model_fwd_args.pop("target")
            full_psp = model_fwd_args.pop("full_packed_seq_params", None)
            replay_ctx = (
                self._maybe_router_replay(
                    model,
                    batches,
                    seq_length,
                    packed_seq_params=full_psp,
                ) if enable_r3 else nullcontext()
            )
            with replay_ctx:
                pre_shifted = full_psp is not None
                response_mask = (
                    build_grpo_compact_ce_mask(batches, target)
                    if self.training_config.ce_compaction else None
                )
                with self._maybe_rl_linear_ce_context(
                    model,
                    target,
                    pre_shifted=pre_shifted,
                    return_entropy=False,
                    response_mask=response_mask,
                ):
                    head_out = model(**model_fwd_args).logits
                logprobs = self._rl_logprobs_from_head_output(
                    head_out,
                    target,
                    pre_shifted=pre_shifted,
                    allow_compile=False,
                )
            if full_psp is None:
                logprobs_list.append(logprobs)
                continue

            restored_logprobs = unpack_thd_log_probs(
                logprobs.cpu(),
                full_psp.cu_seqlens_q,
                full_psp.cu_seqlens_q_padded,
            )
            assert len(restored_logprobs) == len(batches)
            for sample, restored in zip(batches, restored_logprobs, strict=True):
                logprobs_list[sample_order[id(sample)]] = restored

        if pack_seq:
            assert all(logprob is not None for logprob in logprobs_list)
            logprobs = logprobs_list
        else:
            logprobs = torch.cat(logprobs_list) if len(logprobs_list) > 0 else None
            assert logprobs.shape[0] == total_samples
            logprobs = [logprob.squeeze(0) for logprob in logprobs.cpu().chunk(total_samples)]
        clear_memory()
        return logprobs

    def _maybe_rl_linear_ce_context(
        self,
        model,
        target: torch.Tensor,
        *,
        pre_shifted: bool,
        return_entropy: bool,
        response_mask: torch.Tensor | None = None,
    ):
        if not self.training_config.use_linear_ce:
            return nullcontext()
        local_labels = self._prepare_rl_linear_ce_labels(target, pre_shifted=pre_shifted)
        local_mask = None
        if self.training_config.ce_compaction:
            assert response_mask is not None
            if pre_shifted:
                assert response_mask.shape == target.shape
                local_mask = response_mask
            else:
                assert response_mask.shape == (target.shape[0], target.shape[1] - 1)
                # The final hidden state has no next-token policy loss.
                local_mask = torch.cat(
                    (response_mask, response_mask.new_zeros((target.shape[0], 1))), dim=-1
                )
            local_mask = self.prepare_data.rl_train_cp_chunk_single_data(local_mask)
        return linear_ce_forward_context(
            model,
            local_labels,
            self.training_config.linear_ce_backend,
            return_entropy=return_entropy,
            loss_mask=local_mask,
        )

    def _rl_logprobs_from_head_output(
        self,
        head_output: torch.Tensor,
        targets: torch.Tensor,
        *,
        pre_shifted: bool,
        allow_compile: bool,
    ) -> torch.Tensor:
        """Map lm_head output to policy-axis log-probs (fused or dense gather)."""
        if self.training_config.use_linear_ce:
            assert isinstance(head_output, torch.Tensor)
            return self._finalize_rl_linear_ce_log_probs(head_output, pre_shifted=pre_shifted)
        return self.gather_log_probs_packed(
            head_output.float(),
            targets,
            allow_compile=allow_compile,
            pre_shifted=pre_shifted,
        )

    def _rl_logprobs_and_entropy_from_head_output(
        self,
        head_output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        response_mask: torch.Tensor,
        *,
        pre_shifted: bool,
        allow_compile: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map lm_head output to policy-axis log-probs and entropy."""
        if self.training_config.use_linear_ce:
            assert (
                isinstance(head_output, tuple) and len(head_output) == 2
            ), f"expected (log_probs, entropy), got {type(head_output)}"
            local_log_probs, local_entropy = head_output
        else:
            assert isinstance(head_output, torch.Tensor)
            local_log_probs = head_output
            logits = head_output.float()
            local_entropy = -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1)

        log_probs = self._rl_logprobs_from_head_output(
            local_log_probs,
            targets,
            pre_shifted=pre_shifted,
            allow_compile=allow_compile,
        )
        entropy = self._finalize_rl_policy_entropy(
            local_entropy,
            response_mask,
            pre_shifted=pre_shifted,
        )
        return log_probs, entropy

    def _prepare_rl_linear_ce_labels(
        self,
        targets: torch.Tensor,
        *,
        pre_shifted: bool,
    ) -> torch.Tensor:
        """Build CP-local next-token labels aligned with lm_head hidden states."""
        shifted = targets if pre_shifted else targets.roll(shifts=-1, dims=-1)
        return self.prepare_data.rl_train_cp_chunk_single_data(shifted)

    def _finalize_rl_linear_ce_log_probs(
        self,
        local_log_probs: torch.Tensor,
        *,
        pre_shifted: bool,
    ) -> torch.Tensor:
        """CP all-gather fused log-probs and align to the policy-loss token axis."""
        log_probs = self._all_gather_cp_aware(local_log_probs)
        if pre_shifted:
            return log_probs.contiguous()
        return log_probs[:, :-1].contiguous()

    def _finalize_rl_policy_entropy(
        self,
        local_entropy: torch.Tensor,
        response_mask: torch.Tensor,
        *,
        pre_shifted: bool,
    ) -> torch.Tensor:
        """CP all-gather per-token entropy and align to the policy-loss token axis."""
        entropy = self._all_gather_cp_aware(local_entropy)
        if not pre_shifted:
            entropy = entropy[:, :-1]
        assert entropy.shape == response_mask.shape, (
            f"entropy shape {tuple(entropy.shape)} must match response mask "
            f"shape {tuple(response_mask.shape)}"
        )
        return entropy

    def _all_gather_cp_aware(self, local_tensor: torch.Tensor, gather_dim: int = 1) -> torch.Tensor:
        """All-gather across CP ranks, dispatching the correct pattern.

        Qwen4/DSv4 use contiguous splits; standard models use mirror-pair
        zigzag.
        """
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size <= 1:
            return local_tensor
        cp_partition_mode = self.prepare_data.cp_partition_mode
        if cp_partition_mode == "contiguous":
            return all_gather_from_context_parallel_region_no_zigzag(local_tensor, gather_dim)
        if cp_partition_mode == "zigzag":
            return all_gather_from_context_parallel_region(local_tensor, gather_dim)
        raise ValueError(f"Unsupported CP partition mode: {cp_partition_mode!r}")

    def gather_log_probs_packed(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        allow_compile: bool,
        cu_seqlens: torch.Tensor | float | None = None,
        temperature: torch.Tensor | None = None,
        pre_shifted: bool = False,
    ) -> torch.Tensor:
        """Gather next-token log probs for packed sequences.

        Parameters:
            logits: ``[B, S, V]``.
            targets: ``[B, S]`` input tokens in BSHD mode, or already shifted
                target tokens in THD mode.
            cu_seqlens: when provided, marks ``targets`` as pre-shifted.
            pre_shifted: skip the cross-sample ``roll(-1)`` shift.

        Returns:
            ``[B, S-1]`` in BSHD mode, or ``[B, S]`` in pre-shifted THD mode.
        """
        # Handle batch dimension - logits should be [batch_size, seq_len, vocab_size]
        assert logits.dim() == 3
        assert targets.dim() == 2
        if cu_seqlens is not None:
            pre_shifted = True

        if temperature is not None:
            logits = logits.div(temperature)

        shifted_targets = targets if pre_shifted else targets.roll(shifts=-1, dims=-1)
        local_targets = self.prepare_data.rl_train_cp_chunk_single_data(shifted_targets)

        assert logits.shape[:2] == local_targets.shape, f"{logits.shape=} {local_targets.shape=}"

        # Gather log probs for targets
        selective_log_softmax = selective_log_softmax_compiled if allow_compile else selective_log_softmax_raw
        curr_log_probs = selective_log_softmax(logits, local_targets)
        curr_log_probs = self._all_gather_cp_aware(curr_log_probs)

        if pre_shifted:
            return curr_log_probs.contiguous()
        return curr_log_probs[:, :-1].contiguous()

    def get_logprob_and_entropy(
        self,
        logits: torch.Tensor,
        target_tokens: torch.Tensor,
        allow_compile: bool,
        temperature: float | None = None,
        return_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities and entropy.

        Parameters:
            logits: ``[B, S, V]``.
            target_tokens: ``[B, S]``.
            allow_compile:
            temperature: optional.

        Returns:
            log_probs: ``[B, S-1]``.
            entropy: ``[B, S-1]``.
        """
        # shifted_logits = logits[:, :-1, :]
        log_probs = self.gather_log_probs_packed(
            logits, target_tokens, allow_compile=allow_compile, temperature=temperature
        )
        if return_entropy:
            shifted_logits = logits[:, :-1, :]
            log_probs_full = torch.log_softmax(shifted_logits, dim=-1)
            probs = torch.softmax(shifted_logits, dim=-1)
            entropy = -(probs * log_probs_full).sum(dim=-1)
        else:
            entropy = None
        return log_probs, entropy

    def _calc_dynamic_mbs(self, len_batch, max_seq_length):
        dynamic_mbs = self.config.training.seq_length // max_seq_length
        while dynamic_mbs >= 1:
            if len_batch % dynamic_mbs == 0:
                break
            dynamic_mbs -= 1
        assert dynamic_mbs >= 1
        return dynamic_mbs

    def _prepare_online_dspark_batch(
        self,
        prep_batch: Dict[str, Any],
        prep_fwd_kwargs: Dict[str, Any],
        device: torch.device | int,
    ) -> Any:
        assert prepare_dspark_batch is not None
        full_input_ids = prep_batch.pop("full_input_ids")
        full_labels = prep_batch["full_labels"]
        full_loss_mask = prep_batch["full_loss_mask"]
        packed_seq_params = prep_fwd_kwargs.get("packed_seq_params")
        seed_value = int(
            torch.empty((),
                        dtype=torch.int64).random_(generator=self._dspark_anchor_generator).item()
        )
        if self.cp_size > 1:
            assert (full_input_ids.shape[1] == prep_fwd_kwargs["input_ids"].shape[1] * self.cp_size)
            cp_group = self.model._cp_group
            cp_rank = dist.get_rank(cp_group)
            assert cp_rank == self.model._cp_rank
            anchor_seed = torch.tensor(seed_value, dtype=torch.int64, device=device)
            dist.broadcast(
                anchor_seed,
                src=dist.get_global_rank(cp_group, 0),
                group=cp_group,
            )
            seed_value = int(anchor_seed.item())
        rng = torch.Generator(device=full_input_ids.device)
        rng.manual_seed(seed_value)
        dspark_batch = prepare_dspark_batch(
            full_input_ids,
            full_labels,
            full_loss_mask,
            num_anchors=self.training_config.dspark_num_anchors,
            block_size=self.hf_config.dspark_block_size,
            packed_seq_params=packed_seq_params,
            rng=rng,
        )
        if self.cp_size > 1:
            assert shard_dspark_batch_for_contiguous_cp is not None
            dspark_batch = shard_dspark_batch_for_contiguous_cp(
                dspark_batch,
                sequence_length=full_input_ids.shape[1],
                cp_rank=cp_rank,
                cp_size=self.cp_size,
            )
        prep_fwd_kwargs["dspark_batch"] = dspark_batch
        return dspark_batch

    def _maybe_log_thd_pack_layout(
        self,
        *,
        already_logged: bool,
        full_psp: Any,
        local_seqlen: int,
        prep_fwd_kwargs: Dict[str, Any],
        online_train_dspark: bool,
    ) -> bool:
        if (already_logged or not self.config.debug.log_thd_pack_layout or dist.get_rank() != 0):
            return already_logged
        if full_psp is None:
            log(f"[THD pack/CP] is_thd=0 local={local_seqlen} cp={self.cp_size}")
        else:
            raw_lens = (full_psp.cu_seqlens_q[1:] - full_psp.cu_seqlens_q[:-1]).tolist()
            pad_lens = (full_psp.cu_seqlens_q_padded[1:] -
                        full_psp.cu_seqlens_q_padded[:-1]).tolist()
            dspark_extra = ""
            if online_train_dspark:
                keep = prep_fwd_kwargs["dspark_batch"].block_keep_mask
                dspark_extra = (f" dspark_owned={int(keep.sum().item())}/{int(keep.numel())}")
            log(
                f"[THD pack/CP] is_thd=1 n_seg={len(raw_lens)} "
                f"raw_lens={raw_lens} pad_lens={pad_lens} "
                f"T={int(full_psp.total_seqlen)} local={local_seqlen} "
                f"cp={self.cp_size} n_cross="
                f"{count_cross_cp_segments(full_psp.cu_seqlens_q_padded, self.cp_size)}"
                f"{dspark_extra}"
            )
        return True

    def _finetune_step(
        self, batch: List[Dict[str, Any]], num_microbatches: int, forward_only: bool = False
    ):
        training_config = self.config.training
        dp_size = mpu.get_data_parallel_world_size()
        dynamic_cp = self.policy_config.dist_config.dynamic_context_parallel

        is_dpo = training_config.loss_func == "dpo"
        use_linear_ce = training_config.use_linear_ce
        if use_linear_ce:
            assert not is_dpo, "FSDP2 linear CE currently supports SFT only"
            assert training_config.loss_func == "cross_entropy", (
                "FSDP2 linear CE requires loss_func='cross_entropy'"
            )
        loss_fn = get_loss_fn("fsdp2", training_config.loss_func)
        gbs_factor = 2 if is_dpo else 1
        if not dynamic_cp:
            assert training_config.train_gbs * gbs_factor == len(
                batch
            ) * dp_size, f"{training_config.train_gbs=} {gbs_factor=} {len(batch)=} {dp_size=}"

        if is_dpo:
            assert not training_config.online_mtp_sft, (
                "DPO + online MTP training is not yet supported on FSDP2"
            )
            assert not training_config.online_train_dspark, "DPO + DSpark is not supported on FSDP2"
            assert not training_config.use_dynamic_mbs, "DPO + dynamic_mbs not yet supported"

        pack_seq = self.policy_config.ppo_pack_seq
        dynamic_cp_microbatches = None
        if dynamic_cp:
            assert training_config.train_gbs == len(batch) * self.dp_size
            max_seq_length = get_batches_max_seqlen(batch, self.training_config.pad_to_mulitiple_of)
            max_seq_length = get_max_seqlen_within_dp(max_seq_length)
            max_seq_length = min(max_seq_length, training_config.seq_length)
            dynamic_cp_microbatches, num_microbatches, seqlen_sum, seqlen_sq_sum = (
                self.prepare_data.sft_reroute_data_for_dynamic_cp(
                    list(batch),
                    self.tokenizer.pad_token_id,
                    vocab_size=self._get_vocab_size(),
                    pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                )
            )
            data_iter = [[microbatch] for microbatch in dynamic_cp_microbatches]
        elif pack_seq:
            assert not is_dpo, "DPO + pack_seq not supported"
            assert not training_config.use_dynamic_mbs, (
                "use_dynamic_mbs is incompatible with ppo_pack_seq "
                "(pack-seq does its own token-budget micro-batch splitting)"
            )
            dp_group = mpu.get_data_parallel_group()
            data_iter = convert_mbs_for_pack_seq(
                batch,
                max_token_len=training_config.seq_length,
                pad_each_doc_to_multi_of=self.prepare_data._pad_each_doc_to_multi_of,
                dp_group=dp_group,
                cp_size=mpu.get_context_parallel_world_size(),
            )
            num_microbatches = len(data_iter)
            max_seq_length = training_config.seq_length
        else:
            if self.config.debug.experimental_pad_to_max_length:
                max_seq_length = self.config.training.seq_length
            else:
                max_seq_length = get_batches_max_seqlen(
                    batch, self.training_config.pad_to_mulitiple_of
                )
                max_seq_length = get_max_seqlen_within_dp(max_seq_length)
                max_seq_length = min(max_seq_length, training_config.seq_length)
            if training_config.loss_func in ["square_averaging_cross_entropy"]:
                assert False
                update_square_averaging_token_len(batch, max_seq_length)

            dynamic_mbs = 1
            if training_config.use_dynamic_mbs:
                dynamic_mbs = self._calc_dynamic_mbs(len(batch), max_seq_length)
                num_microbatches = len(batch) // dynamic_mbs

            data_iter = get_k_split_list(batch, num_microbatches)
        if is_dpo:
            dpo_mbs = len(batch) // num_microbatches
            assert dpo_mbs == training_config.train_mbs * 2, (
                f"DPO microbatch must have 2*train_mbs samples, got {dpo_mbs} "
                f"(train_mbs={training_config.train_mbs})"
            )
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        # Switch-style MoE load-balancing loss (FSDP2). Computed for the metric
        # in both train and eval; only added to backward when training. Reads
        # router_logits captured via forward hooks; these keep their grad_fn
        # only under non-reentrant gradient checkpointing (the transformers
        # default), so recompute is supported. Under use_reentrant=True the hook
        # captures the no-grad first-pass tensor and the balance gradient is
        # silently dropped, so recompute must stay non-reentrant.
        enable_balance_loss = training_config.moe_balance_loss_coef > 0

        # 在 dp rank 与 micro batch 之间统计 valid token 数量，用于后面 loss 的归一化。
        # MTP 额外统计逐 depth 的全局 valid token 数，作为无偏 den 替换
        # calculate_mtp_loss 里的 per-micro-batch den。
        enable_mtp_loss = training_config.online_mtp_sft
        online_train_dspark = training_config.online_train_dspark
        mtp_num_depth = int(self.hf_config.num_nextn_predict_layers) if enable_mtp_loss else 0
        mtp_cp_rank = dist.get_rank(self.model._cp_group) if self.cp_size > 1 else 0
        device = torch.cuda.current_device()
        global_n = torch.zeros((), device=device)
        global_n_for_mtp = torch.zeros(mtp_num_depth, device=device) if enable_mtp_loss else None
        global_n_for_dspark = torch.zeros((), device=device) if online_train_dspark else None
        prepared_dspark_data = [] if online_train_dspark else None
        pack_n_seg_max = 0
        pack_n_cross_max = 0
        pack_is_thd_min = 1 if pack_seq or dynamic_cp else 0
        pack_total_max = 0
        pack_local_max = 0
        pack_raw_max = 0
        dspark_owned_max = 0
        dspark_slots_max = 0
        logged_thd_pack_layout = False
        prepare_data_func = (
            self.prepare_data.sft_train_with_dynamic_cp
            if dynamic_cp else self.prepare_data.sft_train
        )
        for mb_samples in data_iter:
            prepare_samples = [dict(mb_samples[0])] if dynamic_cp else mb_samples
            prep_batch, prep_fwd_kwargs = prepare_data_func(
                prepare_samples,
                max_seq_length,
                self.tokenizer.pad_token_id,
                comput_attn_mask=False,
                pad_with_random_token=False,
                input_teacher_hidden_states=False,
                vocab_size=self._get_vocab_size(),
            )
            if dynamic_cp:
                prep_batch["cp_partition_mode"] = self._dynamic_cp_partition_mode(
                    prep_batch,
                    prep_fwd_kwargs,
                )
            local_seqlen = prep_fwd_kwargs["input_ids"].shape[1]
            pack_local_max = max(pack_local_max, local_seqlen)
            full_psp = prep_batch.get("full_packed_seq_params")
            if full_psp is None:
                pack_is_thd_min = 0
                pack_total_max = max(pack_total_max, local_seqlen * self.cp_size)
            else:
                cu_pad = full_psp.cu_seqlens_q_padded
                cu_raw = full_psp.cu_seqlens_q
                n_seg = int(cu_pad.numel() - 1)
                pack_n_seg_max = max(pack_n_seg_max, n_seg)
                total_seqlen = int(full_psp.total_seqlen)
                pack_total_max = max(pack_total_max, total_seqlen)
                pack_raw_max = max(pack_raw_max, int((cu_raw[1:] - cu_raw[:-1]).sum().item()))
                n_cross = count_cross_cp_segments(cu_pad, self.cp_size)
                pack_n_cross_max = max(pack_n_cross_max, n_cross)
            global_n += (prep_batch["labels"] != -100).sum()
            if enable_mtp_loss:
                global_n_for_mtp += mtp_per_depth_valid_count(
                    prep_batch["full_labels"],
                    prep_batch["full_loss_mask"],
                    num_depth=mtp_num_depth,
                    cp_size=self.cp_size,
                    cp_rank=mtp_cp_rank,
                    packed_seq_params=prep_batch.get("full_packed_seq_params"),
                )
            if online_train_dspark:
                dspark_batch = self._prepare_online_dspark_batch(
                    prep_batch, prep_fwd_kwargs, device
                )
                dspark_owned_max = max(
                    dspark_owned_max, int(dspark_batch.block_keep_mask.sum().item())
                )
                dspark_slots_max = max(dspark_slots_max, int(dspark_batch.block_keep_mask.numel()))
                global_n_for_dspark += dspark_loss_denominator(
                    dspark_batch,
                    block_size=self.hf_config.dspark_block_size,
                    loss_decay_gamma=training_config.dspark_loss_decay_gamma,
                )
                prepared_dspark_data.append((prep_batch, prep_fwd_kwargs))
            logged_thd_pack_layout = self._maybe_log_thd_pack_layout(
                already_logged=logged_thd_pack_layout,
                full_psp=full_psp,
                local_seqlen=local_seqlen,
                prep_fwd_kwargs=prep_fwd_kwargs,
                online_train_dspark=online_train_dspark,
            )
        pack_stats = torch.tensor(
            [
                pack_n_seg_max,
                pack_n_cross_max,
                pack_total_max,
                pack_local_max,
                pack_raw_max,
                dspark_owned_max,
                dspark_slots_max,
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(pack_stats, op=dist.ReduceOp.MAX)
        pack_is_thd_t = torch.tensor([pack_is_thd_min], dtype=torch.float64, device=device)
        dist.all_reduce(pack_is_thd_t, op=dist.ReduceOp.MIN)
        dist.all_reduce(global_n)
        global_n = global_n.clamp_min(1.0)
        if enable_mtp_loss:
            dist.all_reduce(global_n_for_mtp)
            global_n_for_mtp = global_n_for_mtp.clamp_min(1.0)
        if online_train_dspark:
            dist.all_reduce(global_n_for_dspark)
            global_n_for_dspark = global_n_for_dspark.clamp_min(1.0)

        dpo_metric_sums = {} if is_dpo else None

        report_loss = 0.
        report_main_loss = 0.
        report_mtp_loss = 0.
        report_mtp_depth_loss = None
        report_dspark_metrics = {} if online_train_dspark else None
        report_balance_loss = 0.
        train_iter = prepared_dspark_data if online_train_dspark else data_iter
        for microbatch_index, train_item in enumerate(tqdm(train_iter, disable=True)):
            if online_train_dspark:
                batch, fwd_kwargs = train_item
            else:
                prepare_samples = [dict(train_item[0])] if dynamic_cp else train_item
                batch, fwd_kwargs = prepare_data_func(
                    prepare_samples,
                    max_seq_length,
                    self.tokenizer.pad_token_id,
                    comput_attn_mask=self.training_config.comput_attn_mask,
                    pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                    input_teacher_hidden_states=getattr(
                        self.config.training,
                        "enable_teacher_kl_loss",
                        False,
                    ),
                    vocab_size=self._get_vocab_size(),
                )
                if dynamic_cp:
                    batch["cp_partition_mode"] = self._dynamic_cp_partition_mode(
                        batch,
                        fwd_kwargs,
                    )

            if isinstance(self.model, HpModule):
                self.model.maybe_dump_prepared_microbatch_diagnostics(
                    batch,
                    fwd_kwargs,
                    microbatch_index,
                )

            loss_mask = batch["loss_mask"]
            labels = batch["labels"]
            if use_linear_ce:
                fwd_kwargs["labels"] = None
            if enable_balance_loss:
                fwd_kwargs["output_router_logits"] = True
            train_forward_context = (
                nullcontext() if self.training_config.recompute else self.checkpoint_context_fn()[0]
            )
            linear_ce_context = (
                linear_ce_forward_context(
                    self.model,
                    labels,
                    training_config.linear_ce_backend,
                    return_entropy=False,
                    loss_mask=loss_mask if training_config.ce_compaction else None,
                ) if use_linear_ce else nullcontext()
            )
            # Keep linear_ce context through loss.backward so recomputed lm_head
            # (if ever checkpointed) still sees labels; MTP nests rolled labels.
            with self._dynamic_cp_model_context(self.model, batch):
                with linear_ce_context:
                    if self.config.debug.debug_dtype:
                        from gpatch_v4.utils.debug_dtype import debug_dump_dtype
                        debug_dump_dtype(
                            model=self.model,
                            fwd_kwargs=fwd_kwargs,
                            dump_path="debug-tmp/sft_dtype.md",
                            watched_files=["modeling_deepseek_v4.py", "mtp.py"],
                            skip_ops=[],
                        )
                    with train_forward_context:
                        outputs = self.model(**fwd_kwargs)
                    if use_linear_ce:
                        # SFT uses return_entropy=False → logits is log_probs tensor.
                        log_probs = outputs.logits
                        assert isinstance(log_probs, torch.Tensor) and log_probs.dim() >= 2
                        per_token_linear_ce_loss = -log_probs.float()
                        logits = None
                    else:
                        # Cast to fp32 for numerically stable cross entropy (log-sum-exp in
                        # bf16/fp16 is lossy); consistent with logprob path above.
                        logits = outputs.logits.float()
                        per_token_linear_ce_loss = None

                    mtp_scale = float(
                        getattr(
                            outputs,
                            "mtp_loss_scaling_factor",
                            training_config.mtp_loss_scaling_factor,
                        )
                    )
                    loss_input = Fsdp2FinetuneLossInput(
                        labels_2d=labels,
                        loss_mask_2d=loss_mask,
                        batch=batch,
                        dp_size=self.dp_size,
                        logits=logits,
                        use_linear_ce=use_linear_ce,
                        per_token_linear_ce_loss=per_token_linear_ce_loss,
                        linear_ce_backend=(
                            training_config.linear_ce_backend if use_linear_ce else None
                        ),
                        global_n=global_n,
                        vocab_size=self._get_vocab_size(),
                        loss_fct=loss_fct,
                        online_train_dspark=online_train_dspark,
                        dspark_output=getattr(outputs, "dspark_output", None)
                        if online_train_dspark else None,
                        global_n_for_dspark=global_n_for_dspark,
                        enable_mtp=enable_mtp_loss,
                        mtp_per_depth_h=getattr(outputs, "mtp_per_depth_h", None)
                        if enable_mtp_loss else None,
                        lm_head=self.model.lm_head if enable_mtp_loss else None,
                        cp_group=(self.model._cp_group if self.cp_size > 1 else None)
                        if enable_mtp_loss else None,
                        packed_seq_params=fwd_kwargs.get("packed_seq_params")
                        if enable_mtp_loss else None,
                        global_n_for_mtp=global_n_for_mtp,
                        mtp_loss_scaling_factor=mtp_scale,
                        all_gather_fn=self._all_gather_cp_aware if is_dpo else None,
                    )
                    result = loss_fn(self.config, loss_input)
                    loss = result.loss
                    main_loss = result.main_loss
                    mtp_loss = result.mtp_loss
                    mtp_depth_losses = result.mtp_depth_losses
                    mtp_depth_loss_metrics = result.mtp_depth_loss_metrics
                    if is_dpo:
                        for k, v in result.metrics.items():
                            dpo_metric_sums[k] = dpo_metric_sums.get(k, 0.0) + v

                    balance_loss = None
                    if enable_balance_loss:
                        balance_loss = load_balancing_loss_func(
                            gate_logits=outputs.router_logits,
                            num_experts=self.model.num_experts,
                            top_k=self.model.num_experts_per_tok,
                            cp_group=(
                                batch["cp_group"] if dynamic_cp else
                                self.model._cp_group if self.cp_size > 1 else None
                            ),
                        )

                        # The current computation method follows HF.transformers, which introduces a bias due to varying sequence lengths.
                        # For example:
                        # ```
                        # s1 = [
                        #   [1, 2, ..., 10000],
                        #   [1],
                        # ]
                        # ```
                        # will produce a different result compared to
                        # ```
                        # s2 = [[1, 2, ...., 10000, 1]]
                        # ```
                        loss = (
                            loss +
                            training_config.moe_balance_loss_coef * balance_loss / num_microbatches
                        )

                    if not forward_only:
                        loss.backward()

            tmp = loss.detach().clone()
            tmp_main = main_loss.detach().clone()
            if self.cp_size > 1 and not is_dpo:
                dist.all_reduce(tmp, group=self.model._cp_group)
                dist.all_reduce(tmp_main, group=self.model._cp_group)
            report_loss += tmp.item()
            report_main_loss += tmp_main.item()
            if balance_loss is not None:
                # Already CP-global (identical on every CP rank); accumulate the
                # per-step mean contribution.
                report_balance_loss += balance_loss.detach().item() / num_microbatches
            if not is_dpo and mtp_loss is not None:
                tmp_mtp = mtp_loss.detach().clone()
                if self.cp_size > 1:
                    dist.all_reduce(tmp_mtp, group=self.model._cp_group)
                report_mtp_loss += tmp_mtp.item()
                if report_mtp_depth_loss is None:
                    report_mtp_depth_loss = [0.0 for _ in range(len(mtp_depth_losses))]
                for i, dloss in enumerate(mtp_depth_loss_metrics):
                    dtmp = dloss.detach().clone()
                    report_mtp_depth_loss[i] += dtmp.item()
            if online_train_dspark:
                assert report_dspark_metrics is not None
                assert result.dspark_result is not None
                accumulate_dspark_report_metrics(
                    report_dspark_metrics,
                    result.dspark_result,
                    cp_group=self.model._cp_group if self.cp_size > 1 else None,
                )

        if isinstance(self.model, HpModule):
            self.model.finish_step0_diagnostics()

        report_loss = torch.tensor(report_loss).to(torch.cuda.current_device())
        torch.distributed.all_reduce(report_loss, op=torch.distributed.ReduceOp.AVG)
        report_main_loss_t = torch.tensor(report_main_loss).to(torch.cuda.current_device())
        torch.distributed.all_reduce(report_main_loss_t, op=torch.distributed.ReduceOp.AVG)
        report_mtp_loss_t = torch.tensor(report_mtp_loss).to(torch.cuda.current_device())
        torch.distributed.all_reduce(report_mtp_loss_t, op=torch.distributed.ReduceOp.AVG)

        metric_prefix = "finetune"
        if forward_only:
            metric_prefix = "eval"
        metrics = {
            f"{metric_prefix}/lm_loss": report_main_loss_t.item(),
            f"{metric_prefix}/total_loss": report_loss.item(),
            f"{metric_prefix}/seq_length": max_seq_length,
        }
        if is_dpo:
            n_mb = max(num_microbatches, 1)
            for k, v in dpo_metric_sums.items():
                t = torch.tensor(v / n_mb).to(torch.cuda.current_device())
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                metrics[f"{metric_prefix}/dpo-metrics/{k}"] = t.item()
            metrics[f"{metric_prefix}/dpo-metrics/loss"] = report_loss.item()
        if enable_balance_loss:
            report_balance_loss_t = torch.tensor(report_balance_loss).to(
                torch.cuda.current_device()
            )
            torch.distributed.all_reduce(report_balance_loss_t, op=torch.distributed.ReduceOp.AVG)
            metrics[f"{metric_prefix}/balance_loss"] = report_balance_loss_t.item()
        if enable_mtp_loss:
            metrics[f"{metric_prefix}/mtp_loss"] = report_mtp_loss_t.item()
            if report_mtp_depth_loss is not None:
                for i, value in enumerate(report_mtp_depth_loss):
                    depth_t = torch.tensor(value).to(torch.cuda.current_device())
                    torch.distributed.all_reduce(depth_t, op=torch.distributed.ReduceOp.AVG)
                    metrics[f"{metric_prefix}/mtp_depth_{i}_loss"] = depth_t.item()
        if online_train_dspark:
            assert report_dspark_metrics is not None
            loss_components = report_dspark_metrics["loss_components"]
            reduce_dspark_metrics(
                metrics,
                metric_prefix=metric_prefix,
                block_size=self.hf_config.dspark_block_size,
                device=device,
                loss=loss_components[0].item(),
                ce_loss=loss_components[1].item(),
                l1_loss=loss_components[2].item(),
                confidence_loss=loss_components[3].item(),
                accept_rate_sums=report_dspark_metrics["accept_rate_sums"],
                accept_rate_counts=report_dspark_metrics["accept_rate_counts"],
                tau_sum=report_dspark_metrics["tau_sum"],
                block_count=report_dspark_metrics["block_count"],
            )
        if training_config.use_dynamic_mbs:
            metrics[f"{metric_prefix}/dynamic_mbs"] = dynamic_mbs
        metrics[f"{metric_prefix}/num_micro_batches"] = num_microbatches
        metrics[f"{metric_prefix}/pack_is_thd"] = pack_is_thd_t.item()
        metrics[f"{metric_prefix}/pack_n_segments"] = pack_stats[0].item()
        metrics[f"{metric_prefix}/pack_n_cross_cp_segs"] = pack_stats[1].item()
        metrics[f"{metric_prefix}/pack_total_seqlen"] = pack_stats[2].item()
        metrics[f"{metric_prefix}/cp_local_seqlen"] = pack_stats[3].item()
        metrics[f"{metric_prefix}/pack_raw_tokens"] = pack_stats[4].item()
        if dynamic_cp:
            assert dynamic_cp_microbatches is not None
            metrics[f"{metric_prefix}/dyn_cp_seqlen_sum"] = seqlen_sum
            metrics[f"{metric_prefix}/dyn_cp_seqlen_sq_sum"] = seqlen_sq_sum
            metrics[f"{metric_prefix}/dyn_cp_num_micro_batches"] = num_microbatches
            metrics.update(self._dynamic_cp_schedule_stats(dynamic_cp_microbatches, metric_prefix))
        if online_train_dspark:
            metrics[f"{metric_prefix}/dspark_n_owned_anchors"] = pack_stats[5].item()
            metrics[f"{metric_prefix}/dspark_n_slots"] = pack_stats[6].item()
        return metrics

    def clip_grad_norm_(self):
        if not isinstance(self.model, HpModule):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.optimizer.max_grad_norm
            )
            grad_norm = grad_norm.full_tensor()
            return grad_norm
        else:
            # `apply_hp` binds an EP-aware clip_grad_norm_ that aggregates
            # expert grad norms across the EP group before clipping. It
            # returns a Python float (`total_norm.item()` in
            # `_clip_grad_norm_multi_mesh`); the caller in
            # `fsdp2_engine_lm.py` expects a Tensor (does
            # `torch.isfinite(grad_norm)`), so wrap in a 0-d cuda tensor
            # to match the stock-path contract.
            gn = self.model.clip_grad_norm_(self.config.optimizer.max_grad_norm)
            return torch.tensor(float(gn), device=torch.cuda.current_device())
