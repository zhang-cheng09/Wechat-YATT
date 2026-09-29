# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Qwen3.8-Flash-Next vision SFT and GRPO data preparation."""
import math
from typing import Any, Dict, List, Tuple

import torch
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.extended_model.qwen4_exp import Qwen4ExpPrepareDataForwardLLM
from gpatch_v4.utils import metadata_scalar, pad_or_truncate_last_dim
from gpatch_v4.utils.dynamic_cp_utils import (
    _round_up,
    compute_dyn_cp_response_span,
    dyn_cp_schedule_default,
)

__all__ = ["Qwen4ExpPrepareDataForwardVL"]


class Qwen4ExpPrepareDataForwardVL(Qwen4ExpPrepareDataForwardLLM):
    """Adapt Qwen-VL batches to the upstream Qwen4-Exp VLM interface.

    Keep upstream's three mRoPE axes; explicit masks and CP boundaries replace
    HF's packed-sequence inference from a fourth, text-only position axis.
    """
    @override
    def sft_reroute_data_for_dynamic_cp(
        self,
        gbs_batches: List[Dict[str, Any]],
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[List[Dict[str, torch.Tensor]], int, float, float]:
        if "position_ids" not in gbs_batches[0]:
            return super().sft_reroute_data_for_dynamic_cp(
                gbs_batches,
                pad_token_id,
                pad_with_random_token=pad_with_random_token,
                **kwargs,
            )

        dp_group = mpu.get_data_parallel_group()
        tp_group = mpu.get_tensor_model_parallel_group()
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        cp_size = dp_cp_group.size() // dp_group.size()
        dist_config = self.role_config.dist_config
        pad_div = math.lcm(2 * dp_cp_group.size(), self._pad_each_doc_to_multi_of)
        has_loss_weights = any("loss_weights" in sample for sample in gbs_batches)
        if has_loss_weights and not all("loss_weights" in sample for sample in gbs_batches):
            raise ValueError("loss_weights must be present on every packed sample")

        samples = []
        for sample in gbs_batches:
            token_count = int(sample["sequence_lengths"])
            if token_count < 2:
                raise ValueError("VLM dynamic CP needs at least two tokens for SFT shifting")
            positions = torch.as_tensor(sample["position_ids"], dtype=torch.long)
            if (
                positions.ndim != 3 or positions.shape[:2] != (3, 1) or
                positions.shape[-1] < token_count
            ):
                raise ValueError(
                    "VLM dynamic CP position_ids must be [3, 1, S] covering the tokens"
                )
            tokens, labels, actual_len = self._prepare_tokens_and_labels(
                torch.as_tensor(sample["tokens"]).reshape(-1)[:token_count],
                torch.as_tensor(sample["labels"]).reshape(-1)[:token_count],
                token_count - 1,
                pad_token_id,
                vocab_size=kwargs.get("vocab_size", 0),
            )
            pad_len = (actual_len + pad_div - 1) // pad_div * pad_div
            # Token-major triples survive the scheduler's flat concatenation.
            positions = pad_or_truncate_last_dim(positions[..., :actual_len], pad_len, 0)
            vision_data = sample["vision_data"]
            routed = dict(
                tokens=pad_or_truncate_last_dim(tokens, pad_len, pad_token_id),
                labels=pad_or_truncate_last_dim(labels, pad_len, -100),
                position_ids=positions.permute(1, 2, 0).reshape(-1).contiguous(),
                original_seq_len=torch.tensor([actual_len], dtype=torch.int32),
                padded_seq_len=torch.tensor([pad_len], dtype=torch.int32),
                vision_data=vision_data.to(dtype=torch.float32).reshape(-1)
                if vision_data is not None else None,
                vision_grid_thw=sample["vision_grid_thw"].to(dtype=torch.long).reshape(-1)
                if vision_data is not None else None,
            )
            if has_loss_weights:
                weights = torch.as_tensor(sample["loss_weights"], dtype=torch.float32).reshape(-1)
                routed["loss_weights"] = pad_or_truncate_last_dim(
                    weights[1:token_count], pad_len, 0
                )
            samples.append(routed)

        packed_keys = ["tokens", "labels", "position_ids"]
        if has_loss_weights:
            packed_keys.append("loss_weights")
        cat_keys = ["vision_data", "vision_grid_thw"]
        new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum, _ = dyn_cp_schedule_default(
            samples,
            dp_group,
            tp_group,
            dp_cp_group,
            cp_size,
            dp_group.size(),
            dist_config,
            torch.cuda.current_device(),
            packed_keys=packed_keys,
            cat_keys=cat_keys,
            global_id_seqlens_keys=packed_keys + cat_keys + ["original_seq_len", "padded_seq_len"],
            dtype_map={
                "vision_data": torch.float32,
                "vision_grid_thw": torch.long
            },
            max_seqlen_per_dp_cp_rank=kwargs.get(
                "max_seqlen_per_dp_cp_rank", dist_config.max_seqlen_per_dp_cp_rank
            ),
            need_routing_info=False,
        )
        for sample in new_samples:
            if sample["vision_data"] is None or sample["vision_data"].numel() == 0:
                sample["vision_data"] = None
                sample["vision_grid_thw"] = None
                continue
            grid = sample["vision_grid_thw"].reshape(-1, 3)
            sample["vision_grid_thw"] = grid
            sample["vision_data"] = sample["vision_data"].reshape(int(grid.prod(dim=-1).sum()), -1)
        return new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum

    @override
    def sft_train_with_dynamic_cp(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
        comput_attn_mask: bool = True,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        batch, fwd_kwargs = super().sft_train_with_dynamic_cp(
            batches,
            seq_len,
            pad_token_id,
            comput_attn_mask=comput_attn_mask,
            pad_with_random_token=pad_with_random_token,
            **kwargs,
        )
        if "vision_data" in batches[0]:
            self._add_dynamic_cp_vision_inputs(batches, batch, fwd_kwargs)
        return batch, fwd_kwargs

    @override
    def rl_reroute_data_for_dynamic_cp(
        self,
        gbs_batches: List[Dict[str, Any]],
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[List[Dict[str, torch.Tensor]], int, float, float, Dict[str, Any]]:
        dp_group = mpu.get_data_parallel_group()
        tp_group = mpu.get_tensor_model_parallel_group()
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)

        dp_cp_size = dp_cp_group.size()
        tp_size = tp_group.size()
        dp_cp_pad = 2 * dp_cp_size if dp_cp_size > 1 else 1
        tp_pad = tp_size if tp_size > 1 else 1
        pad_div = math.lcm(
            dp_cp_pad * tp_pad,
            int(self._pad_each_doc_to_multi_of),
        )

        dp_size = dp_group.size()
        cp_size = dp_cp_group.size() // dp_size

        dist_config = self.role_config.dist_config
        scheduler_max_seqlen = kwargs.get(
            "max_seqlen_per_dp_cp_rank", dist_config.max_seqlen_per_dp_cp_rank
        )

        first = gbs_batches[0]
        global_retention_ratio = first.get("global_retention_ratio")
        entropy_aux_figures = first.get("entropy_aux_figures")

        # (src_key, dst_key) — keys present in the input batch that should be
        # padded to (seq_len - 1) and included in the rerouted sample / packing.
        _optional_rl_keys = [
            ("mask", "loss_mask"),
            ("advantages", "advantages"),
            ("logprobs", "prev_log_probs"),
            ("ref_logprobs", "ref_log_probs"),
            ("rollout_log_probs", "rollout_log_probs"),
            ("prev_per_token_entropy", "prev_per_token_entropy"),
            ("token_weights", "token_weights"),
        ]
        # OPD teacher logprobs: detect teacher_logprobs_* keys and unify.
        teacher_logprobs_keys = [k for k in first if k.startswith("teacher_logprobs_")]
        if teacher_logprobs_keys:
            _opd_teacher_names = [k[len("teacher_logprobs_"):] for k in teacher_logprobs_keys]
            _opd_single_teacher = len(_opd_teacher_names) == 1
            _opd_routing_field = "teacher_type"
        else:
            _opd_teacher_names = None

        rl_key_map = [(src, dst) for src, dst in _optional_rl_keys if src in first]
        has_sample_mask = first.get("sample_mask") is not None
        has_routed_experts = first.get("routed_experts") is not None

        # Shift each sample before packing (input = tokens[:-1], target =
        # tokens[1:]); a post-pack roll(-1) would leak labels across sub-sequence
        # boundaries. RL per-token fields are already (seq_len - 1) long.
        for i, batch in enumerate(gbs_batches):
            tokens = batch["tokens"]
            shifted_tokens = tokens[:-1]
            shifted_labels = tokens[1:]
            actual_len = shifted_tokens.shape[-1]
            pad_len = _round_up(actual_len, pad_div)
            prompt_len = metadata_scalar(batch, "prompt_lengths")
            sequence_len = metadata_scalar(batch, "sequence_lengths")
            response_start, response_len = compute_dyn_cp_response_span(
                prompt_len=prompt_len,
                sequence_len=sequence_len,
                actual_len=actual_len,
                # Optional: logprob-only reroute runs before create_response_mask.
                response_mask=batch.get("mask"),
            )
            positions = batch["position_ids"]
            if (
                positions.ndim != 3 or positions.shape[:2] != (3, 1) or
                positions.shape[-1] < tokens.numel()
            ):
                raise ValueError("VLM GRPO position_ids must be [3, 1, S] covering the tokens")
            positions = pad_or_truncate_last_dim(positions[..., :actual_len], pad_len, 0)
            sample = dict(
                tokens=pad_or_truncate_last_dim(shifted_tokens, pad_len, pad_token_id),
                labels=pad_or_truncate_last_dim(shifted_labels, pad_len, 0),
                position_ids=positions.permute(1, 2, 0).reshape(-1).contiguous(),
                original_seq_len=torch.tensor([actual_len], dtype=torch.int32),
                padded_seq_len=torch.tensor([pad_len], dtype=torch.int32),
                dyn_cp_response_start=torch.tensor([response_start], dtype=torch.int32),
                dyn_cp_response_length=torch.tensor([response_len], dtype=torch.int32),
            )
            pixels = batch["vision_data"]
            sample["vision_data"] = (
                pixels.to(dtype=torch.float32).reshape(-1) if pixels is not None else None
            )
            sample["vision_grid_thw"] = (
                batch["vision_grid_thw"].to(dtype=torch.long).reshape(-1)
                if pixels is not None else None
            )
            for src_key, dst_key in rl_key_map:
                value = batch[src_key]
                if src_key == "token_weights" and value.numel() == 1:
                    value = value.reshape(1).expand(actual_len)
                sample[dst_key] = pad_or_truncate_last_dim(value, pad_len, 0).to(torch.float32)
            if _opd_teacher_names is not None:
                if _opd_single_teacher:
                    tname = _opd_teacher_names[0]
                else:
                    tname = batch[_opd_routing_field]
                sample["teacher_log_probs"] = pad_or_truncate_last_dim(
                    batch[f"teacher_logprobs_{tname}"], pad_len, 0
                ).to(torch.float32)
                if "ref_log_probs" not in sample:
                    sample["ref_log_probs"] = sample["teacher_log_probs"].clone()
            if has_sample_mask:
                sm = batch["sample_mask"]
                if sm.dim() == 0:
                    sm = sm.unsqueeze(0)
                sm = sm.expand(actual_len).to(torch.float32).contiguous()
                sample["sample_mask"] = pad_or_truncate_last_dim(sm, pad_len, 0)
            if has_routed_experts:
                routed_experts = batch["routed_experts"]
                assert routed_experts.ndim == 3
                assert routed_experts.shape[0] == actual_len + 1
                routed_experts = routed_experts[:-1]
                route_index = torch.arange(
                    pad_len,
                    device=routed_experts.device,
                ) % actual_len
                padded_routes = routed_experts[route_index]
                sample["routed_experts"] = padded_routes.reshape(-1)
                sample["routed_experts_shape"] = torch.tensor(
                    [routed_experts.shape[1], routed_experts.shape[2]],
                    dtype=torch.int32,
                    device=routed_experts.device,
                )
            gbs_batches[i] = sample

        # Schedule and pack with default dynamic CP scheduler.
        dev = torch.cuda.current_device()
        packed_keys = [
            "tokens",
            "labels",
            "position_ids",
            "dyn_cp_response_start",
            "dyn_cp_response_length",
        ]
        packed_keys.extend(dst for _, dst in rl_key_map)
        if _opd_teacher_names is not None:
            packed_keys.append("teacher_log_probs")
            if "ref_log_probs" not in packed_keys:
                packed_keys.append("ref_log_probs")
        if has_sample_mask:
            packed_keys.append("sample_mask")
        cat_keys = ["vision_data", "vision_grid_thw"]
        dtype_map = {"vision_data": torch.float32, "vision_grid_thw": torch.long}
        if has_routed_experts:
            packed_keys.append("routed_experts")
            cat_keys.append("routed_experts_shape")

        global_id_seqlens_keys = (packed_keys + cat_keys + ["original_seq_len", "padded_seq_len"])
        need_routing_info = kwargs.get("need_routing_info", True)
        new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum, routing_info = (
            dyn_cp_schedule_default(
                gbs_batches,
                dp_group,
                tp_group,
                dp_cp_group,
                cp_size,
                dp_size,
                dist_config,
                dev,
                packed_keys=packed_keys,
                cat_keys=cat_keys,
                global_id_seqlens_keys=global_id_seqlens_keys,
                dtype_map=dtype_map,
                max_seqlen_per_dp_cp_rank=scheduler_max_seqlen,
                need_routing_info=need_routing_info,
            )
        )

        for sample in new_samples:
            pixels = sample["vision_data"]
            if pixels is None or pixels.numel() == 0:
                sample["vision_data"] = None
                sample["vision_grid_thw"] = None
            else:
                grid = sample["vision_grid_thw"].reshape(-1, 3)
                sample["vision_grid_thw"] = grid
                sample["vision_data"] = pixels.reshape(int(grid.prod(dim=-1).sum()), -1)

        if global_retention_ratio is not None:
            for sample in new_samples:
                sample["global_retention_ratio"] = global_retention_ratio
        if entropy_aux_figures is not None:
            for sample in new_samples:
                sample["entropy_aux_figures"] = entropy_aux_figures

        return new_samples, num_micro_batches, seqlen_sum, seqlen_sq_sum, routing_info

    @override
    def grpo_train_with_dynamic_cp(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        ppo_pack_seq: bool,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        batch, fwd_kwargs = super().grpo_train_with_dynamic_cp(
            batches,
            seqlen,
            pad_token_id,
            ppo_pack_seq,
            pad_with_random_token=pad_with_random_token,
            **kwargs,
        )
        self._add_dynamic_cp_vision_inputs(batches, batch, fwd_kwargs)
        return batch, fwd_kwargs

    def _add_dynamic_cp_vision_inputs(
        self,
        batches: List[Dict[str, Any]],
        batch: Dict[str, Any],
        fwd_kwargs: Dict[str, Any],
    ) -> None:
        source = batches[0]
        positions = source["position_ids"].reshape(-1, 3)
        lengths = source["cu_seqlens"].diff().tolist()
        starts = source["cu_seqlens_padded"][:-1].tolist()
        # Match the text parent's removal of per-document scheduler padding.
        positions = torch.cat(
            [
                positions[start:start + length]
                for start, length in zip(starts, lengths, strict=True)
            ],
            dim=0,
        ).T.unsqueeze(1)
        packed_length = batch["full_packed_seq_params"].total_seqlen
        positions = pad_or_truncate_last_dim(positions, packed_length, 0)
        cp_context = fwd_kwargs["_qwen4_exp_cp_context"]
        fwd_kwargs["position_ids"] = positions[
            ..., cp_context.local_sequence_start:cp_context.local_sequence_end].contiguous()
        self._add_vision_inputs(batches, fwd_kwargs)

    @staticmethod
    @override
    def _hf_padding_mask(
        batches: List[Dict[str, Any]],
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if "attention_mask" not in batches[0]:
            return Qwen4ExpPrepareDataForwardLLM._hf_padding_mask(batches, seq_len, device)
        return torch.stack(
            [pad_or_truncate_last_dim(sample["attention_mask"], seq_len, 0) for sample in batches]
        ).view(len(batches), -1).to(device=device, dtype=torch.long)

    @override
    def sft_train(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
        comput_attn_mask: bool = True,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if "position_ids" not in batches[0]:
            return super().sft_train(
                batches,
                seq_len,
                pad_token_id,
                comput_attn_mask=comput_attn_mask,
                pad_with_random_token=pad_with_random_token,
                **kwargs,
            )
        if self.role_config.ppo_pack_seq:
            return self._sft_train_thd(batches, seq_len, pad_token_id)

        batch, fwd_kwargs = super().sft_train(
            batches,
            seq_len,
            pad_token_id,
            comput_attn_mask=comput_attn_mask,
            pad_with_random_token=pad_with_random_token,
            **kwargs,
        )
        position_ids = []
        for sample in batches:
            positions = sample["position_ids"]
            if positions.shape[-1] < seq_len:
                raise ValueError("VLM position_ids must cover the padded sequence length")
            position_ids.append(positions[..., :seq_len])
        position_ids = torch.cat(position_ids, dim=1).cuda(non_blocking=True)
        cp_context = fwd_kwargs.get("_qwen4_exp_cp_context")
        if cp_context is not None:
            position_ids = position_ids[
                ..., cp_context.local_sequence_start:cp_context.local_sequence_end].contiguous()
        fwd_kwargs["position_ids"] = position_ids
        self._add_vision_inputs(batches, fwd_kwargs)
        return batch, fwd_kwargs

    @override
    def _sft_train_thd(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if "position_ids" not in batches[0]:
            return super()._sft_train_thd(batches, seq_len, pad_token_id)
        unpadded_batches = []
        for sample in batches:
            token_count = int(sample["sequence_lengths"])
            unpadded = dict(sample)
            for key in ("tokens", "labels", "loss_weights"):
                if key == "loss_weights" and key not in sample:
                    continue
                unpadded[key] = torch.as_tensor(sample[key]).reshape(-1)[:token_count]
            unpadded_batches.append(unpadded)
        batches = unpadded_batches
        batch, fwd_kwargs = super()._sft_train_thd(batches, seq_len, pad_token_id)

        document_positions = []
        for index, sample in enumerate(batches):
            token_count = torch.as_tensor(sample["tokens"]).numel()
            positions = torch.as_tensor(sample["position_ids"], dtype=torch.long)
            if (
                positions.ndim != 3 or positions.shape[:2] != (3, 1) or
                positions.shape[-1] < token_count
            ):
                raise ValueError(
                    "packed VLM position_ids must be [3, 1, S] with S at least the "
                    f"token length, got {tuple(positions.shape)} for sample {index}"
                )
            positions = positions[..., :token_count]
            if token_count > seq_len + 1:
                positions = positions[..., -(seq_len + 1):]
            document_positions.append(positions[:, 0, :-1].cuda(non_blocking=True))

        packed_seq_params = batch["full_packed_seq_params"]
        packed_positions = torch.cat(document_positions, dim=-1)
        packed_positions = torch.nn.functional.pad(
            packed_positions,
            (0, packed_seq_params.total_seqlen - packed_positions.shape[-1]),
        ).unsqueeze(1)
        cp_context = fwd_kwargs["_qwen4_exp_cp_context"]
        fwd_kwargs["position_ids"] = packed_positions[
            ...,
            cp_context.local_sequence_start:cp_context.local_sequence_end,
        ].contiguous()
        self._add_vision_inputs(batches, fwd_kwargs)
        return batch, fwd_kwargs

    @override
    def model_forward_only(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        fwd_kwargs = super().model_forward_only(
            batches,
            seqlen,
            pad_token_id,
            pad_with_random_token=pad_with_random_token,
            **kwargs,
        )
        self._add_rl_vision_inputs(batches, fwd_kwargs)
        return fwd_kwargs

    @override
    def grpo_train(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        ppo_pack_seq: bool,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        batch, fwd_kwargs = super().grpo_train(
            batches,
            seqlen,
            pad_token_id,
            ppo_pack_seq,
            pad_with_random_token=pad_with_random_token,
            **kwargs,
        )
        self._add_rl_vision_inputs(batches, fwd_kwargs)
        return batch, fwd_kwargs

    def _add_rl_vision_inputs(
        self,
        batches: List[Dict[str, Any]],
        fwd_kwargs: Dict[str, Any],
    ) -> None:
        cp_context = fwd_kwargs.get("_qwen4_exp_cp_context")
        position_rows = []
        for sample in batches:
            token_count = sample["tokens"].numel()
            positions = torch.as_tensor(sample["position_ids"], dtype=torch.long)
            if (
                positions.ndim != 3 or positions.shape[:2] != (3, 1) or
                positions.shape[-1] < token_count
            ):
                raise ValueError("VLM GRPO position_ids must be [3, 1, S] covering the tokens")
            if cp_context is not None:
                # THD inputs drop each document's final token before packing.
                positions = positions[..., :token_count - 1]
            else:
                positions = pad_or_truncate_last_dim(
                    positions,
                    fwd_kwargs["input_ids"].shape[-1],
                    0,
                )
            position_rows.append(positions)
        if cp_context is not None:
            positions = torch.cat(position_rows, dim=-1)
            positions = pad_or_truncate_last_dim(
                positions,
                cp_context.global_input_ids.shape[-1],
                0,
            )
            positions = positions[
                ..., cp_context.local_sequence_start:cp_context.local_sequence_end].contiguous()
        else:
            positions = torch.cat(position_rows, dim=1)
        fwd_kwargs["position_ids"] = positions.cuda(non_blocking=True)
        self._add_vision_inputs(batches, fwd_kwargs)

    def _add_vision_inputs(
        self,
        batches: List[Dict[str, Any]],
        fwd_kwargs: Dict[str, torch.Tensor],
    ) -> None:
        pixel_values = None
        image_grid_thw = None
        vision_data = []
        vision_grid_thw = []
        video_token_id = self.role_config.hf_config.video_token_id
        for sample in batches:
            if (torch.as_tensor(sample["tokens"]) == video_token_id).any():
                raise NotImplementedError("qwen4_exp vision training does not support video data")
            if sample["vision_data"] is not None:
                vision_data.append(sample["vision_data"])
                vision_grid_thw.append(sample["vision_grid_thw"])
        if vision_data:
            pixel_values = torch.cat(vision_data, dim=0).cuda(non_blocking=True)
            image_grid_thw = torch.cat(vision_grid_thw, dim=0).cuda(non_blocking=True)

        fwd_kwargs.update(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
