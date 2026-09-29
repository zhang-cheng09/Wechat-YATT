# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Qwen3.8-Flash-Next (``qwen4_exp``) engine hooks.

Overrides on top of the generic LLM path:

* :class:`Qwen4ExpPrepareDataForwardLLM` — HF **2-D padding mask** (not Megatron 4-D);
  SFT + GRPO THD pack via :func:`qwen4_exp_pack_sequences`.
* :class:`Qwen4ExpPostInitModel` — keep the Engram-owning layer out of activation
  checkpointing, which can only be done after the engine enables it.
"""
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from typing_extensions import override

from megatron.core import mpu, parallel_state

from gpatch_v4.core.constants import PrepareDataRole
from gpatch_v4.extended_model.base import PostInitModel
from gpatch_v4.extended_model.llm import PrepareDataForwardLLM
from gpatch_v4.models.qwen4_exp import (
    Qwen4ExpConfig,
    freeze_router_weights,
    set_activation_checkpointing,
)
from gpatch_v4.models.qwen4_exp.cp import (
    Qwen4ExpCPContext,
    Qwen4ExpPackedSeqParams,
    qwen4_exp_cp_chunk_data,
    qwen4_exp_pack_sequences,
)

__all__ = [
    "Qwen4ExpPostInitModel",
    "Qwen4ExpPrepareDataForwardLLM",
]


class Qwen4ExpPrepareDataForwardLLM(PrepareDataForwardLLM):
    """Prepare SFT / GRPO data with the 2-D padding mask expected by HuggingFace models."""

    cp_partition_mode = "contiguous"
    _cp_global_padding_mask: torch.Tensor
    _cp_context: Qwen4ExpCPContext | None

    def __init__(self, config, role: PrepareDataRole = PrepareDataRole.POLICY) -> None:
        super().__init__(config, role=role)
        model_config = Qwen4ExpConfig.from_pretrained(self.role_config.hf_model_path)
        self._pad_each_doc_to_multi_of = (model_config.get_text_config().indexer_compress_ratio)

    @override
    def sft_reroute_data_for_dynamic_cp(
        self,
        gbs_batches: List[Dict[str, Any]],
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ):
        kwargs["pad_to_multiple_of"] = self._pad_each_doc_to_multi_of
        return super().sft_reroute_data_for_dynamic_cp(
            gbs_batches,
            pad_token_id,
            pad_with_random_token=pad_with_random_token,
            **kwargs,
        )

    @override
    def rl_reroute_data_for_dynamic_cp(
        self,
        gbs_batches: List[Dict[str, Any]],
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ):
        kwargs["pad_to_multiple_of"] = self._pad_each_doc_to_multi_of
        return super().rl_reroute_data_for_dynamic_cp(
            gbs_batches,
            pad_token_id,
            pad_with_random_token=pad_with_random_token,
            **kwargs,
        )

    @staticmethod
    def _unpack_dynamic_cp_documents(
        batch: Dict[str, Any],
        field_names: tuple[str, ...],
    ) -> tuple[list[Dict[str, torch.Tensor]], torch.Tensor | None]:
        cu_seqlens = batch["cu_seqlens"].reshape(-1).tolist()
        cu_seqlens_padded = batch["cu_seqlens_padded"].reshape(-1).tolist()
        if len(cu_seqlens) != len(cu_seqlens_padded):
            raise ValueError("dynamic CP packed sequence boundaries must have equal lengths")

        route_values = batch.get("routed_experts")
        if route_values is not None:
            route_shapes = batch["routed_experts_shape"].reshape(-1, 2)
            if not torch.equal(route_shapes, route_shapes[:1].expand_as(route_shapes)):
                raise ValueError("dynamic CP routed_experts shapes must match across documents")
            num_layers, topk = map(int, route_shapes[0].tolist())
            route_values = route_values.view(-1, num_layers, topk)

        documents = []
        for index, (start, end, padded_start) in enumerate(
            zip(cu_seqlens[:-1], cu_seqlens[1:], cu_seqlens_padded[:-1], strict=True)
        ):
            length = end - start
            if length <= 0:
                raise ValueError(f"dynamic CP document {index} has non-positive length {length}")
            padded_end = cu_seqlens_padded[index + 1]
            if padded_start + length > padded_end:
                raise ValueError(f"dynamic CP document {index} exceeds its padded extent")
            document = {
                "tokens": batch["tokens"].reshape(-1)[padded_start:padded_start + length],
                "labels": batch["labels"].reshape(-1)[padded_start:padded_start + length],
            }
            for field_name in field_names:
                if field_name in batch:
                    document[field_name] = batch[field_name].reshape(-1)[padded_start:padded_start +
                                                                         length]
            if route_values is not None:
                document["routed_experts"] = route_values[padded_start:padded_start + length]
            documents.append(document)
        return documents, route_values

    def _dynamic_cp_group(self, batch: Dict[str, Any]):
        local_cp_size = int(torch.as_tensor(batch["local_cp_size"]).item())
        return parallel_state.get_dynamic_data_context_parallel_groups(group_size=local_cp_size)

    def _pack_dynamic_cp_documents(
        self,
        documents: list[Dict[str, torch.Tensor]],
        cp_size: int,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Qwen4ExpPackedSeqParams]:
        return qwen4_exp_pack_sequences(
            [document["tokens"] for document in documents],
            [document["labels"] for document in documents],
            cp_size=cp_size,
            pad_multiple=self._pad_each_doc_to_multi_of,
            pad_token_id=pad_token_id,
        )

    def _dynamic_cp_context(
        self,
        packed_ids: torch.Tensor,
        packed_labels: torch.Tensor,
        packed_positions: torch.Tensor,
        packed_seq_params: Qwen4ExpPackedSeqParams,
        cp_group,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Qwen4ExpCPContext]:
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()
        global_padding_mask = torch.arange(
            packed_seq_params.total_seqlen,
            device=packed_ids.device,
        ).unsqueeze(0) >= int(packed_seq_params.cu_seqlens_q[-1])
        return qwen4_exp_cp_chunk_data(
            cp_rank,
            cp_size,
            cp_group,
            tokens=packed_ids,
            labels=packed_labels,
            loss_mask=(packed_labels != -100).float(),
            position_ids=packed_positions,
            global_padding_mask=global_padding_mask,
            global_cu_seqlens=packed_seq_params.cu_seqlens_q,
        )

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
        del seq_len, comput_attn_mask, pad_with_random_token, kwargs
        if len(batches) != 1:
            raise ValueError("qwen4_exp dynamic CP expects exactly one packed microbatch")
        source_batch = batches[0]
        for key, value in list(source_batch.items()):
            if isinstance(value, torch.Tensor) and not value.is_cuda:
                source_batch[key] = value.cuda(non_blocking=True)
        cp_group = self._dynamic_cp_group(source_batch)
        documents, _ = self._unpack_dynamic_cp_documents(source_batch, ("loss_weights", ))
        packed_ids, packed_positions, packed_labels, packed_seq_params = self._pack_dynamic_cp_documents(
            documents,
            cp_group.size(),
            pad_token_id,
        )
        local_ids, local_labels, local_loss_mask, local_positions, cp_context = (
            self._dynamic_cp_context(
                packed_ids,
                packed_labels,
                packed_positions,
                packed_seq_params,
                cp_group,
            )
        )
        batch = {
            "labels": local_labels,
            "loss_mask": local_loss_mask,
            "full_labels": packed_labels,
            "full_loss_mask": (packed_labels != -100).float(),
            "full_packed_seq_params": packed_seq_params,
            "cp_group": cp_group,
        }
        if "loss_weights" in documents[0]:
            packed_loss_weights = torch.zeros(
                (1, packed_seq_params.total_seqlen),
                dtype=torch.float32,
                device=packed_ids.device,
            )
            for start, document in zip(
                packed_seq_params.cu_seqlens_q[:-1].tolist(),
                documents,
                strict=True,
            ):
                weights = document["loss_weights"].to(dtype=torch.float32)
                packed_loss_weights[0, start:start + weights.numel()] = weights
            batch["loss_weights"] = packed_loss_weights[:,
                                                        cp_context.local_sequence_start:cp_context.
                                                        local_sequence_end].contiguous()
        fwd_kwargs = {
            "input_ids": local_ids,
            "position_ids": local_positions,
            "attention_mask": cp_context.local_attention_mask.to(torch.long),
            "labels": None,
            "_qwen4_exp_cp_context": cp_context,
        }
        return batch, fwd_kwargs

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
        del seqlen, ppo_pack_seq, pad_with_random_token, kwargs
        if len(batches) != 1:
            raise ValueError("qwen4_exp dynamic CP expects exactly one packed microbatch")
        source_batch = batches[0]
        for key, value in list(source_batch.items()):
            if isinstance(value, torch.Tensor) and not value.is_cuda:
                source_batch[key] = value.cuda(non_blocking=True)
        cp_group = self._dynamic_cp_group(source_batch)
        field_names = (
            "loss_mask",
            "advantages",
            "prev_log_probs",
            "ref_log_probs",
            "rollout_log_probs",
            "teacher_log_probs",
            "prev_per_token_entropy",
            "sample_mask",
            "token_weights",
        )
        documents, _ = self._unpack_dynamic_cp_documents(source_batch, field_names)
        packed_ids, packed_positions, packed_labels, packed_seq_params = self._pack_dynamic_cp_documents(
            documents,
            cp_group.size(),
            pad_token_id,
        )
        local_ids, local_labels, _, local_positions, cp_context = self._dynamic_cp_context(
            packed_ids,
            packed_labels,
            packed_positions,
            packed_seq_params,
            cp_group,
        )
        local_target = local_labels.masked_fill(local_labels == -100, 0)
        batch: Dict[str, Any] = {
            "target": local_target,
            "cu_seqlens": packed_seq_params.cu_seqlens_q,
            "cu_seqlens_padded": packed_seq_params.cu_seqlens_q_padded,
            "dyn_cp_response_start": source_batch["dyn_cp_response_start"].reshape(-1),
            "dyn_cp_response_length": source_batch["dyn_cp_response_length"].reshape(-1),
            "local_cp_size": source_batch["local_cp_size"],
            "full_packed_seq_params": packed_seq_params,
            "cp_group": cp_group,
        }
        for field_name in field_names:
            if field_name not in documents[0]:
                continue
            packed_field = torch.zeros(
                (1, packed_seq_params.total_seqlen),
                dtype=documents[0][field_name].dtype,
                device=packed_ids.device,
            )
            for start, document in zip(
                packed_seq_params.cu_seqlens_q[:-1].tolist(),
                documents,
                strict=True,
            ):
                values = document[field_name]
                packed_field[0, start:start + values.numel()] = values
            batch["mask" if field_name == "loss_mask" else field_name] = packed_field
        if "routed_experts" in documents[0]:
            packed_routes = torch.cat(
                [document["routed_experts"] for document in documents],
                dim=0,
            )
            padding = packed_seq_params.total_seqlen - packed_routes.shape[0]
            if padding:
                packed_routes = torch.cat(
                    (
                        packed_routes,
                        packed_routes[torch.arange(padding, device=packed_routes.device) %
                                      packed_routes.shape[0]]
                    ),
                    dim=0,
                )
            batch["routed_experts"] = packed_routes[cp_context.local_sequence_start:cp_context.
                                                    local_sequence_end].contiguous()
        for field_name in ("global_retention_ratio", "entropy_aux_figures"):
            if field_name in source_batch:
                batch[field_name] = source_batch[field_name]
        fwd_kwargs = {
            "input_ids": local_ids,
            "position_ids": local_positions,
            "attention_mask": cp_context.local_attention_mask.to(torch.long),
            "labels": None,
            "_qwen4_exp_cp_context": cp_context,
        }
        return batch, fwd_kwargs

    @override
    def _sft_train_cp_chunk_data(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        if attention_mask is not None:
            raise RuntimeError("qwen4_exp CP data prep unexpectedly received a 4-D mask")
        cp_group = mpu.get_context_parallel_group()
        cp_size = dist.get_world_size(cp_group)
        cp_rank = mpu.get_context_parallel_rank()
        tokens, labels, loss_mask, position_ids, self._cp_context = (
            qwen4_exp_cp_chunk_data(
                cp_rank,
                cp_size,
                cp_group,
                tokens=tokens,
                labels=labels,
                loss_mask=loss_mask,
                position_ids=position_ids,
                global_padding_mask=self._cp_global_padding_mask,
            )
        )
        return tokens, labels, loss_mask, position_ids, None

    @override
    def rl_train_cp_chunk_single_data(self, data: torch.Tensor) -> torch.Tensor:
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size <= 1:
            return data
        if data.shape[1] % cp_size != 0:
            raise ValueError(
                f"qwen4_exp contiguous CP requires seq_len divisible by cp_size, "
                f"got shape={tuple(data.shape)} cp_size={cp_size}"
            )
        local_len = data.shape[1] // cp_size
        start = mpu.get_context_parallel_rank() * local_len
        return data[:, start:start + local_len].contiguous()

    @override
    def model_forward_only(
        self,
        batches: List[Dict[str, Any]],
        seqlen: int,
        pad_token_id: int,
        pad_with_random_token: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        if not self.role_config.ppo_pack_seq:
            cp_size = mpu.get_context_parallel_world_size()
            if cp_size > 1:
                raise RuntimeError("qwen4_exp RL with CP>1 requires policy.ppo_pack_seq=True")
            model_fwd_args = super().model_forward_only(
                batches,
                seqlen,
                pad_token_id,
                pad_with_random_token=pad_with_random_token,
                **kwargs,
            )
            model_fwd_args["attention_mask"] = self._hf_padding_mask_from_sequence_lengths(
                batches, seqlen, model_fwd_args["input_ids"].device
            )
            return model_fwd_args

        batch, fwd_kwargs = self._grpo_train_thd(
            batches,
            seqlen,
            pad_token_id,
            include_rl_fields=False,
            pad_with_random_token=pad_with_random_token,
            vocab_size=kwargs.get("vocab_size", 0),
        )
        fwd_kwargs["target"] = batch["target"]
        fwd_kwargs["full_packed_seq_params"] = batch["full_packed_seq_params"]
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
        if ppo_pack_seq:
            if not self.role_config.ppo_pack_seq:
                raise RuntimeError("grpo_train pack path requires policy.ppo_pack_seq=True")
            return self._grpo_train_thd(
                batches,
                seqlen,
                pad_token_id,
                include_rl_fields=True,
                pad_with_random_token=pad_with_random_token,
                vocab_size=kwargs.get("vocab_size", 0),
            )

        cp_size = mpu.get_context_parallel_world_size()
        if cp_size > 1:
            raise RuntimeError(
                "qwen4_exp GRPO with CP>1 requires policy.ppo_pack_seq=True "
                "(contiguous THD packing)"
            )
        batch, fwd_kwargs = super().grpo_train(
            batches,
            seqlen,
            pad_token_id,
            ppo_pack_seq=False,
            pad_with_random_token=pad_with_random_token,
            **kwargs,
        )
        fwd_kwargs["attention_mask"] = self._hf_padding_mask_from_sequence_lengths(
            batches, seqlen, fwd_kwargs["input_ids"].device
        )
        return batch, fwd_kwargs

    def _grpo_train_thd(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
        *,
        include_rl_fields: bool,
        pad_with_random_token: bool = False,
        vocab_size: int = 0,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Pack pre-shifted GRPO tensors into one Qwen THD row.

        Documents are contiguous; only a trailing CP/QSA alignment pad is appended
        (unlike DSV4 per-document padding). RL fields stay on the global packed axis.
        """
        del pad_with_random_token, vocab_size  # reserved; random pad not used yet
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
        cp_group = mpu.get_context_parallel_group()

        ids_list: list[torch.Tensor] = []
        targets_list: list[torch.Tensor] = []
        actual_lens: list[int] = []
        for sample_idx, batch in enumerate(batches):
            tokens = batch["tokens"]
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.reshape(-1)
            current_seq_len = int(batch["sequence_lengths"].item())
            if current_seq_len != tokens.numel():
                raise ValueError(
                    f"sample {sample_idx}: sequence_lengths={current_seq_len} != "
                    f"tokens.numel()={tokens.numel()}"
                )
            if current_seq_len > seq_len:
                raise ValueError(
                    f"sample {sample_idx} sequence length {current_seq_len} exceeds "
                    f"seq_len={seq_len}"
                )
            if current_seq_len < 2:
                raise ValueError(f"sample {sample_idx} needs at least two tokens for GRPO")

            shifted_input = tokens[:-1].cuda(non_blocking=True)
            shifted_target = tokens[1:].cuda(non_blocking=True)
            ids_list.append(shifted_input)
            targets_list.append(shifted_target)
            actual_lens.append(shifted_input.numel())

        packed_ids, packed_pos, packed_labels, packed_seq_params = (
            qwen4_exp_pack_sequences(
                ids_list,
                targets_list,
                cp_size=cp_size,
                pad_multiple=self._pad_each_doc_to_multi_of,
                pad_token_id=pad_token_id,
            )
        )
        if packed_seq_params.total_seqlen > seq_len:
            raise ValueError(
                f"packed row length {packed_seq_params.total_seqlen} exceeds seq_len={seq_len}"
            )

        packed_target = packed_labels.clone()
        packed_target = packed_target.masked_fill(packed_labels == -100, 0)

        rl_key_map = (
            ("advantages", "advantages"),
            ("mask", "mask"),
            ("logprobs", "prev_log_probs"),
            ("ref_logprobs", "ref_log_probs"),
            ("rollout_log_probs", "rollout_log_probs"),
            ("token_weights", "token_weights"),
        )
        present_rl_keys = (
            [(src, dst)
             for src, dst in rl_key_map if src in batches[0]] if include_rl_fields else []
        )
        if include_rl_fields:
            for required in ("advantages", "mask", "logprobs"):
                if required not in batches[0]:
                    raise ValueError(f"THD GRPO batch is missing {required}")

        packed_rl = {
            dst:
                torch.zeros(
                    (1, packed_seq_params.total_seqlen),
                    dtype=torch.float32,
                    device=packed_ids.device,
                )
            for _, dst in present_rl_keys
        }
        cu = packed_seq_params.cu_seqlens_q
        for sample_idx, (batch, actual_len) in enumerate(zip(batches, actual_lens, strict=True)):
            offset = int(cu[sample_idx].item())
            for src, dst in present_rl_keys:
                if src not in batch:
                    raise ValueError(f"sample {sample_idx} is missing {src}")
                value = batch[src]
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value)
                if value.dim() > 1:
                    raise ValueError(
                        f"sample {sample_idx} {src} has shape {tuple(value.shape)}; "
                        "qwen4_exp THD GRPO only supports 1D token-level fields"
                    )
                value = value.reshape(-1)
                if src == "rollout_log_probs" and value.numel() == actual_len + 1:
                    value = value[:actual_len]
                if src == "token_weights" and value.numel() == 1:
                    value = value.expand(actual_len)
                if value.numel() != actual_len:
                    raise ValueError(
                        f"sample {sample_idx} {src} length {value.numel()} "
                        f"!= shifted token length {actual_len}"
                    )
                packed_rl[dst][0, offset:offset + actual_len] = value.to(
                    device=packed_ids.device, dtype=torch.float32
                )

        global_padding_mask = torch.arange(
            packed_seq_params.total_seqlen, device=packed_ids.device
        ).unsqueeze(0) >= int(packed_seq_params.cu_seqlens_q[-1])
        dummy_loss_mask = (~global_padding_mask).float()
        local_ids, _, _, local_positions, cp_context = qwen4_exp_cp_chunk_data(
            cp_rank,
            cp_size,
            cp_group,
            tokens=packed_ids,
            labels=packed_target,
            loss_mask=dummy_loss_mask,
            position_ids=packed_pos,
            global_padding_mask=global_padding_mask,
            global_cu_seqlens=packed_seq_params.cu_seqlens_q,
        )

        batch_out: Dict[str, Any] = {
            "target": packed_target,
            "full_packed_seq_params": packed_seq_params,
            "cu_seqlens": packed_seq_params.cu_seqlens_q,
            "cu_seqlens_padded": packed_seq_params.cu_seqlens_q_padded,
            **packed_rl,
        }
        if include_rl_fields and "sample_mask" in batches[0]:
            assert all("sample_mask" in batch for batch in batches)
            sample_masks = [torch.as_tensor(batch["sample_mask"]).reshape(-1) for batch in batches]
            assert all(sample_mask.numel() == 1 for sample_mask in sample_masks)
            batch_out["sample_mask"] = torch.cat(sample_masks).to(
                device=packed_ids.device,
                dtype=torch.float32,
            )
        if include_rl_fields and "entropy_aux_figures" in batches[0]:
            batch_out["entropy_aux_figures"] = torch.as_tensor(batches[0]["entropy_aux_figures"]
                                                              ).to(device=packed_ids.device)

        fwd_kwargs: Dict[str, Any] = {
            "input_ids": local_ids,
            "position_ids": local_positions,
            "attention_mask": cp_context.local_attention_mask.to(torch.long),
            "labels": None,
            "_qwen4_exp_cp_context": cp_context,
        }
        return batch_out, fwd_kwargs

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
        if self.role_config.ppo_pack_seq:
            return self._sft_train_thd(batches, seq_len, pad_token_id)

        cp_size = mpu.get_context_parallel_world_size()
        global_loss_weights = None
        if cp_size > 1 and any("loss_weights" in sample for sample in batches):
            if not all("loss_weights" in sample for sample in batches):
                raise ValueError("loss_weights must be present on every sample in a batch")
            global_loss_weights = torch.stack(
                [self.prepare_loss_weights(sample["loss_weights"], seq_len) for sample in batches]
            ).view(len(batches), -1).cuda(non_blocking=True)

        if cp_size > 1:
            device = torch.device("cuda", torch.cuda.current_device())
            global_attention_mask = self._hf_padding_mask(batches, seq_len, device).bool()
            self._cp_global_padding_mask = global_attention_mask.logical_not()
            self._cp_context = None
        try:
            batch, fwd_kwargs = super().sft_train(
                batches,
                seq_len,
                pad_token_id,
                # The generic 4-D Megatron mask has opposite polarity and is replaced below.
                comput_attn_mask=False,
                pad_with_random_token=pad_with_random_token,
                **kwargs,
            )
            if cp_size > 1:
                cp_context = self._cp_context
                if cp_context is None:
                    raise RuntimeError("qwen4_exp CP data prep did not build its batch context")
        finally:
            if cp_size > 1:
                del self._cp_global_padding_mask
                del self._cp_context

        if cp_size == 1:
            fwd_kwargs["attention_mask"] = self._hf_padding_mask(
                batches, seq_len, fwd_kwargs["input_ids"].device
            )
        else:
            fwd_kwargs["attention_mask"] = cp_context.local_attention_mask.to(torch.long)
            fwd_kwargs["_qwen4_exp_cp_context"] = cp_context
            if global_loss_weights is not None:
                batch["loss_weights"] = global_loss_weights[:, cp_context.
                                                            local_sequence_start:cp_context.
                                                            local_sequence_end].contiguous()
        return batch, fwd_kwargs

    def _sft_train_thd(
        self,
        batches: List[Dict[str, Any]],
        seq_len: int,
        pad_token_id: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Pack shifted SFT documents into one boundary-aware THD row."""
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
        cp_group = mpu.get_context_parallel_group()

        document_ids = []
        document_labels = []
        document_loss_weights = []
        has_loss_weights = any("loss_weights" in sample for sample in batches)
        if has_loss_weights and not all("loss_weights" in sample for sample in batches):
            raise ValueError("loss_weights must be present on every packed sample")

        for index, sample in enumerate(batches):
            tokens = torch.as_tensor(sample["tokens"], dtype=torch.long).reshape(-1)
            labels = torch.as_tensor(sample["labels"], dtype=torch.long).reshape(-1)
            if tokens.shape != labels.shape:
                raise ValueError(
                    f"sample {index} token/label shapes differ: {tokens.shape} and {labels.shape}"
                )
            if not torch.equal(tokens == labels, labels >= 0):
                raise ValueError(f"sample {index} labels must be unshifted token IDs or -100")
            if tokens.numel() > seq_len + 1:
                tokens = tokens[-(seq_len + 1):]
                labels = labels[-(seq_len + 1):]
            if tokens.numel() < 2:
                raise ValueError(f"sample {index} needs at least two tokens for SFT shifting")

            document_ids.append(tokens[:-1].cuda(non_blocking=True))
            document_labels.append(labels[1:].cuda(non_blocking=True))
            if has_loss_weights:
                loss_weights = torch.as_tensor(sample["loss_weights"],
                                               dtype=torch.float32).reshape(-1)
                if loss_weights.numel() != torch.as_tensor(sample["tokens"]).numel():
                    raise ValueError(f"sample {index} loss_weights must match its unshifted tokens")
                if loss_weights.numel() > seq_len + 1:
                    loss_weights = loss_weights[-(seq_len + 1):]
                document_loss_weights.append(loss_weights[1:].cuda(non_blocking=True))

        packed_ids, packed_positions, packed_labels, packed_seq_params = (
            qwen4_exp_pack_sequences(
                document_ids,
                document_labels,
                cp_size=cp_size,
                pad_multiple=self._pad_each_doc_to_multi_of,
                pad_token_id=pad_token_id,
            )
        )
        if packed_seq_params.total_seqlen > seq_len:
            raise ValueError(
                f"packed row length {packed_seq_params.total_seqlen} exceeds seq_len={seq_len}"
            )

        global_padding_mask = torch.arange(
            packed_seq_params.total_seqlen, device=packed_ids.device
        ).unsqueeze(0) >= int(packed_seq_params.cu_seqlens_q[-1])
        loss_mask = (packed_labels != -100).float()
        local_ids, local_labels, local_loss_mask, local_positions, cp_context = (
            qwen4_exp_cp_chunk_data(
                cp_rank,
                cp_size,
                cp_group,
                tokens=packed_ids,
                labels=packed_labels,
                loss_mask=loss_mask,
                position_ids=packed_positions,
                global_padding_mask=global_padding_mask,
                global_cu_seqlens=packed_seq_params.cu_seqlens_q,
            )
        )

        local_loss_weights = None
        if document_loss_weights:
            packed_loss_weights = torch.zeros(
                (1, packed_seq_params.total_seqlen),
                dtype=torch.float32,
                device=packed_ids.device,
            )
            for start, weights in zip(
                packed_seq_params.cu_seqlens_q[:-1].tolist(),
                document_loss_weights,
            ):
                packed_loss_weights[0, start:start + weights.numel()] = weights
            local_loss_weights = packed_loss_weights[:, cp_context.local_sequence_start:cp_context.
                                                     local_sequence_end].contiguous()

        batch = {
            "labels": local_labels,
            "loss_mask": local_loss_mask,
            "loss_weights": local_loss_weights,
            "full_labels": packed_labels,
            "full_loss_mask": loss_mask,
            "full_packed_seq_params": packed_seq_params,
        }
        fwd_kwargs = {
            "input_ids": local_ids,
            "position_ids": local_positions,
            "attention_mask": cp_context.local_attention_mask.to(torch.long),
            "labels": None,
            "_qwen4_exp_cp_context": cp_context,
        }
        return batch, fwd_kwargs

    @staticmethod
    def _hf_padding_mask(
        batches: List[Dict[str, Any]],
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """``[batch, seq_len]`` int mask, ``1`` for real tokens and ``0`` for right padding.

        Lengths come from the unpadded ``tokens`` entries, so this stays correct however
        the base class chose to pad.
        """
        mask = torch.zeros(len(batches), seq_len, dtype=torch.long, device=device)
        for index, sample in enumerate(batches):
            length = min(int(sample["tokens"].shape[0]), seq_len)
            mask[index, :length] = 1
        return mask

    @staticmethod
    def _hf_padding_mask_from_sequence_lengths(
        batches: List[Dict[str, Any]],
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """HF 2-D padding mask for RL batches that carry ``sequence_lengths``."""
        mask = torch.zeros(len(batches), seq_len, dtype=torch.long, device=device)
        for index, sample in enumerate(batches):
            length = min(int(sample["sequence_lengths"].item()), seq_len)
            mask[index, :length] = 1
        return mask


class Qwen4ExpPostInitModel(PostInitModel):
    """Keep Engram out of recompute, which would replay its owner-sharded collectives."""
    @override
    def __call__(self, model) -> None:
        if self.config.training.freeze_moe_router:
            freeze_router_weights(model)
        if model.mtp is not None:
            model.mtp.requires_grad_(self.config.training.online_mtp_sft)
        set_activation_checkpointing(
            model,
            enabled=bool(self.config.training.recompute),
            selective_recompute=bool(self.config.training.selective_recompute),
            selective_recompute_max_layers=self.config.training.selective_recompute_max_layers,
        )
