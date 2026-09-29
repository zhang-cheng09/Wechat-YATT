# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""TE zigzag / THD attention for Qwen3.5."""
from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from typing_extensions import override

from .cp import Qwen3_5CPContext
from .modeling_qwen3_5 import (
    Qwen3_5Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

__all__ = ["Qwen3_5CPAttention"]


class Qwen3_5CPAttention(Qwen3_5Attention):
    """Upstream QKV / RoPE / gated output, TE DotProductAttention for the core."""

    _cp_stream: torch.cuda.Stream | None = None

    def bind_te(self) -> None:
        if getattr(self, "te_attn", None) is not None:
            return
        self.te_attn = te.DotProductAttention(
            num_attention_heads=self.config.num_attention_heads,
            kv_channels=self.head_dim,
            attention_dropout=self.attention_dropout,
            attn_mask_type="causal",
            sequence_parallel=False,
            tp_size=1,
            layer_number=self.layer_idx + 1,
            num_gqa_groups=self.config.num_key_value_heads,
            attention_type="self",
            softmax_scale=self.scaling,
        )

    def bind_cp(self, cp_group: dist.ProcessGroup | None, cp_size: int) -> None:
        self.bind_te()
        if Qwen3_5CPAttention._cp_stream is None:
            Qwen3_5CPAttention._cp_stream = torch.cuda.Stream()
        if cp_size <= 1 or cp_group is None:
            self.te_attn.set_context_parallel_group(
                None,
                [],
                Qwen3_5CPAttention._cp_stream,
                "p2p",
            )
            return
        self.te_attn.set_context_parallel_group(
            cp_group,
            dist.get_process_group_ranks(cp_group),
            Qwen3_5CPAttention._cp_stream,
            "p2p",
        )

    def _te_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cp_context: Qwen3_5CPContext,
    ) -> torch.Tensor:
        """RoPE-applied ``[B, S, H, D]`` QKV → TE → ``[B, S, H, D]``."""
        if getattr(self, "te_attn", None) is None:
            raise RuntimeError("qwen3_5 TE attention is not bound")
        if query_states.shape[1] != cp_context.local_sequence_length:
            raise ValueError(
                "attention local sequence length disagrees with its CP context, "
                f"got {query_states.shape[1]} and {cp_context.local_sequence_length}"
            )
        if query_states.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "TE attention requires fp16 or bf16 activations, "
                f"got {query_states.dtype}"
            )

        if cp_context.global_cu_seqlens_padded is None:
            return self.te_attn(
                query_states,
                key_states,
                value_states,
                None,
                attn_mask_type="causal",
                qkv_format="bshd",
            )

        if query_states.shape[0] != 1:
            raise ValueError(
                "TE THD attention requires batch size 1, "
                f"got {query_states.shape[0]}"
            )
        query = query_states.squeeze(0)
        key = key_states.squeeze(0)
        value = value_states.squeeze(0)
        cu = cp_context.global_cu_seqlens_padded.to(device=query.device, dtype=torch.int32)
        max_seqlen = int(cu.diff().max())
        output = self.te_attn(
            query,
            key,
            value,
            None,
            attn_mask_type="padding_causal",
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )
        return output.unsqueeze(0)

    @override
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cp_context: Optional[Qwen3_5CPContext] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if past_key_values is not None and cp_context is not None:
            raise NotImplementedError("qwen3_5 CP training does not support a KV cache")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        cos, sin = position_embeddings

        if cp_context is not None:
            if hidden_states.shape[1] != cp_context.local_sequence_length:
                raise ValueError(
                    "attention local sequence length disagrees with its CP context, "
                    f"got {hidden_states.shape[1]} and {cp_context.local_sequence_length}"
                )
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=2
            )
            attn_output = self._te_attention(query_states, key_states, value_states, cp_context)
        else:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx
                )
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), None
