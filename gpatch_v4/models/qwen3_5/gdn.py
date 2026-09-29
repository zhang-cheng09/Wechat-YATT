# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Headwise / chunkwise context parallelism for Qwen3.5 GDN.

Same names as Megatron ``linear_cp_mode``:
``headwise`` all-to-alls sequence onto a head shard; ``chunkwise`` keeps the
sequence shard and runs FLA ring kernels after zigzag→contiguous reshuffle.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from typing_extensions import override

from megatron.core.context_parallel_layout.conversion import convert_cp_partition_mode
from megatron.core.ssm.gated_delta_net import (
    get_parameter_local_cp,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
)
from megatron.core.ssm.gated_delta_net.common import build_cp_context
from megatron.core.ssm.gated_delta_net.common import causal_conv1d as fla_causal_conv1d
from megatron.core.ssm.gated_delta_net.common import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)

from .cp import Qwen3_5CPContext
from .modeling_qwen3_5 import Qwen3_5GatedDeltaNet, apply_mask_to_padding_states

__all__ = ["Qwen3_5CPGatedDeltaNet"]


class Qwen3_5CPGatedDeltaNet(Qwen3_5GatedDeltaNet):
    """Upstream GDN weights; Megatron headwise / chunkwise CP in ``forward``."""
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.linear_cp_comm_type = "headwise"

    @override
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        cp_context: Qwen3_5CPContext | None = None,
        **kwargs,
    ):
        if cp_context is not None:
            return self._forward_cp(
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                cp_context=cp_context,
            )
        return super().forward(
            hidden_states,
            cache_params=cache_params,
            attention_mask=attention_mask,
            **kwargs,
        )

    def _forward_cp(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None,
        attention_mask: torch.Tensor | None,
        cp_context: Qwen3_5CPContext,
    ) -> torch.Tensor:
        assert self.linear_cp_comm_type in ("headwise", "chunkwise"), (
            f"linear_cp_comm_type must be 'headwise' or 'chunkwise', "
            f"got {self.linear_cp_comm_type!r}"
        )
        if self.linear_cp_comm_type == "headwise":
            return self._forward_cp_headwise(
                hidden_states, cache_params, attention_mask, cp_context
            )
        return self._forward_cp_chunkwise(hidden_states, cache_params, attention_mask, cp_context)

    def _forward_cp_headwise(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None,
        attention_mask: torch.Tensor | None,
        cp_context: Qwen3_5CPContext,
    ) -> torch.Tensor:
        if cache_params is not None:
            raise NotImplementedError("qwen3_5 CP training does not support recurrent caches")
        if hidden_states.shape[1] != cp_context.local_sequence_length:
            raise ValueError(
                "GDN local sequence length disagrees with its CP context, "
                f"got {hidden_states.shape[1]} and {cp_context.local_sequence_length}"
            )
        cp_size = cp_context.size
        assert self.num_k_heads % cp_size == 0, (
            f"linear_num_key_heads={self.num_k_heads} must be divisible by CP size {cp_size}"
        )
        assert self.num_v_heads % cp_size == 0, (
            f"linear_num_value_heads={self.num_v_heads} must be divisible by CP size {cp_size}"
        )

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size = hidden_states.shape[0]
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        padded_cu = cp_context.global_cu_seqlens_padded
        if padded_cu is None:
            local_ranges = None
            global_ranges = None
            cu_seqlens = None
        else:
            assert batch_size == 1, "packed THD requires a batch of one row"
            local_cu = (padded_cu // cp_size).tolist()
            local_ranges = list(zip(local_cu[:-1], local_cu[1:], strict=True))
            global_cu = padded_cu.tolist()
            global_ranges = list(zip(global_cu[:-1], global_cu[1:], strict=True))
            cu_seqlens = padded_cu.to(device=hidden_states.device, dtype=torch.long)

        qkv_sections = [self.key_dim, self.key_dim, self.value_dim]

        def _to_hp(tensor: torch.Tensor, split_sections: list[int] | None = None) -> torch.Tensor:
            sbh = tensor.transpose(0, 1).contiguous()
            if local_ranges is None:
                sbh = tensor_a2a_cp2hp(
                    sbh,
                    seq_dim=0,
                    head_dim=-1,
                    cp_group=cp_context.group,
                    split_sections=split_sections,
                )
            else:
                parts = [
                    tensor_a2a_cp2hp(
                        sbh[start:end],
                        seq_dim=0,
                        head_dim=-1,
                        cp_group=cp_context.group,
                        split_sections=split_sections,
                    ) for start, end in local_ranges
                ]
                sbh = torch.cat(parts, dim=0)
            return sbh.transpose(0, 1).contiguous()

        def _to_cp(tensor: torch.Tensor) -> torch.Tensor:
            sbh = tensor.transpose(0, 1).contiguous()
            if global_ranges is None:
                sbh = tensor_a2a_hp2cp(
                    sbh,
                    seq_dim=0,
                    head_dim=-1,
                    cp_group=cp_context.group,
                )
            else:
                parts = [
                    tensor_a2a_hp2cp(
                        sbh[start:end],
                        seq_dim=0,
                        head_dim=-1,
                        cp_group=cp_context.group,
                    ) for start, end in global_ranges
                ]
                sbh = torch.cat(parts, dim=0)
            return sbh.transpose(0, 1).contiguous()

        mixed_qkv = _to_hp(mixed_qkv, qkv_sections)
        z = _to_hp(z)
        b = _to_hp(b)
        a = _to_hp(a)
        seq_len_full = mixed_qkv.shape[1]

        conv_weight = get_parameter_local_cp(
            self.conv1d.weight,
            dim=0,
            cp_group=cp_context.group,
            split_sections=qkv_sections,
        )
        mixed_qkv, _ = fla_causal_conv1d(
            x=mixed_qkv,
            weight=conv_weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            cu_seqlens=cu_seqlens,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim // cp_size, self.key_dim // cp_size, self.value_dim // cp_size],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len_full, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len_full, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len_full, -1, self.head_v_dim)
        z = z.reshape(batch_size, seq_len_full, -1, self.head_v_dim)
        a_log = get_parameter_local_cp(self.A_log, dim=0, cp_group=cp_context.group)
        dt_bias = get_parameter_local_cp(self.dt_bias, dim=0, cp_group=cp_context.group)
        beta = b.sigmoid()
        g = -a_log.float().exp() * F.softplus(a.float() + dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        core_attn_out, _ = fla_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        core_attn_out = self.norm(
            core_attn_out.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        ).reshape(batch_size, seq_len_full, -1)
        core_attn_out = _to_cp(core_attn_out)
        return self.out_proj(core_attn_out)

    def _forward_cp_chunkwise(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None,
        attention_mask: torch.Tensor | None,
        cp_context: Qwen3_5CPContext,
    ) -> torch.Tensor:
        if cache_params is not None:
            raise NotImplementedError("qwen3_5 CP training does not support recurrent caches")
        if hidden_states.shape[1] != cp_context.local_sequence_length:
            raise ValueError(
                "GDN local sequence length disagrees with its CP context, "
                f"got {hidden_states.shape[1]} and {cp_context.local_sequence_length}"
            )

        if build_cp_context is None or fla_causal_conv1d is None:
            raise ImportError("GDN chunkwise CP requires FLA build_cp_context / causal_conv1d")

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        def _remap_layout(tensor: torch.Tensor, *, to_contiguous: bool) -> torch.Tensor:
            if cp_context.size == 1:
                return tensor
            return convert_cp_partition_mode(
                tensor,
                source_partition_mode="zigzag" if to_contiguous else "contiguous",
                target_partition_mode="contiguous" if to_contiguous else "zigzag",
                seq_dim=1,
                cu_seqlens=cp_context.global_cu_seqlens_padded,
                cp_group=cp_context.group,
            )

        mixed_qkv = _remap_layout(mixed_qkv, to_contiguous=True)
        z = _remap_layout(z, to_contiguous=True)
        b = _remap_layout(b, to_contiguous=True)
        a = _remap_layout(a, to_contiguous=True)

        padded_cu = cp_context.global_cu_seqlens_padded
        if cp_context.size > 1 and padded_cu is not None:
            assert batch_size == 1, "packed THD requires a batch of one row"
            fla_cp_context = build_cp_context(
                cu_seqlens=padded_cu,
                group=cp_context.group,
                conv1d_kernel_size=self.conv_kernel_size,
            )
            conv_cu_seqlens = padded_cu
        elif cp_context.size > 1:
            fla_cp_context = build_cp_context(
                cu_seqlens=cp_context.global_segment_boundaries.to(hidden_states.device),
                group=cp_context.group,
                conv1d_kernel_size=self.conv_kernel_size,
            )
            conv_cu_seqlens = None
        else:
            fla_cp_context = None
            conv_cu_seqlens = cp_context.global_segment_boundaries.to(hidden_states.device)

        if fla_cp_context is not None and conv_cu_seqlens is None and batch_size > 1:
            conv_outputs = []
            for batch_idx in range(batch_size):
                output, _ = fla_causal_conv1d(
                    x=mixed_qkv[batch_idx:batch_idx + 1],
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    cp_context=fla_cp_context,
                    cu_seqlens=None,
                )
                conv_outputs.append(output)
            mixed_qkv = torch.cat(conv_outputs, dim=0)
        else:
            mixed_qkv, _ = fla_causal_conv1d(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                cp_context=fla_cp_context,
                cu_seqlens=conv_cu_seqlens,
            )

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        if fla_cp_context is not None and conv_cu_seqlens is None and batch_size > 1:
            outputs = []
            for batch_idx in range(batch_size):
                output, _ = fla_chunk_gated_delta_rule(
                    query[batch_idx:batch_idx + 1],
                    key[batch_idx:batch_idx + 1],
                    value[batch_idx:batch_idx + 1],
                    g=g[batch_idx:batch_idx + 1],
                    beta=beta[batch_idx:batch_idx + 1],
                    initial_state=None,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                    cp_context=fla_cp_context,
                    cu_seqlens=None,
                )
                outputs.append(output)
            core_attn_out = torch.cat(outputs, dim=0)
        else:
            core_attn_out, _ = fla_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cp_context=fla_cp_context,
                cu_seqlens=conv_cu_seqlens,
            )
        core_attn_out = self.norm(
            core_attn_out.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        ).reshape(batch_size, seq_len, -1)
        core_attn_out = _remap_layout(core_attn_out, to_contiguous=False)
        return self.out_proj(core_attn_out)
