# coding=utf-8
"""Qwen3.5 dense VL HP builder."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import Module

from gpatch_v4.models.qwen3_5 import apply_hp as apply_hp_qwen3_5
from gpatch_v4.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from gpatch_v4.training_backend.fsdp2_backend.hp_builder.base import HpModelBuilder
from gpatch_v4.utils import log


class Qwen3_5HpBuilder(HpModelBuilder):
    def validate(self, engine: Any) -> None:
        if engine.training_config.enable_mtp:
            raise NotImplementedError(
                "qwen3_5 FSDP2 does not load the MTP head; set enable_mtp=False"
            )
        if engine.training_config.enable_dspark:
            raise NotImplementedError("DSpark is only implemented for DeepSeek-V4")
        if engine.policy_config.dist_config.dynamic_context_parallel:
            if engine.policy_config.dist_config.context_parallel_size != 1:
                raise NotImplementedError(
                    "qwen3_5 FSDP2 dynamic CP requires context_parallel_size=1"
                )
        if engine.policy_config.dist_config.expert_model_parallel_size != 1:
            raise NotImplementedError("qwen3_5 dense requires expert_model_parallel_size=1")
        cp_size = engine.policy_config.dist_config.context_parallel_size
        if cp_size > 1 or engine.policy_config.dist_config.dynamic_context_parallel:
            attn_impl = engine.policy_config.attn_implementation
            if attn_impl == "eager":
                raise NotImplementedError(
                    "qwen3_5 CP>1 requires a TE-capable attn_implementation "
                    f"(got {attn_impl!r})"
                )
        unsupported_flags = [
            name for name, enabled in (
                ("fp8_qat", engine.policy_config.fp8_qat),
                ("fp4_qat", engine.policy_config.fp4_qat),
                ("fp4_qat_indexer", engine.policy_config.fp4_qat_indexer),
                ("fp8", engine.policy_config.fp8),
                ("fsdp_fp8_gather", engine.policy_config.fsdp_fp8_gather),
                (
                    "moe_router_force_load_balancing",
                    engine.policy_config.moe_router_force_load_balancing,
                ),
            ) if enabled
        ]
        if unsupported_flags:
            raise NotImplementedError(f"{unsupported_flags} are not implemented for qwen3_5 FSDP2")

    def prepare_config(self, model_cls: type, hf_model_path: str, engine: Any) -> Any:
        del model_cls
        cfg = Qwen3_5Config.from_pretrained(hf_model_path)
        cfg.use_cache = False
        cfg.text_config.use_cache = False
        # HF checkpoints advertise bf16; apply_hp requires fp32 master weights.
        cfg.text_config.dtype = torch.float32
        cfg.vision_config.dtype = torch.float32

        n_layers_dbg = engine.config.debug.debug_truncate_num_hidden_layers
        if n_layers_dbg is not None:
            log(
                f"DEBUG: truncating qwen3_5 from num_hidden_layers="
                f"{cfg.text_config.num_hidden_layers} -> {n_layers_dbg}",
                rank=0,
            )
            cfg.text_config.num_hidden_layers = n_layers_dbg
            cfg.text_config.layer_types = list(cfg.text_config.layer_types[:n_layers_dbg])
        return cfg

    def apply_hp(self, model: Module, engine: Any) -> Module:
        return apply_hp_qwen3_5(
            model,
            engine.ep_2d_mesh,
            cp_mesh=engine.cp_mesh_for_hp,
            attn_backend=engine.policy_config.attn_implementation,
            linear_cp_mode=engine.policy_config.fsdp_linear_cp_mode,
        )
