# coding=utf-8
"""Qwen4-Exp HP builder."""

from __future__ import annotations

from typing import Any

from torch.nn import Module

from gpatch_v4.training_backend.fsdp2_backend.hp_builder.base import HpModelBuilder
from gpatch_v4.utils import log

_QWEN4_EXP_IMPORT_ERROR = None
try:
    from gpatch_v4.models.qwen4_exp import apply_hp as apply_hp_qwen4_exp
except Exception as error:
    apply_hp_qwen4_exp = None
    _QWEN4_EXP_IMPORT_ERROR = error


class Qwen4ExpHpBuilder(HpModelBuilder):
    def validate(self, engine: Any) -> None:
        if apply_hp_qwen4_exp is None:
            raise ImportError(
                "model_arch=qwen4_exp requires gpatch_v4.models.qwen4_exp"
            ) from _QWEN4_EXP_IMPORT_ERROR
        if engine.training_config.online_mtp_sft:
            raise NotImplementedError(
                "qwen4_exp online MTP training is not supported; use "
                "enable_mtp=True with online_mtp_sft=False for load-only MTP"
            )
        if engine.training_config.enable_dspark:
            raise NotImplementedError("DSpark is only implemented for DeepSeek-V4")
        if engine.policy_config.indexer_backend != "eager":
            raise NotImplementedError(
                "qwen4_exp has no separate indexer kernel backend; use "
                "indexer_backend='eager'"
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
            raise NotImplementedError(
                f"{unsupported_flags} are DeepSeek-V4 features and are not implemented "
                "for qwen4_exp"
            )

    def prepare_config(
        self,
        model_cls: type,
        hf_model_path: str,
        engine: Any,
        model_only_inference: bool,
    ) -> Any:
        cfg = model_cls.config_class.from_pretrained(hf_model_path)
        # The released config enables a DynamicCache, which SFT never consumes.
        text_config = cfg.get_text_config()
        text_config.use_cache = False

        # model_only_inference 控制是否只 reference_model, 是的则一直不加载 mtp
        load_mtp = engine.training_config.enable_mtp and not model_only_inference
        if load_mtp:
            if text_config.mtp_num_hidden_layers != 1:
                raise ValueError(
                    "qwen4_exp load-only MTP requires mtp_num_hidden_layers=1, got "
                    f"{text_config.mtp_num_hidden_layers}"
                )
            if text_config.mtp_use_dedicated_embeddings:
                raise NotImplementedError("qwen4_exp dedicated MTP embeddings are not supported")
            if text_config.hc_count != 4:
                raise ValueError(
                    f"qwen4_exp load-only MTP requires hc_count=4, got {text_config.hc_count}"
                )
            if text_config.mtp is None or text_config.mtp.get("layer_types") != ["full_attention"]:
                raise ValueError(
                    "qwen4_exp load-only MTP requires mtp.layer_types=['full_attention']"
                )
        else:
            text_config.mtp_num_hidden_layers = 0
            text_config.mtp = None

        n_layers_dbg = engine.config.debug.debug_truncate_num_hidden_layers
        if n_layers_dbg is not None:
            log(
                f"DEBUG: truncating qwen4_exp from num_hidden_layers="
                f"{text_config.num_hidden_layers} -> {n_layers_dbg}",
                rank=0,
            )
            text_config.num_hidden_layers = n_layers_dbg
            text_config.layer_types = list(text_config.layer_types[:n_layers_dbg])
            # `ple_layer_ids` is one-indexed; drop any Engram layer past the new depth.
            text_config.ple_layer_ids = [i for i in text_config.ple_layer_ids if i <= n_layers_dbg]
        return cfg

    def apply_hp(self, model: Module, engine: Any) -> Module:
        training = engine.training_config
        if training.freeze_llm:
            model.model.language_model.requires_grad_(False)
            model.lm_head.requires_grad_(False)
            # Text-only microbatches still need a backward graph with a frozen decoder.
            model.enable_input_require_grads()
        if training.freeze_vit:
            model.model.visual.requires_grad_(False)
            model.model.visual.merger.requires_grad_(True)
        if training.freeze_projector:
            model.model.visual.merger.requires_grad_(False)
        return apply_hp_qwen4_exp(
            model,
            engine.ep_2d_mesh,
            cp_mesh=engine.cp_mesh_for_hp,
            attn_backend=engine.policy_config.attn_implementation,
            qsa_score_query_chunk_size=engine.policy_config.qsa_score_query_chunk_size,
            qsa_compile_score=engine.policy_config.qsa_compile_score,
            ep_backend=engine.policy_config.ep_backend,
            deepep_num_sms=engine.policy_config.deepep_num_sms,
            deepep_async_finish=engine.policy_config.deepep_async_finish,
            ep_weight_before_combine=engine.policy_config.ep_weight_before_combine,
        )
