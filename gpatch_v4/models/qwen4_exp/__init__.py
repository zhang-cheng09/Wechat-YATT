# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Qwen3.8-Flash-Next text and vision modeling for the FSDP2 backend.

The configuration and modeling files are vendored from transformers. CP branches live
in the modeling classes; QSA, EP experts, and Engram are swapped after construction.
"""
from transformers import Qwen4ExpConfig

from gpatch_v4.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from gpatch_v4.models.qwen4_exp.engram import (
    OwnerShardedNGramEmbedding,
    Qwen4ExpEngramEmbedding,
)
from gpatch_v4.models.qwen4_exp.hp import (
    apply_hp,
    bind_dynamic_cp,
    set_activation_checkpointing,
    swap_parallel_modules,
)
from gpatch_v4.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpForCausalLM,
    Qwen4ExpPreTrainedModel,
    Qwen4ExpTextModel,
)
from gpatch_v4.models.qwen4_exp.moe import Qwen4ExpEPExperts, freeze_router_weights
from gpatch_v4.models.qwen4_exp.mtp import Qwen4ExpMTPModule
from gpatch_v4.models.qwen4_exp.qsa import Qwen4ExpQSAAttention, Qwen4ExpQSAIndexer
from gpatch_v4.models.qwen4_exp.vl import Qwen4ExpHpForConditionalGeneration

__all__ = [
    "OwnerShardedNGramEmbedding",
    "Qwen4ExpConfig",
    "Qwen4ExpEPExperts",
    "Qwen4ExpEngramEmbedding",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpHpForConditionalGeneration",
    "Qwen4ExpMTPModule",
    "Qwen4ExpPreTrainedModel",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAIndexer",
    "Qwen4ExpTextConfig",
    "Qwen4ExpTextModel",
    "apply_hp",
    "bind_dynamic_cp",
    "freeze_router_weights",
    "set_activation_checkpointing",
    "swap_parallel_modules",
]
