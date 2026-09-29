# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Qwen3.5 dense VL modeling for the FSDP2 backend."""
from gpatch_v4.models.qwen3_5.attn import Qwen3_5CPAttention
from gpatch_v4.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from gpatch_v4.models.qwen3_5.gdn import Qwen3_5CPGatedDeltaNet
from gpatch_v4.models.qwen3_5.hp import (
    Qwen3_5HpForConditionalGeneration,
    apply_hp,
    bind_dynamic_cp,
    swap_parallel_modules,
)
from gpatch_v4.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5PreTrainedModel,
    Qwen3_5TextModel,
)

__all__ = [
    "Qwen3_5CPAttention",
    "Qwen3_5CPGatedDeltaNet",
    "Qwen3_5Config",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5HpForConditionalGeneration",
    "Qwen3_5PreTrainedModel",
    "Qwen3_5TextConfig",
    "Qwen3_5TextModel",
    "Qwen3_5VisionConfig",
    "apply_hp",
    "bind_dynamic_cp",
    "swap_parallel_modules",
]
