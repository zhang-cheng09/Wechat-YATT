# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
from __future__ import annotations

import copy

import torch.nn as nn

from .modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRMSNorm,
)


class Qwen4ExpMTPModule(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if config.mtp_num_hidden_layers != 1:
            raise ValueError(
                "Qwen4Exp load-only MTP requires mtp_num_hidden_layers=1, got "
                f"{config.mtp_num_hidden_layers}"
            )
        if config.mtp_use_dedicated_embeddings:
            raise NotImplementedError("Qwen4Exp dedicated MTP embeddings are not supported")

        hidden_size = config.hidden_size
        hc_hidden_size = config.hc_count * hidden_size
        self.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(
            hidden_size,
            eps=config.rms_norm_eps,
        )
        self.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(
            hc_hidden_size,
            eps=config.rms_norm_eps,
        )
        self.fc_embedding = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size, bias=False)

        mtp_config = copy.deepcopy(config)
        mtp_config.num_hidden_layers = 1
        mtp_config.layer_types = ["qwen_sparse_attention"]
        mtp_config.ple_layer_ids = []
        self.config = mtp_config
        self.layers = nn.ModuleList([Qwen4ExpTextDecoderLayer(mtp_config, 0)])
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(
            mtp_config,
            use_combine=False,
        )


__all__ = ["Qwen4ExpMTPModule"]
