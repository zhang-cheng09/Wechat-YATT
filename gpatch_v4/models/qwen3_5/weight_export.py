# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Qwen3.5 dense VL weight export for RL ``update_weights``.

SGLang reload expects the HF disk layout. Visual params are full replicas
(not DTensor); text params are gathered from FSDP shards.
"""
from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.distributed.tensor import DTensor
from torch.nn import Module

from .checkpoint import _dtensor_full_tensor_on_cuda, _is_tied_lm_head

__all__ = ["export_qwen3_5_bf16_weights"]


def _to_export_tensor(tensor: torch.Tensor) -> torch.Tensor:
    out = tensor.detach()
    if out.dtype.is_floating_point and out.dtype != torch.bfloat16:
        out = out.to(torch.bfloat16)
    if not out.is_cuda:
        out = out.cuda()
    return out.contiguous()


def export_qwen3_5_bf16_weights(model: Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(disk_key, cuda_bf16_tensor)`` for SGLang weight update."""
    tied = _is_tied_lm_head(model)
    for name, param in model.state_dict().items():
        if ".te_attn." in name:
            continue
        if tied and name == "lm_head.weight":
            continue
        if name.startswith("model.visual.") and isinstance(param, DTensor):
            raise RuntimeError(f"{name} is a DTensor; ViT parameters must stay unsharded")
        if isinstance(param, DTensor):
            value = _dtensor_full_tensor_on_cuda(param)
        else:
            value = param
        yield name, _to_export_tensor(value)
        del value
