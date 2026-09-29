# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Qwen3.8-Flash-Next (``qwen4_exp``) weight export for RL ``update_weights``.

SGLang reload expects the **HF disk** layout produced by
:func:`.checkpoint.save_checkpoint_hp`:

* ``model.language_model.*`` and ``model.visual.*`` keep their native names
* ``lm_head.weight`` unchanged
* Engram stays split as ``...ngram_embedding.shard_{i}.weight`` with
  ``i in [0, split_ngram_parts)`` (never a full 102 GB table; never ``shard_{owner}``)
* MoE experts stay packed ``gate_up_proj`` / ``down_proj``, EP-gathered into model
  expert order

Tensors are yielded on **every** rank (collectives for DTensor / EP / Engram broadcast
run on all ranks) and stay on CUDA in bf16 for SGLang IPC.
"""
from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.distributed.tensor import DTensor
from torch.nn import Module

from .checkpoint import (
    _EXPERT_SUFFIXES,
    _dtensor_full_tensor_on_cuda,
    _gather_expert_for_save,
)
from .engram import OwnerShardedNGramEmbedding, iter_engram_hf_shards

__all__ = [
    "export_qwen4_exp_bf16_weights",
]


def _to_export_tensor(tensor: torch.Tensor) -> torch.Tensor:
    out = tensor.detach()
    if out.dtype.is_floating_point and out.dtype != torch.bfloat16:
        out = out.to(torch.bfloat16)
    if not out.is_cuda:
        out = out.cuda()
    return out.contiguous()


def _split_ngram_parts(model: Module) -> int:
    parts = int(model.config.get_text_config().split_ngram_parts)
    if parts <= 0:
        raise ValueError(f"split_ngram_parts must be positive, got {parts}")
    return parts


def export_qwen4_exp_bf16_weights(model: Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(disk_key, cuda_bf16_tensor)`` for SGLang weight update.

    Parameters
    ----------
    model : Module
        An ``apply_hp``-wrapped :class:`Qwen4ExpHpForConditionalGeneration` (must expose
        ``_ep_size`` / ``_ep_group`` / ``_ep_fsdp_mesh``).

    Yields
    ------
    tuple[str, torch.Tensor]
        HF disk-form name and a contiguous CUDA bf16 tensor. Engram shards follow
        ``split_ngram_parts`` (SGLang / HF), not train ``world_size``. VLM exports
        visual weights alongside the text backbone. Loaded MTP weights keep their
        top-level ``mtp.*`` names.
    """
    if not hasattr(model, "_ep_size"):
        raise AttributeError(
            "export_qwen4_exp_bf16_weights expects an apply_hp-wrapped model with "
            "_ep_size; got "
            f"{type(model).__name__}"
        )

    engram_tables = {
        name: module
        for name, module in model.named_modules() if isinstance(module, OwnerShardedNGramEmbedding)
    }
    engram_keys = {f"{name}.weight" for name in engram_tables}
    split_parts = _split_ngram_parts(model) if engram_tables else 0

    state = model.state_dict()
    for name, param in state.items():
        if name in engram_keys:
            continue
        if name.endswith(_EXPERT_SUFFIXES):
            value = _gather_expert_for_save(model, param)
        elif isinstance(param, DTensor):
            # Same as DSV4: full_tensor on a CPU DTensor shard hits NCCL-only PG.
            value = _dtensor_full_tensor_on_cuda(param)
        else:
            value = param
        yield name, _to_export_tensor(value)
        del value

    for name, table in engram_tables.items():
        for shard_key, shard in iter_engram_hf_shards(
            table,
            f"{name}.weight",
            split_parts,
            out_device="cuda",
            yield_on_all_ranks=True,
        ):
            yield shard_key, shard
            del shard
