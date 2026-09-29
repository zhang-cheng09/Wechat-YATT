# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Streaming checkpoint I/O for Qwen3.5 dense VL.

Bound onto the model by :func:`.hp.apply_hp` as ``load_checkpoint_hp`` /
``save_checkpoint_hp``.

HF disk keys already match the VL module tree:

===================================================== =========================
disk                                                  model
===================================================== =========================
``model.language_model.X``                            unchanged
``model.visual.X``                                    unchanged (not DTensor)
``lm_head.weight``                                    unchanged (may be tied)
``mtp.*``                                             skipped
===================================================== =========================
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor, distribute_tensor
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from gpatch_v4.utils import log

__all__ = ["load_checkpoint_hp", "save_checkpoint_hp"]

_SKIP_PATTERNS = (re.compile(r"^mtp\."), )
_INDEX_FILE = "model.safetensors.index.json"
_SAVE_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
)


def _read_weight_map(hf_path: Path) -> dict:
    if not (hf_path / "config.json").is_file():
        raise FileNotFoundError(
            f"{hf_path}/config.json is missing. Note that transformers would silently "
            "fall back to default config values here rather than raising, so this is "
            "checked explicitly."
        )
    index_path = hf_path / _INDEX_FILE
    if not index_path.is_file():
        raise FileNotFoundError(f"{index_path} is missing; expected a sharded checkpoint.")
    return json.loads(index_path.read_text())["weight_map"]


def _is_tied_lm_head(model: nn.Module) -> bool:
    if model.config.tie_word_embeddings or model.config.get_text_config().tie_word_embeddings:
        return True
    return model.lm_head.weight is model.model.language_model.embed_tokens.weight


def load_checkpoint_hp(
    self: nn.Module,
    hf_path: str,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Stream the HF checkpoint into this rank's FSDP shard plus a full ViT."""
    hf_dir = Path(hf_path).expanduser()
    weight_map = _read_weight_map(hf_dir)
    rank = dist.get_rank() if dist.is_initialized() else 0

    param_device = _load_device()
    target_state = dict(self.state_dict())
    tied = _is_tied_lm_head(self)
    if tied:
        target_state.pop("lm_head.weight", None)

    keys_by_shard: dict[str, list[str]] = {}
    skipped = 0
    for disk_key, shard in weight_map.items():
        if any(pattern.match(disk_key) for pattern in _SKIP_PATTERNS):
            skipped += 1
            continue
        keys_by_shard.setdefault(shard, []).append(disk_key)

    loaded: dict[str, torch.Tensor] = {}
    for shard, disk_keys in sorted(keys_by_shard.items()):
        shard_path = hf_dir / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"{shard_path} referenced by {_INDEX_FILE} is missing")
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for disk_key in disk_keys:
                if disk_key not in target_state:
                    skipped += 1
                    continue
                loaded[disk_key] = _load_regular_tensor(
                    handle,
                    disk_key,
                    target_state[disk_key],
                    dtype,
                    param_device,
                )

    missing = {key for key in set(target_state) - set(loaded) if ".te_attn." not in key}
    if missing:
        raise RuntimeError(
            f"{len(missing)} parameters were not present in the checkpoint, e.g. "
            f"{sorted(missing)[:5]}"
        )

    self.load_state_dict(loaded, strict=False, assign=True)
    if tied:
        self.lm_head.weight = self.model.language_model.embed_tokens.weight
    _materialize_meta_buffers(self)
    log(
        f"[rank {rank}] loaded {len(loaded)} tensors from {hf_dir} "
        f"(skipped {skipped} tensors not in the model)",
        rank=0,
    )


def _load_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _load_regular_tensor(
    handle,
    model_key: str,
    target: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    value = handle.get_tensor(model_key)
    if value.dtype.is_floating_point:
        value = value.to(dtype)

    if model_key.startswith("model.visual."):
        if isinstance(target, DTensor):
            raise RuntimeError(f"{model_key} is a DTensor; ViT parameters must stay unsharded")
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"{model_key}: checkpoint shape {tuple(value.shape)} != model shape "
                f"{tuple(target.shape)}"
            )
        return value.to(device)

    if isinstance(target, DTensor):
        return distribute_tensor(value, target.device_mesh, target.placements)
    if tuple(value.shape) != tuple(target.shape):
        raise ValueError(
            f"{model_key}: checkpoint shape {tuple(value.shape)} != model shape "
            f"{tuple(target.shape)}"
        )
    return value.to(device)


def _materialize_meta_buffers(model: nn.Module) -> None:
    device: Optional[torch.device] = None
    for param in model.parameters():
        if not param.is_meta:
            device = param.to_local().device if isinstance(param, DTensor) else param.device
            break
    if device is None:
        raise RuntimeError("_materialize_meta_buffers must run after parameters are loaded")

    for module in model.modules():
        if type(module).__name__ == "Qwen3_5TextRotaryEmbedding":
            rope_type = module.config.rope_parameters["rope_type"]
            rope_init_fn = (
                type(module).compute_default_rope_parameters
                if rope_type == "default" else ROPE_INIT_FUNCTIONS[rope_type]
            )
            inv_freq, module.attention_scaling = rope_init_fn(module.config)
            inv_freq = inv_freq.to(device)
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            module.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        elif type(module).__name__ == "Qwen3_5VisionRotaryEmbedding":
            inv_freq = 1.0 / (
                module.theta**(torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim)
            )
            module.register_buffer("inv_freq", inv_freq.to(device), persistent=False)

        remaining = sorted(
            name for name, buffer in module.named_buffers(recurse=False) if buffer.is_meta
        )
        if remaining:
            raise RuntimeError(
                f"unhandled meta buffers in {type(module).__name__}: {remaining}. "
                "Add a branch here — silent zero-init is a correctness bug."
            )


class _ShardWriter:
    def __init__(self, out_dir: Path, max_bytes: int = 4 * 1024**3) -> None:
        self.out_dir = out_dir
        self.max_bytes = max_bytes
        self.pending: dict[str, torch.Tensor] = {}
        self.pending_bytes = 0
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0
        self.shard_index = 0

    def add(self, key: str, tensor: torch.Tensor) -> None:
        self.pending[key] = tensor
        nbytes = tensor.numel() * tensor.element_size()
        self.pending_bytes += nbytes
        self.total_bytes += nbytes
        if self.pending_bytes >= self.max_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        self.shard_index += 1
        name = f"model-{self.shard_index:05d}-of-PENDING.safetensors"
        save_file(self.pending, str(self.out_dir / name), metadata={"format": "pt"})
        for key in self.pending:
            self.weight_map[key] = name
        self.pending.clear()
        self.pending_bytes = 0

    def finalize(self) -> None:
        self.flush()
        total = self.shard_index
        renames = {}
        for index in range(1, total + 1):
            old = f"model-{index:05d}-of-PENDING.safetensors"
            new = f"model-{index:05d}-of-{total:05d}.safetensors"
            (self.out_dir / old).rename(self.out_dir / new)
            renames[old] = new
        weight_map = {key: renames[shard] for key, shard in self.weight_map.items()}
        (self.out_dir / _INDEX_FILE).write_text(
            json.dumps(
                {
                    "metadata": {
                        "total_size": self.total_bytes
                    },
                    "weight_map": weight_map
                },
                indent=2,
            )
        )


def _dtensor_full_tensor_on_cuda(param: DTensor) -> torch.Tensor:
    local = param.to_local()
    if local.is_cuda:
        return param.full_tensor()
    dt_cuda = param.cuda()
    torch.cuda.synchronize()
    try:
        return dt_cuda.full_tensor()
    finally:
        del dt_cuda


def save_checkpoint_hp(
    self: nn.Module,
    save_path: str,
    orig_ckpt_dir: Optional[str] = None,
    preserve_mtp: bool = False,
) -> None:
    """Write a ``from_pretrained``-ready bf16 HF directory."""
    if preserve_mtp:
        raise NotImplementedError(
            "preserve_mtp=True is meaningless for qwen3_5 FSDP2: the MTP head is "
            "never loaded, so it cannot be written back."
        )
    if orig_ckpt_dir is None:
        raise ValueError(
            "orig_ckpt_dir is required so config/tokenizer files can be copied; without "
            "them the output directory is not loadable."
        )

    rank = dist.get_rank() if dist.is_initialized() else 0
    out_dir = Path(save_path).expanduser()
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    writer = _ShardWriter(out_dir) if rank == 0 else None
    tied = _is_tied_lm_head(self)

    for name, param in self.state_dict().items():
        if ".te_attn." in name:
            continue
        if tied and name == "lm_head.weight":
            continue
        if name.startswith("model.visual.") and isinstance(param, DTensor):
            raise RuntimeError(f"{name} is a DTensor; ViT parameters must stay unsharded")
        value = (_dtensor_full_tensor_on_cuda(param) if isinstance(param, DTensor) else param)
        if rank == 0:
            gathered = value.detach().to("cpu")
            if gathered.dtype.is_floating_point:
                gathered = gathered.to(torch.bfloat16)
            writer.add(name, gathered)
        del value

    if rank == 0:
        writer.finalize()
        source = Path(orig_ckpt_dir).expanduser()
        for filename in _SAVE_FILES:
            candidate = source / filename
            if candidate.is_file():
                shutil.copy2(candidate, out_dir / filename)
        log(
            f"save_checkpoint_hp wrote {len(writer.weight_map)} tensors in "
            f"{writer.shard_index} shards ({writer.total_bytes / 1e9:.1f} GB) to {out_dir}",
            rank=0,
        )

    if dist.is_initialized():
        dist.barrier()
