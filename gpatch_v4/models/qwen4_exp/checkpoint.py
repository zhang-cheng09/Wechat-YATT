# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Streaming checkpoint I/O for Qwen3.8-Flash-Next.

Bound onto the model by :func:`.hp.apply_hp` as ``load_checkpoint_hp`` /
``save_checkpoint_hp``.

The released checkpoint is 1658 tensors / 360 GB across 131 safetensors shards. Loading it
through ``from_pretrained`` would need the whole thing on one rank, so each rank streams
the shards itself and keeps only what it owns.

The model uses the checkpoint's native names: ``model.language_model.*``,
``model.visual.*``, ``lm_head.weight``, and ``mtp.*`` when MTP is loaded.
Engram ``shard_{i}`` keys are streamed into owner-local rows instead of using
HF's conversion mapping, which would concatenate the full table on one device.
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

from gpatch_v4.models.expert_gather import gather_ep_partition
from gpatch_v4.utils import log

from .engram import (
    OwnerShardedNGramEmbedding,
    iter_engram_hf_shards,
    materialize_engram_tables,
)

__all__ = ["load_checkpoint_hp", "save_checkpoint_hp"]

_MTP_PATTERN = re.compile(r"^mtp\.")
_ENGRAM_SHARD_RE = re.compile(r"^(?P<prefix>.*ngram_embedding)\.shard_(?P<index>\d+)\.weight$")
_EXPERT_SUFFIXES = (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")
_INDEX_FILE = "model.safetensors.index.json"


def _read_weight_map(hf_path: Path) -> dict:
    """Load the safetensors index, failing loudly if the checkpoint looks wrong.

    ``PretrainedConfig.from_pretrained`` silently returns *library defaults* for a
    directory with no ``config.json`` — a typo'd path would otherwise train a
    40-layer/2048-hidden model instead of raising. Check both files explicitly.
    """
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


def load_checkpoint_hp(
    self: nn.Module,
    hf_path: str,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Stream the HF checkpoint into this rank's shard of an EP + FSDP2 model.

    Parameters
    ----------
    hf_path : str
        Directory holding ``config.json`` and ``model.safetensors.index.json``.
    dtype : torch.dtype, default ``torch.float32``
        Target dtype for floating-point parameters. Disk is BF16; ``apply_hp`` requires
        an fp32 master, hence the default.

    Notes
    -----
    Three routing cases, all driven by the target's own type/shape rather than by name
    heuristics beyond the expert/Engram suffixes:

    * **Engram table** — FSDP-ignored, holding rows no other rank owns. Only overlaps
      with this rank's row range are read, through ``local_weight`` because ``apply_hp``
      wraps the parameter in a global ``DTensor(Shard(0))``.
    * **Expert weights** — read only this rank's ``[ep_rank * n : (ep_rank+1) * n)``
      slice off disk via ``get_slice``, so the full 512-expert tensor (3.35 GB per key
      per layer) is never materialized.
    * **Everything else** — read whole, then ``distribute_tensor`` onto the parameter's
      own mesh/placements. Peak transient is the largest single tensor
      (``embed_tokens``: 248320x2560, 2.5 GB at fp32).
    """
    hf_dir = Path(hf_path).expanduser()
    weight_map = _read_weight_map(hf_dir)
    rank = dist.get_rank() if dist.is_initialized() else 0

    param_device = _load_device()
    materialize_engram_tables(self, param_device, dtype)

    target_state = self.state_dict()
    expected_mtp_keys = {key for key in target_state if key.startswith("mtp.")}
    if expected_mtp_keys:
        checkpoint_mtp_keys = {key for key in weight_map if key.startswith("mtp.")}
        if checkpoint_mtp_keys != expected_mtp_keys:
            raise RuntimeError(
                "Qwen4Exp MTP checkpoint keys do not match the model: "
                f"missing={sorted(expected_mtp_keys - checkpoint_mtp_keys)}, "
                f"unexpected={sorted(checkpoint_mtp_keys - expected_mtp_keys)}"
            )
    engram_plan = _plan_engram(self)

    # Group by shard file so each of the 131 files is opened at most once.
    keys_by_shard: dict[str, list[str]] = {}
    skipped = 0
    for disk_key, shard in weight_map.items():
        if _MTP_PATTERN.match(disk_key) and self.mtp is None:
            skipped += 1
            continue
        keys_by_shard.setdefault(shard, []).append(disk_key)

    loaded: dict[str, torch.Tensor] = {}
    engram_rows_filled = 0
    for shard, disk_keys in sorted(keys_by_shard.items()):
        shard_path = hf_dir / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"{shard_path} referenced by {_INDEX_FILE} is missing")
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for disk_key in disk_keys:
                engram_match = _ENGRAM_SHARD_RE.match(disk_key)
                if engram_match is not None:
                    if engram_match.group("prefix") not in engram_plan["tables"]:
                        # A debug-truncated model can drop the only Engram-owning layer.
                        skipped += 1
                        continue
                    engram_rows_filled += _load_engram_shard(
                        handle, disk_key, engram_match, engram_plan, dtype
                    )
                    continue
                if disk_key not in target_state:
                    # debug_truncate_num_hidden_layers leaves extra disk layers.
                    skipped += 1
                    continue
                loaded[disk_key] = _load_regular_tensor(
                    handle,
                    disk_key,
                    target_state[disk_key],
                    self,
                    dtype,
                    param_device,
                )

    missing = set(target_state) - set(loaded) - engram_plan["target_keys"]
    if missing:
        raise RuntimeError(
            f"{len(missing)} parameters were not present in the checkpoint, e.g. "
            f"{sorted(missing)[:5]}"
        )
    if engram_plan["target_keys"] and engram_rows_filled != engram_plan["total_local_rows"]:
        raise RuntimeError(
            f"Engram load filled {engram_rows_filled} rows but this rank owns "
            f"{engram_plan['total_local_rows']}"
        )
    self.load_state_dict(loaded, strict=False, assign=True)
    _materialize_meta_buffers(self)
    self._engram_param_ids = {
        id(module.weight)
        for module in self.modules() if isinstance(module, OwnerShardedNGramEmbedding)
    }
    log(
        f"[rank {rank}] loaded {len(loaded)} tensors + "
        f"{engram_rows_filled} Engram rows from {hf_dir} "
        f"(skipped {skipped} tensors not in the model)",
        rank=0,
    )


def _load_device() -> torch.device:
    # fully_shard leaves parameters on meta until assign; do not read target.device.
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _load_regular_tensor(
    handle,
    disk_key: str,
    target: torch.Tensor,
    model: nn.Module,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Read one non-Engram tensor, EP-slicing experts, and match the target's layout."""
    is_expert = disk_key.endswith(_EXPERT_SUFFIXES)
    if is_expert:
        ep_size = model._ep_size
        ep_rank = model._ep_rank
        full_experts = handle.get_slice(disk_key).get_shape()[0]
        if full_experts % ep_size != 0:
            raise ValueError(
                f"{disk_key} has {full_experts} experts, not divisible by ep_size={ep_size}"
            )
        num_local = full_experts // ep_size
        start = ep_rank * num_local
        # Slice on disk: the full tensor is 3.35 GB per key per layer.
        value = handle.get_slice(disk_key)[start:start + num_local]
    else:
        value = handle.get_tensor(disk_key)

    if value.dtype.is_floating_point:
        value = value.to(dtype)

    if isinstance(target, DTensor):
        return distribute_tensor(value, target.device_mesh, target.placements)
    if tuple(value.shape) != tuple(target.shape):
        raise ValueError(
            f"{disk_key}: checkpoint shape {tuple(value.shape)} != model shape "
            f"{tuple(target.shape)}"
        )
    # Hash buffers are not DTensors; safetensors + assign=True would leave them
    # on CPU. target.device is also CPU if they were born there.
    return value.to(device)


def _plan_engram(model: nn.Module) -> dict:
    """Locate the owner-sharded tables and how many rows this rank must fill.

    No alignment constraint between owner ranges and checkpoint shards is needed:
    :func:`_load_engram_shard` intersects the two in **global row space**, so any
    combination works — a rank may read part of one shard, or several whole shards.
    (NVIDIA's implementation does require whole-shard alignment, because it narrows
    write-through views into its own DTensor shard instead of copying overlaps. Don't
    port that restriction; it doesn't apply here.)
    """
    tables = {
        name: module
        for name, module in model.named_modules() if isinstance(module, OwnerShardedNGramEmbedding)
    }
    plan = {"tables": tables, "target_keys": set(), "total_local_rows": 0}
    for name, table in tables.items():
        plan["target_keys"].add(f"{name}.weight")
        plan["total_local_rows"] += table.local_rows
    return plan


def _load_engram_shard(
    handle,
    disk_key: str,
    match: re.Match,
    plan: dict,
    dtype: torch.dtype,
) -> int:
    """Copy the overlap between one checkpoint shard and this rank's owned rows.

    Returns the number of rows written (0 when the shard belongs to another rank).
    """
    module_name = match.group("prefix")
    shard_index = int(match.group("index"))
    table = plan["tables"].get(module_name)
    if table is None:
        raise KeyError(f"{disk_key} has no owner-sharded table at {module_name!r}")

    rows_per_shard = handle.get_slice(disk_key).get_shape()[0]
    global_start = shard_index * rows_per_shard
    global_end = global_start + rows_per_shard
    owned_start = table.global_row_start
    owned_end = owned_start + table.local_rows

    overlap_start = max(global_start, owned_start)
    overlap_end = min(global_end, owned_end)
    if overlap_start >= overlap_end:
        return 0

    value = handle.get_slice(disk_key)[overlap_start - global_start:overlap_end - global_start]
    with torch.no_grad():
        # `local_weight` unwraps the DTensor `apply_hp` puts around the shard.
        table.local_weight[overlap_start - owned_start:overlap_end - owned_start].copy_(
            value.to(dtype)
        )
    return overlap_end - overlap_start


def _materialize_meta_buffers(model: nn.Module) -> None:
    """Restore the non-persistent buffers, which the checkpoint does not contain.

    The text and vision rotary modules have non-persistent ``inv_freq`` buffers. The
    Engram metadata buffers (``layer_multipliers``,
    ``ngram_heads_vocab_sizes``, ``ngram_heads_offsets``) are persistent and arrive from
    the checkpoint.

    These are recomputed **unconditionally**, not only when still on meta. They are pure
    functions of the config and are absent from the checkpoint, so recomputing is always
    correct — whereas keying off ``is_meta`` silently leaves uninitialized garbage behind
    if anything called ``to_empty()`` on the way here.

    Any *other* meta buffer raises: a silent zero-init would only show up later as a bad
    loss curve.
    """
    device: Optional[torch.device] = None
    for param in model.parameters():
        if not param.is_meta:
            device = param.to_local().device if isinstance(param, DTensor) else param.device
            break
    if device is None:
        raise RuntimeError("_materialize_meta_buffers must run after parameters are loaded")

    for module in model.modules():
        if type(module).__name__ == "Qwen4ExpTextRotaryEmbedding":
            rope_type = module.config.rope_parameters["rope_type"]
            rope_init_fn = (
                type(module).compute_default_rope_parameters
                if rope_type == "default" else ROPE_INIT_FUNCTIONS[rope_type]
            )
            inv_freq, module.attention_scaling = rope_init_fn(module.config)
            inv_freq = inv_freq.to(device)
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            module.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        elif type(module).__name__ == "Qwen4ExpVisionRotaryEmbedding":
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


_SAVE_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)


class _ShardWriter:
    """Accumulate tensors and flush a safetensors shard once past ``max_bytes``.

    Necessary rather than convenient: this model is ~352 GB in bf16, so neither a single
    output file nor a rank-0 dict holding every tensor is viable. Files are named with a
    placeholder count and renamed once the total is known, matching HF's layout.
    """
    def __init__(self, out_dir: Path, max_bytes: int = 4 * 1024**3) -> None:
        self.out_dir = out_dir
        self.max_bytes = max_bytes
        self.pending: dict[str, torch.Tensor] = {}
        self.pending_bytes = 0
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0
        self.shard_index = 0
        self.expert_files: list[str] = []

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
        files = [
            f"model-{index:05d}-of-PENDING.safetensors" for index in range(1, self.shard_index + 1)
        ] + self.expert_files
        total = len(files)
        renames = {}
        for index, old in enumerate(files, 1):
            new = f"model-{index:05d}-of-{total:05d}.safetensors"
            (self.out_dir / old).rename(self.out_dir / new)
            renames[old] = new
        self.shard_index = total
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
    """``full_tensor()`` after ensuring the DTensor lives on CUDA.

    NCCL-only process groups reject collectives on CPU shards
    (``No backend type associated with device type cpu``), including empty
    FSDP local shards that may still report ``device_type == "cpu"``.
    """
    local = param.to_local()
    if local.is_cuda:
        return param.full_tensor()
    dt_cuda = param.cuda()
    torch.cuda.synchronize()
    try:
        return dt_cuda.full_tensor()
    finally:
        del dt_cuda


def _gather_ep_local_expert(model: nn.Module, param: torch.Tensor) -> torch.Tensor:
    if isinstance(param, DTensor):
        if model._ep_size == 1:
            return _dtensor_full_tensor_on_cuda(param)
        local = param.to_local().contiguous()
        if not local.is_cuda:
            local = local.cuda()
            torch.cuda.synchronize()
        return gather_ep_partition(param, local, model._ep_fsdp_mesh)
    return param.detach()


def _gather_expert_for_save(model: nn.Module, param: torch.Tensor) -> torch.Tensor:
    """Reassemble an expert tensor in model expert order on every EP rank."""
    ep_local = _gather_ep_local_expert(model, param)
    if model._ep_size == 1:
        return ep_local

    ep_chunks = [torch.empty_like(ep_local) for _ in range(model._ep_size)]
    dist.all_gather(ep_chunks, ep_local, group=model._ep_group)
    value = torch.cat(ep_chunks, dim=0)
    del ep_chunks, ep_local
    return value


def _write_ep_expert_shard(
    model: nn.Module,
    value: torch.Tensor,
    disk_key: str,
    path: Path,
) -> int:
    shape = list(value.shape)
    shape[0] *= model._ep_size
    partition_bytes = value.numel() * torch.bfloat16.itemsize
    total_bytes = partition_bytes * model._ep_size
    # HF keeps all experts in one key; owners write disjoint byte ranges, not duplicate keys.
    header = json.dumps(
        {
            "__metadata__": {
                "format": "pt"
            },
            disk_key: {
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [0, total_bytes]
            },
        }
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    data_start = 8 + len(header)
    if dist.get_rank() == 0:
        with path.open("wb") as handle:
            handle.write(len(header).to_bytes(8, "little"))
            handle.write(header)
            handle.truncate(data_start + total_bytes)
    dist.barrier()
    if model._ep_fsdp_mesh.get_local_rank() == 0:
        partition = value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        with path.open("r+b") as handle:
            handle.seek(data_start + model._ep_rank * partition_bytes)
            handle.write(memoryview(partition.view(torch.uint8).numpy()).cast("B"))
    dist.barrier()
    return total_bytes


def save_checkpoint_hp(
    self: nn.Module,
    save_path: str,
    orig_ckpt_dir: Optional[str] = None,
    preserve_mtp: bool = False,
    save_distributed_ep_shard: bool = False,
) -> None:
    """Write a ``from_pretrained``-ready bf16 HF directory.

    By default rank 0 writes fully gathered experts. With ``save_distributed_ep_shard``,
    EP partitions write disjoint ranges of shared shards; rank 0 writes other weights
    and the index. Every rank must call this collectively.
    Parameter names stay unchanged; the Engram table is re-emitted as
    HF ``shard_{i}`` keys (``i < split_ngram_parts``) so the output reloads through
    :func:`load_checkpoint_hp` and SGLang ``load_qwen4_exp_ple_shard``.

    Parameters
    ----------
    save_path : str
        Output directory, shared by all ranks when EP is enabled.
    orig_ckpt_dir : str, optional
        Source checkpoint whose config / tokenizer files are copied alongside. Required:
        without them the output cannot be reloaded.
    preserve_mtp : bool, default False
        Require the model to contain the MTP head when saving.
    save_distributed_ep_shard : bool, default False
        Write experts per EP partition without cross-EP gather; requires a shared directory.

    """
    if preserve_mtp and self.mtp is None:
        raise ValueError("preserve_mtp=True requires a loaded qwen4_exp MTP head")
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

    engram_tables = {
        name: module
        for name, module in self.named_modules() if isinstance(module, OwnerShardedNGramEmbedding)
    }
    engram_keys = {f"{name}.weight" for name in engram_tables}
    writer = _ShardWriter(out_dir) if rank == 0 else None

    state = self.state_dict()
    for name, param in state.items():
        if name in engram_keys:
            continue
        if name.endswith(_EXPERT_SUFFIXES):
            if save_distributed_ep_shard and self._ep_size > 1:
                value = _gather_ep_local_expert(self, param)
                filename = f"{name}.safetensors"
                total_bytes = _write_ep_expert_shard(self, value, name, out_dir / filename)
                if rank == 0:
                    writer.expert_files.append(filename)
                    writer.weight_map[name] = filename
                    writer.total_bytes += total_bytes
                del value
                continue
            value = _gather_expert_for_save(self, param)
        else:
            # Collective for DTensors, so it runs on every rank; only rank 0 keeps it.
            value = (_dtensor_full_tensor_on_cuda(param) if isinstance(param, DTensor) else param)
        if rank == 0:
            gathered = value.detach().to("cpu")
            if gathered.dtype.is_floating_point:
                gathered = gathered.to(torch.bfloat16)
            writer.add(name, gathered)
        del value

    # Owner-sharded and FSDP-ignored, so gather explicitly — one owner broadcast at a
    # time, re-packed into HF ``split_ngram_parts`` shards (never materialize 102 GB).
    split_parts = 0
    if engram_tables:
        text_config = self.config.get_text_config()
        split_parts = int(text_config.split_ngram_parts)
    for name, table in engram_tables.items():
        for disk_key, shard in _iter_engram_shards(table, f"{name}.weight", rank, split_parts):
            if rank == 0:
                writer.add(disk_key, shard)

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


def _iter_engram_shards(
    table: OwnerShardedNGramEmbedding,
    target_key: str,
    rank: int,
    split_ngram_parts: int,
):
    """Yield ``(shard_key, cpu_tensor)`` in HF ``split_ngram_parts`` layout."""
    return iter_engram_hf_shards(
        table,
        target_key,
        split_ngram_parts,
        out_device="cpu",
        yield_on_all_ranks=False,
    )
