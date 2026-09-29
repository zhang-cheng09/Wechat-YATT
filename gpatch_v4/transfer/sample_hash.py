"""Put-vs-restore sample hashes for TransferQueue correctness dumps."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from gpatch_v4.utils import log
from gpatch_v4.utils.training_utils import expand_rollout_batch

TQ_SAMPLE_HASH_FIELDS: Tuple[str, ...] = (
    "tokens",
    "rollout_log_probs",
    "rollout_logprobs",
    "rewards",
    "group_id",
    "mask",
    "prompt_lengths",
    "sequence_lengths",
    "sample_mask",
    "token_weights",
)
UNREAD_TQ_SAMPLE_HASH_FIELDS: frozenset[str] = frozenset(("rollout_log_probs", "rollout_logprobs"))


@dataclass
class StageHashDump:
    samples: Dict[str, Dict[str, Dict[str, Any]]]
    order: List[str] = field(default_factory=list)
    microbatches: List[List[str]] = field(default_factory=list)
    field_catalog: Dict[str, Any] = field(default_factory=dict)
    group_chunk_errors: List[str] = field(default_factory=list)


def sample_identity(sample: Mapping[str, Any]) -> str:
    """Stable join key across AgentLoop put and train restore.

    Prefers ``unique_id``; ``traj_id`` disambiguates sampling repeats that
    share a prompt id. Falls back to ``group_id:traj_id``.
    """
    traj = sample.get("traj_id")
    if "unique_id" in sample:
        uid = sample["unique_id"]
        if traj is None:
            return str(uid)
        return f"{uid}:{traj}"
    assert "group_id" in sample, "sample hash identity needs unique_id or group_id"
    return f"{sample['group_id']}:{0 if traj is None else traj}"


def hash_field_value(value: Any) -> Dict[str, Any]:
    """Hash one sample field; tensors are hashed on CPU bits, not device.

    Parameters
    ----------
    value : Any
        ``None``, ``torch.Tensor``, ``numpy.ndarray``, or a control-plane
        scalar/string. Nested Python containers are not supported.

    Returns
    -------
    dict
        ``kind`` / ``sha256`` plus ``dtype`` and ``shape`` for array values.
    """
    if value is None:
        return {"kind": "none", "sha256": hashlib.sha256(b"none").hexdigest()}
    if isinstance(value, torch.Tensor):
        payload, dtype, shape = _tensor_payload(value)
        digest = hashlib.sha256(payload)
        digest.update(dtype.encode("utf-8"))
        digest.update(str(tuple(shape)).encode("utf-8"))
        return {
            "kind": "torch",
            "dtype": dtype,
            "shape": shape,
            "sha256": digest.hexdigest(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256(array.tobytes())
        dtype = str(array.dtype)
        shape = list(array.shape)
        digest.update(dtype.encode("utf-8"))
        digest.update(str(tuple(shape)).encode("utf-8"))
        return {
            "kind": "numpy",
            "dtype": dtype,
            "shape": shape,
            "sha256": digest.hexdigest(),
        }
    if isinstance(value, np.generic):
        return hash_field_value(np.asarray(value))
    blob = repr(value).encode("utf-8")
    return {
        "kind": type(value).__name__,
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def hash_sample_fields(
    sample: Mapping[str, Any],
    fields: Sequence[str] = TQ_SAMPLE_HASH_FIELDS,
) -> Dict[str, Dict[str, Any]]:
    hashed = {}
    for field_name in fields:
        if field_name not in sample:
            continue
        hashed[field_name] = hash_field_value(sample[field_name])
    return hashed


def hash_rollout_batches(
    rollout_batches: Sequence[Mapping[str, List[Any]]],
    fields: Sequence[str] = TQ_SAMPLE_HASH_FIELDS,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Expand field-major batches and hash each sample, keyed by identity."""
    return hash_rollout_layout(rollout_batches, fields=fields).samples


def hash_rollout_layout(
    rollout_batches: Sequence[Mapping[str, List[Any]]],
    fields: Sequence[str] = TQ_SAMPLE_HASH_FIELDS,
    *,
    chunk_size: Optional[int] = None,
) -> "StageHashDump":
    """Hash samples and keep per-microbatch identity order for GRPO layout checks."""
    keyed: Dict[str, Dict[str, Dict[str, Any]]] = {}
    order: List[str] = []
    microbatches: List[List[str]] = []
    for rollout_batch in rollout_batches:
        batch_ids: List[str] = []
        for sample in expand_rollout_batch(dict(rollout_batch)):
            key = sample_identity(sample)
            assert key not in keyed, f"duplicate sample identity {key!r}"
            keyed[key] = hash_sample_fields(sample, fields)
            order.append(key)
            batch_ids.append(key)
        microbatches.append(batch_ids)
    return StageHashDump(
        samples=keyed,
        order=order,
        microbatches=microbatches,
        field_catalog=rollout_field_catalog(rollout_batches),
        group_chunk_errors=grpo_positional_group_errors(rollout_batches, chunk_size)
        if chunk_size is not None else [],
    )


def _scalar_id(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        assert value.numel() == 1, f"expected scalar tensor, got shape {tuple(value.shape)}"
        return value.detach().cpu().reshape(-1)[0].item()
    if isinstance(value, np.ndarray):
        assert value.size == 1, f"expected scalar ndarray, got shape {value.shape}"
        return value.reshape(-1)[0].item()
    return value


def _value_spec(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"kind": "none"}
    if isinstance(value, torch.Tensor):
        return {
            "kind": "torch",
            "dtype": str(value.dtype).replace("torch.", ""),
            "shape": list(value.shape),
        }
    if isinstance(value, np.ndarray):
        return {"kind": "numpy", "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return _value_spec(np.asarray(value))
    return {"kind": type(value).__name__}


def _spec_sort_key(spec: Mapping[str, Any]) -> str:
    return json.dumps(dict(spec), sort_keys=True, default=str)


def _collapse_field_specs(variants: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(variants) == 1:
        return variants[0]
    kinds = {spec.get("kind") for spec in variants}
    dtypes = {spec.get("dtype") for spec in variants}
    if len(kinds) == 1 and len(dtypes) == 1:
        collapsed = {"kind": variants[0]["kind"]}
        if "dtype" in variants[0]:
            collapsed["dtype"] = variants[0]["dtype"]
        shapes = [spec["shape"] for spec in variants if "shape" in spec]
        if shapes:
            # Sort so catalog is independent of sample / shard encounter order.
            collapsed["shapes"] = sorted(shapes)
        return collapsed
    return {"variants": sorted(variants, key=_spec_sort_key)}


def _meta_to_value_spec(meta: Mapping[str, Any]) -> Dict[str, Any]:
    spec: Dict[str, Any] = {"kind": meta["kind"]}
    if "dtype" in meta:
        spec["dtype"] = meta["dtype"]
    if "shape" in meta:
        spec["shape"] = list(meta["shape"])
    return spec


def catalog_from_sample_hashes(
    samples: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Rebuild partition-independent catalog from hashed sample field metas."""
    specs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for fields in samples.values():
        for field_name, meta in fields.items():
            spec = _meta_to_value_spec(meta)
            if spec not in specs[field_name]:
                specs[field_name].append(spec)
    return {
        field_name: _collapse_field_specs(variants)
        for field_name, variants in sorted(specs.items())
    }


def rollout_field_catalog(rollout_batches: Sequence[Mapping[str, List[Any]]], ) -> Dict[str, Any]:
    """Per-field kind/dtype; same dtype with ragged shapes keeps ``dtype``."""
    specs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rollout_batch in rollout_batches:
        for sample in expand_rollout_batch(dict(rollout_batch)):
            for field_name, value in sample.items():
                spec = _value_spec(value)
                if spec not in specs[field_name]:
                    specs[field_name].append(spec)
    return {
        field_name: _collapse_field_specs(variants)
        for field_name, variants in sorted(specs.items())
    }


def grpo_positional_group_errors(
    rollout_batches: Sequence[Mapping[str, List[Any]]],
    chunk_size: int,
) -> List[str]:
    """Errors if a microbatch is not ``chunk_size``-aligned same-``group_id`` chunks."""
    assert chunk_size > 0
    errors: List[str] = []
    for batch_i, rollout_batch in enumerate(rollout_batches):
        samples = list(expand_rollout_batch(dict(rollout_batch)))
        n = len(samples)
        if n % chunk_size != 0:
            errors.append(f"batch{batch_i}: n={n} not divisible by chunk_size={chunk_size}")
            continue
        for chunk_i in range(0, n, chunk_size):
            chunk = samples[chunk_i:chunk_i + chunk_size]
            group_ids = [_scalar_id(sample["group_id"]) for sample in chunk]
            if any(gid != group_ids[0] for gid in group_ids[1:]):
                errors.append(
                    f"batch{batch_i} chunk{chunk_i // chunk_size}: mixed group_id {group_ids}"
                )
    return errors


def assert_grpo_positional_groups(
    rollout_batches: Sequence[Mapping[str, List[Any]]],
    chunk_size: int,
) -> None:
    errors = grpo_positional_group_errors(rollout_batches, chunk_size)
    assert not errors, "TQ restore GRPO positional groups are mixed: " + "; ".join(errors)


def dump_tq_sample_hashes(
    dump_dir: str,
    *,
    stage: str,
    ppo_step: int,
    shard: str,
    samples: Mapping[str, Dict[str, Dict[str, Any]]],
    order: Optional[Sequence[str]] = None,
    microbatches: Optional[Sequence[Sequence[str]]] = None,
    field_catalog: Optional[Mapping[str, Any]] = None,
    group_chunk_errors: Optional[Sequence[str]] = None,
) -> Path:
    """Write one shard JSON; filename encodes stage/step for later glob."""
    assert stage in ("put", "restore")
    assert shard
    root = Path(dump_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{stage}_step{ppo_step}_{shard}.json"
    payload = {
        "stage": stage,
        "ppo_step": ppo_step,
        "shard": shard,
        "order": list(order) if order is not None else list(samples),
        "microbatches": [list(mb) for mb in microbatches] if microbatches is not None else [],
        "field_catalog": dict(field_catalog) if field_catalog is not None else {},
        "group_chunk_errors": list(group_chunk_errors) if group_chunk_errors is not None else [],
        "samples": {
            key: fields
            for key, fields in samples.items()
        },
    }
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)
    log(f"[TQ-hash] dumped stage={stage} ppo_step={ppo_step} n={len(samples)} path={path}")
    return path


def maybe_dump_tq_sample_hashes(
    config: Any,
    rollout_batches: Sequence[Mapping[str, List[Any]]],
    *,
    stage: str,
    ppo_step: int,
    shard: str,
    chunk_size: Optional[int] = None,
) -> Optional[Path]:
    debug = config.debug
    if not debug.dump_tq_sample_hashes:
        return None
    layout = hash_rollout_layout(rollout_batches, chunk_size=chunk_size)
    return dump_tq_sample_hashes(
        debug.tq_sample_hash_dir,
        stage=stage,
        ppo_step=ppo_step,
        shard=shard,
        samples=layout.samples,
        order=layout.order,
        microbatches=layout.microbatches,
        field_catalog=layout.field_catalog,
        group_chunk_errors=layout.group_chunk_errors,
    )


def load_stage_hashes(
    dump_dir: str,
    stage: str,
    ppo_step: int,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Union all shards for ``stage`` at ``ppo_step``; duplicate keys must match."""
    return load_stage_hash_dump(dump_dir, stage, ppo_step).samples


def load_stage_hash_dump(
    dump_dir: str,
    stage: str,
    ppo_step: int,
) -> "StageHashDump":
    """Union shards and concatenate per-shard microbatch identity lists.

    ``field_catalog`` is rebuilt from the merged sample hashes so put worker
    shards and restore DP shards compare without partition-tree artifacts.
    """
    root = Path(dump_dir)
    pattern = f"{stage}_step{ppo_step}_*.json"
    keyed: Dict[str, Dict[str, Dict[str, Any]]] = {}
    order: List[str] = []
    microbatches: List[List[str]] = []
    group_chunk_errors: List[str] = []
    matches = sorted(root.glob(pattern))
    assert matches, f"no {stage} hash shards matching {root / pattern}"
    for path in matches:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["stage"] == stage
        assert int(payload["ppo_step"]) == ppo_step
        for key, fields in payload["samples"].items():
            if key in keyed:
                assert keyed[key] == fields, f"conflicting hashes for {key!r} in {path}"
                continue
            keyed[key] = fields
        order.extend(payload.get("order") or [])
        for microbatch in payload.get("microbatches") or []:
            microbatches.append(list(microbatch))
        group_chunk_errors.extend(payload.get("group_chunk_errors") or [])
    return StageHashDump(
        samples=keyed,
        order=order,
        microbatches=microbatches,
        field_catalog=catalog_from_sample_hashes(keyed),
        group_chunk_errors=group_chunk_errors,
    )


@dataclass
class SampleHashCompareResult:
    missing_in_restore: List[str] = field(default_factory=list)
    missing_in_put: List[str] = field(default_factory=list)
    field_mismatch: Dict[str, List[str]] = field(default_factory=dict)
    unread_mismatch: Dict[str, List[str]] = field(default_factory=dict)
    matched_samples: int = 0
    order_mismatch: List[List[str]] = field(default_factory=list)
    microbatch_membership_mismatch: List[List[str]] = field(default_factory=list)
    group_chunk_errors: List[str] = field(default_factory=list)
    catalog_extra_in_restore: List[str] = field(default_factory=list)
    catalog_missing_in_restore: List[str] = field(default_factory=list)
    catalog_dtype_mismatch: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return (
            not self.missing_in_restore and not self.missing_in_put and not self.field_mismatch and
            not self.unread_mismatch and not self.order_mismatch and
            not self.microbatch_membership_mismatch and not self.group_chunk_errors and
            not self.catalog_dtype_mismatch
        )


def compare_sample_hashes(
    put_hashes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    restore_hashes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    allow_mismatch: Iterable[str] = (),
    put_microbatches: Optional[Sequence[Sequence[str]]] = None,
    restore_microbatches: Optional[Sequence[Sequence[str]]] = None,
    put_catalog: Optional[Mapping[str, Any]] = None,
    restore_catalog: Optional[Mapping[str, Any]] = None,
    put_group_chunk_errors: Optional[Sequence[str]] = None,
    restore_group_chunk_errors: Optional[Sequence[str]] = None,
) -> SampleHashCompareResult:
    """Join put/restore by sample identity, then optional microbatch order.

    Unread logprob columns always fail on mismatch. Other hashed columns fail
    unless named in ``allow_mismatch`` (Controller-mutated whitelist fields).
    Identity join still passes if rows are permuted; pass both microbatch
    lists to fail when a restore batch is a permutation of a put batch.
    """
    allowed = frozenset(allow_mismatch)
    assert allowed.isdisjoint(UNREAD_TQ_SAMPLE_HASH_FIELDS), (
        "unread TQ fields cannot be allow-listed: "
        f"{sorted(allowed & UNREAD_TQ_SAMPLE_HASH_FIELDS)}"
    )
    result = SampleHashCompareResult()
    put_keys = set(put_hashes)
    restore_keys = set(restore_hashes)
    result.missing_in_restore = sorted(put_keys - restore_keys)
    result.missing_in_put = sorted(restore_keys - put_keys)
    for key in sorted(put_keys & restore_keys):
        put_fields = put_hashes[key]
        restore_fields = restore_hashes[key]
        sample_ok = True
        for field_name, put_value in put_fields.items():
            restore_value = restore_fields.get(field_name)
            if restore_value == put_value:
                continue
            sample_ok = False
            bucket = (
                result.unread_mismatch
                if field_name in UNREAD_TQ_SAMPLE_HASH_FIELDS else result.field_mismatch
            )
            if field_name in allowed:
                continue
            bucket.setdefault(field_name, []).append(key)
        if sample_ok:
            result.matched_samples += 1
    if put_microbatches and restore_microbatches:
        _compare_microbatch_order(
            put_microbatches,
            restore_microbatches,
            result,
        )
    result.group_chunk_errors = [
        *(put_group_chunk_errors or ()),
        *(restore_group_chunk_errors or ()),
    ]
    if put_catalog is not None and restore_catalog is not None:
        put_fields = set(put_catalog)
        restore_fields = set(restore_catalog)
        result.catalog_extra_in_restore = sorted(restore_fields - put_fields)
        result.catalog_missing_in_restore = sorted(put_fields - restore_fields)
        for field_name in sorted(put_fields & restore_fields):
            if put_catalog[field_name] != restore_catalog[field_name]:
                result.catalog_dtype_mismatch[field_name] = {
                    "put": put_catalog[field_name],
                    "restore": restore_catalog[field_name],
                }
    return result


def _compare_microbatch_order(
    put_microbatches: Sequence[Sequence[str]],
    restore_microbatches: Sequence[Sequence[str]],
    result: SampleHashCompareResult,
) -> None:
    put_remaining = Counter(tuple(mb) for mb in put_microbatches)
    for restore_mb in restore_microbatches:
        restore_seq = tuple(restore_mb)
        if put_remaining[restore_seq]:
            put_remaining[restore_seq] -= 1
            continue
        matched_set = False
        restore_set = set(restore_seq)
        for put_seq, count in list(put_remaining.items()):
            if count and len(put_seq) == len(restore_seq) and set(put_seq) == restore_set:
                put_remaining[put_seq] -= 1
                result.order_mismatch.append(list(restore_seq))
                matched_set = True
                break
        if not matched_set:
            result.microbatch_membership_mismatch.append(list(restore_seq))
    leftover = [list(seq) for seq, count in put_remaining.items() for _ in range(count)]
    result.microbatch_membership_mismatch.extend(leftover)


def _tensor_payload(tensor: torch.Tensor) -> Tuple[bytes, str, List[int]]:
    flat = tensor.detach().cpu().contiguous()
    dtype = str(flat.dtype).replace("torch.", "")
    shape = list(flat.shape)
    if flat.dtype == torch.bfloat16:
        payload = flat.view(torch.uint16).numpy().tobytes()
    else:
        payload = flat.numpy().tobytes()
    return payload, dtype, shape
