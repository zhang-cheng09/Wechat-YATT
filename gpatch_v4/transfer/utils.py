"""Reusable TransferQueue encode / decode / orchestration helpers.

External callers should use :class:`gpatch_v4.transfer.transfer_queue_interface.TQInterface`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tensordict import TensorDict

from gpatch_v4.transfer import get_tq_connector
from gpatch_v4.transfer.tensordict_utils import (
    TqPayloadType,
    data_to_tensordict,
    list_dict_to_tensordict,
    tensordict_to_data,
    tensordict_to_list_dict,
)
from gpatch_v4.transfer.tq_connector import KVBatchMeta, TqConnector

_TQ_DATA_SLOT_FIELD = "_tq_data_slot"
_TQ_TAG_METADATA = "metadata"
_TQ_TAG_PPO_STEP = "ppo_step"
_TQ_TAG_STORED_FIELDS = "stored_fields"

__all__ = [
    "TqDataSlot",
    "TqPayloadType",
    "TqSampleOperateType",
    "get_tq_data_slot",
]


@dataclass(frozen=True)
class TqDataSlot:
    """Picklable reference and codec type for values stored in TransferQueue."""

    keys: Tuple[str, ...]
    payload_type: TqPayloadType


class TqSampleOperateType(str, Enum):
    FIRST_PUT = "first_put"
    UPDATE = "update"


def _resolve_connector(connector: Optional[TqConnector]) -> TqConnector:
    return connector if connector is not None else get_tq_connector()


def get_tq_data_slot(batched_data: Dict[str, Any]) -> Optional[TqDataSlot]:
    """Return the typed TQ slot embedded in ``batched_data``."""
    slots = [value for value in batched_data.values() if isinstance(value, TqDataSlot)]
    assert len(slots) <= 1, f"expected at most one TqDataSlot, got {len(slots)}"
    return slots[0] if slots else None


def _tag_parts(tag: Dict[str, Any]) -> Tuple[int, Tuple[str, ...], Dict[str, Any]]:
    ppo_step = tag[_TQ_TAG_PPO_STEP]
    stored_fields = tuple(tag[_TQ_TAG_STORED_FIELDS])
    metadata = tag[_TQ_TAG_METADATA]
    assert isinstance(ppo_step, int)
    assert stored_fields
    assert isinstance(metadata, dict)
    return ppo_step, stored_fields, metadata


async def async_clear_sample_refs(
    metas: Sequence[KVBatchMeta],
    *,
    connector: Optional[TqConnector] = None,
) -> None:
    """Clear exact consumed keys without touching prefetched rollout steps.

    Parameters
    ----------
    metas : Sequence[KVBatchMeta]
    connector : TqConnector, optional
    """
    connector = _resolve_connector(connector)
    keys: List[str] = []
    for meta in metas:
        assert isinstance(meta, KVBatchMeta)
        keys.extend(meta.keys)
    assert len(keys) == len(set(keys)), "duplicate TQ keys in clear metadata"
    if keys:
        await connector.async_clear(keys)


def _contains_array_payload(value: Any) -> bool:
    return isinstance(value, (torch.Tensor, np.ndarray))


def _validate_lightweight_metadata(
    metadata: Sequence[Dict[str, Any]], stored_fields: Sequence[str]
) -> None:
    stored = set(stored_fields)
    for sample_metadata in metadata:
        for field, value in sample_metadata.items():
            if field in stored:
                assert isinstance(value, dict)
                continue
            assert not _contains_array_payload(value), (
                f"tensor field {field!r} was not encoded into TransferQueue"
            )


async def _async_first_put_samples(
    samples: List[Dict[str, Any]],
    ppo_step: int,
    connector: TqConnector,
) -> KVBatchMeta:
    tensor_dict, metadata = list_dict_to_tensordict(samples)
    stored_fields = tuple(sorted(str(field) for field in tensor_dict.keys()))
    assert stored_fields, "rollout samples contain no TransferQueue-compatible fields"
    _validate_lightweight_metadata(metadata, stored_fields)

    keys = connector.make_keys(len(samples), ppo_step=ppo_step)
    meta = await connector.async_put(tensor_dict, keys, ppo_step)
    assert list(meta.keys) == keys
    return KVBatchMeta(
        keys=list(meta.keys),
        tags=[
            {
                _TQ_TAG_PPO_STEP: ppo_step,
                _TQ_TAG_STORED_FIELDS: stored_fields,
                _TQ_TAG_METADATA: sample_metadata,
            } for sample_metadata in metadata
        ],
        partition_id=meta.partition_id,
        fields=list(stored_fields),
        extra_info=meta.extra_info,
    )


def _metadata_for_field_selection(
    metadata: Dict[str, Any],
    stored_fields: Sequence[str],
    fields: Optional[Sequence[str]],
) -> Dict[str, Any]:
    """Keep selected tensor codecs and control-plane values; drop unused codecs."""
    if fields is None:
        return metadata
    selected = set(fields)
    stored = set(stored_fields)
    assert selected.issubset(stored
                            ), (f"select fields not stored on sample: {sorted(selected - stored)}")
    return {key: value for key, value in metadata.items() if key in selected or key not in stored}


def _requested_stored_fields(
    fields: Optional[Sequence[str]],
    stored_fields: Sequence[str],
) -> Optional[List[str]]:
    if fields is None:
        return None
    stored = set(stored_fields)
    return [field for field in fields if field in stored]


def _empty_row_tensor_dict(n_rows: int) -> TensorDict:
    return TensorDict({}, batch_size=[n_rows])


def _split_restored_kv_batches(
    samples: Sequence[Dict[str, Any]],
    batch_sizes: Sequence[int],
) -> List[Dict[str, List[Any]]]:
    batches = []
    offset = 0
    for batch_size in batch_sizes:
        batch_samples = samples[offset:offset + batch_size]
        assert len(batch_samples) == batch_size and batch_samples
        sample_fields = set(batch_samples[0])
        assert all(set(sample) == sample_fields for sample in batch_samples)
        batches.append(
            {field: [sample[field] for sample in batch_samples]
             for field in sorted(sample_fields)}
        )
        offset += batch_size
    assert offset == len(samples)
    return batches


def _flatten_kv_batch_metas(
    metas: Sequence[KVBatchMeta],
) -> Tuple[List[str], List[Dict[str, Any]], List[int], Tuple[str, ...]]:
    assert metas
    schemas = set()
    keys = []
    tags = []
    batch_sizes = []
    for meta in metas:
        assert isinstance(meta, KVBatchMeta)
        assert meta.keys and len(meta.keys) == len(meta.tags)
        assert meta.fields is not None
        schema = tuple(sorted(meta.fields))
        schemas.add(schema)
        for tag in meta.tags:
            _, stored_fields, _ = _tag_parts(tag)
            assert tuple(sorted(stored_fields)) == schema
        keys.extend(meta.keys)
        tags.extend(meta.tags)
        batch_sizes.append(len(meta.keys))
    assert len(schemas) == 1, f"KVBatchMeta schemas differ: {sorted(schemas)}"
    return keys, tags, batch_sizes, schemas.pop()


def concat_kv_batch_metas(metas: Sequence[KVBatchMeta]) -> KVBatchMeta:
    """Concatenate homogeneous metadata batches without changing row order.

    Parameters
    ----------
    metas : Sequence[KVBatchMeta]
        Non-empty batches from one partition with an identical stored schema.

    Returns
    -------
    KVBatchMeta
        Keys and row tags flattened in input-batch order.
    """
    keys, tags, _, schema = _flatten_kv_batch_metas(metas)
    partition_ids = {meta.partition_id for meta in metas}
    assert len(partition_ids) == 1, f"KVBatchMeta partitions differ: {partition_ids}"
    return KVBatchMeta(
        keys=keys,
        tags=tags,
        partition_id=partition_ids.pop(),
        fields=list(schema),
        extra_info=metas[0].extra_info,
    )


def subset_kv_batch_meta(
    meta: KVBatchMeta,
    indices: Sequence[int],
) -> KVBatchMeta:
    """Select metadata rows by position in the requested order.

    Parameters
    ----------
    meta : KVBatchMeta
    indices : Sequence[int]
        Unique row positions; reordering is allowed.

    Returns
    -------
    KVBatchMeta
        A view-like metadata copy retaining the original partition and schema.
    """
    selected = list(indices)
    assert selected
    assert len(set(selected)) == len(selected), f"duplicate KVBatchMeta indices: {selected}"
    assert min(selected) >= 0 and max(selected) < len(
        meta.keys
    ), (f"KVBatchMeta indices out of range for {len(meta.keys)} rows: {selected}")
    return KVBatchMeta(
        keys=[meta.keys[index] for index in selected],
        tags=[meta.tags[index] for index in selected],
        partition_id=meta.partition_id,
        fields=None if meta.fields is None else list(meta.fields),
        extra_info=meta.extra_info,
    )


def _decode_kv_batches(
    tensor_dict: TensorDict,
    tags: Sequence[Dict[str, Any]],
    batch_sizes: Sequence[int],
    stored_fields: Sequence[str],
    get_fields: Optional[Sequence[str]],
) -> List[Dict[str, List[Any]]]:
    fallback = []
    for tag in tags:
        _, tag_stored_fields, metadata = _tag_parts(tag)
        assert tuple(sorted(tag_stored_fields)) == tuple(sorted(stored_fields))
        fallback.append(_metadata_for_field_selection(metadata, stored_fields, get_fields))
    samples = tensordict_to_list_dict(tensor_dict, fallback)
    return _split_restored_kv_batches(
        samples,
        batch_sizes,
    )


async def async_restore_kv_batches(
    metas: Sequence[KVBatchMeta],
    *,
    fields: Optional[Sequence[str]] = None,
    connector: Optional[TqConnector] = None,
) -> List[Dict[str, List[Any]]]:
    """Restore homogeneous KV metadata batches as field-major dictionaries.

    Parameters
    ----------
    metas : Sequence[KVBatchMeta]
        Input boundaries and row order are preserved in the returned list.
        Every batch must carry the same complete stored-field schema.
    fields : Sequence[str], optional
        Tensor columns to fetch; non-tensor values in row tags remain available.
    connector : TqConnector, optional

    Returns
    -------
    list[dict[str, list[Any]]]
        One field-major dictionary per input metadata batch.
    """
    if not metas:
        return []
    connector = _resolve_connector(connector)
    keys, tags, batch_sizes, stored_fields = _flatten_kv_batch_metas(metas)
    get_fields = _requested_stored_fields(fields, stored_fields)
    if get_fields is not None and not get_fields:
        tensor_dict = _empty_row_tensor_dict(len(keys))
    else:
        tensor_dict = await connector.async_get(keys, fields=get_fields)
    return _decode_kv_batches(
        tensor_dict,
        tags,
        batch_sizes,
        stored_fields,
        get_fields,
    )


def restore_kv_batches(
    metas: Sequence[KVBatchMeta],
    *,
    fields: Optional[Sequence[str]] = None,
    connector: Optional[TqConnector] = None,
) -> List[Dict[str, List[Any]]]:
    """Synchronously restore homogeneous KV metadata batches.

    Parameters
    ----------
    metas : Sequence[KVBatchMeta]
        Input boundaries and row order are preserved in the returned list.
        Every batch must carry the same complete stored-field schema.
    fields : Sequence[str], optional
        Tensor columns to fetch; non-tensor values in row tags remain available.
    connector : TqConnector, optional

    Returns
    -------
    list[dict[str, list[Any]]]
        One field-major dictionary per input metadata batch.
    """
    if not metas:
        return []
    connector = _resolve_connector(connector)
    keys, tags, batch_sizes, stored_fields = _flatten_kv_batch_metas(metas)
    get_fields = _requested_stored_fields(fields, stored_fields)
    if get_fields is not None and not get_fields:
        tensor_dict = _empty_row_tensor_dict(len(keys))
    else:
        tensor_dict = connector.get(keys, fields=get_fields)
    return _decode_kv_batches(
        tensor_dict,
        tags,
        batch_sizes,
        stored_fields,
        get_fields,
    )


def _tensor_state(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, torch.Tensor):
        return ("torch", id(value), value._version)
    if isinstance(value, np.ndarray):
        return ("numpy", id(value))
    if value is None:
        return ("none", )
    return ("other", type(value))


def capture_tq_sample_tensor_state(
    samples: Sequence[Dict[str, Any]],
    metas: Sequence[KVBatchMeta],
) -> Dict[str, Dict[str, Tuple[Any, ...]]]:
    """Capture enough state to avoid rewriting unchanged stored tensors.

    Parameters
    ----------
    samples : Sequence[dict[str, Any]]
        Materialized rows aligned with the flattened metadata batches.
    metas : Sequence[KVBatchMeta]

    Returns
    -------
    dict[str, dict[str, tuple[Any, ...]]]
        Torch entries include mutation versions; NumPy entries are later
        rewritten conservatively because arrays expose no mutation version.
    """
    meta = concat_kv_batch_metas(metas)
    assert len(samples) == len(
        meta.keys
    ), (f"sample/meta row count mismatch: {len(samples)} != {len(meta.keys)}")
    state = {}
    for sample, key, tag in zip(samples, meta.keys, meta.tags, strict=True):
        _, stored_fields, _ = _tag_parts(tag)
        assert key not in state
        state[key] = {
            field: _tensor_state(sample[field])
            for field in stored_fields if field in sample
        }
    return state


def _collect_tq_update_fields(
    samples: Sequence[Dict[str, Any]],
    keys: Sequence[str],
    stored_fields: Sequence[str],
    original_state: Dict[str, Dict[str, Tuple[Any, ...]]],
) -> List[str]:
    """Return dirty stored columns and new array columns for a whole-batch update."""
    update_fields = []
    # 1. 旧 schema 里已有、被物化并且被更新过的列
    for field in stored_fields:
        field_presence = [field in sample for sample in samples]
        assert all(field_presence) or not any(field_presence), (
            f"stored TQ field {field!r} must be materialized for every sample or none"
        )
        if not any(field_presence):
            continue
        values = [sample[field] for sample in samples]
        assert all(value is None or _contains_array_payload(value)
                   for value in values), (f"stored TQ field {field!r} has non-array values")
        field_is_dirty = any(
            field not in original_state[key] or isinstance(value, np.ndarray) or
            _tensor_state(value) != original_state[key][field]
            for key, value in zip(keys, values, strict=True)
        )
        if field_is_dirty:
            assert any(_contains_array_payload(value) for value in values
                      ), (f"stored TQ field {field!r} cannot be updated to an all-None column")
            update_fields.append(field)

    # 2. schema 里没有的新 array 列
    sample_fields = set().union(*(sample.keys() for sample in samples))
    for field in sorted(sample_fields - set(stored_fields)):
        values = [sample.get(field) for sample in samples]
        if not any(_contains_array_payload(value) for value in values):
            continue
        assert all(field in sample for sample in samples
                  ), (f"new TQ field {field!r} must be present in every sample")
        assert all(value is None or _contains_array_payload(value)
                   for value in values), (f"new TQ field {field!r} has non-array values")
        update_fields.append(field)
    return sorted(update_fields)


def _rebuild_tq_update_tags(
    samples: Sequence[Dict[str, Any]],
    tags: Sequence[Dict[str, Any]],
    *,
    ppo_step: int,
    stored_fields: Sequence[str],
    new_stored_fields: Sequence[str],
    updated_field_metadata: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rebuild row tags after a whole-batch TQ update."""
    assert len(samples) == len(tags) == len(updated_field_metadata)
    updated_tags = []
    for sample, tag, field_metadata_by_name in zip(
        samples,
        tags,
        updated_field_metadata,
        strict=True,
    ):
        row_ppo_step, old_stored_fields, old_metadata = _tag_parts(tag)
        assert row_ppo_step == ppo_step
        assert old_stored_fields == tuple(stored_fields)
        metadata = {
            field: value
            for field, value in sample.items() if field not in new_stored_fields
        }
        for field in new_stored_fields:
            if field in field_metadata_by_name:
                metadata[field] = field_metadata_by_name[field]
            else:
                field_metadata = old_metadata[field]
                assert isinstance(field_metadata, dict)
                metadata[field] = field_metadata
        _validate_lightweight_metadata([metadata], new_stored_fields)
        updated_tags.append(
            {
                _TQ_TAG_PPO_STEP: ppo_step,
                _TQ_TAG_STORED_FIELDS: tuple(new_stored_fields),
                _TQ_TAG_METADATA: metadata,
            }
        )
    return updated_tags


async def _async_update_samples(
    samples: List[Dict[str, Any]],
    meta: KVBatchMeta,
    original_state: Dict[str, Dict[str, Tuple[Any, ...]]],
    connector: TqConnector,
) -> KVBatchMeta:
    assert samples
    assert len(samples) == len(meta.keys) == len(meta.tags)
    assert meta.fields is not None
    assert meta.partition_id == connector.tq_config.partition_id

    ppo_steps = set()
    schemas = set()
    seen_keys = set()
    for sample, key, tag in zip(samples, meta.keys, meta.tags, strict=True):
        ppo_step, stored_fields, _ = _tag_parts(tag)
        assert key in original_state, f"unknown TQ sample key {key!r}"
        assert key not in seen_keys, f"duplicate TQ sample key {key!r}"
        seen_keys.add(key)
        ppo_steps.add(ppo_step)
        schemas.add(stored_fields)
        loaded_fields = original_state[key]
        assert all(field in sample
                   for field in loaded_fields), (f"sample {key!r} dropped loaded TQ fields")

    assert len(ppo_steps) == 1, f"sample ppo_steps differ: {sorted(ppo_steps)}"
    assert len(schemas) == 1, f"sample schemas differ: {sorted(schemas)}"
    ppo_step = ppo_steps.pop()
    stored_fields = schemas.pop()
    assert tuple(sorted(meta.fields)) == stored_fields

    update_fields = _collect_tq_update_fields(
        samples,
        meta.keys,
        stored_fields,
        original_state,
    )
    new_stored_fields = tuple(sorted(set(stored_fields).union(update_fields)))
    updated_field_metadata: List[Dict[str, Any]] = [{} for _ in samples]
    if update_fields:
        tensor_dict, metadata = list_dict_to_tensordict(
            samples,
            fields=update_fields,
        )
        assert set(tensor_dict.keys()) == set(update_fields), (
            f"failed to encode TQ update fields: expected {update_fields}, "
            f"got {tuple(tensor_dict.keys())}"
        )
        await connector.async_put(
            tensor_dict,
            meta.keys,
            ppo_step,
            schema_fields=new_stored_fields,
        )
        for index, sample_metadata in enumerate(metadata):
            for field in update_fields:
                field_metadata = sample_metadata[field]
                assert isinstance(field_metadata, dict)
                updated_field_metadata[index][field] = field_metadata

    return KVBatchMeta(
        keys=list(meta.keys),
        tags=_rebuild_tq_update_tags(
            samples,
            meta.tags,
            ppo_step=ppo_step,
            stored_fields=stored_fields,
            new_stored_fields=new_stored_fields,
            updated_field_metadata=updated_field_metadata,
        ),
        partition_id=meta.partition_id,
        fields=list(new_stored_fields),
        extra_info=meta.extra_info,
    )


async def async_put_samples(
    samples: List[Dict[str, Any]],
    operate_type: TqSampleOperateType,
    *,
    ppo_step: Optional[int] = None,
    meta: Optional[KVBatchMeta] = None,
    original_state: Optional[Dict[str, Dict[str, Tuple[Any, ...]]]] = None,
    connector: Optional[TqConnector] = None,
) -> KVBatchMeta:
    """Create or update independently addressable TQ sample rows.

    Parameters
    ----------
    samples : list[dict[str, Any]]
        Rows aligned with ``meta`` for updates.
    operate_type : TqSampleOperateType
        Selects key creation/full encoding or existing-key differential update.
    ppo_step : int, optional
        Required only for ``FIRST_PUT``.
    meta : KVBatchMeta, optional
        Existing rows aligned one-to-one with ``samples`` for ``UPDATE``.
    original_state : dict[str, dict[str, tuple[Any, ...]]], optional
        Required only for ``UPDATE`` to skip unchanged materialized tensors.
    connector : TqConnector, optional

    Returns
    -------
    KVBatchMeta
        Complete keys, refreshed row tags, and the homogeneous stored schema.
    """
    connector = _resolve_connector(connector)
    if operate_type is TqSampleOperateType.FIRST_PUT:
        assert isinstance(ppo_step, int)
        assert meta is None
        assert original_state is None
        return await _async_first_put_samples(samples, ppo_step, connector)
    if operate_type is TqSampleOperateType.UPDATE:
        assert ppo_step is None
        assert isinstance(meta, KVBatchMeta)
        assert original_state is not None
        return await _async_update_samples(samples, meta, original_state, connector)
    raise ValueError(f"unsupported sample TQ operate type: {operate_type!r}")


async def async_offload_to_tq(
    data: Dict[str, Any],
    ppo_step: int,
    payload_type: TqPayloadType,
    fields: Optional[List[str]] = None,
    *,
    connector: Optional[TqConnector] = None,
) -> Dict[str, Any]:
    """Offload selected fields and embed a self-describing TQ slot."""
    connector = _resolve_connector(connector)
    tensor_dict, fallback = data_to_tensordict(
        data,
        payload_type,
        fields=fields,
    )
    if len(tensor_dict.keys()) == 0:
        return fallback

    assert _TQ_DATA_SLOT_FIELD not in fallback
    keys = tuple(connector.make_keys(
        int(tensor_dict.batch_size[0]),
        ppo_step=ppo_step,
    ))
    await connector.async_put(tensor_dict, keys, ppo_step)

    result: Dict[str, Any] = dict(fallback)
    result[_TQ_DATA_SLOT_FIELD] = TqDataSlot(
        keys=keys,
        payload_type=payload_type,
    )
    return result


async def async_restore_from_tq(
    data: Dict[str, Any],
    *,
    connector: Optional[TqConnector] = None,
) -> Dict[str, Any]:
    """Restore a payload according to the type recorded in its TQ slot."""
    connector = _resolve_connector(connector)
    slot = get_tq_data_slot(data)
    if slot is None:
        return data

    fallback = {field: value for field, value in data.items() if not isinstance(value, TqDataSlot)}
    tensor_dict = await connector.async_get(slot.keys)
    return tensordict_to_data(
        tensor_dict,
        fallback,
        slot.payload_type,
    )


def restore_from_tq(
    data: Dict[str, Any],
    *,
    connector: Optional[TqConnector] = None,
) -> Dict[str, Any]:
    """Synchronously restore a payload according to its recorded type."""
    connector = _resolve_connector(connector)
    slot = get_tq_data_slot(data)
    if slot is None:
        return data

    fallback = {field: value for field, value in data.items() if not isinstance(value, TqDataSlot)}
    tensor_dict = connector.get(slot.keys)
    return tensordict_to_data(
        tensor_dict,
        fallback,
        slot.payload_type,
    )


def restore_tq_payloads_into_samples(
    samples: List[Dict[str, Any]],
    payload_field: str,
    *,
    connector: Optional[TqConnector] = None,
) -> None:
    """Restore one complete TQ payload into each sample in-place."""
    connector = _resolve_connector(connector)
    for sample in samples:
        payload = sample[payload_field]
        assert isinstance(
            payload, dict
        ), f"sample[{payload_field!r}] must be a dict, got {type(payload)}"
        restored = restore_from_tq(payload, connector=connector)
        sample.update(restored)
