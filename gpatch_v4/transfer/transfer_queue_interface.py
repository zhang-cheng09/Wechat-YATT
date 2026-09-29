"""External TransferQueue orchestration API.

Callers should use :class:`TQInterface` instead of importing helpers from
``gpatch_v4.transfer.utils``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from gpatch_v4.transfer import utils as _tq_utils
from gpatch_v4.transfer.tensordict_utils import TqPayloadType
from gpatch_v4.transfer.tq_connector import KVBatchMeta, TqConnector
from gpatch_v4.transfer.utils import TqDataSlot, TqSampleOperateType, get_tq_data_slot

__all__ = [
    "TQInterface",
    "TqDataSlot",
    "TqPayloadType",
    "TqSampleOperateType",
    "get_tq_data_slot",
]


class TQInterface:
    """Process-facing TransferQueue put / get / update / clear entrypoints."""
    @staticmethod
    async def async_put_samples(
        samples: List[Dict[str, Any]],
        operate_type: TqSampleOperateType,
        *,
        ppo_step: Optional[int] = None,
        meta: Optional[KVBatchMeta] = None,
        original_state: Optional[Dict[str, Dict[str, Tuple[Any, ...]]]] = None,
        connector: Optional[TqConnector] = None,
    ) -> KVBatchMeta:
        """Create sample rows (``FIRST_PUT``) or rewrite dirty columns (``UPDATE``).

        Parameters
        ----------
        samples : list[dict[str, Any]]
            One dict per row; for ``UPDATE`` MUST align 1:1 with ``meta.keys``.
        operate_type : TqSampleOperateType
            ``FIRST_PUT`` allocates new keys; ``UPDATE`` reuses ``meta.keys``.
        ppo_step : int, optional
            Required for ``FIRST_PUT``; MUST be ``None`` for ``UPDATE``.
        meta : KVBatchMeta, optional
            Required for ``UPDATE``; MUST be ``None`` for ``FIRST_PUT``.
        original_state : dict[str, dict[str, tuple[Any, ...]]], optional
            From :meth:`capture_tq_sample_tensor_state`; required for ``UPDATE``
            so unchanged tensor columns are skipped.
        connector : TqConnector, optional

        Returns
        -------
        KVBatchMeta
            Keys, refreshed row tags, and the homogeneous stored-field schema.
        """
        return await _tq_utils.async_put_samples(
            samples,
            operate_type,
            ppo_step=ppo_step,
            meta=meta,
            original_state=original_state,
            connector=connector,
        )

    @staticmethod
    async def async_restore_kv_batches(
        metas: Sequence[KVBatchMeta],
        *,
        fields: Optional[Sequence[str]] = None,
        connector: Optional[TqConnector] = None,
    ) -> List[Dict[str, List[Any]]]:
        """Fetch TQ rows and rebuild field-major microbatches.

        Parameters
        ----------
        metas : Sequence[KVBatchMeta]
            Output list preserves this order and each batch's row boundaries.
            All batches MUST share one stored-field schema.
        fields : Sequence[str], optional
            Tensor columns to fetch; ``None`` fetches the full stored schema.
            Non-tensor tag metadata is always available on restored rows.
        connector : TqConnector, optional

        Returns
        -------
        list[dict[str, list[Any]]]
            One field-major dict per input meta (``field -> list over rows``).
        """
        return await _tq_utils.async_restore_kv_batches(
            metas,
            fields=fields,
            connector=connector,
        )

    @staticmethod
    def restore_kv_batches(
        metas: Sequence[KVBatchMeta],
        *,
        fields: Optional[Sequence[str]] = None,
        connector: Optional[TqConnector] = None,
    ) -> List[Dict[str, List[Any]]]:
        """Sync counterpart of :meth:`async_restore_kv_batches`.

        Parameters
        ----------
        metas : Sequence[KVBatchMeta]
            Same contract as the async restore.
        fields : Sequence[str], optional
            Same contract as the async restore.
        connector : TqConnector, optional

        Returns
        -------
        list[dict[str, list[Any]]]
            One field-major dict per input meta.
        """
        return _tq_utils.restore_kv_batches(
            metas,
            fields=fields,
            connector=connector,
        )

    @staticmethod
    async def async_clear_sample_refs(
        metas: Sequence[KVBatchMeta],
        *,
        connector: Optional[TqConnector] = None,
    ) -> None:
        """Delete exactly the keys listed in ``metas`` (no prefetch side effects).

        Parameters
        ----------
        metas : Sequence[KVBatchMeta]
            Keys across metas MUST be unique; duplicates assert.
        connector : TqConnector, optional
        """
        await _tq_utils.async_clear_sample_refs(metas, connector=connector)

    @staticmethod
    def capture_tq_sample_tensor_state(
        samples: Sequence[Dict[str, Any]],
        metas: Sequence[KVBatchMeta],
    ) -> Dict[str, Dict[str, Tuple[Any, ...]]]:
        """Snapshot materialized tensors for later dirty-column ``UPDATE``.

        Parameters
        ----------
        samples : Sequence[dict[str, Any]]
            Row-major samples aligned with ``concat`` of ``metas`` (same length).
        metas : Sequence[KVBatchMeta]
            Homogeneous batches; flattened in order before pairing with samples.

        Returns
        -------
        dict[str, dict[str, tuple[Any, ...]]]
            ``tq_key -> field -> state``. Torch uses mutation version; NumPy is
            treated conservatively (no stable mutation version).
        """
        return _tq_utils.capture_tq_sample_tensor_state(samples, metas)

    @staticmethod
    def concat_kv_batch_metas(metas: Sequence[KVBatchMeta]) -> KVBatchMeta:
        """Flatten microbatch metas into one meta without reordering rows.

        Parameters
        ----------
        metas : Sequence[KVBatchMeta]
            Non-empty; same ``partition_id`` and identical stored-field schema.

        Returns
        -------
        KVBatchMeta
            Keys/tags concatenated in input-batch order; schema unchanged.
        """
        return _tq_utils.concat_kv_batch_metas(metas)

    @staticmethod
    def subset_kv_batch_meta(
        meta: KVBatchMeta,
        indices: Sequence[int],
    ) -> KVBatchMeta:
        """Keep selected rows; order follows ``indices`` (reorder allowed).

        Parameters
        ----------
        meta : KVBatchMeta
        indices : Sequence[int]
            Non-empty, unique, in ``[0, len(meta.keys))``.

        Returns
        -------
        KVBatchMeta
            Same partition/schema; keys/tags sliced by ``indices``.
        """
        return _tq_utils.subset_kv_batch_meta(meta, indices)

    @staticmethod
    async def async_offload_to_tq(
        data: Dict[str, Any],
        ppo_step: int,
        payload_type: TqPayloadType,
        fields: Optional[List[str]] = None,
        *,
        connector: Optional[TqConnector] = None,
    ) -> Dict[str, Any]:
        """Store selected fields in TQ and leave a ``TqDataSlot`` in the dict.

        Parameters
        ----------
        data : dict[str, Any]
            Source payload; non-offloaded fields stay inline in the result.
        ppo_step : int
            Used when allocating TQ keys.
        payload_type : TqPayloadType
            Codec used on restore (recorded inside the slot).
        fields : list[str], optional
            Columns to offload; ``None`` follows payload-type defaults.
        connector : TqConnector, optional

        Returns
        -------
        dict[str, Any]
            Lightweight dict with a ``TqDataSlot`` when anything was stored;
            unchanged ``data``-like fallback when there is nothing to offload.
        """
        return await _tq_utils.async_offload_to_tq(
            data,
            ppo_step,
            payload_type,
            fields=fields,
            connector=connector,
        )

    @staticmethod
    async def async_restore_from_tq(
        data: Dict[str, Any],
        *,
        connector: Optional[TqConnector] = None,
    ) -> Dict[str, Any]:
        """Materialize fields referenced by an embedded ``TqDataSlot``.

        Parameters
        ----------
        data : dict[str, Any]
            If no slot is present, returned as-is.
        connector : TqConnector, optional

        Returns
        -------
        dict[str, Any]
            Decoded tensors merged with non-slot fallback fields.
        """
        return await _tq_utils.async_restore_from_tq(data, connector=connector)

    @staticmethod
    def restore_from_tq(
        data: Dict[str, Any],
        *,
        connector: Optional[TqConnector] = None,
    ) -> Dict[str, Any]:
        """Sync counterpart of :meth:`async_restore_from_tq`.

        Parameters
        ----------
        data : dict[str, Any]
            Same contract as the async restore.
        connector : TqConnector, optional

        Returns
        -------
        dict[str, Any]
            Decoded tensors merged with non-slot fallback fields.
        """
        return _tq_utils.restore_from_tq(data, connector=connector)

    @staticmethod
    def restore_tq_payloads_into_samples(
        samples: List[Dict[str, Any]],
        payload_field: str,
        *,
        connector: Optional[TqConnector] = None,
    ) -> None:
        """In-place expand each sample's nested TQ payload via :meth:`restore_from_tq`.

        Parameters
        ----------
        samples : list[dict[str, Any]]
            Mutated: restored fields are ``update``-merged into each sample.
        payload_field : str
            Sample key whose value is a dict that may embed a ``TqDataSlot``.
        connector : TqConnector, optional
        """
        _tq_utils.restore_tq_payloads_into_samples(
            samples,
            payload_field,
            connector=connector,
        )
