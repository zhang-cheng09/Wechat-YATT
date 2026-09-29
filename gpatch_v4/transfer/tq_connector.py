"""Direct payload adapter around TransferQueue's native KV batch API."""
from __future__ import annotations

import logging
import os
import subprocess
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence
from urllib.parse import urlparse

import torch
from tensordict import TensorDict

try:
    import transfer_queue as tq
    from transfer_queue.metadata import KVBatchMeta
except ImportError:
    tq = None
    KVBatchMeta = None

from gpatch_v4.configs.tq_config import TransferQueueConfig
from gpatch_v4.orches.utils import get_current_node_ip
from gpatch_v4.utils import log

__all__ = ["KVBatchMeta", "TqConnector"]


def _td_nbytes(tensor_dict: TensorDict) -> int:
    total = 0
    for value in tensor_dict.values():
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
        elif hasattr(value, "nbytes"):
            total += int(value.nbytes)
    return total


class TqConnector:
    """Offload and restore payload data through TransferQueue."""
    def __init__(self, tq_config: TransferQueueConfig) -> None:
        self.tq_config = tq_config
        if not self.tq_config.partition_id:
            raise ValueError("tq.partition_id must be set before initializing TqConnector")

        self._apply_transfer_queue_log_level()
        if self.tq_config.backend == "MooncakeStore":
            os.environ["MC_TCP_BIND_ADDRESS"] = get_current_node_ip()
        tq.init(self.tq_config.to_tq_config())
        log(
            f"[TQ] initialized backend={self.tq_config.backend} "
            f"partition_id={self.tq_config.partition_id}"
        )

    def _apply_transfer_queue_log_level(self) -> None:
        """Set only ``transfer_queue*`` loggers; leave gcore / root loggers alone."""
        level_name = self.tq_config.log_level
        os.environ["TQ_LOGGING_LEVEL"] = level_name
        logging.getLogger("transfer_queue").setLevel(level_name)
        for name in logging.root.manager.loggerDict:
            if name == "transfer_queue" or name.startswith("transfer_queue."):
                logging.getLogger(name).setLevel(level_name)

    def close(self) -> None:
        """Close the driver-owned TQ runtime and shared backend services."""
        tq.close()
        if self.tq_config.backend == "MooncakeStore":
            self._terminate_mooncake_processes()

    def _terminate_mooncake_processes(self) -> None:
        """Terminate mooncake_master and metadata_server matching this session's ports."""
        mc = self.tq_config.mooncake
        master_port = urlparse(
            mc.master_server_address if "://" in mc.master_server_address else "//" +
            mc.master_server_address
        ).port
        metadata_port = urlparse(
            mc.metadata_server if "://" in mc.metadata_server else "//" + mc.metadata_server
        ).port

        patterns = []
        if master_port:
            patterns.append(f"[m]ooncake_master.*--rpc_port={master_port}")
        if metadata_port:
            patterns.append(f"[m]ooncake_http_metadata_server.*--port.*{metadata_port}")

        for pattern in patterns:
            subprocess.run(
                ["pkill", "-TERM", "-f", pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        time.sleep(2)
        for pattern in patterns:
            subprocess.run(
                ["pkill", "-KILL", "-f", pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

    @staticmethod
    def _key_prefix(ppo_step: int) -> str:
        return f"step{ppo_step}:uid:"

    @staticmethod
    def make_keys(count: int, *, ppo_step: int) -> List[str]:
        """Generate *count* unique user keys for a given step."""
        prefix = TqConnector._key_prefix(ppo_step)
        return [f"{prefix}{uuid.uuid4().hex}" for _ in range(count)]

    async def async_put(
        self,
        tensor_dict: TensorDict,
        key: str | Sequence[str],
        ppo_step: int,
        schema_fields: Sequence[str] | None = None,
    ) -> KVBatchMeta:
        """Upsert TensorDict rows while retaining their complete field schema.

        Parameters
        ----------
        tensor_dict : TensorDict
            Batch dimension must align one-to-one with ``key``.
        key : str or Sequence[str]
        ppo_step : int
            Rollout-production step retained in backend tags.
        schema_fields : Sequence[str], optional
            Complete post-upsert schema; defaults to fields in ``tensor_dict``.

        Returns
        -------
        KVBatchMeta
            Backend metadata for the affected rows.
        """
        keys = [key] if isinstance(key, str) else list(key)
        assert len(keys) == int(tensor_dict.batch_size[0])
        put_fields = sorted(str(name) for name in tensor_dict.keys())
        field_names = put_fields if schema_fields is None else sorted(set(schema_fields))
        assert set(put_fields).issubset(field_names)
        meta = await tq.async_kv_batch_put(
            keys=keys,
            partition_id=self.tq_config.partition_id,
            fields=tensor_dict,
            tags=[{
                "ppo_step": ppo_step,
                "fields": field_names,
            }] * len(keys),
        )
        log(
            f"[TQ] put n_keys={len(keys)} fields={put_fields} "
            f"schema_fields={field_names} "
            f"nbytes={_td_nbytes(tensor_dict)} ppo_step={ppo_step} "
            f"backend={self.tq_config.backend} "
            f"partition_id={self.tq_config.partition_id}"
        )
        return meta

    async def async_get(
        self,
        key: str | Sequence[str],
        fields: Sequence[str] | None = None,
    ) -> TensorDict:
        """Restore TensorDict rows by keys, optionally restricted to ``fields``.

        Parameters
        ----------
        key : str or Sequence[str]
        fields : Sequence[str], optional
            Column subset passed to TransferQueue as ``select_fields``.
            ``None`` restores every stored field on those keys.
        """
        keys = [key] if isinstance(key, str) else list(key)
        kwargs: Dict[str, Any] = {
            "keys": keys,
            "partition_id": self.tq_config.partition_id,
        }
        if fields is not None:
            kwargs["select_fields"] = list(fields)
        tensor_dict = await tq.async_kv_batch_get(**kwargs)
        field_names = list(fields) if fields is not None else "ALL"
        log(
            f"[TQ] get n_keys={len(keys)} fields={field_names} "
            f"nbytes={_td_nbytes(tensor_dict)} "
            f"backend={self.tq_config.backend} "
            f"partition_id={self.tq_config.partition_id}"
        )
        return tensor_dict

    def get(
        self,
        key: str | Sequence[str],
        fields: Sequence[str] | None = None,
    ) -> TensorDict:
        """Synchronously restore TensorDict rows, optionally column-filtered."""
        keys = [key] if isinstance(key, str) else list(key)
        kwargs: Dict[str, Any] = {
            "keys": keys,
            "partition_id": self.tq_config.partition_id,
        }
        if fields is not None:
            kwargs["select_fields"] = list(fields)
        tensor_dict = tq.kv_batch_get(**kwargs)
        field_names = list(fields) if fields is not None else "ALL"
        log(
            f"[TQ] get n_keys={len(keys)} fields={field_names} "
            f"nbytes={_td_nbytes(tensor_dict)} "
            f"backend={self.tq_config.backend} "
            f"partition_id={self.tq_config.partition_id}"
        )
        return tensor_dict

    async def async_clear(self, key: str | Sequence[str]) -> None:
        """Clear caller-selected keys using their latest field schemas."""
        keys = [key] if isinstance(key, str) else list(key)
        partition_id = self.tq_config.partition_id
        assert partition_id is not None
        partition_info = await tq.async_kv_list(partition_id=partition_id)
        key_meta = partition_info.get(partition_id, {})
        for grouped_keys in self._group_keys_by_fields(key_meta, keys):
            await tq.async_kv_clear(
                keys=grouped_keys,
                partition_id=partition_id,
            )
        log(
            f"[TQ] clear n_keys={len(keys)} "
            f"backend={self.tq_config.backend} "
            f"partition_id={partition_id}"
        )

    @staticmethod
    def _group_keys_by_fields(
        key_meta: Dict[str, Any],
        keys: Sequence[str],
    ) -> List[List[str]]:
        """Group selected keys by the latest TransferQueue field schema.

        ``async_kv_clear`` intersects fields across each call, so heterogeneous
        schemas must be cleared separately to remove every stored field.
        """
        missing_keys = [key for key in keys if key not in key_meta]
        assert not missing_keys, f"TQ keys missing before clear: {missing_keys}"
        groups: dict[tuple[str, ...], List[str]] = defaultdict(list)
        ungrouped: List[str] = []
        for key in keys:
            meta = key_meta[key]
            fields = meta.get("fields") if isinstance(meta, dict) else None
            if fields:
                groups[tuple(sorted(str(field) for field in fields))].append(key)
            else:
                ungrouped.append(key)
        grouped_keys = list(groups.values())
        grouped_keys.extend([[key] for key in ungrouped])
        return grouped_keys

    async def async_clear_step(self, ppo_step: int) -> None:
        """Clear all keys generated for one training step."""
        partition_id = self.tq_config.partition_id
        assert partition_id is not None
        partition_info = await tq.async_kv_list(partition_id=partition_id)
        prefix = self._key_prefix(ppo_step)
        key_meta = partition_info.get(partition_id, {})
        keys = [key for key in key_meta if key.startswith(prefix)]
        for grouped_keys in self._group_keys_by_fields(
            key_meta,
            keys,
        ):
            await tq.async_kv_clear(
                keys=grouped_keys,
                partition_id=partition_id,
            )
