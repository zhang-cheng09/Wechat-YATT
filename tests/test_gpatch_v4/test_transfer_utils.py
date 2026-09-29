import asyncio
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import torch
from tensordict import TensorDict

from gpatch_v4.configs.tq_config import (
    TransferQueueConfig,
    TqMooncakeConfig,
    TqSimpleStorageConfig,
)
from gpatch_v4.rollout_generator.async_rollout.rollout_controller import (
    _tq_fields_for_controller_restore,
)
from gpatch_v4.transfer.tq_connector import KVBatchMeta, TqConnector
from gpatch_v4.transfer.transfer_queue_interface import (
    TQInterface,
    TqDataSlot,
    TqPayloadType,
    TqSampleOperateType,
    get_tq_data_slot,
)

async_clear_sample_refs = TQInterface.async_clear_sample_refs
async_offload_to_tq = TQInterface.async_offload_to_tq
async_put_samples = TQInterface.async_put_samples
async_restore_from_tq = TQInterface.async_restore_from_tq
async_restore_kv_batches = TQInterface.async_restore_kv_batches
capture_tq_sample_tensor_state = TQInterface.capture_tq_sample_tensor_state
restore_kv_batches = TQInterface.restore_kv_batches
restore_tq_payloads_into_samples = TQInterface.restore_tq_payloads_into_samples
subset_kv_batch_meta = TQInterface.subset_kv_batch_meta


class _FakeConnector:
    def __init__(self):
        self.tq_config = SimpleNamespace(partition_id="fake")
        self.data = {}
        self.cleared = []
        self.clear_calls = []
        self.async_get_calls = []
        self.get_calls = []
        self.put_calls = []

    @staticmethod
    def make_keys(count, *, ppo_step):
        return [f"step_{ppo_step}:sample_{index}" for index in range(count)]

    @staticmethod
    def _as_keys(key):
        if isinstance(key, str):
            return [key]
        return list(key)

    @staticmethod
    def _unbatched_row(tensor_dict, index):
        row = tensor_dict[index]
        if row.batch_dims == 0:
            return row
        assert int(row.batch_size[0]) == 1
        return row.squeeze(0)

    @staticmethod
    def _stack_unbatched_rows(rows):
        encoded = {}
        for field in rows[0].keys():
            tensors = [row[field] for row in rows]
            shapes = {tuple(tensor.shape) for tensor in tensors}
            if len(shapes) == 1:
                encoded[field] = torch.stack(tensors)
            else:
                encoded[field] = torch.nested.as_nested_tensor(
                    tensors,
                    layout=torch.jagged,
                )
        return TensorDict(encoded, batch_size=[len(rows)])

    def _gather(self, keys, fields=None):
        rows = []
        for stored_key in keys:
            row = self.data[stored_key]
            if fields is not None:
                row = row.select(*fields)
            if row.batch_dims != 0:
                row = self._unbatched_row(row, 0)
            rows.append(row)
        return self._stack_unbatched_rows(rows)

    async def async_put(self, tensor_dict, key, ppo_step, schema_fields=None):
        del ppo_step
        keys = self._as_keys(key)
        assert int(tensor_dict.batch_size[0]) == len(keys)
        self.put_calls.append({
            "keys": keys,
            "fields": sorted(tensor_dict.keys()),
            "batch_size": int(tensor_dict.batch_size[0]),
            "schema_fields": None if schema_fields is None else sorted(schema_fields),
        })
        for index, stored_key in enumerate(keys):
            update_row = self._unbatched_row(tensor_dict, index)
            if stored_key not in self.data:
                self.data[stored_key] = update_row
                continue
            current = self.data[stored_key]
            for field, value in update_row.items():
                current[field] = value
        fields = list(schema_fields) if schema_fields is not None else list(tensor_dict.keys())
        return KVBatchMeta(
            keys=keys,
            tags=[{} for _ in keys],
            partition_id="fake",
            fields=fields,
        )

    async def async_get(self, key, fields=None):
        keys = self._as_keys(key)
        self.async_get_calls.append((keys, fields))
        return self._gather(keys, fields)

    def get(self, key, fields=None):
        keys = self._as_keys(key)
        self.get_calls.append((keys, fields))
        return self._gather(keys, fields)

    async def async_clear(self, key):
        keys = self._as_keys(key)
        self.clear_calls.append(keys)
        self.cleared.extend(keys)


def test_tq_slot_batch_round_trip():
    async def run_test():
        connector = _FakeConnector()
        batched_data = {
            "tensor": [torch.arange(4).reshape(2, 2), torch.arange(3)],
            "numpy": [np.arange(2), np.arange(4).reshape(2, 2)],
            "metadata": ["first", "second"],
        }

        payload = await async_offload_to_tq(
            batched_data,
            ppo_step=3,
            payload_type=TqPayloadType.ONE_KEY_PER_DICT,
            connector=connector,
        )
        slot = get_tq_data_slot(payload)

        assert slot is not None
        assert pickle.loads(pickle.dumps(slot)) == slot
        assert slot.keys == ("step_3:sample_0",)
        assert slot.payload_type is TqPayloadType.ONE_KEY_PER_DICT
        assert payload["metadata"] is batched_data["metadata"]

        restored = await async_restore_from_tq(
            payload, connector=connector
        )
        assert restored["metadata"] == batched_data["metadata"]
        for expected, actual in zip(batched_data["tensor"], restored["tensor"]):
            torch.testing.assert_close(actual, expected)
        for expected, actual in zip(batched_data["numpy"], restored["numpy"]):
            np.testing.assert_array_equal(actual, expected)

    asyncio.run(run_test())


def test_tq_slot_respects_field_selection():
    async def run_test():
        connector = _FakeConnector()
        batched_data = {
            "offloaded": [torch.arange(2), torch.arange(2)],
            "inline": [torch.arange(3), torch.arange(3)],
        }

        payload = await async_offload_to_tq(
            batched_data,
            ppo_step=4,
            payload_type=TqPayloadType.ONE_KEY_PER_DICT,
            fields=["offloaded"],
            connector=connector,
        )

        assert isinstance(get_tq_data_slot(payload), TqDataSlot)
        assert payload["inline"] is batched_data["inline"]
        restored = await async_restore_from_tq(
            payload, connector=connector
        )
        for field in batched_data:
            for expected, actual in zip(batched_data[field], restored[field]):
                torch.testing.assert_close(actual, expected)

    asyncio.run(run_test())


def test_restore_complete_hidden_state_into_sample():
    async def build_payload():
        connector = _FakeConnector()
        hidden_state = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
        payload = await async_offload_to_tq(
            {
                "teacher_hidden_states": hidden_state,
                "metadata": ["single", "dict", "value"],
            },
            ppo_step=5,
            payload_type=TqPayloadType.ONE_KEY_PER_DICT,
            fields=["teacher_hidden_states"],
            connector=connector,
        )
        return connector, hidden_state, payload

    connector, hidden_state, payload = asyncio.run(build_payload())
    samples = [{"teacher_hidden_states": payload}]
    restore_tq_payloads_into_samples(
        samples,
        "teacher_hidden_states",
        connector=connector,
    )

    torch.testing.assert_close(samples[0]["teacher_hidden_states"], hidden_state)
    assert samples[0]["metadata"] == ["single", "dict", "value"]


def test_tq_sample_rows_update_existing_keys_and_restore_train_steps():
    async def build_train_batch():
        connector = _FakeConnector()
        samples = [
            {
                "tokens": torch.arange(4),
                "rollout_logprobs": torch.arange(3, dtype=torch.float32),
                "image": np.arange(4).reshape(2, 2),
                "group_id": "group-0",
            },
            {
                "tokens": torch.arange(6),
                "rollout_logprobs": torch.arange(5, dtype=torch.float32),
                "image": None,
                "group_id": "group-0",
            },
        ]
        sample_batch = await async_put_samples(
            samples,
            TqSampleOperateType.FIRST_PUT,
            ppo_step=7,
            connector=connector,
        )
        restored_batch = (await async_restore_kv_batches(
            [sample_batch],
            connector=connector,
        ))[0]
        restored = [
            {
                field: values[index]
                for field, values in restored_batch.items()
            }
            for index in range(2)
        ]
        original_state = capture_tq_sample_tensor_state(restored, [sample_batch])
        for index, sample in enumerate(restored):
            sample["mask"] = torch.ones(len(sample["tokens"]) - 1)
            sample["normalized_rewards"] = torch.tensor(float(index))
            sample["_train_step_id"] = index
        updated_meta = await async_put_samples(
            restored,
            TqSampleOperateType.UPDATE,
            meta=sample_batch,
            original_state=original_state,
            connector=connector,
        )
        train_metas = (
            subset_kv_batch_meta(updated_meta, [0]),
            subset_kv_batch_meta(updated_meta, [1]),
        )
        return connector, sample_batch, updated_meta, train_metas

    connector, sample_batch, updated_meta, train_metas = asyncio.run(build_train_batch())

    assert updated_meta.keys == sample_batch.keys
    restored_steps = [
        [
            {
                field: values[0]
                for field, values in batch.items()
            }
        ]
        for batch in restore_kv_batches(train_metas, connector=connector)
    ]
    assert [step[0]["_train_step_id"] for step in restored_steps] == [0, 1]
    for index, step in enumerate(restored_steps):
        sample = step[0]
        torch.testing.assert_close(sample["tokens"], torch.arange(4 + index * 2))
        torch.testing.assert_close(
            sample["normalized_rewards"],
            torch.tensor(float(index)),
        )
    np.testing.assert_array_equal(
        restored_steps[0][0]["image"],
        np.arange(4).reshape(2, 2),
    )
    assert restored_steps[1][0]["image"] is None

    asyncio.run(async_clear_sample_refs(
        train_metas,
        connector=connector,
    ))
    assert connector.cleared == sample_batch.keys


def test_tq_update_writes_one_full_column_when_one_row_is_dirty():
    async def run_test():
        connector = _FakeConnector()
        meta = await async_put_samples(
            [
                {"tokens": torch.arange(3), "rewards": torch.tensor(1.0)},
                {"tokens": torch.arange(4), "rewards": torch.tensor(2.0)},
            ],
            TqSampleOperateType.FIRST_PUT,
            ppo_step=8,
            connector=connector,
        )
        restored_batch = (await async_restore_kv_batches(
            [meta],
            connector=connector,
        ))[0]
        samples = [
            {
                field: values[index]
                for field, values in restored_batch.items()
            } for index in range(2)
        ]
        original_state = capture_tq_sample_tensor_state(samples, [meta])
        samples[0]["rewards"] += 10
        connector.put_calls.clear()

        updated_meta = await async_put_samples(
            samples,
            TqSampleOperateType.UPDATE,
            meta=meta,
            original_state=original_state,
            connector=connector,
        )

        assert connector.put_calls == [{
            "keys": meta.keys,
            "fields": ["rewards"],
            "batch_size": 2,
            "schema_fields": ["rewards", "tokens"],
        }]
        restored = (await async_restore_kv_batches(
            [updated_meta],
            connector=connector,
        ))[0]
        torch.testing.assert_close(restored["rewards"][0], torch.tensor(11.0))
        torch.testing.assert_close(restored["rewards"][1], torch.tensor(2.0))

    asyncio.run(run_test())


def test_tq_update_without_dirty_columns_still_refreshes_tags():
    async def run_test():
        connector = _FakeConnector()
        meta = await async_put_samples(
            [
                {"tokens": torch.arange(3), "sample_id": "a"},
                {"tokens": torch.arange(4), "sample_id": "b"},
            ],
            TqSampleOperateType.FIRST_PUT,
            ppo_step=9,
            connector=connector,
        )
        restored_batch = (await async_restore_kv_batches(
            [meta],
            connector=connector,
        ))[0]
        samples = [
            {
                field: values[index]
                for field, values in restored_batch.items()
            } for index in range(2)
        ]
        original_state = capture_tq_sample_tensor_state(samples, [meta])
        for index, sample in enumerate(samples):
            sample["_train_step_id"] = index
        connector.put_calls.clear()

        updated_meta = await async_put_samples(
            samples,
            TqSampleOperateType.UPDATE,
            meta=meta,
            original_state=original_state,
            connector=connector,
        )

        assert connector.put_calls == []
        restored = (await async_restore_kv_batches(
            [updated_meta],
            connector=connector,
        ))[0]
        assert restored["_train_step_id"] == [0, 1]

    asyncio.run(run_test())


def test_tq_sample_rows_select_fields_restore():
    async def run_test():
        connector = _FakeConnector()
        samples = [
            {
                "tokens": torch.arange(4),
                "image": np.arange(4).reshape(2, 2),
                "group_id": "group-0",
            },
            {
                "tokens": torch.arange(6),
                "image": None,
                "group_id": "group-1",
            },
        ]
        sample_batch = await async_put_samples(
            samples,
            TqSampleOperateType.FIRST_PUT,
            ppo_step=9,
            connector=connector,
        )
        restored_batch = (await async_restore_kv_batches(
            [sample_batch],
            fields=["tokens"],
            connector=connector,
        ))[0]
        assert set(restored_batch) == {"tokens", "group_id"}
        torch.testing.assert_close(restored_batch["tokens"][0], torch.arange(4))
        torch.testing.assert_close(restored_batch["tokens"][1], torch.arange(6))
        assert restored_batch["group_id"] == ["group-0", "group-1"]

        full_batch = (await async_restore_kv_batches(
            [sample_batch],
            connector=connector,
        ))[0]
        assert "image" in full_batch
        np.testing.assert_array_equal(
            full_batch["image"][0],
            np.arange(4).reshape(2, 2),
        )
        assert full_batch["image"][1] is None

    asyncio.run(run_test())


def test_tq_sample_batches_restore_coalesces_get_and_preserves_boundaries():
    async def run_test():
        connector = _FakeConnector()
        first = await async_put_samples(
            [
                {"tokens": torch.arange(3), "sample_id": "a"},
                {"tokens": torch.arange(4), "sample_id": "b"},
            ],
            TqSampleOperateType.FIRST_PUT,
            ppo_step=1,
            connector=connector,
        )
        second = await async_put_samples(
            [{"tokens": torch.arange(5), "sample_id": "c"}],
            TqSampleOperateType.FIRST_PUT,
            ppo_step=2,
            connector=connector,
        )

        restored = await async_restore_kv_batches(
            [first, second],
            connector=connector,
        )
        assert len(connector.async_get_calls) == 1
        assert [batch["sample_id"] for batch in restored] == [["a", "b"], ["c"]]
        assert [len(batch["tokens"]) for batch in restored] == [2, 1]

        sync_restored = restore_kv_batches(
            [first, second],
            connector=connector,
        )
        assert len(connector.get_calls) == 1
        assert [batch["sample_id"] for batch in sync_restored] == [["a", "b"], ["c"]]
        assert [len(batch["tokens"]) for batch in sync_restored] == [2, 1]

    asyncio.run(run_test())


def test_tq_metadata_only_restore_skips_storage_get():
    async def run_test():
        connector = _FakeConnector()
        sample_batch = await async_put_samples(
            [
                {"tokens": torch.arange(3), "unique_id": "u0"},
                {"tokens": torch.arange(4), "unique_id": "u1"},
            ],
            TqSampleOperateType.FIRST_PUT,
            ppo_step=3,
            connector=connector,
        )

        restored = await async_restore_kv_batches(
            [sample_batch],
            fields=[],
            connector=connector,
        )
        assert connector.async_get_calls == []
        assert set(restored[0]) == {"unique_id"}
        assert restored[0]["unique_id"] == ["u0", "u1"]

        samples = [
            {
                field: values[index]
                for field, values in restored[0].items()
            } for index in range(2)
        ]
        original_state = capture_tq_sample_tensor_state(samples, [sample_batch])
        for index, sample in enumerate(samples):
            sample["cached_feat"] = torch.tensor([float(index)])
        updated_meta = await async_put_samples(
            samples,
            TqSampleOperateType.UPDATE,
            meta=sample_batch,
            original_state=original_state,
            connector=connector,
        )
        train_batch = restore_kv_batches(
            [updated_meta],
            connector=connector,
        )[0]
        torch.testing.assert_close(train_batch["tokens"][0], torch.arange(3))
        torch.testing.assert_close(train_batch["tokens"][1], torch.arange(4))
        torch.testing.assert_close(train_batch["cached_feat"][0], torch.tensor([0.0]))
        torch.testing.assert_close(train_batch["cached_feat"][1], torch.tensor([1.0]))

    asyncio.run(run_test())


def test_tq_partial_restore_update_keeps_unloaded_fields():
    async def run_test():
        connector = _FakeConnector()
        samples = [
            {
                "tokens": torch.arange(4),
                "rollout_logprobs": torch.arange(3, dtype=torch.float32),
                "image": np.arange(4).reshape(2, 2),
                "group_id": "group-0",
            },
        ]
        sample_batch = await async_put_samples(
            samples,
            TqSampleOperateType.FIRST_PUT,
            ppo_step=4,
            connector=connector,
        )
        restored_batch = (await async_restore_kv_batches(
            [sample_batch],
            fields=["tokens"],
            connector=connector,
        ))[0]
        restored = [
            {
                field: values[0]
                for field, values in restored_batch.items()
            }
        ]
        assert "image" not in restored[0]
        assert "rollout_logprobs" not in restored[0]
        original_state = capture_tq_sample_tensor_state(restored, [sample_batch])
        restored[0]["mask"] = torch.ones(3)
        updated_meta = await async_put_samples(
            restored,
            TqSampleOperateType.UPDATE,
            meta=sample_batch,
            original_state=original_state,
            connector=connector,
        )
        full = (await async_restore_kv_batches(
            [updated_meta],
            connector=connector,
        ))[0]
        torch.testing.assert_close(full["tokens"][0], torch.arange(4))
        torch.testing.assert_close(
            full["rollout_logprobs"][0],
            torch.arange(3, dtype=torch.float32),
        )
        np.testing.assert_array_equal(full["image"][0], np.arange(4).reshape(2, 2))
        torch.testing.assert_close(full["mask"][0], torch.ones(3))

    asyncio.run(run_test())


def test_tq_subset_update_clears_full_original_key_set():
    async def run_test():
        connector = _FakeConnector()
        meta = await async_put_samples(
            [
                {"tokens": torch.arange(3), "sample_id": "a"},
                {"tokens": torch.arange(4), "sample_id": "b"},
                {"tokens": torch.arange(5), "sample_id": "c"},
            ],
            TqSampleOperateType.FIRST_PUT,
            ppo_step=6,
            connector=connector,
        )
        restored_batch = (await async_restore_kv_batches(
            [meta],
            connector=connector,
        ))[0]
        restored = [
            {
                field: values[index]
                for field, values in restored_batch.items()
            } for index in range(3)
        ]
        original_state = capture_tq_sample_tensor_state(restored, [meta])

        survivor_indices = [2, 0]
        survivors = [restored[index] for index in survivor_indices]
        survivor_meta = subset_kv_batch_meta(meta, survivor_indices)
        for index, sample in enumerate(survivors):
            sample["normalized_rewards"] = torch.tensor(float(index))
        updated_meta = await async_put_samples(
            survivors,
            TqSampleOperateType.UPDATE,
            meta=survivor_meta,
            original_state=original_state,
            connector=connector,
        )

        assert updated_meta.keys == [meta.keys[2], meta.keys[0]]
        updated = (await async_restore_kv_batches(
            [updated_meta],
            connector=connector,
        ))[0]
        torch.testing.assert_close(updated["normalized_rewards"][0], torch.tensor(0.0))
        torch.testing.assert_close(updated["normalized_rewards"][1], torch.tensor(1.0))

        await async_clear_sample_refs([meta], connector=connector)
        assert connector.clear_calls == [meta.keys]
        assert sorted(connector.cleared) == sorted(meta.keys)

    asyncio.run(run_test())


def test_native_tq_ragged_rows_and_same_key_update():
    pytest.importorskip("transfer_queue")
    owns_ray_runtime = not ray.is_initialized()
    connector = None
    if owns_ray_runtime:
        # Head from mpirun-init-ray may listen on :6379 without exporting RAY_ADDRESS.
        # Joining an existing cluster must not pass num_cpus/num_gpus.
        if os.environ.get("RAY_ADDRESS"):
            ray.init(address="auto")
        else:
            try:
                ray.init(num_cpus=4)
            except ValueError as exc:
                if "existing cluster" not in str(exc):
                    raise
                ray.init(address="auto")
    try:
        connector = TqConnector(
            TransferQueueConfig(
                enable=True,
                backend="SimpleStorage",
                simple_storage=TqSimpleStorageConfig(
                    num_storage_units=1,
                    total_storage_size=16,
                ),
            )
        )

        async def run_test():
            samples = [
                {
                    "tokens": torch.arange(3),
                    "array": np.arange(2),
                    "sample_id": 0,
                },
                {
                    "tokens": torch.arange(5),
                    "array": np.arange(3),
                    "sample_id": 1,
                },
            ]
            batch = await async_put_samples(
                samples,
                TqSampleOperateType.FIRST_PUT,
                ppo_step=9,
                connector=connector,
            )
            restored_batch = (await async_restore_kv_batches(
                [batch],
                connector=connector,
            ))[0]
            restored = [
                {
                    field: values[index]
                    for field, values in restored_batch.items()
                } for index in range(2)
            ]
            original_state = capture_tq_sample_tensor_state(restored, [batch])
            restored[0]["array"] += 10
            for index, sample in enumerate(restored):
                sample["normalized_rewards"] = torch.tensor(float(index))
            updated_meta = await async_put_samples(
                restored,
                TqSampleOperateType.UPDATE,
                meta=batch,
                original_state=original_state,
                connector=connector,
            )
            return restore_kv_batches(
                [updated_meta],
                connector=connector,
            )

        restored_steps = asyncio.run(run_test())
        restored_samples = [
            {
                field: values[index]
                for field, values in restored_steps[0].items()
            }
            for index in range(2)
        ]
        assert [len(sample["tokens"]) for sample in restored_samples] == [3, 5]
        assert [
            sample["normalized_rewards"].item()
            for sample in restored_samples
        ] == [0.0, 1.0]
        np.testing.assert_array_equal(
            restored_samples[0]["array"],
            np.arange(2) + 10,
        )
    finally:
        if connector is not None:
            connector.close()
        if owns_ray_runtime:
            ray.shutdown()


@pytest.mark.skipif(
    os.environ.get("GCORE_RUN_MOONCAKE_TQ_TEST") != "1",
    reason="set GCORE_RUN_MOONCAKE_TQ_TEST=1 to run MooncakeStore integration",
)
def test_mooncake_tq_variable_length_audio_and_none_round_trip():
    pytest.importorskip("transfer_queue")
    connector = None
    try:
        connector = TqConnector(
            TransferQueueConfig(
                enable=True,
                backend="MooncakeStore",
                mooncake=TqMooncakeConfig(
                    protocol="tcp",
                    global_segment_size_gb=1,
                    local_buffer_size_gb=1,
                ),
            )
        )
        expected_audio = [
            torch.arange(5, dtype=torch.float32),
            torch.arange(8, dtype=torch.float32) + 10,
            None,
        ]
        samples = [
            {
                "tokens": torch.arange(3 + index),
                "audio": audio,
                "sample_id": index,
            }
            for index, audio in enumerate(expected_audio)
        ]

        async def run_test():
            sample_batch = await async_put_samples(
                samples,
                TqSampleOperateType.FIRST_PUT,
                ppo_step=11,
                connector=connector,
            )
            async_restored = (await async_restore_kv_batches(
                [sample_batch],
                connector=connector,
            ))[0]
            sync_restored = restore_kv_batches(
                [sample_batch],
                connector=connector,
            )[0]
            await async_clear_sample_refs(
                [sample_batch],
                connector=connector,
            )
            return async_restored, sync_restored

        async_restored, sync_restored = asyncio.run(run_test())
        for restored in (async_restored, sync_restored):
            for expected, actual in zip(
                expected_audio[:2],
                restored["audio"][:2],
                strict=True,
            ):
                torch.testing.assert_close(actual, expected)
            assert restored["audio"][2] is None
    finally:
        if connector is not None:
            connector.close()


def test_restore_kv_batches_returns_clean_batches():
    async def run_test():
        connector = _FakeConnector()
        samples = [
            {
                "tokens": torch.arange(4),
                "group_id": "g0",
            },
            {
                "tokens": torch.arange(6),
                "group_id": "g0",
            },
        ]
        sample_batch = await async_put_samples(
            samples,
            TqSampleOperateType.FIRST_PUT,
            ppo_step=3,
            connector=connector,
        )
        async_batch = (await async_restore_kv_batches(
            [sample_batch],
            connector=connector,
        ))[0]
        sync_batch = restore_kv_batches(
            [sample_batch],
            connector=connector,
        )[0]
        assert set(async_batch) == {"tokens", "group_id"}
        assert set(sync_batch) == {"tokens", "group_id"}
        torch.testing.assert_close(sync_batch["tokens"][0], samples[0]["tokens"])
        torch.testing.assert_close(sync_batch["tokens"][1], samples[1]["tokens"])
        assert sync_batch["group_id"] == ["g0", "g0"]

    asyncio.run(run_test())


def test_controller_whitelist_then_sync_train_restore_keeps_unread_logprobs():
    async def run_test():
        connector = _FakeConnector()
        samples = [
            {
                "tokens": torch.arange(4),
                "mask": torch.ones(3),
                "prompt_lengths": torch.tensor(2),
                "sequence_lengths": torch.tensor(4),
                "rewards": torch.tensor(0.25),
                "rollout_log_probs": torch.tensor([0.1, 0.2, 0.3]),
                "group_id": "g0",
                "unique_id": "u0",
            },
            {
                "tokens": torch.arange(6),
                "mask": torch.ones(5),
                "prompt_lengths": torch.tensor(2),
                "sequence_lengths": torch.tensor(6),
                "rewards": torch.tensor(-0.5),
                "rollout_log_probs": torch.tensor([0.4, 0.5, 0.6, 0.7, 0.8]),
                "group_id": "g0",
                "unique_id": "u1",
            },
        ]
        sample_batch = await async_put_samples(
            samples,
            TqSampleOperateType.FIRST_PUT,
            ppo_step=5,
            connector=connector,
        )
        controller_fields = _tq_fields_for_controller_restore(
            SimpleNamespace(
                training=SimpleNamespace(
                    use_bt_rm_reward=False,
                    dynamic_batch_train=True,
                    metrics_report=None,
                ),
                tq=SimpleNamespace(controller_restore_fields=None),
            )
        )
        assert "tokens" in controller_fields
        assert "rewards" in controller_fields
        assert "rollout_log_probs" not in controller_fields

        controller_batch = (await async_restore_kv_batches(
            [sample_batch],
            fields=controller_fields,
            connector=connector,
        ))[0]
        restored = [
            {
                field: values[index]
                for field, values in controller_batch.items()
            } for index in range(len(samples))
        ]
        for sample in restored:
            assert "rollout_log_probs" not in sample
            sample.pop("unique_id", None)
        original_state = capture_tq_sample_tensor_state(restored, [sample_batch])
        for index, sample in enumerate(restored):
            sample["rewards"] = sample["rewards"] + 1.0
            sample["cached_feat"] = torch.tensor([float(index)])
        updated_meta = await async_put_samples(
            restored,
            TqSampleOperateType.UPDATE,
            meta=sample_batch,
            original_state=original_state,
            connector=connector,
        )
        train_batch = restore_kv_batches(
            [updated_meta],
            connector=connector,
        )[0]
        assert train_batch["group_id"] == ["g0", "g0"]
        torch.testing.assert_close(train_batch["tokens"][0], samples[0]["tokens"])
        torch.testing.assert_close(train_batch["tokens"][1], samples[1]["tokens"])
        torch.testing.assert_close(train_batch["mask"][0], samples[0]["mask"])
        torch.testing.assert_close(
            train_batch["rollout_log_probs"][0],
            samples[0]["rollout_log_probs"],
        )
        torch.testing.assert_close(
            train_batch["rollout_log_probs"][1],
            samples[1]["rollout_log_probs"],
        )
        torch.testing.assert_close(train_batch["rewards"][0], torch.tensor(1.25))
        torch.testing.assert_close(train_batch["rewards"][1], torch.tensor(0.5))
        torch.testing.assert_close(train_batch["cached_feat"][0], torch.tensor([0.0]))
        torch.testing.assert_close(train_batch["cached_feat"][1], torch.tensor([1.0]))

    asyncio.run(run_test())


def test_tq_enable_allows_single_controller_without_dynamic_batch():
    from gpatch_v4.configs.config import RlConfig
    from gpatch_v4.configs.training_config import RLTrainingConfig

    cfg = RlConfig(
        training=RLTrainingConfig(single_controller=True),
        tq=TransferQueueConfig(enable=True),
    )
    assert cfg.tq.enable
    assert cfg.training.single_controller
    assert not cfg.training.dynamic_batch_train


def test_tq_enable_still_requires_single_controller():
    from gpatch_v4.configs.config import RlConfig

    with pytest.raises(AssertionError, match="single_controller"):
        RlConfig(tq=TransferQueueConfig(enable=True))
