import json

import pytest
import torch

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.configs.debug_config import DebugConfig
from gpatch_v4.configs.tq_config import TransferQueueConfig
from gpatch_v4.configs.training_config import RLTrainingConfig
from gpatch_v4.transfer.sample_hash import (
    assert_grpo_positional_groups,
    compare_sample_hashes,
    dump_tq_sample_hashes,
    hash_field_value,
    hash_rollout_batches,
    hash_rollout_layout,
    load_stage_hash_dump,
    load_stage_hashes,
    sample_identity,
)


def test_sample_identity_uses_unique_id_and_traj():
    assert sample_identity({"unique_id": "u0", "traj_id": 1}) == "u0:1"
    assert sample_identity({"group_id": "g0", "traj_id": 2}) == "g0:2"


def test_hash_field_value_stable_for_tokens_and_logprobs():
    tokens = torch.arange(4)
    logprobs = torch.tensor([0.1, 0.2, 0.3])
    first = hash_field_value(tokens)
    second = hash_field_value(tokens.clone())
    assert first == second
    assert first["shape"] == [4]
    assert hash_field_value(logprobs) != hash_field_value(logprobs + 1e-3)


def test_unread_logprobs_mismatch_fails_even_if_tokens_match(tmp_path):
    put_rb = {
        "unique_id": ["u0", "u1"],
        "traj_id": [0, 0],
        "tokens": [torch.arange(4), torch.arange(6)],
        "rollout_log_probs": [
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([0.4, 0.5, 0.6, 0.7, 0.8]),
        ],
        "rewards": [torch.tensor(0.25), torch.tensor(-0.5)],
        "group_id": ["g0", "g0"],
        "mask": [torch.ones(3), torch.ones(5)],
    }
    restore_rb = {
        "unique_id": ["u1", "u0"],
        "traj_id": [0, 0],
        "tokens": [torch.arange(6), torch.arange(4)],
        "rollout_log_probs": [
            torch.tensor([0.4, 0.5, 0.6, 0.7, 0.8]),
            torch.tensor([0.1, 0.2, 0.9]),
        ],
        "rewards": [torch.tensor(-0.5), torch.tensor(1.25)],
        "group_id": ["g0", "g0"],
        "mask": [torch.ones(5), torch.ones(3)],
    }
    put_hashes = hash_rollout_batches([put_rb])
    restore_hashes = hash_rollout_batches([restore_rb])
    result = compare_sample_hashes(
        put_hashes,
        restore_hashes,
        allow_mismatch=["rewards"],
    )
    assert not result.ok
    assert result.missing_in_restore == []
    assert result.missing_in_put == []
    assert "u0:0" in result.unread_mismatch["rollout_log_probs"]
    assert "rewards" not in result.field_mismatch

    dump_tq_sample_hashes(
        str(tmp_path),
        stage="put",
        ppo_step=0,
        shard="w0",
        samples=put_hashes,
    )
    dump_tq_sample_hashes(
        str(tmp_path),
        stage="restore",
        ppo_step=0,
        shard="dp0",
        samples=restore_hashes,
    )
    loaded_put = load_stage_hashes(str(tmp_path), "put", 0)
    loaded_restore = load_stage_hashes(str(tmp_path), "restore", 0)
    assert loaded_put == put_hashes
    assert loaded_restore == restore_hashes
    shard = json.loads((tmp_path / "put_step0_w0.json").read_text())
    assert shard["stage"] == "put"
    assert "u0:0" in shard["samples"]


def test_compare_pass_when_restore_order_differs_but_bits_match():
    rb = {
        "unique_id": ["u0", "u1"],
        "traj_id": [0, 0],
        "tokens": [torch.arange(4), torch.arange(6)],
        "rollout_log_probs": [
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([0.4, 0.5, 0.6, 0.7, 0.8]),
        ],
        "rewards": [torch.tensor(0.25), torch.tensor(-0.5)],
        "group_id": ["g0", "g0"],
        "mask": [torch.ones(3), torch.ones(5)],
    }
    restored = {
        "unique_id": ["u1", "u0"],
        "traj_id": [0, 0],
        "tokens": [torch.arange(6), torch.arange(4)],
        "rollout_log_probs": [
            torch.tensor([0.4, 0.5, 0.6, 0.7, 0.8]),
            torch.tensor([0.1, 0.2, 0.3]),
        ],
        "rewards": [torch.tensor(-0.5), torch.tensor(0.25)],
        "group_id": ["g0", "g0"],
        "mask": [torch.ones(5), torch.ones(3)],
    }
    result = compare_sample_hashes(
        hash_rollout_batches([rb]),
        hash_rollout_batches([restored]),
    )
    assert result.ok
    assert result.matched_samples == 2


def test_dump_tq_sample_hashes_requires_mooncake():
    with pytest.raises(AssertionError, match="MooncakeStore"):
        RlConfig(
            training=RLTrainingConfig(single_controller=True),
            tq=TransferQueueConfig(enable=True),
            debug=DebugConfig(
                dump_tq_sample_hashes=True,
                tq_sample_hash_dir="/tmp/tq-hash",
            ),
        )


def test_allow_mismatch_cannot_cover_unread_logprobs():
    put_hashes = hash_rollout_batches(
        [{
            "unique_id": ["u0"],
            "traj_id": [0],
            "tokens": [torch.arange(2)],
            "rollout_log_probs": [torch.tensor([0.1])],
        }]
    )
    with pytest.raises(AssertionError, match="unread"):
        compare_sample_hashes(
            put_hashes,
            put_hashes,
            allow_mismatch=["rollout_log_probs"],
        )


def test_hash_field_value_includes_dtype():
    i64 = torch.arange(4, dtype=torch.int64)
    i32 = torch.arange(4, dtype=torch.int32)
    assert hash_field_value(i64)["dtype"] == "int64"
    assert hash_field_value(i64) != hash_field_value(i32)


def _layout_rb(group_ids, *, reverse=False):
    n = len(group_ids)
    tokens = [torch.arange(i + 2) for i in range(n)]
    prompt_lengths = [torch.tensor(1) for _ in range(n)]
    sequence_lengths = [torch.tensor(i + 2) for i in range(n)]
    rb = {
        "unique_id": [f"u{i}" for i in range(n)],
        "traj_id": [0] * n,
        "tokens": tokens,
        "rewards": [torch.tensor(float(i)) for i in range(n)],
        "group_id": list(group_ids),
        "mask": [torch.ones(i + 1) for i in range(n)],
        "prompt_lengths": prompt_lengths,
        "sequence_lengths": sequence_lengths,
    }
    if reverse:
        return {key: list(reversed(value)) for key, value in rb.items()}
    return rb


def test_assert_grpo_positional_groups_rejects_mixed_chunk():
    with pytest.raises(AssertionError, match="mixed group_id"):
        assert_grpo_positional_groups([_layout_rb(["g0", "g0", "g1", "g0"])], 2)


def test_assert_grpo_positional_groups_accepts_aligned_chunks():
    assert_grpo_positional_groups([_layout_rb(["g0", "g0", "g1", "g1"])], 2)


def test_compare_fails_when_restore_permutes_microbatch_order():
    put_rb = _layout_rb(["g0", "g0"])
    restore_rb = _layout_rb(["g0", "g0"], reverse=True)
    put_layout = hash_rollout_layout([put_rb], chunk_size=2)
    restore_layout = hash_rollout_layout([restore_rb], chunk_size=2)
    identity = compare_sample_hashes(put_layout.samples, restore_layout.samples)
    assert identity.ok
    assert identity.matched_samples == 2
    ordered = compare_sample_hashes(
        put_layout.samples,
        restore_layout.samples,
        put_microbatches=put_layout.microbatches,
        restore_microbatches=restore_layout.microbatches,
    )
    assert not ordered.ok
    assert ordered.order_mismatch


def test_dump_records_order_catalog_and_prompt_lengths(tmp_path):
    rb = _layout_rb(["g0", "g0"])
    rb["token_weights"] = [torch.ones(1), torch.ones(1)]
    layout = hash_rollout_layout([rb], chunk_size=2)
    assert "prompt_lengths" in next(iter(layout.samples.values()))
    assert layout.field_catalog["tokens"]["dtype"] == "int64"
    assert layout.field_catalog["token_weights"]["kind"] == "torch"
    dump_tq_sample_hashes(
        str(tmp_path),
        stage="put",
        ppo_step=0,
        shard="w0",
        samples=layout.samples,
        order=layout.order,
        microbatches=layout.microbatches,
        field_catalog=layout.field_catalog,
        group_chunk_errors=layout.group_chunk_errors,
    )
    dump_tq_sample_hashes(
        str(tmp_path),
        stage="restore",
        ppo_step=0,
        shard="dp0",
        samples=layout.samples,
        order=layout.order,
        microbatches=layout.microbatches,
        field_catalog=layout.field_catalog,
        group_chunk_errors=layout.group_chunk_errors,
    )
    loaded = load_stage_hash_dump(str(tmp_path), "put", 0)
    assert loaded.order == layout.order
    assert loaded.microbatches == layout.microbatches
    assert loaded.field_catalog["prompt_lengths"]["kind"] == "torch"
    result = compare_sample_hashes(
        loaded.samples,
        load_stage_hashes(str(tmp_path), "restore", 0),
        put_microbatches=loaded.microbatches,
        restore_microbatches=load_stage_hash_dump(str(tmp_path), "restore", 0).microbatches,
        put_catalog=loaded.field_catalog,
        restore_catalog=load_stage_hash_dump(str(tmp_path), "restore", 0).field_catalog,
    )
    assert result.ok


def test_catalog_reports_extra_restore_field_without_failing_identity():
    put_rb = _layout_rb(["g0"])
    restore_rb = _layout_rb(["g0"])
    restore_rb["token_weights"] = [torch.ones(1)]
    put_layout = hash_rollout_layout([put_rb])
    restore_layout = hash_rollout_layout([restore_rb])
    result = compare_sample_hashes(
        put_layout.samples,
        restore_layout.samples,
        put_catalog=put_layout.field_catalog,
        restore_catalog=restore_layout.field_catalog,
    )
    assert result.ok
    assert result.catalog_extra_in_restore == ["token_weights"]


def test_load_catalog_stable_across_put_worker_vs_dp_shards(tmp_path):
    def _rb(uids, lens):
        n = len(uids)
        return {
            "unique_id": list(uids),
            "traj_id": [0] * n,
            "tokens": [torch.arange(length) for length in lens],
            "rewards": [torch.tensor(0.0) for _ in range(n)],
            "group_id": [f"g{i}" for i in range(n)],
            "mask": [torch.ones(max(length - 1, 1)) for length in lens],
            "rollout_log_probs": [torch.zeros(max(length - 1, 1)) for length in lens],
            "prompt_lengths": [torch.tensor(1) for _ in range(n)],
            "sequence_lengths": [torch.tensor(length) for length in lens],
        }

    shard_a = hash_rollout_layout([_rb(["a0", "a1"], [4, 6])])
    shard_b = hash_rollout_layout([_rb(["b0", "b1"], [8, 10])])
    # put: worker order a then b; restore: DP order b then a (different partition).
    dump_tq_sample_hashes(
        str(tmp_path),
        stage="put",
        ppo_step=0,
        shard="w0",
        samples=shard_a.samples,
        order=shard_a.order,
        microbatches=shard_a.microbatches,
        field_catalog=shard_a.field_catalog,
    )
    dump_tq_sample_hashes(
        str(tmp_path),
        stage="put",
        ppo_step=0,
        shard="w1",
        samples=shard_b.samples,
        order=shard_b.order,
        microbatches=shard_b.microbatches,
        field_catalog=shard_b.field_catalog,
    )
    dump_tq_sample_hashes(
        str(tmp_path),
        stage="restore",
        ppo_step=0,
        shard="dp0",
        samples=shard_b.samples,
        order=shard_b.order,
        microbatches=shard_b.microbatches,
        field_catalog=shard_b.field_catalog,
    )
    dump_tq_sample_hashes(
        str(tmp_path),
        stage="restore",
        ppo_step=0,
        shard="dp1",
        samples=shard_a.samples,
        order=shard_a.order,
        microbatches=shard_a.microbatches,
        field_catalog=shard_a.field_catalog,
    )
    put_dump = load_stage_hash_dump(str(tmp_path), "put", 0)
    restore_dump = load_stage_hash_dump(str(tmp_path), "restore", 0)
    assert put_dump.field_catalog == restore_dump.field_catalog
    assert put_dump.field_catalog["tokens"]["shapes"] == [[4], [6], [8], [10]]
    result = compare_sample_hashes(
        put_dump.samples,
        restore_dump.samples,
        put_microbatches=put_dump.microbatches,
        restore_microbatches=restore_dump.microbatches,
        put_catalog=put_dump.field_catalog,
        restore_catalog=restore_dump.field_catalog,
    )
    assert result.ok
    assert result.matched_samples == 4
    assert result.catalog_dtype_mismatch == {}
