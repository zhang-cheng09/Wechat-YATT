"""Compare AgentLoop put hashes against train restore hashes."""
from __future__ import annotations

import argparse
import sys

from gpatch_v4.transfer.sample_hash import (
    SampleHashCompareResult,
    compare_sample_hashes,
    load_stage_hash_dump,
)


def _fail_reasons(result: SampleHashCompareResult) -> list[str]:
    reasons: list[str] = []
    if result.missing_in_restore:
        reasons.append("missing_in_restore")
    if result.missing_in_put:
        reasons.append("missing_in_put")
    if result.unread_mismatch:
        reasons.append("unread_mismatch")
    if result.field_mismatch:
        reasons.append("field_mismatch")
    if result.order_mismatch:
        reasons.append("order_mismatch")
    if result.microbatch_membership_mismatch:
        reasons.append("microbatch_membership_mismatch")
    if result.group_chunk_errors:
        reasons.append("group_chunk_errors")
    if result.catalog_dtype_mismatch:
        reasons.append("catalog_dtype_mismatch")
    return reasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Join TQ sample hashes dumped at AgentLoop put and train restore. "
            "Unread logprob fields always fail on mismatch. Microbatch identity "
            "order and group_id chunks fail separately from the identity join."
        )
    )
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--ppo-step", type=int, default=0)
    parser.add_argument(
        "--allow-mismatch",
        nargs="*",
        default=[],
        help="Whitelist fields allowed to differ (e.g. rewards). Unread logprobs cannot be listed.",
    )
    args = parser.parse_args(argv)
    put_dump = load_stage_hash_dump(args.dump_dir, "put", args.ppo_step)
    restore_dump = load_stage_hash_dump(args.dump_dir, "restore", args.ppo_step)
    result = compare_sample_hashes(
        put_dump.samples,
        restore_dump.samples,
        allow_mismatch=args.allow_mismatch,
        put_microbatches=put_dump.microbatches,
        restore_microbatches=restore_dump.microbatches,
        put_catalog=put_dump.field_catalog,
        restore_catalog=restore_dump.field_catalog,
        put_group_chunk_errors=put_dump.group_chunk_errors,
        restore_group_chunk_errors=restore_dump.group_chunk_errors,
    )
    print(f"put_samples={len(put_dump.samples)} restore_samples={len(restore_dump.samples)}")
    print(
        f"put_microbatches={len(put_dump.microbatches)} "
        f"restore_microbatches={len(restore_dump.microbatches)}"
    )
    print(f"matched_samples={result.matched_samples}")
    print(f"missing_in_restore={result.missing_in_restore}")
    print(f"missing_in_put={result.missing_in_put}")
    print(f"unread_mismatch={result.unread_mismatch}")
    print(f"field_mismatch={result.field_mismatch}")
    print(f"order_mismatch={result.order_mismatch}")
    print(f"microbatch_membership_mismatch={result.microbatch_membership_mismatch}")
    print(f"group_chunk_errors={result.group_chunk_errors}")
    print(f"catalog_extra_in_restore={result.catalog_extra_in_restore}")
    print(f"catalog_missing_in_restore={result.catalog_missing_in_restore}")
    print(f"catalog_dtype_mismatch={result.catalog_dtype_mismatch}")
    if result.ok:
        print("PASS: put vs restore hashes match (unread fields required)")
        return 0
    print(f"FAIL: TQ put/restore hashes differ; reasons={_fail_reasons(result)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
