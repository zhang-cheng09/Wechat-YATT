import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


def compare_mtp_checkpoint(source: Path, saved: Path) -> dict:
    source_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    saved_map = json.loads((saved / "model.safetensors.index.json").read_text())["weight_map"]
    source_keys = {key for key in source_map if key.startswith("mtp.")}
    saved_keys = {key for key in saved_map if key.startswith("mtp.")}
    assert source_keys, "source checkpoint has no MTP tensors"
    assert source_keys == saved_keys, {
        "missing": sorted(source_keys - saved_keys),
        "unexpected": sorted(saved_keys - source_keys),
    }
    rows = []
    for key in sorted(source_keys | {"lm_head.weight"}):
        with (
            safe_open(source / source_map[key], framework="pt") as original,
            safe_open(saved / saved_map[key], framework="pt") as checkpoint,
        ):
            expected, actual = original.get_slice(key), checkpoint.get_slice(key)
            shape = expected.get_shape()
            assert shape == actual.get_shape(), key
            assert expected.get_dtype() == actual.get_dtype(), key
            chunk_rows = max(1, 4 * 1024 * 1024 // max(1, math.prod(shape[1:])))
            unequal = 0
            total = 0
            for start in range(0, shape[0], chunk_rows):
                a = expected[start:start + chunk_rows]
                b = actual[start:start + chunk_rows]
                unequal += int(torch.count_nonzero(a != b))
                total += a.numel()
            row = {
                "key": key,
                "shape": shape,
                "dtype": expected.get_dtype(),
                "numel": total,
                "unequal_elements": unequal
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    mtp_unchanged = all(
        row["unequal_elements"] == 0 for row in rows if row["key"].startswith("mtp.")
    )
    backbone_changed = next(
        row["unequal_elements"] for row in rows if row["key"] == "lm_head.weight"
    ) > 0
    return {
        "source": str(source),
        "saved": str(saved),
        "mtp_tensor_count": len(source_keys),
        "mtp_unchanged": mtp_unchanged,
        "backbone_changed": backbone_changed,
        "tensors": rows
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--saved", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = compare_mtp_checkpoint(args.source, args.saved)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    assert report["mtp_unchanged"], "MTP weights changed"
    assert report["backbone_changed"], "lm_head did not change at checkpoint precision"
    print("PASS: all MTP tensors unchanged; backbone updated", flush=True)
