"""Copy DAPO-17K and add a synthetic integer ``level`` column.

Each record is assigned ``0`` / ``1`` / ``2`` with probabilities ``5:3:2``.
Source files are not modified.

Usage
-----
python3 tasks/math_rl_v4/add_dapo_e2h_levels.py
python3 tasks/math_rl_v4/add_dapo_e2h_levels.py \\
    --src hf-hub/zhuzilin/dapo-math-17k \\
    --dst hf-hub/zhuzilin/dapo-math-17k-e2h
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_SRC = os.path.join(_REPO_ROOT, "hf-hub", "zhuzilin", "dapo-math-17k")
_DEFAULT_DST = os.path.join(_REPO_ROOT, "hf-hub", "zhuzilin", "dapo-math-17k-e2h")

LEVEL_VALUES = np.array([0, 1, 2], dtype=np.int64)
LEVEL_PROBS = np.array([0.5, 0.3, 0.2], dtype=np.float64)


def sample_levels(n: int, seed: int) -> np.ndarray:
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    rng = np.random.default_rng(seed)
    return rng.choice(LEVEL_VALUES, size=n, p=LEVEL_PROBS)


def attach_levels(records: list[dict[str, Any]], levels: np.ndarray) -> list[dict[str, Any]]:
    if len(records) != int(levels.shape[0]):
        raise ValueError(
            f"records {len(records)} vs levels {int(levels.shape[0])}"
        )
    out = []
    for rec, lv in zip(records, levels):
        row = dict(rec)
        row["level"] = int(lv)
        out.append(row)
    return out


def _list_data_files(src: str) -> tuple[str, list[str]]:
    if os.path.isfile(src):
        if src.endswith(".jsonl"):
            return "jsonl", [src]
        if src.endswith(".parquet"):
            return "parquet", [src]
        raise ValueError(f"unsupported file: {src}")
    jsonl_files = sorted(glob.glob(os.path.join(src, "*.jsonl")))
    parquet_files = sorted(glob.glob(os.path.join(src, "*.parquet")))
    if jsonl_files:
        return "jsonl", jsonl_files
    if parquet_files:
        return "parquet", parquet_files
    raise FileNotFoundError(f"no *.jsonl or *.parquet under {src}")


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fin:
        for line_i, line in enumerate(fin):
            if not line.strip():
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError(f"{path}:{line_i} is not a JSON object")
            rows.append(rec)
    return rows


def _write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_parquet(path: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=path, split="train")
    return [dict(row) for row in ds]


def _write_parquet(path: str, records: list[dict[str, Any]]) -> None:
    from datasets import Dataset

    Dataset.from_list(records).to_parquet(path)


def add_levels_under_dir(src: str, dst: str, seed: int) -> dict[str, int]:
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if src == dst:
        raise ValueError("src and dst must be different paths")
    fmt, files = _list_data_files(src)
    chunks: list[tuple[str, list[dict[str, Any]]]] = []
    for path in files:
        if fmt == "jsonl":
            chunks.append((os.path.basename(path), _load_jsonl(path)))
        else:
            chunks.append((os.path.basename(path), _load_parquet(path)))
    n = sum(len(recs) for _, recs in chunks)
    if n == 0:
        raise ValueError(f"no records in {src}")
    levels = sample_levels(n, seed)
    os.makedirs(dst, exist_ok=True)
    offset = 0
    for name, recs in chunks:
        k = len(recs)
        labeled = attach_levels(recs, levels[offset:offset + k])
        out_path = os.path.join(dst, name)
        if fmt == "jsonl":
            _write_jsonl(out_path, labeled)
        else:
            _write_parquet(out_path, labeled)
        offset += k
    hist = {int(v): int((levels == v).sum()) for v in LEVEL_VALUES}
    return hist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=_DEFAULT_SRC)
    parser.add_argument("--dst", default=_DEFAULT_DST)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    hist = add_levels_under_dir(args.src, args.dst, args.seed)
    total = sum(hist.values())
    print(f"wrote {total} records -> {os.path.abspath(args.dst)}")
    for k in sorted(hist):
        print(f"  level {k}: {hist[k]} ({hist[k] / total:.3f})")


if __name__ == "__main__":
    main()
