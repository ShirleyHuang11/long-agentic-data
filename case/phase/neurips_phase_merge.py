#!/usr/bin/env python3
"""Merge sharded sweep outputs into one directory.

Useful when running many seed shards in parallel.
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge multiple sweep shard directories")
    ap.add_argument("--inputs", type=str, required=True, help="Comma-separated sweep directories")
    ap.add_argument("--out-dir", type=str, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    in_dirs = [Path(x.strip()) for x in args.inputs.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_frames = []
    sum_frames = []

    for d in in_dirs:
        raw_path = d / "raw_metrics.csv"
        sum_path = d / "run_summary.csv"
        if raw_path.exists():
            raw_frames.append(pd.read_csv(raw_path))
        if sum_path.exists():
            sum_frames.append(pd.read_csv(sum_path))

    if not raw_frames or not sum_frames:
        raise ValueError("No raw_metrics.csv / run_summary.csv found in inputs")

    raw = pd.concat(raw_frames, ignore_index=True)
    summary = pd.concat(sum_frames, ignore_index=True)

    raw = raw.drop_duplicates(subset=["beta", "gamma", "seed", "eval_len"], keep="last")
    summary = summary.drop_duplicates(subset=["beta", "gamma", "seed"], keep="last")

    raw = raw.sort_values(["beta", "gamma", "seed", "eval_len"]).reset_index(drop=True)
    summary = summary.sort_values(["beta", "gamma", "seed"]).reset_index(drop=True)

    raw_out = out_dir / "raw_metrics.csv"
    sum_out = out_dir / "run_summary.csv"
    raw.to_csv(raw_out, index=False)
    summary.to_csv(sum_out, index=False)

    print("[Done] Merged sweep shards:")
    print(f"  - {raw_out} ({len(raw)} rows)")
    print(f"  - {sum_out} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
