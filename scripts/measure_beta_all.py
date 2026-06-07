"""Measure beta for the diverse corpora in reference/_lz_Hinf_all.csv
(code / math / web / wiki / stories / Paloma domains), same byte-level
protocol as measure_beta.py. Appends to data/gamma_beta_all.csv.

Usage: python scripts/measure_beta_all.py --start 0 --end 12
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle
from measure_beta import LAGS, FIT_MAX, corr_opnorm

REF = "reference/_lz_Hinf_all.csv"
OUT = "data/gamma_beta_all.csv"
TEXT_KEYS = ("text", "content", "raw_content", "code")


def get_text(row):
    for k in TEXT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    for v in row.values():
        if isinstance(v, str) and len(v) > 50:
            return v
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    args = ap.parse_args()

    refs = list(csv.DictReader(open(REF)))[args.start:args.end]
    exists = os.path.exists(OUT)
    fields = ["dataset", "subset_or_config", "beta", "n_bytes"]
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for r in refs:
            path, cfg = r["path"], r["subset_or_config"] or None
            name = r["dataset"]
            ds = None
            for split in ("train", "validation", "val", "test"):
                try:
                    ds = load_dataset(path, cfg, split=split, streaming=True)
                    break
                except Exception as e:
                    err = e
            if ds is None:
                print(f"{name}/{cfg}: LOAD FAIL {type(err).__name__}: {str(err)[:100]}",
                      flush=True)
                continue
            try:
                docs, size = [], 0
                for row in ds:
                    t = get_text(row)
                    if not t:
                        continue
                    docs.append(t)
                    size += len(t)
                    if size >= lz_oracle.MAX_BYTES or len(docs) >= lz_oracle.MAX_DOCS:
                        break
                corpus = lz_oracle.build_corpus(docs)
                arr = np.frombuffer(corpus, dtype=np.uint8)
                if len(arr) < 200_000:
                    print(f"{name}/{cfg}: corpus too small ({len(arr)}B) — skip",
                          flush=True)
                    continue
                ops = {n: corr_opnorm(arr, n) for n in LAGS}
                xs = [math.log(n) for n in LAGS if n <= FIT_MAX]
                ys = [math.log(ops[n]) for n in LAGS if n <= FIT_MAX]
                beta = float(-np.polyfit(xs, ys, 1)[0])
            except Exception as e:
                print(f"{name}/{cfg}: FAIL {type(e).__name__}: {str(e)[:100]}",
                      flush=True)
                continue
            w.writerow({"dataset": name, "subset_or_config": cfg or "",
                        "beta": round(beta, 4), "n_bytes": len(arr)})
            f.flush()
            print(f"{name}/{cfg}: beta={beta:.3f}", flush=True)


if __name__ == "__main__":
    main()
