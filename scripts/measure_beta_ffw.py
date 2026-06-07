"""Measure beta (byte-level two-point correlation decay) for the FineFineWeb
domain subsets listed in reference/all-lz_Hinf_ffw.csv, matching the agentic
beta protocol (measure_beta.py): 8 MB corpus, opnorm of byte co-occurrence
covariance at geometric lags, power-law fit over lags 1..64.

gamma for these domains = the `alpha` column already in the reference CSV.

Usage: python scripts/measure_beta_ffw.py --start 0 --end 17
Appends to data/gamma_beta_ffw.csv.
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

REF = "reference/all-lz_Hinf_ffw.csv"
OUT = "data/gamma_beta_ffw.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    args = ap.parse_args()

    subsets = [r["subset_or_config"] for r in csv.DictReader(open(REF))]
    subsets = subsets[args.start:args.end]

    exists = os.path.exists(OUT)
    fields = ["subset", "beta", "n_bytes"] + [f"opnorm_lag{n}" for n in LAGS]
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for sub in subsets:
            try:
                ds = load_dataset(
                    "json",
                    data_files=f"hf://datasets/m-a-p/FineFineWeb/{sub}/{sub}_000000.jsonl",
                    split="train", streaming=True)
                docs, size = [], 0
                for row in ds:
                    t = row.get("text") or ""
                    docs.append(t)
                    size += len(t)
                    if size >= lz_oracle.MAX_BYTES or len(docs) >= lz_oracle.MAX_DOCS:
                        break
                corpus = lz_oracle.build_corpus(docs)
                arr = np.frombuffer(corpus, dtype=np.uint8)
                ops = {n: corr_opnorm(arr, n) for n in LAGS}
                xs = [math.log(n) for n in LAGS if n <= FIT_MAX]
                ys = [math.log(ops[n]) for n in LAGS if n <= FIT_MAX]
                beta = float(-np.polyfit(xs, ys, 1)[0])
            except Exception as e:
                print(f"{sub}: FAIL {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            w.writerow({"subset": sub, "beta": round(beta, 4), "n_bytes": len(arr),
                        **{f"opnorm_lag{n}": f"{ops[n]:.6g}" for n in LAGS}})
            f.flush()
            print(f"{sub}: beta={beta:.3f}", flush=True)


if __name__ == "__main__":
    main()
