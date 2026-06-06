"""Measure the token-token correlation decay exponent beta (Cagnetta et al.,
arXiv:2602.07488 Eq. 7) for registry datasets, at byte level.

C_{mu,nu}(n) = P(X_i=mu, X_{i+n}=nu) - P(mu)P(nu) over the same 8 MB
doc-per-episode corpus the (alpha, H_inf) oracle uses; we record the operator
norm ||C(n)||_op at geometric lags and fit ||C(n)||_op ~ n^-beta over the
initial decay regime (lags 1..64), mirroring the paper's WikiText procedure.

Usage: python scripts/measure_beta.py --slug toucan-15m-kimi-k2 [...]
Appends to data/gamma_beta.csv.
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle
import score_agentic_datasets as sad

LAGS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
FIT_MAX = 64  # initial-decay regime, as in the paper's WikiText fit


def corr_opnorm(arr, lag):
    x = arr[:-lag].astype(np.int64)
    y = arr[lag:].astype(np.int64)
    joint = np.bincount(x * 256 + y, minlength=65536).reshape(256, 256)
    joint = joint / len(x)
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    C = joint - np.outer(px, py)
    return float(np.linalg.svd(C, compute_uv=False)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", action="append", required=True)
    args = ap.parse_args()

    out = "data/gamma_beta.csv"
    exists = os.path.exists(out)
    fields = ["slug", "beta", "n_bytes"] + [f"opnorm_lag{n}" for n in LAGS]
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for slug in args.slug:
            entry = next(e for e in sad.REGISTRY if e[4] == slug)
            path, cfg, splits, ser = entry[:4]
            group_key = entry[5] if len(entry) > 5 else None
            docs, size = [], 0
            for doc, _ in sad.iter_docs(path, cfg, splits, ser, group_key=group_key):
                docs.append(doc)
                size += len(doc)
                if len(docs) >= lz_oracle.MAX_DOCS or size >= lz_oracle.MAX_BYTES:
                    break
            corpus = lz_oracle.build_corpus(docs)
            arr = np.frombuffer(corpus, dtype=np.uint8)
            ops = {n: corr_opnorm(arr, n) for n in LAGS}
            xs = [math.log(n) for n in LAGS if n <= FIT_MAX]
            ys = [math.log(ops[n]) for n in LAGS if n <= FIT_MAX]
            beta = -np.polyfit(xs, ys, 1)[0]
            row = {"slug": slug, "beta": round(float(beta), 4),
                   "n_bytes": len(arr),
                   **{f"opnorm_lag{n}": f"{ops[n]:.6g}" for n in LAGS}}
            w.writerow(row)
            f.flush()
            print(f"{slug}: beta={beta:.3f}  opnorms="
                  f"{[f'{ops[n]:.4f}' for n in LAGS[:6]]}", flush=True)


if __name__ == "__main__":
    main()
