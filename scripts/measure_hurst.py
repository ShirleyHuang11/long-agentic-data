"""Hurst exponent of the surprisal increment process (Alabdulmohsin et al.,
arXiv:2402.01825) for registry datasets.

Protocol mirror: the paper computes z_t = -log p(w_t | prefix) with an LLM,
normalizes to zero-mean/unit-variance increments, and estimates H via
rescaled-range analysis R(n)/S(n) ~ n^H. We replace the LLM with an in-sample
order-3 byte n-gram model (add-0.5 smoothing) over the same 8 MB
doc-per-episode corpus the (alpha, H_inf) oracle uses — a compressor-class
proxy, so absolute levels are not comparable to the paper's PaLM2 numbers
(language H=0.70+-0.09, GitHub 0.79, DM-Math 0.50); contrasts between
datasets are the meaningful output.

Usage: python scripts/measure_hurst.py --slug <slug> [...]
Appends to data/hurst.csv.
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

WINDOWS = [2 ** k for k in range(6, 17)]  # 64 .. 65536


def surprisal_series(arr):
    a = arr.astype(np.int64)
    idx3 = a[:-2] * 65536 + a[1:-1] * 256 + a[2:]
    idx2 = a[:-2] * 256 + a[1:-1]
    c3 = np.bincount(idx3, minlength=256 ** 3)
    c2 = np.bincount(idx2, minlength=256 ** 2)
    p = (c3[idx3] + 0.5) / (c2[idx2] + 128.0)
    z = -np.log2(p)
    z = (z - z.mean()) / z.std()
    return z


def rs_hurst(z):
    pts = []
    for n in WINDOWS:
        m = len(z) // n
        if m < 8:
            break
        w = z[: m * n].reshape(m, n)
        w = w - w.mean(axis=1, keepdims=True)
        Y = np.cumsum(w, axis=1)
        R = Y.max(axis=1) - Y.min(axis=1)
        S = w.std(axis=1)
        ok = S > 0
        pts.append((n, float(np.mean(R[ok] / S[ok]))))
    xs = [math.log(n) for n, _ in pts]
    ys = [math.log(rs) for _, rs in pts]
    h = float(np.polyfit(xs, ys, 1)[0])
    return h, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", action="append", required=True)
    args = ap.parse_args()

    out = "data/hurst.csv"
    exists = os.path.exists(out)
    fields = ["slug", "hurst", "n_bytes"] + [f"rs_n{n}" for n in WINDOWS]
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
            z = surprisal_series(arr)
            h, pts = rs_hurst(z)
            row = {"slug": slug, "hurst": round(h, 4), "n_bytes": len(arr),
                   **{f"rs_n{n}": f"{rs:.4g}" for n, rs in pts}}
            w.writerow(row)
            f.flush()
            print(f"{slug}: H={h:.3f}  ({len(pts)} window sizes)", flush=True)


if __name__ == "__main__":
    main()
