"""5-seed sigma runner: re-score a registry entry on disjoint episode samples.

"Seed" = sampling offset: sample k uses episodes [k*N, k*N + cap) in streaming
order, where N is the canonical run's episode count, so the 5 corpora are
disjoint. Offset 0 reproduces the canonical registry number exactly.

Appends rows to data/seed_sigma.csv (slug, seed_offset, alpha, h_inf, ...).
Does NOT touch the canonical provenance sidecars or samples_cache.

Usage:
    python scripts/seed_sigma.py --slug toucan-15m-kimi-k2 --step 553 --seeds 5
"""

import argparse
import csv
import itertools
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle
import score_agentic_datasets as sad

OUT = "data/seed_sigma.csv"
FIELDS = ["slug", "seed_offset", "alpha", "h_inf",
          "bpc_128", "bpc_2048", "bpc_32768",
          "n_episodes", "mean_turns", "n_bytes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--step", type=int, required=True,
                    help="episodes per sample (canonical n_episodes)")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    entry = next(e for e in sad.REGISTRY if e[4] == args.slug)
    path, cfg, splits, ser = entry[:4]
    group_key = entry[5] if len(entry) > 5 else None

    exists = os.path.exists(OUT)
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for k in range(args.seeds):
            off = k * args.step
            docs, turns, size = [], [], 0
            it = itertools.islice(
                sad.iter_docs(path, cfg, splits, ser, group_key=group_key),
                off, None)
            for doc, n in it:
                docs.append(doc)
                turns.append(n)
                size += len(doc)
                if len(docs) >= lz_oracle.MAX_DOCS or size >= lz_oracle.MAX_BYTES:
                    break
            if len(docs) < 50:  # ran off the end of the dataset
                print(f"{args.slug} off={off}: only {len(docs)} eps — skip",
                      flush=True)
                continue
            res = lz_oracle.score(docs)
            w.writerow({
                "slug": args.slug, "seed_offset": off,
                "alpha": res["alpha"], "h_inf": res["h_inf"],
                "bpc_128": res["bpc_128"], "bpc_2048": res["bpc_2048"],
                "bpc_32768": res["bpc_32768"], "n_episodes": len(docs),
                "mean_turns": round(statistics.mean(turns), 1) if turns else 0,
                "n_bytes": res["n_bytes"],
            })
            f.flush()
            print(f"{args.slug} off={off}: alpha={res['alpha']:.3f} "
                  f"H_inf={res['h_inf']:.3f} n={len(docs)}", flush=True)


if __name__ == "__main__":
    main()
