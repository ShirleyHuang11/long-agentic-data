"""Diversity-dose curve on DTap (finding 17 quantified).

Fix the corpus budget (lz_oracle caps) and vary the number of distinct
domain/{benign,malicious} groups g: sample round-robin across the first g
groups (sorted; balanced quota emerges naturally), score (alpha, H_inf) per g.
Prediction from finding 17: H_inf rises with g and collapses at small g.

Usage: python scripts/dtap_diversity_dose.py [--config claudesdk/claude-opus-4-6]
Appends to data/dtap_diversity_dose.csv.
"""

import argparse
import csv
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle
from huggingface_hub import HfApi, hf_hub_download

REPO = "AI-Secure/DTap-Bench-Agent-Trajectories"
GS = [1, 2, 4, 8, 16, 24]


def ser_dtap(traj):
    parts = []
    for s in traj:
        text = s.get("state") or s.get("action") or ""
        parts.append(f"[{s.get('role', '?')}]\n{text}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="claudesdk/claude-opus-4-6")
    args = ap.parse_args()

    api = HfApi()
    files = [s.rfilename for s in api.dataset_info(REPO).siblings
             if s.rfilename.startswith(args.config + "/")
             and s.rfilename.endswith(".json")
             and "judge_result" not in s.rfilename]
    groups = {}
    for f in sorted(files):
        groups.setdefault("/".join(f.split("/")[2:4]), []).append(f)
    names = sorted(groups)

    out = "data/dtap_diversity_dose.csv"
    exists = os.path.exists(out)
    fields = ["config", "n_groups", "alpha", "h_inf",
              "bpc_128", "bpc_2048", "bpc_32768", "n_episodes", "n_bytes"]
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            w.writeheader()
        for g in GS:
            sel = names[:g]
            order = [f for batch in itertools.zip_longest(
                         *[groups[n] for n in sel])
                     for f in batch if f is not None]
            docs, size = [], 0
            for f in order:
                try:
                    p = hf_hub_download(REPO, f, repo_type="dataset")
                    doc = ser_dtap(json.load(open(p, encoding="utf-8"))
                                   .get("trajectory") or [])
                except Exception:
                    continue
                if not doc:
                    continue
                docs.append(doc)
                size += len(doc)
                if len(docs) >= lz_oracle.MAX_DOCS or size >= lz_oracle.MAX_BYTES:
                    break
            res = lz_oracle.score(docs)
            w.writerow({"config": args.config, "n_groups": g,
                        "alpha": res["alpha"], "h_inf": res["h_inf"],
                        "bpc_128": res["bpc_128"], "bpc_2048": res["bpc_2048"],
                        "bpc_32768": res["bpc_32768"],
                        "n_episodes": len(docs), "n_bytes": res["n_bytes"]})
            fh.flush()
            print(f"g={g:>2}: alpha={res['alpha']:.3f} H_inf={res['h_inf']:.3f} "
                  f"eps={len(docs)} bytes={res['n_bytes']}", flush=True)


if __name__ == "__main__":
    main()
