"""Rebuild data/agentic_alpha_hinf.csv from data/provenance/*.json sidecars.

The provenance sidecar is the source of truth for each scoring run (written
atomically per dataset); the CSV is a derived view. Used after the 2026-06-05
disk-full incident killed the CSV write mid-run.
"""

import csv
import glob
import json

FIELDS = ["dataset", "config", "splits", "slug", "alpha", "h_inf",
          "bpc_128", "bpc_2048", "bpc_32768",
          "n_episodes", "mean_turns", "mean_doc_bytes", "n_bytes",
          "hf_revision_sha", "collected_at_utc"]

rows = []
for path in sorted(glob.glob("data/provenance/*.json")):
    p = json.load(open(path, encoding="utf-8"))
    r = p["result"]
    rows.append({
        "dataset": p["hf_dataset"], "config": p["config"],
        "splits": "+".join(p["splits"]), "slug": path.split("/")[-1][:-5],
        "hf_revision_sha": p["hf_revision_sha"],
        "collected_at_utc": p["collected_at_utc"],
        **{k: r[k] for k in FIELDS if k in r},
    })

with open("data/agentic_alpha_hinf.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows -> data/agentic_alpha_hinf.csv")
