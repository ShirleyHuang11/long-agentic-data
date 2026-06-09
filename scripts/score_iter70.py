"""iter-70 loop batch: FRONTIER Terminal-Bench-2 rollouts.

Pairs against iter-69's mid-size Qwen3-32B on Terminal-Bench-2 (H_inf=0.00) to
give a clean frontier-vs-mid generator contrast on the SAME benchmark:
  GPT-5 on Terminal-Bench-2          (frontier)
  Claude-Sonnet-4.5 on Terminal-Bench-2 (frontier)

Reference-exact lz_oracle.score; appends full 17-column rows to
data/agentic_alpha_hinf.csv. Same serializer/schema as iter 69.
"""
import csv
import datetime
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle
from datasets import load_dataset
from huggingface_hub import HfApi


def ser_conversations_rc(row):
    msgs = row["conversations"]
    return "\n\n".join(f"[{m.get('role', '?')}]\n{m.get('content') or ''}" for m in msgs), len(msgs)


ENTRIES = [
    ("DCAgent/GPT-5-terminal-bench-2", None, "train", "terminal-bench2-gpt5"),
    ("DCAgent/claude-sonnet-4-5-terminal-bench-2", None, "train", "terminal-bench2-claude-sonnet45"),
]

FIELDS = ["dataset", "config", "splits", "slug", "alpha", "h_inf", "h_inf_raw",
          "bpc_128", "bpc_2048", "bpc_32768", "n_episodes", "mean_turns",
          "mean_doc_bytes", "n_bytes", "h_inf_v3", "h_inf_stderr", "resolved"]
OUT = "data/agentic_alpha_hinf.csv"
rows = []
for path, cfg, split, slug in ENTRIES:
    print(f"-> {path} ({slug})", flush=True)
    docs, turns, size = [], [], 0
    for row in load_dataset(path, cfg, split=split, streaming=True):
        doc, n = ser_conversations_rc(row)
        if not doc:
            continue
        docs.append(doc); turns.append(n); size += len(doc)
        if len(docs) >= lz_oracle.MAX_DOCS or size >= lz_oracle.MAX_BYTES:
            break
    s = lz_oracle.score(docs)
    try:
        sha = HfApi().dataset_info(path).sha
    except Exception:
        sha = "unresolved"
    rec = {"dataset": path, "config": cfg or "", "splits": split, "slug": slug,
           "alpha": s["alpha"], "h_inf": s["h_inf"], "h_inf_raw": s["h_inf_raw"],
           "bpc_128": s["bpc_128"], "bpc_2048": s["bpc_2048"], "bpc_32768": s["bpc_32768"],
           "n_episodes": len(docs),
           "mean_turns": round(statistics.mean(turns), 1) if turns else 0,
           "mean_doc_bytes": round(statistics.mean(len(d) for d in docs)) if docs else 0,
           "n_bytes": s["n_bytes"], "h_inf_v3": "", "h_inf_stderr": "", "resolved": ""}
    rows.append(rec)
    print(f"   alpha={s['alpha']:.3f} H_inf={s['h_inf']:.3f} (raw {s['h_inf_raw']:.3f}) "
          f"BPC@32K={s['bpc_32768']:.3f} ep={len(docs)} turns={rec['mean_turns']}", flush=True)
    os.makedirs("data/provenance", exist_ok=True)
    prov = {"hf_dataset": path, "hf_url": f"https://huggingface.co/datasets/{path}",
            "hf_revision_sha": sha, "splits": [split], "serializer": "ser_conversations_rc",
            "sampling": {"mode": "streaming first-N", "max_docs": lz_oracle.MAX_DOCS,
                         "max_bytes": lz_oracle.MAX_BYTES},
            "oracle": {"compressor": f"zstd level {lz_oracle.ZSTD_LEVEL}",
                       "context_points": list(lz_oracle.N_POINTS)},
            "collected_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "result": {k: rec[k] for k in ("alpha", "h_inf", "bpc_32768", "n_episodes", "mean_turns", "n_bytes")}}
    with open(f"data/provenance/{slug}.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)

with open(OUT, "a", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=FIELDS).writerows(rows)
print(f"\nappended {len(rows)} rows -> {OUT}")
