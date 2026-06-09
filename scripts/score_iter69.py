"""iter-69 loop batch: benchmark EVAL rollouts across three benchmarks.

Adds three eval-trajectory corpora (agent rollouts collected ON a benchmark),
all from mid-size generators, to enrich the under-represented eval side of the
registry and test the merge-paper thesis that mid-model eval rollouts collapse
to the template floor exactly as distilled training mixtures do:

  CoderForge-32B on SWE-bench-Verified  (swe)
  Qwen3-32B on Terminal-Bench-2         (terminal -- NEW benchmark category)
  R2EGym-32B on GAIA-127                 (search; pairs vs frontier ii-agent 1.25)

Scores with the reference-exact lz_oracle.score (canonical, clamped H_inf) plus
the supplementary score_v3 (resolved flag), and appends full 17-column rows to
data/agentic_alpha_hinf.csv -- matching the existing header so DictReader
consumers (build_merged_table, figures) stay aligned. Writes provenance sidecars.
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


def _turns_chatml(msgs, rk, tk):
    return "\n\n".join(f"[{m.get(rk, '?')}]\n{m.get(tk) or ''}" for m in msgs)


def ser_msgs_jsonstr(row):
    msgs = row["messages"]
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    return _turns_chatml(msgs, "role", "content"), len(msgs)


def ser_conversations_rc(row):
    msgs = row["conversations"]
    return _turns_chatml(msgs, "role", "content"), len(msgs)


# (path, cfg, split, serializer, slug)
ENTRIES = [
    ("togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories",
     None, "train", ser_msgs_jsonstr, "coderforge-32b-swebench-verified-eval"),
    ("DCAgent3/terminal_bench_2_Qwen3_32B_47000_46_20260609_022622",
     None, "train", ser_conversations_rc, "terminal-bench2-qwen3-32b"),
    ("DCAgent2/gaia_127_R2EGym_32B_Agent_20260505_060909",
     None, "train", ser_conversations_rc, "gaia127-r2egym-32b-eval"),
]

FIELDS = ["dataset", "config", "splits", "slug", "alpha", "h_inf", "h_inf_raw",
          "bpc_128", "bpc_2048", "bpc_32768", "n_episodes", "mean_turns",
          "mean_doc_bytes", "n_bytes", "h_inf_v3", "h_inf_stderr", "resolved"]

OUT = "data/agentic_alpha_hinf.csv"
rows = []
for path, cfg, split, ser, slug in ENTRIES:
    print(f"-> {path} ({slug})", flush=True)
    docs, turns = [], []
    size = 0
    ds = load_dataset(path, cfg, split=split, streaming=True)
    for row in ds:
        try:
            doc, n = ser(row)
        except Exception as e:
            print("   skip row:", type(e).__name__, str(e)[:80])
            continue
        if not doc:
            continue
        docs.append(doc)
        turns.append(n)
        size += len(doc)
        if len(docs) >= lz_oracle.MAX_DOCS or size >= lz_oracle.MAX_BYTES:
            break
    s = lz_oracle.score(docs)
    try:  # supplementary diagnostic; skip if scipy unavailable
        v3 = lz_oracle.score_v3(docs)
    except ModuleNotFoundError:
        v3 = {}
    try:
        sha = HfApi().dataset_info(path).sha
    except Exception:
        sha = "unresolved"
    rec = {
        "dataset": path, "config": cfg or "", "splits": split, "slug": slug,
        "alpha": s["alpha"], "h_inf": s["h_inf"], "h_inf_raw": s["h_inf_raw"],
        "bpc_128": s["bpc_128"], "bpc_2048": s["bpc_2048"], "bpc_32768": s["bpc_32768"],
        "n_episodes": len(docs),
        "mean_turns": round(statistics.mean(turns), 1) if turns else 0,
        "mean_doc_bytes": round(statistics.mean(len(d) for d in docs)) if docs else 0,
        "n_bytes": s["n_bytes"],
        "h_inf_v3": v3.get("h_inf", ""), "h_inf_stderr": v3.get("h_inf_stderr", ""),
        "resolved": v3.get("resolved", ""),
    }
    rows.append(rec)
    print(f"   alpha={s['alpha']:.3f} H_inf={s['h_inf']:.3f} (raw {s['h_inf_raw']:.3f}) "
          f"BPC@32K={s['bpc_32768']:.3f} ep={len(docs)} turns={rec['mean_turns']} "
          f"v3={v3.get('h_inf')} resolved={v3.get('resolved')}", flush=True)
    os.makedirs("samples_cache", exist_ok=True)
    with open(f"samples_cache/{slug}.txt", "w", encoding="utf-8") as f:
        f.write("\n\n===== EP BREAK =====\n\n".join(docs[:3]))
    prov = {"hf_dataset": path, "hf_url": f"https://huggingface.co/datasets/{path}",
            "hf_revision_sha": sha, "config": cfg or "default", "splits": [split],
            "serializer": ser.__name__,
            "sampling": {"mode": "streaming first-N", "max_docs": lz_oracle.MAX_DOCS,
                         "max_bytes": lz_oracle.MAX_BYTES},
            "oracle": {"script": "scripts/lz_oracle.py",
                       "compressor": f"zstd level {lz_oracle.ZSTD_LEVEL}",
                       "context_points": list(lz_oracle.N_POINTS)},
            "collected_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "result": {k: rec[k] for k in ("alpha", "h_inf", "bpc_32768",
                                           "n_episodes", "mean_turns", "n_bytes")}}
    os.makedirs("data/provenance", exist_ok=True)
    with open(f"data/provenance/{slug}.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)

with open(OUT, "a", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=FIELDS).writerows(rows)
print(f"\nappended {len(rows)} rows -> {OUT}")
