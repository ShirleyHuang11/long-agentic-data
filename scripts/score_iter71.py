"""iter-71 loop batch: FRONTIER SWE-bench-Verified rollout.

Completes the per-benchmark frontier-vs-mid contrast: Claude-Sonnet-4.5 on
SWE-bench-Verified (frontier) vs iter-69 CoderForge-32B on the same benchmark
(mid, H_inf=0.83). Reference-exact lz_oracle.score; appends a full 17-column row.
"""
import csv
import datetime
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle
from huggingface_hub import HfApi, hf_hub_download

# This repo is one .traj.json per instance; the HF auto-JSON loader chokes on a
# few malformed files (loader-inference issue, finding 10), so we read files
# directly and skip unparseable ones. Each file's top-level "messages" is the
# trajectory.
ENTRIES = [
    ("livesweagent/claude-sonnet-4-5_swebench_verified_traj", None, "validation",
     "messages", "swebench-verified-claude-sonnet45-eval"),
]
FIELDS = ["dataset", "config", "splits", "slug", "alpha", "h_inf", "h_inf_raw",
          "bpc_128", "bpc_2048", "bpc_32768", "n_episodes", "mean_turns",
          "mean_doc_bytes", "n_bytes", "h_inf_v3", "h_inf_stderr", "resolved"]
OUT = "data/agentic_alpha_hinf.csv"
rows = []
for path, cfg, split, mkey, slug in ENTRIES:
    print(f"-> {path} ({slug})", flush=True)
    api = HfApi()
    traj_files = sorted(f for f in api.list_repo_files(path, repo_type="dataset")
                        if f.endswith(".traj.json"))
    docs, turns, size, skipped = [], [], 0, 0
    for fn in traj_files:
        try:
            fp = hf_hub_download(path, fn, repo_type="dataset")
            with open(fp) as fh:
                obj = json.load(fh)
        except Exception:
            skipped += 1
            continue
        msgs = obj.get(mkey) or []
        doc = "\n\n".join(f"[{m.get('role', '?')}]\n{m.get('content') or ''}" for m in msgs)
        if not doc:
            continue
        docs.append(doc); turns.append(len(msgs)); size += len(doc)
        if len(docs) >= lz_oracle.MAX_DOCS or size >= lz_oracle.MAX_BYTES:
            break
    print(f"   loaded {len(docs)} docs, skipped {skipped} unparseable", flush=True)
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
            "hf_revision_sha": sha, "splits": [split], "serializer": "messages_rc",
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
