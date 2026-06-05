"""Direct loader + LZ scoring for AI-Secure/DTap-Bench-Agent-Trajectories.

The repo is JSON-per-episode (no parquet / loading script), so `datasets`
cannot stream it — same situation as the Nemotron-v2 JSONL direct-read
(loop iter 17). One episode file = one trajectory:
    {task_info, traj_info, trajectory: [{role, state|action, metadata, step_id}]}

We score one model slice (config = e.g. "claudesdk/claude-opus-4-6"),
round-robin across the <domain>/<benign|malicious> groups so the 8 MB corpus
cap doesn't collapse onto a single domain. Document = `[role]\ntext` blocks,
matching the registry serialization protocol.

Usage:
    python scripts/score_dtap_direct.py [--config claudesdk/claude-opus-4-6] [--slug dtap-claude-opus-46]
"""

import argparse
import csv
import datetime
import itertools
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle
from huggingface_hub import HfApi, hf_hub_download

REPO = "AI-Secure/DTap-Bench-Agent-Trajectories"


def ser_dtap(traj):
    parts = []
    for s in traj:
        text = s.get("state") or s.get("action") or ""
        parts.append(f"[{s.get('role', '?')}]\n{text}")
    return "\n\n".join(parts), len(traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="claudesdk/claude-opus-4-6")
    ap.add_argument("--slug", default="dtap-claude-opus-46")
    args = ap.parse_args()

    api = HfApi()
    info = api.dataset_info(REPO)
    files = [s.rfilename for s in info.siblings
             if s.rfilename.startswith(args.config + "/")
             and s.rfilename.endswith(".json")
             and "judge_result" not in s.rfilename]
    # group by <domain>/<benign|malicious>, round-robin across groups
    groups = {}
    for f in sorted(files):
        key = "/".join(f.split("/")[2:4])
        groups.setdefault(key, []).append(f)
    order = [f for batch in itertools.zip_longest(*groups.values())
             for f in batch if f is not None]

    docs, turn_counts = [], []
    size = 0
    for f in order:
        try:
            p = hf_hub_download(REPO, f, repo_type="dataset")
            d = json.load(open(p, encoding="utf-8"))
            doc, n_turns = ser_dtap(d.get("trajectory") or [])
        except Exception as e:
            print(f"  skip {f}: {type(e).__name__}: {e}", flush=True)
            continue
        if not doc:
            continue
        docs.append(doc)
        turn_counts.append(n_turns)
        size += len(doc)
        if len(docs) >= lz_oracle.MAX_DOCS or size >= lz_oracle.MAX_BYTES:
            break
    print(f"sampled {len(docs)} episodes / {size} bytes "
          f"from {len(groups)} domain-label groups", flush=True)

    res = lz_oracle.score(docs)
    res.update(
        dataset=REPO, config=args.config, splits="train", slug=args.slug,
        mean_turns=round(statistics.mean(turn_counts), 1) if turn_counts else 0,
        mean_doc_bytes=round(statistics.mean(len(d) for d in docs)) if docs else 0,
        n_episodes=len(docs),
    )
    print(f"alpha={res['alpha']:.3f} H_inf={res['h_inf']:.3f} "
          f"episodes={res['n_episodes']} mean_turns={res['mean_turns']} "
          f"mean_bytes={res['mean_doc_bytes']}", flush=True)

    os.makedirs("samples_cache", exist_ok=True)
    with open(f"samples_cache/{args.slug}.txt", "w", encoding="utf-8") as f:
        f.write("\n\n========== EPISODE BREAK ==========\n\n".join(docs[:3]))

    prov = {
        "hf_dataset": REPO,
        "hf_url": f"https://huggingface.co/datasets/{REPO}",
        "hf_revision_sha": info.sha,
        "config": args.config,
        "group_key": None,
        "splits": ["train"],
        "serializer": "ser_dtap (direct JSON-per-episode loader, scripts/score_dtap_direct.py; "
                      "round-robin across domain/{benign,malicious} groups)",
        "serializer_doc": "",
        "sampling": {
            "mode": "round-robin over domain-label groups, first-k file per group (no shuffle)",
            "max_docs": lz_oracle.MAX_DOCS,
            "max_bytes": lz_oracle.MAX_BYTES,
            "seed_offset": 0,
        },
        "oracle": {
            "script": "scripts/lz_oracle.py",
            "compressor": f"zstd level {lz_oracle.ZSTD_LEVEL}",
            "context_points": list(lz_oracle.N_POINTS),
        },
        "collected_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "result": {k: res[k] for k in
                   ("alpha", "h_inf", "bpc_128", "bpc_2048", "bpc_32768",
                    "n_episodes", "mean_turns", "mean_doc_bytes", "n_bytes")},
    }
    os.makedirs("data/provenance", exist_ok=True)
    with open(f"data/provenance/{args.slug}.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)

    fields = ["dataset", "config", "splits", "slug", "alpha", "h_inf",
              "bpc_128", "bpc_2048", "bpc_32768",
              "n_episodes", "mean_turns", "mean_doc_bytes", "n_bytes"]
    with open("data/agentic_alpha_hinf.csv", "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(
            {k: res.get(k, "") for k in fields})
    print(f"appended -> data/agentic_alpha_hinf.csv", flush=True)


if __name__ == "__main__":
    main()
