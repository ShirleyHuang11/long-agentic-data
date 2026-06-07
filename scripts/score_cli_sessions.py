"""Score cfahlgren1/agent-sessions-list (raw JSONL direct-read).

First real release in the coding-CLI session category (Claude Code / Codex /
pi session dumps). One document = one session file; text = recursively
collected `content`/`text` string values from each event line (uuids,
timestamps and other metadata excluded so they don't inflate H_inf).

Heavy caveats recorded in the registry row: n=10 sessions, mixed formats,
one 3.2 MB pi session dominates the corpus.
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

REPO = "cfahlgren1/agent-sessions-list"
SLUG = "cli-agent-sessions-sampler"


def collect_text(x, out):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in ("content", "text") and isinstance(v, str):
                out.append(v)
            else:
                collect_text(v, out)
    elif isinstance(x, list):
        for v in x:
            collect_text(v, out)


def main():
    api = HfApi()
    files = sorted(s.rfilename for s in api.dataset_info(REPO).siblings
                   if s.rfilename.startswith("sessions/")
                   and s.rfilename.endswith(".jsonl"))
    docs, turns = [], []
    for f in files:
        p = hf_hub_download(REPO, f, repo_type="dataset")
        parts, n_events = [], 0
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            n_events += 1
            out = []
            collect_text(d, out)
            if out:
                parts.append("\n".join(out))
        if parts:
            docs.append("\n\n".join(parts))
            turns.append(n_events)
            print(f"  {f}: events={n_events} text={len(docs[-1])}B", flush=True)

    res = lz_oracle.score(docs)
    res.update(dataset=REPO, config="sessions/*", splits="raw", slug=SLUG,
               mean_turns=round(statistics.mean(turns), 1) if turns else 0,
               mean_doc_bytes=round(statistics.mean(len(d) for d in docs)),
               n_episodes=len(docs))
    print(f"alpha={res['alpha']:.3f} H_inf={res['h_inf']:.3f} "
          f"episodes={res['n_episodes']} mean_turns={res['mean_turns']} "
          f"mean_bytes={res['mean_doc_bytes']}", flush=True)

    os.makedirs("samples_cache", exist_ok=True)
    with open(f"samples_cache/{SLUG}.txt", "w", encoding="utf-8") as f:
        f.write("\n\n========== EPISODE BREAK ==========\n\n".join(docs[:3]))
    prov = {
        "hf_dataset": REPO,
        "hf_url": f"https://huggingface.co/datasets/{REPO}",
        "hf_revision_sha": api.dataset_info(REPO).sha,
        "config": "sessions/* (claude x4, codex x3, pi x3)",
        "group_key": None, "splits": ["raw"],
        "serializer": "score_cli_sessions.py collect_text (content/text values only)",
        "serializer_doc": "",
        "sampling": {"mode": "all 10 session files", "max_docs": lz_oracle.MAX_DOCS,
                     "max_bytes": lz_oracle.MAX_BYTES, "seed_offset": 0},
        "oracle": {"script": "scripts/lz_oracle.py",
                   "compressor": f"zstd level {lz_oracle.ZSTD_LEVEL}",
                   "context_points": list(lz_oracle.N_POINTS)},
        "collected_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "result": {k: res[k] for k in
                   ("alpha", "h_inf", "bpc_128", "bpc_2048", "bpc_32768",
                    "n_episodes", "mean_turns", "mean_doc_bytes", "n_bytes")},
    }
    with open(f"data/provenance/{SLUG}.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)

    fields = ["dataset", "config", "splits", "slug", "alpha", "h_inf",
              "bpc_128", "bpc_2048", "bpc_32768",
              "n_episodes", "mean_turns", "mean_doc_bytes", "n_bytes"]
    with open("data/agentic_alpha_hinf.csv", "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(
            {k: res.get(k, "") for k in fields})
    print("appended -> data/agentic_alpha_hinf.csv", flush=True)


if __name__ == "__main__":
    main()
