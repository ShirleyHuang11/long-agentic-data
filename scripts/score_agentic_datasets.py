"""Score long-horizon agentic datasets with the LZ (alpha, H_inf) oracle.

Pilot registry of common long-horizon agentic benchmarks / trajectory corpora
on HuggingFace. Each entry defines how to serialize one episode (trajectory)
into a single text document: turns rendered as `[role]\ntext` blocks joined by
blank lines, matching the doc-per-episode protocol of the formal-math survey
(~1500 docs or 8 MB cap per dataset, see lz_oracle.py).

Usage:
    python scripts/score_agentic_datasets.py            # score all pilot entries
    python scripts/score_agentic_datasets.py --only nebius/SWE-agent-trajectories
Outputs:
    data/agentic_alpha_hinf.csv   (one row per dataset)
    samples_cache/<slug>.txt      (first 3 serialized episodes, for inspection)
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
from datasets import load_dataset
from huggingface_hub import HfApi


def _turns_chatml(msgs, role_key, text_key):
    return "\n\n".join(f"[{m.get(role_key, '?')}]\n{m.get(text_key) or ''}" for m in msgs)


def ser_nebius(row):
    return _turns_chatml(row["trajectory"], "role", "text"), len(row["trajectory"])


def ser_swesmith(row):
    msgs = row["messages"]
    if isinstance(msgs, str):  # stored as a JSON-encoded string
        msgs = json.loads(msgs)
    return _turns_chatml(msgs, "role", "content"), len(msgs)


def ser_conversations(row):
    return _turns_chatml(row["conversations"], "from", "value"), len(row["conversations"])


def ser_swebench(row):
    doc = (f"[problem_statement]\n{row['problem_statement']}\n\n"
           f"[hints]\n{row['hints_text']}\n\n"
           f"[patch]\n{row['patch']}\n\n[test_patch]\n{row['test_patch']}")
    return doc, 1


def ser_mind2web(row):
    # Compact action-trajectory view: task + per-step action representations.
    # (Raw `actions[*].cleaned_html` observations are ~MB-scale per step and
    # would let 2-3 episodes fill the whole 8 MB corpus; excluded in pilot.)
    steps = "\n".join(f"[step {i}] {a}" for i, a in enumerate(row["action_reprs"]))
    return f"[task]\n{row['confirmed_task']}\n\n{steps}", len(row["action_reprs"])


# (hf_path, config, split(s), serializer, slug)
REGISTRY = [
    ("nebius/SWE-agent-trajectories", None, ["train"], ser_nebius, "nebius-swe-agent-traj"),
    ("SWE-bench/SWE-smith-trajectories", None, ["xml"], ser_swesmith, "swe-smith-traj-xml"),
    ("THUDM/AgentInstruct", "default",
     ["os", "db", "alfworld", "webshop", "mind2web", "kg"],
     ser_conversations, "agentinstruct-all"),
    ("AgentGym/AgentTraj-L", None, ["train"], ser_conversations, "agentgym-agenttraj-l"),
    ("princeton-nlp/SWE-bench_Verified", None, ["test"], ser_swebench, "swe-bench-verified"),
    ("osunlp/Mind2Web", None, ["train"], ser_mind2web, "mind2web-actions"),
]


def iter_docs(path, cfg, splits, ser):
    for split in splits:
        ds = load_dataset(path, cfg, split=split, streaming=True)
        for row in ds:
            yield ser(row)


def score_entry(path, cfg, splits, ser, slug, seed_offset=0):
    docs, turn_counts = [], []
    size = 0
    it = iter_docs(path, cfg, splits, ser)
    if seed_offset:
        it = itertools.islice(it, seed_offset, None)
    for doc, n_turns in it:
        docs.append(doc)
        turn_counts.append(n_turns)
        size += len(doc)
        if len(docs) >= lz_oracle.MAX_DOCS or size >= lz_oracle.MAX_BYTES:
            break
    res = lz_oracle.score(docs)
    res.update(
        dataset=path, config=cfg or "", splits="+".join(splits), slug=slug,
        mean_turns=round(statistics.mean(turn_counts), 1) if turn_counts else 0,
        mean_doc_bytes=round(statistics.mean(len(d) for d in docs)) if docs else 0,
        n_episodes=len(docs),
    )
    os.makedirs("samples_cache", exist_ok=True)
    with open(f"samples_cache/{slug}.txt", "w", encoding="utf-8") as f:
        f.write("\n\n========== EPISODE BREAK ==========\n\n".join(docs[:3]))

    # Provenance sidecar: exact source revision + sampling protocol, so every
    # number in the registry is traceable to a pinned upstream snapshot.
    try:
        sha = HfApi().dataset_info(path).sha
    except Exception:
        sha = "unresolved"
    prov = {
        "hf_dataset": path,
        "hf_url": f"https://huggingface.co/datasets/{path}",
        "hf_revision_sha": sha,
        "config": cfg or "default",
        "splits": splits,
        "serializer": ser.__name__,
        "serializer_doc": (ser.__doc__ or "").strip(),
        "sampling": {
            "mode": "streaming, first-N episodes per split (no shuffle)",
            "max_docs": lz_oracle.MAX_DOCS,
            "max_bytes": lz_oracle.MAX_BYTES,
            "seed_offset": seed_offset,
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
    with open(f"data/provenance/{slug}.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    fields = ["dataset", "config", "splits", "slug", "alpha", "h_inf",
              "bpc_128", "bpc_2048", "bpc_32768",
              "n_episodes", "mean_turns", "mean_doc_bytes", "n_bytes"]
    rows = []
    for path, cfg, splits, ser, slug in REGISTRY:
        if args.only and args.only not in (path, slug):
            continue
        print(f"-> scoring {path} ({slug}) ...", flush=True)
        try:
            res = score_entry(path, cfg, splits, ser, slug)
            rows.append({k: res.get(k, "") for k in fields})
            print(f"   alpha={res['alpha']:.3f} H_inf={res['h_inf']:.3f} "
                  f"episodes={res['n_episodes']} mean_turns={res['mean_turns']} "
                  f"mean_bytes={res['mean_doc_bytes']}", flush=True)
        except Exception as e:
            print(f"   FAIL: {type(e).__name__}: {e}", flush=True)

    os.makedirs("data", exist_ok=True)
    out = "data/agentic_alpha_hinf.csv"
    write_header = not os.path.exists(out) or not args.only
    mode = "w" if not args.only else "a"
    with open(out, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
