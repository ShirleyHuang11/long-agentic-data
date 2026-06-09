"""Measure beta (Cagnetta correlation-decay) for iter69-73 benchmark eval
rollouts, reusing measure_beta.py's operator-norm + initial-decay fit so the
new agentic points can join fig9 (the gamma-beta phase plane). Appends to
data/gamma_beta.csv. Self-contained loaders (the new slugs are not in
score_agentic_datasets.REGISTRY)."""
import csv, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle
from datasets import load_dataset
from measure_beta import corr_opnorm, LAGS, FIT_MAX


def ser_conv(row):  # conversations / messages : list of {role,content}
    msgs = row.get("conversations") or row.get("messages") or []
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    return "\n\n".join(f"[{m.get('role','?')}]\n{m.get('content') or ''}" for m in msgs)


# (path, split, key-hint, slug) ; key-hint unused (ser_conv auto-detects)
ENTRIES = [
    ("DCAgent/GPT-5-terminal-bench-2", "train", "terminal-bench2-gpt5"),
    ("DCAgent3/terminal_bench_2_Qwen3_32B_47000_46_20260609_022622", "train", "terminal-bench2-qwen3-32b"),
    ("AgentSuite/tau-bench-trajectories", "train", "taubench-deepseek-r1-eval"),
    ("togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories", "train", "coderforge-32b-swebench-verified-eval"),
]
out = "data/gamma_beta.csv"
fields = ["slug", "beta", "n_bytes"] + [f"opnorm_lag{n}" for n in LAGS]
have = {r["slug"] for r in csv.DictReader(open(out))}
rows = []
for path, split, slug in ENTRIES:
    if slug in have:
        print(f"skip {slug} (already measured)"); continue
    docs, size = [], 0
    for row in load_dataset(path, split=split, streaming=True):
        d = ser_conv(row)
        if not d:
            continue
        docs.append(d); size += len(d)
        if len(docs) >= lz_oracle.MAX_DOCS or size >= lz_oracle.MAX_BYTES:
            break
    arr = np.frombuffer(lz_oracle.build_corpus(docs), dtype=np.uint8)
    ops = {n: corr_opnorm(arr, n) for n in LAGS}
    xs = [math.log(n) for n in LAGS if n <= FIT_MAX]
    ys = [math.log(ops[n]) for n in LAGS if n <= FIT_MAX]
    beta = -np.polyfit(xs, ys, 1)[0]
    rows.append({"slug": slug, "beta": round(float(beta), 4), "n_bytes": len(arr),
                 **{f"opnorm_lag{n}": f"{ops[n]:.6g}" for n in LAGS}})
    print(f"{slug}: beta={beta:.3f}", flush=True)
with open(out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writerows(rows)
print(f"appended {len(rows)} beta rows")
