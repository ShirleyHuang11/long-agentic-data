"""Resilient beta batch (step 3 of the holographic bridge): measure beta
(Cagnetta op-norm correlation decay) for REGISTRY corpora that have alpha/H_inf
but no beta yet, so more real corpora can be placed on the holographic (beta,gamma)
phase diagram. Per-slug try/except + SKIP (loader-broken / OOM-crasher), like
measure_hurst_batch.py. Appends to data/gamma_beta.csv.
"""
import csv, os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle, score_agentic_datasets as sad
from measure_beta import corr_opnorm, LAGS, FIT_MAX

SKIP = {"app1-agentic-safety-sft", "factory-agent-task-rollouts",
        "nemotron-rl-conv-tool-pivot", "nemotron-rl-injection-v1",
        "nemotron-rl-swe-pivot", "nemotron-sft-v2-interactive",
        "nemotron-sft-v2-search", "nemotron-sft-v2-tool", "opencua-text"}
have_beta = {r["slug"] for r in csv.DictReader(open("data/gamma_beta.csv"))}
merged = {r["slug"] for r in csv.DictReader(open("data/merged_analysis.csv"))}
reg = {t[4]: t for t in sad.REGISTRY}
todo = sorted((set(reg) & merged) - have_beta - SKIP)
print(f"{len(todo)} corpora to attempt\n", flush=True)

out = "data/gamma_beta.csv"
fields = ["slug", "beta", "n_bytes"] + [f"opnorm_lag{n}" for n in LAGS]
ok = bad = 0
with open(out, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    for slug in todo:
        try:
            entry = reg[slug]; path, cfg, splits, ser = entry[:4]
            gk = entry[5] if len(entry) > 5 else None
            docs, size = [], 0
            for doc, _ in sad.iter_docs(path, cfg, splits, ser, group_key=gk):
                docs.append(doc); size += len(doc)
                if len(docs) >= lz_oracle.MAX_DOCS or size >= lz_oracle.MAX_BYTES:
                    break
            arr = np.frombuffer(lz_oracle.build_corpus(docs), dtype=np.uint8)
            ops = {n: corr_opnorm(arr, n) for n in LAGS}
            xs = [math.log(n) for n in LAGS if n <= FIT_MAX]
            ys = [math.log(ops[n]) for n in LAGS if n <= FIT_MAX]
            beta = -np.polyfit(xs, ys, 1)[0]
            w.writerow({"slug": slug, "beta": round(float(beta), 4), "n_bytes": len(arr),
                        **{f"opnorm_lag{n}": f"{ops[n]:.6g}" for n in LAGS}})
            f.flush(); ok += 1
            print(f"OK  {slug}: beta={beta:.3f}", flush=True)
        except Exception as e:
            bad += 1
            print(f"SKIP {slug}: {type(e).__name__}: {str(e)[:70]}", flush=True)
print(f"\ndone: {ok} ok, {bad} skipped", flush=True)
