"""Validate the 4 structural-axis proxies (scaffold_frac, structure_density,
neardup, reuse_dist) that were estimated from only 3 cached episodes. Reload up to
MAX_EP episodes per corpus via the REGISTRY and recompute each axis, then check
whether the many-episode estimate rank-agrees with the 3-episode one
(Spearman across corpora). High agreement => the 3-episode proxies are stable and
can be promoted from 'directional' to 'validated'. Resilient per-corpus.
"""
import os, sys, csv, re
from itertools import combinations
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle, score_agentic_datasets as sad

MAX_EP = 25
WORD = re.compile(r"\w+")
CWORD = re.compile(r"[a-zA-Z][a-zA-Z_]{4,}")
K = 5
SKIP = {"app1-agentic-safety-sft", "factory-agent-task-rollouts",
        "nemotron-rl-conv-tool-pivot", "nemotron-rl-injection-v1",
        "nemotron-rl-swe-pivot", "nemotron-sft-v2-interactive",
        "nemotron-sft-v2-search", "nemotron-sft-v2-tool", "opencua-text"}

def axes(eps):
    # scaffold_frac: byte share of lines shared across >=2 episodes
    seen, lines_per = {}, []
    for i, e in enumerate(eps):
        ls = [l.strip() for l in e.splitlines() if len(l.strip()) > 3]
        lines_per.append(ls)
        for l in set(ls):
            seen.setdefault(l, set()).add(i)
    shared = {l for l, ids in seen.items() if len(ids) >= 2}
    tot = sum(len(l) for ls in lines_per for l in ls)
    sca = sum(len(l) for ls in lines_per for l in ls if l in shared) / tot if tot else None
    # structure_density
    ns = [c for e in eps for c in e if not c.isspace()]
    stru = sum(1 for c in ns if not c.isalpha()) / len(ns) if ns else None
    # neardup: mean pairwise word-5-gram Jaccard
    sh = []
    for e in eps:
        t = [w.lower() for w in WORD.findall(e)]
        sh.append({tuple(t[i:i+K]) for i in range(len(t)-K+1)} if len(t) >= K else set())
    sims = []
    for a, b in combinations(sh, 2):
        u = len(a | b); sims.append(len(a & b)/u if u else 0.0)
    nd = sum(sims)/len(sims) if sims else None
    # reuse_dist
    rds = []
    for e in eps:
        t = [w.lower() for w in CWORD.findall(e)]
        n = len(t)
        if n < 20: continue
        pos = {}
        for i, w in enumerate(t): pos.setdefault(w, []).append(i)
        sp = [(p[-1]-p[0])/n for p in pos.values() if len(p) >= 2]
        if sp: rds.append(sum(sp)/len(sp))
    rd = sum(rds)/len(rds) if rds else None
    return sca, stru, nd, rd

have = {r["slug"] for r in csv.DictReader(open("data/scaffold_frac.csv"))}
reg = [t for t in sad.REGISTRY if t[4] in have and t[4] not in SKIP]
print(f"{len(reg)} corpora to re-measure (MAX_EP={MAX_EP})\n", flush=True)
rows = []
for t in reg:
    slug = t[4]
    try:
        path, cfg, splits, ser = t[:4]; gk = t[5] if len(t) > 5 else None
        eps = []
        for doc, _ in sad.iter_docs(path, cfg, splits, ser, group_key=gk):
            eps.append(doc)
            if len(eps) >= MAX_EP: break
        if len(eps) < 5:
            print(f"skip {slug}: only {len(eps)} eps", flush=True); continue
        sca, stru, nd, rd = axes(eps)
        rows.append({"slug": slug, "n_ep": len(eps), "scaffold_m": sca,
                     "structure_m": stru, "neardup_m": nd, "reuse_m": rd})
        print(f"OK {slug} ({len(eps)} eps)", flush=True)
    except Exception as e:
        print(f"skip {slug}: {type(e).__name__}", flush=True)

mm = pd.DataFrame(rows)
mm.to_csv("data/proxies_multiep.csv", index=False)
print(f"\nre-measured {len(mm)} corpora -> data/proxies_multiep.csv\n", flush=True)

# stability: Spearman(3-episode value, many-episode value) across corpora
ref = {"scaffold_m": ("scaffold_frac.csv", "scaffold_frac"),
       "structure_m": ("structure_density.csv", "structure_density"),
       "neardup_m": ("neardup.csv", "neardup"),
       "reuse_m": ("credit_horizon.csv", "reuse_dist")}
print("=== rank-stability: Spearman(3-ep proxy, ~25-ep estimate) ===")
for mcol, (f, c) in ref.items():
    r3 = pd.read_csv(f"data/{f}")[["slug", c]]
    j = mm.merge(r3, on="slug").dropna(subset=[mcol, c])
    print(f"  {c:18s}: {j[mcol].corr(j[c], method='spearman'):+.2f}  (n={len(j)})")
