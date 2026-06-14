"""Resilient Hurst batch: measures Hurst for ALL remaining REGISTRY corpora that
have reuse_dist but no Hurst yet, wrapping each in try/except so a single bad
loader (e.g. datasets 'Feature type Json' metadata errors) skips that corpus
instead of aborting the whole run. Reuses measure_hurst's math. Appends to
data/hurst.csv with per-row flush.
"""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle
import score_agentic_datasets as sad
from measure_hurst import surprisal_series, rs_hurst, WINDOWS

have_hurst = {r["slug"] for r in csv.DictReader(open("data/hurst.csv"))}
have_reuse = {r["slug"] for r in csv.DictReader(open("data/credit_horizon.csv"))}
reg_by_slug = {t[4]: t for t in sad.REGISTRY}
todo = sorted((set(reg_by_slug) & have_reuse) - have_hurst)
print(f"{len(todo)} corpora to attempt\n", flush=True)

out = "data/hurst.csv"
fields = ["slug", "hurst", "n_bytes"] + [f"rs_n{n}" for n in WINDOWS]
ok = bad = 0
with open(out, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    for slug in todo:
        try:
            entry = reg_by_slug[slug]
            path, cfg, splits, ser = entry[:4]
            gk = entry[5] if len(entry) > 5 else None
            docs, size = [], 0
            for doc, _ in sad.iter_docs(path, cfg, splits, ser, group_key=gk):
                docs.append(doc); size += len(doc)
                if len(docs) >= lz_oracle.MAX_DOCS or size >= lz_oracle.MAX_BYTES:
                    break
            arr = np.frombuffer(lz_oracle.build_corpus(docs), dtype=np.uint8)
            h, pts = rs_hurst(surprisal_series(arr))
            w.writerow({"slug": slug, "hurst": round(h, 4), "n_bytes": len(arr),
                        **{f"rs_n{n}": f"{rs:.4g}" for n, rs in pts}})
            f.flush(); ok += 1
            print(f"OK  {slug}: H={h:.3f}", flush=True)
        except Exception as e:
            bad += 1
            print(f"SKIP {slug}: {type(e).__name__}: {str(e)[:80]}", flush=True)
print(f"\ndone: {ok} ok, {bad} skipped", flush=True)
