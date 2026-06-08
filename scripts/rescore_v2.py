"""Re-score the registry with lz_oracle.score_v2 (7-point floor fit) to correct
the 3-point negative-clamp artifact. Writes data/agentic_hinf_v2.csv alongside
the original (preserved). One row per REGISTRY entry reachable via iter_docs.
"""
import sys, os, csv
sys.path.insert(0, "scripts")
import lz_oracle as lz, score_agentic_datasets as sad

old = {r["slug"]: r for r in csv.DictReader(open("data/agentic_alpha_hinf.csv"))}
out = "data/agentic_hinf_v2.csv"
fields = ["slug","h_inf_old","h_inf_resolved","h_inf_v2","alpha_v2","fit_r2","n_points","n_bytes"]
done = set()
if os.path.exists(out):
    done = {r["slug"] for r in csv.DictReader(open(out))}
f = open(out, "a", newline="")
w = csv.DictWriter(f, fieldnames=fields)
if not done: w.writeheader()

for e in sad.REGISTRY:
    path,cfg,splits,ser,slug = e[:5]
    gk = e[5] if len(e)>5 else None
    drop = e[6] if len(e)>6 else None
    if slug in done: continue
    try:
        docs,size=[],0
        for d,_ in sad.iter_docs(path,cfg,splits,ser,group_key=gk,drop_cols=drop):
            docs.append(d); size+=len(d)
            if size>=lz.MAX_BYTES or len(docs)>=lz.MAX_DOCS: break
        r = lz.score_v2(docs)
        w.writerow({"slug":slug,
                    "h_inf_old": old.get(slug,{}).get("h_inf",""),
                    "h_inf_resolved": round(r["h_inf_resolved"],4),
                    "h_inf_v2": round(r["h_inf"],4),
                    "alpha_v2": round(r["alpha"],4),
                    "fit_r2": round(r["fit_r2"],4) if r["fit_r2"]==r["fit_r2"] else "",
                    "n_points": r["n_points"], "n_bytes": r["n_bytes"]})
        f.flush()
        print(f"{slug:40s} old={old.get(slug,{}).get('h_inf','?'):>6} -> resolved={r['h_inf_resolved']:+.3f} R2={r['fit_r2']:.3f} pts={r['n_points']}", flush=True)
    except Exception as ex:
        print(f"{slug:40s} FAIL {type(ex).__name__}: {str(ex)[:80]}", flush=True)
f.close()
print("DONE")
