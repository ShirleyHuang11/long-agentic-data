import sys, os, csv
sys.path.insert(0,"scripts")
import lz_oracle as lz, score_agentic_datasets as sad
out="data/agentic_hinf_v3.csv"
done=set()
if os.path.exists(out): done={r["slug"] for r in csv.DictReader(open(out))}
fields=["slug","h_inf_v3","h_inf_stderr","resolved","alpha_v3","bpc_32k","n_points","n_bytes"]
f=open(out,"a",newline=""); w=csv.DictWriter(f,fieldnames=fields)
if not done: w.writeheader()
for e in sad.REGISTRY:
    path,cfg,splits,ser,slug=e[:5]; gk=e[5] if len(e)>5 else None; drop=e[6] if len(e)>6 else None
    if slug in done: continue
    try:
        dd,sz=[],0
        for d,_ in sad.iter_docs(path,cfg,splits,ser,group_key=gk,drop_cols=drop):
            dd.append(d); sz+=len(d)
            if sz>=lz.MAX_BYTES or len(dd)>=lz.MAX_DOCS: break
        r=lz.score_v3(dd)
        w.writerow({"slug":slug,"h_inf_v3":round(r.get("h_inf",float('nan')),4),
            "h_inf_stderr":round(r.get("h_inf_stderr",float('nan')),4),
            "resolved":r.get("resolved",False),"alpha_v3":round(r.get("alpha",float('nan')),4),
            "bpc_32k":round(r.get("bpc_32k",float('nan')),4),"n_points":r.get("n_points",0),"n_bytes":r.get("n_bytes",0)})
        f.flush()
        print(f"{slug:38s} h_inf={r.get('h_inf',float('nan')):+.3f} ±{r.get('h_inf_stderr',float('nan')):.3f} resolved={r.get('resolved')}",flush=True)
    except Exception as ex:
        print(f"{slug:38s} FAIL {type(ex).__name__}: {str(ex)[:70]}",flush=True)
f.close(); print("DONE")
