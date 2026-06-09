"""iter-100: two new SWE training-trajectory sources.
ElenaFu/SWE-agent-trajectories (swe-agent-llama-70b; col 'trajectory') and
SWE-Factory/DeepSWE-Agent-Kimi-K2-Trajectories-2.8K (Kimi-K2; col 'messages').
Reference-exact score; full 17-col rows."""
import csv, datetime, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle
from datasets import load_dataset
from huggingface_hub import HfApi
ENTRIES=[("ElenaFu/SWE-agent-trajectories","train","trajectory","swe-agent-llama70b-traj"),
         ("SWE-Factory/DeepSWE-Agent-Kimi-K2-Trajectories-2.8K","train","messages","deepswe-kimi-k2-traj")]
FIELDS=["dataset","config","splits","slug","alpha","h_inf","h_inf_raw","bpc_128","bpc_2048",
        "bpc_32768","n_episodes","mean_turns","mean_doc_bytes","n_bytes","h_inf_v3","h_inf_stderr","resolved"]
rows=[]
for path,split,col,slug in ENTRIES:
    print(f"-> {path} ({slug})",flush=True)
    docs,turns,size=[],[],0
    for row in load_dataset(path,split=split,streaming=True):
        msgs=row[col]
        if isinstance(msgs,str): msgs=json.loads(msgs)
        doc="\n\n".join(f"[{m.get('role','?')}]\n{m.get('content') or ''}" for m in msgs)
        if not doc: continue
        docs.append(doc); turns.append(len(msgs)); size+=len(doc)
        if len(docs)>=lz_oracle.MAX_DOCS or size>=lz_oracle.MAX_BYTES: break
    s=lz_oracle.score(docs)
    try: sha=HfApi().dataset_info(path).sha
    except Exception: sha="unresolved"
    rec={"dataset":path,"config":"","splits":split,"slug":slug,"alpha":s["alpha"],"h_inf":s["h_inf"],
         "h_inf_raw":s["h_inf_raw"],"bpc_128":s["bpc_128"],"bpc_2048":s["bpc_2048"],"bpc_32768":s["bpc_32768"],
         "n_episodes":len(docs),"mean_turns":round(statistics.mean(turns),1) if turns else 0,
         "mean_doc_bytes":round(statistics.mean(len(d) for d in docs)) if docs else 0,
         "n_bytes":s["n_bytes"],"h_inf_v3":"","h_inf_stderr":"","resolved":""}
    rows.append(rec)
    print(f"   alpha={s['alpha']:.3f} H_inf={s['h_inf']:.3f} BPC@32K={s['bpc_32768']:.3f} ep={len(docs)} turns={rec['mean_turns']}",flush=True)
    os.makedirs("data/provenance",exist_ok=True)
    json.dump({"hf_dataset":path,"hf_revision_sha":sha,"serializer":f"{col}_rc",
               "oracle":{"compressor":f"zstd {lz_oracle.ZSTD_LEVEL}","context_points":list(lz_oracle.N_POINTS)},
               "collected_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "result":{k:rec[k] for k in ("alpha","h_inf","bpc_32768","n_episodes","mean_turns","n_bytes")}},
              open(f"data/provenance/{slug}.json","w"),indent=2)
with open("data/agentic_alpha_hinf.csv","a",newline="") as f:
    csv.DictWriter(f,fieldnames=FIELDS).writerows(rows)
print(f"appended {len(rows)} rows")
