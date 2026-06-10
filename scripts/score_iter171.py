"""iter-171: FleetAI/trajectories-tool_use-difficult-envs-raw.
Frontier-model MIX (gemini-3-pro, claude-sonnet-4.5, gemini-3-flash, kimi-k2.5,
opus-4.6/4.5, gpt-5.2) tool-use rollouts on *difficult* environments — harder-env tool
domain. Reference-exact score; full 17-col row. (iter-172: relabeled gemini3pro ->
frontiermix after auditing the model composition — the row pools 7 frontier models.)"""
import csv, datetime, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle
from datasets import load_dataset
from huggingface_hub import HfApi
ENTRIES=[("FleetAI/trajectories-tool_use-difficult-envs-raw","tool_use","messages","fleet-tooluse-difficult-frontiermix")]
FIELDS=["dataset","config","splits","slug","alpha","h_inf","h_inf_raw","bpc_128","bpc_2048",
        "bpc_32768","n_episodes","mean_turns","mean_doc_bytes","n_bytes","h_inf_v3","h_inf_stderr","resolved"]
rows=[]
for path,split,col,slug in ENTRIES:
    print(f"-> {path} ({slug})",flush=True)
    docs,turns,size=[],[],0
    for row in load_dataset(path,split=split,streaming=True):
        msgs=row[col]
        if isinstance(msgs,str): msgs=json.loads(msgs)
        parts=[]
        for m in msgs:
            c=m.get('content') or ''
            if isinstance(c,list): c="".join(p.get('text','') if isinstance(p,dict) else str(p) for p in c)
            parts.append(f"[{m.get('role','?')}]\n{c}")
        doc="\n\n".join(parts)
        if not doc.strip(): continue
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
