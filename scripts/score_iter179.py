"""iter-179: deep-research agent SFT (openresearcher). Loaded by reading the parquet
DIRECTLY via pyarrow — load_dataset fails on this repo's multi-seed split metadata
(`SplitInfo got unexpected data_files`), so we bypass it. A very-long-horizon deep-research
corpus (web research + reasoning, mean ~119 turns/ep, individual episodes to 331).
Reference-exact score; full 17-col row."""
import csv, datetime, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi
REPO="AmanPriyanshu/tool-reasoning-sft-RESEARCH-openresearcher-dataset-sft-deep-research-agent-data-cleaned"
SLUG="openresearcher-deepresearch-sft"
FIELDS=["dataset","config","splits","slug","alpha","h_inf","h_inf_raw","bpc_128","bpc_2048",
        "bpc_32768","n_episodes","mean_turns","mean_doc_bytes","n_bytes","h_inf_v3","h_inf_stderr","resolved"]
p=hf_hub_download(REPO,"seed_42.parquet",repo_type="dataset")
t=pq.read_table(p,columns=["messages"])
docs,turns,size=[],[],0
for batch in t.to_batches(max_chunksize=200):
    for row in batch.to_pylist():
        msgs=row["messages"]
        if isinstance(msgs,str): msgs=json.loads(msgs)
        parts=[]
        for m in msgs:
            c=m.get('content') or ''
            if isinstance(c,list): c="".join(x.get('text','') if isinstance(x,dict) else str(x) for x in c)
            parts.append(f"[{m.get('role','?')}]\n{c}")
        doc="\n\n".join(parts)
        if not doc.strip(): continue
        docs.append(doc); turns.append(len(msgs)); size+=len(doc)
    if len(docs)>=lz_oracle.MAX_DOCS or size>=lz_oracle.MAX_BYTES: break
s=lz_oracle.score(docs)
try: sha=HfApi().dataset_info(REPO).sha
except Exception: sha="unresolved"
rec={"dataset":REPO,"config":"","splits":"seed_42","slug":SLUG,"alpha":s["alpha"],"h_inf":s["h_inf"],
     "h_inf_raw":s["h_inf_raw"],"bpc_128":s["bpc_128"],"bpc_2048":s["bpc_2048"],"bpc_32768":s["bpc_32768"],
     "n_episodes":len(docs),"mean_turns":round(statistics.mean(turns),1) if turns else 0,
     "mean_doc_bytes":round(statistics.mean(len(d) for d in docs)) if docs else 0,
     "n_bytes":s["n_bytes"],"h_inf_v3":"","h_inf_stderr":"","resolved":""}
print(f"alpha={s['alpha']:.3f} H_inf={s['h_inf']:.3f} BPC@32K={s['bpc_32768']:.3f} ep={len(docs)} turns={rec['mean_turns']}",flush=True)
os.makedirs("data/provenance",exist_ok=True)
json.dump({"hf_dataset":REPO,"hf_revision_sha":sha,"serializer":"messages_rc_parquet_direct",
           "oracle":{"compressor":f"zstd {lz_oracle.ZSTD_LEVEL}","context_points":list(lz_oracle.N_POINTS)},
           "collected_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "result":{k:rec[k] for k in ("alpha","h_inf","bpc_32768","n_episodes","mean_turns","n_bytes")}},
          open(f"data/provenance/{SLUG}.json","w"),indent=2)
with open("data/agentic_alpha_hinf.csv","a",newline="") as f:
    csv.DictWriter(f,fieldnames=FIELDS).writerow(rec)
print(f"appended {SLUG}")
