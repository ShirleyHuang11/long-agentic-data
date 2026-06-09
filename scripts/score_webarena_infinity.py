"""Score webarena-x/webarena-infinity-trajectories — real web-agent rollouts.

Three generators (gemini, kimi, qwen) x 13 sites. One document = one task's
history.json (text only: model reasoning + actions + extracted content + page
url/title; screenshots excluded so images don't enter the byte stream). One
registry row per generator -> frontier-vs-mid web contrast on identical tasks.
"""
import csv, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(__file__))
import lz_oracle
from huggingface_hub import HfApi, hf_hub_download

R = "webarena-x/webarena-infinity-trajectories"
SOURCE = {"kimi": "frontier", "qwen": "mid"}  # gemini already appended (browser-use schema)
CAP = 130  # tasks per generator (oracle caps at 8MB/1500 docs anyway)

api = HfApi()
allh = [s.rfilename for s in api.dataset_info(R).siblings
        if s.rfilename.endswith("history.json")]


def ser(hist):
    out = []
    for step in hist.get("history", []):
        # computer-use schema (kimi/qwen): thought + actions
        if step.get("thought"):
            out.append(step["thought"])
        if step.get("actions"):
            out.append("actions: " + json.dumps(step["actions"], ensure_ascii=False))
        # browser-use schema (gemini): model_output + result + state
        mo = step.get("model_output") or {}
        for k in ("evaluation_previous_goal", "next_goal", "memory"):
            if mo.get(k):
                out.append(f"{k}: {mo[k]}")
        if mo.get("action"):
            out.append("action: " + json.dumps(mo["action"], ensure_ascii=False))
        for r in step.get("result") or []:
            for k in ("extracted_content", "long_term_memory"):
                if r.get(k):
                    out.append(f"{k}: {r[k]}")
        st = step.get("state") or {}
        if st.get("url") or st.get("title"):
            out.append(f"page: {st.get('title','')} {st.get('url','')}")
    return "\n".join(out)


fields = ["dataset", "config", "splits", "slug", "alpha", "h_inf", "h_inf_raw",
          "bpc_128", "bpc_2048", "bpc_32768", "n_episodes", "mean_turns",
          "mean_doc_bytes", "n_bytes", "h_inf_v3", "h_inf_stderr", "resolved"]

for model, src in SOURCE.items():
    mfiles = [f for f in allh if f.split("/")[1] == model][:CAP]
    docs, turns = [], []
    for f in mfiles:
        p = hf_hub_download(R, f, repo_type="dataset")
        try:
            h = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        d = ser(h)
        if d:
            docs.append(d)
            turns.append(len(h.get("history", [])))
    if not docs:
        print(f"{model}: no docs"); continue
    res = lz_oracle.score(docs)
    slug = f"webarena-infinity-{model}"
    row = {"dataset": R, "config": f"data/{model}/** history.json (13 sites, text only)",
           "splits": "raw", "slug": slug,
           "alpha": res["alpha"], "h_inf": res["h_inf"], "h_inf_raw": res["h_inf_raw"],
           "bpc_128": res["bpc_128"], "bpc_2048": res["bpc_2048"], "bpc_32768": res["bpc_32768"],
           "n_episodes": len(docs), "mean_turns": round(statistics.mean(turns), 1),
           "mean_doc_bytes": round(statistics.mean(len(d.encode()) for d in docs)),
           "n_bytes": res.get("n_bytes", ""), "h_inf_v3": "", "h_inf_stderr": "", "resolved": "True"}
    with open("data/agentic_alpha_hinf.csv", "a", newline="") as fp:
        csv.DictWriter(fp, fieldnames=fields).writerow({k: row[k] for k in fields})
    print(f"{slug} ({src}): H_inf={res['h_inf']:.3f} alpha={res['alpha']:.3f} "
          f"bpc32k={res['bpc_32768']:.3f} n_docs={len(docs)} mean_turns={row['mean_turns']}")
    prov = {"hf_dataset": R, "hf_url": f"https://huggingface.co/datasets/{R}",
            "config": row["config"], "generator": model, "splits": ["raw"],
            "serializer": "score_webarena_infinity.py (model_output+result+state, screenshots excluded)",
            "sampling": {"max_docs": lz_oracle.MAX_DOCS, "max_bytes": lz_oracle.MAX_BYTES,
                         "cap_per_model": CAP},
            "oracle": {"script": "scripts/lz_oracle.py",
                       "compressor": f"zstd level {lz_oracle.ZSTD_LEVEL}",
                       "context_points": list(lz_oracle.N_POINTS)},
            "result": {k: res[k] for k in ("alpha", "h_inf", "bpc_128", "bpc_2048", "bpc_32768")}}
    json.dump(prov, open(f"data/provenance/{slug}.json", "w"), indent=2)
print("done")
