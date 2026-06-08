"""Debug plot: BPC vs context length for representative datasets.
Shows WHY H_inf resolves (curve flattens -> asymptote visible) or not (still
descending at 524KB -> floor unresolved). Marks 32KB (canonical) and 128KB.
"""
import sys; sys.path.insert(0,"scripts")
import lz_oracle as lz, score_agentic_datasets as sad
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":13,"axes.titlesize":16,"axes.labelsize":14,
    "legend.fontsize":10.5,"savefig.dpi":200,"font.family":"DejaVu Sans"})
NS=[128,512,2048,8192,32768,131072,524288]
PICK=[("ii-agent-gaia-traj","II-Agent GAIA (content-rich)","#1f77b4"),
      ("weblinx-actions","WebLINX actions (human, resolves)","#9467bd"),
      ("swe-zero-oh-traj","SWE-ZERO-OH (healthy traj)","#2ca02c"),
      ("openthoughts-agent-v1-sft","OpenThoughts (template, unresolved)","#ff7f0e"),
      ("apigen-mt-5k","APIGen (template)","#d62728"),
      ("agentinstruct-all","AgentInstruct (template)","#9e9e9e"),
      ("aider-polyglot-sweagentlm7b","aider-7B (failure loops)","#8c564b")]
REG={e[4]:e for e in sad.REGISTRY}
def curve(slug):
    e=REG[slug]; path,cfg,sp,ser=e[:4]; gk=e[5] if len(e)>5 else None
    dd,sz=[],0
    for d,_ in sad.iter_docs(path,cfg,sp,ser,group_key=gk):
        dd.append(d); sz+=len(d)
        if sz>=lz.MAX_BYTES or len(dd)>=lz.MAX_DOCS: break
    c=lz.build_corpus(dd)
    return [(n,lz._bpc_at(c,n)) for n in NS if len(c)//n>=8]
fig,ax=plt.subplots(figsize=(11,7.5))
for slug,lab,col in PICK:
    if slug not in REG: continue
    pts=curve(slug); xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    ax.plot(xs,ys,"o-",color=col,label=lab,lw=2,ms=6)
    print(f"{slug}: "+" ".join(f"{n//1024 if n>=1024 else n}{'K' if n>=1024 else ''}:{b:.2f}" for n,b in pts),flush=True)
ax.set_xscale("log")
ax.axvline(32768,color="k",ls="--",lw=1,alpha=.6); ax.text(34000,ax.get_ylim()[1]*.95,"32KB\n(canonical metric)",fontsize=9)
ax.axvline(131072,color="gray",ls=":",lw=1,alpha=.6); ax.text(140000,ax.get_ylim()[1]*.80,"128KB\n(~32K tok)",fontsize=9,color="gray")
ax.set_xlabel("context length (bytes, log scale)"); ax.set_ylabel("BPC (bits per char) — lower = more compressible")
ax.set_title("Debug: BPC vs context length\nflattening curve → H∞ resolvable · still-falling → unresolved (floor not in window)",loc="left")
ax.legend(loc="upper right"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figures/debug_bpc_curves.png",bbox_inches="tight")
print("wrote figures/debug_bpc_curves.png")
