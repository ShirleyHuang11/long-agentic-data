"""Readable re-render of the corrected (BPC@32K) figures.
Large fonts, large canvas, high DPI, de-cluttered labels.
"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12.5,
    "figure.dpi": 130, "savefig.dpi": 200, "axes.grid": True,
    "grid.alpha": 0.25, "font.family": "DejaVu Sans",
})
OUT="figures"; os.makedirs(OUT, exist_ok=True)
rows={r["slug"]:r for r in csv.DictReader(open("data/agentic_alpha_hinf.csv"))}

def B(slug): return float(rows[slug]["bpc_32768"])

# ---------------- fig: content ranking (the headline) ----------------
GEN=[("II-Agent GAIA (frontier search)","ii-agent-gaia-traj"),
     ("GDPval (human pro tasks)","gdpval-tasks"),
     ("Opus 4.8 CLI sessions","opus48-pi-traces"),
     ("Nemotron search","nemotron-sft-v2-search"),
     ("smolagents GAIA (gpt-4o)","smolagents-gaia-traces"),
     ("JetBrains GPT-5.2 (real issues)","jetbrains-swe-test-minus-verified"),
     ("Toucan Kimi-K2 (MCP)","toucan-15m-kimi-k2"),
     ("GLM-4.7 terminus","dcagent-glm47-terminus2"),
     ("APIGen-MT","apigen-mt-5k"),
     ("OpenThoughts-Agent SFT","openthoughts-agent-v1-sft"),
     ("AgentInstruct","agentinstruct-all"),
     ("aider-7B (failure loops)","aider-polyglot-sweagentlm7b")]
GEN=[(l,s) for l,s in GEN if s in rows]
vals=[B(s) for _,s in GEN]
labs=[l for l,_ in GEN]
order=sorted(range(len(vals)), key=lambda i: vals[i])
vals=[vals[i] for i in order]; labs=[labs[i] for i in order]
cols=["#2ca02c" if v>=2.0 else "#ff7f0e" if v>=1.3 else "#9e9e9e" for v in vals]
fig,ax=plt.subplots(figsize=(13,8))
y=range(len(vals))
ax.barh(list(y),vals,color=cols,height=0.7,edgecolor="black",linewidth=0.6)
for yi,v in zip(y,vals): ax.text(v+0.04,yi,f"{v:.2f}",va="center",fontsize=14,fontweight="bold")
ax.set_yticks(list(y)); ax.set_yticklabels(labs)
ax.set_xlim(0,3.4); ax.set_xlabel("content density  =  BPC@32K  (bits/char at 32 KB context, directly measured)")
ax.set_title("Agentic data content ranking (corrected metric)\nhigh = real content · low = templated / boilerplate",
             loc="left")
import matplotlib.patches as mp
ax.legend(handles=[mp.Patch(color="#2ca02c",label="content-rich (≥2.0)"),
                   mp.Patch(color="#ff7f0e",label="mixed (1.3–2.0)"),
                   mp.Patch(color="#9e9e9e",label="templated (<1.3)")],
          loc="lower right", framealpha=0.95)
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_content_ranking.png"); plt.close(fig)
print("wrote fig_content_ranking.png")

# ---------------- fig: signature scatter, decluttered ----------------
EX={"nemotron-agentic-v1","rlenv-appworld-train","taubench-sonnet-proxy",
    "aguvis-s2-androidctl-text","saital-browser-reasoning-action","saital-browser-action-only"}
import importlib.util
src=open("scripts/make_figures.py").read()
ns={}; exec(src[src.index("CAT = {"):src.index("data = [")], ns); CAT=ns["CAT"]
COL={"SWE traj (frontier)":"#1f77b4","tool/search traj":"#2ca02c","mid-size generator":"#d62728",
     "template SFT":"#9e9e9e","compact action view":"#9467bd","full-obs / annotated view":"#8c564b",
     "agent-text-only view":"#17becf","task corpus":"#ff7f0e"}
pts=[]
for s,r in rows.items():
    if s in EX or s not in CAT: continue
    try: pts.append((B(s),float(r["alpha"]),CAT[s],s))
    except: pass
fig,ax=plt.subplots(figsize=(13,8.5))
for cat,c in COL.items():
    g=[p for p in pts if p[2]==cat]
    if g: ax.scatter([p[0] for p in g],[p[1] for p in g],s=130,c=c,alpha=0.8,
                     edgecolors="white",linewidths=1.0,label=cat)
# label only a few anchors, well spaced
ANN={"ii-agent-gaia-traj":"II-Agent GAIA","gdpval-tasks":"GDPval","opus48-pi-traces":"Opus4.8 CLI",
     "weblinx-actions":"WebLINX","aider-polyglot-sweagentlm7b":"aider-7B flail",
     "agentgym-agenttraj-l":"AgentTraj","openthoughts-agent-v1-sft":"OpenThoughts","apigen-mt-5k":"APIGen"}
for x,a,cat,s in pts:
    if s in ANN: ax.annotate(ANN[s],(x,a),fontsize=12,fontweight="bold",
                             xytext=(7,6),textcoords="offset points")
ax.axvline(1.5,color="gray",lw=1.2,ls="--")
ax.text(1.54,ax.get_ylim()[1]*0.96,"templated  ◀ | ▶  content-rich",fontsize=12,color="gray",va="top")
ax.set_xlabel("content density  =  BPC@32K  (directly measured)")
ax.set_ylabel(r"$\alpha$  (context-scaling exponent = structure)")
ax.set_title("Signature map of 80+ agentic datasets (corrected content axis)", loc="left")
ax.legend(loc="upper right", framealpha=0.95, ncol=1)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_signature_map.png"); plt.close(fig)
print("wrote fig_signature_map.png")
