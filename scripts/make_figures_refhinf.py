"""Canonical figures on the REFERENCE-EXACT H_inf (data_format.md, clamped).
Content axis = h_inf column. ~40% of agentic datasets sit at H_inf=0 = the
reference 'template-degenerate' cluster (honest, reference-consistent).
BPC@32K versions (fig1/2 from make_all_figures.py) are kept as supplementary
finer-resolution views within that cluster.
"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.size":14,"axes.titlesize":16,"axes.labelsize":15,
    "xtick.labelsize":12.5,"ytick.labelsize":12.5,"legend.fontsize":11.5,
    "savefig.dpi":200,"axes.grid":True,"grid.alpha":0.25,"font.family":"DejaVu Sans"})
OUT="figures"
R={r["slug"]:r for r in csv.DictReader(open("data/agentic_alpha_hinf.csv"))}
EX={"nemotron-agentic-v1","rlenv-appworld-train","taubench-sonnet-proxy",
    "aguvis-s2-androidctl-text","saital-browser-reasoning-action","saital-browser-action-only"}
ns={}; src=open("scripts/make_figures.py").read()
exec(src[src.index("CAT = {"):src.index("data = [")], ns); CAT=ns["CAT"]
COL={"SWE traj (frontier)":"#1f77b4","tool/search traj":"#2ca02c","mid-size generator":"#d62728",
     "template SFT":"#9e9e9e","compact action view":"#9467bd","full-obs / annotated view":"#8c564b",
     "agent-text-only view":"#17becf","task corpus":"#ff7f0e"}
rng=np.random.default_rng(0)
pts=[]
for s,r in R.items():
    if s in EX or s not in CAT: continue
    try: h=float(r["h_inf"]); a=float(r["alpha"])
    except: continue
    pts.append((h,a,CAT[s],s))

# fig1ref: signature map H_inf(x) x alpha(y), jitter the H_inf=0 pile
fig,ax=plt.subplots(figsize=(13,8.5))
for cat,c in COL.items():
    g=[p for p in pts if p[2]==cat]
    if not g: continue
    xs=[p[0]+(rng.uniform(-0.015,0.015) if p[0]==0 else 0) for p in g]
    ax.scatter(xs,[p[1] for p in g],s=130,c=c,alpha=.8,edgecolors="white",lw=1,label=cat)
ANN={"ii-agent-gaia-traj":"II-Agent GAIA","gdpval-tasks":"GDPval","weblinx-actions":"WebLINX",
     "swe-bench-verified":"SWE-bench-V","openthoughts-agent-v1-sft":"OpenThoughts(=0)",
     "apigen-mt-5k":"APIGen(=0)","fireact-multitask":"FireAct"}
for h,a,cat,s in pts:
    if s in ANN: ax.annotate(ANN[s],(h,a),fontsize=11,fontweight="bold",xytext=(7,5),textcoords="offset points")
ax.axvspan(-0.05,0.05,color="red",alpha=.05)
ax.text(0.0,0.52,"H∞≈0\ntemplate-\ndegenerate\ncluster",fontsize=10,color="#b22",ha="center")
ax.set_xlabel("H∞  (reference-exact clamped, BPC) — content floor"); ax.set_ylabel(r"$\alpha$ (structure)")
ax.set_title("Fig 1 (reference metric) · Signature map: H∞ × α\n~40% of agentic data at H∞=0 = reference 'template-degenerate' signal",loc="left")
ax.legend(loc="upper right",framealpha=.95); fig.tight_layout(); fig.savefig(f"{OUT}/fig1_signature_refhinf.png"); plt.close()

# fig2ref: content ranking by reference H_inf
GEN=[("II-Agent GAIA","ii-agent-gaia-traj"),("WebLINX actions","weblinx-actions"),
     ("FireAct","fireact-multitask"),("SWE-bench-V","swe-bench-verified"),
     ("GDPval","gdpval-tasks"),("Nemotron-search","nemotron-sft-v2-search"),
     ("Toucan Kimi-K2","toucan-15m-kimi-k2"),("SWE-ZERO-OH","swe-zero-oh-traj"),
     ("GLM-4.7 terminus","dcagent-glm47-terminus2"),("OpenThoughts SFT","openthoughts-agent-v1-sft"),
     ("APIGen","apigen-mt-5k"),("AgentInstruct","agentinstruct-all")]
GEN=[(l,s) for l,s in GEN if s in R]; GEN.sort(key=lambda t:float(R[t[1]]["h_inf"]))
fig,ax=plt.subplots(figsize=(13,8)); y=range(len(GEN))
for yi,(l,s) in zip(y,GEN):
    v=float(R[s]["h_inf"]); c="#2ca02c" if v>=1.0 else "#ff7f0e" if v>0 else "#9e9e9e"
    ax.barh(yi,max(v,0.0),color=c,height=.7,edgecolor="black",lw=.6)
    ax.text(max(v,0)+.02,yi,f"{v:.2f}",va="center",fontweight="bold",fontsize=13)
ax.set_yticks(list(y)); ax.set_yticklabels([l for l,_ in GEN]); ax.set_xlim(0,2.6)
ax.set_xlabel("H∞ (reference-exact, BPC) — content floor")
ax.set_title("Fig 2 (reference metric) · Content ranking by H∞\nOpenThoughts/APIGen at 0 = template-degenerate (reference signal)",loc="left")
ax.legend(handles=[plt.Rectangle((0,0),1,1,color=c) for c in ["#2ca02c","#ff7f0e","#9e9e9e"]],
          labels=["content-rich (≥1.0)","mixed (0–1.0)","H∞=0 template-degenerate"],loc="lower right")
ax.grid(axis="y",visible=False); fig.tight_layout(); fig.savefig(f"{OUT}/fig2_ranking_refhinf.png"); plt.close()
print("wrote fig1_signature_refhinf.png, fig2_ranking_refhinf.png")
