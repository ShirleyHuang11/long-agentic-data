"""Master figure regeneration (2026-06-07) — all on the CORRECTED content metric
BPC@32768 (directly measured), readable styling. Supersedes the H_inf-based
figures. Naming: fig1..fig8 canonical set.
"""
import csv, os, statistics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15.5,
    "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 12,
    "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.25,
    "font.family": "DejaVu Sans"})
OUT="figures"; os.makedirs(OUT, exist_ok=True)
R={r["slug"]:r for r in csv.DictReader(open("data/agentic_alpha_hinf.csv"))}
def B(s): return float(R[s]["bpc_32768"])
def A(s): return float(R[s]["alpha"])
EX={"nemotron-agentic-v1","rlenv-appworld-train","taubench-sonnet-proxy",
    "aguvis-s2-androidctl-text","saital-browser-reasoning-action","saital-browser-action-only"}
# category map
ns={}; src=open("scripts/make_figures.py").read()
exec(src[src.index("CAT = {"):src.index("data = [")], ns); CAT=ns["CAT"]
COL={"SWE traj (frontier)":"#1f77b4","tool/search traj":"#2ca02c","mid-size generator":"#d62728",
     "template SFT":"#9e9e9e","compact action view":"#9467bd","full-obs / annotated view":"#8c564b",
     "agent-text-only view":"#17becf","task corpus":"#ff7f0e"}

# ---------- fig1: signature map ----------
pts=[(B(s),A(s),CAT[s],s) for s,r in R.items() if s not in EX and s in CAT]
fig,ax=plt.subplots(figsize=(13,8.5))
for cat,c in COL.items():
    g=[p for p in pts if p[2]==cat]
    if g: ax.scatter([p[0] for p in g],[p[1] for p in g],s=130,c=c,alpha=.8,edgecolors="white",lw=1,label=cat)
ANN={"ii-agent-gaia-traj":"II-Agent GAIA","gdpval-tasks":"GDPval","weblinx-actions":"WebLINX",
     "aider-polyglot-sweagentlm7b":"aider-7B flail","agentgym-agenttraj-l":"AgentTraj",
     "openthoughts-agent-v1-sft":"OpenThoughts","apigen-mt-5k":"APIGen"}
for x,a,cat,s in pts:
    if s in ANN: ax.annotate(ANN[s],(x,a),fontsize=12,fontweight="bold",xytext=(7,6),textcoords="offset points")
ax.axvline(1.5,color="gray",lw=1.2,ls="--"); ax.text(1.54,ax.get_ylim()[1]*.96,"templated ◀ | ▶ content-rich",fontsize=12,color="gray",va="top")
ax.set_xlabel("content density = BPC@32K (directly measured)"); ax.set_ylabel(r"$\alpha$ (context-scaling = structure)")
ax.set_title("Fig 1 · Signature map of agentic datasets (corrected metric)",loc="left")
ax.legend(loc="upper right",framealpha=.95); fig.tight_layout(); fig.savefig(f"{OUT}/fig1_signature_map.png"); plt.close()

# ---------- fig2: content ranking ----------
GEN=[("II-Agent GAIA","ii-agent-gaia-traj"),("GDPval (human tasks)","gdpval-tasks"),
     ("Nemotron search","nemotron-sft-v2-search"),("smolagents GAIA","smolagents-gaia-traces"),
     ("JetBrains GPT-5.2","jetbrains-swe-test-minus-verified"),("SWE-ZERO-OH","swe-zero-oh-traj"),
     ("Toucan Kimi-K2","toucan-15m-kimi-k2"),("GLM-4.7 terminus","dcagent-glm47-terminus2"),
     ("APIGen-MT","apigen-mt-5k"),("OpenThoughts SFT","openthoughts-agent-v1-sft"),
     ("AgentInstruct","agentinstruct-all"),("aider-7B (flail)","aider-polyglot-sweagentlm7b")]
GEN=[(l,s) for l,s in GEN if s in R]; GEN.sort(key=lambda t:B(t[1]))
fig,ax=plt.subplots(figsize=(13,8)); y=range(len(GEN))
for yi,(l,s) in zip(y,GEN):
    v=B(s); c="#2ca02c" if v>=2 else "#ff7f0e" if v>=1.3 else "#9e9e9e"
    ax.barh(yi,v,color=c,height=.7,edgecolor="black",lw=.6); ax.text(v+.04,yi,f"{v:.2f}",va="center",fontweight="bold",fontsize=13)
ax.set_yticks(list(y)); ax.set_yticklabels([l for l,_ in GEN]); ax.set_xlim(0,3.4)
ax.set_xlabel("content density = BPC@32K"); ax.set_title("Fig 2 · Content ranking (corrected)\nhigh=real content · low=templated",loc="left")
ax.legend(handles=[mp.Patch(color=c,label=l) for c,l in [("#2ca02c","content-rich ≥2.0"),("#ff7f0e","mixed 1.3–2.0"),("#9e9e9e","templated <1.3")]],loc="lower right")
ax.grid(axis="y",visible=False); fig.tight_layout(); fig.savefig(f"{OUT}/fig2_content_ranking.png"); plt.close()

# ---------- fig3: view decomposition (content on BPC@32K) ----------
PAIRS=[("mind2web-actions","mind2web-fullobs","Mind2Web","web"),
       ("weblinx-actions","weblinx-fullobs","WebLINX","web"),
       ("agentnet-actions","agentnet-text","AgentNet","desktop GUI"),
       ("jetbrains-swe-assistant-only","jetbrains-swe-test-minus-verified","JetBrains","SWE"),
       ("swe-rebench-oh-assistant-only","swe-rebench-oh-traj","SWE-rebench-OH","SWE")]
fig,ax=plt.subplots(figsize=(11,6))
for i,(a,b,lab,dom) in enumerate(PAIRS):
    if a not in R or b not in R: continue
    ba,bb=B(a),B(b); up=ba>bb; col="#2ca02c" if up else "#d62728"
    ax.annotate("",xy=(i,ba),xytext=(i,bb),arrowprops=dict(arrowstyle="-|>",color=col,lw=2.5))
    ax.scatter([i],[bb],c="#555",s=70,zorder=3); ax.scatter([i],[ba],c=col,s=70,zorder=3)
    ax.text(i,max(ba,bb)+.08,f"{lab}\n({dom})",ha="center",fontsize=11)
ax.set_xticks([]); ax.set_ylabel("content density = BPC@32K"); ax.set_ylim(0,2.6)
ax.set_title("View decomposition (corrected): strip observations →\n"
             "green=content recovered (web/GUI) · red=content lost (SWE obs ARE content)",loc="left")
ax.text(.02,.02,"grey=full view → colored=stripped view",transform=ax.transAxes,fontsize=11,alpha=.8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_view_decomposition.png"); plt.close()

# ---------- fig4: horizon vs content ----------
fig,ax=plt.subplots(figsize=(12,7.5))
traj=[(float(R[s]["mean_doc_bytes"]),B(s),CAT[s],s) for s in R if s not in EX and s in CAT
      and float(R[s]["mean_turns"])>1]
for cat,c in COL.items():
    g=[p for p in traj if p[2]==cat]
    if g: ax.scatter([p[0] for p in g],[p[1] for p in g],s=110,c=c,alpha=.8,edgecolors="white",lw=.8,label=cat)
for x,b,cat,s in traj:
    if s in ("aider-polyglot-sweagentlm7b","openhands-feedback","ii-agent-gaia-traj","opus48-pi-traces"):
        ax.annotate(s.split("-")[0],(x,b),fontsize=10,xytext=(5,4),textcoords="offset points")
ax.set_xscale("log"); ax.axhline(1.5,color="gray",lw=1,ls="--")
ax.set_xlabel("bytes per episode (log) — horizon"); ax.set_ylabel("content density = BPC@32K")
ax.set_title("Fig 4 · Horizon vs content: long episodes are NOT high-content\n(failure-loop datasets at high bytes/ep, low BPC)",loc="left")
ax.legend(fontsize=10,framealpha=.95); fig.tight_layout(); fig.savefig(f"{OUT}/fig4_horizon_vs_content.png"); plt.close()

# ---------- fig5: Hurst vs content ----------
hu={r["slug"]:float(r["hurst"]) for r in csv.DictReader(open("data/hurst.csv"))}
P={"toucan-15m-kimi-k2":("Toucan","#2ca02c"),"jetbrains-swe-test-minus-verified":("JetBrains","#1f77b4"),
   "swe-zero-12m-traj":("SWE-ZERO","#1f77b4"),"glaive-fc-v2":("glaive-FC","#2ca02c"),
   "weblinx-actions":("WebLINX (human)","#9467bd"),"apigen-mt-5k":("APIGen","#9e9e9e"),
   "ko-agent-traj-train":("Ko-Agent","#9e9e9e"),"aider-polyglot-r2egym32b":("aider-32B flail","#d62728"),
   "agentnet-text":("AgentNet annot.","#8c564b"),
   # iter-176: +8 corpora spanning H∞ (grew Hurst n 9->17)
   "agent-flan-all":("Agent-FLAN","#9e9e9e"),"agentgym-agenttraj-l":("AgentGym","#9e9e9e"),
   "aider-polyglot-qwen3coder30b":("aider-qwen3","#d62728"),"rebel-alfworld-actions":("ALFWorld","#9e9e9e"),
   "swe-hero-oh-traj":("SWE-Hero","#1f77b4"),"miroverse-agentic-sft-new":("MiroVerse","#2ca02c"),
   "gui-odyssey-actions":("GUI-Odyssey (human)","#9467bd"),"fireact-multitask":("FireAct","#2ca02c")}
fig,ax=plt.subplots(figsize=(11,7))
for s,(l,c) in P.items():
    if s in R and s in hu:
        ax.scatter([B(s)],[hu[s]],c=c,s=160,edgecolors="k",zorder=3)
        ax.annotate(l,(B(s),hu[s]),fontsize=12,xytext=(7,5),textcoords="offset points")
ax.axvline(1.5,color="gray",lw=1.2,ls="--")
ax.set_xlabel("content density = BPC@32K"); ax.set_ylabel("Hurst exponent H (long-range organization)")
ax.set_title("Hurst vs content: templates & healthy data span similar Hurst\n→ Hurst alone can't rate data (repetition IS long-range dependence)",loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_hurst_vs_content.png"); plt.close()

# ---------- fig6: gamma-beta plane colored by content ----------
gb={r["slug"]:float(r["beta"]) for r in csv.DictReader(open("data/gamma_beta.csv"))}
import numpy as np
fig,ax=plt.subplots(figsize=(11.5,8))
bb=np.linspace(.05,1.6,200); gg=np.linspace(0,.6,200); BB,GG=np.meshgrid(bb,gg)
cs=ax.contour(BB,GG,GG/(2*BB),levels=[.05,.1,.14,.19,.3,.5,.8],colors="gray",linewidths=.8,alpha=.6)
ax.clabel(cs,fmt=lambda v:f"$\\alpha_D$={v:g}",fontsize=9)
L={"toucan-15m-kimi-k2":"Toucan","jetbrains-swe-test-minus-verified":"JetBrains","apigen-mt-5k":"APIGen",
   "swe-zero-12m-traj":"SWE-ZERO","weblinx-actions":"WebLINX","agentnet-text":"AgentNet annot.","glaive-fc-v2":"glaive-FC"}
xs=[gb[s] for s in L if s in gb and s in R]; ys=[A(s) for s in L if s in gb and s in R]; cc=[B(s) for s in L if s in gb and s in R]
sc=ax.scatter(xs,ys,c=cc,cmap="viridis",s=200,vmin=0,vmax=3,marker="D",edgecolors="k",lw=1.2)
for s in L:
    if s in gb and s in R: ax.annotate(L[s],(gb[s],A(s)),fontsize=12,xytext=(7,5),textcoords="offset points")
fig.colorbar(sc,ax=ax,shrink=.8,label="content density = BPC@32K")
ax.axvspan(.05,.55,alpha=.05,color="tab:blue"); ax.text(.3,.55,"agentic phase",fontsize=12,ha="center",color="tab:blue")
ax.text(1.2,.55,"natural-language phase",fontsize=12,ha="center",color="tab:green")
ax.set_xlabel(r"$\beta$ (byte-correlation decay)"); ax.set_ylabel(r"$\gamma$ (LZ-oracle $\alpha$)")
ax.set_title("γ–β phase plane (color = corrected content density)",loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_gamma_beta.png"); plt.close()

print("regenerated fig1..fig6 on BPC@32K")
