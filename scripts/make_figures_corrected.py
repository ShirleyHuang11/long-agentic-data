"""Corrected figures (2026-06-07): content axis = directly-measured BPC@32768,
replacing the deprecated extrapolated/clamped H_inf. BPC@32K is validated on
synthetic controls (random 2.47 / template 0.02 / mixed 1.40) and needs no
fitting. Lower = templated/compressible, higher = real content.
Axis convention kept: content on x, structure (alpha) on y.
"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT="figures"; os.makedirs(OUT, exist_ok=True)
rows={r["slug"]:r for r in csv.DictReader(open("data/agentic_alpha_hinf.csv"))}
EX={"nemotron-agentic-v1","rlenv-appworld-train","taubench-sonnet-proxy",
    "aguvis-s2-androidctl-text","saital-browser-reasoning-action","saital-browser-action-only"}

# reuse the category map from make_figures.py
import importlib.util
spec=importlib.util.spec_from_file_location("mf","scripts/make_figures.py")
# we only need CAT/COLORS; import safely by exec of those dicts
ns={}
src=open("scripts/make_figures.py").read()
start=src.index("CAT = {"); end=src.index("data = [")
exec(src[src.index("COLORS = {"):start] if False else src[start:end], ns)
CAT=ns["CAT"];
COLORS={"SWE traj (frontier)":"#1f77b4","tool/search traj":"#2ca02c","mid-size generator":"#d62728",
        "template SFT":"#7f7f7f","compact action view":"#9467bd","full-obs / annotated view":"#8c564b",
        "agent-text-only view":"#17becf","task corpus":"#ff7f0e"}

data=[]
for slug,r in rows.items():
    if slug in EX or slug not in CAT: continue
    try: bpc=float(r["bpc_32768"]); a=float(r["alpha"]); be=float(r["mean_doc_bytes"])
    except: continue
    data.append(dict(slug=slug,cat=CAT[slug],bpc=bpc,alpha=a,bytes_ep=be))

# ---- fig1c: signature map on BPC@32K ----
fig,ax=plt.subplots(figsize=(10,7))
for cat,col in COLORS.items():
    pts=[d for d in data if d["cat"]==cat]
    if not pts: continue
    ax.scatter([d["bpc"] for d in pts],[d["alpha"] for d in pts],
               s=[18+14*(d["bytes_ep"]**0.33) for d in pts],
               c=col,alpha=0.75,edgecolors="white",lw=0.5,label=cat)
ann={"weblinx-actions":"WebLINX act","gdpval-tasks":"GDPval","ii-agent-gaia-traj":"II-Agent GAIA",
     "opus48-pi-traces":"Opus4.8 CLI","jetbrains-swe-test-minus-verified":"JetBrains GPT-5.2",
     "openthoughts-agent-v1-sft":"OpenThoughts SFT","apigen-mt-5k":"APIGen",
     "aider-polyglot-sweagentlm7b":"aider-7B(flail)","agentgym-agenttraj-l":"AgentTraj",
     "nemotron-sft-v2-tool":"Nemotron-tool","deep-research-sft-0406":"deep-research"}
for d in data:
    if d["slug"] in ann:
        ax.annotate(ann[d["slug"]],(d["bpc"],d["alpha"]),fontsize=7,xytext=(4,4),textcoords="offset points")
ax.axvline(1.5,color="k",lw=0.6,ls="--",alpha=0.5)
ax.text(1.52,0.02,"content threshold ~1.5 BPC",fontsize=7,alpha=0.7,rotation=90)
ax.set_xlabel("BPC@32K (directly-measured content density)  — low=templated, high=real content")
ax.set_ylabel("alpha (context-scaling exponent) — structure")
ax.set_title("Signature map (CORRECTED): content axis = directly-measured BPC@32K\n"
             "replaces deprecated clamped/extrapolated H_inf")
ax.legend(fontsize=7.5,loc="upper right",framealpha=0.9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1c_signature_bpc32k.png",dpi=160)
print("wrote fig1c_signature_bpc32k.png")

# ---- fig4c: content ranking bar (replaces generator spectrum H_inf) ----
GEN=[("Opus 4.8 CLI","opus48-pi-traces"),("II-Agent GAIA","ii-agent-gaia-traj"),
     ("GDPval tasks","gdpval-tasks"),("JetBrains GPT-5.2","jetbrains-swe-test-minus-verified"),
     ("smolagents GAIA","smolagents-gaia-traces"),("Nemotron-search","nemotron-sft-v2-search"),
     ("Toucan Kimi-K2","toucan-15m-kimi-k2"),("OpenThoughts SFT","openthoughts-agent-v1-sft"),
     ("GLM-4.7 terminus","dcagent-glm47-terminus2"),("APIGen (template)","apigen-mt-5k"),
     ("AgentInstruct","agentinstruct-all"),("aider-7B (flail)","aider-polyglot-sweagentlm7b")]
fig,ax=plt.subplots(figsize=(8,6.5))
ys=range(len(GEN))[::-1]
for y,(lab,slug) in zip(ys,GEN):
    if slug not in rows: continue
    b=float(rows[slug]["bpc_32768"])
    col="#2ca02c" if b>=2.0 else ("#ff7f0e" if b>=1.3 else "#7f7f7f")
    ax.barh(y,b,color=col,height=0.62,alpha=0.85)
    ax.text(b+0.02,y,f"{b:.2f}",va="center",fontsize=8)
ax.set_yticks(list(ys)); ax.set_yticklabels([g[0] for g in GEN],fontsize=8)
ax.set_xlabel("BPC@32K (content density)")
ax.set_title("Content ranking (CORRECTED, directly measured):\n"
             "note OpenThoughts/GLM sit MID, not at zero — the clamp had hidden this")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4c_content_bpc32k.png",dpi=160)
print("wrote fig4c_content_bpc32k.png")
