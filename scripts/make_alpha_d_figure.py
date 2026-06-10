"""Figure + result for the α_D = γ/2β training validation (paper §9 #1).

(a) measured loss-vs-D curves (log-log) for the 4 corpora, single-pass from-scratch
    29M GPT (data-limited regime).
(b) predicted α/(2β) vs measured data-scaling exponent — rank-correlated (Spearman),
    a directional confirmation that low-β agentic data is more sample-efficient.
Writes figures/fig_alpha_d.png and results/alpha_d_validation.md.
"""
import json, csv, statistics as st
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":13,"axes.titlesize":14,"axes.labelsize":13,
    "savefig.dpi":200,"axes.grid":True,"grid.alpha":0.25,"font.family":"DejaVu Sans"})

reg={r['slug']:r for r in csv.DictReader(open("data/agentic_alpha_hinf.csv"))}
gb={(r.get('slug') or r.get('dataset')):float(r[[k for k in r if 'beta' in k.lower()][0]])
    for r in csv.DictReader(open("data/gamma_beta.csv"))}
M={"coderforge":"coderforge-32b-swebench-verified-eval","swezero":"swe-zero-12m-traj",
   "jetbrains":"jetbrains-swe-test-minus-verified","agentnet":"agentnet-text",
   "weblinx":"weblinx-actions","glaive":"glaive-fc-v2","apigen":"apigen-mt-5k",
   "smolagents":"smolagents-gaia-traces","taubench":"taubench-deepseek-r1-eval"}
LAB={"coderforge":"CoderForge","swezero":"SWE-ZERO","jetbrains":"JetBrains","agentnet":"AgentNet",
     "weblinx":"WebLINX","glaive":"Glaive-FC","apigen":"APIGen","smolagents":"smolagents-GAIA","taubench":"tau-bench"}
import matplotlib.cm as _cm
COL={k:_cm.tab10(i % 10) for i,k in enumerate(M)}

import os
rows=[]
for c,slug in M.items():
    if not os.path.exists(f"data/alpha_d_{c}.json"): continue
    d=json.load(open(f"data/alpha_d_{c}.json"))
    D=np.array([p['D'] for p in d['curve']]); L=np.array([p['val_loss'] for p in d['curve']])
    a=float(reg[slug]['alpha']); b=gb[slug]; pred=a/(2*b)
    best=(1e9,None)
    for Linf in np.linspace(0, min(L)*0.98, 60):
        y=L-Linf
        if (y<=0).any(): continue
        co=np.polyfit(np.log(D),np.log(y),1); res=np.sum((np.log(y)-np.polyval(co,np.log(D)))**2)
        if res<best[0]: best=(res,(-co[0],Linf))
    meas=best[1][0]
    rows.append({"corpus":c,"beta":b,"alpha":a,"pred_aD":pred,"meas_aD":meas,"D":D.tolist(),"L":L.tolist()})

def spear(xs,ys):
    n=len(xs)
    def rk(v):
        o=sorted(range(n),key=lambda i:v[i]); r=[0]*n
        for k,i in enumerate(o): r[i]=k+1
        return r
    rx,ry=rk(xs),rk(ys); mx,my=st.fmean(rx),st.fmean(ry)
    return sum((p-mx)*(q-my) for p,q in zip(rx,ry))/((sum((p-mx)**2 for p in rx)*sum((q-my)**2 for q in ry))**.5)
rho=spear([r['pred_aD'] for r in rows],[r['meas_aD'] for r in rows])

fig,(axA,axB)=plt.subplots(1,2,figsize=(13,5.2))
for r in rows:
    axA.plot(r['D'],r['L'],"o-",color=COL[r['corpus']],label=f"{LAB[r['corpus']]} (β={r['beta']:.2f})")
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlabel("training tokens D"); axA.set_ylabel("held-out val loss")
axA.set_title("(a) Data-limited loss curves\n(single-pass from-scratch 29M GPT)",loc="left")
axA.legend(fontsize=10.5)
for r in rows:
    axB.scatter(r['pred_aD'],r['meas_aD'],s=120,color=COL[r['corpus']],edgecolors="white",lw=1,zorder=3)
    axB.annotate(LAB[r['corpus']],(r['pred_aD'],r['meas_aD']),fontsize=10,fontweight="bold",
                 xytext=(7,3),textcoords="offset points")
axB.set_xlabel("predicted α_D = α/(2β)"); axB.set_ylabel("measured data-scaling exponent")
axB.set_title(f"(b) Predicted vs measured (n={len(rows)})\nSpearman = {rho:.2f} (positive but modest)",loc="left")
fig.suptitle("α_D = γ/2β training validation: modest positive support (direction, not precise prediction)",
             fontsize=14,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("figures/fig_alpha_d.png")
print(f"wrote figures/fig_alpha_d.png  Spearman={rho:.2f}")

with open("results/alpha_d_validation.md","w") as f:
    f.write("# α_D = γ/2β training validation (paper §9 #1) — RESULT\n\n")
    f.write("Single-pass from-scratch 29M GPT on each corpus's GPT-2 token stream (data-limited "
            "regime); val loss logged at D=0.25–3M tokens. Job 20718622 (gpu_requeue). "
            "Data: `data/alpha_d_*.json`; scripts: `alpha_d_dataprep.py`, `alpha_d_train.py`.\n\n")
    f.write("| corpus | β | α(γ) | predicted α_D=α/2β | measured exponent |\n| :-- | --: | --: | --: | --: |\n")
    for r in sorted(rows,key=lambda x:-x['pred_aD']):
        f.write(f"| {LAB[r['corpus']]} | {r['beta']:.2f} | {r['alpha']:.2f} | {r['pred_aD']:.2f} | {r['meas_aD']:.2f} |\n")
    f.write(f"\n**Spearman(predicted α_D, measured exponent) = {rho:.2f} (n={len(rows)}).** Positive but "
            "modest: the prediction's *direction* holds — low-β agentic data tends to be more sample-efficient "
            "(WebLINX, Glaive, SWE-ZERO decay faster than high-β AgentNet) — but the quantitative α_D=γ/2β "
            "prediction is far from precise at this scale, with notable discordant points (CoderForge predicted "
            "0.93 but measured 0.22; tau-bench predicted 0.25 but measured 0.39). An earlier n=4 subset gave a "
            "stronger Spearman 0.80; the larger sample reveals that was optimistic. Absolute exponents are also "
            "compressed (measured 0.15–0.61 vs predicted 0.05–0.93): a single small model over a narrow D range "
            "(0.25–3M) recovers a weak ordering, not the theoretical scale. **Honest read: preliminary, modest "
            "positive support for the pattern statistics as a predictive signal — not a confirmation.** A fuller "
            "test (larger models, wider D, β/α measured on the exact training serialization) is forward work.\n")
print("wrote results/alpha_d_validation.md")
