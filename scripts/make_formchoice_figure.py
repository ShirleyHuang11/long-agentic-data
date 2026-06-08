import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.size":14,"axes.titlesize":16,"axes.labelsize":14,
    "xtick.labelsize":12.5,"ytick.labelsize":12.5,"legend.fontsize":12,"savefig.dpi":200,
    "axes.grid":True,"grid.alpha":0.25,"font.family":"DejaVu Sans"})
r=json.load(open("data/formchoice_eval.json"))
models=["base","template","healthy"]
labels=["base\n(no SFT)","template\n(OpenThoughts)","healthy\n(GLM/JetBrains/SWE-ZERO)"]
form=[r[m]["form"] for m in models]; dec=[r[m]["decision"] for m in models]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,6))
cols=["#9e9e9e","#ff7f0e","#2ca02c"]
ax1.bar(range(3),form,color=cols,edgecolor="black",lw=.6)
for i,v in enumerate(form): ax1.text(i,v+.01,f"{v:.2f}",ha="center",fontweight="bold")
ax1.set_xticks(range(3)); ax1.set_xticklabels(labels); ax1.set_ylim(0,1.08)
ax1.set_ylabel("FORM score (valid terminus-2 JSON action)")
ax1.set_title("FORM: CONFIRMED ✓\nSFT lifts skeleton to ceiling",loc="left")
ax1.axhline(form[0],color="gray",ls="--",lw=1,alpha=.7); ax1.text(2.4,form[0]+.01,"base",fontsize=10,color="gray")
ax2.bar(range(3),dec,color=cols,edgecolor="black",lw=.6)
for i,v in enumerate(dec): ax2.text(i,v+.002,f"{v:.3f}",ha="center",fontweight="bold")
ax2.set_xticks(range(3)); ax2.set_xticklabels(labels); ax2.set_ylim(0,0.10)
ax2.set_ylabel("DECISION score (correct tool/action)")
ax2.set_title("DECISION: INCONCLUSIVE\nall floored incl base — no dynamic range",loc="left")
ax2.axhline(dec[0],color="gray",ls="--",lw=1,alpha=.7)
fig.suptitle("Form-vs-Choices (finding 20 / 21): Qwen2.5-1.5B SFT, matched 30M tokens\n"
             "form taught by both corpora · decision untestable at this scale",fontsize=15,y=1.03)
fig.tight_layout(); fig.savefig("figures/fig7_formchoice_result.png",bbox_inches="tight")
print("wrote fig7_formchoice_result.png")
