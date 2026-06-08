"""Diagnose WHY OpenThoughts scores low (reference H_inf=0).
Three views: (A) BPC vs context length — OT curve keeps plunging (no asymptote)
vs a healthy dataset that flattens; (B) the 3-point extrapolation going negative
-> clamped 0; (C) descaffold: how much of the low score is cross-episode shared
system-prompt pooling vs genuinely template content.
"""
import sys; sys.path.insert(0,"scripts")
import lz_oracle as lz, score_agentic_datasets as sad
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.size":12,"axes.titlesize":13,"savefig.dpi":190,"font.family":"DejaVu Sans"})
NS=[128,512,2048,8192,32768,131072,524288]
REG={e[4]:e for e in sad.REGISTRY}
def get(slug,lim=700):
    e=REG[slug]; p,c,s,ser=e[:4]; gk=e[5] if len(e)>5 else None
    dd=[]
    for d,_ in sad.iter_docs(p,c,s,ser,group_key=gk):
        dd.append(d)
        if len(dd)>=lim: break
    return dd
def bcurve(docs):
    c=lz.build_corpus(docs); return [(n,lz._bpc_at(c,n)) for n in NS if len(c)//n>=8]
def strip(docs,t=0.5):
    cnt=Counter()
    for d in docs:
        for ln in set(d.split("\n")): cnt[ln]+=1
    n=len(docs); sh={l for l,k in cnt.items() if k>t*n}
    return ["\n".join(l for l in d.split("\n") if l not in sh) for d in docs], len(sh)

ot=get("openthoughts-agent-v1-sft"); healthy=get("ii-agent-gaia-traj")
ot_c=bcurve(ot); h_c=bcurve(healthy)
ot_st,nsh=strip(ot); ot_st_c=bcurve(ot_st)
print("OT pooled:",[(n,round(b,2)) for n,b in ot_c])
print(f"OT stripped {nsh} shared lines:",[(n,round(b,2)) for n,b in ot_st_c])

fig,axes=plt.subplots(1,2,figsize=(14,5.8))
# Panel A: curves
ax=axes[0]
ax.plot([n for n,_ in ot_c],[b for _,b in ot_c],"o-",color="#ff7f0e",lw=2.5,ms=7,label="OpenThoughts (pooled): keeps falling")
ax.plot([n for n,_ in ot_st_c],[b for _,b in ot_st_c],"s--",color="#d62728",lw=2,ms=6,label=f"OpenThoughts (scaffold stripped, -{nsh} shared lines)")
ax.plot([n for n,_ in h_c],[b for _,b in h_c],"^-",color="#1f77b4",lw=2.5,ms=7,label="II-Agent GAIA (healthy): flattens ~2.7")
ax.axvline(32768,color="k",ls=":",lw=1,alpha=.6); ax.text(35000,5.5,"32K\n(reference\nmetric)",fontsize=9)
ax.set_xscale("log"); ax.set_xlabel("context length (bytes, log)"); ax.set_ylabel("BPC (bits/char)")
ax.set_title("A. WHY low: OpenThoughts curve never flattens →\nextrapolated floor goes negative → reference clamps to 0")
ax.legend(fontsize=9.5,loc="upper right"); ax.grid(alpha=.3)
# Panel B: the 3-point extrapolation arithmetic
ax=axes[1]
b=dict(ot_c); B1,B2,B3=b[128],b[2048],b[32768]
d12,d23=B1-B2,B2-B3; import math
a=math.log(d12/d23)/math.log(16); hraw=B3-d23**2/(d12-d23)
xs=np.array([128,2048,32768]); ys=np.array([B1,B2,B3])
ax.plot(xs,ys,"o",color="#ff7f0e",ms=12,zorder=3,label="3 measured points")
xf=np.logspace(np.log10(128),6,100); ax.plot(xf,hraw+ (B3-hraw)*(xf/32768)**(-a) if False else hraw+(ys[0]-hraw)*(xf/128.0)**(-a),"--",color="#888",label=f"power-law fit (α={a:.2f})")
ax.axhline(0,color="green",lw=1.5,ls="-",alpha=.5); ax.text(2e5,0.08,"reference floor = 0",color="green",fontsize=10)
ax.axhline(hraw,color="red",lw=1.2,ls=":",); ax.text(2e3,hraw+.3,f"raw extrapolated floor = {hraw:.2f}\n(curve still steep → negative)",color="red",fontsize=10)
ax.scatter([1e6],[max(hraw,0)],marker="*",s=300,color="green",zorder=4,label=f"clamped H∞ = {max(hraw,0):.2f}")
ax.set_xscale("log"); ax.set_ylim(hraw-0.5,7); ax.set_xlabel("context length (bytes, log)"); ax.set_ylabel("BPC / extrapolated floor")
ax.set_title(f"B. Reference 3-point arithmetic:\nH∞ = B₃ − d23²/(d12−d23) = {hraw:.2f} → clamped to 0")
ax.legend(fontsize=9.5,loc="upper right"); ax.grid(alpha=.3)
fig.suptitle("Why OpenThoughts scores H∞=0 (reference metric): template-heavy data whose compressibility keeps\n"
             "improving with context (no irreducible-content asymptote) — partly real template, partly shared-scaffold pooling",fontsize=13,y=1.04)
fig.tight_layout(); fig.savefig("figures/fig_why_openthoughts_low.png",bbox_inches="tight")
print("wrote figures/fig_why_openthoughts_low.png")
