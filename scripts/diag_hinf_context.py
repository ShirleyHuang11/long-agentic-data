"""Diagnose H_inf=0 clamping: measure BPC at larger context sizes and refit.
The 3-point oracle caps at n=32768; for highly-templated agentic data the BPC
curve is still falling there, so the power-law floor extrapolates negative and
is clamped to 0. Test: extend to n=131072/524288 and refit over all points.
"""
import sys, os, math
sys.path.insert(0, "scripts")
import numpy as np, lz_oracle
import score_agentic_datasets as sad

NS = [128, 2048, 32768, 131072, 524288]

def bpc_curve(slug):
    e = next(x for x in sad.REGISTRY if x[4]==slug)
    path,cfg,splits,ser = e[:4]; gk = e[5] if len(e)>5 else None
    docs, size = [], 0
    for doc,_ in sad.iter_docs(path,cfg,splits,ser,group_key=gk):
        docs.append(doc); size += len(doc)
        if size>=lz_oracle.MAX_BYTES or len(docs)>=lz_oracle.MAX_DOCS: break
    corpus = lz_oracle.build_corpus(docs)
    return {n: lz_oracle._bpc_at(corpus, n) for n in NS if n < len(corpus)}, len(corpus)

def fit_floor(curve):
    ns = sorted(curve); b = [curve[n] for n in ns]
    # 3-point (original protocol, first three)
    r=16; d12,d23=b[0]-b[1],b[1]-b[2]
    a3 = math.log(d12/d23)/math.log(r) if d12>0 and d23>0 else float('nan')
    h3 = b[2]-d23/(r**a3-1) if a3==a3 else float('nan')
    # full nonlinear-ish: fit log(BPC-Hf)=log c - a log n by scanning Hf
    best=None
    for Hf in np.linspace(-0.5, min(b)-0.01, 400):
        y=np.log(np.array(b)-Hf); x=np.log(np.array(ns))
        A=np.vstack([x,np.ones_like(x)]).T
        sol,res,*_=np.linalg.lstsq(A,y,rcond=None)
        r2 = 1-(res[0] if len(res) else ((y-A@sol)**2).sum())/((y-y.mean())**2).sum()
        if best is None or r2>best[0]: best=(r2,Hf,-sol[0])
    return a3,h3,best

for slug in ["openthoughts-agent-v1-sft","apigen-mt-5k","weblinx-actions"]:
    curve,clen = bpc_curve(slug)
    a3,h3,(r2,Hf,a_full)=fit_floor(curve)
    print(f"\n=== {slug}  (corpus {clen} bytes) ===")
    print("  BPC:", {n:round(v,3) for n,v in curve.items()})
    print(f"  3-point: alpha={a3:.3f} h_inf_raw={h3:.3f} (clamped {max(h3,0):.3f})")
    print(f"  multi-point floor fit: H_inf={Hf:.3f} alpha={a_full:.3f} R2={r2:.4f}")
