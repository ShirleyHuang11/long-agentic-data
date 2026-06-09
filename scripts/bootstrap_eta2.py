"""Bootstrap the source-vs-role variance decomposition of H_inf (paper section 5.1).
Resamples the active corpora with replacement B times, recomputes eta^2 for
generator source and train/eval role, and reports point estimates, 95% CIs, and
P(eta2_source > eta2_role). Confirms the central dissociation is robust."""
import csv, statistics as st, random
rows=[r for r in csv.DictReader(open("data/merged_analysis.csv")) if r["role"]!="EXCLUDED"]
def eta2(sample, key):
    H=[float(r["h_inf"]) for r in sample]; gm=st.fmean(H); sst=sum((h-gm)**2 for h in H)
    if sst==0: return 0.0
    g={}
    for r in sample: g.setdefault(r[key],[]).append(float(r["h_inf"]))
    return sum(len(v)*(st.fmean(v)-gm)**2 for v in g.values())/sst
rng=random.Random(0); n=len(rows); B=2000
src=[]; rol=[]; wins=0
for _ in range(B):
    samp=[rows[rng.randrange(n)] for _ in range(n)]
    es,er=eta2(samp,"source"),eta2(samp,"role"); src.append(es); rol.append(er); wins+=es>er
def ci(x): x=sorted(x); return x[int(.025*len(x))], x[int(.975*len(x))]
print(f"n={n} B={B}")
print(f"eta2(source) point={eta2(rows,'source'):.3f} 95%CI [{ci(src)[0]:.3f},{ci(src)[1]:.3f}]")
print(f"eta2(role)   point={eta2(rows,'role'):.3f} 95%CI [{ci(rol)[0]:.3f},{ci(rol)[1]:.3f}]")
print(f"P(source>role)={wins/B:.3f}")
