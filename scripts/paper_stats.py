"""Single source of truth for the paper's headline numbers, recomputed from
data/merged_analysis.csv (+ gamma_beta*.csv, hurst via merged). Run after any
registry change and diff the output against the paper to catch drift.
Consolidates the recompute snippets used across the iteration audits."""
import csv, statistics as st, random
rows=[r for r in csv.DictReader(open("data/merged_analysis.csv")) if r["role"]!="EXCLUDED"]
csv_rows=sum(1 for _ in csv.DictReader(open("data/agentic_alpha_hinf.csv")))
F=lambda r,k:float(r[k])
def med(g,k="h_inf"): return st.median([F(r,k) for r in g])
print(f"ACTIVE={len(rows)}  CSV_ROWS={csv_rows}")
print("\n[role]  n / medH∞ / medα")
for role in ["TRAIN","EVAL_TASK","EVAL_TRAJ"]:
    g=[r for r in rows if r["role"]==role]; print(f"  {role:10s} {len(g):3d}  {med(g):.2f}  {med(g,'alpha'):.2f}")
print(f"  alpha band: {min(med([r for r in rows if r['role']==x],'alpha') for x in ['TRAIN','EVAL_TASK','EVAL_TRAJ']):.2f}-{max(med([r for r in rows if r['role']==x],'alpha') for x in ['TRAIN','EVAL_TASK','EVAL_TRAJ']):.2f}")
print("\n[source]  n / medH∞ / mean / pos>0.3")
for s in ["human_task","human_demo","frontier","synth_task","mid","distill"]:
    g=[r for r in rows if r["source"]==s]
    print(f"  {s:11s} {len(g):3d}  {med(g):.2f}  {st.fmean([F(r,'h_inf') for r in g]):.2f}  {sum(1 for r in g if F(r,'h_inf')>0.3)}/{len(g)}")
print(f"\ncontent gap: EVAL_TASK {med([r for r in rows if r['role']=='EVAL_TASK']):.2f} vs TRAIN {med([r for r in rows if r['role']=='TRAIN']):.2f}")
H=[F(r,'h_inf') for r in rows]; print(f"ALL: median {st.median(H):.2f} mean {st.fmean(H):.2f} floor%(<0.05) {sum(1 for h in H if h<0.05)/len(H):.0%}")
# eta2
gm=st.fmean(H); sst=sum((h-gm)**2 for h in H)
def eta2(k):
    g={}
    for r in rows: g.setdefault(r[k],[]).append(F(r,'h_inf'))
    return sum(len(v)*(st.fmean(v)-gm)**2 for v in g.values())/sst
print(f"\neta2: source {eta2('source'):.3f}  role {eta2('role'):.3f}  domain {eta2('domain'):.3f}")
# spearman
def spear(xs,ys):
    n=len(xs)
    def rk(v):
        o=sorted(range(n),key=lambda i:v[i]);r=[0]*n;i=0
        while i<n:
            j=i
            while j+1<n and v[o[j+1]]==v[o[i]]:j+=1
            for k in range(i,j+1):r[o[k]]=(i+j)/2+1
            i=j+1
        return r
    rx,ry=rk(xs),rk(ys);mx,my=st.fmean(rx),st.fmean(ry)
    return sum((a-mx)*(b-my) for a,b in zip(rx,ry))/((sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))**.5)
a=[F(r,'alpha') for r in rows]; b=[F(r,'bpc_32768') for r in rows]
hur=[(F(r,'hurst'),F(r,'h_inf')) for r in rows if r['hurst']]
print(f"spearman(alpha,H)={spear(a,H):+.2f}  (BPC32k,H)={spear(b,H):+.2f}  (Hurst,H)={spear([x for x,_ in hur],[y for _,y in hur]):+.2f} (n={len(hur)})")
# length
mb=[F({'x':r['mean_doc_bytes']},'x') if False else float(r['mean_doc_bytes']) for r in rows]
import csv as _c
reg={r['slug']:r for r in _c.DictReader(open('data/agentic_alpha_hinf.csv'))}
B=[float(reg[r['slug']]['mean_doc_bytes']) for r in rows]; T=[float(reg[r['slug']]['mean_turns']) for r in rows]
print(f"spearman(bytes/ep,H)={spear(B,H):+.2f}  (turns,H)={spear(T,H):+.2f}")
print("\n[domain]  n / medH∞ / medBPC / medα")
doms={}
for r in rows: doms.setdefault(r['domain'],[]).append(r)
for d,g in sorted(doms.items(),key=lambda kv:-med(kv[1])):
    print(f"  {d:9s} {len(g):3d}  {med(g):.2f}  {med(g,'bpc_32768'):.2f}  {med(g,'alpha'):.2f}")

# two-way partial eta^2 (source controlling for domain, and vice versa) -- section 5.1
def _resid(key):
    g={}
    for r in rows: g.setdefault(r[key],[]).append(F(r,'h_inf'))
    mean={k:st.fmean(v) for k,v in g.items()}
    return [F(r,'h_inf')-mean[r[key]] for r in rows]
def _eta2_resid(vals,key):
    m=st.fmean(vals); g={}
    for v,r in zip(vals,rows): g.setdefault(r[key],[]).append(v)
    ss=sum((v-m)**2 for v in vals)
    return (sum(len(v)*(st.fmean(v)-m)**2 for v in g.values())/ss) if ss else 0
print(f"\npartial eta2(source|domain)={_eta2_resid(_resid('domain'),'source'):.2f}  partial eta2(domain|source)={_eta2_resid(_resid('source'),'domain'):.2f}")

# --check-paper: assert the count-bearing table rows appear literally in the paper.
# These n's drift on every registry add (the recurring stale-number bug); this
# flags them automatically so each loop catches drift without a manual diff.
import sys
if "--check-paper" in sys.argv:
    txt=open("paper/long_horizon_agentic_data.md").read()
    n=lambda role:sum(1 for r in rows if r["role"]==role)
    ns=lambda s:sum(1 for r in rows if r["source"]==s)
    checks=[
        (f"| TRAIN | {n('TRAIN')} |","role TRAIN n"),
        (f"| EVAL_TASK | {n('EVAL_TASK')} |","role EVAL_TASK n"),
        (f"| EVAL_TRAJ | {n('EVAL_TRAJ')} |","role EVAL_TRAJ n"),
        (f"| human task (written problems) | {ns('human_task')} |","src human_task n"),
        (f"| human demo (action streams) | {ns('human_demo')} |","src human_demo n"),
        (f"| frontier-model rollout | {ns('frontier')} |","src frontier n"),
        (f"| mid-size-model rollout | {ns('mid')} |","src mid n"),
        (f"| distilled SFT mixture | {ns('distill')} |","src distill n"),
        (f"human task n = {ns('human_task')}","limitations human_task n"),
    ]
    print("\n[check-paper]")
    bad=0
    for needle,label in checks:
        ok=needle in txt
        bad+=not ok
        print(f"  {'OK ' if ok else 'MISS'} {label}: '{needle}'")
    print(f"  {bad} mismatch(es)")
    sys.exit(1 if bad else 0)
