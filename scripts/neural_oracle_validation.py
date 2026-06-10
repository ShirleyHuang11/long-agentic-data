"""Agentic LZ-vs-neural validation (paper §9 experiment #5).

For each registry corpus with a cached text sample, run a small pretrained LM
(Qwen2.5-0.5B) and compute its mean bits-per-token — a *neural* content/
incompressibility estimate. Correlating this across corpora with the LZ-derived
H∞ tests whether the Spearman-0.97 LZ↔neural agreement established on the
formal-math survey carries to agentic trajectories.

Output: data/neural_oracle_bpc.csv  (slug, neural_bpc, lz_h_inf)
Run on a GPU node (sbatch); ~5-10 min.
"""
import csv, math, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
WIN = 2048          # token window
MAX_WIN = 30        # cap windows/corpus (~60K tokens) to bound compute
DEV = "cuda" if torch.cuda.is_available() else "cpu"

reg = {r["slug"]: float(r["h_inf"]) for r in csv.DictReader(open("data/agentic_alpha_hinf.csv"))}
active = {r["slug"] for r in csv.DictReader(open("data/merged_analysis.csv")) if r["role"] != "EXCLUDED"}
slugs = sorted(s for s in (f[:-4] for f in os.listdir("samples_cache") if f.endswith(".txt"))
               if s in active and s in reg)
print(f"{len(slugs)} corpora to score on {DEV}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()

@torch.no_grad()
def neural_bpc(text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if len(ids) < 8:
        return None
    tot_nll, tot_tok = 0.0, 0
    for w in range(min(MAX_WIN, (len(ids) + WIN - 1) // WIN)):
        chunk = ids[w * WIN:(w + 1) * WIN]
        if len(chunk) < 2:
            break
        x = chunk.unsqueeze(0).to(DEV)
        logits = model(x).logits
        # next-token CE over positions 0..n-2 predicting 1..n-1
        ce = torch.nn.functional.cross_entropy(
            logits[0, :-1], x[0, 1:], reduction="sum")
        tot_nll += ce.item()
        tot_tok += len(chunk) - 1
    return tot_nll / tot_tok / math.log(2) if tot_tok else None  # nats→bits

rows = []
for i, s in enumerate(slugs):
    text = open(f"samples_cache/{s}.txt", encoding="utf-8", errors="ignore").read()
    bpc = neural_bpc(text)
    if bpc is not None:
        rows.append({"slug": s, "neural_bpc": round(bpc, 4), "lz_h_inf": reg[s]})
        print(f"  [{i+1}/{len(slugs)}] {s}: neural_bpc={bpc:.3f} lz_H∞={reg[s]:.2f}", flush=True)

with open("data/neural_oracle_bpc.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["slug", "neural_bpc", "lz_h_inf"])
    w.writeheader(); w.writerows(rows)
print(f"wrote data/neural_oracle_bpc.csv ({len(rows)} corpora)", flush=True)

# quick Spearman if scipy available
try:
    from scipy.stats import spearmanr
    import numpy as np
    nb = [r["neural_bpc"] for r in rows]; lz = [r["lz_h_inf"] for r in rows]
    rho, p = spearmanr(nb, lz)
    print(f"SPEARMAN(neural_bpc, LZ_H∞) = {rho:.3f}  (p={p:.1e}, n={len(rows)})", flush=True)
except Exception as e:
    print("scipy spearman skipped:", e, flush=True)
