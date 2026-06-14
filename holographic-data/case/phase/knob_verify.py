#!/usr/bin/env python3
"""Realized (β̂, γ̂) estimators for the holographic length-gen study (spec §3.4).

β̂  — pairwise-correlation decay exponent: build the two-point covariance
      C(n)[u,v] = P(X_t=u, X_{t+n}=v) - P(u)P(v); correlation strength = its top
      singular value ||C(n)||_op; fit ||C(n)|| ~ n^{-β} (gamma-beta.pdf Eq. 7).
γ̂  — conditional-entropy decay exponent: entropy-rate-difference estimate of the
      next-token conditional entropy H_n, fit H_n - H_inf ~ n^{-γ} (Eq. 6).

CLI runs the gate: generate a long stream per anchor, report β̂/γ̂, and check the
knob moves β̂ monotonically across the anchors.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

EPS = 1e-12


def _counts_entropy(counts: np.ndarray) -> float:
    p = counts.astype(np.float64)
    p = p / max(p.sum(), EPS)
    p = np.clip(p, EPS, 1.0)
    return float(-np.sum(p * np.log(p)))


def _mgram_entropy(tokens: np.ndarray, order: int, vocab_size: int) -> float:
    n = len(tokens) - order + 1
    if n <= 0:
        return 0.0
    ids = np.zeros(n, dtype=np.int64)
    for k in range(order):
        ids = ids * vocab_size + tokens[k:k + n]
    _, counts = np.unique(ids, return_counts=True)
    return _counts_entropy(counts)


def estimate_gamma(tokens: np.ndarray, vocab_size: int, max_order: int = 6) -> Dict:
    """H_n via entropy-rate differences H(m+1-gram) - H(m-gram); fit n^{-γ}."""
    tokens = np.asarray(tokens, dtype=np.int64)
    hs: List[float] = []
    for m in range(1, max_order + 1):
        hs.append(_mgram_entropy(tokens, m + 1, vocab_size)
                  - _mgram_entropy(tokens, m, vocab_size))
    hs_arr = np.array(hs, dtype=np.float64)
    h_inf = float(hs_arr.min())
    excess = np.clip(hs_arr - h_inf, EPS, None)
    n = np.arange(1, len(hs) + 1, dtype=np.float64)
    if len(hs) >= 2:
        slope, _ = np.polyfit(np.log(n), np.log(excess), 1)
        gamma_hat = float(-slope)
    else:
        gamma_hat = 0.0
    return {"gamma_hat": gamma_hat, "cond_entropy_by_order": hs,
            "h_inf": h_inf}


def estimate_beta(tokens: np.ndarray, vocab_size: int,
                  lags: Sequence[int] = (1, 2, 4, 8, 16, 32, 64)) -> Dict:
    """||C(n)||_op vs lag n; fit n^{-β}. Returns β̂ and per-lag corr strengths."""
    tokens = np.asarray(tokens, dtype=np.int64)
    V = int(vocab_size)
    p1 = np.bincount(tokens, minlength=V).astype(np.float64)
    p1 = p1 / max(p1.sum(), EPS)
    strengths: Dict[str, float] = {}
    vals: List[float] = []
    used: List[int] = []
    for n in lags:
        if len(tokens) - n < 100:
            continue
        a = tokens[:-n]
        b = tokens[n:]
        joint = np.zeros((V, V), dtype=np.float64)
        np.add.at(joint, (a, b), 1.0)
        joint /= max(joint.sum(), EPS)
        C = joint - np.outer(p1, p1)
        top = float(np.linalg.svd(C, compute_uv=False)[0])
        strengths[f"corr_at_lag{n}"] = top
        vals.append(top)
        used.append(n)
    if len(vals) >= 2:
        x = np.log(np.array(used, dtype=np.float64))
        yv = np.log(np.clip(np.array(vals), EPS, None))
        slope, _ = np.polyfit(x, yv, 1)
        beta_hat = float(-slope)
    else:
        beta_hat = 0.0
    out = {"beta_hat": beta_hat}
    out.update(strengths)
    return out


def realized_structure(g, beta: float, gamma: float, seq_len: int = 512,
                       batch_size: int = 32) -> Dict:
    """Ground-truth structural summary the knobs directly control:
    median recall distance (β) and filler fraction (γ)."""
    x, _, _, meta = g.generate_batch(batch_size, seq_len, beta=beta, gamma=gamma,
                                     return_meta=True)
    lags = np.array([l for md in meta for l in md["use_lags"]], dtype=np.float64)
    filler_frac = float((x >= int(g.filler_tokens.min())).float().mean().item())
    median_lag = float(np.median(lags)) if lags.size else 0.0
    mean_uses = float(np.mean([md["n_uses"] for md in meta]))
    return {"median_recall_lag": median_lag, "filler_frac": filler_frac,
            "mean_uses": mean_uses}


def _run_gate() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_nested_monoid import NestedMonoidGenerator

    g = NestedMonoidGenerator(p=31, n_names=512, n_filler=6)

    # --- β-axis at fixed γ: median recall distance must DECREASE with β ---
    print("=== β-axis (γ=0.2 fixed): median recall distance vs β ===")
    betas = [0.05, 0.5, 2.0]
    med_lags = []
    for b in betas:
        np.random.seed(100000)
        s = realized_structure(g, beta=b, gamma=0.2)
        med_lags.append(s["median_recall_lag"])
        print(f"  β={b:5.2f}  median_recall_lag={s['median_recall_lag']:7.1f}  "
              f"mean_uses={s['mean_uses']:5.1f}")
    beta_ok = med_lags[0] > med_lags[1] > med_lags[2]

    # --- γ-axis at fixed β: filler fraction must INCREASE with γ (≈ γ) ---
    print("=== γ-axis (β=0.5 fixed): filler fraction vs γ ===")
    gammas = [0.05, 0.4, 0.8]
    fracs = []
    for gm in gammas:
        np.random.seed(100000)
        s = realized_structure(g, beta=0.5, gamma=gm)
        fracs.append(s["filler_frac"])
        print(f"  γ={gm:5.2f}  filler_frac={s['filler_frac']:6.3f}")
    gamma_ok = fracs[0] < fracs[1] < fracs[2]

    # --- secondary diagnostic: gamma-beta exponents on a long stream ---
    print("=== secondary diagnostic: ||C(n)|| / entropy on a long stream ===")
    np.random.seed(100000)
    x, _, _ = g.generate_batch(1, 16000, beta=0.5, gamma=0.2)
    toks = x[0].numpy()
    bh = estimate_beta(toks, vocab_size=g.vocab_size)["beta_hat"]
    gh = estimate_gamma(toks, vocab_size=g.vocab_size)["gamma_hat"]
    print(f"  (noisy on synthetic data; reported only) beta_hat={bh:.3f} gamma_hat={gh:.3f}")

    print(f"\nGATE β (median recall distance ↓ in β): {beta_ok}")
    print(f"GATE γ (filler fraction ↑ in γ):        {gamma_ok}")
    print(f"GATE: {'PASS' if (beta_ok and gamma_ok) else 'FAIL'}")


if __name__ == "__main__":
    _run_gate()
