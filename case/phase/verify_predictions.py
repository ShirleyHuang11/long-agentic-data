#!/usr/bin/env python3
"""Falsifiability check: verify hypothesis-2 predictions against current data.

The boundary hypothesis in
``case/phase/results/emergent_boundary_hypothesis.md`` makes specific
claims about which (β, γ) cells should be chaos vs emergent. This script
loads every ``runs/*/run_summary.csv``, classifies cells with the same
fixed-threshold rule the rest of the pipeline uses
(``utils.classify_fixed``), and prints a table of:

  | id | description | matched cells | observed phase | predicted | status |

Status is one of:
  confirmed — at least one matching cell observed and matches prediction
  refuted   — at least one matching cell observed and contradicts prediction
  pending   — no matching cell observed yet

Used by the autonomous loop to track scientific progress without
hand-eyeballing CSVs each iteration. Falsifiability is the bar for
publishable claims; this enforces it programmatically.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_phase_diagram import aggregate
from utils import PHASE_NAMES, classify_fixed


_CODE_TO_LABEL = {0: "chaos", 1: "emergent", 2: "super_gen", 3: "rote"}


def _label(row) -> str:
    return _CODE_TO_LABEL[classify_fixed(row)]


# Each prediction is a dict with:
#   id:       short stable name
#   desc:     human-readable claim
#   variants: list of variant names whose data is relevant (None = any)
#   where:    callable(row) -> bool selecting matching aggregate cells
#   expected: "chaos" | "emergent" | "rote" | "super_gen" | set thereof
#   min_seeds: int — only count cells with at least this many seeds
PREDICTIONS: List[Dict] = [
    {
        "id": "P1",
        "desc": "(β=0.4, γ=0.02) chaos despite low α_theory=0.025 — refutes single-α hypothesis",
        "variants": ["gamma_axis_b0p4"],
        "where": lambda r: abs(r.beta - 0.40) < 1e-6 and abs(r.gamma - 0.02) < 1e-6,
        "expected": "chaos",
        "min_seeds": 1,
    },
    {
        "id": "P2",
        "desc": "(β=8, γ=0.95) chaos because γ exceeds γ*(β=8)",
        "variants": ["corners"],
        "where": lambda r: abs(r.beta - 8.0) < 1e-3 and abs(r.gamma - 0.95) < 1e-3,
        "expected": "chaos",
        "min_seeds": 1,
    },
    {
        "id": "P3",
        "desc": "alpha_iso_1p0 entirely chaos — α_theory=1.0 always exceeds γ* ceiling",
        "variants": ["alpha_iso_1p0"],
        "where": lambda r: True,
        "expected": "chaos",
        "min_seeds": 1,
    },
    {
        "id": "P4",
        "desc": "gamma_axis_b0p4 entirely chaos — β=0.4 below β* threshold",
        "variants": ["gamma_axis_b0p4"],
        "where": lambda r: True,
        "expected": "chaos",
        "min_seeds": 1,
    },
    {
        "id": "P5",
        "desc": "refine_b0p4_g0p3 entirely chaos — β centered on 0.4 below β*",
        "variants": ["refine_b0p4_g0p3"],
        "where": lambda r: True,
        "expected": "chaos",
        "min_seeds": 1,
    },
    {
        "id": "P6",
        "desc": "alpha_iso_0p1 chaos at β < ~1.0",
        "variants": ["alpha_iso_0p1"],
        "where": lambda r: r.beta < 1.0,
        "expected": "chaos",
        "min_seeds": 1,
    },
    {
        "id": "P7",
        "desc": "alpha_iso_0p1 emergent at β > ~1.5 (high-β + low-γ both satisfied)",
        "variants": ["alpha_iso_0p1"],
        "where": lambda r: r.beta > 1.5,
        "expected": "emergent",
        "min_seeds": 1,
    },
    {
        "id": "P8",
        "desc": "beta_axis_g0p3 chaos at β < ~1.0",
        "variants": ["beta_axis_g0p3"],
        "where": lambda r: r.beta < 1.0,
        "expected": "chaos",
        "min_seeds": 1,
    },
    {
        "id": "P9",
        "desc": "beta_axis_g0p3 emergent at β > ~1.0 (γ=0.3 < γ*(β=1.4)=0.345)",
        "variants": ["beta_axis_g0p3"],
        "where": lambda r: r.beta > 1.0,
        "expected": "emergent",
        "min_seeds": 1,
    },
    {
        "id": "P10",
        "desc": "refine_b2p0_g0p3 emergent at (β=1.4, γ ≤ 0.30) — already verified",
        "variants": ["refine_b2p0_g0p3"],
        "where": lambda r: abs(r.beta - 1.4) < 1e-3 and r.gamma <= 0.30 + 1e-6,
        "expected": "emergent",
        "min_seeds": 3,
    },
    {
        "id": "P11",
        "desc": "refine_b2p0_g0p3 emergent at (β=1.4, γ=0.39) — γ*(1.4) ≥ 0.39",
        "variants": ["refine_b2p0_g0p3"],
        "where": lambda r: abs(r.beta - 1.4) < 1e-3 and abs(r.gamma - 0.39) < 1e-3,
        "expected": "emergent",
        "min_seeds": 1,
    },
    {
        "id": "P12",
        "desc": "standard_complement: γ*(β) increases with β (low-β chaos at high γ, high-β emergent at high γ)",
        "variants": ["standard_complement"],
        "where": lambda r: True,
        "expected": "any",  # this is a structural claim, not a per-cell phase
        "min_seeds": 1,
    },
    {
        "id": "P13",
        "desc": "gamma_axis_b8p0: γ*(β=8) somewhere between observed emergent (γ=0.02) and chaos (γ=0.95)",
        "variants": ["gamma_axis_b8p0"],
        "where": lambda r: True,
        "expected": "any",  # structural
        "min_seeds": 1,
    },
    {
        "id": "P14",
        "desc": "(β=1.4, γ=0.345) emergent at N=3 seeds — robustness of γ*(1.4) lower bound",
        "variants": ["refine_b2p0_g0p3"],
        "where": lambda r: abs(r.beta - 1.4) < 1e-3 and abs(r.gamma - 0.345) < 1e-3,
        "expected": "emergent",
        "min_seeds": 3,
    },
    {
        # Linear regression on the 4 N=3 cells at β=1.4 gives
        #   train_acc(γ) = 0.269 − 0.185 γ   (R² > 0.99)
        # crosses the train_acc=0.20 chaos threshold at γ ≈ 0.373.
        # P15 supersedes P11: contradicts P11's "γ=0.39 emergent" with the
        # quantitative extrapolation. Exactly one of P11 / P15 will confirm.
        "id": "P15",
        "desc": "(β=1.4, γ=0.39) chaos — linear γ*(1.4)≈0.373 from monotone train_acc trend; supersedes P11",
        "variants": ["refine_b2p0_g0p3"],
        "where": lambda r: abs(r.beta - 1.4) < 1e-3 and abs(r.gamma - 0.39) < 1e-3,
        "expected": "chaos",
        "min_seeds": 1,
    },
    {
        # Linear-in-log-β regression on the 4 N=3 alpha_iso_0p1 cells:
        #   train_acc(β | α=0.1) ≈ 0.167 + 0.0224 · log(β)   (R² > 0.97)
        # crosses train_acc=0.20 at log(β) = (0.20 - 0.167)/0.0224 = 1.47,
        # i.e. β ≈ 4.4. So along α=0.1, only the highest-β cell (β=5.0)
        # is predicted emergent — *not* "all β > 1.5".
        # P16 supersedes P7's qualitative "β > 1.5 emergent at α=0.1" by
        # claiming β ∈ (1.5, 4.0) are still chaos. Both P7 and P16 cover
        # the β ∈ (1.5, 4.0) range; both pending until alpha_iso_0p1
        # progresses to those cells.
        "id": "P16",
        "desc": "alpha_iso_0p1 cells at β ∈ (1.5, 4.0) are CHAOS — linear-in-log-β fit gives β*(α=0.1) ≈ 4.4; supersedes P7 partially",
        "variants": ["alpha_iso_0p1"],
        "where": lambda r: 1.5 < r.beta < 4.0,
        "expected": "chaos",
        "min_seeds": 1,
    },
    {
        # 2D linear fit train_acc(β, γ) = 0.262 + 0.0224·log(β) − 0.185·γ
        # predicts train_acc(4.0, 0.2) = 0.262 + 0.031 − 0.037 = 0.256.
        # 0.256 > 0.20 chaos threshold ⇒ emergent.
        # Job 9030513 (single-cell probe at this point, 3 seeds, 4h wall)
        # tests the 2D linear extrapolation at a (β, γ) point not covered
        # by any other variant. Distinguishes:
        #   - linear separable fit holds out-of-strip   ⇒ ta ∈ [0.24, 0.27]
        #   - interaction term needed (high-β favoured) ⇒ ta > 0.27
        #   - linear fit fails at extrapolation         ⇒ ta < 0.24
        "id": "P17",
        "desc": "(β=4.0, γ=0.2) emergent — 2D linear fit predicts train_acc≈0.256",
        "variants": ["pilot_b4p0_g0p2"],
        "where": lambda r: True,
        "expected": "emergent",
        "min_seeds": 1,
    },
    {
        # 2D linear fit predicts train_acc(4.0, 0.6) = 0.262 + 0.031 − 0.111 = 0.182
        # 0.182 < 0.20 ⇒ chaos. Job 9033655 (single, 1 seed, 2h wall).
        # If the (β=8, γ=0.02) corner-anomaly is a general high-β bonus
        # (interaction term hypothesis), this cell would come out
        # emergent (refuting P18). If linear fit holds at moderate-β /
        # moderate-γ, it stays chaos (confirming P18).
        "id": "P18",
        "desc": "(β=4.0, γ=0.6) chaos — 2D linear fit predicts train_acc≈0.182, just below threshold",
        "variants": ["pilot_b4p0_g0p6"],
        "where": lambda r: True,
        "expected": "chaos",
        "min_seeds": 1,
    },
    {
        # 2D linear fit predicts train_acc(6.4, 0.7) = 0.262 + 0.042 − 0.130 = 0.174
        # 0.174 < 0.20 ⇒ chaos. Job 9033656 (single, 1 seed, 2h wall).
        # Tests the same interaction-vs-linear question at a higher β
        # closer to the (β=8, γ=0.02) anomalous corner. If interaction
        # term is needed, P19 refutes; if linear fit holds, P19 confirms.
        "id": "P19",
        "desc": "(β=6.4, γ=0.7) chaos — 2D linear fit predicts train_acc≈0.174",
        "variants": ["pilot_b6p4_g0p7"],
        "where": lambda r: True,
        "expected": "chaos",
        "min_seeds": 1,
    },
    {
        # iter-19 corrected fit on the 5 emergent cells:
        #   train_acc(β, γ) = 0.254 + 0.0523·log(β) − 0.197·γ   (R²=0.9999)
        # predicts train_acc(4, 0.6) = 0.254 + 0.0725 − 0.118 = 0.214 ⇒ emergent.
        # Direct contradiction with P18 (chaos at 0.182 from iter-13 fit).
        # Whichever fit extrapolates better wins when 9033655 lands.
        "id": "P21",
        "desc": "(β=4.0, γ=0.6) emergent — corrected (iter-19) 2D fit predicts train_acc≈0.214; supersedes P18",
        "variants": ["pilot_b4p0_g0p6"],
        "where": lambda r: True,
        "expected": "emergent",
        "min_seeds": 1,
    },
    {
        # iter-19 corrected fit predicts train_acc(6.4, 0.7) = 0.215 ⇒ emergent.
        # Direct contradiction with P19. Same falsifiability test as P21
        # but at higher β.
        "id": "P22",
        "desc": "(β=6.4, γ=0.7) emergent — corrected (iter-19) 2D fit predicts train_acc≈0.215; supersedes P19",
        "variants": ["pilot_b6p4_g0p7"],
        "where": lambda r: True,
        "expected": "emergent",
        "min_seeds": 1,
    },
    {
        # iter-19 corrected fit predicts γ*(β=8) ≈ 0.825. So at β=8,
        # γ=0.755 should still be just emergent: train_acc(8, 0.755) =
        # 0.254 + 0.109 − 0.149 = 0.214. Job 9016098 (gamma_axis_b8p0)
        # samples γ ∈ {0.02, 0.265, 0.51, 0.755, 1.0} at β=8; the
        # boundary should fall between γ=0.755 (emergent) and γ=1.0
        # (chaos), confirming the quantitative formula.
        "id": "P23",
        "desc": "(β=8, γ=0.755) emergent — corrected fit predicts γ*(β=8)≈0.825 (from γ*(β)≈0.274+0.265·log(β))",
        "variants": ["gamma_axis_b8p0"],
        "where": lambda r: abs(r.beta - 8.0) < 1e-3 and abs(r.gamma - 0.755) < 1e-3,
        "expected": "emergent",
        "min_seeds": 1,
    },
    {
        # iter-46 anchor pilot (Natural, β=2.0, γ=0.8) — proposal's
        # Region I "learnable / Wikipedia-like" anchor.
        # 18-cell weighted OLS predicts:
        #   train_acc(2.0, 0.8) = 0.270 + 0.0475·log(2) − 0.254·0.8
        #                       = 0.270 + 0.033 − 0.203 = 0.100
        # Below 0.20 chaos threshold ⇒ predicted chaos.
        # Mechanistic reading: γ=0.8 means 80% of tokens are noise, the
        # model's locality (β=2) has nothing to retrieve from.
        "id": "P26",
        "desc": "(β=2.0, γ=0.8) chaos — Natural anchor; weighted 18-cell fit predicts train_acc≈0.100",
        "variants": ["anchor_natural_b2p0_g0p8"],
        "where": lambda r: True,
        "expected": "chaos",
        "min_seeds": 1,
    },
    {
        # iter-46 anchor pilot (Edge-of-Chaos, β=0.05, γ=0.05) —
        # proposal's "sweet spot" anchor for strong reasoning. 18-cell
        # weighted fit predicts:
        #   train_acc(0.05, 0.05) = 0.270 + 0.0475·log(0.05) − 0.254·0.05
        #                         = 0.270 − 0.142 − 0.013 = 0.115
        # Below 0.20 chaos threshold ⇒ predicted chaos.
        # If confirmed: Transformer's locality bias is useless when
        # retrieval distance is uniform; this is the regime where
        # Mamba should escape chaos (the proposal's central claim).
        "id": "P27",
        "desc": "(β=0.05, γ=0.05) chaos — Edge-of-Chaos anchor; weighted 18-cell fit predicts train_acc≈0.115",
        "variants": ["anchor_edge_b0p05_g0p05"],
        "where": lambda r: True,
        "expected": "chaos",
        "min_seeds": 1,
    },
    {
        # iter-39 Mamba at Natural anchor (β=2.0, γ=0.8). Transformer is
        # chaos here (P26 confirmed N=3 ta=0.085) because γ=0.8 means 80%
        # noise — no architecture should save the model. Mechanism failure
        # is at the data side, not the model side. Predicting chaos for
        # Mamba too as a control: if it's emergent, the proposal's
        # "no signal genuinely" claim about high-γ chaos is wrong.
        "id": "P30",
        "desc": "(β=2.0, γ=0.8) chaos for Mamba — high-γ chaos is data-limited, not architecture-limited",
        "variants": ["mamba_natural_b2p0_g0p8"],
        "where": lambda r: True,
        "expected": "chaos",
        "min_seeds": 1,
    },
    {
        # iter-39 Mamba at strip cell (β=1.4, γ=0.21). Transformer is
        # firmly emergent here (P10 confirmed N=3 ta=0.230, lengthgen
        # ratio 0.385 = real retrieval). Mamba should also retrieve here
        # (signal is present, β=1.4 is not extreme low). Whether Mamba
        # achieves higher ta than Transformer or just matches is the
        # interesting question; either way the cell should be emergent.
        "id": "P32",
        "desc": "(β=1.4, γ=0.21) emergent for Mamba — strip cell where Transformer already retrieves; signal present",
        "variants": ["mamba_strip_b1p4_g0p21"],
        "where": lambda r: True,
        "expected": "emergent",
        "min_seeds": 1,
    },
    {
        # iter-33 boundary probe (β=0.5, γ=0.05) — discriminator between
        # the linear 18-cell fit and the concavity-corrected model.
        #   linear fit:   train_acc(0.5, 0.05) = 0.270 + 0.0475·log(0.5)
        #                                       − 0.254·0.05 = 0.224  → emergent
        #   concavity:    expected ~0.18 (interior bias ~-0.04 at β=0.5)
        #                                                         → chaos
        # If chaos: linear fit's interior over-prediction (-0.055 at β=0.2,
        # -0.059 at β=0.3) extends to β=0.5; concavity-in-log(β) confirmed.
        # If emergent (ta ≥ 0.20): linear fit valid here; concavity only
        # in β ∈ [0.05, 0.3].
        "id": "P29",
        "desc": "(β=0.5, γ=0.05) chaos — concavity hypothesis predicts ta≈0.18 < 0.20 (linear fit said 0.224 emergent)",
        "variants": ["probe_b0p5_g0p05"],
        "where": lambda r: True,
        "expected": "chaos",
        "min_seeds": 1,
    },
    {
        # iter-47 anchor pilot (CoT, β=0.5, γ=0.4) — proposal's
        # transition anchor (Region II / yellow). Weighted fit predicts:
        #   train_acc(0.5, 0.4) = 0.270 + 0.0475·log(0.5) − 0.254·0.4
        #                      = 0.270 − 0.033 − 0.102 = 0.135
        # Below 0.20 chaos threshold ⇒ predicted chaos.
        # CoT is the third anchor and the boundary discriminator: if
        # this one lands emergent (which would surprise the fit by
        # 0.07), the 18-cell fit's γ-slope is wrong and we revisit.
        "id": "P28",
        "desc": "(β=0.5, γ=0.4) chaos — CoT anchor; weighted 18-cell fit predicts train_acc≈0.135",
        "variants": ["anchor_cot_b0p5_g0p4"],
        "where": lambda r: True,
        "expected": "chaos",
        "min_seeds": 1,
    },
    {
        # User-submitted pilot (job 9029967) trains the same (β=1.4, γ=0.345)
        # cell but for 30000 steps (6× the 5000-step default). Tests
        # whether 5000 steps already captures the model's capacity at
        # this cell or if more compute pushes train_acc up.
        # AULC analysis (iter-23, Result 10) shows emergent cells have
        # AULC≈0.04 — model is still learning at end-of-training. So 6×
        # more steps may improve train_acc, but the loss reduction
        # potential is bounded by AULC × 5000-step plateau.
        # The 5k baseline at this cell is train_acc=0.205 (just-emergent).
        # Predict 30k stays emergent: small monotone train_acc rise but
        # no jump to rote (which would need train_acc ≥ 0.40). If 30k
        # gives train_acc ≥ 0.40, the 5k-based phase boundary is
        # fundamentally compromised and we'd need to redo the analysis
        # at the longer training horizon.
        "id": "P24",
        "desc": "(β=1.4, γ=0.345, 30k steps) emergent — 5k loss-mostly-converged; 6× more should slightly improve but not flip phase",
        "variants": ["pilot_30k_b1p4_g0p345"],
        "where": lambda r: True,
        "expected": "emergent",
        "min_seeds": 1,
    },
    {
        # standard_complement (job 9008462, RUNNING since iter-31) covers
        # β ∈ {0.8, 1.6, 3.2, 6.4} × γ ∈ {0.05, 0.208, 0.367, 0.525,
        # 0.683, 0.842, 1.0}. The corrected 2D fit predicts:
        #   train_acc(3.2, 0.525) = 0.254 + 0.0523·log(3.2) − 0.197·0.525
        #                         = 0.254 + 0.0608 − 0.1034
        #                         = 0.212
        # This is the most discriminating cell in the complement —
        # only 0.012 above the chaos threshold of 0.20. If the corrected
        # fit's β-extrapolation holds beyond the original β=1.4 strip
        # data, this cell is JUST emergent. If the fit overestimates
        # train_acc at higher β (e.g. due to a saturating effect), this
        # cell flips to chaos.
        # Also tests the iter-19 vs iter-13 fit comparison: iter-13 fit
        # gives 0.262 + 0.0224·log(3.2) − 0.185·0.525 = 0.262 + 0.0260 −
        # 0.0971 = 0.191 (chaos). So this cell is exactly where the two
        # fits disagree at the chaos boundary — a clean discriminator.
        "id": "P25",
        "desc": "(β=3.2, γ=0.525) emergent — corrected fit predicts train_acc=0.212; iter-13 fit said 0.191 chaos; cleanest discriminator on standard_complement",
        "variants": ["standard_complement"],
        "where": lambda r: abs(r.beta - 3.2) < 1e-3 and abs(r.gamma - 0.525) < 1e-3,
        "expected": "emergent",
        "min_seeds": 1,
    },
]


def collect_per_variant_aggregates(runs_dir: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for vdir in sorted(runs_dir.iterdir()):
        csv = vdir / "run_summary.csv"
        if not csv.exists() or not vdir.is_dir():
            continue
        df = pd.read_csv(csv)
        if df.empty:
            continue
        out[vdir.name] = aggregate(df)
    return out


def evaluate_one(pred: Dict, by_variant: Dict[str, pd.DataFrame]) -> Dict:
    matched: List[Dict] = []
    for vname in pred["variants"]:
        agg = by_variant.get(vname)
        if agg is None:
            continue
        for _, r in agg.iterrows():
            if int(r["n_seeds"]) < pred["min_seeds"]:
                continue
            if not pred["where"](r):
                continue
            matched.append({
                "variant": vname,
                "beta": float(r["beta"]),
                "gamma": float(r["gamma"]),
                "n_seeds": int(r["n_seeds"]),
                "phase": _label(r),
                "train_acc": float(r["train_acc_mean"]),
                "long_acc": float(r["long_acc_mean"]),
            })

    if not matched:
        status = "pending"
        observed = "—"
    elif pred["expected"] == "any":
        status = "observed"
        observed = sorted({m["phase"] for m in matched})
    else:
        phases = {m["phase"] for m in matched}
        if pred["expected"] in phases and len(phases) == 1:
            status = "confirmed"
        elif pred["expected"] not in phases:
            status = "refuted"
        else:
            status = "mixed"
        observed = sorted(phases)

    return {
        "id": pred["id"],
        "desc": pred["desc"],
        "expected": pred["expected"],
        "observed": observed,
        "status": status,
        "n_matched": len(matched),
        "matched": matched,
    }


def render_markdown(results: List[Dict]) -> str:
    lines = ["# Predictions verification\n",
             "_See case/phase/results/emergent_boundary_hypothesis.md for hypothesis._\n"]
    counts: Dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines.append(f"\n**Status counts:** {counts}\n")
    lines.append("\n| id | status | n_matched | expected | observed | description |")
    lines.append("|---|---|---:|---|---|---|")
    for r in results:
        obs = ",".join(r["observed"]) if isinstance(r["observed"], list) else str(r["observed"])
        lines.append(
            f"| `{r['id']}` | **{r['status']}** | {r['n_matched']} | "
            f"`{r['expected']}` | `{obs}` | {r['desc']} |"
        )

    refuted = [r for r in results if r["status"] == "refuted"]
    if refuted:
        lines.append("\n## Refuted predictions — needs hypothesis revision\n")
        for r in refuted:
            lines.append(f"\n### `{r['id']}`: {r['desc']}\n")
            lines.append("| variant | β | γ | n_seeds | observed phase | train_acc | long_acc |")
            lines.append("|---|---:|---:|---:|---|---:|---:|")
            for m in r["matched"]:
                lines.append(
                    f"| `{m['variant']}` | {m['beta']:.4g} | {m['gamma']:.4g} | "
                    f"{m['n_seeds']} | `{m['phase']}` | {m['train_acc']:.3f} | "
                    f"{m['long_acc']:.3f} |"
                )
    return "\n".join(lines)


def render_terminal_summary(results: List[Dict]) -> str:
    """One-screen status — easier to skim than the full markdown table."""
    counts: Dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines = [f"verify_predictions  ({sum(counts.values())} total)"]
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    lines.append("  " + ", ".join(parts))
    by_status: Dict[str, List[Dict]] = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r)
    for status in ("refuted", "mixed", "confirmed", "observed", "pending"):
        for r in by_status.get(status, []):
            tag = f"[{status.upper():>9}]"
            lines.append(f"  {tag} {r['id']}: {r['desc']}")
    return "\n".join(lines)


def main() -> None:
    runs_dir = Path(__file__).resolve().parent / "runs"
    out_path = Path(__file__).resolve().parent / "results" / "predictions_verification.md"

    by_variant = collect_per_variant_aggregates(runs_dir)
    results = [evaluate_one(p, by_variant) for p in PREDICTIONS]

    md = render_markdown(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)

    # Print compact terminal summary by default; full markdown is on disk.
    if "--full" in sys.argv:
        print(md)
    else:
        print(render_terminal_summary(results))
    print(f"\n[wrote {out_path}]")


if __name__ == "__main__":
    main()
