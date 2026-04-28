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


def main() -> None:
    runs_dir = Path(__file__).resolve().parent / "runs"
    out_path = Path(__file__).resolve().parent / "results" / "predictions_verification.md"

    by_variant = collect_per_variant_aggregates(runs_dir)
    results = [evaluate_one(p, by_variant) for p in PREDICTIONS]

    md = render_markdown(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)

    print(md)
    print(f"\n[wrote {out_path}]")


if __name__ == "__main__":
    main()
