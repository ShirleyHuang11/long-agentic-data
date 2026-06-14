#!/usr/bin/env python3
"""KV ↔ Logical Folding universality test at the 4 paper anchors.

For each anchor (β, γ), pulls per-seed cells from every
``case/phase/runs/*/run_summary.csv`` matching that point, splits by task
(``kv`` if no logical-folding marker in run_summary or run-dir name, else
``logical_folding``) and architecture (``transformer`` / ``mamba``), and
prints/writes a side-by-side comparison plus the ``classify_fixed``
phase-code per (task, arch) cell. Mismatched phase codes between KV and LF
falsify the task-universal-boundary claim.

Usage::

    python case/phase/cross_task_compare.py
    python case/phase/cross_task_compare.py --out case/phase/results/cross_task_universality.md
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PHASE_NAMES, classify_fixed


ANCHORS: List[Tuple[str, float, float]] = [
    ("edge",    0.05, 0.05),
    ("strip",   1.4,  0.21),
    ("cot",     0.5,  0.4),
    ("natural", 2.0,  0.8),
]
TOL = 5e-3


def _classify_row(mean_row: Dict[str, float]) -> int:
    return classify_fixed({
        "train_acc_mean": mean_row["train_acc"],
        "long_acc_mean":  mean_row["long_acc"],
        "gap_mean":       mean_row["generalization_gap"],
        "retention_mean": mean_row["retention_ratio"],
    })


def _label_run(path: Path) -> Tuple[str, str]:
    """Return (task, arch) labels for a run directory.

    KV is the default; LF is identified by a leading 'lf_' in the run-dir
    name. Mamba is identified by a leading 'mamba_'. The trainer doesn't
    yet write task/arch into the CSV, so directory naming is the source of
    truth (matches sweep.sh + sweep.slurm conventions).
    """
    name = path.parent.name
    arch = "mamba" if name.startswith("mamba_") else "transformer"
    task = "logical_folding" if name.startswith("lf_") else "kv"
    return task, arch


def collect(anchors=ANCHORS, runs_dir: Path = Path("case/phase/runs")) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for csv_path in sorted(runs_dir.glob("*/run_summary.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        task, arch = _label_run(csv_path)
        for label, b, g in anchors:
            mask = df["beta"].between(b - TOL, b + TOL) & df["gamma"].between(g - TOL, g + TOL)
            sub = df[mask]
            if not len(sub):
                continue
            for _, r in sub.iterrows():
                rows.append({
                    "anchor": label, "task": task, "arch": arch,
                    "beta": float(r["beta"]), "gamma": float(r["gamma"]),
                    "seed": int(r["seed"]),
                    "train_acc": float(r["train_acc"]),
                    "long_acc":  float(r["long_acc"]),
                    "generalization_gap": float(r["generalization_gap"]),
                    "retention_ratio":    float(r["retention_ratio"]),
                    "source": csv_path.parent.name,
                })
    return pd.DataFrame(rows)


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds within each (anchor, task, arch) cell, plus phase code."""
    if per_seed.empty:
        return pd.DataFrame()
    agg = (per_seed
           .groupby(["anchor", "task", "arch"], as_index=False)
           .agg(n_seeds=("seed", "nunique"),
                train_acc=("train_acc", "mean"),
                train_acc_std=("train_acc", "std"),
                long_acc=("long_acc", "mean"),
                long_acc_std=("long_acc", "std"),
                generalization_gap=("generalization_gap", "mean"),
                retention_ratio=("retention_ratio", "mean")))
    agg["phase_code"] = agg.apply(lambda r: _classify_row(r.to_dict()), axis=1)
    agg["phase"] = agg["phase_code"].map(PHASE_NAMES)
    anchor_order = {a: i for i, (a, *_rest) in enumerate(ANCHORS)}
    agg["_a"] = agg["anchor"].map(anchor_order)
    agg = agg.sort_values(["_a", "task", "arch"]).drop(columns=["_a"]).reset_index(drop=True)
    return agg


def universality_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one-row-per-anchor with task-arch columns, plus a 'kv vs lf' verdict."""
    if agg.empty:
        return pd.DataFrame()
    out_rows: List[Dict[str, object]] = []
    for label, b, g in ANCHORS:
        row: Dict[str, object] = {"anchor": label, "beta": b, "gamma": g}
        kv_xfmr = agg[(agg.anchor == label) & (agg.task == "kv") & (agg.arch == "transformer")]
        lf_xfmr = agg[(agg.anchor == label) & (agg.task == "logical_folding") & (agg.arch == "transformer")]
        for tag, sub in (("kv_xfmr", kv_xfmr), ("lf_xfmr", lf_xfmr)):
            if len(sub):
                r = sub.iloc[0]
                row[f"{tag}_phase"] = r["phase"]
                row[f"{tag}_train"] = round(float(r["train_acc"]), 3)
                row[f"{tag}_long"]  = round(float(r["long_acc"]), 3)
                row[f"{tag}_gap"]   = round(float(r["generalization_gap"]), 3)
                row[f"{tag}_n"]     = int(r["n_seeds"])
            else:
                row[f"{tag}_phase"] = "—"
                row[f"{tag}_train"] = np.nan
                row[f"{tag}_long"]  = np.nan
                row[f"{tag}_gap"]   = np.nan
                row[f"{tag}_n"]     = 0
        if row["kv_xfmr_n"] and row["lf_xfmr_n"]:
            same = row["kv_xfmr_phase"] == row["lf_xfmr_phase"]
            row["universal"] = "✓ same phase" if same else "✗ different phase"
        else:
            row["universal"] = "pending"
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _df_to_md(df: pd.DataFrame, cols: List[str]) -> str:
    sub = df[cols]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, r in sub.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and not np.isnan(v):
                cells.append(f"{v:.4g}")
            elif isinstance(v, float):
                cells.append("—")
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


def to_markdown(agg: pd.DataFrame, univ: pd.DataFrame) -> str:
    lines: List[str] = ["# KV ↔ Logical Folding universality at 4 anchors\n"]
    if univ.empty:
        return "\n".join(lines + ["No data found in case/phase/runs/.\n"])

    lines.append("## Per-anchor verdict\n")
    cols = ["anchor", "beta", "gamma",
            "kv_xfmr_phase", "kv_xfmr_train", "kv_xfmr_long", "kv_xfmr_n",
            "lf_xfmr_phase", "lf_xfmr_train", "lf_xfmr_long", "lf_xfmr_n",
            "universal"]
    lines.append(_df_to_md(univ, cols))
    lines.append("")

    lines.append("## All cells (incl. mamba where present)\n")
    show_cols = ["anchor", "task", "arch", "n_seeds",
                 "train_acc", "long_acc", "generalization_gap",
                 "retention_ratio", "phase"]
    lines.append(_df_to_md(agg.round(4), show_cols))
    lines.append("")

    pending = (univ["universal"] == "pending").sum()
    if pending:
        lines.append(f"_{pending}/4 anchors still missing one or both task cells._\n")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None,
                    help="optional Markdown output path")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = repo_root / "case" / "phase" / "runs"
    per_seed = collect(runs_dir=runs_dir)
    agg = aggregate(per_seed)
    univ = universality_table(agg)

    md = to_markdown(agg, univ)
    print(md)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        print(f"\n[wrote] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
