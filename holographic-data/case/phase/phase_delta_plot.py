#!/usr/bin/env python3
"""Δ phase diagram: Mamba − Transformer at matched (β, γ) cells.

`phase_continuous_plot.py` excludes `mamba_*` runs (different
architecture, different config) so the main diagram is Transformer-only.
This script does the architecture comparison directly: collect both
cohorts separately, inner-join on (β, γ), and color by metric difference.

Diverging colormap centered at 0:
  blue = Mamba > Transformer    red = Transformer > Mamba

Output: case/phase/figures/phase_delta_mamba_minus_transformer.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_phase_diagram import aggregate
from utils import load_config


# Mamba runs that should NOT be aggregated: smoke tests, aborted full runs,
# anything not at the 5000-step canonical config.
MAMBA_EXCLUDE = {"mamba_smoke_edge", "mamba_edge_full"}


def _collect(runs_dir: Path, predicate) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for vdir in sorted(runs_dir.iterdir()):
        if not vdir.is_dir() or not predicate(vdir.name):
            continue
        csv = vdir / "run_summary.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if df.empty:
            continue
        agg = aggregate(df)
        agg["variant"] = vdir.name
        parts.append(agg)
    if not parts:
        return pd.DataFrame()
    all_rows = pd.concat(parts, ignore_index=True)

    def _wmean(s, w):
        w = np.asarray(w, dtype=float)
        s = np.asarray(s, dtype=float)
        ws = w.sum()
        return float((s * w).sum() / ws) if ws > 0 else float("nan")

    out = []
    for (b, g), grp in all_rows.groupby(["beta", "gamma"]):
        out.append({
            "beta": float(b), "gamma": float(g),
            "n_seeds_total": int(grp["n_seeds"].sum()),
            "variants": ",".join(sorted(grp["variant"].unique())),
            "train_acc_mean": _wmean(grp["train_acc_mean"], grp["n_seeds"]),
            "long_acc_mean": _wmean(grp["long_acc_mean"], grp["n_seeds"]),
            "gap_mean": _wmean(grp["gap_mean"], grp["n_seeds"]),
            "retention_mean": _wmean(grp["retention_mean"], grp["n_seeds"]),
        })
    return pd.DataFrame(out)


def _is_transformer(name: str) -> bool:
    if name.startswith("mamba_"):
        return False
    if name.startswith("pilot_30k_"):
        return False
    return True


def _is_mamba(name: str) -> bool:
    return name.startswith("mamba_") and name not in MAMBA_EXCLUDE


def _plot_delta_panel(ax, joined: pd.DataFrame, metric: str, vabs: float, title: str):
    col_mb = f"{metric}_mean_mb"
    col_tx = f"{metric}_mean_tx"
    delta = joined[col_mb] - joined[col_tx]
    sizes = 80 + 40 * np.sqrt(joined["n_seeds_total_mb"].clip(lower=1))
    sc = ax.scatter(
        joined["beta"], joined["gamma"],
        c=delta, cmap="RdBu", vmin=-vabs, vmax=vabs,
        s=sizes, edgecolors="black", linewidth=0.6, alpha=0.95, zorder=3,
    )
    for _, r in joined.iterrows():
        d = r[col_mb] - r[col_tx]
        ax.annotate(
            f"({r['beta']:g}, {r['gamma']:g})\nΔ={d:+.3f}",
            (r["beta"], r["gamma"]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=7.5, color="black",
        )
    ax.set_xscale("log")
    ax.set_xlabel("β  (long-range decay sharpness)")
    ax.set_ylabel("γ  (noise fraction)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.4)
    return sc


def main() -> None:
    _ = load_config(sys.argv[1:])  # honor configs/default.yaml convention even if unused
    runs_dir = Path(__file__).resolve().parent / "runs"
    out_path = Path(__file__).resolve().parent / "figures" / "phase_delta_mamba_minus_transformer.png"

    tx = _collect(runs_dir, _is_transformer)
    mb = _collect(runs_dir, _is_mamba)
    if tx.empty or mb.empty:
        print(f"[warn] tx={len(tx)} mb={len(mb)} — not enough data for delta plot")
        return

    joined = mb.merge(tx, on=["beta", "gamma"], suffixes=("_mb", "_tx"))
    if joined.empty:
        print("[warn] no overlapping (β, γ) cells between Mamba and Transformer")
        return
    print(f"[info] tx={len(tx)} cells, mb={len(mb)} cells, overlap={len(joined)}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=True)

    sc_a = _plot_delta_panel(
        axes[0], joined, "train_acc",
        vabs=0.20,
        title=f"Δ train_acc(L=512)  —  {len(joined)} matched cells",
    )
    plt.colorbar(sc_a, ax=axes[0], label="Mamba − Transformer", shrink=0.82)

    sc_b = _plot_delta_panel(
        axes[1], joined, "long_acc",
        vabs=0.20,
        title="Δ long_acc(L=2048)  —  length generalization",
    )
    plt.colorbar(sc_b, ax=axes[1], label="Mamba − Transformer", shrink=0.82)

    sc_c = _plot_delta_panel(
        axes[2], joined, "retention",
        vabs=0.60,
        title="Δ retention = long_acc / train_acc",
    )
    plt.colorbar(sc_c, ax=axes[2], label="Mamba − Transformer", shrink=0.82)

    fig.suptitle(
        "Δ phase diagram — Mamba minus Transformer at matched (β, γ) cells\n"
        "blue = Mamba better,  red = Transformer better,  white ≈ tied",
        y=1.00, fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"  wrote {out_path}")
    show = joined.rename(columns={
        "n_seeds_total_mb": "n_mb", "n_seeds_total_tx": "n_tx",
        "train_acc_mean_mb": "tr_mb", "train_acc_mean_tx": "tr_tx",
        "long_acc_mean_mb": "lg_mb", "long_acc_mean_tx": "lg_tx",
        "retention_mean_mb": "rt_mb", "retention_mean_tx": "rt_tx",
    })
    cols = ["beta", "gamma", "n_mb", "n_tx",
            "tr_mb", "tr_tx", "lg_mb", "lg_tx", "rt_mb", "rt_tx"]
    print(show[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
