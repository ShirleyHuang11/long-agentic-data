#!/usr/bin/env python3
"""Scatter all (β, γ) cells from every runs/*/run_summary.csv on one plot.

`plot_phase_diagram.py` produces a heatmap on a regular β×γ grid and is
the natural visualization for a single variant's structured sweep
(`standard`, `refine`, etc.). It cannot draw the union across variants,
because corners + alpha_iso + refine + …  taken together do not form a
grid.

This script does that union view. It re-uses ``plot_phase_diagram.aggregate``
and ``utils.classify_fixed`` so the cell-aggregation and threshold rules
are identical to the per-variant figures (no second source of truth).

Output:
  case/phase/figures/cross_variant_scatter.png
  case/phase/figures/cross_variant_summary.csv

CLI: standard OmegaConf dotlist via configs/default.yaml.
Example::

    python case/phase/cross_variant_scatter.py
    python case/phase/cross_variant_scatter.py plot.thresholds.chaos_train=0.10
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_phase_diagram import aggregate
from utils import PHASE_COLORS, PHASE_NAMES, classify_fixed, load_config


def _phase_code(row, t) -> int:
    return classify_fixed(
        row,
        chaos_train=float(t.chaos_train),
        chaos_long=float(t.chaos_long),
        rote_train=float(t.rote_train),
        rote_gap=float(t.rote_gap),
        super_long=float(t.super_long),
        super_ret=float(t.super_ret),
    )


def collect_all_cells(runs_dir: Path) -> pd.DataFrame:
    """Aggregate every variant's run_summary.csv into one (β, γ) DataFrame.

    Cells appearing in multiple variants are folded into a single row by
    re-grouping on (β, γ) and averaging across seeds — assumes the (β, γ)
    pair has identical semantics regardless of which variant produced it.
    Carries a `variants` column that lists which variants contributed.
    """
    parts: List[pd.DataFrame] = []
    for vdir in sorted(runs_dir.iterdir()):
        # Skip cells from non-default training regimes — mixing them in
        # would average across heterogeneous configs (different
        # architecture or different train_steps).
        if vdir.name.startswith("mamba_"):
            continue
        if vdir.name.startswith("pilot_30k_"):  # 30k-step Transformer pilot
            continue
        csv = vdir / "run_summary.csv"
        if not csv.exists() or not vdir.is_dir():
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

    # Re-aggregate (β, γ) across variants — average means weighted by n_seeds.
    def _wmean(s, w):
        w = np.asarray(w, dtype=float)
        s = np.asarray(s, dtype=float)
        wsum = w.sum()
        return float((s * w).sum() / wsum) if wsum > 0 else float("nan")

    out_rows = []
    for (b, g), grp in all_rows.groupby(["beta", "gamma"]):
        out_rows.append({
            "beta": float(b),
            "gamma": float(g),
            "n_seeds_total": int(grp["n_seeds"].sum()),
            "n_variants": int(len(grp)),
            "variants": ",".join(sorted(grp["variant"].unique())),
            "train_acc_mean": _wmean(grp["train_acc_mean"], grp["n_seeds"]),
            "long_acc_mean": _wmean(grp["long_acc_mean"], grp["n_seeds"]),
            "gap_mean": _wmean(grp["gap_mean"], grp["n_seeds"]),
            "retention_mean": _wmean(grp["retention_mean"], grp["n_seeds"]),
        })
    out = pd.DataFrame(out_rows)
    out["alpha_theory"] = out["gamma"] / (2.0 * out["beta"].clip(lower=1e-12))
    return out


def overlay_alpha_lines(ax, betas, gammas, alphas=(0.1, 0.4, 1.0)) -> None:
    if len(betas) < 2 or len(gammas) < 2:
        return
    bgrid = np.geomspace(min(betas), max(betas), 100)
    for a in alphas:
        gline = 2.0 * a * bgrid
        m = (gline >= min(gammas)) & (gline <= max(gammas))
        if m.any():
            ax.plot(bgrid[m], gline[m], color="black", linestyle=":",
                    linewidth=0.7, alpha=0.5)
            # label at the right edge
            i = np.argmax(bgrid * m)
            if m[i]:
                ax.text(bgrid[i] * 1.02, gline[i],
                        f"α={a}", fontsize=7, color="gray",
                        verticalalignment="center")


def fit_emergent_boundary(cells: pd.DataFrame, threshold: float = 0.20):
    """Re-fit the 2D linear law on all emergent cells (weighted by √n_seeds).

    Uses every (β, γ) cell classified as emergent regardless of seed
    count. Per-cell weights w_i = √n_seeds_i give 3-seed cells √3× the
    influence of 1-seed cells in the OLS objective — proper handling of
    heteroscedastic sample sizes without discarding evidence.

    Returns (intercept_a, log_beta_coef_b, gamma_coef_c, n_used, R2)
    or (None, None, None, 0, None) if fewer than 3 emergent cells.

    Boundary line: train_acc = threshold ⇒
       γ*(β) = (a + b·log(β) − threshold) / (−c)
    """
    em = cells[cells["phase_code"] == 1]
    if len(em) < 3:
        return None, None, None, 0, None
    log_b = np.log(em["beta"].to_numpy())
    g = em["gamma"].to_numpy()
    y = em["train_acc_mean"].to_numpy()
    w = np.sqrt(em["n_seeds_total"].to_numpy(dtype=float))
    X = np.column_stack([np.ones(len(em)), log_b, g])
    # weighted least squares: sqrt(w) scaling
    Xw = X * w[:, None]
    yw = y * w
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    pred = X @ coef
    # weighted R² (accounting for sample sizes)
    ss_res = ((w * (y - pred)) ** 2).sum()
    y_wmean = (w * y).sum() / w.sum()
    ss_tot = ((w * (y - y_wmean)) ** 2).sum()
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    return float(coef[0]), float(coef[1]), float(coef[2]), int(len(em)), float(r2)


def overlay_boundary(ax, cells: pd.DataFrame, threshold: float = 0.20):
    """Overlay the predicted chaos→emergent boundary γ*(β) on the scatter.

    Only draws when ≥3 emergent cells are available to fit the 2D
    linear law (weighted by √n_seeds). Returns (a, b, c, n) used so
    the plotter can annotate.
    """
    a, b, c, n, r2 = fit_emergent_boundary(cells, threshold)
    if a is None:
        return None
    if c >= 0:  # boundary slope must be negative for line to make sense
        return None
    bgrid = np.geomspace(cells["beta"].min(), cells["beta"].max(), 200)
    gstar = (a + b * np.log(bgrid) - threshold) / (-c)
    m = (gstar >= cells["gamma"].min()) & (gstar <= cells["gamma"].max())
    if not m.any():
        return None
    ax.plot(bgrid[m], gstar[m], color="#cc4422", linestyle="--",
            linewidth=1.6, alpha=0.85, zorder=4,
            label=f"γ*(β) fit: train_acc={threshold:.2f}  (n_em={n}, R²={r2:.3f})")
    return (a, b, c, n)


def plot_scatter(cells: pd.DataFrame, thresholds, out_path: Path) -> None:
    cells = cells.copy()
    cells["phase_code"] = cells.apply(lambda r: _phase_code(r, thresholds), axis=1)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for code in sorted(PHASE_NAMES):
        sub = cells[cells["phase_code"] == code]
        if sub.empty:
            continue
        sizes = 30 + 25 * np.sqrt(sub["n_seeds_total"].clip(lower=1))
        ax.scatter(sub["beta"], sub["gamma"],
                   s=sizes, c=PHASE_COLORS[code],
                   edgecolors="black", linewidth=0.5,
                   label=f"{PHASE_NAMES[code]} ({len(sub)})",
                   alpha=0.85, zorder=3)

    overlay_alpha_lines(ax, cells["beta"], cells["gamma"])
    overlay_boundary(ax, cells, threshold=float(thresholds.chaos_train))
    ax.set_xscale("log")
    ax.set_xlabel("β  (long-range decay sharpness)")
    ax.set_ylabel("γ  (noise fraction)")
    ax.set_title(
        f"Cross-variant phase scatter — {len(cells)} unique cells, "
        f"{cells['n_seeds_total'].sum()} seeds total"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    cfg = load_config(sys.argv[1:])
    runs_dir = Path(__file__).resolve().parent / "runs"
    out_dir = Path(__file__).resolve().parent / "figures"

    cells = collect_all_cells(runs_dir)
    if cells.empty:
        print(f"[warn] no run_summary.csv files under {runs_dir}")
        return

    csv_path = out_dir / "cross_variant_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    cells.sort_values(["beta", "gamma"]).to_csv(csv_path, index=False)

    png_path = out_dir / "cross_variant_scatter.png"
    plot_scatter(cells, cfg.plot.thresholds, png_path)

    print(f"[done] {len(cells)} unique (β, γ) cells across "
          f"{cells['n_variants'].max()} variant(s)")
    print(f"  {csv_path}")
    print(f"  {png_path}")


if __name__ == "__main__":
    main()
