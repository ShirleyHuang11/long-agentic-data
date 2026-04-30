#!/usr/bin/env python3
"""Continuous phase diagram: 2-panel scatter colored by performance.

The existing cross_variant_scatter.py colors cells by discrete phase
classification (chaos/emergent). This figure shows the same (β, γ)
cells but colored by **continuous** train_acc and long_acc — the
underlying performance the classifier thresholds. Lets you see the
phase boundary as a smooth gradient rather than a hard cutoff.

Output: case/phase/figures/phase_continuous.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_variant_scatter import collect_all_cells, fit_emergent_boundary, _phase_code
from utils import load_config


def overlay_alpha_lines(ax, betas, gammas, alphas=(0.1, 0.4, 1.0)) -> None:
    if len(betas) < 2 or len(gammas) < 2:
        return
    bgrid = np.geomspace(min(betas), max(betas), 100)
    for a in alphas:
        gline = 2.0 * a * bgrid
        m = (gline >= min(gammas)) & (gline <= max(gammas))
        if m.any():
            ax.plot(bgrid[m], gline[m], color="black", linestyle=":",
                    linewidth=0.6, alpha=0.4)


def overlay_boundary_line(ax, cells: pd.DataFrame, threshold: float = 0.20):
    a, b, c, n, r2 = fit_emergent_boundary(cells, threshold)
    if a is None or c >= 0:
        return
    bgrid = np.geomspace(cells["beta"].min(), cells["beta"].max(), 200)
    gstar = (a + b * np.log(bgrid) - threshold) / (-c)
    m = (gstar >= cells["gamma"].min()) & (gstar <= cells["gamma"].max())
    if m.any():
        ax.plot(bgrid[m], gstar[m], color="black", linestyle="--",
                linewidth=1.4, alpha=0.85,
                label=f"γ*(β) at train_acc=0.20  (n_em={n}, R²={r2:.3f})")


def plot_panel(ax, cells: pd.DataFrame, metric: str, vmax: float, title: str) -> None:
    sizes = 28 + 20 * np.sqrt(cells["n_seeds_total"].clip(lower=1))
    sc = ax.scatter(
        cells["beta"], cells["gamma"],
        c=cells[metric], cmap="RdYlGn",
        vmin=0.0, vmax=vmax,
        s=sizes, edgecolors="black", linewidth=0.4, alpha=0.92, zorder=3,
    )
    overlay_alpha_lines(ax, cells["beta"], cells["gamma"])
    overlay_boundary_line(ax, cells)
    ax.set_xscale("log")
    ax.set_xlabel("β  (long-range decay sharpness)")
    ax.set_ylabel("γ  (noise fraction)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.4)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    return sc


def main() -> None:
    cfg = load_config(sys.argv[1:])
    runs_dir = Path(__file__).resolve().parent / "runs"
    out_path = Path(__file__).resolve().parent / "figures" / "phase_continuous.png"

    cells = collect_all_cells(runs_dir)
    if cells.empty:
        print(f"[warn] no data under {runs_dir}")
        return
    cells["phase_code"] = cells.apply(lambda r: _phase_code(r, cfg.plot.thresholds), axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=True)

    sc_a = plot_panel(axes[0], cells, "train_acc_mean", 0.40,
                      f"train_acc(L=512) — {len(cells)} cells, "
                      f"{cells['n_seeds_total'].sum()} seeds")
    plt.colorbar(sc_a, ax=axes[0], label="train_acc", shrink=0.82)

    sc_b = plot_panel(axes[1], cells, "long_acc_mean", 0.20,
                      f"long_acc(L=2048) — length-generalization performance")
    plt.colorbar(sc_b, ax=axes[1], label="long_acc", shrink=0.82)

    sc_c = plot_panel(axes[2], cells, "gap_mean", 0.25,
                      f"gap = train_acc − long_acc")
    plt.colorbar(sc_c, ax=axes[2], label="gap", shrink=0.82)

    fig.suptitle(
        "Continuous phase diagram — (β, γ) plane colored by performance metrics",
        y=1.00, fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {len(cells)} cells, train_acc range "
          f"[{cells['train_acc_mean'].min():.3f}, {cells['train_acc_mean'].max():.3f}]")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
