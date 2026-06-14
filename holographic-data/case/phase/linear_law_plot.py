#!/usr/bin/env python3
"""Paper figure 2: linear law for train_acc(γ) at fixed β.

Shows the iter-11 + iter-12 falsifiability story visually:

* The 4 N=3-seed cells at β=1.4 (γ ∈ {0.21, 0.255, 0.30, 0.345}) lie
  on a line `train_acc(γ) ≈ a − b·γ` with R² > 0.99.
* Linear extrapolation to γ=0.39 predicted train_acc=0.197.
* The held-out cell at γ=0.39 (3 seeds) gave train_acc≈0.193 — a
  generalisation error of 0.004 across the strip→chaos boundary.

Saves:  case/phase/figures/linear_law_b1p4.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_b1p4_cells(runs_dir: Path) -> pd.DataFrame:
    """Per-seed rows of refine_b2p0_g0p3 at β=1.4."""
    csv = runs_dir / "refine_b2p0_g0p3" / "run_summary.csv"
    if not csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv)
    return df[(df["beta"] - 1.4).abs() < 1e-3].copy()


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """OLS y = a + b*x. Returns (a, b, R²)."""
    n = len(x)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    x_mean, y_mean = x.mean(), y.mean()
    b = ((x - x_mean) * (y - y_mean)).sum() / max(((x - x_mean) ** 2).sum(), 1e-12)
    a = y_mean - b * x_mean
    pred = a + b * x
    r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y_mean) ** 2).sum(), 1e-12)
    return float(a), float(b), float(r2)


def plot_linear_law(out_path: Path, runs_dir: Path) -> None:
    df = load_b1p4_cells(runs_dir)
    if df.empty:
        print(f"[skip] no β=1.4 data under {runs_dir}/refine_b2p0_g0p3")
        return

    # Per-seed scatter; cell-mean overlay
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Fit on N=3 cells only (γ ≤ 0.345 was the in-strip range)
    in_strip = df[df["gamma"] <= 0.345 + 1e-6]
    fit_gammas = in_strip["gamma"].to_numpy()
    fit_acc = in_strip["train_acc"].to_numpy()
    a, b, r2 = fit_line(fit_gammas, fit_acc)

    ax.scatter(in_strip["gamma"], in_strip["train_acc"],
               s=22, c="#2c7fb8", alpha=0.6, edgecolors="none",
               label=f"in-strip seeds (n={len(in_strip)}, β=1.4)", zorder=2)

    held_out = df[(df["gamma"] - 0.39).abs() < 1e-3]
    if not held_out.empty:
        ax.scatter(held_out["gamma"], held_out["train_acc"],
                   s=42, c="#cc4422", marker="x", linewidth=1.8,
                   label=f"held-out γ=0.39 seeds (n={len(held_out)})", zorder=3)

    # Cell means
    cell_means = df.groupby("gamma", as_index=False)["train_acc"].mean()
    ax.plot(cell_means["gamma"], cell_means["train_acc"],
            color="#2c7fb8", marker="o", linewidth=0, markersize=8,
            markerfacecolor="white", markeredgewidth=1.5,
            label="cell means", zorder=4)

    # Regression line + extrapolation past fit range
    g_fit = np.linspace(fit_gammas.min() - 0.02,
                        max(fit_gammas.max(), df["gamma"].max()) + 0.02, 100)
    pred = a + b * g_fit
    in_fit_range = g_fit <= fit_gammas.max() + 1e-6
    ax.plot(g_fit[in_fit_range], pred[in_fit_range],
            color="#2c7fb8", linewidth=1.4, alpha=0.8,
            label=f"OLS fit: train_acc = {a:.3f} − {-b:.3f}·γ  (R²={r2:.3f})",
            zorder=2)
    ax.plot(g_fit[~in_fit_range], pred[~in_fit_range],
            color="#2c7fb8", linewidth=1.4, alpha=0.8, linestyle="--",
            label="extrapolation (no fit data)", zorder=2)

    # Chaos threshold line
    ax.axhline(0.20, color="#666666", linewidth=0.8, linestyle=":",
               alpha=0.6, zorder=1)
    ax.text(df["gamma"].min(), 0.205, "chaos threshold = 0.20",
            fontsize=8, color="#666666")

    # Annotate held-out prediction
    if not held_out.empty:
        gamma_held = float(held_out["gamma"].iloc[0])
        pred_held = a + b * gamma_held
        obs_held = float(held_out["train_acc"].mean())
        err = obs_held - pred_held
        ax.annotate(
            f"predicted {pred_held:.3f}\nobserved  {obs_held:.3f}\nerror     {err:+.3f}",
            xy=(gamma_held, obs_held),
            xytext=(gamma_held + 0.005, obs_held - 0.02),
            fontsize=8, color="#cc4422",
            arrowprops=dict(arrowstyle="-", color="#cc4422",
                           linewidth=0.6, alpha=0.6),
        )

    ax.set_xlabel("γ  (noise rate)")
    ax.set_ylabel("train_acc (eval at L=512)")
    ax.set_title(
        "Linear law for train_acc(γ) at β=1.4\n"
        "fit interpolates strip; extrapolation predicts chaos onset"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[done] fit on {len(in_strip)} in-strip seeds:")
    print(f"  train_acc(γ | β=1.4) = {a:.4f} + ({b:+.4f})·γ   R²={r2:.4f}")
    if not held_out.empty:
        print(f"  held-out γ=0.39 (n={len(held_out)}): "
              f"predicted {a + b * 0.39:.4f}, observed {held_out['train_acc'].mean():.4f}")
    print(f"  wrote {out_path}")


def main() -> None:
    runs_dir = Path(__file__).resolve().parent / "runs"
    out_path = Path(__file__).resolve().parent / "figures" / "linear_law_b1p4.png"
    plot_linear_law(out_path, runs_dir)


if __name__ == "__main__":
    main()
