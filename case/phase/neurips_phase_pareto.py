#!/usr/bin/env python3
"""Compute Pareto frontier over aggregated beta-gamma cells."""

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pareto analysis for phase sweep results")
    ap.add_argument("--in-summary", type=str, required=True, help="Path to run_summary.csv")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory")
    ap.add_argument(
        "--aulc-col",
        type=str,
        default="aulc_train_to_final_norm",
        help="AULC column in run_summary.csv (fallbacks to first aulc_* if missing)",
    )
    return ap.parse_args()


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = float(np.mean(x))
    s = float(np.std(x))
    if s < EPS:
        return np.zeros_like(x)
    return (x - m) / s


def pareto_front_mask(values_max: np.ndarray) -> np.ndarray:
    """True for non-dominated points for maximization objectives."""
    n = values_max.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            ge_all = np.all(values_max[j] >= values_max[i])
            gt_any = np.any(values_max[j] > values_max[i])
            if ge_all and gt_any:
                is_pareto[i] = False
                break
    return is_pareto


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_summary)
    if df.empty:
        raise ValueError("Input summary CSV is empty")

    aulc_col = args.aulc_col
    if aulc_col not in df.columns:
        cands = [c for c in df.columns if c.startswith("aulc_")]
        if not cands:
            raise ValueError("No AULC column found in input summary")
        aulc_col = sorted(cands)[0]

    required = ["beta", "gamma", "long_acc", "generalization_gap", "final_train_step_loss", aulc_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg = (
        df.groupby(["beta", "gamma"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            long_acc_mean=("long_acc", "mean"),
            gap_mean=("generalization_gap", "mean"),
            final_loss_mean=("final_train_step_loss", "mean"),
            aulc_mean=(aulc_col, "mean"),
            long_acc_std=("long_acc", "std"),
            aulc_std=(aulc_col, "std"),
        )
        .sort_values(["gamma", "beta"])
        .reset_index(drop=True)
    )
    agg["long_acc_std"] = agg["long_acc_std"].fillna(0.0)
    agg["aulc_std"] = agg["aulc_std"].fillna(0.0)
    agg["long_acc_ci95"] = 1.96 * agg["long_acc_std"] / np.sqrt(np.clip(agg["n_runs"], 1, None))
    agg["aulc_ci95"] = 1.96 * agg["aulc_std"] / np.sqrt(np.clip(agg["n_runs"], 1, None))

    # Objectives (all maximization):
    # 1) long_acc_mean  (higher better)
    # 2) aulc_mean      (higher better)
    # 3) -final_loss    (lower loss better)
    # 4) -gap           (smaller gap better)
    vals = np.stack(
        [
            agg["long_acc_mean"].to_numpy(dtype=float),
            agg["aulc_mean"].to_numpy(dtype=float),
            -agg["final_loss_mean"].to_numpy(dtype=float),
            -agg["gap_mean"].to_numpy(dtype=float),
        ],
        axis=1,
    )
    is_pareto = pareto_front_mask(vals)
    agg["is_pareto"] = is_pareto.astype(int)

    # Composite score (for ranking, not replacing Pareto)
    score = (
        zscore(agg["long_acc_mean"].to_numpy(dtype=float))
        + zscore(agg["aulc_mean"].to_numpy(dtype=float))
        - zscore(agg["final_loss_mean"].to_numpy(dtype=float))
        - zscore(agg["gap_mean"].to_numpy(dtype=float))
    )
    agg["composite_score"] = score
    agg = agg.sort_values(["is_pareto", "composite_score"], ascending=[False, False]).reset_index(drop=True)

    pareto = agg[agg["is_pareto"] == 1].copy().reset_index(drop=True)

    agg_csv = out_dir / "pareto_cells_scored.csv"
    pareto_csv = out_dir / "pareto_front_cells.csv"
    report_md = out_dir / "pareto_report.md"
    scatter_png = out_dir / "pareto_longacc_vs_aulc.png"

    agg.to_csv(agg_csv, index=False)
    pareto.to_csv(pareto_csv, index=False)

    # Plot: long_acc vs aulc, color by final_loss
    fig, ax = plt.subplots(figsize=(7.8, 6.0), dpi=220)
    sc = ax.scatter(
        agg["long_acc_mean"],
        agg["aulc_mean"],
        c=agg["final_loss_mean"],
        cmap="viridis_r",
        s=45,
        alpha=0.75,
        edgecolors="none",
        label="all cells",
    )
    if len(pareto) > 0:
        ax.scatter(
            pareto["long_acc_mean"],
            pareto["aulc_mean"],
            s=85,
            facecolors="none",
            edgecolors="crimson",
            linewidths=1.3,
            label="Pareto front",
        )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("final_loss_mean (lower is better)")
    ax.set_xlabel("long_acc_mean")
    ax.set_ylabel(f"{aulc_col}_mean")
    ax.set_title("Pareto Frontier on Cell Means")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(scatter_png)
    plt.close(fig)

    with report_md.open("w", encoding="utf-8") as f:
        f.write("# Pareto Report\n\n")
        f.write("Objectives:\n")
        f.write("- maximize `long_acc_mean`\n")
        f.write(f"- maximize `{aulc_col}_mean`\n")
        f.write("- minimize `final_loss_mean`\n")
        f.write("- minimize `gap_mean`\n\n")
        f.write(f"- Cells total: {len(agg)}\n")
        f.write(f"- Pareto cells: {len(pareto)}\n\n")

        f.write("## Top Pareto Cells (by composite score)\n")
        top = pareto.sort_values("composite_score", ascending=False).head(15)
        if len(top) == 0:
            f.write("- None\n")
        else:
            for _, r in top.iterrows():
                f.write(
                    "- beta={:.6g}, gamma={:.6g}, long_acc={:.4f}, aulc={:.4f}, final_loss={:.4f}, gap={:.4f}, score={:.4f}\n".format(
                        float(r["beta"]),
                        float(r["gamma"]),
                        float(r["long_acc_mean"]),
                        float(r["aulc_mean"]),
                        float(r["final_loss_mean"]),
                        float(r["gap_mean"]),
                        float(r["composite_score"]),
                    )
                )

        f.write("\n## Artifacts\n")
        f.write("- pareto_cells_scored.csv\n")
        f.write("- pareto_front_cells.csv\n")
        f.write("- pareto_longacc_vs_aulc.png\n")

    print("[Done] Pareto artifacts:")
    print(f"  - {agg_csv}")
    print(f"  - {pareto_csv}")
    print(f"  - {scatter_png}")
    print(f"  - {report_md}")


if __name__ == "__main__":
    main()
