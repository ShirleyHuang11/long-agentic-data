#!/usr/bin/env python3
"""Aggregate and analyze large phase sweeps for paper-quality reporting."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

EPS = 1e-8


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze sweep results and create paper figures")
    ap.add_argument("--in-summary", type=str, required=True, help="Path to run_summary.csv")
    ap.add_argument("--out-dir", type=str, default="plots/phase_neurips_analysis")
    ap.add_argument("--phase-mode", type=str, default="fixed", choices=["fixed", "quantile"])

    # fixed thresholds
    ap.add_argument("--chaos-train-thresh", type=float, default=0.15)
    ap.add_argument("--chaos-long-thresh", type=float, default=0.10)
    ap.add_argument("--rote-train-thresh", type=float, default=0.18)
    ap.add_argument("--rote-gap-thresh", type=float, default=0.05)
    ap.add_argument("--super-long-thresh", type=float, default=0.15)
    ap.add_argument("--super-ret-thresh", type=float, default=0.78)

    ap.add_argument("--permutation-trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    # near-origin constrained limit analysis:
    # beta, gamma -> 0 and beta decays faster than gamma  <=> beta/gamma -> 0
    ap.add_argument("--limit-beta-max", type=float, default=0.25)
    ap.add_argument("--limit-gamma-max", type=float, default=0.25)
    ap.add_argument("--limit-beta-over-gamma-max", type=float, default=0.60)
    return ap.parse_args()


def phase_name(code: int) -> str:
    return {
        0: "Chaos/Saturation",
        1: "Emergent",
        2: "Super-Generalization",
        3: "Rote Memorization",
    }[code]


def classify_fixed(row: pd.Series, args: argparse.Namespace) -> int:
    if row["train_acc_mean"] < args.chaos_train_thresh and row["long_acc_mean"] < args.chaos_long_thresh:
        return 0
    if row["train_acc_mean"] >= args.rote_train_thresh and row["gap_mean"] >= args.rote_gap_thresh:
        return 3
    if row["long_acc_mean"] >= args.super_long_thresh and row["retention_mean"] >= args.super_ret_thresh:
        return 2
    return 1


def classify_quantile(df: pd.DataFrame) -> List[int]:
    q_train_low = df["train_acc_mean"].quantile(0.25)
    q_train_high = df["train_acc_mean"].quantile(0.70)
    q_long_med = df["long_acc_mean"].quantile(0.50)
    q_long_high = df["long_acc_mean"].quantile(0.75)
    q_gap_med = df["gap_mean"].quantile(0.50)
    q_gap_high = df["gap_mean"].quantile(0.75)
    q_ret_med = df["retention_mean"].quantile(0.50)

    codes = []
    for _, r in df.iterrows():
        if r["train_acc_mean"] <= q_train_low and r["long_acc_mean"] <= q_long_med:
            codes.append(0)
        elif r["train_acc_mean"] >= q_train_high and r["gap_mean"] >= q_gap_high:
            codes.append(3)
        elif r["long_acc_mean"] >= q_long_high and r["gap_mean"] <= q_gap_med and r["retention_mean"] >= q_ret_med:
            codes.append(2)
        else:
            codes.append(1)
    return codes


def ols_fit(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    n, p = X.shape
    dof = max(n - p, 1)
    s2 = float((resid @ resid) / dof)
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * s2, EPS, None))

    sst = float(((y - y.mean()) ** 2).sum())
    ssr = float((resid ** 2).sum())
    r2 = 1.0 - (ssr / max(sst, EPS))

    return {
        "beta": beta,
        "se": se,
        "r2": np.array([r2]),
        "y_hat": y_hat,
        "resid": resid,
    }


def permutation_pvalue_for_term(
    y: np.ndarray,
    X_full: np.ndarray,
    X_reduced: np.ndarray,
    term_idx: int,
    trials: int,
    rng: np.random.Generator,
) -> float:
    fit_full = ols_fit(X_full, y)
    fit_red = ols_fit(X_reduced, y)
    obs_delta_r2 = float(fit_full["r2"][0] - fit_red["r2"][0])

    count = 0
    for _ in range(trials):
        yp = rng.permutation(y)
        delta = float(ols_fit(X_full, yp)["r2"][0] - ols_fit(X_reduced, yp)["r2"][0])
        if delta >= obs_delta_r2:
            count += 1

    return (count + 1.0) / (trials + 1.0)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if np.isnan(c) else c


def build_heatmap_matrix(
    df: pd.DataFrame, betas: List[float], gammas: List[float], key: str, fill_value: float = np.nan
) -> np.ndarray:
    z = np.full((len(gammas), len(betas)), fill_value=fill_value, dtype=float)
    for _, r in df.iterrows():
        i = gammas.index(float(r["gamma"]))
        j = betas.index(float(r["beta"]))
        z[i, j] = float(r[key])
    return z


def save_heatmap(z: np.ndarray, betas: List[float], gammas: List[float], title: str, cbar: str, path: Path, cmap: str) -> None:
    bb, gg = np.meshgrid(np.array(betas), np.array(gammas))
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=200)
    im = ax.pcolormesh(bb, gg, z, shading="auto", cmap=cmap)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar)
    ax.set_title(title)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")
    if len(betas) > 1:
        ax.set_xlim(min(betas), max(betas))
    if len(gammas) > 1:
        ax.set_ylim(min(gammas), max(gammas))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_phase_diagram(df: pd.DataFrame, betas: List[float], gammas: List[float], path: Path) -> None:
    z = build_heatmap_matrix(df, betas, gammas, "phase_code", fill_value=0.0)
    alpha = build_heatmap_matrix(df, betas, gammas, "alpha_theory")
    bb, gg = np.meshgrid(np.array(betas), np.array(gammas))

    bx0, bx1 = float(min(betas)), float(max(betas))
    gy0, gy1 = float(min(gammas)), float(max(gammas))
    if len(betas) == 1:
        pad = max(1e-6, abs(bx0) * 0.1)
        bx0, bx1 = bx0 - pad, bx1 + pad
    if len(gammas) == 1:
        pad = max(1e-6, abs(gy0) * 0.1)
        gy0, gy1 = gy0 - pad, gy1 + pad

    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    cmap = ListedColormap(["#9aa0a6", "#9ddf9b", "#f4c76b", "#ef6b6b"])
    ax.imshow(
        z,
        origin="lower",
        extent=(bx0, bx1, gy0, gy1),
        aspect="auto",
        cmap=cmap,
        alpha=0.78,
    )

    if len(betas) >= 2 and len(gammas) >= 2 and np.isfinite(alpha).sum() >= 4:
        cs = ax.contour(bb, gg, alpha, levels=[0.1, 0.2, 0.4, 0.8, 1.2], colors="black", linewidths=1.0)
        ax.clabel(cs, inline=True, fontsize=8, fmt=lambda v: rf"$\alpha={v:.2f}$")

    for _, r in df.iterrows():
        ax.text(float(r["beta"]), float(r["gamma"]), int(r["phase_code"]), ha="center", va="center", fontsize=8)

    ax.set_title("Aggregated Phase Diagram (across seeds)")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_alpha_scatter(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=200)
    x = df["alpha_theory"].to_numpy(dtype=float)
    y = df["long_acc_mean"].to_numpy(dtype=float)
    yerr = df["long_acc_ci95"].to_numpy(dtype=float)
    ax.errorbar(x, y, yerr=yerr, fmt="o", alpha=0.85, capsize=3)

    # linear trend
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    xx = np.linspace(float(x.min()), float(x.max()), 100)
    yy = coef[0] * xx + coef[1]
    ax.plot(xx, yy, "--", linewidth=1.5)

    corr = safe_corr(x, y)
    ax.set_title(f"Alpha-Theory vs Long Accuracy (corr={corr:.3f})")
    ax.set_xlabel(r"$\alpha_{theory}=\gamma/(2\beta)$")
    ax.set_ylabel("mean long_acc")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_summary)
    if df.empty:
        raise ValueError("Input summary CSV is empty")

    for col in [
        "beta",
        "gamma",
        "seed",
        "alpha_theory",
        "train_acc",
        "long_acc",
        "generalization_gap",
        "retention_ratio",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    aulc_cols = [c for c in df.columns if c.startswith("aulc_")]
    aulc_cols = sorted(aulc_cols)
    for c in aulc_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    preferred_aulc = "aulc_train_to_final_norm" if "aulc_train_to_final_norm" in aulc_cols else (aulc_cols[0] if aulc_cols else "")

    renyi_cols = [c for c in df.columns if c.startswith("renyi_D_rate")]
    renyi_cols = sorted(renyi_cols)
    for c in renyi_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # aggregated cell stats
    agg_spec = {
        "n_runs": ("seed", "count"),
        "train_acc_mean": ("train_acc", "mean"),
        "train_acc_std": ("train_acc", "std"),
        "long_acc_mean": ("long_acc", "mean"),
        "long_acc_std": ("long_acc", "std"),
        "gap_mean": ("generalization_gap", "mean"),
        "gap_std": ("generalization_gap", "std"),
        "retention_mean": ("retention_ratio", "mean"),
        "retention_std": ("retention_ratio", "std"),
        "time_mean": ("train_time_sec", "mean"),
    }
    for c in aulc_cols:
        agg_spec[f"{c}_mean"] = (c, "mean")
        agg_spec[f"{c}_std"] = (c, "std")
    for c in renyi_cols:
        agg_spec[f"{c}_mean"] = (c, "mean")
        agg_spec[f"{c}_std"] = (c, "std")

    agg = (
        df.groupby(["beta", "gamma", "alpha_theory"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["gamma", "beta"])
        .reset_index(drop=True)
    )

    for std_col in [c for c in agg.columns if c.endswith("_std")]:
        base = std_col[:-4]
        sem_col = f"{base}_sem"
        ci_col = f"{base}_ci95"
        agg[std_col] = agg[std_col].fillna(0.0)
        agg[sem_col] = agg[std_col] / np.sqrt(np.clip(agg["n_runs"], 1, None))
        agg[ci_col] = 1.96 * agg[sem_col]

    # phase assignment on aggregated means
    if args.phase_mode == "fixed":
        agg["phase_code"] = agg.apply(lambda r: classify_fixed(r, args), axis=1)
    else:
        agg["phase_code"] = classify_quantile(agg)
    agg["phase_name"] = agg["phase_code"].apply(phase_name)

    # regression with interaction on run-level data
    y = df["long_acc"].to_numpy(dtype=float)
    x1 = np.log(np.clip(df["beta"].to_numpy(dtype=float), EPS, None))
    x2 = df["gamma"].to_numpy(dtype=float)
    x3 = x1 * x2

    X_full = np.stack([np.ones_like(x1), x1, x2, x3], axis=1)
    fit = ols_fit(X_full, y)

    coef_names = ["intercept", "log_beta", "gamma", "log_beta_x_gamma"]
    coef_df = pd.DataFrame(
        {
            "term": coef_names,
            "coef": fit["beta"],
            "se": fit["se"],
        }
    )

    # permutation p-values for each non-intercept term
    rng = np.random.default_rng(args.seed)
    pvals = []
    # reduced models by removing each term
    for i, term in enumerate(coef_names):
        if i == 0:
            pvals.append(np.nan)
            continue
        keep = [0, 1, 2, 3]
        keep.remove(i)
        X_red = X_full[:, keep]
        p = permutation_pvalue_for_term(
            y=y,
            X_full=X_full,
            X_reduced=X_red,
            term_idx=i,
            trials=args.permutation_trials,
            rng=rng,
        )
        pvals.append(p)
    coef_df["perm_pvalue"] = pvals
    model_r2 = float(fit["r2"][0])

    # overall correlations (run-level and cell-level)
    run_corr = safe_corr(df["alpha_theory"].to_numpy(dtype=float), df["long_acc"].to_numpy(dtype=float))
    cell_corr = safe_corr(agg["alpha_theory"].to_numpy(dtype=float), agg["long_acc_mean"].to_numpy(dtype=float))

    # save tables
    agg_csv = out_dir / "aggregated_cell_stats.csv"
    coef_csv = out_dir / "regression_coefficients.csv"
    phase_counts_csv = out_dir / "phase_counts.csv"
    report_md = out_dir / "analysis_report.md"

    agg.to_csv(agg_csv, index=False)
    coef_df.to_csv(coef_csv, index=False)

    phase_counts = (
        agg.groupby("phase_name", as_index=False)
        .size()
        .rename(columns={"size": "num_cells"})
        .sort_values("phase_name")
    )
    phase_counts.to_csv(phase_counts_csv, index=False)

    # plots
    betas = sorted([float(x) for x in agg["beta"].unique()])
    gammas = sorted([float(x) for x in agg["gamma"].unique()])

    save_heatmap(
        build_heatmap_matrix(agg, betas, gammas, "long_acc_mean"),
        betas,
        gammas,
        title="Mean Long-Context Accuracy",
        cbar="mean long_acc",
        path=out_dir / "heatmap_long_acc_mean.png",
        cmap="viridis",
    )

    save_heatmap(
        build_heatmap_matrix(agg, betas, gammas, "long_acc_ci95"),
        betas,
        gammas,
        title="95% CI Width of Long-Context Accuracy",
        cbar="CI95 width",
        path=out_dir / "heatmap_long_acc_ci95.png",
        cmap="magma",
    )

    save_heatmap(
        build_heatmap_matrix(agg, betas, gammas, "gap_mean"),
        betas,
        gammas,
        title="Mean Generalization Gap (train - long)",
        cbar="gap mean",
        path=out_dir / "heatmap_gap_mean.png",
        cmap="plasma",
    )

    for c in aulc_cols:
        cm = f"{c}_mean"
        if cm not in agg.columns:
            continue
        save_heatmap(
            build_heatmap_matrix(agg, betas, gammas, cm),
            betas,
            gammas,
            title=f"Mean {cm}",
            cbar=cm,
            path=out_dir / f"heatmap_{cm}.png",
            cmap="viridis_r",
        )

    # Renyi dimension heatmaps (if available)
    for c in renyi_cols:
        cm = f"{c}_mean"
        if cm not in agg.columns:
            continue
        save_heatmap(
            build_heatmap_matrix(agg, betas, gammas, cm),
            betas,
            gammas,
            title=f"Mean {c}",
            cbar=cm,
            path=out_dir / f"heatmap_{cm}.png",
            cmap="cividis",
        )

    save_phase_diagram(agg, betas, gammas, out_dir / "phase_diagram_aggregated.png")
    save_alpha_scatter(agg, out_dir / "scatter_alpha_vs_long_acc.png")

    # report
    with report_md.open("w", encoding="utf-8") as f:
        f.write("# NeurIPS-Scale Sweep Analysis\n\n")
        f.write("## Data Coverage\n")
        f.write(f"- Runs: {len(df)}\n")
        f.write(f"- Cells: {len(agg)}\n")
        f.write(f"- Betas: {len(betas)} ({betas})\n")
        f.write(f"- Gammas: {len(gammas)} ({gammas})\n")
        f.write(f"- Min runs/cell: {int(agg['n_runs'].min())}\n")
        f.write(f"- Max runs/cell: {int(agg['n_runs'].max())}\n\n")

        f.write("## Reliability\n")
        f.write(f"- Mean CI95(long_acc): {float(agg['long_acc_ci95'].mean()):.6f}\n")
        f.write(f"- Median CI95(long_acc): {float(agg['long_acc_ci95'].median()):.6f}\n")
        f.write(f"- Max CI95(long_acc): {float(agg['long_acc_ci95'].max()):.6f}\n\n")

        f.write("## Correlations\n")
        f.write(f"- Corr(alpha_theory, long_acc) run-level: {run_corr:.4f}\n")
        f.write(f"- Corr(alpha_theory, long_acc_mean) cell-level: {cell_corr:.4f}\n\n")

        if renyi_cols:
            f.write("## Renyi Dimensions\n")
            for c in renyi_cols:
                if c in df.columns:
                    rc = safe_corr(df[c].to_numpy(dtype=float), df["long_acc"].to_numpy(dtype=float))
                    cm = f"{c}_mean"
                    rc_cell = safe_corr(agg[cm].to_numpy(dtype=float), agg["long_acc_mean"].to_numpy(dtype=float))
                    f.write(f"- Corr({c}, long_acc) run-level: {rc:.4f}\n")
                    f.write(f"- Corr({cm}, long_acc_mean) cell-level: {rc_cell:.4f}\n")
            f.write("\n")

        if aulc_cols:
            f.write("## AULC Metrics\n")
            f.write("- Definition: area between training loss curve and horizontal line y=final_train_step_loss.\n")
            f.write("- Primary column: `aulc_train_to_final_norm` (area normalized by number of steps).\n")
            for c in aulc_cols:
                rc = safe_corr(df[c].to_numpy(dtype=float), df["long_acc"].to_numpy(dtype=float))
                cm = f"{c}_mean"
                if cm in agg.columns:
                    rc_cell = safe_corr(agg[cm].to_numpy(dtype=float), agg["long_acc_mean"].to_numpy(dtype=float))
                    f.write(f"- Corr({c}, long_acc) run-level: {rc:.4f}\n")
                    f.write(f"- Corr({cm}, long_acc_mean) cell-level: {rc_cell:.4f}\n")
            f.write("\n")

        # limit regime: beta,gamma -> 0 with beta/gamma small (beta decays faster)
        eps = 1e-12
        df["beta_over_gamma"] = df["beta"] / np.clip(df["gamma"], eps, None)
        limit_mask = (
            (df["beta"] <= args.limit_beta_max)
            & (df["gamma"] <= args.limit_gamma_max)
            & (df["beta_over_gamma"] <= args.limit_beta_over_gamma_max)
        )
        limit_df = df[limit_mask].copy()
        limit_runs_csv = out_dir / "limit_subset_runs.csv"
        limit_cell_csv = out_dir / "limit_subset_cell_stats.csv"
        limit_plot = out_dir / "limit_subset_gamma_trend.png"
        if len(limit_df) > 0:
            limit_df.to_csv(limit_runs_csv, index=False)
            limit_agg = (
                limit_df.groupby(["beta", "gamma"], as_index=False)
                .agg(
                    n_runs=("seed", "count"),
                    alpha_theory=("alpha_theory", "mean"),
                    long_acc_mean=("long_acc", "mean"),
                    long_acc_std=("long_acc", "std"),
                    gap_mean=("generalization_gap", "mean"),
                    retention_mean=("retention_ratio", "mean"),
                    beta_over_gamma_mean=("beta_over_gamma", "mean"),
                    aulc_primary_mean=(preferred_aulc, "mean") if preferred_aulc else ("long_acc", "mean"),
                )
                .sort_values(["gamma", "beta"])
                .reset_index(drop=True)
            )
            limit_agg["long_acc_std"] = limit_agg["long_acc_std"].fillna(0.0)
            limit_agg["long_acc_ci95"] = 1.96 * limit_agg["long_acc_std"] / np.sqrt(np.clip(limit_agg["n_runs"], 1, None))
            limit_agg.to_csv(limit_cell_csv, index=False)

            gstats = (
                limit_df.groupby("gamma", as_index=False)
                .agg(
                    n=("seed", "count"),
                    long_acc_mean=("long_acc", "mean"),
                    long_acc_std=("long_acc", "std"),
                    aulc_mean=(preferred_aulc, "mean") if preferred_aulc else ("long_acc", "mean"),
                )
                .sort_values("gamma")
            )
            gstats["long_acc_std"] = gstats["long_acc_std"].fillna(0.0)
            gstats["long_acc_ci95"] = 1.96 * gstats["long_acc_std"] / np.sqrt(np.clip(gstats["n"], 1, None))

            fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=200)
            ax.errorbar(
                gstats["gamma"].to_numpy(dtype=float),
                gstats["long_acc_mean"].to_numpy(dtype=float),
                yerr=gstats["long_acc_ci95"].to_numpy(dtype=float),
                fmt="o-",
                label="long_acc_mean",
                capsize=3,
            )
            if preferred_aulc:
                ax.plot(
                    gstats["gamma"].to_numpy(dtype=float),
                    gstats["aulc_mean"].to_numpy(dtype=float),
                    "s--",
                    label=f"{preferred_aulc}_mean",
                )
            ax.set_xscale("log")
            ax.set_xlabel(r"$\gamma$ (log scale)")
            ax.set_ylabel("metric")
            ax.set_title(r"Limit Regime Trend: $\beta,\gamma \to 0$ with small $\beta/\gamma$")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(limit_plot)
            plt.close(fig)

            f.write("## Limit Regime: beta,gamma -> 0 with beta/gamma small\n")
            f.write(
                f"- Constraint: beta <= {args.limit_beta_max}, gamma <= {args.limit_gamma_max}, "
                f"beta/gamma <= {args.limit_beta_over_gamma_max}\n"
            )
            f.write(f"- Runs in subset: {len(limit_df)}\n")
            f.write(f"- Cells in subset: {len(limit_agg)}\n")
            f.write(f"- Mean(beta/gamma): {float(limit_df['beta_over_gamma'].mean()):.4f}\n")
            f.write(f"- Mean alpha_theory: {float(limit_df['alpha_theory'].mean()):.4f}\n")
            f.write(f"- Mean long_acc: {float(limit_df['long_acc'].mean()):.4f}\n")
            if preferred_aulc:
                f.write(f"- Mean {preferred_aulc}: {float(limit_df[preferred_aulc].mean()):.4f}\n")
            if len(limit_df) > 1:
                lg = np.log(np.clip(limit_df["gamma"].to_numpy(dtype=float), 1e-12, None))
                lacc = limit_df["long_acc"].to_numpy(dtype=float)
                corr_lg_lacc = safe_corr(lg, lacc)
                f.write(f"- Corr(log(gamma), long_acc) in subset: {corr_lg_lacc:.4f}\n")
                if preferred_aulc:
                    aulc = limit_df[preferred_aulc].to_numpy(dtype=float)
                    corr_lg_aulc = safe_corr(lg, aulc)
                    f.write(f"- Corr(log(gamma), {preferred_aulc}) in subset: {corr_lg_aulc:.4f}\n")
            f.write("\n")
        else:
            f.write("## Limit Regime: beta,gamma -> 0 with beta/gamma small\n")
            f.write(
                f"- Constraint: beta <= {args.limit_beta_max}, gamma <= {args.limit_gamma_max}, "
                f"beta/gamma <= {args.limit_beta_over_gamma_max}\n"
            )
            f.write("- No runs found in this subset.\n\n")

        f.write("## Regression (long_acc ~ log(beta) + gamma + interaction)\n")
        f.write(f"- R^2: {model_r2:.4f}\n")
        for _, r in coef_df.iterrows():
            ptxt = "NA" if pd.isna(r["perm_pvalue"]) else f"{float(r['perm_pvalue']):.4f}"
            f.write(
                f"- {r['term']}: coef={float(r['coef']):.6f}, se={float(r['se']):.6f}, perm_p={ptxt}\n"
            )
        f.write("\n")

        f.write("## Phase Counts\n")
        for _, r in phase_counts.iterrows():
            f.write(f"- {r['phase_name']}: {int(r['num_cells'])}\n")
        f.write("\n")

        f.write("## Artifacts\n")
        f.write("- aggregated_cell_stats.csv\n")
        f.write("- regression_coefficients.csv\n")
        f.write("- phase_counts.csv\n")
        f.write("- heatmap_long_acc_mean.png\n")
        f.write("- heatmap_long_acc_ci95.png\n")
        f.write("- heatmap_gap_mean.png\n")
        for c in aulc_cols:
            cm = f"{c}_mean"
            if cm in agg.columns:
                f.write(f"- heatmap_{cm}.png\n")
        f.write("- phase_diagram_aggregated.png\n")
        f.write("- scatter_alpha_vs_long_acc.png\n")
        for c in renyi_cols:
            f.write(f"- heatmap_{c}_mean.png\n")
        if (out_dir / "limit_subset_runs.csv").exists():
            f.write("- limit_subset_runs.csv\n")
        if (out_dir / "limit_subset_cell_stats.csv").exists():
            f.write("- limit_subset_cell_stats.csv\n")
        if (out_dir / "limit_subset_gamma_trend.png").exists():
            f.write("- limit_subset_gamma_trend.png\n")

    meta = {
        "model_r2": model_r2,
        "run_corr_alpha_long": run_corr,
        "cell_corr_alpha_long": cell_corr,
        "aulc_columns": aulc_cols,
        "renyi_columns": renyi_cols,
        "phase_mode": args.phase_mode,
        "phase_counts": {str(k): int(v) for k, v in phase_counts.set_index("phase_name")["num_cells"].to_dict().items()},
    }
    with (out_dir / "analysis_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[Done] Analysis artifacts:")
    print(f"  - {agg_csv}")
    print(f"  - {coef_csv}")
    print(f"  - {phase_counts_csv}")
    print(f"  - {report_md}")
    print(f"  - {out_dir / 'phase_diagram_aggregated.png'}")


if __name__ == "__main__":
    main()
