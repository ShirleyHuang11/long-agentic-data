#!/usr/bin/env python3
"""Aggregate run_summary.csv across every case/phase/runs/<variant>/ subdir
and emit a Markdown report.

Designed to be safe to run mid-sweep: any variant whose CSV is missing or
short is reported as "partial" and still folded into the report.

Sections written:
  1. submission inventory + completion status (incl. sacct if available)
  2. phase-diagram highlights from `standard`
  3. critical-line behavior across `alpha_iso_*`
  4. limit / near-origin behavior from `fast_beta_*`
  5. refine zoomed-in cells (`refine_*`)
  6. unusual cells worth deeper investigation
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import classify_fixed  # single source of truth for thresholds


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO_ROOT / "case" / "phase" / "runs"
REPORT_PATH = REPO_ROOT / "case" / "phase" / "REPORT_sweep_summary.md"

# Per-variant expected number of unique (β, γ) pairs (× seeds = total runs).
# Drives the "completion %" column. Seeds default to 3 unless overridden.
EXPECTED_PAIRS = {
    "standard": 49,
    "corners": 5,
    "alpha_iso_0p1": 12,
    "alpha_iso_0p4": 12,
    "alpha_iso_1p0": 12,
    "beta_axis_g0p3": 12,
    "gamma_axis_b0p4": 12,
    "fast_beta_p2": 12,
    "fast_beta_p3": 12,
    "refine_b0p4_g0p3": 25,
    "refine_b2p0_g0p3": 25,
}
EXPECTED_SEEDS = 3


# -------------------- phase classification ----------------------------------


_CODE_TO_LABEL = {0: "chaos", 1: "emergent", 2: "super_gen", 3: "rote"}


def classify_phase(row) -> str:
    """Fixed-threshold 4-phase classifier on per-cell means.

    Delegates to ``utils.classify_fixed`` so the chaos/rote/super-gen
    thresholds live in exactly one place. Returns the string label this
    file's downstream filters expect.
    """
    return _CODE_TO_LABEL[classify_fixed(row)]


def aggregate_cells(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["beta", "gamma"], as_index=False).agg(
        n_seeds=("train_acc", "count"),
        train_acc_mean=("train_acc", "mean"),
        train_acc_std=("train_acc", "std"),
        long_acc_mean=("long_acc", "mean"),
        long_acc_std=("long_acc", "std"),
        gap_mean=("generalization_gap", "mean"),
        retention_mean=("retention_ratio", "mean"),
    )
    grouped["alpha_theory"] = grouped["gamma"] / (2.0 * grouped["beta"].clip(lower=1e-12))
    grouped["phase"] = grouped.apply(classify_phase, axis=1)
    return grouped


# -------------------- per-variant summary -----------------------------------


def variant_summary(name: str, csv_path: Path) -> Dict[str, object]:
    info: Dict[str, object] = {"variant": name}
    expected_total = EXPECTED_PAIRS.get(name, 0) * EXPECTED_SEEDS
    info["expected"] = expected_total
    info["row_count"] = 0
    if not csv_path.exists():
        info["status"] = "missing"
        info["completion_pct"] = 0.0
        return info

    df = pd.read_csv(csv_path)
    info["row_count"] = int(len(df))
    info["completion_pct"] = (
        100.0 * len(df) / expected_total if expected_total else float("nan")
    )

    if df.empty:
        info["status"] = "empty"
        return info

    info["status"] = "complete" if info["completion_pct"] >= 99.0 else "partial"
    info["mean_long_acc"] = float(df["long_acc"].mean())
    info["mean_train_acc"] = float(df["train_acc"].mean())
    info["mean_gap"] = float(df["generalization_gap"].mean())
    info["alpha_min"] = float(df["alpha_theory"].min())
    info["alpha_max"] = float(df["alpha_theory"].max())
    info["agg"] = aggregate_cells(df)
    return info


# -------------------- sacct helper ------------------------------------------


def sacct_lookup(job_ids: List[str]) -> Dict[str, Tuple[str, str, str]]:
    """jobid → (state, elapsed, exit_code). Empty dict if sacct unavailable."""
    if not shutil.which("sacct") or not job_ids:
        return {}
    try:
        out = subprocess.check_output(
            ["sacct", "-j", ",".join(job_ids), "--noheader",
             "--format=JobID,State,Elapsed,ExitCode", "-P"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return {}
    result: Dict[str, Tuple[str, str, str]] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        jid = parts[0].split(".")[0]  # strip .batch / .extern steps
        if jid in job_ids and jid not in result:
            result[jid] = (parts[1], parts[2], parts[3])
    return result


# -------------------- markdown rendering ------------------------------------


def fmt_pct(x: float) -> str:
    return f"{x:5.1f}%" if x == x else "  n/a"


def fmt_float(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "  n/a"
    if not (v == v):
        return "  n/a"
    if abs(v) >= 0.01:
        return f"{v:.3f}"
    return f"{v:.2e}"


def section_inventory(infos: List[Dict[str, object]],
                      sacct_map: Dict[str, Tuple[str, str, str]]) -> str:
    lines = ["## 1. Submission inventory + completion status\n"]
    if sacct_map:
        lines.append("| variant | rows | expected | completion | mean long_acc | sacct state | elapsed |")
        lines.append("|---|---:|---:|---:|---:|---|---|")
    else:
        lines.append("| variant | rows | expected | completion | mean long_acc |")
        lines.append("|---|---:|---:|---:|---:|")
    for info in infos:
        name = info["variant"]
        rows = info.get("row_count", 0)
        expected = info.get("expected", 0)
        pct = info.get("completion_pct", float("nan"))
        long_acc = fmt_float(info.get("mean_long_acc"))
        if sacct_map:
            # match by job NAME (phase-<variant>); we don't have ID directly,
            # so fall back to "—" if not present.
            state = elapsed = "—"
            for jid, (st, el, _) in sacct_map.items():
                if name in (jid, f"phase-{name}"):
                    state, elapsed = st, el
                    break
            lines.append(
                f"| `{name}` | {rows} | {expected} | {fmt_pct(pct)} | {long_acc} | {state} | {elapsed} |"
            )
        else:
            lines.append(
                f"| `{name}` | {rows} | {expected} | {fmt_pct(pct)} | {long_acc} |"
            )
    return "\n".join(lines) + "\n"


def section_standard(info: Optional[Dict[str, object]]) -> str:
    lines = ["## 2. Phase-diagram highlights from `standard` (full 7×7 grid)\n"]
    if info is None or "agg" not in info:
        lines.append("_No `standard` data available._\n")
        return "\n".join(lines)
    agg: pd.DataFrame = info["agg"]
    counts = agg["phase"].value_counts().to_dict()
    lines.append(f"- {len(agg)} cells aggregated (across "
                 f"{int(agg['n_seeds'].max())} seeds max)")
    lines.append(f"- Phase counts: {counts}")
    fig = RUNS_DIR / "standard" / "phase_diagram_long_acc.png"
    if fig.exists():
        lines.append(f"\n![standard phase diagram]({fig.relative_to(REPORT_PATH.parent)})\n")
    panels = RUNS_DIR / "standard" / "phase_diagram_panels.png"
    if panels.exists():
        lines.append(f"![standard panels]({panels.relative_to(REPORT_PATH.parent)})\n")
    classified = RUNS_DIR / "standard" / "phase_diagram_classified.png"
    if classified.exists():
        lines.append(f"![standard classified]({classified.relative_to(REPORT_PATH.parent)})\n")

    top_rote = agg[agg["phase"] == "rote"].nlargest(5, "gap_mean")
    top_super = agg[agg["phase"] == "super_gen"].nlargest(5, "long_acc_mean")
    if not top_rote.empty:
        lines.append("\n**Top rote-memorisation cells (largest gap):**\n")
        lines.append("| β | γ | α_theory | train_acc | long_acc | gap |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for _, r in top_rote.iterrows():
            lines.append(
                f"| {fmt_float(r['beta'])} | {fmt_float(r['gamma'])} | "
                f"{fmt_float(r['alpha_theory'])} | {fmt_float(r['train_acc_mean'])} | "
                f"{fmt_float(r['long_acc_mean'])} | {fmt_float(r['gap_mean'])} |"
            )
    if not top_super.empty:
        lines.append("\n**Top super-generalisation cells (largest long_acc):**\n")
        lines.append("| β | γ | α_theory | train_acc | long_acc | retention |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for _, r in top_super.iterrows():
            lines.append(
                f"| {fmt_float(r['beta'])} | {fmt_float(r['gamma'])} | "
                f"{fmt_float(r['alpha_theory'])} | {fmt_float(r['train_acc_mean'])} | "
                f"{fmt_float(r['long_acc_mean'])} | {fmt_float(r['retention_mean'])} |"
            )
    return "\n".join(lines) + "\n"


def section_alpha_iso(infos: List[Dict[str, object]]) -> str:
    lines = ["## 3. Critical-line behavior (γ = 2αβ)\n"]
    relevant = [i for i in infos if i["variant"].startswith("alpha_iso_") and "agg" in i]
    if not relevant:
        lines.append("_No alpha_iso data available._\n")
        return "\n".join(lines)
    for info in relevant:
        name = info["variant"]
        agg = info["agg"]
        counts = agg["phase"].value_counts().to_dict()
        lines.append(f"### `{name}`")
        lines.append(f"- {len(agg)} cells, phases: {counts}")
        lines.append(f"- mean long_acc {fmt_float(info['mean_long_acc'])}, "
                     f"α range [{fmt_float(info['alpha_min'])}, "
                     f"{fmt_float(info['alpha_max'])}]")
        fig = RUNS_DIR / name / "phase_diagram_long_acc.png"
        if fig.exists():
            lines.append(f"\n![{name}]({fig.relative_to(REPORT_PATH.parent)})\n")
    return "\n".join(lines) + "\n"


def section_fast_beta(infos: List[Dict[str, object]]) -> str:
    lines = ["## 4. Limit / near-origin behavior (β = γ^p, p > 1)\n"]
    relevant = [i for i in infos if i["variant"].startswith("fast_beta_") and "agg" in i]
    if not relevant:
        lines.append("_No fast_beta data available._\n")
        return "\n".join(lines)
    for info in relevant:
        name = info["variant"]
        agg = info["agg"].sort_values("gamma")
        lines.append(f"### `{name}`")
        lines.append(f"- {len(agg)} cells along γ ∈ [{agg['gamma'].min():.3g}, "
                     f"{agg['gamma'].max():.3g}], β ∈ [{agg['beta'].min():.3g}, "
                     f"{agg['beta'].max():.3g}]")
        lines.append("\n| γ | β | long_acc | train_acc | gap | phase |")
        lines.append("|---:|---:|---:|---:|---:|---|")
        for _, r in agg.iterrows():
            lines.append(
                f"| {fmt_float(r['gamma'])} | {fmt_float(r['beta'])} | "
                f"{fmt_float(r['long_acc_mean'])} | {fmt_float(r['train_acc_mean'])} | "
                f"{fmt_float(r['gap_mean'])} | {r['phase']} |"
            )
        fig = RUNS_DIR / name / "phase_diagram_long_acc.png"
        if fig.exists():
            lines.append(f"\n![{name}]({fig.relative_to(REPORT_PATH.parent)})\n")
    return "\n".join(lines) + "\n"


def section_refine(infos: List[Dict[str, object]]) -> str:
    lines = ["## 5. Refine zoomed-in cells\n"]
    relevant = [i for i in infos if i["variant"].startswith("refine_") and "agg" in i]
    if not relevant:
        lines.append("_No refine data available._\n")
        return "\n".join(lines)
    for info in relevant:
        name = info["variant"]
        agg = info["agg"]
        counts = agg["phase"].value_counts().to_dict()
        lines.append(f"### `{name}`")
        lines.append(f"- {len(agg)} cells, phases: {counts}")
        lines.append(f"- long_acc mean={fmt_float(info['mean_long_acc'])}, "
                     f"std across cells={fmt_float(agg['long_acc_mean'].std())}")
        fig = RUNS_DIR / name / "phase_diagram_long_acc.png"
        if fig.exists():
            lines.append(f"\n![{name}]({fig.relative_to(REPORT_PATH.parent)})\n")
    return "\n".join(lines) + "\n"


def section_top_nonchaos(infos: List[Dict[str, object]]) -> str:
    """Surface the cells that escape `chaos`, sorted by long_acc.

    Without this, a single emergent corner like (β=8, γ=0.02) inside
    `corners` is invisible: the unusual-cell rules below only fire on
    rote-with-collapse / contradictory / high-seed-variance patterns.
    """
    lines = ["## 5b. Top non-chaos cells across all variants (by long_acc)\n"]
    rows: List[Dict[str, object]] = []
    for info in infos:
        if "agg" not in info:
            continue
        agg: pd.DataFrame = info["agg"]
        nonchaos = agg[agg["phase"] != "chaos"]
        for _, r in nonchaos.iterrows():
            rows.append({
                "variant": info["variant"],
                "beta": float(r["beta"]),
                "gamma": float(r["gamma"]),
                "alpha_theory": float(r["alpha_theory"]),
                "n_seeds": int(r["n_seeds"]),
                "train_acc": float(r["train_acc_mean"]),
                "long_acc": float(r["long_acc_mean"]),
                "gap": float(r["gap_mean"]),
                "retention": float(r["retention_mean"]),
                "phase": str(r["phase"]),
            })
    if not rows:
        lines.append("_All aggregated cells classified as `chaos`. "
                     "Either the model is undertrained at every (β, γ) "
                     "or the fixed thresholds need recalibration "
                     "(try `plot.phase_mode=quantile`)._\n")
        return "\n".join(lines)

    rows.sort(key=lambda r: r["long_acc"], reverse=True)
    lines.append(f"{len(rows)} cell(s) escaped chaos. Top {min(15, len(rows))} by long_acc:\n")
    lines.append("| variant | β | γ | α_theory | n_seeds | train_acc | long_acc | gap | retention | phase |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows[:15]:
        lines.append(
            f"| `{r['variant']}` | {fmt_float(r['beta'])} | "
            f"{fmt_float(r['gamma'])} | {fmt_float(r['alpha_theory'])} | "
            f"{r['n_seeds']} | {fmt_float(r['train_acc'])} | "
            f"{fmt_float(r['long_acc'])} | {fmt_float(r['gap'])} | "
            f"{fmt_float(r['retention'])} | {r['phase']} |"
        )
    return "\n".join(lines) + "\n"


def section_unusual(infos: List[Dict[str, object]]) -> str:
    lines = ["## 6. Unusual cells worth deeper investigation\n"]
    pieces: List[str] = []
    for info in infos:
        if "agg" not in info:
            continue
        name = info["variant"]
        agg: pd.DataFrame = info["agg"]
        # 1. high training, very low long: mode-collapse or pathological
        odd1 = agg[(agg["train_acc_mean"] >= 0.5) & (agg["long_acc_mean"] <= 0.05)]
        # 2. low training, high long: contradictory (likely random fluke)
        odd2 = agg[(agg["train_acc_mean"] <= 0.10) & (agg["long_acc_mean"] >= 0.20)]
        # 3. very high seed std on long_acc: unstable
        if "long_acc_std" in agg.columns:
            odd3 = agg[agg["long_acc_std"] > 0.20]
        else:
            odd3 = agg.iloc[0:0]
        for label, sub in (("rote-with-collapse", odd1),
                           ("contradictory", odd2),
                           ("high-seed-variance", odd3)):
            if not sub.empty:
                pieces.append(f"### `{name}` — {label} ({len(sub)} cell(s))")
                pieces.append("| β | γ | train_acc | long_acc | gap | seed_std |")
                pieces.append("|---:|---:|---:|---:|---:|---:|")
                for _, r in sub.head(8).iterrows():
                    seed_std = r.get("long_acc_std", float("nan"))
                    pieces.append(
                        f"| {fmt_float(r['beta'])} | {fmt_float(r['gamma'])} | "
                        f"{fmt_float(r['train_acc_mean'])} | "
                        f"{fmt_float(r['long_acc_mean'])} | "
                        f"{fmt_float(r['gap_mean'])} | {fmt_float(seed_std)} |"
                    )
                pieces.append("")
    if not pieces:
        lines.append("_No unusual cells flagged._\n")
    else:
        lines.extend(pieces)
    return "\n".join(lines) + "\n"


# -------------------- main --------------------------------------------------


def main() -> None:
    job_ids = [a for a in sys.argv[1:] if a.isdigit()]
    sacct_map = sacct_lookup(job_ids)

    if not RUNS_DIR.exists():
        REPORT_PATH.write_text(f"# Sweep summary\n\n_No runs directory at {RUNS_DIR}._\n")
        print(f"[warn] no runs dir; wrote empty report to {REPORT_PATH}")
        return

    infos: List[Dict[str, object]] = []
    for name in sorted(EXPECTED_PAIRS):
        csv_path = RUNS_DIR / name / "run_summary.csv"
        infos.append(variant_summary(name, csv_path))

    parts = ["# Phase-sweep aggregate report\n",
             f"_Generated from `{RUNS_DIR}`._\n"]
    if job_ids:
        parts.append(f"_Slurm jobs tracked: {', '.join(job_ids)}._\n")
    parts.append(section_inventory(infos, sacct_map))
    parts.append(section_standard(next((i for i in infos if i["variant"] == "standard"), None)))
    parts.append(section_alpha_iso(infos))
    parts.append(section_fast_beta(infos))
    parts.append(section_refine(infos))
    parts.append(section_top_nonchaos(infos))
    parts.append(section_unusual(infos))

    REPORT_PATH.write_text("\n".join(parts))
    print(f"[done] wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
