#!/usr/bin/env python3
"""Paper Figure 3: 2×2 multi-diagnostic panel showing four independent
diagnostics that all separate emergent from chaos cells.

  panel A: train_acc(L=512)        vs cell index (sorted)
  panel B: length-gen ratio acc(2048)/acc(512)
  panel C: Rényi D_q=1
  panel D: AULC (training-loss area)

Each cell colored by phase classification. Visual goal: show that
all four columns put emergent cells in a different cluster from
chaos cells, ruling out "this is just the train_acc cutoff".

Output: case/phase/figures/multi_diagnostic.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import classify_fixed, PHASE_COLORS, PHASE_NAMES


def collect_cells(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for vdir in sorted(runs_dir.iterdir()):
        # Skip non-default training regimes (Mamba + 30k-step pilots).
        if vdir.name.startswith("mamba_") or vdir.name.startswith("pilot_30k_"):
            continue
        sumcsv = vdir / "run_summary.csv"
        rawcsv = vdir / "raw_metrics.csv"
        if not sumcsv.exists() or not rawcsv.exists() or not vdir.is_dir():
            continue
        s = pd.read_csv(sumcsv)
        r = pd.read_csv(rawcsv)
        for (b, g), grp in s.groupby(["beta", "gamma"]):
            n = len(grp)
            if n < 1 or float(grp["train_acc"].mean()) < 0.04:
                continue
            ta = float(grp["train_acc"].mean())
            la = float(grp["long_acc"].mean())
            gap = float(grp["generalization_gap"].mean())
            ret = float(grp["retention_ratio"].mean())
            aulc = float(grp["aulc_train_to_final_norm"].mean())
            d_q1 = float(grp["renyi_D_rate_q1p0"].mean())
            r_sub = r[(r["beta"] == b) & (r["gamma"] == g)]
            a512 = float(r_sub[r_sub["eval_len"] == 512]["acc"].mean())
            a2048 = float(r_sub[r_sub["eval_len"] == 2048]["acc"].mean())
            ratio = a2048 / max(a512, 1e-6)
            ph = classify_fixed({
                "train_acc_mean": ta, "long_acc_mean": la,
                "gap_mean": gap, "retention_mean": ret,
            })
            rows.append({
                "variant": vdir.name, "beta": b, "gamma": g, "n": n,
                "train_acc": ta, "long_acc": la, "ratio": ratio,
                "D_q1": d_q1, "aulc": aulc, "phase": ph,
            })
    return pd.DataFrame(rows)


def plot_panels(cells: pd.DataFrame, out_path: Path) -> None:
    cells = cells.sort_values("train_acc").reset_index(drop=True)
    cells["idx"] = range(len(cells))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    panels = [
        ("train_acc", "train_acc(L=512)", 0.20, "chaos threshold"),
        ("ratio", "length-gen ratio  acc(2048)/acc(512)", 0.42,
         "emergent ≈ 0.39 / chaos ≈ 0.47"),
        ("D_q1", "Rényi dimension D_q=1 (data side)", None, None),
        ("aulc", "AULC (norm)  training-loss curve", None, None),
    ]
    for ax, (col, title, hline, hline_label) in zip(axes.flatten(), panels):
        for code in sorted(PHASE_NAMES):
            sub = cells[cells["phase"] == code]
            if sub.empty:
                continue
            ax.scatter(sub["idx"], sub[col],
                       s=22, c=PHASE_COLORS[code], alpha=0.85,
                       edgecolors="black", linewidth=0.4,
                       label=f"{PHASE_NAMES[code]} ({len(sub)})", zorder=3)
        if hline is not None:
            ax.axhline(hline, color="#666666", linestyle=":",
                       linewidth=0.7, alpha=0.6)
            if hline_label:
                ax.text(2, hline + 0.005, hline_label, fontsize=7,
                        color="#666666")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("cell index (sorted by train_acc)")
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)

    axes[0, 0].legend(fontsize=8, framealpha=0.9, loc="upper left")
    fig.suptitle(
        "Four independent diagnostics each separate emergent from chaos\n"
        f"({len(cells)} cells across {cells['variant'].nunique()} variants)",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {len(cells)} cells; emergent={(cells['phase']==1).sum()},"
          f" chaos={(cells['phase']==0).sum()}")
    print(f"  wrote {out_path}")


def main() -> None:
    runs_dir = Path(__file__).resolve().parent / "runs"
    out_path = Path(__file__).resolve().parent / "figures" / "multi_diagnostic.png"
    cells = collect_cells(runs_dir)
    plot_panels(cells, out_path)


if __name__ == "__main__":
    main()
