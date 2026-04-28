#!/usr/bin/env python3
"""End-to-end phase-diagram sweep.

Resolves a (β, γ) sweep plan via ``data_generator.py``, trains
``TinyCausalTransformer`` for each (β, γ, seed) using
``phase_core.train_one_model``, writes ``run_summary.csv`` and
``raw_metrics.csv`` (compatible with ``plot_phase_diagram.py``), and unless
``sweep.no_plot=true`` is passed runs the plotter to drop the final figures
alongside the CSVs in one shot.

All configuration lives in ``configs/config.yaml``. Override on the CLI with
OmegaConf dotlist syntax (``key=value`` or ``nested.key=[1,2,3]``).

Examples
--------

    # CPU smoke test (corners plan, single seed, 100 train steps)
    python case/phase/phase_sweep.py plan.name=corners sweep.seeds=[1] \\
        sweep.device=cpu model.train_steps=100 sweep.out_dir=plots/sweep_smoke

    # 7×7 standard grid on GPU, 3 seeds, then auto-plot
    python case/phase/phase_sweep.py plan.name=standard sweep.device=cuda \\
        sweep.out_dir=plots/sweep_standard

    # Critical-line scan along α=0.4
    python case/phase/phase_sweep.py plan.name=alpha_iso plan.alpha=0.4 \\
        sweep.device=cuda sweep.out_dir=plots/sweep_alpha04

    # Boundary refinement around (β=2.0, γ=0.3)
    python case/phase/phase_sweep.py plan.name=refine plan.beta=2.0 plan.gamma=0.3 \\
        sweep.device=cuda sweep.out_dir=plots/sweep_refine
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_generator as dg
from model import MIN_NON_EMBED_PARAMS, count_non_embedding_params
from phase_core import train_one_model
from utils import choose_device, load_config


SUMMARY_FIELDS = [
    "beta", "gamma", "seed", "alpha_theory",
    "train_len", "long_len",
    "train_acc", "long_acc", "train_loss", "long_loss",
    "generalization_gap", "retention_ratio",
    "objective",
    "initial_train_step_loss", "final_train_step_loss", "best_train_step_loss",
    "aulc_train_to_final", "aulc_train_to_final_norm",
    "train_time_sec", "train_steps", "d_model", "num_layers",
]
RAW_FIELDS = [
    "beta", "gamma", "seed", "eval_len", "acc", "loss",
    "alpha_theory", "train_steps", "d_model", "num_layers",
]


def resolve_plan(plan_cfg) -> dg.SweepPlan:
    """Resolve a sweep plan name (with optional CLI args) to a SweepPlan."""
    name = plan_cfg.name
    if name == "recommended":
        plans = dg.recommended()
        seen, merged = set(), []
        for pl in plans:
            for pair in pl.pairs:
                key = (round(pair[0], 8), round(pair[1], 8))
                if key not in seen:
                    seen.add(key)
                    merged.append(pair)
        return dg.SweepPlan(
            name="recommended",
            description=("Union of standard_grid + corners + "
                         "alpha_iso(0.1, 0.4, 1.0) + fast_beta_p2"),
            pairs=tuple(merged),
        )
    return dg._FACTORIES[name](plan_cfg)


def build_config(model_cfg, long_len) -> Dict[str, object]:
    """Convert the (now fully-populated) cfg.model OmegaConf node into the
    plain dict that train_one_model expects, then enforce the 100M floor."""
    from omegaconf import OmegaConf

    cfg = dict(OmegaConf.to_container(model_cfg, resolve=True))

    train_len = int(cfg["train_len"])
    if long_len is not None:
        cfg["eval_lengths"] = sorted({train_len, int(long_len)})
    else:
        base_evals = [int(x) for x in cfg["eval_lengths"]]
        cfg["eval_lengths"] = sorted({train_len, *base_evals})

    n_non_embed = count_non_embedding_params(
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        ff_mult=int(cfg["ff_mult"]),
        num_layers=int(cfg["num_layers"]),
        vocab_size=int(cfg["vocab_size"]),
    )
    if n_non_embed < MIN_NON_EMBED_PARAMS:
        raise ValueError(
            f"non-embedding params = {n_non_embed:,} (~{n_non_embed/1e6:.2f}M) "
            f"< floor {MIN_NON_EMBED_PARAMS:,} (100M). "
            f"Bump model.d_model / model.num_layers (current "
            f"d_model={cfg['d_model']}, num_layers={cfg['num_layers']}, "
            f"nhead={cfg['nhead']}, ff_mult={cfg['ff_mult']})."
        )
    cfg["_non_embed_params"] = int(n_non_embed)
    return cfg


def write_csv(path: Path, rows: List[Dict[str, object]],
              preferred: Sequence[str]) -> None:
    """Write rows with `preferred` columns first, then any extras."""
    seen = set(preferred)
    extras = []
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                extras.append(k)
    fields = list(preferred) + extras
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def run_sweep(plan: dg.SweepPlan, seeds: Sequence[int],
              cfg: Dict[str, object], device, out_dir: Path,
              no_tqdm: bool = False) -> Path:
    summary_csv = out_dir / "run_summary.csv"
    raw_csv = out_dir / "raw_metrics.csv"
    meta_json = out_dir / "sweep_meta.json"

    train_len = int(cfg["train_len"])
    long_len = max(int(x) for x in cfg["eval_lengths"])

    print(f"[plan] {plan.name} ({len(plan.pairs)} pairs)")
    print(f"[plan] {plan.description}")
    print(f"[seeds] {list(seeds)}")
    print(f"[device] {device}")
    n_non_embed = int(cfg.get("_non_embed_params", 0))
    print(f"[config] train_len={train_len}, long_len={long_len}, "
          f"train_steps={cfg['train_steps']}, d_model={cfg['d_model']}, "
          f"layers={cfg['num_layers']}, nhead={cfg['nhead']}, "
          f"ff_mult={cfg['ff_mult']}")
    print(f"[size]   non-embedding params = {n_non_embed:,} "
          f"(~{n_non_embed/1e6:.2f}M; floor 100M)")

    jobs = [(b, g, s) for b, g in plan.pairs for s in seeds]
    total = len(jobs)
    print(f"[jobs] {total} training runs\n")

    # Write meta.json eagerly so mid-run aggregators (e.g.
    # aggregate_report.py) can read the authoritative expected count
    # before the sweep finishes. Updated again at end with elapsed_sec.
    meta_json.write_text(json.dumps({
        "plan": plan.to_dict(),
        "seeds": list(seeds),
        "device": str(device),
        "config": cfg,
        "num_runs": total,
        "elapsed_sec": 0.0,
        "status": "running",
    }, indent=2, default=str))

    iterable = jobs
    pbar = None
    if not no_tqdm:
        try:
            from tqdm import tqdm
            pbar = tqdm(jobs, total=total,
                        desc=f"sweep[{plan.name}]", dynamic_ncols=True)
            iterable = pbar
        except ImportError:
            pass

    summary_rows: List[Dict[str, object]] = []
    raw_rows: List[Dict[str, object]] = []
    t0 = time.time()

    for beta, gamma, seed in iterable:
        if pbar is not None:
            pbar.set_postfix(b=f"{beta:.4g}", g=f"{gamma:.3g}", s=seed)
        result = train_one_model(beta=beta, gamma=gamma, seed=int(seed),
                                 config=cfg, device=device)
        eval_by_len = result["eval_by_len"]
        if train_len not in eval_by_len:
            raise ValueError(
                f"train_len={train_len} missing from eval_by_len={list(eval_by_len)}")

        train_acc = float(eval_by_len[train_len]["acc"])
        train_loss = float(eval_by_len[train_len]["loss"])
        long_acc = float(eval_by_len[long_len]["acc"])
        long_loss = float(eval_by_len[long_len]["loss"])
        alpha_theory = gamma / (2.0 * max(beta, 1e-8))

        row = {
            "beta": float(beta), "gamma": float(gamma), "seed": int(seed),
            "alpha_theory": float(alpha_theory),
            "train_len": train_len, "long_len": long_len,
            "train_acc": train_acc, "long_acc": long_acc,
            "train_loss": train_loss, "long_loss": long_loss,
            "generalization_gap": train_acc - long_acc,
            "retention_ratio": long_acc / max(train_acc, 1e-8),
            "objective": str(result.get("objective", "next_token_pretraining")),
            "initial_train_step_loss": float(result.get("initial_train_step_loss", 0.0)),
            "final_train_step_loss": float(result.get("final_train_step_loss", 0.0)),
            "best_train_step_loss": float(result.get("best_train_step_loss", 0.0)),
            "aulc_train_to_final": float(result.get("aulc_train_to_final", 0.0)),
            "aulc_train_to_final_norm": float(result.get("aulc_train_to_final_norm", 0.0)),
            "train_time_sec": float(result["train_time_sec"]),
            "train_steps": int(cfg["train_steps"]),
            "d_model": int(cfg["d_model"]),
            "num_layers": int(cfg["num_layers"]),
        }
        if isinstance(result.get("renyi_diag"), dict):
            row.update(result["renyi_diag"])
        summary_rows.append(row)

        for L in sorted(eval_by_len):
            raw_rows.append({
                "beta": float(beta), "gamma": float(gamma), "seed": int(seed),
                "eval_len": int(L),
                "acc": float(eval_by_len[L]["acc"]),
                "loss": float(eval_by_len[L]["loss"]),
                "alpha_theory": float(alpha_theory),
                "train_steps": int(cfg["train_steps"]),
                "d_model": int(cfg["d_model"]),
                "num_layers": int(cfg["num_layers"]),
            })

        write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
        write_csv(raw_csv, raw_rows, RAW_FIELDS)

    if pbar is not None:
        pbar.close()

    elapsed = time.time() - t0
    meta_json.write_text(json.dumps({
        "plan": plan.to_dict(),
        "seeds": list(seeds),
        "device": str(device),
        "config": cfg,
        "num_runs": total,
        "elapsed_sec": elapsed,
        "status": "complete",
    }, indent=2, default=str))

    print(f"\n[done] {total} runs in {elapsed/60:.1f} min")
    print(f"  {summary_csv}")
    print(f"  {raw_csv}")
    print(f"  {meta_json}")
    return summary_csv


def call_plotter(summary_csv: Path, out_dir: Path) -> None:
    plotter = Path(__file__).parent / "plot_phase_diagram.py"
    if not plotter.exists():
        print(f"[warn] {plotter} not found — skipping auto-plot")
        return
    cmd = [sys.executable, str(plotter),
           f"plot.in_summary={summary_csv}",
           f"plot.out_dir={out_dir}"]
    print(f"\n[plot] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[warn] plot_phase_diagram.py failed: {e}")


def main() -> None:
    cfg = load_config(sys.argv[1:])

    plan = resolve_plan(cfg.plan)
    seeds = [int(s) for s in cfg.sweep.seeds]
    train_cfg = build_config(cfg.model, cfg.sweep.long_len)
    device = choose_device(str(cfg.sweep.device))

    out_dir = Path(str(cfg.sweep.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = run_sweep(plan, seeds, train_cfg, device, out_dir,
                            no_tqdm=bool(cfg.sweep.no_tqdm))
    if not bool(cfg.sweep.no_plot):
        call_plotter(summary_csv, out_dir)


if __name__ == "__main__":
    main()
