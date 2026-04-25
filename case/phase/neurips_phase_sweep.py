#!/usr/bin/env python3
"""Large-scale beta-gamma sweeps for paper-grade results.

Outputs:
- raw_metrics.csv: one row per (beta,gamma,seed,eval_len)
- run_summary.csv: one row per (beta,gamma,seed), including train/long metrics
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from tqdm.auto import tqdm

from phase_core import (
    choose_device,
    default_config_dict,
    parse_float_list,
    parse_int_list,
    train_one_model,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run large beta-gamma sweeps with multiple seeds")

    ap.add_argument("--out-dir", type=str, default="plots/phase_neurips_sweep")
    ap.add_argument("--config-json", type=str, default="", help="Optional JSON config file")
    ap.add_argument("--resume", action="store_true", help="Skip runs already in run_summary.csv")
    ap.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bar")

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # Sweep axes
    ap.add_argument("--betas", type=str, default="0.1,0.2,0.4,0.8,1.6,3.2,6.4")
    ap.add_argument("--gammas", type=str, default="0.05,0.15,0.30,0.45,0.60,0.80,1.00")
    ap.add_argument(
        "--beta-gamma-pairs",
        type=str,
        default="",
        help="Optional paired scan: 'b1:g1,b2:g2,...' (overrides cartesian betas x gammas)",
    )
    ap.add_argument("--seeds", type=str, default="1,2,3")

    # Model/training config (overrides JSON/default)
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--train-len", type=int, default=None)
    ap.add_argument("--eval-lengths", type=str, default=None)

    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--nhead", type=int, default=None)
    ap.add_argument("--ff-mult", type=int, default=None)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)

    ap.add_argument("--train-steps", type=int, default=None)
    ap.add_argument("--train-batch-size", type=int, default=None)
    ap.add_argument("--eval-batch-size", type=int, default=None)
    ap.add_argument("--eval-batches", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--deterministic", action="store_true", default=None)
    ap.add_argument("--renyi-seq-len", type=int, default=None)
    ap.add_argument("--renyi-qs", type=str, default=None, help="e.g. 0.5,1.0,2.0")
    ap.add_argument("--renyi-orders", type=str, default=None, help="e.g. 1,2,3")

    return ap.parse_args()


def load_json_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object")
    return data


def merge_config(args: argparse.Namespace) -> Dict[str, object]:
    cfg = default_config_dict()
    if args.config_json:
        cfg.update(load_json_config(Path(args.config_json)))

    # CLI overrides
    if args.vocab_size is not None:
        cfg["vocab_size"] = int(args.vocab_size)
    if args.train_len is not None:
        cfg["train_len"] = int(args.train_len)
    if args.eval_lengths is not None:
        cfg["eval_lengths"] = parse_int_list(args.eval_lengths)
    if args.d_model is not None:
        cfg["d_model"] = int(args.d_model)
    if args.nhead is not None:
        cfg["nhead"] = int(args.nhead)
    if args.ff_mult is not None:
        cfg["ff_mult"] = int(args.ff_mult)
    if args.num_layers is not None:
        cfg["num_layers"] = int(args.num_layers)
    if args.dropout is not None:
        cfg["dropout"] = float(args.dropout)
    if args.train_steps is not None:
        cfg["train_steps"] = int(args.train_steps)
    if args.train_batch_size is not None:
        cfg["train_batch_size"] = int(args.train_batch_size)
    if args.eval_batch_size is not None:
        cfg["eval_batch_size"] = int(args.eval_batch_size)
    if args.eval_batches is not None:
        cfg["eval_batches"] = int(args.eval_batches)
    if args.lr is not None:
        cfg["lr"] = float(args.lr)
    if args.grad_clip is not None:
        cfg["grad_clip"] = float(args.grad_clip)
    if args.deterministic is not None:
        cfg["deterministic"] = bool(args.deterministic)
    if args.renyi_seq_len is not None:
        cfg["renyi_seq_len"] = int(args.renyi_seq_len)
    if args.renyi_qs is not None:
        cfg["renyi_qs"] = parse_float_list(args.renyi_qs)
    if args.renyi_orders is not None:
        cfg["renyi_orders"] = parse_int_list(args.renyi_orders)
    return cfg


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def infer_fieldnames(rows: List[Dict[str, object]], preferred_prefix: Sequence[str]) -> List[str]:
    keys = []
    seen = set()
    for k in preferred_prefix:
        if k not in seen:
            keys.append(k)
            seen.add(k)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    return keys


def run_key(beta: float, gamma: float, seed: int) -> str:
    return f"b={beta:.10f}|g={gamma:.10f}|s={seed}"


def parse_beta_gamma_pairs(s: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    if not s.strip():
        return pairs
    for item in s.split(","):
        it = item.strip()
        if not it:
            continue
        if ":" not in it:
            raise ValueError(f"Invalid pair '{it}', expected 'beta:gamma'")
        btxt, gtxt = it.split(":", 1)
        pairs.append((float(btxt.strip()), float(gtxt.strip())))
    if not pairs:
        raise ValueError("--beta-gamma-pairs provided but no valid pairs parsed")
    return pairs


def main() -> None:
    args = parse_args()
    cfg = merge_config(args)

    pair_scan = parse_beta_gamma_pairs(args.beta_gamma_pairs)
    if pair_scan:
        betas = sorted({float(b) for b, _ in pair_scan})
        gammas = sorted({float(g) for _, g in pair_scan})
    else:
        betas = parse_float_list(args.betas)
        gammas = parse_float_list(args.gammas)
    seeds = parse_int_list(args.seeds)

    if len(cfg["eval_lengths"]) == 0:
        raise ValueError("eval_lengths must not be empty")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = out_dir / "raw_metrics.csv"
    summary_csv = out_dir / "run_summary.csv"
    meta_json = out_dir / "sweep_meta.json"

    existing_raw = read_csv_rows(raw_csv)
    existing_summary = read_csv_rows(summary_csv)

    done_keys = set()
    if args.resume:
        for r in existing_summary:
            try:
                done_keys.add(run_key(float(r["beta"]), float(r["gamma"]), int(float(r["seed"]))))
            except Exception:
                continue

    device = choose_device(args.device)
    print(f"[Device] {device}")

    if pair_scan:
        total = len(pair_scan) * len(seeds)
        print(
            f"[Sweep] paired mode: {len(pair_scan)} beta:gamma pairs x {len(seeds)} seeds = {total} runs"
        )
    else:
        total = len(betas) * len(gammas) * len(seeds)
        print(f"[Sweep] {len(betas)} betas x {len(gammas)} gammas x {len(seeds)} seeds = {total} runs")

    raw_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    # retain old rows to support resume + append semantics
    for r in existing_raw:
        raw_rows.append(dict(r))
    for r in existing_summary:
        summary_rows.append(dict(r))

    if pair_scan:
        jobs = [(beta, gamma, seed) for beta, gamma in pair_scan for seed in seeds]
    else:
        jobs = [(beta, gamma, seed) for beta in betas for gamma in gammas for seed in seeds]
    iterable = jobs
    pbar = None
    if not args.no_tqdm:
        pbar = tqdm(jobs, total=len(jobs), desc="phase-sweep", dynamic_ncols=True)
        iterable = pbar

    run_idx = 0
    for beta, gamma, seed in iterable:
        run_idx += 1
        if pbar is not None:
            pbar.set_postfix(beta=f"{beta:.3f}", gamma=f"{gamma:.3f}", seed=seed)
        key = run_key(beta, gamma, seed)
        if key in done_keys:
            print(f"[Skip {run_idx}/{total}] beta={beta:.3f}, gamma={gamma:.3f}, seed={seed}")
            continue

        msg = f"[Run  {run_idx}/{total}] beta={beta:.3f}, gamma={gamma:.3f}, seed={seed}"
        if pbar is not None:
            tqdm.write(msg)
        else:
            print(msg)

        result = train_one_model(
            beta=beta,
            gamma=gamma,
            seed=seed,
            config=cfg,
            device=device,
        )

        eval_by_len = result["eval_by_len"]
        train_len = int(cfg["train_len"])
        eval_lengths = [int(x) for x in cfg["eval_lengths"]]
        long_len = max(eval_lengths)

        if train_len not in eval_by_len:
            raise ValueError("train_len must be included in eval_lengths for gap metrics")

        train_acc = float(eval_by_len[train_len]["acc"])
        train_loss = float(eval_by_len[train_len]["loss"])
        long_acc = float(eval_by_len[long_len]["acc"])
        long_loss = float(eval_by_len[long_len]["loss"])
        gap = train_acc - long_acc
        retention = long_acc / max(train_acc, 1e-8)
        alpha_theory = gamma / (2.0 * max(beta, 1e-8))
        initial_train_step_loss = float(result.get("initial_train_step_loss", 0.0))
        final_train_step_loss = float(result.get("final_train_step_loss", 0.0))
        best_train_step_loss = float(result.get("best_train_step_loss", 0.0))
        aulc_train_to_final = float(result.get("aulc_train_to_final", 0.0))
        aulc_train_to_final_norm = float(result.get("aulc_train_to_final_norm", 0.0))
        objective = str(result.get("objective", "next_token_pretraining"))
        result_msg = (
            f"[Result] beta={beta:.3f}, gamma={gamma:.3f}, seed={seed} | "
            f"train_acc={train_acc:.4f}, long_acc={long_acc:.4f}, "
            f"gap={gap:.4f}, retention={retention:.4f}, alpha={alpha_theory:.4f}, "
            f"final_loss={final_train_step_loss:.4f}, aulc_to_final={aulc_train_to_final_norm:.4f}"
        )
        if pbar is not None:
            tqdm.write(result_msg)
        else:
            print(result_msg)

        summary_rows.append(
            {
                "beta": beta,
                "gamma": gamma,
                "seed": seed,
                "alpha_theory": alpha_theory,
                "train_len": train_len,
                "long_len": long_len,
                "train_acc": train_acc,
                "long_acc": long_acc,
                "train_loss": train_loss,
                "long_loss": long_loss,
                "generalization_gap": gap,
                "retention_ratio": retention,
                "objective": objective,
                "initial_train_step_loss": initial_train_step_loss,
                "final_train_step_loss": final_train_step_loss,
                "best_train_step_loss": best_train_step_loss,
                "aulc_train_to_final": aulc_train_to_final,
                "aulc_train_to_final_norm": aulc_train_to_final_norm,
                "train_time_sec": float(result["train_time_sec"]),
                "train_steps": int(cfg["train_steps"]),
                "d_model": int(cfg["d_model"]),
                "num_layers": int(cfg["num_layers"]),
            }
        )
        if "renyi_diag" in result and isinstance(result["renyi_diag"], dict):
            summary_rows[-1].update(result["renyi_diag"])

        for L in sorted(eval_by_len.keys()):
            raw_rows.append(
                {
                    "beta": beta,
                    "gamma": gamma,
                    "seed": seed,
                    "eval_len": int(L),
                    "acc": float(eval_by_len[L]["acc"]),
                    "loss": float(eval_by_len[L]["loss"]),
                    "alpha_theory": alpha_theory,
                    "train_steps": int(cfg["train_steps"]),
                    "d_model": int(cfg["d_model"]),
                    "num_layers": int(cfg["num_layers"]),
                }
            )

        # incremental checkpoint to avoid losing long sweeps
        write_csv(
            summary_csv,
            summary_rows,
            fieldnames=infer_fieldnames(
                summary_rows,
                preferred_prefix=[
                    "beta",
                    "gamma",
                    "seed",
                    "alpha_theory",
                    "train_len",
                    "long_len",
                    "train_acc",
                    "long_acc",
                    "train_loss",
                    "long_loss",
                    "generalization_gap",
                    "retention_ratio",
                    "objective",
                    "initial_train_step_loss",
                    "final_train_step_loss",
                    "best_train_step_loss",
                    "aulc_train_to_final",
                    "aulc_train_to_final_norm",
                    "train_time_sec",
                    "train_steps",
                    "d_model",
                    "num_layers",
                ],
            ),
        )
        write_csv(
            raw_csv,
            raw_rows,
            fieldnames=[
                "beta",
                "gamma",
                "seed",
                "eval_len",
                "acc",
                "loss",
                "alpha_theory",
                "train_steps",
                "d_model",
                "num_layers",
            ],
        )

    if pbar is not None:
        pbar.close()

    meta = {
        "betas": betas,
        "gammas": gammas,
        "beta_gamma_pairs": pair_scan,
        "seeds": seeds,
        "objective": "next_token_pretraining",
        "device": str(device),
        "config": cfg,
        "num_runs_total": total,
        "num_runs_recorded": len(summary_rows),
    }
    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n[Done] Sweep artifacts:")
    print(f"  - {raw_csv}")
    print(f"  - {summary_csv}")
    print(f"  - {meta_json}")


if __name__ == "__main__":
    main()
