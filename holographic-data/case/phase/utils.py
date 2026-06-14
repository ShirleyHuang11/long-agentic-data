#!/usr/bin/env python3
"""Misc utilities shared across phase-diagram tooling.

Pulled out of phase_core / data_generator / plot_phase_diagram / phase_sweep
so each entry-point imports from one place. Sections:

  - config:         load_config (OmegaConf YAML + dotlist CLI overrides)
  - env:            set_all_seeds, choose_device
  - formatting:     fmt_float, round_list
  - numeric:        detect_scale
  - 2-D plotting:   cell_edges, build_grid
  - CSV I/O:        write_csv
  - phase taxonomy: PHASE_NAMES, PHASE_COLORS, phase_name,
                    classify_fixed, classify_quantile
  - column choice:  pick_aulc_column
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"
MODELS_DIR = CONFIG_PATH.parent / "models"


def load_config(argv: Optional[Sequence[str]] = None):
    """Load configs/default.yaml, resolve `model_preset`, merge CLI overrides.

    Resolution order:
      1. Load default.yaml
      2. Apply `model_preset=...` from argv if present (so the right preset
         file is selected before its defaults are pulled in)
      3. Load configs/models/<model_preset>.yaml into cfg.model
      4. Apply the full CLI dotlist on top (so `model.train_steps=10000`
         overrides the preset's value for one run)

    Example::

        cfg = load_config(sys.argv[1:])
        print(cfg.plan.name, cfg.model.d_model, cfg.sweep.out_dir)
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(CONFIG_PATH)

    # Stage 1: pull a leading model_preset override out of argv so we know
    # which preset file to load before merging the rest.
    argv_list = list(argv or ())
    for token in argv_list:
        if token.startswith("model_preset="):
            cfg.model_preset = token.split("=", 1)[1]
            break

    preset = cfg.get("model_preset")
    if preset:
        preset_path = MODELS_DIR / f"{preset}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(
                f"model preset {preset!r} not found at {preset_path}"
            )
        cfg.model = OmegaConf.load(preset_path)

    # Stage 2: apply all CLI overrides on top (this lets users tweak preset
    # fields with `model.lr=1e-4` style overrides).
    if argv_list:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(argv_list))

    return cfg


def set_all_seeds(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and (if importable) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def choose_device(device_arg: str):
    """Resolve --device {cpu|cuda|auto} → torch.device."""
    import torch
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- formatting ---------------------------------------------


def fmt_float(x: float) -> str:
    """Compact, lossless-enough float formatting for CLI strings."""
    if x == 0:
        return "0"
    if abs(x) >= 0.01:
        return f"{x:.4g}"
    return f"{x:.3e}"


def round_list(values: Iterable[float], decimals: int = 4) -> List[float]:
    return [round(float(v), decimals) for v in values]


# -------------------- numeric ------------------------------------------------


def detect_scale(values: Sequence[float]) -> str:
    """Infer 'log' vs 'linear' spacing from a 1-D array of axis values."""
    v = np.asarray(sorted(set(float(x) for x in values)), dtype=float)
    if len(v) < 3 or v.min() <= 0:
        return "linear"
    if v.max() / v.min() >= 10.0:
        return "log"
    diffs_log = np.diff(np.log(v))
    diffs_lin = np.diff(v)
    cv_log = float(np.std(diffs_log) / max(abs(np.mean(diffs_log)), 1e-12))
    cv_lin = float(np.std(diffs_lin) / max(abs(np.mean(diffs_lin)), 1e-12))
    return "log" if cv_log < cv_lin else "linear"


# -------------------- 2-D plotting -------------------------------------------


def cell_edges(values: Sequence[float], scale: str) -> np.ndarray:
    """Cell-boundary positions for pcolormesh in the given scale.

    Midpoints in the appropriate scale (log or linear), with the outermost
    edges reflected so the edge cells span the same width as their neighbour.
    """
    v = np.asarray(sorted(set(float(x) for x in values)), dtype=float)
    if len(v) == 1:
        pad = max(abs(v[0]) * 0.1, 1e-3)
        return np.array([v[0] - pad, v[0] + pad])
    if scale == "log":
        log_v = np.log(v)
        mids = 0.5 * (log_v[:-1] + log_v[1:])
        first = log_v[0] - (mids[0] - log_v[0])
        last = log_v[-1] + (log_v[-1] - mids[-1])
        return np.exp(np.concatenate([[first], mids, [last]]))
    mids = 0.5 * (v[:-1] + v[1:])
    first = v[0] - (mids[0] - v[0])
    last = v[-1] + (v[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def build_grid(agg: Any, key: str,
               betas: Sequence[float], gammas: Sequence[float]) -> np.ndarray:
    """Index a (β, γ) aggregation DataFrame into a (Ngamma, Nbeta) grid."""
    z = np.full((len(gammas), len(betas)), np.nan, dtype=float)
    b_idx = {float(b): i for i, b in enumerate(betas)}
    g_idx = {float(g): i for i, g in enumerate(gammas)}
    for _, r in agg.iterrows():
        b = float(r["beta"])
        g = float(r["gamma"])
        if b in b_idx and g in g_idx:
            z[g_idx[g], b_idx[b]] = float(r[key])
    return z


# -------------------- CSV I/O ------------------------------------------------


def write_csv(path: Path, rows: List[Dict[str, object]],
              preferred: Sequence[str]) -> None:
    """Write rows with `preferred` columns first, then any extra keys."""
    seen = set(preferred)
    extras: List[str] = []
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


# -------------------- phase taxonomy -----------------------------------------


PHASE_NAMES: Dict[int, str] = {
    0: "Chaos / Saturation",
    1: "Emergent",
    2: "Super-Generalization",
    3: "Rote Memorization",
}
PHASE_COLORS: Dict[int, str] = {
    0: "#9aa0a6",
    1: "#7fbf7b",
    2: "#1f78b4",
    3: "#e08171",
}


def phase_name(code: int) -> str:
    return PHASE_NAMES[int(code)]


def classify_fixed(row: Any, *,
                   chaos_train: float = 0.20,
                   chaos_long: float = 0.10,
                   rote_train: float = 0.40,
                   rote_gap: float = 0.10,
                   super_long: float = 0.50,
                   super_ret: float = 0.85) -> int:
    """Threshold-based 4-phase classification of an aggregated cell."""
    if row["train_acc_mean"] < chaos_train and row["long_acc_mean"] < chaos_long:
        return 0
    if row["train_acc_mean"] >= rote_train and row["gap_mean"] >= rote_gap:
        return 3
    if row["long_acc_mean"] >= super_long and row["retention_mean"] >= super_ret:
        return 2
    return 1


def classify_quantile(df: Any) -> List[int]:
    """Quantile-based 4-phase classification of an aggregation DataFrame."""
    q_train_low = df["train_acc_mean"].quantile(0.25)
    q_train_high = df["train_acc_mean"].quantile(0.70)
    q_long_med = df["long_acc_mean"].quantile(0.50)
    q_long_high = df["long_acc_mean"].quantile(0.75)
    q_gap_med = df["gap_mean"].quantile(0.50)
    q_gap_high = df["gap_mean"].quantile(0.75)
    q_ret_med = df["retention_mean"].quantile(0.50)

    out: List[int] = []
    for _, r in df.iterrows():
        if r["train_acc_mean"] <= q_train_low and r["long_acc_mean"] <= q_long_med:
            out.append(0)
        elif r["train_acc_mean"] >= q_train_high and r["gap_mean"] >= q_gap_high:
            out.append(3)
        elif (r["long_acc_mean"] >= q_long_high
              and r["gap_mean"] <= q_gap_med
              and r["retention_mean"] >= q_ret_med):
            out.append(2)
        else:
            out.append(1)
    return out


# -------------------- column selection ---------------------------------------


def pick_aulc_column(agg: Any) -> str:
    """Find the preferred AULC mean column in an aggregation DataFrame."""
    candidates = [c for c in agg.columns
                  if c.startswith("aulc_") and c.endswith("_mean")]
    for pref in ("aulc_train_to_final_norm_mean", "aulc_train_to_final_mean"):
        if pref in candidates:
            return pref
    return candidates[0] if candidates else ""
