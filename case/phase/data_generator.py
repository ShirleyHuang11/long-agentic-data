#!/usr/bin/env python3
"""Elegant (β, γ)-parametrized data generator and sweep plans for phase diagrams.

Two layers:

  1. ``PhaseDataGenerator`` — a friendly wrapper around
     ``phase_core.AlgorithmicKVGenerator``.  Re-uses the canonical generator
     so phase-diagram experiments stay consistent with training.

  2. Sweep plans — named, documented (β, γ) selections that probe the regimes
     a phase diagram needs:

         standard_grid()          7×7 log-β × linear-γ — full phase map
         boundary_corners()       4 extreme corners + center — sanity probes
         alpha_iso(α)             along γ/(2β)=α — the theoretical critical line
         beta_axis(γ) / gamma_axis(β)  1-D cross-sections
         fast_beta_limit(p)       β = γ^p with p>1 — near-origin probe
         refine_around(β,γ)       fine grid around a hot cell — boundary refinement
         recommended()            sensible union of the above (a "first sweep")

The CLI prints any plan, can emit a ``--beta-gamma-pairs`` string ready to
hand to ``phase_sweep.py``, and can dump a small sample batch from each
(β, γ) for visual sanity checks.

All configuration lives in ``configs/config.yaml``. Override on the CLI with
OmegaConf dotlist syntax (``key=value`` or ``nested.key=[1,2,3]``).

Examples
--------

    # See every plan and what (β, γ) it proposes
    python case/phase/data_generator.py plan.name=all

    # Build the canonical 7×7 grid as space-separated lists for the sweep CLI
    python case/phase/data_generator.py plan.name=standard data.as_cli=true

    # Explore the α=0.4 critical line, emit pair format
    python case/phase/data_generator.py plan.name=alpha_iso plan.alpha=0.4 data.as_pairs=true

    # Refinement grid around (β=0.4, γ=0.3), saved as JSON
    python case/phase/data_generator.py plan.name=refine plan.beta=0.4 plan.gamma=0.3 \\
        data.out=plots/refine_b0p4_g0p3.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_core import AlgorithmicKVGenerator
from utils import fmt_float, load_config, round_list, set_all_seeds


# -------------------- sweep plan ---------------------------------------------


@dataclass(frozen=True)
class SweepPlan:
    """A documented set of (β, γ) pairs."""

    name: str
    description: str
    pairs: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)

    @property
    def betas(self) -> List[float]:
        return sorted({float(b) for b, _ in self.pairs})

    @property
    def gammas(self) -> List[float]:
        return sorted({float(g) for _, g in self.pairs})

    def as_grid(self) -> Tuple[List[float], List[float]]:
        return self.betas, self.gammas

    def as_cli_grid(self) -> Tuple[str, str]:
        """Comma-separated `--betas` / `--gammas` strings for sweep CLI."""
        return (",".join(fmt_float(b) for b in self.betas),
                ",".join(fmt_float(g) for g in self.gammas))

    def as_pairs_arg(self) -> str:
        """`b1:g1,b2:g2,...` string for `--beta-gamma-pairs`."""
        return ",".join(f"{fmt_float(b)}:{fmt_float(g)}" for b, g in self.pairs)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "n_pairs": len(self.pairs),
            "betas": self.betas,
            "gammas": self.gammas,
            "pairs": [[float(b), float(g)] for b, g in self.pairs],
        }


# -------------------- factories ----------------------------------------------


def standard_grid(
    n_beta: int = 7,
    n_gamma: int = 7,
    beta_range: Tuple[float, float] = (0.1, 6.4),
    gamma_range: Tuple[float, float] = (0.05, 1.0),
) -> SweepPlan:
    """Default NeurIPS-scale 7×7 sweep: log-β × linear-γ.

    Produces the canonical full phase map.  β log-spaced because the relevant
    behaviour spans 2 decades; γ linear because it is naturally a fraction.
    """
    betas = round_list(np.geomspace(*beta_range, n_beta), 3)
    gammas = round_list(np.linspace(*gamma_range, n_gamma), 3)
    pairs = tuple((b, g) for b in betas for g in gammas)
    return SweepPlan(
        name="standard_grid",
        description=(f"{n_beta}×{n_gamma} log-β × linear-γ grid covering the "
                     f"full phase map. β∈{beta_range}, γ∈{gamma_range}."),
        pairs=pairs,
    )


def boundary_corners(
    beta_lo: float = 0.01,
    beta_hi: float = 8.0,
    gamma_lo: float = 0.02,
    gamma_hi: float = 0.95,
    beta_mid: float = 1.0,
    gamma_mid: float = 0.5,
) -> SweepPlan:
    """The 4 extreme regimes plus the center.  Cheap sanity probes."""
    pairs = (
        (beta_lo, gamma_lo),  # uniform long-range, near-noiseless: emergent regime
        (beta_lo, gamma_hi),  # uniform long-range, very noisy: chaos
        (beta_hi, gamma_lo),  # local-only, noiseless: easy / rote
        (beta_hi, gamma_hi),  # local-only, very noisy: rote-but-noisy
        (beta_mid, gamma_mid),  # canonical mid-grid reference
    )
    return SweepPlan(
        name="boundary_corners",
        description=("4 extreme corners + center. Quick probes for the four "
                     "qualitative regimes (chaos / emergent / rote / mid)."),
        pairs=pairs,
    )


def alpha_iso(
    alpha: float,
    n: int = 10,
    gamma_range: Tuple[float, float] = (0.02, 1.0),
) -> SweepPlan:
    """Sample (β, γ) along the iso-curve γ = 2αβ.

    α = γ/(2β) is the theoretical scaling parameter logged in run_summary.csv;
    sweeping along constant α traces the predicted critical line.  Useful for
    locating the *boundary* where phase transitions are sharpest.

    The β range is derived from ``gamma_range`` so every pair lands in the
    physical region γ ∈ (0, 1].
    """
    g_lo, g_hi = gamma_range
    a = max(float(alpha), 1e-6)
    b_lo, b_hi = g_lo / (2.0 * a), g_hi / (2.0 * a)
    betas = np.geomspace(b_lo, b_hi, n)
    gammas = 2.0 * a * betas
    pairs = tuple((round(float(b), 5), round(float(g), 4))
                  for b, g in zip(betas, gammas))
    return SweepPlan(
        name=f"alpha_iso_{alpha:g}",
        description=(f"{len(pairs)} points along γ=2αβ with α={alpha}. Traces "
                     "the theoretical critical line γ/(2β)=const for boundary mapping."),
        pairs=pairs,
    )


def beta_axis(
    gamma: float,
    n: int = 8,
    beta_range: Tuple[float, float] = (0.05, 6.4),
) -> SweepPlan:
    """1-D cross-section: vary β at fixed γ."""
    betas = round_list(np.geomspace(*beta_range, n), 4)
    pairs = tuple((b, gamma) for b in betas)
    return SweepPlan(
        name=f"beta_axis_g{gamma:g}",
        description=f"β log-sweep at fixed γ={gamma}. Use with a γ near the "
                    "phase boundary (e.g. 0.30) to localise the β-transition.",
        pairs=pairs,
    )


def gamma_axis(
    beta: float,
    n: int = 8,
    gamma_range: Tuple[float, float] = (0.02, 1.0),
) -> SweepPlan:
    """1-D cross-section: vary γ at fixed β."""
    gammas = round_list(np.linspace(*gamma_range, n), 4)
    pairs = tuple((beta, g) for g in gammas)
    return SweepPlan(
        name=f"gamma_axis_b{beta:g}",
        description=f"γ linear-sweep at fixed β={beta}. Use with a β near the "
                    "boundary to localise the γ-transition.",
        pairs=pairs,
    )


def fast_beta_limit(
    p: float = 2.0,
    n: int = 8,
    gamma_range: Tuple[float, float] = (0.05, 0.6),
) -> SweepPlan:
    """β = γ^p with p>1 — explore near-origin behaviour.

    Forces β to decay faster than γ as both go to 0.  Mirrors the existing
    ``phase_limit_fastbeta`` experiment.

    γ_lo defaults to 0.05 (not 0): with p=3 a γ=0.01 would yield β=1e-6,
    which is degenerate (essentially zero long-range decay) and breaks
    diagnostics. 0.05 keeps β in the numerically meaningful range for p ≤ 3.
    """
    gammas = np.geomspace(*gamma_range, n)
    betas = np.power(gammas, p)
    pairs = tuple((round(float(b), 5), round(float(g), 4))
                  for b, g in zip(betas, gammas))
    return SweepPlan(
        name=f"fast_beta_p{p:g}",
        description=(f"β = γ^{p} with p>1 — near-origin probe. {n} points "
                     "geometrically spaced in γ. Tests behaviour as both knobs vanish."),
        pairs=pairs,
    )


def refine_around(
    beta: float,
    gamma: float,
    factors: Sequence[float] = (0.70, 0.85, 1.00, 1.15, 1.30),
) -> SweepPlan:
    """Fine grid centred on a single point — for sharpening a boundary."""
    pairs = tuple(
        (round(beta * fb, 5), round(gamma * fg, 5))
        for fb in factors for fg in factors
    )
    return SweepPlan(
        name=f"refine_b{beta:g}_g{gamma:g}",
        description=(f"{len(factors)}×{len(factors)} multiplicative refinement "
                     f"grid around (β={beta}, γ={gamma}). Use after the standard "
                     "grid identifies an interesting cell."),
        pairs=pairs,
    )


def recommended() -> List[SweepPlan]:
    """Curated set: full grid + corners + α-iso lines + a near-origin probe.

    A reasonable "first sweep" that yields a populated phase diagram **and**
    extra resolution on the boundary regions.
    """
    plans: List[SweepPlan] = [
        standard_grid(),
        boundary_corners(),
        alpha_iso(0.10),
        alpha_iso(0.40),
        alpha_iso(1.00),
        fast_beta_limit(p=2.0),
    ]
    return plans


# -------------------- generator ----------------------------------------------


class PhaseDataGenerator:
    """Friendly façade over phase_core.AlgorithmicKVGenerator.

    Usage::

        gen = PhaseDataGenerator(vocab_size=60, seed=1)
        x, y, m = gen.sample(beta=0.4, gamma=0.3, batch_size=8, seq_len=128)

        for beta, gamma, x, y, m in gen.iter_plan(standard_grid(), batch=4, seq_len=64):
            ...
    """

    def __init__(self, vocab_size: int = 60, seed: int | None = None) -> None:
        self.vocab_size = int(vocab_size)
        self.gen = AlgorithmicKVGenerator(vocab_size=self.vocab_size)
        self.seed = seed
        if seed is not None:
            set_all_seeds(int(seed))

    def sample(self, beta: float, gamma: float, batch_size: int, seq_len: int):
        return self.gen.generate_batch(
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            beta=float(beta),
            gamma=float(gamma),
        )

    def iter_plan(self, plan: SweepPlan, batch_size: int, seq_len: int):
        for b, g in plan.pairs:
            x, y, m = self.sample(b, g, batch_size, seq_len)
            yield float(b), float(g), x, y, m

    def diagnose(self, beta: float, gamma: float, seq_len: int = 1024) -> dict:
        """Cheap statistics from a single long sample — useful for unit-checking
        a (β, γ) before paying for training.
        """
        x, _, m = self.sample(beta, gamma, batch_size=1, seq_len=seq_len)
        tokens = x[0].cpu().numpy()
        noise_ids = np.arange(1, 20)
        key_ids = np.arange(20, 40)
        value_ids = np.arange(40, 60)
        n_total = len(tokens)
        return {
            "beta": float(beta),
            "gamma": float(gamma),
            "alpha_theory": float(gamma) / (2.0 * max(float(beta), 1e-12)),
            "seq_len": int(n_total),
            "frac_noise": float(np.isin(tokens, noise_ids).mean()),
            "frac_keys": float(np.isin(tokens, key_ids).mean()),
            "frac_values": float(np.isin(tokens, value_ids).mean()),
            "frac_retrieval_targets": float(m[0].cpu().numpy().mean()),
            "unique_tokens": int(len(set(tokens.tolist()))),
        }


# -------------------- CLI ----------------------------------------------------


def _standard_from_cfg(c) -> SweepPlan:
    """Build standard_grid honoring optional plan.n_beta/n_gamma/beta_range/
    gamma_range overrides. Defaults match the original 7×7 grid."""
    kwargs = {}
    if hasattr(c, "n_beta") and c.get("n_beta") is not None:
        kwargs["n_beta"] = int(c.n_beta)
    if hasattr(c, "n_gamma") and c.get("n_gamma") is not None:
        kwargs["n_gamma"] = int(c.n_gamma)
    if hasattr(c, "beta_range") and c.get("beta_range") is not None:
        kwargs["beta_range"] = tuple(float(x) for x in c.beta_range)
    if hasattr(c, "gamma_range") and c.get("gamma_range") is not None:
        kwargs["gamma_range"] = tuple(float(x) for x in c.gamma_range)
    return standard_grid(**kwargs)


_FACTORIES = {
    "standard": _standard_from_cfg,
    "corners": lambda c: boundary_corners(),
    "alpha_iso": lambda c: alpha_iso(c.alpha, n=c.n),
    "beta_axis": lambda c: beta_axis(c.gamma, n=c.n),
    "gamma_axis": lambda c: gamma_axis(c.beta, n=c.n),
    "fast_beta": lambda c: fast_beta_limit(p=c.p, n=c.n),
    "refine": lambda c: refine_around(c.beta, c.gamma),
}


def _resolve_plans(plan_cfg) -> List[SweepPlan]:
    name = plan_cfg.name
    if name == "all":
        return [
            standard_grid(),
            boundary_corners(),
            alpha_iso(0.1), alpha_iso(0.4), alpha_iso(1.0),
            beta_axis(plan_cfg.gamma, n=plan_cfg.n),
            gamma_axis(plan_cfg.beta, n=plan_cfg.n),
            fast_beta_limit(p=plan_cfg.p, n=plan_cfg.n),
            refine_around(plan_cfg.beta, plan_cfg.gamma),
        ]
    if name == "recommended":
        return recommended()
    return [_FACTORIES[name](plan_cfg)]


def _print_plan(plan: SweepPlan, data_cfg) -> None:
    print(f"\n=== {plan.name} ({len(plan.pairs)} pairs) ===")
    print(plan.description)
    print(f"  βs ({len(plan.betas)}): {plan.betas}")
    print(f"  γs ({len(plan.gammas)}): {plan.gammas}")

    if data_cfg.as_cli:
        b_str, g_str = plan.as_cli_grid()
        print(f"  --betas {b_str}")
        print(f"  --gammas {g_str}")
    if data_cfg.as_pairs:
        print(f"  --beta-gamma-pairs {plan.as_pairs_arg()}")

    if data_cfg.sample:
        gen = PhaseDataGenerator(seed=int(data_cfg.seed))
        print(f"  per-pair stats (seq_len={int(data_cfg.seq_len)}, "
              f"seed={int(data_cfg.seed)}):")
        print("    {:>8} {:>8} {:>8}  {:>6} {:>6} {:>6} {:>8}".format(
            "beta", "gamma", "alpha", "noise", "keys", "vals", "targets"))
        for b, g in plan.pairs:
            d = gen.diagnose(b, g, seq_len=int(data_cfg.seq_len))
            print("    {beta:8.4g} {gamma:8.4g} {alpha_theory:8.4g}  "
                  "{frac_noise:6.3f} {frac_keys:6.3f} {frac_values:6.3f} "
                  "{frac_retrieval_targets:8.4f}".format(**d))


def main() -> None:
    cfg = load_config(sys.argv[1:])
    plans = _resolve_plans(cfg.plan)
    for p in plans:
        _print_plan(p, cfg.data)

    if cfg.data.out is not None:
        out = Path(str(cfg.data.out))
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = [p.to_dict() for p in plans]
        out.write_text(json.dumps(payload, indent=2))
        print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    main()
