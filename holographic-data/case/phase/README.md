# Phase-diagram sweeps for (β, γ)-parametrized algorithmic data

Train a 100M-scale causal Transformer on a key-value retrieval task with
controllable long-range access decay (β) and noise fraction (γ), sweep across
a grid of (β, γ), and produce a phase diagram that classifies each cell as
*chaos / emergent / super-generalisation / rote-memorisation*.

## Layout

```
case/phase/
├── configs/
│   ├── default.yaml          # plan / sweep / data / plot + model_preset
│   └── models/
│       └── 100m.yaml         # ~100.8M non-embed preset (loaded by default)
├── model.py                  # TinyCausalTransformer + 100M-param floor
├── phase_core.py             # AlgorithmicKVGenerator + train_one_model + Rényi
├── data_generator.py         # named (β, γ) sweep plans + CLI for plan inspection
├── phase_sweep.py            # entry point: train models for one plan, write CSVs, auto-plot
├── plot_phase_diagram.py     # 2-D heatmap + 4-phase classification + α iso-lines
├── aggregate_report.py       # roll all runs/<variant>/run_summary.csv into one Markdown report
├── utils.py                  # OmegaConf load_config, phase taxonomy, CSV/grid helpers
├── sweep.sh                  # submit one independent slurm job per variant
├── sweep.slurm               # generic GPU runner used by sweep.sh
├── report.slurm              # CPU partition; runs aggregate_report.py
├── runs/<variant>/           # output CSVs + sweep_meta.json + phase_diagram_*.png
└── logs/                     # slurm stdout / stderr per job
```

## Configs (OmegaConf)

Single hierarchical YAML with model presets pulled in by name. Override
anything on the CLI with dotlist syntax — no argparse anywhere.

```yaml
# configs/default.yaml
plan:    { name: standard, alpha, beta, gamma, p, n }
sweep:   { seeds, out_dir, device, long_len, no_plot, no_tqdm }
model_preset: 100m            # ⟶ configs/models/100m.yaml is merged into cfg.model
data:    { ... }              # data_generator.py CLI extras
plot:    { in_summary, out_dir, phase_mode, thresholds, ... }
```

`sweep.out_dir` defaults to `case/phase/runs/${plan.name}` via OmegaConf
interpolation; `plot.in_summary` and `plot.out_dir` cascade off
`sweep.out_dir`. So overriding `plan.name` reroutes everything automatically.

To swap model size, drop a new `configs/models/<name>.yaml` and pass
`model_preset=<name>`. To tweak one field, pass `model.train_steps=10000`.

## Quick start

```bash
source /n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/flow/bin/activate

# 1. inspect a sweep plan without training
python case/phase/data_generator.py plan.name=corners data.as_pairs=true

# 2. run one sweep locally (auto-plots to the same out_dir)
python case/phase/phase_sweep.py plan.name=alpha_iso plan.alpha=0.4 \
    sweep.device=cuda

# 3. submit ALL variants to slurm (one independent job each)
bash case/phase/sweep.sh

# 4. dry-run mode to see what would be submitted
DRY_RUN=1 bash case/phase/sweep.sh
```

## Sweep variants (sweep.sh)

| variant            | what it covers                                      | runs (3 seeds) |
|--------------------|-----------------------------------------------------|---------------:|
| `standard`         | full 7×7 log-β × linear-γ grid                      | 147            |
| `corners`          | 4 extreme regimes + center                          | 15             |
| `alpha_iso_{0p1, 0p4, 1p0}` | critical line γ = 2αβ at three α values    | 36 each        |
| `beta_axis_g0p3`   | 1-D scan of β at fixed γ=0.3                        | 36             |
| `gamma_axis_b0p4`  | 1-D scan of γ at fixed β=0.4                        | 36             |
| `fast_beta_p{2, 3}`| near-origin limit β = γ^p                           | 36 each        |
| `refine_*`         | 5×5 refinement grid around a hot cell               | 75 each        |

Override seeds globally with `SEEDS='[1,2,3,4,5]' bash case/phase/sweep.sh`.
Submit just one variant with `bash case/phase/sweep.sh standard`.

## Slurm

`sweep.slurm` requests 1× A100, 16 CPU, 64G RAM, 2-day walltime on
`seas_gpu` / `barak_lab`. `sweep.sh` exports `RUN_NAME` and `OVERRIDES` per
variant. CSVs are flushed after every (β, γ, seed), so any timeout still
leaves a usable partial dataset.

`report.slurm` runs `aggregate_report.py` on the `shared` partition (no GPU,
30 min, 16G). Submit with both a 6h begin-time and a `--dependency=afterany`
on all sweep job IDs:

```bash
sbatch \
  --begin=now+6hours \
  --dependency=afterany:8233498:8233499:...:8233508 \
  --export=ALL,TRACKED_JOBS="8233498 8233499 ... 8233508" \
  case/phase/report.slurm
```

The two conditions are AND'd — the report fires when ≥6h have passed AND
all sweeps have settled. It writes
`case/phase/REPORT_sweep_summary.md` with sections for inventory + the
standard grid + critical-line scans + near-origin limit + refine cells +
flagged unusual cells, embedding the auto-produced
`phase_diagram_*.png` figures from each variant.

## Phase classification

Per-cell aggregation of seeds → fixed-threshold 4-phase code:

| code | name                  | rule                                            |
|-----:|-----------------------|-------------------------------------------------|
| 0    | chaos / saturation    | `train_acc<0.20` & `long_acc<0.10`              |
| 3    | rote memorisation     | `train_acc≥0.40` & `gap≥0.10`                   |
| 2    | super-generalisation  | `long_acc≥0.50` & `retention≥0.85`              |
| 1    | emergent              | otherwise                                       |

Thresholds live under `plot.thresholds` in `default.yaml` and can be
overridden per-run. Use `plot.phase_mode=quantile` for a data-driven
classifier instead.

## Decoupled model file

`model.py` is intentionally minimal — it only imports `torch` and exposes
`TinyCausalTransformer`, `count_non_embedding_params`, and the 100M floor
constant. Swap it for an alternative architecture by editing this file
alone; `phase_core.py` re-exports the same names for backwards
compatibility, and `phase_sweep.py` imports them directly from `model`.

## Outputs

Each `runs/<variant>/` contains:

```
run_summary.csv               # per-(β, γ, seed) train/long metrics + Rényi diag
raw_metrics.csv               # per-eval-length acc/loss
sweep_meta.json               # exact config + plan metadata
phase_diagram_long_acc.png    # main heatmap with α iso-lines + boundary contour
phase_diagram_panels.png      # 2×2: long_acc | train_acc | gap | AULC
phase_diagram_classified.png  # 4-phase categorical map
phase_summary.csv             # per-cell aggregation used for the figures
```

`REPORT_sweep_summary.md` (top-level) cross-references every variant and
links the figures.

## Common overrides

```bash
# smaller model (must clear the 100M floor — try a different preset instead)
python case/phase/phase_sweep.py model_preset=200m sweep.device=cuda

# tweak training without changing the preset
python case/phase/phase_sweep.py model.train_steps=20000 model.lr=1e-4

# evaluate at exactly {train_len, long_len}
python case/phase/phase_sweep.py sweep.long_len=4096

# explicit out_dir, skip the auto-plot
python case/phase/phase_sweep.py sweep.out_dir=plots/ad_hoc sweep.no_plot=true
```
