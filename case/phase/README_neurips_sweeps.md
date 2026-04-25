# NeurIPS-Scale Beta-Gamma Sweeps

This folder provides a paper-grade pipeline for large sweeps on algorithmic reasoning:

1. `neurips_phase_sweep.py`: run multi-seed grid sweeps.
2. `neurips_phase_analysis.py`: aggregate statistics, confidence intervals, phase maps, and regression significance.
3. `phase_core.py`: shared generator/model/train/eval logic.

The sweep now also computes **generalized Rényi dimensions** (`renyi_D_rate_q*`) for each `(beta,gamma,seed)` dataset sample and carries them into analysis/reporting.
Training objective is **causal next-token prediction pre-training** (`objective=next_token_pretraining` in `run_summary.csv`).
It also logs **AULC metrics** from the training curve:
- `aulc_train_to_final`: area between the training-loss curve and `y=final_train_step_loss`
- `aulc_train_to_final_norm`: the same area normalized by number of training steps

## 1) Smoke Test (end-to-end)

```bash
source /n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/flow/bin/activate

python case/phase/neurips_phase_sweep.py \
  --config-json case/phase/configs/smoke.json \
  --out-dir plots/phase_neurips_smoke \
  --betas 0.2,1.6 \
  --gammas 0.1,0.6 \
  --seeds 1,2 \
  --device cpu

python case/phase/neurips_phase_analysis.py \
  --in-summary plots/phase_neurips_smoke/run_summary.csv \
  --out-dir plots/phase_neurips_smoke_analysis \
  --phase-mode quantile \
  --permutation-trials 100
```

## 2) Main NeurIPS Sweep (recommended)

### Primary model

```bash
source /n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/flow/bin/activate

python case/phase/neurips_phase_sweep.py \
  --config-json case/phase/configs/neurips_main_gpu.json \
  --out-dir plots/phase_neurips_main \
  --betas 0.1,0.2,0.4,0.8,1.6,3.2,6.4 \
  --gammas 0.05,0.15,0.30,0.45,0.60,0.80,1.00 \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --device cuda \
  --resume
```

### Aggregation + significance

```bash
python case/phase/neurips_phase_analysis.py \
  --in-summary plots/phase_neurips_main/run_summary.csv \
  --out-dir plots/phase_neurips_main_analysis \
  --phase-mode fixed \
  --permutation-trials 500
```

Optional Rényi overrides in sweep:

```bash
  --renyi-seq-len 20000 \
  --renyi-orders 1,2,3,4 \
  --renyi-qs 0.5,1.0,2.0
```

Optional paired scan mode (no cartesian product):

```bash
python case/phase/neurips_phase_sweep.py \
  --config-json case/phase/configs/neurips_main_gpu.json \
  --out-dir plots/phase_neurips_pairs \
  --beta-gamma-pairs 0.02:0.05,0.01:0.03,0.005:0.02 \
  --seeds 1,2,3 \
  --device cuda \
  --resume
```

## 3) Sharded Runs and Merge

For large clusters, run seed shards in parallel by changing `--seeds` and `--out-dir` per shard, then merge:

```bash
python case/phase/neurips_phase_merge.py \
  --inputs plots/phase_neurips_main_s1_3,plots/phase_neurips_main_s4_6,plots/phase_neurips_main_s7_10 \
  --out-dir plots/phase_neurips_main_merged
```

Then analyze `plots/phase_neurips_main_merged/run_summary.csv`.

## 4) Ablation (smaller model)

```bash
python case/phase/neurips_phase_sweep.py \
  --config-json case/phase/configs/neurips_ablation_smallmodel.json \
  --out-dir plots/phase_neurips_ablation_small \
  --betas 0.1,0.2,0.4,0.8,1.6,3.2,6.4 \
  --gammas 0.05,0.15,0.30,0.45,0.60,0.80,1.00 \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --device cuda \
  --resume
```

Then run the same analysis script with `--in-summary plots/phase_neurips_ablation_small/run_summary.csv`.

## 5) Limit Experiment (beta,gamma -> 0 with beta decaying faster)

This experiment enforces `beta = gamma^p` with `p>1` and studies near-origin behavior.

```bash
bash case/phase/phase_limit_fastbeta.sh plots/phase_limit_fastbeta
```

Slurm:

```bash
sbatch case/phase/phase_limit_fastbeta.slurm
```

Artifacts include:
- `limit_subset_runs.csv`
- `limit_subset_cell_stats.csv`
- `limit_subset_gamma_trend.png`
- plus standard phase outputs and AULC heatmaps

## 6) One-Job Adaptive Refinement + Pareto

This pipeline performs:
1. adaptive local refinement around top coarse-grid cells (by long_acc + AULC + low gap/loss),
2. merge with base sweep,
3. analysis + Pareto frontier extraction.

```bash
sbatch --export=ALL,BASE_OUT_DIR=plots/phase_neurips_main_slurm_3474331 case/phase/phase_adaptive_pareto.slurm
```

Optional knobs:
- `TOP_K` (default 12)
- `MAX_PAIRS` (default 140)
- `SEEDS` (default `4,5,6`)
- `BETA_FACTORS`, `GAMMA_FACTORS` (default `0.70,0.85,1.00,1.15,1.30`)

## Output Artifacts

From sweep:
- `raw_metrics.csv`: per `(beta,gamma,seed,eval_len)` metrics.
- `run_summary.csv`: per `(beta,gamma,seed)` train/long summary.
- `sweep_meta.json`: exact config + grid metadata.
  - includes AULC + Rényi diagnostics columns

From analysis:
- `aggregated_cell_stats.csv`
- `regression_coefficients.csv`
- `phase_counts.csv`
- `analysis_report.md`
- `heatmap_long_acc_mean.png`
- `heatmap_long_acc_ci95.png`
- `heatmap_gap_mean.png`
- `heatmap_aulc_*_mean.png` (if AULC columns exist)
- `phase_diagram_aggregated.png`
- `scatter_alpha_vs_long_acc.png`
- `heatmap_renyi_D_rate_q*_mean.png`
- `heatmap_renyi_D_rate_clip_q*_mean.png`
- `pareto_cells_scored.csv` / `pareto_front_cells.csv` / `pareto_longacc_vs_aulc.png`

## Notes for Credibility

- Use at least 8-10 seeds per cell for stable CI95.
- Keep `train_len` included in `eval_lengths` to compute generalization gap consistently.
- Report both run-level and cell-level correlations.
- Keep `sweep_meta.json` and commit exact command lines for reproducibility.
