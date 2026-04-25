#!/usr/bin/env bash
#SBATCH -J phase-report
#SBATCH -o case/phase/logs/%x-%j.out
#SBATCH -e case/phase/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=shared
#SBATCH --account=barak_lab
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Aggregate report job. Submitted with `--begin=now+6hours` (and optionally a
# dependency string on the sweep job IDs) by case/phase/sweep.sh's report-only
# entry. Reads case/phase/runs/*/run_summary.csv and writes
# case/phase/REPORT_sweep_summary.md.

set -euo pipefail

cd /n/home07/hanlinzhang/projects/holographic-data
mkdir -p case/phase/logs

# shellcheck disable=SC1091
source /n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/flow/bin/activate

echo "[SLURM] job_id=${SLURM_JOB_ID}  start=$(date -Is)"
echo "[run]   tracking sweep jobs: ${TRACKED_JOBS:-(none)}"

python case/phase/aggregate_report.py ${TRACKED_JOBS:-}

echo "[SLURM] end=$(date -Is)  done"
