#!/usr/bin/env bash
#SBATCH -J phase-adaptive-pareto
#SBATCH -o case/phase/logs/phase-adaptive-pareto-%j.out
#SBATCH -e case/phase/logs/phase-adaptive-pareto-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=seas_gpu
#SBATCH --account=barak_lab
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

cd /n/home07/hanlinzhang/projects/holographic-data
mkdir -p case/phase/logs

echo "[SLURM] job_id=${SLURM_JOB_ID}"
echo "[SLURM] host=$(hostname)"
echo "[SLURM] start_time=$(date -Is)"
echo "[SLURM] cwd=$(pwd)"
echo "[SLURM] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

export WORK_OUT_DIR="${WORK_OUT_DIR:-plots/phase_adaptive_pareto_slurm_${SLURM_JOB_ID}}"
export NO_TQDM="${NO_TQDM:-1}"

echo "[Run] WORK_OUT_DIR=${WORK_OUT_DIR}"
echo "[Run] BASE_OUT_DIR=${BASE_OUT_DIR:-default}"
echo "[Run] REFINE_OUT_DIR=${REFINE_OUT_DIR:-default}"
echo "[Run] MERGED_OUT_DIR=${MERGED_OUT_DIR:-default}"
echo "[Run] ANALYSIS_OUT_DIR=${ANALYSIS_OUT_DIR:-default}"
echo "[Run] TOP_K=${TOP_K:-default}"
echo "[Run] MAX_PAIRS=${MAX_PAIRS:-default}"
echo "[Run] SEEDS=${SEEDS:-default}"
echo "[Run] CONFIG_JSON=${CONFIG_JSON:-default}"

bash case/phase/phase_adaptive_pareto.sh "${WORK_OUT_DIR}"

echo "[SLURM] end_time=$(date -Is)"
echo "[SLURM] done"
