#!/usr/bin/env bash
#SBATCH -J phase-limit-fastbeta
#SBATCH -o case/phase/logs/phase-limit-fastbeta-%j.out
#SBATCH -e case/phase/logs/phase-limit-fastbeta-%j.err
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

export OUT_DIR="${OUT_DIR:-plots/phase_limit_fastbeta_slurm_${SLURM_JOB_ID}}"
export NO_TQDM="${NO_TQDM:-1}"

echo "[Run] OUT_DIR=${OUT_DIR}"
echo "[Run] GAMMAS=${GAMMAS:-default}"
echo "[Run] FAST_POWERS=${FAST_POWERS:-default}"
echo "[Run] SEEDS=${SEEDS:-default}"
echo "[Run] DEVICE=${DEVICE:-default}"
echo "[Run] CONFIG_JSON=${CONFIG_JSON:-default}"
echo "[Run] LIMIT_BETA_OVER_GAMMA_MAX=${LIMIT_BETA_OVER_GAMMA_MAX:-default}"

bash case/phase/phase_limit_fastbeta.sh "${OUT_DIR}"

echo "[SLURM] end_time=$(date -Is)"
echo "[SLURM] done"
