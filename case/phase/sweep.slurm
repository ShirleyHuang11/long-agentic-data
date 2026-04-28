#!/usr/bin/env bash
#SBATCH -J phase-sweep
#SBATCH -o case/phase/logs/%x-%j.out
#SBATCH -e case/phase/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=seas_gpu
#SBATCH --account=barak_lab
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=64G
#SBATCH --time=2-00:00

# Generic phase_sweep.py runner. Submit via case/phase/sweep.sh, which sets
# RUN_NAME and OVERRIDES for each sweep variant. Direct invocation:
#
#   sbatch --export=ALL,RUN_NAME=alpha04,OVERRIDES="plan.name=alpha_iso plan.alpha=0.4" \
#       case/phase/sweep.slurm

set -euo pipefail

cd /n/home07/hanlinzhang/projects/holographic-data
mkdir -p case/phase/logs case/phase/runs

# shellcheck disable=SC1091
source /n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/flow/bin/activate

RUN_NAME="${RUN_NAME:-sweep_${SLURM_JOB_ID}}"
if [ -n "${OVERRIDES_B64:-}" ]; then
    OVERRIDES="$(printf '%s' "${OVERRIDES_B64}" | base64 -d)"
else
    OVERRIDES="${OVERRIDES:-}"
fi
OUT_DIR="${OUT_DIR:-case/phase/runs/${RUN_NAME}}"

echo "[SLURM] job_id=${SLURM_JOB_ID}  host=$(hostname)"
echo "[SLURM] start_time=$(date -Is)"
echo "[run]   RUN_NAME=${RUN_NAME}"
echo "[run]   OUT_DIR=${OUT_DIR}"
echo "[run]   OVERRIDES=${OVERRIDES}"

if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi || true; fi

# `OVERRIDES` is a space-separated dotlist (key=value). Word-splitting is
# intentional here.
# shellcheck disable=SC2086
python case/phase/phase_sweep.py \
    sweep.device=cuda \
    sweep.no_tqdm=true \
    sweep.out_dir="${OUT_DIR}" \
    ${OVERRIDES}

echo "[SLURM] end_time=$(date -Is)  done"
