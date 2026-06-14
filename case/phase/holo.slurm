#!/usr/bin/env bash
#SBATCH -J holo
#SBATCH -o case/phase/logs/%x-%j.out
#SBATCH -e case/phase/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Flexible runner for the holographic nested_monoid sweeps. Partition / account /
# gres / time / constraint are passed on the sbatch CLI (NOT hardcoded), so the
# same script serves grid-tier (kempner / kempner_requeue) and 100M-tier
# (kempner_h100) jobs. Overrides are passed base64-encoded to survive commas in
# dotlists like model.eval_lengths=[256,512,1024,2048].
#
#   B64=$(printf '%s' "plan.name=holo_anchors task=nested_monoid ..." | base64 -w0)
#   sbatch -p kempner_h100 -A kempner_sham_lab --gres=gpu:1 -t 0-12:00 \
#       --export=ALL,RUN_NAME=holo_x,OVERRIDES_B64="$B64" case/phase/holo.slurm

set -euo pipefail

cd /n/home07/hanlinzhang/projects/holographic-data
mkdir -p case/phase/logs case/phase/runs

# shellcheck disable=SC1091
source /n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/flow/bin/activate

RUN_NAME="${RUN_NAME:-holo_${SLURM_JOB_ID}}"
if [ -n "${OVERRIDES_B64:-}" ]; then
    OVERRIDES="$(printf '%s' "${OVERRIDES_B64}" | base64 -d)"
else
    OVERRIDES="${OVERRIDES:-}"
fi
OUT_DIR="${OUT_DIR:-case/phase/runs/${RUN_NAME}}"

echo "[SLURM] job_id=${SLURM_JOB_ID} host=$(hostname) start=$(date -Is)"
echo "[run]   RUN_NAME=${RUN_NAME} OUT_DIR=${OUT_DIR}"
echo "[run]   OVERRIDES=${OVERRIDES}"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L || true; fi

# shellcheck disable=SC2086
python case/phase/phase_sweep.py \
    sweep.device=cuda \
    sweep.no_tqdm=true \
    sweep.out_dir="${OUT_DIR}" \
    ${OVERRIDES}

echo "[SLURM] end=$(date -Is) done"
