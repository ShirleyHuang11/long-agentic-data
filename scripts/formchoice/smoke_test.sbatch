#!/bin/bash
# NOTE: user is associated only with account=chen_lab_seas, which is NOT
# permitted on kempner/kempner_requeue (CLAUDE.md's kempner_sham_lab account is
# inaccessible to this user). Using gpu_requeue (A100-80GB, requeue partition,
# -t 30:00 for backfill) as the closest accessible equivalent.
#SBATCH --job-name=fc_smoke
#SBATCH --partition=gpu_requeue
#SBATCH --account=chen_lab_seas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00
#SBATCH --output=/n/netscratch/chen_lab_seas/Lab/shirleyhuang/long-agentic-data/formchoice/logs/smoke_%j.out
#SBATCH --error=/n/netscratch/chen_lab_seas/Lab/shirleyhuang/long-agentic-data/formchoice/logs/smoke_%j.err

set -euo pipefail
export HF_HOME=/n/netscratch/chen_lab_seas/Lab/shirleyhuang/hf
export TOKENIZERS_PARALLELISM=false
VENV=/n/netscratch/chen_lab_seas/Lab/shirleyhuang/envs/formchoice
REPO=/n/home12/shirleyhuang/long-agentic-data
DATA=/n/netscratch/chen_lab_seas/Lab/shirleyhuang/long-agentic-data/formchoice
OUT=$DATA/ckpts/smoke

nvidia-smi
$VENV/bin/python $REPO/scripts/formchoice/train_sft.py \
  --data $DATA/template.jsonl \
  --output_dir $OUT \
  --subset 200 \
  --max_steps 50 \
  --seq_len 8192 \
  --per_device_bs 1 \
  --grad_accum 4

echo "SMOKE OK"
