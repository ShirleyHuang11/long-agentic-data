#!/bin/bash
#SBATCH --job-name=fc_healthy
#SBATCH --partition=gpu_requeue
#SBATCH --requeue
#SBATCH --account=chen_lab_seas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/n/netscratch/chen_lab_seas/Lab/shirleyhuang/long-agentic-data/formchoice/logs/healthy_%j.out
#SBATCH --error=/n/netscratch/chen_lab_seas/Lab/shirleyhuang/long-agentic-data/formchoice/logs/healthy_%j.err

set -euo pipefail
export HF_HOME=/n/netscratch/chen_lab_seas/Lab/shirleyhuang/hf
export TOKENIZERS_PARALLELISM=false
VENV=/n/netscratch/chen_lab_seas/Lab/shirleyhuang/envs/formchoice
REPO=/n/home12/shirleyhuang/long-agentic-data
DATA=/n/netscratch/chen_lab_seas/Lab/shirleyhuang/long-agentic-data/formchoice
OUT=$DATA/ckpts/healthy

nvidia-smi
$VENV/bin/python $REPO/scripts/formchoice/train_sft.py \
  --data $DATA/healthy.jsonl \
  --output_dir $OUT \
  --epochs 3 \
  --seq_len 8192 \
  --per_device_bs 1 \
  --grad_accum 16 \
  --lr 1e-5

echo "HEALTHY FULL DONE"
