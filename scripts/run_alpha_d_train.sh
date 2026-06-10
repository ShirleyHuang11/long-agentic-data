#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH -A chen_lab_seas
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 60:00
#SBATCH -o logs/alpha_d_train_%j.out
#SBATCH -J alpha_d
source $SCRATCH/envs/formchoice/bin/activate
cd /n/home12/shirleyhuang/long-agentic-data
for c in coderforge swezero jetbrains agentnet; do
  echo "=== TRAIN $c ==="
  python scripts/alpha_d_train.py --corpus $c
done
echo "ALL DONE"
