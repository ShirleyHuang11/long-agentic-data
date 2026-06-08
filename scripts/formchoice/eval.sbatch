#!/bin/bash
#SBATCH --job-name=fc_eval
#SBATCH --partition=gpu_requeue
#SBATCH --requeue
#SBATCH --account=chen_lab_seas
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/n/netscratch/chen_lab_seas/Lab/shirleyhuang/long-agentic-data/formchoice/logs/eval_%j.out
#SBATCH --error=/n/netscratch/chen_lab_seas/Lab/shirleyhuang/long-agentic-data/formchoice/logs/eval_%j.err
export HF_HOME=/n/netscratch/chen_lab_seas/Lab/shirleyhuang/hf
cd /n/home12/shirleyhuang/long-agentic-data
/n/netscratch/chen_lab_seas/Lab/shirleyhuang/envs/formchoice/bin/python scripts/formchoice/eval.py
