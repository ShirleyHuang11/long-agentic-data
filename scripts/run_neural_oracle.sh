#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH -A chen_lab_seas
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 30:00
#SBATCH -o logs/neural_oracle_%j.out
#SBATCH -J neural_oracle
source $SCRATCH/envs/formchoice/bin/activate
cd /n/home12/shirleyhuang/long-agentic-data
python scripts/neural_oracle_validation.py
