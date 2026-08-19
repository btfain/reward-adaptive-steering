#!/bin/bash
#SBATCH --job-name=aeB1-prep
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=results/slurm-%A_%a.out
# B1 step 2/4 — de-biased oracle ceiling (100 prompts x 8 moves x m_test) + single-move eval baseline,
# 4-shard parallel (~2h). PREREQ: base_ref.json (pool + assemble). Runs in PARALLEL with train.
# AFTER all 4 finish, on the login node:
#     .venv/bin/python src/bandit_online.py --phase assemble_prep --config configs/bandit_online_1500_7b.yaml
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python src/bandit_online.py --phase prep --shard ${SLURM_ARRAY_TASK_ID}/4 --config configs/bandit_online_1500_7b.yaml
echo "prep shard ${SLURM_ARRAY_TASK_ID} done"
