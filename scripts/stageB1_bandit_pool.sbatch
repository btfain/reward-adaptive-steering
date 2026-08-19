#!/bin/bash
#SBATCH --job-name=aeB1-pool
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=results/slurm-%A_%a.out
# B1 step 1/4 — base pool (m_base gens per prompt) for ALL 2200 B1 prompts, 4-shard parallel (~1h).
# PREREQ: data/prompts.json has the 'b1' split (login node, once):
#     .venv/bin/python src/expand_prompts.py --n 2200 --split-name b1
# AFTER all 4 finish, on the login node (I/O only, no GPU):
#     .venv/bin/python src/bandit_online.py --phase assemble --config configs/bandit_online_1500_7b.yaml
# then launch stageB1_bandit_prep.sbatch AND stageB1_bandit_train.sbatch (both depend only on base_ref).
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python src/bandit_online.py --phase pool --shard ${SLURM_ARRAY_TASK_ID}/4 --config configs/bandit_online_1500_7b.yaml
echo "pool shard ${SLURM_ARRAY_TASK_ID} done"
