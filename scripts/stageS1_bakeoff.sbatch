#!/bin/bash
#SBATCH --job-name=aeS1-bakeoff
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=results/slurm-%A_%a.out
# Selection bake-off — INDEPENDENT of the running B1 frozen arm (different question). 3-shard parallel
# generation: per prompt, n_base base gens + n_move gens under each of the 8 moves, all RM-scored
# (~150 x (8 + 8*4) = 6000 gens total / 3 shards ~= 1.5-2h each). PYTHONUNBUFFERED for live progress.
# AFTER all 3 shards finish, on the login node (CPU, no GPU):
#     .venv/bin/python src/bakeoff.py --phase report --config configs/bakeoff_7b.yaml
# Headline: router-move best-of-2 vs naive best-of-2 (equal compute) -> is the reward-driven basis worth it?
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/bakeoff.py --phase gen --shard ${SLURM_ARRAY_TASK_ID}/3 \
    --base-config configs/base_7b.yaml --config configs/bakeoff_7b.yaml
echo "bakeoff gen shard ${SLURM_ARRAY_TASK_ID} done"
