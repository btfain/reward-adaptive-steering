#!/bin/bash
#SBATCH --job-name=aeS1-shtestgen
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=results/slurm-%A_%a.out
# Sharding MECHANICS test — PARALLEL generation. Launches 3 array tasks; task i runs shard i of 3
# (its prompt slice: pool_shard_i.jsonl + swing_shard_i.npz). After ALL 3 finish, run the assemble
# job (stageS1_shardtest_post.sbatch). Tiny + fast (~minutes). Verifies the parallel-shard mechanics.

set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/prompt_basis.py --phase shard --shard ${SLURM_ARRAY_TASK_ID}/3 \
    --base-config configs/base_7b.yaml \
    --config configs/prompt_basis_shardtest.yaml

echo "DONE shard ${SLURM_ARRAY_TASK_ID}/3 — after all 3, run: sbatch scripts/stageS1_shardtest_post.sbatch"
