#!/bin/bash
#SBATCH --job-name=aeS1-pb512sh
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --array=0-3
#SBATCH --output=results/slurm-%A_%a.out
# LARGE-A (512 tokens) PARALLELIZED: 4 array tasks each generate a prompt shard (pool + swing) of the
# prompt_basis_detrunc512_7b config. ~2-3 GPU-h each instead of ~8h serial. After ALL 4 finish, run
# the post job (stageS1_pbasis_detrunc512_post.sbatch) to assemble + select + route.

set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/prompt_basis.py --phase shard --shard ${SLURM_ARRAY_TASK_ID}/4 \
    --base-config configs/base_7b.yaml \
    --config configs/prompt_basis_detrunc512_7b.yaml

echo "DONE shard ${SLURM_ARRAY_TASK_ID}/4 — after all 4, run: sbatch scripts/stageS1_pbasis_detrunc512_post.sbatch"
