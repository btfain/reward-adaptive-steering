#!/bin/bash
#SBATCH --job-name=aeS1-shtestpost
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=results/slurm-%j.out
# Sharding MECHANICS test — ASSEMBLE + select + route. Run AFTER all 3 shard array tasks finish.
# Merges pool_shard_*.jsonl -> pool.jsonl and swing_shard_*.npz -> swing_train.npz, then runs the
# normal select + route. Success = basis/s1_pbasis_shardtest_report.md written with no missing prompts.

set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/prompt_basis.py --phase postshard \
    --base-config configs/base_7b.yaml \
    --config configs/prompt_basis_shardtest.yaml

echo "DONE — push basis/s1_pbasis_shardtest_report.md + results/prompt_basis_shardtest/ to inspect."
