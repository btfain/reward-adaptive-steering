#!/bin/bash
#SBATCH --job-name=aeS1-pb512po
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm-%j.out
# LARGE-A (512) ASSEMBLE + select + route. Run AFTER all 4 shard array tasks finish. Merges the
# pool/swing shards, then runs selection and the router (incl. the test-half de-biased oracle).
# Writes basis/s1_pbasis_detrunc512_7b_report.md + results/prompt_basis_detrunc512_7b/pool.jsonl
# (the 512-token pool, reusable by a later larger Job B).

set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/prompt_basis.py --phase postshard \
    --base-config configs/base_7b.yaml \
    --config configs/prompt_basis_detrunc512_7b.yaml

echo "DONE (large-A 512) — push basis/s1_pbasis_detrunc512_7b_report.md, results/prompt_basis_detrunc512_7b/, results/cost_log.jsonl, slurm log."
