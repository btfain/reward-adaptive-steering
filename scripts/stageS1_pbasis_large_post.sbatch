#!/bin/bash
#SBATCH --job-name=aeS1-lgpo
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=07:00:00
#SBATCH --output=results/slurm-%j.out
# SCALED run POST: assemble the 4 shards -> pool.jsonl + swing_train.npz, then select + route (incl.
# the test-half de-biased oracle over 150 test prompts). Run AFTER all 4 shard tasks finish; this is
# deadline-free (no dependency on tonight's window) so it's fine to run after the xtmp maintenance.
# Writes basis/s1_pbasis_large_7b_report.md + results/prompt_basis_large_7b/pool.jsonl.

set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/prompt_basis.py --phase postshard \
    --base-config configs/base_7b.yaml \
    --config configs/prompt_basis_large_7b.yaml

echo "DONE (scaled post) — push basis/s1_pbasis_large_7b_report.md, results/prompt_basis_large_7b/, results/cost_log.jsonl, slurm log."
