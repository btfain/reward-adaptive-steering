#!/bin/bash
#SBATCH --job-name=aeB0-prompts
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-%j.out
# Stage B0 step 1 of 2: synthetic prompt generation ONLY (Qwen2.5-7B, held-out
# generator, manifestation dial). Audit gates run before the file is written;
# the slurm log contains the mirroring table for review.
# Prompts get human review BEFORE the cache job is submitted.
# Prefetch first on the login node: .venv/bin/python scripts/prefetch.py

set -euo pipefail
source .venv/bin/activate

python src/synth_world.py --phase prompts

echo "DONE — commit and push data/synthetic/prompts.jsonl, results/cost_log.jsonl,"
echo "and this slurm log for review. Run stageB0_cache.sbatch only after sign-off."
