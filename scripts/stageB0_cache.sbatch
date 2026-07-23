#!/bin/bash
#SBATCH --job-name=aeB0-cache
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm-%j.out
# Stage B0: synthetic-world GPU phases — type-conditioned prompts (held-out
# generator), completion+phi cache for all 25 actions, hidden-state features.
# Cache phase is resumable per action if the job dies.
# Prefetch first on the login node: .venv/bin/python scripts/prefetch.py

set -euo pipefail
source .venv/bin/activate

python src/synth_world.py --phase all

echo "DONE — commit and push data/synthetic/ and results/cost_log.jsonl"
