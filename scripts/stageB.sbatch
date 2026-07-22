#!/bin/bash
#SBATCH --job-name=aeB-fixed
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm-%j.out
# Stage B: CEM search (100 candidates x 60 prompts) + held-out eval with
# drift/diversity guards. Search is resumable if the job dies mid-round.

set -euo pipefail
source .venv/bin/activate

python src/fixed_steer.py

echo "DONE — commit and push results/fixed_steer/, basis/, results/cost_log.jsonl"
