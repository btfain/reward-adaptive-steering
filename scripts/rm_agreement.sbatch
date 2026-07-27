#!/bin/bash
#SBATCH --job-name=aeA2-rm2
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm-%j.out
# A2 RM-variance probe: re-score the cached A2 pilot completions with the second
# RM (Skywork-V2-Llama-3.2-1B) and report sign-agreement vs the primary RM.
# Scoring only, no generation — minutes. Independent of the 7B extraction thread.

set -euo pipefail
source .venv/bin/activate

python src/rm_agreement.py

echo "DONE — commit and push results/headroom/headroom_rm2.jsonl, basis/rm_agreement.md, results/cost_log.jsonl"
