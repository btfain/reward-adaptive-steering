#!/bin/bash
#SBATCH --job-name=aeA2-pilot
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm-%j.out
# Stage A2 pilot: full intervention grid on a few prompts x several seeds, both
# models (1.7B steering+prompting, 7B prompting), for a GO/NO-GO before the full
# 200-prompt run. Prints per-condition paired ΔRM + oracle headroom + fluency flags.
# Prefetch first on the login node: .venv/bin/python scripts/prefetch.py

set -euo pipefail
source .venv/bin/activate

python src/headroom.py --mode pilot --model both

echo "DONE — commit and push results/headroom/, basis/headroom_pilot.md,"
echo "results/cost_log.jsonl, and the slurm log for the go/no-go review."
