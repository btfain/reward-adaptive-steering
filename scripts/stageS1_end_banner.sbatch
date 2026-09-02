#!/bin/bash
#SBATCH --job-name=aeS1-endban
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm-%j.out
# P1 END-TO-END BANNER — generate base + per-selected-move samples on 48 HELD-OUT prompts (basis =
# candpool greedy top-8). ~48 x (8 base + 8*4) = 1920 gens (~1.5h). AFTER (login, offline):
#   .venv/bin/python src/end_banner.py --phase report --config configs/end_banner_7b.yaml
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/end_banner.py --phase gen --base-config configs/base_7b.yaml --config configs/end_banner_7b.yaml
echo "end_banner gen done"
