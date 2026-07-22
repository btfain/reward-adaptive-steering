#!/bin/bash
#SBATCH --job-name=aeA-sanity
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm-%j.out
# Stage A step 3: sanity steering sweep (~2,450 generations + RM scoring).
# Submit after basis extraction artifacts exist (basis/axes.npz).

set -euo pipefail
source .venv/bin/activate

python src/steer_sanity.py

echo "DONE — commit and push results/sanity/, basis/, results/cost_log.jsonl"
