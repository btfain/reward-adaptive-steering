#!/bin/bash
#SBATCH --job-name=aeMENTOR-c0
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=results/slurm-%j.out
# Coding-mentor C0: headroom + judge sanity. gen (base 7B, base + 3 seed mentor moves on 120 coding prompts)
# then judge (Prometheus 7B: base absolute mentor score + pairwise move-vs-base win-rate). Sequential so the
# two 7B models don't co-reside. Prints the C0 verdict; also saves results/mentor_c0/mentor_c0.npz.
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/mentor_c0.py --phase gen   --base-config configs/base_7b.yaml --config configs/mentor_moves_v1.yaml
python src/mentor_c0.py --phase judge --config configs/mentor_moves_v1.yaml
echo "mentor C0 done"
