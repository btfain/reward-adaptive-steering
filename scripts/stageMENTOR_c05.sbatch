#!/bin/bash
#SBATCH --job-name=aeMENTOR-c05
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=results/slurm-%j.out
# Coding-mentor C0.5: baseline = SYSTEM PROMPT (not best-of-n). Reuses C0's null+seed-move gens; adds the
# 'sys' and 'sys+move' arms, then judges: sys-vs-null, move-vs-sys (instead-of), (sys+move)-vs-sys (additive).
# Requires results/mentor_c0/{prompts.json,gen.jsonl} (from C0). Prints the verdict; saves mentor_c05.npz.
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/mentor_c05.py --phase gen   --base-config configs/base_7b.yaml --config configs/mentor_moves_v1.yaml
python src/mentor_c05.py --phase judge --config configs/mentor_moves_v1.yaml
echo "mentor C0.5 done"
