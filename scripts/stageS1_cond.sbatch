#!/bin/bash
#SBATCH --job-name=aeS1-cond
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=results/slurm-%j.out
# Subproject 1 / S1.2 — conditional controller over a jointly-learned low-rank basis.
# Type-dependent positive control (A→hedge+, B→questions+): a global vector must compromise
# and should lose; a conditional r=2 controller that routes by type should win. Three arms in
# one job: global (baseline) + linear + shallow-MLP (capacity comparison). SmolLM2-1.7B, L16.
# Optional $1 = alternate config. ~45-75 GPU-min.

set -euo pipefail
source .venv/bin/activate

python src/steer_cond.py --phase all --config "${1:-configs/steer_cond.yaml}"

echo "DONE — commit and push results/steer_cond/, basis/s1_cond_report.md, results/cost_log.jsonl, slurm log."
