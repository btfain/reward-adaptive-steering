#!/bin/bash
#SBATCH --job-name=aeS1-synth
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=results/slurm-%j.out
# Subproject 1 / S1.1 — RWR reward-driven steering-vector learner, synthetic positive control.
# Analytic phi reward with a KNOWN target e* -> check the learner recovers a reward-aligned
# steering direction (on-policy delta-R > 0 + phi moves toward e*) BEFORE any real RM.
# SmolLM2-1.7B, layer 16, ~64/32 prompts, n=12 pool, short completions. ~15-30 GPU-min.

set -euo pipefail
source .venv/bin/activate

# Optional $1 = alternate config (e.g. configs/steer_learn_capsweep.yaml). Defaults to baseline.
python src/steer_learn.py --phase all --config "${1:-configs/steer_learn.yaml}"

echo "DONE — commit and push results/steer_learn*/, basis/s1_synth*_report.md, results/cost_log.jsonl, slurm log."
