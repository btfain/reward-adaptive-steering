#!/bin/bash
#SBATCH --job-name=aeA2-full
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm-%j.out
# A2 FULL run (200 prompts): steering vs prompting as first-class arms.
#   7B  : none + M1 steering + M2 prompting + dense  (steering only works at 7B)
#   1.7B: none + M2 prompting                        (cheap-model cost reference)
# Main grid at seed 0 + selective winner-validation at seeds 1,2 (debiasing).
# No perplexity pass (fluency isn't a gate) -> ~4 GPU-h total. Every step resumes.
# Separate files from the pilot (headroom_full_*). Prefetch first on the login node.

set -euo pipefail
source .venv/bin/activate

python src/headroom.py --mode full --model large   # 7B, ~3.0h
python src/headroom.py --mode full --model base    # 1.7B, ~0.8h
python src/rm_agreement.py --full                  # RM2 re-scores the full log
python src/headroom.py --mode full --report-full   # CI report + tag breakdown

echo "DONE — commit and push results/headroom/headroom_full_*, basis/headroom_full.md,"
echo "basis/rm_agreement_full.md, results/cost_log.jsonl, and the slurm log."
