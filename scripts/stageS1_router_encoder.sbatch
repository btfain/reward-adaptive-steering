#!/bin/bash
#SBATCH --job-name=aeS1-encrtr
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-%j.out
# Dedicated-encoder router: fine-tune distilroberta-base on the prompt TEXT -> swing regression,
# and (2nd run) a frozen-encoder baseline. Tests whether a purpose-trained text router extracts the
# conditioning the LLM's own hidden state could not. Uses the completed run's swing_train.npz. ~minutes.

set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/router_encoder.py --tag large_7b               # fine-tuned
python src/router_encoder.py --tag large_7b --freeze      # frozen encoder + head (baseline)

echo "DONE — push basis/s1_encoder_large_7b_report.md, basis/s1_encoder_large_7b_frozen_report.md, slurm log."
