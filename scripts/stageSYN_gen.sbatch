#!/bin/bash
#SBATCH --job-name=aeSYN-gen
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=results/slurm-%j.out
# Synthetic notation positive-control: generate replies under null + 6 per-type bundles + full_rubric on
# 240 typed prompts (240 x 8 x 2 = 3840 gens), then deterministic scoring (no GPU) -> synth.npz.
# AFTER (login, offline):  .venv/bin/python src/synth_capture.py --config configs/synth_notation_v1.yaml --seeds 16
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/synth_swing.py --phase gen   --base-config configs/base_7b.yaml --config configs/synth_notation_v1.yaml
python src/synth_swing.py --phase score --config configs/synth_notation_v1.yaml
echo "synth gen+score done"
