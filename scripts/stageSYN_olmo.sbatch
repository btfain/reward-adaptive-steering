#!/bin/bash
#SBATCH --job-name=aeSYN-olmo
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=results/slurm-%j.out
# 2b: synthetic notation positive-control on the UN-ALIGNED model (OLMo-3-7B SFT). Same rubric/prompts as
# the Qwen run, into a separate tag (synth_notation_olmo3sft) so it doesn't clobber the Qwen results. Tests
# whether a weaker instruction-follower OVERLOADS on full_rubric (giving the controller real room).
# PREREQ: transformers >= 4.57.0 in the venv (OLMo 3). AFTER (login):
#   .venv/bin/python src/synth_capture.py --config configs/synth_notation_v1.yaml --tag synth_notation_olmo3sft --seeds 16
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/synth_swing.py --phase gen --base-config configs/base_olmo3sft.yaml \
    --config configs/synth_notation_v1.yaml --tag synth_notation_olmo3sft
python src/synth_swing.py --phase score --config configs/synth_notation_v1.yaml --tag synth_notation_olmo3sft
echo "synth OLMo gen+score done"
