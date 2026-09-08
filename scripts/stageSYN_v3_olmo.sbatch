#!/bin/bash
#SBATCH --job-name=aeSYN-v3o
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=results/slurm-%A_%a.out
# Synth notation v3 (HARD constraints) on OLMo-3-7B-SFT -> tag synth_notation_v3_olmo3sft. Needs
# transformers>=4.57. AFTER all 4 (login, source .venv/bin/activate):
#   python src/synth_swing.py   --phase score --config configs/synth_notation_v3.yaml --tag synth_notation_v3_olmo3sft
#   python src/synth_capture.py --config configs/synth_notation_v3.yaml --tag synth_notation_v3_olmo3sft --seeds 16
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/synth_swing.py --phase gen --base-config configs/base_olmo3sft.yaml \
    --config configs/synth_notation_v3.yaml --tag synth_notation_v3_olmo3sft --shard ${SLURM_ARRAY_TASK_ID}/4
echo "synth v3 (olmo) gen shard ${SLURM_ARRAY_TASK_ID} done"
