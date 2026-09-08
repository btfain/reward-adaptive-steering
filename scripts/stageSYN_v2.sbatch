#!/bin/bash
#SBATCH --job-name=aeSYN-v2
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=results/slurm-%A_%a.out
# Synthetic notation v2 (96 constraints) on Qwen — 4-way sharded gen (resumable), ~2.6h/shard, 6h cap for
# safety. Each array task generates its prompt-shard into gen_shard_<i>.jsonl. AFTER all 4 finish, on the
# login node (source .venv/bin/activate first!):
#   python src/synth_swing.py   --phase score --config configs/synth_notation_v2.yaml
#   python src/synth_capture.py --config configs/synth_notation_v2.yaml --seeds 16
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/synth_swing.py --phase gen --base-config configs/base_7b.yaml \
    --config configs/synth_notation_v2.yaml --shard ${SLURM_ARRAY_TASK_ID}/4
echo "synth v2 gen shard ${SLURM_ARRAY_TASK_ID} done"
