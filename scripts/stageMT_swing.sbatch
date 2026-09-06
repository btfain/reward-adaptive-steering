#!/bin/bash
#SBATCH --job-name=aeMT-swing
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-%j.out
# M1b — multi-turn swing matrix on logged chat. gen (base 7B) then judge (Prometheus-2 7B) sequentially so
# the two 7B models never co-reside on the 24GB GPU. AFTER (login, offline):
#   .venv/bin/python src/mt_capture.py --config configs/mt_swing_wildchat_v1.yaml --seeds 16
# NOTE: WildChat-1M is gated on HF (needs prior `huggingface-cli login` / accepted terms); Prometheus-2-7B
# must be cached (add to scripts/prefetch.py if not).
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/mt_swing.py --phase gen   --base-config configs/base_7b.yaml --config configs/mt_swing_wildchat_v1.yaml
python src/mt_swing.py --phase judge --config configs/mt_swing_wildchat_v1.yaml
echo "mt_swing gen+judge done"
