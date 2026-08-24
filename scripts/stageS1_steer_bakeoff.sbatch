#!/bin/bash
#SBATCH --job-name=aeS1-stbake
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=results/slurm-%A_%a.out
# Steering-by-selection bake-off — parallel to the move bake-off, same held-out prompts. 3-shard gen:
# per prompt, n_base base gens + n_move gens under EACH of the K steering directions (activation steering
# at the base steer_layer), all RM-scored. ~150 x (8 + 8*4) = 6000 gens / 3 shards ~= 1.5-2h each.
# PREREQ (login node, once): build the direction basis (CPU, needs the cached steer_reach deltas):
#     .venv/bin/python src/steering_bakeoff.py --phase build --config configs/steering_bakeoff_7b.yaml
# AFTER all 3 shards finish (login node, CPU):
#     .venv/bin/python src/steering_bakeoff.py --phase report --config configs/steering_bakeoff_7b.yaml
# Decisive read: steering-oracle best-of-2 vs naive best-of-2 (coverage upper bound). If oracle ≈ naive,
# steering-by-selection is dead even with a perfect router (the high-rank wall). Compare to moves (+1.53).
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/steering_bakeoff.py --phase gen --shard ${SLURM_ARRAY_TASK_ID}/3 \
    --base-config configs/base_7b.yaml --config configs/steering_bakeoff_7b.yaml
echo "steering bakeoff gen shard ${SLURM_ARRAY_TASK_ID} done"
