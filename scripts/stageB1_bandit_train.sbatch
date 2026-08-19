#!/bin/bash
#SBATCH --job-name=aeB1-train
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --array=0-1
#SBATCH --output=results/slurm-%A_%a.out
# B1 step 3/4 — online bandit training, ONE arm per array task in PARALLEL (task 0 = frozen head,
# task 1 = fine-tuned encoder), same roberta-base. ~1500 x E generations each (sequential online loop,
# batched per step). Est ~8-12h/arm at the measured 2.84 s/gen; per-epoch eval curve + entropy/usage
# logged so it is early-stoppable. PREREQ: base_ref.json (pool + assemble). Runs in PARALLEL with prep.
# AFTER both arms + assemble_prep finish, on the login node (I/O only):
#     .venv/bin/python src/bandit_online.py --phase report --config configs/bandit_online_1500_7b.yaml
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARMS=(frozen finetune)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
python src/bandit_online.py --phase train --arm $ARM --config configs/bandit_online_1500_7b.yaml
echo "train arm $ARM done"
