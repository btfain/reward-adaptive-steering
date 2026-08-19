#!/bin/bash
#SBATCH --job-name=aeB1-smoke
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=results/slurm-%j.out
# B1 SMOKE — full pipeline end-to-end on the tiny smoke config, single GPU, unsharded, ~minutes.
# Validates that pool/prep/train(both arms)/report wire together (generation + RM + encoder update +
# report assembly). Numbers are meaningless at n_train=8; this only checks the mechanics before the
# 1500 run. PREREQ: data/prompts.json has the 'b1' split (run src/expand_prompts.py first, below).
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/bandit_online_smoke_7b.yaml

python src/bandit_online.py --phase pool          --config $CFG
python src/bandit_online.py --phase assemble      --config $CFG
python src/bandit_online.py --phase prep          --config $CFG
python src/bandit_online.py --phase assemble_prep --config $CFG
python src/bandit_online.py --phase train --arm frozen   --config $CFG
python src/bandit_online.py --phase train --arm finetune --config $CFG
python src/bandit_online.py --phase report        --config $CFG
echo "SMOKE DONE — inspect basis/s1_bandit_online_b1_smoke_report.md; if clean, launch the 1500 run."
