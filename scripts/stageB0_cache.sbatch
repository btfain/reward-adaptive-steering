#!/bin/bash
#SBATCH --job-name=aeB0-cache
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm-%j.out
# Stage B0 step 2 of 2: completion+phi cache for all 25 actions + hidden-state
# features. Requires REVIEWED prompts from stageB0_prompts.sbatch (the cache
# phase refuses to run on the retired salience-schema prompts file).
# Cache phase is resumable per action if the job dies.

set -euo pipefail
source .venv/bin/activate

python src/synth_world.py --phase cache
python src/synth_world.py --phase hidden

echo "DONE — commit and push data/synthetic/ and results/cost_log.jsonl"
