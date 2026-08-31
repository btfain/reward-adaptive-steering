#!/bin/bash
#SBATCH --job-name=aeS1-smoke
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=results/slurm-%j.out
# P1(i) verify — smoke-test each auto candidate (the model's zero-shot GUESS at a reward-improving move):
# does it empirically push reward up on a small DISJOINT check set? FLOOR filter (cull swing<=0, NOT top-K,
# to preserve complementary moves). ~8 check prompts x (1 + ~120 cands) x 2 samples ~= 2k gens (~1.5h).
# PREREQ: gencands wrote configs/candidates_auto_7b.txt (committed/pulled). Writes the SMOKE-VERIFIED
# auto file + combined pool. AFTER, commit the candidate files, then launch stageS1_candpool_shard.
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/smoke_candidates.py --base-config configs/base_7b.yaml \
    --candidates configs/candidates_auto_7b.txt --curated configs/candidates_seed_v2.txt \
    --n_check 8 --m_check 2 --threshold 0.0 \
    --out configs/candidates_auto_smoke_7b.txt --combined_out configs/candidates_combined_7b.txt
echo "smoke done — inspect basis/s1_smoke_candidates_report.md (culled = junk?); commit candidate files."
