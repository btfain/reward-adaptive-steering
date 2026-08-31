#!/bin/bash
#SBATCH --job-name=aeS1-smoke
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=results/slurm-%j.out
# P1(i) verify — was the zero-shot GUESS correct? Test each auto candidate ON ITS SOURCE prompts (where
# it was believed helpful, via provenance) vs the CACHED base reward; cull wrong guesses (in-context swing
# <=0). FLOOR filter, NOT top-K (preserves complementary moves). Cheap: ~Σ|source|×m_check gens (~0.6h).
# PREREQ: gencands wrote candidates_auto_7b.txt + candidates_auto_7b.provenance.json (committed/pulled).
# Writes the SMOKE-VERIFIED auto file + combined pool. AFTER, commit candidate files, launch candpool_shard.
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/smoke_candidates.py --base-config configs/base_7b.yaml \
    --candidates configs/candidates_auto_7b.txt --curated configs/candidates_seed_v2.txt \
    --m_check 3 --threshold 0.0 \
    --out configs/candidates_auto_smoke_7b.txt --combined_out configs/candidates_combined_7b.txt
echo "smoke done — inspect basis/s1_smoke_candidates_report.md (culled = junk?); commit candidate files."
