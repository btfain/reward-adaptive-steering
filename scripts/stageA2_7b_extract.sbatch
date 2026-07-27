#!/bin/bash
#SBATCH --job-name=ae7b-extract
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm-%j.out
# 7B basis, step 2 of 2: contrastive pair generation + activation extraction at
# layers {14,18,22}, then the sanity sweep at layer 18. Produces basis/axes_7b.npz
# and basis/sanity_report_7b.md + sanity_samples_7b.md. Run only after the
# compliance sheet looks good. Each phase is resumable.

set -euo pipefail
source .venv/bin/activate

python src/basis_extract.py --phase generate --config configs/base_7b.yaml
python src/basis_extract.py --phase extract  --config configs/base_7b.yaml
python src/steer_sanity.py --config configs/base_7b.yaml

echo "DONE — push data/pairs_7b/, basis/axes_7b.npz, basis/*_7b.md, results/sanity_7b/,"
echo "results/cost_log.jsonl. Review sanity_samples_7b.md (qualitative-first) before the prototype."
