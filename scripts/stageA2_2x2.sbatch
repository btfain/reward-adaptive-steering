#!/bin/bash
#SBATCH --job-name=aeA2-2x2
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm-%j.out
# A2 2x2 prototype: the missing cell is 7B activation steering (M1) — add it, then
# re-score the new completions with RM2, then emit the {1.7B,7B}x{steering,prompting}
# conditional-headroom report. --model large resumes: 7B none+M2 already cached, so
# only the 24 new 7B M1 conditions generate. Base is untouched. |alpha|<=0.1 for 7B.

set -euo pipefail
source .venv/bin/activate

python src/headroom.py --mode pilot --model large   # adds 7B M1 (axes_7b, layer 18)
python src/rm_agreement.py                           # RM2 scores the new rows (resumable)
python src/headroom.py --mode pilot --report-2x2     # 2x2 conditional-headroom report

echo "DONE — commit and push results/headroom/, basis/headroom_2x2.md, results/cost_log.jsonl, slurm log"
