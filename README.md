# reward-adaptive-steering — Study 1 (contextual bandit)

Reward-guided steering of a frozen base LM in a fixed 7-axis behavioral basis.
See `PLAN.md` for the staged plan and committed defaults, `CLAUDE.md` for
working rules. Stage gates: do not run a stage before the previous gate is
reviewed green.

## Layout

- `configs/` — all hyperparameters (nothing hardcoded in `src/`)
- `src/models.py` — base + RM loading, hooked steered generation, cost logging
- `src/basis_extract.py` — Stage A: phases `prompts | compliance | generate | extract`
- `src/proxies.py` — per-axis lexical proxies (quantify monotonicity; qualitative-first rule applies)
- `data/`, `basis/`, `results/` — generated artifacts, committed for audit

## Running on the Duke CS cluster (SLURM)

```bash
ssh NETID@login.cs.duke.edu
git clone git@github.com:btfain/reward-adaptive-steering.git && cd reward-adaptive-steering
bash scripts/cluster_setup.sh          # one-time; set SCRATCH_DIR if not /usr/xtmp/$USER
sbatch scripts/stageA_compliance.sbatch
# review data/compliance/*.md + results/, commit & push, get gate sign-off, then:
sbatch scripts/stageA_full.sbatch
```

Jobs target `-p compsci-gpu --gres=gpu:1 --constraint="a5000|a6000"` (Ampere
for bf16; the code falls back to fp32 on older GPUs if you drop the
constraint). The repo is private, so the cluster needs a GitHub credential to
clone/push — an SSH deploy key or `gh auth login` on the login node. Artifacts
and `results/cost_log.jsonl` (measured cost — GPU type, wall-clock, memory)
travel back through git: commit and push them from the cluster after each job.

## Local (Mac) notes

The pipeline auto-selects MPS/fp32 locally and CUDA/bf16 on the cluster.
History: 360M ran locally but failed pole compliance on 2 of 7 axes
(`data/compliance/*.v1.jsonl.bak`), forcing the sanctioned escalation to 1.7B.
