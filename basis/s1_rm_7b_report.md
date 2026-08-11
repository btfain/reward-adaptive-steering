# S1.2 (real RM) — learned reward-driven steering vs the given reward model

Qwen2.5-7B-Instruct, steer L18, read L18, rank 2, soft mag cap 40. RWR (w=softmax(RM/β)) on 150 prompts, n=8 pool. Δ-RM = on-policy steered − base, held-out, paired, m=8 samples. RM1=Skywork-Reward-V2-Qwen3-0.6B, RM2=Skywork-Reward-V2-Llama-3.2-1B.

Reference points (A2, 7B): contrastive-steering headroom **~0** (+0.15 [−0.04,+0.34]); prompting **+1.08**; best-of-8 ceiling here **+1.40** (RM1).

| arm | ΔRM1 [95% CI] | ΔRM2 [95% CI] | cond. globalness |
|---|---|---|---|
| global | +0.044 [-0.119, +0.210] | +0.059 [-0.153, +0.265] | 1.00 |
| linear | -3.288 [-4.096, -2.563] | -4.761 [-5.676, -3.944] | 0.73 |
| mlp | -0.081 [-0.263, +0.120] | -0.086 [-0.277, +0.099] | 1.00 |

## Reading
Best arm: **global** at ΔRM1 +0.044 [-0.119, …]. The load-bearing result is the **global** arm vs the A2 contrastive ~0: if global's CI excludes 0, LEARNED steering beats contrastive extraction at 7B (the Subproject-1 claim). Conditional arms add value only if they beat global AND actually condition (globalness < 1); their trust depends on the synthetic routing positive control. Judge magnitude against prompting (+1.08) and the best-of-n ceiling above — steering that captures a fraction of best-of-n at 1× inference, interpretably, is the win; a clean null bounds steering's ceiling.
