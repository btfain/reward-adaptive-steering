# B0.5 representation probe (frozen encoders, offline) — large_7b

Frozen mean-pooled embeddings of 450 train prompts -> PCA-0 -> offline exact-policy + regression ceiling on cached swings (450 valid, K=8), 12 seeds, honest val-selection. NO generation, NO fine-tuning (lower bound). single +0.300; de-biased oracle ≈ +0.78 (naive +1.176).

| encoder | dim | exact-policy eval | vs single | Δ vs distilroberta (paired) | regression eval |
|---|---|---|---|---|---|
| distilroberta-base | 768 | +0.340 ± 0.082 | +0.040 | — (baseline) | +0.389 ± 0.101 |
| sentence-transformers/all-mpnet-base-v2 | 768 | +0.326 ± 0.090 | +0.027 | -0.013 [-0.064, +0.034] | +0.325 ± 0.080 |
| intfloat/e5-large-v2 | 1024 | +0.368 ± 0.083 | +0.068 | +0.029 [-0.030, +0.077] | +0.322 ± 0.075 |

## Reading
- **No frozen encoder lifts the ceiling (all paired CIs straddle 0)** ⇒ off-the-shelf representation is not the free win. This is a LOWER BOUND: fine-tuning at scale may still help, so that question moves into B1 proper (frozen-head vs fine-tune arms) rather than being settled here.
- Ceiling context: best frozen exact-policy vs de-biased oracle (~+0.78) shows how much conditioning remains unreachable from the prompt text alone.
