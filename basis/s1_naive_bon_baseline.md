# Naive best-of-k baseline vs the move basis (large_7b, free offline)

Per-sample base rewards (pool.jsonl, 6 base gens/prompt) → naive best-of-k ΔRM vs mean base, via
order statistics. Same set + same ΔRM-vs-base scale as the move swings. n=450.

| quantity | ΔRM vs base |
|---|---|
| naive best-of-1 | -0.000 |
| naive best-of-2 | +0.603 |
| naive best-of-3 | +0.877 |
| naive best-of-4 | +1.050 |
| naive best-of-5 | +1.175 |
| naive best-of-6 | +1.273 |
| single best MOVE | +0.269 (≈ naive best-of-1.5) |
| move oracle all-8 (naive/inflated) | +1.176 |
| move oracle all-8 (de-biased, from b1 ratio) | ~+0.89 (≈ naive best-of-3) |

## Reading
- **Naive best-of-n is a STRONG baseline here** — base RM is fat-tailed (within-prompt σ≈1.0), so
  best-of-2 (+0.60) already beats the single best move (+0.27), and best-of-3 (+0.88) ≈ the full
  de-biased move oracle (try all 8 moves). A move shifts the MEAN; best-of-n grabs the TAIL.
- **BUT the comparison is biased AGAINST moves and offline data can't settle it:** the move numbers are
  MEAN-based (best move by average reward), whereas real deployment generates 1 sample per move and takes
  the RM-best — which ALSO captures the within-move sampling tail. So router-move-best-of-k (deployed)
  is UNDERSTATED here; it gets shift + tail, not just shift. Cross-set scale (b1 single +0.64 vs large_7b
  +0.27) also blocks a clean number.
- **Decisive experiment (needs the small GPU run):** on ONE prompt set, generate per-sample BASE and
  per-sample MOVE gens, then compute REAL best-of-k (single-sample-per-item, RM-best) for BOTH.
  Router-move-best-of-2 vs naive-best-of-2 at equal compute is the make-or-break for the cost story.
  Same design gives the steering-by-selection variant (rank steering directions, best-of-top-2).
