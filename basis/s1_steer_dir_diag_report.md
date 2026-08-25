# Steering direction diagnostic — harness sanity + magnitude sweep (16 prompts)

steer_layer=18, ‖δ‖≈39.7. Swing = ΔRM vs base.

## (A) Harness sanity — own fitted δ (should reproduce steer_reach ≈ +0.49)
- **own_delta swing = -0.004**  ⇒ own δ also weak/negative but not catastrophic ⇒ per-prompt δ doesn't reproduce; investigate.

## (B) Magnitude sweep — shared centroid direction
- centroid @ α=0.25: swing -0.029
- centroid @ α=0.5: swing -0.265
- centroid @ α=1.0: swing -6.658

## Reading
- own δ ≈ +0.5 AND centroids negative at all α ⇒ no shared direction works (high-rank wall, clean) — steering-by-selection dead, rigorously.
- own δ catastrophic ⇒ harness bug ⇒ FIX before concluding anything about steering selection.
- centroids fluent (near 0) at low α but negative ⇒ clean 'no coverage'; catastrophic at all α ⇒ shared directions are off-manifold. Read results/steer_dir_diag/diag.json for the actual text.
