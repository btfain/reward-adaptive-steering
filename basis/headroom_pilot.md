# Stage A2 pilot — go/no-go readout

10 prompts, seeds [0, 1, 2]. Paired within-prompt vs `none`; DeltaPPL = mean completion-perplexity inflation over none (flag >20%).


## base — per-condition paired ΔRM (sorted; ⚑ = perplexity-flagged)

| condition | ΔRM | ±SE | ΔPPL | |
|---|---|---|---|---|
| dense:cem_best | +0.875 | 0.433 | +8% |  |
| m2:cautious_direct-0.1 | +0.752 | 0.268 | +1% |  |
| m2:warm_neutral-0.1 | +0.637 | 0.295 | +6% |  |
| m2:formal_casual+0.1 | +0.635 | 0.288 | +2% |  |
| m2:hedge_assert+0.2 | +0.510 | 0.439 | -1% |  |
| m2:warm_neutral-0.2 | +0.351 | 0.382 | -2% |  |
| m1:inquire_proceed-0.1 | +0.343 | 0.302 | +1% |  |
| m1:elaborate_concise+0.1 | +0.282 | 0.243 | +6% |  |
| m2:cautious_direct+0.1 | +0.276 | 0.569 | +38% | ⚑ |
| m2:inquire_proceed+0.1 | +0.202 | 0.372 | +10% |  |
| m2:hedge_assert-0.1 | +0.191 | 0.349 | +6% |  |
| m2:hedge_assert-0.2 | +0.161 | 0.420 | +1% |  |
| m2:hedge_assert+0.1 | +0.120 | 0.368 | +15% |  |
| m1:cautious_direct+0.1 | +0.035 | 0.310 | +26% | ⚑ |
| m2:cautious_direct+0.2 | +0.014 | 0.479 | -0% |  |
| m2:formal_casual-0.1 | -0.011 | 0.384 | +20% |  |
| m2:warm_neutral+0.1 | -0.024 | 0.354 | +6% |  |
| m1:warm_neutral-0.1 | -0.118 | 0.452 | +22% | ⚑ |
| dense:prior | -0.157 | 0.572 | +49% | ⚑ |
| m1:hedge_assert+0.1 | -0.162 | 0.528 | +36% | ⚑ |
| m1:formal_casual+0.1 | -0.173 | 0.397 | +16% |  |
| m2:elaborate_concise-0.1 | -0.240 | 0.406 | +16% |  |
| m2:formal_casual+0.2 | -0.251 | 0.479 | +30% | ⚑ |
| m2:inquire_proceed-0.1 | -0.382 | 0.547 | -1% |  |
| m1:hedge_assert-0.1 | -0.427 | 0.420 | +23% | ⚑ |
| m2:elaborate_concise-0.2 | -0.471 | 0.405 | +9% |  |
| m2:inquire_proceed-0.2 | -0.583 | 0.452 | -5% |  |
| m2:elaborate_concise+0.1 | -0.587 | 0.583 | -4% |  |
| m2:cautious_direct-0.2 | -0.625 | 0.310 | +17% |  |
| m2:elaborate_concise+0.2 | -0.668 | 0.575 | +23% | ⚑ |
| m1:cautious_direct-0.1 | -0.707 | 0.254 | +13% |  |
| m1:warm_neutral+0.1 | -1.001 | 0.479 | +45% | ⚑ |
| m2:warm_neutral+0.2 | -1.036 | 0.535 | +47% | ⚑ |
| m1:elaborate_concise-0.1 | -1.153 | 0.255 | +18% |  |
| m1:inquire_proceed-0.2 | -1.392 | 0.634 | +87% | ⚑ |
| m1:inquire_proceed+0.1 | -1.955 | 0.624 | +89% | ⚑ |
| m1:cautious_direct+0.2 | -2.111 | 0.536 | +170% | ⚑ |
| m1:elaborate_concise+0.2 | -2.281 | 1.084 | +110% | ⚑ |
| m2:formal_casual-0.2 | -2.284 | 0.707 | +71% | ⚑ |
| dense:rand4 | -2.483 | 0.554 | +48% | ⚑ |
| m1:formal_casual+0.2 | -2.583 | 0.639 | +115% | ⚑ |
| m1:hedge_assert+0.2 | -2.614 | 0.633 | +202% | ⚑ |
| m1:hedge_assert-0.2 | -2.724 | 0.364 | +63% | ⚑ |
| m1:formal_casual-0.1 | -2.757 | 0.445 | +77% | ⚑ |
| m2:inquire_proceed+0.2 | -3.367 | 0.835 | +219% | ⚑ |
| m1:elaborate_concise-0.2 | -3.607 | 0.613 | +117% | ⚑ |
| dense:rand0 | -3.762 | 0.821 | +144% | ⚑ |
| m1:inquire_proceed+0.2 | -4.171 | 0.737 | +328% | ⚑ |
| m1:warm_neutral-0.2 | -4.422 | 0.659 | +69% | ⚑ |
| m1:cautious_direct-0.2 | -4.449 | 0.646 | +81% | ⚑ |
| m1:warm_neutral+0.2 | -4.513 | 0.977 | +209% | ⚑ |
| dense:rand2 | -4.629 | 0.971 | +336% | ⚑ |
| dense:rand5 | -5.037 | 0.875 | +413% | ⚑ |
| dense:rand3 | -6.583 | 0.570 | +270% | ⚑ |
| dense:rand1 | -7.443 | 1.182 | +379% | ⚑ |
| m1:formal_casual-0.2 | -9.125 | 0.761 | +339% | ⚑ |

- **best mean condition:** dense:cem_best ΔRM +0.875 ± 0.433 (fluent)
- **oracle per-prompt headroom** (fluent conds): +2.391 (mean best-per-prompt ΔRM)
- **argmax winners:** 15 distinct conditions win across prompts×seeds — prompt-dependent

## large — per-condition paired ΔRM (sorted; ⚑ = perplexity-flagged)

| condition | ΔRM | ±SE | ΔPPL | |
|---|---|---|---|---|
| m2:hedge_assert+0.1 | +0.700 | 0.316 | +57% | ⚑ |
| m2:formal_casual+0.1 | +0.588 | 0.332 | +28% | ⚑ |
| m2:cautious_direct-0.1 | +0.584 | 0.435 | +27% | ⚑ |
| m2:inquire_proceed-0.1 | +0.564 | 0.355 | +28% | ⚑ |
| m2:warm_neutral-0.1 | +0.495 | 0.405 | +31% | ⚑ |
| m2:inquire_proceed-0.2 | +0.445 | 0.398 | +23% | ⚑ |
| m2:warm_neutral-0.2 | +0.256 | 0.577 | +80% | ⚑ |
| m2:hedge_assert-0.1 | +0.084 | 0.392 | +35% | ⚑ |
| m2:cautious_direct-0.2 | -0.164 | 0.767 | +152% | ⚑ |
| m2:warm_neutral+0.1 | -0.254 | 0.270 | +85% | ⚑ |
| m2:formal_casual+0.2 | -0.453 | 0.268 | +260% | ⚑ |
| m2:warm_neutral+0.2 | -0.481 | 0.289 | +123% | ⚑ |
| m2:hedge_assert+0.2 | -0.607 | 0.305 | +201% | ⚑ |
| m2:elaborate_concise-0.1 | -0.622 | 0.876 | +146% | ⚑ |
| m2:cautious_direct+0.1 | -0.633 | 0.319 | +99% | ⚑ |
| m2:cautious_direct+0.2 | -1.017 | 0.363 | +125% | ⚑ |
| m2:elaborate_concise+0.1 | -1.159 | 0.391 | +46% | ⚑ |
| m2:formal_casual-0.1 | -1.539 | 0.480 | +248% | ⚑ |
| m2:elaborate_concise+0.2 | -1.590 | 0.505 | +50% | ⚑ |
| m2:hedge_assert-0.2 | -1.633 | 0.946 | +132% | ⚑ |
| m2:inquire_proceed+0.1 | -1.934 | 0.709 | +175% | ⚑ |
| m2:formal_casual-0.2 | -2.052 | 1.016 | +462% | ⚑ |
| m2:elaborate_concise-0.2 | -2.228 | 0.488 | +175% | ⚑ |
| m2:inquire_proceed+0.2 | -4.054 | 0.721 | +464% | ⚑ |

- **best mean condition:** m2:hedge_assert+0.1 ΔRM +0.700 ± 0.316 (flagged)
- **oracle per-prompt headroom** (fluent conds): +nan (mean best-per-prompt ΔRM)
- **argmax winners:** 0 distinct conditions win across prompts×seeds — concentrated

## Reading
GO if any model shows a fluent condition with ΔRM > ~2·SE and positive oracle headroom — there is something to optimize. REBRIEF if everything is flat or only flagged (degradation) conditions move the RM (early H3 signal); discuss before the full 200-prompt run.
