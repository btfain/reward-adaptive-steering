# Basis extraction report

| axis | pairs used | pairs total | proxy pos | proxy neg |
|---|---|---|---|---|
| hedge_assert | 200 | 200 | 1.01 | 0.29 |
| elaborate_concise | 200 | 200 | 157.19 | 19.44 |
| formal_casual | 200 | 200 | -0.28 | -5.09 |
| cautious_direct | 200 | 200 | 0.86 | 0.05 |
| warm_neutral | 200 | 200 | 0.55 | 0.05 |
| inquire_proceed | 197 | 200 | 5.30 | 0.10 |

## Cosine similarity, layer 14

| | hedge_assert | elaborate_co | formal_casua | cautious_dir | warm_neutral | inquire_proc |
|---|---|---|---|---|---|---|
| hedge_assert | 1.00 | 0.41 | -0.06 | 0.83 | 0.42 | 0.44 |
| elaborate_co | 0.41 | 1.00 | 0.28 | 0.62 | 0.08 | -0.17 |
| formal_casua | -0.06 | 0.28 | 1.00 | 0.19 | -0.61 | -0.25 |
| cautious_dir | 0.83 | 0.62 | 0.19 | 1.00 | 0.24 | 0.18 |
| warm_neutral | 0.42 | 0.08 | -0.61 | 0.24 | 1.00 | 0.55 |
| inquire_proc | 0.44 | -0.17 | -0.25 | 0.18 | 0.55 | 1.00 |

## Cosine similarity, layer 18

| | hedge_assert | elaborate_co | formal_casua | cautious_dir | warm_neutral | inquire_proc |
|---|---|---|---|---|---|---|
| hedge_assert | 1.00 | 0.47 | -0.10 | 0.85 | 0.44 | 0.45 |
| elaborate_co | 0.47 | 1.00 | 0.24 | 0.65 | 0.11 | -0.15 |
| formal_casua | -0.10 | 0.24 | 1.00 | 0.14 | -0.57 | -0.26 |
| cautious_dir | 0.85 | 0.65 | 0.14 | 1.00 | 0.25 | 0.19 |
| warm_neutral | 0.44 | 0.11 | -0.57 | 0.25 | 1.00 | 0.57 |
| inquire_proc | 0.45 | -0.15 | -0.26 | 0.19 | 0.57 | 1.00 |

**COLLAPSE-RULE FLAGS (|cos| >= 0.7 at default layer):** [('hedge_assert', 'cautious_direct', np.float32(0.8474596))]

## Cosine similarity, layer 22

| | hedge_assert | elaborate_co | formal_casua | cautious_dir | warm_neutral | inquire_proc |
|---|---|---|---|---|---|---|
| hedge_assert | 1.00 | 0.55 | 0.02 | 0.86 | 0.40 | 0.44 |
| elaborate_co | 0.55 | 1.00 | 0.28 | 0.72 | 0.13 | -0.05 |
| formal_casua | 0.02 | 0.28 | 1.00 | 0.22 | -0.54 | -0.18 |
| cautious_dir | 0.86 | 0.72 | 0.22 | 1.00 | 0.23 | 0.23 |
| warm_neutral | 0.40 | 0.13 | -0.54 | 0.23 | 1.00 | 0.54 |
| inquire_proc | 0.44 | -0.05 | -0.18 | 0.23 | 0.54 | 1.00 |
