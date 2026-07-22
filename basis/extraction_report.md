# Basis extraction report

| axis | pairs used | pairs total | proxy pos | proxy neg |
|---|---|---|---|---|
| challenge_accommodate | 83 | 200 | 0.43 | -0.02 |
| hedge_assert | 200 | 200 | 1.24 | 0.16 |
| elaborate_concise | 200 | 200 | 146.00 | 26.47 |
| formal_casual | 200 | 200 | -0.50 | -5.26 |
| cautious_direct | 200 | 200 | 0.78 | 0.04 |
| warm_neutral | 200 | 200 | 1.11 | 0.16 |
| inquire_proceed | 106 | 200 | 3.00 | 0.11 |

## Cosine similarity, layer 12

| | challenge_ac | hedge_assert | elaborate_co | formal_casua | cautious_dir | warm_neutral | inquire_proc |
|---|---|---|---|---|---|---|---|
| challenge_ac | 1.00 | 0.74 | 0.20 | -0.16 | 0.59 | 0.56 | 0.66 |
| hedge_assert | 0.74 | 1.00 | 0.38 | -0.40 | 0.81 | 0.77 | 0.59 |
| elaborate_co | 0.20 | 0.38 | 1.00 | 0.02 | 0.69 | 0.22 | -0.11 |
| formal_casua | -0.16 | -0.40 | 0.02 | 1.00 | -0.17 | -0.64 | -0.29 |
| cautious_dir | 0.59 | 0.81 | 0.69 | -0.17 | 1.00 | 0.56 | 0.32 |
| warm_neutral | 0.56 | 0.77 | 0.22 | -0.64 | 0.56 | 1.00 | 0.61 |
| inquire_proc | 0.66 | 0.59 | -0.11 | -0.29 | 0.32 | 0.61 | 1.00 |

## Cosine similarity, layer 16

| | challenge_ac | hedge_assert | elaborate_co | formal_casua | cautious_dir | warm_neutral | inquire_proc |
|---|---|---|---|---|---|---|---|
| challenge_ac | 1.00 | 0.73 | 0.21 | -0.13 | 0.61 | 0.53 | 0.65 |
| hedge_assert | 0.73 | 1.00 | 0.37 | -0.33 | 0.82 | 0.74 | 0.60 |
| elaborate_co | 0.21 | 0.37 | 1.00 | 0.04 | 0.64 | 0.21 | -0.06 |
| formal_casua | -0.13 | -0.33 | 0.04 | 1.00 | -0.13 | -0.61 | -0.26 |
| cautious_dir | 0.61 | 0.82 | 0.64 | -0.13 | 1.00 | 0.54 | 0.37 |
| warm_neutral | 0.53 | 0.74 | 0.21 | -0.61 | 0.54 | 1.00 | 0.58 |
| inquire_proc | 0.65 | 0.60 | -0.06 | -0.26 | 0.37 | 0.58 | 1.00 |

**COLLAPSE-RULE FLAGS (|cos| >= 0.7 at default layer):** [('challenge_accommodate', 'hedge_assert', np.float32(0.7305497)), ('hedge_assert', 'cautious_direct', np.float32(0.8188211)), ('hedge_assert', 'warm_neutral', np.float32(0.741194))]

## Cosine similarity, layer 20

| | challenge_ac | hedge_assert | elaborate_co | formal_casua | cautious_dir | warm_neutral | inquire_proc |
|---|---|---|---|---|---|---|---|
| challenge_ac | 1.00 | 0.70 | 0.13 | -0.01 | 0.62 | 0.43 | 0.64 |
| hedge_assert | 0.70 | 1.00 | 0.28 | -0.23 | 0.80 | 0.68 | 0.61 |
| elaborate_co | 0.13 | 0.28 | 1.00 | 0.13 | 0.53 | 0.10 | -0.09 |
| formal_casua | -0.01 | -0.23 | 0.13 | 1.00 | -0.04 | -0.58 | -0.20 |
| cautious_dir | 0.62 | 0.80 | 0.53 | -0.04 | 1.00 | 0.48 | 0.42 |
| warm_neutral | 0.43 | 0.68 | 0.10 | -0.58 | 0.48 | 1.00 | 0.53 |
| inquire_proc | 0.64 | 0.61 | -0.09 | -0.20 | 0.42 | 0.53 | 1.00 |
