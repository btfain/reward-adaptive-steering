# Stage B report — fixed global steering

Best action (combined norm 0.091):

- hedge_assert: -0.038
- elaborate_concise: +0.085
- formal_casual: +0.015
- cautious_direct: +0.003
- warm_neutral: +0.037
- inquire_proceed: -0.027

| arm | seed | mean RM | mean words | distinct-1 | distinct-2 |
|---|---|---|---|---|---|
| sft | 0 | -0.424 | 116 | 0.387 | 0.778 |
| sft | 1 | -0.058 | 111 | 0.393 | 0.795 |
| fixed | 0 | -0.198 | 126 | 0.373 | 0.779 |
| fixed | 1 | -0.058 | 124 | 0.392 | 0.804 |

mean KL(steered||base) per token, fixed arm seed 0: 0.0622 nats

**fixed − sft held-out mean RM: +0.113**
