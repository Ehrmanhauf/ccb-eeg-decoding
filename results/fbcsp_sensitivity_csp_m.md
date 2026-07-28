# FBCSP CSP component-count (m) sensitivity sweep

Tested values: m ∈ {1, 2, 3, 4}. 2b capped at m=3 (only 3 channels).

| Dataset | Protocol | m=1 | m=2 | m=3 | m=4 |
|---|---|---|---|---|---|
| BCI-IV-2a | within | 0.467 ± 0.223 | 0.518 ± 0.253 | 0.542 ± 0.264 | 0.532 ± 0.248 |
| BCI-IV-2a | official | 0.369 ± 0.219 | 0.401 ± 0.240 | 0.427 ± 0.236 | 0.468 ± 0.246 |
| BCI-IV-2b | within | 0.215 ± 0.136 | 0.291 ± 0.151 | 0.292 ± 0.152 | — |
| BCI-IV-2b | official | 0.199 ± 0.124 | 0.248 ± 0.139 | 0.267 ± 0.131 | — |

Values are mean ± std of per-subject κ over 9 subjects. Phase-2 baseline used m=2 (see results/fbcsp_baseline.md).
