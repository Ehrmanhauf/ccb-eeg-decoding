# CCB time-window arm-grid sensitivity (BCI-IV-2b screening)

Cells (only ``_TIME_WINDOWS`` varied; everything else held at the
Phase-5 Stage-1 best-factorial CCB cell):

| cell | windows | n_arms (pre-prune) |
|---|---|---|
| full | 0.0-4.0, 0.5-2.5, 1.0-3.0 | 162 |
| 0_4 | 0.0-4.0 | 54 |
| 05_25 | 0.5-2.5 | 54 |
| 1_3 | 1.0-3.0 | 54 |

Mean κ ± std across 9 subjects × seeds × 1 fold:

| cell | within | official |
|---|---|---|
| full | +0.178 ± 0.210 | +0.172 ± 0.193 |
| 0_4 | +0.153 ± 0.223 | +0.148 ± 0.156 |
| 05_25 | +0.173 ± 0.238 | +0.156 ± 0.204 |
| 1_3 | +0.105 ± 0.190 | +0.059 ± 0.135 |

Hyperparameters held fixed at the Phase-5 Stage-1 best cell (α=0.5, calibration_frac=0.3, window_size=50, arm_pool=pruned, include_recent_rewards=False); see open-justifications.md.
