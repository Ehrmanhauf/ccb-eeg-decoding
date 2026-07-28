# CCB spatial-filter family sensitivity (BCI-IV-2b screening)

Cells (only ``_SPATIAL_FILTERS`` varied; everything else held at the
Phase-5 Stage-1 best-factorial CCB cell):

| cell | spatial filters | n_arms (pre-prune) |
|---|---|---|
| full | csp, laplacian, identity | 162 |
| csp_only | csp | 54 |
| laplacian_only | laplacian | 54 |
| identity_only | identity | 54 |

Mean κ ± std across 9 subjects × seeds × 1 fold:

| cell | within | official |
|---|---|---|
| full | +0.178 ± 0.210 | +0.172 ± 0.193 |
| csp_only | +0.175 ± 0.199 | +0.150 ± 0.213 |
| laplacian_only | +0.074 ± 0.161 | +0.047 ± 0.111 |
| identity_only | +0.174 ± 0.223 | +0.145 ± 0.186 |

Hyperparameters held fixed at the Phase-5 Stage-1 best cell (α=0.5, calibration_frac=0.3, window_size=50, arm_pool=pruned, include_recent_rewards=False); see open-justifications.md.
