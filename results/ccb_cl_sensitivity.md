# CL Sensitivity Sweeps — STEW + WAUC (Phase F)

Single-parameter sweeps around the locked CCB default (workload context, within-subject
5-fold CV, seeds {0,1,2,3,42}). Locked default: α=0.5, calibration_frac=0.3,
window_size=50, per_round_cap=∞.

Producing scripts: `scripts/run_ccb_stew.py`, `scripts/run_ccb_wauc.py`.
Source CSVs: `results/ccb_{stew,wauc}_sens_{alpha,calibration,window,cap}.csv`.
κ filtered to non-NA rows; mean ± std over subjects × seeds (n reported).

WAUC contributes 43 usable subjects (S39, S48 error out at LDA fit after the ASR
NaN-filter collapses their label distribution to a single class), so n = 215 = 43 × 5.
STEW contributes 45 subjects, n = 225 = 45 × 5.

## Exploration parameter α

| α | STEW κ | WAUC κ |
|---|---|---|
| 0.1 | 0.774 ± 0.247 | 0.456 ± 0.222 |
| 0.5 (default) | 0.744 ± 0.251 | 0.426 ± 0.207 |
| 1.0 | 0.703 ± 0.279 | 0.413 ± 0.206 |
| 2.0 | 0.708 ± 0.275 | 0.411 ± 0.209 |

Lower α better on both, monotone. Best-vs-default: +0.030 (STEW), +0.030 (WAUC). **Driver, both.**

## Calibration fraction

| frac | STEW κ | WAUC κ |
|---|---|---|
| 0.2 | 0.720 ± 0.270 | 0.412 ± 0.212 |
| 0.3 (default) | 0.744 ± 0.251 | 0.426 ± 0.207 |
| 0.5 | 0.778 ± 0.214 | 0.446 ± 0.201 |

Higher frac better on both, monotone. 0.2→0.5 span: +0.058 (STEW), +0.033 (WAUC). **Driver, both.**

## Sliding-window size

| W | STEW κ | WAUC κ |
|---|---|---|
| ∞ (stationary) | 0.744 ± 0.251 | 0.427 ± 0.210 |
| 25 | 0.751 ± 0.248 | 0.445 ± 0.211 |
| 50 (default) | 0.744 ± 0.251 | 0.426 ± 0.207 |
| 100 | 0.744 ± 0.251 | 0.421 ± 0.212 |

Flat for W ≥ 50 on both; small W=25 bump (+0.007 STEW, +0.019 WAUC), within one σ, non-monotone. **Ruled out, both.**

## Per-round cost cap

| cap | STEW κ | WAUC κ |
|---|---|---|
| ∞ (default) | 0.744 ± 0.251 | 0.426 ± 0.207 |
| 4 | 0.741 ± 0.245 | 0.414 ± 0.193 |
| 3 | 0.770 ± 0.255 | 0.452 ± 0.222 |
| 2 | 0.770 ± 0.255 | 0.452 ± 0.222 |

Binding caps {2,3} recover +0.026 over default on both (identical to 2 d.p.); cap=4 ≈ default. **Mild driver, both.**

## Cross-paradigm verdict

| Mechanism (param) | STEW | WAUC |
|---|---|---|
| Exploration cost (α) | Driver | Driver |
| Calibration overhead (frac) | Driver | Driver |
| Non-stationarity (window W) | Ruled out | Ruled out |
| Arm-pool composition (cap) | Mild driver | Mild driver |

The WAUC profile replicates the STEW profile cell-for-cell, despite different subjects,
hardware, tasks, label operationalisations, and leakage status (STEW within-CV is
ceiling-saturated, WAUC is not). The gap mechanism is therefore **paradigm-level**
(a structural cost of online per-trial adaptation), not a dataset artefact. No single
parameter closes the ≈0.23 κ gap to the WAUC fixed pipeline (best single cell: α=0.1, κ=0.456,
still ≈0.20 below B1's κ=0.658).
