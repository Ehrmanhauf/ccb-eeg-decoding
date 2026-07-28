# Cross-paradigm gap-mechanism comparison: cognitive load vs motor imagery

**Question.** The thesis establishes (Chapter 4 §results-mechanism-summary) that the
CCB-vs-fixed-pipeline gap-mechanism profile *replicates cell-for-cell across the two
cognitive-load datasets* (STEW, WAUC) — exploration cost and calibration overhead are
genuine drivers, arm-pool composition a mild driver, non-stationarity ruled out — making
it a *paradigm-level* property of the CCB-on-cognitive-load setting. Does that profile
also hold on the motor-imagery entry case (BCI-IV-2b)?

This file answers it from the **already-committed** single-parameter sweep CSVs — no new
experiments. All figures are within-subject mean κ (the only protocol the CL datasets
have), computed identically across datasets.

## Source CSVs (committed)

| Mechanism | swept param | 2b (MI) | STEW | WAUC |
|---|---|---|---|---|
| Exploration cost | `alpha` | `ccb_sens_alpha.csv` | `ccb_stew_sens_alpha.csv` | `ccb_wauc_sens_alpha.csv` |
| Calibration overhead | `calibration_frac` | `ccb_sens_calibration.csv` | `ccb_stew_sens_calibration.csv` | `ccb_wauc_sens_calibration.csv` |
| Arm-pool composition | `per_round_cap` | `ccb_sens_perround.csv` | `ccb_stew_sens_cap.csv` | `ccb_wauc_sens_cap.csv` |
| Non-stationarity | `window_size` | `ccb_nonstationary.csv` (official) | `ccb_stew_sens_window.csv` | `ccb_wauc_sens_window.csv` |

Locked default cell: α = 0.5, calibration fraction = 0.3, per-round cap = ∞.

## Per-mechanism Δκ (gain over the locked default), within-subject

| Mechanism (CL-favourable tuning) | STEW | WAUC | **2b (MI)** | transfers? |
|---|---|---|---|---|
| Exploration cost (α 0.1 vs 0.5)        | +0.030 | +0.030 | **−0.093** | **reversed** |
| Calibration overhead (frac 0.5 vs 0.3) | +0.034 | +0.019 | +0.045 | **yes** |
| Arm-pool cap (cap 2 vs ∞)              | +0.026 | +0.026 | **−0.037** | **reversed** |
| Non-stationarity (sliding window)      | +0.007 (within, flat) | +0.018 (within, flat) | n/a within; **+0.049 official** | MI-only role |

(2b α sweep n = 45; calibration n = 27; cap n = 180; CL sweeps n = 675–900. 2b
non-stationarity was swept on the official session-gap protocol only — the within
protocol has no session gap to forget.)

## Finding: the CL mechanism profile does **not** transfer cleanly to MI

- **Exploration cost reverses.** On both CL datasets, lowering α below the default
  recovers κ (+0.030). On 2b the default α = 0.5 is the optimum; lowering it to 0.1
  *costs* 0.093 κ. The exploration optimum is paradigm-dependent.
- **Arm-pool cap reverses.** A tight cost cap recovers +0.026 κ on both CL datasets; on
  2b the same tightening *costs* 0.037 κ (the leakage-pruning rationale that makes a tight
  cap helpful on CL has no honest-signal analog on 2b).
- **Calibration overhead is the one shared driver.** Raising the calibration fraction
  above the default helps on all three (+0.045 on 2b, comparable to CL) — consistent with
  it being the most intrinsic cost of the online-adaptation regime (the policy must
  reserve trials to fit the per-arm heads regardless of paradigm). On 2b the curve is
  non-monotone (the 0.3 default is a local dip, n = 27, noisy), but the higher-than-default
  direction agrees with CL.
- **Non-stationarity is MI-specific.** Ruled out within-subject on all three datasets, it
  becomes a genuine driver only on the MI *official* session-gap protocol (+0.049 at a
  50-round sliding window), which the single-session CL datasets have no analog for.

**Verdict:** two of the four mechanisms reverse direction between cognitive load and
motor imagery, one (calibration overhead) is shared, and one (non-stationarity) acquires
an MI-only role. The "paradigm-level" mechanism profile the thesis establishes across
STEW and WAUC is therefore a property of the **cognitive-load** paradigm, not a universal
property of the CCB — it does not extrapolate to the MI entry case. This vindicates the
thesis's per-paradigm-diagnosis stance: the gap must be re-diagnosed per paradigm, not
inferred from one.

**Caveat (confound).** 2b is also the lowest within-subject-κ cell of the panel
(≈ 0.18, vs ≈ 0.43 WAUC and ≈ 0.74 STEW), and two of its sweeps are smaller-n. The
CL-vs-MI difference is therefore partly confounded with signal strength and is noisier on
the MI side; the directionally-clear reversals (exploration −0.093, arm-pool −0.037) are
the dependable results, the calibration non-monotonicity the least.

## Reproduction

```bash
# Recomputes every figure above from the committed sweep CSVs.
PYTHONPATH=src .venv/bin/python - <<'PY'
import pandas as pd, numpy as np
def mean_by(f, col, prot='within', fixwin=False):
    d = pd.read_csv(f)
    if 'protocol' in d.columns: d = d[d.protocol == prot]
    if fixwin and 'discount_gamma' in d.columns: d = d[d.discount_gamma == 1.0]
    s = d.groupby(col)['kappa'].mean()
    s.index = [np.inf if str(i).strip().lower()=='inf' else float(i) for i in s.index]
    return s.sort_index()
# e.g. exploration on 2b:
print(mean_by('results/ccb_sens_alpha.csv', 'alpha'))
PY
```
