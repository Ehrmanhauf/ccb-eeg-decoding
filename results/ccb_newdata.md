# CCB within-session on the near-ear reframe datasets — Phase 3

**Producer:** `scripts/run_ccb_newdata.py` · **Output:** `results/ccb_newdata.csv`
(450 rows, 0 errors). Locked cell: α=0.5, calibration 0.3, sliding window 50, cap ∞,
workload context, `include_recent_rewards=False`, `n_components=4` (auto-capped to the
channel count). 5 seeds {0,1,2,3,42}, one within-subject fold. COG-BCI session S1 only.

## CCB κ vs the Phase-2 fixed-pipeline bar (within-session)

| Dataset | Montage | Best fixed (B1-B5) | **CCB κ** | **Δκ (fixed − CCB)** |
|---|---|---|---|---|
| UAB | full (14-ch) | 0.963 | 0.776 ± 0.128 | **+0.187** |
| UAB | near-ear T7/T8 | 0.714 | 0.375 ± 0.143 | **+0.339** |
| COG-BCI | full (62-ch) | 0.987 | 0.793 ± 0.186 | **+0.194** |
| COG-BCI | near-ear T7/T8 | 0.615 | 0.318 ± 0.186 | **+0.297** |

## Findings

**The CCB underperforms the fixed pipeline on every new cell**, within-session, by
+0.19 to +0.34 κ — reproducing the thesis's cross-paradigm negative-result direction on
two new cognitive-load datasets at both the full and the near-ear montage. The gap is
**larger at near-ear** (+0.30, +0.34) than at full montage (+0.19, +0.19): when the signal
is already scarce (2 channels), the bandit's calibration-reservation + exploration
overhead bites proportionally harder.

**Per-seed variance is high on the wide montage** (COG-BCI full σ = 0.186 across
subjects×seeds; single-subject seed spreads of 0.2–0.7 are common): the OPLB bandit over
~100 arms on a 62-channel calibration set with a short within-subject stream is unstable
— itself consistent with the "exploration cost dominates at short horizon" mechanism the
thesis identifies on MI.

**Caveat (carried from Phase 2):** the within-session numbers are leak-confounded for both
new datasets (UAB 3-block, COG-BCI 3-file label structure) — both fixed *and* CCB ride the
same file/segment-identity ceiling. The Δκ here is a clean *relative* comparison (both
sides leak equally), but the *absolute* κ is not an honest workload-decoding score. The
leakage-clean evaluation is the cross-session test (Phase 4) and cross-subject LOSO.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/run_ccb_newdata.py --output results/ccb_newdata.csv
```
Crash-safe: checkpoints per (dataset, montage, subject, seed), resumes from an existing CSV.
