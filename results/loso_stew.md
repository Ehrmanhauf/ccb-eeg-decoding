# Leave-One-Subject-Out (LOSO) on STEW — within-segment-leakage test

**Producer:** `scripts/run_loso_stew.py` · **Per-row output:** `results/loso_stew.csv`
**Protocol primitive:** `thesis.protocols.leave_one_subject_out` · **Design doc:** `ccb-formulation.md` §8.1 (protocol 3)

## Why this experiment

STEW's within-subject CV is structurally ceiling-saturated. The loader assigns *one*
workload bin to every epoch of a subject's 2.5-minute rest segment and *one* bin to
every epoch of the multitask segment (`stew_load.py`), so the label is constant within
a continuous recording. Random K-fold CV then trains and tests on epochs cut from the
*same* two segments, and any classifier can score high by recognising "which of my two
segments is this window from" — a segment-identity task, not workload decoding. This
inflates within-CV κ to ≈ 0.94 (Chapter 4, Table `tab:fixed-baselines`).

LOSO removes the leakage by construction: pool all 45 usable subjects, hold one out for
test, train on the other 44. The held-out subject's segments are unseen at fit time, so
the segment-identity shortcut is unavailable and the reported κ measures genuine
cross-subject workload decoding — hard mode.

## Configuration

- **Subjects:** 45 (STEW minus 5/24/42, which lack ratings). 45 LOSO folds.
- **Fixed baselines (B1 FBCSP+sLDA, B2 BandPower+sLDA):** fitted on the full pooled
  44-subject training set, scored on the held-out subject. Deterministic — no seed.
- **CCB (OPLB):** re-cast as a population-trained cross-subject policy — calibration +
  bandit stream drawn from the pooled training subjects, frozen policy evaluated on the
  held-out subject. α = 0.5, calibration fraction 0.3, sliding window 50, workload
  context. Under the matched-conditions discipline (`src/thesis/matched.py`) the per-fold
  training pool is capped at **4,000 epochs** identically for all three method families —
  effectively the full ≈ 3.3k-epoch held-in set for STEW — and the CCB is averaged over
  **5 seeds** (0, 1, 2, 3, 42). The subsample is training-subjects-only (no leakage).
- Two κ figures reported: **per-subject** mean ± std over the 45 held-out subjects
  (0 degenerate single-class folds — every subject's two segments fall in different
  bins), and **pooled** global κ over all held-out predictions jointly (the more robust
  figure, since each subject's test set is coarse, ≤ 2 classes over ≈ 74 epochs).

## Result

| Method | within-CV κ | LOSO per-subject κ | LOSO pooled κ | LOSO acc | Δκ (within−LOSO) |
|---|---|---|---|---|---|
| B1 FBCSP + sLDA     | 0.937 ± 0.105 | 0.277 ± 0.260 | 0.277 | 0.550 | −0.660 |
| B2 BandPower + sLDA | 0.953 ± 0.092 | 0.328 ± 0.262 | 0.314 | 0.571 | −0.625 |
| CCB (OPLB)          | 0.744 ± 0.251 | 0.235 ± 0.297 | 0.243 | 0.538 | −0.509 |

Pooled test-label distribution across the 45 held-out subjects: low 1554, medium 851,
high 925 (3-class).

**Dispersion convention (this table vs. Chapter 4).** The CCB is run at 5 seeds, so two
different standard deviations are defensible and they must not be confused. The `± 0.297`
above is the spread over all 225 subject×seed rows (`ddof=1`). Chapter 4 Table
`tab:loso-stew` instead reports `± 0.267`, the spread over the 45 per-subject means
(`ddof=1`) — the correct figure there, because the subject is the unit of analysis and
seed variation is averaged out before the spread is taken. Both derive from
`results/loso_stew.csv`; the deterministic fixed baselines carry no seed, so their single
value is unambiguous.

## Findings

1. **The within-CV ceiling was mostly segment-leakage — confirmed.** Every method loses
   ≈ 0.6 κ when the held-out subject's segments are removed from training. The fixed
   pipelines fall from ≈ 0.94 to ≈ 0.28–0.33; the within-CV κ ≈ 0.94 was a
   segment-identity score, not a workload-classification score, exactly as the thesis
   argued under caveat.
2. **A modest but genuine cross-subject workload signal survives.** LOSO κ ≈ 0.28–0.33
   for the fixed pipelines is well above the chance floor (κ = 0): STEW contains real,
   person-independent workload structure — it is not pure noise. The honest cross-subject
   number sits below the published Lim 2018 within-subject κ = 0.46 (a different protocol).
3. **The CCB still underperforms the fixed pipelines under LOSO** (pooled 0.243 vs
   0.277/0.314; per-subject 0.235, below the best fixed pipeline by ≈ 0.07 κ),
   reproducing the within-CV and WAUC ordering now that the matched-conditions discipline
   trains all three methods on the identical pool. The gap is narrower in this
   cross-subject regime than within-CV, but consistently signed — the bandit's
   calibration-reservation + exploration overhead is not recovered. A percentile bootstrap
   over subjects places the paired Δκ 95 % CI at [+0.05, +0.14] (excludes zero).
4. **Under the honest protocol, the CL-specific BandPower features edge ahead of the
   MI-derived FBCSP features** (0.328 vs 0.277, +0.051 κ) — a separation that the
   leakage-saturated within-CV protocol hid (there both sat at ≈ 0.94, indistinguishable).

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/run_loso_stew.py \
    --ccb-seeds 0,1,2,3,42 --output results/loso_stew.csv   # --train-cap defaults to 4000
```

The CSV stores per-fold `y_true_seq` / `y_pred_seq`, so the pooled κ is recomputable
from the committed file without re-running. The runner checkpoints after every method
and seed, so the artifact survives an interrupted run.
