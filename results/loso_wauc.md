# Leave-One-Subject-Out (LOSO) on WAUC — block-design-leakage test

**Producer:** `scripts/run_loso_wauc.py` · **Per-row output:** `results/loso_wauc.csv`
**Protocol primitive:** `thesis.protocols.leave_one_subject_out` · **Design doc:** `ccb-formulation.md` §8.1 (protocol 3)

## Why this experiment

WAUC's within-subject CV is block-design-confounded in the same way as STEW. The loader
(`wauc_load.py`) assigns *one* binary mental-workload label to every epoch of a subject's
session, and each subject contributes six sessions (2 mental-workload × 3 physical-exertion
cells). Random K-fold CV then trains and tests on epochs cut from the *same* sessions, so a
classifier can score by session identity ("which of my six sessions is this window from")
rather than by workload. The within-CV κ ≈ 0.65 (not ceiling-saturated only because six
harder-to-separate blocks leak less completely than STEW's two) is consequently a *partial*
recording-identity score, not a clean workload score. A moderate κ is therefore **not**
evidence that a block-designed cell is leakage-free.

LOSO removes the leakage by construction: pool all subjects, hold one out for test, train on
the others. The held-out subject's six sessions are unseen at fit time, so session
recognition is impossible and the reported κ measures genuine cross-subject workload
decoding. (LOSO is chosen over leave-one-session-out because each WAUC session is
single-MW-class, which makes a per-session test fold degenerate for κ; a held-out subject
always contains both classes.)

## Configuration

- **Subjects:** 45 loaded → 45 LOSO folds; **43 usable** (S39 and S48 lose enough epochs to
  the strict NaN-from-ASR filter that their held-out fold collapses to a single class, so
  their κ is undefined and excluded — counted, not silently dropped).
- **Fixed baselines (B1 FBCSP+sLDA, B2 BandPower+sLDA):** fitted on a class-stratified
  4,000-epoch subsample of the pooled training subjects, scored on the held-out subject.
  Deterministic — no seed. WAUC pools ≈ 34,000 epochs across 44 training subjects (≈ 10×
  STEW); the subsample keeps the nine-band FBCSP fit tractable and, drawn from training
  subjects only, carries no leakage.
- **CCB (OPLB):** population-trained cross-subject policy — calibration + bandit stream drawn
  from the pooled training subjects, frozen policy evaluated on the held-out subject. α = 0.5,
  calibration fraction 0.3, sliding window 50, workload context. Under the matched-conditions
  discipline (`src/thesis/matched.py`) the per-fold training pool is capped at **4,000 epochs**
  identically for all three method families (training-subjects-only, no leakage) and the CCB is
  averaged over **5 seeds** (0, 1, 2, 3, 42).
- Two κ figures: **per-subject** mean ± std over the 43 usable held-out subjects, and
  **pooled** global κ over all held-out predictions jointly (the robust headline).

## Result

| Method | within-CV κ | LOSO per-subject κ | LOSO pooled κ | Δκ (within−LOSO) |
|---|---|---|---|---|
| B1 FBCSP + sLDA     | 0.658 ± 0.169 | 0.006 ± 0.126 | −0.002 | −0.652 |
| B2 BandPower + sLDA | 0.644 ± 0.189 | 0.006 ± 0.174 | −0.005 | −0.638 |
| CCB (OPLB)          | 0.426 ± 0.207 | 0.002 ± 0.095 | −0.003 | −0.424 |

Pooled n = 37,436 held-out predictions per method (all 45 folds; S39 and S48 are
excluded from the per-subject mean but their predictions still contribute to the
pooled κ). Per-subject κ is the mean ± standard deviation (population, ddof = 0,
matching `loso_stew.md`) over the 43 usable subjects.

## Findings

1. **The within-CV κ ≈ 0.65 was almost entirely session leakage.** Every method falls to the
   chance floor (per-subject κ ≈ 0.006, pooled κ within 0.007 of zero) once the held-out
   subject's six sessions leave the training set. The within-CV κ was a session-identity
   score, not a workload-classification score — confirming the block-design-leakage caveat.
2. **WAUC carries no person-independent workload signal at 8 channels.** Unlike STEW
   (cross-subject fixed-pipeline κ ≈ 0.28–0.33 survives LOSO), WAUC's clean cross-subject κ
   is indistinguishable from chance: there is no transferable binary-workload structure in the
   8-channel Enobio montage once session identity is removed.
3. **The CCB still does not exceed the fixed pipelines** (all three at the floor), reproducing
   the panel-wide direction under the leakage-clean protocol.

The two datasets bracket the leakage-clean cognitive-load regime: STEW retains a little
cross-subject structure, WAUC none. On neither does the bandit recover the decoding the
within-CV ceiling appeared to promise.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/run_loso_wauc.py \
    --ccb-seeds 0,1,2,3,42 --train-cap 4000 \
    --output results/loso_wauc.csv
```

The CSV stores per-fold `y_true_seq` / `y_pred_seq`, so the pooled κ is recomputable from the
committed file without re-running. The runner checkpoints after every method and seed, so the
artifact survives an interrupted run.
