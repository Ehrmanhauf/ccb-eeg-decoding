# Best-arm-as-frozen-pipeline diagnostic — Phase 5 (Fig 4.2 / Contribution 4)

**Producers:** `scripts/run_best_arm_diagnostic.py` (new cells, `results/best_arm_diagnostic.csv`,
450 rows, 0 errors) + `scripts/run_best_arm_existing.py` (leakage-clean existing cells,
`results/best_arm_existing.csv`, 270 rows; 10 errors = WAUC S39/S48, the known
single-class-collapse subjects the fixed baselines also drop).

**Method.** For each cell, the *single best arm* in the calibrated pool (the arm with the
highest calibration κ, `surviving[0]` from `fit_heads_on_calibration`) is used as a
**frozen pipeline** — no bandit selection, no exploration — and scored on the held-out test
split. Each dataset uses the *same* arm bank its committed CCB used (`enumerate_arms_2b`
for 2b, `enumerate_arms_generic` otherwise). Within-subject CV, 5 seeds.

## Decomposition: fixed − best-arm (arm-bank gap) vs best-arm − CCB (selection gap)

| Cell | best fixed | best-arm | CCB | **arm-bank gap** (fixed−best-arm) | **selection gap** (best-arm−CCB) |
|---|---|---|---|---|---|
| BCI-IV-2b (clean) | 0.292 | 0.122 | 0.184 | **+0.170** | −0.062 |
| WAUC (clean) | 0.785 | 0.452 | 0.426 | **+0.333** | +0.026 |
| UAB full (leak) | 0.963 | 0.831 | 0.776 | **+0.132** | +0.055 |
| UAB near-ear (leak) | 0.714 | 0.400 | 0.375 | **+0.314** | +0.025 |
| COG-BCI full (leak) | 0.987 | 0.839 | 0.793 | **+0.148** | +0.046 |
| COG-BCI near-ear (leak) | 0.615 | 0.312 | 0.318 | **+0.303** | −0.006 |

(The four "leak" cells are within-CV leak-confounded — best-arm rides the leak like fixed and
CCB, so they read as a *relative* decomposition only. The two clean cells, 2b and WAUC, give
the unconfounded read, and show the same pattern.)

## Finding — Contribution 4 is revised by the data

The thesis (Chapter 4) **asserts** Contribution 4: *"the deficit is the bandit's
selection-and-exploration machinery, not the arm bank's features."* The diagnostic, run to
test exactly that assertion, **does not support it — it reverses it:**

1. **The arm-bank gap dominates on every cell** (+0.13 to +0.33 κ): a single best arm — one
   band, one spatial filter, shrinkage LDA — sits far below the *best of the full B1–B5
   fixed pipeline* (9-band FBCSP + random-forest / SVM). The CCB selects among these single
   arms, so it is structurally capped below the full pipeline by the arm bank's
   representational ceiling.
2. **The selection gap is small and mixed** (−0.06 to +0.07 κ): the bandit's online
   selection adds only a modest cost over a static best arm on WAUC (+0.026), the new full
   cells (+0.05–0.07), and the near-ear cells — and is *negative* on 2b (−0.062) and COG-BCI
   near-ear (−0.006), where the adaptive policy actually **beats** the static best arm.

**Corrected claim:** the CCB-vs-fixed-pipeline deficit is **primarily a property of the
arm-bank design** — each arm is a single-pipeline CSP+LDA classifier, structurally weaker
than the full multi-band FBCSP ensemble (B1) or the SVM/RF heads (B3–B5) — **not** primarily
the bandit's adaptation machinery. The exploration / calibration costs the sensitivity sweeps
identify are real but small (≈ 0.03 κ each); they sit on top of the dominant arm-bank ceiling.
The Chapter-4 Contribution-4 wording must be updated to this (tracked for the Phase-7 prose +
Phase-8 audit). The advisor anticipated this outcome: *"if best-arm sits below the baseline it
honestly shows the arm bank is feature-impoverished, and you adjust the claim — either way it
closes the gap."*

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/run_best_arm_diagnostic.py   # UAB + COG-BCI
PYTHONPATH=src .venv/bin/python scripts/run_best_arm_existing.py --datasets bci2b,wauc
```
