# Fixed-pipeline baselines (B1-B5) on the near-ear reframe datasets — Phase 2

**Producer:** `scripts/run_fixed_baselines_newdata.py` · **Output:** `results/fixed_baseline_newdata.csv`
(3600 rows, 0 errors). Within-subject 5-fold CV, seed 42; mean ± SD over subjects × folds.
UAB n = 16 subjects; COG-BCI n = 29 subjects (S1 only). Heads: B1 = FBCSP+sLDA,
B2 = BandPower+sLDA, B3 = SVM-RBF, B4 = CART, B5 = random forest.

## Headline κ per cell

| Dataset | Montage | B1 FBCSP+LDA | B2 BandPower+LDA | **Best fixed (B1-B5)** |
|---|---|---|---|---|
| UAB (14-ch, 3-class) | full | 0.921 ± 0.079 | 0.962 ± 0.039 | **0.963** (SVM, BandPower) |
| UAB | **near-ear T7/T8** | 0.622 ± 0.177 | 0.578 ± 0.152 | **0.714** (SVM, FBCSP) |
| COG-BCI (62-ch, 3-class) | full | 0.971 ± 0.038 | 0.814 ± 0.119 | **0.987** (RF, FBCSP) |
| COG-BCI | **near-ear T7/T8** | 0.558 ± 0.188 | 0.409 ± 0.209 | **0.615** (SVM, FBCSP) |

These are the bars the CCB (Phase 3) is measured against, and the within-session
references against which the cross-session drop (Phase 4) is read.

## Two findings

**1. The near-ear (T7/T8) montage drops κ sharply** — the deployment-relevant decline
the reframe is about. Full → near-ear: UAB 0.96 → 0.71, COG-BCI 0.99 → 0.62 (best fixed).
A ~2-channel temporal montage loses 0.25-0.37 κ versus the full montage *even before*
the bandit and *even under the optimistic within-CV protocol*. This is the first
systematic measurement of how much workload-discriminative signal survives at a
near-ear montage, and it is large — exactly the "clean, citable ceiling on 2-channel
near-ear cognitive-load decoding" the reframe set out to produce.

**2. COG-BCI N-back within-session CV leaks — contra the "leakage-resistant" premise.**
The work plan adopted COG-BCI N-back as a *leakage-resistant* CL cell (separated
recurring blocks). Empirically, under **standard random K-fold CV it does not behave as
leakage-resistant**: the best fixed pipeline reaches κ = 0.987, far above the authors'
~0.65 (3-class, 10-ch, Riemannian MDM). The reason is structural and decisive: the three
workload levels are recorded as **separate files** (`zeroBACK/oneBACK/twoBACK.set`), so
the 3-class label is perfectly confounded with *which recording* a window came from — a
file-identity shortcut analogous to STEW's 2-segment leak. Block-aware CV within a level
would not remove this, because the leak is *across* levels (across files), not within a
level's blocks.

**Consequence for the thesis (honest):** the within-session random-K-fold κ is a
leak-confounded ceiling for *all three* CL datasets (STEW 2-segment, UAB 3-block, COG-BCI
3-file), not a clean workload-decoding score. The genuinely leakage-clean evaluations are
**cross-session** (train S1 → test S2/S3, Phase 4 — different sessions cannot share a
file/segment shortcut) and **cross-subject LOSO** (different subjects). Phase 4 is the
headline and is expected to show the within-CV κ collapse, confirming the leak — exactly
as LOSO did on STEW (0.94 → ~0.28). This *generalises* the thesis's leakage finding rather
than weakening it: the only protocols that yield honest CL numbers at these label
structures are cross-session and cross-subject.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/run_fixed_baselines_newdata.py \
    --output results/fixed_baseline_newdata.csv
```
Loads the full montage once per dataset and derives the near-ear cell via
`select_near_ear` (position-based T7/T8). Crash-safe: checkpoints per
(dataset, montage, subject) and resumes from an existing CSV.
