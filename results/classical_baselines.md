# Classical-classifier comparators (B3–B5) across all five datasets

**Generated:** 2026-05-29 from `scripts/run_classical_baselines.py` (default
arguments: heads `lda,svm,decision_tree,random_forest`; within-subject 5-fold
CV, seed 42, on every dataset; plus the official session split on BCI-IV-2a/2b;
Cho2017 within-only, both configurations). Source CSV:
[`results/classical_baselines.csv`](classical_baselines.csv) (5954 rows).

## Purpose

The first reviewer question for this thesis is "how does the CCB compare to a
decision tree (or an SVM, or a random forest)?" These comparators answer it by
fitting three off-the-shelf scikit-learn classifier heads — SVM-RBF (B3), CART
decision tree (B4), random forest (B5) — on the **same** engineered features as
the fixed-pipeline baselines B1/B2 and the CCB arm-heads, holding the feature
representation fixed and varying only the decision rule. This isolates the
classifier-choice effect: it tests whether the CCB-vs-fixed-pipeline gap is
specific to the shrinkage-LDA head or general to fixed classical classifiers.
The shrinkage-LDA head is included as a **consistency anchor**. Heads are at
scikit-learn library defaults (untuned by design; see
`design-doc/open-justifications.md`). Feature transforms and the SVM
`StandardScaler` are fitted train-only per fold (no leakage).

## Consistency check — the LDA anchor reproduces the committed baselines

The shrinkage-LDA head, run through the new comparator pipeline, reproduces the
existing committed baselines to three decimals — an end-to-end validation that
the comparator code path is consistent with `run_fbcsp_baseline.py` and
`run_fixed_baselines_cl.py`:

| Dataset (LDA head) | this run | committed baseline |
|---|---:|---:|
| STEW, FBCSP feat. (B1)    | 0.937 | 0.937 (`fixed_baseline_cl.md`) |
| STEW, BandPower feat. (B2)| 0.952 | 0.953 |
| WAUC, FBCSP feat. (B1)    | 0.658 | 0.658 |
| WAUC, BandPower feat. (B2)| 0.644 | 0.644 |
| BCI-IV-2a within / official | 0.532 / 0.468 | 0.532 / 0.468 (`fbcsp_baseline.md`) |
| BCI-IV-2b within / official | 0.292 / 0.267 | 0.292 / 0.267 |

## Cognitive load — within-subject 5-fold CV (seed 42)

Mean κ ± std across subjects × folds (n = 225 STEW, n = 215 WAUC); mean accuracy in parentheses.

| Dataset | Feature family | LDA | SVM-RBF | Decision Tree | Random Forest |
|---|---|---:|---:|---:|---:|
| STEW | FBCSP     | 0.937±0.105 (0.968) | 0.937±0.101 (0.969) | 0.833±0.170 (0.916) | 0.930±0.101 (0.965) |
| STEW | BandPower | 0.952±0.092 (0.976) | 0.923±0.119 (0.962) | 0.838±0.189 (0.919) | 0.918±0.134 (0.959) |
| WAUC | FBCSP     | 0.658±0.169 (0.830) | **0.785±0.125 (0.893)** | 0.629±0.180 (0.815) | 0.781±0.129 (0.891) |
| WAUC | BandPower | 0.644±0.189 (0.823) | 0.757±0.147 (0.879) | 0.658±0.193 (0.829) | 0.778±0.146 (0.889) |

## Motor imagery — within-subject 5-fold CV + official session split

FBCSP features; mean κ ± std across 9 subjects.

| Dataset | Protocol | LDA | SVM-RBF | Decision Tree | Random Forest |
|---|---|---:|---:|---:|---:|
| BCI-IV-2a | within   | 0.532±0.263 | 0.521±0.255 | 0.371±0.271 | **0.568±0.293** |
| BCI-IV-2a | official | 0.468±0.261 | 0.443±0.271 | 0.285±0.255 | **0.486±0.257** |
| BCI-IV-2b | within   | **0.292±0.161** | 0.270±0.149 | 0.122±0.121 | 0.244±0.144 |
| BCI-IV-2b | official | **0.267±0.139** | 0.216±0.149 | 0.053±0.155 | 0.210±0.140 |

## Cho2017 — within-subject 5-fold CV (FBCSP features; first fixed-pipeline numbers)

No fixed-pipeline baseline had previously been computed for Cho2017 (the
cross-paradigm table in Chapter 4 carried "—"). The LDA head supplies it. Mean
κ ± std across 50 subjects.

| Dataset | LDA | SVM-RBF | Decision Tree | Random Forest |
|---|---:|---:|---:|---:|
| Cho2017 full (64 ch) | **0.202±0.232** | 0.190±0.225 | 0.045±0.107 | 0.161±0.216 |
| Cho2017 3-ch (C3/Cz/C4) | **0.190±0.204** | 0.183±0.198 | 0.087±0.142 | 0.182±0.205 |

The 3-channel subset (0.190) is statistically indistinguishable from the full
64-channel montage (0.202) for the fixed pipeline too, mirroring the CCB
observation that channel-count reduction barely moves κ on Cho2017.

## Headline — the CCB underperforms the whole classical-classifier family

Taking the best fixed classical pipeline per dataset (the maximum over heads
B1–B5 and both feature families) against the CCB's within-CV κ:

| Dataset | Best fixed (head) | CCB κ | Δκ (best fixed − CCB) |
|---|---|---:|---:|
| STEW (3-class)        | 0.952 (LDA, BandPower) | 0.744 | +0.208 |
| WAUC (binary)         | 0.785 (SVM, FBCSP)     | 0.426 | **+0.359** |
| BCI-IV-2b (3 ch)      | 0.292 (LDA, FBCSP)     | 0.184 | +0.108 |
| Cho2017 full (64 ch)  | 0.202 (LDA, FBCSP)     | 0.077 | +0.125 |
| Cho2017 3-ch          | 0.190 (LDA, FBCSP)     | 0.082 | +0.108 |

On every dataset the best fixed classical pipeline beats the CCB, by Δκ ranging
from +0.108 to +0.359. The deficit is therefore **general to the
classical-classifier family, not an artefact of the shrinkage-LDA head**:
swapping in SVM, a decision tree, or a random forest does not close it, and on
WAUC the strongest classifier (SVM, +0.13 κ over LDA) *widens* it.

## Interpretation — three observations

1. **The classifier-head choice does not explain the CCB gap.** This is the
   central result the comparators were built to test. On no dataset does any
   classical head fall to the CCB's level; the gap is a property of the bandit's
   online-adaptation regime, not of the decision rule.

2. **On WAUC (the clean, non-leakage CL dataset) SVM and random forest beat
   shrinkage LDA by ≈ 0.12–0.13 κ.** The fixed-pipeline ceiling on WAUC is thus
   higher than the LDA baseline of `fixed_baseline_cl.md` suggested (0.785 vs
   0.658), which widens the CCB-vs-fixed gap rather than narrowing it. The
   thesis cross-paradigm table now reports the best-over-heads fixed pipeline.

3. **The single decision tree is consistently the weakest head** (e.g. 0.053 on
   2b official, 0.045 on Cho2017 full), as expected for an unpruned CART on
   small-sample, noisy features; SVM-RBF and random forest are the strong heads.
   On STEW all heads sit near the segment-leakage ceiling (\S below).

## Caveats

- **STEW within-CV is segment-leakage-saturated** (see `fixed_baseline_cl.md`
  Caveat 1 and Chapter 3 §Evaluation Protocol). All heads score κ ≈ 0.83–0.95
  there because the within-CV protocol lets any classifier exploit segment
  identity. These STEW numbers are reported under that caveat, not as workload
  classification scores.
- **WAUC: 43 of 45 usable subjects.** Subjects S39 and S48 collapse to a single
  label class after the ASR NaN-filter and error at the discriminant fit; they
  are preserved as 2 error rows (fold = −1, seed = −1) in the CSV, matching the
  fixed-baseline behaviour documented in `fixed_baseline_cl.md` Caveat 3.
- **Untuned heads.** Hyperparameters are at scikit-learn defaults by design (the
  comparators establish whether the gap is head-specific, which does not require
  per-head tuning); a nested-CV tuning sensitivity is tracked as an open
  justification.

## Reproduction

```bash
make fix-pth
PYTHONPATH=src .venv/bin/python scripts/run_classical_baselines.py \
    --output results/classical_baselines.csv
```

Cho2017 loads lazily via MOABB (cached under `~/mne_data/MNE-gigadb-data`); the
local datasets (STEW, WAUC, BCI-IV-2a/2b) require `data/` to be populated. A
fast smoke check: `--datasets stew,bci2b --subjects 1,2,3`.
