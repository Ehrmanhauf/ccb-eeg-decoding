# Phase C — Fixed-pipeline CL baselines on STEW and WAUC

**Generated:** 2026-05-19 from `scripts/run_fixed_baselines_cl.py` (default
arguments: 5-fold within-subject CV, single seed = 42, both datasets,
both baselines). Source CSV: [`results/fixed_baseline_cl.csv`](fixed_baseline_cl.csv) (884 rows).

## Per-(dataset, baseline) summary

| Dataset | Baseline | Mean κ | Std κ | Mean accuracy | Std accuracy | n_evaluations |
|---|---|---:|---:|---:|---:|---:|
| STEW | B1: FBCSP + sLDA       | **0.9366** | 0.1054 | 0.9684 | 0.0525 | 225 |
| STEW | B2: BandPower + sLDA   | **0.9525** | 0.0924 | 0.9763 | 0.0462 | 225 |
| WAUC | B1: FBCSP + sLDA       | **0.6580** | 0.1685 | 0.8296 | 0.0840 | 215 |
| WAUC | B2: BandPower + sLDA   | **0.6440** | 0.1891 | 0.8225 | 0.0945 | 215 |

Per-subject κ distribution (subject means across folds):

| Dataset | Baseline | min | 25% | median | 75% | max |
|---|---|---:|---:|---:|---:|---:|
| STEW | B1: FBCSP    | 0.429 | 0.867 | 1.000 | 1.000 | 1.000 |
| STEW | B2: BandPower| 0.595 | 0.867 | 1.000 | 1.000 | 1.000 |
| WAUC | B1: FBCSP    | 0.200 | 0.554 | 0.644 | 0.762 | 1.000 |
| WAUC | B2: BandPower| 0.174 | 0.499 | 0.652 | 0.760 | 1.000 |

## Published reference baselines

| Dataset | Reference | Protocol | Reference κ |
|---|---|---|---:|
| STEW | Lim et al. 2018 [`lim2018stew`] (SVR + NCA, 3-class) | not directly comparable to within-CV used here; the published number uses a different evaluation protocol | 0.46 |
| WAUC | Albuquerque et al. 2020 [`albuquerque2020wauc`] | reference κ to be verified directly against the paper body before any thesis claim that depends on it | TBV |

## Interpretation — critical caveats

### Caveat 1 — STEW within-CV is methodologically saturated (segment-leakage)

Both baselines on STEW score within ~0.01 κ of the K = 1.0 ceiling. This is **not a real workload-classification score** in the sense of Lim 2018's κ = 0.46. The reason is structural:

- STEW provides only two long EEG segments per subject (a 2.5-min rest and a 2.5-min SIMKAP multitask block), each carrying a single rating that the loader maps to one workload class.
- After 4 s windowing the subject has ≈ 74 epochs: ≈ 37 labelled by the rest rating and ≈ 37 by the multitask rating.
- Random K-fold within-subject CV draws train and test trials from both segments simultaneously. A classifier can therefore distinguish "epoch from segment A" from "epoch from segment B" using any within-segment correlated noise (drift, residual artifact, subject-specific oscillatory state), without learning a *workload* representation. The labels happen to coincide with the segment identity.

The observed κ ≈ 0.95 reflects this segment-discrimination signal more than workload per se. The within-CV STEW κ is therefore reported here as **evidence of the protocol issue, not as a baseline for the CCB to beat**.

Implications for the rest of the plan:

1. **STEW evaluation should additionally include a leave-one-subject-out (LOSO) protocol**, which would remove the within-segment leakage by holding out a subject's entire 74-epoch budget and training on the remaining 44 subjects. Adding this to `thesis.protocols` is a tracked follow-up.
2. The Phase D / E CCB experiments on STEW will inherit the same within-CV semantics; their κ will need to be interpreted in the same caveated way.
3. The MI 2b numbers from `results/ccb_baseline.csv` use a comparable within-CV protocol on 2b, but the MI streams there contain ~120 trials with labels distributed across multiple class-balanced events per session — so the segment-leakage failure mode does not apply identically on 2b.

### Caveat 2 — WAUC within-CV produces meaningful, non-saturated numbers

WAUC has 6 sessions per subject with alternating low / high MW labels (mw_labels ∈ {0, 1}). After windowing each subject has hundreds of epochs across both label classes; random K-fold CV samples both classes from multiple sessions within the same fold. The within-segment leakage failure mode does not apply at the same intensity, and the resulting κ ≈ 0.65 ± 0.18 looks like a real workload-classification number.

WAUC's B1 (FBCSP + sLDA) is **+0.014 κ above** B2 (BandPower + sLDA) on average, well within one standard deviation of either cell. The two baselines are statistically indistinguishable at this protocol. This is itself a per-paradigm finding: on the ASR-processed WAUC EEG, the MI-derived FBCSP feature pipeline does **not** underperform the CL-specific band-power pipeline. The MI machinery generalises adequately to the CL paradigm in this configuration.

### Caveat 3 — Subject-level data quality on WAUC

Of the 45 nominally usable WAUC subjects (48 filesystem subjects minus S28 missing ratings, S23 and S26 missing the P4 channel), **2 additional subjects collapsed to a single label class after the NaN-filter** that drops ASR-uncovered windows: S39 (only session 6 surviving, all "high") and S48 (only sessions 4 and 6 surviving, both "low"). Both baselines error on these subjects (with the multiline `LinearDiscriminantAnalysis` warning surfaced through the error column of the CSV), reducing the per-baseline successful-evaluation count to 215 out of an expected 225 (43 of 45 subjects × 5 folds). The 4 errored rows are preserved in the CSV (fold = -1, seed = -1) for full traceability.

The post-filter single-class collapse is a known side-effect of the ASR cleaning + per-trial NaN drop: subjects whose recording quality was poor enough that ASR rejected most of one workload-condition's windows lose that class entirely. This is a defensible per-trial quality-control choice — see `design-doc/ccb-formulation.md` §2.7 "NaN-marked windows from ASR" — but the cohort-level consequence is worth noting in the Discussion chapter as part of the WAUC characterization.

## Reproduction

```bash
make stew-check && make wauc-check       # validate local data layouts
PYTHONPATH=src .venv/bin/python scripts/run_fixed_baselines_cl.py
```

Default arguments produce the table above in ~15 minutes on the
reference machine (Apple Silicon M-series, single thread). For a
faster smoke check use `--subjects-stew 1,2,3 --subjects-wauc 1,2,3 --n-folds 1`.

## What this resolves and what it does not

**Resolved:**

- Phase C deliverable per the model-development plan: both fixed-pipeline CL baselines are implemented, tested, and run end-to-end on both datasets.
- The Strategy B 2 × 2 design has its first two cells populated (Baselines B1 and B2). The remaining two cells (CCB-generic in Phase D, CCB-workload in Phase E) will fill out the table.
- The "FBCSP machinery generalises to CL" empirical question has a first-pass answer **for the WAUC paradigm**: yes, FBCSP+sLDA is statistically indistinguishable from a CL-domain-specific BandPower+sLDA at the within-CV protocol used here.

**Not resolved by Phase C:**

1. The STEW within-CV ceiling effect — a LOSO protocol must be added. Open follow-up.
2. The Albuquerque 2020 WAUC reference κ — must be looked up directly against the paper body. The `albuquerque2020wauc` BibTeX note carries this as a pending verification.
3. Whether the workload-context feature set (frontal-θ, parietal-α, frontal-α asymmetry, engagement index) helps the CCB *adaptively*. That contrast lives in Phase E (Workload-context wiring) once the CCB pipeline is run with both context choices.
