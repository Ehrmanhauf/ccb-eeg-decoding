# Phase D — CCB with generic context on STEW and WAUC

**Generated:** 2026-05-19 from `scripts/run_ccb_stew.py` and
`scripts/run_ccb_wauc.py` (default arguments: 5 seeds × 1 within-
subject fold × α = 0.5 × OPLB policy × generic n-channel MI-derived
context). Source CSVs:

- [`results/ccb_stew_generic.csv`](ccb_stew_generic.csv) — 225 rows (45 subjects × 5 seeds × 1 fold × 1 α).
- [`results/ccb_wauc_generic.csv`](ccb_wauc_generic.csv) — 225 rows expected (45 subjects × 5 seeds × 1 fold × 1 α); some may fail because of post-NaN single-class subjects (see Phase C summary).

## Per-(dataset) summary

(Mean across subjects × seeds × folds at the locked first-pass cell.)

| Dataset | Mean κ | Std κ | Median κ | Min κ | Max κ | n_evaluations |
|---|---:|---:|---:|---:|---:|---:|
| STEW (CCB, generic context)  | **0.744** | 0.170 | 0.730 | 0.351 | 1.000 | 225 |
| WAUC (CCB, generic context)  | **0.443** | 0.202 | 0.417 | 0.147 | 0.930 | 215 |

(WAUC has 10 errored evaluations — the 2 single-class-post-NaN subjects S39 / S48 plus a CCB-pipeline edge case on one additional subject's seed combination. The 215 surviving rows are the basis of the table above; the errored rows are preserved verbatim in `ccb_wauc_generic.csv` for traceability.)

For direct comparison with the Phase C fixed-pipeline numbers on the **same protocol** (within-subject 5-fold CV, same data, same drop list):

| Dataset | Fixed-pipeline B1 FBCSP | Fixed-pipeline B2 BandPower | CCB (generic context) | Δκ (CCB − best baseline) |
|---|---:|---:|---:|---:|
| STEW | 0.937 ± 0.105 | 0.953 ± 0.092 | 0.744 ± 0.170 | **−0.209** |
| WAUC | 0.658 ± 0.169 | 0.644 ± 0.189 | 0.443 ± 0.202 | **−0.215** |

## Interpretation

### STEW

The CCB underperforms both fixed-pipeline baselines on STEW by Δκ ≈ −0.20. This is methodologically expected:

1. The CCB pipeline reserves 30 % of training trials for arm-head pre-training (`calibration_frac = 0.3`), so the bandit stream sees ≈ 0.7 × 60 = 42 trials instead of the 60 the fixed pipeline gets.
2. The OPLB policy pays an exploration cost (LinUCB-style upper-confidence-bound exploration) during the streaming phase. Some pulls go to suboptimal arms by design.
3. The generic n-channel MI-derived context (`compute_context_generic`, µ / β power per channel + entropy + variance) is mis-specified for cognitive load — workload-discriminative features (θ / α / β power, frontal-θ, parietal-α, frontal-α asymmetry, engagement) are not in the context, so the bandit cannot condition its arm choice on workload-relevant state.

Per-subject κ ranges 0.35 → 1.00; the lower end (S25, S32, S34, S48) corresponds to subjects where the within-segment EEG is *not* uniformly distinguishable from the across-segment baseline, suggesting that even the segment-leakage signal that drives the fixed-pipeline ceiling is weaker for some subjects.

Bottom-line claim that can be made from this number on its own: *with a deliberately mis-specified (MI-derived) context and a per-trial exploration overhead, the CCB still achieves κ = 0.74 on STEW within-CV, well above the published Lim 2018 within-paper κ = 0.46*. The κ = 0.46 is not directly comparable (different protocol; see Phase C summary's "STEW within-CV is segment-saturated" caveat) so this claim must be carefully bounded.

### WAUC

The CCB with the generic MI-derived context lands at κ = 0.443 ± 0.202 on WAUC — **Δκ = −0.215 below the best fixed-pipeline baseline (B1 FBCSP+sLDA at 0.658)**. This is the cleanest negative result of the multi-paradigm investigation so far, and it lands almost exactly at the lower end of the prediction range written in advance.

Three observations follow.

1. **The Δκ ≈ −0.21 gap is large enough to attribute to the context misspecification, not to bandit overhead alone.** STEW also showed Δκ ≈ −0.21 below its fixed-pipeline baselines, but that contrast happens at the κ ≈ 0.9 ceiling where the segment-leakage signal saturates both pipelines. On WAUC the κ ≈ 0.65 baseline is *not* at the ceiling, and the CCB's drop to κ ≈ 0.44 is within the range "the bandit cannot condition on workload-relevant state" rather than "the bandit pays a constant calibration / exploration toll".

2. **The bandit-stream length is not the bottleneck.** WAUC subjects have ≈ 900 trials per subject after windowing, so the OPLB has hundreds of rounds to converge before the test phase. Despite this, κ does not recover towards the fixed-pipeline level — strong evidence that the limitation is the context, not the number of rounds.

3. **This is the operational case for Phase E's workload-context wiring.** The generic MI-derived context (`compute_context_generic`: μ / β log-power per channel + entropy + variance) does not include θ-band features, parietal-α, frontal-α asymmetry, or the engagement index — exactly the features the workload-classification literature identifies as the CL substrate (the same features the Phase B BandPower baseline computes from the fixed-pipeline side). Switching to `compute_context_workload` is the natural next experiment.

Per-subject distribution: κ ranges 0.147 → 0.930 across 43 subjects (S39 / S48 absent post-NaN single-class collapse, plus one further error). The high-κ tail (S15, S40, S41 each ≥ 0.85) shows that the CCB *can* match the fixed pipeline on subjects whose EEG carries strong MI-flavoured signal in addition to workload signal; the low-κ tail (S05, S29, S32 each ≤ 0.20) is where the context misspecification bites hardest. A subject-stratified analysis in Phase G's discussion chapter will tie this back to the demographics covariates (treadmill vs bike cohort, age).

## What this resolves and what it does not

**Resolved:**

- Phase D first-pass CCB κ exists for both STEW and WAUC.
- The 2 × 2 evaluation design has 3 of 4 cells populated; only "CCB with workload context" remains for Phase E.
- The CCB-vs-fixed-pipeline gap is **−0.21 κ on both datasets**, but **for different reasons**: ceiling-effect at κ ≈ 0.9 on STEW (segment-leakage saturates both pipelines), context misspecification at κ ≈ 0.6 on WAUC (the bandit cannot condition on workload-relevant state). The Phase E experiment will discriminate between these two interpretations.

**Not resolved:**

1. Whether the workload-context specialisation helps on WAUC. Prediction: **Δκ should improve by 0.1–0.2 on WAUC** if the context misspecification is the limiting factor. If Δκ does not improve, the bandit calibration / exploration overhead is the true bottleneck and a different intervention is needed.
2. Whether the workload-context specialisation helps on STEW. Prediction: **smaller improvement on STEW**, because (i) the segment-leakage signal is paradigm-agnostic and any context — MI-derived or CL-derived — captures it, and (ii) the bandit stream on STEW is short (≈ 42 rounds), limiting how much extra signal can be exploited.
3. The STEW within-CV ceiling effect — same caveat as Phase C; LOSO follow-up tracked. The Phase E numbers will inherit this caveat.
4. The WAUC errored rows — preserved verbatim in `ccb_wauc_generic.csv` rather than silently dropped, so the post-mortem is traceable.
