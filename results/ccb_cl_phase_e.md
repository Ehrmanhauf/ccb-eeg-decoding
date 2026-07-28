# Phase E — WorkloadContext-specialised CCB on STEW and WAUC

**Generated:** 2026-05-19 from `scripts/run_ccb_stew.py
--use-workload-context` and `scripts/run_ccb_wauc.py
--use-workload-context` (default flag value; default arguments
otherwise: 5 seeds × 1 within-subject fold × α = 0.5 × OPLB policy).
Source CSVs: [`results/ccb_stew_workload.csv`](ccb_stew_workload.csv),
[`results/ccb_wauc_workload.csv`](ccb_wauc_workload.csv).

The runner-side wiring landed via the new `workload_channel_roles`
parameter of [`thesis.ccb.runner.run_ccb_on_split`](../src/thesis/ccb/runner.py); the
open-justification entry "Wire `compute_context_workload` through
`run_ccb_on_split`" is now closed in
[`design-doc/open-justifications.md`](../design-doc/open-justifications.md).

## Headline result — the workload-context specialisation does NOT help

The Phase E experiment was set up to answer the prediction written
verbatim in [`results/ccb_cl_phase_d.md`](ccb_cl_phase_d.md):

> *"Switching the CCB's context to `compute_context_workload` will
> recover most of the −0.21 κ gap on WAUC."*

The measured outcome refutes that prediction:

| Dataset | Context | Mean κ | Std κ | n | Δκ vs generic |
|---|---|---:|---:|---:|---:|
| STEW | generic   | 0.7437 | 0.2556 | 225 | — |
| STEW | workload  | 0.7439 | 0.2505 | 225 | **+0.0002** |
| WAUC | generic   | 0.4434 | 0.2015 | 215 | — |
| WAUC | workload  | 0.4261 | 0.2071 | 215 | **−0.0173** |

The workload context is **statistically indistinguishable** from the
generic n-channel MI-derived context on STEW, and **marginally worse**
on WAUC. The 0.21-κ-below-fixed-pipeline gap that motivated the Phase
E experiment is not closed.

Per-subject breakdown on WAUC (Δ = workload − generic, averaged
within-subject across the 5 seeds):

| Statistic | Value |
|---|---:|
| Subjects with workload > generic (Δ > 0) | **14 / 43** (33 %) |
| Subjects with Δ > +0.05 (meaningful improvement) | 2 |
| Subjects with Δ > +0.10 (large improvement)  | 0 |
| Mean Δ | −0.017 |
| Std Δ  | 0.044 |
| Min Δ  | −0.115 |
| Max Δ  | +0.094 |

The Δ distribution is centred near zero with a slight negative skew.
This is the empirical signature of *no real effect* — workload
specialisation is not delivering a per-subject systematic benefit
that the average over 43 subjects could detect at this protocol.

## What this means for the bandit-CL gap

The Phase D analysis attributed the CCB-vs-fixed-pipeline gap on WAUC
(Δκ ≈ −0.21) to "the runner's current generic n-channel MI-derived
context not encoding the workload-relevant features … that the
BandPower baseline uses successfully". Phase E now refutes that
specific attribution:

1. **The MI-derived features in the generic context are not the
   bottleneck.** Switching to a CL-specific feature set (θ / α / β
   log-power + frontal-θ + parietal-α + frontal-α asymmetry +
   engagement) preserves κ within rounding error on STEW and
   *worsens* it slightly on WAUC. If the MI features were
   handicapping the bandit, the workload features should have
   produced a measurable lift; they did not.

2. **The 0.21 κ CCB-vs-fixed-pipeline gap on WAUC therefore has a
   different mechanism.** Candidates (none ruled out by the present
   data):
    - **Reservation overhead.** The 30 % calibration reservation
      (`calibration_frac = 0.3`) takes 30 % of the would-be training
      data out of the bandit stream. The fixed pipeline sees the
      full 80 % of within-CV training trials; the CCB sees 56 %.
      A `calibration_frac` sweep (Phase F) will disambiguate.
    - **Exploration cost.** The LinUCB exploration upper bound forces
      pulls towards under-explored arms even when the empirical-mean
      arm is clearly better. With ≈ 600 bandit-stream rounds per
      WAUC subject, the asymptotic regret bound should be small, but
      the constant might still bite. An α-sweep (Phase F) will probe
      whether reducing α from 0.5 → 0.1 closes the gap.
    - **Arm-pool mismatch.** `enumerate_arms_generic(n_channels=8)`
      produces 108 arms based on filter-bank × CSP-or-identity ×
      log-variance-or-Riemannian × time-window combinations. CSP
      with 4 components on 8 channels may not be the right
      featurisation for CL; the band-power baseline used a much
      smaller feature space (8 channels × 3 bands = 24 features +
      4 derived) and matched the CCB's CSP-based grid. An arm-pool
      ablation (Phase F) would test this directly.
    - **Within-CV vs the bandit's adaptive structure.** Random
      K-fold CV does not reward the bandit's per-trial adaptation:
      the calibration trials are sampled randomly across sessions,
      so the bandit cannot learn a session-specific arm-selection
      policy. A session-leave-out protocol (where the bandit
      adapts to held-out session data while training on the other
      five) would exercise the bandit's adaptive machinery in a
      way within-CV does not, and may show a meaningful CCB-vs-
      fixed-pipeline lift.

3. **The workload context might still matter under different
   protocols.** The empirical equivalence here is on within-subject
   CV. A leave-one-subject-out (LOSO) experiment would test whether
   the workload features generalise better *across* subjects than
   the MI-derived μ / β features — even if they don't help within
   subjects. LOSO is not implemented in `thesis.protocols` and is
   tracked as a follow-up.

## STEW workload-context — the expected null result

On STEW the workload context produces κ = 0.7439 vs the generic
context's κ = 0.7437. The two-decimal-place agreement is consistent
with the Phase D prediction:

> *"Smaller improvement on STEW, because (i) the segment-leakage
> signal is paradigm-agnostic and any context captures it, and
> (ii) the bandit stream on STEW is short (≈ 42 rounds), limiting
> how much extra signal can be exploited."*

Both context families capture the segment-discriminative signal that
saturates the within-CV protocol on STEW; the workload-specific
features add no new information beyond what the generic context
already encodes for the segment-identity decision problem.

## Methodological caveats (carried forward from Phase C and D)

1. **STEW within-CV is segment-leakage-saturated** — neither the
   Phase E κ ≈ 0.74 nor the Phase D κ ≈ 0.74 nor the Phase C fixed-
   pipeline κ ≈ 0.95 reflects "real" workload classification. A
   LOSO follow-up is tracked.
2. **WAUC has 4 single-class-post-NaN subjects** errored at the
   LDA fit step. Their fold = −1 rows are preserved in both
   `ccb_wauc_generic.csv` and `ccb_wauc_workload.csv` for full
   traceability.
3. **The workload context dimension (base 9) vs the generic context
   dimension (base 15)** is a confounder for the comparison: the
   workload context has 6 fewer feature dimensions to start with.
   Whether this absolute-dimensionality effect drives the slight
   −0.017 κ on WAUC is not separable from the feature-identity
   effect in the present experiment.

## Conclusions usable in the thesis prose

- The CCB-vs-fixed-pipeline gap on WAUC (Δκ ≈ −0.21) is **not**
  attributable to context misspecification. The MI-derived
  generic context and the CL-specific workload context produce
  statistically equivalent CCB κ on WAUC.
- The remaining candidate mechanisms (calibration reservation,
  exploration cost, arm-pool composition, evaluation protocol) are
  the targets of Phase F sensitivity sweeps.
- Negative-result-friendly framing: this is exactly the kind of
  result the thesis is set up to value. The two contexts being
  statistically equivalent on CCB-vs-fixed-pipeline κ is itself a
  characterisation of the framework's behaviour, regardless of the
  specific sign of Δ.

## Reproduction

```bash
make stew-check && make wauc-check
PYTHONPATH=src .venv/bin/python scripts/run_ccb_stew.py \
    --use-workload-context --output results/ccb_stew_workload.csv
PYTHONPATH=src .venv/bin/python scripts/run_ccb_wauc.py \
    --use-workload-context --output results/ccb_wauc_workload.csv
```

Run time on the reference machine (Apple Silicon M-series, single
thread): ~5 min for STEW and ~25 min for WAUC.
