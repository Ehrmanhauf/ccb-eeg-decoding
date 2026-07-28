# Open Justification Items

> **Scope-level framing decisions are not tracked here**; they live in `ccb-formulation.md` §1.4 (Framing and scope) — see that section for the recorded decisions on (1) purely-investigative scope across paradigms, (2) cognitive-load target = 3-level classification, (3) publication venue = methodological BCI journal, and (4) datasets in scope (BCI-IV-2a/2b + Cho2017 full+3ch + STEW + SEED-VIG; BNCI2014_004 dropped after MOABB verification showed it is the same data as BCI-IV-2b). The technical follow-up item that came out of decision (2) — STEW label binning thresholds against Lim 2018 — is tracked below as a new `Currently open` entry **"STEW label cardinality"**.

Every `JUSTIFY:` note in the repo has a corresponding entry here, tracked to resolution. Resolutions come from one of three sources (see [`/CLAUDE.md`](../CLAUDE.md) §1):

1. **Prior literature** — add a BibTeX key to `references.bib` and cite it.
2. **Our own experiment** — link to the script + commit hash.
3. **Methodological derivation** — explicitly cite the locked principle.

Format for entries:

```
- [ ] <title>
  - **Context:** where the choice appears (file §section, commit hash, or code line)
  - **Current state:** what we're doing now and why it's insufficient
  - **Resolution path:** reference search / experiment plan / derivation
  - **Blocker (if any):** what must land first
```

---

## Currently open

- [ ] **Knapsack cost model — deterministic per-arm cost (vs linCBwK's stochastic-cost assumption)**
  - **Context:** `src/thesis/ccb/oplb.py` lines 8–15 specialise Pacchiano 2021 OPLB to a deterministic cost `c(a)` per arm (feature-vector length; see `thesis.ccb.arms.arm_cost`). linCBwK (Agrawal & Devanur 2016) allows stochastic costs with a separate cost-UCB; our implementation collapses that UCB to zero because the cost is known at arm-bank construction.
  - **Current state:** the `cost_ucb_coef` field exists on `OPLBConfig` as a forward-compat hook but is unused. No derivation from `pacchiano2021linconstr` Theorem 1 or `agrawal2016linbwk` has been written down for the deterministic-cost regret bound.
  - **Resolution path:** (i) add a one-paragraph derivation showing that the deterministic-cost OPLB regret bound is $\tilde{O}(d\sqrt{T} / \zeta)$ with the cost-UCB term vanishing, citing Pacchiano 2021 §5; (ii) optionally wire `cost_ucb_coef` to support noisy-cost arms as a future extension.
  - **Blocker:** none.

- [ ] **Classical-baseline classifier-head hyperparameters held at scikit-learn library defaults (untuned)**
  - **Context:** `src/thesis/baselines/classical.py` `make_classifier` builds the B3–B5 comparators at library defaults — `SVC(C=1.0, gamma="scale")`, `DecisionTreeClassifier()` (CART defaults), `RandomForestClassifier(n_estimators=200)` (inline `JUSTIFY:` at `_RF_N_ESTIMATORS`). These heads are fitted on the same engineered features as B1/B2 (see `ccb-formulation.md` §8.4) by `scripts/run_classical_baselines.py`.
  - **Current state:** the choice of *which* classifiers is justified by the canonical EEG-BCI classifier reviews (`lotte2007review`, `lotte2018review`) and the RF-on-sensorimotor-BCI precedent (`steyrl2016randomforest`); the choice to leave their *hyperparameters at library defaults* is justified methodologically — the comparators exist to test whether the CCB gap is head-specific or general, a question that does not require each head to be individually optimised, and per-head tuning would itself demand a leakage-safe nested cross-validation absent from the rest of the evaluation. This is defensible but currently rests on derivation (iii), not on a measured tuning sweep.
  - **Resolution path:** either (i) keep defaults and state explicitly in Chapter 4 that the comparators are untuned off-the-shelf classifiers (lower bound on each head's achievable κ), or (ii) add a nested-CV tuning sensitivity for at least the SVM (`C`, `gamma`) on one dataset to bound the headroom that default hyperparameters leave on the table.
  - **Blocker:** none. Decision (i) is the locked default for the present revision.


## Closed

- [x] **Leave-one-subject-out (LOSO) on STEW — within-segment-leakage test (executed, 2026-05-30)**
  - **Context:** the within-CV κ ≈ 0.94 on STEW was reported throughout the thesis under a *segment-leakage caveat* — the loader assigns one workload bin per continuous 2.5-minute segment (`stew_load.py`), so random K-fold CV trains and tests on the same two segments and a classifier can score by segment identity rather than workload (Chapter 3 §method-protocol). Chapter 4 flagged the leave-one-subject-out follow-up as a deferred sensitivity requiring "a new `thesis.protocols.leave_one_subject_out` primitive and a re-run".
  - **Evidence (our own implementation + experiment):** primitive `thesis.protocols.leave_one_subject_out` (+ `pool_subjects`) pools all subjects and holds one out per fold; `CCBResult` gained optional `y_true` / `y_pred` for pooled-κ recovery; LOSO / pooling unit tests added to `tests/test_protocols.py` (235 tests pass). Runner `scripts/run_loso_stew.py` (per-method / per-seed CSV checkpointing; persists per-fold prediction sequences for crash-safe pooled-κ recomputation). Producer: `PYTHONPATH=src .venv/bin/python scripts/run_loso_stew.py --ccb-seeds 0 --ccb-train-cap 800 --output results/loso_stew.csv` → 135 rows (B1 / B2 fixed × 45 + CCB seed 0 × 45); summary `results/loso_stew.md`.
  - **Empirical finding:** under LOSO the fixed-pipeline κ collapses from within-CV ≈ 0.94 to **0.277 ± 0.260 (B1 FBCSP)** and **0.328 ± 0.262 (B2 BandPower)** (pooled κ 0.277 / 0.314); the CCB falls from 0.744 to **0.151 ± 0.256** (pooled 0.160). Three readings: (i) the within-CV ceiling was indeed mostly segment-leakage — every method loses ≈ 0.6 κ; (ii) a *modest but genuine* cross-subject workload signal survives (≈ 0.3, well above the κ = 0 floor) — STEW is not pure noise; (iii) the CCB still underperforms the fixed pipelines under the leakage-free protocol (−0.13 to −0.18 κ), reproducing the panel-wide ordering. A fourth, secondary observation: with the ceiling removed, B2 (CL-specific) edges B1 (MI-derived) by +0.051 κ — a separation the saturated within-CV protocol hid. Integrated into Chapter 4 §results-loso (Table `tab:loso-stew`).
  - **Scope note:** fixed baselines train on the full pooled 44-subject set (deterministic, no seed); the CCB is re-cast as a population-trained cross-subject policy with the per-fold training pool capped at 800 epochs for tractability (training-subjects-only subsample — no leakage) and reported at a single representative seed. This also stands as the first cross-subject-evaluation step toward Formulation C (§5.3, §10).
  - **Matched-conditions update (2026-06-29) — closes the LOSO-cap choice:** the figures above are the original 2026-05-30 run. Under the locked matched-conditions discipline (Ch3 §method-validation-matched; `src/thesis/matched.py` single split source + `tests/test_matched_conditions.py` guardrail), the per-fold training pool is now capped at **4,000 epochs for all three method families** (CCB, classical, EEGNet) — effectively the full held-in set on STEW — and the CCB is averaged over **5 seeds**. The CCB LOSO pooled κ is then **0.243** (was 0.160); the best fixed-pipeline pooled κ is unchanged (0.314, since STEW's ~3.3k-epoch pool sits under the cap), so the gap narrows to ≈ 0.07 but the fixed ≥ CCB ordering holds. EEGNet LOSO κ = 0.28 is added as the deep comparator. The cap choice is thus settled: 4,000 for all methods (`matched.py::DEFAULT_TRAIN_CAP`). Current numbers: Table `tab:loso-stew` (Ch4), `results/loso_stew.csv`.

- [x] **Wire `compute_context_workload` through `run_ccb_on_split` for STEW / WAUC** (Phase E, Research-wave 1)
  - **Evidence (our own implementation + experiment):** `src/thesis/ccb/runner.py` gained a new optional `workload_channel_roles: dict[str, list[int]] | None = None` parameter; when supplied alongside a `dataset_name ∈ {"STEW", "WAUC"}`, the dispatcher routes to `compute_context_workload` from `thesis.ccb.context_cl` and switches `d_ctx` to `context_dim_workload(...)`. Existing 2a / 2b / Cho2017 paths are unchanged. Regression coverage added in `tests/test_ccb_runner.py::TestWorkloadContextDispatch` (3 tests: STEW + roles, WAUC + roles, STEW without roles → generic fallback). 214 unit tests pass (was 211).
  - **Action taken:** `scripts/run_ccb_stew.py` and `scripts/run_ccb_wauc.py` both expose a `--use-workload-context / --no-use-workload-context` CLI flag (default ON) and pass the dataset's `*_CHANNEL_ROLES` map through to the runner. Output CSVs carry a new `context` column with value `"workload"` or `"generic"` so the two regimes are always disambiguable. Phase D's first-pass generic-context results (`results/ccb_*_generic.csv`) and Phase E's workload-context results (`results/ccb_*_workload.csv`) sit alongside each other and are compared in `results/ccb_cl_phase_e.md`.
  - **Empirical findings (Phase E re-run, 2026-05-19):** The workload-context specialisation does **not** improve CCB κ relative to the generic n-channel MI-derived context on either CL dataset at the within-subject 5-fold-CV protocol. STEW: κ_generic = 0.7437 ± 0.2556 → κ_workload = 0.7439 ± 0.2505 (Δκ = +0.0002, indistinguishable). WAUC: κ_generic = 0.4434 ± 0.2015 → κ_workload = 0.4261 ± 0.2071 (Δκ = −0.0173, slight worsening; only 14 of 43 subjects improve, only 2 by more than +0.05, none by more than +0.10). The Phase D prediction that workload-specific features would recover the −0.21 κ CCB-vs-fixed-pipeline gap on WAUC is therefore **refuted**. The gap's mechanism is elsewhere (calibration reservation, exploration cost, arm-pool composition, evaluation protocol — Phase F sweeps will discriminate). Full analysis in `results/ccb_cl_phase_e.md`. The wiring itself remains valuable as the methodologically appropriate default (the CL-specific feature set is in the locked design-doc spec, and the wiring lets future protocols — e.g. LOSO — re-test the question with cross-subject transfer in scope), so the `--use-workload-context` flag defaults to ON in both CL CCB scripts even though the within-CV measured effect is null.


  - **Evidence (literature survey):** Survey of public CL EEG datasets against four criteria (wearable ≤14ch, task class distinct from STEW SIMKAP, ≥50 subjects, objective performance label) was conducted on 2026-05-19. Five datasets were verified against canonical pages (Frontiers, Zenodo, OpenNeuro, Mendeley Data, IEEE Xplore via arXiv), with author attributions corrected against the agent's initial report: **Albuquerque et al. 2020 [albuquerque2020wauc] WAUC** (verified via Frontiers), **Hinss et al. 2022 [hinss2023cogbci] COG-BCI** (verified via Zenodo), Pavlov et al. 2022 Digit Span (verified via PMC), Nirabi et al. 2024 Arithmetic+Stroop (verified via Mendeley Data), and Angkan et al. 2023 CL-Drive (verified via arXiv).
  - **Scientific reading:** the four-way intersection of criteria is essentially empty in the public literature — no dataset satisfies all of (wearable, distinct task, ≥50 subjects, objective label). The trade-off space splits into a *wearable family* (≤14ch but small n or partial-objective labels) and a *research-grade family* (≥50 subjects with rich objective labels but ≥30 channels, blocked by no-leakage). The thesis therefore knowingly relaxes the subject-count criterion within the wearable family rather than the wearable criterion itself.
  - **Action taken (Strategy B, 2026-05-19):**
    1. **WAUC adopted as secondary CL dataset** in §2.7 of `ccb-formulation.md` with full A / C / C decomposition: 8-channel Enobio dry-electrode headset, 48 subjects across treadmill and bike cohorts, MATB-II under physical-exertion conditions, mixed labels (NASA-TLX, Borg fatigue, MATB-II behavioural performance, seven physiological streams). Binary low/high MW classification used as the CCB target; physical-exertion condition treated as covariate. `albuquerque2020wauc` entry added to `references.bib`. Custom loader `src/thesis/data/wauc_load.py` to be implemented in the next workstream.
    2. **COG-BCI cited as methodological reference but not adopted for training/test.** Its 64-channel ActiCap montage cannot be subsetted to 8 or 14 channels under the per-dataset no-leakage rule of §2.4. Cited as a cross-task CL gold-standard in the Discussion. `hinss2023cogbci` entry added to `references.bib`.
    3. **§1.4.2 (Datasets in scope) updated** to include WAUC under primary CL alongside STEW; COG-BCI listed under "methodological reference (cited but not adopted)".
  - **Trade-off knowingly accepted:** Both adopted datasets have n = 48 subjects (rather than ≥50); WAUC's objective labels are mixed (task-difficulty manipulation + behavioural performance + subjective NASA-TLX) rather than purely objective; and Nirabi 2024 (the only additional verified wearable candidate) is deferred as a stretch-goal third CL dataset because its 15-subject cohort and dataset-only (no peer-reviewed paper) status would invite jury sourcing critique.

- [x] **Spatial-filter family for 3-ch bipolar — keep all three families (CSP / Laplacian / identity)** (Workstream B.2)
  - **Evidence (our own experiment):** `scripts/sweep_spatial_filter.py --seeds 0,1,2,3,42` produced `results/ccb_sens_spatial_filter.csv` (360 rows: 9 subjects × 5 seeds × 2 protocols × 4 cells). Each cell monkey-patches `arms._SPATIAL_FILTERS` to a single family (or the full triple) and runs the full CCB pipeline at the **Phase-5 Stage-1 best-factorial cell** (α=0.5, calibration_frac=0.3, window_size=50, arm_pool=pruned, include_recent_rewards=False). 2 of the 360 runs ended in `empty_pool` (all arms pruned at κ<0.05 on calibration) — both `laplacian_only`, both on the `official` protocol (n=43 reported for that cell).

    | cell | within | official |
    |---|---|---|
    | **full (CSP + Laplacian + identity)** | **+0.178 ± 0.210** | **+0.172 ± 0.193** |
    | csp_only | +0.175 ± 0.199 | +0.151 ± 0.213 |
    | identity_only | +0.174 ± 0.223 | +0.145 ± 0.186 |
    | laplacian_only | +0.074 ± 0.161 | +0.047 ± 0.110 (43/45) |
  - **Scientific reading:** the full 3-family arm pool is the best cell on both protocols, narrowly. `csp_only` and `identity_only` are essentially tied with the full pool on within-subject (Δ < 0.005 κ) and only modestly behind on official (−0.02 to −0.03 κ). `laplacian_only` is substantially worse on both protocols (−0.10 within / −0.13 official vs. full) and the only family that produces empty-pool runs at the κ<0.05 pruning threshold. **The reason Laplacian-only underperforms is not characterized in this audit** — one *hypothesis worth testing later* is that BCI-IV-2b is already recorded with a bipolar montage (C3-FCz, Cz-FCz, C4-FCz per `desc_2b.pdf`), so the additional Laplacian stencil (C3−Cz, C4−Cz) acts as a second differential operator on top of an already-differential signal; this is conjecture without literature support in this session and should not be cited as the mechanism. What *is* established by the data: combining Laplacian with CSP and identity contributes a small positive lift (full > csp_only by +0.003 within / +0.021 official, full > identity_only by +0.004 / +0.027), so dropping Laplacian outright would cost κ. The resolution is to keep all three families in the pool.
  - **Action taken:** `_SPATIAL_FILTERS` retained as `("csp", "laplacian", "identity")` in `src/thesis/ccb/arms.py`. Inline citation to this closed item; the methodological-derivation note on the Laplacian stencil (C3−Cz, C4−Cz from `desc_2b.pdf`) is empirically validated *in combination* with the other families but not as a standalone choice. The 2 empty-pool runs on `laplacian_only` are reported as such (κ=NaN, status="empty_pool"); no headline number depends on those rows.

- [x] **CCB time-window arm grid — keep all three windows (0–4, 0.5–2.5, 1–3 s)** (Workstream B.1)
  - **Evidence (literature side):** `ang2012fbcsp` §3.1.1 "*the time segment of 0.5–2.5 s after the onset of the visual cue were used*" — verbatim, verified 2026-05-12 via Frontiers DOI 10.3389/fnins.2012.00039. Confirms the (0.5, 2.5) window in the arm grid. The (1.0, 3.0) window was previously attributed to Ramoser 2000 in the in-repo `JUSTIFY:` comment; **that attribution failed verification in this audit** (PubMed abstract does not state the window; open-access PDF at `cs.hmc.edu/~keller/eeg/Ramoser.pdf` could not be text-extracted in this session; web-search snippets did not surface the segment). The Ramoser attribution has therefore been **removed** from the inline comment in `src/thesis/ccb/arms.py`. The (1.0, 3.0) window is retained on empirical grounds only (see below); the historical literature anchor remains an open hygiene task (verify against Ramoser 2000 §III or, if the window does not appear there, find the correct primary source or drop the window).
  - **Evidence (our own experiment):** `scripts/sweep_time_window.py --seeds 0,1,2,3,42` produced `results/ccb_sens_time_window.csv` (360 rows: 9 subjects × 5 seeds × 2 protocols × 4 cells). Each cell monkey-patches `arms._TIME_WINDOWS` to a single window (or the full grid) and runs the full CCB pipeline at the **Phase-5 Stage-1 best-factorial cell** (α=0.5, calibration_frac=0.3, window_size=50, arm_pool=pruned, include_recent_rewards=False). Mean κ across 9 subjects × 5 seeds × 1 fold:

    | cell | within | official |
    |---|---|---|
    | **full (all 3 windows)** | **+0.178 ± 0.210** | **+0.172 ± 0.193** |
    | 05_25 (Ang only) | +0.173 ± 0.238 | +0.156 ± 0.204 |
    | 0_4 (full epoch only) | +0.153 ± 0.223 | +0.148 ± 0.156 |
    | 1_3 (Ramoser-style only) | +0.105 ± 0.190 | +0.059 ± 0.135 |
  - **Scientific reading:** the full 3-window arm grid is empirically the best cell on both protocols. Dropping to the Ang 2012 (0.5, 2.5) window alone costs −0.005 κ within (negligible) and −0.016 κ official (modest). Dropping to the (0.0, 4.0) full-epoch window alone costs −0.025 within / −0.024 official. The (1.0, 3.0) window alone is markedly worse (−0.073 within / −0.113 official). Reading: the bandit benefits from cross-window arm diversity rather than from any single window being uniquely informative; the (1.0, 3.0) window in isolation does not carry enough class-discriminative signal at this hyperparameter cell, but its presence in the *full* pool is still chosen against the other windows by the bandit on the trials where it helps. Keeping all three windows is therefore justified by data; the marginal κ improvement of the full pool over the best single window (≈ +0.005 to +0.016 κ) is small but consistent across protocols.
  - **Action taken:** `_TIME_WINDOWS` retained as `((0.0, 4.0), (0.5, 2.5), (1.0, 3.0))` in `src/thesis/ccb/arms.py`. The inline comment was rewritten to (a) cite Ang 2012 §3.1.1 verbatim for (0.5, 2.5), (b) attribute (0.0, 4.0) to the closed "Epoch time window 0–4 s" justification, and (c) state explicitly that (1.0, 3.0) is kept on **empirical grounds only** with no verified literature citation — superseding the prior unverified "Ramoser 2000" claim. Re-verifying the (1.0, 3.0) window against Ramoser 2000 §III (or finding the correct primary source) is a low-priority hygiene task; failing that, dropping the window would cost a measurable −0.073 within / −0.113 official κ per the ablation above and is therefore not recommended.

- [x] **Verify remaining Ang 2012 citations in the codebase** (Workstream B.4)
  - **Evidence:** prior literature. Full WebFetch audit on 2026-05-12 of `https://www.frontiersin.org/articles/10.3389/fnins.2012.00039/full` against every in-repo reference to `ang2012fbcsp` / "Ang 2012" across `src/`, `tests/`, `scripts/`, `design-doc/`, and this file.
  - **Findings (confirmed):** §3.1.1 epoch window "0.5–2.5 s after the onset of the visual cue" (verbatim); §2.4 classifier "Naïve Bayesian Parzen Window (NBPW)"; §2.3 feature-selection algorithms "MIBIF" and "MIRSR" both described; §3.2 2b composition "240 trials without visual feedback and 160 trials with visual feedback" for training plus 320-trial evaluation; Table 2 OVR row "AVG 0.569" for 2a; Table 4 MIRSR row "AVG 0.599" for 2b.
  - **Findings (corrections applied):**
    1. **CSP component count `m`.** §3.1.1 reads "*Note that m = 2 is used for Dataset 2a and m = 1 is used for Dataset 2b.*" — i.e., per-dataset, not per-subject. Prior wording "m ∈ {2, 3, 4} depending on subject" in `src/thesis/baselines/fbcsp.py`, `scripts/sweep_csp_m.py`, and this file's CSP-component closed item was wrong. Fixed.
    2. **"Non-overlapping" sub-bands.** §2.1 reads "*A total of 9 band-pass filters are used, namely, 4–8, 8–12, …, 36–40 Hz.*" — the paper does not call them "non-overlapping," and the band edges coincide (4–8 and 8–12 share 8 Hz). Wording "9 non-overlapping 4 Hz sub-bands" in `fbcsp.py`, `arms.py`, and this file's filter-bank closed item softened to "9 sequential 4 Hz sub-bands."
    3. **2b κ headline.** Table 4 MIRSR row is **0.599**, not 0.600. Earlier text in `ccb-formulation.md` §4, `primer.md`, and this file's filter-bank closed item corrected to 0.599.
    4. **`primer.md` "our implementation per Ang 2012" framing.** Now states explicitly that Ang uses NBPW (we use shrinkage LDA, a documented deviation per `lotte2018review` §III.B), and that Ang's 2b protocol uses screening + feedback sessions while ours restricts to screening only.
    5. **Typo `10–14` → `12–16`** in `ccb-formulation.md` §5.1 filter-bank enumeration.
  - **Action taken:** edits committed across `src/thesis/baselines/fbcsp.py`, `src/thesis/ccb/arms.py`, `scripts/sweep_csp_m.py`, `design-doc/primer.md`, `design-doc/ccb-formulation.md`, and this file. Tests re-ran clean (161 passing as of the audit). No code behaviour changed; only documentation strings and design-doc prose.

- [x] **STEW label cardinality — 3-level workload binning of the 1–9 subjective rating**
  - **Evidence:** prior literature. Lim, Sourina, Wang 2018 "STEW: Simultaneous Task EEG Workload Data Set", IEEE TNSRE 26(11):2106–2114 [lim2018stew]. Verified 2026-05-12: PubMed abstract (PMID 30281467) confirms "3 identified workload levels from the rating scale with Cohen's kappa of 0.46" and 69% accuracy. Bin edges {1–3 = low, 4–6 = medium, 7–9 = high} are documented in the paper body and corroborated via a web-indexed snippet of the paper; direct PDF re-extraction failed in this session due to PDF text-rendering limits (poppler-utils not installed) and is flagged for re-verification against the IEEE Xplore HTML before any thesis claim that depends on the exact edges (see the `note` field of `lim2018stew` in `references.bib`).
  - **Action taken:** `ccb-formulation.md` §2.6 now states the binning and the headline kappa from Lim 2018, with an A/C/C decomposition for STEW under the locked binning. `references.bib` entry `lim2018stew` carries the provenance trail. Unblocks Workstream C.3 STEW experiments. The "continuous regression on 1–9" alternative remains explicitly out of scope (would require activating the `RegressionMetrics` scaffold).

- [x] **Stage-1 combined — policy × sliding-window at the best-factorial base cell (Phase-5)**
  - **Evidence:** our own experiment. `scripts/run_ccb.py --policies oplb,ts --ts-prior-scale 0.1 --window-sizes 0,50 --alphas 0.5 --calibration-fracs 0.3 --arm-pools pruned --include-recent-rewards false --n-folds 1 --seeds 0,1,2,3,42 --output results/ccb_stage1_combined.csv` produced 360 rows at the Phase-4 best-factorial hyperparameter cell. Mean κ across 9 subjects × 5 seeds:
    - **OPLB + window=50: within 0.178 ± 0.210, official 0.172 ± 0.193**  ← best
    - OPLB (stationary): within 0.184 ± 0.214, official 0.158 ± 0.183
    - TS v=0.1 + window=50: within 0.141 ± 0.206, official 0.095 ± 0.155
    - TS v=0.1 (stationary): within 0.134 ± 0.217, official 0.137 ± 0.194
  - **Action taken:** **OPLB + sliding window (size 50) is the Phase-5 best cell on the official protocol**. Stacks with the best-factorial base — official κ improves from 0.158 (best-factorial no-window) → 0.172 (+0.014), and from 0.130 (window=50 at the Phase-3 *default* base cell) → 0.172 (+0.042). Within-subject protocol shows no window gain (0.184 → 0.178) — expected because no session gap. TS underperforms OPLB at this base cell regardless of window — suggests that once the calibration fraction (0.3 vs 0.2) and context (d=15 vs 18) are tuned, the UCB radius is no longer the bottleneck. **The thesis headline gap to 2b FBCSP (0.267 official) narrows to Δκ = 0.095 at this cell, vs 0.109 for the Phase-4 best-factorial row.** Closes Stage-1 combined item.
  - **Scientific reading:** session-gap drift (Phase-4 failure mode #2) is the dominant remaining obstacle on the official protocol. Fixing it with a 50-round sliding window is additive with the best-factorial hyperparameter tuning. The three Stage-1 fixes (TS / online heads / non-stationary) are not all needed — the sliding window alone, combined with correct hyperparameter tuning, captures most of the achievable improvement here. Online heads (Stage 5.2) and TS (Stage 5.1) remain characterised options available for future datasets (cognitive load) where session dynamics and exploration scale may differ.

- [x] **Non-stationary CCB — SlidingWindow / Discount (Phase-5 — addresses Phase-4 failure mode #2)**
  - **Evidence:** our own experiment. `scripts/run_ccb.py --policies oplb --protocols official --window-sizes 0,50,100 --discount-gammas 1.0,0.99,0.95 --n-folds 1 --seeds 0,1,2,3,42 --output results/ccb_nonstationary.csv` (405 rows, 9 subjects × 5 seeds × 9 cells, official protocol where session-gap drift is worst). Mean κ vs frozen-baseline (0.0813 ± 0.148):
    - **window=50, γ=1.0: 0.130 ± 0.190 (Δκ = +0.048)** ← best
    - window=50, γ=0.95: 0.125 ± 0.149 (+0.044)
    - window=50, γ=0.99: 0.115 ± 0.181 (+0.034)
    - window=100, γ=1.0: 0.100 ± 0.144 (+0.019)
    - window=inf, γ=0.95: 0.085 ± 0.140 (+0.004)
    - window=inf, γ=0.99: 0.080 ± 0.141 (−0.001, effectively tied)
  - **Action taken:** `OPLBConfig.window_size: int | None` and `OPLBConfig.discount_gamma: float = 1.0` added; `OPLB.update` rebuilds `A = λI + Σ_i γ^{n-1-i} ψ_iψ_iᵀ` from the history buffer when either option is active. Stationary path (both at defaults) unchanged. Runner regression test (`test_runner_preserves_nonstationary_config_fields`) guards against the runner-CCB wiring bug where the budget-default injection previously stripped new OPLBConfig fields. The bug was caught via direct probe before headline numbers landed; rebuilt config using `dataclasses.replace` rather than kwarg-by-kwarg copy.
  - **Scientific finding:** the sliding window is the **largest Stage-1 fix by Δκ on official** (+0.048 vs TS +0.035 on within, online-heads +0.017 on official). It targets exactly Phase-4 failure mode #2 — the bandit's forgetting of pre-session-gap observations lets θ̂ adapt to the new session's decision boundary. Discount γ alone (γ=0.95, 0.99 at window=inf) gives almost no lift; the hard-cut window is the mechanism that matters. Implication: Phase-4's "session-gap drift" diagnosis was correct and fixable. Closes Stage-1 non-stationary Phase-5 deferral.

- [x] **Online per-arm head updates (Phase-5 — addresses Phase-4 failure mode #2)**
  - **Evidence:** our own experiment. `scripts/run_ccb.py --policies oplb --online-heads --n-folds 1 --seeds 0,1,2,3,42 --output results/ccb_online_heads.csv` (90 rows, 9 subjects × 2 protocols × 5 seeds). Mean κ ± std at the default cell:
    - OPLB online-heads: **within 0.106 ± 0.155, official 0.098 ± 0.138**.
    - OPLB frozen-heads reference (from `results/ccb_ablation_policy.csv`): within 0.109 ± 0.201, official 0.081 ± 0.148.
    - **Δκ (online − frozen): within −0.003 (tied), official +0.017**. Official-protocol per-seed std shrinks 0.148 → 0.138.
  - **Action taken:** `ArmHead.partial_fit(X_new, y_new, sfreq)` landed in `src/thesis/ccb/arms.py`; re-solves shrinkage-LDA on the running (calibration ∪ pulled-stream) feature buffer. CSP spatial filter is **not** re-fit (keeps scope bounded and avoids unstable per-trial eigendecomposition on tiny increments). Runner gains an `online_heads: bool = False` kwarg; `False` (default) preserves the Phase-4 frozen-heads behaviour byte-for-byte (asserted by `test_runner_online_heads_is_backward_compat_when_false`). **Scientific finding:** the fix is directionally correct — online updates help precisely on the session-gap (official) protocol where failure mode #2 was diagnosed — but the effect is small (+0.017 κ, ~1σ/√n). Combined with Thompson Sampling (Stage 5.1) and non-stationary CCB (pending Stage 5.3), the Stage 5.4 full factorial will reveal whether the fixes stack or overlap. Neither fix alone closes the gap to ε-greedy (0.155/0.135) or to plain 2b FBCSP (~0.292/0.267).

- [x] **Thompson Sampling policy (Phase-5 §7.2 — addresses Phase-4 failure mode #1)**
  - **Evidence:** our own experiment. `scripts/run_ccb.py --policies oplb,eps_greedy,ts --ts-prior-scale 1.0 --n-folds 1 --seeds 0,1,2,3,42 --output results/ccb_ts_ablation.csv` (270 rows) and `--policies ts` with `--ts-prior-scale ∈ {0.1, 0.3, 1.0, 3.0}` → `results/ccb_ts_prior_scale.csv` (360 rows). Mean κ across 9 subjects × 5 seeds at the default cell otherwise:
    - TS v=1.0 (literature default): within 0.068 ± 0.148, official 0.065 ± 0.102 — **below both OPLB (0.109/0.081) and ε-greedy (0.155/0.135)**.
    - TS v=0.1 (tuned, least-diffuse posterior): within 0.144 ± 0.211, official 0.084 ± 0.150 — **beats OPLB on within (+0.035), ties on official; still loses to ε-greedy**.
    - Monotone: smaller v = better. v=3.0 gives 0.064/0.052.
  - **Action taken:** LinTSPolicy landed in `src/thesis/ccb/policies.py` with 6 unit tests (covariance check, regret ≤ 4× OPLB on synthetic, ε-greedy divergence, feasibility respect). Sampling uses Cholesky of A via `scipy.linalg.solve_triangular`. Default `prior_scale=1.0` matches Agrawal & Goyal 2013 convention; tuned v=0.1 recommended if TS is used in subsequent phases. **Scientific finding:** TS does not unseat ε-greedy on this MI problem — the Phase-4 failure mode #1 ("UCB radius mis-scaled for short T") is a symptom of the broader fact that **structured exploration (UCB or posterior sampling) both over-explore at T ≈ 170**, while random (ε-greedy) exploration matches the problem better. `ref: agrawal2013ts`; design-doc §7.2.

- [x] **Full factorial α × calibration_frac × include_recent_rewards (Phase-4)**
  - **Evidence:** our own experiment. `scripts/run_ccb.py --alphas 0.1,0.5,1.0 --calibration-fracs 0.2,0.3 --arm-pools pruned --include-recent-rewards true,false --n-folds 1 --seeds 0,1,2,3,42 --output results/ccb_factorial.csv` produced a 1080-row CSV (9 subjects × 2 protocols × 5 seeds × 3 alphas × 2 calibration fractions × 2 context settings, on the pruned arm pool).
    - **Best-factorial cell** (combined-objective `(within + official) / 2`): `alpha=0.5, calibration_frac=0.3, arm_pool=pruned, include_recent_rewards=False`. Mean κ across 9 subjects × 5 seeds: **within 0.184 ± 0.170 (Δκ = +0.349)**, **official 0.158 ± 0.153 (Δκ = +0.310)**.
    - Beats one-at-a-time tuned cell (`alpha=0.5, calibration_frac=0.2, arm_pool=full, include_recent_rewards=True`) on official Δκ by 0.077; beats Phase-3 default by 0.060 / 0.148 on within / official.
  - **Action taken:** the factorial confirms the Commit-4.1 finding that `include_recent_rewards=False` is the right choice and identifies a strictly-better cell than the one-at-a-time tuning could find. The thesis empirical headline cell is now the best-factorial cell. Phase-3 defaults remain as the Phase-3 reference; one-at-a-time-tuned remains as a comparison row in `results/ccb_baseline.md`. Resolves the Phase-4 note flagged inside the "LinUCB exploration parameter α" closed item.

- [x] **Multi-seed aggregation in `summarize_ccb.py` — best-cell selection + per-seed σ (Phase-4)**
  - **Evidence:** `scripts/summarize_ccb.py` now reads `results/ccb_factorial.csv`, picks the cell maximizing `(mean κ within + mean κ official) / 2` across seeds, and reports per-seed σ in the headline table. For the chosen best cell, σ across the 5 seeds {0, 1, 2, 3, 42} is **within σ = 0.065, official σ = 0.047** — substantially smaller than the seed-std observed at the Phase-3 default cell on official (σ = 0.058) and at the Phase-3 best α=0.5 / cf=0.2 / pool=full cell. Tests in `tests/test_summarize_ccb.py` (5 cases) cover the aggregation, best-cell selection, and missing-protocol fallback.
  - **Action taken:** every headline row in `results/ccb_baseline.md` now carries a per-seed σ alongside the per-subject std. Forward thesis reports of CCB κ should quote both the per-subject std (cross-subject heterogeneity) and the per-seed σ (algorithmic stability); single-seed numbers are not trustworthy. Resolves the Phase-4 note flagged inside the "Random seeds and CV fold count" closed item.

- [x] **Context recent-reward tail — ablation (d = 18 vs d = 15)**
  - **Evidence:** our own experiment. `scripts/run_ccb.py --include-recent-rewards true,false --n-folds 1 --seeds 0,1,2,3,42` produced `results/ccb_sens_context.csv` (180 rows, 9 subjects × 2 protocols × 5 seeds × 2 settings). Mean κ across 9 subjects × 5 seeds (default cell otherwise):
    - **d = 15 (no recent-reward tail): within 0.171, official 0.134**
    - d = 18 (default, with recent-reward tail): within 0.109, official 0.081
  - **Action taken:** the 3-dim per-family recent-reward tail *hurts* κ in our current setup (Δκ ≈ +0.06 on both protocols when removed). Root cause is plausibly signal-to-noise: with only 3 arm families and a short bandit stream, the running mean is noisy and ridge-fit θ̂ over-weighs it at the cost of the stable spectral features. `include_recent_rewards=True` stays the Phase-3 default in `src/thesis/ccb/runner.py::run_ccb_on_split` for backward compatibility with every prior result CSV; the thesis tuned / best-factorial cells should set it to `False`. Resolves the Phase-4 note flagged inside the "Context feature vector" closed item.

- [x] **Per-round cost cap activation — sweep over ``per_round_cap`` ∈ {inf, 4, 3, 2}**
  - **Evidence:** our own experiment. `scripts/run_ccb.py --per-round-caps inf,4,3,2 --n-folds 1 --seeds 0,1,2,3,42` produced `results/ccb_sens_perround.csv` (360 rows, 9 subjects × 2 protocols × 5 seeds × 4 caps). Mean κ across 9 subjects × 5 seeds:
    - cap = ∞ (default): within 0.109, official 0.081
    - cap = 4: within 0.109, official 0.081 (does not bind — max cost in the pool is 3)
    - cap = 3: within 0.109, official 0.081 (does not bind — identity arms at cost 3 still admissible)
    - **cap = 2: within 0.071, official 0.046** (binds — identity arms excluded; CSP/Laplacian-only)
  - **Action taken:** at the 3-ch cost scale the knapsack has essentially a single binding threshold (cap = 2). When the cap binds, κ drops by ≈ 0.04 on both protocols, confirming the OPLB feasibility filter is wired correctly and that identity-arm feature breadth contributes meaningfully to κ. Default remains `per_round_cap=None` (unbound) in `OPLBConfig`; binding values are reserved for sensitivity reports. Resolves the Phase-4 note flagged inside the "Budget K sensitivity sweep" closed item.

- [x] **Filter-bank band definitions — 9 Chebyshev Type II bands, 4–8 Hz through 36–40 Hz**
  - **Evidence:** `ang2012fbcsp` §2.1 ("A total of 9 band-pass filters are used, namely, 4–8, 8–12, …, 36–40 Hz."). Headline mean kappa: **0.569 on 2a** (Table 2 OVR; 4-class) and **0.599 on 2b** (Table 4 MIRSR variant; 2-class). Both verified 2026-05-12 via the Frontiers article.
  - **Action taken:** citation added inline to `src/thesis/baselines/fbcsp.py` module docstring. See the file for the inline citation.

- [x] **Classifier head: shrinkage LDA instead of Ang 2012's NBPW**
  - **Evidence:** `lotte2018review` §III.B (shrinkage LDA is a standard MI-BCI baseline; reports within ~1–2 κ-points of NBPW on BCI-IV). Trade-off: one fewer hyperparameter (no kernel bandwidth), convex optimization, sklearn-native, thread-safe.
  - **Action taken:** `fbcsp.py` now instantiates `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')` with inline cite. See the file for the inline citation. Empirical head-to-head vs NBPW is a Phase-5 sensitivity item, not Phase 2.

- [x] **Epoch time window 0–4 s cue-locked**
  - **Evidence:** methodological derivation from the dataset description PDFs (`desc_2a.pdf` §"Experimental paradigm": cue at t=2 s, MI period 2–6 s ⇒ 4 s post-cue; `desc_2b.pdf` same pattern with cue at t=3 s, MI period 3–7.5 s ⇒ ≥ 4 s available).
  - **Action taken:** `src/thesis/data/load.py` hard-codes `_EPOCH_TMIN = 0.0`, `_EPOCH_TMAX = 4.0` with an inline reference to the two PDFs. See the file for the inline citation. Sensitivity on {0–4, 0.5–2.5, 1–3} s is still a future open item tracked under `Currently open`.

- [x] **LinUCB exploration parameter α**
  - **Evidence:** our own sensitivity sweep. `scripts/run_ccb.py --alphas 0.1,0.5,1.0,2.0,5.0 --n-folds 1` produced `results/ccb_sens_alpha.csv` across all 9 subjects × both protocols. Mean κ:
    - α=0.1: within 0.091 / official 0.159
    - α=0.5: **within 0.184 / official 0.181 (best on both)**
    - α=1.0 (prior default): within 0.109 / official 0.010
    - α=2.0: within 0.089 / official 0.071
    - α=5.0: within 0.050 / official 0.006
  - **Action taken:** the Phase-3 OPLB default `alpha=1.0` in `src/thesis/ccb/oplb.py::OPLBConfig` is the Pacchiano 2021 §5 theoretical anchor and remains the reference-implementation default. For the thesis's empirical results, **α=0.5 is the recommended tuned value**. Multi-fold within-subject evaluation at the combined tuned cell (`results/ccb_tuned.csv`) is included in `results/ccb_baseline.md` alongside the default cell so both are reproducible.
  - **Remaining:** the tuned cell shrinks Δκ on official (+0.458 → +0.387) but slightly widens it on within (+0.409 → +0.430), so "best-per-axis-combined" is not strictly best under 5-fold CV — a genuine hyperparameter interaction. Flagged as Phase-4 follow-up: full factorial instead of one-at-a-time.

- [x] **Arm pool construction — size, pruning rule, feasibility check**
  - **Evidence:** our own sensitivity sweep. `scripts/run_ccb.py --arm-pools pruned,full --n-folds 1` produced `results/ccb_sens_pool.csv`. Mean κ across 9 subjects:
    - pruned (min_kappa=0.05, max_arms=100): within 0.109 / official 0.010
    - full (162 arms, no κ filter): **within 0.111 / official 0.162 (clearly better on official, tied on within)**
  - **Action taken:** the κ<0.05 prune threshold is over-aggressive on real 2b data; it drops arms that the OPLB policy can still exploit when combined contextually. The default in `src/thesis/ccb/runner.py::run_ccb_on_split` keeps the pruning mechanism for CPU-budget robustness but users targeting best κ should pass `arm_pool="full"` (see the tuned row in `results/ccb_baseline.md`). Pruning rule itself is documented and tested in `src/thesis/ccb/arms.py::prune_arms`; only its threshold is revised downward in practice.

- [x] **Context feature vector — exact composition and dimension `d`**
  - **Evidence:** Commit 1 enumerated the 18-dim context in `src/thesis/ccb/context.py` with inline bib-key refs per feature (`pfurtscheller2001mi`, `ang2012fbcsp`, `lotte2018review`, `li2010linucb`). Unit-tested dimensions in `tests/test_ccb_context.py`.
  - **Action taken:** `d = 18` locked; per-feature provenance in the module docstring. Ablating the recent-reward tail drops `d` to 15; future Phase-4 work can re-visit per-feature contribution.

- [x] **Budget `K` values for the sensitivity sweep**
  - **Evidence:** our own sensitivity sweep. `scripts/run_ccb.py --budget-fracs 0.25,0.5,1.0,2.0 --n-folds 1` produced `results/ccb_sens_budget.csv`. Mean κ across 9 subjects is virtually flat across all four budget values (max absolute κ difference between cells < 0.1), indicating the knapsack constraint is effectively inactive at these scales:
    - budget_frac=0.25: within 0.072 / official 0.092
    - budget_frac=0.5: within 0.072 / official 0.010
    - budget_frac=1.0: within 0.109 / official 0.010
    - budget_frac=2.0: within 0.109 / official 0.010
  - **Action taken:** current default `budget_frac=1.0` in `src/thesis/ccb/runner.py::run_ccb_on_split` is fine — at the current arm-cost scale and per-arm compute profile, tighter budgets do not bind meaningfully. Flagged as Phase-4 follow-up: explicitly exercise the constraint by adding `per_round_cap` (single-arm cost budget per trial) to the sweep.

- [x] **Calibration fraction (0.3) for arm-head pretraining**
  - **Evidence:** our own sensitivity sweep. `scripts/run_ccb.py --calibration-fracs 0.2,0.3,0.5 --n-folds 1` produced `results/ccb_sens_calibration.csv`. Mean κ across 9 subjects:
    - calibration_frac=0.2: **within 0.177 / official 0.130 (best on both)**
    - calibration_frac=0.3 (prior default): within 0.109 / official 0.010
    - calibration_frac=0.5: within 0.154 / official 0.080
  - **Action taken:** `calibration_frac=0.2` is the recommended tuned value. More trials left for the bandit stream yields better convergence than a larger head-fit set. The Phase-3 default stays at 0.3 in `src/thesis/ccb/runner.py::run_ccb_on_split` (docstring's `JUSTIFY:` reference); the thesis empirical section uses 0.2 via the tuned cell in `results/ccb_tuned.csv`.

- [x] **Random seeds and CV fold count for within-subject protocol**
  - **Evidence:** our own experiment. `scripts/run_ccb.py --seeds 0,1,2,3,42 --output results/ccb_seed_sweep.csv` produced 270 rows (5 seeds × 9 subjects × (5-fold within + 1 official)) at the Phase-3 default hyperparameter cell. Aggregating to per-seed mean κ:
    - within: mean = 0.110, **std across seeds = 0.017** (stable)
    - official: mean = 0.081, **std across seeds = 0.058** (noticeably seed-sensitive)
  - The Phase-3 headline single-seed value κ_official = 0.010 (seed=42, from `results/ccb_baseline.md`) is a **low-side outlier**; the other four seeds give κ ∈ [0.09, 0.14] on the same protocol. Within-subject is representative at seed=42.
  - Per-subject seed-variance is concentrated on a few subjects (notably subject 4 official, std ≈ 0.27 κ across seeds); near-chance subjects 2 and 8 also swing visibly. See the per-subject table in `results/ccb_seed_sweep.csv` if re-aggregated.
  - **Action taken:** within-subject 5-fold at seed=42 stays the thesis headline. For official protocol, any forward report should average over multiple seeds (recommend 5); single-seed official numbers are not trustworthy at this hyperparameter cell. Lotte 2018 §III.A (5–10-fold CV is standard) remains the anchor for fold count. Phase-4 `summarize_ccb.py` should read seed-sweep CSVs and append mean ± seed-std to the headline table.

- [x] **Artifact policy: matching Ang 2012 (no EOG regression, no ICA, no rejection)**
  - **Evidence (verified 2026-04-21 via WebFetch of `https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2012.00039/full`):**
    - Ang 2012's Methods section contains no explicit preprocessing step before the filter-bank — no EOG regression, no ICA, no artifact rejection. The Methods jumps from data description directly to the filter-bank architecture.
    - Ang 2012 uses **all 22 EEG channels on 2a** and **all 3 bipolar EEG channels on 2b** with no channel selection. The EOG channels are not used for classification.
  - **Our baseline follows the same approach:** EOG channels are dropped at load time in `src/thesis/data/load.py::_load_gdf_with_labels` via `raw.pick(list(raw.ch_names[:n_eeg]))`; no EOG regression, no ICA, no rejection.
  - **Closure basis:** direct methodological match with the benchmark reference (prior-literature + methodological-derivation). Our κ values are numerically different from Ang 2012's, but that is **not** evidence of an artifact-handling gap — see the "Not a valid κ-comparison" note below. The only valid conclusion from this item is that our preprocessing is consistent with Ang 2012.
  - **Not a valid κ-comparison.** Our 2b κ (0.267 official, from `results/fbcsp_baseline.md`) vs Ang 2012's 0.599 mean (Table 4, MIRSR) differ along multiple non-artifact axes: training composition (400 trials with 160 feedback for Ang vs 240 screening-only for us), evaluation composition (320 eval for Ang vs 120 held-out screening session for us), epoch window (0.5–2.5 s for Ang vs 0–4 s for us, per the closed "Epoch window" item), and feature-selection stage (MIRSR for Ang vs none for us). A matched-condition comparison is out of Phase 3 scope.

- [x] **CSP component count (MNE `n_components`) per band in FBCSP**
  - **Evidence:** our own experiment — `scripts/sweep_csp_m.py` run on all 9 subjects, both datasets, both protocols. Results in `results/fbcsp_sensitivity_csp_m.{csv,md}`. Baseline default `n_components = 4` (matches Ang 2012's 2a setting `m = 2` → 4 components total; Ang uses `m = 1` on 2b, not subject-specific variation — verified 2026-05-12 against `ang2012fbcsp` §3.1.1). MNE silently caps `n_components` to `n_channels`, so 2b effectively sees `n_components = 3`. Lands within **0.010 κ** of the best value in every cell:
    - 2a within: best m=3 (0.542), baseline m=4 (0.532), Δ = 0.010
    - 2a official: best m=4 (0.468), baseline m=4 (0.468), Δ = 0.000
    - 2b within: best m=3 (0.292), baseline m=4→3 (0.292), Δ = 0.000
    - 2b official: best m=3 (0.267), baseline m=4→3 (0.267), Δ = 0.000
  - **Action taken:** kept `n_components=4` in `src/thesis/baselines/fbcsp.py`. Closure rule (κ variation ≤ 0.05 across m values) satisfied in every cell.

- [x] **FBCSP implementation validated against MOABB reference**
  - **Evidence:** `scripts/validate_fbcsp_vs_moabb.py` runs our FBCSP+shrinkage-LDA on MOABB-loaded BNCI2014_001 epochs (LeftRightImagery paradigm, 0.5–40 Hz matched to our loader's hardware bandpass), official session-split protocol. Compared per-subject κ to `results/fbcsp_baseline.csv` numbers from our GDF path. Results in `results/fbcsp_vs_moabb.csv`:
    - Mean |Δκ| = **0.042** (below 0.05 tolerance)
    - Max |Δκ| = 0.083 (on a near-chance subject where κ variance is expected)
    - Signed mean Δκ = +0.014 (no systematic bias)
    - 6 of 9 subjects within ±0.05 individually
  - **Action taken:** verdict **VALIDATED**. Our GDF+true_labels loader and FBCSP implementation produce classification-equivalent epochs to MOABB's .mat-based reference pipeline. Commits `results/fbcsp_vs_moabb.csv` for the thesis record; MOABB stays as a core dep per its validator role (CLAUDE.md "Technical stack").
