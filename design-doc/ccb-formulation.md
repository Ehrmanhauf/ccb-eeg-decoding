# Constrained Contextual Bandits on EEG — A Multi-Paradigm Investigation

**Thesis design document**

| | |
|---|---|
| **Author** | Nurullah |
| **Institution** | Sabancı University, Computer Science and Engineering |
| **Version** | 1.3 |
| **Date** | 2026-04-19 (last update: 2026-05-19, top-level framing re-anchored: cognitive load = primary paradigm, motor imagery = classical entry point, fatigue = additional investigation; see §1.4.1) |
| **Status** | Phase 4 numbers landed in §8.3–§8.4; Phase 5 items tracked in `open-justifications.md`; framing re-anchored 2026-05-19 to CL-primary multi-paradigm (this supersedes the original 3ch-vs-22ch MI gap-closure framing at the top-level; the MI material is preserved as a classical diagnostic case, not the centerpiece). |

---

## Abstract

EEG-based brain–computer interfaces are constrained by per-subject signal heterogeneity, channel-count limits of portable hardware, and the absence of a single pipeline that performs well across cognitive paradigms. This thesis is a **purely investigative study of Constrained Contextual Bandit (CCB) behaviour on EEG**, with **cognitive load on wearable-grade hardware as the primary paradigm** (STEW [lim2018stew], plus additional CL datasets to be sourced in the next research wave), motor imagery (BCI Competition IV-2a/2b, Cho2017 [cho2017mieeg]) retained as a **classical entry point and methodological diagnostic**, and fatigue / vigilance (SEED-VIG [zheng2017vigilance]) as an additional cross-paradigm investigation. The core methodological commitment is **no data leakage**: no high-channel or high-resolution recording is allowed to inform a lower-channel / lower-resolution pipeline, applied per dataset and per configuration. This document (i) formalizes the per-paradigm characterization objective, (ii) reviews the relevant literature on contextual bandits with constraints and on adaptive / bandit-driven BCI, (iii) proposes three candidate CCB formulations and recommends one (per-trial filter-bank and feature-set selection with a feature/compute budget), (iv) specifies a concrete algorithm (a LinUCB variant under a knapsack-style constraint, with Thompson Sampling as a Bayesian alternative), and (v) lays out a multi-protocol evaluation plan (within-subject, official BCI Competition where applicable, and leave-one-subject-out) with sensitivity analysis over budget, arm-pool size, and exploration parameters. The narrative is **characterization, not benchmark-victory**: negative or null findings, scientifically demonstrated across datasets, are accepted as substantive contributions. References are given in `references.bib` and cited with `[key]` below.

---

## 1. Problem Statement and Research Question

### 1.1 Research question

*How does a Constrained Contextual Bandit (CCB) policy behave when applied across multiple EEG-based classification paradigms — primarily cognitive load on low-channel wearable hardware, with motor imagery as a classical entry point, and fatigue / vigilance as an additional investigation — and what are its characteristic strengths, failure modes, and dataset-specific trade-offs?*

The investigation is **paradigm-broad and CL-emphasized**, not gap-closure. The previously-formulated narrow form ("can a CCB on 3-channel BCI-IV-2b approach the 22-channel BCI-IV-2a benchmark in MI?") is retained as **§1.2.b** below — one of several diagnostic gap definitions, not the thesis's headline question.

### 1.2 Formal objective

Let $\mathcal{P}$ be the set of paradigms in scope (currently cognitive load, motor imagery, fatigue / vigilance). For each paradigm $p \in \mathcal{P}$ let $\mathcal{D}_p$ be its dataset (or set of datasets), and let $\pi_\theta^{(p)}$ be a CCB policy instantiated on $\mathcal{D}_p$ following the action / constraint / context definitions from §2 and §5. The headline objective is per-paradigm **characterization**:

$$
\kappa^{(p)}(\pi_\theta^{(p)}) \;=\; \mathbb{E}_{(x,y)\sim\mathcal{D}_p}\!\left[\,\text{Cohen's }\kappa\!\left(\pi_\theta^{(p)}(x),\,y\right)\right],
$$

reported alongside variance over subjects, seeds, and the within-subject / official / LOSO evaluation protocols of §6.

#### 1.2.a Primary characterization (CL)

For the CL paradigm the thesis primarily reports $\kappa^{(\text{CL})}$ on STEW (and any additional CL datasets sourced in the next research wave), against published reference baselines on the same dataset and label definition (e.g., Lim 2018 reports headline $\kappa = 0.46$ on STEW with SVR + NCA). The investigation asks: *under what data and constraint regimes does the CCB framework match, fall short of, or qualitatively differ from a fixed pipeline?*

#### 1.2.b Diagnostic gap (MI, retained)

The original MI gap definition is preserved as a per-paradigm diagnostic. Let $\mathcal{D}_{22}$ be the 22-channel BCI IV-2a benchmark distribution (filtered to two MI classes: left hand 769, right hand 770) and $\mathcal{D}_{3}$ the 3-channel BCI IV-2b working distribution (screening sessions only). With $\pi_{22}$ a fixed benchmark classifier on $\mathcal{D}_{22}$ (FBCSP+LDA [ang2012fbcsp] or EEGNet [lawhern2018eegnet]) and $\pi_\theta$ acting on 3-channel trials:

$$
\Delta(\pi_\theta) \;=\; \underbrace{\mathbb{E}_{(x,y)\sim\mathcal{D}_{22}}\!\left[\mathbf{1}\{\pi_{22}(x)=y\}\right]}_{\text{22ch benchmark accuracy (fixed)}} \;-\; \underbrace{\mathbb{E}_{(x',y')\sim\mathcal{D}_{3}}\!\left[\mathbf{1}\{\pi_\theta(x')=y'\}\right]}_{\text{CCB policy accuracy on 3ch}}.
$$

$\Delta$ is reported as one diagnostic among several; it is *not* the thesis's headline metric. The no-leakage commitment is preserved: $\pi_\theta$ **never observes** samples from $\mathcal{D}_{22}$ (and analogous independence is maintained for each new paradigm; see §2.4).

### 1.3 Why this matters

Three drivers, in the priority that the new framing imposes:

1. **Application — cognitive workload monitoring.** Wearable-grade EEG (Emotiv, Muse, OpenBCI Galea, dry-electrode caps) is now broadly deployed for operator vigilance, learning-state monitoring, automotive driver assistance, and aviation crew supervision. These applications need *per-trial decision policies that adapt under hardware constraints* — exactly the regime CCB algorithms target. CL is the primary paradigm of investigation for this reason.
2. **Methodological — characterizing constrained online policies on EEG.** Classical MI-BCI work fixes a per-subject pipeline after calibration; CCB algorithms instead choose per trial under a per-trial constraint budget. Whether that flexibility helps, hurts, or is neutral on real EEG is an open methodological question. Motor imagery provides the classical entry point because the CCB literature [pacchiano2021linconstr, agrawal2019knapsacks] developed against MI-style streams.
3. **Operational — extending to fatigue, vigilance, and beyond.** A characterized CCB framework on CL and MI generalizes naturally to fatigue / drowsiness and to clinical state-monitoring streams. This generalization is investigated explicitly on SEED-VIG; further paradigms remain stretch goals.

### 1.4 Framing and scope — purely investigative across multiple paradigms

The framing-level decisions below were recorded on **2026-05-12** after a planning round; their original advisor-side phrasing is in `advisor-update-2026-04-21.md`. They supersede earlier "we will close the gap" framing.

**1.4.1 Purpose of the thesis.** The thesis is a **purely investigative study** of Constrained Contextual Bandit (CCB) behaviour on EEG across multiple paradigms. The narrative is *diagnostic and characterization*, not benchmark-victory. **Cognitive load** (STEW [lim2018stew] plus one or two additional CL datasets to be sourced in the next research wave) is the **primary paradigm of investigation** — the application target that motivates the framework and the regime in which deployable EEG hardware actually operates. **Motor imagery** on BCI-IV-2a/2b, with Cho2017 [cho2017mieeg] as an independent cohort (evaluated in full 64-channel and C3/Cz/C4 monopolar-subset configurations), provides a **classical CCB entry point and methodological diagnostic**, because the constrained-bandit literature [pacchiano2021linconstr, agrawal2019knapsacks] was first developed against MI-style streams. **Fatigue / vigilance** on SEED-VIG [zheng2017vigilance] serves as an **additional cross-paradigm investigation**. None of the three is presumed to produce a "positive" CCB-beats-baseline result. **Negative or null findings, when scientifically demonstrated across multiple datasets, are accepted as substantive contributions.** This binding follows the persistent project rule "*investigate bandit behavior, do not chase benchmark wins*" (memory: `feedback_research_framing.md`).

**1.4.2 Datasets in scope.** The locked dataset set is:

- **STEW** [lim2018stew] — primary CL dataset (14-channel Emotiv EPOC, 48 subjects, SIMKAP multitasking + rest, subjective 1–9 workload binned to 3-class; custom loader `src/thesis/data/stew_load.py`).
- **WAUC** [albuquerque2020wauc] — secondary CL dataset (8-channel Enobio dry-electrode wireless headset, 48 subjects across treadmill and stationary-bike cohorts, MATB-II under varying physical exertion; mixed labels: NASA-TLX, Borg fatigue, MATB-II behavioural performance, and seven synchronized physiological streams). Adopted 2026-05-19 (Research-wave 1 dataset survey, resolving the "CL primary-paradigm dataset set" open justification). Custom loader to be added in the next workstream as `src/thesis/data/wauc_load.py`.
- **BCI-IV-2b screening** — 3-channel CCB classical MI entry / diagnostic case (the MI dataset on which the original CCB stack was developed and Phase-4 numbers were validated).
- **BCI-IV-2a** — 22-channel MI benchmark (fixed reference, never seen by CCB).
- **Cho2017** — independent MI cohort (52 subjects, 64 channels, MI left/right hand via `moabb.datasets.Cho2017`). Run in **two configurations**: full 64-channel montage, and a C3/Cz/C4 monopolar subset that gives a same-position monopolar-vs-bipolar comparison against 2b across a ~6× larger subject pool. The 3-channel subset is hardware-deployment-driven (low-channel wearable montage) and does not consult labels or Cho2017-specific signal statistics — *no leakage* in the sense of §2.4.
- **SEED-VIG** — fatigue / vigilance (17-channel, custom loader, data access pending).

**Methodological reference (cited but not adopted for training/test):**

- **COG-BCI** [hinss2023cogbci] — 64-channel research-grade dataset (29 subjects across 3 sessions; MATB-II + N-back + Flanker + PVT; objective behavioural + subjective KSS/RSME labels). **Status changed in the 2026 near-ear reframe:** wave 1 cited it as a reference-only gold-standard (the worry being that subsetting a 64-channel montage to 8/14 channels to *mimic* STEW/WAUC would condition a low-channel decision on a richer recording). The reframe adopts it as a **primary CL dataset** without violating §2.4, because the near-ear cell is the fixed **T7/T8** pair selected by electrode *position* at load time — this removes channels, it does not distil the 64-channel signal into the 2-channel pipeline (the same no-leakage operation as the Cho2017 C3/Cz/C4 subset). COG-BCI's three sessions one week apart supply the thesis's leakage-clean **cross-session deployment headline** (train S1 → test S2/S3). It is evaluated at full montage and at the T7/T8 near-ear subset.

Stretch goals (only if Workstreams B–D close under budget): Lee2019_MI, additional wearable-grade CL datasets such as the OpenBCI-Cyton-based Arithmetic + Stroop set [Nirabi et al. 2024 Mendeley Data, dataset-only publication]. The dataset-survey rationale and the BNCI2014_004 = BCI-IV-2b resolution note were recorded in the (unpublished) planning notes; the surviving summary of the 2026-05-19 wearable-CL survey that resulted in WAUC adoption is in `open-justifications.md` ("CL primary-paradigm dataset set").

**1.4.3 Cognitive-load target space.** STEW provides subject-rated workload on a **1–9 scale**; for CCB experiments these are binned into a **3-level classification target (low / medium / high)** so that the existing classification pipeline (`ClassificationMetrics`, `run_ccb_on_split`) is directly reusable without standing up `RegressionMetrics`. The exact binning thresholds remain to be locked against Lim 2018 (IEEE TNSRE 26(11), 2018) before STEW experiments run — open in `open-justifications.md` as **"STEW label cardinality"**.

**1.4.4 Publication venue.** The chapter-level narrative is written for a **methodological BCI journal** audience (e.g., *Journal of Neural Engineering*, *IEEE Transactions on Neural Systems and Rehabilitation Engineering*). This shapes the Discussion chapter toward methodological characterization rather than ML benchmark gains, and binds the negative-results framing of §1.4.1.

**1.4.5 Implications for chapter structure.** The CL-first chapter list is: Introduction → Background → CCB Formulation → **Cognitive Load Investigation (STEW + additional CL datasets)** → MI Classical Entry (BCI-IV-2a/2b Diagnostic Case Study) → Independent MI Cohort (Cho2017, full + 3ch subset) → Fatigue Investigation (SEED-VIG) → Cross-Paradigm Discussion → Conclusion. The CL chapter is positioned first after the formulation chapter to reflect §1.4.1; the MI material is preserved (it contains the strongest validated baselines and the original CCB instantiation) but is no longer the centerpiece. Each empirical chapter is structured as *investigation* — Action / Constraint / Context defined upfront (memory: `feedback_ccb_acc_definitions.md`), signal-processing choices justified per dataset (memory: `feedback_signal_processing_exploration.md`), and operational-definition of the dataset-specific label documented (memory: `feedback_cognitive_load_methodology.md` for STEW, analogous treatment for SEED-VIG).

**1.4.6 Near-ear reframe (2026).** A 2026 advisor reframe narrows the deployment target to its real point of interest: **cognitive-load estimation from a minimal, near-ear EEG montage (T7/T8 proxy), evaluated *including under cross-session drift*** — the regime that motivates wearable deployment. Two public datasets fill the previously-empty low-channel near-ear CL cell (the "one or two additional CL datasets" placeholder of §1.4.1): **UAB Flight-Deck** [hernandez2022pilots] (14-channel Emotiv EPOC X consumer headset, graded *n*-back, 16 subjects) and **COG-BCI** [hinss2023cogbci] (64-channel, *n*-back, 29 subjects × 3 sessions). Both are evaluated at full montage and at the **T7/T8** near-ear subset (position-based selection at load time, §2.4-compliant — channels removed, not distilled). The headline becomes the leakage-clean **cross-session deployment test** on COG-BCI (train S1 → test S2/S3, no domain adaptation); the earlier five-dataset cross-paradigm negative result is retained as supporting evidence, and "near-ear" is stated plainly as a **proxy** (genuine in-ear hardware was not accessible). No primary data collection. The empirical outcome: near-ear cross-session CL decoding is at chance (best fixed κ ≈ 0.06, CCB κ ≈ 0.01), and the best-arm diagnostic localises the within-session CCB-vs-fixed gap to the **arm-bank representational ceiling**, not the bandit's selection machinery. Full plan and figure list: `design-doc/near-ear-reframe-workplan.md`.

---

## 2. Data Setup

### 2.1 Benchmark — BCI IV-2a (22-channel monopolar)

- **Channels.** 22 Ag/AgCl, monopolar, left mastoid reference, right mastoid ground; 3 EOG channels for artifact handling (not for classification) [desc_2a.pdf].
- **Subjects.** 9; two sessions per subject on different days; 6 runs × 48 trials per session = 288 trials.
- **Classes.** Originally 4-class (769/770/771/772). **Filtered here to 2 classes: 769 (left hand) and 770 (right hand)**, to match 2b.
- **Sampling.** 250 Hz; bandpass 0.5–100 Hz; 50 Hz notch; ±100 µV dynamic range.
- **Feedback.** None — matches 2b screening sessions.

### 2.2 CCB working data — BCI IV-2b (3-channel bipolar)

- **Channels.** 3 bipolar (C3, Cz, C4); Fz ground [desc_2b.pdf].
- **Subjects.** 9; 5 sessions per subject. Sessions 1–2 are screening (no feedback), sessions 3–5 are smiley feedback.
- **This thesis uses only sessions 1–2** — the feedback sessions would introduce distributional confounders absent from 2a.
- **Classes.** 2 (769 left hand, 770 right hand).
- **Sampling.** 250 Hz; bandpass 0.5–100 Hz; 50 Hz notch; ±100 µV in screening (matches 2a).

### 2.3 Harmonization for fair comparison

| Factor | 2a (benchmark) | 2b (CCB) | Harmonization |
|---|---|---|---|
| Classes | 4 → 2 | 2 | Filter 2a to {769, 770} |
| Feedback | none | mixed | Use only 2b screening |
| Sampling | 250 Hz | 250 Hz | Native match |
| Dynamic range | ±100 µV | ±100 µV (screening) | Match |
| Reference scheme | Monopolar | Bipolar | **Accepted as a realistic hardware constraint, not a confounder** |

The monopolar vs bipolar difference is framed as an inherent feature of low-cost BCI hardware, not as a nuisance to correct for.

### 2.4 No leakage — a commitment, not a preference

Prior work on "channel reduction" sometimes subsets a high-channel recording to its best K channels and calls the result a low-channel pipeline. This violates the constraint of real deployment: the channel-selection decision is itself conditioned on the 22ch data. **This thesis rejects that approach.** 2a and 2b were recorded independently and are used by separate pipelines that do not share parameters.

### 2.5 Extension cohort — Cho2017 MI (independent low-channel + full-montage comparison)

The 9 subjects of BCI-IV-2b are a small statistical base. To probe whether the CCB findings generalize, the thesis adds **Cho 2017** [cho2017mieeg] as an independent MI cohort, accessed through MOABB (`moabb.datasets.Cho2017`) and the `LeftRightImagery` paradigm.

- **Channels.** 64 Ag/AgCl, monopolar (Biosemi ActiveTwo, 10–10 montage).
- **Subjects.** 52 recorded; **50 usable** after excluding subjects s29 and s33 per the dataset authors [cho2017mieeg]. Independent from BCI-IV-2a/2b subject pools.
- **Classes.** 2 (left hand, right hand — same labels as 2b).
- **Sampling.** 512 Hz; resampled to 250 Hz on load to match the 2a/2b pipeline.
- **Trials.** 100–120 per class per subject [cho2017mieeg]; ≈200–240 trials per subject — sufficient for the CCB stream length used on 2b.
- **Feedback.** None reported in the dataset — matches the screening-only constraint imposed on 2a/2b.

**Two configurations** are evaluated:

**Cho2017-full.** Use all 64 monopolar channels. The arm bank (band × spatial-filter × feature × time-window) operates on the full montage; CSP spatial filters draw from the richer channel set. Acts as a sanity-check of CCB at *higher* effective bandwidth than 2b and as a 50-subject independent MI cohort.

**Cho2017-3ch.** Subset the recording to **C3, Cz, C4 monopolar** at load time, before any modeling. The selection is a **fixed deployment-style montage choice** (the same three positions used by the 2b bipolar hardware), not a data-driven channel selection. This produces a low-channel cohort at the same channel positions as 2b but with a *different reference scheme* (monopolar vs. bipolar). The comparison isolates the effect of reference scheme at fixed channel count and fixed task.

**A / C / C definitions for Cho2017.** Per the project's CCB instantiation discipline:

- **Action (Cho2017-full).** The same per-trial arm space as 2b — selecting one configuration (filter band × spatial filter × feature type × time window) from the generic arm enumerator. With 64 input channels, the CSP spatial-filter slot has access to a wider eigendecomposition. The action set size is bounded by the same arm-pool pruning rule as 2b.
- **Action (Cho2017-3ch).** Identical to 2b's action space — three input channels, same arm parameterization, same pruning rule. The only difference from 2b is the underlying recording.
- **Constraint.** Same knapsack-style per-trial feature/compute budget as 2b. Held *fixed* across the three datasets (2b, Cho2017-full, Cho2017-3ch) so that comparisons reflect the algorithm and the recording, not a budget change.
- **Context.** The MI context vector (`src/thesis/ccb/context.py`, d = 18) is used unchanged. For Cho2017-full the ERD indices use the C3 / C4 channel indices within the 64-channel layout; for Cho2017-3ch the indices are the same as in 2b. The recent-reward tail and bias dimensions are dataset-agnostic.

**Why this matters.** Cho2017-3ch is the most informative comparison: it tests whether the CCB findings on 2b are intrinsic to the C3/Cz/C4 motor-cortex projection or specific to the bipolar reference. Cho2017-full tests whether the CCB's value proposition survives when the deployment constraint is relaxed.

The 3-channel subset is **not** the kind of channel reduction §2.4 forbids: the selection is hardware-deployment-driven (a wearable-cap montage), not conditioned on Cho2017's labels or signal statistics. The CCB never observes the dropped channels — they are removed at load time.

### 2.6 Extension cohort — STEW cognitive workload (Lim 2018)

For the cognitive-load arm of the multi-paradigm investigation the thesis adopts **STEW** [lim2018stew], a publicly-released wearable-grade workload dataset.

- **Channels.** 14 Ag/AgCl, monopolar (Emotiv EPOC: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4) [lim2018stew].
- **Subjects.** 48 (45 with usable subjective ratings) [lim2018stew]; independent from BCI-IV-2a/2b and Cho2017 subject pools.
- **Paradigm.** Each subject completes **two** 2.5-minute EEG segments: (i) eyes-open rest baseline and (ii) the SIMKAP multitasking test [lim2018stew]. After each segment the participant reports a single subjective workload score on a **1–9 scale**.
- **Sampling.** 128 Hz. Resampled to 250 Hz on load to match the 2a/2b/Cho2017 pipeline.

**Operational definition of "cognitive load" for this thesis.** Following Lim 2018, the subjective 1–9 ratings are binned into a **3-level workload classification** with edges:

| Bin | Rating range | Label |
|-----|--------------|-------|
| 1   | 1, 2, 3      | low workload |
| 2   | 4, 5, 6      | medium workload |
| 3   | 7, 8, 9      | high workload |

This binning is the operationalization used in the original paper to report a headline kappa of κ = 0.46 (3-class classification accuracy 69 %, SVR + NCA pipeline) [lim2018stew]. We adopt it unmodified so that any CCB result can be compared to a published reference number on the same dataset and same label definition. **Provenance note:** the headline metrics (3-class κ = 0.46, accuracy 0.69) are verified verbatim against the paper's PubMed abstract (PMID 30281467); the exact edge values {1–3, 4–6, 7–9} are documented in the paper body and were corroborated via a web-indexed snippet but could not be re-extracted directly from the PDF in this session — flagged in `references.bib :: lim2018stew` for a follow-up re-verification against the IEEE Xplore HTML.

**A / C / C definitions for STEW.** Per the project's CCB instantiation discipline:

- **Action.** Same per-trial arm space template as 2b — selecting one configuration (filter band × spatial filter × feature type × time window) from the generic arm enumerator. The arm enumerator adapts to STEW's 14-channel layout (CSP spatial filters operate on 14 channels rather than the 3 bipolar derivations of 2b).
- **Constraint.** Same knapsack-style per-trial feature/compute budget as 2b and Cho2017. Held fixed across datasets so cross-dataset comparisons reflect the algorithm and the recording, not a budget change.
- **Context.** The **workload context vector** (`src/thesis/ccb/context_cl.py`, channel-layout-aware: θ/α/β log-bandpower, frontal-θ, parietal-α, frontal-α asymmetry, engagement index, artifact flag, bias, recent reward tail) replaces the MI-specific context used for 2a/2b/Cho2017. The 14-channel Emotiv frontal-temporal-occipital montage maps cleanly onto the frontal/parietal subselections in `WorkloadContext`.
- **Reward.** Binary correctness against the 3-class workload label assigned to the trial — the same reward type used on 2a/2b/Cho2017 (binary classification correctness).

**Trial definition.** STEW provides two long segments per subject rather than discrete trials. We split each 2.5-min segment into non-overlapping 4-s windows (matching the 2a/2b epoch length, §2.1–§2.2), giving ≈37 trials per condition per subject and ≈74 trials per subject across both conditions. This is *smaller* than 2b's per-subject trial budget; the choice between window length and stream length is a methodological derivation to keep one parameter (window) constant across datasets at the cost of a shorter bandit stream on STEW. Flagged as a future-work sensitivity if the κ values are noisy.

The conditional decision flagged in §1.4.3 (3-class vs. continuous regression for the CL target) is **resolved here as 3-class classification** with the binning above. Continuous regression on the 1–9 rating remains an alternative experiment (would require activating the `RegressionMetrics` scaffold from Phase 5.5), but is **not** in the locked scope.

**Measured results (Phases C–F; canonical home is Chapter 4 §4.1–§4.4 + the committed CSVs).** Within-subject 5-fold CV. Fixed-pipeline baselines (`results/fixed_baseline_cl.csv`): B1 FBCSP+sLDA κ = 0.937, B2 band-power+sLDA κ = 0.953. CCB (`results/ccb_stew_{generic,workload}.csv`): generic context κ = 0.744, workload context κ = 0.744 — context specialisation is null (Δκ = +0.000). The CCB underperforms the best fixed pipeline by **Δκ ≈ 0.21**. **Caveat:** this within-CV protocol is ceiling-saturated by within-segment leakage (random folds draw train/test epochs from the same two 2.5-min segments, making "which segment" trivially separable), so fixed κ ≈ 0.95 is a segment-identity score, *not* a workload-classification score in the Lim 2018 (κ = 0.46) sense; the directional gap is reported under this caveat. Sensitivity sweeps (`results/ccb_cl_sensitivity.md`): exploration cost and calibration overhead are genuine drivers of the gap, arm-pool composition a mild driver, non-stationarity ruled out.

### 2.7 Secondary CL cohort — WAUC (Albuquerque et al. 2020)

For the secondary CL paradigm investigation the thesis adopts **WAUC** [albuquerque2020wauc], a multimodal wearable-grade workload dataset recorded under physical activity. WAUC complements STEW along three axes: a different task family (MATB-II vs. SIMKAP multitasking), a different recording system (8-channel Enobio dry electrodes vs. 14-channel Emotiv EPOC), and the additional ecological-validity dimension of *physical-cognitive dual-task* conditions absent from STEW. The 2026-05-19 dataset survey (criteria, ranking, and the five candidates verified against their canonical pages) is summarised in `open-justifications.md`; the adoption resolved the "CL primary-paradigm dataset set" open justification.

- **Channels.** 8 dry electrodes (Enobio wireless headset by Neuroelectrics). The paper's Materials and Methods text lists the electrode positions as P3, T9, AF7, FP1, FP2, AF8, T10, P4; the on-disk column order in the released `enobio_eeg_asr.csv` files is **AF8, Fp2, Fp1, AF7, T10, T9, P4, P3** (note lower-case `Fp1` / `Fp2`). Both refer to the same eight electrodes; the loader pins the on-disk order in `WAUC_EEG_CHANNELS` (`src/thesis/data/wauc_load.py`). Wearable-grade montage, distinct from STEW's frontal-temporal-occipital Emotiv layout.
- **Subjects.** 48 filesystem subjects `S01..S48` (mapping to `Participant ID = 1001..1048` in `subjective_ratings_with_labels.csv`). After data-completeness quality control, **45 are usable** for the EEG-only CCB path:
  - **Subject 1028** has zero rows in the ratings CSV and is silently dropped (loader: `_WAUC_MISSING_RATINGS`).
  - **Subjects S23 and S26** are missing the `P4` channel from `enobio_eeg_asr.csv` (only 7 of 8 EEG channels present) and are silently dropped rather than imputed; imputation would create a channel a classifier could learn to recognize as "subject 23 / 26" (loader: `_WAUC_MISSING_CHANNELS`). Verified 2026-05-19 by inspecting all 48 EEG file headers.
  - **Subject 1020** is kept despite the upstream README's "no data" flag: verification on 2026-05-19 confirmed `S20/enobio_eeg_asr.csv` exists and has the standard 8-channel layout, and the ratings CSV does carry six rows for `1020`. The README's "no data" remark most plausibly refers to the BioHarness / Empatica streams in the raw archive.

  Cohort breakdown of the 45 surviving subjects: derived from `demographics.csv` (treadmill / bike split; both cohorts well represented).
- **Paradigm.** Each subject performs the NASA Revised Multi-Attribute Task Battery II (MATB-II) at two mental-workload levels (low / high task-difficulty) under three physical-exertion conditions (rest / medium / high) on either a treadmill or a stationary bike. The factorial structure gives $2 \times 3 = 6$ session-condition cells per subject, plus two baseline conditions (`baseline-1` eyes-closed/still and `baseline-2` movement-only) that are not part of the CCB target.
- **Sampling.** 500 Hz at acquisition. Resampled to 250 Hz on load to match the BCI-IV / Cho2017 / STEW pipeline.
- **Inherited preprocessing.** The thesis adopts the `process.rar` release variant; the EEG it contains has already been cleaned with **Artifact Subspace Reconstruction (ASR)** [mullen2015asr]. ASR is an adaptive component-subspace artifact-rejection method that operates on principal-component covariance statistics and is well-established for wearable-grade dry-electrode EEG. The CCB and fixed-pipeline baselines therefore consume the same ASR-cleaned signal; a sensitivity comparison against the un-processed `raw.rar` variant is a tracked stretch-goal sensitivity but is not in the locked scope. The Mullen et al. 2015 bibliographic fields were verified 2026-05-19 against the references list of Blum et al. 2019 (Frontiers in Human Neuroscience 13:141); the IEEE Xplore HTML was not directly re-extracted and is flagged in the `mullen2015asr` `note` field for follow-up authoritative verification before any deeper methodological claim is made.
- **NaN-marked windows from ASR.** A documented side-effect of ASR is that some windows --- those the component subspace reconstruction could not recover --- are emitted with `NaN` samples instead of imputed values (Mullen et al. 2015 §III.C). The loader applies a strict per-epoch NaN filter: any 4-s window containing *any* NaN on *any* channel is silently dropped before that epoch reaches the CCB pipeline or the fixed-pipeline baseline. Per-subject NaN impact is heterogeneous on the WAUC processed archive (verified 2026-05-19 across S01 / S02 / S03 / S05 / S10: 0 % / 35.5 % / 16.8 % / 0 % / 0 % of pre-filter epochs); the drop is logged in the loader docstring and tested in `tests/test_wauc_load.py`.
- **Labels.** Two label columns in `subjective_ratings_with_labels.csv`: (i) binary `mw_labels` ∈ {0.0, 1.0} — the **mental-workload target** for our CCB / baseline experiments, canonicalised by the loader to the strings `"low"` / `"high"`; and (ii) ternary `pw_labels` ∈ {0.0, 1.0, 2.0} — the **physical-workload covariate** (rest / medium / high), carried in `SubjectData.metadata` as the `run` column. NASA-TLX subscales (Mental Demand, Physical Demand, Temporal Demand, Performance, Effort, Frustration) and Borg perceived-exertion ratings are preserved verbatim and are available for downstream correlation analysis but are not the classifier target.

**Operational definition of "cognitive load" for WAUC.** The MW label is **binary low / high** as manipulated by MATB-II task difficulty (per the primary paper). The CCB target is the 2-class workload classification; NASA-TLX is reserved for sanity-check correlation with predicted load and is not used as a training target. Continuous-rating regression on NASA-TLX remains an alternative experiment but is **not** in the locked scope (would require activating the `RegressionMetrics` scaffold). The physical-exertion condition is treated as a *covariate*, not a target: experiments report per-physical-condition CCB performance to characterize whether the policy degrades under physical-cognitive interference.

**A / C / C definitions for WAUC.**

- **Action.** Same per-trial arm space template as STEW — selecting one configuration (filter band × spatial filter × feature type × time window) from the generic arm enumerator. The spatial-filter slot adapts to WAUC's 8-channel layout (CSP draws from 8 inputs rather than 14).
- **Constraint.** Same knapsack-style per-trial feature/compute budget as STEW and the MI cohorts. Held fixed across datasets so cross-dataset comparisons reflect the algorithm and the recording, not a budget change.
- **Context.** The **workload context vector** (`src/thesis/ccb/context_cl.py`) is reused. The Enobio frontal-parietal-temporal montage covers θ / α / β log-bandpower, frontal-θ, parietal-α, and frontal-α-asymmetry features; the engagement-index, artifact-flag, bias, and recent-reward-tail dimensions are dataset-agnostic. The channel-roles map is pinned in `WAUC_CHANNEL_ROLES` (`src/thesis/data/wauc_load.py`): frontal = [AF8, Fp2, Fp1, AF7]; parietal = [P4, P3]; left-frontal asymmetry proxy `f3` = AF7; right-frontal asymmetry proxy `f4` = AF8 (Enobio has no dedicated F3 / F4 electrodes, so the leftmost / rightmost frontal positions are used).
- **Reward.** Binary correctness against the 2-class workload label (low / high) — the same reward type used on STEW, the MI cohorts, and SEED-VIG.

**Trial definition.** MATB-II blocks in WAUC are continuous within each condition. As with STEW, we split each block into non-overlapping 4-s windows (matching the 2a/2b epoch length and the STEW window choice, §2.1–§2.6). Each block is resampled from 500 Hz native to 250 Hz before windowing, again matching the rest of the pipeline. Per-subject trial counts depend on the per-session continuous recording length; the loader processes whatever continuous EEG is tagged with `info == "session"` for each `session_no ∈ {1..6}`. Baseline rows (`info ∈ {"baseline-1", "baseline-2"}`) are excluded from the CCB target by default (`include_baselines=False`).

**Loader status (Phase A, 2026-05-19).** Custom loader `src/thesis/data/wauc_load.py` is implemented and tested against the actual `process.rar` extraction; the layout-validation script `scripts/check_wauc_data.py` (Makefile target `make wauc-check`) confirms the loader's contract on the on-disk files. Offline tests in `tests/test_wauc_load.py` exercise the channel-column resolver, the labels parser (binary `mw_labels` → low / high, ternary `pw_labels` → 0 / 1 / 2 covariate), the session-to-epoch reshape, and the end-to-end loader against synthetic data. Subject-ID translation between filesystem (`S01..S48`) and ratings (`Participant ID = 1001..1048`) is handled by `subject_id_to_partid`. The BioHarness 3 streams (`bh3_br.csv`, `bh3_ecg.csv`, `bh3_rr.csv`) remain accessible inside each subject folder but are not consumed by the CCB classifier path.

**Why WAUC and not COG-BCI [hinss2023cogbci] (wave 1; superseded by the 2026 reframe).** The 2026-05-19 survey identified the COG-BCI database (29 subjects, 64-channel ActiCap, four cognitive tasks) as the methodologically richest public CL dataset. Wave 1 held it reference-only on the worry that subsetting its 64-channel montage to an 8/14-channel target *to mimic STEW/WAUC* would condition a low-channel decision on a richer recording's statistics. **The 2026 near-ear reframe adopts COG-BCI as a primary CL dataset** and dissolves that worry: the near-ear cell is the fixed **T7/T8** pair selected by electrode position at load time (channels removed, not distilled — no-leakage-compliant under §2.4, identical to the Cho2017 C3/Cz/C4 operation), and the dataset is also used at full montage. Its multi-session design is precisely what makes the leakage-clean cross-session deployment test possible; see the near-ear reframe scope note in §1.4 and the work plan `design-doc/near-ear-reframe-workplan.md`.

**Measured results (Phases C–F; canonical home is Chapter 4 §4.1–§4.4 + the committed CSVs).** Within-subject 5-fold CV, 43 usable subjects (S39/S48 error out at LDA fit after the ASR NaN-filter collapses their label distribution to one class). Fixed-pipeline baselines (`results/fixed_baseline_cl.csv`): B1 FBCSP+sLDA κ = 0.658, B2 band-power+sLDA κ = 0.644. CCB (`results/ccb_wauc_{generic,workload}.csv`): generic context κ = 0.443, workload context κ = 0.426 — context specialisation does **not** help (Δκ = −0.017), refuting the Phase-D context-misspecification hypothesis. The CCB underperforms the best fixed pipeline by **Δκ ≈ 0.23**. Unlike STEW, WAUC is **not** ceiling-saturated, so this is a clean (leakage-free) instance of the gap. Sensitivity sweeps (`results/ccb_cl_sensitivity.md`): the WAUC mechanism profile **replicates the STEW profile cell-for-cell** (exploration cost + calibration overhead drivers, arm-pool composition mild driver, non-stationarity ruled out; cap effect identical at +0.026), establishing the gap mechanism as **paradigm-level** rather than dataset-specific.

### 2.8 Extension cohort — SEED-VIG fatigue / vigilance (Zheng & Lu 2017)

For the fatigue / vigilance arm the thesis adopts **SEED-VIG** [zheng2017vigilance], the SJTU BCMI driving-simulator dataset. SEED-VIG complements STEW by providing a *long-duration* (≈2 h per subject) drowsiness paradigm that stresses the non-stationarity machinery already validated on MI (sliding-window +0.042 κ in `open-justifications.md` :: "Non-stationary CCB").

- **Channels.** 17 EEG (6 temporal — FT7, FT8, T7, T8, TP7, TP8 — plus 12 posterior; full list confirmed in the SJTU BCMI page). 4 additional forehead EOG channels are released separately and are not used by the CCB classifier path.
- **Subjects.** 23 (mean age 23, 12 female), recorded with a 17-channel EEG cap during a virtual driving task [zheng2017vigilance].
- **Duration.** ≈2 h per subject; experiments scheduled in the early afternoon to elicit fatigue.
- **Sampling.** 1000 Hz at acquisition. The SJTU distribution publishes pre-extracted features (see "Distributed format" below), not raw 1 kHz time-domain signal.
- **Labels.** PERCLOS-based vigilance score, originally a continuous value in [0, 1]. For our CCB experiments PERCLOS is binarized to **alert (PERCLOS < 0.35) vs. drowsy (PERCLOS > 0.7)**, with intermediate values discarded; this matches the convention used by the FigShare "extracted SEED-VIG" variant and by recent cross-dataset benchmarks. The dataset itself ships the continuous PERCLOS values; the alert/drowsy binarization is a thesis-specific operationalization documented here.

**Distributed format and pipeline implication.** The SJTU release ships **differential-entropy (DE) features** in a $(17 \times 885 \times 25)$ tensor per subject (17 channels × 885 segments of 8 s × 25 frequency bands at 2 Hz resolution). It does **not** ship the raw 1 kHz time-domain EEG. This is a substantive constraint on the CCB instantiation: the existing arm bank (filter-band × spatial-filter × feature × time-window) is parameterized over *time-domain* signals, and its filter-band slot would be redundant on top of the DE features' 25 pre-binned bands. The two adaptation paths are:

- **Adaptation A — DE-feature arms.** Reframe the SEED-VIG arm space as (frequency-band subset × spatial-filter family on the DE tensor × temporal pooling), bypassing the filter-bank slot. This keeps the CCB framing but is a SEED-VIG-specific arm template; explicit re-derivation in `src/thesis/ccb/arms.py` would be required.
- **Adaptation B — request raw EEG via SJTU application.** Some prior work suggests SJTU may release raw `.mat` data upon request alongside the public DE features. If raw 1 kHz signal becomes available, the existing arm bank applies after a 1000 → 250 Hz resample, matching the 2a/2b/Cho2017 paths.

The choice between A and B is **gated on data access** (see §A/C/C below); both routes are valid investigations.

**Access constraint.** The SJTU distribution requires an application form (https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/). The thesis cannot run SEED-VIG experiments until the data arrives. The application is tracked as a workstream-level action item; until the data lands, only the loader interface, design-doc spec, and bib entry are committed.

**A / C / C definitions for SEED-VIG.**

- **Action.** Conditional on adaptation:
  - *Adaptation A (DE features):* select one configuration (frequency-band subset over the 25 SEED-VIG bands × spatial-filter family × temporal-pooling rule) per 8-s segment.
  - *Adaptation B (raw EEG):* same per-trial arm space as 2a/2b/Cho2017 — (filter band × spatial filter × feature type × time window).
- **Constraint.** Same knapsack-style per-trial feature/compute budget held fixed across datasets so cross-dataset comparisons isolate dataset effects.
- **Context.** Workload context (`src/thesis/ccb/context_cl.py`) generalises to vigilance because both rely on slow-wave band power and frontal-α modulation. The channel-layout-aware paths in `WorkloadContext` cover the 17-channel SEED-VIG montage. For Adaptation A the band-power features in the context are read directly from the DE tensor's 8–13 Hz / 4–8 Hz bins.
- **Reward.** Binary correctness against the alert/drowsy label, with intermediate-PERCLOS segments held out (matches the existing classification pipeline; no `RegressionMetrics` activation needed).

**Operational definition of "fatigue" for this thesis.** Vigilance is operationalized as the binary alert/drowsy PERCLOS classification with thresholds 0.35 and 0.7. Continuous PERCLOS regression is recognized as a valid alternative but is **not** in the locked scope, consistent with the §1.4.1 framing that this thesis investigates CCB behaviour and characterizes outcomes rather than maximizing fit to any single label scheme.

---

## 3. Why Bandits? Framing the Adaptation Problem

Classical MI-BCI pipelines fix a preprocessing/feature/classifier stack per subject after a calibration phase [lotte2018review]. Performance stalls when the pipeline is mis-specified for a subject or for a particular signal regime (drift, artifact bursts, mu/beta band shifts).

Contextual bandits [li2010linucb] offer a principled exploration/exploitation mechanism for **online configuration selection**: at each trial the agent observes context, picks a configuration (arm), observes a reward, and updates its policy. **Constraints** (budget, safety, feasibility) [badanidiyuru2018bwk, wu2015cccb, agrawal2016linbwk, pacchiano2021linconstr] reflect deployment realities — limited compute, calibration-trial budgets, or the wish to avoid catastrophic configurations.

Bandit-driven BCI work is sparse but informative:

- Fruitet, Carpentier, Munos, and Clerc [fruitet2013ucbclassif, fruitet2012neurips] introduced UCB-classif to select the best motor-imagery task for each subject online, halving calibration time for a brain-controlled button without loss in precision.
- Ma, Huggins, and Kang [ma2021tsp300] applied Thompson Sampling to stimulus selection in a P300 speller.
- Heskebeck, Bergeling, and Bernhardsson [heskebeck2022mabbci] surveyed MABs in BCI and concluded that "*MAB optimization in the context of BCI is still relatively unexplored*."

No prior work applies a **Constrained Contextual Bandit** specifically to bridge the low-channel vs high-channel gap in MI-BCI. That is the gap this thesis fills.

---

## 4. Literature Review

### 4.1 Contextual bandits with constraints

- **LinUCB** [li2010linucb]: the foundational linear contextual bandit; upper-confidence-bound policy over a linear-reward hypothesis with context-dependent arm embeddings. $\tilde{O}(d\sqrt{T})$ regret.
- **OFUL** [abbasi2011oful]: tightened confidence sets via self-normalizing martingale inequality; the de facto regret bound for linear CB.
- **Thompson Sampling for linear CB** [agrawal2013ts]: Bayesian alternative with competitive empirical performance and $\tilde{O}(d^{3/2}\sqrt{T})$ regret.
- **Bandits with Knapsacks (BwK)** [badanidiyuru2018bwk]: non-contextual setting with $d$ budget resources; each arm pull consumes resources stochastically; the game ends when any budget is exhausted.
- **Linear Contextual Bandits with Knapsacks (linCBwK)** [agrawal2016linbwk]: contextual extension of BwK where both reward and resource consumption depend linearly on context.
- **Constrained Contextual Bandits** [wu2015cccb]: the direct namesake of this thesis; logarithmic or sublinear regret for contextual bandits under a running constraint expressed as an expected cost threshold.
- **Stochastic Bandits with Linear Constraints / OPLB** [pacchiano2021linconstr]: upper-confidence algorithm for a CCB setting where the expected cost of the played policy must satisfy a linear constraint; regret scales inversely with the slack to the feasibility boundary.

### 4.2 Bandits in BCI / EEG

- [fruitet2013ucbclassif]: UCB-classif for motor-imagery task selection (left hand vs right hand vs feet vs tongue per subject); reduced calibration time by ~50%.
- [fruitet2012neurips]: NeurIPS precursor to the above, framing task-selection in MI-BCI as a best-arm identification problem.
- [ma2021tsp300]: Thompson Sampling for adaptive stimulus selection in a P300 speller; shows bandit methods accelerate visual-evoked BCI, but not motor imagery.
- [heskebeck2022mabbci]: review of MABs in BCI; notes that contextual, constrained, and non-stationary variants are largely untouched in MI-BCI.

### 4.3 Low-channel / channel-selection EEG

- [abdullah2022channelsel]: review of channel-selection techniques for MI-BCI; typical finding is that 10–30% of channels recover most of the performance of the full set, but **the selection itself is conditioned on the full data**. This confirms the leakage concern and motivates the independent-recording setup of the present thesis.
- Dry-electrode studies consistently report degraded signal quality relative to research-grade wet electrodes; adaptive/robust pipelines partially close that gap.

### 4.4 BCI-IV 2a / 2b baseline state-of-the-art

- **CSP** [koles1990csp, ramoser2000csp]: the spatial-filter workhorse; maximizes variance of one class while minimizing the other.
- **FBCSP** [ang2008fbcsp, ang2012fbcsp]: filter-bank CSP; the canonical winner on BCI-IV. The 2012 paper reports **mean κ = 0.569 on 2a (4-class, Table 2 OVR)** and **mean κ = 0.599 on 2b (2-class, Table 4 MIRSR variant)** under the official protocol. These are the reference numbers this thesis will target on the **2-class-filtered** 2a and on 2b.
- **Riemannian** [barachant2012riemann]: minimum-distance-to-mean on SPD covariance matrices; near-FBCSP performance without explicit spatial filtering.
- **Deep ConvNets / ShallowConvNet** [schirrmeister2017deep]: deep-learning decoders that match FBCSP; ShallowConvNet specifically mimics the FBCSP feature pipeline with learned filters.
- **EEGNet** [lawhern2018eegnet]: compact depthwise-separable CNN; generalizes across paradigms with limited data.
- **Lotte et al. 2018** [lotte2018review]: 10-year review of BCI classification; CSP+LDA, FBCSP, Riemannian, and deep methods are the modern standards.
- **Pfurtscheller & Neuper 2001** [pfurtscheller2001mi]: foundational treatment of MI-driven mu/beta ERD and its use in BCI.

---

## 5. Three Candidate CCB Formulations

Each formulation is specified by a 5-tuple $(\mathcal{A}, x_t, r_t, \text{constraint}, \text{regret target})$.

### 5.1 Formulation A — Per-trial filter-bank + feature-set selection *(recommended)*

- **Arms $\mathcal{A}$.** Predefined configurations of the form $(b, s, f, w)$:
  - $b$ = sub-band of the filter bank (4–8, 8–12, 12–16, …, 36–40 Hz; 9 Chebyshev-II bands as in [ang2012fbcsp] §2.1).
  - $s$ = spatial filter family (CSP variant, Laplacian, surface Laplacian approximation, or identity given only 3 bipolar channels).
  - $f$ = feature family (log-variance, Riemannian tangent vectors, log-bandpower, CSP log-variance).
  - $w$ = time window within the MI epoch (e.g., 0.5–2.5 s, 1–3 s, 2–4 s).
  Each arm maps a raw 3-channel epoch to a feature vector, followed by a pre-trained lightweight linear head.
- **Context $x_t$.** Per-trial signal statistics on the 3 channels: band-power ratios, spectral entropy, artifact indicator, variance, and a running estimate of recent arm-wise reward.
- **Reward $r_t$.** $\mathbf{1}\{\hat{y}_t = y_t\}$ (supervised; labels come from the cue).
- **Constraint.** (i) A **per-round sparsity** $|\text{active arms}| \le K$, or (ii) a **global feature budget** $\sum_t c(a_t) \le B$ where $c(a)$ is the feature-vector length or compute cost of arm $a$.
- **Regret target.** Sub-linear regret against the best fixed arm in hindsight, while respecting the constraint: $R_T = \tilde{O}(d\sqrt{T})$ with feasibility preserved.

### 5.2 Formulation B — Per-subject spatial-filter / augmentation policy

- **Arms.** Combinations of spatial filter and data-augmentation policy (e.g., additive noise, channel dropout, time-window jitter).
- **Context.** Subject-level signal profile (aggregate statistics on the calibration portion).
- **Reward.** Validation accuracy on a held-out calibration subset.
- **Constraint.** Runtime or complexity budget.
- **Regret target.** Same as A, but per-subject with $T \ll T_{\text{A}}$.

### 5.3 Formulation C — Cross-subject transfer policy *(stretch)*

- **Arms.** Transfer strategies: source-subject subset, domain alignment method (Riemannian alignment, Euclidean alignment, CORAL), or no transfer.
- **Context.** Target-subject calibration stats.
- **Reward.** Lift in accuracy over a no-transfer baseline.
- **Constraint.** Knapsack-style bound on number of source subjects or cumulative alignment cost (linCBwK-style [agrawal2016linbwk]).
- **Regret target.** Sub-linear against the best transfer strategy in hindsight.

### 5.4 Comparison

| Dimension | A (rec.) | B | C |
|---|---|---|---|
| Granularity | Per-trial | Per-subject | Per-subject |
| Context richness | High (per-trial stats) | Low (aggregate) | Moderate |
| Data efficiency | Needs many trials to learn | Efficient | Requires source subjects |
| Deployment realism | High (online BCI) | Moderate | High for clinical scale |
| Implementation complexity | Moderate | Low | High |
| Regret analysis | LinUCB / OPLB variants apply directly | Simple bandit | linCBwK needed |
| Thesis-scope fit | **Strong** | Strong but thin contribution | Stretch; likely future work |

---

## 6. Recommended Formulation — Formulation A, Precisely

### 6.1 Notation

- Trial index $t \in \{1, \ldots, T\}$ within a subject-session.
- Epoch $E_t \in \mathbb{R}^{3 \times L}$: 3 channels × $L$ samples (250 Hz, 4 s → $L=1000$).
- True label $y_t \in \{+1, -1\}$ (left vs right hand).
- Arm $a \in \mathcal{A}$, $|\mathcal{A}|=M$, $M$ on the order of $9 \times 4 \times 4 \times 3 \approx 432$ initially; pruned to $\le 100$ by design feasibility.

### 6.2 Arm feature map

Each arm $a$ deterministically maps $E_t \to \phi_a(E_t) \in \mathbb{R}^{d_a}$. A shared per-arm linear head $w_a \in \mathbb{R}^{d_a}$ produces a score $s_a(E_t) = \langle w_a, \phi_a(E_t)\rangle$; the predicted label is $\hat{y}_t = \text{sgn}(s_a(E_t))$.

### 6.3 Context vector

$x_t = \big[\bar{P}_\mu, \bar{P}_\beta, \text{ratio}_{\mu/\beta}, H_{\text{spec}}, \sigma^2(E_t), \mathbf{1}\{\text{artifact}\}, \hat{r}_{t-1,\cdot}\big] \in \mathbb{R}^{d}$, stacking channel-averaged band-power features, spectral entropy, variance, artifact flag, and a running estimate of arm-wise recent reward. Expected dimension $d \in [12, 40]$.

### 6.4 Linear CCB model

Assume $\mathbb{E}[r_t \mid x_t, a_t=a] = \langle \psi(x_t, a), \theta^\star \rangle$ for an unknown $\theta^\star$ with $\|\theta^\star\| \le S$, and assume a stochastic cost $c(a)$ with known mean. Using OFUL-style confidence sets [abbasi2011oful], define the optimistic index
$$
\text{UCB}_t(a) = \langle \psi(x_t, a), \hat{\theta}_t \rangle + \alpha \sqrt{\psi(x_t, a)^\top A_t^{-1} \psi(x_t, a)},
$$
where $A_t$ is the regularized design matrix and $\alpha$ is the exploration scale.

### 6.5 Constraint

A global knapsack on cumulative feature/compute cost: $\sum_{t=1}^T c(a_t) \le B$. This yields the **linCBwK** regime of [agrawal2016linbwk]; alternatively, if per-round sparsity is preferred, restrict $a_t$ to arms with $c(a) \le K$ at each $t$, which reduces to constrained LinUCB [pacchiano2021linconstr].

### 6.6 Regret target

Sub-linear regret against the best feasible policy in hindsight:
$$
R_T = \mathbb{E}\!\left[\sum_{t=1}^T r^\star(x_t) - r_t\right] = \tilde{O}\!\left(d\sqrt{T}\right),
$$
with feasibility maintained in expectation.

### 6.7 Why A over B and C

1. It directly operationalizes the "intelligent feature optimization" stage of the processing pipeline.
2. The constraint (compute/feature budget) is physically meaningful for low-cost BCI hardware — the whole motivation of the thesis.
3. The per-trial structure matches realistic online BCI deployment; the bandit adapts on the same time-scale as the user's trial sequence.
4. B reduces to a special case of A with coarser arms; C is interesting but requires additional datasets and a second algorithmic layer (transfer), and is therefore deferred to future work.

---

## 7. Algorithm

### 7.1 Primary candidate — LinUCB with knapsack constraint

Adapt the OPLB algorithm of [pacchiano2021linconstr] to the linCBwK setting. At each $t$:

1. **Observe** $x_t$.
2. **Feasibility filter.** Retain only arms with estimated cost UCB below the current per-round budget.
3. **Select** $a_t = \arg\max_a \text{UCB}_t(a)$ over the feasible set.
4. **Act / observe** $r_t = \mathbf{1}\{\hat{y}_t = y_t\}$ and realized cost $c_t$.
5. **Update** $A_{t+1} \gets A_t + \psi(x_t, a_t)\psi(x_t, a_t)^\top$ and $\hat{\theta}_{t+1}$ via regularized least squares.
6. **Decrement** budget $B \gets B - c_t$; terminate if $B \le 0$.

### 7.2 Bayesian alternative — Thompson Sampling with budget

Maintain a posterior over $\theta^\star$ (Gaussian with conjugate updates). At each $t$, sample $\tilde{\theta}_t$ from the posterior, play $a_t = \arg\max_a \langle \psi(x_t, a), \tilde{\theta}_t\rangle$ over feasible arms. Theoretical regret $\tilde{O}(d^{3/2}\sqrt{T})$ [agrawal2013ts]; often stronger empirical performance than UCB methods, at the cost of a weaker worst-case bound.

### 7.3 Regret-bound sketch

Under the linear reward assumption and standard sub-Gaussian noise, and under a feasibility-slack assumption $\zeta > 0$ (OPLB), regret is bounded as
$$
R_T \le C \cdot \frac{d\sqrt{T\log T}}{\zeta} + O(1),
$$
(see [pacchiano2021linconstr] Theorem 1 for the precise statement). The thesis will reproduce this bound for the specific cost structure of Formulation A.

### 7.4 Implementation sketch

Pseudocode (Python-like):

```
initialize A = λ I_d,  b = 0,  theta_hat = 0,  B_remaining = B
for t in 1..T:
    x_t = compute_context(epoch_t, history)
    feasible = [a for a in A if cost_ucb(a) <= B_remaining]
    scores = [dot(theta_hat, psi(x_t,a)) + alpha*sqrt(psi(x_t,a).T @ inv(A) @ psi(x_t,a))
              for a in feasible]
    a_t = feasible[argmax(scores)]
    y_hat = predict_with_arm(a_t, epoch_t)
    observe y_t; r_t = int(y_hat == y_t); c_t = realized_cost(a_t)
    psi_t = psi(x_t, a_t)
    A += outer(psi_t, psi_t); b += r_t * psi_t
    theta_hat = solve(A, b)
    B_remaining -= c_t
    if B_remaining <= 0: break
```

Libraries: NumPy / SciPy for linear algebra and signal processing; MNE for GDF I/O and epoching (the primary MI-side data path, against the committed BCI-IV distribution under `data/`); scikit-learn for LDA and cross-validation; MOABB on the MI path beyond BCI-IV (Cho2017) and as a cross-validator against `moabb.pipelines.FBCSP_SVM` [lotte2018review]. Custom loaders under `src/thesis/data/` (e.g., `stew_load.py`) cover paradigms MOABB does not index — currently STEW (CL primary), with additional CL loaders to follow in the next research wave, and SEED-VIG (fatigue) once data access lands. Arms that require deep-learning heads (e.g., EEGNet variants) will be pre-trained offline and used as fixed scorers.

---

## 8. Evaluation Plan

### 8.1 Three protocols (per-subject, then aggregated)

1. **Within-subject split.** For each of 9 subjects, partition trials into train (70%) / validation (15%) / test (15%); evaluate κ and accuracy. Primary protocol.
2. **Official BCI Competition protocol.** 2a: session 1 → train, session 2 → test; 2b screening sessions: session 1 → train, session 2 → test. Reproducibility anchor; directly comparable to [ang2012fbcsp] and subsequent literature.
3. **Leave-one-subject-out (LOSO).** Pool subjects, hold one out, train on the rest; repeat. Probes cross-subject generalization — hard mode. **Executed on STEW (Research-wave 1):** implemented as `thesis.protocols.leave_one_subject_out` (runner `scripts/run_loso_stew.py`; per-row output `results/loso_stew.csv`; summary `results/loso_stew.md`). The fixed baselines train on the full pooled 44-subject set; the CCB is re-cast as a population-trained cross-subject policy (per-fold training pool capped for tractability — a stated choice, not a leakage concern). The result confirms the STEW within-segment-leakage diagnosis directly: the within-CV fixed-pipeline κ ≈ 0.94 collapses to κ ≈ 0.28–0.33 once each held-out subject's two segments are unseen at training time — a *modest but genuine* cross-subject workload signal, far below the leakage-inflated ceiling but well above the chance floor. The CCB falls from 0.744 to ≈ 0.24 (per-subject mean 0.235, pooled 0.243, averaged over 5 seeds under the locked 4,000-epoch matched-conditions cap), still below the fixed pipelines (per-method table in Chapter 4 §results-loso / `results/loso_stew.md`). The earlier figure of ≈ 0.15 quoted here predated the matched-conditions re-run and is superseded. This is also the first cross-subject-evaluation step toward Formulation C (§5.3, §10).

### 8.2 Metrics

| Metric | Definition | Purpose |
|---|---|---|
| Accuracy | $\frac{1}{N}\sum_t \mathbf{1}\{\hat{y}_t = y_t\}$ | Primary classification score |
| Cohen's κ | $(p_o - p_e)/(1 - p_e)$ | Chance-corrected; BCI-IV standard |
| Cumulative regret | $\sum_t [r^\star(x_t) - r_t]$ | Bandit-specific convergence |
| **Gap-to-benchmark** | $\Delta\kappa = \kappa_{2a,\pi_{22}} - \kappa_{2b,\pi_\theta}$ | The thesis headline number |

Expected direction: $\Delta\kappa \ge 0$ (benchmark wins), with the thesis contribution being how small $\Delta\kappa$ can be made.

### 8.3 Sensitivity analysis

Sweep over:
- Budget $B$ (or per-round cap $K$): $K \in \{1, 2, 4, 8\}$ or $B \in [0.25T, 0.75T, T, 2T]$.
- Arm-pool size $M$: 20, 50, 100, 432 (full).
- Exploration parameter $\alpha$ (UCB) or prior variance (TS).
- Context dimension $d$: with / without running-reward features.

### 8.4 Ablations and comparisons

**Phase-4 policy ablation (9 subjects × 5 seeds, default cell otherwise; `results/ccb_ablation_policy.csv`):**

| Policy | within κ | official κ | Notes |
|---|---|---|---|
| OPLB (default) | 0.109 ± 0.201 | 0.081 ± 0.148 | Reference |
| Fixed-arm (top-κ calibration) | 0.122 ± 0.233 | **0.165 ± 0.209** | Beats OPLB on official by Δκ = +0.084 — session-gap-robust |
| ε-greedy (ε = 0.1) | **0.155 ± 0.203** | **0.135 ± 0.172** | Random exploration beats UCB at our T by Δκ = +0.046 / +0.054 |
| Unconstrained LinUCB | 0.109 ± 0.201 | 0.081 ± 0.148 | Identical to OPLB → knapsack slack at `budget_frac = 1.0` |

- **No-CCB baseline on 2b.** FBCSP + shrinkage LDA trained once per subject; see `results/fbcsp_baseline.md` — main in-2b comparator (official κ = 0.267). Riemannian and EEGNet are listed as Phase-5 extensions in §10.
- **Fixed-arm bandit.** Always pulls the top-κ calibration arm; no online updates. On 2b the fixed-arm baseline outperforms full OPLB on the official session-gap protocol (+0.084 κ). OPLB's online updates drift across the session gap; fixed-arm is robust because its decision rule is frozen after calibration. On within-subject splits (no session gap), fixed-arm and OPLB are essentially tied.
- **CCB without constraint.** Unconstrained LinUCB (`budget = ∞`, no per-round cap). At the default cell the knapsack is slack, so unconstrained and OPLB give identical κ. The constraint only binds when `per_round_cap ≤ 2` on our 3-channel cost scale, at which point κ drops by ≈ 0.04 (see `results/ccb_sens_perround.csv`).
- **CCB with random exploration.** ε-greedy at ε = 0.1 shares OPLB's ridge θ̂ but samples uniform-feasible on the ε-branch instead of using the UCB radius. Beats OPLB by +0.046 κ within / +0.054 κ official — evidence the OFUL-style UCB radius is mis-scaled for our short horizon T ≈ 170 trials.
- **Thompson Sampling (§7.2).** Deferred to Phase 5; no Phase-4 evidence.

**Phase-4 best-factorial cell** (`alpha = 0.5, calibration_frac = 0.3, arm_pool = pruned, include_recent_rewards = False`; from `results/ccb_factorial.csv`): within κ = 0.184 ± 0.170 (Δκ = +0.349), official κ = 0.158 ± 0.153 (Δκ = +0.310). Per-seed σ over {0, 1, 2, 3, 42}: within 0.065, official 0.047. Beats the one-at-a-time-tuned cell on official by Δκ = +0.077.

Reporting includes per-subject tables and cumulative-regret curves in `results/ccb_baseline.md`; violin plots are deferred.

**Classical-classifier comparators (B3–B5, Research-wave 1, 2026-05-29).** To test whether the CCB-vs-fixed-pipeline gap is specific to the shrinkage-LDA head or general to fixed classical classifiers, three off-the-shelf scikit-learn heads — SVM-RBF (B3), CART decision tree (B4), random forest (B5) — are fitted on the *same* engineered features as B1/B2 (FBCSP on the MI datasets; FBCSP and band-power on the CL datasets), holding the feature representation fixed and varying only the decision rule. This is the orthogonal axis to the B1-vs-B2 feature contrast. Heads are held at scikit-learn library defaults (untuned by design — see the open-justifications entry); the feature transform and the SVM's `StandardScaler` are fitted train-only per fold (no leakage). Protocol mirrors the fixed baselines exactly: within-subject 5-fold CV (seed 42) on all datasets, plus the official session split on BCI-IV-2a/2b; on Cho2017 the LDA head additionally supplies the first FBCSP+sLDA fixed-pipeline reference. Implementation: `thesis.baselines.classical`, `thesis.baselines.feature_transformers`; producer `scripts/run_classical_baselines.py`; per-row output `results/classical_baselines.csv`; summary and headline κ table in `results/classical_baselines.md`. The thesis integrates the comparator rows into the Chapter 4 baseline and cross-paradigm tables.

---

## 9. Threats to Validity

- **Monopolar vs bipolar difference.** 2a is monopolar; 2b is bipolar. Not a confounder per se (the thesis frames it as part of the low-cost constraint) but the spatial-pattern assumptions of CSP-style arms differ. Mitigation: adapt spatial-filter arms for bipolar; flag effect size in ablation.
- **Small per-subject sample sizes.** ~288 trials on 2a, ~240 on 2b screening, split across train/test. Regret convergence may be slow. Mitigation: report regret as a function of $t$; do not over-fit arm pools.
- **Inter-subject variability.** High κ variance between subjects is well-documented [ang2012fbcsp, lotte2018review]. Mitigation: report both mean and median, plus per-subject tables.
- **Artifact handling.** EOG is present in 2a but not 2b; pipelines must be artifact-aware. Mitigation: use automatic artifact flags (variance thresholds, kurtosis, ICA) consistently across protocols.
- **Non-stationarity within and across sessions.** Mitigation: per-session normalization; optional sliding-window adaptation of $\hat{\theta}$.
- **Algorithm / hyperparameter over-tuning.** Sensitivity sweep is the defense; report the full grid, not just the best cell.
- **BCI illiteracy.** 15–30% of MI-BCI users show chance-level performance regardless of pipeline [lotte2018review]. Mitigation: identify and report these subjects separately; do not treat them as noise.
- **Label leakage via reward signal.** In a supervised setting the reward is the label itself; care is needed when reporting "online" learning to avoid implicit test-time label access. Mitigation: evaluate on held-out trials only; report regret on training portion.

---

## 10. Future Extensions

This list mixes two kinds of item, kept distinct so the status is unambiguous. The first group was **prototyped during Phase-5, characterised, and set aside** — each was implemented and swept, none changed the cross-paradigm conclusion, and all are retained as reusable options for future datasets where the session dynamics or the exploration scale may differ. The second group is **genuinely future** (unstarted), gated mostly on external data access or on a new formulation spec.

### Prototyped, characterised, and archived (available for reuse)

- **Non-stationary CCB.** SlidingWindow-UCB / Discounted-UCB variants [heskebeck2022mabbci] for drifting EEG were added to `OPLBConfig` (`window_size`, `discount_gamma`); `OPLB.update` rebuilds the design matrix from a history buffer when either is active. A 50-round sliding window is the largest single Stage-1 fix on the 2b session-gap (official) protocol; pure discounting barely moves κ, and the gain does not alter the headline. Full sweep in `open-justifications.md` :: "Non-stationary CCB".
- **Joint arm-and-classifier learning.** Online per-arm head updates landed as `ArmHead.partial_fit` plus the `run_ccb_on_split(..., online_heads=True)` path (the CSP spatial filter stays frozen; only the shrinkage-LDA head re-solves on the running feature buffer). The effect is small and confined to the session-gap protocol, so frozen heads remain the default. See `open-justifications.md` :: "Online per-arm head updates" and `results/archive/ccb_online_heads.csv`.
- **Thompson-sampling exploration.** A LinTS policy landed in `thesis.ccb.policies` and was swept against OPLB and ε-greedy; it does not unseat ε-greedy on the short MI horizon ($T \approx 170$), confirming that *structured* exploration over-explores at this scale. See `open-justifications.md` :: "Thompson Sampling policy". Frozen-OPLB remains the headline policy.

### Genuinely future (unstarted)

- **Additional cognitive-load datasets.** The CL primary investigation currently rests on STEW alone (§2.6); the open-justifications entry "CL primary-paradigm dataset set" tracks the survey and adoption of one or two additional CL datasets (candidates include MATB-derived workload sets, n-back EEG corpora, and learning-state datasets) to broaden the CL chapter's empirical base. Gated on dataset access. Each new CL dataset will receive its own §2.x section with operational label definition and A/C/C decomposition.
- **Transfer to fatigue and drowsiness.** Beyond SEED-VIG (itself blocked on the SJTU BCMI data-access application, §2.7), additional fatigue / drowsiness datasets (with longer streams, different paradigms, or different label provenance) can extend the fatigue chapter; the CCB framework is label-agnostic and supports them via the same `enumerate_arms_generic` + `WorkloadContext` wiring.
- **Formulation C (cross-subject transfer).** Once Formulation A is characterized per paradigm, combine with transfer-learning arms (Riemannian / Euclidean / CORAL alignment, source-subject subset selection — §5.3) to investigate whether cross-subject pre-training narrows the per-paradigm κ variance. The leave-one-subject-out protocol on STEW (now **implemented** — §8.1 protocol 3; runner `scripts/run_loso_stew.py`; result in Chapter 4) is a first cross-subject *evaluation* step in this direction, though it reuses the existing arm bank and a population-trained policy rather than dedicated transfer arms.

---

## 11. References

See `references.bib` (BibTeX). Inline citations follow the `[key]` convention. Key references by bucket:

- **CCB families**: li2010linucb, abbasi2011oful, agrawal2013ts, badanidiyuru2018bwk, wu2015cccb, agrawal2016linbwk, pacchiano2021linconstr.
- **Bandits in BCI**: fruitet2013ucbclassif, fruitet2012neurips, ma2021tsp300, heskebeck2022mabbci.
- **Low-channel / channel selection**: abdullah2022channelsel.
- **MI-BCI foundations and baselines**: koles1990csp, ramoser2000csp, pfurtscheller2001mi, ang2008fbcsp, ang2012fbcsp, barachant2012riemann, schirrmeister2017deep, lawhern2018eegnet, lotte2018review.
