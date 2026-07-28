# CLAUDE.md — Project Instructions

Persistent instructions for Claude (and any contributor) working in this repo. Read first before making changes.

> **This is the public code mirror.** It carries the implementation, the experimental record and the
> design documents. The thesis manuscript (LaTeX), the defense deck, and all EEG data — including the
> BCI Competition IV label files — are **not** distributed here; see [`README.md`](README.md) for how
> to obtain each dataset from its original source.

## What this is

Code for an MSc thesis at Sabancı University (Computer Science and Engineering). Research focus:

> **Investigating whether a Constrained Contextual Bandit (CCB) can decode cognitive states from EEG — foregrounding cognitive load from minimal, near-ear, low-channel montages (the wearable / low-resource deployment regime, evaluated under cross-session drift), with motor imagery as a classical entry point and a vigilance probe realised via the COG-BCI Psychomotor Vigilance Task. The CCB — our model — is compared, in each benchmark's own metric and under a strict no-leakage discipline, against (a) each dataset's published benchmark, (b) a classical-ML battery (FBCSP / band-power × {shrinkage-LDA, SVM-RBF, decision tree, random forest}), and (c) a compact CNN (EEGNet-8,2), and is profiled for computational efficiency.**

The finding is **negative across the panel** — the CCB does not beat any comparator on any dataset, and the deployment-regime cell (near-ear, cross-session cognitive load) is near-undecodable for *every* method (fixed, CCB, and deep). The narrative is *characterization and diagnosis*, not benchmark-victory: this rigorous, multi-paradigm, leakage-controlled demonstration of where the approach helps, where it fails, and why is the substantive contribution. The original narrow form ("3-channel 2b vs. 22-channel 2a in MI") is retained as one diagnostic gap definition under §1.2.b of the design doc (the 2a-vs-2b channel gap, run at 2-class), **not** as the headline question; 2a is *additionally* run at its full four classes for the like-for-like published-benchmark comparison.

**SEED-VIG / a separate fatigue dataset was dropped** (no data access); vigilance is covered by the realised COG-BCI PVT cell. Do not re-introduce SEED-VIG "pending access" language.

Full spec: [`design-doc/ccb-formulation.md`](design-doc/ccb-formulation.md) (see §1.4 for the locked CL-primary multi-paradigm scope).
Background primer: [`design-doc/primer.md`](design-doc/primer.md).
Open decisions: [`design-doc/open-justifications.md`](design-doc/open-justifications.md).

## Working principles, in priority order

### 1. Justification discipline — the main rule

**Every non-trivial choice must answer *"why this instead of anything else"* with evidence.** Evidence must come from one of three sources, in preference order:

1. **Prior literature** — cite a paper (BibTeX key in [`design-doc/references.bib`](design-doc/references.bib); add the entry if missing). If two papers disagree, note it and pick with reasoning.
2. **Our own experiment** — link to the script + commit hash that produced the result we rely on.
3. **Methodological derivation** — derive from an already-locked principle (e.g., *no-leakage* ⇒ a high-channel or high-resolution recording cannot inform a lower-channel / lower-resolution pipeline ⇒ no channel-subsetting of richer datasets to mimic deployment hardware; *matched conditions across datasets* ⇒ screening-only BCI-IV-2b to avoid feedback confounders absent from 2a).

If none of those is available yet:

4. **`JUSTIFY:` (DEFERRED)** — flag it inline (code comment, design-doc note) and add a tracked entry to [`design-doc/open-justifications.md`](design-doc/open-justifications.md). Never silent.

**Non-trivial** (must be justified): dataset selection, preprocessing parameters (filter cutoffs, notch, epoch windows, reference scheme, artifact policy), feature extractor, CSP/FBCSP configuration, classifier choice, hyperparameter values, evaluation protocol specifics, arm-pool composition, reward and cost definitions, algorithm variant, metric choices, regularization, transformation ordering, data-split ratios, random-seed policy.

**Trivial** (no justification needed): code style, variable naming, test scaffolding, tooling choice (uv, Makefile, ruff), commit-message wording, directory layout, CI config.

When writing a design-doc section or a code block with a non-trivial choice, explicitly state three things:

- the **choice**,
- the **alternatives considered**,
- the **reason this one wins** — ref key, experiment link, or methodological requirement.

If those three are not present, the change isn't ready to commit.

### 2. No data leakage (generalized)

A higher-channel or higher-resolution recording **never** informs a lower-channel / lower-resolution pipeline — not for channel subsetting, not for normalization statistics, not for arm pretraining, not transiently. The rule applies *per dataset and per configuration*:

- 22-channel BCI-IV-2a never reaches a CCB acting on 3-channel BCI-IV-2b (the thesis's original raison d'être; preserved as a per-paradigm diagnostic, §1.2.b of the formulation doc).
- Cho2017-full's 64-channel statistics never inform the C3/Cz/C4 monopolar-subset pipeline.
- The rich-montage near-ear CL datasets (UAB 14-channel, COG-BCI 62-channel) never inform their T7/T8 near-ear pipelines: the near-ear montage is obtained by electrode **position** at load time, never distilled from the dense montage.

Derivation: the thesis investigates how a CCB behaves under *real* deployment-hardware constraints; any use of high-resource data to inform a low-resource decision space violates that premise and would silently invalidate cross-paradigm claims.

### 3. Design-doc-first

Major new directions (new CCB variant, new evaluation protocol, new dataset, new algorithm family) → a written spec in `design-doc/` before code. Small tweaks inside an existing spec → direct code is fine.

### 4. Bilingual style

Turkish narrative and conversational replies are fine. English for every technical artifact (design docs, BibTeX, code, comments, commit messages). Don't translate technical terminology — CCB stays CCB, not "kısıtlı bağlamsal bandit."

### 5. Research integrity

- **Code is the source of truth.** When paper text (or design-doc prose) and code disagree, the code is correct unless the user explicitly states otherwise. Fix the prose, or fix the code — never leave them in silent conflict.
- **Never state a numerical result without tracing it** to a specific file, log, or code output (commit hash + script path, or a committed `results/*.csv|md`). If you cannot find the source, say so explicitly rather than repeating a remembered number.
- **Never assume domain-specific technical behavior.** Verify against code before making claims about how methods, losses, pipelines, CSP/FBCSP internals, bandit algorithms, or training procedures work here. Our implementations may diverge from textbook defaults.
- **When editing paper text, change ONLY what was requested.** Do not remove existing content, add unrequested sections, or "improve" surrounding prose unless asked. The same rule applies to design docs during review passes.
- **Citation integrity.** Verify author names, venue, and year against DBLP (or the publisher's canonical page) before adding or modifying any BibTeX entry in `design-doc/references.bib`. Flag any citation detail you cannot verify rather than guessing.

## How this plays out in practice

### When you start a task

Read, in order:

1. This file.
2. [`design-doc/ccb-formulation.md`](design-doc/ccb-formulation.md) sections relevant to the task.
3. [`design-doc/open-justifications.md`](design-doc/open-justifications.md) — is the task resolving one of these?

### When you make a non-trivial choice

Say it out loud in the chat or as a code comment, in this shape:

> **Choice:** use shrinkage LDA as the FBCSP head.
> **Alternatives considered:** (a) Naïve Bayesian Parzen Window (Ang 2012), (b) logistic regression, (c) SVM.
> **Reason:** Lotte et al. 2018 §III.B reports shrinkage LDA is a standard MI-BCI baseline with performance within ~1 % of NBPW across datasets; LDA is closed-form and thread-safe (matters for per-arm head pre-training in Formulation A). Captured in `design-doc/ccb-formulation.md` §5.1. (ref: `lotte2018review`, `ang2012fbcsp`)

### When you can't justify a choice right now

Add to [`design-doc/open-justifications.md`](design-doc/open-justifications.md) using the template there, and leave a `JUSTIFY:` comment at the choice site referencing the tracked item. Don't ship unjustified choices silently.

### Closing a justification item

Move it to the "Closed" section of `open-justifications.md` with the evidence link (ref key or commit hash), and remove the inline `JUSTIFY:` comment.

## Technical stack

- Python 3.12 via `uv` (`pyproject.toml`, `.python-version`).
- **Primary MI data path:** MNE GDF reader against the BCI-IV distribution the user places under `data/` (see `src/thesis/data/load.py`; download instructions in `README.md`). No MOABB on this path.
- **MOABB (`moabb>=1.5`, core dep)** is used for (a) cross-checking our FBCSP against a reference pipeline (`scripts/validate_fbcsp_vs_moabb.py`, BNCI2014_001 / LeftRightImagery), and (b) reaching the independent Cho2017 MI cohort. Not on the MI primary I/O path.
- **CL data path:** custom loaders under `src/thesis/data/` (`stew_load.py`, `wauc_load.py`, `emotiv_uab_load.py`, `cogbci_load.py`, `cogbci_pvt_load.py`, `near_ear.py`) for the cognitive-load and near-ear datasets MOABB doesn't cover.
- scikit-learn for the classical heads; **torch (optional `--extra benchmark`) for the EEGNet CNN comparator** (`src/thesis/baselines/cnn.py`); pyriemann is deferred until we need a Riemannian baseline.
- `Makefile` entry points: `make sync | qa | smoke | baseline | test | lint | format | fix-pth`.
- macOS quirk: uv editable `.pth` files get the `UF_HIDDEN` flag; use `make fix-pth` after any `uv sync`.

## Repo map

```
CLAUDE.md                 # This file
README.md                 # Project summary + dev setup + dataset acquisition
design-doc/
  ccb-formulation.md      # Full spec (CL-primary multi-paradigm scope locked in §1.4)
  primer.md               # Bandits + EEG (MI, CL, vigilance) + metrics background
  references.bib          # Cited bibliography
  open-justifications.md  # Tracked JUSTIFY items
src/thesis/               # Package: data/, baselines/ (fbcsp, bandpower_cl, classical [SVM/DT/RF heads], cnn [EEGNet]), ccb/ (arms, oplb, policies, runner, context, context_cl), matched, metrics, protocols
scripts/                  # QA + runners: CCB, classical battery, EEGNet (run_eegnet_*), efficiency benchmark, near-ear/cross-session, 4-class 2a, LOSO, sensitivity sweeps, summarisers
tests/                    # pytest unit tests (offline / synthetic; 273 tests, no EEG data required)
results/                  # Experiment CSVs (raw) + .md summaries; results/archive/ holds set-aside directions
data/                     # Acquisition + layout notes only (*.README.md); no EEG data is distributed
```

The BCI-IV dataset description PDFs (`desc_2a.pdf`, `desc_2b.pdf`, cited throughout the code
comments) are the competition organisers' documents — download them from
<https://www.bbci.de/competition/iv/> rather than expecting them in this repo.
