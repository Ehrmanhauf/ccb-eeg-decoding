# Constrained Contextual Bandits for Resource-Limited EEG Decoding

A research codebase for a leakage-controlled, multi-paradigm evaluation of a **Constrained
Contextual Bandit (CCB)** as an online EEG pipeline selector. This is the public code mirror of an
MSc thesis at Sabancı University (Computer Science and Engineering).

> **What is in this repo:** the implementation, the full experimental record (result CSVs and
> summaries), and the design documents that justify every methodological choice.
> **What is not:** the thesis manuscript, the defense material, and **any EEG data** — every
> dataset must be obtained from its original provider (see [Getting the data](#getting-the-data)).

---

## What this is

At each trial, a bandit picks one EEG feature-extraction pipeline — a combination of
**frequency band × spatial filter × feature family × time window** — from a bank of arms, observes
whether the resulting classification was correct, and updates its policy, all under a fixed
per-round compute/feature-vector budget. The bandit is the **Optimistic–Pessimistic Linear Bandit
(OPLB)**; the *context* is a per-trial vector of spectral and temporal signal statistics, the
*reward* is classification correctness, and the *constraint* is the feature budget.

The question is whether that adaptivity buys anything in the regime that matters for wearables:
**cognitive load decoded from a minimal, near-ear, low-channel montage, under cross-session drift.**
Motor imagery serves as the classical entry point and a vigilance probe (COG-BCI PVT) as a
secondary cell. The CCB is compared against each dataset's published benchmark, a classical-ML
battery (FBCSP / band-power × {shrinkage-LDA, SVM-RBF, decision tree, random forest}), and a compact
CNN (EEGNet-8,2), then profiled for computational efficiency.

### The finding is negative — and that is the point

Across a seven-dataset, two-paradigm panel (channel counts 2–64), the CCB does **not** reach
fixed-pipeline Cohen's κ on any leakage-clean cell, and near-ear cross-session cognitive load is
near-undecodable for *every* method tested — fixed pipelines, the CCB, and the CNN alike. The
contribution is the rigour and breadth of that demonstration, plus a methodological result of
independent value: **the near-perfect within-subject-CV numbers commonly reported for low-channel
cognitive-load decoding are a recording-identity leakage artefact that collapses under
leakage-clean protocols** (STEW: κ 0.95 → 0.31 going from within-CV to LOSO).

This repo is organised so those claims can be checked rather than taken on trust: every reported
number traces to a committed CSV under [`results/`](results/), and
[`scripts/make_tables.py`](scripts/make_tables.py) recomputes them from those CSVs.

---

## Quick start

No EEG data is needed to install the project and run the full test suite.

```bash
brew install uv
```

```bash
uv sync --all-extras
```

```bash
make fix-pth
```

```bash
make test
```

That is 273 tests, fully offline/synthetic — no EEG files required. (`uv` install docs:
<https://docs.astral.sh/uv/>. The `--all-extras` flag pulls in torch for the EEGNet comparator and
Plotly for the demos.)

**Invoke Python directly, not through `uv run`.** On macOS, `uv run` re-syncs the venv on every
call, and uv's installer sets the `UF_HIDDEN` flag on editable-install `.pth` files, which makes
CPython's `site.py` silently skip them (`import thesis` then fails). `make fix-pth` strips the flag
— rerun it after any `uv sync`. The working invocation is:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_ccb.py --help
```

`make help` lists the other entry points (`qa`, `smoke`, `baseline`, `lint`, `format`,
`stew-check`, `wauc-check`).

---

## Getting the data

**No EEG recordings, label files, or dataset description documents are redistributed here.** Each
dataset is under its provider's own terms; download them yourself into `data/` or `new_datasets/`
(both gitignored). The three `data/*.README.md` notes carry the per-dataset extraction commands,
the expected on-disk layout, and known quirks.

| Dataset | Paradigm | Ch. | Subj. | Source | Lands in |
|---|---|---|---|---|---|
| **STEW** (Lim 2018) | Cognitive load | 14 (Emotiv EPOC) | 45 | [IEEE DataPort](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset) (free account) | `data/STEW/` — [notes](data/STEW.README.md), verify with `make stew-check` |
| **WAUC** (Albuquerque 2020) | Cognitive load | 8 (Enobio dry) | 43 | [MuSAE Lab](https://musaelab.ca/wauc-dataset/) — `process.rar`, ~4.5 GB | `data/WAUC/process/` — [notes](data/WAUC.README.md), verify with `make wauc-check` |
| **UAB Flight-Deck** | Cognitive load | 14 (EPOC X) → T7/T8 | 16 | UAB DDD, DOI `10.5565/ddd.uab.cat/259591` (CC-BY) | `new_datasets/workload_dataset/` — [notes](data/NEW_DATASETS.README.md) |
| **COG-BCI** (Hinss/Roy 2022) | Cognitive load + vigilance | 62 (ActiCap) → T7/T8 | 29 × 3 sessions | Zenodo record `7413650` | `new_datasets/7413650/` |
| **COG-BCI competition split** | Cognitive load (MATB) | 61 → T7/T8 | 2 sessions | Zenodo record `5055046` | `new_datasets/5055046/` |
| **BCI-IV-2a** | Motor imagery | 22 monopolar | 9 | [bbci.de/competition/iv](https://www.bbci.de/competition/iv/download/) (agree-and-submit) + [true labels](https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip) | `data/BCICIV_2a_gdf/`, `data/true_labels/2a/` |
| **BCI-IV-2b** | Motor imagery | 3 bipolar | 9 | same page, `BCICIV_2b_gdf.zip` + [ds2b true labels](https://www.bbci.de/competition/iv/results/ds2b/true_labels.zip) | `data/BCICIV_2b_gdf/`, `data/true_labels/2b/` |
| **Cho2017** | Motor imagery | 64 / C3-Cz-C4 | 50 | fetched on demand via **MOABB** (cached in `~/mne_data`) | no manual step |

The **near-ear cells** are produced by subsetting any T7/T8-bearing montage to those two channels
**by electrode position, at load time** ([`src/thesis/data/near_ear.py`](src/thesis/data/near_ear.py))
— never by distilling a dense montage. That is a hard rule, not an implementation detail; see
[No leakage](#no-leakage--the-rule-that-shapes-the-code) below. "Near-ear" is a stated **proxy**:
genuine in-ear hardware was not accessible, so the near-ear signal is approximated by the temporal
pair T7/T8, which is present on every montage in the panel.

The BCI-IV description PDFs (`desc_2a.pdf`, `desc_2b.pdf`) — cited throughout the code comments as
the authority for montage order and trial timing — are the competition organisers' documents; get
them from <https://www.bbci.de/competition/iv/>.

---

## Architecture

### `src/thesis/ccb/` — the bandit

| Module | Role |
|---|---|
| [`arms.py`](src/thesis/ccb/arms.py) | The arm bank: each arm is a (band, spatial filter, feature family, window) pipeline with its own classifier head and a feature-vector cost. Per-montage enumerators. |
| [`context.py`](src/thesis/ccb/context.py) / [`context_cl.py`](src/thesis/ccb/context_cl.py) | Per-trial context vectors (spectral + temporal statistics) for the MI and cognitive-load paradigms. |
| [`oplb.py`](src/thesis/ccb/oplb.py) | Optimistic–Pessimistic Linear Bandit: optimistic reward estimate, pessimistic cost estimate, budget constraint. |
| [`policies.py`](src/thesis/ccb/policies.py) | Selection policies and the ablation variants. |
| [`runner.py`](src/thesis/ccb/runner.py) | Drives a bandit over a stream — per-round arm pull, reward, update — then a frozen-arm test pass. |
| [`preprocessing.py`](src/thesis/ccb/preprocessing.py) | Shared filtering/epoching used by the arms. |

### `src/thesis/baselines/` — the comparators

[`fbcsp.py`](src/thesis/baselines/fbcsp.py) (filter-bank CSP),
[`bandpower_cl.py`](src/thesis/baselines/bandpower_cl.py) (band-power features for cognitive load),
[`classical.py`](src/thesis/baselines/classical.py) (shrinkage-LDA / SVM-RBF / decision-tree /
random-forest heads), [`cnn.py`](src/thesis/baselines/cnn.py) (EEGNet-8,2 — needs the `benchmark`
extra for torch), [`feature_transformers.py`](src/thesis/baselines/feature_transformers.py).

### Evaluation

- [`protocols.py`](src/thesis/protocols.py) — within-subject 5-fold CV, the official session split
  (MI anchor), **leave-one-subject-out**, and **cross-session** (train session 1 → test 2/3, the
  deployment-drift test). On block-homogeneous cognitive-load labels, within-CV is
  leakage-confounded; the honest CL protocols are LOSO and cross-session.
- [`matched.py`](src/thesis/matched.py) — the fairness guarantee: all three method families (CCB,
  classical battery, EEGNet) train on the *identical* per-fold pool from a single split source.
  Regression-tested by [`tests/test_matched_conditions.py`](tests/test_matched_conditions.py).
- [`metrics.py`](src/thesis/metrics.py) — Cohen's κ (headline), accuracy, cumulative regret.

### No leakage — the rule that shapes the code

A higher-channel or higher-resolution recording **never** informs a lower-channel pipeline: not for
channel subsetting, not for normalisation statistics, not for arm pretraining, not transiently.
22-channel BCI-IV-2a never reaches a CCB acting on 3-channel 2b; Cho2017's 64-channel statistics
never inform the C3/Cz/C4 subset; the dense CL montages never inform their T7/T8 near-ear
pipelines. This is enforced mechanically and regression-tested — see [`CLAUDE.md`](CLAUDE.md) §2
for the derivation.

---

## Reproducing a result

Every runner writes a CSV under `results/`. A few representative entry points — each has `--help`,
and most accept `--subjects` for a fast smoke run:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_loso_stew.py --subjects 1,2,3,4 --output results/loso_stew_smoke.csv
```

```bash
PYTHONPATH=src .venv/bin/python scripts/run_cogbci_crosssession.py --task matb --montages nearear --subjects 1,2 --seeds 0 --output results/crosssession_smoke.csv
```

```bash
PYTHONPATH=src .venv/bin/python scripts/run_classical_baselines.py --datasets stew,wauc --subjects 1,2,3 --output results/classical_smoke.csv
```

```bash
PYTHONPATH=src .venv/bin/python scripts/run_eegnet_benchmark.py --datasets bci2a --subjects 1,2 --seeds 42 --epochs 10 --output results/eegnet_smoke.csv
```

Two aggregation gates run off the **committed** CSVs, so they need no EEG data at all:

```bash
PYTHONPATH=src .venv/bin/python scripts/make_tables.py
```

```bash
PYTHONPATH=src .venv/bin/python scripts/consolidate_results.py
```

`make_tables.py` recomputes every reported cell from `results/*.csv` and flags discrepancies — it
exists because hand-copying numbers between analysis and write-up had caused exactly that. Figures
rebuild the same way, into a generated `figures/` directory:

```bash
PYTHONPATH=src .venv/bin/python scripts/figures/make_figures.py
```

### Interactive demos

[`scripts/build_demos.py`](scripts/build_demos.py) builds three self-contained offline HTML demos
(the leakage collapse, the best-arm diagnostic, and a live OPLB stream) into a generated `demos/`
directory; `--check` verifies their numbers against the CSVs without rendering. Requires the
`demos` extra (Plotly). [`scripts/make_demo_videos.py`](scripts/make_demo_videos.py) renders the
same three to MP4.

```bash
PYTHONPATH=src .venv/bin/python scripts/build_demos.py --check
```

---

## Results index

[`results/results_master.csv`](results/results_master.csv) is the consolidated panel — one row per
(dataset, paradigm, channels, montage, protocol) cell, with columns for the best **fixed** pipeline
κ, the **CCB** κ, **EEGNet** κ, the **published** benchmark where one exists, a `leak` flag marking
leakage-confounded protocols, and `delta_ccb` (fixed − CCB).

Alongside each raw CSV, the `results/*.md` files narrate what the run showed and how to read it.
Notable ones: [`loso_stew.md`](results/loso_stew.md) and [`loso_wauc.md`](results/loso_wauc.md)
(the leakage collapse), [`crosssession_cogbci.md`](results/crosssession_cogbci.md) and
[`crosssession_matb.md`](results/crosssession_matb.md) (the deployment-regime cells),
[`best_arm_diagnostic.md`](results/best_arm_diagnostic.md) (localising the CCB-vs-fixed gap to the
arm bank's representational ceiling rather than the bandit's online selection),
[`hardware_efficiency.md`](results/hardware_efficiency.md) (the compute profile), and
[`citation_audit.md`](results/citation_audit.md). `results/archive/` holds set-aside directions,
kept for provenance.

---

## Repo layout

```
src/thesis/       ccb/ (arms, context, context_cl, oplb, policies, preprocessing, runner)
                  baselines/ (fbcsp, bandpower_cl, classical, cnn, feature_transformers)
                  data/ (load, moabb_load, stew_load, wauc_load, emotiv_uab_load,
                         cogbci_load, cogbci_matb_load, cogbci_pvt_load, near_ear)
                  matched.py, metrics.py, protocols.py, features.py
scripts/          runners (run_ccb*, run_loso_*, run_cogbci_crosssession, run_matb_competition,
                  run_classical_baselines, run_fixed_baselines_*, run_eegnet_*, run_2a_4class,
                  run_pvt, run_hardware_efficiency_benchmark), sensitivity sweeps (sweep_*),
                  aggregation (make_tables, consolidate_results, bootstrap_ci, summarize_*),
                  figures/make_figures.py, demo builders, data checkers
tests/            273 offline/synthetic pytest tests
results/          experiment CSVs + .md summaries + results_master.csv + archive/
design-doc/       ccb-formulation.md (full spec), primer.md (bandits + EEG background),
                  open-justifications.md (tracked open items), references.bib — see
                  design-doc/README.md for the index
data/             acquisition + layout notes only — no data is distributed
```

## Contributing

[`CLAUDE.md`](CLAUDE.md) is the contributor guide (it doubles as the instruction file for Claude
Code). Two rules matter most. **Every non-trivial choice must be justified** — by a citation, an
experiment link, or a derivation from a locked principle; anything unjustified gets a `JUSTIFY:`
marker and a tracked entry in
[`design-doc/open-justifications.md`](design-doc/open-justifications.md). And **the no-leakage rule
is not negotiable**. New directions get a written spec in `design-doc/` before code.

The written specification lives in [`design-doc/`](design-doc/README.md) — start with
[`primer.md`](design-doc/primer.md) if bandits or EEG signal processing are new to you, then
[`ccb-formulation.md`](design-doc/ccb-formulation.md) for the full spec.

## License

Code, documentation and result files: **MIT** — see [`LICENSE`](LICENSE).

Datasets are **not** covered by that licence and are not distributed here; each remains under the
terms set by its original providers — see [`NOTICE.md`](NOTICE.md). If you use one, cite its source
paper; BibTeX entries for all of them are in
[`design-doc/references.bib`](design-doc/references.bib).

## Citation

If this code is useful in your work, please cite the thesis:

> Cebeci, M. N. (2026). *Constrained Contextual Bandits for Resource-Limited EEG Decoding:
> A Leakage-Controlled, Multi-Paradigm Evaluation*. MSc thesis, Sabancı University.
