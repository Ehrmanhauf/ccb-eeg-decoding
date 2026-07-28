# WAUC data — access + manual extraction instructions

The **WAUC** dataset (Albuquerque et al. 2020, *Frontiers in Neuroscience* 14:549524; `albuquerque2020wauc`) is not vendored in this repo. It is the secondary cognitive-load dataset adopted in Research-wave 1 (2026-05-19); the spec is locked in `design-doc/ccb-formulation.md` §2.7.

## 1. Access procedure

The new CSV release of the WAUC dataset is hosted by the **MuSAE Lab** at INRS-EMT (P.I.: Tiago H. Falk) at <https://musaelab.ca/wauc-dataset/>. As of 2026-05-19 the dataset is **openly downloadable** (no application form). Three archive variants are available:

| Variant | Size | Contents |
|---|---|---|
| `raw.rar`     | 4.21 GB | Unprocessed multi-modal recordings (EEG + BioHarness + Empatica) |
| `process.rar` | 4.45 GB | **ASR-processed EEG**, filtered ECG + BR, derived RR intervals |
| `features.rar`| 109 MB  | Features extracted on baseline-1 and baseline-2 only |

Plus the two top-level CSVs that the dataset distributes openly without login:

| CSV | Size | Contents |
|---|---|---|
| `subjective_ratings_with_labels.csv` | ≈10 KB | NASA-TLX subscales + binary `mw_labels` + ternary `pw_labels` per (participant, session_no) |
| `demographics.csv`                   | ≈1 KB  | Age, sex, height, weight, activity (treadmill / bike) per participant |

This thesis uses **`process.rar` (ASR-applied EEG)**, with the inherited preprocessing documented in `design-doc/ccb-formulation.md` §2.7. The choice trades methodological control for reproducibility against Albuquerque 2020's published pipeline; ASR is a well-known artifact-rejection technique (Mullen et al. 2015 — citation pending verification in the BibTeX before any thesis claim that depends on it).

## 2. Extract `process.rar` under `data/WAUC/process/`

```bash
# One-time tool install (free; no licence restrictions).
brew install unar

# Unrar in place. Resulting layout: data/WAUC/process/S01/ ... S48/.
cd data/WAUC && unar -force-overwrite process.rar
```

Other rar-capable tools (`rar`, `7z`, `unrar`, the GUI app **The Unarchiver**, …) work equivalently. After extraction the layout under the repo root must look like:

```
data/WAUC/
├── demographics.csv
├── features.rar                       # may be left as .rar; not used by the loader
├── process/
│   ├── S01/
│   │   ├── enobio_eeg_asr.csv         # 8-ch ASR-processed EEG, 500 Hz native
│   │   ├── bh3_br.csv                 # BioHarness 3 breathing rate
│   │   ├── bh3_ecg.csv                # BioHarness 3 ECG
│   │   └── bh3_rr.csv                 # BioHarness 3 RR intervals
│   ├── S02/
│   …
│   └── S48/
├── process.rar                        # keep or delete after extraction; not used by the loader
├── raw.rar                            # not used by the locked pipeline
└── subjective_ratings_with_labels.csv
```

Subject IDs on the filesystem are `S01..S48` (2-digit zero-padded). In `subjective_ratings_with_labels.csv` and `demographics.csv` the same subjects are referenced as `Participant ID = 1001..1048`; the loader maps between them via `partid = 1000 + sid` (see `thesis.data.wauc_load.subject_id_to_partid`).

The `data/WAUC/` directory is `.gitignore`-d.

### EEG file column structure (verified 2026-05-19 against `S01/enobio_eeg_asr.csv`)

| Order on disk | Column | Meaning |
|---|---|---|
| 1–8 | `AF8, Fp2, Fp1, AF7, T10, T9, P4, P3` | The 8 EEG channels (note: `Fp1`/`Fp2` use lower-case `p`) |
| 9   | `fs`         | Native sampling frequency (constant 500.0 Hz) |
| 10  | `info`       | One of `session`, `baseline-1` (eyes-closed + still), `baseline-2` (movement only) |
| 11  | `session_no` | Session identifier; 1..6 over the 6 condition cells; 0 for baseline rows |

(The two-stage layout — channels first, metadata last — differs from the original GitHub README's prose, which placed metadata before channels. The loader and `_resolve_channel_columns` were aligned to the actual on-disk order during Phase A.)

### Labels CSV structure

`subjective_ratings_with_labels.csv` carries one row per (participant, session_no):

| Column | Meaning |
|---|---|
| `Participant ID` | Integer 1001..1048 (subject) |
| `Mental Demand`, `Physical Demand`, `Temporal Demand`, `Performance`, `Effort`, `Frustration` | NASA-TLX subscale ratings |
| `Perceived Exertion (1)`, `Perceived Exertion (2)` | Borg fatigue ratings |
| `mw_labels` | **Binary mental-workload target** (float `0.0` = low, `1.0` = high). Loader canonicalises to `"low"` / `"high"` string. |
| `pw_labels` | **Ternary physical-workload covariate** (float `0.0` / `1.0` / `2.0`). Carried in `SubjectData.metadata` as the `run` column for stratification; not the classifier target. |
| `session_no` | 1..6 |

## 3. Validate the layout

```bash
make wauc-check
```

The script (`scripts/check_wauc_data.py`):

- Parses `subjective_ratings_with_labels.csv` and reports the number of (participant, session_no) records and the drop list (currently `{1028}`).
- Spot-checks 5 random filesystem subjects: confirms `process/S{NN:02d}/enobio_eeg_asr.csv` exists, has all 8 EEG channel columns, and that `info` and `session_no` take only the documented values.
- Exits non-zero on any failure.

## 4. What the loader does with the files

See `src/thesis/data/wauc_load.py` and `design-doc/ccb-formulation.md` §2.7 (the canonical operational definition for WAUC in this thesis):

- For each subject, the six session blocks are read from `enobio_eeg_asr.csv`, resampled from 500 Hz → 250 Hz (matching the rest of the pipeline), and split into non-overlapping `epoch_seconds`-wide windows (default 4 s).
- Each window inherits the **binary low / high MW label** of its session from `subjective_ratings_with_labels.csv` (column `mw_labels`). The physical-workload condition (`pw_labels`) travels with the trial as the `run` metadata column.
- Baseline-1 / baseline-2 rows are excluded from CCB training data by default (`include_baselines=False`).

### Data-integrity caveats from the release notes + 2026-05-19 verification

- **Subject 1028**: no rows in the ratings CSV → loader drops `S28` silently.
- **Subjects S23 and S26**: each `enobio_eeg_asr.csv` is missing the `P4` column (7 of 8 EEG channels present). The loader silently drops these two subjects rather than imputing a phantom `P4` channel (imputation would create a channel a classifier could learn to recognize as "subject 23 / 26", confounding any per-subject analysis). Documented in `src/thesis/data/wauc_load.py` :: `_WAUC_MISSING_CHANNELS`.
- **Subject 1020**: GitHub README claims "no data" but the ratings CSV does contain 6 rows for `1020` and `S20/enobio_eeg_asr.csv` exists in the processed archive. The README's "no data" remark most plausibly refers to BioHarness / Empatica streams in the raw archive; for the EEG-only CCB path the loader **keeps** `S20`. If a downstream analysis specifically requires non-EEG modalities, re-evaluate.
- **NaN-marked EEG windows**: ASR can leave windows that the component-subspace reconstruction could not recover as `NaN`-marked samples (Mullen et al. 2015 §III.C). The loader filters out any epoch containing *any* NaN sample on *any* channel — a strict per-trial quality-control step. Per-subject NaN impact is heterogeneous (verified 2026-05-19 across S01/S02/S03/S05/S10: 0–35% of pre-filter epochs); see `_session_to_epochs` for the implementation.
- BioHarness 3 missing sessions for 1004 / 1019 / 1032 / 1035: does not affect the EEG path.
- Empatica E4 missing entirely for 1001 / 1025 / 1028 / 1038: does not affect the EEG path.

**Net usable WAUC subject count: 45** (48 filesystem - 1 missing ratings - 2 missing channels).

## 5. License / attribution / citation

The dataset is openly distributed by the MuSAE Lab. The suggested citation per the MuSAE Lab release notes is:

```bibtex
@article{albuquerque2020wauc,
  author  = {Albuquerque, Isabela and Tiwari, Abhishek and Parent, Mark
             and Cassani, Raymundo and Gagnon, Jean-Fran\c{c}ois
             and Lafond, Daniel and Tremblay, S\'ebastien and Falk, Tiago H.},
  title   = {{WAUC}: A Multi-Modal Database for Mental Workload
             Assessment Under Physical Activity},
  journal = {Frontiers in Neuroscience},
  volume  = {14},
  pages   = {549524},
  year    = {2020},
  doi     = {10.3389/fnins.2020.549524},
}
```

The full entry, with the locked verification note, is already in `design-doc/references.bib` under the key `albuquerque2020wauc`.
