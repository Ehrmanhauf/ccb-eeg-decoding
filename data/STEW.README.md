# STEW data — manual download instructions

The STEW workload dataset (Lim, Sourina, Wang 2018, IEEE TNSRE 26(11):2106–2114; `lim2018stew`) is not vendored in this repo. It is open-access on IEEE DataPort but requires a free account to download. Follow these steps after creating an IEEE DataPort account.

## 1. Download the archive

1. Open the dataset page:
   <https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset>
2. Sign in (free IEEE DataPort account).
3. Download `STEW Dataset.zip` (~52 MB).

## 2. Extract under `data/STEW/`

From the repo root:

```bash
mkdir -p data/STEW
unzip -j ~/Downloads/STEW\ Dataset.zip -d data/STEW/
```

`-j` (junk paths) flattens any internal directory structure so the
extracted files land directly in `data/STEW/`. After extraction the
layout should be:

```
data/STEW/
├── ratings.txt
├── sub01_lo.txt
├── sub01_hi.txt
├── sub02_lo.txt
├── sub02_hi.txt
…
└── sub48_hi.txt
```

The directory is `.gitignore`-d (≈50 MB of EEG .txt files).

## 3. Validate the layout

```bash
make stew-check
```

The script (`scripts/check_stew_data.py`):

- Parses `ratings.txt` and reports the count of usable subjects (45 of 48;
  subjects 5, 24, 42 are flagged with unavailable ratings on the
  IEEE DataPort page and are silently dropped by `load_stew`).
- Spot-checks 5 random subjects: confirms both `sub{NN}_lo.txt` and
  `sub{NN}_hi.txt` exist, can be parsed as `(n_samples, 14)`, and have
  ≈19 200 rows (128 Hz × 150 s, ±5 %).
- Exits non-zero on any failure.

## 4. What the loader does with the files

See `src/thesis/data/stew_load.py` and `design-doc/ccb-formulation.md`
§2.6 (canonical operational definition for cognitive load in this thesis):

- Each subject's two 2.5-min segments (rest + multitask) are resampled to
  250 Hz (matching the 2a/2b/Cho2017 pipeline) and split into
  non-overlapping 4-s windows → 37 trials per condition per subject.
- Each window inherits its segment's subjective workload rating from
  `ratings.txt`, binned per Lim 2018: ``{1,2,3} → low``,
  ``{4,5,6} → medium``, ``{7,8,9} → high``.
- The CCB pipeline then consumes the resulting `SubjectData` exactly as
  it does for 2a/2b — same arms, same OPLB, same protocols.

## 5. License / attribution

STEW is released open-access on IEEE DataPort. Cite Lim, Sourina, Wang
2018 (DOI 10.1109/TNSRE.2018.2872924) in any publication. The BibTeX
key is `lim2018stew` in `design-doc/references.bib`.
