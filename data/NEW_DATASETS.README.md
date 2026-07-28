# New Datasets — Phase 0 Verification Note (near-ear reframe)

> Phase 0 definition-of-done for `design-doc/near-ear-reframe-workplan.md`.
> Confirms the raw data is present, the sampling rates, channel lists, and known
> quirks — **before** any loader is written. Verified 2026-06-09 against the data
> on disk under `new_datasets/` (gitignored, ~33 GB, reproducible via the DOIs
> below). `pyarrow` added to the project for the UAB parquet path.

## Summary verdict

| Dataset | Raw time series? | Sampling | Montage | Near-ear T7/T8 | Headline role |
|---|---|---|---|---|---|
| **UAB** (n-back) | ✅ **YES** (`EEG.*` cols) | 128 Hz | EPOC X 14-ch (= STEW) | ✅ both present | CL on consumer near-ear (leak-caveated) |
| **COG-BCI** (N-back) | ✅ YES (`.set/.fdt`) | 500 Hz | 64-ch ActiCap (63 for subj 1–9) | ✅ both present | **leakage-clean CL + cross-session headline** |
| **COG-BCI competition split** | ✅ but pre-epoched | 250 Hz, 2-s epochs | 61-ch cleaned | ✅ both present | MATB leaderboard anchor |

**UAB Phase-0 gate RESOLVED:** the parquet contains the 14 raw µV channels, so UAB
supports the full B1–B5 + CCB pipeline (not just B2).

---

## 1. UAB Flight-Deck — `new_datasets/workload_dataset/`

- **DOI** 10.5565/ddd.uab.cat/259591 (CC-BY). N-back cell:
  `data_n_back_test/eeg/eeg.parquet` — **single 639 MB parquet, all subjects
  concatenated**, 15,294,488 rows × **142 columns**.
- **Raw EEG (confirmed):** columns `EEG.AF3 … EEG.AF4` (indices 3–16) = 14 raw
  channels, EPOC X order `AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8,
  AF4` — **identical to STEW** → reuse `STEW_CHANNEL_ROLES`; near-ear `T7=idx4,
  T8=idx9`. Raw values carry the usual Emotiv ~4200 µV DC offset (mean≈4211,
  σ≈25–29 µV); FBCSP's band-pass removes it.
- **Also present (not used by our pipeline):** `POW.*` (70 cols = 14ch × {Theta,
  Alpha, BetaL, BetaH, Gamma} onboard band-power), `PM.*` (engagement/excitement/
  stress/… performance metrics), `CQ.*` (contact quality), `EEG.Counter`,
  `EEG.Interpolated`, markers.
- **Structure:** `subject` ∈ {subject_01 … subject_16} (16 subjects); `test` ∈
  {1,2,3} = the three difficulty variants (**1 = position 1-back → low, 2 =
  arithmetic 1-back → med, 3 = dual 2-back → high**); `phase` ∈ {1,2,3} = baseline
  / **task** / recovery. ~973k rows/subject median ≈ 126 min/subject (~34 h total).
- **Sampling:** 128 Hz (per spec + `EEG.Counter`). Resample 128→250 Hz in the loader.
- **Leak (confirmed):** within a subject each difficulty is **one continuous task
  block** (`phase==2` of each `test`) → STEW-like segment-identity leak under naïve
  within-CV. Report within-CV with the caveat; this is **not** a headline-clean cell.
- **Loader plan:** read `EEG.*` (14 ch) → filter `phase==2` (task) → per
  `(subject, test)` epoch into 4-s windows → label = `test` mapped to low/med/high.
  `data_flight_simulator/` cross-task transfer is ~2 pilots → **excluded** from κ claims.

## 2. COG-BCI database — `new_datasets/7413650/` (Zenodo record 7413650)

- **CC-BY 4.0.** Per-subject `sub-01.zip … sub-29.zip` (29 subjects) →
  `sub-NN/ses-S{1,2,3}/{eeg,behavioral,chanlocs}/`. EEGLAB `.set/.fdt`, read with
  **MNE `read_raw_eeglab`** (verified on `sub-01/ses-S1/eeg/zeroBACK.set`).
- **EEG:** 500 Hz, ref Fpz, RAW (no acquisition filtering). Resample 500→250 Hz.
- **N-back cells (verified present, all 3 sessions, every subject):**
  `zeroBACK.set`, `oneBACK.set`, `twoBACK.set` → 3-class workload (0/1/2-back),
  **leakage-resistant** (separated recurring 48-trial blocks). Also present:
  `PVT.set` (Phase-6 vigilance), `MATB{easy,med,diff}.set`, `Flanker.set`, `RS_*.set`.
- **Quirks (confirmed on sub-01):** nominal 64-ch ActiCap, but **sub-01 has 63
  channels — Cz absent** (the documented "Cz not recorded for subjects 1–9") and
  **TP9 replaced by `ECG1`** (must be dropped — not EEG). For full-montage cells:
  intersect channels across subjects (drop Cz + ECG1 → common ~62 EEG ch) or handle
  per documented constants. **Near-ear T7/T8 is unaffected** (both present in all).
- **Channel order (sub-01, 63 ch):** `Fp1 Fz F3 F7 FT9 FC5 FC1 C3 T7 ECG1 CP5 CP1
  Pz P3 P7 O1 Oz O2 P4 P8 TP10 CP6 CP2 FCz C4 T8 FT10 FC6 FC2 F4 F8 Fp2 AF7 AF3 AFz
  F1 F5 FT7 FC3 C1 C5 TP7 CP3 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2 CPz CP4 TP8 C6 C2 FC4
  FT8 F6 AF8 AF4 F2` (T7=idx8, T8=idx25, ECG1=idx9).
- **Trial structure:** annotations carry the N-back block/trial triggers (e.g.
  codes 601*/602*/603* on zeroBACK, ~203 annotations incl. an EEGLAB `boundary`).
  Map codes via `new_datasets/7413650/triggerlist.txt` in the loader. Behaviour +
  RSME/KSS in `behavioral/*.mat`, `RSME.txt`, `KSS.txt`.
- **Cross-session:** S1/S2/S3 one week apart → **the deployment-regime headline**
  (train S1 → test S2/S3). Reference: authors' Riemannian MDM ~65 % 3-class
  within-session; cross-session competition leaderboard tops out **< 60 %** (11
  expert teams) — the field-wide difficulty anchor.
- **Extraction note:** 30 GB of per-subject zips. Loader should extract a subject's
  needed `.set/.fdt` on demand to a temp/scratch dir (or expect a pre-extracted
  tree); do **not** commit the extracted tree (it lands under the gitignored
  `new_datasets/`).

## 3. COG-BCI cross-session competition split — `new_datasets/5055046/` (Zenodo 5055046)

- **MATB-II, 2 sessions provided** (`PNN/S{1,2}/eeg/alldata_sbjNN_sessN_{MATBdiff,
  MATBeasy,MATBmed,RS,RSraw}.set`). The `.set` are **already epoched + preprocessed**:
  e.g. MATBmed = (149 epochs, **61 ch**, 500 samples) at **250 Hz** = 2-s epochs,
  cleaned to 61 EEG channels (no Cz/ECG), T7/T8 present.
- **Session 3 is the withheld competition test** (predict + score) — hence
  `estimation_results_session3.csv`, `documentation_pBCI_hackathon.pdf`, and toy code
  `Example_python.pdf` / `Example_matlab.pdf`, plus `chan_locs_standard`.
- **Role:** the **MATB leaderboard anchor / canonical-preprocessing reference**. Our
  cross-session MATB cell uses the full database (record 7413650, which *has* S3
  labels) processed through our own pipeline; the competition split documents the
  authors' epoching and is cited against the published < 60 % leaderboard. It is
  **not** the leakage-clean N-back headline cell (decision: N-back primary, MATB anchor).

---

## Reproduction / downloads

- UAB: `workload_dataset.zip` from UAB DDD, DOI 10.5565/ddd.uab.cat/259591.
- COG-BCI database: Zenodo record **7413650**.
- COG-BCI competition split: Zenodo record **5055046**.

All three are open-access; place under `new_datasets/` (gitignored). Phase 1
loaders (`emotiv_uab_load.py`, `cogbci_load.py`) consume them into `SubjectData`.
