"""WAUC (Albuquerque et al. 2020) cognitive-workload dataset loader.

Custom loader for the secondary CL paradigm dataset adopted in
Research-wave 1 (2026-05-19); spec locked in
``design-doc/ccb-formulation.md`` §2.7. Not MOABB-backed (WAUC is not
exposed by MOABB at the time of writing).

Expected data layout under the repo root (after extracting
``process.rar`` from the MuSAE Lab release into ``data/WAUC/process/``):

    data/WAUC/
    ├── process/
    │   ├── S01/
    │   │   ├── enobio_eeg_asr.csv   # ASR-processed 8-channel EEG, 500 Hz
    │   │   ├── bh3_br.csv           # BioHarness 3 breathing rate
    │   │   ├── bh3_ecg.csv          # BioHarness 3 ECG
    │   │   └── bh3_rr.csv           # BioHarness 3 RR intervals
    │   ├── S02/
    │   …
    │   └── S48/
    ├── subjective_ratings_with_labels.csv
    └── demographics.csv

Subject IDs on the filesystem are ``S{NN:02d}`` for ``NN ∈ 1..48``;
in ``subjective_ratings_with_labels.csv`` and ``demographics.csv`` they
are encoded as ``Participant ID = 1000 + NN`` (so ``S01`` ↔ ``1001``,
…, ``S48`` ↔ ``1048``).

The CCB classifier path uses only ``enobio_eeg_asr.csv``; the
BioHarness-3 streams remain accessible for sanity-check correlation
analyses outside the CCB loop.

Canonical sources:

- Paper: Albuquerque et al. 2020, *Front. Neurosci.* 14:549524, DOI
  10.3389/fnins.2020.549524 (ref: ``albuquerque2020wauc``).
- Data release: https://github.com/MuSAELab/WAUC-A-Multi-Modal-Database-for-Mental-Workload-Assessment-Under-Physical-Activity
  (raw / process / features archives + 2 top-level CSVs hosted by
  the MuSAE Lab; see ``data/WAUC.README.md`` for the access procedure).

File format (verified 2026-05-19 against the extracted ``process``
archive, n = 48 subjects):

- ``enobio_eeg_asr.csv``: 8 EEG channel columns followed by 3
  metadata columns. Channel order on-disk (NOT the order in the
  Albuquerque 2020 §Materials and Methods text): ``AF8, Fp2, Fp1,
  AF7, T10, T9, P4, P3``. Note ``Fp1, Fp2`` use lower-case ``p``.
  Metadata columns: ``fs, info, session_no``. The ``info`` field is
  one of ``baseline-1`` (eyes closed + still), ``baseline-2``
  (movement only), or ``session``; ``session_no`` ∈ {1..6} indexes
  the 6 condition cells (2 mental-workload levels × 3 physical-
  exertion levels). Native sampling 500 Hz.

- ``subjective_ratings_with_labels.csv``: per-(participant, session)
  ground-truth labels and NASA-TLX subscale ratings. Columns:
  ``Participant ID, Mental Demand, Physical Demand, Temporal Demand,
  Performance, Effort, Frustration, Perceived Exertion (1),
  Perceived Exertion (2), mw_labels, pw_labels, session_no``.
  ``mw_labels`` is binary 0.0/1.0 (mental-workload low/high) and is
  this loader's classification target. ``pw_labels`` is the ternary
  physical-workload condition (0.0 / 1.0 / 2.0) and is carried in
  the metadata as a covariate.

- ``demographics.csv``: age, height, weight, sex, activity (Treadmill
  or Bike) per subject. Not consumed by ``load_wauc`` but referenced
  by ``scripts/check_wauc_data.py``.

Data-integrity notes (verified 2026-05-19 against the ratings CSV):

- Subject ``1028`` is **not** in the ratings CSV (no usable labels);
  the loader silently drops it. (The MuSAE Lab GitHub README also
  states this.)
- Subject ``1020`` *is* in the ratings CSV (6 session rows), and the
  filesystem folder ``S20`` exists with an ``enobio_eeg_asr.csv``;
  the loader keeps it. (The GitHub README's claim that 1020 has "no
  data" appears to refer to the BioHarness / Empatica streams in the
  raw archive, not to the EEG path in the processed archive.)

Operational definition (``design-doc/ccb-formulation.md`` §2.7, ref:
``albuquerque2020wauc``):

- Binary low / high mental-workload classification per session, using
  the dataset's own ``mw_labels`` column rather than a hard-coded
  ``session_no → label`` map.
- Each session's EEG is split into non-overlapping ``epoch_seconds``-
  wide windows. Each window inherits the session's MW label.
- The physical-workload condition (``pw_labels``) is treated as a
  covariate — it travels in ``SubjectData.metadata`` so downstream
  analyses can stratify by physical-exertion level, but is not the
  classifier's target.

ref: ``albuquerque2020wauc``, design-doc/ccb-formulation.md §2.7.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from thesis.data.load import SubjectData

# Channel order as the EEG CSV stores them (verified 2026-05-19 against
# data/WAUC/process/S01/enobio_eeg_asr.csv header).
WAUC_EEG_CHANNELS: tuple[str, ...] = (
    "AF8", "Fp2", "Fp1", "AF7", "T10", "T9", "P4", "P3",
)

WAUC_NATIVE_SFREQ: float = 500.0
WAUC_SESSIONS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

# Metadata columns appended after the channel columns in each
# ``enobio_eeg_asr.csv``. Verified 2026-05-19.
_WAUC_METADATA_COLUMNS: tuple[str, ...] = ("fs", "info", "session_no")

# Allowed values of the ``info`` field.
_WAUC_SESSION_INFO_VALUES: tuple[str, ...] = ("session",)
_WAUC_BASELINE_INFO_VALUES: tuple[str, ...] = ("baseline-1", "baseline-2")

# Subjects flagged in the MuSAE Lab release notes (and verified
# against the ratings CSV on 2026-05-19) as missing usable data:
#
# - 1020: GitHub README notes "no data" but the ratings CSV does
#   contain 6 session rows for 1020 and the filesystem folder S20
#   contains ``enobio_eeg_asr.csv``. The "no data" comment most
#   plausibly refers to the BioHarness/Empatica streams in the raw
#   archive; for the EEG-only CCB path, S20 is usable. We keep it.
# - 1028: NOT in the ratings CSV (zero rows). The folder S28 exists
#   but without a label we cannot train; the loader drops it.
_WAUC_MISSING_RATINGS: frozenset[int] = frozenset({1028})

# Filesystem subject IDs whose ``enobio_eeg_asr.csv`` is missing one or
# more of the 8 expected EEG channels. Verified 2026-05-19 across all
# 48 folders:
#
# - S23, S26: each missing the ``P4`` column (7 channels instead of 8).
#
# These subjects are silently dropped at load time rather than imputed,
# on the grounds that adding a zeroed-out P4 column would create a
# channel that the classifier might learn to *use* (e.g., as a "missing-
# channel flag"), which is harder to defend in a methodological
# evaluation than simply restricting the analysis to subjects with
# complete recordings.
_WAUC_MISSING_CHANNELS: frozenset[int] = frozenset({23, 26})

# Map from MW label values (as they appear in the ratings CSV) to the
# canonical lower-case strings used elsewhere in the pipeline.
_WAUC_MW_LABEL_TO_STR: dict[int, str] = {0: "low", 1: "high"}

# Channel-role map for the workload context (used by
# ``thesis.ccb.context_cl.compute_context_workload`` and by the
# CL band-power baseline in Phase B). WAUC has no dedicated F3/F4
# electrodes — we approximate left / right frontal-alpha asymmetry
# with AF7 / AF8, the leftmost / rightmost frontal positions in the
# 8-channel Enobio layout.
WAUC_CHANNEL_ROLES: dict[str, list[int]] = {
    "frontal": [
        WAUC_EEG_CHANNELS.index("AF8"),
        WAUC_EEG_CHANNELS.index("Fp2"),
        WAUC_EEG_CHANNELS.index("Fp1"),
        WAUC_EEG_CHANNELS.index("AF7"),
    ],
    "parietal": [
        WAUC_EEG_CHANNELS.index("P4"),
        WAUC_EEG_CHANNELS.index("P3"),
    ],
    "f3": [WAUC_EEG_CHANNELS.index("AF7")],
    "f4": [WAUC_EEG_CHANNELS.index("AF8")],
}

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_WAUC_ROOT = _REPO_ROOT / "data" / "WAUC"


def subject_id_to_partid(sid: int) -> int:
    """Map filesystem subject ID (1..48) to the ratings-CSV ``Participant ID``.

    On the filesystem, subjects are folders ``S01`` … ``S48``; in
    ``subjective_ratings_with_labels.csv`` and ``demographics.csv``
    they are referenced as ``1001`` … ``1048``. The mapping is
    ``partid = 1000 + sid``.
    """
    if not 1 <= sid <= 48:
        raise ValueError(f"WAUC subject id must be in 1..48, got {sid}")
    return 1000 + sid


def _subject_eeg_path(sid: int, root: Path) -> Path:
    """Return the path to ``S{sid:02d}/enobio_eeg_asr.csv`` under ``root``."""
    return root / "process" / f"S{sid:02d}" / "enobio_eeg_asr.csv"


def _resolve_channel_columns(df: pd.DataFrame) -> list[str]:
    """Return the EEG channel column names present in ``df``.

    Verified against the extracted release on 2026-05-19: the eight
    columns of ``WAUC_EEG_CHANNELS`` appear as bare names. This helper
    additionally tolerates a hypothetical ``eeg_<channel>`` variant in
    case a future release changes the convention.
    """
    cols = list(df.columns)
    if all(ch in cols for ch in WAUC_EEG_CHANNELS):
        return list(WAUC_EEG_CHANNELS)
    lower_to_orig = {c.lower(): c for c in cols}
    resolved: list[str] = []
    for ch in WAUC_EEG_CHANNELS:
        key = f"eeg_{ch}".lower()
        if key in lower_to_orig:
            resolved.append(lower_to_orig[key])
    if len(resolved) == len(WAUC_EEG_CHANNELS):
        return resolved
    raise ValueError(
        "could not resolve WAUC EEG channel columns. Expected one of "
        f"(i) bare names {WAUC_EEG_CHANNELS} or (ii) eeg_-prefixed names. "
        f"Got columns: {cols}. Inspect a sample enobio_eeg_asr.csv via "
        "scripts/check_wauc_data.py and update _resolve_channel_columns "
        "or WAUC_EEG_CHANNELS as needed."
    )


def _load_wauc_labels(labels_path: Path) -> pd.DataFrame:
    """Parse ``subjective_ratings_with_labels.csv`` into a labels frame.

    Returns a DataFrame indexed by ``(partid, session_no)`` with at
    minimum:

    - ``mw_label`` (str) — ``"low"`` or ``"high"`` (canonicalised from
      the numeric ``mw_labels`` column).
    - ``pw_label`` (int) — physical-workload condition 0 / 1 / 2
      (carried as a covariate).

    The original NASA-TLX subscale columns are preserved verbatim.
    """
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} missing. Did you complete the WAUC download "
            "and unrar process.rar under data/WAUC/? See data/WAUC.README.md."
        )
    df = pd.read_csv(labels_path)
    required = {"Participant ID", "session_no", "mw_labels"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{labels_path}: missing required columns {sorted(missing)}. "
            f"Got columns: {list(df.columns)}"
        )

    # Canonicalise mw_labels (numeric float 0.0/1.0) → str low/high.
    mw_int = df["mw_labels"].astype(float).round().astype(int)
    bad = mw_int[~mw_int.isin(_WAUC_MW_LABEL_TO_STR.keys())]
    if len(bad) > 0:
        raise ValueError(
            f"{labels_path}: unexpected mw_labels values {sorted(set(bad))!r}; "
            f"expected {sorted(_WAUC_MW_LABEL_TO_STR.keys())}."
        )
    df = df.copy()
    df["mw_label"] = mw_int.map(_WAUC_MW_LABEL_TO_STR)

    # Canonicalise pw_labels if present (kept as int covariate).
    if "pw_labels" in df.columns:
        df["pw_label"] = df["pw_labels"].astype(float).round().astype(int)
    else:
        df["pw_label"] = -1  # missing → -1 sentinel

    df = df.rename(columns={"Participant ID": "partid"})
    return df.set_index(["partid", "session_no"]).sort_index()


def _read_eeg_csv(path: Path) -> pd.DataFrame:
    """Read an ``enobio_eeg_asr.csv``, validating the metadata columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Did you extract data/WAUC/process.rar?"
        )
    df = pd.read_csv(path)
    missing = set(_WAUC_METADATA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing required metadata columns {sorted(missing)}. "
            f"Got columns: {list(df.columns)}"
        )
    return df


def _session_to_epochs(
    session_df: pd.DataFrame,
    *,
    channel_columns: list[str],
    native_sfreq: float,
    target_sfreq: float,
    epoch_seconds: float,
) -> np.ndarray:
    """Resample + window a session's EEG into ``(n_epochs, n_channels, n_samples)``.

    Mirrors :func:`thesis.data.stew_load._segment_to_epochs` so the
    CCB pipeline sees identical sample-rate semantics on WAUC and STEW.

    **NaN handling.** ASR-processed EEG can leave windows that the
    component-subspace reconstruction could not recover as NaN-marked
    samples (Mullen et al. 2015 [mullen2015asr] §III.C). This helper
    drops any epoch that contains *any* NaN sample on *any* channel
    --- a strict per-trial quality-control step. Per-subject NaN
    impact is heterogeneous on the WAUC processed archive (verified
    2026-05-19 across S01/S02/S03/S05/S10: 0–35 % of epochs); the
    drop is silent (the loader returns only the surviving epochs).
    Methodologically: dropping at the trial granularity preserves
    every clean trial while removing only the corrupted ones, and
    keeps the labelling structure intact (each surviving epoch still
    belongs to a single session_no and therefore inherits a single
    MW label).
    """
    eeg = session_df[channel_columns].to_numpy(dtype=np.float64, copy=False).T
    info = mne.create_info(channel_columns, native_sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg, info, verbose=False)
    if target_sfreq != native_sfreq:
        raw.resample(target_sfreq, verbose=False)
    data = raw.get_data()
    samples_per_epoch = int(round(epoch_seconds * target_sfreq))
    n_epochs = data.shape[1] // samples_per_epoch
    if n_epochs == 0:
        raise ValueError(
            f"session segment too short ({data.shape[1]} samples) for "
            f"epoch_seconds={epoch_seconds} at {target_sfreq} Hz"
        )
    trimmed = data[:, : n_epochs * samples_per_epoch]
    epochs = trimmed.reshape(len(channel_columns), n_epochs, samples_per_epoch)
    epochs = np.transpose(epochs, (1, 0, 2))
    # Drop NaN-contaminated epochs.
    nan_mask = np.isnan(epochs).any(axis=(1, 2))
    return epochs[~nan_mask]


def load_wauc(
    subjects: list[int] | None = None,
    *,
    data_root: Path | None = None,
    epoch_seconds: float = 4.0,
    target_sfreq: float = 250.0,
    include_baselines: bool = False,
) -> list[SubjectData]:
    """Load WAUC, filtered to binary low / high mental-workload classification.

    Each subject contributes the six session segments (one per
    MW × physical-exertion cell). Each session is resampled to
    ``target_sfreq`` and split into non-overlapping
    ``epoch_seconds``-wide windows. Windows inherit the per-session
    binary MW label from ``subjective_ratings_with_labels.csv``.

    Parameters
    ----------
    subjects : list of WAUC filesystem subject IDs (1..48), or
        ``None`` for all subjects with usable labels (1028 dropped —
        no ratings row; 1020 *is* kept because EEG + labels exist for
        it in the processed archive, despite a more general "no data"
        flag in the upstream README).
    data_root : path to the WAUC data directory. Defaults to
        ``data/WAUC`` under the repo root.
    epoch_seconds : trial-window length. Default 4.0 to match
        2a/2b/Cho2017/STEW.
    target_sfreq : sampling rate to resample to. Default 250.0 to
        match the rest of the pipeline (WAUC records natively at 500 Hz).
    include_baselines : whether to also load the two baseline
        conditions (``baseline-1`` eyes-closed/still, ``baseline-2``
        movement-only) as additional unlabeled trials carried in
        ``metadata``. **Default ``False``**: baselines are not part
        of the binary MW target and so are excluded from training /
        test data by default.

    Returns
    -------
    list[SubjectData] — one per loaded subject.

    Notes
    -----
    ``SubjectData.metadata`` carries:

    - ``subject`` : the filesystem subject ID (1..48).
    - ``session`` : ``session_no`` (1..6) as a string.
    - ``run``     : ``pw_label`` (0 / 1 / 2) as a string — the
      physical-workload condition covariate.

    ref: ``albuquerque2020wauc``, design-doc/ccb-formulation.md §2.7.
    """
    root = Path(data_root) if data_root is not None else _DEFAULT_WAUC_ROOT
    labels = _load_wauc_labels(root / "subjective_ratings_with_labels.csv")

    # Subjects available for loading: filesystem 1..48 minus drop list,
    # further filtered to those that appear in the ratings.
    partids_in_ratings = {int(p) for p in labels.index.get_level_values("partid").unique()}
    all_filesystem_subjects = list(range(1, 49))
    available_subjects = sorted(
        sid for sid in all_filesystem_subjects
        if subject_id_to_partid(sid) in partids_in_ratings
        and subject_id_to_partid(sid) not in _WAUC_MISSING_RATINGS
        and sid not in _WAUC_MISSING_CHANNELS
    )
    if subjects is None:
        wanted = available_subjects
    else:
        wanted = [int(s) for s in subjects if int(s) in available_subjects]

    out: list[SubjectData] = []
    for sid in wanted:
        partid = subject_id_to_partid(sid)
        eeg_df = _read_eeg_csv(_subject_eeg_path(sid, root))
        channel_columns = _resolve_channel_columns(eeg_df)

        # Subset to session rows (and baselines if requested).
        accepted_info = set(_WAUC_SESSION_INFO_VALUES)
        if include_baselines:
            accepted_info |= set(_WAUC_BASELINE_INFO_VALUES)
        subset = eeg_df[eeg_df["info"].isin(accepted_info)].copy()

        X_blocks: list[np.ndarray] = []
        y_blocks: list[np.ndarray] = []
        session_meta: list[str] = []
        run_meta: list[str] = []

        for session_no in WAUC_SESSIONS:
            if (partid, session_no) not in labels.index:
                continue
            session_chunk = subset[subset["session_no"] == session_no]
            if session_chunk.empty:
                continue
            row = labels.loc[(partid, session_no)]
            mw_label = str(row["mw_label"])
            pw_label = int(row["pw_label"])
            epochs = _session_to_epochs(
                session_chunk,
                channel_columns=channel_columns,
                native_sfreq=WAUC_NATIVE_SFREQ,
                target_sfreq=target_sfreq,
                epoch_seconds=epoch_seconds,
            )
            X_blocks.append(epochs)
            y_blocks.append(np.array([mw_label] * epochs.shape[0]))
            session_meta.extend([str(session_no)] * epochs.shape[0])
            run_meta.extend([str(pw_label)] * epochs.shape[0])

        if not X_blocks:
            continue

        X = np.concatenate(X_blocks, axis=0)
        y = np.concatenate(y_blocks)
        meta = pd.DataFrame({
            "subject": [sid] * X.shape[0],
            "session": session_meta,
            "run": run_meta,
        })
        out.append(
            SubjectData(
                subject=sid,
                dataset_name="WAUC",
                X=X,
                y=y,
                metadata=meta,
                sfreq=float(target_sfreq),
            )
        )
    return out
