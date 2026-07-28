"""Load BCI Competition IV-2a and IV-2b directly from GDF files + the true-label release.

Neither the GDF signal files nor the true-label .mat files are distributed with this
repository — download both from the canonical sources below and place them under the
repo root in this layout:

    data/BCICIV_2a_gdf/A{01..09}{T,E}.gdf
    data/BCICIV_2b_gdf/B{01..09}{01..05}{T,E}.gdf
    data/true_labels/2a/A{01..09}{T,E}.mat
    data/true_labels/2b/B{01..09}{01..05}{T,E}.mat

Canonical sources:

    Signal (GDF):  https://www.bbci.de/competition/iv/download/ (agree-and-submit page;
                   BCICIV_2a_gdf.zip and BCICIV_2b_gdf.zip)
    Labels (.mat): https://www.bbci.de/competition/iv/results/ds{2a,2b}/true_labels.zip
                   (post-competition ground-truth release; the labels for the
                   evaluation session were withheld during the competition)

Thesis-specific constraints applied here (and nowhere else):

  - 2a (22-channel, originally 4-class) is filtered to left_hand (cue 769 / label 1)
    and right_hand (cue 770 / label 2) to match 2b. Feet (771/3) and tongue (772/4)
    trials are dropped.
  - 2b (3-channel bipolar, 5 sessions) uses **only sessions 01 and 02** — the
    screening (no-feedback) sessions. Sessions 03–05 include smiley feedback and
    therefore introduce a distributional confounder absent from 2a, which has no
    feedback at all. (ref: `project_thesis_design.md`.)

No additional digital filtering is applied here. Signals were already
bandpass-filtered 0.5–100 Hz with a 50 Hz notch at recording time per the BCI-IV
specification (desc_2a.pdf §2, desc_2b.pdf §2); FBCSP's sub-band filter bank
handles all further frequency selection downstream (ref: `ang2012fbcsp`).

Epoch window is 0–4 s post-cue, locked to the design-doc choice
(ccb-formulation.md §6.1). See open-justifications.md for the epoch-window
sensitivity-sweep plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.io as sio

CLASS_LABELS: tuple[str, str] = ("left_hand", "right_hand")

# Sampling rate both datasets were recorded at, per BCI Competition IV spec.
NATIVE_SFREQ: float = 250.0

# Repo-relative data root. __file__ is `src/thesis/data/load.py` → parents[3] is repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO_ROOT / "data"
_GDF_2A = _DATA_DIR / "BCICIV_2a_gdf"
_GDF_2B = _DATA_DIR / "BCICIV_2b_gdf"
_LABELS_2A = _DATA_DIR / "true_labels" / "2a"
_LABELS_2B = _DATA_DIR / "true_labels" / "2b"

# MI cue-onset event codes per BCI Competition IV.
#   769 = left hand       770 = right hand
#   771 = feet (2a only)  772 = tongue (2a only)
#   783 = "unknown cue" marker in evaluation sessions (labels withheld during
#         the competition; supplied here by the true_labels.mat companion).
_CUE_EVENT_CODES = frozenset({"769", "770", "771", "772", "783"})

# 2-class filter applied to the `classlabel` integers (1..4). Keep left-hand (1)
# and right-hand (2). (Convention: label value == cue code − 768.)
_TWOCLASS_LABEL_TO_NAME: dict[int, str] = {1: "left_hand", 2: "right_hand"}

# Full 4-class 2a label map (left/right hand, feet, tongue). Used for the
# benchmark-faithful 4-class 2a evaluation that matches the published 2a numbers
# (Ang 2012 FBCSP κ = 0.569; Lawhern 2018 EEGNet κ = 0.70). 2b has only
# left/right hand, so this map applies to 2a alone.
_FOURCLASS_LABEL_TO_NAME: dict[int, str] = {
    1: "left_hand", 2: "right_hand", 3: "feet", 4: "tongue",
}

# Epoch window relative to cue onset (seconds).
#   ref: desc_2a.pdf §"Experimental paradigm": cue at t=2 s, MI 2–6 s ⇒ 4 s post-cue.
#   ref: desc_2b.pdf §"Experimental paradigm": cue at t=3 s, MI 3–7.5 s ⇒ ≥4 s post-cue.
# We use 4 s for both, locked by ccb-formulation.md §6.1. See
# open-justifications.md for the sensitivity-sweep plan.
_EPOCH_TMIN = 0.0
_EPOCH_TMAX = 4.0


@dataclass(frozen=True)
class SubjectData:
    """All 2-class MI trials for a single subject from a single dataset slice."""

    subject: int
    dataset_name: str  # "BCICIV-2a" or "BCICIV-2b-screening"
    X: np.ndarray  # (n_trials, n_channels, n_samples) float64, MNE native (volts)
    y: np.ndarray  # (n_trials,) string array from CLASS_LABELS
    metadata: pd.DataFrame  # columns: subject, session, run
    sfreq: float

    @property
    def n_trials(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[2])

    @property
    def class_balance(self) -> dict[str, int]:
        labels, counts = np.unique(self.y, return_counts=True)
        return {str(label): int(count) for label, count in zip(labels, counts, strict=True)}

    @property
    def sessions(self) -> list[str]:
        return sorted(self.metadata["session"].unique().tolist())


def load_bci2a(subjects: list[int] | None = None, *, n_classes: int = 2) -> list[SubjectData]:
    """Load BCI Competition IV-2a (22ch benchmark) motor imagery (2- or 4-class).

    Each subject's two sessions — ``A##T.gdf`` ("training", session 0) and
    ``A##E.gdf`` ("evaluation", session 1) — are concatenated into a single
    :class:`SubjectData`. Session strings ``"0"`` and ``"1"`` are stored in
    the metadata so :func:`thesis.protocols.session_split` can separate them
    for the official BCI-IV protocol.

    Parameters
    ----------
    subjects:
        List of 1-indexed subject IDs (1..9). Defaults to all 9.
    n_classes:
        ``2`` (default) keeps the left/right-hand subset (matches 2b); ``4`` keeps
        the full left-hand/right-hand/feet/tongue set for the benchmark-faithful
        comparison against the published 4-class 2a numbers (Ang 2012, Lawhern 2018).
    """
    if n_classes not in (2, 4):
        raise ValueError(f"n_classes must be 2 or 4; got {n_classes}")
    label_map = _TWOCLASS_LABEL_TO_NAME if n_classes == 2 else _FOURCLASS_LABEL_TO_NAME
    ids = list(range(1, 10)) if subjects is None else list(subjects)
    result: list[SubjectData] = []
    for subject in ids:
        X_T, y_T = _load_gdf_with_labels(
            gdf=_GDF_2A / f"A{subject:02d}T.gdf",
            labels=_LABELS_2A / f"A{subject:02d}T.mat",
            n_eeg=22, label_map=label_map,
        )
        X_E, y_E = _load_gdf_with_labels(
            gdf=_GDF_2A / f"A{subject:02d}E.gdf",
            labels=_LABELS_2A / f"A{subject:02d}E.mat",
            n_eeg=22, label_map=label_map,
        )
        X = np.concatenate([X_T, X_E], axis=0)
        y = np.concatenate([y_T, y_E], axis=0)
        metadata = pd.DataFrame(
            {
                "subject": [subject] * len(y),
                "session": ["0"] * len(y_T) + ["1"] * len(y_E),
                "run": ["0"] * len(y),
            }
        )
        result.append(
            SubjectData(
                subject=subject,
                dataset_name="BCICIV-2a",
                X=X,
                y=y,
                metadata=metadata,
                sfreq=NATIVE_SFREQ,
            )
        )
    return result


def load_bci2b_screening(subjects: list[int] | None = None) -> list[SubjectData]:
    """Load BCI Competition IV-2b *screening* sessions only (3ch CCB working data).

    The dataset has 5 sessions per subject; sessions 01 and 02 are screening
    (no feedback), while sessions 03–05 involve smiley feedback. This loader
    uses only 01 and 02, matching the thesis no-feedback condition of 2a.
    Session strings are ``"0"`` (B##01T.gdf) and ``"1"`` (B##02T.gdf).

    Parameters
    ----------
    subjects:
        List of 1-indexed subject IDs (1..9). Defaults to all 9.
    """
    ids = list(range(1, 10)) if subjects is None else list(subjects)
    result: list[SubjectData] = []
    for subject in ids:
        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        sess_col: list[str] = []
        for sess_idx, sess_num in enumerate(("01", "02")):
            Xi, yi = _load_gdf_with_labels(
                gdf=_GDF_2B / f"B{subject:02d}{sess_num}T.gdf",
                labels=_LABELS_2B / f"B{subject:02d}{sess_num}T.mat",
                n_eeg=3,
            )
            X_parts.append(Xi)
            y_parts.append(yi)
            sess_col.extend([str(sess_idx)] * len(yi))
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        metadata = pd.DataFrame(
            {
                "subject": [subject] * len(y),
                "session": sess_col,
                "run": ["0"] * len(y),
            }
        )
        result.append(
            SubjectData(
                subject=subject,
                dataset_name="BCICIV-2b-screening",
                X=X,
                y=y,
                metadata=metadata,
                sfreq=NATIVE_SFREQ,
            )
        )
    return result


def _load_gdf_with_labels(
    *, gdf: Path, labels: Path, n_eeg: int, label_map: dict[int, str] = _TWOCLASS_LABEL_TO_NAME
) -> tuple[np.ndarray, np.ndarray]:
    """Read one GDF file + its true-labels .mat and return (X, y).

    Returns
    -------
    X: (n_trials_kept, n_eeg, n_samples) float64 — epochs around cue onset.
    y: (n_trials_kept,) ndarray of strings from :data:`CLASS_LABELS`.
    """
    if not gdf.exists():
        raise FileNotFoundError(
            f"Missing GDF file: {gdf}. Download BCICIV_{gdf.parent.name}.zip from "
            "https://www.bbci.de/competition/iv/download/ and extract into data/."
        )
    if not labels.exists():
        raise FileNotFoundError(
            f"Missing true-labels .mat: {labels}. Download "
            "https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip (and ds2b) "
            "and extract into data/true_labels/{2a,2b}/. "
            "See design-doc/ccb-formulation.md §2 for provenance."
        )

    # 1. Load GDF. MNE warns about the GDF header using 'EEG' for every channel
    #    name (it auto-renames duplicates with running numbers) and about
    #    ambiguous filter-cutoff metadata — both benign for BCI-IV.
    raw = mne.io.read_raw_gdf(str(gdf), preload=True, verbose="ERROR")

    # 2. Keep the first n_eeg channels; the BCI-IV channel order is EEG first,
    #    then EOG. ref: desc_2a.pdf §2 (22 EEG + 3 EOG), desc_2b.pdf §2 (3
    #    bipolar EEG + 3 EOG).
    raw.pick(list(raw.ch_names[:n_eeg]))

    # 3. Extract cue-onset events. For T files, event codes carry the class
    #    (769/770/771/772) directly; for E files only 783 ("unknown cue")
    #    appears and the true class comes from the companion .mat file.
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    cue_code_ids = {event_id[k] for k in _CUE_EVENT_CODES if k in event_id}
    cue_events = events[np.isin(events[:, 2], list(cue_code_ids))]

    # 4. Load canonical labels and sanity-check the counts line up. If they
    #    don't, the GDF and .mat are from different releases — hard fail rather
    #    than silently produce garbage labels.
    label_mat = sio.loadmat(str(labels))
    all_labels = label_mat["classlabel"].ravel().astype(int)
    if len(cue_events) != len(all_labels):
        raise RuntimeError(
            f"{gdf.name}: cue-event count {len(cue_events)} != label count "
            f"{len(all_labels)} from {labels.name}. GDF and true_labels must "
            "come from the same BCI-IV release."
        )

    # 5. Filter to the requested class subset. Default is the 2-class {left, right
    #    hand} map; the 4-class map (2a only) additionally keeps feet/tongue for the
    #    benchmark-faithful comparison against the published 4-class 2a numbers.
    keep_mask = np.isin(all_labels, list(label_map))
    kept_events = cue_events[keep_mask]
    kept_labels = np.array([label_map[int(lbl)] for lbl in all_labels[keep_mask]])

    # 6. Epoch around the cue at [_EPOCH_TMIN, _EPOCH_TMAX] s.
    epochs = mne.Epochs(
        raw,
        kept_events,
        tmin=_EPOCH_TMIN,
        tmax=_EPOCH_TMAX,
        baseline=None,
        picks="all",
        preload=True,
        verbose="ERROR",
    )
    X = epochs.get_data(copy=False)
    return X, kept_labels
