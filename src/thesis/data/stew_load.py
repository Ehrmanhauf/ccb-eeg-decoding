"""STEW (Lim 2018) workload dataset loader (Phase-5 §2.6 / Workstream C.3).

Custom loader; **not** MOABB-backed (STEW is not exposed by MOABB at the
time of writing).

Expected data layout under the repo root:

    data/STEW/sub{NN}_lo.txt   # rest segment       (NN in 01..48)
    data/STEW/sub{NN}_hi.txt   # multitask segment
    data/STEW/ratings.txt      # CSV "subject, rest_rating, test_rating"

Canonical source: open-access on IEEE DataPort
(https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset).
A free IEEE DataPort account is required to download; manual placement under
``data/STEW/`` is expected. The repo is **not** distributed with STEW data.

File format (verified 2026-05-12 against the IEEE DataPort page):

- ``sub{NN}_{lo,hi}.txt``: 14 columns (channel order: AF3, F7, F3, FC5, T7,
  P7, O1, O2, P8, T8, FC6, F4, F8, AF4) × 19 200 rows (128 Hz × 150 s).
  No header.
- ``ratings.txt``: one row per subject ``"subject_no, rest_rating, test_rating"``
  (1–9 scale). Subjects 5, 24, 42 are listed with unavailable ratings on
  the IEEE DataPort page and are silently dropped by this loader.

Operational definition (`design-doc/ccb-formulation.md` §2.6, ref: `lim2018stew`):

- 3-level workload classification of the 1–9 subjective rating:
  {1, 2, 3} → ``low``, {4, 5, 6} → ``medium``, {7, 8, 9} → ``high``.
- Each 2.5-min segment is split into non-overlapping ``epoch_seconds``-wide
  windows. Each window inherits its segment's bin as the classification
  label.

ref: `lim2018stew` (verified 2026-05-12 via PubMed + IEEE DataPort).
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from thesis.data.load import SubjectData

# Channel order per IEEE DataPort page (verified 2026-05-12).
STEW_CHANNELS: tuple[str, ...] = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
)

STEW_NATIVE_SFREQ: float = 128.0
STEW_SEGMENT_SECONDS: float = 150.0  # 2.5 minutes per condition

# 3-level workload bins. ref: lim2018stew + ccb-formulation.md §2.6.
STEW_BIN_EDGES: tuple[tuple[int, int, str], ...] = (
    (1, 3, "low"),
    (4, 6, "medium"),
    (7, 9, "high"),
)

# Subjects flagged on the IEEE DataPort page as having unavailable ratings.
_STEW_MISSING_RATINGS: frozenset[int] = frozenset({5, 24, 42})

# Channel-role map for the workload context and the CL band-power baseline
# (Phase B). Indices into ``STEW_CHANNELS``. STEW's 14-channel Emotiv EPOC
# layout (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4) has
# no mid-line Fz / Cz / Pz electrodes; frontal-θ is approximated over the
# eight available frontal positions, parietal-α over the two parietal
# positions, and the F3 / F4 single-channel asymmetry pair uses the
# dedicated F3 (idx 2) and F4 (idx 11) electrodes that *are* in the
# montage. Consumed by ``thesis.baselines.bandpower_cl.BandPowerCL`` and
# (in the Phase E wiring) by ``thesis.ccb.context_cl.compute_context_workload``.
STEW_CHANNEL_ROLES: dict[str, list[int]] = {
    "frontal": [0, 1, 2, 3, 10, 11, 12, 13],  # AF3, F7, F3, FC5, FC6, F4, F8, AF4
    "parietal": [5, 8],                         # P7, P8
    "f3": [2],                                  # F3
    "f4": [11],                                 # F4
}

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_STEW_ROOT = _REPO_ROOT / "data" / "STEW"


def _bin_rating(rating: int) -> str:
    """Map a 1–9 workload rating to its low/medium/high bin (ref: lim2018stew)."""
    for lo, hi, label in STEW_BIN_EDGES:
        if lo <= rating <= hi:
            return label
    raise ValueError(f"workload rating {rating!r} outside the 1..9 range")


def _load_ratings(ratings_path: Path) -> dict[int, tuple[int, int]]:
    """Parse the STEW ``ratings.txt`` file.

    Returns ``{subject_id: (rest_rating, multitask_rating)}``. Subjects in
    ``_STEW_MISSING_RATINGS`` are excluded from the returned mapping.

    The file format on IEEE DataPort is documented as CSV; this loader also
    tolerates whitespace-separated values for robustness against minor
    distribution variants.
    """
    table = pd.read_csv(
        ratings_path,
        header=None,
        names=["subject", "rating_rest", "rating_test"],
        sep=r"[,\s]+",
        skipinitialspace=True,
        engine="python",
    )
    out: dict[int, tuple[int, int]] = {}
    for _, row in table.iterrows():
        sid = int(row["subject"])
        if sid in _STEW_MISSING_RATINGS:
            continue
        out[sid] = (int(row["rating_rest"]), int(row["rating_test"]))
    return out


def _read_segment(path: Path) -> np.ndarray:
    """Read a STEW EEG ``.txt`` file into a ``(n_channels, n_samples)`` array."""
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] != len(STEW_CHANNELS):
        raise ValueError(
            f"{path}: expected (n_samples, {len(STEW_CHANNELS)}) layout, "
            f"got shape {arr.shape}"
        )
    return arr.T.astype(np.float64, copy=False)


def _segment_to_epochs(
    segment: np.ndarray,
    *,
    native_sfreq: float,
    target_sfreq: float,
    epoch_seconds: float,
) -> np.ndarray:
    """Resample + cut a continuous segment into ``(n_epochs, n_channels, n_samples)``.

    Resampling routes through MNE's polyphase resampler so the downstream
    spectral arms see the same sampling rate as the 2a/2b/Cho2017 paths.
    """
    info = mne.create_info(list(STEW_CHANNELS), native_sfreq, ch_types="eeg")
    raw = mne.io.RawArray(segment, info, verbose=False)
    if target_sfreq != native_sfreq:
        raw.resample(target_sfreq, verbose=False)
    data = raw.get_data()
    samples_per_epoch = int(round(epoch_seconds * target_sfreq))
    n_epochs = data.shape[1] // samples_per_epoch
    if n_epochs == 0:
        raise ValueError(
            f"segment too short ({data.shape[1]} samples) for "
            f"epoch_seconds={epoch_seconds} at {target_sfreq} Hz"
        )
    trimmed = data[:, : n_epochs * samples_per_epoch]
    epochs = trimmed.reshape(len(STEW_CHANNELS), n_epochs, samples_per_epoch)
    return np.transpose(epochs, (1, 0, 2))


def load_stew(
    subjects: list[int] | None = None,
    *,
    data_root: Path | None = None,
    epoch_seconds: float = 4.0,
    target_sfreq: float = 250.0,
) -> list[SubjectData]:
    """Load STEW filtered to 3-class workload classification.

    Each subject contributes both 2.5-min segments (rest + multitask),
    split into non-overlapping ``epoch_seconds``-wide windows. Every
    window inherits its segment's subjective rating, binned per
    ``STEW_BIN_EDGES``.

    Parameters
    ----------
    subjects : list of 1-indexed subject IDs (1..48), or ``None`` for all
        subjects with usable ratings (45 of 48; subjects 5, 24, 42 dropped).
    data_root : path to the STEW data directory. Defaults to ``data/STEW``
        under the repo root.
    epoch_seconds : trial window length. Default 4.0 to match 2a/2b.
    target_sfreq : sampling rate to resample to. Default 250.0 to match
        the 2a/2b/Cho2017 pipeline (STEW records natively at 128 Hz).

    Returns
    -------
    list[SubjectData] — one per loaded subject.

    Notes
    -----
    The ``metadata`` DataFrame's ``session`` column carries ``"rest"`` or
    ``"multitask"`` per window so :func:`thesis.protocols.session_split`
    can be used for a condition-leave-out protocol if desired.

    ref: `lim2018stew`, design-doc/ccb-formulation.md §2.6.
    """
    root = Path(data_root) if data_root is not None else _DEFAULT_STEW_ROOT
    ratings = _load_ratings(root / "ratings.txt")

    if subjects is None:
        wanted = sorted(ratings.keys())
    else:
        wanted = [int(s) for s in subjects if int(s) in ratings]

    out: list[SubjectData] = []
    for sid in wanted:
        rest_rating, test_rating = ratings[sid]
        rest = _read_segment(root / f"sub{sid:02d}_lo.txt")
        test = _read_segment(root / f"sub{sid:02d}_hi.txt")
        rest_epochs = _segment_to_epochs(
            rest,
            native_sfreq=STEW_NATIVE_SFREQ,
            target_sfreq=target_sfreq,
            epoch_seconds=epoch_seconds,
        )
        test_epochs = _segment_to_epochs(
            test,
            native_sfreq=STEW_NATIVE_SFREQ,
            target_sfreq=target_sfreq,
            epoch_seconds=epoch_seconds,
        )
        X = np.concatenate([rest_epochs, test_epochs], axis=0)
        y_rest = np.array([_bin_rating(rest_rating)] * rest_epochs.shape[0])
        y_test = np.array([_bin_rating(test_rating)] * test_epochs.shape[0])
        y = np.concatenate([y_rest, y_test])
        meta = pd.DataFrame({
            "subject": [sid] * X.shape[0],
            "session": (
                ["rest"] * rest_epochs.shape[0]
                + ["multitask"] * test_epochs.shape[0]
            ),
            "run": ["0"] * X.shape[0],
        })
        out.append(
            SubjectData(
                subject=sid,
                dataset_name="STEW",
                X=X,
                y=y,
                metadata=meta,
                sfreq=float(target_sfreq),
            )
        )
    return out
