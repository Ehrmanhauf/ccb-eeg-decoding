"""UAB Flight-Deck workload dataset loader (Hernández-Sabaté et al., 2024).

Dataset 6 of the near-ear reframe (``design-doc/near-ear-reframe-workplan.md``
§3.1). Cognitive load on the **consumer near-ear** montage: the same Emotiv
EPOC X 14-channel layout as our deployment target, so the T7/T8 near-ear subset
applies directly and the channel-roles map is shared verbatim with STEW.

Expected layout under the repo (gitignored, ~970 MB):

    new_datasets/workload_dataset/data_n_back_test/eeg/eeg.parquet

verified 2026-06-09 (see ``data/NEW_DATASETS.README.md``):

- Single parquet, all 16 subjects concatenated, 15,294,488 rows × 142 cols.
- Raw EEG = columns ``EEG.AF3 … EEG.AF4`` (14 channels, EPOC X order, **identical
  to STEW**). Also present but unused: ``POW.*`` (onboard band-power), ``PM.*``
  (performance metrics), ``CQ.*`` (contact quality).
- ``subject`` ∈ {``subject_01`` … ``subject_16``}; ``test`` ∈ {1,2,3} = the three
  graded n-back variants (1 = position 1-back → low, 2 = arithmetic 1-back → med,
  3 = dual 2-back → high); ``phase`` ∈ {1,2,3} = baseline / **task** / recovery.
- Native 128 Hz. Raw values carry the Emotiv ~4200 µV reference DC offset.

Operational definition (3-level workload, parallel to STEW §2.6):

- Per subject, each ``test`` is one continuous **task** block (``phase == 2``);
  the task EEG is split into non-overlapping ``epoch_seconds`` windows, each
  inheriting the ``test``'s difficulty bin.

**Leak caveat (state in the thesis).** Each difficulty is a single continuous
block per subject → STEW-like *segment-identity* leakage under naïve within-CV.
UAB is therefore **not** a headline-clean cell; report within-CV with the caveat,
and treat the leakage-clean COG-BCI N-back **cross-session** cell as the headline
(COG-BCI's within-session CV is itself file-identity-confounded, not clean).

**Choice — per-epoch DC removal at load.** *Choice:* centre each epoch by its own
per-channel mean (computed over that epoch's own samples only). *Alternatives:*
(a) leave the raw ~4200 µV Emotiv reference offset in place; (b) per-*block*
mean-subtraction; (c) high-pass filter. *Reason:* the offset is a known Emotiv
raw-export reference artefact, not neural signal; per-epoch centering is **strictly
train/test-independent** (no cross-trial statistic touches any other trial, so it
cannot leak), unlike per-block centering whose mean would be computed over train
and test epochs together. It is centering, not frequency-domain filtering, so it
neither informs a low-channel decision nor alters the θ/α/β band-power or the
band-pass CSP features (both already discard DC) — it only un-saturates the
context's amplitude-based artefact flag. This is a methodological-derivation
justification (``CLAUDE.md`` §§1.3, 2); the BCI-IV/STEW loaders need no such step
because their distributions are already centred.

ref: Hernández-Sabaté et al. 2024 (UAB DDD, DOI 10.5565/ddd.uab.cat/259591);
``design-doc/near-ear-reframe-workplan.md`` §3.1.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from thesis.data.load import SubjectData
from thesis.data.near_ear import select_near_ear
from thesis.data.stew_load import STEW_CHANNEL_ROLES

# EPOC X channel order — identical to STEW (verified 2026-06-09 against the
# parquet ``EEG.<ch>`` column order). T7 = index 4, T8 = index 9.
UAB_CHANNELS: tuple[str, ...] = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
)
# Parquet column names holding the raw µV time series.
UAB_RAW_EEG_COLUMNS: tuple[str, ...] = tuple(f"EEG.{c}" for c in UAB_CHANNELS)

UAB_NATIVE_SFREQ: float = 128.0
UAB_TASK_PHASE: int = 2  # 1 = baseline, 2 = task, 3 = recovery

# ``test`` value → 3-level workload bin (ref work-plan §3.1).
UAB_DIFFICULTY: dict[int, str] = {1: "low", 2: "medium", 3: "high"}

# Same montage as STEW → reuse the channel-roles map verbatim (advisor §4).
UAB_CHANNEL_ROLES: dict[str, list[int]] = STEW_CHANNEL_ROLES

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_UAB_PARQUET = (
    _REPO_ROOT / "new_datasets" / "workload_dataset" / "data_n_back_test" / "eeg" / "eeg.parquet"
)


def _subject_str_to_id(s: str) -> int:
    """``"subject_07"`` → ``7``."""
    return int(str(s).split("_")[-1])


def _block_to_epochs(
    block: np.ndarray,
    *,
    native_sfreq: float,
    target_sfreq: float,
    epoch_seconds: float,
) -> np.ndarray:
    """Resample + window one continuous ``(n_channels, n_samples)`` block.

    Removes the per-channel DC offset (Emotiv reference artefact) before
    resampling, then cuts into non-overlapping ``epoch_seconds`` windows. Mirrors
    :func:`thesis.data.stew_load._segment_to_epochs` so the CCB pipeline sees
    identical sample-rate semantics across datasets.
    """
    block = block.astype(np.float64, copy=True)
    info = mne.create_info(list(UAB_CHANNELS), native_sfreq, ch_types="eeg")
    raw = mne.io.RawArray(block, info, verbose=False)
    if target_sfreq != native_sfreq:
        raw.resample(target_sfreq, verbose=False)
    data = raw.get_data()
    samples_per_epoch = int(round(epoch_seconds * target_sfreq))
    n_epochs = data.shape[1] // samples_per_epoch
    if n_epochs == 0:
        raise ValueError(
            f"task block too short ({data.shape[1]} samples) for "
            f"epoch_seconds={epoch_seconds} at {target_sfreq} Hz"
        )
    trimmed = data[:, : n_epochs * samples_per_epoch]
    epochs = trimmed.reshape(len(UAB_CHANNELS), n_epochs, samples_per_epoch)
    epochs = np.transpose(epochs, (1, 0, 2))
    # Per-epoch DC removal: centre each epoch by its own per-channel mean, using
    # only that epoch's own samples. This is strictly train/test-independent (no
    # cross-trial statistic touches any other trial), unlike a per-block mean that
    # would be computed over train+test epochs together. Removes the Emotiv ~4200 µV
    # reference offset; FBCSP/band-power are already DC-immune, so this only affects
    # the context's amplitude-based artefact flag.
    epochs -= epochs.mean(axis=2, keepdims=True)
    return epochs


def _read_uab_parquet(path: Path, subjects: list[int] | None) -> pd.DataFrame:
    """Read only the raw-EEG + metadata columns of the task phase from the parquet."""
    import pyarrow.parquet as pq

    cols = list(UAB_RAW_EEG_COLUMNS) + ["subject", "test", "phase"]
    filters: list = [("phase", "==", UAB_TASK_PHASE)]
    if subjects is not None:
        # Push the subject filter into the parquet scan so partial loads do not
        # materialise all ~7.5M task rows.
        wanted_str = [f"subject_{int(s):02d}" for s in subjects]
        filters.append(("subject", "in", wanted_str))
    table = pq.read_table(path, columns=cols, filters=filters)
    df = table.to_pandas()
    df["subject_id"] = df["subject"].map(_subject_str_to_id)
    return df


def load_emotiv_uab(
    subjects: list[int] | None = None,
    *,
    data_path: Path | None = None,
    near_ear: bool = False,
    epoch_seconds: float = 4.0,
    target_sfreq: float = 250.0,
) -> list[SubjectData]:
    """Load UAB n-back, filtered to 3-level workload classification.

    Each subject contributes the three difficulty *task* blocks (``test`` ∈
    {1,2,3}, ``phase == 2``), each split into non-overlapping ``epoch_seconds``
    windows labelled ``low`` / ``medium`` / ``high``.

    Parameters
    ----------
    subjects : list of 1-indexed subject IDs (1..16), or ``None`` for all 16.
    data_path : path to ``eeg.parquet``. Defaults to the gitignored location
        under ``new_datasets/`` (see module docstring).
    near_ear : if ``True``, subset every subject to the T7/T8 near-ear pair at
        load time (position-based; ``thesis.data.near_ear.select_near_ear``).
        ``dataset_name`` becomes ``"UAB-nearear"``.
    epoch_seconds : trial-window length. Default 4.0 to match the rest of the
        pipeline.
    target_sfreq : resample target. Default 250.0 (UAB records natively at 128 Hz).

    Returns
    -------
    list[SubjectData] — one per loaded subject. ``metadata`` carries
    ``subject`` (1..16), ``session`` = the difficulty-block id ``str(test)``
    (parallels STEW's segment-as-session, documenting the leak structure), and
    ``run`` = ``"0"``.
    """
    path = Path(data_path) if data_path is not None else _DEFAULT_UAB_PARQUET
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Download UAB workload_dataset (DOI "
            "10.5565/ddd.uab.cat/259591) under new_datasets/ — see "
            "data/NEW_DATASETS.README.md."
        )
    df = _read_uab_parquet(path, subjects)
    if df.empty:
        return []

    out: list[SubjectData] = []
    for sid in sorted(df["subject_id"].unique()):
        sub = df[df["subject_id"] == sid]
        X_blocks: list[np.ndarray] = []
        y_blocks: list[np.ndarray] = []
        session_meta: list[str] = []
        for test in sorted(sub["test"].unique()):
            if int(test) not in UAB_DIFFICULTY:
                continue
            block_df = sub[sub["test"] == test]
            block = block_df[list(UAB_RAW_EEG_COLUMNS)].to_numpy(dtype=np.float64).T
            epochs = _block_to_epochs(
                block,
                native_sfreq=UAB_NATIVE_SFREQ,
                target_sfreq=target_sfreq,
                epoch_seconds=epoch_seconds,
            )
            label = UAB_DIFFICULTY[int(test)]
            X_blocks.append(epochs)
            y_blocks.append(np.array([label] * epochs.shape[0]))
            session_meta.extend([str(int(test))] * epochs.shape[0])

        if not X_blocks:
            continue
        X = np.concatenate(X_blocks, axis=0)
        y = np.concatenate(y_blocks)
        meta = pd.DataFrame({
            "subject": [int(sid)] * X.shape[0],
            "session": session_meta,
            "run": ["0"] * X.shape[0],
        })
        data = SubjectData(
            subject=int(sid),
            dataset_name="UAB",
            X=X,
            y=y,
            metadata=meta,
            sfreq=float(target_sfreq),
        )
        if near_ear:
            data = select_near_ear(data, UAB_CHANNELS)
        out.append(data)
    return out
