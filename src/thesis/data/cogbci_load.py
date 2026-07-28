"""COG-BCI cross-session workload dataset loader (Hinss, Roy et al., 2022).

Dataset 7 of the near-ear reframe (``design-doc/near-ear-reframe-workplan.md``
§3.2). The **headline** source: cognitive load (N-back = 0/1/2-back) carrying a
genuine *cross-session drift* axis (3 sessions one week apart, n=29). The
leakage-clean cell here is the **cross-session** split (train S1 → test S2/S3);
within-session within-subject CV is *not* leakage-clean — see the operational
definition below.

Expected layout under the repo (gitignored, ~30 GB; Zenodo record 7413650):

    new_datasets/7413650/sub-NN.zip
        └─ sub-NN/ses-S{1,2,3}/eeg/{zero,one,two}BACK.set + .fdt   (+ PVT, MATB, …)

verified 2026-06-09 (see ``data/NEW_DATASETS.README.md``):

- 64-ch ActiCap nominal, 500 Hz, ref Fpz, **RAW** (no acquisition filtering).
  Read with MNE ``read_raw_eeglab``.
- **Quirks:** ``Cz`` is **not recorded for subjects 1–9**; ``TP9`` is **replaced
  by ``ECG1``** (a non-EEG channel, dropped here). The loader therefore reduces
  every subject to a fixed **canonical 62-channel EEG montage** (the channels
  common to all subjects — sub-01's 63 minus ``ECG1``, which already lacks Cz),
  so index-based channel-roles are consistent across subjects. T7/T8 are present
  in the canonical montage → the near-ear subset is always available.
- N-back trial structure lives in the annotations; the EEGLAB ``boundary``
  markers split each recording into contiguous segments, epoched independently.

Operational definition (3-class workload):

- Per (subject, session), ``zeroBACK`` / ``oneBACK`` / ``twoBACK`` are loaded,
  resampled 500→250 Hz, split at ``boundary`` annotations, and cut into
  non-overlapping ``epoch_seconds`` windows. Each window's label is the N-back
  level (``0back`` / ``1back`` / ``2back``). **Within-session within-subject CV
  is leakage-confounded, not clean:** the three N-back levels are recorded as
  *separate files*, so the label aligns perfectly with file/recording identity
  and a random K-fold scores by file identity rather than workload (full-montage
  κ ≈ 0.99 vs the authors' ≈ 0.65). The separated recurring blocks only rule out
  a single-contiguous-segment shortcut, not the file-identity confound. The
  genuinely leakage-clean cell is the **cross-session** split (below); the
  within-session number is retained only as a marked leak-confounded reference.

**Cross-session axis.** ``SubjectData.metadata['session']`` carries ``S1`` /
``S2`` / ``S3`` so a session split (train S1 → test S2/S3, Phase 4) isolates
drift. The within-session cells (Phase 2/3) restrict to a single session.

ref: Hinss, Roy et al. 2022 (COG-BCI, Zenodo 7413650);
``design-doc/near-ear-reframe-workplan.md`` §3.2.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from thesis.data.load import SubjectData
from thesis.data.near_ear import select_near_ear

COGBCI_NATIVE_SFREQ: float = 500.0
COGBCI_SESSIONS: tuple[str, ...] = ("S1", "S2", "S3")
COGBCI_N_SUBJECTS: int = 29

# .set stem → N-back label (3-class workload target; leakage-clean only under
# the cross-session split — within-session CV is file-identity-confounded).
COGBCI_NBACK_FILES: dict[str, str] = {
    "zeroBACK": "0back",
    "oneBACK": "1back",
    "twoBACK": "2back",
}
# Secondary near-ear targets (Phase 6). PVT = vigilance/fatigue.
COGBCI_PVT_FILE: str = "PVT"

# Non-EEG channels to drop (TP9 was replaced by ECG on this montage).
COGBCI_NON_EEG: frozenset[str] = frozenset({"ECG1", "ECG", "ECG2", "EKG"})

# Canonical 62-channel EEG montage: sub-01's channel order minus ``ECG1`` (sub-01
# already lacks Cz, which is also absent for subjects 1–9, so this set is the one
# common to all subjects). Verified 2026-06-09 against sub-01/ses-S1/eeg. Every
# subject is reduced + reordered to this list at load time → index-based roles
# below are consistent across subjects. T7 and T8 are present.
COGBCI_CANONICAL_EEG: tuple[str, ...] = (
    "Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "Pz",
    "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "TP10", "CP6", "CP2", "FCz", "C4",
    "T8", "FT10", "FC6", "FC2", "F4", "F8", "Fp2", "AF7", "AF3", "AFz", "F1", "F5",
    "FT7", "FC3", "C1", "C5", "TP7", "CP3", "P1", "P5", "PO7", "PO3", "POz", "PO4",
    "PO8", "P6", "P2", "CPz", "CP4", "TP8", "C6", "C2", "FC4", "FT8", "F6", "AF8",
    "AF4", "F2",
)

# Channel-roles for the workload context / band-power baseline, by name then
# resolved to canonical indices. Standard frontal + parietal clusters (matching
# the context's "Fp1/Fp2/AF3/AF4" frontal-θ and "P3/P4/Pz" parietal-α intent);
# F3/F4 drive the frontal-α asymmetry.
_COGBCI_ROLE_NAMES: dict[str, list[str]] = {
    "frontal": ["Fp1", "Fp2", "AF3", "AF4", "Fz", "F3", "F4"],
    "parietal": ["P3", "P4", "Pz"],
    "f3": ["F3"],
    "f4": ["F4"],
}
COGBCI_CHANNEL_ROLES: dict[str, list[int]] = {
    role: [COGBCI_CANONICAL_EEG.index(ch) for ch in names]
    for role, names in _COGBCI_ROLE_NAMES.items()
}

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_COGBCI_ROOT = _REPO_ROOT / "new_datasets" / "7413650"


def _pick_canonical(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Reduce + reorder a raw recording to the canonical 62-ch EEG montage.

    Drops non-EEG channels (ECG) and any extra channel (e.g. Cz on subjects
    10–29) not in :data:`COGBCI_CANONICAL_EEG`, and reorders to that exact list.
    Raises if a canonical channel is missing.
    """
    present = set(raw.ch_names)
    missing = [ch for ch in COGBCI_CANONICAL_EEG if ch not in present]
    if missing:
        raise ValueError(
            f"COG-BCI recording missing canonical channels {missing}; "
            f"has {sorted(present)}"
        )
    return raw.copy().pick(list(COGBCI_CANONICAL_EEG))


def _raw_to_epochs(
    raw: mne.io.BaseRaw,
    *,
    target_sfreq: float,
    epoch_seconds: float,
) -> np.ndarray:
    """Resample, split at ``boundary`` annotations, window → ``(n_ep, n_ch, n_samp)``.

    Epoching never crosses an EEGLAB ``boundary`` (a recording discontinuity);
    each contiguous segment is windowed independently and the remainder dropped.
    """
    if float(raw.info["sfreq"]) != float(target_sfreq):
        raw = raw.copy().resample(target_sfreq, verbose=False)
    data = raw.get_data()  # (n_ch, n_samples)
    sfreq = float(raw.info["sfreq"])
    spe = int(round(epoch_seconds * sfreq))
    n_total = data.shape[1]
    # Contiguous-segment boundaries from the 'boundary' annotations.
    cut_samples = sorted(
        int(round(onset * sfreq))
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description, strict=False)
        if str(desc).lower() == "boundary"
    )
    cuts = [0, *[c for c in cut_samples if 0 < c < n_total], n_total]
    segments: list[np.ndarray] = []
    for a, b in zip(cuts[:-1], cuts[1:], strict=False):
        seg = data[:, a:b]
        n_ep = seg.shape[1] // spe
        if n_ep > 0:
            trimmed = seg[:, : n_ep * spe].reshape(data.shape[0], n_ep, spe)
            segments.append(np.transpose(trimmed, (1, 0, 2)))
    if not segments:
        raise ValueError(
            f"no {epoch_seconds}s epochs in recording ({n_total} samples at {sfreq} Hz)"
        )
    return np.concatenate(segments, axis=0)


def _load_set(set_path: Path) -> mne.io.BaseRaw:
    """``read_raw_eeglab`` with EEG-only channel types, picked to canonical."""
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
    return _pick_canonical(raw)


def _subject_dir_name(sid: int) -> str:
    return f"sub-{sid:02d}"


def _match_eeg_members(names: list[str], ses: str, stem: str) -> list[str]:
    """Zip members for one (session, stem), robust to nesting depth.

    The COG-BCI per-subject zips are inconsistently packaged: some subjects use
    ``sub-NN/ses-S1/eeg/zeroBACK.set`` while others double-nest as
    ``sub-NN/sub-NN/ses-S1/eeg/zeroBACK.set``. Matching on the path *suffix*
    rather than a fixed prefix handles both (verified 2026-06-09: sub-01/02 are
    single-nested, sub-03+ double-nested).
    """
    set_suffix = f"ses-{ses}/eeg/{stem}.set"
    fdt_suffix = f"ses-{ses}/eeg/{stem}.fdt"
    return [n for n in names if n.endswith(set_suffix) or n.endswith(fdt_suffix)]


def load_cogbci(
    subjects: list[int] | None = None,
    *,
    data_root: Path | None = None,
    sessions: tuple[str, ...] = COGBCI_SESSIONS,
    set_stems: dict[str, str] | None = None,
    near_ear: bool = False,
    epoch_seconds: float = 4.0,
    target_sfreq: float = 250.0,
) -> list[SubjectData]:
    """Load COG-BCI N-back into 3-class workload ``SubjectData`` (one per subject).

    Each per-subject zip is extracted to a temp dir on demand, the requested
    ``.set`` recordings are read + reduced to the canonical 62-ch montage,
    epoched, and the temp dir removed.

    Parameters
    ----------
    subjects : 1-indexed subject IDs (1..29), or ``None`` for all 29.
    data_root : directory of ``sub-NN.zip`` files. Defaults to
        ``new_datasets/7413650`` (gitignored).
    sessions : which sessions to include. Within-session cells pass
        ``("S1",)``; the cross-session cell (Phase 4) passes all three. The
        ``session`` metadata column records each epoch's session.
    set_stems : ``{stem: label}`` map of which ``.set`` files to load. Defaults
        to the N-back map :data:`COGBCI_NBACK_FILES`; pass ``{"PVT": "pvt"}`` for
        the Phase-6 vigilance cell.
    near_ear : subset to the T7/T8 pair at load time (position-based).
    epoch_seconds, target_sfreq : window length and resample target.

    Returns
    -------
    list[SubjectData] — ``metadata`` carries ``subject`` (1..29),
    ``session`` (``S1``/``S2``/``S3``), and ``run`` = the ``.set`` stem.
    """
    root = Path(data_root) if data_root is not None else _DEFAULT_COGBCI_ROOT
    stems = dict(set_stems) if set_stems is not None else dict(COGBCI_NBACK_FILES)
    wanted = list(range(1, COGBCI_N_SUBJECTS + 1)) if subjects is None else [int(s) for s in subjects]

    out: list[SubjectData] = []
    for sid in wanted:
        zip_path = root / f"{_subject_dir_name(sid)}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(
                f"{zip_path} missing. Download COG-BCI (Zenodo 7413650) under "
                "new_datasets/7413650/ — see data/NEW_DATASETS.README.md."
            )
        tmp = Path(tempfile.mkdtemp(prefix=f"cogbci_{sid:02d}_"))
        try:
            X_blocks: list[np.ndarray] = []
            y_blocks: list[np.ndarray] = []
            session_meta: list[str] = []
            run_meta: list[str] = []
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                for ses in sessions:
                    for stem, label in stems.items():
                        members = _match_eeg_members(names, ses, stem)
                        if not members:
                            continue
                        zf.extractall(tmp, members=members)
                        set_member = next((n for n in members if n.endswith(".set")), None)
                        if set_member is None:
                            continue
                        raw = _load_set(tmp / set_member)
                        epochs = _raw_to_epochs(
                            raw, target_sfreq=target_sfreq, epoch_seconds=epoch_seconds
                        )
                        X_blocks.append(epochs)
                        y_blocks.append(np.array([label] * epochs.shape[0]))
                        session_meta.extend([ses] * epochs.shape[0])
                        run_meta.extend([stem] * epochs.shape[0])
            if not X_blocks:
                # A present zip yielding no epochs almost always signals a member
                # path-structure mismatch, not a legitimately-absent subject —
                # fail loudly rather than silently dropping the subject.
                raise ValueError(
                    f"COG-BCI {zip_path.name}: no epochs for sessions={list(sessions)} "
                    f"stems={list(stems)}. Member structure may differ — first 3 members: "
                    f"{names[:3]}"
                )
            X = np.concatenate(X_blocks, axis=0)
            y = np.concatenate(y_blocks)
            meta = pd.DataFrame({
                "subject": [sid] * X.shape[0],
                "session": session_meta,
                "run": run_meta,
            })
            data = SubjectData(
                subject=sid,
                dataset_name="COGBCI",
                X=X,
                y=y,
                metadata=meta,
                sfreq=float(target_sfreq),
            )
            if near_ear:
                data = select_near_ear(data, COGBCI_CANONICAL_EEG)
            out.append(data)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return out
