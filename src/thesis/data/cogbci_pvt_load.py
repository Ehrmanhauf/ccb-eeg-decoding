r"""COG-BCI PVT vigilance loader (Phase 6 — the second vital sign).

Points the near-ear pipeline at a *different paradigm* — **vigilance / fatigue** —
using the Psychomotor Vigilance Task (PVT) recordings of the COG-BCI database
(Zenodo 7413650; 29 subjects × 3 sessions). Each ``ses-S{n}/eeg/PVT.set`` is a
continuous ~10-minute recording (63 ch, 500 Hz) with stimulus-onset event markers,
paired with ``ses-S{n}/behavioral/PVT.mat`` carrying the per-trial reaction times.

**Operational definition of vigilance (documented, per the CL-methodology rule).**
The canonical PVT vigilance metric is the *lapse* (reaction time > 500 ms; Dinges &
Powell 1985, Basner & Dinges 2011). In this rested cohort lapses are rare (≈ 5–6 %
of trials), which is too imbalanced for a two-class decoding target. We therefore
use the standard *balanced* operationalization: a **within-session median split of
single-trial reaction time** — the slower half = lower vigilance (``low``), the faster
half = higher vigilance (``high``). RT is the canonical PVT vigilance variable (Lim &
Dinges 2008); the median is a label-definition threshold, not a feature, so it leaks
no EEG information into the decision. (For a cross-session split, derive the threshold
on the training session only — see ``threshold`` below.)

**Epoching.** Each trial is the **2 s window immediately before its stimulus onset**
(``pre_stim=(-2.0, 0.0)``), i.e. the ongoing brain state that precedes — and is used to
predict — the upcoming response's vigilance level. This is the deployment-relevant
"detect low vigilance from ongoing EEG" framing. Windows are resampled to 250 Hz to
match the rest of the panel (500 samples, 2 s). Near-ear is the T7/T8 subset by
electrode position, exactly as elsewhere.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

from .load import SubjectData
from .near_ear import select_near_ear

mne.set_log_level("ERROR")

COGBCI_PVT_ROOT = Path("new_datasets/7413650")
PVT_SESSIONS = ("S1", "S2", "S3")
_STIM_ANNOT = "13"  # EEGLAB event code for the PVT stimulus onset (verified sub-01)

# 62-channel PVT canonical EEG montage. Every session is reduced (via ``pick``) to
# this exact set, dropping the non-EEG ECG1 and Cz (Cz is present only on subjects
# 10–29), so the montage matches the workload loader's canonical 62 channels. The
# name-based roles below reserve T7/T8 for the near-ear subset.
COGBCI_PVT_EEG = (
    "Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "CP5", "CP1",
    "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "TP10", "CP6", "CP2", "FCz", "C4",
    "T8", "FT10", "FC6", "FC2", "F4", "F8", "Fp2", "AF7", "AF3", "AFz", "F1", "F5",
    "FT7", "FC3", "C1", "C5", "TP7", "CP3", "P1", "P5", "PO7", "PO3", "POz", "PO4",
    "PO8", "P6", "P2", "CPz", "CP4", "TP8", "C6", "C2", "FC4", "FT8", "F6", "AF8", "AF4", "F2",
)


def _pvt_roles() -> dict[str, list[int]]:
    """Workload/vigilance channel-roles for the full PVT montage (frontal-θ and
    parietal-α aggregates, F3/F4 asymmetry pair; temporal channels excluded)."""
    names = COGBCI_PVT_EEG
    frontal = [i for i, c in enumerate(names)
               if c.upper().startswith(("F", "AF", "FP", "FC", "FT")) and c.upper() not in ("F3", "F4")]
    parietal = [i for i, c in enumerate(names) if c.upper().startswith(("P", "PO", "CP"))]
    return {"frontal": frontal, "parietal": parietal,
            "f3": [names.index("F3")], "f4": [names.index("F4")]}


COGBCI_PVT_ROLES = _pvt_roles()


def _zip_for(sid: int, root: Path) -> Path:
    return root / f"sub-{sid:02d}.zip"


def _extract_session(zf: zipfile.ZipFile, sid: int, ses: str, dest: Path) -> tuple[Path, Path] | None:
    """Extract PVT .set/.fdt + .mat for one session; return (set_path, mat_path) or None."""
    want = [m for m in zf.namelist()
            if f"ses-{ses}/" in m and "PVT" in m
            and (m.endswith(".set") or m.endswith(".fdt") or m.endswith(".mat"))]
    if not any(m.endswith(".set") for m in want):
        return None
    for m in want:
        zf.extract(m, dest)
    set_path = next(dest.rglob(f"*ses-{ses}*/eeg/PVT.set"), None)
    mat_path = next(dest.rglob(f"*ses-{ses}*/behavioral/PVT.mat"), None)
    return (set_path, mat_path) if (set_path and mat_path) else None


def _session_epochs(
    set_path: Path, mat_path: Path, *, pre_stim: tuple[float, float], target_sfreq: float
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X[n,ch,t], reaction_times[n], ch_names) for one PVT session."""
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True)
    missing = [c for c in COGBCI_PVT_EEG if c not in raw.ch_names]
    if missing:
        raise ValueError(f"PVT session missing canonical channels {missing}")
    raw.pick(list(COGBCI_PVT_EEG))
    events, evid = mne.events_from_annotations(raw)
    if _STIM_ANNOT not in evid:
        raise ValueError(f"PVT stimulus code '{_STIM_ANNOT}' absent; have {sorted(evid)}")
    stim = events[events[:, 2] == evid[_STIM_ANNOT]]
    rt = np.atleast_1d(loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)["PVT"].reaction_times).astype(float)
    n = min(len(stim), len(rt))
    if n == 0:
        raise ValueError("no PVT stimulus epochs")
    epochs = mne.Epochs(raw, stim[:n], tmin=pre_stim[0], tmax=pre_stim[1],
                        baseline=None, preload=True, reject=None, verbose=False)
    epochs.resample(target_sfreq, verbose=False)
    X = epochs.get_data(copy=True)
    kept = epochs.selection  # indices of trials that survived bounds
    rt = rt[kept[kept < n]]
    X = X[: len(rt)]
    return X.astype(np.float64), rt[: len(X)], list(epochs.ch_names)


def load_cogbci_pvt(
    subjects: list[int] | None = None,
    *,
    sessions: tuple[str, ...] = PVT_SESSIONS,
    near_ear: bool = False,
    pre_stim: tuple[float, float] = (-2.0, 0.0),
    target_sfreq: float = 250.0,
    root: Path = COGBCI_PVT_ROOT,
) -> list[SubjectData]:
    """Load COG-BCI PVT as one :class:`SubjectData` per subject (vigilance target).

    ``metadata['session']`` carries S1/S2/S3 so within-session CV and cross-session
    splits both work. The label is the within-session RT median split (``high`` =
    faster/more-vigilant half, ``low`` = slower half).
    """
    zips = sorted(p for p in (root.glob("sub-*.zip")))
    if subjects is not None:
        want = {f"sub-{int(s):02d}.zip" for s in subjects}
        zips = [z for z in zips if z.name in want]
    out: list[SubjectData] = []

    for zp in zips:
        sid = int(zp.stem.split("-")[1])
        Xs, ys, sess_col, ch_names = [], [], [], None
        with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(zp) as zf:
            for ses in sessions:
                got = _extract_session(zf, sid, ses, Path(td))
                if got is None:
                    continue
                X, rt, names = _session_epochs(*got, pre_stim=pre_stim, target_sfreq=target_sfreq)
                if ch_names is None:
                    ch_names = names
                elif names != ch_names:
                    idx = [names.index(c) for c in ch_names]
                    X = X[:, idx, :]
                # within-session median split: slower half -> low vigilance
                med = float(np.median(rt))
                y = np.where(rt > med, "low", "high")
                Xs.append(X)
                ys.extend(y.tolist())
                sess_col.extend([ses] * len(y))
        if not Xs:
            raise ValueError(f"no PVT sessions loaded for subject {sid}")
        X = np.concatenate(Xs, axis=0)
        meta = pd.DataFrame({"subject": sid, "session": sess_col, "run": 0})
        data = SubjectData(
            subject=sid, dataset_name="COGBCI-PVT", X=X,
            y=np.asarray(ys, dtype=object), metadata=meta, sfreq=float(target_sfreq),
        )
        out.append(select_near_ear(data, ch_names) if near_ear else data)
    return out
