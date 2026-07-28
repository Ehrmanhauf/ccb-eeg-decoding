r"""COG-BCI MATB *competition split* loader (Zenodo 5055046) — the leaderboard anchor.

Distinct from ``cogbci_load.py`` (the 3-session COG-BCI database, Zenodo 7413650):
this is the **passive-BCI hackathon competition split**, pre-epoched, two sessions
(S1, S2) with MATB-II at three difficulties (easy/med/diff); session S3 ships only
resting-state, so the MATB cross-session test is **train S1 → test S2**. It is the
*directly leaderboard-comparable* MATB cell the near-ear work plan calls the "anchor"
(``design-doc/near-ear-reframe-workplan.md`` §3; the published leaderboard tops out
< 60 % 3-class accuracy under calibration-permitted conditions).

Archive layout (per subject)::

    P{NN}.zip
      └─ P{NN}/S{1,2}/eeg/alldata_sbj{NN}_sess{1,2}_MATB{easy,med,diff}.set (+ .fdt)

Each ``.set`` is **pre-epoched** EEGLAB data: ~149 epochs × 61 channels × 500 samples
(2 s @ 250 Hz). T7/T8 are present, so the near-ear subset is the same position-based
operation as every other dataset (no leakage).

**A/C/C and protocol are identical to the 3-session MATB cell** (\S3.5 of the thesis);
only the archive format and the train/test session pair differ. Labels easy/med/diff
map to the 3-class workload target low/medium/high, matching STEW/UAB/COG-BCI.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .load import SubjectData
from .near_ear import select_near_ear

mne.set_log_level("ERROR")

MATB_COMP_ROOT = Path("new_datasets/5055046")
MATB_DIFFICULTY = {"MATBeasy": "low", "MATBmed": "medium", "MATBdiff": "high"}
MATB_COMP_SESSIONS = ("S1", "S2")  # S3 ships resting-state only → MATB withheld

# Canonical 61-channel montage of the competition split (uniform across subjects;
# verified against P01/S1 MATBeasy). T7/T8 (the near-ear pair) are at indices 8/23.
COGBCI_MATBCOMP_EEG = (
    "Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "Pz",
    "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "CP6", "CP2", "FCz", "C4", "T8",
    "FT8", "FC6", "FC2", "F4", "F8", "Fp2", "AF7", "AF3", "AFz", "F1", "F5", "FT7",
    "FC3", "C1", "C5", "TP7", "CP3", "P1", "P5", "PO7", "PO3", "POz", "PO4", "PO8",
    "P6", "P2", "CPz", "CP4", "TP8", "C6", "C2", "FC4", "FT10", "F6", "AF8", "AF4", "F2",
)


def _matbcomp_roles() -> dict[str, list[int]]:
    """Workload channel-roles for the full competition montage (mirrors the STEW/
    COG-BCI role convention: frontal-θ aggregate, parietal-α aggregate, F3/F4 the
    frontal-α asymmetry pair)."""
    names = COGBCI_MATBCOMP_EEG
    frontal = [i for i, c in enumerate(names)
               if c.upper().startswith(("F", "AF", "FP", "FC", "FT")) and c.upper() not in ("F3", "F4")]
    parietal = [i for i, c in enumerate(names) if c.upper().startswith(("P", "PO", "CP"))]
    return {"frontal": frontal, "parietal": parietal,
            "f3": [names.index("F3")], "f4": [names.index("F4")]}


COGBCI_MATBCOMP_ROLES = _matbcomp_roles()


def _read_one(set_path: Path) -> tuple[np.ndarray, list[str], float]:
    """Read one pre-epoched EEGLAB MATB ``.set`` → (X[n,ch,t], ch_names, sfreq)."""
    ep = mne.io.read_epochs_eeglab(str(set_path))
    return ep.get_data(copy=True), list(ep.ch_names), float(ep.info["sfreq"])


def load_cogbci_matb(
    subjects: list[int] | None = None,
    *,
    sessions: tuple[str, ...] = MATB_COMP_SESSIONS,
    near_ear: bool = False,
    root: Path = MATB_COMP_ROOT,
) -> list[SubjectData]:
    """Load the COG-BCI MATB competition split as one :class:`SubjectData` per subject.

    Each subject's ``metadata['session']`` carries ``S1`` / ``S2`` per epoch, so
    ``thesis.protocols.cross_session_split(d, train_sessions=['S1'],
    test_sessions=['S2'])`` gives the leaderboard-comparable cross-session test.

    Parameters
    ----------
    subjects : 1-based subject ids; ``None`` = all ``P*.zip`` present.
    near_ear : restrict to the T7/T8 near-ear subset (position-based, at load time).
    """
    zips = sorted(root.glob("P*.zip"))
    if subjects is not None:
        want = {f"P{int(s):02d}.zip" for s in subjects}
        zips = [z for z in zips if z.name in want]
    out: list[SubjectData] = []

    for zp in zips:
        sid = int(zp.stem[1:])  # "P01" -> 1
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(td)
            base = Path(td)
            Xs, ys, sess_col, ch_names, sfreq = [], [], [], None, None
            for sess in sessions:
                snum = sess[1:]  # "S1" -> "1"
                for stem, label in MATB_DIFFICULTY.items():
                    matches = list(base.glob(f"P{sid:02d}/{sess}/eeg/alldata_sbj{sid:02d}_sess{snum}_{stem}.set"))
                    if not matches:
                        continue
                    X, names, sf = _read_one(matches[0])
                    if ch_names is None:
                        ch_names, sfreq = names, sf
                    elif names != ch_names:
                        # align by shared names (defensive; the archive is uniform)
                        idx = [names.index(c) for c in ch_names]
                        X = X[:, idx, :]
                    Xs.append(X.astype(np.float64))
                    ys.extend([label] * X.shape[0])
                    sess_col.extend([sess] * X.shape[0])
            if not Xs:
                raise ValueError(f"no MATB .set found for subject {sid} in {zp}")
            X = np.concatenate(Xs, axis=0)
            meta = pd.DataFrame({"subject": sid, "session": sess_col, "run": 0})
            data = SubjectData(
                subject=sid,
                dataset_name="COGBCI-MATBcomp",
                X=X,
                y=np.asarray(ys, dtype=object),
                metadata=meta,
                sfreq=float(sfreq),
            )
            out.append(select_near_ear(data, ch_names) if near_ear else data)
    return out
