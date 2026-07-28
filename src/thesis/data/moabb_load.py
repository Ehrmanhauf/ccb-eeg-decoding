"""MOABB-backed loaders for Phase-5 §2.2 / §2.4 / §2.5 extensions.

These loaders bridge MOABB's Paradigm.get_data() output to the project's
:class:`thesis.data.SubjectData` schema so the existing CCB runner and
FBCSP baseline can consume cross-dataset experiments without per-dataset
glue code in the runner itself.

Datasets covered:
- :func:`load_physionet_mi` — Schalk et al. 2004 PhysioNet EEG MMIDB
  (109 subjects, 64 channels, left/right hand MI). Phase-5 §2.2.
- :func:`load_bnci2015_004` — Faller et al. 2015 mental-tasks dataset
  (9 subjects, 30 channels, 5 cognitive tasks). Phase-5 §2.4.
- :func:`load_cho2017` — Cho et al. 2017 MI dataset (52 subjects, 64
  channels, left/right hand MI). Phase-5 §2.5 / Workstream C.1. Supports
  ``channels="full"`` and ``channels="c3_cz_c4"`` for the monopolar-vs-2b
  comparison described in `design-doc/ccb-formulation.md` §2.5.

ref (PhysioNet): `Schalk et al. 2004 IEEE Trans. Biomed. Eng. 51(6),
DOI 10.1109/TBME.2004.827072` — to be added to references.bib.
ref (BNCI2015_004): `Faller, Vidaurre, Solis-Escalante, Neuper, Scherer
2015 PLoS One 10(5):e0123727, DOI 10.1371/journal.pone.0123727` — to be
added to references.bib.
ref (Cho2017): `cho2017mieeg` (already in references.bib).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from moabb.datasets import BNCI2015_004, Cho2017, PhysionetMI
from moabb.paradigms import LeftRightImagery, MotorImagery

from thesis.data.load import SubjectData


def load_physionet_mi(subjects: list[int]) -> list[SubjectData]:
    """Load PhysioNet EEG MMIDB filtered to 2-class left-vs-right hand MI.

    Uses MOABB's :class:`LeftRightImagery` paradigm which (a) filters to
    the two-class subproblem and (b) extracts only the imagined-fists runs
    (T3, T4, T7, T8, T11, T12 in PhysioNet's labelling). 64 EEG channels at
    160 Hz; ~45 trials per subject (15 trials × 3 imagined-fist runs).

    Parameters
    ----------
    subjects : list of 1-indexed subject IDs (1..109).

    Returns
    -------
    list[SubjectData] — one per requested subject.
    """
    ds = PhysionetMI()
    paradigm = LeftRightImagery()
    out: list[SubjectData] = []
    for sid in subjects:
        X, y, meta = paradigm.get_data(dataset=ds, subjects=[sid])
        # MOABB returns shape (n_trials, n_channels, n_samples) and labels
        # already mapped to {'left_hand', 'right_hand'} strings — matches
        # our CLASS_LABELS convention.
        out.append(
            SubjectData(
                subject=int(sid),
                dataset_name="PhysioNet-MI-LeftRight",
                X=X,
                y=np.asarray(y),
                metadata=meta.reset_index(drop=True),
                sfreq=160.0,
            )
        )
    return out


def load_bnci2015_004(
    subjects: list[int],
    *,
    classes: Literal["all", "two"] = "two",
) -> list[SubjectData]:
    """Load Faller et al. 2015 mental-tasks dataset.

    5 mental tasks per subject — 'right_hand', 'feet', 'navigation',
    'subtraction', 'word_ass'. By default we extract the 2-class subproblem
    (right_hand vs subtraction) which most cleanly separates motor-only
    from cognitive-only demand and gives a binary CCB target compatible with
    the existing reward signal. Pass ``classes="all"`` for the 5-class
    setup (requires multi-class CCB head — outside §2.4 MVP scope).

    Parameters
    ----------
    subjects : list of 1-indexed subject IDs (1..9).
    classes : "two" (default) or "all".

    Returns
    -------
    list[SubjectData] — one per requested subject.
    """
    ds = BNCI2015_004()
    if classes == "two":
        # Two-class: motor-task (right_hand) vs cognitive-task (subtraction).
        # MOABB's MotorImagery paradigm with explicit `events` selects the
        # subset and re-encodes y to those two strings.
        paradigm = MotorImagery(events=["right_hand", "subtraction"], n_classes=2)
    else:
        paradigm = MotorImagery(n_classes=5)
    out: list[SubjectData] = []
    for sid in subjects:
        X, y, meta = paradigm.get_data(dataset=ds, subjects=[sid])
        out.append(
            SubjectData(
                subject=int(sid),
                dataset_name=f"BNCI2015-004-{classes}",
                X=X,
                y=np.asarray(y),
                metadata=meta.reset_index(drop=True),
                sfreq=256.0,
            )
        )
    return out


# Subjects flagged as unusable by Cho et al. 2017 (the dataset authors).
# ref: cho2017mieeg note in references.bib.
_CHO2017_EXCLUDED = (29, 33)

# Hardware-deployment-driven 3-channel subset matching the 2b motor-cortex
# montage. See ccb-formulation.md §2.5 for the no-leakage rationale.
_CHO2017_C3CZC4 = ("C3", "Cz", "C4")


def load_cho2017(
    subjects: list[int],
    *,
    channels: Literal["full", "c3_cz_c4"] = "full",
    tmin: float = 0.0,
    tmax: float = 4.0,
    resample_hz: float = 250.0,
) -> list[SubjectData]:
    """Load Cho 2017 MI dataset filtered to left/right hand.

    Uses MOABB's :class:`LeftRightImagery` paradigm which (a) filters to
    the two-class subproblem and (b) returns labels already mapped to
    ``{'left_hand', 'right_hand'}`` — matches the project ``CLASS_LABELS``
    convention used by 2a/2b.

    Two channel configurations are supported (see ccb-formulation.md §2.5):

    - ``channels="full"`` (default) keeps all 64 monopolar channels.
    - ``channels="c3_cz_c4"`` restricts to the three motor-cortex
      positions matching the 2b montage. The selection is a fixed
      hardware-deployment choice (a wearable-cap stencil), **not** a
      data-driven channel reduction — applied at load time before the CCB
      ever sees the trials.

    Epoch window ``[tmin, tmax] = [0, 4]`` s post-cue and resampling to
    250 Hz match the project 2a/2b pipeline. Subjects flagged as unusable
    by Cho et al. (s29 and s33) are skipped automatically.

    Parameters
    ----------
    subjects : list of 1-indexed subject IDs (1..52). s29 and s33 are
        silently dropped if requested.
    channels : "full" or "c3_cz_c4".
    tmin, tmax : float, default 0.0, 4.0. Trial epoch window in seconds
        relative to cue.
    resample_hz : float, default 250.0. Target sampling rate (Cho2017
        records natively at 512 Hz).

    Returns
    -------
    list[SubjectData] — one per loaded subject (excluded subjects dropped).

    ref: `cho2017mieeg` (verified 2026-05-12 against the journal page).
    """
    if channels == "full":
        ch_list: list[str] | None = None
        suffix = "full"
    elif channels == "c3_cz_c4":
        ch_list = list(_CHO2017_C3CZC4)
        suffix = "3ch"
    else:
        raise ValueError(
            f"unknown channels={channels!r}; expected 'full' or 'c3_cz_c4'"
        )

    ds = Cho2017()
    paradigm = LeftRightImagery(
        tmin=tmin,
        tmax=tmax,
        channels=ch_list,
        resample=resample_hz,
    )
    out: list[SubjectData] = []
    for sid in subjects:
        if int(sid) in _CHO2017_EXCLUDED:
            continue
        X, y, meta = paradigm.get_data(dataset=ds, subjects=[int(sid)])
        out.append(
            SubjectData(
                subject=int(sid),
                dataset_name=f"Cho2017-{suffix}",
                X=X,
                y=np.asarray(y),
                metadata=meta.reset_index(drop=True),
                sfreq=float(resample_hz),
            )
        )
    return out
