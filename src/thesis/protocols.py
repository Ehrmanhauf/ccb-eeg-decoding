"""Evaluation protocols for MI classification.

Three protocols are implemented here, matching what the CCB design document
commits to (``ccb-formulation.md`` §8.1):

- Within-subject stratified K-fold CV. Primary protocol.
- Session split (train on session ``i``, test on session ``j``). This is how
  the original BCI Competition IV evaluated 2a (session 1 → session 2) and is
  what this thesis adopts for 2b adapted to screening-only.
- Leave-one-subject-out (LOSO). Cross-subject hard mode: pool every subject,
  hold one out for test, train on the rest. Unlike the first two — which
  partition *within* one subject's trials — LOSO operates *across* subjects,
  so it returns a single pooled :class:`~thesis.data.SubjectData` plus a
  :class:`Split` that holds the test subject out. This structurally removes
  the within-segment leakage that inflates within-subject CV on datasets
  whose label is constant within a continuous recording segment (STEW: one
  rest bin + one multitask bin per subject — see ``stew_load.py``).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from thesis.data import SubjectData


@dataclass(frozen=True)
class Split:
    """One train/test index split of a subject's trials."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    name: str


def within_subject_cv(data: SubjectData, *, n_splits: int = 5, seed: int = 42) -> Iterator[Split]:
    """Stratified K-fold CV over all of one subject's trials."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(data.X, data.y)):
        yield Split(
            train_idx=np.asarray(train_idx),
            test_idx=np.asarray(test_idx),
            name=f"within_fold{fold_i}",
        )


def session_split(
    data: SubjectData, *, train_session_idx: int = 0, test_session_idx: int = 1
) -> Split:
    """Train on one session's trials, test on another — the official protocol."""
    sessions = data.sessions
    if train_session_idx >= len(sessions) or test_session_idx >= len(sessions):
        raise ValueError(
            f"Subject {data.subject} has {len(sessions)} sessions; cannot use indices "
            f"{train_session_idx}/{test_session_idx}."
        )
    train_sess = sessions[train_session_idx]
    test_sess = sessions[test_session_idx]
    session_col = data.metadata["session"].to_numpy()
    train_idx = np.where(session_col == train_sess)[0]
    test_idx = np.where(session_col == test_sess)[0]
    return Split(
        train_idx=train_idx,
        test_idx=test_idx,
        name=f"official[{train_sess}→{test_sess}]",
    )


def cross_session_split(
    data: SubjectData,
    *,
    train_sessions: Sequence[str],
    test_sessions: Sequence[str],
) -> Split:
    """Train on one set of sessions, test on another — the cross-session drift test.

    The deployment-regime protocol of the near-ear reframe (``design-doc/
    near-ear-reframe-workplan.md`` §5 Phase 4): on COG-BCI, train on session S1
    and test on the later sessions S2/S3 recorded a week apart, so the only thing
    that changes between train and test is across-session drift — the one regime
    where the CCB's online adaptation could earn its cost.

    Unlike :func:`session_split` (which takes session *indices* into
    ``data.sessions`` for the two-session BCI-IV protocol), this takes explicit
    session *labels* (e.g. ``"S1"``) matched against the ``session`` metadata
    column, and allows multiple sessions on either side. Train and test session
    sets must be disjoint and both non-empty in the data.
    """
    train_set, test_set = set(train_sessions), set(test_sessions)
    if train_set & test_set:
        raise ValueError(
            f"train and test sessions must be disjoint; got overlap "
            f"{sorted(train_set & test_set)}"
        )
    sess = data.metadata["session"].to_numpy()
    train_idx = np.where(np.isin(sess, list(train_sessions)))[0]
    test_idx = np.where(np.isin(sess, list(test_sessions)))[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        present = sorted(set(sess.tolist()))
        raise ValueError(
            f"cross_session_split: empty train ({len(train_idx)}) or test "
            f"({len(test_idx)}); requested train={sorted(train_set)}, "
            f"test={sorted(test_set)}, present={present}"
        )
    return Split(
        train_idx=train_idx,
        test_idx=test_idx,
        name=f"crosssession[{'+'.join(train_sessions)}→{'+'.join(test_sessions)}]",
    )


def pool_subjects(data_list: list[SubjectData], *, subject: int = -1) -> SubjectData:
    """Concatenate several subjects' trials into one pooled :class:`SubjectData`.

    All inputs must agree on ``dataset_name``, ``sfreq``, channel count, and
    sample count (only the trial axis may differ). The pooled ``metadata``
    keeps each trial's original ``subject`` id so a cross-subject protocol can
    recover provenance and build held-out masks. ``subject`` sets the pooled
    object's scalar id (default ``-1`` = "pooled / not a single subject").

    The pooled ``X`` / ``y`` are fresh concatenations; ``metadata`` is a
    re-indexed concat. No per-subject array is mutated.
    """
    if not data_list:
        raise ValueError("pool_subjects requires at least one SubjectData")
    ref = data_list[0]
    for d in data_list[1:]:
        if d.dataset_name != ref.dataset_name:
            raise ValueError(
                f"cannot pool across datasets: {d.dataset_name!r} != {ref.dataset_name!r}"
            )
        if d.sfreq != ref.sfreq:
            raise ValueError(f"sfreq mismatch in pool: {d.sfreq} != {ref.sfreq}")
        if d.X.shape[1:] != ref.X.shape[1:]:
            raise ValueError(
                f"channel/sample shape mismatch in pool: {d.X.shape[1:]} != {ref.X.shape[1:]}"
            )
    return SubjectData(
        subject=subject,
        dataset_name=ref.dataset_name,
        X=np.concatenate([d.X for d in data_list], axis=0),
        y=np.concatenate([d.y for d in data_list], axis=0),
        metadata=pd.concat([d.metadata for d in data_list], ignore_index=True),
        sfreq=ref.sfreq,
    )


def leave_one_subject_out(
    data_list: list[SubjectData],
) -> Iterator[tuple[SubjectData, Split]]:
    """Yield ``(pooled_data, Split)`` once per held-out subject — the LOSO protocol.

    Every subject is pooled into a single :class:`SubjectData` (via
    :func:`pool_subjects`); each yielded :class:`Split` puts the held-out
    subject's trials in ``test_idx`` and every *other* subject's trials in
    ``train_idx``. No trial from the test subject ever appears in train, so
    the within-segment leakage that inflates within-subject CV on
    segment-homogeneous datasets is structurally removed: a fixed pipeline can
    no longer "recognise the segment" because the test subject's segments are
    unseen at fit time.

    The pooled object's scalar ``subject`` is rebound to the held-out id per
    fold (cheap ``dataclasses.replace`` — the large ``X`` / ``y`` arrays are
    shared, not copied), so a downstream :class:`~thesis.ccb.runner.CCBResult`
    records the correct subject. The same ``(pooled_data, Split)`` pair drives
    both the fixed baselines (``clf.fit(X[train_idx]); clf.predict(X[test_idx])``)
    and the CCB (``run_ccb_on_split(pooled_data, split)`` — calibration is
    carved from ``train_idx``, i.e. cross-subject, and the test stream is the
    held-out subject).

    Subjects must have distinct ids. Order of yielded folds follows
    ``data_list``.
    """
    ids = [d.subject for d in data_list]
    if len(set(ids)) != len(ids):
        raise ValueError(f"leave_one_subject_out requires distinct subject ids; got {ids}")
    pooled = pool_subjects(data_list)
    subj_col = pooled.metadata["subject"].to_numpy()
    for sid in ids:
        test_idx = np.where(subj_col == sid)[0]
        train_idx = np.where(subj_col != sid)[0]
        fold_data = replace(pooled, subject=sid)
        yield fold_data, Split(train_idx=train_idx, test_idx=test_idx, name=f"loso[s{sid}]")
