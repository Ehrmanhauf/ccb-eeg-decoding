"""Unit tests for thesis.protocols.

These tests build synthetic ``SubjectData`` fixtures so they run offline — no
GDF/label files touched, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis.data import SubjectData
from thesis.protocols import (
    cross_session_split,
    leave_one_subject_out,
    pool_subjects,
    session_split,
    within_subject_cv,
)


def _make_subject(
    n_per_session: int = 60,
    n_sessions: int = 2,
    n_channels: int = 3,
    n_samples: int = 1000,
) -> SubjectData:
    """Synthetic balanced 2-class MI subject with ``n_sessions`` sessions."""
    rng = np.random.default_rng(seed=0)
    n = n_per_session * n_sessions
    X = rng.standard_normal((n, n_channels, n_samples))
    y = np.tile(
        np.concatenate(
            [
                np.array(["left_hand"] * (n_per_session // 2)),
                np.array(["right_hand"] * (n_per_session // 2)),
            ]
        ),
        n_sessions,
    )
    sessions = np.repeat([f"{i}train" for i in range(n_sessions)], n_per_session)
    metadata = pd.DataFrame({"subject": [1] * n, "session": sessions, "run": ["0run"] * n})
    return SubjectData(
        subject=1,
        dataset_name="synthetic",
        X=X,
        y=y,
        metadata=metadata,
        sfreq=250.0,
    )


def test_within_subject_cv_covers_all_trials_exactly_once():
    data = _make_subject(n_per_session=60, n_sessions=2)
    test_indices: list[int] = []
    for split in within_subject_cv(data, n_splits=5):
        # train and test are disjoint
        assert set(split.train_idx) & set(split.test_idx) == set()
        test_indices.extend(split.test_idx.tolist())
    # test folds together partition the trials
    assert sorted(test_indices) == list(range(data.n_trials))


def test_within_subject_cv_is_stratified():
    data = _make_subject(n_per_session=60, n_sessions=2)
    for split in within_subject_cv(data, n_splits=5):
        train_labels = data.y[split.train_idx]
        train_balance = dict(zip(*np.unique(train_labels, return_counts=True), strict=True))
        # stratification keeps class balance within one trial in each fold
        assert abs(train_balance["left_hand"] - train_balance["right_hand"]) <= 1


def test_session_split_uses_first_two_sessions():
    data = _make_subject(n_per_session=60, n_sessions=2)
    split = session_split(data, train_session_idx=0, test_session_idx=1)
    # train on session 0, test on session 1; both contain 60 trials
    assert len(split.train_idx) == 60
    assert len(split.test_idx) == 60
    # names carry through
    assert "0train" in split.name and "1train" in split.name


def test_session_split_rejects_out_of_range_indices():
    data = _make_subject(n_per_session=60, n_sessions=2)
    with pytest.raises(ValueError):
        session_split(data, train_session_idx=0, test_session_idx=5)


# --------------------------------------------------------------------------
# LOSO (leave-one-subject-out) + pooling
# --------------------------------------------------------------------------


def _make_named_subject(
    subject: int,
    *,
    n_trials: int = 20,
    n_channels: int = 3,
    n_samples: int = 500,
    dataset_name: str = "synthetic",
) -> SubjectData:
    """One synthetic subject whose label is constant per 'segment'.

    Mimics STEW's structure: the first half of the trials are one segment
    (label ``"low"``), the second half another (label ``"high"``), so the
    label is perfectly confounded with the segment within a subject — the
    setup LOSO is meant to defeat.
    """
    rng = np.random.default_rng(seed=subject)
    X = rng.standard_normal((n_trials, n_channels, n_samples))
    half = n_trials // 2
    y = np.array(["low"] * half + ["high"] * (n_trials - half))
    meta = pd.DataFrame(
        {
            "subject": [subject] * n_trials,
            "session": ["rest"] * half + ["multitask"] * (n_trials - half),
            "run": ["0"] * n_trials,
        }
    )
    return SubjectData(
        subject=subject, dataset_name=dataset_name, X=X, y=y, metadata=meta, sfreq=250.0
    )


def test_pool_subjects_concatenates_and_keeps_provenance():
    subs = [_make_named_subject(s, n_trials=20) for s in (1, 2, 3)]
    pooled = pool_subjects(subs)
    assert pooled.n_trials == 60
    assert pooled.X.shape == (60, 3, 500)
    # Each original subject's id survives in the pooled metadata.
    assert sorted(pooled.metadata["subject"].unique().tolist()) == [1, 2, 3]
    assert (pooled.metadata["subject"].to_numpy()[:20] == 1).all()
    # Default pooled scalar id flags "not a single subject".
    assert pooled.subject == -1


def test_pool_subjects_rejects_mismatched_datasets():
    a = _make_named_subject(1, dataset_name="STEW")
    b = _make_named_subject(2, dataset_name="WAUC")
    with pytest.raises(ValueError):
        pool_subjects([a, b])


def test_pool_subjects_rejects_shape_mismatch():
    a = _make_named_subject(1, n_channels=3)
    b = _make_named_subject(2, n_channels=4)
    with pytest.raises(ValueError):
        pool_subjects([a, b])


def test_loso_yields_one_fold_per_subject_no_leakage():
    subs = [_make_named_subject(s, n_trials=20) for s in (1, 2, 3, 4)]
    folds = list(leave_one_subject_out(subs))
    assert len(folds) == 4
    seen_test_subjects = []
    for fold_data, split in folds:
        # train and test partition the whole pooled set, disjointly.
        assert set(split.train_idx) & set(split.test_idx) == set()
        assert len(split.train_idx) + len(split.test_idx) == 80
        # The held-out subject's trials are ALL in test and NONE in train.
        subj_col = fold_data.metadata["subject"].to_numpy()
        held = fold_data.subject
        seen_test_subjects.append(held)
        assert set(subj_col[split.test_idx].tolist()) == {held}
        assert held not in set(subj_col[split.train_idx].tolist())
        # 4 subjects × 20 trials → 20 test, 60 train.
        assert len(split.test_idx) == 20
        assert "loso" in split.name
    # Every subject is held out exactly once.
    assert sorted(seen_test_subjects) == [1, 2, 3, 4]


def test_loso_rejects_duplicate_subject_ids():
    subs = [_make_named_subject(1), _make_named_subject(1)]
    with pytest.raises(ValueError):
        list(leave_one_subject_out(subs))


# --------------------------------------------------------------------------
# cross_session_split (the deployment-regime drift test)
# --------------------------------------------------------------------------


def _make_3session_subject(per_session: int = 20) -> SubjectData:
    rng = np.random.default_rng(0)
    n = per_session * 3
    X = rng.standard_normal((n, 3, 500))
    y = np.tile(np.array(["0back", "1back"] * (per_session // 2)), 3)
    sessions = np.repeat(["S1", "S2", "S3"], per_session)
    meta = pd.DataFrame({"subject": [1] * n, "session": sessions, "run": ["zeroBACK"] * n})
    return SubjectData(subject=1, dataset_name="COGBCI", X=X, y=y, metadata=meta, sfreq=250.0)


def test_cross_session_train_s1_test_s2_s3():
    data = _make_3session_subject(per_session=20)
    split = cross_session_split(data, train_sessions=["S1"], test_sessions=["S2", "S3"])
    assert len(split.train_idx) == 20  # S1
    assert len(split.test_idx) == 40  # S2 + S3
    assert set(split.train_idx) & set(split.test_idx) == set()
    sess = data.metadata["session"].to_numpy()
    assert set(sess[split.train_idx]) == {"S1"}
    assert set(sess[split.test_idx]) == {"S2", "S3"}
    assert "crosssession" in split.name


def test_cross_session_rejects_overlap():
    data = _make_3session_subject()
    with pytest.raises(ValueError, match="disjoint"):
        cross_session_split(data, train_sessions=["S1"], test_sessions=["S1", "S2"])


def test_cross_session_rejects_absent_session():
    data = _make_3session_subject()
    with pytest.raises(ValueError, match="empty train|empty"):
        cross_session_split(data, train_sessions=["S9"], test_sessions=["S2"])
