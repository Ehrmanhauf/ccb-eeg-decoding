"""Guardrail: matched-conditions splits are identical across method families.

This test exists because of a real methodological bug: the EEGNet LOSO path once
trained on the full ~34k-epoch WAUC pool while the fixed baselines used a 4000-epoch
subsample and the CCB used 800 --- a silent, comparison-invalidating mismatch. Every
method now draws its fold indices AND its post-cap training subsample from the single
source ``thesis.matched``; these tests assert that source produces byte-identical
splits/caps regardless of which method (or how many algorithm seeds) consumes them, so
the mismatch cannot recur. Runs offline on synthetic ``SubjectData`` (no real data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thesis.data import SubjectData
from thesis.matched import (
    DEFAULT_TRAIN_CAP,
    SPLIT_SEED,
    matched_cross_session,
    matched_loso,
    matched_session_split,
    matched_within_cv,
    subsample_train,
)
from thesis.protocols import Split, within_subject_cv


def _subject(sid: int, *, n: int = 200, n_sessions: int = 1, seed: int = 0) -> SubjectData:
    """Synthetic balanced 2-class subject; small arrays (n x 2 x 64)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2, 64))
    half = n // 2
    y = np.array([0] * half + [1] * (n - half))
    if n_sessions == 1:
        sessions = np.array(["S1"] * n)
    else:
        sessions = np.repeat([f"S{i + 1}" for i in range(n_sessions)], n // n_sessions)[:n]
    meta = pd.DataFrame({"subject": [sid] * n, "session": sessions})
    return SubjectData(subject=sid, dataset_name="synthetic", X=X, y=y, metadata=meta, sfreq=250.0)


def _split_eq(a: Split, b: Split) -> bool:
    return np.array_equal(a.train_idx, b.train_idx) and np.array_equal(a.test_idx, b.test_idx)


# --- The headline guarantee: identical pools across method "profiles" ---------


def test_loso_pool_identical_across_methods():
    """Calling matched_loso for each method profile yields byte-identical (data, split)."""
    cohort = [_subject(s, n=200, seed=s) for s in range(1, 7)]  # pool 1000 per fold
    fixed = list(matched_loso(cohort, cap=400))
    ccb = list(matched_loso(cohort, cap=400))
    eegnet = list(matched_loso(cohort, cap=400))
    assert len(fixed) == len(ccb) == len(eegnet) == 6
    for (df_f, sf), (df_c, sc), (df_e, se) in zip(fixed, ccb, eegnet, strict=True):
        assert df_f.subject == df_c.subject == df_e.subject
        assert _split_eq(sf, sc) and _split_eq(sc, se)
        # n_train (the column the bug corrupted) is identical for all three.
        assert len(sf.train_idx) == len(sc.train_idx) == len(se.train_idx)


def test_loso_cap_bites_and_test_untouched():
    """The common cap actually subsamples the train pool; the held-out test set is full."""
    cohort = [_subject(s, n=200, seed=s) for s in range(1, 7)]  # held-in pool = 1000
    for fold_data, split in matched_loso(cohort, cap=400):
        assert len(split.train_idx) <= 400  # cap bites (pool 1000 -> <=400)
        # test_idx is the entire held-out subject, never subsampled
        held = np.where(fold_data.metadata["subject"].to_numpy() == fold_data.subject)[0]
        assert np.array_equal(split.test_idx, held)
        # no train/test overlap
        assert set(split.train_idx.tolist()).isdisjoint(split.test_idx.tolist())


def test_loso_default_cap_is_4000():
    """The matched LOSO cap is the locked 4000 (was Fixed=4k / EEGNet=full / CCB=800)."""
    assert DEFAULT_TRAIN_CAP["loso"] == 4000
    # With a pool under 4000 the default cap is a no-op (full held-in pool used by all).
    cohort = [_subject(s, n=200, seed=s) for s in range(1, 5)]  # pool 600
    for _, split in matched_loso(cohort):  # no cap arg -> default 4000
        assert len(split.train_idx) == 600


# --- Within-CV: fixed fold partition, independent of algorithm seed -----------


def test_within_cv_fold_partition_is_fixed_and_deterministic():
    data = _subject(1, n=200)
    a = list(matched_within_cv(data, n_splits=5))
    b = list(matched_within_cv(data, n_splits=5))
    assert len(a) == len(b) == 5
    for sa, sb in zip(a, b, strict=True):
        assert _split_eq(sa, sb)
    # The fixed fold seed equals SPLIT_SEED (so classical/FBCSP/EEGNet share folds).
    ref = list(within_subject_cv(data, n_splits=5, seed=SPLIT_SEED))
    for sm, sr in zip(a, ref, strict=True):
        assert _split_eq(sm, sr)


def test_within_cv_uncapped():
    """Within-CV pools are small; the cap must be a no-op (every train trial kept)."""
    data = _subject(1, n=200)
    for split in matched_within_cv(data, n_splits=5):
        assert len(split.train_idx) + len(split.test_idx) == 200  # 80/20 per fold, nothing dropped


# --- Deterministic-split protocols: uncapped, identical for all methods -------


def test_session_and_cross_session_uncapped_and_deterministic():
    data = _subject(1, n=200, n_sessions=2)  # S1, S2
    s1 = matched_session_split(data, train_session_idx=0, test_session_idx=1)
    s2 = matched_session_split(data, train_session_idx=0, test_session_idx=1)
    assert _split_eq(s1, s2)
    assert len(s1.train_idx) == 100  # all of session S1, uncapped
    cs = matched_cross_session(data, train_sessions=["S1"], test_sessions=["S2"])
    assert len(cs.train_idx) == 100 and len(cs.test_idx) == 100


# --- The shared subsampler primitive -----------------------------------------


def test_subsample_is_stratified_and_seed_fixed():
    rng_idx = np.arange(10000)
    y = np.array([0] * 3000 + [1] * 7000)
    sp = Split(train_idx=rng_idx, test_idx=np.arange(10000, 10100), name="x")
    a = subsample_train(sp, y, 4000)
    b = subsample_train(sp, y, 4000)
    assert np.array_equal(a.train_idx, b.train_idx)  # seed-fixed -> identical for all methods
    assert len(a.train_idx) <= 4000
    assert abs((y[a.train_idx] == 1).mean() - 0.70) < 0.02  # class proportions preserved
    assert np.array_equal(a.test_idx, sp.test_idx)  # test never touched


def test_subsample_noop_when_under_cap():
    y = np.array([0, 1] * 100)
    sp = Split(train_idx=np.arange(200), test_idx=np.arange(200, 210), name="x")
    assert np.array_equal(subsample_train(sp, y, 4000).train_idx, sp.train_idx)
    assert np.array_equal(subsample_train(sp, y, 0).train_idx, sp.train_idx)  # 0 = uncapped


def test_split_seed_is_fixed_constant():
    assert SPLIT_SEED == 42
