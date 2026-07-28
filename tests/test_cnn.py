"""Offline structural tests for the EEGNet baseline (auto-skipped if torch is absent)."""

from __future__ import annotations

import numpy as np
import pytest

from thesis.baselines.cnn import _TORCH, EEGNet

pytestmark = pytest.mark.skipif(not _TORCH, reason="torch not installed (benchmark extra)")


def _lateralized(seed: int = 0, n: int = 60, sf: float = 250.0, t_samp: int = 400):
    rng = np.random.default_rng(seed)
    t = np.arange(t_samp) / sf
    def make(ch):
        X = rng.standard_normal((n, 8, t_samp)) * 1e-6
        X[:, ch] += 5e-6 * np.sin(2 * np.pi * 12 * t)
        return X
    X = np.concatenate([make(0), make(4)])
    y = np.array(["left"] * n + ["right"] * n)
    return X, y


def test_fit_predict_shape_and_labels():
    X, y = _lateralized()
    clf = EEGNet(sfreq=250.0, epochs=15, seed=42).fit(X, y)
    yp = clf.predict(X)
    assert yp.shape == (len(y),)
    assert set(yp.tolist()) <= {"left", "right"}


def test_decodes_above_chance_and_is_deterministic():
    X, y = _lateralized(seed=1)
    perm = np.random.default_rng(1).permutation(len(y))
    tr, te = perm[: int(0.7 * len(y))], perm[int(0.7 * len(y)):]
    p1 = EEGNet(sfreq=250.0, epochs=20, seed=7).fit(X[tr], y[tr]).predict(X[te])
    p2 = EEGNet(sfreq=250.0, epochs=20, seed=7).fit(X[tr], y[tr]).predict(X[te])
    assert (p1 == p2).all(), "same seed must give identical predictions"
    assert (p1 == y[te]).mean() > 0.6, "must beat chance on a clean lateralized signal"
