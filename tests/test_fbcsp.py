"""Unit tests for thesis.baselines.fbcsp.

Uses synthetic EEG-shaped tensors with an injected class-dependent band-power
imbalance so the classifier has a real, reproducible signal to latch onto —
if FBCSP can't beat chance on this, something's wrong with the pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from thesis.baselines.fbcsp import DEFAULT_BANDS, FBCSP


@pytest.fixture
def synthetic_mi(seed: int = 0):
    """Two classes differ in 10–14 Hz band power on distinct channels."""
    rng = np.random.default_rng(seed)
    n_trials_per_class = 60
    n_channels = 3
    n_samples = 1000
    sfreq = 250.0
    t = np.arange(n_samples) / sfreq

    # class 0: more power at 12 Hz on channel 0
    # class 1: more power at 12 Hz on channel 2
    X0 = rng.standard_normal((n_trials_per_class, n_channels, n_samples)) * 0.5
    X0[:, 0, :] += 2.0 * np.sin(2 * np.pi * 12 * t)

    X1 = rng.standard_normal((n_trials_per_class, n_channels, n_samples)) * 0.5
    X1[:, 2, :] += 2.0 * np.sin(2 * np.pi * 12 * t)

    X = np.concatenate([X0, X1], axis=0)
    y = np.array(["left"] * n_trials_per_class + ["right"] * n_trials_per_class)

    shuffle = rng.permutation(len(y))
    return X[shuffle], y[shuffle], sfreq


def test_default_bands_span_4_to_40_hz():
    assert DEFAULT_BANDS[0][0] == 4
    assert DEFAULT_BANDS[-1][1] == 40
    assert len(DEFAULT_BANDS) == 9


def test_fbcsp_beats_chance_on_synthetic_mi(synthetic_mi):
    X, y, sfreq = synthetic_mi
    split = len(y) // 2
    clf = FBCSP(sfreq=sfreq, n_components=2).fit(X[:split], y[:split])
    acc = clf.score(X[split:], y[split:])
    assert acc > 0.75, f"expected > 0.75 on synthetic MI; got {acc:.3f}"


def test_fbcsp_feature_dim_matches_nbands_times_ncomponents(synthetic_mi):
    # 3 channels → CSP can return at most 3 components; use 2 to stay in range.
    X, y, sfreq = synthetic_mi
    n_components = 2
    clf = FBCSP(sfreq=sfreq, n_components=n_components).fit(X, y)
    feats = clf._transform_features(X)
    assert feats.shape == (X.shape[0], len(DEFAULT_BANDS) * n_components)
