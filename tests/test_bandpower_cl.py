"""Unit tests for thesis.baselines.bandpower_cl.

Uses synthetic EEG-shaped tensors with an injected class-dependent
band-power imbalance, mirroring the test_fbcsp.py pattern. The
classifier must beat chance on a clean synthetic signal — if not,
something is wrong with the band-power pipeline before we let it
loose on STEW or WAUC.
"""

from __future__ import annotations

import numpy as np
import pytest

from thesis.baselines.bandpower_cl import (
    DEFAULT_CL_BANDS,
    BandPowerCL,
    _log_band_power,
    _safe_mean_across_indices,
    _welch_psd_per_channel,
)


@pytest.fixture
def synthetic_cl_3class(seed: int = 0):
    """Three workload classes differ in θ / α / β power across channels.

    Mimics the STEW 3-class low/medium/high target. Class identity drives
    band-specific amplitude on overlapping channels so the classifier has
    a clear signal to latch onto.
    """
    rng = np.random.default_rng(seed)
    n_per_class = 40
    n_channels = 8
    n_samples = 1000
    sfreq = 250.0
    t = np.arange(n_samples) / sfreq

    def make(amp_theta: float, amp_alpha: float, amp_beta: float) -> np.ndarray:
        X = rng.standard_normal((n_per_class, n_channels, n_samples)) * 0.5
        X[:, :, :] += amp_theta * np.sin(2 * np.pi * 6 * t)   # θ ≈ 6 Hz
        X[:, :, :] += amp_alpha * np.sin(2 * np.pi * 10 * t)  # α ≈ 10 Hz
        X[:, :, :] += amp_beta * np.sin(2 * np.pi * 20 * t)   # β ≈ 20 Hz
        return X

    X_low = make(amp_theta=0.5, amp_alpha=2.0, amp_beta=0.5)   # high α → engaged → low load
    X_med = make(amp_theta=1.0, amp_alpha=1.0, amp_beta=1.0)   # balanced
    X_high = make(amp_theta=2.0, amp_alpha=0.5, amp_beta=2.0)  # high θ + β → high load

    X = np.concatenate([X_low, X_med, X_high], axis=0)
    y = np.array(
        ["low"] * n_per_class + ["medium"] * n_per_class + ["high"] * n_per_class
    )
    shuffle = rng.permutation(len(y))
    return X[shuffle], y[shuffle], sfreq


@pytest.fixture
def synthetic_cl_binary(seed: int = 0):
    """Binary low / high workload target — mimics WAUC's `mw_labels`."""
    rng = np.random.default_rng(seed)
    n_per_class = 60
    n_channels = 8
    n_samples = 1000
    sfreq = 250.0
    t = np.arange(n_samples) / sfreq

    X_low = rng.standard_normal((n_per_class, n_channels, n_samples)) * 0.5
    X_low += 2.0 * np.sin(2 * np.pi * 10 * t)  # α dominant
    X_high = rng.standard_normal((n_per_class, n_channels, n_samples)) * 0.5
    X_high += 2.0 * np.sin(2 * np.pi * 20 * t)  # β dominant

    X = np.concatenate([X_low, X_high], axis=0)
    y = np.array(["low"] * n_per_class + ["high"] * n_per_class)
    shuffle = rng.permutation(len(y))
    return X[shuffle], y[shuffle], sfreq


def test_default_bands_are_theta_alpha_beta():
    assert DEFAULT_CL_BANDS == ((4.0, 7.0), (8.0, 12.0), (13.0, 30.0))


def test_welch_returns_per_channel_psd():
    epoch = np.random.randn(8, 1000)
    freqs, psd = _welch_psd_per_channel(epoch, sfreq=250.0)
    assert psd.shape[0] == 8
    assert psd.shape[1] == freqs.shape[0]
    assert (freqs >= 0).all()


def test_log_band_power_per_channel_shape():
    freqs = np.array([0.0, 5.0, 10.0, 20.0])
    psd = np.ones((8, 4))
    out = _log_band_power(freqs, psd, fmin=4.0, fmax=7.0)
    assert out.shape == (8,)


def test_log_band_power_floor():
    # All-zero PSD should produce a finite log (clamped to _LOG_FLOOR), not -inf.
    freqs = np.array([0.0, 5.0, 10.0])
    psd = np.zeros((4, 3))
    out = _log_band_power(freqs, psd, fmin=4.0, fmax=7.0)
    assert np.isfinite(out).all()


def test_log_band_power_empty_range_raises():
    freqs = np.array([0.0, 5.0])
    psd = np.ones((2, 2))
    with pytest.raises(ValueError, match="no PSD bins"):
        _log_band_power(freqs, psd, fmin=100.0, fmax=200.0)


def test_safe_mean_empty_indices_returns_zero():
    arr = np.array([1.0, 2.0, 3.0])
    assert _safe_mean_across_indices(arr, []) == 0.0
    assert _safe_mean_across_indices(arr, None) == 0.0


def test_feature_dim_without_channel_roles():
    n_channels, n_samples = 8, 1000
    X = np.random.randn(5, n_channels, n_samples)
    clf = BandPowerCL(sfreq=250.0)
    feats = clf._extract_features(X)
    # 5 trials × (n_channels × n_bands) — no derived aggregates.
    assert feats.shape == (5, n_channels * len(DEFAULT_CL_BANDS))


def test_feature_dim_with_channel_roles():
    n_channels, n_samples = 14, 1000
    X = np.random.randn(5, n_channels, n_samples)
    roles = {"frontal": [0, 1, 2], "parietal": [10, 11], "f3": [2], "f4": [11]}
    clf = BandPowerCL(sfreq=250.0, channel_roles=roles)
    feats = clf._extract_features(X)
    # 14 × 3 = 42 per-channel band-power + 4 derived aggregates = 46.
    assert feats.shape == (5, n_channels * len(DEFAULT_CL_BANDS) + 4)


def test_bandpower_beats_chance_on_synthetic_3class(synthetic_cl_3class):
    X, y, sfreq = synthetic_cl_3class
    split = len(y) // 2
    clf = BandPowerCL(sfreq=sfreq).fit(X[:split], y[:split])
    acc = clf.score(X[split:], y[split:])
    # 3-class chance = 1/3 ≈ 0.333; require well above chance.
    assert acc > 0.70, f"expected > 0.70 on synthetic 3-class CL; got {acc:.3f}"


def test_bandpower_beats_chance_on_synthetic_binary(synthetic_cl_binary):
    X, y, sfreq = synthetic_cl_binary
    split = len(y) // 2
    clf = BandPowerCL(sfreq=sfreq).fit(X[:split], y[:split])
    acc = clf.score(X[split:], y[split:])
    # Binary chance = 0.5.
    assert acc > 0.80, f"expected > 0.80 on synthetic binary CL; got {acc:.3f}"


def test_bandpower_with_channel_roles_classifies(synthetic_cl_binary):
    X, y, sfreq = synthetic_cl_binary
    n_channels = X.shape[1]
    roles = {
        "frontal": list(range(n_channels // 2)),
        "parietal": list(range(n_channels // 2, n_channels)),
        "f3": [0],
        "f4": [n_channels - 1],
    }
    split = len(y) // 2
    clf = BandPowerCL(sfreq=sfreq, channel_roles=roles).fit(X[:split], y[:split])
    acc = clf.score(X[split:], y[split:])
    assert acc > 0.80, f"expected > 0.80 with channel_roles; got {acc:.3f}"


def test_predict_proba_sums_to_one(synthetic_cl_binary):
    X, y, sfreq = synthetic_cl_binary
    clf = BandPowerCL(sfreq=sfreq).fit(X[:60], y[:60])
    proba = clf.predict_proba(X[60:65])
    assert proba.shape == (5, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_returns_known_classes(synthetic_cl_3class):
    X, y, sfreq = synthetic_cl_3class
    clf = BandPowerCL(sfreq=sfreq).fit(X[:60], y[:60])
    preds = clf.predict(X[60:65])
    assert set(preds).issubset({"low", "medium", "high"})


def test_classes_attribute_after_fit(synthetic_cl_3class):
    X, y, sfreq = synthetic_cl_3class
    clf = BandPowerCL(sfreq=sfreq).fit(X, y)
    assert set(clf.classes_) == {"low", "medium", "high"}


def test_wrong_input_shape_raises():
    X_2d = np.random.randn(10, 1000)  # missing channels dim
    clf = BandPowerCL(sfreq=250.0)
    with pytest.raises(ValueError, match=r"3-D"):
        clf._extract_features(X_2d)
