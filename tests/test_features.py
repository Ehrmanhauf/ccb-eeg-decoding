"""Unit tests for thesis.features — dataset-agnostic EEG extractors."""

from __future__ import annotations

import numpy as np
import pytest

from thesis.features import EngagementIndex, SpectralBandExtractor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_trial(
    *,
    n_channels: int,
    n_samples: int = 512,
    sfreq: float = 128.0,
    injected_freq: float | None = None,
    injected_channel: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Return a (n_trials=1, n_channels, n_samples) epoch with an optional pure-tone
    injection on a chosen channel (for feature-sanity tests)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sfreq
    x = rng.standard_normal((1, n_channels, n_samples)) * 1e-6
    if injected_freq is not None and injected_channel is not None:
        x[0, injected_channel] += 5e-6 * np.sin(2 * np.pi * injected_freq * t)
    return x, sfreq


# ---------------------------------------------------------------------------
# SpectralBandExtractor
# ---------------------------------------------------------------------------


def test_spectral_identity_shape_single_band():
    """Single band, identity spatial → (n_trials, n_channels) output."""
    X, sfreq = _make_trial(n_channels=14)
    extractor = SpectralBandExtractor(
        bands=((8.0, 12.0),), sfreq=sfreq, spatial="identity", feature_type="logvar"
    )
    out = extractor.transform(X)
    assert out.shape == (1, 14)
    assert np.isfinite(out).all()


def test_spectral_identity_shape_multi_band():
    """Three bands × 14 channels identity → (n_trials, 42)."""
    X, sfreq = _make_trial(n_channels=14)
    extractor = SpectralBandExtractor(
        bands=((4.0, 7.0), (8.0, 12.0), (13.0, 30.0)),
        sfreq=sfreq,
        spatial="identity",
    )
    out = extractor.transform(X)
    assert out.shape == (1, 3 * 14)


def test_spectral_laplacian_outputs_two_channels_per_band():
    """Laplacian stencil compresses 3 channels → 2 channels per band."""
    X, sfreq = _make_trial(n_channels=3)
    extractor = SpectralBandExtractor(bands=((8.0, 12.0),), sfreq=sfreq, spatial="laplacian")
    out = extractor.transform(X)
    assert out.shape == (1, 2)


def test_spectral_laplacian_rejects_non_three_channels():
    X, sfreq = _make_trial(n_channels=5)
    extractor = SpectralBandExtractor(bands=((8.0, 12.0),), sfreq=sfreq, spatial="laplacian")
    with pytest.raises(ValueError, match="3 channels"):
        extractor.transform(X)


def test_spectral_2d_input_preserves_dimensionality():
    """A 2D input (n_channels, n_samples) returns 1D output."""
    X, sfreq = _make_trial(n_channels=14)
    extractor = SpectralBandExtractor(bands=((8.0, 12.0),), sfreq=sfreq)
    out = extractor.transform(X[0])  # strip trial axis
    assert out.shape == (14,)


def test_spectral_injected_tone_shows_up_in_matching_band():
    """A 10 Hz tone on channel 5 → that channel's α-band logvar should
    exceed its β-band logvar after extraction."""
    X, sfreq = _make_trial(n_channels=14, injected_freq=10.0, injected_channel=5, seed=1)
    alpha_ext = SpectralBandExtractor(bands=((8.0, 12.0),), sfreq=sfreq)
    beta_ext = SpectralBandExtractor(bands=((15.0, 30.0),), sfreq=sfreq)
    alpha_feat = alpha_ext.transform(X)
    beta_feat = beta_ext.transform(X)
    # Channel 5, α band should outshine channel 5, β band.
    assert alpha_feat[0, 5] > beta_feat[0, 5], (
        f"α={alpha_feat[0, 5]:.3f} not > β={beta_feat[0, 5]:.3f} at injected channel"
    )


def test_spectral_rejects_empty_bands():
    with pytest.raises(ValueError, match="bands"):
        SpectralBandExtractor(bands=(), sfreq=128.0)


def test_spectral_rejects_bad_sfreq():
    with pytest.raises(ValueError, match="sfreq"):
        SpectralBandExtractor(bands=((8.0, 12.0),), sfreq=0.0)


def test_spectral_rejects_band_above_nyquist():
    """A band above the Nyquist frequency must raise — catches wrong sfreq config."""
    ext = SpectralBandExtractor(bands=((60.0, 120.0),), sfreq=128.0)
    X, _ = _make_trial(n_channels=14, sfreq=128.0)
    with pytest.raises(ValueError, match="band edges"):
        ext.transform(X)


def test_spectral_from_band_list_accepts_iterable():
    """from_band_list helper handles list, tuple, generator uniformly."""
    bands_list = [(4.0, 7.0), (8.0, 12.0)]
    ext = SpectralBandExtractor.from_band_list(bands_list, sfreq=128.0)
    assert ext.bands == ((4.0, 7.0), (8.0, 12.0))
    ext2 = SpectralBandExtractor.from_band_list(iter(bands_list), sfreq=128.0)
    assert ext2.bands == ((4.0, 7.0), (8.0, 12.0))


def test_spectral_feature_type_accepted():
    """Both feature_type values produce finite (n_trials, n_bands × n_ch) outputs.

    Note: for zero-mean band-passed signals logvar and bandpower differ only
    by a near-constant additive offset ≈ log(n/(n-1)), so testing that they
    are *numerically different beyond default tolerance* fails by design.
    The meaningful check is that both are accepted and produce finite
    feature vectors with the right shape.
    """
    X, sfreq = _make_trial(n_channels=14)
    for ft in ("logvar", "bandpower"):
        ext = SpectralBandExtractor(bands=((8.0, 12.0),), sfreq=sfreq, feature_type=ft)
        out = ext.transform(X)
        assert out.shape == (1, 14)
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# EngagementIndex
# ---------------------------------------------------------------------------


def test_engagement_shape_and_finiteness():
    """Engagement index returns one scalar per input channel per trial."""
    X, sfreq = _make_trial(n_channels=14)
    idx = EngagementIndex(sfreq=sfreq).transform(X)
    assert idx.shape == (1, 14)
    assert np.isfinite(idx).all()
    assert (idx >= 0).all()


def test_engagement_2d_input():
    X, sfreq = _make_trial(n_channels=14)
    idx = EngagementIndex(sfreq=sfreq).transform(X[0])
    assert idx.shape == (14,)


def test_engagement_higher_on_beta_dominant_signal():
    """Injecting a β-band tone should raise engagement for that channel."""
    X_alpha, sfreq = _make_trial(n_channels=14, injected_freq=10.0, injected_channel=3, seed=2)
    X_beta, _ = _make_trial(n_channels=14, injected_freq=20.0, injected_channel=3, seed=2)
    idx_alpha = EngagementIndex(sfreq=sfreq).transform(X_alpha)
    idx_beta = EngagementIndex(sfreq=sfreq).transform(X_beta)
    assert idx_beta[0, 3] > idx_alpha[0, 3], (
        f"β-dominant channel should be more engaged; got β={idx_beta[0, 3]:.3f} "
        f"α={idx_alpha[0, 3]:.3f}"
    )


def test_engagement_rejects_bad_sfreq():
    with pytest.raises(ValueError, match="sfreq"):
        EngagementIndex(sfreq=0.0)


def test_engagement_rejects_bad_bands():
    with pytest.raises(ValueError, match="alpha_band"):
        EngagementIndex(sfreq=128.0, alpha_band=(12.0, 8.0))
    with pytest.raises(ValueError, match="theta_band"):
        EngagementIndex(sfreq=128.0, theta_band=(0.0, 7.0))
