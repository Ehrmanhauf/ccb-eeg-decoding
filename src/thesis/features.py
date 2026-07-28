"""Dataset-agnostic EEG feature extractors.

Phase-5 Stage-2 addition. Provides building blocks reusable outside the
motor-imagery pipeline — specifically for the cognitive-load arm bank
we will assemble on STEW. The MI-specific FBCSP pipeline in
:mod:`thesis.baselines.fbcsp` is **not** modified: it continues to use
CSP (`mne.decoding.CSP`) directly for the 2a/2b baselines.

Modules:

- :class:`SpectralBandExtractor` — bandpass a raw epoch into one or more
  sub-bands, apply a non-CSP spatial transform (identity or C3-Cz / C4-Cz
  Laplacian stencil — the same one used in the MI arm bank), and collapse
  the time axis into log-variance or log-bandpower per (band × output
  channel). Works on any channel count and any band list. **Not a sklearn
  estimator** — stateless, no fit step required.

- :class:`EngagementIndex` — beta / (alpha + theta) engagement index per
  channel. A widely-used cognitive-load / alertness marker. ref: Pope,
  Bogart & Bartolome (1995) *Biological Psychology*, "Biocybernetic system
  evaluates indices of operator engagement in automated task."

Both classes are intentionally minimal — the bandit runner reuses
them by composition through its arm enumeration, which is dataset-aware.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import cheby2, sosfiltfilt

SpatialMode = Literal["identity", "laplacian"]
FeatureType = Literal["logvar", "bandpower"]


def _chebyshev_bandpass(
    data: np.ndarray, fmin: float, fmax: float, sfreq: float, order: int = 4, rs: float = 40.0
) -> np.ndarray:
    """Zero-phase Chebyshev Type-II bandpass (same design as FBCSP module).

    Matches [src/thesis/baselines/fbcsp.py](src/thesis/baselines/fbcsp.py)
    but re-implemented here so cognitive-load arms don't depend on an
    MI-specific module.
    """
    nyq = sfreq / 2.0
    if not (0 < fmin < fmax < nyq):
        raise ValueError(f"band edges must satisfy 0 < {fmin} < {fmax} < {nyq}")
    sos = cheby2(N=order, rs=rs, Wn=[fmin / nyq, fmax / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def _laplacian_stencil_3ch(X: np.ndarray) -> np.ndarray:
    """2-output Laplacian for the canonical 3-channel bipolar layout (C3, Cz, C4).

    Matches the stencil in :mod:`thesis.ccb.arms`; re-exported here so the
    cognitive-load features module does not import from the MI arm bank.
    Input (..., 3, n_samples) → output (..., 2, n_samples): C3-Cz, C4-Cz.
    """
    if X.shape[-2] != 3:
        raise ValueError(f"Laplacian stencil expects 3 channels; got shape {X.shape}")
    ch3_prime = X[..., 0, :] - X[..., 1, :]
    ch4_prime = X[..., 2, :] - X[..., 1, :]
    return np.stack([ch3_prime, ch4_prime], axis=-2)


@dataclass(frozen=True)
class SpectralBandExtractor:
    """Bandpass → (optional spatial) → (log-variance | log-bandpower) collapse.

    The extractor returns a feature matrix of shape ``(n_trials, n_bands *
    n_output_channels)`` where ``n_output_channels`` depends on ``spatial``:
    ``identity`` keeps every input channel; ``laplacian`` applies the C3-Cz /
    C4-Cz stencil (only valid when the input has exactly three channels,
    interpreted in the canonical order).

    Use cases:
    - Cognitive-load arms where bandpower on θ / α / β at frontal / parietal
      sites is the expected signal (no class-discriminative spatial filter).
    - As a fallback feature map for low-channel pipelines where CSP's
      generalized-eigenvalue solve would be unstable.

    This is NOT a replacement for CSP-based MI-BCI classification; the
    FBCSP pipeline stays in :mod:`thesis.baselines.fbcsp` unchanged.

    Parameters
    ----------
    bands : one or more ``(fmin, fmax)`` tuples in Hz.
    sfreq : sampling rate (Hz).
    spatial : ``"identity"`` (default, pass-through) or ``"laplacian"``.
    feature_type : ``"logvar"`` (default, log-variance per band) or
        ``"bandpower"`` (log mean-squared amplitude per band). Equivalent up
        to an additive constant when the signal is zero-mean.
    """

    bands: tuple[tuple[float, float], ...]
    sfreq: float
    spatial: SpatialMode = "identity"
    feature_type: FeatureType = "logvar"

    @classmethod
    def from_band_list(
        cls,
        bands: Iterable[tuple[float, float]],
        sfreq: float,
        *,
        spatial: SpatialMode = "identity",
        feature_type: FeatureType = "logvar",
    ) -> SpectralBandExtractor:
        """Convenience: accept any iterable of bands (list, tuple, generator)."""
        return cls(
            bands=tuple((float(lo), float(hi)) for lo, hi in bands),
            sfreq=float(sfreq),
            spatial=spatial,
            feature_type=feature_type,
        )

    def __post_init__(self) -> None:  # dataclass validation
        if not self.bands:
            raise ValueError("bands must be non-empty")
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive; got {self.sfreq}")
        if self.spatial not in ("identity", "laplacian"):
            raise ValueError(f"unknown spatial mode {self.spatial!r}")
        if self.feature_type not in ("logvar", "bandpower"):
            raise ValueError(f"unknown feature_type {self.feature_type!r}")

    def _apply_spatial(self, X: np.ndarray) -> np.ndarray:
        if self.spatial == "identity":
            return X
        return _laplacian_stencil_3ch(X)

    def _collapse(self, X: np.ndarray) -> np.ndarray:
        """Time-axis collapse: logvar or log-mean-squared-amplitude."""
        if self.feature_type == "logvar":
            var = np.var(X, axis=-1, ddof=1)
            return np.log(var + 1e-12)
        power = np.mean(X**2, axis=-1)
        return np.log(power + 1e-12)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract features.

        Parameters
        ----------
        X : shape ``(n_trials, n_channels, n_samples)`` or
            ``(n_channels, n_samples)``.

        Returns
        -------
        features : shape ``(n_trials, n_bands * n_output_channels)`` or
            ``(n_bands * n_output_channels,)`` — matches the input
            dimensionality convention.
        """
        X = np.asarray(X, dtype=float)
        squeeze_trial = X.ndim == 2
        if squeeze_trial:
            X = X[np.newaxis]
        if X.ndim != 3:
            raise ValueError(f"X must be 2- or 3-D; got shape {X.shape}")

        bands_feats: list[np.ndarray] = []
        for fmin, fmax in self.bands:
            X_filt = _chebyshev_bandpass(X, fmin, fmax, self.sfreq)
            X_spatial = self._apply_spatial(X_filt)
            bands_feats.append(self._collapse(X_spatial))
        out = np.concatenate(bands_feats, axis=-1)
        if squeeze_trial:
            out = out[0]
        return out


@dataclass(frozen=True)
class EngagementIndex:
    """Pope-Bogart-Bartolome β / (α + θ) engagement index per channel.

    A classical cognitive-load / alertness marker: higher values correlate
    with higher task engagement. ref: Pope, A. T., Bogart, E. H., &
    Bartolome, D. S. (1995). *Biocybernetic system evaluates indices of
    operator engagement in automated task.* Biological Psychology, 40,
    187–195.

    Computed from zero-phase Chebyshev-II bandpassed signals to be
    consistent with :class:`SpectralBandExtractor`.

    Output shape: one scalar per input channel per trial; broadcasts
    `(n_trials, n_channels, n_samples)` → `(n_trials, n_channels)`.
    """

    sfreq: float
    alpha_band: tuple[float, float] = (8.0, 12.0)
    theta_band: tuple[float, float] = (4.0, 7.0)
    beta_band: tuple[float, float] = (13.0, 30.0)

    def __post_init__(self) -> None:
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive; got {self.sfreq}")
        for name, (lo, hi) in (
            ("alpha_band", self.alpha_band),
            ("theta_band", self.theta_band),
            ("beta_band", self.beta_band),
        ):
            if not (0 < lo < hi):
                raise ValueError(f"{name} must satisfy 0 < lo < hi; got {(lo, hi)}")

    @staticmethod
    def _bandpower(X: np.ndarray) -> np.ndarray:
        return np.mean(X**2, axis=-1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        squeeze_trial = X.ndim == 2
        if squeeze_trial:
            X = X[np.newaxis]
        if X.ndim != 3:
            raise ValueError(f"X must be 2- or 3-D; got shape {X.shape}")

        alpha = self._bandpower(_chebyshev_bandpass(X, *self.alpha_band, self.sfreq))
        theta = self._bandpower(_chebyshev_bandpass(X, *self.theta_band, self.sfreq))
        beta = self._bandpower(_chebyshev_bandpass(X, *self.beta_band, self.sfreq))
        engagement = beta / (alpha + theta + 1e-24)
        if squeeze_trial:
            engagement = engagement[0]
        return engagement


__all__: Sequence[str] = (
    "SpectralBandExtractor",
    "EngagementIndex",
    "SpatialMode",
    "FeatureType",
)
