r"""Cognitive-load band-power baseline classifier (Phase B, 2026-05-19).

Baseline B2 of the locked CL evaluation design (Strategy B,
`design-doc/ccb-formulation.md` §2.7):

    Features:    per-channel log-band-power in θ (4--7 Hz), α (8--12 Hz),
                 and β (13--30 Hz), plus optional channel-role-aware
                 derived aggregates (frontal-θ, parietal-α, frontal-α
                 asymmetry, engagement index).
    Classifier:  scikit-learn shrinkage Linear Discriminant Analysis
                 (LinearDiscriminantAnalysis with solver='lsqr',
                 shrinkage='auto') --- the same classifier as in
                 ``thesis.baselines.fbcsp.FBCSP``, so the FBCSP-vs-
                 band-power contrast isolates the *feature pipeline*
                 effect, not classifier variance.

Together with FBCSP+sLDA (Baseline B1) this forms a 2 × 2 design across
(feature family × decision regime) for the CL paradigm; see the Phase 2
plan for the full motivation.

Justification for the feature set:

- **θ / α / β band-power per channel** is the canonical EEG-based CL
  feature set in the published literature. Lim et al. 2018
  [`lim2018stew`] use frequency-domain features on STEW in their
  reference $\kappa = 0.46$ baseline; the same band decomposition is
  the substrate of countless workload-classification studies. Using it
  here lets the BandPowerCL baseline serve as a paradigm-appropriate
  comparator against which the FBCSP+sLDA B1 baseline (paradigm-
  consistent with MI) and the CCB framework (Phase D / Phase E) can be
  characterised.
- **Frontal θ, parietal α, frontal-α asymmetry, engagement index**
  are the four derived aggregates encoded in the workload context
  vector (`thesis.ccb.context_cl.compute_context_workload`). They are
  drawn from the established cognitive-workload literature:
  frontal-midline θ increases with working-memory load; parietal-α
  decreases with attentional engagement; frontal-α asymmetry
  correlates with approach / withdrawal motivation; and the
  engagement index $\beta / (\alpha + \theta)$ tracks task engagement
  (Pope, Bogart & Bartolome 1995; see also Klimesch 1999 reviews).
  Including them as explicit features --- rather than only as
  bandit-context features --- ensures the fixed-pipeline baseline is
  not handicapped relative to the CCB on derived-feature access.

Channel-role mappings are supplied per-dataset at construction time:
``STEW_CHANNEL_ROLES`` lives in ``scripts/run_ccb_stew.py`` and
``WAUC_CHANNEL_ROLES`` lives in ``thesis.data.wauc_load`` (both added
for Phase A / B / D / E). When ``channel_roles=None`` the derived
aggregates are not appended, and the feature vector is just the per-
channel band-power grid; this is the right default for any dataset
whose role-aware indices are not yet defined.

ref: `lim2018stew`, `albuquerque2020wauc`,
``design-doc/ccb-formulation.md`` §2.7,
``src/thesis/ccb/context_cl.py``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.signal import welch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Default CL band schedule. Verified against `thesis.ccb.context_cl`
# (theta 4--7, alpha 8--12, beta 13--30) and against the broader
# workload-classification literature [lim2018stew, ang2012fbcsp
# §2.1's sub-band edges as a sanity-check; we use the canonical
# CL ranges here rather than Ang's 9-band MI bank, since this baseline
# is intentionally domain-specific].
DEFAULT_CL_BANDS: tuple[tuple[float, float], ...] = (
    (4.0, 7.0),    # θ
    (8.0, 12.0),   # α
    (13.0, 30.0),  # β
)

# Floor for log-bandpower (avoid log(0) on a numerically empty band).
_LOG_FLOOR: float = 1e-20

# Welch window cap; matches `thesis.ccb.context_cl._WELCH_NPERSEG_CAP`
# so derived aggregates are computed on the same nperseg as the
# bandit-context features.
_WELCH_NPERSEG_CAP: int = 256


def _welch_psd_per_channel(epoch: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD per channel for one trial.

    Returns ``(freqs, psd)`` with ``psd`` shape ``(n_channels, n_freqs)``.
    """
    n_samples = epoch.shape[-1]
    nperseg = min(_WELCH_NPERSEG_CAP, n_samples)
    freqs, psd = welch(
        epoch,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        axis=-1,
        scaling="density",
    )
    return freqs, psd


def _log_band_power(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Log of band-integrated power per channel."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        raise ValueError(f"no PSD bins in [{fmin}, {fmax}] Hz; check sfreq / nperseg.")
    band_power = psd[..., mask].sum(axis=-1)
    return np.log(np.maximum(band_power, _LOG_FLOOR))


def _safe_mean_across_indices(per_channel: np.ndarray, indices: list[int] | None) -> float:
    """Mean of ``per_channel`` over ``indices``; 0.0 if empty / missing."""
    if not indices:
        return 0.0
    return float(np.mean(per_channel[indices]))


class BandPowerCL(BaseEstimator, ClassifierMixin):
    """Cognitive-load band-power + shrinkage-LDA baseline.

    Parameters
    ----------
    sfreq : sampling rate of the input epochs (Hz). Default 250.0 to
        match the rest of the pipeline.
    bands : sequence of ``(fmin, fmax)`` tuples. Default ``DEFAULT_CL_BANDS``
        (θ, α, β).
    channel_roles : dataset-specific role mapping, e.g.
        ``{"frontal": [0, 1, 2, 3], "parietal": [5, 8], "f3": [2],
        "f4": [11]}``. When ``None`` (default) the derived aggregates
        are not appended and the feature vector is just per-channel
        band-power.
    """

    def __init__(
        self,
        *,
        sfreq: float = 250.0,
        bands: tuple[tuple[float, float], ...] = DEFAULT_CL_BANDS,
        channel_roles: Mapping[str, list[int]] | None = None,
    ):
        self.sfreq = sfreq
        self.bands = bands
        self.channel_roles = channel_roles

    def _trial_features(self, epoch: np.ndarray) -> np.ndarray:
        """Compute the feature vector for one ``(n_channels, n_samples)`` trial."""
        freqs, psd = _welch_psd_per_channel(epoch, self.sfreq)

        # Per-channel × per-band log-power. Stored band-by-channel so the
        # flatten order is (band0_ch0, band0_ch1, ..., band1_ch0, ...).
        per_band: list[np.ndarray] = []
        for fmin, fmax in self.bands:
            per_band.append(_log_band_power(freqs, psd, fmin, fmax))
        per_band_arr = np.stack(per_band, axis=0)  # (n_bands, n_channels)
        flat = per_band_arr.flatten()

        if self.channel_roles is None:
            return flat

        # Derived aggregates (only when channel_roles supplied).
        theta_per_ch = per_band_arr[0]  # bands[0] is θ
        alpha_per_ch = per_band_arr[1]  # bands[1] is α
        beta_per_ch = per_band_arr[2]   # bands[2] is β

        frontal = list(self.channel_roles.get("frontal", []))
        parietal = list(self.channel_roles.get("parietal", []))
        f3 = list(self.channel_roles.get("f3", []))
        f4 = list(self.channel_roles.get("f4", []))

        frontal_theta = _safe_mean_across_indices(theta_per_ch, frontal)
        parietal_alpha = _safe_mean_across_indices(alpha_per_ch, parietal)
        if f3 and f4:
            frontal_alpha_asym = float(alpha_per_ch[f4[0]] - alpha_per_ch[f3[0]])
        else:
            frontal_alpha_asym = 0.0
        # Engagement index: β / (α + θ) — averaged across channels.
        # Compute on the linear-domain band power to keep the ratio
        # interpretable; clip the denominator for numerical safety.
        theta_lin = np.exp(theta_per_ch)
        alpha_lin = np.exp(alpha_per_ch)
        beta_lin = np.exp(beta_per_ch)
        engagement = float(np.mean(beta_lin / np.maximum(alpha_lin + theta_lin, _LOG_FLOOR)))

        derived = np.array([frontal_theta, parietal_alpha, frontal_alpha_asym, engagement])
        return np.concatenate([flat, derived])

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Compute per-trial feature vectors for an ``(n_trials, n_channels, n_samples)`` array."""
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"expected 3-D (n_trials, n_channels, n_samples); got {X.shape}")
        feats = np.stack([self._trial_features(X[i]) for i in range(X.shape[0])], axis=0)
        return feats

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BandPowerCL":
        y = np.asarray(y)
        feats = self._extract_features(X)
        # Shrinkage LDA (lsqr + Ledoit-Wolf automatic shrinkage). Same
        # classifier as in ``thesis.baselines.fbcsp.FBCSP`` to isolate
        # the feature-pipeline contrast between B1 and B2.
        self.lda_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.lda_.fit(feats, y)
        self.classes_ = self.lda_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.lda_.predict(self._extract_features(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.lda_.predict_proba(self._extract_features(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())
