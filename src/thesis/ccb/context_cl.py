"""Cognitive-load / workload context vector for the CCB policy.

Counterpart of :mod:`thesis.ccb.context` (which is MI-specific: μ/β ERD
on 3 channels around C3/Cz/C4). This module targets workload-labelled
EEG on higher-channel consumer devices (Emotiv EPOC / STEW, 14 channels;
SEED-VIG, 17 channels).

Feature list (workload-specific, grouped by brain region rather than
hardcoded to MI topography):

  θ (4–7 Hz) mean log-bandpower    — rises with sustained mental effort
  α (8–12 Hz) mean log-bandpower    — drops during task engagement
  β (13–30 Hz) mean log-bandpower   — rises with alertness
  frontal θ log-bandpower          — operator-workload marker (Fp1/Fp2/AF3/AF4 average)
  parietal α log-bandpower         — cognitive-load marker (P3/P4/Pz average)
  frontal α asymmetry              — F3 − F4 alpha log-bandpower
  engagement index                 — β / (α + θ) averaged over all channels
  artifact flag                    — 1 if |epoch| > 150 μV (looser than MI's 100 μV
                                     because Emotiv EPOC dynamic range differs)
  bias                             — constant 1.0 (linear-CB convention)
  running mean reward per arm family — running tail, length = n_recent_arms
                                       (set by the caller / runner).

The exact indices depend on the provided ``channel_roles`` mapping, so
this context is **channel-layout-aware** by design. If a dataset lacks
some regions (e.g. no frontal electrodes), the corresponding features
are silently zeroed and the caller is expected to document it.

ref: Pope, Bogart & Bartolome 1995 (engagement index); Klimesch 1999
(alpha suppression under attention); Berka et al. 2007 (workload
markers in consumer EEG). Full bib entries in
``design-doc/references.bib`` (pending Stage-5-Commit-5.14 audit).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.signal import welch

# Canonical band edges (Hz). Workload-specific cuts vs the MI context
# module: θ here is 4–7 (not 4–8), β here is 13–30 (not 16–24), because
# workload literature uses these wider bands.
_THETA: tuple[float, float] = (4.0, 7.0)
_ALPHA: tuple[float, float] = (8.0, 12.0)
_BETA: tuple[float, float] = (13.0, 30.0)

_ARTIFACT_VOLTS_WORKLOAD: float = 150e-6  # Emotiv EPOC dynamic range is wider
_WELCH_NPERSEG_CAP: int = 256


def _welch_psd(epoch: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(_WELCH_NPERSEG_CAP, epoch.shape[-1])
    return welch(epoch, fs=sfreq, nperseg=nperseg, axis=-1)


def _log_band_from_psd(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        raise ValueError(f"No PSD bins in [{fmin}, {fmax}] Hz; check sfreq / nperseg.")
    band = psd[..., mask].mean(axis=-1)
    return np.log(band + 1e-24)


def _safe_mean_across_indices(per_channel: np.ndarray, indices: list[int] | None) -> float:
    """Mean across the given channel indices; 0.0 if the list is empty or None."""
    if not indices:
        return 0.0
    return float(per_channel[indices].mean())


def context_dim_workload(
    *,
    include_recent_rewards: bool = True,
    n_recent_arms: int = 3,
) -> int:
    """Return the workload-context dimensionality.

    Base features: 9 (θ mean, α mean, β mean, frontal θ, parietal α,
    frontal-α asymmetry, engagement, artifact, bias). Plus recent-reward
    tail of length ``n_recent_arms`` when ``include_recent_rewards``.
    """
    base = 9
    return base + (n_recent_arms if include_recent_rewards else 0)


def compute_context_workload(
    epoch: np.ndarray,
    *,
    sfreq: float,
    channel_roles: Mapping[str, list[int]],
    recent_arm_rewards: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the workload-context vector for one trial.

    Parameters
    ----------
    epoch : shape ``(n_channels, n_samples)`` — any channel count.
    sfreq : sampling rate in Hz.
    channel_roles : mapping from role name to a list of channel indices.
        Recognised keys (missing keys silently zero the corresponding
        feature):

          - ``"frontal"``: e.g. ``[idx_Fp1, idx_Fp2, idx_AF3, idx_AF4]``
            used for frontal θ.
          - ``"parietal"``: indices for parietal α (P3/P4/Pz / similar).
          - ``"f3"`` and ``"f4"``: **single**-element lists for left /
            right frontal alpha asymmetry (first element is taken).

        All other keys are ignored here. (STEW and SEED-VIG will supply
        their own mappings at load time in Stage 5.8.)
    recent_arm_rewards : optional array of running-mean rewards per arm
        family. If provided, appended to the context tail.

    Returns
    -------
    ``np.ndarray`` of shape ``(d,)``.
    """
    if epoch.ndim != 2:
        raise ValueError(
            f"compute_context_workload expects a 2-D epoch (n_ch, n_samples); got {epoch.shape}"
        )
    freqs, psd = _welch_psd(epoch, sfreq)  # (n_ch, n_freqs)

    theta_per_ch = _log_band_from_psd(freqs, psd, *_THETA)
    alpha_per_ch = _log_band_from_psd(freqs, psd, *_ALPHA)
    beta_per_ch = _log_band_from_psd(freqs, psd, *_BETA)

    theta_mean = float(theta_per_ch.mean())
    alpha_mean = float(alpha_per_ch.mean())
    beta_mean = float(beta_per_ch.mean())
    frontal_theta = _safe_mean_across_indices(theta_per_ch, list(channel_roles.get("frontal", [])))
    parietal_alpha = _safe_mean_across_indices(
        alpha_per_ch, list(channel_roles.get("parietal", []))
    )

    f3_list = list(channel_roles.get("f3", []))
    f4_list = list(channel_roles.get("f4", []))
    if f3_list and f4_list:
        asymmetry = float(alpha_per_ch[f3_list[0]] - alpha_per_ch[f4_list[0]])
    else:
        asymmetry = 0.0

    # Engagement index in linear (not log) bandpower averaged over
    # channels, then logged for scale parity with the bandpower features.
    alpha_lin = float(np.exp(alpha_per_ch).mean())
    theta_lin = float(np.exp(theta_per_ch).mean())
    beta_lin = float(np.exp(beta_per_ch).mean())
    engagement = float(np.log(beta_lin / (alpha_lin + theta_lin + 1e-24) + 1e-24))

    artifact = float(np.max(np.abs(epoch)) > _ARTIFACT_VOLTS_WORKLOAD)
    bias = 1.0

    base = np.array(
        [
            theta_mean,
            alpha_mean,
            beta_mean,
            frontal_theta,
            parietal_alpha,
            asymmetry,
            engagement,
            artifact,
            bias,
        ],
        dtype=float,
    )

    if recent_arm_rewards is not None:
        rec = np.asarray(recent_arm_rewards, dtype=float).ravel()
        return np.concatenate([base, rec], axis=0)
    return base
