"""Per-trial context vector ``x_t`` for the CCB policy.

The context bundles per-trial signal statistics with a tiny running-state
memory (mean reward of each arm family over the recent past) so the bandit
can condition its arm choice on the specific trial's regime. See
`design-doc/ccb-formulation.md` §6.3.

Dimension ``d = 18`` is the resolution of `open-justifications.md`
"Context feature vector". Breakdown of the 18 components:

  indices 0–5    μ and β log-bandpower per channel (3 ch × 2 bands = 6)
  index 6        mean μ/β ratio across channels              (1)
  index 7        mean spectral entropy across channels       (1)
  index 8        mean trial variance                          (1)
  index 9        artifact flag (1 if |epoch| > 100 µV)        (1)
  index 10       θ-band mean log-power                        (1)
  index 11       low-γ-band mean log-power                    (1)
  index 12       ERD index at C3 (early vs late μ-power)      (1)
  index 13       ERD index at C4 (early vs late μ-power)      (1)
  index 14       bias term (1.0)                              (1)
  indices 15–17  running mean reward per arm family           (3)

Per-feature provenance / refs are listed in the docstrings below.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

# Canonical band edges (Hz).
# refs: `pfurtscheller2001mi` §III.B (μ/β for MI); `ang2012fbcsp` §2.1
# (θ and low-γ complements).
_MU: tuple[float, float] = (8.0, 12.0)
_BETA: tuple[float, float] = (16.0, 24.0)
_THETA: tuple[float, float] = (4.0, 8.0)
_GAMMA_LOW: tuple[float, float] = (24.0, 40.0)

# Number of arm families tracked in the recent-reward tail of the context.
# One scalar per family in {csp, laplacian, identity}.
N_ARM_FAMILIES: int = 3

# Artifact detection threshold (volts). BCI-IV-2b screening dynamic range
# is ±100 µV per desc_2b.pdf §2; anything past that is a hardware clip.
_ARTIFACT_VOLTS: float = 100e-6

# Split point (seconds, relative to cue onset) for the ERD early/late windows.
# ref: `pfurtscheller2001mi` §III.A (ERD computed as pre-vs-post μ-power).
_ERD_SPLIT_S: float = 1.0

# Welch nperseg ceiling; keeps the computation O(n_samples log n_samples)
# and avoids numerical artefacts on very short windows.
_WELCH_NPERSEG_CAP: int = 256


def context_dim(
    *,
    include_recent_rewards: bool = True,
    n_recent_arms: int = N_ARM_FAMILIES,
) -> int:
    """Return the context-vector dimension ``d``.

    With recent-reward features (default): ``d = 15 + n_recent_arms = 18``.
    Without: ``d = 15``.
    """
    base = 15  # per the module-level breakdown.
    return base + (n_recent_arms if include_recent_rewards else 0)


def _welch_psd(epoch: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD per channel. Returns (freqs, psd) with psd shape (..., n_ch, n_freqs)."""
    nperseg = min(_WELCH_NPERSEG_CAP, epoch.shape[-1])
    return welch(epoch, fs=sfreq, nperseg=nperseg, axis=-1)


def _log_band_from_psd(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Log-bandpower in [fmin, fmax] from a precomputed PSD. Preserves leading axes."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        raise ValueError(f"No PSD bins in [{fmin}, {fmax}] Hz; check sfreq / nperseg.")
    band = psd[..., mask].mean(axis=-1)
    return np.log(band + 1e-24)


def _spectral_entropy_from_psd(psd: np.ndarray) -> np.ndarray:
    """Normalized Shannon spectral entropy per channel. Shape input-preserving (drops freq axis).

    ref: `lotte2018review` §III.D (entropy features for EEG classifiers).
    """
    eps = 1e-24
    total = psd.sum(axis=-1, keepdims=True) + eps
    p = psd / total
    ent = -(p * np.log(p + eps)).sum(axis=-1)
    # Normalize to [0, 1] by dividing by log(n_bins).
    return ent / np.log(psd.shape[-1])


def _erd_index_from_epoch(epoch: np.ndarray, sfreq: float, channel: int) -> float:
    """Event-related desynchronization index at one channel.

    ERD = (P_early − P_late) / P_early, with P = mean μ-band power.
    Positive → desynchronization (expected for the contralateral side
    during MI). ref: `pfurtscheller2001mi` §III.A.
    """
    split = int(round(_ERD_SPLIT_S * sfreq))
    if split <= 0 or split >= epoch.shape[-1]:
        return 0.0
    early = epoch[channel : channel + 1, :split]
    late = epoch[channel : channel + 1, split:]
    f_e, p_e = _welch_psd(early, sfreq)
    f_l, p_l = _welch_psd(late, sfreq)
    p_mu_e = float(np.exp(_log_band_from_psd(f_e, p_e, *_MU).item()))
    p_mu_l = float(np.exp(_log_band_from_psd(f_l, p_l, *_MU).item()))
    return (p_mu_e - p_mu_l) / (p_mu_e + 1e-24)


def compute_context(
    epoch: np.ndarray,
    *,
    sfreq: float = 250.0,
    recent_arm_rewards: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the 18-dim context vector for one trial (3-ch 2b layout).

    Parameters
    ----------
    epoch : shape ``(3, n_samples)`` — 2b's 3 bipolar channels for one trial.
    sfreq : sampling rate in Hz.
    recent_arm_rewards : length-``N_ARM_FAMILIES`` array of running mean
        rewards per arm family. Pass ``None`` to drop those entries (then
        the output has dim 15).

    Returns
    -------
    ``np.ndarray`` of shape ``(d,)`` where ``d`` matches :func:`context_dim`.
    """
    if epoch.ndim != 2 or epoch.shape[0] != 3:
        raise ValueError(
            f"compute_context expects 3-channel epoch of shape (3, n_samples); got {epoch.shape}"
        )

    freqs, psd = _welch_psd(epoch, sfreq)  # (3, n_freqs)

    # 0–2: μ log-power per channel.
    mu_per_ch = _log_band_from_psd(freqs, psd, *_MU)
    # 3–5: β log-power per channel.
    beta_per_ch = _log_band_from_psd(freqs, psd, *_BETA)
    # 6: mean μ/β ratio across channels (in linear units before logging).
    mu_lin = np.exp(mu_per_ch).mean()
    beta_lin = np.exp(beta_per_ch).mean()
    mu_beta_ratio = mu_lin / (beta_lin + 1e-24)
    # 7: mean spectral entropy across channels.
    spec_ent = float(_spectral_entropy_from_psd(psd).mean())
    # 8: mean trial variance across channels (volt²).
    trial_var = float(np.var(epoch, axis=-1).mean())
    # 9: artifact flag (hardware clip check).
    artifact = float(np.max(np.abs(epoch)) > _ARTIFACT_VOLTS)
    # 10: θ-band mean log-power across channels.
    theta_mean = float(_log_band_from_psd(freqs, psd, *_THETA).mean())
    # 11: low-γ mean log-power across channels.
    gamma_mean = float(_log_band_from_psd(freqs, psd, *_GAMMA_LOW).mean())
    # 12, 13: ERD indices at C3 and C4. ref: `pfurtscheller2001mi`.
    erd_c3 = _erd_index_from_epoch(epoch, sfreq, channel=0)
    erd_c4 = _erd_index_from_epoch(epoch, sfreq, channel=2)
    # 14: bias term. ref: `li2010linucb` §3.1 (LinUCB context with a 1.0 bias).
    bias = 1.0

    parts: list[np.ndarray] = [
        mu_per_ch,
        beta_per_ch,
        np.array(
            [
                mu_beta_ratio,
                spec_ent,
                trial_var,
                artifact,
                theta_mean,
                gamma_mean,
                erd_c3,
                erd_c4,
                bias,
            ]
        ),
    ]

    if recent_arm_rewards is not None:
        rec = np.asarray(recent_arm_rewards, dtype=float).ravel()
        if rec.size != N_ARM_FAMILIES:
            raise ValueError(
                f"recent_arm_rewards must have length {N_ARM_FAMILIES}; got {rec.size}"
            )
        parts.append(rec)

    return np.concatenate(parts, axis=0)


def compute_context_generic(
    epoch: np.ndarray,
    *,
    sfreq: float = 250.0,
    recent_arm_rewards: np.ndarray | None = None,
) -> np.ndarray:
    """18-dim context for an arbitrary multi-channel epoch (Phase-5 §2.2 / §2.4).

    Maintains the same 18-dim layout as :func:`compute_context` and
    :func:`compute_context_2a` so OPLB / TS feature ψ(x, a) keeps a
    dataset-agnostic dimensionality. For datasets without a canonical
    sensorimotor C3/Cz/C4 triad (e.g. cognitive-load setups), the per-
    channel slots 0–5 use the **first three channels** as a generic
    placeholder; ERD indices at slots 12, 13 likewise default to channels
    0 and 2. The global aggregates at 6–11 still summarise the full
    n-channel signal.

    Parameters
    ----------
    epoch : shape ``(n_channels, n_samples)`` for any n_channels ≥ 3.
    sfreq : sampling rate in Hz.
    recent_arm_rewards : as in :func:`compute_context`.
    """
    if epoch.ndim != 2 or epoch.shape[0] < 3:
        raise ValueError(
            "compute_context_generic expects (n_channels, n_samples) with "
            f"n_channels ≥ 3; got {epoch.shape}"
        )

    freqs, psd = _welch_psd(epoch, sfreq)

    mu_full = _log_band_from_psd(freqs, psd, *_MU)
    mu_per_ch = mu_full[:3]
    beta_full = _log_band_from_psd(freqs, psd, *_BETA)
    beta_per_ch = beta_full[:3]
    mu_lin = np.exp(mu_full).mean()
    beta_lin = np.exp(beta_full).mean()
    mu_beta_ratio = mu_lin / (beta_lin + 1e-24)
    spec_ent = float(_spectral_entropy_from_psd(psd).mean())
    trial_var = float(np.var(epoch, axis=-1).mean())
    artifact = float(np.max(np.abs(epoch)) > _ARTIFACT_VOLTS)
    theta_mean = float(_log_band_from_psd(freqs, psd, *_THETA).mean())
    gamma_mean = float(_log_band_from_psd(freqs, psd, *_GAMMA_LOW).mean())
    erd_c3 = _erd_index_from_epoch(epoch, sfreq, channel=0)
    erd_c4 = _erd_index_from_epoch(epoch, sfreq, channel=2)
    bias = 1.0

    parts: list[np.ndarray] = [
        mu_per_ch,
        beta_per_ch,
        np.array(
            [
                mu_beta_ratio, spec_ent, trial_var, artifact,
                theta_mean, gamma_mean, erd_c3, erd_c4, bias,
            ]
        ),
    ]
    if recent_arm_rewards is not None:
        rec = np.asarray(recent_arm_rewards, dtype=float).ravel()
        if rec.size != N_ARM_FAMILIES:
            raise ValueError(
                f"recent_arm_rewards must have length {N_ARM_FAMILIES}; got {rec.size}"
            )
        parts.append(rec)
    return np.concatenate(parts, axis=0)


# 2a's 22-channel layout (BCI-IV-2a, international 10-20 derived). The
# `_load_gdf_with_labels` loader picks the first 22 EEG channels in the
# canonical recording order documented in `desc_2a.pdf` §3 ("Electrode
# montage"). Channels 8 (C3), 10 (Cz), 12 (C4) are the canonical sensorimotor
# triad — the same sites that 2b's 3 bipolar channels approximate.
# ref: `desc_2a.pdf` Table 2; international 10-20 montage (Klem 1999).
_2A_C3_IDX, _2A_CZ_IDX, _2A_C4_IDX = 7, 9, 11


def compute_context_2a(
    epoch: np.ndarray,
    *,
    sfreq: float = 250.0,
    recent_arm_rewards: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the 18-dim context vector for one 2a trial (22-ch layout).

    Maintains the *same* 18-dim structure as :func:`compute_context` so the
    OPLB / TS feature ψ(x, a) keeps the same dimensionality across datasets:

      indices 0–5    μ and β log-bandpower at C3, Cz, C4 (the canonical
                     sensorimotor triad — matches 2b's 3 bipolar channels)
      index 6        mean μ/β ratio across **all 22 channels**
      index 7        mean spectral entropy across **all 22 channels**
      index 8        mean trial variance across **all 22 channels**
      index 9        artifact flag (any channel above 100 µV)
      index 10       θ-band mean log-power across all channels
      index 11       low-γ-band mean log-power across all channels
      index 12       ERD index at C3 (early vs late μ-power)
      index 13       ERD index at C4 (early vs late μ-power)
      index 14       bias term (1.0)
      indices 15–17  running mean reward per arm family

    The dimension and the per-channel-spectral-summary positions are
    deliberately identical to :func:`compute_context` so that downstream
    OPLB / context wiring is dataset-agnostic. The 22-ch information enters
    via the global aggregates at indices 6–11 and via the broader arm space
    rather than by enlarging ``d``.

    Parameters
    ----------
    epoch : shape ``(22, n_samples)`` — 2a's 22 EEG channels for one trial.
    sfreq : sampling rate in Hz.
    recent_arm_rewards : length-``N_ARM_FAMILIES`` array of running mean
        rewards per arm family. Pass ``None`` to drop those entries.

    Returns
    -------
    ``np.ndarray`` of shape ``(d,)`` where ``d`` matches :func:`context_dim`.
    """
    if epoch.ndim != 2 or epoch.shape[0] != 22:
        raise ValueError(
            f"compute_context_2a expects 22-channel epoch of shape (22, n_samples); "
            f"got {epoch.shape}"
        )

    freqs, psd = _welch_psd(epoch, sfreq)  # (22, n_freqs)

    # 0–2: μ log-power at C3, Cz, C4 (sensorimotor triad).
    mu_full = _log_band_from_psd(freqs, psd, *_MU)
    mu_per_ch = mu_full[[_2A_C3_IDX, _2A_CZ_IDX, _2A_C4_IDX]]
    # 3–5: β log-power at C3, Cz, C4.
    beta_full = _log_band_from_psd(freqs, psd, *_BETA)
    beta_per_ch = beta_full[[_2A_C3_IDX, _2A_CZ_IDX, _2A_C4_IDX]]
    # 6: mean μ/β ratio across all 22 channels.
    mu_lin = np.exp(mu_full).mean()
    beta_lin = np.exp(beta_full).mean()
    mu_beta_ratio = mu_lin / (beta_lin + 1e-24)
    # 7: mean spectral entropy across all 22 channels.
    spec_ent = float(_spectral_entropy_from_psd(psd).mean())
    # 8: mean trial variance across all channels.
    trial_var = float(np.var(epoch, axis=-1).mean())
    # 9: artifact flag.
    artifact = float(np.max(np.abs(epoch)) > _ARTIFACT_VOLTS)
    # 10: θ across all channels.
    theta_mean = float(_log_band_from_psd(freqs, psd, *_THETA).mean())
    # 11: low-γ across all channels.
    gamma_mean = float(_log_band_from_psd(freqs, psd, *_GAMMA_LOW).mean())
    # 12, 13: ERD at C3, C4.
    erd_c3 = _erd_index_from_epoch(epoch, sfreq, channel=_2A_C3_IDX)
    erd_c4 = _erd_index_from_epoch(epoch, sfreq, channel=_2A_C4_IDX)
    # 14: bias term.
    bias = 1.0

    parts: list[np.ndarray] = [
        mu_per_ch,
        beta_per_ch,
        np.array(
            [
                mu_beta_ratio,
                spec_ent,
                trial_var,
                artifact,
                theta_mean,
                gamma_mean,
                erd_c3,
                erd_c4,
                bias,
            ]
        ),
    ]

    if recent_arm_rewards is not None:
        rec = np.asarray(recent_arm_rewards, dtype=float).ravel()
        if rec.size != N_ARM_FAMILIES:
            raise ValueError(
                f"recent_arm_rewards must have length {N_ARM_FAMILIES}; got {rec.size}"
            )
        parts.append(rec)

    return np.concatenate(parts, axis=0)
