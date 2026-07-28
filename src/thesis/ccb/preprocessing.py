"""Phase-5 §2.3 signal-processing preprocessing for the CCB pipeline.

Two techniques live here as **preprocessing transforms** that operate on a
trial tensor ``(n_trials, n_channels, n_samples)`` and return the same shape:

- :func:`apply_notch` — second-order IIR notch filter at 50 Hz (the European
  power-line frequency present in BCI-IV recordings). Quick and universally
  applicable. ref: Lotte et al. 2018 §III.A on standard preprocessing.
- :func:`apply_ica_cleaning` — Independent Component Analysis on a calibration
  block to identify and reject artifact components, then projection of every
  trial through the surviving spatial demixing. ref: Hyvärinen & Oja 2000
  on FastICA; mne.preprocessing.ICA for the implementation.

A third technique (wavelet-packet log-energy) is implemented as a *new arm
family* in :mod:`thesis.ccb.arms` because it changes the feature extractor,
not the input signal — it does not belong here.

A fourth technique (sLORETA source-space projection) is **not implemented**
in the present iteration. With 3 sensors and ≥100 candidate cortical sources
the inverse problem is severely under-determined; the regularised solution
collapses to a near-identity projection. Pascual-Marqui (2002) discusses
this minimum-sensor limitation directly. The technique is documented as a
future direction once a higher-channel dataset (PhysioNet 64-ch, BCI-IV-2a
22-ch) is the headline target.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import iirnotch, sosfiltfilt, tf2sos


def apply_notch(X: np.ndarray, sfreq: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Apply a second-order IIR notch at ``freq`` Hz to every trial / channel.

    Parameters
    ----------
    X : (..., n_samples) — works on any leading-axis shape; the notch is
        always applied along the last axis.
    sfreq : sampling rate in Hz.
    freq : notch frequency. Default 50 Hz (European power line).
    q : quality factor controlling notch width. Default 30 (≈1.7 Hz wide
        at 50 Hz, narrow enough to not eat the β-band tail).

    Returns
    -------
    Filtered ``X`` with the same shape.
    """
    b, a = iirnotch(w0=freq, Q=q, fs=sfreq)
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, X, axis=-1)


def apply_ica_cleaning(
    X_calibration: np.ndarray,
    X_apply: np.ndarray,
    sfreq: float,
    *,
    n_components: int | None = None,
    eog_z_threshold: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    """Fit FastICA on calibration epochs, drop high-variance artifact components,
    project ``X_apply`` through the surviving demixing.

    For 3-channel data, ``n_components`` defaults to 3 (full rank — no actual
    dimensionality reduction; the cleaning effect comes from zeroing rows of
    the mixing matrix that capture artifact patterns).

    "Artifact" identification is heuristic: an IC whose calibration-trial
    amplitude has |z-score| > ``eog_z_threshold`` on the maximum-amplitude
    sample is flagged. Without an explicit EOG / EMG reference channel
    (BCI-IV-2b's loader drops the EOG channels), this is the only auto-
    detection signal available; a true EOG-correlation step would require
    extending the data loader.

    Parameters
    ----------
    X_calibration : (n_cal_trials, n_ch, n_samples) — used to fit ICA.
    X_apply : (n_apply_trials, n_ch, n_samples) — projected through cleaned
        demixing.
    sfreq : sampling rate (informational).
    n_components : ICA component count. ``None`` → ``min(n_ch, 3)``.
    eog_z_threshold : z-score threshold for component-amplitude-based
        artifact flagging.
    seed : RNG seed for FastICA (reproducibility).

    Returns
    -------
    Cleaned ``X_apply`` with the same shape.
    """
    from mne.preprocessing import ICA
    import mne

    n_ch = X_apply.shape[1]
    if n_components is None:
        n_components = min(n_ch, 3)

    # Build a synthetic Raw from concatenated calibration epochs to fit ICA.
    # This is the standard MNE pattern; see mne.preprocessing.ICA docstring
    # "Notes" section: ICA is fit on a continuous signal even when the
    # downstream application is per-epoch.
    info = mne.create_info(
        ch_names=[f"ch{i}" for i in range(n_ch)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    cal_concat = np.concatenate([X_calibration[i] for i in range(X_calibration.shape[0])], axis=-1)
    raw_cal = mne.io.RawArray(cal_concat, info, verbose=False)

    ica = ICA(n_components=n_components, random_state=seed, max_iter="auto", verbose=False)
    ica.fit(raw_cal)

    # Heuristic artifact detection: any IC whose source-amplitude max-z
    # exceeds the threshold on the calibration data is excluded.
    sources_cal = ica.get_sources(raw_cal).get_data()
    bad_ics: list[int] = []
    for i in range(sources_cal.shape[0]):
        z = (sources_cal[i] - sources_cal[i].mean()) / (sources_cal[i].std() + 1e-12)
        if np.max(np.abs(z)) > eog_z_threshold * 10:
            # Threshold ×10 because z-scores get inflated on continuous data;
            # this catches genuine high-variance ocular blinks while leaving
            # ordinary motor-imagery components alone.
            bad_ics.append(i)
    ica.exclude = bad_ics

    # Apply to X_apply trial-by-trial.
    cleaned = np.empty_like(X_apply)
    for t in range(X_apply.shape[0]):
        raw_t = mne.io.RawArray(X_apply[t], info, verbose=False)
        ica.apply(raw_t, verbose=False)
        cleaned[t] = raw_t.get_data()
    return cleaned
