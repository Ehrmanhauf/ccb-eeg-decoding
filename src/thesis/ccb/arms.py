"""Arm bank for the CCB (Formulation A) — per-trial (band × spatial × feature × window) selection.

Each :class:`Arm` is a configuration; each :class:`ArmHead` is that arm's
pre-trained linear scorer. The CCB policy (commit 2) chooses which head to
consult per trial based on context, subject to a knapsack constraint on
cumulative feature-vector cost.

Pipeline, per arm:

  raw epoch (3, n_samples) →
    [time window slice] →
    [Chebyshev-II bandpass (sub-band)] →
    [spatial filter: CSP | Laplacian | identity] →
    [feature: log-variance | band-power] →
    [shrinkage-LDA head → decision score / label]

Reuse (from `src/thesis/baselines/fbcsp.py`): ``DEFAULT_BANDS`` (the nine
sequential 4 Hz sub-bands from Ang 2012 §2.1) and ``_chebyshev_bandpass``
(zero-phase Chebyshev-II N=4 rs=40 dB via ``sosfiltfilt``).

See `design-doc/ccb-formulation.md` §5.1 and §6.1 for the formulation; see
`design-doc/open-justifications.md` for the pruning rule and the two new
``JUSTIFY:`` markers introduced in this module (time-window grid; Laplacian
stencil for 3-ch bipolar).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pywt
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from thesis.baselines.fbcsp import DEFAULT_BANDS, FBCSP, _chebyshev_bandpass
from thesis.metrics import compute_metrics

# Wavelet packet decomposition parameters for the wavelet_logenergy feature.
# Daubechies-4 (db4) is the canonical compact-support wavelet for EEG decoding;
# level=2 gives 2² = 4 sub-band packets per channel, balancing feature richness
# against the bandit's short-horizon over-fitting risk identified in
# Phase-5 §2.3 (the Riemannian-arms result showed that arm-pool expansion
# at this scale costs more than it returns; the same lesson sets level=2
# rather than level=3 here). ref: Mallat 1989; Lotte et al. 2018 §III.D.
_WAVELET_NAME: str = "db4"
_WAVELET_LEVEL: int = 2
_WAVELET_N_PACKETS: int = 2 ** _WAVELET_LEVEL  # 4 packets per channel

# "fbcsp" spatial is the Phase-5 H1 hybrid arm: the full 9-band FBCSP +
# shrinkage-LDA pipeline exposed as a single (expensive, strong) option
# inside the CCB arm pool. When spatial="fbcsp", the arm's ``band`` /
# ``feature`` / ``n_components`` fields describe the **configuration** of
# the internal FBCSP — not a single filter.
SpatialFilter = Literal["csp", "laplacian", "identity", "fbcsp"]
# "riemann_tangent": covariance → Riemannian tangent-space projection.
# Operates on multi-channel signal (laplacian or identity spatial only —
# CSP collapses to scalars and breaks covariance structure). Output dim =
# n_ch_out * (n_ch_out + 1) / 2 (upper-triangular tangent vector).
# ref: `barachant2012riemann` IEEE TBME 59(4) — robust at low channel
# counts because covariance retains class-discriminative structure even
# when spatial filtering is rank-limited.
FeatureType = Literal["logvar", "bandpower", "riemann_tangent", "wavelet_logenergy"]

# Arm-grid dimensions before pruning.
# ref (time-window grid): `design-doc/ccb-formulation.md` §5.1 lists the
# same three windows. Justifications:
#   (0.0, 4.0) — full cue-locked epoch; the 2a/2b loader's native window
#     and our Phase-2 FBCSP baseline default (`design-doc/open-justifications.md`
#     closed "Epoch time window 0–4 s cue-locked").
#   (0.5, 2.5) — Ang 2012 §3.1.1 verbatim ("the time segment of 0.5–2.5 s
#     after the onset of the visual cue were used"), verified 2026-05-12.
#   (1.0, 3.0) — kept on empirical grounds only: the full 3-window grid
#     beats every single-window restriction at the Phase-5 Stage-1 best
#     cell (`results/ccb_sens_time_window.csv`). No verified literature
#     citation; the historical "Ramoser 2000 / Graz convention" attribution
#     could not be confirmed in this audit.
# Tracked in `open-justifications.md` closed "CCB time-window arm grid".
_TIME_WINDOWS: tuple[tuple[float, float], ...] = (
    (0.0, 4.0),
    (0.5, 2.5),
    (1.0, 3.0),
)
_SPATIAL_FILTERS: tuple[SpatialFilter, ...] = ("csp", "laplacian", "identity")
_FEATURE_TYPES: tuple[FeatureType, ...] = ("logvar", "bandpower")
# Riemannian features only enumerate against multi-channel spatial filters
# (laplacian, identity). CSP is excluded because its log=True output is
# already scalar — no covariance structure to project.
_RIEMANN_SPATIAL_FILTERS: tuple[SpatialFilter, ...] = ("laplacian", "identity")

# Sentinel band for the H1 FBCSP arm — the full 4–40 Hz span covered by
# DEFAULT_BANDS. Used only as an identifying field on the Arm; the
# internal FBCSP uses DEFAULT_BANDS regardless.
_FBCSP_BAND_SENTINEL: tuple[float, float] = (
    float(DEFAULT_BANDS[0][0]),
    float(DEFAULT_BANDS[-1][1]),
)

# 2b's 3 bipolar channels correspond to sensorimotor sites near C3 / Cz / C4
# per `desc_2b.pdf`. We assume the canonical ordering:
#   channel 0 = C3 area, channel 1 = Cz area, channel 2 = C4 area.
_C3_IDX, _CZ_IDX, _C4_IDX = 0, 1, 2


@dataclass(frozen=True)
class Arm:
    """One pipeline configuration in the CCB arm bank.

    Parameters
    ----------
    arm_id : unique integer identifier (assigned by enumerate_arms_2b).
    band : (fmin, fmax) in Hz for the Chebyshev-II bandpass.
    spatial : "csp" | "laplacian" | "identity".
    feature : "logvar" | "bandpower".
    window : (tmin, tmax) in seconds, relative to cue onset.
    cost : scalar knapsack cost; feature-vector length proxy.
    n_components : CSP components (ignored for non-CSP spatial filters).
    """

    arm_id: int
    band: tuple[float, float]
    spatial: SpatialFilter
    feature: FeatureType
    window: tuple[float, float]
    cost: float
    n_components: int


def arm_cost(
    *,
    spatial: SpatialFilter,
    feature: FeatureType,
    n_components: int,
    n_channels: int,
) -> float:
    """Deterministic cost of running one arm: feature-vector length.

    For ``feature in {"logvar", "bandpower"}`` the per-channel scalar
    collapses the time axis: cost equals the spatial-stage output channel
    count. For ``feature == "riemann_tangent"`` the cost is the upper-
    triangular tangent-space dimension ``n_out * (n_out + 1) / 2`` where
    ``n_out`` is the post-spatial channel count.

    CSP outputs ``n_components`` features; Laplacian on 3-ch bipolar reduces
    to 2 channels (C3−Cz and C4−Cz); identity passes all ``n_channels``
    through.

    refs: `ang2012fbcsp` §2.2 on FBCSP feature-vector sizing;
    `barachant2012riemann` Eq. 4 on tangent-space dimensionality.
    """
    if spatial == "csp":
        n_out = min(n_components, n_channels)
    elif spatial == "laplacian":
        n_out = max(n_channels - 1, 1)
    elif spatial == "identity":
        n_out = n_channels
    elif spatial == "fbcsp":
        # Full 9-band FBCSP: per-band CSP produces min(n_components, n_channels)
        # features, concatenated across the full filter bank. Much more
        # expensive than any single-band arm — this is the point (H1 hybrid).
        return float(len(DEFAULT_BANDS) * min(n_components, n_channels))
    else:
        raise ValueError(f"Unknown spatial filter: {spatial!r}")

    if feature == "riemann_tangent":
        # Upper-triangular tangent-vector dim from a (n_out × n_out) SPD covariance.
        # ref: `barachant2012riemann` Eq. 4.
        return float(n_out * (n_out + 1) // 2)
    if feature == "wavelet_logenergy":
        # Wavelet packet decomposition at level _WAVELET_LEVEL produces
        # 2^level packets per channel. Output dim = n_out × 4 with the
        # default level=2. ref: Mallat 1989.
        return float(n_out * _WAVELET_N_PACKETS)
    return float(n_out)


def _laplacian_transform(X: np.ndarray) -> np.ndarray:
    """Surface-Laplacian-like spatial filter for 3-channel bipolar data.

    Produces 2 output channels:  C3'=C3−Cz, C4'=C4−Cz.  Reduces lateral
    drift common to Cz while preserving left/right sensorimotor contrast.
    Input (..., 3, n_samples) → output (..., 2, n_samples).

    JUSTIFY: the exact stencil is a methodological derivation for the
    canonical C3/Cz/C4 bipolar layout (desc_2b.pdf). Alternatives (e.g.,
    full surface Laplacian with Gaussian weights) require a 2-D electrode
    layout that 2b doesn't have. Tracked in `open-justifications.md` as
    "Spatial-filter family for 3-ch bipolar".
    """
    if X.shape[-2] != 3:
        raise ValueError(f"Laplacian expects 3 channels; got shape {X.shape}")
    ch3_prime = X[..., _C3_IDX, :] - X[..., _CZ_IDX, :]
    ch4_prime = X[..., _C4_IDX, :] - X[..., _CZ_IDX, :]
    return np.stack([ch3_prime, ch4_prime], axis=-2)


def _identity_transform(X: np.ndarray) -> np.ndarray:
    """Pass-through spatial filter."""
    return X


def _window_to_slice(window: tuple[float, float], sfreq: float) -> slice:
    lo = int(round(window[0] * sfreq))
    hi = int(round(window[1] * sfreq))
    return slice(lo, hi)


def _logvar(X: np.ndarray) -> np.ndarray:
    """Log-variance feature along the time axis. Input (..., n_ch, n_samples)."""
    var = np.var(X, axis=-1, ddof=1)
    return np.log(var + 1e-12)


def _bandpower(X: np.ndarray) -> np.ndarray:
    """Log-bandpower (mean of squared amplitude) along the time axis."""
    power = np.mean(X**2, axis=-1)
    return np.log(power + 1e-12)


def _wavelet_logenergy(X: np.ndarray) -> np.ndarray:
    """Per-channel wavelet-packet log-energy features.

    Decomposes each channel with Daubechies-4 wavelet packets at
    :data:`_WAVELET_LEVEL` levels (4 sub-band packets per channel by default).
    Returns log-energy (sum of squared coefficients, log-transformed) per
    packet, concatenated across channels.

    Input  : ``(..., n_ch, n_samples)``
    Output : ``(..., n_ch * 2^level)`` — flattened over (channel × packet).

    ref: Mallat 1989 wavelet theory; Lotte et al. 2018 §III.D on wavelet
    features for EEG.
    """
    leading = X.shape[:-2]
    n_ch = X.shape[-2]
    n_samples = X.shape[-1]
    flat = X.reshape(-1, n_ch, n_samples)
    feats = np.empty((flat.shape[0], n_ch * _WAVELET_N_PACKETS), dtype=float)
    for i in range(flat.shape[0]):
        for ch in range(n_ch):
            wp = pywt.WaveletPacket(
                data=flat[i, ch], wavelet=_WAVELET_NAME, mode="symmetric",
                maxlevel=_WAVELET_LEVEL,
            )
            packets = [node.data for node in wp.get_level(_WAVELET_LEVEL, order="natural")]
            for j, p in enumerate(packets):
                energy = float(np.sum(p ** 2))
                feats[i, ch * _WAVELET_N_PACKETS + j] = np.log(energy + 1e-12)
    return feats.reshape(*leading, n_ch * _WAVELET_N_PACKETS)


class ArmHead:
    """Pre-trained scorer for one :class:`Arm`.

    Default mode: fit once on calibration data, then used as a frozen scorer.
    Phase-5 ``partial_fit`` extension: after each newly-labelled trial, the
    LDA head can be re-fit on the accumulated (calibration + stream-so-far)
    feature buffer. The CSP spatial filter is **not** re-fit online (that
    would require re-solving the generalised eigenvalue problem each step);
    only the linear head adapts. This matches the Phase-4 diagnosis that
    session-gap drift is in the classifier decision boundary, not the
    spatial pattern (CSP is fitted once per session, drift happens across
    sessions).

    The head internally holds:
      - an optional :class:`mne.decoding.CSP` spatial filter (for CSP arms),
      - a shrinkage LDA trained on the spatial + feature output,
      - an optional running feature / label buffer (only if ``partial_fit``
        is used).

    ref: `design-doc/ccb-formulation.md` §6.2 (per-arm feature map);
    Phase-5 §7.3 online heads note.
    """

    def __init__(self, arm: Arm):
        self.arm = arm
        self._csp: CSP | None = None
        self._lda: LinearDiscriminantAnalysis | None = None
        self.classes_: np.ndarray | None = None
        # Phase-5 H1 hybrid: internal FBCSP estimator for arms with
        # ``spatial == "fbcsp"``. None for all other arms.
        self._fbcsp: FBCSP | None = None
        # Phase-5 §2.3: covariance estimator + tangent-space projection
        # for ``feature == "riemann_tangent"`` arms. Both are fitted on
        # calibration covariances during fit() and reused at inference.
        # ref: `barachant2012riemann`.
        self._riemann_cov: Covariances | None = None
        self._riemann_ts: TangentSpace | None = None
        # Lazily-allocated online buffers. ``None`` means ``partial_fit`` has
        # not been used; that keeps backward-compat with every frozen-head
        # code path.
        self._feats_buffer: list[np.ndarray] | None = None
        self._y_buffer: list = None  # type: ignore[assignment]

    @property
    def is_fitted(self) -> bool:
        return self._lda is not None or self._fbcsp is not None

    def _apply_window_and_band(self, X: np.ndarray, sfreq: float) -> np.ndarray:
        """Time-window slice + Chebyshev-II bandpass."""
        X_win = X[..., _window_to_slice(self.arm.window, sfreq)]
        return _chebyshev_bandpass(X_win, self.arm.band[0], self.arm.band[1], sfreq)

    def _apply_spatial(self, X_filt: np.ndarray) -> np.ndarray:
        if self.arm.spatial == "csp":
            if self._csp is None:
                raise RuntimeError("CSP arm not fitted; call fit() first.")
            # mne.decoding.CSP with log=True returns log-variance features directly.
            return self._csp.transform(X_filt)
        if self.arm.spatial == "laplacian":
            return _laplacian_transform(X_filt)
        if self.arm.spatial == "identity":
            return _identity_transform(X_filt)
        raise ValueError(f"Unknown spatial filter: {self.arm.spatial!r}")

    def _apply_feature(self, X_spatial: np.ndarray) -> np.ndarray:
        """Feature extraction. For CSP arms, MNE already returns log-variance;
        for non-CSP arms we apply logvar / bandpower / riemann-tangent ourselves.

        Riemannian path requires the covariance and tangent-space estimators
        to have been fitted on calibration data via :meth:`fit`. ``feature_vec``
        delegates to this method at inference; the estimators raise if unfit.
        """
        if self.arm.spatial == "csp":
            # CSP log=True already emits log-variance regardless of
            # self.arm.feature. The feature field is a label for accounting.
            return X_spatial
        if self.arm.feature == "logvar":
            return _logvar(X_spatial)
        if self.arm.feature == "bandpower":
            return _bandpower(X_spatial)
        if self.arm.feature == "riemann_tangent":
            if self._riemann_cov is None or self._riemann_ts is None:
                raise RuntimeError(
                    "Riemannian arm not fitted; call fit() before feature_vec()."
                )
            covs = self._riemann_cov.transform(X_spatial)
            return self._riemann_ts.transform(covs)
        if self.arm.feature == "wavelet_logenergy":
            return _wavelet_logenergy(X_spatial)
        raise ValueError(f"Unknown feature type: {self.arm.feature!r}")

    def fit(self, X: np.ndarray, y: np.ndarray, sfreq: float) -> ArmHead:
        """Fit spatial filter (if CSP / FBCSP) and shrinkage LDA on calibration data.

        Returns self for chaining.
        """
        if self.arm.spatial == "fbcsp":
            # H1 hybrid path: internal full-FBCSP pipeline on the time-windowed
            # epoch. The FBCSP class runs its own 9-band filter-bank + CSP +
            # shrinkage LDA — we only need to pre-slice the time window.
            X_win = X[..., _window_to_slice(self.arm.window, sfreq)]
            n_ch = X.shape[-2]
            k = min(self.arm.n_components, n_ch)
            self._fbcsp = FBCSP(sfreq=sfreq, bands=DEFAULT_BANDS, n_components=k)
            self._fbcsp.fit(X_win, y)
            # Mirror the LDA reference so is_fitted + classes_ behave uniformly.
            self._lda = self._fbcsp.lda_
            self.classes_ = self._fbcsp.classes_
            # Online buffer not supported for FBCSP arms (would require
            # re-fitting the per-band CSPs — out of scope for H1).
            self._feats_buffer = None
            self._y_buffer = None
            return self

        X_filt = self._apply_window_and_band(X, sfreq)

        if self.arm.spatial == "csp":
            # Cap components at rank of input to avoid MNE's silent truncation
            # or generalized-eigenvalue failures on rank-deficient inputs.
            n_ch = X.shape[-2]
            k = min(self.arm.n_components, n_ch)
            self._csp = CSP(n_components=k, reg=None, log=True, norm_trace=False)
            self._csp.fit(X_filt, y)

        # Riemannian feature: fit Covariances + TangentSpace on the spatial-
        # filtered calibration epochs before extracting features. SCM
        # estimator is the standard sample-covariance matrix; the Riemannian
        # metric on the tangent space is the affine-invariant default.
        # ref: `barachant2012riemann` §III.B.
        if self.arm.feature == "riemann_tangent":
            if self.arm.spatial == "csp":
                raise ValueError(
                    "feature='riemann_tangent' is incompatible with spatial='csp' "
                    "(CSP log=True returns scalars; no covariance structure). "
                    "Use spatial in {laplacian, identity}."
                )
            X_spatial_cal = self._apply_spatial(X_filt)
            self._riemann_cov = Covariances(estimator="scm")
            covs_cal = self._riemann_cov.fit_transform(X_spatial_cal)
            self._riemann_ts = TangentSpace(metric="riemann")
            self._riemann_ts.fit(covs_cal)

        feats = self._apply_feature(self._apply_spatial(X_filt))
        # ref: `lotte2018review` §III.B (shrinkage LDA as MI-BCI standard).
        self._lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self._lda.fit(feats, y)
        self.classes_ = self._lda.classes_
        # Seed the online buffer with calibration features so that the first
        # partial_fit call re-fits on a superset of the calibration data.
        self._feats_buffer = [feats]
        self._y_buffer = [np.asarray(y)]
        return self

    def partial_fit(self, X_new: np.ndarray, y_new: np.ndarray, sfreq: float) -> ArmHead:
        """Append labelled trials and re-fit the LDA head.

        The CSP spatial filter is **not** re-fit (keeps Phase-5 scope bounded
        and avoids unstable per-trial eigendecompositions on tiny increments).
        The LDA is re-solved via closed-form shrinkage on the running buffer
        ``(calibration ∪ stream-so-far)``.

        Parameters
        ----------
        X_new : shape ``(n_new_trials, n_ch, n_samples)`` — raw epochs (pre
            window + band), same shape convention as :meth:`fit`.
        y_new : shape ``(n_new_trials,)`` — labels.
        sfreq : sampling rate.
        """
        if not self.is_fitted:
            raise RuntimeError("partial_fit requires a prior fit() call.")
        if self.arm.spatial == "fbcsp":
            # Online updates for full-FBCSP arms would require re-fitting the
            # 9 per-band CSPs on the running buffer — expensive and numerically
            # fragile. Explicitly unsupported; runner must avoid this path
            # when online_heads=True and an FBCSP arm is in the pool.
            raise RuntimeError(
                "partial_fit is not supported for spatial='fbcsp' arms "
                "(H1 hybrid keeps FBCSP arms frozen after calibration)."
            )
        if self._feats_buffer is None or self._y_buffer is None:
            # Defensive: reseed if buffers were dropped.
            raise RuntimeError("online buffers uninitialised; was fit() called?")
        X_filt = self._apply_window_and_band(X_new, sfreq)
        new_feats = self._apply_feature(self._apply_spatial(X_filt))
        self._feats_buffer.append(new_feats)
        self._y_buffer.append(np.asarray(y_new))
        feats_all = np.concatenate(self._feats_buffer, axis=0)
        y_all = np.concatenate(self._y_buffer, axis=0)
        # Re-solve shrinkage LDA; same hyperparameters as fit().
        self._lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self._lda.fit(feats_all, y_all)
        self.classes_ = self._lda.classes_
        return self

    def feature_vec(self, X: np.ndarray, sfreq: float) -> np.ndarray:
        """Return feature vectors for X. Shape: (n_trials, d_a)."""
        if not self.is_fitted:
            raise RuntimeError("ArmHead not fitted; call fit() first.")
        if self.arm.spatial == "fbcsp":
            assert self._fbcsp is not None  # guaranteed by is_fitted
            X_win = X[..., _window_to_slice(self.arm.window, sfreq)]
            # FBCSP._transform_features is the fit-time feature map (private
            # helper) — matches what the LDA was trained on.
            return self._fbcsp._transform_features(X_win)
        X_filt = self._apply_window_and_band(X, sfreq)
        return self._apply_feature(self._apply_spatial(X_filt))

    def score(self, X: np.ndarray, sfreq: float) -> np.ndarray:
        """Signed LDA decision score per trial. Shape: (n_trials,)."""
        # feature_vec() raises RuntimeError if unfitted — let that propagate.
        feats = self.feature_vec(X, sfreq)
        assert self._lda is not None  # guaranteed by is_fitted check in feature_vec
        return self._lda.decision_function(feats)

    def predict(self, X: np.ndarray, sfreq: float) -> np.ndarray:
        """Predicted class labels per trial. Shape: (n_trials,)."""
        feats = self.feature_vec(X, sfreq)
        assert self._lda is not None
        return self._lda.predict(feats)


def enumerate_arms_2b(
    sfreq: float = 250.0,  # noqa: ARG001 — sfreq reserved for future window-vs-sfreq validation
    *,
    include_fbcsp_arm: bool = False,
    include_riemann_arms: bool = False,
    include_wavelet_arms: bool = False,
) -> list[Arm]:
    """Enumerate the arm grid for 2b (3-ch bipolar).

    Base grid: 9 bands × 3 spatial filters × 2 feature types × 3 time
    windows = 162 single-configuration arms.

    Optional extensions:

    - ``include_fbcsp_arm=True`` (Phase-5 H1 hybrid): adds 1 arm for the full
      9-band FBCSP + shrinkage-LDA pipeline. Total = 163.
    - ``include_riemann_arms=True`` (Phase-5 §2.3): adds 9 bands × 2
      multi-channel spatial filters (laplacian, identity) × 1 Riemannian
      feature × 3 time windows = 54 arms. CSP is excluded because its
      log=True output is scalar (no covariance to project). Total = 216
      (or 217 with both extensions).

    For CSP arms, ``n_components`` is capped at 3 (the 2b channel count).
    Per-arm cost is the feature-vector dimension — the FBCSP arm is the
    most expensive single arm at 9 × 3 = 27 features; Riemannian-on-identity
    is the most expensive Phase-5 §2.3 arm at 3 × 4 / 2 = 6 features.

    The returned list is the **un-pruned** base grid; downstream callers
    should run :func:`prune_arms` with calibration data to reduce to ≤ 100.

    ref: `design-doc/ccb-formulation.md` §5.1; Phase-5 H1 hybrid adds the
    FBCSP arm; Phase-5 §2.3 adds Riemannian arms (`barachant2012riemann`).
    """
    arms: list[Arm] = []
    arm_id = 0
    for band in DEFAULT_BANDS:
        for spatial in _SPATIAL_FILTERS:
            for feature in _FEATURE_TYPES:
                for window in _TIME_WINDOWS:
                    # CSP on 3-channel bipolar: n_components capped at 3.
                    n_components = 3 if spatial == "csp" else 3
                    cost = arm_cost(
                        spatial=spatial,
                        feature=feature,
                        n_components=n_components,
                        n_channels=3,
                    )
                    arms.append(
                        Arm(
                            arm_id=arm_id,
                            band=band,
                            spatial=spatial,
                            feature=feature,
                            window=window,
                            cost=cost,
                            n_components=n_components,
                        )
                    )
                    arm_id += 1
    if include_riemann_arms:
        # Phase-5 §2.3: Riemannian arms.
        # 9 bands × 2 spatial filters (laplacian, identity) × 1 feature
        # (riemann_tangent) × 3 windows = 54 additional arms.
        # CSP is excluded because its log=True output is scalar; no covariance
        # structure is left to project to the tangent space.
        # ref: `barachant2012riemann`.
        for band in DEFAULT_BANDS:
            for spatial in _RIEMANN_SPATIAL_FILTERS:
                for window in _TIME_WINDOWS:
                    n_components = 3
                    cost = arm_cost(
                        spatial=spatial,
                        feature="riemann_tangent",
                        n_components=n_components,
                        n_channels=3,
                    )
                    arms.append(
                        Arm(
                            arm_id=arm_id,
                            band=band,
                            spatial=spatial,
                            feature="riemann_tangent",
                            window=window,
                            cost=cost,
                            n_components=n_components,
                        )
                    )
                    arm_id += 1
    if include_wavelet_arms:
        # Phase-5 §2.3 item 3: wavelet-packet log-energy arms.
        # Same multi-channel spatial filters as Riemannian (CSP collapses
        # to scalar — no time series left for wavelet decomposition).
        # 9 × 2 × 1 × 3 = 54 additional arms with feature dim
        # n_out × 4 (level=2) per arm.
        # ref: Mallat 1989 wavelet packet theory; Lotte et al. 2018 §III.D.
        for band in DEFAULT_BANDS:
            for spatial in _RIEMANN_SPATIAL_FILTERS:
                for window in _TIME_WINDOWS:
                    n_components = 3
                    cost = arm_cost(
                        spatial=spatial,
                        feature="wavelet_logenergy",
                        n_components=n_components,
                        n_channels=3,
                    )
                    arms.append(
                        Arm(
                            arm_id=arm_id,
                            band=band,
                            spatial=spatial,
                            feature="wavelet_logenergy",
                            window=window,
                            cost=cost,
                            n_components=n_components,
                        )
                    )
                    arm_id += 1
    if include_fbcsp_arm:
        # Single FBCSP arm spanning the full 4–40 Hz filter bank, on the full
        # 0–4 s cue-locked window. 9 bands × 3 components = 27 features.
        fbcsp_components = 3
        fbcsp_cost = arm_cost(
            spatial="fbcsp",
            feature="logvar",
            n_components=fbcsp_components,
            n_channels=3,
        )
        arms.append(
            Arm(
                arm_id=arm_id,
                band=_FBCSP_BAND_SENTINEL,
                spatial="fbcsp",
                feature="logvar",
                window=(0.0, 4.0),
                cost=fbcsp_cost,
                n_components=fbcsp_components,
            )
        )
        arm_id += 1
    return arms


def enumerate_arms_generic(
    n_channels: int,
    *,
    n_components: int = 4,
    include_riemann_arms: bool = False,
) -> list[Arm]:
    """Enumerate a CCB arm grid for any high-channel dataset.

    Generalises :func:`enumerate_arms_2a` to arbitrary channel counts.
    Used by Phase-5 §2.2 (PhysioNet 64-ch) and §2.4 (BNCI2015_004 30-ch).

    Spatial families: CSP and identity (Laplacian omitted because the
    per-electrode Hjorth stencil requires a verified neighbour map per
    montage; deferred). Per-arm cost is the feature-vector length.

    Base grid: 9 bands × 2 spatial × 2 features × 3 windows = 108 arms.
    With Riemannian: + 9 × 1 × 1 × 3 = 27 (total 135).

    Parameters
    ----------
    n_channels : number of EEG channels in the dataset.
    n_components : CSP components (capped to n_channels at fit time).
    include_riemann_arms : add 27 Riemannian-on-identity arms.
    """
    arms: list[Arm] = []
    arm_id = 0
    spatial_set: tuple[SpatialFilter, ...] = ("csp", "identity")
    for band in DEFAULT_BANDS:
        for spatial in spatial_set:
            for feature in _FEATURE_TYPES:
                for window in _TIME_WINDOWS:
                    cost = arm_cost(
                        spatial=spatial,
                        feature=feature,
                        n_components=n_components,
                        n_channels=n_channels,
                    )
                    arms.append(
                        Arm(
                            arm_id=arm_id,
                            band=band,
                            spatial=spatial,
                            feature=feature,
                            window=window,
                            cost=cost,
                            n_components=n_components,
                        )
                    )
                    arm_id += 1
    if include_riemann_arms:
        for band in DEFAULT_BANDS:
            for window in _TIME_WINDOWS:
                cost = arm_cost(
                    spatial="identity",
                    feature="riemann_tangent",
                    n_components=n_components,
                    n_channels=n_channels,
                )
                arms.append(
                    Arm(
                        arm_id=arm_id,
                        band=band,
                        spatial="identity",
                        feature="riemann_tangent",
                        window=window,
                        cost=cost,
                        n_components=n_components,
                    )
                )
                arm_id += 1
    return arms


def enumerate_arms_2a(
    sfreq: float = 250.0,  # noqa: ARG001 — sfreq reserved for future window-vs-sfreq validation
    *,
    n_components: int = 4,
    include_riemann_arms: bool = False,
) -> list[Arm]:
    """Enumerate the arm grid for BCI-IV-2a (22-ch benchmark dataset).

    Phase-5 §2.1 extension. The 22-channel arm bank deliberately omits the
    Laplacian spatial family because the canonical 22-ch Hjorth (1975)
    stencil requires a verified per-electrode neighbour mapping that this
    first-pass implementation does not assume; CSP and identity cover the
    classical-MI and broad-spectrum cases respectively, and Riemannian-on-
    identity provides the rich-covariance variant.

    Base grid: 9 bands × 2 spatial filters (csp, identity) × 2 feature types
    (logvar, bandpower) × 3 time windows = 108 arms.

    With ``include_riemann_arms=True``: + 9 bands × 1 spatial (identity) × 1
    feature (riemann_tangent) × 3 windows = 27 arms (total 135). Riemannian
    on CSP is invalid (CSP collapses to scalar); Riemannian on the absent
    Laplacian family is therefore not enumerated either.

    For CSP arms, the default ``n_components=4`` matches Ang 2012 §2.2
    (``m_per_side = 2`` ⇒ 4 components total) — the FBCSP baseline value
    used by ``src/thesis/baselines/fbcsp.py``.

    The returned list is the **un-pruned** base grid; downstream callers
    should run :func:`prune_arms` with calibration data to reduce to ≤ 100.

    refs: `ang2012fbcsp` §2.2 (CSP component default); `barachant2012riemann`
    (Riemannian feature family); `desc_2a.pdf` (22-ch montage).
    """
    arms: list[Arm] = []
    arm_id = 0
    # 2a-specific spatial set: CSP (rank up to 22) + identity (passes 22-ch).
    # Laplacian is intentionally omitted in this first-pass 2a arm bank.
    spatial_2a: tuple[SpatialFilter, ...] = ("csp", "identity")
    n_channels = 22
    for band in DEFAULT_BANDS:
        for spatial in spatial_2a:
            for feature in _FEATURE_TYPES:
                for window in _TIME_WINDOWS:
                    cost = arm_cost(
                        spatial=spatial,
                        feature=feature,
                        n_components=n_components,
                        n_channels=n_channels,
                    )
                    arms.append(
                        Arm(
                            arm_id=arm_id,
                            band=band,
                            spatial=spatial,
                            feature=feature,
                            window=window,
                            cost=cost,
                            n_components=n_components,
                        )
                    )
                    arm_id += 1
    if include_riemann_arms:
        # Riemannian on identity only (CSP excluded; Laplacian absent for 2a).
        # Identity preserves 22 channels → tangent dim = 22 × 23 / 2 = 253.
        for band in DEFAULT_BANDS:
            for window in _TIME_WINDOWS:
                cost = arm_cost(
                    spatial="identity",
                    feature="riemann_tangent",
                    n_components=n_components,
                    n_channels=n_channels,
                )
                arms.append(
                    Arm(
                        arm_id=arm_id,
                        band=band,
                        spatial="identity",
                        feature="riemann_tangent",
                        window=window,
                        cost=cost,
                        n_components=n_components,
                    )
                )
                arm_id += 1
    return arms


def build_arm_heads(
    arms: list[Arm],
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    sfreq: float = 250.0,
) -> dict[int, ArmHead]:
    """Fit an :class:`ArmHead` for each arm on calibration (X_cal, y_cal).

    Returns a dict keyed by ``arm_id`` so downstream code can look up heads
    without list indexing games.
    """
    return {arm.arm_id: ArmHead(arm).fit(X_cal, y_cal, sfreq) for arm in arms}


def prune_arms(
    arms: list[Arm],
    heads: dict[int, ArmHead],
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    sfreq: float = 250.0,
    *,
    min_kappa: float = 0.05,
    max_arms: int = 100,
) -> list[Arm]:
    """Drop arms with held-out κ < ``min_kappa``; cap at ``max_arms`` by κ desc.

    This is the resolution of the `open-justifications.md` "Arm pool
    construction" item:

    1. Fit heads on a calibration split (caller's responsibility).
    2. Score each head on a disjoint holdout split.
    3. Keep only arms with κ ≥ 0.05 (chance-level tolerance from
       `lotte2018review` §III.C on BCI-illiteracy κ ≈ 0).
    4. Truncate to the top ``max_arms`` by κ.

    Returns arms **sorted by κ descending**.
    """
    scored: list[tuple[float, Arm]] = []
    for arm in arms:
        head = heads[arm.arm_id]
        y_pred = head.predict(X_holdout, sfreq)
        kappa = compute_metrics(y_holdout, y_pred).kappa
        scored.append((kappa, arm))

    survivors = sorted(
        (pair for pair in scored if pair[0] >= min_kappa),
        key=lambda p: -p[0],
    )[:max_arms]
    return [arm for _, arm in survivors]
