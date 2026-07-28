"""Filter-Bank Common Spatial Pattern + shrinkage LDA.

Follows Ang et al. 2012 (ref: ``ang2012fbcsp``) on BCI-IV 2a/2b:

1. Split the broadband MI signal into nine 4 Hz sub-bands from 4–40 Hz with
   Chebyshev Type-II IIR filters (4–8, 8–12, …, 36–40 Hz).
   ref: ``ang2012fbcsp`` §2.1 quotes the same nine bands verbatim.
2. Fit a CSP (``n_components=4``) independently per band.
   ref: ``ang2012fbcsp`` §3.1.1 uses fixed ``m = 2`` on Dataset 2a (i.e., 4
   components total) and ``m = 1`` on Dataset 2b (2 components total) —
   NOT subject-specific. We fix ``n_components = 4`` (matches Ang's 2a
   setting); on 2b MNE caps to ``n_channels = 3``, which our sensitivity
   sweep showed is within 0.01 κ of every other tested value (see
   ``design-doc/open-justifications.md`` closed item "CSP component count").
3. Concatenate the log-variance features across bands → 36-dim feature vector.
4. Classify with scikit-learn's shrinkage LDA.

**Classifier choice:** Ang 2012 use a Naïve Bayesian Parzen Window (NBPW)
classifier; we use shrinkage LDA instead. ref: ``lotte2018review`` §III.B
reports shrinkage LDA as a standard MI-BCI baseline with performance within
~1–2 κ-points of NBPW across BCI-IV, plus a cleaner convex optimization that
removes one kernel-bandwidth hyperparameter.

**Feature selection:** Ang 2012 apply a Mutual Information-based Best
Individual Feature (MIBIF) step to shortlist the top-k features per subject.
We feed the full 36-dim vector into LDA — shrinkage LDA regularizes the
covariance estimate, which covers the role of feature selection for a
low-dimensional 2-class problem. A MIBIF extension is a design-doc §10
sensitivity follow-up, not a Phase-2 baseline decision.
"""

from __future__ import annotations

import numpy as np
from mne.decoding import CSP
from scipy.signal import cheby2, sosfiltfilt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 9 sequential 4 Hz sub-bands covering 4–40 Hz (Ang 2012 §2.1).
DEFAULT_BANDS: tuple[tuple[float, float], ...] = tuple(
    (float(lo), float(lo + 4)) for lo in range(4, 37, 4)
)


def _chebyshev_bandpass(data: np.ndarray, fmin: float, fmax: float, sfreq: float) -> np.ndarray:
    """Zero-phase Chebyshev Type-II bandpass applied along the time axis."""
    nyq = sfreq / 2.0
    sos = cheby2(N=4, rs=40, Wn=[fmin / nyq, fmax / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


class FBCSP(BaseEstimator, ClassifierMixin):
    """Filter-bank CSP followed by LDA."""

    def __init__(
        self,
        *,
        sfreq: float = 250.0,
        bands: tuple[tuple[float, float], ...] = DEFAULT_BANDS,
        n_components: int = 4,
    ):
        self.sfreq = sfreq
        self.bands = bands
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray) -> FBCSP:
        X = np.asarray(X)
        y = np.asarray(y)
        self.csps_: list[CSP] = []
        features: list[np.ndarray] = []
        for fmin, fmax in self.bands:
            X_filt = _chebyshev_bandpass(X, fmin, fmax, self.sfreq)
            csp = CSP(n_components=self.n_components, reg=None, log=True, norm_trace=False)
            # Shrinkage LDA is locked in below; configuring it here so all
            # per-band CSP objects stay stateless w.r.t. classifier choice.
            csp.fit(X_filt, y)
            self.csps_.append(csp)
            features.append(csp.transform(X_filt))
        feat = np.concatenate(features, axis=1)
        # Shrinkage LDA (lsqr + automatic shrinkage). ref: ``lotte2018review``
        # §III.B; Ledoit-Wolf shrinkage is the default for ``shrinkage='auto'``.
        self.lda_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.lda_.fit(feat, y)
        self.classes_ = self.lda_.classes_
        return self

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        feats: list[np.ndarray] = []
        for (fmin, fmax), csp in zip(self.bands, self.csps_, strict=True):
            X_filt = _chebyshev_bandpass(X, fmin, fmax, self.sfreq)
            feats.append(csp.transform(X_filt))
        return np.concatenate(feats, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.lda_.predict(self._transform_features(np.asarray(X)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.lda_.predict_proba(self._transform_features(np.asarray(X)))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())
