"""Leakage-safe feature-transformer wrappers around the fixed-pipeline baselines.

These expose the *feature-extraction* stage of the two fixed-pipeline baselines
(FBCSP, BandPowerCL) as scikit-learn transformers, so that an arbitrary
classifier head can be fitted on the **same** engineered features the baselines
and the CCB arm-heads consume. This isolates the classifier-choice effect from
the feature-engineering effect: the thesis's B1-vs-B2 design holds the
classifier fixed and varies the features; these wrappers do the orthogonal
thing — hold the features fixed and vary the classifier (B3--B5, see
``thesis.baselines.classical``).

No-leakage discipline (``CLAUDE.md`` §2):

- ``FBCSPTransformer`` fits a *supervised* per-band CSP, so its ``fit`` consumes
  ``y`` and must be re-fit per fold on the training split only. scikit-learn's
  ``Pipeline`` / cross-validation machinery guarantees this when the transformer
  is the first step of a pipeline that is itself fitted per fold.
- ``BandPowerTransformer`` is unsupervised (per-trial Welch spectral features),
  so its ``fit`` is a no-op beyond storing configuration.

Both wrappers **reuse** the verified feature code of the baseline classes
(``FBCSP._transform_features``, ``BandPowerCL._extract_features``) rather than
re-implementing it, so the produced features are identical to B1 / B2 by
construction. The wrappers add no new signal-processing choice; they only
re-expose an existing one through the scikit-learn transformer API.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from thesis.baselines.bandpower_cl import DEFAULT_CL_BANDS, BandPowerCL
from thesis.baselines.fbcsp import DEFAULT_BANDS, FBCSP


class FBCSPTransformer(BaseEstimator, TransformerMixin):
    """Filter-bank CSP log-variance features as a scikit-learn transformer.

    Wraps :class:`thesis.baselines.fbcsp.FBCSP`: ``fit`` fits the per-band
    supervised CSP filters (the FBCSP's shrinkage-LDA head is fitted too, but is
    never used here — the cost is negligible and we keep the verified
    ``FBCSP.fit`` path intact), and ``transform`` applies the fitted CSPs via the
    baseline's own ``_transform_features``. The output is the identical
    ``len(bands) * n_components``-dim feature matrix that Baseline B1 classifies.
    """

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

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "FBCSPTransformer":
        if y is None:
            raise ValueError(
                "FBCSPTransformer.fit requires labels y; the per-band CSP is supervised."
            )
        self.fbcsp_ = FBCSP(sfreq=self.sfreq, bands=self.bands, n_components=self.n_components)
        self.fbcsp_.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # _transform_features only *reads* the fitted ``csps_`` (no mutation),
        # so applying it to a held-out fold cannot leak training information.
        return self.fbcsp_._transform_features(np.asarray(X))


class BandPowerTransformer(BaseEstimator, TransformerMixin):
    """Cognitive-load band-power features as a scikit-learn transformer.

    Wraps :class:`thesis.baselines.bandpower_cl.BandPowerCL`. The feature map is
    unsupervised (per-trial Welch band-power plus optional channel-role-aware
    aggregates), so ``fit`` only stores configuration; ``transform`` calls the
    baseline's own ``_extract_features``. The output is the identical feature
    matrix that Baseline B2 classifies.
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

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "BandPowerTransformer":
        self.bandpower_ = BandPowerCL(
            sfreq=self.sfreq, bands=self.bands, channel_roles=self.channel_roles
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        bp = getattr(self, "bandpower_", None)
        if bp is None:  # transform without fit: construct an equivalent extractor.
            bp = BandPowerCL(
                sfreq=self.sfreq, bands=self.bands, channel_roles=self.channel_roles
            )
        return bp._extract_features(np.asarray(X))
