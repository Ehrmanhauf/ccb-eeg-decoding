"""Baseline classifiers.

- B1: FBCSP + shrinkage LDA (`ang2012fbcsp`) — :class:`FBCSP`.
- B2: cognitive-load band-power + shrinkage LDA — :class:`BandPowerCL`.
- B3--B5: classical-classifier comparators (SVM / Decision Tree / Random Forest)
  on the same engineered features, via :mod:`thesis.baselines.classical` and the
  leakage-safe transformers in :mod:`thesis.baselines.feature_transformers`.
"""

from thesis.baselines.bandpower_cl import DEFAULT_CL_BANDS, BandPowerCL
from thesis.baselines.classical import (
    CLASSICAL_CLASSIFIERS,
    CLASSIFIER_LABELS,
    FEATURE_FAMILIES,
    make_classical_pipeline,
    make_classifier,
    make_feature_transformer,
)
from thesis.baselines.fbcsp import DEFAULT_BANDS, FBCSP
from thesis.baselines.feature_transformers import BandPowerTransformer, FBCSPTransformer

__all__ = [
    "FBCSP",
    "DEFAULT_BANDS",
    "BandPowerCL",
    "DEFAULT_CL_BANDS",
    "FBCSPTransformer",
    "BandPowerTransformer",
    "make_classifier",
    "make_feature_transformer",
    "make_classical_pipeline",
    "CLASSICAL_CLASSIFIERS",
    "FEATURE_FAMILIES",
    "CLASSIFIER_LABELS",
]
