"""Unit tests for the classical-classifier comparators (B3--B5).

Mirrors the synthetic-signal pattern of ``test_fbcsp.py`` and
``test_bandpower_cl.py``: inject a clean, class-dependent band-power signal so
each (feature-family x classifier-head) pairing has a real signal to latch
onto. If a head cannot beat chance here, the wiring is wrong before we run it on
real data. Also asserts the leakage-safety property the comparison relies on:
the FBCSP feature transform is pure (transforming a held-out fold neither
mutates the fitted CSP state nor changes previously-computed train features).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from thesis.baselines.classical import (
    CLASSICAL_CLASSIFIERS,
    make_classical_pipeline,
    make_classifier,
    make_feature_transformer,
)
from thesis.baselines.feature_transformers import BandPowerTransformer, FBCSPTransformer


@pytest.fixture
def synthetic_mi(seed: int = 0):
    """Binary MI: classes differ in 12 Hz power on distinct channel groups.

    Eight channels so the default FBCSP ``n_components=4`` is comfortably within
    range (no MNE capping in the test).
    """
    rng = np.random.default_rng(seed)
    n_per_class = 60
    n_channels = 8
    n_samples = 1000
    sfreq = 250.0
    t = np.arange(n_samples) / sfreq

    X0 = rng.standard_normal((n_per_class, n_channels, n_samples)) * 0.5
    X0[:, 0:2, :] += 2.0 * np.sin(2 * np.pi * 12 * t)  # left group active
    X1 = rng.standard_normal((n_per_class, n_channels, n_samples)) * 0.5
    X1[:, 6:8, :] += 2.0 * np.sin(2 * np.pi * 12 * t)  # right group active

    X = np.concatenate([X0, X1], axis=0)
    y = np.array(["left"] * n_per_class + ["right"] * n_per_class)
    shuffle = rng.permutation(len(y))
    return X[shuffle], y[shuffle], sfreq


@pytest.fixture
def synthetic_cl(seed: int = 0):
    """Binary CL: low = alpha-dominant, high = beta-dominant (mimics WAUC)."""
    rng = np.random.default_rng(seed)
    n_per_class = 60
    n_channels = 8
    n_samples = 1000
    sfreq = 250.0
    t = np.arange(n_samples) / sfreq

    X_low = rng.standard_normal((n_per_class, n_channels, n_samples)) * 0.5
    X_low += 2.0 * np.sin(2 * np.pi * 10 * t)  # alpha dominant
    X_high = rng.standard_normal((n_per_class, n_channels, n_samples)) * 0.5
    X_high += 2.0 * np.sin(2 * np.pi * 20 * t)  # beta dominant

    X = np.concatenate([X_low, X_high], axis=0)
    y = np.array(["low"] * n_per_class + ["high"] * n_per_class)
    shuffle = rng.permutation(len(y))
    return X[shuffle], y[shuffle], sfreq


# --------------------------------------------------------------------------- #
# Factory contracts
# --------------------------------------------------------------------------- #


def test_classical_classifiers_constant():
    assert CLASSICAL_CLASSIFIERS == ("svm", "decision_tree", "random_forest")


def test_make_classifier_types():
    assert isinstance(make_classifier("svm"), Pipeline)  # scaler + SVC
    assert isinstance(make_classifier("decision_tree"), DecisionTreeClassifier)
    assert isinstance(make_classifier("random_forest"), RandomForestClassifier)
    assert isinstance(make_classifier("lda"), LinearDiscriminantAnalysis)


def test_make_classifier_unknown_raises():
    with pytest.raises(ValueError, match="unknown classifier"):
        make_classifier("xgboost")


def test_make_feature_transformer_types():
    assert isinstance(make_feature_transformer("fbcsp"), FBCSPTransformer)
    assert isinstance(make_feature_transformer("bandpower"), BandPowerTransformer)


def test_make_feature_transformer_unknown_raises():
    with pytest.raises(ValueError, match="unknown feature family"):
        make_feature_transformer("riemann")


def test_svm_has_scaler_trees_do_not():
    svm = make_classifier("svm")
    assert "scale" in dict(svm.named_steps)
    assert not isinstance(make_classifier("decision_tree"), Pipeline)
    assert not isinstance(make_classifier("random_forest"), Pipeline)


# --------------------------------------------------------------------------- #
# Each (feature family x head) beats chance on a clean synthetic signal
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("clf_name", CLASSICAL_CLASSIFIERS)
def test_fbcsp_features_each_head_beats_chance(synthetic_mi, clf_name):
    X, y, sfreq = synthetic_mi
    split = len(y) // 2
    pipe = make_classical_pipeline("fbcsp", clf_name, sfreq=sfreq)
    pipe.fit(X[:split], y[:split])
    acc = (pipe.predict(X[split:]) == y[split:]).mean()
    assert acc > 0.70, f"fbcsp+{clf_name}: expected > 0.70 on synthetic MI; got {acc:.3f}"


@pytest.mark.parametrize("clf_name", CLASSICAL_CLASSIFIERS)
def test_bandpower_features_each_head_beats_chance(synthetic_cl, clf_name):
    X, y, sfreq = synthetic_cl
    split = len(y) // 2
    pipe = make_classical_pipeline("bandpower", clf_name, sfreq=sfreq)
    pipe.fit(X[:split], y[:split])
    acc = (pipe.predict(X[split:]) == y[split:]).mean()
    assert acc > 0.70, f"bandpower+{clf_name}: expected > 0.70 on synthetic CL; got {acc:.3f}"


# --------------------------------------------------------------------------- #
# Leakage-safety of the supervised FBCSP transform
# --------------------------------------------------------------------------- #


def test_fbcsp_transform_is_pure_and_does_not_mutate_fitted_state(synthetic_mi):
    X, y, sfreq = synthetic_mi
    split = len(y) // 2
    X_train, y_train = X[:split], y[:split]
    X_test = X[split:]

    tr = FBCSPTransformer(sfreq=sfreq).fit(X_train, y_train)
    feats_train_before = tr.transform(X_train)
    csp0_before = tr.fbcsp_.csps_[0].filters_.copy()

    # Transforming a held-out fold must not change fitted CSP filters …
    _ = tr.transform(X_test)
    csp0_after = tr.fbcsp_.csps_[0].filters_
    np.testing.assert_array_equal(csp0_before, csp0_after)

    # … and re-transforming the train fold must reproduce the same features.
    feats_train_after = tr.transform(X_train)
    np.testing.assert_allclose(feats_train_before, feats_train_after)


def test_fbcsp_transformer_requires_labels(synthetic_mi):
    X, _y, sfreq = synthetic_mi
    with pytest.raises(ValueError, match="requires labels"):
        FBCSPTransformer(sfreq=sfreq).fit(X, None)


def test_bandpower_transform_matches_baseline_features(synthetic_cl):
    """The transformer must reproduce BandPowerCL's own feature matrix exactly."""
    from thesis.baselines.bandpower_cl import BandPowerCL

    X, _y, sfreq = synthetic_cl
    tr = BandPowerTransformer(sfreq=sfreq).fit(X)
    direct = BandPowerCL(sfreq=sfreq)._extract_features(X)
    np.testing.assert_allclose(tr.transform(X), direct)


def test_tree_head_predict_proba_sums_to_one(synthetic_cl):
    X, y, sfreq = synthetic_cl
    pipe = make_classical_pipeline("bandpower", "random_forest", sfreq=sfreq)
    pipe.fit(X[:80], y[:80])
    proba = pipe.predict_proba(X[80:85])
    assert proba.shape == (5, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
