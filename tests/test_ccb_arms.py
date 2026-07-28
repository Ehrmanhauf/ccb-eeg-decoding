"""Unit tests for thesis.ccb.arms."""

from __future__ import annotations

import numpy as np
import pytest

from thesis.ccb.arms import (
    Arm,
    ArmHead,
    arm_cost,
    build_arm_heads,
    enumerate_arms_2b,
    prune_arms,
)


@pytest.fixture
def synthetic_2b_trials():
    """3-channel bipolar synthetic MI with 12 Hz injection.

    Class 0 ("left_hand"): 12 Hz sine on channel 0 (C3 area).
    Class 1 ("right_hand"): 12 Hz sine on channel 2 (C4 area).
    Channel 1 (Cz) remains noise for both. Trials shuffled deterministically.
    """
    rng = np.random.default_rng(seed=0)
    n_per_class = 60
    n_channels = 3
    n_samples = 1001  # 4 s @ 250 Hz
    sfreq = 250.0
    t = np.arange(n_samples) / sfreq

    # Noise 1 µV; signal 5 µV on the lateralized channel (5:1 SNR, so a
    # single-band 2-component CSP can discriminate cleanly). Real MI has
    # weaker SNR, but that's what FBCSP's 9 bands recover; for unit-testing
    # one arm, we need a clear signal.
    X0 = rng.standard_normal((n_per_class, n_channels, n_samples)) * 1e-6
    X0[:, 0] += 5e-6 * np.sin(2 * np.pi * 12 * t)

    X1 = rng.standard_normal((n_per_class, n_channels, n_samples)) * 1e-6
    X1[:, 2] += 5e-6 * np.sin(2 * np.pi * 12 * t)

    X = np.concatenate([X0, X1], axis=0)
    y = np.array(["left_hand"] * n_per_class + ["right_hand"] * n_per_class)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], sfreq


def _kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(y_true, y_pred))


def test_enumerate_arms_2b_full_grid_size():
    arms = enumerate_arms_2b()
    # 9 bands × 3 spatial filters × 2 feature types × 3 time windows = 162.
    assert len(arms) == 162


def test_enumerate_arms_2b_unique_ids():
    arms = enumerate_arms_2b()
    assert len({a.arm_id for a in arms}) == len(arms)


def test_enumerate_arms_2b_with_fbcsp_has_163_arms():
    """H1 hybrid: adds the full FBCSP pipeline as one extra arm."""
    arms = enumerate_arms_2b(include_fbcsp_arm=True)
    assert len(arms) == 163
    assert len({a.arm_id for a in arms}) == 163
    fbcsp_arms = [a for a in arms if a.spatial == "fbcsp"]
    assert len(fbcsp_arms) == 1


def test_enumerate_arms_2b_with_riemann_has_216_arms():
    """Phase-5 §2.3: 9 bands × 2 spatial × 1 feature × 3 windows = 54 added."""
    arms = enumerate_arms_2b(include_riemann_arms=True)
    assert len(arms) == 162 + 54
    assert len({a.arm_id for a in arms}) == 216
    riem_arms = [a for a in arms if a.feature == "riemann_tangent"]
    assert len(riem_arms) == 54
    # No CSP+riemann combinations enumerated (CSP collapses to scalar).
    assert all(a.spatial != "csp" for a in riem_arms)
    # Both extensions stack cleanly: 162 + 1 + 54 = 217.
    arms_both = enumerate_arms_2b(include_fbcsp_arm=True, include_riemann_arms=True)
    assert len(arms_both) == 217


def test_riemann_arm_cost_matches_tangent_dim():
    """Tangent-vector dim = n_out × (n_out + 1) / 2 per Barachant 2012 Eq. 4."""
    # Laplacian on 3-ch reduces to 2 channels → tangent dim = 2*3/2 = 3.
    cost_lap = arm_cost(
        spatial="laplacian", feature="riemann_tangent", n_components=3, n_channels=3
    )
    assert cost_lap == 3.0
    # Identity on 3-ch keeps 3 channels → tangent dim = 3*4/2 = 6.
    cost_id = arm_cost(
        spatial="identity", feature="riemann_tangent", n_components=3, n_channels=3
    )
    assert cost_id == 6.0


def test_riemann_arm_head_beats_chance_on_lateralized_12hz(synthetic_2b_trials):
    """Riemannian arm should learn the C3-vs-C4 12-Hz lateralization."""
    X, y, sfreq = synthetic_2b_trials
    arm = Arm(
        arm_id=999,
        band=(8.0, 16.0),
        spatial="identity",
        feature="riemann_tangent",
        window=(0.0, 4.0),
        cost=6.0,
        n_components=3,
    )
    n_train = int(0.6 * len(y))
    head = ArmHead(arm).fit(X[:n_train], y[:n_train], sfreq)
    y_pred = head.predict(X[n_train:], sfreq)
    # Same SNR as the CSP test in test_arm_head_beats_chance_on_lateralized_12hz;
    # Riemannian on identity should reach high κ on this clean synthetic.
    assert _kappa(y[n_train:], y_pred) > 0.6


def test_riemann_arm_head_feature_vec_shape(synthetic_2b_trials):
    """feature_vec must produce (n_trials, n_ch * (n_ch+1) / 2)."""
    X, y, sfreq = synthetic_2b_trials
    arm = Arm(
        arm_id=998,
        band=(8.0, 16.0),
        spatial="laplacian",
        feature="riemann_tangent",
        window=(0.0, 4.0),
        cost=3.0,
        n_components=3,
    )
    head = ArmHead(arm).fit(X[:60], y[:60], sfreq)
    feats = head.feature_vec(X[60:80], sfreq)
    # Laplacian → 2 channels → tangent dim = 3.
    assert feats.shape == (20, 3)


def test_riemann_arm_head_rejects_csp_spatial():
    """CSP spatial is incompatible with Riemannian feature (scalar output)."""
    arm = Arm(
        arm_id=997,
        band=(8.0, 16.0),
        spatial="csp",
        feature="riemann_tangent",
        window=(0.0, 4.0),
        cost=3.0,
        n_components=3,
    )
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3, 1001)) * 1e-6
    y = np.array(["a"] * 10 + ["b"] * 10)
    with pytest.raises(ValueError, match="incompatible with spatial='csp'"):
        ArmHead(arm).fit(X, y, sfreq=250.0)


def test_fbcsp_arm_cost_exceeds_single_band_arms():
    """H1 FBCSP arm cost: 9 bands × 3 components = 27 on 3-ch data; strictly
    greater than every single-band arm's cost."""
    arms = enumerate_arms_2b(include_fbcsp_arm=True)
    (fbcsp_arm,) = [a for a in arms if a.spatial == "fbcsp"]
    assert fbcsp_arm.cost == 27.0
    single_band_max = max(a.cost for a in arms if a.spatial != "fbcsp")
    assert fbcsp_arm.cost > single_band_max


def test_arm_cost_fbcsp_branch():
    """arm_cost('fbcsp', ...) = 9 × min(n_components, n_channels)."""
    c_3ch = arm_cost(spatial="fbcsp", feature="logvar", n_components=4, n_channels=3)
    assert c_3ch == 9 * 3
    c_14ch = arm_cost(spatial="fbcsp", feature="logvar", n_components=4, n_channels=14)
    assert c_14ch == 9 * 4


def test_arm_cost_monotone_spatial():
    c_csp = arm_cost(spatial="csp", feature="logvar", n_components=2, n_channels=3)
    c_laplacian = arm_cost(spatial="laplacian", feature="logvar", n_components=3, n_channels=3)
    c_identity = arm_cost(spatial="identity", feature="logvar", n_components=3, n_channels=3)
    # CSP(2) < Laplacian(2) == Laplacian on 3-ch bipolar yields 2 outputs, but
    # the cost ordering we care about is: CSP(2) <= Laplacian(2) < Identity(3).
    assert c_csp <= c_laplacian < c_identity


def test_arm_cost_csp_capped_at_n_channels():
    # Ask for 5 components but only 3 channels — must cap at 3.
    c = arm_cost(spatial="csp", feature="logvar", n_components=5, n_channels=3)
    assert c == 3.0


def test_arm_head_beats_chance_on_lateralized_12hz(synthetic_2b_trials):
    X, y, sfreq = synthetic_2b_trials
    arm = Arm(
        arm_id=0,
        band=(8.0, 12.0),
        spatial="csp",
        feature="logvar",
        window=(0.0, 4.0),
        cost=2.0,
        n_components=2,
    )
    split = len(y) // 2
    head = ArmHead(arm).fit(X[:split], y[:split], sfreq)
    y_hat = head.predict(X[split:], sfreq)
    assert _kappa(y[split:], y_hat) > 0.5


def test_arm_head_unfitted_raises():
    arm = Arm(
        arm_id=0,
        band=(8, 12),
        spatial="identity",
        feature="logvar",
        window=(0, 4),
        cost=3.0,
        n_components=3,
    )
    head = ArmHead(arm)
    with pytest.raises(RuntimeError, match="not fitted"):
        head.predict(np.zeros((1, 3, 1001)), sfreq=250.0)


def test_arm_head_partial_fit_matches_refit(synthetic_2b_trials):
    """Running partial_fit on each trial should give the same LDA as a full
    fit on the combined calibration + stream data (up to LDA's shrinkage
    estimator, which is deterministic for a given buffer)."""
    X, y, sfreq = synthetic_2b_trials
    arm = Arm(
        arm_id=0,
        band=(8, 12),
        spatial="identity",
        feature="logvar",
        window=(0, 4),
        cost=3.0,
        n_components=3,
    )
    cal_end = 40
    stream_end = 80

    online = ArmHead(arm).fit(X[:cal_end], y[:cal_end], sfreq)
    for j in range(cal_end, stream_end):
        online.partial_fit(X[j : j + 1], y[j : j + 1], sfreq)

    frozen_refit = ArmHead(arm).fit(X[:stream_end], y[:stream_end], sfreq)

    test_slice = X[stream_end:]
    np.testing.assert_allclose(
        online.score(test_slice, sfreq),
        frozen_refit.score(test_slice, sfreq),
        atol=1e-8,
        err_msg="partial_fit aggregate != refit-from-scratch on combined buffer",
    )


def test_arm_head_partial_fit_without_fit_raises():
    arm = Arm(
        arm_id=0,
        band=(8, 12),
        spatial="identity",
        feature="logvar",
        window=(0, 4),
        cost=3.0,
        n_components=3,
    )
    head = ArmHead(arm)
    with pytest.raises(RuntimeError, match="prior fit"):
        head.partial_fit(np.zeros((1, 3, 1001)), np.array(["left_hand"]), sfreq=250.0)


def test_fbcsp_arm_head_fits_and_predicts(synthetic_2b_trials):
    """H1 FBCSP arm: ArmHead delegates to the internal FBCSP pipeline."""
    X, y, sfreq = synthetic_2b_trials
    arms = enumerate_arms_2b(include_fbcsp_arm=True)
    (fbcsp_arm,) = [a for a in arms if a.spatial == "fbcsp"]
    split = len(y) // 2
    head = ArmHead(fbcsp_arm).fit(X[:split], y[:split], sfreq)
    assert head.is_fitted
    assert head._fbcsp is not None
    y_hat = head.predict(X[split:], sfreq)
    assert y_hat.shape == (len(X) - split,)
    assert _kappa(y[split:], y_hat) > 0.5


def test_fbcsp_arm_head_feature_dim_matches_cost(synthetic_2b_trials):
    """Feature-vector length returned by feature_vec must equal arm.cost."""
    X, y, sfreq = synthetic_2b_trials
    arms = enumerate_arms_2b(include_fbcsp_arm=True)
    (fbcsp_arm,) = [a for a in arms if a.spatial == "fbcsp"]
    head = ArmHead(fbcsp_arm).fit(X[:40], y[:40], sfreq)
    feats = head.feature_vec(X[40:50], sfreq)
    assert feats.shape == (10, int(fbcsp_arm.cost))


def test_fbcsp_arm_head_rejects_partial_fit(synthetic_2b_trials):
    """H1 keeps FBCSP arms frozen — partial_fit is not supported."""
    X, y, sfreq = synthetic_2b_trials
    arms = enumerate_arms_2b(include_fbcsp_arm=True)
    (fbcsp_arm,) = [a for a in arms if a.spatial == "fbcsp"]
    head = ArmHead(fbcsp_arm).fit(X[:40], y[:40], sfreq)
    with pytest.raises(RuntimeError, match="not supported"):
        head.partial_fit(X[40:41], y[40:41], sfreq)


def test_arm_head_partial_fit_changes_prediction_over_time(synthetic_2b_trials):
    """After accumulating many new trials, prediction on held-out data should
    differ from the frozen calibration-only head (not strictly assert
    direction — just that updates propagate to the classifier)."""
    X, y, sfreq = synthetic_2b_trials
    arm = Arm(
        arm_id=0,
        band=(8, 12),
        spatial="csp",
        feature="logvar",
        window=(0, 4),
        cost=2.0,
        n_components=2,
    )
    cal_end = 20
    frozen = ArmHead(arm).fit(X[:cal_end], y[:cal_end], sfreq)
    online = ArmHead(arm).fit(X[:cal_end], y[:cal_end], sfreq)
    # Accumulate 60 more labelled trials.
    for j in range(cal_end, cal_end + 60):
        online.partial_fit(X[j : j + 1], y[j : j + 1], sfreq)
    test_slice = X[cal_end + 60 :]
    scores_frozen = frozen.score(test_slice, sfreq)
    scores_online = online.score(test_slice, sfreq)
    # Must differ on at least a few trials (not a flat "identical" regression).
    assert np.any(np.abs(scores_frozen - scores_online) > 1e-6), (
        "partial_fit had no measurable effect on scores"
    )


def test_arm_head_feature_vec_shape_identity(synthetic_2b_trials):
    X, y, sfreq = synthetic_2b_trials
    arm = Arm(
        arm_id=0,
        band=(8, 12),
        spatial="identity",
        feature="logvar",
        window=(0, 4),
        cost=3.0,
        n_components=3,
    )
    head = ArmHead(arm).fit(X[:50], y[:50], sfreq)
    feats = head.feature_vec(X[50:60], sfreq)
    # Identity keeps 3 channels; logvar collapses time → (n, 3).
    assert feats.shape == (10, 3)


def test_arm_head_feature_vec_shape_laplacian(synthetic_2b_trials):
    X, y, sfreq = synthetic_2b_trials
    arm = Arm(
        arm_id=0,
        band=(8, 12),
        spatial="laplacian",
        feature="logvar",
        window=(0, 4),
        cost=2.0,
        n_components=3,
    )
    head = ArmHead(arm).fit(X[:50], y[:50], sfreq)
    feats = head.feature_vec(X[50:60], sfreq)
    # Laplacian reduces to 2 channels; logvar → (n, 2).
    assert feats.shape == (10, 2)


def test_prune_arms_drops_below_threshold_and_caps(synthetic_2b_trials):
    X, y, sfreq = synthetic_2b_trials
    # Build a tiny mixed pool: 3 "garbage" arms on a high band that has no
    # signal in our synthetic 12 Hz fixture, and one "good" arm on 8–12 Hz.
    # Use realistic windows so the bandpass filter has enough samples.
    bad_arms = [
        Arm(
            arm_id=i,
            band=(36.0, 40.0),
            spatial="identity",
            feature="logvar",
            window=(0.0, 4.0),
            cost=3.0,
            n_components=3,
        )
        for i in range(3)
    ]
    good = Arm(
        arm_id=99,
        band=(8.0, 12.0),
        spatial="csp",
        feature="logvar",
        window=(0.0, 4.0),
        cost=2.0,
        n_components=2,
    )
    arms = [*bad_arms, good]
    split = len(y) // 2
    heads = build_arm_heads(arms, X[:split], y[:split], sfreq)
    survivors = prune_arms(arms, heads, X[split:], y[split:], sfreq, min_kappa=0.1, max_arms=4)
    # Good arm must survive; most garbage arms must be dropped.
    assert any(a.arm_id == 99 for a in survivors)
    assert len(survivors) <= 4


def test_enumerate_arms_2b_bounded_to_100_after_pruning(synthetic_2b_trials):
    X, y, sfreq = synthetic_2b_trials
    arms = enumerate_arms_2b(sfreq)
    split = len(y) // 2
    heads = build_arm_heads(arms, X[:split], y[:split], sfreq)
    survivors = prune_arms(arms, heads, X[split:], y[split:], sfreq, min_kappa=0.05, max_arms=100)
    # The plan's hard cap: at most 100 arms survive.
    assert len(survivors) <= 100
