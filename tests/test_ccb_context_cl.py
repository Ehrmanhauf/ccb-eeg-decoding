"""Unit tests for thesis.ccb.context_cl — workload-specific context."""

from __future__ import annotations

import numpy as np
import pytest

from thesis.ccb.context_cl import (
    compute_context_workload,
    context_dim_workload,
)


def _epoch(n_channels: int, n_samples: int = 512, sfreq: float = 128.0, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_channels, n_samples)) * 1e-6, sfreq


def test_dim_without_recent_rewards():
    assert context_dim_workload(include_recent_rewards=False) == 9


def test_dim_with_recent_rewards_default_3():
    assert context_dim_workload(include_recent_rewards=True) == 12


def test_dim_with_recent_rewards_custom():
    assert context_dim_workload(include_recent_rewards=True, n_recent_arms=5) == 14


def test_compute_shape_without_recent_rewards():
    X, sfreq = _epoch(14)
    ctx = compute_context_workload(
        X, sfreq=sfreq, channel_roles={"frontal": [0, 1], "parietal": [5, 6]}
    )
    assert ctx.shape == (9,)
    assert np.isfinite(ctx).all()


def test_compute_shape_with_recent_rewards():
    X, sfreq = _epoch(14)
    ctx = compute_context_workload(
        X,
        sfreq=sfreq,
        channel_roles={"frontal": [0, 1], "parietal": [5, 6]},
        recent_arm_rewards=np.zeros(3),
    )
    assert ctx.shape == (12,)


def test_compute_rejects_non_2d_input():
    X = np.zeros((1, 14, 512))
    with pytest.raises(ValueError, match="2-D epoch"):
        compute_context_workload(X, sfreq=128.0, channel_roles={})


def test_missing_channel_roles_zero_out():
    X, sfreq = _epoch(14)
    ctx_empty = compute_context_workload(X, sfreq=sfreq, channel_roles={})
    # frontal_theta (index 3), parietal_alpha (index 4), asymmetry (index 5)
    # all zero when no roles provided.
    assert ctx_empty[3] == 0.0
    assert ctx_empty[4] == 0.0
    assert ctx_empty[5] == 0.0
    # Base bands (θ, α, β means) still computed from all channels.
    assert np.isfinite(ctx_empty[0:3]).all()


def test_artifact_flag_triggers_above_threshold():
    X, sfreq = _epoch(14)
    X[0, 100] = 200e-6  # above 150 μV threshold
    ctx = compute_context_workload(X, sfreq=sfreq, channel_roles={})
    assert ctx[7] == 1.0  # artifact flag position


def test_artifact_flag_off_when_clean():
    X, sfreq = _epoch(14)
    ctx = compute_context_workload(X, sfreq=sfreq, channel_roles={})
    assert ctx[7] == 0.0


def test_bias_is_constant_one():
    X, sfreq = _epoch(14)
    ctx = compute_context_workload(X, sfreq=sfreq, channel_roles={})
    assert ctx[8] == 1.0


def test_frontal_asymmetry_responds_to_lateralised_alpha():
    """Inject a 10 Hz (α-band) tone on F3 — asymmetry F3-F4 must go positive."""
    rng = np.random.default_rng(0)
    sfreq = 128.0
    n_samples = 512
    t = np.arange(n_samples) / sfreq
    X = rng.standard_normal((14, n_samples)) * 1e-6
    X[2] += 5e-6 * np.sin(2 * np.pi * 10 * t)  # channel 2 plays F3
    roles = {"f3": [2], "f4": [3]}
    ctx = compute_context_workload(X, sfreq=sfreq, channel_roles=roles)
    assert ctx[5] > 0, f"expected positive F3-F4 α asymmetry; got {ctx[5]:.3f}"


def test_engagement_rises_with_beta_dominant_signal():
    """More β power relative to α+θ → engagement (index 6) should rise."""
    rng = np.random.default_rng(0)
    sfreq = 128.0
    n_samples = 512
    t = np.arange(n_samples) / sfreq

    X_alpha = rng.standard_normal((14, n_samples)) * 1e-6
    X_alpha[:3] += 10e-6 * np.sin(2 * np.pi * 10 * t)[None, :]
    X_beta = rng.standard_normal((14, n_samples)) * 1e-6
    X_beta[:3] += 10e-6 * np.sin(2 * np.pi * 20 * t)[None, :]

    e_alpha = compute_context_workload(X_alpha, sfreq=sfreq, channel_roles={})[6]
    e_beta = compute_context_workload(X_beta, sfreq=sfreq, channel_roles={})[6]
    assert e_beta > e_alpha, f"expected e_beta > e_alpha; got {e_beta=:.3f} {e_alpha=:.3f}"
