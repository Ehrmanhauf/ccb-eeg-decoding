"""Unit tests for thesis.ccb.context."""

from __future__ import annotations

import numpy as np
import pytest

from thesis.ccb.context import (
    N_ARM_FAMILIES,
    compute_context,
    context_dim,
)


def _make_epoch(seed: int = 0, n_samples: int = 1001, scale: float = 1e-6) -> np.ndarray:
    """Synthetic 3-channel bipolar epoch at volt scale (MNE convention)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((3, n_samples)) * scale


def test_context_dim_with_recent_rewards_equals_18():
    assert context_dim(include_recent_rewards=True, n_recent_arms=N_ARM_FAMILIES) == 18


def test_context_dim_without_recent_rewards_equals_15():
    assert context_dim(include_recent_rewards=False) == 15


def test_compute_context_shape_with_recent_rewards():
    epoch = _make_epoch()
    ctx = compute_context(
        epoch,
        sfreq=250.0,
        recent_arm_rewards=np.zeros(N_ARM_FAMILIES),
    )
    assert ctx.shape == (18,)


def test_compute_context_shape_without_recent_rewards():
    epoch = _make_epoch()
    ctx = compute_context(epoch, sfreq=250.0, recent_arm_rewards=None)
    assert ctx.shape == (15,)


def test_compute_context_finite():
    epoch = _make_epoch()
    ctx = compute_context(epoch, sfreq=250.0, recent_arm_rewards=np.zeros(N_ARM_FAMILIES))
    assert np.all(np.isfinite(ctx))


def test_artifact_flag_fires_over_100uv():
    epoch = _make_epoch()
    epoch[1, 500] = 150e-6  # inject a 150 µV spike
    ctx = compute_context(epoch, sfreq=250.0, recent_arm_rewards=np.zeros(N_ARM_FAMILIES))
    # Index 9 per the feature layout in the module docstring.
    assert ctx[9] == 1.0


def test_artifact_flag_stays_off_below_100uv():
    epoch = _make_epoch(scale=5e-6)  # comfortably under 100 µV
    ctx = compute_context(epoch, sfreq=250.0, recent_arm_rewards=np.zeros(N_ARM_FAMILIES))
    assert ctx[9] == 0.0


def test_bias_term_is_one():
    epoch = _make_epoch()
    ctx = compute_context(epoch, sfreq=250.0, recent_arm_rewards=np.zeros(N_ARM_FAMILIES))
    # Bias at index 14 per the feature layout.
    assert ctx[14] == 1.0


def test_recent_rewards_appended_at_end():
    epoch = _make_epoch()
    recent = np.array([0.3, 0.7, 0.5])
    ctx = compute_context(epoch, sfreq=250.0, recent_arm_rewards=recent)
    np.testing.assert_allclose(ctx[15:], recent)


def test_wrong_channel_count_raises():
    bad_epoch = np.zeros((5, 1001))
    with pytest.raises(ValueError, match="3-channel"):
        compute_context(bad_epoch, sfreq=250.0, recent_arm_rewards=None)


def test_wrong_recent_rewards_length_raises():
    epoch = _make_epoch()
    with pytest.raises(ValueError, match="must have length"):
        compute_context(epoch, sfreq=250.0, recent_arm_rewards=np.zeros(7))


def test_context_reproducible():
    epoch = _make_epoch(seed=123)
    recent = np.array([0.1, 0.2, 0.3])
    c1 = compute_context(epoch, sfreq=250.0, recent_arm_rewards=recent)
    c2 = compute_context(epoch, sfreq=250.0, recent_arm_rewards=recent)
    np.testing.assert_array_equal(c1, c2)
