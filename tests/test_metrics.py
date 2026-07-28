"""Unit tests for thesis.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from thesis.metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    compute_metrics,
    compute_regression_metrics,
    gap_to_benchmark,
)


def test_perfect_prediction_gives_unit_accuracy_and_kappa():
    y_true = np.array(["left_hand", "right_hand", "left_hand", "right_hand"])
    y_pred = y_true.copy()
    m = compute_metrics(y_true, y_pred)
    assert m.accuracy == 1.0
    assert m.kappa == 1.0
    assert m.n_trials == 4


def test_chance_prediction_gives_zero_kappa():
    # 50/50 chance prediction on balanced classes → κ ≈ 0
    y_true = np.array(["left_hand"] * 100 + ["right_hand"] * 100)
    rng = np.random.default_rng(seed=0)
    y_pred = rng.choice(["left_hand", "right_hand"], size=200)
    m = compute_metrics(y_true, y_pred)
    assert abs(m.kappa) < 0.2  # seed-specific, but should be near zero


def test_gap_to_benchmark_sign():
    assert gap_to_benchmark(0.6, 0.4) == pytest.approx(0.2)  # benchmark wins
    assert gap_to_benchmark(0.3, 0.5) == pytest.approx(-0.2)  # policy wins


def test_metrics_is_hashable_dataclass():
    m = compute_metrics(
        np.array(["left_hand", "right_hand"]),
        np.array(["left_hand", "right_hand"]),
    )
    assert isinstance(m, ClassificationMetrics)
    # frozen → hashable
    assert {m: 1}[m] == 1


# ---------------------------------------------------------------------------
# Phase-5: RegressionMetrics stub (forward-compat only, not wired in)
# ---------------------------------------------------------------------------


def test_regression_metrics_perfect_prediction():
    y = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    m = compute_regression_metrics(y, y)
    assert isinstance(m, RegressionMetrics)
    assert m.mse == pytest.approx(0.0)
    assert m.rmse == pytest.approx(0.0)
    assert m.r2 == pytest.approx(1.0)
    assert m.n_trials == 5


def test_regression_metrics_mean_prediction_gives_zero_r2():
    """Predicting the mean every time → R² = 0 by definition."""
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_hat = np.full_like(y, y.mean())
    m = compute_regression_metrics(y, y_hat)
    assert m.r2 == pytest.approx(0.0)
    assert m.mse == pytest.approx(y.var())


def test_regression_metrics_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_regression_metrics(np.zeros(5), np.zeros(4))


def test_regression_metrics_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        compute_regression_metrics(np.array([]), np.array([]))


def test_regression_metrics_constant_target_r2_is_zero():
    """When the target is constant, R² falls back to 0 (guarded, no div-by-zero)."""
    y = np.full(5, 0.5)
    y_hat = np.array([0.4, 0.5, 0.6, 0.5, 0.5])
    m = compute_regression_metrics(y, y_hat)
    assert m.r2 == 0.0
    assert m.mse > 0
