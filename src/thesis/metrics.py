"""Metrics for thesis evaluation.

Phase-3/4 headline is the **gap-to-benchmark** Δκ between the 22-ch MI
benchmark and the 3-ch CCB policy. Phase-5 Stage-3 adds cognitive-load
ingestion, which is still classification (STEW 9-point → low/med/high per
the Phase-5 plan) and stays on :class:`ClassificationMetrics`.

The :class:`RegressionMetrics` dataclass is a forward-compat stub for a
possible future regression-bandit extension (continuous workload). It is
**not** wired into any Phase-5 code path — if a future commit starts using
it the runner + summariser will need parallel updates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


@dataclass(frozen=True)
class ClassificationMetrics:
    """Accuracy and Cohen's κ for one evaluation fold."""

    accuracy: float
    kappa: float
    n_trials: int

    def __repr__(self) -> str:
        return f"acc={self.accuracy:.3f} κ={self.kappa:.3f} n={self.n_trials}"


@dataclass(frozen=True)
class RegressionMetrics:
    """Regression scores for continuous-label targets (e.g. continuous workload).

    Phase-5 forward-compat only: currently unused by any runner or script.
    Added here so a future continuous-reward bandit extension doesn't have
    to refactor the metrics module. The thesis plan explicitly scopes
    cognitive load to classification; this dataclass is scaffolding.
    """

    mse: float
    rmse: float
    r2: float
    n_trials: int

    def __repr__(self) -> str:
        return f"mse={self.mse:.4f} rmse={self.rmse:.4f} r²={self.r2:.3f} n={self.n_trials}"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        kappa=float(cohen_kappa_score(y_true, y_pred)),
        n_trials=int(len(y_true)),
    )


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Compute MSE / RMSE / R² on a continuous target.

    Kept minimal on purpose — Phase-5 stays classification-first. If a
    future phase wires this in, reconsider the signature (e.g. weights,
    multi-output).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("empty input to compute_regression_metrics")
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    # R² falls back to 0 when the target is constant (ss_tot = 0) — any
    # non-zero residual at that point is arbitrarily bad; guard explicitly.
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return RegressionMetrics(mse=mse, rmse=rmse, r2=float(r2), n_trials=int(len(y_true)))


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


def gap_to_benchmark(benchmark_kappa: float, policy_kappa: float) -> float:
    """Δκ = κ_benchmark − κ_policy. Positive values mean the benchmark wins."""
    return float(benchmark_kappa - policy_kappa)
