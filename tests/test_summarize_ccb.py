"""Unit tests for scripts/summarize_ccb.py factorial / aggregation logic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# pylint: disable=wrong-import-position
from summarize_ccb import (  # noqa: E402
    _factorial_cell_aggregate,
    _factorial_cell_label,
    _find_best_factorial_cell,
    _policy_ablation_rows,
)


def _make_factorial_fixture(
    alphas=(0.1, 1.0),
    cal_fracs=(0.2, 0.3),
    arm_pools=("pruned",),
    include_recent_rewards=(True, False),
    per_round_caps=("inf",),
    budget_fracs=(1.0,),
    policies=("oplb",),
    seeds=(0, 1, 42),
    subjects=(1, 2, 3),
    protocols=("within", "official"),
    base_kappa: float = 0.1,
    noise: float = 0.01,
) -> pd.DataFrame:
    """Construct a synthetic factorial CSV-like DataFrame.

    κ = base_kappa + 0.1 * alpha + 0.2 * (not cal_frac) + noise(seed, subject).
    Designed so that the cell (alpha=1.0, cal_frac=0.2, pool=pruned,
    include_recent_rewards=True) sits near the top but not deterministically.
    """
    rng = np.random.default_rng(123)
    rows = []
    for alpha in alphas:
        for cal in cal_fracs:
            for pool in arm_pools:
                for irr in include_recent_rewards:
                    for cap in per_round_caps:
                        for bf in budget_fracs:
                            for pol in policies:
                                for seed in seeds:
                                    for subj in subjects:
                                        for prot in protocols:
                                            kappa = (
                                                base_kappa
                                                + 0.1 * float(alpha)
                                                + 0.2 * (0.3 - cal)
                                                + rng.normal(0, noise)
                                            )
                                            rows.append(
                                                {
                                                    "dataset": "BCI-IV-2b",
                                                    "subject": subj,
                                                    "protocol": prot,
                                                    "fold_name": prot,
                                                    "seed": seed,
                                                    "budget_frac": bf,
                                                    "alpha": alpha,
                                                    "arm_pool": pool,
                                                    "calibration_frac": cal,
                                                    "include_recent_rewards": irr,
                                                    "per_round_cap": cap,
                                                    "policy": pol,
                                                    "epsilon": float("nan"),
                                                    "kappa": round(kappa, 4),
                                                    "accuracy": 0.6,
                                                    "n_test": 30,
                                                    "n_arms_surviving": 50,
                                                    "stream_rounds_run": 50,
                                                    "final_regret": 0.0,
                                                    "budget_remaining": 100.0,
                                                }
                                            )
    return pd.DataFrame(rows)


def test_factorial_aggregate_shape_and_keys():
    df = _make_factorial_fixture()
    agg = _factorial_cell_aggregate(df)
    # 2 alphas × 2 cal_fracs × 1 pool × 2 irr × 1 cap × 1 bf × 1 pol × 2 protocols = 16 rows.
    assert len(agg) == 16
    assert {"kappa_mean", "kappa_seed_std", "n_seeds", "n_subjects"}.issubset(agg.columns)
    # All cells have 3 seeds × 3 subjects, so n_seeds = 3, n_subjects = 3.
    assert (agg["n_seeds"] == 3).all()
    assert (agg["n_subjects"] == 3).all()


def test_find_best_factorial_cell_picks_highest_combined_kappa():
    """With κ = base + 0.1·α + 0.2·(0.3 − cal), α=1.0 / cal=0.2 should win."""
    df = _make_factorial_fixture(noise=0.0)  # deterministic
    best, rows = _find_best_factorial_cell(df)
    assert best is not None
    assert best["alpha"] == 1.0
    assert best["calibration_frac"] == 0.2
    assert rows is not None and not rows.empty
    # The selected slice should have exactly the rows matching the best cell.
    for _, r in rows.iterrows():
        assert r["alpha"] == 1.0
        assert r["calibration_frac"] == 0.2


def test_find_best_factorial_cell_handles_missing_protocol():
    """If a cell is missing one protocol, the function should return (None, None)."""
    df = _make_factorial_fixture()
    df = df[df["protocol"] == "within"].copy()  # drop official rows
    best, rows = _find_best_factorial_cell(df)
    assert best is None
    assert rows is None


def test_factorial_cell_label_includes_known_keys():
    cell = {
        "alpha": 0.5,
        "calibration_frac": 0.2,
        "arm_pool": "full",
        "include_recent_rewards": False,
        "per_round_cap": "inf",
        "budget_frac": 1.0,
        "policy": "oplb",
    }
    label = _factorial_cell_label(cell)
    assert "alpha=0.5" in label
    assert "calibration_frac=0.2" in label
    assert "arm_pool=full" in label
    assert "include_recent_rewards=False" in label


def test_policy_ablation_rows_shape():
    """Given a synthetic policy-ablation CSV, per-policy/per-protocol rows are emitted."""
    rows = []
    for policy in ("oplb", "fixed", "eps_greedy", "unconstrained"):
        for protocol in ("within", "official"):
            for subj in range(1, 4):
                for seed in (0, 1):
                    rows.append(
                        {
                            "policy": policy,
                            "protocol": protocol,
                            "subject": subj,
                            "seed": seed,
                            "kappa": 0.1,
                        }
                    )
    df = pd.DataFrame(rows)
    out = _policy_ablation_rows(df)
    # 4 policies × 2 protocols = 8 rows.
    assert len(out) == 8
    for r in out:
        assert r["policy"] in ("oplb", "fixed", "eps_greedy", "unconstrained")
        assert r["protocol"] in ("within", "official")
        assert r["n_subjects"] == 3
        assert r["n"] == 3 * 2  # subjects × seeds
