"""Unit tests for thesis.ccb.runner.

Uses a synthetic 2b-shaped SubjectData so the tests run offline and fast —
no EEG files touched, no 2a code path exercised.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from thesis.ccb import runner as runner_mod
from thesis.ccb.arms import Arm, enumerate_arms_2b
from thesis.ccb.runner import (
    CCBResult,
    _calibration_stream_split,
    fit_heads_on_calibration,
    run_ccb_on_split,
)
from thesis.data import SubjectData
from thesis.protocols import Split


def _make_2b_synthetic(
    *,
    n_per_class_per_session: int = 60,
    n_sessions: int = 2,
    n_samples: int = 1001,
    seed: int = 0,
) -> SubjectData:
    """3-channel bipolar-shaped SubjectData with a 12 Hz lateralized signal.

    Mirrors the fixture pattern in ``tests/test_ccb_arms.py`` but wraps
    everything into a proper SubjectData with session metadata so Phase-2
    protocols (within_subject_cv, session_split) work unchanged.
    """
    rng = np.random.default_rng(seed)
    sfreq = 250.0
    t = np.arange(n_samples) / sfreq

    parts_X: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []
    parts_sess: list[str] = []
    for sess_idx in range(n_sessions):
        X0 = rng.standard_normal((n_per_class_per_session, 3, n_samples)) * 1e-6
        X0[:, 0] += 5e-6 * np.sin(2 * np.pi * 12 * t)
        X1 = rng.standard_normal((n_per_class_per_session, 3, n_samples)) * 1e-6
        X1[:, 2] += 5e-6 * np.sin(2 * np.pi * 12 * t)
        X_sess = np.concatenate([X0, X1], axis=0)
        y_sess = np.array(
            ["left_hand"] * n_per_class_per_session + ["right_hand"] * n_per_class_per_session
        )
        parts_X.append(X_sess)
        parts_y.append(y_sess)
        parts_sess.extend([str(sess_idx)] * len(y_sess))

    X = np.concatenate(parts_X, axis=0)
    y = np.concatenate(parts_y, axis=0)
    metadata = pd.DataFrame({"subject": [1] * len(y), "session": parts_sess, "run": ["0"] * len(y)})
    return SubjectData(
        subject=1,
        dataset_name="BCICIV-2b-screening",
        X=X,
        y=y,
        metadata=metadata,
        sfreq=sfreq,
    )


# ---------------------------------------------------------------------------
# Leakage & safety
# ---------------------------------------------------------------------------


def test_runner_no_leakage_of_2a():
    """The runner module must not import the 2a loader."""
    assert not hasattr(runner_mod, "load_bci2a"), (
        "thesis.ccb.runner must not import load_bci2a (CLAUDE.md §2)"
    )
    # Also check: no `load_bci2a` reference in the source (except possibly
    # in a comment). Strip comments before checking.
    src = inspect.getsource(runner_mod)
    non_comment = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    assert "load_bci2a" not in non_comment, (
        "thesis.ccb.runner source must not reference load_bci2a outside comments"
    )


def test_summarize_reads_only_2a_from_fbcsp_csv():
    """scripts/summarize_ccb.py must explicitly filter FBCSP rows to 2a only.

    This test greps the source rather than loading/running it because running
    requires the 2a CSV to exist. The filter line is referenced by
    _FBCSP_2A_DATASET_TAG and the actual DataFrame filter. Both must be present.
    """
    from pathlib import Path

    src_path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_ccb.py"
    src = src_path.read_text()
    # Non-leakage guard: the dataset-tag constant must be defined and used.
    assert '_FBCSP_2A_DATASET_TAG = "BCI-IV-2a"' in src, (
        "summarize_ccb.py must define _FBCSP_2A_DATASET_TAG = 'BCI-IV-2a'"
    )
    assert 'df["dataset"] == _FBCSP_2A_DATASET_TAG' in src, (
        "summarize_ccb.py must filter the FBCSP CSV by dataset == BCI-IV-2a before reading κ values"
    )
    # And the runner never imports it either (belt and suspenders).
    assert "summarize_ccb" not in inspect.getsource(runner_mod), (
        "runner must not depend on summarize_ccb"
    )


def test_runner_rejects_2a_subject_data():
    """A SubjectData tagged as 2a must be rejected up front."""
    data = _make_2b_synthetic()
    data_2a = SubjectData(
        subject=data.subject,
        dataset_name="BCICIV-2a",  # pretend it's 2a
        X=data.X,
        y=data.y,
        metadata=data.metadata,
        sfreq=data.sfreq,
    )
    split = Split(train_idx=np.arange(60), test_idx=np.arange(60, 120), name="dummy")
    with pytest.raises(RuntimeError, match="2a"):
        run_ccb_on_split(data_2a, split)


# ---------------------------------------------------------------------------
# Calibration / stream split
# ---------------------------------------------------------------------------


def test_calibration_stream_split_is_disjoint_and_covers_train():
    train = np.arange(100)
    cal, stream = _calibration_stream_split(train, calibration_frac=0.3, seed=1)
    assert len(set(cal.tolist()) & set(stream.tolist())) == 0
    assert set(cal.tolist()) | set(stream.tolist()) == set(train.tolist())
    assert len(cal) == 30 and len(stream) == 70


def test_calibration_stream_split_minimum_calibration():
    # Even tiny train sets get ≥ 2 calibration trials.
    cal, stream = _calibration_stream_split(np.arange(5), calibration_frac=0.01, seed=1)
    assert len(cal) >= 2


# ---------------------------------------------------------------------------
# Arm-head fitting + pruning
# ---------------------------------------------------------------------------


def test_fit_heads_on_calibration_returns_disjoint_stream():
    data = _make_2b_synthetic()
    # Use a small arm set so the test is fast.
    arms = [
        Arm(
            arm_id=0,
            band=(8.0, 12.0),
            spatial="csp",
            feature="logvar",
            window=(0.0, 4.0),
            cost=2.0,
            n_components=2,
        ),
        Arm(
            arm_id=1,
            band=(16.0, 24.0),
            spatial="identity",
            feature="logvar",
            window=(0.0, 4.0),
            cost=3.0,
            n_components=3,
        ),
    ]
    train_idx = np.arange(len(data.y) // 2)
    surviving, heads, stream_idx = fit_heads_on_calibration(data, train_idx, arms, seed=0)
    assert len(surviving) <= len(arms)
    for arm in surviving:
        assert arm.arm_id in heads
    # Stream ∩ calibration should be empty; calibration = train_idx − stream_idx.
    cal_set = set(train_idx.tolist()) - set(stream_idx.tolist())
    assert len(cal_set & set(stream_idx.tolist())) == 0


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------


@pytest.fixture
def subset_2b_arms():
    """A small, fast arm set that still exercises all spatial families."""
    return [
        Arm(
            arm_id=0,
            band=(8.0, 12.0),
            spatial="csp",
            feature="logvar",
            window=(0.0, 4.0),
            cost=2.0,
            n_components=2,
        ),
        Arm(
            arm_id=1,
            band=(8.0, 12.0),
            spatial="laplacian",
            feature="logvar",
            window=(0.0, 4.0),
            cost=2.0,
            n_components=3,
        ),
        Arm(
            arm_id=2,
            band=(16.0, 24.0),
            spatial="identity",
            feature="logvar",
            window=(0.0, 4.0),
            cost=3.0,
            n_components=3,
        ),
        Arm(
            arm_id=3,
            band=(36.0, 40.0),
            spatial="identity",
            feature="logvar",
            window=(0.0, 4.0),
            cost=3.0,
            n_components=3,
        ),
    ]


def test_run_ccb_on_split_produces_valid_result(subset_2b_arms):
    data = _make_2b_synthetic()
    # Use a single session's trials as train, the other as test.
    train_idx = np.where(data.metadata["session"].to_numpy() == "0")[0]
    test_idx = np.where(data.metadata["session"].to_numpy() == "1")[0]
    split = Split(train_idx=train_idx, test_idx=test_idx, name="test_session_split")

    result = run_ccb_on_split(data, split, arms=subset_2b_arms, seed=0)
    assert isinstance(result, CCBResult)
    assert -1.0 <= result.kappa <= 1.0
    assert 0.0 <= result.accuracy <= 1.0
    assert result.n_test == len(test_idx)
    # Cumulative regret is non-decreasing (rewards ≥ 0 always).
    if result.cumulative_regret.size >= 2:
        diffs = np.diff(result.cumulative_regret)
        assert np.all(diffs >= -1e-9), f"regret must be non-decreasing; diffs={diffs}"
    # Arm pulls are valid ids in the pool.
    valid_ids = {arm.arm_id for arm in subset_2b_arms}
    for pulled in result.arm_pulls:
        assert int(pulled) in valid_ids


def test_run_ccb_beats_chance_on_synthetic(subset_2b_arms):
    """On the 12 Hz-lateralized synthetic fixture, CCB should land κ > 0.3."""
    data = _make_2b_synthetic()
    train_idx = np.where(data.metadata["session"].to_numpy() == "0")[0]
    test_idx = np.where(data.metadata["session"].to_numpy() == "1")[0]
    split = Split(train_idx=train_idx, test_idx=test_idx, name="synth_session_split")
    result = run_ccb_on_split(data, split, arms=subset_2b_arms, seed=0)
    assert result.kappa > 0.3, f"expected κ > 0.3 on synthetic, got {result.kappa}"


def test_run_ccb_uses_default_arms_when_none_given():
    """Default behavior enumerates the full 2b arm bank."""
    data = _make_2b_synthetic(n_per_class_per_session=40)
    train_idx = np.where(data.metadata["session"].to_numpy() == "0")[0]
    test_idx = np.where(data.metadata["session"].to_numpy() == "1")[0]
    split = Split(train_idx=train_idx, test_idx=test_idx, name="default_arms")
    # arms=None triggers enumerate_arms_2b().
    result = run_ccb_on_split(data, split, seed=0)
    # Full pool (162) pruned to ≤ 100 by the hard cap.
    assert result.n_arms_surviving <= 100
    assert result.n_arms_surviving >= 1


def test_enumerate_arms_2b_integrates_with_runner():
    """Sanity: the default arm bank runs end-to-end without errors."""
    data = _make_2b_synthetic(n_per_class_per_session=30)
    arms = enumerate_arms_2b(data.sfreq)
    assert len(arms) == 162
    train_idx = np.where(data.metadata["session"].to_numpy() == "0")[0]
    test_idx = np.where(data.metadata["session"].to_numpy() == "1")[0]
    split = Split(train_idx=train_idx, test_idx=test_idx, name="full_pool")
    result = run_ccb_on_split(data, split, arms=arms, seed=0)
    assert isinstance(result, CCBResult)


# ---------------------------------------------------------------------------
# Phase-4: context ablation and per-round cap
# ---------------------------------------------------------------------------


def test_runner_include_recent_rewards_false_uses_d15(subset_2b_arms):
    """With the recent-reward tail disabled, OPLB operates on d_psi = 15 + n_arms."""
    from thesis.ccb import runner as runner_mod
    from thesis.ccb.oplb import OPLB

    data = _make_2b_synthetic()
    train_idx = np.where(data.metadata["session"].to_numpy() == "0")[0]
    test_idx = np.where(data.metadata["session"].to_numpy() == "1")[0]
    split = Split(train_idx=train_idx, test_idx=test_idx, name="ctx_ablation")

    captured: dict[str, int] = {}
    orig_init = OPLB.__init__

    def spy_init(self, d_psi, n_arms, config):  # type: ignore[no-redef]
        captured["d_psi"] = d_psi
        captured["n_arms"] = n_arms
        return orig_init(self, d_psi, n_arms, config)

    runner_mod.OPLB.__init__ = spy_init  # type: ignore[method-assign]
    try:
        result = run_ccb_on_split(
            data,
            split,
            arms=subset_2b_arms,
            seed=0,
            include_recent_rewards=False,
        )
    finally:
        runner_mod.OPLB.__init__ = orig_init  # type: ignore[method-assign]

    assert captured["d_psi"] == 15 + captured["n_arms"], (
        f"d_psi should be 15 + n_arms when include_recent_rewards=False; "
        f"got d_psi={captured['d_psi']}, n_arms={captured['n_arms']}"
    )
    assert isinstance(result, CCBResult)


def test_runner_include_recent_rewards_true_default_uses_d18(subset_2b_arms):
    """Default (backward-compat) still uses d_psi = 18 + n_arms."""
    from thesis.ccb import runner as runner_mod
    from thesis.ccb.oplb import OPLB

    data = _make_2b_synthetic()
    train_idx = np.where(data.metadata["session"].to_numpy() == "0")[0]
    test_idx = np.where(data.metadata["session"].to_numpy() == "1")[0]
    split = Split(train_idx=train_idx, test_idx=test_idx, name="ctx_default")

    captured: dict[str, int] = {}
    orig_init = OPLB.__init__

    def spy_init(self, d_psi, n_arms, config):  # type: ignore[no-redef]
        captured["d_psi"] = d_psi
        captured["n_arms"] = n_arms
        return orig_init(self, d_psi, n_arms, config)

    runner_mod.OPLB.__init__ = spy_init  # type: ignore[method-assign]
    try:
        run_ccb_on_split(data, split, arms=subset_2b_arms, seed=0)
    finally:
        runner_mod.OPLB.__init__ = orig_init  # type: ignore[method-assign]

    assert captured["d_psi"] == 18 + captured["n_arms"]


def test_runner_preserves_nonstationary_config_fields(subset_2b_arms):
    """Regression: the runner's budget-default injection MUST keep Phase-5
    ``window_size`` / ``discount_gamma`` fields on the OPLBConfig it hands
    to the policy. Previously the rebuild used a keyword-by-keyword copy
    that dropped new fields silently — 0/N arm-pulls differed between
    stationary and window=50 configs on real data."""
    from thesis.ccb import runner as runner_mod
    from thesis.ccb.oplb import OPLB, OPLBConfig

    data = _make_2b_synthetic()
    train_idx = np.where(data.metadata["session"].to_numpy() == "0")[0]
    test_idx = np.where(data.metadata["session"].to_numpy() == "1")[0]
    split = Split(train_idx=train_idx, test_idx=test_idx, name="nonstat_regress")

    captured: dict[str, OPLBConfig] = {}
    orig_init = OPLB.__init__

    def spy_init(self, d_psi, n_arms, config):  # type: ignore[no-redef]
        captured["config"] = config
        return orig_init(self, d_psi, n_arms, config)

    runner_mod.OPLB.__init__ = spy_init  # type: ignore[method-assign]
    try:
        caller_cfg = OPLBConfig(alpha=1.0, lambda_reg=1.0, window_size=50, discount_gamma=0.9)
        run_ccb_on_split(data, split, arms=subset_2b_arms, config=caller_cfg, seed=0)
    finally:
        runner_mod.OPLB.__init__ = orig_init  # type: ignore[method-assign]

    c = captured["config"]
    assert c.window_size == 50, f"window_size lost: got {c.window_size}"
    assert c.discount_gamma == 0.9, f"discount_gamma lost: got {c.discount_gamma}"
    # And the budget default DID get injected (was inf on input).
    assert c.budget != float("inf")


def test_runner_online_heads_is_backward_compat_when_false(subset_2b_arms):
    """online_heads=False must reproduce the pre-Phase-5 frozen behaviour."""
    data = _make_2b_synthetic()
    train_idx = np.where(data.metadata["session"].to_numpy() == "0")[0]
    test_idx = np.where(data.metadata["session"].to_numpy() == "1")[0]
    split = Split(train_idx=train_idx, test_idx=test_idx, name="online_false")
    baseline = run_ccb_on_split(data, split, arms=subset_2b_arms, seed=0)
    explicit = run_ccb_on_split(data, split, arms=subset_2b_arms, seed=0, online_heads=False)
    assert baseline.kappa == explicit.kappa
    assert baseline.n_test == explicit.n_test
    np.testing.assert_array_equal(baseline.arm_pulls, explicit.arm_pulls)


def test_runner_online_heads_changes_test_predictions(subset_2b_arms):
    """Enabling online_heads should produce a different CCBResult (different
    arm_pulls or kappa) vs frozen heads — proving partial_fit propagates."""
    data = _make_2b_synthetic()
    train_idx = np.where(data.metadata["session"].to_numpy() == "0")[0]
    test_idx = np.where(data.metadata["session"].to_numpy() == "1")[0]
    split = Split(train_idx=train_idx, test_idx=test_idx, name="online_true")
    frozen = run_ccb_on_split(data, split, arms=subset_2b_arms, seed=0)
    online = run_ccb_on_split(data, split, arms=subset_2b_arms, seed=0, online_heads=True)
    # Any observable downstream change is sufficient: arm pull ids or
    # test-set kappa. Runner + partial_fit must interact.
    changed = frozen.kappa != online.kappa or not np.array_equal(frozen.arm_pulls, online.arm_pulls)
    assert changed, "online_heads=True must measurably change runner output"


def test_runner_accepts_custom_policy_factory(subset_2b_arms):
    """A non-default policy_factory (FixedArmPolicy) runs end-to-end and
    pulls only the fixed arm in the stream phase."""
    from thesis.ccb.policies import FixedArmPolicy

    data = _make_2b_synthetic()
    train_idx = np.where(data.metadata["session"].to_numpy() == "0")[0]
    test_idx = np.where(data.metadata["session"].to_numpy() == "1")[0]
    split = Split(train_idx=train_idx, test_idx=test_idx, name="fixed_arm")

    def fixed_factory(d_psi, n_arms, cfg):  # noqa: ARG001
        return FixedArmPolicy(d_psi=d_psi, n_arms=n_arms, config=cfg, fixed_arm_idx=0)

    result = run_ccb_on_split(
        data,
        split,
        arms=subset_2b_arms,
        seed=0,
        policy_factory=fixed_factory,
    )
    # Stream pulls should all map to arm-index-0 in the surviving list.
    # Because surviving is sorted by calibration κ, arm_pulls contains the
    # arm_id of the top survivor — a single value repeated.
    assert isinstance(result, CCBResult)
    if result.arm_pulls.size > 0:
        assert len(set(result.arm_pulls.tolist())) == 1, (
            "FixedArmPolicy must pull exactly one arm_id during the stream"
        )


def test_runner_binding_per_round_cap_never_pulls_expensive_arms(subset_2b_arms):
    """With per_round_cap=2, only arms with cost ≤ 2 may ever be pulled."""
    from thesis.ccb.oplb import OPLBConfig

    # subset_2b_arms has costs {2, 2, 3, 3}: cap=2 binds the identity arms out.
    cheap_arm_ids = {arm.arm_id for arm in subset_2b_arms if arm.cost <= 2.0}
    assert cheap_arm_ids, "fixture should have at least one cost ≤ 2 arm"

    data = _make_2b_synthetic()
    train_idx = np.where(data.metadata["session"].to_numpy() == "0")[0]
    test_idx = np.where(data.metadata["session"].to_numpy() == "1")[0]
    split = Split(train_idx=train_idx, test_idx=test_idx, name="cap_binding")

    config = OPLBConfig(alpha=1.0, lambda_reg=1.0, per_round_cap=2.0)
    result = run_ccb_on_split(data, split, arms=subset_2b_arms, config=config, seed=0)

    pulled = {int(aid) for aid in result.arm_pulls}
    # Only arms that survived pruning AND respect the cap could be pulled.
    # If calibration pruned all cost ≤ 2 arms this test degenerates — the
    # runner would have raised. Otherwise, every pulled arm must be cheap.
    for aid in pulled:
        matched = [arm for arm in subset_2b_arms if arm.arm_id == aid]
        assert matched, f"pulled arm_id {aid} not in fixture"
        assert matched[0].cost <= 2.0, (
            f"per_round_cap=2.0 violated: arm {aid} has cost {matched[0].cost}"
        )


def _make_cl_synthetic(
    *,
    dataset_name: str = "STEW",
    n_channels: int = 14,
    n_per_class_per_session: int = 30,
    n_sessions: int = 2,
    n_samples: int = 1000,
    seed: int = 0,
) -> SubjectData:
    """Synthetic CL-shaped SubjectData with a workload-style signal.

    Generates ``n_channels``-channel epochs with a class-dependent
    band-dominant signal — just enough structure that the workload
    context (which sums θ/α/β log-powers) sees a real signal. Both
    classes appear in every session so within_subject_cv and the CCB
    runner's calibration / stream split both see balanced data.
    """
    rng = np.random.default_rng(seed)
    sfreq = 250.0
    t = np.arange(n_samples) / sfreq

    parts_X: list[np.ndarray] = []
    parts_y: list[str] = []
    parts_sess: list[str] = []
    for sess_idx in range(n_sessions):
        X_low = rng.standard_normal((n_per_class_per_session, n_channels, n_samples)) * 1e-6
        X_low += 2e-6 * np.sin(2 * np.pi * 10 * t)  # α-dominant (low workload)
        X_high = rng.standard_normal((n_per_class_per_session, n_channels, n_samples)) * 1e-6
        X_high += 2e-6 * np.sin(2 * np.pi * 20 * t)  # β-dominant (high workload)
        parts_X += [X_low, X_high]
        parts_y += ["low"] * n_per_class_per_session + ["high"] * n_per_class_per_session
        parts_sess += [str(sess_idx)] * (2 * n_per_class_per_session)

    X = np.concatenate(parts_X, axis=0)
    y = np.array(parts_y)
    meta = pd.DataFrame({
        "subject": [1] * len(y),
        "session": parts_sess,
        "run": ["0"] * len(y),
    })
    perm = rng.permutation(len(y))
    return SubjectData(
        subject=1,
        dataset_name=dataset_name,
        X=X[perm],
        y=y[perm],
        metadata=meta.iloc[perm].reset_index(drop=True),
        sfreq=sfreq,
    )


class TestWorkloadContextDispatch:
    """Phase E regression coverage for the CL-context dispatch in run_ccb_on_split."""

    def test_stew_with_workload_roles_uses_cl_context(self):
        """STEW dataset_name + workload_channel_roles → CL context path runs end-to-end."""
        from thesis.ccb.arms import enumerate_arms_generic
        from thesis.ccb.oplb import OPLBConfig

        data = _make_cl_synthetic(dataset_name="STEW", n_channels=14)
        train_idx = np.where(data.metadata["session"].to_numpy() == "0")[0]
        test_idx = np.where(data.metadata["session"].to_numpy() == "1")[0]
        split = Split(train_idx=train_idx, test_idx=test_idx, name="cl_dispatch")

        roles = {"frontal": [0, 1, 2, 3], "parietal": [10, 11], "f3": [2], "f4": [11]}
        arms = enumerate_arms_generic(n_channels=14, n_components=4)[:6]

        result = run_ccb_on_split(
            data,
            split,
            arms=arms,
            config=OPLBConfig(alpha=0.5, lambda_reg=1.0),
            calibration_frac=0.3,
            seed=0,
            include_recent_rewards=False,
            workload_channel_roles=roles,
        )
        assert isinstance(result, CCBResult)
        assert result.n_test > 0

    def test_wauc_with_workload_roles_uses_cl_context(self):
        """WAUC dataset_name + workload_channel_roles → CL context path runs end-to-end."""
        from thesis.ccb.arms import enumerate_arms_generic
        from thesis.ccb.oplb import OPLBConfig

        data = _make_cl_synthetic(dataset_name="WAUC", n_channels=8)
        train_idx = np.where(data.metadata["session"].to_numpy() == "0")[0]
        test_idx = np.where(data.metadata["session"].to_numpy() == "1")[0]
        split = Split(train_idx=train_idx, test_idx=test_idx, name="cl_dispatch")

        roles = {"frontal": [0, 1, 2, 3], "parietal": [6, 7], "f3": [3], "f4": [0]}
        arms = enumerate_arms_generic(n_channels=8, n_components=4)[:6]

        result = run_ccb_on_split(
            data,
            split,
            arms=arms,
            config=OPLBConfig(alpha=0.5, lambda_reg=1.0),
            calibration_frac=0.3,
            seed=0,
            include_recent_rewards=False,
            workload_channel_roles=roles,
        )
        assert isinstance(result, CCBResult)
        assert result.n_test > 0

    def test_cl_without_roles_falls_through_to_generic(self):
        """STEW dataset_name but no roles → generic n-channel MI context (back-compat)."""
        from thesis.ccb.arms import enumerate_arms_generic
        from thesis.ccb.oplb import OPLBConfig

        data = _make_cl_synthetic(dataset_name="STEW", n_channels=14)
        train_idx = np.where(data.metadata["session"].to_numpy() == "0")[0]
        test_idx = np.where(data.metadata["session"].to_numpy() == "1")[0]
        split = Split(train_idx=train_idx, test_idx=test_idx, name="cl_dispatch")

        arms = enumerate_arms_generic(n_channels=14, n_components=4)[:6]
        result = run_ccb_on_split(
            data,
            split,
            arms=arms,
            config=OPLBConfig(alpha=0.5, lambda_reg=1.0),
            calibration_frac=0.3,
            seed=0,
            include_recent_rewards=False,
        )
        assert isinstance(result, CCBResult)
        assert result.n_test > 0
