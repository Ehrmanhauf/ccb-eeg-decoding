"""Unit tests for thesis.ccb.policies — the §8.4 ablations.

Offline, synthetic-only; no EEG files touched.
"""

from __future__ import annotations

import numpy as np
import pytest

from thesis.ccb.oplb import INFEASIBLE, OPLB, OPLBConfig
from thesis.ccb.policies import (
    EpsilonGreedyPolicy,
    FixedArmPolicy,
    LinTSPolicy,
    make_unconstrained_linucb,
)

# ---------------------------------------------------------------------------
# FixedArmPolicy
# ---------------------------------------------------------------------------


def test_fixed_arm_always_pulls_same_arm():
    """select() returns the configured arm index every call when feasible."""
    n_arms = 5
    policy = FixedArmPolicy(
        d_psi=10,
        n_arms=n_arms,
        config=OPLBConfig(budget=1000.0),
        fixed_arm_idx=2,
    )
    arm_costs = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
    contexts = np.zeros((n_arms, 10))
    for _ in range(50):
        a = policy.select(contexts=contexts, arm_costs=arm_costs)
        assert a == 2
        policy.update(a, contexts[a], reward=1.0, realized_cost=float(arm_costs[a]))


def test_fixed_arm_respects_per_round_cap():
    """A fixed arm whose cost exceeds the per-round cap returns INFEASIBLE."""
    policy = FixedArmPolicy(
        d_psi=10,
        n_arms=3,
        config=OPLBConfig(budget=1000.0, per_round_cap=2.0),
        fixed_arm_idx=1,  # cost = 3.0 (above cap)
    )
    arm_costs = np.array([1.0, 3.0, 2.0])
    contexts = np.zeros((3, 10))
    assert policy.select(contexts=contexts, arm_costs=arm_costs) == INFEASIBLE


def test_fixed_arm_exhausts_budget():
    """Once budget drains below the fixed arm's cost, select() returns INFEASIBLE."""
    policy = FixedArmPolicy(
        d_psi=4,
        n_arms=2,
        config=OPLBConfig(budget=5.0),
        fixed_arm_idx=0,
    )
    arm_costs = np.array([2.0, 1.0])
    contexts = np.zeros((2, 4))
    pulls = 0
    for _ in range(10):
        a = policy.select(contexts=contexts, arm_costs=arm_costs)
        if a == INFEASIBLE:
            break
        policy.update(a, contexts[a], reward=1.0, realized_cost=float(arm_costs[a]))
        pulls += 1
    assert pulls in (2, 3), f"expected 2 or 3 pulls on budget 5 / cost 2, got {pulls}"
    assert policy.select(contexts=contexts, arm_costs=arm_costs) == INFEASIBLE


def test_fixed_arm_rejects_bad_idx():
    with pytest.raises(ValueError, match="out of range"):
        FixedArmPolicy(
            d_psi=4,
            n_arms=3,
            config=OPLBConfig(),
            fixed_arm_idx=7,
        )


# ---------------------------------------------------------------------------
# EpsilonGreedyPolicy
# ---------------------------------------------------------------------------


def test_epsilon_greedy_exploration_fraction_converges_to_epsilon():
    """With all-equal reward, the random-branch fraction should ≈ ε over many rounds."""
    n_arms = 4
    d_psi = 8
    epsilon = 0.2
    policy = EpsilonGreedyPolicy(
        d_psi=d_psi,
        n_arms=n_arms,
        config=OPLBConfig(budget=float("inf")),
        epsilon=epsilon,
        rng_seed=123,
    )
    contexts = np.zeros((n_arms, d_psi))
    arm_costs = np.ones(n_arms)
    # Inject a clearly best arm via θ̂ updates so greedy would always pick arm 0.
    for _ in range(10):
        psi = contexts[0].copy()
        psi[0] = 1.0  # inject a bias so arm 0 gets greedy-picked
        policy.update(0, psi, reward=1.0, realized_cost=1.0)

    # Build contexts so arm 0 has a clearly higher ⟨ψ, θ̂⟩ than the rest.
    contexts = np.zeros((n_arms, d_psi))
    contexts[0, 0] = 1.0  # only arm 0 has that feature

    # Count how often select returns a non-greedy choice.
    n_trials = 2000
    non_greedy = 0
    for _ in range(n_trials):
        a = policy.select(contexts=contexts, arm_costs=arm_costs)
        if a != 0:
            non_greedy += 1
    frac = non_greedy / n_trials
    # Random ε-branch selects any of n_arms uniformly; only (n_arms-1)/n_arms
    # of those produce a non-greedy pick. Expected frac = ε · (n_arms-1)/n_arms.
    expected = epsilon * (n_arms - 1) / n_arms
    assert abs(frac - expected) < 0.05, (
        f"non-greedy fraction {frac:.3f} deviates from expected {expected:.3f}"
    )


def test_epsilon_greedy_epsilon_zero_is_pure_greedy():
    """ε=0 → never diverges from greedy argmax on non-degenerate input."""
    n_arms = 3
    d_psi = 4
    policy = EpsilonGreedyPolicy(
        d_psi=d_psi,
        n_arms=n_arms,
        config=OPLBConfig(budget=float("inf")),
        epsilon=0.0,
        rng_seed=0,
    )
    # Make arm 1 dominate via updates.
    psi_arm1 = np.array([0.0, 1.0, 0.0, 0.0])
    for _ in range(20):
        policy.update(1, psi_arm1, reward=1.0, realized_cost=1.0)

    contexts = np.stack(
        [
            np.array([0.5, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.5, 0.0]),
        ]
    )
    arm_costs = np.ones(n_arms)
    for _ in range(20):
        a = policy.select(contexts=contexts, arm_costs=arm_costs)
        assert a == 1, f"ε=0 should always pick greedy best (arm 1); got {a}"


def test_epsilon_greedy_rejects_out_of_range():
    with pytest.raises(ValueError, match="epsilon"):
        EpsilonGreedyPolicy(d_psi=4, n_arms=3, config=OPLBConfig(), epsilon=1.5)


def test_epsilon_greedy_respects_budget_and_cap():
    """Feasibility filter applies to both random and greedy branches."""
    policy = EpsilonGreedyPolicy(
        d_psi=4,
        n_arms=3,
        config=OPLBConfig(budget=float("inf"), per_round_cap=1.0),
        epsilon=1.0,  # always random branch
        rng_seed=0,
    )
    arm_costs = np.array([0.5, 2.0, 0.8])  # arms 1 exceeds cap
    contexts = np.zeros((3, 4))
    for _ in range(100):
        a = policy.select(contexts=contexts, arm_costs=arm_costs)
        assert a in (0, 2), f"ε=1 random branch should never pick over-cap arm 1; got {a}"


# ---------------------------------------------------------------------------
# Unconstrained LinUCB
# ---------------------------------------------------------------------------


def test_unconstrained_linucb_never_infeasible():
    """On non-degenerate inputs, unconstrained LinUCB always selects an arm."""
    config = OPLBConfig(budget=5.0, per_round_cap=1.0)  # input has constraints
    policy = make_unconstrained_linucb(d_psi=4, n_arms=3, config=config)
    assert isinstance(policy, OPLB)
    assert policy.config.budget == float("inf")
    assert policy.config.per_round_cap is None

    arm_costs = np.array([10.0, 20.0, 30.0])  # all above the ORIGINAL per-round cap
    contexts = np.zeros((3, 4))
    for _ in range(20):
        a = policy.select(contexts=contexts, arm_costs=arm_costs)
        assert a != INFEASIBLE, "unconstrained policy should never be infeasible here"
        policy.update(a, contexts[a], reward=1.0, realized_cost=float(arm_costs[a]))
    # Budget remains +inf after arbitrarily many expensive pulls.
    assert policy.budget_remaining == float("inf")


def test_unconstrained_preserves_alpha_and_lambda():
    """Hyperparameters other than budget/cap should pass through unchanged."""
    config = OPLBConfig(alpha=2.5, lambda_reg=3.0, delta=0.01, budget=100.0, per_round_cap=1.0)
    policy = make_unconstrained_linucb(d_psi=4, n_arms=3, config=config)
    assert policy.config.alpha == 2.5
    assert policy.config.lambda_reg == 3.0
    assert policy.config.delta == 0.01


# ---------------------------------------------------------------------------
# LinTSPolicy (Thompson Sampling)
# ---------------------------------------------------------------------------


def test_lints_sample_covariance_matches_prior():
    """Monte-Carlo: sampled θ's empirical cov should be ≈ v² · A⁻¹ (prior Cov)."""
    d, n = 6, 4
    policy = LinTSPolicy(
        d_psi=d,
        n_arms=n,
        config=OPLBConfig(lambda_reg=2.0, budget=float("inf")),
        prior_scale=1.0,
        rng_seed=42,
    )
    # Before any update: A = λI, θ̂ = 0 → samples ~ N(0, v²/λ · I)
    samples = np.stack([policy._sample_theta() for _ in range(5000)])
    emp_cov = np.cov(samples, rowvar=False)
    expected = (policy.prior_scale**2 / 2.0) * np.eye(d)  # λ=2.0
    # 5000 samples, d=6: ~3-5% error expected.
    assert np.allclose(emp_cov, expected, atol=0.10), (
        f"empirical cov deviates from v²·A⁻¹; max |err| = {np.max(np.abs(emp_cov - expected)):.3f}"
    )


def test_lints_select_respects_budget_and_cap():
    """TS feasibility filter matches OPLB's: expensive arms never pulled under tight cap."""
    policy = LinTSPolicy(
        d_psi=4,
        n_arms=3,
        config=OPLBConfig(budget=float("inf"), per_round_cap=1.0),
        rng_seed=0,
    )
    arm_costs = np.array([0.5, 2.0, 0.8])  # arm 1 over cap
    contexts = np.random.default_rng(0).standard_normal((3, 4))
    for _ in range(100):
        a = policy.select(contexts=contexts, arm_costs=arm_costs)
        assert a in (0, 2), f"per_round_cap violated: picked arm {a} with cost 2.0"


def test_lints_converges_toward_best_arm_on_synthetic():
    """With updates, TS should concentrate mass on the truly-best arm."""
    d, n_arms = 4, 3
    true_theta = np.array([1.0, 0.0, 0.0, 0.0])
    contexts = np.stack(
        [
            np.array([1.0, 0.0, 0.0, 0.0]),  # arm 0: high expected reward
            np.array([0.2, 0.0, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0, 0.0]),
        ]
    )
    arm_costs = np.ones(n_arms)
    policy = LinTSPolicy(
        d_psi=d,
        n_arms=n_arms,
        config=OPLBConfig(lambda_reg=1.0, budget=float("inf")),
        prior_scale=1.0,
        rng_seed=7,
    )
    rng = np.random.default_rng(7)
    for _ in range(200):
        a = policy.select(contexts=contexts, arm_costs=arm_costs)
        assert a != INFEASIBLE
        mean_r = float(contexts[a] @ true_theta)
        r = mean_r + 0.1 * rng.standard_normal()
        policy.update(a, contexts[a], reward=r, realized_cost=1.0)

    # After 200 pulls, the majority in the tail should be arm 0 (best).
    tail_picks = [policy.select(contexts=contexts, arm_costs=arm_costs) for _ in range(500)]
    best_frac = sum(1 for a in tail_picks if a == 0) / len(tail_picks)
    assert best_frac > 0.7, (
        f"expected TS to concentrate > 70% on arm 0 after learning; got {best_frac:.2f}"
    )


def test_lints_rejects_non_positive_prior_scale():
    with pytest.raises(ValueError, match="prior_scale"):
        LinTSPolicy(d_psi=4, n_arms=3, config=OPLBConfig(), prior_scale=0.0)
    with pytest.raises(ValueError, match="prior_scale"):
        LinTSPolicy(d_psi=4, n_arms=3, config=OPLBConfig(), prior_scale=-1.0)


def test_lints_regret_within_4x_oplb_on_synthetic():
    """Soft check: TS regret should not be catastrophically worse than OPLB.

    Agrawal & Goyal 2013 Thm 1 proves TS regret is ``Õ(d^{3/2} √T)`` while
    OPLB/LinUCB is ``Õ(d √T)``. On a clean synthetic with d=5 the worst-case
    ratio is ~√d ≈ 2.2×; 4× is a loose safety threshold to catch egregious
    sign flips / Cholesky direction bugs while allowing the expected gap.
    """
    d, n_arms, T = 5, 8, 300
    rng_true = np.random.default_rng(0)
    true_theta = rng_true.standard_normal(d)
    arm_costs = np.ones(n_arms)

    def run(policy, contexts_per_round, noise_rng):
        regret = 0.0
        for t in range(T):
            ctx = contexts_per_round[t]
            mean_rs = ctx @ true_theta
            best = mean_rs.max()
            a = policy.select(contexts=ctx, arm_costs=arm_costs)
            if a == INFEASIBLE:
                break
            r = mean_rs[a] + 0.1 * noise_rng.standard_normal()
            policy.update(a, ctx[a], reward=r, realized_cost=1.0)
            regret += best - mean_rs[a]
        return regret

    rng_ctx = np.random.default_rng(1)
    contexts = [rng_ctx.standard_normal((n_arms, d)) for _ in range(T)]

    oplb = OPLB(d_psi=d, n_arms=n_arms, config=OPLBConfig(alpha=1.0, budget=float("inf")))
    ts = LinTSPolicy(d_psi=d, n_arms=n_arms, config=OPLBConfig(budget=float("inf")), rng_seed=11)
    r_oplb = run(oplb, contexts, np.random.default_rng(100))
    r_ts = run(ts, contexts, np.random.default_rng(100))
    assert r_ts < 4.0 * max(r_oplb, 1.0), (
        f"TS regret {r_ts:.1f} unreasonably larger than OPLB regret {r_oplb:.1f}"
    )


def test_lints_differs_from_epsilon_greedy():
    """LinTSPolicy should not share ε-greedy's random-branch structure.

    With contexts that strongly distinguish arm 0, ε-greedy (ε=1.0) picks
    uniformly, while TS (after training) concentrates on arm 0. This test
    just ensures the two policies have different select-distributions under
    the same inputs post-training.
    """
    d, n_arms = 4, 3
    contexts = np.stack(
        [
            np.array([2.0, 0.0, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0, 0.0]),
        ]
    )
    arm_costs = np.ones(n_arms)
    true_theta = np.array([1.0, 0.0, 0.0, 0.0])

    # Train TS for 100 rounds.
    ts = LinTSPolicy(d_psi=d, n_arms=n_arms, config=OPLBConfig(budget=float("inf")), rng_seed=0)
    rng_noise = np.random.default_rng(0)
    for _ in range(100):
        a = ts.select(contexts=contexts, arm_costs=arm_costs)
        r = float(contexts[a] @ true_theta) + 0.05 * rng_noise.standard_normal()
        ts.update(a, contexts[a], reward=r, realized_cost=1.0)

    ts_picks = [ts.select(contexts=contexts, arm_costs=arm_costs) for _ in range(200)]
    eps_picks = []
    eps_policy = EpsilonGreedyPolicy(
        d_psi=d, n_arms=n_arms, config=OPLBConfig(budget=float("inf")), epsilon=1.0, rng_seed=0
    )
    for _ in range(200):
        eps_picks.append(eps_policy.select(contexts=contexts, arm_costs=arm_costs))

    ts_best = sum(1 for a in ts_picks if a == 0) / 200
    eps_best = sum(1 for a in eps_picks if a == 0) / 200
    # ε=1.0 is pure random → best-arm fraction ≈ 1/3. Trained TS should be > 0.5.
    assert ts_best > 0.5
    assert eps_best < 0.5
