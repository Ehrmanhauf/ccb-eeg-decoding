"""Unit tests for thesis.ccb.oplb.

Pure synthetic linear-bandit environments — no EEG, no file I/O. Keeps
the bandit correctness story independent of the domain for Phase 3.
"""

from __future__ import annotations

import numpy as np
import pytest

from thesis.ccb.oplb import INFEASIBLE, OPLB, OPLBConfig, default_alpha, make_psi

# ---------------------------------------------------------------------------
# Synthetic linear-bandit environment
# ---------------------------------------------------------------------------


def _make_env(*, d: int, n_arms: int, T: int, noise: float, seed: int):
    """Generate a linear-reward environment with a fixed θ*.

    Returns
    -------
    theta_star : (d,)
    gen : callable(t: int) -> (contexts[n_arms, d], expected[n_arms], costs[n_arms])
    """
    rng = np.random.default_rng(seed)
    theta_star = rng.standard_normal(d)
    theta_star /= np.linalg.norm(theta_star)

    # Pre-generate one context bank per round so re-runs under the same
    # seed are fully deterministic.
    bank = rng.standard_normal((T, n_arms, d))
    bank /= np.linalg.norm(bank, axis=-1, keepdims=True)
    noise_draws = rng.standard_normal(T) * noise

    def gen(t: int):
        contexts = bank[t]
        expected = contexts @ theta_star  # (n_arms,)
        costs = np.ones(n_arms)
        return contexts, expected, costs, noise_draws[t]

    return theta_star, gen


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------


def test_default_alpha_monotone_in_t():
    a_10 = default_alpha(d=5, t=10)
    a_1000 = default_alpha(d=5, t=1000)
    assert a_1000 > a_10 > 0


def test_default_alpha_at_t_zero_degenerates_gracefully():
    a0 = default_alpha(d=5, t=0)
    assert np.isfinite(a0) and a0 > 0


def test_oplb_rejects_bad_init():
    with pytest.raises(ValueError, match="d_psi"):
        OPLB(d_psi=0, n_arms=5, config=OPLBConfig())
    with pytest.raises(ValueError, match="n_arms"):
        OPLB(d_psi=3, n_arms=0, config=OPLBConfig())


def test_select_rejects_wrong_shapes():
    oplb = OPLB(d_psi=3, n_arms=4, config=OPLBConfig())
    with pytest.raises(ValueError, match="contexts shape"):
        oplb.select(contexts=np.zeros((3, 3)), arm_costs=np.ones(4))
    with pytest.raises(ValueError, match="arm_costs shape"):
        oplb.select(contexts=np.zeros((4, 3)), arm_costs=np.ones(3))


def test_select_rejects_negative_cost():
    oplb = OPLB(d_psi=3, n_arms=4, config=OPLBConfig())
    with pytest.raises(ValueError, match="non-negative"):
        oplb.select(contexts=np.zeros((4, 3)), arm_costs=np.array([1, 1, -1, 1]))


def test_theta_hat_converges_on_noiseless():
    """With zero noise, θ̂ should approach θ* after a few hundred rounds."""
    d, n_arms, T = 4, 8, 500
    theta_star, gen = _make_env(d=d, n_arms=n_arms, T=T, noise=0.0, seed=1)
    oplb = OPLB(d_psi=d, n_arms=n_arms, config=OPLBConfig(alpha=1.0))
    for t in range(T):
        ctx, expected, costs, _noise = gen(t)
        a = oplb.select(contexts=ctx, arm_costs=costs)
        assert a != INFEASIBLE
        oplb.update(a, ctx[a], reward=float(expected[a]), realized_cost=float(costs[a]))
    assert np.linalg.norm(oplb.theta_hat - theta_star) < 0.1


def test_synthetic_linear_bandit_regret_ratio_decreases():
    """Sublinear-regret proxy: R(T)/T should drop noticeably as T grows."""
    d, n_arms, T = 5, 10, 2000
    theta_star, gen = _make_env(d=d, n_arms=n_arms, T=T, noise=0.1, seed=2)
    oplb = OPLB(d_psi=d, n_arms=n_arms, config=OPLBConfig(alpha=1.0))
    cum = 0.0
    curve = np.zeros(T)
    for t in range(T):
        ctx, expected, costs, noise = gen(t)
        a = oplb.select(contexts=ctx, arm_costs=costs)
        assert a != INFEASIBLE
        r_t = float(expected[a] + noise)
        oplb.update(a, ctx[a], reward=r_t, realized_cost=float(costs[a]))
        cum += float(expected.max() - expected[a])
        curve[t] = cum

    early_rate = curve[T // 10 - 1] / (T // 10)
    late_rate = curve[T - 1] / T
    # Sub-linear regret: per-round regret at T is at most ~80% of the early-phase rate.
    assert late_rate < 0.8 * early_rate, f"early_rate={early_rate:.4f} late_rate={late_rate:.4f}"
    # Sanity: the inferred theta is at least aligned with the truth direction.
    cos_sim = float(oplb.theta_hat @ theta_star) / (np.linalg.norm(oplb.theta_hat) + 1e-12)
    assert cos_sim > 0.5


def test_per_round_cap_filters_infeasible_arms():
    """Arms whose cost exceeds per_round_cap must never be played."""
    d, n_arms, T = 3, 6, 200
    _theta, gen = _make_env(d=d, n_arms=n_arms, T=T, noise=0.0, seed=3)
    config = OPLBConfig(alpha=1.0, per_round_cap=1.0)
    oplb = OPLB(d_psi=d, n_arms=n_arms, config=config)
    # Half the arms are cheap (cost 1), half are expensive (cost 2).
    cost_override = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    pulls: list[int] = []
    for t in range(T):
        ctx, _expected, _costs, _noise = gen(t)
        a = oplb.select(contexts=ctx, arm_costs=cost_override)
        assert a != INFEASIBLE
        pulls.append(a)
        oplb.update(a, ctx[a], reward=0.0, realized_cost=float(cost_override[a]))
    # No arm ≥ index 3 was ever pulled.
    assert max(pulls) < 3


def test_budget_exhaustion_returns_infeasible():
    """Once the knapsack is empty, select returns INFEASIBLE."""
    d, n_arms = 2, 3
    config = OPLBConfig(alpha=1.0, budget=5.0)
    oplb = OPLB(d_psi=d, n_arms=n_arms, config=config)
    ctx = np.eye(n_arms, d)[:, :d] if d <= n_arms else np.eye(n_arms, d)
    costs = np.array([2.0, 2.0, 2.0])
    # Two pulls consume 4 of 5; third pull would overshoot.
    # After each pull budget shrinks by the cost.
    oplb.update(0, ctx[0], reward=0.0, realized_cost=2.0)  # budget → 3
    oplb.update(1, ctx[1], reward=0.0, realized_cost=2.0)  # budget → 1
    # Now all three arms (cost 2) are infeasible because remaining is 1.
    a = oplb.select(contexts=ctx, arm_costs=costs)
    assert a == INFEASIBLE


def test_reproducibility_under_fixed_seed():
    """Two runs with identical inputs produce identical arm sequences."""
    d, n_arms, T = 4, 5, 300
    _theta, gen = _make_env(d=d, n_arms=n_arms, T=T, noise=0.1, seed=7)

    def run():
        oplb = OPLB(d_psi=d, n_arms=n_arms, config=OPLBConfig(alpha=1.0))
        seq = []
        for t in range(T):
            ctx, expected, costs, noise = gen(t)
            a = oplb.select(contexts=ctx, arm_costs=costs)
            seq.append(a)
            oplb.update(a, ctx[a], reward=float(expected[a] + noise), realized_cost=float(costs[a]))
        return seq

    assert run() == run()


def test_alpha_zero_plays_pure_exploitation():
    """α=0 at select time should pick the argmax predicted reward."""
    d, n_arms = 3, 4
    oplb = OPLB(d_psi=d, n_arms=n_arms, config=OPLBConfig(alpha=1.0))
    # Inject a known θ̂ ≈ [1, 0, 0] via a few directed updates, then test.
    psi = np.array([1.0, 0.0, 0.0])
    for _ in range(20):
        oplb.update(0, psi, reward=1.0, realized_cost=0.0)
    # Now θ̂ should lean heavily toward [1, 0, 0]-ish.
    ctx = np.array([[0.1, 0, 0], [0.9, 0, 0], [0.5, 0, 0], [0.3, 0, 0]])
    a = oplb.select(contexts=ctx, arm_costs=np.ones(n_arms), alpha=0.0)
    assert a == 1  # highest ctx[:, 0] wins under pure exploitation


def test_make_psi_concatenates_and_shape():
    ctx = np.array([1.0, 2.0, 3.0])
    onehot = np.array([0.0, 1.0, 0.0])
    psi = make_psi(ctx, onehot)
    assert psi.shape == (6,)
    np.testing.assert_array_equal(psi, np.array([1, 2, 3, 0, 1, 0]))


# ---------------------------------------------------------------------------
# Phase-5 Stage-1.3: non-stationary OPLB
# ---------------------------------------------------------------------------


def test_window_size_forgets_old_rounds():
    """With window_size=3, A after 5 updates should equal A from just the last 3."""
    d = 4
    cfg = OPLBConfig(lambda_reg=1.0, budget=float("inf"), window_size=3)
    oplb = OPLB(d_psi=d, n_arms=2, config=cfg)

    rng = np.random.default_rng(0)
    psis = [rng.standard_normal(d) for _ in range(5)]
    for psi in psis:
        oplb.update(0, psi, reward=0.5, realized_cost=0.0)

    # A should equal λI + sum_{last 3 ψψᵀ}.
    expected = 1.0 * np.eye(d) + sum(np.outer(p, p) for p in psis[-3:])
    np.testing.assert_allclose(oplb._A, expected, atol=1e-12)


def test_window_size_invalid_raises():
    with pytest.raises(ValueError, match="window_size"):
        OPLB(d_psi=4, n_arms=2, config=OPLBConfig(window_size=0))
    with pytest.raises(ValueError, match="window_size"):
        OPLB(d_psi=4, n_arms=2, config=OPLBConfig(window_size=-5))


def test_discount_gamma_exponential_weighting():
    """γ=0.5: after 2 updates, the first update's contribution should be γ²·ψ₁ψ₁ᵀ."""
    d = 3
    cfg = OPLBConfig(lambda_reg=1.0, budget=float("inf"), discount_gamma=0.5)
    oplb = OPLB(d_psi=d, n_arms=2, config=cfg)

    psi1 = np.array([1.0, 0.0, 0.0])
    psi2 = np.array([0.0, 1.0, 0.0])
    oplb.update(0, psi1, reward=1.0, realized_cost=0.0)
    oplb.update(0, psi2, reward=2.0, realized_cost=0.0)

    # After the two updates with γ=0.5:
    # Step 1: A ← 0.5·(λI − λI) + λI + ψ1ψ1ᵀ = λI + ψ1ψ1ᵀ,  b ← 0.5·0 + ψ1 = ψ1
    # Step 2: A ← 0.5·(λI + ψ1ψ1ᵀ − λI) + λI + ψ2ψ2ᵀ = 0.5·ψ1ψ1ᵀ + λI + ψ2ψ2ᵀ
    #         b ← 0.5·ψ1 + 2·ψ2
    expected_A = 1.0 * np.eye(d) + 0.5 * np.outer(psi1, psi1) + np.outer(psi2, psi2)
    expected_b = 0.5 * psi1 + 2.0 * psi2
    np.testing.assert_allclose(oplb._A, expected_A, atol=1e-12)
    np.testing.assert_allclose(oplb._b, expected_b, atol=1e-12)


def test_discount_gamma_bounds():
    with pytest.raises(ValueError, match="discount_gamma"):
        OPLB(d_psi=4, n_arms=2, config=OPLBConfig(discount_gamma=0.0))
    with pytest.raises(ValueError, match="discount_gamma"):
        OPLB(d_psi=4, n_arms=2, config=OPLBConfig(discount_gamma=1.5))


def test_stationary_default_unchanged():
    """Default config (no window, γ=1.0) should reproduce the original
    (A, b) exactly — backward-compat with every Phase-3/4 test."""
    d = 4
    cfg = OPLBConfig(lambda_reg=1.0, budget=float("inf"))
    oplb = OPLB(d_psi=d, n_arms=2, config=cfg)
    rng = np.random.default_rng(0)
    psis = [rng.standard_normal(d) for _ in range(10)]
    for psi in psis:
        oplb.update(0, psi, reward=0.5, realized_cost=0.0)
    expected_A = 1.0 * np.eye(d) + sum(np.outer(p, p) for p in psis)
    expected_b = sum(0.5 * p for p in psis)
    np.testing.assert_allclose(oplb._A, expected_A, atol=1e-12)
    np.testing.assert_allclose(oplb._b, expected_b, atol=1e-12)


def test_window_and_discount_combine():
    """Both options active: old rounds drop AND remaining rounds get discounted."""
    d = 3
    cfg = OPLBConfig(lambda_reg=1.0, budget=float("inf"), window_size=2, discount_gamma=0.5)
    oplb = OPLB(d_psi=d, n_arms=2, config=cfg)
    psi1 = np.array([1.0, 0.0, 0.0])
    psi2 = np.array([0.0, 1.0, 0.0])
    psi3 = np.array([0.0, 0.0, 1.0])
    oplb.update(0, psi1, reward=1.0, realized_cost=0.0)
    oplb.update(0, psi2, reward=1.0, realized_cost=0.0)
    oplb.update(0, psi3, reward=1.0, realized_cost=0.0)
    # After step 3: window drops psi1 first, then γ=0.5 multiplies remaining,
    # then psi3 added. Buffer = [psi2, psi3].
    # Step 3:
    #   drop psi1: A -= ψ1ψ1ᵀ, b -= ψ1
    #     → A = λI + ψ2ψ2ᵀ,  b = ψ2
    #   γ=0.5: A ← 0.5·(A − λI) + λI = λI + 0.5·ψ2ψ2ᵀ
    #          b ← 0.5·ψ2
    #   add psi3: A += ψ3ψ3ᵀ = λI + 0.5·ψ2ψ2ᵀ + ψ3ψ3ᵀ
    #            b += ψ3      = 0.5·ψ2 + ψ3
    expected_A = np.eye(d) + 0.5 * np.outer(psi2, psi2) + np.outer(psi3, psi3)
    expected_b = 0.5 * psi2 + psi3
    np.testing.assert_allclose(oplb._A, expected_A, atol=1e-12)
    np.testing.assert_allclose(oplb._b, expected_b, atol=1e-12)
