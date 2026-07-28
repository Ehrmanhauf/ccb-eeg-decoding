"""Alternative CCB policies — ablations from ``design-doc/ccb-formulation.md`` §8.4
and Phase-5 Stage-1 algorithmic fixes.

Four drop-in alternatives to the default :class:`thesis.ccb.oplb.OPLB` policy:

- :class:`FixedArmPolicy` — commits to a single arm (typically the top-κ arm
  from calibration pruning) and pulls it every round. Isolates the
  contribution of online exploration: any κ improvement OPLB gets over
  fixed-arm is attributable to the bandit itself, not the arm-pool
  calibration.
- :class:`EpsilonGreedyPolicy` — with probability ε pick a uniform feasible
  arm, otherwise pick greedy ``argmax ⟨ψ, θ̂⟩`` (no UCB radius). Shares
  OPLB's ridge update rule. Isolates the value of structured (UCB)
  exploration over random exploration at our horizon T.
- :class:`LinTSPolicy` — Phase-5 Thompson Sampling (Agrawal & Goyal 2013).
  Bayesian alternative to OPLB's UCB: sample ``θ̃ ~ N(θ̂, v² A^{-1})`` per
  round and greedy-argmax. Addresses Phase-4 failure mode #1 (UCB radius
  mis-scaled for short T ≈ 170). Reuses OPLB's ridge bookkeeping via
  composition. ref: `agrawal2013ts`.
- :func:`make_unconstrained_linucb` — constructs an :class:`OPLB` with the
  knapsack stripped (``budget=inf``, ``per_round_cap=None``). Isolates the
  cost of the constraint itself.

All policies satisfy the :class:`Policy` structural interface consumed by
:func:`thesis.ccb.runner.run_ccb_on_split` via a factory callable. They
return :data:`thesis.ccb.oplb.INFEASIBLE` when no arm fits the remaining
budget or the per-round cap.

ref: `design-doc/ccb-formulation.md` §8.4 (the four ablations).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.linalg import cho_factor, solve_triangular

from thesis.ccb.oplb import INFEASIBLE, OPLB, OPLBConfig


class Policy(Protocol):
    """Structural interface the CCB runner consumes.

    :class:`OPLB` already satisfies this contract; the classes in this module
    add alternative implementations for the §8.4 ablations.
    """

    def select(
        self,
        contexts: np.ndarray,
        arm_costs: np.ndarray,
        *,
        alpha: float | None = None,
    ) -> int: ...

    def update(
        self,
        arm_idx: int,
        psi: np.ndarray,
        reward: float,
        realized_cost: float,
    ) -> None: ...

    @property
    def budget_remaining(self) -> float: ...


class FixedArmPolicy:
    """Commit to a single arm for every round; no updates, no exploration.

    The default ``fixed_arm_idx=0`` targets the top-κ surviving arm because
    :func:`thesis.ccb.arms.prune_arms` returns its output sorted by κ
    descending.

    Per-round feasibility still applies — if the fixed arm's cost exceeds
    remaining budget or the per-round cap, :meth:`select` returns
    :data:`INFEASIBLE`.
    """

    def __init__(
        self,
        d_psi: int,  # noqa: ARG002 — unused; kept for uniform factory signature.
        n_arms: int,
        config: OPLBConfig,
        *,
        fixed_arm_idx: int = 0,
    ):
        if not 0 <= fixed_arm_idx < n_arms:
            raise ValueError(f"fixed_arm_idx={fixed_arm_idx} out of range [0, {n_arms})")
        self.n_arms = n_arms
        self.fixed_arm_idx = fixed_arm_idx
        self.config = config
        self._budget_remaining = float(config.budget)
        self._t = 0

    def select(
        self,
        contexts: np.ndarray,  # noqa: ARG002 — context ignored; fixed-arm policy.
        arm_costs: np.ndarray,
        *,
        alpha: float | None = None,  # noqa: ARG002 — kept for Policy-interface parity.
    ) -> int:
        if self._budget_remaining <= 0:
            return INFEASIBLE
        cost = float(np.asarray(arm_costs)[self.fixed_arm_idx])
        if cost > self._budget_remaining:
            return INFEASIBLE
        cap = self.config.per_round_cap
        if cap is not None and cost > cap:
            return INFEASIBLE
        return self.fixed_arm_idx

    def update(
        self,
        arm_idx: int,  # noqa: ARG002 — fixed-arm ignores identity.
        psi: np.ndarray,  # noqa: ARG002 — no θ̂ to update.
        reward: float,  # noqa: ARG002 — reward not used.
        realized_cost: float,
    ) -> None:
        if realized_cost < 0:
            raise ValueError(f"realized_cost must be non-negative; got {realized_cost}")
        self._budget_remaining -= float(realized_cost)
        self._t += 1

    @property
    def budget_remaining(self) -> float:
        return self._budget_remaining


class EpsilonGreedyPolicy:
    """Epsilon-greedy linear bandit sharing OPLB's ridge estimator.

    With probability ``epsilon`` pick a uniform feasible arm; otherwise pick
    ``argmax_a ⟨ψ(x, a), θ̂⟩`` (greedy, no UCB bonus). Update rule is
    identical to OPLB's — :class:`OPLB` is reused internally for θ̂ and the
    budget accounting.
    """

    def __init__(
        self,
        d_psi: int,
        n_arms: int,
        config: OPLBConfig,
        *,
        epsilon: float = 0.1,
        rng_seed: int = 0,
    ):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1]; got {epsilon}")
        self._oplb = OPLB(d_psi=d_psi, n_arms=n_arms, config=config)
        self.epsilon = float(epsilon)
        self._rng = np.random.default_rng(rng_seed)

    def _feasible_mask(self, arm_costs: np.ndarray) -> np.ndarray:
        feasible = arm_costs <= self._oplb.budget_remaining
        cap = self._oplb.config.per_round_cap
        if cap is not None:
            feasible &= arm_costs <= cap
        return feasible

    def select(
        self,
        contexts: np.ndarray,
        arm_costs: np.ndarray,
        *,
        alpha: float | None = None,  # noqa: ARG002 — ε-greedy ignores the UCB radius.
    ) -> int:
        arm_costs = np.asarray(arm_costs, dtype=float)
        if self._oplb.budget_remaining <= 0:
            return INFEASIBLE

        if self._rng.random() < self.epsilon:
            feasible = self._feasible_mask(arm_costs)
            feasible_ids = np.where(feasible)[0]
            if feasible_ids.size == 0:
                return INFEASIBLE
            return int(self._rng.choice(feasible_ids))

        # Greedy branch: reuse OPLB.select with α=0 for feasibility-aware argmax.
        return self._oplb.select(contexts=contexts, arm_costs=arm_costs, alpha=0.0)

    def update(
        self,
        arm_idx: int,
        psi: np.ndarray,
        reward: float,
        realized_cost: float,
    ) -> None:
        self._oplb.update(arm_idx, psi, reward, realized_cost)

    @property
    def budget_remaining(self) -> float:
        return self._oplb.budget_remaining


class LinTSPolicy:
    """Thompson Sampling for linear contextual bandits (Agrawal & Goyal 2013).

    At each round we sample ``θ̃_t ~ N(θ̂_t, v² A_t^{-1})`` and play
    ``argmax_a ⟨ψ(x_t, a), θ̃_t⟩`` over feasible arms. No exploration bonus
    term — exploration comes from posterior sampling. Update rule is
    identical to OPLB's ridge estimator (reused via composition).

    Phase-4 evidence (`results/ccb_ablation_policy.csv`) showed that UCB with
    the Pacchiano-default ``alpha`` is mis-scaled for our short horizon
    T ≈ 170 — ε-greedy beat OPLB by Δκ = +0.046 / +0.054. TS removes the
    explicit UCB-radius choice, replacing it with a posterior-scale
    hyperparameter ``prior_scale`` (``v``). Expected Phase-5 outcome: TS is
    competitive with or better than ε-greedy on the same cell.

    Parameters
    ----------
    d_psi, n_arms, config
        Same as :class:`OPLB`. ``config.alpha`` is ignored by TS (no UCB
        radius); ``config.lambda_reg`` / ``budget`` / ``per_round_cap`` are
        honored via the composed OPLB instance.
    prior_scale : ``v`` in the notation above. Default 1.0 matches
        Agrawal & Goyal 2013's unit-noise / unit-prior convention; larger
        values explore more aggressively.
    rng_seed : RNG seed for the Gaussian sample.
    """

    def __init__(
        self,
        d_psi: int,
        n_arms: int,
        config: OPLBConfig,
        *,
        prior_scale: float = 1.0,
        rng_seed: int = 0,
    ):
        if prior_scale <= 0:
            raise ValueError(f"prior_scale must be positive; got {prior_scale}")
        self._oplb = OPLB(d_psi=d_psi, n_arms=n_arms, config=config)
        self.prior_scale = float(prior_scale)
        self._rng = np.random.default_rng(rng_seed)

    def _sample_theta(self) -> np.ndarray:
        """Draw ``θ̃ ~ N(θ̂, v² A^{-1})``.

        Uses Cholesky of A: if A = LLᵀ, then A⁻¹ = L⁻ᵀL⁻¹ and a sample is
        ``θ̂ + v · L⁻ᵀ · z`` where ``z ~ N(0, I_d)``. Cov check:
        ``v · L⁻ᵀ · I_d · (v · L⁻ᵀ)ᵀ = v² L⁻ᵀ L⁻¹ = v² A⁻¹``. ✓
        """
        L_and_lower = cho_factor(self._oplb._A, lower=True)
        L = L_and_lower[0]
        z = self._rng.standard_normal(self._oplb.d)
        # Solve Lᵀ x = z  →  x = L⁻ᵀ z
        x = solve_triangular(L, z, lower=True, trans="T")
        return self._oplb.theta_hat + self.prior_scale * x

    def select(
        self,
        contexts: np.ndarray,
        arm_costs: np.ndarray,
        *,
        alpha: float | None = None,  # noqa: ARG002 — TS ignores UCB radius
    ) -> int:
        contexts = np.asarray(contexts, dtype=float)
        arm_costs = np.asarray(arm_costs, dtype=float)
        if contexts.shape != (self._oplb.n_arms, self._oplb.d):
            raise ValueError(
                f"contexts shape {contexts.shape} != (n_arms={self._oplb.n_arms}, d={self._oplb.d})"
            )
        if arm_costs.shape != (self._oplb.n_arms,):
            raise ValueError(f"arm_costs shape {arm_costs.shape} != ({self._oplb.n_arms},)")
        if self._oplb.budget_remaining <= 0:
            return INFEASIBLE

        feasible = arm_costs <= self._oplb.budget_remaining
        cap = self._oplb.config.per_round_cap
        if cap is not None:
            feasible &= arm_costs <= cap
        if not np.any(feasible):
            return INFEASIBLE

        theta_tilde = self._sample_theta()
        scores = contexts @ theta_tilde
        scores = np.where(feasible, scores, -np.inf)
        return int(np.argmax(scores))

    def update(
        self,
        arm_idx: int,
        psi: np.ndarray,
        reward: float,
        realized_cost: float,
    ) -> None:
        self._oplb.update(arm_idx, psi, reward, realized_cost)

    @property
    def budget_remaining(self) -> float:
        return self._oplb.budget_remaining


def make_unconstrained_linucb(d_psi: int, n_arms: int, config: OPLBConfig) -> OPLB:
    """Construct an :class:`OPLB` with the knapsack stripped.

    Preserves caller's ``alpha``, ``lambda_reg``, ``delta``, and
    ``cost_ucb_coef``; replaces ``budget`` with ``+inf`` and
    ``per_round_cap`` with ``None``. Used as the §8.4 "CCB without
    constraint" ablation.
    """
    unconstrained = OPLBConfig(
        lambda_reg=config.lambda_reg,
        alpha=config.alpha,
        delta=config.delta,
        budget=float("inf"),
        per_round_cap=None,
        cost_ucb_coef=config.cost_ucb_coef,
    )
    return OPLB(d_psi=d_psi, n_arms=n_arms, config=unconstrained)
