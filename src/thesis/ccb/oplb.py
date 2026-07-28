"""OPLB — Optimistic Pessimistic Linear Bandit under knapsack constraints.

Reference: Pacchiano, Ghavamzadeh, Bartlett, Jiang. *Stochastic Bandits with
Linear Constraints*. AISTATS 2021. (arXiv:2006.10185, ``pacchiano2021linconstr``
in our ``references.bib``).

We specialize the OPLB template to our CCB setup — see
`design-doc/ccb-formulation.md` §7.1 — with three simplifications:

  1. Deterministic cost ``c(a)`` per arm (known at arm-bank construction).
     Pacchiano's OPLB handles stochastic costs via an UCB on cost; for us,
     the cost-UCB radius collapses to zero. ``JUSTIFY:`` item
     "Knapsack cost model — deterministic per-arm cost vs linCBwK's
     stochastic-cost assumption" documented in
     ``design-doc/open-justifications.md``.
  2. Reward is binary (classification correctness) — still sub-Gaussian, so
     the OFUL confidence radius applies unchanged.
  3. Contextual features ``ψ(x, a)`` are constructed externally (the caller
     passes ``contexts[a] = ψ(x_t, a)`` to ``select`` and ``update``).
     ``make_psi`` provides the default concat-with-one-hot encoding.

Algorithm (per round t):

  A_0 = λI,  b_0 = 0
  select arm a_t = argmax_{a ∈ feasible} <ψ_{t,a}, θ̂_t> + α · √(ψ_{t,a}^T A_t^{-1} ψ_{t,a})
  observe reward r_t, realized cost c_t
  A_{t+1} = A_t + ψ_{t,a_t} ψ_{t,a_t}^T
  b_{t+1} = b_t + r_t · ψ_{t,a_t}
  θ̂_{t+1} = A_{t+1}^{-1} b_{t+1}  (Cholesky solve)
  budget_remaining -= c_t

``select`` returns the sentinel :data:`INFEASIBLE = -1` when no arm fits
the remaining budget or the optional per-round cap.

No EEG imports — this module is domain-agnostic and unit-tested on
synthetic linear-bandit environments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

#: Sentinel returned by :meth:`OPLB.select` when no feasible arm exists.
INFEASIBLE: int = -1


@dataclass
class OPLBConfig:
    """Hyperparameters for :class:`OPLB`.

    Parameters
    ----------
    lambda_reg : ridge regularization on the design matrix A_0 = λI. Default 1.0.
    alpha : exploration scale. Use :func:`default_alpha` for the OFUL /
        Pacchiano 2021 §5 theoretical value, or a constant for empirical
        tuning via the Commit-4 sensitivity sweep. Default 1.0.
    delta : confidence level (used only by :func:`default_alpha`). Default 0.05.
    budget : global knapsack ``B``. Default ``inf`` (unconstrained LinUCB).
    per_round_cap : optional ``K`` — arms with ``cost > K`` are infeasible
        at every round. ``None`` disables.
    cost_ucb_coef : reserved for a future stochastic-cost extension. Default 1.0.
    window_size : Phase-5 non-stationary option. If set, ``A`` and ``b`` only
        aggregate the last ``window_size`` ``(ψ, r)`` pairs — older rounds'
        contributions are subtracted as the buffer rolls. ``None`` (default)
        disables the window; semantics equivalent to ``window_size = inf``.
        ref: SlidingWindow-UCB variants surveyed in `heskebeck2022mabbci`.
    discount_gamma : Phase-5 non-stationary option. Before each
        :meth:`update`, ``A`` and ``b`` are multiplied by ``γ``, yielding
        an exponentially-weighted least-squares θ̂. ``1.0`` (default)
        disables discounting. Rounds older than ``log ε / log γ`` drop
        below an ``ε`` weight. ``window_size`` and ``discount_gamma`` can
        coexist — if both are set the window subtraction happens first,
        then the discount multiplies the result.
    """

    lambda_reg: float = 1.0
    alpha: float = 1.0
    delta: float = 0.05
    budget: float = float("inf")
    per_round_cap: float | None = None
    cost_ucb_coef: float = 1.0
    window_size: int | None = None
    discount_gamma: float = 1.0


def default_alpha(
    d: int,
    t: int,
    *,
    delta: float = 0.05,
    S: float = 1.0,
    lambda_reg: float = 1.0,
    sigma: float = 1.0,
    psi_bound: float = 1.0,
) -> float:
    """OFUL / Pacchiano 2021 §5 confidence-radius default for α at round t.

    α_t = σ · √(d · log((1 + t·L²/λ) / δ)) + √λ · S

    where ``σ`` bounds the sub-Gaussian noise, ``L`` bounds ‖ψ‖, and ``S``
    bounds ‖θ*‖. Defaults σ = L = S = 1 are the conventional unit-scale
    choices; scale your features to match or override explicitly.

    ref: `pacchiano2021linconstr` §5; `abbasi2011oful` Thm 2.
    """
    if t <= 0:
        # At t=0 the log argument degenerates; use the no-data value √λ·S.
        return float(np.sqrt(lambda_reg) * S)
    log_term = np.log((1.0 + t * psi_bound * psi_bound / lambda_reg) / delta)
    return float(sigma * np.sqrt(d * log_term) + np.sqrt(lambda_reg) * S)


class OPLB:
    """Contextual bandit with shared θ and a knapsack feasibility filter.

    Intended use from the CCB runner (Commit 3):

    >>> oplb = OPLB(d_psi=118, n_arms=64, config=OPLBConfig(budget=128.0))
    >>> for t in range(T):
    ...     contexts = np.stack([make_psi(x_t, onehot(a)) for a in arms])
    ...     a_idx = oplb.select(contexts=contexts, arm_costs=costs)
    ...     if a_idx == INFEASIBLE:
    ...         break
    ...     r_t, c_t = env.step(arms[a_idx])
    ...     oplb.update(a_idx, contexts[a_idx], r_t, realized_cost=c_t)
    """

    def __init__(self, d_psi: int, n_arms: int, config: OPLBConfig):
        if d_psi <= 0:
            raise ValueError(f"d_psi must be positive; got {d_psi}")
        if n_arms <= 0:
            raise ValueError(f"n_arms must be positive; got {n_arms}")
        if config.window_size is not None and config.window_size <= 0:
            raise ValueError(
                f"window_size must be a positive int or None; got {config.window_size}"
            )
        if not 0.0 < config.discount_gamma <= 1.0:
            raise ValueError(f"discount_gamma must be in (0, 1]; got {config.discount_gamma}")
        self.d = d_psi
        self.n_arms = n_arms
        self.config = config

        self._A = config.lambda_reg * np.eye(d_psi)
        self._b = np.zeros(d_psi)
        self._theta_hat = np.zeros(d_psi)
        self._t = 0
        self._budget_remaining = float(config.budget)
        # Non-stationary ring buffer: list of (psi, reward) pairs ordered
        # oldest-first. Only allocated when window_size or discount_gamma
        # is non-default; the stationary path uses incremental updates.
        self._nonstationary = config.window_size is not None or config.discount_gamma < 1.0
        self._history: list[tuple[np.ndarray, float]] | None = [] if self._nonstationary else None

    @property
    def theta_hat(self) -> np.ndarray:
        """Current ridge estimate of the latent θ*. Shape ``(d_psi,)``."""
        return self._theta_hat.copy()

    @property
    def budget_remaining(self) -> float:
        """Remaining budget after all :meth:`update` calls so far."""
        return self._budget_remaining

    @property
    def t(self) -> int:
        """Number of completed update steps."""
        return self._t

    def select(
        self,
        contexts: np.ndarray,
        arm_costs: np.ndarray,
        *,
        alpha: float | None = None,
    ) -> int:
        """Choose the argmax-UCB arm among feasible ones, or :data:`INFEASIBLE`.

        Parameters
        ----------
        contexts : shape ``(n_arms, d_psi)`` — ψ(x_t, a) for each arm.
        arm_costs : shape ``(n_arms,)`` — deterministic ``c(a)``. All must be
            non-negative.
        alpha : optional override for ``OPLBConfig.alpha``. Pass 0.0 to play
            pure exploitation (useful to freeze the policy on a test split).

        Returns
        -------
        Arm index in ``[0, n_arms)`` or :data:`INFEASIBLE` if no arm fits.
        """
        contexts = np.asarray(contexts, dtype=float)
        arm_costs = np.asarray(arm_costs, dtype=float)
        if contexts.shape != (self.n_arms, self.d):
            raise ValueError(
                f"contexts shape {contexts.shape} != (n_arms={self.n_arms}, d={self.d})"
            )
        if arm_costs.shape != (self.n_arms,):
            raise ValueError(f"arm_costs shape {arm_costs.shape} != ({self.n_arms},)")
        if np.any(arm_costs < 0):
            raise ValueError("arm_costs must be non-negative")

        if self._budget_remaining <= 0:
            return INFEASIBLE

        # Feasibility: cost must fit remaining budget (and optional per-round cap).
        feasible = arm_costs <= self._budget_remaining
        if self.config.per_round_cap is not None:
            feasible &= arm_costs <= self.config.per_round_cap
        if not np.any(feasible):
            return INFEASIBLE

        alpha_t = self.config.alpha if alpha is None else alpha

        # Vectorized UCB: solve A^{-1} Ψ^T once via Cholesky, then quadratic
        # form per row. Cheaper than per-arm triangular solves for large n_arms.
        cho = cho_factor(self._A)
        means = contexts @ self._theta_hat  # (n_arms,)
        A_inv_psi = cho_solve(cho, contexts.T).T  # (n_arms, d_psi)
        quad = np.einsum("ad,ad->a", contexts, A_inv_psi)
        quad = np.clip(quad, 0.0, None)  # numerical safety
        ucbs = means + alpha_t * np.sqrt(quad)

        # Mask infeasible arms with -inf so argmax skips them.
        ucbs = np.where(feasible, ucbs, -np.inf)
        return int(np.argmax(ucbs))

    def update(
        self,
        arm_idx: int,
        psi: np.ndarray,
        reward: float,
        realized_cost: float,
    ) -> None:
        """Online ridge update after pulling ``arm_idx`` with feature ``psi``.

        Updates A_t, b_t, θ̂, and decrements the budget by ``realized_cost``.
        """
        if not 0 <= arm_idx < self.n_arms:
            raise ValueError(f"arm_idx out of range: {arm_idx}")
        psi = np.asarray(psi, dtype=float)
        if psi.shape != (self.d,):
            raise ValueError(f"psi shape {psi.shape} != ({self.d},)")
        if realized_cost < 0:
            raise ValueError(f"realized_cost must be non-negative; got {realized_cost}")

        if self._nonstationary:
            # Rebuild A / b from scratch over the admissible history window
            # with per-entry discount weights. O(window · d²) per update,
            # tractable for the window sizes we use (≤ 170 for 2b MI).
            assert self._history is not None
            self._history.append((psi.copy(), float(reward)))
            window = self.config.window_size
            if window is not None and len(self._history) > window:
                self._history = self._history[-window:]
            gamma = self.config.discount_gamma
            lam = self.config.lambda_reg
            A = lam * np.eye(self.d)
            b = np.zeros(self.d)
            n = len(self._history)
            for i, (p, r) in enumerate(self._history):
                # Weight = γ^(rounds since entry was inserted). Newest entry
                # (i = n-1) has weight 1; oldest (i = 0) has weight γ^(n-1).
                w = gamma ** (n - 1 - i)
                A = A + w * np.outer(p, p)
                b = b + w * r * p
            self._A = A
            self._b = b
        else:
            self._A = self._A + np.outer(psi, psi)
            self._b = self._b + float(reward) * psi

        cho = cho_factor(self._A)
        self._theta_hat = cho_solve(cho, self._b)
        self._budget_remaining -= float(realized_cost)
        self._t += 1


def make_psi(context: np.ndarray, arm_onehot: np.ndarray) -> np.ndarray:
    """Default shared-θ feature: ``ψ(x, a) = [x ; onehot(a)]``.

    Dim: ``len(context) + len(arm_onehot)``. Simple and widely used; see
    `li2010linucb` §4 (hybrid model), and `design-doc/ccb-formulation.md`
    §6.4. Alternative embeddings (dense arm features by band/spatial/…)
    are future work.
    """
    context = np.asarray(context, dtype=float).ravel()
    arm_onehot = np.asarray(arm_onehot, dtype=float).ravel()
    return np.concatenate([context, arm_onehot], axis=0)
