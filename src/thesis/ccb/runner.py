"""CCB runner — glues Phase 2 (data/protocols) to Phase 3 (arms + OPLB).

One call to :func:`run_ccb_on_split` fits arm heads on a calibration portion
of the train split, runs the OPLB bandit online on the remaining train
portion (with hindsight regret tracking), and then scores the test split
under a frozen policy (α=0, no updates). Returns a :class:`CCBResult`
with κ, accuracy, per-round arm pulls, cumulative regret, and the
budget trace.

**Leakage invariant.** The 2a loader is *never* imported here — this
module only touches 2b via ``SubjectData`` passed in from the caller.
A defensive check at the top of ``run_ccb_on_split`` also rejects any
SubjectData whose ``dataset_name`` starts with ``BCICIV-2a``. See
`CLAUDE.md` §2 and the test
``tests/test_ccb_runner.py::test_runner_no_leakage_of_2a``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from thesis.ccb.arms import (
    Arm,
    ArmHead,
    build_arm_heads,
    enumerate_arms_2a,
    enumerate_arms_2b,
    enumerate_arms_generic,
    prune_arms,
)
from thesis.ccb.context import N_ARM_FAMILIES, compute_context, compute_context_2a, context_dim
from thesis.ccb.oplb import INFEASIBLE, OPLB, OPLBConfig
from thesis.ccb.policies import Policy
from thesis.data import SubjectData
from thesis.metrics import compute_metrics
from thesis.protocols import Split

PolicyFactory = Callable[[int, int, OPLBConfig], Policy]


@dataclass(frozen=True)
class CCBResult:
    """Outcome of one :func:`run_ccb_on_split` call."""

    subject: int
    protocol: str
    kappa: float
    accuracy: float
    n_test: int
    cumulative_regret: np.ndarray  # (n_stream_used,) hindsight regret curve
    arm_pulls: np.ndarray  # (n_stream_used,) arm_id for each pulled round
    budget_trace: np.ndarray  # (n_stream_used,) budget remaining post-step
    n_arms_surviving: int
    # Test-split ground truth + predictions. Default ``None`` keeps every
    # existing caller untouched; populated so cross-subject protocols (LOSO)
    # can pool per-fold predictions into one global κ. Shape ``(n_test,)``.
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None


# Map spatial-filter names to the 0-based index of the running-reward feature
# slot in the context vector (indices 15, 16, 17 per ``context.py``).
_ARM_FAMILY_INDEX: dict[str, int] = {"csp": 0, "laplacian": 1, "identity": 2}


def _calibration_stream_split(
    train_idx: np.ndarray, calibration_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Partition ``train_idx`` into (calibration, bandit-stream) halves.

    The two returned arrays are disjoint and together cover ``train_idx``.
    See ``CLAUDE.md`` §2 and the runtime-assertion below.
    """
    rng = np.random.default_rng(seed)
    perm = train_idx.copy()
    rng.shuffle(perm)
    n_cal = int(round(calibration_frac * len(perm)))
    n_cal = max(n_cal, 2)  # need at least 2 trials for CSP class covariances
    return perm[:n_cal], perm[n_cal:]


def fit_heads_on_calibration(
    data: SubjectData,
    train_idx: np.ndarray,
    arms: list[Arm],
    *,
    calibration_frac: float = 0.3,
    head_fit_frac: float = 0.6,
    min_kappa: float = 0.05,
    max_arms: int = 100,
    seed: int = 42,
) -> tuple[list[Arm], dict[int, ArmHead], np.ndarray]:
    """Fit arm heads on a calibration portion of ``train_idx`` and prune.

    Calibration is split internally into a **head-fit** part (default 60 %)
    and a **prune-holdout** part (default 40 %); the prune rule drops arms
    whose held-out κ on the prune-holdout is below ``min_kappa`` and then
    keeps the top-``max_arms`` by κ.

    Returns
    -------
    surviving_arms : pruned list sorted by κ desc.
    heads : dict keyed by ``arm_id`` (only for surviving arms).
    bandit_stream_idx : remaining train indices not used for calibration.
    """
    cal_idx, stream_idx = _calibration_stream_split(train_idx, calibration_frac, seed)

    # Head-fit / prune-holdout split inside the calibration set.
    rng = np.random.default_rng(seed + 1)
    shuffled = cal_idx.copy()
    rng.shuffle(shuffled)
    n_fit = max(int(round(head_fit_frac * len(shuffled))), 2)
    fit_idx = shuffled[:n_fit]
    ho_idx = shuffled[n_fit:]

    heads = build_arm_heads(arms, data.X[fit_idx], data.y[fit_idx], data.sfreq)
    surviving = prune_arms(
        arms,
        heads,
        data.X[ho_idx],
        data.y[ho_idx],
        data.sfreq,
        min_kappa=min_kappa,
        max_arms=max_arms,
    )
    surviving_heads = {arm.arm_id: heads[arm.arm_id] for arm in surviving}
    return surviving, surviving_heads, stream_idx


def _build_per_arm_contexts(
    base_context: np.ndarray,
    recent_arm_rewards: np.ndarray,
    n_arms: int,
    d_ctx: int,
    *,
    include_recent_rewards: bool = True,
) -> np.ndarray:
    """Stack per-arm ψ(x, a) = [context ; one-hot(a)] rows.

    With ``include_recent_rewards=True`` (default): ``d_ctx = 18``.
    With ``include_recent_rewards=False`` (Phase-4 ablation): ``d_ctx = 15`` and
    ``recent_arm_rewards`` is ignored — the context drops the per-family
    running-reward tail. Returns shape ``(n_arms, d_ctx + n_arms)``.
    """
    if include_recent_rewards:
        ctx_full = np.concatenate([base_context, recent_arm_rewards], axis=0)
    else:
        ctx_full = base_context
    assert ctx_full.shape == (d_ctx,), f"unexpected context dim {ctx_full.shape}"
    tiled = np.tile(ctx_full, (n_arms, 1))
    return np.concatenate([tiled, np.eye(n_arms)], axis=1)


def run_ccb_on_split(
    data: SubjectData,
    split: Split,
    *,
    arms: list[Arm] | None = None,
    config: OPLBConfig | None = None,
    calibration_frac: float = 0.3,
    min_kappa: float = 0.05,
    max_arms: int = 100,
    budget_frac: float = 1.0,
    seed: int = 42,
    include_recent_rewards: bool = True,
    policy_factory: PolicyFactory | None = None,
    online_heads: bool = False,
    allow_22ch_research_question: bool = False,
    workload_channel_roles: dict[str, list[int]] | None = None,
    reward_mode: str = "accuracy",
) -> CCBResult:
    """End-to-end CCB on one (subject, split).

    Steps:
      1. Reject 2a data up front (leakage guard).
      2. Fit + prune arms on a calibration portion of ``split.train_idx``.
      3. Build OPLB over the surviving arm pool with ``budget = n_test × median_cost × budget_frac``.
      4. Run OPLB online over the bandit-stream portion; track hindsight regret.
      5. Freeze OPLB (``α=0``, no updates) and score ``split.test_idx``.

    Parameters
    ----------
    data : subject's :class:`SubjectData` — must be from 2b (``dataset_name``
        starts with ``"BCICIV-2b"``).
    split : train/test index split from ``thesis.protocols``.
    arms : explicit arm list. Defaults to ``enumerate_arms_2b(data.sfreq)``.
    config : explicit :class:`OPLBConfig`. Defaults to ``OPLBConfig(alpha=1.0)``
        with ``budget`` derived from ``budget_frac`` (see below).
    calibration_frac : fraction of ``split.train_idx`` used for
        arm-head fitting + pruning. Default 0.3; the remaining 0.7 is the
        bandit stream. ``JUSTIFY:`` tracked as "Calibration fraction for
        arm-head pretraining" in open-justifications.md (sweep in Commit 4).
    min_kappa : prune arms whose held-out κ falls below this.
    max_arms : hard cap on the surviving arm pool.
    budget_frac : multiplier on ``n_test_trials × median_arm_cost`` when
        ``config.budget`` is not explicitly set.
    seed : RNG seed for the calibration / bandit-stream permutation.
    include_recent_rewards : whether the per-trial context includes the
        3-dim running-mean reward tail per arm family (``csp``, ``laplacian``,
        ``identity``). Default ``True`` (``d_ctx = 18``); set to ``False``
        for the Phase-4 context ablation (``d_ctx = 15``). Phase-4 deferral
        from `open-justifications.md` "Context feature vector" (line 68).
    policy_factory : ``(d_psi, n_arms, config) -> Policy`` factory for the
        online-learning algorithm. Default ``None`` → constructs the canonical
        :class:`thesis.ccb.oplb.OPLB`. Pass a factory from
        :mod:`thesis.ccb.policies` for the §8.4 ablations (fixed-arm,
        ε-greedy, unconstrained LinUCB).
    online_heads : if ``True``, after each bandit pull the pulled arm's
        head is updated via :meth:`ArmHead.partial_fit` with the newly-
        labelled trial. Default ``False`` (Phase-4 frozen-heads behaviour).
        Phase-5 addition to target failure mode #2 (session-gap drift). Only
        the pulled arm updates — standard bandit partial-feedback framing.
    reward_mode : bandit reward shaping. ``"accuracy"`` (default) uses the 0/1
        correctness reward r_t = 1{ŷ_t = y_t} — byte-identical to all prior
        results. ``"balanced"`` weights each correct prediction by the inverse
        training-class frequency w_c = 1/(K·freq_train(c)), a per-trial
        balanced-accuracy proxy that aligns arm *selection* with Cohen's κ
        rather than raw accuracy. Class frequencies are computed train-only (no
        leakage). Used for the reward-objective robustness check; it changes only
        which arms the policy learns to select, not the test-phase κ scoring.
    """
    # No-leakage guard. The thesis's headline question is the **3-channel
    # low-channel** deployment scenario; for that question, 2a (22-ch) data
    # must not enter the CCB pipeline (`CLAUDE.md` §2). Phase-5 §2.1 adds
    # a separate research sub-question — "how does the CCB perform when the
    # channel budget is relaxed?" — for which 2a is the appropriate dataset.
    # The guard is therefore reframed as conditional: it blocks 2a by
    # default (preserving every existing 3-ch result) and is opt-in via the
    # explicit `allow_22ch_research_question=True` flag for the §2.1 sub-
    # experiment only. The 3-ch thesis pipeline never sets this flag.
    if data.dataset_name.startswith("BCICIV-2a") and not allow_22ch_research_question:
        raise RuntimeError(
            f"CCB must not run on 2a data (got dataset_name={data.dataset_name!r}) "
            "for the 3-channel thesis pipeline (CLAUDE.md §2 no-leakage). "
            "Pass allow_22ch_research_question=True only for the Phase-5 §2.1 "
            "sub-experiment, which asks a separate research question about "
            "high-channel CCB behaviour."
        )

    is_2a = data.dataset_name.startswith("BCICIV-2a")
    is_2b = data.dataset_name.startswith("BCICIV-2b")
    # Cognitive-load datasets use the workload context. The near-ear reframe adds
    # UAB and COG-BCI (and their position-based ``-nearear`` T7/T8 subsets, whose
    # dataset_name is e.g. ``"UAB-nearear"``); split on ``-`` to match the base.
    is_cl = data.dataset_name.split("-")[0] in {"STEW", "WAUC", "UAB", "COGBCI"}
    # 3-ch 2b uses compute_context (validates shape (3, n_samples)).
    # 22-ch 2a uses compute_context_2a (validates shape (22, n_samples)).
    # CL datasets (STEW, WAUC) with an explicit ``workload_channel_roles``
    # map use the CL-specific workload context (θ/α/β per channel +
    # frontal-θ, parietal-α, frontal-α asymmetry, engagement, artifact
    # flag, bias) — methodologically appropriate for cognitive-load
    # classification per `design-doc/ccb-formulation.md` §2.6 / §2.7.
    # Other multi-channel datasets (PhysioNet 64ch, BNCI2015_004 30ch)
    # require an n_channels-aware context. For Phase-5 §2.2 / §2.4 MVP
    # we reuse compute_context_generic for any non-2a/non-2b dataset
    # without a workload-roles map.
    n_ch_data = data.X.shape[1]
    cl_context = is_cl and workload_channel_roles is not None
    if is_2b:
        context_fn = compute_context
    elif is_2a:
        context_fn = compute_context_2a
    elif cl_context:
        from thesis.ccb.context_cl import compute_context_workload
        _roles = workload_channel_roles  # bind in closure
        context_fn = lambda epoch, sfreq, recent_arm_rewards: compute_context_workload(  # noqa: E731
            epoch, sfreq=sfreq, channel_roles=_roles, recent_arm_rewards=recent_arm_rewards
        )
    else:
        # Generic n-channel MI-derived context fallback.
        from thesis.ccb.context import compute_context_generic
        context_fn = lambda epoch, sfreq, recent_arm_rewards: compute_context_generic(  # noqa: E731
            epoch, sfreq=sfreq, recent_arm_rewards=recent_arm_rewards
        )

    if arms is None:
        if is_2b:
            arms = enumerate_arms_2b(data.sfreq)
        elif is_2a:
            arms = enumerate_arms_2a(data.sfreq)
        else:
            arms = enumerate_arms_generic(n_ch_data)

    sfreq = data.sfreq
    X_all = data.X
    y_all = data.y

    # Calibration + prune.
    surviving, heads, stream_idx = fit_heads_on_calibration(
        data,
        split.train_idx,
        arms,
        calibration_frac=calibration_frac,
        min_kappa=min_kappa,
        max_arms=max_arms,
        seed=seed,
    )
    n_arms = len(surviving)
    if n_arms == 0:
        raise RuntimeError(
            "All arms pruned; pool is empty. Lower min_kappa or check calibration data."
        )

    # Runtime leakage guard: calibration ∩ stream must be empty (also
    # ``split.test_idx`` is disjoint from train by protocol construction;
    # no need to re-check).
    assert not (
        set(np.atleast_1d(stream_idx).tolist())
        & set(np.atleast_1d(np.setdiff1d(split.train_idx, stream_idx)).tolist())
    ), "calibration ∩ bandit-stream must be empty"

    arm_costs = np.array([arm.cost for arm in surviving], dtype=float)

    # Budget default: scaled to the total number of rounds the policy will
    # face (bandit stream + test phase) so ``budget_frac = 1.0`` means "one
    # median-cost pull per round on average — no scarcity". Values < 1 bite
    # the knapsack; values > 1 provide slack. Stream length is known now,
    # test length is fixed by the split.
    n_test = len(split.test_idx)
    n_stream_expected = len(stream_idx)
    default_budget = (n_stream_expected + n_test) * float(np.median(arm_costs)) * budget_frac
    if config is None:
        config = OPLBConfig(alpha=1.0, lambda_reg=1.0, budget=default_budget)
    elif config.budget == float("inf"):
        # Inject the budget default but preserve every other caller-set
        # field (including Phase-5 window_size / discount_gamma). Use
        # ``dataclasses.replace`` so newly-added OPLBConfig fields are
        # never silently dropped from the rebuilt config — that was the
        # original 5.3 runner wiring bug.
        config = replace(config, budget=default_budget)

    # The CL workload context has a different base dimensionality than the
    # MI-derived contexts (CL: base 9; MI: base 15). Switch dimensionality
    # calculation accordingly so that the policy's d_psi matches the
    # context-vector shape that ``context_fn`` will actually emit.
    if cl_context:
        from thesis.ccb.context_cl import context_dim_workload
        d_ctx = context_dim_workload(
            include_recent_rewards=include_recent_rewards,
            n_recent_arms=N_ARM_FAMILIES,
        )
    else:
        d_ctx = context_dim(
            include_recent_rewards=include_recent_rewards,
            n_recent_arms=N_ARM_FAMILIES,
        )
    d_psi = d_ctx + n_arms
    if policy_factory is None:
        policy: Policy = OPLB(d_psi=d_psi, n_arms=n_arms, config=config)
    else:
        policy = policy_factory(d_psi, n_arms, config)

    # ------------------------------------------------------------------
    # Stream (bandit) phase
    # ------------------------------------------------------------------
    n_stream = len(stream_idx)
    X_stream = X_all[stream_idx]
    y_stream = y_all[stream_idx]

    # Reward shaping (§4.x objective-robustness check). "accuracy" leaves the 0/1
    # reward untouched (default; byte-identical to all prior results). "balanced"
    # up-weights a correct prediction on a rarer class by the inverse train-class
    # frequency w_c = 1/(K·freq_train(c)) = N/(K·count_c) — a per-trial
    # balanced-accuracy proxy that pushes arm selection toward κ rather than raw
    # accuracy. Frequencies are train-only (no leakage).
    reward_weight: dict[str, float] | None = None
    if reward_mode == "balanced":
        _tr = y_all[split.train_idx]
        _cls, _cnt = np.unique(_tr, return_counts=True)
        _K = len(_cls)
        reward_weight = {str(c): float(len(_tr) / (_K * n)) for c, n in zip(_cls, _cnt, strict=True)}
    elif reward_mode != "accuracy":
        raise ValueError(f"reward_mode must be 'accuracy' or 'balanced', got {reward_mode!r}")

    # With frozen heads we can precompute per-arm predictions once; with
    # online heads we must recompute per trial (heads drift as pulls accrue).
    if online_heads:
        per_arm_preds_static = None
    else:
        per_arm_preds_static = np.stack(
            [heads[arm.arm_id].predict(X_stream, sfreq) for arm in surviving],
            axis=1,
        )  # (n_stream, n_arms) of string labels

    # Pre-compute base (15-dim) contexts per stream trial; we'll splice in
    # running arm-family rewards on the fly.
    base_contexts = np.stack(
        [
            context_fn(X_stream[j], sfreq=sfreq, recent_arm_rewards=None)
            for j in range(n_stream)
        ],
        axis=0,
    )

    arm_pulls = np.full(n_stream, -1, dtype=int)
    rewards = np.zeros(n_stream)
    budget_trace = np.zeros(n_stream)
    per_arm_rewards = np.zeros((n_stream, n_arms), dtype=float)
    running_rewards = np.zeros(N_ARM_FAMILIES)
    running_counts = np.zeros(N_ARM_FAMILIES)

    stop_j: int = n_stream  # trim point if budget runs out early
    for j in range(n_stream):
        recent = running_rewards / np.maximum(running_counts, 1.0)
        contexts_mat = _build_per_arm_contexts(
            base_contexts[j],
            recent,
            n_arms,
            d_ctx,
            include_recent_rewards=include_recent_rewards,
        )
        # Populate per-arm rewards for this trial — frozen path hits the
        # precomputed matrix; online path queries each arm's current head.
        if per_arm_preds_static is not None:
            per_arm_rewards[j] = (per_arm_preds_static[j] == y_stream[j]).astype(float)
        else:
            trial = X_stream[j : j + 1]
            for k, arm in enumerate(surviving):
                pred = heads[arm.arm_id].predict(trial, sfreq)[0]
                per_arm_rewards[j, k] = float(pred == y_stream[j])

        if reward_weight is not None:  # balanced reward: scale this trial by 1/(K·freq) of its true class
            per_arm_rewards[j] *= reward_weight.get(str(y_stream[j]), 1.0)

        a_idx = policy.select(contexts=contexts_mat, arm_costs=arm_costs)
        if a_idx == INFEASIBLE:
            stop_j = j
            break
        arm_pulls[j] = surviving[a_idx].arm_id
        r = float(per_arm_rewards[j, a_idx])
        rewards[j] = r
        policy.update(
            a_idx,
            contexts_mat[a_idx],
            reward=r,
            realized_cost=float(arm_costs[a_idx]),
        )
        if online_heads:
            # Update the pulled arm's head with the newly labelled trial.
            heads[surviving[a_idx].arm_id].partial_fit(
                X_stream[j : j + 1], y_stream[j : j + 1], sfreq
            )
        # Per-family running-reward bookkeeping. The Phase-5 H1 "fbcsp" arm
        # is outside the three single-pipeline families tracked here; its pulls
        # skip this update. That tail is only consumed when
        # include_recent_rewards=True; H1 uses False, so the skip has no
        # downstream effect. Pre-H1 pools never see "fbcsp" — byte-identical.
        family = _ARM_FAMILY_INDEX.get(surviving[a_idx].spatial)
        if family is not None:
            running_rewards[family] += r
            running_counts[family] += 1
        budget_trace[j] = policy.budget_remaining

    # Trim to used portion.
    arm_pulls = arm_pulls[:stop_j]
    rewards = rewards[:stop_j]
    budget_trace = budget_trace[:stop_j]
    per_arm_rewards_used = per_arm_rewards[:stop_j]

    # Hindsight per-round regret: compare to the per-trial best arm reward.
    if stop_j > 0:
        per_round_oracle = per_arm_rewards_used.max(axis=1)
        cumulative_regret = np.cumsum(per_round_oracle - rewards)
    else:
        cumulative_regret = np.zeros(0)

    # ------------------------------------------------------------------
    # Test phase — frozen policy (α=0, no updates)
    # ------------------------------------------------------------------
    test_idx = split.test_idx
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    per_arm_preds_test = np.stack(
        [heads[arm.arm_id].predict(X_test, sfreq) for arm in surviving],
        axis=1,
    )
    base_contexts_test = np.stack(
        [
            context_fn(X_test[j], sfreq=sfreq, recent_arm_rewards=None)
            for j in range(len(test_idx))
        ],
        axis=0,
    )

    # Recent-reward features are frozen at the end-of-stream state.
    recent = running_rewards / np.maximum(running_counts, 1.0)
    y_pred = np.empty(len(test_idx), dtype=y_all.dtype)
    for j in range(len(test_idx)):
        contexts_mat = _build_per_arm_contexts(
            base_contexts_test[j],
            recent,
            n_arms,
            d_ctx,
            include_recent_rewards=include_recent_rewards,
        )
        a_idx = policy.select(contexts=contexts_mat, arm_costs=arm_costs, alpha=0.0)
        if a_idx == INFEASIBLE:
            # Fallback: best calibration-κ arm (first in the sorted surviving list).
            a_idx = 0
        y_pred[j] = per_arm_preds_test[j, a_idx]

    m = compute_metrics(y_test, y_pred)
    return CCBResult(
        subject=data.subject,
        protocol=split.name,
        kappa=float(m.kappa),
        accuracy=float(m.accuracy),
        n_test=int(m.n_trials),
        cumulative_regret=cumulative_regret,
        arm_pulls=arm_pulls,
        budget_trace=budget_trace,
        n_arms_surviving=n_arms,
        y_true=y_test,
        y_pred=y_pred,
    )
