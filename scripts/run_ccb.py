"""CCB evaluation runner — mirrors scripts/run_fbcsp_baseline.py.

Runs the CCB on BCI-IV-2b screening for a cross product of
``(subjects × protocols × budget_fracs × alphas × arm_pools × calibration_fracs
× include_recent_rewards × per_round_caps × seeds)``
and writes per-row results to a CSV. Supports sweep use cases by passing
comma-separated values for any hyperparameter.

Examples::

    # Main baseline: all 9 subjects, both protocols, default hyperparameters,
    # 5-fold within-subject + 1 official session split. 54 rows.
    PYTHONPATH=src .venv/bin/python scripts/run_ccb.py

    # Budget-frac sensitivity sweep (fold 0 only, to keep runtime bounded).
    PYTHONPATH=src .venv/bin/python scripts/run_ccb.py \\
        --budget-fracs 0.25,0.5,1.0,2.0 --n-folds 1 \\
        --output results/ccb_sens_budget.csv

    # Alpha sensitivity sweep.
    PYTHONPATH=src .venv/bin/python scripts/run_ccb.py \\
        --alphas 0.1,0.5,1.0,2.0,5.0 --n-folds 1 \\
        --output results/ccb_sens_alpha.csv

    # Phase-4 context ablation (d=18 vs d=15).
    PYTHONPATH=src .venv/bin/python scripts/run_ccb.py \\
        --include-recent-rewards true,false --n-folds 1 --seeds 0,1,2,3,42 \\
        --output results/ccb_sens_context.csv

    # Phase-4 per-round cost cap (binding the knapsack).
    PYTHONPATH=src .venv/bin/python scripts/run_ccb.py \\
        --per-round-caps inf,4,3,2 --n-folds 1 --seeds 0,1,2,3,42 \\
        --output results/ccb_sens_perround.csv

No 2a data is loaded here. 2a numbers live in ``results/fbcsp_baseline.csv``
and are only read by ``scripts/summarize_ccb.py`` for the gap analysis.
"""

from __future__ import annotations

import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.progress import track

from thesis.ccb.arms import enumerate_arms_2b
from thesis.ccb.oplb import OPLBConfig
from thesis.ccb.policies import (
    EpsilonGreedyPolicy,
    FixedArmPolicy,
    LinTSPolicy,
    make_unconstrained_linucb,
)
from thesis.ccb.runner import PolicyFactory, run_ccb_on_split
from thesis.data import SubjectData, load_bci2b_screening
from thesis.protocols import session_split, within_subject_cv

_VALID_POLICIES = ("oplb", "fixed", "eps_greedy", "unconstrained", "ts")


def _build_policy_factory(
    policy_name: str,
    *,
    epsilon: float,
    rng_seed: int,
    ts_prior_scale: float = 1.0,
) -> PolicyFactory | None:
    """Resolve a CLI policy name to a factory. ``None`` → runner default (OPLB)."""
    if policy_name == "oplb":
        return None  # runner instantiates OPLB itself
    if policy_name == "fixed":
        return lambda d, n, c: FixedArmPolicy(d_psi=d, n_arms=n, config=c, fixed_arm_idx=0)
    if policy_name == "eps_greedy":
        return lambda d, n, c: EpsilonGreedyPolicy(
            d_psi=d, n_arms=n, config=c, epsilon=epsilon, rng_seed=rng_seed
        )
    if policy_name == "unconstrained":
        return lambda d, n, c: make_unconstrained_linucb(d, n, c)
    if policy_name == "ts":
        return lambda d, n, c: LinTSPolicy(
            d_psi=d, n_arms=n, config=c, prior_scale=ts_prior_scale, rng_seed=rng_seed
        )
    raise ValueError(f"unknown policy {policy_name!r}; choose from {_VALID_POLICIES}")


mne.set_log_level("WARNING")


def _parse_csv_int(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_csv_float(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _parse_csv_str(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_csv_bool(value: str) -> list[bool]:
    out: list[bool] = []
    for v in value.split(","):
        v_s = v.strip().lower()
        if not v_s:
            continue
        if v_s in ("true", "1", "yes", "y"):
            out.append(True)
        elif v_s in ("false", "0", "no", "n"):
            out.append(False)
        else:
            raise ValueError(f"Cannot parse {v!r} as bool; use true/false.")
    return out


def _parse_csv_cap(value: str) -> list[float | None]:
    """Parse a comma-separated list of per-round caps.

    Accepts numbers (``"4"``, ``"3.5"``) and the sentinels ``"inf"`` / ``"none"``
    — both map to ``None`` (no per-round cap, the CCB default).
    """
    out: list[float | None] = []
    for v in value.split(","):
        v_s = v.strip().lower()
        if not v_s:
            continue
        if v_s in ("inf", "none", "null"):
            out.append(None)
        else:
            out.append(float(v_s))
    return out


def _build_arm_pool(
    data: SubjectData,
    arm_pool: str,
    *,
    include_fbcsp_arm: bool = False,
    include_riemann_arms: bool = False,
):
    arms = enumerate_arms_2b(
        data.sfreq,
        include_fbcsp_arm=include_fbcsp_arm,
        include_riemann_arms=include_riemann_arms,
    )
    if arm_pool == "full":
        # 'full' disables the κ<0.05 filter but still caps at the base grid.
        return arms, 0.0, len(arms)  # min_kappa=0 → keep everything that fits
    if arm_pool == "pruned":
        return arms, 0.05, 100
    raise ValueError(f"arm_pool must be 'pruned' or 'full'; got {arm_pool!r}")


def _splits_for_subject(data: SubjectData, protocol: str, n_folds: int, seed: int):
    if protocol == "within":
        yield from list(within_subject_cv(data, n_splits=5, seed=seed))[:n_folds]
    elif protocol == "official":
        yield session_split(data, train_session_idx=0, test_session_idx=1)
    else:
        raise ValueError(f"protocol must be 'within' or 'official'; got {protocol!r}")


def main(
    subjects: str = typer.Option("all", help="Comma-separated subject IDs, or 'all'."),
    protocols: str = typer.Option("within,official", help="'within', 'official', or both."),
    budget_fracs: str = typer.Option("1.0", help="Comma-separated budget_frac values."),
    alphas: str = typer.Option("1.0", help="Comma-separated OPLB alpha values."),
    arm_pools: str = typer.Option("pruned", help="Comma-separated: 'pruned' and/or 'full'."),
    calibration_fracs: str = typer.Option("0.3", help="Comma-separated calibration fractions."),
    include_recent_rewards: str = typer.Option(
        "true",
        help="Comma-separated booleans controlling the 3-dim recent-reward tail "
        "in the context (true → d_ctx=18, false → d_ctx=15).",
    ),
    per_round_caps: str = typer.Option(
        "inf",
        help="Comma-separated per-round cost caps. Use 'inf' or 'none' for no cap; "
        "numeric values bind arms whose cost exceeds the cap.",
    ),
    policies: str = typer.Option(
        "oplb",
        help="Comma-separated policies to evaluate. Choose from "
        "'oplb' (default), 'fixed' (top-κ calibration arm), 'eps_greedy' "
        "(random exploration at rate --epsilons), 'unconstrained' "
        "(OPLB with budget stripped). See design-doc §8.4.",
    ),
    epsilons: str = typer.Option(
        "0.1",
        help="Comma-separated ε values for eps_greedy. Ignored for other policies.",
    ),
    ts_prior_scale: float = typer.Option(
        1.0,
        help="Prior scale v for Thompson Sampling (θ̃ ~ N(θ̂, v²·A⁻¹)). "
        "Ignored for non-TS policies. Default 1.0 matches Agrawal & Goyal 2013.",
    ),
    online_heads: bool = typer.Option(
        False,
        help="If set, the pulled arm's LDA head is updated via partial_fit "
        "on each trial. Phase-5 Stage-1 fix for session-gap drift. Default "
        "False (Phase-4 frozen-head behaviour).",
    ),
    include_fbcsp_arm: bool = typer.Option(
        False,
        help="Phase-5 H1 hybrid: add a full 9-band FBCSP + shrinkage-LDA "
        "pipeline as a single (expensive, strong) arm in the CCB pool. "
        "Default False (Phase-4 behaviour; all existing result CSVs were "
        "produced with this off).",
    ),
    include_riemann_arms: bool = typer.Option(
        False,
        help="Phase-5 §2.3: add 54 Riemannian-tangent arms (9 bands × 2 "
        "spatial filters × 3 windows) to the CCB pool. ref: "
        "barachant2012riemann. Default False (Phase-4 behaviour).",
    ),
    window_sizes: str = typer.Option(
        "0",
        help="Comma-separated sliding-window sizes for non-stationary OPLB / TS. "
        "Use 0 for no window (stationary). Numeric values > 0 make A, b "
        "aggregate only the last N (psi, r) pairs.",
    ),
    discount_gammas: str = typer.Option(
        "1.0",
        help="Comma-separated discount factors for non-stationary OPLB / TS. "
        "Use 1.0 for no discount (stationary). Values in (0, 1) down-weight "
        "past updates exponentially.",
    ),
    seeds: str = typer.Option(
        "42",
        help="Comma-separated seeds for calibration/stream/fold shuffles.",
    ),
    n_folds: int = typer.Option(5, help="Within-subject CV folds to run (max 5)."),
    output: Path = typer.Option(
        Path("results/ccb_baseline.csv"),
        help="CSV path for per-row results.",
    ),
) -> None:
    console = Console()
    subj_ids = list(range(1, 10)) if subjects == "all" else _parse_csv_int(subjects)
    prot_list = _parse_csv_str(protocols)
    bf_list = _parse_csv_float(budget_fracs)
    alpha_list = _parse_csv_float(alphas)
    pool_list = _parse_csv_str(arm_pools)
    cf_list = _parse_csv_float(calibration_fracs)
    irr_list = _parse_csv_bool(include_recent_rewards)
    cap_list = _parse_csv_cap(per_round_caps)
    policy_list = _parse_csv_str(policies)
    for p in policy_list:
        if p not in _VALID_POLICIES:
            raise ValueError(f"unknown policy {p!r}; choose from {_VALID_POLICIES}")
    eps_list = _parse_csv_float(epsilons) if "eps_greedy" in policy_list else [0.1]
    window_list: list[int | None] = [
        (None if int(v) == 0 else int(v)) for v in _parse_csv_int(window_sizes)
    ]
    gamma_list = _parse_csv_float(discount_gammas)
    seed_list = _parse_csv_int(seeds)

    console.log(
        f"subjects={subj_ids}  protocols={prot_list}  budget_fracs={bf_list}  "
        f"alphas={alpha_list}  arm_pools={pool_list}  calibration_fracs={cf_list}  "
        f"include_recent_rewards={irr_list}  per_round_caps={cap_list}  "
        f"policies={policy_list}  epsilons={eps_list}  online_heads={online_heads}  "
        f"window_sizes={window_list}  discount_gammas={gamma_list}  "
        f"seeds={seed_list}  n_folds={n_folds}"
    )

    console.log("Loading BCI-IV-2b screening for all requested subjects …")
    data_by_subj = {s: load_bci2b_screening(subjects=[s])[0] for s in subj_ids}
    console.log(f"  → {len(data_by_subj)} subjects loaded.")

    # ε is only meaningful when policy == "eps_greedy"; for other policies we
    # collapse the ε axis to a single sentinel so the sweep doesn't explode.
    def _eps_for(policy_name: str) -> list[float]:
        return eps_list if policy_name == "eps_greedy" else [float("nan")]

    cells = [
        (subj, prot, bf, alpha, pool, cf, irr, cap, pol, eps, win, gamma, seed)
        for subj in subj_ids
        for prot in prot_list
        for bf in bf_list
        for alpha in alpha_list
        for pool in pool_list
        for cf in cf_list
        for irr in irr_list
        for cap in cap_list
        for pol in policy_list
        for eps in _eps_for(pol)
        for win in window_list
        for gamma in gamma_list
        for seed in seed_list
    ]
    console.log(f"Total hyperparameter cells: {len(cells)}")

    rows: list[dict] = []
    t_global = time.perf_counter()

    for (
        subj,
        protocol,
        bf,
        alpha,
        arm_pool,
        cal_frac,
        irr,
        cap,
        pol,
        eps,
        win,
        gamma,
        seed,
    ) in track(cells, description="CCB sweep", console=console):
        data = data_by_subj[subj]
        arms, min_kappa, max_arms = _build_arm_pool(
            data,
            arm_pool,
            include_fbcsp_arm=include_fbcsp_arm,
            include_riemann_arms=include_riemann_arms,
        )
        factory = _build_policy_factory(
            pol,
            epsilon=0.0 if not np.isfinite(eps) else eps,
            rng_seed=seed,
            ts_prior_scale=ts_prior_scale,
        )
        for split in _splits_for_subject(data, protocol, n_folds=n_folds, seed=seed):
            config = OPLBConfig(
                alpha=alpha,
                lambda_reg=1.0,
                per_round_cap=cap,
                window_size=win,
                discount_gamma=gamma,
            )
            result = run_ccb_on_split(
                data,
                split,
                arms=arms,
                config=config,
                calibration_frac=cal_frac,
                min_kappa=min_kappa,
                max_arms=max_arms,
                budget_frac=bf,
                seed=seed,
                include_recent_rewards=irr,
                policy_factory=factory,
                online_heads=online_heads,
            )
            final_regret = (
                float(result.cumulative_regret[-1]) if result.cumulative_regret.size else 0.0
            )
            budget_remaining = (
                float(result.budget_trace[-1]) if result.budget_trace.size else float("nan")
            )
            rows.append(
                {
                    "dataset": "BCI-IV-2b",
                    "subject": subj,
                    "protocol": protocol,
                    "fold_name": result.protocol,
                    "seed": seed,
                    "budget_frac": bf,
                    "alpha": alpha,
                    "arm_pool": arm_pool,
                    "calibration_frac": cal_frac,
                    "include_recent_rewards": bool(irr),
                    "per_round_cap": "inf" if cap is None else cap,
                    "policy": pol,
                    "epsilon": eps,
                    "online_heads": bool(online_heads),
                    "window_size": "inf" if win is None else win,
                    "discount_gamma": gamma,
                    "include_fbcsp_arm": bool(include_fbcsp_arm),
                    "include_riemann_arms": bool(include_riemann_arms),
                    "kappa": round(result.kappa, 4),
                    "accuracy": round(result.accuracy, 4),
                    "n_test": result.n_test,
                    "n_arms_surviving": result.n_arms_surviving,
                    "stream_rounds_run": int(result.arm_pulls.size),
                    "final_regret": round(final_regret, 2),
                    "budget_remaining": round(budget_remaining, 2),
                }
            )

    elapsed = time.perf_counter() - t_global
    console.log(f"Sweep completed in {elapsed:.1f} s  ({len(rows)} result rows).")

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    console.log(f"Saved per-row CSV → {output}")

    # Brief console summary by (protocol, arm_pool): mean κ across subjects.
    console.rule("Mean κ by protocol × arm_pool (over all cells × folds)")
    pivot = df.groupby(["protocol", "arm_pool"])["kappa"].agg(["mean", "std", "count"]).round(3)
    console.print(pivot)


if __name__ == "__main__":
    typer.run(main)
