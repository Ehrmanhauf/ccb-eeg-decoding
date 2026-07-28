"""Run the CCB on WAUC (Albuquerque et al. 2020) — secondary CL paradigm.

Research-wave 1 first-pass CCB κ on WAUC (Phases D and E of the
model-development plan). Mirrors ``scripts/run_ccb_stew.py`` for CLI
ergonomics and output schema — the two CL CCB scripts are intentionally
parallel so the same downstream summary tooling applies.

The ``--use-workload-context`` flag (default ON) routes through
``compute_context_workload`` via ``run_ccb_on_split``'s
``workload_channel_roles`` parameter; pass
``--no-use-workload-context`` to fall back to the generic n-channel
MI-derived context for ablation comparison.

Examples::

    # Default: all 45 usable WAUC subjects, 1 fold, 5 seeds.
    PYTHONPATH=src .venv/bin/python scripts/run_ccb_wauc.py

    # Single-subject smoke test.
    PYTHONPATH=src .venv/bin/python scripts/run_ccb_wauc.py \\
        --subjects 1 --n-folds 1 --seeds 0 \\
        --output results/ccb_wauc_smoke.csv

    # α sensitivity sweep (fold 0 only).
    PYTHONPATH=src .venv/bin/python scripts/run_ccb_wauc.py \\
        --alphas 0.1,0.5,1.0,2.0 --n-folds 1 \\
        --output results/ccb_wauc_sens_alpha.csv

Prerequisites: ``data/WAUC/process/`` must contain the extracted
ASR-processed archive; run ``make wauc-check`` to validate the layout.
The drop list (subjects 1028 / S23 / S26 — see
``thesis.data.wauc_load``) is applied automatically by ``load_wauc``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from thesis.ccb.arms import enumerate_arms_generic
from thesis.ccb.oplb import OPLBConfig
from thesis.ccb.runner import run_ccb_on_split
from thesis.data.wauc_load import (
    WAUC_CHANNEL_ROLES,
    WAUC_EEG_CHANNELS,
    load_wauc,
)
from thesis.protocols import within_subject_cv

# Phase E wiring closed: the ``--use-workload-context`` flag passes
# ``WAUC_CHANNEL_ROLES`` (`thesis.data.wauc_load`) through
# ``run_ccb_on_split``'s ``workload_channel_roles`` parameter, which
# routes to ``compute_context_workload`` in ``thesis.ccb.context_cl``.


def _splits(data, protocol: str, n_folds: int, seed: int):
    if protocol == "within":
        yield from list(within_subject_cv(data, n_splits=5, seed=seed))[:n_folds]
    else:
        # WAUC has 6 session-condition cells per subject; a session-leave-
        # out protocol could in principle be defined (train on sessions
        # 1–4, test on 5–6, etc.) but is deferred. For Phase D the within-
        # subject CV is the only supported protocol — consistent with
        # `scripts/run_ccb_stew.py` and the Phase C fixed-pipeline runner.
        raise ValueError(
            f"unknown / unsupported protocol for WAUC: {protocol!r}. "
            "Use 'within' (no session-leave-out protocol implemented yet)."
        )


def _parse_csv_int(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_float(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main(
    subjects: str = typer.Option(
        "all",
        help="Comma-separated WAUC filesystem IDs (1..48, omitting 23/26/28), or 'all'.",
    ),
    protocols: str = typer.Option(
        "within",
        help="Evaluation protocol. WAUC currently supports only 'within' (within-subject CV).",
    ),
    seeds: str = typer.Option("0,1,2,3,42", help="Comma-separated seeds."),
    n_folds: int = typer.Option(1, help="Within-subject CV folds (max 5)."),
    alphas: str = typer.Option(
        "0.5",
        help="Comma-separated OPLB α (exploration). Phase-5 best on 2b was 0.5.",
    ),
    calibration_fracs: str = typer.Option(
        "0.3",
        help="Comma-separated calibration fractions. Default 0.3.",
    ),
    window_sizes: str = typer.Option(
        "50",
        help="Comma-separated OPLB sliding-window sizes (0 → stationary).",
    ),
    per_round_caps: str = typer.Option(
        "inf",
        help="Comma-separated per-round cost caps ('inf' for unbounded).",
    ),
    n_components: int = typer.Option(
        4,
        help="CSP components per band (capped to n_channels at fit time).",
    ),
    include_riemann_arms: bool = typer.Option(
        False,
        help="Add 27 Riemannian-on-identity arms (135 total). Default disabled.",
    ),
    epoch_seconds: float = typer.Option(
        4.0,
        help="Trial-window length in seconds. Default 4.0 to match 2a/2b/STEW.",
    ),
    use_workload_context: bool = typer.Option(
        True,
        help=(
            "Use the CL-specific workload context (compute_context_workload, "
            "θ/α/β + frontal-θ + parietal-α + frontal-α asymmetry + engagement). "
            "Default ON. Disable to fall back to the generic n-channel MI-derived "
            "context for ablation comparison against the Phase D first-pass numbers."
        ),
    ),
    reward_mode: str = typer.Option(
        "accuracy",
        help="CCB reward: 'accuracy' (0/1; default, the headline) or 'balanced' "
        "(inverse train-class-frequency weighted — the κ-aligned robustness check).",
    ),
    output: Path = typer.Option(
        Path("results/ccb_wauc.csv"),
        help="Output CSV path.",
    ),
) -> None:
    console = Console()
    subject_filter = None if subjects == "all" else _parse_csv_int(subjects)
    prot_list = [p.strip() for p in protocols.split(",") if p.strip()]
    seed_list = _parse_csv_int(seeds)
    alpha_list = _parse_csv_float(alphas)
    calibration_frac_list = _parse_csv_float(calibration_fracs)
    window_size_list = [int(w) for w in window_sizes.split(",") if w.strip()]
    per_round_cap_list = [
        float("inf") if v.strip().lower() == "inf" else float(v)
        for v in per_round_caps.split(",")
        if v.strip()
    ]

    console.log("Loading WAUC subjects...")
    subject_data_list = load_wauc(
        subjects=subject_filter,
        epoch_seconds=epoch_seconds,
    )
    if not subject_data_list:
        console.print("[red]✗ No usable WAUC subjects loaded. Check data/WAUC/process/ and run `make wauc-check`.[/red]")
        raise typer.Exit(code=1)
    console.log(f"Loaded {len(subject_data_list)} subjects.")

    n_channels = subject_data_list[0].X.shape[1]
    assert n_channels == len(WAUC_EEG_CHANNELS), (
        f"unexpected WAUC channel count: got {n_channels}, expected {len(WAUC_EEG_CHANNELS)}"
    )

    rows: list[dict] = []
    for data in subject_data_list:
        arms = enumerate_arms_generic(
            n_channels=n_channels,
            n_components=n_components,
            include_riemann_arms=include_riemann_arms,
        )
        console.log(
            f"Subject S{data.subject:02d}: {data.X.shape[0]} epochs, {len(arms)} arms enumerated."
        )
        for protocol in prot_list:
            for seed in seed_list:
                for alpha in alpha_list:
                    for cal_frac in calibration_frac_list:
                        for window_size_val in window_size_list:
                            win = window_size_val if window_size_val > 0 else None
                            for per_round_cap in per_round_cap_list:
                                splits = list(_splits(data, protocol, n_folds, seed))
                                for fold_i, split in enumerate(splits):
                                    config = OPLBConfig(
                                        alpha=alpha,
                                        lambda_reg=1.0,
                                        budget=float("inf"),
                                        window_size=win,
                                        discount_gamma=1.0,
                                        per_round_cap=None if per_round_cap == float("inf") else per_round_cap,
                                    )
                                    common_row = {
                                        "dataset": "WAUC",
                                        "subject": data.subject,
                                        "protocol": protocol,
                                        "fold_name": f"{protocol}_fold{fold_i}",
                                        "seed": seed,
                                        "alpha": alpha,
                                        "calibration_frac": cal_frac,
                                        "window_size": "inf" if win is None else win,
                                        "per_round_cap": "inf" if per_round_cap == float("inf") else per_round_cap,
                                        "n_components": n_components,
                                        "include_riemann_arms": bool(include_riemann_arms),
                                        "epoch_seconds": epoch_seconds,
                                        "context": "workload" if use_workload_context else "generic",
                                        "reward_mode": reward_mode,
                                    }
                                    try:
                                        res = run_ccb_on_split(
                                            data,
                                            split,
                                            arms=arms,
                                            config=config,
                                            calibration_frac=cal_frac,
                                            seed=seed,
                                            include_recent_rewards=False,
                                            workload_channel_roles=WAUC_CHANNEL_ROLES if use_workload_context else None,
                                            reward_mode=reward_mode,
                                        )
                                        rows.append(
                                            {
                                                **common_row,
                                                "kappa": res.kappa,
                                                "accuracy": res.accuracy,
                                                "n_test": res.n_test,
                                                "n_arms_surviving": res.n_arms_surviving,
                                                "stream_rounds_run": int(res.arm_pulls.size),
                                                "final_regret": float(res.cumulative_regret[-1])
                                                if res.cumulative_regret.size
                                                else 0.0,
                                            }
                                        )
                                    except Exception as exc:
                                        rows.append(
                                            {
                                                **common_row,
                                                "kappa": float("nan"),
                                                "accuracy": float("nan"),
                                                "n_test": 0,
                                                "n_arms_surviving": 0,
                                                "stream_rounds_run": 0,
                                                "final_regret": 0.0,
                                                "error": str(exc),
                                            }
                                        )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    console.log(f"Saved {len(rows)} rows → {output}")

    df = pd.DataFrame(rows).dropna(subset=["kappa"])
    print()
    print(f"=== Mean κ on WAUC ({'workload' if use_workload_context else 'generic'} context) ===")
    print(df.groupby("protocol")["kappa"].agg(["mean", "std", "count"]).round(4))
    print()
    print("=== Reference baselines (within-subject CV, same protocol) ===")
    print("Phase C WAUC B1 FBCSP+sLDA:                κ = 0.658 ± 0.169 (n = 215)")
    print("Phase C WAUC B2 BandPower+sLDA:            κ = 0.644 ± 0.189 (n = 215)")
    print("Phase D WAUC CCB (generic context):        κ = 0.443 ± 0.202 (n = 215)")


if __name__ == "__main__":
    typer.run(main)
