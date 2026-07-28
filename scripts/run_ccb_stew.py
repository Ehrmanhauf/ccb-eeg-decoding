"""Run the CCB on STEW (Lim 2018) — the primary CL paradigm dataset.

Research-wave 1 first-pass CCB κ on STEW (Phases D and E of the
model-development plan). Headline target of the locked CL-primary
scope (design-doc/ccb-formulation.md §2.6).

Mirrors ``scripts/run_ccb_2a.py`` and ``scripts/run_ccb_moabb.py``
for CLI ergonomics and output schema. The ``--use-workload-context``
flag (default ON) routes through ``compute_context_workload`` via
``run_ccb_on_split``'s ``workload_channel_roles`` parameter; pass
``--no-use-workload-context`` to fall back to the generic n-channel
MI-derived context for ablation comparison.

Examples::

    # Default: all 45 usable subjects, both protocols, 5 seeds.
    PYTHONPATH=src .venv/bin/python scripts/run_ccb_stew.py

    # Single-subject smoke test.
    PYTHONPATH=src .venv/bin/python scripts/run_ccb_stew.py \\
        --subjects 1 --n-folds 1 --seeds 0 \\
        --output results/ccb_stew_smoke.csv

    # α sensitivity sweep (fold 0 only).
    PYTHONPATH=src .venv/bin/python scripts/run_ccb_stew.py \\
        --alphas 0.1,0.5,1.0,2.0 --n-folds 1 \\
        --output results/ccb_stew_sens_alpha.csv

The STEW loader requires the dataset to be manually placed under
``data/STEW/`` (see ``data/STEW.README.md``); run ``make stew-check``
to validate the local layout before running this script.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from thesis.ccb.arms import enumerate_arms_generic
from thesis.ccb.oplb import OPLBConfig
from thesis.ccb.runner import run_ccb_on_split
from thesis.data.stew_load import (
    STEW_CHANNEL_ROLES,
    STEW_CHANNELS,
    load_stew,
)
from thesis.protocols import within_subject_cv

# STEW_CHANNEL_ROLES now lives in thesis.data.stew_load (single source of
# truth, paralleling thesis.data.wauc_load.WAUC_CHANNEL_ROLES). The
# ``--use-workload-context`` flag passes it through ``run_ccb_on_split``'s
# ``workload_channel_roles`` parameter (Phase E wiring).


def _splits(data, protocol: str, n_folds: int, seed: int):
    if protocol == "within":
        yield from list(within_subject_cv(data, n_splits=5, seed=seed))[:n_folds]
    else:
        # STEW has no "official" competition protocol; only within-subject CV
        # is meaningful here. The two segments per subject (rest, multitask)
        # could be used as a session-leave-out protocol if labels differ
        # across them; that is a deferred secondary protocol.
        raise ValueError(
            f"unknown / unsupported protocol for STEW: {protocol!r}. "
            "Use 'within' (no 'official' protocol exists for STEW)."
        )


def _parse_csv_int(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_float(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main(
    subjects: str = typer.Option(
        "all",
        help="Comma-separated STEW subject IDs (1..48, omitting 5/24/42), or 'all'.",
    ),
    protocols: str = typer.Option(
        "within",
        help="Evaluation protocol. STEW only supports 'within' (within-subject CV).",
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
        help="Trial-window length in seconds. Default 4.0 to match 2a/2b.",
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
    output: Path = typer.Option(
        Path("results/ccb_stew.csv"),
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

    console.log("Loading STEW subjects...")
    subject_data_list = load_stew(
        subjects=subject_filter,
        epoch_seconds=epoch_seconds,
    )
    if not subject_data_list:
        console.print("[red]✗ No usable STEW subjects loaded. Check data/STEW/ and run `make stew-check`.[/red]")
        raise typer.Exit(code=1)
    console.log(f"Loaded {len(subject_data_list)} subjects.")

    n_channels = subject_data_list[0].X.shape[1]
    assert n_channels == len(STEW_CHANNELS), (
        f"unexpected STEW channel count: got {n_channels}, expected {len(STEW_CHANNELS)}"
    )

    rows: list[dict] = []
    for data in subject_data_list:
        arms = enumerate_arms_generic(
            n_channels=n_channels,
            n_components=n_components,
            include_riemann_arms=include_riemann_arms,
        )
        console.log(
            f"Subject {data.subject}: {data.X.shape[0]} epochs, {len(arms)} arms enumerated."
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
                                    res = run_ccb_on_split(
                                        data,
                                        split,
                                        arms=arms,
                                        config=config,
                                        calibration_frac=cal_frac,
                                        seed=seed,
                                        include_recent_rewards=False,
                                        workload_channel_roles=STEW_CHANNEL_ROLES if use_workload_context else None,
                                    )
                                    rows.append(
                                        {
                                            "dataset": "STEW",
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

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    console.log(f"Saved {len(rows)} rows → {output}")

    # Quick aggregate summary.
    df = pd.DataFrame(rows)
    print()
    print(f"=== Mean κ on STEW ({'workload' if use_workload_context else 'generic'} context) ===")
    grouped = df.groupby("protocol")["kappa"].agg(["mean", "std", "count"]).round(4)
    print(grouped)
    print()
    print("=== Reference baselines (within-subject CV, same protocol) ===")
    print("Lim 2018 (SVR + NCA, 3-class):              κ = 0.46  [different protocol]")
    print("Phase C STEW B1 FBCSP+sLDA:                 κ = 0.937 ± 0.105 (n = 225)")
    print("Phase C STEW B2 BandPower+sLDA:             κ = 0.953 ± 0.092 (n = 225)")
    print("Phase D STEW CCB (generic context):         κ = 0.744 ± 0.170 (n = 225)")
    print()
    print("Note: STEW within-CV is segment-leakage-saturated for the fixed pipelines;")
    print("the CCB drops to ≈ 0.74 mainly because of its calibration + exploration")
    print("overhead, not because of context misspecification (per Phase D analysis).")


if __name__ == "__main__":
    typer.run(main)
