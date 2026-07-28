"""Run CCB on MOABB-backed datasets (Phase-5 §2.2, §2.4, §2.5).

Adapts :func:`thesis.ccb.runner.run_ccb_on_split` to non-2a/2b datasets via
the generic context + generic arm enumerator. The only dataset-specific
logic is the loader; everything else is shared with the 2a/2b paths.

Usage examples:
    # PhysioNet MI, 5 subjects, 5 seeds, both protocols
    python scripts/run_ccb_moabb.py --dataset physionet --subjects 1,2,3,4,5
    # BNCI2015_004 mental tasks (right_hand vs subtraction), 5 subjects
    python scripts/run_ccb_moabb.py --dataset bnci2015_004 --subjects 1,2,3,4,5
    # Cho2017 MI L/R hand, full 64-channel montage
    python scripts/run_ccb_moabb.py --dataset cho2017 --subjects 1,2,3,4,5
    # Cho2017 subsetted to C3/Cz/C4 (deployment-style 3-channel cap)
    python scripts/run_ccb_moabb.py --dataset cho2017_3ch --subjects 1,2,3,4,5
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from thesis.ccb.arms import enumerate_arms_generic
from thesis.ccb.oplb import OPLBConfig
from thesis.ccb.runner import run_ccb_on_split
from thesis.data.moabb_load import (
    load_bnci2015_004,
    load_cho2017,
    load_physionet_mi,
)
from thesis.protocols import within_subject_cv


def _load_with_retry(
    loader, subject: int, *, console: Console, max_retries: int = 3, base_delay_s: float = 30.0
):
    """Call ``loader([subject])`` with exponential-backoff retries.

    Wraps MOABB's network-bound downloads so a transient ChunkedEncodingError
    (or any network exception) does not abort an N-subject sweep. Returns the
    SubjectData on success, or None if the loader returned an empty list (the
    Cho2017 loader silently drops s29/s33) or all retries failed.
    """
    for attempt in range(1, max_retries + 1):
        try:
            datas = loader([subject])
            if not datas:
                console.log(f"  S{subject} loader returned [] (excluded subject); skipping")
                return None
            return datas[0]
        except Exception as exc:  # noqa: BLE001 — broad: MOABB raises many network types
            if attempt == max_retries:
                console.log(
                    f"  [red]S{subject} load failed after {max_retries} attempts: {exc!r}[/red]"
                )
                return None
            wait = base_delay_s * (2 ** (attempt - 1))
            console.log(
                f"  S{subject} load attempt {attempt}/{max_retries} failed: "
                f"{type(exc).__name__}; retrying in {wait:.0f}s"
            )
            time.sleep(wait)
    return None  # unreachable


def _splits(data, protocol: str, n_folds: int, seed: int):
    if protocol == "within":
        yield from list(within_subject_cv(data, n_splits=5, seed=seed))[:n_folds]
    else:
        # MOABB datasets often lack a clean two-session split. Use within for now.
        raise ValueError("Only 'within' protocol is supported for MOABB datasets.")


def _parse_csv_int(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(
    dataset: str = typer.Option(
        "physionet",
        help=(
            "Dataset key: 'physionet' | 'bnci2015_004' | 'cho2017' (full 64ch) "
            "| 'cho2017_3ch' (C3/Cz/C4 monopolar subset)."
        ),
    ),
    subjects: str = typer.Option("1,2,3,4,5", help="Comma-separated subject IDs."),
    seeds: str = typer.Option("0,1,2,3,42", help="Comma-separated seeds."),
    n_folds: int = typer.Option(1, help="Within-subject CV folds."),
    alpha: float = typer.Option(0.5, help="OPLB exploration α."),
    calibration_frac: float = typer.Option(0.3, help="Calibration fraction."),
    window_size: int = typer.Option(50, help="Sliding window (0 → stationary)."),
    output: Path = typer.Option(
        Path("results/ccb_moabb.csv"), help="Output CSV."
    ),
) -> None:
    console = Console()
    subj_ids = _parse_csv_int(subjects)
    seed_list = _parse_csv_int(seeds)
    win = None if window_size == 0 else window_size

    if dataset == "physionet":
        loader = load_physionet_mi
        ds_label = "PhysioNet-MI"
    elif dataset == "bnci2015_004":
        loader = load_bnci2015_004
        ds_label = "BNCI2015-004"
    elif dataset == "cho2017":
        loader = lambda subj: load_cho2017(subj, channels="full")
        ds_label = "Cho2017-full"
    elif dataset == "cho2017_3ch":
        loader = lambda subj: load_cho2017(subj, channels="c3_cz_c4")
        ds_label = "Cho2017-3ch"
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    console.log(f"Running CCB on {ds_label} for subjects {subj_ids} …")
    console.log(
        "Per-subject lazy load with retry-on-network-failure; "
        "incremental CSV writes after every subject."
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for subject in subj_ids:
        data = _load_with_retry(loader, subject, console=console)
        if data is None:
            continue
        n_ch = data.X.shape[1]
        arms = enumerate_arms_generic(n_ch)
        console.log(
            f"S{subject}: {len(arms)} arms (n_ch={n_ch}, n_trials={data.n_trials})"
        )
        for seed in seed_list:
            for fold_i, split in enumerate(_splits(data, "within", n_folds, seed)):
                config = OPLBConfig(
                    alpha=alpha, lambda_reg=1.0, budget=float("inf"),
                    window_size=win, discount_gamma=1.0,
                )
                try:
                    res = run_ccb_on_split(
                        data, split, arms=arms, config=config,
                        calibration_frac=calibration_frac, seed=seed,
                        include_recent_rewards=False,
                    )
                except Exception as e:
                    console.log(f"  S{subject} seed={seed} fold={fold_i} FAILED: {e}")
                    continue
                rows.append({
                    "dataset": ds_label, "subject": subject, "protocol": "within",
                    "fold_name": f"within_fold{fold_i}", "seed": seed,
                    "alpha": alpha, "calibration_frac": calibration_frac,
                    "window_size": "inf" if win is None else win,
                    "kappa": res.kappa, "accuracy": res.accuracy,
                    "n_test": res.n_test, "n_arms_surviving": res.n_arms_surviving,
                    "stream_rounds_run": int(res.arm_pulls.size),
                    "n_channels": n_ch, "n_trials_total": data.n_trials,
                })
                console.log(
                    f"  S{subject} seed={seed} fold={fold_i} → "
                    f"κ={res.kappa:.4f} acc={res.accuracy:.4f}"
                )
        # Incremental checkpoint after each subject completes — keeps
        # partial results recoverable if the sweep is interrupted later.
        pd.DataFrame(rows).to_csv(output, index=False)

    pd.DataFrame(rows).to_csv(output, index=False)
    df = pd.DataFrame(rows)
    print()
    print(f"=== Mean κ for {ds_label} (avg over subjects × seeds) ===")
    print(df.groupby("subject")["kappa"].agg(["mean", "std", "count"]).round(4))
    print(f"\nGRAND MEAN: κ = {df['kappa'].mean():.4f}  (n = {len(df)})")


if __name__ == "__main__":
    typer.run(main)
