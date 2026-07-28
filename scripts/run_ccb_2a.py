"""Run the CCB on BCI Competition IV-2a (22-channel benchmark dataset).

Phase-5 §2.1 sub-experiment. Asks a separate research question from the
3-channel thesis pipeline: how does the CCB perform when the channel budget
is relaxed? This is *not* a comparison with 2b (no-leakage invariant for the
3-ch thesis intact); it characterises adaptive-policy behaviour at higher
electrode density independently.

The runner's `allow_22ch_research_question=True` flag is required to bypass
the no-leakage guard (which still defaults to OFF for every other code path,
preserving the 3-ch thesis pipeline byte-for-byte).

Headline comparator: the 22-ch FBCSP+sLDA baseline already in
`results/fbcsp_baseline.csv` (κ_within = 0.532, κ_official = 0.468).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from thesis.ccb.arms import enumerate_arms_2a
from thesis.ccb.oplb import OPLBConfig
from thesis.ccb.runner import run_ccb_on_split
from thesis.data import load_bci2a
from thesis.protocols import session_split, within_subject_cv


def _splits(data, protocol: str, n_folds: int, seed: int):
    if protocol == "within":
        yield from list(within_subject_cv(data, n_splits=5, seed=seed))[:n_folds]
    elif protocol == "official":
        yield session_split(data, train_session_idx=0, test_session_idx=1)
    else:
        raise ValueError(f"unknown protocol: {protocol!r}")


def _parse_csv_int(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_str(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(
    subjects: str = typer.Option("all", help="Comma-separated subject IDs (1..9), or 'all'."),
    protocols: str = typer.Option("within,official", help="'within', 'official', or both."),
    seeds: str = typer.Option("0,1,2,3,42", help="Comma-separated seeds."),
    n_folds: int = typer.Option(1, help="Within-subject CV folds (max 5)."),
    alpha: float = typer.Option(0.5, help="OPLB exploration parameter."),
    calibration_frac: float = typer.Option(0.3, help="Arm-head calibration fraction."),
    window_size: int = typer.Option(
        50,
        help="OPLB sliding window size (0 disables → stationary). Phase-5 best for 2b.",
    ),
    include_riemann_arms: bool = typer.Option(
        False,
        help="Add Riemannian-tangent arms (9 bands × 1 spatial × 3 windows = 27 added).",
    ),
    output: Path = typer.Option(
        Path("results/ccb_2a.csv"),
        help="CSV path for per-row results.",
    ),
) -> None:
    """Run CCB on BCI-IV-2a using the Phase-5 best-cell hyperparameters."""
    console = Console()
    subj_ids = list(range(1, 10)) if subjects == "all" else _parse_csv_int(subjects)
    prot_list = _parse_csv_str(protocols)
    seed_list = _parse_csv_int(seeds)
    win = None if window_size == 0 else window_size

    console.log(
        f"subjects={subj_ids}  protocols={prot_list}  seeds={seed_list}  "
        f"alpha={alpha}  cal={calibration_frac}  window={win}  "
        f"include_riemann_arms={include_riemann_arms}"
    )

    console.log("Loading BCI-IV-2a (22-channel) for all requested subjects …")
    data_by_subj = {s: load_bci2a(subjects=[s])[0] for s in subj_ids}
    console.log(f"  → {len(data_by_subj)} subjects loaded.")

    rows: list[dict] = []
    for subject in subj_ids:
        data = data_by_subj[subject]
        arms = enumerate_arms_2a(data.sfreq, include_riemann_arms=include_riemann_arms)
        console.log(
            f"Subject {subject}: {len(arms)} arms enumerated (n_components=4)."
        )
        for protocol in prot_list:
            for seed in seed_list:
                for fold_i, split in enumerate(_splits(data, protocol, n_folds, seed)):
                    config = OPLBConfig(
                        alpha=alpha,
                        lambda_reg=1.0,
                        budget=float("inf"),  # injected by runner
                        window_size=win,
                        discount_gamma=1.0,
                    )
                    res = run_ccb_on_split(
                        data,
                        split,
                        arms=arms,
                        config=config,
                        calibration_frac=calibration_frac,
                        seed=seed,
                        include_recent_rewards=False,  # Phase-5 best
                        allow_22ch_research_question=True,  # §2.1 opt-in
                    )
                    rows.append(
                        {
                            "dataset": "BCI-IV-2a",
                            "subject": subject,
                            "protocol": protocol,
                            "fold_name": f"{protocol}_fold{fold_i}",
                            "seed": seed,
                            "alpha": alpha,
                            "calibration_frac": calibration_frac,
                            "window_size": "inf" if win is None else win,
                            "include_riemann_arms": bool(include_riemann_arms),
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
                    console.log(
                        f"  S{subject} {protocol} seed={seed} fold={fold_i} "
                        f"→ κ={res.kappa:.4f} acc={res.accuracy:.4f} "
                        f"arms_surv={res.n_arms_surviving}"
                    )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    console.log(f"Saved {len(rows)} rows → {output}")

    # Quick aggregate.
    df = pd.DataFrame(rows)
    print()
    print("=== Mean κ by protocol (avg over subjects × seeds × folds) ===")
    print(df.groupby("protocol")["kappa"].agg(["mean", "std", "count"]).round(4))


if __name__ == "__main__":
    typer.run(main)
