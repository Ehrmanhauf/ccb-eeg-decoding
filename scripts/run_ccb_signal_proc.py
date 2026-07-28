"""Phase-5 §2.3 signal-processing alternatives — combined runner.

Runs the CCB on BCI-IV-2b screening with one of three §2.3 techniques
applied (or with the wavelet arm family enabled), producing one CSV per
technique for direct comparison against the Phase-5 best baseline
(results/ccb_stage1_combined.csv, oplb+window=50: within 0.178 / official 0.172).

Techniques:
- "notch": apply a 50 Hz notch to every trial before any arm runs.
- "ica":   fit FastICA on calibration-block trials, drop high-variance
           artifact components, project all subsequent trials.
- "wavelet": add the wavelet-packet log-energy arm family (54 new arms)
             to the standard CCB pool.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from thesis.ccb.arms import enumerate_arms_2b
from thesis.ccb.oplb import OPLBConfig
from thesis.ccb.preprocessing import apply_ica_cleaning, apply_notch
from thesis.ccb.runner import run_ccb_on_split
from thesis.data import SubjectData, load_bci2b_screening
from thesis.protocols import session_split, within_subject_cv


def _splits(data, protocol: str, n_folds: int, seed: int):
    if protocol == "within":
        yield from list(within_subject_cv(data, n_splits=5, seed=seed))[:n_folds]
    elif protocol == "official":
        yield session_split(data, train_session_idx=0, test_session_idx=1)


def _parse_csv_int(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _apply_preproc(
    data: SubjectData,
    technique: str,
    train_idx: np.ndarray,
    seed: int,
) -> SubjectData:
    """Return a SubjectData with the chosen preprocessing applied to ``data.X``.

    For "ica", the calibration block (train_idx) is used to fit ICA, then the
    full X tensor is projected through. This keeps the no-leakage invariant:
    the test set is *projected* through a model fit on training data only,
    matching the standard ICA-as-preprocessing convention.
    """
    if technique == "notch":
        X_new = apply_notch(data.X, sfreq=data.sfreq, freq=50.0)
    elif technique == "ica":
        X_cal = data.X[train_idx]
        X_new = apply_ica_cleaning(X_cal, data.X, sfreq=data.sfreq, seed=seed)
    elif technique == "wavelet" or technique == "none":
        X_new = data.X
    else:
        raise ValueError(f"unknown technique: {technique!r}")
    return replace(data, X=X_new)


def main(
    technique: str = typer.Option(
        "notch",
        help="One of: 'notch', 'ica', 'wavelet', 'none' (no preprocessing, no wavelet arms).",
    ),
    subjects: str = typer.Option("all", help="Comma IDs or 'all'."),
    seeds: str = typer.Option("0,1,2,3,42", help="Comma seeds."),
    output: Path = typer.Option(
        Path("results/ccb_signal_proc.csv"), help="Output CSV."
    ),
) -> None:
    console = Console()
    if technique not in {"notch", "ica", "wavelet", "none"}:
        raise ValueError(f"unknown technique: {technique!r}")
    subj_ids = list(range(1, 10)) if subjects == "all" else _parse_csv_int(subjects)
    seed_list = _parse_csv_int(seeds)

    console.log(f"Technique: {technique}  |  subjects={subj_ids}  seeds={seed_list}")
    console.log("Loading BCI-IV-2b screening …")
    data_by_subj = {s: load_bci2b_screening(subjects=[s])[0] for s in subj_ids}

    rows: list[dict] = []
    for subject in subj_ids:
        data = data_by_subj[subject]
        # Phase-5 best cell hyperparameters (matches results/ccb_stage1_combined.csv).
        for protocol in ["within", "official"]:
            for seed in seed_list:
                for fold_i, split in enumerate(_splits(data, protocol, n_folds=1, seed=seed)):
                    # Apply preprocessing using *training* indices only as fit data.
                    data_proc = _apply_preproc(data, technique, split.train_idx, seed=seed)
                    arms = enumerate_arms_2b(
                        data.sfreq,
                        include_wavelet_arms=(technique == "wavelet"),
                    )
                    config = OPLBConfig(
                        alpha=0.5, lambda_reg=1.0, budget=float("inf"),
                        window_size=50, discount_gamma=1.0,
                    )
                    res = run_ccb_on_split(
                        data_proc, split,
                        arms=arms, config=config,
                        calibration_frac=0.3, seed=seed,
                        include_recent_rewards=False,
                    )
                    rows.append({
                        "technique": technique,
                        "subject": subject,
                        "protocol": protocol,
                        "fold_name": f"{protocol}_fold{fold_i}",
                        "seed": seed,
                        "kappa": res.kappa,
                        "accuracy": res.accuracy,
                        "n_test": res.n_test,
                        "n_arms_surviving": res.n_arms_surviving,
                        "stream_rounds_run": int(res.arm_pulls.size),
                    })
                    console.log(
                        f"  S{subject} {protocol} seed={seed} → "
                        f"κ={res.kappa:.4f}"
                    )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    df = pd.DataFrame(rows)
    print()
    print(f"=== Mean κ for technique={technique} ===")
    print(df.groupby("protocol")["kappa"].agg(["mean", "std", "count"]).round(4))
    print(f"\nGrand mean (subj-avg): {df['kappa'].mean():.4f}")


if __name__ == "__main__":
    typer.run(main)
