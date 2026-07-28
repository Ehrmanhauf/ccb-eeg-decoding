"""FBCSP+LDA baseline over both datasets, both protocols, all subjects.

Produces:
- a per-subject table (dataset × subject × protocol → κ, accuracy, n_test)
- a summary table (dataset × protocol → mean ± std of κ)

Usage:
    uv run python scripts/run_fbcsp_baseline.py
    uv run python scripts/run_fbcsp_baseline.py --subjects 1,3 --output results/fbcsp_quick.csv
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.progress import track

from thesis.baselines import FBCSP
from thesis.data import SubjectData, load_bci2a, load_bci2b_screening
from thesis.matched import matched_session_split, matched_within_cv
from thesis.metrics import compute_metrics

# Silence MNE's per-band CSP covariance-estimation chatter so the baseline
# progress output stays readable. Warnings and errors still surface.
mne.set_log_level("WARNING")


def _eval_within(data: SubjectData, *, n_splits: int = 5) -> tuple[float, float]:
    """Mean κ and accuracy across K-fold CV on the subject's trials."""
    kappas: list[float] = []
    accs: list[float] = []
    for split in matched_within_cv(data, n_splits=n_splits):
        clf = FBCSP(sfreq=data.sfreq)
        clf.fit(data.X[split.train_idx], data.y[split.train_idx])
        y_hat = clf.predict(data.X[split.test_idx])
        m = compute_metrics(data.y[split.test_idx], y_hat)
        kappas.append(m.kappa)
        accs.append(m.accuracy)
    return float(np.mean(kappas)), float(np.mean(accs))


def _eval_official(data: SubjectData) -> tuple[float, float, int]:
    """κ, accuracy, and n_test for session_0 → session_1."""
    split = matched_session_split(data, train_session_idx=0, test_session_idx=1)
    clf = FBCSP(sfreq=data.sfreq)
    clf.fit(data.X[split.train_idx], data.y[split.train_idx])
    y_hat = clf.predict(data.X[split.test_idx])
    m = compute_metrics(data.y[split.test_idx], y_hat)
    return m.kappa, m.accuracy, m.n_trials


def _process_dataset(
    data_list: list[SubjectData], dataset_label: str, *, console: Console
) -> list[dict]:
    rows = []
    for data in track(data_list, description=f"{dataset_label} per-subject", console=console):
        kappa_within, acc_within = _eval_within(data)
        kappa_official, acc_official, n_test_official = _eval_official(data)
        rows.append(
            {
                "dataset": dataset_label,
                "subject": data.subject,
                "kappa_within": kappa_within,
                "acc_within": acc_within,
                "kappa_official": kappa_official,
                "acc_official": acc_official,
                "n_test_official": n_test_official,
                "n_trials_total": data.n_trials,
                "n_channels": data.n_channels,
            }
        )
    return rows


def main(
    subjects: str = typer.Option("all", help="Comma-separated subject IDs, or 'all'"),
    output: Path = typer.Option(
        Path("results/fbcsp_baseline.csv"), help="CSV path for the per-subject table"
    ),
) -> None:
    console = Console()
    subj_list = None if subjects == "all" else [int(s) for s in subjects.split(",")]

    console.log(f"Loading BCI-IV 2a (22ch benchmark) subjects={subj_list or 'all'} …")
    data_2a = load_bci2a(subjects=subj_list)
    console.log(f"  → {len(data_2a)} subjects, trial tensor shape {data_2a[0].X.shape}")

    console.log(f"Loading BCI-IV 2b screening (3ch CCB data) subjects={subj_list or 'all'} …")
    data_2b = load_bci2b_screening(subjects=subj_list)
    console.log(f"  → {len(data_2b)} subjects, trial tensor shape {data_2b[0].X.shape}")

    rows: list[dict] = []
    rows += _process_dataset(data_2a, "BCI-IV-2a", console=console)
    rows += _process_dataset(data_2b, "BCI-IV-2b", console=console)
    df = pd.DataFrame(rows)

    console.rule("Per-subject κ")
    console.print(df)

    console.rule("Summary (mean ± std κ)")
    summary = (
        df.groupby("dataset")[["kappa_within", "kappa_official"]].agg(["mean", "std"]).round(3)
    )
    console.print(summary)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    console.log(f"Saved per-subject table → {output}")


if __name__ == "__main__":
    typer.run(main)
