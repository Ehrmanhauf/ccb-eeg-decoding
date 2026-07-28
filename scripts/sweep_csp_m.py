"""CSP component-count sensitivity sweep for FBCSP baseline.

Runs the full 9-subject FBCSP+shrinkage-LDA baseline at multiple values of
the CSP ``n_components`` hyperparameter (``m``), across both BCI-IV-2a and 2b
and both evaluation protocols (within-subject 5-fold and official
session-split). Aggregates per-(m, dataset, protocol) mean±std κ.

Purpose: close the "CSP component count m" open-justifications item
(design-doc/open-justifications.md). Ang 2012 §3.1.1 uses a fixed m = 2 on
Dataset 2a and m = 1 on Dataset 2b (NOT subject-specific). Our Phase-2
baseline fixes n_components = 4 (Ang's 2a value); this sweep tests
whether that choice generalizes across the 9-subject population for both
datasets before Phase 3 builds the CCB on top.

Constraint: MNE's CSP cannot return more components than input channels.
For 2b (3 channels) we cap m at 3; m=4 is a 2a-only value.

Output:
  results/fbcsp_sensitivity_csp_m.csv  — long-form (m, dataset, protocol, κ_mean, κ_std)
  results/fbcsp_sensitivity_csp_m.md   — Markdown pivot table
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
from thesis.metrics import compute_metrics
from thesis.protocols import session_split, within_subject_cv

mne.set_log_level("ERROR")


def _kappa_within(data: SubjectData, m: int, n_splits: int = 5) -> float:
    ks: list[float] = []
    for split in within_subject_cv(data, n_splits=n_splits):
        clf = FBCSP(sfreq=data.sfreq, n_components=m).fit(
            data.X[split.train_idx], data.y[split.train_idx]
        )
        y_hat = clf.predict(data.X[split.test_idx])
        ks.append(compute_metrics(data.y[split.test_idx], y_hat).kappa)
    return float(np.mean(ks))


def _kappa_official(data: SubjectData, m: int) -> float:
    split = session_split(data, train_session_idx=0, test_session_idx=1)
    clf = FBCSP(sfreq=data.sfreq, n_components=m).fit(
        data.X[split.train_idx], data.y[split.train_idx]
    )
    y_hat = clf.predict(data.X[split.test_idx])
    return float(compute_metrics(data.y[split.test_idx], y_hat).kappa)


def main(
    m_values: str = typer.Option("1,2,3,4", help="Comma-separated m values."),
    output_csv: Path = typer.Option(Path("results/fbcsp_sensitivity_csp_m.csv")),
    output_md: Path = typer.Option(Path("results/fbcsp_sensitivity_csp_m.md")),
) -> None:
    console = Console()
    m_list = sorted({int(m) for m in m_values.split(",") if m.strip()})

    console.log("Loading BCI-IV-2a + 2b screening (all 9 subjects each) …")
    data_2a = load_bci2a()
    data_2b = load_bci2b_screening()

    rows: list[dict] = []
    for m in m_list:
        console.rule(f"m = {m}")
        for dataset_list, dataset_label, max_chan in [
            (data_2a, "BCI-IV-2a", 22),
            (data_2b, "BCI-IV-2b", 3),
        ]:
            if m > max_chan:
                console.log(f"  skip {dataset_label} (m={m} > n_channels={max_chan})")
                continue
            ks_within: list[float] = []
            ks_official: list[float] = []
            for data in track(dataset_list, description=f"{dataset_label} m={m}", console=console):
                ks_within.append(_kappa_within(data, m=m))
                ks_official.append(_kappa_official(data, m=m))
            rows.append(
                {
                    "m": m,
                    "dataset": dataset_label,
                    "protocol": "within",
                    "kappa_mean": float(np.mean(ks_within)),
                    "kappa_std": float(np.std(ks_within)),
                }
            )
            rows.append(
                {
                    "m": m,
                    "dataset": dataset_label,
                    "protocol": "official",
                    "kappa_mean": float(np.mean(ks_official)),
                    "kappa_std": float(np.std(ks_official)),
                }
            )

    df = pd.DataFrame(rows).round(3)
    console.print(df.to_string(index=False))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    console.log(f"Saved long-form CSV → {output_csv}")

    # Render a pivot-style Markdown table: rows = (dataset, protocol), cols = m.
    lines: list[str] = [
        "# FBCSP CSP component-count (m) sensitivity sweep",
        "",
        f"Tested values: m ∈ {{{', '.join(str(m) for m in m_list)}}}. 2b capped at m=3 (only 3 channels).",
        "",
        "| Dataset | Protocol | " + " | ".join(f"m={m}" for m in m_list) + " |",
        "|---|---|" + "|".join("---" for _ in m_list) + "|",
    ]
    for dataset in sorted(df["dataset"].unique()):
        for protocol in ["within", "official"]:
            sub = df[(df["dataset"] == dataset) & (df["protocol"] == protocol)]
            cells = []
            for m in m_list:
                row = sub[sub["m"] == m]
                if row.empty:
                    cells.append("—")
                else:
                    cells.append(
                        f"{row['kappa_mean'].iloc[0]:.3f} ± {row['kappa_std'].iloc[0]:.3f}"
                    )
            lines.append(f"| {dataset} | {protocol} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(
        "Values are mean ± std of per-subject κ over 9 subjects. "
        "Phase-2 baseline used m=2 (see results/fbcsp_baseline.md)."
    )
    output_md.write_text("\n".join(lines) + "\n")
    console.log(f"Saved Markdown summary → {output_md}")


if __name__ == "__main__":
    typer.run(main)
