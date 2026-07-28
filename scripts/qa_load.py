"""Phase 2 QA smoke test: load 2a and 2b for selected subjects, print shape summary.

Usage:
    uv run python scripts/qa_load.py                 # subject 1, both datasets
    uv run python scripts/qa_load.py --subject 3
    uv run python scripts/qa_load.py --subjects 1,3,5
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from thesis.data import SubjectData, load_bci2a, load_bci2b_screening


def _row(data: SubjectData) -> tuple[str, ...]:
    return (
        data.dataset_name,
        str(data.subject),
        str(data.n_trials),
        str(data.n_channels),
        str(data.n_samples),
        f"{data.sfreq:.0f} Hz",
        ",".join(data.sessions),
        str(data.class_balance),
    )


def main(
    subjects: str = typer.Option("1", help="Comma-separated subject IDs, or 'all'"),
) -> None:
    console = Console()
    subj_list = None if subjects == "all" else [int(s) for s in subjects.split(",")]

    console.log(f"Loading BCI-IV 2a (22ch, 2-class) for subjects={subj_list or 'all'} …")
    data_2a = load_bci2a(subjects=subj_list)

    console.log(
        f"Loading BCI-IV 2b screening (3ch, 2-class, first 2 sessions) for subjects={subj_list or 'all'} …"
    )
    data_2b = load_bci2b_screening(subjects=subj_list)

    table = Table(title="Phase 2 data-loader QA")
    for col in (
        "Dataset",
        "Subj",
        "n_trials",
        "n_ch",
        "n_samples",
        "sfreq",
        "sessions",
        "class balance",
    ):
        table.add_column(col)
    for data in data_2a:
        table.add_row(*_row(data))
    for data in data_2b:
        table.add_row(*_row(data))

    console.print(table)


if __name__ == "__main__":
    typer.run(main)
