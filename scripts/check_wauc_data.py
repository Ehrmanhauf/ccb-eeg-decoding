"""Validate the local WAUC data layout against the loader's contract.

Run AFTER extracting ``data/WAUC/process.rar`` (the ASR-processed
release variant of the MuSAE Lab dataset; see ``data/WAUC.README.md``)
in place under ``data/WAUC/process/``. The script checks:

  - ``subjective_ratings_with_labels.csv`` is parseable; the
    ``mw_labels`` column canonicalises to ``low`` / ``high`` and
    ``pw_labels`` to integer 0 / 1 / 2.
  - For a sample of filesystem subjects (``S{NN:02d}``), the
    corresponding ``process/S{NN:02d}/enobio_eeg_asr.csv`` exists and
    contains all 8 EEG channel columns plus the three metadata
    columns ``(fs, info, session_no)``.
  - The ``info`` field takes only the documented values
    (``session``, ``baseline-1``, ``baseline-2``) and ``session_no``
    falls in 1..6.
  - The drop list (``1028``) is correctly handled by the loader.

Exits non-zero on any failure so it can be wired into CI or pre-
experiment gates.

ref: design-doc/ccb-formulation.md §2.7, src/thesis/data/wauc_load.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from thesis.data.wauc_load import (
    WAUC_EEG_CHANNELS,
    WAUC_NATIVE_SFREQ,
    WAUC_SESSIONS,
    _WAUC_BASELINE_INFO_VALUES,
    _WAUC_MISSING_RATINGS,
    _WAUC_SESSION_INFO_VALUES,
    _load_wauc_labels,
    _read_eeg_csv,
    _resolve_channel_columns,
    _subject_eeg_path,
    subject_id_to_partid,
)


def _check_subject(root: Path, sid: int) -> tuple[bool, str]:
    """Return (ok, message) for one filesystem subject's enobio_eeg_asr.csv."""
    eeg_path = _subject_eeg_path(sid, root)
    if not eeg_path.exists():
        return False, f"missing: {eeg_path.relative_to(root)}"
    try:
        df = _read_eeg_csv(eeg_path)
        channels = _resolve_channel_columns(df)
    except Exception as exc:
        return False, f"parse error: {exc}"

    info_values = set(df["info"].astype(str).unique())
    allowed_info = set(_WAUC_SESSION_INFO_VALUES) | set(_WAUC_BASELINE_INFO_VALUES)
    unexpected_info = info_values - allowed_info
    if unexpected_info:
        return False, f"unexpected `info` values: {sorted(unexpected_info)}"

    session_values = set(int(s) for s in df["session_no"].dropna().unique())
    session_only = df.loc[df["info"].astype(str) == "session"]
    session_values_in_session = set(int(s) for s in session_only["session_no"].dropna().unique())
    unexpected_sessions = session_values_in_session - set(WAUC_SESSIONS)
    if unexpected_sessions:
        return False, f"unexpected session_no values among session rows: {sorted(unexpected_sessions)}"

    n_rows = len(df)
    note = (
        f"{n_rows} rows · {len(session_values_in_session)}/6 session_no values · "
        f"channels resolved as `{channels[0]}` style"
    )
    return True, note


def main(
    data_root: Path = typer.Option(
        Path("data/WAUC"),
        help="Path to the directory containing the extracted WAUC files.",
    ),
    sample_size: int = typer.Option(
        5, help="How many filesystem subjects to spot-check end-to-end (>0)."
    ),
) -> None:
    console = Console()
    if not data_root.exists():
        console.print(f"[red]✗ data root not found: {data_root}[/red]")
        console.print(
            "See data/WAUC.README.md for the access + extraction procedure. "
            "The full dataset requires a `brew install unar` (or equivalent) "
            "to unrar process.rar."
        )
        raise typer.Exit(code=2)

    labels_path = data_root / "subjective_ratings_with_labels.csv"
    if not labels_path.exists():
        console.print(f"[red]✗ missing {labels_path}[/red]")
        raise typer.Exit(code=2)

    try:
        labels = _load_wauc_labels(labels_path)
    except Exception as exc:
        console.print(f"[red]✗ labels CSV unparseable: {exc}[/red]")
        raise typer.Exit(code=2)

    n_partids = labels.index.get_level_values("partid").nunique()
    n_records = len(labels)
    console.print(
        f"[green]✓[/green] subjective_ratings_with_labels.csv parsed: "
        f"{n_records} (partid, session_no) records across {n_partids} participants. "
        f"Drop list: {sorted(_WAUC_MISSING_RATINGS)}."
    )

    # Spot-check filesystem subjects.
    available_sids = sorted(
        sid for sid in range(1, 49)
        if subject_id_to_partid(sid) in {int(p) for p in labels.index.get_level_values("partid").unique()}
        and subject_id_to_partid(sid) not in _WAUC_MISSING_RATINGS
    )
    rng = np.random.default_rng(0)
    sample = sorted(
        rng.choice(available_sids, size=min(sample_size, len(available_sids)), replace=False).tolist()
    )

    table = Table(title=f"WAUC spot-check ({len(sample)} subjects)")
    table.add_column("filesystem")
    table.add_column("partid")
    table.add_column("n_session_labels")
    table.add_column("status", justify="left")
    table.add_column("notes")

    all_ok = True
    for sid in sample:
        partid = subject_id_to_partid(int(sid))
        try:
            n_lbl = labels.loc[(partid, slice(None))].shape[0]
        except KeyError:
            n_lbl = 0
        ok, note = _check_subject(data_root, int(sid))
        all_ok &= ok
        status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        table.add_row(f"S{sid:02d}", str(partid), str(n_lbl), status, note)

    console.print(table)
    console.print(
        f"\nExpected EEG channel set ({len(WAUC_EEG_CHANNELS)}): "
        f"{', '.join(WAUC_EEG_CHANNELS)}"
    )
    console.print(
        f"Expected native sfreq: {WAUC_NATIVE_SFREQ:.0f} Hz · "
        f"sessions in {WAUC_SESSIONS} · baseline `info` in "
        f"{_WAUC_BASELINE_INFO_VALUES}"
    )

    if all_ok:
        console.print(
            "[green]✓ WAUC layout looks consistent with the loader's contract.[/green]"
        )
        console.print("Ready to run CCB experiments on WAUC.")
    else:
        console.print(
            "[red]✗ at least one subject failed the spot-check; "
            "see the table above and fix before running experiments. "
            "If the EEG channel columns inside enobio_eeg_asr.csv use a third "
            "naming convention not covered by _resolve_channel_columns, "
            "patch src/thesis/data/wauc_load.py to accept it.[/red]"
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
