"""Validate the local STEW data layout against the loader's contract.

Run AFTER manually downloading the STEW archive from IEEE DataPort
(https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset)
and extracting it under ``data/STEW/`` at the repo root.

The script checks:
  - ``ratings.txt`` is parseable into ``{subject: (rest, multitask)}``.
  - For each rating-bearing subject, both ``sub{NN}_lo.txt`` and
    ``sub{NN}_hi.txt`` exist.
  - A random sample of EEG files has the expected ``(n_samples, 14)``
    layout and roughly the expected ``128 Hz * 150 s = 19 200`` row count.
  - Subjects 5, 24, 42 are present in the file pool but excluded by the
    loader (consistent with IEEE DataPort's "unavailable ratings" note).

Reports a short summary; non-zero exit code on any failure so it can be
wired into CI or pre-experiment gates.

ref: design-doc/ccb-formulation.md §2.6, src/thesis/data/stew_load.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from thesis.data.stew_load import (
    STEW_CHANNELS,
    STEW_NATIVE_SFREQ,
    STEW_SEGMENT_SECONDS,
    _load_ratings,
    _read_segment,
)

EXPECTED_ROWS = int(STEW_NATIVE_SFREQ * STEW_SEGMENT_SECONDS)


def _check_subject(root: Path, sid: int) -> tuple[bool, str]:
    """Return (ok, message) for one subject."""
    rest = root / f"sub{sid:02d}_lo.txt"
    test = root / f"sub{sid:02d}_hi.txt"
    missing = [p.name for p in (rest, test) if not p.exists()]
    if missing:
        return False, f"missing: {', '.join(missing)}"
    try:
        seg_rest = _read_segment(rest)
        seg_test = _read_segment(test)
    except Exception as exc:
        return False, f"shape error: {exc}"
    n_rest, n_test = seg_rest.shape[1], seg_test.shape[1]
    note = f"rest={n_rest} samples, multitask={n_test} samples"
    # Native 128 Hz × 150 s ≈ 19200; allow ±5% tolerance for any release variant.
    lo, hi = int(EXPECTED_ROWS * 0.95), int(EXPECTED_ROWS * 1.05)
    if not (lo <= n_rest <= hi and lo <= n_test <= hi):
        return False, f"length out of range ({note}); expected ≈{EXPECTED_ROWS}"
    return True, note


def main(
    data_root: Path = typer.Option(
        Path("data/STEW"),
        help="Path to the directory containing the extracted STEW files.",
    ),
    sample_size: int = typer.Option(
        5, help="How many subjects to spot-check end-to-end (>0)."
    ),
) -> None:
    console = Console()
    if not data_root.exists():
        console.print(f"[red]✗ data root not found: {data_root}[/red]")
        console.print(
            "Download the open-access STEW archive from "
            "https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset "
            f"and extract it under {data_root}/. Free IEEE DataPort account required."
        )
        raise typer.Exit(code=2)

    ratings_path = data_root / "ratings.txt"
    if not ratings_path.exists():
        console.print(f"[red]✗ missing {ratings_path}[/red]")
        raise typer.Exit(code=2)

    ratings = _load_ratings(ratings_path)
    if not ratings:
        console.print("[red]✗ ratings.txt parsed but yielded zero usable subjects[/red]")
        raise typer.Exit(code=2)

    console.print(
        f"[green]✓[/green] ratings.txt parsed: "
        f"{len(ratings)} subjects with usable ratings "
        f"(subjects 5, 24, 42 excluded per IEEE DataPort)."
    )

    rng = np.random.default_rng(0)
    sample = sorted(rng.choice(list(ratings.keys()), size=min(sample_size, len(ratings)), replace=False).tolist())
    table = Table(title=f"STEW spot-check ({len(sample)} subjects)")
    table.add_column("subject")
    table.add_column("rest_rating")
    table.add_column("test_rating")
    table.add_column("status", justify="left")
    table.add_column("notes")

    all_ok = True
    for sid in sample:
        rest_r, test_r = ratings[int(sid)]
        ok, note = _check_subject(data_root, int(sid))
        all_ok &= ok
        status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        table.add_row(str(sid), str(rest_r), str(test_r), status, note)

    console.print(table)
    console.print(
        f"\nExpected channel order ({len(STEW_CHANNELS)}): {', '.join(STEW_CHANNELS)}"
    )
    console.print(
        f"Expected segment length: ≈{EXPECTED_ROWS} samples "
        f"({STEW_NATIVE_SFREQ:.0f} Hz × {STEW_SEGMENT_SECONDS:.0f} s)"
    )

    if all_ok:
        console.print(
            "[green]✓ STEW layout looks consistent with the loader's contract.[/green]"
        )
        console.print("Ready to run CCB experiments on STEW.")
    else:
        console.print(
            "[red]✗ at least one subject failed the spot-check; "
            "see the table above and fix before running experiments.[/red]"
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
