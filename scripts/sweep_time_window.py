"""Time-window arm-grid sensitivity sweep for CCB on BCI-IV-2b.

Closes the "CCB time-window arm grid" open-justifications item by
comparing the full 3-window grid (currently in production) against three
single-window restrictions.

Cells:

- ``full``: ``_TIME_WINDOWS = ((0.0, 4.0), (0.5, 2.5), (1.0, 3.0))`` — the
  Phase-2/3 default; arm pool offers all three windows to the bandit.
- ``0_4``: only ``(0.0, 4.0)`` — full cue-locked epoch (our 2a/2b loader
  default).
- ``05_25``: only ``(0.5, 2.5)`` — matches Ang 2012 §3.1.1 verbatim.
- ``1_3``: only ``(1.0, 3.0)`` — historically the Ramoser 2000 / Graz
  convention (the Ramoser 2000 attribution is *not* directly verified
  against the paper PDF in this audit; see open-justifications closed
  item).

Each cell is evaluated at the **Phase-5 Stage-1 best-factorial CCB cell**
(``alpha=0.5``, ``calibration_frac=0.3``, ``window_size=50``,
``arm_pool=pruned``, ``include_recent_rewards=False``) so the result
isolates the time-window grid as the only varying axis.

Output: ``results/ccb_sens_time_window.csv`` + ``.md`` aggregate.
"""

from __future__ import annotations

import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.progress import track

from thesis.ccb import arms as arms_mod
from thesis.ccb.arms import enumerate_arms_2b
from thesis.ccb.oplb import OPLBConfig
from thesis.ccb.runner import run_ccb_on_split
from thesis.data.load import load_bci2b_screening
from thesis.protocols import session_split, within_subject_cv

mne.set_log_level("ERROR")


_FULL_WINDOWS: tuple[tuple[float, float], ...] = ((0.0, 4.0), (0.5, 2.5), (1.0, 3.0))

_CELLS: dict[str, tuple[tuple[float, float], ...]] = {
    "full": _FULL_WINDOWS,
    "0_4": ((0.0, 4.0),),
    "05_25": ((0.5, 2.5),),
    "1_3": ((1.0, 3.0),),
}


def _splits_for_subject(data, protocol: str, *, seed: int):
    if protocol == "within":
        return list(within_subject_cv(data, n_splits=5, seed=seed))[:1]
    if protocol == "official":
        return [session_split(data, train_session_idx=0, test_session_idx=1)]
    raise ValueError(f"unknown protocol {protocol!r}")


def main(
    seeds: str = typer.Option(
        "0,1,2,3,42", help="Comma-separated seeds (default = best-factorial set)."
    ),
    output_csv: Path = typer.Option(Path("results/ccb_sens_time_window.csv")),
    output_md: Path = typer.Option(Path("results/ccb_sens_time_window.md")),
) -> None:
    console = Console()
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    subj_ids = list(range(1, 10))

    console.log("Loading BCI-IV-2b screening (all 9 subjects) …")
    data_by_subj = {s: load_bci2b_screening(subjects=[s])[0] for s in subj_ids}
    console.log(f"  → {len(data_by_subj)} subjects.")

    saved_windows = arms_mod._TIME_WINDOWS
    rows: list[dict] = []
    t0 = time.perf_counter()
    try:
        for cell_name, windows in _CELLS.items():
            console.rule(f"cell={cell_name}  windows={windows}")
            arms_mod._TIME_WINDOWS = windows  # type: ignore[attr-defined]
            cells = [
                (subj, prot, seed)
                for subj in subj_ids
                for prot in ("within", "official")
                for seed in seed_list
            ]
            for subj, protocol, seed in track(
                cells, description=cell_name, console=console
            ):
                data = data_by_subj[subj]
                base_arms = enumerate_arms_2b(sfreq=data.sfreq)
                for split in _splits_for_subject(data, protocol, seed=seed):
                    config = OPLBConfig(
                        alpha=0.5,
                        lambda_reg=1.0,
                        budget=float("inf"),
                        window_size=50,
                        discount_gamma=1.0,
                    )
                    result = run_ccb_on_split(
                        data,
                        split,
                        arms=base_arms,
                        config=config,
                        calibration_frac=0.3,
                        min_kappa=0.05,
                        max_arms=100,
                        seed=seed,
                        include_recent_rewards=False,
                    )
                    rows.append(
                        {
                            "cell": cell_name,
                            "windows": ";".join(
                                f"{lo}-{hi}" for lo, hi in windows
                            ),
                            "subject": subj,
                            "protocol": protocol,
                            "seed": seed,
                            "kappa": result.kappa,
                            "accuracy": result.accuracy,
                            "n_test": result.n_test,
                            "n_arms_surviving": result.n_arms_surviving,
                            "n_arms_total": len(base_arms),
                            "stream_rounds_run": int(result.arm_pulls.size),
                        }
                    )
    finally:
        arms_mod._TIME_WINDOWS = saved_windows  # type: ignore[attr-defined]

    elapsed = time.perf_counter() - t0
    console.log(f"Sweep completed in {elapsed:.1f}s — {len(rows)} rows.")

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    console.log(f"Saved → {output_csv}")

    summary = (
        df.groupby(["cell", "protocol"])["kappa"]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    console.print(summary)

    lines = [
        "# CCB time-window arm-grid sensitivity (BCI-IV-2b screening)",
        "",
        "Cells (only ``_TIME_WINDOWS`` varied; everything else held at the",
        "Phase-5 Stage-1 best-factorial CCB cell):",
        "",
        "| cell | windows | n_arms (pre-prune) |",
        "|---|---|---|",
    ]
    sample_counts = (
        df.drop_duplicates(["cell", "subject"])
        .groupby("cell")["n_arms_total"]
        .median()
        .astype(int)
        .to_dict()
    )
    for cell_name, ws in _CELLS.items():
        windows_str = ", ".join(f"{lo}-{hi}" for lo, hi in ws)
        lines.append(
            f"| {cell_name} | {windows_str} | {sample_counts.get(cell_name, '—')} |"
        )
    lines.append("")
    lines.append("Mean κ ± std across 9 subjects × seeds × 1 fold:")
    lines.append("")
    lines.append("| cell | within | official |")
    lines.append("|---|---|---|")
    for cell_name in _CELLS:
        wm = summary.loc[(cell_name, "within"), "mean"]
        ws = summary.loc[(cell_name, "within"), "std"]
        om = summary.loc[(cell_name, "official"), "mean"]
        os = summary.loc[(cell_name, "official"), "std"]
        lines.append(
            f"| {cell_name} | {wm:+.3f} ± {ws:.3f} | {om:+.3f} ± {os:.3f} |"
        )
    lines.append("")
    lines.append(
        "Hyperparameters held fixed at the Phase-5 Stage-1 best cell "
        "(α=0.5, calibration_frac=0.3, window_size=50, arm_pool=pruned, "
        "include_recent_rewards=False); see open-justifications.md."
    )
    output_md.write_text("\n".join(lines) + "\n")
    console.log(f"Saved → {output_md}")


if __name__ == "__main__":
    typer.run(main)
