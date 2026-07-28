r"""Dump per-round cumulative OPLB regret for the regret figure (Phase 5, Fig 4.3).

The thesis defines the OPLB regret bound (§3.5.2) but never shows a regret curve;
the result CSVs persist only the scalar ``final_regret``. This script re-runs the
CCB on the headline cells and saves the **full** per-round cumulative regret
``R(t) = hindsight-best-arm reward − bandit reward`` (already computed inside
``run_ccb_on_split`` as ``CCBResult.cumulative_regret`` — this is a small re-run,
not a new experiment). Seed 42 only (one representative seed is enough for the
mean curve); all subjects.

Output (long): ``results/regret_curves.csv`` — one row per
(dataset, montage, subject, round) with the cumulative regret. The figure script
aggregates to a per-(cell, round) mean ± SD across subjects.

Example::

    PYTHONPATH=src .venv/bin/python scripts/run_regret_curves.py
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import typer
from rich.console import Console

from thesis.ccb.arms import enumerate_arms_generic
from thesis.ccb.oplb import OPLBConfig
from thesis.ccb.runner import run_ccb_on_split
from thesis.data import select_near_ear
from thesis.data.cogbci_load import COGBCI_CANONICAL_EEG, COGBCI_CHANNEL_ROLES, load_cogbci
from thesis.data.emotiv_uab_load import UAB_CHANNELS, UAB_CHANNEL_ROLES, load_emotiv_uab
from thesis.data.near_ear import NEAR_EAR_ROLES
from thesis.protocols import within_subject_cv

mne.set_log_level("WARNING")


def _parse_str(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main(
    datasets: str = typer.Option("uab,cogbci"),
    montages: str = typer.Option("full,nearear"),
    seed: int = typer.Option(42),
    alpha: float = typer.Option(0.5),
    calibration_frac: float = typer.Option(0.3),
    window_size: int = typer.Option(50),
    n_components: int = typer.Option(4),
    output: Path = typer.Option(Path("results/regret_curves.csv")),
) -> None:
    console = Console()
    ds_keys = _parse_str(datasets)
    montage_list = _parse_str(montages)
    win = window_size if window_size > 0 else None
    output.parent.mkdir(parents=True, exist_ok=True)

    configs = {
        "uab": (lambda: load_emotiv_uab(), UAB_CHANNEL_ROLES, UAB_CHANNELS),
        "cogbci": (lambda: load_cogbci(sessions=("S1",)), COGBCI_CHANNEL_ROLES, COGBCI_CANONICAL_EEG),
    }
    rows: list[dict] = []
    done: set[tuple[str, str, int]] = set()
    if output.exists():
        prev = pd.read_csv(output)
        rows = prev.to_dict("records")
        done = {(str(r["dataset"]), str(r["montage"]), int(r["subject"])) for r in rows}
        console.log(f"Resuming: {len(done)} (dataset, montage, subject) curves done.")

    for ds in ds_keys:
        if ds not in configs:
            continue
        loader, full_roles, chan_names = configs[ds]
        console.log(f"Loading {ds.upper()} …")
        full_list = loader()
        for montage in montage_list:
            roles = full_roles if montage == "full" else NEAR_EAR_ROLES
            console.rule(f"{ds.upper()} · {montage} · regret")
            for data in full_list:
                if (ds.upper(), montage, int(data.subject)) in done:
                    continue
                cell = data if montage == "full" else select_near_ear(data, chan_names)
                try:
                    split = list(within_subject_cv(cell, n_splits=5, seed=seed))[0]
                    arms = enumerate_arms_generic(n_channels=cell.n_channels, n_components=n_components)
                    cfg = OPLBConfig(alpha=alpha, lambda_reg=1.0, budget=float("inf"),
                                     window_size=win, discount_gamma=1.0, per_round_cap=None)
                    res = run_ccb_on_split(cell, split, arms=arms, config=cfg,
                                           calibration_frac=calibration_frac, seed=seed,
                                           include_recent_rewards=False, workload_channel_roles=roles)
                    curve = np.asarray(res.cumulative_regret, dtype=float)
                    for t, r in enumerate(curve):
                        rows.append({"dataset": ds.upper(), "montage": montage,
                                     "subject": data.subject, "seed": seed,
                                     "round": int(t), "cumulative_regret": float(r)})
                    console.log(f"  {montage} s{data.subject}: {curve.size} rounds, final R={curve[-1] if curve.size else 0:.2f}")
                except Exception as exc:  # noqa: BLE001
                    console.log(f"[red]  ✗ {ds} {montage} s{data.subject}: {exc}[/red]")
                pd.DataFrame(rows).to_csv(output, index=False)

    pd.DataFrame(rows).to_csv(output, index=False)
    console.log(f"Saved {len(rows)} regret-curve rows → {output}")


if __name__ == "__main__":
    typer.run(main)
