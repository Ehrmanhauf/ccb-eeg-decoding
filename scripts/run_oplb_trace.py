r"""Log ONE real held-out-subject OPLB stream for the defense demo (demos/oplb_stream.html).

``run_ccb_on_split`` already produces the per-round arm-selection trace
(``CCBResult.arm_pulls``) and the per-round cumulative regret (``CCBResult.cumulative_regret``);
every production runner discards the arm-pull trace. This script re-runs the CCB on a SINGLE
real subject under the same locked recipe as ``run_regret_curves.py`` / ``run_ccb_newdata.py``
and persists the full trace to ``demos/oplb_trajectory.json`` (marked ``"real": true``), so
``build_demos.py`` renders a real stream instead of the synthetic placeholder. This is a small
single-subject re-run, not a new experiment.

Recommended cell (UAB, full 14-channel montage, subject 1, within-session) — matches the real
UAB regret curve in Fig. 4.3, which is the demo's static fallback::

    PYTHONPATH=src .venv/bin/python scripts/run_oplb_trace.py

Then rebuild the stream demo::

    PYTHONPATH=src .venv/bin/python scripts/build_demos.py --only oplb

Pick a different cell with ``--dataset {uab,cogbci} --montage {full,nearear} --subject N``.
The stream showcases the online *selection* machinery (the best-arm diagnostic shows selection
is competitive); the within-session absolute kappa is not a leakage-clean decoding score.
"""

from __future__ import annotations

import json
from pathlib import Path

import mne
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

_NICE = {"uab": "UAB", "cogbci": "COG-BCI"}
_MONT = {"full": "full", "nearear": "near-ear (T7/T8)"}


def main(
    dataset: str = typer.Option("uab", help="uab or cogbci"),
    montage: str = typer.Option("full", help="full or nearear"),
    subject: int = typer.Option(1, help="subject id to trace"),
    seed: int = typer.Option(42),
    alpha: float = typer.Option(0.5),
    calibration_frac: float = typer.Option(0.3),
    window_size: int = typer.Option(50),
    n_components: int = typer.Option(4),
    output: Path = typer.Option(Path("demos/oplb_trajectory.json")),
) -> None:
    console = Console()
    configs = {
        "uab": (load_emotiv_uab, UAB_CHANNEL_ROLES, UAB_CHANNELS),
        "cogbci": (lambda: load_cogbci(sessions=("S1",)), COGBCI_CHANNEL_ROLES, COGBCI_CANONICAL_EEG),
    }
    if dataset not in configs:
        raise typer.BadParameter(f"dataset must be one of {list(configs)}")
    loader, full_roles, chan_names = configs[dataset]

    console.log(f"Loading {dataset.upper()} …")
    full_list = loader()
    match = [d for d in full_list if int(d.subject) == subject]
    if not match:
        avail = sorted(int(d.subject) for d in full_list)
        raise typer.BadParameter(f"subject {subject} not found; available: {avail}")
    data = match[0]

    roles = full_roles if montage == "full" else NEAR_EAR_ROLES
    cell = data if montage == "full" else select_near_ear(data, chan_names)
    win = window_size if window_size > 0 else None

    split = list(within_subject_cv(cell, n_splits=5, seed=seed))[0]
    arms = enumerate_arms_generic(n_channels=cell.n_channels, n_components=n_components)
    cfg = OPLBConfig(alpha=alpha, lambda_reg=1.0, budget=float("inf"),
                     window_size=win, discount_gamma=1.0, per_round_cap=None)
    console.log(f"Streaming {dataset.upper()} · {montage} · S{subject} over {len(arms)} arms …")
    res = run_ccb_on_split(cell, split, arms=arms, config=cfg,
                           calibration_frac=calibration_frac, seed=seed,
                           include_recent_rewards=False, workload_channel_roles=roles)

    traj = {
        "real": True,
        "cell": f"real: {_NICE[dataset]} · {_MONT[montage]} · S{subject} (within-session)",
        "dataset": _NICE[dataset], "montage": montage, "subject": int(subject), "seed": seed,
        "arm_pulls": [int(a) for a in res.arm_pulls],
        "cumulative_regret": [float(r) for r in res.cumulative_regret],
        "n_arms": int(res.n_arms_surviving), "kappa": float(res.kappa),
        "accuracy": float(res.accuracy), "n_test": int(res.n_test),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(traj, indent=2))
    console.log(f"Wrote {output}: {len(traj['arm_pulls'])} rounds, "
                f"frozen-test κ={traj['kappa']:.3f}, {traj['n_arms']} arms survived.")
    console.log("Next:  PYTHONPATH=src .venv/bin/python scripts/build_demos.py --only oplb")


if __name__ == "__main__":
    typer.run(main)
