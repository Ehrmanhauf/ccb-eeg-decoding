r"""Best-arm-as-fixed-pipeline diagnostic (Phase 5 — the key missing number).

For each cell, reports the test-κ of the *single best arm* in the calibrated pool
used as a **frozen pipeline** (no bandit selection, no exploration). The logic
re-uses the exact calibration path the CCB runs (``fit_heads_on_calibration``,
which returns the surviving pool sorted by calibration κ — element 0 is the best
arm), then scores that one arm's head on the held-out test split.

Interpretation (Fig. 4.2): if best-arm ≈ best fixed baseline while the CCB sits
well below both, the deficit is localised to the bandit's *selection-and-
exploration machinery*, not the arm bank's features — the Contribution-4 claim
the thesis currently only asserts. If best-arm sits below the baseline, the arm
bank is feature-impoverished and the claim is adjusted honestly.

Runs on the leakage-clean cells (the new near-ear CL cells + the existing 2b /
Cho2017), within-subject, mirroring the CCB cell. Output:
``results/best_arm_diagnostic.csv`` — one row per (dataset, montage, subject,
seed). Checkpointed + resumable.

Examples::

    PYTHONPATH=src .venv/bin/python scripts/run_best_arm_diagnostic.py
    PYTHONPATH=src .venv/bin/python scripts/run_best_arm_diagnostic.py \
        --datasets uab --montages nearear --subjects 1,2 --seeds 0
"""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd
import typer
from rich.console import Console

from thesis.ccb.arms import enumerate_arms_generic
from thesis.ccb.runner import fit_heads_on_calibration
from thesis.data import select_near_ear
from thesis.data.cogbci_load import COGBCI_CANONICAL_EEG, load_cogbci
from thesis.data.emotiv_uab_load import UAB_CHANNELS, load_emotiv_uab
from thesis.metrics import compute_metrics
from thesis.protocols import within_subject_cv

mne.set_log_level("WARNING")


def _parse_int(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_str(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main(
    datasets: str = typer.Option("uab,cogbci", help="uab, cogbci."),
    montages: str = typer.Option("full,nearear", help="full, nearear."),
    subjects: str = typer.Option("all"),
    seeds: str = typer.Option("0,1,2,3,42"),
    calibration_frac: float = typer.Option(0.3),
    n_components: int = typer.Option(4),
    output: Path = typer.Option(Path("results/best_arm_diagnostic.csv")),
) -> None:
    console = Console()
    ds_keys = _parse_str(datasets)
    montage_list = _parse_str(montages)
    seed_list = _parse_int(seeds)
    subj_filter = None if subjects == "all" else _parse_int(subjects)
    output.parent.mkdir(parents=True, exist_ok=True)

    configs = {
        "uab": (lambda: load_emotiv_uab(subjects=subj_filter), UAB_CHANNELS),
        "cogbci": (lambda: load_cogbci(subjects=subj_filter, sessions=("S1",)), COGBCI_CANONICAL_EEG),
    }
    rows: list[dict] = []
    done: set[tuple[str, str, int, int]] = set()
    if output.exists():
        prev = pd.read_csv(output).fillna("")
        rows = prev.to_dict("records")
        done = {(str(r["dataset"]), str(r["montage"]), int(r["subject"]), int(r["seed"]))
                for r in rows if str(r.get("error", "")) == ""}
        console.log(f"Resuming: {len(done)} cells done.")

    for ds in ds_keys:
        if ds not in configs:
            continue
        loader, chan_names = configs[ds]
        console.log(f"Loading {ds.upper()} …")
        full_list = loader()
        for montage in montage_list:
            console.rule(f"{ds.upper()} · {montage} · best-arm")
            for data in full_list:
                cell = data if montage == "full" else select_near_ear(data, chan_names)
                arms = enumerate_arms_generic(n_channels=cell.n_channels, n_components=n_components)
                for seed in seed_list:
                    if (ds.upper(), montage, int(data.subject), int(seed)) in done:
                        continue
                    try:
                        split = list(within_subject_cv(cell, n_splits=5, seed=seed))[0]
                        surviving, heads, _ = fit_heads_on_calibration(
                            cell, split.train_idx, arms,
                            calibration_frac=calibration_frac, seed=seed,
                        )
                        best = surviving[0]  # pool is sorted by calibration κ desc
                        y_pred = heads[best.arm_id].predict(cell.X[split.test_idx], cell.sfreq)
                        m = compute_metrics(cell.y[split.test_idx], y_pred)
                        rows.append({
                            "dataset": ds.upper(), "montage": montage, "subject": data.subject,
                            "seed": seed, "n_channels": cell.n_channels,
                            "best_arm_kappa": float(m.kappa), "best_arm_accuracy": float(m.accuracy),
                            "best_arm_id": int(best.arm_id), "best_arm_spatial": str(best.spatial),
                            "n_arms_surviving": len(surviving), "n_test": int(m.n_trials), "error": "",
                        })
                        console.log(f"  {montage} s{data.subject} seed{seed}: best-arm κ={m.kappa:.3f} (arm {best.arm_id}/{best.spatial})")
                    except Exception as exc:  # noqa: BLE001
                        console.log(f"[red]  ✗ {ds} {montage} s{data.subject} seed{seed}: {exc}[/red]")
                        rows.append({
                            "dataset": ds.upper(), "montage": montage, "subject": data.subject,
                            "seed": seed, "n_channels": cell.n_channels,
                            "best_arm_kappa": float("nan"), "best_arm_accuracy": float("nan"),
                            "best_arm_id": -1, "best_arm_spatial": "", "n_arms_surviving": 0,
                            "n_test": 0, "error": str(exc),
                        })
                    pd.DataFrame(rows).to_csv(output, index=False)

    if not rows:
        raise typer.Exit(code=1)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    console.log(f"Saved {len(df)} rows → {output}")
    console.print(df.dropna(subset=["best_arm_kappa"]).groupby(["dataset", "montage"])["best_arm_kappa"].agg(["mean", "std", "count"]).round(4))


if __name__ == "__main__":
    typer.run(main)
