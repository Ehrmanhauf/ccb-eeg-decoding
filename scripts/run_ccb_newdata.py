r"""CCB within-session on the near-ear reframe datasets (Phase 3).

Runs the Constrained Contextual Bandit on UAB and COG-BCI N-back, at full montage
and the T7/T8 near-ear subset, in the stationary within-session regime, and
records the same-dataset Δκ versus the Phase-2 fixed pipelines.

Locked first-pass cell (matches the CL CCB headline): α=0.5, calibration 0.3,
sliding window 50, cap ∞, workload context, ``include_recent_rewards=False``,
``n_components=4`` (auto-capped to the channel count — 2 for the near-ear cells).
Five seeds {0,1,2,3,42}, one within-subject fold (mirrors ``run_ccb_stew.py``).

COG-BCI uses **session S1 only** (the within-session number shares its training
session with the Phase-4 cross-session split). Crash-safe: checkpoints per
(dataset, montage, subject, seed) and resumes from an existing CSV.

Output: ``results/ccb_newdata.csv``.

Examples::

    PYTHONPATH=src .venv/bin/python scripts/run_ccb_newdata.py
    PYTHONPATH=src .venv/bin/python scripts/run_ccb_newdata.py \
        --datasets uab --montages nearear --subjects 1,2 --seeds 0 \
        --output results/ccb_newdata_smoke.csv
"""

from __future__ import annotations

from pathlib import Path

import mne
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


def _parse_csv_int(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_str(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(
    datasets: str = typer.Option("uab,cogbci", help="Comma-separated: uab, cogbci."),
    montages: str = typer.Option("full,nearear", help="Comma-separated: full, nearear."),
    subjects: str = typer.Option("all", help="Subject IDs (or 'all')."),
    seeds: str = typer.Option("0,1,2,3,42", help="Comma-separated seeds."),
    alpha: float = typer.Option(0.5, help="OPLB α."),
    calibration_frac: float = typer.Option(0.3, help="Calibration fraction."),
    window_size: int = typer.Option(50, help="Sliding-window size (0 → stationary)."),
    n_components: int = typer.Option(4, help="CSP components per band (capped to n_channels)."),
    n_folds: int = typer.Option(1, help="Within-subject CV folds used (max 5)."),
    output: Path = typer.Option(Path("results/ccb_newdata.csv")),
) -> None:
    console = Console()
    ds_keys = _parse_csv_str(datasets)
    montage_list = _parse_csv_str(montages)
    seed_list = _parse_csv_int(seeds)
    subj_filter = None if subjects == "all" else _parse_csv_int(subjects)
    win = window_size if window_size > 0 else None
    output.parent.mkdir(parents=True, exist_ok=True)

    configs = {
        "uab": (lambda: load_emotiv_uab(subjects=subj_filter), UAB_CHANNEL_ROLES, UAB_CHANNELS),
        "cogbci": (
            lambda: load_cogbci(subjects=subj_filter, sessions=("S1",)),
            COGBCI_CHANNEL_ROLES, COGBCI_CANONICAL_EEG,
        ),
    }

    rows: list[dict] = []
    done: set[tuple[str, str, int, int]] = set()
    if output.exists():
        prev = pd.read_csv(output).fillna("")
        rows = prev.to_dict("records")
        done = {
            (str(r["dataset"]), str(r["montage"]), int(r["subject"]), int(r["seed"]))
            for r in rows
            if str(r.get("error", "")) == ""
        }
        console.log(f"Resuming: {len(done)} (dataset, montage, subject, seed) cells done.")

    for ds in ds_keys:
        if ds not in configs:
            console.print(f"[red]Unknown dataset {ds!r}; skipping.[/red]")
            continue
        loader, full_roles, chan_names = configs[ds]
        console.log(f"Loading {ds.upper()} (full) subjects={subj_filter or 'all'} …")
        full_list = loader()
        console.log(f"  → {len(full_list)} subjects.")
        for montage in montage_list:
            roles = full_roles if montage == "full" else NEAR_EAR_ROLES
            console.rule(f"{ds.upper()} · {montage} · CCB")
            for data in full_list:
                cell = data if montage == "full" else select_near_ear(data, chan_names)
                n_ch = cell.n_channels
                arms = enumerate_arms_generic(n_channels=n_ch, n_components=n_components)
                for seed in seed_list:
                    if (ds.upper(), montage, int(data.subject), int(seed)) in done:
                        continue
                    try:
                        split = list(within_subject_cv(cell, n_splits=5, seed=seed))[:n_folds][0]
                        cfg = OPLBConfig(
                            alpha=alpha, lambda_reg=1.0, budget=float("inf"),
                            window_size=win, discount_gamma=1.0, per_round_cap=None,
                        )
                        res = run_ccb_on_split(
                            cell, split, arms=arms, config=cfg,
                            calibration_frac=calibration_frac, seed=seed,
                            include_recent_rewards=False, workload_channel_roles=roles,
                        )
                        rows.append({
                            "dataset": ds.upper(), "montage": montage, "subject": data.subject,
                            "protocol": "within", "seed": seed, "alpha": alpha,
                            "calibration_frac": calibration_frac, "window_size": window_size,
                            "n_components": n_components, "n_channels": n_ch, "context": "workload",
                            "kappa": float(res.kappa), "accuracy": float(res.accuracy),
                            "n_test": int(res.n_test), "n_arms_surviving": int(res.n_arms_surviving),
                            "stream_rounds_run": int(res.arm_pulls.size),
                            "final_regret": float(res.cumulative_regret[-1]) if res.cumulative_regret.size else 0.0,
                            "error": "",
                        })
                        console.log(f"  {montage} s{data.subject} seed{seed}: κ={res.kappa:.3f} ({n_ch}ch, {res.n_arms_surviving} arms)")
                    except Exception as exc:  # noqa: BLE001
                        console.log(f"[red]  ✗ {ds} {montage} s{data.subject} seed{seed}: {exc}[/red]")
                        rows.append({
                            "dataset": ds.upper(), "montage": montage, "subject": data.subject,
                            "protocol": "within", "seed": seed, "alpha": alpha,
                            "calibration_frac": calibration_frac, "window_size": window_size,
                            "n_components": n_components, "n_channels": n_ch, "context": "workload",
                            "kappa": float("nan"), "accuracy": float("nan"), "n_test": 0,
                            "n_arms_surviving": 0, "stream_rounds_run": 0, "final_regret": 0.0,
                            "error": str(exc),
                        })
                    pd.DataFrame(rows).to_csv(output, index=False)  # checkpoint per (subj, seed)

    if not rows:
        console.print("[red]No rows generated.[/red]")
        raise typer.Exit(code=1)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    console.log(f"Saved {len(df)} rows → {output}")
    console.rule("Mean κ per (dataset, montage)")
    console.print(df.dropna(subset=["kappa"]).groupby(["dataset", "montage"])["kappa"].agg(["mean", "std", "count"]).round(4))


if __name__ == "__main__":
    typer.run(main)
