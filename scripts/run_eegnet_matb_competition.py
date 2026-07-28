r"""EEGNet on the COG-BCI MATB *competition split* — the deep-learning leaderboard anchor.

Mirrors ``scripts/run_matb_competition.py`` (same Zenodo competition data, same 15-subject
cohort, same S1->S2 cross-session split, full 61-channel montage + the T7/T8 near-ear
subset) but fits the EEGNet-8,2 deep comparator instead of the fixed pipelines / CCB. This
fills the EEGNet entry of the published-benchmark MATB-competition cell (tab:master /
tab:eegnet) and tests, on the leaderboard split itself, the Roy 2022 finding that deep
networks fared worst under cross-session drift.

No-leakage (CLAUDE.md §2): EEGNet is fit on session S1 only and scored on S2; per-trial
standardisation is intrinsic to EEGNet (each trial's own channel x time statistics, no
cross-trial moments); the near-ear montage is obtained by electrode position at load time
(``near_ear=True``), never distilled from the dense montage. EEGNet is training-stochastic,
so the training seed varies over ``--seeds`` and we report the mean over seeds (never a
single-seed point estimate), exactly as the rest of the EEGNet panel. Checkpointed per
(montage, subject); resumes from an existing CSV.

Output: ``results/eegnet_matb_competition.csv`` (dataset=COGBCI, task=matb-competition,
protocol=cross) -> aggregated by ``scripts/summarize_eegnet.py`` into ``eegnet_summary.csv``.

Run::

    PYTHONPATH=src .venv/bin/python scripts/run_eegnet_matb_competition.py
    # smoke: one subject, one seed, few epochs
    PYTHONPATH=src .venv/bin/python scripts/run_eegnet_matb_competition.py \
        --montages full --subjects 1 --seeds 42 --epochs 4 \
        --output results/eegnet_matbcomp_smoke.csv
"""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd
import typer
from rich.console import Console

from thesis.data.cogbci_matb_load import load_cogbci_matb
from thesis.metrics import compute_metrics
from thesis.protocols import cross_session_split

mne.set_log_level("WARNING")


def _eegnet():
    """Lazy import: torch is the optional 'benchmark' extra; keep the module importable without it."""
    from thesis.baselines.cnn import EEGNet
    return EEGNet


def _row(montage, cell, split, *, seed, kappa, acc, n_channels, error=""):
    return {
        "dataset": "COGBCI", "task": "matb-competition", "protocol": "cross",
        "montage": montage, "method": "eegnet", "subject": int(cell.subject), "seed": int(seed),
        "kappa": float(kappa), "accuracy": float(acc),
        "n_train": int(len(split.train_idx)), "n_test": int(len(split.test_idx)),
        "n_channels": int(n_channels), "error": error,
    }


def _parse_int(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main(
    montages: str = typer.Option("full,nearear"),
    subjects: str = typer.Option("all"),
    seeds: str = typer.Option("42,7", help="EEGNet training seeds; reported as the mean over them."),
    epochs: int = typer.Option(50, help="EEGNet training epochs (matches the rest of the panel)."),
    output: Path = typer.Option(Path("results/eegnet_matb_competition.csv")),
) -> None:
    console = Console()
    EEGNet = _eegnet()
    montage_list = [m.strip() for m in montages.split(",") if m.strip()]
    seed_list = _parse_int(seeds)
    subj_filter = None if subjects == "all" else _parse_int(subjects)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    done: set[tuple[str, int]] = set()
    if output.exists():
        prev = pd.read_csv(output).fillna("")
        rows = prev.to_dict("records")
        done = {(str(r["montage"]), int(r["subject"]))
                for r in rows if str(r.get("error", "")) == ""}
        console.log(f"Resuming: {len(done)} (montage, subject) cells done.")

    for montage in montage_list:
        console.rule(f"EEGNet · MATB competition · {montage} · cross-session S1->S2")
        console.log(f"Loading MATB competition ({montage}) subjects={subj_filter or 'all'} …")
        cells = load_cogbci_matb(subjects=subj_filter, near_ear=(montage == "nearear"))
        for cell in cells:
            if (montage, int(cell.subject)) in done:
                continue
            try:
                split = cross_session_split(cell, train_sessions=["S1"], test_sessions=["S2"])
                Xtr, ytr = cell.X[split.train_idx], cell.y[split.train_idx]
                Xte, yte = cell.X[split.test_idx], cell.y[split.test_idx]
                for seed in seed_list:
                    yp = EEGNet(sfreq=cell.sfreq, epochs=epochs, seed=seed).fit(Xtr, ytr).predict(Xte)
                    m = compute_metrics(yte, yp)
                    rows.append(_row(montage, cell, split, seed=seed, kappa=m.kappa, acc=m.accuracy,
                                     n_channels=cell.n_channels))
                console.log(f"  {montage} s{cell.subject} done ({cell.n_channels}ch)")
            except Exception as exc:  # noqa: BLE001
                console.log(f"[red]  ✗ {montage} s{cell.subject}: {exc}[/red]")
                rows.append(_row(montage, cell,
                                 type("S", (), {"train_idx": [], "test_idx": []})(),
                                 seed=-1, kappa=float("nan"), acc=float("nan"),
                                 n_channels=getattr(cell, "n_channels", 0), error=str(exc)))
            pd.DataFrame(rows).to_csv(output, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    console.log(f"Saved {len(df)} rows → {output}")
    valid = df[df.method == "eegnet"].dropna(subset=["kappa"])
    for montage in montage_list:
        sub = valid[valid.montage == montage]
        if not sub.empty:
            console.print(f"  {montage}: EEGNet mean κ={sub.kappa.mean():+.3f} "
                          f"(n={sub.subject.nunique()} subj, {len(sub)} fits)")


if __name__ == "__main__":
    typer.run(main)
