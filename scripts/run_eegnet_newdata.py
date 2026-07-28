r"""EEGNet decoding comparator on the near-ear datasets + PVT (the all-datasets CNN column).

Companion to ``run_eegnet_benchmark.py`` (which covers the core MI/CL cells: BCI-IV-2a/2b,
Cho2017, STEW, WAUC). This runner adds the deep-learning comparator to the cells those miss:

  * UAB        — within-session, full + near-ear (mirrors ``run_ccb_newdata.py``)
  * COG-BCI N-back — within-session (S1) + cross-session (S1→S2/S3), full + near-ear
  * COG-BCI MATB  — cross-session (S1→S2/S3), full + near-ear (the leaderboard-anchored cell)
  * COG-BCI PVT   — within-session (S1) + cross-session, full + near-ear (the vigilance cell)

EEGNet is run under the **same protocol as the CCB/classical cell** it sits beside, so its row
drops straight into the Chapter-4 comparison tables next to the CCB and the fixed pipelines.
The within cells mirror the CCB's *fold-0-per-seed* protocol (``run_ccb_newdata.py``); the
cross-session cells use the single S1→S2/S3 split (``run_cogbci_crosssession.py``). EEGNet is
training-stochastic, so we vary the training seed over ``--seeds`` and report mean ± std.

No-leakage (CLAUDE.md §2): EEGNet is fit on the training split only; per-trial standardisation
uses each trial's own channel×time statistics (``cnn.py``); near-ear is the T7/T8 subset by
electrode position; single-thread CPU for reproducibility. Crash-safe: checkpoints per
(dataset/task, protocol, montage, subject, seed) and resumes from an existing CSV.

Output: ``results/eegnet_newdata.csv``.

Examples::

    uv sync --extra benchmark
    # Smoke: one UAB subject, near-ear, one seed, few epochs.
    PYTHONPATH=src .venv/bin/python scripts/run_eegnet_newdata.py \
        --cells uab_within --montages nearear --subjects 1 --seeds 42 --epochs 8 \
        --output results/eegnet_newdata_smoke.csv
    # Full (heavy; background, caffeinated).
    caffeinate -i make fix-pth && PYTHONPATH=src .venv/bin/python scripts/run_eegnet_newdata.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mne
import pandas as pd
import typer
from rich.console import Console

from thesis.data import select_near_ear
from thesis.data.cogbci_load import COGBCI_CANONICAL_EEG, COGBCI_NBACK_FILES, load_cogbci
from thesis.data.cogbci_pvt_load import load_cogbci_pvt
from thesis.data.emotiv_uab_load import UAB_CHANNELS, load_emotiv_uab
from thesis.metrics import compute_metrics
from thesis.protocols import cross_session_split, within_subject_cv

mne.set_log_level("WARNING")

# The cells this runner covers, each = (dataset_label, task, protocol). "within" cells use
# fold-0-per-seed within_subject_cv; "cross" cells use the single S1→S2/S3 split.
_MATB_STEMS = {"MATBeasy": "easy", "MATBmed": "med", "MATBdiff": "diff"}
_ALL_CELLS = ("uab_within", "cogbci_nback_within", "cogbci_nback_cross",
              "cogbci_matb_cross", "pvt_within", "pvt_cross")


def _eegnet():
    """Import EEGNet lazily (torch is the optional `benchmark` extra); single-core."""
    import torch

    from thesis.baselines.cnn import EEGNet

    torch.set_num_threads(1)
    return EEGNet


def _row(*, dataset, task, protocol, montage, subject, seed, n_channels, kappa, accuracy,
         n_train, n_test, error="") -> dict:
    return {
        "dataset": dataset, "task": task, "protocol": protocol, "montage": montage,
        "method": "eegnet", "subject": int(subject), "seed": int(seed),
        "kappa": float(kappa), "accuracy": float(accuracy), "n_train": int(n_train),
        "n_test": int(n_test), "n_channels": int(n_channels), "error": error,
    }


def _fit_eval(EEGNet, base, split, *, sfreq, epochs, seed):
    """Train EEGNet on the split's train idx; return (kappa, acc, n_train, n_test)."""
    Xtr, ytr = base.X[split.train_idx], base.y[split.train_idx]
    Xte, yte = base.X[split.test_idx], base.y[split.test_idx]
    yp = EEGNet(sfreq=sfreq, epochs=epochs, seed=seed).fit(Xtr, ytr).predict(Xte)
    m = compute_metrics(yte, yp)
    return float(m.kappa), float(m.accuracy), len(split.train_idx), int(m.n_trials)


def _session_subset(cell, ses: str):
    mask = (cell.metadata["session"] == ses).to_numpy()
    return replace(cell, X=cell.X[mask], y=cell.y[mask],
                   metadata=cell.metadata[mask].reset_index(drop=True))


def _near_ear(cell, dataset_label: str):
    """Near-ear subset by electrode position; PVT loads near-ear at source so it is a no-op."""
    if dataset_label == "UAB":
        return select_near_ear(cell, UAB_CHANNELS)
    return select_near_ear(cell, COGBCI_CANONICAL_EEG)


def _load_cells(cell_key: str, montage: str, subj_filter):
    """Return (dataset_label, task, protocol, list[SubjectData]) for a covered cell."""
    near = montage == "nearear"
    if cell_key == "uab_within":
        data = load_emotiv_uab(subjects=subj_filter)
        return "UAB", "nback", "within", [d if not near else _near_ear(d, "UAB") for d in data]
    if cell_key.startswith("cogbci_nback"):
        proto = "within" if cell_key.endswith("within") else "cross"
        sessions = ("S1",) if proto == "within" else ("S1", "S2", "S3")
        data = load_cogbci(subjects=subj_filter, sessions=sessions, set_stems=COGBCI_NBACK_FILES)
        cells = [d if not near else _near_ear(d, "COGBCI") for d in data]
        return "COGBCI", "nback", proto, cells
    if cell_key == "cogbci_matb_cross":
        data = load_cogbci(subjects=subj_filter, sessions=("S1", "S2", "S3"), set_stems=_MATB_STEMS)
        cells = [d if not near else _near_ear(d, "COGBCI") for d in data]
        return "COGBCI", "matb", "cross", cells
    if cell_key.startswith("pvt"):
        proto = "within" if cell_key.endswith("within") else "cross"
        # the PVT loader subsets to near-ear at source (near_ear flag), all 3 sessions.
        cells = load_cogbci_pvt(subjects=subj_filter, near_ear=near)
        return "COGBCI-PVT", "pvt", proto, cells
    raise ValueError(f"unknown cell {cell_key!r}")


def _split_for(base, protocol: str, seed: int):
    """fold-0-per-seed within CV, or the single S1→S2/S3 cross-session split."""
    if protocol == "within":
        return list(within_subject_cv(base, n_splits=5, seed=seed))[0]
    return cross_session_split(base, train_sessions=["S1"], test_sessions=["S2", "S3"])


def _parse_int(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_str(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main(
    cells: str = typer.Option(",".join(_ALL_CELLS), help=f"Subset of {_ALL_CELLS}."),
    montages: str = typer.Option("full,nearear", help="full, nearear."),
    subjects: str = typer.Option("all", help="Subject IDs (or 'all')."),
    seeds: str = typer.Option("42,7,123", help="EEGNet training seeds; mean ± std reported."),
    epochs: int = typer.Option(50, help="EEGNet training epochs."),
    output: Path = typer.Option(Path("results/eegnet_newdata.csv")),
) -> None:
    console = Console()
    EEGNet = _eegnet()
    cell_keys = _parse_str(cells)
    montage_list = _parse_str(montages)
    seed_list = _parse_int(seeds)
    subj_filter = None if subjects == "all" else _parse_int(subjects)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    done: set[tuple] = set()
    if output.exists():
        prev = pd.read_csv(output).fillna("")
        rows = prev.to_dict("records")
        done = {(str(r["dataset"]), str(r["task"]), str(r["protocol"]), str(r["montage"]),
                 int(r["subject"]), int(r["seed"])) for r in rows if str(r.get("error", "")) == ""}
        console.log(f"Resuming: {len(done)} cells done.")

    for cell_key in cell_keys:
        for montage in montage_list:
            label, task, proto, data_cells = _load_cells(cell_key, montage, subj_filter)
            console.rule(f"{label} {task} · {proto} · {montage} · EEGNet ({len(data_cells)} subj)")
            for cell in data_cells:
                base = _session_subset(cell, "S1") if (proto == "within" and label != "UAB") else cell
                n_ch = base.n_channels
                for seed in seed_list:
                    key = (label, task, proto, montage, int(base.subject), int(seed))
                    if key in done:
                        continue
                    try:
                        split = _split_for(base, proto, seed)
                        k, a, ntr, nte = _fit_eval(EEGNet, base, split, sfreq=base.sfreq,
                                                   epochs=epochs, seed=seed)
                        rows.append(_row(dataset=label, task=task, protocol=proto, montage=montage,
                                         subject=base.subject, seed=seed, n_channels=n_ch,
                                         kappa=k, accuracy=a, n_train=ntr, n_test=nte))
                        console.log(f"  {montage} s{base.subject} seed{seed}: κ={k:.3f} acc={a:.3f} ({n_ch}ch)")
                    except Exception as exc:  # noqa: BLE001
                        console.log(f"[red]  ✗ {cell_key} {montage} s{base.subject} seed{seed}: {exc}[/red]")
                        rows.append(_row(dataset=label, task=task, protocol=proto, montage=montage,
                                         subject=base.subject, seed=seed, n_channels=n_ch,
                                         kappa=float("nan"), accuracy=float("nan"), n_train=0,
                                         n_test=0, error=str(exc)))
                    pd.DataFrame(rows).to_csv(output, index=False)  # checkpoint per (subj, seed)

    if not rows:
        console.print("[red]No rows generated.[/red]")
        raise typer.Exit(code=1)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    console.log(f"Saved {len(df)} rows → {output}")
    console.rule("EEGNet mean κ per (dataset, task, protocol, montage)")
    console.print(df.dropna(subset=["kappa"])
                  .groupby(["dataset", "task", "protocol", "montage"])["kappa"]
                  .agg(["mean", "std", "count"]).round(4))


if __name__ == "__main__":
    typer.run(main)
