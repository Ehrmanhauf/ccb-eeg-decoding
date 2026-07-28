r"""Best-arm-as-frozen-pipeline diagnostic on the EXISTING leakage-clean cells (Phase 5).

Companion to ``run_best_arm_diagnostic.py`` (which covers the new UAB / COG-BCI
cells). For a clean Fig 4.2 the diagnostic must run on the *leakage-clean* cells
the advisor lists — BCI-IV-2b, WAUC, Cho2017-3ch — where best-arm-vs-fixed-vs-CCB
is interpretable (unlike the within-CV-leak-confounded new cells). Each dataset
uses the *same* arm bank its committed CCB used (``enumerate_arms_2b`` for 2b,
``enumerate_arms_generic`` otherwise), so the best-arm bar is comparable to the
committed CCB bar.

Output: ``results/best_arm_existing.csv``. Within-subject CV (fold 0), 5 seeds.
Checkpointed + resumable.

Example::

    PYTHONPATH=src .venv/bin/python scripts/run_best_arm_existing.py --datasets bci2b,wauc
"""

from __future__ import annotations

import time
from pathlib import Path

import mne
import pandas as pd
import typer
from rich.console import Console

from thesis.ccb.arms import enumerate_arms_2b, enumerate_arms_generic
from thesis.ccb.runner import fit_heads_on_calibration
from thesis.data import load_bci2b_screening
from thesis.data.moabb_load import load_cho2017
from thesis.data.wauc_load import load_wauc
from thesis.metrics import compute_metrics
from thesis.protocols import within_subject_cv

mne.set_log_level("WARNING")


def _arms_for(ds: str, data):
    if ds == "bci2b":
        return enumerate_arms_2b(data.sfreq)
    return enumerate_arms_generic(n_channels=data.n_channels, n_components=4)


def _parse_int(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_str(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def _best_arm_row(ds_label, data, seed, calibration_frac):
    arms = _arms_for(ds_label_key(ds_label), data)
    split = list(within_subject_cv(data, n_splits=5, seed=seed))[0]
    surviving, heads, _ = fit_heads_on_calibration(
        data, split.train_idx, arms, calibration_frac=calibration_frac, seed=seed
    )
    best = surviving[0]
    y_pred = heads[best.arm_id].predict(data.X[split.test_idx], data.sfreq)
    m = compute_metrics(data.y[split.test_idx], y_pred)
    return {
        "dataset": ds_label, "montage": "full", "subject": data.subject, "seed": seed,
        "n_channels": data.n_channels, "best_arm_kappa": float(m.kappa),
        "best_arm_accuracy": float(m.accuracy), "best_arm_id": int(best.arm_id),
        "best_arm_spatial": str(best.spatial), "n_arms_surviving": len(surviving),
        "n_test": int(m.n_trials), "error": "",
    }


_LABELS = {"bci2b": "BCI-IV-2b", "wauc": "WAUC", "cho2017_3ch": "Cho2017-3ch"}


def ds_label_key(label: str) -> str:
    return {v: k for k, v in _LABELS.items()}.get(label, "")


def main(
    datasets: str = typer.Option("bci2b,wauc,cho2017_3ch", help="bci2b, wauc, cho2017_3ch."),
    subjects: str = typer.Option("all"),
    cho_subjects: str = typer.Option("1-52"),
    seeds: str = typer.Option("0,1,2,3,42"),
    calibration_frac: float = typer.Option(0.3),
    output: Path = typer.Option(Path("results/best_arm_existing.csv")),
) -> None:
    console = Console()
    ds_keys = _parse_str(datasets)
    seed_list = _parse_int(seeds)
    subj_filter = None if subjects == "all" else _parse_int(subjects)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    done: set[tuple[str, int, int]] = set()
    if output.exists():
        prev = pd.read_csv(output).fillna("")
        rows = prev.to_dict("records")
        done = {(str(r["dataset"]), int(r["subject"]), int(r["seed"]))
                for r in rows if str(r.get("error", "")) == ""}
        console.log(f"Resuming: {len(done)} cells done.")

    def run_subject(ds_key, data):
        label = _LABELS[ds_key]
        for seed in seed_list:
            if (label, int(data.subject), int(seed)) in done:
                continue
            try:
                rows.append(_best_arm_row(label, data, seed, calibration_frac))
                console.log(f"  {label} s{data.subject} seed{seed}: best-arm κ={rows[-1]['best_arm_kappa']:.3f}")
            except Exception as exc:  # noqa: BLE001
                console.log(f"[red]  ✗ {label} s{data.subject} seed{seed}: {exc}[/red]")
                rows.append({"dataset": label, "montage": "full", "subject": data.subject, "seed": seed,
                             "n_channels": data.n_channels, "best_arm_kappa": float("nan"),
                             "best_arm_accuracy": float("nan"), "best_arm_id": -1, "best_arm_spatial": "",
                             "n_arms_surviving": 0, "n_test": 0, "error": str(exc)})
            pd.DataFrame(rows).to_csv(output, index=False)

    for ds in ds_keys:
        if ds == "bci2b":
            console.rule("BCI-IV-2b · best-arm")
            for data in load_bci2b_screening(subjects=subj_filter):
                run_subject(ds, data)
        elif ds == "wauc":
            console.rule("WAUC · best-arm")
            for data in load_wauc(subjects=subj_filter):
                run_subject(ds, data)
        elif ds == "cho2017_3ch":
            console.rule("Cho2017-3ch · best-arm (lazy MOABB)")
            cho_ids = _parse_int(cho_subjects.replace("-", ",")) if "-" not in cho_subjects else list(
                range(int(cho_subjects.split("-")[0]), int(cho_subjects.split("-")[1]) + 1))
            for sid in cho_ids:
                try:
                    datas = load_cho2017([sid], channels="c3_cz_c4")
                except Exception as exc:  # noqa: BLE001
                    console.log(f"  Cho2017 S{sid} load failed: {exc}"); time.sleep(2); continue
                if datas:
                    run_subject(ds, datas[0])

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    console.log(f"Saved {len(df)} rows → {output}")
    console.print(df.dropna(subset=["best_arm_kappa"]).groupby("dataset")["best_arm_kappa"].agg(["mean", "std", "count"]).round(4))


if __name__ == "__main__":
    typer.run(main)
