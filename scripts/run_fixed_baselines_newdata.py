r"""Fixed-pipeline baselines B1-B5 on the near-ear reframe datasets (Phase 2).

Establishes the fixed-pipeline reference kappa on UAB and COG-BCI N-back, at
**full montage and the T7/T8 near-ear subset**, before the CCB is run. The bar
the CCB (Phase 3-4) is measured against.

Heads (``thesis.baselines.classical``): ``lda`` (shrinkage-LDA = B1 on FBCSP
features / B2 on band-power features), ``svm`` (RBF, B3), ``decision_tree`` (B4),
``random_forest`` (B5). Feature families: FBCSP **and** band-power (both CL
datasets). Features extracted once per fold, reused across heads. Protocol:
within-subject 5-fold CV, seed 42 (mirrors ``run_classical_baselines.py``).

COG-BCI uses **session S1 only** for the within-session number, so it shares its
training session with the Phase-4 cross-session split (train S1 -> test S2/S3) and
the within-vs-cross contrast isolates drift, not training data.

No-leakage: the supervised FBCSP CSP and the SVM ``StandardScaler`` are fitted
train-only per fold; the near-ear subset is selected by electrode position at load
time. UAB carries the STEW-like segment-leak caveat (each difficulty = one block);
COG-BCI N-back is leakage-resistant (separated recurring blocks).

Output: ``results/fixed_baseline_newdata.csv`` — one row per
``(dataset, montage, feature_family, classifier, subject, fold, seed)``.
Checkpointed per subject (crash-safe across a session resume).

Examples::

    PYTHONPATH=src .venv/bin/python scripts/run_fixed_baselines_newdata.py
    # smoke:
    PYTHONPATH=src .venv/bin/python scripts/run_fixed_baselines_newdata.py \
        --datasets uab --subjects 1,2 --output results/fixed_newdata_smoke.csv
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import typer
from rich.console import Console

from thesis.baselines.classical import make_classifier, make_feature_transformer
from thesis.data import SubjectData, select_near_ear
from thesis.data.cogbci_load import COGBCI_CANONICAL_EEG, COGBCI_CHANNEL_ROLES, load_cogbci
from thesis.data.emotiv_uab_load import UAB_CHANNELS, UAB_CHANNEL_ROLES, load_emotiv_uab
from thesis.data.near_ear import NEAR_EAR_ROLES
from thesis.metrics import compute_metrics
from thesis.protocols import within_subject_cv

mne.set_log_level("WARNING")

_FEATURE_FAMILIES = ("fbcsp", "bandpower")
_DEFAULT_HEADS = ("lda", "svm", "decision_tree", "random_forest")


def _eval_subject(
    data: SubjectData,
    *,
    dataset: str,
    montage: str,
    channel_roles: dict[str, list[int]],
    classifiers: list[str],
    seed: int,
    n_folds: int,
) -> list[dict]:
    rows: list[dict] = []
    splits = list(within_subject_cv(data, n_splits=5, seed=seed))[:n_folds]
    for fold_i, split in enumerate(splits):
        X_train, y_train = data.X[split.train_idx], data.y[split.train_idx]
        X_test, y_test = data.X[split.test_idx], data.y[split.test_idx]
        for family in _FEATURE_FAMILIES:
            roles = channel_roles if family == "bandpower" else None
            tr = make_feature_transformer(family, sfreq=data.sfreq, channel_roles=roles)
            tr.fit(X_train, y_train)
            Xtr, Xte = tr.transform(X_train), tr.transform(X_test)
            for clf_name in classifiers:
                clf = make_classifier(clf_name, random_state=seed)
                clf.fit(Xtr, y_train)
                m = compute_metrics(y_test, clf.predict(Xte))
                rows.append({
                    "dataset": dataset, "montage": montage, "feature_family": family,
                    "classifier": clf_name, "subject": data.subject, "fold": fold_i,
                    "seed": seed, "kappa": float(m.kappa), "accuracy": float(m.accuracy),
                    "n_train": int(len(split.train_idx)), "n_test": int(m.n_trials),
                    "n_channels": int(data.n_channels), "error": "",
                })
    return rows


def _parse_csv_int(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_str(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(
    datasets: str = typer.Option("uab,cogbci", help="Comma-separated: uab, cogbci."),
    montages: str = typer.Option("full,nearear", help="Comma-separated: full, nearear."),
    classifiers: str = typer.Option(",".join(_DEFAULT_HEADS), help="Heads to fit."),
    subjects: str = typer.Option("all", help="Subject IDs (or 'all')."),
    seed: int = typer.Option(42, help="Single CV seed (mirrors the fixed baselines)."),
    n_folds: int = typer.Option(5, help="Within-subject CV folds (max 5)."),
    output: Path = typer.Option(Path("results/fixed_baseline_newdata.csv")),
) -> None:
    console = Console()
    ds_keys = _parse_csv_str(datasets)
    montage_list = _parse_csv_str(montages)
    clf_list = _parse_csv_str(classifiers)
    subj_filter = None if subjects == "all" else _parse_csv_int(subjects)
    output.parent.mkdir(parents=True, exist_ok=True)

    # (dataset key) -> (loader, full channel-roles, full channel names)
    configs = {
        "uab": (lambda: load_emotiv_uab(subjects=subj_filter), UAB_CHANNEL_ROLES, UAB_CHANNELS),
        "cogbci": (
            lambda: load_cogbci(subjects=subj_filter, sessions=("S1",)),
            COGBCI_CHANNEL_ROLES, COGBCI_CANONICAL_EEG,
        ),
    }
    # Resume support: keep prior rows and skip (dataset, montage, subject) cells
    # already completed, so a session-resume kill does not force a full recompute.
    rows: list[dict] = []
    done: set[tuple[str, str, int]] = set()
    if output.exists():
        prev = pd.read_csv(output).fillna("")
        rows = prev.to_dict("records")
        done = {
            (str(r["dataset"]), str(r["montage"]), int(r["subject"]))
            for r in rows
            if str(r.get("error", "")) == "" and int(r.get("fold", -1)) >= 0
        }
        console.log(f"Resuming: {len(done)} (dataset, montage, subject) cells already done.")

    for ds in ds_keys:
        if ds not in configs:
            console.print(f"[red]Unknown dataset {ds!r}; skipping.[/red]")
            continue
        loader, full_roles, chan_names = configs[ds]
        console.log(f"Loading {ds.upper()} (full montage) subjects={subj_filter or 'all'} …")
        full_list = loader()
        console.log(f"  → {len(full_list)} subjects.")
        for montage in montage_list:
            roles = full_roles if montage == "full" else NEAR_EAR_ROLES
            console.rule(f"{ds.upper()} · {montage}")
            for data in full_list:
                if (ds.upper(), montage, int(data.subject)) in done:
                    continue
                cell_data = data if montage == "full" else select_near_ear(data, chan_names)
                try:
                    rows += _eval_subject(
                        cell_data, dataset=ds.upper(), montage=montage, channel_roles=roles,
                        classifiers=clf_list, seed=seed, n_folds=n_folds,
                    )
                    console.log(f"  s{data.subject} ({cell_data.n_channels}ch) done")
                except Exception as exc:  # noqa: BLE001
                    console.log(f"[red]  ✗ {ds} {montage} s{data.subject}: {exc}[/red]")
                    rows.append({
                        "dataset": ds.upper(), "montage": montage, "feature_family": "",
                        "classifier": "", "subject": data.subject, "fold": -1, "seed": -1,
                        "kappa": float("nan"), "accuracy": float("nan"), "n_train": 0,
                        "n_test": 0, "n_channels": int(cell_data.n_channels), "error": str(exc),
                    })
                pd.DataFrame(rows).to_csv(output, index=False)  # checkpoint per subject

    if not rows:
        console.print("[red]No rows generated.[/red]")
        raise typer.Exit(code=1)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    console.log(f"Saved {len(df)} rows → {output}")
    console.rule("Mean κ per (dataset, montage, feature_family, classifier)")
    summary = (
        df.dropna(subset=["kappa"])
        .groupby(["dataset", "montage", "feature_family", "classifier"])["kappa"]
        .agg(["mean", "std", "count"]).round(4)
    )
    console.print(summary)


if __name__ == "__main__":
    typer.run(main)
