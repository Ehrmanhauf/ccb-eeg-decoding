r"""Classical-classifier comparators (B3--B5) across all five datasets.

Answers the recurring reviewer question — "how does the CCB compare to a
decision tree (or an SVM, or a random forest)?" — by fitting off-the-shelf
scikit-learn classifier heads on the **same engineered features** the
fixed-pipeline baselines (B1 FBCSP, B2 band-power) and the CCB arm-heads use.
Holding the feature representation fixed and varying only the classifier head
isolates the classifier-choice effect (the orthogonal axis to the thesis's
B1-vs-B2 feature contrast).

Heads (``thesis.baselines.classical``): ``svm`` (RBF, B3), ``decision_tree``
(CART, B4), ``random_forest`` (B5), plus ``lda`` (shrinkage-LDA *consistency
anchor* — reproduces B1/B2 on the same features, and supplies the
fixed-pipeline number on Cho2017 which had none).

Feature families: FBCSP for the motor-imagery datasets; **both** FBCSP and
band-power for the cognitive-load datasets (matching B1 and B2). Features are
extracted **once per fold** and reused across heads.

Protocols mirror the existing fixed baselines exactly so the new rows slot into
the same Chapter-4 tables with the same per-cell cardinality:

- CL (STEW, WAUC): within-subject 5-fold CV, seed 42 (cf. ``run_fixed_baselines_cl.py``).
- MI (BCI-IV-2a/2b): within-subject 5-fold CV (seed 42) **and** the official
  session-0 -> session-1 split (cf. ``run_fbcsp_baseline.py``).
- Cho2017 (full + C3/Cz/C4): within-subject 5-fold CV, per-subject lazy MOABB
  load with retry (cf. ``run_ccb_moabb.py``); no clean two-session split.

No-leakage: the supervised FBCSP CSP and the SVM ``StandardScaler`` are fitted
train-only per fold; each dataset / configuration is fitted only on its own
trials.

Output: ``results/classical_baselines.csv`` — one row per
``(dataset, feature_family, classifier, subject, protocol, fold, seed)``.

Examples::

    # Everything (heaviest: Cho2017 full 64ch x 50 subjects).
    make fix-pth && PYTHONPATH=src .venv/bin/python scripts/run_classical_baselines.py

    # Just the CL datasets, smoke subset.
    PYTHONPATH=src .venv/bin/python scripts/run_classical_baselines.py \
        --datasets stew,wauc --subjects 1,2,3 --output results/classical_smoke.csv
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

from thesis.baselines.classical import make_classifier, make_feature_transformer
from thesis.data import SubjectData, load_bci2a, load_bci2b_screening
from thesis.data.moabb_load import load_cho2017
from thesis.data.stew_load import STEW_CHANNEL_ROLES, load_stew
from thesis.data.wauc_load import WAUC_CHANNEL_ROLES, load_wauc
from thesis.matched import SPLIT_SEED, matched_session_split, matched_within_cv
from thesis.metrics import compute_metrics

# Silence MNE's per-band CSP covariance-estimation chatter.
mne.set_log_level("WARNING")

# Datasets and the engineered feature families evaluated on each.
_MI_DATASETS = {"bci2a", "bci2b", "cho2017", "cho2017_3ch"}
_CL_DATASETS = {"stew", "wauc"}
_FEATURE_FAMILIES = {
    "bci2a": ("fbcsp",),
    "bci2b": ("fbcsp",),
    "cho2017": ("fbcsp",),
    "cho2017_3ch": ("fbcsp",),
    "stew": ("fbcsp", "bandpower"),
    "wauc": ("fbcsp", "bandpower"),
}
_DATASET_LABELS = {
    "bci2a": "BCI-IV-2a",
    "bci2b": "BCI-IV-2b",
    "cho2017": "Cho2017-full",
    "cho2017_3ch": "Cho2017-3ch",
    "stew": "STEW",
    "wauc": "WAUC",
}
_CHANNEL_ROLES = {"stew": STEW_CHANNEL_ROLES, "wauc": WAUC_CHANNEL_ROLES}


def _extract_once(
    family: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    sfreq: float,
    channel_roles: dict[str, list[int]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the feature transformer on train only, transform both splits (once)."""
    tr = make_feature_transformer(family, sfreq=sfreq, channel_roles=channel_roles)
    tr.fit(X_train, y_train)
    return tr.transform(X_train), tr.transform(X_test)


def _rows_for_split(
    data: SubjectData,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    dataset_label: str,
    families: tuple[str, ...],
    classifiers: list[str],
    channel_roles: dict[str, list[int]] | None,
    protocol: str,
    fold: int,
    seed: int,
) -> list[dict]:
    """Evaluate every (family x head) on one train/test split; reuse features."""
    X_train, y_train = data.X[train_idx], data.y[train_idx]
    X_test, y_test = data.X[test_idx], data.y[test_idx]
    rows: list[dict] = []
    for family in families:
        roles = channel_roles if family == "bandpower" else None
        Xtr_feat, Xte_feat = _extract_once(
            family, X_train, y_train, X_test, sfreq=data.sfreq, channel_roles=roles
        )
        for clf_name in classifiers:
            clf = make_classifier(clf_name, random_state=seed)
            clf.fit(Xtr_feat, y_train)
            m = compute_metrics(y_test, clf.predict(Xte_feat))
            rows.append(
                {
                    "dataset": dataset_label,
                    "feature_family": family,
                    "classifier": clf_name,
                    "subject": data.subject,
                    "protocol": protocol,
                    "fold": fold,
                    "seed": seed,
                    "kappa": float(m.kappa),
                    "accuracy": float(m.accuracy),
                    "n_train": int(len(train_idx)),
                    "n_test": int(m.n_trials),
                    "n_channels": int(data.n_channels),
                    "error": "",
                }
            )
    return rows


def _error_row(data: SubjectData, dataset_label: str, exc: Exception) -> dict:
    return {
        "dataset": dataset_label,
        "feature_family": "",
        "classifier": "",
        "subject": data.subject,
        "protocol": "",
        "fold": -1,
        "seed": -1,
        "kappa": float("nan"),
        "accuracy": float("nan"),
        "n_train": 0,
        "n_test": 0,
        "n_channels": int(data.n_channels),
        "error": str(exc),
    }


def _eval_subject(
    data: SubjectData,
    *,
    dataset_key: str,
    classifiers: list[str],
    seeds: list[int],
    n_folds: int,
) -> list[dict]:
    """All folds/seeds/protocols for one subject; per-subject try/except outside."""
    dataset_label = _DATASET_LABELS[dataset_key]
    families = _FEATURE_FAMILIES[dataset_key]
    channel_roles = _CHANNEL_ROLES.get(dataset_key)
    rows: list[dict] = []

    # Within-subject CV (all datasets). The fold PARTITION is fixed at the shared
    # matched-conditions split seed (so classical sees byte-identical folds to FBCSP and
    # EEGNet); only the classifier random_state varies with the loop seed.
    splits = list(matched_within_cv(data, n_splits=5, fold_seed=SPLIT_SEED))[:n_folds]
    for seed in seeds:
        for fold_i, split in enumerate(splits):
            rows += _rows_for_split(
                data,
                split.train_idx,
                split.test_idx,
                dataset_label=dataset_label,
                families=families,
                classifiers=classifiers,
                channel_roles=channel_roles,
                protocol="within",
                fold=fold_i,
                seed=seed,
            )

    # Official session-0 -> session-1 split (BCI-IV-2a/2b only — they ship two
    # sessions; CL datasets and Cho2017 have no clean equivalent).
    if dataset_key in {"bci2a", "bci2b"}:
        split = matched_session_split(data, train_session_idx=0, test_session_idx=1)
        rows += _rows_for_split(
            data,
            split.train_idx,
            split.test_idx,
            dataset_label=dataset_label,
            families=families,
            classifiers=classifiers,
            channel_roles=None,
            protocol="official",
            fold=0,
            seed=seeds[0],
        )
    return rows


def _load_cho_with_retry(
    channels: str, subject: int, *, console: Console, max_retries: int = 3, base_delay_s: float = 30.0
) -> SubjectData | None:
    """Lazy per-subject Cho2017 load with exponential-backoff network retries."""
    for attempt in range(1, max_retries + 1):
        try:
            datas = load_cho2017([subject], channels=channels)
            if not datas:
                console.log(f"  Cho2017 S{subject} excluded by loader; skipping")
                return None
            return datas[0]
        except Exception as exc:  # noqa: BLE001 — MOABB raises many network types
            if attempt == max_retries:
                console.log(f"  [red]Cho2017 S{subject} load failed after {max_retries}: {exc!r}[/red]")
                return None
            wait = base_delay_s * (2 ** (attempt - 1))
            console.log(f"  Cho2017 S{subject} attempt {attempt} failed; retry in {wait:.0f}s")
            time.sleep(wait)
    return None


def _parse_csv_int(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_str(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_subject_range(s: str) -> list[int]:
    """Parse '1-52' or '1,2,3' (or mixed) into a sorted unique int list."""
    out: set[int] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo, hi = tok.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(tok))
    return sorted(out)


def main(
    datasets: str = typer.Option(
        "stew,wauc,bci2a,bci2b,cho2017,cho2017_3ch",
        help="Comma-separated dataset keys to run.",
    ),
    classifiers: str = typer.Option(
        "lda,svm,decision_tree,random_forest",
        help="Heads to fit. 'lda' is the consistency anchor; svm/decision_tree/random_forest are B3-B5.",
    ),
    subjects: str = typer.Option(
        "all", help="Comma-separated subject IDs for STEW/WAUC/BCI-IV (or 'all')."
    ),
    cho_subjects: str = typer.Option(
        "1-52", help="Cho2017 subject range/list (loader drops 29,33)."
    ),
    seeds: str = typer.Option("42", help="Comma-separated seeds (default single seed 42)."),
    n_folds: int = typer.Option(5, help="Within-subject CV folds (max 5)."),
    output: Path = typer.Option(
        Path("results/classical_baselines.csv"),
        help="Output CSV path (long format).",
    ),
) -> None:
    console = Console()
    ds_keys = _parse_csv_str(datasets)
    clf_list = _parse_csv_str(classifiers)
    seed_list = _parse_csv_int(seeds)
    subj_filter = None if subjects == "all" else _parse_csv_int(subjects)

    for ds in ds_keys:
        if ds not in _DATASET_LABELS:
            console.print(f"[red]Unknown dataset {ds!r}; skipping.[/red]")

    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # --- Local-data datasets (eager batch load) --------------------------- #
    eager_loaders = {
        "stew": lambda: load_stew(subjects=subj_filter),
        "wauc": lambda: load_wauc(subjects=subj_filter),
        "bci2a": lambda: load_bci2a(subjects=subj_filter),
        "bci2b": lambda: load_bci2b_screening(subjects=subj_filter),
    }
    for ds in [d for d in ds_keys if d in eager_loaders]:
        console.log(f"Loading {_DATASET_LABELS[ds]} subjects={subj_filter or 'all'} …")
        data_list = eager_loaders[ds]()
        console.log(f"  → {len(data_list)} subjects.")
        for data in track(data_list, description=f"{_DATASET_LABELS[ds]}", console=console):
            try:
                rows += _eval_subject(
                    data, dataset_key=ds, classifiers=clf_list, seeds=seed_list, n_folds=n_folds
                )
            except Exception as exc:  # noqa: BLE001
                console.log(f"[red]✗ {_DATASET_LABELS[ds]} S{data.subject} failed: {exc}[/red]")
                rows.append(_error_row(data, _DATASET_LABELS[ds], exc))
        pd.DataFrame(rows).to_csv(output, index=False)  # checkpoint after each dataset

    # --- Cho2017 (lazy per-subject MOABB load) ---------------------------- #
    cho_channel_map = {"cho2017": "full", "cho2017_3ch": "c3_cz_c4"}
    for ds in [d for d in ds_keys if d in cho_channel_map]:
        cho_ids = _parse_subject_range(cho_subjects)
        console.log(f"Running {_DATASET_LABELS[ds]} for {len(cho_ids)} subjects (lazy load) …")
        for sid in cho_ids:
            data = _load_cho_with_retry(cho_channel_map[ds], sid, console=console)
            if data is None:
                continue
            try:
                rows += _eval_subject(
                    data, dataset_key=ds, classifiers=clf_list, seeds=seed_list, n_folds=n_folds
                )
                console.log(f"  {_DATASET_LABELS[ds]} S{sid} done (n_ch={data.n_channels}).")
            except Exception as exc:  # noqa: BLE001
                console.log(f"[red]✗ {_DATASET_LABELS[ds]} S{sid} failed: {exc}[/red]")
                rows.append(_error_row(data, _DATASET_LABELS[ds], exc))
            pd.DataFrame(rows).to_csv(output, index=False)  # checkpoint after each subject

    if not rows:
        console.print("[red]No rows generated. Check arguments and data availability.[/red]")
        raise typer.Exit(code=1)

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    console.log(f"Saved {len(df)} rows → {output}")

    summary = (
        df.dropna(subset=["kappa"])
        .groupby(["dataset", "feature_family", "classifier", "protocol"])["kappa"]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    console.rule("Mean κ per (dataset, feature_family, classifier, protocol)")
    console.print(summary)


if __name__ == "__main__":
    typer.run(main)
