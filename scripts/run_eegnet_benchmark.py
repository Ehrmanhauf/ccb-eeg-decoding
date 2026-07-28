r"""EEGNet decoding benchmark — the deep-learning comparator across all datasets.

Closes gap **G1** of `design-doc/benchmark-comparison-plan.md`: EEGNet (Lawhern 2018,
the standard compact EEG CNN) was previously run only on one BCI-IV-2a subject as a
*compute* comparator (`run_hardware_efficiency_benchmark.py`). Here it is run as a
first-class **decoding** benchmark on every core dataset, under the *same* protocols as
the classical suite (`run_classical_baselines.py`), so its rows drop into the same
Chapter-4 tables beside the CCB and B1-B5.

Protocols (mirroring the existing comparators):
  - MI (BCI-IV-2a/2b, Cho2017 full + C3/Cz/C4): within-subject 5-fold CV; 2a/2b also the
    official session-0 -> session-1 split.
  - CL (STEW, WAUC): within-subject CV (leakage-caveated) **and** leave-one-subject-out
    (leakage-clean) -- the LOSO pooled kappa is the honest CL benchmark number.

Multi-seed: EEGNet is training-stochastic (BCI-IV-2a s1: kappa 0.655 at seed 42 vs 0.207
at seed 7; `results/hardware_efficiency_kappa_2a.csv`). The **CV split is fixed at seed
42** (so EEGNet sees the *same* folds as `classical_baselines.csv`) and only the EEGNet
**training seed** varies over ``--seeds``; we report mean +/- std over those seeds and
never a single-seed point estimate.

Matched conditions: all folds and the LOSO training subsample come from
``thesis.matched`` (shared with the classical suite and the CCB), so EEGNet trains on the
*identical* per-fold pool --- e.g. the common 4000-epoch class-stratified LOSO cap, not
the full ~34k-epoch WAUC pool. Per-trial standardisation is intrinsic to EEGNet and is
disclosed as such (it cannot be applied to FBCSP/band-power features).

No-leakage (CLAUDE.md §2): EEGNet is fit on the training split only; per-trial
standardisation uses each trial's own channel x time statistics (no cross-trial moments,
see `cnn.py`); single-thread CPU for reproducibility.

Output: ``results/eegnet_benchmark.csv`` -- one row per
``(dataset, protocol, subject, seed)``; LOSO rows also carry the prediction sequences so
the pooled global kappa is recomputable (cf. `loso_stew.csv`).

Examples::

    uv sync --extra benchmark   # installs CPU torch
    # Smoke: two 2a subjects, one seed, few epochs.
    PYTHONPATH=src .venv/bin/python scripts/run_eegnet_benchmark.py \
        --datasets bci2a --subjects 1,2 --seeds 42 --epochs 10 \
        --output results/eegnet_smoke.csv
    # Full sweep (heavy; background).
    make fix-pth && PYTHONPATH=src .venv/bin/python scripts/run_eegnet_benchmark.py
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

from thesis.data import SubjectData, load_bci2a, load_bci2b_screening
from thesis.data.moabb_load import load_cho2017
from thesis.data.stew_load import load_stew
from thesis.data.wauc_load import load_wauc
from thesis.matched import (
    SPLIT_SEED,
    matched_loso,
    matched_session_split,
    matched_within_cv,
)
from thesis.metrics import compute_metrics

mne.set_log_level("WARNING")

_MI_DATASETS = {"bci2a", "bci2b", "cho2017", "cho2017_3ch"}
_CL_DATASETS = {"stew", "wauc"}
_DATASET_LABELS = {
    "bci2a": "BCI-IV-2a",
    "bci2b": "BCI-IV-2b",
    "cho2017": "Cho2017-full",
    "cho2017_3ch": "Cho2017-3ch",
    "stew": "STEW",
    "wauc": "WAUC",
}
def _eegnet():
    """Import EEGNet lazily (torch is the optional `benchmark` extra)."""
    import torch

    from thesis.baselines.cnn import EEGNet

    torch.set_num_threads(1)  # single-core, reproducible
    return EEGNet


def _row(
    *,
    dataset_label: str,
    protocol: str,
    subject: int,
    seed: int,
    n_channels: int,
    kappa: float,
    accuracy: float,
    n_train: int,
    n_test: int,
    y_true_seq: str = "",
    y_pred_seq: str = "",
    error: str = "",
) -> dict:
    return {
        "dataset": dataset_label,
        "method": "eegnet",
        "montage": "full",
        "protocol": protocol,
        "subject": subject,
        "seed": seed,
        "kappa": kappa,
        "accuracy": accuracy,
        "n_train": n_train,
        "n_test": n_test,
        "n_channels": n_channels,
        "y_true_seq": y_true_seq,
        "y_pred_seq": y_pred_seq,
        "error": error,
    }


def _fit_predict(EEGNet, Xtr, ytr, Xte, *, sfreq: float, epochs: int, seed: int):
    """Train EEGNet on (Xtr, ytr); return predictions on Xte."""
    clf = EEGNet(sfreq=sfreq, epochs=epochs, seed=seed).fit(Xtr, ytr)
    return clf.predict(Xte)


def _eval_within_and_official(
    EEGNet, data: SubjectData, *, dataset_key: str, seeds: list[int], epochs: int, n_folds: int,
    want: set[str],
) -> list[dict]:
    """Within-subject CV (fixed split seed) + official split (MI); EEGNet seed varies."""
    label = _DATASET_LABELS[dataset_key]
    rows: list[dict] = []
    # Folds and any subsample come from the shared matched-conditions source so EEGNet
    # sees byte-identical splits to the classical suite and the CCB (within/official are
    # uncapped). See src/thesis/matched.py.
    splits = list(matched_within_cv(data, n_splits=5, fold_seed=SPLIT_SEED))[:n_folds]
    for seed in seeds:
        if "within" in want:
            for split in splits:
                Xtr, ytr = data.X[split.train_idx], data.y[split.train_idx]
                Xte, yte = data.X[split.test_idx], data.y[split.test_idx]
                yp = _fit_predict(EEGNet, Xtr, ytr, Xte, sfreq=data.sfreq, epochs=epochs, seed=seed)
                m = compute_metrics(yte, yp)
                rows.append(_row(dataset_label=label, protocol="within", subject=data.subject,
                                 seed=seed, n_channels=data.n_channels, kappa=float(m.kappa),
                                 accuracy=float(m.accuracy), n_train=len(split.train_idx),
                                 n_test=int(m.n_trials)))
        if "official" in want and dataset_key in {"bci2a", "bci2b"}:
            split = matched_session_split(data, train_session_idx=0, test_session_idx=1)
            Xtr, ytr = data.X[split.train_idx], data.y[split.train_idx]
            Xte, yte = data.X[split.test_idx], data.y[split.test_idx]
            yp = _fit_predict(EEGNet, Xtr, ytr, Xte, sfreq=data.sfreq, epochs=epochs, seed=seed)
            m = compute_metrics(yte, yp)
            rows.append(_row(dataset_label=label, protocol="official", subject=data.subject,
                             seed=seed, n_channels=data.n_channels, kappa=float(m.kappa),
                             accuracy=float(m.accuracy), n_train=len(split.train_idx),
                             n_test=int(m.n_trials)))
    return rows


def _eval_loso(
    EEGNet, data_list: list[SubjectData], *, dataset_key: str, seeds: list[int], epochs: int,
    console: Console,
) -> list[dict]:
    """Leave-one-subject-out (leakage-clean) for the CL datasets; EEGNet seed varies.

    The per-fold training pool is the shared matched-conditions LOSO subsample (default
    cap 4000, fixed split seed) so EEGNet trains on the *identical* held-in data as the
    fixed baselines and the CCB --- not the full ~34k-epoch pool, which was the
    matched-conditions bug this routing fixes. See src/thesis/matched.py.
    """
    label = _DATASET_LABELS[dataset_key]
    folds = list(matched_loso(data_list))
    rows: list[dict] = []
    for seed in seeds:
        for pooled, split in folds:
            Xtr, ytr = pooled.X[split.train_idx], pooled.y[split.train_idx]
            yte = pooled.y[split.test_idx]
            try:
                yp = _fit_predict(EEGNet, Xtr, ytr, pooled.X[split.test_idx],
                                  sfreq=pooled.sfreq, epochs=epochs, seed=seed)
            except Exception as exc:  # noqa: BLE001
                console.log(f"[red]✗ {label} LOSO s{pooled.subject} seed{seed}: {exc}[/red]")
                rows.append(_row(dataset_label=label, protocol="loso", subject=pooled.subject,
                                 seed=seed, n_channels=pooled.n_channels, kappa=float("nan"),
                                 accuracy=float("nan"), n_train=len(split.train_idx),
                                 n_test=len(yte), error=str(exc)))
                continue
            # kappa is ill-defined for a single-class held-out fold; store NaN but keep
            # the sequences so the pooled global kappa still uses them (cf. loso_stew).
            kappa = float("nan") if len(set(yte.tolist())) < 2 else float(compute_metrics(yte, yp).kappa)
            acc = float((yp == yte).mean())
            rows.append(_row(dataset_label=label, protocol="loso", subject=pooled.subject,
                             seed=seed, n_channels=pooled.n_channels, kappa=kappa, accuracy=acc,
                             n_train=len(split.train_idx), n_test=len(yte),
                             y_true_seq="|".join(map(str, yte.tolist())),
                             y_pred_seq="|".join(map(str, yp.tolist()))))
    return rows


def _load_cho_with_retry(channels, subject, *, console, max_retries=3, base_delay_s=30.0):
    for attempt in range(1, max_retries + 1):
        try:
            datas = load_cho2017([subject], channels=channels)
            return datas[0] if datas else None
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                console.log(f"  [red]Cho2017 S{subject} load failed: {exc!r}[/red]")
                return None
            time.sleep(base_delay_s * (2 ** (attempt - 1)))
    return None


def _parse_int(s: str) -> list[int]:
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


def _parse_str(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _pooled_loso_kappa(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled global kappa per (dataset, seed) from the stored LOSO sequences."""
    loso = df[(df.protocol == "loso") & (df.y_true_seq.astype(str) != "")]
    out = []
    for (ds, seed), g in loso.groupby(["dataset", "seed"]):
        yt = np.concatenate([np.array(s.split("|")) for s in g.y_true_seq])
        yp = np.concatenate([np.array(s.split("|")) for s in g.y_pred_seq])
        out.append({"dataset": ds, "seed": seed, "pooled_kappa": float(compute_metrics(yt, yp).kappa),
                    "n": len(yt)})
    return pd.DataFrame(out)


def main(
    datasets: str = typer.Option("bci2a,bci2b,cho2017,cho2017_3ch,stew,wauc"),
    protocols: str = typer.Option("within,official,loso", help="Subset of {within,official,loso}."),
    subjects: str = typer.Option("all", help="Subject IDs for STEW/WAUC/BCI-IV ('all' or list)."),
    cho_subjects: str = typer.Option("1-52", help="Cho2017 range/list (loader drops 29,33)."),
    seeds: str = typer.Option("42,7,123", help="EEGNet training seeds (CV split is fixed at 42)."),
    epochs: int = typer.Option(50, help="EEGNet training epochs."),
    n_folds: int = typer.Option(5),
    output: Path = typer.Option(Path("results/eegnet_benchmark.csv")),
) -> None:
    console = Console()
    EEGNet = _eegnet()
    ds_keys = _parse_str(datasets)
    want = set(_parse_str(protocols))
    seed_list = _parse_int(seeds)
    subj_filter = None if subjects == "all" else _parse_int(subjects)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    eager = {
        "stew": lambda: load_stew(subjects=subj_filter),
        "wauc": lambda: load_wauc(subjects=subj_filter),
        "bci2a": lambda: load_bci2a(subjects=subj_filter, n_classes=4),  # 4-class: match CCB + published EEGNet κ=0.70
        "bci2b": lambda: load_bci2b_screening(subjects=subj_filter),
    }
    for ds in [d for d in ds_keys if d in eager]:
        console.log(f"Loading {_DATASET_LABELS[ds]} …")
        data_list = eager[ds]()
        console.log(f"  → {len(data_list)} subjects.")
        if want & {"within", "official"}:
            for data in track(data_list, description=f"{_DATASET_LABELS[ds]} within", console=console):
                try:
                    rows += _eval_within_and_official(
                        EEGNet, data, dataset_key=ds, seeds=seed_list, epochs=epochs,
                        n_folds=n_folds, want=want,
                    )
                except Exception as exc:  # noqa: BLE001
                    console.log(f"[red]✗ {_DATASET_LABELS[ds]} S{data.subject}: {exc}[/red]")
        if "loso" in want and ds in _CL_DATASETS and len(data_list) > 2:
            console.log(f"  {_DATASET_LABELS[ds]} LOSO ({len(data_list)} folds × {len(seed_list)} seeds) …")
            rows += _eval_loso(EEGNet, data_list, dataset_key=ds, seeds=seed_list, epochs=epochs,
                               console=console)
        pd.DataFrame(rows).to_csv(output, index=False)  # checkpoint per dataset

    cho_map = {"cho2017": "full", "cho2017_3ch": "c3_cz_c4"}
    for ds in [d for d in ds_keys if d in cho_map]:
        if not (want & {"within"}):
            continue
        for sid in _parse_int(cho_subjects):
            data = _load_cho_with_retry(cho_map[ds], sid, console=console)
            if data is None:
                continue
            try:
                rows += _eval_within_and_official(
                    EEGNet, data, dataset_key=ds, seeds=seed_list, epochs=epochs,
                    n_folds=n_folds, want=want,
                )
                console.log(f"  {_DATASET_LABELS[ds]} S{sid} done.")
            except Exception as exc:  # noqa: BLE001
                console.log(f"[red]✗ {_DATASET_LABELS[ds]} S{sid}: {exc}[/red]")
            pd.DataFrame(rows).to_csv(output, index=False)  # checkpoint per subject

    if not rows:
        console.print("[red]No rows generated.[/red]")
        raise typer.Exit(code=1)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    console.log(f"Saved {len(df)} rows → {output}")

    per_cell = (df.dropna(subset=["kappa"]).groupby(["dataset", "protocol"])["kappa"]
                .agg(["mean", "std", "count"]).round(4))
    console.rule("EEGNet mean κ per (dataset, protocol) — within/official: per-subject; loso: per-fold")
    console.print(per_cell)
    pooled = _pooled_loso_kappa(df)
    if not pooled.empty:
        console.rule("LOSO pooled global κ (the clean CL headline)")
        console.print(pooled.groupby("dataset")["pooled_kappa"].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    typer.run(main)
