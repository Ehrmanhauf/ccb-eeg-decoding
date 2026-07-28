r"""Aggregate the EEGNet decoding-comparator CSVs into one per-cell summary.

Reads whichever of the three EEGNet decoding outputs exist
(``results/eegnet_benchmark.csv`` — core MI/CL; ``results/eegnet_cho.csv`` — Cho2017;
``results/eegnet_newdata.csv`` — near-ear UAB/COG-BCI + PVT) and emits one tidy row per
``(dataset, task, protocol, montage)`` with mean ± std Cohen's κ and accuracy over the
training seeds (and per-subject/per-fold cells). For the leakage-clean LOSO cells (STEW,
WAUC), the **pooled global κ** is recomputed from the stored prediction sequences — the
honest CL headline, exactly as ``loso_stew.csv`` does.

This is the single source the Chapter-4 EEGNet column draws from; re-run it as the
background jobs accumulate rows.

Output: ``results/eegnet_summary.csv`` (+ a printed table).

Usage::

    PYTHONPATH=src .venv/bin/python scripts/summarize_eegnet.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from thesis.metrics import compute_metrics

_RES = Path("results")
_SOURCES = ("eegnet_benchmark.csv", "eegnet_cho.csv", "eegnet_newdata.csv")


def _load() -> pd.DataFrame:
    frames = []
    for p in sorted(_RES.glob("eegnet_*.csv")):
        if p.name == "eegnet_summary.csv":
            continue
        d = pd.read_csv(p)
        if d.empty or "kappa" not in d.columns:
            continue
        if "task" not in d.columns:
            d["task"] = "-"
        d["source"] = p.name
        frames.append(d)
    if not frames:
        raise SystemExit("no EEGNet CSVs found yet (results/eegnet_*.csv).")
    df = pd.concat(frames, ignore_index=True)
    df["task"] = df["task"].fillna("-")
    df["montage"] = df.get("montage", "full").fillna("full")
    return df


def _pooled_loso(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled global κ per (dataset, montage, seed) from stored LOSO sequences, then mean over seeds."""
    if "y_true_seq" not in df.columns:
        return pd.DataFrame()
    loso = df[(df.protocol == "loso") & (df.y_true_seq.astype(str).str.len() > 0)
              & (df.y_true_seq.astype(str) != "nan")]
    rows = []
    for (ds, mt, seed), g in loso.groupby(["dataset", "montage", "seed"]):
        yt = np.concatenate([np.array(str(s).split("|")) for s in g.y_true_seq])
        yp = np.concatenate([np.array(str(s).split("|")) for s in g.y_pred_seq])
        rows.append({"dataset": ds, "montage": mt, "seed": seed,
                     "pooled_kappa": float(compute_metrics(yt, yp).kappa), "n": len(yt)})
    if not rows:
        return pd.DataFrame()
    per_seed = pd.DataFrame(rows)
    return (per_seed.groupby(["dataset", "montage"])["pooled_kappa"]
            .agg(["mean", "std"]).round(4).reset_index()
            .rename(columns={"mean": "loso_pooled_kappa_mean", "std": "loso_pooled_kappa_std"}))


def main() -> None:
    console = Console()
    df = _load()
    valid = df[df.get("error", "").fillna("") == ""].dropna(subset=["kappa"]) if "error" in df else df.dropna(subset=["kappa"])

    summary = (valid.groupby(["dataset", "task", "protocol", "montage"])
               .agg(kappa_mean=("kappa", "mean"), kappa_std=("kappa", "std"),
                    acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                    n_cells=("kappa", "size"))
               .round(4).reset_index())

    pooled = _pooled_loso(df)
    if not pooled.empty:
        summary = summary.merge(pooled, on=["dataset", "montage"], how="left")

    out = _RES / "eegnet_summary.csv"
    summary.to_csv(out, index=False)
    console.rule("EEGNet decoding summary — mean over seeds/subjects (per cell)")
    console.print(summary.to_string(index=False))
    if not pooled.empty:
        console.rule("LOSO pooled global κ (leakage-clean CL headline)")
        console.print(pooled.to_string(index=False))
    console.log(f"Saved → {out}  ({len(summary)} cells from {df['source'].nunique()} source file(s))")


if __name__ == "__main__":
    main()
