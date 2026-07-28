r"""4-class BCI-IV-2a run — benchmark-faithful comparison vs the published 4-class numbers.

The thesis evaluates 2a at 2-class (to match 2b); the published 2a benchmarks are 4-class
(Ang 2012 FBCSP κ = 0.569; Lawhern 2018 EEGNet κ = 0.70). This runs our own pipelines on
the FULL 4-class 2a (``load_bci2a(n_classes=4)``) so the comparison is like-for-like.

Methods (``--methods``): ``classical`` (FBCSP × {LDA, SVM, Decision Tree, Random Forest}),
``ccb`` (OPLB, 108-arm 2a bank, §2.1 22-channel research-question opt-in), ``eegnet`` (the
compact-CNN deep comparator; slow). Protocols: within-subject 5-fold CV and the official
session-0→1 split (the protocol the published benchmarks use; this is the cell to compare).
Metric: Cohen's κ + accuracy. No-leakage: feature transforms / CSP / CCB calibration fit on
the train split only.

Examples::

    # fast: classical + CCB, both protocols, 2 seeds
    PYTHONPATH=src .venv/bin/python scripts/run_2a_4class.py --methods classical,ccb
    # slow: deep comparator only, append to a separate file (run caffeinated/background)
    PYTHONPATH=src .venv/bin/python scripts/run_2a_4class.py --methods eegnet \
        --output results/bci2a_4class_eegnet.csv

Output: results/bci2a_4class.csv (one row per method/classifier/protocol/subject/fold/seed).
"""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd
import typer
from rich.console import Console
from rich.progress import track

from thesis.baselines.classical import make_classifier, make_feature_transformer
from thesis.ccb.arms import enumerate_arms_2a
from thesis.ccb.oplb import OPLBConfig
from thesis.ccb.runner import run_ccb_on_split
from thesis.data import load_bci2a
from thesis.matched import matched_session_split, matched_within_cv
from thesis.metrics import compute_metrics

mne.set_log_level("ERROR")
_HEADS = ("lda", "svm", "decision_tree", "random_forest")


def _splits(data, protocol: str):
    # Fold partition fixed at the shared matched-conditions seed; the per-method
    # algorithm seed (classifier random_state / EEGNet init / CCB) varies separately.
    if protocol == "within":
        return list(matched_within_cv(data, n_splits=5))
    return [matched_session_split(data, train_session_idx=0, test_session_idx=1)]


def _row(method, clf, protocol, subject, fold, seed, m_kappa, m_acc, ntr, nte):
    return {"dataset": "BCI-IV-2a", "n_classes": 4, "method": method, "classifier": clf,
            "protocol": protocol, "subject": subject, "fold": fold, "seed": seed,
            "kappa": float(m_kappa), "accuracy": float(m_acc), "n_train": int(ntr),
            "n_test": int(nte)}


def _classical_rows(data, subject, protocol, seed):
    rows = []
    for fold_i, sp in enumerate(_splits(data, protocol)):
        Xtr, ytr = data.X[sp.train_idx], data.y[sp.train_idx]
        Xte, yte = data.X[sp.test_idx], data.y[sp.test_idx]
        tr = make_feature_transformer("fbcsp", sfreq=data.sfreq)
        tr.fit(Xtr, ytr)
        Ftr, Fte = tr.transform(Xtr), tr.transform(Xte)
        for head in _HEADS:
            clf = make_classifier(head, random_state=seed).fit(Ftr, ytr)
            m = compute_metrics(yte, clf.predict(Fte))
            rows.append(_row("classical", head, protocol, subject, fold_i, seed,
                             m.kappa, m.accuracy, len(sp.train_idx), m.n_trials))
    return rows


def _ccb_rows(data, subject, protocol, seed):
    arms = enumerate_arms_2a(data.sfreq)
    rows = []
    for fold_i, sp in enumerate(_splits(data, protocol)):
        config = OPLBConfig(alpha=0.5, lambda_reg=1.0, budget=float("inf"),
                            window_size=50, discount_gamma=1.0)
        res = run_ccb_on_split(data, sp, arms=arms, config=config, calibration_frac=0.3,
                               seed=seed, include_recent_rewards=False,
                               allow_22ch_research_question=True)
        rows.append(_row("ccb", "oplb", protocol, subject, fold_i, seed,
                         res.kappa, res.accuracy, len(sp.train_idx), res.n_test))
    return rows


def _eegnet_rows(data, subject, protocol, seed):
    import torch

    from thesis.baselines.cnn import EEGNet
    torch.set_num_threads(1)
    rows = []
    for fold_i, sp in enumerate(_splits(data, protocol)):
        Xtr, ytr = data.X[sp.train_idx], data.y[sp.train_idx]
        Xte, yte = data.X[sp.test_idx], data.y[sp.test_idx]
        yp = EEGNet(sfreq=data.sfreq, epochs=50, seed=seed).fit(Xtr, ytr).predict(Xte)
        m = compute_metrics(yte, yp)
        rows.append(_row("eegnet", "eegnet", protocol, subject, fold_i, seed,
                         m.kappa, m.accuracy, len(sp.train_idx), m.n_trials))
    return rows


def _parse_int(s: str) -> list[int]:
    out: set[int] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if "-" in tok:
            lo, hi = tok.split("-")
            out.update(range(int(lo), int(hi) + 1))
        elif tok:
            out.add(int(tok))
    return sorted(out)


def main(
    subjects: str = typer.Option("1-9"),
    seeds: str = typer.Option("42,7", help="CCB/EEGNet training seeds; classical uses the first."),
    protocols: str = typer.Option("within,official"),
    methods: str = typer.Option("classical,ccb", help="Subset of classical,ccb,eegnet."),
    output: Path = typer.Option(Path("results/bci2a_4class.csv")),
) -> None:
    console = Console()
    subj = _parse_int(subjects)
    seed_list = _parse_int(seeds)
    prots = [p.strip() for p in protocols.split(",") if p.strip()]
    want = {m.strip() for m in methods.split(",") if m.strip()}
    output.parent.mkdir(parents=True, exist_ok=True)

    console.log(f"Loading 4-class BCI-IV-2a for subjects {subj} …")
    data_by = {s: load_bci2a(subjects=[s], n_classes=4)[0] for s in subj}
    console.log(f"  → {len(data_by)} subjects, {data_by[subj[0]].class_balance} classes/subj.")

    rows: list[dict] = []
    for s in track(subj, description="2a 4-class", console=console):
        data = data_by[s]
        for protocol in prots:
            if "classical" in want:
                rows += _classical_rows(data, s, protocol, seed_list[0])
            if "ccb" in want:
                for seed in seed_list:
                    rows += _ccb_rows(data, s, protocol, seed)
            if "eegnet" in want:
                for seed in seed_list:
                    rows += _eegnet_rows(data, s, protocol, seed)
        pd.DataFrame(rows).to_csv(output, index=False)  # checkpoint per subject
        console.log(f"  subject {s} done ({len(rows)} rows).")

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    console.log(f"Saved {len(df)} rows → {output}")
    summary = (df.dropna(subset=["kappa"])
               .groupby(["method", "classifier", "protocol"])[["kappa", "accuracy"]]
               .mean().round(3))
    console.rule("4-class 2a — mean κ / accuracy")
    console.print(summary)


if __name__ == "__main__":
    typer.run(main)
