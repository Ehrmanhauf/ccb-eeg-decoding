r"""COG-BCI cross-session: fixed pipeline vs CCB (Phase 4 — the headline cell).

The deployment-regime test: train on session S1, test on the later sessions
S2/S3 recorded a week apart, so the only change between train and test is
across-session drift — the one regime where the CCB's online adaptation could
earn its cost. Run at full montage and at the T7/T8 near-ear subset, on the
leakage-clean N-back cell (and, with ``--task matb``, on the MATB cell that is
directly comparable to the published sub-60% expert-team leaderboard).

For each (montage, subject): the fixed pipelines B1-B5 are fitted on S1 and scored
on S2+S3; the CCB calibrates + streams on S1 and is scored, frozen, on S2+S3
(5 seeds). The same-cell Δκ (best fixed − CCB) is the headline number.

No-leakage: fixed CSP / SVM scaler fitted on S1 only; near-ear by electrode
position. Crash-safe: checkpoints per (task, montage, subject) and resumes.

Output: ``results/crosssession_cogbci.csv``.

Examples::

    PYTHONPATH=src .venv/bin/python scripts/run_cogbci_crosssession.py
    PYTHONPATH=src .venv/bin/python scripts/run_cogbci_crosssession.py \
        --task matb --montages nearear --subjects 1,2 --seeds 0 \
        --output results/crosssession_smoke.csv
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import typer
from rich.console import Console

from thesis.baselines.classical import make_classifier, make_feature_transformer
from thesis.ccb.arms import enumerate_arms_generic
from thesis.ccb.oplb import OPLBConfig
from thesis.ccb.runner import run_ccb_on_split
from thesis.data import select_near_ear
from thesis.data.cogbci_load import (
    COGBCI_CANONICAL_EEG,
    COGBCI_CHANNEL_ROLES,
    COGBCI_NBACK_FILES,
    load_cogbci,
)
from thesis.data.near_ear import NEAR_EAR_ROLES
from thesis.metrics import compute_metrics
from thesis.protocols import cross_session_split

mne.set_log_level("WARNING")

_HEADS = ("lda", "svm", "decision_tree", "random_forest")
_FEATURE_FAMILIES = ("fbcsp", "bandpower")
_TASK_STEMS = {
    "nback": COGBCI_NBACK_FILES,
    "matb": {"MATBeasy": "easy", "MATBmed": "med", "MATBdiff": "diff"},
}


def _fixed_rows(cell, split, *, task, montage, roles, seed) -> list[dict]:
    """B1-B5 fixed pipelines: fit on S1 (train), score S2+S3 (test)."""
    Xtr_raw, ytr = cell.X[split.train_idx], cell.y[split.train_idx]
    Xte_raw, yte = cell.X[split.test_idx], cell.y[split.test_idx]
    rows: list[dict] = []
    for family in _FEATURE_FAMILIES:
        fr = roles if family == "bandpower" else None
        tr = make_feature_transformer(family, sfreq=cell.sfreq, channel_roles=fr)
        tr.fit(Xtr_raw, ytr)
        Xtr, Xte = tr.transform(Xtr_raw), tr.transform(Xte_raw)
        for head in _HEADS:
            clf = make_classifier(head, random_state=seed)
            clf.fit(Xtr, ytr)
            m = compute_metrics(yte, clf.predict(Xte))
            rows.append(_row(task, montage, "fixed", cell, split, kappa=m.kappa,
                             acc=m.accuracy, feature_family=family, classifier=head, seed=-1))
    return rows


def _row(task, montage, method, cell, split, *, kappa, acc, feature_family="",
         classifier="", seed=-1, n_arms=0, regret=0.0, error="") -> dict:
    return {
        "dataset": "COGBCI", "task": task, "montage": montage, "method": method,
        "feature_family": feature_family, "classifier": classifier,
        "subject": cell.subject, "seed": seed,
        "kappa": float(kappa), "accuracy": float(acc),
        "n_train": int(len(split.train_idx)), "n_test": int(len(split.test_idx)),
        "n_channels": int(cell.n_channels), "n_arms_surviving": int(n_arms),
        "final_regret": float(regret), "error": error,
    }


def _parse_int(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_str(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main(
    task: str = typer.Option("nback", help="nback (leakage-clean headline) or matb (leaderboard anchor)."),
    montages: str = typer.Option("full,nearear", help="full, nearear."),
    subjects: str = typer.Option("all", help="Subject IDs (or 'all')."),
    seeds: str = typer.Option("0,1,2,3,42", help="CCB seeds."),
    alpha: float = typer.Option(0.5),
    calibration_frac: float = typer.Option(0.3),
    window_size: int = typer.Option(50),
    n_components: int = typer.Option(4),
    output: Path = typer.Option(Path("results/crosssession_cogbci.csv")),
) -> None:
    console = Console()
    montage_list = _parse_str(montages)
    seed_list = _parse_int(seeds)
    subj_filter = None if subjects == "all" else _parse_int(subjects)
    win = window_size if window_size > 0 else None
    stems = _TASK_STEMS[task]
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    done: set[tuple[str, str, int]] = set()
    if output.exists():
        prev = pd.read_csv(output).fillna("")
        rows = prev.to_dict("records")
        done = {
            (str(r["task"]), str(r["montage"]), int(r["subject"]))
            for r in rows if str(r.get("error", "")) == ""
        }
        console.log(f"Resuming: {len(done)} (task, montage, subject) cells done.")

    console.log(f"Loading COG-BCI {task} all 3 sessions subjects={subj_filter or 'all'} …")
    full_list = load_cogbci(subjects=subj_filter, sessions=("S1", "S2", "S3"), set_stems=stems)
    console.log(f"  → {len(full_list)} subjects.")

    for montage in montage_list:
        roles = COGBCI_CHANNEL_ROLES if montage == "full" else NEAR_EAR_ROLES
        console.rule(f"COG-BCI {task} · {montage} · cross-session S1→S2/S3")
        for data in full_list:
            if (task, montage, int(data.subject)) in done:
                continue
            cell = data if montage == "full" else select_near_ear(data, COGBCI_CANONICAL_EEG)
            try:
                split = cross_session_split(cell, train_sessions=["S1"], test_sessions=["S2", "S3"])
                # Fixed pipelines (deterministic).
                rows += _fixed_rows(cell, split, task=task, montage=montage, roles=roles, seed=42)
                # CCB (5 seeds).
                arms = enumerate_arms_generic(n_channels=cell.n_channels, n_components=n_components)
                for seed in seed_list:
                    cfg = OPLBConfig(alpha=alpha, lambda_reg=1.0, budget=float("inf"),
                                     window_size=win, discount_gamma=1.0, per_round_cap=None)
                    res = run_ccb_on_split(cell, split, arms=arms, config=cfg,
                                           calibration_frac=calibration_frac, seed=seed,
                                           include_recent_rewards=False, workload_channel_roles=roles)
                    rows.append(_row(task, montage, "ccb", cell, split, kappa=res.kappa,
                                     acc=res.accuracy, seed=seed, n_arms=res.n_arms_surviving,
                                     regret=float(res.cumulative_regret[-1]) if res.cumulative_regret.size else 0.0))
                console.log(f"  {montage} s{data.subject} done ({cell.n_channels}ch)")
            except Exception as exc:  # noqa: BLE001
                console.log(f"[red]  ✗ {task} {montage} s{data.subject}: {exc}[/red]")
                rows.append(_row(task, montage, "error", cell, type("S", (), {"train_idx": [], "test_idx": []})(),
                                 kappa=float("nan"), acc=float("nan"), error=str(exc)))
            pd.DataFrame(rows).to_csv(output, index=False)  # checkpoint per subject

    if not rows:
        console.print("[red]No rows generated.[/red]")
        raise typer.Exit(code=1)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    console.log(f"Saved {len(df)} rows → {output}")
    console.rule(f"Cross-session κ ({task}): best fixed vs CCB, per montage")
    valid = df[df.method.isin(["fixed", "ccb"])].dropna(subset=["kappa"])
    for montage in montage_list:
        sub = valid[valid.montage == montage]
        if sub.empty:
            continue
        best_fixed = sub[sub.method == "fixed"].groupby(["feature_family", "classifier"])["kappa"].mean().max()
        ccb_mean = sub[sub.method == "ccb"]["kappa"].mean()
        console.print(f"  {montage}: best fixed κ={best_fixed:.3f} | CCB κ={ccb_mean:.3f} | Δκ={best_fixed - ccb_mean:+.3f}")


if __name__ == "__main__":
    typer.run(main)
