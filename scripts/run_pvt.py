r"""COG-BCI PVT vigilance — near-ear pipeline on a second vital sign (Phase 6).

Runs the identical fixed-vs-CCB machinery on the COG-BCI PVT (vigilance) recordings,
target = within-session RT median split (high/low vigilance; see
``thesis.data.cogbci_pvt_load`` for the documented operational definition). Two
protocols, two montages:

  * within-session  : session S1 only, 5-fold within-subject CV ("is near-ear
    vigilance decodable at all?")
  * cross-session   : train S1 -> test S2/S3 ("does it survive drift, and does the
    CCB help?" — the deployment test, mirroring the N-back/MATB cells)

Fixed B1-B5 fit train-only; CCB calibrate+stream on train, frozen on test (5 seeds).
Near-ear = T7/T8 by position. Checkpointed per (protocol, montage, subject).

Output: ``results/pvt_vigilance.csv``.
"""

from __future__ import annotations

from dataclasses import replace
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
from thesis.data.cogbci_pvt_load import COGBCI_PVT_ROLES, load_cogbci_pvt
from thesis.data.near_ear import NEAR_EAR_ROLES
from thesis.metrics import compute_metrics
from thesis.protocols import cross_session_split, within_subject_cv

mne.set_log_level("WARNING")

_HEADS = ("lda", "svm", "decision_tree", "random_forest")
_FEATURE_FAMILIES = ("fbcsp", "bandpower")


def _row(protocol, montage, method, cell, split, *, kappa, acc, feature_family="",
         classifier="", seed=-1, n_arms=0, regret=0.0, error=""):
    return {
        "dataset": "COGBCI-PVT", "protocol": protocol, "montage": montage, "method": method,
        "feature_family": feature_family, "classifier": classifier,
        "subject": cell.subject, "seed": seed, "kappa": float(kappa), "accuracy": float(acc),
        "n_train": int(len(split.train_idx)), "n_test": int(len(split.test_idx)),
        "n_channels": int(cell.n_channels), "n_arms_surviving": int(n_arms),
        "final_regret": float(regret), "error": error,
    }


def _fixed_rows(protocol, cell, split, *, montage, roles):
    Xtr, ytr = cell.X[split.train_idx], cell.y[split.train_idx]
    Xte, yte = cell.X[split.test_idx], cell.y[split.test_idx]
    out = []
    for family in _FEATURE_FAMILIES:
        fr = roles if family == "bandpower" else None
        tr = make_feature_transformer(family, sfreq=cell.sfreq, channel_roles=fr)
        tr.fit(Xtr, ytr)
        Ztr, Zte = tr.transform(Xtr), tr.transform(Xte)
        for head in _HEADS:
            clf = make_classifier(head, random_state=42)
            clf.fit(Ztr, ytr)
            m = compute_metrics(yte, clf.predict(Zte))
            out.append(_row(protocol, montage, "fixed", cell, split, kappa=m.kappa,
                            acc=m.accuracy, feature_family=family, classifier=head))
    return out


def _ccb_rows(protocol, cell, split, *, montage, roles, seeds, alpha, calib, win):
    arms = enumerate_arms_generic(n_channels=cell.n_channels, n_components=min(4, cell.n_channels))
    out = []
    for seed in seeds:
        cfg = OPLBConfig(alpha=alpha, lambda_reg=1.0, budget=float("inf"),
                         window_size=win, discount_gamma=1.0, per_round_cap=None)
        res = run_ccb_on_split(cell, split, arms=arms, config=cfg, calibration_frac=calib,
                               seed=seed, include_recent_rewards=False, workload_channel_roles=roles)
        out.append(_row(protocol, montage, "ccb", cell, split, kappa=res.kappa, acc=res.accuracy,
                        seed=seed, n_arms=res.n_arms_surviving,
                        regret=float(res.cumulative_regret[-1]) if res.cumulative_regret.size else 0.0))
    return out


def _session_subset(cell, ses: str):
    mask = (cell.metadata["session"] == ses).to_numpy()
    return replace(cell, X=cell.X[mask], y=cell.y[mask], metadata=cell.metadata[mask].reset_index(drop=True))


def _parse_int(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main(
    montages: str = typer.Option("nearear,full"),
    protocols: str = typer.Option("within,cross"),
    subjects: str = typer.Option("all"),
    seeds: str = typer.Option("0,1,2,3,42"),
    alpha: float = typer.Option(0.5),
    calibration_frac: float = typer.Option(0.3),
    window_size: int = typer.Option(50),
    output: Path = typer.Option(Path("results/pvt_vigilance.csv")),
) -> None:
    console = Console()
    montage_list = [m.strip() for m in montages.split(",") if m.strip()]
    proto_list = [p.strip() for p in protocols.split(",") if p.strip()]
    seed_list = _parse_int(seeds)
    subj_filter = None if subjects == "all" else _parse_int(subjects)
    win = window_size if window_size > 0 else None
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    done: set[tuple[str, str, int]] = set()
    if output.exists():
        prev = pd.read_csv(output).fillna("")
        rows = prev.to_dict("records")
        done = {(str(r["protocol"]), str(r["montage"]), int(r["subject"]))
                for r in rows if str(r.get("error", "")) == ""}
        console.log(f"Resuming: {len(done)} cells done.")

    for montage in montage_list:
        roles = COGBCI_PVT_ROLES if montage == "full" else NEAR_EAR_ROLES
        console.log(f"Loading PVT ({montage}) subjects={subj_filter or 'all'} …")
        cells = load_cogbci_pvt(subjects=subj_filter, near_ear=(montage == "nearear"))
        console.log(f"  → {len(cells)} subjects.")
        for cell in cells:
            for proto in proto_list:
                if (proto, montage, int(cell.subject)) in done:
                    continue
                try:
                    if proto == "within":
                        s1 = _session_subset(cell, "S1")
                        split = list(within_subject_cv(s1, n_splits=5, seed=42))[0]
                        base = s1
                    else:
                        split = cross_session_split(cell, train_sessions=["S1"], test_sessions=["S2", "S3"])
                        base = cell
                    rows += _fixed_rows(proto, base, split, montage=montage, roles=roles)
                    rows += _ccb_rows(proto, base, split, montage=montage, roles=roles,
                                      seeds=seed_list, alpha=alpha, calib=calibration_frac, win=win)
                    console.log(f"  {proto} {montage} s{cell.subject} done ({cell.n_channels}ch)")
                except Exception as exc:  # noqa: BLE001
                    console.log(f"[red]  ✗ {proto} {montage} s{cell.subject}: {exc}[/red]")
                    rows.append(_row(proto, montage, "error",
                                     cell, type("S", (), {"train_idx": [], "test_idx": []})(),
                                     kappa=float("nan"), acc=float("nan"), error=str(exc)))
                pd.DataFrame(rows).to_csv(output, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    console.log(f"Saved {len(df)} rows → {output}")
    valid = df[df.method.isin(["fixed", "ccb"])].dropna(subset=["kappa"])
    for proto in proto_list:
        for montage in montage_list:
            sub = valid[(valid.protocol == proto) & (valid.montage == montage)]
            if sub.empty:
                continue
            bf = sub[sub.method == "fixed"].groupby(["feature_family", "classifier"]).kappa.mean().max()
            cc = sub[sub.method == "ccb"].kappa.mean()
            console.print(f"  {proto}/{montage}: best fixed κ={bf:.3f} | CCB κ={cc:.3f} | Δκ={bf-cc:+.3f}")


if __name__ == "__main__":
    typer.run(main)
