r"""COG-BCI MATB *competition split* cross-session — the leaderboard-comparable anchor.

Train session S1 → test session S2 (S3 MATB is withheld in the hackathon split), on
the pre-epoched MATB-II competition data (Zenodo 5055046; loader
``thesis.data.cogbci_matb_load``). Full 61-channel montage (directly comparable to the
published passive-BCI leaderboard, which tops out < 60 % 3-class accuracy under
calibration-permitted conditions) and the T7/T8 near-ear subset (the deployment cell).

Same eval as the 3-session cross-session runner: fixed pipelines B1–B5 fitted on S1 and
scored on S2; CCB calibrated + streamed on S1, scored frozen on S2 (5 seeds). No
domain adaptation. No-leakage: feature fits train-only, near-ear by electrode position.
Checkpointed per (montage, subject); resumes.

Output: ``results/crosssession_matb_competition.csv``.
"""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd
import typer
from rich.console import Console

from thesis.baselines.classical import make_classifier, make_feature_transformer
from thesis.ccb.arms import enumerate_arms_generic
from thesis.ccb.oplb import OPLBConfig
from thesis.ccb.runner import run_ccb_on_split
from thesis.data.cogbci_matb_load import COGBCI_MATBCOMP_ROLES, load_cogbci_matb
from thesis.data.near_ear import NEAR_EAR_ROLES
from thesis.metrics import compute_metrics
from thesis.protocols import cross_session_split

mne.set_log_level("WARNING")

_HEADS = ("lda", "svm", "decision_tree", "random_forest")
_FEATURE_FAMILIES = ("fbcsp", "bandpower")


def _row(montage, method, cell, split, *, kappa, acc, feature_family="", classifier="",
         seed=-1, n_arms=0, regret=0.0, error=""):
    return {
        "dataset": "COGBCI-MATBcomp", "task": "matb-competition", "montage": montage,
        "method": method, "feature_family": feature_family, "classifier": classifier,
        "subject": cell.subject, "seed": seed, "kappa": float(kappa), "accuracy": float(acc),
        "n_train": int(len(split.train_idx)), "n_test": int(len(split.test_idx)),
        "n_channels": int(cell.n_channels), "n_arms_surviving": int(n_arms),
        "final_regret": float(regret), "error": error,
    }


def _fixed_rows(cell, split, *, montage, roles):
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
            out.append(_row(montage, "fixed", cell, split, kappa=m.kappa, acc=m.accuracy,
                            feature_family=family, classifier=head))
    return out


def _parse_int(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main(
    montages: str = typer.Option("full,nearear"),
    subjects: str = typer.Option("all"),
    seeds: str = typer.Option("0,1,2,3,42"),
    alpha: float = typer.Option(0.5),
    calibration_frac: float = typer.Option(0.3),
    window_size: int = typer.Option(50),
    output: Path = typer.Option(Path("results/crosssession_matb_competition.csv")),
) -> None:
    console = Console()
    montage_list = [m.strip() for m in montages.split(",") if m.strip()]
    seed_list = _parse_int(seeds)
    subj_filter = None if subjects == "all" else _parse_int(subjects)
    win = window_size if window_size > 0 else None
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
        roles = COGBCI_MATBCOMP_ROLES if montage == "full" else NEAR_EAR_ROLES
        console.rule(f"MATB competition · {montage} · cross-session S1→S2")
        console.log(f"Loading MATB competition ({montage}) subjects={subj_filter or 'all'} …")
        cells = load_cogbci_matb(subjects=subj_filter, near_ear=(montage == "nearear"))
        for cell in cells:
            if (montage, int(cell.subject)) in done:
                continue
            try:
                split = cross_session_split(cell, train_sessions=["S1"], test_sessions=["S2"])
                rows += _fixed_rows(cell, split, montage=montage, roles=roles)
                arms = enumerate_arms_generic(n_channels=cell.n_channels, n_components=4)
                for seed in seed_list:
                    cfg = OPLBConfig(alpha=alpha, lambda_reg=1.0, budget=float("inf"),
                                     window_size=win, discount_gamma=1.0, per_round_cap=None)
                    res = run_ccb_on_split(cell, split, arms=arms, config=cfg,
                                           calibration_frac=calibration_frac, seed=seed,
                                           include_recent_rewards=False, workload_channel_roles=roles)
                    rows.append(_row(montage, "ccb", cell, split, kappa=res.kappa, acc=res.accuracy,
                                     seed=seed, n_arms=res.n_arms_surviving,
                                     regret=float(res.cumulative_regret[-1]) if res.cumulative_regret.size else 0.0))
                console.log(f"  {montage} s{cell.subject} done ({cell.n_channels}ch)")
            except Exception as exc:  # noqa: BLE001
                console.log(f"[red]  ✗ {montage} s{cell.subject}: {exc}[/red]")
                rows.append(_row(montage, "error", cell,
                                 type("S", (), {"train_idx": [], "test_idx": []})(),
                                 kappa=float("nan"), acc=float("nan"), error=str(exc)))
            pd.DataFrame(rows).to_csv(output, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    console.log(f"Saved {len(df)} rows → {output}")
    valid = df[df.method.isin(["fixed", "ccb"])].dropna(subset=["kappa"])
    for montage in montage_list:
        sub = valid[valid.montage == montage]
        if sub.empty:
            continue
        bf = sub[sub.method == "fixed"].groupby(["feature_family", "classifier"]).kappa.mean().max()
        cc = sub[sub.method == "ccb"].kappa.mean()
        console.print(f"  {montage}: best fixed κ={bf:.3f} | CCB κ={cc:.3f} | Δκ={bf-cc:+.3f}")


if __name__ == "__main__":
    typer.run(main)
