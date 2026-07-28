"""Leave-one-subject-out (LOSO) on STEW — the deferred within-segment-leakage test.

STEW's within-subject CV is structurally ceiling-saturated: the loader assigns
*one* workload bin to every epoch of a subject's 2.5-min rest segment and *one*
bin to every epoch of the multitask segment (``stew_load.py`` — the label is
constant within a continuous recording). Random K-fold CV therefore trains and
tests on epochs cut from the *same* two continuous segments, so a fixed pipeline
only has to recognise "which of my two segments is this window from" — a trivial
task that inflates κ to ≈ 0.94. This was flagged in the thesis as a deferred
sensitivity (Chapter 4, "Limitations").

LOSO removes the leakage structurally: pool every subject, hold one out for test,
train on the other 44. The held-out subject's segments are unseen at fit time, so
"segment recognition" is impossible and the reported κ measures genuine
cross-subject workload decoding — hard mode (``ccb-formulation.md`` §8.1,
protocol 3).

Three conditions, matching the STEW within-CV panel in Chapter 4:

  - B1 fixed: FBCSP + shrinkage LDA  (``thesis.baselines.fbcsp.FBCSP``)
  - B2 fixed: per-channel band-power + shrinkage LDA  (``BandPowerCL``)
  - CCB:      the OPLB contextual bandit (``run_ccb_on_split``)

The fixed baselines fit on the full pooled 44-subject training set (LDA/CSP fits
are fast and benefit from all data). The CCB is a *secondary* cross-subject probe:
its cross-subject training pool (~3.3k epochs) is ~55× the within-subject CCB's
stream (~40 trials), so for tractability — and to keep the CCB's data regime
comparable to its within-subject form — the per-fold training pool is subsampled
to ``--ccb-train-cap`` epochs before calibration/stream. This is a deliberate,
stated tractability choice, not a leakage concern (the subsample is drawn from
training subjects only). The CCB-under-LOSO is also a first empirical probe of the
Formulation-C "cross-subject transfer" direction (``ccb-formulation.md`` §10):
the policy + arm heads are population-trained, frozen, and tested on a new subject.

Two κ figures are reported per method:

  - **per-subject** mean ± std over the held-out subjects (directly comparable to
    the within-CV table format; degenerate single-class test subjects are
    excluded and counted);
  - **pooled** global κ over every held-out prediction (the robust LOSO headline —
    per-subject test sets are coarse, ≤ 2 classes × ~74 epochs).

Output: ``results/loso_stew.csv`` (one row per (method, subject, seed)).

Examples::

    # Full run: all 45 subjects, all three methods, CCB seed 42.
    PYTHONPATH=src .venv/bin/python scripts/run_loso_stew.py

    # Fast smoke on a handful of subjects.
    PYTHONPATH=src .venv/bin/python scripts/run_loso_stew.py \\
        --subjects 1,2,3,4 --output results/loso_stew_smoke.csv

    # Fixed baselines only (the decisive leakage test), skip the CCB.
    PYTHONPATH=src .venv/bin/python scripts/run_loso_stew.py --methods fbcsp,bandpower

Requires data/STEW/ populated; run ``make stew-check`` first.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import typer
from rich.console import Console

from thesis.baselines.bandpower_cl import BandPowerCL
from thesis.baselines.fbcsp import FBCSP
from thesis.ccb.arms import enumerate_arms_generic
from thesis.ccb.oplb import OPLBConfig
from thesis.ccb.runner import run_ccb_on_split
from thesis.data.stew_load import STEW_CHANNEL_ROLES, STEW_CHANNELS, load_stew
from thesis.matched import matched_loso
from thesis.metrics import compute_metrics
from thesis.protocols import Split

mne.set_log_level("WARNING")


def _parse_csv_int(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_str(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _n_classes(y: np.ndarray) -> int:
    return int(len(np.unique(y)))


def _fixed_fold(
    name: str, pooled, split: Split, *, channel_roles
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Fit one fixed baseline on the pooled train split, score the held-out subject."""
    if name == "fbcsp":
        clf = FBCSP(sfreq=pooled.sfreq)
    elif name == "bandpower":
        clf = BandPowerCL(sfreq=pooled.sfreq, channel_roles=channel_roles)
    else:
        raise ValueError(f"unknown fixed baseline {name!r}")
    clf.fit(pooled.X[split.train_idx], pooled.y[split.train_idx])
    y_true = pooled.y[split.test_idx]
    y_pred = clf.predict(pooled.X[split.test_idx])
    n_cls = _n_classes(y_true)
    # κ is ill-defined for a single-class test fold (held-out subject whose two
    # segments fall in the same workload bin). Mark NaN; the pooled κ still uses
    # these predictions, and the per-subject aggregate drops them.
    kappa = float(compute_metrics(y_true, y_pred).kappa) if n_cls >= 2 else float("nan")
    acc = float((y_true == y_pred).mean())
    return (
        {
            "kappa": kappa,
            "accuracy": acc,
            "n_train": int(len(split.train_idx)),
            "n_test": int(len(split.test_idx)),
            "n_classes_test": n_cls,
        },
        y_true,
        y_pred,
    )


def main(
    subjects: str = typer.Option(
        "all", help="Comma-separated STEW subject IDs (1..48, omitting 5/24/42), or 'all'."
    ),
    methods: str = typer.Option(
        "fbcsp,bandpower,ccb",
        help="Methods to run: comma-separated from 'fbcsp', 'bandpower', 'ccb'.",
    ),
    ccb_seeds: str = typer.Option(
        "42",
        help="Comma-separated CCB seeds (calibration/stream permutation). Default single seed.",
    ),
    train_cap: int = typer.Option(
        4000,
        help="Common per-fold training-pool cap applied IDENTICALLY to every method "
        "(0 = use all). STEW pools ~3.3k epochs across 45 subjects, under the cap, so all "
        "methods train on the full held-in pool; the cap is the shared matched-conditions "
        "value (src/thesis/matched.py) so STEW and WAUC use one policy.",
    ),
    alpha: float = typer.Option(0.5, help="OPLB α (matches within-CV STEW headline)."),
    calibration_frac: float = typer.Option(0.3, help="CCB calibration fraction."),
    window_size: int = typer.Option(50, help="OPLB sliding-window size (0 → stationary)."),
    n_components: int = typer.Option(4, help="CSP components per band."),
    use_workload_context: bool = typer.Option(
        True, help="Use the CL workload context (default ON; matches the within-CV headline)."
    ),
    reward_mode: str = typer.Option(
        "accuracy",
        help="CCB reward: 'accuracy' (0/1; default, the headline) or 'balanced' "
        "(inverse train-class-frequency weighted — the κ-aligned robustness check).",
    ),
    output: Path = typer.Option(Path("results/loso_stew.csv"), help="Output CSV path."),
) -> None:
    console = Console()
    subject_filter = None if subjects == "all" else _parse_csv_int(subjects)
    method_list = _parse_csv_str(methods)
    seed_list = _parse_csv_int(ccb_seeds)

    console.log("Loading STEW subjects...")
    stew = load_stew(subjects=subject_filter)
    if len(stew) < 2:
        console.print("[red]✗ LOSO needs ≥2 subjects. Check data/STEW/ and run `make stew-check`.[/red]")
        raise typer.Exit(code=1)
    console.log(f"Loaded {len(stew)} subjects → {len(stew)} LOSO folds.")

    n_channels = stew[0].X.shape[1]
    assert n_channels == len(STEW_CHANNELS), f"unexpected channel count {n_channels}"

    # Matched-conditions folds: train_idx is the common class-stratified train_cap
    # subsample (fixed split seed), shared identically by every method. STEW's pool is
    # under the cap, so this is the full held-in set for all methods.
    folds = list(matched_loso(stew, cap=train_cap))  # [(pooled_data, split)] per held-out subject
    rows: list[dict] = []
    # Accumulate (y_true, y_pred) per (method, seed) for the pooled global κ.
    pooled_preds: dict[tuple[str, int], list[tuple[np.ndarray, np.ndarray]]] = {}

    # ---- Fixed baselines (deterministic; one pass over folds) ----
    for name in [m for m in method_list if m in ("fbcsp", "bandpower")]:
        console.rule(f"LOSO · {name}")
        for fold_data, split in folds:
            sid = fold_data.subject
            try:
                rec, yt, yp = _fixed_fold(
                    name, fold_data, split, channel_roles=STEW_CHANNEL_ROLES
                )
                rows.append({"dataset": "STEW", "method": name, "subject": sid, "seed": -1, **rec,
                             "y_true_seq": ";".join(map(str, yt.tolist())),
                             "y_pred_seq": ";".join(map(str, yp.tolist()))})
                pooled_preds.setdefault((name, -1), []).append((yt, yp))
                console.log(f"  s{sid}: κ={rec['kappa']:.3f} acc={rec['accuracy']:.3f} (n_cls={rec['n_classes_test']})")
            except Exception as exc:  # noqa: BLE001
                console.log(f"[red]  ✗ s{sid} {name} failed: {exc}[/red]")
                rows.append({"dataset": "STEW", "method": name, "subject": sid, "seed": -1,
                             "kappa": float("nan"), "accuracy": float("nan"), "n_train": 0,
                             "n_test": 0, "n_classes_test": 0, "error": str(exc)})
        # Checkpoint after each method completes (crash-safe: a session resume /
        # kill mid-run no longer wipes finished work, since the script otherwise
        # writes the CSV only at the very end).
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
        console.log(f"[green]  ✓ checkpoint: {name} done — {len(rows)} rows → {output}[/green]")

    # ---- CCB (cross-subject probe; seed-dependent, stream-capped) ----
    if "ccb" in method_list:
        arms = enumerate_arms_generic(n_channels=n_channels, n_components=n_components)
        win = window_size if window_size > 0 else None
        for seed in seed_list:
            console.rule(f"LOSO · ccb (seed {seed}, train-cap {train_cap})")
            for fold_data, split in folds:
                sid = fold_data.subject
                cfg = OPLBConfig(alpha=alpha, lambda_reg=1.0, budget=float("inf"),
                                 window_size=win, discount_gamma=1.0, per_round_cap=None)
                # Same matched split as the fixed baselines; the CCB carves its
                # calibration/stream from this identical train pool using its own seed.
                try:
                    res = run_ccb_on_split(
                        fold_data, split, arms=arms, config=cfg,
                        calibration_frac=calibration_frac, seed=seed,
                        include_recent_rewards=False,
                        workload_channel_roles=STEW_CHANNEL_ROLES if use_workload_context else None,
                        reward_mode=reward_mode,
                    )
                    n_cls = _n_classes(res.y_true)
                    kappa = res.kappa if n_cls >= 2 else float("nan")
                    rows.append({"dataset": "STEW", "method": "ccb", "subject": sid, "seed": seed,
                                 "reward_mode": reward_mode,
                                 "kappa": float(kappa), "accuracy": float(res.accuracy),
                                 "n_train": int(len(split.train_idx)), "n_test": int(res.n_test),
                                 "n_classes_test": n_cls,
                                 "y_true_seq": ";".join(map(str, res.y_true.tolist())),
                                 "y_pred_seq": ";".join(map(str, res.y_pred.tolist()))})
                    pooled_preds.setdefault(("ccb", seed), []).append((res.y_true, res.y_pred))
                    console.log(f"  s{sid}: κ={kappa:.3f} acc={res.accuracy:.3f} (arms={res.n_arms_surviving}, n_cls={n_cls})")
                except Exception as exc:  # noqa: BLE001
                    console.log(f"[red]  ✗ s{sid} ccb seed{seed} failed: {exc}[/red]")
                    rows.append({"dataset": "STEW", "method": "ccb", "subject": sid, "seed": seed,
                                 "kappa": float("nan"), "accuracy": float("nan"), "n_train": 0,
                                 "n_test": 0, "n_classes_test": 0, "error": str(exc)})
            # Checkpoint after each seed completes (crash-safe).
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(output, index=False)
            console.log(f"[green]  ✓ checkpoint: ccb seed {seed} done — {len(rows)} rows → {output}[/green]")

    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    console.log(f"Saved {len(df)} rows → {output}")

    # ---- Summary ----
    console.rule("LOSO-on-STEW — per-subject mean ± std κ (degenerate folds excluded)")
    valid = df.dropna(subset=["kappa"])
    per_subj = valid.groupby("method")["kappa"].agg(["mean", "std", "count"]).round(4)
    console.print(per_subj)

    console.rule("LOSO-on-STEW — pooled global κ (all held-out predictions)")
    for (method, seed), pairs in sorted(pooled_preds.items()):
        yt = np.concatenate([p[0] for p in pairs])
        yp = np.concatenate([p[1] for p in pairs])
        m = compute_metrics(yt, yp)
        tag = method if seed == -1 else f"{method}[seed{seed}]"
        console.print(f"  {tag:18s} pooled κ = {m.kappa:.4f}  acc = {m.accuracy:.4f}  (n = {m.n_trials})")

    console.print()
    console.print("[bold]Within-CV reference (leakage-inflated, from Chapter 4 / results):[/bold]")
    console.print("  B1 FBCSP+sLDA   within-CV κ = 0.937 ± 0.105")
    console.print("  B2 BandPower    within-CV κ = 0.953 ± 0.092")
    console.print("  CCB             within-CV κ = 0.744 ± 0.170")
    console.print("[dim]LOSO κ far below within-CV κ ⇒ the within-CV number was segment-leakage.[/dim]")


if __name__ == "__main__":
    typer.run(main)
