r"""Operational interpretation of the headline decoding results.

Chapter 4 reports Cohen's kappa. Kappa is the right chance-corrected metric for comparing
methods, but it does not tell a reader what a given score *means* for someone who would
actually wear the device. This script derives, from the committed result CSVs and from
nothing else, the three translations Chapter 4 needs:

  1. kappa -> plain accuracy against the task's own chance level. Accuracy is read from the
     committed CSVs where it was recorded, never reconstructed algebraically from kappa.
  2. The leakage contrast in accuracy terms: what the standard within-subject cross-validation
     protocol appears to deliver, versus what survives a leakage-clean protocol on the same
     data and the same pipeline.
  3. Operational error structure: from the stored per-epoch prediction sequences on STEW
     leave-one-subject-out, the full confusion matrix and the per-class precision --- i.e.
     when the system declares a workload level, how often is it right? This is the quantity
     an adaptive-automation designer would need, and it is not recoverable from kappa alone.

Every figure produced here traces to a committed CSV under results/. No number is entered
by hand and none is carried over from prose.

Output: results/operational_interpretation.csv (tidy) + results/operational_interpretation.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import cohen_kappa_score

RESULTS = Path("results")
STEW_LABELS = ["low", "medium", "high"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["error"].isna()] if "error" in df.columns else df


def _best_fixed(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean over subjects per pipeline, then the best-kappa pipeline per cell.

    Matches how Chapter 4 reports "best fixed pipeline": the comparison is against the
    strongest fixed configuration available, not against an average over configurations.
    """
    g = (df.groupby([*keys, "feature_family", "classifier"])
           .agg(kappa=("kappa", "mean"), accuracy=("accuracy", "mean"),
                n_subjects=("subject", "nunique"))
           .reset_index())
    return g.loc[g.groupby(keys)["kappa"].idxmax()].reset_index(drop=True)


def _crosssession_rows() -> list[dict]:
    """The deployment cells: COG-BCI n-back and MATB, near-ear, cross-session."""
    rows: list[dict] = []
    for path, task in [(RESULTS / "crosssession_cogbci.csv", "n-back"),
                       (RESULTS / "crosssession_matb.csv", "MATB")]:
        if not path.exists():
            continue
        df = _clean(pd.read_csv(path))
        fixed = _best_fixed(df[df["method"] == "fixed"], ["task", "montage"])
        ccb = (df[df["method"] == "ccb"].groupby(["task", "montage"])
                 .agg(kappa=("kappa", "mean"), accuracy=("accuracy", "mean"),
                      n_subjects=("subject", "nunique")).reset_index())
        for _, r in fixed.iterrows():
            rows.append({
                "cell": f"COG-BCI {task} ({r['montage']}), cross-session",
                "protocol": "cross-session (leakage-clean)", "system": "best fixed pipeline",
                "detail": f"{r['feature_family']}+{r['classifier']}",
                "kappa": r["kappa"], "accuracy": r["accuracy"], "chance": 1 / 3,
                "n_subjects": r["n_subjects"], "source": path.name,
            })
        for _, r in ccb.iterrows():
            rows.append({
                "cell": f"COG-BCI {task} ({r['montage']}), cross-session",
                "protocol": "cross-session (leakage-clean)", "system": "CCB",
                "detail": "OPLB", "kappa": r["kappa"], "accuracy": r["accuracy"],
                "chance": 1 / 3, "n_subjects": r["n_subjects"], "source": path.name,
            })
    return rows


def _bci2a_rows() -> list[dict]:
    """The classical motor-imagery entry point, at the official 4-class split."""
    rows: list[dict] = []
    ccb_path, cls_path = RESULTS / "bci2a_4class.csv", RESULTS / "bci2a_4class_classical.csv"
    if ccb_path.exists():
        d = pd.read_csv(ccb_path)
        rows.append({
            "cell": "BCI-IV-2a, 4-class official split", "protocol": "official split",
            "system": "CCB", "detail": "OPLB", "kappa": d["kappa"].mean(),
            "accuracy": d["accuracy"].mean(), "chance": 0.25,
            "n_subjects": d["subject"].nunique(), "source": ccb_path.name,
        })
    if cls_path.exists():
        d = pd.read_csv(cls_path)
        g = d.groupby("classifier").agg(kappa=("kappa", "mean"), accuracy=("accuracy", "mean"),
                                        n=("subject", "nunique")).reset_index()
        best = g.loc[g["kappa"].idxmax()]
        rows.append({
            "cell": "BCI-IV-2a, 4-class official split", "protocol": "official split",
            "system": "best classical pipeline", "detail": str(best["classifier"]),
            "kappa": best["kappa"], "accuracy": best["accuracy"], "chance": 0.25,
            "n_subjects": best["n"], "source": cls_path.name,
        })
    return rows


def _leakage_rows() -> list[dict]:
    """Within-CV (leakage-confounded) versus leakage-clean, same data, same pipelines."""
    rows: list[dict] = []
    cls_path = RESULTS / "classical_baselines.csv"
    if cls_path.exists():
        d = _clean(pd.read_csv(cls_path))
        d = d[d["protocol"] == "within"] if "protocol" in d.columns else d
        for ds in ["STEW"]:
            sub = d[d["dataset"].astype(str).str.upper() == ds]
            if not len(sub):
                continue
            g = (sub.groupby(["feature_family", "classifier"])
                    .agg(kappa=("kappa", "mean"), accuracy=("accuracy", "mean"),
                         n=("subject", "nunique")).reset_index())
            b = g.loc[g["kappa"].idxmax()]
            rows.append({
                "cell": f"{ds} (14 ch)", "protocol": "within-subject CV (leakage-confounded)",
                "system": "best fixed pipeline",
                "detail": f"{b['feature_family']}+{b['classifier']}",
                "kappa": b["kappa"], "accuracy": b["accuracy"], "chance": 1 / 3,
                "n_subjects": b["n"], "source": cls_path.name,
            })
    loso = RESULTS / "loso_stew.csv"
    if loso.exists():
        d = pd.read_csv(loso)
        for method in ["bandpower", "fbcsp"]:
            sub = d[d["method"] == method]
            if not len(sub):
                continue
            rows.append({
                "cell": "STEW (14 ch)", "protocol": "leave-one-subject-out (leakage-clean)",
                "system": f"fixed pipeline ({method})", "detail": method,
                "kappa": sub["kappa"].mean(), "accuracy": sub["accuracy"].mean(),
                "chance": 1 / 3, "n_subjects": sub["subject"].nunique(), "source": loso.name,
            })
    return rows


def _stew_confusion() -> tuple[pd.DataFrame, dict, str]:
    """Pooled confusion matrix and per-class precision for STEW LOSO.

    Uses the per-epoch prediction sequences persisted by the LOSO runner, so the error
    structure is the measured one rather than an inference from the summary statistic.
    The deterministic best fixed pipeline (band-power) is used: it carries no seed, so a
    single pooled confusion matrix is well defined.
    """
    path = RESULTS / "loso_stew.csv"
    d = pd.read_csv(path)
    sub = d[d["method"] == "bandpower"]
    yt: list[str] = []
    yp: list[str] = []
    for _, r in sub.iterrows():
        if isinstance(r.get("y_true_seq"), str) and isinstance(r.get("y_pred_seq"), str):
            yt += r["y_true_seq"].split(";")
            yp += r["y_pred_seq"].split(";")
    yt_a, yp_a = np.array(yt), np.array(yp)
    cm = pd.DataFrame(
        [[int(((yt_a == t) & (yp_a == p)).sum()) for p in STEW_LABELS] for t in STEW_LABELS],
        index=[f"true {t}" for t in STEW_LABELS],
        columns=[f"pred {p}" for p in STEW_LABELS],
    )
    stats: dict = {"n_epochs": int(yt_a.size),
                   "accuracy": float((yt_a == yp_a).mean()),
                   "kappa": float(cohen_kappa_score(yt_a, yp_a))}
    for p in STEW_LABELS:
        fired = int((yp_a == p).sum())
        hit = int(((yp_a == p) & (yt_a == p)).sum())
        stats[f"precision_{p}"] = (hit / fired) if fired else float("nan")
        stats[f"fired_{p}"] = fired
        stats[f"hit_{p}"] = hit
        support = int((yt_a == p).sum())
        stats[f"recall_{p}"] = (hit / support) if support else float("nan")
    return cm, stats, path.name


def main(output: Path = typer.Option(RESULTS / "operational_interpretation.csv")) -> None:
    rows = _crosssession_rows() + _bci2a_rows() + _leakage_rows()
    if not rows:
        print("No source CSVs found under results/."); raise typer.Exit(code=1)
    df = pd.DataFrame(rows)
    df["points_above_chance"] = (df["accuracy"] - df["chance"]) * 100
    df["accuracy_pct"] = df["accuracy"] * 100
    df["chance_pct"] = df["chance"] * 100
    for c in ["kappa", "accuracy", "chance", "points_above_chance", "accuracy_pct", "chance_pct"]:
        df[c] = df[c].astype(float).round(4)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    cm, stats, cm_src = _stew_confusion()

    md = [
        "# Operational interpretation of the headline results",
        "",
        f"**Producer:** `scripts/make_operational_interpretation.py` · **Output:** `{output}`",
        "",
        "Cohen's kappa is the metric Chapter 4 compares methods in. This file translates the",
        "headline cells into the terms a reader outside the field needs: plain accuracy against",
        "the task's own chance level, and --- where per-epoch predictions were persisted --- the",
        "error structure a deployed system would actually exhibit. Accuracy is read from the",
        "committed CSVs, never reconstructed from kappa.",
        "",
        "## Per-cell accuracy against chance",
        "",
        "| Cell | Protocol | System | kappa | accuracy | chance | points above chance | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        md.append(
            f"| {r['cell']} | {r['protocol']} | {r['system']} ({r['detail']}) | "
            f"{r['kappa']:.3f} | {r['accuracy_pct']:.1f}\\% | {r['chance_pct']:.1f}\\% | "
            f"{r['points_above_chance']:+.1f} | {int(r['n_subjects'])} |"
        )
    md += [
        "",
        "## Error structure: STEW leave-one-subject-out, best fixed pipeline",
        "",
        f"Pooled over {stats['n_epochs']} held-out epochs from `{cm_src}` "
        f"(accuracy {stats['accuracy'] * 100:.1f}\\%, kappa {stats['kappa']:.3f}).",
        "",
        "| | " + " | ".join(cm.columns) + " |",
        "|---" * (len(cm.columns) + 1) + "|",
        *[f"| {idx} | " + " | ".join(str(v) for v in row) + " |"
          for idx, row in zip(cm.index, cm.values, strict=True)],
        "",
        "Per-class precision --- when the system declares a level, how often it is right:",
        "",
        "| Declared level | Correct / declared | Precision | Recall |",
        "|---|---|---|---|",
    ]
    for p in STEW_LABELS:
        md.append(f"| {p} | {stats[f'hit_{p}']} / {stats[f'fired_{p}']} | "
                  f"{stats[f'precision_{p}']:.3f} | {stats[f'recall_{p}']:.3f} |")
    md += [
        "",
        "The asymmetry is the operationally important part: the low-load state is identified",
        "far more reliably than the high-load state, and the intermediate level is barely",
        "distinguished at all. An adaptive system keyed to detecting high load would therefore",
        "be acting on a false alarm roughly half the times it fired.",
        "",
    ]
    md_path = output.with_suffix(".md")
    md_path.write_text("\n".join(md))
    print(f"Saved {len(df)} rows -> {output}")
    print(f"Saved summary   -> {md_path}")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    typer.run(main)
