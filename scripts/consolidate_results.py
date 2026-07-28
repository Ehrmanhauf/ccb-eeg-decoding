r"""Build the unified master results table from the committed CSVs.

Consolidates every (dataset, montage, protocol) cell of the panel into ONE table:
best fixed pipeline (max over B1--B5), CCB, EEGNet, and the published benchmark, in the
benchmark's own metric, with channel count, subject count, and leakage status. Numbers are
read from ``results/*.csv`` --- never hand-copied --- so the master table cannot drift from
the detailed per-paradigm tables (the discrepancy the advisor flagged). Reuses the verified
readers in ``scripts/make_tables.py``.

Outputs:
  - ``results/results_master.csv``  (flat, one row per cell --- analysis/figures)
  - prints the LaTeX ``tab:master`` body to stdout (paste into Chapter 4)

Run:  PYTHONPATH=src .venv/bin/python scripts/consolidate_results.py
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.metrics import compute_metrics


def _round3(x: float) -> float:
    """Round to 3 decimals half-up from the 4-decimal value, matching the convention used in
    the hand-set thesis tables (e.g. STEW's 0.9525 -> 0.953), rather than the float-floor /
    round-half-to-even that ``%.3f`` and NumPy apply. Keeps the master table consistent with
    the per-paradigm tables at rounding boundaries."""
    return float(Decimal(repr(round(float(x), 4))).quantize(Decimal("0.001"), ROUND_HALF_UP))

_RES = Path(__file__).resolve().parents[1] / "results"


def _csv(name: str) -> pd.DataFrame | None:
    p = _RES / name
    return pd.read_csv(p) if p.exists() else None


def _best_fixed_within_cl(ds: str) -> float:
    """Max within-CV kappa over EVERY fixed baseline for a CL set: the SVM/DT/RF heads in
    classical_baselines.csv AND the B1/B2 shrinkage-LDA pipelines in fixed_baseline_cl.csv."""
    cands: list[float] = []
    c = _csv("classical_baselines.csv")
    if c is not None:
        cc = c[(c.dataset == ds) & (c.protocol == "within")].dropna(subset=["kappa"])
        if not cc.empty:
            cands.append(cc.groupby(["feature_family", "classifier"]).kappa.mean().max())
    f = _csv("fixed_baseline_cl.csv")
    if f is not None:
        ff = f[f.dataset == ds].dropna(subset=["kappa"])
        if not ff.empty:
            cands.append(float(ff.groupby("baseline").kappa.mean().max()))
    return float(max(cands))


def _best_fixed_within_mi(ds: str) -> float:
    """Max within-CV kappa over FBCSP x heads for an MI set (classical_baselines)."""
    d = _csv("classical_baselines.csv")
    d = d[(d.dataset == ds) & (d.protocol == "within")].dropna(subset=["kappa"])
    if d.empty:
        return float("nan")
    return float(d.groupby(["feature_family", "classifier"]).kappa.mean().max())


def _best_fixed_newdata(ds: str, montage: str) -> float:
    d = _csv("fixed_baseline_newdata.csv").dropna(subset=["kappa"])
    d = d[(d.dataset == ds) & (d.montage == montage)]
    return float(d.groupby(["feature_family", "classifier"]).kappa.mean().max())


def _ccb_within(csv: str, **filt) -> float:
    d = _csv(csv)
    if d is None:
        return float("nan")
    for k, v in filt.items():
        if k in d.columns:
            d = d[d[k] == v]
    return float(d.kappa.mean())


def _ccb_newdata(ds: str, montage: str) -> float:
    d = _csv("ccb_newdata.csv")
    return float(d[(d.dataset == ds) & (d.montage == montage)].kappa.mean())


def _loso_pooled(csv: str, method: str) -> float:
    """Pooled global kappa over all held-out predictions for one method (loso_*.csv)."""
    d = _csv(csv)
    d = d[(d.method == method)].copy()
    d = d[d.y_true_seq.astype(str) != ""]
    yt = np.concatenate([np.array(str(s).split(";")) for s in d.y_true_seq])
    yp = np.concatenate([np.array(str(s).split(";")) for s in d.y_pred_seq])
    return float(compute_metrics(yt, yp).kappa)


def _best_fixed_loso(csv: str) -> float:
    return max(_loso_pooled(csv, "fbcsp"), _loso_pooled(csv, "bandpower"))


def _cross(csv: str, task: str, montage: str, method: str) -> float:
    """Mean kappa for a cross-session cell (crosssession_*.csv)."""
    d = _csv(csv)
    if d is None:
        return float("nan")
    d = d[(d.task == task) & (d.montage == montage)].dropna(subset=["kappa"])
    if method == "fixed":  # fixed rows carry method=="fixed" with feature_family in {fbcsp,bandpower}
        f = d[d.method == "fixed"]
        return float(f.groupby(["feature_family", "classifier"]).kappa.mean().max())
    return float(d[d.method == method].kappa.mean())


def _eegnet_pooled_loso(csv: str, dataset_contains: str) -> float:
    d = _csv(csv)
    if d is None:
        return float("nan")
    d = d[(d.protocol == "loso") & (d.dataset.astype(str).str.contains(dataset_contains, case=False))]
    d = d[d.y_true_seq.astype(str) != ""]
    if d.empty:
        return float("nan")
    ks = []
    for _, g in d.groupby("seed"):
        yt = np.concatenate([np.array(str(s).split("|")) for s in g.y_true_seq])
        yp = np.concatenate([np.array(str(s).split("|")) for s in g.y_pred_seq])
        ks.append(compute_metrics(yt, yp).kappa)
    return float(np.mean(ks))


def _eegnet_mean(csv: str, **filt) -> float:
    d = _csv(csv)
    if d is None:
        return float("nan")
    for k, v in filt.items():
        if k in d.columns:
            d = d[d[k].astype(str).str.contains(str(v), case=False) if k == "dataset" else d[k] == v]
    d = d.dropna(subset=["kappa"])
    return float(d.kappa.mean()) if not d.empty else float("nan")


def _eegnet_summary(dataset: str, protocol: str, montage: str, task: str | None = None) -> float:
    """Per-cell EEGNet kappa from the committed aggregation (scripts/summarize_eegnet.py ->
    results/eegnet_summary.csv). The canonical source for the cross-session and MI within-CV
    EEGNet cells, so the master table cannot drift from the dedicated EEGNet table."""
    d = _csv("eegnet_summary.csv")
    if d is None:
        return float("nan")
    m = (d.dataset == dataset) & (d.protocol == protocol) & (d.montage == montage)
    if task is not None:
        m &= d.task.astype(str) == task
    r = d[m]
    return float(r.kappa_mean.iloc[0]) if len(r) else float("nan")


NAN = float("nan")


def cells() -> list[dict]:
    """One dict per master-table row. kappa fields NaN where a method/benchmark is absent."""
    return [
        # dataset, paradigm, ch, n, montage, protocol, leak(True=confounded), fixed, ccb, eegnet, published, pub_note
        {"ds": "STEW", "par": "CL", "ch": 14, "n": 45, "mon": "full", "prot": "within-CV", "leak": True,
             "fixed": _best_fixed_within_cl("STEW"), "ccb": _ccb_within("ccb_stew_workload.csv"),
             "eeg": _eegnet_mean("eegnet_clloso.csv", dataset="STEW", protocol="within"),
             "pub": "0.46", "pubnote": "Lim SVR+NCA (within-CV)"},
        {"ds": "STEW", "par": "CL", "ch": 14, "n": 45, "mon": "full", "prot": "LOSO", "leak": False,
             "fixed": _best_fixed_loso("loso_stew.csv"), "ccb": _loso_pooled("loso_stew.csv", "ccb"),
             "eeg": _eegnet_pooled_loso("eegnet_clloso.csv", "STEW"), "pub": "--", "pubnote": ""},
        {"ds": "WAUC", "par": "CL", "ch": 8, "n": 45, "mon": "full", "prot": "within-CV", "leak": True,
             "fixed": _best_fixed_within_cl("WAUC"), "ccb": _ccb_within("ccb_wauc_workload.csv"),
             "eeg": NAN,  # EEGNet WAUC within-CV not run (only LOSO); within-CV is the leak exhibit
             "pub": "--", "pubnote": "no published decoding benchmark"},
        {"ds": "WAUC", "par": "CL", "ch": 8, "n": 43, "mon": "full", "prot": "LOSO", "leak": False,
             "fixed": _best_fixed_loso("loso_wauc.csv"), "ccb": _loso_pooled("loso_wauc.csv", "ccb"),
             "eeg": _eegnet_pooled_loso("eegnet_wauc.csv", "WAUC"), "pub": "--", "pubnote": ""},
        {"ds": "UAB", "par": "CL", "ch": 14, "n": 16, "mon": "full", "prot": "within-CV", "leak": True,
             "fixed": _best_fixed_newdata("UAB", "full"), "ccb": _ccb_newdata("UAB", "full"),
             "eeg": _eegnet_mean("eegnet_newdata.csv", dataset="UAB", montage="full", protocol="within"),
             "pub": "--", "pubnote": ""},
        {"ds": "UAB near-ear", "par": "CL", "ch": 2, "n": 16, "mon": "T7/T8", "prot": "within-CV", "leak": True,
             "fixed": _best_fixed_newdata("UAB", "nearear"), "ccb": _ccb_newdata("UAB", "nearear"),
             "eeg": _eegnet_mean("eegnet_newdata.csv", dataset="UAB", montage="nearear", protocol="within"),
             "pub": "--", "pubnote": ""},
        {"ds": "COG-BCI n-back", "par": "CL", "ch": 62, "n": 29, "mon": "full", "prot": "cross-session", "leak": False,
             "fixed": _cross("crosssession_cogbci.csv", "nback", "full", "fixed"),
             "ccb": _cross("crosssession_cogbci.csv", "nback", "full", "ccb"),
             "eeg": _eegnet_summary("COGBCI", "cross", "full", task="nback"),
             "pub": "0.65", "pubnote": "Hinss MDM (5-fold)"},
        {"ds": "COG-BCI n-back near-ear", "par": "CL", "ch": 2, "n": 29, "mon": "T7/T8", "prot": "cross-session", "leak": False,
             "fixed": _cross("crosssession_cogbci.csv", "nback", "nearear", "fixed"),
             "ccb": _cross("crosssession_cogbci.csv", "nback", "nearear", "ccb"),
             "eeg": _eegnet_summary("COGBCI", "cross", "nearear", task="nback"),
             "pub": "--", "pubnote": ""},
        {"ds": "COG-BCI MATB (compet.)", "par": "CL", "ch": 61, "n": 15, "mon": "full", "prot": "cross-session", "leak": False,
             "fixed": _cross("crosssession_matb_competition.csv", "matb-competition", "full", "fixed"),
             "ccb": _cross("crosssession_matb_competition.csv", "matb-competition", "full", "ccb"),
             "eeg": _eegnet_summary("COGBCI", "cross", "full", task="matb-competition"),
             "pub": "0.54", "pubnote": "Roy 2022 leaderboard (acc.); competition split"},
        {"ds": "COG-BCI MATB near-ear", "par": "CL", "ch": 2, "n": 29, "mon": "T7/T8", "prot": "cross-session", "leak": False,
             "fixed": _cross("crosssession_matb.csv", "matb", "nearear", "fixed"),
             "ccb": _cross("crosssession_matb.csv", "matb", "nearear", "ccb"),
             "eeg": _eegnet_summary("COGBCI", "cross", "nearear", task="matb"),
             "pub": "--", "pubnote": "no-adaptation near-ear"},
        {"ds": "BCI-IV-2a", "par": "MI", "ch": 22, "n": 9, "mon": "full", "prot": "within-CV (2-cl)", "leak": False,
             "fixed": _best_fixed_within_mi("BCI-IV-2a"), "ccb": _ccb_within("ccb_2a.csv", protocol="within"),
             "eeg": NAN,  # EEGNet 2a was run on the 4-class official split, not this 2-class within-CV cell
             "pub": "0.569", "pubnote": "Ang FBCSP (4-cl official)"},
        {"ds": "BCI-IV-2b", "par": "MI", "ch": 3, "n": 9, "mon": "full", "prot": "within-CV", "leak": False,
             "fixed": _best_fixed_within_mi("BCI-IV-2b"),
             # Locked best-factorial cell has recent-reward DISABLED (see tab:ccb-hparams); without
             # this constraint the include_recent_rewards=True factorial seeds dilute 0.184 -> 0.183.
             "ccb": _ccb_within("ccb_factorial.csv", alpha=0.5, calibration_frac=0.3, protocol="within", include_recent_rewards=False),
             "eeg": _eegnet_summary("BCI-IV-2b", "within", "full"),
             "pub": "0.600", "pubnote": "Ang FBCSP"},
        {"ds": "Cho2017 full", "par": "MI", "ch": 64, "n": 50, "mon": "full", "prot": "within-CV", "leak": False,
             "fixed": _best_fixed_within_mi("Cho2017-full"), "ccb": _ccb_within("ccb_cho2017_full.csv"),
             "eeg": NAN,  # Cho2017 EEGNet not reported: MI decoding established on full-cohort 2a/2b; the 2-of-50 pilot is not cohort-comparable
             "pub": "0.675", "pubnote": "Cho CSP+FLDA (acc.)"},
        {"ds": "Cho2017 3-ch", "par": "MI", "ch": 3, "n": 50, "mon": "C3/Cz/C4", "prot": "within-CV", "leak": False,
             "fixed": _best_fixed_within_mi("Cho2017-3ch"), "ccb": _ccb_within("ccb_cho2017_3ch.csv"),
             "eeg": NAN,  # EEGNet Cho2017 run on the 64-ch full montage only, not the 3-ch subset
             "pub": "--", "pubnote": ""},
    ]


def _fmt(x) -> str:
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    return f"{_round3(x):.3f}"


def main() -> None:
    rows = cells()
    df = pd.DataFrame(rows)
    df["delta_ccb"] = df.apply(
        lambda r: (r.fixed - r.ccb) if not (np.isnan(r.fixed) or np.isnan(r.ccb)) else np.nan, axis=1)
    out = _RES / "results_master.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} cells)\n")

    # LaTeX body
    print(r"% --- tab:master body (generated by scripts/consolidate_results.py) ---")
    for r in rows:
        leak = r"$^{\dagger}$" if r["leak"] else ""
        dk = r["fixed"] - r["ccb"] if not (np.isnan(r["fixed"]) or np.isnan(r["ccb"])) else NAN
        print(f'{r["ds"]} & {r["par"]} & {r["ch"]} & {r["n"]} & {r["prot"]}{leak} & '
              f'{_fmt(r["fixed"])} & {_fmt(r["ccb"])} & {_fmt(r["eeg"])} & {_fmt(r["pub"])} & {_fmt(dk)} \\\\')

    # self-check against known thesis values (catch reader bugs before trusting the table)
    print("\n=== self-check vs known thesis values (|Δ|>0.005 flagged) ===")
    known = {("STEW", "within-CV"): (0.953, 0.744), ("WAUC", "within-CV"): (0.785, 0.426),
             ("STEW", "LOSO"): (0.314, 0.243), ("UAB", "within-CV"): (0.963, 0.776),
             ("BCI-IV-2b", "within-CV"): (0.292, 0.184)}
    for r in rows:
        key = (r["ds"], r["prot"])
        if key in known:
            ef, ec = known[key]
            fb = "OK" if abs(r["fixed"] - ef) <= 0.005 else f"FIXED {r['fixed']:.3f}≠{ef}"
            cb = "OK" if abs(r["ccb"] - ec) <= 0.005 else f"CCB {r['ccb']:.3f}≠{ec}"
            print(f'  {r["ds"]:16s} {r["prot"]:14s} {fb:18s} {cb}')


if __name__ == "__main__":
    main()
