r"""Regenerate / verify the data-driven Chapter-4 tables from committed CSVs.

The advisor flagged that hand-copying table numbers caused discrepancies (e.g. the
Table 4.14 standard-deviation typo). This script recomputes the numeric cells of the
cross-paradigm consolidation table (``tab:cross-paradigm``) directly from the source
CSVs and compares them to the values currently printed in the thesis, flagging any
mismatch beyond a rounding tolerance. Run it after editing results or tables::

    PYTHONPATH=src .venv/bin/python scripts/make_tables.py

Exit status is non-zero if any cell mismatches, so it can gate a build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_RES = Path(__file__).resolve().parents[1] / "results"
TOL = 0.004  # rounding tolerance (tables print 3 decimals)


def _mean_kappa(csv: str, **filt) -> float | None:
    p = _RES / csv
    if not p.exists():
        return None
    d = pd.read_csv(p)
    for k, v in filt.items():
        if k in d.columns:
            d = d[d[k] == v]
    col = "kappa" if "kappa" in d.columns else ("kappa_within" if "kappa_within" in d.columns else None)
    if col is None or d.empty:
        return None
    return float(d[col].mean())


def _best_fixed_newdata(ds: str, montage: str) -> float:
    d = pd.read_csv(_RES / "fixed_baseline_newdata.csv").dropna(subset=["kappa"])
    d = d[(d.dataset == ds) & (d.montage == montage)]
    return float(d.groupby(["feature_family", "classifier"]).kappa.mean().max())


def _ccb_newdata(ds: str, montage: str) -> float:
    d = pd.read_csv(_RES / "ccb_newdata.csv")
    return float(d[(d.dataset == ds) & (d.montage == montage)].kappa.mean())


# (label, recomputed best-fixed, recomputed CCB, thesis best-fixed, thesis CCB)
def rows() -> list[tuple]:
    out = []

    def add(label, fixed_fn, ccb_fn, t_fixed, t_ccb):
        out.append((label, fixed_fn, ccb_fn, t_fixed, t_ccb))

    # New near-ear cells (fully recomputable from the new-data CSVs).
    add("UAB full", _best_fixed_newdata("UAB", "full"), _ccb_newdata("UAB", "full"), 0.963, 0.776)
    add("UAB near-ear", _best_fixed_newdata("UAB", "nearear"), _ccb_newdata("UAB", "nearear"), 0.714, 0.375)
    add("COG-BCI full", _best_fixed_newdata("COGBCI", "full"), _ccb_newdata("COGBCI", "full"), 0.987, 0.793)
    add("COG-BCI near-ear", _best_fixed_newdata("COGBCI", "nearear"), _ccb_newdata("COGBCI", "nearear"), 0.615, 0.318)
    # CCB cells for the original panel (best-fixed for these lives in mixed CL/MI CSVs;
    # the CCB side is the load-bearing, most-edited number, so we verify it).
    add("STEW CCB", None, _mean_kappa("ccb_stew_workload.csv"), 0.953, 0.744)
    add("WAUC CCB", None, _mean_kappa("ccb_wauc_workload.csv"), 0.785, 0.426)
    # 2b best-factorial cell (alpha=0.5, calib=0.3, within, recent-reward OFF) lives in
    # ccb_factorial.csv, NOT ccb_baseline.csv (an older single-config run at 0.124). The
    # include_recent_rewards=False constraint is the locked config; omitting it averages in
    # the recent-reward=True factorial seeds and dilutes 0.184 -> 0.183.
    add("BCI-IV-2b CCB", None, _mean_kappa("ccb_factorial.csv", alpha=0.5, calibration_frac=0.3, protocol="within", include_recent_rewards=False), 0.292, 0.184)
    add("Cho2017 full CCB", None, _mean_kappa("ccb_cho2017_full.csv"), 0.202, 0.077)
    add("Cho2017 3-ch CCB", None, _mean_kappa("ccb_cho2017_3ch.csv"), 0.190, 0.082)
    return out


def main() -> int:
    print(f"{'cell':22s} {'recomputed (fix/CCB)':24s} {'thesis (fix/CCB)':18s} status")
    bad = 0
    for label, fixed, ccb, t_fixed, t_ccb in rows():
        msgs = []
        if fixed is not None and abs(fixed - t_fixed) > TOL:
            msgs.append(f"FIXED {fixed:.3f}≠{t_fixed:.3f}")
        if ccb is not None and abs(ccb - t_ccb) > TOL:
            msgs.append(f"CCB {ccb:.3f}≠{t_ccb:.3f}")
        status = "OK" if not msgs else "⚠ " + "; ".join(msgs)
        bad += bool(msgs)
        fr = f"{fixed:.3f}" if fixed is not None else "  – "
        cr = f"{ccb:.3f}" if ccb is not None else "  – "
        print(f"{label:22s} {fr+' / '+cr:24s} {f'{t_fixed:.3f} / {t_ccb:.3f}':18s} {status}")
    print("\n" + ("ALL CELLS MATCH THE CSVs" if not bad else f"{bad} MISMATCH(ES) — fix the table or the CSV"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
