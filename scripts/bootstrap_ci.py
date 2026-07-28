r"""Bootstrap 95% confidence intervals for the leakage-clean headline cells.

Addresses the small-n objection: the leakage-clean cells (STEW/WAUC LOSO, COG-BCI
cross-session n-back, COG-BCI MATB competition) have only n = 15--45 subjects, and the
thesis otherwise reports mean ± SD. This script resamples SUBJECTS with replacement
(percentile bootstrap, B = 10 000, fixed seed) to put a 95% CI on:

  (a) the best fixed-pipeline per-subject mean κ,
  (b) the CCB per-subject mean κ,
  (c) the paired Δκ = best-fixed − CCB (the directional claim) — a CI that excludes 0
      means the CCB is significantly below the fixed pipeline; a CI straddling 0 (e.g.
      WAUC LOSO, where everything is at chance) means the comparison is not meaningful.

No model re-runs: per-subject κ is read from the committed ``results/*.csv``. The CCB
per-subject κ is averaged over its seeds first, then bootstrapped over subjects.

Output: ``results/bootstrap_ci.csv`` + a printed summary.

Run:  PYTHONPATH=src .venv/bin/python scripts/bootstrap_ci.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_RES = Path(__file__).resolve().parents[1] / "results"
B = 10_000
SEED = 12345


def _mean_ci(vals: np.ndarray, rng, lo: float = 2.5, hi: float = 97.5):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    boot = vals[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return float(vals.mean()), float(np.percentile(boot, lo)), float(np.percentile(boot, hi)), n


def _paired_delta_ci(a: np.ndarray, b: np.ndarray, rng, lo: float = 2.5, hi: float = 97.5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = a[keep] - b[keep]
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    boot = d[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boot, lo)), float(np.percentile(boot, hi)), n


def _loso_fixed_ccb(csv: str):
    """LOSO CSVs: fixed methods are 'fbcsp'/'bandpower' (B1/B2); pick the aggregate-best."""
    d = pd.read_csv(_RES / csv)
    fixed_methods = [m for m in ("fbcsp", "bandpower") if m in d.method.unique()]
    best = max(fixed_methods, key=lambda m: d[d.method == m].kappa.mean())
    fixed = d[d.method == best].dropna(subset=["kappa"]).groupby("subject").kappa.mean()
    ccb = d[d.method == "ccb"].dropna(subset=["kappa"]).groupby("subject").kappa.mean()
    return best, fixed, ccb


def _cross_fixed_ccb(csv: str, task: str, montage: str):
    """Cross-session CSVs: fixed rows carry method='fixed' + (feature_family, classifier)."""
    d = pd.read_csv(_RES / csv)
    d = d[(d.task == task) & (d.montage == montage)]
    fx = d[d.method == "fixed"].dropna(subset=["kappa"])
    key = fx.groupby(["feature_family", "classifier"]).kappa.mean().idxmax()
    fxb = fx[(fx.feature_family == key[0]) & (fx.classifier == key[1])].groupby("subject").kappa.mean()
    ccb = d[d.method == "ccb"].dropna(subset=["kappa"]).groupby("subject").kappa.mean()
    return f"{key[0]}/{key[1]}", fxb, ccb


CELLS = [
    ("STEW LOSO", lambda: _loso_fixed_ccb("loso_stew.csv")),
    ("WAUC LOSO", lambda: _loso_fixed_ccb("loso_wauc.csv")),
    ("COG-BCI n-back cross (full)", lambda: _cross_fixed_ccb("crosssession_cogbci.csv", "nback", "full")),
    ("COG-BCI n-back cross (near-ear)", lambda: _cross_fixed_ccb("crosssession_cogbci.csv", "nback", "nearear")),
    ("COG-BCI MATB competition (full)", lambda: _cross_fixed_ccb("crosssession_matb_competition.csv", "matb-competition", "full")),
]


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for name, fn in CELLS:
        best, fixed, ccb = fn()
        common = sorted(set(fixed.index) & set(ccb.index))
        fa = fixed.reindex(common).to_numpy()
        ca = ccb.reindex(common).to_numpy()
        fm, flo, fhi, _ = _mean_ci(fa, rng)
        cm, clo, chi, _ = _mean_ci(ca, rng)
        dm, dlo, dhi, n = _paired_delta_ci(fa, ca, rng)
        sig = "Δκ>0 (95% CI excludes 0)" if dlo > 0 else "Δκ CI includes 0"
        rows.append({
            "cell": name, "best_fixed": best, "n": n,
            "fixed_mean": round(fm, 4), "fixed_ci_lo": round(flo, 4), "fixed_ci_hi": round(fhi, 4),
            "ccb_mean": round(cm, 4), "ccb_ci_lo": round(clo, 4), "ccb_ci_hi": round(chi, 4),
            "delta_mean": round(dm, 4), "delta_ci_lo": round(dlo, 4), "delta_ci_hi": round(dhi, 4),
            "delta_excludes_zero": bool(dlo > 0),
        })
        print(f"{name}  (n={n}, best fixed = {best})")
        print(f"    best-fixed κ = {fm:+.3f}  95% CI [{flo:+.3f}, {fhi:+.3f}]")
        print(f"    CCB        κ = {cm:+.3f}  95% CI [{clo:+.3f}, {chi:+.3f}]")
        print(f"    Δκ fixed−CCB = {dm:+.3f}  95% CI [{dlo:+.3f}, {dhi:+.3f}]   → {sig}")
    out = _RES / "bootstrap_ci.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out} ({len(rows)} cells); B={B}, seed={SEED}")


if __name__ == "__main__":
    main()
