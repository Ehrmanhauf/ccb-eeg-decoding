# COG-BCI cross-session: fixed pipeline vs CCB — Phase 4 (the deployment headline)

**Producer:** `scripts/run_cogbci_crosssession.py --task nback` · **Output:**
`results/crosssession_cogbci.csv` (754 rows, 0 errors). Protocol: **train on session
S1, test on sessions S2 + S3** (one week apart) — the deployment-regime drift test.
29 subjects, full 62-channel montage and the T7/T8 near-ear subset. Fixed pipelines
B1–B5 fitted on S1 and scored on S2+S3; CCB calibrated + streamed on S1, scored frozen
on S2+S3 (5 seeds). No domain adaptation.

## Result: cross-session collapses to chance

| Montage | within-CV fixed / CCB | **cross-session fixed / CCB** | Δ (within → cross) |
|---|---|---|---|
| Near-ear T7/T8 | 0.615 / 0.318 | **0.062 / 0.013** | collapse to chance |
| Full 62-channel | 0.987 / 0.793 | **0.078 / 0.012** | collapse to chance |

(best fixed = max over B1–B5; CCB = mean over 5 seeds. Near-ear CCB 0.013 ± 0.066,
full CCB 0.012 ± 0.079 — both indistinguishable from the κ = 0 chance floor.)

## Findings

**1. The within-CV numbers were leakage, confirmed decisively.** The full-montage κ
falls from 0.987 to 0.078 once train and test come from *different sessions* — the
within-session 0.987 was almost entirely the file-identity shortcut (the 3 N-back levels
are separate recordings). Cross-session removes the shortcut and the honest score is
~chance. This is the strongest possible confirmation of the leakage caveat carried since
Phase 2 (and parallels LOSO on STEW: 0.94 → ~0.28).

**2. Near-ear cross-session is the deployment ceiling — and it is chance.** At a
2-channel near-ear montage, cross-session cognitive-load decoding does not work (best
fixed κ = 0.062). This is the single most deployment-relevant number in the thesis: a
clean, citable ceiling that no within-session dataset could provide. 2-channel near-ear
workload monitoring across sessions, on this evidence, is not viable with these pipelines.

**3. The CCB does not rescue the deployment regime.** The bandit's online per-trial
adaptation — the one capability that could in principle exploit drift — leaves it at
chance (κ ≈ 0.01), alongside the fixed pipeline. Online adaptation does **not** bridge
cross-session drift here; both methods fail together. Per the work plan's outcome guide,
this is the "CCB loses within **and** cross-session → the negative result covers the
deployment regime — bulletproof" case (here both methods are at the floor, so the
sharper reading is that *the regime itself is undecodable* at these montages without
domain adaptation).

**4. Context (honest scope).** This is a *pure* cross-session test with **no domain
adaptation** (train S1, test S2/S3 directly), which is harder than the COG-BCI
competition's leaderboard setting (MATB task, calibration/alignment permitted; expert
teams topped out < 60 %). The chance result is therefore expected and honest, not a bug:
it characterises fixed-vs-CCB under raw drift, not a leaderboard attempt. The MATB
competition split (`run_cogbci_crosssession.py --task matb`) is the directly
leaderboard-comparable anchor.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/run_cogbci_crosssession.py \
    --task nback --montages nearear,full --output results/crosssession_cogbci.csv
```
Crash-safe: checkpoints per (task, montage, subject), resumes from an existing CSV.
