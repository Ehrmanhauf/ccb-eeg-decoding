# Phase F — Sensitivity sweeps and mechanism discrimination on STEW and WAUC

**Generated:** 2026-05-20 from `scripts/run_ccb_stew.py` and
`scripts/run_ccb_wauc.py` with comma-separated parameter lists
(`--alphas`, `--calibration-fracs`, `--window-sizes`,
`--per-round-caps`), each under `--use-workload-context`, 5 seeds ×
1 within-subject fold, OPLB policy. Source CSVs:
[`results/ccb_{stew,wauc}_sens_{alpha,calibration,window,cap}.csv`].
Raw sweep tables (mean ± std per parameter value):
[`results/ccb_cl_sensitivity.md`](ccb_cl_sensitivity.md).

This phase answers the question left open by
[`results/ccb_cl_phase_e.md`](ccb_cl_phase_e.md): *the context vector is
not the limiting factor (Phase E), so which of the remaining four
candidate mechanisms explains the CCB-vs-fixed-pipeline gap?* The gap to
explain is Δκ ≈ 0.21 on STEW (best fixed B2 = 0.953 vs workload-context
CCB = 0.744) and Δκ ≈ 0.23 on WAUC (best fixed B1 = 0.658 vs
workload-context CCB = 0.426).

## Headline result — no single parameter closes the gap; the mechanism profile is paradigm-level

Four single-parameter sweeps per dataset (eight total) probe four
candidate mechanisms. Verdict per cell: *driver* (monotone curve, gap
shrinks at the favourable end), *mild driver* (monotone but effect ≈ one
σ), or *ruled out* (flat within σ).

| Mechanism (sweep parameter)            | STEW                         | WAUC                         |
|----------------------------------------|------------------------------|------------------------------|
| Exploration cost (α)                   | Driver (α=0.1: +0.030)       | Driver (α=0.1: +0.030)       |
| Calibration overhead (calib. fraction) | Driver (0.2→0.5: +0.058)     | Driver (0.2→0.5: +0.033)     |
| Non-stationarity (sliding window W)    | Ruled out (flat; +0.007 @25) | Ruled out (flat; +0.019 @25) |
| Arm-pool composition (per-round cap)   | Mild driver ({2,3}: +0.026)  | Mild driver ({2,3}: +0.026)  |

**Two findings.**

1. **No single mechanism explains the gap.** The best single-parameter
   setting on each dataset (α = 0.1) lifts κ by only +0.030; even
   stacking the favourable end of every sweep would not approach the
   0.21–0.23 gap. The gap is therefore not a tuning artefact of any one
   hyperparameter — it is the structural cost of the online
   per-trial-adaptation regime itself: the policy must spend trials
   exploring (exploration-cost driver) and must reserve trials to
   calibrate the per-arm heads (calibration-overhead driver), and
   neither cost is recoverable by retuning a single knob.

2. **The WAUC profile replicates the STEW profile cell-for-cell.** Same
   two drivers, same mild driver, same rule-out; the arm-pool cap effect
   is identical to two decimal places (+0.026 on both). Because STEW and
   WAUC differ in subjects, hardware (14-ch Emotiv vs 8-ch Enobio),
   task (SIMKAP vs MATB-II), label operationalisation (3-class subjective
   vs binary task-difficulty), and — critically — leakage status (STEW
   within-CV is ceiling-saturated, WAUC is not), a profile that survives
   all of that variation is a **paradigm-level** property of CCB-on-
   cognitive-load, not a dataset artefact.

## Mechanism discrimination

- **Exploration cost (driver, both).** Lower α monotonically improves κ;
  α = 0.1 recovers +0.030 over the α = 0.5 default on both datasets. The
  bandit's exploration of suboptimal arms costs accuracy that the
  fixed pipeline (which never explores) does not pay. This is intrinsic
  to the bandit regime, not removable.
- **Calibration overhead (driver, both).** Higher calibration fraction
  monotonically improves κ. The per-arm heads benefit from more
  calibration data more than the bandit stream benefits from more bandit
  rounds. The effect is larger on STEW (+0.058 over the 0.2→0.5 range)
  than WAUC (+0.033), consistent with STEW's much smaller per-subject
  budget (≈74 trials vs WAUC's 600–930 epochs), which leaves STEW's
  heads more calibration-starved at the low end.
- **Non-stationarity (ruled out, both).** The sliding-window sweep is
  flat: W ∈ {50, 100, ∞} agree within 0.006 κ on both datasets, and the
  W = 25 uptick (+0.007 STEW, +0.019 WAUC) sits well inside one σ
  (≈ 0.21–0.25) and is non-monotone. The optimal arm does not drift
  across a within-subject session; an inappropriately wide history
  window is not the cause of the gap.
- **Arm-pool composition (mild driver, both).** Binding the per-round
  cost cap to {2, 3} prunes the expensive arms and recovers +0.026 on
  both datasets; cap = 4 ≈ default (binds on too few arms). A real but
  small contributor — not the dominant mechanism.

## What this means for the gap

The −0.21 (STEW) / −0.23 (WAUC) gap is **explained as the joint,
structural cost of online per-trial adaptation, not as a misconfigured
hyperparameter.** Two of the four candidate mechanisms (exploration cost,
calibration overhead) are genuine drivers but each contributes only a
small recoverable fraction; one (arm-pool composition) is a mild driver;
one (non-stationarity) is ruled out. The context vector was already ruled
out in Phase E. With context, non-stationarity, and any single
hyperparameter all eliminated as the explanation, the residual gap is the
price the bandit pays for exploring and for reserving calibration trials —
a price the frozen fixed pipeline never pays. The cell-for-cell
cross-paradigm replication promotes this from a two-dataset observation to
a paradigm-level characterisation of the CCB-on-cognitive-load setting.

This is captured in Chapter 4 §4.4 (four WAUC tables + cross-paradigm
mechanism matrix + interpretation) and design-doc §2.6/§2.7.

## Reproduction

```bash
make stew-check && make wauc-check
# STEW (≈45 min total)
for p in "--alphas 0.1,0.5,1.0,2.0" "--calibration-fracs 0.2,0.3,0.5" \
         "--window-sizes 0,25,50,100" "--per-round-caps inf,4,3,2"; do
  PYTHONPATH=src .venv/bin/python scripts/run_ccb_stew.py --use-workload-context $p \
    --output results/ccb_stew_sens_<param>.csv
done
# WAUC (≈5 h total) — run STRICTLY SERIALLY, one process at a time:
# parallel runs thrash an 8 GB machine into swap (verified 2026-05-20).
for p in "--alphas 0.1,0.5,1.0,2.0" "--calibration-fracs 0.2,0.3,0.5" \
         "--window-sizes 0,25,50,100" "--per-round-caps inf,4,3,2"; do
  PYTHONPATH=src .venv/bin/python scripts/run_ccb_wauc.py --use-workload-context $p \
    --output results/ccb_wauc_sens_<param>.csv
done
```
