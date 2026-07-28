# COG-BCI PVT vigilance — the second vital sign (optional)

**Producer:** `scripts/run_pvt.py` · **Output:** `results/pvt_vigilance.csv`
(1504 rows, 29 subjects × 2 montages × 2 protocols; **1 exception**, see below). Loader:
`thesis.data.cogbci_pvt_load`.

**Montage.** The full montage is the **canonical 62-channel EEG set** — every PVT session is
reduced (via `pick`) to the same 62 channels as the COG-BCI workload cells, dropping the
non-EEG `ECG1` and `Cz` (`Cz` is recorded only for subjects 10–29). The near-ear cell is the
T7/T8 pair by electrode position. (Earlier runs evaluated a non-uniform 63/64-channel montage
that retained ECG1 and, for subjects 10–29, Cz; this was corrected to the uniform 62.)

**Operational definition (documented per the CL-methodology rule).** Target = vigilance, from
the COG-BCI Psychomotor Vigilance Task. The canonical lapse label (RT > 500 ms; Dinges & Powell
1985, Basner & Dinges 2011) is too rare in this rested cohort (≈ 5–6 %) for a balanced two-class
decode, so we use the standard balanced operationalization: a **within-session reaction-time
median split** (slower half = low vigilance, faster half = high; RT is the canonical PVT
vigilance variable, Lim & Dinges 2008). Each trial is the **2 s pre-stimulus window** (ongoing
state → predict the upcoming response's vigilance), resampled to 250 Hz. The median is a label
threshold, not a feature (no EEG leakage).

## Result — vigilance is near chance at *every* montage and protocol

| Protocol | Montage | best fixed κ | CCB κ | Δκ |
|---|---|---|---|---|
| within-session (S1, 5-fold)   | full 62-ch     | 0.077 | −0.021 | +0.097 |
| within-session (S1, 5-fold)   | near-ear T7/T8 | 0.065 | −0.025 | +0.090 |
| cross-session (S1→S2/S3)       | full 62-ch     | 0.018 | −0.012 | +0.030 |
| cross-session (S1→S2/S3)       | near-ear T7/T8 | 0.030 | +0.001 | +0.029 |

Best fixed = max κ over the fixed feature-family × classifier grid; CCB = mean over seeds.
Chance κ = 0 (balanced target).

## Findings

1. **Vigilance is undecodable here at every montage and protocol.** The best fixed pipeline tops
   out at κ = 0.077 (within-session, full 62-ch) — "slight" agreement on the Landis–Koch scale,
   far below any usable level. This is *not* a near-ear-montage result: the failure is in the
   target (a rested cohort with a tight RT distribution, ~90 trials/session, and a pre-stimulus
   window carrying little vigilance-discriminative EEG), and is reported as a null rather than
   tuned post hoc. Dropping the ECG1 noise channel nudged the within-full fixed κ from the earlier
   63-ch run's 0.034 to 0.077, but the cell remains near chance.
2. **The CCB underperforms the fixed pipeline throughout** (Δκ = +0.03 to +0.10; the CCB is
   mildly negative within-session). On an undecodable target the bandit's exploration and
   calibration reservation push it slightly below the chance floor, so the panel-wide
   fixed-≥-CCB direction holds on a second paradigm even where neither method finds signal.
3. **One exception (recorded, not dropped):** subject 24's within-session full-montage CCB cell
   prunes to an empty arm pool ("all arms pruned; pool is empty") — no arm clears the calibration
   κ threshold on this undecodable target. It is preserved as an exception row in the CSV and
   excluded from the CCB mean.

The cell's value is this consistency demonstration (the same fixed-≥-CCB direction on a second
cognitive state), not a positive vigilance result.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/run_pvt.py        # → results/pvt_vigilance.csv
```

Montages `nearear,full`; protocols within-session (S1 5-fold) and cross-session (S1→S2/S3).
The near-ear cells are independent of the ECG1/Cz drop (T7/T8 only) and reproduce the earlier
run's numbers exactly — a regression check on the montage correction.
