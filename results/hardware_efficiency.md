# Hardware-efficiency benchmark — CPU calibration + inference profile

**Producer:** `scripts/run_hardware_efficiency_benchmark.py` · **Output:** `results/hardware_efficiency.csv`
**Host:** Apple M3, 8 cores, 8 GB RAM, Darwin 25.5.0 · Python 3.12.3, NumPy 2.4.4, torch 2.12.0
**Threading:** single thread throughout — `torch.set_num_threads(1)` *and* `OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS=1`
set before NumPy is imported, so the classical pipelines are held to the same one-core budget as torch.

**Cells:** six (dataset, montage) pairs spanning 2 → 62 channels, subject 1 of each, first fold of the
matched-conditions 5-fold split. Timing: 10 repeats over the test set per system after 5 warm-up calls;
CCB online adaptation, 5 repeats with a fresh policy each time.

## ⚠️ Caveat (read first)

This is a hardware **profile**, **not** a rescue of the thesis's negative decoding result. A cheaper
way to reach a low κ is **not** a deployment win. The κ column is a single-subject, single-fold anchor
carried only so the cost numbers are never misread as a decoding contribution — it is **not** a decoding
result. Two cells (STEW, COG-BCI full) read κ = 1.000 because their within-CV protocol is
leakage-saturated on a block-labelled dataset; that is exactly the artefact Chapter 4 diagnoses, and
those κ values carry no information. In the regime the thesis is actually about — near-ear
cross-session cognitive load — **every** method is near chance, so efficiency there buys nothing.

## Result

| Cell (ch) | System | κ† | p50 (ms) | p99 (ms) | fit (s) | param bytes | energy 15 W / 2 W (mJ) | online (ms) |
|---|---|---|---|---|---|---|---|---|
| BCI-IV-2a (22) | FBCSP + LDA | 0.483 | 8.01 | 19.73 | 3.27 | 85,496 | 125.2 / 16.7 | — |
| | BandPower + LDA | 0.483 | 0.33 | 1.00 | 0.08 | 36,536 | 5.5 / 0.7 | — |
| | CCB | 0.241 | 1.63 | 3.15 | 3.24 | 11,520 | 27.0 / 3.6 | 0.43 |
| | EEGNet | 0.655 | 0.59 | 2.17 | 45.98 | 12,176 | 11.4 / 1.5 | — |
| BCI-IV-2b (3) | FBCSP + LDA | 0.292 | 4.94 | 6.57 | 0.28 | 9,464 | 73.8 / 9.8 | — |
| | BandPower + LDA | 0.458 | 0.21 | 0.55 | 0.05 | 968 | 3.3 / 0.4 | — |
| | CCB | 0.083 | 1.18 | 2.25 | 0.83 | 2,488 | 19.0 / 2.5 | 0.26 |
| | EEGNet | 0.250 | 0.25 | 0.46 | 19.58 | 10,960 | 3.8 / 0.5 | — |
| STEW (14) | FBCSP + LDA | 1.000† | 6.81 | 10.06 | 0.36 | 42,392 | 100.9 / 13.4 | — |
| | BandPower + LDA | 1.000† | 0.26 | 0.57 | 0.02 | 15,176 | 4.2 / 0.6 | — |
| | CCB | 1.000† | 1.02 | 1.70 | 0.90 | 4,264 | 16.2 / 2.2 | 0.23 |
| | EEGNet | 1.000† | 0.37 | 1.40 | 10.27 | 11,616 | 6.5 / 0.9 | — |
| UAB near-ear (2) | FBCSP + LDA | 0.607 | 4.84 | 9.01 | 0.57 | 5,232 | 74.3 / 9.9 | — |
| | BandPower + LDA | 0.686 | 0.19 | 0.38 | 0.13 | 696 | 2.9 / 0.4 | — |
| | CCB | 0.465 | 0.72 | 1.60 | 1.22 | 5,912 | 11.6 / 1.5 | 0.79 |
| | EEGNet | 0.717 | 0.23 | 0.46 | 73.72 | 12,876 | 3.5 / 0.5 | — |
| COG-BCI (62) | FBCSP + LDA | 1.000† | 14.36 | 25.99 | 20.58 | 571,320 | 214.3 / 28.6 | — |
| | BandPower + LDA | 0.850 | 0.55 | 1.06 | 0.15 | 285,804 | 8.7 / 1.2 | — |
| | CCB | 0.900 | 1.62 | 2.89 | 53.94 | 56,024 | 25.7 / 3.4 | 0.56 |
| | EEGNet | 1.000† | 2.99 | 6.79 | 137.81 | 16,704 | 45.0 / 6.0 | — |
| COG-BCI near-ear (2) | FBCSP + LDA | 0.725 | 4.70 | 7.00 | 0.22 | 5,112 | 70.5 / 9.4 | — |
| | BandPower + LDA | −0.125 | 0.19 | 0.42 | 0.04 | 684 | 2.9 / 0.4 | — |
| | CCB | 0.453 | 0.69 | 1.24 | 0.97 | 1,956 | 10.9 / 1.4 | 0.34 |
| | EEGNet | 0.328 | 0.20 | 0.44 | 22.87 | 12,864 | 3.1 / 0.4 | — |

† Single subject / single fold; the STEW and COG-BCI-full cells are leakage-saturated (see caveat).
Energy is **derived** as CPU-time × TDP, so it is perfectly collinear with CPU time by construction;
it is reported at two TDPs bracketing embedded and laptop class rather than as an independent axis.

## Findings

1. **Every method is comfortably real-time, with a large margin.** The slowest single measurement in
   the entire table is FBCSP's p99 of 26.0 ms at 62 channels, against a 4-second epoch — 0.65 % of the
   real-time budget, i.e. more than two orders of magnitude inside it. Real-time feasibility is
   therefore *not* a discriminating axis among these methods on this class of hardware.

2. **Channel count costs memory and calibration, not latency.** The COG-BCI pair is a controlled
   comparison — same subject, same trials, same fold, only the montage differs. Going from 2 to 62
   channels (31×) multiplies FBCSP's inference latency by only **3.1×** (4.70 → 14.36 ms), strongly
   *sub*-linear, while its fitted parameter footprint grows **112×** (5.1 KB → 571 KB), roughly
   quadratic as expected for covariance-based spatial filtering, and its calibration time grows
   **94×** (0.22 s → 20.58 s). The common
   claim that dense-montage feature extractors scale super-linearly in channel count is therefore true
   of *footprint and fit cost* but **not** of per-trial inference latency, which is dominated by the
   fixed cost of nine zero-phase filter passes over the time axis.

3. **Calibration is where the deep comparator is expensive.** Per subject-fold: band-power 0.02–0.15 s,
   FBCSP 0.22–20.6 s, CCB 0.83–53.9 s, EEGNet **10.3–137.8 s**. EEGNet costs roughly two to three orders
   of magnitude more to calibrate than band-power, and that gap — not inference — is its deployment
   burden. Extrapolating the measured per-fit cost over a 29-subject leave-one-subject-out panel gives
   well over an hour of single-thread compute for EEGNet against under five seconds for band-power;
   that extrapolation is derived from the per-fit measurements here, not separately timed.

4. **The CCB is not the cheapest to run, on any axis except stored size.** It has a small model
   (2.0–56 KB) but its per-trial inference costs **2.9–5.6× more than band-power + LDA** in both latency
   and energy — roughly a quarter to a fifth as efficient, not half. Its online adaptation is genuinely
   affordable (0.23–0.79 ms/trial median, p95 ≤ 1.62 ms; 2.3–7.9 µs per arm over 56–100 arms), and it
   is a capability none of the static comparators has at all.

5. **Lightness is not a win — on either axis.** The CCB is the least accurate system on the
   leakage-clean cells while costing more per trial than band-power. The honest reading: the CCB is an
   attractive lightweight, online, constraint-aware *policy*, but on this thesis's evidence it neither
   decodes better nor runs cheaper than a simple fixed baseline. The hardware profile is a property to
   record, not a contribution that offsets the negative result.

## Methodology notes / limitations

- **Tail latency is reported, not just the median.** A deployed decoder misses deadlines on its tail.
  The p99/p50 ratio reaches 2.5× (2a FBCSP), so a median-only report would understate worst case.
- **Peak resident memory is deliberately absent.** `ru_maxrss` is a monotone process-level high-water
  mark, so within one process the first system fitted absorbs all shared allocation and later systems
  read near zero. `param_bytes` — the summed bytes of fitted numeric state — is the honest footprint
  measure and is what an embedded target actually needs to hold.
- **Energy is a derived proxy**, not a power-meter reading; treat the mJ figures as order-of-magnitude.
- **Single subject, single fold per cell.** Between-fit variation (differing arm counts, CSP ranks) is
  not captured; only within-fit timing variation is repeated. The κ column inherits this and is
  illustrative only.
- **The ranking is regime-dependent.** These are per-trial (batch size 1) numbers — the online
  deployment regime. Under batched throughput FBCSP vectorises across the batch while EEGNet's forward
  becomes cache-bound, which can reverse their order. Per-trial is reported because that is how a
  deployed BCI runs.
- The CCB row times context computation plus the selected arm's pipeline; arm selection itself is
  negligible linear algebra and is timed separately in the online-adaptation column.

## Reproduction

```bash
uv sync --extra benchmark   # installs CPU torch
PYTHONPATH=src .venv/bin/python scripts/run_hardware_efficiency_benchmark.py   # → results/hardware_efficiency.csv
PYTHONPATH=src .venv/bin/python scripts/run_kappa_robustness_2a.py             # → results/hardware_efficiency_kappa_2a.csv (κ robustness)
```
