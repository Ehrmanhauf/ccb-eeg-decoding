#!/usr/bin/env bash
# Re-run ONLY the cells whose numbers change under the matched-conditions fix
# (src/thesis/matched.py): the leakage-clean LOSO cells, where the per-fold
# training pool is now the common 4000-epoch class-stratified subsample for ALL
# three method families (was Fixed=4k / EEGNet=full~34k / CCB=800).
#
# What is NOT here, and why:
#   - Fixed WAUC LOSO was already 4k@seed42 -> identical pool; re-run only to
#     regenerate a self-consistent CSV alongside the changed CCB/EEGNet rows.
#   - STEW pools ~3.3k (< 4k) so the cap is a no-op there; only the CCB STEW LOSO
#     number changes (its old 800 cap is lifted to the full pool).
#   - Within-CV / official / cross-session numbers are UNCHANGED: the fixed/FBCSP/
#     EEGNet within-CV runners now draw folds from matched_within_cv at the same
#     seed (42) they already used, so those CSVs are byte-identical (verified by
#     make test + the guardrail). No re-run needed.
#
# RUN IT IN YOUR OWN TERMINAL (sleep disabled, lid open) so nothing kills it:
#
#     caffeinate -dimsu bash scripts/rerun_fairness.sh 2>&1 | tee /tmp/rerun_fairness.log
#
# Resumable: the CCB/fixed LOSO writers checkpoint per method/seed. The EEGNet
# WAUC cell is the long pole (~hours at the 4k cap, vs ~days on full data).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=1
PY=.venv/bin/python
log(){ echo "[$(date '+%H:%M:%S')] === $* ==="; }

# Don't contend with any stray EEGNet worker.
while pgrep -f 'run_eegnet_(newdata|benchmark)\.py' >/dev/null 2>&1; do
  echo "[$(date '+%H:%M:%S')] an eegnet worker is running; waiting 60s…"; sleep 60
done

# 1. WAUC LOSO — fixed + CCB at the common 4k matched pool (fast; no torch).
log "1/5 WAUC LOSO: fbcsp,bandpower,ccb @ matched 4k"
$PY scripts/run_loso_wauc.py --methods fbcsp,bandpower,ccb \
  --ccb-seeds 0,1,2,3,42 --output results/loso_wauc.csv

# 2. STEW LOSO — CCB changes (800 -> full ~3.3k); re-run all three for a
#    consistent CSV (fixed numbers are unchanged).
log "2/5 STEW LOSO: fbcsp,bandpower,ccb @ matched (full <=4k)"
$PY scripts/run_loso_stew.py --methods fbcsp,bandpower,ccb \
  --ccb-seeds 0,1,2,3,42 --output results/loso_stew.csv

# 3. EEGNet WAUC LOSO — the corrected deep comparator at the 4k cap (THE fix).
#    Separate output file; summarize_eegnet.py globs results/eegnet_*.csv and the
#    assistant folds the WAUC rows into the EEGNet table.
log "3/5 EEGNet WAUC LOSO @ matched 4k (long pole, ~hours)"
$PY scripts/run_eegnet_benchmark.py --datasets wauc --protocols loso \
  --seeds 42,7 --epochs 50 --output results/eegnet_wauc.csv

# 4. EEGNet WAUC within-CV — the leak-confounded reference cell for the
#    validation-sanity table (optional; comment out to skip the ~9h within sweep).
log "4/5 EEGNet WAUC within-CV @ matched (reference cell)"
$PY scripts/run_eegnet_benchmark.py --datasets wauc --protocols within \
  --seeds 42,7 --epochs 50 --output results/eegnet_wauc_within.csv

# 5. Efficiency profile — MUST run LAST, on the now-quiet CPU: per-trial inference cost
#    (all four methods) + the CCB's per-trial online-adaptation cost (select+update), across
#    a representative span (MI 22-ch, MI 3-ch, CL 14-ch, near-ear 2-ch). Timing under CPU
#    contention is meaningless, which is why this is sequential, not concurrent.
log "5/5 efficiency profile (quiet CPU): 2a,2b,stew,uab_nearear x 4 methods (inference + online cost)"
$PY scripts/run_hardware_efficiency_benchmark.py \
  --cells 2a,2b,stew,uab_nearear --output results/hardware_efficiency.csv

log "ALL MATCHED RE-RUNS + EFFICIENCY DONE — next: summarize_eegnet.py + make_tables.py, then fold numbers"
