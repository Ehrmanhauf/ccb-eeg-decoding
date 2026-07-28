#!/usr/bin/env bash
# Sequential EEGNet decoding sweep — ONE process at a time (no contention → no segfault),
# resume-safe where possible. Built to firm up the deep-learning comparator (tab:eegnet)
# overnight at full subject counts.
#
# RUN IT IN YOUR OWN TERMINAL (not via the assistant) so nothing kills it mid-run, with
# sleep fully disabled and the lid open:
#
#     caffeinate -dimsu bash scripts/run_eegnet_overnight.sh 2>&1 | tee /tmp/eegnet_overnight.log
#
# It writes to results/eegnet_*.csv; re-running resumes the near-ear/cross-session cells and
# re-does the per-file MI/CL cells. When it finishes, the assistant aggregates everything with
# scripts/summarize_eegnet.py and folds the final numbers into Chapter 4 (tab:eegnet).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=1
PY=.venv/bin/python
log(){ echo "[$(date '+%H:%M:%S')] === $* ==="; }

# 0. Don't run concurrently with an already-running worker (avoid the segfault-under-contention).
while pgrep -f 'run_eegnet_(newdata|benchmark)\.py' >/dev/null 2>&1; do
  echo "[$(date '+%H:%M:%S')] another eegnet worker is running; waiting 60s…"; sleep 60
done

# 1. Near-ear + cross-session + PVT (run_eegnet_newdata is RESUME-SAFE: skips done cells).
log "1/4 newdata: COG-BCI cross (nback+matb), UAB near, PVT (near+full)"
$PY scripts/run_eegnet_newdata.py \
  --cells cogbci_nback_cross,cogbci_matb_cross,uab_within,pvt_cross,pvt_within \
  --montages nearear,full --seeds 42,7 --epochs 50 --output results/eegnet_newdata.csv

# 2. Motor imagery: 2b (within+official), 2a (4-class official).
log "2/4 MI: 2b + 2a"
$PY scripts/run_eegnet_benchmark.py --datasets bci2b --protocols within,official \
  --seeds 42,7 --epochs 50 --output results/eegnet_2b.csv
$PY scripts/run_eegnet_benchmark.py --datasets bci2a --protocols official \
  --seeds 42,7 --epochs 50 --output results/eegnet_2a.csv

# 3. Cognitive load, leakage-clean: STEW/WAUC LOSO (+ within for the leak-confounded reference).
log "3/4 CL: STEW/WAUC LOSO + within"
$PY scripts/run_eegnet_benchmark.py --datasets stew,wauc --protocols within,loso \
  --seeds 42,7 --epochs 50 --output results/eegnet_clloso.csv

# 4. Motor imagery: Cho2017 within-CV, full + C3/Cz/C4 (representative 16-subject subset;
#    the full 52 would run ~17h alone — 16 gives a stable mean with an n= note in the thesis).
log "4/4 MI: Cho2017 1-16 (full + 3-ch)"
$PY scripts/run_eegnet_benchmark.py --datasets cho2017,cho2017_3ch --protocols within \
  --cho-subjects 1-16 --seeds 42,7 --epochs 50 --output results/eegnet_cho.csv

log "ALL EEGNET CELLS DONE — next: scripts/summarize_eegnet.py, then fold into tab:eegnet"
