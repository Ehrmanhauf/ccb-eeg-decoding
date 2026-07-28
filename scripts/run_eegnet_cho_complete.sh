#!/usr/bin/env bash
# =============================================================================
# Complete the EEGNet Cho2017 cell at the FULL ~50-subject cohort
#   (full 64-ch montage + the C3/Cz/C4 3-channel subset, within-subject 5-fold CV)
#
# WHY THIS RUN EXISTS
#   The committed results/eegnet_cho.csv holds only 2 of 50 subjects (a pilot stub),
#   so its kappa is NOT comparable to the n=50 fixed-pipeline / CCB Cho2017 cells.
#   The master table therefore leaves the EEGNet Cho entry blank on purpose (see the
#   "incomplete stub" note in scripts/consolidate_results.py). This run replaces the
#   stub with the full cohort so the cell becomes comparable and the MI EEGNet panel
#   is complete.
#
# MATCHED-CONDITIONS DISCIPLINE (unchanged from the rest of the panel)
#   - CV split is FIXED at seed 42 inside the runner (matched_within_cv); only the
#     EEGNet *training* seed varies over --seeds.
#   - epochs / folds / seeds are identical to every other EEGNet cell.
#
# COST  ~17 h for the 52-subject full montage alone (per the overnight-script note),
#        plus the lighter 3-channel pass -> budget roughly ONE DAY. The runner
#        CHECKPOINTS PER SUBJECT to the output CSV, so an interruption keeps the
#        subjects already done (a restart re-does from subject 1 -- it is not
#        skip-resumable).
#
# HOW TO RUN  In YOUR OWN terminal (NOT via the assistant), caffeinated, lid open:
#
#       caffeinate -dimsu bash scripts/run_eegnet_cho_complete.sh
#
#   All stdout+stderr is tee'd to logs/eegnet_cho_<timestamp>.log (git-ignored).
#   When it finishes, tell the assistant: it re-aggregates (summarize_eegnet.py),
#   repoints the consolidator's Cho cell, and folds the number into tab:master /
#   tab:eegnet + the scoreboard.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=1
PY=.venv/bin/python

# ---- parameters -------------------------------------------------------------
DATASETS="cho2017,cho2017_3ch"   # full 64-ch montage + C3/Cz/C4 subset
CHO_SUBJECTS="1-52"              # loader drops 29 & 33 -> ~50 subjects (matches the n=50 cell)
SEEDS="42,7"                    # 2 training seeds, as in tab:eegnet ("mean over two random seeds")
EPOCHS="50"                     # identical to every other EEGNet cell
N_FOLDS="5"
OUT="results/eegnet_cho.csv"

# ---- logging / traceability -------------------------------------------------
mkdir -p logs
TS="$(date '+%Y%m%d_%H%M%S')"
LOG="logs/eegnet_cho_${TS}.log"
exec > >(tee -a "$LOG") 2>&1          # everything below goes to console AND the log

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] === $* ==="; }
trap 'log "ERROR near line $LINENO (exit $?). Partial results (subjects done so far) are in $OUT; a rerun restarts from subject 1."' ERR

log "EEGNet Cho2017 completion run — START"
echo "  host:        $(hostname)"
echo "  pwd:         $(pwd)"
echo "  git HEAD:    $(git rev-parse --short HEAD 2>/dev/null) on $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
echo "  git dirty:   $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') uncommitted file(s)"
echo "  python:      $($PY --version 2>&1)"
echo "  log file:    $LOG"
echo "  params:      datasets=$DATASETS  cho_subjects=$CHO_SUBJECTS  seeds=$SEEDS  epochs=$EPOCHS  folds=$N_FOLDS  (CV split seed fixed at 42)"
echo "  output:      $OUT  (checkpointed per subject)"

# ---- preflight: torch must be present (EEGNet needs the optional 'benchmark' extra) ----
if ! $PY -c "import torch; print('  torch:      '+torch.__version__)" 2>/dev/null; then
  log "torch is NOT importable. Install the benchmark extra first:  uv sync --extra benchmark   (then re-run)."
  exit 1
fi

# ---- 0. do not contend with another EEGNet worker (avoids the under-contention segfault) ----
while pgrep -f 'run_eegnet_(newdata|benchmark)\.py' >/dev/null 2>&1; do
  log "another EEGNet worker is running; waiting 60s…"; sleep 60
done

# ---- 1. back up the 2-subject stub before it is overwritten (git also retains it) ----
if [ -f "$OUT" ]; then
  cp "$OUT" "logs/eegnet_cho_stub_backup_${TS}.csv"
  log "backed up existing stub ($(($(wc -l < "$OUT") - 1)) data rows) -> logs/eegnet_cho_stub_backup_${TS}.csv"
fi

# ---- 2. the run: Cho2017 full + C3/Cz/C4, within-CV, ~50 subjects ----
log "launching run_eegnet_benchmark.py (this is the multi-hour step; progress is logged per subject)"
SECONDS=0
$PY scripts/run_eegnet_benchmark.py \
  --datasets "$DATASETS" --protocols within \
  --cho-subjects "$CHO_SUBJECTS" --seeds "$SEEDS" --epochs "$EPOCHS" --n-folds "$N_FOLDS" \
  --output "$OUT"
log "run finished in $((SECONDS/3600))h $(((SECONDS%3600)/60))m $((SECONDS%60))s"

# ---- 3. sanity: subject coverage + per-montage mean kappa, straight from the fresh CSV ----
log "coverage + mean kappa from $OUT:"
$PY - "$OUT" <<'PYEOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
for mont, g in d.groupby("montage"):
    k = g.dropna(subset=["kappa"])
    print(f"  Cho2017 [{mont:8s}]  subjects={g.subject.nunique():2d}  mean kappa={k.kappa.mean():+.3f}  (rows={len(g)})")
print(f"  -> expect ~50 subjects per montage; 2 means the run did not complete.")
PYEOF

# ---- 4. re-aggregate the committed EEGNet summary so it carries the new Cho cell ----
log "re-aggregating -> results/eegnet_summary.csv"
$PY scripts/summarize_eegnet.py

log "DONE. Log saved to $LOG"
echo ""
echo "  NEXT (assistant will do this once you report back):"
echo "    1. repoint the Cho cell in scripts/consolidate_results.py to _eegnet_summary(...)"
echo "    2. regenerate results_master.csv + the scoreboard figure"
echo "    3. add the Cho rows to tab:master / tab:eegnet and commit"
