# Operational interpretation of the headline results

**Producer:** `scripts/make_operational_interpretation.py` · **Output:** `results/operational_interpretation.csv`

Cohen's kappa is the metric Chapter 4 compares methods in. This file translates the
headline cells into the terms a reader outside the field needs: plain accuracy against
the task's own chance level, and --- where per-epoch predictions were persisted --- the
error structure a deployed system would actually exhibit. Accuracy is read from the
committed CSVs, never reconstructed from kappa.

## Per-cell accuracy against chance

| Cell | Protocol | System | kappa | accuracy | chance | points above chance | n |
|---|---|---|---|---|---|---|---|
| COG-BCI n-back (full), cross-session | cross-session (leakage-clean) | best fixed pipeline (bandpower+svm) | 0.078 | 38.5\% | 33.3\% | +5.1 | 29 |
| COG-BCI n-back (nearear), cross-session | cross-session (leakage-clean) | best fixed pipeline (bandpower+random_forest) | 0.062 | 37.4\% | 33.3\% | +4.1 | 29 |
| COG-BCI n-back (full), cross-session | cross-session (leakage-clean) | CCB (OPLB) | 0.012 | 34.1\% | 33.3\% | +0.7 | 29 |
| COG-BCI n-back (nearear), cross-session | cross-session (leakage-clean) | CCB (OPLB) | 0.013 | 34.1\% | 33.3\% | +0.8 | 29 |
| COG-BCI MATB (nearear), cross-session | cross-session (leakage-clean) | best fixed pipeline (bandpower+svm) | 0.218 | 47.8\% | 33.3\% | +14.5 | 29 |
| COG-BCI MATB (nearear), cross-session | cross-session (leakage-clean) | CCB (OPLB) | 0.042 | 36.1\% | 33.3\% | +2.8 | 29 |
| BCI-IV-2a, 4-class official split | official split | CCB (OPLB) | 0.163 | 37.2\% | 25.0\% | +12.2 | 9 |
| BCI-IV-2a, 4-class official split | official split | best classical pipeline (random_forest) | 0.431 | 57.3\% | 25.0\% | +32.3 | 9 |
| STEW (14 ch) | within-subject CV (leakage-confounded) | best fixed pipeline (bandpower+lda) | 0.953 | 97.6\% | 33.3\% | +64.3 | 45 |
| STEW (14 ch) | leave-one-subject-out (leakage-clean) | fixed pipeline (bandpower) (bandpower) | 0.329 | 57.1\% | 33.3\% | +23.8 | 45 |
| STEW (14 ch) | leave-one-subject-out (leakage-clean) | fixed pipeline (fbcsp) (fbcsp) | 0.277 | 55.0\% | 33.3\% | +21.7 | 45 |

## Error structure: STEW leave-one-subject-out, best fixed pipeline

Pooled over 3330 held-out epochs from `loso_stew.csv` (accuracy 57.1\%, kappa 0.314).

| | pred low | pred medium | pred high |
|---|---|---|---|
| true low | 1223 | 202 | 129 |
| true medium | 297 | 213 | 341 |
| true high | 270 | 189 | 466 |

Per-class precision --- when the system declares a level, how often it is right:

| Declared level | Correct / declared | Precision | Recall |
|---|---|---|---|
| low | 1223 / 1790 | 0.683 | 0.787 |
| medium | 213 / 604 | 0.353 | 0.250 |
| high | 466 / 936 | 0.498 | 0.504 |

The asymmetry is the operationally important part: the low-load state is identified
far more reliably than the high-load state, and the intermediate level is barely
distinguished at all. An adaptive system keyed to detecting high load would therefore
be acting on a false alarm roughly half the times it fired.
