# COG-BCI MATB cross-session near-ear — the second-task anchor (Phase 5/8)

**Producer:** `scripts/run_cogbci_crosssession.py --task matb --montages nearear` ·
**Output:** `results/crosssession_matb.csv` (377 rows, 29 subjects, 0 errors). Protocol:
**train session S1, test sessions S2+S3**, T7/T8 near-ear subset, 3-class MATB difficulty
(easy/med/diff). Same runner, loader, and split as the N-back cross-session headline —
only the task differs.

> Scope note: this is the MATB task from the **3-session** COG-BCI database (Zenodo
> 7413650), run under the thesis's pure no-adaptation cross-session protocol. It is the
> *runnable* MATB anchor. The separate **2-session competition split** (Zenodo 5055046,
> pre-epoched), directly comparable to the published leaderboard, is now also done — see
> [`crosssession_matb_competition.md`](crosssession_matb_competition.md) (full κ=0.296 /
> 53 % acc, inside the expert band; CCB trails at 0.125).

## Result

| Task (near-ear T7/T8, cross-session) | Best fixed κ | CCB κ | Δκ (fixed − CCB) |
|---|---|---|---|
| N-back (the leakage-clean headline) | 0.062 | 0.013 | +0.049 |
| **MATB (second-task anchor)** | **0.218** | **0.042 ± 0.099** | **+0.176** |

## Findings

**1. The near-ear cross-session regime is hard and task-dependent — not universally at
chance.** N-back near-ear cross-session collapses to chance (0.062); MATB retains *modest*
signal (best fixed 0.218). The two tasks manipulate workload differently (discrete n-back
memory levels vs continuous multi-attribute task load), and the continuous MATB load leaves
more cross-session-stable near-ear structure. The honest headline is therefore **"near-ear
cross-session cognitive-load decoding is hard (κ ≈ 0.06–0.22, task-dependent)"**, not
"uniformly impossible".

**2. The CCB underperforms the fixed pipeline on both tasks.** Δκ = +0.05 (N-back) and
+0.18 (MATB): wherever there is cross-session signal at the near-ear montage, the fixed
pipeline captures more of it than the CCB. The bandit's online adaptation does not convert
the modest MATB signal into a competitive score (CCB 0.042 vs fixed 0.218). The panel-wide
fixed > CCB direction holds in the deployment regime on a second, independent task.

**3. Absolute level is consistent with the field's difficulty.** MATB best-fixed κ = 0.218
(≈ low-50 % balanced accuracy, 3-class) is in line with the COG-BCI MATB leaderboard, where
expert teams topped out under 60 % accuracy under *easier* conditions (calibration/alignment
permitted). The chance-to-modest range is honest and expected, not a pipeline defect.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/run_cogbci_crosssession.py \
    --task matb --montages nearear --output results/crosssession_matb.csv
```
