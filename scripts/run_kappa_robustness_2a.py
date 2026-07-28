r"""Multi-subject κ robustness check for the hardware-efficiency benchmark's decoding anchor.

The hardware benchmark (``run_hardware_efficiency_benchmark.py``) reports per-system κ on a
single BCI-IV-2a subject/fold purely to anchor the caveat "efficiency only matters where
decoding works". This script checks how *robust* that κ ordering is, because a single-fold,
single-seed κ is noisy (and EEGNet adds training stochasticity).

For all nine 2a subjects (first within-subject 5-fold split, seed 42) it scores FBCSP+LDA,
band-power+LDA, and EEGNet at two training seeds, and reports the per-subject κ + means.

Finding (committed CSV): EEGNet decodes better *on average* (mean κ ≈ 0.70-0.77 vs ~0.52 for
the LDA-head fixed pipelines and ~0.27 for the CCB), consistent with the literature -- but it is
seed-sensitive at the single-subject level (e.g. s1: 0.66 at seed 42 vs 0.21 at seed 7). So the
qualitative ranking CCB < fixed < CNN is robust; the single-cell point estimate is not.

Output: results/hardware_efficiency_kappa_2a.csv
"""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd
import typer

mne.set_log_level("ERROR")


def main(
    seeds: str = typer.Option("42,7", help="Comma-separated EEGNet training seeds."),
    cnn_epochs: int = typer.Option(50),
    output: Path = typer.Option(Path("results/hardware_efficiency_kappa_2a.csv")),
) -> None:
    import torch

    from thesis.baselines.bandpower_cl import BandPowerCL
    from thesis.baselines.cnn import EEGNet
    from thesis.baselines.fbcsp import FBCSP
    from thesis.data.load import load_bci2a
    from thesis.metrics import compute_metrics
    from thesis.protocols import within_subject_cv

    torch.set_num_threads(1)
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    rows: list[dict] = []
    print(f"{'subj':5s} {'FBCSP+LDA':>10s} {'BandPwr+LDA':>12s} "
          + " ".join(f"EEGNet-s{s:<2d}".rjust(11) for s in seed_list))
    for subj in range(1, 10):
        cell = load_bci2a([subj])[0]
        sp = next(iter(within_subject_cv(cell, n_splits=5, seed=42)))
        Xtr, ytr = cell.X[sp.train_idx], cell.y[sp.train_idx]
        Xte, yte = cell.X[sp.test_idx], cell.y[sp.test_idx]
        sf = cell.sfreq
        kfb = float(compute_metrics(yte, FBCSP(sfreq=sf).fit(Xtr, ytr).predict(Xte)).kappa)
        kbp = float(compute_metrics(yte, BandPowerCL(sfreq=sf).fit(Xtr, ytr).predict(Xte)).kappa)
        knn = {s: float(compute_metrics(yte, EEGNet(sfreq=sf, epochs=cnn_epochs, seed=s)
                                        .fit(Xtr, ytr).predict(Xte)).kappa) for s in seed_list}
        rows.append({"subject": subj, "fbcsp_lda": kfb, "bandpower_lda": kbp,
                     **{f"eegnet_seed{s}": knn[s] for s in seed_list}})
        print(f"s{subj:<4d} {kfb:>10.3f} {kbp:>12.3f} "
              + " ".join(f"{knn[s]:>11.3f}" for s in seed_list))

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    means = df.drop(columns="subject").mean().round(3)
    print(f"{'MEAN':5s} {means['fbcsp_lda']:>10.3f} {means['bandpower_lda']:>12.3f} "
          + " ".join(f"{means[f'eegnet_seed{s}']:>11.3f}" for s in seed_list))
    s0 = seed_list[0]
    best_fixed = df[["fbcsp_lda", "bandpower_lda"]].max(axis=1)
    print(f"EEGNet(seed{s0}) > best LDA-head fixed: {int((df[f'eegnet_seed{s0}'] > best_fixed).sum())}/9")
    if len(seed_list) > 1:
        spread = (df[f"eegnet_seed{seed_list[0]}"] - df[f"eegnet_seed{seed_list[1]}"]).abs()
        print(f"EEGNet seed spread |Δκ|: mean {spread.mean():.3f}, max {spread.max():.3f} (subject {int(df.loc[spread.idxmax(),'subject'])})")
    print(f"Saved → {output}")


if __name__ == "__main__":
    typer.run(main)
