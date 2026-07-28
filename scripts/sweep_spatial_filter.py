"""Spatial-filter family sensitivity sweep for CCB on BCI-IV-2b.

Closes the "Spatial-filter family for 3-ch bipolar" open-justifications
item by comparing the full 3-family arm pool (currently in production)
against three single-family restrictions.

Cells (only ``_SPATIAL_FILTERS`` varied; Riemannian-on-laplacian/identity
arms are not enabled by default and are unaffected):

- ``full``: ``("csp", "laplacian", "identity")`` — Phase-2/3 default.
- ``csp``: CSP-only. ``n_components`` capped at 3 (2b's channel count) per
  the closed "CSP component count" justification.
- ``laplacian``: Laplacian-only — C3-Cz and C4-Cz bipolar differencing on
  top of the already-bipolar 2b recording. Output is 2-dim per band.
- ``identity``: identity-only — pass the 3 bipolar channels through
  un-modified. Output is 3-dim per band.

Each cell is evaluated at the **Phase-5 Stage-1 best-factorial CCB cell**
(``alpha=0.5``, ``calibration_frac=0.3``, ``window_size=50``,
``arm_pool=pruned``, ``include_recent_rewards=False``) so the result
isolates the spatial-filter family as the only varying axis.

Output: ``results/ccb_sens_spatial_filter.csv`` + ``.md`` aggregate.
"""

from __future__ import annotations

import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.progress import track

from thesis.ccb import arms as arms_mod
from thesis.ccb.arms import enumerate_arms_2b
from thesis.ccb.oplb import OPLBConfig
from thesis.ccb.runner import run_ccb_on_split
from thesis.data.load import load_bci2b_screening
from thesis.protocols import session_split, within_subject_cv

mne.set_log_level("ERROR")


_FULL_SPATIAL: tuple[str, ...] = ("csp", "laplacian", "identity")

_CELLS: dict[str, tuple[str, ...]] = {
    "full": _FULL_SPATIAL,
    "csp_only": ("csp",),
    "laplacian_only": ("laplacian",),
    "identity_only": ("identity",),
}


def _splits_for_subject(data, protocol: str, *, seed: int):
    if protocol == "within":
        return list(within_subject_cv(data, n_splits=5, seed=seed))[:1]
    if protocol == "official":
        return [session_split(data, train_session_idx=0, test_session_idx=1)]
    raise ValueError(f"unknown protocol {protocol!r}")


def main(
    seeds: str = typer.Option(
        "0,1,2,3,42", help="Comma-separated seeds (default = best-factorial set)."
    ),
    output_csv: Path = typer.Option(Path("results/ccb_sens_spatial_filter.csv")),
    output_md: Path = typer.Option(Path("results/ccb_sens_spatial_filter.md")),
) -> None:
    console = Console()
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    subj_ids = list(range(1, 10))

    console.log("Loading BCI-IV-2b screening (all 9 subjects) …")
    data_by_subj = {s: load_bci2b_screening(subjects=[s])[0] for s in subj_ids}
    console.log(f"  → {len(data_by_subj)} subjects.")

    saved_spatial = arms_mod._SPATIAL_FILTERS
    rows: list[dict] = []
    t0 = time.perf_counter()
    try:
        for cell_name, spatials in _CELLS.items():
            console.rule(f"cell={cell_name}  spatials={spatials}")
            arms_mod._SPATIAL_FILTERS = spatials  # type: ignore[attr-defined]
            cells = [
                (subj, prot, seed)
                for subj in subj_ids
                for prot in ("within", "official")
                for seed in seed_list
            ]
            for subj, protocol, seed in track(
                cells, description=cell_name, console=console
            ):
                data = data_by_subj[subj]
                base_arms = enumerate_arms_2b(sfreq=data.sfreq)
                for split in _splits_for_subject(data, protocol, seed=seed):
                    config = OPLBConfig(
                        alpha=0.5,
                        lambda_reg=1.0,
                        budget=float("inf"),
                        window_size=50,
                        discount_gamma=1.0,
                    )
                    try:
                        result = run_ccb_on_split(
                            data,
                            split,
                            arms=base_arms,
                            config=config,
                            calibration_frac=0.3,
                            min_kappa=0.05,
                            max_arms=100,
                            seed=seed,
                            include_recent_rewards=False,
                        )
                    except RuntimeError as exc:
                        # "All arms pruned" — record as an empty-pool outcome
                        # (the ablation cell itself is the finding).
                        if "All arms pruned" not in str(exc):
                            raise
                        rows.append(
                            {
                                "cell": cell_name,
                                "spatial_filters": ";".join(spatials),
                                "subject": subj,
                                "protocol": protocol,
                                "seed": seed,
                                "kappa": float("nan"),
                                "accuracy": float("nan"),
                                "n_test": 0,
                                "n_arms_surviving": 0,
                                "n_arms_total": len(base_arms),
                                "stream_rounds_run": 0,
                                "status": "empty_pool",
                            }
                        )
                        continue
                    rows.append(
                        {
                            "cell": cell_name,
                            "spatial_filters": ";".join(spatials),
                            "subject": subj,
                            "protocol": protocol,
                            "seed": seed,
                            "kappa": result.kappa,
                            "accuracy": result.accuracy,
                            "n_test": result.n_test,
                            "n_arms_surviving": result.n_arms_surviving,
                            "n_arms_total": len(base_arms),
                            "stream_rounds_run": int(result.arm_pulls.size),
                            "status": "ok",
                        }
                    )
    finally:
        arms_mod._SPATIAL_FILTERS = saved_spatial  # type: ignore[attr-defined]

    elapsed = time.perf_counter() - t0
    console.log(f"Sweep completed in {elapsed:.1f}s — {len(rows)} rows.")

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    console.log(f"Saved → {output_csv}")

    summary = (
        df.groupby(["cell", "protocol"])["kappa"]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    console.print(summary)
    empty = df[df["status"] == "empty_pool"]
    if len(empty):
        console.log(
            f"[yellow]{len(empty)} runs ended in empty-pool "
            f"(all arms pruned at κ<0.05 on calibration):[/yellow]"
        )
        console.print(empty.groupby(["cell", "protocol"]).size())

    lines = [
        "# CCB spatial-filter family sensitivity (BCI-IV-2b screening)",
        "",
        "Cells (only ``_SPATIAL_FILTERS`` varied; everything else held at the",
        "Phase-5 Stage-1 best-factorial CCB cell):",
        "",
        "| cell | spatial filters | n_arms (pre-prune) |",
        "|---|---|---|",
    ]
    sample_counts = (
        df.drop_duplicates(["cell", "subject"])
        .groupby("cell")["n_arms_total"]
        .median()
        .astype(int)
        .to_dict()
    )
    for cell_name, ss in _CELLS.items():
        lines.append(
            f"| {cell_name} | {', '.join(ss)} | {sample_counts.get(cell_name, '—')} |"
        )
    lines.append("")
    lines.append("Mean κ ± std across 9 subjects × seeds × 1 fold:")
    lines.append("")
    lines.append("| cell | within | official |")
    lines.append("|---|---|---|")
    for cell_name in _CELLS:
        def _fmt(row):
            if pd.isna(row["mean"]):
                return f"n/a ({int(empty_per_cell.get((cell_name, row.name), 0))} empty-pool)"
            return f"{row['mean']:+.3f} ± {row['std']:.3f}"
        empty_per_cell = (
            df[df["status"] == "empty_pool"].groupby(["cell", "protocol"]).size().to_dict()
        )
        w_row = summary.loc[(cell_name, "within")] if (cell_name, "within") in summary.index else None
        o_row = summary.loc[(cell_name, "official")] if (cell_name, "official") in summary.index else None
        w_str = _fmt(w_row) if w_row is not None else "—"
        o_str = _fmt(o_row) if o_row is not None else "—"
        lines.append(f"| {cell_name} | {w_str} | {o_str} |")
    lines.append("")
    lines.append(
        "Hyperparameters held fixed at the Phase-5 Stage-1 best cell "
        "(α=0.5, calibration_frac=0.3, window_size=50, arm_pool=pruned, "
        "include_recent_rewards=False); see open-justifications.md."
    )
    output_md.write_text("\n".join(lines) + "\n")
    console.log(f"Saved → {output_md}")


if __name__ == "__main__":
    typer.run(main)
