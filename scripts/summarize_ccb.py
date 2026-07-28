"""Gap-closure summary from CCB baseline + optional sensitivity CSVs.

Reads ``results/ccb_baseline.csv`` (required), optional sensitivity CSVs
(budget, alpha, arm_pool, calibration), and ``results/fbcsp_baseline.csv``
(for the 22-channel 2a benchmark numbers, filtered to
``dataset == "BCI-IV-2a"``). Produces ``results/ccb_baseline.md`` with:

  - Headline Δκ table (2a FBCSP − 2b CCB, both protocols, mean ± std).
  - Per-subject κ table comparing 2b CCB to 2b FBCSP (the in-2b reference).
  - Per-sensitivity-axis summary tables when the corresponding CSVs exist.

**No-leakage guard.** The 2a numbers are read only from the frozen
``results/fbcsp_baseline.csv`` and filtered to the 2a subset. A test in
``tests/test_ccb_runner.py`` greps this source to confirm the filter line
is present (see ``test_summarize_reads_only_2a_from_fbcsp_csv``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console

# No-leakage filter — do not remove. Asserted by tests.
_FBCSP_2A_DATASET_TAG = "BCI-IV-2a"


def _load_fbcsp_2a(fbcsp_csv: Path) -> pd.DataFrame:
    """Read the frozen FBCSP baseline CSV and filter to 2a rows.

    No 2b rows are ever read here; CCB's own numbers come from ccb_csv.
    """
    if not fbcsp_csv.exists():
        raise FileNotFoundError(f"FBCSP baseline CSV not found: {fbcsp_csv}")
    df = pd.read_csv(fbcsp_csv)
    # Explicit filter — referenced by the no-leakage test.
    df_2a = df[df["dataset"] == _FBCSP_2A_DATASET_TAG].copy()
    if df_2a.empty:
        raise RuntimeError(f"No rows with dataset == {_FBCSP_2A_DATASET_TAG!r} in {fbcsp_csv}")
    return df_2a


def _default_rows(ccb_df: pd.DataFrame) -> pd.DataFrame:
    """Filter CCB rows to the "default hyperparameters" slice for the headline.

    Phase-4 columns (``include_recent_rewards``, ``per_round_cap``) are filtered
    to their defaults when present, and silently skipped on older CSVs that
    pre-date those columns.
    """
    mask = (
        (ccb_df["budget_frac"] == 1.0)
        & (ccb_df["alpha"] == 1.0)
        & (ccb_df["arm_pool"] == "pruned")
        & (np.isclose(ccb_df["calibration_frac"], 0.3))
    )
    if "include_recent_rewards" in ccb_df.columns:
        mask &= ccb_df["include_recent_rewards"].astype(bool)
    if "per_round_cap" in ccb_df.columns:
        # "inf" is our sentinel for no cap; numeric caps bind the knapsack.
        mask &= ccb_df["per_round_cap"].astype(str).str.lower() == "inf"
    default = ccb_df[mask].copy()
    if default.empty:
        raise RuntimeError(
            "No default-hyperparameter rows found in CCB CSV. Expected "
            "budget_frac=1.0, alpha=1.0, arm_pool=pruned, calibration_frac=0.3, "
            "include_recent_rewards=True, per_round_cap=inf."
        )
    return default


def _per_subject_kappa(ccb_default: pd.DataFrame, protocol: str) -> pd.Series:
    """Mean κ per subject for ``protocol`` (averages over folds if multiple)."""
    sub = ccb_default[ccb_default["protocol"] == protocol]
    return sub.groupby("subject")["kappa"].mean()


def _aggregate_delta_kappa(ccb_default: pd.DataFrame, fbcsp_2a: pd.DataFrame) -> pd.DataFrame:
    """Per-subject Δκ = κ_2a_FBCSP − κ_2b_CCB per protocol."""
    rows = []
    fbcsp_2a_idx = fbcsp_2a.set_index("subject")
    for protocol in ("within", "official"):
        ccb_kappa = _per_subject_kappa(ccb_default, protocol)
        # FBCSP column naming from run_fbcsp_baseline.py.
        fbcsp_col = "kappa_within" if protocol == "within" else "kappa_official"
        if fbcsp_col not in fbcsp_2a_idx.columns:
            continue
        for subj in sorted(set(ccb_kappa.index) & set(fbcsp_2a_idx.index)):
            k_ccb = float(ccb_kappa.loc[subj])
            k_2a = float(fbcsp_2a_idx.loc[subj, fbcsp_col])
            rows.append(
                {
                    "subject": subj,
                    "protocol": protocol,
                    "kappa_2a_fbcsp": round(k_2a, 3),
                    "kappa_2b_ccb": round(k_ccb, 3),
                    "delta_kappa": round(k_2a - k_ccb, 3),
                }
            )
    return pd.DataFrame(rows)


def _fmt_mean_std(s: pd.Series) -> str:
    return f"{s.mean():.3f} ± {s.std():.3f}"


def _sensitivity_table_rows(ccb_df: pd.DataFrame, axis_name: str, axis_col: str) -> list[dict]:
    rows = []
    for value, group in ccb_df.groupby(axis_col):
        for protocol in ("within", "official"):
            sub = group[group["protocol"] == protocol]
            if sub.empty:
                continue
            rows.append(
                {
                    "axis": axis_name,
                    "value": value,
                    "protocol": protocol,
                    "kappa_mean": round(float(sub["kappa"].mean()), 3),
                    "kappa_std": round(float(sub["kappa"].std()), 3),
                    "n_subjects": int(sub["subject"].nunique()),
                }
            )
    return rows


_FACTORIAL_CELL_KEYS: tuple[str, ...] = (
    "alpha",
    "calibration_frac",
    "arm_pool",
    "include_recent_rewards",
    "per_round_cap",
    "budget_frac",
    "policy",
)


def _factorial_cell_label(cell: dict) -> str:
    """Human-readable label for a factorial hyperparameter cell."""
    return ", ".join(f"{k}={cell[k]}" for k in _FACTORIAL_CELL_KEYS if k in cell)


def _factorial_cell_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """For each factorial cell, compute per-seed mean κ per protocol and roll up.

    The return frame has one row per (cell, protocol) with:
      - ``kappa_mean`` — mean across seeds of the per-seed subject-averaged κ
      - ``kappa_seed_std`` — std across seeds of that same per-seed subject-averaged κ
      - ``n_seeds`` — distinct seeds present
      - ``n_subjects`` — distinct subjects present
    """
    # Step 1: reduce to per-(cell, protocol, seed) mean κ across subjects.
    present_keys = [k for k in _FACTORIAL_CELL_KEYS if k in df.columns]
    group_keys_a = [*present_keys, "protocol", "seed"]
    per_seed = df.groupby(group_keys_a)["kappa"].mean().reset_index()

    # Step 2: roll up across seeds.
    group_keys_b = [*present_keys, "protocol"]
    agg = per_seed.groupby(group_keys_b)["kappa"].agg(["mean", "std", "count"]).reset_index()
    agg = agg.rename(columns={"mean": "kappa_mean", "std": "kappa_seed_std", "count": "n_seeds"})

    # n_subjects per cell is constant across protocol; compute separately.
    n_subjects = df.groupby(present_keys)["subject"].nunique().rename("n_subjects")
    agg = agg.merge(n_subjects, on=list(present_keys))
    return agg


def _find_best_factorial_cell(df: pd.DataFrame) -> tuple[dict, pd.DataFrame] | tuple[None, None]:
    """Return the cell with the highest combined (within + official) mean κ.

    Combined objective: ``(kappa_mean_within + kappa_mean_official) / 2``. Cells
    missing either protocol are skipped.
    """
    agg = _factorial_cell_aggregate(df)
    if agg.empty:
        return None, None
    present_keys = [k for k in _FACTORIAL_CELL_KEYS if k in df.columns]
    pivot = agg.pivot_table(
        index=present_keys,
        columns="protocol",
        values="kappa_mean",
    )
    if not {"within", "official"}.issubset(pivot.columns):
        return None, None
    pivot["combined"] = (pivot["within"] + pivot["official"]) / 2.0
    best = pivot["combined"].idxmax()
    best_cell = dict(zip(present_keys, best, strict=True))

    # Collect the rows corresponding to that cell for downstream reporting.
    mask = pd.Series([True] * len(df), index=df.index)
    for k, v in best_cell.items():
        if k == "include_recent_rewards":
            mask &= df[k].astype(bool) == bool(v)
        elif k == "per_round_cap":
            mask &= df[k].astype(str).str.lower() == str(v).lower()
        else:
            mask &= df[k] == v
    return best_cell, df[mask].copy()


def _policy_ablation_rows(policy_df: pd.DataFrame) -> list[dict]:
    """Per-policy × protocol mean ± std κ across subjects × seeds."""
    rows: list[dict] = []
    for (policy, protocol), group in policy_df.groupby(["policy", "protocol"]):
        rows.append(
            {
                "policy": str(policy),
                "protocol": str(protocol),
                "kappa_mean": round(float(group["kappa"].mean()), 3),
                "kappa_std": round(float(group["kappa"].std()), 3),
                "n": int(len(group)),
                "n_subjects": int(group["subject"].nunique()),
            }
        )
    return rows


def _render_markdown(
    ccb_default: pd.DataFrame,
    fbcsp_2a: pd.DataFrame,
    delta_default: pd.DataFrame,
    sensitivity_rows: list[dict],
    fbcsp_baseline_csv: Path,
    ccb_tuned: pd.DataFrame | None = None,
    delta_tuned: pd.DataFrame | None = None,
    tuned_cell: str | None = None,
    policy_rows: list[dict] | None = None,
    ccb_factorial_best: pd.DataFrame | None = None,
    delta_factorial_best: pd.DataFrame | None = None,
    factorial_best_cell: str | None = None,
    factorial_best_seed_std: dict[str, float] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# CCB evaluation on BCI-IV-2b — headline + gap-closure")
    lines.append("")
    lines.append(
        f"Reads the 2a benchmark from `{fbcsp_baseline_csv}` "
        f"(filter `dataset == '{_FBCSP_2A_DATASET_TAG}'` — no-leakage enforced)."
    )
    lines.append("")

    # --- Summary ---
    lines.append("## Summary (mean ± std κ over 9 subjects)")
    lines.append("")
    lines.append("| Dataset / policy | κ within | κ official |")
    lines.append("|---|---|---|")
    for dataset, protocol_col_within, protocol_col_official in [
        ("BCI-IV-2a FBCSP (22-ch benchmark)", "kappa_within", "kappa_official"),
    ]:
        kw = _fmt_mean_std(fbcsp_2a[protocol_col_within])
        ko = _fmt_mean_std(fbcsp_2a[protocol_col_official])
        lines.append(f"| {dataset} | {kw} | {ko} |")
    k_w_ccb = ccb_default[ccb_default["protocol"] == "within"].groupby("subject")["kappa"].mean()
    k_o_ccb = ccb_default[ccb_default["protocol"] == "official"].groupby("subject")["kappa"].mean()
    lines.append(
        f"| BCI-IV-2b CCB (3-ch, default hyperparams) | {_fmt_mean_std(k_w_ccb)} | {_fmt_mean_std(k_o_ccb)} |"
    )
    if ccb_tuned is not None:
        k_w_t = ccb_tuned[ccb_tuned["protocol"] == "within"].groupby("subject")["kappa"].mean()
        k_o_t = ccb_tuned[ccb_tuned["protocol"] == "official"].groupby("subject")["kappa"].mean()
        tuned_label = (
            f"BCI-IV-2b CCB (3-ch, tuned: {tuned_cell})"
            if tuned_cell
            else "BCI-IV-2b CCB (3-ch, tuned)"
        )
        lines.append(f"| {tuned_label} | {_fmt_mean_std(k_w_t)} | {_fmt_mean_std(k_o_t)} |")
    if ccb_factorial_best is not None:
        k_w_f = (
            ccb_factorial_best[ccb_factorial_best["protocol"] == "within"]
            .groupby("subject")["kappa"]
            .mean()
        )
        k_o_f = (
            ccb_factorial_best[ccb_factorial_best["protocol"] == "official"]
            .groupby("subject")["kappa"]
            .mean()
        )
        label = (
            f"BCI-IV-2b CCB (3-ch, best-factorial: {factorial_best_cell})"
            if factorial_best_cell
            else "BCI-IV-2b CCB (3-ch, best-factorial)"
        )
        lines.append(f"| {label} | {_fmt_mean_std(k_w_f)} | {_fmt_mean_std(k_o_f)} |")
        if factorial_best_seed_std is not None:
            lines.append("")
            lines.append(
                "Best-factorial seed-std (across seeds {0, 1, 2, 3, 42}, "
                "subject-averaged κ per seed):"
            )
            for protocol in ("within", "official"):
                s = factorial_best_seed_std.get(protocol)
                if s is not None:
                    lines.append(f"- **{protocol}**: σ = **{s:.3f}**")
    lines.append("")

    # --- Gap-closure headline ---
    lines.append("## Gap-to-benchmark Δκ = κ<sub>2a FBCSP</sub> − κ<sub>2b CCB</sub>")
    lines.append("")
    lines.append(
        "**Default hyperparameters** (budget_frac=1.0, alpha=1.0, arm_pool=pruned, calibration_frac=0.3):"
    )
    for protocol in ("within", "official"):
        sub = delta_default[delta_default["protocol"] == protocol]
        if sub.empty:
            continue
        mean_delta = float(sub["delta_kappa"].mean())
        lines.append(
            f"- **{protocol}**: Δκ = **{mean_delta:+.3f}** "
            f"(per-subject mean across {len(sub)} subjects)"
        )
    if delta_tuned is not None and not delta_tuned.empty:
        lines.append("")
        lines.append(f"**Tuned** ({tuned_cell}):")
        for protocol in ("within", "official"):
            sub = delta_tuned[delta_tuned["protocol"] == protocol]
            if sub.empty:
                continue
            mean_delta = float(sub["delta_kappa"].mean())
            lines.append(
                f"- **{protocol}**: Δκ = **{mean_delta:+.3f}** "
                f"(per-subject mean across {len(sub)} subjects)"
            )
    if delta_factorial_best is not None and not delta_factorial_best.empty:
        lines.append("")
        lines.append(f"**Best-factorial** ({factorial_best_cell}):")
        for protocol in ("within", "official"):
            sub = delta_factorial_best[delta_factorial_best["protocol"] == protocol]
            if sub.empty:
                continue
            mean_delta = float(sub["delta_kappa"].mean())
            lines.append(
                f"- **{protocol}**: Δκ = **{mean_delta:+.3f}** "
                f"(per-subject mean across {len(sub)} subjects)"
            )
    lines.append("")

    # --- Per-subject Δκ ---
    lines.append("## Per-subject Δκ")
    lines.append("")
    lines.append("| Subject | Protocol | κ 2a FBCSP | κ 2b CCB (default) | Δκ |")
    lines.append("|---|---|---|---|---|")
    for _, r in delta_default.sort_values(["protocol", "subject"]).iterrows():
        lines.append(
            f"| {r['subject']} | {r['protocol']} | {r['kappa_2a_fbcsp']:.3f} | "
            f"{r['kappa_2b_ccb']:.3f} | {r['delta_kappa']:+.3f} |"
        )
    lines.append("")

    # --- Sensitivity sweep summaries ---
    if sensitivity_rows:
        lines.append("## Sensitivity sweeps (mean ± std κ per protocol × axis value)")
        lines.append("")
        for axis_name in sorted({r["axis"] for r in sensitivity_rows}):
            axis_data = [r for r in sensitivity_rows if r["axis"] == axis_name]
            if not axis_data:
                continue
            lines.append(f"### {axis_name}")
            lines.append("")
            lines.append("| value | protocol | κ (mean ± std) | n_subjects |")
            lines.append("|---|---|---|---|")
            for r in sorted(axis_data, key=lambda d: (str(d["value"]), d["protocol"])):
                lines.append(
                    f"| {r['value']} | {r['protocol']} | "
                    f"{r['kappa_mean']:.3f} ± {r['kappa_std']:.3f} | {r['n_subjects']} |"
                )
            lines.append("")

    # --- Policy ablation (§8.4) ---
    if policy_rows:
        lines.append("## Policy ablation — design-doc §8.4")
        lines.append("")
        lines.append(
            "Compares OPLB (default) against three drop-in alternatives: "
            "**fixed** (top-κ calibration arm, no exploration), "
            "**eps_greedy** (random ε-exploration instead of UCB), and "
            "**unconstrained** (OPLB with knapsack stripped). κ is the "
            "mean ± std across 9 subjects × seeds at the default "
            "hyperparameter cell."
        )
        lines.append("")
        lines.append("| policy | protocol | κ (mean ± std) | n rows | n_subjects |")
        lines.append("|---|---|---|---|---|")
        policy_order = {"oplb": 0, "fixed": 1, "eps_greedy": 2, "unconstrained": 3}
        for r in sorted(
            policy_rows,
            key=lambda d: (policy_order.get(d["policy"], 99), d["protocol"]),
        ):
            lines.append(
                f"| {r['policy']} | {r['protocol']} | "
                f"{r['kappa_mean']:.3f} ± {r['kappa_std']:.3f} | {r['n']} | "
                f"{r['n_subjects']} |"
            )
        lines.append("")

    lines.append("---")
    lines.append(
        "The CCB thesis contribution is to shrink Δκ on 2b while obeying the "
        "no-leakage constraint (never touching 2a at training time). Values "
        "reported here correspond to the default hyperparameter cell "
        "`(budget_frac=1.0, alpha=1.0, arm_pool=pruned, calibration_frac=0.3)`; "
        "other cells are tabulated in the sensitivity sections above."
    )
    return "\n".join(lines) + "\n"


def main(
    ccb_csv: Path = typer.Option(Path("results/ccb_baseline.csv")),
    fbcsp_csv: Path = typer.Option(Path("results/fbcsp_baseline.csv")),
    tuned_csv: Path = typer.Option(
        Path("results/ccb_tuned.csv"),
        help="Optional — tuned-hyperparameter CCB results; skipped if missing.",
    ),
    sensitivity_budget: Path = typer.Option(
        Path("results/ccb_sens_budget.csv"), help="Optional — skipped if missing."
    ),
    sensitivity_alpha: Path = typer.Option(
        Path("results/ccb_sens_alpha.csv"), help="Optional — skipped if missing."
    ),
    sensitivity_pool: Path = typer.Option(
        Path("results/ccb_sens_pool.csv"), help="Optional — skipped if missing."
    ),
    sensitivity_calibration: Path = typer.Option(
        Path("results/ccb_sens_calibration.csv"), help="Optional — skipped if missing."
    ),
    sensitivity_context: Path = typer.Option(
        Path("results/ccb_sens_context.csv"),
        help="Optional Phase-4 context-ablation CSV (d=18 vs d=15); skipped if missing.",
    ),
    sensitivity_perround: Path = typer.Option(
        Path("results/ccb_sens_perround.csv"),
        help="Optional Phase-4 per-round cost-cap CSV; skipped if missing.",
    ),
    policy_ablation: Path = typer.Option(
        Path("results/ccb_ablation_policy.csv"),
        help="Optional Phase-4 §8.4 policy ablation CSV (oplb/fixed/eps_greedy/"
        "unconstrained); skipped if missing.",
    ),
    factorial_csv: Path = typer.Option(
        Path("results/ccb_factorial.csv"),
        help="Optional Phase-4 full-factorial CSV. If present, the best cell "
        "(highest mean κ averaged over within + official) is reported in the "
        "headline with per-seed σ.",
    ),
    output_md: Path = typer.Option(Path("results/ccb_baseline.md")),
) -> None:
    console = Console()

    ccb_df = pd.read_csv(ccb_csv)
    fbcsp_2a = _load_fbcsp_2a(fbcsp_csv)

    ccb_default = _default_rows(ccb_df)
    delta_default = _aggregate_delta_kappa(ccb_default, fbcsp_2a)

    ccb_tuned: pd.DataFrame | None = None
    delta_tuned: pd.DataFrame | None = None
    tuned_cell: str | None = None
    if tuned_csv.exists():
        ccb_tuned = pd.read_csv(tuned_csv)
        delta_tuned = _aggregate_delta_kappa(ccb_tuned, fbcsp_2a)
        # Infer the tuned cell label from the CSV (all rows should share the same hyperparam values).
        row = ccb_tuned.iloc[0]
        tuned_cell = (
            f"budget_frac={row['budget_frac']}, alpha={row['alpha']}, "
            f"arm_pool={row['arm_pool']}, calibration_frac={row['calibration_frac']}"
        )
    else:
        console.log(f"[dim]No tuned CSV at {tuned_csv}; skipping tuned row.[/dim]")

    sensitivity_rows: list[dict] = []
    for axis_name, axis_col, path in [
        ("budget_frac", "budget_frac", sensitivity_budget),
        ("alpha", "alpha", sensitivity_alpha),
        ("arm_pool", "arm_pool", sensitivity_pool),
        ("calibration_frac", "calibration_frac", sensitivity_calibration),
        ("include_recent_rewards", "include_recent_rewards", sensitivity_context),
        ("per_round_cap", "per_round_cap", sensitivity_perround),
    ]:
        if path.exists():
            df_sens = pd.read_csv(path)
            if axis_col not in df_sens.columns:
                console.log(
                    f"[dim]Skipping sensitivity axis {axis_name!r}: "
                    f"column {axis_col!r} not in {path}[/dim]"
                )
                continue
            sensitivity_rows.extend(_sensitivity_table_rows(df_sens, axis_name, axis_col))
        else:
            console.log(f"[dim]Skipping sensitivity axis {axis_name!r} (no file at {path})[/dim]")

    policy_rows: list[dict] | None = None
    if policy_ablation.exists():
        policy_df = pd.read_csv(policy_ablation)
        if "policy" in policy_df.columns:
            policy_rows = _policy_ablation_rows(policy_df)
        else:
            console.log(
                f"[dim]Policy ablation CSV at {policy_ablation} lacks a 'policy' "
                "column; skipping §8.4 table.[/dim]"
            )
    else:
        console.log(f"[dim]No policy-ablation CSV at {policy_ablation}; skipping §8.4.[/dim]")

    # --- Factorial best cell (Phase 4) ---
    ccb_factorial_best: pd.DataFrame | None = None
    delta_factorial_best: pd.DataFrame | None = None
    factorial_best_cell: str | None = None
    factorial_best_seed_std: dict[str, float] | None = None
    if factorial_csv.exists():
        fac_df = pd.read_csv(factorial_csv)
        best_cell, best_rows = _find_best_factorial_cell(fac_df)
        if best_cell is not None and best_rows is not None:
            ccb_factorial_best = best_rows
            delta_factorial_best = _aggregate_delta_kappa(ccb_factorial_best, fbcsp_2a)
            factorial_best_cell = _factorial_cell_label(best_cell)
            # Compute per-seed σ for the best cell, across the 5 seeds.
            per_seed = (
                ccb_factorial_best.groupby(["protocol", "seed"])["kappa"].mean().reset_index()
            )
            factorial_best_seed_std = {
                protocol: float(per_seed[per_seed["protocol"] == protocol]["kappa"].std(ddof=1))
                for protocol in ("within", "official")
                if not per_seed[per_seed["protocol"] == protocol].empty
            }
            console.log(f"[green]Best factorial cell: {factorial_best_cell}[/green]")
        else:
            console.log(
                f"[dim]Factorial CSV at {factorial_csv} could not resolve a best cell "
                "(missing within/official columns?); skipping best-factorial row.[/dim]"
            )
    else:
        console.log(f"[dim]No factorial CSV at {factorial_csv}; skipping best-factorial row.[/dim]")

    md = _render_markdown(
        ccb_default,
        fbcsp_2a,
        delta_default,
        sensitivity_rows,
        fbcsp_csv,
        ccb_tuned=ccb_tuned,
        delta_tuned=delta_tuned,
        tuned_cell=tuned_cell,
        policy_rows=policy_rows,
        ccb_factorial_best=ccb_factorial_best,
        delta_factorial_best=delta_factorial_best,
        factorial_best_cell=factorial_best_cell,
        factorial_best_seed_std=factorial_best_seed_std,
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md)
    console.log(f"Wrote {output_md}")

    console.rule("Headline (default)")
    for protocol in ("within", "official"):
        sub = delta_default[delta_default["protocol"] == protocol]
        if sub.empty:
            continue
        console.log(f"{protocol}: mean Δκ = {sub['delta_kappa'].mean():+.3f} ({len(sub)} subjects)")
    if delta_tuned is not None:
        console.rule(f"Headline (tuned: {tuned_cell})")
        for protocol in ("within", "official"):
            sub = delta_tuned[delta_tuned["protocol"] == protocol]
            if sub.empty:
                continue
            console.log(
                f"{protocol}: mean Δκ = {sub['delta_kappa'].mean():+.3f} ({len(sub)} subjects)"
            )


if __name__ == "__main__":
    typer.run(main)
