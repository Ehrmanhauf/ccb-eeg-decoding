r"""Generate the manuscript figures from committed result CSVs.

Each figure is built from a committed ``results/*.csv`` into ``figures/*.pdf`` (a
generated, gitignored directory — rerun this script to rebuild). Figure
*interpretation* lives in the manuscript caption, not in the image: the plots carry
short, neutral titles only, as is standard for a scientific manuscript. Every function
is guarded — a figure whose feeder data is absent is skipped with a notice — so the
script can be run incrementally.

Sizing/readability: the target page is A4 with \textwidth ~= 5.9in at 12pt body text, so
each figure is authored close to its on-page display width (minimal down-scaling) and
uses a 12pt base font with a ~9.5pt annotation floor, so text stays legible on the page.

Figures (the sensitivity/best-arm/regret set plus the leakage-collapse / scoreboard / channel panels):
  3.1 dataset coverage grid          3.2 CCB stream-split flow          3.3 electrode montages
  4.1 sensitivity sweeps             4.2 best-arm vs CCB vs fixed       4.3 cumulative regret
  4.4 Delta-kappa forest             4.5 leakage collapse               4.6 scoreboard heatmap
  4.7 channel-count vs kappa

Run::

    PYTHONPATH=src .venv/bin/python scripts/figures/make_figures.py            # all
    PYTHONPATH=src .venv/bin/python scripts/figures/make_figures.py --only 4.2
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_RES = _REPO / "results"
_OUT = _REPO / "figures"
_OUT.mkdir(parents=True, exist_ok=True)

# Consistent, publication-oriented defaults. Element sizes are set here (not per call)
# so every figure shares one type scale; in-axes annotations use the FS_* floors below.
plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlepad": 8,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#3C4043",
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "lines.linewidth": 2.0,
})

# Vivid, colour-blind-safe qualitative palette — validated (worst adjacent CVD ΔE 25.0, well
# clear of the ≥12 target; run scripts/validate_palette.js). Distinguishable under the common
# colour-vision deficiencies and in greyscale print, while reading brighter than the old
# Okabe–Ito set. ONE semantic mapping is reused across every figure so a colour always means
# the same thing. (Low-contrast slots aqua/yellow always carry a direct value label — the
# "relief rule" — so identity is never colour-alone.)
C_FIXED = "#2a78d6"   # blue    — best fixed pipeline / reference series / full montage
C_ARM = "#1baf7a"     # aqua    — best single arm (frozen)
C_CCB = "#eb6834"     # orange  — CCB (our model), the warm focal series
C_EEG = "#4a3aa7"     # violet  — EEGNet
C_CLEAN = "#2a78d6"   # leakage-clean cell — blue (= fixed)
C_LEAK = "#9a9891"    # neutral grey — within-CV leak exhibit (muted, not trusted)
C_NEAR = "#eb6834"    # orange  — near-ear T7/T8 accent / 2nd dataset (visible in lines & scatter,
#                       unlike a pale yellow; never co-occurs with C_CCB in a single figure)

FS_ANNOT = 9.5        # in-axes annotation floor (readable after page down-scaling)


def _save(fig, name: str) -> None:
    p = _OUT / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ wrote {p.relative_to(_REPO)}")


def _have(*names: str) -> bool:
    missing = [n for n in names if not (_RES / n).exists()]
    if missing:
        print(f"  … skipped (missing {missing})")
    return not missing


# --------------------------------------------------------------------------- #
# Fig 3.1 — dataset coverage grid
# --------------------------------------------------------------------------- #
def fig_3_1() -> None:
    print("Fig 3.1 — dataset coverage grid")
    # (dataset, channels, paradigm row, hardware glyph, near-ear?)
    pts = [
        ("STEW", 14, "CL", "consumer", False),
        ("WAUC", 8, "CL", "consumer", False),
        ("UAB", 14, "CL", "consumer", False),
        ("UAB T7/T8", 2, "CL", "consumer", True),
        ("COG-BCI", 64, "CL", "research", False),
        ("COG-BCI T7/T8", 2, "CL", "research", True),
        ("BCI-IV-2b", 3, "MI", "bipolar", True),
        ("BCI-IV-2a", 22, "MI", "research", False),
        ("Cho2017", 64, "MI", "research", False),
        ("Cho2017 3ch", 3, "MI", "research", True),
    ]
    glyph = {"consumer": "o", "research": "s", "bipolar": "D"}
    rowy = {"MI": 0.0, "CL": 1.0}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    # Group datasets that share (paradigm, channel count) so co-located markers get
    # a deterministic vertical separation instead of overprinting.
    groups: dict[tuple, list] = defaultdict(list)
    for p in pts:
        groups[(p[2], p[1])].append(p)
    for (par, ch), members in groups.items():
        k = len(members)
        span = 0.30
        offs = [0.0] if k == 1 else [(-span / 2 + span * i / (k - 1)) for i in range(k)]
        for (name, _ch, _par, hw, near), off in zip(members, offs, strict=False):
            y = rowy[par] + off
            ax.scatter(ch, y, s=215 if near else 95, marker=glyph[hw],
                       facecolors=(C_NEAR if near else "white"),
                       edgecolors=(C_NEAR if near else C_FIXED),
                       linewidths=1.8, zorder=4)
            if k == 1:
                dy, va = -14, "top"
            elif off >= 0:
                dy, va = 13, "bottom"
            else:
                dy, va = -13, "top"
            ax.annotate(name, (ch, y), fontsize=FS_ANNOT, xytext=(0, dy),
                        textcoords="offset points", ha="center", va=va,
                        color=(C_NEAR if near else "#222222"),
                        fontweight=("bold" if near else "normal"))
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 8, 14, 22, 64])
    ax.set_xticklabels([2, 3, 8, 14, 22, 64])
    ax.set_xlim(1.6, 125)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Motor\nimagery", "Cognitive\nload"], fontweight="bold")
    ax.set_ylim(-0.95, 1.8)
    ax.set_xlabel("channel count (log scale)")
    ax.grid(axis="y", alpha=0)
    ax.grid(axis="x", alpha=0.16)
    handles = [
        mpatches.Patch(facecolor=C_NEAR, edgecolor=C_NEAR, label="near-ear T7/T8 subset"),
        plt.Line2D([], [], marker="o", color=C_FIXED, ls="", mfc="white", label="consumer dry/semi-dry"),
        plt.Line2D([], [], marker="s", color=C_FIXED, ls="", mfc="white", label="research wet"),
        plt.Line2D([], [], marker="D", color=C_FIXED, ls="", mfc="white", label="bipolar screening"),
    ]
    ax.legend(handles=handles, fontsize=9.0, loc="lower center", ncol=2,
              framealpha=0.96, edgecolor="#cccccc")
    _save(fig, "fig_3_1_coverage.pdf")


# --------------------------------------------------------------------------- #
# Fig 3.2 — CCB calibration / stream / frozen-test flow
# --------------------------------------------------------------------------- #
def fig_3_2() -> None:
    print("Fig 3.2 — CCB stream-split flow")
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    GREEN, YELL, RED, LAV, EDGE = "#dcefd9", "#fcf2cf", "#f7dcdc", "#e7e8f6", "#3a3a3a"

    def box(cx, cy, w, h, text, fc, fs=9.5):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=EDGE, lw=1.1, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3)
        return (cx, cy, w, h)

    def edge(b, s):
        cx, cy, w, h = b
        return {"r": (cx + w / 2, cy), "l": (cx - w / 2, cy),
                "t": (cx, cy + h / 2), "b": (cx, cy - h / 2)}[s]

    def arrow(b1, b2, s1="r", s2="l"):
        (x1, y1), (x2, y2) = edge(b1, s1), edge(b2, s2)
        ax.annotate("", (x2, y2), (x1, y1),
                    arrowprops={"arrowstyle": "-|>", "lw": 1.3, "color": EDGE, "shrinkA": 1, "shrinkB": 1})

    cx0, cx1, cx2, cx3 = 1.35, 4.35, 7.55, 10.7
    yt, yb, ym = 4.3, 1.5, 2.9
    train = box(cx0, ym, 2.1, 1.4, "Training\ntrials", LAV)
    calib = box(cx1, yt, 2.4, 1.35, "Calibration\n(30%)", GREEN)
    strm = box(cx1, yb, 2.4, 1.35, "Bandit stream\n(70%)", YELL)
    fitp = box(cx2, yt, 2.7, 1.45, "Fit + prune\nper-arm heads\n($\\kappa\\geq0.05$)", GREEN, fs=8.6)
    oplb = box(cx2, yb, 3.0, 2.0,
               "OPLB loop:\n$\\varphi(x_t)\\!\\to\\!$ feasible\n$\\to$ UCB select\n$\\to$ frozen head $\\to$ reward",
               YELL, fs=8.2)
    test = box(cx3, ym, 2.2, 1.6, "Frozen test\n$\\alpha\\!\\to\\!0$\nscore $\\kappa$", RED)
    arrow(train, calib)
    arrow(train, strm)
    arrow(calib, fitp)
    arrow(strm, oplb)
    arrow(fitp, oplb, "b", "t")
    arrow(oplb, test)
    ax.text(cx2, 0.30, "exploration cost paid here", fontsize=8.5, color="#9a6a00",
            ha="center", style="italic")
    ax.text(cx3, 1.72, "$\\kappa$ measured here\n(exploration off)", fontsize=8.5,
            color=C_CCB, ha="center", va="top", style="italic")
    _save(fig, "fig_3_2_streamsplit.pdf")


# --------------------------------------------------------------------------- #
# Fig 4.1 — single-parameter sensitivity sweeps (load-bearing)
# --------------------------------------------------------------------------- #
def fig_4_1() -> None:
    print("Fig 4.1 — sensitivity sweeps")
    # Each parameter is rendered on an evenly spaced categorical axis so that the
    # stationary / no-cap setting (inf) — the locked default — is shown alongside
    # the finite values instead of being dropped by a log axis.
    sweeps = [("alpha", "alpha", r"$\alpha$ (exploration)"),
              ("calibration", "calibration_frac", "calibration fraction"),
              ("window", "window_size", "sliding window (rounds)"),
              ("cap", "per_round_cap", "per-round cost cap")]
    datasets = {"STEW": ("ccb_stew_sens_{}.csv", C_FIXED, 0.953),
                "WAUC": ("ccb_wauc_sens_{}.csv", C_NEAR, 0.785)}
    if not _have(*[f"ccb_stew_sens_{s[0]}.csv" for s in sweeps]):
        return

    def _fmt(v):
        return r"$\infty$" if not np.isfinite(v) else f"{v:g}"

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 6.4))
    for ax, (skey, col, xlabel) in zip(axes.ravel(), sweeps, strict=False):
        frames, catvals = {}, set()
        for dname, (tmpl, _color, _fixed) in datasets.items():
            f = _RES / tmpl.format(skey)
            if not f.exists():
                continue
            d = pd.read_csv(f)
            d = d[d.protocol == "within"] if "protocol" in d.columns else d
            if "discount_gamma" in d.columns:
                d = d[d.discount_gamma == 1.0]
            d = d.copy()
            d[col] = d[col].astype(float)  # "inf" parses to np.inf
            frames[dname] = d
            catvals.update(d[col].unique().tolist())
        cats = sorted(catvals)  # inf sorts last
        pos = {c: i for i, c in enumerate(cats)}
        for dname, d in frames.items():
            color, fixed = datasets[dname][1], datasets[dname][2]
            g = d.groupby(col)["kappa"].agg(["mean", "std"])
            xs = [pos[c] for c in g.index]
            ax.errorbar(xs, g["mean"].values, yerr=g["std"].values, marker="o", ms=6,
                        color=color, capsize=3, lw=1.8, label=dname, zorder=3)
            ax.axhline(fixed, ls="--", color=color, alpha=0.55, lw=1.2, zorder=1)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([_fmt(c) for c in cats])
        ax.set_xlim(-0.35, len(cats) - 0.65)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\kappa$")
        ax.set_title(skey)
        ax.set_ylim(0.1, 1.05)
    axes[0, 0].text(0.03, 0.06, "dashed = best fixed-pipeline $\\kappa$",
                    transform=axes[0, 0].transAxes, fontsize=8.5, color="#555555")
    handles = [plt.Line2D([], [], color=C_FIXED, marker="o", lw=1.8, label="STEW (CCB)"),
               plt.Line2D([], [], color=C_NEAR, marker="o", lw=1.8, label="WAUC (CCB)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=10.5,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "fig_4_1_sweeps.pdf")


# --------------------------------------------------------------------------- #
# Fig 4.2 — best-arm vs CCB vs fixed (key diagnostic)
# --------------------------------------------------------------------------- #
_NICE = {"UAB": "UAB", "COGBCI": "COG-BCI"}
_MONT = {"full": "full", "nearear": "near-ear"}


def fig_4_2() -> None:
    print("Fig 4.2 — best-arm vs CCB vs fixed")
    if not _have("best_arm_diagnostic.csv", "ccb_newdata.csv", "fixed_baseline_newdata.csv"):
        return
    ba = pd.read_csv(_RES / "best_arm_diagnostic.csv")
    ccb = pd.read_csv(_RES / "ccb_newdata.csv")
    fx = pd.read_csv(_RES / "fixed_baseline_newdata.csv")
    cells, best_fixed, best_arm, ccb_k = [], [], [], []
    # Leakage-clean existing cells first (fixed/CCB traced to the committed Ch4 tables).
    if (_RES / "best_arm_existing.csv").exists():
        be = pd.read_csv(_RES / "best_arm_existing.csv")
        for label, fixed, cc in [("BCI-IV-2b\n(clean)", 0.292, 0.184), ("WAUC\n(clean)", 0.785, 0.426)]:
            key = label.split("\n")[0]
            g = be[be.dataset == key].dropna(subset=["best_arm_kappa"])
            if len(g):
                cells.append(label)
                best_fixed.append(fixed)
                best_arm.append(g["best_arm_kappa"].mean())
                ccb_k.append(cc)
    for (ds, mt), g in ba.groupby(["dataset", "montage"]):
        cells.append(f"{_NICE.get(ds, ds)}\n{_MONT.get(mt, mt)}")
        best_arm.append(g["best_arm_kappa"].mean())
        ccb_k.append(ccb[(ccb.dataset == ds) & (ccb.montage == mt)]["kappa"].mean())
        ff = fx[(fx.dataset == ds) & (fx.montage == mt)].dropna(subset=["kappa"])
        best_fixed.append(ff.groupby(["feature_family", "classifier"])["kappa"].mean().max())
    y = np.arange(len(cells))
    h = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 0.74 * len(cells) + 1.1))
    ax.barh(y + h, best_fixed, h, label="best fixed pipeline (B1–B5)", color=C_FIXED)
    ax.barh(y, best_arm, h, label="best single arm (frozen)", color=C_ARM)
    ax.barh(y - h, ccb_k, h, label="CCB", color=C_CCB)
    ax.set_yticks(y)
    ax.set_yticklabels(cells, fontsize=10.0)
    ax.set_xlabel(r"$\kappa$ (within-subject CV)")
    ax.set_xlim(0, 1.02)
    ax.grid(axis="y", alpha=0)
    ax.legend(loc="lower right")
    _save(fig, "fig_4_2_bestarm.pdf")


# --------------------------------------------------------------------------- #
# Fig 4.3 — cumulative regret
# --------------------------------------------------------------------------- #
def fig_4_3() -> None:
    print("Fig 4.3 — cumulative regret")
    if not _have("regret_curves.csv"):
        return
    d = pd.read_csv(_RES / "regret_curves.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    palette = {"full": C_FIXED, "nearear": C_NEAR}
    for (ds, mt), g in d.groupby(["dataset", "montage"]):
        piv = g.pivot_table("cumulative_regret", "round", "subject")
        # Plot only rounds where *every* subject is still streaming, so each mean
        # averages the same cohort (avoids the drop-out tail artefact).
        full_rows = piv.dropna(axis=0, how="any")
        if full_rows.empty:
            full_rows = piv.loc[piv.notna().sum(axis=1) >= 0.9 * piv.shape[1]].ffill()
        m = full_rows.mean(axis=1)
        s = full_rows.std(axis=1)
        col = palette.get(mt, "C0")
        ax.plot(m.index, m.values, color=col, lw=1.8, label=f"{_NICE.get(ds, ds)} / {_MONT.get(mt, mt)}")
        ax.fill_between(m.index, m - s, m + s, alpha=0.15, color=col)
    ax.set_xlabel("bandit-stream round $t$")
    ax.set_ylabel("cumulative regret $R(t)$")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    _save(fig, "fig_4_3_regret.pdf")


# --------------------------------------------------------------------------- #
# Fig 4.4 — cross-dataset Δκ forest plot
# --------------------------------------------------------------------------- #
def fig_4_4() -> None:
    print("Fig 4.4 — Δκ forest plot")
    rows = [("STEW (within, leak exhibit)", 0.209, "leak"),
            ("WAUC (within, leak exhibit)", 0.359, "leak"),
            ("BCI-IV-2b (within)", 0.108, "clean"),
            ("BCI-IV-2a (within)", 0.251, "clean"),
            ("Cho2017-full (within)", 0.125, "clean"),
            ("Cho2017-3ch (within)", 0.108, "clean")]
    try:
        fx = pd.read_csv(_RES / "fixed_baseline_newdata.csv")
        ccb = pd.read_csv(_RES / "ccb_newdata.csv")
        for ds, mt in [("UAB", "full"), ("UAB", "nearear"), ("COGBCI", "full"), ("COGBCI", "nearear")]:
            bf = fx[(fx.dataset == ds) & (fx.montage == mt)].dropna(subset=["kappa"]).groupby(
                ["feature_family", "classifier"])["kappa"].mean().max()
            ck = ccb[(ccb.dataset == ds) & (ccb.montage == mt)]["kappa"].mean()
            tag = "near" if mt == "nearear" else "leak"
            rows.append((f"{_NICE.get(ds, ds)} {_MONT[mt]} (within)", bf - ck, tag))
        cs = pd.read_csv(_RES / "crosssession_cogbci.csv")
        for mt in ["nearear", "full"]:
            v = cs[(cs.montage == mt) & (cs.method.isin(["fixed", "ccb"]))].dropna(subset=["kappa"])
            bf = v[v.method == "fixed"].groupby(["feature_family", "classifier"])["kappa"].mean().max()
            ck = v[v.method == "ccb"]["kappa"].mean()
            rows.append((f"COG-BCI {_MONT[mt]} (cross-session)", bf - ck, "near" if mt == "nearear" else "clean"))
    except FileNotFoundError:
        pass
    rows.sort(key=lambda r: r[1])
    colors = {"clean": C_CLEAN, "leak": C_LEAK, "near": C_NEAR}
    taglab = {"clean": "leakage-clean cell", "leak": "leak exhibit (within-CV)",
              "near": "near-ear T7/T8"}
    fig, ax = plt.subplots(figsize=(6.8, 0.46 * len(rows) + 1.2))
    for i in range(0, len(rows), 2):  # subtle alternating bands aid row tracking
        ax.axhspan(i - 0.5, i + 0.5, color="#f5f6f8", zorder=0)
    ax.axvline(0, color="#888888", lw=0.9, zorder=1)
    vmax = max(v for _, v, _ in rows)
    for i, (_lab, v, tag) in enumerate(rows):
        ax.plot([0, v], [i, i], color=colors[tag], lw=2.6, alpha=0.9,
                solid_capstyle="round", zorder=2)
        ax.scatter(v, i, color=colors[tag], s=75, zorder=3, edgecolors="white", linewidths=0.9)
        ax.text(v + 0.008, i, f"{v:+.2f}", va="center", ha="left", fontsize=FS_ANNOT, color="#333333")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(r"$\Delta\kappa$ (best fixed $-$ CCB)  —  positive: CCB underperforms")
    ax.set_xlim(-0.02, vmax + 0.07)
    ax.grid(axis="y", alpha=0)
    ax.grid(axis="x", alpha=0.18)
    handles = [mpatches.Patch(color=colors[t], label=taglab[t]) for t in ("clean", "leak", "near")]
    ax.legend(handles=handles, fontsize=9.5, loc="lower right", framealpha=0.95, edgecolor="#cccccc")
    _save(fig, "fig_4_4_forest.pdf")


# --------------------------------------------------------------------------- #
# Fig 4.5 — leakage collapse: within-CV ceiling vs leakage-clean protocols (added)
# --------------------------------------------------------------------------- #
def fig_4_5() -> None:
    print("Fig 4.5 — leakage collapse")
    if not _have("crosssession_cogbci.csv", "loso_stew.csv", "fixed_baseline_newdata.csv"):
        return
    # Best fixed-pipeline kappa: within-CV ceiling vs the leakage-clean protocol.
    cs = pd.read_csv(_RES / "crosssession_cogbci.csv")

    def _bf(df):
        df = df[df.method == "fixed"].dropna(subset=["kappa"]) if "method" in df.columns else df.dropna(subset=["kappa"])
        return float(df.groupby(["feature_family", "classifier"])["kappa"].mean().max())

    fx_new = pd.read_csv(_RES / "fixed_baseline_newdata.csv")

    def _bf_new(ds, mt):
        d = fx_new[(fx_new.dataset == ds) & (fx_new.montage == mt)].dropna(subset=["kappa"])
        return float(d.groupby(["feature_family", "classifier"])["kappa"].mean().max())

    loso = pd.read_csv(_RES / "loso_stew.csv")
    # Best fixed-pipeline LOSO kappa for STEW = max over the fixed feature families.
    lf = loso[loso.method.isin(["fbcsp", "bandpower"])].dropna(subset=["kappa"])
    loso_fixed = float(lf.groupby("method").kappa.mean().max()) if len(lf) else None

    # WAUC LOSO (same block-design-leakage test); within-CV ceiling = best B1/B2 sLDA
    # within-CV kappa (max(0.658, 0.644) from results/fixed_baseline_cl.csv).
    loso_w_fixed = None
    if _have("loso_wauc.csv"):
        lw = pd.read_csv(_RES / "loso_wauc.csv")
        lwf = lw[lw.method.isin(["fbcsp", "bandpower"])].dropna(subset=["kappa"])
        loso_w_fixed = float(lwf.groupby("method").kappa.mean().max()) if len(lwf) else None

    cells = []  # (label with clean protocol, within_ceiling, clean_value)
    cells.append(("COG-BCI full\n(cross-session)", _bf_new("COGBCI", "full"), _bf(cs[cs.montage == "full"])))
    cells.append(("COG-BCI near-ear\n(cross-session)", _bf_new("COGBCI", "nearear"), _bf(cs[cs.montage == "nearear"])))
    if loso_fixed is not None:
        cells.append(("STEW\n(LOSO)", 0.953, loso_fixed))
    if loso_w_fixed is not None:
        cells.append(("WAUC\n(LOSO)", 0.658, loso_w_fixed))

    labels = [c[0] for c in cells]
    within = [c[1] for c in cells]
    clean = [c[2] for c in cells]
    x = np.arange(len(cells))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    b1 = ax.bar(x - w / 2, within, w, label="within-subject CV (leakage-confounded ceiling)", color=C_LEAK)
    b2 = ax.bar(x + w / 2, clean, w, label="leakage-clean protocol (cross-session / LOSO)", color=C_CLEAN)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=FS_ANNOT)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.0)
    ax.set_ylabel(r"best fixed-pipeline $\kappa$")
    ax.set_ylim(0, 1.30)
    ax.grid(axis="x", alpha=0)
    ax.legend(fontsize=9.5, loc="upper center", ncol=1, framealpha=0.95)
    _save(fig, "fig_4_5_leakage.pdf")


def _draw_head(ax) -> None:
    """Schematic top-down head: outline circle, nose triangle, two ears (nose up)."""
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#555555", lw=1.5, zorder=1))
    ax.plot([-0.13, 0, 0.13], [0.99, 1.16, 0.99], color="#555555", lw=1.5, zorder=1)  # nose
    for sgn in (-1, 1):  # ears
        ax.add_patch(mpatches.FancyBboxPatch(
            (sgn * 1.0 - 0.03, -0.13), 0.06, 0.26,
            boxstyle="round,pad=0.02", fill=False, color="#555555", lw=1.4, zorder=1))


def fig_3_3() -> None:
    """EEG montages across the panel — the near-ear T7/T8 pair as the 2-channel subset.

    Pure-geometry figure (no results CSV): electrode positions from the MNE
    standard_10-05 montage, orthographic top-down projection (nose up, +x = right,
    so T7 is left / T8 is right), with the near-ear deployment pair highlighted.
    Laid out 2x2 (rather than 1x4) so each head is large enough on the A4 page for the
    electrode markers and T7/T8 labels to read clearly.
    """
    import mne

    EMOTIV14 = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8",
                "FC6", "F4", "F8", "AF4"]
    ENOBIO8 = ["AF8", "Fp2", "Fp1", "AF7", "T10", "T9", "P4", "P3"]
    COGBCI62 = ["Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "CP5", "CP1",
                "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "TP10", "CP6", "CP2",
                "FCz", "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8", "Fp2", "AF7", "AF3",
                "AFz", "F1", "F5", "FT7", "FC3", "C1", "C5", "TP7", "CP3", "P1", "P5",
                "PO7", "PO3", "POz", "PO4", "PO8", "P6", "P2", "CPz", "CP4", "TP8", "C6",
                "C2", "FC4", "FT8", "F6", "AF8", "AF4", "F2"]
    NEAR = {"T7", "T8"}

    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    used = set(EMOTIV14) | set(ENOBIO8) | set(COGBCI62) | NEAR
    scale = max(float(np.hypot(pos[c][0], pos[c][1])) for c in used if c in pos)

    def to2d(ch):
        v = pos.get(ch)
        if v is None:
            return None
        return 0.92 * v[0] / scale, 0.92 * v[1] / scale

    panels = [
        ("COG-BCI\n62-ch research cap", COGBCI62, NEAR),
        ("STEW / UAB\n14-ch consumer (Emotiv)", EMOTIV14, NEAR),
        ("WAUC\n8-ch (Enobio)", ENOBIO8, set()),
        ("Near-ear deployment\nmontage (T7/T8)", ["T7", "T8"], NEAR),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.4))
    for ax, (title, chs, hi) in zip(axes.ravel(), panels, strict=False):
        _draw_head(ax)
        for ch in chs:
            p = to2d(ch)
            if p is None:
                continue
            near = ch in hi
            ax.scatter(*p, s=150 if near else 40,
                       c=(C_NEAR if near else "white"),
                       edgecolors=(C_NEAR if near else "#555555"),
                       linewidths=1.8 if near else 1.0, zorder=(5 if near else 3))
            if near:
                ax.annotate(ch, p, textcoords="offset points", xytext=(0, 11),
                            ha="center", fontsize=11, fontweight="bold",
                            color=C_NEAR, zorder=6)
        ax.set_title(f"{title}  ({len(chs)} ch)", fontsize=11.5)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.40)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(
        "Electrode montages across the panel — the near-ear T7/T8 pair (orange) is the "
        "two-channel deployment subset of the richer montages",
        fontsize=12.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "fig_3_3_montage.pdf")


def fig_4_6() -> None:
    """Master scoreboard heatmap: every (dataset, protocol) cell x method, coloured by kappa.

    The visual twin of tab:master (same source, results_master.csv). The leakage-clean cells
    (LOSO, cross-session) read pale; the within-CV exhibits (marked) saturate dark --- the
    collapse, at a glance.
    """
    if not _have("results_master.csv"):
        return
    df = pd.read_csv(_RES / "results_master.csv")
    M = df[["fixed", "ccb", "eeg"]].to_numpy(dtype=float)
    labels = [f'{r.ds} · {r.prot}' + (" †" if r.leak else "") for r in df.itertuples()]
    fig, ax = plt.subplots(figsize=(5.6, 7.1))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#E9E9E9")
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Best fixed", "CCB", "EEGNet"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "---" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                    fontsize=10.0, color="white" if (not np.isnan(v) and v > 0.55) else "#222222")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"Cohen's $\kappa$")
    ax.set_title(r"Per-cell decoding scoreboard ($\dagger$ = within-CV leak exhibit)", fontsize=11.5)
    ax.grid(False)
    _save(fig, "fig_4_6_scoreboard.pdf")


def fig_4_7() -> None:
    """Leakage-clean best-fixed kappa vs channel count: decodability tracks task, not channels."""
    if not _have("results_master.csv"):
        return
    df = pd.read_csv(_RES / "results_master.csv")
    d = df[(~df.leak.astype(bool)) & df.fixed.notna()].copy()
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    colors = {"CL": C_NEAR, "MI": C_FIXED}
    for par, g in d.groupby("par"):
        ax.scatter(g.ch, g.fixed, s=85, color=colors.get(par, "#888888"), label=par,
                   zorder=3, edgecolors="white", linewidths=1.0)
    for r in d.itertuples():
        ax.annotate(r.ds.replace("COG-BCI ", "").replace(" (compet.)", ""),
                    (r.ch, r.fixed), fontsize=8.5, xytext=(6, 3), textcoords="offset points")
    ax.axhline(0, color="#999999", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 3, 8, 14, 22, 64])
    ax.set_xticklabels(["2", "3", "8", "14", "22", "64"])
    ax.set_xlabel("channel count (log scale)")
    ax.set_ylabel(r"leakage-clean best-fixed $\kappa$")
    ax.set_title("Channel count does not determine deployment-regime decodability", fontsize=11.5)
    ax.legend(title="paradigm", loc="upper left")
    _save(fig, "fig_4_7_channels.pdf")


def fig_deck_guards() -> None:
    """Deck slide-7 schematic (English): the no-leakage discipline, enforced in code.
    Two guards in one flow — (1) high-channel never informs low-channel (near-ear by position),
    and (2) one train-only, byte-identical split feeds every method family."""
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, *, tc="#141414", fs=10, weight="bold"):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.16",
            fc=fc, ec=ec, lw=1.7, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight=weight, zorder=3)

    def arrow(x1, y1, x2, y2, color="#8a8880", lw=2.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2))

    # --- Guard 1: a high-channel recording never informs a low-channel pipeline ---
    box(0.3, 5.9, 3.1, 1.4, "Dense cap\n62-channel", "#e9f1fb", C_FIXED)
    box(6.6, 5.9, 3.1, 1.4, "Near-ear\nT7/T8 · 2-channel", "#fdeee4", C_NEAR)
    arrow(3.5, 6.6, 6.5, 6.6, color="#b6b4ad", lw=2.4)
    cx, cy = 5.0, 6.6
    ax.add_patch(plt.Circle((cx, cy), 0.44, fc="white", ec="#d03b3b", lw=2.6, zorder=4))
    ax.plot([cx - 0.31, cx + 0.31], [cy + 0.31, cy - 0.31], color="#d03b3b", lw=2.6, zorder=5)
    ax.text(5.0, 5.25, "near-ear by electrode position — dense cap never distilled",
            ha="center", va="center", fontsize=8.6, color="#333333", style="italic")

    # --- Guard 2: one train-only, byte-identical split feeds every method (single box, no
    #     fanning arrows that would cross the method boxes) ---
    box(0.3, 3.0, 2.3, 1.1, "TRAIN", "#e9f1fb", C_FIXED)
    box(2.6, 3.0, 1.5, 1.1, "TEST", "#f3f3f1", "#9a9891", tc="#555555")
    arrow(1.45, 2.9, 1.45, 1.85, color=C_FIXED, lw=1.8)
    ax.text(1.45, 1.55, "CSP + standardisers\nfit on TRAIN only", ha="center", va="top",
            fontsize=8.3, color=C_FIXED, weight="bold")
    arrow(4.3, 3.55, 5.75, 3.55, color="#b6b4ad", lw=2.2)
    ax.add_patch(mpatches.FancyBboxPatch(
        (5.9, 3.0), 3.8, 1.1, boxstyle="round,pad=0.06,rounding_size=0.16",
        fc="white", ec="#8a8880", lw=1.6, zorder=2))
    for i, (lab, col) in enumerate([("CCB", C_CCB), ("Fixed", C_FIXED), ("EEGNet", C_EEG)]):
        ax.text(6.55 + i * 1.28, 3.55, lab, ha="center", va="center",
                fontsize=9.5, color=col, weight="bold", zorder=3)
    ax.text(7.8, 2.35, "byte-identical splits — one source", ha="center", va="center",
            fontsize=8.6, color="#333333", style="italic")

    ax.set_title("No-leakage safeguards, enforced mechanically", fontsize=11.5, weight="bold")
    _save(fig, "fig_deck_guards.pdf")


def fig_deck_leakage() -> None:
    """Deck slide-5 schematic: recording-identity leakage under random k-fold vs. holding out
    the whole recording (cross-session / LOSO). One continuous recording per load level."""
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))
    seg = 15
    for ax, (title, mode) in zip(axes, [("Random $k$-fold CV", "leak"),
                                        ("Hold out the whole recording", "clean")], strict=True):
        ax.set_xlim(-0.5, seg + 0.5)
        ax.set_ylim(-1.6, 3.2)
        ax.axis("off")
        ax.set_title(title, fontsize=10.5, weight="bold", pad=4)
        if mode == "leak":
            rng = np.random.default_rng(1)
            assign = rng.integers(0, 2, seg)
            for i in range(seg):
                ax.add_patch(plt.Rectangle((i, 1.3), 0.9, 0.95,
                             fc=(C_FIXED if assign[i] == 0 else C_CCB), ec="white", lw=0.8))
            ax.annotate("", xy=(seg + 0.2, 1.05), xytext=(-0.2, 1.05),
                        arrowprops=dict(arrowstyle="-", color="#bbb", lw=1.0))
            ax.text(seg / 2, 2.7, "one continuous recording (one load level)",
                    ha="center", fontsize=8.6, color="#333333")
            ax.text(seg / 2, 0.5, "train & test drawn from the same recording",
                    ha="center", va="top", fontsize=8.7, color="#c0392b")
            ax.text(seg / 2, -0.75, "model scores by recording identity",
                    ha="center", fontsize=9.0, color="#c0392b", weight="bold")
        else:
            per = seg // 3
            for i in range(seg):
                held = (i // per) >= 2
                ax.add_patch(plt.Rectangle((i, 1.3), 0.9, 0.95,
                             fc=(C_CCB if held else C_FIXED), ec="white", lw=0.8))
            ax.text(seg / 2, 2.7, "three recordings — one held out entirely",
                    ha="center", fontsize=8.6, color="#333333")
            ax.text(per, 0.5, "TRAIN", ha="center", va="top", fontsize=8.8,
                    color=C_FIXED, weight="bold")
            ax.text(per * 2 + per / 2, 0.5, "TEST\n(unseen recording)", ha="center", va="top",
                    fontsize=8.4, color=C_CCB, weight="bold")
            ax.text(seg / 2, -0.75, "leakage-clean estimate",
                    ha="center", fontsize=9.0, color="#1e8449", weight="bold")
    axes[1].legend(handles=[mpatches.Patch(fc=C_FIXED, label="train window"),
                            mpatches.Patch(fc=C_CCB, label="test window")],
                   loc="lower center", bbox_to_anchor=(-0.08, -0.34), ncol=2,
                   fontsize=8.5, frameon=False)
    fig.suptitle("Recording-identity leakage — hold out the unit, not random windows",
                 fontsize=11.5, weight="bold", y=1.0)
    _save(fig, "fig_deck_leakage.pdf")


def fig_deck_arms() -> None:
    """Deck slide-10 schematic: the factorial arm bank + context-driven per-trial selection."""
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    factors = [("9\nsub-bands", C_FIXED), ("2 spatial\n(CSP, identity)", C_ARM),
               ("2 features\n(log-var,\nlog-power)", C_CCB), ("3\nwindows", C_EEG)]
    x = 0.2
    for i, (lab, col) in enumerate(factors):
        ax.add_patch(mpatches.FancyBboxPatch((x, 2.5), 1.9, 1.8,
                     boxstyle="round,pad=0.05,rounding_size=0.14", fc="white", ec=col, lw=1.9))
        ax.text(x + 0.95, 3.4, lab, ha="center", va="center", fontsize=8.8, color=col, weight="bold")
        x += 1.9
        ax.text(x + 0.28, 3.4, "$\\times$" if i < 3 else "$=$", ha="center", va="center",
                fontsize=16, color="#555555")
        x += 0.62
    ax.add_patch(mpatches.FancyBboxPatch((x, 2.5), 2.3, 1.8,
                 boxstyle="round,pad=0.05,rounding_size=0.14", fc="#fdeee4", ec=C_CCB, lw=2.4))
    ax.text(x + 1.15, 3.4, "108\npipeline arms", ha="center", va="center",
            fontsize=11, color="#111111", weight="bold")
    ax.text(6.5, 1.35, "context $\\phi$  $\\rightarrow$  OPLB policy selects one arm per trial",
            ha="center", va="center", fontsize=9.6, color="#333333")
    ax.text(6.5, 0.5, "(162 arms on the 3-channel BCI-IV-2b bank, which adds a Laplacian filter)",
            ha="center", va="center", fontsize=8.0, color="#777777", style="italic")
    ax.set_title("The arm bank: a factorial of feature-extraction pipelines",
                 fontsize=11.5, weight="bold")
    _save(fig, "fig_deck_arms.pdf")


_FIGS = {"3.1": fig_3_1, "3.2": fig_3_2, "3.3": fig_3_3, "4.1": fig_4_1, "4.2": fig_4_2,
         "4.3": fig_4_3, "4.4": fig_4_4, "4.5": fig_4_5, "4.6": fig_4_6, "4.7": fig_4_7,
         "guards": fig_deck_guards, "leakage": fig_deck_leakage, "arms": fig_deck_arms}


def main(only: str = typer.Option("", help="Comma-separated figure ids (e.g. 4.1,4.4); empty = all.")) -> None:
    wanted = [x.strip() for x in only.split(",") if x.strip()] or list(_FIGS)
    for fid in wanted:
        if fid in _FIGS:
            _FIGS[fid]()
        else:
            print(f"  unknown figure {fid!r}")


if __name__ == "__main__":
    typer.run(main)
