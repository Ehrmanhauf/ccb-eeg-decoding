r"""Build three self-contained, offline-portable HTML demos for the thesis defense.

Each demo is a single ``.html`` file with Plotly inlined (``include_plotlyjs=True``)
and the data baked in -- no server, no internet. They are a *visualization layer*
over the committed result CSVs (demos 1-2) plus one cached OPLB run (demo 3); they
introduce no new science and the numbers match the thesis figures/tables.

  demos/leakage_collapse.html  -- toggle within-CV <-> leakage-clean; the ceiling collapses
  demos/best_arm.html          -- pick a cell; best fixed / best arm / CCB + the two gaps
  demos/oplb_stream.html        -- animate one OPLB run: arm pulls + cumulative regret + final kappa

Design: colourblind-safe Okabe-Ito palette, a single typographic system, Landis-Koch
interpretation bands so the kappa axis is readable, and semantic encoding (the illusory
within-CV ceiling is rendered hollow/hatched; the leakage-clean truth is solid).

Run:  PYTHONPATH=src .venv/bin/python scripts/build_demos.py            (build all)
      PYTHONPATH=src .venv/bin/python scripts/build_demos.py --check     (verify numbers only)
"""

# Plotly's figure/layout API is idiomatically written with dict(...) calls.
# ruff: noqa: C408
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import typer

RESULTS = Path("results")
DEMOS = Path("demos")

# --------------------------------------------------------------------------- #
# Design system
# --------------------------------------------------------------------------- #
FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"
INK = "#1f2433"          # primary text
MUTED = "#737a8c"        # secondary text
GRID = "#eef1f5"         # faint gridlines
PAPER = "#ffffff"
PANEL = "#fbfcfe"        # plot area

# Okabe-Ito colourblind-safe categorical palette
BLUE = "#2a78d6"
SKY = "#56B4E9"
GREEN = "#1baf7a"
ORANGE = "#E69F00"
VERM = "#eb6834"
PURPLE = "#4a3aa7"
GOLD = "#C99700"         # the "illusory ceiling" (looks valuable, isn't)
SLATE = "#9aa6b2"

# Landis-Koch agreement thresholds (for the kappa axis)
_LK = [(0.0, 0.2, "slight"), (0.2, 0.4, "fair"), (0.4, 0.6, "moderate"),
       (0.6, 0.8, "substantial"), (0.8, 1.0, "almost perfect")]


def _kappa_band_shapes(*, x0: float = 0.0, x1: float = 1.0) -> list[dict]:
    """Faint alternating bands + threshold lines for a [0,1] kappa axis (y)."""
    shapes = []
    for i, (lo, hi, _) in enumerate(_LK):
        shapes.append(dict(type="rect", xref="paper", yref="y", x0=x0, x1=x1, y0=lo, y1=hi,
                           fillcolor="#0072B2" if i % 2 else "#56B4E9",
                           opacity=0.05, line_width=0, layer="below"))
    for lo, _hi, _ in _LK[1:]:
        shapes.append(dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=lo, y1=lo,
                           line=dict(color="#d4dae3", width=1, dash="dot"), layer="below"))
    return shapes


def _kappa_band_labels() -> list[dict]:
    return [dict(xref="paper", yref="y", x=1.005, xanchor="left", y=(lo + hi) / 2,
                 text=name, showarrow=False, font=dict(size=10.5, color=MUTED))
            for lo, hi, name in _LK]


def _source(text: str) -> dict:
    return dict(xref="paper", yref="paper", x=0, xanchor="left", y=-0.14, yanchor="top",
                text=f"<span style='color:{MUTED};font-size:11px'>{text}</span>", showarrow=False)


def _theme(fig: go.Figure, *, title: str, subtitle: str = "", height: int = 600,
           margin_t: int = 120) -> None:
    head = f"<b>{title}</b>"
    if subtitle:
        head += (f"<br><span style='font-size:13.5px;color:{MUTED};"
                 f"font-weight:400'>{subtitle}</span>")
    fig.update_layout(
        template="plotly_white", font=dict(family=FONT, size=14, color=INK),
        paper_bgcolor=PAPER, plot_bgcolor=PANEL,
        title=dict(text=head, x=0.5, xanchor="center", y=0.96, yanchor="top",
                   font=dict(size=21, color=INK)),
        margin=dict(l=95, r=135, t=margin_t, b=95), height=height,
        hoverlabel=dict(font=dict(family=FONT, size=13), bgcolor="white"),
    )


# --------------------------------------------------------------------------- #
# Data extraction (mirrors the fig_4_5 / fig_4_2 helpers in make_figures.py)
# --------------------------------------------------------------------------- #
def _bf_grid(df: pd.DataFrame) -> float:
    df = df.dropna(subset=["kappa"])
    return float(df.groupby(["feature_family", "classifier"])["kappa"].mean().max())


def _bf_newdata(ds: str, mt: str) -> float:
    d = pd.read_csv(RESULTS / "fixed_baseline_newdata.csv")
    return _bf_grid(d[(d.dataset == ds) & (d.montage == mt)])


def _bf_cl(ds: str) -> float:
    d = pd.read_csv(RESULTS / "fixed_baseline_cl.csv").dropna(subset=["kappa"])
    return float(d[d.dataset == ds].groupby("baseline")["kappa"].mean().max())


def _loso_best_fixed(fname: str) -> float:
    d = pd.read_csv(RESULTS / fname)
    d = d[d.method.isin(["fbcsp", "bandpower"])].dropna(subset=["kappa"])
    return float(d.groupby("method")["kappa"].mean().max())


def _crosssession_best_fixed(montage: str = "full") -> float:
    cs = pd.read_csv(RESULTS / "crosssession_cogbci.csv")
    return _bf_grid(cs[(cs.montage == montage) & (cs.method == "fixed")])


def _master(ds: str, prot: str) -> float:
    """Best-fixed kappa for one (dataset, protocol) cell, read straight from the master table
    (``results_master.csv``) so the demo matches the scoreboard exactly (max over the B1--B5
    battery, including the SVM/RF heads that ``fixed_baseline_cl.csv`` lacks)."""
    m = pd.read_csv(RESULTS / "results_master.csv")
    return float(m[(m.ds == ds) & (m.prot == prot)]["fixed"].iloc[0])


def leakage_data() -> list[dict]:
    # Best-fixed (max over B1--B5), sourced to match the master scoreboard (tab:master). COG-BCI
    # within-CV is not a row in the master table, so it comes from the new-data battery (0.99).
    return [
        {"name": "COG-BCI", "within": _bf_newdata("COGBCI", "full"),
         "clean": _master("COG-BCI n-back", "cross-session"), "protocol": "cross-session"},
        {"name": "STEW", "within": _master("STEW", "within-CV"),
         "clean": _master("STEW", "LOSO"), "protocol": "LOSO"},
        {"name": "WAUC", "within": _master("WAUC", "within-CV"),
         "clean": _master("WAUC", "LOSO"), "protocol": "LOSO"},
    ]


def best_arm_data() -> list[dict]:
    ba = pd.read_csv(RESULTS / "best_arm_diagnostic.csv")
    ccb = pd.read_csv(RESULTS / "ccb_newdata.csv")
    cells = []
    # Leakage-clean existing cells first (fixed/CCB traced to the Ch4 tables, matching fig_4_2);
    # the best single arm is the mean over subjects/seeds in best_arm_existing.csv.
    be_path = RESULTS / "best_arm_existing.csv"
    if be_path.exists():
        be = pd.read_csv(be_path)
        for label, key, bf, cc in [("BCI-IV-2b · 3-ch (clean)", "BCI-IV-2b", 0.292, 0.184),
                                    ("WAUC · 8-ch (clean)", "WAUC", 0.785, 0.426)]:
            g = be[be.dataset == key].dropna(subset=["best_arm_kappa"])
            if len(g):
                arm = float(g["best_arm_kappa"].mean())
                cells.append({"label": label, "fixed": bf, "arm": arm, "ccb": cc,
                              "arm_bank_gap": bf - arm, "selection_gap": arm - cc})
    for ds, mt, label in [("COGBCI", "full", "COG-BCI · full (62-ch)"),
                          ("COGBCI", "nearear", "COG-BCI · near-ear (T7/T8)"),
                          ("UAB", "full", "UAB · full (14-ch)"),
                          ("UAB", "nearear", "UAB · near-ear (T7/T8)")]:
        bf = _bf_newdata(ds, mt)
        arm = float(ba[(ba.dataset == ds) & (ba.montage == mt)]["best_arm_kappa"].dropna().mean())
        cc = float(ccb[(ccb.dataset == ds) & (ccb.montage == mt)]["kappa"].dropna().mean())
        cells.append({"label": label, "fixed": bf, "arm": arm, "ccb": cc,
                      "arm_bank_gap": bf - arm, "selection_gap": arm - cc})
    return cells


# --------------------------------------------------------------------------- #
# Synthetic 2b stream (self-contained — no external data for the OPLB animation)
# --------------------------------------------------------------------------- #
def _synthetic_2b(*, n_per_class_per_session: int = 80, n_sessions: int = 2,
                  n_samples: int = 1001, seed: int = 42):
    from thesis.data.load import SubjectData
    rng = np.random.default_rng(seed)
    sfreq = 250.0
    t = np.arange(n_samples) / sfreq
    Xs, ys, sess = [], [], []
    for s in range(n_sessions):
        X0 = rng.standard_normal((n_per_class_per_session, 3, n_samples)) * 1e-6
        X0[:, 0] += 5e-6 * np.sin(2 * np.pi * 12 * t)
        X1 = rng.standard_normal((n_per_class_per_session, 3, n_samples)) * 1e-6
        X1[:, 2] += 5e-6 * np.sin(2 * np.pi * 12 * t)
        Xs.append(np.concatenate([X0, X1], axis=0))
        ys.append(np.array(["left_hand"] * n_per_class_per_session
                           + ["right_hand"] * n_per_class_per_session))
        sess.extend([str(s)] * (2 * n_per_class_per_session))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    meta = pd.DataFrame({"subject": [1] * len(y), "session": sess, "run": ["0"] * len(y)})
    return SubjectData(subject=1, dataset_name="synthetic-2b", X=X, y=y,
                       metadata=meta, sfreq=sfreq)


def oplb_trajectory(seed: int = 42) -> dict:
    # Prefer a real held-out-subject trace if one has been logged (scripts/run_oplb_trace.py);
    # otherwise fall back to the self-contained synthetic 2-class run below.
    cached = DEMOS / "oplb_trajectory.json"
    if cached.exists():
        d = json.loads(cached.read_text())
        if d.get("real"):
            return d

    import mne

    from thesis.ccb.arms import enumerate_arms_2b
    from thesis.ccb.oplb import OPLBConfig
    from thesis.ccb.runner import run_ccb_on_split
    from thesis.protocols import session_split

    mne.set_log_level("ERROR")
    data = _synthetic_2b(seed=seed)
    split = session_split(data, train_session_idx=0, test_session_idx=1)
    arms = enumerate_arms_2b(data.sfreq)
    res = run_ccb_on_split(data, split, arms=arms, config=OPLBConfig(alpha=0.5), seed=seed)
    return {
        "arm_pulls": [int(a) for a in res.arm_pulls],
        "cumulative_regret": [float(r) for r in res.cumulative_regret],
        "n_arms": int(res.n_arms_surviving), "kappa": float(res.kappa),
        "accuracy": float(res.accuracy), "n_test": int(res.n_test),
    }


# --------------------------------------------------------------------------- #
# Demo 1 — leakage collapse (the centrepiece)
# --------------------------------------------------------------------------- #
def build_leakage_collapse() -> go.Figure:
    rows = leakage_data()
    names = [r["name"] for r in rows]
    within = [r["within"] for r in rows]
    clean = [r["clean"] for r in rows]

    def bars(state: str) -> go.Bar:
        if state == "within":
            y = within
            return go.Bar(
                x=names, y=y, width=0.56, cliponaxis=False,
                marker=dict(color="rgba(201,151,0,0.18)", line=dict(color=GOLD, width=2.2),
                            pattern=dict(shape="/", fgcolor=GOLD, size=7, solidity=0.28)),
                text=[f"<b>{v:.2f}</b>" for v in y], textposition="outside",
                textfont=dict(size=15, color=GOLD),
                hovertemplate="%{x} · within-CV<br>κ = %{y:.3f}<extra></extra>")
        y = clean
        # A near-chance (slightly negative) kappa would render as an invisible sub-axis bar that
        # "disappears"; floor the plotted height at the baseline and keep the true value in the hover.
        y_plot = [max(v, 0.0) for v in y]
        return go.Bar(
            x=names, y=y_plot, width=0.56, cliponaxis=False,
            marker=dict(color=BLUE, line=dict(color="#004c78", width=1)),
            text=[f"<b>{max(v, 0.0):.2f}</b><br><span style='font-size:11px;color:{MUTED}'>{r['protocol']}</span>"
                  for v, r in zip(y, rows, strict=True)], textposition="outside",
            textfont=dict(size=15, color=BLUE), customdata=y,
            hovertemplate="%{x} · leakage-clean<br>κ = %{customdata:.3f}<extra></extra>")

    def verdict(state: str) -> dict:
        if state == "within":
            txt = (f"<b style='color:{GOLD}'>Within-subject CV</b> — a leakage-confounded ceiling: "
                   "the classifier scores by <i>recording identity</i>, not workload.")
        else:
            txt = (f"<b style='color:{BLUE}'>Leakage-clean protocol</b> — hold out the whole "
                   "recording (session → cross-session, subject → LOSO) and the ceiling collapses "
                   "toward chance.")
        return dict(xref="paper", yref="paper", x=0.5, xanchor="center", y=1.04, yanchor="bottom",
                    text=txt, showarrow=False, font=dict(size=13))

    src = _source("Best fixed-pipeline κ. Source: results/{fixed_baseline_*, crosssession_cogbci, "
                  "loso_stew, loso_wauc}.csv · mirrors thesis Fig. 4.5")
    fig = go.Figure(
        data=[bars("within")],
        frames=[go.Frame(name="within", data=[bars("within")],
                         layout=go.Layout(annotations=_kappa_band_labels() + [verdict("within"), src])),
                go.Frame(name="clean", data=[bars("clean")],
                         layout=go.Layout(annotations=_kappa_band_labels() + [verdict("clean"), src]))],
    )
    _theme(fig, title="The leakage collapse", height=620, margin_t=150)
    fig.update_layout(
        shapes=_kappa_band_shapes(),
        annotations=_kappa_band_labels() + [verdict("within"), src],
        xaxis=dict(showgrid=False, tickfont=dict(size=15, color=INK), ticklen=0),
        yaxis=dict(title="Cohen's κ", range=[-0.05, 1.08], showgrid=False, zeroline=True,
                   zerolinecolor="#c2c8d2", tickformat=".1f"),
        bargap=0.45,
        updatemenus=[dict(
            type="buttons", direction="right", x=0.5, xanchor="center", y=1.12, yanchor="bottom",
            pad=dict(b=4), showactive=True, bgcolor="#f1f4f8", bordercolor="#d4dae3",
            font=dict(size=12.5),
            buttons=[
                dict(label="  Within-CV  ", method="animate",
                     args=[["within"], {"frame": {"duration": 650, "redraw": True},
                                        "transition": {"duration": 550, "easing": "cubic-in-out"}}]),
                dict(label="  Leakage-clean  ", method="animate",
                     args=[["clean"], {"frame": {"duration": 650, "redraw": True},
                                       "transition": {"duration": 550, "easing": "cubic-in-out"}}]),
            ])],
    )
    return fig


# --------------------------------------------------------------------------- #
# Demo 2 — best-arm decomposition
# --------------------------------------------------------------------------- #
def build_best_arm() -> go.Figure:
    cells = best_arm_data()
    cats = ["best fixed<br>pipeline", "best single arm<br>(frozen)", "CCB"]
    colors = [BLUE, GREEN, VERM]

    def trace(c: dict) -> go.Bar:
        y = [c["fixed"], c["arm"], c["ccb"]]
        return go.Bar(x=cats, y=y, width=0.6, marker=dict(color=colors, line=dict(width=0)),
                      text=[f"<b>{v:.3f}</b>" for v in y], textposition="outside",
                      textfont=dict(size=15, color=INK),
                      hovertemplate="%{x}<br>κ = %{y:.3f}<extra></extra>")

    def gap_anns(c: dict) -> list[dict]:
        f, a, cc = c["fixed"], c["arm"], c["ccb"]
        sel = c["selection_gap"]
        sel_col = GREEN if sel < 0 else MUTED
        sel_txt = (f"selection gap {sel:+.3f}"
                   + ("<br>✓ bandit beats its static arm" if sel < 0 else ""))
        return [
            # arm-bank gap bracket (fixed -> arm)
            dict(x=0.5, y=max(f, a) + 0.06, xref="x", yref="y", xanchor="center", showarrow=False,
                 text=f"<b style='color:{VERM}'>arm-bank gap {c['arm_bank_gap']:+.3f}</b>",
                 font=dict(size=12.5)),
            dict(x=0, y=f, ax=1, ay=a, xref="x", yref="y", axref="x", ayref="y",
                 showarrow=True, arrowhead=0, arrowwidth=1.4, arrowcolor=VERM, opacity=0.6),
            # selection gap (arm -> CCB)
            dict(x=1.5, y=max(a, cc) + 0.06, xref="x", yref="y", xanchor="center", showarrow=False,
                 text=f"<b style='color:{sel_col}'>{sel_txt}</b>", font=dict(size=12.5)),
            dict(x=1, y=a, ax=2, ay=cc, xref="x", yref="y", axref="x", ayref="y",
                 showarrow=True, arrowhead=0, arrowwidth=1.4, arrowcolor=sel_col, opacity=0.6),
        ]

    src = _source("κ per cell. Source: results/{best_arm_diagnostic, ccb_newdata, "
                  "fixed_baseline_newdata}.csv · mirrors thesis Fig. 4.2")
    fig = go.Figure(
        data=[trace(cells[0])],
        frames=[go.Frame(name=c["label"], data=[trace(c)],
                         layout=go.Layout(annotations=_kappa_band_labels() + gap_anns(c) + [src]))
                for c in cells],
    )
    _theme(fig, title="Where does the CCB-vs-fixed gap live?",
           subtitle="The arm-bank gap dominates; the selection gap is small (and sometimes the bandit wins)",
           height=600)
    fig.update_layout(
        shapes=_kappa_band_shapes(),
        annotations=_kappa_band_labels() + gap_anns(cells[0]) + [src],
        xaxis=dict(showgrid=False, tickfont=dict(size=13.5, color=INK), ticklen=0),
        yaxis=dict(title="Cohen's κ", range=[0, 1.08], showgrid=False, zeroline=True,
                   zerolinecolor="#c2c8d2", tickformat=".1f"),
        bargap=0.4,
        updatemenus=[dict(
            type="dropdown", x=0.0, xanchor="left", y=1.15, yanchor="bottom", showactive=True,
            bgcolor="#f1f4f8", bordercolor="#d4dae3", font=dict(size=12.5),
            buttons=[dict(label=f"  {c['label'].replace('<br>', ' ')}  ", method="animate",
                          args=[[c["label"]], {"frame": {"duration": 400, "redraw": True},
                                               "transition": {"duration": 350}}]) for c in cells])],
    )
    return fig


# --------------------------------------------------------------------------- #
# Demo 3 — OPLB stream animation
# --------------------------------------------------------------------------- #
def build_oplb_stream(traj: dict) -> go.Figure:
    pulls = traj["arm_pulls"]
    regret = traj["cumulative_regret"]
    n = len(pulls)
    rounds = list(range(1, n + 1))
    inc = np.diff([0.0] + regret)               # per-round regret increment (0 = matched the oracle)
    mcol = [GREEN if d <= 0 else VERM for d in inc]   # green round = no regret paid

    def frame_data(t: int) -> list:
        return [
            go.Scatter(x=rounds[:t], y=pulls[:t], mode="markers",
                       marker=dict(size=8, color=mcol[:t], line=dict(width=0.5, color="white")),
                       xaxis="x", yaxis="y",
                       hovertemplate="round %{x}<br>arm %{y}<extra></extra>"),
            go.Scatter(x=rounds[:t], y=regret[:t], mode="lines",
                       line=dict(color=BLUE, width=2.6, shape="spline"),
                       fill="tozeroy", fillcolor="rgba(0,114,178,0.10)",
                       xaxis="x2", yaxis="y2",
                       hovertemplate="round %{x}<br>cumulative regret %{y:.0f}<extra></extra>"),
        ]

    fig = go.Figure(
        data=frame_data(1),
        frames=[go.Frame(name=str(t), data=frame_data(t)) for t in range(1, n + 1)],
    )
    _theme(fig, title="The OPLB bandit, live",
           subtitle=f"One stream of {n} rounds ({traj.get('cell', 'synthetic 2-class')}) — arms "
                    f"explored, regret paid, then the frozen-test verdict",
           height=640)
    fig.update_layout(
        margin=dict(l=95, r=60, t=120, b=80),
        xaxis=dict(domain=[0, 1], anchor="y", title="", showticklabels=False, showgrid=False,
                   range=[0, n + 1]),
        yaxis=dict(domain=[0.56, 1.0], title="selected arm (ID)",
                   range=[min(pulls) - 5, max(pulls) + 6], showgrid=True, gridcolor=GRID),
        xaxis2=dict(domain=[0, 1], anchor="y2", title="round", showgrid=False, range=[0, n + 1]),
        yaxis2=dict(domain=[0.0, 0.40], title="cumulative regret",
                    range=[0, max(max(regret), 1.0) * 1.12], showgrid=True, gridcolor=GRID),
        showlegend=False,
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, xanchor="left", y=1.13, yanchor="bottom",
            bgcolor="#f1f4f8", bordercolor="#d4dae3", font=dict(size=12.5), pad=dict(b=4),
            buttons=[
                dict(label="▶  Play", method="animate",
                     args=[None, {"frame": {"duration": 45, "redraw": True}, "fromcurrent": True}]),
                dict(label="❚❚  Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])])],
        sliders=[dict(active=0, x=0.18, len=0.82, y=1.135, yanchor="bottom",
                      currentvalue=dict(prefix="round ", font=dict(size=12, color=MUTED)),
                      pad=dict(b=2), ticklen=0, font=dict(size=9),
                      steps=[dict(method="animate", label="",
                                  args=[[str(t)], {"frame": {"duration": 0, "redraw": True},
                                                   "mode": "immediate"}]) for t in range(1, n + 1)])],
    )
    # legend chips for the marker colours
    for col, lab, xpos in [(GREEN, "round matched the oracle", 0.62), (VERM, "round paid regret", 0.86)]:
        fig.add_annotation(xref="paper", yref="paper", x=xpos, y=1.135, yanchor="bottom",
                           xanchor="left", showarrow=False, font=dict(size=11, color=col),
                           text=f"● {lab}")
    # top-left of the arm panel: the only corner with no arm-pull points (high IDs
    # only ever appear in the right half, rounds >= 56), so the badge never overlaps data
    fig.add_annotation(xref="paper", yref="paper", x=0.015, y=0.985, xanchor="left", yanchor="top",
                       showarrow=False, align="left", borderpad=8, bgcolor="rgba(0,158,115,0.12)",
                       bordercolor=GREEN, borderwidth=1,
                       text=f"<b style='color:{GREEN}'>frozen-test κ = {traj['kappa']:.3f}</b>"
                            f"<br><span style='font-size:11px;color:{MUTED}'>"
                            f"{'real held-out subject' if traj.get('real') else 'synthetic'} · "
                            f"acc {traj['accuracy']:.0%} · "
                            f"{traj['n_arms']} arms survived · n={traj['n_test']}</span>")
    return fig


# --------------------------------------------------------------------------- #
def _write(fig: go.Figure, name: str) -> None:
    DEMOS.mkdir(exist_ok=True)
    out = DEMOS / f"{name}.html"
    pio.write_html(fig, file=str(out), include_plotlyjs=True, full_html=True, auto_play=False,
                   config=dict(displayModeBar=False, responsive=True))
    print(f"  wrote {out}  ({out.stat().st_size // 1024} KB)")


def _check() -> None:
    # Leakage demo now uses best-fixed (max over B1--B5), scoreboard-consistent (tab:master).
    lk = {r["name"]: (round(r["within"], 2), round(abs(r["clean"]), 2)) for r in leakage_data()}
    exp = {"COG-BCI": (0.99, 0.08), "STEW": (0.95, 0.31), "WAUC": (0.78, 0.00)}
    for k, v in exp.items():
        assert lk[k] == v, f"leakage {k}: got {lk[k]}, expected {v}"
    ba = {c["label"]: c for c in best_arm_data()}
    assert round(ba["COG-BCI · full (62-ch)"]["fixed"], 2) == 0.99, ba["COG-BCI · full (62-ch)"]
    assert ba["COG-BCI · near-ear (T7/T8)"]["selection_gap"] < 0, "COG-BCI near-ear selection gap < 0"
    b2b = ba["BCI-IV-2b · 3-ch (clean)"]  # the rhetorically strongest cell (bandit beats its arm)
    assert abs(b2b["arm_bank_gap"] - 0.170) < 0.006 and b2b["selection_gap"] < -0.03, b2b
    print("  demo numbers match the thesis figures/tables ✓")


def main(check: bool = typer.Option(False, help="Only verify numbers; do not write HTML."),
         only: str = typer.Option("", help="Comma list of demos to (re)build: leakage,best_arm,oplb.")) -> None:
    _check()
    if check:
        return
    want = {s.strip() for s in only.split(",") if s.strip()} or {"leakage", "best_arm", "oplb"}
    print("Building demos ...")
    if "leakage" in want:
        _write(build_leakage_collapse(), "leakage_collapse")
    if "best_arm" in want:
        _write(build_best_arm(), "best_arm")
    if "oplb" in want:
        traj = oplb_trajectory(seed=42)
        if not traj.get("real"):  # never overwrite a real logged trace with the synthetic one
            (DEMOS / "oplb_trajectory.json").write_text(json.dumps(traj, indent=2))
        _write(build_oplb_stream(traj), "oplb_stream")
    print("Done. Open the demos/*.html files in any browser (offline).")


if __name__ == "__main__":
    typer.run(main)
