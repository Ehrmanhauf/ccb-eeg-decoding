r"""Render the three presentation demos as self-contained MP4 videos.

These are the *same* visualisations as ``scripts/build_demos.py``'s HTML demos, over the
*same* committed data (``results/*.csv``, plus the optional ``demos/oplb_trajectory.json``
cache written by ``scripts/run_oplb_trace.py``), re-rendered with matplotlib + ffmpeg so they
can be embedded directly in a slide deck and played *in-slide* (no browser, no internet, no
broken external link). Output lands in the generated, gitignored ``demos/`` directory.

Every number comes from ``build_demos.py``'s data helpers (``leakage_data``, ``best_arm_data``,
``oplb_trajectory``), so the videos match the HTML demos and the reported figures/tables exactly
(``build_demos --check`` guards those numbers).

Output (all under ``demos/``):
  leakage_collapse.mp4  + _poster.png   -- within-CV ceiling collapses to the leakage-clean truth
  best_arm.mp4          + _poster.png   -- cycle the cells; arm-bank gap dominates the selection gap
  oplb_stream.mp4       + _poster.png   -- one real OPLB run streamed: arm pulls + cumulative regret

Run:  PYTHONPATH=src .venv/bin/python scripts/make_demo_videos.py
      PYTHONPATH=src .venv/bin/python scripts/make_demo_videos.py --only leakage,oplb
"""
# matplotlib's animation API is stateful; a clear-and-redraw per frame is used for robustness.
# matplotlib's bbox/arrowprops kwargs are idiomatically written with dict(...) calls.
# ruff: noqa: C408
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as manim  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import typer  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

# reuse the exact data + palette from the HTML demo builder (scripts/ is on sys.path[0])
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_demos import (  # noqa: E402
    BLUE,
    GOLD,
    GREEN,
    GRID,
    INK,
    MUTED,
    VERM,
    best_arm_data,
    leakage_data,
    oplb_trajectory,
)

DEMOS = Path("demos")
FIGSIZE = (12.8, 7.2)  # 16:9
DPI = 150              # -> 1920x1080

# Landis-Koch agreement bands for the kappa axis (readability, matches the HTML demos)
_LK = [(0.0, 0.2, "slight"), (0.2, 0.4, "fair"), (0.4, 0.6, "moderate"),
       (0.6, 0.8, "substantial"), (0.8, 1.0, "almost\nperfect")]


def _ease(t: float) -> float:
    """Smoothstep easing in [0,1]."""
    t = min(max(t, 0.0), 1.0)
    return 3 * t**2 - 2 * t**3


def _blend(c1, c2, t: float):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple((1 - t) * a + t * b)


def _kappa_bands(ax, x1: float) -> None:
    for i, (lo, hi, name) in enumerate(_LK):
        ax.axhspan(lo, hi, color="#0072B2" if i % 2 else "#56B4E9", alpha=0.05, zorder=0)
        if lo > 0:
            ax.axhline(lo, color="#d4dae3", ls=":", lw=1, zorder=0)
        ax.text(x1 + 0.04, (lo + hi) / 2, name, color=MUTED, fontsize=10, va="center", ha="left")


def _titles(fig, title: str, subtitle: str, subtitle_color: str, source: str) -> None:
    fig.text(0.5, 0.945, title, ha="center", va="top", fontsize=23, fontweight="bold", color=INK)
    fig.text(0.5, 0.88, subtitle, ha="center", va="top", fontsize=13, color=subtitle_color)
    fig.text(0.09, 0.03, source, ha="left", va="bottom", fontsize=9.5, color=MUTED)


def _save(anim: manim.FuncAnimation, fig, frame_fn, out: Path, fps: int) -> None:
    DEMOS.mkdir(exist_ok=True)
    frame_fn(0)  # draw the first frame so it can double as the PowerPoint poster
    poster = out.with_name(out.stem + "_poster.png")
    fig.savefig(poster, dpi=DPI)
    writer = manim.FFMpegWriter(
        fps=fps, codec="libx264",
        extra_args=["-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    anim.save(str(out), writer=writer, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out}  ({out.stat().st_size // 1024} KB)  + {poster.name}")


# --------------------------------------------------------------------------- #
# Demo 1 — the leakage collapse (within-CV gold ceiling -> leakage-clean blue truth)
# --------------------------------------------------------------------------- #
def render_leakage(out: Path) -> None:
    rows = leakage_data()
    names = [r["name"] for r in rows]
    within = np.array([r["within"] for r in rows])
    clean_true = np.array([r["clean"] for r in rows])
    clean = np.maximum(clean_true, 0.0)          # floor near-chance bars at the axis (as the HTML does)
    prot = [r["protocol"] for r in rows]
    x = np.arange(len(names))

    F_HOLD1, F_TRANS, F_HOLD2 = 40, 45, 70
    total = F_HOLD1 + F_TRANS + F_HOLD2

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.subplots_adjust(left=0.09, right=0.86, top=0.80, bottom=0.12)

    def frame(f: int):
        if f < F_HOLD1:
            t = 0.0
        elif f < F_HOLD1 + F_TRANS:
            t = _ease((f - F_HOLD1) / F_TRANS)
        else:
            t = 1.0
        ax.clear()
        _kappa_bands(ax, x1=len(names) - 0.5)
        h = within * (1 - t) + clean * t
        face = _blend(GOLD, BLUE, t)
        edge = _blend(GOLD, "#004c78", t)
        facealpha = 0.18 * (1 - t) + 1.0 * t
        bars = ax.bar(x, h, width=0.56, zorder=3)
        for b in bars:
            b.set_facecolor((*face, facealpha))
            b.set_edgecolor(edge)
            b.set_linewidth(2.2 if t < 0.5 else 1.0)
            if t < 0.4:
                b.set_hatch("////")
        lab_col = GOLD if t < 0.5 else BLUE
        for k in range(len(names)):
            val = within[k] * (1 - t) + clean_true[k] * t
            ax.text(x[k], h[k] + 0.02, f"{max(val, 0.0):.2f}", ha="center", va="bottom",
                    fontsize=16, fontweight="bold", color=lab_col)
            if t > 0.55:
                ax.text(x[k], h[k] + 0.085, prot[k], ha="center", va="bottom",
                        fontsize=10.5, color=MUTED, alpha=(t - 0.55) / 0.45)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=15, color=INK)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylim(-0.02, 1.12)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylabel("Cohen's κ", fontsize=14)
        ax.tick_params(length=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if t < 0.5:
            sub = ("Within-subject CV — a leakage-confounded ceiling: "
                   "the classifier scores by recording identity, not workload")
            scol = GOLD
        else:
            sub = ("Leakage-clean — hold out the whole recording (session→cross-session, "
                   "subject→LOSO) and the ceiling collapses toward chance")
            scol = BLUE
        # subtitle is redrawn each frame so its colour can switch; title/source are static
        for txt in list(fig.texts):
            if getattr(txt, "_role", None) == "sub":
                txt.remove()
        st = fig.text(0.5, 0.875, sub, ha="center", va="top", fontsize=12.5, color=scol)
        st._role = "sub"
        return ()

    fig.text(0.5, 0.955, "The leakage collapse", ha="center", va="top",
             fontsize=24, fontweight="bold", color=INK)
    fig.text(0.09, 0.03, "Best fixed-pipeline κ · source results/*.csv · mirrors thesis Fig. 4.5",
             ha="left", va="bottom", fontsize=9.5, color=MUTED)
    anim = manim.FuncAnimation(fig, frame, frames=total, interval=40, blit=False)
    _save(anim, fig, frame, out, fps=25)


# --------------------------------------------------------------------------- #
# Demo 2 — best-arm decomposition (cycle the cells)
# --------------------------------------------------------------------------- #
def render_best_arm(out: Path) -> None:
    cells = best_arm_data()
    cats = ["best fixed\npipeline", "best single arm\n(frozen)", "CCB"]
    colors = [BLUE, GREEN, VERM]
    HOLD, TRANS = 34, 15
    seg = HOLD + TRANS
    total = len(cells) * seg

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.subplots_adjust(left=0.09, right=0.86, top=0.76, bottom=0.14)

    def vals(c):
        return np.array([c["fixed"], c["arm"], c["ccb"]])

    def frame(f: int):
        seg_idx = min(f // seg, len(cells) - 1)
        within = f - seg_idx * seg
        c0 = cells[seg_idx]
        if within < HOLD or seg_idx == len(cells) - 1:
            c1, tt = c0, 0.0
        else:
            c1, tt = cells[seg_idx + 1], _ease((within - HOLD) / TRANS)
        v = vals(c0) * (1 - tt) + vals(c1) * tt
        cc = c0 if tt < 0.5 else c1
        ax.clear()
        _kappa_bands(ax, x1=2.5)
        ax.bar(np.arange(3), v, width=0.6, color=colors, zorder=3)
        for k in range(3):
            ax.text(k, v[k] + 0.02, f"{v[k]:.3f}", ha="center", va="bottom",
                    fontsize=16, fontweight="bold", color=INK)
        f_, a_, g_ = cc["fixed"], cc["arm"], cc["ccb"]
        sel = cc["selection_gap"]
        sel_col = GREEN if sel < 0 else MUTED
        # connecting arrows link the bar tops; the gap values live in a fixed upper-centre
        # callout (the region above the arm/CCB bars is empty in every cell, so no collision)
        ax.annotate("", xy=(1, a_), xytext=(0, f_),
                    arrowprops=dict(arrowstyle="-", color=VERM, lw=1.5, alpha=0.55))
        ax.annotate("", xy=(2, g_), xytext=(1, a_),
                    arrowprops=dict(arrowstyle="-", color=sel_col, lw=1.5, alpha=0.55))
        ax.text(0.60, 0.965, f"arm-bank gap {cc['arm_bank_gap']:+.3f}", transform=ax.transAxes,
                ha="center", va="top", color=VERM, fontsize=14.5, fontweight="bold")
        sel_txt = f"selection gap {sel:+.3f}" + ("  ✓ bandit beats its arm" if sel < 0 else "")
        ax.text(0.60, 0.905, sel_txt, transform=ax.transAxes,
                ha="center", va="top", color=sel_col, fontsize=14.5, fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_xticklabels(cats, fontsize=13.5, color=INK)
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Cohen's κ", fontsize=14)
        ax.tick_params(length=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for txt in list(fig.texts):
            if getattr(txt, "_role", None) == "cell":
                txt.remove()
        ct = fig.text(0.5, 0.80, cc["label"].replace("<br>", " "), ha="center", va="top",
                      fontsize=15, fontweight="bold", color=INK)
        ct._role = "cell"
        return ()

    fig.text(0.5, 0.955, "Where does the CCB-vs-fixed gap live?", ha="center", va="top",
             fontsize=24, fontweight="bold", color=INK)
    fig.text(0.5, 0.895, "The arm-bank gap dominates; the selection gap is small "
             "(and sometimes the bandit wins)", ha="center", va="top", fontsize=13, color=MUTED)
    fig.text(0.09, 0.03, "κ per cell · source results/{best_arm_diagnostic, ccb_newdata, "
             "fixed_baseline_newdata}.csv · mirrors thesis Fig. 4.2",
             ha="left", va="bottom", fontsize=9.5, color=MUTED)
    anim = manim.FuncAnimation(fig, frame, frames=total, interval=40, blit=False)
    _save(anim, fig, frame, out, fps=25)


# --------------------------------------------------------------------------- #
# Demo 3 — OPLB stream (one real held-out run: arm pulls + cumulative regret)
# --------------------------------------------------------------------------- #
def render_oplb(out: Path) -> None:
    traj = oplb_trajectory(seed=42)
    pulls = np.array(traj["arm_pulls"])
    regret = np.array(traj["cumulative_regret"])
    n = len(pulls)
    rounds = np.arange(1, n + 1)
    inc = np.diff(np.concatenate([[0.0], regret]))
    mcol = np.where(inc <= 0, GREEN, VERM)
    kind = "real held-out subject" if traj.get("real") else "synthetic 2-class"

    step = max(1, int(np.ceil(n / 140)))
    stream = list(range(step, n + 1, step))
    if stream[-1] != n:
        stream.append(n)
    HOLD = 24
    frames_list = stream + [n] * HOLD

    fig, (axT, axB) = plt.subplots(2, 1, figsize=FIGSIZE, dpi=DPI, height_ratios=[1.25, 1])
    fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.10, hspace=0.30)

    def frame(i: int):
        t = frames_list[i]
        is_hold = i >= len(stream)
        axT.clear()
        axB.clear()
        axT.scatter(rounds[:t], pulls[:t], s=26, c=mcol[:t],
                    edgecolors="white", linewidths=0.4, zorder=3)
        axT.set_xlim(0, n + 1)
        axT.set_ylim(pulls.min() - 5, pulls.max() + 6)
        axT.set_ylabel("selected arm (ID)", fontsize=13)
        axT.set_xticklabels([])
        axT.grid(True, color=GRID)
        for s in ("top", "right"):
            axT.spines[s].set_visible(False)
        # legend chips
        axT.scatter([], [], s=40, c=GREEN, label="round matched the oracle")
        axT.scatter([], [], s=40, c=VERM, label="round paid regret")
        axT.legend(loc="upper left", fontsize=10.5, frameon=False, ncol=2,
                   bbox_to_anchor=(0.0, 1.14))
        axB.plot(rounds[:t], regret[:t], color=BLUE, lw=2.6)
        axB.fill_between(rounds[:t], regret[:t], color=BLUE, alpha=0.10)
        axB.set_xlim(0, n + 1)
        axB.set_ylim(0, max(regret.max(), 1.0) * 1.12)
        axB.set_xlabel("round", fontsize=13)
        axB.set_ylabel("cumulative regret", fontsize=13)
        axB.grid(True, color=GRID)
        for s in ("top", "right"):
            axB.spines[s].set_visible(False)
        if is_hold:
            # solid callout in the sparse top-right (highest arm IDs are rare there)
            axT.text(0.985, 0.965,
                     f"frozen-test κ = {traj['kappa']:.3f}\n"
                     f"acc {traj['accuracy']:.0%} · {traj['n_arms']} arms · n={traj['n_test']}",
                     transform=axT.transAxes, ha="right", va="top", linespacing=1.6,
                     fontsize=14, fontweight="bold", color=GREEN,
                     bbox=dict(boxstyle="round,pad=0.5", fc=_blend("#ffffff", GREEN, 0.12),
                               ec=GREEN, lw=1.5))
        return ()

    fig.text(0.5, 0.955, "The OPLB bandit, live", ha="center", va="top",
             fontsize=24, fontweight="bold", color=INK)
    fig.text(0.5, 0.905,
             f"One real OPLB run over {n} rounds ({kind}) — arms explored, "
             f"regret paid, then the frozen-test verdict",
             ha="center", va="top", fontsize=12, color=MUTED)
    anim = manim.FuncAnimation(fig, frame, frames=len(frames_list), interval=42, blit=False)
    _save(anim, fig, frame, out, fps=24)


# --------------------------------------------------------------------------- #
_RENDERERS = {
    "leakage": (render_leakage, "leakage_collapse.mp4"),
    "best_arm": (render_best_arm, "best_arm.mp4"),
    "oplb": (render_oplb, "oplb_stream.mp4"),
}


def main(only: str = typer.Option("", help="Comma list of demos to render: leakage,best_arm,oplb.")) -> None:
    want = {s.strip() for s in only.split(",") if s.strip()} or set(_RENDERERS)
    print("Rendering demo videos (matplotlib + ffmpeg) ...")
    for key, (fn, name) in _RENDERERS.items():
        if key in want:
            fn(DEMOS / name)
    print("Done. MP4s (and *_poster.png) written under demos/.")


if __name__ == "__main__":
    typer.run(main)
