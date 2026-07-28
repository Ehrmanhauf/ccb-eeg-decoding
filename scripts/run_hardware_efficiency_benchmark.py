r"""Hardware-efficiency benchmark: the full deployment cost profile of each method.

Profiles, on a quiet single-thread CPU, what each of the four method families costs to
*calibrate* and to *run* across the channel-count span of the panel --- from the
two-channel near-ear deployment montage up to a 62-channel research cap:

  - FBCSP + shrinkage LDA          (fixed pipeline; nine sub-bands)
  - band-power + shrinkage LDA     (fixed pipeline)
  - CCB                            context + selected arm's pipeline (inference) AND the
                                   bandit's per-trial select+update (online adaptation)
  - EEGNet                         compact CNN (compute-heavy comparator)

Three distinct costs are reported, because a wearable pays all three:

  * CALIBRATION (all four): wall-clock to fit the model on the training split. This is
    the up-front cost the user sits through before the device is usable, and it is where
    the deep comparator is expensive while the classical and bandit pipelines are cheap.
  * INFERENCE (all four): per-trial latency (median AND tail percentiles), CPU time,
    deployed-parameter footprint, and a real-time factor --- the cost of classifying one
    epoch on the frozen/deployed model.
  * ONLINE ADAPTATION (CCB only): per-trial OPLB ``select`` + ``update`` latency, also
    normalised per arm --- the cost the bandit pays to adapt online that the three STATIC
    methods do not pay at all (they would need a full offline refit). ``online_heads=False``
    is the headline default, so per-trial head ``partial_fit`` is not on this path.

IMPORTANT (thesis caveat): this is a hardware *profile*, NOT a rescue of the negative
decoding result. A cheaper route to a low kappa is not a deployment win. The kappa column
is carried so the efficiency numbers are never read as a decoding contribution --- the CCB
is typically the least accurate classical system here, and its online-adaptation capability
does not translate into an accuracy win (best-arm diagnostic, Chapter 4). Measuring the
capability is rigour; it is not a performance claim.

Methodology notes:
  - Single thread throughout. BLAS threading is pinned via environment variables set
    BEFORE numpy is imported (see the top of this file), so the classical pipelines are
    held to the same one-core budget as torch. Without this pin the classical numbers are
    optimistic against a genuine embedded single-core target.
  - Latency is reported as p50/p95/p99 plus IQR, not just a median: a real-time system is
    governed by its tail, not its typical case.
  - Model footprint is reported twice: ``param_bytes`` sums the actual fitted numeric
    state (the number an embedded engineer needs) while ``model_footprint_kb`` keeps the
    serialised-pickle figure for continuity with earlier runs.
  - Energy is a DERIVED quantity (CPU-time x TDP), not an independent measurement, and is
    therefore reported at two TDPs bracketing the plausible range rather than as a third
    axis. It is perfectly collinear with CPU time by construction.
  - Peak resident memory is deliberately NOT reported: ``ru_maxrss`` is a monotone
    process-level high-water mark, so within a single process the first system fitted
    absorbs all shared allocation and later systems read near zero. A misleading column is
    worse than an absent one; ``param_bytes`` is the honest footprint measure.

Matched conditions: the split comes from ``thesis.matched`` (same fold seed as decoding).
Run on an IDLE machine (efficiency timing is meaningless under CPU contention) --- e.g. as
the final step of ``scripts/rerun_fairness.sh``, after the decoding runs finish.

Output: results/hardware_efficiency.csv
"""

from __future__ import annotations

# Pin every BLAS backend to a single thread BEFORE numpy (and anything importing it) is
# loaded --- these variables are read at library load time, so setting them later is a
# no-op. This is what makes the classical pipelines a fair single-core comparison against
# torch, which is pinned separately via torch.set_num_threads(1).
import os  # noqa: E402

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = "1"

import functools  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402

from thesis.baselines.bandpower_cl import BandPowerCL  # noqa: E402
from thesis.baselines.fbcsp import FBCSP  # noqa: E402
from thesis.ccb.arms import enumerate_arms_2a, enumerate_arms_generic  # noqa: E402
from thesis.ccb.context import N_ARM_FAMILIES, compute_context, compute_context_2a  # noqa: E402
from thesis.ccb.context_cl import compute_context_workload  # noqa: E402
from thesis.ccb.oplb import OPLB, OPLBConfig  # noqa: E402
from thesis.ccb.runner import _build_per_arm_contexts, fit_heads_on_calibration  # noqa: E402
from thesis.data import select_near_ear  # noqa: E402
from thesis.data.cogbci_load import COGBCI_CHANNEL_ROLES, load_cogbci  # noqa: E402
from thesis.data.emotiv_uab_load import UAB_CHANNELS, load_emotiv_uab  # noqa: E402
from thesis.data.load import load_bci2a, load_bci2b_screening  # noqa: E402
from thesis.data.near_ear import NEAR_EAR_ROLES  # noqa: E402
from thesis.data.stew_load import STEW_CHANNEL_ROLES, load_stew  # noqa: E402
from thesis.matched import matched_within_cv  # noqa: E402
from thesis.metrics import compute_metrics  # noqa: E402

# TDP bracket for the derived energy proxy: a low-power embedded/wearable-class core and a
# laptop-class core. Reporting both makes the order-of-magnitude nature explicit instead of
# implying a single measured wattage.
TDP_EMBEDDED_W = 2.0
TDP_LAPTOP_W = 15.0


# --- cell registry: each entry sets up one (dataset, montage) profiling cell ---------------
# context_fn matches what the headline CCB runs per dataset (runner.py dispatch):
# 2a -> compute_context_2a (22-ch), 2b -> compute_context (3-ch), CL -> workload context.
def _cell_2a(subject):
    cell = load_bci2a([subject])[0]
    return "BCI-IV-2a", "MI", cell, enumerate_arms_2a(cell.sfreq), compute_context_2a


def _cell_2b(subject):
    cell = load_bci2b_screening([subject])[0]
    arms = enumerate_arms_generic(n_channels=cell.X.shape[1], n_components=4)
    return "BCI-IV-2b", "MI", cell, arms, compute_context


def _cell_stew(subject):
    cell = load_stew([subject])[0]
    arms = enumerate_arms_generic(n_channels=cell.X.shape[1], n_components=4)
    ctx = functools.partial(compute_context_workload, channel_roles=STEW_CHANNEL_ROLES)
    return "STEW", "CL", cell, arms, ctx


def _cell_uab_nearear(subject):
    full = load_emotiv_uab([subject])[0]
    cell = select_near_ear(full, UAB_CHANNELS)  # T7/T8 by position -> 2 channels
    arms = enumerate_arms_generic(n_channels=cell.X.shape[1], n_components=4)
    ctx = functools.partial(compute_context_workload, channel_roles=NEAR_EAR_ROLES)
    return "UAB near-ear", "CL", cell, arms, ctx


def _cell_cogbci(subject):
    # Highest channel count in the panel (62 ch) -- the upper anchor of the scaling curve.
    cell = load_cogbci([subject], sessions=("S1",))[0]
    arms = enumerate_arms_generic(n_channels=cell.X.shape[1], n_components=4)
    ctx = functools.partial(compute_context_workload, channel_roles=COGBCI_CHANNEL_ROLES)
    return "COG-BCI", "CL", cell, arms, ctx


def _cell_cogbci_nearear(subject):
    cell = load_cogbci([subject], sessions=("S1",), near_ear=True)[0]
    arms = enumerate_arms_generic(n_channels=cell.X.shape[1], n_components=4)
    ctx = functools.partial(compute_context_workload, channel_roles=NEAR_EAR_ROLES)
    return "COG-BCI near-ear", "CL", cell, arms, ctx


_CELLS = {
    "2a": _cell_2a,
    "2b": _cell_2b,
    "stew": _cell_stew,
    "uab_nearear": _cell_uab_nearear,
    "cogbci": _cell_cogbci,
    "cogbci_nearear": _cell_cogbci_nearear,
}


def _host_info() -> dict:
    """Machine identity, so the latency numbers can be contextualised and reproduced."""
    cpu = platform.processor() or "unknown"
    ram_gb = float("nan")
    n_cores = os.cpu_count() or -1
    if sys.platform == "darwin":  # platform.processor() returns just "arm" on macOS
        try:
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True, timeout=10,
            ).stdout.strip() or cpu
            mem = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True, timeout=10,
            ).stdout.strip()
            ram_gb = round(int(mem) / 1024**3, 1)
        except (subprocess.SubprocessError, OSError, ValueError):
            pass
    try:
        import torch  # noqa: PLC0415

        torch_v = torch.__version__
    except ImportError:
        torch_v = "not-installed"
    return {
        "host_cpu": cpu,
        "host_cores": n_cores,
        "host_ram_gb": ram_gb,
        "host_os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch_v,
        "blas_threads": 1,
    }


def _param_bytes(model) -> float:
    """Bytes of actual fitted numeric state --- the deployed-model footprint.

    Recursively walks the object graph summing ``ndarray.nbytes`` (and torch parameter
    bytes), so the figure excludes pickle/class overhead. This is the number that matters
    for an embedded target; ``pickle.dumps`` conflates parameters with serialisation
    metadata and with sklearn's stored hyper-parameters.
    """
    seen: set[int] = set()
    total = 0

    def walk(obj, depth: int = 0) -> None:
        nonlocal total
        if depth > 8 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            total += int(obj.nbytes)
            return
        if hasattr(obj, "parameters") and callable(obj.parameters):  # torch module
            try:
                total += sum(p.numel() * p.element_size() for p in obj.parameters())
                total += sum(b.numel() * b.element_size() for b in obj.buffers())
                return
            except (TypeError, AttributeError):
                pass
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v, depth + 1)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                walk(v, depth + 1)
        elif hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                walk(v, depth + 1)

    walk(model)
    return float(total)


def _n_params(model) -> float:
    """Count of fitted scalar parameters (torch tensors or numpy arrays)."""
    seen: set[int] = set()
    total = 0

    def walk(obj, depth: int = 0) -> None:
        nonlocal total
        if depth > 8 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            total += int(obj.size)
            return
        if hasattr(obj, "parameters") and callable(obj.parameters):
            try:
                total += sum(p.numel() for p in obj.parameters())
                return
            except (TypeError, AttributeError):
                pass
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v, depth + 1)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                walk(v, depth + 1)
        elif hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                walk(v, depth + 1)

    walk(model)
    return float(total)


def _profile_per_trial(predict_one, X: np.ndarray, *, repeats: int = 10) -> dict:
    """Per-trial (one-at-a-time) inference cost -- the deployment regime.

    Returns the full latency distribution, not just a median: a deployed real-time
    decoder misses its deadline on the tail, not on the typical trial.
    """
    for _ in range(5):
        predict_one(X[:1])  # warm-up
    lat: list[float] = []
    u0 = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    for _ in range(repeats):
        for i in range(len(X)):
            t0 = time.perf_counter()
            predict_one(X[i : i + 1])
            lat.append((time.perf_counter() - t0) * 1e3)
    cpu_per_trial = (resource.getrusage(resource.RUSAGE_SELF).ru_utime - u0) / (repeats * len(X))
    a = np.asarray(lat)
    return {
        "latency_ms_per_trial": float(np.median(a)),  # kept as the headline column
        "latency_p50_ms": float(np.percentile(a, 50)),
        "latency_p95_ms": float(np.percentile(a, 95)),
        "latency_p99_ms": float(np.percentile(a, 99)),
        "latency_iqr_ms": float(np.percentile(a, 75) - np.percentile(a, 25)),
        "latency_mean_ms": float(a.mean()),
        "cpu_ms_per_trial": float(cpu_per_trial * 1e3),
        "n_timing_samples": int(a.size),
    }


def _profile_online_update(
    base_contexts: np.ndarray, arm_costs: np.ndarray, d_ctx: int, n_arms: int,
    config: OPLBConfig, *, repeats: int = 5,
) -> dict:
    """Per-trial cost of the CCB's online adaptation: OPLB select + update (ms).

    This is the cost the bandit pays each trial to adapt online (build the per-arm context,
    choose an arm, update the linear posterior) that the static FBCSP/band-power/EEGNet
    pipelines never pay -- they would need a full offline refit instead. Drives the same
    ``select``/``update`` calls as the runner's stream loop (runner.py).
    """
    recent = np.zeros(N_ARM_FAMILIES)
    lat: list[float] = []
    for _ in range(repeats):
        policy = OPLB(d_psi=d_ctx + n_arms, n_arms=n_arms, config=config)
        for bc in base_contexts:
            ctx = _build_per_arm_contexts(bc, recent, n_arms, d_ctx, include_recent_rewards=False)
            t0 = time.perf_counter()
            a = policy.select(contexts=ctx, arm_costs=arm_costs)
            if a < 0:  # INFEASIBLE (cannot happen with a non-binding budget) -- skip timing
                continue
            policy.update(a, ctx[a], reward=1.0, realized_cost=float(arm_costs[a]))
            lat.append((time.perf_counter() - t0) * 1e3)
    if not lat:
        return {"online_update_ms_per_trial": float("nan"), "online_update_p95_ms": float("nan")}
    arr = np.asarray(lat)
    return {
        "online_update_ms_per_trial": float(np.median(arr)),
        "online_update_p95_ms": float(np.percentile(arr, 95)),
    }


def _model_kb(model) -> float:
    """Serialized-model footprint in KB (kept for continuity with earlier runs)."""
    try:
        return len(pickle.dumps(model)) / 1024.0
    except Exception:  # noqa: BLE001 -- torch fallback: raw parameter bytes
        return sum(p.numel() * p.element_size() for p in model.model_.parameters()) / 1024.0


def main(
    cells: str = typer.Option(
        "2a,2b,stew,uab_nearear,cogbci,cogbci_nearear",
        help="Comma list from " + ",".join(_CELLS),
    ),
    subjects: str = typer.Option("1", help="Comma list of subject ids to profile per cell."),
    seed: int = typer.Option(42),
    cnn_epochs: int = typer.Option(50, help="EEGNet training epochs."),
    repeats: int = typer.Option(10, help="Timing repeats over the test set per system."),
    output: Path = typer.Option(Path("results/hardware_efficiency.csv")),
) -> None:
    import torch  # noqa: PLC0415

    from thesis.baselines.cnn import EEGNet  # local import; torch is the optional `benchmark` extra
    torch.set_num_threads(1)  # single-core comparison (the embedded-deployment regime)

    host = _host_info()
    print("Host: {host_cpu} | {host_cores} cores | {host_ram_gb} GB | {host_os}".format(**host))
    print(f"      python {host['python']} numpy {host['numpy']} torch {host['torch']} "
          f"| BLAS threads pinned to 1\n")

    # CCB online config: the headline cell (alpha=0.5, window 50), budget non-binding for
    # timing (we measure select+update cost, not the knapsack gate).
    ccb_cfg = OPLBConfig(alpha=0.5, lambda_reg=1.0, budget=float("inf"),
                         window_size=50, discount_gamma=1.0, per_round_cap=None)
    subject_ids = [int(s.strip()) for s in subjects.split(",") if s.strip()]
    rows: list[dict] = []
    for key in [c.strip() for c in cells.split(",") if c.strip()]:
        if key not in _CELLS:
            print(f"  unknown cell {key!r}; skipping")
            continue
        for subject in subject_ids:
            try:
                label, paradigm, cell, arms, context_fn = _CELLS[key](subject)
            except Exception as exc:  # noqa: BLE001 -- a missing subject must not kill the sweep
                print(f"  {key} s{subject}: load failed ({type(exc).__name__}: {exc}); skipping")
                continue
            sf = cell.sfreq
            split = next(iter(matched_within_cv(cell, n_splits=5, fold_seed=seed)))
            Xtr, ytr = cell.X[split.train_idx], cell.y[split.train_idx]
            Xte, yte = cell.X[split.test_idx], cell.y[split.test_idx]
            n_ch = cell.X.shape[1]
            epoch_seconds = float(cell.X.shape[2]) / float(sf)
            print(f"\n=== {label} ({paradigm}, {n_ch} ch) s{subject}: "
                  f"{len(Xtr)} train / {len(Xte)} test, {epoch_seconds:.1f}s epochs ===")

            # --- CALIBRATION COST: wall-clock to fit each system on the training split ---
            t0 = time.perf_counter()
            fbcsp = FBCSP(sfreq=sf).fit(Xtr, ytr)
            fit_fbcsp = time.perf_counter() - t0

            t0 = time.perf_counter()
            bandp = BandPowerCL(sfreq=sf).fit(Xtr, ytr)
            fit_bandp = time.perf_counter() - t0

            t0 = time.perf_counter()
            surviving, heads, _ = fit_heads_on_calibration(cell, split.train_idx, arms, seed=seed)
            fit_ccb = time.perf_counter() - t0
            best = surviving[0]

            t0 = time.perf_counter()
            eegnet = EEGNet(sfreq=sf, epochs=cnn_epochs, seed=seed).fit(Xtr, ytr)
            fit_eegnet = time.perf_counter() - t0

            def ccb_predict(X, _heads=heads, _best=best, _ctx=context_fn, _sf=sf):
                for x in X:  # the per-trial inference cost: context + the selected arm's pipeline
                    _ctx(x, sfreq=_sf, recent_arm_rewards=None)
                return _heads[_best.arm_id].predict(X, _sf)

            systems = [
                ("FBCSP + LDA", fbcsp, fbcsp.predict, fit_fbcsp),
                ("BandPower + LDA", bandp, bandp.predict, fit_bandp),
                ("CCB", heads[best.arm_id], ccb_predict, fit_ccb),
                ("EEGNet (CNN)", eegnet, eegnet.predict, fit_eegnet),
            ]
            # CCB online-adaptation cost (select+update), measured once for the cell.
            arm_costs = np.array([a.cost for a in surviving], dtype=float)
            base_ctx = np.stack([context_fn(x, sfreq=sf, recent_arm_rewards=None) for x in Xte], axis=0)
            d_ctx = int(base_ctx.shape[1])  # derive from the actual context (2a/2b/workload dims differ)
            online = _profile_online_update(base_ctx, arm_costs, d_ctx, len(surviving), ccb_cfg)
            n_arms = len(surviving)

            for name, model, predict, fit_s in systems:
                kappa = float(compute_metrics(yte, predict(Xte)).kappa)
                prof = _profile_per_trial(predict, Xte, repeats=repeats)
                cpu_s = prof["cpu_ms_per_trial"] / 1e3
                is_ccb = name == "CCB"
                rows.append({
                    "dataset": label, "paradigm": paradigm, "n_channels": n_ch,
                    "subject": subject, "system": name, "kappa": round(kappa, 3),
                    # --- inference cost ---
                    "latency_ms_per_trial": round(prof["latency_ms_per_trial"], 4),
                    "latency_p50_ms": round(prof["latency_p50_ms"], 4),
                    "latency_p95_ms": round(prof["latency_p95_ms"], 4),
                    "latency_p99_ms": round(prof["latency_p99_ms"], 4),
                    "latency_iqr_ms": round(prof["latency_iqr_ms"], 4),
                    "latency_mean_ms": round(prof["latency_mean_ms"], 4),
                    "cpu_ms_per_trial": round(prof["cpu_ms_per_trial"], 4),
                    # --- real-time feasibility: fraction of the epoch's own duration ---
                    "epoch_seconds": round(epoch_seconds, 3),
                    "realtime_factor": round(prof["latency_ms_per_trial"] / (epoch_seconds * 1e3), 6),
                    "realtime_factor_p99": round(prof["latency_p99_ms"] / (epoch_seconds * 1e3), 6),
                    # --- footprint ---
                    "model_footprint_kb": round(_model_kb(model), 1),
                    "param_bytes": int(_param_bytes(model)),
                    "n_params": int(_n_params(model)),
                    # --- calibration cost ---
                    "fit_seconds": round(fit_s, 4),
                    # --- online-adaptation cost: CCB only (static methods pay none per trial) ---
                    "online_update_ms_per_trial": (
                        round(online["online_update_ms_per_trial"], 4) if is_ccb else 0.0),
                    "online_update_p95_ms": (
                        round(online["online_update_p95_ms"], 4) if is_ccb else 0.0),
                    "online_update_us_per_arm": (
                        round(online["online_update_ms_per_trial"] * 1e3 / n_arms, 4)
                        if is_ccb and n_arms else 0.0),
                    "n_arms": n_arms if is_ccb else 0,
                    # --- derived energy proxy, bracketed (collinear with cpu time by construction) ---
                    "energy_mj_per_trial": round(cpu_s * TDP_LAPTOP_W * 1e3, 3),
                    "energy_mj_embedded_2w": round(cpu_s * TDP_EMBEDDED_W * 1e3, 3),
                    "tdp_watts": TDP_LAPTOP_W,
                    # --- provenance ---
                    "n_train": len(Xtr), "n_test": len(Xte),
                    "n_timing_samples": prof["n_timing_samples"],
                    **host,
                })
                tag = f"  +{online['online_update_ms_per_trial']:.3f} ms online" if is_ccb else ""
                print(f"  {name:18s} k={kappa:+.3f}  fit {fit_s:8.2f}s  "
                      f"infer p50 {prof['latency_ms_per_trial']:7.3f} / p99 "
                      f"{prof['latency_p99_ms']:7.3f} ms{tag}")

    if not rows:
        print("No rows generated."); raise typer.Exit(code=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"\nSaved {len(rows)} rows -> {output}")


if __name__ == "__main__":
    typer.run(main)
