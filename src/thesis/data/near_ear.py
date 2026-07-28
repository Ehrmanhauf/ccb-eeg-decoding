"""Near-ear (T7/T8) montage subset — the deployable proxy for around-ear EEG.

The thesis's near-ear cells are produced by subsetting any montage that contains
the temporal pair **T7/T8** down to those two channels, *at load time, by
electrode position, before any model sees the data*.

**No-leakage rationale (state explicitly in the thesis).** This is the same
position-based operation already used for the Cho2017 C3/Cz/C4 subset
(``moabb_load``): it is driven by deployment-hardware geometry, never by labels
or signal statistics, so it does **not** violate the generalized no-leakage rule
(``CLAUDE.md`` §2). We remove channels; we never use the full montage to inform a
low-channel decision.

T7/T8 is a *proxy* for around-ear hardware (in-ear / cEEGrid), **not** in-ear
recording — the thesis claims it as a near-ear proxy plainly. T7/T8 sit just
above and anterior to the ears on every montage in the reframe (STEW / UAB EPOC X
14-ch; COG-BCI 64-ch ActiCap), giving a matched near-ear position across three
amplifiers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from thesis.data.load import SubjectData

# The near-ear deployable pair (advisor work plan §4). Exercised by the offline
# tests; consumed by ``select_near_ear`` and by every near-ear runner cell.
NEAR_EAR: tuple[str, str] = ("T7", "T8")

# Channel-roles for the 2-channel near-ear montage, consumed by the workload
# context (``compute_context_workload``) and the band-power baseline. T7/T8 are
# *temporal*, not frontal/parietal, so the frontal-θ and parietal-α aggregates
# are deliberately empty (the context zeroes them, see ``context_cl`` —
# ``_safe_mean_across_indices`` returns 0.0 on an empty list); the left/right
# temporal pair T7/T8 supplies the only available asymmetry proxy. Index 0 = T7
# (left), index 1 = T8 (right) in the subset emitted by ``select_near_ear``.
NEAR_EAR_ROLES: dict[str, list[int]] = {
    "frontal": [],
    "parietal": [],
    "f3": [0],  # T7 (left temporal) — asymmetry left arm
    "f4": [1],  # T8 (right temporal) — asymmetry right arm
}


def select_near_ear(data: SubjectData, channel_names: Sequence[str]) -> SubjectData:
    """Return a copy of ``data`` subset to the T7/T8 channels (position-based).

    Parameters
    ----------
    data : full-montage :class:`SubjectData`.
    channel_names : the full-montage channel order matching ``data.X``'s channel
        axis (axis 1). The two near-ear channels are located **by name**.

    The returned object keeps ``y`` / ``metadata`` / ``subject`` / ``sfreq`` and
    sets ``X = data.X[:, [idx_T7, idx_T8], :]`` (channels in T7, T8 order) and
    ``dataset_name = f"{data.dataset_name}-nearear"`` so downstream code can tell
    the near-ear cell apart.
    """
    names = list(channel_names)
    if data.X.shape[1] != len(names):
        raise ValueError(
            f"channel_names length {len(names)} != X channel axis {data.X.shape[1]}"
        )
    try:
        idx = [names.index(ch) for ch in NEAR_EAR]
    except ValueError as exc:
        raise ValueError(
            f"near-ear subset needs channels {NEAR_EAR}; montage has {names}"
        ) from exc
    return replace(
        data,
        X=data.X[:, idx, :].copy(),
        dataset_name=f"{data.dataset_name}-nearear",
    )
