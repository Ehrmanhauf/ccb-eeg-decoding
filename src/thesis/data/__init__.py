"""Data loading for the thesis dataset panel.

MI: BCI Competition IV-2a/2b (``load``), Cho2017 (``moabb_load``).
CL: STEW (``stew_load``), WAUC (``wauc_load``), and the near-ear reframe datasets
UAB (``emotiv_uab_load``) + COG-BCI (``cogbci_load``). Every loader emits the
shared :class:`SubjectData` interface. ``near_ear`` selects the deployable T7/T8
proxy subset (``near_ear`` module).
"""

from thesis.data.load import CLASS_LABELS, SubjectData, load_bci2a, load_bci2b_screening
from thesis.data.near_ear import NEAR_EAR, NEAR_EAR_ROLES, select_near_ear

__all__ = [
    "CLASS_LABELS",
    "NEAR_EAR",
    "NEAR_EAR_ROLES",
    "SubjectData",
    "load_bci2a",
    "load_bci2b_screening",
    "load_cogbci",
    "load_emotiv_uab",
    "select_near_ear",
]


def __getattr__(name: str):
    # Lazy imports: the UAB loader pulls in pyarrow and the COG-BCI loader pulls
    # in heavy MNE paths; keep ``import thesis.data`` cheap for the MI path.
    if name == "load_emotiv_uab":
        from thesis.data.emotiv_uab_load import load_emotiv_uab

        return load_emotiv_uab
    if name == "load_cogbci":
        from thesis.data.cogbci_load import load_cogbci

        return load_cogbci
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
