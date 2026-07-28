"""Offline structural tests for the COG-BCI PVT vigilance loader.

The loader reads gitignored real EEGLAB + behavioral data; these cover the parts
that do not need the archive: the montage (a 62-channel canonical EEG set with the
non-EEG ECG1 channel dropped to match the workload loader), the workload/vigilance
roles map (which reserves T7/T8 for the near-ear subset), and the stimulus event code.
"""

from __future__ import annotations

from thesis.data.cogbci_pvt_load import (
    _STIM_ANNOT,
    COGBCI_PVT_EEG,
    COGBCI_PVT_ROLES,
    PVT_SESSIONS,
)


def test_montage_is_62ch_no_ecg_with_near_ear_pair():
    assert len(COGBCI_PVT_EEG) == 62
    assert len(set(COGBCI_PVT_EEG)) == 62
    assert "T7" in COGBCI_PVT_EEG and "T8" in COGBCI_PVT_EEG
    assert "ECG1" not in COGBCI_PVT_EEG  # non-EEG channel dropped to match the workload montage


def test_roles_exclude_near_ear_channels():
    roles = COGBCI_PVT_ROLES
    t7, t8 = COGBCI_PVT_EEG.index("T7"), COGBCI_PVT_EEG.index("T8")
    for agg in ("frontal", "parietal"):
        assert t7 not in roles[agg] and t8 not in roles[agg], "near-ear pair reserved"
    assert roles["frontal"] and roles["parietal"]
    assert set(roles["frontal"]).isdisjoint(roles["parietal"])


def test_f3_f4_asymmetry_pair_resolves():
    assert COGBCI_PVT_EEG[COGBCI_PVT_ROLES["f3"][0]] == "F3"
    assert COGBCI_PVT_EEG[COGBCI_PVT_ROLES["f4"][0]] == "F4"
    assert COGBCI_PVT_ROLES["f3"][0] not in COGBCI_PVT_ROLES["frontal"]


def test_stimulus_code_and_sessions():
    assert _STIM_ANNOT == "13"
    assert PVT_SESSIONS == ("S1", "S2", "S3")
