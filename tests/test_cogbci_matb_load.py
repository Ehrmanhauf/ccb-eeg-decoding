"""Offline structural tests for the COG-BCI MATB competition-split loader.

The loader reads gitignored real EEGLAB data, so these tests cover the parts that
do not need the archive: the canonical montage, the workload channel-roles map, and
the difficulty→3-class label mapping.
"""

from __future__ import annotations

from thesis.data.cogbci_matb_load import (
    COGBCI_MATBCOMP_EEG,
    COGBCI_MATBCOMP_ROLES,
    MATB_COMP_SESSIONS,
    MATB_DIFFICULTY,
)


def test_montage_is_61_channels_with_near_ear_pair():
    assert len(COGBCI_MATBCOMP_EEG) == 61
    assert len(set(COGBCI_MATBCOMP_EEG)) == 61  # no duplicates
    assert "T7" in COGBCI_MATBCOMP_EEG and "T8" in COGBCI_MATBCOMP_EEG


def test_roles_map_indices_are_valid_and_disjoint_where_expected():
    roles = COGBCI_MATBCOMP_ROLES
    n = len(COGBCI_MATBCOMP_EEG)
    for key in ("frontal", "parietal", "f3", "f4"):
        assert key in roles
        assert all(0 <= i < n for i in roles[key])
    # F3 / F4 resolve to the named electrodes
    assert COGBCI_MATBCOMP_EEG[roles["f3"][0]] == "F3"
    assert COGBCI_MATBCOMP_EEG[roles["f4"][0]] == "F4"
    # F3 / F4 are the asymmetry pair, held out of the frontal-theta aggregate
    assert roles["f3"][0] not in roles["frontal"]
    assert roles["f4"][0] not in roles["frontal"]
    # frontal and parietal aggregates are both populated and disjoint
    assert roles["frontal"] and roles["parietal"]
    assert set(roles["frontal"]).isdisjoint(roles["parietal"])


def test_near_ear_channels_are_not_in_frontal_or_parietal_aggregates():
    # T7/T8 are temporal — they must not leak into the frontal/parietal CL aggregates
    t7 = COGBCI_MATBCOMP_EEG.index("T7")
    t8 = COGBCI_MATBCOMP_EEG.index("T8")
    for agg in ("frontal", "parietal"):
        assert t7 not in COGBCI_MATBCOMP_ROLES[agg]
        assert t8 not in COGBCI_MATBCOMP_ROLES[agg]


def test_difficulty_maps_to_three_class_workload():
    assert MATB_DIFFICULTY == {"MATBeasy": "low", "MATBmed": "medium", "MATBdiff": "high"}
    assert set(MATB_DIFFICULTY.values()) == {"low", "medium", "high"}


def test_competition_sessions_are_the_two_matb_sessions():
    # S3 ships resting-state only, so the MATB cross-session pair is S1 → S2
    assert MATB_COMP_SESSIONS == ("S1", "S2")
