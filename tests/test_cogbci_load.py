"""Offline tests for the COG-BCI loader core (synthetic MNE Raw — no zips/real data).

The zip-extraction glue in ``load_cogbci`` is exercised by the Phase-2 smoke run
against the real data; here we test the channel-normalisation and boundary-aware
epoching logic that does the scientific work.
"""

from __future__ import annotations

import mne
import numpy as np

from thesis.data.cogbci_load import (
    COGBCI_CANONICAL_EEG,
    COGBCI_CHANNEL_ROLES,
    _match_eeg_members,
    _pick_canonical,
    _raw_to_epochs,
)

mne.set_log_level("ERROR")


def _make_raw(ch_names: list[str], *, sfreq: float = 500.0, n_samples: int = 6000) -> mne.io.BaseRaw:
    data = np.random.default_rng(0).standard_normal((len(ch_names), n_samples)) * 1e-5
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def test_canonical_montage_has_t7_t8_and_62_channels():
    assert len(COGBCI_CANONICAL_EEG) == 62
    assert "T7" in COGBCI_CANONICAL_EEG and "T8" in COGBCI_CANONICAL_EEG
    assert "Cz" not in COGBCI_CANONICAL_EEG  # absent for subjects 1–9
    assert "ECG1" not in COGBCI_CANONICAL_EEG


def test_channel_roles_resolve_to_valid_indices():
    for role, idxs in COGBCI_CHANNEL_ROLES.items():
        for i in idxs:
            assert 0 <= i < 62, f"{role} index {i} out of range"
    # F3/F4 asymmetry pair are single channels
    assert len(COGBCI_CHANNEL_ROLES["f3"]) == 1
    assert COGBCI_CANONICAL_EEG[COGBCI_CHANNEL_ROLES["f3"][0]] == "F3"
    assert COGBCI_CANONICAL_EEG[COGBCI_CHANNEL_ROLES["f4"][0]] == "F4"


def test_pick_canonical_drops_ecg_cz_and_reorders():
    # Build a raw with canonical channels in REVERSED order + extra ECG1 + Cz.
    raw = _make_raw(list(reversed(COGBCI_CANONICAL_EEG)) + ["ECG1", "Cz"])
    picked = _pick_canonical(raw)
    assert picked.ch_names == list(COGBCI_CANONICAL_EEG)  # reordered, extras dropped


def test_pick_canonical_raises_on_missing_channel():
    import pytest

    raw = _make_raw(list(COGBCI_CANONICAL_EEG[:-1]))  # drop one canonical channel
    with pytest.raises(ValueError, match="missing canonical"):
        _pick_canonical(raw)


def test_raw_to_epochs_respects_boundary():
    # 12 s at 500 Hz, a boundary at 6 s → two contiguous 6 s segments.
    raw = _make_raw(list(COGBCI_CANONICAL_EEG), sfreq=500.0, n_samples=6000)
    raw.set_annotations(mne.Annotations(onset=[6.0], duration=[0.0], description=["boundary"]))
    ep = _raw_to_epochs(raw, target_sfreq=250.0, epoch_seconds=4.0)
    # each 6 s segment → one 4 s epoch (1000 samples) → 2 epochs, 62 channels.
    assert ep.shape == (2, 62, 1000)


def test_match_eeg_members_handles_single_and_double_nesting():
    # sub-01/02 are single-nested; sub-03+ double-nest sub-NN/sub-NN/...
    single = ["sub-01/ses-S1/eeg/zeroBACK.set", "sub-01/ses-S1/eeg/zeroBACK.fdt",
              "sub-01/ses-S1/eeg/PVT.set", "sub-01/ses-S2/eeg/zeroBACK.set"]
    double = ["sub-03/sub-03/ses-S1/eeg/zeroBACK.set", "sub-03/sub-03/ses-S1/eeg/zeroBACK.fdt"]
    assert set(_match_eeg_members(single, "S1", "zeroBACK")) == {single[0], single[1]}
    assert set(_match_eeg_members(double, "S1", "zeroBACK")) == set(double)
    # session- and stem-specific (no cross-matching)
    assert _match_eeg_members(single, "S1", "oneBACK") == []
    assert _match_eeg_members(single, "S3", "zeroBACK") == []


def test_raw_to_epochs_no_boundary_single_run():
    raw = _make_raw(list(COGBCI_CANONICAL_EEG), sfreq=500.0, n_samples=5000)  # 10 s
    ep = _raw_to_epochs(raw, target_sfreq=250.0, epoch_seconds=4.0)
    # 10 s → resample 250 Hz → 2500 samples → 2 windows of 1000.
    assert ep.shape == (2, 62, 1000)
