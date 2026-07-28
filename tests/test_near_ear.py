"""Offline tests for the near-ear (T7/T8) subset helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis.data import SubjectData
from thesis.data.near_ear import NEAR_EAR, NEAR_EAR_ROLES, select_near_ear


def _make(channel_names: list[str], n_trials: int = 12) -> SubjectData:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_trials, len(channel_names), 500))
    y = np.array(["low", "high"] * (n_trials // 2))
    meta = pd.DataFrame({"subject": [1] * n_trials, "session": ["0"] * n_trials, "run": ["0"] * n_trials})
    return SubjectData(subject=1, dataset_name="X", X=X, y=y, metadata=meta, sfreq=250.0)


def test_selects_t7_t8_in_order():
    names = ["AF3", "F7", "T7", "P7", "T8", "F4"]  # T7 at idx2, T8 at idx4
    data = _make(names)
    near = select_near_ear(data, names)
    assert near.X.shape == (12, 2, 500)
    # channel 0 of the subset is T7, channel 1 is T8
    assert np.allclose(near.X[:, 0, :], data.X[:, 2, :])
    assert np.allclose(near.X[:, 1, :], data.X[:, 4, :])
    assert near.dataset_name == "X-nearear"
    # y / metadata / subject unchanged
    assert np.array_equal(near.y, data.y)
    assert near.subject == 1


def test_raises_when_t7_or_t8_absent():
    names = ["AF3", "F7", "C3", "P7", "F4", "AF4"]  # no T7/T8
    data = _make(names)
    with pytest.raises(ValueError, match="near-ear"):
        select_near_ear(data, names)


def test_raises_on_channel_name_length_mismatch():
    data = _make(["AF3", "T7", "T8"])
    with pytest.raises(ValueError, match="channel_names length"):
        select_near_ear(data, ["AF3", "T7"])  # too few


def test_near_ear_roles_are_two_channel_safe():
    # f3/f4 index the 2-channel subset (T7=0, T8=1); frontal/parietal empty.
    assert NEAR_EAR == ("T7", "T8")
    assert NEAR_EAR_ROLES["f3"] == [0]
    assert NEAR_EAR_ROLES["f4"] == [1]
    assert NEAR_EAR_ROLES["frontal"] == []
    assert NEAR_EAR_ROLES["parietal"] == []
