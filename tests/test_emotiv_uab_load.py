"""Offline tests for the UAB loader (synthetic parquet — no real data needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis.data.emotiv_uab_load import (
    UAB_CHANNELS,
    UAB_DIFFICULTY,
    UAB_RAW_EEG_COLUMNS,
    _block_to_epochs,
    load_emotiv_uab,
)


def _make_uab_parquet(path: Path, *, n_subjects: int = 2, task_samples: int = 1500) -> Path:
    """Write a tiny UAB-shaped parquet: EEG.* cols + subject/test/phase, +4200 µV offset."""
    rng = np.random.default_rng(0)
    frames = []
    for s in range(1, n_subjects + 1):
        for test in (1, 2, 3):
            for phase in (1, 2, 3):
                n = task_samples if phase == 2 else 80
                cols = {c: rng.standard_normal(n) * 25.0 + 4200.0 for c in UAB_RAW_EEG_COLUMNS}
                cols["subject"] = [f"subject_{s:02d}"] * n
                cols["test"] = [test] * n
                cols["phase"] = [phase] * n
                frames.append(pd.DataFrame(cols))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(path, engine="pyarrow")
    return path


def test_block_to_epochs_removes_dc_and_shapes():
    rng = np.random.default_rng(1)
    block = rng.standard_normal((14, 1500)) * 25.0 + 4200.0  # Emotiv-like offset
    ep = _block_to_epochs(block, native_sfreq=128.0, target_sfreq=250.0, epoch_seconds=4.0)
    # 1500 samples @128Hz ≈ 11.7s → resample 250Hz ≈ 2930 samples → 2 windows of 1000.
    assert ep.shape[1] == 14 and ep.shape[2] == 1000
    assert ep.shape[0] >= 2
    # DC offset removed: per-epoch mean far below the 4200 raw offset.
    assert np.abs(ep.mean()) < 50.0


def test_load_emitssubjectdata_full_and_near_ear(tmp_path: Path):
    p = _make_uab_parquet(tmp_path / "eeg.parquet")
    full = load_emotiv_uab(data_path=p)
    assert len(full) == 2
    s = full[0]
    assert s.dataset_name == "UAB"
    assert s.n_channels == 14
    assert s.sfreq == 250.0
    # three difficulty blocks → three labels present
    assert set(s.y) == {"low", "medium", "high"}
    # session metadata records the difficulty-block id (1/2/3)
    assert set(s.metadata["session"]) == {"1", "2", "3"}

    near = load_emotiv_uab(data_path=p, near_ear=True)
    assert near[0].n_channels == 2
    assert near[0].dataset_name == "UAB-nearear"


def test_subject_filter_and_difficulty_map(tmp_path: Path):
    p = _make_uab_parquet(tmp_path / "eeg.parquet", n_subjects=3)
    only2 = load_emotiv_uab(subjects=[2], data_path=p)
    assert len(only2) == 1 and only2[0].subject == 2
    assert UAB_DIFFICULTY == {1: "low", 2: "medium", 3: "high"}
    assert UAB_RAW_EEG_COLUMNS[4] == "EEG.T7" and UAB_CHANNELS[4] == "T7"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_emotiv_uab(data_path=Path("/nonexistent/eeg.parquet"))
