"""Offline tests for the STEW loader.

These tests don't touch the actual STEW dataset (which is not vendored).
They cover the loader's logic — workload binning, segment-to-epoch
reshaping, and end-to-end loading against a synthetic data layout — so
regressions on Phase-5 §2.6 / Workstream C.3 are caught without
network access or IEEE DataPort credentials.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis.data.stew_load import (
    STEW_BIN_EDGES,
    STEW_CHANNELS,
    STEW_NATIVE_SFREQ,
    _bin_rating,
    _load_ratings,
    _read_segment,
    _segment_to_epochs,
    load_stew,
)


class TestBinRating:
    """3-level workload binning per ccb-formulation.md §2.6."""

    @pytest.mark.parametrize(
        "rating, expected",
        [
            (1, "low"), (2, "low"), (3, "low"),
            (4, "medium"), (5, "medium"), (6, "medium"),
            (7, "high"), (8, "high"), (9, "high"),
        ],
    )
    def test_in_range(self, rating: int, expected: str) -> None:
        assert _bin_rating(rating) == expected

    @pytest.mark.parametrize("rating", [0, -1, 10, 100])
    def test_out_of_range_raises(self, rating: int) -> None:
        with pytest.raises(ValueError, match="outside the 1..9 range"):
            _bin_rating(rating)

    def test_bin_edges_partition_1_to_9(self) -> None:
        """Every integer rating 1..9 falls into exactly one bin."""
        for r in range(1, 10):
            hits = [
                label for lo, hi, label in STEW_BIN_EDGES if lo <= r <= hi
            ]
            assert len(hits) == 1


class TestLoadRatings:
    """ratings.txt parsing — must tolerate both comma-CSV and whitespace forms."""

    def test_csv_form(self, tmp_path: Path) -> None:
        path = tmp_path / "ratings.txt"
        path.write_text("1,2,8\n2,3,5\n3,1,9\n")
        ratings = _load_ratings(path)
        assert ratings == {1: (2, 8), 2: (3, 5), 3: (1, 9)}

    def test_csv_with_spaces(self, tmp_path: Path) -> None:
        # IEEE DataPort example uses "1, 2, 8" — comma followed by space.
        path = tmp_path / "ratings.txt"
        path.write_text("1, 2, 8\n2, 3, 5\n")
        ratings = _load_ratings(path)
        assert ratings == {1: (2, 8), 2: (3, 5)}

    def test_drops_missing_rating_subjects(self, tmp_path: Path) -> None:
        # Subjects 5, 24, 42 are flagged on IEEE DataPort as unavailable.
        path = tmp_path / "ratings.txt"
        path.write_text("1,2,8\n5,0,0\n6,3,6\n24,0,0\n42,0,0\n")
        ratings = _load_ratings(path)
        assert set(ratings.keys()) == {1, 6}


class TestReadSegment:
    def test_shape(self, tmp_path: Path) -> None:
        arr = np.random.randn(1000, 14).astype(np.float64)
        path = tmp_path / "sub01_lo.txt"
        np.savetxt(path, arr)
        out = _read_segment(path)
        assert out.shape == (14, 1000)
        np.testing.assert_allclose(out, arr.T)

    def test_wrong_channel_count_raises(self, tmp_path: Path) -> None:
        arr = np.random.randn(1000, 8).astype(np.float64)
        path = tmp_path / "sub01_lo.txt"
        np.savetxt(path, arr)
        with pytest.raises(ValueError, match=r"expected .*14.* layout"):
            _read_segment(path)


class TestSegmentToEpochs:
    """Segment → epoch reshape + resample correctness."""

    def test_full_segment_yields_37_epochs(self) -> None:
        seg = np.random.randn(14, int(STEW_NATIVE_SFREQ * 150)).astype(np.float64)
        epochs = _segment_to_epochs(
            seg,
            native_sfreq=STEW_NATIVE_SFREQ,
            target_sfreq=250.0,
            epoch_seconds=4.0,
        )
        # 150 s / 4 s/epoch = 37 (with one extra 2 s dangling, trimmed).
        assert epochs.shape == (37, 14, 1000)

    def test_no_resample_passthrough(self) -> None:
        # native == target → resample should be a no-op shape-wise.
        seg = np.random.randn(14, int(STEW_NATIVE_SFREQ * 8)).astype(np.float64)
        epochs = _segment_to_epochs(
            seg,
            native_sfreq=STEW_NATIVE_SFREQ,
            target_sfreq=STEW_NATIVE_SFREQ,
            epoch_seconds=4.0,
        )
        # 8 s / 4 s = 2 epochs, 14 channels, 4 * 128 = 512 samples each.
        assert epochs.shape == (2, 14, 512)

    def test_too_short_raises(self) -> None:
        seg = np.random.randn(14, 100).astype(np.float64)
        with pytest.raises(ValueError, match="too short"):
            _segment_to_epochs(
                seg,
                native_sfreq=STEW_NATIVE_SFREQ,
                target_sfreq=250.0,
                epoch_seconds=4.0,
            )


def _write_fake_stew_layout(root: Path, subjects: list[int], ratings: dict[int, tuple[int, int]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    n_samples = int(STEW_NATIVE_SFREQ * 150)
    rng = np.random.default_rng(0)
    for sid in subjects:
        rest = rng.standard_normal((n_samples, 14))
        test = rng.standard_normal((n_samples, 14))
        np.savetxt(root / f"sub{sid:02d}_lo.txt", rest)
        np.savetxt(root / f"sub{sid:02d}_hi.txt", test)
    lines = []
    for sid in sorted({*subjects, 5, 24, 42}):
        if sid in ratings:
            rest_r, test_r = ratings[sid]
            lines.append(f"{sid},{rest_r},{test_r}")
        else:
            lines.append(f"{sid},0,0")  # missing-rating entries
    (root / "ratings.txt").write_text("\n".join(lines) + "\n")


class TestLoadStew:
    """End-to-end loader test against a synthetic STEW layout."""

    def test_returns_subject_data_per_subject(self, tmp_path: Path) -> None:
        subjects = [1, 2, 3]
        ratings = {1: (2, 8), 2: (4, 5), 3: (7, 9)}
        _write_fake_stew_layout(tmp_path, subjects, ratings)

        out = load_stew(data_root=tmp_path)
        assert len(out) == 3
        for s in out:
            assert s.dataset_name == "STEW"
            assert s.sfreq == 250.0
            assert s.n_channels == 14
            # 150 s / 4 s = 37 epochs per segment × 2 segments = 74.
            assert s.n_trials == 74
            # session column carries rest/multitask split.
            assert set(s.metadata["session"].unique()) == {"rest", "multitask"}

    def test_label_binning_matches_ratings(self, tmp_path: Path) -> None:
        subjects = [1]
        ratings = {1: (2, 8)}  # rest=low, test=high
        _write_fake_stew_layout(tmp_path, subjects, ratings)
        out = load_stew(data_root=tmp_path)
        rest_mask = out[0].metadata["session"].values == "rest"
        assert set(out[0].y[rest_mask]) == {"low"}
        assert set(out[0].y[~rest_mask]) == {"high"}

    def test_missing_rating_subjects_dropped_silently(self, tmp_path: Path) -> None:
        # Subjects 5, 24, 42 have 0,0 ratings and should be skipped.
        subjects = [5, 24, 42]
        ratings: dict[int, tuple[int, int]] = {}
        _write_fake_stew_layout(tmp_path, subjects, ratings)
        out = load_stew(subjects=[5, 24, 42], data_root=tmp_path)
        assert out == []
