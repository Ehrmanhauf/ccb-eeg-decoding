"""Offline tests for the WAUC loader.

These tests do not touch the actual WAUC dataset (which is not vendored;
the full archive requires extraction under data/WAUC/process/, see
``data/WAUC.README.md``). They cover the loader's logic — channel-column
resolution, labels-CSV parsing, session-to-epoch reshaping, subject ID
mapping, and end-to-end loading — against a synthetic data layout that
mirrors the actual ``process.rar`` extraction.

ref: design-doc/ccb-formulation.md §2.7, src/thesis/data/wauc_load.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis.data.wauc_load import (
    WAUC_CHANNEL_ROLES,
    WAUC_EEG_CHANNELS,
    WAUC_NATIVE_SFREQ,
    WAUC_SESSIONS,
    _load_wauc_labels,
    _read_eeg_csv,
    _resolve_channel_columns,
    _session_to_epochs,
    load_wauc,
    subject_id_to_partid,
)


class TestSubjectIdMapping:
    """Filesystem subject ID ↔ ratings ``Participant ID`` mapping."""

    @pytest.mark.parametrize(
        "sid, expected_partid",
        [(1, 1001), (2, 1002), (20, 1020), (28, 1028), (48, 1048)],
    )
    def test_in_range(self, sid: int, expected_partid: int) -> None:
        assert subject_id_to_partid(sid) == expected_partid

    @pytest.mark.parametrize("sid", [0, -1, 49, 100])
    def test_out_of_range_raises(self, sid: int) -> None:
        with pytest.raises(ValueError, match=r"1\.\.48"):
            subject_id_to_partid(sid)


class TestChannelRoles:
    """WAUC_CHANNEL_ROLES uses channel positions inside the 8-channel layout."""

    def test_keys(self) -> None:
        assert set(WAUC_CHANNEL_ROLES.keys()) == {"frontal", "parietal", "f3", "f4"}

    def test_indices_within_bounds(self) -> None:
        for role, idx_list in WAUC_CHANNEL_ROLES.items():
            for i in idx_list:
                assert 0 <= i < len(WAUC_EEG_CHANNELS), f"role {role!r} index {i} out of bounds"

    def test_f3_f4_are_singletons(self) -> None:
        assert len(WAUC_CHANNEL_ROLES["f3"]) == 1
        assert len(WAUC_CHANNEL_ROLES["f4"]) == 1

    def test_f3_points_to_left_frontal_proxy(self) -> None:
        # AF7 is the leftmost frontal electrode in WAUC's 8-channel layout.
        assert WAUC_CHANNEL_ROLES["f3"] == [WAUC_EEG_CHANNELS.index("AF7")]

    def test_f4_points_to_right_frontal_proxy(self) -> None:
        # AF8 is the rightmost frontal electrode in WAUC's 8-channel layout.
        assert WAUC_CHANNEL_ROLES["f4"] == [WAUC_EEG_CHANNELS.index("AF8")]


class TestResolveChannelColumns:
    """Channel column auto-resolution against the on-disk header."""

    def test_bare_channel_names_match_on_disk_order(self) -> None:
        # The actual extracted file has channel cols in this order; the
        # helper must return them in that same order.
        cols = list(WAUC_EEG_CHANNELS) + ["fs", "info", "session_no"]
        df = pd.DataFrame({c: [0.0] for c in cols})
        resolved = _resolve_channel_columns(df)
        assert resolved == list(WAUC_EEG_CHANNELS)

    def test_eeg_prefixed_names_tolerated(self) -> None:
        cols = {f"eeg_{c}": [0.0] for c in WAUC_EEG_CHANNELS}
        cols.update({"fs": [0.0], "info": ["session"], "session_no": [1]})
        df = pd.DataFrame(cols)
        resolved = _resolve_channel_columns(df)
        assert resolved == [f"eeg_{c}" for c in WAUC_EEG_CHANNELS]

    def test_missing_channel_raises(self) -> None:
        cols = {c: [0.0] for c in WAUC_EEG_CHANNELS[:6]}  # only 6 of 8
        df = pd.DataFrame(cols)
        with pytest.raises(ValueError, match="could not resolve WAUC EEG channel"):
            _resolve_channel_columns(df)

    def test_unrecognised_naming_raises(self) -> None:
        df = pd.DataFrame({f"ch{i}": [0.0] for i in range(8)})
        with pytest.raises(ValueError, match="could not resolve"):
            _resolve_channel_columns(df)


class TestLoadWaucLabels:
    """subjective_ratings_with_labels.csv parsing."""

    def test_minimal_well_formed_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "subjective_ratings_with_labels.csv"
        path.write_text(
            "Participant ID,session_no,mw_labels,pw_labels\n"
            "1001,1,0.0,0.0\n"
            "1001,2,1.0,0.0\n"
            "1002,1,0.0,1.0\n"
            "1002,2,1.0,2.0\n"
        )
        df = _load_wauc_labels(path)
        assert df.index.names == ["partid", "session_no"]
        assert df.loc[(1001, 1), "mw_label"] == "low"
        assert df.loc[(1001, 2), "mw_label"] == "high"
        assert df.loc[(1002, 2), "pw_label"] == 2

    def test_missing_pw_labels_defaults_to_sentinel(self, tmp_path: Path) -> None:
        path = tmp_path / "subjective_ratings_with_labels.csv"
        path.write_text(
            "Participant ID,session_no,mw_labels\n1001,1,0.0\n"
        )
        df = _load_wauc_labels(path)
        assert df.loc[(1001, 1), "pw_label"] == -1

    def test_unknown_mw_label_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "subjective_ratings_with_labels.csv"
        path.write_text(
            "Participant ID,session_no,mw_labels\n1001,1,3.0\n"
        )
        with pytest.raises(ValueError, match="unexpected mw_labels values"):
            _load_wauc_labels(path)

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "subjective_ratings_with_labels.csv"
        path.write_text("Participant ID,mw_labels\n1001,0.0\n")
        with pytest.raises(ValueError, match="missing required columns"):
            _load_wauc_labels(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="WAUC download"):
            _load_wauc_labels(tmp_path / "does_not_exist.csv")


class TestReadEegCsv:
    """enobio_eeg_asr.csv parsing with metadata-column validation."""

    def test_well_formed_passthrough(self, tmp_path: Path) -> None:
        n = 10
        cols: dict[str, np.ndarray | list] = {ch: np.random.randn(n) for ch in WAUC_EEG_CHANNELS}
        cols["fs"] = [500.0] * n
        cols["info"] = ["session"] * n
        cols["session_no"] = [1] * n
        path = tmp_path / "enobio_eeg_asr.csv"
        pd.DataFrame(cols).to_csv(path, index=False)
        df_out = _read_eeg_csv(path)
        assert {"fs", "info", "session_no"}.issubset(df_out.columns)
        assert len(df_out) == n

    def test_missing_metadata_column_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({ch: [0.0] for ch in WAUC_EEG_CHANNELS})
        path = tmp_path / "enobio_eeg_asr.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing required metadata"):
            _read_eeg_csv(path)


class TestSessionToEpochs:
    """Session → epoch reshape + resample correctness."""

    def _make_session_df(self, n_samples: int, sfreq: float = WAUC_NATIVE_SFREQ) -> pd.DataFrame:
        cols: dict[str, np.ndarray | list] = {
            ch: np.random.randn(n_samples) for ch in WAUC_EEG_CHANNELS
        }
        cols["fs"] = np.full(n_samples, sfreq)
        cols["info"] = ["session"] * n_samples
        cols["session_no"] = np.ones(n_samples, dtype=int)
        return pd.DataFrame(cols)

    def test_8s_segment_yields_2_epochs(self) -> None:
        df = self._make_session_df(int(WAUC_NATIVE_SFREQ * 8))
        epochs = _session_to_epochs(
            df,
            channel_columns=list(WAUC_EEG_CHANNELS),
            native_sfreq=WAUC_NATIVE_SFREQ,
            target_sfreq=250.0,
            epoch_seconds=4.0,
        )
        # 8 s / 4 s = 2 epochs; resampled 500 → 250 Hz → 1000 samples/epoch.
        assert epochs.shape == (2, len(WAUC_EEG_CHANNELS), 1000)

    def test_no_resample_passthrough(self) -> None:
        df = self._make_session_df(int(WAUC_NATIVE_SFREQ * 8))
        epochs = _session_to_epochs(
            df,
            channel_columns=list(WAUC_EEG_CHANNELS),
            native_sfreq=WAUC_NATIVE_SFREQ,
            target_sfreq=WAUC_NATIVE_SFREQ,
            epoch_seconds=4.0,
        )
        assert epochs.shape == (2, len(WAUC_EEG_CHANNELS), 2000)

    def test_too_short_raises(self) -> None:
        df = self._make_session_df(50)
        with pytest.raises(ValueError, match="too short"):
            _session_to_epochs(
                df,
                channel_columns=list(WAUC_EEG_CHANNELS),
                native_sfreq=WAUC_NATIVE_SFREQ,
                target_sfreq=250.0,
                epoch_seconds=4.0,
            )

    def test_drops_nan_contaminated_epochs(self) -> None:
        # 8 s × 500 Hz = 4000 samples = 2 raw epochs of 4 s. Mark the
        # second epoch (samples 2000+) NaN on a single channel; the
        # NaN-drop should leave 1 surviving epoch.
        df = self._make_session_df(int(WAUC_NATIVE_SFREQ * 8))
        df.loc[2000:, WAUC_EEG_CHANNELS[0]] = np.nan
        epochs = _session_to_epochs(
            df,
            channel_columns=list(WAUC_EEG_CHANNELS),
            native_sfreq=WAUC_NATIVE_SFREQ,
            target_sfreq=WAUC_NATIVE_SFREQ,
            epoch_seconds=4.0,
        )
        assert epochs.shape == (1, len(WAUC_EEG_CHANNELS), 2000)
        assert not np.isnan(epochs).any()

    def test_all_nan_session_returns_empty_array(self) -> None:
        df = self._make_session_df(int(WAUC_NATIVE_SFREQ * 8))
        df.loc[:, list(WAUC_EEG_CHANNELS)] = np.nan
        epochs = _session_to_epochs(
            df,
            channel_columns=list(WAUC_EEG_CHANNELS),
            native_sfreq=WAUC_NATIVE_SFREQ,
            target_sfreq=WAUC_NATIVE_SFREQ,
            epoch_seconds=4.0,
        )
        assert epochs.shape == (0, len(WAUC_EEG_CHANNELS), 2000)


def _write_fake_wauc_layout(
    root: Path,
    subjects: list[int],
    labels: dict[tuple[int, int], tuple[int, int]],
    samples_per_session: int = int(WAUC_NATIVE_SFREQ * 8),
) -> None:
    """Build a synthetic WAUC tree under ``root`` for end-to-end tests.

    Parameters
    ----------
    subjects : filesystem subject IDs (1..48) to materialise.
    labels   : ``{(partid, session_no): (mw_label_int, pw_label_int)}``.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    rows = [
        {
            "Participant ID": partid,
            "session_no": sess,
            "mw_labels": float(mw),
            "pw_labels": float(pw),
        }
        for (partid, sess), (mw, pw) in labels.items()
    ]
    pd.DataFrame(rows).to_csv(root / "subjective_ratings_with_labels.csv", index=False)

    pd.DataFrame({"partId": [subject_id_to_partid(s) for s in subjects], "age": [25] * len(subjects)}).to_csv(
        root / "demographics.csv", index=False
    )

    for sid in subjects:
        partid = subject_id_to_partid(sid)
        subject_dir = root / "process" / f"S{sid:02d}"
        subject_dir.mkdir(parents=True, exist_ok=True)

        blocks: list[pd.DataFrame] = []
        for sess in WAUC_SESSIONS:
            if (partid, sess) not in labels:
                continue
            n = samples_per_session
            block: dict[str, np.ndarray | list] = {
                ch: rng.standard_normal(n) for ch in WAUC_EEG_CHANNELS
            }
            block["fs"] = np.full(n, WAUC_NATIVE_SFREQ)
            block["info"] = ["session"] * n
            block["session_no"] = [sess] * n
            blocks.append(pd.DataFrame(block))

        if blocks:
            pd.concat(blocks, axis=0, ignore_index=True).to_csv(
                subject_dir / "enobio_eeg_asr.csv", index=False
            )


class TestLoadWauc:
    """End-to-end loader test against a synthetic WAUC layout."""

    def test_returns_subject_data_per_subject(self, tmp_path: Path) -> None:
        subjects = [1, 2]
        labels = {
            (subject_id_to_partid(sid), sess): (sess % 2, (sess - 1) // 2)
            for sid in subjects for sess in WAUC_SESSIONS
        }
        _write_fake_wauc_layout(tmp_path, subjects, labels)
        out = load_wauc(data_root=tmp_path)
        assert len(out) == 2
        for s in out:
            assert s.dataset_name == "WAUC"
            assert s.sfreq == 250.0
            assert s.n_channels == len(WAUC_EEG_CHANNELS)
            assert s.n_trials == len(WAUC_SESSIONS) * 2  # 6 sessions × 2 epochs

    def test_label_assignment_matches_csv(self, tmp_path: Path) -> None:
        partid = 1001
        labels = {
            (partid, 1): (0, 0),  # low MW, no physical
            (partid, 2): (1, 0),  # high MW, no physical
            (partid, 3): (0, 1),  # low, medium
            (partid, 4): (1, 1),
            (partid, 5): (0, 2),
            (partid, 6): (1, 2),
        }
        _write_fake_wauc_layout(tmp_path, [1], labels)
        out = load_wauc(data_root=tmp_path)
        assert len(out) == 1
        s = out[0]
        # Every epoch's label and pw_label covariate must match the labels CSV.
        for sess_str, run_str, mw_label in zip(s.metadata["session"], s.metadata["run"], s.y):
            sess = int(sess_str)
            expected_mw, expected_pw = labels[(partid, sess)]
            expected_mw_str = "low" if expected_mw == 0 else "high"
            assert mw_label == expected_mw_str
            assert int(run_str) == expected_pw

    def test_subject_1028_skipped(self, tmp_path: Path) -> None:
        # Subject 1028 (filesystem S28, partid 1028) is in the loader's
        # drop list. Even if we write a labels row for it the loader must
        # skip it.
        labels = {
            (1001, 1): (0, 0),
            (1028, 1): (1, 0),
        }
        _write_fake_wauc_layout(tmp_path, [1, 28], labels)
        out = load_wauc(data_root=tmp_path)
        assert {s.subject for s in out} == {1}

    def test_subject_1020_kept(self, tmp_path: Path) -> None:
        # Despite the upstream README's "no data" flag, 1020 has ratings
        # and EEG in the processed release; the loader keeps it.
        labels = {(1020, 1): (0, 0)}
        _write_fake_wauc_layout(tmp_path, [20], labels)
        out = load_wauc(data_root=tmp_path)
        assert {s.subject for s in out} == {20}

    def test_subjects_filter_passthrough(self, tmp_path: Path) -> None:
        subjects = [1, 2, 3]
        labels = {
            (subject_id_to_partid(sid), sess): (0, 0)
            for sid in subjects for sess in WAUC_SESSIONS
        }
        _write_fake_wauc_layout(tmp_path, subjects, labels)
        out = load_wauc(subjects=[1, 3], data_root=tmp_path)
        assert {s.subject for s in out} == {1, 3}
