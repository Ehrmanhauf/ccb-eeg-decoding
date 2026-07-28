"""Single source of truth for matched-conditions train/test splits.

Every method family in this thesis --- the CCB, the classical battery (B1--B5),
and the EEGNet comparator --- must draw both its fold indices *and* its post-cap
training subsample from THIS module, so that every per-``(dataset, protocol)``
comparison is apples-to-apples *by construction*. The thesis's contribution is a
matched-conditions comparison (``CLAUDE.md`` §"matched conditions"); a method
that quietly trains on a different pool (the EEGNet-LOSO-on-full-data bug this
module was written to kill) silently invalidates the comparison.

The discipline this module enforces:

- **Identical splits.** All methods obtain ``(train_idx, test_idx)`` for a cell
  from the same wrapper here, which delegates to ``thesis.protocols``.
- **Identical training pool.** Large LOSO pools (WAUC pools ~34k epochs across
  44 subjects) are capped to a single common size via a *class-stratified*
  subsample drawn with a *fixed* seed --- so the capped pool is byte-identical
  across methods, independent of each method's own algorithm seed (EEGNet weight
  init, CCB exploration). Within-subject CV, the official session split, and the
  cross-session split use per-subject / per-session pools small enough to need no
  cap.
- **Separation of seeds.** The *split / subsample* seed is fixed (``SPLIT_SEED``);
  only the *algorithm* seed varies across reruns. This keeps the data identical
  while still letting stochastic methods report mean +/- std.

The guardrail ``tests/test_matched_conditions.py`` asserts the wrappers here are
deterministic and produce identical pools across the three method profiles.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from thesis.data import SubjectData
from thesis.protocols import (
    Split,
    cross_session_split,
    leave_one_subject_out,
    session_split,
    within_subject_cv,
)

# --- Locked matched-conditions constants -------------------------------------

#: Fixed seed for the *split / subsample* (NOT the algorithm seed). Held constant
#: across every method and every rerun so the training pool is identical.
SPLIT_SEED: int = 42

#: Default per-fold training-pool cap (class-stratified subsample of ``train_idx``)
#: keyed by protocol. ``0`` means uncapped. The LOSO cap of 4000 is the matched
#: value: it was previously Fixed=4000 / EEGNet=full(~34k) / CCB=800 --- the
#: mismatch this module removes. 4000 keeps the 9-band FBCSP fit and EEGNet
#: training tractable while giving every method the same data (cf. the cap
#: rationale already in ``scripts/run_loso_wauc.py``). Within / official /
#: cross-session pools are small enough to use in full.
DEFAULT_TRAIN_CAP: dict[str, int] = {
    "within": 0,
    "official": 0,
    "cross_session": 0,
    "loso": 4000,
}


def _cap(protocol: str, override: int | None) -> int:
    if override is not None:
        return override
    try:
        return DEFAULT_TRAIN_CAP[protocol]
    except KeyError as exc:  # pragma: no cover - programmer error
        raise ValueError(
            f"unknown protocol {protocol!r}; expected one of {sorted(DEFAULT_TRAIN_CAP)}"
        ) from exc


def subsample_train(
    split: Split, y_all: np.ndarray, cap: int, *, seed: int = SPLIT_SEED
) -> Split:
    """Class-stratified subsample of ``split.train_idx`` down to ``cap`` trials.

    ``test_idx`` is never touched (the evaluation set is identical for all
    methods). ``cap <= 0`` or a pool already at/under ``cap`` returns the split
    unchanged. The subsample is drawn with a *fixed* ``seed`` (default
    :data:`SPLIT_SEED`) so the capped pool is identical across methods regardless
    of each method's algorithm seed. Lifted from the original per-runner helper in
    ``scripts/run_loso_wauc.py`` so it is shared, not copied.
    """
    train_idx = split.train_idx
    if cap <= 0 or len(train_idx) <= cap:
        return split
    rng = np.random.default_rng(seed)
    y_tr = y_all[train_idx]
    classes, counts = np.unique(y_tr, return_counts=True)
    picked: list[np.ndarray] = []
    for c, cnt in zip(classes, counts, strict=True):
        k = max(1, int(round(cap * cnt / len(train_idx))))
        c_idx = train_idx[y_tr == c]
        picked.append(rng.choice(c_idx, size=min(k, len(c_idx)), replace=False))
    sub = np.sort(np.concatenate(picked))
    return Split(train_idx=sub, test_idx=split.test_idx, name=split.name)


# --- Matched protocol wrappers -----------------------------------------------
# Each wrapper yields/returns the split a method MUST use for that cell. Because
# every method calls the same wrapper with the same protocol cap and SPLIT_SEED,
# the training pool is identical by construction.


def matched_within_cv(
    data: SubjectData,
    *,
    n_splits: int = 5,
    fold_seed: int = SPLIT_SEED,
    cap: int | None = None,
) -> Iterator[Split]:
    """Within-subject K-fold CV with the shared fold seed (uncapped by default)."""
    c = _cap("within", cap)
    for sp in within_subject_cv(data, n_splits=n_splits, seed=fold_seed):
        yield subsample_train(sp, data.y, c)


def matched_session_split(
    data: SubjectData,
    *,
    train_session_idx: int = 0,
    test_session_idx: int = 1,
    cap: int | None = None,
) -> Split:
    """Official session split (uncapped by default); deterministic for all methods."""
    sp = session_split(
        data, train_session_idx=train_session_idx, test_session_idx=test_session_idx
    )
    return subsample_train(sp, data.y, _cap("official", cap))


def matched_cross_session(
    data: SubjectData,
    *,
    train_sessions: Sequence[str],
    test_sessions: Sequence[str],
    cap: int | None = None,
) -> Split:
    """Cross-session drift split (uncapped by default); deterministic for all methods."""
    sp = cross_session_split(
        data, train_sessions=train_sessions, test_sessions=test_sessions
    )
    return subsample_train(sp, data.y, _cap("cross_session", cap))


def matched_loso(
    data_list: list[SubjectData], *, cap: int | None = None
) -> Iterator[tuple[SubjectData, Split]]:
    """LOSO folds with the common matched training cap (default 4000) for every method.

    Yields ``(pooled_data, capped_split)`` per held-out subject. The cap is applied
    with the fixed :data:`SPLIT_SEED`, so Fixed, CCB, and EEGNet all train on the
    identical class-stratified subsample of the held-in pool.
    """
    c = _cap("loso", cap)
    for fold_data, sp in leave_one_subject_out(data_list):
        yield fold_data, subsample_train(sp, fold_data.y, c)
