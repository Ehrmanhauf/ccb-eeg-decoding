"""Constrained Contextual Bandit (CCB) — Formulation A implementation.

See `design-doc/ccb-formulation.md` §6–7 for the formal specification and
`design-doc/open-justifications.md` for tracked hyperparameter decisions.

This package implements Phase 3 of the thesis. The CCB operates on BCI-IV-2b
(3-channel bipolar MI) only; any attempt to import from 2a is a leakage bug.
"""

from thesis.ccb.arms import (
    Arm,
    ArmHead,
    arm_cost,
    build_arm_heads,
    enumerate_arms_2b,
    prune_arms,
)
from thesis.ccb.context import N_ARM_FAMILIES, compute_context, context_dim
from thesis.ccb.oplb import INFEASIBLE, OPLB, OPLBConfig, default_alpha, make_psi

__all__ = [
    "INFEASIBLE",
    "N_ARM_FAMILIES",
    "OPLB",
    "Arm",
    "ArmHead",
    "OPLBConfig",
    "arm_cost",
    "build_arm_heads",
    "compute_context",
    "context_dim",
    "default_alpha",
    "enumerate_arms_2b",
    "make_psi",
    "prune_arms",
]
