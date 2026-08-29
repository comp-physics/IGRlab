#!/usr/bin/env python3
"""Run the production HDG CUDA accepted-step paired performance campaign.

This driver deliberately does not configure, build, acquire a GPU, or change
GPU settings.  Invoke it around an already-built benchmark while holding the
single external exclusive GPU lease required by the campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, NamedTuple, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CONTROL_COMMIT = "30afa08f8ad19a3ad37e627463ef63c40da1ae91"
FROZEN_CONTROL_TREE = "03313cf6b5b17d05fc3d768a6c77770295ced343"
TRACE_PRECONDITIONER_BY_POLICY_ENGINE_AND_MODE = {
    ("engine-default", "ordinary", "conventional"): "hypre_ilu0",
    ("engine-default", "ordinary", "igr"): "hypre_ilu0",
    ("engine-default", "tensor", "conventional"): "hypre_diag_scale",
    ("common-fast", "ordinary", "conventional"): "hypre_diag_scale",
    ("common-fast", "ordinary", "igr"): "hypre_ilu0",
    ("common-fast", "tensor", "conventional"): "hypre_diag_scale",
    ("common-fast", "tensor", "igr"): "hypre_ilu0",
    ("equilibrated-mgr", "tensor", "igr"): "hypre_mgr_field_split",
    ("native-element-schwarz-fp64", "tensor", "igr"):
        "native_element_schwarz_fp64",
    ("native-element-schwarz-matrix-free-fp64", "tensor", "igr"):
        "native_element_schwarz_fp64",
    ("native-element-schwarz-equilibrated-field-split-fp64", "tensor", "igr"):
        "native_element_schwarz_equilibrated_field_split_fp64",
    ("native-element-schwarz-matrix-free-equilibrated-field-split-fp64",
     "tensor", "igr"):
        "native_element_schwarz_equilibrated_field_split_fp64",
    ("native-element-schwarz-matrix-free-warp-equilibrated-field-split-fp64",
     "tensor", "igr"):
        "native_element_schwarz_warp_equilibrated_field_split_fp64",
    ("native-element-schwarz-matrix-free-equilibrated-field-split-"
     "sigma-jacobi3-fp64", "tensor", "igr"):
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi3_fp64",
    ("native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
     "sigma-jacobi3-fp64", "tensor", "igr"):
        "native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64",
}
TRACE_ILU_POLICY_BY_POLICY_ENGINE_AND_MODE = {
    ("engine-default", "ordinary", "conventional"): (0, 0, 0, 1),
    ("engine-default", "ordinary", "igr"): (0, 0, 1, 1),
    ("engine-default", "tensor", "conventional"): (-1, -1, -1, 0),
    ("engine-default", "tensor", "igr"): (-1, -1, -1, 0),
    ("common-fast", "ordinary", "conventional"): (-1, -1, -1, 0),
    ("common-fast", "ordinary", "igr"): (0, 0, 1, 1),
    ("common-fast", "tensor", "conventional"): (-1, -1, -1, 0),
    ("common-fast", "tensor", "igr"): (0, 0, 1, 1),
    ("equilibrated-mgr", "tensor", "igr"): (-1, -1, -1, 0),
    ("native-element-schwarz-fp64", "tensor", "igr"):
        (-1, -1, -1, 0),
    ("native-element-schwarz-matrix-free-fp64", "tensor", "igr"):
        (-1, -1, -1, 0),
    ("native-element-schwarz-equilibrated-field-split-fp64", "tensor", "igr"):
        (-1, -1, -1, 0),
    ("native-element-schwarz-matrix-free-equilibrated-field-split-fp64",
     "tensor", "igr"):
        (-1, -1, -1, 0),
    ("native-element-schwarz-matrix-free-warp-equilibrated-field-split-fp64",
     "tensor", "igr"):
        (-1, -1, -1, 0),
    ("native-element-schwarz-matrix-free-equilibrated-field-split-"
     "sigma-jacobi3-fp64", "tensor", "igr"):
        (-1, -1, -1, 0),
    ("native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
     "sigma-jacobi3-fp64", "tensor", "igr"):
        (-1, -1, -1, 0),
}
TRACE_ILU_FACTOR_REUSE_BY_POLICY_ENGINE_AND_MODE = {
    ("engine-default", "ordinary", "conventional"):
        "refresh_each_trace_solve",
    ("engine-default", "ordinary", "igr"): "refresh_each_trace_solve",
    ("engine-default", "tensor", "conventional"): "not_applicable",
    ("engine-default", "tensor", "igr"): "not_applicable",
    ("common-fast", "ordinary", "conventional"): "not_applicable",
    ("common-fast", "ordinary", "igr"): "first_trace_solve_per_attempt",
    ("common-fast", "tensor", "conventional"): "not_applicable",
    ("common-fast", "tensor", "igr"): "first_trace_solve_per_attempt",
    ("equilibrated-mgr", "tensor", "igr"): "not_applicable",
    ("native-element-schwarz-fp64", "tensor", "igr"):
        "not_applicable",
    ("native-element-schwarz-matrix-free-fp64", "tensor", "igr"):
        "not_applicable",
    ("native-element-schwarz-equilibrated-field-split-fp64", "tensor", "igr"):
        "not_applicable",
    ("native-element-schwarz-matrix-free-equilibrated-field-split-fp64",
     "tensor", "igr"):
        "not_applicable",
    ("native-element-schwarz-matrix-free-warp-equilibrated-field-split-fp64",
     "tensor", "igr"):
        "not_applicable",
    ("native-element-schwarz-matrix-free-equilibrated-field-split-"
     "sigma-jacobi3-fp64", "tensor", "igr"):
        "not_applicable",
    ("native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
     "sigma-jacobi3-fp64", "tensor", "igr"):
        "not_applicable",
}
TRACE_ILU_TRIANGULAR_SOLVE_BY_POLICY_ENGINE_AND_MODE = {
    ("engine-default", "ordinary", "conventional"): ("direct", 0, 0, 0),
    ("engine-default", "ordinary", "igr"): ("direct", 0, 0, 0),
    ("engine-default", "tensor", "conventional"):
        ("not_applicable", 0, 0, 0),
    ("engine-default", "tensor", "igr"):
        ("not_applicable", 0, 0, 0),
    ("common-fast", "ordinary", "conventional"):
        ("not_applicable", 0, 0, 0),
    ("common-fast", "ordinary", "igr"):
        ("iterative_jacobi", 5, 5, 8),
    ("common-fast", "tensor", "conventional"):
        ("not_applicable", 0, 0, 0),
    ("common-fast", "tensor", "igr"):
        ("iterative_jacobi", 5, 5, 8),
    ("equilibrated-mgr", "tensor", "igr"):
        ("not_applicable", 0, 0, 0),
    ("native-element-schwarz-fp64", "tensor", "igr"):
        ("not_applicable", 0, 0, 0),
    ("native-element-schwarz-matrix-free-fp64", "tensor", "igr"):
        ("not_applicable", 0, 0, 0),
    ("native-element-schwarz-equilibrated-field-split-fp64", "tensor", "igr"):
        ("not_applicable", 0, 0, 0),
    ("native-element-schwarz-matrix-free-equilibrated-field-split-fp64",
     "tensor", "igr"):
        ("not_applicable", 0, 0, 0),
    ("native-element-schwarz-matrix-free-warp-equilibrated-field-split-fp64",
     "tensor", "igr"):
        ("not_applicable", 0, 0, 0),
    ("native-element-schwarz-matrix-free-equilibrated-field-split-"
     "sigma-jacobi3-fp64", "tensor", "igr"):
        ("not_applicable", 0, 0, 0),
    ("native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
     "sigma-jacobi3-fp64", "tensor", "igr"):
        ("not_applicable", 0, 0, 0),
}
# Solver name and the three mutually exclusive rank-one/rank-two
# configuration predicates emitted in each bounded sample.  Keeping this
# explicit prevents the MGR experiment from being mistaken for the native
# tensor trace solver merely because both select the tensor CUDA engine.
TRACE_SOLVER_CONFIGURATION_BY_POLICY_ENGINE_AND_MODE = {
    ("engine-default", "ordinary", "conventional"):
        ("hypre-parcsr-gmres", False, False, False),
    ("engine-default", "ordinary", "igr"):
        ("hypre-parcsr-gmres", False, False, False),
    ("engine-default", "tensor", "conventional"):
        ("native-rank1-fp64", True, False, False),
    ("engine-default", "tensor", "igr"):
        ("native-rank1-fp64", True, False, False),
    ("common-fast", "ordinary", "conventional"):
        ("hypre-parcsr-gmres", False, False, False),
    ("common-fast", "ordinary", "igr"):
        ("hypre-parcsr-gmres", False, False, False),
    ("common-fast", "tensor", "conventional"):
        ("native-rank1-fp64", True, False, False),
    ("common-fast", "tensor", "igr"):
        ("native-rank1-fp64", True, False, False),
    ("equilibrated-mgr", "tensor", "igr"):
        ("hypre-fgmres-mgr-rank1", False, True, False),
    ("native-element-schwarz-fp64", "tensor", "igr"):
        ("native-rank1-fp64", True, False, False),
    ("native-element-schwarz-matrix-free-fp64", "tensor", "igr"):
        ("native-rank1-fp64", True, False, False),
    ("native-element-schwarz-equilibrated-field-split-fp64", "tensor", "igr"):
        ("native-rank1-fp64", True, False, False),
    ("native-element-schwarz-matrix-free-equilibrated-field-split-fp64",
     "tensor", "igr"):
        ("native-rank1-fp64", True, False, False),
    ("native-element-schwarz-matrix-free-warp-equilibrated-field-split-fp64",
     "tensor", "igr"):
        ("native-rank1-fp64", True, False, False),
    ("native-element-schwarz-matrix-free-equilibrated-field-split-"
     "sigma-jacobi3-fp64", "tensor", "igr"):
        ("native-rank1-fp64", True, False, False),
    ("native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
     "sigma-jacobi3-fp64", "tensor", "igr"):
        ("native-rank1-fp64", True, False, False),
}


class FixedSigmaJacobiPolicy(NamedTuple):
    sweeps: int
    preconditioner: str
    warp: bool


NO_FIXED_SIGMA_JACOBI_POLICY = FixedSigmaJacobiPolicy(0, "", False)
FIXED_SIGMA_JACOBI_POLICY = {
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi1-fp64": FixedSigmaJacobiPolicy(
        1,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi1_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi1-fp64": FixedSigmaJacobiPolicy(
        1,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi1_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi2-fp64": FixedSigmaJacobiPolicy(
        2,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi2_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi2-fp64": FixedSigmaJacobiPolicy(
        2,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi2_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi3-fp64": FixedSigmaJacobiPolicy(
        3,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi3_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi3-fp64": FixedSigmaJacobiPolicy(
        3,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi3_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi4-fp64": FixedSigmaJacobiPolicy(
        4,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi4_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi4-fp64": FixedSigmaJacobiPolicy(
        4,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi4_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi5-fp64": FixedSigmaJacobiPolicy(
        5,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi5_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi5-fp64": FixedSigmaJacobiPolicy(
        5,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi5_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi6-fp64": FixedSigmaJacobiPolicy(
        6,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi6_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi6-fp64": FixedSigmaJacobiPolicy(
        6,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi6_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi7-fp64": FixedSigmaJacobiPolicy(
        7,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi7_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi7-fp64": FixedSigmaJacobiPolicy(
        7,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi7_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi8-fp64": FixedSigmaJacobiPolicy(
        8,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi8_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi8-fp64": FixedSigmaJacobiPolicy(
        8,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi8_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi9-fp64": FixedSigmaJacobiPolicy(
        9,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi9_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi9-fp64": FixedSigmaJacobiPolicy(
        9,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi9_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi10-fp64": FixedSigmaJacobiPolicy(
        10,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi10_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi10-fp64": FixedSigmaJacobiPolicy(
        10,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi10_fp64",
        True,
    ),
}
PRODUCTION_TENSOR_IGR_TRACE_POLICY_BY_ORDER = {
    1: (
        "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
        "sigma-jacobi10-fp64"
    ),
    2: (
        "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
        "sigma-jacobi10-fp64"
    ),
    3: (
        "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
        "sigma-jacobi2-fp64"
    ),
}
for _fixed_policy, (_fixed_sweeps, _fixed_preconditioner, _fixed_warp) in \
        FIXED_SIGMA_JACOBI_POLICY.items():
    _fixed_key = (_fixed_policy, "tensor", "igr")
    TRACE_PRECONDITIONER_BY_POLICY_ENGINE_AND_MODE[_fixed_key] = \
        _fixed_preconditioner
    TRACE_ILU_POLICY_BY_POLICY_ENGINE_AND_MODE[_fixed_key] = (-1, -1, -1, 0)
    TRACE_ILU_FACTOR_REUSE_BY_POLICY_ENGINE_AND_MODE[_fixed_key] = \
        "not_applicable"
    TRACE_ILU_TRIANGULAR_SOLVE_BY_POLICY_ENGINE_AND_MODE[_fixed_key] = \
        ("not_applicable", 0, 0, 0)
    TRACE_SOLVER_CONFIGURATION_BY_POLICY_ENGINE_AND_MODE[_fixed_key] = \
        ("native-rank1-fp64", True, False, False)


def fixed_sigma_jacobi_policy(trace_policy: str) -> FixedSigmaJacobiPolicy:
    return FIXED_SIGMA_JACOBI_POLICY.get(
        trace_policy, NO_FIXED_SIGMA_JACOBI_POLICY
    )


def production_tensor_igr_trace_policy(
        order: int | None) -> tuple[str, FixedSigmaJacobiPolicy]:
    """Return the exact rank-one engine-default winner for one p1--p3 cell."""
    if isinstance(order, bool):
        raise CampaignError("production tensor-IGR policy order is invalid")
    try:
        policy = PRODUCTION_TENSOR_IGR_TRACE_POLICY_BY_ORDER[order]
    except (KeyError, TypeError) as error:
        raise CampaignError(
            "production tensor-IGR policy requires polynomial order 1, 2, or 3"
        ) from error
    descriptor = fixed_sigma_jacobi_policy(policy)
    if descriptor.sweeps <= 0 or not descriptor.warp:
        raise CampaignError("production tensor-IGR policy table is inconsistent")
    return policy, descriptor


SAME_POLICY_MEMBERS = {
    "O_base": {"engine": "ordinary", "trace_policy": "engine-default"},
    "O_common": {"engine": "ordinary", "trace_policy": "common-fast"},
    "T_common": {"engine": "tensor", "trace_policy": "common-fast"},
}
SAME_POLICY_MEMBER_ORDER = tuple(SAME_POLICY_MEMBERS)
FINAL_BENCHMARK_NAME = "dg_igr_hdg_cuda_accepted_step"
FINAL_WARMUP_STEPS = 1
FINAL_MEASURED_STEPS = 5
UINT64_MAX = (1 << 64) - 1
FINAL_GPU_COMPUTE_CAPABILITY = 120
FROZEN_OVERLAY_PATH = Path(
    "tests/hdg/runtime/cuda/benchmark_generic_euler_accepted_step.cu"
)
FROZEN_OVERLAY_DIFF_SHA256 = (
    "534bcded1f38bf5a5d6dc3cf2c7f58400243c6d596e3b132ff5f9b9123136e25"
)
BENCHMARK_BUILD_RELATIVE_PATH = Path(
    "tests/hdg/runtime/cuda/"
    "dg_igr_hdg_runtime_cuda_accepted_step_benchmark"
)
REQUIRED_HARDWARE_IDENTITY_FIELDS = (
    "index", "uuid", "name", "driver_version", "compute_cap"
)
CELLS = tuple((mode, order) for mode in ("conventional", "igr")
              for order in (1, 2, 3))
QUALIFICATION_PROFILES = {
    "saturated": {
        "workload_label": "sm120-mesh52-saturated",
        "mesh_n": 52,
        "domain_periods": 1,
        "dt": 1.0e-5,
        "export_control_mode": "igr",
        "export_control_order": 3,
        "pairs": 6,
    },
    "break-even": {
        "workload_label": "sm120-mesh4-break-even",
        "mesh_n": 4,
        "domain_periods": 1,
        "dt": 1.0e-5,
        "export_control_mode": "igr",
        "export_control_order": 3,
        "pairs": 6,
    },
}
NVIDIA_SMI_FIELDS = (
    "index",
    "uuid",
    "name",
    "pci.bus_id",
    "driver_version",
    "compute_cap",
    "memory.total",
    "timestamp",
    "pstate",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks.max.sm",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "utilization.gpu",
    "utilization.memory",
    "clocks_event_reasons.active",
    "clocks_throttle_reasons.active",
)


class CampaignError(RuntimeError):
    """An evidence-invalidating campaign failure."""


def expected_resolved_trace_policy(
        engine: str, mode: str,
        requested_trace_policy: str = "engine-default", *,
        order: int | None = None) -> str:
    """Return the deterministic rank-one production policy resolution."""
    if requested_trace_policy != "engine-default":
        return requested_trace_policy
    if engine == "tensor" and mode == "igr":
        return production_tensor_igr_trace_policy(order)[0]
    return "engine-default"


def expected_trace_preconditioner(
        engine: str, mode: str,
        trace_policy: str = "engine-default", *,
        order: int | None = None) -> str:
    """Return the exact typed engine-and-mode trace policy."""
    if trace_policy == "engine-default" and engine == "tensor" and \
            mode == "igr":
        return production_tensor_igr_trace_policy(order)[1].preconditioner
    try:
        return TRACE_PRECONDITIONER_BY_POLICY_ENGINE_AND_MODE[
            (trace_policy, engine, mode)
        ]
    except KeyError as error:
        raise CampaignError(
            f"unsupported CUDA engine/mode preconditioner policy: "
            f"{trace_policy!r}/{engine!r}/{mode!r}"
        ) from error


def expected_trace_ilu_policy(
        engine: str, mode: str,
        trace_policy: str = "engine-default") -> tuple[int, int, int, int]:
    """Return type, fill, local reordering, and application iterations."""
    try:
        return TRACE_ILU_POLICY_BY_POLICY_ENGINE_AND_MODE[
            (trace_policy, engine, mode)
        ]
    except KeyError as error:
        raise CampaignError(
            "unsupported trace-policy/engine/mode ILU policy: "
            f"{trace_policy!r}/{engine!r}/{mode!r}"
        ) from error


def expected_trace_ilu_factor_reuse_policy(
        engine: str, mode: str,
        trace_policy: str = "engine-default") -> str:
    """Return the exact aggregate-only ILU factor lifetime policy."""
    try:
        return TRACE_ILU_FACTOR_REUSE_BY_POLICY_ENGINE_AND_MODE[
            (trace_policy, engine, mode)
        ]
    except KeyError as error:
        raise CampaignError(
            "unsupported CUDA engine/mode ILU factor-reuse policy: "
            f"{trace_policy!r}/{engine!r}/{mode!r}"
        ) from error


def expected_trace_ilu_triangular_solve_policy(
        engine: str, mode: str,
        trace_policy: str = "engine-default") -> tuple[str, int, int, int]:
    """Return kind, side iteration counts, and factor-SpMV updates/apply."""
    try:
        return TRACE_ILU_TRIANGULAR_SOLVE_BY_POLICY_ENGINE_AND_MODE[
            (trace_policy, engine, mode)
        ]
    except KeyError as error:
        raise CampaignError(
            "unsupported CUDA engine/mode ILU triangular-solve policy: "
            f"{trace_policy!r}/{engine!r}/{mode!r}"
        ) from error


def expected_trace_solver_configuration(
        engine: str, mode: str,
        trace_policy: str = "engine-default") -> tuple[str, bool, bool, bool]:
    """Return solver name plus native/MGR/rank-two configuration flags."""
    try:
        return TRACE_SOLVER_CONFIGURATION_BY_POLICY_ENGINE_AND_MODE[
            (trace_policy, engine, mode)
        ]
    except KeyError as error:
        raise CampaignError(
            "unsupported trace-policy/engine/mode solver configuration: "
            f"{trace_policy!r}/{engine!r}/{mode!r}"
        ) from error


def tensor_comparison_attribution(mode: str, order: int) -> dict[str, Any]:
    """Describe the policy scope of an ordinary-versus-tensor speedup."""
    ordinary_preconditioner = expected_trace_preconditioner(
        "ordinary", mode, order=order
    )
    tensor_preconditioner = expected_trace_preconditioner(
        "tensor", mode, order=order
    )
    ordinary_ilu = expected_trace_ilu_policy("ordinary", mode)
    tensor_ilu = expected_trace_ilu_policy("tensor", mode)
    ordinary_reuse = expected_trace_ilu_factor_reuse_policy("ordinary", mode)
    tensor_reuse = expected_trace_ilu_factor_reuse_policy("tensor", mode)
    ordinary_triangular = expected_trace_ilu_triangular_solve_policy(
        "ordinary", mode
    )
    tensor_triangular = expected_trace_ilu_triangular_solve_policy(
        "tensor", mode
    )
    configuration_matched = (
        ordinary_preconditioner == tensor_preconditioner and
        ordinary_ilu == tensor_ilu and
        ordinary_triangular == tensor_triangular
    )
    return {
        "scope": "complete_engine_package",
        "polynomial_order": order,
        "ordinary_trace_policy": "engine-default",
        "tensor_trace_policy": "engine-default",
        "ordinary_trace_preconditioner": ordinary_preconditioner,
        "tensor_trace_preconditioner": tensor_preconditioner,
        "ordinary_ilu_factor_reuse_policy": ordinary_reuse,
        "tensor_ilu_factor_reuse_policy": tensor_reuse,
        "ordinary_ilu_triangular_solve_policy": {
            "kind": ordinary_triangular[0],
            "lower_jacobi_iterations": ordinary_triangular[1],
            "upper_jacobi_iterations": ordinary_triangular[2],
            "jacobi_factor_spmv_updates_per_application": ordinary_triangular[3],
        },
        "tensor_ilu_triangular_solve_policy": {
            "kind": tensor_triangular[0],
            "lower_jacobi_iterations": tensor_triangular[1],
            "upper_jacobi_iterations": tensor_triangular[2],
            "jacobi_factor_spmv_updates_per_application": tensor_triangular[3],
        },
        "trace_preconditioner_configuration_matched": configuration_matched,
        "trace_policy_matched": configuration_matched and
        ordinary_reuse == tensor_reuse,
        "includes_tensor_condensation": True,
        "mma_contribution_requires_separate_profiler_attribution": True,
    }


def trace_policy_configuration(
        engine: str, mode: str, trace_policy: str, *,
        order: int) -> dict[str, Any]:
    """Return the exact policy dimensions required from one benchmark run."""
    ilu = expected_trace_ilu_policy(engine, mode, trace_policy)
    triangular = expected_trace_ilu_triangular_solve_policy(
        engine, mode, trace_policy
    )
    resolved_trace_policy = expected_resolved_trace_policy(
        engine, mode, trace_policy, order=order
    )
    return {
        "trace_policy": trace_policy,
        "requested_trace_policy": trace_policy,
        "resolved_trace_policy": resolved_trace_policy,
        "trace_preconditioner": expected_trace_preconditioner(
            engine, mode, trace_policy, order=order
        ),
        "ilu_type": ilu[0],
        "ilu_level_of_fill": ilu[1],
        "ilu_local_reordering": ilu[2],
        "ilu_application_iterations": ilu[3],
        "ilu_triangular_solve_kind": triangular[0],
        "ilu_lower_jacobi_iterations": triangular[1],
        "ilu_upper_jacobi_iterations": triangular[2],
        "ilu_accounting_kind": (
            "derived_bounds_v1" if ilu[3] else "unavailable"
        ),
        "ilu_factor_reuse_policy":
            expected_trace_ilu_factor_reuse_policy(
                engine, mode, trace_policy
            ),
    }


def record_trace_policy_configuration(record: dict[str, Any]) -> dict[str, Any]:
    """Extract bounded top-level policy metadata, never solver event history."""
    return {
        "trace_policy": record.get("trace_policy"),
        "requested_trace_policy": record.get("requested_trace_policy"),
        "resolved_trace_policy": record.get("resolved_trace_policy"),
        "trace_preconditioner": record.get("trace_preconditioner"),
        "ilu_type": record.get("trace_ilu_type"),
        "ilu_level_of_fill": record.get("trace_ilu_level_of_fill"),
        "ilu_local_reordering": record.get("trace_ilu_local_reordering"),
        "ilu_application_iterations": record.get(
            "trace_ilu_application_iterations"
        ),
        "ilu_triangular_solve_kind": record.get(
            "trace_ilu_triangular_solve_kind"
        ),
        "ilu_lower_jacobi_iterations": record.get(
            "trace_ilu_lower_jacobi_iterations"
        ),
        "ilu_upper_jacobi_iterations": record.get(
            "trace_ilu_upper_jacobi_iterations"
        ),
        "ilu_accounting_kind": record.get("trace_ilu_accounting_kind"),
        "ilu_factor_reuse_policy": record.get(
            "trace_ilu_factor_reuse_policy"
        ),
    }


def same_policy_attribution(mode: str, order: int) -> dict[str, Any]:
    """Describe the bounded, provisional three-member attribution contract."""
    members = {
        name: {
            **selection,
            "configuration": trace_policy_configuration(
                selection["engine"], mode, selection["trace_policy"],
                order=order
            ),
        }
        for name, selection in SAME_POLICY_MEMBERS.items()
    }
    common_matches = (
        members["O_common"]["configuration"] ==
        members["T_common"]["configuration"]
    )
    if not common_matches:
        raise CampaignError(
            f"common-fast policy table is not engine-invariant for {mode!r}"
        )
    return {
        "scope": "bounded_same_policy_triplet",
        "status": "provisional_non_promotion_attribution",
        "promotion_eligible": False,
        "members": members,
        "policy_gain_definition": "O_base median / O_common median",
        "tensor_local_gain_definition": "O_common median / T_common median",
        "package_gain_definition": "O_base median / T_common median",
        "common_fast_policy_exactly_matched": True,
        "includes_tensor_condensation_in_T_common": True,
        "mma_contribution_requires_separate_profiler_attribution": True,
        "bounded_aggregate_records_only": True,
    }


def validate_qualification_profile(arguments: argparse.Namespace) -> None:
    """Bind promotion evidence to one of the two declared campaign profiles."""
    expected = QUALIFICATION_PROFILES[arguments.workload_class]
    observed = {
        "workload_label": arguments.workload_label,
        "mesh_n": arguments.mesh_n,
        "domain_periods": arguments.domain_periods,
        "dt": arguments.dt,
        "export_control_mode": arguments.export_control_mode,
        "export_control_order": arguments.export_control_order,
        "pairs": arguments.pairs,
    }
    mismatches = [
        f"{name}={observed[name]!r} (expected {value!r})"
        for name, value in expected.items()
        if observed[name] != value
    ]
    if mismatches:
        raise CampaignError(
            "qualification evidence does not match the declared "
            f"{arguments.workload_class} profile: " + "; ".join(mismatches)
        )


def utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def parse_metadata(entries: Iterable[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in entries:
        key, separator, value = entry.partition("=")
        if not separator or not key or not value:
            raise CampaignError(f"{label} entries must be KEY=VALUE: {entry!r}")
        if key in result:
            raise CampaignError(f"duplicate {label} key: {key}")
        result[key] = value
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise CampaignError(f"executable is not a regular file: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise CampaignError(f"executable is not executable: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "mtime_ns": resolved.stat().st_mtime_ns,
    }


def regular_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise CampaignError(f"path is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "mtime_ns": resolved.stat().st_mtime_ns,
    }


def manifest_identity(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise CampaignError(f"build manifest is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "mtime_ns": resolved.stat().st_mtime_ns,
    }


def validate_executable_build_relationship(
        executable: dict[str, Any], build: dict[str, Any], *,
        source_modified_mtime_ns: int | None = None) -> dict[str, Any]:
    cache = Path(build["cmake_cache"]["path"])
    expected = (cache.parent / BENCHMARK_BUILD_RELATIVE_PATH).resolve()
    observed = Path(executable["path"]).resolve()
    if observed != expected:
        raise CampaignError(
            f"benchmark executable {observed} is not the canonical target "
            f"under supplied build cache {expected}"
        )
    if executable["mtime_ns"] < build["cmake_cache"]["mtime_ns"]:
        raise CampaignError("benchmark executable predates its supplied build cache")
    if source_modified_mtime_ns is not None and \
            executable["mtime_ns"] < source_modified_mtime_ns:
        raise CampaignError("frozen benchmark executable predates its approved overlay")
    return {
        "canonical_target_path": str(expected),
        "executable_not_older_than_cache": True,
        "executable_not_older_than_overlay": (
            True if source_modified_mtime_ns is not None else "not applicable"
        ),
    }


def parse_typed_cmake_cache(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if not raw_line or raw_line.startswith(("#", "//")) or "=" not in raw_line:
            continue
        key_and_type, value = raw_line.split("=", 1)
        key, separator, _ = key_and_type.partition(":")
        if separator and key:
            result[key] = value
    return result


def parse_key_value_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        key, separator, value = raw_line.partition("=")
        if not separator or not key or key in result:
            raise CampaignError(f"invalid key/value manifest line in {path}: {raw_line!r}")
        result[key] = value
    return result


def parse_semicolon_key_values(value: str, field: str) -> dict[str, str]:
    """Parse an exact semicolon-delimited build-identity binding."""
    result: dict[str, str] = {}
    for entry in value.split(";"):
        key, separator, item = entry.partition("=")
        if not separator or not key or not item or key in result:
            raise CampaignError(
                f"invalid {field} key/value entry: {entry!r}"
            )
        result[key] = item
    return result


def validated_hypre_identity(mfem: dict[str, str]) -> dict[str, Any]:
    """Bind the solver-work contract to one exact HYPRE Spack DAG."""
    spec = mfem.get("hypre_spec", "")
    dag_hash = mfem.get("hypre_hash", "")
    prefix = mfem.get("hypre_prefix", "")
    spec_tokens = spec.split()
    root_spec = spec_tokens[0] if spec_tokens else ""
    def enabled_root_variant(name: str) -> bool:
        marker = f"+{name}"
        position = root_spec.find(marker)
        end = position + len(marker)
        return position >= 0 and (
            end == len(root_spec) or root_spec[end] in "+~"
        )

    if not root_spec.startswith("hypre@") or \
            not enabled_root_variant("mpi") or "~mpi" in root_spec or \
            not enabled_root_variant("cuda") or "~cuda" in root_spec or \
            "cuda_arch=120" not in spec_tokens:
        raise CampaignError(
            "MFEM identity HYPRE spec must bind an MPI/CUDA SM120 build"
        )
    spack_base32 = set("abcdefghijklmnopqrstuvwxyz234567")
    if len(dag_hash) != 32 or any(
            character not in spack_base32 for character in dag_hash):
        raise CampaignError(
            "MFEM identity does not record an exact HYPRE Spack DAG hash"
        )
    prefix_path = Path(prefix)
    if not prefix_path.is_absolute() or \
            not prefix_path.name.endswith(f"-{dag_hash}"):
        raise CampaignError(
            "MFEM identity HYPRE prefix is not bound to its Spack DAG hash"
        )
    reuse_abi = parse_semicolon_key_values(
        mfem.get("reuse_abi", ""), "MFEM reuse_abi"
    )
    if reuse_abi.get("hypre_spec") != spec or \
            reuse_abi.get("hypre") != dag_hash:
        raise CampaignError(
            "MFEM reuse ABI does not bind the exact HYPRE spec and DAG hash"
        )

    manifest_fields = {
        key: value for key, value in sorted(mfem.items())
        if key.startswith("hypre_")
    }
    if any(not value for value in manifest_fields.values()):
        raise CampaignError("MFEM identity has an empty HYPRE provenance field")
    hexadecimal = set("0123456789abcdef")
    for key, value in manifest_fields.items():
        if "revision" in key or "commit" in key:
            if len(value) not in (40, 64) or any(
                    character not in hexadecimal for character in value):
                raise CampaignError(
                    f"MFEM identity {key} is not an exact source revision"
                )
        if key.endswith("sha256") and (
                len(value) != 64 or any(
                    character not in hexadecimal for character in value)):
            raise CampaignError(
                f"MFEM identity {key} is not an exact SHA-256"
            )
    return {
        "spack_spec": spec,
        "spack_dag_hash": dag_hash,
        "install_prefix": prefix,
        "reuse_abi_binding": {
            "hypre_spec": reuse_abi["hypre_spec"],
            "hypre": reuse_abi["hypre"],
        },
        # Retain source-revision/configuration fields automatically when a
        # newer canonical MFEM identity manifest exposes them.
        "manifest_fields": manifest_fields,
    }


def validated_build_provenance(*, source_root: Path, cmake_cache_path: str,
                               mfem_identity_path: str) -> dict[str, Any]:
    cache_path = Path(cmake_cache_path).expanduser().resolve(strict=True)
    mfem_path = Path(mfem_identity_path).expanduser().resolve(strict=True)
    cache = parse_typed_cmake_cache(cache_path)
    expected_cache = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CUDA_ARCHITECTURES": "120",
        "DG_IGR_ENABLE_HDG_RUNTIME": "ON",
        "DG_IGR_ENABLE_OBSERVABILITY": "OFF",
        "DG_IGR_OBSERVABILITY_USE_NVTX": "OFF",
    }
    for key, expected in expected_cache.items():
        if cache.get(key) != expected:
            raise CampaignError(
                f"build cache {key}={cache.get(key)!r}; expected {expected!r}"
            )
    cache_source = Path(cache.get("CMAKE_HOME_DIRECTORY", "")).resolve()
    if cache_source != source_root.resolve():
        raise CampaignError(
            f"build cache source {cache_source} does not match {source_root.resolve()}"
        )
    for key in ("CMAKE_CUDA_COMPILER", "CMAKE_CXX_COMPILER", "MFEM_DIR"):
        if not cache.get(key):
            raise CampaignError(f"build cache does not record {key}")

    mfem = parse_key_value_manifest(mfem_path)
    expected_mfem = {
        "schema": "dg-igr-mfem-identity",
        "version": "1",
        "mode": "gpu",
        "profile": "cuda-sm120",
        "build_type": "Release",
        "dependency_cuda_arch": "120",
        "mfem_cuda_archs": "80-virtual",
    }
    for key, expected in expected_mfem.items():
        if mfem.get(key) != expected:
            raise CampaignError(
                f"MFEM identity {key}={mfem.get(key)!r}; expected {expected!r}"
            )
    if Path(mfem.get("source_root", "")).resolve() != \
            (source_root / "tpl" / "mfem").resolve():
        raise CampaignError("MFEM identity source root does not match the source tree")
    hypre_identity = validated_hypre_identity(mfem)

    selected_cache = {
        key: cache[key] for key in (
            "CMAKE_HOME_DIRECTORY", "CMAKE_BUILD_TYPE",
            "CMAKE_CUDA_ARCHITECTURES", "CMAKE_CUDA_COMPILER",
            "CMAKE_CXX_COMPILER", "MFEM_DIR", "DG_IGR_ENABLE_HDG_RUNTIME",
            "DG_IGR_ENABLE_OBSERVABILITY", "DG_IGR_OBSERVABILITY_USE_NVTX",
        )
    }
    return {
        "cmake_cache": manifest_identity(str(cache_path)),
        "validated_cmake_values": selected_cache,
        "mfem_identity": manifest_identity(str(mfem_path)),
        "validated_mfem_values": mfem,
        "validated_hypre_identity": hypre_identity,
        "tracked_build_sh": regular_file_identity(source_root / "build.sh"),
    }


def artifact_identity(path: Path, recorded_path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise CampaignError(f"artifact is not a regular file: {resolved}")
    return {
        "path": recorded_path.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def validate_raw_artifacts(output_dir: Path,
                           invocations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for invocation in invocations:
        for path_field, identity_field in (
                ("raw_json", "raw_json_identity"),
                ("stderr", "stderr_identity")):
            relative = Path(invocation[path_field])
            observed = artifact_identity(output_dir / relative, relative)
            if observed != invocation[identity_field]:
                raise CampaignError(
                    f"retained artifact changed after capture: {relative}"
                )
            entries.append(observed)
    entries.sort(key=lambda item: item["path"])
    encoded = json.dumps(
        entries, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "artifact_count": len(entries),
        "entries": entries,
        "ledger_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def run_read_only(command: Sequence[str], timeout_seconds: float = 30.0
                  ) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
    )


def git_output(source_root: Path, *arguments: str) -> str:
    command = [
        "git", "--no-optional-locks", "-C", str(source_root), *arguments
    ]
    result = run_read_only(command)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise CampaignError(f"git query failed ({shlex.join(command)}): {detail}")
    return result.stdout.strip()


def git_identity(source_root: Path) -> dict[str, Any]:
    root = Path(git_output(source_root, "rev-parse", "--show-toplevel")).resolve()
    return {
        "root": str(root),
        "commit": git_output(root, "rev-parse", "HEAD"),
        "tree": git_output(root, "rev-parse", "HEAD^{tree}"),
        "status_short": git_output(
            root, "status", "--short", "--untracked-files=all"
        ),
    }


def git_bytes(source_root: Path, *arguments: str) -> bytes:
    command = [
        "git", "--no-optional-locks", "-C", str(source_root), *arguments
    ]
    result = subprocess.run(
        command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise CampaignError(
            f"git query failed ({shlex.join(command)}): {detail}"
        )
    return result.stdout


def frozen_overlay_identity(source_root: Path) -> dict[str, Any]:
    root = Path(git_output(source_root, "rev-parse", "--show-toplevel")).resolve()
    identity = git_identity(root)
    if identity["commit"] != FROZEN_CONTROL_COMMIT or \
            identity["tree"] != FROZEN_CONTROL_TREE:
        raise CampaignError("frozen source root is not the declared control commit/tree")
    changed = [line for line in git_output(
        root, "diff", "--name-only", "--"
    ).splitlines() if line]
    staged = git_output(root, "diff", "--cached", "--name-only", "--")
    untracked = git_output(root, "ls-files", "--others", "--exclude-standard")
    if changed != [FROZEN_OVERLAY_PATH.as_posix()] or staged or untracked:
        raise CampaignError(
            "frozen source must contain exactly the unstaged benchmark-order overlay"
        )
    overlay = git_bytes(
        root, "diff", "--binary", "--", FROZEN_OVERLAY_PATH.as_posix()
    )
    if not overlay or b"--order" not in overlay or b"options.order" not in overlay:
        raise CampaignError("frozen benchmark overlay lacks the declared order selector")
    overlay_sha256 = hashlib.sha256(overlay).hexdigest()
    if overlay_sha256 != FROZEN_OVERLAY_DIFF_SHA256:
        raise CampaignError(
            "frozen benchmark overlay differs from the reviewed order-only diff: "
            f"{overlay_sha256}"
        )
    modified_path = root / FROZEN_OVERLAY_PATH
    base = git_bytes(root, "show", f"HEAD:{FROZEN_OVERLAY_PATH.as_posix()}")
    return {
        "source": identity,
        "overlay_kind": "benchmark_order_cli_p1_p3_only",
        "changed_paths": changed,
        "diff_size_bytes": len(overlay),
        "diff_sha256": overlay_sha256,
        "expected_diff_sha256": FROZEN_OVERLAY_DIFF_SHA256,
        "base_file_sha256": hashlib.sha256(base).hexdigest(),
        "modified_file": regular_file_identity(modified_path),
    }


def query_nvidia_smi(executable: Path | None, gpu_index: int
                     ) -> dict[str, Any]:
    if executable is None:
        return {"available": False, "reason": "nvidia-smi was not found"}
    fields: dict[str, Any] = {}
    for field in NVIDIA_SMI_FIELDS:
        command = [
            str(executable),
            f"--id={gpu_index}",
            f"--query-gpu={field}",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = run_read_only(command, timeout_seconds=10.0)
            fields[field] = {
                "returncode": result.returncode,
                "value": result.stdout.strip() if result.returncode == 0 else None,
                "stderr": result.stderr.strip(),
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            fields[field] = {
                "returncode": None,
                "value": None,
                "stderr": str(error),
            }
    successful = {
        key: item["value"] for key, item in fields.items()
        if item["returncode"] == 0 and item["value"] not in (None, "", "N/A")
    }
    return {
        "available": bool(successful),
        "executable": str(executable),
        "gpu_index": gpu_index,
        "queried_utc": utc_now(),
        "values": successful,
        "queries": fields,
    }


def validate_hardware_snapshot(snapshot: dict[str, Any], required: bool) -> None:
    if not required:
        return
    values = snapshot.get("values", {})
    required_fields = (*REQUIRED_HARDWARE_IDENTITY_FIELDS,
                       "clocks.current.sm", "pstate", "temperature.gpu",
                       "power.draw", "power.limit")
    missing = [field for field in required_fields if not values.get(field)]
    if not snapshot.get("available") or missing:
        raise CampaignError(
            "required nvidia-smi hardware/clock metadata is unavailable: "
            + ", ".join(missing or ("all fields",))
        )


def validate_hardware_identity(reference: dict[str, Any],
                               observed: dict[str, Any]) -> None:
    reference_values = reference.get("values", {})
    observed_values = observed.get("values", {})
    changed = [
        field for field in REQUIRED_HARDWARE_IDENTITY_FIELDS
        if reference_values.get(field) != observed_values.get(field)
    ]
    if changed:
        raise CampaignError(
            "GPU identity changed during campaign: " + ", ".join(changed)
        )


def write_exclusive(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(content)


def write_json_exclusive(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(content, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise CampaignError("cannot compute a quantile of an empty sample")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def sample_statistics(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise CampaignError("cannot summarize an empty sample")
    checked = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= 0.0 for value in checked):
        raise CampaignError("timing samples must be finite and positive")
    median = statistics.median(checked)
    return {
        "count": len(checked),
        "median": median,
        "population_stddev": statistics.pstdev(checked),
        "median_absolute_deviation": statistics.median(
            [abs(value - median) for value in checked]
        ),
        "minimum": min(checked),
        "first_quartile": quantile(checked, 0.25),
        "third_quartile": quantile(checked, 0.75),
        "maximum": max(checked),
    }


def bootstrap_median_interval(values: Sequence[float], resamples: int,
                              seed: int) -> dict[str, Any]:
    if not values:
        raise CampaignError("cannot bootstrap an empty sample")
    generator = random.Random(seed)
    estimates = []
    count = len(values)
    for _ in range(resamples):
        estimates.append(statistics.median(
            values[generator.randrange(count)] for _ in range(count)
        ))
    return {
        "method": "paired_nonparametric_bootstrap_of_median",
        "confidence_level": 0.95,
        "resamples": resamples,
        "seed": seed,
        "lower": quantile(estimates, 0.025),
        "upper": quantile(estimates, 0.975),
    }


def classify_speedup(interval: dict[str, Any]) -> str:
    lower = checked_float(interval.get("lower"), "confidence_interval.lower")
    upper = checked_float(interval.get("upper"), "confidence_interval.upper")
    if lower > 1.0:
        return "win_beyond_observed_noise"
    if upper < 1.0:
        return "regression_beyond_observed_noise"
    return "break_even_or_indeterminate"


def checked_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignError(f"benchmark field {field} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignError(f"benchmark field {field} is not finite")
    return result


def checked_int(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CampaignError(
            f"benchmark field {field} is not an integer >= {minimum}"
        )
    return value


def checked_policy_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CampaignError(f"benchmark field {field} is not an integer")
    return value


def checked_uint64(value: Any, field: str) -> int:
    result = checked_int(value, field)
    if result > UINT64_MAX:
        raise CampaignError(f"benchmark field {field} exceeds uint64")
    return result


def tensor_volume_work_plan(*, mode: str, order: int, elements: int,
                            volume_points: int) -> dict[str, int]:
    """Independently reproduce the bounded production volume-WMMA plan."""
    if mode not in {"conventional", "igr"} or order not in {1, 2, 3} or \
            elements <= 0 or volume_points <= 0:
        raise CampaignError("tensor-volume work shape is invalid")
    interior_components = 7 if mode == "igr" else 4
    trace_components = 5 if mode == "igr" else 4
    scalar_interior_dofs = (order + 1) ** 2
    scalar_face_dofs = order + 1
    interior_dofs = interior_components * scalar_interior_dofs
    trace_dofs = 4 * trace_components * scalar_face_dofs
    coefficient_stride = 66 if mode == "igr" else 44
    workspace_doubles = interior_dofs * (trace_dofs + 1)
    capacity_points = workspace_doubles // coefficient_stride
    if capacity_points <= 0:
        raise CampaignError("tensor-volume workspace holds no coefficient")
    chunk_points = volume_points if volume_points <= capacity_points else \
        capacity_points - capacity_points % 4
    if chunk_points <= 0 or \
            (volume_points > capacity_points and chunk_points < 4):
        raise CampaignError("tensor-volume workspace cannot form an MMA chunk")

    chunks: list[int] = []
    remaining = volume_points
    quadrature_groups = 0
    while remaining:
        count = min(remaining, chunk_points)
        if count <= 0 or count > capacity_points or \
                (remaining > count and count % 4):
            raise CampaignError("tensor-volume chunk plan is inconsistent")
        chunks.append(count)
        quadrature_groups += (count + 3) // 4
        remaining -= count
    mma_tiles = (
        elements * interior_components *
        ((scalar_interior_dofs + 7) // 8) *
        ((interior_dofs + 8) // 8) * 3 * quadrature_groups
    )
    if mma_tiles <= 0 or mma_tiles > UINT64_MAX:
        raise CampaignError("tensor-volume MMA work exceeds uint64")
    return {
        "workspace_doubles": workspace_doubles,
        "coefficient_stride": coefficient_stride,
        "capacity_points": capacity_points,
        "chunk_points": chunk_points,
        "contraction_launches_per_assembly": len(chunks),
        "mma_tiles_per_assembly": mma_tiles,
    }


def _ilu_derived_bounds_are_valid(
        outer_iterations: int, linear_solve_attempts: int,
        krylov_restart_dimension: int,
        configured_iterations: int,
        callback_lower: int, callback_upper: int,
        inner_lower: int, inner_upper: int) -> bool:
    """Validate the pinned-GMRES derived bounds without uint64 overflow."""
    values = (
        outer_iterations, linear_solve_attempts, krylov_restart_dimension,
        callback_lower, callback_upper,
        inner_lower, inner_upper,
    )
    if krylov_restart_dimension <= 0 or configured_iterations <= 0 or any(
            value < 0 or value > UINT64_MAX for value in values):
        return False
    if outer_iterations == 0:
        return not any((callback_lower, callback_upper,
                        inner_lower, inner_upper))
    if linear_solve_attempts == 0 or outer_iterations > UINT64_MAX // 2 or \
            callback_lower <= outer_iterations or \
            callback_upper != 2 * outer_iterations:
        return False
    # Bounds are derived per successful trace solve and accumulated.  Check
    # whether the aggregate restart-cycle count is feasible without retaining
    # the per-solve iteration partition (which would be event history).
    restart_cycles = callback_lower - outer_iterations
    positive_solves = min(
        linear_solve_attempts, outer_iterations, restart_cycles
    )
    if positive_solves == 0 or \
            restart_cycles > UINT64_MAX // krylov_restart_dimension:
        return False
    maximum_partition_iterations = (
        restart_cycles * krylov_restart_dimension
    )
    minimum_partition_iterations = (
        (restart_cycles - positive_solves) * krylov_restart_dimension +
        positive_solves
    )
    if not (minimum_partition_iterations <= outer_iterations <=
            maximum_partition_iterations):
        return False
    if callback_upper > UINT64_MAX // configured_iterations:
        return False
    return (
        inner_lower == callback_lower * configured_iterations and
        inner_upper == callback_upper * configured_iterations
    )


def _ilu_jacobi_factor_spmv_bounds_are_valid(
        callback_lower: int, callback_upper: int,
        configured_iterations: int, updates_per_application: int,
        update_lower: int, update_upper: int) -> bool:
    """Validate bounded aggregate factor-SpMV work for iterative Jacobi."""
    values = (
        callback_lower, callback_upper, configured_iterations,
        updates_per_application, update_lower, update_upper,
    )
    if any(value < 0 or value > UINT64_MAX for value in values):
        return False
    if updates_per_application == 0:
        return update_lower == 0 and update_upper == 0
    if configured_iterations == 0 or \
            configured_iterations > UINT64_MAX // updates_per_application:
        return False
    updates_per_callback = configured_iterations * updates_per_application
    if callback_upper > UINT64_MAX // updates_per_callback:
        return False
    return (
        update_lower == callback_lower * updates_per_callback and
        update_upper == callback_upper * updates_per_callback
    )


def _ilu_factor_accounting_is_valid(
        trace_solves: int, linear_solve_attempts: int, policy: str,
        factor_setups: int,
        factor_reuses: int, maximum_factor_age: int) -> bool:
    """Validate aggregate counts for the typed factor-lifetime policy."""
    if trace_solves <= 0 or linear_solve_attempts < trace_solves:
        return False
    if policy == "not_applicable":
        return not any((factor_setups, factor_reuses, maximum_factor_age))
    if policy == "refresh_each_trace_solve":
        return factor_setups == linear_solve_attempts and \
            factor_reuses == 0 and \
            maximum_factor_age == 0
    if policy == "first_trace_solve_per_attempt":
        return factor_setups == 1 and \
            factor_reuses == linear_solve_attempts - 1 and \
            maximum_factor_age == trace_solves - 1
    return False


def _mgr_accounting_is_valid(
        *, expected_mgr: bool, expected_element_field_split: bool,
        trace_solves: int, linear_iterations: int,
        equilibration_builds: int, equilibration_passes: int,
        mgr_setup_calls: int, mgr_preconditioner_applications: int,
        mgr_f_ilu_setups: int, mgr_f_ilu_applications: int,
        minimum_left_scale: float, maximum_left_scale: float,
        minimum_right_scale: float, maximum_right_scale: float) -> bool:
    """Validate constant-size MGR/equilibration aggregates, never history."""
    counters = (
        equilibration_builds,
        equilibration_passes,
        mgr_setup_calls,
        mgr_preconditioner_applications,
        mgr_f_ilu_setups,
        mgr_f_ilu_applications,
    )
    scales = (
        minimum_left_scale,
        maximum_left_scale,
        minimum_right_scale,
        maximum_right_scale,
    )
    if any(value < 0 or value > UINT64_MAX for value in counters) or \
            any(not math.isfinite(value) for value in scales):
        return False
    if expected_element_field_split:
        return (
            not expected_mgr and
            equilibration_builds == trace_solves and
            equilibration_passes == 2 * trace_solves and
            mgr_setup_calls == 0 and
            mgr_preconditioner_applications == 0 and
            mgr_f_ilu_setups == 0 and mgr_f_ilu_applications == 0 and
            all(value == 0.0 for value in scales)
        )
    if not expected_mgr:
        return not any(counters) and all(value == 0.0 for value in scales)
    return (
        equilibration_builds == trace_solves and
        equilibration_passes == 2 * trace_solves and
        mgr_setup_calls == trace_solves and
        mgr_f_ilu_setups == mgr_setup_calls and
        mgr_f_ilu_applications == mgr_preconditioner_applications and
        (linear_iterations == 0 or mgr_preconditioner_applications > 0) and
        0.0 < minimum_left_scale <= maximum_left_scale and
        0.0 < minimum_right_scale <= maximum_right_scale
    )


def _element_schwarz_accounting_is_valid(
        *, expected_schwarz: bool, order: int, trace_solves: int,
        linear_iterations: int, krylov_restart_dimension: int,
        patch_setups: int, preconditioner_applications: int,
        patch_rows: int, minimum_multiplicity: int,
        maximum_multiplicity: int) -> bool:
    """Validate bounded native element-Schwarz aggregates, never history."""
    values = (
        trace_solves,
        linear_iterations,
        krylov_restart_dimension,
        patch_setups,
        preconditioner_applications,
        patch_rows,
        minimum_multiplicity,
        maximum_multiplicity,
    )
    if any(value < 0 or value > UINT64_MAX for value in values):
        return False
    if not expected_schwarz:
        return not any((
            patch_setups,
            preconditioner_applications,
            patch_rows,
            minimum_multiplicity,
            maximum_multiplicity,
        ))
    expected_patch_rows = 4 * 5 * (order + 1)
    if order not in (1, 2, 3) or trace_solves == 0 or \
            patch_setups != trace_solves or \
            patch_rows != expected_patch_rows or \
            minimum_multiplicity < 1 or \
            minimum_multiplicity > maximum_multiplicity or \
            maximum_multiplicity > 2 or \
            krylov_restart_dimension == 0:
        return False
    if linear_iterations == 0:
        return preconditioner_applications == 0
    if preconditioner_applications <= linear_iterations:
        return False
    # Native right-preconditioned GMRES applies the patch solve once per
    # Krylov iteration and once per restart update.  The aggregate restart
    # count is therefore exact even though no per-solve partition is retained.
    restart_cycles = preconditioner_applications - linear_iterations
    return (
        restart_cycles <= linear_iterations and
        restart_cycles <= UINT64_MAX // krylov_restart_dimension and
        linear_iterations <= restart_cycles * krylov_restart_dimension
    )


def validate_memory_snapshot(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignError(f"benchmark field {field} is not an object")
    free = checked_int(value.get("free_bytes"), f"{field}.free_bytes")
    total = checked_int(value.get("total_bytes"), f"{field}.total_bytes", 1)
    used = checked_int(value.get("used_bytes"), f"{field}.used_bytes")
    if free > total or used != total - free:
        raise CampaignError(f"benchmark field {field} is internally inconsistent")
    return value


def validate_memory_controls(record: dict[str, Any]) -> None:
    snapshots = record.get("device_memory_snapshots")
    if not isinstance(snapshots, dict):
        raise CampaignError("schema-2 benchmark has no device-memory snapshots")
    for name in (
        "after_device_selection_before_problem",
        "after_initialization",
        "after_samples",
    ):
        validate_memory_snapshot(snapshots.get(name), f"device_memory_snapshots.{name}")

    umpire = record.get("umpire_allocator_high_watermarks_after_samples")
    if not isinstance(umpire, dict) or not isinstance(umpire.get("available"), bool):
        raise CampaignError("schema-2 benchmark has no Umpire memory evidence")
    for allocator in ("mfem", "hypre", "sum"):
        available_field = f"{allocator}_available"
        bytes_field = f"{allocator}_bytes"
        available = umpire.get(available_field)
        if not isinstance(available, bool):
            raise CampaignError(
                f"benchmark field umpire_allocator_high_watermarks_after_samples."
                f"{available_field} is not boolean"
            )
        if available:
            checked_int(
                umpire.get(bytes_field),
                "umpire_allocator_high_watermarks_after_samples." + bytes_field,
            )
        elif umpire.get(bytes_field) is not None:
            raise CampaignError(
                f"unavailable Umpire allocator {allocator} reported a byte count"
            )
    if umpire["sum_available"] and umpire["mfem_available"] and \
            umpire["hypre_available"] and \
            umpire["sum_bytes"] != umpire["mfem_bytes"] + umpire["hypre_bytes"]:
        raise CampaignError("Umpire allocator high-watermark sum is inconsistent")


def validate_final_sample(sample: Any, *, engine: str, mode: str, order: int,
                          trace_policy: str,
                          expected_preconditioner: str,
                          volume_mma_tiles_per_assembly: int,
                          krylov_restart_dimension: int,
                          true_residual_tightening_enabled: bool,
                          expected_phase: str, expected_index: int,
                          expected_duration: float | None) -> None:
    if not isinstance(sample, dict):
        raise CampaignError("schema-2 benchmark sample is not an object")
    if sample.get("phase") != expected_phase or \
            sample.get("measured_index") != expected_index:
        raise CampaignError("schema-2 benchmark sample ordering is invalid")
    duration = checked_float(sample.get("duration_seconds"),
                             "samples.duration_seconds")
    if duration <= 0.0:
        raise CampaignError("benchmark sample duration must be positive")
    if expected_duration is not None and not math.isclose(
            duration, expected_duration, rel_tol=1.0e-13, abs_tol=0.0):
        raise CampaignError("sample duration differs from raw duration evidence")

    work = {
        field: checked_int(sample.get(field), f"samples.{field}")
        for field in (
            "retry_index",
            "device_ticket_count",
            "nonlinear_updates",
            "trace_solves",
            "linear_iterations",
            "linear_solve_attempts",
            "newton_correction_linear_iterations",
            "final_coherent_linear_iterations",
            "true_residual_tightenings",
            "line_search_trials",
            "rejected_line_search_trials",
            "iterative_refinement_steps",
            "residual_replacements",
            "precision_corrections",
        )
    }
    checked_int(sample.get("attempt_sequence"), "samples.attempt_sequence", 1)
    if work["trace_solves"] <= 0:
        raise CampaignError("accepted sample reported no true-trace solve")
    if work["linear_iterations"] != \
            work["newton_correction_linear_iterations"] + \
            work["final_coherent_linear_iterations"] or \
            work["linear_solve_attempts"] != \
            work["trace_solves"] + work["true_residual_tightenings"]:
        raise CampaignError("role-specific linear work accounting is invalid")
    if work["true_residual_tightenings"] > 0 and not \
            true_residual_tightening_enabled:
        raise CampaignError(
            "sample reports a true-residual tightening that the recorded "
            "role-specific tolerance policy cannot request"
        )
    if work["rejected_line_search_trials"] > work["line_search_trials"]:
        raise CampaignError("rejected line-search work exceeds total trials")
    if any(work[field] != 0 for field in (
            "iterative_refinement_steps", "residual_replacements",
            "precision_corrections")):
        raise CampaignError(
            "the implemented all-FP64 policy reported unexpected correction work"
        )
    if sample.get("all_fp64_policy") is not True or \
            sample.get("correction_accounting_complete") is not True:
        raise CampaignError(
            "sample lacks complete solver-sourced all-FP64 correction accounting"
        )

    for field in (
        "initial_residual",
        "final_residual",
        "true_relative_residual",
        "maximum_conservation_error",
        "maximum_conservation_relative_error",
    ):
        if checked_float(sample.get(field), f"samples.{field}") < 0.0:
            raise CampaignError(f"benchmark sample field {field} is negative")
    for field in ("maximum_wave_speed", "backend_stable_dt"):
        if checked_float(sample.get(field), f"samples.{field}") <= 0.0:
            raise CampaignError(f"benchmark sample field {field} is not positive")

    products = sample.get("product_generations")
    if not isinstance(products, dict):
        raise CampaignError("measured sample lacks product-generation evidence")
    for field in ("residual", "matrix", "factor", "solution"):
        checked_int(products.get(field), f"samples.product_generations.{field}")
    checked_int(products.get("issued_mask"),
                "samples.product_generations.issued_mask", 1)

    tensor = sample.get("tensor_core")
    if not isinstance(tensor, dict):
        raise CampaignError("sample lacks tensor-core evidence")
    if tensor.get("selected_engine") != engine:
        raise CampaignError("sample engine evidence differs from selection")
    resolved_trace_policy = expected_resolved_trace_policy(
        engine, mode, trace_policy, order=order
    )
    if tensor.get("requested_trace_policy") != trace_policy:
        raise CampaignError(
            "sample requested trace-policy evidence differs from selection"
        )
    if tensor.get("resolved_trace_policy") != resolved_trace_policy:
        raise CampaignError(
            "sample resolved trace-policy evidence differs from resolution"
        )
    (
        expected_trace_solver,
        expected_native,
        expected_mgr,
        expected_rank2,
    ) = expected_trace_solver_configuration(engine, mode, trace_policy)
    expected_sigma_jacobi_sweeps, _, _ = fixed_sigma_jacobi_policy(
        resolved_trace_policy
    )
    expected_fixed_sigma_jacobi = expected_sigma_jacobi_sweeps > 0
    expected_schwarz = resolved_trace_policy in {
        "native-element-schwarz-fp64",
        "native-element-schwarz-matrix-free-fp64",
        "native-element-schwarz-equilibrated-field-split-fp64",
        "native-element-schwarz-matrix-free-equilibrated-field-split-fp64",
        "native-element-schwarz-matrix-free-warp-equilibrated-field-split-fp64",
    } or expected_fixed_sigma_jacobi
    expected_matrix_free = resolved_trace_policy in {
        "native-element-schwarz-matrix-free-fp64",
        "native-element-schwarz-matrix-free-equilibrated-field-split-fp64",
        "native-element-schwarz-matrix-free-warp-equilibrated-field-split-fp64",
    } or expected_fixed_sigma_jacobi
    expected_element_field_split = resolved_trace_policy in {
        "native-element-schwarz-equilibrated-field-split-fp64",
        "native-element-schwarz-matrix-free-equilibrated-field-split-fp64",
        "native-element-schwarz-matrix-free-warp-equilibrated-field-split-fp64",
    } or expected_fixed_sigma_jacobi
    if tensor.get("trace_solver") != expected_trace_solver:
        raise CampaignError(
            f"sample trace_solver={tensor.get('trace_solver')!r}; expected "
            f"{expected_trace_solver!r} for rank-one {engine!r}"
        )
    native_trace_solves = checked_uint64(
        tensor.get("native_rank1_trace_solve_calls"),
        "samples.tensor_core.native_rank1_trace_solve_calls",
    )
    hypre_trace_solves = checked_uint64(
        tensor.get("hypre_trace_solve_calls"),
        "samples.tensor_core.hypre_trace_solve_calls",
    )
    matrix_free_operator_applications = checked_uint64(
        tensor.get("native_matrix_free_trace_operator_applications"),
        "samples.tensor_core.native_matrix_free_trace_operator_applications",
    )
    matrix_free_accounting_exact = (
        matrix_free_operator_applications >= work["linear_iterations"] and
        ((work["linear_iterations"] == 0) ==
         (matrix_free_operator_applications == 0))
    ) if expected_matrix_free else matrix_free_operator_applications == 0
    if tensor.get("native_rank1_trace_solver_configured") is not \
            expected_native or \
            tensor.get("hypre_mgr_rank1_trace_solver_configured") is not \
            expected_mgr or \
            tensor.get("native_element_schwarz_configured") is not \
            expected_schwarz or \
            tensor.get("native_matrix_free_trace_operator_configured") is \
            not expected_matrix_free or \
            tensor.get("hypre_host_staged_rank2_trace_solver_configured") \
            is not expected_rank2 or \
            tensor.get("trace_solver_accounting_complete") is not True or \
            not matrix_free_accounting_exact or \
            native_trace_solves != (work["linear_solve_attempts"]
                                     if expected_native else 0) or \
            hypre_trace_solves != (0 if expected_native else
                                   work["linear_solve_attempts"]):
        raise CampaignError(
            "sample recurring trace-solver accounting does not match the "
            f"rank-one {engine!r} engine contract"
        )
    if tensor.get("trace_preconditioner_configured") is not True:
        raise CampaignError("sample trace preconditioner was not configured")
    if tensor.get("trace_preconditioner") != expected_preconditioner:
        raise CampaignError(
            "sample trace_preconditioner="
            f"{tensor.get('trace_preconditioner')!r}; expected "
            f"{expected_preconditioner!r} for {engine!r}"
        )
    expected_ilu = expected_trace_ilu_policy(engine, mode, trace_policy)
    observed_ilu = tuple(
        checked_policy_int(tensor.get(field), f"samples.tensor_core.{field}")
        for field in (
            "ilu_type",
            "ilu_level_of_fill",
            "ilu_local_reordering",
            "ilu_application_iterations",
        )
    )
    if observed_ilu != expected_ilu:
        raise CampaignError(
            f"sample trace ILU policy={observed_ilu!r}; expected "
            f"{expected_ilu!r} for {engine!r}/{mode!r}"
        )
    expected_triangular = expected_trace_ilu_triangular_solve_policy(
        engine, mode, trace_policy
    )
    observed_triangular = (
        tensor.get("ilu_triangular_solve_kind"),
        checked_int(
            tensor.get("ilu_lower_jacobi_iterations"),
            "samples.tensor_core.ilu_lower_jacobi_iterations",
        ),
        checked_int(
            tensor.get("ilu_upper_jacobi_iterations"),
            "samples.tensor_core.ilu_upper_jacobi_iterations",
        ),
    )
    if observed_triangular != expected_triangular[:3]:
        raise CampaignError(
            f"sample ILU triangular-solve policy={observed_triangular!r}; "
            f"expected {expected_triangular[:3]!r} for {engine!r}/{mode!r}"
        )
    factor_setups = checked_uint64(
        tensor.get("ilu_factor_setups"),
        "samples.tensor_core.ilu_factor_setups",
    )
    factor_reuses = checked_uint64(
        tensor.get("ilu_factor_reuses"),
        "samples.tensor_core.ilu_factor_reuses",
    )
    callback_lower = checked_uint64(
        tensor.get("ilu_preconditioner_callbacks_lower_bound"),
        "samples.tensor_core.ilu_preconditioner_callbacks_lower_bound",
    )
    callback_upper = checked_uint64(
        tensor.get("ilu_preconditioner_callbacks_upper_bound"),
        "samples.tensor_core.ilu_preconditioner_callbacks_upper_bound",
    )
    inner_lower = checked_uint64(
        tensor.get("ilu_inner_iterations_lower_bound"),
        "samples.tensor_core.ilu_inner_iterations_lower_bound",
    )
    inner_upper = checked_uint64(
        tensor.get("ilu_inner_iterations_upper_bound"),
        "samples.tensor_core.ilu_inner_iterations_upper_bound",
    )
    jacobi_update_lower = checked_uint64(
        tensor.get("ilu_jacobi_factor_spmv_updates_lower_bound"),
        "samples.tensor_core."
        "ilu_jacobi_factor_spmv_updates_lower_bound",
    )
    jacobi_update_upper = checked_uint64(
        tensor.get("ilu_jacobi_factor_spmv_updates_upper_bound"),
        "samples.tensor_core."
        "ilu_jacobi_factor_spmv_updates_upper_bound",
    )
    factor_age = checked_uint64(
        tensor.get("ilu_factor_age_generations"),
        "samples.tensor_core.ilu_factor_age_generations",
    )
    ilu_iterations = expected_ilu[3]
    expected_accounting_kind = (
        "derived_bounds_v1" if ilu_iterations else "unavailable"
    )
    expected_factor_reuse = expected_trace_ilu_factor_reuse_policy(
        engine, mode, trace_policy
    )
    if tensor.get("ilu_accounting_kind") != expected_accounting_kind or \
            tensor.get("ilu_factor_reuse_policy") != expected_factor_reuse or \
            tensor.get("ilu_accounting_complete") is not True:
        raise CampaignError("sample ILU accounting contract is incomplete")
    if ilu_iterations == 0:
        if not _ilu_factor_accounting_is_valid(
                work["trace_solves"], work["linear_solve_attempts"],
                expected_factor_reuse,
                factor_setups, factor_reuses, factor_age) or \
                any((callback_lower, callback_upper,
                     inner_lower, inner_upper,
                     jacobi_update_lower, jacobi_update_upper)):
            raise CampaignError("non-ILU sample reported ILU work")
    elif not _ilu_factor_accounting_is_valid(
            work["trace_solves"], work["linear_solve_attempts"],
            expected_factor_reuse,
            factor_setups, factor_reuses, factor_age) or \
            not _ilu_derived_bounds_are_valid(
                work["linear_iterations"], work["linear_solve_attempts"],
                krylov_restart_dimension,
                ilu_iterations,
                callback_lower, callback_upper, inner_lower, inner_upper
            ):
        raise CampaignError("sample ILU work accounting is inconsistent")
    if not _ilu_jacobi_factor_spmv_bounds_are_valid(
            callback_lower, callback_upper, ilu_iterations,
            expected_triangular[3], jacobi_update_lower,
            jacobi_update_upper):
        raise CampaignError(
            "sample ILU Jacobi factor-SpMV accounting is inconsistent"
        )
    mgr_counters = {
        field: checked_uint64(
            tensor.get(field), f"samples.tensor_core.{field}"
        )
        for field in (
            "equilibration_builds",
            "equilibration_passes",
            "mgr_setup_calls",
            "mgr_preconditioner_applications",
            "mgr_f_ilu_setups",
            "mgr_f_ilu_applications",
        )
    }
    mgr_scales = {
        field: checked_float(
            tensor.get(field), f"samples.tensor_core.{field}"
        )
        for field in (
            "minimum_left_scale",
            "maximum_left_scale",
            "minimum_right_scale",
            "maximum_right_scale",
        )
    }
    if not _mgr_accounting_is_valid(
            expected_mgr=expected_mgr,
            expected_element_field_split=expected_element_field_split,
            trace_solves=work["trace_solves"],
            linear_iterations=work["linear_iterations"],
            **mgr_counters,
            **mgr_scales):
        raise CampaignError(
            "sample bounded MGR/equilibration accounting is inconsistent"
        )
    schwarz_counters = {
        field: checked_uint64(
            tensor.get(field), f"samples.tensor_core.{field}"
        )
        for field in (
            "element_schwarz_patch_setups",
            "element_schwarz_preconditioner_applications",
        )
    }
    schwarz_shape = {
        field: checked_int(
            tensor.get(field), f"samples.tensor_core.{field}"
        )
        for field in (
            "element_schwarz_patch_rows",
            "element_schwarz_minimum_multiplicity",
            "element_schwarz_maximum_multiplicity",
        )
    }
    if not _element_schwarz_accounting_is_valid(
            expected_schwarz=expected_schwarz,
            order=order,
            trace_solves=work["trace_solves"],
            linear_iterations=work["linear_iterations"],
            krylov_restart_dimension=krylov_restart_dimension,
            patch_setups=schwarz_counters[
                "element_schwarz_patch_setups"
            ],
            preconditioner_applications=schwarz_counters[
                "element_schwarz_preconditioner_applications"
            ],
            patch_rows=schwarz_shape["element_schwarz_patch_rows"],
            minimum_multiplicity=schwarz_shape[
                "element_schwarz_minimum_multiplicity"
            ],
            maximum_multiplicity=schwarz_shape[
                "element_schwarz_maximum_multiplicity"
            ]):
        raise CampaignError(
            "sample bounded native element-Schwarz accounting is "
            "inconsistent"
        )
    sigma_jacobi_sweeps = checked_int(
        tensor.get("field_split_sigma_jacobi_sweeps"),
        "samples.tensor_core.field_split_sigma_jacobi_sweeps",
    )
    sigma_jacobi_damping = checked_float(
        tensor.get("field_split_sigma_jacobi_damping"),
        "samples.tensor_core.field_split_sigma_jacobi_damping",
    )
    sigma_jacobi_updates = checked_uint64(
        tensor.get("field_split_sigma_jacobi_updates"),
        "samples.tensor_core.field_split_sigma_jacobi_updates",
    )
    expected_sigma_jacobi_damping = 2.0 / 3.0 \
        if expected_fixed_sigma_jacobi else 0.0
    if expected_sigma_jacobi_sweeps > 0 and \
            schwarz_counters[
                "element_schwarz_preconditioner_applications"
            ] > UINT64_MAX // expected_sigma_jacobi_sweeps:
        raise CampaignError("fixed-sigma Jacobi aggregate work exceeds uint64")
    expected_sigma_jacobi_updates = (
        schwarz_counters["element_schwarz_preconditioner_applications"] *
        expected_sigma_jacobi_sweeps
    )
    if sigma_jacobi_sweeps != expected_sigma_jacobi_sweeps or \
            not math.isclose(
                sigma_jacobi_damping, expected_sigma_jacobi_damping,
                rel_tol=0.0, abs_tol=0.0) or \
            sigma_jacobi_updates != expected_sigma_jacobi_updates:
        raise CampaignError(
            "sample fixed-sigma Jacobi work does not match the selected "
            f"trace policy {trace_policy!r}"
        )
    launches = checked_int(
        tensor.get("condensation_launches"),
        "samples.tensor_core.condensation_launches",
    )
    tiles = checked_int(tensor.get("mma_tiles"),
                        "samples.tensor_core.mma_tiles")
    volume_launches = checked_uint64(
        tensor.get("volume_assembly_launches"),
        "samples.tensor_core.volume_assembly_launches",
    )
    volume_tiles = checked_uint64(
        tensor.get("volume_mma_tiles"),
        "samples.tensor_core.volume_mma_tiles",
    )
    if not isinstance(tensor.get("hardware_supported"), bool):
        raise CampaignError("sample tensor hardware-support evidence is missing")
    if tensor.get("fallback_count") != 0 or \
            tensor.get("fallback_accounting_complete") is not True or \
            tensor.get("no_fallback") is not True:
        raise CampaignError("fallback execution is not valid campaign evidence")
    if engine == "tensor":
        if tensor.get("hardware_supported") is not True:
            raise CampaignError("tensor hardware support was not reported")
        if launches <= 0 or tiles <= 0 or volume_launches <= 0:
            raise CampaignError("tensor run did not report production MMA work")
        if volume_mma_tiles_per_assembly <= 0 or \
                volume_launches > UINT64_MAX // \
                volume_mma_tiles_per_assembly or \
                volume_tiles != volume_launches * \
                volume_mma_tiles_per_assembly:
            raise CampaignError(
                "tensor-volume aggregate differs from exact shape/chunk work"
            )
    elif launches != 0 or tiles != 0 or volume_launches != 0 or \
            volume_tiles != 0:
        raise CampaignError("ordinary CUDA reported tensor work")

    transfers = sample.get("transfers")
    if not isinstance(transfers, dict) or transfers.get("complete") is not True:
        raise CampaignError("sample lacks complete transfer-accounting evidence")
    transfer_values = {}
    for field in (
        "host_to_device_bytes",
        "device_to_host_bytes",
        "device_to_device_bytes",
        "mpi_staging_bytes",
        "synchronization_events",
        "event_count",
    ):
        transfer_values[field] = checked_int(
            transfers.get(field), f"samples.transfers.{field}"
        )
    if transfer_values["device_to_host_bytes"] <= 0 or \
            transfer_values["device_to_device_bytes"] <= 0 or \
            transfer_values["synchronization_events"] <= 0 or \
            transfer_values["event_count"] <= 0:
        raise CampaignError("accepted sample omitted recurring transfer/sync work")

    gates = sample.get("gates")
    required_gates = (
        "candidate_ready",
        "finite",
        "admissible",
        "recovery",
        "true_residual",
        "conservation",
        "telemetry",
        "products_coupled_and_ordered",
        "success",
    )
    if not isinstance(gates, dict) or any(
            gates.get(field) is not True for field in required_gates):
        raise CampaignError("sample did not pass every acceptance gate")


def validate_frozen_sample(sample: Any, *, expected_phase: str,
                           expected_index: int,
                           expected_duration: float | None) -> None:
    if not isinstance(sample, dict) or sample.get("phase") != expected_phase or \
            sample.get("measured_index") != expected_index:
        raise CampaignError("frozen sample ordering is invalid")
    duration = checked_float(sample.get("duration_seconds"),
                             "frozen.samples.duration_seconds")
    if duration <= 0.0 or (expected_duration is not None and not math.isclose(
            duration, expected_duration, rel_tol=1.0e-13, abs_tol=0.0)):
        raise CampaignError("frozen sample duration evidence is invalid")
    for field in (
        "initial_residual", "final_residual", "true_relative_residual",
        "maximum_conservation_error", "maximum_conservation_relative_error",
    ):
        if checked_float(sample.get(field), f"frozen.samples.{field}") < 0.0:
            raise CampaignError(f"frozen sample field {field} is negative")
    for field in ("maximum_wave_speed", "backend_stable_dt"):
        if checked_float(sample.get(field), f"frozen.samples.{field}") <= 0.0:
            raise CampaignError(f"frozen sample field {field} is not positive")
    work = {
        field: checked_int(sample.get(field), f"frozen.samples.{field}")
        for field in (
            "nonlinear_updates", "trace_solves", "linear_iterations",
            "line_search_trials", "rejected_line_search_trials",
        )
    }
    if work["trace_solves"] <= 0 or \
            work["rejected_line_search_trials"] > work["line_search_trials"]:
        raise CampaignError("frozen sample solver-work accounting is invalid")
    transfers = sample.get("transfers")
    if not isinstance(transfers, dict) or transfers.get("complete") is not True:
        raise CampaignError("frozen sample transfer accounting is incomplete")
    for field in (
        "host_to_device_bytes", "device_to_host_bytes",
        "device_to_device_bytes", "mpi_staging_bytes",
        "synchronization_events", "event_count",
    ):
        checked_int(transfers.get(field), f"frozen.samples.transfers.{field}")
    if transfers["device_to_host_bytes"] <= 0 or \
            transfers["device_to_device_bytes"] <= 0 or \
            transfers["synchronization_events"] <= 0 or \
            transfers["event_count"] <= 0:
        raise CampaignError("frozen sample omitted recurring transfer/sync work")
    gates = sample.get("gates")
    required_gates = (
        "candidate_ready", "finite", "admissible", "recovery",
        "true_residual", "conservation", "telemetry",
        "products_coupled_and_ordered", "success",
    )
    if not isinstance(gates, dict) or any(
            gates.get(field) is not True for field in required_gates):
        raise CampaignError("frozen sample did not pass every acceptance gate")


def normalize_gpu_uuid(value: Any) -> str:
    if not isinstance(value, str):
        raise CampaignError("GPU UUID is not a string")
    normalized = value.strip().lower()
    if normalized.startswith("gpu-"):
        normalized = normalized[4:]
    normalized = normalized.replace("-", "")
    if len(normalized) != 32 or any(character not in "0123456789abcdef"
                                    for character in normalized):
        raise CampaignError(f"GPU UUID has an invalid form: {value!r}")
    return normalized


def validate_embedded_final_identity(
        record: dict[str, Any], expected_build: dict[str, str],
        expected_gpu: dict[str, Any]) -> None:
    build = record.get("build_identity")
    if build != expected_build:
        raise CampaignError(
            f"benchmark embedded build identity {build!r} differs from "
            f"the candidate/build binding {expected_build!r}"
        )
    if checked_int(record.get("gpu_device_ordinal"),
                   "gpu_device_ordinal") != expected_gpu["device_ordinal"]:
        raise CampaignError("benchmark executed on an unexpected CUDA ordinal")
    if normalize_gpu_uuid(record.get("gpu_uuid")) != \
            normalize_gpu_uuid(expected_gpu["uuid"]):
        raise CampaignError("benchmark CUDA UUID differs from nvidia-smi lease UUID")
    if checked_int(record.get("gpu_compute_capability"),
                   "gpu_compute_capability") != expected_gpu["compute_capability"]:
        raise CampaignError("benchmark compute capability differs from lease metadata")


def validate_final_policy(record: dict[str, Any], *, mode: str,
                          order: int) -> dict[str, int]:
    exact_top_level = {
        "time_integrator": "backward_euler",
        "time_step_policy": "fixed",
        "initial_state_kind": "projected_smooth_periodic",
        "initial_trace_kind": "projected_from_committed_interior",
        "source_kind": "zero",
        "boundary_kind": "periodic_topology",
        "principal_threads": 128,
        "principal_element_threads": 4,
        "principal_elements_per_block": 32,
        "principal_execution": "four_lane_subwarp_per_element",
        "profiling_observability": False,
        "profile_measured_step": -1,
    }
    for field, expected in exact_top_level.items():
        if record.get(field) != expected:
            raise CampaignError(
                f"benchmark field {field}={record.get(field)!r}; expected {expected!r}"
            )
    if not math.isclose(checked_float(record.get("gamma"), "gamma"), 1.4,
                        rel_tol=0.0, abs_tol=0.0):
        raise CampaignError("benchmark Euler gamma differs from the frozen policy")
    expected_components = (7, 5) if mode == "igr" else (4, 4)
    if (record.get("interior_components"), record.get("trace_components")) != \
            expected_components:
        raise CampaignError("benchmark U/Uhat component contract is invalid")

    elements = checked_int(record.get("elements"), "elements", 1)
    if elements != checked_int(record.get("mesh_n"), "mesh_n", 2) ** 2:
        raise CampaignError("benchmark element count differs from mesh_n squared")
    blocks = checked_int(record.get("principal_blocks"), "principal_blocks", 1)
    if blocks != (elements + 31) // 32:
        raise CampaignError("benchmark principal block count is inconsistent")
    checked_int(record.get("interior_unknowns"), "interior_unknowns", 1)
    checked_int(record.get("true_trace_unknowns"), "true_trace_unknowns", 1)

    precision = record.get("precision_policy")
    expected_precision = {
        "external_state": "fp64",
        "local_lu": "fp64",
        "local_triangular_solve": "fp64",
        "tensor_operands": "fp64",
        "tensor_accumulation": "fp64",
        "acceptance_gates": "fp64",
        "mixed_precision_enabled": False,
        "iterative_refinement_enabled": False,
        "residual_replacement_enabled": False,
        "precision_correction_enabled": False,
    }
    if precision != expected_precision:
        raise CampaignError("benchmark precision policy differs from all-FP64 policy")

    policy = record.get("numerical_policy")
    if not isinstance(policy, dict):
        raise CampaignError("benchmark lacks its numerical policy")
    quadrature = 2 * order + 2
    expected_volume_assembly_order = 12 if mode == "igr" else quadrature
    expected_volume_points = (expected_volume_assembly_order // 2 + 1) ** 2
    if checked_int(record.get("volume_assembly_order"),
                   "volume_assembly_order", 1) != \
            expected_volume_assembly_order or \
            checked_int(record.get("volume_points"),
                        "volume_points", 1) != expected_volume_points:
        raise CampaignError(
            "resolved volume quadrature shape differs from the numerical policy"
        )
    volume_work = tensor_volume_work_plan(
        mode=mode, order=order, elements=elements,
        volume_points=expected_volume_points,
    )
    exact_policy = {
        "volume_projection_quadrature_order": quadrature,
        "true_trace_projection_quadrature_order": quadrature,
        "conventional_assembly_quadrature_order": quadrature,
        "igr_volume_assembly_quadrature_order": 12,
        "igr_face_assembly_quadrature_order": 12,
        "maximum_nonlinear_updates": 80,
        "requested_maximum_linear_iterations": 0,
        "resolved_maximum_linear_iterations": max(
            100, int(record["true_trace_unknowns"])
        ),
        "krylov_restart_dimension": min(
            100, max(20, int(record["true_trace_unknowns"]))
        ),
        "maximum_line_search_trials": 13,
        "regularization": mode if mode == "igr" else "none",
    }
    for field, expected in exact_policy.items():
        if policy.get(field) != expected:
            raise CampaignError(
                f"numerical policy {field}={policy.get(field)!r}; expected {expected!r}"
            )
    expected_reals = {
        "nonlinear_relative_tolerance": 1.0e-10,
        "newton_correction_linear_relative_tolerance": 1.0e-12,
        "final_coherent_linear_relative_tolerance": 1.0e-12,
        "true_residual_tightening_linear_relative_tolerance": 1.0e-12,
        "true_residual_relative_tolerance": 1.0e-10,
        "conservation_relative_tolerance": 1.0e-10,
        "armijo_constant": 1.0e-4,
        "igr_alpha": 0.2 if mode == "igr" else 0.0,
        "igr_beta": 0.8 if mode == "igr" else 0.0,
    }
    for field, expected in expected_reals.items():
        if not math.isclose(
                checked_float(policy.get(field), f"numerical_policy.{field}"),
                expected, rel_tol=0.0, abs_tol=0.0):
            raise CampaignError(f"numerical policy {field} differs from control")
    return volume_work


def validate_record(record: Any, *, source: str, mode: str, order: int,
                    mesh_n: int, domain_periods: int, timestep: float,
                    export: bool, engine: str | None,
                    expected_build: dict[str, str] | None = None,
                    expected_gpu: dict[str, Any] | None = None,
                    trace_policy: str = "engine-default") -> dict[str, Any]:
    if not isinstance(record, dict):
        raise CampaignError("benchmark output is not a JSON object")
    schema = record.get("schema")
    if source == "final" and schema != 2:
        raise CampaignError(f"final-tree benchmark did not emit schema 2: {schema}")
    if source == "frozen" and schema != 1:
        raise CampaignError(
            f"frozen benchmark must be the exact schema-1 historical harness: {schema}"
        )
    if record.get("success") is not True:
        raise CampaignError("benchmark reported failure")
    if record.get("precision") != "fp64":
        raise CampaignError("accepted-step evidence must use FP64 external state")
    expected = {
        "mode": mode,
        "polynomial_order": order,
        "mesh_n": mesh_n,
        "domain_periods": domain_periods,
        "host_export": export,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise CampaignError(
                f"benchmark field {field}={record.get(field)!r}; expected {value!r}"
            )
    if record.get("rank_count") not in (None, 1):
        raise CampaignError("accepted-step benchmark must use exactly one rank")
    if not math.isclose(checked_float(record.get("dt"), "dt"), timestep,
                        rel_tol=1.0e-13, abs_tol=0.0):
        raise CampaignError("benchmark timestep differs from the requested timestep")
    if engine is not None and record.get("cuda_engine") != engine:
        raise CampaignError(
            f"benchmark cuda_engine={record.get('cuda_engine')!r}; expected {engine!r}"
        )
    if source == "final" and record.get("trace_policy") != trace_policy:
        raise CampaignError(
            f"benchmark trace_policy={record.get('trace_policy')!r}; "
            f"expected {trace_policy!r}"
        )
    if source == "final":
        resolved_trace_policy = expected_resolved_trace_policy(
            engine, mode, trace_policy, order=order
        )
        if record.get("requested_trace_policy") != trace_policy or \
                record.get("resolved_trace_policy") != resolved_trace_policy:
            raise CampaignError(
                "benchmark requested/resolved trace-policy provenance is "
                "inconsistent"
            )
        production_selectable = trace_policy == "engine-default"
        expected_variant = "production" if production_selectable else "ablation"
        if record.get("variant") != expected_variant or \
                record.get("production_selectable") is not production_selectable:
            raise CampaignError(
                "benchmark production/ablation provenance does not match "
                f"trace policy {trace_policy!r}"
            )
        if expected_build is None or expected_gpu is None:
            raise CampaignError("final benchmark validation lacks build/GPU binding")
        if record.get("benchmark") != FINAL_BENCHMARK_NAME or \
                record.get("backend") != "cuda" or record.get("rank_count") != 1:
            raise CampaignError("final-tree benchmark identity is invalid")
        gpu_name = record.get("gpu_name")
        if not isinstance(gpu_name, str) or not gpu_name.strip():
            raise CampaignError("final-tree benchmark has no GPU name")
        if checked_int(record.get("gpu_compute_capability"),
                       "gpu_compute_capability") != \
                FINAL_GPU_COMPUTE_CAPABILITY:
            raise CampaignError("final-tree benchmark did not execute on SM120")
        checked_int(record.get("gpu_sm_count"), "gpu_sm_count", 1)
        checked_int(record.get("cuda_driver_version"),
                    "cuda_driver_version", 1)
        checked_int(record.get("cuda_runtime_version"),
                    "cuda_runtime_version", 1)
        validate_embedded_final_identity(record, expected_build, expected_gpu)
        volume_work = validate_final_policy(
            record, mode=mode, order=order
        )
        if record.get("initialization_in_primary_timing") is not False:
            raise CampaignError("benchmark initialization timing scope is invalid")
        initialization = checked_float(
            record.get("initialization_duration_seconds"),
            "initialization_duration_seconds",
        )
        if initialization <= 0.0:
            raise CampaignError("benchmark initialization duration is not positive")
        expected_scope = (
            "Prepare+CapturePreparedObservation+"
            + ("ExportPreparedInteriorD2H+" if export else "")
            + "AcknowledgePreparedOutput+Commit"
        )
        if record.get("timing_scope") != expected_scope:
            raise CampaignError("benchmark accepted-step timing scope is invalid")
        if record.get("warmup_steps") != FINAL_WARMUP_STEPS or \
                record.get("measured_steps") != FINAL_MEASURED_STEPS:
            raise CampaignError("benchmark warmup/measured policy is invalid")
        validate_memory_controls(record)
        expected_preconditioner = expected_trace_preconditioner(
            engine, mode, trace_policy, order=order
        )
        expected_trace_solver = expected_trace_solver_configuration(
            engine, mode, trace_policy
        )[0]
        if record.get("trace_solver") != expected_trace_solver:
            raise CampaignError(
                f"benchmark trace_solver={record.get('trace_solver')!r}; "
                f"expected {expected_trace_solver!r} for rank-one {engine!r}"
            )
        if record.get("trace_preconditioner") != expected_preconditioner:
            raise CampaignError(
                "benchmark trace_preconditioner="
                f"{record.get('trace_preconditioner')!r}; expected "
                f"{expected_preconditioner!r} for {engine!r}/{mode!r}"
            )
        expected_ilu = expected_trace_ilu_policy(
            engine, mode, trace_policy
        )
        observed_ilu = (
            checked_policy_int(record.get("trace_ilu_type"),
                               "trace_ilu_type"),
            checked_policy_int(record.get("trace_ilu_level_of_fill"),
                               "trace_ilu_level_of_fill"),
            checked_policy_int(record.get("trace_ilu_local_reordering"),
                               "trace_ilu_local_reordering"),
            checked_policy_int(record.get("trace_ilu_application_iterations"),
                               "trace_ilu_application_iterations"),
        )
        if observed_ilu != expected_ilu:
            raise CampaignError(
                f"benchmark trace ILU policy={observed_ilu!r}; expected "
                f"{expected_ilu!r} for {engine!r}/{mode!r}"
            )
        expected_triangular = expected_trace_ilu_triangular_solve_policy(
            engine, mode, trace_policy
        )
        observed_triangular = (
            record.get("trace_ilu_triangular_solve_kind"),
            checked_int(
                record.get("trace_ilu_lower_jacobi_iterations"),
                "trace_ilu_lower_jacobi_iterations",
            ),
            checked_int(
                record.get("trace_ilu_upper_jacobi_iterations"),
                "trace_ilu_upper_jacobi_iterations",
            ),
        )
        if observed_triangular != expected_triangular[:3]:
            raise CampaignError(
                "benchmark trace ILU triangular-solve policy="
                f"{observed_triangular!r}; expected "
                f"{expected_triangular[:3]!r} for {engine!r}/{mode!r}"
            )
        expected_accounting_kind = (
            "derived_bounds_v1" if expected_ilu[3] else "unavailable"
        )
        if record.get("trace_ilu_accounting_kind") != \
                expected_accounting_kind:
            raise CampaignError(
                "benchmark trace ILU accounting kind="
                f"{record.get('trace_ilu_accounting_kind')!r}; expected "
                f"{expected_accounting_kind!r} for {engine!r}/{mode!r}"
            )
        expected_factor_reuse = expected_trace_ilu_factor_reuse_policy(
            engine, mode, trace_policy
        )
        if record.get("trace_ilu_factor_reuse_policy") != \
                expected_factor_reuse:
            raise CampaignError(
                "benchmark trace ILU factor-reuse policy="
                f"{record.get('trace_ilu_factor_reuse_policy')!r}; expected "
                f"{expected_factor_reuse!r} for {engine!r}/{mode!r}"
            )
        expected_configuration = trace_policy_configuration(
            engine, mode, trace_policy, order=order
        )
        observed_configuration = record_trace_policy_configuration(record)
        if observed_configuration != expected_configuration:
            raise CampaignError(
                "benchmark trace-policy metadata is not the exact selected "
                f"configuration: observed={observed_configuration!r}; "
                f"expected={expected_configuration!r}"
            )
        expected_sigma_jacobi_sweeps, _, fixed_sigma_warp = \
            fixed_sigma_jacobi_policy(resolved_trace_policy)
        warp_field_split = resolved_trace_policy in {
            "native-element-schwarz-matrix-free-warp-"
            "equilibrated-field-split-fp64",
        } or fixed_sigma_warp
        scalar_field_split = resolved_trace_policy in {
            "native-element-schwarz-equilibrated-field-split-fp64",
            "native-element-schwarz-matrix-free-"
            "equilibrated-field-split-fp64",
        } or (expected_sigma_jacobi_sweeps > 0 and not fixed_sigma_warp)
        full_patch = resolved_trace_policy in {
            "native-element-schwarz-fp64",
            "native-element-schwarz-matrix-free-fp64",
        }
        expected_schwarz_apply = (
            "cooperative-pivoted-lu-fp64" if warp_field_split else
            "scalar-pivoted-lu-fp64" if scalar_field_split else
            "scalar-full-patch-pivoted-lu-fp64" if full_patch else
            "not-applicable"
        )
        expected_tile_width = (
            16 if warp_field_split and order == 1 else
            32 if warp_field_split else 0
        )
        if record.get("trace_element_schwarz_apply") != \
                expected_schwarz_apply or checked_int(
                    record.get("trace_element_schwarz_apply_tile_width"),
                    "trace_element_schwarz_apply_tile_width",
                ) != expected_tile_width:
            raise CampaignError(
                "benchmark element-Schwarz apply metadata does not match "
                f"the resolved trace policy {resolved_trace_policy!r}"
            )
        expected_sigma_jacobi_damping = \
            2.0 / 3.0 if expected_sigma_jacobi_sweeps else 0.0
        if checked_int(
                record.get("trace_field_split_sigma_jacobi_sweeps"),
                "trace_field_split_sigma_jacobi_sweeps",
            ) != expected_sigma_jacobi_sweeps or not math.isclose(
                checked_float(
                    record.get("trace_field_split_sigma_jacobi_damping"),
                    "trace_field_split_sigma_jacobi_damping",
                ),
                expected_sigma_jacobi_damping,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
            raise CampaignError(
                "benchmark fixed-sigma Jacobi metadata does not match "
                f"the resolved trace policy {resolved_trace_policy!r}"
            )
    if source == "frozen":
        frozen_exact = {
            "benchmark": "dg_igr_hdg_cuda_accepted_step_fp64",
            "rank_count": 1,
            "initialization_timed": False,
            "warmup_steps": FINAL_WARMUP_STEPS,
            "measured_steps": FINAL_MEASURED_STEPS,
            "principal_threads": 128,
            "principal_element_threads": 4,
            "principal_elements_per_block": 32,
            "principal_execution": "four_lane_subwarp_per_element",
        }
        for field, value in frozen_exact.items():
            if record.get(field) != value:
                raise CampaignError(
                    f"frozen benchmark field {field}={record.get(field)!r}; "
                    f"expected {value!r}"
                )
        expected_scope = (
            "Prepare+CapturePreparedObservation+"
            + ("ExportPreparedInteriorD2H+" if export else "")
            + "AcknowledgePreparedOutput+Commit"
        )
        if record.get("timing_scope") != expected_scope:
            raise CampaignError("frozen accepted-step timing scope is invalid")
        elements = checked_int(record.get("elements"), "frozen.elements", 1)
        if elements != mesh_n * mesh_n or \
                record.get("principal_blocks") != (elements + 31) // 32:
            raise CampaignError("frozen workload geometry is inconsistent")
        checked_int(record.get("interior_unknowns"),
                    "frozen.interior_unknowns", 1)
        checked_int(record.get("true_trace_unknowns"),
                    "frozen.true_trace_unknowns", 1)
        checked_int(record.get("gpu_sm_count"), "frozen.gpu_sm_count", 1)

    durations = record.get("raw_durations_seconds")
    if not isinstance(durations, list) or not durations:
        raise CampaignError("benchmark has no raw measured durations")
    checked_durations = [checked_float(value, "raw_durations_seconds")
                         for value in durations]
    if any(value <= 0.0 for value in checked_durations):
        raise CampaignError("benchmark durations must be positive")

    samples = record.get("samples")
    if source == "final":
        if len(checked_durations) != FINAL_MEASURED_STEPS:
            raise CampaignError("benchmark raw measured-duration count is invalid")
        if not isinstance(samples, list) or len(samples) != \
                FINAL_WARMUP_STEPS + FINAL_MEASURED_STEPS:
            raise CampaignError("schema-2 benchmark sample count is invalid")
        numerical_policy = record.get("numerical_policy")
        if not isinstance(numerical_policy, dict):
            raise CampaignError("benchmark lacks its numerical policy")
        krylov_restart_dimension = checked_uint64(
            numerical_policy.get("krylov_restart_dimension"),
            "numerical_policy.krylov_restart_dimension",
        )
        if krylov_restart_dimension == 0:
            raise CampaignError("Krylov restart dimension must be positive")
        newton_linear_tolerance = checked_float(
            numerical_policy.get(
                "newton_correction_linear_relative_tolerance"
            ),
            "numerical_policy.newton_correction_linear_relative_tolerance",
        )
        final_linear_tolerance = checked_float(
            numerical_policy.get(
                "final_coherent_linear_relative_tolerance"
            ),
            "numerical_policy.final_coherent_linear_relative_tolerance",
        )
        tightening_linear_tolerance = checked_float(
            numerical_policy.get(
                "true_residual_tightening_linear_relative_tolerance"
            ),
            "numerical_policy."
            "true_residual_tightening_linear_relative_tolerance",
        )
        true_residual_tightening_enabled = (
            tightening_linear_tolerance < newton_linear_tolerance or
            tightening_linear_tolerance < final_linear_tolerance
        )
        validate_final_sample(
            samples[0], engine=engine, mode=mode, order=order,
            trace_policy=trace_policy,
            expected_preconditioner=expected_preconditioner,
            volume_mma_tiles_per_assembly=
                volume_work["mma_tiles_per_assembly"],
            krylov_restart_dimension=krylov_restart_dimension,
            true_residual_tightening_enabled=
                true_residual_tightening_enabled,
            expected_phase="warmup", expected_index=-1,
            expected_duration=None,
        )
        if not math.isclose(
                checked_float(record.get("warmup_duration_seconds"),
                              "warmup_duration_seconds"),
                checked_float(samples[0].get("duration_seconds"),
                              "samples.duration_seconds"),
                rel_tol=1.0e-13, abs_tol=0.0):
            raise CampaignError("warmup duration differs from sample evidence")
        for index, duration in enumerate(checked_durations):
            validate_final_sample(
                samples[index + FINAL_WARMUP_STEPS], engine=engine, mode=mode,
                order=order,
                trace_policy=trace_policy,
                expected_preconditioner=expected_preconditioner,
                volume_mma_tiles_per_assembly=
                    volume_work["mma_tiles_per_assembly"],
                krylov_restart_dimension=krylov_restart_dimension,
                true_residual_tightening_enabled=
                    true_residual_tightening_enabled,
                expected_phase="measured", expected_index=index,
                expected_duration=duration,
            )
        reported_median = checked_float(record.get("median_seconds"),
                                        "median_seconds")
        reported_stddev = checked_float(
            record.get("population_stddev_seconds"),
            "population_stddev_seconds",
        )
        if not math.isclose(reported_median, statistics.median(checked_durations),
                            rel_tol=1.0e-13, abs_tol=0.0) or \
                not math.isclose(reported_stddev,
                                 statistics.pstdev(checked_durations),
                                 rel_tol=1.0e-12, abs_tol=1.0e-18):
            raise CampaignError("benchmark timing summary differs from raw samples")
    else:
        if len(checked_durations) != FINAL_MEASURED_STEPS or \
                not isinstance(samples, list) or len(samples) != \
                FINAL_WARMUP_STEPS + FINAL_MEASURED_STEPS:
            raise CampaignError("frozen sample/duration count is invalid")
        validate_frozen_sample(
            samples[0], expected_phase="warmup", expected_index=-1,
            expected_duration=None,
        )
        if not math.isclose(
                checked_float(record.get("warmup_duration_seconds"),
                              "frozen.warmup_duration_seconds"),
                checked_float(samples[0].get("duration_seconds"),
                              "frozen.samples.duration_seconds"),
                rel_tol=1.0e-13, abs_tol=0.0):
            raise CampaignError("frozen warmup duration differs from sample")
        for index, duration in enumerate(checked_durations):
            validate_frozen_sample(
                samples[index + FINAL_WARMUP_STEPS],
                expected_phase="measured", expected_index=index,
                expected_duration=duration,
            )
        if not math.isclose(
                checked_float(record.get("median_seconds"),
                              "frozen.median_seconds"),
                statistics.median(checked_durations),
                rel_tol=1.0e-13, abs_tol=0.0) or not math.isclose(
                    checked_float(record.get("population_stddev_seconds"),
                                  "frozen.population_stddev_seconds"),
                    statistics.pstdev(checked_durations),
                    rel_tol=1.0e-12, abs_tol=1.0e-18):
            raise CampaignError("frozen timing summary differs from raw samples")
    return record


def summarized_sample_work(sample: dict[str, Any], *, schema: int) -> dict[str, Any]:
    """Retain only fields that exist in the selected benchmark schema."""
    result = {
        "measured_index": sample["measured_index"],
        "nonlinear_updates": sample["nonlinear_updates"],
        "trace_solves": sample["trace_solves"],
        "linear_iterations": sample["linear_iterations"],
        "line_search_trials": sample["line_search_trials"],
        "rejected_line_search_trials": sample[
            "rejected_line_search_trials"
        ],
        "transfers": sample["transfers"],
        "gates": sample["gates"],
    }
    if schema == 1:
        return result
    if schema != 2:
        raise CampaignError(f"cannot summarize unknown benchmark schema {schema!r}")
    tensor = sample["tensor_core"]
    result.update({
        "attempt_sequence": sample["attempt_sequence"],
        "retry_index": sample["retry_index"],
        "device_ticket_count": sample["device_ticket_count"],
        "linear_solve_attempts": sample["linear_solve_attempts"],
        "newton_correction_linear_iterations": sample[
            "newton_correction_linear_iterations"
        ],
        "final_coherent_linear_iterations": sample[
            "final_coherent_linear_iterations"
        ],
        "true_residual_tightenings": sample["true_residual_tightenings"],
        "iterative_refinement_steps": sample["iterative_refinement_steps"],
        "residual_replacements": sample["residual_replacements"],
        "precision_corrections": sample["precision_corrections"],
        "product_generations": sample["product_generations"],
        "selected_engine": tensor["selected_engine"],
        "requested_trace_policy": tensor["requested_trace_policy"],
        "resolved_trace_policy": tensor["resolved_trace_policy"],
        "trace_solver": tensor["trace_solver"],
        "native_rank1_trace_solve_calls": tensor[
            "native_rank1_trace_solve_calls"
        ],
        "hypre_trace_solve_calls": tensor["hypre_trace_solve_calls"],
        "native_rank1_trace_solver_configured": tensor[
            "native_rank1_trace_solver_configured"
        ],
        "hypre_mgr_rank1_trace_solver_configured": tensor[
            "hypre_mgr_rank1_trace_solver_configured"
        ],
        "native_element_schwarz_configured": tensor[
            "native_element_schwarz_configured"
        ],
        "native_matrix_free_trace_operator_configured": tensor[
            "native_matrix_free_trace_operator_configured"
        ],
        "native_matrix_free_trace_operator_applications": tensor[
            "native_matrix_free_trace_operator_applications"
        ],
        "hypre_host_staged_rank2_trace_solver_configured": tensor[
            "hypre_host_staged_rank2_trace_solver_configured"
        ],
        "trace_solver_accounting_complete": tensor[
            "trace_solver_accounting_complete"
        ],
        "trace_preconditioner": tensor["trace_preconditioner"],
        "trace_preconditioner_configured": tensor[
            "trace_preconditioner_configured"
        ],
        "ilu_type": tensor["ilu_type"],
        "ilu_level_of_fill": tensor["ilu_level_of_fill"],
        "ilu_local_reordering": tensor["ilu_local_reordering"],
        "ilu_application_iterations": tensor["ilu_application_iterations"],
        "ilu_triangular_solve_kind": tensor[
            "ilu_triangular_solve_kind"
        ],
        "ilu_lower_jacobi_iterations": tensor[
            "ilu_lower_jacobi_iterations"
        ],
        "ilu_upper_jacobi_iterations": tensor[
            "ilu_upper_jacobi_iterations"
        ],
        "ilu_factor_reuse_policy": tensor["ilu_factor_reuse_policy"],
        "ilu_factor_setups": tensor["ilu_factor_setups"],
        "ilu_factor_reuses": tensor["ilu_factor_reuses"],
        "ilu_accounting_kind": tensor["ilu_accounting_kind"],
        "ilu_accounting_complete": tensor["ilu_accounting_complete"],
        "ilu_preconditioner_callbacks_lower_bound": tensor[
            "ilu_preconditioner_callbacks_lower_bound"
        ],
        "ilu_preconditioner_callbacks_upper_bound": tensor[
            "ilu_preconditioner_callbacks_upper_bound"
        ],
        "ilu_inner_iterations_lower_bound": tensor[
            "ilu_inner_iterations_lower_bound"
        ],
        "ilu_inner_iterations_upper_bound": tensor[
            "ilu_inner_iterations_upper_bound"
        ],
        "ilu_jacobi_factor_spmv_updates_lower_bound": tensor[
            "ilu_jacobi_factor_spmv_updates_lower_bound"
        ],
        "ilu_jacobi_factor_spmv_updates_upper_bound": tensor[
            "ilu_jacobi_factor_spmv_updates_upper_bound"
        ],
        "ilu_factor_age_generations": tensor["ilu_factor_age_generations"],
        "equilibration_builds": tensor["equilibration_builds"],
        "equilibration_passes": tensor["equilibration_passes"],
        "mgr_setup_calls": tensor["mgr_setup_calls"],
        "mgr_preconditioner_applications": tensor[
            "mgr_preconditioner_applications"
        ],
        "mgr_f_ilu_setups": tensor["mgr_f_ilu_setups"],
        "mgr_f_ilu_applications": tensor["mgr_f_ilu_applications"],
        "element_schwarz_patch_setups": tensor[
            "element_schwarz_patch_setups"
        ],
        "element_schwarz_preconditioner_applications": tensor[
            "element_schwarz_preconditioner_applications"
        ],
        "field_split_sigma_jacobi_sweeps": tensor[
            "field_split_sigma_jacobi_sweeps"
        ],
        "field_split_sigma_jacobi_damping": tensor[
            "field_split_sigma_jacobi_damping"
        ],
        "field_split_sigma_jacobi_updates": tensor[
            "field_split_sigma_jacobi_updates"
        ],
        "element_schwarz_patch_rows": tensor[
            "element_schwarz_patch_rows"
        ],
        "element_schwarz_minimum_multiplicity": tensor[
            "element_schwarz_minimum_multiplicity"
        ],
        "element_schwarz_maximum_multiplicity": tensor[
            "element_schwarz_maximum_multiplicity"
        ],
        "minimum_left_scale": tensor["minimum_left_scale"],
        "maximum_left_scale": tensor["maximum_left_scale"],
        "minimum_right_scale": tensor["minimum_right_scale"],
        "maximum_right_scale": tensor["maximum_right_scale"],
        "hardware_supported": tensor["hardware_supported"],
        "volume_assembly_launches": tensor["volume_assembly_launches"],
        "volume_mma_tiles": tensor["volume_mma_tiles"],
        "condensation_launches": tensor["condensation_launches"],
        "mma_tiles": tensor["mma_tiles"],
        "fallback_count": tensor["fallback_count"],
        "fallback_accounting_complete": tensor[
            "fallback_accounting_complete"
        ],
        "no_fallback": tensor["no_fallback"],
    })
    return result


def record_summary(record: dict[str, Any]) -> dict[str, Any]:
    durations = [float(value) for value in record["raw_durations_seconds"]]
    schema = record["schema"]
    if schema not in (1, 2):
        raise CampaignError(f"cannot summarize unknown benchmark schema {schema!r}")
    sample_work = [
        summarized_sample_work(sample, schema=schema)
        for sample in record.get("samples", [])
        if isinstance(sample, dict) and sample.get("phase") == "measured"
    ]
    workload_fields = (
        "mode", "polynomial_order", "mesh_n", "domain_periods",
        "domain_length", "cells_per_period", "element_h", "elements",
        "interior_unknowns", "true_trace_unknowns", "principal_blocks",
        "gpu_sm_count", "dt", "host_export",
    ) if schema == 1 else (
        "mode", "polynomial_order", "mesh_n", "domain_periods",
        "domain_length", "cells_per_period", "element_h", "elements",
        "interior_components", "trace_components", "interior_unknowns",
        "true_trace_unknowns", "principal_blocks", "gpu_sm_count",
        "initial_state_kind", "initial_trace_kind", "source_kind",
        "boundary_kind", "time_integrator", "time_step_policy", "dt",
        "host_export", "profiling_observability",
    )
    summary: dict[str, Any] = {
        "schema": schema,
        "workload_identity": {
            field: record[field] for field in workload_fields
        },
        "median_seconds_recomputed": statistics.median(durations),
        "raw_duration_count": len(durations),
        "measured_sample_work": sample_work,
        "measured_linear_iterations_total": sum(
            item["linear_iterations"] for item in sample_work
            if isinstance(item["linear_iterations"], int)
        ),
        "gpu_sm_count": record["gpu_sm_count"],
        "timing_scope": record["timing_scope"],
    }
    if schema == 1:
        summary["initialization_timed"] = record["initialization_timed"]
    else:
        summary.update({
            "trace_policy": record["trace_policy"],
            "requested_trace_policy": record["requested_trace_policy"],
            "resolved_trace_policy": record["resolved_trace_policy"],
            "trace_solver": record["trace_solver"],
            "precision_policy": record["precision_policy"],
            "numerical_policy": record["numerical_policy"],
            "initialization_duration_seconds": record[
                "initialization_duration_seconds"
            ],
            "initialization_in_primary_timing": record[
                "initialization_in_primary_timing"
            ],
            "device_memory_snapshots": record["device_memory_snapshots"],
            "umpire_allocator_high_watermarks_after_samples": record[
                "umpire_allocator_high_watermarks_after_samples"
            ],
            "gpu_name": record["gpu_name"],
            "gpu_compute_capability": record["gpu_compute_capability"],
            "cuda_driver_version": record["cuda_driver_version"],
            "cuda_runtime_version": record["cuda_runtime_version"],
        })
        summary["trace_preconditioner"] = record["trace_preconditioner"]
        summary["trace_field_split_sigma_jacobi_policy"] = {
            "sweeps": record["trace_field_split_sigma_jacobi_sweeps"],
            "damping": record["trace_field_split_sigma_jacobi_damping"],
        }
        summary["trace_ilu_policy"] = {
            "type": record["trace_ilu_type"],
            "level_of_fill": record["trace_ilu_level_of_fill"],
            "local_reordering": record["trace_ilu_local_reordering"],
            "application_iterations": record[
                "trace_ilu_application_iterations"
            ],
            "triangular_solve_kind": record[
                "trace_ilu_triangular_solve_kind"
            ],
            "lower_jacobi_iterations": record[
                "trace_ilu_lower_jacobi_iterations"
            ],
            "upper_jacobi_iterations": record[
                "trace_ilu_upper_jacobi_iterations"
            ],
            "accounting_kind": record["trace_ilu_accounting_kind"],
            "factor_reuse_policy": record[
                "trace_ilu_factor_reuse_policy"
            ],
        }
    return summary


def measured_samples(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for record in records:
        for sample in record.get("samples", []):
            if isinstance(sample, dict) and sample.get("phase") == "measured":
                result.append(sample)
    return result


def work_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    record_list = list(records)
    schemas = {record.get("schema") for record in record_list}
    if not record_list or len(schemas) != 1 or not schemas.issubset({1, 2}):
        raise CampaignError(
            "solver-work summary requires nonempty homogeneous schema-1 or "
            "schema-2 records"
        )
    schema = next(iter(schemas))
    samples = measured_samples(record_list)
    fields = (
        "nonlinear_updates",
        "trace_solves",
        "linear_iterations",
        "linear_solve_attempts",
        "newton_correction_linear_iterations",
        "final_coherent_linear_iterations",
        "true_residual_tightenings",
        "line_search_trials",
        "rejected_line_search_trials",
        "retry_index",
        "device_ticket_count",
        "attempt_sequence",
        "iterative_refinement_steps",
        "residual_replacements",
        "precision_corrections",
    )
    summary: dict[str, Any] = {
        "linear_iterations_by_invocation": [
            [sample.get("linear_iterations")
             for sample in record.get("samples", [])
             if isinstance(sample, dict) and sample.get("phase") == "measured"]
            for record in record_list
        ],
    }
    if schema == 2:
        summary["trace_solvers"] = sorted({
            record["trace_solver"] for record in record_list
            if isinstance(record.get("trace_solver"), str)
        })
        summary["trace_policies"] = sorted({
            record["trace_policy"] for record in record_list
            if isinstance(record.get("trace_policy"), str)
        })
        summary["requested_trace_policies"] = sorted({
            record["requested_trace_policy"] for record in record_list
            if isinstance(record.get("requested_trace_policy"), str)
        })
        summary["resolved_trace_policies"] = sorted({
            record["resolved_trace_policy"] for record in record_list
            if isinstance(record.get("resolved_trace_policy"), str)
        })
        summary["trace_preconditioners"] = sorted({
            record["trace_preconditioner"] for record in record_list
            if isinstance(record.get("trace_preconditioner"), str)
        })
    for field in fields:
        values = [sample.get(field) for sample in samples]
        if values and all(isinstance(value, int) and not isinstance(value, bool)
                          for value in values):
            summary[field] = {
                "count": len(values),
                "minimum": min(values),
                "median": statistics.median(values),
                "maximum": max(values),
                "total": sum(values),
            }
    tensor_fields = (
        "ilu_factor_setups",
        "ilu_factor_reuses",
        "ilu_preconditioner_callbacks_lower_bound",
        "ilu_preconditioner_callbacks_upper_bound",
        "ilu_inner_iterations_lower_bound",
        "ilu_inner_iterations_upper_bound",
        "ilu_jacobi_factor_spmv_updates_lower_bound",
        "ilu_jacobi_factor_spmv_updates_upper_bound",
        "native_rank1_trace_solve_calls",
        "hypre_trace_solve_calls",
        "equilibration_builds",
        "equilibration_passes",
        "mgr_setup_calls",
        "mgr_preconditioner_applications",
        "mgr_f_ilu_setups",
        "mgr_f_ilu_applications",
        "element_schwarz_patch_setups",
        "element_schwarz_preconditioner_applications",
        "field_split_sigma_jacobi_sweeps",
        "field_split_sigma_jacobi_updates",
        "element_schwarz_patch_rows",
        "element_schwarz_minimum_multiplicity",
        "element_schwarz_maximum_multiplicity",
        "volume_assembly_launches",
        "volume_mma_tiles",
        "condensation_launches",
        "mma_tiles",
        "fallback_count",
    )
    if schema == 2:
        tensor_samples = [sample["tensor_core"] for sample in samples]
        summary["tensor_core"] = {
            "selected_engines": sorted({
                tensor.get("selected_engine") for tensor in tensor_samples
                if isinstance(tensor.get("selected_engine"), str)
            }),
            "requested_trace_policies": sorted({
                tensor.get("requested_trace_policy")
                for tensor in tensor_samples
                if isinstance(tensor.get("requested_trace_policy"), str)
            }),
            "resolved_trace_policies": sorted({
                tensor.get("resolved_trace_policy")
                for tensor in tensor_samples
                if isinstance(tensor.get("resolved_trace_policy"), str)
            }),
            "trace_solvers": sorted({
                tensor.get("trace_solver") for tensor in tensor_samples
                if isinstance(tensor.get("trace_solver"), str)
            }),
            "native_rank1_trace_solver_configured_values": sorted({
                tensor.get("native_rank1_trace_solver_configured")
                for tensor in tensor_samples
                if isinstance(
                    tensor.get("native_rank1_trace_solver_configured"), bool
                )
            }),
            "hypre_mgr_rank1_trace_solver_configured_values": sorted({
                tensor.get("hypre_mgr_rank1_trace_solver_configured")
                for tensor in tensor_samples
                if isinstance(
                    tensor.get("hypre_mgr_rank1_trace_solver_configured"), bool
                )
            }),
            "native_element_schwarz_configured_values": sorted({
                tensor.get("native_element_schwarz_configured")
                for tensor in tensor_samples
                if isinstance(
                    tensor.get("native_element_schwarz_configured"), bool
                )
            }),
            "hypre_host_staged_rank2_trace_solver_configured_values": sorted({
                tensor.get("hypre_host_staged_rank2_trace_solver_configured")
                for tensor in tensor_samples
                if isinstance(
                    tensor.get(
                        "hypre_host_staged_rank2_trace_solver_configured"
                    ), bool
                )
            }),
            "trace_solver_accounting_all_complete": bool(tensor_samples) and all(
                tensor.get("trace_solver_accounting_complete") is True
                for tensor in tensor_samples
            ),
            "ilu_policies": sorted({
                (
                    tensor.get("ilu_type"),
                    tensor.get("ilu_level_of_fill"),
                    tensor.get("ilu_local_reordering"),
                    tensor.get("ilu_application_iterations"),
                )
                for tensor in tensor_samples
            }),
            "ilu_accounting_kinds": sorted({
                tensor.get("ilu_accounting_kind") for tensor in tensor_samples
                if isinstance(tensor.get("ilu_accounting_kind"), str)
            }),
            "ilu_triangular_solve_policies": sorted({
                (
                    tensor.get("ilu_triangular_solve_kind"),
                    tensor.get("ilu_lower_jacobi_iterations"),
                    tensor.get("ilu_upper_jacobi_iterations"),
                )
                for tensor in tensor_samples
            }),
            "ilu_factor_reuse_policies": sorted({
                tensor.get("ilu_factor_reuse_policy")
                for tensor in tensor_samples
                if isinstance(tensor.get("ilu_factor_reuse_policy"), str)
            }),
            "ilu_accounting_all_complete": bool(tensor_samples) and all(
                tensor.get("ilu_accounting_complete") is True
                for tensor in tensor_samples
            ),
            "all_no_fallback": bool(tensor_samples) and all(
                tensor.get("no_fallback") is True and
                tensor.get("fallback_count") == 0 and
                tensor.get("fallback_accounting_complete") is True
                for tensor in tensor_samples
            ),
        }
        for field in tensor_fields:
            values = [tensor.get(field) for tensor in tensor_samples]
            if values and all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in values):
                summary["tensor_core"][field] = {
                    "count": len(values),
                    "minimum": min(values),
                    "median": statistics.median(values),
                    "maximum": max(values),
                    "total": sum(values),
                }
        factor_ages = [
            tensor.get("ilu_factor_age_generations")
            for tensor in tensor_samples
        ]
        if factor_ages and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in factor_ages):
            # Each value is already a per-attempt maximum, so it is a
            # distribution rather than additive work.
            summary["tensor_core"]["ilu_factor_age_generations"] = {
                "count": len(factor_ages),
                "minimum": min(factor_ages),
                "median": statistics.median(factor_ages),
                "maximum": max(factor_ages),
            }
        for field in (
                "minimum_left_scale",
                "maximum_left_scale",
                "minimum_right_scale",
                "maximum_right_scale"):
            values = [tensor.get(field) for tensor in tensor_samples]
            if values and all(
                    isinstance(value, (int, float)) and
                    not isinstance(value, bool) and math.isfinite(float(value))
                    for value in values):
                # These extrema are already one bounded aggregate per attempt;
                # summarize their distribution rather than adding them.
                summary["tensor_core"][field] = {
                    "count": len(values),
                    "minimum": min(values),
                    "median": statistics.median(values),
                    "maximum": max(values),
                }
    transfer_fields = (
        "host_to_device_bytes",
        "device_to_host_bytes",
        "device_to_device_bytes",
        "mpi_staging_bytes",
        "synchronization_events",
        "event_count",
    )
    transfers = [sample.get("transfers", {}) for sample in samples]
    summary["transfers"] = {
        "all_complete": bool(transfers) and all(
            transfer.get("complete") is True for transfer in transfers
        ),
        "complete_count": sum(
            transfer.get("complete") is True for transfer in transfers
        ),
        "sample_count": len(transfers),
    }
    for field in transfer_fields:
        values = [transfer.get(field) for transfer in transfers]
        if values and all(isinstance(value, int) and not isinstance(value, bool)
                          for value in values):
            summary["transfers"][field] = {
                "minimum": min(values),
                "median": statistics.median(values),
                "maximum": max(values),
                "total": sum(values),
            }
    required_gates = (
        "candidate_ready",
        "finite",
        "admissible",
        "recovery",
        "true_residual",
        "conservation",
        "telemetry",
        "products_coupled_and_ordered",
        "success",
    )
    gates = [sample.get("gates", {}) for sample in samples]
    summary["acceptance_gates"] = {
        field: bool(gates) and all(gate.get(field) is True for gate in gates)
        for field in required_gates
    }
    return summary


def memory_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    record_list = list(records)
    initialization = [
        checked_float(record.get("initialization_duration_seconds"),
                      "initialization_duration_seconds")
        for record in record_list
    ]
    snapshot_names = (
        "after_device_selection_before_problem",
        "after_initialization",
        "after_samples",
    )
    device: dict[str, Any] = {}
    all_observed_used: list[float] = []
    for name in snapshot_names:
        values = [
            float(record["device_memory_snapshots"][name]["used_bytes"])
            for record in record_list
        ]
        device[name] = sample_statistics(values)
        all_observed_used.extend(values)
    umpire_sum = [
        float(record["umpire_allocator_high_watermarks_after_samples"]
              ["sum_bytes"])
        for record in record_list
        if record["umpire_allocator_high_watermarks_after_samples"]
           .get("sum_available") is True
    ]
    return {
        "initialization_duration_seconds": sample_statistics(initialization),
        "memory_measurement_scope": (
            "three cudaMemGetInfo observations plus allocator-scoped Umpire "
            "high-watermarks; not a global accepted-step peak"
        ),
        "device_used_bytes": device,
        "maximum_observed_snapshot_device_used_bytes": max(all_observed_used),
        "umpire_sum_high_watermark_bytes":
            sample_statistics(umpire_sum) if umpire_sum else None,
    }


def invoke_benchmark(*, executable: Path, command: list[str], output_dir: Path,
                     relative_stem: Path, invocation_id: str, phase: str,
                     comparison_member: str, pair_index: int | None,
                     mode: str, order: int, mesh_n: int,
                     domain_periods: int, timestep: float, export: bool,
                     validation_source: str, validation_engine: str | None,
                     trace_policy: str,
                     expected_build: dict[str, str] | None,
                     expected_gpu: dict[str, Any] | None,
                     timeout_seconds: float,
                     invocations: list[dict[str, Any]]) -> dict[str, Any]:
    del executable  # The resolved executable is already the first command token.
    started_utc = utc_now()
    started = time.monotonic()
    stdout = ""
    stderr = ""
    returncode: int | None = None
    timed_out = False
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
    elapsed = time.monotonic() - started
    raw_path = Path("raw") / relative_stem.with_suffix(".json")
    stderr_path = Path("raw") / relative_stem.with_suffix(".stderr.txt")
    write_exclusive(output_dir / raw_path, stdout)
    write_exclusive(output_dir / stderr_path, stderr)
    invocation: dict[str, Any] = {
        "id": invocation_id,
        "phase": phase,
        "comparison_member": comparison_member,
        "pair_index": pair_index,
        "mode": mode,
        "order": order,
        "host_export": export,
        "trace_policy": trace_policy if validation_source == "final" else None,
        "command": command,
        "command_shell_display": shlex.join(command),
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "wall_seconds": elapsed,
        "returncode": returncode,
        "timed_out": timed_out,
        "raw_json": str(raw_path),
        "stderr": str(stderr_path),
        "raw_json_identity": artifact_identity(
            output_dir / raw_path, raw_path
        ),
        "stderr_identity": artifact_identity(
            output_dir / stderr_path, stderr_path
        ),
    }
    invocations.append(invocation)
    if timed_out:
        raise CampaignError(f"benchmark timed out: {invocation_id}")
    if returncode != 0:
        raise CampaignError(
            f"benchmark exited {returncode}: {invocation_id}; see {stderr_path}"
        )
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise CampaignError(
            f"benchmark did not emit one JSON value: {invocation_id}: {error}"
        ) from error
    record = validate_record(
        parsed,
        source=validation_source,
        mode=mode,
        order=order,
        mesh_n=mesh_n,
        domain_periods=domain_periods,
        timestep=timestep,
        export=export,
        engine=validation_engine,
        expected_build=expected_build,
        expected_gpu=expected_gpu,
        trace_policy=trace_policy,
    )
    invocation["benchmark_summary"] = record_summary(record)
    return record


def final_command(executable: Path, *, engine: str, mode: str, order: int,
                  mesh_n: int, domain_periods: int, timestep: float,
                  export: bool, expected_build: dict[str, str],
                  trace_policy: str = "engine-default") -> list[str]:
    # Fail before launching the benchmark when the explicit policy is not
    # defined for this exact engine/mode tuple.  The MGR and native element-
    # Schwarz experiments are tensor+IGR rank-one policies only.
    trace_policy_configuration(
        engine, mode, trace_policy, order=order
    )
    return [
        str(executable),
        "--mode", mode,
        "--engine", engine,
        "--trace-policy", trace_policy,
        "--order", str(order),
        "--mesh-n", str(mesh_n),
        "--domain-periods", str(domain_periods),
        "--dt", repr(timestep),
        "--export" if export else "--no-export",
        "--expected-source-revision", expected_build["source_revision"],
        "--expected-source-tree", expected_build["source_tree"],
        "--expected-source-status", expected_build["source_status"],
        "--expected-build-directory", expected_build["build_directory"],
        "--expected-cmake-cache", expected_build["cmake_cache"],
    ]


def frozen_command(executable: Path, *, mode: str, order: int, mesh_n: int,
                   domain_periods: int, timestep: float) -> list[str]:
    return [
        str(executable),
        "--mode", mode,
        "--order", str(order),
        "--mesh-n", str(mesh_n),
        "--domain-periods", str(domain_periods),
        "--dt", repr(timestep),
        "--no-export",
    ]


def summarize_tensor_cell(records: dict[str, dict[int, dict[str, Any]]],
                          *, mode: str, order: int,
                          bootstrap_resamples: int,
                          bootstrap_seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {"engines": {}}
    for engine in ("ordinary", "tensor"):
        ordered = [records[engine][index] for index in sorted(records[engine])]
        all_durations = [float(value) for record in ordered
                         for value in record["raw_durations_seconds"]]
        run_medians = [statistics.median(record["raw_durations_seconds"])
                       for record in ordered]
        result["engines"][engine] = {
            "raw_durations_seconds": sample_statistics(all_durations),
            "invocation_medians_seconds": sample_statistics(run_medians),
            "solver_work": work_summary(ordered),
            "initialization_and_memory": memory_summary(ordered),
        }
    common_pairs = sorted(set(records["ordinary"]) & set(records["tensor"]))
    pair_details = []
    speedups = []
    for pair_index in common_pairs:
        ordinary = statistics.median(
            records["ordinary"][pair_index]["raw_durations_seconds"]
        )
        tensor = statistics.median(
            records["tensor"][pair_index]["raw_durations_seconds"]
        )
        speedup = ordinary / tensor
        speedups.append(speedup)
        pair_details.append({
            "pair_index": pair_index,
            "ordinary_median_seconds": ordinary,
            "tensor_median_seconds": tensor,
            "ordinary_over_tensor_speedup": speedup,
        })
    interval = bootstrap_median_interval(
        speedups, bootstrap_resamples, bootstrap_seed
    )
    result["paired_comparison"] = {
        "speedup_definition": "ordinary invocation median / tensor invocation median",
        "attribution": tensor_comparison_attribution(mode, order),
        "pairs": pair_details,
        "speedup": sample_statistics(speedups),
        "median_speedup_95_percent_confidence_interval": interval,
        "noise_disposition": classify_speedup(interval),
        "geometric_mean_speedup": math.exp(
            statistics.mean(math.log(value) for value in speedups)
        ),
    }
    return result


def _ratio_summary(values: Sequence[float], *, definition: str,
                   bootstrap_resamples: int,
                   bootstrap_seed: int) -> dict[str, Any]:
    interval = bootstrap_median_interval(
        values, bootstrap_resamples, bootstrap_seed
    )
    return {
        "ratio_definition": definition,
        "ratio": sample_statistics(values),
        "median_ratio_95_percent_confidence_interval": interval,
        "noise_disposition": classify_speedup(interval),
        "geometric_mean_ratio": math.exp(
            statistics.mean(math.log(value) for value in values)
        ),
    }


def summarize_same_policy_cell(
        records: dict[str, dict[int, dict[str, Any]]], *, mode: str,
        order: int,
        bootstrap_resamples: int, bootstrap_seed: int) -> dict[str, Any]:
    """Summarize one bounded O_base/O_common/T_common triplet cell."""
    result: dict[str, Any] = {
        "status": "provisional_non_promotion_attribution",
        "promotion_eligible": False,
        "attribution": same_policy_attribution(mode, order),
        "members": {},
    }
    for member in SAME_POLICY_MEMBER_ORDER:
        ordered = [records[member][index]
                   for index in sorted(records[member])]
        if not ordered:
            raise CampaignError(
                f"same-policy attribution has no {member} records"
            )
        configurations = [
            record_trace_policy_configuration(record) for record in ordered
        ]
        if any(configuration != configurations[0]
               for configuration in configurations[1:]):
            raise CampaignError(
                f"same-policy member {member} changed trace policy"
            )
        selection = SAME_POLICY_MEMBERS[member]
        expected_configuration = trace_policy_configuration(
            selection["engine"], mode, selection["trace_policy"],
            order=order
        )
        if configurations[0] != expected_configuration:
            raise CampaignError(
                f"same-policy member {member} has unexpected trace metadata"
            )
        all_durations = [
            float(value) for record in ordered
            for value in record["raw_durations_seconds"]
        ]
        run_medians = [
            statistics.median(record["raw_durations_seconds"])
            for record in ordered
        ]
        result["members"][member] = {
            **selection,
            "trace_policy_configuration": configurations[0],
            "raw_durations_seconds": sample_statistics(all_durations),
            "invocation_medians_seconds": sample_statistics(run_medians),
            "solver_work": work_summary(ordered),
            "initialization_and_memory": memory_summary(ordered),
        }

    pair_sets = [set(records[member]) for member in SAME_POLICY_MEMBER_ORDER]
    if any(pair_set != pair_sets[0] for pair_set in pair_sets[1:]):
        raise CampaignError("same-policy attribution has incomplete triplets")
    common_pairs = pair_sets[0]
    triplets: list[dict[str, Any]] = []
    policy_gains: list[float] = []
    tensor_local_gains: list[float] = []
    package_gains: list[float] = []
    for pair_index in sorted(common_pairs):
        selected = {
            member: records[member][pair_index]
            for member in SAME_POLICY_MEMBER_ORDER
        }
        ordinary_common_configuration = record_trace_policy_configuration(
            selected["O_common"]
        )
        tensor_common_configuration = record_trace_policy_configuration(
            selected["T_common"]
        )
        if ordinary_common_configuration != tensor_common_configuration:
            raise CampaignError(
                "O_common and T_common did not report exactly matching "
                f"common-fast metadata in {mode} pair {pair_index}"
            )
        raw = {
            member: [float(value) for value in
                     selected[member]["raw_durations_seconds"]]
            for member in SAME_POLICY_MEMBER_ORDER
        }
        medians = {
            member: statistics.median(raw[member])
            for member in SAME_POLICY_MEMBER_ORDER
        }
        policy_gain = medians["O_base"] / medians["O_common"]
        tensor_local_gain = medians["O_common"] / medians["T_common"]
        package_gain = medians["O_base"] / medians["T_common"]
        if not math.isclose(
                policy_gain * tensor_local_gain, package_gain,
                rel_tol=1.0e-13, abs_tol=0.0):
            raise CampaignError("same-policy triplet ratio identity failed")
        policy_gains.append(policy_gain)
        tensor_local_gains.append(tensor_local_gain)
        package_gains.append(package_gain)
        triplets.append({
            "pair_index": pair_index,
            "raw_durations_seconds": raw,
            "invocation_medians_seconds": medians,
            "policy_gain_O_base_over_O_common": policy_gain,
            "tensor_local_gain_O_common_over_T_common": tensor_local_gain,
            "package_gain_O_base_over_T_common": package_gain,
            "common_fast_policy_configuration":
                ordinary_common_configuration,
            "common_fast_policy_exactly_matched": True,
        })
    result["triplets"] = triplets
    result["comparisons"] = {
        "policy_gain": _ratio_summary(
            policy_gains,
            definition="O_base invocation median / O_common invocation median",
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
        "tensor_local_gain": _ratio_summary(
            tensor_local_gains,
            definition="O_common invocation median / T_common invocation median",
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + 1000,
        ),
        "package_gain": _ratio_summary(
            package_gains,
            definition="O_base invocation median / T_common invocation median",
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + 2000,
        ),
    }
    return result


def summarize_frozen_cell(records: dict[str, dict[int, dict[str, Any]]],
                          *, bootstrap_resamples: int,
                          bootstrap_seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {"controls": {}}
    for member in ("final_ordinary", "frozen_ordinary"):
        ordered = [records[member][index] for index in sorted(records[member])]
        durations = [float(value) for record in ordered
                     for value in record["raw_durations_seconds"]]
        medians = [statistics.median(record["raw_durations_seconds"])
                   for record in ordered]
        result["controls"][member] = {
            "raw_durations_seconds": sample_statistics(durations),
            "invocation_medians_seconds": sample_statistics(medians),
            "solver_work": work_summary(ordered),
        }
    pair_details = []
    speedups = []
    for pair_index in sorted(set(records["final_ordinary"]) &
                             set(records["frozen_ordinary"])):
        final_median = statistics.median(
            records["final_ordinary"][pair_index]["raw_durations_seconds"]
        )
        frozen_median = statistics.median(
            records["frozen_ordinary"][pair_index]["raw_durations_seconds"]
        )
        speedup = frozen_median / final_median
        speedups.append(speedup)
        pair_details.append({
            "pair_index": pair_index,
            "final_ordinary_median_seconds": final_median,
            "frozen_ordinary_median_seconds": frozen_median,
            "final_ordinary_speedup_over_frozen": speedup,
        })
    interval = bootstrap_median_interval(
        speedups, bootstrap_resamples, bootstrap_seed
    )
    result["paired_comparison"] = {
        "speedup_definition": "frozen invocation median / final-tree ordinary invocation median",
        "pairs": pair_details,
        "speedup": sample_statistics(speedups),
        "median_speedup_95_percent_confidence_interval": interval,
        "noise_disposition": classify_speedup(interval),
    }
    return result


def performance_disposition(manifest: dict[str, Any], *,
                            evidence_mode: str,
                            workload_class: str) -> dict[str, Any]:
    """Separate completed measurement from promotion-performance acceptance."""
    expected_cells = tuple(f"{mode}_p{order}" for mode, order in CELLS)

    def disclose(result: dict[str, Any]) -> dict[str, Any]:
        comparison = result["paired_comparison"]
        interval = comparison[
            "median_speedup_95_percent_confidence_interval"
        ]
        classification = classify_speedup(interval)
        if comparison["noise_disposition"] != classification:
            raise CampaignError(
                "recorded performance classification differs from its interval"
            )
        return {
            "median_speedup": checked_float(
                comparison["speedup"]["median"], "speedup.median"
            ),
            "confidence_interval": interval,
            "classification": classification,
        }

    tensor_results = manifest.get("final_tree_results", {})
    tensor_cells = {
        key: disclose(tensor_results[key])
        for key in expected_cells if key in tensor_results
    }
    tensor_wins = [
        key for key in expected_cells
        if tensor_cells.get(key, {}).get("classification") ==
        "win_beyond_observed_noise"
    ]
    tensor_missing = [key for key in expected_cells if key not in tensor_cells]
    tensor_non_wins = [
        key for key in expected_cells
        if key in tensor_cells and key not in tensor_wins
    ]
    is_saturated_qualification = (
        evidence_mode == "qualification" and workload_class == "saturated"
    )
    tensor_saturated_pass = (
        is_saturated_qualification and not tensor_missing and not tensor_non_wins
    )
    if workload_class == "break-even":
        tensor_disposition = "characterization_only_not_a_promotion_gate"
        reported_saturated_pass: bool | None = None
    elif evidence_mode != "qualification":
        tensor_disposition = "not_qualification_evidence"
        reported_saturated_pass = None
    elif tensor_saturated_pass:
        tensor_disposition = "pass_all_six_cells_win_beyond_observed_noise"
        reported_saturated_pass = True
    else:
        tensor_disposition = "fail_not_all_six_cells_win_beyond_observed_noise"
        reported_saturated_pass = False

    frozen_results = manifest.get("frozen_control_results", {})
    frozen_cells = {
        key: disclose(frozen_results[key])
        for key in expected_cells if key in frozen_results
    }
    frozen_missing = [key for key in expected_cells if key not in frozen_cells]
    frozen_regressions = [
        key for key in expected_cells
        if frozen_cells.get(key, {}).get("classification") ==
        "regression_beyond_observed_noise"
    ]
    frozen_nominal_regressions = [
        key for key in expected_cells
        if key in frozen_cells and frozen_cells[key]["median_speedup"] < 1.0
    ]
    frozen_control_pass = not frozen_missing and not frozen_regressions
    if frozen_missing:
        frozen_disposition = "incomplete_control_disclosure"
    elif frozen_regressions:
        frozen_disposition = "final_tree_ordinary_regression_detected"
    else:
        frozen_disposition = "no_regression_beyond_observed_noise"

    promotion_pass = tensor_saturated_pass and frozen_control_pass
    if workload_class == "break-even":
        promotion_disposition = "characterization_only_not_a_promotion_gate"
    elif evidence_mode != "qualification":
        promotion_disposition = "not_qualification_evidence"
    elif promotion_pass:
        promotion_disposition = "pass"
    elif frozen_regressions:
        promotion_disposition = "fail_frozen_ordinary_regression"
    else:
        promotion_disposition = "fail_tensor_six_cell_requirement"

    return {
        "evidence_mode": evidence_mode,
        "workload_class": workload_class,
        "tensor_vs_final_tree_ordinary": {
            "speedup_definition": "ordinary / tensor",
            "attribution_by_cell": {
                f"{mode}_p{order}": tensor_comparison_attribution(mode, order)
                for mode, order in CELLS
            },
            "required_cells": list(expected_cells),
            "cells": tensor_cells,
            "all_six_cells_disclosed": not tensor_missing,
            "missing_cells": tensor_missing,
            "wins_beyond_observed_noise": tensor_wins,
            "non_winning_cells": tensor_non_wins,
            "saturated_pass": reported_saturated_pass,
            "disposition": tensor_disposition,
        },
        "final_tree_ordinary_vs_frozen_control": {
            "speedup_definition": "frozen / final-tree ordinary",
            "required_cells": list(expected_cells),
            "cells": frozen_cells,
            "all_six_cells_disclosed": not frozen_missing,
            "missing_cells": frozen_missing,
            "nominal_median_regressions": frozen_nominal_regressions,
            "has_nominal_median_regression": bool(frozen_nominal_regressions),
            "regressions_beyond_observed_noise": frozen_regressions,
            "has_regression_beyond_observed_noise": bool(frozen_regressions),
            "control_pass": frozen_control_pass,
            "disposition": frozen_disposition,
        },
        "promotion_performance_pass": promotion_pass,
        "promotion_performance_disposition": promotion_disposition,
    }


def same_policy_performance_disposition(
        results: dict[str, dict[str, Any]], *,
        evidence_mode: str) -> dict[str, Any]:
    """Label the ablation as bounded provisional attribution, never a gate."""
    expected_cells = tuple(f"{mode}_p{order}" for mode, order in CELLS)
    missing = [key for key in expected_cells if key not in results]
    if missing:
        raise CampaignError(
            "same-policy attribution is missing cells: " + ", ".join(missing)
        )
    cells: dict[str, Any] = {}
    for key in expected_cells:
        comparisons = results[key].get("comparisons", {})
        cell: dict[str, Any] = {}
        for name in ("policy_gain", "tensor_local_gain", "package_gain"):
            comparison = comparisons.get(name)
            if not isinstance(comparison, dict):
                raise CampaignError(
                    f"same-policy attribution {key} lacks {name}"
                )
            interval = comparison.get(
                "median_ratio_95_percent_confidence_interval"
            )
            classification = classify_speedup(interval)
            if comparison.get("noise_disposition") != classification:
                raise CampaignError(
                    f"same-policy attribution {key}/{name} has an "
                    "inconsistent noise classification"
                )
            cell[name] = {
                "median_ratio": checked_float(
                    comparison["ratio"]["median"], f"{key}.{name}.median"
                ),
                "confidence_interval": interval,
                "classification": classification,
            }
        cells[key] = cell
    return {
        "evidence_mode": evidence_mode,
        "comparison_mode": "same-policy-attribution",
        "status": "provisional_non_promotion_attribution",
        "promotion_eligible": False,
        "required_cells": list(expected_cells),
        "all_six_cells_disclosed": True,
        "cells": cells,
        "promotion_performance_pass": False,
        "promotion_performance_disposition":
            "provisional_same_policy_attribution_not_a_promotion_gate",
        "interpretation": (
            "policy_gain isolates engine-default versus common-fast within "
            "ordinary CUDA; tensor_local_gain compares ordinary and tensor "
            "CUDA under the same common-fast trace policy"
        ),
    }


def resolve_nvidia_smi(argument: str | None) -> Path | None:
    candidate = argument or shutil.which("nvidia-smi")
    if candidate is None:
        return None
    resolved = Path(candidate).expanduser().resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise CampaignError(f"nvidia-smi is not executable: {resolved}")
    return resolved


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run paired engine-package or cyclic same-policy accepted-step "
            "evidence. "
            "The caller must already hold the external exclusive GPU lease."
        )
    )
    result.add_argument("--benchmark", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--source-root", default=str(REPOSITORY_ROOT))
    result.add_argument("--evidence-mode", required=True,
                        choices=("qualification", "smoke"))
    result.add_argument(
        "--comparison-mode",
        choices=("engine-package", "same-policy-attribution"),
        default="engine-package",
        help=(
            "Production engine-package comparison or provisional, "
            "non-promotion O_base/O_common/T_common attribution."
        ),
    )
    result.add_argument("--pairs", required=True, type=parse_positive_int)
    result.add_argument("--workload-class", required=True,
                        choices=("saturated", "break-even"))
    result.add_argument("--workload-label", required=True)
    result.add_argument("--mesh-n", required=True, type=parse_positive_int)
    result.add_argument("--domain-periods", required=True,
                        type=parse_positive_int)
    result.add_argument("--dt", required=True, type=parse_positive_float)
    result.add_argument("--export-control-mode", required=True,
                        choices=("conventional", "igr"))
    result.add_argument("--export-control-order", required=True, type=int,
                        choices=(1, 2, 3))
    result.add_argument("--build-label", required=True)
    result.add_argument("--build-metadata", action="append", default=[],
                        metavar="KEY=VALUE")
    result.add_argument("--build-manifest", required=True)
    result.add_argument("--mfem-identity", required=True)
    result.add_argument("--expected-commit")
    result.add_argument("--expected-tree")
    result.add_argument("--allow-dirty", action="store_true")
    result.add_argument("--gpu-index", required=True, type=int)
    result.add_argument("--nvidia-smi")
    result.add_argument("--require-nvidia-smi", action="store_true")
    result.add_argument("--hardware-metadata", action="append", default=[],
                        metavar="KEY=VALUE")
    result.add_argument("--frozen-benchmark")
    result.add_argument("--frozen-source-root")
    result.add_argument("--frozen-build-label")
    result.add_argument("--frozen-build-metadata", action="append", default=[],
                        metavar="KEY=VALUE")
    result.add_argument("--frozen-build-manifest")
    result.add_argument("--frozen-mfem-identity")
    result.add_argument("--bootstrap-resamples", type=parse_positive_int,
                        default=10000)
    result.add_argument("--bootstrap-seed", type=int, default=20260823)
    result.add_argument("--timeout-seconds", type=parse_positive_float,
                        default=1800.0)
    return result


def run_campaign(arguments: argparse.Namespace) -> Path:
    if arguments.comparison_mode == "engine-package":
        if arguments.pairs < 2:
            raise CampaignError(
                "at least two pairs are required for alternating order"
            )
    elif arguments.pairs < 3 or arguments.pairs % 3 != 0:
        raise CampaignError(
            "same-policy attribution requires at least three triplets and a "
            "triplet count divisible by three for balanced cyclic order"
        )
    if arguments.mesh_n < 2 or \
            arguments.domain_periods > arguments.mesh_n or \
            arguments.mesh_n % arguments.domain_periods != 0:
        raise CampaignError(
            "domain-periods must be positive, no larger than mesh-n, and divide mesh-n"
        )
    if not arguments.workload_label.strip():
        raise CampaignError("--workload-label must be non-empty")
    frozen_arguments = (
        arguments.frozen_benchmark, arguments.frozen_source_root,
        arguments.frozen_build_label, arguments.frozen_build_manifest,
        arguments.frozen_mfem_identity,
    )
    if any(frozen_arguments) and not all(frozen_arguments):
        raise CampaignError(
            "all frozen benchmark/source/build/MFEM arguments must be supplied together"
        )
    if arguments.comparison_mode == "same-policy-attribution":
        if arguments.evidence_mode != "smoke":
            raise CampaignError(
                "same-policy attribution is provisional non-promotion evidence "
                "and requires --evidence-mode smoke"
            )
        if any(frozen_arguments):
            raise CampaignError(
                "same-policy attribution forbids the unrelated frozen-control phase"
            )
        if not arguments.expected_commit or not arguments.expected_tree:
            raise CampaignError(
                "same-policy attribution requires an exact expected commit/tree"
            )
        if arguments.allow_dirty:
            raise CampaignError(
                "same-policy attribution requires a clean source tree"
            )
        if not arguments.require_nvidia_smi:
            raise CampaignError(
                "same-policy attribution requires nvidia-smi metadata"
            )
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(arguments.gpu_index):
            raise CampaignError(
                "same-policy attribution requires CUDA_VISIBLE_DEVICES to equal "
                "the exclusive --gpu-index"
            )
    if arguments.evidence_mode == "qualification":
        validate_qualification_profile(arguments)
        if arguments.bootstrap_resamples < 10000:
            raise CampaignError(
                "qualification evidence requires at least 10000 bootstrap resamples"
            )
        if not all(frozen_arguments):
            raise CampaignError("qualification evidence requires the frozen control")
        if not arguments.expected_commit or not arguments.expected_tree:
            raise CampaignError(
                "qualification evidence requires expected commit and tree"
            )
        if arguments.allow_dirty:
            raise CampaignError("qualification evidence cannot allow a dirty source")
        if not arguments.require_nvidia_smi:
            raise CampaignError("qualification evidence requires nvidia-smi metadata")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(arguments.gpu_index):
            raise CampaignError(
                "qualification evidence requires CUDA_VISIBLE_DEVICES to equal "
                "the exclusive --gpu-index"
            )

    output_dir = Path(arguments.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise CampaignError(f"output directory already exists: {output_dir}")
    source_root = Path(arguments.source_root).expanduser().resolve()
    source_start = git_identity(source_root)
    if source_start["status_short"] and not arguments.allow_dirty:
        raise CampaignError("source tree is dirty; final campaign requires a clean tree")
    if arguments.expected_commit and \
            source_start["commit"] != arguments.expected_commit:
        raise CampaignError("source commit does not match --expected-commit")
    if arguments.expected_tree and source_start["tree"] != arguments.expected_tree:
        raise CampaignError("source tree does not match --expected-tree")

    final_identity = file_identity(Path(arguments.benchmark))
    final_executable = Path(final_identity["path"])
    final_build_provenance = validated_build_provenance(
        source_root=source_root,
        cmake_cache_path=arguments.build_manifest,
        mfem_identity_path=arguments.mfem_identity,
    )
    final_build_relationship = validate_executable_build_relationship(
        final_identity, final_build_provenance
    )
    final_expected_build = {
        "source_revision": source_start["commit"],
        "source_tree": source_start["tree"],
        "source_status": "clean" if not source_start["status_short"] else "dirty",
        "build_directory": str(Path(arguments.build_manifest).resolve().parent),
        "cmake_cache": str(Path(arguments.build_manifest).resolve()),
    }
    frozen_identity = None
    frozen_executable = None
    frozen_source_identity = None
    frozen_build_provenance = None
    if arguments.frozen_benchmark:
        frozen_identity = file_identity(Path(arguments.frozen_benchmark))
        frozen_executable = Path(frozen_identity["path"])
        frozen_root = Path(arguments.frozen_source_root).expanduser().resolve()
        frozen_source_identity = frozen_overlay_identity(frozen_root)
        frozen_build_provenance = validated_build_provenance(
            source_root=frozen_root,
            cmake_cache_path=arguments.frozen_build_manifest,
            mfem_identity_path=arguments.frozen_mfem_identity,
        )
        frozen_build_relationship = validate_executable_build_relationship(
            frozen_identity, frozen_build_provenance,
            source_modified_mtime_ns=frozen_source_identity["modified_file"][
                "mtime_ns"
            ],
        )
        if frozen_identity["sha256"] == final_identity["sha256"]:
            raise CampaignError(
                "final and frozen benchmark executable hashes must be distinct"
            )
    nvidia_smi = resolve_nvidia_smi(arguments.nvidia_smi)
    if arguments.require_nvidia_smi and nvidia_smi is None:
        raise CampaignError("--require-nvidia-smi was set but nvidia-smi was not found")

    build_metadata = parse_metadata(arguments.build_metadata, "build metadata")
    frozen_build_metadata = parse_metadata(
        arguments.frozen_build_metadata, "frozen build metadata"
    )
    supplied_hardware = parse_metadata(
        arguments.hardware_metadata, "hardware metadata"
    )
    driver_path = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "schema": 1,
        "campaign": "production_tensor_core_hdg_paired_accepted_step",
        "evidence_mode": arguments.evidence_mode,
        "comparison_mode": arguments.comparison_mode,
        "promotion_qualification": (
            arguments.evidence_mode == "qualification" and
            arguments.comparison_mode == "engine-package"
        ),
        "attribution_status": (
            "provisional_non_promotion_attribution"
            if arguments.comparison_mode == "same-policy-attribution"
            else "production_engine_package_comparison"
        ),
        "status": "running",
        "execution_status": "running",
        "started_utc": utc_now(),
        "driver": {
            "path": str(driver_path),
            "sha256": sha256_file(driver_path),
            "argv": sys.argv,
            "python": sys.version,
        },
        "method": {
            "external_exclusive_gpu_lease_required": True,
            "gpu_lease_acquired_by_driver": False,
            "build_performed_by_driver": False,
            "cell_order": "forward on even pair rounds, reverse on odd rounds",
            "engine_order": (
                "ordinary/tensor on even pairs, tensor/ordinary on odd pairs"
                if arguments.comparison_mode == "engine-package"
                else "see member_order"
            ),
            "member_order": (
                "ordinary/tensor alternating by pair"
                if arguments.comparison_mode == "engine-package"
                else "O_base/O_common/T_common cyclic Latin order by triplet"
            ),
            "primary_timing": "benchmark schema-2 no-export accepted steps",
            "export_control": (
                "one export-enabled invocation per final-tree engine"
                if arguments.comparison_mode == "engine-package"
                else "not run in bounded same-policy attribution"
            ),
            "noise_classification": (
                "95% paired-bootstrap CI lower>1 win; upper<1 regression; "
                "otherwise break-even/indeterminate"
            ),
            "retained_observability": (
                "bounded benchmark samples and aggregate metadata only; no "
                "event stream, replay, callback capture, or flight recorder"
            ),
        },
        "workload": {
            "class": arguments.workload_class,
            "label": arguments.workload_label,
            "modes": ["conventional", "igr"],
            "orders": [1, 2, 3],
            "pairs_per_cell": arguments.pairs,
            "mesh_n": arguments.mesh_n,
            "domain_periods": arguments.domain_periods,
            "dt": arguments.dt,
            "host_export_primary": False,
            "export_control_mode": arguments.export_control_mode,
            "export_control_order": arguments.export_control_order,
        },
        "source_start": source_start,
        "builds": {
            "final_tree": {
                "label": arguments.build_label,
                "metadata": build_metadata,
                "provenance": final_build_provenance,
                "executable": final_identity,
                "executable_build_relationship": final_build_relationship,
            }
        },
        "host": {
            "platform": platform.platform(),
            "uname": list(platform.uname()),
            "safe_environment": {
                key: os.environ.get(key) for key in (
                    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
                    "OMP_NUM_THREADS", "MFEM_DEVICE"
                ) if key in os.environ
            },
        },
        "hardware": {
            "supplied": supplied_hardware,
            "gpu_index": arguments.gpu_index,
            "nvidia_smi_executable": str(nvidia_smi) if nvidia_smi else None,
            "pair_rounds": [],
        },
        "invocations": [],
    }
    if arguments.comparison_mode == "same-policy-attribution":
        manifest["workload"]["triplets_per_cell"] = arguments.pairs
    if frozen_identity is not None:
        manifest["builds"]["frozen_ordinary"] = {
            "label": arguments.frozen_build_label,
            "metadata": frozen_build_metadata,
            "provenance": frozen_build_provenance,
            "executable": frozen_identity,
            "declared_source_commit": FROZEN_CONTROL_COMMIT,
            "declared_source_tree": FROZEN_CONTROL_TREE,
            "source_and_overlay": frozen_source_identity,
            "executable_build_relationship": frozen_build_relationship,
            "provenance_note": (
                "The executable is bound to the frozen application tree plus the "
                "recorded benchmark-only polynomial-order CLI overlay."
            ),
        }

    output_dir.mkdir(parents=True)
    primary_members = (
        ("ordinary", "tensor")
        if arguments.comparison_mode == "engine-package"
        else SAME_POLICY_MEMBER_ORDER
    )
    final_records = {
        (mode, order): {member: {} for member in primary_members}
        for mode, order in CELLS
    }
    frozen_records = {
        (mode, order): {"final_ordinary": {}, "frozen_ordinary": {}}
        for mode, order in CELLS
    }
    try:
        start_hardware = query_nvidia_smi(nvidia_smi, arguments.gpu_index)
        validate_hardware_snapshot(start_hardware, arguments.require_nvidia_smi)
        manifest["hardware"]["start"] = start_hardware
        hardware_values = start_hardware.get("values", {})
        try:
            expected_compute_capability = int(round(
                float(hardware_values["compute_cap"]) * 10.0
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise CampaignError(
                "nvidia-smi compute capability is unavailable or invalid"
            ) from exc
        final_expected_gpu = {
            "device_ordinal": 0,
            "uuid": hardware_values.get("uuid"),
            "compute_capability": expected_compute_capability,
        }

        for pair_index in range(arguments.pairs):
            cells = CELLS if pair_index % 2 == 0 else tuple(reversed(CELLS))
            if arguments.comparison_mode == "engine-package":
                members = ("ordinary", "tensor") if pair_index % 2 == 0 \
                    else ("tensor", "ordinary")
            else:
                rotation = pair_index % len(SAME_POLICY_MEMBER_ORDER)
                members = (
                    SAME_POLICY_MEMBER_ORDER[rotation:] +
                    SAME_POLICY_MEMBER_ORDER[:rotation]
                )
            for mode, order in cells:
                for member in members:
                    if arguments.comparison_mode == "engine-package":
                        engine = member
                        trace_policy = "engine-default"
                    else:
                        selection = SAME_POLICY_MEMBERS[member]
                        engine = selection["engine"]
                        trace_policy = selection["trace_policy"]
                    invocation_id = (
                        f"final-{mode}-p{order}-pair{pair_index:03d}-{member}"
                    )
                    stem = Path("final") / f"{mode}-p{order}" / \
                        f"pair-{pair_index:03d}-{member}"
                    record = invoke_benchmark(
                        executable=final_executable,
                        command=final_command(
                            final_executable,
                            engine=engine,
                            mode=mode,
                            order=order,
                            mesh_n=arguments.mesh_n,
                            domain_periods=arguments.domain_periods,
                            timestep=arguments.dt,
                            export=False,
                            expected_build=final_expected_build,
                            trace_policy=trace_policy,
                        ),
                        output_dir=output_dir,
                        relative_stem=stem,
                        invocation_id=invocation_id,
                        phase=(
                            "final_tree_primary"
                            if arguments.comparison_mode == "engine-package"
                            else "same_policy_attribution"
                        ),
                        comparison_member=member,
                        pair_index=pair_index,
                        mode=mode,
                        order=order,
                        mesh_n=arguments.mesh_n,
                        domain_periods=arguments.domain_periods,
                        timestep=arguments.dt,
                        export=False,
                        validation_source="final",
                        validation_engine=engine,
                        trace_policy=trace_policy,
                        expected_build=final_expected_build,
                        expected_gpu=final_expected_gpu,
                        timeout_seconds=arguments.timeout_seconds,
                        invocations=manifest["invocations"],
                    )
                    if arguments.workload_class == "saturated" and \
                            record["principal_blocks"] < record["gpu_sm_count"]:
                        raise CampaignError(
                            "saturated workload did not launch at least one principal "
                            "assembly block per SM"
                        )
                    final_records[(mode, order)][member][pair_index] = record
            pair_hardware = query_nvidia_smi(nvidia_smi, arguments.gpu_index)
            validate_hardware_snapshot(
                pair_hardware, arguments.require_nvidia_smi
            )
            validate_hardware_identity(start_hardware, pair_hardware)
            manifest["hardware"]["pair_rounds"].append({
                "phase": (
                    "final_tree_primary"
                    if arguments.comparison_mode == "engine-package"
                    else "same_policy_attribution"
                ),
                "pair_index": pair_index,
                "member_order": list(members),
                "snapshot": pair_hardware,
            })

        export_records: dict[str, dict[str, Any]] = {}
        export_engines = (
            ("ordinary", "tensor")
            if arguments.comparison_mode == "engine-package" else ()
        )
        for engine in export_engines:
            mode = arguments.export_control_mode
            order = arguments.export_control_order
            invocation_id = f"export-{mode}-p{order}-{engine}"
            record = invoke_benchmark(
                executable=final_executable,
                command=final_command(
                    final_executable,
                    engine=engine,
                    mode=mode,
                    order=order,
                    mesh_n=arguments.mesh_n,
                    domain_periods=arguments.domain_periods,
                    timestep=arguments.dt,
                    export=True,
                    expected_build=final_expected_build,
                ),
                output_dir=output_dir,
                relative_stem=Path("export") / f"{mode}-p{order}" / engine,
                invocation_id=invocation_id,
                phase="final_tree_export_control",
                comparison_member=engine,
                pair_index=None,
                mode=mode,
                order=order,
                mesh_n=arguments.mesh_n,
                domain_periods=arguments.domain_periods,
                timestep=arguments.dt,
                export=True,
                validation_source="final",
                validation_engine=engine,
                trace_policy="engine-default",
                expected_build=final_expected_build,
                expected_gpu=final_expected_gpu,
                timeout_seconds=arguments.timeout_seconds,
                invocations=manifest["invocations"],
            )
            export_records[engine] = record

        if frozen_executable is not None:
            for pair_index in range(arguments.pairs):
                cells = CELLS if pair_index % 2 == 0 else tuple(reversed(CELLS))
                members = ("final_ordinary", "frozen_ordinary") \
                    if pair_index % 2 == 0 else \
                    ("frozen_ordinary", "final_ordinary")
                for mode, order in cells:
                    for member in members:
                        is_final = member == "final_ordinary"
                        executable = final_executable if is_final else frozen_executable
                        command = final_command(
                            executable,
                            engine="ordinary",
                            mode=mode,
                            order=order,
                            mesh_n=arguments.mesh_n,
                            domain_periods=arguments.domain_periods,
                            timestep=arguments.dt,
                            export=False,
                            expected_build=final_expected_build,
                        ) if is_final else frozen_command(
                            executable,
                            mode=mode,
                            order=order,
                            mesh_n=arguments.mesh_n,
                            domain_periods=arguments.domain_periods,
                            timestep=arguments.dt,
                        )
                        invocation_id = (
                            f"frozen-compare-{mode}-p{order}-pair{pair_index:03d}-{member}"
                        )
                        stem = Path("frozen-comparison") / f"{mode}-p{order}" / \
                            f"pair-{pair_index:03d}-{member}"
                        record = invoke_benchmark(
                            executable=executable,
                            command=command,
                            output_dir=output_dir,
                            relative_stem=stem,
                            invocation_id=invocation_id,
                            phase="ordinary_frozen_comparison",
                            comparison_member=member,
                            pair_index=pair_index,
                            mode=mode,
                            order=order,
                            mesh_n=arguments.mesh_n,
                            domain_periods=arguments.domain_periods,
                            timestep=arguments.dt,
                            export=False,
                            validation_source="final" if is_final else "frozen",
                            validation_engine="ordinary" if is_final else None,
                            trace_policy="engine-default",
                            expected_build=(
                                final_expected_build if is_final else None
                            ),
                            expected_gpu=(final_expected_gpu if is_final else None),
                            timeout_seconds=arguments.timeout_seconds,
                            invocations=manifest["invocations"],
                        )
                        frozen_records[(mode, order)][member][pair_index] = record
                pair_hardware = query_nvidia_smi(
                    nvidia_smi, arguments.gpu_index
                )
                validate_hardware_snapshot(
                    pair_hardware, arguments.require_nvidia_smi
                )
                validate_hardware_identity(start_hardware, pair_hardware)
                manifest["hardware"]["pair_rounds"].append({
                    "phase": "ordinary_frozen_comparison",
                    "pair_index": pair_index,
                    "snapshot": pair_hardware,
                })

        source_end = git_identity(Path(source_start["root"]))
        if source_end != source_start:
            raise CampaignError("source commit/tree/status changed during the campaign")
        manifest["source_end"] = source_end
        end_hardware = query_nvidia_smi(nvidia_smi, arguments.gpu_index)
        validate_hardware_snapshot(end_hardware, arguments.require_nvidia_smi)
        validate_hardware_identity(start_hardware, end_hardware)
        manifest["hardware"]["end"] = end_hardware

        if file_identity(final_executable) != final_identity:
            raise CampaignError("final-tree benchmark executable changed during campaign")
        if regular_file_identity(driver_path)["sha256"] != \
                manifest["driver"]["sha256"]:
            raise CampaignError("campaign driver changed during campaign")
        final_build_end = validated_build_provenance(
            source_root=source_root,
            cmake_cache_path=arguments.build_manifest,
            mfem_identity_path=arguments.mfem_identity,
        )
        if final_build_end != final_build_provenance:
            raise CampaignError("final-tree build provenance changed during campaign")
        if frozen_executable is not None:
            if file_identity(frozen_executable) != frozen_identity:
                raise CampaignError("frozen benchmark executable changed during campaign")
            frozen_root = Path(arguments.frozen_source_root).expanduser().resolve()
            if frozen_overlay_identity(frozen_root) != frozen_source_identity:
                raise CampaignError("frozen source/overlay changed during campaign")
            frozen_build_end = validated_build_provenance(
                source_root=frozen_root,
                cmake_cache_path=arguments.frozen_build_manifest,
                mfem_identity_path=arguments.frozen_mfem_identity,
            )
            if frozen_build_end != frozen_build_provenance:
                raise CampaignError("frozen build provenance changed during campaign")
        manifest["integrity"] = {
            "executables_rehashed_unchanged": True,
            "build_provenance_rehashed_unchanged": True,
            "sources_rechecked_unchanged": True,
            "raw_artifacts": validate_raw_artifacts(
                output_dir, manifest["invocations"]
            ),
        }

        if arguments.comparison_mode == "engine-package":
            manifest["final_tree_results"] = {}
            for cell_index, (mode, order) in enumerate(CELLS):
                key = f"{mode}_p{order}"
                manifest["final_tree_results"][key] = summarize_tensor_cell(
                    final_records[(mode, order)],
                    mode=mode,
                    order=order,
                    bootstrap_resamples=arguments.bootstrap_resamples,
                    bootstrap_seed=arguments.bootstrap_seed + cell_index,
                )
            ordinary_export = statistics.median(
                export_records["ordinary"]["raw_durations_seconds"]
            )
            tensor_export = statistics.median(
                export_records["tensor"]["raw_durations_seconds"]
            )
            manifest["export_control"] = {
                "mode": arguments.export_control_mode,
                "order": arguments.export_control_order,
                "ordinary": record_summary(export_records["ordinary"]),
                "tensor": record_summary(export_records["tensor"]),
                "ordinary_over_tensor_speedup": ordinary_export / tensor_export,
                "attribution": tensor_comparison_attribution(
                    arguments.export_control_mode,
                    arguments.export_control_order,
                ),
            }
            if frozen_executable is not None:
                manifest["frozen_control_results"] = {}
                for cell_index, (mode, order) in enumerate(CELLS):
                    key = f"{mode}_p{order}"
                    manifest["frozen_control_results"][key] = \
                        summarize_frozen_cell(
                            frozen_records[(mode, order)],
                            bootstrap_resamples=arguments.bootstrap_resamples,
                            bootstrap_seed=(
                                arguments.bootstrap_seed + 100 + cell_index
                            ),
                        )
            manifest["performance_disposition"] = performance_disposition(
                manifest,
                evidence_mode=arguments.evidence_mode,
                workload_class=arguments.workload_class,
            )
        else:
            manifest["same_policy_attribution_results"] = {}
            for cell_index, (mode, order) in enumerate(CELLS):
                key = f"{mode}_p{order}"
                manifest["same_policy_attribution_results"][key] = \
                    summarize_same_policy_cell(
                        final_records[(mode, order)],
                        mode=mode,
                        order=order,
                        bootstrap_resamples=arguments.bootstrap_resamples,
                        bootstrap_seed=arguments.bootstrap_seed + cell_index,
                    )
            manifest["export_control"] = {
                "status": "not_run",
                "reason": "bounded_same_policy_attribution_primary_only",
            }
            manifest["performance_disposition"] = \
                same_policy_performance_disposition(
                    manifest["same_policy_attribution_results"],
                    evidence_mode=arguments.evidence_mode,
                )
        manifest["status"] = "completed"
        manifest["execution_status"] = "completed"
        manifest["finished_utc"] = utc_now()
    except Exception as error:
        manifest["status"] = "failure"
        manifest["execution_status"] = "failure"
        manifest["finished_utc"] = utc_now()
        manifest["error"] = f"{type(error).__name__}: {error}"
        try:
            manifest["hardware"]["end_after_failure"] = query_nvidia_smi(
                nvidia_smi, arguments.gpu_index
            )
        except Exception as hardware_error:
            manifest["hardware"]["end_after_failure_error"] = str(hardware_error)
        write_json_exclusive(output_dir / "campaign.json", manifest)
        raise

    write_json_exclusive(output_dir / "campaign.json", manifest)
    return output_dir / "campaign.json"


def main() -> int:
    try:
        arguments = parser().parse_args()
        manifest = run_campaign(arguments)
        print(json.dumps({
            "execution_completed": True,
            "campaign_manifest": str(manifest),
        }))
        return 0
    except (CampaignError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"hdg_tensor_campaign: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
