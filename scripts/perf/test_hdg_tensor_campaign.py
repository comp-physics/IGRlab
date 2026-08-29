#!/usr/bin/env python3
"""No-build/no-GPU smoke and synthetic qualification campaign tests."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


SCRIPT = Path(__file__).with_name("hdg_tensor_campaign.py").resolve()

MODULE_SPEC = importlib.util.spec_from_file_location(
    "hdg_tensor_campaign_under_test", SCRIPT
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError("could not load hdg_tensor_campaign.py")
CAMPAIGN = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(CAMPAIGN)

SYNTHETIC_HYPRE_HASH = "eofobxw4pr2efngtgfbdmp5lks5yex7i"
SYNTHETIC_HYPRE_REVISION = "7e247a231e000000000000000000000000000000"
SCHEMA1_SAMPLE_SUMMARY_FIELDS = {
    "measured_index", "nonlinear_updates", "trace_solves",
    "linear_iterations", "line_search_trials",
    "rejected_line_search_trials", "transfers", "gates",
}
SCHEMA1_WORK_SUMMARY_FIELDS = {
    "linear_iterations_by_invocation", "nonlinear_updates", "trace_solves",
    "linear_iterations", "line_search_trials",
    "rejected_line_search_trials", "transfers", "acceptance_gates",
}
SCHEMA1_RECORD_SUMMARY_FIELDS = {
    "schema", "workload_identity", "median_seconds_recomputed",
    "raw_duration_count", "measured_sample_work",
    "measured_linear_iterations_total", "gpu_sm_count", "timing_scope",
    "initialization_timed",
}
SCHEMA1_WORKLOAD_SUMMARY_FIELDS = {
    "mode", "polynomial_order", "mesh_n", "domain_periods",
    "domain_length", "cells_per_period", "element_h", "elements",
    "interior_unknowns", "true_trace_unknowns", "principal_blocks",
    "gpu_sm_count", "dt", "host_export",
}
SCHEMA2_ONLY_SUMMARY_FIELDS = {
    "trace_policy", "trace_policies", "trace_solver", "trace_solvers",
    "trace_preconditioner",
    "trace_ilu_policy", "trace_preconditioners",
    "tensor_core", "trace_and_tensor_accounting",
    "trace_policy_metadata_status", "precision_policy", "numerical_policy",
    "initialization_duration_seconds", "initialization_in_primary_timing",
    "device_memory_snapshots",
    "umpire_allocator_high_watermarks_after_samples", "gpu_name",
    "gpu_compute_capability", "cuda_driver_version",
    "cuda_runtime_version",
}
SCHEMA2_ONLY_RECORD_FIELDS = {
    "variant", "production_selectable", "backend", "cuda_engine",
    "trace_policy", "trace_solver", "trace_preconditioner",
    "trace_ilu_type",
    "trace_ilu_level_of_fill", "trace_ilu_local_reordering",
    "trace_ilu_application_iterations", "trace_ilu_accounting_kind",
    "trace_ilu_factor_reuse_policy", "trace_ilu_triangular_solve_kind",
    "trace_ilu_lower_jacobi_iterations",
    "trace_ilu_upper_jacobi_iterations", "time_integrator",
    "time_step_policy", "gamma", "interior_components",
    "trace_components", "gpu_name", "gpu_device_ordinal", "gpu_uuid",
    "gpu_compute_capability", "cuda_driver_version",
    "cuda_runtime_version", "initial_state_kind", "initial_trace_kind",
    "source_kind", "boundary_kind", "precision_policy",
    "numerical_policy", "initialization_in_primary_timing",
    "profiling_observability", "profile_measured_step",
    "initialization_duration_seconds", "device_memory_snapshots",
    "umpire_allocator_high_watermarks_after_samples", "build_identity",
}
SCHEMA2_ONLY_SAMPLE_FIELDS = {
    "attempt_sequence", "retry_index", "device_ticket_count",
    "linear_solve_attempts", "newton_correction_linear_iterations",
    "final_coherent_linear_iterations", "true_residual_tightenings",
    "iterative_refinement_steps", "residual_replacements",
    "precision_corrections", "all_fp64_policy",
    "correction_accounting_complete", "product_generations", "tensor_core",
    "trace_solver", "native_rank1_trace_solve_calls",
    "hypre_trace_solve_calls", "native_rank1_trace_solver_configured",
    "hypre_mgr_rank1_trace_solver_configured",
    "hypre_host_staged_rank2_trace_solver_configured",
    "trace_solver_accounting_complete", "equilibration_builds",
    "equilibration_passes", "mgr_setup_calls",
    "mgr_preconditioner_applications", "mgr_f_ilu_setups",
    "mgr_f_ilu_applications", "minimum_left_scale",
    "maximum_left_scale", "minimum_right_scale", "maximum_right_scale",
}

FAKE_BENCHMARK = r'''#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=("conventional", "igr"))
parser.add_argument("--engine", default="ordinary", choices=("ordinary", "tensor"))
parser.add_argument(
    "--trace-policy", default="engine-default",
    choices=("engine-default", "common-fast", "equilibrated-mgr"),
)
parser.add_argument("--order", required=True, type=int)
parser.add_argument("--mesh-n", required=True, type=int)
parser.add_argument("--domain-periods", required=True, type=int)
parser.add_argument("--dt", required=True, type=float)
parser.add_argument("--expected-source-revision", default="")
parser.add_argument("--expected-source-tree", default="")
parser.add_argument("--expected-source-status", default="")
parser.add_argument("--expected-build-directory", default="")
parser.add_argument("--expected-cmake-cache", default="")
parser.add_argument("--export", dest="export", action="store_true")
parser.add_argument("--no-export", dest="export", action="store_false")
parser.set_defaults(export=False)
args = parser.parse_args()

is_frozen = "frozen" in str(Path(__file__))
common_fast = args.trace_policy == "common-fast"
equilibrated_mgr = args.trace_policy == "equilibrated-mgr"
if equilibrated_mgr and (args.engine != "tensor" or args.mode != "igr"):
    parser.error("equilibrated-mgr requires tensor IGR")
if equilibrated_mgr:
    trace_preconditioner = "hypre_mgr_field_split"
elif args.mode == "conventional" and \
        (common_fast or args.engine == "tensor"):
    trace_preconditioner = "hypre_diag_scale"
else:
    trace_preconditioner = "hypre_ilu0"
ilu_policy = (
    (-1, -1, -1, 0)
    if trace_preconditioner in ("hypre_diag_scale", "hypre_mgr_field_split")
    else (0, 0, 1 if args.mode == "igr" else 0, 1)
)
ilu_factor_reuse_policy = (
    "not_applicable"
    if ilu_policy[3] == 0
    else "first_trace_solve_per_attempt"
    if args.mode == "igr" and (common_fast or args.engine == "tensor")
    else "refresh_each_trace_solve"
)
ilu_triangular_policy = (
    ("not_applicable", 0, 0, 0)
    if not ilu_policy[3]
    else ("iterative_jacobi", 5, 5, 8)
    if args.mode == "igr" and (common_fast or args.engine == "tensor")
    else ("direct", 0, 0, 0)
)
trace_solver = (
    "hypre-fgmres-mgr-rank1"
    if equilibrated_mgr else "native-rank1-fp64"
    if args.engine == "tensor" else "hypre-parcsr-gmres"
)
if is_frozen:
    scale = 2.5
elif args.engine == "ordinary":
    scale = 1.5 if common_fast else 2.0
else:
    scale = 1.0
scale *= 1.0 + 0.1 * (args.order - 1)
scale *= 1.05 if args.mode == "igr" else 1.0
scale *= 1.2 if args.export else 1.0
durations = [scale * value for value in (0.99, 1.0, 1.01, 1.0, 1.0)]
warmup_duration = scale * 1.1
samples = []
for index, duration in [(-1, warmup_duration), *enumerate(durations)]:
    sample = {
        "phase": "warmup" if index == -1 else "measured",
        "measured_index": index,
        "duration_seconds": duration,
        "step": index + 2,
        "generation": index + 2,
        "time": (index + 2) * args.dt,
        "initial_residual": 1.0,
        "final_residual": 1.0e-10,
        "true_relative_residual": 1.0e-12,
        "maximum_conservation_error": 1.0e-14,
        "maximum_conservation_relative_error": 1.0e-14,
        "maximum_wave_speed": 1.5,
        "backend_stable_dt": 0.01,
        "attempt_sequence": index + 2,
        "retry_index": 0,
        "device_ticket_count": 10,
        "nonlinear_updates": 2,
        "trace_solves": 2,
        "linear_iterations": 7,
        "line_search_trials": 2,
        "rejected_line_search_trials": 0,
        "iterative_refinement_steps": 0,
        "residual_replacements": 0,
        "precision_corrections": 0,
        "all_fp64_policy": True,
        "correction_accounting_complete": True,
        "product_generations": {
            "residual": 3,
            "matrix": 3,
            "factor": 3,
            "solution": 3,
            "issued_mask": 15,
        },
        "transfers": {
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 64,
            "device_to_device_bytes": 128,
            "mpi_staging_bytes": 0,
            "synchronization_events": 2,
            "event_count": 4,
            "complete": True,
        },
        "gates": {
            "candidate_ready": True,
            "finite": True,
            "admissible": True,
            "recovery": True,
            "true_residual": True,
            "conservation": True,
            "telemetry": True,
            "products_coupled_and_ordered": True,
            "success": True,
        },
    }
    if not is_frozen:
        sample["linear_solve_attempts"] = 2
        sample["newton_correction_linear_iterations"] = 4
        sample["final_coherent_linear_iterations"] = 3
        sample["true_residual_tightenings"] = 0
        ilu_active = ilu_policy[3] > 0
        krylov_restart_dimension = min(
            100,
            max(20, 2 * args.mesh_n * args.mesh_n * (args.order + 1) *
                (5 if args.mode == "igr" else 4)),
        )
        callback_lower = (
            7 + (7 + krylov_restart_dimension - 1) //
            krylov_restart_dimension
            if ilu_active else 0
        )
        callback_upper = 14 if ilu_active else 0
        sample["tensor_core"] = {
            "selected_engine": args.engine,
            "trace_policy": args.trace_policy,
            "trace_solver": trace_solver,
            "native_rank1_trace_solve_calls": (
                sample["linear_solve_attempts"]
                if args.engine == "tensor" and not equilibrated_mgr else 0
            ),
            "hypre_trace_solve_calls": (
                0 if args.engine == "tensor" and not equilibrated_mgr
                else sample["linear_solve_attempts"]
            ),
            "native_rank1_trace_solver_configured": (
                args.engine == "tensor" and not equilibrated_mgr
            ),
            "hypre_mgr_rank1_trace_solver_configured": equilibrated_mgr,
            "hypre_host_staged_rank2_trace_solver_configured": False,
            "trace_solver_accounting_complete": True,
            "trace_preconditioner": trace_preconditioner,
            "ilu_type": ilu_policy[0],
            "ilu_level_of_fill": ilu_policy[1],
            "ilu_local_reordering": ilu_policy[2],
            "ilu_application_iterations": ilu_policy[3],
            "ilu_triangular_solve_kind": ilu_triangular_policy[0],
            "ilu_lower_jacobi_iterations": ilu_triangular_policy[1],
            "ilu_upper_jacobi_iterations": ilu_triangular_policy[2],
            "ilu_factor_reuse_policy": ilu_factor_reuse_policy,
            "ilu_factor_setups": (
                1 if ilu_factor_reuse_policy ==
                "first_trace_solve_per_attempt"
                else 2 if ilu_active else 0
            ),
            "ilu_factor_reuses": (
                1 if ilu_factor_reuse_policy ==
                "first_trace_solve_per_attempt" else 0
            ),
            "ilu_accounting_kind": (
                "derived_bounds_v1" if ilu_active else "unavailable"
            ),
            "ilu_preconditioner_callbacks_lower_bound": callback_lower,
            "ilu_preconditioner_callbacks_upper_bound": callback_upper,
            "ilu_inner_iterations_lower_bound": (
                callback_lower * ilu_policy[3]
            ),
            "ilu_inner_iterations_upper_bound": (
                callback_upper * ilu_policy[3]
            ),
            "ilu_jacobi_factor_spmv_updates_lower_bound": (
                callback_lower * ilu_policy[3] * ilu_triangular_policy[3]
            ),
            "ilu_jacobi_factor_spmv_updates_upper_bound": (
                callback_upper * ilu_policy[3] * ilu_triangular_policy[3]
            ),
            "ilu_accounting_complete": True,
            "ilu_factor_age_generations": (
                1 if ilu_factor_reuse_policy ==
                "first_trace_solve_per_attempt" else 0
            ),
            "equilibration_builds": (
                sample["trace_solves"] if equilibrated_mgr else 0
            ),
            "equilibration_passes": (
                2 * sample["trace_solves"] if equilibrated_mgr else 0
            ),
            "mgr_setup_calls": (
                sample["trace_solves"] if equilibrated_mgr else 0
            ),
            "mgr_preconditioner_applications": (
                9 if equilibrated_mgr else 0
            ),
            "mgr_f_ilu_setups": (
                sample["trace_solves"] if equilibrated_mgr else 0
            ),
            "mgr_f_ilu_applications": 9 if equilibrated_mgr else 0,
            "minimum_left_scale": 0.5 if equilibrated_mgr else 0.0,
            "maximum_left_scale": 2.0 if equilibrated_mgr else 0.0,
            "minimum_right_scale": 0.25 if equilibrated_mgr else 0.0,
            "maximum_right_scale": 4.0 if equilibrated_mgr else 0.0,
            "trace_preconditioner_configured": True,
            "hardware_supported": True,
            "condensation_launches": 4 if args.engine == "tensor" else 0,
            "mma_tiles": 64 if args.engine == "tensor" else 0,
            "fallback_count": 0,
            "fallback_accounting_complete": True,
            "no_fallback": True,
        }
    else:
        for field in (
            "attempt_sequence", "retry_index", "device_ticket_count",
            "iterative_refinement_steps", "residual_replacements",
            "precision_corrections", "all_fp64_policy",
            "correction_accounting_complete", "product_generations",
        ):
            del sample[field]
    samples.append(sample)
payload = {
    "schema": 1 if is_frozen else 2,
    "benchmark": (
        "dg_igr_hdg_cuda_accepted_step_fp64"
        if is_frozen else "dg_igr_hdg_cuda_accepted_step"
    ),
    "success": True,
    "backend": "cuda",
    "cuda_engine": args.engine,
    "trace_policy": args.trace_policy,
    "trace_solver": trace_solver,
    "trace_preconditioner": trace_preconditioner,
    "trace_ilu_type": ilu_policy[0],
    "trace_ilu_level_of_fill": ilu_policy[1],
    "trace_ilu_local_reordering": ilu_policy[2],
    "trace_ilu_application_iterations": ilu_policy[3],
    "trace_ilu_triangular_solve_kind": ilu_triangular_policy[0],
    "trace_ilu_lower_jacobi_iterations": ilu_triangular_policy[1],
    "trace_ilu_upper_jacobi_iterations": ilu_triangular_policy[2],
    "trace_ilu_accounting_kind": (
        "derived_bounds_v1" if ilu_policy[3] else "unavailable"
    ),
    "trace_ilu_factor_reuse_policy": ilu_factor_reuse_policy,
    "mode": args.mode,
    "precision": "fp64",
    "time_integrator": "backward_euler",
    "time_step_policy": "fixed",
    "gamma": 1.4,
    "rank_count": 1,
    "polynomial_order": args.order,
    "mesh_n": args.mesh_n,
    "domain_periods": args.domain_periods,
    "domain_length": args.domain_periods,
    "cells_per_period": args.mesh_n // args.domain_periods,
    "element_h": args.domain_periods / args.mesh_n,
    "elements": args.mesh_n * args.mesh_n,
    "interior_components": 7 if args.mode == "igr" else 4,
    "trace_components": 5 if args.mode == "igr" else 4,
    "interior_unknowns": (
        args.mesh_n * args.mesh_n * (args.order + 1) ** 2 *
        (7 if args.mode == "igr" else 4)
    ),
    "true_trace_unknowns": (
        2 * args.mesh_n * args.mesh_n * (args.order + 1) *
        (5 if args.mode == "igr" else 4)
    ),
    "dt": args.dt,
    "host_export": args.export,
    "gpu_name": "Fake SM120",
    "gpu_device_ordinal": 0,
    "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000001",
    "gpu_compute_capability": 120,
    "gpu_sm_count": 84,
    "cuda_driver_version": 13000,
    "cuda_runtime_version": 13000,
    "principal_threads": 128,
    "principal_element_threads": 4,
    "principal_elements_per_block": 32,
    "principal_blocks": (args.mesh_n * args.mesh_n + 31) // 32,
    "principal_execution": "four_lane_subwarp_per_element",
    "initial_state_kind": "projected_smooth_periodic",
    "initial_trace_kind": "projected_from_committed_interior",
    "source_kind": "zero",
    "boundary_kind": "periodic_topology",
    "precision_policy": {
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
    },
    "numerical_policy": {
        "volume_projection_quadrature_order": 2 * args.order + 2,
        "true_trace_projection_quadrature_order": 2 * args.order + 2,
        "conventional_assembly_quadrature_order": 2 * args.order + 2,
        "igr_volume_assembly_quadrature_order": 12,
        "igr_face_assembly_quadrature_order": 12,
        "maximum_nonlinear_updates": 80,
        "requested_maximum_linear_iterations": 0,
        "resolved_maximum_linear_iterations": max(
            100,
            2 * args.mesh_n * args.mesh_n * (args.order + 1) *
            (5 if args.mode == "igr" else 4),
        ),
        "krylov_restart_dimension": min(
            100,
            max(20, 2 * args.mesh_n * args.mesh_n * (args.order + 1) *
                (5 if args.mode == "igr" else 4)),
        ),
        "maximum_line_search_trials": 13,
        "nonlinear_relative_tolerance": 1.0e-10,
        "newton_correction_linear_relative_tolerance": 1.0e-12,
        "final_coherent_linear_relative_tolerance": 1.0e-12,
        "true_residual_tightening_linear_relative_tolerance": 1.0e-12,
        "true_residual_relative_tolerance": 1.0e-10,
        "conservation_relative_tolerance": 1.0e-10,
        "armijo_constant": 1.0e-4,
        "regularization": "igr" if args.mode == "igr" else "none",
        "igr_alpha": 0.2 if args.mode == "igr" else 0.0,
        "igr_beta": 0.8 if args.mode == "igr" else 0.0,
    },
    "initialization_in_primary_timing": False,
    "initialization_timed": False,
    "profiling_observability": False,
    "profile_measured_step": -1,
    "initialization_duration_seconds": 0.25,
    "device_memory_snapshots": {
        "after_device_selection_before_problem": {
            "free_bytes": 900,
            "total_bytes": 1000,
            "used_bytes": 100,
        },
        "after_initialization": {
            "free_bytes": 700,
            "total_bytes": 1000,
            "used_bytes": 300,
        },
        "after_samples": {
            "free_bytes": 650,
            "total_bytes": 1000,
            "used_bytes": 350,
        },
    },
    "umpire_allocator_high_watermarks_after_samples": {
        "available": True,
        "mfem_available": True,
        "hypre_available": True,
        "sum_available": True,
        "mfem_bytes": 100,
        "hypre_bytes": 50,
        "sum_bytes": 150,
    },
    "timing_scope": (
        "Prepare+CapturePreparedObservation+"
        + ("ExportPreparedInteriorD2H+" if args.export else "")
        + "AcknowledgePreparedOutput+Commit"
    ),
    "warmup_steps": 1,
    "measured_steps": 5,
    "warmup_duration_seconds": warmup_duration,
    "raw_durations_seconds": durations,
    "median_seconds": scale,
    "population_stddev_seconds": statistics.pstdev(durations),
    "samples": samples,
}
if is_frozen:
    for field in (
        "backend", "cuda_engine", "trace_policy", "trace_solver",
        "trace_preconditioner",
        "trace_ilu_type", "trace_ilu_level_of_fill",
        "trace_ilu_local_reordering", "trace_ilu_application_iterations",
        "trace_ilu_triangular_solve_kind",
        "trace_ilu_lower_jacobi_iterations",
        "trace_ilu_upper_jacobi_iterations",
        "trace_ilu_accounting_kind", "trace_ilu_factor_reuse_policy",
        "time_integrator", "time_step_policy", "gamma",
        "interior_components", "trace_components", "gpu_name",
        "gpu_device_ordinal", "gpu_uuid", "gpu_compute_capability",
        "cuda_driver_version", "cuda_runtime_version",
        "initial_state_kind", "initial_trace_kind", "source_kind",
        "boundary_kind", "precision_policy", "numerical_policy",
        "initialization_in_primary_timing", "profiling_observability",
        "profile_measured_step", "initialization_duration_seconds",
        "device_memory_snapshots",
        "umpire_allocator_high_watermarks_after_samples",
    ):
        del payload[field]
else:
    payload["variant"] = (
        "production" if args.trace_policy == "engine-default" else "ablation"
    )
    payload["production_selectable"] = (
        args.trace_policy == "engine-default"
    )
    payload["build_identity"] = {
        "source_revision": args.expected_source_revision,
        "source_tree": args.expected_source_tree,
        "source_status": args.expected_source_status,
        "build_directory": args.expected_build_directory,
        "cmake_cache": args.expected_cmake_cache,
    }
print(json.dumps(payload))
'''

FAKE_NVIDIA_SMI = r'''#!/usr/bin/env python3
import sys

query = next(argument for argument in sys.argv if argument.startswith("--query-gpu="))
field = query.split("=", 1)[1]
values = {
    "index": "0",
    "uuid": "GPU-00000000-0000-0000-0000-000000000001",
    "name": "Fake SM120",
    "pci.bus_id": "00000000:01:00.0",
    "driver_version": "600.1",
    "compute_cap": "12.0",
    "memory.total": "49140",
    "timestamp": "2026/08/23 00:00:00.000",
    "pstate": "P0",
    "clocks.current.graphics": "2100",
    "clocks.current.sm": "2100",
    "clocks.current.memory": "9000",
    "clocks.max.sm": "2400",
    "temperature.gpu": "52",
    "power.draw": "230.0",
    "power.limit": "600.0",
    "utilization.gpu": "0",
    "utilization.memory": "0",
    "clocks_event_reasons.active": "None",
    "clocks_throttle_reasons.active": "None",
}
print(values[field])
'''


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def write_build_provenance(root: Path, source: Path,
                           build_name: str = "build") -> tuple[Path, Path]:
    build = root / build_name
    build.mkdir()
    cache = build / "CMakeCache.txt"
    cache.write_text(
        "\n".join((
            "CMAKE_BUILD_TYPE:STRING=Release",
            "CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=120",
            "CMAKE_CUDA_COMPILER:FILEPATH=/fake/nvcc",
            "CMAKE_CXX_COMPILER:FILEPATH=/fake/mpicxx",
            "MFEM_DIR:PATH=/fake/mfem",
            "DG_IGR_ENABLE_HDG_RUNTIME:BOOL=ON",
            "DG_IGR_ENABLE_OBSERVABILITY:BOOL=OFF",
            "DG_IGR_OBSERVABILITY_USE_NVTX:BOOL=OFF",
            f"CMAKE_HOME_DIRECTORY:INTERNAL={source}",
            "",
        )),
        encoding="utf-8",
    )
    mfem = build / "dg-igr-mfem-identity-v1.txt"
    mfem.write_text(
        "\n".join((
            "schema=dg-igr-mfem-identity",
            "version=1",
            f"source_root={source / 'tpl' / 'mfem'}",
            "mode=gpu",
            "profile=cuda-sm120",
            "build_type=Release",
            "hypre_spec=hypre@develop+mpi+cuda cuda_arch=120",
            f"hypre_hash={SYNTHETIC_HYPRE_HASH}",
            "hypre_prefix=/fake/spack/hypre-develop-"
            f"{SYNTHETIC_HYPRE_HASH}",
            f"hypre_source_revision={SYNTHETIC_HYPRE_REVISION}",
            "hypre_configuration=mpi+cuda+device-memory+sm120",
            "dependency_cuda_arch=120",
            "mfem_cuda_archs=80-virtual",
            "reuse_abi=build_type=Release;"
            "hypre_spec=hypre@develop+mpi+cuda cuda_arch=120;"
            f"hypre={SYNTHETIC_HYPRE_HASH};"
            "dependency_arch=120;mfem_arch=80-virtual",
            "",
        )),
        encoding="utf-8",
    )
    return cache, mfem


def rewrite_manifest(source: Path, destination: Path, *,
                     replacements: dict[str, str] | None = None,
                     omitted: set[str] | None = None) -> Path:
    replacements = replacements or {}
    omitted = omitted or set()
    observed: set[str] = set()
    output: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        key, separator, _ = line.partition("=")
        if not separator:
            output.append(line)
        elif key in omitted:
            observed.add(key)
        elif key in replacements:
            output.append(f"{key}={replacements[key]}")
            observed.add(key)
        else:
            output.append(line)
    expected = set(replacements) | omitted
    if observed != expected:
        raise AssertionError(
            f"manifest rewrite missed keys: {sorted(expected - observed)}"
        )
    destination.write_text("\n".join((*output, "")), encoding="utf-8")
    return destination


def main() -> int:
    assert CAMPAIGN._ilu_derived_bounds_are_valid(
        7, 1, 5, 1, 9, 14, 9, 14
    )
    assert CAMPAIGN._ilu_derived_bounds_are_valid(
        0, 1, 5, 1, 0, 0, 0, 0
    )
    assert CAMPAIGN._ilu_derived_bounds_are_valid(
        7, 2, 5, 1, 10, 14, 10, 14
    )
    assert not CAMPAIGN._ilu_derived_bounds_are_valid(
        7, 1, 5, 1, 6, 14, 6, 14
    )
    assert not CAMPAIGN._ilu_derived_bounds_are_valid(
        7, 1, 5, 1, 7, 14, 7, 14
    )
    assert not CAMPAIGN._ilu_derived_bounds_are_valid(
        7, 1, 5, 1, 8, 14, 8, 14
    )
    assert not CAMPAIGN._ilu_derived_bounds_are_valid(
        7, 1, 5, 1, 9, 13, 9, 13
    )
    assert not CAMPAIGN._ilu_derived_bounds_are_valid(
        7, 1, 5, 1, 9, 15, 9, 15
    )
    assert not CAMPAIGN._ilu_derived_bounds_are_valid(
        7, 1, 5, 2, 9, 14, 9, 28
    )
    assert CAMPAIGN._ilu_derived_bounds_are_valid(
        CAMPAIGN.UINT64_MAX // 2, 1,
        CAMPAIGN.UINT64_MAX // 2, 1,
        CAMPAIGN.UINT64_MAX // 2 + 1, CAMPAIGN.UINT64_MAX - 1,
        CAMPAIGN.UINT64_MAX // 2 + 1, CAMPAIGN.UINT64_MAX - 1,
    )
    assert not CAMPAIGN._ilu_derived_bounds_are_valid(
        CAMPAIGN.UINT64_MAX // 2, 1,
        CAMPAIGN.UINT64_MAX // 2, 2,
        CAMPAIGN.UINT64_MAX // 2 + 1, CAMPAIGN.UINT64_MAX - 1,
        CAMPAIGN.UINT64_MAX, CAMPAIGN.UINT64_MAX,
    )
    assert not CAMPAIGN._ilu_derived_bounds_are_valid(
        -1, 1, 5, 1, 0, 0, 0, 0
    )
    assert not CAMPAIGN._ilu_derived_bounds_are_valid(
        1, 1, 0, 1, 1, 2, 1, 2
    )
    assert not CAMPAIGN._ilu_derived_bounds_are_valid(
        1, 0, 5, 1, 2, 2, 2, 2
    )
    assert CAMPAIGN._ilu_jacobi_factor_spmv_bounds_are_valid(
        9, 14, 1, 8, 72, 112
    )
    assert CAMPAIGN._ilu_jacobi_factor_spmv_bounds_are_valid(
        9, 14, 1, 0, 0, 0
    )
    assert not CAMPAIGN._ilu_jacobi_factor_spmv_bounds_are_valid(
        9, 14, 1, 8, 71, 112
    )
    assert not CAMPAIGN._ilu_jacobi_factor_spmv_bounds_are_valid(
        9, CAMPAIGN.UINT64_MAX, 1, 8, 72, CAMPAIGN.UINT64_MAX
    )
    assert not CAMPAIGN._ilu_jacobi_factor_spmv_bounds_are_valid(
        9, 14, 0, 8, 0, 0
    )
    assert CAMPAIGN._ilu_factor_accounting_is_valid(
        2, 2, "refresh_each_trace_solve", 2, 0, 0
    )
    assert CAMPAIGN._ilu_factor_accounting_is_valid(
        1, 1, "first_trace_solve_per_attempt", 1, 0, 0
    )
    assert CAMPAIGN._ilu_factor_accounting_is_valid(
        2, 2, "first_trace_solve_per_attempt", 1, 1, 1
    )
    assert CAMPAIGN._ilu_factor_accounting_is_valid(
        2, 2, "not_applicable", 0, 0, 0
    )
    assert not CAMPAIGN._ilu_factor_accounting_is_valid(
        2, 2, "refresh_each_trace_solve", 1, 1, 1
    )
    assert not CAMPAIGN._ilu_factor_accounting_is_valid(
        2, 2, "first_trace_solve_per_attempt", 1, 1, 0
    )
    assert not CAMPAIGN._ilu_factor_accounting_is_valid(
        1, 1, "first_trace_solve_per_attempt", 1, 0, 1
    )
    assert not CAMPAIGN._ilu_factor_accounting_is_valid(
        2, 2, "not_applicable", 0, 1, 0
    )
    valid_mgr_aggregates = {
        "expected_mgr": True,
        "trace_solves": 2,
        "linear_iterations": 7,
        "equilibration_builds": 2,
        "equilibration_passes": 4,
        "mgr_setup_calls": 2,
        "mgr_preconditioner_applications": 9,
        "mgr_f_ilu_setups": 2,
        "mgr_f_ilu_applications": 9,
        "minimum_left_scale": 0.5,
        "maximum_left_scale": 2.0,
        "minimum_right_scale": 0.25,
        "maximum_right_scale": 4.0,
    }
    assert CAMPAIGN._mgr_accounting_is_valid(**valid_mgr_aggregates)
    for field, value in (
            ("equilibration_builds", 1),
            ("equilibration_passes", 3),
            ("mgr_setup_calls", 1),
            ("mgr_f_ilu_setups", 1),
            ("mgr_f_ilu_applications", 8),
            ("minimum_left_scale", 0.0),
            ("maximum_right_scale", 0.1)):
        invalid = {**valid_mgr_aggregates, field: value}
        assert not CAMPAIGN._mgr_accounting_is_valid(**invalid)
    assert CAMPAIGN._mgr_accounting_is_valid(
        expected_mgr=False,
        trace_solves=2,
        linear_iterations=7,
        equilibration_builds=0,
        equilibration_passes=0,
        mgr_setup_calls=0,
        mgr_preconditioner_applications=0,
        mgr_f_ilu_setups=0,
        mgr_f_ilu_applications=0,
        minimum_left_scale=0.0,
        maximum_left_scale=0.0,
        minimum_right_scale=0.0,
        maximum_right_scale=0.0,
    )

    with tempfile.TemporaryDirectory(prefix="hdg-tensor-campaign-test-") as temp:
        root = Path(temp)
        source = root / "source"
        source.mkdir()
        (source / "tpl" / "mfem").mkdir(parents=True)
        (source / "marker.txt").write_text("clean test source\n", encoding="utf-8")
        write_executable(source / "build.sh", "#!/bin/sh\nexit 0\n")
        run(["git", "init", "-q"], source)
        run(["git", "config", "user.name", "Campaign Test"], source)
        run(["git", "config", "user.email", "campaign-test@example.invalid"], source)
        run(["git", "add", "."], source)
        run(["git", "commit", "-q", "-m", "test source"], source)
        commit = run(["git", "rev-parse", "HEAD"], source).stdout.strip()
        tree = run(["git", "rev-parse", "HEAD^{tree}"], source).stdout.strip()

        frozen_source = root / "frozen-source"
        frozen_target = frozen_source / CAMPAIGN.FROZEN_OVERLAY_PATH
        frozen_target.parent.mkdir(parents=True)
        frozen_target.write_text("constexpr int order = 1;\n", encoding="utf-8")
        write_executable(frozen_source / "build.sh", "#!/bin/sh\nexit 0\n")
        run(["git", "init", "-q"], frozen_source)
        run(["git", "config", "user.name", "Campaign Test"], frozen_source)
        run(["git", "config", "user.email", "campaign-test@example.invalid"],
            frozen_source)
        run(["git", "add", "."], frozen_source)
        run(["git", "commit", "-q", "-m", "synthetic frozen"], frozen_source)
        synthetic_frozen_commit = run(
            ["git", "rev-parse", "HEAD"], frozen_source
        ).stdout.strip()
        synthetic_frozen_tree = run(
            ["git", "rev-parse", "HEAD^{tree}"], frozen_source
        ).stdout.strip()
        frozen_target.write_text(
            "struct Options { int order = 1; };\n"
            "// --order\n"
            "int selected = options.order;\n",
            encoding="utf-8",
        )
        saved_commit = CAMPAIGN.FROZEN_CONTROL_COMMIT
        saved_tree = CAMPAIGN.FROZEN_CONTROL_TREE
        saved_overlay = CAMPAIGN.FROZEN_OVERLAY_DIFF_SHA256
        CAMPAIGN.FROZEN_CONTROL_COMMIT = synthetic_frozen_commit
        CAMPAIGN.FROZEN_CONTROL_TREE = synthetic_frozen_tree
        synthetic_overlay = run(
            ["git", "diff", "--binary", "--", str(CAMPAIGN.FROZEN_OVERLAY_PATH)],
            frozen_source,
        ).stdout.encode()
        CAMPAIGN.FROZEN_OVERLAY_DIFF_SHA256 = hashlib.sha256(
            synthetic_overlay
        ).hexdigest()
        try:
            overlay = CAMPAIGN.frozen_overlay_identity(frozen_source)
        finally:
            CAMPAIGN.FROZEN_CONTROL_COMMIT = saved_commit
            CAMPAIGN.FROZEN_CONTROL_TREE = saved_tree
            CAMPAIGN.FROZEN_OVERLAY_DIFF_SHA256 = saved_overlay
        assert overlay["changed_paths"] == [
            CAMPAIGN.FROZEN_OVERLAY_PATH.as_posix()
        ]
        assert overlay["overlay_kind"] == "benchmark_order_cli_p1_p3_only"
        assert len(overlay["diff_sha256"]) == 64

        nvidia_smi = root / "nvidia-smi"
        write_executable(nvidia_smi, FAKE_NVIDIA_SMI)
        build_cache, mfem_identity = write_build_provenance(root, source)
        final = build_cache.parent / CAMPAIGN.BENCHMARK_BUILD_RELATIVE_PATH
        final.parent.mkdir(parents=True)
        write_executable(final, FAKE_BENCHMARK)

        build_provenance = CAMPAIGN.validated_build_provenance(
            source_root=source,
            cmake_cache_path=str(build_cache),
            mfem_identity_path=str(mfem_identity),
        )
        hypre_identity = build_provenance["validated_hypre_identity"]
        assert hypre_identity["spack_spec"] == \
            "hypre@develop+mpi+cuda cuda_arch=120"
        assert hypre_identity["spack_dag_hash"] == SYNTHETIC_HYPRE_HASH
        assert hypre_identity["reuse_abi_binding"] == {
            "hypre_spec": "hypre@develop+mpi+cuda cuda_arch=120",
            "hypre": SYNTHETIC_HYPRE_HASH,
        }
        assert hypre_identity["manifest_fields"][
            "hypre_source_revision"
        ] == SYNTHETIC_HYPRE_REVISION
        assert hypre_identity["manifest_fields"][
            "hypre_configuration"
        ] == "mpi+cuda+device-memory+sm120"
        minimal_hypre_manifest = rewrite_manifest(
            mfem_identity, root / "mfem-no-optional-hypre-fields.txt",
            omitted={"hypre_source_revision", "hypre_configuration"},
        )
        minimal_hypre_identity = CAMPAIGN.validated_build_provenance(
            source_root=source,
            cmake_cache_path=str(build_cache),
            mfem_identity_path=str(minimal_hypre_manifest),
        )["validated_hypre_identity"]
        assert minimal_hypre_identity["manifest_fields"] == {
            "hypre_hash": SYNTHETIC_HYPRE_HASH,
            "hypre_prefix": "/fake/spack/hypre-develop-"
            f"{SYNTHETIC_HYPRE_HASH}",
            "hypre_spec": "hypre@develop+mpi+cuda cuda_arch=120",
        }

        def provenance_rejected(path: Path) -> bool:
            try:
                CAMPAIGN.validated_build_provenance(
                    source_root=source,
                    cmake_cache_path=str(build_cache),
                    mfem_identity_path=str(path),
                )
            except CAMPAIGN.CampaignError:
                return True
            return False

        bad_provenance_cases = (
            rewrite_manifest(
                mfem_identity, root / "mfem-missing-hypre-hash.txt",
                omitted={"hypre_hash"},
            ),
            rewrite_manifest(
                mfem_identity, root / "mfem-malformed-hypre-hash.txt",
                replacements={"hypre_hash": "not-a-spack-dag-hash"},
            ),
            rewrite_manifest(
                mfem_identity, root / "mfem-mismatched-hypre-prefix.txt",
                replacements={
                    "hypre_prefix": "/fake/spack/hypre-develop-"
                    + "a" * 32,
                },
            ),
            rewrite_manifest(
                mfem_identity, root / "mfem-unbound-hypre-reuse.txt",
                replacements={
                    "reuse_abi": "build_type=Release;"
                    "hypre_spec=hypre@develop+mpi+cuda cuda_arch=120;"
                    f"hypre={'a' * 32};dependency_arch=120;"
                    "mfem_arch=80-virtual",
                },
            ),
            rewrite_manifest(
                mfem_identity, root / "mfem-unsupported-hypre-config.txt",
                replacements={
                    "hypre_spec": "hypre@develop+mpi~cuda cuda_arch=120",
                },
            ),
            rewrite_manifest(
                mfem_identity, root / "mfem-wrong-hypre-architecture.txt",
                replacements={
                    "hypre_spec": "hypre@develop+mpi+cuda cuda_arch=1200",
                },
            ),
            rewrite_manifest(
                mfem_identity, root / "mfem-empty-hypre-revision.txt",
                replacements={"hypre_source_revision": ""},
            ),
            rewrite_manifest(
                mfem_identity, root / "mfem-symbolic-hypre-revision.txt",
                replacements={"hypre_source_revision": "develop"},
            ),
        )
        assert all(provenance_rejected(path) for path in bad_provenance_cases)

        explicit_build = {
            "source_revision": commit,
            "source_tree": tree,
            "source_status": "clean",
            "build_directory": str(build_cache.resolve().parent),
            "cmake_cache": str(build_cache.resolve()),
        }
        explicit_gpu = {
            "device_ordinal": 0,
            "uuid": "GPU-00000000-0000-0000-0000-000000000001",
            "compute_capability": 120,
        }
        mgr_command = CAMPAIGN.final_command(
            final,
            engine="tensor",
            mode="igr",
            order=2,
            mesh_n=12,
            domain_periods=3,
            timestep=1.0e-5,
            export=False,
            expected_build=explicit_build,
            trace_policy="equilibrated-mgr",
        )
        assert mgr_command[mgr_command.index("--trace-policy") + 1] == \
            "equilibrated-mgr"
        mgr_record = json.loads(run(mgr_command, root).stdout)
        mgr_validation_arguments = {
            "source": "final",
            "mode": "igr",
            "order": 2,
            "mesh_n": 12,
            "domain_periods": 3,
            "timestep": 1.0e-5,
            "export": False,
            "engine": "tensor",
            "expected_build": explicit_build,
            "expected_gpu": explicit_gpu,
            "trace_policy": "equilibrated-mgr",
        }
        CAMPAIGN.validate_record(mgr_record, **mgr_validation_arguments)
        assert mgr_record["variant"] == "ablation"
        assert mgr_record["production_selectable"] is False
        assert mgr_record["trace_solver"] == "hypre-fgmres-mgr-rank1"
        assert mgr_record["trace_preconditioner"] == \
            "hypre_mgr_field_split"
        assert (
            mgr_record["trace_ilu_type"],
            mgr_record["trace_ilu_level_of_fill"],
            mgr_record["trace_ilu_local_reordering"],
            mgr_record["trace_ilu_application_iterations"],
        ) == (-1, -1, -1, 0)
        assert mgr_record["trace_ilu_factor_reuse_policy"] == \
            "not_applicable"
        assert mgr_record["trace_ilu_triangular_solve_kind"] == \
            "not_applicable"
        for sample in mgr_record["samples"]:
            tensor = sample["tensor_core"]
            assert sample["linear_solve_attempts"] == \
                sample["trace_solves"] + sample["true_residual_tightenings"]
            assert sample["true_residual_tightenings"] == 0
            assert tensor["hypre_trace_solve_calls"] == \
                sample["linear_solve_attempts"]
            assert tensor["native_rank1_trace_solve_calls"] == 0
            assert tensor["native_rank1_trace_solver_configured"] is False
            assert tensor["hypre_mgr_rank1_trace_solver_configured"] is True
            assert tensor[
                "hypre_host_staged_rank2_trace_solver_configured"
            ] is False
            assert tensor["equilibration_builds"] == sample["trace_solves"]
            assert tensor["equilibration_passes"] == \
                2 * sample["trace_solves"]
            assert tensor["mgr_setup_calls"] == sample["trace_solves"]
            assert tensor["mgr_preconditioner_applications"] > 0
            assert tensor["mgr_f_ilu_setups"] == \
                tensor["mgr_setup_calls"]
            assert tensor["mgr_f_ilu_applications"] == \
                tensor["mgr_preconditioner_applications"]
            assert not set(tensor).intersection({
                "mgr_history", "equilibration_history", "event_stream",
                "flight_recorder", "replay",
            })
        mgr_record_summary = CAMPAIGN.record_summary(mgr_record)
        assert all(
            sample["mgr_f_ilu_applications"] ==
            sample["mgr_preconditioner_applications"]
            for sample in mgr_record_summary["measured_sample_work"]
        )
        mgr_work_summary = CAMPAIGN.work_summary([mgr_record])["tensor_core"]
        assert mgr_work_summary["trace_solvers"] == [
            "hypre-fgmres-mgr-rank1"
        ]
        assert mgr_work_summary[
            "hypre_mgr_rank1_trace_solver_configured_values"
        ] == [True]
        assert mgr_work_summary["equilibration_builds"]["total"] == 10
        assert mgr_work_summary["equilibration_passes"]["total"] == 20
        assert mgr_work_summary["mgr_setup_calls"]["total"] == 10
        assert mgr_work_summary["mgr_f_ilu_setups"]["total"] == 10
        assert mgr_work_summary["mgr_preconditioner_applications"][
            "total"
        ] == mgr_work_summary["mgr_f_ilu_applications"]["total"]

        def mgr_rejected(record: dict) -> bool:
            try:
                CAMPAIGN.validate_record(record, **mgr_validation_arguments)
            except CAMPAIGN.CampaignError:
                return True
            return False

        for field, value in (
                ("hypre_trace_solve_calls", 1),
                ("equilibration_builds", 1),
                ("equilibration_passes", 3),
                ("mgr_setup_calls", 1),
                ("mgr_preconditioner_applications", 0),
                ("mgr_f_ilu_setups", 1),
                ("mgr_f_ilu_applications", 8),
                ("minimum_left_scale", 0.0),
                ("maximum_right_scale", 0.1)):
            invalid_mgr = copy.deepcopy(mgr_record)
            invalid_mgr["samples"][1]["tensor_core"][field] = value
            assert mgr_rejected(invalid_mgr), field
        invalid_mgr_flag = copy.deepcopy(mgr_record)
        invalid_mgr_flag["samples"][1]["tensor_core"][
            "hypre_mgr_rank1_trace_solver_configured"
        ] = False
        assert mgr_rejected(invalid_mgr_flag)
        invalid_mgr_solver = copy.deepcopy(mgr_record)
        invalid_mgr_solver["trace_solver"] = "native-rank1-fp64"
        assert mgr_rejected(invalid_mgr_solver)
        impossible_tightening = copy.deepcopy(mgr_record)
        impossible_sample = impossible_tightening["samples"][1]
        impossible_sample["linear_solve_attempts"] = 3
        impossible_sample["true_residual_tightenings"] = 1
        impossible_sample["tensor_core"]["hypre_trace_solve_calls"] = 3
        assert mgr_rejected(impossible_tightening)
        for engine, mode in (
                ("ordinary", "igr"),
                ("ordinary", "conventional"),
                ("tensor", "conventional")):
            try:
                CAMPAIGN.final_command(
                    final,
                    engine=engine,
                    mode=mode,
                    order=2,
                    mesh_n=12,
                    domain_periods=3,
                    timestep=1.0e-5,
                    export=False,
                    expected_build=explicit_build,
                    trace_policy="equilibrated-mgr",
                )
            except CAMPAIGN.CampaignError:
                pass
            else:
                raise AssertionError(
                    "MGR command admitted an unsupported engine/mode"
                )

        output = root / "evidence"
        command = [
            sys.executable,
            str(SCRIPT),
            "--benchmark", str(final),
            "--output-dir", str(output),
            "--source-root", str(source),
            "--evidence-mode", "smoke",
            "--pairs", "2",
            "--workload-class", "break-even",
            "--workload-label", "synthetic-break-even",
            "--mesh-n", "12",
            "--domain-periods", "3",
            "--dt", "1e-5",
            "--export-control-mode", "igr",
            "--export-control-order", "3",
            "--build-label", "synthetic-final",
            "--build-metadata", "cuda_arch=120",
            "--build-manifest", str(build_cache),
            "--mfem-identity", str(mfem_identity),
            "--expected-commit", commit,
            "--expected-tree", tree,
            "--gpu-index", "0",
            "--nvidia-smi", str(nvidia_smi),
            "--require-nvidia-smi",
            "--hardware-metadata", "lease=synthetic",
            "--bootstrap-resamples", "100",
        ]
        completed = run(command, root)
        completion = json.loads(completed.stdout)
        assert completion["execution_completed"] is True
        campaign = json.loads((output / "campaign.json").read_text())
        assert campaign["status"] == "completed"
        assert campaign["execution_status"] == "completed"
        assert campaign["promotion_qualification"] is False
        assert campaign["performance_disposition"][
            "promotion_performance_pass"
        ] is False
        assert campaign["performance_disposition"][
            "promotion_performance_disposition"
        ] == "characterization_only_not_a_promotion_gate"
        assert campaign["performance_disposition"][
            "tensor_vs_final_tree_ordinary"
        ]["attribution_by_mode"]["igr"]["trace_policy_matched"] is False
        assert campaign["export_control"]["attribution"][
            "tensor_trace_preconditioner"
        ] == "hypre_ilu0"
        assert campaign["export_control"]["attribution"][
            "mma_contribution_requires_separate_profiler_attribution"
        ] is True
        assert campaign["source_start"] == campaign["source_end"]
        assert len(campaign["final_tree_results"]) == 6
        assert "frozen_control_results" not in campaign
        assert len(campaign["invocations"]) == 26
        assert len(list((output / "raw").rglob("*.json"))) == 26
        assert CAMPAIGN.expected_trace_preconditioner(
            "ordinary", "conventional"
        ) == "hypre_ilu0"
        assert CAMPAIGN.expected_trace_preconditioner(
            "ordinary", "igr"
        ) == "hypre_ilu0"
        assert CAMPAIGN.expected_trace_preconditioner(
            "tensor", "conventional"
        ) == "hypre_diag_scale"
        assert CAMPAIGN.expected_trace_preconditioner(
            "tensor", "igr"
        ) == "hypre_ilu0"
        assert CAMPAIGN.expected_trace_ilu_policy(
            "ordinary", "conventional"
        ) == (0, 0, 0, 1)
        assert CAMPAIGN.expected_trace_ilu_policy(
            "ordinary", "igr"
        ) == (0, 0, 1, 1)
        assert CAMPAIGN.expected_trace_ilu_policy(
            "tensor", "conventional"
        ) == (-1, -1, -1, 0)
        assert CAMPAIGN.expected_trace_ilu_policy(
            "tensor", "igr"
        ) == (0, 0, 1, 1)
        assert CAMPAIGN.expected_trace_ilu_triangular_solve_policy(
            "ordinary", "conventional"
        ) == ("direct", 0, 0, 0)
        assert CAMPAIGN.expected_trace_ilu_triangular_solve_policy(
            "ordinary", "igr"
        ) == ("direct", 0, 0, 0)
        assert CAMPAIGN.expected_trace_ilu_triangular_solve_policy(
            "tensor", "conventional"
        ) == ("not_applicable", 0, 0, 0)
        assert CAMPAIGN.expected_trace_ilu_triangular_solve_policy(
            "tensor", "igr"
        ) == ("iterative_jacobi", 5, 5, 8)
        assert CAMPAIGN.expected_trace_ilu_factor_reuse_policy(
            "ordinary", "conventional"
        ) == "refresh_each_trace_solve"
        assert CAMPAIGN.expected_trace_ilu_factor_reuse_policy(
            "ordinary", "igr"
        ) == "refresh_each_trace_solve"
        assert CAMPAIGN.expected_trace_ilu_factor_reuse_policy(
            "tensor", "conventional"
        ) == "not_applicable"
        assert CAMPAIGN.expected_trace_ilu_factor_reuse_policy(
            "tensor", "igr"
        ) == "first_trace_solve_per_attempt"
        for engine in ("ordinary", "tensor"):
            assert CAMPAIGN.expected_trace_preconditioner(
                engine, "conventional", "common-fast"
            ) == "hypre_diag_scale"
            assert CAMPAIGN.expected_trace_ilu_policy(
                engine, "conventional", "common-fast"
            ) == (-1, -1, -1, 0)
            assert CAMPAIGN.expected_trace_preconditioner(
                engine, "igr", "common-fast"
            ) == "hypre_ilu0"
            assert CAMPAIGN.expected_trace_ilu_policy(
                engine, "igr", "common-fast"
            ) == (0, 0, 1, 1)
            assert CAMPAIGN.expected_trace_ilu_triangular_solve_policy(
                engine, "igr", "common-fast"
            ) == ("iterative_jacobi", 5, 5, 8)
            assert CAMPAIGN.expected_trace_ilu_factor_reuse_policy(
                engine, "igr", "common-fast"
            ) == "first_trace_solve_per_attempt"
        assert CAMPAIGN.expected_trace_preconditioner(
            "tensor", "igr", "equilibrated-mgr"
        ) == "hypre_mgr_field_split"
        assert CAMPAIGN.expected_trace_ilu_policy(
            "tensor", "igr", "equilibrated-mgr"
        ) == (-1, -1, -1, 0)
        assert CAMPAIGN.expected_trace_ilu_triangular_solve_policy(
            "tensor", "igr", "equilibrated-mgr"
        ) == ("not_applicable", 0, 0, 0)
        assert CAMPAIGN.expected_trace_ilu_factor_reuse_policy(
            "tensor", "igr", "equilibrated-mgr"
        ) == "not_applicable"
        assert CAMPAIGN.expected_trace_solver_configuration(
            "tensor", "igr", "equilibrated-mgr"
        ) == ("hypre-fgmres-mgr-rank1", False, True, False)
        for engine, mode in (
                ("ordinary", "igr"),
                ("ordinary", "conventional"),
                ("tensor", "conventional")):
            try:
                CAMPAIGN.expected_trace_solver_configuration(
                    engine, mode, "equilibrated-mgr"
                )
            except CAMPAIGN.CampaignError:
                pass
            else:
                raise AssertionError(
                    "equilibrated-mgr admitted an unsupported engine/mode"
                )
        for cell_key, result in campaign["final_tree_results"].items():
            comparison = result["paired_comparison"]
            assert math.isclose(comparison["speedup"]["median"], 2.0)
            assert comparison[
                "median_speedup_95_percent_confidence_interval"
            ]["resamples"] == 100
            mode = "igr" if cell_key.startswith("igr_") else "conventional"
            attribution = comparison["attribution"]
            assert attribution["scope"] == "complete_engine_package"
            assert attribution["ordinary_trace_preconditioner"] == "hypre_ilu0"
            assert attribution["tensor_trace_preconditioner"] == (
                "hypre_ilu0" if mode == "igr"
                else "hypre_diag_scale"
            )
            assert attribution[
                "trace_preconditioner_configuration_matched"
            ] is False
            assert attribution["trace_policy_matched"] is False
            assert attribution[
                "ordinary_ilu_factor_reuse_policy"
            ] == "refresh_each_trace_solve"
            assert attribution["tensor_ilu_factor_reuse_policy"] == (
                "first_trace_solve_per_attempt" if mode == "igr"
                else "not_applicable"
            )
            assert attribution[
                "ordinary_ilu_triangular_solve_policy"
            ] == {
                "kind": "direct",
                "lower_jacobi_iterations": 0,
                "upper_jacobi_iterations": 0,
                "jacobi_factor_spmv_updates_per_application": 0,
            }
            assert attribution[
                "tensor_ilu_triangular_solve_policy"
            ] == ({
                "kind": "iterative_jacobi",
                "lower_jacobi_iterations": 5,
                "upper_jacobi_iterations": 5,
                "jacobi_factor_spmv_updates_per_application": 8,
            } if mode == "igr" else {
                "kind": "not_applicable",
                "lower_jacobi_iterations": 0,
                "upper_jacobi_iterations": 0,
                "jacobi_factor_spmv_updates_per_application": 0,
            })
            assert attribution["includes_tensor_condensation"] is True
            assert attribution[
                "mma_contribution_requires_separate_profiler_attribution"
            ] is True
            assert result["engines"]["ordinary"]["solver_work"][
                "trace_preconditioners"
            ] == ["hypre_ilu0"]
            assert result["engines"]["tensor"]["solver_work"][
                "trace_preconditioners"
            ] == [
                "hypre_ilu0" if cell_key.startswith("igr_")
                else "hypre_diag_scale"
            ]
            for engine in ("ordinary", "tensor"):
                work = result["engines"][engine]["solver_work"]
                assert work["trace_policies"] == ["engine-default"]
                expected_trace_solver = (
                    "native-rank1-fp64"
                    if engine == "tensor" else "hypre-parcsr-gmres"
                )
                assert work["trace_solvers"] == [expected_trace_solver]
                assert work["tensor_core"]["trace_solvers"] == [
                    expected_trace_solver
                ]
                assert work["tensor_core"][
                    "native_rank1_trace_solve_calls"
                ]["total"] == (20 if engine == "tensor" else 0)
                assert work["tensor_core"][
                    "hypre_trace_solve_calls"
                ]["total"] == (0 if engine == "tensor" else 20)
                assert work["tensor_core"][
                    "native_rank1_trace_solver_configured_values"
                ] == [engine == "tensor"]
                assert work["tensor_core"][
                    "hypre_host_staged_rank2_trace_solver_configured_values"
                ] == [False]
                assert work["tensor_core"][
                    "trace_solver_accounting_all_complete"
                ] is True
                ilu_active = engine == "ordinary" or cell_key.startswith("igr_")
                assert work["linear_iterations"]["total"] == 70
                assert work["linear_iterations_by_invocation"] == [
                    [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]
                ]
                assert work["tensor_core"]["all_no_fallback"] is True
                assert work["tensor_core"][
                    "ilu_accounting_kinds"
                ] == ["derived_bounds_v1" if ilu_active else "unavailable"]
                assert work["tensor_core"][
                    "ilu_accounting_all_complete"
                ] is True
                tensor_igr = engine == "tensor" and mode == "igr"
                expected_triangular_policy = (
                    ["iterative_jacobi", 5, 5] if tensor_igr
                    else ["direct", 0, 0] if ilu_active
                    else ["not_applicable", 0, 0]
                )
                assert work["tensor_core"][
                    "ilu_triangular_solve_policies"
                ] == [expected_triangular_policy]
                expected_factor_setups = (
                    10 if tensor_igr else 20 if ilu_active else 0
                )
                expected_factor_reuses = 10 if tensor_igr else 0
                expected_factor_age = 1 if tensor_igr else 0
                expected_factor_policy = (
                    "first_trace_solve_per_attempt" if tensor_igr
                    else "refresh_each_trace_solve" if ilu_active
                    else "not_applicable"
                )
                assert work["tensor_core"][
                    "ilu_factor_reuse_policies"
                ] == [expected_factor_policy]
                expected_callback_lower = 80 if ilu_active else 0
                expected_callback_upper = 140 if ilu_active else 0
                assert work["tensor_core"][
                    "ilu_factor_setups"
                ]["total"] == expected_factor_setups
                assert work["tensor_core"][
                    "ilu_factor_reuses"
                ]["total"] == expected_factor_reuses
                assert work["tensor_core"][
                    "ilu_factor_age_generations"
                ]["maximum"] == expected_factor_age
                assert "total" not in work["tensor_core"][
                    "ilu_factor_age_generations"
                ]
                assert work["tensor_core"][
                    "ilu_preconditioner_callbacks_lower_bound"
                ]["total"] == expected_callback_lower
                assert work["tensor_core"][
                    "ilu_preconditioner_callbacks_upper_bound"
                ]["total"] == expected_callback_upper
                assert work["tensor_core"][
                    "ilu_inner_iterations_lower_bound"
                ]["total"] == expected_callback_lower
                assert work["tensor_core"][
                    "ilu_inner_iterations_upper_bound"
                ]["total"] == expected_callback_upper
                assert work["tensor_core"][
                    "ilu_jacobi_factor_spmv_updates_lower_bound"
                ]["total"] == (640 if tensor_igr else 0)
                assert work["tensor_core"][
                    "ilu_jacobi_factor_spmv_updates_upper_bound"
                ]["total"] == (1120 if tensor_igr else 0)
                assert all(work["acceptance_gates"].values())
                assert work["transfers"]["all_complete"] is True
                memory = result["engines"][engine][
                    "initialization_and_memory"
                ]
                assert memory[
                    "maximum_observed_snapshot_device_used_bytes"
                ] == 350.0
                assert memory["umpire_sum_high_watermark_bytes"]["median"] == 150.0
            assert result["engines"]["ordinary"]["solver_work"][
                "tensor_core"
            ]["mma_tiles"]["total"] == 0
            assert result["engines"]["tensor"]["solver_work"][
                "tensor_core"
            ]["mma_tiles"]["total"] == 640
        primary = [item for item in campaign["invocations"]
                   if item["phase"] == "final_tree_primary" and
                   item["mode"] == "conventional" and item["order"] == 1]
        assert [item["comparison_member"] for item in primary] == [
            "ordinary", "tensor", "tensor", "ordinary"
        ]
        for item in campaign["invocations"]:
            assert "--order" in item["command"]
            assert "--mesh-n" in item["command"]
            assert "--domain-periods" in item["command"]
            assert "--dt" in item["command"]
            if item["phase"] != "ordinary_frozen_comparison":
                assert item["trace_policy"] == "engine-default"
                assert item["command"][
                    item["command"].index("--trace-policy") + 1
                ] == "engine-default"
            for path_field, identity_field in (
                ("raw_json", "raw_json_identity"),
                ("stderr", "stderr_identity"),
            ):
                identity = item[identity_field]
                artifact = output / item[path_field]
                assert identity["path"] == item[path_field]
                assert identity["size_bytes"] == artifact.stat().st_size
                assert identity["sha256"] == hashlib.sha256(
                    artifact.read_bytes()
                ).hexdigest()
            summary = item["benchmark_summary"]
            assert summary["trace_policy"] == "engine-default"
            expected_trace_solver = (
                "native-rank1-fp64"
                if item["comparison_member"] == "tensor"
                else "hypre-parcsr-gmres"
            )
            assert summary["trace_solver"] == expected_trace_solver
            expected_preconditioner = CAMPAIGN.expected_trace_preconditioner(
                item["comparison_member"], item["mode"]
            )
            assert summary["trace_preconditioner"] == expected_preconditioner
            assert summary["measured_linear_iterations_total"] == 35
            assert [sample["linear_iterations"]
                    for sample in summary["measured_sample_work"]] == [7] * 5
            assert all(
                sample["trace_preconditioner"] == expected_preconditioner
                for sample in summary["measured_sample_work"]
            )
            assert all(
                sample["trace_preconditioner_configured"] is True
                for sample in summary["measured_sample_work"]
            )
            assert all(
                sample["trace_solver"] == expected_trace_solver and
                sample["native_rank1_trace_solve_calls"] ==
                (sample["trace_solves"]
                 if item["comparison_member"] == "tensor" else 0) and
                sample["hypre_trace_solve_calls"] ==
                (0 if item["comparison_member"] == "tensor"
                 else sample["trace_solves"]) and
                sample["native_rank1_trace_solver_configured"] is
                (item["comparison_member"] == "tensor") and
                sample[
                    "hypre_host_staged_rank2_trace_solver_configured"
                ] is False and
                sample["trace_solver_accounting_complete"] is True
                for sample in summary["measured_sample_work"]
            )
            ilu_active = item["comparison_member"] == "ordinary" or \
                item["mode"] == "igr"
            expected_accounting_kind = (
                "derived_bounds_v1" if ilu_active else "unavailable"
            )
            expected_ilu_policy = CAMPAIGN.expected_trace_ilu_policy(
                item["comparison_member"], item["mode"]
            )
            expected_factor_reuse_policy = \
                CAMPAIGN.expected_trace_ilu_factor_reuse_policy(
                    item["comparison_member"], item["mode"]
                )
            expected_triangular_policy = \
                CAMPAIGN.expected_trace_ilu_triangular_solve_policy(
                    item["comparison_member"], item["mode"]
                )
            assert (
                summary["trace_ilu_policy"]["type"],
                summary["trace_ilu_policy"]["level_of_fill"],
                summary["trace_ilu_policy"]["local_reordering"],
                summary["trace_ilu_policy"]["application_iterations"],
            ) == expected_ilu_policy
            assert summary["trace_ilu_policy"][
                "accounting_kind"
            ] == expected_accounting_kind
            assert summary["trace_ilu_policy"][
                "factor_reuse_policy"
            ] == expected_factor_reuse_policy
            assert (
                summary["trace_ilu_policy"]["triangular_solve_kind"],
                summary["trace_ilu_policy"]["lower_jacobi_iterations"],
                summary["trace_ilu_policy"]["upper_jacobi_iterations"],
            ) == expected_triangular_policy[:3]
            expected_factor_setups = (
                1 if expected_factor_reuse_policy ==
                "first_trace_solve_per_attempt"
                else 2 if ilu_active else 0
            )
            expected_factor_reuses = (
                1 if expected_factor_reuse_policy ==
                "first_trace_solve_per_attempt" else 0
            )
            expected_factor_age = expected_factor_reuses
            assert all(
                sample["ilu_accounting_kind"] == expected_accounting_kind and
                sample["ilu_accounting_complete"] is True and
                sample["ilu_factor_reuse_policy"] ==
                expected_factor_reuse_policy and
                sample["ilu_factor_setups"] == expected_factor_setups and
                sample["ilu_factor_reuses"] == expected_factor_reuses and
                sample["ilu_factor_age_generations"] == expected_factor_age and
                sample["ilu_triangular_solve_kind"] ==
                expected_triangular_policy[0] and
                sample["ilu_lower_jacobi_iterations"] ==
                expected_triangular_policy[1] and
                sample["ilu_upper_jacobi_iterations"] ==
                expected_triangular_policy[2] and
                sample["ilu_preconditioner_callbacks_lower_bound"] ==
                (8 if ilu_active else 0) and
                sample["ilu_preconditioner_callbacks_upper_bound"] ==
                (14 if ilu_active else 0) and
                sample["ilu_inner_iterations_lower_bound"] ==
                (8 if ilu_active else 0) and
                sample["ilu_inner_iterations_upper_bound"] ==
                (14 if ilu_active else 0) and
                sample["ilu_jacobi_factor_spmv_updates_lower_bound"] ==
                8 * expected_ilu_policy[3] * expected_triangular_policy[3] and
                sample["ilu_jacobi_factor_spmv_updates_upper_bound"] ==
                14 * expected_ilu_policy[3] * expected_triangular_policy[3]
                for sample in summary["measured_sample_work"]
            )
            assert summary["device_memory_snapshots"][
                "after_samples"
            ]["used_bytes"] == 350
            assert summary[
                "umpire_allocator_high_watermarks_after_samples"
            ]["sum_bytes"] == 150
            assert all(
                sample["fallback_count"] == 0 and
                sample["fallback_accounting_complete"] is True and
                sample["no_fallback"] is True and
                all(sample["gates"].values())
                for sample in summary["measured_sample_work"]
            )
        assert campaign["hardware"]["start"]["values"][
            "clocks.current.sm"
        ] == "2100"

        ablation_output = root / "same-policy-attribution"
        ablation_command = list(command)
        ablation_command[
            ablation_command.index(str(output))
        ] = str(ablation_output)
        ablation_command[
            ablation_command.index("--pairs") + 1
        ] = "3"
        ablation_command.extend((
            "--comparison-mode", "same-policy-attribution",
        ))
        previous_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        try:
            ablation_completed = run(ablation_command, root)
        finally:
            if previous_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible_devices
        assert json.loads(ablation_completed.stdout)[
            "execution_completed"
        ] is True
        ablation = json.loads(
            (ablation_output / "campaign.json").read_text(encoding="utf-8")
        )
        assert ablation["comparison_mode"] == "same-policy-attribution"
        assert ablation["attribution_status"] == \
            "provisional_non_promotion_attribution"
        assert ablation["promotion_qualification"] is False
        assert "final_tree_results" not in ablation
        assert "frozen_control_results" not in ablation
        assert len(ablation["same_policy_attribution_results"]) == 6
        assert len(ablation["invocations"]) == 54
        assert len(list((ablation_output / "raw").rglob("*.json"))) == 54
        assert ablation["export_control"] == {
            "status": "not_run",
            "reason": "bounded_same_policy_attribution_primary_only",
        }
        assert ablation["performance_disposition"][
            "promotion_performance_pass"
        ] is False
        assert ablation["performance_disposition"][
            "promotion_performance_disposition"
        ] == "provisional_same_policy_attribution_not_a_promotion_gate"
        assert ablation["performance_disposition"][
            "promotion_eligible"
        ] is False
        ablation_primary = [
            item for item in ablation["invocations"]
            if item["phase"] == "same_policy_attribution" and
            item["mode"] == "conventional" and item["order"] == 1
        ]
        assert [item["comparison_member"] for item in ablation_primary] == [
            "O_base", "O_common", "T_common",
            "O_common", "T_common", "O_base",
            "T_common", "O_base", "O_common",
        ]
        for item in ablation["invocations"]:
            selection = CAMPAIGN.SAME_POLICY_MEMBERS[
                item["comparison_member"]
            ]
            assert item["trace_policy"] == selection["trace_policy"]
            assert item["command"][item["command"].index("--engine") + 1] == \
                selection["engine"]
            assert item["command"][
                item["command"].index("--trace-policy") + 1
            ] == selection["trace_policy"]
            summary = item["benchmark_summary"]
            assert summary["trace_policy"] == selection["trace_policy"]
            expected_trace_solver = (
                "native-rank1-fp64"
                if selection["engine"] == "tensor"
                else "hypre-parcsr-gmres"
            )
            assert summary["trace_solver"] == expected_trace_solver
            assert all(
                sample["trace_policy"] == selection["trace_policy"] and
                sample["trace_solver"] == expected_trace_solver
                for sample in summary["measured_sample_work"]
            )
        for key, result in ablation[
                "same_policy_attribution_results"].items():
            assert result["promotion_eligible"] is False
            assert result["attribution"]["common_fast_policy_exactly_matched"] \
                is True
            assert result["members"]["O_common"][
                "trace_policy_configuration"
            ] == result["members"]["T_common"][
                "trace_policy_configuration"
            ]
            assert result["members"]["O_base"]["engine"] == "ordinary"
            assert result["members"]["O_base"]["trace_policy"] == \
                "engine-default"
            comparisons = result["comparisons"]
            assert math.isclose(
                comparisons["policy_gain"]["ratio"]["median"], 4.0 / 3.0
            )
            assert math.isclose(
                comparisons["tensor_local_gain"]["ratio"]["median"], 1.5
            )
            assert math.isclose(
                comparisons["package_gain"]["ratio"]["median"], 2.0
            )
            assert comparisons["policy_gain"][
                "median_ratio_95_percent_confidence_interval"
            ]["resamples"] == 100
            assert len(result["triplets"]) == 3
            assert all(
                set(triplet["raw_durations_seconds"]) ==
                set(CAMPAIGN.SAME_POLICY_MEMBER_ORDER) and
                all(len(values) == 5 for values in
                    triplet["raw_durations_seconds"].values()) and
                triplet["common_fast_policy_exactly_matched"] is True
                for triplet in result["triplets"]
            )
            expected_common = "hypre_ilu0" if key.startswith("igr_") \
                else "hypre_diag_scale"
            assert result["members"]["O_common"][
                "trace_policy_configuration"
            ]["trace_preconditioner"] == expected_common

        ordinary_invocation = next(
            item for item in campaign["invocations"]
            if item["phase"] == "final_tree_primary" and
            item["comparison_member"] == "ordinary"
        )
        ordinary_record = json.loads(
            (output / ordinary_invocation["raw_json"]).read_text()
        )
        assert ordinary_record[
            "trace_ilu_factor_reuse_policy"
        ] == "refresh_each_trace_solve"
        assert ordinary_record["trace_ilu_triangular_solve_kind"] == "direct"
        assert ordinary_record["trace_ilu_lower_jacobi_iterations"] == 0
        assert ordinary_record["trace_ilu_upper_jacobi_iterations"] == 0
        assert all(
            sample["tensor_core"]["ilu_factor_reuse_policy"] ==
            "refresh_each_trace_solve" and
            sample["tensor_core"]["ilu_factor_setups"] == 2 and
            sample["tensor_core"]["ilu_factor_reuses"] == 0 and
            sample["tensor_core"]["ilu_factor_age_generations"] == 0 and
            sample["tensor_core"]["ilu_triangular_solve_kind"] == "direct" and
            sample["tensor_core"]["ilu_lower_jacobi_iterations"] == 0 and
            sample["tensor_core"]["ilu_upper_jacobi_iterations"] == 0 and
            sample["tensor_core"][
                "ilu_jacobi_factor_spmv_updates_lower_bound"
            ] == 0 and
            sample["tensor_core"][
                "ilu_jacobi_factor_spmv_updates_upper_bound"
            ] == 0
            for sample in ordinary_record["samples"]
        )
        validation_arguments = {
            "source": "final",
            "mode": ordinary_invocation["mode"],
            "order": ordinary_invocation["order"],
            "mesh_n": 12,
            "domain_periods": 3,
            "timestep": 1.0e-5,
            "export": False,
            "engine": "ordinary",
            "expected_build": ordinary_record["build_identity"],
            "expected_gpu": {
                "device_ordinal": 0,
                "uuid": "GPU-00000000-0000-0000-0000-000000000001",
                "compute_capability": 120,
            },
        }

        def rejected(record: dict) -> bool:
            try:
                CAMPAIGN.validate_record(record, **validation_arguments)
            except CAMPAIGN.CampaignError:
                return True
            return False

        bad_gate = copy.deepcopy(ordinary_record)
        bad_gate["samples"][1]["gates"]["admissible"] = False
        assert rejected(bad_gate)
        bad_mma = copy.deepcopy(ordinary_record)
        bad_mma["samples"][1]["tensor_core"]["mma_tiles"] = 1
        assert rejected(bad_mma)
        bad_memory = copy.deepcopy(ordinary_record)
        bad_memory["device_memory_snapshots"]["after_samples"][
            "used_bytes"
        ] += 1
        assert rejected(bad_memory)
        bad_scope = copy.deepcopy(ordinary_record)
        bad_scope["timing_scope"] = "Prepare-only"
        assert rejected(bad_scope)
        bad_transfers = copy.deepcopy(ordinary_record)
        bad_transfers["samples"][1]["transfers"]["complete"] = False
        assert rejected(bad_transfers)
        bad_trace_policy = copy.deepcopy(ordinary_record)
        bad_trace_policy["trace_policy"] = "common-fast"
        assert rejected(bad_trace_policy)
        bad_trace_solver = copy.deepcopy(ordinary_record)
        bad_trace_solver["trace_solver"] = "native-rank1-fp64"
        assert rejected(bad_trace_solver)
        bad_native_calls = copy.deepcopy(ordinary_record)
        bad_native_calls["samples"][1]["tensor_core"][
            "native_rank1_trace_solve_calls"
        ] = bad_native_calls["samples"][1]["trace_solves"]
        assert rejected(bad_native_calls)
        for field, value in (
            ("trace_solver", "native-rank1-fp64"),
            ("hypre_trace_solve_calls", 0),
            ("native_rank1_trace_solver_configured", True),
            ("hypre_host_staged_rank2_trace_solver_configured", True),
            ("trace_solver_accounting_complete", False),
        ):
            bad_solver_accounting = copy.deepcopy(ordinary_record)
            bad_solver_accounting["samples"][1]["tensor_core"][field] = value
            assert rejected(bad_solver_accounting)
        bad_variant = copy.deepcopy(ordinary_record)
        bad_variant["variant"] = "ablation"
        assert rejected(bad_variant)
        bad_production_selection = copy.deepcopy(ordinary_record)
        bad_production_selection["production_selectable"] = False
        assert rejected(bad_production_selection)
        bad_sample_trace_policy = copy.deepcopy(ordinary_record)
        bad_sample_trace_policy["samples"][1]["tensor_core"][
            "trace_policy"
        ] = "common-fast"
        assert rejected(bad_sample_trace_policy)
        boolean_ilu_policy = copy.deepcopy(ordinary_record)
        boolean_ilu_policy["trace_ilu_application_iterations"] = True
        assert rejected(boolean_ilu_policy)
        bad_ordinary_reuse_top_level = copy.deepcopy(ordinary_record)
        bad_ordinary_reuse_top_level[
            "trace_ilu_factor_reuse_policy"
        ] = "first_trace_solve_per_attempt"
        assert rejected(bad_ordinary_reuse_top_level)
        bad_ordinary_factor_setups = copy.deepcopy(ordinary_record)
        bad_ordinary_factor_setups["samples"][1]["tensor_core"][
            "ilu_factor_setups"
        ] = 1
        assert rejected(bad_ordinary_factor_setups)
        bad_ordinary_factor_reuses = copy.deepcopy(ordinary_record)
        bad_ordinary_factor_reuses["samples"][1]["tensor_core"][
            "ilu_factor_reuses"
        ] = 1
        assert rejected(bad_ordinary_factor_reuses)
        bad_ordinary_factor_age = copy.deepcopy(ordinary_record)
        bad_ordinary_factor_age["samples"][1]["tensor_core"][
            "ilu_factor_age_generations"
        ] = 1
        assert rejected(bad_ordinary_factor_age)
        bad_ordinary_triangular_top_level = copy.deepcopy(ordinary_record)
        bad_ordinary_triangular_top_level[
            "trace_ilu_triangular_solve_kind"
        ] = "iterative_jacobi"
        assert rejected(bad_ordinary_triangular_top_level)
        bad_ordinary_jacobi_iterations = copy.deepcopy(ordinary_record)
        bad_ordinary_jacobi_iterations["samples"][1]["tensor_core"][
            "ilu_lower_jacobi_iterations"
        ] = 5
        assert rejected(bad_ordinary_jacobi_iterations)
        bad_ordinary_jacobi_updates = copy.deepcopy(ordinary_record)
        bad_ordinary_jacobi_updates["samples"][1]["tensor_core"][
            "ilu_jacobi_factor_spmv_updates_lower_bound"
        ] = 1
        assert rejected(bad_ordinary_jacobi_updates)

        bad_build = copy.deepcopy(ordinary_record)
        bad_build["build_identity"]["source_revision"] = "0" * 40
        assert rejected(bad_build)
        bad_uuid = copy.deepcopy(ordinary_record)
        bad_uuid["gpu_uuid"] = (
            "GPU-00000000-0000-0000-0000-000000000002"
        )
        assert rejected(bad_uuid)

        ordinary_common_invocation = next(
            item for item in ablation["invocations"]
            if item["comparison_member"] == "O_common" and
            item["mode"] == "igr"
        )
        ordinary_common_record = json.loads(
            (ablation_output / ordinary_common_invocation["raw_json"])
            .read_text(encoding="utf-8")
        )
        common_validation_arguments = {
            **validation_arguments,
            "mode": "igr",
            "order": ordinary_common_invocation["order"],
            "trace_policy": "common-fast",
            "expected_build": ordinary_common_record["build_identity"],
        }
        CAMPAIGN.validate_record(
            ordinary_common_record, **common_validation_arguments
        )
        assert ordinary_common_record["variant"] == "ablation"
        assert ordinary_common_record["production_selectable"] is False
        assert ordinary_common_record["trace_ilu_triangular_solve_kind"] == \
            "iterative_jacobi"
        assert ordinary_common_record[
            "trace_ilu_factor_reuse_policy"
        ] == "first_trace_solve_per_attempt"
        bad_common_top_level = copy.deepcopy(ordinary_common_record)
        bad_common_top_level[
            "trace_ilu_triangular_solve_kind"
        ] = "direct"
        try:
            CAMPAIGN.validate_record(
                bad_common_top_level, **common_validation_arguments
            )
        except CAMPAIGN.CampaignError:
            pass
        else:
            raise AssertionError(
                "common-fast ordinary IGR accepted mismatched trace metadata"
            )

        tensor_igr_invocation = next(
            item for item in campaign["invocations"]
            if item["phase"] == "final_tree_primary" and
            item["comparison_member"] == "tensor" and
            item["mode"] == "igr"
        )
        tensor_igr_record = json.loads(
            (output / tensor_igr_invocation["raw_json"]).read_text()
        )
        tensor_igr_validation_arguments = {
            **validation_arguments,
            "mode": "igr",
            "order": tensor_igr_invocation["order"],
            "engine": "tensor",
            "expected_build": tensor_igr_record["build_identity"],
        }
        assert tensor_igr_record[
            "trace_preconditioner"
        ] == "hypre_ilu0"
        assert tensor_igr_record["trace_ilu_type"] == 0
        assert tensor_igr_record["trace_ilu_level_of_fill"] == 0
        assert tensor_igr_record["trace_ilu_application_iterations"] == 1
        assert tensor_igr_record["trace_ilu_local_reordering"] == 1
        assert tensor_igr_record[
            "trace_ilu_accounting_kind"
        ] == "derived_bounds_v1"
        assert tensor_igr_record[
            "trace_ilu_factor_reuse_policy"
        ] == "first_trace_solve_per_attempt"
        assert tensor_igr_record[
            "trace_ilu_triangular_solve_kind"
        ] == "iterative_jacobi"
        assert tensor_igr_record["trace_ilu_lower_jacobi_iterations"] == 5
        assert tensor_igr_record["trace_ilu_upper_jacobi_iterations"] == 5
        assert all(
            sample["tensor_core"]["trace_preconditioner"] ==
            "hypre_ilu0" and
            sample["tensor_core"]["ilu_type"] == 0 and
            sample["tensor_core"]["ilu_level_of_fill"] == 0 and
            sample["tensor_core"]["ilu_application_iterations"] == 1 and
            sample["tensor_core"]["ilu_local_reordering"] == 1 and
            sample["tensor_core"]["ilu_accounting_kind"] ==
            "derived_bounds_v1" and
            sample["tensor_core"]["ilu_factor_reuse_policy"] ==
            "first_trace_solve_per_attempt" and
            sample["tensor_core"]["ilu_factor_setups"] == 1 and
            sample["tensor_core"]["ilu_factor_reuses"] == 1 and
            sample["tensor_core"]["ilu_factor_age_generations"] == 1 and
            sample["tensor_core"]["ilu_triangular_solve_kind"] ==
            "iterative_jacobi" and
            sample["tensor_core"]["ilu_lower_jacobi_iterations"] == 5 and
            sample["tensor_core"]["ilu_upper_jacobi_iterations"] == 5 and
            sample["tensor_core"]["ilu_accounting_complete"] is True and
            sample["tensor_core"][
                "ilu_preconditioner_callbacks_lower_bound"
            ] == 8 and
            sample["tensor_core"][
                "ilu_preconditioner_callbacks_upper_bound"
            ] == 14 and
            sample["tensor_core"]["ilu_inner_iterations_lower_bound"] == 8 and
            sample["tensor_core"]["ilu_inner_iterations_upper_bound"] == 14 and
            sample["tensor_core"][
                "ilu_jacobi_factor_spmv_updates_lower_bound"
            ] == 64 and
            sample["tensor_core"][
                "ilu_jacobi_factor_spmv_updates_upper_bound"
            ] == 112
            for sample in tensor_igr_record["samples"]
        )

        def tensor_igr_rejected(record: dict) -> bool:
            try:
                CAMPAIGN.validate_record(
                    record, **tensor_igr_validation_arguments
                )
            except CAMPAIGN.CampaignError:
                return True
            return False

        wrong_tensor_igr_top_level = copy.deepcopy(tensor_igr_record)
        wrong_tensor_igr_top_level[
            "trace_preconditioner"
        ] = "hypre_diag_scale"
        assert tensor_igr_rejected(wrong_tensor_igr_top_level)
        wrong_tensor_igr_sample = copy.deepcopy(tensor_igr_record)
        wrong_tensor_igr_sample["samples"][1]["tensor_core"][
            "trace_preconditioner"
        ] = "hypre_diag_scale"
        assert tensor_igr_rejected(wrong_tensor_igr_sample)
        wrong_tensor_igr_top_level_iterations = copy.deepcopy(
            tensor_igr_record
        )
        wrong_tensor_igr_top_level_iterations[
            "trace_ilu_application_iterations"
        ] = 2
        assert tensor_igr_rejected(wrong_tensor_igr_top_level_iterations)
        wrong_tensor_igr_top_level_kind = copy.deepcopy(tensor_igr_record)
        wrong_tensor_igr_top_level_kind[
            "trace_ilu_accounting_kind"
        ] = "unavailable"
        assert tensor_igr_rejected(wrong_tensor_igr_top_level_kind)
        wrong_tensor_igr_top_level_reuse = copy.deepcopy(tensor_igr_record)
        wrong_tensor_igr_top_level_reuse[
            "trace_ilu_factor_reuse_policy"
        ] = "refresh_each_trace_solve"
        assert tensor_igr_rejected(wrong_tensor_igr_top_level_reuse)
        wrong_tensor_igr_top_level_triangular = copy.deepcopy(
            tensor_igr_record
        )
        wrong_tensor_igr_top_level_triangular[
            "trace_ilu_triangular_solve_kind"
        ] = "direct"
        assert tensor_igr_rejected(wrong_tensor_igr_top_level_triangular)
        wrong_tensor_igr_top_level_lower_jacobi = copy.deepcopy(
            tensor_igr_record
        )
        wrong_tensor_igr_top_level_lower_jacobi[
            "trace_ilu_lower_jacobi_iterations"
        ] = 4
        assert tensor_igr_rejected(wrong_tensor_igr_top_level_lower_jacobi)
        wrong_tensor_igr_sample_iterations = copy.deepcopy(tensor_igr_record)
        wrong_tensor_igr_sample_iterations["samples"][1]["tensor_core"][
            "ilu_application_iterations"
        ] = 2
        assert tensor_igr_rejected(wrong_tensor_igr_sample_iterations)
        wrong_tensor_igr_top_level_fill = copy.deepcopy(tensor_igr_record)
        wrong_tensor_igr_top_level_fill[
            "trace_ilu_level_of_fill"
        ] = 1
        assert tensor_igr_rejected(wrong_tensor_igr_top_level_fill)
        wrong_tensor_igr_sample_fill = copy.deepcopy(tensor_igr_record)
        wrong_tensor_igr_sample_fill["samples"][1]["tensor_core"][
            "ilu_level_of_fill"
        ] = 1
        assert tensor_igr_rejected(wrong_tensor_igr_sample_fill)
        wrong_tensor_igr_top_level_type = copy.deepcopy(tensor_igr_record)
        wrong_tensor_igr_top_level_type["trace_ilu_type"] = 1
        assert tensor_igr_rejected(wrong_tensor_igr_top_level_type)
        wrong_tensor_igr_sample_reordering = copy.deepcopy(tensor_igr_record)
        wrong_tensor_igr_sample_reordering["samples"][1]["tensor_core"][
            "ilu_local_reordering"
        ] = 0
        assert tensor_igr_rejected(wrong_tensor_igr_sample_reordering)
        wrong_tensor_igr_kind = copy.deepcopy(tensor_igr_record)
        wrong_tensor_igr_kind["samples"][1]["tensor_core"][
            "ilu_accounting_kind"
        ] = "unavailable"
        assert tensor_igr_rejected(wrong_tensor_igr_kind)
        incomplete_tensor_igr = copy.deepcopy(tensor_igr_record)
        incomplete_tensor_igr["samples"][1]["tensor_core"][
            "ilu_accounting_complete"
        ] = False
        assert tensor_igr_rejected(incomplete_tensor_igr)
        wrong_tensor_igr_factor_setups = copy.deepcopy(tensor_igr_record)
        wrong_tensor_igr_factor_setups["samples"][1]["tensor_core"][
            "ilu_factor_setups"
        ] += 1
        assert tensor_igr_rejected(wrong_tensor_igr_factor_setups)
        wrong_tensor_igr_reuse_policy = copy.deepcopy(tensor_igr_record)
        wrong_tensor_igr_reuse_policy["samples"][1]["tensor_core"][
            "ilu_factor_reuse_policy"
        ] = "refresh_each_trace_solve"
        assert tensor_igr_rejected(wrong_tensor_igr_reuse_policy)
        wrong_tensor_igr_reuses = copy.deepcopy(tensor_igr_record)
        wrong_tensor_igr_reuses["samples"][1]["tensor_core"][
            "ilu_factor_reuses"
        ] = 0
        assert tensor_igr_rejected(wrong_tensor_igr_reuses)
        zero_tensor_igr_factor_age = copy.deepcopy(tensor_igr_record)
        zero_tensor_igr_factor_age["samples"][1]["tensor_core"][
            "ilu_factor_age_generations"
        ] = 0
        assert tensor_igr_rejected(zero_tensor_igr_factor_age)
        low_tensor_igr_callback_bound = copy.deepcopy(tensor_igr_record)
        low_tensor_igr_sample = low_tensor_igr_callback_bound[
            "samples"
        ][1]["tensor_core"]
        low_tensor_igr_sample[
            "ilu_preconditioner_callbacks_lower_bound"
        ] = 6
        low_tensor_igr_sample["ilu_inner_iterations_lower_bound"] = 6
        assert tensor_igr_rejected(low_tensor_igr_callback_bound)
        wrong_exact_tensor_igr_lower = copy.deepcopy(tensor_igr_record)
        wrong_exact_tensor_igr_sample = wrong_exact_tensor_igr_lower[
            "samples"
        ][1]["tensor_core"]
        wrong_exact_tensor_igr_sample[
            "ilu_preconditioner_callbacks_lower_bound"
        ] = 10
        wrong_exact_tensor_igr_sample["ilu_inner_iterations_lower_bound"] = 10
        wrong_exact_tensor_igr_sample[
            "ilu_jacobi_factor_spmv_updates_lower_bound"
        ] = 80
        assert tensor_igr_rejected(wrong_exact_tensor_igr_lower)
        nonexact_tensor_igr_callback_bound = copy.deepcopy(tensor_igr_record)
        nonexact_tensor_igr_sample = nonexact_tensor_igr_callback_bound[
            "samples"
        ][1]["tensor_core"]
        nonexact_tensor_igr_sample[
            "ilu_preconditioner_callbacks_upper_bound"
        ] = 13
        nonexact_tensor_igr_sample["ilu_inner_iterations_upper_bound"] = 13
        assert tensor_igr_rejected(nonexact_tensor_igr_callback_bound)
        overflowing_tensor_igr_bound = copy.deepcopy(tensor_igr_record)
        overflowing_tensor_igr_bound["samples"][1]["tensor_core"][
            "ilu_preconditioner_callbacks_upper_bound"
        ] = CAMPAIGN.UINT64_MAX + 1
        assert tensor_igr_rejected(overflowing_tensor_igr_bound)
        wrong_tensor_igr_inner_bound = copy.deepcopy(tensor_igr_record)
        wrong_tensor_igr_inner_bound["samples"][1]["tensor_core"][
            "ilu_inner_iterations_lower_bound"
        ] += 1
        assert tensor_igr_rejected(wrong_tensor_igr_inner_bound)
        wrong_tensor_igr_triangular_kind = copy.deepcopy(tensor_igr_record)
        wrong_tensor_igr_triangular_kind["samples"][1]["tensor_core"][
            "ilu_triangular_solve_kind"
        ] = "direct"
        assert tensor_igr_rejected(wrong_tensor_igr_triangular_kind)
        wrong_tensor_igr_jacobi_iterations = copy.deepcopy(tensor_igr_record)
        wrong_tensor_igr_jacobi_iterations["samples"][1]["tensor_core"][
            "ilu_upper_jacobi_iterations"
        ] = 4
        assert tensor_igr_rejected(wrong_tensor_igr_jacobi_iterations)
        wrong_tensor_igr_jacobi_update_bound = copy.deepcopy(
            tensor_igr_record
        )
        wrong_tensor_igr_jacobi_update_bound["samples"][1]["tensor_core"][
            "ilu_jacobi_factor_spmv_updates_lower_bound"
        ] -= 1
        assert tensor_igr_rejected(wrong_tensor_igr_jacobi_update_bound)

        tensor_conventional_invocation = next(
            item for item in campaign["invocations"]
            if item["phase"] == "final_tree_primary" and
            item["comparison_member"] == "tensor" and
            item["mode"] == "conventional"
        )
        tensor_conventional_record = json.loads(
            (output / tensor_conventional_invocation["raw_json"]).read_text()
        )
        tensor_conventional_validation_arguments = {
            **validation_arguments,
            "mode": "conventional",
            "order": tensor_conventional_invocation["order"],
            "engine": "tensor",
            "expected_build": tensor_conventional_record["build_identity"],
        }
        assert tensor_conventional_record[
            "trace_ilu_factor_reuse_policy"
        ] == "not_applicable"
        assert tensor_conventional_record[
            "trace_ilu_triangular_solve_kind"
        ] == "not_applicable"
        assert tensor_conventional_record[
            "trace_ilu_lower_jacobi_iterations"
        ] == 0
        assert tensor_conventional_record[
            "trace_ilu_upper_jacobi_iterations"
        ] == 0
        assert all(
            sample["tensor_core"]["ilu_factor_reuse_policy"] ==
            "not_applicable" and
            sample["tensor_core"]["ilu_factor_setups"] == 0 and
            sample["tensor_core"]["ilu_factor_reuses"] == 0 and
            sample["tensor_core"]["ilu_factor_age_generations"] == 0 and
            sample["tensor_core"]["ilu_triangular_solve_kind"] ==
            "not_applicable" and
            sample["tensor_core"]["ilu_lower_jacobi_iterations"] == 0 and
            sample["tensor_core"]["ilu_upper_jacobi_iterations"] == 0 and
            sample["tensor_core"][
                "ilu_jacobi_factor_spmv_updates_lower_bound"
            ] == 0 and
            sample["tensor_core"][
                "ilu_jacobi_factor_spmv_updates_upper_bound"
            ] == 0
            for sample in tensor_conventional_record["samples"]
        )

        def tensor_conventional_rejected(record: dict) -> bool:
            try:
                CAMPAIGN.validate_record(
                    record, **tensor_conventional_validation_arguments
                )
            except CAMPAIGN.CampaignError:
                return True
            return False

        bad_tensor_conventional_top_level = copy.deepcopy(
            tensor_conventional_record
        )
        bad_tensor_conventional_top_level[
            "trace_ilu_factor_reuse_policy"
        ] = "refresh_each_trace_solve"
        assert tensor_conventional_rejected(
            bad_tensor_conventional_top_level
        )
        bad_tensor_conventional_reuse = copy.deepcopy(
            tensor_conventional_record
        )
        bad_tensor_conventional_reuse["samples"][1]["tensor_core"][
            "ilu_factor_reuses"
        ] = 1
        assert tensor_conventional_rejected(bad_tensor_conventional_reuse)
        bad_tensor_conventional_triangular = copy.deepcopy(
            tensor_conventional_record
        )
        bad_tensor_conventional_triangular["samples"][1]["tensor_core"][
            "ilu_triangular_solve_kind"
        ] = "direct"
        assert tensor_conventional_rejected(
            bad_tensor_conventional_triangular
        )

        frozen_cache, frozen_mfem = write_build_provenance(
            root, frozen_source, "frozen-build"
        )
        frozen = frozen_cache.parent / CAMPAIGN.BENCHMARK_BUILD_RELATIVE_PATH
        frozen.parent.mkdir(parents=True)
        write_executable(
            frozen, FAKE_BENCHMARK + "\n# distinct frozen schema-1 control\n"
        )

        def qualification_arguments(
                qualification_output: Path, *,
                workload_class: str = "break-even",
                profile_overrides: dict[str, object] | None = None,
                frozen_executable: Path = frozen,
                frozen_cache_path: Path = frozen_cache,
                frozen_mfem_path: Path = frozen_mfem):
            profile = dict(CAMPAIGN.QUALIFICATION_PROFILES[workload_class])
            if profile_overrides:
                profile.update(profile_overrides)
            return CAMPAIGN.parser().parse_args([
                "--benchmark", str(final),
                "--output-dir", str(qualification_output),
                "--source-root", str(source),
                "--evidence-mode", "qualification",
                "--pairs", str(profile["pairs"]),
                "--workload-class", workload_class,
                "--workload-label", str(profile["workload_label"]),
                "--mesh-n", str(profile["mesh_n"]),
                "--domain-periods", str(profile["domain_periods"]),
                "--dt", str(profile["dt"]),
                "--export-control-mode", str(profile["export_control_mode"]),
                "--export-control-order", str(profile["export_control_order"]),
                "--build-label", "synthetic-final",
                "--build-metadata", "cuda_arch=120",
                "--build-manifest", str(build_cache),
                "--mfem-identity", str(mfem_identity),
                "--expected-commit", commit,
                "--expected-tree", tree,
                "--gpu-index", "0",
                "--nvidia-smi", str(nvidia_smi),
                "--require-nvidia-smi",
                "--hardware-metadata", "lease=synthetic",
                "--frozen-benchmark", str(frozen_executable),
                "--frozen-source-root", str(frozen_source),
                "--frozen-build-label", "synthetic-frozen",
                "--frozen-build-metadata", "cuda_arch=120",
                "--frozen-build-manifest", str(frozen_cache_path),
                "--frozen-mfem-identity", str(frozen_mfem_path),
                "--bootstrap-resamples", "10000",
                "--timeout-seconds", "30",
            ])

        def expect_campaign_error(action) -> None:
            try:
                action()
            except CAMPAIGN.CampaignError:
                return
            raise AssertionError("campaign evidence binding unexpectedly succeeded")

        ablation_arguments = CAMPAIGN.parser().parse_args(
            ablation_command[2:]
        )
        invalid_ablation_pairs = copy.copy(ablation_arguments)
        invalid_ablation_pairs.pairs = 2
        invalid_ablation_pairs.output_dir = str(root / "bad-ablation-pairs")
        expect_campaign_error(
            lambda: CAMPAIGN.run_campaign(invalid_ablation_pairs)
        )
        invalid_ablation_evidence = copy.copy(ablation_arguments)
        invalid_ablation_evidence.evidence_mode = "qualification"
        invalid_ablation_evidence.output_dir = str(
            root / "bad-ablation-qualification"
        )
        expect_campaign_error(
            lambda: CAMPAIGN.run_campaign(invalid_ablation_evidence)
        )
        invalid_ablation_dirty = copy.copy(ablation_arguments)
        invalid_ablation_dirty.allow_dirty = True
        invalid_ablation_dirty.output_dir = str(root / "bad-ablation-dirty")
        expect_campaign_error(
            lambda: CAMPAIGN.run_campaign(invalid_ablation_dirty)
        )
        invalid_ablation_frozen = copy.copy(ablation_arguments)
        invalid_ablation_frozen.output_dir = str(root / "bad-ablation-frozen")
        invalid_ablation_frozen.frozen_benchmark = str(frozen)
        invalid_ablation_frozen.frozen_source_root = str(frozen_source)
        invalid_ablation_frozen.frozen_build_label = "synthetic-frozen"
        invalid_ablation_frozen.frozen_build_manifest = str(frozen_cache)
        invalid_ablation_frozen.frozen_mfem_identity = str(frozen_mfem)
        expect_campaign_error(
            lambda: CAMPAIGN.run_campaign(invalid_ablation_frozen)
        )

        for workload_class in ("saturated", "break-even"):
            CAMPAIGN.validate_qualification_profile(
                qualification_arguments(
                    root / f"unused-{workload_class}",
                    workload_class=workload_class,
                )
            )
        invalid_profile_values = {
            "workload_label": "wrong-label",
            "mesh_n": 5,
            "domain_periods": 2,
            "dt": 2.0e-5,
            "export_control_mode": "conventional",
            "export_control_order": 2,
            "pairs": 5,
        }
        canonical_arguments = qualification_arguments(root / "unused-canonical")
        for field, value in invalid_profile_values.items():
            invalid_arguments = copy.copy(canonical_arguments)
            setattr(invalid_arguments, field, value)
            expect_campaign_error(
                lambda candidate=invalid_arguments:
                CAMPAIGN.validate_qualification_profile(candidate)
            )

        final_provenance = CAMPAIGN.validated_build_provenance(
            source_root=source,
            cmake_cache_path=str(build_cache),
            mfem_identity_path=str(mfem_identity),
        )
        misbound_executable = root / "not-the-build-target" / "benchmark"
        misbound_executable.parent.mkdir()
        write_executable(misbound_executable, FAKE_BENCHMARK)
        expect_campaign_error(
            lambda: CAMPAIGN.validate_executable_build_relationship(
                CAMPAIGN.file_identity(misbound_executable), final_provenance
            )
        )
        expect_campaign_error(
            lambda: CAMPAIGN.validated_build_provenance(
                source_root=source,
                cmake_cache_path=str(frozen_cache),
                mfem_identity_path=str(frozen_mfem),
            )
        )

        duplicate_cache, duplicate_mfem = write_build_provenance(
            root, frozen_source, "same-hash-frozen-build"
        )
        duplicate_frozen = (
            duplicate_cache.parent / CAMPAIGN.BENCHMARK_BUILD_RELATIVE_PATH
        )
        duplicate_frozen.parent.mkdir(parents=True)
        write_executable(duplicate_frozen, FAKE_BENCHMARK)

        previous_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        CAMPAIGN.FROZEN_CONTROL_COMMIT = synthetic_frozen_commit
        CAMPAIGN.FROZEN_CONTROL_TREE = synthetic_frozen_tree
        CAMPAIGN.FROZEN_OVERLAY_DIFF_SHA256 = hashlib.sha256(
            synthetic_overlay
        ).hexdigest()
        try:
            wrong_workload = qualification_arguments(root / "bad-workload")
            wrong_workload.workload_label = "sm120-mesh5-break-even"
            expect_campaign_error(
                lambda: CAMPAIGN.run_campaign(wrong_workload)
            )

            CAMPAIGN.FROZEN_OVERLAY_DIFF_SHA256 = "0" * 64
            expect_campaign_error(
                lambda: CAMPAIGN.frozen_overlay_identity(frozen_source)
            )
            CAMPAIGN.FROZEN_OVERLAY_DIFF_SHA256 = hashlib.sha256(
                synthetic_overlay
            ).hexdigest()

            same_hash_arguments = qualification_arguments(
                root / "bad-same-hash",
                frozen_executable=duplicate_frozen,
                frozen_cache_path=duplicate_cache,
                frozen_mfem_path=duplicate_mfem,
            )
            expect_campaign_error(
                lambda: CAMPAIGN.run_campaign(same_hash_arguments)
            )

            qualification_output = root / "qualification-evidence"
            manifest_path = CAMPAIGN.run_campaign(
                qualification_arguments(
                    qualification_output, workload_class="saturated"
                )
            )
        finally:
            CAMPAIGN.FROZEN_CONTROL_COMMIT = saved_commit
            CAMPAIGN.FROZEN_CONTROL_TREE = saved_tree
            CAMPAIGN.FROZEN_OVERLAY_DIFF_SHA256 = saved_overlay
            if previous_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible_devices

        assert manifest_path == qualification_output / "campaign.json"
        qualification = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_cells = {
            f"{mode}_p{order}" for mode, order in CAMPAIGN.CELLS
        }
        assert qualification["status"] == "completed"
        assert qualification["execution_status"] == "completed"
        assert qualification["promotion_qualification"] is True
        assert qualification["workload"]["class"] == "saturated"
        assert qualification["workload"]["label"] == \
            "sm120-mesh52-saturated"
        assert qualification["workload"]["mesh_n"] == 52
        assert qualification["workload"]["domain_periods"] == 1
        assert qualification["workload"]["dt"] == 1.0e-5
        assert qualification["workload"]["pairs_per_cell"] == 6
        assert qualification["export_control"]["mode"] == "igr"
        assert qualification["export_control"]["order"] == 3
        assert qualification["export_control"]["attribution"][
            "trace_policy_matched"
        ] is False
        assert set(qualification["final_tree_results"]) == expected_cells
        assert set(qualification["frozen_control_results"]) == expected_cells
        assert len(qualification["invocations"]) == 146
        assert qualification["integrity"]["raw_artifacts"][
            "artifact_count"
        ] == 292

        final_build = qualification["builds"]["final_tree"]
        frozen_build = qualification["builds"]["frozen_ordinary"]
        assert final_build["executable"]["sha256"] != \
            frozen_build["executable"]["sha256"]
        assert final_build["executable_build_relationship"][
            "canonical_target_path"
        ] == str(final.resolve())
        assert frozen_build["executable_build_relationship"][
            "canonical_target_path"
        ] == str(frozen.resolve())
        assert frozen_build["source_and_overlay"]["diff_sha256"] == \
            hashlib.sha256(synthetic_overlay).hexdigest()
        assert qualification["hardware"]["start"]["values"]["uuid"] == \
            "GPU-00000000-0000-0000-0000-000000000001"

        disposition = qualification["performance_disposition"]
        tensor_disposition = disposition["tensor_vs_final_tree_ordinary"]
        frozen_disposition = disposition[
            "final_tree_ordinary_vs_frozen_control"
        ]
        assert tensor_disposition["all_six_cells_disclosed"] is True
        assert tensor_disposition["attribution_by_mode"]["igr"][
            "tensor_trace_preconditioner"
        ] == "hypre_ilu0"
        assert tensor_disposition["attribution_by_mode"]["igr"][
            "trace_policy_matched"
        ] is False
        assert tensor_disposition["attribution_by_mode"]["igr"][
            "trace_preconditioner_configuration_matched"
        ] is False
        assert tensor_disposition["attribution_by_mode"]["igr"][
            "ordinary_ilu_factor_reuse_policy"
        ] == "refresh_each_trace_solve"
        assert tensor_disposition["attribution_by_mode"]["igr"][
            "tensor_ilu_factor_reuse_policy"
        ] == "first_trace_solve_per_attempt"
        assert tensor_disposition["attribution_by_mode"]["igr"][
            "ordinary_ilu_triangular_solve_policy"
        ]["kind"] == "direct"
        assert tensor_disposition["attribution_by_mode"]["igr"][
            "tensor_ilu_triangular_solve_policy"
        ] == {
            "kind": "iterative_jacobi",
            "lower_jacobi_iterations": 5,
            "upper_jacobi_iterations": 5,
            "jacobi_factor_spmv_updates_per_application": 8,
        }
        assert tensor_disposition["attribution_by_mode"]["igr"][
            "mma_contribution_requires_separate_profiler_attribution"
        ] is True
        assert len(tensor_disposition["cells"]) == 6
        assert tensor_disposition["saturated_pass"] is True
        assert tensor_disposition["disposition"] == \
            "pass_all_six_cells_win_beyond_observed_noise"
        assert frozen_disposition["all_six_cells_disclosed"] is True
        assert len(frozen_disposition["cells"]) == 6
        assert frozen_disposition[
            "has_regression_beyond_observed_noise"
        ] is False
        assert disposition["promotion_performance_pass"] is True

        break_even_disposition = CAMPAIGN.performance_disposition(
            qualification,
            evidence_mode="qualification",
            workload_class="break-even",
        )
        assert break_even_disposition["promotion_performance_pass"] is False
        assert break_even_disposition[
            "promotion_performance_disposition"
        ] == "characterization_only_not_a_promotion_gate"

        regression_manifest = copy.deepcopy(qualification)
        regression_manifest["frozen_control_results"]["conventional_p1"][
            "paired_comparison"
        ]["noise_disposition"] = "regression_beyond_observed_noise"
        regression_comparison = regression_manifest[
            "frozen_control_results"
        ]["conventional_p1"]["paired_comparison"]
        regression_comparison[
            "median_speedup_95_percent_confidence_interval"
        ]["lower"] = 0.8
        regression_comparison[
            "median_speedup_95_percent_confidence_interval"
        ]["upper"] = 0.9
        regression_comparison["speedup"]["median"] = 0.85
        regression_disposition = CAMPAIGN.performance_disposition(
            regression_manifest,
            evidence_mode="qualification",
            workload_class="saturated",
        )
        assert regression_disposition[
            "final_tree_ordinary_vs_frozen_control"
        ]["regressions_beyond_observed_noise"] == ["conventional_p1"]
        assert regression_disposition[
            "final_tree_ordinary_vs_frozen_control"
        ]["nominal_median_regressions"] == ["conventional_p1"]
        assert regression_disposition["promotion_performance_pass"] is False
        assert regression_disposition[
            "promotion_performance_disposition"
        ] == "fail_frozen_ordinary_regression"

        frozen_invocations = [
            item for item in qualification["invocations"]
            if item["comparison_member"] == "frozen_ordinary"
        ]
        assert len(frozen_invocations) == 36
        for result in qualification["frozen_control_results"].values():
            frozen_work = result["controls"]["frozen_ordinary"][
                "solver_work"
            ]
            assert set(frozen_work) == SCHEMA1_WORK_SUMMARY_FIELDS
            assert not SCHEMA2_ONLY_SUMMARY_FIELDS.intersection(frozen_work)
        for invocation in frozen_invocations:
            record = json.loads(
                (qualification_output / invocation["raw_json"]).read_text(
                    encoding="utf-8"
                )
            )
            assert record["schema"] == 1
            assert not SCHEMA2_ONLY_RECORD_FIELDS.intersection(record)
            assert all(
                not SCHEMA2_ONLY_SAMPLE_FIELDS.intersection(sample)
                for sample in record["samples"]
            )
            summary = invocation["benchmark_summary"]
            assert summary["schema"] == 1
            assert set(summary) == SCHEMA1_RECORD_SUMMARY_FIELDS
            assert not SCHEMA2_ONLY_SUMMARY_FIELDS.intersection(summary)
            assert set(summary["workload_identity"]) == \
                SCHEMA1_WORKLOAD_SUMMARY_FIELDS
            assert all(
                value is not None
                for value in summary["workload_identity"].values()
            )
            assert all(
                set(sample) == SCHEMA1_SAMPLE_SUMMARY_FIELDS
                for sample in summary["measured_sample_work"]
            )

        expect_campaign_error(lambda: CAMPAIGN.work_summary([
            ordinary_record, record,
        ]))
        unknown_schema = copy.deepcopy(record)
        unknown_schema["schema"] = 3
        expect_campaign_error(lambda: CAMPAIGN.record_summary(unknown_schema))

        for build in (final_build, frozen_build):
            retained_hypre = build["provenance"][
                "validated_hypre_identity"
            ]
            assert retained_hypre["spack_dag_hash"] == SYNTHETIC_HYPRE_HASH
            assert retained_hypre["manifest_fields"][
                "hypre_source_revision"
            ] == SYNTHETIC_HYPRE_REVISION
    print("hdg_tensor_campaign smoke and qualification tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
