#!/usr/bin/env python3
"""Run a bounded, aggregate-only shock architecture selection matrix.

The caller supplies an already-built unified application executable, an
absolute or resolvable mesh, a fixed physical horizon, and an ordered set of
typed methods.  This driver performs no build, GPU selection, or resource
leasing.  Every sample is a fresh process and the samples are cyclically
rotated to reduce order bias.

Only concise JSON and Markdown aggregates are retained.  Application stdout,
stderr, GNU-time records, VTK files, and optional postprocessor output live in
a temporary directory and are removed after their bounded fields are parsed.
An application failure is selection data, not a harness failure.  A wall
timeout is reported as right-censored and is never presented as a successful
time to solution or as an observed time to failure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 3
REPORT_JSON = "shock-architecture-selection.json"
REPORT_MARKDOWN = "shock-architecture-selection.md"
DEFAULT_TIME_PROGRAM = "/usr/bin/time"
MAX_REPETITIONS = 5
MAX_FAILURE_REASON_CHARS = 240
QOI_FIELDS = (
    "shock_position",
    "shock_standoff_distance",
    "shock_profile_l1_error",
    "shock_profile_l2_error",
    "shock_thickness",
    "density_minimum",
    "density_maximum",
    "pressure_minimum",
    "pressure_maximum",
    "conservation_error",
    "boundary_balance_error",
    "sigma_localization",
)
QOI_FILE_INPUT_OPTIONS = frozenset({
    "--reference-profile",
})
QOI_HARNESS_OWNED_OPTIONS = frozenset({
    "--vtk-root",
    "--method",
    "--target-time",
    "--mesh",
})

DG_RK3SSP = "dg-rk3ssp"
DG_SSPRK104 = "dg-ssprk104"
HDG_RODAS3_ORDINARY = "hdg-rodas3-ordinary"
HDG_RODAS3_TENSOR = "hdg-rodas3-tensor"
METHODS = (
    DG_RK3SSP,
    DG_SSPRK104,
    HDG_RODAS3_ORDINARY,
    HDG_RODAS3_TENSOR,
)
EXPLICIT_METHODS = (DG_RK3SSP, DG_SSPRK104)
METHOD_INTEGRATORS = {
    DG_RK3SSP: "rk3ssp",
    DG_SSPRK104: "ssprk104",
    HDG_RODAS3_ORDINARY: "rodas3",
    HDG_RODAS3_TENSOR: "rodas3",
}
METHOD_ENGINES = {
    HDG_RODAS3_ORDINARY: "ordinary",
    HDG_RODAS3_TENSOR: "tensor",
}

PROTECTED_APP_OPTIONS = frozenset({
    "-dev", "--device", "-m", "--mesh", "-o", "--order",
    "-rs", "--serial-refine", "-rp", "--parallel-refine",
    "-dg", "--dg-backend",
    "-disc", "--discretization", "-reg", "--regularization",
    "-ti", "--time-integrator", "-hce", "--hdg-cuda-engine",
    "-hia", "--hdg-igr-alpha", "-hib", "--hdg-igr-beta",
    "-ds", "--dgsem-stab", "-dvf", "--dgsem-volume-flux",
    "-dvd", "--dgsem-volume-dissipation", "-dsf",
    "--dgsem-surface-flux", "-dej", "--dgsem-entropy-jacobian",
    "-tf", "--t-final", "-dt", "--time-step", "-dtmin", "--dt-min",
    "-c", "--cfl-number", "-ca", "--c-alpha",
    "-ial", "--igr-alpha-length", "-is", "--igr-space", "-ig",
    "--igr-grad", "-im", "--igr-solve-mode", "-ils",
    "--igr-linear-solver", "-ipc", "--igr-preconditioner", "-incp",
    "--igr-nonconvergence-policy", "-iscad", "--igr-sigma-cadence",
    "-isn", "--igr-sigma-update-interval", "-isrtol",
    "--igr-stage-rtol-scale", "-iip", "--igr-ip-kind", "-irtol",
    "--igr-lin-rtol", "-iatol", "--igr-lin-atol", "-imaxit",
    "--igr-lin-maxit", "-iprint", "--igr-lin-print", "-iamgc",
    "--igr-amg-coarsen-type", "-iamgr", "--igr-amg-relax-type",
    "-iamgl", "--igr-amg-max-levels", "-iamgst",
    "--igr-amg-strong-threshold", "-iexface", "--igr-exact-dg-face",
    "-ino-exface", "--no-igr-exact-dg-face", "-isigcfl",
    "--igr-sigma-cfl", "-ino-sigcfl", "--no-igr-sigma-cfl",
    "-isigrus", "--igr-sigma-rusanov", "-ino-sigrus",
    "--no-igr-sigma-rusanov", "-iidflux", "--igr-invariant-domain-flux",
    "-ino-iidflux", "--no-igr-invariant-domain-flux",
    "-tatol", "--time-atol",
    "-trtol", "--time-rtol", "-pl", "--positivity-limiter",
    "-zsp", "--zhang-shu-points", "-zsvtk",
    "--zhang-shu-vtk-points", "-no-zsvtk",
    "--no-zhang-shu-vtk-points", "-vtk", "--vtk_save",
    "-no-vtk", "--no-vtk_save", "-dt_save", "--time-step-save",
    "-vtk-h1", "--vtk-h1-diagnostics", "-no-vtk-h1",
    "--no-vtk-h1-diagnostics", "-vtk-l2", "--vtk-l2-diagnostics",
    "-no-vtk-l2", "--no-vtk-l2-diagnostics", "-vtk-rank",
    "--vtk-mpi-rank-field", "-no-vtk-rank",
    "--no-vtk-mpi-rank-field", "-vtk-zs",
    "--vtk-zhang-shu-diagnostics", "-no-vtk-zs",
    "--no-vtk-zhang-shu-diagnostics",
    "-glvis", "--glvis", "-no-glvis", "--no-glvis",
    "-save-state", "--save-state", "-no-save-state", "--no-save-state",
    "-ea", "--element-assembly-divergence", "-mf",
    "--matrix-free-divergence",
    "-vs", "--visualization-steps", "-as", "--analysis-steps",
    "-nan-scan", "--nan-scan", "-no-nan-scan", "--no-nan-scan",
    "-adm-scan", "--admissibility-scan", "-no-adm-scan",
    "--no-admissibility-scan",
    "-conservation-balance", "--conservation-balance",
    "-no-conservation-balance", "--no-conservation-balance",
    "-restart-vtk", "--restart-vtk", "-restart-cycle",
    "--restart-cycle", "-restart-step", "--restart-step",
    "-hdg-checkpoint", "--hdg-checkpoint", "-hdg-restart-checkpoint",
    "--hdg-restart-checkpoint",
})

SCALAR_CLOSURE_INTEGER_FIELDS = (
    "calls", "converged_calls", "production_calls",
    "production_converged_calls", "output_only_calls",
    "output_only_converged_calls", "output_only_iters", "total_iters",
    "max_iters", "total_cheb_eigen_reuses", "total_cheb_smoother_reuses",
    "total_cheb_smoother_builds", "total_krylov_operator_apply_count",
    "total_krylov_preconditioner_apply_count",
    "total_krylov_preconditioner_operator_apply_count",
    "direct_identity_uses", "exact_face_local_nf", "exact_face_shared_nf",
    "exact_face_operator_uses", "exact_face_last_mult_calls",
)
SCALAR_CLOSURE_FLOAT_FIELDS = (
    "avg_iters", "avg_solve_iter_s", "total_assemble_s", "total_form_s",
    "total_pc_setup_s", "total_cheb_diag_s", "total_cheb_setup_s",
    "total_krylov_operator_apply_s", "total_krylov_preconditioner_apply_s",
    "total_krylov_preconditioner_operator_apply_s", "total_solve_s",
    "total_igr_s", "total_update_s", "solve_pct", "total_igr_pct",
    "solve_pct_stage", "total_igr_pct_stage",
)
SCALAR_CLOSURE_TEXT_FIELDS = ("last_solver", "last_precond")
SCALAR_CLOSURE_BOOLEAN_FIELDS = (
    "direct_identity_system", "exact_face_requested", "exact_face_effective",
)
ARCHITECTURE_WORK_INTEGER_FIELDS = (
    "nonlinear_updates", "linear_iterations", "linear_solve_attempts",
    "trace_solves", "scalar_closure_solves",
    "scalar_closure_linear_iterations", "complete_assemblies",
    "complete_condensations", "complete_recoveries",
    "positivity_limiter_calls", "positivity_limiter_active_calls",
    "positivity_limited_elements",
)
ARCHITECTURE_WORK_FLOAT_FIELDS = (
    "minimum_density", "minimum_pressure", "maximum_wave_speed",
    "maximum_final_residual", "maximum_conservation_error",
    "maximum_final_algebraic_residual", "minimum_positivity_limiter_theta",
)
EXPLICIT_TIME_INTEGER_FIELDS = (
    "steps", "rhs_evaluations", "accepted_attempts", "rejected_attempts",
    "rejected_limiter_elements", "characteristic_speed_probes",
)
EXPLICIT_TIME_FLOAT_FIELDS = (
    "final_time", "minimum_attempted_dt", "maximum_attempted_dt",
    "minimum_accepted_dt", "maximum_accepted_dt",
    "minimum_characteristic_speed", "maximum_characteristic_speed",
)
HDG_INTEGER_FIELDS = (
    "step", "output_cycle", "fallback_count", "rejected_attempts",
    "committed_step", "consecutive_rejections",
)
HDG_FLOAT_FIELDS = (
    "time", "c_alpha", "alpha_effective", "h_eff",
    "euler_trace_penalty", "accepted_dt", "next_dt",
    "minimum_accepted_dt", "maximum_accepted_dt", "committed_t",
    "attempted_dt", "minimum_reported_accepted_dt",
    "maximum_reported_accepted_dt",
)

FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
KEY_VALUE_RE = re.compile(
    r"(?:^|\s)([A-Za-z_][A-Za-z0-9_.-]*)=(\"[^\"]*\"|\S+)"
)
STEP_SUMMARY_RE = re.compile(
    r"steps \(cumulative/segment\):\s*(\d+)\s*/\s*(\d+)"
)
TIME_SUMMARY_RE = re.compile(
    rf"time \(initial/final\):\s*({FLOAT_PATTERN})\s*/\s*({FLOAT_PATTERN})"
)
ELAPSED_SUMMARY_RE = re.compile(rf"elapsed:\s*({FLOAT_PATTERN})\s*s")
ALIVE_RE = re.compile(
    rf"\[alive\]\s+step\s+(\d+)\s*\|\s*t\s+({FLOAT_PATTERN})"
)
TIME_RECORD_RE = re.compile(
    r"^DG_IGR_EXTERNAL_TIME\s+"
    rf"wall_seconds=({FLOAT_PATTERN})\s+"
    r"peak_rss_kib=(\d+)\s+exit_status=(-?\d+)\s*$"
)


class ShockArchitectureError(RuntimeError):
    """A configuration, launch, parser, or report-publication failure."""


@dataclass(frozen=True)
class HarnessControls:
    device: str
    order: int
    serial_refine: int
    parallel_refine: int
    final_time: float
    c_alpha: float
    alpha_length_policy: str
    dgsem_surface_flux: str
    igr_invariant_domain_flux: bool
    igr_sigma_cfl: bool
    igr_sigma_rusanov: bool
    conservation_balance: bool
    zhang_shu_vtk_points: bool
    closure_certificate_max_relative_residual: float
    explicit_dt_by_method: Mapping[str, float]
    explicit_cfl_by_method: Mapping[str, float]
    dt_min: float
    igr_linear_rtol: float
    igr_linear_atol: float
    igr_linear_max_iterations: int
    rodas_initial_dt: float
    rodas_time_atol: float
    rodas_time_rtol: float
    progress_steps: int
    vtk: bool
    vtk_save_interval: float


@dataclass(frozen=True)
class TimedProcessResult:
    returncode: int
    timed_out: bool
    wall_seconds: float
    wall_source: str
    peak_rss_kib: int | None
    time_exit_status: int | None
    stdout_tail: str
    stderr_tail: str
    stdout_bytes: int
    stderr_bytes: int
    stdout_truncated: bool
    stderr_truncated: bool


def _finite_positive(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ShockArchitectureError(f"{name} must be finite and positive")
    return converted


def _finite_nonnegative(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ShockArchitectureError(f"{name} must be finite and nonnegative")
    return converted


def parse_methods(value: str) -> tuple[str, ...]:
    """Parse one caller-ordered, duplicate-free method list."""
    methods = tuple(item.strip() for item in value.split(",") if item.strip())
    if not methods:
        raise ShockArchitectureError("--methods must select at least one method")
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ShockArchitectureError(
            "unknown methods: " + ",".join(unknown) +
            "; choices are " + ",".join(METHODS)
        )
    if len(set(methods)) != len(methods):
        raise ShockArchitectureError("--methods must not contain duplicates")
    return methods


def rotated_schedule(
        methods: Sequence[str], repetitions: int,
        rotation_offset: int = 0) -> tuple[tuple[str, ...], ...]:
    """Return cyclic Latin rows beginning from a caller-selected offset."""
    parsed = parse_methods(",".join(methods))
    if repetitions < 1 or repetitions > MAX_REPETITIONS:
        raise ShockArchitectureError(
            f"repetitions must be in [1,{MAX_REPETITIONS}]"
        )
    start = rotation_offset % len(parsed)
    rows = []
    for repetition in range(repetitions):
        shift = (start + repetition) % len(parsed)
        rows.append(parsed[shift:] + parsed[:shift])
    return tuple(rows)


def _option_name(argument: str) -> str:
    return argument.split("=", 1)[0]


def validate_extra_app_arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    """Prevent caller extras from silently changing typed-card invariants."""
    values = tuple(str(argument) for argument in arguments)
    for argument in values:
        if not argument:
            raise ShockArchitectureError("--app-arg values must be non-empty")
        if _option_name(argument) in PROTECTED_APP_OPTIONS:
            raise ShockArchitectureError(
                f"--app-arg may not override typed option {_option_name(argument)}"
            )
    return values


def build_method_arguments(
        method: str, mesh: str, controls: HarnessControls,
        extra_arguments: Sequence[str] = ()) -> list[str]:
    """Build one exact full-energy, active-IGR typed application card."""
    if method not in METHODS:
        raise ShockArchitectureError(f"unknown method: {method}")
    if controls.device == "cpu" and method == HDG_RODAS3_TENSOR:
        raise ShockArchitectureError("tensor HDG requires --device cuda")
    if method not in EXPLICIT_METHODS and \
            controls.alpha_length_policy == "local":
        raise ShockArchitectureError(
            "HDG methods do not support the element-local alpha policy"
        )
    extras = validate_extra_app_arguments(extra_arguments)
    arguments = [
        "--device", controls.device,
        "--mesh", mesh,
        "--order", str(controls.order),
        "--serial-refine", str(controls.serial_refine),
        "--parallel-refine", str(controls.parallel_refine),
        "--regularization", "igr",
        "--c-alpha", format(controls.c_alpha, ".17g"),
        "--igr-alpha-length", controls.alpha_length_policy,
        "--t-final", format(controls.final_time, ".17g"),
        "--dt-min", format(controls.dt_min, ".17g"),
        "--positivity-limiter", "zhang-shu",
        "--zhang-shu-points", "consumer",
        "--visualization-steps", str(controls.progress_steps),
        "--analysis-steps", "0",
        "--nan-scan",
        "--admissibility-scan",
        "--no-glvis",
        "--no-save-state",
    ]
    if controls.vtk:
        arguments.extend([
            "--vtk_save",
            "--time-step-save", format(controls.vtk_save_interval, ".17g"),
        ])
    else:
        arguments.append("--no-vtk_save")

    if method in EXPLICIT_METHODS:
        arguments.extend([
            "--discretization", "dg",
            "--dg-backend", "dg",
            "--dgsem-surface-flux", controls.dgsem_surface_flux,
            "--time-integrator", METHOD_INTEGRATORS[method],
            "--time-step",
            format(controls.explicit_dt_by_method[method], ".17g"),
            "--cfl-number",
            format(controls.explicit_cfl_by_method[method], ".17g"),
            "--igr-space", "L2",
            "--igr-grad", "BrokenL2",
            "--igr-solve-mode", "krylov",
            "--igr-ip-kind", "sipg",
            "--igr-sigma-cadence", "every-stage",
            "--igr-stage-rtol-scale", "1",
            "--igr-lin-rtol", format(controls.igr_linear_rtol, ".17g"),
            "--igr-lin-atol", format(controls.igr_linear_atol, ".17g"),
            "--igr-lin-maxit", str(controls.igr_linear_max_iterations),
            "--igr-linear-solver", "minres",
            "--igr-preconditioner", "chebyshev",
            "--igr-exact-dg-face",
            "--igr-invariant-domain-flux"
            if controls.igr_invariant_domain_flux
            else "--no-igr-invariant-domain-flux",
            "--igr-sigma-cfl" if controls.igr_sigma_cfl
            else "--no-igr-sigma-cfl",
            "--igr-sigma-rusanov" if controls.igr_sigma_rusanov
            else "--no-igr-sigma-rusanov",
            "--igr-nonconvergence-policy", "error",
        ])
        arguments.append(
            "--zhang-shu-vtk-points"
            if controls.zhang_shu_vtk_points
            else "--no-zhang-shu-vtk-points"
        )
        arguments.append(
            "--conservation-balance"
            if controls.conservation_balance
            else "--no-conservation-balance"
        )
    else:
        arguments.extend([
            "--discretization", "hdg",
            "--time-integrator", "rodas3",
            "--hdg-cuda-engine", METHOD_ENGINES[method],
            "--time-step", format(-controls.rodas_initial_dt, ".17g"),
            "--cfl-number", "0",
            "--time-atol", format(controls.rodas_time_atol, ".17g"),
            "--time-rtol", format(controls.rodas_time_rtol, ".17g"),
        ])
    arguments.extend(extras)
    return arguments


def method_closure_policy(method: str) -> dict[str, Any]:
    """Describe the closure actually requested by each typed card."""
    if method in EXPLICIT_METHODS:
        return {
            "mathematical_model": "instantaneous-igr-index-one-dae",
            "architecture": "constraint-projected-explicit-elliptic",
            "scalar_space": "L2",
            "gradient_reconstruction": "BrokenL2",
            "scalar_solve_mode": "krylov",
            "reconstruction_cadence": "every-stage",
            "linear_nonconvergence_policy": "error",
            "exact_dg_face_operator_requested": True,
            "application_metrics": (
                "explicit and DG scalar-closure summaries plus the final "
                "accepted-state closure certificate and certification-support "
                "status"
            ),
        }
    if method in (HDG_RODAS3_ORDINARY, HDG_RODAS3_TENSOR):
        return {
            "mathematical_model": "instantaneous-igr-index-one-dae",
            "architecture": "coupled-traced-hdg-rodas3",
            "scalar_space": "HDG backend-owned traced scalar closure",
            "gradient_reconstruction": "HDG backend-owned algebraic fields",
            "scalar_solve_mode": "HDG backend-owned coupled solve",
            "reconstruction_cadence": (
                "RODAS3 stage algebraic reconstruction owned by HDG backend"
            ),
            "linear_nonconvergence_policy": "HDG backend policy",
            "exact_dg_face_operator_requested": False,
            "application_metrics": (
                "HDG progress/status only unless additional aggregates are "
                "emitted; DG scalar-closure counters remain null"
            ),
        }
    raise ShockArchitectureError(f"unknown method: {method}")


def _parse_key_values(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in KEY_VALUE_RE.finditer(line):
        value = match.group(2)
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]
        fields[match.group(1)] = value
    return fields


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(str(value))
    except ValueError:
        return None
    return converted if math.isfinite(converted) else None


def _as_float_vector(value: Any) -> list[float] | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = [_as_float(item) for item in value.split(",")]
    if not parsed or any(item is None for item in parsed):
        return None
    return [float(item) for item in parsed]


def _bounded_reason(value: str) -> str:
    flattened = " ".join(value.split())
    if not flattened:
        return "no_diagnostic_reason"
    return flattened[:MAX_FAILURE_REASON_CHARS]


def _structured_lines(log: str, marker: str,
                      event: str | None = None) -> list[dict[str, str]]:
    matches = []
    for raw_line in log.splitlines():
        line = raw_line.strip()
        if marker not in line:
            continue
        fields = _parse_key_values(line)
        if event is None or fields.get("event") == event:
            matches.append(fields)
    return matches


def _typed_fields(
        fields: Mapping[str, str], integer_names: Sequence[str],
        float_names: Sequence[str], text_names: Sequence[str] = (),
        boolean_names: Sequence[str] = ()) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in integer_names:
        value = _as_int(fields.get(name))
        if value is not None:
            result[name] = value
    for name in float_names:
        value = _as_float(fields.get(name))
        if value is not None:
            result[name] = value
    for name in text_names:
        if name in fields:
            result[name] = fields[name]
    for name in boolean_names:
        value = _as_int(fields.get(name))
        if value in (0, 1):
            result[name] = bool(value)
    return result


def _nullable_typed_fields(
        fields: Mapping[str, str], integer_names: Sequence[str],
        float_names: Sequence[str], text_names: Sequence[str] = (),
        boolean_names: Sequence[str] = ()) -> dict[str, Any]:
    """Return a fixed nullable schema without synthesizing missing metrics."""
    result: dict[str, Any] = {
        "reported": bool(fields),
        **{name: None for name in integer_names},
        **{name: None for name in float_names},
        **{name: None for name in text_names},
        **{name: None for name in boolean_names},
    }
    result.update(_typed_fields(
        fields, integer_names, float_names, text_names, boolean_names
    ))
    return result


def _infer_failure_reason(log: str, hdg_rejection: str | None) -> str | None:
    if hdg_rejection:
        return _bounded_reason(hdg_rejection)
    explicit_final_rows = _structured_lines(
        log, "[explicit-time]", "final-status"
    )
    explicit_rows = explicit_final_rows or _structured_lines(
        log, "[explicit-time]", "run-summary"
    )
    if explicit_rows:
        if explicit_rows[-1].get("success") == "1":
            return None
        termination = explicit_rows[-1].get("termination_reason")
        if termination and termination != "target-reached":
            return _bounded_reason(termination)
    patterns = (
        (r"\[dt-check\].*below minimum threshold", "dt_below_minimum"),
        (r"\[dt-check\].*[Nn]on-finite", "nonfinite_timestep"),
        (r"\[dt-check\].*non-positive", "nonpositive_timestep"),
        (r"\[nan-scan\].*Detected", "nonfinite_state"),
        (r"\[admissibility-scan\].*event=euler", "admissibility_violation"),
        (r"\[igr-linear\].*event=nonconvergence-policy", "igr_nonconvergence"),
        (r"MFEM abort", "mfem_abort"),
        (r"terminate called", "uncaught_exception"),
    )
    for pattern, reason in patterns:
        if re.search(pattern, log):
            return reason
    return None


def _last_diagnostic_excerpt(log: str) -> str | None:
    nonempty = [line.strip() for line in log.splitlines() if line.strip()]
    return _bounded_reason(nonempty[-1]) if nonempty else None


def parse_application_log(log: str, method: str) -> dict[str, Any]:
    """Parse bounded synthetic or production output into run aggregates."""
    if method not in METHODS:
        raise ShockArchitectureError(f"unknown method: {method}")
    steps: int | None = None
    segment_steps: int | None = None
    initial_time: float | None = None
    final_time: float | None = None
    internal_wall: float | None = None
    progress: list[tuple[float, int | None]] = []
    authoritative_time: float | None = None
    authoritative_step: int | None = None
    authoritative_source: str | None = None

    step_matches = list(STEP_SUMMARY_RE.finditer(log))
    if step_matches:
        steps = int(step_matches[-1].group(1))
        segment_steps = int(step_matches[-1].group(2))
    time_matches = list(TIME_SUMMARY_RE.finditer(log))
    if time_matches:
        initial_time = float(time_matches[-1].group(1))
        final_time = float(time_matches[-1].group(2))
        progress.append((final_time, steps))
    elapsed_matches = list(ELAPSED_SUMMARY_RE.finditer(log))
    if elapsed_matches:
        internal_wall = float(elapsed_matches[-1].group(1))

    for match in ALIVE_RE.finditer(log):
        progress.append((float(match.group(2)), int(match.group(1))))

    explicit_summary_rows = _structured_lines(
        log, "[explicit-time]", "run-summary"
    )
    explicit_final_rows = _structured_lines(
        log, "[explicit-time]", "final-status"
    )
    legacy_retry_rows = _structured_lines(
        log, "[explicit-retry]", "run-summary"
    )
    explicit_fields = dict(
        explicit_summary_rows[-1] if explicit_summary_rows else (
            legacy_retry_rows[-1] if legacy_retry_rows else {}
        )
    )
    if explicit_final_rows:
        explicit_fields.update(explicit_final_rows[-1])
    explicit_time = _nullable_typed_fields(
        explicit_fields,
        integer_names=EXPLICIT_TIME_INTEGER_FIELDS,
        float_names=EXPLICIT_TIME_FLOAT_FIELDS,
        text_names=("termination_reason",),
        boolean_names=("dt_min_reached", "success"),
    )
    explicit_time["source_marker"] = (
        "explicit-time-final-status" if explicit_final_rows else
        "explicit-time-run-summary" if explicit_summary_rows else
        "explicit-retry" if legacy_retry_rows else None
    )
    explicit_time["run_summary_reported"] = bool(explicit_summary_rows)
    explicit_time["final_status_reported"] = bool(explicit_final_rows)
    if method in EXPLICIT_METHODS and (
            explicit_summary_rows or explicit_final_rows):
        summary_steps = _as_int(explicit_fields.get("steps"))
        summary_time = _as_float(explicit_fields.get("final_time"))
        if summary_steps is not None:
            steps = summary_steps
        if summary_time is not None:
            final_time = summary_time
            authoritative_time = summary_time
            authoritative_step = summary_steps
            authoritative_source = (
                "explicit-final-status" if explicit_final_rows
                else "explicit-run-summary"
            )
            progress.append((summary_time, summary_steps))

    hdg_progress_rows = _structured_lines(
        log, "[hdg-progress]", "accepted"
    )
    hdg_reported_dts: list[float] = []
    for row in hdg_progress_rows:
        value = _as_float(row.get("time"))
        step = _as_int(row.get("step"))
        accepted_dt = _as_float(row.get("accepted_dt"))
        if value is not None:
            progress.append((value, step))
        if accepted_dt is not None:
            hdg_reported_dts.append(accepted_dt)

    hdg_success_fields: dict[str, str] | None = None
    hdg_rejection_fields: dict[str, str] = {}
    hdg_rejection: str | None = None
    for line in log.splitlines():
        if "[hdg-main]" not in line:
            continue
        fields = _parse_key_values(line)
        if fields.get("status") == "Success":
            hdg_success_fields = fields
            value = _as_float(fields.get("time"))
            step = _as_int(fields.get("step"))
            if value is not None:
                authoritative_time = value
                authoritative_step = step
                authoritative_source = "hdg-success"
                progress.append((value, step))
        elif fields.get("status") == "Rejected":
            hdg_rejection_fields = fields
            match = re.search(
                r"\breason=(.*?)(?:\s+cuda_engine=\S+)?\s*$", line
            )
            hdg_rejection = match.group(1) if match else "hdg_rejected"
            committed_time = _as_float(fields.get("committed_t"))
            committed_step = _as_int(fields.get("committed_step"))
            if committed_time is not None:
                authoritative_time = committed_time
                authoritative_step = committed_step
                authoritative_source = "hdg-rejected-committed-state"
                progress.append((committed_time, committed_step))

    runtime_time_patterns = (
        re.compile(rf"\[dt-check\].*\bt=({FLOAT_PATTERN})"),
        re.compile(rf"\[nan-scan\].*\bt=({FLOAT_PATTERN})"),
        re.compile(rf"\[admissibility-scan\].*\bt=({FLOAT_PATTERN})"),
        re.compile(rf"\[time-loop\].*\btime=({FLOAT_PATTERN})"),
        re.compile(rf"\[vtk-output\].*\bt=({FLOAT_PATTERN})"),
    )
    for pattern in runtime_time_patterns:
        for match in pattern.finditer(log):
            progress.append((float(match.group(1)), None))
    diagnostic_progress_time = max(
        (item[0] for item in progress), default=None
    )
    if authoritative_time is not None:
        final_time = authoritative_time
        if steps is None and authoritative_step is not None:
            steps = authoritative_step
    elif progress:
        _, reached_step = max(progress, key=lambda item: item[0])
        final_time = None
        if steps is None and reached_step is not None:
            steps = reached_step

    accepted_steps = _as_int(explicit_fields.get("accepted_attempts")) \
        if method in EXPLICIT_METHODS else None
    rejected_attempts = _as_int(explicit_fields.get("rejected_attempts")) \
        if method in EXPLICIT_METHODS else None
    if hdg_success_fields is not None:
        hdg_steps = _as_int(hdg_success_fields.get("step"))
        if hdg_steps is not None:
            steps = hdg_steps
            accepted_steps = hdg_steps
        rejected_attempts = _as_int(
            hdg_success_fields.get("rejected_attempts")
        )
    elif method not in EXPLICIT_METHODS:
        committed_step = _as_int(hdg_rejection_fields.get("committed_step"))
        last_progress = hdg_progress_rows[-1] if hdg_progress_rows else {}
        accepted_steps = committed_step \
            if committed_step is not None else _as_int(last_progress.get("step"))
        rejected_attempts = _as_int(last_progress.get("rejected_attempts"))

    igr_lines = _structured_lines(log, "[igr-linear]", "run-summary")
    igr_fields = igr_lines[-1] if igr_lines else {}
    closure = _nullable_typed_fields(
        igr_fields,
        integer_names=SCALAR_CLOSURE_INTEGER_FIELDS,
        float_names=SCALAR_CLOSURE_FLOAT_FIELDS,
        text_names=SCALAR_CLOSURE_TEXT_FIELDS,
        boolean_names=SCALAR_CLOSURE_BOOLEAN_FIELDS,
    )

    lifecycle_lines = _structured_lines(
        log, "[igr-sigma-lifecycle]", "run-summary"
    )
    lifecycle = _nullable_typed_fields(
        lifecycle_lines[-1] if lifecycle_lines else {},
        integer_names=(
            "prefetch_solves", "stage0_reuses", "output_only_solves",
            "prefetch_invalidations",
        ),
        float_names=(),
    )

    q_sigma_rows = _structured_lines(
        log, "[igr-q-sigma]", "run-summary"
    )
    q_sigma_telemetry = _nullable_typed_fields(
        q_sigma_rows[-1] if q_sigma_rows else {},
        integer_names=(
            "sample_count", "nonpositive_count", "nonfinite_count",
        ),
        float_names=(
            "minimum_raw_q_sigma", "minimum_q_sigma_over_gamma_p",
        ),
        text_names=("scope", "precision"),
        boolean_names=("rejected_attempts_included",),
    )

    limiter_rows = _structured_lines(log, "[limiter]", "run-summary")
    limiter_integer_names = (
        "calls", "active_calls", "limited_elements",
        "element_call_opportunities",
    )
    limiter_float_names = (
        "total_s", "avg_s", "avg_active_elements", "avg_tensor_points",
        "avg_mfem_volume_points", "avg_mfem_face_points", "avg_points",
        "avg_collapsed_elements", "avg_pressure_bisections",
        "limited_element_fraction", "minimum_theta",
    )
    limiter_text_names = (
        "aggregation", "limited_fraction_denominator",
    )
    limiter = _nullable_typed_fields(
        {}, limiter_integer_names, limiter_float_names, limiter_text_names
    )
    limiter["reported_ranks"] = 0
    if limiter_rows:
        typed_rows = [
            _typed_fields(
                row, limiter_integer_names, limiter_float_names,
                limiter_text_names,
            )
            for row in limiter_rows
        ]
        limiter["reported"] = True
        limiter["reported_ranks"] = len(typed_rows)
        for name in limiter_integer_names:
            values = [row[name] for row in typed_rows if name in row]
            if values:
                limiter[name] = max(values)
        for name in limiter_float_names:
            values = [row[name] for row in typed_rows if name in row]
            if values:
                limiter[name] = min(values) if name == "minimum_theta" \
                    else max(values)
        for name in limiter_text_names:
            values = [row[name] for row in typed_rows if name in row]
            if values:
                limiter[name] = values[0] if len(set(values)) == 1 else "mixed"

    architecture_rows = _structured_lines(log, "ARCHITECTURE_SELECTION_POINT")
    architecture = _nullable_typed_fields(
        architecture_rows[-1] if architecture_rows else {},
        integer_names=ARCHITECTURE_WORK_INTEGER_FIELDS,
        float_names=ARCHITECTURE_WORK_FLOAT_FIELDS,
    )

    certificate_rows = _structured_lines(
        log, "[igr-closure-certificate]", "accepted-state"
    )
    certificate_fields = certificate_rows[-1] if certificate_rows else {}
    closure_certificate = _nullable_typed_fields(
        certificate_fields,
        integer_names=("state_generation",),
        float_names=(
            "time", "residual_rel_rhs", "residual_abs", "relative_limit",
            "absolute_limit",
        ),
        text_names=("scope", "precision"),
        boolean_names=(
            "solved", "supported", "certified", "run_success",
            "skipped_nonfinite",
        ),
    )
    closure_certificate["residual_reported"] = \
        "residual_rel_rhs" in certificate_fields
    closure_certificate["residual_finite"] = \
        closure_certificate["residual_rel_rhs"] is not None

    conservation_rows = _structured_lines(
        log, "[explicit-conservation]", "accepted-trajectory"
    )
    conservation_fields = conservation_rows[-1] if conservation_rows else {}
    conservation = _nullable_typed_fields(
        conservation_fields,
        integer_names=("components", "accepted_attempts", "rejected_attempts"),
        float_names=(
            "domain_measure", "maximum_absolute_error",
            "maximum_relative_error",
        ),
        boolean_names=("supported", "source_supported", "run_success"),
    )
    for name in (
            "initial", "final", "boundary_flux", "boundary_flux_scale",
            "defect", "relative_defect"):
        conservation[name] = _as_float_vector(conservation_fields.get(name))

    regularization_rows = _structured_lines(
        log, "[igr-regularization]", "resolved"
    )
    regularization = _nullable_typed_fields(
        regularization_rows[-1] if regularization_rows else {},
        integer_names=(),
        float_names=(
            "c_alpha", "h_min", "h_average", "h_max",
            "h_eff_reference", "alpha_effective_min",
            "alpha_effective_max",
        ),
        text_names=("h_eff_policy",),
    )

    hdg_fields: dict[str, str] = {}
    if hdg_progress_rows:
        hdg_fields.update(hdg_progress_rows[-1])
    if hdg_rejection_fields:
        hdg_fields.update(hdg_rejection_fields)
    if hdg_success_fields is not None:
        hdg_fields.update(hdg_success_fields)
    hdg = _nullable_typed_fields(
        hdg_fields,
        integer_names=HDG_INTEGER_FIELDS,
        float_names=HDG_FLOAT_FIELDS,
        text_names=(
            "status", "cuda_engine", "time_treatment",
            "requested_trace_policy", "resolved_trace_policy",
            "trace_solver", "trace_preconditioner", "regularization",
            "positivity_limiter", "zhang_shu_points", "alpha_length_policy",
        ),
    )
    hdg["accepted_progress_records"] = len(hdg_progress_rows)
    if hdg_reported_dts:
        hdg["minimum_reported_accepted_dt"] = min(hdg_reported_dts)
        hdg["maximum_reported_accepted_dt"] = max(hdg_reported_dts)

    bad_admissibility = _structured_lines(log, "[admissibility-scan]", "euler")
    admissibility: dict[str, Any] = {"violation_records": len(bad_admissibility)}
    if bad_admissibility:
        bad_rho = [_as_int(row.get("bad_rho")) for row in bad_admissibility]
        bad_pressure = [
            _as_int(row.get("bad_pressure")) for row in bad_admissibility
        ]
        rho_values = [_as_float(row.get("min_rho")) for row in bad_admissibility]
        pressure_values = [
            _as_float(row.get("min_pressure")) for row in bad_admissibility
        ]
        admissibility.update({
            "maximum_bad_density_points": max(
                (value for value in bad_rho if value is not None), default=None
            ),
            "maximum_bad_pressure_points": max(
                (value for value in bad_pressure if value is not None),
                default=None,
            ),
            "minimum_density": min(
                (value for value in rho_values if value is not None),
                default=None,
            ),
            "minimum_pressure": min(
                (value for value in pressure_values if value is not None),
                default=None,
            ),
        })

    reason = _infer_failure_reason(log, hdg_rejection)
    return {
        "steps": steps,
        "segment_steps": segment_steps,
        "accepted_steps": accepted_steps,
        "rejected_attempts": rejected_attempts,
        "initial_time": initial_time,
        "time_reached": final_time,
        "time_reached_authority": authoritative_source,
        "diagnostic_progress_time": diagnostic_progress_time,
        "internal_time_loop_wall_seconds": internal_wall,
        "failure_reason_from_log": reason,
        "explicit_time": explicit_time,
        "scalar_closure": closure,
        "scalar_lifecycle": lifecycle,
        "frozen_sigma_telemetry": q_sigma_telemetry,
        "limiter": limiter,
        "architecture_work": architecture,
        "closure_certificate": closure_certificate,
        "conservation": conservation,
        "regularization": regularization,
        "hdg": hdg,
        "admissibility": admissibility,
    }


def classify_execution(
        parsed: Mapping[str, Any], *, returncode: int, timed_out: bool,
        target_time: float, application_wall_seconds: float) -> dict[str, Any]:
    """Classify only application execution, independent of QoI quality."""
    reached = _as_float(parsed.get("time_reached"))
    explicit_time = parsed.get("explicit_time")
    application_reported_success = explicit_time.get("success") \
        if isinstance(explicit_time, Mapping) else None
    tolerance = max(1.0e-12, 1.0e-10 * abs(target_time))
    target_reached = reached is not None and reached >= target_time - tolerance
    if timed_out:
        return {
            "kind": "censored",
            "reason": "wall_timeout",
            "target_reached": target_reached,
            "time_to_solution_seconds": None,
            "time_to_failure_seconds": None,
            "censoring_wall_seconds": application_wall_seconds,
            "application_reported_success": application_reported_success,
            "target_time": target_time,
        }
    if returncode != 0:
        reason = parsed.get("failure_reason_from_log")
        return {
            "kind": "failure",
            "reason": reason or f"process_exit_{returncode}",
            "target_reached": target_reached,
            "time_to_solution_seconds": None,
            "time_to_failure_seconds": application_wall_seconds,
            "censoring_wall_seconds": None,
            "application_reported_success": application_reported_success,
            "target_time": target_time,
        }
    if application_reported_success is False:
        reason = parsed.get("failure_reason_from_log")
        return {
            "kind": "failure",
            "reason": reason or "application_reported_incomplete",
            "target_reached": target_reached,
            "time_to_solution_seconds": None,
            "time_to_failure_seconds": application_wall_seconds,
            "censoring_wall_seconds": None,
            "application_reported_success": False,
            "target_time": target_time,
        }
    if not target_reached:
        return {
            "kind": "failure",
            "reason": "target_horizon_not_observed",
            "target_reached": False,
            "time_to_solution_seconds": None,
            "time_to_failure_seconds": application_wall_seconds,
            "censoring_wall_seconds": None,
            "application_reported_success": application_reported_success,
            "target_time": target_time,
        }
    return {
        "kind": "success",
        "reason": "target_horizon_reached",
        "target_reached": True,
        "time_to_solution_seconds": application_wall_seconds,
        "time_to_failure_seconds": None,
        "censoring_wall_seconds": None,
        "application_reported_success": application_reported_success,
        "target_time": target_time,
    }


def parse_qoi_requirements(values: Sequence[str]) -> dict[str, dict[str, float]]:
    """Parse FIELD=REFERENCE:ABS_TOL caller-selected quality gates."""
    result: dict[str, dict[str, float]] = {}
    for value in values:
        if "=" not in value or ":" not in value.split("=", 1)[1]:
            raise ShockArchitectureError(
                "--qoi-requirement requires FIELD=REFERENCE:ABS_TOL"
            )
        field, specification = value.split("=", 1)
        reference_text, tolerance_text = specification.split(":", 1)
        if field not in QOI_FIELDS:
            raise ShockArchitectureError(
                f"unknown QoI requirement field {field!r}"
            )
        if field in result:
            raise ShockArchitectureError(
                f"--qoi-requirement repeats field {field}"
            )
        try:
            reference = float(reference_text)
            tolerance = float(tolerance_text)
        except ValueError as error:
            raise ShockArchitectureError(
                f"QoI requirement {field} must contain numeric values"
            ) from error
        if not math.isfinite(reference) or not math.isfinite(tolerance) or \
                tolerance < 0.0:
            raise ShockArchitectureError(
                f"QoI requirement {field} needs a finite reference and "
                "nonnegative absolute tolerance"
            )
        result[field] = {
            "reference": reference,
            "absolute_tolerance": tolerance,
        }
    return result


def evaluate_quality(
        *, qualification_mode: bool, method: str,
        execution: Mapping[str, Any],
        qoi_status: str, qoi_values: Mapping[str, float | None],
        requirements: Mapping[str, Mapping[str, float]],
        qoi_failure: str | None,
        closure_certificate: Mapping[str, Any],
        frozen_sigma_telemetry: Mapping[str, Any] | None = None,
        require_positive_q_sigma: bool = False,
        closure_certificate_max_relative_residual: float = 1.0e-7
        ) -> dict[str, Any]:
    """Evaluate caller-selected QoIs without changing execution outcome."""
    if not qualification_mode:
        status_by_measurement = {
            "not_requested": "not_requested",
            "success": "measured_not_gated",
            "failed": "measurement_failed",
            "censored": "measurement_censored",
            "not_run_application_nonsuccess": "not_evaluated",
        }
        return {
            "mode": "measurement-only",
            "status": status_by_measurement.get(qoi_status, "not_evaluated"),
            "qualified": False,
            "reason": qoi_failure,
            "requirements": {},
        }
    if execution["kind"] != "success":
        return {
            "mode": "qualification",
            "status": "not_evaluated",
            "qualified": False,
            "reason": "application_execution_not_successful",
            "requirements": {},
        }
    if qoi_status == "censored":
        return {
            "mode": "qualification",
            "status": "censored",
            "qualified": False,
            "reason": qoi_failure or "qoi_postprocessor_timeout",
            "requirements": {},
        }
    if qoi_status != "success":
        return {
            "mode": "qualification",
            "status": "failed",
            "qualified": False,
            "reason": qoi_failure or "qoi_postprocessor_failed",
            "requirements": {},
        }
    if method in EXPLICIT_METHODS:
        if not closure_certificate.get("reported"):
            return {
                "mode": "qualification",
                "status": "failed",
                "qualified": False,
                "reason": "missing_dg_accepted_state_closure_certificate",
                "requirements": {},
            }
        if closure_certificate.get("supported") is not True:
            return {
                "mode": "qualification",
                "status": "failed",
                "qualified": False,
                "reason": "unsupported_dg_closure_certificate",
                "requirements": {},
            }
        if closure_certificate.get("solved") is not True:
            return {
                "mode": "qualification",
                "status": "failed",
                "qualified": False,
                "reason": "unsolved_dg_accepted_state_closure",
                "requirements": {},
            }
        if closure_certificate.get("certified") is not True:
            return {
                "mode": "qualification",
                "status": "failed",
                "qualified": False,
                "reason": "uncertified_dg_accepted_state_closure",
                "requirements": {},
            }
        if (closure_certificate.get("run_success") is not True or
                closure_certificate.get("skipped_nonfinite") is True):
            return {
                "mode": "qualification",
                "status": "failed",
                "qualified": False,
                "reason": "unsuccessful_dg_closure_certificate_run",
                "requirements": {},
            }
        if (closure_certificate.get("scope") !=
                "final-committed-accepted-state" or
                closure_certificate.get("precision") != "fp64"):
            return {
                "mode": "qualification",
                "status": "failed",
                "qualified": False,
                "reason": "invalid_dg_closure_certificate_scope",
                "requirements": {},
            }
        certificate_time = _as_float(closure_certificate.get("time"))
        target_time = _as_float(execution.get("target_time"))
        time_tolerance = max(
            1.0e-12,
            1.0e-10 * max(1.0, abs(target_time or 0.0)),
        )
        if (certificate_time is None or target_time is None or
                abs(certificate_time - target_time) > time_tolerance):
            return {
                "mode": "qualification",
                "status": "failed",
                "qualified": False,
                "reason": "wrong_time_dg_closure_certificate",
                "requirements": {},
            }
        state_generation = _as_int(closure_certificate.get("state_generation"))
        if state_generation is None or state_generation < 0:
            return {
                "mode": "qualification",
                "status": "failed",
                "qualified": False,
                "reason": "invalid_dg_closure_state_generation",
                "requirements": {},
            }
        certificate_residual = _as_float(
            closure_certificate.get("residual_rel_rhs")
        )
        reported_limit = _as_float(closure_certificate.get("relative_limit"))
        external_limit = _finite_positive(
            closure_certificate_max_relative_residual,
            "closure certificate maximum relative residual",
        )
        if (certificate_residual is None or certificate_residual < 0.0 or
                reported_limit is None or reported_limit < 0.0 or
                certificate_residual > reported_limit or
                certificate_residual > external_limit):
            return {
                "mode": "qualification",
                "status": "failed",
                "qualified": False,
                "reason": "invalid_dg_closure_certificate_residual",
                "requirements": {},
            }
        if require_positive_q_sigma:
            telemetry = frozen_sigma_telemetry or {}
            if not telemetry.get("reported"):
                return {
                    "mode": "qualification",
                    "status": "failed",
                    "qualified": False,
                    "reason": "missing_exact_iid_q_sigma_telemetry",
                    "requirements": {},
                }
            sample_count = _as_int(telemetry.get("sample_count"))
            nonpositive_count = _as_int(
                telemetry.get("nonpositive_count")
            )
            nonfinite_count = _as_int(telemetry.get("nonfinite_count"))
            minimum_raw = _as_float(telemetry.get("minimum_raw_q_sigma"))
            minimum_normalized = _as_float(
                telemetry.get("minimum_q_sigma_over_gamma_p")
            )
            if (telemetry.get("scope") !=
                    "application-total-attempted-speed-work" or
                    telemetry.get("precision") != "fp64" or
                    telemetry.get("rejected_attempts_included") is not True or
                    sample_count is None or sample_count <= 0 or
                    nonpositive_count is None or nonpositive_count < 0 or
                    nonfinite_count is None or nonfinite_count < 0 or
                    nonpositive_count != 0 or nonfinite_count != 0 or
                    minimum_raw is None or minimum_raw <= 0.0 or
                    minimum_normalized is None or minimum_normalized <= 0.0):
                return {
                    "mode": "qualification",
                    "status": "failed",
                    "qualified": False,
                    "reason": "invalid_exact_iid_q_sigma_telemetry",
                    "requirements": {},
                }
    else:
        return {
            "mode": "qualification",
            "status": "failed",
            "qualified": False,
            "reason": "hdg_quality_contract_not_implemented",
            "requirements": {},
        }
    gates: dict[str, Any] = {}
    all_passed = True
    for field, requirement in requirements.items():
        measured = qoi_values.get(field)
        reference = float(requirement["reference"])
        tolerance = float(requirement["absolute_tolerance"])
        absolute_error = abs(measured - reference) \
            if measured is not None else None
        passed = absolute_error is not None and absolute_error <= tolerance
        all_passed = all_passed and passed
        gates[field] = {
            "measured": measured,
            "reference": reference,
            "absolute_tolerance": tolerance,
            "absolute_error": absolute_error,
            "passed": passed,
        }
    return {
        "mode": "qualification",
        "status": "passed" if all_passed else "failed",
        "qualified": all_passed,
        "reason": "all_qoi_gates_passed" if all_passed else
            "qoi_gate_failed_or_missing",
        "requirements": gates,
    }


def classify_selection(
        execution: Mapping[str, Any], quality: Mapping[str, Any],
        *, qualification_mode: bool,
        end_to_end_wall_seconds: float) -> dict[str, Any]:
    """Expose selection readiness without relabeling execution failures."""
    if execution["kind"] == "failure":
        kind = "execution_failure"
    elif execution["kind"] == "censored":
        kind = "execution_censored"
    elif not qualification_mode:
        kind = "execution_success_unqualified"
    elif quality["status"] == "passed":
        kind = "qualified_success"
    elif quality["status"] == "censored":
        kind = "quality_censored"
    else:
        kind = "quality_failure"
    return {
        "kind": kind,
        "time_to_qualified_solution_seconds":
            end_to_end_wall_seconds if kind == "qualified_success" else None,
    }


def parse_time_record(text: str) -> tuple[float, int, int] | None:
    for line in reversed(text.splitlines()):
        match = TIME_RECORD_RE.match(line.strip())
        if not match:
            continue
        wall = float(match.group(1))
        peak = int(match.group(2))
        status = int(match.group(3))
        if math.isfinite(wall) and wall >= 0.0 and peak >= 0:
            return wall, peak, status
    return None


def _read_tail(path: Path, limit: int) -> tuple[str, int, bool]:
    """Read a bounded whole log, or balanced head/tail when truncated."""
    size = path.stat().st_size if path.exists() else 0
    if size == 0:
        return "", 0, False
    with path.open("rb") as stream:
        if size <= limit:
            data = stream.read(limit)
        else:
            separator = b"\nDG_IGR_BOUNDED_LOG_GAP\n"
            payload = limit - len(separator)
            head_size = payload // 2
            tail_size = payload - head_size
            head = stream.read(head_size)
            stream.seek(size - tail_size)
            data = head + separator + stream.read(tail_size)
    return data.decode("utf-8", errors="replace"), size, size > limit


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(
        process_group: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while _process_group_exists(process_group):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False
        time.sleep(min(0.05, remaining))
    return True


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    """Terminate, reap, and verify the whole session process group."""
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass
    if not _wait_for_process_group_exit(process_group, 1.0):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired as error:
            raise ShockArchitectureError(
                f"unable to reap timed process-group leader {process.pid}"
            ) from error
    if not _wait_for_process_group_exit(process_group, 5.0):
        raise ShockArchitectureError(
            f"process group {process_group} survived SIGKILL"
        )


def run_timed_process(
        command: Sequence[str], *, time_program: str, work_directory: Path,
        environment: Mapping[str, str], timeout_seconds: float,
        max_log_bytes: int, label: str) -> TimedProcessResult:
    """Run one command through GNU time without a shell or retained raw log."""
    stdout_path = work_directory / f"{label}.stdout"
    stderr_path = work_directory / f"{label}.stderr"
    time_path = work_directory / f"{label}.time"
    timed_command = [
        time_program,
        "--quiet",
        "--output", str(time_path),
        "--format",
        "DG_IGR_EXTERNAL_TIME wall_seconds=%e peak_rss_kib=%M exit_status=%x",
        "--",
        *command,
    ]
    started = time.monotonic()
    try:
        with stdout_path.open("wb") as stdout_stream, \
                stderr_path.open("wb") as stderr_stream:
            process = subprocess.Popen(
                timed_command,
                cwd=str(work_directory),
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                start_new_session=True,
            )
            timed_out = False
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_group(process)
            except BaseException:
                _terminate_process_group(process)
                raise
            if not timed_out:
                try:
                    group_exited = _wait_for_process_group_exit(
                        process.pid, 0.25
                    )
                except BaseException:
                    _terminate_process_group(process)
                    raise
                if not group_exited:
                    _terminate_process_group(process)
                    raise ShockArchitectureError(
                        f"{label} command exited with surviving process-group "
                        "members"
                    )
    except OSError as error:
        raise ShockArchitectureError(
            f"unable to launch {label} process: {error}"
        ) from error
    monotonic_wall = time.monotonic() - started
    stdout_tail, stdout_bytes, stdout_truncated = _read_tail(
        stdout_path, max_log_bytes
    )
    stderr_tail, stderr_bytes, stderr_truncated = _read_tail(
        stderr_path, max_log_bytes
    )
    time_text = time_path.read_text(encoding="utf-8", errors="replace") \
        if time_path.exists() else ""
    record = parse_time_record(time_text)
    if record is None:
        wall_seconds = monotonic_wall
        wall_source = "monotonic_timeout" if timed_out else "monotonic_fallback"
        peak_rss_kib = None
        time_exit_status = None
    else:
        wall_seconds, peak_rss_kib, time_exit_status = record
        wall_source = "gnu_time"
    return TimedProcessResult(
        returncode=process.returncode,
        timed_out=timed_out,
        wall_seconds=wall_seconds,
        wall_source=wall_source,
        peak_rss_kib=peak_rss_kib,
        time_exit_status=time_exit_status,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )


def empty_qoi() -> dict[str, float | None]:
    return {name: None for name in QOI_FIELDS}


def parse_qoi_json(text: str) -> dict[str, float | None]:
    """Validate one bounded postprocessor JSON object against the QoI schema."""
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as error:
        raise ShockArchitectureError(
            f"QoI postprocessor did not emit valid JSON: {error}"
        ) from error
    if not isinstance(decoded, dict):
        raise ShockArchitectureError("QoI postprocessor JSON must be an object")
    unknown = sorted(set(decoded) - set(QOI_FIELDS))
    if unknown:
        raise ShockArchitectureError(
            "QoI postprocessor emitted unknown fields: " + ",".join(unknown)
        )
    result = empty_qoi()
    for name, value in decoded.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ShockArchitectureError(f"QoI field {name} must be numeric or null")
        converted = float(value)
        if not math.isfinite(converted):
            raise ShockArchitectureError(f"QoI field {name} must be finite")
        result[name] = converted
    return result


def qoi_command(
        executable: str, *, vtk_root: Path, method: str,
        target_time: float, mesh: str,
        extra_arguments: Sequence[str]) -> list[str]:
    return [
        executable,
        "--vtk-root", str(vtk_root),
        "--method", method,
        "--target-time", format(target_time, ".17g"),
        "--mesh", mesh,
        *extra_arguments,
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def capture_file_identity(path: Path, name: str) -> dict[str, Any]:
    """Hash a resolved input and reject a file that changes while hashing."""
    resolved = path.resolve()
    try:
        before = resolved.stat()
        digest = _sha256_file(resolved)
        after = resolved.stat()
    except OSError as error:
        raise ShockArchitectureError(
            f"unable to hash pinned {name}: {error}"
        ) from error
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field)
           for field in stable_fields):
        raise ShockArchitectureError(f"{name} changed while being hashed")
    return {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "device": after.st_dev,
        "inode": after.st_ino,
    }


def pin_qoi_input_files(
        extra_arguments: Sequence[str], *,
        qualification_mode: bool) -> tuple[tuple[str, ...],
                                           tuple[dict[str, Any], ...]]:
    """Normalize and hash every recognized file-valued QoI input.

    The bundled QoI contract currently has one external file input,
    ``--reference-profile``. Keep the option allowlist explicit so output
    paths such as ``--write-profile`` are never mistaken for immutable input
    evidence. Both ``--option=PATH`` and ``--option PATH`` forms are bound.
    """
    normalized = list(extra_arguments)
    pinned: list[dict[str, Any]] = []
    index = 0
    while index < len(normalized):
        token = normalized[index]
        supplied_option, separator, inline_value = token.partition("=")
        if supplied_option.startswith("--") and any(
                option.startswith(supplied_option)
                for option in QOI_HARNESS_OWNED_OPTIONS):
            raise ShockArchitectureError(
                f"--qoi-arg may not override harness-owned QoI option "
                f"{supplied_option}"
            )
        if supplied_option not in QOI_FILE_INPUT_OPTIONS:
            if supplied_option.startswith("--") and any(
                    option.startswith(supplied_option)
                    for option in QOI_FILE_INPUT_OPTIONS):
                raise ShockArchitectureError(
                    f"abbreviated QoI file input {supplied_option} is not "
                    "permitted; use the complete option name"
                )
            index += 1
            continue
        option = supplied_option

        value_index = index
        if separator:
            value = inline_value
        else:
            value_index = index + 1
            if value_index >= len(normalized) or \
                    normalized[value_index].startswith("-"):
                raise ShockArchitectureError(
                    f"QoI file input {option} requires a path argument"
                )
            value = normalized[value_index]
        if not value:
            raise ShockArchitectureError(
                f"QoI file input {option} requires a non-empty path"
            )

        try:
            resolved = Path(value).expanduser().resolve()
        except (OSError, RuntimeError) as error:
            context = "qualification QoI input" if qualification_mode \
                else "QoI input"
            raise ShockArchitectureError(
                f"unable to resolve {context} {option}: {error}"
            ) from error
        if not resolved.is_file():
            context = "qualification QoI input" if qualification_mode \
                else "QoI input"
            raise ShockArchitectureError(
                f"{context} {option} is missing or not a file: {resolved}"
            )
        identity = capture_file_identity(
            resolved, f"QoI input {option}"
        )
        if separator:
            normalized[index] = f"{option}={identity['path']}"
        else:
            normalized[index] = option
            normalized[value_index] = str(identity["path"])
        pinned.append({
            "option": option,
            "argument_index": index,
            "file": identity,
        })
        index = value_index + 1
    return tuple(normalized), tuple(pinned)


def verify_file_identity(expected: Mapping[str, Any], name: str) -> None:
    current = capture_file_identity(Path(str(expected["path"])), name)
    if current != dict(expected):
        raise ShockArchitectureError(
            f"{name} changed after the campaign inputs were pinned"
        )


def _resolve_executable(value: str, name: str) -> Path:
    candidate = shutil.which(str(Path(value).expanduser()))
    if candidate is None:
        raise ShockArchitectureError(f"{name} is not executable: {value}")
    path = Path(candidate).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ShockArchitectureError(f"{name} is not executable: {path}")
    return path


def _resolve_file(value: str, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ShockArchitectureError(f"{name} is not a file: {path}")
    return path


def _git(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ShockArchitectureError(
            f"unable to read Git identity from {root}: {error}"
        ) from error
    return result.stdout.strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ShockArchitectureError(
            f"unable to read Git identity from {root}: {error}"
        ) from error
    return result.stdout


def _source_worktree_sha256(root: Path) -> str:
    """Hash tracked changes and untracked, nonignored source content."""
    digest = hashlib.sha256()
    tracked_delta = _git_bytes(
        root, "diff", "--no-ext-diff", "--binary", "--full-index", "HEAD", "--"
    )
    digest.update(b"tracked-delta\0")
    digest.update(len(tracked_delta).to_bytes(8, "big"))
    digest.update(tracked_delta)
    untracked = _git_bytes(
        root, "ls-files", "--others", "--exclude-standard", "-z"
    )
    digest.update(b"untracked\0")
    for raw_path in (item for item in untracked.split(b"\0") if item):
        relative = os.fsdecode(raw_path)
        path = root / relative
        metadata = path.lstat()
        digest.update(len(raw_path).to_bytes(8, "big"))
        digest.update(raw_path)
        digest.update(metadata.st_mode.to_bytes(8, "big"))
        if path.is_symlink():
            payload = os.fsencode(os.readlink(path))
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        elif path.is_file():
            before = path.stat()
            digest.update(bytes.fromhex(_sha256_file(path)))
            after = path.stat()
            stable_fields = (
                "st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns"
            )
            if any(getattr(before, field) != getattr(after, field)
                   for field in stable_fields):
                raise ShockArchitectureError(
                    f"untracked source file changed while hashing: {relative}"
                )
        else:
            raise ShockArchitectureError(
                f"unsupported untracked source entry: {relative}"
            )
    return digest.hexdigest()


def git_identity(
        source_root: Path, *, expected_commit: str | None,
        expected_tree: str | None, allow_dirty: bool) -> dict[str, Any]:
    commit = _git(source_root, "rev-parse", "HEAD")
    tree = _git(source_root, "rev-parse", "HEAD^{tree}")
    status = _git(source_root, "status", "--porcelain=v1", "--untracked-files=all")
    origin = _git(source_root, "remote", "get-url", "origin")
    if expected_commit is not None and commit != expected_commit:
        raise ShockArchitectureError(
            f"Git commit {commit} does not match --expected-commit {expected_commit}"
        )
    if expected_tree is not None and tree != expected_tree:
        raise ShockArchitectureError(
            f"Git tree {tree} does not match --expected-tree {expected_tree}"
        )
    if status and not allow_dirty:
        raise ShockArchitectureError(
            "source tree is dirty; commit it or pass --allow-dirty-source"
        )
    worktree_sha256 = _source_worktree_sha256(source_root)
    confirm_commit = _git(source_root, "rev-parse", "HEAD")
    confirm_tree = _git(source_root, "rev-parse", "HEAD^{tree}")
    confirm_status = _git(
        source_root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if (commit, tree, status) != (
            confirm_commit, confirm_tree, confirm_status):
        raise ShockArchitectureError(
            "source state changed while the campaign identity was pinned"
        )
    identity = {
        "source_root": str(source_root),
        "origin": origin,
        "commit": commit,
        "tree": tree,
        "dirty": bool(status),
        "porcelain_status_sha256": _sha256_text(status),
        "worktree_content_sha256": worktree_sha256,
    }
    identity["source_identity_sha256"] = _sha256_text(json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ))
    return identity


def verify_git_identity(expected: Mapping[str, Any]) -> None:
    source_root = Path(str(expected["source_root"]))
    current = git_identity(
        source_root,
        expected_commit=str(expected["commit"]),
        expected_tree=str(expected["tree"]),
        allow_dirty=True,
    )
    if current != dict(expected):
        raise ShockArchitectureError(
            "source state changed after the campaign inputs were pinned"
        )


def parse_key_value_arguments(
        values: Sequence[str], name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ShockArchitectureError(f"{name} requires KEY=VALUE")
        key, item = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", key):
            raise ShockArchitectureError(f"{name} has invalid key: {key!r}")
        if not item:
            raise ShockArchitectureError(f"{name} value for {key} is empty")
        if key in result:
            raise ShockArchitectureError(f"{name} repeats key {key}")
        result[key] = item
    return result


def parse_method_float_overrides(
        values: Sequence[str], *, option_name: str, default: float,
        allow_zero: bool) -> dict[str, float]:
    """Parse METHOD=VALUE overrides for the explicit integrators."""
    result = {method: float(default) for method in EXPLICIT_METHODS}
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ShockArchitectureError(
                f"{option_name} requires METHOD=VALUE"
            )
        method, text = value.split("=", 1)
        if method not in EXPLICIT_METHODS:
            raise ShockArchitectureError(
                f"{option_name} method must be one of " +
                ",".join(EXPLICIT_METHODS)
            )
        if method in seen:
            raise ShockArchitectureError(
                f"{option_name} repeats method {method}"
            )
        try:
            converted = float(text)
        except ValueError as error:
            raise ShockArchitectureError(
                f"{option_name} value for {method} must be numeric"
            ) from error
        if not math.isfinite(converted) or converted < 0.0 or \
                (converted == 0.0 and not allow_zero):
            qualifier = "nonnegative" if allow_zero else "positive"
            raise ShockArchitectureError(
                f"{option_name} value for {method} must be finite and "
                f"{qualifier}"
            )
        result[method] = converted
        seen.add(method)
    return result


def _statistics(values: Sequence[float]) -> dict[str, float | int] | None:
    checked = [float(value) for value in values if math.isfinite(float(value))]
    if not checked:
        return None
    ordered = sorted(checked)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "maximum": ordered[-1],
        "range": ordered[-1] - ordered[0],
    }


def aggregate_runs(
        runs: Sequence[Mapping[str, Any]],
        methods: Sequence[str]) -> dict[str, Any]:
    """Aggregate execution, quality, failure, and censoring independently."""
    result: dict[str, Any] = {}
    for method in methods:
        selected = [run for run in runs if run["method"] == method]
        execution_kinds = [run["execution"]["kind"] for run in selected]
        execution_successes = [
            run for run in selected if run["execution"]["kind"] == "success"
        ]
        execution_failures = [
            run for run in selected if run["execution"]["kind"] == "failure"
        ]
        execution_censored = [
            run for run in selected if run["execution"]["kind"] == "censored"
        ]
        failed_times_reached = [
            float(run["progress"]["time_reached"])
            for run in execution_failures
            if run["progress"].get("time_reached") is not None
        ]
        censored_times_reached = [
            float(run["progress"]["time_reached"])
            for run in execution_censored
            if run["progress"].get("time_reached") is not None
        ]
        execution_success_walls = [
            float(run["execution"]["time_to_solution_seconds"])
            for run in execution_successes
        ]
        execution_failure_walls = [
            float(run["execution"]["time_to_failure_seconds"])
            for run in execution_failures
            if run["execution"].get("time_to_failure_seconds") is not None
        ]
        execution_censoring_walls = [
            float(run["execution"]["censoring_wall_seconds"])
            for run in execution_censored
            if run["execution"].get("censoring_wall_seconds") is not None
        ]
        qualified = [
            run for run in selected
            if run["selection"]["kind"] == "qualified_success"
        ]
        qualified_walls = [
            float(run["selection"]["time_to_qualified_solution_seconds"])
            for run in qualified
        ]
        quality_statuses = [run["quality"]["status"] for run in selected]
        selection_kinds = [run["selection"]["kind"] for run in selected]
        execution_reason_counts: dict[str, int] = {}
        quality_reason_counts: dict[str, int] = {}
        for run in selected:
            if run["execution"]["kind"] != "success":
                reason = str(run["execution"].get("reason") or "unspecified")
                execution_reason_counts[reason] = \
                    execution_reason_counts.get(reason, 0) + 1
            if run["quality"]["status"] in {
                    "failed", "censored", "measurement_failed",
                    "measurement_censored"}:
                reason = str(run["quality"].get("reason") or "unspecified")
                quality_reason_counts[reason] = \
                    quality_reason_counts.get(reason, 0) + 1
        measured_qoi_runs = [
            run for run in execution_successes
            if run["qoi"]["status"] == "success"
        ]
        app_walls = [
            float(run["process"]["application_wall_seconds"])
            for run in selected
        ]
        end_to_end_walls = [
            float(run["process"]["end_to_end_wall_seconds"])
            for run in selected
        ]
        peaks = [
            float(run["process"]["peak_rss_kib"])
            for run in selected
            if run["process"].get("peak_rss_kib") is not None
        ]
        qoi_aggregates = {}
        for field in QOI_FIELDS:
            values = [
                float(run["qoi"]["values"][field])
                for run in measured_qoi_runs
                if run["qoi"]["values"].get(field) is not None
            ]
            qoi_aggregates[field] = _statistics(values)
        result[method] = {
            "attempts": len(selected),
            "execution": {
                "successes": execution_kinds.count("success"),
                "failures": execution_kinds.count("failure"),
                "censored": execution_kinds.count("censored"),
                "all_repetitions_successful": bool(selected) and
                    len(execution_successes) == len(selected),
                "time_to_solution_seconds":
                    _statistics(execution_success_walls),
                "time_to_failure_seconds":
                    _statistics(execution_failure_walls),
                "censoring_wall_seconds":
                    _statistics(execution_censoring_walls),
                "maximum_failure_physical_time_reached":
                    max(failed_times_reached) if failed_times_reached else None,
                "maximum_censored_physical_time_reached":
                    max(censored_times_reached)
                    if censored_times_reached else None,
                "nonsuccess_reason_counts": execution_reason_counts,
            },
            "quality": {
                "qualified_successes": len(qualified),
                "failed": quality_statuses.count("failed"),
                "censored": quality_statuses.count("censored"),
                "not_evaluated": quality_statuses.count("not_evaluated"),
                "measurement_only":
                    quality_statuses.count("measured_not_gated"),
                "measurement_failed":
                    quality_statuses.count("measurement_failed"),
                "measurement_censored":
                    quality_statuses.count("measurement_censored"),
                "not_requested": quality_statuses.count("not_requested"),
                "status_counts": {
                    status: quality_statuses.count(status)
                    for status in sorted(set(quality_statuses))
                },
                "nonsuccess_reason_counts": quality_reason_counts,
                "time_to_qualified_solution_seconds":
                    _statistics(qualified_walls),
            },
            "selection": {
                "kind_counts": {
                    kind: selection_kinds.count(kind)
                    for kind in sorted(set(selection_kinds))
                },
            },
            "application_wall_seconds": _statistics(app_walls),
            "end_to_end_wall_seconds": _statistics(end_to_end_walls),
            "peak_rss_kib": _statistics(peaks),
            "qoi": qoi_aggregates,
        }
    return result


def _markdown_value(value: Any, precision: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return format(value, f".{precision}g")
    return str(value).replace("|", "\\|")


def render_markdown(report: Mapping[str, Any]) -> str:
    provenance = report["provenance"]
    controls = report["controls"]
    qoi_input_files = provenance.get("qoi_input_files", [])
    qoi_input_summary = ", ".join(
        f"`{item['option']}` = `{item['file']['path']}` "
        f"(SHA-256 `{item['file']['sha256']}`)"
        for item in qoi_input_files
    ) if qoi_input_files else "none"
    lines = [
        "# Shock architecture selection",
        "",
        f"Generated: `{report['created_utc']}`",
        "",
        f"Git: `{provenance['git']['commit']}` / tree "
        f"`{provenance['git']['tree']}`"
        f"{' (dirty)' if provenance['git']['dirty'] else ''}",
        "",
        f"Executable SHA-256: `{provenance['executable']['sha256']}`",
        "",
        f"Mesh SHA-256: `{provenance['mesh']['sha256']}`",
        "",
        f"Harness invocation SHA-256: "
        f"`{provenance.get('harness_invocation', {}).get('sha256', '—')}`",
        "",
        f"Pinned QoI input files: {qoi_input_summary}.",
        "",
        f"Target physical time: `{controls['final_time']}`; "
        f"C_alpha: `{controls['c_alpha']}`; h_eff policy: "
        f"`{controls['alpha_length_policy']}`.",
        "",
        "Execution success means a zero application exit code and observed "
        "arrival at the target horizon. Quality qualification is reported "
        "separately and requires every caller-selected QoI gate to pass. "
        "Timeout rows are right-censored; their wall time is not time to "
        "failure.",
        "",
        "## Method aggregates",
        "",
        "| Method | Runs | Exec success | Exec failure | Exec censored | "
        "Quality qualified | Quality failed | Quality censored | Median exec "
        "time to solution (s) | Median qualified time (s) | Peak RSS max "
        "(KiB) | Max failed physical time | Max censored physical time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in report["methods"]:
        item = report["aggregates"][method]
        execution = item["execution"]
        quality = item["quality"]
        timing = execution["time_to_solution_seconds"] or {}
        qualified_timing = quality["time_to_qualified_solution_seconds"] or {}
        peak = item["peak_rss_kib"] or {}
        lines.append(
            f"| {method} | {item['attempts']} | {execution['successes']} | "
            f"{execution['failures']} | {execution['censored']} | "
            f"{quality['qualified_successes']} | {quality['failed']} | "
            f"{quality['censored']} | "
            f"{_markdown_value(timing.get('median'))} | "
            f"{_markdown_value(qualified_timing.get('median'))} | "
            f"{_markdown_value(peak.get('maximum'), 10)} | "
            f"{_markdown_value(execution['maximum_failure_physical_time_reached'])} | "
            f"{_markdown_value(execution['maximum_censored_physical_time_reached'])} |"
        )

    lines.extend([
        "",
        "## Individual fresh processes",
        "",
        "| Run | Repetition | Position | Method | Execution | Quality | "
        "Selection | Physical time reached | Application wall (s) | "
        "End-to-end wall (s) | Peak RSS (KiB) | Execution reason | Quality reason |",
        "|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---|",
    ])
    for run in report["runs"]:
        lines.append(
            f"| {run['run_index']} | {run['repetition']} | "
            f"{run['position']} | {run['method']} | "
            f"{run['execution']['kind']} | {run['quality']['status']} | "
            f"{run['selection']['kind']} | "
            f"{_markdown_value(run['progress']['time_reached'])} | "
            f"{_markdown_value(run['process']['application_wall_seconds'])} | "
            f"{_markdown_value(run['process']['end_to_end_wall_seconds'])} | "
            f"{_markdown_value(run['process']['peak_rss_kib'], 10)} | "
            f"{_markdown_value(run['execution']['reason'])} | "
            f"{_markdown_value(run['quality'].get('reason'))} |"
        )
    lines.extend([
        "",
        "## Resolved regularization",
        "",
        "| Run | Method | h_eff policy | h_eff reference | Effective alpha "
        "min | Effective alpha max | HDG reported effective alpha |",
        "|---:|---|---|---:|---:|---:|---:|",
    ])
    for run in report["runs"]:
        regularization = run["progress"].get("regularization", {})
        hdg = run["progress"].get("hdg", {})
        policy = regularization.get("h_eff_policy") \
            or hdg.get("alpha_length_policy")
        h_eff = regularization.get("h_eff_reference") \
            if regularization.get("reported") else hdg.get("h_eff")
        lines.append(
            f"| {run['run_index']} | {run['method']} | "
            f"{_markdown_value(policy)} | {_markdown_value(h_eff)} | "
            f"{_markdown_value(regularization.get('alpha_effective_min'))} | "
            f"{_markdown_value(regularization.get('alpha_effective_max'))} | "
            f"{_markdown_value(hdg.get('alpha_effective'))} |"
        )
    lines.extend([
        "",
        "## Frozen-sigma speed-sample telemetry",
        "",
        "| Run | Method | Reported | Samples | Nonpositive q_sigma | "
        "Nonfinite q_sigma/ratio | Minimum q_sigma | Minimum "
        "q_sigma/(gamma p) | Scope | Precision | Rejected attempts included |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ])
    for run in report["runs"]:
        telemetry = run["progress"].get("frozen_sigma_telemetry", {})
        lines.append(
            f"| {run['run_index']} | {run['method']} | "
            f"{_markdown_value(telemetry.get('reported', False))} | "
            f"{_markdown_value(telemetry.get('sample_count'))} | "
            f"{_markdown_value(telemetry.get('nonpositive_count'))} | "
            f"{_markdown_value(telemetry.get('nonfinite_count'))} | "
            f"{_markdown_value(telemetry.get('minimum_raw_q_sigma'))} | "
            f"{_markdown_value(telemetry.get('minimum_q_sigma_over_gamma_p'))} | "
            f"{_markdown_value(telemetry.get('scope'))} | "
            f"{_markdown_value(telemetry.get('precision'))} | "
            f"{_markdown_value(telemetry.get('rejected_attempts_included'))} |"
        )
    lines.extend([
        "",
        "The exact IGR route reports an MPI-reduced FP64 aggregate over all "
        "speed evaluations performed by the application, including rejected "
        "attempts. It retains no per-event history. A positive reported "
        "minimum q_sigma establishes only the sampled frozen-sigma local "
        "flux condition, not hyperbolicity of the nonlocal reduced DAE.",
    ])
    lines.extend([
        "",
        "## DG limiter aggregates",
        "",
        "| Run | Method | Reported ranks | Calls | Active calls | Limited "
        "element-visits | Element-call opportunities | Limited-element "
        "fraction | Minimum theta | Limiter time (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for run in report["runs"]:
        limiter = run["progress"].get("limiter", {})
        lines.append(
            f"| {run['run_index']} | {run['method']} | "
            f"{_markdown_value(limiter.get('reported_ranks', 0))} | "
            f"{_markdown_value(limiter.get('calls'))} | "
            f"{_markdown_value(limiter.get('active_calls'))} | "
            f"{_markdown_value(limiter.get('limited_elements'))} | "
            f"{_markdown_value(limiter.get('element_call_opportunities'))} | "
            f"{_markdown_value(limiter.get('limited_element_fraction'))} | "
            f"{_markdown_value(limiter.get('minimum_theta'))} | "
            f"{_markdown_value(limiter.get('total_s'))} |"
        )
    lines.extend([
        "",
        "The limited-element fraction is `limited_elements / "
        "element_call_opportunities` over completed, post-warmup limiter "
        "calls. An element is counted once each call in which its density or "
        "pressure polynomial is blended; this is an element-call incidence "
        "ratio, not unique mesh coverage. Rejected-mean attempts commit no "
        "limiter update and remain represented by the explicit retry totals. "
        "Minimum theta is the smallest effective density-times-pressure "
        "contraction over those element-calls.",
        "",
        "## DG accepted-trajectory conservation balance",
        "",
        "| Run | Method | Balance reported | Supported | Domain measure | "
        "Maximum absolute defect | Maximum relative defect |",
        "|---:|---|---|---|---:|---:|---:|",
    ])
    for run in report["runs"]:
        conservation = run["progress"].get("conservation", {})
        lines.append(
            f"| {run['run_index']} | {run['method']} | "
            f"{_markdown_value(conservation.get('reported', False))} | "
            f"{_markdown_value(conservation.get('supported'))} | "
            f"{_markdown_value(conservation.get('domain_measure'))} | "
            f"{_markdown_value(conservation.get('maximum_absolute_error'))} | "
            f"{_markdown_value(conservation.get('maximum_relative_error'))} |"
        )
    lines.extend([
        "",
        "The defect is the final minus initial extensive Euler state plus "
        "the signed outward numerical boundary-flux integral accumulated "
        "only across committed SSPRK stages. Its componentwise relative "
        "scale is the absolute state change plus the stage-weighted absolute "
        "value of each globally integrated boundary-flux component. Spatial "
        "and rank cancellation therefore occurs before that absolute value. "
        + ("This opt-in diagnostic is included in these end-to-end samples."
           if controls.get("conservation_balance") else
           "This opt-in diagnostic is disabled in these samples; they cannot "
           "support a conservation-qualified claim."),
        "",
        "## DG accepted-state closure certification",
        "",
        "| Run | Method | Marker reported | Solved | Certification supported | "
        "Certified | Time | Scope | Precision | FP64 residual / RHS | "
        "Reported relative limit | Certificate run success |",
        "|---:|---|---|---|---|---|---:|---|---|---:|---:|---|",
    ])
    for run in report["runs"]:
        certificate = run["progress"].get("closure_certificate", {})
        lines.append(
            f"| {run['run_index']} | {run['method']} | "
            f"{_markdown_value(certificate.get('reported', False))} | "
            f"{_markdown_value(certificate.get('solved'))} | "
            f"{_markdown_value(certificate.get('supported'))} | "
            f"{_markdown_value(certificate.get('certified'))} | "
            f"{_markdown_value(certificate.get('time'))} | "
            f"{_markdown_value(certificate.get('scope'))} | "
            f"{_markdown_value(certificate.get('precision'))} | "
            f"{_markdown_value(certificate.get('residual_rel_rhs'))} | "
            f"{_markdown_value(certificate.get('relative_limit'))} | "
            f"{_markdown_value(certificate.get('run_success'))} |"
        )
    closure_policies = controls.get("closure_policy_by_method", {})
    if closure_policies:
        lines.extend([
            "",
            "## Closure architecture",
            "",
            "| Method | Architecture | Scalar representation | Reconstruction |",
            "|---|---|---|---|",
        ])
        for method in report["methods"]:
            policy = closure_policies[method]
            lines.append(
                f"| {method} | {_markdown_value(policy['architecture'])} | "
                f"{_markdown_value(policy['scalar_space'])} | "
                f"{_markdown_value(policy['reconstruction_cadence'])} |"
            )
    lines.extend([
        "",
        "## Measurement policy",
        "",
        "- Every row is a fresh process; schedule rows are cyclic rotations of "
        "the caller-specified method order.",
        "- GNU time encloses application startup, initialization, stepping, "
        "configured output, validation, teardown, and MPI launcher when used.",
        "- Peak RSS is host RSS reported by GNU time, not device-memory usage.",
        "- Application execution failure, application censoring, QoI failure, "
        "and QoI censoring are counted independently.",
        "- Raw logs, VTK fields, and profiler/event histories are not retained.",
        "- Nullable QoI fields mean no qualifying VTK postprocessor value was "
        "supplied; they are not zero measurements.",
        "",
    ])
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def publish_report(
        output_directory: Path, report: Mapping[str, Any],
        *, overwrite: bool) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / REPORT_JSON
    markdown_path = output_directory / REPORT_MARKDOWN
    existing = [path for path in (json_path, markdown_path) if path.exists()]
    if existing and not overwrite:
        raise ShockArchitectureError(
            "refusing to overwrite existing report: " +
            ",".join(str(path) for path in existing)
        )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write(json_path, encoded)
    _atomic_write(markdown_path, render_markdown(report))
    return json_path, markdown_path


def _controls_from_arguments(
        arguments: argparse.Namespace, methods: Sequence[str]) -> HarnessControls:
    final_time = _finite_positive(arguments.final_time, "--final-time")
    explicit_dt = _finite_positive(arguments.explicit_dt, "--explicit-dt")
    explicit_cfl = _finite_nonnegative(arguments.explicit_cfl, "--explicit-cfl")
    explicit_dt_by_method = parse_method_float_overrides(
        arguments.method_explicit_dt,
        option_name="--method-explicit-dt",
        default=explicit_dt,
        allow_zero=False,
    )
    explicit_cfl_by_method = parse_method_float_overrides(
        arguments.method_explicit_cfl,
        option_name="--method-explicit-cfl",
        default=explicit_cfl,
        allow_zero=True,
    )
    dt_min = _finite_positive(arguments.dt_min, "--dt-min")
    rodas_initial_dt = _finite_positive(
        arguments.rodas_initial_dt, "--rodas-initial-dt"
    )
    selected_initial_steps = [
        explicit_dt_by_method[method]
        for method in methods if method in EXPLICIT_METHODS
    ]
    if any(method not in EXPLICIT_METHODS for method in methods):
        selected_initial_steps.append(rodas_initial_dt)
    if dt_min > min(selected_initial_steps):
        raise ShockArchitectureError(
            "--dt-min may not exceed any selected method's initial timestep"
        )
    vtk_interval = arguments.vtk_save_interval
    if vtk_interval is None:
        vtk_interval = final_time
    vtk_interval = _finite_positive(vtk_interval, "--vtk-save-interval")
    return HarnessControls(
        device=arguments.device,
        order=arguments.order,
        serial_refine=arguments.serial_refine,
        parallel_refine=arguments.parallel_refine,
        final_time=final_time,
        c_alpha=_finite_positive(arguments.c_alpha, "--c-alpha"),
        alpha_length_policy=arguments.alpha_length_policy,
        dgsem_surface_flux=arguments.dgsem_surface_flux,
        igr_invariant_domain_flux=arguments.igr_invariant_domain_flux,
        igr_sigma_cfl=arguments.igr_sigma_cfl,
        igr_sigma_rusanov=arguments.igr_sigma_rusanov,
        conservation_balance=arguments.conservation_balance,
        zhang_shu_vtk_points=arguments.zhang_shu_vtk_points,
        closure_certificate_max_relative_residual=_finite_positive(
            arguments.closure_certificate_max_relative_residual,
            "--closure-certificate-max-relative-residual",
        ),
        explicit_dt_by_method=explicit_dt_by_method,
        explicit_cfl_by_method=explicit_cfl_by_method,
        dt_min=dt_min,
        igr_linear_rtol=_finite_positive(
            arguments.igr_linear_rtol, "--igr-linear-rtol"
        ),
        igr_linear_atol=_finite_nonnegative(
            arguments.igr_linear_atol, "--igr-linear-atol"
        ),
        igr_linear_max_iterations=arguments.igr_linear_max_iterations,
        rodas_initial_dt=rodas_initial_dt,
        rodas_time_atol=_finite_positive(
            arguments.rodas_time_atol, "--rodas-time-atol"
        ),
        rodas_time_rtol=_finite_positive(
            arguments.rodas_time_rtol, "--rodas-time-rtol"
        ),
        progress_steps=arguments.progress_steps,
        vtk=arguments.vtk or arguments.qoi_postprocessor is not None,
        vtk_save_interval=vtk_interval,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a rotated, fresh-process shock architecture matrix. The "
            "caller owns any GPU/MPI resource lease and supplies a built "
            "unified application. Application failures are recorded as data."
        )
    )
    parser.add_argument("--executable", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--methods", required=True,
        help="Comma-separated caller order; choices: " + ",".join(METHODS),
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--rotation-offset", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--order", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--serial-refine", type=int, default=0)
    parser.add_argument("--parallel-refine", type=int, default=0)
    parser.add_argument("--final-time", type=float, required=True)
    parser.add_argument("--c-alpha", type=float, default=1.0)
    parser.add_argument(
        "--alpha-length-policy", choices=("min", "average", "max", "local"),
        default="max",
    )
    parser.add_argument(
        "--dgsem-surface-flux",
        choices=("rusanov", "diffusive-rusanov", "es-winters", "es-lf-entropy"),
        default="rusanov",
        help="Typed DG/DGSEM explicit surface flux",
    )
    parser.add_argument(
        "--igr-invariant-domain-flux", action="store_true",
        help=(
            "Use the exact frozen-stage LF split threshold together with the "
            "frozen-sigma characteristic-modulus bound for both explicit CFL "
            "accounting and Rusanov face dissipation"
        ),
    )
    parser.add_argument(
        "--igr-sigma-cfl", action="store_true",
        help="Include the instantaneous IGR scalar scale in explicit CFL",
    )
    parser.add_argument(
        "--igr-sigma-rusanov", action="store_true",
        help="Include the instantaneous IGR scalar scale in Rusanov dissipation",
    )
    parser.add_argument(
        "--conservation-balance", action="store_true",
        help=(
            "Enable accepted-step physical-boundary conservation telemetry "
            "for explicit DG cards; this adds qualification overhead"
        ),
    )
    parser.add_argument(
        "--zhang-shu-vtk-points", action="store_true",
        help=(
            "Enforce Zhang-Shu admissibility at refined VTK output points "
            "for explicit DG cards (default off; HDG cards are unchanged)"
        ),
    )
    parser.add_argument(
        "--closure-certificate-max-relative-residual",
        type=float,
        default=1.0e-7,
        help=(
            "Independent FP64 ceiling applied in addition to the closure "
            "solver's reported accepted-state certificate limit"
        ),
    )
    parser.add_argument(
        "--explicit-dt", type=float, default=1.0e-4,
        help="Default explicit initial dt, or fixed dt when CFL is zero",
    )
    parser.add_argument(
        "--explicit-cfl", type=float, default=0.5,
        help="Default explicit CFL; zero selects the configured fixed dt",
    )
    parser.add_argument(
        "--method-explicit-dt", action="append", default=[],
        metavar="METHOD=DT",
        help="Per-explicit-method initial/fixed dt override",
    )
    parser.add_argument(
        "--method-explicit-cfl", action="append", default=[],
        metavar="METHOD=CFL",
        help="Per-explicit-method CFL override; zero selects its fixed dt",
    )
    parser.add_argument("--dt-min", type=float, default=1.0e-12)
    parser.add_argument("--igr-linear-rtol", type=float, default=1.0e-5)
    parser.add_argument("--igr-linear-atol", type=float, default=0.0)
    parser.add_argument(
        "--igr-linear-max-iterations", type=int, default=500,
    )
    parser.add_argument("--rodas-initial-dt", type=float, default=1.0e-4)
    parser.add_argument("--rodas-time-atol", type=float, default=1.0e-10)
    parser.add_argument("--rodas-time-rtol", type=float, default=1.0e-6)
    parser.add_argument("--progress-steps", type=int, default=50)
    parser.add_argument("--vtk", action="store_true")
    parser.add_argument("--vtk-save-interval", type=float)
    parser.add_argument(
        "--qoi-postprocessor",
        help=(
            "Optional executable receiving --vtk-root, --method, "
            "--target-time, and --mesh; stdout must be one QoI JSON object"
        ),
    )
    parser.add_argument(
        "--qoi-arg", action="append", default=[],
        help=(
            "Additional postprocessor token; use --qoi-arg=VALUE. Known "
            "file-valued inputs are normalized and byte-pinned."
        ),
    )
    parser.add_argument(
        "--qualification-mode", action="store_true",
        help="Require explicit QoI gates before reporting qualified success",
    )
    parser.add_argument(
        "--qoi-requirement", action="append", default=[],
        metavar="FIELD=REFERENCE:ABS_TOL",
        help="Caller-selected QoI gate; repeat for each required field",
    )
    parser.add_argument(
        "--app-arg", action="append", default=[],
        help="Unprotected application token; use --app-arg=--flag",
    )
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--qoi-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-log-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--time-program", default=DEFAULT_TIME_PROGRAM)
    parser.add_argument("--scratch-root")
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--mpi-launcher")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument(
        "--launcher-arg", action="append", default=[],
        help="Additional launcher token after rank selection",
    )
    parser.add_argument(
        "--hardware-field", action="append", default=[], metavar="KEY=VALUE",
    )
    parser.add_argument(
        "--resource-lease", default="caller-managed",
        help="Recorded lease identifier; this driver never acquires it",
    )
    parser.add_argument("--enable-limiter-timing", action="store_true")
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-tree")
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    integer_checks = (
        (arguments.serial_refine, 0, 20, "--serial-refine"),
        (arguments.parallel_refine, 0, 20, "--parallel-refine"),
        (arguments.igr_linear_max_iterations, 1, 100000,
         "--igr-linear-max-iterations"),
        (arguments.progress_steps, 1, 1000000000, "--progress-steps"),
        (arguments.max_log_bytes, 4096, 16 * 1024 * 1024,
         "--max-log-bytes"),
        (arguments.mpi_ranks, 1, 1024, "--mpi-ranks"),
    )
    for value, minimum, maximum, name in integer_checks:
        if value < minimum or value > maximum:
            raise ShockArchitectureError(
                f"{name} must be in [{minimum},{maximum}]"
            )
    _finite_positive(arguments.timeout_seconds, "--timeout-seconds")
    _finite_positive(arguments.qoi_timeout_seconds, "--qoi-timeout-seconds")
    if arguments.mpi_ranks > 1 and arguments.mpi_launcher is None:
        raise ShockArchitectureError("--mpi-ranks > 1 requires --mpi-launcher")
    if arguments.mpi_launcher is None and arguments.launcher_arg:
        raise ShockArchitectureError("--launcher-arg requires --mpi-launcher")
    if not arguments.mpi_numproc_flag:
        raise ShockArchitectureError("--mpi-numproc-flag must be non-empty")
    if arguments.igr_invariant_domain_flux and (
            arguments.igr_sigma_cfl or arguments.igr_sigma_rusanov):
        raise ShockArchitectureError(
            "--igr-invariant-domain-flux is a coupled policy and may not be "
            "combined with the independent sigma-speed heuristics"
        )
    if (arguments.igr_invariant_domain_flux and
            arguments.dgsem_surface_flux != "rusanov"):
        raise ShockArchitectureError(
            "--igr-invariant-domain-flux requires --dgsem-surface-flux "
            "rusanov"
        )


def run_campaign(arguments: argparse.Namespace) -> dict[str, Any]:
    _validate_arguments(arguments)
    methods = parse_methods(arguments.methods)
    schedule = rotated_schedule(
        methods, arguments.repetitions, arguments.rotation_offset
    )
    executable = _resolve_executable(arguments.executable, "application")
    mesh = _resolve_file(arguments.mesh, "mesh")
    time_program = _resolve_executable(arguments.time_program, "GNU time")
    qoi_executable = _resolve_executable(
        arguments.qoi_postprocessor, "QoI postprocessor"
    ) if arguments.qoi_postprocessor is not None else None
    qoi_requirements = parse_qoi_requirements(arguments.qoi_requirement)
    if arguments.qualification_mode and qoi_executable is None:
        raise ShockArchitectureError(
            "--qualification-mode requires --qoi-postprocessor"
        )
    if arguments.qualification_mode and not qoi_requirements:
        raise ShockArchitectureError(
            "--qualification-mode requires at least one --qoi-requirement"
        )
    if qoi_requirements and not arguments.qualification_mode:
        raise ShockArchitectureError(
            "--qoi-requirement requires --qualification-mode"
        )
    normalized_qoi_arguments, qoi_input_files = pin_qoi_input_files(
        arguments.qoi_arg,
        qualification_mode=arguments.qualification_mode,
    )
    launcher = _resolve_executable(
        arguments.mpi_launcher, "MPI launcher"
    ) if arguments.mpi_launcher is not None else None
    source_root = Path(arguments.source_root).expanduser().resolve()
    git = git_identity(
        source_root,
        expected_commit=arguments.expected_commit,
        expected_tree=arguments.expected_tree,
        allow_dirty=arguments.allow_dirty_source,
    )
    harness_path = Path(__file__).resolve()
    pinned_files: dict[str, dict[str, Any]] = {
        "executable": capture_file_identity(executable, "application executable"),
        "mesh": capture_file_identity(mesh, "mesh"),
        "harness": capture_file_identity(harness_path, "selection harness"),
        "time_program": capture_file_identity(time_program, "GNU time"),
    }
    if qoi_executable is not None:
        pinned_files["qoi_postprocessor"] = capture_file_identity(
            qoi_executable, "QoI postprocessor"
        )
    if launcher is not None:
        pinned_files["mpi_launcher"] = capture_file_identity(
            launcher, "MPI launcher"
        )
    controls = _controls_from_arguments(arguments, methods)
    extras = validate_extra_app_arguments(arguments.app_arg)
    hardware_fields = parse_key_value_arguments(
        arguments.hardware_field, "--hardware-field"
    )
    child_environment = dict(os.environ)
    child_environment.pop("DG_IGR_VTK_PREFIX_PATH", None)
    child_environment.pop("DG_IGR_LIMITER_TIMING", None)
    if arguments.enable_limiter_timing:
        child_environment["DG_IGR_LIMITER_TIMING"] = "1"
    selected_environment = {
        name: value for name, value in child_environment.items()
        if name.startswith("DG_IGR_") or name in {
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OMP_PROC_BIND",
            "OMP_PLACES",
        }
    }

    method_cards = {
        method: build_method_arguments(method, str(mesh), controls, extras)
        for method in methods
    }
    launcher_prefix = []
    if launcher is not None:
        launcher_prefix = [
            str(launcher), arguments.mpi_numproc_flag,
            str(arguments.mpi_ranks), *arguments.launcher_arg,
        ]

    runs: list[dict[str, Any]] = []
    scratch_parent = Path(arguments.scratch_root).expanduser().resolve() \
        if arguments.scratch_root else None
    if scratch_parent is not None and not scratch_parent.is_dir():
        raise ShockArchitectureError(
            f"--scratch-root is not a directory: {scratch_parent}"
        )
    with tempfile.TemporaryDirectory(
            prefix="dg-igr-shock-architecture-",
            dir=str(scratch_parent) if scratch_parent else None) as temporary:
        temporary_root = Path(temporary)
        run_index = 0
        for repetition, row in enumerate(schedule, start=1):
            for position, method in enumerate(row, start=1):
                run_index += 1
                run_directory = temporary_root / f"run-{run_index:03d}-{method}"
                run_directory.mkdir()
                vtk_root = run_directory / "vtk"
                run_environment = dict(child_environment)
                run_environment["DG_IGR_VTK_PREFIX_PATH"] = str(vtk_root)
                application_command = [
                    *launcher_prefix, str(executable), *method_cards[method]
                ]
                process = run_timed_process(
                    application_command,
                    time_program=str(time_program),
                    work_directory=run_directory,
                    environment=run_environment,
                    timeout_seconds=arguments.timeout_seconds,
                    max_log_bytes=arguments.max_log_bytes,
                    label="application",
                )
                combined_log = process.stdout_tail + "\n" + process.stderr_tail
                parsed = parse_application_log(combined_log, method)
                if process.returncode != 0 and \
                        parsed["failure_reason_from_log"] is None:
                    parsed["failure_reason_from_log"] = \
                        _last_diagnostic_excerpt(combined_log)
                execution = classify_execution(
                    parsed,
                    returncode=process.returncode,
                    timed_out=process.timed_out,
                    target_time=controls.final_time,
                    application_wall_seconds=process.wall_seconds,
                )

                qoi_values = empty_qoi()
                qoi_status = "not_requested"
                qoi_wall = 0.0
                qoi_peak: int | None = None
                qoi_failure: str | None = None
                qoi_process_command: list[str] | None = None
                qoi_process_record: dict[str, Any] = {
                    "returncode": None,
                    "timed_out": False,
                    "wall_source": None,
                    "captured_stdout_bytes": 0,
                    "captured_stderr_bytes": 0,
                    "stdout_tail_truncated": False,
                    "stderr_tail_truncated": False,
                }
                if qoi_executable is not None and \
                        execution["kind"] == "success":
                    qoi_process_command = qoi_command(
                        str(qoi_executable), vtk_root=vtk_root, method=method,
                        target_time=controls.final_time, mesh=str(mesh),
                        extra_arguments=normalized_qoi_arguments,
                    )
                    qoi_process = run_timed_process(
                        qoi_process_command,
                        time_program=str(time_program),
                        work_directory=run_directory,
                        environment=run_environment,
                        timeout_seconds=arguments.qoi_timeout_seconds,
                        max_log_bytes=arguments.max_log_bytes,
                        label="qoi",
                    )
                    qoi_wall = qoi_process.wall_seconds
                    qoi_peak = qoi_process.peak_rss_kib
                    qoi_process_record = {
                        "returncode": qoi_process.returncode,
                        "timed_out": qoi_process.timed_out,
                        "wall_source": qoi_process.wall_source,
                        "captured_stdout_bytes": qoi_process.stdout_bytes,
                        "captured_stderr_bytes": qoi_process.stderr_bytes,
                        "stdout_tail_truncated": qoi_process.stdout_truncated,
                        "stderr_tail_truncated": qoi_process.stderr_truncated,
                    }
                    if qoi_process.timed_out:
                        qoi_status = "censored"
                        qoi_failure = "qoi_postprocessor_timeout"
                    elif qoi_process.returncode != 0:
                        qoi_status = "failed"
                        qoi_failure = (
                            f"qoi_postprocessor_exit_{qoi_process.returncode}"
                        )
                    elif qoi_process.stdout_truncated:
                        qoi_status = "failed"
                        qoi_failure = "qoi_postprocessor_output_truncated"
                    else:
                        try:
                            qoi_values = parse_qoi_json(qoi_process.stdout_tail)
                            qoi_status = "success"
                        except ShockArchitectureError as error:
                            qoi_status = "failed"
                            qoi_failure = str(error)
                elif qoi_executable is not None:
                    qoi_status = "not_run_application_nonsuccess"

                conservation = parsed.get("conservation", {})
                if isinstance(conservation, Mapping) and \
                        conservation.get("reported"):
                    qoi_values = dict(qoi_values)
                    qoi_values["conservation_error"] = \
                        conservation.get("maximum_absolute_error")
                    qoi_values["boundary_balance_error"] = \
                        conservation.get("maximum_relative_error")

                end_to_end_wall = process.wall_seconds + qoi_wall
                quality = evaluate_quality(
                    qualification_mode=arguments.qualification_mode,
                    method=method,
                    execution=execution,
                    qoi_status=qoi_status,
                    qoi_values=qoi_values,
                    requirements=qoi_requirements,
                    qoi_failure=qoi_failure,
                    closure_certificate=parsed["closure_certificate"],
                    frozen_sigma_telemetry=parsed["frozen_sigma_telemetry"],
                    require_positive_q_sigma=(
                        controls.igr_invariant_domain_flux
                    ),
                    closure_certificate_max_relative_residual=(
                        controls.closure_certificate_max_relative_residual
                    ),
                )
                selection = classify_selection(
                    execution, quality,
                    qualification_mode=arguments.qualification_mode,
                    end_to_end_wall_seconds=end_to_end_wall,
                )
                peak_candidates = [
                    value for value in (process.peak_rss_kib, qoi_peak)
                    if value is not None
                ]
                command_json = json.dumps(
                    application_command, separators=(",", ":"),
                    ensure_ascii=False,
                )
                runs.append({
                    "run_index": run_index,
                    "repetition": repetition,
                    "position": position,
                    "method": method,
                    "command": application_command,
                    "command_sha256": _sha256_text(command_json),
                    "qoi_command": qoi_process_command,
                    "provenance": {
                        "qoi_input_files": list(qoi_input_files),
                        "typed_application_controls": {
                            "zhang_shu_vtk_points": (
                                controls.zhang_shu_vtk_points
                                if method in EXPLICIT_METHODS else None
                            ),
                        },
                    },
                    "process": {
                        "returncode": process.returncode,
                        "timed_out": process.timed_out,
                        "application_wall_seconds": process.wall_seconds,
                        "postprocess_wall_seconds": qoi_wall,
                        "end_to_end_wall_seconds": end_to_end_wall,
                        "wall_source": process.wall_source,
                        "peak_rss_kib": max(peak_candidates)
                            if peak_candidates else None,
                        "application_peak_rss_kib": process.peak_rss_kib,
                        "postprocess_peak_rss_kib": qoi_peak,
                        "gnu_time_exit_status": process.time_exit_status,
                        "captured_stdout_bytes": process.stdout_bytes,
                        "captured_stderr_bytes": process.stderr_bytes,
                        "stdout_tail_truncated": process.stdout_truncated,
                        "stderr_tail_truncated": process.stderr_truncated,
                    },
                    "progress": parsed,
                    "qoi": {
                        "status": qoi_status,
                        "failure_reason": qoi_failure,
                        "values": qoi_values,
                        "process": qoi_process_record,
                    },
                    "execution": execution,
                    "quality": quality,
                    "selection": selection,
                })
                print(
                    "SHOCK_ARCHITECTURE_RUN"
                    f" run={run_index} repetition={repetition}"
                    f" position={position} method={method}"
                    f" execution={execution['kind']}"
                    f" quality={quality['status']}"
                    f" selection={selection['kind']}"
                    f" time_reached={parsed.get('time_reached')}"
                    f" wall_seconds={end_to_end_wall:.9g}"
                    f" reason={execution['reason']}"
                )

    verify_git_identity(git)
    for name, identity in pinned_files.items():
        verify_file_identity(identity, name.replace("_", " "))
    for item in qoi_input_files:
        verify_file_identity(
            item["file"], f"QoI input {item['option']}"
        )

    controls_record = {
        **controls.__dict__,
        "explicit_dt_by_method": {
            method: controls.explicit_dt_by_method[method]
            for method in methods if method in EXPLICIT_METHODS
        },
        "explicit_cfl_by_method": {
            method: controls.explicit_cfl_by_method[method]
            for method in methods if method in EXPLICIT_METHODS
        },
        "instantaneous_igr_model": True,
        "limiter": "zhang-shu",
        "limiter_timing_enabled": arguments.enable_limiter_timing,
        "timeout_seconds": arguments.timeout_seconds,
        "qoi_timeout_seconds": arguments.qoi_timeout_seconds,
        "mpi_ranks": arguments.mpi_ranks,
        "resource_lease": arguments.resource_lease,
        "qualification_mode": arguments.qualification_mode,
        "qoi_requirements": qoi_requirements,
        "closure_policy_by_method": {
            method: method_closure_policy(method) for method in methods
        },
    }
    invocation_arguments = list(getattr(arguments, "_driver_argv", ()))
    invocation = [str(harness_path), *invocation_arguments]
    report = {
        "schema_version": SCHEMA_VERSION,
        "campaign": "shock-architecture-selection",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "methods": list(methods),
        "schedule": [list(row) for row in schedule],
        "controls": controls_record,
        "provenance": {
            "git": git,
            "executable": pinned_files["executable"],
            "mesh": pinned_files["mesh"],
            "harness": pinned_files["harness"],
            "harness_invocation": {
                "argv": invocation,
                "sha256": _sha256_text(json.dumps(
                    invocation, separators=(",", ":"), ensure_ascii=False
                )),
            },
            "time_program": pinned_files["time_program"],
            "qoi_postprocessor": pinned_files.get("qoi_postprocessor"),
            "qoi_input_files": list(qoi_input_files),
            "typed_application_controls": {
                "zhang_shu_vtk_points": {
                    "enabled": controls.zhang_shu_vtk_points,
                    "scope": "explicit-dg-only",
                },
            },
            "mpi_launcher": ({
                **pinned_files["mpi_launcher"],
                "numproc_flag": arguments.mpi_numproc_flag,
                "ranks": arguments.mpi_ranks,
                "arguments": list(arguments.launcher_arg),
            } if launcher is not None else None),
            "verified_unchanged_after_runs": True,
            "hardware": {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor() or None,
                "logical_cpu_count": os.cpu_count(),
                "caller_fields": hardware_fields,
            },
            "selected_environment": selected_environment,
        },
        "method_cards": method_cards,
        "runs": runs,
    }
    report["aggregates"] = aggregate_runs(runs, methods)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    arguments = _argument_parser().parse_args(raw_arguments)
    arguments._driver_argv = raw_arguments
    try:
        report = run_campaign(arguments)
        json_path, markdown_path = publish_report(
            Path(arguments.output_dir).expanduser().resolve(),
            report,
            overwrite=arguments.overwrite,
        )
    except ShockArchitectureError as error:
        print(f"shock architecture matrix failed: {error}", file=sys.stderr)
        return 1
    print(f"SHOCK_ARCHITECTURE_REPORT json={json_path} markdown={markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
