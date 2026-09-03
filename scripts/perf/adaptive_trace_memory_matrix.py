#!/usr/bin/env python3
"""Final memory and fresh-process controls for adaptive trace methods.

This test-only driver performs no build, GPU selection, persistent file
publication, or raw-log retention.  A tracked CTest invocation supplies the
already-built convergence executable while the caller owns the exclusive
visible GPU.  Memory sampling and unperturbed process timing are deliberately
separate modes.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


# Importing the tracked sibling must not create an untracked cache in an exact
# candidate tree during qualification.
sys.dont_write_bytecode = True

from gpu_memory_control import (  # noqa: E402
    MIB,
    MemoryControlError,
    aggregate_three_repeat_controls,
    run_gpu_memory_control,
)


MARKER = "ARCHITECTURE_SELECTION_POINT"
SKIP_RETURN_CODE = 77
POLL_MS = 50
PRE_ROWS = 10
POST_ROWS = 10
CONTROL_TIMEOUT_SECONDS = 600.0
ORDER = 3
ELEMENTS_PER_DIRECTION = 12
FINAL_TIME = 0.08
REGULARIZATION = "igr"
MEMORY_MODE = "memory"
FRESH_PROCESS_TIMING_MODE = "fresh-process-timing"
MODES = (MEMORY_MODE, FRESH_PROCESS_TIMING_MODE)

RODAS3 = "rodas3"
MONOLITHIC_BE = "monolithic_backward_euler"
PARTITIONED_BE = "partitioned_backward_euler"
METHODS = (RODAS3, MONOLITHIC_BE, PARTITIONED_BE)

EXACT_WORK_FIELDS = (
    "steps",
    "accepted_steps",
    "rejected_attempts",
    "controller_revision",
    "nonlinear_updates",
    "linear_iterations",
    "linear_solve_attempts",
    "trace_solves",
    "scalar_closure_solves",
    "scalar_closure_linear_iterations",
    "true_residual_tightenings",
    "complete_assemblies",
    "complete_condensations",
    "complete_recoveries",
    "common_shift_operator_builds",
    "common_shift_operator_reuses",
    "condensed_rhs_refreshes",
    "common_shift_trace_solves",
    "common_shift_preconditioner_setups",
    "common_shift_preconditioner_reuses",
    "native_trace_solve_calls",
    "hypre_trace_solve_calls",
    "patch_setups",
    "preconditioner_applications",
    "matrix_free_operator_applications",
    "output_publications",
)
MAXIMUM_QUALITY_FIELDS = (
    "maximum_final_residual",
    "maximum_true_relative_residual",
    "maximum_conservation_error",
    "maximum_conservation_relative_error",
    "maximum_embedded_temporal_error_norm",
    "maximum_final_algebraic_residual",
    "maximum_final_algebraic_correction_norm",
)
MINIMUM_QUALITY_FIELDS = (
    "minimum_density",
    "minimum_pressure",
)

ENV_BE_STEPS = "DG_IGR_ADAPTIVE_TRACE_MEMORY_BE_STEPS"
ENV_PBE_STEPS = "DG_IGR_ADAPTIVE_TRACE_MEMORY_PBE_STEPS"
ENV_RODAS_ATOL = "DG_IGR_ADAPTIVE_TRACE_MEMORY_RODAS_ATOL"
ENV_RODAS_RTOL = "DG_IGR_ADAPTIVE_TRACE_MEMORY_RODAS_RTOL"
ENV_RODAS_INITIAL_DT = "DG_IGR_ADAPTIVE_TRACE_MEMORY_RODAS_INITIAL_DT"
ENV_RODAS_DT_MIN = "DG_IGR_ADAPTIVE_TRACE_MEMORY_RODAS_DT_MIN"
ENV_GPU_UUID = "DG_IGR_ADAPTIVE_TRACE_MEMORY_GPU_UUID"
ENV_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
REQUIRED_ENVIRONMENT = (
    ENV_BE_STEPS,
    ENV_PBE_STEPS,
    ENV_RODAS_ATOL,
    ENV_RODAS_RTOL,
    ENV_RODAS_INITIAL_DT,
    ENV_RODAS_DT_MIN,
    ENV_GPU_UUID,
    ENV_VISIBLE_DEVICES,
)


class AdaptiveTraceMemoryError(RuntimeError):
    """A bounded driver, parser, or qualification failure."""


class MissingQualificationEnvironment(AdaptiveTraceMemoryError):
    """Required final-selection values are absent, so CTest should skip."""

    def __init__(self, names: Sequence[str]):
        self.names = tuple(names)
        super().__init__("missing required environment: " + ",".join(self.names))


def latin_square_schedule() -> tuple[tuple[str, str, str], ...]:
    """Return the exact three-repeat R/B/P rotation."""
    return (
        (RODAS3, MONOLITHIC_BE, PARTITIONED_BE),
        (MONOLITHIC_BE, PARTITIONED_BE, RODAS3),
        (PARTITIONED_BE, RODAS3, MONOLITHIC_BE),
    )


def _positive_int(text: str, name: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise AdaptiveTraceMemoryError(f"{name} must be an integer") from error
    if value < 1 or value > 512:
        raise AdaptiveTraceMemoryError(f"{name} must be in [1,512]")
    return value


def _positive_float(text: str, name: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise AdaptiveTraceMemoryError(f"{name} must be numeric") from error
    if not math.isfinite(value) or value <= 0.0:
        raise AdaptiveTraceMemoryError(f"{name} must be finite and positive")
    return value


def selected_controls_from_environment(
        environment: Mapping[str, str]) -> dict[str, Any]:
    """Read and validate the exact equal-error selections from the caller."""
    missing = [
        name for name in REQUIRED_ENVIRONMENT
        if not isinstance(environment.get(name), str)
        or not environment[name].strip()
    ]
    if missing:
        raise MissingQualificationEnvironment(missing)

    be_steps = _positive_int(environment[ENV_BE_STEPS].strip(), ENV_BE_STEPS)
    pbe_steps = _positive_int(
        environment[ENV_PBE_STEPS].strip(), ENV_PBE_STEPS
    )
    rodas_atol = _positive_float(
        environment[ENV_RODAS_ATOL].strip(), ENV_RODAS_ATOL
    )
    rodas_rtol = _positive_float(
        environment[ENV_RODAS_RTOL].strip(), ENV_RODAS_RTOL
    )
    rodas_initial_dt = _positive_float(
        environment[ENV_RODAS_INITIAL_DT].strip(), ENV_RODAS_INITIAL_DT
    )
    rodas_dt_min = _positive_float(
        environment[ENV_RODAS_DT_MIN].strip(), ENV_RODAS_DT_MIN
    )
    if rodas_initial_dt > FINAL_TIME:
        raise AdaptiveTraceMemoryError(
            f"{ENV_RODAS_INITIAL_DT} exceeds the fixed horizon"
        )
    if rodas_dt_min > rodas_initial_dt:
        raise AdaptiveTraceMemoryError(
            f"{ENV_RODAS_DT_MIN} exceeds the selected initial dt"
        )
    visible_devices = environment[ENV_VISIBLE_DEVICES].strip()
    if "," in visible_devices:
        raise AdaptiveTraceMemoryError(
            "CUDA_VISIBLE_DEVICES must expose exactly one authorized GPU"
        )
    return {
        "be_steps": be_steps,
        "pbe_steps": pbe_steps,
        "rodas_atol": rodas_atol,
        "rodas_rtol": rodas_rtol,
        "rodas_initial_dt": rodas_initial_dt,
        "rodas_dt_min": rodas_dt_min,
        "gpu_uuid": environment[ENV_GPU_UUID].strip(),
        "visible_devices": visible_devices,
    }


def parse_architecture_selection_point(
        stdout: str, expected_method: str) -> tuple[dict[str, str], int]:
    """Parse exactly one successful single-method point and its CUDA floor."""
    if expected_method not in METHODS:
        raise AdaptiveTraceMemoryError(
            f"unknown expected architecture method: {expected_method}"
        )
    points = []
    for raw_line in stdout.splitlines():
        fields = raw_line.strip().split()
        if fields and fields[0] == MARKER:
            points.append(fields)
    if len(points) != 1:
        raise AdaptiveTraceMemoryError(
            f"expected exactly one {MARKER}; observed {len(points)}"
        )

    parsed: dict[str, str] = {}
    for token in points[0][1:]:
        if "=" not in token:
            raise AdaptiveTraceMemoryError(
                f"architecture point token lacks '=': {token!r}"
            )
        key, value = token.split("=", 1)
        if not key or not value or key in parsed:
            raise AdaptiveTraceMemoryError(
                f"invalid or duplicate architecture point field: {token!r}"
            )
        parsed[key] = value
    required = ("method", "passed", "internal_cuda_used_floor_bytes")
    missing = [name for name in required if name not in parsed]
    if missing:
        raise AdaptiveTraceMemoryError(
            "architecture point omitted required fields: " + ",".join(missing)
        )
    if parsed["method"] != expected_method:
        raise AdaptiveTraceMemoryError(
            f"architecture point method={parsed['method']!r}; "
            f"expected {expected_method!r}"
        )
    if parsed["passed"] != "1":
        raise AdaptiveTraceMemoryError("architecture point did not pass")
    try:
        internal_floor = int(parsed["internal_cuda_used_floor_bytes"])
    except ValueError as error:
        raise AdaptiveTraceMemoryError(
            "internal CUDA used floor must be an integer"
        ) from error
    if internal_floor <= 0:
        raise AdaptiveTraceMemoryError(
            "internal CUDA used floor must be positive"
        )
    return parsed, internal_floor


def _point_int(fields: Mapping[str, str], name: str) -> int:
    if name not in fields:
        raise AdaptiveTraceMemoryError(
            f"architecture point omitted required field: {name}"
        )
    try:
        return int(fields[name])
    except ValueError as error:
        raise AdaptiveTraceMemoryError(
            f"architecture point {name} must be an integer"
        ) from error


def _point_float(fields: Mapping[str, str], name: str) -> float:
    if name not in fields:
        raise AdaptiveTraceMemoryError(
            f"architecture point omitted required field: {name}"
        )
    try:
        value = float(fields[name])
    except ValueError as error:
        raise AdaptiveTraceMemoryError(
            f"architecture point {name} must be numeric"
        ) from error
    if not math.isfinite(value):
        raise AdaptiveTraceMemoryError(
            f"architecture point {name} must be finite"
        )
    return value


def _require_point_flag(fields: Mapping[str, str], name: str,
                        expected: bool) -> None:
    value = _point_int(fields, name)
    if value != int(expected):
        raise AdaptiveTraceMemoryError(
            f"architecture point {name}={value}; expected {int(expected)}"
        )


def validate_selected_point(
        fields: Mapping[str, str], method: str, controls: Mapping[str, Any],
        *, expect_output: bool) -> float:
    """Bind one successful point to the selected controls and full gates."""
    if method not in METHODS:
        raise AdaptiveTraceMemoryError(f"unknown method: {method}")
    expected_text = {
        "method": method,
        "regularization": REGULARIZATION,
        "order": str(ORDER),
        "elements_per_direction": str(ELEMENTS_PER_DIRECTION),
    }
    for name, expected in expected_text.items():
        if fields.get(name) != expected:
            raise AdaptiveTraceMemoryError(
                f"architecture point {name}={fields.get(name)!r}; "
                f"expected {expected!r}"
            )
    if _point_float(fields, "final_time") != FINAL_TIME:
        raise AdaptiveTraceMemoryError(
            "architecture point final_time differs from selected horizon"
        )
    _require_point_flag(fields, "passed", True)
    _require_point_flag(fields, "temporal_reference_error", False)
    if _point_int(fields, "timing_repetitions") != 1:
        raise AdaptiveTraceMemoryError(
            "single-method point must contain exactly one timing repetition"
        )

    gate_names = (
        "finite_gate",
        "admissibility_gate",
        "recovery_gate",
        "true_residual_gate",
        "conservation_gate",
        "no_fallback_gate",
        "trace_policy_gate",
        "common_shift_gate",
    )
    for name in gate_names:
        _require_point_flag(fields, name, True)
    positive_names = (
        "measured_u4_global_l2_error",
        "sigma_coefficient_rms",
        "minimum_density",
        "minimum_pressure",
        "maximum_wave_speed",
        "measured_session_wall_seconds",
        "common_setup_inclusive_wall_seconds",
    )
    for name in positive_names:
        if _point_float(fields, name) <= 0.0:
            raise AdaptiveTraceMemoryError(
                f"architecture point {name} must be positive"
            )
    measured_wall = _point_float(fields, "measured_session_wall_seconds")
    internal_wall = _point_float(
        fields, "common_setup_inclusive_wall_seconds"
    )
    if internal_wall < measured_wall:
        raise AdaptiveTraceMemoryError(
            "common-setup-inclusive wall is below the session wall"
        )
    nonnegative_names = (
        *MAXIMUM_QUALITY_FIELDS,
    )
    for name in nonnegative_names:
        if _point_float(fields, name) < 0.0:
            raise AdaptiveTraceMemoryError(
                f"architecture point {name} must be nonnegative"
            )
    for name in EXACT_WORK_FIELDS:
        if _point_int(fields, name) < 0:
            raise AdaptiveTraceMemoryError(
                f"architecture point {name} must be nonnegative"
            )
    if _point_int(fields, "trace_solves") <= 0:
        raise AdaptiveTraceMemoryError(
            "architecture point trace_solves must be positive"
        )

    steps = _point_int(fields, "steps")
    accepted = _point_int(fields, "accepted_steps")
    rejected = _point_int(fields, "rejected_attempts")
    if method == RODAS3:
        expected_controls = {
            "time_atol": float(controls["rodas_atol"]),
            "time_rtol": float(controls["rodas_rtol"]),
            "adaptive_initial_dt": float(controls["rodas_initial_dt"]),
            "adaptive_minimum_dt": float(controls["rodas_dt_min"]),
            "dt": float(controls["rodas_initial_dt"]),
        }
        for name, expected in expected_controls.items():
            if _point_float(fields, name) != expected:
                raise AdaptiveTraceMemoryError(
                    f"architecture point {name} differs from selected control"
                )
        revision = _point_int(fields, "controller_revision")
        _require_point_flag(fields, "adaptive", True)
        _require_point_flag(fields, "prepared_assessment_authority", True)
        if steps <= 0 or accepted != steps or rejected <= 0 or \
                revision != accepted + rejected:
            raise AdaptiveTraceMemoryError(
                "adaptive point violates accepted/rejected revision accounting"
            )
    else:
        key = "be_steps" if method == MONOLITHIC_BE else "pbe_steps"
        expected_steps = int(controls[key])
        _require_point_flag(fields, "adaptive", False)
        if steps != expected_steps or accepted != expected_steps or rejected != 0:
            raise AdaptiveTraceMemoryError(
                "fixed-step point differs from selected step control"
            )
        if _point_float(fields, "dt") != FINAL_TIME / expected_steps:
            raise AdaptiveTraceMemoryError(
                "fixed-step point dt differs from the selected horizon/steps"
            )

    expected_publications = 3 if expect_output else 0
    if _point_int(fields, "output_publications") != expected_publications:
        raise AdaptiveTraceMemoryError(
            "architecture point output publication count is not selected"
        )
    _require_point_flag(fields, "midpoint_output_landed", expect_output)
    _require_point_flag(fields, "final_output_landed", expect_output)
    return internal_wall


def benchmark_command(
        executable: str, method: str, controls: Mapping[str, Any],
        output_root: str | None = None) -> list[str]:
    """Build one exact selected-method convergence command."""
    if method not in METHODS:
        raise AdaptiveTraceMemoryError(f"unknown method: {method}")
    if output_root is not None and not output_root:
        raise AdaptiveTraceMemoryError("output root must be non-empty")
    command = [
        executable,
        "--order", str(ORDER),
        "--regularization", REGULARIZATION,
        "--architecture-selection-screen",
        "--screen-elements", str(ELEMENTS_PER_DIRECTION),
        "--screen-final-time", format(FINAL_TIME, ".17g"),
        "--screen-memory-method", method,
    ]
    if method == RODAS3:
        command.extend([
            "--adaptive-rodas3-screen",
            "--screen-time-atol", format(float(controls["rodas_atol"]), ".17g"),
            "--screen-time-rtol", format(float(controls["rodas_rtol"]), ".17g"),
            "--screen-initial-dt",
            format(float(controls["rodas_initial_dt"]), ".17g"),
            "--screen-dt-min", format(float(controls["rodas_dt_min"]), ".17g"),
        ])
    else:
        steps_key = "be_steps" if method == MONOLITHIC_BE else "pbe_steps"
        command.extend(["--screen-memory-steps", str(int(controls[steps_key]))])
    if output_root is not None:
        command.extend(["--screen-output-root", output_root])
    return command


def _resolve_program(value: str, label: str) -> str:
    expanded = str(Path(value).expanduser())
    resolved = shutil.which(expanded)
    if resolved is None:
        raise AdaptiveTraceMemoryError(f"{label} is not executable: {value}")
    path = Path(resolved).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise AdaptiveTraceMemoryError(f"{label} is not executable: {path}")
    return str(path)


def _minimal_control(evidence: Mapping[str, Any]) -> dict[str, Any]:
    measurement = evidence.get("measurement")
    if evidence.get("status") != "success" or not isinstance(
            measurement, Mapping):
        raise AdaptiveTraceMemoryError("memory control omitted qualified evidence")
    required = (
        "incremental_peak_mib",
        "absolute_peak_mib",
        "absolute_peak_bytes",
        "headroom_mib",
    )
    if any(name not in measurement for name in required):
        raise AdaptiveTraceMemoryError(
            "memory control omitted an aggregate measurement"
        )
    return {
        "status": "success",
        "measurement": {name: measurement[name] for name in required},
    }


def run_matrix(
        executable: str, nvidia_smi: str, controls: Mapping[str, Any],
        environment: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Run nine fresh controls and return only per-method aggregates."""
    child_environment = dict(environment)
    child_environment["DG_IGR_RUN_ARCHITECTURE_SELECTION_SCREEN"] = "1"
    repeats: dict[str, list[dict[str, Any]]] = {
        method: [] for method in METHODS
    }
    floors: dict[str, list[int]] = {method: [] for method in METHODS}

    for rotation in latin_square_schedule():
        for method in rotation:
            command = benchmark_command(executable, method, controls)
            try:
                evidence = run_gpu_memory_control(
                    command,
                    nvidia_smi=nvidia_smi,
                    selected_gpu_uuid=str(controls["gpu_uuid"]),
                    poll_ms=POLL_MS,
                    timeout_seconds=CONTROL_TIMEOUT_SECONDS,
                    environment=child_environment,
                    pre_rows=PRE_ROWS,
                    post_rows=POST_ROWS,
                )
            except MemoryControlError as error:
                raise AdaptiveTraceMemoryError(
                    f"{method} whole-device control failed: {error}"
                ) from error
            benchmark = evidence.get("benchmark")
            if not isinstance(benchmark, Mapping) or not isinstance(
                    benchmark.get("stdout"), str):
                raise AdaptiveTraceMemoryError(
                    f"{method} control omitted benchmark stdout"
                )
            fields, internal_floor = parse_architecture_selection_point(
                benchmark["stdout"], method
            )
            validate_selected_point(
                fields, method, controls, expect_output=False
            )
            repeats[method].append(_minimal_control(evidence))
            floors[method].append(internal_floor)

    aggregates: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        try:
            aggregate = aggregate_three_repeat_controls(
                repeats[method], floors[method]
            )
        except MemoryControlError as error:
            raise AdaptiveTraceMemoryError(
                f"{method} repeat aggregation failed: {error}"
            ) from error
        aggregates[method] = {
            "external": aggregate,
            "maximum_internal_cuda_used_floor_bytes": max(floors[method]),
        }
    return aggregates


def _run_fresh_process(
        command: Sequence[str], *, timeout_seconds: float,
        environment: Mapping[str, str]) -> dict[str, Any]:
    """Run one unprofiled child and retain only its bounded process result."""
    started_ns = time.monotonic_ns()
    try:
        completed = subprocess.run(
            list(command),
            env=dict(environment),
            text=True,
            encoding="utf-8",
            errors="strict",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
            start_new_session=True,
        )
    except subprocess.TimeoutExpired as error:
        raise AdaptiveTraceMemoryError(
            "fresh-process timing child exceeded its timeout"
        ) from error
    except OSError as error:
        raise AdaptiveTraceMemoryError(
            f"unable to launch fresh-process timing child: {error}"
        ) from error
    finished_ns = time.monotonic_ns()
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": (finished_ns - started_ns) / 1.0e9,
    }


def _bounded_failure_excerpt(value: Any, limit: int = 512) -> str:
    """Flatten one diagnostic stream without retaining an unbounded log."""
    if not isinstance(value, str):
        return "<missing>"
    flattened = " ".join(value.split())
    return flattened[:limit] if flattened else "<empty>"


def _three_repeat_statistics(values: Sequence[float], name: str,
                             *, allow_zero: bool = False) -> dict[str, float]:
    if len(values) != 3:
        raise AdaptiveTraceMemoryError(
            f"{name} requires exactly three repetitions"
        )
    checked = [float(value) for value in values]
    if any(not math.isfinite(value) or
           (value < 0.0 if allow_zero else value <= 0.0)
           for value in checked):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise AdaptiveTraceMemoryError(
            f"{name} repetitions must be finite and {qualifier}"
        )
    ordered = sorted(checked)
    return {
        "minimum": ordered[0],
        "median": ordered[1],
        "maximum": ordered[2],
        "range": ordered[2] - ordered[0],
    }


def run_fresh_process_timing_matrix(
        executable: str, controls: Mapping[str, Any],
        environment: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Run three output-inclusive fresh processes per selected method."""
    child_environment = dict(environment)
    child_environment["DG_IGR_RUN_ARCHITECTURE_SELECTION_SCREEN"] = "1"
    process_walls: dict[str, list[float]] = {
        method: [] for method in METHODS
    }
    internal_walls: dict[str, list[float]] = {
        method: [] for method in METHODS
    }
    process_envelopes: dict[str, list[float]] = {
        method: [] for method in METHODS
    }
    work_signatures: dict[str, list[tuple[int, ...]]] = {
        method: [] for method in METHODS
    }
    u4_metrics: dict[str, list[float]] = {
        method: [] for method in METHODS
    }
    maximum_quality_metrics: dict[str, dict[str, list[float]]] = {
        method: {name: [] for name in MAXIMUM_QUALITY_FIELDS}
        for method in METHODS
    }
    minimum_quality_metrics: dict[str, dict[str, list[float]]] = {
        method: {name: [] for name in MINIMUM_QUALITY_FIELDS}
        for method in METHODS
    }

    with tempfile.TemporaryDirectory(
            prefix="dg-igr-adaptive-trace-timing-") as temporary_root:
        for repetition, rotation in enumerate(latin_square_schedule()):
            for method in rotation:
                output_root = str(
                    Path(temporary_root) /
                    f"repeat-{repetition + 1}-{method}"
                )
                command = benchmark_command(
                    executable, method, controls, output_root
                )
                try:
                    evidence = _run_fresh_process(
                        command,
                        timeout_seconds=CONTROL_TIMEOUT_SECONDS,
                        environment=child_environment,
                    )
                except AdaptiveTraceMemoryError as error:
                    raise AdaptiveTraceMemoryError(
                        f"{method} fresh-process timing failed: {error}"
                    ) from error
                if evidence.get("returncode") != 0:
                    raise AdaptiveTraceMemoryError(
                        f"{method} fresh-process timing exited nonzero: "
                        "stderr=" + _bounded_failure_excerpt(
                            evidence.get("stderr")
                        )
                    )
                stdout = evidence.get("stdout")
                if not isinstance(stdout, str):
                    raise AdaptiveTraceMemoryError(
                        f"{method} fresh-process timing omitted stdout"
                    )
                stderr = evidence.get("stderr")
                if not isinstance(stderr, str) or stderr:
                    raise AdaptiveTraceMemoryError(
                        f"{method} fresh-process timing produced non-empty "
                        "stderr: " + _bounded_failure_excerpt(stderr)
                    )
                fields, _ = parse_architecture_selection_point(stdout, method)
                internal_wall = validate_selected_point(
                    fields, method, controls, expect_output=True
                )
                try:
                    process_wall = float(evidence.get("wall_seconds", math.nan))
                except (TypeError, ValueError) as error:
                    raise AdaptiveTraceMemoryError(
                        f"{method} fresh-process wall must be numeric"
                    ) from error
                if not math.isfinite(process_wall) or process_wall <= 0.0:
                    raise AdaptiveTraceMemoryError(
                        f"{method} fresh-process wall must be finite and positive"
                    )
                envelope = process_wall - internal_wall
                if envelope < 0.0:
                    raise AdaptiveTraceMemoryError(
                        f"{method} process wall is below its internal wall"
                    )
                process_walls[method].append(process_wall)
                internal_walls[method].append(internal_wall)
                process_envelopes[method].append(envelope)
                work_signatures[method].append(tuple(
                    _point_int(fields, name) for name in EXACT_WORK_FIELDS
                ))
                u4_metrics[method].append(
                    _point_float(fields, "measured_u4_global_l2_error")
                )
                for name in MAXIMUM_QUALITY_FIELDS:
                    maximum_quality_metrics[method][name].append(
                        _point_float(fields, name)
                    )
                for name in MINIMUM_QUALITY_FIELDS:
                    minimum_quality_metrics[method][name].append(
                        _point_float(fields, name)
                    )

    for method in METHODS:
        if len(set(work_signatures[method])) != 1:
            raise AdaptiveTraceMemoryError(
                f"{method} work signature changed across timing repetitions"
            )
        reference_u4 = u4_metrics[method][0]
        for value in u4_metrics[method][1:]:
            scale = max(reference_u4, value)
            if abs(reference_u4 - value) > 1.0e-13 + 1.0e-9 * scale:
                raise AdaptiveTraceMemoryError(
                    f"{method} U4 metric changed across timing repetitions"
                )

    return {
        method: {
            "repeat_count": 3,
            "work_signature": dict(zip(
                EXACT_WORK_FIELDS, work_signatures[method][0]
            )),
            "u4_metric": _three_repeat_statistics(
                u4_metrics[method], f"{method} U4 metric"
            ),
            "maximum_quality_metrics": {
                name: _three_repeat_statistics(
                    maximum_quality_metrics[method][name],
                    f"{method} {name}", allow_zero=True,
                )
                for name in MAXIMUM_QUALITY_FIELDS
            },
            "minimum_quality_metrics": {
                name: _three_repeat_statistics(
                    minimum_quality_metrics[method][name],
                    f"{method} {name}",
                )
                for name in MINIMUM_QUALITY_FIELDS
            },
            "process_wall_seconds": _three_repeat_statistics(
                process_walls[method], f"{method} process wall"
            ),
            "internal_wall_seconds": _three_repeat_statistics(
                internal_walls[method], f"{method} internal wall"
            ),
            "process_envelope_seconds": _three_repeat_statistics(
                process_envelopes[method], f"{method} process envelope",
                allow_zero=True,
            ),
        }
        for method in METHODS
    }


def _bytes_from_mib(value: Any) -> int:
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise AdaptiveTraceMemoryError("aggregate MiB value is invalid")
    return int(round(converted * MIB))


def print_aggregates(aggregates: Mapping[str, Mapping[str, Any]]) -> None:
    """Print only compact, aggregate high-water records."""
    for method in METHODS:
        item = aggregates[method]
        external = item["external"]
        incremental = external["incremental_peak_mib"]
        print(
            "ADAPTIVE_TRACE_MEMORY_HIGH_WATER"
            f" method={method}"
            f" repeat_count={external['repeat_count']}"
            " absolute_peak_bytes="
            f"{_bytes_from_mib(external['conservative_absolute_peak_mib'])}"
            " incremental_peak_bytes="
            f"{_bytes_from_mib(external['conservative_incremental_peak_mib'])}"
            " minimum_headroom_bytes="
            f"{_bytes_from_mib(external['minimum_headroom_mib'])}"
            " incremental_peak_range_bytes="
            f"{_bytes_from_mib(incremental['range'])}"
            " internal_cuda_used_floor_bytes="
            f"{item['maximum_internal_cuda_used_floor_bytes']}"
            " qualified=1"
        )


def print_fresh_process_timing_aggregates(
        aggregates: Mapping[str, Mapping[str, Any]],
        controls: Mapping[str, Any]) -> None:
    """Print compact process-wall summaries and an uncertainty-aware result."""
    print(
        "ADAPTIVE_TRACE_FRESH_PROCESS_CONTROLS"
        f" regularization={REGULARIZATION}"
        f" order={ORDER}"
        f" elements_per_direction={ELEMENTS_PER_DIRECTION}"
        f" final_time={FINAL_TIME:.17g}"
        f" monolithic_steps={int(controls['be_steps'])}"
        f" partitioned_steps={int(controls['pbe_steps'])}"
        f" rodas3_time_atol={float(controls['rodas_atol']):.17g}"
        f" rodas3_time_rtol={float(controls['rodas_rtol']):.17g}"
        f" rodas3_initial_dt={float(controls['rodas_initial_dt']):.17g}"
        f" rodas3_minimum_dt={float(controls['rodas_dt_min']):.17g}"
        f" requested_gpu_uuid={controls['gpu_uuid']}"
        f" cuda_visible_devices={controls['visible_devices']}"
        " timing_repetitions=3"
        " output_publications_per_process=3"
    )
    for method in METHODS:
        item = aggregates[method]
        work = item["work_signature"]
        u4 = item["u4_metric"]
        maximum_quality = item["maximum_quality_metrics"]
        minimum_quality = item["minimum_quality_metrics"]
        process = item["process_wall_seconds"]
        internal = item["internal_wall_seconds"]
        envelope = item["process_envelope_seconds"]
        work_text = "".join(
            f" {name}={work[name]}" for name in EXACT_WORK_FIELDS
        )
        maximum_quality_text = "".join(
            f" {name}_max={maximum_quality[name]['maximum']:.17g}"
            f" {name}_range={maximum_quality[name]['range']:.17g}"
            for name in MAXIMUM_QUALITY_FIELDS
        )
        minimum_quality_text = "".join(
            f" {name}_min={minimum_quality[name]['minimum']:.17g}"
            f" {name}_range={minimum_quality[name]['range']:.17g}"
            for name in MINIMUM_QUALITY_FIELDS
        )
        print(
            "ADAPTIVE_TRACE_FRESH_PROCESS_WALL"
            f" method={method}"
            f" repeat_count={item['repeat_count']}"
            f"{work_text}"
            f" u4_metric_median={u4['median']:.17g}"
            f" u4_metric_range={u4['range']:.17g}"
            f"{maximum_quality_text}"
            f"{minimum_quality_text}"
            f" minimum_seconds={process['minimum']:.17g}"
            f" median_seconds={process['median']:.17g}"
            f" maximum_seconds={process['maximum']:.17g}"
            f" range_seconds={process['range']:.17g}"
            f" internal_median_seconds={internal['median']:.17g}"
            f" process_envelope_median_seconds={envelope['median']:.17g}"
            " process_startup_included=1"
            " process_teardown_included=1"
            " output_included=1"
            " nvidia_smi_polling=0"
            " selected_controls_validated=1"
            " numerical_gates_validated=1"
            " qualified=1"
        )

    rodas = aggregates[RODAS3]["process_wall_seconds"]
    monolithic = aggregates[MONOLITHIC_BE]["process_wall_seconds"]
    partitioned = aggregates[PARTITIONED_BE]["process_wall_seconds"]
    median_vs_monolithic = monolithic["median"] / rodas["median"]
    median_vs_partitioned = partitioned["median"] / rodas["median"]
    conservative_vs_monolithic = monolithic["minimum"] / rodas["maximum"]
    conservative_vs_partitioned = partitioned["minimum"] / rodas["maximum"]
    separated = conservative_vs_monolithic > 1.0 and \
        conservative_vs_partitioned > 1.0
    if not separated:
        raise AdaptiveTraceMemoryError(
            "RODAS3 fresh-process timing is not separated beyond the "
            "observed monolithic and partitioned ranges"
        )
    print(
        "ADAPTIVE_TRACE_FRESH_PROCESS_SELECTION"
        f" rodas3_speedup_vs_monolithic_median="
        f"{median_vs_monolithic:.17g}"
        f" rodas3_speedup_vs_partitioned_median="
        f"{median_vs_partitioned:.17g}"
        f" rodas3_speedup_vs_monolithic_conservative="
        f"{conservative_vs_monolithic:.17g}"
        f" rodas3_speedup_vs_partitioned_conservative="
        f"{conservative_vs_partitioned:.17g}"
        f" separated_beyond_observed_ranges={int(separated)}"
        " measurement_qualified=1"
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a selected adaptive/BE/PBE tensor CUDA final matrix; "
            "the caller must own the one visible GPU"
        )
    )
    parser.add_argument("--executable", required=True)
    parser.add_argument("--mode", choices=MODES, default=MEMORY_MODE)
    parser.add_argument("--nvidia-smi")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    try:
        controls = selected_controls_from_environment(os.environ)
    except MissingQualificationEnvironment as error:
        print(
            "ADAPTIVE_TRACE_MEMORY_MATRIX skipped=1"
            f" mode={arguments.mode}"
            f" missing_environment={','.join(error.names)}"
        )
        return SKIP_RETURN_CODE
    try:
        executable = _resolve_program(
            arguments.executable, "convergence executable"
        )
        if arguments.mode == MEMORY_MODE:
            if arguments.nvidia_smi is None:
                raise AdaptiveTraceMemoryError(
                    "memory mode requires --nvidia-smi"
                )
            nvidia_smi = _resolve_program(arguments.nvidia_smi, "nvidia-smi")
            aggregates = run_matrix(
                executable, nvidia_smi, controls, os.environ
            )
            print_aggregates(aggregates)
        else:
            aggregates = run_fresh_process_timing_matrix(
                executable, controls, os.environ
            )
            print_fresh_process_timing_aggregates(aggregates, controls)
    except AdaptiveTraceMemoryError as error:
        print(
            f"adaptive trace {arguments.mode} matrix failed: {error}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
