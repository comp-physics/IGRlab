#!/usr/bin/env python3
"""Whole-device memory control for the selected adaptive trace methods.

This test-only driver performs no build, GPU selection, file publication, or
raw-log retention.  A tracked CTest invocation supplies the already-built
convergence executable while the caller owns the exclusive visible GPU.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import sys
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

RODAS3 = "rodas3"
MONOLITHIC_BE = "monolithic_backward_euler"
PARTITIONED_BE = "partitioned_backward_euler"
METHODS = (RODAS3, MONOLITHIC_BE, PARTITIONED_BE)

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


def benchmark_command(
        executable: str, method: str, controls: Mapping[str, Any]) -> list[str]:
    """Build one exact, output-free convergence memory command."""
    if method not in METHODS:
        raise AdaptiveTraceMemoryError(f"unknown method: {method}")
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
            _, internal_floor = parse_architecture_selection_point(
                benchmark["stdout"], method
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


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the selected adaptive/BE/PBE tensor CUDA memory matrix; "
            "the caller must own the one visible GPU"
        )
    )
    parser.add_argument("--executable", required=True)
    parser.add_argument("--nvidia-smi", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    try:
        controls = selected_controls_from_environment(os.environ)
    except MissingQualificationEnvironment as error:
        print(
            "ADAPTIVE_TRACE_MEMORY_MATRIX skipped=1"
            f" missing_environment={','.join(error.names)}"
        )
        return SKIP_RETURN_CODE
    try:
        executable = _resolve_program(arguments.executable, "convergence executable")
        nvidia_smi = _resolve_program(arguments.nvidia_smi, "nvidia-smi")
        aggregates = run_matrix(
            executable, nvidia_smi, controls, os.environ
        )
        print_aggregates(aggregates)
    except AdaptiveTraceMemoryError as error:
        print(f"adaptive trace memory matrix failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
