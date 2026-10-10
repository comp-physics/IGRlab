#!/usr/bin/env python3
"""Dedicated whole-GPU memory observation for an untimed control run.

The caller is responsible for holding the external GPU lease.  This module
does not build, select, reserve, or modify a GPU.  Its benchmark timings are
intentionally unsuitable for the primary performance matrix because
``nvidia-smi`` is polled concurrently with the child process.
"""

from __future__ import annotations

import csv
import math
import os
import signal
import statistics
import subprocess
import threading
import time
from typing import Any, Mapping, Sequence


MIB = 1024 * 1024
MINIMUM_BASELINE_ROWS = 10
MINIMUM_POST_ROWS = 10
BASELINE_LIMIT_MIB = 64.0
FLOOR_TOLERANCE_BYTES = MIB
REPEAT_RELATIVE_LIMIT = 0.02
MEMORY_QUERY_FIELDS = (
    "timestamp",
    "uuid",
    "memory.total",
    "memory.used",
    "memory.free",
)


class MemoryControlError(RuntimeError):
    """A failed sampler or qualification gate, with retained partial evidence."""

    def __init__(self, message: str, evidence: dict[str, Any] | None = None):
        super().__init__(message)
        self.evidence = evidence


def _utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _normalize_uuid(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("GPU UUID must be a non-empty string")
    normalized = value.strip().lower()
    if normalized.startswith("gpu-"):
        normalized = normalized[4:]
    return normalized.replace("-", "")


def _number(value: str, field: str) -> float:
    try:
        result = float(value.strip())
    except ValueError as error:
        raise ValueError(f"{field} is not numeric: {value!r}") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} is not finite and nonnegative: {value!r}")
    return result


def _statistics(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    checked = [float(value) for value in values]
    if any(not math.isfinite(value) for value in checked):
        raise ValueError("statistics sample contains a non-finite value")
    return {
        "count": len(checked),
        "minimum": min(checked),
        "median": statistics.median(checked),
        "mean": statistics.mean(checked),
        "population_stddev": statistics.pstdev(checked),
        "maximum": max(checked),
        "range": max(checked) - min(checked),
    }


def _signal_process_group(process: subprocess.Popen[str], action: int) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, action)
    except (ProcessLookupError, PermissionError):
        try:
            process.send_signal(action)
        except ProcessLookupError:
            pass


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    _signal_process_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=1.0)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_process_group(process, signal.SIGKILL)
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired as error:
        raise MemoryControlError(
            f"unable to stop child process group {process.pid}"
        ) from error


class _NvidiaSmiPoller:
    def __init__(self, executable: str | os.PathLike[str], gpu_uuid: str,
                 poll_ms: int, environment: Mapping[str, str] | None):
        self.command = [
            str(executable),
            f"--id={gpu_uuid}",
            "--query-gpu=" + ",".join(MEMORY_QUERY_FIELDS),
            "--format=csv,noheader,nounits",
            f"--loop-ms={poll_ms}",
        ]
        self._condition = threading.Condition()
        self.rows: list[dict[str, Any]] = []
        self.raw_stdout: list[str] = []
        self.raw_stderr: list[str] = []
        self.parse_errors: list[str] = []
        try:
            self.process = subprocess.Popen(
                self.command,
                text=True,
                encoding="utf-8",
                errors="strict",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                env=dict(environment) if environment is not None else None,
                start_new_session=True,
            )
        except OSError as error:
            raise MemoryControlError(
                f"unable to start nvidia-smi poller: {error}"
            ) from error
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self._stdout_thread = threading.Thread(
            target=self._read_stdout, name="gpu-memory-control-stdout",
            daemon=True)
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, name="gpu-memory-control-stderr",
            daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for raw_line in self.process.stdout:
            received_ns = time.monotonic_ns()
            with self._condition:
                self.raw_stdout.append(raw_line)
            if not raw_line.strip():
                continue
            try:
                fields = next(csv.reader([raw_line]))
                if len(fields) != len(MEMORY_QUERY_FIELDS):
                    raise ValueError(
                        f"expected {len(MEMORY_QUERY_FIELDS)} CSV fields, "
                        f"received {len(fields)}"
                    )
                row = {
                    "sequence": 0,
                    "received_monotonic_ns": received_ns,
                    "nvidia_timestamp": fields[0].strip(),
                    "gpu_uuid": fields[1].strip(),
                    "total_mib": _number(fields[2], "memory.total"),
                    "used_mib": _number(fields[3], "memory.used"),
                    "free_mib": _number(fields[4], "memory.free"),
                    "raw_line": raw_line,
                }
                if not row["nvidia_timestamp"] or not row["gpu_uuid"]:
                    raise ValueError("timestamp and UUID must be non-empty")
                if row["used_mib"] > row["total_mib"] or \
                        row["free_mib"] > row["total_mib"]:
                    raise ValueError("used/free memory exceeds total memory")
                with self._condition:
                    row["sequence"] = len(self.rows)
                    self.rows.append(row)
                    self._condition.notify_all()
            except (csv.Error, ValueError) as error:
                with self._condition:
                    self.parse_errors.append(
                        f"{type(error).__name__}: {error}; line={raw_line!r}"
                    )
                    self._condition.notify_all()
        with self._condition:
            self._condition.notify_all()

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for chunk in self.process.stderr:
            with self._condition:
                self.raw_stderr.append(chunk)
                self._condition.notify_all()

    def wait_for_rows(self, predicate, timeout_seconds: float,
                      description: str) -> None:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while not predicate(self.rows):
                if self.process.poll() is not None:
                    raise MemoryControlError(
                        f"nvidia-smi exited before {description}: "
                        f"returncode={self.process.returncode}"
                    )
                if self.parse_errors:
                    raise MemoryControlError(
                        f"nvidia-smi emitted invalid memory evidence before "
                        f"{description}: {self.parse_errors[0]}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise MemoryControlError(
                        f"timed out waiting for {description}"
                    )
                self._condition.wait(timeout=min(remaining, 0.25))

    def stop(self) -> None:
        termination_error: Exception | None = None
        try:
            _terminate_process(self.process)
        except Exception as error:
            termination_error = error
        finally:
            self._stdout_thread.join(timeout=1.0)
            self._stderr_thread.join(timeout=1.0)
            if self.process.stdout is not None:
                self.process.stdout.close()
            if self.process.stderr is not None:
                self.process.stderr.close()
        if self._stdout_thread.is_alive() or self._stderr_thread.is_alive():
            raise MemoryControlError("nvidia-smi reader thread did not stop")
        if termination_error is not None:
            raise termination_error

    def evidence(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "pid": self.process.pid,
            "returncode": self.process.poll(),
            "raw_stdout": "".join(self.raw_stdout),
            "raw_stderr": "".join(self.raw_stderr),
            "parse_errors": list(self.parse_errors),
            "rows": [dict(row) for row in self.rows],
        }


def _run_benchmark(command: Sequence[str], *, timeout_seconds: float,
                   environment: Mapping[str, str] | None,
                   cwd: str | os.PathLike[str] | None) -> dict[str, Any]:
    started_utc = _utc_now()
    started_ns = time.monotonic_ns()
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            env=dict(environment) if environment is not None else None,
            text=True,
            encoding="utf-8",
            errors="strict",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        raise MemoryControlError(f"unable to start benchmark: {error}") from error
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process(process)
        stdout, stderr = process.communicate()
    except BaseException:
        _terminate_process(process)
        process.communicate()
        raise
    finished_ns = time.monotonic_ns()
    return {
        "command": list(command),
        "pid": process.pid,
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": finished_ns,
        "wall_seconds": (finished_ns - started_ns) / 1.0e9,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
    }


def _validate_and_summarize(evidence: dict[str, Any], *, selected_uuid: str,
                            poll_ms: int, pre_rows: int,
                            post_rows: int) -> None:
    rows = evidence["poller"]["rows"]
    poller_errors = evidence["poller"]["parse_errors"]
    benchmark = evidence["benchmark"]
    if poller_errors:
        raise MemoryControlError(
            f"nvidia-smi emitted invalid rows: {poller_errors[0]}", evidence
        )
    if benchmark["timed_out"]:
        raise MemoryControlError("benchmark timed out", evidence)
    if benchmark["returncode"] != 0:
        raise MemoryControlError(
            f"benchmark exited with status {benchmark['returncode']}", evidence
        )
    try:
        expected_uuid = _normalize_uuid(selected_uuid)
        uuid_mismatch = any(
            _normalize_uuid(row["gpu_uuid"]) != expected_uuid for row in rows
        )
    except ValueError as error:
        raise MemoryControlError(str(error), evidence) from error
    if uuid_mismatch:
        raise MemoryControlError(
            "nvidia-smi row UUID differs from the selected CUDA UUID", evidence
        )
    totals = [row["total_mib"] for row in rows]
    if not totals or any(not math.isclose(value, totals[0], rel_tol=0.0,
                                          abs_tol=1.0e-9)
                         for value in totals):
        raise MemoryControlError(
            "nvidia-smi total memory changed during the control", evidence
        )

    started_ns = benchmark["started_monotonic_ns"]
    finished_ns = benchmark["finished_monotonic_ns"]
    before = [row for row in rows
              if row["received_monotonic_ns"] < started_ns]
    after = [row for row in rows
             if row["received_monotonic_ns"] > finished_ns]
    active = [row for row in rows
              if started_ns <= row["received_monotonic_ns"] <= finished_ns]
    if len(before) < pre_rows or len(after) < post_rows:
        raise MemoryControlError(
            "sampler did not retain the required pre/post rows", evidence
        )
    if not active:
        raise MemoryControlError(
            "sampler observed no row while the benchmark was active", evidence
        )
    before = before[-pre_rows:]
    after = after[:post_rows]
    peak_window = active + after[:1]

    maximum_gap_ns = 5 * poll_ms * 1_000_000
    cadence_points = [started_ns] + [
        row["received_monotonic_ns"] for row in active
    ] + [finished_ns]
    gaps_ns = [right - left for left, right in
               zip(cadence_points, cadence_points[1:])]
    observed_maximum_gap_ns = max(gaps_ns)
    if observed_maximum_gap_ns > maximum_gap_ns:
        raise MemoryControlError(
            "active sampling gap exceeds five requested polling intervals", evidence
        )

    before_values = [row["used_mib"] for row in before]
    after_values = [row["used_mib"] for row in after]
    before_stats = _statistics(before_values)
    after_stats = _statistics(after_values)
    baseline_drift = abs(before_stats["median"] - after_stats["median"])
    if before_stats["range"] > BASELINE_LIMIT_MIB or \
            after_stats["range"] > BASELINE_LIMIT_MIB or \
            baseline_drift > BASELINE_LIMIT_MIB:
        raise MemoryControlError(
            "global-memory baseline range/drift exceeds 64 MiB", evidence
        )

    absolute_peak_mib = max(row["used_mib"] for row in peak_window)
    incremental_peak_mib = absolute_peak_mib - before_stats["median"]
    headroom_mib = totals[0] - absolute_peak_mib
    absolute_peak_bytes = int(round(absolute_peak_mib * MIB))
    incremental_peak_bytes = int(round(incremental_peak_mib * MIB))
    headroom_bytes = int(round(headroom_mib * MIB))
    total_bytes = int(round(totals[0] * MIB))
    if headroom_mib < 0.0:
        raise MemoryControlError(
            "observed global memory exceeds reported total memory", evidence
        )
    active_values = [row["used_mib"] for row in active]
    evidence["measurement"] = {
        "semantics": (
            "observed whole-device memory.used high-water mark from benchmark "
            "launch through the first post-exit sample; benchmark timings are "
            "excluded from primary performance evidence"
        ),
        "units": "MiB (2^20 bytes)",
        "selected_gpu_uuid": selected_uuid,
        "requested_poll_ms": poll_ms,
        "maximum_allowed_active_gap_ms": 5 * poll_ms,
        "observed_maximum_active_gap_ms": observed_maximum_gap_ns / 1.0e6,
        "pre_rows": before,
        "active_rows": active,
        "post_rows": after,
        "pre_used_mib": before_stats,
        "active_used_mib": _statistics(active_values),
        "post_used_mib": after_stats,
        "baseline_drift_mib": baseline_drift,
        "absolute_peak_mib": absolute_peak_mib,
        "absolute_peak_bytes": absolute_peak_bytes,
        "incremental_peak_mib": incremental_peak_mib,
        "incremental_peak_bytes": incremental_peak_bytes,
        "headroom_mib": headroom_mib,
        "headroom_bytes": headroom_bytes,
        "total_mib": totals[0],
        "total_bytes": total_bytes,
        "qualification_gates": {
            "benchmark_exit_zero": True,
            "selected_uuid_constant": True,
            "total_memory_constant": True,
            "numeric_used_and_free": True,
            "minimum_pre_rows": True,
            "minimum_post_rows": True,
            "active_gap_within_five_poll_intervals": True,
            "baseline_range_and_drift_within_64_mib": True,
        },
    }


def run_gpu_memory_control(
        benchmark_command: Sequence[str], *,
        nvidia_smi: str | os.PathLike[str], selected_gpu_uuid: str,
        poll_ms: int, timeout_seconds: float,
        environment: Mapping[str, str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        pre_rows: int = MINIMUM_BASELINE_ROWS,
        post_rows: int = MINIMUM_POST_ROWS) -> dict[str, Any]:
    """Run one dedicated benchmark control and return raw/qualified evidence.

    ``benchmark_command`` is used verbatim without a shell.  The returned
    benchmark stdout and stderr are unstripped.  On failure, a
    :class:`MemoryControlError` carries any available partial evidence.
    """
    if not benchmark_command or any(not isinstance(item, str) or not item
                                    for item in benchmark_command):
        raise MemoryControlError("benchmark command must contain non-empty strings")
    if not isinstance(poll_ms, int) or isinstance(poll_ms, bool) or poll_ms <= 0:
        raise MemoryControlError("poll_ms must be a positive integer")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise MemoryControlError("timeout_seconds must be finite and positive")
    if pre_rows < MINIMUM_BASELINE_ROWS or post_rows < MINIMUM_POST_ROWS:
        raise MemoryControlError("at least ten pre and ten post rows are required")
    try:
        _normalize_uuid(selected_gpu_uuid)
    except ValueError as error:
        raise MemoryControlError(str(error)) from error

    evidence: dict[str, Any] = {
        "schema": 1,
        "status": "running",
        "started_utc": _utc_now(),
        "benchmark": None,
        "poller": None,
    }
    poller: _NvidiaSmiPoller | None = None
    failure: Exception | None = None
    try:
        poller = _NvidiaSmiPoller(
            nvidia_smi, selected_gpu_uuid, poll_ms, environment
        )
        baseline_timeout = max(2.0, pre_rows * poll_ms * 10.0 / 1000.0)
        poller.wait_for_rows(
            lambda rows: len(rows) >= pre_rows,
            baseline_timeout,
            f"{pre_rows} pre-launch rows",
        )
        evidence["benchmark"] = _run_benchmark(
            benchmark_command,
            timeout_seconds=timeout_seconds,
            environment=environment,
            cwd=cwd,
        )
        finished_ns = evidence["benchmark"]["finished_monotonic_ns"]
        post_timeout = max(2.0, post_rows * poll_ms * 10.0 / 1000.0)
        poller.wait_for_rows(
            lambda rows: len([
                row for row in rows
                if row["received_monotonic_ns"] > finished_ns
            ]) >= post_rows,
            post_timeout,
            f"{post_rows} post-exit rows",
        )
    except Exception as error:  # cleanup and retain evidence before rethrowing
        failure = error
    finally:
        if poller is not None:
            try:
                poller.stop()
            except Exception as error:
                if failure is None:
                    failure = error
            evidence["poller"] = poller.evidence()
        evidence["finished_utc"] = _utc_now()

    if failure is not None:
        evidence["status"] = "failure"
        evidence["error"] = f"{type(failure).__name__}: {failure}"
        if isinstance(failure, MemoryControlError) and \
                failure.evidence is not None:
            raise failure
        raise MemoryControlError(str(failure), evidence) from failure
    try:
        _validate_and_summarize(
            evidence,
            selected_uuid=selected_gpu_uuid,
            poll_ms=poll_ms,
            pre_rows=pre_rows,
            post_rows=post_rows,
        )
    except MemoryControlError as error:
        evidence["status"] = "failure"
        evidence["error"] = f"MemoryControlError: {error}"
        raise
    evidence["status"] = "success"
    return evidence


def aggregate_three_repeat_controls(
        repeats: Sequence[Mapping[str, Any]],
        internal_cuda_used_floor_bytes: Sequence[int]) -> dict[str, Any]:
    """Validate and summarize three dedicated memory-control repetitions."""
    if len(repeats) != 3 or len(internal_cuda_used_floor_bytes) != 3:
        raise MemoryControlError(
            "memory qualification requires exactly three repeats and floors"
        )
    measurements: list[Mapping[str, Any]] = []
    for index, repeat in enumerate(repeats):
        if repeat.get("status") != "success" or \
                not isinstance(repeat.get("measurement"), Mapping):
            raise MemoryControlError(
                f"memory repeat {index} is not qualified"
            )
        measurements.append(repeat["measurement"])

    incremental = [float(item["incremental_peak_mib"])
                   for item in measurements]
    absolute = [float(item["absolute_peak_mib"])
                for item in measurements]
    absolute_bytes = [int(item["absolute_peak_bytes"])
                      for item in measurements]
    headroom = [float(item["headroom_mib"]) for item in measurements]
    if any(not math.isfinite(value) for value in incremental + absolute + headroom):
        raise MemoryControlError("memory repeat contains a non-finite result")
    incremental_stats = _statistics(incremental)
    stability_limit_mib = max(
        BASELINE_LIMIT_MIB,
        REPEAT_RELATIVE_LIMIT * incremental_stats["median"],
    )
    if incremental_stats["range"] > stability_limit_mib:
        raise MemoryControlError(
            "three-repeat incremental peak spread exceeds "
            "max(64 MiB, 2% of median)"
        )

    floor_checks = []
    for index, (observed_bytes, floor) in enumerate(
            zip(absolute_bytes, internal_cuda_used_floor_bytes)):
        if isinstance(floor, bool) or not isinstance(floor, int) or floor < 0:
            raise MemoryControlError(
                f"internal CUDA memory floor {index} is not a nonnegative integer"
            )
        passes = observed_bytes + FLOOR_TOLERANCE_BYTES >= floor
        floor_checks.append({
            "repeat_index": index,
            "observed_absolute_peak_bytes": observed_bytes,
            "internal_cuda_used_floor_bytes": floor,
            "tolerance_bytes": FLOOR_TOLERANCE_BYTES,
            "passes": passes,
        })
        if not passes:
            raise MemoryControlError(
                f"repeat {index} external peak is below its internal "
                "cudaMemGetInfo floor beyond 1 MiB tolerance"
            )
    return {
        "schema": 1,
        "qualified": True,
        "repeat_count": 3,
        "incremental_peak_mib": incremental_stats,
        "absolute_peak_mib": _statistics(absolute),
        "headroom_mib": _statistics(headroom),
        "stability_limit_mib": stability_limit_mib,
        "conservative_absolute_peak_mib": max(absolute),
        "conservative_incremental_peak_mib": max(incremental),
        "minimum_headroom_mib": min(headroom),
        "internal_cuda_floor_checks": floor_checks,
    }


__all__ = [
    "BASELINE_LIMIT_MIB",
    "FLOOR_TOLERANCE_BYTES",
    "MIB",
    "MemoryControlError",
    "aggregate_three_repeat_controls",
    "run_gpu_memory_control",
]
