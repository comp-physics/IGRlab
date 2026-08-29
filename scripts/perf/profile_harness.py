#!/usr/bin/env python3
"""Run an explicit L0--L2 profiling scenario and write auditable artifacts.

The runner never supplies physics, backend, device, MPI, or profiler defaults.
Callers provide an executable plus exact arguments, a scenario label, and an
output directory. L0 runs one smoke command, L1 repeats that command and
records timing, and L2 constructs an Nsight Systems or Nsight Compute command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import stat
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_WRITER = ROOT / "scripts/perf/profile_provenance.py"
PROVENANCE_VALIDATOR = ROOT / "scripts/perf/validate_profile_provenance.py"
EXIT_TOOL_UNAVAILABLE = 3
EXIT_DATA_UNAVAILABLE = 4
EXIT_EXECUTION_FAILURE = 5
RESERVED_ARTIFACTS = frozenset({
    "profile-provenance.json", "profile-manifest.json", "execution.log",
    "ncu-tensor-proof.csv", "ncu-tensor-proof.stderr.txt",
    "ncu-accepted-step-attribution.csv",
    "ncu-accepted-step-attribution.stderr.txt",
    "nsys-nvtx-sum.csv", "nsys-nvtx-sum.stderr.txt",
    "nsys-cuda-api-sum.csv", "nsys-cuda-api-sum.stderr.txt",
    "nsys-cuda-gpu-mem-time-sum.csv",
    "nsys-cuda-gpu-mem-time-sum.stderr.txt",
    "nsys-cuda-gpu-mem-size-sum.csv",
    "nsys-cuda-gpu-mem-size-sum.stderr.txt",
})
EXIT_VALIDATION_FAILURE = 6
UINT64_MAX = (1 << 64) - 1
NCU_DMMA_METRIC = "sm__inst_executed_pipe_fp64_op_dmma.sum"
NCU_DURATION_METRIC = "gpu__time_duration.avg"
NCU_CONDENSATION_KERNEL = "TensorBorrowedCondenseKernel"
NCU_CONDENSATION_FILTER = f"regex:.*{NCU_CONDENSATION_KERNEL}.*"
NCU_CONDENSATION_NVTX_RANGE = "HDG/condensation/"
NCU_VOLUME_KERNEL = "BorrowedEulerTensorVolumeContractionKernel"
NCU_VOLUME_FILTER = f"regex:.*{NCU_VOLUME_KERNEL}.*"
NCU_VOLUME_NVTX_RANGE = "HDG/local-assembly/"
NCU_ACCEPTED_STEP_NVTX_RANGE = "HDG/profiled-accepted-step/"
NCU_INSTRUCTION_QUALIFICATION_KINDS = frozenset({
    "tensor-instructions", "tensor-volume-instructions",
})
PRODUCTION_TRACE_POLICY = "engine-default"
FINAL_IGR_QUADRATURE_BY_ORDER = {1: (6, 4), 2: (8, 6), 3: (10, 8)}
FINAL_LINEAR_ROLE_RELATIVE_TOLERANCES = (1.0e-10, 3.0e-11, 1.0e-12)


class FixedSigmaJacobiProfilePolicy(NamedTuple):
    sweeps: int
    preconditioner: str
    warp: bool


FIXED_SIGMA_JACOBI_PROFILE_POLICIES = {
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi1-fp64": FixedSigmaJacobiProfilePolicy(
        1,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi1_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi1-fp64": FixedSigmaJacobiProfilePolicy(
        1,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi1_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi2-fp64": FixedSigmaJacobiProfilePolicy(
        2,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi2_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi2-fp64": FixedSigmaJacobiProfilePolicy(
        2,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi2_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi3-fp64": FixedSigmaJacobiProfilePolicy(
        3,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi3_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi3-fp64": FixedSigmaJacobiProfilePolicy(
        3,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi3_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi4-fp64": FixedSigmaJacobiProfilePolicy(
        4,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi4_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi4-fp64": FixedSigmaJacobiProfilePolicy(
        4,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi4_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi5-fp64": FixedSigmaJacobiProfilePolicy(
        5,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi5_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi5-fp64": FixedSigmaJacobiProfilePolicy(
        5,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi5_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi6-fp64": FixedSigmaJacobiProfilePolicy(
        6,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi6_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi6-fp64": FixedSigmaJacobiProfilePolicy(
        6,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi6_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi7-fp64": FixedSigmaJacobiProfilePolicy(
        7,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi7_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi7-fp64": FixedSigmaJacobiProfilePolicy(
        7,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi7_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi8-fp64": FixedSigmaJacobiProfilePolicy(
        8,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi8_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi8-fp64": FixedSigmaJacobiProfilePolicy(
        8,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi8_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi9-fp64": FixedSigmaJacobiProfilePolicy(
        9,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi9_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi9-fp64": FixedSigmaJacobiProfilePolicy(
        9,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi9_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-equilibrated-field-split-"
    "sigma-jacobi10-fp64": FixedSigmaJacobiProfilePolicy(
        10,
        "native_element_schwarz_equilibrated_field_split_sigma_jacobi10_fp64",
        False,
    ),
    "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
    "sigma-jacobi10-fp64": FixedSigmaJacobiProfilePolicy(
        10,
        "native_element_schwarz_warp_equilibrated_field_split_"
        "sigma_jacobi10_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-warp-coupled-"
    "sigma-jacobi2-fp64": FixedSigmaJacobiProfilePolicy(
        2,
        "native_element_schwarz_warp_coupled_sigma_jacobi2_fp64",
        True,
    ),
    "native-element-schwarz-matrix-free-warp-coupled-"
    "sigma-jacobi10-fp64": FixedSigmaJacobiProfilePolicy(
        10,
        "native_element_schwarz_warp_coupled_sigma_jacobi10_fp64",
        True,
    ),
}
COUPLED_WARP_BASE_PROFILE_POLICY = \
    "native-element-schwarz-matrix-free-warp-coupled-fp64"
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


class ProfileTracePolicyContract(NamedTuple):
    requested: str
    resolved: str
    preconditioner: str
    variant: str
    production_selectable: bool
    fixed_sigma_jacobi: FixedSigmaJacobiProfilePolicy | None
    element_schwarz_required: bool
    factorization: str
    apply: str
    tile_width: int


def _profile_trace_policy_contract(
        requested: str, mode: str, order: int) -> ProfileTracePolicyContract:
    if mode not in {"conventional", "igr"}:
        raise RuntimeError("profiled trace policy mode is invalid")
    if isinstance(order, bool) or order not in {1, 2, 3}:
        raise RuntimeError("profiled trace policy order is invalid")
    if requested == PRODUCTION_TRACE_POLICY:
        if mode == "igr":
            resolved = PRODUCTION_TENSOR_IGR_TRACE_POLICY_BY_ORDER[order]
            fixed = FIXED_SIGMA_JACOBI_PROFILE_POLICIES[resolved]
            preconditioner = fixed.preconditioner
        else:
            resolved = PRODUCTION_TRACE_POLICY
            preconditioner = "hypre_diag_scale"
            fixed = None
        return ProfileTracePolicyContract(
            requested,
            resolved,
            preconditioner,
            "production",
            True,
            fixed,
            mode == "igr",
            "lower-multiplicative-uhat4-sigmahat"
            if mode == "igr" else "not-applicable",
            "cooperative-pivoted-lu-fp64"
            if mode == "igr" else "not-applicable",
            (16 if order == 1 else 32) if mode == "igr" else 0,
        )
    if requested == COUPLED_WARP_BASE_PROFILE_POLICY and mode == "igr":
        return ProfileTracePolicyContract(
            requested,
            requested,
            "native_element_schwarz_warp_coupled_fp64",
            "ablation",
            False,
            None,
            True,
            "full-coupled-patch-lu",
            "cooperative-full-patch-pivoted-lu-fp64",
            32,
        )
    fixed = FIXED_SIGMA_JACOBI_PROFILE_POLICIES.get(requested)
    if fixed is None or mode != "igr":
        raise RuntimeError(
            "profiled trace policy is neither engine-default nor an exact "
            "native element-Schwarz IGR ablation policy"
        )
    coupled = "-warp-coupled-" in requested
    return ProfileTracePolicyContract(
        requested,
        requested,
        fixed.preconditioner,
        "ablation",
        False,
        fixed,
        True,
        "full-coupled-patch-lu" if coupled
        else "lower-multiplicative-uhat4-sigmahat",
        "cooperative-full-patch-pivoted-lu-fp64" if coupled
        else "cooperative-pivoted-lu-fp64" if fixed.warp
        else "scalar-pivoted-lu-fp64",
        32 if coupled or (fixed.warp and order != 1)
        else 16 if fixed.warp else 0,
    )


NCU_ATTRIBUTION_RANGES = {
    "HDG/local-assembly": "local_assembly",
    "HDG/local-factorization": "local_factorization",
    "HDG/condensation": "condensation",
    "HDG/trace-assembly": "trace_assembly",
    "HDG/trace-solve": "trace_solve",
    "HDG/interior-recovery": "interior_recovery",
    "HDG/reduction": "reductions",
    "HDG/validation": "validation",
    "HDG/packing": "packing_and_conversion",
    "HDG/device-transfer": "device_transfers",
    "HDG/nonlinear-update": "nonlinear_update",
    "HDG/synchronization": "synchronization",
}
NCU_REQUIRED_ATTRIBUTION_CATEGORIES = frozenset({
    "local_assembly", "condensation", "trace_assembly", "trace_solve",
    "interior_recovery", "reductions", "validation",
    "packing_and_conversion", "nonlinear_update",
})
NCU_REPORT = Path("capture.ncu-rep")
NSYS_REPORT = Path("capture.nsys-rep")
NSYS_QUALIFICATION_SUMMARIES = (
    ("nvtx_sum", "nsys-nvtx-sum"),
    ("cuda_api_sum", "nsys-cuda-api-sum"),
    ("cuda_gpu_mem_time_sum", "nsys-cuda-gpu-mem-time-sum"),
    ("cuda_gpu_mem_size_sum", "nsys-cuda-gpu-mem-size-sum"),
)


def safe_relative_path(value: str) -> Path:
    candidate = Path(value)
    if not value or candidate.is_absolute() or ".." in candidate.parts:
        raise argparse.ArgumentTypeError("must be a non-empty relative path without '..'")
    return candidate


def parse_key_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("must be NAME=RELATIVE_PATH")
    name, raw_path = value.split("=", 1)
    if not name or not name.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError("requested data NAME must be alphanumeric, '_' or '-'")
    return name, safe_relative_path(raw_path)


def parse_nsys_capture_range_end(value: str) -> str:
    if value in {"none", "stop", "stop-shutdown"}:
        return value
    parts = value.split(":")
    if parts[0] == "repeat":
        if len(parts) == 1:
            return value
        if len(parts) not in {2, 3}:
            raise argparse.ArgumentTypeError("invalid Nsight Systems repeat capture policy")
        count, mode = parts[1], parts[2] if len(parts) == 3 else ""
        if count and (not count.isdigit() or int(count) <= 0):
            raise argparse.ArgumentTypeError("Nsight Systems repeat count must be positive")
        if not count and not mode:
            raise argparse.ArgumentTypeError("invalid Nsight Systems repeat capture policy")
        if mode and mode not in {"defer", "sync", "async"}:
            raise argparse.ArgumentTypeError("invalid Nsight Systems repeat mode")
        return value
    if parts[0] == "repeat-shutdown" and len(parts) in {2, 3}:
        count = parts[1]
        mode = parts[2] if len(parts) == 3 else ""
        if not count.isdigit() or int(count) <= 0:
            raise argparse.ArgumentTypeError(
                "Nsight Systems repeat-shutdown count must be positive"
            )
        if mode and mode not in {"defer", "sync", "async"}:
            raise argparse.ArgumentTypeError("invalid Nsight Systems repeat mode")
        return value
    raise argparse.ArgumentTypeError("invalid Nsight Systems capture-range end policy")


def quoted(command: list[str]) -> str:
    return " ".join(json.dumps(item) for item in command)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    """Capture a regular file strongly enough to detect replacement or rewrite."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"not a regular file: {path}")
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    final = os.lstat(path)
    if (stat.S_ISLNK(final.st_mode) or final.st_dev != info.st_dev or
            final.st_ino != info.st_ino or final.st_size != size):
        raise RuntimeError(f"file changed while reading: {path}")
    return {
        "path": str(path),
        "bytes": size,
        "sha256": digest.hexdigest(),
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "st_mtime_ns": info.st_mtime_ns,
    }


def command_output(command: list[str], label: str) -> str:
    try:
        result = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise RuntimeError(f"could not query {label}: {exc}") from exc
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"{label} query exited {result.returncode}: {detail or 'no diagnostic'}"
        )
    output = result.stdout.strip() or result.stderr.strip()
    if not output:
        raise RuntimeError(f"{label} query returned no output")
    return output


def parse_typed_cmake_cache(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"could not read CMake cache {path}: {exc}") from exc
    for raw_line in lines:
        if not raw_line or raw_line.startswith(("#", "//")) or "=" not in raw_line:
            continue
        key_and_type, value = raw_line.split("=", 1)
        key, separator, _ = key_and_type.partition(":")
        if separator and key:
            if key in result:
                raise RuntimeError(f"duplicate CMake cache key: {key}")
            result[key] = value
    return result


def parse_key_value_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"could not read MFEM identity {path}: {exc}") from exc
    for raw_line in lines:
        if not raw_line or raw_line.startswith("#"):
            continue
        key, separator, value = raw_line.partition("=")
        if not separator or not key or key in result:
            raise RuntimeError(f"invalid MFEM identity line: {raw_line!r}")
        result[key] = value
    return result


def validated_build_identity(args: argparse.Namespace) -> dict[str, Any]:
    assert args.build_manifest is not None
    cache_path = args.build_manifest.resolve(strict=True)
    if cache_path.name != "CMakeCache.txt":
        raise RuntimeError("qualification build manifest must be CMakeCache.txt")
    cache = parse_typed_cmake_cache(cache_path)
    expected_cache = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CUDA_ARCHITECTURES": "120",
        "DG_IGR_ENABLE_HDG_RUNTIME": "ON",
        "DG_IGR_ENABLE_OBSERVABILITY": "ON",
        "DG_IGR_OBSERVABILITY_USE_NVTX": "ON",
    }
    for key, expected in expected_cache.items():
        if cache.get(key) != expected:
            raise RuntimeError(
                f"qualification build cache {key}={cache.get(key)!r}; "
                f"expected {expected!r}"
            )
    if not cache.get("CMAKE_HOME_DIRECTORY"):
        raise RuntimeError("qualification build cache does not record CMAKE_HOME_DIRECTORY")
    cache_source = Path(cache["CMAKE_HOME_DIRECTORY"]).resolve()
    if cache_source != args.source_root:
        raise RuntimeError(
            f"qualification build source {cache_source} does not match "
            f"{args.source_root}"
        )
    for key in ("CMAKE_CUDA_COMPILER", "CMAKE_CXX_COMPILER", "MFEM_DIR"):
        if not cache.get(key):
            raise RuntimeError(f"qualification build cache does not record {key}")
    if not args.exe.is_relative_to(cache_path.parent):
        raise RuntimeError(
            f"profiled executable {args.exe} is outside supplied build "
            f"directory {cache_path.parent}"
        )

    mfem_path = args.mfem_identity_path.resolve(strict=True)
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
            raise RuntimeError(
                f"MFEM identity {key}={mfem.get(key)!r}; expected {expected!r}"
            )
    if Path(mfem.get("source_root", "")).resolve() != \
            (args.source_root / "tpl" / "mfem").resolve():
        raise RuntimeError("MFEM identity source root does not match source tree")
    if "cuda_arch=120" not in mfem.get("hypre_spec", ""):
        raise RuntimeError("MFEM identity does not bind HYPRE to CUDA architecture 120")
    for key in ("mfem_pin", "compiler", "install_root"):
        if not mfem.get(key):
            raise RuntimeError(f"MFEM identity does not record {key}")
    expected_mfem_dir = (Path(mfem["install_root"]) / "lib/cmake/mfem").resolve()
    if Path(cache["MFEM_DIR"]).resolve() != expected_mfem_dir:
        raise RuntimeError(
            "application MFEM_DIR does not match the supplied MFEM identity"
        )
    selected_cache = {
        key: cache[key] for key in (
            "CMAKE_HOME_DIRECTORY", "CMAKE_BUILD_TYPE",
            "CMAKE_CUDA_ARCHITECTURES", "CMAKE_CUDA_COMPILER",
            "CMAKE_CXX_COMPILER", "MFEM_DIR", "DG_IGR_ENABLE_HDG_RUNTIME",
            "DG_IGR_ENABLE_OBSERVABILITY", "DG_IGR_OBSERVABILITY_USE_NVTX",
        )
    }
    return {
        "cmake_cache": file_identity(cache_path),
        "validated_cmake_values": selected_cache,
        "mfem_identity": file_identity(mfem_path),
        "validated_mfem_values": mfem,
    }


def normalized_compute_capability(value: str) -> str:
    parts = value.strip().split(".", 1)
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise RuntimeError(f"invalid GPU compute capability: {value!r}")
    return f"{int(parts[0])}.{int(parts[1])}"


def query_gpu_identity(args: argparse.Namespace) -> dict[str, Any]:
    if Path(args.resolved_nvidia_smi).name != "nvidia-smi":
        raise RuntimeError("qualification requires the nvidia-smi executable")
    command = [
        args.resolved_nvidia_smi, f"--id={args.gpu_index}",
        "--query-gpu=index,uuid,name,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    output = command_output(command, "nvidia-smi GPU identity")
    try:
        rows = list(csv.reader(io.StringIO(output)))
    except csv.Error as exc:
        raise RuntimeError(f"could not parse nvidia-smi output: {exc}") from exc
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimeError("nvidia-smi must return exactly one five-column GPU row")
    index, uuid, name, driver, compute_capability = (
        value.strip() for value in rows[0]
    )
    if index != str(args.gpu_index):
        raise RuntimeError(
            f"nvidia-smi GPU index mismatch: expected {args.gpu_index}, observed {index}"
        )
    if uuid != args.gpu_identity:
        raise RuntimeError(
            f"GPU UUID mismatch: expected {args.gpu_identity}, observed {uuid}"
        )
    if normalized_compute_capability(compute_capability) != "12.0":
        raise RuntimeError(
            f"qualification requires compute capability 12.0, observed "
            f"{compute_capability}"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible not in {index, uuid}:
        raise RuntimeError(
            "qualification requires CUDA_VISIBLE_DEVICES to select exactly the "
            "queried GPU by index or UUID"
        )
    return {
        "query_command": command,
        "index": index,
        "uuid": uuid,
        "name": name,
        "driver_version": driver,
        "compute_capability": normalized_compute_capability(compute_capability),
        "cuda_visible_devices": visible,
        "nvidia_smi_executable": file_identity(Path(args.resolved_nvidia_smi)),
        "raw_output_sha256": sha256_bytes(output.encode("utf-8")),
    }


def query_profiler_identity(args: argparse.Namespace, tool: str) -> dict[str, Any]:
    output = command_output([tool, "--version"], "profiler version")
    expected_name = "ncu" if args.level == "l2-ncu" else "nsys"
    expected_banner = (
        "Nsight Compute Command Line Profiler"
        if args.level == "l2-ncu"
        else "NVIDIA Nsight Systems"
    )
    if Path(tool).name != expected_name or expected_banner not in output:
        raise RuntimeError(
            f"qualification requires the {expected_name} NVIDIA profiler banner"
        )
    if output != args.tool_version:
        raise RuntimeError(
            f"profiler version mismatch: expected {args.tool_version!r}, "
            f"observed {output!r}"
        )
    return {
        "executable": file_identity(Path(tool)),
        "version_query": [tool, "--version"],
        "version": output,
        "version_output_sha256": sha256_bytes(output.encode("utf-8")),
    }


def available(value: str) -> dict[str, str]:
    return {"status": "available", "value": value}


def unavailable(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def git_value(root: Path, *arguments: str, clean_empty: bool = False) -> dict[str, str]:
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return unavailable(f"git could not run: {exc}")
    if result.returncode == 0:
        value = result.stdout.strip()
        if value or clean_empty:
            return available(value or "clean")
    return unavailable(result.stderr.strip() or f"git exited {result.returncode}")


def live_source_identity(root: Path) -> dict[str, dict[str, str]]:
    return {
        "root": available(str(root)),
        "revision": git_value(root, "rev-parse", "HEAD"),
        "tree": git_value(root, "rev-parse", "HEAD^{tree}"),
        "status": git_value(root, "status", "--short", clean_empty=True),
    }


def available_value(field: Any, label: str) -> str:
    if not isinstance(field, dict) or field.get("status") != "available":
        raise RuntimeError(f"{label} is unavailable")
    value = field.get("value")
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} is empty")
    return value


def capture_execution_identity(args: argparse.Namespace,
                               pin: dict[str, Any],
                               tool: str | None) -> dict[str, Any]:
    source = pin["payload"].get("source")
    if not isinstance(source, dict):
        raise RuntimeError("pinned source identity is unavailable")
    root = available_value(source.get("root"), "source root")
    if root != str(args.source_root):
        raise RuntimeError(
            f"pinned source root mismatch: expected {args.source_root}, observed {root}"
        )
    revision = available_value(source.get("revision"), "source revision")
    tree = available_value(source.get("tree"), "source tree")
    status = available_value(source.get("status"), "source status")
    if args.expected_commit and revision != args.expected_commit:
        raise RuntimeError(
            f"source revision mismatch: expected {args.expected_commit}, observed {revision}"
        )
    if args.expected_tree and tree != args.expected_tree:
        raise RuntimeError(
            f"source tree mismatch: expected {args.expected_tree}, observed {tree}"
        )
    if args.require_clean_source and status != "clean":
        raise RuntimeError(f"source is not clean: {status}")
    qualification: dict[str, Any] | None = None
    if args.require_exact_candidate:
        assert args.build_manifest is not None
        build_root = args.build_manifest.parent
        if not args.exe.is_relative_to(build_root):
            raise RuntimeError(
                f"profiled executable {args.exe} is outside supplied build "
                f"directory {build_root}"
            )
        if tool is None:
            raise RuntimeError("exact-candidate profiling tool is unavailable")
    if args.evidence_mode == "qualification":
        assert tool is not None
        if args.resolved_nvidia_smi is None:
            raise RuntimeError(
                f"nvidia-smi executable not found: {args.nvidia_smi_executable}"
            )
        qualification = {
            "build": validated_build_identity(args),
            "profiler": query_profiler_identity(args, tool),
            "gpu": query_gpu_identity(args),
        }
    return {
        "source": source,
        "expected_commit": args.expected_commit or "not required",
        "expected_tree": args.expected_tree or "not required",
        "clean_source_required": args.require_clean_source,
        "exact_candidate_required": args.require_exact_candidate,
        "executable": file_identity(args.exe),
        "build_manifest": (
            file_identity(args.build_manifest)
            if args.build_manifest is not None
            else {"status": "unavailable", "reason": "not supplied"}
        ),
        "profiler_executable": (
            file_identity(Path(tool))
            if tool is not None
            else {"status": "unavailable", "reason": "not resolved"}
        ),
        "qualification": qualification or {
            "status": "unavailable", "reason": "smoke or non-L2 evidence mode"
        },
    }


def execution_identity_unchanged(args: argparse.Namespace,
                                 start: dict[str, Any]) -> tuple[bool, str]:
    try:
        current_source = live_source_identity(args.source_root)
        current_executable = file_identity(args.exe)
        current_build = (
            file_identity(args.build_manifest)
            if args.build_manifest is not None
            else {"status": "unavailable", "reason": "not supplied"}
        )
        current_tool = (
            file_identity(Path(args.resolved_tool))
            if args.resolved_tool is not None
            else {"status": "unavailable", "reason": "not resolved"}
        )
        current_qualification = (
            {
                "build": validated_build_identity(args),
                "profiler": query_profiler_identity(args, args.resolved_tool),
                "gpu": query_gpu_identity(args),
            }
            if args.evidence_mode == "qualification"
            else {"status": "unavailable", "reason": "smoke or non-L2 evidence mode"}
        )
    except (OSError, RuntimeError) as exc:
        return False, f"could not revalidate execution identity: {exc}"
    mismatches: list[str] = []
    if current_source != start["source"]:
        mismatches.append("source")
    if current_executable != start["executable"]:
        mismatches.append("executable")
    if current_build != start["build_manifest"]:
        mismatches.append("build manifest")
    if current_tool != start["profiler_executable"]:
        mismatches.append("profiler executable")
    if current_qualification != start["qualification"]:
        mismatches.append("qualification build, MFEM, profiler, or GPU identity")
    return (not mismatches,
            "" if not mismatches else "changed during execution: " + ", ".join(mismatches))


def artifact_entry(path: Path, out_dir: Path, command: list[str]) -> dict[str, Any]:
    data = regular_bytes(path, out_dir)
    return {"path": path.relative_to(out_dir).as_posix(), "bytes": len(data),
            "sha256": sha256_bytes(data), "generating_command": command}


def tool_path(value: str) -> str | None:
    if not value:
        return None
    if os.path.sep in value:
        return str(Path(value).resolve()) if Path(value).is_file() and os.access(value, os.X_OK) else None
    found = shutil.which(value)
    return str(Path(found).resolve()) if found else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def regular_snapshot(path: Path, out_dir: Path) -> tuple[bytes, dict[str, int]]:
    """Read a regular, non-symlink artifact with its open-file identity."""
    try:
        relative = path.relative_to(out_dir)
        current = out_dir
        for component in relative.parts:
            current = current / component
            if current.is_symlink():
                raise ValueError(f"symlink path component: {relative}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"not a regular file: {relative}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(out_dir.resolve(strict=True)):
            raise ValueError(f"path escapes output directory: {relative}")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"not a regular file: {relative}")
            data = stream.read()
        final = os.lstat(path)
        if final.st_ino != info.st_ino or final.st_dev != info.st_dev or stat.S_ISLNK(final.st_mode):
            raise ValueError(f"artifact changed while reading: {relative}")
        return data, {"st_dev": info.st_dev, "st_ino": info.st_ino,
                      "st_ctime_ns": info.st_ctime_ns, "st_size": info.st_size}
    except (OSError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


def regular_bytes(path: Path, out_dir: Path) -> bytes:
    """Read a regular, non-symlink artifact confined to out_dir."""
    return regular_snapshot(path, out_dir)[0]


def write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def open_exclusive_binary(path: Path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    return os.fdopen(descriptor, "wb")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("l0", "l1", "l2-nsys", "l2-ncu"), required=True)
    parser.add_argument("--scenario", required=True, help="Caller-owned scenario label; no physics defaults are inferred.")
    parser.add_argument("--exe", required=True, type=Path, help="Exact application executable.")
    parser.add_argument("--arg", action="append", default=[], help="One exact application argument; repeat as needed.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument(
        "--evidence-mode", choices=("smoke", "qualification"), default="",
        help="Required for L2: fake-tool smoke or fail-closed promotion qualification.",
    )
    parser.add_argument("--expected-commit", default="")
    parser.add_argument("--expected-tree", default="")
    parser.add_argument("--require-clean-source", action="store_true")
    parser.add_argument("--require-exact-candidate", action="store_true")
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--environment", action="append", default=[], metavar="KEY", help="Allowlisted environment variable copied into provenance.")
    parser.add_argument("--label", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--mfem-identity", default="")
    parser.add_argument("--gpu-identity", default="")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--nvidia-smi-executable", default="nvidia-smi")
    parser.add_argument("--tool-executable", default="", help="Required for L2; fake tools work in no-capture tests.")
    parser.add_argument("--tool-version", default="")
    parser.add_argument("--repetitions", type=int, default=0, help="L1 measured repetitions; must be 5 through 7.")
    parser.add_argument("--warmups", type=int, default=0, help="L1 warm-up repetitions; exactly one is required.")
    parser.add_argument("--nsys-trace", default="", help="Required L2 Nsys trace set.")
    parser.add_argument("--nsys-sample", default="", help="Required L2 Nsys sample mode.")
    parser.add_argument("--nsys-capture-range", choices=("none", "cudaProfilerApi", "nvtx"), default="", help="Explicit optional Nsys capture trigger.")
    parser.add_argument("--nsys-capture-range-end", type=parse_nsys_capture_range_end, default=None, help="Explicit Nsys capture-range end policy.")
    parser.add_argument("--nsys-nvtx-capture", default="", help="NVTX range/domain trigger used with --nsys-capture-range nvtx.")
    parser.add_argument("--ncu-set", default="", help="Explicit NCU metric set.")
    parser.add_argument("--ncu-target-processes", choices=("application-only", "all"), default="")
    parser.add_argument(
        "--ncu-qualification-kind",
        choices=("tensor-instructions", "tensor-volume-instructions",
                 "accepted-step-attribution"),
        default="",
        help=("Required for NCU qualification: exact condensation or volume "
              "tensor-instruction proof, or duration-only accepted-step "
              "category attribution."),
    )
    parser.add_argument("--ncu-section", action="append", default=[], help="Explicit NCU section; repeat as needed.")
    parser.add_argument("--ncu-metric", action="append", default=[], help="One exact NCU metric; repeat as needed.")
    parser.add_argument("--ncu-kernel-name", default="", help="Explicit NCU kernel-name filter.")
    parser.add_argument("--ncu-kernel-name-base", default="", help="Explicit NCU kernel-name-base semantics.")
    parser.add_argument("--ncu-launch-count", type=int, default=0)
    parser.add_argument("--ncu-launch-skip", type=int, default=0)
    parser.add_argument("--ncu-nvtx-include", default="", help="Explicit NCU NVTX include filter.")
    parser.add_argument("--ncu-nvtx-exclude", default="", help="Explicit NCU NVTX exclusion filter.")
    parser.add_argument("--require-data", action="append", type=parse_key_path, default=[], metavar="NAME=RELATIVE_PATH", help="Tool-produced report/table required after L2 execution.")
    parser.add_argument("--dry-run", action="store_true", help="Write provenance/manifest and print command without execution.")
    return parser


def required_application_value(parser: argparse.ArgumentParser,
                               args: argparse.Namespace, option: str) -> str:
    positions = [index for index, value in enumerate(args.arg) if value == option]
    if len(positions) != 1:
        parser.error(
            f"profiler qualification requires exactly one application {option}"
        )
    position = positions[0]
    if position + 1 >= len(args.arg) or args.arg[position + 1].startswith("--"):
        parser.error(f"application {option} requires a value")
    return args.arg[position + 1]


def validate_accepted_step_qualification_application(
        parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.exe.name != "dg_igr_hdg_runtime_cuda_accepted_step_benchmark":
        parser.error(
            "profiler qualification requires the production accepted-step benchmark"
        )
    expected_application = {
        "--engine": "tensor",
        "--trace-policy": PRODUCTION_TRACE_POLICY,
        "--expected-source-revision": args.expected_commit,
        "--expected-source-tree": args.expected_tree,
        "--expected-source-status": "clean",
        "--expected-build-directory": str(args.build_manifest.parent.resolve()),
        "--expected-cmake-cache": str(args.build_manifest.resolve()),
    }
    for option, expected in expected_application.items():
        value = required_application_value(parser, args, option)
        if value != expected:
            parser.error(
                f"profiler qualification application {option}={value!r}; "
                f"expected {expected!r}"
            )
    mode = required_application_value(parser, args, "--mode")
    if mode not in {"conventional", "igr"}:
        parser.error("profiler qualification requires conventional or IGR mode")
    order_text = required_application_value(parser, args, "--order")
    if order_text not in {"1", "2", "3"}:
        parser.error("profiler qualification requires application order 1, 2, or 3")
    order = int(order_text)
    # Promotion-bound NCU/Nsys captures qualify only the production selector.
    # Explicit coupled and lower-split policies remain available to ordinary
    # smoke/screening runs, but may not be represented as promotable evidence.
    expected_role_tolerances = dict(zip(
        ("--newton-linear-rtol", "--final-linear-rtol",
         "--true-residual-tightening-linear-rtol"),
        FINAL_LINEAR_ROLE_RELATIVE_TOLERANCES,
    ))
    for option, expected in expected_role_tolerances.items():
        try:
            value = float(required_application_value(parser, args, option))
        except ValueError:
            parser.error(f"profiler qualification application {option} is invalid")
        if value != expected:
            parser.error(
                f"profiler qualification application {option}={value!r}; "
                f"expected {expected!r}"
            )
    if mode == "igr":
        volume, face = FINAL_IGR_QUADRATURE_BY_ORDER[order]
        for option, expected in (
                ("--igr-volume-quadrature-order", volume),
                ("--igr-face-quadrature-order", face)):
            if required_application_value(parser, args, option) != str(expected):
                parser.error(
                    f"profiler qualification application {option} must be "
                    f"{expected} for p{order} IGR"
                )
    elif "--igr-volume-quadrature-order" in args.arg or \
            "--igr-face-quadrature-order" in args.arg:
        parser.error(
            "conventional profiler qualification forbids IGR quadrature options"
        )
    measured_step = required_application_value(
        parser, args, "--profile-measured-step"
    )
    if measured_step not in {"0", "1", "2", "3", "4"}:
        parser.error("profiler qualification step must be in [0,4]")
    if args.arg.count("--no-export") != 1 or "--export" in args.arg:
        parser.error(
            "profiler qualification requires exactly one application --no-export"
        )
    if "--igr" in args.arg or "--conventional" in args.arg:
        parser.error("profiler qualification forbids application mode aliases")
    try:
        mesh_n = int(required_application_value(parser, args, "--mesh-n"))
        periods = int(
            required_application_value(parser, args, "--domain-periods")
        )
        timestep = float(required_application_value(parser, args, "--dt"))
    except ValueError:
        parser.error(
            "profiler qualification requires numeric mesh, periods, and timestep"
        )
    if mesh_n < 2 or periods < 1 or periods > mesh_n or mesh_n % periods:
        parser.error("profiler qualification requires a valid explicit periodic mesh")
    if not math.isfinite(timestep) or timestep <= 0.0:
        parser.error("profiler qualification requires a finite positive timestep")


def validate_ncu_qualification_inputs(parser: argparse.ArgumentParser,
                                      args: argparse.Namespace) -> None:
    validate_accepted_step_qualification_application(parser, args)
    if args.ncu_target_processes != "application-only":
        parser.error("NCU qualification requires application-only target processes")
    if args.ncu_set or args.ncu_section:
        parser.error("NCU qualification uses only its exact required metric list")
    if args.ncu_kernel_name_base != "demangled":
        parser.error("NCU qualification requires demangled kernel names")
    normalized_nvtx = args.ncu_nvtx_include.replace("\\/", "/")
    if args.ncu_launch_count <= 0:
        parser.error("NCU qualification requires a positive --ncu-launch-count")
    if args.ncu_launch_skip != 0:
        parser.error("NCU qualification forbids --ncu-launch-skip")
    if args.ncu_qualification_kind in NCU_INSTRUCTION_QUALIFICATION_KINDS:
        if set(args.ncu_metric) != {NCU_DMMA_METRIC, NCU_DURATION_METRIC} or \
                len(args.ncu_metric) != 2:
            parser.error(
                "tensor-instruction qualification requires exactly the DMMA "
                "instruction and GPU duration metrics"
            )
        volume = args.ncu_qualification_kind == \
            "tensor-volume-instructions"
        expected_filter = NCU_VOLUME_FILTER if volume \
            else NCU_CONDENSATION_FILTER
        expected_range = NCU_VOLUME_NVTX_RANGE if volume \
            else NCU_CONDENSATION_NVTX_RANGE
        hotspot = "volume contraction" if volume else "condensation"
        if args.ncu_kernel_name != expected_filter:
            parser.error(
                "tensor-instruction qualification requires the exact tensor "
                f"{hotspot} kernel filter"
            )
        if normalized_nvtx != expected_range or \
                args.ncu_nvtx_exclude:
            parser.error(
                "tensor-instruction qualification requires only the HDG "
                f"{hotspot} NVTX include range"
            )
    elif args.ncu_qualification_kind == "accepted-step-attribution":
        if args.ncu_metric != [NCU_DURATION_METRIC]:
            parser.error(
                "accepted-step attribution requires exactly the GPU duration metric"
            )
        if args.ncu_kernel_name:
            parser.error(
                "accepted-step attribution forbids a kernel filter"
            )
        if normalized_nvtx != NCU_ACCEPTED_STEP_NVTX_RANGE or \
                args.ncu_nvtx_exclude:
            parser.error(
                "accepted-step attribution requires only the outer profiled "
                "accepted-step NVTX include range"
            )
    else:
        parser.error(
            "NCU qualification requires explicit --ncu-qualification-kind"
        )



def validate_inputs(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.scenario.strip():
        parser.error("--scenario must be non-empty")
    if not args.exe.is_file() or not os.access(args.exe, os.X_OK):
        parser.error(f"--exe is not an executable file: {args.exe}")
    if args.out_dir.exists():
        parser.error(f"--out-dir already exists: {args.out_dir}")
    if args.build_manifest is not None and (
            args.build_manifest.is_symlink() or not args.build_manifest.is_file()):
        parser.error(f"--build-manifest is not a regular file: {args.build_manifest}")
    if args.level.startswith("l2-") and not args.evidence_mode:
        parser.error("L2 requires explicit --evidence-mode smoke or qualification")
    if not args.level.startswith("l2-") and args.evidence_mode:
        parser.error("--evidence-mode is valid only for L2 profiling")
    if args.evidence_mode == "qualification":
        args.require_exact_candidate = True
    if args.evidence_mode == "smoke" and args.require_exact_candidate:
        parser.error("exact-candidate evidence requires --evidence-mode qualification")
    if args.require_exact_candidate:
        if not args.level.startswith("l2-"):
            parser.error("--require-exact-candidate is valid only for L2 profiling")
        if args.dry_run:
            parser.error("exact-candidate profiling cannot be a dry run")
        if not args.expected_commit or not args.expected_tree or \
                not args.require_clean_source or args.build_manifest is None:
            parser.error(
                "exact-candidate profiling requires commit, tree, clean source, "
                "and build manifest"
            )
        if not args.tool_version or not args.mfem_identity or not args.gpu_identity:
            parser.error(
                "exact-candidate profiling requires tool version, MFEM identity, "
                "and GPU identity"
            )
        if args.gpu_index is None or args.gpu_index < 0:
            parser.error("qualification requires a non-negative --gpu-index")
        if "CUDA_VISIBLE_DEVICES" not in args.environment:
            parser.error(
                "qualification requires --environment CUDA_VISIBLE_DEVICES"
            )
        mfem_path = Path(args.mfem_identity).expanduser()
        if mfem_path.is_symlink() or not mfem_path.is_file():
            parser.error(
                "qualification --mfem-identity must name a regular identity file"
            )
        args.mfem_identity_path = mfem_path.resolve()
    if args.level == "l1" and not 5 <= args.repetitions <= 7:
        parser.error("L1 requires 5 through 7 measured --repetitions")
    if args.level == "l1" and args.warmups != 1:
        parser.error("L1 requires exactly one --warmups")
    if args.level != "l1" and (args.repetitions or args.warmups):
        parser.error("--repetitions and --warmups are valid only for L1")
    if args.level.startswith("l2-") and not args.tool_executable:
        parser.error("--tool-executable is required for L2")
    if args.level == "l2-nsys" and (not args.nsys_trace or not args.nsys_sample):
        parser.error("L2 Nsys requires explicit --nsys-trace and --nsys-sample")
    if args.level != "l2-nsys" and (
            args.nsys_capture_range or args.nsys_capture_range_end or
            args.nsys_nvtx_capture):
        parser.error("Nsys capture-range options are valid only for L2 Nsys")
    if args.level == "l2-nsys":
        trace_apis = {item.strip() for item in args.nsys_trace.split(",")}
        if args.nsys_capture_range_end and not args.nsys_capture_range:
            parser.error("--nsys-capture-range-end requires --nsys-capture-range")
        if args.nsys_capture_range in {"cudaProfilerApi", "nvtx"} and \
                not args.nsys_capture_range_end:
            parser.error("triggered Nsys capture requires explicit --nsys-capture-range-end")
        if args.nsys_capture_range == "cudaProfilerApi" and "cuda" not in trace_apis:
            parser.error("cudaProfilerApi capture requires CUDA tracing")
        if args.nsys_capture_range == "nvtx":
            if "nvtx" not in trace_apis:
                parser.error("NVTX capture requires NVTX tracing")
            if not args.nsys_nvtx_capture.strip():
                parser.error("NVTX capture requires --nsys-nvtx-capture")
        elif args.nsys_nvtx_capture:
            parser.error("--nsys-nvtx-capture requires --nsys-capture-range nvtx")
        if args.nsys_capture_range == "none" and \
                args.nsys_capture_range_end not in {None, "none"}:
            parser.error("capture range 'none' accepts only range-end 'none'")
    if args.level == "l2-ncu" and not (args.ncu_set or args.ncu_section or args.ncu_metric):
        parser.error("L2 NCU requires explicit --ncu-set, --ncu-section, or --ncu-metric")
    if args.level == "l2-ncu" and not args.ncu_target_processes:
        parser.error("L2 NCU requires explicit --ncu-target-processes")
    if args.level == "l2-ncu":
        if any(not metric.strip() or "," in metric for metric in args.ncu_metric):
            parser.error("--ncu-metric expects one non-empty exact metric without commas")
        if len(set(args.ncu_metric)) != len(args.ncu_metric):
            parser.error("duplicate --ncu-metric")
    elif args.ncu_qualification_kind:
        parser.error("--ncu-qualification-kind is valid only for L2 NCU")
    if args.level == "l2-ncu" and args.evidence_mode == "smoke" and \
            args.ncu_qualification_kind:
        parser.error(
            "--ncu-qualification-kind is reserved for exact qualification"
        )
    if args.level == "l2-ncu" and args.evidence_mode == "qualification" and \
            not args.ncu_qualification_kind:
        parser.error(
            "NCU qualification requires explicit --ncu-qualification-kind"
        )
    if args.level.startswith("l2-") and not args.require_data:
        parser.error("L2 requires at least one explicit --require-data NAME=RELATIVE_PATH")
    names = [name for name, _ in args.require_data]
    if len(set(names)) != len(names):
        parser.error("duplicate --require-data name")
    for _, relative in args.require_data:
        if relative.as_posix() in RESERVED_ARTIFACTS:
            parser.error("--require-data cannot reserve a harness-owned artifact name")
    if args.evidence_mode == "qualification":
        required_report = NCU_REPORT if args.level == "l2-ncu" else NSYS_REPORT
        if dict(args.require_data).get("report") != required_report:
            parser.error(
                "qualification requires --require-data "
                f"report={required_report.as_posix()}"
            )
    if args.evidence_mode == "qualification" and args.level == "l2-ncu":
        validate_ncu_qualification_inputs(parser, args)
    if args.evidence_mode == "qualification" and args.level == "l2-nsys":
        validate_accepted_step_qualification_application(parser, args)
        trace_apis = {item.strip() for item in args.nsys_trace.split(",")}
        if trace_apis != {"cuda", "nvtx"}:
            parser.error(
                "Nsys qualification requires exactly CUDA and NVTX tracing"
            )
        if args.nsys_sample != "none":
            parser.error("Nsys qualification requires --nsys-sample none")
        if args.nsys_capture_range != "cudaProfilerApi" or \
                args.nsys_capture_range_end != "stop" or \
                args.nsys_nvtx_capture:
            parser.error(
                "Nsys qualification requires accepted-step cudaProfilerApi "
                "capture with range-end stop"
            )


def build_command(args: argparse.Namespace, tool: str | None) -> list[str]:
    app_command = [str(args.exe), *args.arg]
    if args.level in {"l0", "l1"}:
        return app_command
    assert tool is not None
    stem = args.out_dir / "capture"
    if args.level == "l2-nsys":
        command = [tool, "profile", "--trace", args.nsys_trace,
                   "--sample", args.nsys_sample]
        if args.nsys_capture_range:
            command.extend(("--capture-range", args.nsys_capture_range))
        if args.nsys_capture_range_end:
            command.extend(("--capture-range-end", args.nsys_capture_range_end))
        if args.nsys_nvtx_capture:
            command.extend(("--nvtx-capture", args.nsys_nvtx_capture))
        return [*command, "--force-overwrite=true", "--output", str(stem),
                *app_command]
    command = [tool, "--target-processes", args.ncu_target_processes, "--force-overwrite", "--export", str(stem)]
    if args.ncu_set:
        command.extend(("--set", args.ncu_set))
    for section in args.ncu_section:
        command.extend(("--section", section))
    if args.ncu_metric:
        command.extend(("--metrics", ",".join(args.ncu_metric)))
    if args.ncu_kernel_name_base:
        command.extend(("--kernel-name-base", args.ncu_kernel_name_base))
    if args.ncu_kernel_name:
        command.extend(("--kernel-name", args.ncu_kernel_name))
    if args.ncu_launch_count:
        command.extend(("--launch-count", str(args.ncu_launch_count)))
    if args.ncu_launch_skip:
        command.extend(("--launch-skip", str(args.ncu_launch_skip)))
    if args.ncu_nvtx_include or args.ncu_nvtx_exclude:
        command.append("--nvtx")
    if args.ncu_nvtx_include:
        command.extend(("--nvtx-include", args.ncu_nvtx_include))
    if args.ncu_nvtx_exclude:
        command.extend(("--nvtx-exclude", args.ncu_nvtx_exclude))
    return [*command, *app_command]


def write_provenance(args: argparse.Namespace, command: list[str], tool: str | None, output: Path) -> None:
    tool_name = "none" if args.level in {"l0", "l1"} else args.level.removeprefix("l2-")
    writer_command = [sys.executable, str(PROVENANCE_WRITER), "--output", str(output), "--tool", tool_name,
                      "--source-root", str(args.source_root), "--output-stem", str(args.out_dir / "capture")]
    if tool:
        writer_command.extend(("--tool-executable", tool))
    if args.tool_version:
        writer_command.extend(("--tool-version", args.tool_version))
    for key in args.environment:
        writer_command.extend(("--environment", key))
    for label in [*args.label, f"scenario={args.scenario}", f"level={args.level}"]:
        writer_command.extend(("--label", label))
    writer_command.extend(("--command", "--", *command))
    result = subprocess.run(writer_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "provenance writer failed")
    required = ["source.root", "source.revision", "source.tree", "source.status", "capture.output_stem"]
    if args.level.startswith("l2-") and tool:
        required.append("tool.executable")
    validator_command = [sys.executable, str(PROVENANCE_VALIDATOR), "--metadata", str(output)]
    for path in required:
        validator_command.extend(("--require", path))
    result = subprocess.run(validator_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "provenance validator failed")
    return pin_provenance(output, args.out_dir)


def pin_provenance(path: Path, out_dir: Path) -> dict[str, Any]:
    data, identity = regular_snapshot(path, out_dir)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"provenance is not valid JSON: {exc}") from exc
    return {"path": path, "bytes": len(data), "sha256": sha256_bytes(data), "payload": payload,
            "identity": identity}


def unchanged_provenance(pin: dict[str, Any], out_dir: Path) -> bool:
    try:
        current = pin_provenance(pin["path"], out_dir)
    except (RuntimeError, ValueError):
        return False
    return (current["bytes"] == pin["bytes"] and current["sha256"] == pin["sha256"]
            and current["payload"] == pin["payload"] and current["identity"] == pin["identity"])


def pinned_environment(pin: dict[str, Any] | None) -> dict[str, Any]:
    if not pin:
        return {}
    environment = pin["payload"].get("environment")
    return environment if isinstance(environment, dict) else {}


def dirty_state(out_dir: Path) -> dict[str, str]:
    try:
        return json.loads((out_dir / "profile-provenance.json").read_text(encoding="utf-8"))["source"]["status"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "reason": f"provenance unavailable: {exc}"}


def recorded_environment(out_dir: Path, provenance_valid: bool) -> dict[str, Any]:
    """Return only availability facts accepted by the provenance validator."""
    if not provenance_valid:
        return {}
    try:
        environment = json.loads((out_dir / "profile-provenance.json").read_text(encoding="utf-8"))["environment"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return {}
    return environment if isinstance(environment, dict) else {}


def safe_existing(paths: list[Path], out_dir: Path) -> list[Path]:
    accepted: list[Path] = []
    for path in paths:
        try:
            regular_bytes(path, out_dir)
        except ValueError:
            continue
        accepted.append(path)
    return sorted(accepted, key=lambda path: path.relative_to(out_dir).as_posix())


def missing_requested(args: argparse.Namespace) -> list[str]:
    missing: list[str] = []
    for name, relative in args.require_data:
        try:
            if not regular_bytes(args.out_dir / relative, args.out_dir):
                missing.append(f"{name}={relative}")
        except ValueError:
            missing.append(f"{name}={relative}")
    return missing


def validate_native_report(args: argparse.Namespace) -> dict[str, Any]:
    relative = NCU_REPORT if args.level == "l2-ncu" else NSYS_REPORT
    data = regular_bytes(args.out_dir / relative, args.out_dir)
    expected_magic = (
        b"NVR\x00" if args.level == "l2-ncu"
        else b"NVIDIA Tegra Profiler Report"
    )
    if not data.startswith(expected_magic):
        raise RuntimeError(
            f"{relative} does not have the expected native "
            f"{args.level.removeprefix('l2-')} report signature"
        )
    return {
        "path": relative.as_posix(),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "signature_hex": expected_magic.hex(),
    }


def positive_float(value: str, label: str) -> float:
    try:
        parsed = float(value.replace(",", "").strip())
    except ValueError as exc:
        raise RuntimeError(f"invalid {label} value in NCU CSV: {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise RuntimeError(f"NCU {label} must be finite and positive, observed {parsed}")
    return parsed


def ncu_import_command(tool: str, report: Path) -> list[str]:
    """Import raw NCU CSV in stable PerfWorks base units."""
    return [
        tool, "--import", str(report), "--csv", "--page", "raw",
        "--print-units", "base",
    ]


def parse_ncu_tensor_csv(data: bytes, expected_launches: int,
                         kernel_substring: str) -> tuple[
        dict[str, Any], list[float]]:
    try:
        text = data.decode("utf-8", errors="strict")
        rows = list(csv.reader(io.StringIO(text)))
    except (UnicodeError, csv.Error) as exc:
        raise RuntimeError(f"could not parse NCU tensor-proof CSV: {exc}") from exc
    required = {"Kernel Name", NCU_DMMA_METRIC, NCU_DURATION_METRIC}
    header_index = next(
        (index for index, row in enumerate(rows) if required.issubset(set(row))),
        None,
    )
    if header_index is None:
        raise RuntimeError(
            "NCU raw CSV lacks kernel name, DMMA instruction, or duration column"
        )
    header = rows[header_index]
    kernel_column = header.index("Kernel Name")
    dmma_column = header.index(NCU_DMMA_METRIC)
    duration_column = header.index(NCU_DURATION_METRIC)
    duration_unit = ""
    if header_index + 1 < len(rows) and \
            len(rows[header_index + 1]) > duration_column:
        duration_unit = rows[header_index + 1][duration_column].strip()
    if duration_unit not in {"", "nsecond", "ns"}:
        raise RuntimeError(
            "NCU tensor-proof duration unit is not nanoseconds: "
            f"{duration_unit!r}"
        )
    last_column = max(kernel_column, dmma_column, duration_column)
    matched: list[dict[str, Any]] = []
    other_data_kernels: list[str] = []
    for row in rows[header_index + 1:]:
        if len(row) <= last_column:
            continue
        kernel = row[kernel_column].strip()
        if not kernel or kernel == "Kernel Name":
            continue
        if kernel_substring not in kernel:
            # Units and tool preambles are not kernel rows. A numeric metric makes
            # this a real, unexpected data row and invalidates the exact filter.
            try:
                float(row[dmma_column].replace(",", "").strip())
            except ValueError:
                continue
            other_data_kernels.append(kernel)
            continue
        matched.append({
            "kernel": kernel,
            "dmma_instructions": positive_float(
                row[dmma_column], "DMMA instruction count"
            ),
            "duration": positive_float(row[duration_column], "GPU duration"),
        })
    if other_data_kernels:
        raise RuntimeError(
            "NCU raw CSV contains kernels outside the exact hotspot filter: "
            + ", ".join(other_data_kernels[:3])
        )
    if len(matched) != expected_launches:
        raise RuntimeError(
            f"NCU raw CSV contains {len(matched)} {kernel_substring} launches; "
            f"expected {expected_launches}"
        )
    dmma_values = [row["dmma_instructions"] for row in matched]
    duration_values = [row["duration"] for row in matched]
    dmma_histogram = [
        {
            "dmma_instructions": value,
            "launches": dmma_values.count(value),
        }
        for value in sorted(set(dmma_values))
    ]
    return ({
        "status": "positive_tensor_instruction_proof",
        "kernel_substring": kernel_substring,
        "matched_launches": len(matched),
        "dmma_metric": NCU_DMMA_METRIC,
        "dmma_instruction_total": sum(dmma_values),
        "dmma_instruction_minimum": min(dmma_values),
        "dmma_instruction_maximum": max(dmma_values),
        "dmma_instruction_histogram": dmma_histogram,
        "duration_metric": NCU_DURATION_METRIC,
        "duration_unit": "nanoseconds",
        "raw_duration_unit_field": duration_unit or "empty (NCU 2024.1)",
        "duration_total": sum(duration_values),
        "duration_minimum": min(duration_values),
    }, dmma_values)


def exact_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{label} must be a nonnegative integer")
    return value


def exact_uint64(value: Any, label: str) -> int:
    result = exact_nonnegative_integer(value, label)
    if result > UINT64_MAX:
        raise RuntimeError(f"{label} exceeds uint64")
    return result


def exact_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            not math.isfinite(float(value)):
        raise RuntimeError(f"{label} must be a finite number")
    return float(value)


def _validated_fixed_sigma_jacobi_work(
        tensor: dict[str, Any],
        contract: ProfileTracePolicyContract) -> dict[str, int | float]:
    descriptor = contract.fixed_sigma_jacobi
    expected_sweeps = descriptor.sweeps if descriptor is not None else 0
    expected_damping = 2.0 / 3.0 if descriptor is not None else 0.0
    applications = exact_uint64(
        tensor.get("element_schwarz_preconditioner_applications"),
        "element-Schwarz preconditioner application count",
    )
    sweeps = exact_nonnegative_integer(
        tensor.get("field_split_sigma_jacobi_sweeps"),
        "fixed-sigma Jacobi sweep count",
    )
    damping = exact_finite_number(
        tensor.get("field_split_sigma_jacobi_damping"),
        "fixed-sigma Jacobi damping",
    )
    updates = exact_uint64(
        tensor.get("field_split_sigma_jacobi_updates"),
        "fixed-sigma Jacobi aggregate update count",
    )
    expected_updates = applications * expected_sweeps
    if expected_updates > UINT64_MAX:
        raise RuntimeError("fixed-sigma Jacobi aggregate work exceeds uint64")
    if sweeps != expected_sweeps or damping != expected_damping or \
            updates != expected_updates:
        raise RuntimeError(
            "profiled fixed-sigma Jacobi aggregate work does not exactly "
            "match the selected typed trace policy"
        )
    if contract.element_schwarz_required and applications == 0:
        raise RuntimeError(
            "profiled native element-Schwarz policy has no material "
            "preconditioner work"
        )
    result: dict[str, int | float] = {
        "element_schwarz_preconditioner_applications": applications,
        "sigma_jacobi_sweeps": sweeps,
        "sigma_jacobi_damping": damping,
        "sigma_jacobi_updates": updates,
    }
    if contract.factorization == "lower-multiplicative-uhat4-sigmahat":
        result.update({
            "field_split_sigma_jacobi_sweeps": sweeps,
            "field_split_sigma_jacobi_damping": damping,
            "field_split_sigma_jacobi_updates": updates,
        })
    return result


def application_argument(args: argparse.Namespace, option: str) -> str:
    """Return one already-validated accepted-step application argument."""
    positions = [index for index, value in enumerate(args.arg) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(args.arg):
        raise RuntimeError(f"profiled application argument {option} is unavailable")
    return args.arg[positions[0] + 1]


def profiled_tensor_work(args: argparse.Namespace) -> dict[str, Any]:
    """Validate one bounded benchmark summary from the profiled process.

    This consumes the existing finite execution log after the process exits. It
    adds no application callback, event stream, or production instrumentation.
    """
    log_path = args.out_dir / "execution.log"
    try:
        text = regular_bytes(log_path, args.out_dir).decode(
            "utf-8", errors="strict"
        )
    except UnicodeError as exc:
        raise RuntimeError(
            f"profile execution log is not UTF-8: {exc}"
        ) from exc
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        object_start = line.find("{")
        if object_start < 0:
            continue
        try:
            candidate = json.loads(line[object_start:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("benchmark") == \
                "dg_igr_hdg_cuda_accepted_step":
            records.append(candidate)
    if len(records) != 1:
        raise RuntimeError(
            "qualification requires exactly one accepted-step benchmark "
            f"summary in the execution log; observed {len(records)}"
        )
    record = records[0]
    measured_step = int(application_argument(
        args, "--profile-measured-step"
    ))
    expected_mode = application_argument(args, "--mode")
    expected_order = int(application_argument(args, "--order"))
    expected_mesh_n = int(application_argument(args, "--mesh-n"))
    expected_periods = int(application_argument(args, "--domain-periods"))
    expected_timestep = float(application_argument(args, "--dt"))
    requested_trace_policy = application_argument(args, "--trace-policy")
    trace_policy_contract = _profile_trace_policy_contract(
        requested_trace_policy, expected_mode, expected_order
    )
    expected_equilibration_passes = (
        2 if trace_policy_contract.factorization ==
        "lower-multiplicative-uhat4-sigmahat" else 0
    )
    expected_euler_rows = (
        16 * (expected_order + 1) if expected_equilibration_passes else 0
    )
    expected_sigma_rows = (
        4 * (expected_order + 1) if expected_equilibration_passes else 0
    )
    exact_record = (
        record.get("schema") == 2 and record.get("success") is True and
        record.get("variant") == trace_policy_contract.variant and
        record.get("production_selectable") is
            trace_policy_contract.production_selectable and
        record.get("backend") == "cuda" and
        record.get("cuda_engine") == "tensor" and
        record.get("trace_policy") == trace_policy_contract.requested and
        record.get("requested_trace_policy") ==
            trace_policy_contract.requested and
        record.get("resolved_trace_policy") ==
            trace_policy_contract.resolved and
        record.get("trace_preconditioner") ==
            trace_policy_contract.preconditioner and
        record.get("trace_element_schwarz_factorization") ==
            trace_policy_contract.factorization and
        record.get("trace_element_schwarz_apply") ==
            trace_policy_contract.apply and
        record.get("trace_element_schwarz_apply_tile_width") ==
            trace_policy_contract.tile_width and
        record.get("trace_element_schwarz_equilibration_passes") ==
            expected_equilibration_passes and
        record.get("trace_element_schwarz_euler_rows") ==
            expected_euler_rows and
        record.get("trace_element_schwarz_sigma_rows") ==
            expected_sigma_rows and
        record.get("mode") == expected_mode and
        record.get("polynomial_order") == expected_order and
        record.get("mesh_n") == expected_mesh_n and
        record.get("domain_periods") == expected_periods and
        record.get("dt") == expected_timestep and
        record.get("rank_count") == 1 and
        record.get("profiling_observability") is True and
        record.get("profile_measured_step") == measured_step and
        record.get("host_export") is False
    )
    if not exact_record:
        if trace_policy_contract.production_selectable:
            raise RuntimeError(
                "profiled benchmark summary does not match the qualified "
                "production tensor accepted-step selection"
            )
        raise RuntimeError(
            "profiled benchmark summary does not match the exact typed tensor "
            "accepted-step trace-policy selection"
        )
    numerical_policy = record.get("numerical_policy")
    if not isinstance(numerical_policy, dict):
        raise RuntimeError(
            "profiled benchmark summary has no reported numerical policy"
        )
    if expected_mode == "igr":
        expected_volume_order, expected_face_order = \
            FINAL_IGR_QUADRATURE_BY_ORDER[expected_order]
        observed_volume_order = exact_nonnegative_integer(
            numerical_policy.get("igr_volume_assembly_quadrature_order"),
            "reported IGR volume quadrature order",
        )
        observed_face_order = exact_nonnegative_integer(
            numerical_policy.get("igr_face_assembly_quadrature_order"),
            "reported IGR face quadrature order",
        )
        if (observed_volume_order, observed_face_order) != \
                (expected_volume_order, expected_face_order):
            raise RuntimeError(
                "profiled benchmark reported IGR quadrature policy does not "
                f"match p{expected_order}: observed "
                f"({observed_volume_order},{observed_face_order}), expected "
                f"({expected_volume_order},{expected_face_order})"
            )
    reported_role_tolerances = tuple(
        exact_finite_number(numerical_policy.get(field), f"reported {field}")
        for field in (
            "newton_correction_linear_relative_tolerance",
            "final_coherent_linear_relative_tolerance",
            "true_residual_tightening_linear_relative_tolerance",
        )
    )
    if reported_role_tolerances != \
            FINAL_LINEAR_ROLE_RELATIVE_TOLERANCES:
        raise RuntimeError(
            "profiled benchmark reported role-specific linear tolerances do "
            "not match the exact final numerical policy"
        )
    expected_sigma_sweeps = (
        trace_policy_contract.fixed_sigma_jacobi.sweeps
        if trace_policy_contract.fixed_sigma_jacobi is not None else 0
    )
    expected_sigma_damping = (
        2.0 / 3.0
        if trace_policy_contract.fixed_sigma_jacobi is not None else 0.0
    )
    generic_sigma_sweeps = exact_nonnegative_integer(
        record.get("trace_sigma_jacobi_sweeps"),
        "top-level sigma Jacobi sweep count",
    )
    generic_sigma_damping = exact_finite_number(
        record.get("trace_sigma_jacobi_damping"),
        "top-level sigma Jacobi damping",
    )
    compatibility_sigma_sweeps = exact_nonnegative_integer(
        record.get("trace_field_split_sigma_jacobi_sweeps"),
        "compatibility fixed-sigma Jacobi sweep count",
    )
    compatibility_sigma_damping = exact_finite_number(
        record.get("trace_field_split_sigma_jacobi_damping"),
        "compatibility fixed-sigma Jacobi damping",
    )
    if generic_sigma_sweeps != expected_sigma_sweeps or \
            compatibility_sigma_sweeps != generic_sigma_sweeps or \
            generic_sigma_damping != expected_sigma_damping or \
            compatibility_sigma_damping != generic_sigma_damping:
        raise RuntimeError(
            "profiled benchmark fixed-sigma Jacobi metadata does not exactly "
            "match the selected typed trace policy"
        )
    samples = record.get("samples")
    if not isinstance(samples, list):
        raise RuntimeError("profiled benchmark summary has no sample array")
    selected = [
        sample for sample in samples
        if isinstance(sample, dict) and
        sample.get("measured_index") == measured_step
    ]
    if len(selected) != 1 or selected[0].get("phase") != "measured":
        raise RuntimeError(
            "profiled benchmark summary does not identify exactly one selected "
            "measured step"
        )
    tensor = selected[0].get("tensor_core")
    if not isinstance(tensor, dict):
        raise RuntimeError("selected benchmark sample has no tensor work summary")
    condensation_launches = exact_uint64(
        tensor.get("condensation_launches"), "condensation launch count"
    )
    mma_tiles = exact_uint64(
        tensor.get("mma_tiles"), "planned MMA tile count"
    )
    volume_assembly_launches = exact_uint64(
        tensor.get("volume_assembly_launches"),
        "tensor-volume assembly launch count",
    )
    volume_mma_tiles = exact_uint64(
        tensor.get("volume_mma_tiles"),
        "planned tensor-volume MMA tile count",
    )
    if condensation_launches <= 0 or mma_tiles <= 0:
        raise RuntimeError(
            "selected benchmark sample has no material tensor condensation work"
        )
    if volume_assembly_launches <= 0 or volume_mma_tiles <= 0:
        raise RuntimeError(
            "selected benchmark sample has no material tensor-volume work"
        )
    if not (
        tensor.get("selected_engine") == "tensor" and
        tensor.get("requested_trace_policy") ==
            trace_policy_contract.requested and
        tensor.get("resolved_trace_policy") ==
            trace_policy_contract.resolved and
        tensor.get("trace_preconditioner") ==
            trace_policy_contract.preconditioner and
        tensor.get("hardware_supported") is True and
        tensor.get("fallback_count") == 0 and
        tensor.get("fallback_accounting_complete") is True and
        tensor.get("no_fallback") is True
    ):
        raise RuntimeError(
            "selected benchmark sample does not prove an admitted no-fallback "
            "tensor engine"
        )
    sigma_jacobi_work = _validated_fixed_sigma_jacobi_work(
        tensor, trace_policy_contract
    )

    elements = exact_nonnegative_integer(record.get("elements"), "element count")
    interior_components = exact_nonnegative_integer(
        record.get("interior_components"), "interior component count"
    )
    trace_components = exact_nonnegative_integer(
        record.get("trace_components"), "trace component count"
    )
    if elements <= 0 or interior_components <= 0 or trace_components <= 0:
        raise RuntimeError("profiled tensor shape is empty")
    expected_components = (4, 4) if expected_mode == "conventional" else (7, 5)
    if elements != expected_mesh_n * expected_mesh_n or \
            (interior_components, trace_components) != expected_components:
        raise RuntimeError(
            "profiled benchmark shape does not match the periodic quadrilateral "
            "Euler/IGR contract"
        )
    scalar_interior_dofs = (expected_order + 1) ** 2
    scalar_trace_dofs = 4 * (expected_order + 1)
    interior_dofs = interior_components * scalar_interior_dofs
    trace_dofs = trace_components * scalar_trace_dofs
    expected_interior_unknowns = elements * interior_dofs
    if record.get("interior_unknowns") != expected_interior_unknowns:
        raise RuntimeError(
            "profiled benchmark interior shape is inconsistent with its order, "
            "components, and element count"
        )

    volume_assembly_order = exact_nonnegative_integer(
        record.get("volume_assembly_order"), "volume assembly order"
    )
    volume_points = exact_nonnegative_integer(
        record.get("volume_points"), "volume point count"
    )
    expected_volume_assembly_order = (
        FINAL_IGR_QUADRATURE_BY_ORDER[expected_order][0]
        if expected_mode == "igr" else 2 * expected_order + 2
    )
    expected_volume_points = (expected_volume_assembly_order // 2 + 1) ** 2
    if volume_assembly_order != expected_volume_assembly_order or \
            volume_points != expected_volume_points:
        raise RuntimeError(
            "profiled benchmark volume quadrature differs from the exact "
            "production numerical policy"
        )

    def tiles(count: int, width: int) -> int:
        return (count + width - 1) // width

    # Independent test-side reproduction of the fixed fused-kernel work
    # decomposition. These are four scalar totals, not per-event telemetry.
    schur = tiles(trace_dofs, 8) ** 2 * tiles(interior_dofs, 4)
    blocked_lu = 0
    blocked_forward = 0
    blocked_backward = 0
    rhs_tiles = tiles(trace_dofs + 1, 8)
    for panel_start in range(0, interior_dofs, 8):
        panel_end = min(panel_start + 8, interior_dofs)
        panel_inner = tiles(panel_end - panel_start, 4)
        trailing_rows = tiles(interior_dofs - panel_end, 8)
        prefix_rows = tiles(panel_start, 8)
        blocked_lu += trailing_rows * trailing_rows * panel_inner
        blocked_forward += trailing_rows * rhs_tiles * panel_inner
        blocked_backward += prefix_rows * rhs_tiles * panel_inner
    per_element = {
        "blocked_lu_trailing_update": blocked_lu,
        "blocked_forward_rhs_update": blocked_forward,
        "blocked_backward_rhs_update": blocked_backward,
        "schur_contraction": schur,
    }
    if any(value <= 0 for value in per_element.values()):
        raise RuntimeError(
            "qualified p1-p3 tensor shape does not exercise every intended "
            "fused local-algebra DMMA stage"
        )
    per_launch = {
        stage: value * elements for stage, value in per_element.items()
    }
    selected_step = {
        stage: value * condensation_launches
        for stage, value in per_launch.items()
    }
    expected_mma_tiles = sum(selected_step.values())
    if mma_tiles != expected_mma_tiles:
        raise RuntimeError(
            f"benchmark planned MMA total {mma_tiles} does not match the "
            f"independent fused-stage total {expected_mma_tiles}"
        )

    # Independently reproduce the zero-allocation tensor-volume chunk plan and
    # dynamic FP64 m8n8k4 work.  The output is a bounded shape histogram, not a
    # per-launch trace.
    coefficient_stride = 66 if expected_mode == "igr" else 44
    workspace_doubles = interior_dofs * (trace_dofs + 1)
    capacity_points = workspace_doubles // coefficient_stride
    if capacity_points <= 0:
        raise RuntimeError("profiled tensor-volume workspace has zero capacity")
    chunk_points = volume_points if volume_points <= capacity_points else \
        capacity_points - capacity_points % 4
    if chunk_points <= 0 or \
            (volume_points > capacity_points and chunk_points < 4):
        raise RuntimeError(
            "profiled tensor-volume workspace cannot form an MMA chunk"
        )
    volume_base_tiles = (
        elements * interior_components * tiles(scalar_interior_dofs, 8) *
        tiles(interior_dofs + 1, 8) * 3
    )
    remaining_points = volume_points
    chunk_work: list[tuple[int, int, int]] = []
    while remaining_points:
        count = min(remaining_points, chunk_points)
        if count <= 0 or count > capacity_points or \
                (remaining_points > count and count % 4):
            raise RuntimeError(
                "profiled tensor-volume chunk plan is inconsistent"
            )
        groups = tiles(count, 4)
        chunk_work.append((count, groups, volume_base_tiles * groups))
        remaining_points -= count
    per_assembly_volume_tiles = sum(item[2] for item in chunk_work)
    expected_volume_tiles = (
        per_assembly_volume_tiles * volume_assembly_launches
    )
    volume_contraction_launches = (
        len(chunk_work) * volume_assembly_launches
    )
    if expected_volume_tiles > UINT64_MAX or \
            volume_contraction_launches > UINT64_MAX or \
            volume_mma_tiles != expected_volume_tiles:
        raise RuntimeError(
            f"benchmark planned tensor-volume MMA total {volume_mma_tiles} "
            "does not match the independent shape/chunk total "
            f"{expected_volume_tiles}"
        )
    chunk_histogram: list[dict[str, int]] = []
    for count, groups, per_launch_tiles in chunk_work:
        existing = next(
            (item for item in chunk_histogram
             if item["quadrature_points"] == count and
             item["quadrature_groups"] == groups and
             item["mma_tiles_per_launch"] == per_launch_tiles),
            None,
        )
        if existing is None:
            chunk_histogram.append({
                "quadrature_points": count,
                "quadrature_groups": groups,
                "mma_tiles_per_launch": per_launch_tiles,
                "launches_per_assembly": 1,
            })
        else:
            existing["launches_per_assembly"] += 1
    return {
        "status": "complete_profiled_tensor_work",
        "profile_measured_step": measured_step,
        "mode": expected_mode,
        "trace_policy": trace_policy_contract.requested,
        "resolved_trace_policy": trace_policy_contract.resolved,
        "trace_preconditioner": trace_policy_contract.preconditioner,
        "trace_element_schwarz_factorization":
            trace_policy_contract.factorization,
        "trace_element_schwarz_apply": trace_policy_contract.apply,
        "trace_element_schwarz_apply_tile_width":
            trace_policy_contract.tile_width,
        "promotion_eligible_trace_policy":
            trace_policy_contract.production_selectable,
        "polynomial_order": expected_order,
        "elements": elements,
        "interior_dofs_per_element": interior_dofs,
        "trace_dofs_per_element": trace_dofs,
        "condensation_launches": condensation_launches,
        "planned_mma_tiles": mma_tiles,
        "planned_mma_tiles_by_fused_stage": selected_step,
        "all_intended_fused_stages_material": True,
        "volume_assembly_order": volume_assembly_order,
        "volume_points": volume_points,
        "volume_coefficient_stride": coefficient_stride,
        "volume_workspace_capacity_points": capacity_points,
        "volume_chunk_points": chunk_points,
        "volume_assembly_launches": volume_assembly_launches,
        "volume_contraction_launches": volume_contraction_launches,
        "planned_volume_mma_tiles_per_assembly":
            per_assembly_volume_tiles,
        "planned_volume_mma_tiles": volume_mma_tiles,
        "planned_volume_contraction_launch_distribution": chunk_histogram,
        **sigma_jacobi_work,
        "fallback_count": 0,
        "execution_log": file_identity(log_path),
    }


def import_ncu_tensor_proof(args: argparse.Namespace, tool: str,
                            application_work: dict[str, Any]) -> tuple[
        dict[str, Any], list[Path], dict[Path, list[str]]]:
    report = args.out_dir / NCU_REPORT
    csv_path = args.out_dir / "ncu-tensor-proof.csv"
    stderr_path = args.out_dir / "ncu-tensor-proof.stderr.txt"
    command = ncu_import_command(tool, report)
    try:
        with open_exclusive_binary(csv_path) as stdout_stream, \
                open_exclusive_binary(stderr_path) as stderr_stream:
            result = subprocess.run(
                command, cwd=args.out_dir, stdout=stdout_stream,
                stderr=stderr_stream,
            )
    except OSError as exc:
        raise RuntimeError(f"could not import NCU report: {exc}") from exc
    if result.returncode:
        raise RuntimeError(f"NCU report import exited {result.returncode}")
    csv_data = regular_bytes(csv_path, args.out_dir)
    volume = args.ncu_qualification_kind == "tensor-volume-instructions"
    if volume:
        kernel_substring = NCU_VOLUME_KERNEL
        hotspot = "tensor-volume-contraction"
        expected_launches = int(
            application_work["volume_contraction_launches"]
        )
        expected_instructions = int(
            application_work["planned_volume_mma_tiles"]
        )
        assembly_launches = int(
            application_work["volume_assembly_launches"]
        )
        expected_values = [
            int(item["mma_tiles_per_launch"])
            for item in application_work[
                "planned_volume_contraction_launch_distribution"
            ]
            for _ in range(
                int(item["launches_per_assembly"]) * assembly_launches
            )
        ]
    else:
        kernel_substring = NCU_CONDENSATION_KERNEL
        hotspot = "fused-local-algebra-condensation"
        expected_launches = int(application_work["condensation_launches"])
        expected_instructions = int(application_work["planned_mma_tiles"])
        expected_per_launch, remainder = divmod(
            expected_instructions, expected_launches
        )
        if remainder:
            raise RuntimeError(
                "profiled condensation MMA work is not uniform by launch"
            )
        expected_values = [expected_per_launch] * expected_launches
    if args.ncu_launch_count != expected_launches:
        raise RuntimeError(
            f"NCU launch count {args.ncu_launch_count} does not match the "
            f"profiled accepted step's {expected_launches} {hotspot} launches"
        )
    proof, observed_values = parse_ncu_tensor_csv(
        csv_data, expected_launches, kernel_substring
    )
    exact_dmma_work = (
        len(expected_values) == expected_launches and
        expected_instructions <= 2 ** 53 and
        all(value <= 2 ** 53 for value in expected_values) and
        sorted(observed_values) == sorted(
            float(value) for value in expected_values
        ) and
        proof["dmma_instruction_total"] == float(expected_instructions) and
        proof["dmma_instruction_minimum"] == float(min(expected_values)) and
        proof["dmma_instruction_maximum"] == float(max(expected_values))
    )
    if not exact_dmma_work:
        raise RuntimeError(
            f"NCU {hotspot} DMMA work does not exactly match the profiled "
            "accepted step's complete planned MMA work and launch "
            "distribution: observed total/min/max "
            f"{proof['dmma_instruction_total']}/"
            f"{proof['dmma_instruction_minimum']}/"
            f"{proof['dmma_instruction_maximum']}, expected "
            f"{expected_instructions}/{min(expected_values)}/"
            f"{max(expected_values)}"
        )
    proof.update({
        "qualified_hotspot": hotspot,
        "complete_planned_dmma_match": True,
        "import_command": command,
        "report": file_identity(report),
        "csv": file_identity(csv_path),
        "import_stderr": file_identity(stderr_path),
    })
    if volume:
        proof.update({
            "volume_assembly_order":
                application_work["volume_assembly_order"],
            "volume_points": application_work["volume_points"],
            "planned_volume_mma_tiles_per_assembly":
                application_work["planned_volume_mma_tiles_per_assembly"],
            "planned_volume_contraction_launch_distribution":
                application_work[
                    "planned_volume_contraction_launch_distribution"
                ],
        })
    else:
        proof["planned_dmma_by_fused_stage"] = \
            application_work["planned_mma_tiles_by_fused_stage"]
    artifacts = [csv_path, stderr_path]
    return proof, artifacts, {path: command for path in artifacts}


def parse_ncu_attribution_csv(data: bytes,
                              launch_limit: int) -> dict[str, Any]:
    try:
        text = data.decode("utf-8", errors="strict")
        rows = list(csv.reader(io.StringIO(text)))
    except (UnicodeError, csv.Error) as exc:
        raise RuntimeError(
            f"could not parse NCU accepted-step CSV: {exc}"
        ) from exc
    required = {"Kernel Name", NCU_DURATION_METRIC}
    header_index = next(
        (index for index, row in enumerate(rows)
         if required.issubset(set(row)) and
         any("Push/Pop_Range" in value for value in row)),
        None,
    )
    if header_index is None:
        raise RuntimeError(
            "NCU raw CSV lacks kernel, NVTX push/pop, or duration columns"
        )
    header = rows[header_index]
    kernel_column = header.index("Kernel Name")
    duration_column = header.index(NCU_DURATION_METRIC)
    nvtx_columns = [
        index for index, value in enumerate(header)
        if "Push/Pop_Range" in value
    ]
    if len(nvtx_columns) != 1:
        raise RuntimeError("NCU raw CSV has ambiguous NVTX push/pop columns")
    nvtx_column = nvtx_columns[0]
    last_column = max(kernel_column, duration_column, nvtx_column)
    duration_unit = ""
    if header_index + 1 < len(rows) and \
            len(rows[header_index + 1]) > duration_column:
        duration_unit = rows[header_index + 1][duration_column].strip()
    # Nsight Compute 2024.1 leaves the raw unit cell empty for this metric;
    # other releases spell the PerfWorks base unit as nsecond or ns.
    if duration_unit not in {"", "nsecond", "ns"}:
        raise RuntimeError(
            "NCU accepted-step attribution requires nanosecond duration data; "
            f"observed unit {duration_unit!r}"
        )

    category_totals = {
        category: {"launches": 0, "duration_ns": 0.0}
        for category in NCU_ATTRIBUTION_RANGES.values()
    }
    category_totals["accepted_step_unclassified"] = {
        "launches": 0, "duration_ns": 0.0,
    }
    kernel_totals: dict[str, dict[str, Any]] = {}
    tensor_condensation_launches = 0
    tensor_condensation_duration = 0.0
    tensor_volume_launches = 0
    tensor_volume_duration = 0.0
    launches = 0
    total_duration = 0.0
    outside_ranges: list[str] = []
    for row in rows[header_index + 1:]:
        if len(row) <= last_column:
            continue
        kernel = row[kernel_column].strip()
        if not kernel or kernel == "Kernel Name":
            continue
        try:
            duration = positive_float(
                row[duration_column], "accepted-step GPU duration"
            )
        except RuntimeError:
            # Unit rows and tool preambles are not launch data.
            try:
                float(row[duration_column].replace(",", "").strip())
            except ValueError:
                continue
            raise
        ranges = row[nvtx_column]
        if "HDG/profiled-accepted-step" not in ranges:
            outside_ranges.append(kernel)
            continue
        matches = [
            (ranges.rfind(region), category)
            for region, category in NCU_ATTRIBUTION_RANGES.items()
            if region in ranges
        ]
        category = max(matches)[1] if matches \
            else "accepted_step_unclassified"
        category_totals[category]["launches"] += 1
        category_totals[category]["duration_ns"] += duration
        kernel_entry = kernel_totals.setdefault(
            kernel, {"kernel": kernel, "launches": 0, "duration_ns": 0.0}
        )
        kernel_entry["launches"] += 1
        kernel_entry["duration_ns"] += duration
        launches += 1
        total_duration += duration
        if NCU_CONDENSATION_KERNEL in kernel:
            tensor_condensation_launches += 1
            tensor_condensation_duration += duration
        if NCU_VOLUME_KERNEL in kernel:
            tensor_volume_launches += 1
            tensor_volume_duration += duration
    if outside_ranges:
        raise RuntimeError(
            "NCU attribution contains launch data outside the exact accepted-step "
            "NVTX selection: " + ", ".join(outside_ranges[:3])
        )
    # A launch limit is a profiler stop condition, not merely an upper bound.
    # Equality cannot distinguish a complete accepted-step capture from a
    # capture truncated at the declared cap, so qualification fails closed.
    if launches <= 0 or launches >= launch_limit:
        raise RuntimeError(
            f"NCU attribution contains {launches} launches; expected at least "
            f"one and strictly fewer than the declared capture limit "
            f"{launch_limit}"
        )
    missing_categories = sorted(
        category for category in NCU_REQUIRED_ATTRIBUTION_CATEGORIES
        if category_totals[category]["launches"] <= 0 or
        category_totals[category]["duration_ns"] <= 0.0
    )
    if missing_categories:
        raise RuntimeError(
            "NCU attribution is missing material accepted-step categories: "
            + ", ".join(missing_categories)
        )
    if tensor_condensation_launches <= 0 or \
            tensor_condensation_duration <= 0.0:
        raise RuntimeError(
            "NCU attribution contains no positive tensor condensation launch"
        )
    if tensor_volume_launches <= 0 or tensor_volume_duration <= 0.0:
        raise RuntimeError(
            "NCU attribution contains no positive tensor-volume contraction "
            "launch"
        )
    categories = []
    for category, values in sorted(category_totals.items()):
        duration = float(values["duration_ns"])
        categories.append({
            "category": category,
            "launches": int(values["launches"]),
            "duration_ns": duration,
            "fraction_of_gpu_kernel_duration": duration / total_duration,
        })
    kernels = sorted(
        kernel_totals.values(), key=lambda value: (-value["duration_ns"],
                                                   value["kernel"])
    )
    for value in kernels:
        value["fraction_of_gpu_kernel_duration"] = \
            value["duration_ns"] / total_duration
    return {
        "status": "positive_accepted_step_attribution",
        "duration_metric": NCU_DURATION_METRIC,
        "duration_unit": "nanoseconds",
        "raw_duration_unit_field": duration_unit or "empty (NCU 2024.1)",
        "capture_launch_limit": launch_limit,
        "matched_launches": launches,
        "total_gpu_kernel_duration_ns": total_duration,
        "categories": categories,
        "required_categories": sorted(NCU_REQUIRED_ATTRIBUTION_CATEGORIES),
        "tensor_condensation": {
            "kernel_substring": NCU_CONDENSATION_KERNEL,
            "launches": tensor_condensation_launches,
            "duration_ns": tensor_condensation_duration,
            "fraction_of_gpu_kernel_duration":
                tensor_condensation_duration / total_duration,
        },
        "tensor_volume_assembly": {
            "kernel_substring": NCU_VOLUME_KERNEL,
            "launches": tensor_volume_launches,
            "duration_ns": tensor_volume_duration,
            "fraction_of_gpu_kernel_duration":
                tensor_volume_duration / total_duration,
        },
        "kernel_totals": kernels,
        "scope_limitations": {
            "gpu_kernel_duration_only": True,
            "cuda_api_and_memcpy_time_requires_nsys": True,
            "host_and_mpi_time_not_included": True,
        },
    }


def import_ncu_attribution(args: argparse.Namespace, tool: str,
                           application_work: dict[str, Any]) -> tuple[
        dict[str, Any], list[Path], dict[Path, list[str]]]:
    report = args.out_dir / NCU_REPORT
    csv_path = args.out_dir / "ncu-accepted-step-attribution.csv"
    stderr_path = args.out_dir / "ncu-accepted-step-attribution.stderr.txt"
    command = ncu_import_command(tool, report)
    try:
        with open_exclusive_binary(csv_path) as stdout_stream, \
                open_exclusive_binary(stderr_path) as stderr_stream:
            result = subprocess.run(
                command, cwd=args.out_dir, stdout=stdout_stream,
                stderr=stderr_stream,
            )
    except OSError as exc:
        raise RuntimeError(f"could not import NCU report: {exc}") from exc
    if result.returncode:
        raise RuntimeError(f"NCU report import exited {result.returncode}")
    csv_data = regular_bytes(csv_path, args.out_dir)
    attribution = parse_ncu_attribution_csv(
        csv_data, args.ncu_launch_count
    )
    expected_tensor_launches = int(application_work["condensation_launches"])
    if attribution["tensor_condensation"]["launches"] != \
            expected_tensor_launches:
        raise RuntimeError(
            "NCU accepted-step attribution tensor launch count does not match "
            f"the profiled benchmark: observed "
            f"{attribution['tensor_condensation']['launches']}, expected "
            f"{expected_tensor_launches}"
        )
    expected_volume_launches = int(
        application_work["volume_contraction_launches"]
    )
    if attribution["tensor_volume_assembly"]["launches"] != \
            expected_volume_launches:
        raise RuntimeError(
            "NCU accepted-step attribution tensor-volume launch count does "
            "not match the profiled benchmark: observed "
            f"{attribution['tensor_volume_assembly']['launches']}, expected "
            f"{expected_volume_launches}"
        )
    attribution.update({
        "import_command": command,
        "report": file_identity(report),
        "csv": file_identity(csv_path),
        "import_stderr": file_identity(stderr_path),
    })
    artifacts = [csv_path, stderr_path]
    return attribution, artifacts, {path: command for path in artifacts}


def import_nsys_summaries(args: argparse.Namespace, tool: str) -> tuple[
        dict[str, Any], list[Path], dict[Path, list[str]]]:
    """Derive bounded, static tables from the already captured native report.

    These files are profiler-report summaries, not application instrumentation:
    no callback, event stream, or production recorder is installed.
    """
    report = args.out_dir / NSYS_REPORT
    summaries: dict[str, Any] = {}
    artifacts: list[Path] = []
    artifact_commands: dict[Path, list[str]] = {}
    for report_name, artifact_stem in NSYS_QUALIFICATION_SUMMARIES:
        csv_path = args.out_dir / f"{artifact_stem}.csv"
        stderr_path = args.out_dir / f"{artifact_stem}.stderr.txt"
        command = [
            tool, "stats", "--report", report_name,
            "--format", "csv", "--output", "-",
            "--force-export", "true", str(report),
        ]
        try:
            with open_exclusive_binary(csv_path) as stdout_stream, \
                    open_exclusive_binary(stderr_path) as stderr_stream:
                result = subprocess.run(
                    command, cwd=args.out_dir, stdout=stdout_stream,
                    stderr=stderr_stream,
                )
        except OSError as exc:
            raise RuntimeError(
                f"could not generate Nsys {report_name} summary: {exc}"
            ) from exc
        if result.returncode:
            raise RuntimeError(
                f"Nsys {report_name} summary exited {result.returncode}"
            )
        csv_data = regular_bytes(csv_path, args.out_dir)
        if not csv_data.strip():
            raise RuntimeError(f"Nsys {report_name} summary is empty")
        summaries[report_name] = {
            "status": "generated_from_native_report",
            "command": command,
            "report": file_identity(report),
            "csv": file_identity(csv_path),
            "stderr": file_identity(stderr_path),
        }
        artifacts.extend((csv_path, stderr_path))
        artifact_commands[csv_path] = command
        artifact_commands[stderr_path] = command
    return summaries, artifacts, artifact_commands


def qualification_evidence_unchanged(args: argparse.Namespace,
                                     qualification: dict[str, Any] | None
                                     ) -> tuple[bool, str]:
    if not qualification or qualification.get("status") != "success":
        return True, ""
    try:
        current_report = validate_native_report(args)
        if current_report != qualification.get("native_report"):
            return False, "native profiler report changed after validation"
        application_work = qualification.get("application_work", {})
        execution_log = application_work.get("execution_log")
        if not isinstance(execution_log, dict) or \
                file_identity(Path(execution_log.get("path", ""))) != \
                execution_log:
            return False, "profiled benchmark execution log changed after validation"
        if args.level == "l2-ncu":
            instruction_kind = args.ncu_qualification_kind in \
                NCU_INSTRUCTION_QUALIFICATION_KINDS
            proof_name = (
                "tensor_instruction_proof"
                if instruction_kind
                else "accepted_step_attribution"
            )
            proof = qualification.get(proof_name, {})
            for key in ("report", "csv", "import_stderr"):
                identity = proof.get(key)
                if not isinstance(identity, dict) or \
                        file_identity(Path(identity.get("path", ""))) != identity:
                    return False, f"NCU {proof_name} {key} changed after validation"
        elif args.level == "l2-nsys":
            summaries = qualification.get("nsys_summaries", {})
            if set(summaries) != {
                    report for report, _ in NSYS_QUALIFICATION_SUMMARIES}:
                return False, "Nsys qualification summary set is incomplete"
            for report_name, summary in summaries.items():
                for key in ("report", "csv", "stderr"):
                    identity = summary.get(key)
                    if not isinstance(identity, dict) or \
                            file_identity(
                                Path(identity.get("path", ""))
                            ) != identity:
                        return False, (
                            f"Nsys {report_name} {key} changed after validation"
                        )
    except (OSError, RuntimeError, ValueError) as exc:
        return False, f"could not revalidate qualification evidence: {exc}"
    return True, ""


def make_manifest(args: argparse.Namespace, command: list[str], tool: str | None,
                  outcome: str, detail: str, artifacts: list[Path],
                  timings: list[float], pin: dict[str, Any] | None,
                  execution_identity: dict[str, Any] | None = None,
                  qualification: dict[str, Any] | None = None,
                  artifact_commands: dict[Path, list[str]] | None = None) -> dict[str, Any]:
    selected_environment = pinned_environment(pin)
    requested = [{"name": name, "path": path.as_posix()} for name, path in sorted(args.require_data)]
    existing = safe_existing(artifacts, args.out_dir)
    executable_identity = (
        execution_identity["executable"]
        if execution_identity is not None
        else file_identity(args.exe)
    )
    timing_statistics: dict[str, float | int] = {"measured_repetitions": len(timings)}
    if timings:
        timing_statistics.update({"median_seconds": statistics.median(timings),
                                  "dispersion_seconds": statistics.pstdev(timings) if len(timings) > 1 else 0.0})
    artifact_commands = artifact_commands or {}
    qualification_payload = qualification or {
        "status": "not_performed",
        "reason": "smoke or non-L2 evidence mode",
    }
    if args.evidence_mode == "qualification" and outcome != "success":
        qualification_payload = {
            "status": "failure",
            "reason": detail,
            "partial_evidence": qualification_payload,
        }
    return {"schema_version": 1, "scenario": args.scenario, "level": args.level,
            "evidence_mode": args.evidence_mode or "not_applicable",
            "ncu_qualification_kind": (
                args.ncu_qualification_kind or "not_applicable"
            ),
            "outcome": outcome, "detail": detail,
            "generating_command": command, "generating_command_display": quoted(command),
            "identity": {"tool": args.level.removeprefix("l2-") if args.level.startswith("l2-") else "none",
                         "tool_executable": tool or "unavailable", "tool_version": args.tool_version or "unavailable",
                         "executable": executable_identity,
                         "build_manifest": (
                             execution_identity["build_manifest"]
                             if execution_identity is not None
                             else {"status": "unavailable", "reason": "not pinned"}
                         ),
                         "profiler_executable": (
                             execution_identity["profiler_executable"]
                             if execution_identity is not None
                             else {"status": "unavailable", "reason": "not pinned"}
                         ),
                         "source_binding": (
                             {key: execution_identity[key] for key in (
                                 "source", "expected_commit", "expected_tree",
                                 "clean_source_required", "exact_candidate_required"
                             )}
                             if execution_identity is not None
                             else {"status": "unavailable", "reason": "not pinned"}
                         ),
                         "qualification": (
                             execution_identity["qualification"]
                             if execution_identity is not None
                             else {"status": "unavailable", "reason": "not pinned"}
                         ),
                         "mfem": args.mfem_identity or "unavailable",
                         "gpu": args.gpu_identity or "unavailable"},
            "environment": selected_environment, "requested_data": requested, "source_dirty_state": pin["payload"].get("source", {}).get("status", {"status": "unavailable", "reason": "no pinned provenance"}) if pin else {"status": "unavailable", "reason": "no pinned provenance"},
            "timing_seconds": timings, "timing_statistics": timing_statistics,
            "qualification": qualification_payload,
            "artifacts": [
                artifact_entry(
                    path, args.out_dir, artifact_commands.get(path, command)
                ) for path in existing
            ]}


def execute(command: list[str], log_path: Path, warmups: int, repetitions: int) -> tuple[int, list[float], str]:
    """Run explicit warm-ups, then record only measured repetition durations."""
    timings: list[float] = []
    try:
        descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError as exc:
        return 127, timings, f"could not create execution log: {exc}"
    with os.fdopen(descriptor, "w", encoding="utf-8") as log:
        for _ in range(warmups):
            try:
                result = subprocess.run(command, cwd=log_path.parent, stdout=log, stderr=subprocess.STDOUT)
            except OSError as exc:
                return 127, timings, f"could not execute command: {exc}"
            if result.returncode:
                return result.returncode, timings, ""
        for _ in range(repetitions):
            started = time.perf_counter()
            try:
                result = subprocess.run(command, cwd=log_path.parent, stdout=log, stderr=subprocess.STDOUT)
            except OSError as exc:
                return 127, timings, f"could not execute command: {exc}"
            timings.append(time.perf_counter() - started)
            if result.returncode:
                return result.returncode, timings, ""
    return 0, timings, ""


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.exe = args.exe.resolve()
    args.out_dir = args.out_dir.resolve()
    args.source_root = args.source_root.resolve()
    if args.build_manifest is not None:
        supplied_build_manifest = args.build_manifest.absolute()
        if supplied_build_manifest.is_symlink():
            parser.error("--build-manifest must not be a symlink")
        args.build_manifest = supplied_build_manifest.resolve()
    if os.path.sep in args.tool_executable:
        args.tool_executable = str(Path(args.tool_executable).resolve())
    if os.path.sep in args.nvidia_smi_executable:
        args.nvidia_smi_executable = str(
            Path(args.nvidia_smi_executable).resolve()
        )
    args.resolved_nvidia_smi = tool_path(args.nvidia_smi_executable)
    validate_inputs(parser, args)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    tool = tool_path(args.tool_executable) if args.level.startswith("l2-") else None
    args.resolved_tool = tool
    command = build_command(args, tool or args.tool_executable if args.level.startswith("l2-") else None)
    provenance = args.out_dir / "profile-provenance.json"
    execution_identity: dict[str, Any] | None = None
    try:
        pin = write_provenance(args, command, tool, provenance)
        execution_identity = capture_execution_identity(args, pin, tool)
    except (OSError, RuntimeError) as exc:
        print(f"profile-harness: provenance validation failure: {exc}", file=sys.stderr)
        try:
            write_exclusive_json(args.out_dir / "profile-manifest.json",
                                 make_manifest(args, command, tool, "validation_failure", str(exc), [provenance], [], None))
        except OSError:
            pass
        return EXIT_VALIDATION_FAILURE
    artifacts: list[Path] = [provenance]
    timings: list[float] = []
    qualification: dict[str, Any] | None = None
    artifact_commands: dict[Path, list[str]] = {}
    command_executed = False
    outcome, detail, exit_code = "dry_run" if args.dry_run else "success", "dry-run command constructed" if args.dry_run else "command completed", 0
    if args.level.startswith("l2-") and not tool and not args.dry_run:
        outcome, detail, exit_code = "tool_unavailable", f"tool executable not found: {args.tool_executable}", EXIT_TOOL_UNAVAILABLE
    elif not args.dry_run:
        command_executed = True
        log_path = args.out_dir / "execution.log"
        artifacts.append(log_path)
        returncode, timings, execution_detail = execute(command, log_path, args.warmups, args.repetitions or 1)
        if returncode:
            outcome, detail, exit_code = "execution_failure", execution_detail or f"command exited {returncode}", EXIT_EXECUTION_FAILURE
        elif args.level.startswith("l2-"):
            missing = missing_requested(args)
            artifacts.extend(args.out_dir / relative for _, relative in args.require_data)
            if missing:
                outcome, detail, exit_code = "data_unavailable", "missing requested data: " + ", ".join(missing), EXIT_DATA_UNAVAILABLE
            else:
                detail = "command completed with all requested data present"
                if args.evidence_mode == "qualification":
                    try:
                        native_report = validate_native_report(args)
                        application_work = profiled_tensor_work(args)
                        qualification = {
                            "status": "native_report_validated",
                            "native_report": native_report,
                            "application_work": application_work,
                        }
                        if args.level == "l2-ncu":
                            assert tool is not None
                            if args.ncu_qualification_kind in \
                                    NCU_INSTRUCTION_QUALIFICATION_KINDS:
                                proof_paths = [
                                    args.out_dir / "ncu-tensor-proof.csv",
                                    args.out_dir / "ncu-tensor-proof.stderr.txt",
                                ]
                            else:
                                proof_paths = [
                                    args.out_dir /
                                    "ncu-accepted-step-attribution.csv",
                                    args.out_dir /
                                    "ncu-accepted-step-attribution.stderr.txt",
                                ]
                            import_command = ncu_import_command(
                                tool, args.out_dir / NCU_REPORT
                            )
                            artifacts.extend(proof_paths)
                            artifact_commands.update(
                                {path: import_command for path in proof_paths}
                            )
                            if args.ncu_qualification_kind in \
                                    NCU_INSTRUCTION_QUALIFICATION_KINDS:
                                tensor_proof, _, _ = import_ncu_tensor_proof(
                                    args, tool, application_work
                                )
                                qualification = {
                                    "status": "success",
                                    "native_report": native_report,
                                    "application_work": application_work,
                                    "tensor_instruction_proof": tensor_proof,
                                }
                                qualified_hotspot = tensor_proof[
                                    "qualified_hotspot"
                                ]
                                detail = (
                                    "qualified native NCU report proves "
                                    "complete planned DMMA execution in the "
                                    f"{qualified_hotspot} kernel"
                                )
                            else:
                                attribution, _, _ = import_ncu_attribution(
                                    args, tool, application_work
                                )
                                qualification = {
                                    "status": "success",
                                    "native_report": native_report,
                                    "application_work": application_work,
                                    "accepted_step_attribution": attribution,
                                }
                                detail = (
                                    "qualified native NCU report attributes the "
                                    "accepted-step GPU kernel duration and the "
                                    "tensor volume/condensation fractions"
                                )
                        else:
                            assert tool is not None
                            summaries, summary_paths, summary_commands = \
                                import_nsys_summaries(args, tool)
                            artifacts.extend(summary_paths)
                            artifact_commands.update(summary_commands)
                            qualification = {
                                "status": "success",
                                "native_report": native_report,
                                "application_work": application_work,
                                "nsys_summaries": summaries,
                            }
                            detail = (
                                "qualified native Nsight Systems report with "
                                "hashed NVTX, CUDA API, and GPU memory-operation "
                                "summaries"
                            )
                    except (OSError, RuntimeError, ValueError) as exc:
                        outcome, detail, exit_code = (
                            "validation_failure",
                            f"qualification evidence validation failed: {exc}",
                            EXIT_VALIDATION_FAILURE,
                        )
    else:
        artifacts.extend(args.out_dir / relative for _, relative in args.require_data)
    if not args.dry_run and not unchanged_provenance(pin, args.out_dir):
        outcome, detail, exit_code = "validation_failure", "pinned provenance changed or disappeared during execution", EXIT_VALIDATION_FAILURE
        artifacts = [path for path in artifacts if path != provenance]
    if command_executed:
        unchanged, reason = execution_identity_unchanged(args, execution_identity)
        if not unchanged:
            outcome, detail, exit_code = "validation_failure", reason, EXIT_VALIDATION_FAILURE
    if args.evidence_mode == "qualification":
        unchanged, reason = qualification_evidence_unchanged(args, qualification)
        if not unchanged:
            outcome, detail, exit_code = (
                "validation_failure", reason, EXIT_VALIDATION_FAILURE
            )
    manifest_path = args.out_dir / "profile-manifest.json"
    try:
        write_exclusive_json(
            manifest_path,
            make_manifest(args, command, tool, outcome, detail, artifacts,
                          timings, pin, execution_identity, qualification,
                          artifact_commands),
        )
    except (OSError, RuntimeError) as exc:
        print(f"profile-harness: manifest write failure: {exc}", file=sys.stderr)
        return EXIT_VALIDATION_FAILURE
    print(f"profile-harness: outcome={outcome} manifest={manifest_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
