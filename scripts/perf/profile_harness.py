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
from typing import Any

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
NCU_DMMA_METRIC = "sm__inst_executed_pipe_fp64_op_dmma.sum"
NCU_DURATION_METRIC = "gpu__time_duration.avg"
NCU_CONDENSATION_KERNEL = "TensorBorrowedCondenseKernel"
NCU_CONDENSATION_FILTER = f"regex:.*{NCU_CONDENSATION_KERNEL}.*"
NCU_CONDENSATION_NVTX_RANGE = "HDG/condensation/"
NCU_ACCEPTED_STEP_NVTX_RANGE = "HDG/profiled-accepted-step/"
PRODUCTION_TRACE_POLICY = "engine-default"
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
        choices=("tensor-instructions", "accepted-step-attribution"),
        default="",
        help=("Required for NCU qualification: exact tensor instruction proof "
              "or duration-only accepted-step category attribution."),
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
    if required_application_value(parser, args, "--mode") not in {
            "conventional", "igr"}:
        parser.error("profiler qualification requires conventional or IGR mode")
    order = required_application_value(parser, args, "--order")
    if order not in {"1", "2", "3"}:
        parser.error("profiler qualification requires application order 1, 2, or 3")
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
    if args.ncu_qualification_kind == "tensor-instructions":
        if set(args.ncu_metric) != {NCU_DMMA_METRIC, NCU_DURATION_METRIC} or \
                len(args.ncu_metric) != 2:
            parser.error(
                "tensor-instruction qualification requires exactly the DMMA "
                "instruction and GPU duration metrics"
            )
        if args.ncu_kernel_name != NCU_CONDENSATION_FILTER:
            parser.error(
                "tensor-instruction qualification requires the exact tensor "
                "condensation kernel filter"
            )
        if normalized_nvtx != NCU_CONDENSATION_NVTX_RANGE or \
                args.ncu_nvtx_exclude:
            parser.error(
                "tensor-instruction qualification requires only the HDG "
                "condensation NVTX include range"
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


def parse_ncu_tensor_csv(data: bytes, expected_launches: int) -> dict[str, Any]:
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
        if NCU_CONDENSATION_KERNEL not in kernel:
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
            "NCU raw CSV contains kernels outside the exact condensation filter: "
            + ", ".join(other_data_kernels[:3])
        )
    if len(matched) != expected_launches:
        raise RuntimeError(
            f"NCU raw CSV contains {len(matched)} tensor condensation launches; "
            f"expected {expected_launches}"
        )
    dmma_values = [row["dmma_instructions"] for row in matched]
    duration_values = [row["duration"] for row in matched]
    return {
        "status": "positive_tensor_instruction_proof",
        "kernel_substring": NCU_CONDENSATION_KERNEL,
        "matched_launches": len(matched),
        "dmma_metric": NCU_DMMA_METRIC,
        "dmma_instruction_total": sum(dmma_values),
        "dmma_instruction_minimum": min(dmma_values),
        "duration_metric": NCU_DURATION_METRIC,
        "duration_unit": "nanoseconds",
        "raw_duration_unit_field": duration_unit or "empty (NCU 2024.1)",
        "duration_total": sum(duration_values),
        "duration_minimum": min(duration_values),
    }


def import_ncu_tensor_proof(args: argparse.Namespace, tool: str) -> tuple[
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
    proof = parse_ncu_tensor_csv(csv_data, args.ncu_launch_count)
    proof.update({
        "import_command": command,
        "report": file_identity(report),
        "csv": file_identity(csv_path),
        "import_stderr": file_identity(stderr_path),
    })
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
    tensor_launches = 0
    tensor_duration = 0.0
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
            tensor_launches += 1
            tensor_duration += duration
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
    if tensor_launches <= 0 or tensor_duration <= 0.0:
        raise RuntimeError(
            "NCU attribution contains no positive tensor condensation launch"
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
            "launches": tensor_launches,
            "duration_ns": tensor_duration,
            "fraction_of_gpu_kernel_duration": tensor_duration / total_duration,
        },
        "kernel_totals": kernels,
        "scope_limitations": {
            "gpu_kernel_duration_only": True,
            "cuda_api_and_memcpy_time_requires_nsys": True,
            "host_and_mpi_time_not_included": True,
        },
    }


def import_ncu_attribution(args: argparse.Namespace, tool: str) -> tuple[
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
        if args.level == "l2-ncu":
            proof_name = (
                "tensor_instruction_proof"
                if args.ncu_qualification_kind == "tensor-instructions"
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
                        qualification = {
                            "status": "native_report_validated",
                            "native_report": native_report,
                        }
                        if args.level == "l2-ncu":
                            assert tool is not None
                            if args.ncu_qualification_kind == \
                                    "tensor-instructions":
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
                            if args.ncu_qualification_kind == \
                                    "tensor-instructions":
                                tensor_proof, _, _ = import_ncu_tensor_proof(
                                    args, tool
                                )
                                qualification = {
                                    "status": "success",
                                    "native_report": native_report,
                                    "tensor_instruction_proof": tensor_proof,
                                }
                                detail = (
                                    "qualified native NCU report proves positive "
                                    "DMMA execution in tensor condensation kernel"
                                )
                            else:
                                attribution, _, _ = import_ncu_attribution(
                                    args, tool
                                )
                                qualification = {
                                    "status": "success",
                                    "native_report": native_report,
                                    "accepted_step_attribution": attribution,
                                }
                                detail = (
                                    "qualified native NCU report attributes the "
                                    "accepted-step GPU kernel duration and tensor "
                                    "condensation fraction"
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
