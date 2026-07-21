#!/usr/bin/env python3
"""Run an explicit L0--L2 profiling scenario and write auditable artifacts.

The runner never supplies physics, backend, device, MPI, or profiler defaults.
Callers provide an executable plus exact arguments, a scenario label, and an
output directory. L0 runs one smoke command, L1 repeats that command and
records timing, and L2 constructs an Nsight Systems or Nsight Compute command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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
EXIT_VALIDATION_FAILURE = 6


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


def quoted(command: list[str]) -> str:
    return " ".join(json.dumps(item) for item in command)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_entry(path: Path, out_dir: Path, command: list[str]) -> dict[str, Any]:
    return {"path": path.relative_to(out_dir).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path), "generating_command": command}


def tool_path(value: str) -> str | None:
    if not value:
        return None
    if os.path.sep in value:
        return str(Path(value).resolve()) if Path(value).is_file() and os.access(value, os.X_OK) else None
    return shutil.which(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("l0", "l1", "l2-nsys", "l2-ncu"), required=True)
    parser.add_argument("--scenario", required=True, help="Caller-owned scenario label; no physics defaults are inferred.")
    parser.add_argument("--exe", required=True, type=Path, help="Exact application executable.")
    parser.add_argument("--arg", action="append", default=[], help="One exact application argument; repeat as needed.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--environment", action="append", default=[], metavar="KEY", help="Allowlisted environment variable copied into provenance.")
    parser.add_argument("--label", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--mfem-identity", default="")
    parser.add_argument("--gpu-identity", default="")
    parser.add_argument("--tool-executable", default="", help="Required for L2; fake tools work in no-capture tests.")
    parser.add_argument("--tool-version", default="")
    parser.add_argument("--repetitions", type=int, default=0, help="L1 measured repetitions; must be 5 through 7.")
    parser.add_argument("--warmups", type=int, default=0, help="L1 warm-up repetitions; exactly one is required.")
    parser.add_argument("--nsys-trace", default="", help="Required L2 Nsys trace set.")
    parser.add_argument("--nsys-sample", default="", help="Required L2 Nsys sample mode.")
    parser.add_argument("--ncu-set", default="", help="Explicit NCU metric set.")
    parser.add_argument("--ncu-section", action="append", default=[], help="Explicit NCU section; repeat as needed.")
    parser.add_argument("--ncu-kernel-name", default="", help="Explicit NCU kernel-name filter.")
    parser.add_argument("--ncu-kernel-name-base", default="", help="Explicit NCU kernel-name-base semantics.")
    parser.add_argument("--ncu-launch-count", type=int, default=0)
    parser.add_argument("--ncu-launch-skip", type=int, default=0)
    parser.add_argument("--ncu-nvtx-include", default="", help="Explicit NCU NVTX include filter.")
    parser.add_argument("--ncu-nvtx-exclude", default="", help="Explicit NCU NVTX exclusion filter.")
    parser.add_argument("--require-data", action="append", type=parse_key_path, default=[], metavar="NAME=RELATIVE_PATH", help="Tool-produced report/table required after L2 execution.")
    parser.add_argument("--dry-run", action="store_true", help="Write provenance/manifest and print command without execution.")
    return parser


def validate_inputs(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.scenario.strip():
        parser.error("--scenario must be non-empty")
    if not args.exe.is_file() or not os.access(args.exe, os.X_OK):
        parser.error(f"--exe is not an executable file: {args.exe}")
    if args.out_dir.exists():
        parser.error(f"--out-dir already exists: {args.out_dir}")
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
    if args.level == "l2-ncu" and not (args.ncu_set or args.ncu_section):
        parser.error("L2 NCU requires explicit --ncu-set or at least one --ncu-section")
    if args.level.startswith("l2-") and not args.require_data:
        parser.error("L2 requires at least one explicit --require-data NAME=RELATIVE_PATH")
    names = [name for name, _ in args.require_data]
    if len(set(names)) != len(names):
        parser.error("duplicate --require-data name")
    for _, relative in args.require_data:
        if relative in {Path("profile-manifest.json"), Path("profile-provenance.json")}:
            parser.error("--require-data cannot reserve a harness-owned artifact name")


def build_command(args: argparse.Namespace, tool: str | None) -> list[str]:
    app_command = [str(args.exe), *args.arg]
    if args.level in {"l0", "l1"}:
        return app_command
    assert tool is not None
    stem = args.out_dir / "capture"
    if args.level == "l2-nsys":
        return [tool, "profile", "--trace", args.nsys_trace, "--sample", args.nsys_sample,
                "--force-overwrite=true", "--output", str(stem), *app_command]
    command = [tool, "--target-processes", "application-only", "--force-overwrite", "--export", str(stem)]
    if args.ncu_set:
        command.extend(("--set", args.ncu_set))
    for section in args.ncu_section:
        command.extend(("--section", section))
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


def make_manifest(args: argparse.Namespace, command: list[str], tool: str | None, outcome: str, detail: str,
                  artifacts: list[Path], timings: list[float], provenance_valid: bool) -> dict[str, Any]:
    selected_environment = recorded_environment(args.out_dir, provenance_valid)
    requested = [{"name": name, "path": path.as_posix()} for name, path in sorted(args.require_data)]
    existing = sorted((path for path in artifacts if path.is_file()), key=lambda path: path.relative_to(args.out_dir).as_posix())
    executable_identity = {"path": str(args.exe), "bytes": args.exe.stat().st_size, "sha256": sha256(args.exe)}
    timing_statistics: dict[str, float | int] = {"measured_repetitions": len(timings)}
    if timings:
        timing_statistics.update({"median_seconds": statistics.median(timings),
                                  "dispersion_seconds": statistics.pstdev(timings) if len(timings) > 1 else 0.0})
    return {"schema_version": 1, "scenario": args.scenario, "level": args.level, "outcome": outcome, "detail": detail,
            "generating_command": command, "generating_command_display": quoted(command),
            "identity": {"tool": args.level.removeprefix("l2-") if args.level.startswith("l2-") else "none",
                         "tool_executable": tool or "unavailable", "tool_version": args.tool_version or "unavailable",
                         "executable": executable_identity, "mfem": args.mfem_identity or "unavailable", "gpu": args.gpu_identity or "unavailable"},
            "environment": selected_environment, "requested_data": requested, "source_dirty_state": dirty_state(args.out_dir),
            "timing_seconds": timings, "timing_statistics": timing_statistics,
            "artifacts": [artifact_entry(path, args.out_dir, command) for path in existing]}


def execute(command: list[str], log_path: Path, warmups: int, repetitions: int) -> tuple[int, list[float], str]:
    """Run explicit warm-ups, then record only measured repetition durations."""
    timings: list[float] = []
    with log_path.open("w", encoding="utf-8") as log:
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
    if os.path.sep in args.tool_executable:
        args.tool_executable = str(Path(args.tool_executable).resolve())
    validate_inputs(parser, args)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    tool = tool_path(args.tool_executable) if args.level.startswith("l2-") else None
    command = build_command(args, tool or args.tool_executable if args.level.startswith("l2-") else None)
    provenance = args.out_dir / "profile-provenance.json"
    try:
        write_provenance(args, command, tool, provenance)
    except RuntimeError as exc:
        print(f"profile-harness: provenance validation failure: {exc}", file=sys.stderr)
        write_json(args.out_dir / "profile-manifest.json",
                   make_manifest(args, command, tool, "validation_failure", str(exc), [provenance], [], False))
        return EXIT_VALIDATION_FAILURE
    artifacts: list[Path] = [provenance]
    timings: list[float] = []
    outcome, detail, exit_code = "dry_run" if args.dry_run else "success", "dry-run command constructed" if args.dry_run else "command completed", 0
    if args.level.startswith("l2-") and not tool and not args.dry_run:
        outcome, detail, exit_code = "tool_unavailable", f"tool executable not found: {args.tool_executable}", EXIT_TOOL_UNAVAILABLE
    elif not args.dry_run:
        log_path = args.out_dir / "execution.log"
        artifacts.append(log_path)
        returncode, timings, execution_detail = execute(command, log_path, args.warmups, args.repetitions or 1)
        if returncode:
            outcome, detail, exit_code = "execution_failure", execution_detail or f"command exited {returncode}", EXIT_EXECUTION_FAILURE
        elif args.level.startswith("l2-"):
            missing = [f"{name}={relative}" for name, relative in args.require_data
                       if not (args.out_dir / relative).is_file() or (args.out_dir / relative).stat().st_size == 0]
            artifacts.extend(args.out_dir / relative for _, relative in args.require_data)
            if missing:
                outcome, detail, exit_code = "data_unavailable", "missing requested data: " + ", ".join(missing), EXIT_DATA_UNAVAILABLE
            else:
                detail = "command completed with all requested data present"
    else:
        artifacts.extend(args.out_dir / relative for _, relative in args.require_data)
    manifest_path = args.out_dir / "profile-manifest.json"
    write_json(manifest_path, make_manifest(args, command, tool, outcome, detail, artifacts, timings, True))
    print(f"profile-harness: outcome={outcome} manifest={manifest_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
