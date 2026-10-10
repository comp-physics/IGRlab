#!/usr/bin/env python3
"""Qualify whole-device peak memory for the production tensor CUDA engine.

This is deliberately a post-timing control.  It polls ``nvidia-smi`` while
the accepted-step benchmark runs, so none of the benchmark durations emitted
here are eligible for the primary performance matrix.

The driver does not configure, build, acquire, or select a GPU.  Its caller
must hold the campaign's one exclusive GPU lease and must expose exactly the
selected physical device through ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shlex
import sys
from types import ModuleType
from typing import Any, Sequence


# Loading the two tracked sibling validators must not dirty an exact candidate
# by creating an untracked ``__pycache__`` during qualification.
sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
QUALIFICATION_WORKLOAD = {
    "label": "sm120-mesh52-saturated",
    "mesh_n": 52,
    "domain_periods": 1,
    "dt": 1.0e-5,
    "repeats": 3,
    "poll_ms": 50,
}
CELLS = tuple((mode, order) for mode in ("conventional", "igr")
              for order in (1, 2, 3))


def _load_sibling(filename: str, module_name: str) -> ModuleType:
    """Load a sibling without depending on the caller's ``sys.path``."""
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load required sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAMPAIGN = _load_sibling(
    "hdg_tensor_campaign.py", "dg_igr_hdg_tensor_campaign_memory_dependency"
)
MEMORY = _load_sibling(
    "gpu_memory_control.py", "dg_igr_gpu_memory_control_dependency"
)


class MemoryCampaignError(RuntimeError):
    """An exact-candidate or whole-device memory qualification failure."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run post-timing whole-device GPU peak-memory controls for every "
            "production tensor campaign cell. The caller must hold the "
            "external exclusive GPU lease."
        )
    )
    result.add_argument("--benchmark", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--source-root", required=True)
    result.add_argument("--evidence-mode", required=True,
                        choices=("qualification",))
    result.add_argument("--workload-label", required=True)
    result.add_argument("--mesh-n", required=True, type=_positive_int)
    result.add_argument("--domain-periods", required=True, type=_positive_int)
    result.add_argument("--dt", required=True, type=_positive_float)
    result.add_argument("--repeats", required=True, type=_positive_int)
    result.add_argument("--poll-ms", required=True, type=_positive_int)
    result.add_argument("--build-label", required=True)
    result.add_argument("--build-metadata", action="append", default=[],
                        metavar="KEY=VALUE")
    result.add_argument("--build-manifest", required=True)
    result.add_argument("--mfem-identity", required=True)
    result.add_argument("--expected-commit", required=True)
    result.add_argument("--expected-tree", required=True)
    result.add_argument("--gpu-index", required=True, type=int)
    result.add_argument("--gpu-uuid", required=True)
    result.add_argument("--nvidia-smi", required=True)
    result.add_argument("--hardware-metadata", action="append", default=[],
                        metavar="KEY=VALUE")
    result.add_argument("--timeout-seconds", type=_positive_float,
                        default=1800.0)
    return result


def _validate_profile(arguments: argparse.Namespace) -> None:
    observed = {
        "label": arguments.workload_label,
        "mesh_n": arguments.mesh_n,
        "domain_periods": arguments.domain_periods,
        "dt": arguments.dt,
        "repeats": arguments.repeats,
        "poll_ms": arguments.poll_ms,
    }
    mismatches = [
        f"{name}={observed[name]!r} (expected {expected!r})"
        for name, expected in QUALIFICATION_WORKLOAD.items()
        if observed[name] != expected
    ]
    if mismatches:
        raise MemoryCampaignError(
            "memory qualification does not match the exact saturated profile: "
            + "; ".join(mismatches)
        )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_live_gpu(snapshot: dict[str, Any], *, gpu_index: int,
                       gpu_uuid: str) -> dict[str, Any]:
    CAMPAIGN.validate_hardware_snapshot(snapshot, True)
    values = snapshot.get("values", {})
    try:
        observed_index = int(values["index"])
        observed_compute_capability = float(values["compute_cap"])
    except (KeyError, TypeError, ValueError) as error:
        raise MemoryCampaignError(
            "nvidia-smi index/compute-capability identity is invalid"
        ) from error
    if observed_index != gpu_index:
        raise MemoryCampaignError(
            f"nvidia-smi selected index {observed_index}; expected {gpu_index}"
        )
    if values.get("uuid") != gpu_uuid:
        raise MemoryCampaignError(
            "nvidia-smi UUID differs from the exact selected --gpu-uuid"
        )
    if not math.isclose(observed_compute_capability, 12.0,
                        rel_tol=0.0, abs_tol=0.0):
        raise MemoryCampaignError(
            "whole-device memory qualification requires compute capability 12.0"
        )
    return {
        "device_ordinal": 0,
        "uuid": values["uuid"],
        "compute_capability": 120,
    }


def _artifact_identity(output_dir: Path, relative: Path) -> dict[str, Any]:
    return CAMPAIGN.artifact_identity(output_dir / relative, relative)


def _write_control_artifacts(output_dir: Path, relative_stem: Path,
                             evidence: dict[str, Any]) -> dict[str, Any]:
    benchmark = evidence.get("benchmark")
    poller = evidence.get("poller")
    benchmark = benchmark if isinstance(benchmark, dict) else {}
    poller = poller if isinstance(poller, dict) else {}
    paths = {
        "control": Path("raw") / relative_stem.with_suffix(".control.json"),
        "benchmark_stdout": (
            Path("raw") / relative_stem.with_suffix(".benchmark.stdout.json")
        ),
        "benchmark_stderr": (
            Path("raw") / relative_stem.with_suffix(".benchmark.stderr.txt")
        ),
        "poller_stdout": (
            Path("raw") / relative_stem.with_suffix(".poller.stdout.csv")
        ),
        "poller_stderr": (
            Path("raw") / relative_stem.with_suffix(".poller.stderr.txt")
        ),
    }
    CAMPAIGN.write_json_exclusive(output_dir / paths["control"], evidence)
    CAMPAIGN.write_exclusive(
        output_dir / paths["benchmark_stdout"],
        str(benchmark.get("stdout", "")),
    )
    CAMPAIGN.write_exclusive(
        output_dir / paths["benchmark_stderr"],
        str(benchmark.get("stderr", "")),
    )
    CAMPAIGN.write_exclusive(
        output_dir / paths["poller_stdout"],
        str(poller.get("raw_stdout", "")),
    )
    CAMPAIGN.write_exclusive(
        output_dir / paths["poller_stderr"],
        str(poller.get("raw_stderr", "")),
    )
    return {
        name: {
            "path": path.as_posix(),
            "identity": _artifact_identity(output_dir, path),
        }
        for name, path in paths.items()
    }


def _raw_artifact_ledger(output_dir: Path) -> dict[str, Any]:
    raw_dir = output_dir / "raw"
    entries = []
    if raw_dir.exists():
        for path in sorted(candidate for candidate in raw_dir.rglob("*")
                           if candidate.is_file()):
            relative = path.relative_to(output_dir)
            entries.append(_artifact_identity(output_dir, relative))
    encoded = json.dumps(
        entries, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "artifact_count": len(entries),
        "entries": entries,
        "ledger_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    manifest_path = output_dir / "memory_campaign.json"
    CAMPAIGN.write_json_exclusive(manifest_path, manifest)
    digest = CAMPAIGN.sha256_file(manifest_path)
    CAMPAIGN.write_exclusive(
        output_dir / "memory_campaign.sha256",
        f"{digest}  {manifest_path.name}\n",
    )
    return manifest_path


def _record_internal_floor(record: dict[str, Any]) -> int:
    snapshots = record["device_memory_snapshots"]
    return max(int(snapshot["used_bytes"]) for snapshot in snapshots.values())


def _module_identities() -> dict[str, Any]:
    return {
        "driver": CAMPAIGN.regular_file_identity(Path(__file__)),
        "accepted_step_campaign_validator": CAMPAIGN.regular_file_identity(
            SCRIPT_DIR / "hdg_tensor_campaign.py"
        ),
        "whole_device_sampler": CAMPAIGN.regular_file_identity(
            SCRIPT_DIR / "gpu_memory_control.py"
        ),
    }


def _same_module_identities(start: dict[str, Any]) -> None:
    if _module_identities() != start:
        raise MemoryCampaignError(
            "memory driver or a validation/sampling dependency changed during run"
        )


def _expected_build(source: dict[str, Any], cache_path: Path) -> dict[str, str]:
    return {
        "source_revision": source["commit"],
        "source_tree": source["tree"],
        "source_status": "clean",
        "build_directory": str(cache_path.parent),
        "cmake_cache": str(cache_path),
    }


def _run_one_control(*, arguments: argparse.Namespace, output_dir: Path,
                     executable: Path, expected_build: dict[str, str],
                     expected_gpu: dict[str, Any], mode: str, order: int,
                     engine: str, repeat_index: int,
                     invocations: list[dict[str, Any]]) -> tuple[
                         dict[str, Any], dict[str, Any], int]:
    command = CAMPAIGN.final_command(
        executable,
        engine=engine,
        mode=mode,
        order=order,
        mesh_n=arguments.mesh_n,
        domain_periods=arguments.domain_periods,
        timestep=arguments.dt,
        export=False,
        expected_build=expected_build,
    )
    relative_stem = (
        Path(f"{mode}-p{order}") / engine / f"repeat-{repeat_index:03d}"
    )
    invocation: dict[str, Any] = {
        "id": f"{mode}-p{order}-{engine}-repeat-{repeat_index:03d}",
        "mode": mode,
        "order": order,
        "engine": engine,
        "repeat_index": repeat_index,
        "host_export": False,
        "command": command,
        "command_shell_display": shlex.join(command),
        "status": "running",
    }
    invocations.append(invocation)
    evidence: dict[str, Any] | None = None
    try:
        evidence = MEMORY.run_gpu_memory_control(
            command,
            nvidia_smi=arguments.nvidia_smi,
            selected_gpu_uuid=arguments.gpu_uuid,
            poll_ms=arguments.poll_ms,
            timeout_seconds=arguments.timeout_seconds,
            environment=os.environ.copy(),
            cwd=arguments.source_root,
        )
    except MEMORY.MemoryControlError as error:
        if isinstance(error.evidence, dict):
            evidence = error.evidence
            invocation["artifacts"] = _write_control_artifacts(
                output_dir, relative_stem, evidence
            )
        invocation["status"] = "failure"
        invocation["error"] = f"{type(error).__name__}: {error}"
        raise MemoryCampaignError(str(error)) from error

    invocation["artifacts"] = _write_control_artifacts(
        output_dir, relative_stem, evidence
    )
    benchmark = evidence.get("benchmark")
    if not isinstance(benchmark, dict):
        invocation["status"] = "failure"
        raise MemoryCampaignError("memory control omitted benchmark evidence")
    stdout = benchmark.get("stdout")
    if not isinstance(stdout, str):
        invocation["status"] = "failure"
        raise MemoryCampaignError("benchmark stdout is not retained text")
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as error:
        invocation["status"] = "failure"
        invocation["error"] = f"{type(error).__name__}: {error}"
        raise MemoryCampaignError(
            f"benchmark did not emit exactly one JSON value: {error}"
        ) from error
    try:
        record = CAMPAIGN.validate_record(
            parsed,
            source="final",
            mode=mode,
            order=order,
            mesh_n=arguments.mesh_n,
            domain_periods=arguments.domain_periods,
            timestep=arguments.dt,
            export=False,
            engine=engine,
            expected_build=expected_build,
            expected_gpu=expected_gpu,
        )
        if record["principal_blocks"] < record["gpu_sm_count"]:
            raise MemoryCampaignError(
                "saturated memory workload launched fewer principal blocks "
                "than SMs"
            )
    except Exception as error:
        invocation["status"] = "failure"
        invocation["error"] = f"{type(error).__name__}: {error}"
        raise
    floor = _record_internal_floor(record)
    invocation.update({
        "status": "success",
        "started_utc": benchmark.get("started_utc"),
        "finished_utc": benchmark.get("finished_utc"),
        "wall_seconds": benchmark.get("wall_seconds"),
        "external_memory_measurement": evidence["measurement"],
        "internal_cuda_used_floor_bytes": floor,
        "benchmark_summary": CAMPAIGN.record_summary(record),
    })
    return evidence, record, floor


def run_campaign(arguments: argparse.Namespace) -> Path:
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise MemoryCampaignError(
            f"output directory already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": 1,
        "campaign": "production_tensor_core_hdg_whole_device_peak_memory",
        "evidence_mode": arguments.evidence_mode,
        "promotion_qualification": True,
        "status": "running",
        "qualification_pass": False,
        "started_utc": CAMPAIGN.utc_now(),
        "method": {
            "post_timing_control": True,
            "primary_performance_eligible": False,
            "reason_excluded_from_primary_timing": (
                "nvidia-smi is intentionally polled concurrently with every "
                "accepted-step benchmark invocation"
            ),
            "external_exclusive_gpu_lease_required": True,
            "gpu_lease_acquired_by_driver": False,
            "build_performed_by_driver": False,
            "whole_device_measurement": (
                "nvidia-smi memory.used from launch through first post-exit row"
            ),
            "internal_floor": (
                "maximum benchmark cudaMemGetInfo used_bytes snapshot"
            ),
        },
        "workload": {
            "label": arguments.workload_label,
            "modes": ["conventional", "igr"],
            "orders": [1, 2, 3],
            "engines": ["ordinary", "tensor"],
            "mesh_n": arguments.mesh_n,
            "domain_periods": arguments.domain_periods,
            "dt": arguments.dt,
            "host_export": False,
            "repeats_per_engine_cell": arguments.repeats,
            "poll_ms": arguments.poll_ms,
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
        "invocations": [],
    }
    try:
        _validate_profile(arguments)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(arguments.gpu_index):
            raise MemoryCampaignError(
                "qualification requires CUDA_VISIBLE_DEVICES to equal the "
                "exclusive --gpu-index"
            )
        source_root = Path(arguments.source_root).expanduser().resolve(strict=True)
        arguments.source_root = str(source_root)
        driver_path = Path(__file__).resolve()
        if not _inside(driver_path, source_root) or \
                not _inside(SCRIPT_DIR / "hdg_tensor_campaign.py", source_root) or \
                not _inside(SCRIPT_DIR / "gpu_memory_control.py", source_root):
            raise MemoryCampaignError(
                "qualification driver and dependencies must come from the exact "
                "candidate source tree"
            )
        module_identities = _module_identities()
        manifest["modules"] = module_identities
        source_start = CAMPAIGN.git_identity(source_root)
        manifest["source_start"] = source_start
        if source_start["status_short"]:
            raise MemoryCampaignError(
                "memory qualification requires an exact clean candidate tree"
            )
        if source_start["commit"] != arguments.expected_commit or \
                source_start["tree"] != arguments.expected_tree:
            raise MemoryCampaignError(
                "source commit/tree differs from the exact expected candidate"
            )

        build_cache = Path(arguments.build_manifest).expanduser().resolve(strict=True)
        build = CAMPAIGN.validated_build_provenance(
            source_root=source_root,
            cmake_cache_path=str(build_cache),
            mfem_identity_path=arguments.mfem_identity,
        )
        executable_identity = CAMPAIGN.file_identity(Path(arguments.benchmark))
        executable = Path(executable_identity["path"])
        build_relationship = CAMPAIGN.validate_executable_build_relationship(
            executable_identity, build
        )
        expected_build = _expected_build(source_start, build_cache)
        manifest["build"] = {
            "label": arguments.build_label,
            "metadata": CAMPAIGN.parse_metadata(
                arguments.build_metadata, "build metadata"
            ),
            "provenance": build,
            "executable": executable_identity,
            "executable_build_relationship": build_relationship,
            "expected_embedded_identity": expected_build,
        }

        nvidia_smi = CAMPAIGN.resolve_nvidia_smi(arguments.nvidia_smi)
        if nvidia_smi is None:
            raise MemoryCampaignError("nvidia-smi was not found")
        arguments.nvidia_smi = str(nvidia_smi)
        start_hardware = CAMPAIGN.query_nvidia_smi(
            nvidia_smi, arguments.gpu_index
        )
        expected_gpu = _validate_live_gpu(
            start_hardware,
            gpu_index=arguments.gpu_index,
            gpu_uuid=arguments.gpu_uuid,
        )
        manifest["hardware"] = {
            "gpu_index": arguments.gpu_index,
            "gpu_uuid": arguments.gpu_uuid,
            "nvidia_smi_executable": str(nvidia_smi),
            "supplied": CAMPAIGN.parse_metadata(
                arguments.hardware_metadata, "hardware metadata"
            ),
            "start": start_hardware,
            "post_invocation": [],
        }

        controls: dict[tuple[str, int, str], list[dict[str, Any]]] = {
            (mode, order, engine): []
            for mode, order in CELLS for engine in ("ordinary", "tensor")
        }
        floors: dict[tuple[str, int, str], list[int]] = {
            key: [] for key in controls
        }
        records: dict[tuple[str, int, str], list[dict[str, Any]]] = {
            key: [] for key in controls
        }
        for repeat_index in range(arguments.repeats):
            cells: Sequence[tuple[str, int]] = (
                CELLS if repeat_index % 2 == 0 else tuple(reversed(CELLS))
            )
            engines = (
                ("ordinary", "tensor") if repeat_index % 2 == 0
                else ("tensor", "ordinary")
            )
            for mode, order in cells:
                for engine in engines:
                    evidence, record, floor = _run_one_control(
                        arguments=arguments,
                        output_dir=output_dir,
                        executable=executable,
                        expected_build=expected_build,
                        expected_gpu=expected_gpu,
                        mode=mode,
                        order=order,
                        engine=engine,
                        repeat_index=repeat_index,
                        invocations=manifest["invocations"],
                    )
                    key = (mode, order, engine)
                    controls[key].append(evidence)
                    records[key].append(record)
                    floors[key].append(floor)
                    observed_hardware = CAMPAIGN.query_nvidia_smi(
                        nvidia_smi, arguments.gpu_index
                    )
                    _validate_live_gpu(
                        observed_hardware,
                        gpu_index=arguments.gpu_index,
                        gpu_uuid=arguments.gpu_uuid,
                    )
                    CAMPAIGN.validate_hardware_identity(
                        start_hardware, observed_hardware
                    )
                    manifest["hardware"]["post_invocation"].append({
                        "invocation_id": manifest["invocations"][-1]["id"],
                        "snapshot": observed_hardware,
                    })

        results: dict[str, Any] = {}
        for mode, order in CELLS:
            cell_key = f"{mode}_p{order}"
            results[cell_key] = {}
            for engine in ("ordinary", "tensor"):
                key = (mode, order, engine)
                aggregation = MEMORY.aggregate_three_repeat_controls(
                    controls[key], floors[key]
                )
                results[cell_key][engine] = {
                    "whole_device_peak_memory": aggregation,
                    "internal_cuda_snapshot_summary": CAMPAIGN.memory_summary(
                        records[key]
                    ),
                    "repeat_invocation_ids": [
                        f"{mode}-p{order}-{engine}-repeat-{index:03d}"
                        for index in range(arguments.repeats)
                    ],
                }
        manifest["results"] = results

        source_end = CAMPAIGN.git_identity(source_root)
        if source_end != source_start:
            raise MemoryCampaignError(
                "source commit/tree/status changed during memory qualification"
            )
        manifest["source_end"] = source_end
        if CAMPAIGN.file_identity(executable) != executable_identity:
            raise MemoryCampaignError(
                "accepted-step benchmark changed during memory qualification"
            )
        build_end = CAMPAIGN.validated_build_provenance(
            source_root=source_root,
            cmake_cache_path=str(build_cache),
            mfem_identity_path=arguments.mfem_identity,
        )
        if build_end != build:
            raise MemoryCampaignError(
                "build/MFEM provenance changed during memory qualification"
            )
        _same_module_identities(module_identities)
        end_hardware = CAMPAIGN.query_nvidia_smi(
            nvidia_smi, arguments.gpu_index
        )
        _validate_live_gpu(
            end_hardware,
            gpu_index=arguments.gpu_index,
            gpu_uuid=arguments.gpu_uuid,
        )
        CAMPAIGN.validate_hardware_identity(start_hardware, end_hardware)
        manifest["hardware"]["end"] = end_hardware
        manifest["integrity"] = {
            "source_rechecked_unchanged": True,
            "executable_rehashed_unchanged": True,
            "build_and_mfem_rehashed_unchanged": True,
            "driver_and_dependencies_rehashed_unchanged": True,
            "raw_artifacts": _raw_artifact_ledger(output_dir),
        }
        manifest["status"] = "completed"
        manifest["qualification_pass"] = True
        manifest["finished_utc"] = CAMPAIGN.utc_now()
    except Exception as error:
        manifest["status"] = "failure"
        manifest["qualification_pass"] = False
        manifest["finished_utc"] = CAMPAIGN.utc_now()
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest.setdefault("integrity", {})["raw_artifacts"] = \
            _raw_artifact_ledger(output_dir)
        _write_manifest(output_dir, manifest)
        raise

    return _write_manifest(output_dir, manifest)


def main() -> int:
    try:
        arguments = parser().parse_args()
        path = run_campaign(arguments)
        print(json.dumps({
            "qualification_completed": True,
            "qualification_pass": True,
            "memory_campaign_manifest": str(path),
        }))
        return 0
    except Exception as error:
        print(f"hdg_tensor_memory_campaign: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
