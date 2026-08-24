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

FAKE_BENCHMARK = r'''#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=("conventional", "igr"))
parser.add_argument("--engine", default="ordinary", choices=("ordinary", "tensor"))
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
if is_frozen:
    scale = 2.5
elif args.engine == "ordinary":
    scale = 2.0
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
        sample["tensor_core"] = {
            "selected_engine": args.engine,
            "trace_preconditioner": (
                "hypre_ilu0"
                if args.engine == "ordinary"
                else "hypre_diag_scale"
            ),
            "trace_preconditioner_configured": True,
            "hardware_supported": True,
            "condensation_launches": 4 if args.engine == "tensor" else 0,
            "mma_tiles": 64 if args.engine == "tensor" else 0,
            "fallback_count": 0,
            "fallback_accounting_complete": True,
            "no_fallback": True,
        }
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
    "trace_preconditioner": (
        "hypre_ilu0" if args.engine == "ordinary" else "hypre_diag_scale"
    ),
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
        "igr_assembly_quadrature_order": 12,
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
        "linear_relative_tolerance": 1.0e-12,
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
    del payload["cuda_engine"]
    del payload["trace_preconditioner"]
else:
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
            "dependency_cuda_arch=120",
            "mfem_cuda_archs=80-virtual",
            "",
        )),
        encoding="utf-8",
    )
    return cache, mfem


def main() -> int:
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
        assert campaign["source_start"] == campaign["source_end"]
        assert len(campaign["final_tree_results"]) == 6
        assert "frozen_control_results" not in campaign
        assert len(campaign["invocations"]) == 26
        assert len(list((output / "raw").rglob("*.json"))) == 26
        for result in campaign["final_tree_results"].values():
            comparison = result["paired_comparison"]
            assert math.isclose(comparison["speedup"]["median"], 2.0)
            assert comparison[
                "median_speedup_95_percent_confidence_interval"
            ]["resamples"] == 100
            assert result["engines"]["ordinary"]["solver_work"][
                "trace_preconditioners"
            ] == ["hypre_ilu0"]
            assert result["engines"]["tensor"]["solver_work"][
                "trace_preconditioners"
            ] == ["hypre_diag_scale"]
            for engine in ("ordinary", "tensor"):
                work = result["engines"][engine]["solver_work"]
                assert work["linear_iterations"]["total"] == 70
                assert work["linear_iterations_by_invocation"] == [
                    [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]
                ]
                assert work["tensor_core"]["all_no_fallback"] is True
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
            expected_preconditioner = (
                "hypre_diag_scale"
                if item["comparison_member"] == "tensor"
                else "hypre_ilu0"
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

        ordinary_invocation = next(
            item for item in campaign["invocations"]
            if item["phase"] == "final_tree_primary" and
            item["comparison_member"] == "ordinary"
        )
        ordinary_record = json.loads(
            (output / ordinary_invocation["raw_json"]).read_text()
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

        bad_build = copy.deepcopy(ordinary_record)
        bad_build["build_identity"]["source_revision"] = "0" * 40
        assert rejected(bad_build)
        bad_uuid = copy.deepcopy(ordinary_record)
        bad_uuid["gpu_uuid"] = (
            "GPU-00000000-0000-0000-0000-000000000002"
        )
        assert rejected(bad_uuid)

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
        for invocation in frozen_invocations:
            record = json.loads(
                (qualification_output / invocation["raw_json"]).read_text(
                    encoding="utf-8"
                )
            )
            assert record["schema"] == 1
            assert "cuda_engine" not in record
            assert "build_identity" not in record
    print("hdg_tensor_campaign smoke and qualification tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
