#!/usr/bin/env python3
"""Deterministic non-capture regression tests for profile_harness.py."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts/perf/profile_harness.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "profile_harness_under_test", HARNESS
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError("could not load profile_harness.py")
PROFILE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(PROFILE)


def run(*arguments: str, expected: int = 0, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run([sys.executable, str(HARNESS), *arguments], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env or os.environ.copy())
    if result.returncode != expected:
        raise RuntimeError(f"expected {expected}, got {result.returncode}: {result.args}\n{result.stdout}\n{result.stderr}")
    return result


def executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(0o755)


def read_manifest(directory: Path) -> dict:
    return json.loads((directory / "profile-manifest.json").read_text(encoding="utf-8"))


def git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def common(app: Path, output: Path) -> list[str]:
    return ["--scenario", "generic-case", "--exe", str(app), "--out-dir", str(output), "--arg=--input", "--arg", "case.yaml"]


def validate_fixed_sigma_jacobi_profile_surface() -> None:
    expected = {
        "native-element-schwarz-matrix-free-equilibrated-field-split-"
        "sigma-jacobi1-fp64": (
            1,
            "native_element_schwarz_equilibrated_field_split_"
            "sigma_jacobi1_fp64",
            False,
        ),
        "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
        "sigma-jacobi1-fp64": (
            1,
            "native_element_schwarz_warp_equilibrated_field_split_"
            "sigma_jacobi1_fp64",
            True,
        ),
        "native-element-schwarz-matrix-free-equilibrated-field-split-"
        "sigma-jacobi2-fp64": (
            2,
            "native_element_schwarz_equilibrated_field_split_"
            "sigma_jacobi2_fp64",
            False,
        ),
        "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
        "sigma-jacobi2-fp64": (
            2,
            "native_element_schwarz_warp_equilibrated_field_split_"
            "sigma_jacobi2_fp64",
            True,
        ),
        "native-element-schwarz-matrix-free-equilibrated-field-split-"
        "sigma-jacobi3-fp64": (
            3,
            "native_element_schwarz_equilibrated_field_split_"
            "sigma_jacobi3_fp64",
            False,
        ),
        "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
        "sigma-jacobi3-fp64": (
            3,
            "native_element_schwarz_warp_equilibrated_field_split_"
            "sigma_jacobi3_fp64",
            True,
        ),
        "native-element-schwarz-matrix-free-equilibrated-field-split-"
        "sigma-jacobi4-fp64": (
            4,
            "native_element_schwarz_equilibrated_field_split_"
            "sigma_jacobi4_fp64",
            False,
        ),
        "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
        "sigma-jacobi4-fp64": (
            4,
            "native_element_schwarz_warp_equilibrated_field_split_"
            "sigma_jacobi4_fp64",
            True,
        ),
        "native-element-schwarz-matrix-free-equilibrated-field-split-"
        "sigma-jacobi5-fp64": (
            5,
            "native_element_schwarz_equilibrated_field_split_"
            "sigma_jacobi5_fp64",
            False,
        ),
        "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
        "sigma-jacobi5-fp64": (
            5,
            "native_element_schwarz_warp_equilibrated_field_split_"
            "sigma_jacobi5_fp64",
            True,
        ),
        "native-element-schwarz-matrix-free-equilibrated-field-split-"
        "sigma-jacobi6-fp64": (
            6,
            "native_element_schwarz_equilibrated_field_split_"
            "sigma_jacobi6_fp64",
            False,
        ),
        "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
        "sigma-jacobi6-fp64": (
            6,
            "native_element_schwarz_warp_equilibrated_field_split_"
            "sigma_jacobi6_fp64",
            True,
        ),
        "native-element-schwarz-matrix-free-equilibrated-field-split-"
        "sigma-jacobi7-fp64": (
            7,
            "native_element_schwarz_equilibrated_field_split_"
            "sigma_jacobi7_fp64",
            False,
        ),
        "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
        "sigma-jacobi7-fp64": (
            7,
            "native_element_schwarz_warp_equilibrated_field_split_"
            "sigma_jacobi7_fp64",
            True,
        ),
    }
    assert PROFILE.FIXED_SIGMA_JACOBI_PROFILE_POLICIES == expected
    conventional_default = PROFILE._profile_trace_policy_contract(
        "engine-default", "conventional"
    )
    assert conventional_default == (
        "engine-default", "engine-default", "hypre_diag_scale",
        "production", True, None,
    )
    igr_default = PROFILE._profile_trace_policy_contract(
        "engine-default", "igr"
    )
    assert igr_default == (
        "engine-default",
        "native-element-schwarz-matrix-free-warp-"
        "equilibrated-field-split-fp64",
        "native_element_schwarz_warp_equilibrated_field_split_fp64",
        "production", True, None,
    )

    def rejected(function: object, *arguments: object) -> bool:
        try:
            function(*arguments)  # type: ignore[operator]
        except RuntimeError:
            return True
        return False

    assert rejected(
        PROFILE._profile_trace_policy_contract,
        "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
        "sigma-jacobi7-fp64-extra",
        "igr",
    )
    assert rejected(
        PROFILE._profile_trace_policy_contract,
        "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
        "sigma-jacobi8-fp64",
        "igr",
    )
    for policy, descriptor in expected.items():
        contract = PROFILE._profile_trace_policy_contract(policy, "igr")
        assert contract == (
            policy, policy, descriptor[1], "ablation", False,
            PROFILE.FixedSigmaJacobiProfilePolicy(*descriptor),
        )
        assert rejected(
            PROFILE._profile_trace_policy_contract, policy, "conventional"
        )
        tensor = {
            "element_schwarz_preconditioner_applications": 9,
            "field_split_sigma_jacobi_sweeps": descriptor[0],
            "field_split_sigma_jacobi_damping": 2.0 / 3.0,
            "field_split_sigma_jacobi_updates": 9 * descriptor[0],
        }
        assert PROFILE._validated_fixed_sigma_jacobi_work(
            tensor, contract
        ) == tensor
        for field, value in (
                ("field_split_sigma_jacobi_sweeps",
                 5 if descriptor[0] != 5 else 4),
                ("field_split_sigma_jacobi_damping", 0.5),
                ("field_split_sigma_jacobi_damping", float("nan")),
                ("field_split_sigma_jacobi_updates",
                 9 * descriptor[0] - 1)):
            invalid = {**tensor, field: value}
            assert rejected(
                PROFILE._validated_fixed_sigma_jacobi_work,
                invalid,
                contract,
            ), (policy, field)
        no_work = {
            **tensor,
            "element_schwarz_preconditioner_applications": 0,
            "field_split_sigma_jacobi_updates": 0,
        }
        assert rejected(
            PROFILE._validated_fixed_sigma_jacobi_work, no_work, contract
        ), policy
        if descriptor[0] == 7:
            overflowing = {
                **tensor,
                "element_schwarz_preconditioner_applications":
                    PROFILE.UINT64_MAX,
                "field_split_sigma_jacobi_updates": PROFILE.UINT64_MAX,
            }
            assert rejected(
                PROFILE._validated_fixed_sigma_jacobi_work,
                overflowing,
                contract,
            ), policy


def main() -> int:
    validate_fixed_sigma_jacobi_profile_surface()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        app = root / "app"
        executable(app, "import sys\nprint('app', *sys.argv[1:])\n")
        env = {**os.environ, "PROFILE_HARNESS_TEST": "allowlisted"}
        l0 = root / "l0"
        run("--level", "l0", *common(app, l0), "--environment", "PROFILE_HARNESS_TEST", env=env)
        l0_manifest = read_manifest(l0)
        assert l0_manifest["outcome"] == "success"
        assert l0_manifest["environment"] == {"PROFILE_HARNESS_TEST": {"status": "available", "value": "allowlisted"}}
        log_entry = next(item for item in l0_manifest["artifacts"] if item["path"] == "execution.log")
        assert log_entry["bytes"] == (l0 / "execution.log").stat().st_size and len(log_entry["sha256"]) == 64
        assert l0_manifest["source_dirty_state"]["status"] == "available"
        assert l0_manifest["identity"]["executable"]["bytes"] == app.stat().st_size
        assert len(l0_manifest["identity"]["executable"]["sha256"]) == 64
        exact_source = root / "exact-source"
        exact_source.mkdir()
        git(exact_source, "init")
        git(exact_source, "config", "user.name", "Profile Harness Test")
        git(exact_source, "config", "user.email", "profile-harness@example.invalid")
        exact_app = exact_source / "app"
        executable(exact_app, "print('exact app')\n")
        build_manifest = exact_source / "CMakeCache.txt"
        build_manifest.write_text("CMAKE_BUILD_TYPE:STRING=Release\n", encoding="utf-8")
        git(exact_source, "add", "app", "CMakeCache.txt")
        git(exact_source, "commit", "-m", "exact fixture")
        exact_commit = git(exact_source, "rev-parse", "HEAD")
        exact_tree = git(exact_source, "rev-parse", "HEAD^{tree}")
        exact_out = root / "exact"
        run("--level", "l0", *common(exact_app, exact_out),
            "--source-root", str(exact_source),
            "--expected-commit", exact_commit, "--expected-tree", exact_tree,
            "--require-clean-source", "--build-manifest", str(build_manifest))
        exact_identity = read_manifest(exact_out)["identity"]
        assert exact_identity["source_binding"]["expected_commit"] == exact_commit
        assert exact_identity["source_binding"]["expected_tree"] == exact_tree
        assert exact_identity["source_binding"]["clean_source_required"] is True
        assert exact_identity["source_binding"]["source"]["status"]["value"] == "clean"
        assert exact_identity["build_manifest"]["path"] == str(build_manifest)
        assert len(exact_identity["build_manifest"]["sha256"]) == 64
        mismatch_out = root / "exact-mismatch"
        run("--level", "l0", *common(exact_app, mismatch_out),
            "--source-root", str(exact_source), "--expected-commit", "0" * 40,
            "--expected-tree", exact_tree, "--require-clean-source",
            "--build-manifest", str(build_manifest), expected=6)
        mismatch_manifest = read_manifest(mismatch_out)
        assert mismatch_manifest["outcome"] == "validation_failure"
        assert "source revision mismatch" in mismatch_manifest["detail"]

        qualification_source = root / "qualification-source"
        qualification_source.mkdir()
        git(qualification_source, "init")
        git(qualification_source, "config", "user.name", "Profile Harness Test")
        git(
            qualification_source, "config", "user.email",
            "profile-harness@example.invalid",
        )
        (qualification_source / "tpl/mfem").mkdir(parents=True)
        (qualification_source / "tpl/mfem/pin.txt").write_text(
            "fixture\n", encoding="utf-8"
        )
        qualification_build = qualification_source / "build"
        qualification_build.mkdir()
        qualification_app = (
            qualification_build /
            "dg_igr_hdg_runtime_cuda_accepted_step_benchmark"
        )
        executable(qualification_app, "print('qualification benchmark')\n")
        qualification_cache = qualification_build / "CMakeCache.txt"
        qualification_cache.write_text(
            "\n".join((
                "CMAKE_BUILD_TYPE:STRING=Release",
                "CMAKE_CUDA_ARCHITECTURES:STRING=120",
                "CMAKE_CUDA_COMPILER:FILEPATH=/opt/cuda/bin/nvcc",
                "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++",
                "MFEM_DIR:PATH=/opt/mfem/lib/cmake/mfem",
                "DG_IGR_ENABLE_HDG_RUNTIME:BOOL=ON",
                "DG_IGR_ENABLE_OBSERVABILITY:BOOL=ON",
                "DG_IGR_OBSERVABILITY_USE_NVTX:BOOL=ON",
                f"CMAKE_HOME_DIRECTORY:INTERNAL={qualification_source}",
                "",
            )),
            encoding="utf-8",
        )
        qualification_mfem = (
            qualification_build / "dg-igr-mfem-identity-v1.txt"
        )
        qualification_mfem.write_text(
            "\n".join((
                "schema=dg-igr-mfem-identity",
                "version=1",
                "mfem_pin=fixture-pin",
                f"source_root={qualification_source / 'tpl/mfem'}",
                "install_root=/opt/mfem",
                "mode=gpu",
                "profile=cuda-sm120",
                "build_type=Release",
                "compiler=gcc@14.2.0",
                "hypre_spec=hypre@2.32.0+cuda cuda_arch=120",
                "dependency_cuda_arch=120",
                "mfem_cuda_archs=80-virtual",
                "",
            )),
            encoding="utf-8",
        )
        git(qualification_source, "add", ".")
        git(qualification_source, "commit", "-m", "qualification fixture")
        qualification_commit = git(qualification_source, "rev-parse", "HEAD")
        qualification_tree = git(
            qualification_source, "rev-parse", "HEAD^{tree}"
        )

        qualification_tools = root / "qualification-tools"
        qualification_tools.mkdir()
        fake_ncu = qualification_tools / "ncu"
        profiler_version = (
            "NVIDIA (R) Nsight Compute Command Line Profiler\nVersion 2026.1"
        )
        executable(
            fake_ncu,
            """import json
import os
import pathlib
import sys

arguments = sys.argv[1:]
if arguments == ["--version"]:
    print("NVIDIA (R) Nsight Compute Command Line Profiler")
    print("Version 2026.1")
elif "--import" in arguments:
    if arguments[-2:] != ["--print-units", "base"]:
        raise SystemExit(19)
    if os.environ.get("FAKE_NCU_ATTRIBUTION"):
        print('"ID","Kernel Name","NVTX Push/Pop_Range","gpu__time_duration.avg"')
        print('"","","","nsecond"')
        categories = [
            ("BorrowedEulerVolumeKernel", "local-assembly", 8000),
            ("BorrowedEulerTensorVolumeContractionKernel", "local-assembly", 1500),
            ("BorrowedEulerTensorVolumeContractionKernel", "local-assembly", 1500),
            ("BorrowedEulerTensorVolumeContractionKernel", "local-assembly", 1500),
            ("TensorBorrowedCondenseKernel", "condensation", 2000),
            ("TensorBorrowedCondenseKernel", "condensation", 2000),
            ("TensorBorrowedCondenseKernel", "condensation", 2000),
            ("OwnedMatrixKernel", "trace-assembly", 500),
            ("NativeRankOneTraceSolveKernel", "trace-solve", 700),
            ("BorrowedRecoverKernel", "interior-recovery", 400),
            ("SquaredNormKernel", "reduction", 100),
            ("AdmissibilityKernel", "validation", 100),
            ("GatherKernel", "packing", 50),
            ("TrialKernel", "nonlinear-update", 50),
        ]
        if os.environ.get("FAKE_NCU_MISSING_CATEGORY"):
            categories = categories[:-1]
        if os.environ.get("FAKE_NCU_MISSING_VOLUME"):
            categories = [item for item in categories
                          if "VolumeContraction" not in item[0]]
        for launch, (kernel, category, duration) in enumerate(categories):
            ranges = ("HDG/profiled-accepted-step HDG/accepted-step "
                      f"HDG/{category}")
            print(f'"{launch}","void {kernel}()","{ranges}","{duration}"')
    else:
        print('"ID","Kernel Name","sm__inst_executed_pipe_fp64_op_dmma.sum","gpu__time_duration.avg"')
        print('"","","inst","nsecond"')
        volume = bool(os.environ.get("FAKE_NCU_VOLUME"))
        p1_igr = bool(os.environ.get("FAKE_APP_P1_IGR"))
        expected_values = (
            [908544, 908544, 908544, 227136] * 3
            if volume and p1_igr else
            [22145760] * 3 if volume else [22827168] * 3
        )
        kernel = ("BorrowedEulerTensorVolumeContractionKernel"
                  if volume else "TensorBorrowedCondenseKernel")
        for launch, expected in enumerate(expected_values):
            value = ("0" if os.environ.get("FAKE_NCU_ZERO") else
                     str(expected - 1)
                     if os.environ.get("FAKE_NCU_WRONG_TOTAL") else
                     str(expected))
            print(f'"{launch}","void {kernel}<double>()","{value}","226912"')
else:
    stem = pathlib.Path(arguments[arguments.index("--export") + 1])
    payload = (b"not-a-native-report" if os.environ.get("FAKE_NCU_BAD_REPORT")
               else b"NVR\\x00synthetic-native-report")
    stem.with_suffix(".ncu-rep").write_bytes(payload)
    launches = int(os.environ.get("FAKE_APP_CONDENSATION_LAUNCHES", "3"))
    volume_launches = int(os.environ.get(
        "FAKE_APP_VOLUME_ASSEMBLY_LAUNCHES", "3"
    ))
    p1_igr = bool(os.environ.get("FAKE_APP_P1_IGR"))
    order = 1 if p1_igr else 3
    interior_unknowns = 75712 if p1_igr else 302848
    condensation_tiles_per_launch = 889616 if p1_igr else 22827168
    volume_tiles_per_assembly = 2952768 if p1_igr else 22145760
    resolved_trace_policy = (
        "native-element-schwarz-matrix-free-warp-"
        "equilibrated-field-split-fp64"
    )
    top_level_resolved_trace_policy = (
        "engine-default" if os.environ.get("FAKE_APP_BAD_TOP_RESOLVED")
        else resolved_trace_policy
    )
    sample_resolved_trace_policy = (
        "engine-default" if os.environ.get("FAKE_APP_BAD_SAMPLE_RESOLVED")
        else resolved_trace_policy
    )
    print(json.dumps({
        "schema": 2,
        "benchmark": "dg_igr_hdg_cuda_accepted_step",
        "success": True,
        "variant": "production",
        "production_selectable": True,
        "backend": "cuda",
        "cuda_engine": "tensor",
        "trace_policy": "engine-default",
        "requested_trace_policy": "engine-default",
        "resolved_trace_policy": top_level_resolved_trace_policy,
        "trace_preconditioner":
            "native_element_schwarz_warp_equilibrated_field_split_fp64",
        "trace_field_split_sigma_jacobi_sweeps": 0,
        "trace_field_split_sigma_jacobi_damping": 0.0,
        "mode": "igr",
        "rank_count": 1,
        "polynomial_order": order,
        "mesh_n": 52,
        "domain_periods": 1,
        "dt": 1e-5,
        "elements": 2704,
        "volume_assembly_order": 12,
        "volume_points": 49,
        "interior_components": 7,
        "trace_components": 5,
        "interior_unknowns": interior_unknowns,
        "profiling_observability": True,
        "profile_measured_step": 2,
        "host_export": False,
        "samples": [{
            "phase": "measured",
            "measured_index": 2,
            "tensor_core": {
                "selected_engine": "tensor",
                "requested_trace_policy": "engine-default",
                "resolved_trace_policy": sample_resolved_trace_policy,
                "trace_preconditioner":
                    "native_element_schwarz_warp_"
                    "equilibrated_field_split_fp64",
                "element_schwarz_preconditioner_applications": 9,
                "field_split_sigma_jacobi_sweeps": 0,
                "field_split_sigma_jacobi_damping": 0.0,
                "field_split_sigma_jacobi_updates": 0,
                "hardware_supported": True,
                "volume_assembly_launches": volume_launches,
                "volume_mma_tiles": volume_tiles_per_assembly * volume_launches,
                "condensation_launches": launches,
                "mma_tiles": condensation_tiles_per_launch * launches,
                "fallback_count": 0,
                "fallback_accounting_complete": True,
                "no_fallback": True,
            },
        }],
    }, separators=(",", ":")))
""",
        )
        fake_nvidia_smi = qualification_tools / "nvidia-smi"
        executable(
            fake_nvidia_smi,
            "print('0, GPU-QUALIFICATION, NVIDIA RTX PRO, 999.1, 12.0')\n",
        )
        fake_qualification_nsys = qualification_tools / "nsys"
        nsys_profiler_version = "NVIDIA Nsight Systems version 2026.1"
        executable(
            fake_qualification_nsys,
            """import json
import os
import pathlib
import sys

arguments = sys.argv[1:]
if arguments == ["--version"]:
    print("NVIDIA Nsight Systems version 2026.1")
elif arguments and arguments[0] == "stats":
    expected_reports = {
        "nvtx_sum", "cuda_api_sum", "cuda_gpu_mem_time_sum",
        "cuda_gpu_mem_size_sum",
    }
    report = arguments[arguments.index("--report") + 1]
    expected = [
        "stats", "--report", report, "--format", "csv", "--output", "-",
        "--force-export", "true", str(pathlib.Path.cwd() / "capture.nsys-rep"),
    ]
    if report not in expected_reports or arguments != expected:
        raise SystemExit(17)
    if os.environ.get("FAKE_NSYS_STATS_FAIL"):
        raise SystemExit(18)
    if os.environ.get("FAKE_NSYS_STATS_EMPTY"):
        raise SystemExit(0)
    if os.environ.get("FAKE_NSYS_STATS_TAMPER") and \
            report == "cuda_gpu_mem_size_sum":
        (pathlib.Path.cwd() / "nsys-nvtx-sum.csv").write_text("tampered")
    print('"Report","Time (%)","Total Time (ns)","Instances"')
    print(f'"{report}","100.0","1234","2"')
else:
    stem = pathlib.Path(arguments[arguments.index("--output") + 1])
    stem.with_suffix(".nsys-rep").write_bytes(
        b"NVIDIA Tegra Profiler Report synthetic"
    )
    print(json.dumps({
        "schema": 2,
        "benchmark": "dg_igr_hdg_cuda_accepted_step",
        "success": True,
        "variant": "production",
        "production_selectable": True,
        "backend": "cuda",
        "cuda_engine": "tensor",
        "trace_policy": "engine-default",
        "requested_trace_policy": "engine-default",
        "resolved_trace_policy": "engine-default",
        "trace_preconditioner": "hypre_diag_scale",
        "trace_field_split_sigma_jacobi_sweeps": 0,
        "trace_field_split_sigma_jacobi_damping": 0.0,
        "mode": "conventional",
        "rank_count": 1,
        "polynomial_order": 2,
        "mesh_n": 52,
        "domain_periods": 1,
        "dt": 1e-5,
        "elements": 2704,
        "volume_assembly_order": 6,
        "volume_points": 16,
        "interior_components": 4,
        "trace_components": 4,
        "interior_unknowns": 97344,
        "profiling_observability": True,
        "profile_measured_step": 2,
        "host_export": False,
        "samples": [{
            "phase": "measured",
            "measured_index": 2,
            "tensor_core": {
                "selected_engine": "tensor",
                "requested_trace_policy": "engine-default",
                "resolved_trace_policy": "engine-default",
                "trace_preconditioner": "hypre_diag_scale",
                "element_schwarz_preconditioner_applications": 0,
                "field_split_sigma_jacobi_sweeps": 0,
                "field_split_sigma_jacobi_damping": 0.0,
                "field_split_sigma_jacobi_updates": 0,
                "hardware_supported": True,
                "volume_assembly_launches": 3,
                "volume_mma_tiles": 3893760,
                "condensation_launches": 3,
                "mma_tiles": 5159232,
                "fallback_count": 0,
                "fallback_accounting_complete": True,
                "no_fallback": True,
            },
        }],
    }, separators=(",", ":")))
""",
        )

        def qualification_arguments(
                output: Path,
                kind: str = "tensor-instructions",
                p1_igr: bool = False) -> list[str]:
            application = [
                "--arg=--mode", "--arg", "igr",
                "--arg=--engine", "--arg", "tensor",
                "--arg=--trace-policy", "--arg", "engine-default",
                "--arg=--order", "--arg", "1" if p1_igr else "3",
                "--arg=--mesh-n", "--arg", "52",
                "--arg=--domain-periods", "--arg", "1",
                "--arg=--dt", "--arg", "1e-5",
                "--arg=--profile-measured-step", "--arg", "2",
                "--arg=--no-export",
                "--arg=--expected-source-revision", "--arg",
                qualification_commit,
                "--arg=--expected-source-tree", "--arg", qualification_tree,
                "--arg=--expected-source-status", "--arg", "clean",
                "--arg=--expected-build-directory", "--arg",
                str(qualification_build.resolve()),
                "--arg=--expected-cmake-cache", "--arg",
                str(qualification_cache.resolve()),
            ]
            if kind == "tensor-instructions":
                ncu_arguments = [
                    "--ncu-metric",
                    "sm__inst_executed_pipe_fp64_op_dmma.sum",
                    "--ncu-metric", "gpu__time_duration.avg",
                    "--ncu-kernel-name-base", "demangled",
                    "--ncu-kernel-name",
                    "regex:.*TensorBorrowedCondenseKernel.*",
                    "--ncu-launch-count", "3",
                    "--ncu-nvtx-include", "HDG\\/condensation/",
                ]
            elif kind == "tensor-volume-instructions":
                ncu_arguments = [
                    "--ncu-metric",
                    "sm__inst_executed_pipe_fp64_op_dmma.sum",
                    "--ncu-metric", "gpu__time_duration.avg",
                    "--ncu-kernel-name-base", "demangled",
                    "--ncu-kernel-name",
                    "regex:.*BorrowedEulerTensorVolumeContractionKernel.*",
                    "--ncu-launch-count", "12" if p1_igr else "3",
                    "--ncu-nvtx-include", "HDG\\/local-assembly/",
                ]
            else:
                ncu_arguments = [
                    "--ncu-metric", "gpu__time_duration.avg",
                    "--ncu-kernel-name-base", "demangled",
                    "--ncu-launch-count", "20",
                    "--ncu-nvtx-include",
                    "HDG\\/profiled-accepted-step/",
                ]
            return [
                "--level", "l2-ncu", "--evidence-mode", "qualification",
                "--scenario", f"igr-p{1 if p1_igr else 3}-{kind}",
                "--exe", str(qualification_app), "--out-dir", str(output),
                "--source-root", str(qualification_source),
                "--expected-commit", qualification_commit,
                "--expected-tree", qualification_tree,
                "--require-clean-source",
                "--build-manifest", str(qualification_cache),
                "--mfem-identity", str(qualification_mfem),
                "--gpu-identity", "GPU-QUALIFICATION", "--gpu-index", "0",
                "--nvidia-smi-executable", str(fake_nvidia_smi),
                "--environment", "CUDA_VISIBLE_DEVICES",
                "--tool-executable", str(fake_ncu),
                "--tool-version", profiler_version,
                "--ncu-target-processes", "application-only",
                "--ncu-qualification-kind", kind,
                *ncu_arguments,
                "--require-data", "report=capture.ncu-rep",
                *application,
            ]

        qualification_env = {
            **os.environ, "CUDA_VISIBLE_DEVICES": "0",
        }
        qualification_out = root / "qualification"
        run(*qualification_arguments(qualification_out), env=qualification_env)
        qualification_manifest = read_manifest(qualification_out)
        assert qualification_manifest["outcome"] == "success"
        assert qualification_manifest["evidence_mode"] == "qualification"
        proof = qualification_manifest["qualification"][
            "tensor_instruction_proof"
        ]
        assert proof["status"] == "positive_tensor_instruction_proof"
        assert proof["matched_launches"] == 3
        assert proof["dmma_instruction_total"] == 68481504.0
        assert proof["dmma_instruction_minimum"] == 22827168.0
        assert proof["dmma_instruction_maximum"] == 22827168.0
        assert proof["complete_planned_dmma_match"] is True
        assert proof["planned_dmma_by_fused_stage"] == {
            "blocked_backward_rhs_update": 16240224,
            "blocked_forward_rhs_update": 16240224,
            "blocked_lu_trailing_update": 13287456,
            "schur_contraction": 22713600,
        }
        application_work = qualification_manifest["qualification"][
            "application_work"
        ]
        assert application_work["condensation_launches"] == 3
        assert application_work["planned_mma_tiles"] == 68481504
        assert application_work["volume_assembly_launches"] == 3
        assert application_work["volume_contraction_launches"] == 3
        assert application_work["planned_volume_mma_tiles"] == 66437280
        assert application_work["all_intended_fused_stages_material"] is True
        assert proof["duration_unit"] == "nanoseconds"
        assert proof["raw_duration_unit_field"] == "nsecond"
        assert proof["import_command"][-2:] == ["--print-units", "base"]
        bound = qualification_manifest["identity"]["qualification"]
        assert bound["gpu"]["uuid"] == "GPU-QUALIFICATION"
        assert bound["gpu"]["compute_capability"] == "12.0"
        assert bound["profiler"]["version"] == profiler_version
        assert len(bound["profiler"]["executable"]["sha256"]) == 64
        assert bound["build"]["validated_cmake_values"][
            "DG_IGR_OBSERVABILITY_USE_NVTX"
        ] == "ON"
        proof_artifact = next(
            item for item in qualification_manifest["artifacts"]
            if item["path"] == "ncu-tensor-proof.csv"
        )
        assert "--import" in proof_artifact["generating_command"]
        assert proof_artifact["generating_command"][-2:] == [
            "--print-units", "base",
        ]

        for suffix, environment, expected_detail in (
            (
                "bad-top-resolved", "FAKE_APP_BAD_TOP_RESOLVED",
                "does not match the qualified production tensor",
            ),
            (
                "bad-sample-resolved", "FAKE_APP_BAD_SAMPLE_RESOLVED",
                "does not prove an admitted no-fallback tensor engine",
            ),
        ):
            rejected_out = root / f"qualification-{suffix}"
            run(
                *qualification_arguments(rejected_out), expected=6,
                env={**qualification_env, environment: "1"},
            )
            assert expected_detail in \
                read_manifest(rejected_out)["detail"]

        volume_qualification_out = root / "qualification-volume"
        run(
            *qualification_arguments(
                volume_qualification_out, "tensor-volume-instructions", True
            ),
            env={
                **qualification_env,
                "FAKE_NCU_VOLUME": "1",
                "FAKE_APP_P1_IGR": "1",
            },
        )
        volume_manifest = read_manifest(volume_qualification_out)
        volume_proof = volume_manifest["qualification"][
            "tensor_instruction_proof"
        ]
        assert volume_proof["status"] == \
            "positive_tensor_instruction_proof"
        assert volume_proof["qualified_hotspot"] == \
            "tensor-volume-contraction"
        assert volume_proof["kernel_substring"] == \
            "BorrowedEulerTensorVolumeContractionKernel"
        assert volume_proof["matched_launches"] == 12
        assert volume_proof["dmma_instruction_total"] == 8858304.0
        assert volume_proof["dmma_instruction_minimum"] == 227136.0
        assert volume_proof["dmma_instruction_maximum"] == 908544.0
        assert volume_proof["dmma_instruction_histogram"] == [
            {"dmma_instructions": 227136.0, "launches": 3},
            {"dmma_instructions": 908544.0, "launches": 9},
        ]
        assert volume_proof["complete_planned_dmma_match"] is True
        assert volume_proof["volume_assembly_order"] == 12
        assert volume_proof["volume_points"] == 49
        assert volume_proof[
            "planned_volume_mma_tiles_per_assembly"
        ] == 2952768
        assert volume_proof[
            "planned_volume_contraction_launch_distribution"
        ] == [
            {
                "quadrature_points": 16,
                "quadrature_groups": 4,
                "mma_tiles_per_launch": 908544,
                "launches_per_assembly": 3,
            },
            {
                "quadrature_points": 1,
                "quadrature_groups": 1,
                "mma_tiles_per_launch": 227136,
                "launches_per_assembly": 1,
            },
        ]
        volume_work = volume_manifest["qualification"]["application_work"]
        assert volume_work["volume_assembly_launches"] == 3
        assert volume_work["volume_workspace_capacity_points"] == 17
        assert volume_work["volume_chunk_points"] == 16
        assert volume_work["volume_contraction_launches"] == 12
        assert volume_work["planned_volume_mma_tiles"] == 8858304

        attribution_out = root / "qualification-attribution"
        run(
            *qualification_arguments(
                attribution_out, "accepted-step-attribution"
            ),
            env={**qualification_env, "FAKE_NCU_ATTRIBUTION": "1"},
        )
        attribution_manifest = read_manifest(attribution_out)
        attribution = attribution_manifest["qualification"][
            "accepted_step_attribution"
        ]
        assert attribution["status"] == \
            "positive_accepted_step_attribution"
        assert attribution["matched_launches"] == 14
        assert attribution["total_gpu_kernel_duration_ns"] == 20400.0
        assert attribution["tensor_condensation"]["launches"] == 3
        assert attribution["tensor_condensation"]["duration_ns"] == 6000.0
        assert attribution["import_command"][-2:] == [
            "--print-units", "base",
        ]
        assert math.isclose(
            attribution["tensor_condensation"][
                "fraction_of_gpu_kernel_duration"
            ],
            6000.0 / 20400.0,
        )
        assert attribution["tensor_volume_assembly"]["launches"] == 3
        assert attribution["tensor_volume_assembly"]["duration_ns"] == 4500.0
        assert math.isclose(
            attribution["tensor_volume_assembly"][
                "fraction_of_gpu_kernel_duration"
            ],
            4500.0 / 20400.0,
        )
        assert set(attribution["required_categories"]) <= {
            item["category"] for item in attribution["categories"]
            if item["launches"] > 0
        }
        attribution_artifact = next(
            item for item in attribution_manifest["artifacts"]
            if item["path"] == "ncu-accepted-step-attribution.csv"
        )
        assert "--import" in attribution_artifact["generating_command"]
        assert attribution_artifact["generating_command"][-2:] == [
            "--print-units", "base",
        ]

        truncated_attribution_arguments = qualification_arguments(
            root / "qualification-attribution-at-cap",
            "accepted-step-attribution",
        )
        launch_count_index = truncated_attribution_arguments.index(
            "--ncu-launch-count"
        ) + 1
        truncated_attribution_arguments[launch_count_index] = "9"
        truncated_attribution = run(
            *truncated_attribution_arguments,
            expected=6,
            env={**qualification_env, "FAKE_NCU_ATTRIBUTION": "1"},
        )
        assert "strictly fewer than the declared capture limit" in \
            read_manifest(
                root / "qualification-attribution-at-cap"
            )["detail"]
        assert truncated_attribution.returncode == 6

        skipped_qualification_arguments = qualification_arguments(
            root / "qualification-launch-skip"
        )
        skipped_qualification_arguments.extend(("--ncu-launch-skip", "1"))
        assert "forbids --ncu-launch-skip" in run(
            *skipped_qualification_arguments,
            expected=2,
            env=qualification_env,
        ).stderr

        missing_trace_policy_arguments = qualification_arguments(
            root / "qualification-missing-trace-policy"
        )
        trace_policy_index = missing_trace_policy_arguments.index(
            "--arg=--trace-policy"
        )
        del missing_trace_policy_arguments[
            trace_policy_index:trace_policy_index + 3
        ]
        assert "requires exactly one application --trace-policy" in run(
            *missing_trace_policy_arguments,
            expected=2,
            env=qualification_env,
        ).stderr

        ablation_trace_policy_arguments = qualification_arguments(
            root / "qualification-ablation-trace-policy"
        )
        ablation_trace_policy_arguments[
            ablation_trace_policy_arguments.index("engine-default")
        ] = "common-fast"
        ablation_trace_policy_error = run(
            *ablation_trace_policy_arguments,
            expected=2,
            env=qualification_env,
        ).stderr
        assert "application --trace-policy='common-fast'" in \
            ablation_trace_policy_error
        assert "expected 'engine-default'" in ablation_trace_policy_error

        fixed_trace_policy_arguments = qualification_arguments(
            root / "qualification-fixed-sigma-jacobi7-trace-policy"
        )
        fixed_trace_policy = (
            "native-element-schwarz-matrix-free-warp-"
            "equilibrated-field-split-sigma-jacobi7-fp64"
        )
        fixed_trace_policy_arguments[
            fixed_trace_policy_arguments.index("engine-default")
        ] = fixed_trace_policy
        fixed_trace_policy_error = run(
            *fixed_trace_policy_arguments,
            expected=2,
            env=qualification_env,
        ).stderr
        assert f"application --trace-policy={fixed_trace_policy!r}" in \
            fixed_trace_policy_error
        assert "expected 'engine-default'" in fixed_trace_policy_error

        missing_category_out = root / "qualification-attribution-missing"
        run(
            *qualification_arguments(
                missing_category_out, "accepted-step-attribution"
            ),
            expected=6,
            env={
                **qualification_env,
                "FAKE_NCU_ATTRIBUTION": "1",
                "FAKE_NCU_MISSING_CATEGORY": "1",
            },
        )
        missing_category_manifest = read_manifest(missing_category_out)
        assert missing_category_manifest["outcome"] == "validation_failure"
        assert "missing material accepted-step categories" in \
            missing_category_manifest["detail"]

        missing_volume_out = root / "qualification-attribution-missing-volume"
        run(
            *qualification_arguments(
                missing_volume_out, "accepted-step-attribution"
            ),
            expected=6,
            env={
                **qualification_env,
                "FAKE_NCU_ATTRIBUTION": "1",
                "FAKE_NCU_MISSING_VOLUME": "1",
            },
        )
        assert "no positive tensor-volume contraction" in read_manifest(
            missing_volume_out
        )["detail"]

        nsys_application = [
            "--arg=--mode", "--arg", "conventional",
            "--arg=--engine", "--arg", "tensor",
            "--arg=--trace-policy", "--arg", "engine-default",
            "--arg=--order", "--arg", "2",
            "--arg=--mesh-n", "--arg", "52",
            "--arg=--domain-periods", "--arg", "1",
            "--arg=--dt", "--arg", "1e-5",
            "--arg=--profile-measured-step", "--arg", "2",
            "--arg=--no-export",
            "--arg=--expected-source-revision", "--arg",
            qualification_commit,
            "--arg=--expected-source-tree", "--arg", qualification_tree,
            "--arg=--expected-source-status", "--arg", "clean",
            "--arg=--expected-build-directory", "--arg",
            str(qualification_build.resolve()),
            "--arg=--expected-cmake-cache", "--arg",
            str(qualification_cache.resolve()),
        ]
        nsys_qualification_out = root / "qualification-nsys"
        nsys_qualification_arguments = [
            "--level", "l2-nsys", "--evidence-mode", "qualification",
            "--scenario", "conventional-p2-accepted-step-nsys",
            "--exe", str(qualification_app),
            "--out-dir", str(nsys_qualification_out),
            "--source-root", str(qualification_source),
            "--expected-commit", qualification_commit,
            "--expected-tree", qualification_tree,
            "--require-clean-source",
            "--build-manifest", str(qualification_cache),
            "--mfem-identity", str(qualification_mfem),
            "--gpu-identity", "GPU-QUALIFICATION", "--gpu-index", "0",
            "--nvidia-smi-executable", str(fake_nvidia_smi),
            "--environment", "CUDA_VISIBLE_DEVICES",
            "--tool-executable", str(fake_qualification_nsys),
            "--tool-version", nsys_profiler_version,
            "--nsys-trace", "cuda,nvtx", "--nsys-sample", "none",
            "--nsys-capture-range", "cudaProfilerApi",
            "--nsys-capture-range-end", "stop",
            "--require-data", "report=capture.nsys-rep",
            *nsys_application,
        ]
        run(*nsys_qualification_arguments, env=qualification_env)
        nsys_qualification_manifest = read_manifest(nsys_qualification_out)
        assert nsys_qualification_manifest["outcome"] == "success"
        assert nsys_qualification_manifest["qualification"]["status"] == \
            "success"
        nsys_summaries = nsys_qualification_manifest["qualification"][
            "nsys_summaries"
        ]
        assert set(nsys_summaries) == {
            "nvtx_sum", "cuda_api_sum", "cuda_gpu_mem_time_sum",
            "cuda_gpu_mem_size_sum",
        }
        for report_name, summary in nsys_summaries.items():
            assert summary["status"] == "generated_from_native_report"
            assert summary["csv"]["bytes"] > 0
            assert len(summary["csv"]["sha256"]) == 64
            assert summary["command"][1:] == [
                "stats", "--report", report_name,
                "--format", "csv", "--output", "-",
                "--force-export", "true",
                str(nsys_qualification_out / "capture.nsys-rep"),
            ]
        nsys_artifact_paths = {
            artifact["path"]
            for artifact in nsys_qualification_manifest["artifacts"]
        }
        assert {
            "nsys-nvtx-sum.csv", "nsys-cuda-api-sum.csv",
            "nsys-cuda-gpu-mem-time-sum.csv",
            "nsys-cuda-gpu-mem-size-sum.csv",
        } <= nsys_artifact_paths
        assert nsys_qualification_manifest["identity"]["qualification"][
            "profiler"
        ]["version"] == nsys_profiler_version

        wrong_nsys_scope = list(nsys_qualification_arguments)
        wrong_nsys_scope[
            wrong_nsys_scope.index("stop")
        ] = "repeat"
        wrong_nsys_scope[
            wrong_nsys_scope.index(str(nsys_qualification_out))
        ] = str(root / "qualification-nsys-wrong-scope")
        assert "requires accepted-step cudaProfilerApi capture" in run(
            *wrong_nsys_scope, expected=2, env=qualification_env
        ).stderr

        for suffix, environment, expected_detail in (
            ("stats-fail", "FAKE_NSYS_STATS_FAIL", "summary exited 18"),
            ("stats-empty", "FAKE_NSYS_STATS_EMPTY", "summary is empty"),
            ("stats-tamper", "FAKE_NSYS_STATS_TAMPER", "changed after validation"),
        ):
            negative_arguments = list(nsys_qualification_arguments)
            negative_arguments[
                negative_arguments.index(str(nsys_qualification_out))
            ] = str(root / f"qualification-nsys-{suffix}")
            run(
                *negative_arguments, expected=6,
                env={**qualification_env, environment: "1"},
            )
            assert expected_detail in read_manifest(
                root / f"qualification-nsys-{suffix}"
            )["detail"]

        wrong_nsys_application = list(nsys_qualification_arguments)
        wrong_nsys_application[
            wrong_nsys_application.index("tensor")
        ] = "ordinary"
        wrong_nsys_application[
            wrong_nsys_application.index(str(nsys_qualification_out))
        ] = str(root / "qualification-nsys-wrong-engine")
        assert "expected 'tensor'" in run(
            *wrong_nsys_application, expected=2, env=qualification_env
        ).stderr

        wrong_nsys_trace_policy = list(nsys_qualification_arguments)
        wrong_nsys_trace_policy[
            wrong_nsys_trace_policy.index("engine-default")
        ] = "common-fast"
        wrong_nsys_trace_policy[
            wrong_nsys_trace_policy.index(str(nsys_qualification_out))
        ] = str(root / "qualification-nsys-ablation-trace-policy")
        wrong_nsys_trace_policy_error = run(
            *wrong_nsys_trace_policy, expected=2, env=qualification_env
        ).stderr
        assert "application --trace-policy='common-fast'" in \
            wrong_nsys_trace_policy_error
        assert "expected 'engine-default'" in \
            wrong_nsys_trace_policy_error

        zero_out = root / "qualification-zero"
        run(
            *qualification_arguments(zero_out), expected=6,
            env={**qualification_env, "FAKE_NCU_ZERO": "1"},
        )
        zero_manifest = read_manifest(zero_out)
        assert zero_manifest["outcome"] == "validation_failure"
        assert zero_manifest["qualification"]["status"] == "failure"
        assert "must be finite and positive" in zero_manifest["detail"]

        wrong_total_out = root / "qualification-wrong-dmma-total"
        run(
            *qualification_arguments(wrong_total_out), expected=6,
            env={**qualification_env, "FAKE_NCU_WRONG_TOTAL": "1"},
        )
        assert "complete planned MMA work" in read_manifest(
            wrong_total_out
        )["detail"]

        wrong_volume_total_out = root / "qualification-wrong-volume-dmma-total"
        run(
            *qualification_arguments(
                wrong_volume_total_out, "tensor-volume-instructions"
            ),
            expected=6,
            env={
                **qualification_env,
                "FAKE_NCU_VOLUME": "1",
                "FAKE_NCU_WRONG_TOTAL": "1",
            },
        )
        assert "tensor-volume-contraction DMMA work" in read_manifest(
            wrong_volume_total_out
        )["detail"]

        truncated_out = root / "qualification-truncated-launches"
        run(
            *qualification_arguments(truncated_out), expected=6,
            env={
                **qualification_env,
                "FAKE_APP_CONDENSATION_LAUNCHES": "4",
            },
        )
        assert "does not match the profiled accepted step's 4 " \
            "fused-local-algebra-condensation launches" \
            in read_manifest(truncated_out)["detail"]

        bad_report_out = root / "qualification-bad-report"
        run(
            *qualification_arguments(bad_report_out), expected=6,
            env={**qualification_env, "FAKE_NCU_BAD_REPORT": "1"},
        )
        bad_report_manifest = read_manifest(bad_report_out)
        assert bad_report_manifest["qualification"]["status"] == "failure"
        assert "expected native ncu report signature" in \
            bad_report_manifest["detail"]

        wrong_kernel_arguments = qualification_arguments(
            root / "qualification-wrong-kernel"
        )
        kernel_index = wrong_kernel_arguments.index(
            "regex:.*TensorBorrowedCondenseKernel.*"
        )
        wrong_kernel_arguments[kernel_index] = "regex:.*OtherKernel.*"
        assert "exact tensor condensation kernel filter" in run(
            *wrong_kernel_arguments, expected=2, env=qualification_env,
        ).stderr

        l1 = root / "l1"
        run("--level", "l1", *common(app, l1), "--warmups", "1", "--repetitions", "5")
        l1_manifest = read_manifest(l1)
        assert len(l1_manifest["timing_seconds"]) == 5
        assert l1_manifest["timing_statistics"]["measured_repetitions"] == 5
        assert "median_seconds" in l1_manifest["timing_statistics"] and "dispersion_seconds" in l1_manifest["timing_statistics"]
        assert (l1 / "execution.log").read_text(encoding="utf-8").count("app --input case.yaml") == 6
        nsys = root / "fake-nsys"
        executable(nsys, "import pathlib, sys\na=sys.argv\nstem=pathlib.Path(a[a.index('--output') + 1])\nstem.with_suffix('.nsys-rep').write_text('report')\n(pathlib.Path.cwd() / 'nvtx.csv').write_text('table')\n")
        nsys_out = root / "nsys"
        run("--level", "l2-nsys", "--evidence-mode", "smoke", *common(app, nsys_out), "--tool-executable", os.path.relpath(nsys, ROOT),
            "--nsys-trace", "cuda,nvtx", "--nsys-sample", "none", "--require-data", "nvtx=nvtx.csv",
            "--nsys-capture-range", "cudaProfilerApi",
            "--nsys-capture-range-end", "stop",
            "--mfem-identity", "mfem-test", "--gpu-identity", "gpu-test")
        nsys_manifest = read_manifest(nsys_out)
        assert nsys_manifest["outcome"] == "success" and nsys_manifest["identity"]["mfem"] == "mfem-test"
        nsys_command = nsys_manifest["generating_command"]
        assert nsys_command[0] == str(nsys.resolve())
        assert nsys_command[nsys_command.index("--trace") + 1] == "cuda,nvtx"
        assert nsys_command[nsys_command.index("--sample") + 1] == "none"
        assert nsys_command[nsys_command.index("--capture-range") + 1] == \
            "cudaProfilerApi"
        assert nsys_command[nsys_command.index("--capture-range-end") + 1] == \
            "stop"
        assert any(item["path"] == "nvtx.csv" for item in nsys_manifest["artifacts"])
        nsys_nvtx = root / "nsys-nvtx"
        run("--level", "l2-nsys", "--evidence-mode", "smoke", *common(app, nsys_nvtx),
            "--tool-executable", str(nsys), "--nsys-trace", "cuda,nvtx",
            "--nsys-sample", "none", "--nsys-capture-range", "nvtx",
            "--nsys-capture-range-end", "repeat::defer",
            "--nsys-nvtx-capture", "accepted-step@dg-igr",
            "--require-data", "nvtx=nvtx.csv", "--dry-run")
        nsys_nvtx_command = read_manifest(nsys_nvtx)["generating_command"]
        assert nsys_nvtx_command[nsys_nvtx_command.index("--capture-range") + 1] == "nvtx"
        assert nsys_nvtx_command[nsys_nvtx_command.index("--capture-range-end") + 1] == "repeat::defer"
        assert nsys_nvtx_command[nsys_nvtx_command.index("--nvtx-capture") + 1] == "accepted-step@dg-igr"
        missing_capture_range = root / "missing-capture-range"
        assert "requires --nsys-capture-range" in run(
            "--level", "l2-nsys", "--evidence-mode", "smoke", *common(app, missing_capture_range),
            "--tool-executable", str(nsys), "--nsys-trace", "cuda",
            "--nsys-sample", "none", "--nsys-capture-range-end", "stop",
            "--require-data", "nvtx=nvtx.csv", expected=2,
        ).stderr
        missing_nvtx_trigger = root / "missing-nvtx-trigger"
        assert "requires --nsys-nvtx-capture" in run(
            "--level", "l2-nsys", "--evidence-mode", "smoke", *common(app, missing_nvtx_trigger),
            "--tool-executable", str(nsys), "--nsys-trace", "cuda,nvtx",
            "--nsys-sample", "none", "--nsys-capture-range", "nvtx",
            "--nsys-capture-range-end", "stop",
            "--require-data", "nvtx=nvtx.csv", expected=2,
        ).stderr
        ncu = root / "fake-ncu"
        executable(ncu, "import pathlib\n(pathlib.Path.cwd() / 'summary.txt').write_text('summary')\n")
        ncu_out = root / "ncu"
        run("--level", "l2-ncu", "--evidence-mode", "smoke", *common(app, ncu_out), "--tool-executable", str(ncu),
            "--ncu-target-processes", "all",
            "--ncu-set", "basic", "--ncu-section", "Occupancy", "--ncu-kernel-name-base", "demangled",
            "--ncu-metric", "sm__inst_executed_pipe_fp64_op_dmma.sum",
            "--ncu-metric", "gpu__time_duration.avg",
            "--ncu-kernel-name", "kernel.*", "--ncu-launch-count", "3", "--ncu-launch-skip", "2",
            "--ncu-nvtx-include", "DG/*", "--require-data", "summary=summary.txt")
        ncu_command = read_manifest(ncu_out)["generating_command"]
        for option in ("--set", "--section", "--metrics", "--kernel-name-base", "--kernel-name", "--launch-count", "--launch-skip", "--nvtx", "--nvtx-include"):
            assert option in ncu_command
        assert ncu_command[ncu_command.index("--target-processes") + 1] == "all"
        assert ncu_command[ncu_command.index("--metrics") + 1] == (
            "sm__inst_executed_pipe_fp64_op_dmma.sum,gpu__time_duration.avg"
        )
        ncu_dry = root / "ncu-dry"
        run("--level", "l2-ncu", "--evidence-mode", "smoke", *common(app, ncu_dry), "--tool-executable", str(ncu),
            "--ncu-target-processes", "application-only",
            "--ncu-metric", "sm__inst_executed_pipe_fp64_op_dmma.sum",
            "--require-data", "summary=summary.txt", "--dry-run")
        dry_ncu_command = read_manifest(ncu_dry)["generating_command"]
        assert dry_ncu_command[dry_ncu_command.index("--target-processes") + 1] == "application-only"
        assert dry_ncu_command[dry_ncu_command.index("--metrics") + 1] == \
            "sm__inst_executed_pipe_fp64_op_dmma.sum"
        duplicate_metric = root / "duplicate-metric"
        assert "duplicate --ncu-metric" in run(
            "--level", "l2-ncu", "--evidence-mode", "smoke", *common(app, duplicate_metric),
            "--tool-executable", str(ncu), "--ncu-target-processes", "all",
            "--ncu-metric", "gpu__time_duration.avg",
            "--ncu-metric", "gpu__time_duration.avg",
            "--require-data", "summary=summary.txt", expected=2,
        ).stderr
        missing_ncu_target = root / "missing-ncu-target"
        assert "ncu-target-processes" in run("--level", "l2-ncu", "--evidence-mode", "smoke", *common(app, missing_ncu_target),
                                               "--tool-executable", str(ncu), "--ncu-set", "basic",
                                               "--require-data", "summary=summary.txt", expected=2).stderr
        dry = root / "dry"
        run("--level", "l2-nsys", "--evidence-mode", "smoke", *common(app, dry), "--tool-executable", str(root / "not-installed"),
            "--nsys-trace", "cuda", "--nsys-sample", "none", "--require-data", "table=table.csv", "--dry-run")
        assert read_manifest(dry)["outcome"] == "dry_run"
        missing_tool = root / "missing-tool"
        run("--level", "l2-nsys", "--evidence-mode", "smoke", *common(app, missing_tool), "--tool-executable", str(root / "no-nsys"),
            "--nsys-trace", "cuda", "--nsys-sample", "none", "--require-data", "table=table.csv", expected=3)
        assert read_manifest(missing_tool)["outcome"] == "tool_unavailable"
        missing_data = root / "missing-data"
        run("--level", "l2-ncu", "--evidence-mode", "smoke", *common(app, missing_data), "--tool-executable", str(ncu),
            "--ncu-target-processes", "all",
            "--ncu-set", "basic", "--require-data", "missing=missing.csv", expected=4)
        assert read_manifest(missing_data)["outcome"] == "data_unavailable"
        empty_ncu = root / "fake-empty-ncu"
        executable(empty_ncu, "import pathlib\n(pathlib.Path.cwd() / 'empty.csv').write_text('')\n")
        empty_data = root / "empty-data"
        run("--level", "l2-ncu", "--evidence-mode", "smoke", *common(app, empty_data), "--tool-executable", str(empty_ncu),
            "--ncu-target-processes", "all",
            "--ncu-set", "basic", "--require-data", "empty=empty.csv", expected=4)
        assert read_manifest(empty_data)["outcome"] == "data_unavailable"
        bad = root / "bad-app"
        executable(bad, "raise SystemExit(7)\n")
        failed = root / "failed"
        run("--level", "l0", *common(bad, failed), expected=5)
        assert read_manifest(failed)["outcome"] == "execution_failure"
        bad_shebang = root / "bad-shebang"
        bad_shebang.write_text("#!/definitely/not/an/interpreter\n", encoding="utf-8")
        bad_shebang.chmod(0o755)
        launch_failure = root / "launch-failure"
        run("--level", "l0", *common(bad_shebang, launch_failure), expected=5)
        assert read_manifest(launch_failure)["outcome"] == "execution_failure"
        assert "could not execute command" in read_manifest(launch_failure)["detail"]
        rejected = root / "rejected"
        assert "relative path" in run("--level", "l2-nsys", "--evidence-mode", "smoke", *common(app, rejected), "--tool-executable", str(nsys),
                                       "--nsys-trace", "cuda", "--nsys-sample", "none", "--require-data", "bad=../escape.csv", expected=2).stderr
        log_alias = root / "log-alias"
        assert "reserve" in run("--level", "l2-nsys", "--evidence-mode", "smoke", *common(app, log_alias), "--tool-executable", str(nsys),
                                 "--nsys-trace", "cuda", "--nsys-sample", "none",
                                 "--require-data", "log=execution.log", expected=2).stderr
        secret = "do-not-write-this-secret"
        sensitive = root / "sensitive"
        sensitive_result = run("--level", "l0", *common(app, sensitive), "--environment", "API_TOKEN", expected=6,
                               env={**os.environ, "API_TOKEN": secret})
        sensitive_files = "".join(path.read_text(encoding="utf-8") for path in sensitive.rglob("*") if path.is_file())
        assert secret not in sensitive_result.stdout + sensitive_result.stderr + sensitive_files
        assert read_manifest(sensitive)["outcome"] == "validation_failure"
        assert read_manifest(sensitive)["environment"] == {}
        relative = root / "relative"
        run("--level", "l0", "--scenario", "relative-inputs", "--exe", os.path.relpath(app, ROOT),
            "--out-dir", os.path.relpath(relative, ROOT), "--source-root", ".")
        relative_manifest = read_manifest(relative)
        assert relative_manifest["outcome"] == "success"
        assert relative_manifest["identity"]["executable"]["path"] == str(app.resolve())
        assert relative_manifest["source_dirty_state"]["status"] == "available"
        victim = root / "outside-victim"
        victim.write_text("unchanged", encoding="utf-8")
        manifest_attacker = root / "manifest-attacker"
        executable(manifest_attacker, "import os\nos.symlink(os.environ['VICTIM'], 'profile-manifest.json')\n")
        manifest_attack = root / "manifest-attack"
        run("--level", "l0", *common(manifest_attacker, manifest_attack), expected=6,
            env={**os.environ, "VICTIM": str(victim)})
        assert victim.read_text(encoding="utf-8") == "unchanged"
        assert (manifest_attack / "profile-manifest.json").is_symlink()
        symlink_ncu = root / "symlink-ncu"
        executable(symlink_ncu, "import os\nos.symlink(os.environ['VICTIM'], 'outside.csv')\n")
        symlink_data = root / "symlink-data"
        run("--level", "l2-ncu", "--evidence-mode", "smoke", *common(app, symlink_data), "--tool-executable", str(symlink_ncu),
            "--ncu-target-processes", "all", "--ncu-set", "basic", "--require-data", "outside=outside.csv",
            expected=4, env={**os.environ, "VICTIM": str(victim)})
        assert victim.read_text(encoding="utf-8") == "unchanged"
        assert (symlink_data / "outside.csv").is_symlink()
        delete_pin = root / "delete-pin-ncu"
        executable(delete_pin, "import pathlib\np=pathlib.Path.cwd()\n(p / 'summary.txt').write_text('ok')\n(p / 'profile-provenance.json').unlink()\n")
        deleted_pin = root / "deleted-pin"
        run("--level", "l2-ncu", "--evidence-mode", "smoke", *common(app, deleted_pin), "--tool-executable", str(delete_pin),
            "--ncu-target-processes", "all", "--ncu-set", "basic", "--require-data", "summary=summary.txt", expected=6)
        assert read_manifest(deleted_pin)["outcome"] == "validation_failure"
        replace_pin = root / "replace-pin-ncu"
        executable(replace_pin, "import pathlib\np=pathlib.Path.cwd()\n(p / 'summary.txt').write_text('ok')\n(p / 'profile-provenance.json').write_text('{}')\n")
        replaced_pin = root / "replaced-pin"
        run("--level", "l2-ncu", "--evidence-mode", "smoke", *common(app, replaced_pin), "--tool-executable", str(replace_pin),
            "--ncu-target-processes", "all", "--ncu-set", "basic", "--require-data", "summary=summary.txt", expected=6)
        replacement_manifest = read_manifest(replaced_pin)
        assert replacement_manifest["outcome"] == "validation_failure"
        recreate_pin = root / "recreate-pin-ncu"
        executable(recreate_pin, "import pathlib\np=pathlib.Path.cwd()\ncontents=(p / 'profile-provenance.json').read_bytes()\n(p / 'summary.txt').write_text('ok')\n(p / 'profile-provenance.json').unlink()\n(p / 'profile-provenance.json').write_bytes(contents)\n")
        recreated_pin = root / "recreated-pin"
        run("--level", "l2-ncu", "--evidence-mode", "smoke", *common(app, recreated_pin), "--tool-executable", str(recreate_pin),
            "--ncu-target-processes", "all", "--ncu-set", "basic", "--require-data", "summary=summary.txt", expected=6)
        assert read_manifest(recreated_pin)["outcome"] == "validation_failure"
    print("profile-harness-test: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
