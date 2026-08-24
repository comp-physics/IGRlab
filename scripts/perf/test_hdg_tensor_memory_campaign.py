#!/usr/bin/env python3
"""No-build/no-GPU exact-binding tests for the peak-memory campaign."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
DRIVER_NAME = "hdg_tensor_memory_campaign.py"
DEPENDENCIES = (
    DRIVER_NAME,
    "hdg_tensor_campaign.py",
    "gpu_memory_control.py",
)


def _load_campaign_test_module():
    path = HERE / "test_hdg_tensor_campaign.py"
    spec = importlib.util.spec_from_file_location(
        "hdg_tensor_campaign_test_fixture_for_memory", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load campaign test fixtures: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXTURE = _load_campaign_test_module()
CAMPAIGN = FIXTURE.CAMPAIGN


FAKE_BENCHMARK = FIXTURE.FAKE_BENCHMARK.replace(
    "print(json.dumps(payload))",
    r'''
import os
import time

if os.environ.get("MEMORY_TEST_BAD_WORKLOAD") == "1":
    payload["mesh_n"] = args.mesh_n - 1
if os.environ.get("MEMORY_TEST_HIGH_INTERNAL_FLOOR") == "1":
    payload["device_memory_snapshots"] = {
        "after_device_selection_before_problem": {
            "free_bytes": 3000000000,
            "total_bytes": 4000000000,
            "used_bytes": 1000000000,
        },
        "after_initialization": {
            "free_bytes": 2000000000,
            "total_bytes": 4000000000,
            "used_bytes": 2000000000,
        },
        "after_samples": {
            "free_bytes": 1000000000,
            "total_bytes": 4000000000,
            "used_bytes": 3000000000,
        },
    }
marker = Path(os.environ["MEMORY_TEST_ACTIVE_MARKER"])
marker.write_text("active\n", encoding="utf-8")
try:
    time.sleep(0.015)
finally:
    marker.unlink(missing_ok=True)
print(json.dumps(payload))
''',
)


FAKE_NVIDIA_SMI = r'''#!/usr/bin/env python3
import datetime
import os
from pathlib import Path
import sys
import time

query = next(argument for argument in sys.argv if argument.startswith("--query-gpu="))
fields = query.split("=", 1)[1]
uuid = "GPU-00000000-0000-0000-0000-000000000001"
values = {
    "index": "0",
    "uuid": uuid,
    "name": "Fake SM120",
    "pci.bus_id": "00000000:01:00.0",
    "driver_version": "600.1",
    "compute_cap": "12.0",
    "memory.total": "49140",
    "timestamp": "2026/08/24 00:00:00.000",
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
if "," not in fields:
    print(values[fields])
    raise SystemExit(0)

marker = Path(os.environ["MEMORY_TEST_ACTIVE_MARKER"])
while True:
    active = marker.exists()
    if active and os.environ.get("MEMORY_TEST_BAD_PEAK") == "1":
        used = "not-a-number"
        free = "not-a-number"
    else:
        used_value = 1400 if active else 1000
        used = str(used_value)
        free = str(49140 - used_value)
    stamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
    print(f"{stamp}, {uuid}, 49140, {used}, {free}", flush=True)
    time.sleep(0.001)
'''


def write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def run_checked(command: list[str], cwd: Path,
                environment: dict[str, str] | None = None
                ) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )


def run_unchecked(command: list[str], cwd: Path,
                  environment: dict[str, str]
                  ) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )


def manifest_and_hash(output: Path) -> dict:
    path = output / "memory_campaign.json"
    content = path.read_bytes()
    sidecar = (output / "memory_campaign.sha256").read_text(
        encoding="utf-8"
    ).strip().split()
    assert sidecar == [hashlib.sha256(content).hexdigest(), path.name]
    return json.loads(content)


def main() -> int:
    with tempfile.TemporaryDirectory(
            prefix="hdg-tensor-memory-campaign-test-") as temp:
        root = Path(temp)
        source = root / "source"
        perf = source / "scripts" / "perf"
        perf.mkdir(parents=True)
        for filename in DEPENDENCIES:
            shutil.copy2(HERE / filename, perf / filename)
        (source / "tpl" / "mfem").mkdir(parents=True)
        (source / "marker.txt").write_text(
            "clean synthetic exact candidate\n", encoding="utf-8"
        )
        write_executable(source / "build.sh", "#!/bin/sh\nexit 0\n")
        run_checked(["git", "init", "-q"], source)
        run_checked(["git", "config", "user.name", "Memory Test"], source)
        run_checked([
            "git", "config", "user.email", "memory-test@example.invalid"
        ], source)
        run_checked(["git", "add", "."], source)
        run_checked(["git", "commit", "-q", "-m", "exact candidate"], source)
        commit = run_checked(
            ["git", "rev-parse", "HEAD"], source
        ).stdout.strip()
        tree = run_checked(
            ["git", "rev-parse", "HEAD^{tree}"], source
        ).stdout.strip()

        build_cache, mfem_identity = FIXTURE.write_build_provenance(
            root, source
        )
        benchmark = (
            build_cache.parent / CAMPAIGN.BENCHMARK_BUILD_RELATIVE_PATH
        )
        write_executable(benchmark, FAKE_BENCHMARK)
        nvidia_smi = root / "nvidia-smi"
        write_executable(nvidia_smi, FAKE_NVIDIA_SMI)
        copied_driver = perf / DRIVER_NAME

        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": "0",
            "MEMORY_TEST_ACTIVE_MARKER": str(root / "gpu-active.marker"),
        })

        def command(output: Path, *, gpu_uuid: str | None = None,
                    mesh_n: int = 52) -> list[str]:
            return [
                sys.executable,
                str(copied_driver),
                "--benchmark", str(benchmark),
                "--output-dir", str(output),
                "--source-root", str(source),
                "--evidence-mode", "qualification",
                "--workload-label", "sm120-mesh52-saturated",
                "--mesh-n", str(mesh_n),
                "--domain-periods", "1",
                "--dt", "1e-5",
                "--repeats", "3",
                "--poll-ms", "50",
                "--build-label", "synthetic-final",
                "--build-metadata", "cuda_arch=120",
                "--build-manifest", str(build_cache),
                "--mfem-identity", str(mfem_identity),
                "--expected-commit", commit,
                "--expected-tree", tree,
                "--gpu-index", "0",
                "--gpu-uuid", gpu_uuid or
                "GPU-00000000-0000-0000-0000-000000000001",
                "--nvidia-smi", str(nvidia_smi),
                "--hardware-metadata", "lease=synthetic",
                "--timeout-seconds", "5",
            ]

        success_output = root / "success"
        completed = run_unchecked(command(success_output), root, environment)
        assert completed.returncode == 0, completed.stderr
        completion = json.loads(completed.stdout)
        assert completion["qualification_completed"] is True
        assert completion["qualification_pass"] is True
        success = manifest_and_hash(success_output)
        assert success["status"] == "completed"
        assert success["qualification_pass"] is True
        assert success["method"]["post_timing_control"] is True
        assert success["method"]["primary_performance_eligible"] is False
        assert success["source_start"] == success["source_end"]
        assert success["workload"]["mesh_n"] == 52
        assert success["workload"]["repeats_per_engine_cell"] == 3
        assert success["workload"]["poll_ms"] == 50
        assert len(success["invocations"]) == 36
        assert len(success["results"]) == 6
        assert success["integrity"]["raw_artifacts"][
            "artifact_count"
        ] == 180
        for cell in success["results"].values():
            assert set(cell) == {"ordinary", "tensor"}
            for engine in cell.values():
                qualified = engine["whole_device_peak_memory"]
                assert qualified["qualified"] is True
                assert qualified["repeat_count"] == 3
                assert qualified["conservative_incremental_peak_mib"] == 400
                assert all(
                    check["passes"] is True
                    for check in qualified["internal_cuda_floor_checks"]
                )
        first = success["invocations"][0]
        assert first["command"][first["command"].index("--mesh-n") + 1] == "52"
        assert "--no-export" in first["command"]
        control_artifact = first["artifacts"]["control"]
        control_path = success_output / control_artifact["path"]
        assert control_artifact["identity"]["sha256"] == hashlib.sha256(
            control_path.read_bytes()
        ).hexdigest()
        control = json.loads(control_path.read_text(encoding="utf-8"))
        assert control["benchmark"]["stdout"]
        assert control["benchmark"]["stderr"] == ""
        assert control["benchmark"]["command"] == first["command"]
        assert len(control["poller"]["rows"]) >= 21
        assert control["poller"]["raw_stdout"]
        assert control["measurement"]["active_rows"]

        def assert_failure(output: Path, failed: subprocess.CompletedProcess[str]
                           ) -> dict:
            assert failed.returncode == 1
            evidence = manifest_and_hash(output)
            assert evidence["status"] == "failure"
            assert evidence["qualification_pass"] is False
            assert evidence["error"]
            assert evidence["integrity"]["raw_artifacts"][
                "ledger_sha256"
            ]
            return evidence

        identity_output = root / "bad-identity"
        identity = assert_failure(
            identity_output,
            run_unchecked(
                command(
                    identity_output,
                    gpu_uuid="GPU-00000000-0000-0000-0000-000000000002",
                ),
                root,
                environment,
            ),
        )
        assert identity["invocations"] == []
        assert "UUID" in identity["error"]

        profile_output = root / "bad-profile"
        profile = assert_failure(
            profile_output,
            run_unchecked(command(profile_output, mesh_n=51), root, environment),
        )
        assert profile["invocations"] == []
        assert "saturated profile" in profile["error"]

        bad_workload_environment = dict(environment)
        bad_workload_environment["MEMORY_TEST_BAD_WORKLOAD"] = "1"
        workload_output = root / "bad-workload-record"
        workload = assert_failure(
            workload_output,
            run_unchecked(
                command(workload_output), root, bad_workload_environment
            ),
        )
        assert len(workload["invocations"]) == 1
        assert workload["integrity"]["raw_artifacts"][
            "artifact_count"
        ] == 5
        assert "mesh_n" in workload["error"]

        bad_peak_environment = dict(environment)
        bad_peak_environment["MEMORY_TEST_BAD_PEAK"] = "1"
        peak_output = root / "bad-peak"
        peak = assert_failure(
            peak_output,
            run_unchecked(command(peak_output), root, bad_peak_environment),
        )
        assert len(peak["invocations"]) == 1
        assert peak["integrity"]["raw_artifacts"]["artifact_count"] == 5
        assert "invalid" in peak["error"] or "numeric" in peak["error"]

        high_floor_environment = dict(environment)
        high_floor_environment["MEMORY_TEST_HIGH_INTERNAL_FLOOR"] = "1"
        floor_output = root / "bad-floor"
        floor = assert_failure(
            floor_output,
            run_unchecked(command(floor_output), root, high_floor_environment),
        )
        assert len(floor["invocations"]) == 36
        assert floor["integrity"]["raw_artifacts"][
            "artifact_count"
        ] == 180
        assert "cudaMemGetInfo floor" in floor["error"]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
