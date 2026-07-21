#!/usr/bin/env python3
"""Deterministic non-capture regression tests for profile_harness.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts/perf/profile_harness.py"


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


def common(app: Path, output: Path) -> list[str]:
    return ["--scenario", "generic-case", "--exe", str(app), "--out-dir", str(output), "--arg=--input", "--arg", "case.yaml"]


def main() -> int:
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
        run("--level", "l2-nsys", *common(app, nsys_out), "--tool-executable", os.path.relpath(nsys, ROOT),
            "--nsys-trace", "cuda,nvtx", "--nsys-sample", "none", "--require-data", "nvtx=nvtx.csv",
            "--mfem-identity", "mfem-test", "--gpu-identity", "gpu-test")
        nsys_manifest = read_manifest(nsys_out)
        assert nsys_manifest["outcome"] == "success" and nsys_manifest["identity"]["mfem"] == "mfem-test"
        nsys_command = nsys_manifest["generating_command"]
        assert nsys_command[0] == str(nsys.resolve())
        assert nsys_command[nsys_command.index("--trace") + 1] == "cuda,nvtx"
        assert nsys_command[nsys_command.index("--sample") + 1] == "none"
        assert any(item["path"] == "nvtx.csv" for item in nsys_manifest["artifacts"])
        ncu = root / "fake-ncu"
        executable(ncu, "import pathlib\n(pathlib.Path.cwd() / 'summary.txt').write_text('summary')\n")
        ncu_out = root / "ncu"
        run("--level", "l2-ncu", *common(app, ncu_out), "--tool-executable", str(ncu),
            "--ncu-set", "basic", "--ncu-section", "Occupancy", "--ncu-kernel-name-base", "demangled",
            "--ncu-kernel-name", "kernel.*", "--ncu-launch-count", "3", "--ncu-launch-skip", "2",
            "--ncu-nvtx-include", "DG/*", "--require-data", "summary=summary.txt")
        ncu_command = read_manifest(ncu_out)["generating_command"]
        for option in ("--set", "--section", "--kernel-name-base", "--kernel-name", "--launch-count", "--launch-skip", "--nvtx", "--nvtx-include"):
            assert option in ncu_command
        dry = root / "dry"
        run("--level", "l2-nsys", *common(app, dry), "--tool-executable", str(root / "not-installed"),
            "--nsys-trace", "cuda", "--nsys-sample", "none", "--require-data", "table=table.csv", "--dry-run")
        assert read_manifest(dry)["outcome"] == "dry_run"
        missing_tool = root / "missing-tool"
        run("--level", "l2-nsys", *common(app, missing_tool), "--tool-executable", str(root / "no-nsys"),
            "--nsys-trace", "cuda", "--nsys-sample", "none", "--require-data", "table=table.csv", expected=3)
        assert read_manifest(missing_tool)["outcome"] == "tool_unavailable"
        missing_data = root / "missing-data"
        run("--level", "l2-ncu", *common(app, missing_data), "--tool-executable", str(ncu),
            "--ncu-set", "basic", "--require-data", "missing=missing.csv", expected=4)
        assert read_manifest(missing_data)["outcome"] == "data_unavailable"
        empty_ncu = root / "fake-empty-ncu"
        executable(empty_ncu, "import pathlib\n(pathlib.Path.cwd() / 'empty.csv').write_text('')\n")
        empty_data = root / "empty-data"
        run("--level", "l2-ncu", *common(app, empty_data), "--tool-executable", str(empty_ncu),
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
        assert "relative path" in run("--level", "l2-nsys", *common(app, rejected), "--tool-executable", str(nsys),
                                       "--nsys-trace", "cuda", "--nsys-sample", "none", "--require-data", "bad=../escape.csv", expected=2).stderr
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
    print("profile-harness-test: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
