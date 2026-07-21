#!/usr/bin/env python3
"""Focused non-capture checks for the profile provenance tools."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WRITER = ROOT / "scripts/perf/profile_provenance.py"
VALIDATOR = ROOT / "scripts/perf/validate_profile_provenance.py"


def run(*command: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ, "PROFILE_PROVENANCE_TEST": "set"})
    if result.returncode != expected:
        raise RuntimeError(f"expected {expected}, got {result.returncode}: {' '.join(command)}\n{result.stdout}\n{result.stderr}")
    return result


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        metadata = Path(temporary) / "sidecar.json"
        run(sys.executable, str(WRITER), "--output", str(metadata), "--tool", "nsys", "--tool-executable", "/opt/nsight/nsys", "--tool-version", "2026.1", "--output-stem", "artifacts/run", "--source-root", str(ROOT), "--environment", "PROFILE_PROVENANCE_TEST", "--label", "purpose=smoke", "--command", "--", "application", "--device", "cuda")
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        assert payload["tool"]["name"] == "nsys"
        assert payload["capture"]["command"] == ["application", "--device", "cuda"]
        assert payload["source"]["revision"]["status"] == "available"
        run(sys.executable, str(VALIDATOR), "--metadata", str(metadata), "--require", "tool.version", "--require", "source.tree", "--require", "environment.PROFILE_PROVENANCE_TEST")
        payload["tool"]["version"] = {"status": "unavailable", "reason": "not installed"}
        rejected = run(sys.executable, str(WRITER), "--output", str(metadata), "--tool", "nsys", "--environment", "API_TOKEN", "--command", "application", expected=2)
        assert "refuses sensitive key" in rejected.stderr

        metadata.write_text(json.dumps(payload), encoding="utf-8")
        run(sys.executable, str(VALIDATOR), "--metadata", str(metadata), "--require", "tool.version", expected=1)
    print("profile-provenance-test: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
