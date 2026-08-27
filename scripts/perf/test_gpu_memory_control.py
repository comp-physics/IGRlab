#!/usr/bin/env python3
"""Focused no-GPU process test for gpu_memory_control.py."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import unittest

from gpu_memory_control import (
    MIB,
    MemoryControlError,
    aggregate_three_repeat_controls,
    run_gpu_memory_control,
)


FAKE_NVIDIA_SMI = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import time

gpu_uuid = next(
    argument.split("=", 1)[1]
    for argument in sys.argv if argument.startswith("--id=")
)
poll_ms = int(next(
    argument.split("=", 1)[1]
    for argument in sys.argv if argument.startswith("--loop-ms=")
))
reported_uuid = os.environ.get("FAKE_REPORTED_UUID", gpu_uuid)
marker = Path(os.environ["FAKE_GPU_ACTIVE_MARKER"])
counter = 0
while True:
    if os.environ.get("FAKE_BASELINE_DRIFT") == "1" and not marker.exists():
        used = 100 + 80 * counter
    else:
        used = 300 if marker.exists() else 100
    total = 1000
    free = total - used
    print(
        f"{time.time_ns()},{reported_uuid},{total},{used},{free}",
        flush=True,
    )
    counter += 1
    time.sleep(max(0.001, poll_ms / 1000.0))
'''


FAKE_BENCHMARK = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import time

marker = Path(os.environ["FAKE_GPU_ACTIVE_MARKER"])
marker.write_text("active\n", encoding="utf-8")
try:
    sys.stderr.write("fake benchmark stderr\n")
    sys.stderr.flush()
    time.sleep(0.35)
    sys.stdout.write('{"schema":2,"success":true}\n')
    sys.stdout.flush()
finally:
    marker.unlink(missing_ok=True)
raise SystemExit(int(os.environ.get("FAKE_BENCHMARK_EXIT", "0")))
'''


def write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


class GpuMemoryControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="gpu-memory-control-test-"
        )
        self.root = Path(self.temporary.name)
        self.nvidia_smi = self.root / "nvidia-smi"
        self.benchmark = self.root / "benchmark"
        self.marker = self.root / "active.marker"
        write_executable(self.nvidia_smi, FAKE_NVIDIA_SMI)
        write_executable(self.benchmark, FAKE_BENCHMARK)
        self.environment = dict(os.environ)
        self.environment["FAKE_GPU_ACTIVE_MARKER"] = str(self.marker)

    def tearDown(self) -> None:
        self.marker.unlink(missing_ok=True)
        self.temporary.cleanup()

    def run_control(self, environment=None):
        return run_gpu_memory_control(
            [str(self.benchmark)],
            nvidia_smi=self.nvidia_smi,
            selected_gpu_uuid="GPU-00112233-4455-6677-8899-aabbccddeeff",
            poll_ms=20,
            timeout_seconds=5.0,
            environment=environment or self.environment,
        )

    def test_control_retains_exact_process_and_memory_evidence(self) -> None:
        evidence = self.run_control()
        self.assertEqual(evidence["status"], "success")
        self.assertEqual(
            evidence["benchmark"]["stdout"],
            '{"schema":2,"success":true}\n',
        )
        self.assertEqual(
            evidence["benchmark"]["stderr"],
            "fake benchmark stderr\n",
        )
        self.assertFalse(evidence["benchmark"]["timed_out"])
        self.assertEqual(evidence["benchmark"]["returncode"], 0)
        measurement = evidence["measurement"]
        self.assertEqual(len(measurement["pre_rows"]), 10)
        self.assertGreaterEqual(len(measurement["active_rows"]), 1)
        self.assertEqual(len(measurement["post_rows"]), 10)
        self.assertEqual(measurement["absolute_peak_mib"], 300.0)
        self.assertEqual(measurement["incremental_peak_mib"], 200.0)
        self.assertEqual(measurement["headroom_mib"], 700.0)
        self.assertLessEqual(
            measurement["observed_maximum_active_gap_ms"], 100.0
        )
        self.assertTrue(all(
            measurement["qualification_gates"].values()
        ))
        self.assertIn("--loop-ms=20", evidence["poller"]["command"])
        self.assertIn(
            "--id=GPU-00112233-4455-6677-8899-aabbccddeeff",
            evidence["poller"]["command"],
        )
        self.assertFalse(self.marker.exists())

    def test_uuid_mismatch_fails_with_raw_evidence(self) -> None:
        environment = dict(self.environment)
        environment["FAKE_REPORTED_UUID"] = (
            "GPU-ffeeddcc-bbaa-9988-7766-554433221100"
        )
        with self.assertRaises(MemoryControlError) as caught:
            self.run_control(environment)
        self.assertIsNotNone(caught.exception.evidence)
        self.assertGreaterEqual(
            len(caught.exception.evidence["poller"]["rows"]), 20
        )
        self.assertEqual(
            caught.exception.evidence["benchmark"]["stdout"],
            '{"schema":2,"success":true}\n',
        )

    def test_nonzero_benchmark_exit_retains_exact_streams(self) -> None:
        environment = dict(self.environment)
        environment["FAKE_BENCHMARK_EXIT"] = "7"
        with self.assertRaisesRegex(MemoryControlError, "status 7") as caught:
            self.run_control(environment)
        evidence = caught.exception.evidence
        self.assertEqual(evidence["benchmark"]["returncode"], 7)
        self.assertEqual(
            evidence["benchmark"]["stdout"],
            '{"schema":2,"success":true}\n',
        )
        self.assertEqual(
            evidence["benchmark"]["stderr"],
            "fake benchmark stderr\n",
        )

    def test_three_repeat_aggregation_and_gates(self) -> None:
        repeats = [self.run_control() for _ in range(3)]
        result = aggregate_three_repeat_controls(
            repeats, [299 * MIB, 300 * MIB, 301 * MIB]
        )
        self.assertTrue(result["qualified"])
        self.assertEqual(result["repeat_count"], 3)
        self.assertEqual(result["incremental_peak_mib"]["range"], 0.0)
        self.assertEqual(result["conservative_absolute_peak_mib"], 300.0)
        self.assertEqual(result["minimum_headroom_mib"], 700.0)
        self.assertTrue(all(
            item["passes"] for item in result["internal_cuda_floor_checks"]
        ))

        unstable = copy.deepcopy(repeats)
        unstable[2]["measurement"]["incremental_peak_mib"] = 300.0
        with self.assertRaisesRegex(MemoryControlError, "spread exceeds"):
            aggregate_three_repeat_controls(
                unstable, [299 * MIB, 300 * MIB, 301 * MIB]
            )

        with self.assertRaisesRegex(MemoryControlError, "below its internal"):
            aggregate_three_repeat_controls(
                repeats, [299 * MIB, 400 * MIB, 301 * MIB]
            )


if __name__ == "__main__":
    unittest.main()
