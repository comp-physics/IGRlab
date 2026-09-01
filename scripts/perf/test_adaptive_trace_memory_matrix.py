#!/usr/bin/env python3
"""No-GPU parser and schedule tests for adaptive_trace_memory_matrix.py."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import os
import unittest
from unittest.mock import patch

from adaptive_trace_memory_matrix import (
    ENV_BE_STEPS,
    ENV_GPU_UUID,
    ENV_PBE_STEPS,
    ENV_RODAS_ATOL,
    ENV_RODAS_DT_MIN,
    ENV_RODAS_INITIAL_DT,
    ENV_RODAS_RTOL,
    ENV_VISIBLE_DEVICES,
    METHODS,
    MONOLITHIC_BE,
    PARTITIONED_BE,
    RODAS3,
    AdaptiveTraceMemoryError,
    MissingQualificationEnvironment,
    SKIP_RETURN_CODE,
    benchmark_command,
    latin_square_schedule,
    main,
    parse_architecture_selection_point,
    selected_controls_from_environment,
)


def valid_environment() -> dict[str, str]:
    return {
        ENV_BE_STEPS: "32",
        ENV_PBE_STEPS: "16",
        ENV_RODAS_ATOL: "2.5e-11",
        ENV_RODAS_RTOL: "2.5e-7",
        ENV_RODAS_INITIAL_DT: "0.08",
        ENV_RODAS_DT_MIN: "1e-6",
        ENV_GPU_UUID: "GPU-00112233-4455-6677-8899-aabbccddeeff",
        ENV_VISIBLE_DEVICES: "0",
    }


class AdaptiveTraceMemoryMatrixTests(unittest.TestCase):
    def test_schedule_is_exact_three_run_latin_square(self) -> None:
        schedule = latin_square_schedule()
        self.assertEqual(schedule, (
            (RODAS3, MONOLITHIC_BE, PARTITIONED_BE),
            (MONOLITHIC_BE, PARTITIONED_BE, RODAS3),
            (PARTITIONED_BE, RODAS3, MONOLITHIC_BE),
        ))
        self.assertEqual(len(schedule), 3)
        for row in schedule:
            self.assertEqual(set(row), set(METHODS))
        for column in range(3):
            self.assertEqual(
                {schedule[row][column] for row in range(3)}, set(METHODS)
            )
        flattened = [method for row in schedule for method in row]
        for method in METHODS:
            self.assertEqual(flattened.count(method), 3)

    def test_parser_requires_one_successful_exact_method_point(self) -> None:
        valid = (
            "unrelated bounded child output\n"
            "ARCHITECTURE_SELECTION_POINT method=rodas3 final_time=0.08 "
            "accepted_steps=5 rejected_attempts=1 "
            "internal_cuda_used_floor_bytes=123456 passed=1\n"
        )
        fields, floor = parse_architecture_selection_point(valid, RODAS3)
        self.assertEqual(fields["method"], RODAS3)
        self.assertEqual(floor, 123456)

        with self.assertRaisesRegex(
                AdaptiveTraceMemoryError, "expected exactly one"):
            parse_architecture_selection_point("no point\n", RODAS3)
        with self.assertRaisesRegex(
                AdaptiveTraceMemoryError, "observed 2"):
            parse_architecture_selection_point(valid + valid, RODAS3)
        with self.assertRaisesRegex(AdaptiveTraceMemoryError, "expected"):
            parse_architecture_selection_point(valid, MONOLITHIC_BE)
        with self.assertRaisesRegex(AdaptiveTraceMemoryError, "did not pass"):
            parse_architecture_selection_point(
                valid.replace("passed=1", "passed=0"), RODAS3
            )
        with self.assertRaisesRegex(AdaptiveTraceMemoryError, "positive"):
            parse_architecture_selection_point(
                valid.replace("123456", "0"), RODAS3
            )

    def test_required_controls_bind_method_specific_commands(self) -> None:
        environment = valid_environment()
        controls = selected_controls_from_environment(environment)
        rodas = benchmark_command("/tmp/convergence", RODAS3, controls)
        be = benchmark_command("/tmp/convergence", MONOLITHIC_BE, controls)
        pbe = benchmark_command("/tmp/convergence", PARTITIONED_BE, controls)
        self.assertIn("--adaptive-rodas3-screen", rodas)
        self.assertNotIn("--screen-memory-steps", rodas)
        self.assertEqual(be[-2:], ["--screen-memory-steps", "32"])
        self.assertEqual(pbe[-2:], ["--screen-memory-steps", "16"])
        for method, command in zip(METHODS, (rodas, be, pbe)):
            index = command.index("--screen-memory-method")
            self.assertEqual(command[index + 1], method)

        del environment[ENV_GPU_UUID]
        with self.assertRaises(MissingQualificationEnvironment):
            selected_controls_from_environment(environment)

    def test_missing_final_selection_environment_returns_skip_77(self) -> None:
        stdout = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(stdout):
            status = main([
                "--executable", "/not/resolved/on-skip",
                "--nvidia-smi", "/not/resolved/on-skip",
            ])
        self.assertEqual(status, SKIP_RETURN_CODE)
        self.assertIn("skipped=1", stdout.getvalue())
        self.assertIn(ENV_BE_STEPS, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
