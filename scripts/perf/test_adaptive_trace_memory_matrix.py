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
    FRESH_PROCESS_TIMING_MODE,
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
    print_fresh_process_timing_aggregates,
    run_fresh_process_timing_matrix,
    selected_controls_from_environment,
    validate_selected_point,
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


def valid_point(method: str, controls: dict[str, object], *,
                output: bool, internal_wall: float = 1.0) -> str:
    if method == RODAS3:
        steps = 5
        accepted = 5
        rejected = 1
        revision = 6
        adaptive = 1
        atol = float(controls["rodas_atol"])
        rtol = float(controls["rodas_rtol"])
        initial_dt = float(controls["rodas_initial_dt"])
        minimum_dt = float(controls["rodas_dt_min"])
        dt = initial_dt
        prepared = 1
    else:
        key = "be_steps" if method == MONOLITHIC_BE else "pbe_steps"
        steps = int(controls[key])
        accepted = steps
        rejected = 0
        revision = 0
        adaptive = 0
        atol = 0.0
        rtol = 0.0
        initial_dt = 0.0
        minimum_dt = 0.0
        dt = 0.08 / steps
        prepared = 0
    fields = {
        "method": method,
        "regularization": "igr",
        "order": "3",
        "elements_per_direction": "12",
        "final_time": format(0.08, ".17g"),
        "steps": str(steps),
        "dt": format(dt, ".17g"),
        "adaptive": str(adaptive),
        "time_atol": format(atol, ".17g"),
        "time_rtol": format(rtol, ".17g"),
        "adaptive_initial_dt": format(initial_dt, ".17g"),
        "adaptive_minimum_dt": format(minimum_dt, ".17g"),
        "controller_revision": str(revision),
        "timing_repetitions": "1",
        "measured_session_wall_seconds": format(
            0.8 * internal_wall, ".17g"
        ),
        "common_setup_inclusive_wall_seconds": format(
            internal_wall, ".17g"
        ),
        "measured_u4_global_l2_error": "0.001",
        "temporal_reference_error": "0",
        "sigma_coefficient_rms": "0.01",
        "accepted_steps": str(accepted),
        "rejected_attempts": str(rejected),
        "nonlinear_updates": "5",
        "linear_iterations": "42",
        "linear_solve_attempts": "6",
        "trace_solves": "6",
        "scalar_closure_solves": "0",
        "scalar_closure_linear_iterations": "0",
        "true_residual_tightenings": "1",
        "complete_assemblies": "7",
        "complete_condensations": "6",
        "complete_recoveries": "6",
        "common_shift_operator_builds": "2",
        "common_shift_operator_reuses": "4",
        "condensed_rhs_refreshes": "4",
        "common_shift_trace_solves": "6",
        "common_shift_preconditioner_setups": "2",
        "common_shift_preconditioner_reuses": "4",
        "native_trace_solve_calls": "6",
        "hypre_trace_solve_calls": "0",
        "patch_setups": "2",
        "preconditioner_applications": "42",
        "matrix_free_operator_applications": "48",
        "maximum_final_residual": "1e-12",
        "maximum_true_relative_residual": "1e-10",
        "maximum_conservation_error": "1e-13",
        "maximum_conservation_relative_error": "1e-13",
        "maximum_embedded_temporal_error_norm": "2.0",
        "maximum_final_algebraic_residual": "1e-12",
        "maximum_final_algebraic_correction_norm": "0.25",
        "minimum_density": "0.9",
        "minimum_pressure": "0.8",
        "maximum_wave_speed": "2.0",
        "internal_cuda_used_floor_bytes": "123456",
        "output_publications": "3" if output else "0",
        "midpoint_output_landed": "1" if output else "0",
        "final_output_landed": "1" if output else "0",
        "prepared_assessment_authority": str(prepared),
        "finite_gate": "1",
        "admissibility_gate": "1",
        "recovery_gate": "1",
        "true_residual_gate": "1",
        "conservation_gate": "1",
        "no_fallback_gate": "1",
        "trace_policy_gate": "1",
        "common_shift_gate": "1",
        "passed": "1",
    }
    return "ARCHITECTURE_SELECTION_POINT " + " ".join(
        f"{name}={value}" for name, value in fields.items()
    ) + "\n"


def changed_point(point: str, replacements: dict[str, str]) -> str:
    tokens = point.strip().split()
    changed = [tokens[0]]
    for token in tokens[1:]:
        name, value = token.split("=", 1)
        changed.append(f"{name}={replacements.get(name, value)}")
    return " ".join(changed) + "\n"


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
        output_command = benchmark_command(
            "/tmp/convergence", RODAS3, controls, "/tmp/output"
        )
        self.assertEqual(
            output_command[-2:], ["--screen-output-root", "/tmp/output"]
        )

        del environment[ENV_GPU_UUID]
        with self.assertRaises(MissingQualificationEnvironment):
            selected_controls_from_environment(environment)

    def test_selected_point_binds_controls_gates_and_output(self) -> None:
        controls = selected_controls_from_environment(valid_environment())
        for method in METHODS:
            fields, _ = parse_architecture_selection_point(
                valid_point(method, controls, output=True), method
            )
            self.assertGreater(
                validate_selected_point(
                    fields, method, controls, expect_output=True
                ),
                0.0,
            )

        rodas_fields, _ = parse_architecture_selection_point(
            valid_point(RODAS3, controls, output=True), RODAS3
        )
        be_fields, _ = parse_architecture_selection_point(
            valid_point(MONOLITHIC_BE, controls, output=True), MONOLITHIC_BE
        )
        mutations = (
            (rodas_fields, "time_atol", "1e-4", RODAS3, "selected control"),
            (rodas_fields, "conservation_gate", "0", RODAS3,
             "conservation_gate"),
            (rodas_fields, "output_publications", "2", RODAS3,
             "publication count"),
            (rodas_fields, "midpoint_output_landed", "0", RODAS3,
             "midpoint_output_landed"),
            (be_fields, "steps", "31", MONOLITHIC_BE,
             "selected step control"),
        )
        for original, name, replacement, method, message in mutations:
            with self.subTest(field=name, method=method):
                changed = dict(original)
                changed[name] = replacement
                with self.assertRaisesRegex(
                        AdaptiveTraceMemoryError, message):
                    validate_selected_point(
                        changed, method, controls, expect_output=True
                    )

    def test_fresh_process_matrix_uses_latin_square_and_compact_output(
            self) -> None:
        environment = valid_environment()
        controls = selected_controls_from_environment(environment)
        process_walls = {
            RODAS3: (1.0, 1.2, 1.1),
            MONOLITHIC_BE: (4.0, 4.2, 4.1),
            PARTITIONED_BE: (5.0, 5.2, 5.1),
        }
        internal_walls = {
            RODAS3: 0.8,
            MONOLITHIC_BE: 3.5,
            PARTITIONED_BE: 4.5,
        }
        counts = {method: 0 for method in METHODS}
        launches: list[str] = []

        def fake_run(command, **_kwargs):
            method_index = command.index("--screen-memory-method") + 1
            method = command[method_index]
            repetition = counts[method]
            counts[method] += 1
            launches.append(method)
            self.assertIn("--screen-output-root", command)
            return {
                "returncode": 0,
                "stdout": valid_point(
                    method, controls, output=True,
                    internal_wall=internal_walls[method],
                ),
                "stderr": "",
                "wall_seconds": process_walls[method][repetition],
            }

        with patch(
                "adaptive_trace_memory_matrix._run_fresh_process",
                side_effect=fake_run):
            aggregates = run_fresh_process_timing_matrix(
                "/tmp/convergence", controls, environment
            )
        self.assertEqual(
            launches,
            [method for rotation in latin_square_schedule()
             for method in rotation],
        )
        self.assertEqual(
            aggregates[RODAS3]["process_wall_seconds"]["median"], 1.1
        )
        rodas_work = aggregates[RODAS3]["work_signature"]
        self.assertEqual(rodas_work["steps"], 5)
        self.assertEqual(rodas_work["accepted_steps"], 5)
        self.assertEqual(rodas_work["rejected_attempts"], 1)
        self.assertEqual(rodas_work["controller_revision"], 6)
        self.assertEqual(rodas_work["trace_solves"], 6)
        self.assertEqual(rodas_work["linear_iterations"], 42)
        self.assertAlmostEqual(
            aggregates[MONOLITHIC_BE]["process_wall_seconds"]["range"], 0.2
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            print_fresh_process_timing_aggregates(aggregates, controls)
        emitted = stdout.getvalue()
        self.assertEqual(
            emitted.count("ADAPTIVE_TRACE_FRESH_PROCESS_CONTROLS"), 1
        )
        self.assertEqual(
            emitted.count("ADAPTIVE_TRACE_FRESH_PROCESS_WALL"), 3
        )
        self.assertIn("regularization=igr", emitted)
        self.assertIn("order=3", emitted)
        self.assertIn("elements_per_direction=12", emitted)
        self.assertIn("monolithic_steps=32", emitted)
        self.assertIn("partitioned_steps=16", emitted)
        self.assertIn("rodas3_time_atol=2.5000000000000001e-11", emitted)
        self.assertIn("accepted_steps=5 rejected_attempts=1", emitted)
        self.assertIn("trace_solves=6", emitted)
        self.assertIn("common_shift_operator_reuses=4", emitted)
        self.assertIn("maximum_final_residual_max=9.9999999999999998e-13",
                      emitted)
        self.assertIn("minimum_density_min=0.90000000000000002", emitted)
        self.assertIn("requested_gpu_uuid=GPU-00112233", emitted)
        self.assertIn("process_startup_included=1", emitted)
        self.assertIn("output_included=1", emitted)
        self.assertIn("nvidia_smi_polling=0", emitted)
        self.assertIn("separated_beyond_observed_ranges=1", emitted)

        aggregates[RODAS3]["process_wall_seconds"]["maximum"] = 5.0
        with self.assertRaisesRegex(
                AdaptiveTraceMemoryError, "not separated beyond"):
            with redirect_stdout(io.StringIO()):
                print_fresh_process_timing_aggregates(aggregates, controls)

    def test_fresh_process_matrix_rejects_bad_process_evidence(self) -> None:
        environment = valid_environment()
        controls = selected_controls_from_environment(environment)
        valid_stdout = valid_point(
            RODAS3, controls, output=True, internal_wall=2.0
        )
        bad_evidence = (
            ({"returncode": 7, "stdout": valid_stdout,
              "stderr": "bounded child failure", "wall_seconds": 3.0},
             "exited nonzero"),
            ({"returncode": 0, "stdout": valid_stdout,
              "stderr": "", "wall_seconds": float("nan")},
             "finite and positive"),
            ({"returncode": 0, "stdout": valid_stdout,
              "stderr": "", "wall_seconds": 1.0},
             "below its internal wall"),
            ({"returncode": 0, "stdout": valid_stdout,
              "stderr": "unexpected diagnostic", "wall_seconds": 3.0},
             "non-empty stderr"),
        )
        for evidence, message in bad_evidence:
            with self.subTest(message=message), patch(
                    "adaptive_trace_memory_matrix._run_fresh_process",
                    return_value=evidence):
                with self.assertRaisesRegex(
                        AdaptiveTraceMemoryError, message):
                    run_fresh_process_timing_matrix(
                        "/tmp/convergence", controls, environment
                    )

        with patch(
                "adaptive_trace_memory_matrix._run_fresh_process",
                side_effect=AdaptiveTraceMemoryError("child timeout")):
            with self.assertRaisesRegex(
                    AdaptiveTraceMemoryError, "child timeout"):
                run_fresh_process_timing_matrix(
                    "/tmp/convergence", controls, environment
                )

    def test_fresh_process_matrix_requires_repeatable_work_and_u4(self) -> None:
        environment = valid_environment()
        controls = selected_controls_from_environment(environment)
        mutations = (
            ({"steps": "6", "accepted_steps": "6",
              "controller_revision": "7"}, "work signature changed"),
            ({"trace_solves": "7"}, "work signature changed"),
            ({"measured_u4_global_l2_error": "0.002"},
             "U4 metric changed"),
        )
        for replacements, message in mutations:
            counts = {method: 0 for method in METHODS}

            def fake_run(command, **_kwargs):
                method_index = command.index("--screen-memory-method") + 1
                method = command[method_index]
                repetition = counts[method]
                counts[method] += 1
                point = valid_point(method, controls, output=True)
                if method == RODAS3 and repetition == 1:
                    point = changed_point(point, replacements)
                return {
                    "returncode": 0,
                    "stdout": point,
                    "stderr": "",
                    "wall_seconds": 2.0,
                }

            with self.subTest(message=message), patch(
                    "adaptive_trace_memory_matrix._run_fresh_process",
                    side_effect=fake_run):
                with self.assertRaisesRegex(
                        AdaptiveTraceMemoryError, message):
                    run_fresh_process_timing_matrix(
                        "/tmp/convergence", controls, environment
                    )

    def test_missing_final_selection_environment_returns_skip_77(self) -> None:
        for arguments in (
                ["--executable", "/not/resolved/on-skip",
                 "--nvidia-smi", "/not/resolved/on-skip"],
                ["--mode", FRESH_PROCESS_TIMING_MODE,
                 "--executable", "/not/resolved/on-skip"]):
            stdout = io.StringIO()
            with self.subTest(arguments=arguments), \
                    patch.dict(os.environ, {}, clear=True), \
                    redirect_stdout(stdout):
                status = main(arguments)
            self.assertEqual(status, SKIP_RETURN_CODE)
            self.assertIn("skipped=1", stdout.getvalue())
            self.assertIn(ENV_BE_STEPS, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
