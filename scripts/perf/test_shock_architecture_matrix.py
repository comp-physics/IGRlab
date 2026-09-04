#!/usr/bin/env python3
"""Synthetic, no-device tests for shock_architecture_matrix.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


sys.dont_write_bytecode = True

from shock_architecture_matrix import (  # noqa: E402
    DG_RK3SSP,
    DG_SSPRK104,
    DGSEM_SSPRK104_EVERY_STAGE,
    DGSEM_SSPRK104_EXTRAPOLATED,
    EXACT_IID_Q_SIGMA_SCOPE,
    EXTRAPOLATED_Q_SIGMA_SCOPE,
    HDG_RODAS3_ORDINARY,
    HDG_RODAS3_TENSOR,
    METHODS,
    PROTECTED_APP_OPTIONS,
    SCHEMA_VERSION,
    HarnessControls,
    ShockArchitectureError,
    _argument_parser,
    aggregate_runs,
    build_method_arguments,
    campaign_child_environment,
    capture_file_identity,
    classify_execution,
    classify_selection,
    evaluate_quality,
    git_identity,
    method_closure_policy,
    parse_application_log,
    parse_key_value_arguments,
    parse_method_float_overrides,
    parse_methods,
    parse_qoi_json,
    parse_qoi_requirements,
    parse_time_record,
    pin_qoi_input_files,
    qoi_command,
    render_markdown,
    required_q_sigma_scope,
    rotated_schedule,
    run_campaign,
    run_timed_process,
    validate_extra_app_arguments,
    verify_file_identity,
    verify_git_identity,
)


def controls(*, device: str = "cuda", vtk: bool = False,
             zhang_shu_vtk_points: bool = False) -> HarnessControls:
    return HarnessControls(
        device=device,
        order=2,
        serial_refine=0,
        parallel_refine=1,
        final_time=0.2,
        c_alpha=1.0,
        alpha_length_policy="max",
        dgsem_surface_flux="rusanov",
        igr_invariant_domain_flux=False,
        igr_sigma_cfl=False,
        igr_sigma_rusanov=False,
        conservation_balance=False,
        zhang_shu_vtk_points=zhang_shu_vtk_points,
        closure_certificate_max_relative_residual=1.0e-7,
        closure_certificate_max_absolute_residual=0.0,
        explicit_dt_by_method={
            DG_RK3SSP: 1.0e-4,
            DG_SSPRK104: 3.0e-4,
            DGSEM_SSPRK104_EVERY_STAGE: 4.0e-4,
            DGSEM_SSPRK104_EXTRAPOLATED: 5.0e-4,
        },
        explicit_cfl_by_method={
            DG_RK3SSP: 0.5,
            DG_SSPRK104: 1.0,
            DGSEM_SSPRK104_EVERY_STAGE: 2.0,
            DGSEM_SSPRK104_EXTRAPOLATED: 3.0,
        },
        dt_min=1.0e-10,
        igr_linear_rtol=1.0e-5,
        igr_linear_atol=0.0,
        igr_linear_max_iterations=500,
        rodas_initial_dt=1.0e-3,
        rodas_time_atol=1.0e-10,
        rodas_time_rtol=1.0e-6,
        progress_steps=25,
        vtk=vtk,
        vtk_save_interval=0.2,
    )


def option_value(arguments: list[str], name: str) -> str:
    index = arguments.index(name)
    return arguments[index + 1]


def certified_closure() -> dict[str, object]:
    return {
        "reported": True,
        "scope": "final-committed-accepted-state",
        "precision": "fp64",
        "time": 0.2,
        "state_generation": 2,
        "solved": True,
        "supported": True,
        "certified": True,
        "run_success": True,
        "residual_rel_rhs": 2.0e-8,
        "residual_abs": 1.0e-9,
        "residual_reported": True,
        "residual_finite": True,
        "relative_limit": 5.0e-5,
        "absolute_limit": 0.0,
        "skipped_nonfinite": False,
    }


def certified_extrapolated_lifecycle(
        *, endpoint_solves: int = 34,
        absolute_only_passes: int = 0) -> dict[str, object]:
    return {
        "reported": True,
        "endpoint_solves": endpoint_solves,
        "exact_anchor_certificates": endpoint_solves + 2,
        "exact_anchor_absolute_only_passes": absolute_only_passes,
        "max_exact_anchor_certificate_acceptance_ratio": 0.5,
        "max_exact_anchor_relative_pass_residual_rel_rhs": 2.5e-8,
        "max_exact_anchor_absolute_only_residual_abs": 0.0,
    }


def certified_extrapolated_scalar_closure() -> dict[str, object]:
    return {
        "reported": True,
        "last_solver": "cg",
        "last_precond": "operatorchebyshev",
        "exact_face_effective": True,
    }


def positive_q_sigma_telemetry(
        scope: str = EXACT_IID_Q_SIGMA_SCOPE) -> dict[str, object]:
    return {
        "reported": True,
        "scope": scope,
        "precision": "fp64",
        "rejected_attempts_included": True,
        "sample_count": 100,
        "nonpositive_count": 0,
        "nonfinite_count": 0,
        "minimum_raw_q_sigma": 0.125,
        "minimum_q_sigma_over_gamma_p": 0.05,
    }


def q_sigma_rejection_summary(
        *, attempts: int = 0, nonpositive: int = 0, nonfinite: int = 0,
        rejected_attempts: int | None = None,
        reason: str | None = None, stage: int | None = None
        ) -> dict[str, object]:
    if rejected_attempts is None:
        rejected_attempts = attempts
    if reason is None:
        if attempts == 0:
            reason = "none"
        elif nonpositive > 0 and nonfinite > 0:
            reason = "q-sigma-nonpositive-and-nonfinite"
        elif nonfinite > 0:
            reason = "q-sigma-nonfinite"
        else:
            reason = "q-sigma-nonpositive"
    if stage is None:
        stage = -1 if attempts == 0 else 6
    return {
        "reported": True,
        "success": True,
        "rejected_attempts": rejected_attempts,
        "rejected_q_sigma_attempts": attempts,
        "rejected_q_sigma_nonpositive_samples": nonpositive,
        "rejected_q_sigma_nonfinite_samples": nonfinite,
        "last_q_sigma_rejection_reason": reason,
        "last_q_sigma_rejection_stage": stage,
    }


def sample_run(
        method: str, kind: str, wall: float, *, time_reached: float | None,
        peak_rss_kib: int | None = 1000,
        qoi_value: float | None = None,
        quality_status: str | None = None) -> dict[str, object]:
    if kind == "success":
        execution = {
            "kind": "success",
            "reason": "target_horizon_reached",
            "time_to_solution_seconds": wall,
            "time_to_failure_seconds": None,
            "censoring_wall_seconds": None,
        }
    elif kind == "failure":
        execution = {
            "kind": "failure",
            "reason": "synthetic_failure",
            "time_to_solution_seconds": None,
            "time_to_failure_seconds": wall,
            "censoring_wall_seconds": None,
        }
    else:
        execution = {
            "kind": "censored",
            "reason": "wall_timeout",
            "time_to_solution_seconds": None,
            "time_to_failure_seconds": None,
            "censoring_wall_seconds": wall,
        }
    if quality_status is None:
        quality_status = "measured_not_gated" \
            if kind == "success" and qoi_value is not None else "not_evaluated"
    quality = {
        "mode": "measurement-only",
        "status": quality_status,
        "qualified": quality_status == "passed",
        "reason": None,
    }
    if kind == "failure":
        selection_kind = "execution_failure"
    elif kind == "censored":
        selection_kind = "execution_censored"
    elif quality_status == "passed":
        selection_kind = "qualified_success"
    else:
        selection_kind = "execution_success_unqualified"
    return {
        "method": method,
        "execution": execution,
        "quality": quality,
        "selection": {
            "kind": selection_kind,
            "time_to_qualified_solution_seconds":
                wall if selection_kind == "qualified_success" else None,
        },
        "progress": {"time_reached": time_reached},
        "process": {
            "application_wall_seconds": wall,
            "end_to_end_wall_seconds": wall,
            "peak_rss_kib": peak_rss_kib,
        },
        "qoi": {
            "status": "success" if qoi_value is not None else "not_requested",
            "values": {
                "shock_position": qoi_value,
                "shock_standoff_distance": None,
                "shock_profile_l1_error": None,
                "shock_profile_l2_error": None,
                "shock_thickness": None,
                "density_minimum": None,
                "density_maximum": None,
                "pressure_minimum": None,
                "pressure_maximum": None,
                "conservation_error": None,
                "boundary_balance_error": None,
                "sigma_localization": None,
            },
        },
    }


class ScheduleAndCardTests(unittest.TestCase):
    def test_cyclic_schedule_preserves_caller_order(self) -> None:
        methods = (DG_SSPRK104, HDG_RODAS3_TENSOR, DG_RK3SSP)
        self.assertEqual(
            rotated_schedule(methods, 3, rotation_offset=1),
            (
                (HDG_RODAS3_TENSOR, DG_RK3SSP, DG_SSPRK104),
                (DG_RK3SSP, DG_SSPRK104, HDG_RODAS3_TENSOR),
                (DG_SSPRK104, HDG_RODAS3_TENSOR, DG_RK3SSP),
            ),
        )
        self.assertEqual(parse_methods(",".join(methods)), methods)

    def test_method_list_rejects_unknown_duplicate_and_empty(self) -> None:
        for value, message in (
            ("", "at least one"),
            ("dg-rk3ssp,dg-rk3ssp", "duplicates"),
            ("fictional", "unknown methods"),
        ):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ShockArchitectureError, message):
                parse_methods(value)
        with self.assertRaisesRegex(ShockArchitectureError, "repetitions"):
            rotated_schedule((DG_RK3SSP,), 6)

    def test_explicit_cards_are_typed_strict_non_h1_every_stage(self) -> None:
        expected_integrators = {
            DG_RK3SSP: "rk3ssp",
            DG_SSPRK104: "ssprk104",
        }
        for method, integrator in expected_integrators.items():
            with self.subTest(method=method):
                arguments = build_method_arguments(
                    method, "/mesh/cylinder.msh", controls(),
                    ("--velocity-x", "3.0"),
                )
                self.assertEqual(option_value(arguments, "--discretization"), "dg")
                self.assertEqual(option_value(arguments, "--dg-backend"), "dg")
                self.assertEqual(
                    option_value(arguments, "--time-integrator"), integrator
                )
                self.assertEqual(option_value(arguments, "--regularization"), "igr")
                self.assertEqual(option_value(arguments, "--igr-space"), "L2")
                self.assertEqual(
                    option_value(arguments, "--igr-grad"), "BrokenL2"
                )
                self.assertEqual(
                    option_value(arguments, "--igr-solve-mode"), "krylov"
                )
                self.assertEqual(option_value(arguments, "--igr-ip-kind"), "sipg")
                self.assertEqual(
                    option_value(arguments, "--igr-sigma-cadence"),
                    "every-stage",
                )
                self.assertEqual(
                    option_value(arguments, "--igr-nonconvergence-policy"),
                    "error",
                )
                self.assertEqual(
                    option_value(arguments, "--igr-linear-solver"), "minres"
                )
                self.assertEqual(
                    option_value(arguments, "--igr-preconditioner"),
                    "chebyshev",
                )
                self.assertIn("--igr-exact-dg-face", arguments)
                self.assertEqual(
                    option_value(arguments, "--dgsem-surface-flux"),
                    "rusanov",
                )
                self.assertIn("--no-igr-sigma-cfl", arguments)
                self.assertIn("--no-igr-sigma-rusanov", arguments)
                self.assertIn("--no-igr-invariant-domain-flux", arguments)
                self.assertIn("--no-conservation-balance", arguments)
                self.assertIn("--nan-scan", arguments)
                self.assertIn("--admissibility-scan", arguments)
                self.assertIn("--no-vtk_save", arguments)
                self.assertIn("--no-zhang-shu-vtk-points", arguments)
                self.assertNotIn("--zhang-shu-vtk-points", arguments)
                self.assertEqual(
                    float(option_value(arguments, "--time-step")),
                    controls().explicit_dt_by_method[method],
                )
                self.assertEqual(
                    float(option_value(arguments, "--cfl-number")),
                    controls().explicit_cfl_by_method[method],
                )

    def test_selected_dgsem_cards_pin_exact_transport_and_closure_tuple(
            self) -> None:
        local = controls()
        local = HarnessControls(**{
            **local.__dict__,
            "alpha_length_policy": "max",
            "dgsem_surface_flux": "es-winters",
            "igr_invariant_domain_flux": True,
            "igr_sigma_cfl": True,
            "igr_sigma_rusanov": True,
        })
        expected_cadences = {
            DGSEM_SSPRK104_EVERY_STAGE: "every-stage",
            DGSEM_SSPRK104_EXTRAPOLATED: "extrapolated-every-step",
        }
        for method, cadence in expected_cadences.items():
            with self.subTest(method=method):
                arguments = build_method_arguments(
                    method, "/mesh/cylinder.msh", local
                )
                expected_options = {
                    "--discretization": "dgsem",
                    "--dg-backend": "dgsem",
                    "--time-integrator": "ssprk104",
                    "--igr-alpha-length": "local",
                    "--igr-space": "L2",
                    "--igr-grad": "BrokenL2",
                    "--igr-solve-mode": "krylov",
                    "--igr-ip-kind": "sipg",
                    "--igr-linear-solver": "cg",
                    "--igr-preconditioner": "chebyshev",
                    "--igr-sigma-cadence": cadence,
                    "--igr-stage-rtol-scale": "1",
                    "--dgsem-stab": "none",
                    "--dgsem-volume-flux": "central",
                    "--dgsem-volume-dissipation": "llf",
                    "--dgsem-surface-flux": "rusanov",
                    "--dgsem-entropy-jacobian": "fd",
                }
                for option, expected in expected_options.items():
                    self.assertEqual(option_value(arguments, option), expected)
                self.assertIn("--igr-exact-dg-face", arguments)
                self.assertIn("--no-igr-invariant-domain-flux", arguments)
                self.assertIn("--no-igr-sigma-cfl", arguments)
                self.assertIn("--no-igr-sigma-rusanov", arguments)
                self.assertIn("--no-vtk-h1-diagnostics", arguments)
                self.assertIn("--no-vtk-l2-diagnostics", arguments)
                self.assertIn("--no-vtk-mpi-rank-field", arguments)
                self.assertIn("--no-vtk-zhang-shu-diagnostics", arguments)
                self.assertEqual(
                    float(option_value(arguments, "--time-step")),
                    local.explicit_dt_by_method[method],
                )
                self.assertEqual(
                    float(option_value(arguments, "--cfl-number")),
                    local.explicit_cfl_by_method[method],
                )

    def test_q_sigma_contract_is_method_specific(self) -> None:
        self.assertIsNone(required_q_sigma_scope(
            DG_RK3SSP, igr_invariant_domain_flux=False
        ))
        self.assertEqual(
            required_q_sigma_scope(
                DG_SSPRK104, igr_invariant_domain_flux=True
            ),
            EXACT_IID_Q_SIGMA_SCOPE,
        )
        self.assertIsNone(required_q_sigma_scope(
            DGSEM_SSPRK104_EVERY_STAGE,
            igr_invariant_domain_flux=True,
        ))
        self.assertEqual(
            required_q_sigma_scope(
                DGSEM_SSPRK104_EXTRAPOLATED,
                igr_invariant_domain_flux=False,
            ),
            EXTRAPOLATED_Q_SIGMA_SCOPE,
        )
        self.assertIsNone(required_q_sigma_scope(
            HDG_RODAS3_ORDINARY, igr_invariant_domain_flux=True
        ))

    def test_zhang_shu_vtk_point_control_is_default_off_and_explicit_only(
            self) -> None:
        base_cli = [
            "--executable", "/app", "--mesh", "/mesh",
            "--output-dir", "/output", "--methods", DG_RK3SSP,
            "--final-time", "0.2",
        ]
        self.assertFalse(
            _argument_parser().parse_args(base_cli).zhang_shu_vtk_points
        )
        self.assertTrue(_argument_parser().parse_args([
            *base_cli, "--zhang-shu-vtk-points",
        ]).zhang_shu_vtk_points)

        explicit = build_method_arguments(
            DG_RK3SSP, "/mesh/cylinder.msh",
            controls(zhang_shu_vtk_points=True),
        )
        self.assertIn("--zhang-shu-vtk-points", explicit)
        self.assertNotIn("--no-zhang-shu-vtk-points", explicit)

        hdg = build_method_arguments(
            HDG_RODAS3_ORDINARY, "/mesh/cylinder.msh",
            controls(zhang_shu_vtk_points=True),
        )
        self.assertNotIn("--zhang-shu-vtk-points", hdg)
        self.assertNotIn("--no-zhang-shu-vtk-points", hdg)

        aliases = {
            "-zsvtk", "--zhang-shu-vtk-points",
            "-no-zsvtk", "--no-zhang-shu-vtk-points",
        }
        self.assertTrue(aliases <= PROTECTED_APP_OPTIONS)
        for alias in aliases:
            with self.subTest(alias=alias), self.assertRaisesRegex(
                    ShockArchitectureError, "may not override"):
                validate_extra_app_arguments((alias,))

    def test_explicit_card_can_type_sigma_aware_cfl_and_dissipation(self) -> None:
        local = controls()
        local = HarnessControls(**{
            **local.__dict__,
            "igr_sigma_cfl": True,
            "igr_sigma_rusanov": True,
        })
        arguments = build_method_arguments(
            DG_SSPRK104, "/mesh/cylinder.msh", local
        )
        self.assertIn("--igr-sigma-cfl", arguments)
        self.assertIn("--igr-sigma-rusanov", arguments)
        self.assertNotIn("--no-igr-sigma-cfl", arguments)
        self.assertNotIn("--no-igr-sigma-rusanov", arguments)

    def test_explicit_card_can_type_coupled_invariant_domain_flux(self) -> None:
        local = controls()
        local = HarnessControls(**{
            **local.__dict__,
            "igr_invariant_domain_flux": True,
        })
        arguments = build_method_arguments(
            DG_SSPRK104, "/mesh/cylinder.msh", local
        )
        self.assertIn("--igr-invariant-domain-flux", arguments)
        self.assertIn("--no-igr-sigma-cfl", arguments)
        self.assertIn("--no-igr-sigma-rusanov", arguments)

    def test_explicit_card_can_enable_conservation_balance(self) -> None:
        local = controls()
        local = HarnessControls(**{
            **local.__dict__,
            "conservation_balance": True,
        })
        arguments = build_method_arguments(
            DG_SSPRK104, "/mesh/cylinder.msh", local
        )
        self.assertIn("--conservation-balance", arguments)
        self.assertNotIn("--no-conservation-balance", arguments)

    def test_hdg_cards_select_only_the_engine(self) -> None:
        ordinary = build_method_arguments(
            HDG_RODAS3_ORDINARY, "/mesh/cylinder.msh", controls(vtk=True)
        )
        tensor = build_method_arguments(
            HDG_RODAS3_TENSOR, "/mesh/cylinder.msh", controls(vtk=True)
        )
        for arguments, engine in ((ordinary, "ordinary"), (tensor, "tensor")):
            self.assertEqual(option_value(arguments, "--discretization"), "hdg")
            self.assertEqual(option_value(arguments, "--time-integrator"), "rodas3")
            self.assertEqual(option_value(arguments, "--hdg-cuda-engine"), engine)
            self.assertEqual(option_value(arguments, "--cfl-number"), "0")
            self.assertTrue(float(option_value(arguments, "--time-step")) < 0.0)
            self.assertIn("--vtk_save", arguments)
        with self.assertRaisesRegex(ShockArchitectureError, "requires --device"):
            build_method_arguments(
                HDG_RODAS3_TENSOR, "/mesh/cylinder.msh", controls(device="cpu")
            )
        local = controls()
        local = HarnessControls(
            **{**local.__dict__, "alpha_length_policy": "local"}
        )
        with self.assertRaisesRegex(ShockArchitectureError, "local.*alpha"):
            build_method_arguments(
                HDG_RODAS3_ORDINARY, "/mesh/cylinder.msh", local
            )

    def test_closure_policy_does_not_apply_dg_claims_to_hdg(self) -> None:
        explicit = method_closure_policy(DG_RK3SSP)
        hdg = method_closure_policy(HDG_RODAS3_TENSOR)
        self.assertEqual(explicit["scalar_space"], "L2")
        self.assertEqual(explicit["reconstruction_cadence"], "every-stage")
        self.assertEqual(hdg["architecture"], "coupled-traced-hdg-rodas3")
        self.assertNotEqual(hdg["scalar_space"], "L2")
        self.assertIn("null", hdg["application_metrics"])

    def test_extra_arguments_cannot_override_typed_controls(self) -> None:
        required_restart_aliases = {
            "-restart-vtk", "--restart-vtk", "-restart-cycle",
            "--restart-cycle", "-restart-step", "--restart-step",
            "-hdg-checkpoint", "--hdg-checkpoint",
            "-hdg-restart-checkpoint", "--hdg-restart-checkpoint",
        }
        self.assertTrue(required_restart_aliases <= PROTECTED_APP_OPTIONS)
        for option in sorted(PROTECTED_APP_OPTIONS):
            with self.subTest(option=option), self.assertRaisesRegex(
                    ShockArchitectureError, "may not override"):
                validate_extra_app_arguments((option,))
            with self.subTest(option=f"{option}=value"), self.assertRaisesRegex(
                    ShockArchitectureError, "may not override"):
                validate_extra_app_arguments((f"{option}=value",))

    def test_key_value_metadata_is_strict(self) -> None:
        self.assertEqual(
            parse_key_value_arguments(
                ("gpu_uuid=GPU-123", "driver=999.1"), "hardware"
            ),
            {"gpu_uuid": "GPU-123", "driver": "999.1"},
        )
        for values, message in (
            (("missing",), "KEY=VALUE"),
            (("bad key=value",), "invalid key"),
            (("gpu=",), "empty"),
            (("gpu=one", "gpu=two"), "repeats"),
        ):
            with self.subTest(values=values), self.assertRaisesRegex(
                    ShockArchitectureError, message):
                parse_key_value_arguments(values, "hardware")

    def test_per_method_explicit_overrides_are_strict(self) -> None:
        values = parse_method_float_overrides(
            ("dg-rk3ssp=0.4",),
            option_name="--method-explicit-cfl", default=0.5,
            allow_zero=True,
        )
        self.assertEqual(values[DG_RK3SSP], 0.4)
        self.assertEqual(values[DG_SSPRK104], 0.5)
        self.assertEqual(values[DGSEM_SSPRK104_EVERY_STAGE], 0.5)
        self.assertEqual(values[DGSEM_SSPRK104_EXTRAPOLATED], 0.5)
        fixed = parse_method_float_overrides(
            ("dg-ssprk104=0",), option_name="--method-explicit-cfl",
            default=0.5, allow_zero=True,
        )
        self.assertEqual(fixed[DG_SSPRK104], 0.0)
        for override, message in (
            ("hdg-rodas3-tensor=1", "must be one of"),
            ("dg-rk3ssp=bad", "numeric"),
            ("dg-rk3ssp=-1", "nonnegative"),
        ):
            with self.subTest(override=override), self.assertRaisesRegex(
                    ShockArchitectureError, message):
                parse_method_float_overrides(
                    (override,), option_name="--method-explicit-cfl",
                    default=0.5, allow_zero=True,
                )


class SyntheticLogParserTests(unittest.TestCase):
    def test_campaign_environment_pins_full_chebyshev_operator(self) -> None:
        source = {
            "PATH": "/closed/path",
            "DG_IGR_VTK_PREFIX_PATH": "/stale/vtk",
            "DG_IGR_LIMITER_TIMING": "stale",
            "DG_IGR_IGR_CHEB_BASE_OPERATOR": "1",
        }
        environment = campaign_child_environment(
            source, enable_limiter_timing=False
        )
        self.assertEqual(environment["PATH"], "/closed/path")
        self.assertEqual(
            environment["DG_IGR_IGR_CHEB_BASE_OPERATOR"], "0"
        )
        self.assertNotIn("DG_IGR_VTK_PREFIX_PATH", environment)
        self.assertNotIn("DG_IGR_LIMITER_TIMING", environment)

    def test_explicit_success_parses_retry_closure_limiter_and_work(self) -> None:
        log = """
[alive] step     20 | t     0.125000 | dt 1.0e-03 | avg/step 1s
steps (cumulative/segment): 32 / 32 | elapsed: 4.250000 s
time (initial/final): 0.000000 / 0.200000
[explicit-retry] event=run-summary accepted_attempts=999 rejected_attempts=999 minimum_accepted_dt=9
[explicit-time] event=run-summary steps=32 final_time=0.2 rhs_evaluations=103 accepted_attempts=32 rejected_attempts=2 rejected_limiter_elements=7 rejected_q_sigma_attempts=1 rejected_q_sigma_nonpositive_samples=3 rejected_q_sigma_nonfinite_samples=0 last_q_sigma_rejection_reason=q-sigma-nonpositive last_q_sigma_rejection_stage=6 characteristic_speed_probes=35 minimum_characteristic_speed=2.25 maximum_characteristic_speed=19.5 minimum_attempted_dt=0.00025 maximum_attempted_dt=0.002 minimum_accepted_dt=0.0005 maximum_accepted_dt=0.001 maximum_accepted_step_growth_ratio=1.75 dt_min_reached=0 success=1 termination_reason=target-reached
[igr-closure-certificate] event=accepted-state scope=final-committed-accepted-state precision=fp64 time=0.2 state_generation=32 solved=1 supported=1 certified=1 residual_rel_rhs=2.5e-8 run_success=1 residual_abs=1e-9 relative_limit=5e-5 absolute_limit=0 skipped_nonfinite=0
[explicit-conservation] event=accepted-trajectory supported=1 source_supported=1 components=4 domain_measure=2 initial=1,2,3,4 final=1.1,2.2,3.3,4.4 boundary_flux=-0.1,-0.2,-0.3,-0.4 boundary_flux_scale=0.2,0.3,0.4,0.5 defect=1e-14,2e-14,3e-14,4e-14 relative_defect=1e-15,2e-15,3e-15,4e-15 maximum_absolute_error=4e-14 maximum_relative_error=4e-15 accepted_attempts=32 rejected_attempts=2 run_success=1
[explicit-time] event=final-status final_time=0.2 success=1 termination_reason=target-reached
[igr-regularization] event=resolved c_alpha=1 h_eff_policy=max h_min=0.04 h_average=0.06 h_max=0.08 h_eff_reference=0.08 alpha_effective_min=0.0064 alpha_effective_max=0.0064
[igr-sigma-lifecycle] event=run-summary prefetch_solves=1 stage0_reuses=2 output_only_solves=0 prefetch_invalidations=2 endpoint_solves=34 extrapolated_stages=321 exact_anchor_certificates=36 exact_anchor_absolute_only_passes=0 max_exact_anchor_certificate_residual_rel_rhs=2.5e-8 max_exact_anchor_certificate_relative_limit=5e-5 max_exact_anchor_certificate_residual_abs=1e-9 max_exact_anchor_certificate_absolute_limit=0 max_exact_anchor_certificate_acceptance_ratio=0.5 max_exact_anchor_relative_pass_residual_rel_rhs=2.5e-8 max_exact_anchor_absolute_only_residual_abs=0 max_endpoint_prediction_defect=0.0125 max_relative_endpoint_prediction_defect=0.025
[igr-q-sigma] event=run-summary scope=application-total-attempted-speed-work rejected_attempts_included=1 precision=fp64 sample_count=15000 nonpositive_count=3 nonfinite_count=0 minimum_raw_q_sigma=-0.125 minimum_q_sigma_over_gamma_p=-0.05
[igr-linear] event=run-summary calls=103 converged_calls=103 production_calls=102 production_converged_calls=102 output_only_calls=1 output_only_converged_calls=1 output_only_iters=3 total_iters=420 avg_iters=4.077 max_iters=9 total_assemble_s=0.4 total_form_s=0.2 total_pc_setup_s=0.3 total_solve_s=1.5 total_igr_s=2.5 total_update_s=2.7 total_krylov_operator_apply_count=525 total_krylov_preconditioner_apply_count=420 total_krylov_preconditioner_operator_apply_count=1260 last_solver=minres last_precond=operatorchebyshev-base exact_face_requested=1 exact_face_effective=1 exact_face_operator_uses=103 exact_face_last_mult_calls=7
[limiter] event=run-summary rank=0 calls=98 active_calls=20 limited_elements=70 element_call_opportunities=1960 limited_element_fraction=0.0357142857143 minimum_theta=0.2 aggregation=element-call-incidence limited_fraction_denominator=element_call_opportunities total_s=0.6 avg_s=0.00612 avg_active_elements=0.714285714286 avg_tensor_points=16
ARCHITECTURE_SELECTION_POINT nonlinear_updates=0 linear_iterations=420 trace_solves=0 scalar_closure_solves=103 complete_assemblies=103 positivity_limiter_calls=98 positivity_limiter_active_calls=20 positivity_limited_elements=70 minimum_density=0.5 minimum_pressure=0.25 maximum_conservation_error=1e-10 minimum_positivity_limiter_theta=0.2
"""
        parsed = parse_application_log(log, DG_RK3SSP)
        self.assertEqual(parsed["steps"], 32)
        self.assertEqual(parsed["segment_steps"], 32)
        self.assertEqual(parsed["accepted_steps"], 32)
        self.assertEqual(parsed["rejected_attempts"], 2)
        self.assertEqual(parsed["time_reached"], 0.2)
        self.assertEqual(parsed["time_reached_authority"],
                         "explicit-final-status")
        self.assertEqual(parsed["internal_time_loop_wall_seconds"], 4.25)
        self.assertEqual(
            parsed["explicit_time"]["source_marker"],
            "explicit-time-final-status",
        )
        self.assertTrue(parsed["explicit_time"]["final_status_reported"])
        self.assertEqual(parsed["explicit_time"]["rhs_evaluations"], 103)
        self.assertEqual(parsed["explicit_time"]["rejected_limiter_elements"], 7)
        self.assertEqual(
            parsed["explicit_time"]["rejected_q_sigma_attempts"], 1
        )
        self.assertEqual(
            parsed["explicit_time"]
            ["rejected_q_sigma_nonpositive_samples"],
            3,
        )
        self.assertEqual(
            parsed["explicit_time"]["rejected_q_sigma_nonfinite_samples"],
            0,
        )
        self.assertEqual(
            parsed["explicit_time"]["last_q_sigma_rejection_reason"],
            "q-sigma-nonpositive",
        )
        self.assertEqual(
            parsed["explicit_time"]["last_q_sigma_rejection_stage"], 6
        )
        self.assertEqual(
            parsed["explicit_time"]["characteristic_speed_probes"], 35
        )
        self.assertEqual(
            parsed["explicit_time"]["minimum_characteristic_speed"], 2.25
        )
        self.assertEqual(
            parsed["explicit_time"]["maximum_characteristic_speed"], 19.5
        )
        self.assertEqual(parsed["explicit_time"]["maximum_accepted_dt"], 0.001)
        self.assertEqual(
            parsed["explicit_time"]["maximum_accepted_step_growth_ratio"],
            1.75,
        )
        self.assertTrue(parsed["explicit_time"]["success"])
        self.assertEqual(
            parsed["explicit_time"]["termination_reason"], "target-reached"
        )
        self.assertTrue(parsed["closure_certificate"]["certified"])
        self.assertTrue(parsed["closure_certificate"]["supported"])
        self.assertTrue(parsed["closure_certificate"]["run_success"])
        self.assertEqual(
            parsed["closure_certificate"]["residual_rel_rhs"], 2.5e-8
        )
        self.assertEqual(parsed["closure_certificate"]["scope"],
                         "final-committed-accepted-state")
        self.assertEqual(parsed["closure_certificate"]["relative_limit"],
                         5.0e-5)
        self.assertTrue(parsed["conservation"]["supported"])
        self.assertEqual(parsed["conservation"]["components"], 4)
        self.assertEqual(parsed["conservation"]["boundary_flux"],
                         [-0.1, -0.2, -0.3, -0.4])
        self.assertEqual(parsed["conservation"]["relative_defect"],
                         [1.0e-15, 2.0e-15, 3.0e-15, 4.0e-15])
        self.assertEqual(
            parsed["conservation"]["maximum_relative_error"], 4.0e-15
        )
        self.assertEqual(parsed["regularization"]["h_eff_policy"], "max")
        self.assertEqual(parsed["regularization"]["h_eff_reference"], 0.08)
        self.assertEqual(
            parsed["regularization"]["alpha_effective_max"], 0.0064
        )
        self.assertEqual(parsed["scalar_closure"]["production_calls"], 102)
        self.assertEqual(parsed["scalar_closure"]["total_iters"], 420)
        self.assertTrue(parsed["scalar_closure"]["exact_face_effective"])
        self.assertEqual(parsed["scalar_lifecycle"]["stage0_reuses"], 2)
        self.assertEqual(parsed["scalar_lifecycle"]["endpoint_solves"], 34)
        self.assertEqual(
            parsed["scalar_lifecycle"]["extrapolated_stages"], 321
        )
        self.assertEqual(
            parsed["scalar_lifecycle"]["exact_anchor_certificates"], 36
        )
        self.assertEqual(
            parsed["scalar_lifecycle"]
            ["exact_anchor_absolute_only_passes"],
            0,
        )
        self.assertEqual(
            parsed["scalar_lifecycle"]
            ["max_exact_anchor_certificate_residual_rel_rhs"],
            2.5e-8,
        )
        self.assertEqual(
            parsed["scalar_lifecycle"]
            ["max_exact_anchor_certificate_relative_limit"],
            5.0e-5,
        )
        self.assertEqual(
            parsed["scalar_lifecycle"]
            ["max_exact_anchor_certificate_residual_abs"],
            1.0e-9,
        )
        self.assertEqual(
            parsed["scalar_lifecycle"]
            ["max_exact_anchor_certificate_absolute_limit"],
            0.0,
        )
        self.assertEqual(
            parsed["scalar_lifecycle"]
            ["max_exact_anchor_certificate_acceptance_ratio"],
            0.5,
        )
        self.assertEqual(
            parsed["scalar_lifecycle"]
            ["max_exact_anchor_relative_pass_residual_rel_rhs"],
            2.5e-8,
        )
        self.assertEqual(
            parsed["scalar_lifecycle"]
            ["max_exact_anchor_absolute_only_residual_abs"],
            0.0,
        )
        self.assertEqual(
            parsed["scalar_lifecycle"]["max_endpoint_prediction_defect"],
            0.0125,
        )
        self.assertEqual(
            parsed["scalar_lifecycle"]
            ["max_relative_endpoint_prediction_defect"],
            0.025,
        )
        q_sigma = parsed["frozen_sigma_telemetry"]
        self.assertTrue(q_sigma["reported"])
        self.assertTrue(q_sigma["rejected_attempts_included"])
        self.assertEqual(q_sigma["precision"], "fp64")
        self.assertEqual(q_sigma["sample_count"], 15000)
        self.assertEqual(q_sigma["nonpositive_count"], 3)
        self.assertEqual(q_sigma["nonfinite_count"], 0)
        self.assertEqual(q_sigma["minimum_raw_q_sigma"], -0.125)
        self.assertEqual(q_sigma["minimum_q_sigma_over_gamma_p"], -0.05)
        self.assertEqual(parsed["limiter"]["calls"], 98)
        self.assertEqual(parsed["limiter"]["active_calls"], 20)
        self.assertEqual(parsed["limiter"]["limited_elements"], 70)
        self.assertEqual(
            parsed["limiter"]["element_call_opportunities"], 1960
        )
        self.assertAlmostEqual(
            parsed["limiter"]["limited_element_fraction"], 70.0 / 1960.0
        )
        self.assertEqual(parsed["limiter"]["minimum_theta"], 0.2)
        self.assertEqual(
            parsed["limiter"]["limited_fraction_denominator"],
            "element_call_opportunities",
        )
        self.assertEqual(parsed["architecture_work"]["positivity_limited_elements"], 70)

    def test_q_sigma_parser_keeps_missing_or_nonfinite_minima_nullable(
            self) -> None:
        parsed = parse_application_log(
            "[igr-q-sigma] event=run-summary sample_count=12 "
            "nonpositive_count=0 nonfinite_count=1 "
            "minimum_raw_q_sigma=nan "
            "minimum_q_sigma_over_gamma_p=inf\n",
            DG_RK3SSP,
        )["frozen_sigma_telemetry"]
        self.assertTrue(parsed["reported"])
        self.assertEqual(parsed["sample_count"], 12)
        self.assertEqual(parsed["nonpositive_count"], 0)
        self.assertEqual(parsed["nonfinite_count"], 1)
        self.assertIsNone(parsed["minimum_raw_q_sigma"])
        self.assertIsNone(parsed["minimum_q_sigma_over_gamma_p"])

    def test_limiter_rank_rows_use_slowest_timing_and_global_minimum_theta(
            self) -> None:
        log = """
[limiter] event=run-summary rank=0 calls=4 active_calls=2 limited_elements=3 element_call_opportunities=40 limited_element_fraction=0.075 minimum_theta=0.4 aggregation=element-call-incidence limited_fraction_denominator=element_call_opportunities total_s=0.7 avg_s=0.175 avg_active_elements=0.75
[limiter] event=run-summary rank=1 calls=4 active_calls=2 limited_elements=3 element_call_opportunities=40 limited_element_fraction=0.075 minimum_theta=0.25 aggregation=element-call-incidence limited_fraction_denominator=element_call_opportunities total_s=0.9 avg_s=0.225 avg_active_elements=0.75
"""
        limiter = parse_application_log(log, DG_RK3SSP)["limiter"]
        self.assertEqual(limiter["reported_ranks"], 2)
        self.assertEqual(limiter["calls"], 4)
        self.assertEqual(limiter["limited_elements"], 3)
        self.assertEqual(limiter["total_s"], 0.9)
        self.assertEqual(limiter["minimum_theta"], 0.25)
        self.assertEqual(limiter["aggregation"], "element-call-incidence")

    def test_missing_explicit_summary_remains_nullable(self) -> None:
        log = """
steps (cumulative/segment): 9 / 9 | elapsed: 1.0 s
time (initial/final): 0 / 0.2
"""
        parsed = parse_application_log(log, DG_SSPRK104)
        self.assertIsNone(parsed["accepted_steps"])
        self.assertIsNone(parsed["rejected_attempts"])
        self.assertFalse(parsed["explicit_time"]["reported"])
        self.assertIsNone(parsed["explicit_time"]["rhs_evaluations"])
        self.assertIsNone(
            parsed["explicit_time"]["rejected_q_sigma_attempts"]
        )
        self.assertIsNone(
            parsed["explicit_time"]["last_q_sigma_rejection_reason"]
        )
        self.assertIsNone(
            parsed["explicit_time"]["characteristic_speed_probes"]
        )
        self.assertIsNone(
            parsed["explicit_time"]["minimum_characteristic_speed"]
        )
        self.assertFalse(parsed["limiter"]["reported"])
        self.assertIsNone(parsed["limiter"]["limited_element_fraction"])
        self.assertIsNone(parsed["limiter"]["minimum_theta"])

    def test_nonfinite_closure_certificate_is_reported_but_not_finite(self) -> None:
        parsed = parse_application_log(
            "[igr-closure-certificate] event=accepted-state time=0.2 "
            "state_generation=5 solved=0 certified=1 residual_rel_rhs=nan\n",
            DG_RK3SSP,
        )
        certificate = parsed["closure_certificate"]
        self.assertTrue(certificate["reported"])
        self.assertTrue(certificate["residual_reported"])
        self.assertFalse(certificate["residual_finite"])
        self.assertIsNone(certificate["residual_rel_rhs"])

    def test_hdg_success_parses_adaptive_and_regularization_fields(self) -> None:
        log = """
[hdg-progress] event=accepted step=10 time=0.12 accepted_dt=0.008 next_dt=0.009 rejected_attempts=2
[hdg-progress] event=accepted step=17 time=0.2 accepted_dt=0.004 next_dt=0.006 rejected_attempts=3
[hdg-main] status=Success step=17 time=0.2 output_cycle=1 cuda_engine=tensor time_treatment=rodas3 requested_trace_policy=engine-default resolved_trace_policy=native trace_solver=gmres fallback_count=0 trace_preconditioner=native regularization=igr positivity_limiter=zhang-shu zhang_shu_points=consumer c_alpha=1 alpha_effective=0.0104992 alpha_length_policy=max h_eff=0.102466 euler_trace_penalty=1 rejected_attempts=3 minimum_accepted_dt=0.003 maximum_accepted_dt=0.02
"""
        parsed = parse_application_log(log, HDG_RODAS3_TENSOR)
        self.assertEqual(parsed["steps"], 17)
        self.assertEqual(parsed["accepted_steps"], 17)
        self.assertEqual(parsed["rejected_attempts"], 3)
        self.assertEqual(parsed["time_reached"], 0.2)
        self.assertEqual(parsed["hdg"]["cuda_engine"], "tensor")
        self.assertAlmostEqual(parsed["hdg"]["alpha_effective"], 0.0104992)
        self.assertEqual(parsed["hdg"]["accepted_progress_records"], 2)
        self.assertEqual(parsed["hdg"]["minimum_accepted_dt"], 0.003)
        self.assertEqual(parsed["hdg"]["minimum_reported_accepted_dt"], 0.004)
        self.assertFalse(parsed["scalar_closure"]["reported"])
        self.assertIsNone(parsed["scalar_closure"]["calls"])
        self.assertFalse(parsed["architecture_work"]["reported"])
        self.assertIsNone(parsed["architecture_work"]["trace_solves"])
        self.assertFalse(parsed["regularization"]["reported"])
        self.assertIsNone(parsed["regularization"]["alpha_effective_min"])

    def test_failure_progress_and_reason_are_distinct(self) -> None:
        explicit_log = """
[alive] step     44 | t     0.137500 | dt 1.0e-08 | avg/step 1s
[dt-check] dt below minimum threshold: dt=9e-11 dt_min=1e-10 stage=after-cfl-update t=0.138
"""
        parsed = parse_application_log(explicit_log, DG_SSPRK104)
        self.assertIsNone(parsed["time_reached"])
        self.assertEqual(parsed["diagnostic_progress_time"], 0.138)
        self.assertIsNone(parsed["time_reached_authority"])
        self.assertEqual(parsed["failure_reason_from_log"], "dt_below_minimum")

        hdg_log = """
[hdg-progress] event=accepted step=11 time=0.137 accepted_dt=1e-7 next_dt=5e-8 rejected_attempts=4
[admissibility-scan] event=euler label=attempt point_set=limiter t=0.138 bad_rho=0 bad_pressure=2 min_rho=0.8 min_pressure=-0.1 first_bad_rank=0
[hdg-main] status=Rejected reason=adaptive RODAS3 minimum dt reached committed_step=11 committed_t=1.375e-1 attempted_dt=5e-8 next_dt=2.5e-8 consecutive_rejections=5 cuda_engine=ordinary
"""
        parsed = parse_application_log(hdg_log, HDG_RODAS3_ORDINARY)
        self.assertIn(
            "adaptive RODAS3 minimum dt reached",
            parsed["failure_reason_from_log"],
        )
        self.assertEqual(parsed["time_reached"], 0.1375)
        self.assertEqual(parsed["accepted_steps"], 11)
        self.assertEqual(parsed["hdg"]["committed_step"], 11)
        self.assertEqual(parsed["hdg"]["committed_t"], 0.1375)

    def test_admissibility_failure_uses_minima(self) -> None:
        log = """
[admissibility-scan] event=euler label=stage point_set=limiter t=0.11 bad_rho=2 bad_pressure=3 min_rho=-0.1 min_pressure=-2 first_bad_rank=0
[admissibility-scan] event=euler label=stage point_set=output t=0.12 bad_rho=1 bad_pressure=4 min_rho=-0.2 min_pressure=-1 first_bad_rank=0
"""
        parsed = parse_application_log(log, DG_RK3SSP)
        self.assertEqual(parsed["admissibility"]["violation_records"], 2)
        self.assertEqual(parsed["admissibility"]["maximum_bad_pressure_points"], 4)
        self.assertEqual(parsed["admissibility"]["minimum_density"], -0.2)
        self.assertEqual(parsed["failure_reason_from_log"], "admissibility_violation")

    def test_successful_explicit_summary_ignores_intermediate_scan_warnings(self) -> None:
        log = """
[admissibility-scan] event=euler label=output t=0.2 bad_rho=0 bad_pressure=3 min_rho=0.8 min_pressure=-0.01 first_bad_rank=0
[explicit-time] event=run-summary steps=2 final_time=0.2 rhs_evaluations=7 accepted_attempts=2 rejected_attempts=0 rejected_limiter_elements=0 minimum_attempted_dt=0.1 maximum_attempted_dt=0.1 minimum_accepted_dt=0.1 maximum_accepted_dt=0.1 dt_min_reached=0 success=1 termination_reason=target-reached
[explicit-time] event=final-status final_time=0.2 success=1 termination_reason=target-reached
"""
        parsed = parse_application_log(log, DG_RK3SSP)
        self.assertIsNone(parsed["failure_reason_from_log"])

    def test_explicit_limiter_terminal_reasons_remain_distinct(self) -> None:
        for reason in ("limiter-retry-limit", "limiter-dt-min"):
            with self.subTest(reason=reason):
                log = (
                    "[explicit-time] event=run-summary steps=4 "
                    "final_time=0.03 rhs_evaluations=19 "
                    "accepted_attempts=4 rejected_attempts=8 "
                    "rejected_limiter_elements=3 "
                    "minimum_attempted_dt=1e-12 maximum_attempted_dt=1e-3 "
                    "minimum_accepted_dt=1e-4 maximum_accepted_dt=1e-3 "
                    f"dt_min_reached=1 success=0 termination_reason={reason}\n"
                )
                parsed = parse_application_log(log, DG_RK3SSP)
                self.assertEqual(parsed["failure_reason_from_log"], reason)

    def test_explicit_terminal_marker_is_committed_time_authority(self) -> None:
        log = """
[alive] step     19 | t     0.029140281 | dt 2.0e-10 | avg/step 1s
[admissibility-scan] event=euler label=attempt point_set=limiter t=0.029699884 bad_rho=0 bad_pressure=3 min_rho=0.7 min_pressure=-0.2 first_bad_rank=0
[explicit-time] event=run-summary steps=19 final_time=0.029140281 rhs_evaluations=437 accepted_attempts=19 rejected_attempts=166 rejected_limiter_elements=450 minimum_attempted_dt=1e-10 maximum_attempted_dt=0.001 minimum_accepted_dt=2e-10 maximum_accepted_dt=0.001 dt_min_reached=1 success=0 termination_reason=limiter-dt-min
[explicit-time] event=final-status final_time=0.029140281 success=0 termination_reason=limiter-dt-min
"""
        parsed = parse_application_log(log, DG_SSPRK104)
        self.assertAlmostEqual(parsed["time_reached"], 0.029140281)
        self.assertEqual(parsed["steps"], 19)
        self.assertEqual(parsed["failure_reason_from_log"], "limiter-dt-min")

    def test_explicit_final_status_overrides_precertificate_summary(self) -> None:
        log = """
[explicit-time] event=run-summary steps=8 final_time=0.2 rhs_evaluations=25 accepted_attempts=8 rejected_attempts=0 rejected_limiter_elements=0 minimum_attempted_dt=0.01 maximum_attempted_dt=0.04 minimum_accepted_dt=0.01 maximum_accepted_dt=0.04 dt_min_reached=0 success=1 termination_reason=target-reached
[igr-closure-certificate] event=accepted-state time=0.2 state_generation=8 solved=1 supported=1 certified=0 residual_rel_rhs=inf run_success=0
[explicit-time] event=final-status final_time=0.2 success=0 termination_reason=igr-closure-certificate-failed
"""
        parsed = parse_application_log(log, DG_RK3SSP)
        self.assertFalse(parsed["explicit_time"]["success"])
        self.assertEqual(
            parsed["explicit_time"]["termination_reason"],
            "igr-closure-certificate-failed",
        )
        self.assertEqual(
            parsed["failure_reason_from_log"],
            "igr-closure-certificate-failed",
        )


class OutcomeAndReportTests(unittest.TestCase):
    def test_execution_never_confuses_success_failure_and_censoring(self) -> None:
        parsed = {"time_reached": 0.2, "failure_reason_from_log": None}
        success = classify_execution(
            parsed, returncode=0, timed_out=False, target_time=0.2,
            application_wall_seconds=3.0,
        )
        self.assertEqual(success["kind"], "success")
        self.assertEqual(success["time_to_solution_seconds"], 3.0)
        self.assertIsNone(success["time_to_failure_seconds"])

        failure = classify_execution(
            {"time_reached": 0.1, "failure_reason_from_log": "dt_min"},
            returncode=4, timed_out=False, target_time=0.2,
            application_wall_seconds=2.0,
        )
        self.assertEqual(failure["kind"], "failure")
        self.assertEqual(failure["time_to_failure_seconds"], 2.0)
        self.assertIsNone(failure["time_to_solution_seconds"])

        censored = classify_execution(
            {"time_reached": 0.15}, returncode=-15, timed_out=True,
            target_time=0.2, application_wall_seconds=10.0,
        )
        self.assertEqual(censored["kind"], "censored")
        self.assertEqual(censored["censoring_wall_seconds"], 10.0)
        self.assertIsNone(censored["time_to_failure_seconds"])

        unobserved = classify_execution(
            {"time_reached": None}, returncode=0, timed_out=False,
            target_time=0.2, application_wall_seconds=1.0,
        )
        self.assertEqual(unobserved["kind"], "failure")
        self.assertEqual(unobserved["reason"], "target_horizon_not_observed")

        reported_failure = classify_execution(
            {
                "time_reached": 0.2,
                "failure_reason_from_log": "limiter-retry-limit",
                "explicit_time": {"success": False},
            },
            returncode=0, timed_out=False, target_time=0.2,
            application_wall_seconds=1.5,
        )
        self.assertEqual(reported_failure["kind"], "failure")
        self.assertEqual(reported_failure["reason"], "limiter-retry-limit")

    def test_qoi_timeout_is_quality_censoring_not_execution_failure(self) -> None:
        execution = classify_execution(
            {"time_reached": 0.2}, returncode=0, timed_out=False,
            target_time=0.2, application_wall_seconds=3.0,
        )
        requirements = parse_qoi_requirements(("shock_position=1.0:0.1",))
        quality = evaluate_quality(
            qualification_mode=True,
            method=DG_RK3SSP,
            execution=execution,
            qoi_status="censored",
            qoi_values={"shock_position": None},
            requirements=requirements,
            qoi_failure="qoi_postprocessor_timeout",
            closure_certificate=certified_closure(),
        )
        selection = classify_selection(
            execution, quality, qualification_mode=True,
            end_to_end_wall_seconds=8.0,
        )
        self.assertEqual(execution["kind"], "success")
        self.assertEqual(quality["status"], "censored")
        self.assertEqual(selection["kind"], "quality_censored")
        self.assertIsNone(selection["time_to_qualified_solution_seconds"])

    def test_quality_requires_each_caller_selected_gate(self) -> None:
        execution = classify_execution(
            {"time_reached": 0.2}, returncode=0, timed_out=False,
            target_time=0.2, application_wall_seconds=3.0,
        )
        requirements = parse_qoi_requirements((
            "shock_position=1.0:0.1",
            "shock_thickness=0.2:0.02",
        ))
        passed = evaluate_quality(
            qualification_mode=True, method=DG_RK3SSP,
            execution=execution,
            qoi_status="success",
            qoi_values={"shock_position": 1.05, "shock_thickness": 0.19},
            requirements=requirements, qoi_failure=None,
            closure_certificate=certified_closure(),
        )
        self.assertTrue(passed["qualified"])
        self.assertEqual(passed["status"], "passed")
        failed = evaluate_quality(
            qualification_mode=True, method=DG_RK3SSP,
            execution=execution,
            qoi_status="success",
            qoi_values={"shock_position": 1.05, "shock_thickness": None},
            requirements=requirements, qoi_failure=None,
            closure_certificate=certified_closure(),
        )
        self.assertFalse(failed["qualified"])
        self.assertEqual(failed["status"], "failed")
        with self.assertRaisesRegex(ShockArchitectureError, "unknown QoI"):
            parse_qoi_requirements(("fictional=1:0.1",))

    def test_dg_quality_requires_finite_certified_accepted_state_closure(self) -> None:
        execution = classify_execution(
            {"time_reached": 0.2}, returncode=0, timed_out=False,
            target_time=0.2, application_wall_seconds=3.0,
        )
        requirements = parse_qoi_requirements(("shock_position=1.0:0.1",))
        cases = (
            ({"reported": False}, "missing_dg"),
            ({
                "reported": True, "supported": False, "certified": False,
                "residual_rel_rhs": None,
            }, "unsupported_dg"),
            ({**certified_closure(), "certified": False}, "uncertified_dg"),
            ({**certified_closure(), "residual_rel_rhs": None,
              "residual_finite": False}, "invalid_dg"),
            ({**certified_closure(), "solved": False}, "unsolved_dg"),
            ({**certified_closure(), "run_success": False}, "unsuccessful_dg"),
            ({**certified_closure(), "time": 0.1}, "wrong_time_dg"),
            ({**certified_closure(), "precision": "fp32"}, "invalid_dg"),
            ({**certified_closure(), "residual_rel_rhs": 2.0e-6},
             "invalid_dg"),
        )
        for certificate, reason in cases:
            with self.subTest(reason=reason):
                quality = evaluate_quality(
                    qualification_mode=True, method=DG_SSPRK104,
                    execution=execution, qoi_status="success",
                    qoi_values={"shock_position": 1.0},
                    requirements=requirements, qoi_failure=None,
                    closure_certificate=certificate,
                )
                self.assertEqual(quality["status"], "failed")
                self.assertIn(reason, quality["reason"])

        hdg_quality = evaluate_quality(
            qualification_mode=True, method=HDG_RODAS3_TENSOR,
            execution=execution, qoi_status="success",
            qoi_values={"shock_position": 1.0},
            requirements=requirements, qoi_failure=None,
            closure_certificate={"reported": False},
        )
        self.assertFalse(hdg_quality["qualified"])
        self.assertEqual(
            hdg_quality["reason"], "hdg_quality_contract_not_implemented"
        )

    def test_exact_iid_quality_requires_reconciled_q_sigma_samples(
            self) -> None:
        execution = classify_execution(
            {"time_reached": 0.2}, returncode=0, timed_out=False,
            target_time=0.2, application_wall_seconds=3.0,
        )
        requirements = parse_qoi_requirements(("shock_position=1.0:0.1",))
        common = {
            "qualification_mode": True,
            "method": DG_SSPRK104,
            "execution": execution,
            "qoi_status": "success",
            "qoi_values": {"shock_position": 1.0},
            "requirements": requirements,
            "qoi_failure": None,
            "closure_certificate": certified_closure(),
            "scalar_lifecycle": certified_extrapolated_lifecycle(),
            "explicit_time": q_sigma_rejection_summary(),
            "required_q_sigma_evidence_scope": EXACT_IID_Q_SIGMA_SCOPE,
        }
        passed = evaluate_quality(
            **common,
            frozen_sigma_telemetry=positive_q_sigma_telemetry(),
        )
        self.assertTrue(passed["qualified"])
        self.assertTrue(passed["q_sigma_containment"]["contained"])

        contained_telemetry = {
            **positive_q_sigma_telemetry(),
            "sample_count": 120,
            "nonpositive_count": 3,
            "nonfinite_count": 2,
            "minimum_raw_q_sigma": -0.125,
            "minimum_q_sigma_over_gamma_p": -0.05,
        }
        contained = evaluate_quality(
            **{
                **common,
                "explicit_time": q_sigma_rejection_summary(
                    attempts=1, nonpositive=3, nonfinite=2
                ),
            },
            frozen_sigma_telemetry=contained_telemetry,
        )
        self.assertTrue(contained["qualified"])
        self.assertTrue(contained["q_sigma_containment"]["contained"])
        self.assertTrue(
            contained["q_sigma_containment"]["counts_reconciled"]
        )

        underreported_attempts_telemetry = {
            **positive_q_sigma_telemetry(),
            "nonpositive_count": 1,
            "minimum_raw_q_sigma": -0.125,
            "minimum_q_sigma_over_gamma_p": -0.05,
        }
        underreported_attempts = evaluate_quality(
            **{
                **common,
                "explicit_time": q_sigma_rejection_summary(
                    attempts=2, nonpositive=1, nonfinite=0
                ),
            },
            frozen_sigma_telemetry=underreported_attempts_telemetry,
        )
        self.assertFalse(underreported_attempts["qualified"])
        self.assertEqual(
            underreported_attempts["reason"],
            "invalid_exact_iid_q_sigma_telemetry",
        )
        self.assertFalse(
            underreported_attempts["q_sigma_containment"]["contained"]
        )

        cases = (
            ({"reported": False}, "missing_exact_iid_q_sigma"),
            ({**positive_q_sigma_telemetry(), "sample_count": 0},
             "invalid_exact_iid_q_sigma"),
            ({**positive_q_sigma_telemetry(), "nonpositive_count": 1},
             "invalid_exact_iid_q_sigma"),
            ({**positive_q_sigma_telemetry(), "nonfinite_count": 1},
             "invalid_exact_iid_q_sigma"),
            ({**positive_q_sigma_telemetry(), "minimum_raw_q_sigma": 0.0},
             "invalid_exact_iid_q_sigma"),
            ({**positive_q_sigma_telemetry(),
              "minimum_q_sigma_over_gamma_p": float("inf")},
             "invalid_exact_iid_q_sigma"),
            ({**positive_q_sigma_telemetry(), "precision": "fp32"},
             "invalid_exact_iid_q_sigma"),
            ({**positive_q_sigma_telemetry(),
              "scope": "accepted-attempts-only"},
             "invalid_exact_iid_q_sigma"),
            ({**positive_q_sigma_telemetry(),
              "rejected_attempts_included": False},
             "invalid_exact_iid_q_sigma"),
        )
        for telemetry, reason in cases:
            with self.subTest(reason=reason, telemetry=telemetry):
                failed = evaluate_quality(
                    **common, frozen_sigma_telemetry=telemetry
                )
                self.assertFalse(failed["qualified"])
                self.assertEqual(failed["status"], "failed")
                self.assertIn(reason, failed["reason"])

        for time_summary in (
            q_sigma_rejection_summary(
                attempts=1, nonpositive=2, nonfinite=2
            ),
            q_sigma_rejection_summary(
                attempts=1, nonpositive=3, nonfinite=4
            ),
            q_sigma_rejection_summary(
                attempts=0, nonpositive=3, nonfinite=2
            ),
            q_sigma_rejection_summary(
                attempts=1, nonpositive=3, nonfinite=2,
                reason="none",
            ),
            q_sigma_rejection_summary(
                attempts=1, nonpositive=3, nonfinite=2, stage=-2
            ),
            {"reported": False},
        ):
            with self.subTest(time_summary=time_summary):
                failed = evaluate_quality(
                    **{**common, "explicit_time": time_summary},
                    frozen_sigma_telemetry=contained_telemetry,
                )
                self.assertFalse(failed["qualified"])
                self.assertEqual(
                    failed["reason"],
                    "invalid_exact_iid_q_sigma_telemetry",
                )
                self.assertFalse(
                    failed["q_sigma_containment"]["contained"]
                )

        failed_certificate = evaluate_quality(
            **{
                **common,
                "closure_certificate": {
                    **certified_closure(), "certified": False
                },
                "explicit_time": q_sigma_rejection_summary(
                    attempts=1, nonpositive=3, nonfinite=2
                ),
            },
            frozen_sigma_telemetry=contained_telemetry,
        )
        self.assertFalse(failed_certificate["qualified"])
        self.assertEqual(
            failed_certificate["reason"],
            "uncertified_dg_accepted_state_closure",
        )

        measurement_only = evaluate_quality(
            **{**common, "qualification_mode": False},
            frozen_sigma_telemetry={"reported": False},
        )
        self.assertEqual(measurement_only["status"], "measured_not_gated")
        hdg = evaluate_quality(
            **{**common, "method": HDG_RODAS3_ORDINARY,
               "closure_certificate": {"reported": False}},
            frozen_sigma_telemetry={"reported": False},
        )
        self.assertFalse(hdg["qualified"])
        self.assertEqual(
            hdg["reason"], "hdg_quality_contract_not_implemented"
        )

    def test_extrapolated_quality_requires_frozen_sigma_scope(self) -> None:
        execution = classify_execution(
            {"time_reached": 0.2}, returncode=0, timed_out=False,
            target_time=0.2, application_wall_seconds=3.0,
        )
        common = {
            "qualification_mode": True,
            "method": DGSEM_SSPRK104_EXTRAPOLATED,
            "execution": execution,
            "qoi_status": "success",
            "qoi_values": {"shock_position": 1.0},
            "requirements": parse_qoi_requirements((
                "shock_position=1.0:0.1",
            )),
            "qoi_failure": None,
            "closure_certificate": certified_closure(),
            "scalar_closure": certified_extrapolated_scalar_closure(),
            "scalar_lifecycle": certified_extrapolated_lifecycle(),
            "explicit_time": q_sigma_rejection_summary(),
            "required_q_sigma_evidence_scope": EXTRAPOLATED_Q_SIGMA_SCOPE,
        }
        passed = evaluate_quality(
            **common,
            frozen_sigma_telemetry=positive_q_sigma_telemetry(
                EXTRAPOLATED_Q_SIGMA_SCOPE
            ),
        )
        self.assertTrue(passed["qualified"])

        contained_telemetry = {
            **positive_q_sigma_telemetry(EXTRAPOLATED_Q_SIGMA_SCOPE),
            "nonpositive_count": 4,
            "minimum_raw_q_sigma": -0.25,
            "minimum_q_sigma_over_gamma_p": -0.1,
        }
        contained = evaluate_quality(
            **{
                **common,
                "explicit_time": q_sigma_rejection_summary(
                    attempts=1, nonpositive=4
                ),
            },
            frozen_sigma_telemetry=contained_telemetry,
        )
        self.assertTrue(contained["qualified"])
        self.assertTrue(contained["q_sigma_containment"]["contained"])

        wrong_scope = evaluate_quality(
            **common,
            frozen_sigma_telemetry=positive_q_sigma_telemetry(
                EXACT_IID_Q_SIGMA_SCOPE
            ),
        )
        self.assertFalse(wrong_scope["qualified"])
        self.assertEqual(
            wrong_scope["reason"], "invalid_frozen_sigma_q_sigma_telemetry"
        )

    def test_closure_certificate_accepts_bounded_absolute_alternative(
            self) -> None:
        execution = classify_execution(
            {"time_reached": 0.2}, returncode=0, timed_out=False,
            target_time=0.2, application_wall_seconds=3.0,
        )
        certificate = {
            **certified_closure(),
            "residual_rel_rhs": None,
            "residual_finite": False,
            "residual_abs": 1.0e-12,
            "absolute_limit": 5.0e-12,
        }
        common = {
            "qualification_mode": True,
            "method": DG_RK3SSP,
            "execution": execution,
            "qoi_status": "success",
            "qoi_values": {},
            "requirements": {},
            "qoi_failure": None,
            "closure_certificate": certificate,
        }
        rejected = evaluate_quality(**common)
        self.assertFalse(rejected["qualified"])
        accepted = evaluate_quality(
            **common,
            closure_certificate_max_absolute_residual=2.0e-12,
        )
        self.assertTrue(accepted["qualified"])

    def test_extrapolated_quality_gates_complete_anchor_history(self) -> None:
        execution = classify_execution(
            {"time_reached": 0.2}, returncode=0, timed_out=False,
            target_time=0.2, application_wall_seconds=3.0,
        )
        common = {
            "qualification_mode": True,
            "method": DGSEM_SSPRK104_EXTRAPOLATED,
            "execution": execution,
            "qoi_status": "success",
            "qoi_values": {},
            "requirements": {},
            "qoi_failure": None,
            "closure_certificate": certified_closure(),
            "scalar_closure": certified_extrapolated_scalar_closure(),
            "explicit_time": q_sigma_rejection_summary(),
            "frozen_sigma_telemetry": positive_q_sigma_telemetry(
                EXTRAPOLATED_Q_SIGMA_SCOPE
            ),
            "required_q_sigma_evidence_scope": EXTRAPOLATED_Q_SIGMA_SCOPE,
        }
        invalid_histories = (
            {"reported": False},
            {**certified_extrapolated_lifecycle(),
             "exact_anchor_certificates": 35},
            {**certified_extrapolated_lifecycle(),
             "max_exact_anchor_certificate_acceptance_ratio": 1.01},
            {**certified_extrapolated_lifecycle(),
             "max_exact_anchor_certificate_acceptance_ratio": float("nan")},
            {**certified_extrapolated_lifecycle(),
             "max_exact_anchor_relative_pass_residual_rel_rhs": 2.0e-6},
            {**certified_extrapolated_lifecycle(absolute_only_passes=1),
             "max_exact_anchor_absolute_only_residual_abs": 1.0e-12},
            {**certified_extrapolated_lifecycle(absolute_only_passes=37)},
        )
        for lifecycle in invalid_histories:
            with self.subTest(lifecycle=lifecycle):
                result = evaluate_quality(
                    **common, scalar_lifecycle=lifecycle
                )
                self.assertFalse(result["qualified"])
                self.assertEqual(
                    result["reason"],
                    "invalid_dg_exact_anchor_certificate_history",
                )
        absolute_history = {
            **certified_extrapolated_lifecycle(absolute_only_passes=1),
            "max_exact_anchor_absolute_only_residual_abs": 1.0e-12,
        }
        accepted = evaluate_quality(
            **common,
            scalar_lifecycle=absolute_history,
            closure_certificate_max_absolute_residual=2.0e-12,
        )
        self.assertTrue(accepted["qualified"])

    def test_extrapolated_quality_requires_full_cg_chebyshev_policy(
            self) -> None:
        execution = classify_execution(
            {"time_reached": 0.2}, returncode=0, timed_out=False,
            target_time=0.2, application_wall_seconds=3.0,
        )
        common = {
            "qualification_mode": True,
            "method": DGSEM_SSPRK104_EXTRAPOLATED,
            "execution": execution,
            "qoi_status": "success",
            "qoi_values": {},
            "requirements": {},
            "qoi_failure": None,
            "closure_certificate": certified_closure(),
            "scalar_lifecycle": certified_extrapolated_lifecycle(),
            "explicit_time": q_sigma_rejection_summary(),
            "frozen_sigma_telemetry": positive_q_sigma_telemetry(
                EXTRAPOLATED_Q_SIGMA_SCOPE
            ),
            "required_q_sigma_evidence_scope": EXTRAPOLATED_Q_SIGMA_SCOPE,
        }
        for scalar_closure in (
            {"reported": False},
            {**certified_extrapolated_scalar_closure(),
             "last_solver": "minres"},
            {**certified_extrapolated_scalar_closure(),
             "last_precond": "operatorchebyshev-base"},
            {**certified_extrapolated_scalar_closure(),
             "exact_face_effective": False},
        ):
            with self.subTest(scalar_closure=scalar_closure):
                result = evaluate_quality(
                    **common, scalar_closure=scalar_closure
                )
                self.assertFalse(result["qualified"])
                self.assertEqual(
                    result["reason"],
                    "invalid_dg_extrapolated_closure_solver_policy",
                )

        passed = evaluate_quality(
            **common,
            scalar_closure=certified_extrapolated_scalar_closure(),
        )
        self.assertTrue(passed["qualified"])

    def test_qoi_contract_is_fixed_numeric_and_nullable(self) -> None:
        parsed = parse_qoi_json(
            json.dumps({"shock_position": 1.25, "shock_thickness": None})
        )
        self.assertEqual(parsed["shock_position"], 1.25)
        self.assertIsNone(parsed["pressure_minimum"])
        for text, message in (
            ('{"unknown": 1}', "unknown fields"),
            ('{"shock_position": "1"}', "numeric or null"),
            ('{"shock_position": NaN}', "finite"),
            ('[]', "must be an object"),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(
                    ShockArchitectureError, message):
                parse_qoi_json(text)

    def test_qoi_command_has_no_shell_template(self) -> None:
        command = qoi_command(
            "/tool/qoi", vtk_root=Path("/scratch/vtk"),
            method=DG_RK3SSP, target_time=0.2,
            mesh="/mesh/cylinder.msh", extra_arguments=("--strict",),
        )
        self.assertEqual(command[0], "/tool/qoi")
        self.assertEqual(option_value(command, "--vtk-root"), "/scratch/vtk")
        self.assertEqual(option_value(command, "--method"), DG_RK3SSP)
        self.assertIn("--strict", command)

    def test_qoi_reference_profile_is_normalized_and_byte_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "nested"
            nested.mkdir()
            reference = root / "reference.json"
            reference.write_bytes(b'{"profile":{"x":[0,1],"density":[1,2]}}\n')
            nonnormalized = nested / ".." / reference.name

            arguments, inputs = pin_qoi_input_files(
                (f"--reference-profile={nonnormalized}", "--case=cylinder"),
                qualification_mode=True,
            )

            self.assertEqual(
                arguments[0], f"--reference-profile={reference.resolve()}"
            )
            self.assertEqual(arguments[1], "--case=cylinder")
            self.assertEqual(len(inputs), 1)
            self.assertEqual(inputs[0]["option"], "--reference-profile")
            self.assertEqual(inputs[0]["argument_index"], 0)
            self.assertEqual(inputs[0]["file"]["path"], str(reference.resolve()))
            self.assertEqual(
                inputs[0]["file"]["sha256"],
                capture_file_identity(reference, "reference")["sha256"],
            )
            verify_file_identity(inputs[0]["file"], "QoI reference profile")
            reference.write_text("mutated\n", encoding="utf-8")
            with self.assertRaisesRegex(ShockArchitectureError, "changed"):
                verify_file_identity(
                    inputs[0]["file"], "QoI reference profile"
                )

    def test_qoi_file_input_supports_split_form_and_excludes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.json"
            reference.write_text("{}\n", encoding="utf-8")
            output = root / "output.json"
            output.write_text("old output\n", encoding="utf-8")
            arguments, inputs = pin_qoi_input_files(
                (
                    "--write-profile", str(output),
                    "--reference-profile", str(reference),
                ),
                qualification_mode=False,
            )
            self.assertEqual(arguments[1], str(output))
            self.assertEqual(arguments[3], str(reference.resolve()))
            self.assertEqual(len(inputs), 1)
            self.assertEqual(inputs[0]["argument_index"], 2)

            with self.assertRaisesRegex(
                    ShockArchitectureError, "abbreviated QoI file input"):
                pin_qoi_input_files(
                    (f"--reference-prof={reference}",),
                    qualification_mode=True,
                )
            with self.assertRaisesRegex(
                    ShockArchitectureError, "harness-owned QoI option --mesh"):
                pin_qoi_input_files(
                    (f"--mesh={reference}",),
                    qualification_mode=True,
                )

    def test_missing_qualification_qoi_input_fails_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing-reference.json"
            with self.assertRaisesRegex(
                    ShockArchitectureError,
                    r"qualification QoI input --reference-profile .*missing"):
                pin_qoi_input_files(
                    (f"--reference-profile={missing}",),
                    qualification_mode=True,
                )
        with self.assertRaisesRegex(
                ShockArchitectureError, "requires a path argument"):
            pin_qoi_input_files(
                ("--reference-profile",), qualification_mode=True
            )

    def test_gnu_time_record_parser_is_strict(self) -> None:
        self.assertEqual(
            parse_time_record(
                "noise\nDG_IGR_EXTERNAL_TIME wall_seconds=2.75 "
                "peak_rss_kib=12345 exit_status=7\n"
            ),
            (2.75, 12345, 7),
        )
        self.assertIsNone(parse_time_record("wall_seconds=2.75 peak=1"))

    def test_aggregates_keep_failure_and_censoring_out_of_success_timing(self) -> None:
        runs = [
            sample_run(DG_RK3SSP, "success", 3.0, time_reached=0.2,
                       qoi_value=1.0),
            sample_run(DG_RK3SSP, "failure", 1.0, time_reached=0.08),
            sample_run(DG_RK3SSP, "censored", 9.0, time_reached=0.1,
                       peak_rss_kib=None),
            sample_run(HDG_RODAS3_TENSOR, "success", 5.0,
                       time_reached=0.2, qoi_value=1.1),
        ]
        aggregate = aggregate_runs(runs, (DG_RK3SSP, HDG_RODAS3_TENSOR))
        explicit = aggregate[DG_RK3SSP]
        self.assertEqual(explicit["execution"]["successes"], 1)
        self.assertEqual(explicit["execution"]["failures"], 1)
        self.assertEqual(explicit["execution"]["censored"], 1)
        self.assertEqual(
            explicit["execution"]["time_to_solution_seconds"]["median"], 3.0
        )
        self.assertEqual(
            explicit["execution"]["maximum_failure_physical_time_reached"],
            0.08,
        )
        self.assertEqual(
            explicit["execution"]["maximum_censored_physical_time_reached"],
            0.1,
        )
        self.assertEqual(
            explicit["selection"]["kind_counts"]["execution_failure"], 1
        )
        self.assertEqual(explicit["qoi"]["shock_position"]["median"], 1.0)

    def test_markdown_labels_censored_measurements(self) -> None:
        runs = [sample_run(DG_RK3SSP, "censored", 9.0, time_reached=0.1)]
        run = runs[0]
        run["progress"]["limiter"] = {
            "reported_ranks": 1,
            "calls": 10,
            "active_calls": 4,
            "limited_elements": 8,
            "element_call_opportunities": 100,
            "limited_element_fraction": 0.08,
            "minimum_theta": 0.3,
            "total_s": 0.5,
        }
        run["progress"]["explicit_time"] = q_sigma_rejection_summary(
            attempts=1, nonpositive=3, rejected_attempts=2
        )
        run["progress"]["scalar_lifecycle"] = {
            "prefetch_solves": 1,
            "stage0_reuses": 2,
            "output_only_solves": 0,
            "prefetch_invalidations": 1,
            "endpoint_solves": 11,
            "extrapolated_stages": 100,
            "exact_anchor_certificates": 13,
            "exact_anchor_absolute_only_passes": 0,
            "max_exact_anchor_certificate_residual_rel_rhs": 2.0e-8,
            "max_exact_anchor_certificate_relative_limit": 5.0e-5,
            "max_exact_anchor_certificate_residual_abs": 1.0e-9,
            "max_exact_anchor_certificate_absolute_limit": 0.0,
            "max_exact_anchor_certificate_acceptance_ratio": 0.5,
            "max_exact_anchor_relative_pass_residual_rel_rhs": 2.0e-8,
            "max_exact_anchor_absolute_only_residual_abs": 0.0,
            "max_endpoint_prediction_defect": 0.01,
            "max_relative_endpoint_prediction_defect": 0.02,
        }
        run["quality"]["q_sigma_containment"] = {
            "counts_reconciled": True,
            "contained": True,
        }
        run.update({
            "run_index": 1,
            "repetition": 1,
            "position": 1,
        })
        report = {
            "created_utc": "2026-09-03T00:00:00+00:00",
            "methods": [DG_RK3SSP],
            "controls": {
                "final_time": 0.2,
                "c_alpha": 1.0,
                "alpha_length_policy": "max",
            },
            "provenance": {
                "git": {"commit": "a" * 40, "tree": "b" * 40,
                        "dirty": False},
                "executable": {"sha256": "c" * 64},
                "mesh": {"sha256": "d" * 64},
                "qoi_input_files": [{
                    "option": "--reference-profile",
                    "argument_index": 0,
                    "file": {
                        "path": "/reference/profile.json",
                        "sha256": "e" * 64,
                    },
                }],
            },
            "runs": runs,
            "aggregates": aggregate_runs(runs, (DG_RK3SSP,)),
        }
        markdown = render_markdown(report)
        self.assertIn("right-censored", markdown)
        self.assertIn("| censored |", markdown)
        self.assertNotIn("Median exec time to solution (s) | 9", markdown)
        self.assertIn("## Explicit DG/DGSEM limiter aggregates", markdown)
        self.assertIn("## Frozen-sigma speed-sample telemetry", markdown)
        self.assertIn("## q_sigma rollback containment", markdown)
        self.assertIn("## Instantaneous IGR sigma lifecycle", markdown)
        self.assertIn(
            "## Explicit DG/DGSEM accepted-trajectory conservation balance",
            markdown,
        )
        self.assertIn("committed SSPRK stages", markdown)
        self.assertIn("Pinned QoI input files", markdown)
        self.assertIn("/reference/profile.json", markdown)
        self.assertIn("e" * 64, markdown)
        self.assertIn(
            "| 1 | dg-rk3ssp | 1 | 3 | 0 | q-sigma-nonpositive | 6 | "
            "True | True |",
            markdown,
        )
        self.assertIn(
            "| 1 | dg-rk3ssp | 1 | 2 | 0 | 1 | 11 | 100 | 13 | 0 | "
            "2e-08 | 5e-05 | 1e-09 | 0 | 0.5 | 2e-08 | 0 | 0.01 | "
            "0.02 |",
            markdown,
        )
        self.assertIn("| 1 | dg-rk3ssp | 1 | 10 | 4 | 8 | 100 | 0.08 | 0.3 | 0.5 |", markdown)
        self.assertIn("element-call incidence ratio", markdown)


class ProcessAndProvenanceIntegrationTests(unittest.TestCase):
    def test_campaign_binds_qoi_input_in_report_and_every_run(self) -> None:
        time_program = Path("/usr/bin/time")
        if not time_program.is_file():
            self.skipTest("GNU time is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(
                ("git", "init", str(root)), check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            subprocess.run(
                ("git", "-C", str(root), "remote", "add", "origin",
                 "https://github.com/comp-physics/dg_igr_miniapp.git"),
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            mesh = root / "mesh.msh"
            mesh.write_text("synthetic mesh\n", encoding="utf-8")
            reference = root / "reference.json"
            reference.write_text("{}\n", encoding="utf-8")
            inputs = root / "inputs"
            inputs.mkdir()
            application = root / "application.py"
            application.write_text(
                """#!/usr/bin/env python3
print("time (initial/final): 0 / 0.2")
print("[explicit-time] event=run-summary steps=2 final_time=0.2 "
      "rhs_evaluations=7 accepted_attempts=2 rejected_attempts=0 "
      "rejected_limiter_elements=0 minimum_attempted_dt=0.1 "
      "maximum_attempted_dt=0.1 minimum_accepted_dt=0.1 "
      "maximum_accepted_dt=0.1 dt_min_reached=0 success=1 "
      "termination_reason=target-reached")
print("[igr-closure-certificate] event=accepted-state "
      "scope=final-committed-accepted-state precision=fp64 time=0.2 "
      "state_generation=2 solved=1 supported=1 certified=1 "
      "residual_rel_rhs=1e-9 run_success=1 residual_abs=1e-10 "
      "relative_limit=5e-5 absolute_limit=0 skipped_nonfinite=0")
print("[explicit-time] event=final-status final_time=0.2 success=1 "
      "termination_reason=target-reached")
""",
                encoding="utf-8",
            )
            qoi = root / "qoi.py"
            qoi.write_text(
                """#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--vtk-root")
parser.add_argument("--method")
parser.add_argument("--target-time")
parser.add_argument("--mesh")
parser.add_argument("--reference-profile", required=True)
arguments = parser.parse_args()
Path(arguments.reference_profile).read_bytes()
print(json.dumps({"shock_position": 1.0}))
""",
                encoding="utf-8",
            )
            application.chmod(0o755)
            qoi.chmod(0o755)
            subprocess.run(
                ("git", "-C", str(root), "add", "."), check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            subprocess.run(
                ("git", "-C", str(root), "-c", "user.name=Harness Test",
                 "-c", "user.email=harness@example.invalid", "commit",
                 "-m", "freeze"),
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )

            raw_reference = inputs / ".." / reference.name
            driver_arguments = [
                "--executable", str(application),
                "--mesh", str(mesh),
                "--output-dir", str(root / "report"),
                "--methods", DG_RK3SSP,
                "--device", "cpu",
                "--final-time", "0.2",
                "--zhang-shu-vtk-points",
                "--qoi-postprocessor", str(qoi),
                f"--qoi-arg=--reference-profile={raw_reference}",
                "--qualification-mode",
                "--qoi-requirement=shock_position=1.0:0.0",
                "--time-program", str(time_program),
                "--source-root", str(root),
            ]
            arguments = _argument_parser().parse_args(driver_arguments)
            arguments._driver_argv = driver_arguments
            report = run_campaign(arguments)

            self.assertEqual(report["schema_version"], SCHEMA_VERSION)
            self.assertEqual(SCHEMA_VERSION, 3)
            self.assertTrue(report["controls"]["zhang_shu_vtk_points"])
            self.assertEqual(
                report["provenance"]["typed_application_controls"]
                ["zhang_shu_vtk_points"],
                {"enabled": True, "scope": "explicit-dg-and-dgsem-only"},
            )
            self.assertTrue(
                report["runs"][0]["provenance"]
                ["typed_application_controls"]
                ["zhang_shu_vtk_points"]
            )
            self.assertIn(
                "--zhang-shu-vtk-points", report["runs"][0]["command"]
            )
            report_inputs = report["provenance"]["qoi_input_files"]
            self.assertEqual(len(report_inputs), 1)
            self.assertEqual(
                report_inputs[0]["file"]["path"], str(reference.resolve())
            )
            self.assertEqual(
                report_inputs[0]["file"]["sha256"],
                capture_file_identity(reference, "reference")["sha256"],
            )
            self.assertEqual(
                report["runs"][0]["provenance"]["qoi_input_files"],
                report_inputs,
            )
            self.assertIn(
                f"--reference-profile={reference.resolve()}",
                report["runs"][0]["qoi_command"],
            )
            self.assertEqual(report["runs"][0]["quality"]["status"], "passed")

    def test_timeout_kills_sigterm_resistant_process_group_member(self) -> None:
        time_program = Path("/usr/bin/time")
        if not time_program.is_file():
            self.skipTest("GNU time is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child_pid_path = root / "child.pid"
            program = """
import os
from pathlib import Path
import signal
import sys
import time

child = os.fork()
if child == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(sys.argv[1]).write_text(str(os.getpid()), encoding="utf-8")
    while True:
        time.sleep(0.05)
while True:
    time.sleep(0.05)
"""
            child_pid: int | None = None
            try:
                result = run_timed_process(
                    (sys.executable, "-c", program, str(child_pid_path)),
                    time_program=str(time_program),
                    work_directory=root,
                    environment=os.environ,
                    timeout_seconds=0.5,
                    max_log_bytes=4096,
                    label="timeout-tree",
                )
                self.assertTrue(result.timed_out)
                child_pid = int(child_pid_path.read_text(encoding="utf-8"))
                deadline = time.monotonic() + 2.0
                while Path(f"/proc/{child_pid}").exists() and \
                        time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertFalse(Path(f"/proc/{child_pid}").exists())
            finally:
                if child_pid is None and child_pid_path.exists():
                    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
                if child_pid is not None and Path(f"/proc/{child_pid}").exists():
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_file_and_source_pins_detect_post_pin_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(
                ("git", "init", str(root)), check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            subprocess.run(
                ("git", "-C", str(root), "remote", "add", "origin",
                 "https://github.com/comp-physics/dg_igr_miniapp.git"),
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            source = root / "source.txt"
            source.write_text("frozen\n", encoding="utf-8")
            subprocess.run(
                ("git", "-C", str(root), "add", "source.txt"), check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            subprocess.run(
                ("git", "-C", str(root), "-c", "user.name=Harness Test",
                 "-c", "user.email=harness@example.invalid", "commit",
                 "-m", "freeze"),
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            source_pin = git_identity(
                root, expected_commit=None, expected_tree=None,
                allow_dirty=False,
            )
            file_pin = capture_file_identity(source, "synthetic source")
            verify_git_identity(source_pin)
            verify_file_identity(file_pin, "synthetic source")
            source.write_text("mutated\n", encoding="utf-8")
            with self.assertRaisesRegex(ShockArchitectureError, "changed"):
                verify_file_identity(file_pin, "synthetic source")
            with self.assertRaisesRegex(ShockArchitectureError, "changed"):
                verify_git_identity(source_pin)


if __name__ == "__main__":
    unittest.main()
