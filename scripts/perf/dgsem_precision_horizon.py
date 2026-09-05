#!/usr/bin/env python3
"""Bounded active-wave full-application precision experiment.

Build with build.sh first. The caller owns the GPU lease. This helper runs the
existing smooth-wave application, retains its ordinary output, and records
whole-process wall time. Compare saved fields with compare_vtk_pvd_fields.py;
successful termination and closure certification alone are not error estimates.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--order", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--refine", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--final-time", type=float, default=0.01)
    parser.add_argument("--linear-rtol", type=float, default=1e-10)
    parser.add_argument("--linear-solver", choices=("cg", "fgmres"), default="cg")
    parser.add_argument("--dense-inverse", choices=("fp64", "fp32", "tf32"))
    parser.add_argument("--two-level-inverse", choices=("dense", "cg"))
    parser.add_argument("--integrator", choices=("rk3ssp", "ssprk104"), default="rk3ssp")
    parser.add_argument("--sigma-space", choices=("L2", "H1"), default="L2")
    parser.add_argument("--sigma-cadence", choices=("every-stage", "extrapolated-every-step"),
                        default="every-stage")
    parser.add_argument("--extrapolation-degree", type=int, choices=(1, 2, 3))
    parser.add_argument("--volume-dissipation", choices=("none", "llf"), default="none")
    parser.add_argument("--positivity-limiter", choices=("none", "zhangshu"), default="none")
    parser.add_argument("--c-alpha", type=float, default=1.0)
    parser.add_argument("--sigma-basis", choices=("lgl",))
    parser.add_argument("--fused-residual", action="store_true")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    for value in (args.dt, args.final_time, args.linear_rtol, args.timeout, args.c_alpha):
        if not math.isfinite(value) or value <= 0:
            parser.error("time, tolerance, and timeout must be finite and positive")
    if args.dense_inverse in ("fp32", "tf32") and args.linear_solver != "fgmres":
        parser.error("mixed precision requires --linear-solver fgmres")
    if args.two_level_inverse and (args.dense_inverse or args.linear_solver != "fgmres"):
        parser.error("two-level inverse requires FGMRES and excludes the global dense inverse")
    if args.extrapolation_degree and args.sigma_cadence != "extrapolated-every-step":
        parser.error("extrapolation degree requires extrapolated-every-step cadence")
    source = Path(__file__).resolve().parents[2]
    executable = args.executable.resolve(strict=True)
    mesh = (args.mesh or source / "examples/dgsem_smooth_advection_periodic/"
            "smooth_advection_periodic_n6.mesh").resolve(strict=True)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    command = [
        str(executable), "--device", "cuda", "--mesh", str(mesh),
        "--active-igr-wave", "--discretization", "dgsem", "--dg-backend", "dgsem",
        "--order", str(args.order), "--serial-refine", str(args.refine),
        "--parallel-refine", "0", "--regularization", "igr", "--c-alpha", str(args.c_alpha),
        "--igr-alpha-length", "local", "--dgsem-stab", "none",
        "--dgsem-volume-flux", "central", "--dgsem-volume-dissipation", args.volume_dissipation,
        "--dgsem-surface-flux", "rusanov", "--time-integrator", args.integrator,
        "--time-step", format(args.dt, ".17g"), "--cfl-number", "0",
        "--t-final", format(args.final_time, ".17g"), "--igr-space", args.sigma_space,
        "--igr-grad", "BrokenL2", "--igr-solve-mode", "krylov",
        "--igr-ip-kind", "sipg", "--igr-exact-dg-face",
        "--igr-sigma-cadence", args.sigma_cadence, "--igr-stage-rtol-scale", "1",
        "--igr-linear-solver", args.linear_solver, "--igr-preconditioner", "chebyshev",
        "--igr-lin-rtol", format(args.linear_rtol, ".17g"),
        "--igr-lin-atol", "0", "--igr-lin-maxit", "2000",
        "--igr-nonconvergence-policy", "error", "--igr-lin-print", "0",
        "--no-igr-sigma-cfl", "--no-igr-sigma-rusanov", "--no-igr-invariant-domain-flux",
        "--positivity-limiter", args.positivity_limiter, "--nan-scan", "--admissibility-scan",
        "--analysis-steps", "0", "--visualization-steps", "1000000",
        "--vtk_save", "--time-step-save", format(args.final_time, ".17g"),
        "--no-glvis", "--no-save-state", "--no-vtk-h1-diagnostics",
        "--no-vtk-l2-diagnostics", "--no-vtk-mpi-rank-field", "--no-vtk-zhang-shu-diagnostics",
    ]
    environment = os.environ.copy()
    for key in ("DG_IGR_IGR_DENSE_INVERSE", "DG_IGR_IGR_TWO_LEVEL_INVERSE",
                "DG_IGR_IGR_TWO_LEVEL_COARSE", "DG_IGR_IGR_EXTRAPOLATION_DEGREE",
                "DG_IGR_IGR_SIGMA_BASIS", "DG_IGR_DGSEM_FUSED_RESIDUAL"):
        environment.pop(key, None)
    environment.update({
        "OMP_NUM_THREADS": "1", "DG_IGR_DGSEM_PRODUCTION_STRICT": "1",
        "DG_IGR_DGSEM_TIMING": "1", "DG_IGR_IGR_PARTIAL_ASSEMBLY": "1",
        "DG_IGR_IGR_DEVICE_QF": "1", "DG_IGR_VTK_PREFIX_PATH": str(out / "vtk"),
    })
    if args.dense_inverse:
        environment["DG_IGR_IGR_DENSE_INVERSE"] = args.dense_inverse
    if args.two_level_inverse:
        environment["DG_IGR_IGR_TWO_LEVEL_INVERSE"] = "fp64"
        environment["DG_IGR_IGR_TWO_LEVEL_COARSE"] = args.two_level_inverse
    if args.extrapolation_degree:
        environment["DG_IGR_IGR_EXTRAPOLATION_DEGREE"] = str(args.extrapolation_degree)
    if args.sigma_basis:
        environment["DG_IGR_IGR_SIGMA_BASIS"] = args.sigma_basis
    if args.fused_residual:
        environment["DG_IGR_DGSEM_FUSED_RESIDUAL"] = "1"
    record = {
        "command": command,
        "environment": {k: v for k, v in environment.items()
                        if k.startswith(("DG_IGR_", "CUDA_", "OMP_"))},
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=source, text=True),
    }
    start = time.perf_counter()
    with (out / "stdout.txt").open("w") as stdout, (out / "stderr.txt").open("w") as stderr:
        process = subprocess.Popen(command, cwd=source, env=environment,
                                   stdout=stdout, stderr=stderr)
        timed_out = threading.Event()
        def expire():
            timed_out.set()
            process.kill()
        timer = threading.Timer(args.timeout, expire)
        timer.daemon = True
        timer.start()
        try:
            record["returncode"] = process.wait()
        finally:
            timer.cancel()
        record["timeout"] = timed_out.is_set()
    record["wall_seconds"] = time.perf_counter() - start
    summaries = {}
    for line in (out / "stdout.txt").read_text().splitlines():
        if line.startswith(("[explicit-time]", "[igr-closure-certificate]",
                            "[igr-linear] event=run-summary", "[igr-dense-inverse]",
                            "[igr-two-level-inverse]", "[igr-sigma-basis]",
                            "[igr-sigma-predictor]", "[dgsem-fused-residual]",
                            "[dgsem-op] event=run-summary")):
            fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
            summaries[line.split()[0] + ":" + fields.get("event", "setup")] = fields
    record["summaries"] = summaries
    final = summaries.get("[explicit-time]:final-status", {})
    certificate = summaries.get("[igr-closure-certificate]:accepted-state", {})
    dense_setup = summaries.get("[igr-dense-inverse]:setup", {})
    linear = summaries.get("[igr-linear]:run-summary", {})
    two_level_setup = summaries.get("[igr-two-level-inverse]:setup", {})
    record["requested_preconditioner_executed"] = (
        dense_setup.get("precision") == args.dense_inverse and
        linear.get("last_precond") == "dense-inverse-" + args.dense_inverse and
        linear.get("last_solver") == args.linear_solver
        if args.dense_inverse else bool(two_level_setup) and
        linear.get("last_precond") == "two-level-inverse-fp64" and
        linear.get("last_solver") == "fgmres" if args.two_level_inverse else not dense_setup and
        not linear.get("last_precond", "").startswith("dense-inverse-"))
    record["requested_execution_executed"] = (
        (not args.sigma_basis or bool(summaries.get("[igr-sigma-basis]:setup"))) and
        (not args.fused_residual or bool(summaries.get("[dgsem-fused-residual]:setup"))) and
        (not args.extrapolation_degree or
         summaries.get("[igr-sigma-predictor]:setup", {}).get("degree") ==
         str(args.extrapolation_degree)))
    record["completed_and_certified"] = (
        record["returncode"] == 0 and not record["timeout"] and
        record["requested_preconditioner_executed"] and record["requested_execution_executed"] and
        final.get("success") == "1" and
        final.get("termination_reason") == "target-reached" and
        abs(float(final.get("final_time", "nan")) - args.final_time) < 1e-12 and
        abs(float(certificate.get("time", "nan")) - args.final_time) < 1e-12 and
        certificate.get("certified") == "1" and certificate.get("run_success") == "1")
    (out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"output_dir": str(out), "wall_seconds": record["wall_seconds"],
                      "completed_and_certified": record["completed_and_certified"]}))
    return 0 if record["completed_and_certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
