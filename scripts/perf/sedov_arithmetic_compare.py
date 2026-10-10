#!/usr/bin/env python3
"""Bounded Sedov arithmetic ablation with the same FP64 trace solver.

Volume contractions select FP64, FP32 or TF32 arithmetic; the nonlinear residual
can independently retain FP64 contraction with a mixed-precision Jacobian. Local condensation
independently selects FP64 WMMA or ordinary FP64 CUDA. Coefficients, condensation,
faces, global trace solve, time integration and persistent storage remain FP64.
In particular, simt-fp32 is not an end-to-end FP32 solver.

Uses the existing Sedov VTK decoder and conservative Q3 reconstruction. Example:
  python sedov_arithmetic_compare.py --binary PATH --source CHECKOUT \
    --mode simt-fp32 --name pilot-simt-fp32 --horizon .005 \
    --reference tests/research/results/sedov_arithmetic_20260927/pilot-fp64/run.json
Offline analysis (does not acquire/use the GPU):
  python sedov_arithmetic_compare.py --analyze-run PATH/run.json \
    --name offline-check --reference PATH/reference/run.json
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import threading
import time

REPO = Path(__file__).resolve().parents[2]
PREVIOUS = REPO / "tests/research/results/cigmo_sedov_20260927"
OUTPUT = REPO / "tests/research/results/sedov_arithmetic_20260927"
MODES = ("fp64-wmma", "simt-fp64", "simt-fp32", "tf32", "tf32x3")
SCOPE = ("Volume Jacobian arithmetic, nonlinear residual contraction precision and FP64 condensation execution are independently controlled; "
         "coefficients, condensation, faces, trace solve, time integration and persistent storage remain FP64. "
         "No mode in this study is an end-to-end FP32 solver.")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def command(argv, cwd=None):
    return subprocess.check_output(argv, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def gpu():
    return command(["nvidia-smi", "--query-gpu=uuid,name,driver_version,utilization.gpu,"
                    "memory.used,temperature.gpu,clocks.sm,power.draw", "--format=csv"])


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def no_compilation():
    active = {line.strip() for line in command(["ps", "-eo", "comm="]).splitlines()}
    forbidden = {"nvcc", "cicc", "cudafe++", "cc1plus", "cc1", "g++", "clang++",
                 "ninja", "make", "cmake", "ptxas", "nvlink", "fatbinary"}
    if active & forbidden:
        raise RuntimeError(f"Compilation active; timing refused: {sorted(active & forbidden)}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze(record, references):
    """Reuse the existing order-aware reconstruction; derivatives are cell-local."""
    import numpy as np

    evaluator_path = PREVIOUS / "analyze_sedov.py"
    evaluator = module(evaluator_path, "sedov_existing_analysis")
    decoder, decoder_path = evaluator.decoder_module()
    pvd = Path(record["pvd"])
    horizon = float(record.get("horizon", record["command"][record["command"].index("--t-final") + 1]))
    raw, files = evaluator.load_frame(pvd, decoder, horizon)
    polynomial_order = int(record.get("order", 3))
    grid, reconstruction = evaluator.Grid.reconstruct(raw, order=polynomial_order)
    initial_raw, initial_files = evaluator.load_frame(pvd, decoder, 0.0)
    initial, _ = evaluator.Grid.reconstruct(initial_raw, order=polynomial_order)
    diagnostics = evaluator.integral_stats(grid, initial, order=8)
    require(diagnostics["min_density_dense17"] > 0 and diagnostics["min_pressure_dense17"] > 0,
            "Nonpositive conservative-derived density/pressure at dense cell samples")
    xy, weight, ids = grid.quadrature(8)
    values = grid.evaluate(xy, ids, gradients=True)
    require(all(np.isfinite(v).all() for v in values.values()), "Nonfinite reconstructed diagnostics")
    report = {"scope": SCOPE, "horizon": horizon, "pvd": str(pvd),
              "reader": {"path": str(decoder_path), "sha256": sha(decoder_path)},
              "evaluator": {"path": str(evaluator_path), "sha256": sha(evaluator_path)},
              "reconstruction": reconstruction, "integrals": diagnostics,
              "sources": {**files, **initial_files}, "references": [],
              "gradient_note": f"Analytic cell-local velocity derivatives reconstructed from Q{polynomial_order} conserved fields; no DG face lift or jump distribution is included.",
              "sampled_ranges": {k: {"min": float(v.min()), "max": float(v.max())}
                                 for k, v in values.items()}}
    for reference_path in references:
        reference_path = reference_path.resolve()
        reference = json.loads(reference_path.read_text())
        require(reference.get("accepted", False), f"Reference was not accepted: {reference_path}")
        reference_raw, reference_files = evaluator.load_frame(Path(reference["pvd"]), decoder, horizon)
        reference_grid, reference_reconstruction = evaluator.Grid.reconstruct(
            reference_raw, order=int(reference.get("order", 3)))
        difference = evaluator.difference(grid, reference_grid, order=8)
        integration = grid if grid.nx >= reference_grid.nx else reference_grid
        qxy, qw, _ = integration.quadrature(8)
        gv = grid.evaluate(qxy, gradients=True)
        rv = reference_grid.evaluate(qxy, gradients=True)
        for field in ("div_velocity", "grad_velocity_x_x", "grad_velocity_x_y",
                      "grad_velocity_y_x", "grad_velocity_y_y"):
            delta = gv[field] - rv[field]
            square, refsquare = float(qw @ (delta * delta)), float(qw @ (rv[field] * rv[field]))
            difference[field] = {"l2": math.sqrt(square),
                "rms": math.sqrt(square / float(qw.sum())),
                "relative_l2": math.sqrt(square / refsquare) if refsquare else None,
                "max_abs_at_quadrature": float(abs(delta).max())}
        report["references"].append({"run_json": str(reference_path), "sha256": sha(reference_path),
            "name": reference.get("name", reference_path.parent.name), "differences": difference,
            "reconstruction": reference_reconstruction, "source_files": reference_files,
            "interpretation": "Difference from numerical reference, not continuum error."})
    return report


def run(args, out):
    binary, source = args.binary.resolve(), args.source.resolve()
    mesh = source / "examples/sedov_taylor_blast/sedov_taylor.msh"
    require(binary.is_file() and mesh.is_file(), "Missing binary or Sedov mesh")
    diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=source)
    (out / "source.diff").write_bytes(diff)
    argv = [str(binary), "--mesh", str(mesh), "--discretization", "hdg", "--device", "cuda",
        "--hdg-cuda-engine", "tensor", "--order", "3", "--regularization", "igr",
        "--c-alpha", "1", "--hdg-igr-beta", "1", "--igr-alpha-length", "average",
        "--igr-sigma-cadence", "every-stage", "--serial-refine", "0", "--parallel-refine", "0",
        "--time-integrator", "rodas3", "--time-step", "-0.0025", "--cfl-number", "0",
        "--time-atol", "1e-8", "--time-rtol", "1e-5", "--dt-min", "1e-8",
        "--t-final", str(args.horizon), "--positivity-limiter", "none", "--visualization-steps",
        "1000000", "--vtk_save", "--time-step-save", str(min(.025, args.horizon)),
        "--no-glvis", "--no-save-state"]
    if not args.omit_conservation_option:
        argv.extend(["--hdg-conservation-rtol", str(args.conservation_rtol),
                     "--hdg-nonlinear-rtol", str(args.nonlinear_rtol)])
    env = {k: v for k, v in os.environ.items() if not k.startswith("DG_IGR_")}
    env.update(OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="0",
               DG_IGR_VTK_PREFIX_PATH=str(out / "vtk"),
               DG_IGR_HDG_VOLUME_ARITHMETIC=args.mode,
               DG_IGR_HDG_VOLUME_RESIDUAL=args.residual,
               DG_IGR_HDG_CONDENSATION_ARITHMETIC=args.condensation)
    executed = argv
    if args.ncu:
        executed = [str(args.ncu.resolve()), "--target-processes", "application-only",
                    "--kernel-name-base", "demangled", "--kernel-name", "regex:" + args.profile_kernel,
                    "--launch-count", "3", "--launch-skip", "0", "--metrics", args.profile_metrics,
                    "--export", str(out / "capture"), *argv]
    selected_env = {k: v for k, v in env.items() if k.startswith(("DG_IGR_", "OMP_", "CUDA_"))
                    or k in ("PATH", "LD_LIBRARY_PATH")}
    (out / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(source)) +
        "\n" + " ".join(f"{k}={shlex.quote(v)}" for k, v in selected_env.items()) +
        " " + shlex.join(executed) + "\n")
    record = {"name": args.name, "mode": args.mode, "residual": args.residual, "condensation": args.condensation, "scope": SCOPE,
        "engine": "tensor", "horizon": args.horizon, "rtol": 1e-5, "atol": 1e-8,
        "conservation_rtol": None if args.omit_conservation_option else args.conservation_rtol,
        "conservation_option_omitted": args.omit_conservation_option,
        "nonlinear_rtol": None if args.omit_conservation_option else args.nonlinear_rtol,
        "case_tolerance_options_omitted": args.omit_conservation_option,
        "linear_solver_tolerance_control": "Unchanged case default; no linear-solver tolerance override is passed.",
        "physical_alpha": 1 / 1024, "source_directory": str(source),
        "source_commit": command(["git", "rev-parse", "HEAD"], source),
        "source_tree": command(["git", "rev-parse", "HEAD^{tree}"], source),
        "source_tracked_status": command(["git", "status", "--porcelain", "--untracked-files=no"], source),
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(), "binary": str(binary),
        "binary_sha256": sha(binary), "mesh_sha256": sha(mesh), "command": argv,
        "executed_command": executed, "profiled": bool(args.ncu),
        "environment": selected_env, "runner_sha256": sha(__file__), "timeout_seconds": args.timeout,
        "status": "running", "accepted": False, "started_utc": utc(), "timed_out": False,
        "timing_scope": "Complete fresh process including setup, rejected work and VTK output; excludes post-run field analysis."}
    record["source_identity_note"] = "Source commit/tree/diff above describe the checkout at launch; only a supplied frozen build manifest binds compiled source to the binary."
    if args.build_manifest:
        manifest_path = args.build_manifest.resolve()
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        require({"binary_sha256", "source_commit", "source_tree", "source_diff_sha256"} <= manifest.keys(),
                "Build manifest must contain binary_sha256, source_commit, source_tree and source_diff_sha256")
        require(manifest["binary_sha256"] == record["binary_sha256"], "Frozen build manifest binary hash mismatch")
        (out / "build_manifest.json").write_bytes(manifest_bytes)
        record["build_manifest"] = {"original_path": str(manifest_path),
                                    "sha256": hashlib.sha256(manifest_bytes).hexdigest(), "contents": manifest}
        record["launch_checkout_matches_build"] = all(record[key] == manifest[key]
            for key in ("source_commit", "source_tree", "source_diff_sha256"))
    cache = binary.parents[2] / "CMakeCache.txt"
    if cache.exists():
        record["build_cache"] = {"path": str(cache), "sha256": sha(cache)}
    write(out / "run.json", record)
    try:
        with open("/tmp/dg-igr-gpu0.lock", "a") as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("GPU lease busy; no run started") from None
            no_compilation()
            record["gpu_before"] = gpu()
            with (out / "stdout.txt").open("w") as stdout, (out / "stderr.txt").open("w") as stderr:
                record["process_started_utc"] = utc()
                raw_start = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                perf_start = time.perf_counter_ns()
                process = subprocess.Popen(executed, cwd=source, env=env, stdout=stdout,
                                           stderr=stderr, start_new_session=True)
                def expire():
                    record["timed_out"] = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                timer = threading.Timer(args.timeout, expire)
                timer.start()
                try:
                    record["returncode"] = process.wait()
                finally:
                    record["perf_counter_seconds"] = (time.perf_counter_ns() - perf_start) * 1e-9
                    record["monotonic_raw_seconds"] = (time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - raw_start) * 1e-9
                    timer.cancel()
                    record["process_finished_utc"] = utc()
            record["gpu_after"] = gpu()
        logs = (out / "stdout.txt").read_text()
        record["summaries"] = {line.split()[0] + ":" + dict(re.findall(r"(\w+)=(\S+)", line)).get("event", "setup"):
                               dict(re.findall(r"(\w+)=(\S+)", line)) for line in logs.splitlines() if line.startswith("[")}
        record["rejected_log_lines"] = [s for s in logs.splitlines() if "event=rejected" in s or "event=reject" in s]
        record["log_sha256"] = {name: sha(out / name) for name in ("stdout.txt", "stderr.txt")}
        if record["returncode"] != 0 or record["timed_out"]:
            record["failure_log_first"] = {name: (out / name).read_text().splitlines()[:12]
                                           for name in ("stdout.txt", "stderr.txt")}
            record["failure_log_tail"] = {name: (out / name).read_text().splitlines()[-12:]
                                          for name in ("stdout.txt", "stderr.txt")}
            failure_events = [{"stream": name, "line": line} for name in ("stdout.txt", "stderr.txt")
                              for line in (out / name).read_text().splitlines()
                              if re.search(r"fail|error|reject|nan|nonfinite|non-finite", line, re.IGNORECASE)]
            record["failure_events"] = {"count": len(failure_events), "first": failure_events[:12],
                                        "last": failure_events[-12:]}
        require(record["binary_sha256"] == sha(binary), "Binary changed during run")
        require(record["returncode"] == 0 and not record["timed_out"], "Process failed or timed out")
        summary = record["summaries"].get("[hdg-main]:setup", {})
        require(summary.get("status") == "Success" and abs(float(summary.get("time", -1)) - args.horizon) < 1e-12,
                "Solver did not reach requested horizon")
        pvds = list(out.rglob("*.pvd"))
        require(len(pvds) == 1, "Missing or ambiguous PVD")
        record["pvd"] = str(pvds[0])
        reader = module(REPO / "tests/scripts/compare_vtk_pvd_fields.py", "sedov_existing_reader")
        frames = reader.parse_pvd(pvds[0])
        require(abs(max(f["timestep"] for f in frames) - args.horizon) < 1e-12, "Saved fields did not reach horizon")
        record["frames"] = []
        for frame in frames:
            arrays = reader.load_point_arrays(Path(frame["file"]), None)
            require({"density", "pressure", "energy", "momentum", "velocity", "sigma"} <= arrays.keys(), "Missing physical fields")
            require(all(len(v) and all(math.isfinite(x) for x in v) for v in arrays.values()), "Nonfinite or empty saved fields")
            require(min(arrays["density"]) > 0 and min(arrays["pressure"]) > 0, "Nonpositive saved density/pressure")
            record["frames"].append({"time": frame["timestep"], "file": frame["file"],
                                     "fields": {k: reader.array_stats(v) for k, v in arrays.items()}})
        analysis = analyze(record, args.reference)
        write(out / "analysis.json", analysis)
        record.update(accepted=True, status="complete", analysis=str(out / "analysis.json"))
    except Exception as error:
        record.update(status="failed", validation_error=str(error))
    finally:
        if args.ncu:
            record["profile_files"] = {str(p): sha(p) for p in out.glob("*.ncu-rep")}
            record["profile_note"] = ("Profiled process times are not performance baselines. Captured kernel data may be usable "
                                      "if the subsequent application fails; that does not count as reaching the physical horizon.")
        record["finished_utc"] = utc()
        write(out / "run.json", record)
    brief = {k: record.get(k) for k in ("name", "mode", "residual", "condensation", "status", "monotonic_raw_seconds",
                                       "perf_counter_seconds", "validation_error")}
    brief["summary"] = record.get("summaries", {}).get("[hdg-main]:setup")
    print(json.dumps(brief), flush=True)
    return 0 if record["accepted"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--residual", choices=("matched", "fp64"), default="matched",
                        help="Volume nonlinear residual uses selected arithmetic or FP64; Jacobian keeps --mode")
    parser.add_argument("--condensation", choices=("fp64-wmma", "simt-fp64"), default="fp64-wmma")
    parser.add_argument("--build-manifest", type=Path,
                        help="Frozen JSON with binary_sha256, source_commit, source_tree and source_diff_sha256")
    parser.add_argument("--name", required=True)
    parser.add_argument("--horizon", type=float, default=.005)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--conservation-rtol", type=float, default=1e-10,
                        help="Separate HDG conservation acceptance tolerance; time/linear solver tolerances stay unchanged")
    parser.add_argument("--nonlinear-rtol", type=float, default=1e-10,
                        help="Separate HDG nonlinear tolerance; time/linear solver tolerances stay unchanged")
    parser.add_argument("--omit-conservation-option", action="store_true",
                        help="Compatibility with old binaries; omit both conservation and nonlinear case tolerance flags")
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--analyze-run", type=Path)
    parser.add_argument("--ncu", type=Path, help="Optional profiler; process wall times then include instrumentation")
    parser.add_argument("--profile-kernel", default=r".*Volume(Contraction|Coefficient)Kernel.*")
    parser.add_argument("--profile-metrics", default="gpu__time_duration.sum,sm__inst_executed_pipe_fp64_op_dmma.sum,sm__inst_executed_pipe_tensor_subpipe_hmma_op_hmma.sum")
    args = parser.parse_args()
    require(re.fullmatch(r"[A-Za-z0-9_-]+", args.name), "Unsafe run name")
    require(args.horizon > 0 and args.timeout > 0, "Horizon/timeout must be positive")
    require(math.isfinite(args.conservation_rtol) and args.conservation_rtol > 0,
            "Conservation tolerance must be finite and positive")
    require(math.isfinite(args.nonlinear_rtol) and args.nonlinear_rtol > 0,
            "Nonlinear tolerance must be finite and positive")
    if not args.analyze_run:
        require(args.binary and args.source and args.mode, "Run requires --binary, --source and --mode")
    out = OUTPUT / args.name
    out.mkdir(parents=True, exist_ok=False)
    if args.analyze_run:
        record = json.loads(args.analyze_run.read_text())
        require(record.get("accepted", False), "Requested run was not accepted")
        write(out / "analysis.json", analyze(record, args.reference))
        print(out / "analysis.json")
        return 0
    return run(args, out)


if __name__ == "__main__":
    raise SystemExit(main())
