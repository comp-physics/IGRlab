#!/usr/bin/env python3
"""Run a focused CPU/CUDA L2 IGR PVD drift check."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import compare_vtk_pvd_fields as vtk_compare


DATASET_RE = re.compile(r"^VTK dataset:\s*(\S+)\s*$", re.MULTILINE)
RUN_SUMMARY_RE = re.compile(r"^\[igr-linear\]\s+event=run-summary\s+(.*)$",
                            re.MULTILINE)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_env(entries: Iterable[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"environment entry must be KEY=VALUE, got {entry!r}")
        key, value = entry.split("=", 1)
        if not key:
            raise ValueError(f"environment entry has empty key: {entry!r}")
        env[key] = value
    return env


def parse_limits(entries: Iterable[str]) -> Dict[str, float]:
    limits: Dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"limit must be ARRAY=VALUE, got {entry!r}")
        name, value = entry.split("=", 1)
        limits[name] = float(value)
    return limits


def sci(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.12e}"
    except (TypeError, ValueError):
        return str(value)


def plain(value) -> str:
    return "n/a" if value is None else str(value)


def role_value(args: argparse.Namespace, role: str, name: str):
    specific = getattr(args, f"{role}_{name}", None)
    if specific is not None:
        return specific
    return getattr(args, name)


def app_command_for_role(args: argparse.Namespace,
                         role: str,
                         device: str) -> List[str]:
    command = [
        str(Path(args.exe).resolve()),
        "--device",
        device,
        "--dg-backend",
        "dg",
        "-o",
        str(args.order),
        "-rs",
        str(args.serial_ref),
        "-rp",
        str(args.parallel_ref),
        "-is",
        str(args.igr_space),
        "-ig",
        str(args.igr_grad),
        "-ca",
        str(args.c_alpha),
        "-tf",
        str(args.tf),
        "-dt",
        str(args.dt),
        "-c",
        str(args.cfl),
        "--igr-sigma-cadence",
        "every-stage",
        "--igr-stage-rtol-scale",
        str(role_value(args, role, "igr_stage_rtol_scale")),
        "--igr-lin-rtol",
        str(role_value(args, role, "igr_lin_rtol")),
        "-vtk",
        "-dt_save",
        str(args.dt_save),
        "-no-glvis",
        "-vs",
        "1000000",
        "-iprint",
        "0",
    ]
    if args.nan_scan:
        command.append("-nan-scan")
    if args.admissibility_scan:
        command.append("-adm-scan")
    lin_atol = role_value(args, role, "igr_lin_atol")
    if lin_atol is not None:
        command.extend(["--igr-lin-atol", str(lin_atol)])
    lin_maxit = role_value(args, role, "igr_lin_maxit")
    if lin_maxit is not None:
        command.extend(["--igr-lin-maxit", str(lin_maxit)])
    linear_solver = role_value(args, role, "igr_linear_solver")
    if linear_solver is not None:
        command.extend(["--igr-linear-solver", str(linear_solver)])
    preconditioner = role_value(args, role, "igr_preconditioner")
    if preconditioner is not None:
        command.extend(["--igr-preconditioner", str(preconditioner)])
    exact_face = args.igr_exact_dg_face
    if role == "cpu":
        exact_face = exact_face or args.cpu_igr_exact_dg_face
    elif role == "cuda":
        exact_face = exact_face or args.cuda_igr_exact_dg_face
    if exact_face:
        command.append("--igr-exact-dg-face")
    if args.disable_vtk_diagnostics:
        command.extend([
            "--no-vtk-h1-diagnostics",
            "--no-vtk-l2-diagnostics",
            "--no-vtk-mpi-rank-field",
            "--no-vtk-zhang-shu-diagnostics",
        ])
    return command


def command_for_role(args: argparse.Namespace, role: str, device: str) -> List[str]:
    app_command = app_command_for_role(args, role, device)
    if args.mpi_ranks <= 1:
        return app_command
    command = [
        str(args.mpiexec),
        str(args.mpi_numproc_flag),
        str(args.mpi_ranks),
    ]
    command.extend(args.mpi_preflag)
    command.extend(app_command)
    command.extend(args.mpi_postflag)
    return command


def expected_dataset(args: argparse.Namespace) -> str:
    total_ref = args.serial_ref + args.parallel_ref
    c_alpha = float(args.c_alpha)
    if c_alpha.is_integer():
        c_alpha_text = str(int(c_alpha))
    else:
        c_alpha_text = str(args.c_alpha)
    return (
        "supersonic_cylinder_w_igr"
        f"_p_{args.order}_ref_{total_ref}_is_{args.igr_space}"
        f"_ig_{args.igr_grad}_ca_{c_alpha_text}"
    )


def run_role(args: argparse.Namespace,
             role: str,
             device: str,
             extra_env: Dict[str, str]) -> Path:
    root = repo_root()
    dataset = expected_dataset(args)
    generated_dir = root / "out" / "supersonic_cylinder_w_igr" / dataset
    shutil.rmtree(generated_dir, ignore_errors=True)

    role_dir = Path(args.out_dir) / role
    shutil.rmtree(role_dir, ignore_errors=True)
    role_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(extra_env)
    env.setdefault("OMP_NUM_THREADS", "1")
    command = command_for_role(args, role, device)
    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = Path(args.out_dir) / f"{role}.log"
    log_path.write_text(
        f"[cpu-cuda-pvd-drift] role={role} command={shlex.join(command)}\n" +
        completed.stdout,
        encoding="utf-8")
    if completed.returncode != 0:
        print(completed.stdout)
        raise RuntimeError(f"{role} run failed with exit code {completed.returncode}")

    match = DATASET_RE.search(completed.stdout)
    if match:
        dataset = match.group(1)
        generated_dir = root / "out" / "supersonic_cylinder_w_igr" / dataset
    pvd = generated_dir / f"{dataset}.pvd"
    if not pvd.exists():
        raise RuntimeError(f"{role} run did not produce expected PVD: {pvd}")

    copied_dir = role_dir / dataset
    shutil.copytree(generated_dir, copied_dir, dirs_exist_ok=True)
    return copied_dir / f"{dataset}.pvd"


def load_run_metadata(pvd: Path) -> Dict:
    metadata_path = pvd.parent / "run_metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(f"missing run metadata: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def timing_payload_for_role(role: str,
                            pvd: Path,
                            log_path: Path) -> Dict:
    metadata = load_run_metadata(pvd)
    parsed_log_summary = parse_run_summary(log_path)
    payload = {
        "role": role,
        "pvd": str(pvd),
        "metadata": str(pvd.parent / "run_metadata.json"),
        "runtime": metadata.get("runtime", {}),
        "performance": metadata.get("performance", {}),
        "igr_linear": metadata.get("igr_linear", {}),
        "command": metadata.get("command", {}),
        "log_igr_summary": parsed_log_summary,
    }
    perf = payload["performance"]
    igr = payload["igr_linear"]
    print(
        "[timing]"
        f" role={role}"
        f" device={payload['runtime'].get('device_backend', 'n/a')}"
        f" ranks={perf.get('ranks', payload['runtime'].get('mpi_ranks', 'n/a'))}"
        f" steps={perf.get('steps', 'n/a')}"
        f" elapsed_s={sci(perf.get('elapsed'))}"
        f" avg_step_per_dof_local={sci(perf.get('avg_step_per_dof_local'))}"
        f" local_dofs={perf.get('local_dofs', 'n/a')}"
        f" global_dofs={perf.get('global_dofs', 'n/a')}"
        f" igr_calls={igr.get('calls', parsed_log_summary.get('calls', 'n/a'))}"
        f" igr_converged_calls="
        f"{igr.get('converged_calls', parsed_log_summary.get('converged_calls', 'n/a'))}"
        f" avg_iters={sci(igr.get('avg_iters', parsed_log_summary.get('avg_iters')))}"
        f" total_solve_s={sci(igr.get('total_solve_s'))}"
        f" total_igr_s={sci(igr.get('total_igr_work_s'))}"
        f" total_cheb_diag_s={sci(igr.get('total_cheb_diag_s'))}"
        f" total_cheb_setup_s={sci(igr.get('total_cheb_setup_s'))}"
        f" total_cheb_eigen_reuses="
        f"{plain(igr.get('total_cheb_eigen_reuses'))}"
        f" total_cheb_smoother_reuses="
        f"{plain(igr.get('total_cheb_smoother_reuses'))}"
        f" total_cheb_smoother_builds="
        f"{plain(igr.get('total_cheb_smoother_builds'))}"
        f" total_krylov_operator_apply_s="
        f"{sci(igr.get('total_krylov_operator_apply_s'))}"
        f" total_krylov_preconditioner_apply_s="
        f"{sci(igr.get('total_krylov_preconditioner_apply_s'))}"
        f" total_krylov_preconditioner_operator_apply_s="
        f"{sci(igr.get('total_krylov_preconditioner_operator_apply_s'))}"
        f" total_krylov_operator_apply_count="
        f"{plain(igr.get('total_krylov_operator_apply_count'))}"
        f" total_krylov_preconditioner_apply_count="
        f"{plain(igr.get('total_krylov_preconditioner_apply_count'))}"
        f" total_krylov_preconditioner_operator_apply_count="
        f"{plain(igr.get('total_krylov_preconditioner_operator_apply_count'))}"
        f" last_solver={igr.get('last_solver', parsed_log_summary.get('last_solver', 'n/a'))}"
        f" last_precond={igr.get('last_precond', parsed_log_summary.get('last_precond', 'n/a'))}"
    )
    return payload


def write_timing_summary(args: argparse.Namespace,
                         cpu_pvd: Path,
                         cuda_pvd: Path) -> None:
    out_dir = Path(args.out_dir)
    roles = [
        timing_payload_for_role("cpu", cpu_pvd, out_dir / "cpu.log"),
        timing_payload_for_role("cuda", cuda_pvd, out_dir / "cuda.log"),
    ]
    payload = {
        "schema_version": 1,
        "benchmark": "cpu_cuda_l2_timing_summary",
        "mpi_ranks": args.mpi_ranks,
        "roles": roles,
    }
    summary_json = out_dir / "cpu_cuda_timing_summary.json"
    summary_md = out_dir / "cpu_cuda_timing_summary.md"
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")

    headers = [
        "role",
        "device",
        "ranks",
        "steps",
        "elapsed_s",
        "avg_step_per_dof_local",
        "local/global dofs",
        "IGR calls",
        "avg iters",
        "IGR solve/total_s",
        "Cheb diag/setup/eig/sm/builds",
        "Krylov op/pc/pc-op time_s",
        "Krylov op/pc/pc-op applies",
        "solver/precond",
        "direct path",
        "exact face",
    ]
    lines = [
        "# CPU/CUDA Timing Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in roles:
        runtime = row["runtime"]
        perf = row["performance"]
        igr = row["igr_linear"]
        direct = igr.get("direct_identity_system", {})
        exact = igr.get("exact_face", {})
        lines.append(
            "| "
            + " | ".join([
                row["role"],
                plain(runtime.get("device_backend")),
                plain(perf.get("ranks", runtime.get("mpi_ranks"))),
                plain(perf.get("steps")),
                sci(perf.get("elapsed")),
                sci(perf.get("avg_step_per_dof_local")),
                f"{plain(perf.get('local_dofs'))} / "
                f"{plain(perf.get('global_dofs'))}",
                f"{plain(igr.get('calls'))} / "
                f"{plain(igr.get('converged_calls'))}",
                sci(igr.get("avg_iters")),
                f"{sci(igr.get('total_solve_s'))} / "
                f"{sci(igr.get('total_igr_work_s'))}",
                f"{sci(igr.get('total_cheb_diag_s'))} / "
                f"{sci(igr.get('total_cheb_setup_s'))} / "
                f"{plain(igr.get('total_cheb_eigen_reuses'))} / "
                f"{plain(igr.get('total_cheb_smoother_reuses'))} / "
                f"{plain(igr.get('total_cheb_smoother_builds'))}",
                f"{sci(igr.get('total_krylov_operator_apply_s'))} / "
                f"{sci(igr.get('total_krylov_preconditioner_apply_s'))} / "
                f"{sci(igr.get('total_krylov_preconditioner_operator_apply_s'))}",
                f"{plain(igr.get('total_krylov_operator_apply_count'))} / "
                f"{plain(igr.get('total_krylov_preconditioner_apply_count'))} / "
                f"{plain(igr.get('total_krylov_preconditioner_operator_apply_count'))}",
                f"{plain(igr.get('last_solver'))} / "
                f"{plain(igr.get('last_precond'))}",
                f"{plain(direct.get('effective'))}; "
                f"{plain(direct.get('reason'))}; "
                f"uses {plain(direct.get('uses'))}",
                f"{plain(exact.get('effective'))}; "
                f"{plain(exact.get('reason'))}; "
                f"faces {plain(exact.get('local_faces'))}/"
                f"{plain(exact.get('shared_faces'))}; "
                f"uses {plain(exact.get('operator_uses'))}",
            ])
            + " |"
        )
    lines.extend(["", f"JSON: `{summary_json}`", ""])
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[timing] json={summary_json}")
    print(f"[timing] markdown={summary_md}")


def compare_and_check(args: argparse.Namespace, cpu_pvd: Path, cuda_pvd: Path) -> int:
    frames_a = vtk_compare.parse_pvd(cpu_pvd)
    frames_b = vtk_compare.parse_pvd(cuda_pvd)
    pairs = vtk_compare.common_timesteps(frames_a, frames_b, args.time_tol)
    if len(pairs) < args.min_matched_frames:
        raise RuntimeError(
            f"matched only {len(pairs)} frames, need {args.min_matched_frames}"
        )

    frame_results = [
        vtk_compare.compare_frames(a, b, args.arrays)
        for a, b in pairs
    ]
    aggregate = vtk_compare.aggregate_frame_diffs(frame_results)
    payload = {
        "schema_version": 1,
        "benchmark": "cpu_cuda_l2_pvd_drift",
        "cpu_pvd": str(cpu_pvd),
        "cuda_pvd": str(cuda_pvd),
        "mpi_ranks": args.mpi_ranks,
        "matched_frames": len(pairs),
        "arrays": args.arrays,
        "aggregate": aggregate,
        "worst": vtk_compare.worst_aggregate_metrics(aggregate),
    }

    out_dir = Path(args.out_dir)
    drift_json = out_dir / "cpu_cuda_pvd_drift.json"
    drift_md = out_dir / "cpu_cuda_pvd_drift.md"
    drift_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    vtk_compare.write_markdown(drift_md, {
        **payload,
        "a_pvd": str(cpu_pvd),
        "b_pvd": str(cuda_pvd),
        "a_frames": len(frames_a),
        "b_frames": len(frames_b),
        "frames": frame_results,
    })

    array_limits = parse_limits(args.array_rel_rms_limit)
    failures: List[str] = []
    for name in args.arrays:
        row = aggregate.get(name)
        if not isinstance(row, dict):
            continue
        limit = array_limits.get(name, args.default_rel_rms_limit)
        value = float(row.get("max_rel_rms", 0.0))
        print(f"[drift] {name} max_rel_rms={value:.12e} limit={limit:.12e}")
        if value > limit:
            failures.append(f"{name}: {value:.12e} > {limit:.12e}")

    print(f"[drift] matched_frames={len(pairs)}")
    print(f"[drift] json={drift_json}")
    print(f"[drift] markdown={drift_md}")
    if failures:
        raise RuntimeError("drift limits failed: " + "; ".join(failures))
    return 0


def check_required_log_patterns(log_path: Path, patterns: Iterable[str]) -> None:
    text = log_path.read_text(encoding="utf-8")
    missing = [pattern for pattern in patterns if re.search(pattern, text) is None]
    if missing:
        raise RuntimeError(
            f"{log_path} missing required pattern(s): " + ", ".join(missing)
        )


def parse_run_summary(log_path: Path) -> Dict[str, str]:
    text = log_path.read_text(encoding="utf-8")
    matches = RUN_SUMMARY_RE.findall(text)
    if not matches:
        raise RuntimeError(f"{log_path} missing igr-linear run-summary")
    summary: Dict[str, str] = {}
    for token in matches[-1].split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        summary[key] = value
    return summary


def check_igr_converged(log_path: Path) -> None:
    summary = parse_run_summary(log_path)
    calls = int(summary.get("calls", "-1"))
    converged = int(summary.get("converged_calls", "-1"))
    if calls < 0 or converged < 0:
        raise RuntimeError(
            f"{log_path} run-summary missing calls/converged_calls"
        )
    print(
        "[igr-convergence]"
        f" {log_path.name} calls={calls} converged_calls={converged}"
        f" avg_iters={summary.get('avg_iters', 'n/a')}"
        f" last_solver={summary.get('last_solver', 'n/a')}"
        f" last_precond={summary.get('last_precond', 'n/a')}"
    )
    if calls != converged:
        raise RuntimeError(
            f"{log_path} IGR convergence failed: "
            f"calls={calls}, converged_calls={converged}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--mpiexec", default=os.environ.get("MPIEXEC_EXECUTABLE",
                                                            "mpirun"))
    parser.add_argument("--mpi-numproc-flag",
                        default=os.environ.get("MPIEXEC_NUMPROC_FLAG", "-n"))
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--serial-ref", type=int, default=0)
    parser.add_argument("--parallel-ref", type=int, default=0)
    parser.add_argument("--igr-space", default="L2")
    parser.add_argument("--igr-grad", default="L2")
    parser.add_argument("--tf", type=float, default=2.0e-4)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--dt-save", type=float, default=1.0e-4)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--c-alpha", type=float, default=1.0)
    parser.add_argument("--igr-stage-rtol-scale", type=float, default=10.0)
    parser.add_argument("--igr-lin-rtol", type=float, default=1.0e-8)
    parser.add_argument("--igr-lin-atol", type=float)
    parser.add_argument("--igr-lin-maxit", type=int)
    parser.add_argument("--igr-linear-solver")
    parser.add_argument("--igr-preconditioner")
    parser.add_argument("--igr-exact-dg-face", action="store_true")
    parser.add_argument(
        "--disable-vtk-diagnostics",
        action="store_true",
        help="Disable optional H1/L2/MPI-rank/Zhang-Shu VTK diagnostic fields.",
    )
    parser.add_argument("--cpu-igr-stage-rtol-scale", type=float)
    parser.add_argument("--cuda-igr-stage-rtol-scale", type=float)
    parser.add_argument("--cpu-igr-lin-rtol", type=float)
    parser.add_argument("--cuda-igr-lin-rtol", type=float)
    parser.add_argument("--cpu-igr-lin-atol", type=float)
    parser.add_argument("--cuda-igr-lin-atol", type=float)
    parser.add_argument("--cpu-igr-lin-maxit", type=int)
    parser.add_argument("--cuda-igr-lin-maxit", type=int)
    parser.add_argument("--cpu-igr-linear-solver")
    parser.add_argument("--cuda-igr-linear-solver")
    parser.add_argument("--cpu-igr-preconditioner")
    parser.add_argument("--cuda-igr-preconditioner")
    parser.add_argument("--cpu-igr-exact-dg-face", action="store_true")
    parser.add_argument("--cuda-igr-exact-dg-face", action="store_true")
    parser.add_argument("--nan-scan", action="store_true")
    parser.add_argument("--admissibility-scan", action="store_true")
    parser.add_argument("--cpu-env", action="append", default=[])
    parser.add_argument("--cuda-env", action="append", default=[])
    parser.add_argument("--require-cpu-log", action="append", default=[])
    parser.add_argument("--require-cuda-log", action="append", default=[])
    parser.add_argument("--require-cpu-igr-converged", action="store_true")
    parser.add_argument("--require-cuda-igr-converged", action="store_true")
    parser.add_argument("--default-rel-rms-limit", type=float, default=5.0e-6)
    parser.add_argument("--array-rel-rms-limit", action="append", default=[])
    parser.add_argument("--time-tol", type=float, default=1.0e-12)
    parser.add_argument("--min-matched-frames", type=int, default=3)
    parser.add_argument("--arrays", nargs="+", default=vtk_compare.DEFAULT_ARRAYS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cpu_env = parse_env(args.cpu_env)
    cuda_env = parse_env(args.cuda_env)
    cpu_pvd = run_role(args, "cpu", "cpu", cpu_env)
    cuda_pvd = run_role(args, "cuda", "cuda", cuda_env)
    check_required_log_patterns(out_dir / "cpu.log", args.require_cpu_log)
    check_required_log_patterns(out_dir / "cuda.log", args.require_cuda_log)
    if args.require_cpu_igr_converged:
        check_igr_converged(out_dir / "cpu.log")
    if args.require_cuda_igr_converged:
        check_igr_converged(out_dir / "cuda.log")
    write_timing_summary(args, cpu_pvd, cuda_pvd)
    return compare_and_check(args, cpu_pvd, cuda_pvd)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"cpu-cuda-pvd-drift: ERROR: {exc}", file=sys.stderr)
        raise
