#!/usr/bin/env python3
"""Compare CUDA exact-face runs with direct identity system off and on."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import compare_vtk_pvd_fields as vtk_compare


DATASET_RE = re.compile(r"^VTK dataset:\s*(\S+)\s*$", re.MULTILINE)


def sci(value) -> str:
    try:
        return f"{float(value):.12e}"
    except (TypeError, ValueError):
        return str(value)


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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def expected_dataset(args: argparse.Namespace) -> str:
    total_ref = args.serial_ref + args.parallel_ref
    c_alpha = float(args.c_alpha)
    c_alpha_text = str(int(c_alpha)) if c_alpha.is_integer() else str(args.c_alpha)
    return (
        "supersonic_cylinder_w_igr"
        f"_p_{args.order}_ref_{total_ref}_is_L2_ig_L2_ca_{c_alpha_text}"
    )


def command(args: argparse.Namespace) -> List[str]:
    cmd = [
        str(Path(args.exe).resolve()),
        "--device",
        "cuda",
        "--dg-backend",
        "dg",
        "-o",
        str(args.order),
        "-rs",
        str(args.serial_ref),
        "-rp",
        str(args.parallel_ref),
        "-is",
        "L2",
        "-ig",
        "L2",
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
        str(args.igr_stage_rtol_scale),
        "--igr-lin-rtol",
        str(args.igr_lin_rtol),
        "-vtk",
        "-dt_save",
        str(args.dt_save),
        "-no-glvis",
        "-vs",
        "1000000",
        "-iprint",
        "0",
        "--igr-exact-dg-face",
    ]
    if args.igr_lin_atol is not None:
        cmd.extend(["--igr-lin-atol", str(args.igr_lin_atol)])
    if args.igr_linear_solver is not None:
        cmd.extend(["--igr-linear-solver", str(args.igr_linear_solver)])
    if args.igr_preconditioner is not None:
        cmd.extend(["--igr-preconditioner", str(args.igr_preconditioner)])
    return cmd


def run_role(args: argparse.Namespace,
             role: str,
             direct_enabled: bool,
             extra_env: Dict[str, str]) -> Path:
    root = repo_root()
    dataset = expected_dataset(args)
    generated_dir = root / "out" / "supersonic_cylinder_w_igr" / dataset
    shutil.rmtree(generated_dir, ignore_errors=True)

    role_dir = Path(args.out_dir) / role
    if role_dir.exists():
        shutil.rmtree(role_dir)
    role_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(extra_env)
    env["DG_IGR_IGR_PARTIAL_ASSEMBLY"] = "1"
    env["DG_IGR_IGR_DEVICE_QF"] = "1"
    env["DG_IGR_DEBUG_QF"] = "1"
    env["DG_IGR_IGR_DIRECT_IDENTITY_SYSTEM"] = "1" if direct_enabled else "0"
    env.setdefault("OMP_NUM_THREADS", "1")

    log_path = Path(args.out_dir) / f"{role}.log"
    cmd = command(args)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(
            f"[cuda-direct-identity] role={role} "
            f"direct={int(direct_enabled)} command={' '.join(cmd)}\n"
        )
        proc = subprocess.run(
            cmd,
            cwd=root,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"{role} run failed with code {proc.returncode}; see {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = DATASET_RE.search(text)
    if match:
        dataset = match.group(1)
        generated_dir = root / "out" / "supersonic_cylinder_w_igr" / dataset
    pvd = generated_dir / f"{dataset}.pvd"
    if not pvd.exists():
        raise RuntimeError(f"{role} run did not produce expected PVD: {pvd}")
    copied_dir = role_dir / dataset
    shutil.copytree(generated_dir, copied_dir, dirs_exist_ok=True)
    return copied_dir / f"{dataset}.pvd"


def load_metadata(pvd: Path) -> Dict:
    metadata_path = pvd.parent / "run_metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(f"missing run metadata: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def require_path(metadata: Dict, expected: bool, role: str) -> None:
    direct = metadata.get("igr_linear", {}).get("direct_identity_system", {})
    effective = direct.get("effective")
    if effective is not expected:
        raise RuntimeError(
            f"{role} direct identity effective={effective!r}, expected {expected!r}"
        )


def compare(args: argparse.Namespace, off_pvd: Path, on_pvd: Path) -> Dict:
    frames_a = vtk_compare.parse_pvd(off_pvd)
    frames_b = vtk_compare.parse_pvd(on_pvd)
    pairs = vtk_compare.common_timesteps(frames_a, frames_b, args.time_tolerance)
    if not pairs:
        raise RuntimeError("no matching timesteps between direct-off and direct-on runs")

    arrays: Optional[List[str]] = args.array if args.array else None
    frame_diffs = [
        vtk_compare.compare_frames(frame_a, frame_b, arrays)
        for frame_a, frame_b in pairs
    ]
    aggregate = vtk_compare.aggregate_frame_diffs(frame_diffs)
    worst = vtk_compare.worst_aggregate_metrics(aggregate)
    worst_rel = worst.get("max_rel_rms", {}).get("value")
    if worst_rel is None or float(worst_rel) > args.default_rel_rms_limit:
        raise RuntimeError(
            "direct identity parity drift exceeds limit: "
            f"worst_rel_rms={sci(worst_rel)} "
            f"limit={sci(args.default_rel_rms_limit)}"
        )
    return {
        "schema_version": 1,
        "benchmark": "cuda_direct_identity_parity",
        "generated_utc": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "direct_off_pvd": str(off_pvd),
        "direct_on_pvd": str(on_pvd),
        "matched_frames": len(pairs),
        "arrays": list(aggregate.keys()),
        "aggregate": aggregate,
        "worst": worst,
        "default_rel_rms_limit": args.default_rel_rms_limit,
    }


def write_summary(out_dir: Path, payload: Dict) -> None:
    json_path = out_dir / "cuda_direct_identity_parity.json"
    md_path = out_dir / "cuda_direct_identity_parity.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    worst = payload.get("worst", {})
    lines = [
        "# CUDA Direct Identity Parity",
        "",
        f"- Direct-off PVD: `{payload['direct_off_pvd']}`",
        f"- Direct-on PVD: `{payload['direct_on_pvd']}`",
        f"- Matched frames: {payload['matched_frames']}",
        f"- Relative RMS limit: `{sci(payload['default_rel_rms_limit'])}`",
        "",
        "| metric | array | value |",
        "| --- | --- | ---: |",
    ]
    for key in ("max_rel_rms", "max_rms", "max_abs"):
        row = worst.get(key, {})
        lines.append(
            f"| {key} | {row.get('array', 'n/a')} | `{sci(row.get('value'))}` |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[cuda-direct-identity] json={json_path}")
    print(f"[cuda-direct-identity] markdown={md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--serial-ref", type=int, default=0)
    parser.add_argument("--parallel-ref", type=int, default=0)
    parser.add_argument("--tf", type=float, default=2.0e-4)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--dt-save", type=float, default=1.0e-4)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--c-alpha", default="1.0")
    parser.add_argument("--igr-stage-rtol-scale", type=float, default=10.0)
    parser.add_argument("--igr-lin-rtol", type=float, default=1.0e-8)
    parser.add_argument("--igr-lin-atol", type=float)
    parser.add_argument("--igr-linear-solver")
    parser.add_argument("--igr-preconditioner")
    parser.add_argument("--array", action="append")
    parser.add_argument("--time-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--default-rel-rms-limit", type=float, default=1.0e-10)
    parser.add_argument("--env", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_env = parse_env(args.env)
    try:
        off_pvd = run_role(args, "direct_off", False, extra_env)
        on_pvd = run_role(args, "direct_on", True, extra_env)
        require_path(load_metadata(off_pvd), False, "direct_off")
        require_path(load_metadata(on_pvd), True, "direct_on")
        payload = compare(args, off_pvd, on_pvd)
        payload["direct_off_metadata"] = str(off_pvd.parent / "run_metadata.json")
        payload["direct_on_metadata"] = str(on_pvd.parent / "run_metadata.json")
        write_summary(out_dir, payload)
        return 0
    except Exception as exc:
        print(f"cuda-direct-identity-parity: ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
