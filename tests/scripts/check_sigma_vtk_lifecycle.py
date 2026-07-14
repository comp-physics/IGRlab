#!/usr/bin/env python3
"""Verify that VTK sigma preparation replaces, rather than duplicates, stage 0."""

from __future__ import annotations

import argparse
import base64
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path


def require(pattern: str, text: str, label: str) -> re.Match[str]:
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"missing {label}: /{pattern}/")
    return match


def base64_chars(byte_count: int) -> int:
    return ((4 * byte_count // 3) + 3) & ~3


def decode_vtk_binary(root: ET.Element, data_array: ET.Element) -> bytes:
    encoded = "".join((data_array.text or "").split())
    byte_order = root.get("byte_order", "LittleEndian")
    endian = "<" if byte_order == "LittleEndian" else ">"
    compressor = root.get("compressor")

    if compressor:
        first_header_chars = base64_chars(4)
        first_header = base64.b64decode(encoded[:first_header_chars])
        (block_count,) = struct.unpack(f"{endian}I", first_header)
        header_bytes = (3 + block_count) * 4
        header_chars = base64_chars(header_bytes)
        header = base64.b64decode(encoded[:header_chars])
        values = struct.unpack(f"{endian}{3 + block_count}I", header)
        block_size, final_block_size = values[1:3]
        compressed_sizes = values[3:]
        compressed = base64.b64decode(encoded[header_chars:])
        decoded = bytearray()
        offset = 0
        for block, compressed_size in enumerate(compressed_sizes):
            block_data = compressed[offset : offset + compressed_size]
            expected = (
                final_block_size
                if block == block_count - 1 and final_block_size
                else block_size
            )
            uncompressed = zlib.decompress(block_data)
            if len(uncompressed) != expected:
                raise RuntimeError(
                    "VTK compressed block size mismatch: "
                    f"expected {expected}, found {len(uncompressed)}"
                )
            decoded.extend(uncompressed)
            offset += compressed_size
        if offset != len(compressed):
            raise RuntimeError("VTK compressed payload has trailing bytes")
        return bytes(decoded)

    header_chars = base64_chars(4)
    header = base64.b64decode(encoded[:header_chars])
    (expected_bytes,) = struct.unpack(f"{endian}I", header)
    decoded = base64.b64decode(encoded[header_chars:])
    if len(decoded) != expected_bytes:
        raise RuntimeError(
            "VTK binary payload size mismatch: "
            f"expected {expected_bytes}, found {len(decoded)}"
        )
    return decoded


def sigma_values(root: ET.Element, sigma: ET.Element) -> list[float]:
    vtk_type = sigma.get("type")
    if vtk_type != "Float64":
        raise RuntimeError(f"unsupported sigma VTK type: {vtk_type}")
    data_format = sigma.get("format", "ascii").lower()
    if data_format == "ascii":
        return [float(token) for token in (sigma.text or "").split()]
    if data_format != "binary":
        raise RuntimeError(f"unsupported sigma VTK format: {data_format}")

    raw = decode_vtk_binary(root, sigma)
    if len(raw) % 8 != 0:
        raise RuntimeError("sigma VTK payload is not a whole number of doubles")
    byte_order = root.get("byte_order", "LittleEndian")
    endian = "<" if byte_order == "LittleEndian" else ">"
    return [value[0] for value in struct.iter_unpack(f"{endian}d", raw)]


def verify_sigma_arrays(
    output_root: Path, mpi_ranks: int, expected_cycles: int = 3
) -> None:
    vtu_files = sorted(output_root.rglob("*.vtu"))
    expected = expected_cycles * mpi_ranks
    if len(vtu_files) != expected:
        raise RuntimeError(
            f"expected {expected} VTU rank frames, found {len(vtu_files)}"
        )
    for path in vtu_files:
        root = ET.parse(path).getroot()
        sigma = root.find(".//DataArray[@Name='sigma']")
        if sigma is None:
            raise RuntimeError(f"{path} does not contain a sigma array")
        values = sigma_values(root, sigma)
        if not values:
            raise RuntimeError(f"{path} sigma array is empty")
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"{path} sigma contains a non-finite value")


def run_case(command: list[str], env: dict[str, str], log_path: Path) -> str:
    completed = subprocess.run(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"application failed with {completed.returncode}\n{completed.stdout}"
        )
    return completed.stdout


def verify_required_solve_failure(
    command: list[str], env: dict[str, str], output_root: Path
) -> None:
    failure_root = output_root / "required-solve-failure"
    failure_root.mkdir()
    failure_env = dict(env)
    failure_env["DG_IGR_VTK_PREFIX_PATH"] = str(failure_root / "vtk")
    failure_command = list(command)
    failure_command[failure_command.index("-ils") + 1] = "gmres"
    failure_command[failure_command.index("-ipc") + 1] = "none"
    failure_command[failure_command.index("--igr-lin-rtol") + 1] = "1.0e-16"
    failure_command.extend(["--igr-lin-maxit", "1"])
    completed = subprocess.run(
        failure_command,
        env=failure_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (failure_root / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode == 0:
        raise RuntimeError("required sigma solve failure control unexpectedly passed")
    if "did not converge" not in completed.stdout:
        raise RuntimeError("required sigma solve failure was not classified")
    if list(failure_root.rglob("*.vtu")):
        raise RuntimeError("failed required sigma solve wrote a VTK field")


def verify_solve_purposes(
    log: str, linear_rtol: float, stage_rtol_scale: float
) -> None:
    events = re.findall(
        r"\[igr-linear\] event=solve .*?purpose=(\S+).*?"
        r"rk_stage=(-?\d+).*?effective_rtol=([0-9.eE+-]+)",
        log,
    )
    prefetch = [event for event in events if event[0] == "stage0-prefetch"]
    output_only = [event for event in events if event[0] == "output-only"]
    if len(prefetch) != 2 or len(output_only) != 1:
        raise RuntimeError(
            "unexpected solve-purpose events: "
            f"prefetch={len(prefetch)} output_only={len(output_only)}"
        )
    stage_rtol = linear_rtol * stage_rtol_scale
    for _, stage, effective_rtol in prefetch:
        if int(stage) != 0 or not math.isclose(
            float(effective_rtol), stage_rtol, rel_tol=1.0e-12
        ):
            raise RuntimeError(
                "stage-0 prefetch did not preserve stage tolerance: "
                f"stage={stage} rtol={effective_rtol} expected={stage_rtol}"
            )
    _, stage, effective_rtol = output_only[0]
    if int(stage) != -1 or not math.isclose(
        float(effective_rtol), linear_rtol, rel_tol=1.0e-12
    ):
        raise RuntimeError(
            "output-only solve did not use base tolerance: "
            f"stage={stage} rtol={effective_rtol} expected={linear_rtol}"
        )


def grid_function_values(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = text.split("\n\n", 1)[1]
    except IndexError as exc:
        raise RuntimeError(f"invalid MFEM GridFunction file: {path}") from exc
    return [float(token) for token in payload.split()]


def verify_matched_final_state(vtk_root: Path, control_root: Path) -> tuple[float, float]:
    vtk_files = sorted(
        path
        for path in vtk_root.rglob("euler-*-final.*")
        if not path.name.startswith("euler-mesh-")
    )
    control_files = sorted(
        path
        for path in control_root.rglob("euler-*-final.*")
        if not path.name.startswith("euler-mesh-")
    )
    if not vtk_files or len(vtk_files) != len(control_files):
        raise RuntimeError(
            "expected matching final conservative component files, "
            f"found vtk={len(vtk_files)} control={len(control_files)}"
        )
    max_abs = 0.0
    diff_sq = 0.0
    ref_sq = 0.0
    for vtk_path, control_path in zip(vtk_files, control_files):
        if vtk_path.name != control_path.name:
            raise RuntimeError(
                f"final-state file mismatch: {vtk_path.name} != {control_path.name}"
            )
        vtk_values = grid_function_values(vtk_path)
        control_values = grid_function_values(control_path)
        if len(vtk_values) != len(control_values):
            raise RuntimeError(f"value count differs in {vtk_path.name}")
        for vtk_value, control_value in zip(vtk_values, control_values):
            diff = vtk_value - control_value
            max_abs = max(max_abs, abs(diff))
            diff_sq += diff * diff
            ref_sq += control_value * control_value
    rel_l2 = math.sqrt(diff_sq / max(ref_sq, 1.0e-300))
    if max_abs > 1.0e-13 or rel_l2 > 1.0e-12:
        raise RuntimeError(
            "matched-schedule final state exceeds tolerance: "
            f"max_abs={max_abs:.6e} rel_l2={rel_l2:.6e}"
        )
    return max_abs, rel_l2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--mpi-numproc-flag", default="-np")
    parser.add_argument("--failure-control", action="store_true")
    parser.add_argument("--igr-stage-rtol-scale", type=float, default=1.0)
    args = parser.parse_args()

    output_root = Path(args.out_dir).resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    env = os.environ.copy()
    vtk_root = output_root / "vtk"
    control_root = output_root / "control"
    env.update(
        {
            "DG_IGR_VTK_PREFIX_PATH": str(vtk_root),
            "DG_IGR_IGR_PARTIAL_ASSEMBLY": "1",
            "DG_IGR_IGR_DEVICE_QF": "1",
            "DG_IGR_IGR_CHEB_ORDER": "3",
            "DG_IGR_IGR_CHEB_BASE_OPERATOR": "0",
            "DG_IGR_IGR_EVENT_LOG_MIN_ITERS": "1",
        }
    )
    linear_rtol = 1.0e-5 if args.device == "cuda" else 1.0e-8
    app_command = [
        str(Path(args.exe).resolve()),
        "--device",
        args.device,
        "--dg-backend",
        "dg",
        "-o",
        "1",
        "-rs",
        "0",
        "-rp",
        "0",
        "-is",
        "L2",
        "-ig",
        "BrokenL2",
        "-ca",
        "1.0",
        "-tf",
        "2.0e-4",
        "-dt",
        "1.0e-4",
        "-c",
        "0.0",
        "--igr-sigma-cadence",
        "every-stage",
        "--igr-stage-rtol-scale",
        str(args.igr_stage_rtol_scale),
        "--igr-lin-rtol",
        str(linear_rtol),
        "-ils",
        "minres",
        "-ipc",
        "chebyshev",
        "-iexface",
        "-vtk",
        "-dt_save",
        "1.0e-4",
        "-no-glvis",
        "-vs",
        "1000000",
        "-iprint",
        "0",
        "-save-state",
    ]
    command = app_command
    if args.mpi_ranks > 1:
        command = [
            args.mpiexec,
            args.mpi_numproc_flag,
            str(args.mpi_ranks),
            *app_command,
        ]
    log = run_case(command, env, output_root / "vtk.log")
    verify_solve_purposes(log, linear_rtol, args.igr_stage_rtol_scale)

    lifecycle = require(
        r"\[igr-sigma-lifecycle\] event=run-summary "
        r"prefetch_solves=(\d+) stage0_reuses=(\d+) output_only_solves=(\d+) "
        r"prefetch_invalidations=(\d+)",
        log,
        "sigma lifecycle summary",
    )
    values = tuple(int(value) for value in lifecycle.groups())
    if values != (2, 2, 1, 0):
        raise RuntimeError(f"unexpected lifecycle counts: {values}")

    summary = require(
        r"\[igr-linear\] event=run-summary calls=(\d+) converged_calls=(\d+) "
        r"production_calls=(\d+) production_converged_calls=(\d+) "
        r"output_only_calls=(\d+) output_only_converged_calls=(\d+)",
        log,
        "IGR accounting summary",
    )
    counts = tuple(int(value) for value in summary.groups())
    if counts != (7, 7, 6, 6, 1, 1):
        raise RuntimeError(f"unexpected IGR accounting: {counts}")

    verify_sigma_arrays(vtk_root, args.mpi_ranks)
    if args.failure_control:
        verify_required_solve_failure(command, env, output_root)

    control_command = list(command)
    control_command[control_command.index("-vtk")] = "-no-vtk"
    control_env = dict(env)
    control_env["DG_IGR_VTK_PREFIX_PATH"] = str(control_root)
    control_log = run_case(
        control_command, control_env, output_root / "control.log"
    )
    control_summary = require(
        r"\[igr-linear\] event=run-summary calls=(\d+) converged_calls=(\d+) "
        r"production_calls=(\d+) production_converged_calls=(\d+) "
        r"output_only_calls=(\d+) output_only_converged_calls=(\d+)",
        control_log,
        "matched-schedule control accounting",
    )
    control_counts = tuple(int(value) for value in control_summary.groups())
    if control_counts != (6, 6, 6, 6, 0, 0):
        raise RuntimeError(f"unexpected control accounting: {control_counts}")
    max_abs, rel_l2 = verify_matched_final_state(vtk_root, control_root)

    print(
        "sigma-vtk-lifecycle: PASS"
        f" device={args.device} lifecycle={values} accounting={counts}"
        f" control={control_counts} max_abs={max_abs:.6e}"
        f" rel_l2={rel_l2:.6e}"
        f" failure_control={int(args.failure_control)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - test driver must print context.
        print(f"sigma-vtk-lifecycle: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
