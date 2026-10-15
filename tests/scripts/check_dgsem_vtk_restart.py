#!/usr/bin/env python3
"""Verify exact Gauss-Lobatto DGSEM VTK cold-restart semantics.

The default mode exercises the established host DGSEM route and does not claim
packed-production DGSEM Mult coverage.  Only conservative state is
reconstructed; solver-owned IGR sigma is certified as a cold recomputation.
The optional extrapolated-LLF mode verifies the production candidate's exact
cold reconstruction and intentional extrapolation-history reset contract.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import check_vtk_restart as vtk


LOCK_PATH = Path("/tmp/dg_igr-sm120.lock")
FINAL_TIME = 2.0e-6
SAVE_DT = 1.0e-6
GLOBAL_ELEMENTS = 2
# Restarting extrapolated cadence deliberately discards the predecessor sigma
# endpoint, so the first resumed step is not bitwise equivalent to an
# uninterrupted step.  This short two-step qualification nevertheless requires
# the resulting conservative-state drift to remain well below one part per
# million in both a scale-aware max norm and the global relative L2 norm.
COLD_HISTORY_RESET_MAX_REL_LINF = 5.0e-7
COLD_HISTORY_RESET_REL_L2 = 5.0e-8
MESH_TEXT = """MFEM mesh v1.0

dimension
2

elements
2
1 3 0 1 4 3
1 3 1 2 5 4

boundary
6
1 1 0 1
1 1 1 2
2 1 2 5
3 1 5 4
3 1 4 3
4 1 3 0

vertices
6
2
0.0 0.0
0.5 0.0
1.0 0.0
0.0 1.0
0.5 1.0
1.0 1.0
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_attempt(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    for serial in range(1000):
        attempt = output / f"attempt-{stamp}-{os.getpid()}-{serial:03d}"
        try:
            attempt.mkdir()
            return attempt
        except FileExistsError:
            continue
    raise RuntimeError("unable to allocate a unique preserved attempt directory")


class Lease:
    def __init__(self, attempt: Path, required: bool):
        self.attempt = attempt
        self.required = required
        self.handle = None
        self.record: dict[str, object] = {
            "path": str(LOCK_PATH),
            "required": required,
            "status": "not-required",
        }

    def acquire(self) -> bool:
        if not self.required:
            write_json(self.attempt / "lease.json", self.record)
            return True
        self.handle = LOCK_PATH.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.record.update({"status": "busy", "observed_at": utc_now()})
            write_json(self.attempt / "lease.json", self.record)
            self.handle.close()
            self.handle = None
            print(f"DGSEM_RESTART_LEASE BUSY {LOCK_PATH}")
            return False
        holder = {
            "task": "DGSEM-RESTART-01",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": utc_now(),
        }
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps(holder, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.record.update({"status": "acquired", **holder})
        write_json(self.attempt / "lease.json", self.record)
        print(f"DGSEM_RESTART_LEASE ACQUIRED {LOCK_PATH}")
        return True

    def release(self) -> None:
        if self.handle is None:
            return
        self.record.update({"status": "released", "released_at": utc_now()})
        write_json(self.attempt / "lease.json", self.record)
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None
        print(f"DGSEM_RESTART_LEASE RELEASED {LOCK_PATH}")


class Runner:
    def __init__(self, attempt: Path, environment: dict[str, str]):
        self.attempt = attempt
        self.environment = environment
        self.records: list[dict[str, object]] = []

    def run(
        self,
        command: list[str],
        log: Path,
        vtk_output: Path,
        succeed: bool = True,
    ) -> str:
        env = dict(self.environment)
        env["DG_IGR_VTK_PREFIX_PATH"] = str(vtk_output)
        started = utc_now()
        completed = subprocess.run(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout, encoding="utf-8")
        record = {
            "command": command,
            "command_shell": shlex.join(command),
            "expected": "success" if succeed else "failure",
            "exit_code": completed.returncode,
            "log": str(log),
            "started_at": started,
            "finished_at": utc_now(),
            "vtk_output": str(vtk_output),
        }
        self.records.append(record)
        write_json(self.attempt / "commands.json", self.records)
        if succeed != (completed.returncode == 0):
            expectation = "success" if succeed else "failure"
            raise RuntimeError(
                f"expected {expectation}, got rc={completed.returncode}: {log}\n"
                f"{completed.stdout}"
            )
        return completed.stdout


def make_mesh(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MESH_TEXT, encoding="utf-8")


def base_command(args: argparse.Namespace, mesh: Path) -> list[str]:
    linear_rtol = "1.0e-5" if args.device == "cuda" else "1.0e-10"
    volume_dissipation = "llf" if args.extrapolated_llf else "none"
    time_integrator = "ssprk104" if args.extrapolated_llf else "rk3ssp"
    alpha_policy = "local" if args.extrapolated_llf else "max"
    igr_space = "L2" if args.extrapolated_llf else "H1"
    igr_grad = "BrokenL2" if args.extrapolated_llf else "H1"
    linear_solver = "cg" if args.extrapolated_llf else "minres"
    app = [
        str(Path(args.exe).resolve()),
        "--device", args.device,
        "-m", str(mesh),
        "-rs", "0",
        "-rp", "0",
        "-o", str(args.order),
        "-tf", f"{FINAL_TIME:.17g}",
        "-dt", f"{SAVE_DT:.17g}",
        "-c", "0.0",
        "-dg", "dgsem",
        "-ds", "none",
        "-dvf", "central",
        "-dvd", volume_dissipation,
        "-dsf", "rusanov",
        "--time-integrator", time_integrator,
        "-as", "0",
        "-no-glvis",
        "-vtk",
        "-dt_save", f"{SAVE_DT:.17g}",
        "-no-vtk-h1",
        "-no-vtk-l2",
        "-no-vtk-rank",
        "-no-vtk-zs",
        "-no-save-state",
        "-ca", "1.0",
        "-ial", alpha_policy,
        "-is", igr_space,
        "-ig", igr_grad,
        "-ils", linear_solver,
        "-ipc", "chebyshev",
        "--igr-nonconvergence-policy", "error",
        "-irtol", linear_rtol,
        "-imaxit", "1000",
        "-iprint", "0",
    ]
    if args.extrapolated_llf:
        app.extend(
            [
                "--igr-exact-dg-face",
                "--igr-sigma-cadence", "extrapolated-every-step",
            ]
        )
    if args.mpi_ranks > 1 or args.mpi_single_rank:
        return [args.mpiexec, args.mpi_numproc_flag, str(args.mpi_ranks), *app]
    return app


def replace_option(command: list[str], option: str, value: str) -> list[str]:
    changed = list(command)
    changed[changed.index(option) + 1] = value
    return changed


def entries_contract(pvd: Path, args: argparse.Namespace) -> dict[str, object]:
    entries = vtk.pvd_entries(pvd)
    expected = [(0, 0.0), (1, SAVE_DT), (2, FINAL_TIME)]
    if len(entries) != len(expected):
        raise RuntimeError(f"unexpected PVD entry count: {entries}")
    manifests: list[dict[str, object]] = []
    piece_names: list[list[str]] = []
    for (cycle, time, pvtu), (want_cycle, want_time) in zip(entries, expected):
        if cycle != want_cycle or not math.isclose(time, want_time, abs_tol=1.0e-15):
            raise RuntimeError(f"unexpected PVD cycle/time: {(cycle, time)}")
        manifest_path = pvtu.parent / "restart.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sigma_policy = (
            "cold-recompute-and-reset-extrapolation-history"
            if args.extrapolated_llf
            else "cold-recompute-from-conservative-state"
        )
        required = {
            "schema_version": 4,
            "complete": True,
            "case": "dgsem_sedov_taylor_blast",
            "cycle": cycle,
            "time": time,
            "step": cycle,
            "ranks": args.mpi_ranks,
            "global_elements": GLOBAL_ELEMENTS,
            "order": args.order,
            "serial_refinement": 0,
            "parallel_refinement": 0,
            "dg_backend": "dgsem",
            "time_integrator": (
                "ssprk104" if args.extrapolated_llf else "rk3ssp"
            ),
            "dgsem_stabilization": "none",
            "dgsem_volume_flux": "central",
            "dgsem_volume_dissipation": (
                "llf" if args.extrapolated_llf else "none"
            ),
            "dgsem_surface_flux": "rusanov",
            "dgsem_entropy_jacobian": "fd",
            "basis": "GaussLobatto",
            "ordering": "byNODES",
            "data_type": "Float64",
            "vtk_format": "binary",
            "piece_pattern": "proc%06d.vtu",
            "igr_enabled": True,
            "igr_space": "l2" if args.extrapolated_llf else "h1",
            "igr_grad": "brokenl2" if args.extrapolated_llf else "h1",
            "alpha_length_policy": "local" if args.extrapolated_llf else "max",
            "exact_dg_face": args.extrapolated_llf,
            "sigma_cadence": (
                "extrapolated-every-step"
                if args.extrapolated_llf
                else "every-stage"
            ),
            "sigma_update_interval": 1,
            "igr_invariant_domain_flux": False,
            "zhang_shu_limiter": True,
            "rho_floor": 5.0e-7,
            "pressure_floor": 1.0e-6,
            "sigma_saved": True,
            "sigma_restart_policy": sigma_policy,
            "static_source_present": False,
            "static_source_fingerprint": "none",
        }
        for key, want in required.items():
            if manifest.get(key) != want:
                raise RuntimeError(
                    f"manifest {manifest_path} has {key}={manifest.get(key)!r}, expected {want!r}"
                )
        local_elements = manifest.get("local_elements")
        if (
            not isinstance(local_elements, list)
            or len(local_elements) != args.mpi_ranks
            or sum(local_elements) != GLOBAL_ELEMENTS
            or any(int(count) <= 0 for count in local_elements)
        ):
            raise RuntimeError(f"invalid same-rank element partition: {local_elements}")
        pieces = vtk.piece_paths(pvtu)
        names = [piece.name for piece in pieces]
        expected_names = [
            f"proc{rank:06d}.vtu" for rank in range(args.mpi_ranks)
        ]
        if names != expected_names or any(not piece.is_file() for piece in pieces):
            raise RuntimeError(f"invalid same-rank piece identity: {names}")
        manifests.append(manifest)
        piece_names.append(names)
    fingerprints = {str(item["mesh_fingerprint"]) for item in manifests}
    if len(fingerprints) != 1:
        raise RuntimeError(f"mesh fingerprint changed across cycles: {fingerprints}")
    return {
        "entries": [[cycle, time] for cycle, time, _ in entries],
        "mesh_fingerprint": next(iter(fingerprints)),
        "local_elements": manifests[0]["local_elements"],
        "piece_names": piece_names,
    }


def canonical_hash(items: list[float | int], kind: str) -> str:
    digest = hashlib.sha256()
    code = {"Float64": "d", "Int32": "i", "UInt8": "B"}[kind]
    for item in items:
        digest.update(struct.pack(f"<{code}", item))
    return digest.hexdigest()


def piece_arrays(path: Path) -> tuple[dict[str, list[float | int]], dict[str, list[float | int]]]:
    root = ET.parse(path).getroot()
    fields: dict[str, list[float | int]] = {}
    for name in ("density", "momentum", "energy"):
        array = root.find(f".//DataArray[@Name='{name}']")
        if array is None:
            raise RuntimeError(f"{path} lacks {name}")
        fields[name] = vtk.values(root, array)
    mesh: dict[str, list[float | int]] = {}
    points = root.find(".//Points/DataArray")
    if points is None:
        raise RuntimeError(f"{path} lacks points")
    mesh["points"] = vtk.values(root, points)
    for name in ("connectivity", "offsets", "types", "attribute"):
        array = root.find(f".//DataArray[@Name='{name}']")
        if array is None:
            raise RuntimeError(f"{path} lacks {name}")
        mesh[name] = vtk.values(root, array)
    return fields, mesh


def compare_final(
    control_pvtu: Path,
    restart_pvtu: Path,
    device: str,
    cold_history_reset: bool = False,
) -> dict[str, object]:
    control_pieces = vtk.piece_paths(control_pvtu)
    restart_pieces = vtk.piece_paths(restart_pvtu)
    expected_names = [f"proc{rank:06d}.vtu" for rank in range(len(control_pieces))]
    if [path.name for path in control_pieces] != expected_names:
        raise RuntimeError("control PVTU piece ordering is not same-rank")
    if [path.name for path in restart_pieces] != expected_names:
        raise RuntimeError("restart PVTU piece ordering is not same-rank")
    backend_max_limit = 2.0e-8 if device == "cuda" else 5.0e-10
    backend_rel_limit = 1.0e-9 if device == "cuda" else 2.0e-11
    max_abs = 0.0
    diff_sq = 0.0
    ref_sq = 0.0
    ref_linf = 0.0
    field_hashes: dict[str, list[dict[str, str]]] = {
        name: [] for name in ("density", "momentum", "energy")
    }
    mesh_hashes: list[dict[str, str]] = []
    for rank, (control_piece, restart_piece) in enumerate(
        zip(control_pieces, restart_pieces)
    ):
        lhs, lhs_mesh = piece_arrays(control_piece)
        rhs, rhs_mesh = piece_arrays(restart_piece)
        if lhs_mesh != rhs_mesh:
            raise RuntimeError(f"rank {rank} final mesh arrays differ")
        combined_mesh = hashlib.sha256()
        for name, kind in (
            ("points", "Float64"),
            ("connectivity", "Int32"),
            ("offsets", "Int32"),
            ("types", "UInt8"),
            ("attribute", "Int32"),
        ):
            combined_mesh.update(bytes.fromhex(canonical_hash(lhs_mesh[name], kind)))
        mesh_digest = combined_mesh.hexdigest()
        mesh_hashes.append({"control": mesh_digest, "restart": mesh_digest})
        for name in lhs:
            if len(lhs[name]) != len(rhs[name]):
                raise RuntimeError(f"rank {rank} {name} value counts differ")
            field_hashes[name].append(
                {
                    "control": canonical_hash(lhs[name], "Float64"),
                    "restart": canonical_hash(rhs[name], "Float64"),
                }
            )
            for expected, found in zip(lhs[name], rhs[name]):
                diff = float(found) - float(expected)
                max_abs = max(max_abs, abs(diff))
                diff_sq += diff * diff
                ref_sq += float(expected) * float(expected)
                ref_linf = max(ref_linf, abs(float(expected)))
    rel_l2 = math.sqrt(diff_sq / max(ref_sq, 1.0e-300))
    relative_linf = max_abs / max(ref_linf, 1.0)
    if (
        not math.isfinite(max_abs)
        or not math.isfinite(relative_linf)
        or not math.isfinite(rel_l2)
    ):
        raise RuntimeError(
            "DGSEM split-run drift is non-finite: "
            f"max_abs={max_abs} relative_linf={relative_linf} rel_l2={rel_l2}"
        )
    if cold_history_reset:
        comparison_policy = "bounded-cold-history-reset"
        max_limit = max(
            backend_max_limit,
            COLD_HISTORY_RESET_MAX_REL_LINF * max(ref_linf, 1.0),
        )
        relative_linf_limit = COLD_HISTORY_RESET_MAX_REL_LINF
        rel_limit = max(backend_rel_limit, COLD_HISTORY_RESET_REL_L2)
    else:
        comparison_policy = "split-equivalence"
        max_limit = backend_max_limit
        relative_linf_limit = max_limit / max(ref_linf, 1.0)
        rel_limit = backend_rel_limit
    if (
        max_abs > max_limit
        or relative_linf > relative_linf_limit
        or rel_l2 > rel_limit
    ):
        raise RuntimeError(
            "DGSEM split-run drift exceeds "
            f"{comparison_policy} tolerance: max_abs={max_abs:.6e} "
            f"relative_linf={relative_linf:.6e} rel_l2={rel_l2:.6e}"
        )
    return {
        "policy": comparison_policy,
        "max_abs": max_abs,
        "reference_linf": ref_linf,
        "relative_linf": relative_linf,
        "rel_l2": rel_l2,
        "max_abs_tolerance": max_limit,
        "relative_linf_tolerance": relative_linf_limit,
        "rel_l2_tolerance": rel_limit,
        "cold_history_reset": cold_history_reset,
        "field_sha256": field_hashes,
        "mesh_sha256": mesh_hashes,
        "same_rank_piece_names": expected_names,
    }


def restart_diagnostics(log: str) -> dict[str, object]:
    restart = vtk.require(
        r"\[vtk-restart\] source=.*? cycle=1 time=1e-06 step=1 legacy=0 .*?"
        r"reconstruction_max_abs=([0-9.eE+-]+) reconstruction_rel_l2=([0-9.eE+-]+) "
        r"mesh_point_max_abs=([0-9.eE+-]+) condition_inf=([0-9.eE+-]+) "
        r"rho_min=([0-9.eE+-]+) p_min=([0-9.eE+-]+) "
        r"sigma_policy=cold-recompute output_mode=new next_cycle=2 next_time=2e-06",
        log,
        "DGSEM restart certificate",
    )
    values = [float(value) for value in restart.groups()]
    if values[0] > 5.0e-11 or values[1] > 5.0e-12:
        raise RuntimeError(f"reconstruction certificate exceeds loader tolerance: {values[:2]}")
    if values[2] > 2.0e-12 or values[4] <= 0.0 or values[5] <= 0.0:
        raise RuntimeError(f"restart mesh/admissibility certificate is invalid: {values}")
    sigma = vtk.require(
        r"\[vtk-restart\] event=sigma-initialization solved=(\d+) cold_start=(\d+) "
        r"reusable_stage0=(\d+) certification_supported=(\d+) "
        r"independent_residual_certified=(\d+) "
        r"independent_residual_rel_rhs=([0-9.eE+-]+)",
        log,
        "cold sigma recomputation certificate",
    )
    sigma_values = [int(value) for value in sigma.groups()[:5]]
    if sigma_values != [1, 1, 1, 1, 1]:
        raise RuntimeError(f"sigma was not cold-recomputed and certified: {sigma.groups()}")
    lifecycle = vtk.require(
        r"\[igr-sigma-lifecycle\] event=run-summary prefetch_solves=(\d+) "
        r"stage0_reuses=(\d+) output_only_solves=(\d+) prefetch_invalidations=(\d+)",
        log,
        "restart sigma lifecycle",
    )
    lifecycle_values = [int(value) for value in lifecycle.groups()]
    if lifecycle_values[0] < 1 or lifecycle_values[1] < 1:
        raise RuntimeError(f"unexpected restart sigma lifecycle: {lifecycle_values}")
    return {
        "reconstruction_max_abs": values[0],
        "reconstruction_rel_l2": values[1],
        "mesh_point_max_abs": values[2],
        "interpolation_condition_inf": values[3],
        "rho_min": values[4],
        "pressure_min": values[5],
        "sigma_initialization": {
            "solved": sigma_values[0],
            "cold_start": sigma_values[1],
            "reusable_stage0": sigma_values[2],
            "certification_supported": sigma_values[3],
            "independent_residual_certified": sigma_values[4],
            "independent_residual_rel_rhs": float(sigma.group(6)),
        },
        "sigma_lifecycle": {
            "prefetch_solves": lifecycle_values[0],
            "stage0_reuses": lifecycle_values[1],
            "output_only_solves": lifecycle_values[2],
            "prefetch_invalidations": lifecycle_values[3],
        },
    }


def packed_dispatch_diagnostics(
    log: str, args: argparse.Namespace, label: str
) -> dict[str, object]:
    dispatch = vtk.require(
        r"\[dgsem-production\] selected=mfem-native-packed strict=1 "
        r"backend=(cpu|cuda) .*?volume_dissipation=llf "
        r"llf_volume_active=1 frozen_sigma_spectral_speed=1 "
        r"order=(\d+) dim=2",
        log,
        f"{label} strict packed DGSEM dispatch",
    )
    backend, order = dispatch.groups()
    if backend != args.device or int(order) != args.order:
        raise RuntimeError(
            f"{label} packed dispatch mismatch: backend={backend} order={order}"
        )
    return {
        "selected": "mfem-native-packed",
        "strict": True,
        "backend": backend,
        "volume_dissipation": "llf",
        "frozen_sigma_spectral_speed": True,
        "order": int(order),
        "dimension": 2,
    }


def selected_cycle(pvd: Path, cycle: int = 1) -> tuple[int, float, Path]:
    return next(entry for entry in vtk.pvd_entries(pvd) if entry[0] == cycle)


def copy_variant(source_pvd: Path, root: Path) -> Path:
    shutil.copytree(source_pvd.parent, root)
    return root / source_pvd.name


def update_manifest(pvd: Path, **changes: object) -> None:
    manifest_path = selected_cycle(pvd)[2].parent / "restart.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(changes)
    write_json(manifest_path, manifest)


def set_field_first(pvd: Path, field: str, value: float) -> None:
    piece = vtk.piece_paths(selected_cycle(pvd)[2])[0]
    tree = ET.parse(piece)
    root = tree.getroot()
    array = root.find(f".//DataArray[@Name='{field}']")
    if array is None:
        raise RuntimeError(f"negative source lacks {field}")
    items = vtk.values(root, array)
    items[0] = value
    endian = "<" if root.get("byte_order", "LittleEndian") == "LittleEndian" else ">"
    array.text = "\n" + vtk.binary_text(vtk.encode_values(items, "Float64", endian), endian) + "\n"
    tree.write(piece, encoding="utf-8", xml_declaration=True)


def offset_first_coordinate(pvd: Path, delta: float) -> None:
    piece = vtk.piece_paths(selected_cycle(pvd)[2])[0]
    tree = ET.parse(piece)
    root = tree.getroot()
    array = root.find(".//Points/DataArray")
    if array is None:
        raise RuntimeError("coordinate negative source lacks points")
    items = vtk.values(root, array)
    items[0] = float(items[0]) + delta
    endian = "<" if root.get("byte_order", "LittleEndian") == "LittleEndian" else ">"
    array.text = "\n" + vtk.binary_text(vtk.encode_values(items, "Float64", endian), endian) + "\n"
    tree.write(piece, encoding="utf-8", xml_declaration=True)


def update_pvtu_field(
    pvd: Path,
    field: str,
    attribute: str,
    value: str,
) -> None:
    pvtu = selected_cycle(pvd)[2]
    tree = ET.parse(pvtu)
    declaration = tree.getroot().find(f".//PDataArray[@Name='{field}']")
    if declaration is None:
        raise RuntimeError(f"PVTU negative source lacks {field}")
    declaration.set(attribute, value)
    tree.write(pvtu, encoding="utf-8", xml_declaration=True)


def run_negative_controls(
    runner: Runner,
    base: list[str],
    source_pvd: Path,
    root: Path,
) -> dict[str, object]:
    results: dict[str, object] = {}

    def execute(
        name: str,
        pvd: Path,
        command: list[str],
        pattern: str,
        contract: dict[str, object],
    ) -> None:
        source_before = vtk.tree_hashes(pvd.parent)
        output = root / name / "resumed" / "vtk"
        text = runner.run(command, root / name / "run.log", output, succeed=False)
        vtk.require(pattern, text, name)
        source_after = vtk.tree_hashes(pvd.parent)
        if source_before != source_after:
            raise RuntimeError(f"{name} modified its source dataset")
        misleading = [
            path for path in output.rglob("*")
            if path.is_file() and path.suffix in {".pvd", ".pvtu", ".vtu", ".json"}
        ] if output.exists() else []
        if misleading:
            raise RuntimeError(f"{name} created misleading resumed output: {misleading}")
        results[name] = {
            **contract,
            "reason_code": name.replace("-", "_").upper(),
            "exit_code": runner.records[-1]["exit_code"],
            "matched_pattern": pattern,
            "source_unchanged": True,
            "resumed_dataset_files": 0,
            "transactional_contract": "rejected before final caller-state commit",
        }

    def restart(pvd: Path, active: list[str] | None = None, legacy: bool = False) -> list[str]:
        command = [*(active or base), "--restart-vtk", str(pvd), "--restart-cycle", "1"]
        if legacy:
            command.extend(["--restart-step", "1"])
        return command

    pvd = copy_variant(source_pvd, root / "dgsem-gausslegendre" / "source")
    update_manifest(pvd, basis="GaussLegendre")
    execute(
        "dgsem-gausslegendre",
        pvd,
        restart(pvd),
        r"manifest is incompatible",
        {"active_backend": "dgsem", "manifest_basis": "GaussLegendre"},
    )

    pvd = copy_variant(source_pvd, root / "wrong-dgsem-transport" / "source")
    update_manifest(pvd, dgsem_volume_dissipation="llf")
    execute(
        "wrong-dgsem-transport",
        pvd,
        restart(pvd),
        r"manifest is incompatible",
        {
            "active_volume_dissipation": "none",
            "manifest_volume_dissipation": "llf",
        },
    )

    pvd = copy_variant(source_pvd, root / "dg-gausslobatto" / "source")
    update_manifest(pvd, dg_backend="dg")
    execute(
        "dg-gausslobatto",
        pvd,
        restart(pvd, replace_option(base, "-dg", "dg")),
        r"manifest is incompatible",
        {"active_backend": "dg", "manifest_basis": "GaussLobatto"},
    )

    pvd = copy_variant(source_pvd, root / "wrong-dimension" / "source")
    update_manifest(pvd, dimension=3)
    execute(
        "wrong-dimension",
        pvd,
        restart(pvd),
        r"manifest is incompatible",
        {"active_dimension": 2, "manifest_dimension": 3},
    )

    pvd = copy_variant(source_pvd, root / "wrong-ordering" / "source")
    update_manifest(pvd, ordering="byVDIM")
    execute(
        "wrong-ordering",
        pvd,
        restart(pvd),
        r"manifest is incompatible",
        {"active_ordering": "byNODES", "manifest_ordering": "byVDIM"},
    )

    pvd = copy_variant(source_pvd, root / "wrong-manifest-order" / "source")
    update_manifest(pvd, order=2)
    execute(
        "wrong-manifest-order",
        pvd,
        restart(pvd),
        r"manifest is incompatible",
        {"active_order": 1, "manifest_order": 2},
    )

    pvd = copy_variant(source_pvd, root / "wrong-mesh-fingerprint" / "source")
    update_manifest(pvd, mesh_fingerprint="0000000000000000")
    execute(
        "wrong-mesh-fingerprint",
        pvd,
        restart(pvd),
        r"manifest is incompatible",
        {"manifest_mesh_fingerprint": "0000000000000000"},
    )

    pvd = copy_variant(source_pvd, root / "wrong-coordinates" / "source")
    offset_first_coordinate(pvd, 0.125)
    execute(
        "wrong-coordinates",
        pvd,
        restart(pvd),
        r"physical points do not match the active mesh",
        {"coordinate_offset": 0.125},
    )

    pvd = copy_variant(source_pvd, root / "wrong-order" / "source")
    update_manifest(pvd, order=5)
    execute(
        "wrong-order",
        pvd,
        restart(pvd),
        r"manifest is incompatible",
        {
            "active_order": 1,
            "manifest_order": 5,
            "supported_orders": [1, 2, 3, 4],
        },
    )

    pvd = copy_variant(source_pvd, root / "missing-dgsem-manifest" / "source")
    (selected_cycle(pvd)[2].parent / "restart.json").unlink()
    execute(
        "missing-dgsem-manifest",
        pvd,
        restart(pvd, legacy=True),
        r"DGSEM VTK restart requires a complete restart manifest",
        {"active_backend": "dgsem", "explicit_legacy_step": 1},
    )

    pvd = copy_variant(source_pvd, root / "wrong-refinement" / "source")
    execute(
        "wrong-refinement",
        pvd,
        restart(pvd, replace_option(base, "-rp", "1")),
        r"manifest is incompatible",
        {"active_parallel_refinement": 1},
    )

    pvd = copy_variant(source_pvd, root / "wrong-ranks" / "source")
    update_manifest(pvd, ranks=2)
    execute(
        "wrong-ranks",
        pvd,
        restart(pvd),
        r"manifest is incompatible",
        {"active_ranks": 1, "manifest_ranks": 2},
    )

    pvd = copy_variant(source_pvd, root / "wrong-field-name" / "source")
    update_pvtu_field(pvd, "density", "Name", "not_density")
    execute(
        "wrong-field-name",
        pvd,
        restart(pvd),
        r"required field declaration is incompatible",
        {"required_field": "density", "declared_field": "not_density"},
    )

    pvd = copy_variant(source_pvd, root / "wrong-field-components" / "source")
    update_pvtu_field(pvd, "momentum", "NumberOfComponents", "3")
    execute(
        "wrong-field-components",
        pvd,
        restart(pvd),
        r"required field declaration is incompatible",
        {"field": "momentum", "active_components": 2, "declared_components": 3},
    )

    pvd = copy_variant(source_pvd, root / "missing-piece" / "source")
    vtk.piece_paths(selected_cycle(pvd)[2])[0].unlink()
    execute(
        "missing-piece",
        pvd,
        restart(pvd),
        r"missing restart rank piece",
        {"payload": "missing rank-0 piece"},
    )

    pvd = copy_variant(source_pvd, root / "nonfinite-state" / "source")
    set_field_first(pvd, "density", math.nan)
    execute(
        "nonfinite-state",
        pvd,
        restart(pvd),
        r"non-finite",
        {"payload": "density[0]=NaN"},
    )

    pvd = copy_variant(source_pvd, root / "inadmissible-state" / "source")
    set_field_first(pvd, "energy", 0.0)
    execute(
        "inadmissible-state",
        pvd,
        restart(pvd),
        r"not consumer-point admissible",
        {"payload": "energy[0]=0"},
    )

    pvd = copy_variant(source_pvd, root / "corrupt-payload" / "source")
    piece = vtk.piece_paths(selected_cycle(pvd)[2])[0]
    tree = ET.parse(piece)
    density = tree.getroot().find(".//DataArray[@Name='density']")
    if density is None:
        raise RuntimeError("corrupt source lacks density")
    density.text = "!!!!"
    tree.write(piece, encoding="utf-8", xml_declaration=True)
    execute(
        "corrupt-payload",
        pvd,
        restart(pvd),
        r"base64|binary header|byte count",
        {"payload": "invalid base64 density"},
    )

    pvd = copy_variant(source_pvd, root / "truncated-payload" / "source")
    piece = vtk.piece_paths(selected_cycle(pvd)[2])[0]
    tree = ET.parse(piece)
    density = tree.getroot().find(".//DataArray[@Name='density']")
    if density is None:
        raise RuntimeError("truncated source lacks density")
    encoded = "".join((density.text or "").split())
    density.text = encoded[:-4]
    tree.write(piece, encoding="utf-8", xml_declaration=True)
    execute(
        "truncated-payload",
        pvd,
        restart(pvd),
        r"payload size|truncated|byte count",
        {"payload": "truncated density binary"},
    )

    pvd = copy_variant(source_pvd, root / "incompatible-manifest" / "source")
    update_manifest(pvd, schema_version=99)
    execute(
        "incompatible-manifest",
        pvd,
        restart(pvd),
        r"unsupported schema",
        {"manifest_schema": 99},
    )
    return results


def run_rank_piece_reorder_negative(
    runner: Runner,
    base: list[str],
    source_pvd: Path,
    root: Path,
) -> dict[str, object]:
    name = "pvtu-rank-piece-reorder"
    pvd = copy_variant(source_pvd, root / name / "source")
    pvtu = selected_cycle(pvd)[2]
    tree = ET.parse(pvtu)
    pieces = tree.getroot().findall(".//Piece")
    if len(pieces) < 2:
        raise RuntimeError("rank-piece reorder control requires at least two pieces")
    first = pieces[0].get("Source", "")
    second = pieces[1].get("Source", "")
    pieces[0].set("Source", second)
    pieces[1].set("Source", first)
    tree.write(pvtu, encoding="utf-8", xml_declaration=True)
    source_before = vtk.tree_hashes(pvd.parent)
    output = root / name / "resumed" / "vtk"
    command = [*base, "--restart-vtk", str(pvd), "--restart-cycle", "1"]
    text = runner.run(command, root / name / "run.log", output, succeed=False)
    pattern = r"piece ordering/naming is incompatible with same-rank restart"
    vtk.require(pattern, text, name)
    if vtk.tree_hashes(pvd.parent) != source_before:
        raise RuntimeError("rank-piece reorder rejection modified its source dataset")
    misleading = [
        path for path in output.rglob("*")
        if path.is_file() and path.suffix in {".pvd", ".pvtu", ".vtu", ".json"}
    ] if output.exists() else []
    if misleading:
        raise RuntimeError(f"rank-piece reorder created misleading output: {misleading}")
    return {
        name: {
            "reason_code": "PVTU_RANK_PIECE_REORDER",
            "exit_code": runner.records[-1]["exit_code"],
            "matched_pattern": pattern,
            "original_piece_order": [first, second],
            "reordered_piece_order": [second, first],
            "source_unchanged": True,
            "resumed_dataset_files": 0,
        }
    }


def sha256_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name != "sha256_manifest.json"
    }


def execute(args: argparse.Namespace, attempt: Path, runner: Runner) -> dict[str, object]:
    mesh = attempt / "mesh" / "dgsem_restart_square.mesh"
    make_mesh(mesh)
    base = base_command(args, mesh)

    control_root = attempt / "control" / "vtk"
    control_log = runner.run(base, attempt / "control" / "run.log", control_root)
    source_pvd = vtk.dataset_pvd(control_root)
    source_contract = entries_contract(source_pvd, args)
    source_hashes = vtk.tree_hashes(source_pvd.parent)

    restart_root = attempt / "restart" / "vtk"
    restart_command = [
        *base,
        "--restart-vtk", str(source_pvd),
        "--restart-cycle", "1",
    ]
    restart_log = runner.run(
        restart_command,
        attempt / "restart" / "run.log",
        restart_root,
    )
    diagnostics = restart_diagnostics(restart_log)
    packed_dispatch: dict[str, object] | None = None
    if args.extrapolated_llf:
        packed_dispatch = {
            "control": packed_dispatch_diagnostics(
                control_log, args, "control"
            ),
            "restart": packed_dispatch_diagnostics(
                restart_log, args, "restart"
            ),
        }
    if vtk.tree_hashes(source_pvd.parent) != source_hashes:
        raise RuntimeError("successful cold restart modified its source dataset")

    restart_pvd = vtk.dataset_pvd(restart_root)
    restart_entries = vtk.pvd_entries(restart_pvd)
    if len(restart_entries) != 1 or restart_entries[0][0] != 2 or not math.isclose(
        restart_entries[0][1], FINAL_TIME, abs_tol=1.0e-15
    ):
        raise RuntimeError(f"restart output cycle/time numbering is wrong: {restart_entries}")
    restart_manifest = json.loads(
        (restart_entries[0][2].parent / "restart.json").read_text(encoding="utf-8")
    )
    if restart_manifest.get("step") != 2 or restart_manifest.get("next_cycle") != 3:
        raise RuntimeError("restart manifest step/cycle continuation is wrong")
    expected_sigma_policy = (
        "cold-recompute-and-reset-extrapolation-history"
        if args.extrapolated_llf
        else "cold-recompute-from-conservative-state"
    )
    if restart_manifest.get("sigma_restart_policy") != expected_sigma_policy:
        raise RuntimeError("restart manifest has the wrong sigma cold-start policy")
    metadata = json.loads(
        (restart_pvd.parent / "run_metadata.json").read_text(encoding="utf-8")
    )
    runtime = metadata.get("runtime", {})
    igr_linear = metadata.get("igr_linear", {})
    if (
        metadata.get("schema_version") != 3
        or metadata.get("restart", {}).get("cycle") != 1
        or metadata.get("performance", {}).get("segment_steps") != 1
        or runtime.get("dgsem_stabilization") != "none"
        or runtime.get("dgsem_volume_flux") != "central"
        or runtime.get("dgsem_volume_dissipation")
        != ("llf" if args.extrapolated_llf else "none")
        or runtime.get("dgsem_surface_flux") != "rusanov"
        or runtime.get("dgsem_entropy_jacobian") != "fd"
        or runtime.get("igr_sigma_cadence")
        != (
            "extrapolated-every-step"
            if args.extrapolated_llf
            else "every-stage"
        )
        or runtime.get("igr_sigma_update_interval") != 1
        or runtime.get("igr_alpha_length_policy")
        != ("local" if args.extrapolated_llf else "max")
        or runtime.get("igr_exact_dg_face_operator")
        != args.extrapolated_llf
        or runtime.get("igr_frozen_sigma_spectral_speed")
        != args.extrapolated_llf
        or (
            args.extrapolated_llf
            and (
                igr_linear.get("last_solver") != "cg"
                or igr_linear.get("last_precond") != "operatorchebyshev"
            )
        )
    ):
        raise RuntimeError(
            "restart metadata lacks effective method/source/segment provenance"
        )

    control_final = next(entry[2] for entry in vtk.pvd_entries(source_pvd) if entry[0] == 2)
    comparison = compare_final(
        control_final,
        restart_entries[0][2],
        args.device,
        cold_history_reset=args.extrapolated_llf,
    )
    negatives: dict[str, object] = {}
    if args.negative_controls:
        if args.device != "cpu" or args.mpi_ranks != 1:
            raise RuntimeError("negative controls require serial CPU")
        negatives = run_negative_controls(
            runner,
            base,
            source_pvd,
            attempt / "negative",
        )
        if vtk.tree_hashes(source_pvd.parent) != source_hashes:
            raise RuntimeError("negative controls modified the positive source dataset")

    if args.rank_piece_reorder_negative:
        if args.mpi_ranks < 2:
            raise RuntimeError("rank-piece reorder negative requires at least two ranks")
        negatives.update(
            run_rank_piece_reorder_negative(
                runner,
                base,
                source_pvd,
                attempt / "negative",
            )
        )
        if vtk.tree_hashes(source_pvd.parent) != source_hashes:
            raise RuntimeError("rank reorder control modified the positive source dataset")

    production = vtk.require(
        r"production_calls=(\d+) production_converged_calls=(\d+)",
        control_log,
        "control IGR solves",
    )
    production_calls = int(production.group(1))
    production_converged_calls = int(production.group(2))
    if production_calls < 1 or production_converged_calls != production_calls:
        raise RuntimeError(
            "control IGR production solves did not all converge: "
            f"{production.groups()}"
        )
    return {
        "schema_version": 1,
        "status": "pass",
        "task": "DGSEM-RESTART-01",
        "claim_scope": (
            "DGSEM extrapolated-LLF cold restart with exact reconstruction and "
            "intentional history reset"
            if args.extrapolated_llf
            else "pre-composition host DGSEM cold-restart evidence; not "
            "packed-production Mult evidence"
        ),
        "case": "dgsem_sedov_taylor_blast",
        "boundary_condition": "all-slip",
        "tuple": {
            "backend": "dgsem",
            "basis": "GaussLobatto",
            "order": args.order,
            "volume_flux": "central",
            "volume_dissipation": "llf" if args.extrapolated_llf else "none",
            "surface_flux": "rusanov",
            "stabilization": "none",
            "sigma_cadence": (
                "extrapolated-every-step"
                if args.extrapolated_llf
                else "every-stage"
            ),
        },
        "device": args.device,
        "ranks": args.mpi_ranks,
        "mesh": {
            "path": str(mesh),
            "sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
            "global_elements": GLOBAL_ELEMENTS,
            **source_contract,
        },
        "control": {
            "final_time": FINAL_TIME,
            "checkpoint_cycle": 1,
            "checkpoint_time": SAVE_DT,
            "checkpoint_step": 1,
            "source_dataset_sha256": source_hashes,
        },
        "restart": {
            "output_entries": [[entry[0], entry[1]] for entry in restart_entries],
            "final_step": restart_manifest["step"],
            "final_cycle": restart_manifest["cycle"],
            **diagnostics,
        },
        "comparison": comparison,
        "igr_production_solves": {
            "calls": production_calls,
            "converged_calls": production_converged_calls,
        },
        "dgsem_packed_dispatch": packed_dispatch,
        "sigma_truth": {
            "serialized_sigma_persistence_claimed": False,
            "conservative_state_reconstructed": True,
            "policy": expected_sigma_policy,
            "extrapolation_history_restored": False,
            "independent_residual_certified": True,
        },
        "negative_controls_run": (
            args.negative_controls or args.rank_piece_reorder_negative
        ),
        "negative": negatives,
        "commands_file": str(attempt / "commands.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--order", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-single-rank", action="store_true",
                        help="Use the selected MPI launcher even for one rank")
    parser.add_argument("--negative-controls", action="store_true")
    parser.add_argument("--rank-piece-reorder-negative", action="store_true")
    parser.add_argument(
        "--extrapolated-llf",
        action="store_true",
        help=(
            "exercise DGSEM LLF with local-alpha exact-face IGR and "
            "extrapolated-every-step cold-history reset"
        ),
    )
    args = parser.parse_args()
    if args.mpi_ranks < 1:
        raise RuntimeError("--mpi-ranks must be positive")
    if args.extrapolated_llf and args.negative_controls:
        raise RuntimeError(
            "--negative-controls exercise the legacy/default transport tuple"
        )

    attempt = create_attempt(Path(args.out_dir).resolve())
    write_json(
        attempt / "attempt.json",
        {"status": "running", "started_at": utc_now(), "argv": sys.argv},
    )
    environment = os.environ.copy()
    environment.update(
        {
            "DG_IGR_IGR_CHEB_ORDER": "3",
            "DG_IGR_IGR_CHEB_BASE_OPERATOR": (
                "0" if args.extrapolated_llf else "1"
            ),
        }
    )
    if args.extrapolated_llf:
        environment["DG_IGR_DGSEM_PRODUCTION_STRICT"] = "1"
    else:
        # Retain the legacy closure controls only for the pre-composition
        # restart surface. Exact DG faces make both policies intrinsic to the
        # selected extrapolated tuple.
        environment["DG_IGR_IGR_PARTIAL_ASSEMBLY"] = "1"
        environment["DG_IGR_IGR_DEVICE_QF"] = "1"
    runner = Runner(attempt, environment)
    lease = Lease(attempt, args.mpi_ranks > 1 or args.device == "cuda")
    if not lease.acquire():
        write_json(
            attempt / "attempt.json",
            {"status": "lease-busy", "finished_at": utc_now(), "argv": sys.argv},
        )
        return 75
    try:
        metrics = execute(args, attempt, runner)
    except Exception as exc:
        write_json(
            attempt / "attempt.json",
            {
                "status": "fail",
                "finished_at": utc_now(),
                "error": str(exc),
                "argv": sys.argv,
            },
        )
        raise
    finally:
        lease.release()

    metrics["lease"] = lease.record
    metrics["attempt"] = str(attempt)
    metrics["finished_at"] = utc_now()
    write_json(attempt / "metrics.json", metrics)
    write_json(
        attempt / "attempt.json",
        {"status": "pass", "finished_at": utc_now(), "argv": sys.argv},
    )
    write_json(attempt / "sha256_manifest.json", sha256_manifest(attempt))
    manifest_hash = hashlib.sha256(
        (attempt / "sha256_manifest.json").read_bytes()
    ).hexdigest()
    comparison = metrics["comparison"]
    print(
        "dgsem-vtk-restart: PASS"
        f" device={args.device} ranks={args.mpi_ranks} order={args.order}"
        f" max_abs={comparison['max_abs']:.6e} rel_l2={comparison['rel_l2']:.6e}"
        f" negatives={len(metrics['negative'])}"
        f" metrics={attempt / 'metrics.json'}"
        f" sha256_manifest={manifest_hash}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - preserve and report the full attempt.
        print(f"dgsem-vtk-restart: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
