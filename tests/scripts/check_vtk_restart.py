#!/usr/bin/env python3
"""Exercise production VTK restart through the cylinder executable."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
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


TYPE_FORMATS = {"Float64": "d", "Int32": "i", "UInt8": "B"}


def base64_chars(byte_count: int) -> int:
    return ((4 * byte_count // 3) + 3) & ~3


def decode_binary(root: ET.Element, array: ET.Element) -> bytes:
    encoded = "".join((array.text or "").split())
    endian = "<" if root.get("byte_order", "LittleEndian") == "LittleEndian" else ">"
    header_type = root.get("header_type", "UInt32")
    header_code = "I" if header_type == "UInt32" else "Q"
    header_size = struct.calcsize(header_code)
    compressor = root.get("compressor")
    if compressor:
        first_chars = base64_chars(header_size)
        first = base64.b64decode(encoded[:first_chars], validate=True)
        (blocks,) = struct.unpack(f"{endian}{header_code}", first)
        header_count = 3 + blocks
        header_chars = base64_chars(header_count * header_size)
        header = base64.b64decode(encoded[:header_chars], validate=True)
        values = struct.unpack(f"{endian}{header_count}{header_code}", header)
        block_size, final_size = values[1:3]
        sizes = values[3:]
        compressed = base64.b64decode(encoded[header_chars:], validate=True)
        result = bytearray()
        offset = 0
        for index, size in enumerate(sizes):
            block = zlib.decompress(compressed[offset : offset + size])
            expected = final_size if index == blocks - 1 and final_size else block_size
            if len(block) != expected:
                raise RuntimeError("compressed VTK block has the wrong size")
            result.extend(block)
            offset += size
        if offset != len(compressed):
            raise RuntimeError("compressed VTK payload has trailing bytes")
        return bytes(result)

    header_chars = base64_chars(header_size)
    header = base64.b64decode(encoded[:header_chars], validate=True)
    (size,) = struct.unpack(f"{endian}{header_code}", header)
    payload = base64.b64decode(encoded[header_chars:], validate=True)
    if len(payload) != size:
        raise RuntimeError("binary VTK payload has the wrong size")
    return payload


def values(root: ET.Element, array: ET.Element) -> list[float | int]:
    vtk_type = array.get("type")
    if vtk_type not in TYPE_FORMATS:
        raise RuntimeError(f"unsupported test VTK type {vtk_type}")
    code = TYPE_FORMATS[vtk_type]
    if array.get("format", "ascii").lower() == "ascii":
        caster = float if code == "d" else int
        return [caster(token) for token in (array.text or "").split()]
    raw = decode_binary(root, array)
    endian = "<" if root.get("byte_order", "LittleEndian") == "LittleEndian" else ">"
    return [item[0] for item in struct.iter_unpack(f"{endian}{code}", raw)]


def encode_values(items: list[float | int], vtk_type: str, endian: str) -> bytes:
    code = TYPE_FORMATS[vtk_type]
    return b"".join(struct.pack(f"{endian}{code}", item) for item in items)


def binary_text(raw: bytes, endian: str) -> str:
    return base64.b64encode(struct.pack(f"{endian}I", len(raw))).decode() + base64.b64encode(raw).decode()


def rewrite_vtu(path: Path, mode: str) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    arrays = list(root.findall(".//DataArray"))
    decoded = [(array, values(root, array)) for array in arrays]
    if mode == "ascii":
        root.attrib.pop("compressor", None)
        root.attrib.pop("header_type", None)
        for array, items in decoded:
            array.set("format", "ascii")
            if array.get("type") == "Float64":
                array.text = "\n" + " ".join(f"{float(item):.17g}" for item in items) + "\n"
            else:
                array.text = "\n" + " ".join(str(int(item)) for item in items) + "\n"
    elif mode == "big-endian":
        root.set("byte_order", "BigEndian")
        root.attrib.pop("compressor", None)
        root.attrib.pop("header_type", None)
        for array, items in decoded:
            raw = encode_values(items, array.get("type", ""), ">")
            array.set("format", "binary")
            array.text = "\n" + binary_text(raw, ">") + "\n"
    elif mode == "compressed":
        root.set("byte_order", "LittleEndian")
        root.set("compressor", "vtkZLibDataCompressor")
        root.attrib.pop("header_type", None)
        for array, items in decoded:
            raw = encode_values(items, array.get("type", ""), "<")
            compressed = zlib.compress(raw)
            header = struct.pack("<4I", 1, len(raw), len(raw), len(compressed))
            array.set("format", "binary")
            array.text = "\n" + base64.b64encode(header).decode() + base64.b64encode(compressed).decode() + "\n"
    else:
        raise RuntimeError(f"unknown rewrite mode {mode}")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def require(pattern: str, text: str, label: str) -> re.Match[str]:
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"missing {label}: /{pattern}/")
    return match


def run(command: list[str], env: dict[str, str], log: Path, succeed: bool = True) -> str:
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
    if succeed != (completed.returncode == 0):
        expectation = "success" if succeed else "failure"
        raise RuntimeError(
            f"expected {expectation}, got rc={completed.returncode}: {log}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def dataset_pvd(root: Path) -> Path:
    paths = list(root.rglob("*.pvd"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one PVD below {root}, found {len(paths)}")
    return paths[0]


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def pvd_entries(path: Path) -> list[tuple[int, float, Path]]:
    root = ET.parse(path).getroot()
    result = []
    for item in root.findall(".//DataSet"):
        piece = (path.parent / item.get("file", "")).resolve()
        cycle_match = re.fullmatch(r"Cycle(\d+)/data\.pvtu", item.get("file", ""))
        if not cycle_match:
            raise RuntimeError(f"unexpected PVD entry {item.attrib}")
        result.append((int(cycle_match.group(1)), float(item.get("timestep", "nan")), piece))
    return result


def piece_paths(pvtu: Path) -> list[Path]:
    root = ET.parse(pvtu).getroot()
    return [(pvtu.parent / piece.get("Source", "")).resolve() for piece in root.findall(".//Piece")]


def conservative_values(path: Path) -> dict[str, list[float | int]]:
    root = ET.parse(path).getroot()
    result = {}
    for name in ("density", "momentum", "energy"):
        array = root.find(f".//DataArray[@Name='{name}']")
        if array is None:
            raise RuntimeError(f"{path} has no {name} array")
        result[name] = values(root, array)
    return result


def compare_final(
    control_pvtu: Path, restart_pvtu: Path, device: str
) -> tuple[float, float]:
    control = piece_paths(control_pvtu)
    restart = piece_paths(restart_pvtu)
    if len(control) != len(restart):
        raise RuntimeError("control/restart piece counts differ")
    max_abs = 0.0
    diff_sq = 0.0
    ref_sq = 0.0
    for control_piece, restart_piece in zip(control, restart):
        lhs = conservative_values(control_piece)
        rhs = conservative_values(restart_piece)
        for name in lhs:
            if len(lhs[name]) != len(rhs[name]):
                raise RuntimeError(f"{name} value counts differ")
            for expected, found in zip(lhs[name], rhs[name]):
                diff = float(found) - float(expected)
                max_abs = max(max_abs, abs(diff))
                diff_sq += diff * diff
                ref_sq += float(expected) * float(expected)
    rel_l2 = math.sqrt(diff_sq / max(ref_sq, 1.0e-300))
    max_limit = 2.0e-8 if device == "cuda" else 5.0e-10
    rel_limit = 1.0e-9 if device == "cuda" else 2.0e-11
    if max_abs > max_limit or rel_l2 > rel_limit:
        raise RuntimeError(
            f"split-run drift too large: max_abs={max_abs:.6e} rel_l2={rel_l2:.6e}"
        )
    return max_abs, rel_l2


def manifest_checks(
    pvd: Path,
    ranks: int,
    time_integrator: str,
    invariant_domain_flux: bool,
    zhang_shu_limiter: bool,
    zhang_shu_vtk_points: bool,
) -> None:
    entries = pvd_entries(pvd)
    if [(cycle, time) for cycle, time, _ in entries] != [
        (0, 0.0),
        (1, 1.0e-4),
        (2, 2.0e-4),
    ]:
        raise RuntimeError(f"unexpected control PVD entries: {entries}")
    for cycle, time, pvtu in entries:
        manifest_path = pvtu.parent / "restart.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 4,
            "complete": True,
            "cycle": cycle,
            "time": time,
            "step": cycle,
            "ranks": ranks,
            "time_integrator": time_integrator,
            "igr_invariant_domain_flux": invariant_domain_flux,
            "zhang_shu_limiter": zhang_shu_limiter,
            "zhang_shu_vtk_points": zhang_shu_vtk_points,
            "rho_floor": 5.0e-7,
            "pressure_floor": 1.0e-6,
            "static_source_present": False,
            "static_source_fingerprint": "none",
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"manifest {manifest_path} has {key}={manifest.get(key)!r}, expected {value!r}")
        if manifest.get("piece_pattern") != "proc%06d.vtu":
            raise RuntimeError(f"manifest {manifest_path} lacks the rank-piece contract")
        local_elements = manifest.get("local_elements")
        if not isinstance(local_elements, list) or len(local_elements) != ranks or sum(local_elements) != 1322:
            raise RuntimeError(f"manifest {manifest_path} has unexpected per-rank elements")
        if manifest.get("data_type") != "Float64" or manifest.get("vtk_format") != "binary":
            raise RuntimeError(f"manifest {manifest_path} has the wrong VTK precision/format")
        if manifest.get("compression") != "none" or manifest.get("compression_level") != 0:
            raise RuntimeError(f"manifest {manifest_path} has the wrong compression contract")
        if manifest.get("next_cycle") != cycle + 1 or manifest.get("next_save_time") != time + 1.0e-4:
            raise RuntimeError(f"manifest {manifest_path} has the wrong continuation metadata")


def base_command(
    exe: Path,
    device: str,
    ranks: int,
    mpiexec: str,
    np_flag: str,
    order: int,
    with_igr: bool,
    cfl_driven: bool,
    time_integrator: str,
    invariant_domain_flux: bool,
    zhang_shu_vtk_points: bool,
) -> list[str]:
    linear_rtol = "1.0e-5" if device == "cuda" else "1.0e-8"
    app = [
        str(exe), "--device", device, "-rs", "0", "-rp", "0", "-o", str(order),
        "-tf", "2.0e-4", "-dt", "-0.01" if cfl_driven else "1.0e-4",
        "-c", "0.3" if cfl_driven else "0.0", "-dg", "dg", "-as", "0",
        "-no-glvis", "-vtk", "-dt_save", "1.0e-4",
        "-no-vtk-h1", "-no-vtk-l2", "-no-vtk-rank", "-no-vtk-zs",
        "-no-save-state", "-iprint", "0", "--time-integrator",
        time_integrator,
    ]
    if with_igr:
        app.extend([
            "-ca", "1.0", "-ial", "max", "-is", "L2", "-ig", "BrokenL2",
            "-ils", "minres", "-ipc", "chebyshev", "-irtol", linear_rtol,
            "-imaxit", "1000",
            "-iexface",
        ])
        if invariant_domain_flux:
            app.append("--igr-invariant-domain-flux")
    if zhang_shu_vtk_points:
        app.append("--zhang-shu-vtk-points")
    return [mpiexec, np_flag, str(ranks), *app] if ranks > 1 else app


def with_output(env: dict[str, str], root: Path) -> dict[str, str]:
    result = dict(env)
    result["DG_IGR_VTK_PREFIX_PATH"] = str(root)
    return result


def failure(command: list[str], env: dict[str, str], root: Path, name: str, pattern: str) -> None:
    target = root / name
    text = run(command, with_output(env, target / "vtk"), target / "run.log", succeed=False)
    require(pattern, text, name)
    if list((target / "vtk").rglob("*.vtu")):
        raise RuntimeError(f"{name} wrote VTK output before rejection")


def variant_copy(source_pvd: Path, root: Path, cycle: int = 1) -> Path:
    shutil.copytree(source_pvd.parent, root)
    copied_pvd = root / source_pvd.name
    selected = next(item for item in pvd_entries(copied_pvd) if item[0] == cycle)
    (selected[2].parent / "restart.json").unlink(missing_ok=True)
    return copied_pvd


def run_format_controls(base: list[str], env: dict[str, str], source_pvd: Path, root: Path) -> None:
    for mode in ("ascii", "big-endian", "compressed"):
        case_root = root / mode
        pvd = variant_copy(source_pvd, case_root / "source")
        selected = next(item for item in pvd_entries(pvd) if item[0] == 1)
        for piece in piece_paths(selected[2]):
            rewrite_vtu(piece, mode)
        if mode == "big-endian":
            pvtu_tree = ET.parse(selected[2])
            pvtu_tree.getroot().set("byte_order", "BigEndian")
            pvtu_tree.write(selected[2], encoding="utf-8", xml_declaration=True)
        command = [*base, "--restart-vtk", str(pvd), "--restart-cycle", "1", "--restart-step", "1"]
        text = run(command, with_output(env, case_root / "vtk"), case_root / "run.log")
        require(r"reconstruction_max_abs=([0-9.eE+-]+)", text, f"{mode} reconstruction")


def run_negative_controls(
    base: list[str],
    env: dict[str, str],
    source_pvd: Path,
    root: Path,
    time_integrator: str,
    invariant_domain_flux: bool,
    with_igr: bool,
) -> None:
    restart = [*base, "--restart-vtk", str(source_pvd), "--restart-cycle", "1"]

    def incompatible_manifest(name: str, key: str, value: object) -> None:
        source_root = root / f"{name}-source"
        shutil.copytree(source_pvd.parent, source_root)
        pvd = source_root / source_pvd.name
        selected = next(item for item in pvd_entries(pvd) if item[0] == 1)
        manifest_path = selected[2].parent / "restart.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest[key] = value
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        command = [*base, "--restart-vtk", str(pvd), "--restart-cycle", "1"]
        failure(
            command,
            env,
            root,
            name,
            r"manifest (?:static source )?is incompatible",
        )

    source_hashes = tree_hashes(source_pvd.parent)
    same_prefix = source_pvd.parent.parent
    same_prefix_command = list(restart)
    same_prefix_command[same_prefix_command.index("-no-save-state")] = "-save-state"
    text = run(same_prefix_command, with_output(env, same_prefix), root / "same-prefix" / "run.log")
    require(r"output_mode=new", text, "same-prefix non-destructive mode")
    if tree_hashes(source_pvd.parent) != source_hashes:
        raise RuntimeError("same-prefix restart modified the source dataset")
    restart_dirs = list(same_prefix.glob(f"{source_pvd.stem}_restart_c1_s1"))
    if (
        len(restart_dirs) != 1
        or not (restart_dirs[0] / "run_metadata.json").is_file()
        or not (restart_dirs[0] / "glvis_state").is_dir()
    ):
        raise RuntimeError("same-prefix restart did not isolate output metadata")

    tf_equal = list(restart)
    tf_equal[tf_equal.index("-tf") + 1] = "1.0e-4"
    failure(tf_equal, env, root, "tf-not-forward", r"must be greater than the restart time")

    missing_cycle = [*base, "--restart-vtk", str(source_pvd), "--restart-cycle", "99"]
    failure(missing_cycle, env, root, "missing-cycle", r"is not present in the PVD")

    wrong_order = list(restart)
    wrong_order[wrong_order.index("-o") + 1] = "2"
    failure(wrong_order, env, root, "wrong-order", r"manifest is incompatible")

    wrong_refinement = list(restart)
    wrong_refinement[wrong_refinement.index("-rp") + 1] = "1"
    failure(wrong_refinement, env, root, "wrong-refinement", r"manifest is incompatible")

    incompatible_manifest("wrong-case", "case", "not_the_active_case")
    incompatible_manifest("wrong-dimension", "dimension", 3)
    incompatible_manifest("wrong-equations", "equations", 5)
    incompatible_manifest("wrong-basis", "basis", "GaussLobatto")
    incompatible_manifest("wrong-mesh", "mesh_fingerprint", "0000000000000000")
    incompatible_manifest(
        "wrong-effective-zhang-shu",
        "zhang_shu_limiter",
        not with_igr,
    )
    incompatible_manifest(
        "wrong-zhang-shu-vtk-points",
        "zhang_shu_vtk_points",
        "--zhang-shu-vtk-points" not in base,
    )
    incompatible_manifest("wrong-rho-floor", "rho_floor", 1.0e-8)
    incompatible_manifest("wrong-pressure-floor", "pressure_floor", 2.0e-8)
    incompatible_manifest("wrong-static-source-presence", "static_source_present", True)
    incompatible_manifest(
        "wrong-static-source-fingerprint",
        "static_source_fingerprint",
        "0123456789abcdef",
    )

    wrong_time_integrator = list(restart)
    time_index = wrong_time_integrator.index("--time-integrator") + 1
    wrong_time_integrator[time_index] = (
        "rk3ssp" if time_integrator == "ssprk104" else "ssprk104"
    )
    failure(
        wrong_time_integrator,
        env,
        root,
        "wrong-time-integrator",
        r"manifest is incompatible",
    )

    if with_igr:
        wrong_invariant_flux = list(restart)
        invariant_flag = "--igr-invariant-domain-flux"
        if invariant_domain_flux:
            wrong_invariant_flux.remove(invariant_flag)
        else:
            wrong_invariant_flux.append(invariant_flag)
        failure(
            wrong_invariant_flux,
            env,
            root,
            "wrong-invariant-domain-flux",
            r"manifest is incompatible",
        )

    wrong_backend = list(restart)
    wrong_backend[wrong_backend.index("-dg") + 1] = "dgsem"
    failure(
        wrong_backend,
        env,
        root,
        "wrong-backend",
        r"manifest is incompatible|supports the L2 DG backend only|requires the ordinary DG backend",
    )

    wrong_alpha = list(restart)
    wrong_alpha[wrong_alpha.index("-ca") + 1] = "2.0"
    failure(wrong_alpha, env, root, "wrong-alpha", r"manifest is incompatible")

    wrong_limiter = [*restart, "--zhang-shu-points", "quadrature"]
    failure(
        wrong_limiter,
        env,
        root,
        "wrong-limiter-points",
        r"manifest is incompatible|requires --zhang-shu-points consumer",
    )

    no_exact_face = [item for item in restart if item != "-iexface"]
    failure(no_exact_face, env, root, "missing-exact-face", r"requires exact DG faces")

    thinned_sigma = [*restart, "--igr-sigma-cadence", "every-step"]
    failure(thinned_sigma, env, root, "thinned-sigma", r"requires every-stage")

    changed_solver = list(restart)
    changed_solver[changed_solver.index("-irtol") + 1] = "1.0e-6"
    changed_solver[changed_solver.index("-imaxit") + 1] = "1200"
    text = run(changed_solver, with_output(env, root / "solver-change" / "vtk"), root / "solver-change" / "run.log")
    require(r"independent_residual_certified=1", text, "allowed solver-control change")

    no_path = [*base, "--restart-cycle", "1"]
    failure(no_path, env, root, "restart-option-without-path", r"require --restart-vtk")

    nonmonotone_root = root / "nonmonotone-source"
    shutil.copytree(source_pvd.parent, nonmonotone_root)
    nonmonotone_pvd = nonmonotone_root / source_pvd.name
    pvd_tree = ET.parse(nonmonotone_pvd)
    pvd_items = pvd_tree.getroot().findall(".//DataSet")
    pvd_items[-1].set("timestep", "0.00005")
    pvd_tree.write(nonmonotone_pvd, encoding="utf-8", xml_declaration=True)
    command = [*base, "--restart-vtk", str(nonmonotone_pvd), "--restart-cycle", "1"]
    failure(command, env, root, "nonmonotone-pvd", r"must be strictly increasing")

    duplicate_root = root / "duplicate-cycle-source"
    shutil.copytree(source_pvd.parent, duplicate_root)
    duplicate_pvd = duplicate_root / source_pvd.name
    pvd_tree = ET.parse(duplicate_pvd)
    pvd_items = pvd_tree.getroot().findall(".//DataSet")
    pvd_items[1].set("file", pvd_items[0].get("file", ""))
    pvd_tree.write(duplicate_pvd, encoding="utf-8", xml_declaration=True)
    command = [*base, "--restart-vtk", str(duplicate_pvd), "--restart-cycle", "1"]
    failure(command, env, root, "duplicate-cycle", r"must be strictly increasing")

    incomplete_root = root / "incomplete-source"
    shutil.copytree(source_pvd.parent, incomplete_root)
    incomplete = incomplete_root / source_pvd.name
    selected = next(item for item in pvd_entries(incomplete) if item[0] == 1)
    manifest_path = selected[2].parent / "restart.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    command = [*base, "--restart-vtk", str(incomplete), "--restart-cycle", "1"]
    failure(command, env, root, "incomplete-manifest", r"manifest is incomplete")

    fallback_root = root / "latest-incomplete-source"
    shutil.copytree(source_pvd.parent, fallback_root)
    fallback_pvd = fallback_root / source_pvd.name
    latest_entry = next(item for item in pvd_entries(fallback_pvd) if item[0] == 2)
    latest_manifest_path = latest_entry[2].parent / "restart.json"
    latest_manifest = json.loads(latest_manifest_path.read_text(encoding="utf-8"))
    latest_manifest["complete"] = False
    latest_manifest_path.write_text(json.dumps(latest_manifest), encoding="utf-8")
    command = [*base, "--restart-vtk", str(fallback_pvd)]
    text = run(command, with_output(env, root / "latest-complete-fallback" / "vtk"), root / "latest-complete-fallback" / "run.log")
    require(r"cycle=1 time=0\.0001 step=1 legacy=0", text, "latest complete fallback")

    schema2_root = root / "schema2-source"
    shutil.copytree(source_pvd.parent, schema2_root)
    schema2_pvd = schema2_root / source_pvd.name
    schema2_entry = next(item for item in pvd_entries(schema2_pvd) if item[0] == 1)
    schema2_path = schema2_entry[2].parent / "restart.json"
    schema2 = json.loads(schema2_path.read_text(encoding="utf-8"))
    schema2["schema_version"] = 2
    schema2.pop("static_source_present", None)
    schema2.pop("static_source_fingerprint", None)
    schema2_path.write_text(json.dumps(schema2), encoding="utf-8")
    schema2_command = [
        *base,
        "--restart-vtk",
        str(schema2_pvd),
        "--restart-cycle",
        "1",
    ]
    text = run(
        schema2_command,
        with_output(env, root / "schema2-compatible" / "vtk"),
        root / "schema2-compatible" / "run.log",
    )
    require(r"legacy=0", text, "source-free schema-2 compatibility")

    schema1_root = root / "schema1-source"
    shutil.copytree(source_pvd.parent, schema1_root)
    schema1_pvd = schema1_root / source_pvd.name
    schema1_entry = next(item for item in pvd_entries(schema1_pvd) if item[0] == 1)
    schema1_path = schema1_entry[2].parent / "restart.json"
    schema1 = json.loads(schema1_path.read_text(encoding="utf-8"))
    schema1["schema_version"] = 1
    for key in (
        "time_integrator",
        "igr_invariant_domain_flux",
        "zhang_shu_limiter",
        "zhang_shu_vtk_points",
        "rho_floor",
        "pressure_floor",
        "static_source_present",
        "static_source_fingerprint",
    ):
        schema1.pop(key, None)
    schema1_path.write_text(json.dumps(schema1), encoding="utf-8")
    schema1_command = [
        *base,
        "--restart-vtk",
        str(schema1_pvd),
        "--restart-cycle",
        "1",
    ]
    if (
        time_integrator == "rk3ssp"
        and not invariant_domain_flux
        and "--zhang-shu-vtk-points" not in base
    ):
        text = run(
            schema1_command,
            with_output(env, root / "schema1-compatible" / "vtk"),
            root / "schema1-compatible" / "run.log",
        )
        require(r"legacy=0", text, "schema-1 compatibility")
    else:
        failure(
            schema1_command,
            env,
            root,
            "schema1-new-method",
            r"schema 1 is compatible only with rk3ssp",
        )

    legacy = variant_copy(source_pvd, root / "legacy-source")
    command = [*base, "--restart-vtk", str(legacy), "--restart-cycle", "1"]
    failure(command, env, root, "legacy-missing-step", r"provide --restart-step")
    command.extend(["--restart-step", "1"])
    text = run(command, with_output(env, root / "legacy-explicit-step" / "vtk"), root / "legacy-explicit-step" / "run.log")
    require(r"legacy=1", text, "legacy explicit-step import")

    missing_field = variant_copy(source_pvd, root / "missing-field-source")
    selected = next(item for item in pvd_entries(missing_field) if item[0] == 1)
    pvtu_tree = ET.parse(selected[2])
    density = pvtu_tree.getroot().find(".//PDataArray[@Name='density']")
    if density is None:
        raise RuntimeError("test source has no PVTU density declaration")
    density.set("Name", "missing_density")
    pvtu_tree.write(selected[2], encoding="utf-8", xml_declaration=True)
    command = [*base, "--restart-vtk", str(missing_field), "--restart-cycle", "1", "--restart-step", "1"]
    failure(command, env, root, "missing-density", r"required field declaration is incompatible")

    wrong_components = variant_copy(source_pvd, root / "wrong-components-source")
    selected = next(item for item in pvd_entries(wrong_components) if item[0] == 1)
    pvtu_tree = ET.parse(selected[2])
    momentum = pvtu_tree.getroot().find(".//PDataArray[@Name='momentum']")
    if momentum is None:
        raise RuntimeError("test source has no PVTU momentum declaration")
    momentum.set("NumberOfComponents", "3")
    pvtu_tree.write(selected[2], encoding="utf-8", xml_declaration=True)
    command = [*base, "--restart-vtk", str(wrong_components), "--restart-cycle", "1", "--restart-step", "1"]
    failure(command, env, root, "wrong-components", r"required field declaration is incompatible")

    binary32 = variant_copy(source_pvd, root / "binary32-source")
    selected = next(item for item in pvd_entries(binary32) if item[0] == 1)
    pvtu_tree = ET.parse(selected[2])
    density_decl = pvtu_tree.getroot().find(".//PDataArray[@Name='density']")
    if density_decl is None:
        raise RuntimeError("test source has no PVTU density declaration")
    density_decl.set("type", "Float32")
    pvtu_tree.write(selected[2], encoding="utf-8", xml_declaration=True)
    command = [*base, "--restart-vtk", str(binary32), "--restart-cycle", "1", "--restart-step", "1"]
    failure(command, env, root, "binary32", r"required field declaration is incompatible")

    wrong_ranks = variant_copy(source_pvd, root / "wrong-ranks-source")
    selected = next(item for item in pvd_entries(wrong_ranks) if item[0] == 1)
    pvtu_tree = ET.parse(selected[2])
    grid = pvtu_tree.getroot().find(".//PUnstructuredGrid")
    if grid is None:
        raise RuntimeError("test source has no PUnstructuredGrid")
    ET.SubElement(grid, "Piece", {"Source": "proc000001.vtu"})
    pvtu_tree.write(selected[2], encoding="utf-8", xml_declaration=True)
    command = [*base, "--restart-vtk", str(wrong_ranks), "--restart-cycle", "1", "--restart-step", "1"]
    failure(command, env, root, "wrong-ranks", r"piece count does not match")

    missing_piece = variant_copy(source_pvd, root / "missing-piece-source")
    selected = next(item for item in pvd_entries(missing_piece) if item[0] == 1)
    piece_paths(selected[2])[0].unlink()
    command = [*base, "--restart-vtk", str(missing_piece), "--restart-cycle", "1", "--restart-step", "1"]
    failure(command, env, root, "missing-piece", r"missing restart rank piece")

    nonfinite = variant_copy(source_pvd, root / "nonfinite-source")
    selected = next(item for item in pvd_entries(nonfinite) if item[0] == 1)
    piece = piece_paths(selected[2])[0]
    tree = ET.parse(piece)
    vtk_root = tree.getroot()
    density = vtk_root.find(".//DataArray[@Name='density']")
    if density is None:
        raise RuntimeError("test source has no density array")
    density_values = values(vtk_root, density)
    density_values[0] = math.nan
    density.text = "\n" + binary_text(encode_values(density_values, "Float64", "<"), "<") + "\n"
    tree.write(piece, encoding="utf-8", xml_declaration=True)
    command = [*base, "--restart-vtk", str(nonfinite), "--restart-cycle", "1", "--restart-step", "1"]
    failure(command, env, root, "nonfinite-state", r"non-finite")

    inadmissible = variant_copy(source_pvd, root / "inadmissible-source")
    selected = next(item for item in pvd_entries(inadmissible) if item[0] == 1)
    piece = piece_paths(selected[2])[0]
    tree = ET.parse(piece)
    vtk_root = tree.getroot()
    energy = vtk_root.find(".//DataArray[@Name='energy']")
    if energy is None:
        raise RuntimeError("test source has no energy array")
    energy_values = values(vtk_root, energy)
    energy_values[0] = 0.0
    energy.text = "\n" + binary_text(encode_values(energy_values, "Float64", "<"), "<") + "\n"
    tree.write(piece, encoding="utf-8", xml_declaration=True)
    command = [*base, "--restart-vtk", str(inadmissible), "--restart-cycle", "1", "--restart-step", "1"]
    failure(command, env, root, "inadmissible-state", r"not consumer-point admissible")

    corrupt_base64 = variant_copy(source_pvd, root / "corrupt-base64-source")
    selected = next(item for item in pvd_entries(corrupt_base64) if item[0] == 1)
    piece = piece_paths(selected[2])[0]
    tree = ET.parse(piece)
    density = tree.getroot().find(".//DataArray[@Name='density']")
    if density is None:
        raise RuntimeError("test source has no density array")
    density.text = "!!!!"
    tree.write(piece, encoding="utf-8", xml_declaration=True)
    command = [*base, "--restart-vtk", str(corrupt_base64), "--restart-cycle", "1", "--restart-step", "1"]
    failure(command, env, root, "corrupt-base64", r"base64|binary header|byte count")

    corrupt_zlib = variant_copy(source_pvd, root / "corrupt-zlib-source")
    selected = next(item for item in pvd_entries(corrupt_zlib) if item[0] == 1)
    piece = piece_paths(selected[2])[0]
    rewrite_vtu(piece, "compressed")
    tree = ET.parse(piece)
    density = tree.getroot().find(".//DataArray[@Name='density']")
    if density is None:
        raise RuntimeError("test source has no density array")
    encoded = "".join((density.text or "").split())
    header_chars = 4 * ((4 * 4 + 2) // 3)
    compressed = bytearray(base64.b64decode(encoded[header_chars:]))
    compressed[len(compressed) // 2] ^= 0xFF
    density.text = "\n" + encoded[:header_chars] + base64.b64encode(compressed).decode() + "\n"
    tree.write(piece, encoding="utf-8", xml_declaration=True)
    command = [*base, "--restart-vtk", str(corrupt_zlib), "--restart-cycle", "1", "--restart-step", "1"]
    failure(command, env, root, "corrupt-zlib", r"zlib failed")

    truncated = variant_copy(source_pvd, root / "truncated-source")
    selected = next(item for item in pvd_entries(truncated) if item[0] == 1)
    piece = piece_paths(selected[2])[0]
    tree = ET.parse(piece)
    density = tree.getroot().find(".//DataArray[@Name='density']")
    if density is None:
        raise RuntimeError("test source has no density array")
    encoded = "".join((density.text or "").split())
    density.text = encoded[:-4]
    tree.write(piece, encoding="utf-8", xml_declaration=True)
    command = [*base, "--restart-vtk", str(truncated), "--restart-cycle", "1", "--restart-step", "1"]
    failure(command, env, root, "truncated-array", r"payload size|truncated")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--order", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--mpi-numproc-flag", default="-np")
    parser.add_argument("--format-controls", action="store_true")
    parser.add_argument("--negative-controls", action="store_true")
    parser.add_argument("--no-igr", action="store_true")
    parser.add_argument("--cfl-driven", action="store_true")
    parser.add_argument(
        "--time-integrator",
        choices=("rk3ssp", "ssprk104"),
        default="rk3ssp",
    )
    parser.add_argument("--invariant-domain-flux", action="store_true")
    parser.add_argument("--zhang-shu-vtk-points", action="store_true")
    args = parser.parse_args()

    if args.no_igr and args.invariant_domain_flux:
        raise RuntimeError("--invariant-domain-flux requires IGR in this harness")

    output = Path(args.out_dir).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    env = os.environ.copy()
    env.update({
        "DG_IGR_IGR_PARTIAL_ASSEMBLY": "1",
        "DG_IGR_IGR_DEVICE_QF": "1",
        "DG_IGR_IGR_CHEB_ORDER": "3",
        "DG_IGR_IGR_CHEB_BASE_OPERATOR": "1",
    })
    if args.cfl_driven:
        env["DG_IGR_TIME_DT_TRACE"] = "1"
    base = base_command(
        Path(args.exe).resolve(),
        args.device,
        args.mpi_ranks,
        args.mpiexec,
        args.mpi_numproc_flag,
        args.order,
        not args.no_igr,
        args.cfl_driven,
        args.time_integrator,
        args.invariant_domain_flux,
        args.zhang_shu_vtk_points,
    )

    control_root = output / "control" / "vtk"
    control_log = run(base, with_output(env, control_root), output / "control" / "run.log")
    source_pvd = dataset_pvd(control_root)
    manifest_checks(
        source_pvd,
        args.mpi_ranks,
        args.time_integrator,
        args.invariant_domain_flux,
        not args.no_igr,
        args.zhang_shu_vtk_points,
    )

    restart_root = output / "restart" / "vtk"
    restart_command = [*base, "--restart-vtk", str(source_pvd), "--restart-cycle", "1"]
    restart_log = run(restart_command, with_output(env, restart_root), output / "restart" / "run.log")
    require(r"cycle=1 time=0\.0001 step=1 legacy=0", restart_log, "restart selection")
    if args.cfl_driven:
        control_dt = require(
            r"\[time-loop\] event=nominal-dt stage=update step=1 "
            r"time=0\.0001 dt=([0-9.eE+-]+)",
            control_log,
            "control CFL timestep at checkpoint",
        )
        restart_dt = require(
            r"\[time-loop\] event=nominal-dt stage=initial step=1 "
            r"time=0\.0001 dt=([0-9.eE+-]+)",
            restart_log,
            "restart initial CFL timestep",
        )
        control_value = float(control_dt.group(1))
        restart_value = float(restart_dt.group(1))
        relative_dt = abs(restart_value - control_value) / max(abs(restart_value), 1.0e-300)
        if relative_dt > 5.0e-3:
            raise RuntimeError(
                "restart/control CFL timestep mismatch: "
                f"control={control_value:.17e} restart={restart_value:.17e} "
                f"relative={relative_dt:.6e}"
            )
    if not args.no_igr:
        require(r"independent_residual_certified=1", restart_log, "sigma certificate")
        lifecycle = require(r"prefetch_solves=(\d+) stage0_reuses=(\d+) output_only_solves=(\d+) prefetch_invalidations=(\d+)", restart_log, "restart sigma lifecycle")
        if tuple(map(int, lifecycle.groups())) != (1, 1, 1, 0):
            raise RuntimeError(f"unexpected restart sigma lifecycle {lifecycle.groups()}")

    restart_pvd = dataset_pvd(restart_root)
    restart_entries = pvd_entries(restart_pvd)
    if [(cycle, time) for cycle, time, _ in restart_entries] != [(2, 2.0e-4)]:
        raise RuntimeError(f"restart output duplicated or misnumbered a cycle: {restart_entries}")
    metadata_path = restart_pvd.parent / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("restart", {}).get("cycle") != 1 or metadata.get("performance", {}).get("segment_steps") != 1:
        raise RuntimeError("restart run metadata lacks source or segment provenance")
    control_final = next(item[2] for item in pvd_entries(source_pvd) if item[0] == 2)
    max_abs, rel_l2 = compare_final(
        control_final, restart_entries[0][2], args.device
    )

    latest_root = output / "latest" / "vtk"
    latest = list(base)
    latest[latest.index("-tf") + 1] = "3.0e-4"
    latest.extend(["--restart-vtk", str(source_pvd)])
    latest_log = run(latest, with_output(env, latest_root), output / "latest" / "run.log")
    require(r"cycle=2 time=0\.0002 step=2 legacy=0", latest_log, "latest complete cycle")
    if [(c, t) for c, t, _ in pvd_entries(dataset_pvd(latest_root))] != [(3, 3.0e-4)]:
        raise RuntimeError("latest-cycle restart did not continue at cycle 3")

    if args.format_controls:
        run_format_controls(base, env, source_pvd, output / "formats")
    if args.negative_controls:
        if args.mpi_ranks != 1:
            raise RuntimeError("negative controls are intended for one rank")
        run_negative_controls(
            base,
            env,
            source_pvd,
            output / "negative",
            args.time_integrator,
            args.invariant_domain_flux,
            not args.no_igr,
        )

    if not args.no_igr:
        production_calls = 20 if args.time_integrator == "ssprk104" else 6
        require(
            rf"production_calls={production_calls} "
            rf"production_converged_calls={production_calls}",
            control_log,
            "control solve accounting",
        )
    print(
        "vtk-restart: PASS"
        f" device={args.device} ranks={args.mpi_ranks} order={args.order}"
        f" igr={int(not args.no_igr)}"
        f" cfl={int(args.cfl_driven)}"
        f" time_integrator={args.time_integrator}"
        f" invariant_domain_flux={int(args.invariant_domain_flux)}"
        f" zhang_shu_vtk_points={int(args.zhang_shu_vtk_points)}"
        f" max_abs={max_abs:.6e} rel_l2={rel_l2:.6e}"
        f" formats={int(args.format_controls)} negatives={int(args.negative_controls)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - test driver reports full context.
        print(f"vtk-restart: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
