#!/usr/bin/env python3
"""Bounded MFEM ParaView postprocessor for shock-architecture QoIs.

The program deliberately emits one small JSON object on stdout.  Diagnostics
go to stderr so it can be used directly as shock_architecture_matrix.py's QoI
executable.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET
import zlib


QOI_FIELDS = (
    "shock_position",
    "shock_standoff_distance",
    "shock_profile_l1_error",
    "shock_profile_l2_error",
    "shock_thickness",
    "density_minimum",
    "density_maximum",
    "pressure_minimum",
    "pressure_maximum",
    "conservation_error",
    "boundary_balance_error",
    "sigma_localization",
)

FLOAT_TYPES = {
    "Float32": ("f", 4),
    "Float64": ("d", 8),
}
HEADER_TYPES = {
    "UInt32": ("I", 4),
    "UInt64": ("Q", 8),
}


class ShockQOIError(RuntimeError):
    """A controlled input or scientific-observability failure."""


@dataclass(frozen=True)
class Limits:
    max_file_bytes: int
    max_total_points: int
    max_pieces: int
    max_pvds: int


@dataclass(frozen=True)
class Frame:
    time: float
    path: Path
    collection: Path


@dataclass
class Fields:
    points: list[tuple[float, float, float]]
    density: list[float]
    momentum: list[tuple[float, ...]]
    energy: list[float]
    sigma: list[float]
    # Retained only as optional compatibility data. Reported pressure QoIs are
    # always reconstructed independently from density, momentum, and energy.
    pressure: list[float] | None = None


@dataclass(frozen=True)
class Profile:
    x: tuple[float, ...]
    density: tuple[float, ...]


@dataclass(frozen=True)
class ReferenceProfile:
    """A profile plus the scientific identity needed to consume it safely."""

    profile: Profile
    case: str
    method: str
    target_time: float
    mesh: str
    mesh_sha256: str
    gamma: float
    extraction: dict[str, float]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> Iterable[ET.Element]:
    return (child for child in element if _local_name(child.tag) == name)


def _descendants(element: ET.Element, name: str) -> Iterable[ET.Element]:
    return (item for item in element.iter() if _local_name(item.tag) == name)


def _only_child(element: ET.Element, name: str, context: str) -> ET.Element:
    found = list(_children(element, name))
    if len(found) != 1:
        raise ShockQOIError(
            f"{context} must contain exactly one {name}, found {len(found)}"
        )
    return found[0]


def _positive_int(value: str, label: str, *, allow_zero: bool = False) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ShockQOIError(f"{label} is not an integer") from error
    if parsed < 0 or (parsed == 0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ShockQOIError(f"{label} must be {qualifier}")
    return parsed


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ShockQOIError(f"{label} is not numeric")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ShockQOIError(f"{label} is not numeric") from error
    if not math.isfinite(parsed):
        raise ShockQOIError(f"{label} must be finite")
    return parsed


def _read_xml(path: Path, limits: Limits) -> ET.Element:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ShockQOIError(f"cannot stat VTK file {path}: {error}") from error
    if size > limits.max_file_bytes:
        raise ShockQOIError(
            f"VTK file {path} is {size} bytes, exceeding the configured bound"
        )
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ShockQOIError(f"cannot read VTK file {path}: {error}") from error
    lowered = raw.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise ShockQOIError(f"DTD/entity declarations are forbidden in {path}")
    try:
        return ET.fromstring(raw)
    except ET.ParseError as error:
        raise ShockQOIError(f"malformed XML in {path}: {error}") from error


def _safe_reference(base: Path, value: str | None, boundary: Path,
                    label: str) -> Path:
    if value is None or not value.strip() or "\x00" in value:
        raise ShockQOIError(f"missing {label} path")
    candidate = (base / value).resolve()
    try:
        candidate.relative_to(boundary)
    except ValueError as error:
        raise ShockQOIError(f"{label} escapes VTK root: {value}") from error
    if not candidate.is_file():
        raise ShockQOIError(f"missing {label} file {candidate}")
    return candidate


def _time_tolerance(target: float, absolute: float, relative: float) -> float:
    return max(absolute, relative * max(1.0, abs(target)))


def select_frame(vtk_root: Path, target_time: float, method: str,
                 limits: Limits, absolute_tolerance: float,
                 relative_tolerance: float) -> Frame:
    """Select the unique PVD frame closest to the requested final time."""
    requested = vtk_root.resolve()
    if requested.is_file():
        if requested.suffix.lower() != ".pvd":
            raise ShockQOIError("--vtk-root file must be an MFEM .pvd collection")
        boundary = requested.parent.resolve()
        pvds = [requested]
    elif requested.is_dir():
        boundary = requested
        pvds = []
        for item in requested.rglob("*.pvd"):
            resolved = item.resolve()
            try:
                resolved.relative_to(boundary)
            except ValueError as error:
                raise ShockQOIError(f"PVD path escapes VTK root: {item}") from error
            pvds.append(resolved)
            if len(pvds) > limits.max_pvds:
                raise ShockQOIError("too many PVD collections below --vtk-root")
    else:
        raise ShockQOIError(f"--vtk-root does not exist: {vtk_root}")
    if not pvds:
        raise ShockQOIError(f"no .pvd collection found below {vtk_root}")

    tolerance = _time_tolerance(
        target_time, absolute_tolerance, relative_tolerance
    )
    candidates: list[Frame] = []
    for pvd in sorted(set(pvds)):
        root = _read_xml(pvd, limits)
        if root.get("type") != "Collection":
            raise ShockQOIError(f"{pvd} is not a VTK Collection")
        timed_datasets: list[tuple[float, int, ET.Element]] = []
        datasets = list(_descendants(root, "DataSet"))
        if len(datasets) > limits.max_pieces * 16:
            raise ShockQOIError(f"too many DataSet entries in {pvd}")
        for index, dataset in enumerate(datasets):
            time = _finite_float(dataset.get("timestep"),
                                 f"timestep in {pvd}")
            timed_datasets.append((time, index, dataset))
        # The harness asks for a fixed-horizon result.  Do not silently select
        # an intermediate target-time frame from a stale, longer collection.
        if (not timed_datasets or
                abs(max(item[0] for item in timed_datasets) - target_time) >
                tolerance):
            continue
        entries: list[tuple[float, int, Path]] = []
        for time, index, dataset in timed_datasets:
            if abs(time - target_time) <= tolerance:
                frame_path = _safe_reference(
                    pvd.parent, dataset.get("file"), boundary, "PVD frame"
                )
                if frame_path.suffix.lower() not in (".pvtu", ".vtu"):
                    raise ShockQOIError(
                        f"target PVD frame is not VTU/PVTU: {frame_path}"
                    )
                entries.append((time, index, frame_path))
        if entries:
            # Prefer the closest representation of target_time.  For repeated
            # final-time saves, the last collection entry is the committed one.
            time, _, path = min(
                entries, key=lambda item: (abs(item[0] - target_time), -item[1])
            )
            candidates.append(Frame(time, path, pvd))

    unique = {item.path: item for item in candidates}
    if not unique:
        raise ShockQOIError(
            f"no PVD frame matches target time {target_time:.17g} "
            f"within tolerance {tolerance:.3g}"
        )
    if len(unique) > 1:
        # A method-labelled collection is a safe disambiguator, but never pick
        # a frame merely by modification time or directory traversal order.
        method_key = "".join(char for char in method.casefold() if char.isalnum())
        labelled = [
            item for item in unique.values()
            if method_key and method_key in "".join(
                char for char in str(item.collection).casefold()
                if char.isalnum()
            )
        ]
        if len(labelled) == 1:
            return labelled[0]
        listed = ", ".join(str(item.path) for item in unique.values())
        raise ShockQOIError(f"ambiguous target-time frames: {listed}")
    return next(iter(unique.values()))


def _base64_chars(byte_count: int) -> int:
    return ((4 * byte_count // 3) + 3) & ~3


def _decode_b64(value: str, label: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ShockQOIError(f"invalid base64 in {label}") from error


def _decode_binary(root: ET.Element, array: ET.Element, expected_bytes: int,
                   label: str) -> bytes:
    encoded = "".join((array.text or "").split())
    if not encoded:
        raise ShockQOIError(f"empty binary DataArray for {label}")
    byte_order = root.get("byte_order", "LittleEndian")
    if byte_order == "LittleEndian":
        endian = "<"
    elif byte_order == "BigEndian":
        endian = ">"
    else:
        raise ShockQOIError(f"unsupported VTK byte_order {byte_order!r}")
    header_name = root.get("header_type", "UInt32")
    if header_name not in HEADER_TYPES:
        raise ShockQOIError(f"unsupported VTK header_type {header_name!r}")
    header_code, header_size = HEADER_TYPES[header_name]
    first_chars = _base64_chars(header_size)
    if len(encoded) < first_chars:
        raise ShockQOIError(f"truncated binary header for {label}")
    first = _decode_b64(encoded[:first_chars], label)
    # In compressed MFEM output the first integer is a prefix of one base64
    # encoding of the complete multi-integer header.  A character-aligned
    # prefix can therefore decode to one or two extra bytes.
    if len(first) < header_size:
        raise ShockQOIError(f"wrong binary header size for {label}")
    (first_value,) = struct.unpack(f"{endian}{header_code}",
                                   first[:header_size])

    compressor = root.get("compressor")
    if compressor is None:
        payload_size = first_value
        if payload_size != expected_bytes:
            raise ShockQOIError(
                f"{label} advertises {payload_size} bytes; expected {expected_bytes}"
            )
        payload = _decode_b64(encoded[first_chars:], label)
        if len(payload) != payload_size:
            raise ShockQOIError(f"binary payload has the wrong size for {label}")
        return payload
    if compressor != "vtkZLibDataCompressor":
        raise ShockQOIError(f"unsupported VTK compressor {compressor!r}")

    blocks = first_value
    if blocks <= 0 or blocks > max(1, expected_bytes):
        raise ShockQOIError(f"invalid compressed block count for {label}")
    header_count = 3 + blocks
    header_bytes = header_count * header_size
    header_chars = _base64_chars(header_bytes)
    if len(encoded) < header_chars:
        raise ShockQOIError(f"truncated compressed header for {label}")
    header = _decode_b64(encoded[:header_chars], label)
    if len(header) != header_bytes:
        raise ShockQOIError(f"wrong compressed header size for {label}")
    values = struct.unpack(f"{endian}{header_count}{header_code}", header)
    block_bytes, last_bytes = values[1:3]
    compressed_sizes = values[3:]
    if block_bytes <= 0 or any(size <= 0 for size in compressed_sizes):
        raise ShockQOIError(f"invalid compressed block sizes for {label}")
    final_bytes = last_bytes if last_bytes else block_bytes
    advertised = (blocks - 1) * block_bytes + final_bytes
    if advertised != expected_bytes:
        raise ShockQOIError(
            f"{label} compressed header advertises {advertised} bytes; "
            f"expected {expected_bytes}"
        )
    compressed = _decode_b64(encoded[header_chars:], label)
    if sum(compressed_sizes) != len(compressed):
        raise ShockQOIError(f"compressed payload has the wrong size for {label}")
    result = bytearray()
    offset = 0
    for index, size in enumerate(compressed_sizes):
        expected = final_bytes if index == blocks - 1 else block_bytes
        chunk = compressed[offset:offset + size]
        inflater = zlib.decompressobj()
        try:
            decoded = inflater.decompress(chunk, expected + 1)
        except zlib.error as error:
            raise ShockQOIError(f"invalid zlib block for {label}") from error
        if (len(decoded) != expected or not inflater.eof or
                inflater.unconsumed_tail or inflater.unused_data):
            raise ShockQOIError(f"compressed VTK block has the wrong size for {label}")
        result.extend(decoded)
        offset += size
    if offset != len(compressed) or len(result) != expected_bytes:
        raise ShockQOIError(f"compressed VTK payload is inconsistent for {label}")
    return bytes(result)


def _array_values(root: ET.Element, array: ET.Element, expected_count: int,
                  label: str) -> list[float]:
    vtk_type = array.get("type")
    if vtk_type not in FLOAT_TYPES:
        raise ShockQOIError(f"{label} must use Float32 or Float64")
    code, item_size = FLOAT_TYPES[vtk_type]
    data_format = array.get("format", "ascii").casefold()
    if data_format == "ascii":
        tokens = (array.text or "").split()
        if len(tokens) != expected_count:
            raise ShockQOIError(
                f"{label} contains {len(tokens)} values; expected {expected_count}"
            )
        values = [_finite_float(token, label) for token in tokens]
    elif data_format == "binary":
        raw = _decode_binary(root, array, expected_count * item_size, label)
        byte_order = root.get("byte_order", "LittleEndian")
        endian = "<" if byte_order == "LittleEndian" else ">"
        values = [item[0] for item in struct.iter_unpack(f"{endian}{code}", raw)]
    else:
        raise ShockQOIError(f"unsupported DataArray format {data_format!r} for {label}")
    if len(values) != expected_count or any(not math.isfinite(item) for item in values):
        raise ShockQOIError(f"{label} contains a wrong count or non-finite value")
    return values


def _component_count(array: ET.Element, label: str) -> int:
    return _positive_int(array.get("NumberOfComponents", "1"),
                         f"{label} NumberOfComponents")


def _named_arrays(point_data: ET.Element, path: Path) -> dict[str, ET.Element]:
    result: dict[str, ET.Element] = {}
    for array in _children(point_data, "DataArray"):
        name = (array.get("Name") or "").strip().casefold()
        if not name:
            raise ShockQOIError(f"unnamed PointData array in {path}")
        if name in result:
            raise ShockQOIError(f"duplicate PointData array {name!r} in {path}")
        result[name] = array
    return result


def _required_array(arrays: dict[str, ET.Element], aliases: Sequence[str],
                    path: Path) -> ET.Element:
    found = [arrays[name] for name in aliases if name in arrays]
    if len(found) != 1:
        raise ShockQOIError(
            f"{path} requires exactly one PointData field among {tuple(aliases)}"
        )
    return found[0]


def _optional_array(arrays: dict[str, ET.Element], aliases: Sequence[str],
                    path: Path) -> ET.Element | None:
    found = [arrays[name] for name in aliases if name in arrays]
    if len(found) > 1:
        raise ShockQOIError(
            f"{path} contains multiple PointData fields among {tuple(aliases)}"
        )
    return found[0] if found else None


def _parse_vtu(path: Path, limits: Limits) -> Fields:
    root = _read_xml(path, limits)
    if root.get("type") != "UnstructuredGrid":
        raise ShockQOIError(f"{path} is not a VTK UnstructuredGrid")
    pieces = list(_descendants(root, "Piece"))
    if len(pieces) != 1:
        raise ShockQOIError(f"{path} must contain exactly one Piece")
    piece = pieces[0]
    count = _positive_int(piece.get("NumberOfPoints", ""),
                          f"NumberOfPoints in {path}")
    if count > limits.max_total_points:
        raise ShockQOIError(f"point count in {path} exceeds configured bound")

    points_node = _only_child(piece, "Points", str(path))
    coordinate_arrays = list(_children(points_node, "DataArray"))
    if len(coordinate_arrays) != 1:
        raise ShockQOIError(f"{path} must contain one Points DataArray")
    coordinate_array = coordinate_arrays[0]
    components = _component_count(coordinate_array, "Points")
    if components not in (2, 3):
        raise ShockQOIError(f"{path} Points must have two or three components")
    raw_points = _array_values(
        root, coordinate_array, count * components, f"Points in {path}"
    )
    points = [
        (raw_points[index], raw_points[index + 1],
         raw_points[index + 2] if components == 3 else 0.0)
        for index in range(0, len(raw_points), components)
    ]

    point_data = _only_child(piece, "PointData", str(path))
    arrays = _named_arrays(point_data, path)
    density_array = _required_array(arrays, ("density", "rho"), path)
    momentum_array = _required_array(arrays, ("momentum", "mom"), path)
    energy_array = _required_array(
        arrays, ("energy", "total_energy"), path
    )
    sigma_array = _required_array(arrays, ("sigma", "igr_sigma"), path)
    pressure_array = _optional_array(arrays, ("pressure", "p"), path)
    for name, array in (
            ("density", density_array), ("energy", energy_array),
            ("sigma", sigma_array)):
        if _component_count(array, name) != 1:
            raise ShockQOIError(f"{name} in {path} must be scalar PointData")
    momentum_components = _component_count(momentum_array, "momentum")
    if momentum_components not in (1, 2, 3):
        raise ShockQOIError(
            f"momentum in {path} must have one, two, or three components"
        )
    if pressure_array is not None and \
            _component_count(pressure_array, "pressure") != 1:
        raise ShockQOIError(f"pressure in {path} must be scalar PointData")
    momentum_values = _array_values(
        root, momentum_array, count * momentum_components,
        f"momentum in {path}",
    )
    return Fields(
        points=points,
        density=_array_values(root, density_array, count, f"density in {path}"),
        momentum=[
            tuple(momentum_values[offset:offset + momentum_components])
            for offset in range(0, len(momentum_values), momentum_components)
        ],
        energy=_array_values(root, energy_array, count, f"energy in {path}"),
        sigma=_array_values(root, sigma_array, count, f"sigma in {path}"),
        pressure=_array_values(
            root, pressure_array, count, f"pressure in {path}"
        ) if pressure_array is not None else None,
    )


def load_frame(frame: Frame, vtk_boundary: Path, limits: Limits) -> Fields:
    if frame.path.suffix.lower() == ".vtu":
        piece_paths = [frame.path]
    else:
        root = _read_xml(frame.path, limits)
        if root.get("type") != "PUnstructuredGrid":
            raise ShockQOIError(f"{frame.path} is not a VTK PUnstructuredGrid")
        pieces = list(_descendants(root, "Piece"))
        if not pieces or len(pieces) > limits.max_pieces:
            raise ShockQOIError(
                f"{frame.path} has an invalid number of rank pieces: {len(pieces)}"
            )
        piece_paths = [
            _safe_reference(frame.path.parent, piece.get("Source"), vtk_boundary,
                            "PVTU rank piece")
            for piece in pieces
        ]
        if len(set(piece_paths)) != len(piece_paths):
            raise ShockQOIError(f"duplicate rank piece in {frame.path}")

    combined = Fields([], [], [], [], [], None)
    momentum_components: int | None = None
    pressure_present: bool | None = None
    for path in piece_paths:
        parsed = _parse_vtu(path, limits)
        if len(combined.points) + len(parsed.points) > limits.max_total_points:
            raise ShockQOIError("aggregate point count exceeds configured bound")
        parsed_momentum_components = len(parsed.momentum[0])
        if momentum_components is None:
            momentum_components = parsed_momentum_components
        elif parsed_momentum_components != momentum_components:
            raise ShockQOIError(
                "MPI rank pieces have inconsistent momentum component counts"
            )
        parsed_pressure_present = parsed.pressure is not None
        if pressure_present is None:
            pressure_present = parsed_pressure_present
            combined.pressure = [] if pressure_present else None
        elif parsed_pressure_present != pressure_present:
            raise ShockQOIError(
                "MPI rank pieces inconsistently provide projected pressure"
            )
        combined.points.extend(parsed.points)
        combined.density.extend(parsed.density)
        combined.momentum.extend(parsed.momentum)
        combined.energy.extend(parsed.energy)
        combined.sigma.extend(parsed.sigma)
        if combined.pressure is not None and parsed.pressure is not None:
            combined.pressure.extend(parsed.pressure)
    count = len(combined.points)
    required_values = (
        combined.density, combined.momentum, combined.energy, combined.sigma
    )
    if count == 0 or not all(len(values) == count for values in required_values):
        raise ShockQOIError("inconsistent or empty aggregate PointData")
    if combined.pressure is not None and len(combined.pressure) != count:
        raise ShockQOIError("inconsistent projected pressure PointData")
    return combined


def _group_profile(samples: list[tuple[float, float]], width: float) -> Profile:
    samples.sort(key=lambda item: item[0])
    grouped_x: list[float] = []
    grouped_density: list[float] = []
    index = 0
    while index < len(samples):
        anchor = samples[index][0]
        stop = index + 1
        while stop < len(samples) and samples[stop][0] - anchor <= width:
            stop += 1
        group = samples[index:stop]
        grouped_x.append(sum(item[0] for item in group) / len(group))
        grouped_density.append(sum(item[1] for item in group) / len(group))
        index = stop
    if len(grouped_x) < 3:
        raise ShockQOIError("near-centerline density profile has fewer than three samples")
    if any(grouped_x[i] >= grouped_x[i + 1]
           for i in range(len(grouped_x) - 1)):
        raise ShockQOIError("density profile coordinates are not strictly increasing")
    return Profile(tuple(grouped_x), tuple(grouped_density))


def make_profile(fields: Fields, *, y: float, x_min: float | None,
                 x_max: float | None, half_width: float | None,
                 bin_width: float | None, coordinate_tolerance: float,
                 length_scale: float) -> Profile:
    candidates = [
        (point[0], rho, abs(point[1] - y))
        for point, rho in zip(fields.points, fields.density)
        if (x_min is None or point[0] >= x_min - coordinate_tolerance)
        and (x_max is None or point[0] <= x_max + coordinate_tolerance)
    ]
    if not candidates:
        raise ShockQOIError("no points lie in the requested profile x interval")
    if half_width is None:
        requested_width = max(0.02 * length_scale, 10.0 * coordinate_tolerance)
        near_count = sum(item[2] <= requested_width for item in candidates)
        if near_count < 8:
            distances = sorted(item[2] for item in candidates)
            requested_width = max(
                requested_width,
                distances[min(7, len(distances) - 1)] + coordinate_tolerance,
            )
    else:
        requested_width = half_width
    selected = [(x, rho) for x, rho, distance in candidates
                if distance <= requested_width]
    if len(selected) < 3:
        raise ShockQOIError("too few points in the requested centerline strip")
    grouping_width = bin_width
    if grouping_width is None:
        grouping_width = max(0.02 * length_scale, coordinate_tolerance)
    return _group_profile(selected, grouping_width)


def shock_metrics(profile: Profile, nose_x: float,
                  thickness_length_scale: float) -> tuple[float | None,
                                                          float | None,
                                                          float | None]:
    gradients: list[tuple[float, int]] = []
    for index in range(len(profile.x) - 1):
        dx = profile.x[index + 1] - profile.x[index]
        gradient = abs(profile.density[index + 1] - profile.density[index]) / dx
        gradients.append((gradient, index))
    maximum_gradient, index = max(gradients)
    jump = max(profile.density) - min(profile.density)
    density_scale = max(1.0, max(abs(item) for item in profile.density))
    if maximum_gradient <= 0.0 or jump <= 1.0e-14 * density_scale:
        return None, None, None
    position = 0.5 * (profile.x[index] + profile.x[index + 1])
    standoff = nose_x - position
    if standoff < -1.0e-10 * max(1.0, abs(nose_x)):
        raise ShockQOIError("detected cylinder shock lies downstream of the nose")
    standoff = max(0.0, standoff)
    thickness = jump / maximum_gradient / thickness_length_scale
    return position, standoff, thickness


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise ShockQOIError(f"cannot hash file {path}: {error}") from error
    return digest.hexdigest()


def _reference_profile_from_json(path: Path) -> ReferenceProfile:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ShockQOIError(f"cannot read reference profile {path}: {error}") from error
    if len(raw) > 64 * 1024 * 1024:
        raise ShockQOIError("reference profile exceeds 64 MiB bound")
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShockQOIError(f"malformed reference profile {path}") from error
    if not isinstance(decoded, dict):
        raise ShockQOIError("reference profile must be a JSON object")
    if decoded.get("schema_version") != 2:
        raise ShockQOIError("reference profile requires schema_version 2")
    case = decoded.get("case")
    method = decoded.get("method")
    mesh = decoded.get("mesh")
    target_time = _finite_float(
        decoded.get("target_time"), "reference target_time"
    )
    gamma = _finite_float(decoded.get("gamma"), "reference gamma")
    extraction = decoded.get("extraction")
    if not isinstance(case, str) or not case:
        raise ShockQOIError("reference profile requires a nonempty case")
    if not isinstance(method, str) or not method:
        raise ShockQOIError("reference profile requires a nonempty method")
    if not isinstance(mesh, dict):
        raise ShockQOIError("reference profile requires mesh provenance")
    mesh_path = mesh.get("path")
    mesh_sha256 = mesh.get("sha256")
    if not isinstance(mesh_path, str) or not mesh_path:
        raise ShockQOIError("reference profile requires a nonempty mesh path")
    if (not isinstance(mesh_sha256, str) or len(mesh_sha256) != 64 or
            any(character not in "0123456789abcdef" for character in mesh_sha256)):
        raise ShockQOIError("reference profile requires a lowercase SHA-256 mesh hash")
    if gamma <= 1.0:
        raise ShockQOIError("reference gamma must be greater than one")
    required_extraction = (
        "profile_y", "profile_x_min", "profile_x_max",
        "centerline_half_width", "profile_bin_width",
        "coordinate_tolerance", "thickness_length_scale",
    )
    if not isinstance(extraction, dict) or set(extraction) != set(required_extraction):
        raise ShockQOIError("reference profile has an incomplete extraction contract")
    checked_extraction = {
        name: _finite_float(extraction[name], f"reference extraction {name}")
        for name in required_extraction
    }

    payload = decoded.get("profile")
    if not isinstance(payload, dict):
        raise ShockQOIError("reference profile payload must be an object")
    if payload.get("coordinate") != "x":
        raise ShockQOIError("reference profile coordinate must be x")
    x_values = payload.get("x")
    density_values = payload.get("density")
    if not isinstance(x_values, list) or not isinstance(density_values, list):
        raise ShockQOIError("reference profile requires x and density arrays")
    if len(x_values) != len(density_values) or len(x_values) < 8:
        raise ShockQOIError("reference profile arrays have incompatible lengths")
    x = tuple(_finite_float(value, "reference x") for value in x_values)
    density = tuple(
        _finite_float(value, "reference density") for value in density_values
    )
    if any(x[index] >= x[index + 1] for index in range(len(x) - 1)):
        raise ShockQOIError("reference x coordinates must be strictly increasing")
    return ReferenceProfile(
        profile=Profile(x, density),
        case=case,
        method=method,
        target_time=target_time,
        mesh=mesh_path,
        mesh_sha256=mesh_sha256,
        gamma=gamma,
        extraction=checked_extraction,
    )


def _profile_from_json(path: Path) -> Profile:
    """Compatibility helper returning the validated schema-v2 profile."""
    return _reference_profile_from_json(path).profile


def _interpolate(profile: Profile, coordinate: float) -> float:
    if coordinate <= profile.x[0]:
        return profile.density[0]
    if coordinate >= profile.x[-1]:
        return profile.density[-1]
    low = 0
    high = len(profile.x) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if profile.x[middle] <= coordinate:
            low = middle
        else:
            high = middle
    fraction = (coordinate - profile.x[low]) / (profile.x[high] - profile.x[low])
    return profile.density[low] + fraction * (
        profile.density[high] - profile.density[low]
    )


def profile_errors(
        profile: Profile, reference: Profile, *,
        requested_interval: tuple[float, float] | None = None,
        minimum_overlap_fraction: float = 0.99,
        endpoint_tolerance: float = 0.0) -> tuple[float, float]:
    if (not math.isfinite(minimum_overlap_fraction) or
            minimum_overlap_fraction <= 0.0 or minimum_overlap_fraction > 1.0):
        raise ShockQOIError("minimum profile overlap fraction must lie in (0,1]")
    if not math.isfinite(endpoint_tolerance) or endpoint_tolerance < 0.0:
        raise ShockQOIError("profile endpoint tolerance must be nonnegative")
    lower = max(profile.x[0], reference.x[0])
    upper = min(profile.x[-1], reference.x[-1])
    if upper <= lower:
        raise ShockQOIError("computed and reference profiles do not overlap")
    overlap = upper - lower
    computed_span = profile.x[-1] - profile.x[0]
    reference_span = reference.x[-1] - reference.x[0]
    if (overlap + 2.0 * endpoint_tolerance <
            minimum_overlap_fraction * computed_span or
            overlap + 2.0 * endpoint_tolerance <
            minimum_overlap_fraction * reference_span):
        raise ShockQOIError(
            "computed and reference profile overlap is below the required fraction"
        )
    if requested_interval is not None:
        requested_lower, requested_upper = requested_interval
        if (not math.isfinite(requested_lower) or
                not math.isfinite(requested_upper) or
                requested_lower >= requested_upper):
            raise ShockQOIError("requested profile interval is invalid")
        for label, candidate in (("computed", profile), ("reference", reference)):
            if (candidate.x[0] > requested_lower + endpoint_tolerance or
                    candidate.x[-1] < requested_upper - endpoint_tolerance):
                raise ShockQOIError(
                    f"{label} profile does not cover the requested interval"
                )
    knots = sorted({
        lower, upper,
        *(item for item in profile.x if lower < item < upper),
        *(item for item in reference.x if lower < item < upper),
    })
    l1_integral = 0.0
    l2_integral = 0.0
    for left, right in zip(knots, knots[1:]):
        d_left = _interpolate(profile, left) - _interpolate(reference, left)
        d_right = _interpolate(profile, right) - _interpolate(reference, right)
        dx = right - left
        if d_left * d_right >= 0.0:
            l1_integral += 0.5 * dx * (abs(d_left) + abs(d_right))
        else:
            denominator = abs(d_left) + abs(d_right)
            l1_integral += 0.5 * dx * (
                d_left * d_left + d_right * d_right
            ) / denominator
        l2_integral += dx * (
            d_left * d_left + d_left * d_right + d_right * d_right
        ) / 3.0
    interval = upper - lower
    return l1_integral / interval, math.sqrt(max(0.0, l2_integral / interval))


def _write_profile(path: Path, profile: Profile, *, method: str,
                   target_time: float, case: str, mesh: str, gamma: float,
                   extraction: dict[str, float], overwrite: bool) -> None:
    destination = path.resolve()
    if destination.exists() and not overwrite:
        raise ShockQOIError(
            f"profile output already exists (use --overwrite-profile): {destination}"
        )
    if len(profile.x) != len(profile.density) or len(profile.x) < 8:
        raise ShockQOIError(
            "profile output requires at least eight paired profile locations"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh_path = Path(mesh).expanduser().resolve()
    if not mesh_path.is_file():
        raise ShockQOIError(f"profile source mesh does not exist: {mesh_path}")
    document = {
        "schema_version": 2,
        "case": case,
        "method": method,
        "target_time": target_time,
        "mesh": {
            "path": str(mesh_path),
            "sha256": _sha256_file(mesh_path),
        },
        "gamma": gamma,
        "extraction": extraction,
        "profile": {
            "coordinate": "x",
            "x": list(profile.x),
            "density": list(profile.density),
        },
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _reference_extraction(
        *, profile_y: float | None, profile_x_min: float | None,
        profile_x_max: float | None, centerline_half_width: float | None,
        profile_bin_width: float | None, coordinate_tolerance: float,
        thickness_length_scale: float | None) -> dict[str, float]:
    values = {
        "profile_y": profile_y,
        "profile_x_min": profile_x_min,
        "profile_x_max": profile_x_max,
        "centerline_half_width": centerline_half_width,
        "profile_bin_width": profile_bin_width,
        "coordinate_tolerance": coordinate_tolerance,
        "thickness_length_scale": thickness_length_scale,
    }
    if any(value is None for value in values.values()):
        raise ShockQOIError(
            "reference profile use requires explicit profile y/x bounds, "
            "strip width, bin width, coordinate tolerance, and thickness scale"
        )
    return {name: float(value) for name, value in values.items()
            if value is not None}


def _validate_reference_profile(
        reference: ReferenceProfile, *, case: str, target_time: float,
        gamma: float, extraction: dict[str, float], expected_method: str,
        expected_mesh_sha256: str, time_tolerance: float,
        coordinate_tolerance: float) -> None:
    if reference.case != case:
        raise ShockQOIError(
            f"reference case mismatch: {reference.case!r} != {case!r}"
        )
    if reference.method != expected_method:
        raise ShockQOIError(
            "reference method metadata does not match --reference-method"
        )
    if reference.mesh_sha256 != expected_mesh_sha256:
        raise ShockQOIError(
            "reference mesh metadata does not match --reference-mesh-sha256"
        )
    if abs(reference.target_time - target_time) > time_tolerance:
        raise ShockQOIError("reference target time does not match requested target")
    if not math.isclose(
            reference.gamma, gamma, rel_tol=1.0e-13,
            abs_tol=10.0 * sys.float_info.epsilon):
        raise ShockQOIError("reference gamma does not match requested gamma")
    for name, requested in extraction.items():
        stored = reference.extraction[name]
        tolerance = max(
            coordinate_tolerance,
            32.0 * sys.float_info.epsilon * max(1.0, abs(stored), abs(requested)),
        )
        if abs(stored - requested) > tolerance:
            raise ShockQOIError(
                f"reference extraction mismatch for {name}: {stored} != {requested}"
            )


def reconstruct_pressure(fields: Fields, gamma: float) -> list[float]:
    """Recompute thermodynamic pressure from conserved values in FP64."""
    checked_gamma = _finite_float(gamma, "gamma")
    if checked_gamma <= 1.0:
        raise ShockQOIError("gamma must be finite and greater than one")
    count = len(fields.points)
    if count == 0 or not all(
            len(values) == count for values in
            (fields.density, fields.momentum, fields.energy)):
        raise ShockQOIError(
            "pressure reconstruction requires equally sized nonempty "
            "density, momentum, and energy fields"
        )
    component_count = len(fields.momentum[0])
    if component_count not in (1, 2, 3):
        raise ShockQOIError(
            "pressure reconstruction requires one-, two-, or "
            "three-component momentum"
        )

    pressure: list[float] = []
    for index, (rho, momentum, energy) in enumerate(zip(
            fields.density, fields.momentum, fields.energy)):
        if not math.isfinite(rho) or not math.isfinite(energy):
            raise ShockQOIError(
                f"non-finite conserved scalar at point {index}"
            )
        if rho <= 0.0:
            raise ShockQOIError(
                f"nonpositive density prevents pressure reconstruction at point {index}"
            )
        if len(momentum) != component_count or not all(
                math.isfinite(component) for component in momentum):
            raise ShockQOIError(
                f"malformed or non-finite momentum at point {index}"
            )
        try:
            momentum_squared = math.fsum(
                component * component for component in momentum
            )
            kinetic_energy = momentum_squared / (2.0 * rho)
            value = (checked_gamma - 1.0) * (energy - kinetic_energy)
        except (OverflowError, ValueError, ZeroDivisionError) as error:
            raise ShockQOIError(
                f"pressure reconstruction failed at point {index}"
            ) from error
        if not all(math.isfinite(item) for item in (
                momentum_squared, kinetic_energy, value)):
            raise ShockQOIError(
                f"pressure reconstruction is non-finite at point {index}"
            )
        pressure.append(value)
    return pressure


def compute_qoi(fields: Fields, *, case: str, center_x: float,
                center_y: float, radius: float, profile_y: float | None,
                profile_x_min: float | None, profile_x_max: float | None,
                centerline_half_width: float | None,
                profile_bin_width: float | None,
                coordinate_tolerance: float,
                thickness_length_scale: float | None,
                reference_profile: Path | None,
                gamma: float = 1.4,
                target_time: float | None = None,
                reference_method: str | None = None,
                reference_mesh_sha256: str | None = None,
                reference_time_tolerance: float = 0.0,
                minimum_profile_overlap_fraction: float = 0.99
                ) -> tuple[dict[str, float | None],
                                             Profile | None]:
    result: dict[str, float | None] = {name: None for name in QOI_FIELDS}
    pressure = reconstruct_pressure(fields, gamma)
    result["density_minimum"] = min(fields.density)
    result["density_maximum"] = max(fields.density)
    result["pressure_minimum"] = min(pressure)
    result["pressure_maximum"] = max(pressure)
    absolute_sigma = [abs(value) for value in fields.sigma]
    sigma_mean = sum(absolute_sigma) / len(absolute_sigma)
    result["sigma_localization"] = (
        max(absolute_sigma) / sigma_mean if sigma_mean > 0.0 else 0.0
    )

    profile: Profile | None = None
    if case == "cylinder":
        nose_x = center_x - radius
        upper = nose_x if profile_x_max is None else min(profile_x_max, nose_x)
        profile = make_profile(
            fields, y=center_y if profile_y is None else profile_y,
            x_min=profile_x_min, x_max=upper,
            half_width=centerline_half_width, bin_width=profile_bin_width,
            coordinate_tolerance=coordinate_tolerance, length_scale=radius,
        )
        scale = radius if thickness_length_scale is None else thickness_length_scale
        position, standoff, thickness = shock_metrics(profile, nose_x, scale)
        result["shock_position"] = position
        result["shock_standoff_distance"] = standoff
        result["shock_thickness"] = thickness
    elif reference_profile is not None:
        xs = [point[0] for point in fields.points]
        extent = max(xs) - min(xs)
        if extent <= 0.0:
            raise ShockQOIError("generic profile has zero x extent")
        profile = make_profile(
            fields, y=center_y if profile_y is None else profile_y,
            x_min=profile_x_min, x_max=profile_x_max,
            half_width=centerline_half_width, bin_width=profile_bin_width,
            coordinate_tolerance=coordinate_tolerance, length_scale=extent,
        )
    if reference_profile is not None:
        if profile is None:
            raise ShockQOIError("reference comparison requires a computed profile")
        if (target_time is None or reference_method is None or
                reference_mesh_sha256 is None):
            raise ShockQOIError(
                "reference comparison requires target, method, and mesh identity"
            )
        extraction = _reference_extraction(
            profile_y=profile_y, profile_x_min=profile_x_min,
            profile_x_max=profile_x_max,
            centerline_half_width=centerline_half_width,
            profile_bin_width=profile_bin_width,
            coordinate_tolerance=coordinate_tolerance,
            thickness_length_scale=thickness_length_scale,
        )
        reference = _reference_profile_from_json(reference_profile)
        _validate_reference_profile(
            reference, case=case, target_time=target_time, gamma=gamma,
            extraction=extraction, expected_method=reference_method,
            expected_mesh_sha256=reference_mesh_sha256,
            time_tolerance=reference_time_tolerance,
            coordinate_tolerance=coordinate_tolerance,
        )
        requested_upper = profile_x_max
        if case == "cylinder":
            requested_upper = min(profile_x_max, center_x - radius)
        if profile_x_min is None or requested_upper is None:
            raise ShockQOIError("reference comparison requires explicit x bounds")
        l1, l2 = profile_errors(
            profile, reference.profile,
            requested_interval=(profile_x_min, requested_upper),
            minimum_overlap_fraction=minimum_profile_overlap_fraction,
            endpoint_tolerance=profile_bin_width or 0.0,
        )
        result["shock_profile_l1_error"] = l1
        result["shock_profile_l2_error"] = l2
    if set(result) != set(QOI_FIELDS):
        raise AssertionError("internal QoI schema mismatch")
    for name, value in result.items():
        if value is not None and not math.isfinite(value):
            raise ShockQOIError(f"computed QoI {name} is non-finite")
    # Conservation and boundary balance are intentionally null: point-sampled
    # VTK output does not contain the integration and flux data needed to
    # reconstruct either quantity honestly.
    return result, profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract bounded shock QoIs from an MFEM ParaView collection"
    )
    parser.add_argument("--vtk-root", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--target-time", required=True, type=float)
    parser.add_argument("--mesh", required=True)
    parser.add_argument(
        "--gamma", type=float, default=1.4,
        help="Ratio of specific heats used for conserved-state pressure",
    )
    parser.add_argument(
        "--case", choices=("cylinder", "planar", "shock-vortex", "generic"),
        default="cylinder",
    )
    parser.add_argument("--center-x", type=float, default=0.6)
    parser.add_argument("--center-y", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--profile-y", type=float)
    parser.add_argument("--profile-x-min", type=float)
    parser.add_argument("--profile-x-max", type=float)
    parser.add_argument("--centerline-half-width", type=float)
    parser.add_argument("--profile-bin-width", type=float)
    parser.add_argument("--coordinate-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--thickness-length-scale", type=float)
    parser.add_argument("--reference-profile", type=Path)
    parser.add_argument(
        "--reference-method",
        help="Exact method identity stored in --reference-profile",
    )
    parser.add_argument(
        "--reference-mesh-sha256",
        help="Exact lowercase mesh SHA-256 stored in --reference-profile",
    )
    parser.add_argument(
        "--minimum-profile-overlap-fraction", type=float, default=0.99,
        help=(
            "Minimum mutual computed/reference profile coverage; requested "
            "bounds must also be covered within one bin width"
        ),
    )
    parser.add_argument("--write-profile", type=Path)
    parser.add_argument("--overwrite-profile", action="store_true")
    parser.add_argument("--time-absolute-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--time-relative-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--max-file-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--max-total-points", type=int, default=20_000_000)
    parser.add_argument("--max-pieces", type=int, default=4096)
    parser.add_argument("--max-pvds", type=int, default=128)
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> Limits:
    if not arguments.method.strip():
        raise ShockQOIError("--method must not be empty")
    arguments.target_time = _finite_float(arguments.target_time, "--target-time")
    arguments.gamma = _finite_float(arguments.gamma, "--gamma")
    if arguments.gamma <= 1.0:
        raise ShockQOIError("--gamma must be finite and greater than one")
    for name in (
            "center_x", "center_y", "profile_y", "profile_x_min",
            "profile_x_max"):
        value = getattr(arguments, name)
        if value is not None:
            setattr(arguments, name, _finite_float(value, f"--{name.replace('_', '-')}"))
    for name in (
            "radius", "coordinate_tolerance", "time_absolute_tolerance",
            "time_relative_tolerance", "minimum_profile_overlap_fraction"):
        value = _finite_float(getattr(arguments, name),
                              f"--{name.replace('_', '-')}")
        if value <= 0.0 and name == "radius":
            raise ShockQOIError("--radius must be positive")
        if value < 0.0 and name != "radius":
            raise ShockQOIError(f"--{name.replace('_', '-')} must be nonnegative")
        setattr(arguments, name, value)
    if not 0.0 < arguments.minimum_profile_overlap_fraction <= 1.0:
        raise ShockQOIError(
            "--minimum-profile-overlap-fraction must lie in (0,1]"
        )
    for name in ("centerline_half_width", "profile_bin_width",
                 "thickness_length_scale"):
        value = getattr(arguments, name)
        if value is not None:
            value = _finite_float(value, f"--{name.replace('_', '-')}")
            if value <= 0.0:
                raise ShockQOIError(f"--{name.replace('_', '-')} must be positive")
            setattr(arguments, name, value)
    if (arguments.profile_x_min is not None and
            arguments.profile_x_max is not None and
            arguments.profile_x_min >= arguments.profile_x_max):
        raise ShockQOIError("--profile-x-min must be less than --profile-x-max")
    mesh = Path(arguments.mesh).expanduser()
    if not mesh.is_file():
        raise ShockQOIError(f"--mesh does not name a readable file: {arguments.mesh}")
    if arguments.reference_profile is not None and \
            not arguments.reference_profile.is_file():
        raise ShockQOIError(
            f"--reference-profile does not exist: {arguments.reference_profile}"
        )
    if arguments.reference_profile is not None:
        if not arguments.reference_method:
            raise ShockQOIError(
                "--reference-profile requires --reference-method"
            )
        digest = arguments.reference_mesh_sha256
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(character not in "0123456789abcdef" for character in digest)):
            raise ShockQOIError(
                "--reference-profile requires a lowercase "
                "--reference-mesh-sha256"
            )
        _reference_extraction(
            profile_y=arguments.profile_y,
            profile_x_min=arguments.profile_x_min,
            profile_x_max=arguments.profile_x_max,
            centerline_half_width=arguments.centerline_half_width,
            profile_bin_width=arguments.profile_bin_width,
            coordinate_tolerance=arguments.coordinate_tolerance,
            thickness_length_scale=arguments.thickness_length_scale,
        )
    elif arguments.reference_method is not None or \
            arguments.reference_mesh_sha256 is not None:
        raise ShockQOIError(
            "--reference-method/--reference-mesh-sha256 require "
            "--reference-profile"
        )
    if arguments.write_profile is not None:
        _reference_extraction(
            profile_y=arguments.profile_y,
            profile_x_min=arguments.profile_x_min,
            profile_x_max=arguments.profile_x_max,
            centerline_half_width=arguments.centerline_half_width,
            profile_bin_width=arguments.profile_bin_width,
            coordinate_tolerance=arguments.coordinate_tolerance,
            thickness_length_scale=arguments.thickness_length_scale,
        )
    limits = Limits(
        _positive_int(str(arguments.max_file_bytes), "--max-file-bytes"),
        _positive_int(str(arguments.max_total_points), "--max-total-points"),
        _positive_int(str(arguments.max_pieces), "--max-pieces"),
        _positive_int(str(arguments.max_pvds), "--max-pvds"),
    )
    return limits


def run(arguments: argparse.Namespace) -> dict[str, float | None]:
    limits = _validate_arguments(arguments)
    frame = select_frame(
        arguments.vtk_root, arguments.target_time, arguments.method, limits,
        arguments.time_absolute_tolerance, arguments.time_relative_tolerance,
    )
    requested = arguments.vtk_root.resolve()
    boundary = requested.parent if requested.is_file() else requested
    fields = load_frame(frame, boundary, limits)
    qoi, profile = compute_qoi(
        fields, case=arguments.case, center_x=arguments.center_x,
        center_y=arguments.center_y, radius=arguments.radius,
        profile_y=arguments.profile_y, profile_x_min=arguments.profile_x_min,
        profile_x_max=arguments.profile_x_max,
        centerline_half_width=arguments.centerline_half_width,
        profile_bin_width=arguments.profile_bin_width,
        coordinate_tolerance=arguments.coordinate_tolerance,
        thickness_length_scale=arguments.thickness_length_scale,
        reference_profile=arguments.reference_profile,
        gamma=arguments.gamma,
        target_time=arguments.target_time,
        reference_method=arguments.reference_method,
        reference_mesh_sha256=arguments.reference_mesh_sha256,
        reference_time_tolerance=_time_tolerance(
            arguments.target_time,
            arguments.time_absolute_tolerance,
            arguments.time_relative_tolerance,
        ),
        minimum_profile_overlap_fraction=(
            arguments.minimum_profile_overlap_fraction
        ),
    )
    if arguments.write_profile is not None:
        if profile is None:
            xs = [point[0] for point in fields.points]
            extent = max(xs) - min(xs)
            if extent <= 0.0:
                raise ShockQOIError("cannot write a profile with zero x extent")
            profile = make_profile(
                fields, y=arguments.center_y if arguments.profile_y is None
                else arguments.profile_y,
                x_min=arguments.profile_x_min, x_max=arguments.profile_x_max,
                half_width=arguments.centerline_half_width,
                bin_width=arguments.profile_bin_width,
                coordinate_tolerance=arguments.coordinate_tolerance,
                length_scale=extent,
            )
        _write_profile(
            arguments.write_profile, profile, method=arguments.method,
            target_time=frame.time, case=arguments.case, mesh=arguments.mesh,
            gamma=arguments.gamma,
            extraction=_reference_extraction(
                profile_y=arguments.profile_y,
                profile_x_min=arguments.profile_x_min,
                profile_x_max=arguments.profile_x_max,
                centerline_half_width=arguments.centerline_half_width,
                profile_bin_width=arguments.profile_bin_width,
                coordinate_tolerance=arguments.coordinate_tolerance,
                thickness_length_scale=arguments.thickness_length_scale,
            ),
            overwrite=arguments.overwrite_profile,
        )
    return qoi


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        result = run(arguments)
    except (ShockQOIError, OSError, OverflowError, struct.error) as error:
        print(f"shock_qoi: {error}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, sort_keys=True, separators=(",", ":"),
              allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
