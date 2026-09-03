#!/usr/bin/env python3
"""Synthetic, no-device tests for shock_qoi.py."""

from __future__ import annotations

import base64
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib


sys.dont_write_bytecode = True

from shock_qoi import (  # noqa: E402
    QOI_FIELDS,
    Fields,
    Frame,
    Limits,
    Profile,
    ShockQOIError,
    _parse_vtu,
    _profile_from_json,
    _reference_profile_from_json,
    _validate_reference_profile,
    _write_profile,
    compute_qoi,
    load_frame,
    main,
    profile_errors,
    reconstruct_pressure,
    select_frame,
)


TEST_LIMITS = Limits(
    max_file_bytes=4 * 1024 * 1024,
    max_total_points=10_000,
    max_pieces=16,
    max_pvds=8,
)


def _encoded(values: list[float], mode: str, *, endian: str,
             header_type: str) -> str:
    raw = struct.pack(f"{endian}{len(values)}d", *values)
    header_code = "I" if header_type == "UInt32" else "Q"
    if mode == "binary":
        header = struct.pack(f"{endian}{header_code}", len(raw))
        return base64.b64encode(header).decode() + base64.b64encode(raw).decode()
    if mode == "compressed":
        # Match MFEM's WriteVTKEncodedCompressed one-block convention: zero
        # means the final block is full-sized.
        compressed = zlib.compress(raw)
        header_values = [1, len(raw), 0, len(compressed)]
        header = struct.pack(
            f"{endian}{len(header_values)}{header_code}", *header_values
        )
        return (base64.b64encode(header).decode() +
                base64.b64encode(compressed).decode())
    if mode == "ascii":
        return " ".join(format(value, ".17g") for value in values)
    raise AssertionError(mode)


def _data_array(name: str | None, values: list[float], mode: str, *,
                components: int = 1, endian: str = "<",
                header_type: str = "UInt32") -> str:
    name_attribute = "" if name is None else f' Name="{name}"'
    vtk_format = "ascii" if mode == "ascii" else "binary"
    return (
        f'<DataArray type="Float64"{name_attribute} '
        f'NumberOfComponents="{components}" format="{vtk_format}">'
        f'{_encoded(values, mode, endian=endian, header_type=header_type)}'
        '</DataArray>'
    )


def _write_vtu(path: Path, points: list[tuple[float, float, float]],
               density: list[float], momentum: list[tuple[float, ...]],
               energy: list[float], sigma: list[float],
               pressure: list[float] | None = None, *, mode: str = "ascii",
               endian: str = "<", header_type: str = "UInt32",
               omit: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = [component for point in points for component in point]
    momentum_components = len(momentum[0]) if momentum else 0
    flattened_momentum = [
        component for value in momentum for component in value
    ]
    arrays: list[tuple[str, list[float], int]] = [
        ("density", density, 1),
        ("momentum", flattened_momentum, momentum_components),
        ("energy", energy, 1),
        ("sigma", sigma, 1),
    ]
    if pressure is not None:
        arrays.append(("pressure", pressure, 1))
    point_data = "".join(
        _data_array(
            name, values, mode, components=components,
            endian=endian, header_type=header_type,
        )
        for name, values, components in arrays if name != omit
    )
    byte_order = "LittleEndian" if endian == "<" else "BigEndian"
    compressor = ' compressor="vtkZLibDataCompressor"' \
        if mode == "compressed" else ""
    document = (
        f'<VTKFile type="UnstructuredGrid" byte_order="{byte_order}" '
        f'header_type="{header_type}"{compressor}>'
        '<UnstructuredGrid>'
        f'<Piece NumberOfPoints="{len(points)}" NumberOfCells="0">'
        '<Points>'
        f'{_data_array(None, flattened, mode, components=3, endian=endian, header_type=header_type)}'
        '</Points><PointData>'
        f'{point_data}'
        '</PointData></Piece></UnstructuredGrid></VTKFile>'
    )
    path.write_text(document, encoding="utf-8")


def _write_pvtu(path: Path, sources: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pieces = "".join(f'<Piece Source="{source}"/>' for source in sources)
    path.write_text(
        '<VTKFile type="PUnstructuredGrid" byte_order="LittleEndian">'
        f'<PUnstructuredGrid>{pieces}</PUnstructuredGrid></VTKFile>',
        encoding="utf-8",
    )


def _write_pvd(path: Path, entries: list[tuple[float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets = "".join(
        f'<DataSet timestep="{format(time, ".17g")}" file="{source}"/>'
        for time, source in entries
    )
    path.write_text(
        '<VTKFile type="Collection" byte_order="LittleEndian">'
        f'<Collection>{datasets}</Collection></VTKFile>',
        encoding="utf-8",
    )


def _energy_from_pressure(
        density: list[float], momentum: list[tuple[float, ...]],
        pressure: list[float], gamma: float = 1.4) -> list[float]:
    return [
        p / (gamma - 1.0) +
        sum(component * component for component in mom) / (2.0 * rho)
        for rho, mom, p in zip(density, momentum, pressure)
    ]


def _simple_fields(offset: float = 0.0, gamma: float = 1.4) -> tuple[
        list[tuple[float, float, float]], list[float],
        list[tuple[float, ...]], list[float], list[float], list[float]]:
    points = [
        (offset + 0.0, 0.0, 0.0),
        (offset + 0.1, 0.0, 0.0),
        (offset + 0.2, 0.0, 0.0),
    ]
    density = [1.0, 1.5, 2.0]
    momentum = [(1.0, 0.0), (1.5, 0.0), (2.0, 0.0)]
    pressure = [3.0, 3.5, 4.0]
    energy = _energy_from_pressure(density, momentum, pressure, gamma)
    return points, density, momentum, energy, [0.0, 1.0, 2.0], pressure


class VTKParsingTest(unittest.TestCase):
    def test_ascii_binary_and_mfem_zlib_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points, density, momentum, energy, sigma, pressure = _simple_fields()
            cases = (
                ("ascii", "<", "UInt32"),
                ("binary", "<", "UInt32"),
                ("compressed", ">", "UInt64"),
            )
            for index, (mode, endian, header_type) in enumerate(cases):
                with self.subTest(mode=mode, endian=endian,
                                  header_type=header_type):
                    path = root / f"piece-{index}.vtu"
                    _write_vtu(
                        path, points, density, momentum, energy, sigma,
                        pressure, mode=mode, endian=endian,
                        header_type=header_type,
                    )
                    parsed = _parse_vtu(path, TEST_LIMITS)
                    self.assertEqual(parsed.points, points)
                    self.assertEqual(parsed.density, density)
                    self.assertEqual(parsed.momentum, momentum)
                    self.assertEqual(parsed.energy, energy)
                    self.assertEqual(parsed.pressure, pressure)
                    self.assertEqual(parsed.sigma, sigma)

    def test_target_selection_uses_last_equally_close_final_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = _simple_fields()
            first = root / "Cycle000001" / "data.vtu"
            stale = root / "Cycle000002" / "stale.vtu"
            final = root / "Cycle000003" / "final.vtu"
            for path in (first, stale, final):
                _write_vtu(path, *values)
            collection = root / "result.pvd"
            _write_pvd(collection, [
                (0.0, "Cycle000001/data.vtu"),
                (0.2, "Cycle000002/stale.vtu"),
                (0.2, "Cycle000003/final.vtu"),
            ])
            selected = select_frame(
                root, 0.2, "dg-ssprk104", TEST_LIMITS, 1.0e-12, 1.0e-9
            )
            self.assertEqual(selected.path, final.resolve())
            self.assertEqual(selected.time, 0.2)

    def test_pvtu_combines_rank_pieces_without_deduplicating_dg_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cycle = root / "Cycle000007"
            values0 = _simple_fields()
            values1 = _simple_fields(0.2)
            _write_vtu(cycle / "proc000000.vtu", *values0, mode="binary")
            _write_vtu(cycle / "proc000001.vtu", *values1,
                       mode="compressed")
            _write_pvtu(cycle / "data.pvtu",
                        ["proc000000.vtu", "proc000001.vtu"])
            frame = Frame(0.25, cycle / "data.pvtu", root / "result.pvd")
            fields = load_frame(frame, root.resolve(), TEST_LIMITS)
            self.assertEqual(len(fields.points), 6)
            self.assertEqual(fields.points[2], fields.points[3])
            self.assertEqual(min(fields.density), 1.0)
            self.assertEqual(len(fields.momentum), 6)
            self.assertEqual(len(fields.energy), 6)
            self.assertAlmostEqual(max(reconstruct_pressure(fields, 1.4)), 4.0)


class QOITest(unittest.TestCase):
    def test_cylinder_shock_profile_position_standoff_and_thickness(self) -> None:
        x = [0.0, 0.1, 0.2, 0.3, 0.34]
        density = [1.0, 1.0, 1.0, 2.0, 2.0]
        momentum = [(rho, 0.0) for rho in density]
        physical_pressure = [1.0, 1.0, 1.0, 4.0, 4.0]
        fields = Fields(
            points=[(value, 0.0, 0.0) for value in x],
            density=density,
            momentum=momentum,
            energy=_energy_from_pressure(
                density, momentum, physical_pressure
            ),
            sigma=[0.0, 1.0, 2.0, 1.0, 0.0],
            pressure=[-100.0, -90.0, -80.0, -70.0, -60.0],
        )
        qoi, profile = compute_qoi(
            fields, case="cylinder", center_x=0.6, center_y=0.0,
            radius=0.25, profile_y=None, profile_x_min=None,
            profile_x_max=None, centerline_half_width=1.0e-6,
            profile_bin_width=1.0e-6, coordinate_tolerance=1.0e-12,
            thickness_length_scale=None, reference_profile=None,
            gamma=1.4,
        )
        self.assertIsNotNone(profile)
        self.assertAlmostEqual(qoi["shock_position"], 0.25)
        self.assertAlmostEqual(qoi["shock_standoff_distance"], 0.10)
        self.assertAlmostEqual(qoi["shock_thickness"], 0.4)
        self.assertEqual(qoi["density_minimum"], 1.0)
        self.assertEqual(qoi["density_maximum"], 2.0)
        self.assertAlmostEqual(qoi["pressure_minimum"], 1.0)
        self.assertAlmostEqual(qoi["pressure_maximum"], 4.0)
        self.assertAlmostEqual(qoi["sigma_localization"], 2.5)
        self.assertIsNone(qoi["conservation_error"])
        self.assertIsNone(qoi["boundary_balance_error"])

    def test_pressure_reconstruction_rejects_nonpositive_density_and_bad_gamma(
            self) -> None:
        points, density, momentum, energy, sigma, pressure = _simple_fields()
        zero_density = Fields(
            points=points,
            density=[0.0, *density[1:]],
            momentum=momentum,
            energy=energy,
            sigma=sigma,
            pressure=pressure,
        )
        with self.assertRaisesRegex(ShockQOIError, "nonpositive density"):
            reconstruct_pressure(zero_density, 1.4)
        negative_density = Fields(
            points=points,
            density=[-1.0, *density[1:]],
            momentum=momentum,
            energy=energy,
            sigma=sigma,
            pressure=pressure,
        )
        with self.assertRaisesRegex(ShockQOIError, "nonpositive density"):
            reconstruct_pressure(negative_density, 1.4)
        valid = Fields(points, density, momentum, energy, sigma, pressure)
        nonfinite_momentum = Fields(
            points, density,
            [(float("nan"), 0.0), *momentum[1:]],
            energy, sigma, pressure,
        )
        with self.assertRaisesRegex(ShockQOIError, "non-finite momentum"):
            reconstruct_pressure(nonfinite_momentum, 1.4)
        overflowing_momentum = Fields(
            points, density,
            [(1.0e308, 0.0), *momentum[1:]],
            energy, sigma, pressure,
        )
        with self.assertRaisesRegex(ShockQOIError, "non-finite"):
            reconstruct_pressure(overflowing_momentum, 1.4)
        for gamma in (1.0, float("nan"), float("inf")):
            with self.subTest(gamma=gamma), self.assertRaises(ShockQOIError):
                reconstruct_pressure(valid, gamma)

    def test_cli_rejects_nonphysical_or_nonfinite_gamma(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.mesh"
            mesh.write_text("synthetic", encoding="utf-8")
            for gamma in ("1.0", "nan", "inf"):
                stderr = io.StringIO()
                with self.subTest(gamma=gamma), redirect_stderr(stderr):
                    status = main([
                        "--vtk-root", str(root), "--method", "dg",
                        "--target-time", "0.1", "--mesh", str(mesh),
                        "--gamma", gamma,
                    ])
                self.assertEqual(status, 2)
                self.assertIn("gamma", stderr.getvalue())

    def test_reference_interpolation_has_exact_piecewise_linear_norms(self) -> None:
        computed = Profile((0.0, 1.0), (1.0, 2.0))
        reference = Profile((0.0, 0.25, 0.75, 1.0),
                            (0.0, 0.25, 0.75, 1.0))
        l1, l2 = profile_errors(computed, reference)
        self.assertAlmostEqual(l1, 1.0)
        self.assertAlmostEqual(l2, 1.0)

        tiny_overlap = Profile((0.99, 1.0, 1.01), (1.0, 1.0, 1.0))
        with self.assertRaisesRegex(ShockQOIError, "overlap"):
            profile_errors(computed, tiny_overlap)

    def test_reference_json_and_atomic_profile_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            mesh = Path(directory) / "mesh.mesh"
            mesh.write_text("synthetic mesh", encoding="utf-8")
            profile = Profile(
                tuple(index / 7.0 for index in range(8)),
                tuple(1.0 + index / 7.0 for index in range(8)),
            )
            extraction = {
                "profile_y": 0.0,
                "profile_x_min": 0.0,
                "profile_x_max": 1.0,
                "centerline_half_width": 0.1,
                "profile_bin_width": 0.01,
                "coordinate_tolerance": 1.0e-10,
                "thickness_length_scale": 1.0,
            }
            _write_profile(
                path, profile, method="dg-ssprk104", target_time=0.2,
                case="planar", mesh=str(mesh), gamma=1.4,
                extraction=extraction, overwrite=False,
            )
            self.assertEqual(_profile_from_json(path), profile)
            reference = _reference_profile_from_json(path)
            _validate_reference_profile(
                reference, case="planar", target_time=0.2, gamma=1.4,
                extraction=extraction, expected_method="dg-ssprk104",
                expected_mesh_sha256=reference.mesh_sha256,
                time_tolerance=1.0e-12, coordinate_tolerance=1.0e-10,
            )
            with self.assertRaisesRegex(ShockQOIError, "case mismatch"):
                _validate_reference_profile(
                    reference, case="shock-vortex", target_time=0.2,
                    gamma=1.4, extraction=extraction,
                    expected_method="dg-ssprk104",
                    expected_mesh_sha256=reference.mesh_sha256,
                    time_tolerance=1.0e-12,
                    coordinate_tolerance=1.0e-10,
                )
            with self.assertRaises(ShockQOIError):
                _write_profile(
                    path, profile, method="dg-ssprk104", target_time=0.2,
                    case="planar", mesh=str(mesh), gamma=1.4,
                    extraction=extraction, overwrite=False,
                )

    def test_harness_cli_emits_exact_qoi_schema_and_null_unobservables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.mesh"
            mesh.write_text("synthetic", encoding="utf-8")
            cycle = root / "Cycle000004"
            gamma = 1.6
            points0, density0, momentum0, energy0, sigma0, pressure0 = \
                _simple_fields(gamma=gamma)
            points1, density1, momentum1, energy1, sigma1, pressure1 = \
                _simple_fields(0.3, gamma)
            density1 = [0.8, 1.2, 2.5]
            momentum1 = [(0.8, 0.0), (1.2, 0.0), (2.5, 0.0)]
            physical_pressure1 = [2.0, 5.0, 6.0]
            energy1 = _energy_from_pressure(
                density1, momentum1, physical_pressure1, gamma
            )
            # Deliberately invalid as a diagnostic: the registered pressure
            # projection must not influence the independently reconstructed QoI.
            pressure1 = [-200.0, -300.0, -400.0]
            sigma1 = [4.0, 0.0, 0.0]
            _write_vtu(cycle / "proc000000.vtu", points0, density0,
                       momentum0, energy0, sigma0, pressure0, mode="ascii")
            _write_vtu(cycle / "proc000001.vtu", points1, density1,
                       momentum1, energy1, sigma1, pressure1,
                       mode="compressed")
            _write_pvtu(cycle / "data.pvtu",
                        ["proc000000.vtu", "proc000001.vtu"])
            _write_pvd(root / "run.pvd", [(0.4, "Cycle000004/data.pvtu")])
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main([
                    "--vtk-root", str(root), "--method", "dg-ssprk104",
                    "--target-time", "0.4", "--mesh", str(mesh),
                    "--case", "planar", "--gamma", str(gamma),
                ])
            self.assertEqual(status, 0, stderr.getvalue())
            self.assertEqual(stderr.getvalue(), "")
            decoded = json.loads(stdout.getvalue())
            self.assertEqual(set(decoded), set(QOI_FIELDS))
            self.assertEqual(decoded["density_minimum"], 0.8)
            self.assertEqual(decoded["density_maximum"], 2.5)
            self.assertAlmostEqual(decoded["pressure_minimum"], 2.0)
            self.assertAlmostEqual(decoded["pressure_maximum"], 6.0)
            self.assertIsNone(decoded["shock_position"])
            self.assertIsNone(decoded["conservation_error"])
            self.assertTrue(math.isfinite(decoded["sigma_localization"]))


class MalformedInputTest(unittest.TestCase):
    def test_missing_conserved_fields_nonfinite_and_counts_are_rejected(
            self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = _simple_fields()

            for field in ("density", "momentum", "energy"):
                missing = root / f"missing-{field}.vtu"
                _write_vtu(missing, *values, omit=field)
                with self.subTest(field=field), self.assertRaisesRegex(
                        ShockQOIError, "PointData field"):
                    _parse_vtu(missing, TEST_LIMITS)

            nonfinite = root / "nonfinite.vtu"
            bad_density = [1.0, float("nan"), 2.0]
            _write_vtu(
                nonfinite, values[0], bad_density, values[2], values[3],
                values[4], values[5],
            )
            with self.assertRaisesRegex(ShockQOIError, "finite"):
                _parse_vtu(nonfinite, TEST_LIMITS)

            wrong_count = root / "wrong-count.vtu"
            _write_vtu(
                wrong_count, values[0], values[1], values[2],
                values[3][:-1], values[4], values[5],
            )
            with self.assertRaisesRegex(ShockQOIError, "expected"):
                _parse_vtu(wrong_count, TEST_LIMITS)

            vector_energy = root / "vector-energy.vtu"
            _write_vtu(vector_energy, *values)
            vector_energy.write_text(
                vector_energy.read_text(encoding="utf-8").replace(
                    'Name="energy" NumberOfComponents="1"',
                    'Name="energy" NumberOfComponents="2"',
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ShockQOIError, "energy.*scalar"):
                _parse_vtu(vector_energy, TEST_LIMITS)

            bad_momentum = root / "bad-momentum.vtu"
            _write_vtu(
                bad_momentum, values[0], values[1], [(), (), ()],
                values[3], values[4], values[5],
            )
            with self.assertRaisesRegex(
                    ShockQOIError, "momentum NumberOfComponents"):
                _parse_vtu(bad_momentum, TEST_LIMITS)

            missing_projection = root / "no-projected-pressure.vtu"
            _write_vtu(missing_projection, *values[:-1])
            parsed = _parse_vtu(missing_projection, TEST_LIMITS)
            self.assertIsNone(parsed.pressure)
            self.assertAlmostEqual(max(reconstruct_pressure(parsed, 1.4)), 4.0)

    def test_target_time_mismatch_and_path_escape_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "vtk"
            root.mkdir()
            outside = parent / "outside.vtu"
            _write_vtu(outside, *_simple_fields())
            _write_pvd(root / "mismatch.pvd", [(0.1, "missing.vtu")])
            with self.assertRaisesRegex(ShockQOIError, "no PVD frame"):
                select_frame(root, 0.2, "dg", TEST_LIMITS, 1.0e-12, 1.0e-9)

            (root / "mismatch.pvd").unlink()
            _write_pvd(root / "escape.pvd", [(0.2, "../outside.vtu")])
            with self.assertRaisesRegex(ShockQOIError, "escapes VTK root"):
                select_frame(root, 0.2, "dg", TEST_LIMITS, 1.0e-12, 1.0e-9)

    def test_corrupt_binary_and_malformed_reference_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "corrupt.vtu"
            _write_vtu(path, *_simple_fields(), mode="binary")
            text = path.read_text(encoding="utf-8")
            first_payload = text.index("</DataArray>")
            text = text[:first_payload - 4] + "!!!!" + text[first_payload:]
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ShockQOIError, "base64"):
                _parse_vtu(path, TEST_LIMITS)

            reference = root / "reference.json"
            reference.write_text(
                json.dumps({
                    "schema_version": 2,
                    "case": "planar",
                    "method": "reference",
                    "target_time": 0.2,
                    "mesh": {"path": "mesh.mesh", "sha256": "0" * 64},
                    "gamma": 1.4,
                    "extraction": {
                        "profile_y": 0.0,
                        "profile_x_min": 0.0,
                        "profile_x_max": 1.0,
                        "centerline_half_width": 0.1,
                        "profile_bin_width": 0.01,
                        "coordinate_tolerance": 1.0e-10,
                        "thickness_length_scale": 1.0,
                    },
                    "profile": {
                        "coordinate": "x",
                        "x": [0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
                        "density": [1.0] * 8,
                    },
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ShockQOIError, "strictly increasing"):
                _profile_from_json(reference)

    def test_dtd_is_rejected_before_xml_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entity.vtu"
            path.write_text(
                '<!DOCTYPE x [<!ENTITY y "z">]><VTKFile type="UnstructuredGrid"/>',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ShockQOIError, "forbidden"):
                _parse_vtu(path, TEST_LIMITS)


if __name__ == "__main__":
    unittest.main()
