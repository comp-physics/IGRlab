#!/usr/bin/env python3
"""Compare saved DGSEM polynomial fields at common physical quadrature points.

Supported input is rank-one MFEM XML VTK: one type-70 Lagrange quadrilateral
per DG cell, equally spaced local output nodes, a uniform axis-aligned Cartesian
mesh, and density, two-component momentum, total energy, and sigma point data.
The compared objects are the saved tensor-polynomial interpolants, not raw point
arrays or derived pressure/velocity interpolants. Declare each run's native
order and sigma space. An optional output order describes oversampled VTK;
output below either native state or known sigma degree is rejected.

Uses NumPy and the repository's existing PVD/piece/binary readers. Default
quadrature is five Gauss points per direction on 24 subdivisions, augmented
with every cell boundary of both inputs so DG jumps never cross panel interiors.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


READER_PATH = Path(__file__).resolve().parents[2] / "tests/scripts/compare_vtk_pvd_fields.py"
_SPEC = importlib.util.spec_from_file_location("vtk_field_readers", READER_PATH)
READERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(READERS)
FIELDS = ("density", "momentum_x", "momentum_y", "energy", "sigma")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_array(node, count, components=1):
    require(node is not None, "missing VTK DataArray")
    require(int(node.get("NumberOfComponents", "1")) == components,
            "incorrect VTK component count")
    kind = node.get("type", "")
    if node.get("format") == "binary":
        compact = "".join((node.text or "").split())
        require(len(compact) >= 8, "missing VTK binary header")
        header = base64.b64decode(compact[:8], validate=True)
        payload = base64.b64decode(compact[8:], validate=True)
        require(len(header) == 4 and int.from_bytes(header, "little") == len(payload),
                "VTK binary payload length disagrees with header")
        result = np.asarray(READERS.read_vtk_binary_data(node.text, kind))
    elif node.get("format") == "ascii":
        dtype = float if kind in ("Float32", "Float64") else int
        READERS.typecode_for_vtk_type(kind)  # Reject unsupported types.
        result = np.asarray([dtype(v) for v in (node.text or "").split()])
    else:
        raise ValueError("only inline binary/ASCII VTK arrays are supported")
    require(result.size == count * components, "VTK array length mismatch")
    require(np.isfinite(result).all(), "nonfinite VTK data")
    return result.reshape(count, components) if components != 1 else result


def clusters(values, tolerance):
    groups = []
    for value in sorted(values):
        if not groups or value - groups[-1][0] > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return np.array([math.fsum(group) / len(group) for group in groups])


def vtk_quad_index(i, j, p):
    """MFEM/VTK tensor node convention; cf. tpl/mfem/mesh/vtk.cpp."""
    if i in (0, p) and j in (0, p):
        return (2 if j else 1) if i else (3 if j else 0)
    if j in (0, p):
        return 4 + i - 1 + (2 * (p - 1) if j else 0)
    if i in (0, p):
        return 4 + j - 1 + ((p - 1) if i else 3 * (p - 1))
    return 4 + 4 * (p - 1) + i - 1 + (p - 1) * (j - 1)


def load_mesh(dataset, order):
    require(1 <= order <= 8, "supported polynomial orders are 1 through 8")
    paths = list(READERS.iter_piece_paths(dataset))
    require(len(paths) == 1, "only one rank/piece is supported")
    root = ET.parse(paths[0]).getroot()
    require(root.get("type") == "UnstructuredGrid" and
            root.get("byte_order", "LittleEndian") == "LittleEndian" and
            root.get("header_type", "UInt32") == "UInt32" and
            root.get("compressor") is None, "unsupported VTK encoding/grid")
    pieces = root.findall("./UnstructuredGrid/Piece")
    require(len(pieces) == 1, "only one UnstructuredGrid piece is supported")
    piece = pieces[0]
    npoints, ncells = int(piece.get("NumberOfPoints")), int(piece.get("NumberOfCells"))
    n = order + 1
    require(ncells > 0 and npoints == ncells * n * n,
            "declared native order disagrees with disconnected VTK cell nodes")
    points = read_array(piece.find("./Points/DataArray"), npoints, 3)
    offsets = read_array(piece.find("./Cells/DataArray[@Name='offsets']"), ncells)
    types = read_array(piece.find("./Cells/DataArray[@Name='types']"), ncells)
    connectivity = read_array(piece.find("./Cells/DataArray[@Name='connectivity']"), npoints)
    require(np.array_equal(offsets, np.arange(1, ncells + 1) * n * n),
            "cell offsets do not describe uniform tensor cells of declared order")
    require(np.all(types == 70), "only VTK_LAGRANGE_QUADRILATERAL (70) is supported")
    require(np.issubdtype(connectivity.dtype, np.integer) and
            np.array_equal(np.sort(connectivity), np.arange(npoints)),
            "connectivity must use every disconnected point exactly once")
    arrays = []
    for name, components in (("density", 1), ("momentum", 2), ("energy", 1), ("sigma", 1)):
        nodes = piece.findall(f"./PointData/DataArray[@Name='{name}']")
        require(len(nodes) == 1, f"missing/duplicate point field {name}")
        arrays.append(read_array(nodes[0], npoints, components).reshape(npoints, components))
    fields = np.concatenate(arrays, axis=1)
    tolerance = 256 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(points))))
    require(np.max(np.abs(points[:, 2])) <= tolerance, "mesh must lie in z=0")
    cells = []
    for ids in connectivity.reshape(ncells, n * n):
        xy = points[ids, :2]
        axes = [clusters(xy[:, axis], tolerance) for axis in range(2)]
        require(all(len(axis) == n for axis in axes), "cell is not an axis-aligned tensor grid")
        for axis in axes:
            require(axis[-1] - axis[0] > 100 * tolerance and
                    np.max(np.abs(axis - np.linspace(axis[0], axis[-1], n))) <= tolerance,
                    "cell output nodes must be uniformly spaced")
        local = np.empty((n, n, len(FIELDS)))
        seen = set()
        for position, point_id in enumerate(ids):
            i, j = [int(np.argmin(np.abs(axes[a] - xy[position, a]))) for a in range(2)]
            require((i, j) not in seen and vtk_quad_index(i, j, order) == position,
                    "malformed or unsupported rotated VTK Lagrange connectivity")
            seen.add((i, j))
            local[j, i] = fields[point_id]
        cells.append((axes[0][0], axes[0][-1], axes[1][0], axes[1][-1], local))
    x_edges = clusters([v for cell in cells for v in cell[:2]], tolerance)
    y_edges = clusters([v for cell in cells for v in cell[2:4]], tolerance)
    for edges in (x_edges, y_edges):
        require(np.max(np.abs(np.diff(edges) - (edges[-1] - edges[0]) / (len(edges) - 1)))
                <= tolerance, "mesh is not uniform Cartesian")
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    require(nx * ny == ncells, "Cartesian mesh has missing/overlapping cells")
    data = np.empty((ny, nx, n, n, len(FIELDS)))
    seen = set()
    for x0, x1, y0, y1, local in cells:
        ix = int(np.argmin(abs(x_edges - x0)))
        iy = int(np.argmin(abs(y_edges - y0)))
        require(ix < nx and iy < ny and (ix, iy) not in seen and
                abs(x_edges[ix + 1] - x1) <= tolerance and
                abs(y_edges[iy + 1] - y1) <= tolerance, "invalid Cartesian cell coverage")
        seen.add((ix, iy))
        data[iy, ix] = local
    return {"x": x_edges, "y": y_edges, "values": data, "order": order,
            "tolerance": tolerance, "piece": str(paths[0]),
            "piece_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest()}


def lagrange_basis(coordinate, order):
    nodes = np.linspace(0, 1, order + 1)
    basis = np.ones((len(coordinate), order + 1))
    for j in range(order + 1):
        for k in range(order + 1):
            if j != k:
                basis[:, j] *= (coordinate - nodes[k]) / (nodes[j] - nodes[k])
    return basis


def evaluate(mesh, x, y):
    indices = [np.searchsorted(mesh[a], point, side="right") - 1
               for a, point in (("x", x), ("y", y))]
    for axis, index in zip(("x", "y"), indices):
        require(np.all((index >= 0) & (index < len(mesh[axis]) - 1)),
                "sample outside the mesh interior")
    ix, iy = indices
    tx = (x - mesh["x"][ix]) / (mesh["x"][ix + 1] - mesh["x"][ix])
    ty = (y - mesh["y"][iy]) / (mesh["y"][iy + 1] - mesh["y"][iy])
    values = np.einsum("qi,qj,qjic->qc", lagrange_basis(tx, mesh["order"]),
                       lagrange_basis(ty, mesh["order"]), mesh["values"][iy, ix])
    require(np.isfinite(values).all(), "nonfinite reconstructed polynomial")
    return values


def compare(a, b, subdivisions=24, gauss=5):
    require(subdivisions > 0 and gauss >= max(a["order"], b["order"]) + 1,
            "Gauss count must exceed both polynomial orders for exact squared-error integration")
    tolerance = max(a["tolerance"], b["tolerance"])
    coordinates, weights, panels = [], [], []
    nodes, gauss_weights = np.polynomial.legendre.leggauss(gauss)
    for axis in ("x", "y"):
        require(np.max(abs(a[axis][[0, -1]] - b[axis][[0, -1]])) <= tolerance,
                "physical domains differ")
        edges = clusters(np.concatenate((a[axis], b[axis],
            np.linspace(a[axis][0], a[axis][-1], subdivisions + 1))), tolerance)
        half = np.diff(edges) / 2
        coordinates.append(((edges[:-1, None] + edges[1:, None]) / 2 +
                            half[:, None] * nodes).ravel())
        weights.append((half[:, None] * gauss_weights).ravel())
        panels.append(len(half))
    x, y = np.meshgrid(*coordinates)
    weight = np.outer(weights[1], weights[0]).ravel()
    av, bv = evaluate(a, x.ravel(), y.ravel()), evaluate(b, x.ravel(), y.ravel())
    difference = bv - av
    result = {"quadrature": {"requested_subdivisions_per_axis": subdivisions,
              "actual_panels_xy": panels, "gauss_points_per_direction": gauss,
              "sample_count": len(weight), "squared_polynomial_error_exact_in_exact_arithmetic": True},
              "fields": {}}
    for column, name in enumerate(FIELDS):
        l2 = math.sqrt(float(np.dot(weight, difference[:, column] ** 2)))
        reference_l2 = math.sqrt(float(np.dot(weight, av[:, column] ** 2)))
        result["fields"][name] = {"l2_absolute": l2, "reference_l2": reference_l2,
            "l2_relative": l2 / reference_l2 if reference_l2 else (0.0 if l2 == 0 else None),
            "max_sample_absolute": float(np.max(abs(difference[:, column]))),
            "reference_integral": float(np.dot(weight, av[:, column])),
            "candidate_integral": float(np.dot(weight, bv[:, column])),
            "integral_difference": float(np.dot(weight, difference[:, column]))}
    result["conserved_integrals"] = {}
    for role, key in (("reference", "reference_integral"), ("candidate", "candidate_integral"),
                      ("difference", "integral_difference")):
        result["conserved_integrals"][role] = {
            "mass": result["fields"]["density"][key],
            "momentum_xy": [result["fields"][name][key] for name in FIELDS[1:3]],
            "total_energy": result["fields"]["energy"][key]}
    return result


def select_frame(pvd, target_time, tolerance):
    require(pvd.suffix.lower() == ".pvd", "PVD input is required for time validation")
    collection = ET.parse(pvd).getroot()
    datasets = collection.findall("./Collection/DataSet")
    require(collection.get("type") == "Collection" and datasets and
            all(record.get("file") and record.get("timestep") is not None for record in datasets),
            "malformed PVD collection")
    frames = READERS.parse_pvd(pvd)
    times = [float(frame["timestep"]) for frame in frames]
    require(times and all(math.isfinite(t) for t in times) and
            all(b > a for a, b in zip(times, times[1:])), "invalid/nonmonotone PVD times")
    if target_time is None:
        return frames[-1]
    matches = [frame for frame in frames if abs(float(frame["timestep"]) - target_time) <= tolerance]
    require(len(matches) == 1, "requested time is absent or ambiguous")
    return matches[0]


def validate_orders(native, output, sigma_space):
    require(1 <= native <= 8, "supported native polynomial orders are 1 through 8")
    represented = native if output is None else output
    sigma_order = max(native, 3) if sigma_space == "H1" else native
    require(1 <= represented <= 8 and represented >= native,
            "VTK output order must be at least the native state order and at most 8")
    require(represented >= sigma_order,
            "H1 sigma at p1/p2 is undersampled; output order must reach sigma degree 3")
    return represented, sigma_order


def self_test():
    """Exercise the real XML reader with reordered point IDs and unequal grids."""
    def polynomial(x, y):
        return np.array([1 + x + y*y, x*y - .2, x*x + .3*y, 4 + x*x*y*y, 2*x - y*y])

    def write_fixture(path, p, nx):
        points, values, connectivity = [], [], []
        n = p + 1
        for cy in range(nx):
            for cx in range(nx):
                local = [None] * (n*n)
                for j in range(n):
                    for i in range(n):
                        x, y = (cx + i/p)/nx, (cy + j/p)/nx
                        local[vtk_quad_index(i, j, p)] = len(points)
                        points.append([x, y, 0]); values.append(polynomial(x, y))
                connectivity.extend(local)
        # Deliberately reverse global point storage, preserving connectivity.
        points, values = np.array(points)[::-1], np.array(values)[::-1]
        connectivity = len(points) - 1 - np.array(connectivity)
        root = ET.Element("VTKFile", type="UnstructuredGrid", byte_order="LittleEndian")
        piece = ET.SubElement(ET.SubElement(root, "UnstructuredGrid"), "Piece",
                              NumberOfPoints=str(len(points)), NumberOfCells=str(nx*nx))
        def array(parent, data, kind, name=None, components=1):
            attributes = {"type": kind, "format": "ascii", "NumberOfComponents": str(components)}
            if name: attributes["Name"] = name
            ET.SubElement(parent, "DataArray", **attributes).text = " ".join(map(str, np.ravel(data)))
        array(ET.SubElement(piece, "Points"), points, "Float64", components=3)
        cells = ET.SubElement(piece, "Cells")
        array(cells, connectivity, "Int32", "connectivity")
        array(cells, np.arange(1, nx*nx+1)*n*n, "Int32", "offsets")
        array(cells, np.full(nx*nx, 70), "UInt8", "types")
        data = ET.SubElement(piece, "PointData")
        for name, columns in (("density", [0]), ("momentum", [1, 2]), ("energy", [3]), ("sigma", [4])):
            array(data, values[:, columns], "Float64", name, len(columns))
        ET.ElementTree(root).write(path)
    with tempfile.TemporaryDirectory(prefix="dgsem-physical-fields-") as temporary:
        pa, pb = Path(temporary)/"a.vtu", Path(temporary)/"b.vtu"
        write_fixture(pa, 2, 2); write_fixture(pb, 4, 3)
        represented, sigma_order = validate_orders(2, 4, "L2")
        require(represented == 4 and sigma_order == 2, "native/output order metadata self-test failed")
        a, b = load_mesh(pa, 2), load_mesh(pb, represented)
        report = compare(a, b, 4, 5)
        expected = (11/6, .05, 29/60, 37/9, 2/3)
        for name, integral in zip(FIELDS, expected):
            require(report["fields"][name]["l2_absolute"] < 1e-13 and
                    abs(report["fields"][name]["reference_integral"] - integral) < 1e-13,
                    "polynomial reconstruction/integration self-test failed")
        # A discontinuous cellwise constant offset tests partition augmentation.
        b["values"][0, 0, :, :, 0] += .25
        shifted = compare(a, b, 4, 5)["fields"]["density"]
        require(abs(shifted["l2_absolute"] - .25/3) < 1e-13 and
                abs(shifted["integral_difference"] - .25/9) < 1e-13,
                "piecewise-polynomial quadrature self-test failed")
        tree = ET.parse(pb)
        tree.find(".//Cells/DataArray[@Name='types']").text = " ".join(["9"]*9)
        tree.write(pb)
        try:
            load_mesh(pb, 4)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed cell-type self-test failed")
    return {"self_test": "passed", "coverage": ["unequal orders/meshes", "permuted point storage",
            "native/output order distinction", "analytic integrals", "DG discontinuity integration",
            "cell type rejection"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, nargs="?")
    parser.add_argument("candidate", type=Path, nargs="?")
    for role in ("reference", "candidate"):
        parser.add_argument(f"--{role}-order", type=int)
        parser.add_argument(f"--{role}-output-order", type=int,
                            help="represented VTK order; defaults to the native order")
        parser.add_argument(f"--{role}-sigma-space", choices=("L2", "H1"))
        parser.add_argument(f"--{role}-c-alpha", type=float)
    parser.add_argument("--time", type=float, help="default: final saved frame in each PVD")
    parser.add_argument("--time-tolerance", type=float, default=1e-12)
    parser.add_argument("--subdivisions", type=int, default=24)
    parser.add_argument("--gauss", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            report = self_test()
        else:
            require(math.isfinite(args.time_tolerance) and args.time_tolerance >= 0 and
                    (args.time is None or math.isfinite(args.time)), "invalid requested time/tolerance")
            meshes, identities = [], []
            for role in ("reference", "candidate"):
                path, order, space = (getattr(args, role), getattr(args, role+"_order"),
                                      getattr(args, role+"_sigma_space"))
                require(path is not None and order is not None and space is not None,
                        "both PVD paths, native orders, and sigma spaces must be explicit")
                output_order, sigma_order = validate_orders(
                    order, getattr(args, role+"_output_order"), space)
                frame = select_frame(path, args.time, args.time_tolerance)
                mesh = load_mesh(Path(frame["file"]), output_order)
                meshes.append(mesh)
                identities.append({"pvd": str(path.resolve()), "time": frame["timestep"],
                    "piece": mesh["piece"], "piece_sha256": mesh["piece_sha256"],
                    "declared_native_order": order, "represented_order": output_order,
                    "declared_sigma_space": space, "declared_sigma_order": sigma_order,
                    "cells_xy": [len(mesh["x"])-1, len(mesh["y"])-1],
                    "domain_xy": [mesh["x"][[0, -1]].tolist(), mesh["y"][[0, -1]].tolist()]})
            gap = abs(identities[0]["time"] - identities[1]["time"])
            require(gap <= args.time_tolerance, "saved physical times differ")
            ca = (args.reference_c_alpha, args.candidate_c_alpha)
            require((ca[0] is None) == (ca[1] is None), "supply both c-alpha values or neither")
            alpha = None
            if ca[0] is not None:
                require(all(math.isfinite(c) and c > 0 for c in ca), "invalid c-alpha")
                alpha = [c * min(np.diff(mesh["x"])[0], np.diff(mesh["y"])[0])**2
                         for c, mesh in zip(ca, meshes)]
                require(math.isclose(*alpha, rel_tol=1e-12, abs_tol=0),
                        "local physical alpha=C*min(dx,dy)^2 differs; rescale C with mesh refinement")
            report = compare(*meshes, args.subdivisions, args.gauss)
            report.update({"reference": identities[0], "candidate": identities[1],
                "time_difference": gap, "time_tolerance": args.time_tolerance,
                "interpretation": "errors of represented conservative/sigma tensor-polynomial VTK fields",
                "alpha_check": {"status": "matched" if alpha is not None else "not checked",
                    "policy": "local C*min(dx,dy)^2 for Cartesian affine quads", "values": alpha}})
        rendered = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            args.output.write_text(rendered + "\n")
        print(rendered)
    except (ValueError, TypeError, OSError, ET.ParseError, OverflowError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
