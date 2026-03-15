#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

TOL = 1.0e-10
QUAD_CONFIGS = {
    10: {
        "name": "quadratic",
        "line_type": 8,
        "mirror": [3, 2, 1, 0, 6, 5, 4, 7, 8],
        "edge_patterns": [
            (0, 1, 4),
            (1, 2, 5),
            (2, 3, 6),
            (3, 0, 7),
        ],
    },
    36: {
        "name": "cubic",
        "line_type": 26,
        "mirror": [3, 2, 1, 0, 9, 8, 7, 6, 5, 4, 11, 10, 15, 14, 13, 12],
        "edge_patterns": [
            (0, 1, 4, 5),
            (1, 2, 6, 7),
            (2, 3, 8, 9),
            (3, 0, 10, 11),
        ],
    },
}
PHYSICAL_NAMES = [
    (1, 1, "bottom"),
    (1, 2, "right"),
    (1, 3, "top"),
    (1, 4, "left"),
    (1, 5, "circle"),
    (2, 6, "domain"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the full supersonic-cylinder mesh by meshing the upper half, "
            "mirroring the quads, and rebuilding the boundary edges from the final 2D mesh."
        )
    )
    parser.add_argument("--geo", required=True, help="Upper-half Gmsh source (.geo).")
    parser.add_argument("--out", required=True, help="Output full-domain .msh path.")
    return parser.parse_args()


def parse_half_mesh(path: Path):
    lines = path.read_text().splitlines()
    nodes: dict[int, tuple[float, float, float]] = {}
    quads: list[tuple[list[int], list[int]]] = []
    surface_types: Counter[int] = Counter()
    i = 0
    while i < len(lines):
        tag = lines[i].strip()
        i += 1
        if tag == "$Nodes":
            count = int(lines[i].strip())
            i += 1
            for _ in range(count):
                parts = lines[i].split()
                i += 1
                nodes[int(parts[0])] = (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                )
            i += 1  # $EndNodes
        elif tag == "$Elements":
            count = int(lines[i].strip())
            i += 1
            for _ in range(count):
                parts = lines[i].split()
                i += 1
                etype = int(parts[1])
                ntags = int(parts[2])
                tags = list(map(int, parts[3 : 3 + ntags]))
                conn = list(map(int, parts[3 + ntags :]))
                if etype in {2, 3, 9, 10, 16, 20, 21, 22, 23, 24, 25, 36, 37, 38, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51}:
                    surface_types[etype] += 1
                if etype in QUAD_CONFIGS:
                    quads.append((tags, conn))
            i += 1  # $EndElements
    return nodes, quads, surface_types


def mirror_nodes(nodes: dict[int, tuple[float, float, float]]):
    mirrored: dict[int, int] = {}
    full_nodes = dict(nodes)
    next_id = max(nodes) + 1
    for nid, (x, y, z) in nodes.items():
        if abs(y) <= TOL:
            mirrored[nid] = nid
        else:
            mirrored[nid] = next_id
            full_nodes[next_id] = (x, -y, z)
            next_id += 1
    return full_nodes, mirrored


def build_full_quads(quads, mirrored, quad_type: int):
    full_quads = list(quads)
    mirror = QUAD_CONFIGS[quad_type]["mirror"]
    for tags, conn in quads:
        reflected = [mirrored[nid] for nid in conn]
        reflected = [reflected[idx] for idx in mirror]
        full_quads.append((tags, reflected))
    return full_quads


def classify_boundary(edge_nodes, coords):
    xs = [coords[nid][0] for nid in edge_nodes]
    ys = [coords[nid][1] for nid in edge_nodes]
    if max(abs(y - 1.0) for y in ys) < 1.0e-8:
        return 3
    if max(abs(y + 1.0) for y in ys) < 1.0e-8:
        return 1
    if max(abs(x - 0.0) for x in xs) < 1.0e-8:
        return 4
    if max(abs(x - 4.0) for x in xs) < 1.0e-8:
        return 2
    return 5


def rebuild_boundary(full_quads, coords, quad_type: int):
    edge_counts: Counter[tuple[int, int]] = Counter()
    oriented_edges: dict[tuple[int, int], tuple[int, ...]] = {}
    edge_patterns = QUAD_CONFIGS[quad_type]["edge_patterns"]
    for _, conn in full_quads:
        for pattern in edge_patterns:
            edge = tuple(conn[idx] for idx in pattern)
            key = tuple(sorted(edge[:2]))
            edge_counts[key] += 1
            oriented_edges.setdefault(key, edge)

    boundary = []
    for key, count in edge_counts.items():
        if count != 1:
            continue
        edge = oriented_edges[key]
        boundary.append((classify_boundary(edge, coords), edge))
    return boundary


def write_full_mesh(path: Path, nodes, boundary, quads, quad_type: int):
    line_type = QUAD_CONFIGS[quad_type]["line_type"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        handle.write("$PhysicalNames\n")
        handle.write(f"{len(PHYSICAL_NAMES)}\n")
        for dim, tag, name in PHYSICAL_NAMES:
            handle.write(f'{dim} {tag} "{name}"\n')
        handle.write("$EndPhysicalNames\n")

        handle.write("$Nodes\n")
        handle.write(f"{len(nodes)}\n")
        for nid in sorted(nodes):
            x, y, z = nodes[nid]
            handle.write(f"{nid} {x:.16g} {y:.16g} {z:.16g}\n")
        handle.write("$EndNodes\n")

        total_elements = len(boundary) + len(quads)
        handle.write("$Elements\n")
        handle.write(f"{total_elements}\n")
        eid = 1
        for phys, edge in boundary:
            conn = " ".join(map(str, edge))
            handle.write(f"{eid} {line_type} 2 {phys} {phys} {conn}\n")
            eid += 1
        for tags, conn in quads:
            conn_text = " ".join(map(str, conn))
            phys = tags[0] if tags else 6
            handle.write(f"{eid} {quad_type} 2 {phys} 1 {conn_text}\n")
            eid += 1
        handle.write("$EndElements\n")


def main() -> None:
    args = parse_args()
    geo = Path(args.geo).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        half_mesh = Path(tmpdir) / "half.msh"
        subprocess.run(
            ["gmsh", "-2", str(geo), "-o", str(half_mesh)],
            check=True,
        )
        nodes, quads, surface_types = parse_half_mesh(half_mesh)
        if not quads:
            raise RuntimeError(
                f"No supported quadrilateral elements were found in {half_mesh}. "
                f"Supported types: {sorted(QUAD_CONFIGS)}"
            )
        quad_types = {etype for etype, count in surface_types.items() if count and etype in QUAD_CONFIGS}
        if len(quad_types) != 1:
            raise RuntimeError(
                f"Half mesh {half_mesh} must contain exactly one supported quad type; found {sorted(quad_types)}"
            )
        quad_type = next(iter(quad_types))
        unsupported = {
            etype: count
            for etype, count in surface_types.items()
            if count and etype != quad_type
        }
        if unsupported:
            raise RuntimeError(
                f"Half mesh {half_mesh} contains unsupported 2D element types: {unsupported}"
            )
        full_nodes, mirrored = mirror_nodes(nodes)
        full_quads = build_full_quads(quads, mirrored, quad_type)
        boundary = rebuild_boundary(full_quads, full_nodes, quad_type)
        write_full_mesh(out, full_nodes, boundary, full_quads, quad_type)

        counts = Counter()
        counts[QUAD_CONFIGS[quad_type]["line_type"]] = len(boundary)
        counts[quad_type] = len(full_quads)
        print(f"wrote {out}")
        print(dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
