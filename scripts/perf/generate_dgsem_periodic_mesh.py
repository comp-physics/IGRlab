#!/usr/bin/env python3
"""Generate the miniapp's uniform periodic L2-P1 geometry mesh convention."""
import argparse
from pathlib import Path


def mesh_text(n):
    if n < 3:
        raise ValueError("Use at least three cells per direction")
    vertex = lambda i, j: (j % n) * n + (i % n)
    lines = ["MFEM mesh v1.0", "", "dimension", "2", "", "elements", str(n*n)]
    for j in range(n):
        for i in range(n):
            corners = [vertex(i, j), vertex(i+1, j),
                       vertex(i+1, j+1), vertex(i, j+1)]
            lines.append("1 3 " + " ".join(map(str, corners)))
    lines += ["", "boundary", str(4*n)]
    for attribute, j in ((1, 0), (3, n-1)):
        for i in range(n):
            lines.append(f"{attribute} 1 {vertex(i, j)} {vertex(i+1, j)}")
    for attribute, i in ((2, 0), (4, n-1)):
        for j in range(n):
            lines.append(f"{attribute} 1 {vertex(i, j)} {vertex(i, j+1)}")
    lines += ["", "vertices", str(n*n), "", "nodes", "FiniteElementSpace",
              "FiniteElementCollection: L2_T1_2D_P1", "VDim: 2", "Ordering: 1", ""]
    for j in range(n):
        for i in range(n):
            for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
                lines.append(f"{(i+di)/n:.17g} {(j+dj)/n:.17g}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-n6", type=Path)
    args = parser.parse_args()
    if args.verify_n6:
        assert mesh_text(6).encode() == args.verify_n6.read_bytes(), "N6 differs"
        print("Existing N6 fixture reproduced byte for byte")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(mesh_text(args.n))
    print(f"Wrote {args.n} x {args.n} periodic quadrilateral mesh: {args.output}")
