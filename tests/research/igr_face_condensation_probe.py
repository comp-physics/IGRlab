#!/usr/bin/env python3
"""Bounded algebraic screen of element-local IGR bubble elimination.

Consumes a frozen *production* operator export; it never constructs physics.
Dense arrays and eigensolves are research references, not a GPU implementation.
See docs/research/igr_face_locality_derivation.md for interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy import linalg
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, spsolve
from scipy.spatial import cKDTree


def norm(x):
    return float(linalg.norm(x))


def ratio(x, y):
    return float(x / y) if y else (0.0 if not x else None)


def lagrange_values(nodes, points):
    """Cardinal interpolation, including exact equality at nodal points."""
    nodes, points = np.asarray(nodes), np.asarray(points)
    weights = np.array([1.0 / np.prod(nodes[j] - np.delete(nodes, j))
                        for j in range(len(nodes))])
    values = np.empty((len(points), len(nodes)))
    for i, point in enumerate(points):
        hits = np.flatnonzero(np.abs(nodes - point) < 8 * np.finfo(float).eps)
        if len(hits):
            values[i] = 0
            values[i, hits[0]] = 1
        else:
            row = weights / (point - nodes)
            values[i] = row / row.sum()
    return values


def load_export(path):
    raw = json.loads(Path(path).read_text())
    matrix = raw.get("A", raw.get("matrix"))
    if isinstance(matrix, dict):
        a = sparse.coo_matrix((matrix["values"], (matrix["rows"], matrix["cols"])),
                              shape=(raw["n"], raw["n"])).tocsr()
        if raw["n"] <= 1600:
            a = a.toarray()
    else:
        a = np.asarray(matrix, dtype=float)
    rhs = np.asarray(raw.get("rhs", raw.get("b")), dtype=float)
    if "elements" in raw:
        elements = raw["elements"]
        dofs = [np.asarray(e["dofs"], dtype=int) for e in elements]
        nodes = [np.asarray(e.get("nodes", raw.get("reference_nodes")), dtype=float)
                 for e in elements]
        masses = [np.asarray(e["mass"], dtype=float) for e in elements]
    else:
        dofs = [np.asarray(e, dtype=int) for e in raw["element_dofs"]]
        pts = raw["reference_nodes"]
        nodes = [np.asarray(pts, dtype=float) for _ in dofs]
        masses = [np.asarray(m, dtype=float)
                  for m in raw.get("mass_blocks", raw.get("element_mass"))]
    if a.ndim != 2 or a.shape[0] != a.shape[1] or rhs.shape != (a.shape[0],):
        raise ValueError("Expected a square matrix and one matching closure RHS")
    if not np.array_equal(np.sort(np.concatenate(dofs)), np.arange(a.shape[0])):
        raise ValueError("Probe requires unsigned disjoint element DOFs")
    return raw, a, rhs, dofs, nodes, masses


def hermite_jet_values(p, points):
    """Cubic endpoint value/derivative functions plus double-zero bubbles."""
    x = np.asarray(points)
    columns = [2*x**3 - 3*x**2 + 1, -2*x**3 + 3*x**2,
               x**3 - 2*x**2 + x, x**3 - x**2]
    columns += [16*x*x*(1-x)**2 * Legendre.basis(k)(2*x-1)
                for k in range(p - 3)]
    return np.column_stack(columns)


def volume_face_attenuation(raw, a, rhs, dofs, masses):
    """A_theta=(1-theta)A_volume+theta*A, a changed discrete closure."""
    if "volume_blocks" not in raw:
        return {"available": False}
    row, col, values = [], [], []
    reaction_one = np.zeros_like(rhs)
    for ids, block, reaction in zip(dofs, raw["volume_blocks"], raw["reaction_blocks"]):
        block = np.asarray(block)
        reaction_one[ids] = np.asarray(reaction) @ np.ones(len(ids))
        for i, r in enumerate(ids):
            for j, c in enumerate(ids):
                row.append(r); col.append(c); values.append(block[i, j])
    volume = sparse.coo_matrix((values, (row, col)), shape=a.shape).tocsr()
    is_sparse = sparse.issparse(a)
    if not is_sparse:
        volume = volume.toarray()

    def solve(matrix, vector):
        return spsolve(matrix, vector) if is_sparse else linalg.solve(matrix, vector, assume_a="pos")

    def physical_norm(vector):
        return np.sqrt(max(0.0, sum(float(vector[ids] @ m @ vector[ids])
                                    for ids, m in zip(dofs, masses))))

    reference = solve(a, rhs)
    reference_l2, bnorm = physical_norm(reference), norm(rhs)
    results = []
    for theta in (0.0, 0.25, 0.5, 0.75, 0.99, 0.999, 1.0):
        matrix = (1 - theta) * volume + theta * a
        value = solve(matrix, rhs)
        results.append({"theta": theta,
            "modified_matrix_residual_relative": ratio(norm(matrix @ value - rhs), bnorm),
            "original_residual_relative": ratio(norm(a @ value - rhs), bnorm),
            "physical_l2_error_relative": ratio(physical_norm(value - reference), reference_l2),
            "constant_response_residual_relative": ratio(
                norm(matrix @ np.ones(len(rhs)) - reaction_one), norm(reaction_one))})
    return {"available": True, "family": "A_theta=(1-theta)A_volume+theta*A",
            "same_source_rhs": True,
            "claim": "theta!=1 changes discrete spatial closure; constant preservation alone is insufficient",
            "samples": results}


def sparse_p1_face_screen(raw, a, rhs, dofs, nodes, masses, tolerance, max_iterations,
                         previous_path=None):
    """P1 has no bubbles, but its exact basis change enables tiny vertex blocks."""
    vertices = raw.get("element_vertices")
    if vertices is None:
        return {"available": False, "reason": "curved-mesh face blocks require vertex topology"}
    rows, columns, values = [], [], []
    grouped = {}
    for ids, points, vertex in zip(dofs, nodes, vertices):
        lx = lagrange_values([0.0, 1.0], points[:, 0])
        ly = lagrange_values([0.0, 1.0], points[:, 1])
        t = np.einsum("aj,ai->aji", ly, lx).reshape(4, 4)
        for i, row in enumerate(ids):
            for j, column in enumerate(ids):
                rows.append(row); columns.append(column); values.append(t[i, j])
        # MFEM quadrilateral vertices are cyclic; tensor columns are x-fast.
        for dof, vertex_id in zip(ids, np.asarray(vertex)[[0, 1, 3, 2]]):
            grouped.setdefault(int(vertex_id), []).append(int(dof))
    transform = sparse.coo_matrix((values, (rows, columns)), shape=a.shape).tocsr()
    transformed = (transform.T @ a @ transform).tocsr()

    def block_inverse(matrix, groups):
        row, col, value = [], [], []
        max_block = 0
        for group in groups:
            block = matrix[np.ix_(group, group)].toarray()
            inverse = linalg.cho_solve(linalg.cho_factor(block, lower=True), np.eye(len(group)))
            max_block = max(max_block, len(group))
            for i, r in enumerate(group):
                for j, c in enumerate(group):
                    row.append(r); col.append(c); value.append(inverse[i, j])
        return sparse.coo_matrix((value, (row, col)), shape=a.shape).tocsr(), max_block

    face_inverse, max_block = block_inverse(transformed, list(grouped.values()))
    element_inverse, _ = block_inverse(a, dofs)
    reference = spsolve(a, rhs)
    bnorm = norm(rhs)
    initial = np.zeros_like(rhs)
    previous_identity = None
    if previous_path is not None:
        previous_raw, previous_a, previous_rhs, previous_dofs, _, _ = load_export(previous_path)
        if (previous_a.shape != a.shape or len(previous_dofs) != len(dofs) or
                not np.allclose(previous_raw["dof_coordinates"], raw["dof_coordinates"],
                                rtol=0, atol=1e-12) or
                not all(np.array_equal(x, y) for x, y in zip(previous_dofs, dofs))):
            raise ValueError("Warm-start export must use the same element DOFs")
        initial = spsolve(previous_a, previous_rhs)
        previous_identity = {"path": str(Path(previous_path).resolve()),
            "sha256": hashlib.sha256(Path(previous_path).read_bytes()).hexdigest()}

    def physical_norm(vector):
        return np.sqrt(max(0.0, sum(float(vector[ids] @ m @ vector[ids])
                                    for ids, m in zip(dofs, masses))))

    def run(label, apply):
        count = [0]

        def callback(_):
            count[0] += 1

        begin = time.perf_counter()
        value, status = cg(a, rhs, x0=initial, M=LinearOperator(a.shape, matvec=apply), tol=tolerance,
                           atol=0.0, maxiter=max_iterations, callback=callback)
        return {"method": label, "iterations": count[0], "status": int(status),
                "original_residual_relative": norm(a @ value - rhs) / bnorm,
                "physical_l2_error_relative": physical_norm(value - reference) / physical_norm(reference),
                "host_solve_seconds": time.perf_counter() - begin}

    results = [run("point-jacobi", lambda x: x / a.diagonal()),
               run("original-element-block", lambda x: element_inverse @ x),
               run("transformed-vertex-block", lambda x: transform @ (face_inverse @ (transform.T @ x)))]
    return {"available": True, "reference_original_residual_relative": norm(a @ reference - rhs) / bnorm,
            "previous_export": previous_identity,
            "initial_original_residual_relative": norm(a @ initial - rhs) / bnorm,
            "block_count": len(grouped), "block_max_dofs": max_block,
            "inverse_entries": face_inverse.nnz, "transform_entries": transform.nnz,
            "transformed_matrix_entries": transformed.nnz, "solves": results,
            "claim": "same production matrix, CPU algebraic screen, no GPU speedup"}


def analyze(path, tolerance, max_iterations, basis="lgl", previous_path=None):
    started = time.perf_counter()
    raw, a, rhs, dofs, nodes, masses = load_export(path)
    n = a.shape[0]
    p = int(round(np.sqrt(len(dofs[0])))) - 1
    if previous_path is not None and not sparse.issparse(a):
        raise ValueError("--previous-export is currently supported only for the sparse p1 screen")
    if basis == "hermite-jet" and p < 3:
        raise ValueError("The value/normal-derivative basis requires p>=3")
    if sparse.issparse(a):
        if p != 1:
            raise ValueError("Dense condensation screen capped at 1600 DOFs; large p1 is structural only")
        return {"schema": "igr-face-condensation-v1", "input": str(Path(path).resolve()),
                "input_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "metadata": raw.get("metadata", raw.get("meta", {k: v for k, v in raw.items()
                    if isinstance(v, (str, int, float, bool)) or v is None})),
                "n": n, "elements": len(dofs), "p": p, "basis": basis,
                "interior_dofs": 0,
                "trace_dofs": n, "eliminated_fraction": 0.0,
                "exact_condensed": None, "element_block_full_cg": None,
                "element_block_schur_cg": None,
                "sparse_p1_face_blocks": sparse_p1_face_screen(raw, a, rhs, dofs, nodes,
                    masses, tolerance, max_iterations, previous_path),
                "volume_plus_scaled_faces": volume_face_attenuation(raw, a, rhs, dofs, masses),
                "host_screen_seconds": time.perf_counter() - started,
                "claim": "p1 has no bubble unknowns; exact bubble condensation eliminates nothing"}
    if n > 1600 or p < 1 or any(len(e) != (p + 1) ** 2 for e in dofs):
        raise ValueError("Bounded probe supports <=1600 DOFs on uniform tensor quads")
    if any(x.shape != ((p + 1) ** 2, 2) for x in nodes):
        raise ValueError("Need the reference (x,y) coordinates of every local GL node")
    symmetry = ratio(norm(a - a.T), norm(a))
    if symmetry > 1e-11:
        raise ValueError("This probe's energy and CG claims require a symmetric operator")
    a_factor = linalg.cho_factor(a, lower=True, check_finite=True)
    reference = linalg.cho_solve(a_factor, rhs)
    bnorm = norm(rhs)

    def physical_norm(x):
        value = sum(float(x[ids] @ mass @ x[ids]) for ids, mass in zip(dofs, masses))
        return float(np.sqrt(max(0.0, value)))

    def metrics(x):
        err = x - reference
        return {"original_residual_relative": ratio(norm(a @ x - rhs), bnorm),
                "physical_l2_error_relative": ratio(physical_norm(err),
                                                     physical_norm(reference)),
                "energy_error_relative": ratio(np.sqrt(max(0.0, err @ a @ err)),
                                                np.sqrt(max(0.0, reference @ rhs)))}

    # The export uses [0,1]^2 reference coordinates, independent of MFEM's
    # element DOF ordering. Columns use an explicitly defined x-fast LGL basis.
    lgl = np.concatenate(([0.0], (Legendre.basis(p).deriv().roots() + 1) / 2, [1.0]))
    inside_1d = range(1, p) if basis == "lgl" else range(4, p + 1)
    local_inside = np.array([j * (p + 1) + i for j in inside_1d
                             for i in inside_1d], dtype=int)
    local_boundary = np.setdiff1d(np.arange((p + 1) ** 2), local_inside)
    transform = np.zeros_like(a)
    interior_groups, boundary_groups = [], []
    basis_condition = []
    for ids, points in zip(dofs, nodes):
        if basis == "lgl":
            lx = lagrange_values(lgl, points[:, 0])
            ly = lagrange_values(lgl, points[:, 1])
        else:
            lx = hermite_jet_values(p, points[:, 0])
            ly = hermite_jet_values(p, points[:, 1])
        t = np.einsum("aj,ai->aji", ly, lx).reshape(len(ids), len(ids))
        transform[np.ix_(ids, ids)] = t
        interior_groups.append(ids[local_inside])
        boundary_groups.append(ids[local_boundary])
        basis_condition.append(float(np.linalg.cond(t)))
    transformed = transform.T @ a @ transform
    transformed_rhs = transform.T @ rhs
    exterior_interior_square = 0.0
    for group, ids in zip(interior_groups, dofs):
        other = np.setdiff1d(np.arange(n), ids)
        exterior_interior_square += norm(transformed[np.ix_(group, other)]) ** 2
    interior = np.concatenate(interior_groups)
    boundary = np.concatenate(boundary_groups)
    ai = transformed[np.ix_(interior, interior)]
    aib = transformed[np.ix_(interior, boundary)]
    abi = transformed[np.ix_(boundary, interior)]
    abb = transformed[np.ix_(boundary, boundary)]
    ai_block = np.zeros_like(ai)
    local_sizes = [len(group) for group in interior_groups]
    offset = 0
    x = np.zeros_like(aib)
    y = np.zeros(len(interior))
    factor_flops, solve_flops = 0.0, 0.0
    for size in local_sizes:
        sl = slice(offset, offset + size)
        ai_block[sl, sl] = ai[sl, sl]
        if size:
            factor = linalg.cho_factor(ai[sl, sl], lower=True)
            x[sl] = linalg.cho_solve(factor, aib[sl])
            y[sl] = linalg.cho_solve(factor, transformed_rhs[interior[sl]])
            factor_flops += size ** 3 / 3
            # Charge the deliberately dense screen, including zero columns.
            solve_flops += 2 * size ** 2 * (len(boundary) + 1)
        offset += size
    off_diagonal = ratio(norm(ai - ai_block), norm(transformed))
    if off_diagonal > 1e-11:
        raise ValueError("Cross-element bubbles couple: element-local elimination invalid")
    schur = abb - abi @ x
    schur_rhs = transformed_rhs[boundary] - abi @ y
    schur_factor = linalg.cho_factor(schur, lower=True)
    trace_reference = linalg.cho_solve(schur_factor, schur_rhs)

    def reconstruct(trace):
        coefficients = np.zeros(n)
        coefficients[boundary] = trace
        coefficients[interior] = y - x @ trace
        return transform @ coefficients

    inverse_boundary = {int(d): i for i, d in enumerate(boundary)}
    trace_groups = [np.array([inverse_boundary[int(d)] for d in group])
                    for group in boundary_groups]

    def block_setup(matrix, groups):
        diagonal = np.zeros_like(matrix)
        factors = []
        for group in groups:
            block = matrix[np.ix_(group, group)]
            diagonal[np.ix_(group, group)] = block
            factors.append(linalg.cho_factor(block, lower=True))

        def apply(vector):
            out = np.zeros_like(vector)
            for group, factor in zip(groups, factors):
                out[group] = linalg.cho_solve(factor, vector[group])
            return out

        eigenvalues = linalg.eigvalsh(matrix, diagonal)
        lo, hi = float(eigenvalues[0]), float(eigenvalues[-1])
        return diagonal, apply, {"lambda_min": lo, "lambda_max": hi,
                                 "condition": hi / lo,
                                 "optimal_richardson_weight": 2 / (lo + hi),
                                 "optimal_richardson_contraction": (hi - lo) / (hi + lo)}

    full_diagonal, full_preconditioner, full_spectrum = block_setup(a, dofs)
    diagonal, preconditioner, spectrum = block_setup(schur, trace_groups)
    linear_operator = LinearOperator(schur.shape, matvec=preconditioner)
    iteration = [0]
    accepted_trace = [None]

    class ReachedOriginalResidual(Exception):
        pass

    def count(_):
        iteration[0] += 1

    def count_trace(value):
        iteration[0] += 1
        if norm(a @ reconstruct(value) - rhs) <= tolerance * bnorm:
            accepted_trace[0] = value.copy()
            raise ReachedOriginalResidual

    try:
        # The Schur Euclidean norm is basis-dependent. Stop at the same original
        # GL current-operator residual as full CG, not at a different norm.
        trace, status = cg(schur, schur_rhs, M=linear_operator, tol=tolerance * 1e-5,
                           atol=0.0, maxiter=max_iterations, callback=count_trace)
    except ReachedOriginalResidual:
        trace, status = accepted_trace[0], 0
    cg_result = {"iterations": iteration[0], "status": int(status),
                 **metrics(reconstruct(trace))}
    iteration[0] = 0
    full_result, status = cg(a, rhs, M=LinearOperator(a.shape, matvec=full_preconditioner),
                            tol=tolerance, atol=0.0, maxiter=max_iterations, callback=count)
    full_cg_result = {"iterations": iteration[0], "status": int(status),
                      **metrics(full_result)}

    # Alternative face coupling: group duplicate geometric LGL trace locations
    # across elements. Two-sided face nodes and multi-element vertices become
    # tiny local blocks. Every coefficient belongs to exactly one group.
    face_node_result = None
    if basis == "lgl" and "dof_coordinates" in raw:
        coordinates = np.asarray(raw["dof_coordinates"])
        new_coordinates = np.zeros_like(coordinates)
        for ids in dofs:
            new_coordinates[ids] = linalg.solve(transform[np.ix_(ids, ids)], coordinates[ids])
        trace_coordinates = new_coordinates[boundary]
        coordinate_tolerance = 1e-10 * max(1.0, float(np.ptp(coordinates, axis=0).max()))
        parents = np.arange(len(boundary))

        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        for i, j in cKDTree(trace_coordinates).query_pairs(coordinate_tolerance):
            parents[root(j)] = root(i)
        grouped = {}
        for i in range(len(boundary)):
            grouped.setdefault(root(i), []).append(i)
        face_node_groups = [np.asarray(value, dtype=int) for value in grouped.values()]
        node_diagonal, node_preconditioner, node_spectrum = block_setup(schur, face_node_groups)
        iteration[0], accepted_trace[0] = 0, None
        try:
            node_trace, node_status = cg(schur, schur_rhs,
                M=LinearOperator(schur.shape, matvec=node_preconditioner),
                tol=tolerance * 1e-5, atol=0.0, maxiter=max_iterations, callback=count_trace)
        except ReachedOriginalResidual:
            node_trace, node_status = accepted_trace[0], 0
        face_node_result = {"block_count": len(face_node_groups),
            "block_max_dofs": max(map(len, face_node_groups)),
            "factor_entries": sum(len(group) ** 2 for group in face_node_groups),
            "coordinate_pair_tolerance": coordinate_tolerance,
            "coordinates_are_polynomial_interpolants": True,
            "spectrum": node_spectrum, "cg_iterations": iteration[0],
            "cg_status": int(node_status), **metrics(reconstruct(node_trace))}
        iteration[0], accepted_trace[0] = 0, None
        try:
            mixed_trace, mixed_status = cg(schur, schur_rhs,
                M=LinearOperator(schur.shape, matvec=lambda r: preconditioner(r) + node_preconditioner(r)),
                tol=tolerance * 1e-5, atol=0.0, maxiter=max_iterations, callback=count_trace)
        except ReachedOriginalResidual:
            mixed_trace, mixed_status = accepted_trace[0], 0
        face_node_result["additive_element_plus_node"] = {"iterations": iteration[0],
            "status": int(mixed_status), **metrics(reconstruct(mixed_trace))}
        lo, hi = node_spectrum["lambda_min"], node_spectrum["lambda_max"]
        midpoint, halfwidth = (lo + hi) / 2, (hi - lo) / 2
        trace, previous = np.zeros(len(boundary)), np.zeros(len(boundary))
        rho = halfwidth / midpoint
        chebyshev, first_hit = [], None
        for degree in range(1, 257):
            residual = node_preconditioner(schur_rhs - schur @ trace)
            if degree == 1:
                current = trace + residual / midpoint
            else:
                rho_new = 1 / (2 * midpoint / halfwidth - rho)
                current = trace + 2 * rho_new / halfwidth * residual + rho_new * rho * (trace - previous)
                rho = rho_new
            previous, trace = trace, current
            values = metrics(reconstruct(trace))
            if first_hit is None and values["original_residual_relative"] <= tolerance:
                first_hit = degree
            if degree in (1, 2, 4, 8, 16, 32, 64, 128, 256) or degree == first_hit:
                chebyshev.append({"degree": degree, **values})
        face_node_result["chebyshev"] = {"first_original_residual_tolerance_hit": first_hit,
            "exact_spectrum_used": True,
            "analysis_checks_every_step": True, "samples": chebyshev}

    # One trace sweep propagates only through the explicit Schur graph. Exact
    # local interiors and all residual/recovery work are included in this screen.
    sweeps, trace = [], np.zeros(len(boundary))
    omega = spectrum["optimal_richardson_weight"]
    for k in range(1, 33):
        trace += omega * preconditioner(schur_rhs - schur @ trace)
        if k in (1, 2, 4, 8, 16, 32):
            sweeps.append({"sweeps": k, **metrics(reconstruct(trace))})

    # Joint route-2/route-3 screen: eliminate bubbles first, then use a local
    # damped-wave (heavy-ball) interface recurrence with exact spectral bounds.
    lo, hi = spectrum["lambda_min"], spectrum["lambda_max"]
    contraction = (np.sqrt(hi) - np.sqrt(lo)) / (np.sqrt(hi) + np.sqrt(lo))
    step, momentum = 4 / (np.sqrt(hi) + np.sqrt(lo)) ** 2, contraction ** 2
    heavy_ball, trace, previous = [], np.zeros(len(boundary)), np.zeros(len(boundary))
    heavy_ball_first_hit = None
    for k in range(1, 257):
        current = trace + momentum * (trace - previous) + step * preconditioner(
            schur_rhs - schur @ trace)
        previous, trace = trace, current
        if k in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            values = metrics(reconstruct(trace))
            heavy_ball.append({"sweeps": k, **values})
            if (heavy_ball_first_hit is None and
                    values["original_residual_relative"] <= tolerance):
                heavy_ball_first_hit = k

    # Deliberately changed discrete models: convexly attenuate off-element
    # Schur coupling. SPD is retained for eta in [0,1], consistency is not.
    attenuated = []
    for eta in (0.0, 0.25, 0.5, 0.75, 0.99, 0.999, 1.0):
        matrix = (1 - eta) * diagonal + eta * schur
        value = linalg.solve(matrix, schur_rhs, assume_a="pos")
        attenuated.append({"eta": eta, **metrics(reconstruct(value))})

    # Condensation can create a two-hop trace graph. Count block adjacency using
    # a relative Frobenius threshold, with diagnostic threshold sensitivity.
    def block_graph(matrix, groups):
        result = np.zeros((len(groups), len(groups)), dtype=bool)
        matrix_scale = norm(matrix)
        for i, row in enumerate(groups):
            for j, column in enumerate(groups):
                result[i, j] = norm(matrix[np.ix_(row, column)]) > 1e-12 * matrix_scale
        return result

    full_graph = block_graph(a, dofs)
    trace_graph = block_graph(schur, trace_groups)
    useful_a_entries = sum(len(r) * len(c) for i, r in enumerate(dofs)
                           for j, c in enumerate(dofs) if full_graph[i, j])
    useful_s_entries = sum(len(r) * len(c) for i, r in enumerate(trace_groups)
                           for j, c in enumerate(trace_groups) if trace_graph[i, j])
    interior_trace_nonzero_columns = sum(int(np.count_nonzero(np.max(np.abs(aib[
        sum(local_sizes[:e]):sum(local_sizes[:e + 1])]), axis=0) > 1e-12 * norm(aib)))
        for e, size in enumerate(local_sizes) if size)
    identity_probe = np.sin(np.arange(len(boundary)) * 0.17)
    lifted = np.zeros(n)
    lifted[boundary] = identity_probe
    lifted[interior] = -x @ identity_probe
    identity_relative_error = ratio(abs(lifted @ transformed @ lifted -
                                        identity_probe @ schur @ identity_probe),
                                    abs(identity_probe @ schur @ identity_probe))
    return {"schema": "igr-face-condensation-v1", "input": str(Path(path).resolve()),
            "input_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "metadata": raw.get("metadata", raw.get("meta", {k: v for k, v in raw.items()
                if isinstance(v, (str, int, float, bool)) or v is None})),
            "n": n, "elements": len(dofs),
            "p": p, "basis": basis, "interior_dofs": len(interior),
            "trace_dofs": len(boundary),
            "eliminated_fraction": len(interior) / n,
            "basis_condition_max": max(basis_condition), "symmetry_relative": symmetry,
            "cross_element_bubble_block_relative": off_diagonal,
            "cross_element_interior_to_any_relative":
                np.sqrt(exterior_interior_square) / norm(transformed),
            "schur_energy_identity_relative": identity_relative_error,
            "exact_condensed": metrics(reconstruct(trace_reference)),
            "element_block_full_spectrum": full_spectrum,
            "element_block_schur_spectrum": spectrum,
            "element_block_full_cg": full_cg_result, "element_block_schur_cg": cg_result,
            "matched_face_node_block_schur": face_node_result,
            "optimal_damping_uses_exact_spectrum": True,
            "original_residual_target": tolerance,
            "fixed_trace_sweeps": sweeps,
            "interface_damped_wave": {"step": float(step), "momentum": float(momentum),
                "asymptotic_spectral_contraction": float(contraction),
                "first_sampled_tolerance_hit": heavy_ball_first_hit, "samples": heavy_ball},
            "changed_discrete_coupling": attenuated,
            "volume_plus_scaled_faces": volume_face_attenuation(raw, a, rhs, dofs, masses),
            "work": {"full_block_graph_edges_directed_including_diagonal": int(full_graph.sum()),
                     "trace_block_graph_edges_directed_including_diagonal": int(trace_graph.sum()),
                     "full_block_stored_entries": useful_a_entries,
                     "explicit_schur_block_stored_entries": useful_s_entries,
                     "explicit_schur_entry_ratio": useful_s_entries / useful_a_entries,
                     "bubble_factor_flops_estimate": factor_flops,
                     "dense_screen_bubble_solve_flops_estimate": solve_flops,
                     "interior_trace_nonzero_columns_summed": interior_trace_nonzero_columns,
                     "dense_screen_a_and_transform_and_schur_bytes":
                         int(a.nbytes + transform.nbytes + schur.nbytes)},
            "host_screen_seconds": time.perf_counter() - started,
            "claim": "frozen production operator algebra only; no GPU or horizon speedup"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--basis", choices=("lgl", "hermite-jet", "both"), default="both")
    parser.add_argument("--previous-export", type=Path,
                        help="Exact prior closure as an initial guess for the sparse p1 screen")
    args = parser.parse_args()
    results = []
    for path in args.inputs:
        bases = [args.basis] if args.basis != "both" else ["lgl", "hermite-jet"]
        for basis in bases:
            if basis == "hermite-jet":
                raw = json.loads(path.read_text())
                element = raw["element_dofs"][0] if "element_dofs" in raw else raw["elements"][0]["dofs"]
                if len(element) < 16:
                    continue
            results.append(analyze(path, args.tolerance, args.max_iterations, basis, args.previous_export))
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    for result in results:
        print(json.dumps({k: result[k] for k in ("input", "n", "p", "basis", "trace_dofs",
            "exact_condensed", "element_block_full_cg", "element_block_schur_cg")},
            allow_nan=False))


if __name__ == "__main__":
    main()
