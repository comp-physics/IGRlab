#!/usr/bin/env python3
"""Bounded p-scaled-alpha / face-conditioned polynomial screen.

Uses current production A and b, never synthesizes an IGR source. Dense
condensation and spectral audits are explicitly charged research oracles.
Local matrix bounds need no global eigenproblem, but their matrix construction
is not free. No time-to-solution or tensor-throughput claim follows from this.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy import linalg as la
from scipy.spatial import cKDTree

from igr_face_condensation_probe import lagrange_values
from igr_relaxation_probe import load, norm


def block_factor(matrix, groups):
    """B and W with W B W.T=I. Every factor is local to one listed group."""
    started = time.perf_counter()
    b, w = np.zeros_like(matrix), np.zeros_like(matrix)
    for group in groups:
        block = matrix[np.ix_(group, group)]
        lower = la.cholesky(block, lower=True)
        b[np.ix_(group, group)] = block
        w[np.ix_(group, group)] = la.solve_triangular(lower, np.eye(len(group)), lower=True)
    return b, w, {"host_factor_seconds": time.perf_counter() - started,
                  "factor_entries": sum(len(g) ** 2 for g in groups),
                  "factor_flops_estimate": sum(len(g) ** 3 / 3 for g in groups),
                  "factor_apply_products": sum(len(g) * (len(g) + 1) for g in groups),
                  "max_block_size": max(map(len, groups))}


def condense(data):
    started = time.perf_counter()
    a, b = data["A"].toarray(), data["b"]
    n, dofs, raw = len(b), data["dofs"], data["raw"]
    p = int(raw["order"])
    n1 = p + 1
    points = np.asarray(raw["reference_nodes"])
    lgl = np.r_[0., (Legendre.basis(p).deriv().roots() + 1) / 2, 1.]
    local_i = np.asarray([j*n1+i for j in range(1, p) for i in range(1, p)], int)
    local_t = np.setdiff1d(np.arange(n1*n1), local_i)
    transform = np.zeros_like(a)
    internal, trace, interior_groups = [], [], []
    physical_face_groups = {}
    coords = np.asarray(raw["dof_coordinates"])
    trace_coords = []
    for element, ids in enumerate(dofs):
        lx, ly = lagrange_values(lgl, points[:, 0]), lagrange_values(lgl, points[:, 1])
        local = np.einsum("aj,ai->aji", ly, lx).reshape(len(ids), len(ids))
        transform[np.ix_(ids, ids)] = local
        internal.extend(ids[local_i]); trace.extend(ids[local_t])
        interior_groups.append(ids[local_i])
        trace_coords.extend(la.solve(local, coords[ids])[local_t])
        vertices = raw.get("element_vertices", [])[element]
        for offset, index in enumerate(local_t):
            j, i = divmod(int(index), n1)
            if i in (0, p) and j in (0, p):
                corner = {(0, 0): 0, (p, 0): 1, (p, p): 2, (0, p): 3}[(i, j)]
                key = ("vertex", int(vertices[corner]))
            else:
                edge = (0, 1) if j == 0 else (3, 2) if j == p else (0, 3) if i == 0 else (1, 2)
                key = ("edge", *sorted(int(vertices[k]) for k in edge))
            physical_face_groups.setdefault(key, []).append(element*len(local_t)+offset)
    internal, trace = np.asarray(internal, int), np.asarray(trace, int)
    at, rt = transform.T @ a @ transform, transform.T @ data["reaction"] @ transform
    bt = transform.T @ b
    ai, ait = at[np.ix_(internal, internal)], at[np.ix_(internal, trace)]
    offset, block_error, x, y = 0, np.array(ai, copy=True), np.zeros_like(ait), np.zeros(len(internal))
    for group in interior_groups:
        size = len(group)
        sl = slice(offset, offset+size)
        if size:
            factor = la.cho_factor(ai[sl, sl], lower=True)
            x[sl] = la.cho_solve(factor, ait[sl])
            y[sl] = la.cho_solve(factor, bt[internal[sl]])
        block_error[sl, sl] = 0
        offset += size
    bubble_coupling = norm(block_error) / norm(a)
    if bubble_coupling > 1e-11:
        raise ValueError("Interior elimination is not element local")
    s = at[np.ix_(trace, trace)] - ait.T @ x
    g = bt[trace] - ait.T @ y
    ri = rt[np.ix_(internal, internal)]
    rit = rt[np.ix_(internal, trace)]
    sr = rt[np.ix_(trace, trace)] - (rit.T @ la.solve(ri, rit, assume_a="pos") if len(internal) else 0)
    trace_groups = [np.arange(e*len(local_t), (e+1)*len(local_t)) for e in range(len(dofs))]
    parents = np.arange(len(trace))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, j in cKDTree(np.asarray(trace_coords)).query_pairs(1e-10*max(1., np.ptp(coords, axis=0).max())):
        parents[root(j)] = root(i)
    groups = {}
    for i in range(len(trace)):
        groups.setdefault(root(i), []).append(i)
    face_groups = [np.asarray(g, int) for g in groups.values()]

    def reconstruct(value):
        coefficients = np.zeros(n)
        coefficients[trace] = value
        coefficients[internal] = y - x @ value
        return transform @ coefficients

    # Reaction condensation is also local. S(A)>=S(R) follows from A>=R.
    return {"S": s, "g": g, "SR": sr, "reconstruct": reconstruct,
            "trace_groups": trace_groups, "face_groups": face_groups,
            "physical_face_groups": [np.asarray(v, int) for v in physical_face_groups.values()],
            "host_condensation_seconds": time.perf_counter()-started,
            "trace_dofs": len(trace), "interior_dofs": len(internal),
            "cross_element_bubble_relative": bubble_coupling,
            "dense_assembly_bytes": a.nbytes + at.nbytes + transform.nbytes + s.nbytes,
            "basis_transform_products_per_action": len(dofs) * (p+1)**4,
            "dense_elimination_products_estimate": len(internal)*len(trace)**2,
            "interior_factor_entries": sum(len(i)**2 for i in interior_groups)}


def local_bounds(h, b, sr, element_groups):
    """Finite-neighborhood certificates; no Ritz/eigen estimate of global H."""
    started = time.perf_counter()
    # Upper bound of R_S^-1/2 B R_S^-1/2 gives a positive lower bound on
    # B^-1/2 S B^-1/2, conditional on the separately audited A>=R.
    _, wr, _ = block_factor(sr, element_groups)
    contrast = wr @ b @ wr.T
    lower_reaction = 1 / float(np.max(np.sum(abs(contrast), axis=1)))
    upper = float(np.max(np.sum(abs(h), axis=1)))
    result = [{"name": "local-reaction-row", "lower": lower_reaction, "upper": upper,
               "power": 1, "matrix_products": 0,
               "assumption": "A-R positive semidefinite; Schur monotonicity"}]
    # Symmetry makes ||(H-cI)^k||_infinity^(1/k) a two-sided spectral radius
    # certificate. Matrix powers extend neighborhoods and may become dense.
    hpower = np.array(h, copy=True)
    base_graph = abs(h) > 1e-13*max(1., abs(h).max())
    graph = np.array(base_graph, copy=True)
    for degree in (1, 2, 4, 8):
        if degree > 1:
            hpower = hpower @ hpower
            graph = (graph.astype(np.int64) @ graph.astype(np.int64)) > 0
        # First tighten the upper endpoint, so the centered radius need not
        # waste resolution compensating for an overly large one-row bound.
        current_upper = min(upper, float(np.max(np.sum(abs(hpower), axis=1))) ** (1/degree))
        center = current_upper/2
        power = h-center*np.eye(len(h))
        for _ in range(int(math.log2(degree))):
            power = power @ power
        radius = float(np.max(np.sum(abs(power), axis=1))) ** (1/degree)
        result.append({"name": f"local-centered-power-{degree}",
                       "lower": max(lower_reaction, center-radius),
                       "upper": min(current_upper, center+radius), "power": degree,
                       "matrix_products": 2*int(math.log2(degree)),
                       "structural_entries": int(graph.sum()),
                       "dense_setup_multiply_products": 2*int(math.log2(degree))*len(h)**3,
                       "assumption": "symmetry; reaction lower additionally assumes A-R PSD"})
    return result, time.perf_counter()-started


def solve_metrics(data, x, reference):
    error = x-reference
    energy = float(error @ (data["A"] @ error))
    mass = float(error @ (data["mass"] @ error))
    return {"current_residual_relative": norm(data["A"] @ x-data["b"])/norm(data["b"]),
            "energy_error_relative": math.sqrt(max(0., energy)/float(reference @ data["b"])),
            "physical_l2_error_relative": math.sqrt(max(0., mass)/float(reference @ (data["mass"] @ reference)))}


def cheb_path(h, f, w, lower, upper, cap, reconstruct, metrics, tolerance):
    center, radius = (upper+lower)/2, (upper-lower)/2
    rho = radius/center
    y, correction = np.zeros_like(f), np.zeros_like(f)
    first_residual, first_energy = None, None
    samples = []
    started = time.perf_counter()
    for step in range(1, cap+1):
        residual = f-h@y
        if step == 1 or radius <= 1e-14*center:
            correction = residual/center
        else:
            following = 1/(2*center/radius-rho)
            correction = following*rho*correction + 2*following/radius*residual
            rho = following
        y += correction
        # These checks deliberately inspect every step to find first passage;
        # a proposed endpoint-only implementation would omit them.
        row = metrics(reconstruct(w.T@y))
        if first_residual is None and row["current_residual_relative"] <= tolerance:
            first_residual = step
        if first_energy is None and row["energy_error_relative"] <= 1e-3:
            first_energy = step
        if step in (1, 2, 4, 8, 16, 32, 64, 128, 256, cap) or step in (first_residual, first_energy):
            samples.append({"stages": step, **row})
        if first_residual is not None:
            break
    return {"first_residual_target": first_residual, "first_energy_1e-3": first_energy,
            "test_only_original_operator_checks": step,
            "host_recurrence_and_diagnostics_seconds": time.perf_counter()-started,
            "samples": samples}


def cg_path(h, f, w, cap, reconstruct, metrics, tolerance):
    y, r = np.zeros_like(f), f.copy()
    d, rr = r.copy(), float(r@r)
    for k in range(1, cap+1):
        hd = h@d
        step = rr/float(d@hd)
        y += step*d
        r -= step*hd
        new_rr = float(r@r)
        row = metrics(reconstruct(w.T@y))
        if row["current_residual_relative"] <= tolerance:
            break
        d = r+(new_rr/rr)*d
        rr = new_rr
    return {"iterations": k, "complete": row["current_residual_relative"] <= tolerance,
            "recurrence_scalar_reductions": 1+2*k,
            "test_only_original_operator_checks": k, **row}


def composed_screen(h, w, lower, upper, s_entries):
    """Exact algebraic precomposition of a fixed Chebyshev inverse polynomial."""
    n = len(h)
    center, radius = (upper+lower)/2, (upper-lower)/2
    rho = radius/center
    q, correction = np.zeros_like(h), np.zeros_like(h)
    started = time.perf_counter()
    rows = []
    for degree in range(1, 9):
        residual = np.eye(n)-h@q
        if degree == 1 or radius <= 1e-14*center:
            correction = residual/center
        else:
            following = 1/(2*center/radius-rho)
            correction = following*rho*correction + 2*following/radius*residual
            rho = following
        q += correction
        if degree not in (2, 4, 8):
            continue
        composed = w.T @ q @ w
        threshold = 1e-12*max(1., abs(composed).max())
        significant = int(np.count_nonzero(abs(composed)>threshold))
        # No dropping is performed; sparse support count is descriptive only.
        rows.append({"stages": degree, "stored_dense_entries": n*n,
                     "significant_entries_at_relative_1e-12": significant,
                     "no_entries_dropped": True,
                     "recurrence_operator_products_lower_estimate": degree*s_entries,
                     "precomposed_apply_products": n*n,
                     "dense_setup_products_lower_estimate": degree*n**3,
                     "host_cumulative_setup_seconds": time.perf_counter()-started,
                     "inverse_symmetry_relative": norm(composed-composed.T)/norm(composed)})
    return rows


def analyze(path, tolerance, cap, compose):
    started = time.perf_counter()
    data = load(path)
    n = len(data["b"])
    if n > 1600:
        raise ValueError("Dense oracle screen capped at 1600 DOFs")
    a, reaction = data["A"].toarray(), data["reaction"].toarray()
    reference_started = time.perf_counter()
    reference = la.solve(a, data["b"], assume_a="pos")
    reference_seconds = time.perf_counter()-reference_started
    metric = lambda x: solve_metrics(data, x, reference)
    ref_metrics = metric(reference)
    if ref_metrics["current_residual_relative"] > 1e-9:
        raise ValueError("Numerical reference does not meet residual gate")
    audit_started = time.perf_counter()
    diffusion_min = float(la.eigvalsh(a-reaction, reaction, subset_by_index=[0, 0])[0])
    if diffusion_min < -1e-8:
        raise ValueError("A>=R certificate fails")
    audit_seconds = time.perf_counter()-audit_started
    c = condense(data)
    rows = []
    for label, groups, source in (("condensed-face-node", c["face_groups"], c["S"]),
                                   ("condensed-full-face", c["physical_face_groups"], c["S"]),
                                   ("condensed-element-trace", c["trace_groups"], c["S"]),
                                   ("condensed-reaction", c["trace_groups"], c["SR"])):
        b, w, costs = block_factor(source, groups)
        h, f = w@c["S"]@w.T, w@c["g"]
        eigen_started = time.perf_counter()
        eigen = la.eigvalsh(h)
        eigen_seconds = time.perf_counter()-eigen_started
        lo, hi = float(eigen[0]), float(eigen[-1])
        bounds, bound_seconds = local_bounds(h, b, c["SR"], c["trace_groups"])
        for bound in bounds:
            if not (bound["lower"] <= lo*(1+1e-8) and bound["upper"] >= hi*(1-1e-8)):
                raise ValueError("Local interval failed independent spectral audit")
            q = (math.sqrt(bound["upper"]/bound["lower"])-1)/(math.sqrt(bound["upper"]/bound["lower"])+1)
            bound["guaranteed_energy_1e-3_stages"] = math.ceil(math.log(5e-4)/math.log(q)) if q else 1
        best = min(bounds, key=lambda x: x["upper"]/x["lower"])
        runs = []
        for label_bounds, low, high in (("oracle", lo*(1-1e-12), hi*(1+1e-12)),
                                      (best["name"], best["lower"]*(1-1e-12), best["upper"]*(1+1e-12))):
            runs.append({"bounds": label_bounds, "lower": low, "upper": high,
                         **cheb_path(h, f, w, low, high, cap, c["reconstruct"], metric, tolerance)})
        row = {"metric": label, **costs, "oracle_lower": lo, "oracle_upper": hi,
               "oracle_condition": hi/lo, "oracle_eigen_seconds": eigen_seconds,
               "local_bounds": bounds, "local_bound_setup_seconds": bound_seconds,
               "cg": cg_path(h, f, w, cap, c["reconstruct"], metric, tolerance),
               "chebyshev": runs}
        if compose and label == "condensed-face-node":
            row["composed_fixed_polynomials"] = composed_screen(h, w, lo, hi,
                int(np.count_nonzero(abs(c["S"]) > 1e-12*abs(c["S"]).max())))
        rows.append(row)
    local = np.zeros(n)
    for ids, block in zip(data["dofs"], data["raw"]["volume_blocks"]):
        local[ids] = la.solve(np.asarray(block), data["b"][ids], assume_a="pos")
    condensation = {k: v for k, v in c.items() if not callable(v) and
                    not isinstance(v, (np.ndarray, list))}
    raw = data["raw"]
    return {"schema": "igr-hp-face-screen-v1", "input": str(Path(path).resolve()),
            "input_sha256": data["sha256"], "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "metadata": {k: v for k, v in raw.items() if isinstance(v, (str, int, float, bool))},
            "reference_seconds": reference_seconds, "reference": ref_metrics,
            "diffusion_min_reaction_eigenvalue": diffusion_min, "audit_seconds": audit_seconds,
            "independent_element_neumann": metric(local), "condensation": condensation,
            "metrics": rows, "host_total_seconds": time.perf_counter()-started,
            "scope": "Frozen current production closure; CPU work screen, no GPU or advancing accuracy claim"}


def selftest():
    # Independent random SPD form with A>=R checks the local bound algebra.
    rng = np.random.default_rng(43)
    n = 12
    q = rng.normal(size=(n, n))
    r = np.diag(np.linspace(.3, 2., n))
    a = r+q.T@q
    groups = [np.arange(i, i+3) for i in range(0, n, 3)]
    b, w, _ = block_factor(a, groups)
    h = w@a@w.T
    bounds, _ = local_bounds(h, b, r, [np.array([i]) for i in range(n)])
    spectrum = la.eigvalsh(h)
    for row in bounds:
        assert row["lower"] <= spectrum[0]*(1+1e-12)
        assert row["upper"] >= spectrum[-1]*(1-1e-12)
    assert norm(w@b@w.T-np.eye(n)) < 1e-12
    print(json.dumps({"selftest": "pass", "independent_random_spd_bounds": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-stages", type=int, default=256)
    parser.add_argument("--compose", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return selftest()
    if not args.inputs or args.output is None:
        parser.error("inputs and --output required")
    if not 0 < args.tolerance < 1 or not 1 <= args.max_stages <= 1024:
        parser.error("Invalid bounded controls")
    results = []
    for path in args.inputs:
        row = analyze(path, args.tolerance, args.max_stages, args.compose)
        results.append(row)
        print(json.dumps({"input": str(path), "metrics": [{"metric": r["metric"],
              "cg": r["cg"]["iterations"], "cheb": r["chebyshev"][0]["first_residual_target"],
              "condition": r["oracle_condition"]} for r in row["metrics"]]}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"command": [sys.executable]+sys.argv, "results": results},
                                        indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
