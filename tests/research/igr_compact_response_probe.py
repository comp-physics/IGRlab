#!/usr/bin/env python3
"""Bounded CPU screen of finite-range responses, never a production IGR solver.

Consumes the frozen production-operator JSON from igr_screened_locality_probe.
The exact inverse is an explicitly charged, small-matrix oracle for tapering;
the symmetric patch maps use only local principal matrices. Applying any map
once changes the discrete closure. Iterating to a current-operator tolerance
would instead retain the target closure, but is not timed by this probe.

OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_compact_response_probe.py \
    --input /tmp/frozen.json --output /tmp/compact.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy import integrate, linalg, sparse
from scipy.sparse import linalg as spla


def ratio(a, b):
    return float(a / b) if b > 0 else (0.0 if a == 0 else None)


def wendland(r):
    r = np.asarray(r)
    return np.maximum(1.0 - r, 0.0) ** 4 * (1.0 + 4.0 * r)


def continuum_screen():
    """Constant-density, one-dimensional response; length ell=1."""
    radius = math.sqrt(28.0)
    second = 3 * radius**2 * integrate.quad(lambda s: s**2 * wendland(s), 0, 1)[0]
    fourth = 3 * radius**4 * integrate.quad(lambda s: s**4 * wendland(s), 0, 1)[0]
    transfers = []
    for k in (0, .1, .25, .5, 1, 2, 4, 8):
        compact = 3 * integrate.quad(
            lambda s: wendland(s) * math.cos(k * radius * s), 0, 1,
            epsabs=1e-13, epsrel=1e-13)[0]
        exact = 1 / (1 + k * k)
        transfers.append({"k_ell": k, "elliptic_transfer": exact,
                          "compact_transfer": compact,
                          "relative_transfer_error": abs(compact - exact) / exact})
    # Gaussian load: compare Fourier responses using Parseval, not time advancement.
    loads = []
    for width in (.25, 1, 4):
        k = np.linspace(0, 32 / width, 32769)
        nodes, weights = np.polynomial.legendre.leggauss(96)
        s, weights = (nodes + 1) / 2, weights / 2
        # Chunk to avoid a large quadrature-by-frequency temporary.
        compact = np.concatenate([
            np.cos(radius * block[:, None] * s) @ (3 * weights * wendland(s))
            for block in np.array_split(k, 32)])
        exact = 1 / (1 + k * k)
        source_squared = np.exp(-(width * k)**2)
        error = np.trapz(source_squared * (compact - exact)**2, k)
        scale = np.trapz(source_squared * exact**2, k)
        loads.append({"gaussian_width_over_ell": width,
                      "response_relative_l2": math.sqrt(error / scale)})
    exponential_windows = []
    for length in (2, 4, 6, 8, 12):
        q = math.exp(-length)
        uniform_bound = 4 * q / (1 - q)**2
        transfers_window = []
        for k in (0, .1, .25, .5, 1, 2, 4, 8):
            relative = 2 * q * (1 - math.cos(k * length)) / (1 - q)**2
            transfers_window.append({"k_ell": k, "relative_transfer_error": relative})
        gaussian = []
        for width in (.25, 1, 4):
            numerator = integrate.quad(
                lambda k: math.exp(-(width * k)**2) / (1 + k*k)**2 *
                (2 * q * (1 - math.cos(k * length)) / (1 - q)**2)**2,
                0, 16 / width, epsabs=1e-15, epsrel=1e-11, limit=300)[0]
            denominator = integrate.quad(
                lambda k: math.exp(-(width * k)**2) / (1 + k*k)**2,
                0, 16 / width, epsabs=1e-15, epsrel=1e-11)[0]
            gaussian.append({"gaussian_width_over_ell": width,
                             "response_relative_l2": math.sqrt(numerator / denominator)})
        exponential_windows.append({"support_radius_over_ell": length,
                                    "uniform_relative_fourier_error_bound": uniform_bound,
                                    "second_moment_over_ell_squared": 2 - 2*q*length**2/(1-q)**2,
                                    "transfers": transfers_window, "gaussian_loads": gaussian})
    return {"scope": "changed constant-density 1D continuum response; no PDE advancement",
            "radius_over_ell": radius, "zeroth_moment": 1.0,
            "second_moment_over_ell_squared": second,
            "fourth_moment_over_ell_fourth": fourth,
            "elliptic_fourth_moment_over_ell_fourth": 24.0,
            "transfers": transfers, "gaussian_loads": loads,
            "truncated_exponential_autocorrelation": exponential_windows,
            "autocorrelation_radius_for_uniform_relative_error": [
                {"relative_error": eps, "radius_over_ell": 2 * math.asinh(1 / math.sqrt(eps))}
                for eps in (.1, .01, 1e-4, 1e-6)]}


def from_coo(spec, n):
    return sparse.coo_matrix((spec["values"], (spec["rows"], spec["cols"])),
                             shape=(n, n)).tocsr()


def assemble_blocks(blocks, element_dofs, n):
    rows, cols, values = [], [], []
    for dofs, block in zip(element_dofs, blocks):
        q = len(dofs)
        rows.extend(np.repeat(dofs, q))
        cols.extend(np.tile(dofs, q))
        values.extend(np.asarray(block).ravel())
    return sparse.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()


def patch_indices(element_dofs, neighbors, radius):
    result = []
    for center in range(len(element_dofs)):
        found, frontier = {center}, {center}
        for _ in range(radius):
            frontier = {int(j) for i in frontier for j in neighbors[i]} - found
            found |= frontier
        result.append(np.concatenate([element_dofs[e] for e in sorted(found)]))
    return result


def local_maps(a, m, element_dofs, neighbors, radius, max_patch_dofs):
    """SPD additive map and SPD constant-response-preserving secant variant."""
    patches = patch_indices(element_dofs, neighbors, radius)
    sizes = np.asarray([len(ids) for ids in patches])
    if sizes.max() > max_patch_dofs:
        return None, {"status": "skipped_patch_cap", "max_patch_dofs": int(sizes.max())}
    begin = time.perf_counter()
    coverage = np.zeros(a.shape[0])
    for ids in patches:
        coverage[ids] += 1
    partition = 1 / coverage
    rows, cols, ordinary, corrected = [], [], [], []
    max_secant_error, min_secant_curvature = 0.0, float("inf")
    for ids in patches:
        q = len(ids)
        local = a[ids][:, ids].toarray()
        factor = linalg.cho_factor(local, lower=True, check_finite=False)
        h = linalg.cho_solve(factor, np.eye(q), check_finite=False)
        target = partition[ids]
        scale = np.sqrt(target)
        h = scale[:, None] * h * scale[None, :]
        y = m[ids]
        curvature = float(y @ target)
        if curvature <= 0:
            raise ValueError("Local secant has nonpositive curvature; assumptions failed")
        # Inverse BFGS written with O(q^2) outer products, not dense I-products.
        hy = h @ y
        hp = h - (np.outer(target, hy) + np.outer(hy, target)) / curvature
        hp += ((1 + float(y @ hy) / curvature) / curvature) * np.outer(target, target)
        max_secant_error = max(max_secant_error, np.linalg.norm(hp @ y - target)
                              / np.linalg.norm(target))
        min_secant_curvature = min(min_secant_curvature, curvature)
        rows.append(np.repeat(ids, q))
        cols.append(np.tile(ids, q))
        ordinary.append(h.ravel())
        corrected.append(hp.ravel())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    maps = [sparse.coo_matrix((np.concatenate(values), (rows, cols)), shape=a.shape).tocsr()
            for values in (ordinary, corrected)]
    products = int(np.sum(sizes.astype(np.int64)**2))
    padded = int(np.sum((16 * ((sizes + 15) // 16)).astype(np.int64)**2))
    info = {"status": "completed", "radius_in_element_graph": radius,
            "response_graph_support_bound": 2 * radius,
            "patch_count": len(patches), "max_patch_dofs": int(sizes.max()),
            "local_factorization_flops_cholesky_estimate": float(np.sum(sizes.astype(float)**3) / 3),
            "explicit_local_inverse_extra_work": "q triangular solves per patch; not included in factorization FLOPs",
            "local_apply_products": products, "tile16_padded_products": padded,
            "tile16_useful_fraction": products / padded,
            "local_float64_matrix_bytes": 8 * products,
            "patch_construction_seconds_for_both_maps": time.perf_counter() - begin,
            "max_local_secant_relative_error": max_secant_error,
            "min_local_secant_curvature": min_secant_curvature,
            "scatter_note": "symmetric additive patches overlap outputs; gather/scatter/reduction costs are required"}
    return maps, info


def response_metrics(bmap, a, rhs, exact, mass, coords, m, spectral_factor=None):
    begin = time.perf_counter()
    sigma = bmap @ rhs
    apply_seconds = time.perf_counter() - begin
    error = sigma - exact
    norm = lambda x, op: math.sqrt(max(0, float(x @ (op @ x))))
    constant = bmap @ m - np.ones(len(rhs))
    weights = mass @ np.ones(len(rhs))
    moment_errors = []
    for axis in range(coords.shape[1]):
        x = coords[:, axis] - .5 * (coords[:, axis].min() + coords[:, axis].max())
        extent = float(np.max(np.abs(x)))
        if extent > 0:
            x = x / extent
        for degree in (0, 1, 2):
            multiplier = x**degree
            moment_errors.append({"axis": axis, "degree": degree,
                                  "absolute_error": abs(float((weights * multiplier) @ error)),
                                  "relative_to_absolute_reference_moment": ratio(
                                      abs(float((weights * multiplier) @ error)),
                                      float(np.abs(weights * multiplier * exact).sum()))})
    out = {"relative_physical_l2": ratio(norm(error, mass), norm(exact, mass)),
           "relative_operator_energy": ratio(norm(error, a), norm(exact, a)),
           "relative_current_closure_residual": ratio(np.linalg.norm(a @ sigma - rhs), np.linalg.norm(rhs)),
           "constant_response_relative_l2": norm(constant, mass) / norm(np.ones(len(rhs)), mass),
           "weak_constant_balance_absolute_error": abs(float(m @ sigma - rhs.sum())),
           "weak_constant_balance_relative_error": ratio(abs(float(m @ sigma - rhs.sum())), float(np.abs(rhs).sum())),
           "response_moments": moment_errors,
           "response_min": float(sigma.min()), "response_max": float(sigma.max()),
           "probe_map_apply_seconds": apply_seconds,
           "timing_scope": "single CPU application of diagnostic assembled map; not GPU or application timing"}
    if spectral_factor is not None:
        dense = bmap.toarray() if sparse.issparse(bmap) else bmap
        # A=L L^T, so L^T B L has the eigenvalues of B A.
        spectrum = linalg.eigvalsh(spectral_factor.T @ dense @ spectral_factor,
                                  check_finite=False)
        raw = linalg.eigvalsh(dense, check_finite=False)
        out.update({"response_min_eigenvalue": float(raw[0]),
                    "preconditioned_min_eigenvalue": float(spectrum[0]),
                    "preconditioned_max_eigenvalue": float(spectrum[-1]),
                    "worst_case_relative_energy_operator_error": float(np.max(np.abs(spectrum - 1))),
                    "optimal_richardson_contraction_from_exact_spectrum":
                        float((spectrum[-1] - spectrum[0]) / (spectrum[-1] + spectrum[0]))})
    else:
        out["spectrum_status"] = "not computed above bounded dense-oracle cap; SPD follows construction assumptions"
    return out


def spectral_wave_maps(a, reaction, horizons=(2, 4, 6, 8)):
    """Exact small-matrix oracle for the finite-wave-horizon filter.

    This dense semidiscrete oracle is not strictly graph-local. Finite continuum
    propagation and a numerically local wave evaluation are separate questions.
    """
    begin = time.perf_counter()
    eigenvalues, vectors = linalg.eigh(a.toarray(), reaction.toarray(), check_finite=False)
    lam = eigenvalues - 1
    if lam[0] < -1e-9 * max(1, abs(lam[-1])):
        raise ValueError("Wave-filter bound requires positive-semidefinite diffusion A-R")
    raw_min = float(lam[0])
    lam = np.maximum(lam, 0)
    setup = time.perf_counter() - begin
    out = []
    for horizon in horizons:
        q = math.exp(-horizon)
        delta = 4 * q * np.sin(.5 * horizon * np.sqrt(lam))**2 / (1-q)**2
        multiplier = (1 + delta) / (1 + lam)
        bmap = (vectors * multiplier[None, :]) @ vectors.T
        out.append((bmap, {"family": "finite_wave_horizon_spectral_oracle",
                           "dimensionless_wave_horizon": horizon,
                           "uniform_relative_energy_error_bound": 4*q/(1-q)**2,
                           "actual_spectral_relative_energy_error": float(delta.max()),
                           "diffusion_generalized_min_eigenvalue": raw_min,
                           "diffusion_generalized_max_eigenvalue": float(lam[-1]),
                           "minimum_central_difference_steps_for_stability_only":
                               int(math.ceil(horizon * math.sqrt(lam[-1]) / 2)),
                           "shared_generalized_eigendecomposition_seconds": setup,
                           "support_status": "dense semidiscrete spectral oracle; continuum finite propagation is not discrete graph compactness",
                           "cost_status": "global spectral setup and dense application; no scalable execution or tensor speedup claim"}))
    return out


def frozen_screen(path, patch_radii, taper_radii, oracle_cap, patch_cap):
    source = json.loads(path.read_text())
    n = int(source["n"])
    a = from_coo(source["A"], n)
    rhs = np.asarray(source["b"], float)
    coords = np.asarray(source["dof_coordinates"], float)
    elements = [np.asarray(dofs, int) for dofs in source["element_dofs"]]
    neighbors = source["neighbors"]
    mass = assemble_blocks(source["mass_blocks"], elements, n)
    m = a @ np.ones(n)
    symmetry = spla.norm(a - a.T) / spla.norm(a)
    if symmetry > 1e-10:
        raise ValueError(f"SPD response screen requires symmetric A; defect {symmetry}")
    begin = time.perf_counter()
    exact = spla.spsolve(a.tocsc(), rhs)
    ref_seconds = time.perf_counter() - begin
    reference_residual = ratio(np.linalg.norm(a @ exact - rhs), np.linalg.norm(rhs))
    if reference_residual is not None and reference_residual > 1e-8:
        raise ValueError("Reference residual exceeds bounded screen qualification")
    spectral_factor, inverse = None, None
    oracle_seconds = None
    if n <= oracle_cap:
        begin = time.perf_counter()
        spectral_factor = linalg.cholesky(a.toarray(), lower=True)
        inverse = linalg.cho_solve((spectral_factor, True), np.eye(n), check_finite=False)
        oracle_seconds = time.perf_counter() - begin
    excluded = {"A", "b", "mass_blocks", "reaction_blocks", "dof_coordinates", "element_dofs",
                "element_centers", "neighbors", "reference_nodes", "element_alpha", "volume_blocks"}
    metadata = {key: value for key, value in source.items() if key not in excluded}
    if "element_alpha" in source:
        metadata["element_alpha_min"] = min(source["element_alpha"])
        metadata["element_alpha_max"] = max(source["element_alpha"])
    out = {"input_path": str(path.resolve()), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "metadata": metadata, "n": n, "ne": len(elements),
           "order": source.get("order"), "operator_symmetry_relative_defect": symmetry,
           "reaction_load_min": float(m.min()), "reference_relative_residual": reference_residual,
           "reference_sparse_solve_seconds": ref_seconds,
           "oracle_dense_inverse_seconds": oracle_seconds,
           "oracle_dense_inverse_bytes": 8 * n * n if inverse is not None else 0,
           "oracle_cost_scope": "global construction; tapering is a model-effect oracle, not a scalable setup algorithm",
           "responses": []}
    if "reaction_blocks" in source:
        reaction = assemble_blocks(source["reaction_blocks"], elements, n)
        reaction_m = reaction @ np.ones(n)
        out["A_one_vs_reaction_constant_load_relative_error"] = ratio(np.linalg.norm(m - reaction_m), np.linalg.norm(reaction_m))
        if n <= oracle_cap:
            for bmap, info in spectral_wave_maps(a, reaction):
                metrics = response_metrics(bmap, a, rhs, exact, mass, coords, m)
                metrics["spectrum_status"] = "exact generalized diffusion spectrum computed; raw response eigenvalues not separately evaluated"
                out["responses"].append({**info, **metrics})
    for radius in patch_radii:
        maps, info = local_maps(a, m, elements, neighbors, radius, patch_cap)
        if maps is None:
            out["responses"].append({"family": "symmetric_patch", "radius": radius, **info})
            continue
        for family, bmap in zip(("symmetric_principal_patch", "symmetric_moment_preserving_patch"), maps):
            out["responses"].append({"family": family, **info,
                                     "assembled_union_products": int(bmap.nnz),
                                     **response_metrics(bmap, a, rhs, exact, mass, coords, m, spectral_factor)})
    if inverse is not None:
        distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        centers = np.asarray(source["element_centers"], float)
        adjacent = [np.linalg.norm(centers[e] - centers[j])
                    for e in range(len(elements)) for j in neighbors[e] if j > e]
        if not adjacent:
            raise ValueError("Taper screen requires a connected multi-element geometry")
        h = float(np.median(adjacent))
        out["taper_length_unit_median_adjacent_center_distance"] = h
        for radius in taper_radii:
            begin = time.perf_counter()
            taper = wendland(distances / (radius * h))
            bmap = inverse * taper
            setup = time.perf_counter() - begin
            out["responses"].append({"family": "oracle_wendland_taper",
                                     "radius_in_length_units": radius,
                                     "physical_support_radius": radius * h,
                                     "taper_and_product_seconds_excluding_oracle": setup,
                                     "nonzero_products": int(np.count_nonzero(bmap)),
                                     **response_metrics(bmap, a, rhs, exact, mass, coords, m, spectral_factor)})
    else:
        out["taper_status"] = "skipped: exact inverse would exceed bounded global oracle cap"
    return out


def self_test():
    n = 12
    lap = sparse.diags([-np.ones(n - 1), np.r_[1, np.full(n - 2, 2), 1], -np.ones(n - 1)], [-1, 0, 1])
    m = np.linspace(.3, 1.1, n)
    a = (sparse.diags(m) + 2 * lap).tocsr()
    elements = [np.arange(2 * i, 2 * i + 2) for i in range(n // 2)]
    neighbors = [[j for j in (e - 1, e + 1) if 0 <= j < len(elements)] for e in range(len(elements))]
    for radius in (0, 1, 2):
        maps, _ = local_maps(a, m, elements, neighbors, radius, 64)
        for bmap in maps:
            assert linalg.eigvalsh(bmap.toarray())[0] > 0
            assert spla.norm(bmap - bmap.T) < 1e-12
        assert np.linalg.norm(maps[1] @ m - 1) < 1e-12
        load = np.sin(np.arange(n))
        assert abs(m @ (maps[1] @ load) - load.sum()) < 1e-12
    inverse = linalg.inv(a.toarray())
    x = np.repeat(np.arange(n // 2), 2).astype(float)  # duplicated DG coordinates are permitted
    bmap = inverse * wendland(np.abs(x[:, None] - x[None, :]) / 2)
    assert linalg.eigvalsh(bmap)[0] > 0
    assert np.all(bmap[np.abs(x[:, None] - x[None, :]) >= 2] == 0)
    continuum = continuum_screen()
    assert abs(continuum["second_moment_over_ell_squared"] - 2) < 1e-12
    assert abs(continuum["fourth_moment_over_ell_fourth"] - 11.2) < 1e-11
    for length in (2, 6, 12):
        q = math.exp(-length)
        kernel = lambda r: (math.exp(-r) - math.exp(-(2 * length - r))) / (2 * (1 - q)**2)
        assert abs(2 * integrate.quad(kernel, 0, length)[0] - 1) < 1e-12
        for k in (.1, 1, 4):
            transform = 2 * integrate.quad(lambda r: kernel(r) * math.cos(k*r), 0, length,
                                          epsabs=1e-13, epsrel=1e-13)[0]
            formula = (1 + q*q - 2*q*math.cos(k*length)) / ((1-q)**2 * (1+k*k))
            assert abs(transform - formula) < 1e-12
    for bmap, info in spectral_wave_maps(a, sparse.diags(m)):
        assert np.linalg.norm(bmap @ m - 1) < 1e-12
        assert linalg.eigvalsh(bmap)[0] > 0
        assert info["actual_spectral_relative_energy_error"] <= info["uniform_relative_energy_error_bound"] + 1e-14
    return {"passed": True, "checks": ["symmetric positive patch maps", "constant-response secant",
            "weak constant balance", "taper SPD with duplicate coordinates", "strict finite support", "continuum moments",
            "exponential autocorrelation normalization and Fourier formula", "variable-reaction spectral wave filter"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-radii", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--taper-radii", type=float, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--max-oracle-dofs", type=int, default=1024)
    parser.add_argument("--max-patch-dofs", type=int, default=512)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--wave-only", action="store_true",
                        help="Evaluate only finite-wave-horizon spectral oracles on frozen cases")
    args = parser.parse_args()
    if min(args.patch_radii) < 0 or min(args.taper_radii) <= 0:
        parser.error("Patch radii must be nonnegative and taper radii positive")
    begin = time.perf_counter()
    result = {"schema": "igr-compact-response-probe-v1", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "command": [sys.executable] + sys.argv,
              "numpy_version": np.__version__, "scipy_version": scipy.__version__,
              "classification": "finite-range research responses; no production continuum adoption or tensor performance claim",
              "continuum": continuum_screen(), "frozen_cases": []}
    if args.self_test:
        result["self_test"] = self_test()
    for path in args.input:
        result["frozen_cases"].append(frozen_screen(path, [] if args.wave_only else args.patch_radii,
                                                  [] if args.wave_only else args.taper_radii,
                                                  args.max_oracle_dofs, args.max_patch_dofs))
    result["total_probe_seconds"] = time.perf_counter() - begin
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(result["frozen_cases"]),
                      "seconds": result["total_probe_seconds"], "self_test": result.get("self_test")}))


if __name__ == "__main__":
    main()
