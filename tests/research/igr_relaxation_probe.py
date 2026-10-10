#!/usr/bin/env python3
"""Bounded frozen-production-operator artificial-time relaxation experiment.

This is a local-metric damped-wave / heavy-ball algebraic solver, not a new
hyperbolic DG discretization. JSON input contains A, b, mass, element_dofs;
optional reaction supplies a local coercivity bound, and --previous supplies
an actual previous frozen state for warm starting. Dense sampling, reference
factorization and eigenvalue audits are test-only and separately charged.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def norm(x):
    return float(la.norm(x))


def ratio(a, b):
    return float(a / b) if b else (0.0 if a == 0 else None)


def matrix(value):
    if isinstance(value, dict):
        if "indptr" in value:
            return sp.csr_matrix((value["data"], value["indices"],
                                  value["indptr"]), shape=value["shape"])
        vals = value.get("data", value.get("values"))
        rows = value.get("row", value.get("rows"))
        cols = value.get("col", value.get("cols"))
        shape = value.get("shape")
        return sp.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()
    return sp.csr_matrix(np.asarray(value, dtype=float))


def block_matrix(dofs, blocks, n):
    row, col, val = [], [], []
    for idx, block in zip(dofs, blocks):
        row.extend(np.repeat(idx, len(idx)))
        col.extend(np.tile(idx, len(idx)))
        val.extend(np.asarray(block).ravel())
    return sp.coo_matrix((val, (row, col)), shape=(n, n)).tocsr()


def load(path):
    start = time.perf_counter()
    raw = json.loads(Path(path).read_text())
    a = matrix(raw["A"])
    b = np.asarray(raw["b"], dtype=float)
    dofs = [np.asarray(x, dtype=int) for x in raw["element_dofs"]]
    n = len(b)
    mass = (matrix(raw["mass"]) if "mass" in raw else
            block_matrix(dofs, raw["mass_blocks"], n))
    reaction = (matrix(raw["reaction"]) if "reaction" in raw else
                block_matrix(dofs, raw["reaction_blocks"], n)
                if "reaction_blocks" in raw else None)
    if a.shape != (n, n) or mass.shape != (n, n):
        raise ValueError("A, mass and b shapes differ")
    if not np.array_equal(np.sort(np.concatenate(dofs)), np.arange(n)):
        raise ValueError("Element DOFs must be an unsigned disjoint partition")
    if not np.isfinite(a.data).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite source data")
    symmetry = norm((a - a.T).data) / max(norm(a.data), 1e-300)
    if symmetry > 1e-11:
        raise ValueError(f"Damped-wave SPD derivation requires symmetry: {symmetry}")
    # Preserve the actual sampled A action. Only the eigenvalue audit uses its
    # symmetric part, whose discrepancy is checked above and reported.
    return {"raw": raw, "A": a, "b": b, "mass": mass,
            "reaction": reaction, "dofs": dofs, "symmetry": symmetry,
            "read_seconds": time.perf_counter() - start,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


class LocalMetric:
    def __init__(self, data, kind):
        start = time.perf_counter()
        source = data["A"] if kind == "element" else data["mass"]
        if kind == "reaction":
            source = data["reaction"]
            if source is None:
                raise ValueError("reaction metric requires reaction matrix")
        self.dofs = data["dofs"]
        self.blocks = [source[idx][:, idx].toarray() for idx in self.dofs]
        self.factors = [la.cho_factor((x + x.T) * 0.5, lower=True,
                                      check_finite=True) for x in self.blocks]
        self.sparse = block_matrix(self.dofs, self.blocks, source.shape[0])
        self.factor_seconds = time.perf_counter() - start
        self.apply_calls = 0
        self.apply_seconds = 0.0
        self.factor_flops = float(sum(len(idx) ** 3 / 3 for idx in self.dofs))
        self.apply_flops = int(sum(2 * len(idx) ** 2 for idx in self.dofs))

    def apply(self, rhs):
        start = time.perf_counter()
        out = np.empty_like(rhs)
        for idx, factor in zip(self.dofs, self.factors):
            out[idx] = la.cho_solve(factor, rhs[idx], check_finite=False)
        self.apply_calls += 1
        self.apply_seconds += time.perf_counter() - start
        return out

    def dual_norm(self, rhs):
        return math.sqrt(max(float(rhs @ self.apply(rhs)), 0.0))


def spectrum(data, metric):
    eig = None
    audit_seconds = 0.0
    if len(data["b"]) <= 1600:
        start = time.perf_counter()
        dense_a = data["A"].toarray()
        eig = la.eigvalsh((dense_a + dense_a.T) * 0.5, metric.sparse.toarray(),
                         check_finite=False)
        if eig[0] <= 0:
            raise ValueError(f"Current operator is not SPD: lambda_min={eig[0]}")
        audit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    # B is element-block diagonal; form only occupied local/face columns.
    upper, workspace, rhs_count = 0.0, 0, 0
    for idx, factor in zip(metric.dofs, metric.factors):
        arows = data["A"][idx]
        cols = np.unique(arows.indices)
        local = arows[:, cols].toarray()
        scaled = la.cho_solve(factor, local, check_finite=False)
        upper = max(upper, float(np.max(np.sum(np.abs(scaled), axis=1))))
        workspace = max(workspace, local.nbytes + scaled.nbytes)
        rhs_count += len(cols)
    reaction = data["reaction"]
    lower = None
    reaction_psd_gap = None
    coercivity_audit_seconds = 0.0
    if reaction is not None:
        lower = min(float(la.eigvalsh(reaction[idx][:, idx].toarray(), block,
                                      check_finite=False)[0])
                    for idx, block in zip(metric.dofs, metric.blocks))
    local_bounds_seconds = time.perf_counter() - start
    if reaction is not None and eig is not None:
        start = time.perf_counter()
        # Audit of the sufficient assumption A >= R. This is NOT a local
        # implementation and cannot be silently amortized in performance.
        k = dense_a - reaction.toarray()
        reaction_psd_gap = float(la.eigvalsh((k + k.T) * 0.5,
                                            metric.sparse.toarray(),
                                            check_finite=False)[0])
        coercivity_audit_seconds = time.perf_counter() - start
        if reaction_psd_gap < -1e-9 * max(1.0, float(eig[-1])):
            lower = None
    return {"exact_min": float(eig[0]) if eig is not None else None,
            "exact_max": float(eig[-1]) if eig is not None else None,
            "condition": float(eig[-1] / eig[0]) if eig is not None else None,
            "oracle_audit_seconds": audit_seconds,
            "local_bound_seconds": local_bounds_seconds,
            "local_upper": upper, "local_lower": lower,
            "diffusion_psd_gap": reaction_psd_gap,
            "coercivity_audit_seconds": coercivity_audit_seconds,
            "bounds_dense_workspace_bytes": int(workspace),
            "bounds_local_inverse_rhs_count": int(rhs_count),
            "lower_bound_status": "globally_audited" if eig is not None else "conditional_on_SIPG_diffusion_PSD",
            "bounds_note": "upper row-sum bound; lower requires A>=reaction; global PSD audit charged when <=1600 DOFs"}


def parameters(lower, upper):
    if not (0 < lower <= upper):
        raise ValueError("Invalid spectral enclosure")
    lo, hi = math.sqrt(lower), math.sqrt(upper)
    q = (hi - lo) / (hi + lo)
    return {"lower": lower, "upper": upper,
            "step": 4 / (hi + lo) ** 2, "momentum": q * q,
            "pseudo_dt": 2 / math.sqrt(upper + lower),
            "damping_eta": math.sqrt(lower * upper / (lower + upper)),
            "asymptotic_factor": q}


def relaxation(data, metric, reference, initial, params, tolerance, cap,
               checkpoint, label):
    a, b, mass = data["A"], data["b"], data["mass"]
    x = initial.copy()
    previous = x.copy()  # zero backward artificial-time velocity
    bnorm = norm(b)
    refnorm = math.sqrt(max(float(reference @ (mass @ reference)), 0.0))
    matvecs, certificates, reductions = 0, 0, 1  # RHS norm
    iterate_seconds, certificate_seconds = 0.0, 0.0
    metric_calls0, metric_seconds0 = metric.apply_calls, metric.apply_seconds
    rows = []
    complete = False
    initial_residual = None
    peak = 0.0
    for k in range(cap + 1):
        start = time.perf_counter()
        residual = b - a @ x
        matvecs += 1
        if k == 0:
            initial_residual = norm(residual)
        iterate_seconds += time.perf_counter() - start
        if k % checkpoint == 0 or k == cap:
            start = time.perf_counter()
            current = initial_residual if k == 0 else norm(residual)
            certificates += 1
            reductions += 1
            peak = max(peak, current / max(bnorm, 1e-300))
            error = x - reference
            error_l2 = math.sqrt(max(float(error @ (mass @ error)), 0.0))
            bound = (k + 1 + k * params["asymptotic_factor"]) * params["asymptotic_factor"] ** k
            rows.append({"sweeps": k, "residual_relative": ratio(current, bnorm),
                         "physical_l2_error_relative": ratio(error_l2, refnorm),
                         "error_l2_absolute": error_l2,
                         "dual_B_residual_bound_factor": bound})
            complete = current <= tolerance * bnorm
            certificate_seconds += time.perf_counter() - start
            if complete or not np.isfinite(current) or current > 1e10 * max(bnorm, 1e-300):
                break
        if k == cap:
            break
        start = time.perf_counter()
        correction = metric.apply(residual)
        new = x + params["momentum"] * (x - previous) + params["step"] * correction
        previous, x = x, new
        iterate_seconds += time.perf_counter() - start
    # Last residual multiplication is certification only; earlier products
    # both advance and can certify the current iterate without another A call.
    return {"method": "centered-damped-wave-heavy-ball", "initialization": label,
            "parameters": params, "complete": complete, "sweeps": k,
            "current_residual_relative": rows[-1]["residual_relative"],
            "physical_l2_error_relative": rows[-1]["physical_l2_error_relative"],
            "initial_residual_relative": ratio(initial_residual, bnorm),
            "peak_sampled_residual_relative": peak,
            "A_applications": matvecs, "B_inverse_applications": metric.apply_calls - metric_calls0,
            "matvec_nonzero_products": int(matvecs * a.nnz),
            "local_solve_flops": int((metric.apply_calls - metric_calls0) * metric.apply_flops),
            "metric_apply_seconds_in_iteration": metric.apply_seconds - metric_seconds0,
            "iterate_seconds": iterate_seconds, "certificate_seconds": certificate_seconds,
            "certificate_reductions": reductions, "certificate_count": certificates,
            "solver_vector_storage_bytes": int(5 * len(b) * 8),
            "new_rhs_influence_hops_bound": max(k - 1, 0),
            "diagnostic_physical_norms_are_test_only": True,
            "history": rows}


def cg_control(data, metric, reference, initial, tolerance, cap):
    # Explicit PCG permits unambiguous work/reduction counts and checks the
    # true current residual when the recursive residual first meets tolerance.
    a, b, mass = data["A"], data["b"], data["mass"]
    start = time.perf_counter()
    x = initial.copy()
    r = b - a @ x
    matvecs, reductions = 1, 2  # initial RHS and residual norms
    bnorm = norm(b)
    calls0 = metric.apply_calls
    complete = norm(r) <= tolerance * bnorm
    p = None
    old = None
    k = 0
    for k in range(1, cap + 1):
        if complete:
            k -= 1
            break
        z = metric.apply(r)
        rz = float(r @ z)
        reductions += 1
        p = z.copy() if p is None else z + (rz / old) * p
        ap = a @ p
        matvecs += 1
        denom = float(p @ ap)
        reductions += 1
        if denom <= 0:
            raise ValueError("PCG encountered nonpositive curvature")
        step = rz / denom
        x += step * p
        r -= step * ap
        old = rz
        reductions += 1
        if norm(r) <= tolerance * bnorm:
            r = b - a @ x
            matvecs += 1
            reductions += 1
            complete = norm(r) <= tolerance * bnorm
    r = b - a @ x
    matvecs += 1
    reductions += 1
    complete = norm(r) <= tolerance * bnorm
    error = x - reference
    en = math.sqrt(max(float(error @ (mass @ error)), 0.0))
    rn = math.sqrt(max(float(reference @ (mass @ reference)), 0.0))
    return {"method": "local-metric-PCG", "complete": complete,
            "iterations": k, "A_applications": matvecs,
            "B_inverse_applications": metric.apply_calls - calls0,
            "global_scalar_reductions_unfused": reductions,
            "current_residual_relative": ratio(norm(r), bnorm),
            "physical_l2_error_relative": ratio(en, rn),
            "iterate_and_certificate_seconds": time.perf_counter() - start}


def selftest():
    # Each eigenmode must agree with the analytical characteristic recurrence.
    lower, upper = 1.0, 25.0
    p = parameters(lower, upper)
    for lam in (lower, 7.0, upper):
        x, prev = 1.0, 1.0
        for k in range(1, 25):
            x, prev = x - p["step"] * lam * x + p["momentum"] * (x - prev), x
            assert abs(x) <= (k + 1 + k * p["asymptotic_factor"]) * p["asymptotic_factor"] ** k + 1e-12
    # Nearest-neighbor matrix and block-local inverse preserve finite-sweep
    # propagation; the exact inverse generally does not have this support.
    n = 11
    a = sp.diags((-np.ones(n - 1), 3 * np.ones(n), -np.ones(n - 1)), (-1, 0, 1)).tocsr()
    b = np.zeros(n)
    b[5] = 1
    x, prev = np.zeros(n), np.zeros(n)
    for k in range(1, 5):
        new = x + p["momentum"] * (x - prev) + p["step"] * (b - a @ x)
        x, prev = new, x
        assert np.all(x[np.abs(np.arange(n) - 5) > k - 1] == 0)
    assert np.count_nonzero(la.solve(a.toarray(), b)) == n
    print(json.dumps({"selftest": "pass", "modal_bound": True,
                      "finite_sweep_support": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics", default="mass,element")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--checkpoint", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        selftest()
        return
    if args.input is None or args.output is None:
        parser.error("--input and --output are required")
    if not (0 < args.tolerance < 1 and 1 <= args.max_iterations <= 10000 and args.checkpoint >= 1):
        parser.error("Invalid bounded iteration controls")
    begin = time.perf_counter()
    data = load(args.input)
    start = time.perf_counter()
    reference = spla.spsolve(data["A"].tocsc(), data["b"])
    reference_seconds = time.perf_counter() - start
    ref_residual = ratio(norm(data["A"] @ reference - data["b"]), norm(data["b"]))
    if ref_residual is None or ref_residual > 1e-9:
        raise ValueError("FP64 reference fails current-operator residual gate")
    initials = {"zero": np.zeros_like(reference)}
    previous_meta = None
    if args.previous:
        old = load(args.previous)
        if old["A"].shape != data["A"].shape or len(old["dofs"]) != len(data["dofs"]) or any(not np.array_equal(x, y)
                for x, y in zip(old["dofs"], data["dofs"])):
            raise ValueError("Warm start requires identical finite-element DOF layout")
        start = time.perf_counter()
        initials["previous_frozen_closure"] = spla.spsolve(old["A"].tocsc(), old["b"])
        previous_solve_seconds = time.perf_counter() - start
        previous_sigma = initials["previous_frozen_closure"]
        previous_residual = ratio(norm(old["b"] - old["A"] @ previous_sigma), norm(old["b"]))
        if previous_residual is None or previous_residual > 1e-9:
            raise ValueError("Previous FP64 closure fails its original residual gate")
        current_residual = data["b"] - data["A"] @ previous_sigma
        residual_identity = (data["b"] - old["b"]) - (data["A"] - old["A"]) @ previous_sigma
        previous_meta = {"input": str(args.previous), "sha256": old["sha256"],
                         "reference_solve_seconds": previous_solve_seconds,
                         "reference_current_residual_relative": previous_residual,
                         "next_rhs_relative_change": ratio(norm(data["b"] - old["b"]), norm(old["b"])),
                         "next_operator_relative_frobenius_change": ratio(norm((data["A"] - old["A"]).data), norm(old["A"].data)),
                         "warm_residual_identity_error_relative_rhs": ratio(norm(current_residual - residual_identity), norm(data["b"])),
                         "read_seconds": old["read_seconds"],
                         "note": "previous closure solve is charged separately, never a free synthetic perturbation"}
    output = {"schema": "igr-frozen-relaxation-v1", "input": str(args.input),
              "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "command": [sys.executable] + sys.argv,
              "input_sha256": data["sha256"],
              "source_metadata": data["raw"].get("metadata", {
                  key: value for key, value in data["raw"].items()
                  if not isinstance(value, (list, dict))}),
              "source_symmetry_relative": data["symmetry"],
              "input_read_seconds": data["read_seconds"],
              "reference_solve_seconds": reference_seconds,
              "reference_current_residual_relative": ref_residual,
              "previous": previous_meta, "tolerance": args.tolerance,
              "n": len(reference), "nnz": data["A"].nnz,
              "implementation": "NumPy/SciPy CPU; frozen algebraic mechanism only, no GPU timing or advancing-system claim",
              "metric_results": []}
    for kind in args.metrics.split(","):
        metric = LocalMetric(data, kind)
        spec = spectrum(data, metric)
        choices = {}
        if spec["exact_min"] is not None:
            choices["exact_spectrum_oracle"] = parameters(spec["exact_min"] * (1 - 1e-12),
                                                          spec["exact_max"] * (1 + 1e-12))
        if spec["local_lower"] is not None:
            choices["local_coercivity_and_row_sum"] = parameters(spec["local_lower"], spec["local_upper"])
        row = {"metric": kind, "metric_factor_seconds": metric.factor_seconds,
               "metric_factor_flops_estimate": metric.factor_flops,
               "spectrum": spec, "runs": []}
        for label, initial in initials.items():
            for bounds_name, par in choices.items():
                run = relaxation(data, metric, reference, initial, par,
                                 args.tolerance, args.max_iterations, args.checkpoint, label)
                run["bounds"] = bounds_name
                row["runs"].append(run)
            control = cg_control(data, metric, reference, initial, args.tolerance, args.max_iterations)
            control["initialization"] = label
            row["runs"].append(control)
        output["metric_results"].append(row)
    output["total_process_work_seconds"] = time.perf_counter() - begin
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "n": len(reference),
                      "metrics": len(output["metric_results"]),
                      "seconds": output["total_process_work_seconds"]}))


if __name__ == "__main__":
    main()
