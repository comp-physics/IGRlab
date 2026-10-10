#!/usr/bin/env python3
"""Finite-neighborhood inverse polynomials on frozen production IGR operators.

CPU research only. Compare ordinary Chebyshev correction with a positive
finite geometric-autocorrelation polynomial. References/eigenvalue oracles
are separately charged; neither is part of a proposed GPU implementation.
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

from igr_relaxation_probe import LocalMetric, cg_control, load, norm, spectrum


def finite_weights(lower, upper, stages):
    """Coefficients of F(A') = sum w_k T_k((a-A')/b)."""
    if not (0 < lower <= upper and stages >= 1):
        raise ValueError("Invalid spectrum/stage count")
    if upper == lower:
        return np.array([1 / lower]), 0.0
    s = math.sqrt(upper / lower)
    z = (s - 1) / (s + 1)
    logz = math.log(z)
    d = lower * (s + 1) ** 2 / 4
    denominator = d * (-math.expm1(stages * logz)) ** 2 * (-math.expm1(2 * logz))
    weights = np.array([(1 if k == 0 else 2) * math.exp(k * logz) *
                        (-math.expm1(2 * (stages - k) * logz)) / denominator
                        for k in range(stages)])
    q = math.exp(stages * logz)
    return weights, 4 * q / (1 - q) ** 2


def finite_apply(apply_scaled, rhs, lower, upper, stages):
    weights, bound = finite_weights(lower, upper, stages)
    previous = rhs.copy()
    result = weights[0] * previous
    if len(weights) == 1:
        return result, bound, 0
    center, radius = (upper + lower) / 2, (upper - lower) / 2
    current = (center * previous - apply_scaled(previous)) / radius
    result += weights[1] * current
    for k in range(2, len(weights)):
        following = 2 * (center * current - apply_scaled(current)) / radius - previous
        result += weights[k] * following
        previous, current = current, following
    return result, bound, len(weights) - 1


def measurements(data, x, reference, metric):
    residual = data["b"] - data["A"] @ x
    error = x - reference
    def relative(v, r, op):
        return math.sqrt(max(float(v @ (op @ v)), 0) /
                         max(float(r @ (op @ r)), 1e-300))
    return {"current_residual_relative": norm(residual) / max(norm(data["b"]), 1e-300),
            "physical_l2_error_relative": relative(error, reference, data["mass"]),
            "energy_error_relative": relative(error, reference, data["A"]),
            "B_dual_residual_relative": metric.dual_norm(residual) /
                                        max(metric.dual_norm(data["b"]), 1e-300)}


def chebyshev(data, metric, initial, reference, lower, upper, cap, tolerance, checkpoint):
    start = time.perf_counter()
    a, b = data["A"], data["b"]
    bnorm = max(norm(b), 1e-300)
    x = initial.copy()
    center, radius = (upper + lower) / 2, (upper - lower) / 2
    rho = radius / center
    correction = np.zeros_like(b)
    matvecs, metric_calls, reductions = 0, 0, 1
    diagnostics = 0.0
    history = []
    for k in range(cap + 1):
        r = b - a @ x
        matvecs += 1
        if k % checkpoint == 0 or k == cap:
            current_relative = norm(r) / bnorm
            diagnostic_start = time.perf_counter()
            row = measurements(data, x, reference, metric)
            row["stages"] = k
            history.append(row)
            diagnostics += time.perf_counter() - diagnostic_start
            reductions += 1
            if current_relative <= tolerance:
                break
        if k == cap:
            break
        z = metric.apply(r)
        metric_calls += 1
        if k == 0 or radius == 0:
            correction = z / center
        else:
            next_rho = 1 / (2 * center / radius - rho)
            correction = next_rho * rho * correction + 2 * next_rho / radius * z
            rho = next_rho
        x += correction
    q = (math.sqrt(upper / lower) - 1) / (math.sqrt(upper / lower) + 1)
    return {"method": "Chebyshev-correction", "stages": k,
            "complete": history[-1]["current_residual_relative"] <= tolerance,
            "A_applications_solver": matvecs, "B_inverse_applications_solver": metric_calls,
            "certificate_reductions": reductions, "new_rhs_influence_hops_bound": max(k - 1, 0),
            "energy_error_bound_relative_initial": min(1.0, 2 * q ** k),
            "iteration_and_certificate_seconds": time.perf_counter() - start - diagnostics,
            "test_only_diagnostics_seconds": diagnostics,
            "test_only_extra_A_applications": 3 * len(history),
            "test_only_extra_B_inverse_applications": 2 * len(history),
            "final": history[-1], "history": history}


def finite_run(data, metric, initial, reference, lower, upper, stages):
    start = time.perf_counter()
    r = data["b"] - data["A"] @ initial
    rhs = metric.apply(r)
    delta, bound, products = finite_apply(lambda v: metric.apply(data["A"] @ v),
                                         rhs, lower, upper, stages)
    elapsed = time.perf_counter() - start
    diagnostic_start = time.perf_counter()
    row = measurements(data, initial + delta, reference, metric)
    return {"method": "positive-finite-window-polynomial", "stages": stages,
            "degree": products, "new_rhs_influence_hops_bound": products,
            "uniform_relative_inverse_error_bound": bound,
            "bound_norm": "B-dual residual or A-energy error, relative to initial error",
            "A_applications_solver": products + 1,
            "B_inverse_applications_solver": products + 1,
            "certificate_A_applications": 1, "certificate_reductions": 2,
            "test_only_extra_A_applications": 2,
            "test_only_extra_B_inverse_applications": 2,
            "iteration_seconds": elapsed,
            "test_only_diagnostics_seconds": time.perf_counter() - diagnostic_start,
            "final": row}


def selftest():
    # Independent scalar transfer checks across four decades of conditioning.
    for upper in (1.0, 4.0, 100.0, 10000.0):
        eigenvalues = np.geomspace(1, upper, 301)
        for m in (1, 3, 17, 101):
            actual, bound, _ = finite_apply(lambda v: eigenvalues * v,
                                            np.ones(301), 1, upper, m)
            transfer = eigenvalues * actual
            assert np.min(transfer) >= 1 - 1e-10
            assert np.max(transfer - 1) <= bound + 1e-9
            assert abs(actual[0] - 1) < 2e-12
    # Finite support and constant preservation on a variable-coefficient graph,
    # whose exact inverse has nonzero influence beyond the polynomial radius.
    n = 15
    weights = np.linspace(.2, 2, n - 1)
    edge = sp.diags((-np.ones(n-1), np.ones(n-1)), (0, 1), shape=(n-1, n)).tocsr()
    reaction = np.linspace(.5, 2, n)
    a = sp.diags(reaction) + edge.T @ sp.diags(weights) @ edge
    upper = float(np.max(np.asarray(abs(sp.diags(1/reaction) @ a).sum(axis=1))))
    impulse = np.eye(n)[7]
    response, _, _ = finite_apply(lambda v: (a @ v) / reaction,
                                  impulse / reaction, 1, upper, 4)
    assert np.all(response[np.abs(np.arange(n) - 7) > 3] == 0)
    constant, _, _ = finite_apply(lambda v: (a @ v) / reaction, np.ones(n), 1, upper, 7)
    assert norm(constant - 1) < 1e-12
    assert np.count_nonzero(spla.spsolve(a.tocsc(), impulse)) == n
    print(json.dumps({"selftest": "pass", "uniform_transfer_bound": True,
                      "variable_reaction_constant_mode": True, "finite_support": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics", default="reaction,element")
    parser.add_argument("--stages", default="8,32,128,512")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=512)
    parser.add_argument("--checkpoint", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        selftest()
        return
    if args.input is None or args.output is None:
        parser.error("--input and --output required")
    stages = [int(x) for x in args.stages.split(",")]
    if not (0 < args.tolerance < 1 and 1 <= args.max_iterations <= 4096 and
            args.checkpoint >= 1 and all(1 <= x <= 4096 for x in stages)):
        parser.error("Invalid bounded controls")
    data = load(args.input)
    start = time.perf_counter()
    reference = spla.spsolve(data["A"].tocsc(), data["b"])
    ref_seconds = time.perf_counter() - start
    ref_residual = norm(data["A"] @ reference - data["b"]) / norm(data["b"])
    if ref_residual > 1e-9:
        raise ValueError("Reference residual fails gate")
    initials = {"zero": np.zeros_like(reference)}
    previous = None
    if args.previous:
        old = load(args.previous)
        if old["A"].shape != data["A"].shape or len(old["dofs"]) != len(data["dofs"]) or any(
                not np.array_equal(x, y) for x, y in zip(old["dofs"], data["dofs"])):
            raise ValueError("Previous layout differs")
        start = time.perf_counter()
        previous_solution = spla.spsolve(old["A"].tocsc(), old["b"])
        previous_seconds = time.perf_counter() - start
        residual = norm(old["A"] @ previous_solution - old["b"]) / norm(old["b"])
        if residual > 1e-9:
            raise ValueError("Previous reference fails gate")
        initials["previous_frozen_closure"] = previous_solution
        previous = {"sha256": old["sha256"], "reference_seconds": previous_seconds,
                    "reference_residual_relative": residual}
    out = {"schema": "igr-local-polynomial-v1", "command": [sys.executable] + sys.argv,
           "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "helper_sha256": hashlib.sha256(Path(__file__).with_name("igr_relaxation_probe.py").read_bytes()).hexdigest(),
           "input_sha256": data["sha256"], "n": len(reference), "nnz": data["A"].nnz,
           "reference_seconds": ref_seconds, "reference_residual_relative": ref_residual,
           "read_seconds": data["read_seconds"], "previous": previous,
           "implementation": "CPU frozen-operator mechanism screen; no GPU or time-to-solution claim",
           "metric_results": []}
    for kind in args.metrics.split(","):
        metric = LocalMetric(data, kind)
        spec = spectrum(data, metric)
        choices = {"local_conditional": (spec["local_lower"], spec["local_upper"])}
        if spec["exact_min"] is not None:
            choices["oracle"] = (spec["exact_min"] * (1 - 1e-12), spec["exact_max"] * (1 + 1e-12))
        row = {"metric": kind, "spectrum": spec, "factor_seconds": metric.factor_seconds,
               "factor_flops": metric.factor_flops, "apply_flops": metric.apply_flops, "runs": []}
        for label, initial in initials.items():
            control = cg_control(data, metric, reference, initial, args.tolerance, args.max_iterations)
            control["initialization"] = label
            row["runs"].append(control)
            for name, (lower, upper) in choices.items():
                if lower is None:
                    continue
                # Reaction scaling's known constant eigenvalue is one. Use it
                # exactly in the finite filter; oracle roundoff must not move it.
                finite_lower = 1.0 if kind == "reaction" else lower
                runs = [chebyshev(data, metric, initial, reference, lower, upper,
                                  args.max_iterations, args.tolerance, args.checkpoint)]
                runs += [finite_run(data, metric, initial, reference, finite_lower, upper, m)
                         for m in stages]
                for run in runs:
                    run.update({"initialization": label, "bounds": name,
                                "lower": finite_lower if run["method"].startswith("positive") else lower,
                                "upper": upper,
                                "filter_exact_on_constant_eigenmode": kind == "reaction" and run["method"].startswith("positive")})
                    row["runs"].append(run)
        out["metric_results"].append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "n": len(reference)}))


if __name__ == "__main__":
    main()
