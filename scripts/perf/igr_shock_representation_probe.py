#!/usr/bin/env python3
"""CPU-only approximation-capacity probe for an actual full-energy IGR shock.

Reference: https://arxiv.org/html/2607.12693v1, equations 19--23 and Remark 19.
The ideal-gas Mach-2 profile is reconstructed from its integrating factor,
not a synthetic cusp. x is measured in sqrt(alpha), so alpha=1 below.
Independent component L2 projections are compared at equal coefficient counts.
These are best approximations, not numerical solutions or runtime measurements.
No claim is made that a stationary profile is dynamically attracting.

Requires Python 3, NumPy and SciPy. Example:
  python3 scripts/perf/igr_shock_representation_probe.py --output /tmp/igr-profile.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import roots_legendre


class Profile:
    """Stationary rho-=p-=1, gamma=1.4, Mach=2, full-energy profile."""

    gamma = 1.4
    mach = 2.0
    rho_lo = 1.0
    rho_hi = (gamma + 1) * mach**2 / ((gamma - 1) * mach**2 + 2)
    mass_flux = mach * math.sqrt(gamma)
    momentum_flux = mass_flux**2 + 1
    e0 = gamma / (gamma - 1) + mass_flux**2 / 2
    energy_flux = mass_flux * e0
    a = 2 * (gamma - 1) / (gamma + 1)
    beta = 1 / (1 + a / 2)
    aa = (gamma + 1) * mass_flux**2 / 2
    bb = (gamma - 1) * e0
    rho_s = math.sqrt(aa / bb)

    def __init__(self, panels: int, quadrature: int):
        self.panels = panels
        self.quadrature = quadrature
        self.g, self.w = roots_legendre(quadrature)
        self.branches = {}
        # d=delta*v^6 removes the gamma=1.4 fractional powers from dx/dv
        # near the sonic point. Integrate each panel by Gauss quadrature.
        v = np.linspace(0, 0.999 ** (1 / 6), panels + 1)
        pg, pw = roots_legendre(12)
        for sign, endpoint in ((-1, self.rho_lo), (1, self.rho_hi)):
            delta = abs(endpoint - self.rho_s)
            increments = []
            for start in range(0, panels, 128):
                left, right = v[start:start + 128], v[start + 1:start + 129]
                vp = (left[:, None] + right[:, None]) / 2
                vp = vp + (right - left)[:, None] * pg / 2
                d = delta * vp**6
                z = self.slope_squared(d, sign)
                dx_dv = 6 * delta * vp**5 / np.sqrt(z)
                increments.extend((dx_dv @ pw) * (right - left) / 2)
            x = np.r_[0.0, np.cumsum(increments)]
            d = delta * v**6
            # Interpolate the regularized dependent variable d(|x|^beta).
            # Keeping d separate avoids subtracting nearly equal densities.
            interpolator = CubicSpline(x**self.beta, d, extrapolate=False)
            self.branches[sign] = (interpolator, x[-1])

    def sigma(self, rho):
        # Factored form avoids cancellation near the outer states.
        return self.bb * (rho - self.rho_lo) * (self.rho_hi - rho) / rho

    def sigma_r(self, rho):
        return self.aa / rho**2 - self.bb

    def slope_squared(self, d, sign):
        """Z=(d rho/dx)^2; positive integral after t=d^a substitution."""
        d = np.asarray(d)
        endpoint = self.rho_lo if sign < 0 else self.rho_hi
        delta = abs(endpoint - self.rho_s)
        lo, hi = d**self.a, delta**self.a
        t = (hi + lo[..., None]) / 2 + (hi - lo[..., None]) * self.g / 2
        rr = self.rho_s + sign * t ** (1 / self.a)
        integrand = (2 * self.sigma(rr) * (self.bb * (self.rho_s + rr)) **
                     (self.a - 1) * rr ** (-2 * self.a) / self.a)
        integral = (integrand @ self.w) * (hi - lo) / 2
        rho = self.rho_s + sign * d
        factor = (self.bb * d * (self.rho_s + rho)) ** self.a / rho ** (2 + 2 * self.a)
        with np.errstate(divide="ignore", invalid="ignore"):
            return integral / factor

    def density_distance(self, x):
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        for sign in (-1, 1):
            mask = x * sign > 0
            spline, xmax = self.branches[sign]
            if np.any(np.abs(x[mask]) > xmax):
                raise ValueError(f"Profile interpolation requested beyond {xmax}")
            result[mask] = spline(np.abs(x[mask])**self.beta)
        return result

    def fields(self, x):
        x = np.asarray(x, dtype=float)
        rho = self.rho_s + np.sign(x) * self.density_distance(x)
        energy = self.e0 * rho - self.momentum_flux + self.mass_flux**2 / rho
        return np.stack((rho, np.full_like(rho, self.mass_flux), energy, self.sigma(rho)), axis=-1)

    def density_derivative(self, x):
        x = np.asarray(x, dtype=float)
        if np.any(x == 0):
            raise ValueError("The exact density derivative is infinite at the sonic point")
        d = self.density_distance(x)
        result = np.empty_like(x)
        for sign in (-1, 1):
            mask = x * sign > 0
            result[mask] = np.sqrt(self.slope_squared(d[mask], sign))
        return result

    def source_moments(self, width):
        """Independent adaptive integration in density, with cusp weight removed."""
        values = np.zeros(3)
        absolute = np.zeros(3)
        errors = np.zeros(3)
        for sign in (-1, 1):
            dmax = float(self.density_distance(np.array([sign * width]))[0])
            spline, _ = self.branches[sign]
            # An independent inverse lookup is needed only for source moments.
            # x(d) is smooth in d^(1/beta); sample from the existing branch spline
            # via scalar bracketing, avoiding a second approximate inverse table.
            from scipy.optimize import brentq

            def integrand(t, power, absolute_value):
                # d=t^(1/(1-a/2)) makes sqrt(Z) dd integrable and nonsingular.
                exponent = 1 / (1 - self.a / 2)
                d = t**exponent
                rho = self.rho_s + sign * d
                z = float(self.slope_squared(d, sign))
                y = brentq(lambda y: float(spline(y)) - d, 0, width**self.beta,
                           xtol=2e-14, rtol=1e-14)
                x = sign * y**(1 / self.beta)
                moment = abs(x)**power if absolute_value else x**power
                return 2 * self.mass_flux**2 / rho**4 * math.sqrt(z) * exponent * t**(exponent - 1) * moment

            for power in range(3):
                upper = dmax**(1 - self.a / 2)
                val, err = quad(integrand, 0, upper, args=(power, False),
                                epsabs=2e-11, epsrel=2e-11, limit=150)
                av, _ = quad(integrand, 0, upper, args=(power, True),
                             epsabs=2e-11, epsrel=2e-11, limit=150)
                values[power] += val
                absolute[power] += av
                errors[power] += err
        return values, absolute, errors


def integration_rule(breaks, order):
    """Composite Gauss rule graded toward each possible singularity or DG face."""
    g, w = roots_legendre(order)
    points, weights = [], []
    for left, right in zip(breaks[:-1], breaks[1:]):
        middle = (left + right) / 2
        for end, length, sign in ((left, middle - left, 1), (right, right - middle, -1)):
            # Explicitly split small endpoint neighborhoods, retaining every
            # interval's measure. Order refinement detects quadrature error.
            radial = np.r_[0.0, np.geomspace(1e-12, 1.0, 13)]
            for lo, hi in zip(radial[:-1], radial[1:]):
                t = (lo + hi) / 2 + (hi - lo) * g / 2
                xx = end + sign * length * t
                # Avoid representational equality at nonzero shifted cusps.
                xx = np.where(xx == end, np.nextafter(end, middle), xx)
                points.extend(xx)
                weights.extend(length * (hi - lo) * w / 2)
    return np.asarray(points), np.asarray(weights)


def legendre_matrix(z, count, derivative=False):
    value = np.polynomial.legendre.legvander(z, count - 1)
    if not derivative:
        return value
    deriv = np.zeros_like(value)
    if count > 1:
        deriv[:, 1] = 1
    for n in range(2, count):
        deriv[:, n] = ((2*n-1) * (value[:, n-1] + z*deriv[:, n-1]) - (n-1)*deriv[:, n-2]) / n
    return deriv


class Representation:
    def __init__(self, kind, count, width, location, beta):
        self.kind, self.count = kind, count
        self.width, self.location, self.beta = width, location, beta
        if kind == "global_polynomial" or kind == "cusp_enriched":
            self.breaks = [-width, width]
            self.counts = [count if kind == "global_polynomial" else count - 4]
        elif kind == "split_polynomial":
            self.breaks = [-width, location, width]
            self.counts = [count // 2, count - count // 2]
        elif kind == "uniform_p3":
            self.breaks = np.linspace(-width, width, count // 4 + 1).tolist()
            self.counts = [4] * (count // 4)
        elif kind in ("graded_p3", "graded_hp"):
            levels = count // 8 if kind == "graded_p3" else min(count // 4, math.ceil(math.sqrt(count/2)))
            self.breaks = sorted(set([-width, width, location] +
                [location - (width + location) * 0.25**j for j in range(levels)] +
                [location + (width - location) * 0.25**j for j in range(levels)]))
            if kind == "graded_p3":
                self.counts = [4] * (2 * levels)
            else:
                # Geometric h refinement plus increasing p away from the cusp.
                # Count every local coefficient; no hidden continuity reduction.
                quota = (count/2-2*levels)*np.arange(levels, 0, -1)/sum(range(1, levels+1))
                allocation = np.floor(quota).astype(int)+2
                remaining = count//2-int(sum(allocation))
                for index in np.argsort(-(quota-np.floor(quota)))[:remaining]:
                    allocation[index] += 1
                self.counts = allocation.tolist()+allocation[::-1].tolist()
        else:
            raise ValueError(kind)
        self.quad_breaks = sorted(set(self.breaks + [0.0, location]))

    def basis(self, x, derivative=False):
        if derivative and self.kind == "cusp_enriched" and np.any(x == self.location):
            raise ValueError("The enriched density basis derivative is undefined at its cusp; integrate it weakly")
        result = np.zeros((len(x), self.count))
        offset = 0
        for cell, count in enumerate(self.counts):
            left, right = self.breaks[cell:cell+2]
            mask = (x >= left) & ((x < right) if cell < len(self.counts)-1 else (x <= right))
            z = (2*x[mask] - right - left) / (right - left)
            result[mask, offset:offset+count] = legendre_matrix(z, count, derivative) * (2/(right-left) if derivative else 1)
            offset += count
        if self.kind == "cusp_enriched":
            # Four independent coefficients: each one-sided beta and 2*beta
            # power. No exact profile values enter these basis functions.
            d = (x - self.location) / self.width
            for j, (sign, exponent) in enumerate(((-1, self.beta), (1, self.beta), (-1, 2*self.beta), (1, 2*self.beta))):
                mask = d * sign > 0
                result[mask, offset+j] = (sign*exponent/self.width * abs(d[mask])**(exponent-1)
                                               if derivative else abs(d[mask])**exponent)
        return result

    def project(self, profile, order):
        x, w = integration_rule(self.quad_breaks, order)
        matrix = self.basis(x)
        weighted = np.sqrt(w)[:, None] * matrix
        scales = np.linalg.norm(weighted, axis=0)
        coefficients, _, rank, singular = np.linalg.lstsq(weighted/scales, np.sqrt(w)[:, None]*profile.fields(x), rcond=None)
        self.coefficients = coefficients/scales[:, None]
        self.rank = int(rank)
        self.condition = float(singular[0]/singular[-1])

    def evaluate(self, x, derivative=False):
        return self.basis(np.asarray(x), derivative) @ self.coefficients


def fluxes(fields, gamma):
    rho, mom, energy, sigma = fields.T
    pressure = (gamma-1)*(energy-mom*mom/(2*rho))
    return np.stack((mom, mom*mom/rho + pressure + sigma,
                     (energy+pressure+sigma)*mom/rho), axis=-1), pressure


def measure(rep, profile, order, reference_moments, absolute_moments):
    x, w = integration_rule(rep.quad_breaks, order)
    truth, fit = profile.fields(x), rep.evaluate(x)
    derivative = rep.evaluate(x, True)
    delta = fit-truth
    l2 = np.sqrt(w @ (delta*delta))
    reference_l2 = np.sqrt(w @ (truth*truth))
    velocity_derivative = (derivative[:, 1]*fit[:, 0]-fit[:, 1]*derivative[:, 0])/fit[:, 0]**2
    source = 2*velocity_derivative**2
    moments = np.array([w @ (source*x**j) for j in range(3)])
    flux, pressure = fluxes(fit, profile.gamma)
    exact_flux = np.array([profile.mass_flux, profile.momentum_flux, profile.energy_flux])
    flux_l2 = np.sqrt(w @ ((flux-exact_flux)**2)/(2*rep.width))/np.abs(exact_flux)
    # A separate dense sample gives a sampled maximum, never a supremum claim.
    samples = np.unique(np.r_[np.linspace(-rep.width, rep.width, 8193), rep.quad_breaks])
    sample_error = np.max(np.abs(rep.evaluate(samples)-profile.fields(samples)), axis=0)
    jumps = []
    for boundary in rep.breaks[1:-1]:
        a = rep.evaluate([np.nextafter(boundary, -math.inf)])[0]
        b = rep.evaluate([np.nextafter(boundary, math.inf)])[0]
        jumps.append(np.abs(a-b))
    return {"l2_absolute_rho_momentum_energy_sigma": l2.tolist(),
            "l2_relative_rho_momentum_energy_sigma": (l2/reference_l2).tolist(),
            "sampled_max_absolute_rho_momentum_energy_sigma": sample_error.tolist(),
            "broken_gradient_source_moments_0_1_2": moments.tolist(),
            "source_moment_error_over_absolute_reference_moment": ((moments-reference_moments)/absolute_moments).tolist(),
            "relative_rms_mass_momentum_energy_flux_defect": flux_l2.tolist(),
            "max_interior_face_jump_rho_momentum_energy_sigma": np.max(jumps, axis=0).tolist() if jumps else [0.0]*4,
            "minimum_sampled_density": float(np.min(fit[:, 0])),
            "minimum_sampled_pressure": float(np.min(pressure)),
            "finite": bool(np.isfinite(fit).all() and np.isfinite(derivative).all() and np.isfinite(moments).all())}


def run(args):
    started = time.perf_counter()
    coarse = Profile(args.panels, args.profile_quadrature)
    profile = Profile(2*args.panels, 2*args.profile_quadrature)
    x, w = integration_rule([-args.width, 0.0, args.width], 64)
    coarse_fields, fields = coarse.fields(x), profile.fields(x)
    derivative = profile.density_derivative(x)
    source = 2*profile.mass_flux**2/fields[:, 0]**4*derivative**2
    moments, absolute_moments, moment_quad_errors = profile.source_moments(args.width)
    moment_rule = np.array([w @ (source*x**j) for j in range(3)])
    sigma_derivative = profile.sigma_r(fields[:, 0])*derivative
    boundary_x = np.array([-args.width, args.width])
    boundary_rho = profile.fields(boundary_x)[:, 0]
    boundary_flux = profile.sigma_r(boundary_rho)*profile.density_derivative(boundary_x)/boundary_rho
    weak = []
    for j in range(3):
        value = w @ ((fields[:, 3]/fields[:, 0]-source)*x**j)
        if j:
            value += w @ (sigma_derivative/fields[:, 0]*j*x**(j-1))
        value -= boundary_flux[1]*args.width**j-boundary_flux[0]*(-args.width)**j
        weak.append(float(value))
    reference_change = np.max(np.abs(coarse_fields-fields), axis=0)
    coarse_moments = coarse.source_moments(args.width)[0]
    reference_flux, reference_pressure = fluxes(fields, profile.gamma)
    exact_flux = np.array([profile.mass_flux, profile.momentum_flux, profile.energy_flux])
    valid = (np.max(reference_change) < args.reference_tolerance and
             np.max(abs(coarse_moments-moments)/absolute_moments) < args.reference_tolerance and
             np.max(abs(moment_rule-moments)/absolute_moments) < args.reference_tolerance and
             np.max(abs(np.asarray(weak)))/absolute_moments[0] < args.reference_tolerance and
             np.min(reference_pressure) > 0 and np.isfinite(fields).all() and
             np.isfinite(derivative).all() and np.isfinite(moments).all() and
             np.isfinite(weak).all())
    records = []
    if valid:
        for count in args.counts:
            for kind in ("global_polynomial", "uniform_p3", "split_polynomial", "graded_p3", "graded_hp", "cusp_enriched"):
                shifts = [0.0] if kind in ("global_polynomial", "uniform_p3") else args.location_errors
                for shift in shifts:
                    rep = Representation(kind, count, args.width, shift, profile.beta)
                    order = max(args.projection_quadrature, max(rep.counts)+4)
                    rep.project(profile, order)
                    result = measure(rep, profile, order*2, moments, absolute_moments)
                    # Refit and evaluate on finer quadrature to expose projection
                    # and derivative/source-moment quadrature sensitivity.
                    rep_fine = Representation(kind, count, args.width, shift, profile.beta)
                    rep_fine.project(profile, order*2)
                    fine = measure(rep_fine, profile, order*4, moments, absolute_moments)
                    l2_change = np.array(fine["l2_absolute_rho_momentum_energy_sigma"])-result["l2_absolute_rho_momentum_energy_sigma"]
                    source_change = (np.array(fine["broken_gradient_source_moments_0_1_2"])-result["broken_gradient_source_moments_0_1_2"])/absolute_moments
                    numerical_checks_pass = (rep_fine.rank == count and fine["finite"] and
                        fine["minimum_sampled_density"] > 0 and fine["minimum_sampled_pressure"] > 0 and
                        np.max(abs(l2_change)) < 1e-8 and np.max(abs(source_change)) < 1e-6)
                    records.append({"kind": kind, "coefficients_per_field": count,
                        "state_and_sigma_coefficient_count": 4*count,
                        "cusp_location_error_in_sqrt_alpha": shift,
                        "cusp_location_is_oracle": kind not in ("global_polynomial", "uniform_p3") and shift == 0,
                        "extra_geometry_parameters": 1 if kind in ("split_polynomial", "graded_p3", "graded_hp", "cusp_enriched") else 0,
                        "rank": rep_fine.rank, "scaled_design_condition": rep_fine.condition,
                        "numerical_checks_pass": bool(numerical_checks_pass),
                        "coarse_projection_quadrature": order, "fine_projection_quadrature": 2*order,
                        "local_polynomial_coefficients_per_field": rep.counts,
                        "nonpolynomial_enrichment_coefficients_per_field": 4 if kind == "cusp_enriched" else 0,
                        "smallest_element_width_in_sqrt_alpha": float(np.min(np.diff(rep.breaks))),
                        "breaks": rep.breaks, "result": fine,
                        "quadrature_refinement_l2_error_change": l2_change.tolist(),
                        "quadrature_refinement_source_moment_change": source_change.tolist()})
    script = Path(__file__).resolve()
    identity = subprocess.run(["git", "rev-parse", "HEAD"], cwd=script.parent, text=True, capture_output=True, check=True).stdout.strip()
    return {"schema": "igr-shock-representation-v1", "source_commit": identity,
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "configuration": vars(args), "reference_pass": bool(valid),
        "representation_numerical_check_thresholds": {"absolute_l2_error_quadrature_change": 1e-8,
            "source_moment_quadrature_change_over_absolute_reference": 1e-6,
            "require_full_rank_finite_values_positive_sampled_density_and_pressure": True},
        "scope": "Stationary full-energy ideal-gas IGR profile; best representation only. No PDE advancement, GPU work, speedup, dynamic stability, or novelty claim.",
        "accounting": "Equal independent coefficients per rho,momentum,energy,sigma field; adaptive representations additionally consume the reported cusp-location parameter. Uniform p3 control happens to put the cusp on an interior face. Graded meshes can impose severe explicit CFL restrictions; best approximation is not time-to-solution. All moments use broken reconstructed velocity gradients; no distributional jump-square model is introduced. Face jumps are reported separately.",
        "reference": {"paper": "https://arxiv.org/html/2607.12693v1", "gamma": profile.gamma,
            "upstream_mach": profile.mach, "alpha": 1, "rho_end_states": [profile.rho_lo, profile.rho_hi],
            "rho_sonic": profile.rho_s, "a": profile.a, "density_cusp_exponent": profile.beta,
            "sigma_leading_cusp_exponent": 2*profile.beta,
            "grid_and_quadrature_refinement_max_field_change": reference_change.tolist(),
            "source_moments": moments.tolist(), "absolute_source_moments": absolute_moments.tolist(),
            "adaptive_moment_quadrature_error_estimates": moment_quad_errors.tolist(),
            "source_moment_reference_refinement_change": ((coarse_moments-moments)/absolute_moments).tolist(),
            "independent_x_vs_density_source_moment_relative_change": ((moment_rule-moments)/absolute_moments).tolist(),
            "weak_closure_moment_residuals": weak,
            "maximum_relative_flux_invariant_error": np.max(np.abs(reference_flux-exact_flux)/abs(exact_flux), axis=0).tolist()},
        "representations": records, "cpu_analysis_seconds": time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--panels", type=int, default=2048)
    parser.add_argument("--profile-quadrature", type=int, default=32)
    parser.add_argument("--projection-quadrature", type=int, default=24)
    parser.add_argument("--reference-tolerance", type=float, default=2e-9)
    parser.add_argument("--width", type=float, default=2.0)
    parser.add_argument("--counts", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--location-errors", type=float, nargs="+", default=[0.0, 0.02, 0.1])
    args = parser.parse_args()
    if any(n < 8 or n % 8 for n in args.counts):
        parser.error("Each coefficient count must be a positive multiple of 8")
    if args.panels < 64 or args.profile_quadrature < 8 or args.projection_quadrature < 8:
        parser.error("Insufficient reference/projection resolution")
    if not math.isfinite(args.reference_tolerance) or args.reference_tolerance <= 0:
        parser.error("Reference tolerance must be positive and finite")
    if not math.isfinite(args.width) or not 0 < args.width <= 4 or any(not math.isfinite(x) or abs(x) >= args.width for x in args.location_errors):
        parser.error("Require 0 < width <= 4 and location errors inside the window")
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    all_valid = result["reference_pass"] and all(r["numerical_checks_pass"] for r in result["representations"])
    print(json.dumps({"output": str(output.resolve()), "reference_pass": result["reference_pass"],
                      "all_representation_numerical_checks_pass": all_valid,
                      "representation_count": len(result["representations"]), "cpu_analysis_seconds": result["cpu_analysis_seconds"]}))
    if not all_valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
