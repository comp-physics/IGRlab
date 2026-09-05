# Artificial-time relaxation for the current screened IGR operator

This note derives an inexpensive falsification test for a local artificial-time solver. The accompanying `tests/research/igr_relaxation_probe.py` applies it to exported, frozen production IGR matrices. It does not advance a new physical model or implement a new first-order hyperbolic DG system. The recurrence is the established heavy-ball method for a positive quadratic, interpreted as a centered damped wave. [Polyak's original iteration paper](https://doi.org/10.1016/0041-5553(64)90137-5) is prior art. A separate [hyperbolic DG Euler–Poisson method](https://arxiv.org/abs/2008.10593) demonstrates that an elliptic solve can share a hyperbolic execution engine; it does not establish IGR performance.

## 1. A priori energy, damping and finite propagation

Freeze physical time and the current density and alpha. Write the current symmetric coercive DG closure as

\[
A\sigma=b,\qquad A=R+K,\qquad R_{ij}=\int\rho^{-1}\phi_i\phi_j.
\]

`K` includes diffusion, consistency and penalty faces. Its positive semidefiniteness is an assumption requiring a coercive SIPG choice; the small frozen probes check it independently rather than inferring it from the positive continuum coefficient. Natural boundaries leave the constant diffusion mode, but reaction makes `A` positive definite. A nonsymmetric closure discretization needs different analysis and is rejected by this probe.

For any **element-block diagonal, symmetric positive definite** metric `B`, solve in artificial time `s`

\[
B\sigma_{ss}+2\eta B\sigma_s+A\sigma=b,\qquad \eta>0.
\]

Its equilibrium is precisely the current closure, for every choice of `B`. Defining `e=σ−A⁻¹b` gives the energy identity

\[
\mathcal E=\tfrac12\|e_s\|_B^2+\tfrac12\|e\|_A^2,
\qquad \frac{d\mathcal E}{ds}=-2\eta\|e_s\|_B^2.
\]

In the symmetric coordinates `B^(1/2)e`, a mode of `B⁻¹A` with eigenvalue `λ` obeys `z″+2ηz′+λz=0`. If its spectrum lies in `[μ,L]`, the slowest asymptotic exponent is `η` for `η≤√μ`, and `η−√(η²−μ)` for `η>√μ`. Thus continuous-time optimal worst-mode damping is `η=√μ`. The critical mode has a linear prefactor in time: an exponential alone is not a uniform transient bound.

Choosing `B=R` is particularly transparent at continuum level:

\[
\rho^{-1}\sigma_{ss}+2\eta\rho^{-1}\sigma_s
 +\rho^{-1}\sigma-\nabla\!\cdot(\alpha\rho^{-1}\nabla\sigma)=f.
\]

With coefficients frozen in physical time, its principal artificial-wave speed is `√α`; density cancels from the ratio of principal stiffness and inertia. Spatially varying coefficients, boundaries and discrete penalty factors still affect conditioning. At a finite artificial horizon, the continuum wave has finite propagation; its converged equilibrium retains the original elliptic tail. Confining the artificial evolution is therefore a controlled incomplete solve, not exact compact support of the instantaneous inverse.

## 2. Stable explicit recurrence and the unavoidable local work

Use centered acceleration and damping with artificial step `δ`:

\[
B\frac{x_{k+1}-2x_k+x_{k-1}}{\delta^2}
 +\eta B\frac{x_{k+1}-x_{k-1}}{\delta}+Ax_k=b.
\]

Because `B⁻¹` acts inside each element, this is explicitly local:

\[
x_{k+1}=x_k+\beta(x_k-x_{k-1})+hB^{-1}(b-Ax_k),
\quad h=\frac{\delta^2}{1+\eta\delta},
\quad\beta=\frac{1-\eta\delta}{1+\eta\delta}.
\]

Here `h` denotes an iteration coefficient, not mesh size. Each eigenmode has characteristic polynomial `r²−(1+β−hλ)r+β`. The strict stability condition is `δ²L<4` with positive damping. Minimizing the worst spectral radius over the frozen interval gives

\[
h=\frac4{(\sqrt L+\sqrt\mu)^2},\quad
\beta=\left(\frac{\sqrt L-\sqrt\mu}{\sqrt L+\sqrt\mu}\right)^2,
\quad q=\frac{\sqrt\kappa-1}{\sqrt\kappa+1}.
\]

These correspond to `δ=2/√(L+μ)` and `η=√(Lμ/(L+μ))`. Initializing `x₋₁=x₀`, the degree-`k` error polynomial is `q^k[U_k(t)−q U_(k−1)(t)]`, where `t=(1+q²−hλ)/(2q)∈[−1,1]`. Consequently

\[
\|e_k\|_A\le (k+1+kq)q^k\|e_0\|_A.
\]

The same polynomial bound applies to the residual in the `B⁻¹` norm. Euclidean residuals can show transients. It is incorrect to promise `q^k` contraction from the first sweep. If `L=μ`, the method solves in one step and the limiting formula is interpreted directly.

Every sweep costs one current operator action, one local metric solve and a few vector operations. A fixed batch requires no inner global dot products. Residual certification does require a reduction; checking every sweep forfeits that communication advantage. The probe charges its checks explicitly and supports batched checks. With nearest-face `A` and an element-local inverse, a perturbation introduced through the new RHS affects at most `k−1` face hops after `k` sweeps from a zero correction. A previous global solution does not become compactly supported: only the new correction has this propagation bound.

For affine elements and a standard polynomial inverse estimate, reaction-mass scaling gives the schematic bound

\[
\kappa\lesssim 1+C_{\rm geom,\rho,penalty}\,
 \alpha\,(p+1)^4/h_K^2.
\]

When `α=C h_K²`, the global mesh count need not control the iteration count, but high polynomial order still does: roughly `O(√C (p+1)² log(1/ε))` actions, with a further logarithmic/transient factor from the bound above. This predicts why simply changing diffusion into artificial waves need not beat a good local preconditioner. Replacing `B` with the principal element block of `A` absorbs more internal stiffness; its remaining spectrum must be assessed, since weakly coupled interelement modes and penalties can still deteriorate with `p`.

## 3. Bounds that do not require a global eigenproblem

The small probe computes exact generalized eigenvalues only as a charged calibration oracle. It also constructs local bounds. If `A≥R`, then

\[
\mu\ge\min_K\lambda_{\min}(R_K,B_K),\qquad
L\le\|B^{-1}A\|_\infty.
\]

The first calculation factors only element matrices. The second applies those factors to occupied local/face columns and takes the maximum absolute row sum. The maximum and minimum require setup reductions but no global dense matrix factorization. A sparse matrix can have poor row-sum bounds even when its actual spectrum is favorable; the conservative and oracle runs quantify this gap. Large-case lower bounds remain conditional on coercivity if no global audit is performed. A converged current residual is direct evidence about that run, not a proof that all evolving states share the bound.

The probe preserves the actual sampled operator in every action. A tiny measured asymmetry is only symmetrized for the eigenvalue audit. Reference solutions use a separate sparse FP64 direct solve and must pass the current-operator residual gate. Sampling/export, reference factorization, local factorization, bound construction, eigenvalue audits, iterations and certifications are reported separately. These CPU costs are not GPU performance estimates.

## 4. Warm starts, evolving density and accuracy budgets

For an exact previous closure `A₀σ₀=b₀`, the residual at the next frozen state is exactly

\[
r_0=b_1-A_1\sigma_0
=(b_1-b_0)-(A_1-A_0)\sigma_0.
\]

Thus screening and a small change in both density-dependent coefficients and source can reduce the required correction dramatically. It is insufficient to examine `b₁−b₀` alone. The probe can warm-start from a separately exported previous full-energy state, retaining the complete current `A₁` in all residuals; the previous solve is identified and charged independently. Controlled field perturbations are not mislabeled as an advancing application trajectory.

For any approximate closure with residual `r`, `e=A⁻¹r` satisfies `||e||_A≤||r||_(B⁻¹)/√μ`. In the physical mass norm, a corresponding coercivity bound gives `||e||_M≤||r||_(M⁻¹)/μ_M`. Turning this into transport accuracy additionally requires bounding the momentum gradient/face lifting of `e` and energy work with velocity. Those can amplify errors with `p/h`; a small scalar closure norm alone is not a shock-accuracy certificate. Under an appropriate stable Lipschitz evolution estimate, maintaining order `q` needs the closure-induced RHS error at most `O(Δt^q)` across a fixed horizon. Shock admissibility and quantities of interest still require application tests.

## 5. Meaning of the screen

This route can remove repeated Krylov reductions without changing the converged instantaneous target. Its frozen SPD mathematics gives no better asymptotic iteration rate than a good polynomial solver; heavy-ball and Chebyshev are close competitors on the same interval. It is worth a separate prototype because an artificial-wave state may support useful fusion or a later first-order DG implementation. A single local inverse applied to one vector is still a narrow matrix operation, so success as an ordinary CUDA solver is not a tensor-core result. Simultaneous stage right-hand sides, on-chip coefficient evaluation and reuse must be assessed with their additional work charged.

The same modal calculation gives a mixed-precision caution that can be tested before writing a tensor kernel. If each recurrence introduces a bounded additive disturbance `d_k` in the symmetric coordinates, its impulse response is bounded by `(j+1)q^j`. Hence a uniform disturbance of size `ε` can accumulate to at most `ε/(1−q)²`, a conservative `O(κ ε)` bound. This is not a claim that FP32 necessarily fails; it explains why a current higher-precision residual/correction is necessary when conditioning and the requested error budget demand it. Strong cancellation and local factor scaling determine the actual perturbation size.

## Initial frozen-production measurements

The first screen uses the application's current exact-face L2 operator on a 16-element p3 prescribed full-energy tanh state, and the actual saved p1 cylinder state at physical time 0.5 on 1,322 curved elements. It is a frozen solver comparison, not a new complete time-to-solution measurement. At p3, the audited reaction-mass spectrum is `[1,4438.49]`; principal element scaling changes it to `[0.00477548,1.99522]`. `A−R` passes the small global semidefinite audit to roundoff.

At relative current residual `1e-8`, p3 reaction-mass relaxation fails the 512-sweep cap even with exact spectral parameters (residual `1.86e-6`), while same-metric PCG needs 46 iterations. Exact-spectrum element relaxation needs 224 sweeps versus 76 PCG iterations. Its cheap local lower bound is much more pessimistic than the true minimum eigenvalue and the conservative recurrence fails 512 sweeps at residual `1.07e-4`. These failures support stopping this unfactored high-p route unless a spatial elimination or materially different execution schedule changes its cost.

The p3/p5/p7 sweep makes the scaling visible: reaction-mass condition numbers are `4438, 21233, 65258`, close to `16(p+1)^4`; principal-element condition numbers are `418, 941, 1674`, close to `26.1(p+1)^2`. Exact-spectrum element recurrence needs `224, 336, 456` sweeps, nearly linear in `p+1`, versus `76, 113, 145` same-metric PCG iterations. This is measured scaling on one family, not a uniform theorem for general meshes. The local conservative spectral enclosure deteriorates considerably faster than the actual element-scaled spectrum.

The cylinder's reaction scaling is more competitive. Conditional local bounds `[1,362.40]` give 200 sweeps to `1e-8`, versus 175 same-metric PCG iterations; element scaling needs 216 sweeps versus 101 PCG iterations. The unweighted physical mass is a poor metric on this graded, variable-density case and both methods exhaust 512. Checks every eight sweeps require 27 solver norm reductions for the reaction recurrence, including the initial RHS norm, while the simple PCG implementation accounts for 529 unfused scalar reductions. This is an opportunity to trade additional local work for fewer reductions, not a measured GPU improvement; production Chebyshev-preconditioned CG and fused reductions are stronger controls than this CPU work-count baseline.

At the looser current residual target `1e-3`, reaction and element recurrence both take 88 sweeps. Their relative physical sigma L2 errors are `4.92e-4` and `3.36e-4`; respective PCG controls take 73 and 40 iterations with errors `1.50e-4` and `2.32e-4`. Equal residual targets do not imply identical physical error. Full source hashes, setup times, actual action counts, per-check errors and conditional-bound status are recorded in `/tmp/igr-local-four/relaxation-*.json`.

For a controlled perturbation multiplying all saved conservative fields by `1+0.001 sin(3x)cos(2y)`, the previous cylinder closure has current relative residual `4.74e-4`. At tolerance `1e-3`, both methods correctly need zero corrections. At `1e-8`, reaction recurrence drops from 200 to 120 sweeps and PCG from 175 to 107; element recurrence drops from 216 to 128 and PCG from 101 to 60. This pair is a perturbed realistic state, not two accepted time steps. It illustrates both the value and the limit of warm starting: few updates are plausible at a transport-appropriate tolerance, but not guaranteed at an arbitrarily tight closure tolerance.

The finite-wave-horizon response derived in [the compact-response note](igr_compact_response_derivation.md) supplies another combination: integrate a local wave for a finite horizon to approximate the inverse with a known spectral error. Its explicit evaluation encounters the same high-frequency stability limit. Exact elimination of interior modes or a better local metric should therefore precede an expensive GPU implementation of either wave route. None of these measurements rules out a different first-order hyperbolic discretization with different auxiliary-variable relaxation or local time stepping; those would need their own energy and cost analysis.

Reproduce the independent modal/support checks with:

```bash
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_relaxation_probe.py --self-test
```

Given a frozen export and an optional earlier export:

```bash
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_relaxation_probe.py \
  --input /tmp/current-igr.json --previous /tmp/previous-igr.json \
  --metrics mass,reaction,element --tolerance 1e-8 \
  --max-iterations 3000 --checkpoint 1 --output /tmp/igr-relaxation.json
```

JSON entries distinguish exact-spectrum calibration, local conditional bounds, actual residual/physical error, successful completion, local matrix work, operator applications and reduction counts. Passing the Python process alone is not a convergence certificate; inspect each run's `complete` field.
