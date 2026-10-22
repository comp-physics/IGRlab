# Local tensor Jacobi prototype and measured Sedov result

The optimized two-sweep closure completed the fixed 512×512, p5 ordinary-DG
Sedov comparison through t=0.1. Whole-application time fell from 667.666 s to
503.380 s: **1.326× faster, or 24.606% less time**, in one sequential pair on
the same DeltaAI GPU. The maximum conserved-field relative L2 difference from
the optimized converged control was 1.077e-6; sigma differed by 2.035% in L2.
This establishes a useful approximation on this case, not a general convergence,
stability or scaling result. The prototype remains opt-in.

## Method

Select `--igr-solve-mode element-block-jacobi --igr-jacobi-blocks tensor`, with
`--igr-sweeps 2 --igr-initial-sweeps 50 --igr-jacobi-omega 0.7`. Each regular
stage starts from the existing sigma history. Defaults and the current/reference
block policies are unchanged. This candidate differs from both exact-current
dense block Jacobi and the earlier unstable frozen-initial-inverse trial.

At the first production state, assemble the exact element diagonal D0,
including volume mass/diffusion and exact SIPG face self terms. With the actual
collocated physical mass diagonal M0, normalize

    H0 = M0^(-1/2) D0 M0^(-1/2).

Admit H0 only when it is symmetric and its complete matrix reconstructs as a
Kronecker sum Hx tensor I + I tensor Hy within 1e-11 relative max-entry scale.
Diagonalize the small one-dimensional axes once; their individual gauge-shifted
spectra may be indefinite, but every combined eigenvalue must be positive and
resolved. Rounded hashes only locate candidate factor classes: every reuse
checks the complete normalized element matrix. At most 512 verified factor
classes are admitted. Dense EA, D0 and setup-only permutation storage are freed
after factorization.

Every stage refreshes M from current inverse-density quadrature values and
fixed physical geometry weights. The local correction is

    P_current r = M_current^(-1/2) Q
                  diag(1/(lambda_x + lambda_y)) Q^T
                  M_current^(-1/2) r,

where Q is the product of the two small axis eigenbases. A fused native CUDA
kernel performs four FP64 tensor contractions, diagonal scaling and native
gather/scatter. **This local inverse uses FP64 SIMT, not TF32 tensor cores.**
The structural savings remove repeated dense assembly/LU and triangular solves;
the two-sweep approximation also removes converged Krylov work.

A globally uniform density change reproduces the exact current local inverse
up to floating-point factorization error. Heterogeneous density instead uses an
approximate SPD mass-scaled reference operator. Current A and b remain FP64 and
are refreshed from the current physical state; two sweeps do not certify their
residual. There is no coarse correction, converged cleanup or fallback. Updating
the mass scaling adds no per-stage host coefficient copy, finite scan or reduction.

## Qualification and supported geometry

The backend admits one CUDA rank, uniform scalar GL nodal L2 quads with identity
true/local layout, collocated mass quadrature and D1D <= 8. It requires fixed
affine Cartesian Q1 geometry with positive diagonal Jacobians, verified at every
corner and quadrature point. Initial coefficients must produce separable
normalized blocks. Geometry/topology and alpha policy must remain fixed.
This version does not support curved or general non-Cartesian elements.

Initial output-only/discarded prefetched states cannot seed the production
reference; later output does not commit production history. Native qualification
job 3436508 passed all seven focused checks on source
`52d46caf68d2a3a928cb500b656779f34a2c02d1`, built using canonical `build.sh` on
DeltaAI. These cover independent assembled-MFEM/dense-inverse oracles at p2/p5,
non-unit cells, reordered elements, boundary/interior classes, changed uniform
density, heterogeneous-density surrogate correctness and distinction from exact
current blocks, SPD/symmetry, return-to-prior density, no retained dense matrices,
geometry/separability rejection, lifecycle, parser defaults and CUDA Krylov
regressions. The earlier fixture-ordering failure was repaired by constructing
the standalone fixture after MFEM initialized its geometry caches; production
topology guards were retained.

Short admission job 3436621 passed complete saved-field checks through t=.0025.
Its output-heavy short totals were not used as the full-run performance claim.

## Matched full comparison

Job **3436707**, node **gh115**, ran the control and candidate sequentially on
the same GPU UUID. Both reached t=.1 with **4,939 accepted steps**, zero rejected
attempts and 14,818 RHS/sigma calls. The fixed mesh contains 262,144 cells,
37,748,736 Euler DOFs and 9,437,184 sigma DOFs. Settings were ordinary DG/GL p5,
L2/BrokenL2, exact SIPG faces, SSPRK3, CFL .3, max-h c_alpha=1/6
(physical alpha=6.357828776041666e-7), and VTK output every .02. Optional NaN and
admissibility scans, positivity limiting and modal damping were all disabled.

The control is the optimized converged source
`97ba1681ac27e07145df790d4c23ed837a21caea`, not the original GPU baseline.
It uses the TF32 spectral preconditioner with current FP64 A/RHS and
RHS-relative tolerance 1e-6. All 14,818 independent current-equation FP64
certificates passed; the largest residual ratio was 9.9954522e-7.
The candidate performed 29,684 fixed sweeps with no residual certificates.
Both retained the same optimized FP32 ordinary-DG Euler route.

| Measured scope | Converged control | Local tensor Jacobi | Time reduction |
| --- | ---: | ---: | ---: |
| Whole application, including setup/output | 667.666 s | 503.380 s | 24.606% |
| Native loop stopwatch, including loop output | 587.385 s | 425.386 s | 27.580% |
| Inclusive sigma host update | 348.434 s | 188.155 s | 46.000% |
| Solve host counter | 231.472 s | 78.677 s | 66.010% |

The solve counter is contained in the sigma update scope; do not add them.
Assembly, form-system and preconditioner-setup host counters were respectively
92.729/13.407/3.224 s for the control and 92.430/13.228/3.448 s for the candidate.
These counters include initial sigma preparation before the native loop timer
starts and are not isolated GPU timings. They do not establish an Euler-only or
sigma-to-Euler cost ratio; no new Nsight capture was needed for this bounded trial.

The tensor backend reported nine verified factor classes, reconstruction error
1.3566446e-13, no dense blocks retained after setup, and 189,800,032 logical
persistent bytes (181.007 MiB). That is not total or peak application memory.
Sampled application GPU peaks were 17,840 MiB for the control and 18,310 MiB for
the candidate, including setup; the one-second sampler can miss shorter peaks.

## Final-field comparison

Complete saved Q5 polynomials were reconstructed and compared at GL6×6 points
in every element, not only cell averages. Conserved/sigma L2 and integral
comparisons are exact for those saved polynomials up to roundoff; Linf values
are sampled. Pressure and speed are derived at the GL points.

| Field | Relative L2 difference | Sampled relative Linf difference |
| --- | ---: | ---: |
| Density | 2.2759e-7 | 7.4248e-7 |
| Momentum x | 1.0769e-6 | 1.9817e-6 |
| Momentum y | 1.0769e-6 | 1.9810e-6 |
| Total energy | 3.4257e-7 | 8.9972e-7 |
| Pressure | 3.0446e-7 | 8.3979e-7 |
| Sigma | 2.0348e-2 | 2.6584e-2 |

All conserved components passed the predeclared engineering comparison budgets
of 1e-3 relative L2 and 1e-2 sampled relative Linf. Sigma had no acceptance
budget and remains deliberately approximate. All 9,437,184 final GL points per
arm were finite with positive density and pressure; candidate minima were
rho=.4494201808 and p=1.0003187645. Direct differences in saved conserved
integrals were at roundoff scale. These checks establish neither positivity
between GL nodes/at unsaved times nor a fraction of the discretization error.

## Interpretation and reproducibility

Repeated dense local factorization was an implementation bottleneck, not a
reason to rule out fixed-sweep block Jacobi. This structured approximation
retained the small flow-field differences and beat the optimized converged
control on this case. Its current-density scaling also completed the full
interval where the previous frozen-initial-inverse experiment failed.

This is one control-first pair with unlocked GPU clocks, not a confidence
interval or an asymptotic performance result. Active-sample median SM clocks
were 1950/1980 MHz; memory clocks matched at 2619 MHz. No competing GPU
process was sampled, and the host heartbeat recorded no gap over 1.5 s. Identical output schedules and
between-output loop windows support a solver saving rather than an I/O-only
explanation. Independent review found no source/configuration/timing blocker.

Exact arguments/environment, source and binary identities, field hashes and
measured counters are recorded in
[the case specification](benchmarks/dg_local_tensor_jacobi_case.json) and
[the measurement receipt](benchmarks/dg_local_tensor_jacobi_measurements.json).
Full logs, compressed native fields and inspected density/pressure/sigma maps
and radial plots are preserved with the external experiment artifacts named
there. The tested application source remains `52d46caf`; subsequent result-only
commits do not change that identity.
