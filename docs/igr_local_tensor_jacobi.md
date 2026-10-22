# Local tensor Jacobi feasibility prototype

This opt-in experiment keeps the existing two-sweep ordinary-DG closure and
replaces repeated dense element assembly/LU with a small tensor representation
and current density scaling. It is a separate candidate from both exact-current
block Jacobi and the unstable frozen-initial-inverse trial. Native CUDA validation
and large-case results are pending. No speedup is claimed from host algebra checks.

Select `--igr-solve-mode element-block-jacobi --igr-jacobi-blocks tensor`, with
`--igr-sweeps 2 --igr-initial-sweeps 50 --igr-jacobi-omega 0.7`. Legacy defaults
and current/reference block policies remain unchanged. The intended comparison
is the same512²,p5 ordinary-DG Sedov case through t=0.1, against optimized
source97ba1681, with both optional runtime scans and both limiters disabled.

At the first production state, assemble the exact current element diagonal D0,
including volume mass/diffusion and exact SIPG face self terms. With the actual
collocated physical mass diagonal M0, normalize

    H0 = M0^(-1/2) D0 M0^(-1/2).

Admit H0 only when it is symmetric and its complete matrix reconstructs as a
Kronecker sum Hx tensor I + I tensor Hy within1e-11 relative max-entry scale.
Diagonalize the small one-dimensional axes once; their individual gauge-shifted
spectra may be indefinite, but every combined eigenvalue must be positive and
resolved. Rounded hashes only locate candidate factor classes: every reuse
checks the complete normalized element matrix. At most512 verified factor
classes are admitted. Dense EA, D0 and setup-only permutation storage are freed
after factorization.

Every stage refreshes M from actual current inverse-density quadrature values
and fixed physical geometry weights. The local correction is

    P_current r = M_current^(-1/2) Q
                  diag(1/(lambda_x + lambda_y)) Q^T
                  M_current^(-1/2) r,

where Q is the product of the two small axis eigenbases. A fused native CUDA
kernel performs four FP64 tensor contractions, diagonal scaling, and native
gather/scatter. This implementation is SIMT; it does not claim tensor-core or
TF32 execution. The savings being tested come from removing recurring dense
setup/triangular solves and reducing stored matrix data.

A globally uniform density change is reproduced by the exact current local
inverse up to floating-point factorization error. Heterogeneous density instead
uses an explicitly approximate SPD mass-scaled reference operator. Current A and
b are still rebuilt from the current physical state; two sweeps do not certify
their residual. No hidden coarse correction, converged cleanup or fallback is
used. Scaling adds no per-stage host coefficient copy, finite scan or reduction.

The bounded backend admits one CUDA rank, uniform scalar GL nodal L2 quads with
identity true/local layout, collocated mass quadrature and D1D<=8. This version
requires fixed affine Cartesian geometry with positive diagonal Jacobians,
verified at every Q1 corner and quadrature point. Initial coefficients must also
produce separable normalized blocks. It does not establish a curved/non-Cartesian
solver. Geometry/topology and alpha policy must stay fixed. The existing history
contract prevents initial output-only/discarded prefetched states from choosing
the production reference; later output does not commit production history.

Native qualification includes assembled-MFEM exact-block and independent dense
surrogate oracles atp2/p5, non-unit cell sizes, boundary/interior classes, changed
uniform density rho=.45, heterogeneous density, SPD/symmetry, return-to-prior
density, no retained dense matrices, and explicit admission rejections. Existing
parser, current/reference lifecycle and CUDA Krylov suites remain required.
Host-only algebra checks are not substitutes for these native tests.
