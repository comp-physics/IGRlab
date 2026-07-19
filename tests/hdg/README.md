# Scalar HDG CPU Oracle

This lane-local harness is the review boundary for `HDG-ORACLE-01`. It is a
transparent C++17 CPU reference for one-dimensional scalar Poisson and scalar
advection. It does not register a production backend and does not contain
Euler, IGR, wall thermodynamics, time integration, device code, packing, or a
performance path.

## Frozen formulation

On every segment `K`, the scalar interior space is the discontinuous MFEM
`L2_FECollection(p,1)` space `V_p(K)=P_p(K)`. The skeleton trace has one scalar
value per zero-dimensional face and is represented by the vertex restriction
of `H1_Trace_FECollection(1,1)`. This is the complete trace polynomial space in
one dimension. MFEM supplies curved isoparametric geometry, quadrature, basis
values and derivatives, element/vertex incidence, segment DOF orientation,
parallel partitioning, and dense LU factorization.

The solve trace is zero-dimensional, so its in-mesh orientation is identity. To
freeze the nontrivial reusable skeleton rule, the p1-p4 permutation fixture uses
MFEM `DG_Interface_FECollection(p,2)` on segment faces and checks `s <-> 1-s`
against MFEM orientation `-1`; the local Schur fixture separately reverses the
interior segment and swaps its two trace slots.

For Poisson, `q=-kappa grad(u)` and `div(q)=f`, with `kappa=1`. The local mixed
HDG equations are

```text
(kappa^-1 q,r)_K - (u,grad r)_K + <uhat,r n>_dK = 0,
-(q,grad v)_K + <qhat n,v>_dK = (f,v)_K,
qhat n = q n + tau (u-uhat),
tau = 4 (p+1)^2 / h_K.
```

The trace is prescribed strongly at both Dirichlet endpoints. At each interior
face, the sum of the two outward numerical fluxes is zero.

For positive constant advection speed `beta=1.3`, the local equation and
upwind flux are

```text
-(beta u,grad v)_K + <fhat n,v>_dK = (f,v)_K,
fhat n = beta n uhat + tau_n (u-uhat),
tau_n = max(beta n,0).
```

The inflow trace is essential. Interior traces enforce conservative flux
balance. The outflow flux is natural and the reported outflow trace is the
recovered interior limit; it is not an extra global equation.

With element interior vector `x_K` and local trace vector `lambda_K`, both
forms assemble the explicit residual and constant Jacobian

```text
R_K = A_K x_K + B_K lambda_K - f_K,
G_K = C_K x_K + D_K lambda_K.
```

Local MFEM LU solves give

```text
x_K = A_K^-1 (f_K - B_K lambda_K),
S_K = D_K - C_K A_K^-1 B_K,
b_K = -C_K A_K^-1 f_K,
```

after the stated essential-trace elimination. Global skeleton assembly sums
`S_K` and `b_K`. Recovery is the same stored local solve after the trace
solution. The tests compare this route with a separately assembled monolithic
interior-plus-trace MFEM dense direct solve.

## MFEM hybridization boundary

`mfem::Hybridization` hybridizes an assembled conforming finite-element system
through a constraint matrix, producing `C Ahat^-1 C^T`. That is genuine and is
the right reference for the H(div) constraint semantics in MFEM example 4, but
it is not this HDG method's independently defined numerical trace and flux
system. `mfem::StaticCondensation` likewise follows the finite-element private
DOF split. Using either class here would silently change the formulation.
Therefore this oracle uses the same MFEM v4.9 local LU/static-condensation
algebra explicitly, exposes every `A/B/C/D` block, and compares Schur action,
reduced RHS, trace solve, and recovery against the monolithic MFEM solve. No
claim is made that MFEM supplies native transient nonlinear Euler HDG.

## Verification and tolerances

`run_oracle.sh` requires a unique run directory. It configures and builds only
`tests/hdg` against the immutable CPU MFEM v4.9 install at commit
`d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec`. The compile rejects CUDA, HIP,
RAJA, or Umpire-enabled MFEM. Serial checks run before MPI. The one/two-rank
checks acquire `/tmp/dg_igr-sm120.lock` nonblockingly and exit 75 if busy.

The MFEM direct-solve residual tolerance is `2e-12`, tighter than the general
comparison tolerance `2e-10`. Failed results are still written before the
program exits nonzero. Outputs are machine-readable `serial_checks.json`,
`mpi_{1,2}_checks.json`, and a p1-p4 affine/curved `convergence.csv`.

The deterministic suite independently freezes:

- finite-difference residual/Jacobian agreement;
- monolithic versus condensed assembly, reduced RHS, trace solve, Schur action,
  and local back-substitution;
- MFEM's p1-p4 `DG_Interface_FECollection` trace permutation and
  reversed local Schur algebra;
- strong essential-trace treatment, local balance, and shared-face
  conservation;
- scalar Poisson/advection p1-p4 affine/curved convergence and algebraically
  exact local recovery;
- MFEM `ParMesh` partition counts, lower-element trace ownership, reduced
  matrix/RHS exchange, and owner-to-ghost trace exchange on one and two ranks.

## Later integration request

The coordinator should append `add_subdirectory(hdg)` to `tests/CMakeLists.txt`;
the root file already adds `tests/` under `BUILD_TESTING AND DG_IGR_BUILD_TESTS`.
Register serial, one-rank, and two-rank invocations of the resulting
`hdg_oracle` target with unique binary-directory outputs. The MPI registrations
must call through a nonblocking `/tmp/dg_igr-sm120.lock` wrapper, treat exit 75
as a yielded/skipped lease, and run serially. The existing `run_oracle.sh` is
the exact lane-local command sequence to preserve in CI.
