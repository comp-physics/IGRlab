# Scalar HDG CPU Oracle

This lane-local harness is the review boundary for `HDG-ORACLE-01`. It is a
transparent C++17 CPU reference for one-dimensional scalar Poisson and scalar
advection. It does not register a production backend and does not contain
Euler, IGR, wall thermodynamics, time integration, device code, packing, or a
performance path.

## Frozen formulation

On every segment `K`, the scalar interior space is the discontinuous MFEM
`L2_FECollection(p,1)` space `V_p(K)=P_p(K)`. The operational skeleton trace
has one scalar value per zero-dimensional face and is represented by the
vertex restriction of `H1_Trace_FECollection(1,1)`. This is the complete trace
polynomial space in one dimension. MFEM supplies curved isoparametric geometry,
quadrature, basis values and derivatives, element/vertex incidence, trace
ownership, parallel restriction/prolongation, and local dense factorization.

The operational one-dimensional endpoint trace has identity in-mesh
orientation. The nontrivial reusable orientation rule is tested in a coupled
p1-p4 two-dimensional skeleton fixture. It builds a two-quadrilateral mesh,
uses `DG_Interface_FECollection(p,2)` face DOFs and
`FaceElementTransformations`, assembles interior mass/diffusion and face
couplings, forms `D-B^T A^-1 B`, assembles the global trace operator, reverses
the shared-face parameter, and compares through the actual MFEM orientation
`-1` permutation. This couples orientation to assembly and condensation; it is
not a standalone shape-permutation assertion.

For Poisson, `q=-kappa grad(u)` and `div(q)=f`, with `kappa=1`. The local mixed
HDG equations are

```text
(kappa^-1 q,r)_K - (u,grad r)_K + <uhat,r n>_dK = 0,
-(q,grad v)_K + <qhat n,v>_dK = (f,v)_K,
qhat n = q n + tau (u-uhat),
tau = 4 (p+1)^2 / h_K.
```

The trace is prescribed strongly at both Dirichlet endpoints. At every
interior face, the two outward numerical fluxes sum to zero.

For positive constant advection speed `beta=1.3`, the local equation and
upwind flux are

```text
-(beta u,grad v)_K + <fhat n,v>_dK = (f,v)_K,
fhat n = beta n uhat + tau_n (u-uhat),
tau_n = max(beta n,0).
```

The inflow trace is essential. Interior traces enforce conservative flux
balance. The outflow flux is natural and the reported outflow trace is the
recovered interior limit, not an extra global equation.

With element interior vector `x_K` and local trace vector `lambda_K`, both
forms assemble the explicit residual and constant Jacobian

```text
R_K = A_K x_K + B_K lambda_K - f_K,
G_K = C_K x_K + D_K lambda_K.
```

Local MFEM dense solves give

```text
x_K = A_K^-1 (f_K - B_K lambda_K),
S_K = D_K - C_K A_K^-1 B_K,
b_K = -C_K A_K^-1 f_K,
```

after essential-trace elimination. Global skeleton assembly sums `S_K` and
`b_K`. Recovery repeats the stored local solve after the trace solution. The
suite compares this route with a separately assembled monolithic
interior-plus-trace dense direct solve.

## MFEM hybridization boundary

`mfem::Hybridization` hybridizes an assembled conforming finite-element system
through a constraint matrix, producing `C Ahat^-1 C^T`. That is genuine MFEM
H(div) constraint semantics, but it is not this HDG method with an
independently defined numerical trace and flux system.
`mfem::StaticCondensation` follows the finite-element private-DOF split and is
also not substituted here. The oracle instead uses MFEM v4.9 geometry, spaces,
orientation, integrators, parallel maps, and local factorization where the
semantics match, while keeping every `A/B/C/D` block explicit. It compares the
explicit local/trace algebra to monolithic solves and MFEM operator-level
references. No claim is made that MFEM supplies native transient nonlinear
Euler HDG.

## Independent operator and parallel references

The interior residual and the recovered skeleton residual
`C*x(lambda)+D*lambda` are both central-finite-differenced. Source vectors,
Poisson mass and weak-derivative blocks, and the advection convection block are
also assembled independently with MFEM integrators. These references do not
reuse the monolithic/condensed block-comparison path.

For MPI, the fixture uses an actual MFEM `ParMesh` and
`ParFiniteElementSpace`. Every local trace DOF is recorded with
`GetGlobalTDofNumber`, `GetLocalTDofNumber`, its MFEM owner, and owned/ghost
state. Only owned entries seed a `HypreParVector`; `ParallelProject` and
`ParGridFunction::Distribute` perform owner-to-ghost exchange from true DOFs.
Element-local HDG Schur blocks are inserted in a rank-local `SparseMatrix` and
`ParBilinearForm::ParallelAssemble` constructs `P^T S_local P`. The reduced
right side and a separately evaluated local skeleton residual are reduced by
`P^T`; their true-DOF residuals are compared. MPI all-reductions carry only
scalar errors/counts, while `Gather/Gatherv` records metadata for JSON. No
replicated dense global operator or full-vector all-reduce is used as the
parallel oracle.

## Verification and tolerances

`run_oracle.sh` rejects an existing run path before creating any file. It
configures and builds only `tests/hdg` against the immutable CPU MFEM v4.9
install at commit `d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec`. Compilation
rejects CUDA, HIP, RAJA, or Umpire-enabled MFEM. Serial runs before MPI. The
one-/two-rank checks acquire `/tmp/dg_igr-sm120.lock` nonblockingly and exit 75
if busy.

The local direct-solve residual tolerance is `2e-12`, tighter than the general
comparison tolerance `2e-10`. The finite-difference Jacobian tolerance is
`2e-7`; p1-p4 recovery tolerance is `5e-10`. Failed results are retained in
unique evidence run directories. Machine-readable outputs are
`serial_checks.json`, `mpi_{1,2}_checks.json`, and p1-p4 affine/curved
`convergence.csv`.

Every convergence combination rejects nonfinite quantities, requires all four
errors to decrease strictly, and gates on the final refinement rate at
`p+0.65`; JSON records the observed final rate, threshold, margin, finite flag,
monotonic flag, and pass result.

The deterministic suite freezes:

- interior and skeleton finite-difference residual/Jacobian agreement;
- independent MFEM source, mass, weak-derivative, and convection operators;
- monolithic versus condensed assembly, reduced RHS, trace solve, Schur action,
  and local back-substitution;
- p1-p4 MFEM oriented skeleton assembly and condensation;
- strong essential traces, local balance, and shared-face conservation;
- scalar Poisson/advection p1-p4 affine/curved final-rate convergence and exact
  local recovery;
- one-/two-rank MFEM true-DOF ownership maps, owner/ghost exchange, and
  `P^T S_local P` agreement with the direct skeleton residual.

## Later integration request

The coordinator should add `add_subdirectory(hdg)` to `tests/CMakeLists.txt`;
the root file already adds `tests/` under `BUILD_TESTING AND
DG_IGR_BUILD_TESTS`. Register serial, one-rank, and two-rank invocations of the
lane target with unique binary-directory outputs. MPI registration must use a
nonblocking `/tmp/dg_igr-sm120.lock`, treat exit 75 as a yielded/skipped lease,
and serialize the MPI jobs. The lane-local `run_oracle.sh` is the exact command
sequence to preserve in CI. No shared CMake edit belongs to this lane commit.
