# HDG-ORACLE-01 gate evidence

Final verification command:

```text
tests/hdg/run_oracle.sh /tmp/dg_igr-hdg-oracle-0214-remediation-final06
```

This command created a unique isolated CPU build/output/log tree, ran serial
first, acquired the nonblocking `/tmp/dg_igr-sm120.lock` lease, passed one- and
two-rank MPI, and released the lease. Reusing the same path returned exit 73;
all files remained byte-identical. The final run is preserved under
`evidence/runs/remediation-final-20260718`. Two substantive remediation
failures are also preserved: `remediation-compile02-serial-fd-failure` records
the initial skeleton finite-difference cancellation failure, and
`remediation-compile03-mpi-projection-crash` records the unsupported trace
projection crash.

Canonical machine-readable evidence:

- `evidence/serial_checks.json`: deterministic algebra, independent MFEM
  operator checks, p1-p4 oriented condensation, and convergence gate margins;
- `evidence/convergence.csv`: 64 Poisson/advection, affine/curved, p1-p4 rows;
- `evidence/mpi_1_checks.json` and `evidence/mpi_2_checks.json`: actual MFEM
  local/global true-DOF maps, ownership, ghost exchange, and parallel Schur
  action;
- `evidence/runs/remediation-final-20260718/collision_guard.txt`: exit-73 and
  byte-immutability result.

## Required artifact mapping

1. **Concrete formulation:** `README.md` freezes the interior and trace
   spaces, Poisson/advection fluxes and stabilization, boundary conditions,
   local residual/Jacobian blocks, condensation, Schur action, trace solve,
   and recovery.
2. **MFEM v4.9 semantics:** every JSON records clean MFEM v4.9 commit
   `d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec`. The implementation uses MFEM
   geometry, L2/trace spaces, face transformations, orientation maps,
   integration and finite-element operators, `ParMesh`,
   `ParFiniteElementSpace`, `P^T S_local P`, and local dense factorization.
   Lane CMake rejects accelerator-enabled MFEM.
3. **Hybridization boundary:** `README.md` explains why MFEM
   `Hybridization` and `StaticCondensation` are not substituted for the
   independently defined HDG trace flux, exposes `A/B/C/D`, and makes no
   native transient nonlinear Euler HDG claim.
4. **Small-matrix and independent fixtures:** `serial_checks.json` includes
   interior and skeleton finite-difference residual/Jacobian checks,
   full-versus-condensed assembly, Schur action, reduced RHS, trace solve, and
   recovery for both scalar equations. Independent MFEM source/mass/weak
   derivative/convection integrators check element operators. Skeleton
   residual values are `1.3195812442745591e-08` (Poisson) and
   `9.627187935734582e-12` (advection), below `2e-7`; MFEM element-operator
   errors are at most `2.6645352591003757e-15`, below `2e-10`.
5. **Manufactured convergence/recovery:** `convergence.csv` contains four
   refinements for every required combination. JSON rejects nonfinite values,
   requires strictly decreasing errors, and requires the final rate to exceed
   `p+0.65`. All 16 final-rate gates pass; the smallest recorded margin is
   `0.347606114378221`. Recovery and conservation remain independently
   reported.
6. **Orientation, conservation, BCs, MPI:** the p1-p4 oriented skeleton
   fixture uses actual `DG_Interface_FECollection` face DOFs,
   `FaceElementTransformations`, MFEM element integrators, local condensation,
   global skeleton assembly, and MFEM orientation `-1`; the maximum error is
   `1.1102230246251565e-15`. Essential traces, constant states, conservation,
   and recovery also pass. The two-rank JSON records nine owned true DOFs, one
   ghost, one shared face, and the explicit shared mapping `(rank 1, local 0)
   -> (global 4, owner 0)`. Owner-to-ghost error is zero. The independent
   `P^T S_local P` versus direct `P^T G_local` residual error is
   `2.468025783741723e-12`, below `2e-10`.
7. **CPU/reference constraints:** lane CMake requires C++17, pinned CPU MFEM,
   and warning-clean compilation. The implementation is free-function
   reference algebra with plain data carriers and no custom model vtable,
   Kokkos, direct RAJA/Umpire, CUDA, packing, tensor-core, or performance path.
8. **Production exclusion:** only lane-owned `include/numerics/hdg/**`,
   `src/numerics/hdg/**`, and `tests/hdg/**` changed. Euler/IGR, wall
   thermodynamics, every-stage sigma, lifecycle wiring, shared CMake, device
   work, and later-wave optimization remain untouched.

The local direct-solve residual tolerance is `2e-12`, tighter than the normal
`2e-10` comparison tolerance. The finite-difference Jacobian tolerance is
`2e-7`; p1-p4 recovery tolerance is `5e-10`. No performance claim is made.

## Exact later integration request

The coordinator should add `add_subdirectory(hdg)` to `tests/CMakeLists.txt`
and register lane-local serial, one-rank MPI, and two-rank MPI invocations with
unique binary-directory output paths. MPI registration must use a nonblocking
`/tmp/dg_igr-sm120.lock`, treat exit 75 as a yielded lease, and serialize the
one-/two-rank jobs. No root/shared CMake change is part of this commit.
