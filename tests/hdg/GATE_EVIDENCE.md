# HDG-ORACLE-01 gate evidence

Final verification command:

```text
tests/hdg/run_oracle.sh /tmp/dg_igr-hdg-oracle-0214-wave1-run10
```

The command used an isolated build, output, and log tree under that directory.
It ran serial first, then acquired the nonblocking
`/tmp/dg_igr-sm120.lock` lease, passed one- and two-rank MPI, and released the
lease. Preserved earlier runs are `run01` through `run09`; `run01`,
`run02`, `run03`, `run04`, `run05`, and `run07` retain their original
configure, compile, or numerical failures.

The committed machine-readable evidence is:

- `evidence/serial_checks.json`: deterministic algebra and all scalar gates;
- `evidence/convergence.csv`: 64 Poisson/advection, affine/curved, p1-p4 rows;
- `evidence/mpi_1_checks.json` and `evidence/mpi_2_checks.json`: ownership
  and exchange results.

## Required artifact mapping

1. **Concrete formulation:** `README.md` freezes the interior and trace
   spaces, Poisson and advection fluxes/stabilization, boundary conditions,
   local residual/Jacobian blocks, condensation, Schur action, trace solve, and
   recovery.
2. **MFEM v4.9 semantics:** every JSON file records MFEM v4.9 commit
   `d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec`. The implementation uses MFEM
   curved geometry, L2 and trace spaces, integration rules, finite-element
   shapes, orientation maps, `ParMesh`, `ParFiniteElementSpace`, and dense
   LU. CMake rejects an accelerator-enabled MFEM install.
3. **Hybridization boundary:** `README.md` states why MFEM
   `Hybridization`/`StaticCondensation` do not directly represent this
   independently defined HDG trace flux, exposes `A/B/C/D`, and makes no
   native transient nonlinear Euler HDG claim.
4. **Small-matrix fixtures:** `serial_checks.json` has separate passing
   residual/Jacobian, full-assembly-versus-condensation, Schur-action,
   reduced-RHS, trace-solve, and back-substitution checks for both equations.
   MFEM dense monolithic solves are independent references.
5. **Manufactured convergence/recovery:** `convergence.csv` contains four
   refinements for every Poisson/advection, affine/curved, p1-p4 combination.
   Final observed rates are approximately p+1; every row includes direct
   condensed-versus-monolithic recovery and conservation errors.
6. **Orientation, conservation, BCs, MPI:** `serial_checks.json` freezes the
   actual p1-p4 MFEM `DG_Interface_FECollection` permutation, reversed Schur
   algebra, essential traces, constant states, and conservation.
   `mpi_{1,2}_checks.json` verifies MFEM true-DOF counts, deterministic
   lower-element ownership, matrix/RHS reduction, and owner-to-ghost exchange.
7. **CPU/reference constraints:** `tests/hdg/CMakeLists.txt` requires C++17,
   CPU MFEM, and warning-clean compilation. The implementation is static
   free-function reference algebra with plain data carriers and no custom
   model vtable, Kokkos, direct RAJA/Umpire, CUDA, packing, tensor-core, or
   performance path.
8. **Production exclusion:** only `include/numerics/hdg/**`,
   `src/numerics/hdg/**`, and `tests/hdg/**` are changed. Euler/IGR,
   trace-wall thermodynamics, every-stage sigma, lifecycle wiring, shared
   CMake, device work, and later-wave optimization remain untouched.

The MFEM direct-solve residual tolerance is `2e-12`, tighter than the normal
`2e-10` comparison tolerance. The finite-difference Jacobian comparison uses
`2e-7`; p1-p4 recovery uses `5e-10`. No performance claim is made.
