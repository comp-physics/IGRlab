# Consolidation validation

The tested numerical source is
`0e4d578b0b3bd1acf99f3b30a05cfda5d0d47e8d` on
`codex/igrlab-consolidation`. The final packaging commit changes consolidation
documentation only. MFEM remains pinned to
`d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec`.

All application configuration, compilation and ordinary tests used this
checkout’s canonical `build.sh`, installed Spack profiles and Release builds.
CPU builds used four jobs; CUDA builds used two application/MFEM jobs. Local
GPU execution used the authorized RTX 5080, serialized tests and both shared
GPU lease locks. No A100 allocation or new remote GPU job was used.

| Configuration | Selected result | Coverage |
| --- | --- | --- |
| CPU, HDG ON, research OFF | 68/68 passed | DG/DGSEM, IGR, damping/SSPRK, boundaries, regularization, HDG contracts/runtime/application/MPI |
| CPU, HDG OFF, research ON | 18/18 passed | Ordinary operators/IGR/damping, source oracles, compatibility checks; all enabled CPU research probe targets compiled |
| CUDA SM120, HDG ON, research OFF | 31/31 passed | CPU/CUDA parity, source operators, damping/SSPRK, DGSEM production/manual tensor paths, HDG ordinary/tensor sessions, communicator splits, tensor SASS |
| CUDA SM120, HDG OFF, research OFF | 18/18 passed | DG/DGSEM, IGR source, damping/SSPRK and application parity checks |

The CUDA tuple is dependency architecture `120`, MFEM `80-virtual`, and
application architecture `120`. HDG’s CPU-only collective-projection
rejection test passed in the CPU batch; it is intentionally absent from the
CUDA registration. The screened tensor patch test belongs to the opted-in
HDG runtime and is also absent with HDG disabled. These are selected suites,
not every registered test or every historical campaign.

The compatibility tests exercise canonical/legacy cache synchronization,
runtime environment precedence and both legacy namespace aliases. The
ordinary CPU IGR suite includes a nonzero host-coefficient regression against
direct solver construction for average/local alpha policies and subsequent
alpha/policy updates. Communicator-authority and zero-owned-rank collective
admission gates for superseded PRs #37/#38 passed.

## Repairs required by validation

The first CUDA run passed 27/31 selected tests. Its four failures are retained
in the audit records. Two pre-existing correctness defects were repaired:

- The application factory moved a solver whose coefficients held references
  to that solver’s parameter members. It now constructs at the final heap
  address and returns the owning pointer. Configuration resolution is
  unchanged. Both previously failing DG/DGSEM application parity checks pass.
- Rebuilt inverse-mass/sigma mappings could retain a device-valid view while
  new values were written through unchecked host indexing. Explicit
  `HostWrite()` calls establish the authoritative host copy before device
  reads. The existing refined-operator CUDA damping regression passes.

The tensor assembly check also now selects `UseTensor=true` exactly beside
the preserved SIMT specialization. Native SM120 disassembly records 19 FP64
`DMMA.8x8x4` instruction sites in condensation and three in volume contraction.
These are static instruction sites, not measured execution counts or speedups.
Runtime tensor sessions passed separately.

An exclusive `compute-sanitizer` memory check of the canonical-built CUDA
damping executable passed with **zero reported memory errors**, including
the refined DG/DGSEM paths, in both HDG-enabled and disabled builds. The
final diagnostic binds the disabled-build binary’s unchanged SHA256. CUDA-aware OpenMPI’s host-pointer probing API
messages were excluded with `--report-api-errors no`; device memory checks
remained enabled. The initial failing diagnostic and final command/log are
retained. Independent reviews found no changed numerical formulas, damping
rates, cutoff, mass, quadrature or flux policy in these repairs.

## Bounded application damping comparison

The current-source CPU application comparison uses p3 DGSEM on a 3×3 affine
periodic quadrilateral mesh, a nonuniform active wave, assembled L2/BrokenL2
IGR closure at every SSPRK3 stage, central volume/Rusanov surface fluxes and
no positivity limiter. Coefficient QFunctions, partial assembly and the
exact-face route are explicitly disabled. The two runs differ only in M16
rate/profile: zero versus 80000.

Both runs accepted 128 identical steps of `2^-20` to
`t=0.0001220703125`, rejected none, and converged all 384 stage closure solves.
Final sigma is nonzero; sampled density, pressure and q_sigma remain positive.
The selected high-degree set is `max(a,b)=3` in each cell’s degree-3
Legendre tensor polynomial. Its physical L2 norms in the four conserved fields
decrease 99.983–99.987%. Maximum conservation defects are
`1.47e-14` without damping and `1.73e-14` with damping. Sigma is re-solved and
is not directly damped; its selected high-degree norm increases in this case.

This verifies a bounded application treatment effect and retained
conservation. It does not reproduce the original one-dimensional p39
fine-step ringing study, supply an exact-reference error, or establish
universal pointwise/accuracy improvement. The older pre-fix assembled attempts
with zero sigma and the separate exact-face comparison remain explicitly
labeled in the audit records.

## A100 and remaining scope

The canonical A100 print plan still selects native `80/80/80`. Required SM80
policy, dependency artifact/ABI handling, host-compiler/template workarounds,
CUDA host facade, trace parameter/scheduling fixes and TF32 rounding correction
are retained; see [A100 support](../a100.md).

Historical A100 qualification remains tied to its original source/software
tuple: 9/9 selected tests plus clean p1 trace memcheck, and 80/80 admissible
horizons to `t=0.1`. The archived unfinished PACE p4 experiment passed 12/15
checks. No fresh A100 execution, new performance benchmark, full historical
campaign rerun or new GPU research-probe qualification is claimed here.

## Evidence and recovery

Durable evidence is under
`/home/jarias9/archives/igrlab/consolidation-20261010/audit/`: `implementation/`
contains build plans, exact commands, CMake caches and test/sanitizer logs;
`planning-backends/m16-application/current-assembled-L2/` contains the complete
application fixture, settings, binary/mesh/output hashes and measurements;
`planning-a100/` contains qualified-source inventory and independent repair
reviews; `planning-git/` contains final cleanup guards and preservation checks.

All raw archive members/checksums are verified. The isolated restore drill
checks all named refs/stashes, all 100 retirement records, MFEM and full
historical reflog closure (28,141 reachable objects, zero missing), the complete
PACE worktree and selected primary files. A separate exact 210-blob supplement
repairs pre-existing empty Git objects while preserving raw snapshot bytes.
The original database repair retains every displaced file and passes full
connectivity checks. Independent post-repair checks confirm every original
inventoried worktree file, symlink, status and dirty binary patch matches the
archive. Submodule working files and external dependency installations are
excluded from physical worktree volumes: MFEM source is recoverable from the
pinned archived submodule objects, and literal symlinks preserve their targets
without capturing external target bodies.

The final candidate Git bundle, its checksum and isolated clone verification
are recorded in the archive’s candidate packaging receipt. Deletion,
superseded-PR closure, publication, merge and repository/checkout migration
remain separate approval steps.
