# Exact-face kernel experiments on the fixed large Sedov case

These prototypes did not establish an end-to-end improvement and are retained
on `igr-face-operator-opt` as experiments, not a recommended replacement for
the validated `igr-dg-transport-memory` baseline.

The case remains 512 by 512 ordinary DG p5 GL, t=0.1, CFL0.3,
C_alpha=1/6 with max length, L2/BrokenL2, TF32 spectral inverse, and FP64
current-equation residual certification at RHS-relative1e-6. Optional
nonfinite/admissibility scans and limiting are off.

## Compact geometry prototype

Commit `00aef19b49769ba14f77ea70a80911861871a6ba` adds an independent,
bounded, byte-exact dictionary for normal geometry. After a matching
coefficient refresh, ordinary CUDA arithmetic/max/sum face applications may
reconstruct weighted normals with explicitly rounded multiplication. Initial,
harmonic, stale, unsupported and high-quadrature paths retain dense normals.
Dense tables remain materialized for other consumers. The experimental switch
is `IGRLAB_IGR_COMPACT_FACE_NORMALS` (legacy DG_IGR_ alias also supported).

Native build3413917 passed the existing exact-face CPU and CUDA suites:
416cases each, including bitwise old-device-action parity and independent
assembled-MFEM checks; maximum assembled discrepancy2.08167e-17.

Job3413951 ran the frozen baseline and prototype sequentially on gh082,
GPU `GPU-7e2ecb23-3215-1dbe-c35f-473a51dcac4f`:

| Measurement | Frozen baseline | Compact prototype |
|---|---:|---:|
| Full-process seconds, setup and VTK included | 891.301207 | 895.434423 |
| Steps / outer iterations |4939 /51739|4939 /51739|
| Sampled peak GPU memory, MiB |22444|22444|
| Maximum observed FP64 RHS-relative residual |9.9963081e-7|9.9963081e-7|

This one observation is0.46% slower, not a demonstrated gain. Both final native
VTUs are byte-identical (all stored fields and geometry), SHA256
`ac8b93dba31f0b9032d47decdf58b6c003cd7c14094fcd8e5e81bf69aad41c93`,
1160775042bytes each. No resampling/norm approximation is needed to establish
that saved-field identity. This is not an analytic PDE accuracy claim.

The candidate executable SHA256 is
`d10f87004a05610f684ddebea80ef55fe700776a0f3f7468ac42c8d1ee25470c`;
the frozen baseline is
`c6f91d544b831499ebe96dbcf35306aa6e145d83e1c6333c46f8ec115a145d42`.
CUDA binary resource inspection reports40registers/thread for the baseline
and56for this runtime-selected candidate, with zero stack/local bytes.
These are compiled resources, not measured occupancy or proof of a bottleneck.

## Specialization and large-mesh selection diagnostic

Commit `4a36aadd77ba6a6bef6e9488af4de2657957be9e` separates compact/dense
kernels and adds fixed36-DOF/6-point dispatch with dynamic fallback. Native
build3414224 passed both exact-face suites again. Its executable SHA256 is
`9268a7b0e3059b8501abd85086633cddf25ad94325bd4ecfc088dd94a095b063`.
Compiled compact-p5 register use drops to48; compact-dynamic and both dense
variants use40, without spills.

Test-only commit `9503bf2cf2b2bece839903ffb0c91dfa52ac9935` adds the opt-in
`dg_igr_exact_face_test cuda --benchmark512 /path/to/sedov_taylor.msh` mode.
Job3414369 on gh011 used the exact production mesh file, no serial refinement,
four parallel refinements,262144cells,9437184scalarDOFs and523264interior faces.
This is a record-kernel selection diagnostic with synthetic nonuniform
coefficients/input, not physical evolution or an end-to-end benchmark.

**The normal dictionary was not admitted on the actual refined mesh.** The
shape dictionary had3slots; normal slots were0. A Cartesian mesh alone is
insufficient to guarantee admission under this bounded byte-exact block
packing. Therefore the compact variants were skipped rather than timed as
though active. All executed record variants matched the dense reference bits.

With4warmups and64CUDA-event-timed launches:

| Record kernel | Mean milliseconds |
|---|---:|
| Dense dynamic dimensions |1.890699|
| Dense static36/6 |2.203409|

The static variant was slower in this diagnostic. Neither second full-run
comparison nor fresh profiling was justified by these results. Previously
queued profile3414000 was cancelled while pending; no profiler GPU run occurred.

The next coefficient-preparation work starts from the validated baseline,
without these unhelpful face changes. Read-only analysis identified diagnostic-only
diffusion gradients and repeated equal-rule state interpolation as separate
candidates. They are not implemented or credited as savings in this branch.

Evidence root:
`/home/jarias9/.cache/igrlab-experiments/sedov-face-normal-20261019`.
The complete first-pair receipts and final field identity are in
`comparison-3413951/`; native resource reports are in`resource-inspection/`.
The second source/test receipts and kernel diagnostic are under
`static-kernel-v2/`. All original comparison artifacts are preserved.
