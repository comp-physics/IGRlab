# FP64 IGR face tensor factorization

**The opt-in p5/Q6 1D tensor candidate reduced matched 512² Sedov process
time by 10.518% (846.787 → 757.720 s) over the validated coefficient-preparation
TF32 pipeline; its record kernel was 2.492× faster. Native and full-run FP64
residual audits passed. Complete saved-Q5 comparison found conservative-field
differences of at most 1.963e-9 relative L2 and 8.903e-9 GL-sampled normalized
Linf.** Sampled peak process GPU residency increased by **96 MiB**. Results
apply to this tested case. The earlier reference-table cache prototype was
rejected for performance.

The preceding coefficient-preparation campaign's Nsight Systems capture
(job 3414629) attributes **183.537941 s / 33.9983% of summed GPU kernel time**
to the FP64 current-IGR face family. Its dominant face-record kernel accounts
for **140.146817 s across 96,516 calls**. These motivate the work; they are
prior-capture measurements, not new-implementation or wall-time fractions.

## Measured first prototype: reference-table cache

Candidate `130fc9be598b168b0b04142664152bf35eb48275` shares byte-identical
reference DShape blocks and reconstructs each physical normal derivative
inside the original dense DOF projection and quadrature lift. Job **3421775**
passed **448 cases each on CPU and CUDA**; CUDA maximum normalized relative
L2/Linf errors were `6.1821520235515655e-16` / `9.9235559957760425e-16`.
Earlier job 3421609 stopped at an admission fixture; its repair changed only
test geometry.

On the actual 512² p5 GL Sedov mesh, three slots admitted, but the unprofiled
record kernel took **2.241950512 ms versus 1.365167975 ms dense: 64.2% slower**
(four warmups, 64 launches), with bitwise equal output. Reference, geometry
and slot storage was 20,736 / 100,466,688 / 2,093,056 bytes; one dense normal
table is 1,808,400,384 bytes and remained allocated.

Bounded NCU job **3421790** on `gh126` found lower DRAM throughput
(1.630901 TB/s → 258.935610 GB/s), but L1 load-service utilization rose
79.059291% → 90.140747%; LG/MIO throttle ratios rose
0.077112 / 0.418740 → 2.950119 / 4.812247. Registers increased 40 → 48 and
active warps fell 71.733236% → 61.045037%. These counters support L1
load-service pressure and motivated reducing reference requests and
contraction work. This rejected candidate was not run through a full
physical evolution.

## Current candidate: 1D tensor contractions

Candidate `c4a7498b3aa52778f19a02be9a09f644342fcb02` replaces dense per-DOF
reconstruction with endpoint and tangent 1D contractions. One CUDA warp owns
a face: it contracts the two element traces along the reference normal axis,
evaluates values and both reference derivatives tangentially, applies the
per-quadrature physical geometry and current side coefficients, then performs
the reverse 1D lifts. This is tensor-product **FP64 SIMT arithmetic** and
introduces no Tensor Core operations or reduced precision.

Admission is limited to **n = nq = 6**, 36-DOF scalar VALUE tensor quadrilaterals
with GL or GLL basis. Setup verifies the actual nodal finite element, a
bijective lexicographic-to-native DOF map and its node coordinates. It builds
each side from the actual mapped reference quadrature points and checks the
1D products against native shape and DShape evaluations. Endpoint/tangent
packets include orientation and are shared only when byte-identical, with at
most 64 slots. Failed representation checks decline the tensor path.

FP64 geometry packets remain per face, side and quadrature point:
`nh = adj(J) * (0.5 * w / det(J) * normal)`, copied from the existing builder.
This permits skew and curved physical cells without assuming constant
geometry. State-dependent side coefficients are refreshed each update. The
1D contraction changes accumulation order; measured parity below shows FP64
roundoff differences rather than the first prototype's bitwise equality.

`IGRLAB_IGR_FACTORIZED_FACE_NORMALS=1` (legacy alias
`DG_IGR_IGR_FACTORIZED_FACE_NORMALS=1`) **defaults off** and is sampled at
operator construction. Ordinary CUDA, matching geometry and successful
coefficient-refresh revisions, correctly sized side coefficients and
arithmetic/max/sum scaling are required. Initial applications, harmonic
scaling, unsupported bases/backends, invalid or stale metadata and cache
declines use the dense path. Refresh invalidates its coefficient stamp before
work and stamps it after successful submission. The additional factor cache
and staging budget remains bounded at 256 MiB, including host/device copies;
this guard can decline larger cases.

Dense normal tables, weighted-normal materialization during refresh,
CPU/shared-face paths, diagonal consumers and the deterministic gather are
retained. The reference-table kernel remains available to the test-only
benchmark for comparison.

## Measured current native gate

Job **3421849** completed the canonical `build.sh` workflow and passed
`dg_igr_exact_face_cpu` and `dg_igr_exact_face_cuda`, **448 cases each**.
Measured normalized relative error maxima were:

| Suite | Relative L2 | Relative Linf |
|---|---:|---:|
| CPU | 6.1642605143736054e-16 | 9.9235559957760425e-16 |
| CUDA | 1.0454912948510563e-15 | 1.0667822695459245e-15 |

Both satisfy the `5e-12` tolerance. The suite compares dense and independently
assembled MFEM actions and covers GL/GLL, skew/curved cells, coefficient
refresh, same-size geometry rebuilds, fallback metadata and public-apply
selector wiring. CUDA recorded **36 tensor applications**, both normal axes
(mask 3), and 852 reversed-side observations. Observed native DOF maps were
all identity (`nonidentity_tensor_maps=0`); the implementation checks general
native-map proofs, but this suite does not establish nonidentity-map fixture
coverage.

On the actual Sedov mesh with `-rs 0 -rp 4` (**512² ordinary-DG p5 GL
elements**, 9,437,184 scalar DOFs, 523,264 interior faces and six face
quadrature points), four warmups and 64 launches measured:

| Variant | Mean record kernel | Relative L2 / Linf versus dense |
|---|---:|---:|
| Dense baseline | 1.374529481 ms | 0 / 0 |
| Reference-table cache | 2.238557577 ms | 0 / 0; bitwise equal |
| 1D tensor | 0.551625490 ms | 3.1319331446655432e-16 / 5.593768420680324e-16 |

The tensor kernel is **2.49178× faster (59.868% less time)** than dense in
this benchmark. Actual tensor admission used three slots: endpoint 576 B,
tangent 3,456 B, axes 24 B, face-slot map 2,093,056 B and native-DOF map 144 B.
The common per-quadrature geometry remains 100,466,688 B; dense tables and
weighted-normal refresh are still retained. These timings use
`scope=record_kernel_only` with synthetic nonuniform state and coefficients.
Public apply and full physical evolution were not timed by this benchmark.

The ≥10% record-time improvement gate admitted bounded Nsight Compute
sampling in the same job on GH200 node `gh112`, one selected record kernel
per variant:

| Counter | Dense | 1D tensor |
|---|---:|---:|
| Profiled record kernel | 1.382592 ms | 0.578080 ms |
| DRAM throughput | 1.633905 TB/s | 1.001830 TB/s |
| L1 load-service wavefront throughput, % of sustained peak | 78.765007% | 84.971042% |
| LG throttle ratio | 0.078621 | 0.001627 |
| MIO throttle ratio | 0.449886 | 0.125220 |
| Registers per thread | 40 | 40 |
| Active warps, % of sustained maximum | 72.150492% | 69.418325% |

LG/MIO stalls fall while L1 utilization remains high and increases. The
shorter kernel is consistent with less load/contraction work completing
faster; reduced L1 utilization does not explain the gain. These single
hardware samples diagnose the record kernel and provide no full-Sedov timing
or accuracy result. The unprofiled 64-launch means are the performance gate.

## Measured matched full Sedov pair

Job **3421875** completed one unprofiled run per variant sequentially on
GH200 node `gh134`, GPU `GPU-6d6c43dc-cf98-6ce2-d8ad-b993403bfe0f`. The frozen
coefficient-preparation baseline (`a6cfdc1`) already uses the validated TF32
spectral inverse and FP32 Euler tensor-line pipeline; the candidate
(`c4a7498`) adds the opt-in FP64 face contractions. This is an incremental
comparison over that pipeline, not a new original-FP64 comparison.

| Full-run metric | Coefficient-preparation baseline | 1D tensor candidate |
|---|---:|---:|
| Setup/output-inclusive process time | 846.786549015 s | 757.720057623 s |
| Sampled peak process GPU residency | 21,286 MiB | 21,382 MiB |
| Accepted / rejected time steps | 4,939 / 0 | 4,939 / 0 |
| Converged / total linear solves | 14,818 / 14,818 | 14,818 / 14,818 |
| Production / output-only solves | 14,817 / 1 | 14,817 / 1 |
| Total Krylov iterations | 51,739 | 51,732 |
| Independent FP64 certificates / solves | 14,818 / 14,818 | 14,818 / 14,818 |
| Maximum FP64 RHS-relative residual | 9.9963081e-7 | 9.9998798e-7 |
| Final committed-state FP64 RHS-relative residual | 3.615353934207918e-7 | 3.615794157919136e-7 |

The candidate saved **89.066491392 s, or 10.518175% (1.117545×)**. Both reached
`t=0.10000000000000001`; every solve and the final committed state passed the
common `1e-6` FP64 RHS-relative threshold, with no missing certificate IDs,
residual-replacement retries or rejected steps. The seven-iteration
difference reflects a slight numerical path change; these are not bitwise
identical evolutions. Sampled GPU residency increased by **96 MiB**; there is
no measured memory reduction. Residency was sampled once per second, so
shorter peaks can be missed.

Physical and solver settings match: final time 0.1, L2/BrokenL2 IGR,
`C_alpha=1/6` with max element length, physical alpha
`6.357828776041666e-7`, CFL 0.3 and RHS-relative linear
rtol `1e-6`; `--no-nan-scan --no-admissibility-scan`, positivity limiting and
modal damping off. Timing is one observation per variant on one GPU, with no
repeat-distribution or confidence-interval claim. Performance is established
only for this tested 512² p5/Q6 case; no broader size, order or geometry
speedup is claimed.

## Complete final saved-Q5 comparison

Both native VTUs were transferred and hash-verified, each **1,160,775,042
bytes**. Comparison covers all **262,144 matched cells**, using GL6×6 in
each cell (**9,437,184 points**) at `t=0.1`. Time is selected from each PVD;
matching physical alpha `6.357828776041666e-7` comes from the validated run
and configuration receipts, not from the VTU itself. The files and fields
are not bitwise equal.

Relative L2 divides the difference norm by the baseline L2 norm; normalized
sampled Linf divides the maximum GL-point difference by the baseline maximum
absolute value on those GL points.

| Final field | Relative L2 | GL-sampled normalized Linf |
|---|---:|---:|
| Density | 6.8499793e-10 | 7.2315034e-9 |
| Momentum x | 1.9627596e-9 | 8.6185329e-9 |
| Momentum y | 1.9596664e-9 | 8.9024862e-9 |
| Energy | 5.3779763e-10 | 3.6613770e-9 |
| Derived pressure | 5.2827525e-10 | 3.5028964e-9 |
| Derived speed | 1.4347513e-9 | 7.1137519e-9 |
| IGR sigma | 9.46169365e-8 | 4.07259471e-7 |

All four conservative components pass the engineering budgets of `1e-3`
relative L2 and `1e-2` sampled normalized Linf. Sigma's absolute L2 difference
is `1.8943898e-12`, with maximum GL-point absolute difference `2.3022064e-11`;
no sigma acceptance budget was prescribed. Derived pressure/speed are
reported without applying the conservative-component budgets.

GL6 integrates squared differences of the complete saved Q5 conservative
and sigma polynomials exactly up to FP64 roundoff on this affine mesh.
Derived pressure/speed norms are quadrature estimates. Linf is sampled at
GL points, not the continuous polynomial supremum. These results establish
saved-polynomial agreement between the two runs at the saved final time;
they do not measure analytic Sedov accuracy or unsaved times.

| Evidence for current 1D tensor candidate | Status |
|---|---|
| Native CPU/CUDA suites and measured error maxima | Passed; job 3421849 |
| Actual 512² p5 tensor admission and storage | Admitted; three tensor slots |
| Dense/reference-table/tensor record timing | Measured; tensor 2.49178× faster than dense |
| Bounded hardware profile | Completed; job 3421849 |
| Matched full Sedov runtime, iterations and residual audit | Passed; 10.518175% less process time, job 3421875 |
| Final physical-field comparison | Completed; conservative engineering budgets pass, sigma differences reported |

One DeltaAI GPU is authorized at a time under `bdiy-dtai-gh`, partition
`ghx4`, constraint `projects`, with eight CPUs, 96G host memory and
`OMP_NUM_THREADS=1`.

## Provenance

Branch: `igr-face-factorization-opt`.
Validated coefficient-preparation base:
`d734b926af2cdb581f966f6a42029d8505de832f`, whose tested implementation is
`a6cfdc1d3b3c26a0eaeb3a5412b51cb05796f022`.
Current candidate: `c4a7498b3aa52778f19a02be9a09f644342fcb02`.
The frozen source graph contains **771 files**. Relative to the validated
base, four deployment paths differ: `DGFaceQuadratureKernels.hpp`, its `.cpp`
implementation, `IGR_linear_solve.cpp` and `test_igr_exact_face.cpp`.
The current tensor step changes the header and test. Frozen earlier binaries
and evidence are preserved. Job 3421849 tested source `c4a7498` with SHA256:

- Test binary: `a7af49828fc17d7a1a6e59a2a3b7b802a05f4ee8f7dae337753d472c873e98d0`.
- Application: `5ca34c4e03bd2c04353b0f05555c9f86b5da7c6804cde597e9deb620f07d3fd4`.
- Frozen baseline application: `2f858af42e59acdf66deca72e6c7ebcee309b57eb5b641b3ed410cfdfc759dd6`.

The authoritative full-pair specification is `tensor-v2/case-spec.json`,
SHA256 `be03fdf22f9569ffb05c96f94cdfe8e0e5a6551c4f7443a2db932c0faae8a02d`.
Final native VTU SHA256:

- Baseline: `ac8b93dba31f0b9032d47decdf58b6c003cd7c14094fcd8e5e81bf69aad41c93`.
- Candidate: `c2e6c8cd52a227b30f6125974fa634632a686787a7ea44d448d1ec472c04a5ed`.

Local evidence root:
`/home/jarias9/.cache/igrlab-experiments/sedov-face-factorization-20261020`.
First-prototype receipts are `build-3421775/`, `ncu-3421790/` and
`reference-cache-analysis.json`; NCU CSVs have a units row and one data row.
Current source manifests, job scripts and receipts use `tensor-v2/` beneath
that root; completed gate/profile receipts are `tensor-v2/build-3421849/` and
`tensor-v2/ncu-3421849/`. Full-run `comparison.json` and both per-variant
audit JSONs are in `tensor-v2/comparison-3421875/`, alongside
`field-comparison.json`, `conservative-fields.json`, `sigma-fields.json` and
each variant's `final-vtk/collection-receipt.json`. Remote campaign root:
`/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/sedov-face-factorization-20261020`.
Source manifests record the complete source identity; `build.slurm` records
canonical build/test and benchmark commands. All campaign GPU jobs are
complete.
