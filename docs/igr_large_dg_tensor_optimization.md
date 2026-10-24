# Large single-GPU DG tensor optimization

This experiment optimizes the application on one fixed large smooth DG case.
It is not a mesh-convergence or maximum-capacity study. Small CUDA fixtures
establish correctness; their timings are not HPC performance evidence.

## Workload and comparisons

DeltaAI, one GH200, account `bdiy-dtai-gh`, partition `ghx4`, `projects`
constraint. The fixed periodic unit-square mesh has 512 by 512 elements at p5:
9,437,184 scalar closure DOFs and 37,748,736 conserved-state DOFs. Use the
smooth active IGR wave, RK3 SSP, dt=1e-5, `c_alpha=1`, local alpha length
(alpha=3.814697265625e-6), SIPG exact faces, and the current FP64 operator
for solve acceptance. No positivity limiter or shock problem is involved.
All variants use the same mesh and physics. FP64, FP32 and TF32 controls
separate algorithmic changes from arithmetic precision.

Native initial/final conservative-state exports use round-trip precision and
explicit host synchronization. A streaming comparison uses p5 Gauss weights
for conservative-field L2 differences and integrals; Linf and positivity are
evaluated at saved nodes. Existing acceptance budgets are normalized L2<=1e-3,
Linf<=1e-2, excess invariant drift<=1e-6, and native conservation<=5e-10.
Agreement with the same-discretization FP64 solution is not an analytic PDE
error estimate. Output runs and profiler captures are separate from timing.

## Literature and implementation choices

Bryngelson, *Tensor-cores accelerate compact finite-difference solves and CFD*,
supplied manuscript: a fixed line inverse is reused across many real RHSs;
composing fixed linear stages can remove intermediate work. Reduced-precision
tensor throughput, rather than dense inversion alone, produces the reported
benefit. Its square, well-conditioned compact-FD inverse and inexpensive banded
residual differ from variable-coefficient DG. We retain FP64 state and nonlinear
flux, and measure complete solves including convergence and setup. The supplied
manuscript's tensor arithmetic and state-storage precision are distinct axes.

[Cui, *Acceleration of Tensor-Product Operations with Tensor Cores*](https://arxiv.org/abs/2407.09621)
uses matrix-free tensor products, bank-aware shared-memory access, and mixed
precision multigrid. Its constant-coefficient Cartesian Kronecker structure is
more restrictive than the current IGR operator. The
[multilevel DG companion](https://arxiv.org/abs/2405.18982) additionally motivates
reducing residual/smoother data motion. We borrow these implementation ideas,
without equating our modal restricted-overlap smoother with its patch scheme.

[Tu et al., *Accelerating High-Order Finite Element Simulations at Extreme Scale
with FP64 Tensor Cores*](https://arxiv.org/abs/2603.09038) combines partial
assembly, sum factorization, fusion and shared-memory layouts. Its arithmetic
is FP64 DMMA, not TF32. It motivates reducing stored operators and intermediates
and retaining matched SIMT controls, rather than predicting a speedup here.

## Candidates

`IGRLAB_IGR_TENSOR_MG_OPERATOR=shared` explicitly replaces the sampled frozen
preconditioner matrix by its translation average. On the admitted connected
periodic Cartesian topology, this is an average of permutation congruences of
the SPD screened SIPG matrix. It remains SPD mathematically, and tensor modal
restriction commutes with translation. Variable-coefficient shared MG is an
approximate preconditioner, not an exact replacement for the current equation.
Current-coefficient FP64 outer residual certification remains authoritative.

GPU probes accumulate five shared blocks and per-element diagonal traces.
FP32 level-stencil storage changes from 31,360 bytes per element to 31,360
bytes total; other vectors and coarse CSR still scale with elements. Shared
residual application uses real neighbor vectors as GEMM columns. Pre-coarse
applications compute only retained modal rows and fuse restriction. FP64/FP32
SIMT controls and standard TF32 WMMA implement the same reference model.

`IGRLAB_SHARED_EULER_LAYOUT=tensor-lines` factors conventional DG's dense weak
divergence into directional derivative and transverse mass operations.
Setup reconstructs the original affine Gauss-Legendre matrix and verifies its
agreement. Nonlinear flux, background centering and moment correction remain
FP64. Standard WMMA consumes distinct element/field columns; FP64/FP32 SIMT
controls use the same factorization. Dense layout remains the default.

The tensor-line setup validates every element's affine geometry and nodal basis
before assembling one reference weak-divergence matrix. It avoids both dense
per-element copies (about 10.1 GiB of combined host matrix storage at n512).
The retained matrix is still assembled by the original integrator and checked
against the factorization; unsupported geometry is rejected by this opt-in.

Two common diagnostic costs are also reduced. On ARM, collocated derived-field
projection qualifies the host rounding/FMA mode and uses matching CUDA FP64
arithmetic, retaining exceptional-environment and nonfinite fallbacks. Affine
collocated conservation integration uses the appropriate GL/GLL nodal weights;
other cases retain their quadrature and gather an element only once. These are
separate from tensor-core gains and also apply to the matched FP64 control.

The v4 candidate, `IGRLAB_IGR_TENSOR_MG_OWNED_TILE=4`, uses one 6 by 6 element
input patch to correct a uniquely owned 4 by 4 element interior. Its shared
maps are 576 by 1296 at p5 and 256 by 576 at p3, with 16,384 distinct patch
columns on this mesh. A 64 by 64 output tile reuses shared panels and scatters
corrections directly. This changes the mathematical preconditioner. Native
tests check an independent separable oracle, reordered periodic ownership,
partial CTA batches, and convergence against the current FP64 equation.

The separate `IGRLAB_IGR_TENSOR_MG_RESIDUAL=fp32` control retains TF32 smoothing
while computing shared-reference residuals with FP32 SIMT. It tests whether
TF32 residual arithmetic adds enough convergence or kernel cost to outweigh
its contraction throughput. Both controls preserve the fine physical alpha.

## Global reference inverse

The v5 candidate, `IGRLAB_IGR_TENSOR_MG_OPERATOR=spectral`, removes the patch
hierarchy and coarse CG entirely. It admits a checked periodic Cartesian p5
GL topology, including reordered elements, and verifies that the averaged
five-block modal stencil is a separable sum of two one-dimensional operators.
It diagonalizes those operators once in FP64 with cuSolver and checks the
relative Frobenius residual over all eigenvectors, orthogonality, and positive
screening before inversion.
It never forms the full two-dimensional dense matrix.

On the fixed n512 case, a genuine 3072 by 3072 modal field is transformed as
`Z = Qx^T R Qy`, divided entrywise by `lambda_x + lambda_y - mean`, then
transformed back as `Qx Z Qy^T`. Four cuBLAS GEMMs apply the reusable reference
inverse. Global coordinate packing and nodal/modal changes are included in the
application cost. The original evolving FP64 PDE operator still decides
convergence; the reference inverse is only a preconditioner.

Controls use FP64, pedantic FP32, or explicit `CUBLAS_COMPUTE_32F_FAST_TF32`.
The optional `tf32x3` mode splits both factors and sums high-high, high-low,
and low-high products, for twelve GEMMs per application. The omitted low-low
term means it remains an approximation, not an exact FP32 emulation. The
[cuBLAS compute modes](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html)
select arithmetic; hardware profiling must independently establish Tensor
Core execution. All spectral precision comparisons use identical FP32 Euler
contractions with FP64 state, nonlinear flux, and current-equation acceptance.

This prototype has explicit geometry, topology, separability, index and memory
admission checks. It is not yet a general curved-mesh, multi-GPU, DGSEM, or
arbitrary-order solver.
The dense transforms cost O(Nx*Ny*(Nx+Ny)) operations per application and
O(Nx^2+Ny^2+Nx*Ny) storage. The measured benefit therefore does not establish
asymptotic superiority over sparse or local tensor methods as the mesh grows.
This experiment deliberately holds one large mesh fixed.

## Validation status

Initial source: `60711ded9eac158378bc19c6e44c3df0289ea67b`.
The n512 FP64 admission run (DeltaAI job 3399587) completed 10 steps, all 31
linear solves, and native final acceptance. Process wall time was 225.45 s;
the stepping-loop report was 48.99 s. These are one admission sample, not a
speedup or steady-state performance claim. Peak observed GPU residency was
about 41,010 MiB; the allocated device reported 97,871 MiB usable memory.

The baseline two-step Nsight Systems capture (3399622) shows substantial host
time beyond GPU kernels. Dense weak divergence and Krylov orthogonalization
are material device costs. Setup, final diagnostics, and transfer costs must
therefore accompany any kernel-level comparison.

Seven native suites passed in DeltaAI job 3399714: shared reference maps,
tensor multigrid, center maps, tensor volume, device Krylov, CUDA diagnostics,
and time-default/conservation tests. The tested application SHA256 is
`076bc60443fd858af1f52597f31de2516a6d3e749902875b0013ac48ca91048a`.
The allocator ownership issue found in the first shared coarse CSR construction
was fixed using MFEM-managed storage. ARM control tests use actual FPCR readback
because the hardware may ignore unsupported exception-enable bits.

The v3 source snapshot is recorded as commit
`2bda5540cb07926bc33cf4dccfe86fbf07704675`. Its build receipt binds the frozen
768-file source map to that application hash; the commit mapping comes from
the campaign record. Job 3399745 qualified the original FP64 solver and the
FP64, FP32, and TF32 shared-reference/tensor-line variants for 100 steps to
t=0.001. All 301 solves per run converged. Candidate solves independently
certified the original current FP64 equation at 1e-10; the baseline also
achieved residuals below 1e-10, with its final explicit certificate using
5e-10. Every retained native-state comparison passed.

The v4 build receipt binds its 769-file map to application SHA256
`8a52bdd4873c9bc0f5329fc5a93e1c2255de3ede0831ae76b0519f3ee4064b6b`.
Its original-source map exactly matches the frozen v3 map. The six changed
paths comprise IGR selection/implementation, the owned-patch helper, and its
two CUDA tests. The archived changed-file hashes match the v4 manifest. All
seven native suites passed in build job 3399895; field job 3399918 completed
the five v4 variants below. The independent audits are retained as
`analysis-v3.json`, `analysis-v3-profile-addon.json`, and `analysis-v4.json`.

The table reports field qualification, including saved initial/final state,
301 solves, and an additional diagnostic preconditioner application per solve.
Outer counts include the final output solve; coarse counts additionally
include the diagnostic preconditioner applications. The solve durations are single-run observations with output enabled;
they are **not benchmark results or a performance ranking**. Each error is the
maximum normalized conservative-component difference from the v3 original
FP64 reference on the same mesh and time interval.

| Qualified candidate | Outer / coarse iterations | L2 difference | Node Linf difference | Solve seconds, field run |
| --- | ---: | ---: | ---: | ---: |
| v3 shared FP64 | 2,407 / 17,128 | 1.72e-10 | 2.30e-8 | 36.72 |
| v3 shared FP32 | 2,408 / 17,628 | 1.82e-10 | 2.34e-8 | 30.04 |
| v3 shared TF32 | 3,010 / 32,795 | 1.09e-7 | 9.97e-7 | 41.96 |
| v4 TF32, FP32 residual | 2,408 / 37,264 | 1.09e-7 | 9.76e-7 | 33.90 |
| v4 owned 4 by 4, FP64 | 2,408 / 45,924 | 1.72e-10 | 2.62e-8 | 48.46 |
| v4 owned 4 by 4, FP32 | 2,408 / 46,599 | 1.81e-10 | 2.40e-8 | 37.04 |
| v4 owned 4 by 4, TF32 | 3,311 / 80,999 | 1.09e-7 | 9.17e-7 | 59.07 |
| v4 owned 4 by 4, TF32 with FP32 residual | 2,408 / 53,881 | 1.09e-7 | 9.96e-7 | 41.20 |

All v4 runs completed 100 accepted steps, 300 stage solves and one final
output solve, with no rejected steps, coarse failures, coarse caps, or
fallbacks. All 301 independent current-A certificates per candidate passed;
the largest residual was 1.15e-11. Initial state differences were zero. Maximum
excess invariant drift was 4.23e-16, and native conservation defects stayed
below 4.42e-13. Saved density and pressure remained positive. Sampled GPU
residency was 27.86--28.41 GiB and maximum host RSS 24.56--25.50 GiB. GPU memory
was polled once per second, so shorter peaks may be missed.

These comparisons establish agreement at the saved initial and final times.
Conserved-field integrals and squared L2 differences use exact polynomial
quadrature on the admitted affine cells, up to FP64 rounding. Linf and
positivity describe the saved GL nodes. They do not establish analytic PDE
accuracy, a continuous polynomial supremum, unsaved-time agreement, or a
longer physical horizon. Raw native component files were compared on DeltaAI;
the independent local audit checks the retained comparison records and logs.

## Spectral qualification

The native-tested v5 algorithm is commit
`2ae2661bca31318b9e65fe6752514eee1c2b73af`, application SHA256
`7d723b394e912760c6ce4fe3437cf3fbcfe56f9ec3aad9c48531ddd82379861b`.
All 770 application/configuration/test hashes match the frozen source manifest.
All seven native suites passed in job 3400344, including rectangular 48 by 72
modal layouts, independently assembled constant-operator inversion, changed
current coefficients, extreme RHS scales, nonfinite recovery, and source
operator lifetime.

Field job 3400347 qualified all four spectral modes on the same fixed large
case. Every mode completed 301 independent current-A certificates at 1e-10,
100 accepted steps, and zero rejected steps or fallbacks. The averaged stencil
separability defect was 5.16e-16. The maximum relative Frobenius eigen residual
across the two eigensystems was 1.93e-15, and the corresponding maximum
orthogonality defect was 4.81e-15. No eigenvalues were clipped. Setup took
3.99--4.18 seconds, including reference probing, eigensystems, and validation.

| Spectral precision | Outer iterations | Maximum current-A residual | L2 difference | Node Linf difference | Solve seconds, field run |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP64 | 2,106 | 8.08e-11 | 1.82e-10 | 3.39e-08 | 23.61 |
| FP32 | 2,106 | 8.16e-11 | 1.81e-10 | 2.43e-08 | 23.39 |
| TF32 | 2,407 | 9.9e-11 | 1.83e-10 | 3.53e-08 | 15.20 |
| TF32, three products | 2,106 | 8.79e-11 | 1.82e-10 | 2.4e-08 | 17.51 |

These field-run solve observations are not repeated benchmark results.
All initial states were identical. Maximum excess invariant drift was
3.76e-16 and native conservation error was below 4.51e-13; saved nodal density
and pressure were positive. Sampled GPU residency was 27.51--27.69 GiB and
host RSS 21.08--24.00 GiB. The mesh stayed fixed after storage reductions;
this is a large-workload optimization experiment, not a maximum-memory-capacity
claim. The independent record is `analysis-v5.json`.

## Earlier shared-multigrid profile findings

Job 3399807 added one output-free, unprofiled 100-step run per v3 precision,
plus separate two-step Nsight Systems captures and targeted Nsight Compute
launches. The single FP32/TF32 process observations were 94.14/105.22 seconds,
with 27.98/38.89 seconds in linear solves. They have no repeat-based uncertainty
estimate and do not establish a benchmark winner.

The Systems captures identify two costs in the configured TF32 pipeline:
25% more preconditioner applications and 19.4% more dedicated preconditioner
kernel time per application than FP32. Shared-residual GPU time per
application was 1.81 versus 1.11 ms. Coarse CG required 9.26 versus 7.04
iterations per application while GPU time per coarse iteration was similar,
0.158 versus 0.162 ms. This accounts for additional algorithmic work as well
as kernel cost.

Compute confirms Tensor Core execution, but measured tensor-pipe activity was
only about 2.1--2.4% for the center map, 1.5--1.8% for shared residuals, and
1.06% for Euler tensor lines. Center occupancy was about 72--73%, residual
occupancy 65--71%, and Euler occupancy 42%. The captured kernels reached
2.6--24.8% of sustained DRAM throughput. These launches show substantial
staging/instruction work alongside light matrix-pipe use; occupancy alone
does not establish efficient tensor execution. Missing stall and bank-conflict
counters prevent a more specific bottleneck claim.

The v4 larger patch did not reduce outer iterations and increased coarse
iterations. Using FP32 residuals restored eight outer iterations per solve for
TF32, but its coarse work remained above the v3 FP32 control. The passing
accuracy gates therefore do not supply evidence of a tensor-core speedup.

## Spectral hardware profile

Nsight Systems job 3400362 measured the three floating-point controls on one
GPU. The table excludes cuSolver/setup GEMMs and uses each capture's actual
preconditioner application count. Dedicated inverse time includes products,
normalization, nodal packing/scatter, and diagonal scaling.

| Inverse mode | Applications / GEMM launches | GEMM GPU ms/application | Dedicated inverse GPU ms/application |
| --- | ---: | ---: | ---: |
| FP32 | 48 / 192 | 5.435 | 6.298 |
| TF32 | 55 / 220 | 0.854 | 1.717 |
| TF32, three products | 48 / 576 | 2.668 | 3.690 |

Raw TF32 reduces the GEMM component by 6.37x per application compared with
FP32. It uses more applications, so the measured dedicated GPU totals improve
from 302.3 to 94.4 ms for these two-step captures. About 0.86 ms/application
of packing, nodal transforms, normalization and scaling remains, roughly half
of the TF32 inverse's dedicated GPU time. The three-product variant recovers
the FP32 iteration count but pays for twelve GEMMs per application. Whole
capture face/volume totals also contain 324 setup probes and cannot be read as
steady-state iteration shares.

Nsight Compute job 3400466 recorded four actual GEMM launches per mode. The
raw-TF32 sample covers the four forward/backward products; the FP32 regex
matched only TN FFMA products, and the three-product sample covers one NN
cross-term and the final NT high/low terms. These are selected hardware
observations, not a complete per-iteration timing sample for every mode.

The raw-TF32 kernels exhibit 48.01--81.54% tensor-pipe activity and
15.84--28.50% sustained DRAM utilization. They use 168 registers/thread and
232,448 shared bytes per CTA, limiting residency to one CTA per SM. Barrier,
long-scoreboard waits, and shared-bank conflicts are present. This evidence
supports successful tensor-core use, not an HBM bandwidth saturation claim.
The earlier shared-MG center maps on the same large mesh showed only
2.1--2.4% tensor activity. Kernel replay used cold caches without clock locking;
replay durations are excluded from application performance results.

An additional FP64 Systems control in job 3400466 establishes that cuBLAS
selects IEEE FP64 tensor-core kernels for the FP64 inverse: its 48 NT
`tensor16x8x16` launches match the 48 runtime applications. Pedantic arithmetic
alone does not establish a SIMT implementation. The frozen v5 setup label
`tensor_execution=none` for FP64 is therefore incorrect metadata; use the
observed profile, and the later logging-only correction, rather than that
label. FP32 runtime products were SIMT/FFMA, while both TF32 modes used Hopper
TF32 GEMMs. Full counts, units, phase limits, and source receipts are retained
in `analysis-v5-profile.json`.

## Repeated application measurement

Job 3400467 completed four excluded two-step warmups and sixteen unprofiled
100-step runs on the same GH200 (gh109). All four variants use the same v5
executable, including the common host-diagnostic improvements; the original
FP64 row selects the original solver and operator paths within that binary.
Each variant occurs once in each of four block positions, with directed
predecessor pairs balanced within blocks. Each fresh process includes setup
and final acceptance. Field exports and profiling are disabled.

The application-exit timer excludes evidence parsing and monitor shutdown and
was cross-checked against `/usr/bin/time` within 0.10 s for every run. The
independent audit validates all sixteen runs, the four excluded warmups,
source identity, the prescribed configurations and mesh, the same GPU UUID,
the planned order, and 100 accepted steps with 301 converged solves per run.
Final closure and conservation pass throughout. No failed gates, rejected
steps or candidate coarse failures were found. Candidate
current-equation certification remains at 1e-10. The baseline's final explicit
certificate allows 5e-10; its native every-solve convergence policy is unchanged.

| Variant | Application median [min, max], s | Linear-solve median, s | Outer iterations |
| --- | ---: | ---: | ---: |
| Original FP64 | 257.38 [256.39, 260.75] | 126.12 | 19,148 |
| Shared multigrid, FP32 | 92.61 [92.43, 92.94] | 27.71 | 2,408 |
| Global reference inverse, FP32 | 85.80 [85.29, 86.22] | 21.59 | 2,106 |
| Global reference inverse, TF32 | 78.45 [76.91, 78.62] | 13.37 | 2,407 |

Ratios of application medians give **3.28x** speedup for the TF32
candidate over the original FP64 solver, **15.3% less time**
than shared-multigrid FP32, and **8.6% less time** than the
matched spectral-FP32 control. The last comparison changes only the inverse
precision. Most of the gain over the original solver comes from the operator
and preconditioner reformulation; it must not all be attributed to TF32.
Matched spectral linear-solve time improves by
1.62x, while setup and other application work reduce the full-process benefit.

The within-block TF32/original speedups span 3.278--3.334x (median 3.300x).
All four TF32 runs also beat their matched spectral-FP32 and prior shared-FP32
controls. Four observations on one device support the reported ranges, not a
statistical confidence interval or a claim about other GPUs or geometries.
The audit retains the distinction between paired ratios and ratios of medians.

Cold preconditioner setup medians are 11.69 s for the original solver,
3.99 s for shared FP32, 4.05 s for spectral FP32 and 4.10 s for spectral TF32.
The largest sampled GPU residency across the timing runs falls from
40.28 GiB for the original solver to
27.58 GiB for spectral TF32. GPU memory was polled once per second.
Original coarse iteration totals are unavailable; spectral modes have no
coarse solve, while shared FP32 records 16,390 coarse iterations per run.

The retained `analysis-final-timing.json` includes every run, resource record,
source hash, gate result, median/range and paired ratio. It reports `complete`
with sixteen valid measured runs and no invalid runs.

## Outcome and remaining scope

The promising implementation is the reusable global reference inverse applied
through four large cuBLAS TF32 GEMMs, with FP32 Euler contractions and current
FP64 acceptance. The larger WMMA patch experiments passed accuracy but added
coarse work; they supplied no performance reason to replace the shared FP32
control. The three-product TF32 mode passed accuracy and recovered the FP32
iteration count, but its extra GEMMs made it slower than raw TF32 in the
qualification/profile observations, so it was not selected for the repeated
benchmark.

The next optimization targets are per-stage system formation, current-operator
work, and the remaining packing/nodal transforms. The large GEMMs already use
tensor cores effectively. General curved geometry, other polynomial orders,
DGSEM/HDG, multiple GPUs, longer physical horizons, and a fully FP32 state/outer
solver require separate validation. No production default or merge was changed.

## Final source and reproduction

The measured algorithm and performance binary remain v5/commit `2ae2661`.
The final v6 change only corrects the cuBLAS hardware-execution setup label to
`requires-profile-proof`, including FP64; it changes no numerical operation.
Canonical build/native job 3400590 passed all seven suites in 18.78 seconds.
Its 770-file source map matches the final application source; binary SHA256:
`dcf88fe1693c6826d578a53c0073caa4f6963b7a3c408b1d25ca57bc341ae7e0`.
Large-case timings and physical comparisons were not rerun for this log-only
change and are attributed to the frozen v5 executable above.

The branch is `igr-tensor-multigrid-deltaai`. Reproduction scripts, manifests,
comparisons and audits are retained under
`/home/jarias9/.cache/igrlab-experiments/large-dg-20261017` and remotely under
`/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/large-dg-v1`.
The bundle's `README.md` maps source versions, jobs, native fields, profiler
captures, and the frozen case/environment definitions. Raw large state files
and profiler captures remain on DeltaAI. Slurm scripts reserve one GPU at a
time on `bdiy-dtai-gh`/`ghx4` with the `projects` constraint; builds use the
application's canonical `build.sh` driver.
