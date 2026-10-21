# Sedov: original GPU paths versus the current optimized pipeline

The matched V2 comparison reduces full application time from **4,254.268 s
to 887.865 s**, a **4.79157x speedup / 79.1300% time reduction**, using the
optimized TF32 pipeline. Sampled peak process GPU memory falls from
39,506 to 22,444 MiB: **17,062 MiB / 43.1884% less**. These are completed
timing and solver-audit results. Full conserved-field comparisons pass the existing
engineering budgets; the largest original-to-TF32 conserved relative L2
difference is **3.54403e-9**. Fresh Nsight Systems attribution and bounded
native counters identify TF32 Tensor Core execution in sampled cuBLAS
configurations; counter coverage is partial.

## Matched case and measurements

Ordinary DG on 512 by 512 affine Cartesian elements, p5 Gauss–Legendre,
37,748,736 conserved coefficients and 9,437,184 scalar IGR coefficients.
Use one MPI rank on one GH200 GPU, L2/BrokenL2 IGR, Euler slip walls and
natural scalar boundaries. Preserve the Gaussian Sedov initial condition,
RK3 SSP with sigma updated every stage, CFL 0.3, physical alpha=h²/6
(6.357828776041666e-7), final time 0.1 and VTK cadence 0.02. CUDA FP64
FGMRES uses RHS-relative rtol 1e-6, atol 0 and maxit 1000. Limiting,
modal damping and optional nonfinite/admissibility scans are off.

Job **3407624** ran the three variants sequentially on **gh009**, GPU
`GPU-f0c4e4dd-0b89-e8d0-c0a8-953f5c447d5b`, with one GPU, eight CPUs,
96G host memory and OMP_NUM_THREADS=1. Each row is **one observation**, in
original/FP32/TF32 order. Wall time includes fresh setup, VTK and finalization;
there is no repeat-distribution or confidence-interval claim. Process GPU
memory is sampled once per second and may miss shorter peaks.

| Metric | Original FP64 GPU paths | Optimized spectral FP32 | Optimized spectral TF32 |
|---|---:|---:|---:|
| Full application wall time, s | 4,254.267724 | 1,082.192001 | 887.865337 |
| Sampled peak process GPU memory, MiB | 39,506 | 22,296 | 22,444 |
| Accepted / rejected timesteps | 4,939 / 0 | 4,939 / 0 | 4,939 / 0 |
| Converged linear solves | 14,818 | 14,818 | 14,818 |
| Total outer Krylov iterations | 232,116 | 47,514 | 51,739 |
| Mean / maximum iterations per solve | 15.664 / 36 | 3.207 / 5 | 3.492 / 6 |
| Maximum observed independent relative RHS residual | 9.9992402e-7 | 9.9987858e-7 | 9.9963081e-7 |
| Final committed-state relative RHS residual | 8.8719535e-7 | 2.4499603e-7 | 3.6153539e-7 |

Original to optimized FP32 is 3.93116x / 74.5622% less application time.
Changing only the optimized spectral inverse from pedantic FP32 to TF32
gives **1.21887x / 17.9568% less application time**, despite more outer
iterations. TF32 uses 148 MiB more sampled peak memory than the FP32 control.
Original to current measures cumulative implementation changes; FP32 to TF32
isolates the spectral precision choice, including its effect on iterations.
These ratios are not multiplied by historical smooth-wave or incremental
Sedov speedups.

## Controls and arithmetic

“Original” selects the retained original GPU operator and two-level FP64
preconditioner paths in the **same current executable**. It is not a
historical checkout. Common scan-policy and residual-authority corrections
therefore apply to all three variants.

| Canonical environment selector | Original | Optimized FP32 | Optimized TF32 |
|---|---|---|---|
| `IGRLAB_IGR_TENSOR_MG` | `off` | `fp32` | `tf32` |
| `IGRLAB_IGR_TENSOR_MG_OPERATOR` | `sampled` | `spectral` | `spectral` |
| `IGRLAB_IGR_TWO_LEVEL_INVERSE` | `fp64` | empty | empty |
| `IGRLAB_SHARED_OPERATOR_PRECISION` | `off` | `fp32` | `fp32` |
| `IGRLAB_SHARED_EULER_LAYOUT` | `dense` | `tensor-lines` | `tensor-lines` |
| `IGRLAB_WEAKDIV_SIGMA_STORAGE` | `dense` | `auto` | `auto` |
| `IGRLAB_SHARED_EULER_FLUX_EXECUTION` | `element` | `pointwise` | `pointwise` |
| `IGRLAB_DG_TENSOR_FACE` | `off` | `fp64` | `fp64` |

The original two-level inverse uses element-local FP64 blocks and an order-1
coarse space with CUDA FP64 cooperative CG. Common settings include
`--igr-preconditioner none` (the explicit environment-selected preconditioner
is installed first), `--igr-exact-dg-face`, `--no-nan-scan`,
`--no-admissibility-scan`, `IGRLAB_IGR_TENSOR_VOLUME=off` and spectral rebuild
interval 0. The case specification retains all arguments and both environment
aliases; the optimized controls differ only in spectral precision and output
path.

Each spectral inverse application packs the residual, applies **four
3072 by 3072 cuBLAS GEMMs** for the two forward and two inverse basis
transforms, scales spectral coefficients and unpacks. The TF32 selector
requests `CUBLAS_COMPUTE_32F_FAST_TF32`; the FP32 control uses
`CUBLAS_COMPUTE_32F_PEDANTIC` and `CUBLAS_PEDANTIC_MATH`. Their application /
GEMM counts are 51,739 / 206,956 and 47,514 / 190,056 respectively. Nsight
Systems observes the corresponding native TF32 cuBLAS kernel family;
sampled NN/NT configurations have raw GMMA instruction proof, with partial
counter coverage described below. No new WMMA kernel is claimed.
The TF32 run performs a nominal `4 * 2 * 3072^3 * 51739` =
11,999,734,898,098,176 GEMM FLOPs, counting a multiply-add as two operations.
This counts the four matrix products only, not all application FLOPs or a
Tensor Core fraction.

Conserved state, sigma, the current physical IGR operator, Krylov arithmetic
and residual authority remain FP64. Both optimized variants use FP32 SIMT
Euler tensor-line contractions, FP64 nonlinear volume-flux preparation and
FP64 tensor face traces/flux/lift; those Euler kernels are not counted as
Tensor Core work. Both preconditioners are built once and reused. Every
stage still refreshes the current FP64 physical IGR equation and RHS. Native
host intervals called assembly or solve can include synchronization and
deferred GPU work; they are not independent GPU kernel times, and their
differences do not establish omitted IGR assembly.

## Residual authority and scan policy

All variants enable extended independent current-equation checks with
start=1 and factor=1; TRUE_RESIDUAL remains 0 to avoid an extra preconditioner
probe. Every observed independent residual and final closure is finite and
at most 1e-6, with zero residual-replacement retries. Original caller/final
metadata retains its formal factor-5 gate (5e-6); the analysis applies the
stricter actual 1e-6 threshold to that result as well.

The original initial solve takes zero iterations and omits its extra
post-solve certificate. Its 14,817 unique certificate records account for
all 232,116 reported iterations, proving that the missing call 1 contains
no positive-iteration work. Native initial FP64 b−Ax stopping still applies.
Both spectral variants have all 14,818 certificates, including the initial
zero-iteration call. Extra post-solve check work is therefore not identical
for every call across all variants.

Source `c13928b` wires the existing nan-scan option through IGR routine
U/B/warm-start diagnostics and first-use QF diagnostics, and removes the
successful-certificate solution scan. Mandatory FP64 residual gates and
exceptional failure-recovery checks remain active. All V2 logs pass the
scan-off audit, including absence of device-qf-finite/range/sample events.
Earlier V1 records, including abandoned comparison job 3407145, are preserved
and **excluded** because that binary did not fully honor the scan-off policy.

Canonical native build **3407600** passed all five required checks:
`dg_igr_igr_diagnostic_scans`, `dg_igr_igr_diagnostic_scans_cuda`,
`dg_igr_device_krylov_cuda`, `dg_igr_dg_volume_memory_cuda` and
`dg_igr_dg_tensor_face_cuda`.

## Full final-field comparison

All final t=0.1 collection receipts, run identities, native byte counts and
SHA256 hashes were verified after GPU profiling ended. The unchanged Q5
comparison helper evaluates every saved cell polynomial at GL6×6 points:
262,144 unique cells / 9,437,184 points per comparison. It verifies common
Cartesian geometry and bijective nodal ordering. All three Euler reports
have status=ok and pass the existing conservative engineering budgets of
relative L2≤1e-3 and sampled relative Linf≤1e-2.

| Metric (relative norms use the reference norm) | Original → FP32 | Original → TF32 | FP32 → TF32 |
|---|---:|---:|---:|
| Maximum conserved relative L2 | 3.544500e-9 | 3.544028e-9 | 1.964721e-9 |
| Maximum conserved sampled relative Linf | 1.888837e-8 | 2.084068e-8 | 9.976636e-9 |
| Pressure relative L2 | 5.248301e-10 | 5.247110e-10 | 5.282094e-10 |
| Pressure sampled relative Linf | 3.497897e-9 | 3.476133e-9 | 3.822728e-9 |
| Speed relative L2 | 3.672752e-9 | 3.673555e-9 | 1.436370e-9 |
| Speed sampled relative Linf | 2.782934e-8 | 2.690746e-8 | 7.547075e-9 |
| Nonconservative sigma relative L2 | 3.171047e-7 | 3.381504e-7 | 1.086488e-7 |
| Nonconservative sigma sampled relative Linf | 1.657761e-6 | 1.679678e-6 | 3.860241e-7 |
| Sigma signed integral difference | +4.906783e-13 | +9.329056e-13 | +4.422273e-13 |

GL6 integrates squared Q5 differences and conserved/sigma integrals exactly
up to FP64 roundoff. Pressure and speed metrics are quadrature estimates.
Linf and the following minima are sampled at GL points, not continuous
polynomial bounds. Sigma is not an Euler conserved component and has no
assigned engineering budget.

| Sampled minimum | Original | FP32 | TF32 |
|---|---:|---:|---:|
| Density | 0.449420238551 | 0.449420237251 | 0.449420239038 |
| Pressure | 1.000318764492 | 1.000318764491 | 1.000318764490 |

The saved-state conservative integral differences below integrate
candidate−reference directly. The last row normalizes by the reference
integral of the component's absolute value, avoiding division by nearly
zero net momentum. These are cross-run comparisons, not conservation drift
relative to the initial condition.

| Integral comparison | Original → FP32 | Original → TF32 | FP32 → TF32 |
|---|---:|---:|---:|
| Density signed difference | −4.695198e-19 | −5.870450e-18 | −5.400931e-18 |
| x-momentum signed difference | −3.233877e-12 | −3.234848e-12 | −9.711259e-16 |
| y-momentum signed difference | −3.233487e-12 | −3.232186e-12 | +1.300909e-15 |
| Energy signed difference | +1.881021e-17 | +2.445438e-17 | +5.644167e-18 |
| Maximum absolute difference / reference GL-L1 | 1.824205e-11 | 1.824752e-11 | 7.338328e-15 |

All six local norm processes ran sequentially with `/usr/bin/python3` and
single-threaded BLAS/OMP; peak RSS was 1,110.617 MiB. This establishes saved
field agreement on the fixed affine Cartesian, p5, single-MPI-rank case,
not analytic Sedov accuracy, positivity everywhere or a newly assigned
physical error budget. Q5 cell-center previews remain separate from these
complete field norms.

## Fresh profiling and partial native counter coverage

Nsight Systems job **3407651** profiles the same optimized TF32 binary and
physical case on gh009 through successful t=0.1 completion. Only NVTX and
output location change from the unprofiled control. CUDA/NVTX/cuBLAS/cuSolver
tracing is enabled; CPU sampling, context switches and memory tracking are
off. These profiled times are excluded from the benchmark comparison above.

| Profile interval | Elapsed, s | Any recorded GPU activity / elapsed | TF32 GEMM kernel time, s | TF32 GEMM / elapsed | TF32 GEMM / GPU-active time |
|---|---:|---:|---:|---:|---:|
| Evolution: first RK start through last RK end, including intermediate output/Analysis | 776.711311 | 76.2008% | 44.726953 | 5.7585% | 7.5570% |
| Verified 50-step window, zero-based RK steps 3625–3674 | 6.4339272 | 98.7925% | 0.520475 | 8.0895% | 8.1884% |

The 50-step window has no scheduled VTK save or Analysis callback, verified
against native stdout and source scheduling. It still includes intervening
host work and gaps; it is not a sum of RK host durations. GPU activity is the
union of kernel, memcpy and memset execution intervals, clipped at the host
window boundaries. Lower GPU coverage over the complete evolution does not
by itself identify I/O as the cause of every gap.

The authoritative timeline uses native recording startTime/stopTime bounds:
startup is 67.857875 s, evolution 776.711311 s and finalization 58.038862 s.
This origin correction preserves the original kernel census. Across the
entire 902.608048 s recording, summed GPU kernel durations total
**576.378272 s**. Source-defined families
account for the following work; their percentages use that kernel-duration
sum, rather than evolution elapsed or GPU-active time:

| Native kernel family | Summed duration, s | Share of all recorded kernel duration |
|---|---:|---:|
| Current IGR FP64 diffusion-face coefficient update and record/gather | 182.255098 | 31.6207% |
| MFEM FP64 quadrature interpolation/derivatives/geometry | 71.412065 | 12.3898% |
| Spectral SIMT input scaling, pack, diagonal scale and unpack | 48.625960 | 8.4365% |
| Explicit spectral TF32 cuBLAS GEMMs | 44.731213 | 7.7607% |

The IGR diffusion-face family is distinct from the new Euler face kernels.
MFEM interpolation spans several call sites, and the spectral packing family
includes input-maximum/exponent reductions. Each launched kernel is assigned
once for source attribution; nested host NVTX durations and CUDA API waits
are not added to GPU kernel time. These measurements do not establish that
the physical IGR assembly was omitted or equate native host assembly totals
with device work.

TF32 GEMM fractions cover whole kernel intervals, including their memory
and SIMT work. They are **not** a fraction of all FLOPs, a literal percentage
of Tensor Core cycles, or an application-wide hardware utilization metric.

Nsight Compute job **3408843** on gh059 collected **159 native launches /
15 configurations**, against 171 expected launches / 16 configurations in
the bounded evolved-step plan. Native records contain 24 NN and 12 NT TF32
GEMMs; the expected TN configuration has no native sample. Console progress
names say NN/TN while the exported native records say NN/NT. No records are
renamed or counted as covering the missing configuration.

Joining only configurations with positive sampled raw instruction counts
to their matching Systems intervals gives **35.554389479 s / 4.57755526%**
of evolution elapsed and **0.413664896 s / 6.42943078%** of the 50-step
window. These cover the NN/NT configurations verified by sampled counters;
they do not mean that every matching launch had counters or measure literal
Tensor Core active time. The all-three-configuration source/API GEMM
fractions remain 5.7585% and 8.0895% respectively. Positive tensor-pipe
percentages alone do not enter this instruction-verified join.

The 36 native TF32 GEMM samples contain **53,084,160 raw GMMA instructions
and zero DMMA instructions** in total. These are sampled warp-instruction
counts summed across hardware sub-units, not scalar FLOPs or full-run counts.
Representative NN launch ID 9 takes 211.680 µs and reports 1,769,472 GMMA,
zero DMMA, tensor-pipe activity of 68.38% of peak sustained elapsed and
DRAM throughput of 28.35% of peak sustained elapsed.

For 21 current IGR `InteriorFaceDGDiffusionRecord2DWarp` samples, mean duration is 1.414403 ms,
mean SM throughput 53.5805% and mean DRAM throughput 39.7005% of peak
sustained elapsed; raw GMMA and DMMA counts are zero. Some aggregate
tensor-pipe counters are nonzero even on these source FP64 SIMT kernels.
They are not raw instruction proof, and zero GMMA/DMMA does not establish
absence of uncollected instruction categories such as HMMA. Opaque library
hardware behavior and unsampled configurations remain unresolved.

NCU uses kernel replay, cache flushing, no clock control and dynamic
pipeline boost. Its durations and per-kernel counters are diagnostic,
separate from the unprofiled benchmark and Systems timeline. The preserved
run receipt records an initial CSV-export error; a successful saved-report import
recovery produced the authoritative native metrics without another GPU run.
No separate setup counter capture was taken: the named one-time FP64 library
tensor candidates total only 8.782272 ms in Systems. Their omitted sample
does not prove absence of Tensor Core use.

## Frozen provenance and evidence

- Source: `c13928be284a511a1d170a2b9a5b3cd27cc0fac0`.
- Executable SHA256: `c6f91d544b831499ebe96dbcf35306aa6e145d83e1c6333c46f8ec115a145d42`.
- Build receipt SHA256: `a6dd2612a6cb8172c168cd61d004e8ebfef9e4fa3ff77985143f28858ddbf529`.
- Case specification SHA256: `bfa8aa9f058ae6e4dadbcb2ada82e15247c93671195ae625f37341214d4103ed`.
- Runner SHA256: `0b028ff1b67a7aa0a8ce4b9a4a83f1e20ac290b9ce7f94c294b8fd45a8fd32b8`.
- Comparison audit SHA256: `f9d4dca946a074e59b1e419f26770687af467851186396c9644cb97b3dd540cc`.
- Corrected joined Systems attribution SHA256: `f75eca77d11297ca2d92f8aa7e2f6ac5ae0820cf6675d34da03aaa4b810c23b4`.
- Systems run receipt SHA256: `887de13f11f54df91ed8262c3c82089658792c05e89bf600a8886ec0b0779421`.
- Native NCU metrics SHA256: `7fce24dc2470fce05f653247842444df7360dd53c9035d664be400e0e47e15c0`.

Evidence root:
`/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2`.
The authoritative [comparison audit](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/comparison.json)
uses the [original](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/original-result.json),
[FP32](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/fp32-result.json)
and [TF32](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/tf32-result.json)
results. The [case specification](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/case-spec.json),
[build receipt](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/binary/build-complete.json),
[native test log](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/build-3407600/test-detail.log)
and `comparison-3407624/<variant>/` directories retain the inputs and raw
measurements. The independent timing audit
checks source/binary/build/mesh/runner identity and the actual process GPU UUID.
Fresh profiling uses the [corrected joined Systems attribution](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/nsys-current-3407651/attribution-origin-corrected-with-counters.json)
and [profile run receipt](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/nsys-current-3407651/run.json),
with [native NCU metrics](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/ncu-evolved-3408843/metrics.json)
and the [postprocessing recovery receipt](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/ncu-evolved-3408843/postprocessing-recovery.json).
The full-field receipts are [original/FP32](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/fields-original-vs-fp32.json),
[original/TF32](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/fields-original-vs-tf32.json)
and [FP32/TF32](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/fields-fp32-vs-tf32.json),
with corresponding sigma receipts for [original/FP32](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/sigma-original-vs-fp32.json),
[original/TF32](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/sigma-original-vs-tf32.json)
and [FP32/TF32](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/sigma-fp32-vs-tf32.json).
The unchanged conserved-field helper SHA256 is
`4e4d58131b08d0c6fbd61fd0ef6802619d637d10f67ccb2b15a39d444d063405`;
the optional scalar helper SHA256 is
`bd4544fcea3325209cb02187aa092a652f44025d452359d60c39c27b4cc15a90`.
The [deck field summary](/home/jarias9/.cache/igrlab-experiments/sedov-original-20261018/scanoff-v2/fields-deck-summary.json)
maps the six norm receipts to per-variant minima and per-pair field norms.
Unresolved hardware evidence is not substituted with older campaign results.
