# DG/DGSEM tensor-pipeline hardware counters on DeltaAI

> These hardware counters describe the earlier small prototype workload.
> They do not establish bottlenecks or speedups at HPC scale. See the
> [fixed large DG investigation](igr_large_dg_tensor_optimization.md) for
> the large-case profiles and subsequent code changes.

Date: 2026-10-17. This supplements [the end-to-end and physical results](igr_tensor_pipeline_deltaai_results.md); it changes no application source or defaults.

The sampled kernels support prioritizing the patch smoother and coarse solve.
TF32 transforms beat FP64 but lose to FP32 in these captures; small tiled products
exhibit little tensor-pipe activity. Coarse CG has too few blocks and very low
scheduler issue activity. The fine-level residual
records DRAM read volume approximately equal to its 101.25 MiB FP32 coefficient
footprint; attributing most of those reads to coefficients is a source-supported
inference. The FP64 WMMA volume kernel
shows a useful local improvement, but was only about 33 ms of the earlier n64 DG
profile, versus 331 ms of transforms, 315 ms of coarse CG and 214 ms of residuals.
These earlier Systems totals establish priorities; new NCU replay durations do
not replace unprofiled process timings.

## Collection and validity

One NVIDIA GH200 120GB, 132 SMs, 60 MiB L2, on DeltaAI; account `bdiy-dtai-gh`,
partition `ghx4`, constraint `projects`, one GPU/8 CPUs/48 GiB host memory per job.
NCU `2025.2.0.0`, build `35613519`, kernel replay, caches flushed between passes,
no profiler clock locking, one application process and logical CUDA device 0.
All captures retain the full 100-step, dt=1e-4, p5 smooth periodic workload with
original-current FP64 IGR acceptance. Each admitted run completes 100 accepted
steps, 300 production RHS calls, 301 converged IGR solves, zero rejected steps,
zero coarse caps/failures and the native conservation checks. Field budgets are
inherited from the exact-binary, exact-control qualifications in the linked
report; these output-free captures do not redo field comparisons.

Launches are selected by exact kernel family/precision, matching invocation
ordinal and measured grid/block dimensions. Each transform/spatial/volume case
has two samples, each coarse case three and each residual case one. These are
later invocations, not a distribution across the trajectory or proof of steady
state. Different precision trajectories can reach different stages at the same
MG ordinal. Coarse iteration counts are not available per captured launch, so
its n32/n64 durations must not be interpreted as equal-work scaling.

The matrix collects launch, occupancy, speed-of-light, compute, scheduler,
warp-state and instruction sections. The continuation also collects detailed
memory tables. Three MG cases are repeated specifically for those added memory
counters; their durations are kept as a separate cohort. The
[NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
explains replay/cache effects and why stall reasons need scheduler context.
Active-cycle occupancy, elapsed-cycle tensor activity, and stall ratios have
different denominators. None is a percentage of application wall time.

## Multigrid transforms

Selected n64 cold-cache replay durations, microseconds; values are medians of two
launches, with their observed ranges. These are diagnostic kernel measurements.

| Level | FP64 SIMT | FP32 SIMT | TF32 WMMA |
|---|---:|---:|---:|
| p5 | 23.824 [23.680, 23.968] | 13.408 [13.344, 13.472] | 16.848 [16.768, 16.928] |
| p3 | 8.944 [8.864, 9.024] | 6.864 [6.688, 7.040] | 8.192 [7.936, 8.448] |

TF32 p5/p3 achieved active warp occupancy is 51.88%/35.17%, versus
81.62%/78.13% for FP32. TF32 uses 56 registers/thread versus FP32's 32.
Type-specific tensor activity is only 2.75%/1.23% of elapsed cycles. Scheduler
issue activity is 45.45%/29.91%, versus FP32's 61.03%/54.27%.
The p3 TF32 grid has only 0.65 resident-capacity waves; p5 has 1.94.
This is not evidence of a saturated tensor pipeline.

Source-derived useful/padded WMMA work is 42.1875% for the 18×18 p5 transform
and 56.25% for the 12×12 p3 transform. N columns are all real patch lines with no
N-tail padding here. Increasing mesh size does not improve M/K padding. Four
independent warp tiles per block repeatedly stage and convert the small shared
operator and RHS panels. Padding, staging, occupancy and memory dependencies
are plausible contributors; the measurements do not isolate each contribution.

The added memory-table cohort measures essentially equal p5 DRAM reads:
5,316,096 bytes for FP32 and 5,317,632 for TF32. TF32 additionally records about
229,162 shared-load and 221,767 shared-store bank conflicts per launch, while
FP32 uses no shared memory. The p3 TF32 sample records about 52,431/73,728.
These justify testing shared-memory layouts and operator reuse, but do not
quantify the runtime recoverable by removing conflicts. The memory cohort's
p5 FP32/TF32 medians are 13.456/16.624 µs and p3 TF32 is 8.064 µs; these are
reported separately from the initial cohort.

## Coarse solve and residual

Coarse CG launches only 64 blocks at n64 and 16 at n32, each 256 threads, on a
132-SM GPU. At most that many SMs can hold its blocks simultaneously; actual
placement was not measured. Achieved warp occupancy over active cycles is
12.40%/12.55% for the n64/n32 medians (individual samples 12.35–12.71%),
with only 0.04–0.05 eligible warps per scheduler and about 4%
issue activity. It uses 111 registers/thread, allocated as 112, limiting potential
residency to two blocks/SM, but the grid is already smaller than one block/SM.
Reducing registers alone cannot fix that shortage of parallel work.

Median durations are 136.800 µs at n64 and 124.320 µs at n32 (three samples each).
These are variable-work solves, not per-iteration timings. The barrier stall
ratio is about 27–30 per issued instruction in the first samples, with further
long-scoreboard waits. This supports investigating synchronization and sparse
work organization; it does not prove how much time grid barriers alone consume.
The current persistent sparse CG is not a direct GEMM substitution. More
coarsening, a stronger reusable coarse approximation or a different solver may
be useful, but each changes the convergence/performance tradeoff and needs tests.

The n64 p5 residual takes 40.640 µs in its sampled cold-cache launch, reads
108,132,352 DRAM bytes and reaches 67.94% reported DRAM throughput. Its L2 hit
rate is 16.24%; eligible warps are 0.17/scheduler and issue activity 13.59%, with
a long-scoreboard ratio of 56.62 per issued instruction. These jointly support
memory latency/traffic as a substantial constraint for this sample. The p3
residual takes 16.096 µs, reads 21,585,152 bytes and reaches 33.44% DRAM
throughput. Cache flushing matters: the p3 coefficient footprint fits L2, so
these results do not establish the production warm-cache traffic. Do not replace
this with a blanket claim that every level is bandwidth-saturated.

## Volume and shared spatial maps

The n64 FP64 volume action takes 21.888 µs with SIMT and 14.448 µs with WMMA
(two-sample medians). WMMA DMMA activity is 11.24% of elapsed cycles. Its shared
memory still records about 204,596 load and 589,824 store bank conflicts per
launch. A local improvement and imperfect memory layout coexist; raw conflict
counts alone do not determine the next application-level priority.

Shared Euler contraction medians in µs, two samples per entry:

| Method/mesh | FP64 SIMT | FP32 SIMT | TF32 WMMA |
|---|---:|---:|---:|
| DG n32 | 12.336 | 12.096 | 27.888 |
| DG n64 | 34.256 | 33.808 | 29.104 |
| DGSEM n32 | 5.920 | 5.888 | 7.360 |
| DGSEM n64 | 9.952 | 11.536 | 15.216 |

DG TF32 has only 0.16/0.65 resident-capacity waves at n32/n64 and
9.09%/36.13% active occupancy. Its type-specific tensor elapsed activity is
0.42%/1.60%. It benefits from larger batching at n64 in these samples, while
its earlier Systems aggregate saving versus FP64 was only about 2.1 ms over
300 contractions. This remains no demonstrated material end-to-end TF32 gain.

The DGSEM map has only K=6 and 28.125% useful/padded WMMA work, versus DG's
36×72 map at 75%. DGSEM TF32 is slower than FP64 at both sampled sizes, with
0.77%/1.40% type-specific tensor elapsed activity. Shared DG/DGSEM input/output
storage remains FP64 for all three contraction modes, so reduced arithmetic
precision supplies no automatic halving of those global-array bytes. Both TF32
spatial kernels also record shared-memory conflicts. These observations support
keeping shared DG as the structural improvement and the existing fused DGSEM
spatial path; they do not isolate a single cause of the full-path difference.

## Recommended next prototype

Precompose the existing shared separable patch inverse into its center-response
rows, once per p-level. The current smoother only uses the center element's
correction after four full-patch transforms. Its reference factors and clipped
inverse diagonal are shared across elements; the element-dependent scalar and
0.8 damping remain in the final update. Existing patch packing already produces
one genuine residual patch per dense input column.

| Calculated work per patch | p5 | p3 |
|---|---:|---:|
| Dense center-response map | 36×324 | 16×144 |
| Existing four-transform useful FMAs | 23,328 | 6,912 |
| Dense-map useful FMAs | 11,664 | 2,304 |
| Dense-map useful/padded TF32 tile work | 74.085% | 100% |

A pack → dense map → scaled center-add implementation would reduce smoother
launches from 36 to 12 per V-cycle and remove intermediate transpose/divide
passes. These are source-derived counts, not measured improvements. The dense
map represents the current separable preconditioner. It preserves that map in
exact arithmetic, but changes rounding and potentially outer iteration counts.
Construct the center rows in FP64 from the same factors, clipping and diagonal
indexing; compare dense FP64, FP32 and TF32 against their factorized controls.

The critical layout risk is lower parallelism: reusing four warp tiles per block
would give only 48/192 p5 blocks and 16/64 p3 blocks at n32/n64, while K grows to
324/144. A new tile/block schedule is necessary to investigate; one-warp blocks
or split-K are candidates with their own costs. The fewer operations do not by
themselves imply a faster kernel. Longer dot products, cancellation in the dense
response, range and coefficient changes all need numerical qualification.

Keep the original-current FP64 residual acceptance, physical accuracy budget,
and equivalent FP32/SIMT controls in the next prototype. Measure the complete
patch update including packing/output handling and then the full solver; a
faster GEMM alone would not establish an application gain. Retain the existing
DGSEM fused Euler route unless a full spatial-path comparison reverses its
previously measured advantage. There is no basis here for changing precision
defaults or claiming shock/cylinder, curved-cell, MPI or general-p support.

## Source, jobs and evidence

Application commit `61938a7a8f7f21aef74ada6ade3f77308a1653d3`; executable SHA256
`8209ec67ad4b95fe0d6ba2d95b2fb7b2c8e926e75a9541ab522e611c244cf74a`.
The 60-file source payload matches that commit; the documented remote base plus
overlay and two documentation-only base differences remain unchanged.
Overlay manifest SHA256
`dee3081a947cdc7083b7e72cbc668c58b6be144bcdf76058d8bf8555fff41aa1`.

The admitted set has **27 capture cases / 54 launches**, including three
additional MG memory-table cases. All completed captures pass their native
scientific controls; the largest native relative conservation defect is
4.046e-13 against a 5e-10 limit. Raw CSV SHA256 values were verified locally and
all exports re-parsed to exact equality with the recorded per-launch metrics.
The CSV reader passed 32 host tests; independent source/shape, admission and
interpretation reviews were performed. No local application build was used.

| Job | Outcome and scope |
|---|---|
| `3398984` | Failed preflight before an application launch: inherited build-only `IGRLAB_SPACK_ENV` was incorrectly treated as a runtime experiment selector. Exact-key exemption records it as build metadata. |
| `3398988` | Application and counter capture succeeded; job failed at the CSV reader's long/wide schema assumption. Saved CSV passed corrected offline parsing. Original failed receipt retained. |
| `3399000` on `gh005` | Eight completed cases / 18 admitted launches. Ninth application/capture succeeded, but job failed because NCU's exported residual name shortened the anonymous namespace. Original failed receipt retained. |
| `3399015` on `gh089` | Completed 19 cases / 36 launches, exit 0. Namespace-qualified capture filter retained, separate exact-template export matcher plus measured launch-shape checks added. Full end audits passed. |

Initial full-section captures used 20 replay passes; memory-table captures used
34. Driver `595.71.05`. Before/after GPU snapshots are retained, but do not
measure clocks, power or thermals throughout each capture. Different metric
cohorts/nodes are not pooled as repeated timing samples. The failed residual
capture and smoke are excluded from the 54-launch analysis.

Remote root: `/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016`.
Raw NCU reports and application logs remain below `igr-ncu-results/`, with
immutable helper bundles in `ncu-tools-20261017-v3/` and `ncu-tools-20261017-v4/`.
Helper archive SHA256 values:

- v3: `3044ee6c0a2dad215f0b8f5e4377b8795d1262b5cf0f49b6567a83b156a9495e`.
- v4: `c207fd52af2488a1daa4dcf2e77aaad36568f9aba03bb33e2167618cfdcce50d`.

Local scripts, source hypotheses, exact counter exports, compact receipts,
submission receipts and derived tables are retained under
`/home/jarias9/.cache/igrlab-build/evidence/igr-ncu-20261017`.
The remote report path/hash, exact command/environment, source/binary/mesh and
qualification identities, native checks and all per-launch metric names, units
and values are recorded in the corresponding receipts.

A generic tensor-active counter reports small nonzero values even in SIMT
controls. The type-specific HMMA-family counter prints zero there and positive
values in TF32 kernels. The discrepancy is unresolved; generic activity alone
is not instruction proof. Exact-symbol SASS from the earlier campaign remains
the TF32/FP64 tensor-instruction evidence. No dynamic tensor instruction total
or useful tensor-FLOP efficiency is claimed here.
