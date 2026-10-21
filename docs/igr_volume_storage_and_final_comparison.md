# Direct IGR volume storage and final cumulative Sedov comparison

Steps 1–3 are complete. On the same GH200 GPU, the cumulative solver reduced
full application time from **4,260.630 s to 688.339 s: 6.18973× speedup and
83.8442% less time**. Sampled peak process GPU memory fell from **39,506 to
18,708 MiB: 52.6452% less**. Five matched intervals without VTK saves were
9.88939× faster. Final conserved-field relative L2 differences are at most
3.54206e-9. All observed and final actual FP64 RHS-relative residuals pass
1e-6, with the original zero-iteration extra-certificate exception disclosed
below. This is one original-first pair on one physical GPU.

The last storage pass removes 736 MiB of persistent arrays from the previous
collocated volume implementation. Its final current VTK is byte-identical to
that preceding cached-volume run. The new final pair measures the entire
cumulative pipeline; it does not isolate a physical-run speedup from storage
alone or establish a general-mesh solver default.

## Implementation and scope

Branch `igr-volume-storage-opt`, tested source `e6be5c69250e4db415766216b06119bf5cfccd91`, follows the collocated volume checkpoint `d59f056a3755f7dcec51bcad643db9de5acb3cf7` / production `3125e796afc600d622454856f94ff99ef665439a`. All prior cumulative DG/IGR optimizations are retained. General-mesh/cylinder portability, DGSEM/HDG generalization and dev/main integration remain deferred.

The optional `IGRLAB_IGR_TENSOR_VOLUME_STORAGE=direct` mode (legacy `DG_IGR_IGR_TENSOR_VOLUME_STORAGE`) requires admitted collocated FP64 SIMT. Cached storage remains the default and is retained for unsupported combinations. Direct storage reads current quadrature-function objects on each application, stores compact immutable geometry, and eliminates its owned weighted-coefficient arrays and GPU coefficient-refresh kernel. Refresh validates and rebinds QF objects; it never retains their raw data pointers. Geometry tiles may share storage only after bitwise equality, including signed zero. Separate FP64 geometry-times-coefficient products preserve the former cached intermediate rounding. The original FP64 current operator remains the certificate authority; TF32 spectral GEMMs are unchanged.

## Native validation and isolated operator measurements

Canonical DeltaAI build/test job 3428469 on gh007 completed all commands successfully. The existing CUDA volume suite now covers direct/cached/public-PA/independent-dense comparisons, GL/GLL p1/p3/p5/p7, curved and unequal-rule fallback, nonunit actual B diagonals, coefficient-object rebinding, live buffer replacement, device-only updates, nonfinite visibility and finite recovery. Native proofs include eight admitted direct setups and 26 explicit direct-to-cached fallbacks. The independent dense oracle is unchanged.

The actual 512² p5 benchmark uses 262,144 elements, 9,437,184 scalar DOFs, four warmups, 64 repetitions and two nonuniform coefficient generations. Input and outputs have distinct owning buffers. Setup, coefficient generation and oracle downloads are excluded from apply/refresh timings.

| Mode | Mean apply, ms | Additional refresh, ms |
|---|---:|---:|
| Public PA | 0.580340 | Common PA refresh 0.258918 |
| Generic fused FP64 SIMT | 0.823996 | 0.342038 |
| Fused FP64 WMMA | 0.479325 | 0.340887 |
| Collocated cached FP64 SIMT | 0.226870 | 0.340969 |
| Collocated direct FP64 SIMT | 0.257417 | No GPU refresh kernel |

CUDA events measure application and cached refresh intervals; those intervals can include host submission gaps. Direct refresh is host metadata work. Its measured 13.5 ns/call is noise-scale and is not presented as a precise production cost. Common PA assembly/refresh remains for FP64 certificates.

The actual Sedov geometry did not satisfy whole-mesh bitwise sharing. Direct mode therefore retained all compact per-element values: 288 MiB geometry, 36 MiB DOF map and 3,072 bytes of basis maps, totaling 339,741,696 bytes. Compared with the previous 1,111,493,632-byte fused instance, this removes 771,751,936 bytes (736 MiB). These are owned persistent-array counts, not whole-process sampled peaks. The unadmitted all-shared geometry case would use approximately 36 MiB but is not the measured Sedov configuration.

Direct application is 27.38–33.71 microseconds slower than cached across the two generations. Removing approximately 341 microseconds per refresh offsets this at the preceding case's 5.492 applications per binding: the model predicts 156–191 microseconds saved per binding, or 2.31–2.82 seconds over its 14,818 bindings. Break-even is approximately 10.1–12.5 applications per binding. This is a workload-based estimate, not an isolated physical-run speedup. The final original/current pair measures the entire cumulative change.

All five variants pass both large coefficient generations. Across them the maximum relative L2 difference is 8.978e-17 and sampled vector Linf difference is 3.584e-16. Focused direct/cached comparisons are identical where reported. Independent source, native-test and artifact reviews found no remaining issue.

## Fresh profiling

Systems job 3428494 on gh008 continuously evolved the tested current binary,
then recorded 15.004265 s after a nominal 466 s delay. The application was
intentionally terminated after the bounded diagnostic capture. The independent
SQLite analysis uses actual capture start/stop bounds and unions overlapping
kernel, memcpy and memset intervals. Recorded GPU activity covers 98.0263% of
the recording and 98.3350% of 178 complete RK ranges; their mean duration is
83.4228 ms. This measures activity presence, not SM utilization.

Every complete RK range has the same observed launch counts: 18 direct volume
applications, three solves, six original PA mass/diffusion kernels, 42 compact
face kernels and 12 spectral inverse applications. Thus this interval has six
direct applications per coefficient binding, below the native model's 10.1–12.5
break-even range. No fused coefficient-refresh kernel is recorded. One
32.443 ms inter-step gap contains pressure projection; other inter-step gaps
are at most 4.992 microseconds. The delayed trace does not encode absolute
RK indices. All 10,850 emitted FP64 certificates pass, with maximum observed
RHS-relative residual 9.99286e-7; those logs span the diagnostic evolution,
not just the 15-second capture.

| Observed kernel group | Summed kernel time share |
|---|---:|
| Compact IGR face record/gather | 17.75% |
| TF32 cuBLAS GEMMs | 13.08% |
| Euler tensor weak divergence/packed kernels | 10.00% |
| Euler tensor face trace/flux/lift | 10.00% |
| Volume quadrature interpolation | 7.90% |
| IGR face density/coefficient preparation | 6.87% |
| Direct FP64 volume application | 6.16% |
| Spectral nodal conversions | 4.92% |
| Krylov basis/solution algebra | 4.04% |
| Norm/dot/other reductions | 3.63% |
| Original PA volume certificate cores | 1.84% |

These are selected disjoint groups from the recorded kernel-time total, not
whole-application elapsed partitions or fractions of application FLOPs. The
full census retains all groups. Systems reports possible CUPTI event loss
(driver 13.2 traced with CUDA 12.9 instrumentation), so counts and shares are
observational, not a complete-event guarantee. Existing VTK/Analysis NVTX
coverage is incomplete; absence of those names is not evidence of absence
of output or callbacks. The original FP64 certificate volume path remains
present. TF32 precision/operation selection is unchanged; this pass does
not introduce new Tensor Core work.

Compute job 3428521 on gh114 ran a fresh continuous evolution, selected the
reviewed RK/Step ordinal 3200, and captured one direct-volume launch before
its intentional diagnostic kill. Its last completed native certificate is
step 3198/stage 2; the interrupted later solve has no completed-certificate
claim. All 9,597 emitted certificates pass. The native kernel name,
65536×1×1 grid, 128×1×1 block, 3,840-byte static shared storage and zero
dynamic shared storage match the Systems configuration. All requested
metrics were collected; no unsupported/missing counter is treated as zero.

| Direct-volume counter | Measured value |
|---|---:|
| Profiled kernel duration | 271.488 microseconds |
| DRAM read / write bytes | 566,272,768 / 73,575,168 |
| DRAM throughput / peak sustained | 58.6% |
| L2 traffic bytes | 980,385,184 |
| Active warp occupancy | 70.16% |
| Registers per thread | 40 |
| Register-limited resident blocks | 12 |
| Long / short scoreboard stall percentages | 49.31% / 17.8% |
| Math-pipe / MIO throttle percentages | 1.32% / 2.71% |

The read/write byte rate is approximately 2.36 TB/s for this replayed sample.
Together, the counters point to memory dependencies rather than saturated
arithmetic pipelines; 58.6% DRAM throughput does not establish pure bandwidth
saturation. The 12 register-limited four-warp blocks imply a 75% theoretical
warp occupancy bound, close to the achieved value. Stall percentages refer
to active warp cycles and must not be added as elapsed-time partitions.
Kernel replay with cache flushing perturbs caching; this duration is not
an unprofiled application timing or a matched cached/direct counter comparison.

The profiles support proceeding to the final comparison. Remaining time is
spread across face operations, spectral inversion, coefficient/interpolation
work and Euler operations. This is evidence for future prioritization, not a
reason to change the frozen candidate during the final original/current pair.

## Final original/current physical comparison

Job **3428588** completed both applications sequentially on **gh019**, GPU
`GPU-0da510af-f562-1f2f-f290-a4719fad7cb5`. Original ran first. The fixed case
has 512² elements, p5 ordinary DG, **37,748,736 conserved DOFs** and
**9,437,184 scalar IGR DOFs**, L2/BrokenL2, RK3 SSP/every-stage sigma,
CFL 0.3, alpha=h²/6, t=0.1, RHS-relative rtol=1e-6 and atol=0, and VTK
cadence 0.02. Optional nonfinite/admissibility scans, positivity limiting
and modal damping remained off. Arguments other than executable and the
common environment match; intentional backend overlays are frozen in the
case specification. One GPU, eight CPUs, 96G host memory and OMP=1 were used.

| Measurement | Original GPU paths | Cumulative direct-storage pipeline |
|---|---:|---:|
| Full application wall time, s | 4,260.630241 | 688.338520 |
| Sampled peak process GPU memory, MiB | 39,506 | 18,708 |
| Accepted timesteps | 4,939 | 4,939 |
| Converged linear solve calls | 14,818 | 14,818 |
| Total outer Krylov iterations | 232,116 | 51,742 |
| Mean / maximum outer iterations per solve | 15.6645 / 36 | 3.49183 / 5 |
| Maximum observed actual relative RHS residual | 9.9992402e-7 | 9.9954522e-7 |
| Final committed-state actual relative RHS residual | 8.87195350e-7 | 3.61564397e-7 |

The full process saving is **3,572.291721 s**. Sampled memory saving is
**20,798 MiB (20.31055 GiB)**. Memory is sampled at one-second intervals and
may miss shorter peaks. Application time includes fresh setup, normal output
and finalization. Relative to the preceding cached-volume run's 19,448 MiB,
the new observed peak is 740 MiB lower; that cross-job observation is distinct
from the exact 736 MiB reduction in the operator's owned arrays.

The original is the retained original GPU path selected in frozen scan-off
source `c13928be284a511a1d170a2b9a5b3cd27cc0fac0`, not a historical dev/main
checkout. Both use CUDA FP64 outer FGMRES. Original preconditioning uses
FP64 element blocks and an order-1 coarse space with cooperative CUDA FP64
CG; current preconditioning uses the frozen global-reference spectral TF32
inverse. Both are built once and reused. Current also retains FP32 SIMT
Euler tensor lines, tensor faces, coefficient reuse, compact face data and
lifts, spectral nodal conversions and the new FP64 direct volume mode.
Conserved state, sigma, current IGR operator, Krylov arithmetic and residual
authority remain FP64. Selecting original modes in today's executable would
still inherit newer lazy diagnostic-gradient preparation, so the pinned
original is needed for faithful cumulative attribution. See the earlier
[three-way comparison](igr_sedov_original_gpu_comparison.md) for the original,
FP32 spectral and TF32 spectral control definitions; this new pair does not
repeat or re-estimate the isolated FP32-versus-TF32 precision comparison.

### Compute intervals, timers and system observations

The independent review identifies the same five disjoint checkpoint windows
in both native logs: steps 1000–1500, 2000–2500, 3000–3500, 4000–4500 and
4500–4939. No VTK save occurs between their bounding checkpoint timestamps,
as checked by stdout marker order. They span 2,439 steps and 7,317 stage
certificates in each run. Elapsed time falls from **2,010.022775 to
203.250514 s**, a **9.88939× speedup / 89.8881% reduction**. Their iteration
counts are 125,791 and 28,369. These intervals include analysis callback,
logging and intervening host work; they are not GPU-only timings. The larger
compute-interval ratio shows why a single full-process ratio should not be
used to describe all parts of execution. No elapsed=0 checkpoint is invented.

| Native host interval, s | Original | Current |
|---|---:|---:|
| IGR solve | 1,308.508299 | 236.373091 |
| IGR total | 3,686.450209 | 360.493900 |
| Update total | 3,693.754893 | 367.644318 |
| Assembly | 2,351.226554 | 107.270358 |
| Form | 10.139605 | 13.647642 |
| Preconditioner setup | 16.575752 | 3.202809 |

These native host intervals are nested and can absorb deferred GPU work;
their savings must not be added or called independent kernel-time savings.
There is no separate native Euler timer in these logs, so a remainder is not
labeled as measured Euler time.

Mean sampled SM clocks at utilization >=90% were 1917.760 and 1912.753 MHz
(current 0.261% lower). Clocks were dynamic and sample populations differ;
this is descriptive, not a causal frequency correction. Host monitor maximum
gaps were 0.10854/0.10056 s, with no >1.5 s gaps or sampler/query failures.
Both runs have filesystem-wait samples, and the largest current GPU query
took 1.77226 s. Responsive monitoring does not rule out application/storage
stalls. This is one matched pair with no confidence interval, and historical
incremental speedups are neither added nor multiplied into this result.

### Residual conventions and complete final fields

Original has 14,817 extra certificate records for 14,818 calls. Their
iteration sum exactly accounts for all 232,116 iterations, proving that the
omitted initial call 1 had zero iterations; native initial FP64 b−Ax stopping
still applies. Its formal caller/final metadata permits factor five, while
all observed and final actual relative RHS residuals are <=1e-6. Current has
all 14,818 independent current-A certificates at factor one. Neither run
has a residual-replacement retry. Extra post-solve work is not identical
for every call across the two backends.

After all timed measurements ended, complete final files were hashed and
bound to the new run identities. The new original VTU differs from the
archived original, so it was downloaded. Current matches the preceding
cached-volume VTU exactly and was reused only after verifying its full hash
against the new remote run. Both files contain 1,160,775,042 bytes.

The unchanged native field readers check all 262,144 cells and compare the
saved Q5 polynomials at GL6×6: 9,437,184 points per scalar field. Geometry,
alpha and final time agree. L2 polynomial differences are integrated exactly
up to FP64 roundoff on the common affine cells; Linf is a maximum sampled at
GL points, not a continuous polynomial supremum.

| Final field | Relative L2 difference | Relative sampled Linf difference |
|---|---:|---:|
| Density | 7.65516e-10 | 9.23950e-9 |
| Momentum x | 3.54206e-9 | 2.00518e-8 |
| Momentum y | 3.54151e-9 | 1.95142e-8 |
| Energy | 5.87257e-10 | 4.79325e-9 |
| IGR sigma | 3.34685e-7 | 1.66199e-6 |

All conserved components pass the existing engineering budgets of relative
L2 <=1e-3 and sampled Linf <=1e-2. Sigma has no assigned budget; its absolute
L2 and sampled Linf differences are 6.70095e-12 and 9.39510e-11. Derived
pressure/speed relative L2 differences are 5.24116e-10 / 3.67396e-9; those
nonlinear derived norms are quadrature estimates. These are numerical
implementation comparisons, not analytic Sedov error, conservation drift
from the initial condition, or an intermediate-state/positivity-everywhere
claim. Byte identity with the prior cached-volume run covers its complete
saved final fields and geometry, not every unsaved state.

## Campaign decision

Keep the direct storage candidate opt-in and carry this cumulative checkpoint
into the planned integration review. Native validation, fresh profiling and
the final matched original/current comparison are complete. No defect or
additional optimization requirement emerged that blocks that handoff.
Remaining face, interpolation/coefficient and Euler work can be prioritized
in later passes. This result applies to the fixed p5 Cartesian, single-GPU
case; the global spectral backend's geometry/topology restrictions remain.
No defaults, PRs, merges, cylinder portability or other solver families were
changed as part of this pass.

## Identities and evidence

- New application SHA-256: `e1fec53d5f2202158da8226bdbc454391338656dce6cfeda1c54345c3a5d9957`.
- New native test SHA-256: `f68052d9da9676c2192d293387c2d2328bfb8442f2b281e313ce5457be700efd`.
- New build receipt SHA-256: `ccd62e7cbacb892ca7cf50ab751863ef567fe75c9624022d3d8173818366d7b8`.
- Final original VTU SHA-256: `566e6e8910729a4b9e000f12676423d832fb5338571e01e96f421c6d0aab202d`.
- Final current VTU SHA-256: `81d8bf54ddec7ab4f0b6293d77283d6c1ee0f9d19304c1b04f6846bc191debb3`.
- Final case specification SHA-256: `3bb8ee39be2da3c26243b1b362e6cf538e9dea5e22b49faec1316e9757cf38c0`.
- Original application SHA-256: `c6f91d544b831499ebe96dbcf35306aa6e145d83e1c6333c46f8ec115a145d42`.
- Original build receipt SHA-256: `a6dd2612a6cb8172c168cd61d004e8ebfef9e4fa3ff77985143f28858ddbf529`.
- Remote campaign: `/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/igr-volume-storage-20261021`.
- Durable local campaign: `/home/jarias9/.cache/igrlab-experiments/igr-volume-storage-20261021`.

Native job helpers retain their original hashes separately from a subsequent runtime-only compatibility correction for the older original build-receipt schema. No application change or GPU rerun was required for that helper correction.

Within the retained campaign, `comparison-3428588/comparison.json` preserves
the primary timing/residual checkpoint; its field-pending marker describes
that checkpoint and is intentionally not rewritten. `field-comparison.json`
and the conserved/sigma norm reports complete the field assessment.
`analysis/timing-telemetry-review.json` and `independent-assessment.json`
record the independent full-run review. Separate `nsys-3428494/analysis/`
and `ncu-3428521/analysis/` directories retain the reviewed profile evidence.
Complete final VTUs, collection receipts, native build logs and source/config
hashes remain available. All GPU jobs used one GPU at a time.

Slurm reports all four jobs completed: native 90 s, Systems 499 s, Compute
475 s and the final pair 4,964 s. This pass used **6,028 allocated GPU-seconds
(1.67444 GPU-hours)** on one GPU at a time. This is not the historical campaign
or account-wide usage total. No GPU jobs remain active.
