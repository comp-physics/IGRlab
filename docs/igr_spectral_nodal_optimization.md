# Spectral nodal conversion optimization

**The float-storage conversion pair is 2.1572× faster on the actual 512²
mesh. Complete final-field comparison passes the conservative accuracy
budgets, with maximum conserved-component normalized L2 difference
1.963e-9; every FP64
current-equation certificate passes. The original pair suffered baseline
pauses; its 23.24% process-time reduction is not a valid causal speedup.
A reversed-order repeat without those long pauses measured 0.767% lower
whole-process time and 4.10–5.16% lower time in matched intervals without
scheduled VTK saves. These support a compute-path benefit, but not a large
or statistically established whole-solver speedup.**

This pass targets the FP64 conversions surrounding the existing four TF32
spectral-inverse GEMMs. It follows the validated compact-face source
`14cb9ffea660053d02cfb5584db7279fa5f401f0` and retains its optimized face
preparation, FP32 Euler tensor-line path, and current-equation FP64 residual
authority.

## Implementation and admission

Candidate `01140046e5fa2a35f56bf12d2f1340350f484ab9` adds the default-off
`IGRLAB_IGR_SPECTRAL_TENSOR_NODAL` selector, with the existing
`DG_IGR_IGR_SPECTRAL_TENSOR_NODAL` alias. Its private header is shared by
production and native tests; it adds no public diagnostic interface.

The existing 36×36 nodal matrix is constructed from products of normalized
1D Legendre values. Setup verifies six distinct finite reference coordinates
on each axis, every pair of coordinates, bijective native/tensor permutations,
and all 1,296 matrix entries. Matrix comparisons allow 64 FP64 epsilons
scaled by the entry magnitude (with a unit floor). A failed proof retains
the original dense kernels. Existing shared-node and disjoint field-map
checks remain in the spectral setup.

With native data arranged in a 6×6 array X, the forward multiplication is
`Bxᵀ X By` and the reverse is `Bx X Byᵀ`. Each becomes two length-six
contractions: 432 scalar products per cell and direction rather than 1,296.
The forward kernel loads and power-of-two scales each input once.
Intermediates and sums remain FP64, with one final cast into the existing
float or double storage. Reverse recovery still divides by the operator
scale before restoring the exponent. These are FP64 SIMT kernels; the
TF32 GEMM work is unchanged.

Each candidate kernel uses one 64-thread block per cell and two 36-value
shared arrays. Source and destination permutations preserve native DOF and
global rectangular-field ordering. The nonfinite exponent sentinel writes
NaNs through the existing contract; subsequent calls use fresh inputs.
The regrouped sums can differ in rounding. The setup log records actual
selection, fallback reason and matrix-proof error. There are no new
per-application host synchronizations or diagnostic state scans.

This change does not broaden the spectral inverse's existing Cartesian
topology restriction. The factor proof and tests support GL and GLL node
sets, including permuted native orderings.

## Measurement scope

The full comparison uses the same 512² ordinary-DG p5 GL Sedov case:
37,748,736 conserved coefficients, 9,437,184 scalar IGR coefficients,
L2/BrokenL2, t=0.1, CFL 0.3, C_alpha=1/6 with max element length and physical
alpha 6.357828776041666e-7. RHS-relative tolerance remains 1e-6, with every
solve certified against the current FP64 equation. Nonfinite/admissibility
diagnostic scans and positivity/modal limiting are off. Output cadence is
0.02.

The previous fresh Systems census, job 3422519, attributed 29.304744 s to
`ToSpectral` and 9.019669 s to `FromSpectral`. Those are summed GPU kernel
durations in the preceding tensor-face baseline. The conversion source
bytes match the new compact-face baseline; new hardware sampling uses the
actual compact-face executable and a verified evolved RK window.

Native tests compare both directions separately against dense CUDA and
long-double dense oracles, covering float/double storage, GL/GLL, native and
field permutations, rectangular layouts, tails, zero, extreme power-of-two
scales, sentinel poisoning and finite recovery. The existing inverse suite
also runs with the selector off and on, retaining independent current-A
checks across supported inverse precisions.

The actual 512² microbenchmark measures forward, reverse and combined
conversions using four warmups and 64 CUDA-event-timed calls. Setup,
spectral GEMMs, Krylov work and output are excluded. A full comparison is
admitted only after native tests pass and combined float-storage conversion
time improves by at least 10%. Full application timing uses one unprofiled
run per variant on the same GPU and includes startup and saved output.
Final physical fields are compared from complete native Q5 VTK payloads.

## Native validation and transform benchmark

Job **3423156** passed all three focused native CUDA suites:
`dg_igr_tensor_multigrid_cuda`, `dg_igr_tensor_nodal_cuda`, and
`dg_igr_tensor_multigrid_nodal_cuda`. Production spectral setup reported the
expected dense/tensor selections, with zero matrix-proof error and no
admission fallback in the enabled fixtures.

The private tests use a per-row forward-error bound, plus final float or
subnormal rounding allowances. The extreme subnormal cases can therefore
have a larger normalized relative error despite correct final rounding;
they are not asserted to satisfy a uniform normal-range relative tolerance.

Application SHA-256:
`0acd87c208d6b2de440a72c021d26ca1a87a63c2c2e3377b6a233f9800caa5c5`.
Diagnostic executable SHA-256:
`edda9fd08ec259b74d48709b13a6fdf758834820025f08acdcb7f8f81b3fe0c4`.

| Storage | Direction | Dense, ms | Tensor, ms | Speedup |
|---|---|---:|---:|---:|
| Float, used by TF32 inverse | Forward | 0.532000482 | 0.161971495 | 3.2845× |
| Float, used by TF32 inverse | Reverse | 0.168915004 | 0.161835998 | 1.0437× |
| Float, used by TF32 inverse | Combined | 0.698838472 | 0.323956490 | 2.1572× |
| Double | Forward | 0.534545481 | 0.162258998 | 3.2944× |
| Double | Reverse | 0.206002995 | 0.162037507 | 1.2713× |
| Double | Combined | 0.736859977 | 0.324279010 | 2.2723× |

All contractions use FP64 sums. The double-storage rows are another
conversion benchmark, not a comparison against the original full FP64
solver. Most of the float-path gain comes from the forward conversion.

For the combined fixed-mesh benchmark, tensor versus dense normalized
L2/Linf differences are 2.5388300681e-15 / 4.9425069527e-13 with float
storage, and 3.41002300017e-16 / 1.11072771816e-15 with double storage.
These use the benchmark's synthetic inputs and do not replace full physical
field validation.

The first native build, job 3423090, compiled the application but stopped
before tests because the active Make process retained its old target list
after CMake regenerated the newly added test target. The rerun split
application and test builds into separate canonical `build.sh` invocations.
No source repair or local application build was needed; both attempts are
retained.

## Evolved baseline hardware sample

Job **3423073** continuously evolved the frozen compact-face baseline to
NVTX RK window 3626, captured two forward and two reverse calls, then
intentionally terminated that diagnostic application. It does not claim
physical completion. Native report/CSV coverage passed with 12 replay
passes per selected launch.

| Mean counter | Forward | Reverse |
|---|---:|---:|
| Profiled kernel duration | 554.48 µs | 173.26 µs |
| Long-scoreboard stall | 58.50% | 53.85% |
| Achieved occupancy | 95.28% | 92.01% |
| DRAM throughput, % of sustained peak | 6.57% | 19.76% |
| Registers per thread | 32 | 32 |

The collected local-memory load/store counters measured zero. They do not
alone establish compiler spill counts. The sample supports investigating
load dependencies and request work, without proving one unique bottleneck.
Kernel replay flushes caches and perturbs execution; its timings are
diagnostics, not unprofiled application performance.

## Matched full evolution and physical agreement

Job **3423168** ran the two unprofiled variants sequentially on `gh089`,
GPU `GPU-51c41a08-3a65-616c-1750-aa828380bc66`. Existing tensor, solver and
physical controls match; only the nodal selector, frozen executable identity
and output directory differ. Both production and final accepted-state
equations remain FP64-certified.

| Observed metric | Dense nodal baseline | Tensor nodal candidate |
|---|---:|---:|
| Process wall time, s | 954.059411 | 732.359340 |
| Native simulation stopwatch, s | 775.622908 | 655.148091 |
| Accepted / rejected steps | 4,939 / 0 | 4,939 / 0 |
| Independently certified solves | 14,818 | 14,818 |
| Krylov iterations | 51,732 | 51,747 |
| Maximum relative RHS residual | 9.9998798e-7 | 9.9997421e-7 |
| Final accepted-state relative RHS residual | 3.615794158e-7 | 3.624114523e-7 |
| Observed peak process GPU memory, MiB | 18,376 | 18,376 |
| Process ns per conserved DOF per step | 5.117219 | 3.928102 |

The 15 additional iterations are retained in the candidate timing. Neither
run has rejected steps, missing certificates or residual-replacement
retries. Process time includes startup and saved output. The per-DOF
figures are observed full-process costs, not isolated operator throughput.
The memory sampler requested 1 Hz but had gaps up to 33.128 s in the
baseline; the table does not establish an uninterrupted peak-memory bound.

Both complete native VTUs contain 1,160,775,042 bytes. Final-time selection,
source/configuration identity, physical alpha and transfer hashes were
verified through the run audits and collection receipts:

- Baseline SHA-256:
  `c2e6c8cd52a227b30f6125974fa634632a686787a7ea44d448d1ec472c04a5ed`.
- Candidate SHA-256:
  `9ab51b7545c03104a6fbea77c4b95e16e0e207eecec5814859b690bd229760a2`.

| Complete saved Q5 field | Normalized L2 difference | GL-sampled normalized Linf difference |
|---|---:|---:|
| Density | 6.847857139e-10 | 8.145659677e-9 |
| x momentum | 1.962825084e-9 | 8.770991201e-9 |
| y momentum | 1.962427724e-9 | 8.220244081e-9 |
| Energy | 5.374795047e-10 | 3.950355186e-9 |
| Sigma | 9.478741081e-8 | 3.960492542e-7 |

Normalization is per component: `||candidate − baseline||₂ / ||baseline||₂`
for L2, and `max_GL |candidate − baseline| / max_GL |baseline|` for sampled
Linf.

Every conserved component passes the existing 1e-3 normalized L2 and 1e-2
sampled Linf engineering budgets. Sigma has no assigned physical acceptance
budget; its measured differences are reported separately. GL6×6 integrates
squared Q5 differences exactly up to floating-point roundoff. Linf is
sampled, not the continuous supremum. These are implementation comparisons
at t=0.1 and matching alpha, not analytic Sedov error estimates.

## Timing variability and checkpoint evidence

The identical frozen baseline previously took 739.087664 s on `gh006`
(job 3422767). Its 14,818 residual-certificate lines match the current
baseline byte-for-byte, so different iteration work does not explain the
214.971747 s increase. Of that increase, 112.016137 s lies within the native
stopwatch and 102.955610 s outside it. The installed timer configuration and
frozen binary confirm monotonic wall time; this is not a CPU-time versus
elapsed-time discrepancy.

The native stopwatch begins after initial output and ends after final
output/certification. Baseline checkpoint timing diverged sharply after
step 3000, including a 51.795 s increase over the earlier reference during
3000–3500, where no scheduled VTK save occurs. Its memory sampler also
records 14 gaps longer than 1.5 s, versus none in the candidate. Available
evidence cannot uniquely identify a filesystem, GPU or host cause.

All four adjacent 500-step analysis intervals without scheduled VTK saves
are reported below; the outlier is retained:

| Completed steps | Baseline, s | Candidate, s | Iterations, baseline/candidate |
|---|---:|---:|---:|
| 1000 → 1500 | 41.526158 | 40.234656 | 4,500 / 4,500 |
| 2000 → 2500 | 44.275700 | 40.221135 | 4,500 / 4,500 |
| 3000 → 3500 | 100.010035 | 47.097685 | 5,895 / 5,894 |
| 4000 → 4500 | 50.442838 | 56.212651 | 6,881 / 6,883 |

Actual VTK markers in both logs occur at completed steps
0, 966, 1964, 2975, 3973 and 4939. The checkpoint timestamp is captured at
analysis-hook entry, before host reads and the callback. Differences
therefore include the starting analysis callback and ordinary logging;
they are not GPU-only or entirely output-free intervals.

The four baseline/candidate ratios have descriptive median **1.06645×**,
but span **0.89736× to 2.12346×**, including a slower candidate interval.
Neither that median nor the 23.24% whole-process reduction is a reliable
code-caused overall speedup estimate. The isolated conversion benchmark and
hardware results establish the narrower kernel improvement.

## Same-allocation hardware comparison

After both timed applications and residual audits completed, the same job
captured one float-storage kernel per direction and implementation on the
actual mesh. These use synthetic diagnostic inputs and 12-pass kernel
replay with identical cache/clock controls. All four captures and numeric
counter coverage passed.

| Counter | Forward dense → tensor | Reverse dense → tensor |
|---|---:|---:|
| Profiled duration, µs | 551.712 → 163.328 | 170.944 → 162.784 |
| Executed thread DFMA instructions | 339,738,624 → 113,246,208 | 405,798,912 → 179,306,496 |
| L1/TEX requested bytes | 6,175,064,064 → 1,134,559,232 | 2,470,969,344 → 614,465,536 |
| DRAM bytes read | 114,323,712 → 114,318,848 | 76,588,032 → 76,587,008 |
| Achieved occupancy, % | 95.28 → 75.54 | 92.27 → 67.21 |
| Registers per thread | 32 → 32 | 32 → 32 |

All collected local-memory traffic/instruction counters measured zero.
Forward removes repeated input gather/scaling work in addition to reducing
contraction work. Reverse already performs final scaling once per output;
that scaling, scatter and nearly unchanged DRAM payload remain. Both tensor
directions finish near 162 µs.

Requested cache traffic and arithmetic fall substantially, but the reverse
gain is small. The new block has only 36 value-processing lanes among
64 threads, while measured warp occupancy also falls. Batching several
cells per block is therefore a concrete next hypothesis for improving lane
use and latency hiding. No batching speedup or single dominant bottleneck
has been established; barrier-stall counters were not collected. These
FP64 SIMT samples do not measure the wider solver's TF32 GEMMs.

## Retained evidence and decision

Keep this as an opt-in candidate: large-mesh conversion speed and physical
agreement are established. The reversed-order repeat below supports a modest
compute-path benefit, with only a sub-percent observed whole-process change.
The fresh timeline makes face/volume operator work the next priority; nodal
block packing remains a secondary experiment. No default change is made.

Local campaign root:
`/home/jarias9/.cache/igrlab-experiments/sedov-spectral-nodal-20261020`.
Remote root:
`/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/sedov-spectral-nodal-20261020`.

- `ncu-baseline-3423073/`: evolved baseline native report, raw CSV and counters.
- `build-3423156/`: canonical build, detailed native tests, benchmark/gate,
  frozen executable receipt and case specification.
- `comparison-3423168/`: unprofiled records, residual audits, four hardware
  captures, final VTK receipts and full Q5 comparisons.
- `ncu-transform-review/`, `timing-variation-review/`,
  `matched-window-review/`: independent derived evidence and scope limits.
- `build-3423090/`: initial cached-target build failure, retained separately.

All four allocated jobs used one GPU, eight CPUs and 96G host memory under
`bdiy-dtai-gh`/`ghx4`/`projects`. Actual use was **2,460 GH200-seconds /
0.683333 GPU-hours**, including the build retry and all profiling.
Queue waiting is excluded. Accounting is retained in
`campaign-accounting.json`. No extra physical reruns were submitted during that initial pass.
The separately authorized reversed-order repeat is recorded below.

## Reversed-order repeat and evolved Systems capture — job 3424855

The user authorized a matched repeat after the readiness checks. The frozen
candidate ran first, followed by the frozen dense-nodal baseline on the same
GH200, `gh140`, UUID `GPU-fb152096-3fb4-11a7-9120-fb0d0d24ed10`.
Both binaries, case specification, output cadence, solver controls and scan-off
settings are unchanged. This baseline already contains the earlier TF32 inverse
and compact-face improvements; it is not the original FP64 GPU implementation.
No application rebuild or new production code was needed.

The original application runner and process-memory sampler were retained.
Additional external telemetry used an independent 100 ms host heartbeat,
one-second GPU queries, and application `/proc` samples. These observations
were buffered in RAM until each timed run ended. The new audit accepts the
reversed execution order while retaining baseline/candidate order in result
arrays. All run identity, residual and same-GPU checks passed.

| Measurement | Dense-nodal baseline | Tensor-nodal candidate |
|---|---:|---:|
| Whole application, seconds | 729.635494 | 724.039335 |
| Native simulation stopwatch, seconds | 650.653148 | 643.399586 |
| Native accumulated linear-solve timer, seconds | 286.498402 | 266.800618 |
| Native accumulated update timer, seconds | 417.161829 | 398.148426 |
| Outside native stopwatch, seconds | 78.982346 | 80.639749 |
| Steps / rejected steps | 4,939 / 0 | 4,939 / 0 |
| Converged, certified solves | 14,818 | 14,818 |
| Krylov iterations | 51,732 | 51,747 |
| Maximum current-FP64 RHS-relative residual | 9.9998798e-7 | 9.9997421e-7 |
| Sampled peak GPU memory, MiB | 18,376 | 18,376 |

The observed process-time reduction is **0.76698% (1.007729×)**; the native
stopwatch reduction is **1.11481%**. This is one reversed-order pair, not an
estimate of a repeat distribution or a statistically established sub-percent
whole-solver gain. The original 23.24% observation remains invalid as a causal
speedup estimate.

Using the same mechanical selection of adjacent 500-step intervals without
scheduled VTK saves as the original analysis gives:

| Steps | Baseline seconds | Candidate seconds | Time reduction |
|---|---:|---:|---:|
| 1,000–1,500 | 41.735478 | 40.025054 | 4.098% |
| 2,000–2,500 | 41.773576 | 40.006055 | 4.231% |
| 3,000–3,500 | 47.315255 | 44.877167 | 5.153% |
| 4,000–4,500 | 52.290089 | 49.590294 | 5.163% |

Across these four intervals, measured time falls from 183.114398 to
174.498570 seconds (**4.70516%**). They include ordinary logging and analysis
callback work and are not GPU-only timings. Their consistent direction
supports a compute-path benefit, while total process timing still includes
startup and saved output. There is no per-save timer that would isolate a
filesystem contribution to the difference. The native accumulated linear-solve
timer falls by 19.698 seconds, and the update timer by 19.013 seconds. These
are host-side component timers, not independent GPU kernel totals. Applying
the isolated conversion benchmark to the actual preconditioner call counts
predicts about 19.389 seconds saved; this transferred microbenchmark model
is consistent in scale, but is not a production kernel-time measurement.

The earlier long process-memory-sampler gaps and anomalous baseline
checkpoint interval did not recur. Maximum independent heartbeat spacing
was 0.100156/0.100145 seconds for baseline/candidate, with no
gaps over 1.5 seconds. One baseline GPU query took 3.241 seconds; concurrent
`/proc` samples showed a running application with advancing CPU and write
counters, so that query delay is not evidence of an application pause.
Software power-capping counters increased during the pair; thermal and
hardware power-brake counters did not. Clocks were dynamic, not locked.

Each variant's entire 14,818-line certificate sequence matches its own earlier
run. At t=0.1, each complete 1,160,775,042-byte native VTU also matches its own
earlier payload exactly: baseline `c2e6c8cd…`, candidate `9ab51b75…` (full hashes
in the retained receipts). The two variants are not byte-identical to one
another. The previously verified complete Q5 norms therefore apply unchanged:
maximum conserved-component normalized L2/Linf differences remain
1.962825e-9/8.770991e-9, and sigma remains 9.478741e-8/3.960493e-7.
Conserved-field budgets pass; sigma has no assigned budget. This establishes
implementation reproducibility, not analytic Sedov accuracy.

Only after both unprofiled applications and their audits ended, a separate
continuous candidate evolution generated a **15.004648-second** Systems
capture. Its delay was calibrated from the completed candidate's timing;
the application was intentionally terminated after recording. It reached
step 3,355 in the retained certificates and recorded 164 RK-step events.
That controlled cutoff is not a physical comparison failure. Native report,
SQLite, kernel census, API statistics and solve logs are retained.

The union of GPU kernel/copy/memset intervals covers **14.724932 seconds,
98.1358% of the trace**. Kernel intervals alone cover 14.166877 seconds.
All observed device work is on one stream. Major measured groups are:

| Kernel group | Summed kernel seconds | Share of kernel time |
|---|---:|---:|
| Interior exact-face record | 1.979831 | 13.98% |
| Face record + gather + PA diffusion/mass | 4.195798 | 29.62% |
| TF32-named cuBLAS GEMMs | 1.706217 | 12.04% |
| Nodal tensor conversions, both directions | 0.639255 | 4.51% |

The face-record row is included in the larger operator row; these rows are
not disjoint categories. TF32-named GEMMs occupy 11.37% of elapsed capture
time, and nodal conversions 4.26%. Kernel names and dispatch establish the
GEMM path; Systems does not measure Tensor Core instruction utilization or
achieved tensor FLOP/s. The earlier Compute counters remain separate evidence.

Long host CUDA API durations mostly overlap GPU execution and should not be
counted as CPU-only idle time. Nested NVTX and concurrent OS-runtime calls
also cannot be added as elapsed-time partitions. The frozen source does not
instrument VTK/analysis callbacks, so their absence from NVTX names alone is
not proof of an output-free trace. Absolute RK-step mapping from the delayed
capture remains an inference from retained progress and range counts.

The next performance target should follow the measured current-operator cost:
investigate face application/gather and volume mass/diffusion first.
Fusion and data-layout changes are hypotheses to assess with hardware
counters; this timeline alone does not establish a memory-bandwidth bottleneck.
Block packing is still a valid secondary experiment, but the conversion pair
now accounts for only 4.26%
of this elapsed window. No further code optimization, default change, or GPU
run was performed as part of this repeat.

Retained repeat evidence:
`/home/jarias9/.cache/igrlab-experiments/sedov-spectral-nodal-repeat-20261020`.
Native remote output:
`/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/sedov-spectral-nodal-20261020/repeat-3424855`.
The job completed at **18:38:33 EDT on October 20, 2026**, using
**1,959 GH200-seconds / 0.544167 GPU-hours**. The separate readiness checks
used 103 GPU-seconds. No recurring automation was created.
