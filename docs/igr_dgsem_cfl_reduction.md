# Parallel DGSEM characteristic-speed reductions

The parallel reduction measured **986.736058 → 687.738442 seconds**, **30.302% less full-process time / 1.43475×**, with identical final VTK files and identical per-solve work. This phase replaces the large-array, single-block characteristic-speed maxima in the DGSEM production path with parallel blockwise maxima and a final reduction. The comparison uses the already-qualified TF32 spectral IGR solver in both arms; only the executable and output paths differ. The physical equation, CFL calculation, solver tolerance and diagnostic-scan settings are identical.

The cumulative branch `igr-dgsem-cfl-reduction` starts at `8dcbbff883a2af4188b3f4de580107d7666a4493`. Tested candidate source is `2363d0727b979a3241abfd5177d2c3c226c32c24`, binary SHA-256 `86c0cd82810ef6d002785cc3d065627081c9ec57ca0661438300e2e727157c50`. The control is the frozen executable compiled from `ed8829cfb0075ae4e04aae922faf7ac697d6e669`, SHA-256 `7c7a92a88f77cb8871977085724df6ce5c0c121f1834c20ea1bf6feb25b85737`, qualified in the [preceding DGSEM phase](igr_dgsem_sedov_qualification.md). The intervening `8dcbbff` commit only documents that qualification. Exact configuration and measurements are retained in [the case definition](benchmarks/dgsem_cfl_case.json) and [the measurement record](benchmarks/dgsem_cfl_measurements.json).

## Motivation and implementation

The previous evolved-state Systems capture (3430740) recorded 683 characteristic-speed maximum launches taking 6.452695 seconds, 44.707% of its recorded kernel duration. Each launch used one 256-thread block for an entire array: 9,437,184 nodal speeds or 3,151,872 face-point speeds. This is required CFL work; optional diagnostic scans were off.

The production owner now supplies retained scratch to the maximum operation. Arrays above 4,096 entries use `min(ceil(N/1024), 1024)` blocks of 256 threads for partial maxima, followed by one 256-thread final reduction. Scratch is at most 1,024 FP64 values, **8 KiB**, shared sequentially between the nodal and face reductions. It is prepared with the cache, includes admitted overintegration arrays, and participates in the existing workspace-lifetime checks. No per-RHS scratch resize, new host read or MPI reduction is introduced.

The original two-argument function remains unchanged as the compatibility path and exact microbenchmark control. Small arrays and non-native-CUDA backends retain that path. Positive maxima and positive infinity have an exact representation independent of reduction partitioning. When the final maximum is zero, the implementation repeats the literal original ordered traversal so signed-zero and exceptional-input behavior are preserved. Separate nodal and face results, host combination, MPI maximum and CFL cadence are retained.

This is a parallel reduction optimization and adds no tensor-core operations. Its production dispatch depends on array size/backend, rather than Cartesian mesh or spectral-preconditioner admission. Full-case performance evidence below is nevertheless limited to the fixed Cartesian p5 single-GPU case.

## Native qualification and reduction benchmark

Canonical DeltaAI job **3430873**, node **gh011**, passed `dg_igr_dgsem_maximum_cpu`, `dg_igr_dgsem_maximum_cuda`, `dg_igr_dgsem_operator_cuda` and `dg_igr_dgsem_igr_tests`. The maximum fixture checks 360 inputs on each backend. Both old and new functions are compared bitwise to a separately compiled literal-original loop.

Coverage includes the small-array threshold, block-cap boundaries, ragged grids and later-stride winners, NaNs/infinities/signed zeros, large zero/negative fallback, fresh device writes and decreasing maxima. Oversized scratch is initially poisoned with infinity to expose stale-tail reads, and its size, capacity and device address must remain unchanged. DGSEM integration checks retain the existing residual/CFL and scan-policy checks. This is focused native qualification, not a new full-repository or multi-rank validation.

The same allocation then compared the unchanged old function and the retained-workspace function on finite-positive arrays with the actual production lengths. Four warmups and 32 alternating interleaved repetitions per variant use CUDA events; allocation, literal-original oracle, host reads and pre/post bitwise checks are outside the event intervals.

| Maximum input | Entries | Old single-block mean | Hierarchical mean | Mean paired time reduction |
|---|---:|---:|---:|---:|
| Nodal speeds | 9,437,184 | 14.129521 ms | 0.030551 ms | 99.784% |
| Face-point speeds | 3,151,872 | 1.974151 ms | 0.011109 ms | 99.437% |

All results are bitwise equal and scratch is reused. These are warmed, isolated reductions with synthetic values, not full simulation speedups. In particular, the isolated face array has a different cache environment from the preceding evolved-state profile; its duration should not be substituted into that profile.

## Matched physical comparison

Job **3430887** ran both fresh executables sequentially on **gh069**, baseline first, using the same physical GPU. All numerical equality gates passed.

| Measurement | Original maximum | Hierarchical maximum | Change |
|---|---:|---:|---:|
| Full process including setup/output/finalization | 986.736058 s | 687.738442 s | 30.302% less / 1.43475× |
| Native stopwatch after initial output | 895.984924 s | 609.998284 s | 31.919% less / 1.46883× |
| Five predefined intervals between VTK writes | 339.783663 s | 203.184035 s | 40.202% less / 1.67230× |
| Accepted steps / certified solves | 4,939 / 14,818 | 4,939 / 14,818 | Identical |
| Outer Krylov iterations | 49,175 | 49,175 | Every solve identical |
| Sampled peak process GPU memory | 19,212 MiB | 19,212 MiB | Same sampled value |
| Maximum RHS-relative FP64 residual | 9.9979699e-7 | 9.9979699e-7 | Identical |

The complete ordered certificate payloads match, as do all step/CFL numerical summaries, coefficient-refresh/application counts and final closure. There are no rejected attempts or residual-replacement retries. This phase changes the cost of the same calculation, rather than reducing iteration work or changing the requested tolerance.

The final complete VTUs are both 1,160,775,042 bytes with SHA-256 `8959b8153f4b004e96faea77b389f7725db5f94e8edc8257cbb5800b88953948`, also matching the preceding qualified TF32 run. This covers stored density, momentum, energy, sigma, pressure and velocity at t=0.1. No tolerance-based field comparison is needed for identical complete files; no identity claim is made for every unsaved intermediate field.

All five predefined windows are VTK-free and faster, with 35.329–43.318% less time. They total 2,439 steps with identical iteration work. They retain analysis/logging and host activity, and are not independent repeats or GPU-only timings. The full-process saving is **298.997615 seconds** per fixed-case run.

Telemetry shows no errors or observed host-wide pauses: maximum 100-ms heartbeat gaps are 0.113203/0.106712 seconds, with none above 1.5 seconds. Median SM clocks are 1,980/1,965 MHz; memory clocks are 2,619 MHz in both arms. No clock normalization is applied. The 8 KiB extra scratch is below the granularity of the sampled MiB memory comparison.

Both arms use the unchanged 512 by 512 p5 DGSEM Sedov case: 262,144 Cartesian elements, 37,748,736 conserved DOFs and 9,437,184 scalar IGR DOFs; Euler and sigma use GLL nodes, with L2/BrokenL2 physical spaces. SSPRK3 evolves continuously to t=0.1, CFL 0.3, sigma every stage, max-length `c_alpha=1/6` (physical alpha `6.357828776041666e-7`), RHS-relative FP64 FGMRES tolerance `1e-6`, absolute tolerance zero. Both use the TF32 spectral preconditioner, generic cached FP64 current IGR volume, compact face paths and existing fused FP64 SIMT DGSEM Euler residual. VTK is written every 0.02. Nonfinite/admissibility scans, limiters, modal damping, subcell blending and restart are off.

Acceptance requires identical complete residual-certificate payloads, step/CFL numerical accounting, iteration work, coefficient/application counts, final closure and complete final VTU bytes. Hashing occurs after both timed processes finish. The same predeclared VTK-free intervals are retained: 1000–1500, 2000–2500, 3000–3500, 4000–4500 and 4500 through the common final step; no substitute intervals are selected after seeing timings.

## Evolved hardware evidence

Nsight Systems job **3431084**, on **gh007**, recorded 15.005384 seconds of a fresh continuous candidate evolution, with 197 complete trace-local RK steps. The same qualified binary and numerical settings were used; only NVTX/output prefixes changed. All 4,356 emitted FP64 certificates before the intentional cutoff passed. The unprofiled physical pair completed separately.

Within 595 complete `DG/Mult` scopes, CUDA-launch correlation finds the expected nodal-then-face sequence. Mean partial-plus-final kernel duration is **31.850 microseconds for nodal speeds** and **11.609 microseconds for face-point speeds**, excluding scalar transfers and synchronization.

The trace records **1,192 partial-reduction launches at grid 1024 / block 256**, plus **1,192 finalizations at grid 1 / block 256**. Each kernel uses 2 KiB static shared memory per block. Combined recorded duration is **0.025902 seconds**, **0.184% of recorded kernel time**. No old single-block `DeterministicMaximumKernel` launch was recorded. The preceding trace attributed 44.707% to that old kernel; these traces use different GPUs and physical-step counts, so this is a change in recorded composition, not an independently measured whole-case speedup.

| Recorded kernel group | Summed duration | Share of recorded kernel duration |
|---|---:|---:|
| CFL partial and final maxima | 0.025902 s | 0.184% |
| IGR generic volume and refresh | 2.774022 s | 19.756% |
| IGR compact face record and gather | 2.367748 s | 16.862% |
| TF32 spectral GEMMs | 1.525374 s | 10.863% |
| FP64 tensor nodal maps | 0.576666 s | 4.107% |
| DGSEM fused Euler residual | 0.809258 s | 5.763% |

The generic FP64 current-IGR volume and compact face operations are now the largest identifiable kernel groups. The capture also contains 7,143 actual TF32 cuBLAS kernels and the expected tensor nodal maps; the preconditioner machinery remains active. The final interrupted application can leave partial ranges at the trace boundary, so launch-count equality across every route is not assumed.

The capture reports the same driver/trace-support compatibility warning (CUDA driver 13.2 versus this Nsight build's 12.9 support) and possible missing CUPTI/NVTX/OS-runtime events. Raw reports and SQLite are retained. All percentages above refer only to recorded kernel durations, not complete application-time partitions, SM occupancy, bandwidth or tensor-core utilization. No VTK/Analysis overlap was observed in recorded ranges; incomplete output instrumentation prevents an unconditional output-free claim. No new Nsight Compute counter pass was performed.

## Evidence and limits

All builds and native checks use canonical `build.sh` on DeltaAI, CUDA architecture 90, HDG runtime and NVTX enabled. Resources are one GH200 GPU at a time, account `bdiy-dtai-gh`, partition `ghx4`, constraint `projects`, eight CPUs, 96 GiB requested host memory and OMP=1. Raw evidence is retained under `/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/igr-dgsem-cfl-20261021`; the local evidence cache is `/home/jarias9/.cache/igrlab-experiments/igr-dgsem-cfl-20261021`.

The comparison is one baseline-first pair with unlocked clocks and no repeat-based confidence interval. One-second GPU memory samples may miss brief peaks. Prior preconditioner and DG speedups must not be multiplied into this new measured ratio. Documentation commits after the tested source do not change the qualified implementation.
