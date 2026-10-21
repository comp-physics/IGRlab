# GLL DGSEM qualification of the reusable spectral IGR solver

The large GLL DGSEM case completes with the reusable TF32 spectral IGR preconditioner: **1,581.69 → 973.21 seconds**, **1.625× / 38.47% less full-process time**, with maximum conserved-field relative L2 difference **5.19e-11**. This phase transfers the shared IGR acceleration to one fixed large DGSEM Sedov case. It compares an FP64 two-level preconditioner with the reusable TF32 spectral reference inverse, while keeping the DGSEM Euler operator, current FP64 IGR equation, tolerance, initial data and output settings identical. It is a comparison of preconditioner strategies, not an isolated TF32-versus-FP64 arithmetic comparison or a historical original-GPU baseline.

The branch starts from merged `dev` `1dd0a70aba56a516829a8c5a7210f134d15468db` (PR #77). Tested source is `ed8829cfb0075ae4e04aae922faf7ac697d6e669`, executable SHA-256 `7c7a92a88f77cb8871977085724df6ce5c0c121f1834c20ea1bf6feb25b85737`. The implementation commits add native GLL spectral tests (`6eb836f`) and make routine DGSEM nonfinite/admissibility state scans opt-in (`ed8829c`). Exact configuration and compact evidence are in [the case definition](benchmarks/dgsem_sedov_case.json) and [the measurement record](benchmarks/dgsem_sedov_measurements.json).

## Fixed physical and numerical configuration

The canonical Sedov Gaussian is defined on the 32 by 32 base mesh before four refinements, giving 512 by 512 Cartesian elements at p5: 37,748,736 conserved DOFs and 9,437,184 scalar IGR DOFs. The unified `sedov_taylor_blast --discretization dgsem` driver is used. Both physical Euler and sigma spaces have GLL nodes, with sigma selected by `IGRLAB_IGR_SIGMA_BASIS=lgl`; physical spaces remain L2/BrokenL2. Sigma reaches the DGSEM Euler operator by an admitted exact native-node gather.

Both arms use SSPRK3, sigma every stage, CFL 0.3, maximum-length alpha with `c_alpha=1/6` (physical alpha `6.357828776041666e-7`), RHS-relative FGMRES tolerance `1e-6`, absolute tolerance zero, 1,000 maximum iterations and failure on nonconvergence. They evolve continuously to t=0.1 and save VTK every 0.02. There is no restart, positivity limiter, modal damping, subcell blend, sigma-aware CFL/Rusanov or invariant-domain flux. Nonfinite and admissibility diagnostic scans are explicitly off. Existing flux/CFL checks are unchanged, and FP64 current-equation residual certification remains active.

The DGSEM discretization uses central volume flux, no volume dissipation and Rusanov surface flux. Its existing fused Euler residual is FP64 SIMT. The newer ordinary-DG FP32 conservative-tail fusion is not selected and is not claimed to transfer to DGSEM.

## Shared operator and preconditioner distinction

GLL interpolation nodes do not imply a diagonal mass matrix for the original IGR GL quadrature. Both arms therefore use the generic cached FP64 tensor-volume path, with collocated/direct storage disabled. The current physical operator and its quadratures are preserved. Evaluating the same GLL polynomial at GL integration points is a basis evaluation, not an H1 projection or a change of physical space. The native GLL oracle explicitly rejects the GL-only diagonal-interpolation shortcut.

The control uses FP64 element corrections plus the existing cooperative CUDA FP64 coarse CG. Its auxiliary coarse space is continuous Q1/H1, with 263,169 unknowns. That preconditioning auxiliary space does not project the physical sigma or its BrokenL2 gradient into H1.

The candidate retains a reference operator and applies its inverse as a preconditioner within FP64 FGMRES. FP64 6 by 6 tensor nodal transforms map native GLL coefficients to and from the spectral representation. Four explicit TF32 cuBLAS GEMMs per inverse application perform the global forward/inverse tensor transforms on 3,072 by 3,072 matrices; diagonal spectral scaling is between forward and inverse transforms. These are repeated inverse applications per Krylov iteration, not new matrix inversions per element or per Runge–Kutta stage. Every accepted solve is certified against the current FP64 equation at the same requested tolerance.

Both arms request shared current-state interpolation, factorized face normals, compact face quadrature data and compact tensor face lifting. Native tests cover those paths with GLL spaces. Large-case kernel execution must be attributed from the retained profile, rather than inferred solely from these environment switches.

## Native qualification

Canonical DeltaAI build/check job 3430371 passed all eight selected suites: GLL spectral inverse with dense and tensor nodal transforms, generic IGR tensor volume, exact faces, device Krylov, shared coefficient diagnostics, DGSEM operator and DGSEM/IGR integration. The GLL oracle covers reordered periodic and natural-boundary cells, consistent off-diagonal mass, cold/warm starts, changing coefficients, FP64/TF32 preconditioning and retained reference-operator lifetime. Maximum FP64 inverse residual was `7.14955e-13`, against a fixed `5e-10` bound; maximum changed-current-equation residual was `2.2537e-12`.

Scan-policy fixtures verify bitwise residual and identical CFL with scans off/on for normal, fused and admitted overintegrated paths. Ordinary residuals default to no bulk state scan; explicit diagnostic admission always scans. The separately requested extrapolated/frozen-sigma algorithm retains the current per-RHS telemetry that its policy requires. Its exception is not active in this every-stage case. Structural/cache/MPI consistency, geometry metadata and actual numerical-operation errors are unchanged. Multi-rank runtime and the unsupported p5-Euler/p1-sigma overintegration combination are not newly qualified by this single-GPU campaign.

The two full-size short pilots reached t=0.0025 in 120 steps and passed all current-equation certificates. Baseline pilot 3430395 and TF32 pilot 3430486 were on different GPUs; they are admission evidence only. The first wrapper misinterpreted an intentionally unprinted successful zero-iteration initial anchor, and the second wrapper encountered a disappearing-process permission error in optional host telemetry after successful physics. Both outcomes were reconstructed from retained logs and bound receipts without rerunning completed physical work. The full comparison is a separate fresh same-GPU pair.

## Measured full comparison

Job **3430525** ran the FP64 control followed by the TF32 candidate on **gh019**, GPU `GPU-4f86110d-740a-c230-3939-e777ffba2498`. Both fresh processes use the same frozen executable and exact argv; environment differences are only the three preconditioner selectors and output prefixes. Field hashing and transfers occurred after both timings.

| Measurement | FP64 two-level | TF32 spectral | Change |
|---|---:|---:|---:|
| Full process, including setup/output/finalization | 1,581.692877 s | 973.205781 s | 38.471% less / 1.62524× |
| Native stopwatch, after initial output | 1,488.510498 s | 894.873574 s | 39.881% less / 1.66338× |
| Five predefined intervals between VTK writes | 664.760351 s | 342.071920 s | 48.542% less / 1.94334× |
| Native accumulated IGR time | 972.484614 s | 363.195073 s | 62.653% less |
| Native accumulated linear-solve time | 872.267393 s | 277.545666 s | 68.181% less |
| Sampled peak process GPU memory | 25,502 MiB | 19,212 MiB | 6,290 MiB less |
| Accepted steps / certified solves | 4,939 / 14,818 | 4,939 / 14,818 | Same counts |
| Outer Krylov iterations | 193,166 | 49,175 | 74.543% less |
| Average / maximum iterations per solve | 13.0359 / 32 | 3.3186 / 5 | Fewer iterations |
| Maximum RHS-relative FP64 residual | 9.9999349e-7 | 9.9979699e-7 | Both below 1e-6 |

All 14,818 printed factor-one current-equation certificates passed in each full arm; no accepted-step rejection or residual-replacement retry occurred. Final committed-state closure residuals were `7.8237292465110115e-7` and `2.8228554080321248e-7`. Unlike the VTK-disabled pilots, these full runs print the initial certificate too. The FP64 acceptance equation and tolerance are unchanged.

The five intervals were fixed before running: steps 1000–1500, 2000–2500, 3000–3500, 4000–4500 and 4500–4939. All are VTK-free and all improved, by 34.917–58.276%. Together they cover 2,439 steps. They include IGR, logging, analysis and host work; they are neither isolated GPU time nor independent repeats. Iteration counts intentionally differ because the preconditioner changes; no interval was selected or rejected based on favorable iteration work.

External host heartbeats show maximum gaps of 0.100752/0.100538 s with none above 1.5 s, and telemetry reports no errors. Median SM clocks were 1,965/1,980 MHz; memory clocks were 2,619 MHz in both arms. There is no observed host-wide hitch, but unlocked clocks and fixed order remain limitations. No clock normalization is applied. The native stage timers above are measured application scopes, not exclusive GPU-time partitions.

The principal mechanism is a 3.93× reduction in outer iterations with the spectral reference preconditioner. Native average solve time per iteration actually increases from 4.516 to 5.644 ms, while total linear-solve time drops by about 595 seconds. This result does not isolate the contribution of TF32 precision from the stronger spectral preconditioner. Fixed Euler, output and other work limit the whole-process gain.

### Complete final fields

Both final VTUs at t=0.1 contain the complete conserved and sigma fields, each 1,160,775,042 bytes. Their hashes differ, so both files were transferred and checked against the post-timing remote hashes before numerical comparison. All 262,144 matched Cartesian cells are covered exactly once. The saved Q5 polynomial is reconstructed from validated output-node positions and integrated using GL6×6; squared-polynomial L2 norms and conserved integrals are exact up to FP64 roundoff. This works independently of the original GLL coefficient representation.

| Field | Relative polynomial L2 difference | Relative maximum difference at GL samples |
|---|---:|---:|
| density | 1.896532e-11 | 4.361397e-10 |
| momentum_x | 5.185890e-11 | 6.605690e-10 |
| momentum_y | 5.186015e-11 | 6.606081e-10 |
| energy | 2.027310e-11 | 4.101993e-10 |
| sigma (reported separately) | 2.474857e-07 | 1.981965e-06 |

All four conserved components pass the predefined engineering budgets: relative L2 ≤1e-3 and sampled relative Linf ≤1e-2. Sigma has no invented physical acceptance budget; its norms are reported separately and both solves retain the same FP64 residual requirement. Relative norms use each reference component's own L2 norm or maximum absolute GL sample. Pressure's relative L2 difference is `2.197965e-11`; mass and energy integral differences are about `3.3e-18` and `3.0e-18` in magnitude.

A separate offline check reconstructed every final GLL6×6 node: 9,437,184 samples in each arm, all finite with positive density and pressure. Minimum density is approximately 0.4494202015 and minimum pressure 1.0003187637 in both. This is final-node evidence; it does not establish positivity between nodes or at unsaved times. Field agreement with the FP64 control is not an analytic Sedov-error estimate.

## Hardware evidence

Nsight Systems job **3430740** captured 15.004304 seconds on **gh014** from a fresh continuous evolution with the same qualified binary and settings. Only NVTX and output paths changed. The 281-second delay placed the diagnostic at an evolved state; 113 complete trace-local RK steps were recorded. The separate application was intentionally terminated after the 15-second recording, and all 4,143 emitted FP64 certificates before termination passed. This cutoff is independent of the successful complete physical pair.

The native trace contains 1,023 forward and 1,023 inverse tensor nodal maps, plus **4,092 actual `sm90_xmma_gemm_*tf32*...cublas` kernels**, consistent with four global GEMMs per spectral inverse application. Independent CUDA-launch/NVTX correlation attributes all 4,092 recorded GEMMs to 1,023 `IGR/krylov-preconditioner-mult` ranges, exactly four GEMMs plus one forward and one inverse nodal map per range. All 341 recorded `Values2D` launches belong to `IGR/device-coefficients`, one per range, supporting the admitted shared state-interpolation path. The compact tensor face record/gather kernels, generic FP64 tensor volume and refresh kernel, and existing FP64 fused DGSEM Euler kernel are present. No collocated-volume kernel was recorded. Compact qdata storage remains a frozen-source/selector/admission inference because its refresh kernel name does not encode that storage choice.

| Recorded kernel group | Summed GPU duration | Share of recorded kernel duration |
|---|---:|---:|
| DGSEM characteristic-speed maximum | 6.452695 s | 44.707% |
| IGR generic volume and refresh | 1.571453 s | 10.888% |
| IGR compact face record and gather | 1.345318 s | 9.321% |
| TF32 spectral GEMMs | 0.861188 s | 5.967% |
| FP64 tensor nodal maps | 0.330642 s | 2.291% |
| DGSEM fused Euler residual | 0.461841 s | 3.200% |

The largest remaining bottleneck is the DGSEM characteristic-speed maximum: **683 launches, 6.452695 seconds, 44.707% of recorded kernel duration**. Every observed launch uses one block of 256 threads. `ReducePackedCharacteristicSpeed` reduces 9,437,184 nodal speeds and 3,151,872 face-point speeds separately on each residual application; this is algorithmic CFL work, not the optional diagnostic state scans. A hierarchical device-wide maximum that preserves the existing positive/zero/nonfinite semantics is the next concrete DGSEM optimization target. No performance result for that proposed change is claimed here.

Recorded kernels occupy 14.433236 of the 15.004304 trace seconds. That measures recorded activity, not SM occupancy or bandwidth saturation; a long-running one-block reduction can keep a kernel active while leaving most of the GPU unused. No new Nsight Compute hardware-counter pass was performed for this transfer qualification.

The profiler warns that the installed CUDA 13.2 driver is newer than this Nsight build's CUDA 12.9 trace support and that CUPTI, NVTX and OS-runtime events may be missing. Native reports, SQLite and warning records are retained. Positive kernel witnesses are valid evidence of execution; complete event counts and exact workload fractions are not claimed. The table is a breakdown of **recorded kernel duration**, not an elapsed whole-application partition or tensor-core utilization. No VTK/Analysis overlap is observed in the collected ranges, but output instrumentation is incomplete, so this is not an unconditional output-free claim. NVTX scopes are nested and partly asynchronous; their summed host durations are not additive GPU-time shares.

## Evidence and limits

Application builds and native tests ran only through canonical `build.sh` on DeltaAI, CUDA architecture 90, HDG runtime and NVTX enabled. Allocations use one GH200 GPU at a time, account `bdiy-dtai-gh`, partition `ghx4`, constraint `projects`, eight CPUs, 96 GiB requested host memory and OMP=1. No local application build or automatic follow-up task was used.

The full pair is baseline-first on one GPU with unlocked clocks; it has no repeat-based confidence interval. Memory samples can miss short peaks. Neither prior ordinary-DG speedups nor the short-pilot timings should be multiplied into this result. Cartesian natural-boundary support and this p5 Sedov case do not establish arbitrary geometry, other orders, cylinder robustness, multi-GPU scaling or HDG support.

Raw evidence is retained under `/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/igr-dgsem-sedov-20261021`, with compact local backup under `/home/jarias9/.cache/igrlab-experiments/igr-dgsem-sedov-20261021`.
