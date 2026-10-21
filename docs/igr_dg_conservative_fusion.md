# Conservative Euler volume fusion after the DG checkpoint

On the fixed 512² p5 ordinary-DG Sedov case, the selected fused Euler volume path measured **690.869505 → 668.581089 seconds**, **3.226140% less full application time / 1.033337×**, against the merged checkpoint. Complete final VTK fields are byte-identical and every solve uses the same Krylov iteration count. This is one baseline-first pair on the same physical GPU, with normal VTK output and unlocked clocks; it has no repeat-based confidence interval.

The measured implementation is `97ba1681ac27e07145df790d4c23ed837a21caea` on `igr-dg-face-flux-opt`, derived from merged `dev` `5b913f7c739e61e08aaf2bda5cb4fbe068b35f7e` (PR #76). The [measurement record](benchmarks/dg_conservative_fusion_measurements.json) contains the exact commands, environments, identities, tests, residuals and evidence hashes. This is an opt-in prototype, selected with `IGRLAB_SHARED_EULER_CONSERVATIVE_FUSION=1` alongside the existing FP32 tensor-line Euler configuration.

## Change and scope

The original conservative volume tail launches four kernels: center the physical flux, apply FP32 tensor lines, calculate the FP64 conservative moment correction, and restore the anchors/scatter into the residual. The fused path stages one element's four equations in shared memory and performs those operations in one 64-thread block with shared pitch 73.

The pointwise nonlinear flux remains FP64. Cast points, contraction/accumulation order, native-node ordering, expected/observed FP64 moment sums, mass fractions and anchor restoration are preserved. The FP64 current IGR equation, residual authority and reusable TF32 spectral inverse remain the surrounding solver. This change adds no new tensor-core operations; it reduces intermediate traffic and storage.

Admission is restricted to the existing verified ordinary-DG p5 Gauss–Legendre FP32 tensor-line path. Other admitted orders, precision modes and dense layouts retain the buffered path. Existing basis and geometry rejection rules are unchanged; this does not add curved-mesh or DGSEM support. Selection is frozen until the packed setup is rebuilt; reset/reconfiguration releases the selected storage consistently.

The centered input (576 MiB), contraction output (288 MiB), and moment correction (8 MiB) are not allocated by the fused owner. Exact removed owned scratch is **914,358,272 bytes / 872 MiB**. The raw FP64 flux packet remains. This source accounting is distinct from sampled process memory below.

## Fixed case and exact control

The mesh is 512 by 512 Cartesian elements: a 32² base mesh with four refinements, p5, 9,437,184 scalar IGR DOFs and 37,748,736 conserved DOFs. Both arms use L2/BrokenL2, SSP RK3, sigma every stage, CFL 0.3, max-length alpha with `c_alpha=1/6` (physical alpha `6.357828776041666e-7`), RHS-relative IGR tolerance `1e-6`, absolute tolerance zero, and continuous evolution to `t=0.1`. Normal VTK output is every 0.02. Positivity limiting, modal damping, sigma-aware CFL/Rusanov, invariant-domain flux, and optional nonfinite/admissibility scans are off. There is no restart.

The checkpoint executable was compiled from `5499c0dad18af7940e614019825113f3f52d5906`; its DG arithmetic is the merged `dev` checkpoint. Its SHA-256 is `b18ac258421966c9e627b4c4dc6d1302a4e62d76a86a3440803c38a815b47dc3`. The selected executable SHA-256 is `5a8f1c3e02d214b55bb857fadd8308cb2864b1e46dff5125d66046a0fcf26495`. Both use the current checkpoint environment; only the new fusion switch and executable/output paths differ. The control is the real frozen checkpoint binary, not a newer executable with a mode disabled.

All application builds and checks used canonical `build.sh --gpu --cuda-arch 90 --hdg-runtime --nvtx-profile --iterate` on DeltaAI. Jobs use one GPU at a time, account `bdiy-dtai-gh`, partition `ghx4`, constraint `projects`, eight CPUs, 96 GiB requested host memory and OMP=1. No local application build was used.

## Native qualification and tuning

Job **3429955** qualified all four frozen variants sequentially on **gh107**, the same GH200 GPU. Each passed seven named native suites, including the new conservative-fusion oracles, before the large operator benchmark. The oracles include independent assembled RHS and conservation checks, native-node permutation, nonconstant/high-background states, nonfinite/overflow propagation and finite recovery, and other-order/precision/dense fallbacks. Exception-only negative guard checks were explicitly not executed because the installed MFEM build has no exception support. This is focused qualification, not a new full-repository or multi-rank qualification.

The full-size spatial benchmark uses the actual refined Sedov mesh and canonical initial Gaussian, frozen sigma=0, four warmups and 64 alternating interleaved applications per arm. It includes full DG spatial action and CFL synchronization, with setup outside the CUDA event interval. Before/after outputs are finite, nonzero and bitwise equal. It is not an evolved IGR-solve benchmark.

| Threads / shared pitch | Buffered operator (ms) | Fused operator (ms) | Paired time reduction | Tested source |
|---|---:|---:|---:|---|
| 128 / 72 | 7.028025 | 6.430092 | 8.508% | `bad59e1` |
| 64 / 72 | 6.709750 | 5.862196 | 12.632% | `0c7bf85` |
| 128 / 73 | 6.979950 | 6.212716 | 10.992% | `11d18d3` |
| 64 / 73 | 6.938788 | 5.956373 | 14.158% | `97ba168` |

64/73 was selected for the largest within-variant paired reduction, **14.158%**. 64/72 has the lowest raw absolute time and the buffered controls vary across variants, so this single sweep does not establish a statistically definitive padding optimum. All four configurations passed their numerical gates.

## Matched full simulation

Job **3429991**, node **gh107**, GPU `GPU-89f8ade6-b0a7-6cd0-6a80-1d88a5faf712`, ran the checkpoint followed by the selected candidate. Field hashing was deferred until both timed processes finished.

| Measurement | Checkpoint | Fused | Change |
|---|---:|---:|---:|
| Full process, including setup/output/finalization | 690.869505 s | 668.581089 s | 3.226% less |
| Native stopwatch, after initial output | 612.169423 s | 588.767109 s | 3.823% less |
| Five predefined intervals between VTK writes | 200.492177 s | 192.900915 s | 3.786% less |
| Sampled peak process GPU memory | 18,708 MiB | 17,840 MiB | 868 MiB less |
| Accepted steps / IGR solves | 4,939 / 14,818 | 4,939 / 14,818 | Identical |
| Outer Krylov iterations | 51,742 | 51,742 | Every solve identical |
| Maximum actual RHS-relative residual | 9.9954522e-7 | 9.9954522e-7 | Identical |

There were no residual-replacement retries. All 14,818 factor-one current-equation FP64 certificates and the final committed-state closure certificate passed in each arm. No diagnostic scans were enabled to obtain these results.

Both complete final VTUs are **1,160,775,042 bytes**, SHA-256 `81d8bf54ddec7ab4f0b6293d77283d6c1ee0f9d19304c1b04f6846bc191debb3`. This covers stored density, momentum, energy, sigma, pressure and velocity at `t=0.1`; it also matches the prior integrated checkpoint. Equality of the complete files avoids a tolerance-based field comparison, but does not assert identity of every unsaved state.

The five intervals were fixed before the pair: steps 1000–1500, 2000–2500, 3000–3500, 4000–4500 and 4500–4939. They include 2,439 steps, 7,317 solves and 28,369 iterations in each arm. All five are VTK-free and faster (2.176–4.556% less time), with identical per-solve work. They remain host elapsed intervals including analysis/logging and IGR work, not isolated GPU kernel time or five independent repeat trials. No substitute intervals were selected after seeing results.

External 100 ms host heartbeats show maximum gaps of 0.100260 and 0.100706 s, with none above 1.5 s. Median SM clock is 1,965 MHz in both arms and memory clock is 2,619 MHz in all samples. The preceding VTK-free interval measurements support a solver benefit beyond output timing variation, while leaving fixed-order drift and ordinary run-to-run variability unresolved. One-second memory samples can miss brief peaks; the observed 868 MiB reduction need not equal the exact 872 MiB ownership change.

## Hardware evidence

Baseline Nsight Compute job **3429585** covered 84 launches across 16 configurations at evolved RK range ordinal 3200. The original 128/72 fusion capture **3429846** sampled three fused launches at the same requested ordinal. Center → contraction → moment → restore consumed **3.988439 GiB/RHS** mean measured DRAM traffic; original fusion consumed **1.327813 GiB/RHS**, **66.708% less**. Sum of the four baseline duration medians is 2,085.408 µs/RHS; original fused median is 1,445.696 µs/RHS.

These captures used different GH200 devices, kernel replay, all-cache flush and unlocked clocks. Their durations are profiler evidence, not an unprofiled simulation speedup. Original-fusion barrier and short-scoreboard stall metrics (34.19% and 14.64%) and 41.1 million shared-load bank conflicts per launch motivated the tuning. Stall metrics are not percentages of application wall time, and bank counts lack a normalized request denominator.

Selected 64/73 capture **3429992** completed on **gh006**, with the exact qualified source/binary, a 262,144-block grid, 64 threads/block, 3,584 bytes static shared memory and three matching launches. All 9,600 emitted current-equation certificates before the intended cutoff passed. The profiler intentionally terminated its separate diagnostic application after the requested launches; the full physical comparison above completed independently.

| Fused-kernel metric | Original 128 / 72 | Selected 64 / 73 |
|---|---:|---:|
| Median replay duration | 1,445.696 µs | 1,124.288 µs |
| Mean measured DRAM read + write / RHS | 1.327813 GiB | 1.327344 GiB |
| Median achieved active warps | 59.13% | 59.39% |
| Registers per thread | 43 | 43 |
| Median barrier-stall metric | 34.19% | 20.81% |
| Median short-scoreboard metric | 14.64% | 15.04% |
| Median long-scoreboard metric | 33.56% | 40.05% |
| Median shared-load bank conflicts / launch | 41,108,066 | 7,011,798 |

The selected configuration retains **66.720% less measured DRAM traffic** than the original four-stage tail. Its replay duration is shorter and its barrier/bank-conflict metrics are lower than the first fused prototype, while achieved occupancy and register use are essentially unchanged. These cross-device counters are consistent with improved synchronization/shared-memory service; thread count and padding changed together, so they do not isolate padding's contribution. Memory-dependency stalls and the ordered correction barriers remain. Whole-case improvement is established by the unprofiled pair, not by these replay durations.

## Campaign interpretation

This phase provides an observed **22.288416 seconds** saved per full fixed-case run, plus lower scratch ownership and sampled GPU memory, with identical final saved fields. It does not change alpha, tolerances or the preconditioner to obtain the saving. Keep the initial-state operator, profiler replay and whole-case measurements separate.

The historical original-to-checkpoint **6.189731×** result belongs to job 3428588 and its pinned original/cumulative executables. Do not multiply that independently measured ratio by this phase's ratio or present it as a newly measured original-to-fused result.

Remaining profile targets include IGR face record/gather, side-density projection and Euler face lift. Batching the existing 12-thread side-density projection across faces is the next small arithmetic-preserving hypothesis; broader density-shape reuse and face-lift staging need their own measured qualification. No speedup for those changes is claimed here.

Raw reports, native logs, source manifests, exact case/runner files and complete final VTK outputs are retained under `/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/igr-dg-face-flux-20261021`. Compact evidence and analysis are backed up under `/home/jarias9/.cache/igrlab-experiments/igr-dg-face-flux-20261021`. Documentation commits after the tested source do not change the qualified implementation.
