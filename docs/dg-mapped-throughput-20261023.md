# Mapped ordinary-DG throughput checkpoint — 2026-10-23

This completes the bounded 2D DG optimization batch. On the fixed large mapped Sedov case, the recommended combination reduces application time from **574.632 s to 392.494 s**: **31.70% less time, 1.464× speedup**. Advancing the solution improves from **280.002 s to 248.028 s**: **11.42% less time, 1.129× speedup**. Final fields pass the tight equivalence gates: maximum conserved relative L2 difference **1.60×10⁻¹³**, sigma **1.06×10⁻¹²**.

The comparison is against the **previous optimized mapped ordinary-DG solver**, including its fixed spectral corrections and previous Euler row-reuse work. It is not a comparison against the original unoptimized solver. Both arms use the same frozen application, changing only the six new opt-in switches below. No production defaults or IGR accuracy settings changed.

## Source and fixed case

- Tested source: `bc0df2971a6de2f840d995aa45f75bb664523b85`, branch `dg-mapped-throughput-checkpoint`, based on checkpoint `0899a39`.
- Application SHA-256: `08f468cb1d47d4c88c2e7e86e9f8d51dd708efe30bd1ed5b913e1904f6f12957`.
- Native qualification `3446171`: all **14 selected CPU/CUDA groups passed**, including independent mapped-operator, face, mass, stage-data and VTK checks. Installed MFEM/Spack identities remained unchanged; canonical `build.sh` built the application in node `/tmp`.
- Fixed **512×512 Q2 mapped mesh**, ordinary DG, **p5 GL state basis**, **262,144 elements**, **9,437,184 scalar DOFs**, **37,748,736 conservative DOFs** plus sigma. No refinement or mesh-size sweep.
- Mesh SHA-256: `def0f9eabe4b39fa860c5856a77fa186ef83f859728d96de0fc25dc191657836`.
- L2/BrokenL2 IGR, `c_alpha=1/6`, length policy `max`, SSPRK3, CFL `0.3`, exact SIPG faces, every-stage IGR. The closure uses **50 initial / 2 regular damped spectral corrections**, omega `0.7`, projected local reference, `wmma-factor-fp64`. This is the existing approximate fixed-sweep policy, with exact current FP64 operator applications; it does not claim a Krylov residual tolerance.
- `t_final=0.1`, output every `0.02` including the initial frame: six VTK saves. Positivity limiter, modal damping, nonfinite/admissibility scans, conservation-balance scans, H1/L2 output diagnostics and state-restart output remain off. Direct pressure-coefficient output remains on in both arms.
- One GH200 GPU at a time, account `bdiy-dtai-gh`, partition `ghx4`, constraint `projects`, MPI rank 1, OMP 1. Each baseline/candidate pair shares one physical GPU; different jobs need not share a GPU.

Exact arguments, controlled environments, GPU identities, physical norms and counter values are in [the compact results record](dg-mapped-throughput-20261023-results.json).

## Recommended configuration

All switches have the `IGRLAB_` prefix; the existing `DG_IGR_` aliases are also supported.

| New switch | Baseline | Recommended | Change |
|---|---:|---:|---|
| `DG_INVMASS_ROW_REUSE` | 0 | 1 | Reuse one matrix row across the four Euler components; preserve accumulation order. |
| `DG_TENSOR_FACE_ROW_REUSE` | 0 | 1 | Reuse trace/lift weights and geometry across components. |
| `IGR_STAGE_STATE_DIRECT` | 0 | 1 | Read current native state directly; remove the full-state restriction copy. |
| `DG_MAPPED_WEAKDIV_TENSOR` | 0 | 1 | Apply the original weak integral through reference tensor contractions and full per-quadrature mapped geometry. |
| `VTK_CACHED_EVALUATION` | 0 | 1 | Cache field reference interpolation while retaining the native mesh writer and pressure policy. |
| `IGR_FACE_COALESCED_TRACE` | 0 | **0** | Tested separately; slightly slower here, so excluded. |

Previous optimizations remain on: compact IGR volume, fused current residual and solution update, tensor face-density handling, fused coefficients, projected FP64 WMMA local inverse, Euler dense-row execution and the previous weak-divergence row reuse. When the mapped tensor path is selected it replaces that last dense weak-divergence apply. These new kernels use **FP64**; this batch introduces no new TF32 approximation.

The mapped weak integral evaluates flux at the original state nodes, then interpolates that flux. It preserves the native GL7×7 GradientIntegrator quadrature on this Q2/p5 case. It does not replace the nonlinear flux with flux evaluated at interpolated states. Its weak-divergence coefficient representation falls from 2,592 to 196 FP64 entries per element: **5.0625 to 0.3828 GiB** here, a **4.6797 GiB storage reduction**. This is representation size, not a measured peak-residency reduction.

## Full application result

Unprofiled job **3446347**, GPU `GPU-0db23969-1081-f667-0279-d91e41498832` (`gh017`). Both arms complete **6,663 steps and 19,990 sigma calls** with identical settings and save times. Times exclude queueing, builds, validation and post-run field hashing/archiving.

| Disjoint host-wall scope | Baseline s | Recommended s |
|---|---:|---:|
| Before/after `RunEulerCase` (construction, startup, destruction) | 50.166 | 26.668 |
| Driver setup + initial fields/output | 56.596 | 35.778 |
| Advancing solution + CFL | 280.002 | 248.028 |
| Scheduled output | 150.063 | 65.343 |
| Scheduled diagnostics | 0.486 | 0.476 |
| Final output/post-loop | 37.319 | 16.201 |
| **Total application** | **574.632** | **392.494** |

The six native VTK saves total **219.918→93.296 s**. The large application gain therefore includes output savings; the advancing-time gain is reported separately. The outside-driver bin includes construction/startup/destruction and is not a pure kernel or assembly timer. Host scopes can contain asynchronous GPU work and are not a GPU-time partition.

## Individual contributions and evolved GPU cost

Seven-way Nsight Systems job **3446226** uses one GPU and changes one candidate at a time. Every GPU profile keeps writer caching off, evolves continuously from zero, uses diagnostic save interval `0.1`, delays capture 150 s, and records 15 s. Initial/final output is outside selected steps.

| Candidate alone | Target ms/stage, before→after | Recorded GPU ms/RK | Change versus 40.906 baseline |
|---|---:|---:|---:|
| Inverse-mass reuse | 1.723→1.451 | 40.094 | −1.99% |
| Euler trace + lift reuse | 1.665→0.784 | 38.449 | −6.01% |
| Direct-state preparation chain | 0.743→0.571 | 40.494 | −1.01% |
| Mapped weak divergence | 2.244→2.006 | 40.101 | −1.97% |
| Coalesced IGR trace | 0.973→1.006 | 41.147 | +0.59% |
| All GPU candidates | — | 36.292 | −11.28% |

These isolated gains are not additive application speedups. The combined run's unchanged-kernel total differs by only +0.08%, supporting attribution to the changed kernels rather than a broad system-speed shift.

Fresh paired profiles **3446320** compare baseline with the recommended subset, coalesced trace off. Their strictly complete recorded windows contain 332 and 368 RK steps respectively; each stage has exactly two current-A/residual and two local-inverse/update applications.

| Recorded GPU work | Baseline ms/stage | Recommended ms/stage |
|---|---:|---:|
| Per-stage coefficients/RHS + operator/P refresh | 2.125 | 1.941 |
| Current A + residual | 2.908 | 2.887 |
| P + solution update | 0.441 | 0.437 |
| Euler spatial RHS | 7.283 | 5.886 |
| RK common other | 0.412 | 0.411 |
| Other recorded work | 0.526 | 0.526 |

Recorded GPU work per RK is **41.084→36.268 ms**, a reduction of **11.72%**. Sigma-to-Euler GPU cost changes **0.824→0.984** because Euler improves more; this is not an increase in sigma work. Preparation is included in sigma, and the fused A/residual and P/update kernels cannot be split into independent pure costs.

One final host-complete RK in the recommended capture lacked three kernel activities and one memcpy activity. Strict selection rejects the entire incomplete step, preserving 368 steps with one-to-one recorded API/activity matches. The original failed receipt, trace and audit remain unchanged; separate pinned recovery receipts qualify the observed complete subset. Dropping that step changes the mean by only +0.00252 ms/RK. The baseline loses no steps.

Nsight Systems reports driver 13.2 versus trace-library 12.9 compatibility and possible collection-loss warnings. Complete recorded counts and correlations do not prove lossless acquisition. The unprofiled full application pair is the timing authority. The first ablation audits retain a stale “120-second” explanatory string; their numeric records and actual commands show the common **150-second** delay.

## Nsight Compute hardware confirmation

Paired job **3446448**, same GPU for both arms. All **21 counters** and six expected launches per arm pass name, configuration, order and coverage checks. Two stage samples cover weak divergence, inverse mass and face lift at the bounded RK-range selection. Standard kernel replay, all-cache flush and unlocked clocks make these diagnostic timings distinct from application performance.

| Kernel | Replay ms, before→after | DRAM reads GB | L1TEX bytes GB | FP64 DFMA thread-instructions, billions |
|---|---:|---:|---:|---:|
| Weak divergence | 2.241→2.033 | 6.493→1.468 | 11.475→9.588 | 2.718→2.129 |
| Inverse mass | 1.721→1.452 | 3.171→3.247 | 15.704→8.456 | 1.359 unchanged |
| Face lift | 1.302→0.486 | 0.2856→0.2856 | 5.329→2.783 | 0.679 unchanged |

Mass and lift improve despite lower achieved occupancy: registers rise 32→40 and 48→64 respectively. Their benefit is not a reduction in DRAM bytes. Weak divergence uses 47→40 registers and occupancy rises 59.35→74.09%; DRAM reads fall 77.4% and DFMA instructions 21.7%.

Mapped contraction introduces shared-load/store bank-conflict counts averaging 8.98M/19.99M, short-scoreboard stalls 0.26→21.73%, MIO stalls 0.01→9.455% and barrier stalls 0→4.575%. These support further attention to shared-memory layout/dependencies, but do not establish one exclusive bottleneck or additive time fractions. All four measured local-load/store counters are zero in every sample; reported stack size is still 1,568 bytes, so this is not a claim of zero allocated stack.

## Physical qualification and checkpoint boundary

Native VTK decoding compares the full saved Q5 conservative and sigma polynomials on matching Q2 geometry using GL8×8 quadrature. Maximum conservative relative L2 difference is **1.5953×10⁻¹³**; sigma is **1.0580×10⁻¹²**. Maximum conservative sampled relative Linf is **8.8315×10⁻¹³**; sigma is **4.1995×10⁻¹²**. Density/pressure/q-sigma are finite and positive at sampled GL8 points; this does not assert positivity at every point or time. Tight field gates passed before the NCU capture.

The baseline final VTU is byte-identical to the preceding optimized campaign (`db4d0b27…`). Recommended final SHA-256 is `67ab5896aa1a911824c36ab2f549875df70e4a04eda22f046ea19688a8b812e7`. No new claim against the original/converged sigma solver is made here; the existing approximate closure policy is held fixed.

Close this 2D checkpoint and carry the applicable work into **2D DGSEM, then 3D**. Cached VTK has broad basis support; DGSEM needs its own GLL state/IGR admission checks and its SBP/two-point volume formulation preserved. Its mass path is already separate and often diagonal. Current DGSEM production admits only 1D/2D, and the new Euler tensor kernels are 2D: 3D needs hex-face orientations, five-component flow, three-direction contractions and a memory design that avoids large dense element matrices. Shared-memory findings should inform that design. No DGSEM/3D performance claim follows from this checkpoint.

All new run/build scratch used node `/tmp`; installed dependencies were reused. Only selected reports, current qualified binaries and final comparison fields remain. The duplicate baseline snapshot is a verified hardlink to existing identical bytes, superseded native binaries were removed, and raw failed profiler evidence is retained for the explicit recovery. No merge or new scheduled automation was performed.
