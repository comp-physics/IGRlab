# Mapped DGSEM spectral-correction transfer

This experimental follow-on to the mapped ordinary-DG checkpoint admits p5
GLL state and sigma to the projected element-local spectral inverse and fused
stage/face coefficient preparation. The current global FP64 IGR equation,
exact SIPG faces, and native GL7 mass / GL6 diffusion and source quadrature
remain unchanged on Q2 mapped quadrilaterals. Euler retains its own GLL/SBP
formulation; its diagonal mass does not replace the consistent IGR mass.

The approximate candidate is explicitly selected with
`--igr-solve-mode element-block-jacobi --igr-jacobi-blocks tensor`,
`IGRLAB_IGR_SIGMA_BASIS=lgl`, `IGRLAB_IGR_LOCAL_TENSOR_REFERENCE=projected`,
and `IGRLAB_IGR_LOCAL_TENSOR_EXECUTION=wmma-factor-fp64`. The qualified comparison
uses 50 initial and two regular corrections with damping 0.7. It does not
certify sigma convergence. The default Krylov closure is unchanged; fixed
closure still rejects restart, Zhang–Shu limiting, and RK abort/retry.

GLL's actual GL-quadrature mass has off-diagonal entries even on affine cells.
The projected inverse is therefore checked against an independently assembled
projected surrogate, not claimed equal to the exact element inverse. The
original separable reference remains GL-only.

Fused coefficient refresh and tensor face-density evaluation now use actual
GL/GLL state bases. Face integration remains GL6, independently of the state
and scalar nodal bases. Current-state/native-layout checks and unsupported
fallbacks remain in place. Cached native VTK evaluation gains explicit GLL
parity coverage. Ordinary-DG dense weak-divergence, mass and lift reuse kernels
are not part of DGSEM's already fused Euler residual.

Native qualification covers projected GLL SIMT/WMMA against an independent
oracle, actual consistent mass, FP64 residual/update fusion, current coefficient
refresh, output/prefetch history, mapped DGSEM residual/CFL versus host assembly,
and native VTK parity. Native qualification and the full mapped-case comparison passed as recorded below.

## Native qualification

Canonical DeltaAI job **3446614** passed all **31 selected CPU/CUDA test groups**
on source `96c7ea05a6cd8a99c69008e85e242f66fba31876`, with unchanged installed
MFEM/Spack dependencies. The mapped p5 GLL wrapper fixture preserves 50/2 sweep
budgets, output-only history, and one consumed stage-zero prefetch. Its Euler
residual differs from separate host evaluation by at most 5.62e-16 in relative
maximum norm; the CFL maximum agrees exactly. The host oracle uses the base
operator's mass matrices, rather than independently assembling them.

Projected GLL SIMT and FP64 WMMA oracles, current consistent-mass scaling,
residual/update fusion, stage/face preparation, and native VTK parity all pass.
These small fixtures qualify algebra and execution. Their positive projected
operators do not imply stability of a fixed damping choice for every mapping;
the large-case trajectory and flow comparison are separate requirements.

The [native record](benchmarks/dgsem_mapped_native_validation.json) binds the
source, executable, GPU, test selection, canonical commands and log hashes.
The application binary SHA-256 is
`6338e9b0a1502c26af26706b80a726707da0e2dfe3c4623b667913524658694b`.

All large fields and profiler data remain on DeltaAI. Field analysis uses a
prebuilt ARM VTK wheel unpacked into job-local temporary storage; only compact
metrics and selected render images return to the local machine. Runtime/build
scratch is cleaned, while installed application dependencies are reused.


## Fixed-large short trajectory

DeltaAI job **3446717** completed three fresh same-binary runs on the same
GH200, each reaching t=0.0025 in 162 steps on the 512² Q2-mapped p5 GLL case.
The converged reference uses FP64 FGMRES/two-level preconditioning and
RHS-relative tolerance 1e-6. Both fixed arms use projected FP64 WMMA, 50/2
corrections at omega=0.7; only coefficient/face refresh, direct state and VTK
execution differ between them.

Remote native-VTK Q2/Q5 comparison at GL8×8 gives maximum conserved relative
L2 difference **2.875811589e-8** between the converged and fixed baselines.
Sigma differs by **0.2302145822 relative L2 (23.02%)** and is reported separately
from the flow budget. The two fixed arms have zero measured conserved/sigma
field differences; file hashes differ because writer metadata also differs.
Final sampled density, pressure and q_sigma checks pass. This is short-horizon
correctness evidence, not full-time stability or a performance claim.

The [compact pilot record](benchmarks/dgsem_mapped_short_validation.json) retains
raw timing scopes and the exact receipt hashes. All comparison work ran on
DeltaAI with prebuilt VTK 9.7.1 in job `/tmp`; all short-run snapshots were
removed after qualification. No large field was downloaded to WSL.


## Completed mapped DGSEM comparison

Source `96c7ea05a6cd8a99c69008e85e242f66fba31876` completed the full three-arm
comparison in DeltaAI job **3446752**, using one GH200
`GPU-961a75de-5e0f-dfa6-dbf1-36bcd07b960c`. The frozen application SHA-256 is
`6338e9b0a1502c26af26706b80a726707da0e2dfe3c4623b667913524658694b`.
Each arm starts freshly at t=0, reaches t=0.1 in **6,663 accepted steps**, and
performs **19,990 closure calls**. Of these, 19,989 are production calls and
one is output-only. Five output preparations are consumed as stage-zero
prefetches, with no invalidations. No RK attempts are rejected.

The fixed case is 512² Q2-mapped quadrilaterals, p5 GLL state and sigma,
262,144 elements, 9,437,184 scalar DOFs and 37,748,736 conservative DOFs.
L2/BrokenL2 IGR uses actual GL7×7 mass, GL6×6 diffusion/source and GL6 faces,
exact SIPG, alpha=(1/6)h_max², SSPRK3 and CFL 0.3. DGSEM uses central SBP
volume flux, Rusanov faces, its fused FP64 residual, no volume dissipation,
no overintegration and no Euler DMMA. Limiters, modal damping, routine
nonfinite/admissibility/conservation scans and restart output are off. Native
VTK output is scheduled at 0,.02,.04,.06,.08,.1, with direct pressure output
and H1/L2 output diagnostics off.

The reference uses FP64 FGMRES, RHS-relative tolerance 1e-6, and a retained
initial FP64 element/Q1 two-level preconditioner with cooperative FP64 coarse
CG. Its final independent current-FP64 residual certificate is
**9.29285184e-7**. Both fixed arms use projected local FP64 WMMA, 50 initial
and two regular damped corrections at omega 0.7. The improved arm changes
coefficient refresh, tensor face-density preparation, direct state access and
cached VTK execution. These two fixed arms share the same numerical policy.

| Host-wall scope, seconds | Converged reference | Fixed baseline | Fixed improved |
|---|---:|---:|---:|
| External to RunEulerCase | 25.274 | 24.352 | 23.709 |
| Setup + initial fields/output | 61.583 | 54.895 | 35.308 |
| Advancing solution + CFL | 1325.268 | 273.479 | 188.404 |
| Scheduled output | 149.665 | 146.958 | 65.075 |
| Scheduled diagnostics | 0.411 | 0.421 | 0.421 |
| Final output/post-loop | 37.823 | 36.629 | 16.177 |
| **Whole application** | **1600.025** | **536.734** | **329.094** |

The combined reference/improved comparison is **4.862× faster overall
(79.43% less time)** and **7.034× faster during advancement+CFL**. It includes
a change from a converged closure to fixed corrections. The execution-only
fixed pair is **1.631× faster overall (38.69% less time)** and **1.452× faster
during advancement+CFL (31.11% less time)**. Six native ParaView saves sum
to 216.402→93.186 s in that pair. Output savings therefore contribute to the
end-to-end gain; the overall factor is not a kernel-only speedup. The six
host bins are additive, but may contain asynchronous GPU work and are not a
GPU-time partition. Timed application scopes exclude builds, field hashing
and post-run comparison. This is one sequential same-GPU observation; no
replicate confidence interval or fixed-clock claim is made.

The reference reports 261,550 FGMRES iterations (13.084 per closure call),
301,529 solver-telemetry A applications and 261,550 P applications. Each fixed
arm reports **40,028 A and 40,028 P applications**, including initialization
and two final output-only corrections; production corrections total 40,026.
The reference's A telemetry includes its warm-start and final/restart true
residuals, so A count is greater than iterations. Independent original-A
certification and setup probes outside the telemetry wrapper are not included
in those reported counts. Regular fixed stages have exactly two current-A
residuals and two local-P updates. Fixed acceptance does not certify a residual
tolerance.

## Full saved-field qualification

Both predeclared remote gates pass. Complete saved Q5 conserved and sigma
polynomials are compared on matching ordered Q2 cells with GL8×8 physical
integration. Relative L2 is norm(candidate−reference)/norm(reference), per
component. Sampled Linf uses maxima at those GL8 points.

| Conserved component | Reference vs fixed-baseline relative L2 |
|---|---:|
| Density | 1.73357281e-7 |
| x momentum | 8.20829087e-7 |
| y momentum | 8.20829087e-7 |
| Energy | 2.61618671e-7 |

The maximum conserved relative L2 difference is **8.20829087e-7**; the maximum
conserved sampled relative Linf is **1.96102120e-6**. Sigma relative L2 is
**0.01529111176 (1.5291%)** and is reported separately: no new sigma
method-accuracy budget was approved. Baseline/improved execution differences
are zero in the measured complete saved conservative and sigma fields.
Reference/baseline measures the closure-policy difference; baseline/improved
measures execution equivalence. Finite positive sampled density, pressure and
q_sigma pass at final GL8 points. These are comparisons to a converged
discrete closure, not an analytic solution or discretization-error study, and
do not establish continuous or between-frame positivity.

Retained full raw-run SHA-256:
`72f0d2b766bed7c565c006836defeeedde7f34f150dac7a55d05fc2f1e35b6dd`;
qualification SHA-256:
`7990a8c7edf10216da0d864960510d3f8315ce3d4e647d6cc3d1fb37aa10d437`.
Only the reference and improved finals remain retained on DeltaAI. No large
field was downloaded to WSL.

## Evolved GPU accounting and hardware diagnosis

Qualified Nsight Systems job **3446741** retains 368 baseline and 537 improved
complete evolved RK steps on one GH200, distinct from the full-run GPU. Each
step has three stages and exactly two A/residual and two P/update applications
per stage. The profiles start at t=0, delay collection 120 s and collect 15 s;
the two windows reach different evolved states. Their recorded GPU interval
union is 40.138→27.439 ms/RK (31.64% less), independently descriptive of the
full unprofiled wall result.

The per-stage preparation group falls 6.197→1.939 ms. A/residual is
2.864→2.888 ms, P/update .435→.438 ms, and Euler spatial residual/faces
2.894→2.892 ms. Including preparation, current A, local P and closure copies,
sigma recorded GPU work is 10.024661→5.794110 ms/stage. Dividing by Euler
spatial residual/faces, with CFL excluded from that denominator, gives
**sigma:Euler = 3.4637→2.0033**. The phase-union sums match the complete
recorded GPU union to rounding on this pair. Current A/residual plus local P
alone is about 1.15× Euler; omitting preparation would understate closure cost.
Driver/trace-library compatibility and possible collection-loss warnings
remain; recorded launch reciprocity does not establish lossless acquisition.

A bounded launch-owned excerpt selects the chronologically first qualified
complete improved RK, rather than a duration-ranked step. It contains128
activities (101 kernels and27 copies), with a27.773473-ms device span and
27.476801-ms recorded interval union. Its times retain asynchronous ownership
and may extend beyond the host RK scope. The native2460×524 timeline uses one
linear axis, no widened intervals or gap compression, and three stage labels.
This one-step example is distinct from the capture-wide mean above.

NCU job **3446848** on the full-run GPU qualifies all **21 requested counters
plus LaunchStats for ten captured launches**. The native order is
V,F,V,F,E,V,F,V,F,E (four current-volume, four exact-face-record, two Euler).
The predeclared summary uses V[0,2], F[1,3], E[4,9], two samples per family;
these are six summarized launches, not six captured launches. The NVTX filter
selects RK/Step occurrence 1200; no absolute-step payload is available.

| Summarized family | Replay ms/launch | Long-scoreboard stalls % | MIO stalls % | Barrier stalls % | Active-warps occupancy % |
|---|---:|---:|---:|---:|---:|
| Compact current IGR volume | .779 | 11.33 | 30.15 | 8.95 | 72.75 |
| Exact SIPG face record | .489 | 61.07 | .94 | .00 | 68.22 |
| Fused DGSEM Euler residual | 1.352 | 47.10 | .41 | 25.06 | 44.46 |

The volume kernel also has 18.36% short-scoreboard stalls; Euler uses
60 registers/thread, versus 40/38 for volume/face. These observations point
to compact-volume data dependencies/reuse, face memory latency/reuse, and
Euler synchronization/register tradeoffs as useful next targets. They do not
identify one exclusive bottleneck or additive fractions of kernel wall time.
All four measured local-load/store counters are zero in each summarized
family, while LaunchStats reports a 1,568-byte stack; zero counter values do
not imply zero stack allocation or prove compiler spill counts.

Kernel replay, all-cache flush, dynamic boost and unlocked clocks make these
diagnostic durations distinct from application timing. This capture does not
measure the local-P WMMA kernel or tensor-pipe utilization. Both new stage
preparation and current A remain FP64 SIMT; only the selected local correction
uses FP64 WMMA. There is no new TF32 approximation.

The opt-in method is qualified for this fixed 2D mapped p5 case, with the
listed scans/limiters disabled. Positive projected operators and one completed
trajectory do not prove a general damping stability bound, mesh/p-order/MPI
scaling, or 3D performance. Historical Cartesian global-TF32 spectral-P results
are separate geometry/method comparisons and are not a matched speedup baseline.


Compact reproducible inputs and measurements: [full comparison](benchmarks/dgsem_mapped_full_comparison.json), [Nsight Systems](benchmarks/dgsem_mapped_nsys.json), [Nsight Compute](benchmarks/dgsem_mapped_ncu.json).
