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
and `IGRLAB_IGR_LOCAL_TENSOR_EXECUTION=wmma-factor-fp64`. The proposed comparison
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
and native VTK parity. Native qualification passed as recorded below. Large-case accuracy and
performance remain pending; no measured DGSEM speedup is claimed here.

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


## Evolved Nsight Systems evidence

Job **3446741** captured two continuous t=0 trajectories, each after a common
120-second delay, with 15 seconds of collection on the same GH200. Both use
native initial VTK and diagnostic output interval 0.1; each profiling process
is intentionally terminated after capture. SQLite analysis stays on DeltaAI,
and scratch SQLite is removed. Native reports remain remote.

The strict launch/activity selection retains **368 baseline / 537 improved
complete RK steps**, each with three stages and exactly two current-A/residual
and two local-P/update applications per stage. Recorded GPU interval union
falls **40.138 → 27.439 ms/RK (31.64% less)**. The windows reach different
evolved states; this is descriptive GPU evidence, not full-run wall timing.

| Recorded GPU scope, ms/stage | Baseline | Improved |
|---|---:|---:|
| Coefficients/RHS and A/P refresh | 6.197 | 1.939 |
| Current A and residual | 2.864 | 2.888 |
| Local P and solution update | 0.435 | 0.438 |
| DGSEM Euler spatial residual/faces | 2.894 | 2.892 |
| CFL reductions | 0.043 | 0.044 |
| Common RK work | 0.417 | 0.416 |
| Other recorded closure copies | 0.529 | 0.529 |

Preparation accounts for the gain; other major kernels remain close in cost.
The remaining leading kernels are compact IGR volume (1.539 ms/stage across
two applications), the fused DGSEM Euler residual (1.372 ms/stage), and the
exact face record kernel (0.974 ms/stage across two applications). The local
FP64 WMMA inverse plus update is 0.438 ms/stage; this scope contains tensor-core
operations, but its duration is not a measure of tensor-core FLOPs/utilization.

Driver 13.2 / trace-library 12.9 compatibility and possible collection-loss
warnings remain. Incomplete boundary RK records are excluded by the strict
selection; there is no claim of lossless tracing. The
[profile record](benchmarks/dgsem_mapped_nsys.json) includes source/binary/GPU,
selected counts, native report hashes and warnings. The full three-way t=0.1
comparison is submitted as **3446752**; its physical and end-to-end results
remain pending. No large field or native profile was downloaded to WSL.
