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
