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
and native VTK parity. Numerical qualification and performance results are
pending; no measured DGSEM improvement is claimed by this source change.
