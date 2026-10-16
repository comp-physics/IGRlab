# Experimental DG/DGSEM tensor IGR pipeline

This opt-in prototype accelerates the IGR elliptic solve shared by DG and
DGSEM. The Euler state, current physical coefficients, outer FGMRES vectors,
and final original-operator residual are FP64. Defaults are unchanged.
This document describes the implementation contract; measured results are
recorded separately with the tested source and binary identity.

## Selection

Use the CUDA backend and explicit FGMRES with
`IGRLAB_IGR_KRYLOV_BACKEND=cuda-fp64`. The existing exact DG face operator
and device quadrature coefficients are required.

| Environment variable | Values | Purpose |
| --- | --- | --- |
| `IGRLAB_IGR_TENSOR_MG` | `off`, `fp64`, `fp32`, `tf32` | Select the p-multigrid preconditioner. |
| `IGRLAB_IGR_TENSOR_MG_REBUILD_INTERVAL` | nonnegative integer, default `0` | Rebuild coefficient-dependent frozen preconditioner data every N solves; `0` reuses the first setup. |
| `IGRLAB_IGR_TENSOR_VOLUME` | `off`, `simt`, `wmma` | Use the original volume operator or the fused FP64 volume implementation with ordinary arithmetic or FP64 WMMA. |

Do not combine tensor multigrid with the dense inverse, two-level inverse,
or screened-patch experiment selectors. Keep shared Euler operator selection
`off` when isolating these changes. Every tensor pipeline solve independently
checks the original current-coefficient FP64 equation using the requested
relative-to-RHS or absolute tolerance. A finite unconverged iterate is not an
accuracy result; benchmark configurations must reject nonconvergence.

## Numerical structure

The fine equation retains the current mass, variable diffusion, IPDG face
coupling, penalty, geometry, and quadrature. Fused volume application caches
reference and geometry data, refreshes current scalar coefficients, and adds
the existing exact face action. Mass and diffusion quadratures are separate.
It uses FP64 Tensor Cores; TF32 is reserved for preconditioning.

The frozen preconditioner constructs a p5 → p3 → p1 Galerkin hierarchy in
nested tensor Legendre coordinates. Each V-cycle pre-smooths, restricts the
residual, recursively corrects, prolongates, and post-smooths. Overlapping
patch smoothers use shared separable transforms applied to distinct patch
lines. The mixed modes store level work in FP32; TF32 mode uses actual WMMA
products with FP32 accumulation. The lowest sparse solve remains FP64.
Galerkin coarsening preserves the fine equation's physical alpha; it does
not replace it with an alpha computed from a coarser mesh length.

The first prototype is intentionally limited to one CUDA rank, uniform
axis-aligned periodic quadrilaterals, scalar nodal L2 spaces, and the admitted
p5 hierarchy. The fused volume path has its own explicit tensor-layout and
quadrature bounds. Unsupported configurations must fail admission. Setup
samples and stores the frozen local stencil; its time and memory count in
end-to-end comparisons. Reuse is an explicit preconditioner approximation,
not a license to freeze the physical equation.

## Comparison contract

Compare the existing FP64 solver, all-FP64 multigrid, FP32 multigrid, TF32
multigrid, and the fused-operator controls using identical mesh, timestep,
final time, discretization, output policy, and physical coefficients.
Validate every solve, finite positive density/pressure, conservation, and
saved-polynomial field differences against the same-discretization FP64
reference. These differences are not an analytic PDE error estimate.

Report cold setup and complete simulation wall time, repeated alternating
output-free timings, outer iterations, memory, and actual Tensor Core
execution evidence. Separate accuracy-budget relaxation from precision
changes. A faster contraction alone is not an end-to-end speedup.
