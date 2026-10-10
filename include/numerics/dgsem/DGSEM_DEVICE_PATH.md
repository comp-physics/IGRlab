# DGSEM correctness device path

Scope: Wave 2 correctness only, dimensions 1 and 2, polynomial orders p1--p4.
The accepted CPU oracle and every tolerance in `dgsem_reference.hpp` remain the
reference. This path neither repairs B-SCI-001 nor claims production wall
stability, admissibility, performance, or support for TIGRE/HIGR equations.

## Execution inventory

| Consumer | Device execution in this lane | Host setup/control | Blocked or excluded |
|---|---|---|---|
| LGL basis identities | Moments, derivative action, SBP, and split-form checks use `mfem::forall` over `Vector`/`DenseMatrix` memory. | MFEM builds and caches p1--p4 LGL data once. | No runtime basis construction on device. |
| Curl-form metrics | Packed 2D nodal coordinate derivatives and adjugate rows execute on device. | FE discovery, geometry sampling, and tensor local-to-lex mapping. | 3D curl-form metrics remain outside the honest 1D/2D scope. |
| Volume flux | Static Euler plus accepted stage-frozen scalar-sigma tensor flux and central contravariant contraction execute on device. | Model selection, state/sigma packing, and mesh traversal. | Entropy/dissipative variants still depend on legacy runtime-polymorphic callbacks and are not claimed as device kernels. |
| Local faces | Static Euler Rusanov flux and deterministic equal-and-opposite gather execute on device. | MFEM face discovery, nodal lookup, normal construction, and right-side orientation map. | Diffusive and entropy-stable runtime-selected face models are not claimed. |
| Shared faces | Static Euler Rusanov flux executes on device and updates only the locally owned left entry. | `ParGridFunction::ExchangeFaceNbrData`, neighbor unpacking, and orientation map. | This lane does not replace MFEM exchange or claim GPU-aware MPI. |
| Boundary faces | Static Euler Rusanov flux executes after the ghost state is supplied. | Boundary-integrator selection and boundary-state evaluation. | Runtime-polymorphic boundary callbacks remain host-side. |
| Inverse mass | Packed per-element dense action executes on device. | Existing inverse-mass assembly and packing. | No tensor-core or performance specialization. |
| Auxiliary sigma | A prescribed, stage-frozen scalar is read in the static Euler tensor/face kernels without mutation. | Existing scalar IGR field ownership and stage lifecycle. | No scalar-IGR solver edit, TIGRE/HIGR equation, or wall-stability claim. |
| Reductions | Component sums and nonnegative maxima execute in deterministic MFEM kernels. | Host reads the final scalar/component results for assertions and CFL control. | No parallel-performance reduction claim. |
| Exchange | Packed neighbor states are device consumers after exchange. | MFEM MPI exchange remains host control in this correctness lane. | Multi-GPU execution requires an explicit safe resource map. |
| Output/restart | None. | JSON evidence, diagnostics, output, and restart remain host consumers after explicit synchronization. | No output/restart edit is authorized. |

The face kernel receives explicit left/right element and local-DOF maps. A
`Local` face writes both sides with opposite signs; `Shared` writes only its
locally owned left side; `Boundary` writes only the interior side. The normal
always points out of the left side, and host-resolved right orientation is
consumed without reinterpretation. One packed host preflight checks that every
normal is nonzero before the MFEM kernel launches; it does not synchronize once
per face point.

The focused parity fixture uses one dimension-parameterized implementation for
direct p1--p4 checks in dimensions 1 and 2. Euler plus prescribed stage-frozen
scalar sigma, central volume, constant cancellation, inverse mass,
local/shared/boundary faces, component sums, and maximum reduction execute in
both dimensions. Curl-form metric construction is recorded only for dimension
2; the 1D flux fixture consumes its exact affine contravariant metric as input.
The machine-readable evidence records `dimensions: [1,2]`, per-dimension
errors, and this kernel-specific coverage boundary.

## Integration boundary

`src/numerics/dgsem/dgsem_device.cpp` must be compiled with the configured
accelerator language whenever MFEM CUDA/HIP is enabled. This worker does not
own `src/CMakeLists.txt`; local evidence supplies the file through the existing
`DG_IGR_ACCELERATOR_KERNEL_SOURCES` cache variable. The coordinator-owned
integration is one entry in `_dg_igr_default_accel_sources`:

```cmake
"src/numerics/dgsem/dgsem_device.cpp"
```

No root CMake, test registration, backend registration, or generic DG operator
change is required for review of this lane.

Research extension: `DG_IGR_DGSEM_HIGH_ORDER_RESEARCH=1` admits p5--p7 on the existing 1D/2D packed application path. The 2D fused residual has matching ordinary-CUDA/FP64-tensor instantiations through p7. Independent kernel and application residual parity checks cover p1--p7; this does not qualify every nonlinear shocked trajectory or change the default p1--p4 policy.

`DG_IGR_DGSEM_SUBCELL_BLEND=1` selects a separate experimental numerical method on the fused CUDA p3--p7 affine-quadrilateral path. Density/pressure Legendre modal energy supplies one blend coefficient per element (`DG_IGR_DGSEM_SUBCELL_THRESHOLD`, default `1e-4`). The high-order residual is blended with a conservative adjacent-LGL-subcell LLF residual; both reuse the unique macroface flux and the same instantaneous sigma. Native Euler polynomials are retained. This changes the transport discretization, and is not an algebraically equivalent coordinate transform or an IGR entropy/positivity proof. Setup, sensor computation, private aggregate accounting, and low-order work are included in application timing. The final local-rank summary records the fraction of element/RHS visits that remained entirely DG; it includes rejected work.
