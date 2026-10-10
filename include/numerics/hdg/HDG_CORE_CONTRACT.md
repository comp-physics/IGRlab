# HDG Core Contract

This directory exposes a model-neutral C++17 contract for later HDG CPU,
MPI, and device work. It does not register an HDG operator or choose a
physical model.

The accepted scalar oracle supplies the algebraic boundary. For each element,
the logical blocks remain explicit:

```text
R = A x + B lambda - f
G = C x + D lambda
```

Local residual/Jacobian evaluation, condensation, Schur action, trace
assembly, the global trace solve, and interior recovery are distinct static
dispatch calls. The contract neither substitutes `mfem::Hybridization` nor
requires an independent implementation; later reference and packed routes can
both implement these boundaries.

`HdgTopologyStorage` owns host construction arrays. `HdgTopologyView`, the
metadata records, block views, workspace views, and trace contributions are
trivially copyable. A backend is responsible for mirroring those arrays into
MFEM-managed, RAJA/Umpire-managed, or raw accelerator storage. The contract
does not allocate device memory, move data, or depend on Kokkos. POD view
accessors are host/device callable under CUDA or HIP compilation.

Trace fields are compile-time tuples of named scalar, vector, or tensor
potentials. Their component offsets are fixed by type without creating a
runtime model hierarchy. A `KernelSpec` binds model, trace, stabilization,
time-treatment, backend, precision, and model-owned POD device-policy types.
It selects none of those types itself.

`StridedBlockView` describes logical batched matrices with independent batch,
row, and column strides. Consequently, later implementations can compare
array-of-matrices, interleaved, MFEM `DenseTensor`, vendor-library, and custom
packing without changing the HDG phase interfaces. `WorkspaceLayout` likewise
reserves persistent residual, Jacobian, factorization, condensed-system, and
temporary regions without prescribing an allocator.

Non-goals for this contract are numerical fluxes, stabilization formulas,
Euler/IGR/TIGRE equations, wall treatment, time integration, nonlinear or
linear solver algorithms, MPI exchange, accelerator kernels, tensor-core
packing, runtime selection, production wiring, and default activation. Shared
observability remains independently gated and default-off.
