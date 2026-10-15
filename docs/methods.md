# Methods and research scope

[Project overview](../README.md) · [Get started](getting-started.md) · [Examples](../examples/)

IGRlab uses [MFEM](https://mfem.org/) for finite-element meshes, spaces, and
operators. It provides three spatial discretizations for studying
information-geometric regularization in compressible Euler flows.

| Method | CPU and CUDA implementations | Manual tensor-core kernels |
| --- | --- | --- |
| Discontinuous Galerkin (DG) | Euler and IGR operators | TF32 IGR patch preconditioner |
| DG spectral element method (DGSEM) | Reference and packed operators | FP64 fused residual |
| Hybridizable DG (HDG) | Euler and IGR solvers | FP64 volume and condensation |

## Hardware and order support

Polynomial order is selected at runtime, without rebuilding the application.
The [order comparison workflow](gpu-studies.md#current-architecture-order-comparisons)
provides DG, DGSEM, and HDG commands at comparable conserved DOF counts.

The specialized CUDA kernels use explicit warp-level matrix
multiply-accumulate (WMMA) instructions. Their support depends on element
geometry, polynomial order, arithmetic policy, and GPU architecture.
Availability of a CPU or CUDA method does not imply that every configuration
has a specialized tensor-core kernel.

The generic HDG application supports conforming 2D quadrilateral meshes with
runtime-selected polynomial order, subject to dimension and workspace limits.
Its CUDA path admits one or two MPI ranks. Dense HDG storage remains a scaling
limit. See [A100 support](a100.md) for device architecture, build requirements,
and qualified configurations.

## Conservative modal damping

M16 modal damping is opt-in for affine, uniform-order nodal L2 DG/DGSEM tensor
cells. It preserves physical cell averages, but does not guarantee positivity
or elimination of ringing. It is not an HDG capability.

For example, damping can be enabled with:

```text
--modal-damping-rate 80000 --igr-sigma-cadence every-stage
```

A zero rate disables damping. When damping is enabled, active IGR requires a
closure solve at every stage. For a positive rate, the timestep is capped at
`1/rate`; the coupled system may need a smaller step. The example rate is a
configuration choice, not a general recommendation. See the
[bounded damping validation](consolidation/validation.md#bounded-application-damping-comparison)
for the tested cases and limitations.

## Validation scope

This is experimental research software. [Consolidation validation](consolidation/validation.md)
records selected CPU and SM120 correctness checks. The
[current A100 results](a100.md#current-architecture-multiorder-follow-up-2026-10-15)
include multiorder field comparisons, targeted CUDA/MPI checks, and
DOF-normalized end-to-end timings.

Results apply to the recorded source and configurations. They do not establish
arbitrary mixed-hp support, multi-node performance, or long-time suppression of
ringing. Consult the recorded configuration when choosing a method, order,
geometry, or device for a new study.

- [Numerical validation](numerical-validation.md): accuracy and stability checks for optimization.
- [Scalar HDG oracle](../tests/hdg/README.md): independent CPU checks of the scalar formulation.
- [Consolidation notes](consolidation/README.md): compatibility, retained research, and recovery records.
