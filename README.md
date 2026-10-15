<p align="center">
  <img src="docs/assets/igrlab-logo-adaptive.svg" alt="IGRlab" width="500">
</p>

<p align="center"><strong>Information-geometric regularization for Euler flows</strong></p>

<p align="center">
  <a href="docs/getting-started.md">Get started</a> ·
  <a href="docs/methods.md">Methods</a> ·
  <a href="#examples">Examples</a> ·
  <a href="#validation-and-performance">Validation</a>
</p>

IGRlab explores how **information-geometric regularization (IGR)** shapes
compressible Euler flows, from smooth transport to strong shocks. Built on
[MFEM](https://mfem.org/), it brings DG, DGSEM, and HDG into a shared research
codebase for studying numerical behavior and computational cost on CPUs and GPUs.

**Experimental research software.** Capabilities and validation vary by method,
polynomial order, and hardware; see the [method guide](docs/methods.md) for scope.

## Three approaches to Euler flows

| Method | Implementation |
| --- | --- |
| **DG** — discontinuous Galerkin | Euler and IGR operators |
| **DGSEM** — DG spectral element method | Reference and packed operators |
| **HDG** — hybridizable discontinuous Galerkin | Euler and IGR solvers with local elimination |

CPU and CUDA implementations support comparisons across polynomial orders.
Specialized tensor-core kernels are available for selected configurations.
[Explore the methods and their supported configurations →](docs/methods.md)

## Get started

From a source checkout, with a C++20 compiler, MPI, and [Spack](https://spack.io/):

```bash
git submodule update --init
source /path/to/spack/share/spack/setup-env.sh
./build.sh --cpu-only --hdg-runtime -j 4
```

Then run a short DG Sedov blast with IGR, using the `app_build` path printed
by the build:

```bash
APP_BUILD=/path/printed/as/app_build
"$APP_BUILD/examples/sedov_taylor_blast/sedov_taylor_blast" \
  --device cpu --t-final 0.001 -rs 0 -rp 0 -o 1 -c 0.25
```

[Full setup and CUDA builds](docs/getting-started.md) ·
[PACE](docs/pace.md) · [Delta](docs/delta.md) ·
[Fast application-only rebuilds](docs/iteration.md)

## Examples

| Flow problem | Example |
| --- | --- |
| Strong blast wave | [Sedov–Taylor blast](examples/sedov_taylor_blast/) |
| Shock structure and interaction | [Stationary shock](examples/dgsem_stationary_shock/) · [Shock–vortex interaction](examples/dgsem_shock_vortex_interaction/) |
| Supersonic flow around a body | [Cylinder](examples/dgsem_supersonic_cylinder/) |
| Smooth transport and method comparisons | [Periodic advection](examples/dgsem_smooth_advection_periodic/) |

Browse [all examples](examples/); each executable provides its options with `--help`.

## Validation and performance

Accuracy, conservation, stability, and cost per degree of freedom guide
development. The [current A100 results](docs/a100.md#current-architecture-multiorder-follow-up-2026-10-15)
report field comparisons across orders, targeted CUDA/MPI checks, and
DOF-normalized end-to-end timings for the recorded source and configurations.

- [Numerical validation](docs/numerical-validation.md) — how changes are assessed.
- [GPU studies](docs/gpu-studies.md) — reproducible comparisons and Nsight profiling.
- [Validation records](docs/consolidation/validation.md) — selected CPU and SM120 checks.
- [Method scope and limitations](docs/methods.md#validation-scope) — what the evidence covers.
