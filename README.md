<p align="center">
  <img src="docs/assets/igrlab-logo-adaptive.svg" alt="IGRlab" width="500">
</p>

<p align="center"><strong>Information-geometric regularization for Euler flows</strong></p>

IGRlab is a research code for studying information-geometric regularization
(IGR) in compressible Euler flows, including shocks and smooth transport.
It uses [MFEM](https://mfem.org/) for finite-element meshes, spaces, and
operators, and provides three spatial discretizations:

| Method | CPU and CUDA implementations | Manual tensor-core kernels |
| --- | --- | --- |
| Discontinuous Galerkin (DG) | Euler and IGR operators | TF32 IGR patch preconditioner |
| DG spectral element method (DGSEM) | Reference and packed operators | FP64 fused residual |
| Hybridizable DG (HDG) | Euler and IGR solvers | FP64 volume and condensation |

The specialized CUDA kernels use explicit warp-level matrix
multiply-accumulate (WMMA) instructions. Their support depends on element
geometry, polynomial order, arithmetic policy, and GPU architecture.
The generic HDG application supports conforming 2D quadrilateral meshes with
runtime-selected polynomial order, subject to dimension and workspace limits.
Its CUDA path admits one or two MPI ranks. See [A100 support](docs/a100.md)
for device architecture, build requirements, and qualified configurations.

## Build

You will need a C++20 compiler, [Spack](https://spack.io/), and MPI. CUDA
builds also need the matching NVIDIA toolchain. The initial build installs
dependencies, builds the pinned MFEM submodule and IGRlab, and runs CTest.
From a source checkout:

```bash
git submodule update --init
source /path/to/spack/share/spack/setup-env.sh
./build.sh --cpu-only --hdg-runtime -j 4
```

DG and DGSEM are built by default. `--hdg-runtime` enables the HDG application.
For CUDA, choose the profile for your GPU:

```bash
./build.sh --gpu --cuda-arch 120 --hdg-runtime -j 4  # SM120
./build.sh --a100 --hdg-runtime -j 2                # A100
```

Build outputs live outside the checkout under
`${XDG_CACHE_HOME:-$HOME/.cache}/igrlab-build` by default. The printed build
plan gives the exact `app_build` directory; `--print-plan` shows it without
building. Run `./build.sh --help` for cache and research-test options.
[A100 setup](docs/a100.md#practical-setup) covers site environments and GPU
build settings.

On PACE and Delta, use the checked-in site profiles to submit builds and
validation on compute nodes. The [PACE guide](docs/pace.md) and
[Delta guide](docs/delta.md) cover each site's setup, queues, and runtime.
Keep dependency and build roots separate between sites.

After configuring once, [IGRlab-only iteration](docs/iteration.md) uses
`--iterate --target NAME` to rebuild selected application targets while
reusing dependencies. Polynomial order is selected at runtime and needs no
rebuild. The [GPU study workflow](docs/gpu-studies.md) provides DOF-normalized
comparisons and selective Nsight profiling on both sites.

## First run

Run a short DG Sedov blast with IGR. Set `APP_BUILD` to the `app_build` path
printed by your CPU build:

```bash
APP_BUILD=/path/printed/as/app_build
"$APP_BUILD/examples/sedov_taylor_blast/sedov_taylor_blast" \
  --device cpu --t-final 0.001 -rs 0 -rp 0 -o 1 -c 0.25
```

Executables remain in the build tree under `examples/<case>/`. Use `--help`
for an example's options. The [examples](examples/) include stationary
shocks, shock–vortex interaction, supersonic flow, and smooth periodic
advection. Add `--smoke` to a build command for the driver's short two-rank
Sedov check.

## Numerical notes

This is experimental research software.
[Consolidation validation](docs/consolidation/validation.md) records selected
CPU and SM120 correctness checks.
[Current A100 results](docs/a100.md#current-architecture-multiorder-follow-up-2026-10-15)
include multiorder field comparisons, targeted CUDA/MPI checks, and
DOF-normalized end-to-end timings. Results apply to the recorded source and
configurations; they do not establish arbitrary mixed-hp support, multi-node
performance, or long-time suppression of ringing. Dense HDG storage remains
a scaling limit.

Conservative modal damping (M16) is opt-in for affine, uniform-order nodal
L2 DG/DGSEM tensor cells. It preserves physical cell averages, but does not
guarantee positivity or elimination of ringing. Enable it with
`--modal-damping-rate 80000 --igr-sigma-cadence every-stage`; zero rate
disables damping. When damping is enabled, active IGR requires a closure
solve at every stage. For a positive rate, the timestep is capped at
`1/rate`; the coupled system may need a smaller step. See [bounded damping validation](docs/consolidation/validation.md#bounded-application-damping-comparison).

Further documentation:

- [Numerical validation](docs/numerical-validation.md): accuracy and stability checks for optimization.
- [Validation notes](docs/consolidation/validation.md): tested source and configurations.
- [Scalar HDG oracle](tests/hdg/README.md): independent CPU checks of the scalar formulation.
- [Consolidation notes](docs/consolidation/README.md): compatibility, retained research, and recovery records.
