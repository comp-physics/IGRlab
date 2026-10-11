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
The generic HDG application currently supports conforming 2D
quadrilateral meshes at orders 1–3; its CUDA path admits one or two MPI ranks.
See [A100 support](docs/a100.md) for device architecture and build requirements.

## Build

You will need a C++20 compiler, [Spack](https://spack.io/), and MPI. CUDA
builds also need the matching NVIDIA toolchain. The build driver installs
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
[Historical A100 results](docs/a100.md#historical-hardware-evidence) are tied
to their original source/software tuples. These records do not establish
fresh A100 qualification of the current source, multi-node performance,
or long-time suppression of ringing.

Conservative modal damping (M16) is opt-in for affine, uniform-order nodal
L2 DG/DGSEM tensor cells. It preserves physical cell averages, but does not
guarantee positivity or elimination of ringing. Enable it with
`--modal-damping-rate 80000 --igr-sigma-cadence every-stage`; zero rate
disables damping. When damping is enabled, active IGR requires a closure
solve at every stage. For a positive rate, the timestep is capped at
`1/rate`; the coupled system may need a smaller step. See [bounded damping validation](docs/consolidation/validation.md#bounded-application-damping-comparison).

Further documentation:

- [Validation notes](docs/consolidation/validation.md): tested source and configurations.
- [Scalar HDG oracle](tests/hdg/README.md): independent CPU checks of the scalar formulation.
- [Consolidation notes](docs/consolidation/README.md): compatibility, retained research, and recovery records.
