# Getting started with IGRlab

[Project overview](../README.md) · [Methods](methods.md) · [Examples](../examples/)

## CPU build

You will need a C++20 compiler, [Spack](https://spack.io/), and MPI. From a
source checkout:

```bash
git submodule update --init
source /path/to/spack/share/spack/setup-env.sh
./build.sh --cpu-only --hdg-runtime -j 4
```

The initial build installs dependencies, builds the pinned MFEM submodule and
IGRlab, and runs CTest. DG and DGSEM are built by default; `--hdg-runtime`
enables the HDG application. Use `./build.sh --help` for all build options.

Build outputs live outside the checkout under
`${XDG_CACHE_HOME:-$HOME/.cache}/igrlab-build` by default. The printed build
plan gives the exact `app_build` directory. Add `--print-plan` to a build
command to inspect its paths without building.

## First run

Run a short DG Sedov blast with IGR. Set `APP_BUILD` to the `app_build` path
printed by your CPU build:

```bash
APP_BUILD=/path/printed/as/app_build
"$APP_BUILD/examples/sedov_taylor_blast/sedov_taylor_blast" \
  --device cpu --t-final 0.001 -rs 0 -rp 0 -o 1 -c 0.25
```

Executables remain in the build tree under `examples/<case>/`. Use `--help`
for an example's options. The [examples](../examples/) include stationary
shocks, shock–vortex interaction, supersonic flow, and smooth periodic
advection. Add `--smoke` to a build command for the driver's short two-rank
Sedov check.

## CUDA builds

CUDA builds need a matching NVIDIA toolchain. Choose the profile for your GPU:

```bash
./build.sh --gpu --cuda-arch 120 --hdg-runtime -j 4  # SM120
./build.sh --a100 --hdg-runtime -j 2                # A100
```

Choose parallel build jobs for the host's available memory and CPU allocation.
See [A100 setup](a100.md#practical-setup) for environment requirements and
[method support](methods.md#hardware-and-order-support) for architecture and
configuration limits.

## PACE and Delta

On clusters, use the checked-in site profiles to submit builds and validation
on compute nodes. The [PACE guide](pace.md) and [Delta guide](delta.md) cover
each site's setup, queues, and runtime. Keep dependency and build roots
separate between sites.

## Iterating on the application

After configuring once, [IGRlab-only iteration](iteration.md) uses
`--iterate --target NAME` to rebuild selected application targets while
reusing dependencies. Polynomial order is selected at runtime and needs no
rebuild. The [GPU study workflow](gpu-studies.md) provides DOF-normalized
comparisons and selective Nsight profiling on both sites.
