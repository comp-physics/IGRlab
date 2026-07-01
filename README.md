# dg_igr_miniapp
Repository for experimenting with IGR in a DG context using MFEM as a backend

## Building

MFEM is pinned as the `tpl/mfem` submodule and built locally so this project can
consume its installed CMake target. Spack supplies MFEM's dependencies.

Initialize the submodule and load Spack's shell support once:

```bash
git submodule update --init
source /path/to/spack/share/spack/setup-env.sh
```

Then use the unified build driver:

```bash
./build.sh --cpu-only -j 16
./build.sh --gpu --cuda-arch 120 -j 8
```

Spack environments live under `spack-envs/`. The checked-in profiles are
`cpu` and `cuda-sm120`; the latter uses CUDA 13 and HYPRE `develop`. Future
datacenter profiles follow the same convention, such as `cuda-sm80` for A100,
`cuda-sm90` for H100/H200, and `cuda-sm100` for B200/GB200.

The concrete `--cuda-arch` selects the dependency profile and isolated build
directories. MFEM's CUDA code generation defaults independently to
`80-virtual` PTX to avoid CUDA 13's expensive native `sm_120` compilation.

Both commands install dependencies, rebuild MFEM, build and install the app,
and run CTest. Useful controls include `--no-deps`, `--no-clean`, `--no-tests`,
and `--smoke`. Run `./build.sh --help` for the full interface.
