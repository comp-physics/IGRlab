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
./build.sh --gpu --cuda-arch 120 -j 16
```

Application and locally built MFEM outputs live outside the checkout. By default they are scoped under
`${XDG_CACHE_HOME:-$HOME/.cache}/dg-igr-build` by canonical worktree and
profile; choose another external location with `--output-root DIR` or
`DG_IGR_OUTPUT_ROOT`. A selected output root must be outside every detected Git
checkout.

```bash
./build.sh --cpu-only --output-root /scratch/$USER/dg-igr -j 16
./build.sh --gpu --cuda-arch 120 --print-plan
```

`--print-plan` reports the output root, deterministic worktree/profile scope,
and resolved app/MFEM paths without creating output-root content or acquiring
the output lock. `build_mfem_tpl.sh` is an internal helper; invoke `build.sh`
for all builds.

Spack environments live under `spack-envs/`. The checked-in profiles are
`cpu` and `cuda-sm120`; the latter uses CUDA 13 and HYPRE `develop`. Future
datacenter profiles follow the same convention, such as `cuda-sm80` for A100,
`cuda-sm90` for H100/H200, and `cuda-sm100` for B200/GB200.

GPU architecture selection has three deliberately separate roles. For the
checked-in Blackwell profile, `--cuda-arch 120` selects Spack dependencies and
isolated build directories with `cuda_arch=120`; MFEM is compiled with
`MFEM_CUDA_ARCHS=80-virtual` PTX to avoid CUDA 13's expensive native `sm_120`
compilation; and this project's CUDA sources are compiled natively with
`APP_CUDA_ARCHS=120`. The driver enforces that split before dependency or cache
actions and rejects the ambiguous legacy `CUDA_ARCHS` variable. These values
normally do not need to be set explicitly.

The application and MFEM builds both use `-j 16` in the example above. Use
`--mfem-jobs N` only when MFEM needs different concurrency, such as:

```bash
./build.sh --gpu --cuda-arch 120 -j 16 --mfem-jobs 8
```

`./build.sh --gpu --cuda-arch 120 --print-plan` validates and prints the
resolved architecture and job policy without invoking Spack, CMake, cache
cleanup, or a build.

Both commands install dependencies, rebuild MFEM, build and install the app,
and run CTest. Useful controls include `--no-deps`, `--no-clean`, `--no-tests`,
and `--smoke`. Run `./build.sh --help` for the full interface.
