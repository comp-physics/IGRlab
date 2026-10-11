# Building on PACE

The supported entry point is the repository's `build.sh`. The PACE front end
loads the checked-in site configuration, prepares an isolated Spack environment,
and invokes that same driver. It does not require a campaign script or patches to
MFEM. The profile uses the free `gts-sbryngelson3` account and `embers` background
queue.

Initialize the pinned MFEM submodule after cloning:

```bash
git submodule update --init tpl/mfem
```

Use an absolute scratch output path and the installed Spack core checkout:

```bash
./build.sh --site pace --a100 --hdg-runtime --submit \
  --output-root /absolute/scratch/igrlab-build \
  --spack-root /absolute/path/to/spack/core
```

For a CPU build, replace `--a100` with `--cpu-only`. Add `--nvtx-profile` for CUDA
profiler binaries. Add `--test-regex REGEX` to select a focused validation suite,
or `--no-tests` to compile only. `--no-tests` is a build result, not a validation
result. Use the same command without `--submit` inside an existing free compute
allocation. Builds and timings must run on compute nodes.

Spack can also be initialized in the login shell once; then `--spack-root` is
optional. Alternatively set `IGRLAB_PACE_SPACK_ROOT`. The profile expects the
separate builtin recipe checkout at `../packages/repos/spack_repo/builtin`
relative to the Spack core directory; set `IGRLAB_PACE_RECIPE_ROOT` when the
installation uses another layout. The tested tool is Spack `1.3.0.dev0`; the
dependency environment retains its concrete lock between runs.

`--submit` prints job IDs and log paths. Compilation uses `cpu-small`, one task,
eight CPUs, 48 GiB and an eight-hour limit. When tests are enabled, a dependent
validation job uses two hours, four MPI slots with two CPUs each, and 48 GiB. GPU
validation requests one A100 on `gpu-a100`; CPU validation uses `cpu-small`.
Compilation therefore does not occupy a GPU. The validation stage reruns the
canonical driver with installed dependencies and existing build outputs.
MPI tests use the selected site's `mpiexec` directly from the batch shell.

Inspect a plan without submitting or changing output directories:

```bash
./build.sh --site pace --a100 --hdg-runtime --submit \
  --output-root /absolute/scratch/igrlab-build --print-plan
```

The default is incremental reuse. `--clean` explicitly rebuilds the selected
application and MFEM caches. Dependencies, Spack scopes and submission requests
live under the output root; application and MFEM artifacts retain the driver's
worktree/profile layout. `--site-root DIR` optionally shares a dependency root
across distinct output roots. A changed site manifest is rejected with an
actionable message; select a new dependency root rather than silently changing
an existing graph.

The checked-in [site definition](../cmake/sites/pace.json) selects GCC 12.3.0,
CMake 3.30.2, METIS 5.1.0, OpenBLAS 0.3.23 and HYPRE 3.1.0 in double precision.
CPU uses site OpenMPI 4.1.5 without CUDA. A100 uses site OpenMPI 5.0.9 with CUDA
12.9.1 and Umpire 2025.12.0. Every dependency uses the portable `x86_64_v3`
target, suitable for PACE's AMD A100 hosts. Dependency, MFEM and application CUDA
code generation all target native SM80. The site profile checks the concrete
graph before installing dependencies or compiling MFEM.

The canonical driver refreshes HYPRE discovery and checks the actual cached
headers/libraries, installed MFEM interface and runtime library bindings. Merely
changing `HYPRE_DIR` does not make a CMake cache safe: `FindHYPRE` can retain old
paths. These checks prevent those paths from being accepted as a successful
build. The HDG source preflight also requires MFEM's pinned `v4.9` tag. Include
`refs/tags/v4.9` in MFEM Git bundles; a commit-only bundle loses the release
description used by the face oracle.

PACE CPU/A100 runs are the performance authority for this work. Local builds
may verify compilation and correctness, but local RTX 5080 timings do not guide
these optimizations. The site profile uses ordinary batch children and the
validated MPI installation without overriding MPI transport components.
