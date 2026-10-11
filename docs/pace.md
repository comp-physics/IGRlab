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

Initialize the tested build Spack once. The driver queries its root with
`spack location -r`; then use an absolute scratch output path:

```bash
./build.sh --site pace --a100 --hdg-runtime --submit \
  --output-root /absolute/scratch/igrlab-build
```

For a CPU build, replace `--a100` with `--cpu-only`. Add `--nvtx-profile` for CUDA
profiler binaries. Add `--test-regex REGEX` to select a focused validation suite,
or `--no-tests` to compile only. `--no-tests` is a build result, not a validation
result. Use the same command without `--submit` inside an existing free compute
allocation. Builds and timings must run on compute nodes.

For a shell without initialized build Spack, provide
`--spack-root /absolute/path/to/spack/core` or set `IGRLAB_PACE_SPACK_ROOT`.
For a fresh site root, the driver queries that tool's configured builtin recipes
with `spack location --repo builtin`; it does not assume a recipe checkout layout.
An existing profile retains the repository recorded in its generated manifest
and queries it with `spack -e <environment> location --repo builtin`, isolated from
ambient Spack environment and configuration variables. A new second profile
uses the site's retained shared repository scope (or the existing profile if
that shared scope is absent). Conflicting, malformed, or missing retained
repository metadata fails before dependency preparation.
`IGRLAB_PACE_RECIPE_ROOT` remains an explicit override for fresh roots; an override
on an existing site must match its retained repository. Use a new `--site-root`
to select a different recipe authority without replacing the existing graph.
The tested build tool is Spack `1.3.0.dev0`; the dependency environment retains
its concrete lock between runs.

The worker loads the selected compiler module inside the compute allocation,
locates the shared Spack store from that compiler's resolved executable path,
and queries it with the selected build Spack. This avoids PACE's `spack/latest`
user store, which is separate from the shared store and uses an older database
reader. The site definition contains version, variant, module and target
constraints, with no installation prefixes or package hashes. The worker queries
`spack find --json` and `spack location -i` for each selected external, verifies
an unambiguous match, and writes the discovered paths into its local environment.
Application and MFEM configuration then query `spack -e ENV location -i PACKAGE`.
Missing or ambiguous site packages stop the build before compilation.

`--submit` prints job IDs and log paths. Compilation uses `cpu-small`, one task,
eight CPUs, 48 GiB and an eight-hour limit. When tests are enabled, a dependent
validation job uses two hours, four MPI slots with two CPUs each, and 48 GiB. GPU
validation requests one A100 on `gpu-a100`; CPU validation uses `cpu-small`.
Compilation therefore does not occupy a GPU. The validation stage reruns the
canonical driver with installed dependencies and existing build outputs.
MPI tests use the selected site's `mpiexec` directly from the batch shell.
Direct CTest executable tests linked to MFEM or MPI also use that launcher with
one rank. The CPU OpenMPI 4 stack failed singleton MPI initialization in the
observed Slurm batch runs; this launch rule also gives target tests a consistent
MPI process manager in the GPU profile. Existing tests with explicit rank counts
keep their commands.
This uses CMake's `TEST_LAUNCHER` target property (CMake 3.29 or newer); script
tests continue to control any subprocess launches themselves. The three VTK
restart/lifecycle script families receive a PACE-only opt-in to use the selected
MPI launcher for one rank; other configurations retain their direct launches.

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

The dependency setup follows Spack's documented environment model: retain
`spack.yaml` and `spack.lock`, use ordinary `spack concretize` rather than forcing
a fresh resolution on each build, and install the selected environment. See
[Spack environments](https://spack.readthedocs.io/en/latest/environments.html#concretizing).
Site packages use explicit external specs with `buildable: false`, so Spack
cannot silently replace site MPI with another build; see
[external packages](https://spack.readthedocs.io/en/latest/packages_yaml.html#external-packages).
Prefixes come from Spack's
[location command](https://spack.readthedocs.io/en/latest/command_index.html#spack-location).
The generated `site-packages.json` records discovery for each profile. A changed
site installation requires a new dependency root, preserving the existing lock
and binaries until their replacements have been validated.
