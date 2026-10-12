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
and optionally `--test-label REGEX` to require matching CTest labels as well.
Their intersection must select at least one test. Use
`--no-tests` to compile only. `--no-tests` is a build result, not a validation
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
eight CPUs, 48 GiB and a default eight-hour limit. When tests are enabled, a dependent
validation job defaults to two hours, four MPI slots with two CPUs each, and 48 GiB. GPU
validation requests one A100 on `gpu-a100`, or one V100 on `gpu-v100` for
`--gpu --cuda-arch 70`; CPU validation uses `cpu-small`.
Use `--build-time HH:MM:SS` and `--validation-time HH:MM:SS` to request shorter
allocations, up to those defaults. The submission plan and request receipt retain
the chosen limits. For a focused correctness selection, for example, add
`--validation-time 00:15:00`; choose sufficient time for the selected tests.
Compilation therefore does not occupy a GPU. The validation stage invokes
`--tests-only` through the canonical driver. It verifies the successful build's
source, configuration, locked dependencies, MFEM identity, test graph and runtime
artifact hashes before running tests, with no configure, build or install step.
It also requires unchanged retained site configuration; it does not concretize
or install dependencies. Missing or changed artifacts require a canonical build.
To validate an existing build again, add `--tests-only --submit` to the same
source, build options and output-root command; only a validation job is submitted.
Test selection and wall-time options can change without rebuilding.
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

For interactive CUDA debugging, first submit compilation of the CUDA profile
to CPU compute. Export the chosen paths so they remain available in the later
interactive shell:

```bash
export SOURCE_ROOT=/absolute/path/to/IGRlab
export SITE_ROOT=/absolute/scratch/igrlab-site
export SPACK_TOOL_ROOT=/absolute/path/to/spack/core
export OUTPUT_ROOT=/absolute/scratch/igrlab-build
export MFEM_ROOT=/absolute/scratch/igrlab-mfem-sm80

"$SOURCE_ROOT/build.sh" --site pace --a100 --hdg-runtime --submit --no-tests \
  --site-root "$SITE_ROOT" --spack-root "$SPACK_TOOL_ROOT" \
  --output-root "$OUTPUT_ROOT" --mfem-output-root "$MFEM_ROOT" --no-clean
```

After that build job completes successfully, request one A100, one rank,
eight CPUs and 48 GiB for a bounded interactive session:

```bash
srun --account=gts-sbryngelson3 --qos=embers --partition=gpu-a100 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --gres=gpu:a100:1 \
  --time=00:30:00 --mpi=none --pty bash -l
```

`--mpi=none` starts a shell without Slurm's MPI plugin; the shell is not an MPI
application. This avoids passing a shell-step PMIx context to later tests.
CTest launches MPI tests through the selected site's `mpiexec`, which supplies
their MPI process manager within the allocation. See
[Slurm's MPI option](https://slurm.schedmd.com/srun.html#OPT_mpi) and
[Open MPI's Slurm launch guidance](https://docs.open-mpi.org/en/v5.0.9/launching-apps/slurm.html).

Inside that allocated shell, run a focused selection through the saved build
receipt:

```bash
"$SOURCE_ROOT/build.sh" --site pace --a100 --hdg-runtime --tests-only --no-clean \
  --site-root "$SITE_ROOT" --spack-root "$SPACK_TOOL_ROOT" \
  --output-root "$OUTPUT_ROOT" --mfem-output-root "$MFEM_ROOT" \
  --test-regex '^igrlab_numerical_(conservation_cpu|conservation_cuda|backend_parity_cuda)$' \
  --test-label '^numerical-smoke$'
```

Keep the source, build flags and root paths identical to the completed build;
repeat any additional feature flags, such as `--nvtx-profile`. Compilation stays
in the CPU submission, while the GPU shell performs run-only validation. Choose
a longer allocation for larger convergence or HDG alignment selections. An
interactive `srun` step may expose `SLURM_STEP_GPUS` without `SLURM_JOB_GPUS`;
the driver accepts that assignment and retains Slurm's CUDA-visible mapping.

The one-task, eight-CPU shell can expose only one MPI launch slot. The CUDA
DGSEM checks `dg_igr_dgsem_operator_mpi_2rank_cuda` and
`dg_igr_dgsem_operator_split_mpi_4rank_cuda` require two and four ranks. If MPI
reports insufficient slots, apply the selected Open MPI 5 mapping and binding
settings only to this canonical correctness command:

```bash
PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe \
PRTE_MCA_hwloc_default_binding_policy=none \
"$SOURCE_ROOT/build.sh" --site pace --a100 --hdg-runtime --tests-only --no-clean \
  --site-root "$SITE_ROOT" --spack-root "$SPACK_TOOL_ROOT" \
  --output-root "$OUTPUT_ROOT" --mfem-output-root "$MFEM_ROOT" \
  --test-regex '^dg_igr_dgsem_operator_(mpi_2rank_cuda|split_mpi_4rank_cuda)$' </dev/null
```

Repeat any feature flags from the completed build, including `--nvtx-profile`
when enabled. Redirect standard input so MPI children cannot consume pasted
interactive shell input. Keep Slurm's allocation variables unchanged. These
settings let the existing MPI launchers run within the allocated eight CPUs and one GPU;
they request no extra resources and establish no timing result. See
[Open MPI 5 mapping parameters](https://docs.open-mpi.org/en/v5.0.9/mca.html#converting-mapping-parameters).
The regular `--submit` validation route requests four Slurm tasks.

V100 workflow debugging uses `--gpu --cuda-arch 70`, with a separate native
`cuda-sm70` environment, application outputs and MFEM cache. It uses the same
GCC 12.3 / CUDA 12.9.1 / CUDA-aware OpenMPI 5.0.9 site stack, discovered through
Spack. The SM70 profile has host-fixture coverage; native compilation and V100
execution have not yet been validated. Existing CPU/SM80 environment documents
remain unchanged. With
`SOURCE_ROOT` and `SPACK_TOOL_ROOT` set as above, choose fresh V100 roots and
submit compilation to CPU compute:

```bash
export V100_SITE_ROOT=/absolute/scratch/igrlab-site-v100
export V100_OUTPUT_ROOT=/absolute/scratch/igrlab-build-v100
export V100_MFEM_ROOT=/absolute/scratch/igrlab-mfem-sm70

"$SOURCE_ROOT/build.sh" --site pace --gpu --cuda-arch 70 --submit --no-tests \
  --site-root "$V100_SITE_ROOT" --spack-root "$SPACK_TOOL_ROOT" \
  --output-root "$V100_OUTPUT_ROOT" --mfem-output-root "$V100_MFEM_ROOT" --no-clean
```

After successful build receipt publication, enter a free V100 allocation with
the same shell-launch rule, then run the four DG/DGSEM numerical checks:

```bash
srun --account=gts-sbryngelson3 --qos=embers --partition=gpu-v100 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --gres=gpu:v100:1 \
  --time=00:30:00 --mpi=none --pty bash -l

"$SOURCE_ROOT/build.sh" --site pace --gpu --cuda-arch 70 --tests-only --no-clean \
  --site-root "$V100_SITE_ROOT" --spack-root "$SPACK_TOOL_ROOT" \
  --output-root "$V100_OUTPUT_ROOT" --mfem-output-root "$V100_MFEM_ROOT" \
  --test-regex '^igrlab_numerical_(conservation_cpu|conservation_cuda|backend_parity_cuda|dg_convergence_cpu)$'
```

This selection exercises conservation, spatial convergence and direct CPU/CUDA
agreement in production DG/DGSEM. SM70 does not enable HDG or its tensor route.
Release the allocation when finished. Final HDG/tensor acceptance and hardware
performance comparisons still require A100; the maintained `perf.sh` measurement
provider admits A100 only. V100 workflow results do not establish A100 speedups.

Profiler versions also limit V100 debugging. The observed Nsight Compute
2026.3 does not support Volta/V100; see NVIDIA's
[GPU support table](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html#gpu-support).
Nsight Systems 2025.1.3 predates the documented removal of Volta support in
2025.4; see the [release notes](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html#deprecated-features).
An older Nsight Compute with documented Volta support still needs its local
availability and an actual capture verified. This build path establishes no
V100 profiler readiness. Keep any native captures and trace processing on PACE
compute nodes.

Inspect a plan without submitting or changing output directories:

```bash
./build.sh --site pace --a100 --hdg-runtime --submit \
  --output-root /absolute/scratch/igrlab-build --print-plan
```

The default is incremental reuse. `--clean` explicitly rebuilds the selected
application and MFEM caches. An optional `--mfem-output-root /absolute/cache/path`
shares MFEM independently of application outputs. Its `source`, `build` and
`install` directories are protected by a separate writer lock; matching partial
fills resume, and completed stamped caches are admitted without rebuilding.
Changed source, compiler, dependency graph or MFEM policy rejects that cache
without deleting it. `--clean` affects only application outputs when this shared
cache is selected. The MFEM lock is released before application compilation.
Dependencies, Spack scopes and submission requests
live under the output root; application and MFEM artifacts retain the driver's
worktree/profile layout. `--site-root DIR` optionally shares a dependency root
across distinct output roots. A changed site manifest is rejected with an
actionable message; select a new dependency root rather than silently changing
an existing graph.

The checked-in [site definition](../cmake/sites/pace.json) selects GCC 12.3.0,
CMake 3.30.2, METIS 5.1.0, OpenBLAS 0.3.23 and HYPRE 3.1.0 in double precision.
CPU uses site OpenMPI 4.1.5 without CUDA. A100 uses site OpenMPI 5.0.9 with CUDA
12.9.1 and Umpire 2025.12.0. Every dependency uses the portable `x86_64_v3`
target, suitable for PACE's AMD A100 hosts. A100 dependency, MFEM and application
CUDA code generation all target native SM80. The site profile checks the concrete
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

Use the maintained [GPU study workflow](gpu-studies.md) to compare source
versions with one command, retained source/build caches, separate timing and
profiling stages, and summary-only retrieval.

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

The public submission route was verified on PACE on 2026-10-11 at build-driver
commit `30390d8`, with the working tree's DG/HDG numerical edits and pinned MFEM
`d9d6526` (`v4.9`). Both profiles ran with dependency installation enabled:
ordinary concretization retained the exact manifest and lock, and the final
checks admitted the installed MFEM interfaces and application library bindings.
CPU build job `13957931` completed in 2m34s; native-SM80 HDG/NVTX build job
`13957935` completed in 4m14s. Both used the free CPU queue. CPU validation job
`13957934` passed all three selected tests: `igrlab_mpi_test_launcher`,
`dg_igr_dg_operator_tests`, and `dg_igr_sigma_vtk_lifecycle_cpu`.
The A100 build result covers compilation and library binding checks. Subsequent
free-queue A100 job `13955960` passed all nine selected device correctness tests,
including DG/DGSEM operators, ordinary/tensor HDG sessions and p3 extensions,
p1/p3 CPU/CUDA numerical alignment, and the exact-face cylinder physical drift
comparison. CPU timing job `13955959` passed five selected correctness tests.
Both jobs completed their paired timing and separate physical-output runs on
compute nodes. A later profiler-report filename check failed in the A100 timing
harness after Nsight Compute successfully wrote a compressed `.ncu-repz` report;
this did not require an application or dependency rebuild.

For performance studies, use unprofiled solver or accepted-step elapsed time as
the primary result. Retain independent paired runs, warmup, useful-work counts,
iteration/retry counts, residual/error and physical-field gates, source and
binary identities, compiler/dependency versions, rank/thread placement, and GPU
identity. Define DOF or element updates explicitly when reporting throughput.

On allocated PACE compute nodes, use Nsight Systems to attribute kernel,
launch, synchronization and transfer costs, and inspect idle gaps in the native
timeline. Use bounded Nsight Compute captures of changed and expensive kernels
for theoretical/achieved occupancy, registers/shared-memory limits, grid waves,
compute/memory throughput, memory traffic/cache behavior, scheduler issue
activity, eligible warps, aggregate stall reasons and roofline inputs. Record
unavailable branch/divergence or direct spilling counters as unavailable; local
memory traffic alone does not establish register spilling. Interpret stalls
alongside issue activity and weight kernel findings by their contribution to
solver time. High occupancy alone is not a performance objective.

Keep native traces and all trace processing on PACE compute nodes. Retrieve
compact summaries and provenance only. Record profiler replay, cache and clock
policies separately from normal solver timing, and retain GPU clock/power/thermal
snapshots with their sampling limits. NVIDIA documents these distinctions in
the [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
and [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
CPU studies should also measure instructions per cycle, SIMD use, cache misses
and memory bandwidth on allocated CPU nodes. Multi-rank communication, load
imbalance and strong/weak scaling require separate workloads; the paired timing
campaign above used one rank.
