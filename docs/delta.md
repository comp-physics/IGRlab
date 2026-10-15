# Building on Delta

For routine optimization, use the [fast iteration workflow](iteration.md).
Keep one candidate checkout and build root, and use `--iterate --target NAME`
to rebuild only the application being measured. Dependency setup and full
qualification stay outside this loop.

Use the canonical `build.sh --site delta` front end. Its initial profiles are
CPU and native A100 (`cuda-sm80`), funded by `bdiy-delta-cpu` and
`bdiy-delta-gpu`. CUDA compilation runs on the `cpu` partition; validation
runs separately on `gpuA100x4` or in an already owned A100 allocation.

CPU resources compile CUDA software and dependencies. A successful build is
separate from GPU qualification and paired end-to-end timing/profiling. The
[bounded multiorder suite](gpu-studies.md#current-architecture-order-comparisons)
covers DG, DGSEM and both HDG engines at p1, p3, p5 and p7, with mesh sizes
chosen by conserved DOFs. Select a backend/order subset for an individual edit;
suite availability alone is not native numerical or performance qualification.

The site policy in `cmake/sites/delta.json` selects installed NCSA GCC, CMake,
OpenMPI/HPC-X, CUDA and OpenBLAS modules. Discovery records their actual
prefixes, versions and tool/metadata hashes. `OMPI_CC=gcc`, `OMPI_CXX=g++` and
`OMPI_FC=gfortran` select GNU compilers through the vendor MPI wrappers during
discovery. The driver then passes those verified compiler files by absolute
path so canonical MFEM admission can check the executables and versions. This
does not assert that the vendor MPI binary itself was built with GCC.
The same CUDA-aware system MPI serves both application profiles. CPU
HYPRE/MFEM/application settings still disable CUDA. Spack must reuse these
externals and builds the missing numerical dependencies in a private graph.
It never writes the NCSA installation or queries/upgrades its shared database.
Temporary Git clones use a private node-local directory under `SLURM_TMPDIR`
or `/tmp`. This avoids copying Lustre ACL/stripe attributes from read-only Git
pack files. Set `IGRLAB_DELTA_TMPDIR` to override the temporary base; dependency
stages, caches and installations still stay in the selected site root.

Install private Spack 1.2.2 and its pinned package repository under approved
project/work storage. Git/Python source staging can happen on the login node;
concretization, bootstrap and compilation happen on CPU compute. This setup
uses Spack core tag `v1.2.2` and `spack-packages` tag `v2026.06.0`. After
initializing the pinned MFEM submodule, submit a build-only job:

```bash
export IGRLAB_DELTA_RECIPE_ROOT=/absolute/work/tooling/spack-packages/repos/spack_repo/builtin
./build.sh --site delta --a100 --nvtx-profile --hdg-runtime \
  --spack-root /absolute/work/tooling/spack --site-root /absolute/work/dependencies \
  --output-root /absolute/work/builds --mfem-output-root /absolute/work/mfem-sm80 \
  --submit --no-tests --build-time 06:00:00 --jobs 8
```

Build and validation limits are configurable `HH:MM:SS`, defaulting to one
hour. The scheduler enforces its own partition limits. `--submit` prints the
actual job ID, immutable request path and log paths. Separate build roots can
share a dependency site root; profile locks serialize dependency installation.
Do not change retained environments in place to upgrade a toolchain. Choose
a fresh site root for a different dependency graph.

After a successful build, use identical source, options and paths with
`--tests-only --no-deps --no-clean`. Omit `--submit` to use an owned compute
allocation; include it to schedule separate validation. `--test-regex` may
select the required correctness tests. Tests-only checks source, binary and
concrete dependency identities and writes a genuine canonical result receipt;
it does not configure, install or compile. Validate each changed source and
configuration once before benchmarking it.

For direct rank-one GPU programs under an MPI-none interactive step, the Delta
runtime removes only `SLURM_STEP_ID` and `SLURM_STEPID` from descendants.
Open MPI 4 otherwise treats the inherited step as a direct PMI launch; without
that selector it uses a managed singleton within the real allocation. Job ID,
account, partition, node list, task count, assigned step GPUs and CUDA visibility
stay intact. The worker still verifies live ownership, resources, node, end time
and GPU before running applications. This does not change the parent owner or
fake a Slurm allocation. [Open MPI routing source](https://github.com/open-mpi/ompi/blob/v4.1.8/orte/mca/schizo/slurm/schizo_slurm.c)

The installed HPC-X `orted` shell wrapper retains the singleton report-pipe
writer while waiting for its daemon, so MPI initialization can wait indefinitely
for EOF. The runtime derives `OPAL_BINDIR` from the selected system MPI prefix
and requires executable `bin/.bin/orted` and `bin/.bin/ompi_info` before selecting
that native directory. MPI libraries, compiler wrappers and loader setup stay
unchanged. [Open MPI directory override](https://github.com/open-mpi/ompi/blob/v4.1.8/opal/mca/installdirs/env/opal_installdirs_env.c),
[singleton report-pipe handling](https://github.com/open-mpi/ompi/blob/v4.1.8/orte/mca/ess/singleton/ess_singleton_module.c)

Keep useful GPU allocations running through prepared sequential work. Nsight
Systems full-solver traces and closure-inclusive Nsight Compute captures stay
on Delta and are processed on compute. Request the documented
`--constraint=perf,nvperf` when hardware counters are required. Delta Slurm
currently makes that request node-exclusive even when one GPU is requested;
on `gpuA100x4` this reserves the whole node and charges 4 GPU SUs per hour.
Use ordinary one-GPU allocations for qualification and timings, and agree on
the counter allocation cost before requesting it. [NCSA accounting](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/job_accounting.html)
Retrieve compact
results, report cold process wall time separately from solver time, and compare
baseline/candidate builds on the same allocated GPU. PACE build/test results do
not qualify a new Delta binary or establish a cross-cluster speedup.

Delta policies and modules are documented in the
[NCSA user guide](https://docs.ncsa.illinois.edu/systems/delta/en/latest/).
