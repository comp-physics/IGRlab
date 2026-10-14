# Building on Delta

Use the canonical `build.sh --site delta` front end. Its initial profiles are
CPU and native A100 (`cuda-sm80`), funded by `bdiy-delta-cpu` and
`bdiy-delta-gpu`. CUDA compilation runs on the `cpu` partition; validation
runs separately on `gpuA100x4` or in an already owned A100 allocation.

The current optimization work targets representative order 5. CPU resources
are used to compile CUDA software and dependencies; broad CPU/order sweeps are
outside this work. A successful build is separate from GPU qualification and
paired end-to-end timing/profiling.

The site policy in `cmake/sites/delta.json` selects installed NCSA GCC, CMake,
OpenMPI/HPC-X, CUDA and OpenBLAS modules. Discovery records their actual
prefixes, versions and tool/metadata hashes. `OMPI_CC=gcc`, `OMPI_CXX=g++` and
`OMPI_FC=gfortran` select GNU compilers through the vendor MPI wrappers; this
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

Keep useful GPU allocations running through prepared sequential work. Nsight
Systems full-solver traces and closure-inclusive Nsight Compute captures stay
on Delta and are processed on compute. Request the documented
`--constraint=perf,nvperf` when hardware counters are required. Retrieve compact
results, report cold process wall time separately from solver time, and compare
baseline/candidate builds on the same allocated GPU. PACE build/test results do
not qualify a new Delta binary or establish a cross-cluster speedup.

Delta policies and modules are documented in the
[NCSA user guide](https://docs.ncsa.illinois.edu/systems/delta/en/latest/).
