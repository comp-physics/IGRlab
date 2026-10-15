# IGRlab-only iteration

Use the same configured checkout and application build directory across edits.
`--iterate` goes straight to the existing CMake application build. It does not
run Spack, build or inspect MFEM, hash artifacts, install anything, generate
qualification receipts, or run tests automatically.

Build only the executable being measured and its IGRlab library dependencies:

```bash
./build.sh --site delta --a100 --iterate \
  --site-root /absolute/work/existing-dependencies \
  --output-root /absolute/work/existing-builds \
  --target dgsem_smooth_advection_periodic --jobs 8 --submit
```

That executable supports the full-application DG, DGSEM and HDG cases used in
our comparisons. Use another configured executable target when the experiment
needs it; repeat `--target` for multiple targets. Omitting `--target` builds all
configured IGRlab targets incrementally. On Delta, compilation still runs on
CPU compute. Omit `--submit` inside the CPU allocation.

The same command works on PACE with `--site pace` and that site's retained
dependency/build roots. Choose `--jobs` within the allocated CPU count; extra
cores help only when the changed application targets have independent compile
work. Changing polynomial order is a runtime argument and does not require a
rebuild. See the [bounded order comparison suite](gpu-studies.md#current-architecture-order-comparisons)
for consistent DG, DGSEM and HDG commands and DOF normalization.

The cached configuration supplies the compiler, dependency paths and build
options. Changing those options requires the normal configuration route.
CMake may regenerate its application build files when application CMake files
change; MFEM and other dependencies are imported, not rebuilt by this graph.
A missing build directory fails with an instruction to configure once; it
never falls back to dependency setup. Keep the checkout path unchanged because
it determines the output directory.

For an optional focused test, first build that test's target, then run it in
the appropriate owned compute allocation:

```bash
./build.sh --site delta --a100 --iterate --tests-only \
  --site-root /absolute/work/existing-dependencies \
  --output-root /absolute/work/existing-builds \
  --test-regex '^dg_igr_exact_face_cuda$'
```

Only that selection runs. There is no preliminary test suite, hash pass or
receipt wrapper. An empty selection or failed test fails the command.

For exploratory timings, run the selected binary from `app/build` with the
existing benchmark arguments and runtime environment, and compare baseline
and candidate on the same GPU. Do not route the edit/build loop through the
full study import/qualification machinery. Use a few unprofiled repetitions;
profile only when needed to answer a specific question.

An app-only rebuild removes the old whole-build qualification receipt before
compilation, including if compilation fails. It does not claim that unrelated
targets or tests match the edited source. Once a candidate is worth keeping,
use the normal build route with `--no-clean --no-deps` and the appropriate
checks to rebuild the remaining application targets and qualify it. Dependency
updates remain a separate, explicit operation.
