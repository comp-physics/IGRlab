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

## Shared CUDA Krylov prototype

The shared IGR path has an opt-in, rank-one ordinary-CUDA FGMRES implementation.
The default remains MFEM. Select the implementation independently from the
physical method and preconditioner configuration:

| Environment variable | Default | Opt-in choice |
| --- | --- | --- |
| `IGRLAB_IGR_KRYLOV_BACKEND` | `mfem` | `cuda-fp64` (requires explicit FGMRES) |
| `IGRLAB_IGR_COARSE_CG_BACKEND` | follows the outer backend | `mfem` or `cuda-fp64` |
| `IGRLAB_IGR_COARSE_CG_EXECUTION` | `host-polled` | `cooperative` (admitted CUDA coarse solves) |
| `IGRLAB_IGR_TWO_LEVEL_BLOCK_PRECISION` | `fp64` | `fp32` (requires the owned CUDA FGMRES path) |

The legacy `DG_IGR_` spellings remain accepted. CPU and multiple-rank outer
solves use MFEM. The existing two-level preconditioner remains rank-one CUDA.
Its coarse CG retains the original `1e-2` preconditioned-norm tolerance and
100-iteration limit. Setting the coarse backend explicitly permits comparisons
of the outer implementation alone and the nested implementation alone.

The FP64 control caches workspace and keeps reductions, recurrence scalars,
ordered modified Gram–Schmidt, Givens rotations and the small backsolve on the
GPU. FGMRES uses the existing restart dimension of 50. There is still one host
convergence read per Krylov iteration, and called operators/preconditioners
may have their own synchronization. This is not fully device-controlled
execution. FGMRES retains MFEM's recurrence target
`max(atol, rtol*||b-A*x0||)`, including warm starts. Before accepting an iterate,
it also recomputes the original FP64 residual and requires
`||b-A*x|| <= max(atol, rtol*||b||)`. These are separate gates: a near-solution
warm start must not shrink the true-residual target below attainable FP64
cancellation. `GetInitialNorm()` reports the initial residual and
`GetFinalNorm()` reports the recomputed true residual. A failed certificate
restarts within the unchanged iteration budget. Compare iterations and
end-to-end timing: different reductions and final certification need not match
MFEM bit for bit or perform exactly the same work.

The FP32 option changes only the element inverse-block preconditioner action.
It uses IEEE FP32 FMA, retains FP64 state/coarse algebra/residuals, and retains
the FP64 inverse backup for unsafe-conversion/nonfinite/underflow row recovery.
It therefore adds an FP32 buffer and does not reduce total inverse storage.
Setup and teardown messages report storage and aggregate recovered rows. This
first prototype does not permanently escalate the entire preconditioner after
outer stagnation; reject a candidate that fails its original-operator residual
gate.

Build the focused algebra fixtures incrementally before reserving a GPU:
`--target dg_igr_device_krylov_test` and, with HDG runtime tests configured,
`--target dg_igr_igr_two_level_inverse_cuda_test`. The first fixture checks
restarted flexible solves against MFEM, repeated near-solution warm starts,
inconsistent-operator rejection, happy breakdown, zero RHS,
iteration limits, preconditioned CG norms and invalid curvature. The second
checks the two-level construction/current operator, IEEE FP32 action errors
and extreme-value recovery. Run it with
`IGRLAB_IGR_COARSE_CG_BACKEND=cuda-fp64` to exercise nested device CG. These
algebra fixtures do not establish application speedups or long-horizon accuracy.

See [original-architecture solver options](original-architecture-solvers.md)
for cooperative admission, HDG local-solve selectors, measured results and
validation scope. The application closure certificate is distinct from the
strict internal FGMRES convergence test described above.
