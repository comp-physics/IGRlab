# Numerical acceptance for optimization

These CTest groups exercise the production discretization and time integrators.
They supplement the workload-specific field checks in GPU studies. A faster
kernel alone does not establish preserved accuracy, conserved totals, or solver
time to solution.

Use the canonical `build.sh` driver. `--test-label` intersects
`--test-regex` when both are supplied; selecting no tests is an error. CPU builds
need the usual MPI/HYPRE-enabled FP64 MFEM profile. CUDA tests require the
canonical CUDA application/MFEM profile and an allocated GPU. HDG tests also
require `--hdg-runtime`. These tests do not require research tests or profiler
instrumentation. For PACE, use the portable CPU or native SM80 profile and the
free `embers` queue. Build on CPU compute and execute CUDA checks on the
allocated A100; compilation does not establish numerical acceptance.

For example, after a canonical PACE build, submit validation of the existing
build receipt with the same output, site and Spack roots:

```sh
./build.sh --site pace --cpu-only --tests-only --no-clean \
  --output-root "$OUTPUT_ROOT" --site-root "$SITE_ROOT" --spack-root "$SPACK_ROOT" \
  --submit --test-label '^numerical-smoke$'

./build.sh --site pace --a100 --hdg-runtime --tests-only --no-clean \
  --output-root "$OUTPUT_ROOT" --site-root "$SITE_ROOT" --spack-root "$SPACK_ROOT" \
  --submit --test-label '^numerical-smoke$'
```

The requested test configuration must agree with the saved build receipt. A
profile originally built without HDG must first be built canonically with HDG;
`--tests-only` does not add binaries. Use a separate intentional submission with
`--test-label '^numerical-convergence$'` for the larger ladders, with sufficient
validation wall time. The convergence group is excluded from the quick group.
To restrict a label to one family, also pass a name regex such as
`--test-regex '^igrlab_numerical_'`.

| Group | Tests | Purpose |
| --- | --- | --- |
| `numerical-smoke` | `igrlab_numerical_conservation_cpu` | DG and DGSEM p1/p3 periodic conserved totals at every committed step. |
| `numerical-smoke` | `igrlab_numerical_conservation_cuda`, `igrlab_numerical_backend_parity_cuda` | Same device evolution, plus direct production CPU/CUDA final-field and total agreement. |
| `numerical-smoke` | `dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_p{1,3}_rank1` | Existing CPU/ordinary-CUDA/tensor-CUDA full-field and trace trajectories, conventional and active IGR, numerical health, conservation audits and zero fallback. |
| `numerical-convergence` | `igrlab_numerical_dg_convergence_cpu` | New ordinary DG p1/p3 spatial ladder and finest-step temporal contamination check. |
| `numerical-convergence` | `dg_igr_dgsem_advection_tests` | Existing production DGSEM p1-p4 affine/curved spatial ladders and advection checks. |
| `numerical-convergence` | `hdg_generic_euler_cpu_convergence_{conventional,igr}_p{1,3}` | Existing HDG spatial and temporal convergence with independent exact error and timestep self-differences. |
| `numerical-convergence` | `dg_igr_hdg_runtime_cuda_generic_euler_convergence_{conventional,igr}_p{1,3}_rank1` | Same HDG convergence fixture across CPU, ordinary CUDA and tensor CUDA, with backend error/norm agreement. |
| `numerical-conservation` | `hdg_generic_euler_cpu_igr_quadrature_equivalence_p3` | Existing independent quadrature integrals of all Euler components over one-step and multi-step active-IGR trajectories, plus same-field and solver-health gates. |

All these tests also carry `numerical-acceptance`. `convergence`, `conservation`
and `backend-parity` identify the relevant numerical property; `cpu` and `cuda`
identify the required backends. The broad `conservation` label also selects
other pre-existing tests, so use the focused `numerical-*` labels when choosing
a bounded optimization gate. Historical HDG resource-lock names are mutual
exclusion identifiers, rather than an architectural restriction.

## Periodic evolution contract

The new fixture uses an affine periodic unit square, Gauss-Lobatto p1 and p3
spaces, and the existing smooth Euler contact wave:
`rho = 1 + 0.02*sin(2*pi*(x-0.35*t))*cos(2*pi*(y+0.2*t))`,
constant velocity `(0.35,-0.2)`, pressure `1`, gamma `1.4`. There is no source
or physical boundary flux. The reference is the translated exact solution,
rather than a solution produced by another numerical method.

The small conservation cases use 4x4 elements, final time `0.01` and exactly
20 equal steps of `0.0005` through production SSPRK(3,3). They require 20
completed steps and 60 RHS evaluations, finite coefficients and positive
density/pressure at independent quadrature points. A nonzero physical L2 state
change rejects a stopped or constant trajectory. Production DGSEM must report
strict supported CPU/CUDA execution, 60 production calls and zero fallback.

At the initial state and every committed step, the test independently integrates
mass, both momentum components and energy with Gauss quadrature of degree
`2*p+8`. This exactly integrates each conserved polynomial on the affine mesh.
Host `long double` accumulators reduce cancellation in this test-only oracle;
production arithmetic remains FP64. For each total, the fixed gate is
`abs(delta) <= 1e-12 + 2e-11*max(abs(initial),abs(current))`. The allowance covers
FP64 element/face summation and repeated RK arithmetic; it is substantially
above unit roundoff and tighter than the existing DGSEM `5e-10` conservation
gate. A total is checked against its own initial discrete integral, preventing
projection error from being mistaken for conservation loss.

The parity executable finishes and destroys all CPU meshes, operators and MFEM
vectors before selecting the process-global CUDA device exactly once. Only
plain host observations cross that boundary. Both members use the same mesh,
polynomial order, horizon and step count. Every final coefficient must agree
within `5e-11 + 2e-10*max(abs(cpu),abs(cuda))`; the independent physical L2
difference must be at most `2.5e-10`. Conserved totals additionally meet the
stricter conservation gate. This permits FP64 operation-order differences
without admitting a changed timestep or unsupported DGSEM fallback. Each
backend independently passes conservation and positivity before comparison.

## Convergence contract

The new ordinary-DG ladder uses p1 and p3 on 3x3, 6x6 and 12x12 periodic
meshes, with the same final time and exact contact-wave reference. The expected
spatial order is `p+1`. Both coarse/medium and medium/fine errors must decrease,
be finite and nonzero, and have observed order `log2(error_coarse/error_fine)`
greater than `p+0.25`. This reuses the pre-existing DGSEM contract's fixed
0.75-order allowance for the finite refinement ladder; it is not fitted to the
new run. Logs report initial projection errors, final conservative-vector L2
errors and both observed orders.

The maximum step is `0.05*h^2/(hypot(0.35,-0.2)+sqrt(1.4/0.98))`; the
number of equal steps is rounded up so the exact common horizon is reached.
For a third-order integrator, this policy reduces temporal error faster under
mesh refinement than the targeted p1/p3 spatial error. The finest mesh is
also evolved with twice the step count. Its physical L2 change relative to the
original finest trajectory must be no more than 10% of that trajectory's exact
error. This is a measured timestep-sensitivity check, not a rigorous estimate
of the complete temporal error or a separate temporal-order claim. Existing
HDG ladders retain their own independent temporal self-convergence contracts.

Thresholds and workload sizes are fixed before compute validation. Any failure
should be investigated with its printed errors, drift and work counts; passing
is not established until the selected CPU/A100 tests actually execute on the
tested source and dependency configuration. This fixture covers smooth,
source-free affine evolution. Existing broader curved-geometry, boundary,
limiter, active-IGR and MPI tests remain necessary when a change touches those
features.
