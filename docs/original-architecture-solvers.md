# Original-architecture CUDA solver options

These options accelerate existing DG/DGSEM and HDG solves without changing
their numerical defaults. The measured candidates use FP64. IEEE FP32 options
are retained as default-off comparison experiments; they do not establish a
Tensor Core benefit. The later fused-orthogonalization and TF32 local-solve
experiments are not included here.

## Selecting the FP64 candidates

Set selectors before launching the application, alongside the existing
method, IGR policy, mesh and time-integration arguments. The `DG_IGR_` legacy
environment spellings remain accepted.

| Environment variable | Default | FP64 candidate |
| --- | --- | --- |
| `IGRLAB_IGR_KRYLOV_BACKEND` | `mfem` | `cuda-fp64` |
| `IGRLAB_IGR_COARSE_CG_BACKEND` | follows outer backend | `cuda-fp64` |
| `IGRLAB_IGR_COARSE_CG_EXECUTION` | `host-polled` | `cooperative` |
| `IGRLAB_IGR_TWO_LEVEL_BLOCK_PRECISION` | `fp64` | `fp64` |
| `IGRLAB_HDG_LOCAL_SOLVE` | `current` | `fp64-refine` |
| `IGRLAB_HDG_LOCAL_SOLVE_BACKEND` | `simt` | `cublas-batched` |

The first four selectors apply to shared DG/DGSEM IGR solves; the last two
apply to CUDA HDG local solves. Changing these implementation selectors does
not select a different physical method. See [iteration](iteration.md) for
the canonical build/test route and [GPU studies](gpu-studies.md) for the
application's existing comparison commands.

### DG and DGSEM

The device outer path requires explicit FGMRES and rank-one CUDA execution.
CPU and multiple-rank outer solves use MFEM. Ordered FP64 modified
Gram–Schmidt, reductions, recurrence scalars, Givens rotations and the small
backsolve execute on the GPU; one host convergence read remains per outer
iteration. The existing restart dimension is 50. Coarse CG keeps the current
CSR/Jacobi action, preconditioned-norm relative tolerance `1e-2` and limit of
100 iterations. Cooperative execution requires supported CUDA launch
resources and the admitted operator/preconditioner/initialization policy;
otherwise coarse CG falls back to host-polled execution.

The outer solver keeps separate recurrence and original-operator residual
tests: `max(atol, rtol*||b-A*x0||)` and `max(atol, rtol*||b||)`, respectively.
A failed true-residual test restarts within the requested iteration budget.
The application's existing final closure certificate is unchanged: at the
studied `rtol=1e-10`, it uses `5e-10` and can certify a capped internal iterate.
The application certificate and strict internal solver convergence are
different decisions.

`IGRLAB_IGR_TWO_LEVEL_BLOCK_PRECISION=fp32` changes only the element
inverse-block action to IEEE FP32 FMA. It retains FP64 state, coarse algebra,
residuals and inverse backups for unsafe-conversion/nonfinite/underflow row
recovery. It adds storage, and does not permanently escalate the whole
preconditioner after outer stagnation. No incremental speed or memory benefit
is established for this option.

### HDG

Refined local solves retain the original assembled FP64 matrices and use
FP64 residuals, Schur construction and recovery. The local componentwise
backward-error limit is `1e-13`; the original coupled-residual default limit
is separately `1e-10`. Up to four corrections are allowed before fresh FP64
factorization recovery. Separate global RHS/action assembly preserves small
shared-face contributions under cancellation near convergence.

The cuBLAS backend uses batched LU/solve with pedantic math. Application
qualification below covers rank-one, borrowed monolithic backward Euler.
Refined modes reject unsupported configurations, including multiple ranks,
non-borrowed execution and RODAS. These results do not establish partitioned
backward-Euler performance. `fp32-refine` is an optional IEEE FP32 local
factor/solve mode with the same original-FP64 gates; both it and the FP64
backend keep existing arenas. The separately selectable HDG volume arithmetic
was pinned to original FP64 for these comparisons. cuBLAS selection alone is
not evidence of TF32 Tensor Core execution.

## Conserved output synchronization

After a device state update, saved conserved fields could reuse stale host
views. The common output boundary now refreshes the owning conserved vector
before reading those views. The regression first reads host views, performs
two distinct device updates and checks all five conserved components, without
a derived-field projection masking the synchronization. Derived-field
residency is checked separately. This is a fixed-owner regression, not a
general owner-resize or interactive visualization test.

## Measured results and validation

The 2026-10-15 retained `combined-v5` qualification used DG `c8e42b5`, HDG
`0995647` and output fix `ff36c84`, built by the canonical driver in Delta job
22930220. Integration commit `4ce97bb` has byte-identical contents for all
21 changed compiled/test/build-registration paths. Documentation changes
after that commit do not change the tested implementation.

The comparison used one A100-SXM4-40GB, one MPI rank, eight allocated CPU cores
and one OpenMP/BLAS thread. Each number is the median of three alternating
unprofiled process-time ratios within the same allocation. Controls and
candidates used explicit selectors in the same corrected executable. Process
time includes startup, setup and shutdown; warmups, field runs and profiles
are excluded. Ranges are observed samples, not confidence intervals.

Smooth cases used active IGR with `c_alpha=1` and `dt=1e-5`. DG/DGSEM used
RK3SSP with local alpha; HDG used monolithic backward Euler with minimum-length
alpha. Ordinary HDG retained refreshed ILU0/Jacobi 5+5 and Hypre GMRES. These
are same-method implementation comparisons, not cross-method rankings or an
hp-convergence study.

| Case | FP64 candidate | Median process-time reduction | Observed pair range |
| --- | --- | ---: | ---: |
| DG p1, 128×128, 10 steps | Device FGMRES + cooperative coarse CG | 1.11% | −0.81–1.45% |
| DG p5, 32×32, 100 steps | Device FGMRES + cooperative coarse CG | 55.30% | 54.84–55.34% |
| DGSEM p5, 32×32, 100 steps | Device FGMRES + cooperative coarse CG | 59.65% | 58.88–59.86% |
| Ordinary HDG p5, 32×32, 10 steps | cuBLAS FP64 refinement | 6.80% | 4.87–8.70% |
| Ordinary HDG p7, 32×32, 10 steps | cuBLAS FP64 refinement | 14.50% | 14.31–14.60% |
| Ordinary HDG p5, 32×32, 50 steps | cuBLAS FP64 refinement | 8.88% | −1.17–10.40% |

DG p1 is inconclusive. The longer ordinary-HDG p5 case includes a slower
candidate pair. Tensor-HDG p5 regressed by a median 18.88% with FP64 cuBLAS;
FP32 cuBLAS regressed by 5.96% for ordinary HDG and 45.48% for tensor HDG.
These results support retaining opt-in controls. They do not support a memory
reduction: separate 100-ms samples at ordinary-HDG p5 observed 7393 MiB for
current versus 7639 MiB for either vendor mode, one run each; brief peaks can
be missed.

The retained audit checked 14 summaries covering 239 application runs:
138 paired timing runs, 46 warmups, 52 field runs and three memory runs.
Initial/final density, momentum, energy, pressure, velocity and sigma passed
componentwise `5e-9 + 5e-8*abs(control)` comparisons, matching geometry/shapes,
finite-value and sampled density/pressure positivity checks, and evolving
conserved-state checks. Final sigma was nonzero; HDG initial sigma may be zero.
DG/DGSEM maximum final closure residual was `1.942e-12` against `5e-10`;
smooth/shock conservation maxima were `6.151e-14`/`6.242e-14` against `5e-10`.
The maximum reported HDG original-coupled residual was `3.170e-14`; its
configured limit remained `1e-10`.

Six DG/DGSEM shock runs also passed: MFEM, cooperative FP64 and cooperative
FP32-block controls at p3, 32×1 quads, Mach 2, width 1.25, 100 steps at
`dt=2e-4` to `t=0.02`. These are correctness checks, not shock speedup evidence.
Qualified cooperative runs reported no runtime fallback.

Focused native CUDA tests passed in the retained progression: device Krylov
and two-level inverse (22929472), repaired HDG local solves (22929706), output
alias and diagnostics (22929990), and updated cooperative Krylov/vendor HDG
fixtures on final v5 (22930221). This is not a full test-suite pass. The study
does not establish analytic-solution accuracy, arbitrary hp support, other GPU
architectures, accelerated multi-rank operation or long physical-time accuracy.

### Integration revalidation

The integrated source at `4ce97bb` was rebuilt successfully through the
canonical driver in job 22931989. Job 22931991 passed six focused tests:
`dg_igr_device_krylov_cuda`, `dg_igr_hdg_runtime_cuda_refined_local_solve`,
`dg_igr_igr_two_level_inverse_cuda`, `dg_igr_analysis_device_alias_cuda`,
`dg_igr_diagnostics_cuda` and `dg_igr_diagnostics_tests`. The two-level inverse
also passed a separate cooperative-coarse run. Logs confirmed cooperative
admission and the cuBLAS shared-trace cancellation fixtures.

Job 22932011 passed six fresh A100 application field runs at p5, 32×32,
10 steps: MFEM/cooperative for DG and DGSEM, and current/FP64-cuBLAS for
ordinary HDG. Initial/final comparisons passed the unchanged tolerances above,
including finite evolving conserved fields, sampled density/pressure
positivity and native solver/work checks. DG/DGSEM also passed the native
`5e-10` conservation gate. These checks validate the integrated implementation;
they add no new timing samples to the retained performance results.
