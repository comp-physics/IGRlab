# Tensor-oriented DG + IGR: prior evidence and reusable controls

Inspected 2026-09-05. This is a source/history audit, not a new performance
measurement. Repository origin was verified as
`https://github.com/comp-physics/dg_igr_miniapp.git`. The progress journal was
discovered through project file metadata and its origin verified as
`https://github.com/jarias9/igr_progress_journal.git`; no checkout path is
required by this document.

## Source identities

The initial checkout, `30afa08f8ad19a3ad37e627463ef63c40da1ae91` (PR61),
predated the completed tensor engine. The latest locally inspected candidate
is `f95ee67e4bfdc62ccfc50890400f8b843de53797`, tree
`2cc1d68c091d0127ce211caca1648717f959751c`, on the preserved
`codex/tensor-dg-igr-breakthrough` ref. Journal commit `4b99595` records PR67
as merged at `db4d2ae1af3f55fe87f3b0360804d4834169a691` with that same tree.
The merge object was absent from the local refs at initial inspection; its
publication status here comes from the journal, not a fresh remote query.

The inspected MFEM gitlink is
`d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec`. Historical local GPU work used
RTX 5080/SM120; the canonical build driver lowers dependency/application
architecture 120 and MFEM compatibility code `80-virtual`.

## Reported results and their limits

The quantitative rows below were read in journal briefs under
`tasks/briefs/`. They are previously reported measurements, not reruns. The
temporary raw-report location named by the latest brief was not found on the
present filesystem, so its artifact hashes and raw samples were not rechecked.

| Published result | Reported RTX 5080 measurement | What it establishes |
| --- | --- | --- |
| PR63, `c020f3b3627c36693df8deabaf79365b5e729adf`, tree `47a66ecf112bb67bcffb069f8760801359f80a33` | N52 IGR accepted-step ordinary/tensor ratios: p1 86.88, p2 37.62, p3 40.00. p3 trace solve: 61.94% of GPU kernel time. | Complete engine-package gain, including trace algorithm, fusion and local algebra; not an MMA-only gain, scientific validation, or physical-horizon comparison. |
| PR64, `a9c75040a8de69453fab1f0324d816041db65694`, tree `50ae1e2dca4fb0c7d9db35de3171c71de85c8f62` | p2/N8 tensor partitioned BE: 0.499859 s/unit physical time at dt=0.14; tensor monolithic BE: 1.098504 at dt=0.07. | Bounded cost-per-time/admissibility selection. Partitioned BE is a first-order lagged coupling and is not equivalent to the coupled DAE method. |
| PR65, `b8dc56b064e19a6ff4082863da88ac498fe70482`, tree `6261aada41926aa87bd3ad1cca55fb8aca1163d0` | Active IGR, p3/N12, tf=0.08, C_alpha=1: fresh-process medians adaptive RODAS3 3.1003 s; monolithic BE 6.9912 s; partitioned BE 8.7733 s. | Same requested-error threshold, with startup, rejected attempts and closure charged. RODAS3 error 2.26e-7 versus BE about 1.68e-4 means this is not a matched-realized-error curve. |
| PR67, candidate `f95ee67e4bfdc62ccfc50890400f8b843de53797` | Developed p1 cylinder to t=0.5: every-stage 52.145 s; extrapolated 40.193 s. | Whole-method ordinary-SIMT gain. Coupled extrapolation is second order despite SSPRK(10,4) transport; Sedov final sigma localization differs about 41%. |

The corresponding journal briefs are `HDG-TENSOR-CUDA-CLOSURE-V1.md`,
`HDG-TIME-TO-SOLUTION-V1.md`, `HDG-ADAPTIVE-DAE-TRACE-V1.md`, and
`DG-IGR-SHOCK-REGIME-V1.md`. PR65's refined cylinder failed to reach its
requested long horizon, collapsing near t=0.02; it is not a shock-regime
performance winner. Its inherited three-rank RODAS3 restart failure also
remains outside the qualified surface.

## Algebra, numerical methods and model parameters

The promoted tensor HDG engine retains full-energy Euler and instantaneous
scalar IGR. Its FP64 tensor volume contractions and local condensation change
algebraic execution; native matrix-free Schur actions and Schwarz field splits
change linear-solve implementation under independent FP64 residual checks.

Partitioned BE changes temporal coupling. Stage extrapolation/reduced closure
frequency changes the numerical method and needs its own error and stability
assessment. Endpoint closure certification alone does not establish accurate
within-step sigma or undo its effect on the Euler update. RODAS3 instead treats
Euler traces and IGR fields as algebraic variables in the singular-mass DAE,
reusing its common-shift factors across stages.

PR65 changed application coefficient input to
`alpha = C_alpha * h_eff^2`. Older direct-alpha controls therefore are not
automatically same-physics comparisons. PR67's selected DGSEM uses local h;
legacy DG/HDG comparison cards use global-max h. On nonuniform meshes these
are different regularization policies. No changed continuum model was adopted
by this audit.

Historical all-FP64 and p1-p3 qualification are bounds on those implementations,
not research restrictions. Deployment-GPU measurements, mixed precision,
higher-order and 3D probes remain open.

## Preserved lower-precision work

`codex/hdg-igr-schwarz-tf32-ir` at `16b3672` (equivalent cherry-pick
`e364fae82e8137a152a8fd1d7e6f2a4e6822837e`) contains an equilibrated TF32/FP32
Schwarz-factor experiment. Global Krylov vectors and gather are FP64; patch
triangular solves are scalar FP32; one FP64-defect correction and a first-use
defect gate are charged. `codex/hdg-igr-combine-tf32-matrix-free` also survives.
The latest candidate contains no TF32 path. No quantitative TF32 loss was
located in the inspected journal; code absence does not prove a negative
performance result. Dense inverse application, multi-RHS work, and new layouts
would be distinct experiments from those scalar triangular solves.

## Reusable measurement routes

- `tests/hdg/runtime/cuda/benchmark_generic_euler_accepted_step.cu` exposes
  engine, trace policy, integrator, tolerances and quadrature with source/build
  identities. One warmup and 5-20 accepted-step samples explain mechanisms but
  are not automatically full-process or matched-error evidence. Specify mesh
  size: its default N104 is expensive.
- `scripts/perf/hdg_tensor_campaign.py` provides alternating engine packages
  and cyclic ordinary-base/ordinary-common/tensor-common attribution. Its
  historical promotion workflow is optional for new research.
- `tests/hdg/runtime/cuda/test_generic_euler_convergence.cu` retains active-IGR
  physical-horizon/reference-error RODAS3/BE/PBE screens.
- `scripts/perf/profile_harness.py` provides process and NCU provenance;
  `scripts/perf/shock_architecture_matrix.py` plus `shock_qoi.py` provide
  horizon and QoI controls. The shock driver currently accepts only p1-p3.
- `tests/scripts/compare_vtk_pvd_fields.py` compares FP64 conservative and sigma
  fields on matching saved points/times. Its RMS is a point-sample norm, not a
  quadrature-weighted physical L2 norm. It intersects available arrays/times:
  callers must separately require the requested final horizon and all fields.
- The old disabled DGSEM performance CTest references
  `tests/scripts/dgsem_perf_harness.py`, which is absent on this candidate;
  use the existing generic profiling route instead.

Historical WSL Nsys captures lacked a usable GPU kernel timeline. NCU and
production-linked SASS were the tensor-instruction authorities. Recheck tools
on the new run rather than assuming that limitation persists.

## Proposed bounded full-application control (not run by this audit)

`dgsem_smooth_advection_periodic --active-igr-wave` supplies full-energy Euler
with nonconstant velocity and nonzero IGR forcing. The default Gaussian
advection case has constant velocity and is a weak test of active closure.
The active wave deliberately has no analytic solution-error callback: use
timestep-refined FP64 runs of the same spatial method as references.

Resolve `APP` from the canonical build output, set `P=3` or `P=4`, and give each
run a fresh `OUT`. This command selects the existing packed central-volume
route and forces an error if production execution cannot be selected:

```bash
OMP_NUM_THREADS=1 DG_IGR_DGSEM_PRODUCTION_STRICT=1 \
DG_IGR_DGSEM_TIMING=1 DG_IGR_IGR_PARTIAL_ASSEMBLY=1 \
DG_IGR_IGR_DEVICE_QF=1 DG_IGR_VTK_PREFIX_PATH="$OUT" \
"$APP" --device cuda \
  --mesh "$SOURCE/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n6.mesh" \
  --active-igr-wave --discretization dgsem --dg-backend dgsem --order "$P" \
  --serial-refine 0 --parallel-refine 0 --regularization igr \
  --c-alpha 1 --igr-alpha-length local \
  --dgsem-stab none --dgsem-volume-flux central \
  --dgsem-volume-dissipation none --dgsem-surface-flux rusanov \
  --time-integrator rk3ssp --time-step 0.001 --cfl-number 0 --t-final 0.01 \
  --igr-space L2 --igr-grad BrokenL2 --igr-solve-mode krylov \
  --igr-ip-kind sipg --igr-exact-dg-face --igr-sigma-cadence every-stage \
  --igr-stage-rtol-scale 1 --igr-linear-solver cg --igr-preconditioner chebyshev \
  --igr-lin-rtol 1e-10 --igr-lin-atol 0 --igr-lin-maxit 2000 \
  --igr-nonconvergence-policy error --igr-lin-print 0 \
  --no-igr-sigma-cfl --no-igr-sigma-rusanov --no-igr-invariant-domain-flux \
  --positivity-limiter none --nan-scan --admissibility-scan \
  --analysis-steps 0 --visualization-steps 1000000 \
  --vtk_save --time-step-save 0.01 --no-glvis --no-save-state \
  --no-vtk-h1-diagnostics --no-vtk-l2-diagnostics \
  --no-vtk-mpi-rank-field --no-vtk-zhang-shu-diagnostics
```

The central/no-volume-dissipation tuple is a smooth-method control, not the
PR67 LLF shock tuple. Repeat ordinary FP64 with dt=0.0005 and dt=0.00025;
measure reference self-difference before claiming the coarser dt meets an
error budget. Compare a numerical-kernel candidate at identical dt, mesh and
order to isolate arithmetic changes. A p3-versus-p4 solution-quality claim
also needs common physical sampling or an FE projection; raw VTK point arrays
have different lengths. The production metadata supports p1-p4 in 1D/2D;
the stored p4 restart test covers the host route and is not a p4 CUDA pass.

Conservative arrays `density`, `momentum`, `energy` and closure `sigma` are
saved as FP64. The restart manifest binds time/order/mesh; the existing cold
restart reconstructs the Gauss-Lobatto polynomial and recomputes sigma.
For same-order runs, the existing comparison command is:

```bash
python3 tests/scripts/compare_vtk_pvd_fields.py \
  --a-pvd "$REFERENCE_PVD" --b-pvd "$CANDIDATE_PVD" \
  --arrays density momentum energy sigma pressure \
  --output-json "$COMPARISON_JSON" --output-md "$COMPARISON_MD"
```

Require final saved time 0.01, nonzero finite sigma and successful closure
before interpreting timings. The dt values above are proposed bounded probes,
not measured stability or accuracy limits. Charge setup, closure and output
consistently across implementations.
