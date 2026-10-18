# Large Sedov DG transport optimization

The combined volume and face changes reduce full application time by 29.9%
(1.427x speedup) and sampled peak GPU memory by 20.3% on the fixed large
Sedov case. The largest final conserved-field relative L2 difference is
1.97e-9. Every original-current-FP64 residual certificate passes.

The campaign targets Euler volume-flux preparation and face kernels identified
by the completed [Nsight/IGR stopping investigation](igr_sedov_profile_and_stopping.md).
Two matched, unprofiled comparisons isolate the volume change and measure
the cumulative volume-plus-face result. This is one physical case and one
observation per variant, not a repeated performance distribution or an
analytic accuracy study. New paths remain opt-in with legacy fallback.

## Fixed comparison

Ordinary DG, p5 Gauss–Legendre, 512 by 512 Cartesian elements, 37,748,736
conserved coefficients and 9,437,184 IGR coefficients. Advance from t=0 to
0.1 with RK3 SSP, CFL=0.3, C_alpha=1/6 and max-length policy, giving physical
alpha=6.357828776041666e-7. Euler boundaries are slip walls; IGR uses L2 and
BrokenL2 with natural scalar boundaries. Preserve the existing Gaussian
initial condition and VTK output every 0.02.

Use RHS-relative CUDA FGMRES stopping at 1e-6, the reusable TF32 spectral
inverse, and FP32 Euler tensor-line contractions. Conserved state, nonlinear
fluxes, current IGR equation, and residual certification remain FP64.
Positivity limiting, modal damping, nonfinite scans, and optional
admissibility scans are off. The actual launcher explicitly uses
`--no-nan-scan --no-admissibility-scan`.

All builds/tests run through canonical `build.sh` on DeltaAI. GPU allocations
are serialized: one GPU, eight CPUs, 96G host memory, OMP_NUM_THREADS=1,
account `bdiy-dtai-gh`, partition `ghx4`, constraint `projects`. Each paired
benchmark uses one allocation so both variants run on the same GPU/node.
Small fixtures establish correctness only.

## Volume prototype

Commit `2eced69` adds two opt-in selectors:

| Selector | Original control | Candidate |
|---|---|---|
| `IGRLAB_WEAKDIV_SIGMA_STORAGE` | `dense` (default) | `auto` |
| `IGRLAB_SHARED_EULER_FLUX_EXECUTION` | `element` (default) | `pointwise` |

Auto storage gathers current scalar coefficients directly when same-mesh
scalar nodal L2 GL/GLL spaces have verified coincident nodes, a bijective
local-node map, and supported DOF transformations. It does not infer nodal
values from matching coordinates alone: MFEM IntegratedGLL is histopolant
and retains interpolation. Repeated compatible reference elements share a
CalcShape table; unsupported metadata retains the dense route. This fallback
is a compact full reference table, not staged 1D interpolation.

Pointwise preparation parallelizes the existing FP64 Euler/IGR nodal flux
algebra over element nodes. Existing centering, moment correction and
optional speed/telemetry semantics are retained. Legacy element execution
remains available for a matched comparison.

On the fixed case, source-derived storage removes 2.53125 GiB of per-element
sigma interpolation matrices; the scalar gather map is 36 MiB. These are
allocation counts, not measured peak-memory or bandwidth savings.

Native build/test job 3406571 passed `dg_igr_dg_volume_memory_cuda` in 2.62 s.
The executable SHA256 is
`4233286e004a16fc77bd77a21eb1bd252b4294c57c8b1a17d5b0063b6a62e26e`.
The test compares with independently assembled gradient/mass operators and
covers GL/GLL, node permutations, interpolation/basis fallbacks, current
sigma changes, space rebinding, and optional speeds/telemetry. An initial
failure in build 3406529 was a test-oracle error: an empty public telemetry
summary encodes its minima as NaN, while accumulation needs infinity.
The oracle was corrected without changing numerical tolerances.

Matched full-case job 3406572 ran original control then volume candidate on
node gh031, GPU UUID `GPU-1ff4241d-6dae-496f-f308-5126b246d6d9`.
Arguments, case, binary/build receipts and GPU identity match; the only
application environment changes are the two selectors and output paths.

| Metric | Original control | Volume candidate |
|---|---:|---:|
| Full application wall time | 1,263.613 s | 1,119.798 s |
| Accepted / rejected timesteps | 4,939 / 0 | 4,939 / 0 |
| Converged, FP64-certified solves | 14,818 / 14,818 | 14,818 / 14,818 |
| Total Krylov iterations | 51,745 | 51,741 |
| Sampled peak process GPU memory | 28,006 MiB | 25,412 MiB |
| Final FP64 RHS-relative residual | 3.597e-7 | 3.595e-7 |
| Full-process ns / conserved DOF / timestep | 6.778 | 6.006 |

The candidate uses 11.3813% less total time (1.12843x speedup) and 2,594 MiB
less sampled GPU memory. This is one paired observation, including startup
and VTK output, not a repeated benchmark distribution. The final pre-output
loop averages are 0.2325 and 0.2025 seconds per timestep. All existing
current-equation FP64 residual certificates pass; no residual replacements
or rejected timesteps occurred. A host IGR assembly timer absorbs waits for
previous GPU work: its apparent decrease must not be reported as an IGR
assembly-kernel optimization.

Both native final VTUs were transferred after timed runs ended and SHA256
verified. The GL6x6 comparison evaluates every saved Q5 cell polynomial at
9,437,184 Gauss points. Maximum conservative relative L2 difference is
1.966732942e-9; maximum sampled relative Linf is 8.483622918e-9. Derived
pressure and speed relative L2 differences are 5.271126294e-10 and
1.435378409e-9. These are differences from the matched control, not analytic
Sedov error estimates. The new control also agrees with the prior validated
RHS-relative baseline to max relative L2 1.965508357e-9.

## Face prototype

`IGRLAB_DG_TENSOR_FACE=fp64` requests the new path; default `off` retains
legacy behavior. Admission is bounded to CUDA FP64, rank-one conforming
affine GL VALUE quads with supported Rusanov physics, matching L2 GL sigma
when present, and the original GL face quadrature. Unsupported layouts,
quadrature, or policies fall back. The immediate validation target includes
slip walls and oriented/sheared periodic cells, not curved or distributed
meshes.

The path contracts normal tensor lines into compact traces, evaluates one
canonical FP64 numerical flux per face quadrature point, then gathers the
lift into each element coefficient without residual atomics. Reference
factorization and face orientation are checked against the original finite
element and quadrature. Exactly sparse tangential maps skip zero-weight
loads while retaining the actual nonzero FP64 coefficient. Other admitted
maps retain their full tangential table. Existing characteristic-speed,
telemetry, boundary-capture and intrinsic boundary-ghost rejection helpers
are reused.

For the fixed case, source-derived counts replace 3.375 GiB of Euler/sigma
face-shape tables with 240 MiB of traces, 96.1875 MiB of canonical fluxes, and
compact maps/metadata. About 906 million residual atomic additions per RHS
become 37.75 million coefficient writes. These counts do not establish a
kernel or application speedup. No new tensor-core arithmetic is introduced
by this memory restructuring; the existing TF32 inverse remains active.

Equivalence claims cover finite admitted states. Skipping exact-zero
products can change the spatial footprint of NaN/Inf propagation on invalid
input. No extra always-on nonfinite scans are introduced solely to reproduce
that footprint; intrinsic boundary rejection is preserved.

Independent source review found no remaining finite-state blocker.
Native build 3406821 passed the focused volume and face CUDA suites in
2.94 and 1.38 seconds. The face suite compares full RHS and conservation
against the original path, including boundary flux capture, sigma changes,
orientation and quadrature/space fallbacks. Native validation exposed and
resolved retained periodic boundary records: MFEM returns null boundary
transformations for joined interior seams, which are skipped like the
legacy route while complete element-side coverage remains mandatory.

The frozen executable SHA256 is
`b2828014dd1b464658836cbab22bf2ec3c718a44a02581edfbe342c27b3f5664`.
Its receipt retains 771 source-file hashes; the five changed production/test
files match implementation commit `00f1245`. Job 3406831 ran the original
control and combined volume-plus-face path sequentially on node gh003, GPU
UUID `GPU-efe3a061-7f23-ac1e-7fe2-53e5ec24bbf0`. Physical arguments, case,
binary/build receipts and GPU identity match. Environment differences are
only the three optimization selectors and output paths.

| Metric | Original control | Combined candidate |
|---|---:|---:|
| Full application wall time | 1,303.004 s | 913.302 s |
| Accepted / rejected timesteps | 4,939 / 0 | 4,939 / 0 |
| Converged, FP64-certified solves | 14,818 / 14,818 | 14,818 / 14,818 |
| Total Krylov iterations | 51,742 | 51,739 |
| Mean / maximum Krylov iterations | 3.492 / 6 | 3.492 / 6 |
| Sampled peak process GPU memory | 28,154 MiB | 22,444 MiB |
| Final FP64 RHS-relative residual | 3.606e-7 | 3.615e-7 |
| Full-process ns / conserved DOF / timestep | 6.989 | 4.899 |

The candidate uses 29.9080% less full-process time (1.42670x speedup) and
5,710 MiB less sampled peak GPU memory (20.2813%). The final pre-output loop
averages are 0.2388 and 0.1605 seconds per timestep, a 32.8% reduction.
All residual certificates pass, with no residual-replacement retries or
rejected steps. Linear-solve totals are 348.671 and 346.378 seconds; the gain
does not come from substantially fewer Krylov iterations.

This measures cumulative end-to-end improvement directly. It must not be
presented as an isolated face-only percentage: the volume-only pair used a
different allocation and executable. Retaining a fresh same-GPU control in
each pair avoids attributing cross-node differences to the optimization.

Both native final VTUs were transferred after timing ended and SHA256
verified. The same complete Q5/GL6x6 comparison gives:

| Conserved component | Relative L2 difference | Sampled relative Linf |
|---|---:|---:|
| Density | 6.847e-10 | 6.720e-9 |
| x momentum | 1.965e-9 | 8.572e-9 |
| y momentum | 1.963e-9 | 9.178e-9 |
| Energy | 5.382e-10 | 3.718e-9 |

All pass the engineering budgets. Derived pressure and speed relative L2
norms differ by 5.286e-10 and 1.437e-9. This establishes agreement of the final
saved fields on this case, not accuracy against an analytic Sedov solution
or a bound at all unsaved times. Final VTU SHA256 hashes are
`def7ed969a7b1db42556098c6f6415efc0cb2c110a4d0b811357ce4ee95246c5`
(control) and
`ac8b93dba31f0b9032d47decdf58b6c003cd7c14094fcd8e5e81bf69aad41c93`
(combined).

## Carry-forward choices

Keep the verified configuration opt-in:

```sh
export IGRLAB_WEAKDIV_SIGMA_STORAGE=auto
export IGRLAB_SHARED_EULER_FLUX_EXECUTION=pointwise
export IGRLAB_DG_TENSOR_FACE=fp64
```

These selectors complement the existing TF32 spectral inverse, FP32 Euler
volume tensor lines and RHS-relative stopping; they do not replace those
settings. Optional diagnostic scans remain off in the measured launchers.

The 29.9% improvement uses unchanged precision choices. The new face
contractions and nonlinear fluxes stay FP64. A 2-by-6 endpoint contraction
has little arithmetic relative to its data movement; forcing that individual
operation into a padded WMMA tile was not selected for this round. This is a
source-based screening decision, not a measured rejection of every possible
fused or batched tensor-core design. More aggressive staging or precision
changes require a new measured hypothesis.

No new Nsight capture was collected for these replacement kernels. The
original Systems/Compute results motivated the changes; the new evidence is
native correctness, source/allocation counts, and unprofiled whole-solver
timing. No new SM/DRAM-utilization or per-kernel speedup is claimed. Broader
geometry, MPI, other orders and DGSEM/HDG require separate qualification before broader
use. All campaign GPU jobs have completed; no additional run is queued.

## Retained evidence

The campaign directory is
`/home/jarias9/.cache/igrlab-experiments/dg-transport-20261018`.
It contains frozen source manifests and snapshots, build receipts, exact
case specs, launchers, and progress records. Remote artifacts are under
`/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/dg-transport-20261018`.
The original volume-pair spec SHA256 is
`ea1dc7a733b82037e224e4a80af9a23ddd9e41fdb84b5be92d8e0ef0932dab44`.

All four measured runs achieved target-time success and passed the
original-FP64 residual certificates. Each paired comparison uses unprofiled
application timing and final native fields at the same alpha. The GL6x6 Q5
comparison measures all saved conserved polynomials, with relative L2<=1e-3 and sampled Linf<=1e-2 as
engineering budgets. Observed differences and loop/full-process timing
are reported separately; the budgets are not measured errors.
