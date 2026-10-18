# Curved DGSEM cylinder with IGR and modal damping

This is a functional and physical-admission prototype on the repository's curved
cylinder mesh. It does not replace the large smooth-DG performance experiment
or establish HPC throughput. The mesh has 1,322 Q2 quadrilaterals, 47,592 p5
scalar closure DOFs and 190,368 conserved-state DOFs.

## Configuration and scope

Run `supersonic_cylinder_w_igr` on one DeltaAI GH200 with DGSEM p5, GLL Euler
and Sigma nodes, L2/BrokenL2 spaces, RK3 SSP, gamma=1.4, freestream velocity
(3,0), and normal startup with physical width 0.08. The original IGR baseline
source and exact SIPG interior faces remain active. Use central Euler volume
flux without extra volume dissipation, Rusanov surface flux, every-stage IGR,
and no positivity limiter, subcell blend, sigma-aware wave speed, or
invariant-domain flux. Disable H1 visualization spaces. These explicit choices
override stale example defaults.

The user selected max-length alpha scaling. The DOT study's portable modal
reference uses physical alpha=0.0004 and rates 0, 20000, 80000 (M16), and
320000. Matching that alpha on this mesh gives h_max=0.10246584078453408 and
C_alpha=0.038097964720559922. Rates 0 and 80000 are the initial cylinder
experiments; the study used a different one-dimensional p39 problem and does
not establish this cylinder parameter choice. Rates are inverse model time,
not timestep-dependent filter strengths. At dt=1e-5, nu*dt=0.8 for M16.
At p5 the profile leaves Q3 undamped, damps max-index-four modes at
187.4048, and max-index-five modes at 80000; this is not the same spectral
resolution as the original p39 study.

GL and GLL remain options for the same broken Qp scalar space. This prototype
selects GLL Sigma, and native logging confirms `mode=exact-nodal-gather`.
The IGR weak operator still uses its original Gauss quadrature: D1D=6, mass
Q1D=7, diffusion/RHS Q1D=6 for p5/Q2. GLL endpoint nodes do not remove the
DGSEM inverse-mass boundary correction or IGR SIPG face terms.

## Implementation

Source commit: `68136dd` (full source hashes in the c1 build receipt).

- Extend conservative modal damping to curved 2D polynomial quads with
  solution and geometry degrees at most eight. Existing affine 1D–3D paths
  retain their behavior, including the p39 reference.
- Set the cylinder's H1 construction/diagnostic defaults false and export
  native state at round-trip precision.
- Reuse existing DGSEM q_sigma telemetry when admissibility scanning is
  requested, so every-stage runs report aggregate physical-admission evidence.

For a curved cell, form the exact polynomial physical Gram matrix of tensor
Legendre modes, sort modes by increasing existing M16 rate, and factor
H=U^T U. The nodal decay matrix is V U^-1 diag(q) U V^-1. Ordering all zero-rate
modes first preserves that whole polynomial subspace; the operator also
preserves physical cell means and dissipates the physical mass quadratic
norm. Eigenrates remain between zero and nu. The fixed geometry operator is
cached per curved cell and shared across conservative components; runtime
application uses device arrays and CUDA kernels.

This is a physical-mass extension of the affine filter, not a positivity,
entropy, or raw-gradient guarantee. At p5/Q2, ordinary reference filtering can
already preserve the physical mean because the Jacobian polynomial is low
order; the new construction additionally establishes physical-norm
nonexpansion. Jacobian positivity is checked at several finite point sets,
not certified globally for arbitrary degree-eight maps. Dense curved caches
are limited to 256 MiB per rank. Host long-double setup is currently visible
in short-run wall time and has not been optimized for large curved meshes.

## Backends

| Variant | Current IGR volume | Euler contraction | RAS inverse |
|---|---|---|---|
| reference | Original FP64 partial assembly | FP64 SIMT fused | FP64 |
| tensor | FP64 WMMA | FP64 DMMA 8x8x4 fused | FP64 |
| tensor-fp32 | FP64 WMMA | FP64 DMMA 8x8x4 fused | FP32 SIMT |
| tensor-tf32 | FP64 WMMA | FP64 DMMA 8x8x4 fused | TF32 WMMA |

All variants retain the evolving original FP64 equation and CUDA FP64 FGMRES,
rtol=1e-10, atol=0, maximum 2000 iterations. Generic radius-one RAS reuses
inverses from the first operator only as a preconditioner. It admits curved
GLL cells and physical boundaries. Tensor-volume runs additionally certify
against the independent original current operator after each solve.

The periodic global spectral inverse and shared tensor multigrid are disabled:
their current geometry/topology restrictions exclude this mesh. The previous
3.28x large-DG result cannot be transferred to this cylinder. The TF32 RAS
kernel repeats one RHS across 16 columns, so its tensor execution alone does
not imply efficient GEMM utilization or an end-to-end speedup.

## Validation

Native build job 3402271 used the canonical driver on DeltaAI, CUDA SM90.
Binary SHA256: `8277cadb819c2aaa2821d46f56a0e6173569ec81bf6401611b0739089bdc2b9e`.
All four tests passed: modal damping CPU and CUDA, tensor volume CUDA, and
screened patch CUDA. Modal fixtures cover mixed affine/curved Q2 geometry,
GL/GLL nodes, both vector orderings, constants, the full undamped Q3 subspace,
physical mean, physical-norm dissipation, a rate bound, and the affine law.
Independent source review found no blocking issue.

The one-step admission job 3402297 establishes mesh geometry and max length;
its tiny bootstrap alpha is not used for physical comparisons. Job 3402317
runs the four ten-step pilots at physical alpha=0.0004 and dt=1e-6. They all
passed. Original and tensor FP64 final fields differ by at most 6.67e-16 in
normalized nodal Linf; all initial fields, meshes and headers match exactly.

All six 1,000-step runs (job 3402324) reached t=0.01 at dt=1e-5.

| Backend | Rate | Max normalized nodal RMS / Linf vs matched reference | Min rho / p at final saved nodes | Application wall (s) |
|---|---:|---:|---:|---:|
| reference | 0 | reference | 0.912489 / 0.492716 | 197.850 |
| tensor | 0 | 3.782e-15 / 6.795e-14 | 0.912489 / 0.492716 | 196.963 |
| reference | 80000 | reference | 0.889204 / 0.457792 | 210.419 |
| tensor | 80000 | 5.716e-15 / 6.461e-14 | 0.889204 / 0.457792 | 210.525 |
| tensor-fp32 | 80000 | 5.753e-15 / 7.461e-14 | 0.889204 / 0.457792 | 205.455 |
| tensor-tf32 | 80000 | 5.858e-15 / 6.104e-14 | 0.889204 / 0.457792 | 212.816 |

Every run accepted 1,000 steps with no rejected attempts, all 3,001 IGR
solves converged, and every recorded current-A relative residual was at most
5.746e-11 (tolerance 1e-10).
All sampled q_sigma values were positive: minima were 0.91490 undamped and
0.93086 damped. The maximum native relative boundary-conservation defect was
1.521e-13. There was no logged solver fallback. Sampled peak process GPU
memory was 5,592 MiB; this mesh does not fill a GH200.

The wall times are single diagnostic/output-enabled observations. They show
no established end-to-end advantage for this cylinder tensor path; TF32 RAS
was slower than the matched FP32 preconditioner in this observation. Do not
interpret the few-percent differences as an HPC speedup or a statistically
ranked performance result.

Nsight Compute jobs 3402327 and 3402507 captured actual tensor execution:

| Kernel | Executed instruction class | Tensor active / elapsed cycles |
|---|---|---:|
| IGR tensor volume | FP64 DMMA | 4.42% |
| p5 fused Euler | FP64 DMMA | 1.56% |
| TF32 RAS patch | HMMA/TF32 | 2.95% |

The corresponding DMMA instruction activity is nonzero for the two FP64
kernels, and HMMA instruction activity is nonzero for the TF32 kernel. These
are bounded single-launch hardware checks with kernel replay and cold cache,
not steady-state performance measurements. Their low utilization provides no
basis for claiming that the cylinder fills the GPU. The first two exact
profiler filters did not match NCU's typed template names; broad function-name
filters captured them successfully on the same binary.

The M16 rate did not improve every admission margin: at t=0.01 the FP64
reference's saved-node minimum pressure is 0.45779 with damping versus
0.49272 without, and minimum density is 0.88920 versus 0.91249. Both remain
positive over this tested window. The window t=0.01 is an early transient;
the example's existing t=4 default has not been validated. These observations
do not establish an optimal damping rate or long-time limiter-free stability. The prototype
therefore keeps damping explicit and optional; it does not promote 80000 to a
production default.

Saved-field comparisons are unweighted GLL-node RMS and Linf, normalized per
component by max(1, reference RMS). They are not physical L2 errors or an
analytic PDE error estimate. Physical conservation is assessed separately
using the application's boundary-flux balance. Native strict production
checks density/pressure at state and face panels. Offline acceptance also
requires finite positive q_sigma=gamma*p+(gamma-1)*Sigma at all flux locations
sampled by the aggregate telemetry. No finite sampling establishes positivity
everywhere inside cells. Every-stage
flux telemetry also does not independently scan q_sigma after the final Sigma
reconstruction; its final closure flag cannot be used as proof of that scan.

## Reproduction and evidence

Use the existing DeltaAI environment and canonical driver. On the recorded
remote checkout, pass the same `--output-root` as the original build:

```bash
build_output=/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/builds
./build.sh --gpu --cuda-arch 90 --iterate \
  --target supersonic_cylinder_w_igr \
  --target dg_igr_modal_damping_test --target dg_igr_tensor_volume_test \
  --target dg_igr_igr_screened_patch_cuda_test --output-root "$build_output" --jobs 4
./build.sh --gpu --cuda-arch 90 --iterate --tests-only --output-root "$build_output" \
  --test-regex '^(dg_igr_modal_damping_(cpu|cuda)|dg_igr_tensor_volume_cuda|dg_igr_igr_screened_patch_cuda)$' --jobs 1
```

The experiment ran sequentially on one GPU, account `bdiy-dtai-gh`, partition
`ghx4`, constraint `projects`, eight CPUs and 96 GiB requested host memory.
Every run records the binary/mesh/runner hashes, complete arguments/environment,
GPU identity, saved fields, application output, memory samples, and native
acceptance evidence. Timing here includes diagnostics and state output and
must not be treated as an HPC benchmark.

The durable local bundle is
`/home/jarias9/.cache/igrlab-experiments/cylinder-20261017`; the remote copy is
`/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/cylinder-prototype-20261017`.
The bundle contains `run-cylinder.py`, `compare-cylinder-state.py`,
`check-cylinder-selection.py`, all job scripts and source/binary receipts.
From an allocated DeltaAI GPU shell with the existing environment loaded:

```bash
experiment=/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/cylinder-prototype-20261017
python3 "$experiment/run-cylinder.py" \
  --binary "$experiment/binary-c1/supersonic_cylinder_w_igr" \
  --mesh /projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/source/examples/supersonic_example_w_igr/supersonic_cylinder.msh \
  --output "$experiment/new-unique-run-directory" \
  --variant tensor --steps 1000 --dt 1e-5 --rate 80000 \
  --c-alpha 0.038097964720559922 --fields --timeout 420
```

The driver refuses an existing output directory, clears inherited IGRlab
experiment selectors, and records all effective commands. Change only
`--variant` to `reference`, `tensor-fp32`, or `tensor-tf32` for matched arithmetic
comparisons. Change `--rate` to zero for the undamped control. Do not combine
results with different rates as an arithmetic-accuracy comparison.

For a manual launch, the essential backend selection is:

```bash
export IGRLAB_IGR_SIGMA_BASIS=lgl
export IGRLAB_IGR_KRYLOV_BACKEND=cuda-fp64
export IGRLAB_IGR_TENSOR_VOLUME=wmma
export IGRLAB_IGR_TENSOR_MG=off
export IGRLAB_IGR_PATCH_KIND=ras
export IGRLAB_IGR_PATCH_RADIUS=1
export IGRLAB_IGR_PATCH_PRECISION=fp64
export IGRLAB_IGR_PATCH_REBUILD_INTERVAL=0
export IGRLAB_DGSEM_PRODUCTION_STRICT=1
export IGRLAB_DGSEM_FUSED_RESIDUAL=1
export IGRLAB_DGSEM_FUSED_DMMA=1
export IGRLAB_DGSEM_SUBCELL_BLEND=0
export IGRLAB_SHARED_OPERATOR_PRECISION=off
```

Use the full command from a run record to retain the same physical settings
and diagnostic checks. These opt-ins do not change the example's global
backend or positivity-limiter defaults.

## Allocation usage

The six cylinder build, admission, run and profile jobs allocated one GH200
equivalent for a total of 1,726 seconds: **0.47944 GPU-hours/SUs**. This includes
build time on the GPU partition. Slurm allocation records are retained in the
local bundle. All IGRlab DeltaAI jobs on `bdiy-dtai-gh` since October 16 total
8.14806 GPU-hours at completion of these captures; this does not include older
Delta jobs. One DeltaAI SU corresponds to one reserved GH200-hour, per the
[NCSA accounting guide](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/job-accounting.html).
