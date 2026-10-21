# IGR coefficient preparation: matched large Sedov result

The coefficient-preparation change reduces full application time from
**897.558118 s to 848.447222 s**, saving **49.110896 s / 5.47161%**
(**1.057883x speedup**) in a sequential pair on the same DeltaAI GPU.
Sampled peak process GPU residency falls from **22,444 to 21,286 MiB**,
a reduction of **1,158 MiB / 5.15951%**. Both runs complete 4,939 steps,
14,818 certified linear solves and 51,739 outer Krylov iterations.

This is an **incremental comparison against the validated TF32 pipeline**.
It does not compare against the original FP64 GPU implementation. The pair
measures the combined removal of unused diagnostic gradients and reuse of
matching state interpolation; it does not isolate their individual savings.
The final native VTK field payloads are byte-identical. Fresh Nsight Systems
confirms the redundant interpolation work is removed.

## Matched case and measurement scope

Job **3414609** ran the frozen baseline followed by the candidate on **gh030**,
GPU `GPU-979d161d-25e9-ed7c-1429-f6708bf4251b` (NVIDIA GH200). Resources were
one GPU, one MPI rank, eight CPUs, 96G host memory and `OMP_NUM_THREADS=1`,
using `bdiy-dtai-gh`, partition `ghx4`, constraint `projects`.

Both runs use ordinary DG on a fixed **512 by 512 Cartesian mesh**, p5
Gauss–Legendre, with **37,748,736 conserved coefficients** and **9,437,184
scalar IGR coefficients**. Preserve L2/BrokenL2 IGR, RK3 SSP, every-stage
sigma updates, final time 0.1, CFL 0.3, `C_alpha=1/6` with max element length,
and physical alpha `6.357828776041666e-7`. CUDA FP64 FGMRES uses RHS-relative
rtol `1e-6`, atol 0 and maxit 1000. Positivity limiting and modal damping are
off. Both launchers explicitly use `--no-nan-scan --no-admissibility-scan`.
VTK output is retained at physical-time cadence 0.02.

| Metric | Frozen TF32 baseline | Coefficient-prep candidate |
|---|---:|---:|
| Full application wall time, s | 897.558118 | 848.447222 |
| Sampled peak process GPU memory, MiB | 22,444 | 21,286 |
| Accepted / rejected steps | 4,939 / 0 | 4,939 / 0 |
| Converged / certified linear solves | 14,818 / 14,818 | 14,818 / 14,818 |
| Outer Krylov iterations | 51,739 | 51,739 |
| Maximum observed independent relative RHS residual | 9.9963081e-7 | 9.9963081e-7 |
| Final committed-state relative RHS residual | 3.615353934207918e-7 | 3.615353934207918e-7 |
| Full-process ns per conserved coefficient per step | 4.814167 | 4.550754 |

Each timing is **one unprofiled full run**, including startup, fresh setup,
VTK and finalization. There is no repeat-distribution or confidence-interval
claim. Memory is sampled once per second and can miss shorter peaks. The
last row divides full process time by 37,748,736 conserved coefficients and
4,939 steps; it includes all IGR and output work and is not an isolated
operator throughput. Native component timers are host intervals that may
absorb pending GPU work and are not independent GPU kernel durations.

## Implementation and controls

`grad_u_qf_` is a diffusion-quadrature gradient consumed only by optional
finite diagnostics and ingestion audits. It is now allocated and refreshed
only when ingestion auditing is enabled or an explicitly enabled finite scan
has not yet completed. The mandatory RHS gradient remains computed every
update. Rebuilding quadrature resets the diagnostic gradient buffer without
resetting the lifetime-once scan flag. Enabling scans or ingestion audits
later still produces current-state gradients before those diagnostics read
them.

State values are interpolated once per distinct compatible quadrature layout
within an update. Cache setup proves identical mesh, element coverage,
logical offsets, geometry, and ordered reference point/weight bytes; every
projection uses the same restricted current state and byVDIM layout. Each
geometry's rule is checked once. Different layouts retain separate
interpolations. The proof is cached, while the values themselves are always
computed from the current update.

Production coefficient reads, ingestion audits and finite scans select the
same direct `const QuadratureFunction&` owner. There are no MFEM memory
aliases or stale host/device validity flags to synchronize. Unused value
wrappers retain host allocation but receive no interpolation or device read.
The optimization changes neither coefficient formulas nor FP64
current-equation residual certification.

`DG_IGR_IGR_SHARE_STATE_INTERPOLATION` defaults to true and is sampled when the
quadrature cache is built. The equivalent `IGRLAB_IGR_SHARE_STATE_INTERPOLATION`
alias is supported through the existing environment helper. The pair sets
both aliases to 0 for the frozen baseline and 1 for the candidate. The frozen
baseline predates this selector and keeps its original preparation behavior;
setting the new selector to 0 in the candidate disables sharing but does not
restore unconditional diagnostic-gradient computation.

The existing precision arrangement is unchanged: FP64 current IGR operator,
coefficients, Krylov work and certification; TF32 spectral-inverse cuBLAS
GEMMs; FP32 shared Euler tensor-line operators; FP64 Euler face operations.
Both logs record 51,739 spectral inverse applications and 206,956 GEMMs.
This change does not introduce additional Tensor Core operations.

## Native validation

The canonical DeltaAI build and focused checks passed in job **3414571**:

- `dg_igr_igr_diagnostic_scans` (2.59 s in the detailed test log).
- `dg_igr_igr_diagnostic_scans_cuda` (2.42 s in the detailed test log).

The extended tests exercise sharing off/on, evolving current states,
matching and nonmatching quadrature, same-solver cache rebuilds and late
ingestion auditing. Shared and unshared current RHS and current operator
probe actions match bitwise in the fixtures. Diagnostic diffusion-gradient
extrema are checked against current nonconstant-state gradients, including
late enablement. The existing opt-in scan, actual current-A residual and
CUDA FP64 FGMRES checks remain active.

Both full Sedov runs have complete independent certificate coverage: 14,818
records, no missing solve IDs, all observed relative RHS residuals below
`1e-6`, no residual-replacement retries and no audit warnings. Equal iteration
counts and residual summaries alone do not establish final-field identity.

## Final saved fields

Both complete native VTU files at **t=0.1**, physical alpha
`6.357828776041666e-7`, were hashed remotely and verified after local transfer
against the completed-run collection receipts. Both are **1,160,775,042 bytes**
with SHA-256
`ac8b93dba31f0b9032d47decdf58b6c003cd7c14094fcd8e5e81bf69aad41c93`.

The complete saved payloads, including Float64 density, energy, momentum,
sigma, velocity, pressure and geometry, are **byte-identical**. A separate
numerical norm calculation is unnecessary for identical payloads. This
establishes final saved-output parity; it does not establish analytic PDE
accuracy or equality of every unsaved intermediate state. The evidence is
`comparison-3414609/field-comparison.json` and each variant's
`final-vtk/collection-receipt.json`.

## Fresh Nsight Systems attribution

The fresh Systems capture, job **3414629** on **gh029**, completed normally
through all 4,939 steps and physical time 0.1. All **14,818 coefficient-update
ranges** contain exactly **one large state `Values2D` kernel and one large
state `Derivatives2D` kernel**. There are no selected large-state kernels
outside those ranges or with missing correlations.

| Observed large-state interpolation work | Prior TF32 capture | Candidate capture |
|---|---:|---:|
| `Values2D` calls per coefficient update | 3 | 1 |
| `Derivatives2D` calls per coefficient update | 2 | 1 |
| Total `Values2D` calls | 44,454 | 14,818 |
| Total `Derivatives2D` calls | 29,636 | 14,818 |

This removes **44,454 large interpolation/derivative kernel launches** across
the full run: 29,636 value interpolations and 14,818 derivative interpolations.
Total captured CUDA kernel counts decrease by the same 44,454, from
2,591,996 to 2,547,542; memcpy and memset counts are unchanged. One-time
geometry interpolation configurations are classified separately.
The candidate's summed large-state kernel durations are **9.852006541 s** for
values and **21.280673044 s** for physical derivatives.

The prior capture is job **3407651** on **gh009**, using the same frozen
baseline binary as the timing comparison. Different profile nodes make this
a comparison of observed counts and removed work, **not a matched speedup
measurement from kernel durations**. The 5.47161% full-time reduction comes
only from the unprofiled same-GPU pair on gh030.

The remaining FP64 current-IGR face family consumes **183.537941 s**, or
**33.9983% of 539.844739 s summed GPU kernel time**, in the candidate capture.
That percentage is not a wall-time fraction or a hardware utilization metric.
No new Nsight Compute capture or Tensor Core activity-counter claim is made
for this coefficient-preparation change.

## Source and evidence identities

The candidate branch is `igr-coefficient-prep-opt`, based on validated
`e57119b158145d658b31b77b73c0ced9053d1b29`. The tested source is
`a6cfdc1d3b3c26a0eaeb3a5412b51cb05796f022`. Its production diff contains only
`include/physics/igr/IGR.hpp` and `src/physics/igr/IGR_device_coefficients.cpp`,
plus focused tests in `tests/test_igr.cpp`. All 771 tracked source-file hashes
were checked against the native build receipt after collecting the results;
there was no source drift before this report-only addition.

| Receipt | Identity |
|---|---|
| Frozen baseline executable source | `c13928be284a511a1d170a2b9a5b3cd27cc0fac0` |
| Baseline executable SHA-256 | `c6f91d544b831499ebe96dbcf35306aa6e145d83e1c6333c46f8ec115a145d42` |
| Candidate executable SHA-256 | `2f858af42e59acdf66deca72e6c7ebcee309b57eb5b641b3ed410cfdfc759dd6` |
| Candidate build receipt SHA-256 | `f5c13334e0fc7dfa3a5122159e7964a84572e72b9805b6217ce7fdaa6b26f111` |
| Matched case specification SHA-256 | `4bb1c4df2e6beb3050ab953f5ee654cd8b58e14c619ccb2a27954d4513b60832` |
| Comparison runner SHA-256 | `f145403070ddbcd681a08e4321c46d55b05b2552f5f922ec08186d46ec025ddc` |
| Detailed native test log SHA-256 | `4b3d14107d291e0c5574f7c317b5ca19a89be596ed660034bd95daaee13cd8c4` |
| Fresh native Systems report SHA-256 | `f4c058fb15b852b547b26752e4b9ef014e96e065e170be843a4b857409a62878` |
| Fresh Systems SQLite SHA-256 | `ceeda7f45da4fd0f040096a91421c81cd07dec83e7f0cc1b520ab830ab768d9f` |

Evidence root:
`/home/jarias9/.cache/igrlab-experiments/sedov-coefficient-prep-20261019`.
The build receipt is `build-complete.json`; native test evidence is
`test-detail.log`; matched measurements are
`comparison-3414609/comparison.json`, `baseline-prep-audit.json` and
`shared-prep-audit.json` in that comparison directory. The corresponding
remote campaign is
`/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/sedov-coefficient-prep-20261019`.
Run audits retain complete arguments, both environment aliases, memory
samples, executable receipts and native stdout/stderr hashes. Fresh profiling
records are `nsys-3414629/run.json`, `attribution.json` and
`interpolation-audit.json` in that profile directory, with native
`capture.nsys-rep` and `capture.sqlite`. The historical count cross-check is
`baseline-interpolation-check.json` at the campaign evidence root.

The earlier compact-normal face experiments are preserved separately on
`igr-face-operator-opt` at `565cda8`. They were neutral/slower in the evaluated
large case, and the production mesh did not admit the normal dictionary.
Their production code and test-only benchmark were restored to the validated
baseline before this candidate was built; no face prototype is included in
these coefficient-preparation results.
