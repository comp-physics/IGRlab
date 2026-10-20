# Compact IGR face coefficient updates

**On the matched 512² p5 DG Sedov pair, compact face preparation reduced
full application time by 4.7817% (776.204 → 739.088 s) and sampled peak
GPU residency by 3,006 MiB (21,382 → 18,376 MiB). The complete final native
VTU payloads are byte-identical. CPU/CUDA tests and all current-equation
FP64 residual certificates passed.**

This pass follows the validated FP64 tensor-face implementation. Its target
is the remaining dense face preparation, not an additional Tensor Core
kernel or a change to the solver's precision.

## Matched full Sedov result

Job **3422767** ran baseline then candidate, sequentially on `gh006`,
GPU `GPU-8d1ac016-1267-753e-4674-204cd5dfdf83`. Both use the same physics,
solver, tensor flags and output settings described below. Only the compact
qdata selector, frozen executable identity and output directory differ.

| Metric | Previous tensor-face baseline | Compact qdata |
|---|---:|---:|
| Full process time, s | 776.203667 | 739.087664 |
| Sampled peak process GPU memory, MiB | 21,382 | 18,376 |
| Accepted / rejected steps | 4,939 / 0 | 4,939 / 0 |
| Converged / independently certified linear solves | 14,818 / 14,818 | 14,818 / 14,818 |
| Krylov iterations | 51,732 | 51,732 |
| Maximum FP64 RHS-relative residual | 9.9998798e-7 | 9.9998798e-7 |
| Final committed-state RHS-relative residual | 3.615794158e-7 | 3.615794158e-7 |
| Full-process ns per conserved DOF per step | 4.163267 | 3.964190 |

The candidate saves **37.116003 s**, giving **1.050219×** speedup. The
observed peak reduction is **2.93555 GiB / 14.0586%**. These are one
unprofiled full run per variant, including setup and output; they do not
establish a repeat distribution. GPU memory is sampled once per second and
can miss brief peaks. The per-DOF figure divides full time by 37,748,736
conserved coefficients and 4,939 steps, including all IGR and output work.
There are no certificate gaps or residual-replacement retries.

Both complete final VTUs at t=0.1 and alpha=6.357828776041666e-7 contain
**1,160,775,042 bytes**, with identical SHA-256:
`c2e6c8cd52a227b30f6125974fa634632a686787a7ea44d448d1ec472c04a5ed`.
Original PVD/PVTU manifests select the final time, and source-bound
collection receipts verify the native payload after transfer. This
establishes identity of every saved field, including sigma, and geometry.
Additional numerical norm evaluation is unnecessary; this is implementation
agreement, not error against an analytic Sedov solution.

## Implementation

Candidate `7b612502798864fe90faff371d4a8d46ca3cc68e` adds
`IGRLAB_IGR_COMPACT_FACE_QDATA=1` (legacy alias
`DG_IGR_IGR_COMPACT_FACE_QDATA=1`), default off and sampled at construction.
It also prepares the existing tensor representation when the older
factorized-face selector is unset.

For admitted ordinary-CUDA p5/Q6 tensor faces with arithmetic, max or sum
scaling, each coefficient update retains current side coefficients and
penalties without multiplying every dense normal-derivative entry. It
releases the full shape, geometry-normal and weighted-normal device
allocations. The existing tensor action and deterministic gather are
unchanged. The diagonal reconstructs the needed entries directly from
endpoint/tangent factors, physical geometry and current coefficients.

Independent coefficient generations protect dense fallback reads: every
interior dense reader materializes current weighted normals before use,
even after tensor metadata is rejected. Initial construction, unsupported
orders/backends, shared faces and harmonic scaling preserve dense behavior.
Exact dense host geometry, the reference cache and its admission budget
remain; this pass does not eliminate their construction or host footprint.
The 256 MiB factor-cache budget can still decline larger cases, so the
tensor admission must be checked before applying these performance numbers
to another mesh size.

The precision arrangement remains FP64 current IGR operator, coefficients,
Krylov work and residual certification; TF32 spectral-inverse GEMMs; FP32
Euler tensor-line operations and FP64 Euler face work.

## Memory interpretation

The installed MFEM build and native device log select
`host-umpire,cuda-umpire`. Its `MFEM_DEVICE` allocator uses Umpire QuickPool.
Removing a vector's device allocation returns that block for reuse within
the pool; it need not reduce CUDA process residency. The native benchmark
therefore reports actual MFEM device pointers/bytes separately from
`cudaMemGetInfo`, and the full runs sample process memory using SMI.
An initial dense allocation can also affect the process's high-water mark.

## Matched workload and validation scope

Both variants use the fixed 512² ordinary-DG p5 GL Sedov case on one
DeltaAI GH200, sequentially in the same allocation. Settings remain
L2/BrokenL2, SSP RK3, every-stage IGR updates, t=0.1, CFL 0.3,
C_alpha=1/6 with max element length, physical alpha
6.357828776041666e-7, FP64 FGMRES and RHS-relative tolerance 1e-6.
Nonfinite/admissibility diagnostic scans, positivity limiting and modal
damping are off. Output cadence remains 0.02.

The baseline is the previous tensor-face source
`c4a7498b3aa52778f19a02be9a09f644342fcb02`, binary
`5ca34c4e03bd2c04353b0f05555c9f86b5da7c6804cde597e9deb620f07d3fd4`.
The comparison is incremental over the accelerated pipeline, not a fresh
comparison against the original FP64 GPU solver.

Canonical remote `build.sh` checks cover CPU and native CUDA. Added
fixtures compare evolving-qdata actions and diagonals against independently
rebuilt dense data and an assembled MFEM diagonal; they exercise repeated
updates, fallback/re-entry, stale metadata, same-size geometry/rule rebuilds
and an unsupported order. Device-allocation witnesses confirm that compact
action and diagonal consumers do not silently restore the large arrays.

The separate 512² benchmark measures refresh alone and refresh plus public
face apply (including output reset and gather), with four warmups and
64 calls measured with CUDA events. It excludes coefficient projection,
diagonal assembly, setup and solves. Full application timing includes
setup, continuous evolution, output and finalization; profiled time is not
used as a benchmark. Complete saved-Q5 payload identity at matching alpha
and final time supplies the physical comparison for this pair, alongside
FP64 residual certificates.

## Fresh baseline Nsight Systems

Job **3422519** profiled the complete previous tensor-face application on
`gh107`, with all 4,939 time steps and no rejections. Among 451.040278 s of
summed GPU kernel durations, the current IGR face family remains largest
at 99.586936 s (22.0794%):

| Face work | Calls | Summed GPU time | Share of all kernel time |
|---|---:|---:|---:|
| Tensor record | 96,183 | 55.188817 s | 12.2359% |
| Dense coefficient refresh | 14,817 | 28.441750 s | 6.3058% |
| Deterministic gather | 96,509 | 15.517553 s | 3.4404% |
| Dense record | 326 | 0.438815 s | 0.0973% |

These are work-duration statistics, not application wall-time fractions.
GPU activity covers 60.7736% of the whole recording and 72.4628% of the
evolution interval. In a verified 50-step evolved window with no scheduled
VTK save or analysis callback, activity covers 98.5030% of 5.09294848 s.
The timeline supports removing repeated dense preparation; it does not
predict the measured end-to-end improvement or provide new hardware-counter
evidence. The native report and SQLite remain preserved.

## Native validation and fixed-mesh benchmark

Job **3422755** passed both canonical exact-face suites, retaining the
448-case suite on each backend. New lifecycle coverage exercised 21
compact refreshes, 25 dense fallbacks and 88 diagonal comparisons on CUDA.
Overall normalized relative L2/Linf maxima were 1.045491295e-15 /
1.066782270e-15 on CUDA and 6.164260514e-16 / 9.923555996e-16 on CPU,
below the 5e-12 bound.

Tested source is `14cb9ffea660053d02cfb5584db7279fa5f401f0`; the application
SHA-256 is
`596f341e13da8da88bdff7044d242c47d061440158e84730978b02d980935e44`.
The preceding attempt, job 3422692, built successfully but failed because
the newly independent dense test oracle omitted its gather-incidence setup.
`14cb9ff` fixes only that test setup; the production implementation is
unchanged from `7b61250`. Both attempts' inputs and logs are retained.

The native benchmark ran on `gh027`; the frozen GPU identity is
`GPU-4a013246-e5b8-90db-693b-9dcb303f4f69`.

| Benchmark metric | Dense refresh + tensor action | Compact refresh + tensor action |
|---|---:|---:|
| Refresh only | 2.069881439 ms | 0.172545999 ms |
| Refresh + public apply | 2.836167097 ms | 0.948140025 ms |
| MFEM-owned face scratch device allocations | 4,201,921,004 B | 585,120,236 B |
| Full normal-base + weighted-normal device allocations | 3,616,800,768 B | 0 B |

Refresh is 11.9961× faster; refresh plus apply is 2.9913× faster.
The combined output is byte-identical to the dense-refresh/tensor-action
reference in this synthetic fixed-mesh benchmark. Both variants actually
admit the tensor action. The full shape table is already absent from the
baseline's device allocations; the additional saving is the two normal
tables, **3,616,800,768 B (3.37 GiB)**. `cudaMemGetInfo` reports zero increase
in free memory because the pool retains the released blocks.

## Evidence and next target

Local campaign root:
`/home/jarias9/.cache/igrlab-experiments/sedov-face-qdata-20261020`.
Remote root:
`/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/sedov-face-qdata-20261020`.

- `nsys-baseline-3422519/`: original run/build identities, native report,
  SQLite and timeline attribution; small provenance/analysis files also
  retained locally.
- `build-3422755/`: canonical build, detailed CPU/CUDA tests, fixed-mesh
  benchmark, gate and case specification. Build receipt retained in remote
  `binary/build-complete.json` and in the local build directory.
- `comparison-3422767/`: matched timing and complete certificate audits;
  local final-VTK collection receipts and `field-comparison.json`.
- `attempt1/` and `build-3422692/`: preserved first package and failed
  test-oracle attempt.
- Case specification SHA-256:
  `71ca7999fca65d67e7fe68af4299bcdfa7d6161d0458fe75182cb9a06c840741`.

The four allocated jobs used **2,583 GH200-seconds / 0.7175 GPU-hours**,
including the failed test attempt and profiling. Each requested one GPU,
eight CPUs and 96G host memory under `bdiy-dtai-gh`/`ghx4`/`projects`.
Queue waiting is excluded. The accounting snapshot is
`campaign-accounting.json`.

The next proposed target is the spectral nodal conversion around the TF32
inverse. Fresh baseline profiling measures 29.305 s in `ToSpectral` and
9.020 s in `FromSpectral`. Their dense 36×36 transforms are constructed from
products of 1D Legendre values; cooperative FP64 1D contractions could avoid
repeated input loads and scaling. That is an untested candidate requiring
native comparison and current-equation certification. Tensor face-density
projection is a second candidate, with 20.777 s of measured baseline GPU
work and a path to supporting admitted skew/curved geometry.
