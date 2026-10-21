# Compact FP64 DG face-lift storage

The compact face representation reduced the measured record-plus-gather
kernel time by 10.01% on the actual 512² p5 mesh. One matched full Sedov
pair measured 1.49% lower application time and 4.85% lower native IGR solve
time. The complete final VTK files and all 14,818 independent residual
certificate records are byte-identical. This is a modest observed benefit
from one sequential pair, not a statistically established speedup.

## Change and scope

Tested candidate `519b33997399f2adadfac62d4505bbe3ab536f38` on
`igr-current-operator-opt` adds default-off
`IGRLAB_IGR_COMPACT_FACE_LIFT` (legacy alias
`DG_IGR_IGR_COMPACT_FACE_LIFT`). It stores 24 FP64 face-lift values rather
than 72 expanded native rows, and reconstructs the endpoint products in
the deterministic gather. Per-face FP64 rounding and ordered accumulation
are retained; there are no atomics or new diagnostic state scans.

This pass changes FP64 SIMT work in the current IGR operator. It does not
change the four TF32 preconditioner GEMMs, the current equation, solver
tolerance, or the previous tensor nodal conversion optimization.

The gather uses a validated native-to-tensor inverse and the current face
slot/axis data. Existing geometry/coefficient admission and fallback paths
remain. Component tests include GL/GLL, permutations, skew/curved fixtures,
partial batches, refresh/fallback/recovery, and exceptional arithmetic.
Full evolution evidence remains the single Cartesian Sedov case; those
component tests do not establish general curved-mesh solver performance.

## Baseline Nsight Compute evidence

Job 3425246 continuously evolved frozen source
`01140046e5fa2a35f56bf12d2f1340350f484ab9` to the selected RK window,
then captured two launches each of mass, diffusion, face record and gather.
All eight launches and requested counters passed the independent audit.
The intentional profiler cutoff is separate from the completed physical runs.

| Kernel | Mean duration, µs | DRAM throughput, % of peak | Long-scoreboard stall, % |
|---|---:|---:|---:|
| Mass | 175.808 | 41.08 | 39.05 |
| Diffusion | 300.240 | 37.02 | 32.16 |
| Face record | 569.440 | 25.29 | 52.23 |
| Gather | 156.880 | 78.41 | 86.825 |

This motivated reducing stored face contributions and their memory traffic.
Replay flushes caches and perturbs timing; these are baseline hardware
diagnostics, not full-run timing or proof of a unique bottleneck. No fresh
candidate full-case kernel capture is included in this result.

## Native validation and kernel benchmark

Canonical DeltaAI `build.sh` job 3425410 passed the four suites
`dg_igr_exact_face_cpu`, `dg_igr_exact_face_cuda`,
`dg_igr_tensor_multigrid_cuda`, and `dg_igr_tensor_multigrid_nodal_cuda`
with the selector both off and on: eight passing checks.
The large benchmark also passed bitwise expanded-row and gathered-output
parity, with four warmups and 64 CUDA-event-timed repetitions.

| Actual 512² p5 benchmark | Expanded rows, ms | Compact lift, ms |
|---|---:|---:|
| Record | 0.554515481 | 0.472920507 |
| Gather | 0.157656506 | 0.175993502 |
| Combined | 0.715522528 | 0.643900990 |

Combined speedup is 1.11123× (10.01% less time). Gather alone becomes
slower; the record saving exceeds that cost. Setup, resets, oracle
downloads, solves and full evolution are outside these timed pairs.

Logical face-record storage falls from 301,400,064 to 100,466,688 bytes,
a 200,933,376-byte (191.625 MiB) reduction. Sampled full-process GPU memory
remains 18,376 MiB in both runs. The Umpire pool can retain freed blocks,
so this is not evidence of a lower process high-water allocation.

## Matched full evolution

Job 3425661 ran baseline then candidate sequentially on `gh006`, GPU
`GPU-14e15701-106d-342f-8e76-85e18d9df384`, under the authorized
`bdiy-dtai-gh` / `ghx4` / `projects` allocation. It completed successfully
in 24 min 33 s with one GPU. Application timing excludes wrapper checks
and telemetry teardown but includes startup and VTK output.

Both runs use ordinary DG, 512² elements, p5 GL, 37,748,736 conserved
coefficients and 9,437,184 scalar IGR coefficients; L2/BrokenL2,
t=0.1, CFL=0.3, C_alpha=1/6 with max length,
alpha=6.357828776041666e-7. Nonfinite/admissibility scans and positivity/
modal limiting are off. Both retain RHS-relative rtol=1e-6, atol=0,
current-equation FP64 certificates, and VTK output every 0.02.

| Measured quantity | Expanded rows | Compact lift |
|---|---:|---:|
| Application wall time, s | 734.070374 | 723.136544 |
| Native simulation stopwatch, s | 653.606882 | 640.526429 |
| Native IGR solve total, s | 274.521465 | 261.204374 |
| Native IGR total including preparation, s | 399.690481 | 385.759334 |
| Accepted / rejected steps | 4,939 / 0 | 4,939 / 0 |
| Certified solves | 14,818 | 14,818 |
| Krylov iterations | 51,747 | 51,747 |
| Maximum RHS-relative residual | 9.9997421e-7 | 9.9997421e-7 |
| Final accepted-state RHS-relative residual | 3.6241145233e-7 | 3.6241145233e-7 |

The observed application saving is 10.933829 s (1.49%); native solve
saving is 13.317090 s (4.85%). Native component timers are host intervals
that can include pending GPU work, not independent kernel-duration sums.
Do not multiply this incremental result by historical speedups to claim
a new original-to-current comparison.

Five adjacent analysis intervals with no VTK-save marker between their
endpoints (steps 1000–1500, 2000–2500, 3000–3500, 4000–4500 and
4500–4939) all favor the candidate: 2.34–3.58% less elapsed time.
Their combined 2,439 steps take 222.435549 versus 215.541291 s, a
3.10% reduction. These intervals include ordinary analysis overhead;
they are not GPU-only timings.

The buffered 100-ms host heartbeat has no gaps above 1.5 s (maximum
0.10049/0.10071 s), both telemetry samplers report no errors, and their
threads stopped before the next phase. Before/after health records show
no ECC, thermal or hardware-slowdown indications. Dynamic clocks remain
a confound: in samples with at least 90% GPU utilization, mean SM clock
is 1932.07 versus 1943.49 MHz (+0.59%); memory clock is 2619 MHz in both.
The independent timing audit supports an observed compute-path benefit
without attributing every saved second solely to the code change.

## Physical agreement

Final time 0.1 was selected from each PVD, then the complete single-rank
VTU was streamed through SHA-256. Both files are 1,160,775,042 bytes,
with identical digest:

`9ab51b7545c03104a6fbea77c4b95e16e0e207eecec5814859b690bd229760a2`.

This establishes identical saved density, energy, momentum, sigma,
velocity, pressure, mesh and attribute payloads. It does not add an
analytic PDE-accuracy claim or certify unsaved intermediate values.
The entire 14,818-line independent-certificate sequences also match:

`8703904de75494b092429fd3f9ae02caf4d1c33b01aef6b6853c4304e214cc15`.

All solves pass the original FP64 current-equation tolerance. There are
no residual-replacement retries or rejected steps.

## Source and evidence identities

- Baseline source: `01140046e5fa2a35f56bf12d2f1340350f484ab9`.
- Candidate source: `519b33997399f2adadfac62d4505bbe3ab536f38`.
- Baseline binary: `0acd87c208d6b2de440a72c021d26ca1a87a63c2c2e3377b6a233f9800caa5c5`.
- Candidate binary: `9ca494cfae7acdda163c3dea29f90db19e913e3e90ec20026012882dba1e4c57`.
- Case specification: `df0fe7ae85da4f80b503cf8e141c4ab1b3a298a474eef9d8dadf00f95c120680`.
- Native benchmark gate: `3c1e68ef6a8ccb534b37b05a7ac506f6ca9cb8212ab178c881aaf98e19e4f3c7`.
- Local evidence: `/home/jarias9/.cache/igrlab-experiments/igr-current-operator-20261020`.
- Remote evidence: `/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/igr-current-operator-20261020`.

The first native attempt, 3425389, failed because the harness passed a
GPU-only production flag to the CPU test. The source compiled and CUDA
checks passed; correcting the test environment yielded the eight-pass
3425410 gate. Comparison attempt 3425540 failed before either application
started because a Slurm physical GPU index was reused as an NVML visible
index. UUID-based discovery repaired the launcher for 3425661. Neither
failed attempt is included in the application timing comparison.

The optimization remains opt-in. Its local memory/work reduction and
saved-field equivalence are established here; the precise end-to-end
benefit remains subject to run-to-run variation.
