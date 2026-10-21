# Collocated FP64 IGR volume application

Status: the native suites, large-mesh operator benchmark and complete matched
Sedov comparison are finished. On one GH200, the candidate reduced full
application time from 711.511 s to 694.245 s (2.43%), native IGR solve time
by 8.21%, and five intervals without VTK output by 5.14%. Peak sampled GPU
memory increased by 1,072 MiB (5.83%). All current-equation residual checks
passed. Final conserved-field relative L2 differences were at most 1.964e-9.

The admitted collocated SIMT kernel takes 0.227334 ms versus 0.581040 ms
for the original public partial-assembly volume application on the actual
512² p5 mesh: 2.55588× faster locally. This is an FP64 SIMT optimization;
the existing TF32 spectral inverse is unchanged. Its additional coefficient
refresh and persistent arrays dilute the application benefit and impose a
material capacity cost. Keep this path opt-in while reducing that storage.

## Cumulative baseline and target

This pass follows the validated compact face-lift implementation
`519b33997399f2adadfac62d4505bbe3ab536f38` and report checkpoint
`ccaf56c091a7485dbb4e5f2d9da0e12db4ec4852`. All earlier coefficient,
tensor-face, compact-face-data, spectral nodal and compact-lift improvements
remain enabled. TF32 spectral inverse GEMMs, FP32 Euler tensor lines, FP64
current-equation certification and scan-off policy are retained.

Fresh Nsight Systems job 3426187 continuously evolved the frozen baseline
and recorded 15.004709 s. The diagnostic application was intentionally
terminated after recording. The native trace proves 3,539 launches each of
the compact face record, compact gather, p5 mass and p5 diffusion kernels;
the older expanded face kernels are absent.

| Measured baseline kernel group | Summed time, s | Share of kernel time |
|---|---:|---:|
| Mass and diffusion kernels | 1.6790 | 11.89% |
| Public PA volume including correlated gather/clear/scatter | 2.1009 | 14.87% |
| Compact IGR face record and gather | 2.3317 | 16.51% |
| TF32-named cuBLAS GEMMs | 1.7005 | 12.04% |
| Spectral nodal conversions | 0.6528 | 4.62% |

The public-volume row includes the mass/diffusion row. These are kernel-time
groups, not independent whole-application elapsed partitions. GPU activity
covers 97.96% of the recording and 98.27% of the complete-RK interior.
There are 167 complete RK ranges, averaging 88.773 ms, with maximum
intervening gap 18.08 microseconds. This supports a representative continuous
compute interval. The recording does not encode absolute RK indices.

## Existing fusion controls

The application already contained a fused FP64 volume operator, with SIMT
and FP64 WMMA modes, but the cumulative Sedov configuration kept it off.
Test-only commit `3a0664ed502e7f6c6f331e568868d18d251282e6` adds
`dg_igr_tensor_volume_test --benchmark512 <Sedov mesh>` to compare these
existing paths with public `ParBilinearForm::Mult` on the real mesh.

Canonical DeltaAI job 3426235 passed the original native volume suite and
the large benchmark. Mean public apply times across two nonuniform
coefficient generations were 0.588773 ms for PA, 0.831798 ms for generic
fused SIMT, and 0.481040 ms for FP64 WMMA. Fused refresh added about
0.3404 ms per update. The generic SIMT path regressed; WMMA's apply benefit
was modest after the extra refresh cost. No full physical comparison was
submitted for these two controls.

## New implementation

Candidate `3125e796afc600d622454856f94ff99ef665439a` changes only
`src/physics/igr/IGR_tensor_volume.cu` and its existing native test. The new
`IGRLAB_IGR_TENSOR_VOLUME_COLLOCATED` selector, with legacy
`DG_IGR_IGR_TENSOR_VOLUME_COLLOCATED`, defaults off and is effective only
with `IGRLAB_IGR_TENSOR_VOLUME=simt`.

Construction inspects the actual mass and diffusion tensor maps. Admission
requires equal D1D and both Q1D values, finite B/Bt/G/Gt arrays of the expected
sizes, exactly zero off-diagonal B entries, and exact transpose consistency.
It preserves the actual diagonal entries; they are never snapped to one.
Failed admission retains the generic SIMT path. Existing rejection of
nonfinite forward maps remains unchanged.

Diagonal interpolation reduces twelve dense tensor contractions to four
derivative contractions plus pointwise diagonal scaling. One element per
warp retains its input and two metric-weighted flux fields in shared memory.
The kernel preserves separate mass/diffusion diagonal factors at their
forward and transpose stages, actual per-quadrature geometry/current
coefficients, signed native DOF maps, and final mass plus diffusion
accumulation. A CTA barrier precedes the return of inactive tail warps.

The implementation dispatches compact shared-memory kernels for D1D 1–8;
native off/on oracle coverage includes p1/p3/p5/p7, with the performance
claim limited to p5. Existing generic/WMMA kernels, padded coefficient
storage and refresh remain. The frozen spectral preconditioner is still
constructed from the original FP64 operator, and independent residual
certificates still evaluate that original current operator.

## Native validation and actual-mesh result

Job 3426399 built the application and test through canonical `build.sh`.
The focused CUDA volume suite passed, including the unchanged PA and dense
oracles with the selector off and on. Additional p5 fixtures verify real
admission, distinct nonunit mass/diffusion diagonals on rotated/skew cells,
off-diagonal rejection, GLL and unequal-rule fallback, WMMA selection,
current coefficient updates, nonfinite visibility and finite recovery.
Contrived cached-map fixtures use the matching public PA operator as their
oracle and restore the maps afterward.

The large benchmark uses the actual Sedov mesh with zero serial and four
parallel refinements: 262,144 elements and 9,437,184 scalar DOFs, p5 GL,
separate original 6×6 mass/diffusion rules, and two nonuniform current
coefficient generations. Timings use four warmups and 64 CUDA-event-timed
repetitions per variant and generation, excluding coefficient generation,
setup and host oracle comparisons.

| Path | Mean public apply, ms | Additional fused refresh, ms |
|---|---:|---:|
| Original PA | 0.581040 | — |
| Generic fused FP64 SIMT | 0.824218 | 0.341480 |
| Fused FP64 WMMA | 0.480388 | 0.340620 |
| Admitted collocated FP64 SIMT | 0.227334 | 0.340711 |

Original PA refresh, 0.256621 ms in this fixture, remains common work
because PA is retained for independent certificates. Refresh event intervals
include host submission gaps. The maximum large-case relative L2/Linf
differences remain below 3.59e-16. Both coefficient generations and actual
collocated selection passed.

Each fused instance adds source-derived persistent arrays of 1,111,493,632
bytes: 512 MiB geometry, 512 MiB weighted coefficients, 36 MiB DOF map and
3,072 bytes of basis maps. This is not a measured process-memory saving;
three fused controls coexist only inside the benchmark.

Slurm marked 3426399 failed because the postprocessor looked for the
benchmark field `kernel` instead of `actual_kernel`. All seven build,
native, dependency-inspection and benchmark commands returned zero.
The one-key parser repair was independently reviewed and rerun against the
same pinned raw output. Original failed records are preserved alongside
`benchmark-recovered.json`; no GPU rerun was needed.

## Full matched comparison

Job 3426538 completed both arms on gh091, using the same observed GH200
GPU UUID `GPU-7e58e92a-1765-a91b-5060-930deb107803`: baseline volume off,
then candidate volume SIMT with collocation enabled. All earlier
optimization flags remained enabled in both arms. The case was ordinary
DG, 512² p5 GL, L2/BrokenL2, t=0.1, CFL=0.3, C_alpha=1/6 with max length,
physical alpha=6.357828776041666e-7, RHS-relative rtol=1e-6 and atol=0.
Nonfinite/admissibility scans, positivity limiting and modal damping were
off; VTK cadence was 0.02. Application timing includes setup and output.

| Measurement | Baseline | Candidate | Observed change |
|---|---:|---:|---:|
| Full application, s | 711.511002 | 694.244841 | 2.4267% less |
| Native IGR solve, s | 256.805861 | 235.729663 | 8.2071% less |
| Native update, s | 387.156105 | 366.538776 | 5.3253% less |
| Five intervals without VTK saves, s | 212.891282 | 201.950142 | 5.1393% less |
| Steps | 4,939 | 4,939 | unchanged |
| Krylov iterations | 51,747 | 51,742 | 5 fewer |
| FP64 current-A certificates passed | 14,818 | 14,818 | complete coverage |
| Maximum RHS-relative residual | 9.9997421e-7 | 9.9954522e-7 | both below 1e-6 |
| Sampled process GPU peak, MiB | 18,376 | 19,448 | 1,072 more |

All five disjoint intervals without VTK saves favored the candidate. They
cover 2,439 steps and 28,362 versus 28,369 iterations: the improvement in
these intervals is not explained by fewer iterations. These are native
elapsed-time intervals between analysis callbacks, not GPU-only timings.
The native solve and update timers are nested; their savings must not be
added. The candidate reports 14,818 coefficient refreshes and 81,377
submitted collocated volume applications, with no fallback.

The observed full-application saving is 17.266161 s, or 1.02487× speedup.
Both arms used dynamic clocks; mean SM clock at sampled utilization >=90%
was 1965.55 MHz for the baseline and 1954.86 MHz for the candidate (0.54%
lower). No long external-monitor gaps, sampling failures or hardware error
counter indications were found. Both runs contain filesystem-wait samples;
the monitor does not prove absence of application or storage stalls. This
is one sequential, baseline-first pair, not a repeat distribution or a
frequency-normalized causal estimate. No candidate Nsight kernel hardware
profile is claimed. Historical incremental speedups should not be multiplied.

### Complete final-field comparison

The final native Q5 VTUs at t=0.1 have different hashes. Both complete files
were downloaded and verified against remote whole-file SHA-256 hashes.
The comparison checks all 262,144 cells and reconstructs each saved Q5
polynomial at GL6×6: 9,437,184 points per scalar field, with matching geometry,
alpha and time. The L2 norms of polynomial differences are integrated
exactly up to FP64 roundoff on these affine cells. The infinity values below
are sampled maxima over GL points, not continuous polynomial suprema.

| Field | Relative L2 difference | Relative sampled Linf difference |
|---|---:|---:|
| Density | 6.84347e-10 | 7.54781e-9 |
| Momentum x | 1.96396e-9 | 8.12056e-9 |
| Momentum y | 1.96192e-9 | 8.46937e-9 |
| Energy | 5.37097e-10 | 3.39721e-9 |
| IGR sigma | 9.46184e-8 | 3.77093e-7 |

All conserved components are far below the existing engineering comparison
budgets of 1e-3 relative L2 and 1e-2 sampled Linf. Sigma has no separately
assigned budget: its measured absolute L2 difference is 1.89442e-12 and
sampled Linf difference is 2.13168e-11. Derived pressure and speed relative
L2 differences are 5.27594e-10 and 1.43641e-9, respectively; these derived
norms are quadrature estimates. These results establish close agreement
with the cumulative baseline for this case, not analytic Sedov accuracy
or a claim about other meshes, resolutions or shock problems.

### Decision and next target

Retain the new kernel as an opt-in cumulative improvement. The observed
8.21% reduction in IGR solve time is useful, but the 1.05 GiB sampled GPU
peak increase matters for the goal of fitting larger simulations. The next
priority is reducing padded or duplicated fused geometry/coefficient storage
while retaining original-A certification, before treating this configuration
as the preferred setup. This pass makes no general-mesh portability claim.

Independent review checked production admission and kernel logic, native
test oracles, parser recovery, matched run configuration, residual coverage,
timings and telemetry. The complete numerical field comparison is retained
separately with pinned helper identities and full-file collection receipts.

## Identities and retained evidence

- Baseline application SHA-256: `9ca494cfae7acdda163c3dea29f90db19e913e3e90ec20026012882dba1e4c57`.
- Candidate application SHA-256: `a721149b79c23ce888bfcd6f5b9a5976f0a428b7a54b0f4e1945421f7104218c`.
- Candidate test SHA-256: `49b01408e2bebff78ae39423cb583711ea345300349f0bff13b32b408010475a`.
- Candidate native build receipt: `1f1364b604324c4800e47ede86db1833c50e61358e6dfcd1bb34e343152a7673`.
- Full-case specification: `b091aaa7618446fcdba5aab927747026af471b1b780c1df405938a45b9959c29`.
- Full-case admission gate: `1854bf1d77e09d43f3af82c6980ff5ed702b45d59c7afb8b509462fcfd072ebc`.
- Baseline final VTU SHA-256: `9ab51b7545c03104a6fbea77c4b95e16e0e207eecec5814859b690bd229760a2`.
- Candidate final VTU SHA-256: `81d8bf54ddec7ab4f0b6293d77283d6c1ee0f9d19304c1b04f6846bc191debb3`.
- Local campaign: `/home/jarias9/.cache/igrlab-experiments/igr-volume-operator-20261020`.
- Remote campaign: `/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/igr-volume-operator-20261020`.

Within the local campaign, `comparison-3426538/analysis/independent-timing-review.json`
records the independent timing audit; `comparison-3426538/field-comparison.json`,
`conservative-fields.json` and `sigma-fields.json` record actual final norms.
Final collection receipts and complete VTUs are retained in each variant's
`final-vtk` directory. Build, benchmark and profile evidence remains in
`build-3426235`, `build-collocated-3426399` and `nsys-3426187`.

The optimization remains default-off. Portability and integration into
`dev`/`main` are deferred to the later campaign requested by the user.
