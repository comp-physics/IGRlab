# Local Jacobi tensor-core prototypes

The fastest measured inverse application is the grouped dense TF32 backend:
0.171616 ms versus 0.315488 ms for the existing FP64 SIMT implementation
(1.8383x). The preferred accuracy/performance tradeoff is grouped dense **FP64
tensor cores**: 0.187312 ms (1.6843x), retaining roundoff-level agreement with
the independent dense oracle. It captures 89.1% of the fastest variant's time
saving. These are complete local-inverse timings, not whole-simulation speedups.

## Implementations

The existing surrogate applies

    z_e = S_e B_c S_e r_e,
    B_c = (Q_y tensor Q_x) diag(1 / (lambda_x + lambda_y)) (Q_y tensor Q_x)^T,

with current FP64 inverse-square-root mass scaling S_e and a verified,
initial-state reference class c. Every variant keeps this same surrogate in
exact arithmetic. Changing arithmetic introduces rounding differences; it does
not make two fixed Jacobi sweeps a converged sigma solve.

The default kernel applies the inverse through four directional contractions.
The prototypes either batch those contractions or precompute a complete 36x36
inverse once per class. Grouped kernels process 16 distinct elements per block,
gather their residuals directly into shared memory, multiply, apply the final
current mass factor, and scatter FP64 corrections. The sigma update remains the
existing separate operation. There is no global packed-residual buffer, repeated
LU, per-element dense inverse cache, coarse correction or fallback.

This case has nine factor classes. Dense variants therefore retain only nine
36x36 matrices: 93,312 bytes in FP64. They have 16,389 element groups, including
80 padded slots (0.0305%). Grouped dense variants own 191,007,796 logical
persistent bytes versus 189,800,032 for the existing backend. These are
backend-owned arrays, not process peak GPU memory.

Select the existing tensor Jacobi policy with

    --igr-solve-mode element-block-jacobi --igr-jacobi-blocks tensor

and choose the implementation through `IGRLAB_IGR_LOCAL_TENSOR_EXECUTION`:

| Selector | Representation and arithmetic |
| --- | --- |
| `simt-fp64` | Default unchanged directional FP64 kernel |
| `simt-fp32` | Same directional representation with FP32 contractions |
| `simt-dense-fp32` | Grouped shared dense inverse, ordinary FP32 dot products |
| `wmma-factor-tf32` | Grouped directional transforms, TF32 WMMA and FP32 accumulation |
| `wmma-dense-tf32` | Grouped shared dense inverse, TF32 WMMA and FP32 accumulation |
| `wmma-dense-fp64` | Grouped shared dense inverse, FP64 WMMA |

The legacy selector `DG_IGR_LOCAL_TENSOR_EXECUTION` is also accepted. Selection
is frozen when the backend is constructed. Grouped modes require p5; WMMA
requires SM80 or newer. Geometry remains fixed affine Cartesian GL quadrilaterals
with the original separability checks. This work broadens neither mesh support
nor solver convergence guarantees.

Actual arithmetic is recorded by `[igr-local-tensor-jacobi]` using `backend`,
`precision`, `execution` and `tensor_core`. Older high-level policy names
`tensor-density-fp64` / `tensor-element-density-fp64` remain legacy metadata in
this prototype and do not describe an opt-in backend's arithmetic. Use the
detailed backend record and the instruction measurements below.

## Native qualification and measurement

Native job **3438902** used one GH200 on **gh086**, source
`3508bababd4f4fda9326c2c03fd715f5f31021b0`, built with the canonical remote
`build.sh` driver for CUDA architecture 90. All eight focused suites passed:
the existing parser, lifecycle, Krylov, strict p2/p5 inverse and rejection
checks, plus the six-backend qualification. The latter uses independently
assembled MFEM matrices and pivoted FP64 dense inverses, reordered elements,
uniform/heterogeneous density changes, signed inputs, finite results and
sampled symmetry/positive-quadratic-action checks.

FP32/TF32 cases test residual exponents -80, 0 and +80; FP64 additionally
covers -700 and +700. Float variants use direct casts without power-of-two
rescaling and are not qualified for the full FP64 dynamic range. Their error
envelopes account for operand rounding, accumulation and operator norms;
passing them is not a 1e-6 residual or physical-accuracy certificate.

The full benchmark uses 512x512 ordinary-DG/GL p5 elements, 9,437,184 sigma
unknowns, unit-square geometry, alpha=6.357828776041666e-7, SIPG penalty 144,
GL6x6 mass/diffusion and GL6 face quadrature. These match the fixed Sedov
operator setup. The current density and signed residual are synthetic,
nonconstant inputs, not a saved evolved Sedov state.

All modes run on the same GPU, with one backend live at a time. CUDA events
measure the complete production `Mult`, including gather, current mass
scaling, shared-memory packing, inverse arithmetic and scatter. Setup, the
coefficient-refresh kernel and offline correctness work are excluded. Each
round has eight warmups and 64 timed calls; three rotated/reversed rounds
produce the table below. The interval is the range of round medians, not a
confidence interval. Ordering is not completely balanced: the baseline is first
in two rounds and fifth in the third.

| Backend | Median ms | Round-median range, ms | Speedup | Correction relative L2 | Relative max norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP64 directional SIMT | 0.315488 | 0.313472-0.316384 | 1.0000x | 4.82e-14 | 2.83e-13 |
| FP32 directional SIMT | 0.310976 | 0.310656-0.311984 | 1.0145x | 1.43e-7 | 2.54e-7 |
| FP32 shared dense SIMT | 0.232128 | 0.231456-0.232432 | 1.3591x | 1.12e-7 | 2.55e-7 |
| TF32 directional WMMA | 0.191136 | 0.190256-0.193184 | 1.6506x | 4.38e-4 | 9.08e-4 |
| TF32 shared dense WMMA | 0.171616 | 0.171104-0.172864 | 1.8383x | 1.23e-4 | 1.73e-4 |
| FP64 shared dense WMMA | 0.187312 | 0.186896-0.188368 | 1.6843x | 4.82e-14 | 2.84e-13 |

Errors compare one complete inverse application with the independent compact
nine-class dense oracle. They are not errors in evolved sigma or flow fields.
The maximum normalized initial-class infinity-norm condition estimate is about
1898.02. The current-density scale is identical for all variants.

The dense FP32 SIMT control already achieves 57.9% of the fastest variant's
time saving. Consequently, grouping and representation account for much of the
improvement; lowering precision alone barely changes the directional kernel.
FP64 WMMA is another 19.3% faster than that dense FP32 SIMT control. All tested
kernels fuse the mass factors and avoid global packed-vector intermediates.

## Hardware confirmation

The three WMMA paths are verified by positive hardware instruction counters
and matching SASS from the same frozen benchmark executable. NCU job 3438903
captured the directional TF32 kernel successfully; its exporter parser expected
long-format CSV but this NCU version emitted wide-format CSV. That saved capture
was recovered without a GPU rerun. Continuation 3438927 reused it and captured
only the two dense kernels. All three evidence checks passed.

| Backend | Executed tensor instructions | DRAM bytes | Tensor-pipe active percentage* |
| --- | ---: | ---: | ---: |
| Directional TF32 | 1,573,344 HMMA | 254,426,368 | 3.68% |
| Shared dense TF32 | 983,340 HMMA | 255,153,152 | 2.69% |
| Shared dense FP64 | 1,475,010 DMMA | 254,911,744 | 13.24% |

*The exact counter is
`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`.
TF32 SASS contains `HMMA.1684.F32.TF32`; FP64 SASS contains `DMMA.8x8x4`.
Each count matches the 16,389 groups and the implemented warp/tile schedule.
One selected launch per backend was captured, with cache flushing and dynamic
clocks. These are diagnostic hardware samples; the unprofiled CUDA events above
supply the performance comparison. The first capture and continuation ran in
separate allocations, so their durations are not a controlled timing comparison.

Substantial DRAM traffic and low tensor-pipe activity are consistent with data
movement and instruction overhead limiting the value of TF32's higher peak
arithmetic throughput. These counters alone do not prove saturation of DRAM
bandwidth or isolate every stall cause. No further parameter sweep or physical
simulation was run for this bounded comparison.

## Implication for the full solver

The earlier validated local-FP64 Sedov run used 29,684 inverse applications,
took 503.380437 s overall, and accumulated 78.677081 s in the complete solve
scope. Multiplying the measured per-application differences by that call count
gives the following **projection**, not a measured application speedup:

| Candidate | Projected seconds saved | Fraction of prior application time |
| --- | ---: | ---: |
| Grouped dense FP64 WMMA | 3.805 | 0.756% |
| Grouped dense TF32 WMMA | 4.271 | 0.848% |

The extra TF32 benefit over FP64 WMMA is about 0.466 s, or 0.093% of the earlier
application time. Do not divide the entire 78.7 s solve counter by the inverse
speedup: current-operator applications and vector work are still required.
Different inputs, GPU conditions and surrounding kernels can change the
transferred estimate.

The bounded prototype comparison favors opt-in `wmma-dense-fp64`. A new full
physical pair is not justified for resolving a sub-percent projected timing
difference in this experiment. Evolved flow-field accuracy and actual end-to-end
runtime of the new variants remain unmeasured; a later promotion to production
would require the corresponding physical regression. The original validated
`simt-fp64` remains the default.

Exact measured values, source/executable hashes, profiler evidence and timing
scope are in [the measurement receipt](benchmarks/dg_local_jacobi_tensor_core_measurements.json).
