# Tensor-oriented DG + instantaneous IGR investigation

Measured 2026-09-05 on the local RTX 5080. **The strongest new result is a
reusable dense inverse preconditioner, with about 3.4× lower full-process time
on a small smooth problem at matched solution quality. Tensor arithmetic did
not improve it.** Larger, redundant transport matrices also lost to a strong
ordinary-CUDA control at comparable arithmetic error.

The tested implementation is commit
`c38ac6b5f40a963f62c3e16af3d253645b56c165`, tree
`f98f527f650c09c9ba2afd1a61b973bc59153231`, on local branch
`codex/tensor-precision-investigation`. It starts from completed candidate
`f95ee67e4bfdc62ccfc50890400f8b843de53797`, not the initial PR61 checkout.
[Prior evidence](tensor_prior_evidence.md) distinguishes historical claims
from reruns. [Recorded measurements](tensor_precision_results.json) retain
commands, environments, binary/mesh/output hashes, individual samples,
reference comparisons and profiler evidence. Documentation added after the
tested commit does not change the tested implementation.

## Complete-application result

The existing active-wave application advances full-energy Euler with
instantaneous IGR: 2D periodic quadrilaterals, DGSEM p3/N6, 576 scalar closure
unknowns, central volume flux, Rusanov faces, local
`alpha = C_alpha * h_eff^2`, `C_alpha=1`, L2/BrokenL2 SIPG with exact DG faces,
RK3SSP, dt=0.001, final time=0.08. Every run takes 80 accepted steps,
240 stage RHS/closure evaluations and one final output closure; none rejects
work or falls back. No stage extrapolation, reduced closure frequency or
continuum-model change is used.

Three fresh processes per selected mode, with a populated CUDA runtime cache
and one exclusively leased GPU. Dense modes rotated; the final ordinary
tolerance calibration ran afterward. These are the strongest tested controls
passing the common error gate:

| Current-equation solver/preconditioner | Median process seconds [range] | Total Krylov iterations | Median preconditioner setup seconds |
| --- | ---: | ---: | ---: |
| Ordinary FP64 CG/Chebyshev, rtol=1e-3 | 8.833 [8.681, 9.234] | 14,388 | 0.576 |
| FGMRES + retained FP64 inverse, rtol=1e-6 | 2.570 [2.570, 2.619] | 651 | 0.299 |
| FGMRES + retained FP32 inverse, rtol=1e-6 | 2.570 [2.469, 2.670] | 652 | 0.299 |
| FGMRES + retained TF32 inverse, rtol=1e-6 | 2.770 [2.669, 2.921] | 964 | 0.303 |

The FP64 inverse is **3.44×** faster than the tolerance-relaxed ordinary
control. FP32 and FP64 timing ranges overlap; no precision speedup is resolved.
The Python timeout wait can add up to approximately 50 ms of process-completion
observation overhead, another reason not to interpret their small difference.
TF32 adds correction iterations and is slower. These comparisons include
process initialization, current coefficients and faces, dense construction,
host inversion, allocation/upload, packing/conversions, synchronization,
all Krylov corrections, transport, advancement, scans and saved output.
No failed-to-horizon candidate enters the timing table.

All four modes meet a common measured gate against an FP64 dt=0.00025,
FGMRES rtol=1e-11 reference: maximum pointwise error over density, momentum
and energy below 1e-8, and sigma relative RMS below 1e-6. Actual maxima are
3.20e-9 and 3.88e-7, respectively (dense sigma errors are about 9.66e-8).
The independent dt=0.0005 versus dt=0.00025
self-differences are 3.54e-10 and 1.07e-8. These are matching VTK point-sample
norms at the same spatial discretization, not continuum spatial-error estimates
or quadrature-weighted L2 norms. All five saved fields from all 12 p3 timing
runs were checked; final FP64 closure residual certification and
finite/admissibility checks also pass.

The tolerance sweep matters. The first repeated comparison used CG rtol=1e-6
and dense FGMRES rtol=1e-10, giving medians 14.045/2.720/2.770/3.021 s and an
apparent 5.16× algorithmic gain. Relaxing the ordinary solve reduced that
claim. CG rtol=1e-4 also passes at 10.887 s; rtol=1e-2 reaches the horizon
in 7.931 s but fails the sigma bound with relative RMS error 3.17e-6, despite
passing its configured residual certificate. It is excluded as a winner.
This is a measured tolerance screen, not proof of globally optimal tuning.

The mechanism is fewer expensive global operator applications. Median reported
IGR time falls from 7.154 s to 0.832 s with the FP64 inverse, including setup;
the DGSEM transport operator contributes about 0.18 s in both. Thus a faster
volume contraction alone has little leverage on the initial complete solve.
Those component timers explain attribution; process wall time is the outcome.

The new opt-in `DG_IGR_IGR_DENSE_INVERSE=fp64|fp32|tf32` samples the first
current full operator, forms its inverse in FP64 on the host, and reuses
that inverse **only as a preconditioner**. Every subsequent stage still rebuilds
and solves its current `A(U) sigma = b(U)` in FP64. FGMRES corrects preconditioner
rounding and coefficient drift. This is an algebraic solve change, not lagging
the physical closure. The default solver remains unchanged.

FP64 uses ordinary `Dgemv`; FP32 uses ordinary `Sgemv` plus conversions. TF32
uses a GEMM with 16 repeated RHS columns: at n=576, 663,552 useful matvec FLOPs
become at least 10,616,832 explicit GEMM FLOPs per application, before cuBLAS's
internal tile padding. Allocated device arrays are 2.654/1.332/1.401 MB for
FP64/FP32/TF32; these counts exclude library workspace and transient host setup.
NCU confirms actual TF32 tensor execution: the selected cuBLAS
`cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4` kernel executes 20,736
tensor-pipe instructions. Its profiled 5.312 us duration is not a complete
preconditioner or application timing.

The larger-system screen exposes setup cost. A second, single-run p2/N12
screen has 1,296 closure unknowns and the same dt/horizon. After tolerance
calibration, CG/Chebyshev at rtol=1e-4 takes 9.281 s versus 6.679 s for
retained FP64/FGMRES at rtol=1e-6: **1.39×**. Both pass the gate, with maximum
conservative error 3.21e-9 and sigma relative RMS 1.04e-7. CG rtol=1e-3
finishes in 8.833 s but fails sigma accuracy at 1.68e-6. Dense setup costs
about 2.2 s and FP64 matrix storage is 13.44 MB. Reference self-differences
are 3.54e-10 and 7.76e-9. Earlier stricter FP64/FP32/TF32 screens are retained
in the data. This is a corroborating screen, not a repeated scaling curve or
a p2-versus-p3 accuracy comparison: order, mesh and local alpha also change.

A p4/N6 short horizon also passes: n=900 padded to 912 in TF32, final t=0.01,
maximum conservative difference from the strict ordinary FP64 solve 1.34e-15,
sigma difference 5.55e-17. An earlier FP32 implementation binary agrees
similarly; its distinct binary identity is retained in the data. Neither this
parity result nor endpoint certification establishes shock robustness.

The implementation deliberately admits only CUDA, one rank and n<=2048.
Setup requires n current-operator applications plus O(n^3) inversion; storage
and inverse application are O(n^2). No inverse refresh, compression, scaling
strategy or large-mesh qualification was added. Mixed precision requires
FGMRES; mode/dimension changes and unsupported cases fail explicitly.

## Transport tensor screen and rejected approaches

The test-only probe reuses the application's Euler contravariant flux and LGL
basis. It compares the same 2D central-volume operator against a long-double
reference, using a prescribed nonzero sigma. Orders p=1,3,4,7,15 and 1,025
elements exercise small tiles, nonmultiple padding and higher-order arithmetic.
It compares existing/pair-form FP64, cooperative FP64/FP32, precombined-matrix
FP64/FP32, cuBLAS factorized lines, redundant dense 2D matrices, and fused
TF32/FP16 with or without three-product compensation. The fused forms pack and
convert on chip; the library forms charge their pack and unpack kernels.

Three-run median full GPU operator times, 200 iterations per sample:

| Formulation | p3 time / relative operator error | p15 time / relative operator error |
| --- | ---: | ---: |
| Precombined matrix, ordinary FP32 | **6.292 us / 6.03e-8** | **28.693 us / 2.06e-7** |
| Fused TF32, three products | 11.248 us / 1.19e-7 | 43.992 us / 4.27e-7 |
| Fused FP16, three products | 14.101 us / 1.24e-7 | 49.236 us / 3.25e-7 |
| Redundant dense 2D TF32, three products | 31.577 us / 1.46e-7 | 121.225 us / 5.55e-7 |

Single TF32 at p15 takes 26.406 us but has relative error 1.38e-3, versus
2.06e-7 for ordinary FP32. It is not a matched-quality win. Three-product
compensation computes high-high, high-low and low-high products with FP32
accumulation; it improves error but costs three matrix products. The FP16
path includes range checks, has no scaling/fallback, and makes no claim for
unbounded coefficients or states.

The fair ordinary control precombines the same linear operator as the tensor
variants: off-diagonal `A_ij=-Dsplit_ij/2`, diagonal equal to the sum of those
off-diagonal entries, with quadrature weights applied afterward. This is
algebraically equivalent to the central pair contraction. An initial apparent
tensor gain disappeared when ordinary CUDA received that same transformation.

Useful application work is 1.050 million FLOPs at p3 and 67.174 million at p15.
Both fused corrected variants execute 48.14×/3× that explicit arithmetic;
dense corrected GEMM executes 12×/48×. Dense scratch is 1.32/23.09 MB. Its
O((p+1)^4) 2D application received a fair comparison with the O((p+1)^3)
factorized forms and lost. Static setup, first-use warmup, scratch and phase
measurements are recorded separately; independent phase timings are not
summed. Steady kernel timings exclude reference validation and upstream flux
construction. They are mechanism evidence, not complete DG+IGR speedups.

NCU also confirms tensor instructions in the fused TF32 probe. Compensated
tensor forms lose to ordinary FP32 across the tested orders, so they remain
test-only. The ordinary precombined FP32 operator is a useful control, but
its use in production transport would still require complete solution-error
and stability qualification.

An exact diagonal inverse-mass fast path is integrated into production:
one stored coefficient/multiply per scalar when every off-diagonal entry is
exactly zero, preserving the arbitrary dense fallback. Host and CUDA parity
tests pass, including genuinely dense controls; final application fields are
bit-identical to the old mass path. An actual diagonal production launch was
profiled. The short full-process comparison was 3.1207 versus 3.1201 s, which
supports no material time-to-solution claim.

## What survives from earlier work, and deployment limits

The existing HDG tensor engine remains useful. A bounded p3/N8 IGR rerun with
common-fast trace policy, dt=1e-5 and five measured accepted steps gives
ordinary/tensor medians 0.40476/0.06721 s, with both at 64 Krylov iterations
and true relative residual around 8e-13. Initialization is 0.378/0.870 s.
This verifies a roughly 6× engine-package step benefit at this size, not the
older N52 ratio, an isolated MMA gain or a matched-error full-horizon result.
Historical RODAS3 and extrapolation findings have different accuracy and
coupling qualifications; they are not interchangeable baselines here.

The RTX 5080 dense TF32 peak is only 56.3 TFLOP/s, equal to its ordinary FP32
peak; dense FP16/BF16 with FP32 accumulation is 112.6 TFLOP/s. A three-product
TF32 representation has at most one third of that TF32 rate per useful
product, before padding, conversion and data movement. Using sparse marketing
rates or FP16-accumulation rates would overestimate these experiments.
Source: [NVIDIA RTX Blackwell whitepaper, p49](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).

The target hardware balance is substantially different. H100 SXM/H200 list
989 TFLOP/s TF32 and 1,979 TFLOP/s FP16/BF16 **with sparsity**: corresponding
dense rates are approximately 494.5 and 989.5, versus 67 ordinary FP32 and
67 FP64 tensor TFLOP/s. HGX B200's technical brief lists dense TF32 1.1 PFLOP/s,
FP16/BF16 2.2 PFLOP/s and ordinary FP32 75 TFLOP/s. These are potential compute
ceilings, not forecasts of application speedup. No H100/H200/B200 or remote
measurement was made. Sources: [H100](https://www.nvidia.com/en-us/data-center/h100/),
[H200](https://www.nvidia.com/en-us/data-center/h200/),
[B200 technical brief, p25](https://dam-cdn.nvd.orangelogic.com/AssetLink/gl2l4l4812s5fw0p614s6i8bv6mi3vx5.pdf).

**Next question:** can batched local or multilevel inverse representations
retain the observed reduction in current-equation iterations while avoiding
global cubic setup and quadratic storage? Real multiple RHS columns, reuse
and deployment-GPU dense throughput could make tensor hardware relevant
there. Coefficient drift, shocks and longer horizons must test convergence
and refresh costs; the present evidence does not settle them.

## Reproduction and validation

The canonical build used MFEM `d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec`,
CUDA 13.3, native application architecture 120 and MFEM `80-virtual`, OMP=1,
RTX 5080 16 GiB, driver 610.43.02. MFEM LAPACK is disabled; its pivoted host
elimination builds the inverse. Cold first-use processes were retained in the
data but excluded from the cached-runtime table: the initial ordinary run
took 14.55 s versus 3.12 s warm at t=0.01, and the first new dense FP64 binary
took 10.64 s. Those JIT/startup differences are not algorithmic speedups.

From the repository root, using already installed canonical dependencies:

```bash
source /home/jarias9/spack/share/spack/setup-env.sh
flock -n /tmp/dg-igr-gpu0.lock ./build.sh --gpu --cuda-arch 120 \
  --hdg-runtime --no-clean --no-deps -j 16 \
  --test-regex '^(dg_igr_dgsem_operator_tests|dg_igr_dgsem_operator_cuda|dg_igr_dgsem_igr_tests|dg_igr_igr_dense_inverse_cuda)$'
```

All four tests pass. The inverse test covers n=33 nonsymmetric packing,
transpose, FP64/FP32/TF32, retaining an inverse after changing the operator,
and FGMRES correction against an independent long-double reference. The
horizon runner also rejected a negative control in which the requested dense
mode was absent from the executable. Independent source and evidence reviews
found no blocking flaw in the tested configuration.

Set `BUILD` to the build directory printed by that command. Its measured value
was `/home/jarias9/.cache/dg-igr-build/worktrees/wt-1f1e8ddf9e6432a2/cuda-sm120/app/build`.
Use fresh output directories, and run GPU commands exclusively:

```bash
APP="$BUILD/examples/dgsem_smooth_advection_periodic/dgsem_smooth_advection_periodic"
RUNROOT=$(mktemp -d /tmp/dg-igr-precision.XXXXXX)
flock -n /tmp/dg-igr-gpu0.lock python3 scripts/perf/dgsem_precision_horizon.py \
  --executable "$APP" --output-dir "$RUNROOT/ordinary" \
  --final-time .08 --linear-rtol 1e-3
flock -n /tmp/dg-igr-gpu0.lock python3 scripts/perf/dgsem_precision_horizon.py \
  --executable "$APP" --output-dir "$RUNROOT/fp64" \
  --final-time .08 --linear-rtol 1e-6 --linear-solver fgmres --dense-inverse fp64
flock -n /tmp/dg-igr-gpu0.lock python3 scripts/perf/dgsem_precision_horizon.py \
  --executable "$APP" --output-dir "$RUNROOT/reference-quarter" \
  --final-time .08 --dt .00025 --linear-rtol 1e-11 \
  --linear-solver fgmres --dense-inverse fp64
flock -n /tmp/dg-igr-gpu0.lock \
  "$BUILD/tests/hdg/runtime/cuda/dg_igr_dgsem_tensor_volume_benchmark" \
  --elements 1025 --iterations 200 > "$RUNROOT/probe.jsonl"
```

Replace dense mode with `fp32` or `tf32`; repeat the reference at dt=.0005.
The initial stricter-tolerance three-process rotation was `[ordinary,fp64,fp32,tf32]`,
`[fp32,tf32,ordinary,fp64]`, `[tf32,fp64,fp32,ordinary]`. The final dense
rtol=1e-6 rotation was `[fp64,fp32,tf32]`, `[tf32,fp64,fp32]`,
`[fp32,tf32,fp64]`, followed by the ordinary tolerance sweep/repeats.
All use the same application binary; final calibration ran with only research
documentation changes in the checkout. Add `--order 2
--refine 1` for N12 or `--order 4 --final-time .01` for the padded p4 check.
The runner retains exact expanded arguments and validates horizon and closure;
field error is a separate required check:

```bash
python3 tests/scripts/compare_vtk_pvd_fields.py \
  --a-pvd "$REFERENCE_PVD" --b-pvd "$CANDIDATE_PVD" \
  --arrays density momentum energy sigma pressure \
  --output-json "$RUNROOT/comparison.json" --output-md "$RUNROOT/comparison.md"
```

Resolve the two PVD paths under their run directories. Require every listed
field, the final requested timestamp, and the stated error bounds; the
comparison utility alone does not enforce them. The recorded JSON includes
the exact checks' values. Profiler evidence used NCU metrics
`sm__inst_executed_pipe_tensor.sum,gpu__time_duration.sum`, demangled kernel
filters `regex:Fused` or `regex:.*gemm.*`, and bounded launch counts 2 or 1.
