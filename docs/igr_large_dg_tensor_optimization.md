# Large single-GPU DG tensor optimization

This experiment optimizes the application on one fixed large smooth DG case.
It is not a mesh-convergence or maximum-capacity study. Small CUDA fixtures
establish correctness; their timings are not HPC performance evidence.

## Workload and comparisons

DeltaAI, one GH200, account `bdiy-dtai-gh`, partition `ghx4`, `projects`
constraint. The fixed periodic unit-square mesh has 512 by 512 elements at p5:
9,437,184 scalar closure DOFs and 37,748,736 conserved-state DOFs. Use the
smooth active IGR wave, RK3 SSP, dt=1e-5, `c_alpha=1`, local alpha length
(alpha=3.814697265625e-6), SIPG exact faces, and the current FP64 operator
for solve acceptance. No positivity limiter or shock problem is involved.
All variants use the same mesh and physics. FP64, FP32 and TF32 controls
separate algorithmic changes from arithmetic precision.

Native initial/final conservative-state exports use round-trip precision and
explicit host synchronization. A streaming comparison uses p5 Gauss weights
for conservative-field L2 differences and integrals; Linf and positivity are
evaluated at saved nodes. Existing acceptance budgets are normalized L2<=1e-3,
Linf<=1e-2, excess invariant drift<=1e-6, and native conservation<=5e-10.
Agreement with the same-discretization FP64 solution is not an analytic PDE
error estimate. Output runs and profiler captures are separate from timing.

## Literature and implementation choices

Bryngelson, *Tensor-cores accelerate compact finite-difference solves and CFD*,
supplied manuscript: a fixed line inverse is reused across many real RHSs;
composing fixed linear stages can remove intermediate work. Reduced-precision
tensor throughput, rather than dense inversion alone, produces the reported
benefit. Its square, well-conditioned compact-FD inverse and inexpensive banded
residual differ from variable-coefficient DG. We retain FP64 state and nonlinear
flux, and measure complete solves including convergence and setup. The supplied
manuscript's tensor arithmetic and state-storage precision are distinct axes.

[Cui, *Acceleration of Tensor-Product Operations with Tensor Cores*](https://arxiv.org/abs/2407.09621)
uses matrix-free tensor products, bank-aware shared-memory access, and mixed
precision multigrid. Its constant-coefficient Cartesian Kronecker structure is
more restrictive than the current IGR operator. The
[multilevel DG companion](https://arxiv.org/abs/2405.18982) additionally motivates
reducing residual/smoother data motion. We borrow these implementation ideas,
without equating our modal restricted-overlap smoother with its patch scheme.

[Tu et al., *Accelerating High-Order Finite Element Simulations at Extreme Scale
with FP64 Tensor Cores*](https://arxiv.org/abs/2603.09038) combines partial
assembly, sum factorization, fusion and shared-memory layouts. Its arithmetic
is FP64 DMMA, not TF32. It motivates reducing stored operators and intermediates
and retaining matched SIMT controls, rather than predicting a speedup here.

## Candidates

`IGRLAB_IGR_TENSOR_MG_OPERATOR=shared` explicitly replaces the sampled frozen
preconditioner matrix by its translation average. On the admitted connected
periodic Cartesian topology, this is an average of permutation congruences of
the SPD screened SIPG matrix. It remains SPD mathematically, and tensor modal
restriction commutes with translation. Variable-coefficient shared MG is an
approximate preconditioner, not an exact replacement for the current equation.
Current-coefficient FP64 outer residual certification remains authoritative.

GPU probes accumulate five shared blocks and per-element diagonal traces.
FP32 level-stencil storage changes from 31,360 bytes per element to 31,360
bytes total; other vectors and coarse CSR still scale with elements. Shared
residual application uses real neighbor vectors as GEMM columns. Pre-coarse
applications compute only retained modal rows and fuse restriction. FP64/FP32
SIMT controls and standard TF32 WMMA implement the same reference model.

`IGRLAB_SHARED_EULER_LAYOUT=tensor-lines` factors conventional DG's dense weak
divergence into directional derivative and transverse mass operations.
Setup reconstructs the original affine Gauss-Legendre matrix and verifies its
agreement. Nonlinear flux, background centering and moment correction remain
FP64. Standard WMMA consumes distinct element/field columns; FP64/FP32 SIMT
controls use the same factorization. Dense layout remains the default.

The tensor-line setup validates every element's affine geometry and nodal basis
before assembling one reference weak-divergence matrix. It avoids both dense
per-element copies (about 10.1 GiB of combined host matrix storage at n512).
The retained matrix is still assembled by the original integrator and checked
against the factorization; unsupported geometry is rejected by this opt-in.

Two common diagnostic costs are also reduced. On ARM, collocated derived-field
projection qualifies the host rounding/FMA mode and uses matching CUDA FP64
arithmetic, retaining exceptional-environment and nonfinite fallbacks. Affine
collocated conservation integration uses the appropriate GL/GLL nodal weights;
other cases retain their quadrature and gather an element only once. These are
separate from tensor-core gains and also apply to the matched FP64 control.

## Validation status

Initial source: `60711ded9eac158378bc19c6e44c3df0289ea67b`.
The n512 FP64 admission run (DeltaAI job 3399587) completed 10 steps, all 31
linear solves, and native final acceptance. Process wall time was 225.45 s;
the stepping-loop report was 48.99 s. These are one admission sample, not a
speedup or steady-state performance claim. Peak observed GPU residency was
about 41,010 MiB; the allocated device reported 97,871 MiB usable memory.

The baseline two-step Nsight Systems capture (3399622) shows substantial host
time beyond GPU kernels. Dense weak divergence and Krylov orthogonalization
are material device costs. Setup, final diagnostics, and transfer costs must
therefore accompany any kernel-level comparison.

Seven native suites passed in DeltaAI job 3399714: shared reference maps,
tensor multigrid, center maps, tensor volume, device Krylov, CUDA diagnostics,
and time-default/conservation tests. The tested application SHA256 is
`076bc60443fd858af1f52597f31de2516a6d3e749902875b0013ac48ca91048a`.
The allocator ownership issue found in the first shared coarse CSR construction
was fixed using MFEM-managed storage. ARM control tests use actual FPCR readback
because the hardware may ignore unsupported exception-enable bits.

Large-case field comparisons, repeated timings and targeted hardware profiles
are pending. No candidate speedup is claimed.
