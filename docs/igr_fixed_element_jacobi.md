# Fixed element-block Jacobi closure experiment

This opt-in ordinary-DG experiment compares an approximate sigma closure with
the latest optimized, converged ordinary-DG solver. The control uses the TF32
spectral preconditioner inside FP64 FGMRES and certifies the current equation at
the requested RHS-relative tolerance of `1e-6`. The historical original GPU
solver is not the control for this experiment.

For the current stage equation `A sigma = b`, let `D` contain the complete
diagonal element blocks of `A`, including mass, volume diffusion and SIPG face
self contributions. Each sweep computes

```
r = b - A sigma
D z = r
sigma = sigma + omega z
```

The requested trial uses `omega=0.7`, 50 sweeps at initial time zero, then two
sweeps per stage, warm-started from the preceding production sigma. Element
matrices and FP64 pivoted LU factors are refreshed for every new state;
topology, maps and storage are reused. MFEM element assembly and the exact face
data supply the blocks. There is no coarse correction or Krylov fallback.

Two regular sweeps perform two current-operator actions and two block solves.
Block preparation/factorization is part of the closure cost. No tolerance test
or true-residual action is required for fixed-sweep acceptance. The existing
optional `IGRLAB_IGR_TRUE_RESIDUAL` diagnostic can measure the original current
FP64 residual, but cannot retry or replace this approximate solution. Routine
nonfinite/admissibility scans remain opt-in. LU singularity/error statuses are
checked as part of factorization.

Select the trial with these additions to the matched ordinary-DG case command:

```
--igr-solve-mode element-block-jacobi --igr-sweeps 2 \
--igr-initial-sweeps 50 --igr-jacobi-omega 0.7 \
--igr-sigma-cadence every-stage
```

The default `--igr-jacobi-blocks current` refreshes and factors the current
element diagonal each update. The separately selected
`--igr-jacobi-blocks reference` instead forms an explicit FP64 inverse of the
initial production element blocks, then reuses it:

```
sigma = sigma + omega D_reference_inverse (b_current - A_current sigma)
```

This reference-block Richardson iteration is a different approximation from
current-block Jacobi. It keeps the current FP64 equation and warm-start history;
it does not refresh the local inverse as density evolves. An initial output-only
solve or discarded prefetch cannot select the production reference matrix.
The reference mode releases EA and LU construction storage after the checked
inverse is complete. At 512² p5 it retains 2592 MiB of inverse matrices; its
matrix-only setup peak is 7776 MiB, excluding common operator/state storage,
maps, vectors and library workspaces. Reference-mode native and physical
qualification are pending; the results below describe the current-block trial.

Use CUDA, L2/BrokenL2, exact SIPG faces, and the direct identity layout. Disable
global tensor, two-level, dense and patch preconditioners for this mode. Retain
the optimized FP64 current-volume operator and the optimized Euler kernels.
The prototype admits one rank with uniform nodal quadrilateral elements and
natural scalar boundary conditions. The element-block construction does not
require Cartesian or affine cells; other retained optimizations can have their
own geometry restrictions. DGSEM, HDG, restart, constrained scalar spaces and
other sigma update cadences are outside this trial.

Initialization is a reusable stage-zero closure even when VTK is disabled.
Output-only solves do not commit production warm-start history, and a prefetched
stage-zero solution commits history only when that stage consumes it. Final
output is explicitly approximate and uncertified; it is not replaced by a
converged cleanup solve. Legacy defaults and the converged control's residual
certification are unchanged.

The fixed physical comparison is
512 by 512 elements, p5 ordinary DG/GL, SSPRK3, CFL 0.3, max-length
`c_alpha=1/6`, final time 0.1, VTK every 0.02, with scans and limiters disabled.
Compare complete conserved polynomial fields and derived flow quantities at
matching physical times; report sigma differences separately. A single fixed
mesh does not establish error relative to the discretization error. Fresh
Nsight attribution and unprofiled end-to-end timing determine the sigma/flow
cost ratio and benefit.

## Initial native validation and short comparison

Native job `3431521` passed the current-block CUDA oracle, fixed-sweep lifecycle
checks, mode/configuration checks and existing CUDA Krylov check. Tested source
`829a988ec5e82ef543592f4000b067ec21d9e701`, application SHA256
`a637586982a4e0fbb4e5b420fe6dff8c3eb4b7ce1d895d0675db4dcf4247c848`.
Skewed/curved block and current-operator comparisons were within approximately
`5.2e-16` relative error; FP64 block solves were within `5.5e-15`. The lifecycle
check proves 50 initialization sweeps, two subsequent sweeps, output-only history
invariance, and stage-zero prefetch consumption without a second solve.

Job `3431542` ran both methods sequentially on one GH200 GPU through **t=0.0025
only**, with 120 accepted steps each. Its reference is optimized source
`97ba1681ac27e07145df790d4c23ed837a21caea`, binary SHA256
`5a8f1c3e02d214b55bb857fadd8308cb2864b1e46dff5125d66046a0fcf26495`.
Exact identities, configuration and field norms are in
[the measurement record](benchmarks/dg_fixed_element_jacobi_short_measurements.json).

| Short-run quantity | Optimized converged control | Two-sweep candidate |
|---|---:|---:|
| Native elapsed time, including output | 55.703 s | 143.831 s |
| Entire application wall time | 133.726 s | 221.429 s |
| Sigma updates | 361 | 361 |
| Krylov iterations / Jacobi sweeps | 896 | 770 |
| Host block/preconditioner setup counter | 3.252 s | 86.147 s |
| Host linear solve counter | 4.637 s | 5.409 s |

The candidate is slower in this short trial. Its block setup counter averages
238.634 ms/update. These host counters are not a GPU cost partition: the LU
status synchronization can wait for preceding asynchronous assembly. A bounded
setup profile is the next step before a much longer full run.

All saved conservative values are finite; density and pressure are positive at
all 9,437,184 reconstructed native GL points. Full Q5 comparisons give maximum
conserved relative L2 difference `3.647e-8`, pressure relative L2 `2.992e-9`, and
sigma relative L2 **0.28847**. Sigma's reference L2 magnitude is only `1.185e-7`
at this early time, so the small flow difference does not establish later-time
accuracy or accuracy relative to discretization error. The t=0.1 comparison and
fresh evolved-state sigma-to-flow profile remain pending.

## Bounded setup diagnostic

Job `3431621` captured five seconds of a separate continuous candidate process,
after a nominal 140-second delay, then deliberately terminated that process.
This is not a failed physical comparison. The trace contains 19 complete
regular update scopes, each with one block assembly, one factorization and two
sweeps. Actual absolute physical-time endpoints are not encoded by these NVTX
ranges.

The recorded dominant kernels averaged **137.111 ms** for cuBLAS `getrf_panel`,
**62.872 ms** for exact face-block assembly, **29.164 ms** for MFEM diffusion EA,
and **3.381 ms** for mass EA (19 recorded occurrences each). These observations
show that optimizing only volume assembly would leave substantial LU and face
assembly costs. The host factor scope averages 236.415 ms because its final
scalar synchronization also waits for preceding assembly.

**CUPTI warned of possible missing records.** The profiler also reported that
driver 13.2 was unsupported by its bundled CUDA 12.9 tracing libraries. A complete GPU phase
partition and sigma/flow ratio are withheld; the listed times describe recorded
kernel occurrences, not an authoritative percentage of all GPU work. See the
[profile record](benchmarks/dg_fixed_element_jacobi_setup_profile.json).

An important method choice remains: the existing two-level path retains
reference blocks, whereas this first trial deliberately refreshes exact current
blocks. Reusing reference blocks would remove recurring assembly/factorization
but changes the approximate iteration and needs a separate flow comparison.
Optimizing current blocks preserves this method but must address LU and face
assembly as well as volume EA. No full t=0.1 candidate result is claimed.
