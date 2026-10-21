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

Validation and measurements are pending. The fixed physical comparison is
512 by 512 elements, p5 ordinary DG/GL, SSPRK3, CFL 0.3, max-length
`c_alpha=1/6`, final time 0.1, VTK every 0.02, with scans and limiters disabled.
Compare complete conserved polynomial fields and derived flow quantities at
matching physical times; report sigma differences separately. A single fixed
mesh does not establish error relative to the discretization error. Fresh
Nsight attribution and unprofiled end-to-end timing determine the sigma/flow
cost ratio and benefit.
