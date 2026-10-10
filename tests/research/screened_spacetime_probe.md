# Screen-aware space-time Schwarz probe

`screened_spacetime_probe.cpp` is a CPU numerical/work probe. It does not
execute tensor instructions or add a production integrator. Its spatial
problem is periodic, pseudo-one-dimensional full-energy Euler (four conserved
components, fields constant in the second direction), coupled to instantaneous
IGR. It shares the application LGL basis, MFEM Euler flux, equation of state,
IGR source, sigma flux, and frozen-sigma admission algebra.

The Euler discretization is the strong DGSEM residual with a unique Rusanov
flux at each face. The scalar closure is a continuous nodal LGL discretization
with lumped mass, using the same weak equation

\[
 (\rho^{-1}\sigma,v)+(\alpha\rho^{-1}\partial_x\sigma,\partial_xv)
 = (\alpha q(\nabla u),v).
\]

This reduced spatial formulation differs from the complete application's
two-dimensional, generally overintegrated H1 or SIPG L2 closure. The probe uses
broken conservative gradients to construct the shared quadratic source. It
does not add a jump reconstruction or claim shock convergence.

The time method has two Gauss nodes. Each overlapping patch solves a local
space-time collocation predictor by Picard sweeps, holding the exterior Euler
and scalar temporal waveforms from the previous outer Schwarz sweep. A factor
of the patch's accepted-state closure matrix is reused for two genuine temporal
right-hand sides. Defect correction uses the full current matrix: density
changes in **both reaction and diffusion** are included. After each outer sweep,
only the center element's prediction is retained. Each continuous scalar node
has one owner. Rebuilt global current-closure and collocation residuals are
checked before accepting the predictor. These checks and their communication
requirements are charged.

The corrector gathers the retained predictions and computes each physical face
flux once per temporal node, then applies that same flux to its two cells.
Conservation therefore does not depend on intermediate overlapping-patch flux
agreement. An exact global closure is computed at every accepted endpoint and
charged. A failed attempt is not committed to the state or prediction history.

This is **space-time Schwarz collocation**, not classic element-volume-only
ADER. It has actual local space-time work but retains outer coupling because
the elliptic closure is instantaneous. Constant initial waveforms can leave an
error proportional to `dt * contraction^outer_sweeps`; a fixed sweep count need
not retain fourth-order convergence. Cubic prediction from four accepted
endpoints is exposed separately for sigma and the conservative state. Its first
three RK4 startup steps, exact endpoint closures, and history work are charged.
Nonuniform endpoint times are used, including a shortened final timestep.

Use `--fixed-outer 1` or `2` only as an explicit accuracy/stability ablation.
Such a row may reach the horizon but report `certified=false`. Its reported
closure/collocation defects must not be hidden when comparing it with RK.
The adaptive sweep mode (`--fixed-outer 0`) fails the attempt if it cannot meet
the original equations' tolerances within `--outer-max` sweeps.

An additional `--inner 0` ablation requires fixed outer sweeps and joint cubic
history. It keeps the extrapolated conservative temporal states, corrects their
current patch closures, and integrates the resulting unique fluxes with two-node
Gauss quadrature. This is an explicit history-based general-linear method,
**not a collocation solve or ADER predictor**. Its collocation defect is reported
only as a diagnostic, not used as a target equation; the current closure gate
still applies. Four-point state extrapolation and Gauss quadrature suggest
fourth-order smooth accuracy, but stability and common-horizon error must be
measured. For a linear ODE without closure this construction reduces to the
fourth-order Adams-Bashforth update; it does not inherit RK4 or implicit Gauss
stability.

After a canonical build provides the executable, useful short screens are:

```bash
PROBE --self-test
PROBE --method both --elements 12 --order 3 --halo 1 --alpha .0025 \
  --dt .002 --final-time .02 --reference-dt .00003125
PROBE --method patch --fixed-outer 1 --sigma-predictor history3 \
  --state-predictor history3 --dt .002 --final-time .02 \
  --reference-dt .00003125
PROBE --case compression --method both --dt .001 --final-time .02
```

Repeat a common horizon with halved timesteps, halve the RK reference timestep,
and vary halo/physical alpha. Compare errors, completed horizons, equation
defects, conservation, and all work counts. `wall_seconds` includes predictor
setup, allocations, coefficient rebuilding, corrections, certificates, startup,
and final closure; the independent RK reference has a separate time. Dense
factor/solve FLOPs are estimates for those operations alone. Patch factor
storage is reported separately from its retained source matrix; temporary
fields, current matrices, and global reference storage are additional. No CPU
timing or work estimate establishes a GPU/application speedup.

## Joint chord-Newton screen

`--method rk4-joint` compares RK4 with a materially different solution of the
same two-node Gauss equations. It treats both conservative stage states and
both scalar stage responses as unknowns. Each nonlinear residual constructs
the **current** reaction, diffusion, quadratic source and Euler flux; it does
not call an elliptic solve inside the stage iteration. A finite-difference
Jacobian is factored once and reused as a chord-Newton approximation. The
current closure and collocation residuals must both pass before accepting a
step. The accepted endpoint is independently reclosed and checked.

`--joint-block-elements 0` factors the complete small Jacobian as an oracle.
Positive values retain only contiguous element-block principal Jacobians,
with the two stages, all four Euler components and owned scalar nodes in each
block. This is a nonoverlapping block-Jacobi nonlinear correction, not exact
nonlinear elimination or a converged independent-element method. Current
residual evaluations retain the interblock flux and elliptic coupling.
`--joint-rebuild 0` retains these factors for the entire run; a positive value
rebuilds after that many accepted joint steps. A changed timestep rebuilds the
factors. A backtracking line search checks current equations and physical
admission; failed trial evaluations count toward the total work. Failure to
converge ends the candidate at its last accepted horizon. There is no hidden
RK fallback or timestep retry.

The current finite-difference constructor evaluates the complete residual for
every block column. All of that coefficient and flux work is charged; it is a
small research implementation, not a claim of an optimal sparse Jacobian
constructor. `joint_jacobian_columns`, `joint_factors`, `joint_iterations`,
`joint_residual_calls`, `joint_line_searches`, and rejected step counts expose
its mechanism. Existing `patch_setup_seconds` and retained-factor byte fields
include the joint factors too, for compatibility with the earlier JSON schema.

`--alpha-h-power s` sets `alpha=h^2/(p+1)^s`. The default remains an explicitly
specified physical `--alpha`; it does not secretly depend on polynomial order.
Comparisons between temporal methods at one alpha solve the same semidiscrete
problem. Comparisons across these alpha values change the regularization and
must not be reported as matched-physics time-to-solution improvements.

```bash
PROBE --method rk4-joint --elements 8 --order 2 --case compression \
  --dt .004 --final-time .04 --reference-dt .00003125 \
  --joint-block-elements 2 --joint-rebuild 0 --alpha-h-power 2
```

`--joint-block-offset 1` rotates the same contiguous joint blocks by one
periodic element. This changes only the nonlinear preconditioner partition;
it preserves the DG mesh, initial condition, alpha and target equations. For
eight elements and two-element blocks, the compression fronts at x=0 and x=.5
lie at block cuts with offset zero and inside blocks with offset one. This
is a same-physics test of source proximity to a block boundary.

## Analytic two-derivative factor reuse

`--method rk4-tdrk4` compares RK4 with a two-stage fourth-order two-derivative
method. It differentiates the current constraint analytically:
`A dot_sigma = dot_b - dot_A sigma`, including density changes in both the
reaction and diffusion. The current factor supplies sigma and its physical-time
tangent; no dense joint Jacobian or nonlinear stage iteration is used. Both
Rusanov flux and dissipation-speed derivatives are included. Exact max/abs ties
use forward directional derivatives, which do not imply smooth fourth-order
accuracy through a wave-speed switch.

The accepted endpoint factor is retained for the next step. A completed run
uses `1+2*steps` current matrix constructions/factors and `2*steps` additional
tangent RHS solves, versus RK4's `1+4*steps` current factors. The total scalar
RHS count remains the same. Tangent source, flux and face work have separate
JSON counters; `reuse_setup_seconds` includes current factor construction and
its sigma solve. Endpoint closure is directly solved and physical admission is
checked. Independent self-tests audit analytic closure/flux tangents, a
nonzero-jump speed switch, conservation, factor counts and smooth order.

```bash
"$PROBE" --method rk4-tdrk4 --elements 12 --order 3 --halo 0 \
  --case compression --alpha .0025 --dt .001 --final-time .2 \
  --reference-dt .00003125
```

The two-derivative method is neither the previous Gauss predictor nor an ADER
implementation. Its linear stability polynomial equals RK4's; the hypothesis
is cheaper current coefficient/factor construction, not a larger linear CFL.

`--speed-smoothing epsilon` is a separate numerical-flux control. Positive
epsilon replaces absolute velocity and the face maximum with smooth upper
envelopes. The face dissipation remains no smaller than the original bound,
and its excess is at most `1.5*epsilon`. This changes DG numerical dissipation;
it is not an algebraic optimization. For every positive-epsilon run, the
executable also computes an unsmoothed reference and reports
`state_error_to_unsmoothed_reference` and
`reference_flux_change_relative_l2`, so improved temporal order cannot conceal
the changed spatial solution. Those independent reference costs are excluded
from candidate timing and reported separately.

The calibrated analytic-reuse study is documented in
[`igr_spacetime_reuse_findings.md`](../../docs/research/igr_spacetime_reuse_findings.md),
with all successful, failed and less accurate rows in its accompanying JSON.
