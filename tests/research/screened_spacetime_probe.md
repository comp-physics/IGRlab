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
