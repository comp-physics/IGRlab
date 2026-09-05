# Four locality routes for shocked DG + instantaneous IGR

This follow-up derives and prototypes all four proposed routes on the existing
full-energy application's closure. The central distinction is between solving
element interiors locally, approximating a screened inverse over a finite
neighborhood, and changing the closure. These are different mathematical choices.
The detailed derivations are in the [face note](igr_face_locality_derivation.md),
[artificial-time note](igr_relaxation_derivation.md), and
[compact-response note](igr_compact_response_derivation.md).

## What can be predicted before implementation

For a frozen admissible Euler state, write the current closure as

\[
A\sigma=b,\qquad
A=R+K,\qquad
R\sim\rho^{-1},\quad K\sim-\nabla\cdot(\alpha\rho^{-1}\nabla).
\]

Here the discrete matrices include the current SIPG faces and the application's
source. The following spectral claims require positive reaction and a coercive
DG diffusion form; the actual small matrices are audited for these properties.
The large-case lower bounds are conditional on diffusion positivity, not a new
proof of coercivity for every curved mesh and penalty.

**Physical range.** At constant coefficients the normalized operator
\(I-\ell^2\partial_{xx}\) has Green function
\(e^{-|x|/\ell}/(2\ell)\), \(\ell=\sqrt\alpha\); the unnormalized inverse has
an additional factor ρ. Thus weak distant influence is
compatible with a globally supported exact inverse. If \(\alpha=C h^2\), the
screening length in elements is \(\sqrt C\), independently of p. This explains
why p refinement alone does not shorten the physical smoothing width.

An exact element Dirichlet-to-Neumann map gives a sharper transmission test:

\[
D_K=\frac{\ell}{\rho}
\begin{bmatrix}\coth(h/\ell)&-\operatorname{csch}(h/\ell)\\
-\operatorname{csch}(h/\ell)&\coth(h/\ell)\end{bmatrix}.
\]

The opposite-face/self-face DtN coupling ratio is \(\operatorname{sech}(h/\ell)\).
It is 0.0845, 0.648, and 0.952 for C=0.1, 1, and 10. At C=1, an element cannot
generally ignore its neighbors and still reproduce the current closure.
Variable coefficients, geometry and tangential modes require further bounds.

**Correction work.** A block-local SPD metric B gives the self-adjoint similar
operator \(B^{-1/2}AB^{-1/2}\), with spectral interval \([\mu,L]\). A degree
\(m-1\) inverse polynomial communicates new source information at most m−1
face hops when B is element-block diagonal. Ordinary Chebyshev correction has

\[
\|e_m\|_A\le2q^m\|e_0\|_A,
\qquad q=\frac{\sqrt\kappa-1}{\sqrt\kappa+1},\quad\kappa=L/\mu.
\]

This provides a cost estimate before kernel tuning. A local metric that leaves
\(\kappa\sim p^4\) costs approximately \(p^2\log(1/\varepsilon)\) stages;
one reducing it to \(p^2\) costs approximately \(p\log(1/\varepsilon)\).
Actual source spectra can make CG much faster than a uniform polynomial bound.
Polynomial approximation and inverse decay are established ideas; see
[Demko, Moss and Smith (1984)](https://www.ams.org/mcom/1984-43-168/S0025-5718-1984-0758197-9/S0025-5718-1984-0758197-9.pdf).

**Warm updates.** The exact identity

\[
r_{n+1}=b_{n+1}-A_{n+1}\sigma_n
=\Delta b-\Delta A\,\sigma_n+r_n
\]

shows why a previous closure can need few corrections. It depends on the
actual update size and chosen accuracy, not just elliptic conditioning.
For the saved cylinder state with the controlled perturbation
\(U_{\rm new}=(1+0.001\sin(3x)\cos(2y))U_{\rm old}\),
the previous accurate closure has relative current
residual \(4.74\times10^{-4}\). At tolerance \(10^{-3}\), zero correction is
appropriate; at \(10^{-8}\), substantial work remains. This perturbation is a
controlled frozen-state experiment, not a sampled application time step.

**The accuracy norm matters.** The exact closure energy error is
\(\|e_\sigma\|_A=\|r\|_{A^{-1}}\). Euler responds to closure gradients and
face fluxes, so a small sigma L2 error alone is insufficient. For a fixed state,
one can bound the transport perturbation by
\(\|\delta R_E\|\le\|\partial_\sigma R_E A^{-1/2}\|\|r\|_{A^{-1}}\).
That constant includes differentiation and traces; with sigma-dependent
dissipation it must be a derivative supremum along the closure segment.
The tested configuration uses sigma-independent dissipation. Even in constant-coefficient
Fourier analysis, \(\max_k |k|/(1+\ell^2k^2)=1/(2\ell)\): shorter screening
can increase sensitivity to closure error. A frozen bound does not replace
an advancing shock accuracy/stability test.

## An explicitly finite-range response

Set \(H=R^{-1/2}KR^{-1/2}\). The response

\[
F_T(H)=\int_0^T\frac{e^{-t}-e^{t-2T}}{(1-e^{-T})^2}
\cos(t\sqrt H)\,dt
=\frac{(1+e^{-2T})I-2e^{-T}\cos(T\sqrt H)}
{(1-e^{-T})^2(I+H)}
\]

preserves the reaction-scaled constant mode and satisfies

\[
0\preceq (I+H)F_T(H)-I
\preceq\frac{4e^{-T}}{(1-e^{-T})^2}I.
\]

The continuum wave has speed \(\sqrt\alpha\); finite artificial horizon T
therefore gives finite physical influence. T=6 gives a uniform relative energy
error below 1%, over approximately six screening lengths. It does **not** make
the current exact elliptic inverse compact. It defines a controlled approximate
response, or a changed continuum closure if selected as an exact fixed model.
No changed continuum response has been adopted for production.

There is also a genuinely local discrete evaluation. For
\(A'=R^{-1}A\), spectrum \([1,L]\), let
\(z=(\sqrt L-1)/(\sqrt L+1)\) and
\(C=((L+1)I-2A')/(L-1)\). Then

\[
P_{m-1}(A')=
\frac{(1+z^{2m})I-2z^mT_m(C)}{(1-z^m)^2A'}.
\]

The numerator vanishes at A′=0, so this is an **actual polynomial of degree
m−1**, not an instruction to apply an inverse. It is evaluated by a local
Chebyshev recurrence in [igr_polynomial_probe.py](../../tests/research/igr_polynomial_probe.py).
Its relative inverse error lies between zero and
\(4z^m/(1-z^m)^2\), and \(P_{m-1}(1)=1\). Thus finite graph range,
positive spectrum, constant response and a uniform error bound can coexist.
Those properties do not imply pointwise positivity on arbitrary DG meshes.
The scalar geometric-autocorrelation algebra is not claimed as novel.
Neither the energy bound nor positive spectrum proves pointwise admissibility,
the original geometric variational structure, or stable coupled shock evolution.

The cost is decisive: at the measured p3 reaction condition number 4438, the
uniform 0.1% bound needs about 277 stages. Explicit wave evaluation has the
same high-frequency stiffness problem. Face/interior preconditioning is a
more promising companion than simply doing this many unpreconditioned sweeps.

## Element-local interiors with coupled faces

The existing GL basis has no trace-zero nodal subset. Change coordinates to
LGL trace and bubble functions without changing the polynomial space or
quadrature. The interior block is element-block diagonal, and exact elimination
leaves

\[
S\sigma_\Gamma=g,\qquad
S=A_{\Gamma\Gamma}-A_{\Gamma I}A_{II}^{-1}A_{I\Gamma}.
\]

Interior solves are then local; face coupling remains. SIPG normal-derivative
terms create some two-hop Schur fill. In 2D the retained unknown count is 4p
per element, versus \((p+1)^2\); at p1 there are no bubbles to eliminate.
This is exact algebraic equivalence when the Schur solve is accurate.

A new hybridized DG discretization can express the same architectural idea
directly. On each element, solve
\(\rho^{-1}\sigma+\nabla\cdot q=f\),
\(q=-\alpha\rho^{-1}\nabla\sigma\), given a face trace \(\widehat\sigma\),
with numerical normal flux
\(\widehat q\cdot n=q\cdot n+\tau(\sigma-\widehat\sigma)\).
Enforce the sum of the two outward numerical fluxes to be zero on each interior
face. Element fields are local functions of the traces; the trace equations
retain transmission. Function spaces, weak forms and stabilization must be
chosen consistently; this schematic system is not an assertion of equivalence
to the current SIPG discretization. [Unified hybridization](https://doi.org/10.1137/070706616)
provides an established mathematical framework.

For deliberately weakened coupling, the better-behaved family is

\[
A_\theta=A_0+\theta(A-A_0),\qquad A_0=R+K_{\rm volume},\quad0\le\theta\le1.
\]

It is SPD by convexity and preserves \(A_\theta\mathbf1=R\mathbf1\).
At θ=0 it gives independent element Neumann problems. At θ=1 it is the current
closure. Values in between change the discrete regularization: scaling the
SIPG consistency terms generally loses consistency with the same fixed-alpha
continuum operator. A mesh-dependent alpha tending to zero is a separate limit,
not a proof that the finite-grid shock is unchanged. Making θ small does not
make the remaining inverse strictly compact.

There is an a priori modal sensitivity formula. If λ is an eigenvalue of
\(A_0^{-1/2}AA_0^{-1/2}\), changing A to Aθ multiplies its response by
\(\lambda/[1+\theta(\lambda-1)]\) relative to the original response.
The relative change is
\((1-\theta)(\lambda-1)/[1+\theta(\lambda-1)]\).
This predicts when apparently modest weakening can substantially change a
mode; the actual source determines how much that mode contributes.

This local-interior idea also applies to joint Euler/IGR space-time Newton
blocks, potentially producing useful dense multi-field, multi-stage work.
That extension needs actual nonlinear and interface solves; the present scalar
condensation screen does not establish an ADER engine.

## Frozen production-operator screens

The shared C++ test export calls the existing IGR solver, full-energy physics,
BrokenDG gradient, local alpha and exact faces. It checks sampled graph actions
against the current operator. It also exports physical mass, reaction and
volume blocks; independently, A·1=R·1 and volume·1=R·1 agree to roundoff.
The input cases are a prescribed compression, the saved planar shock at t=6,
and the saved curved-cylinder shock at t=0.5 (1322 quadrilaterals, 5288 scalar
DOFs). Saved polynomial samples are mapped back to their original mesh cells,
with independently checked physical coordinates. Raising the planar source
from p2 to p3 preserves its represented state, but changes the closure space.

| Route | Measured result | Interpretation |
|---|---|---|
| Local inverse polynomials | Cylinder reaction Chebyshev: 184 stages versus 175 same-metric PCG iterations, true residual 1e−8; warm 112 versus 107 | Competitive work/reduction tradeoff in this case; hundreds of stages at higher p are unattractive |
| Exact interior/face elimination | On p3/p5/p7 compression, element-block CG 76/113/145 versus condensed face-node CG 17/25/33 | Condensation alone did little; grouping opposite face values removed important stiff modes |
| Artificial-time relaxation | Cylinder reaction metric: 200 sweeps versus PCG 175; 27 versus 529 unfused scalar reductions | Fewer reductions at the cost of more operator work; no standalone GPU claim |
| Compact response | T=6 spectral oracle: energy errors 0.53–0.68% on several small shock/compression matrices | Explicit mathematical control; naive local wave evaluation needs 200/438/767 stability stages at p3/p5/p7, before accuracy cost |

The same face-node approach gave 21 rather than 101 element-block CG iterations
on the actual p1 cylinder operator, and 13 rather than 60 for its controlled
warm perturbation. These counts compare at the same current residual 1e−8.
They do not compare against the stronger production preconditioners or include
an advancing flow; the application follow-up below makes that comparison.

Several losses are informative. Condensation alone did not remove the stiff
face modes. An exact Hermite value/derivative basis reduced interface fill but
increased basis conditioning and did not improve iterations. Adding the element
and face-node inverses weakened the better standalone method in every small
case. The saved planar shock needed 68 face-node iterations versus 59 with
element blocks: a good worst-case condition number is insufficient to predict
performance for one source.

One-shot moment-preserving compact patch inverses gave cylinder sigma L2 errors
of 79.5%, 27.8%, 17.3%, and 24.2% at radii 0/1/2/4. Large overlap costs and
nonmonotonic response error stopped this branch as a direct closure replacement.
The θ family is much better behaved than arbitrary off-diagonal attenuation,
but independent elements still have about 80% cylinder error. θ=0.5/0.75/0.99
gave approximately 2.04%/0.63%/0.0177% cylinder sigma L2 error. Its full Euler
accuracy and consistency have not been established.

## Complete cylinder application follow-up

The useful new implementation is an opt-in FP64 p1 vertex preconditioner in the
existing screened-patch component. It uses the production face-block sampling
and current-operator correction. Setup forms the GL/corner congruence, factors
the vertex groups and composes the basis transforms into inverse rows. One
ordinary CUDA kernel applies those rows. No continuum model, transport flux,
closure cadence or time integrator changes.

The complete test uses 1322 quadrilaterals around a cylinder, p1, full-energy Euler,
instantaneous closure at every SSPRK104 stage, local alpha=C h² with C=1,
LLF volume/Rusanov face dissipation, Zhang–Shu limiting, dt=0.005 and t=0.5.
There are 100 accepted steps, 1001 closure calls and zero rejected attempts in
every run. The control uses the previously calibrated ordinary FP64 CUDA
CG/Chebyshev degree 2, three power iterations, rtol=1e−2. RAS and vertex use
rtol=1e−3, refreshed every 100 closure solves. RAS uses FGMRES; vertex is SPD
and uses CG. Quality is checked against the same-grid FP64 CG rtol=1e−6,
dt=0.0025 reference from the prior campaign.

| Configuration | Runs | Complete process wall times, s | Median, s | Mean including every first process, s |
|---|---:|---|---:|---:|
| Ordinary CG/Chebyshev | 3 | 18.622, 16.584, 16.445 | 16.584 | 17.217 |
| Previous radius-3 RAS | 4 | 13.916, 12.884, 12.921, 12.942 | 12.931 | 13.166 |
| New vertex CG | 4 | 19.867, 12.119, 12.419, 12.285 | 12.352 | 14.172 |

These are full subprocess times, including initialization, coefficient
construction, factor setup/rebuilding, conversions, transport, faces,
advancement, certification, output and process termination. The vertex median
is 25.5% below ordinary CG and 4.5% below RAS. However, **the first observed
vertex process was slower**, and its all-run mean is worse than RAS. The
19.867-second first process had an advancement timer of only 11.175 seconds;
this difference is charged, not attributed to an unverified JIT explanation.
Later vertex advancement timers were 10.59–11.36 seconds versus RAS
12.10–12.14 seconds. Thus the new method is a useful recurring-work candidate,
with a modest complete-cost improvement over the already strong RAS method
and an unresolved first-process cost. It is not an unconditional performance
winner for a one-shot short run.

The mechanism is clear. Over the horizon, vertex CG takes 8877 iterations
versus RAS 6447 and control 24027. Despite **more iterations than RAS**, its
repeated preconditioner setup costs approximately 0.17 seconds rather than
0.87, and its solve costs approximately 1.6–1.85 seconds rather than
2.1–2.2. Current operator construction still takes approximately 6.2–6.5
seconds. The first vertex run has extra setup/initial-closure costs, retained
in the archive. Phase timings include different scopes and must not be added
to the complete wall time as though they were independent.

Vertex factors contain 20,440 entries, with maximum group dimension five;
RAS's sampled principal factors contain 11,549,280 entries, maximum dimension
112. The expanded vertex action stores 1,663,724 device bytes and executes
181,920 products per application, versus RAS's 4,430,828 bytes and 489,376
products. The unexpanded vertex inverse plus two basis changes would need
62,744 products: composition deliberately performs about 2.9 times as much
matrix arithmetic to avoid intermediate vectors and extra kernels. These are
ordinary FP64 products, with warp-reduction overhead additional; there is no
tensor throughput claim and no hidden tensor padding counted as useful work.

Against the tighter same-grid reference, the largest relative RMS error among
the four conserved fields is 2.52e−7 for vertex, 2.67e−7 for RAS and 5.52e−7
for the control. Sigma relative RMS errors are 3.66e−7, 4.01e−7 and 9.47e−7.
These are unweighted errors over identical VTK samples, including duplicate
face points; they are not volume-weighted norms or proof of spatial convergence.
All repeats of each method give identical saved numerical fields. Saved
density and reconstructed pressure remain positive, accepted-state closure
certification passes, and conservation is checked using the application's
boundary-balance telemetry. This qualifies solver accuracy on this horizon,
not all shock configurations or a fully hp-converged method.

The higher-p face-condensed method remains a CPU algebraic screen. Its small
matrices use separately charged FP64 direct references and spectral oracles.
Neither those CPU times nor the RTX 5080 result establishes H100/H200/B200
performance. The p1 vertex factors are too small and narrow to demonstrate
tensor-core value. Higher-p interior or joint space-time solves could provide
larger matrix work, but construction, extra arithmetic, precision correction
and interface iterations still require measurement at the actual precision.

## Next question

The most useful next experiment combines the successful face treatment with
local **joint Euler/IGR space-time interiors**, and tests whether it reduces
the number of current coefficient constructions and interface exchanges on a
moving shock. That is where matrix hardware could change the balance. A
face-conditioned polynomial or artificial-wave corrector is another bounded
option: the measured p3/p5/p7 stages drop to 49/67/83, but still exceed
face-PCG's 17/25/33. Changing the metric requires new bounds; the
reaction-scaled filter's exact constant-mode property and physical wave cone
do not automatically carry over to an arbitrary face preconditioner.

Locality also does not by itself solve hp shock representation. Reducing alpha
with p narrows the physical front and can worsen underresolution and source
evaluation. Front/subcell adaptation or enrichment must be tested jointly with
the closure and conservative faces. The prior representation experiments are
evidence for that separate uncertainty, not validation of an hp production
method here.

## Tested identities and reproduction

Implementation commit: `b7942c8049a78401fe75ba65fc268b762f104ceb`, tree
`1445f4acaf21efc26b44eecf86d30372fe1ea419`. Every new candidate and RAS
application measurement uses this binary/source; the first control uses the
previous implementation binary and produces identical fields and iteration
counts to the rebuilt control. Per-run binary hashes are in the
[results archive](igr_locality_four_routes_results.json). All nine frozen
operator cases were re-exported from the final committed implementation; A,
b, mass, reaction, geometry and DOF arrays are exactly equal to the inputs used
by the Python screens. Individual input/script hashes and commands are retained.

The machine is the local RTX 5080, driver 610.62, CUDA 13.3, Release, rank one,
OMP_NUM_THREADS=1, app SM120 and MFEM 80-virtual. MFEM pin is
`d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec`. No remote GPU was used.
Six focused canonical checks pass: local operator export, screened patch CUDA
(including vertex action, symmetry/SPD and retained-factor current solves),
stationary correction CUDA, DGSEM operator CUDA, DGSEM IGR and IGR tests.
An initial oracle test exposed an unsafe host write after a device read;
explicit HostWrite fixed the test, with all accuracy thresholds unchanged.
Polynomial, relaxation and compact-response independent selftests also pass.

```bash
source /home/jarias9/spack/share/spack/setup-env.sh
flock -n /tmp/dg-igr-gpu0.lock ./build.sh --gpu --cuda-arch 120 --hdg-runtime \
  --no-clean --no-deps -j 16 --test-regex \
  '^dg_igr_(igr_screened_patch_cuda|local_operator_cpu|igr_stationary_correction_cuda|dgsem_igr_tests|dgsem_operator_cuda|igr_tests)$'

# BUILD is the build directory printed by build.sh. Run GPU/MPI outside sandbox.
flock -n /tmp/dg-igr-gpu0.lock python3 scripts/perf/dgsem_precision_horizon.py \
  --executable "$BUILD/examples/supersonic_example_w_igr/supersonic_cylinder_w_igr" \
  --case supersonic-cylinder --order 1 --dt .005 --final-time .5 \
  --integrator ssprk104 --sigma-space L2 --volume-dissipation llf \
  --positivity-limiter zhangshu --diagnostic-scans off --conservation-balance \
  --fused-residual --cheb-order 2 --cheb-power-iterations 3 --timeout 90 \
  --linear-solver cg --linear-rtol 1e-3 --patch-radius 0 \
  --patch-kind vertex-p1 --patch-rebuild-interval 100 --output-dir /tmp/igr-vertex

# Shared small production-operator input for all four research scripts.
OMP_NUM_THREADS=1 "$BUILD/tests/dg_igr_local_operator_probe" \
  --nx 4 --ny 4 --order 3 --output /tmp/igr-tanh-p3.json
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_polynomial_probe.py \
  --input /tmp/igr-tanh-p3.json --output /tmp/igr-polynomial.json
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_relaxation_probe.py \
  --input /tmp/igr-tanh-p3.json --metrics reaction,element \
  --max-iterations 512 --checkpoint 8 --output /tmp/igr-relaxation.json
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_compact_response_probe.py \
  --input /tmp/igr-tanh-p3.json --output /tmp/igr-compact.json
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_face_condensation_probe.py \
  /tmp/igr-tanh-p3.json --basis both --output /tmp/igr-faces.json
```

For actual shocked inputs, first use `igr_saved_state_to_nodes.py` on the
existing application VTK and pass `--mesh` and `--nodal-state` to the C++ probe.
VTK sampling degree must be at least the known original state degree; target
order must be at least the input sampling degree. This is a bounded test export,
not a production restart or capture system. The archive contains all case
commands, node hashes, original reference identities, and a reference to the
existing archived cylinder postprocessor. Raw temporary outputs remain under
`/tmp/igr-local-four`; the bounded results are preserved in the repository.
