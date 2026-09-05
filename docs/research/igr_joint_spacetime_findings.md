# Joint current-closure space-time screen

This experiment replaces the previous repeated local Picard/closure iteration
with a coupled chord-Newton solve. It is a bounded numerical and work screen in
`tests/research/screened_spacetime_probe.cpp`; it is not a production integrator,
a complete ADER implementation, or a tensor-core benchmark.

## Hypothesis and equations

For two Gauss stages, write the semidiscrete equations as

\[
 G_{U,i}=U_i-U_n-\Delta t\sum_j Q_{ij}F(U_j,\sigma_j)=0,\qquad
 G_{\sigma,i}=A(U_i)\sigma_i-b(U_i)=0.
\]

The prior scheme alternated closure correction and Euler Picard iteration.
The new scheme solves for all stage states and scalar responses jointly. A
retained Jacobian supplies corrections to the **current** equations. Thus
retaining a factor is a solver approximation, not a lagged closure model.
The scalar blocks are not solved separately during residual evaluation.

The Jacobian has the schematic form

\[
 J=\begin{bmatrix}
 I-\Delta t QF_U&-\Delta t QF_\sigma\\
 C_U&A
 \end{bmatrix},\qquad
 C_U=(\partial_U A)\sigma-\partial_U b.
\]

Eliminating sigma gives the closed Euler-stage Jacobian
\(I-\Delta t Q(F_U-F_\sigma A^{-1}C_U)\). A useful joint approximation must
represent this feedback; reducing the count of standalone elliptic solves by
itself does not guarantee less work. Density changes in both reaction and
diffusion and derivatives of the nonlinear source enter the finite-difference
Jacobian and every current residual in this prototype.

For a short-screened closure, the scalar inverse couples mainly over
\(\sqrt{\alpha}\), whereas Euler information in a stage couples over a distance
of order \(a\Delta t\), where \(a\) is a characteristic speed. Choosing
\(\alpha=h^2/(p+1)^s\) reduces the former relative to an element to
\((p+1)^{-s/2}\), but does not reduce the latter. Independent element stage
solves therefore do not follow merely from smaller alpha. The numerical trace
and differentiation operators introduce additional high-order stiffness.

## Implementation and scope

The full-Jacobian factor is a small global oracle. Disjoint contiguous blocks
retain all Euler variables, owned continuous sigma nodes, and both stages for
the selected elements; their factors provide a block-Jacobi chord correction.
This tests whether local joint matrices can converge with the actual interface
coupling left in the residual. It is not exact interior condensation.

Factors can be retained across accepted steps. Backtracking uses current
closure/collocation residuals and rejects inadmissible Euler or frozen-sigma
flux states. Failure to converge stops the candidate at its last accepted
horizon. There are no unreported retries or RK fallbacks. Every accepted step
uses unique conservative face fluxes and independently recomputes its endpoint
closure. A converged stage solve has the original semidiscrete instantaneous
closure at the two Gauss nodes to the stated tolerance; no accuracy claim is
made between those nodes.

The finite-difference constructor evaluates the entire residual for each
column. That residual explicitly assembles the reduced dense current scalar
matrix before multiplying sigma; a fused matrix-free evaluation is another
untested implementation opportunity. The construction is intentionally simple
and fully charged, so timing
is not an estimate of an optimal sparse or analytic Jacobian implementation.
The screen includes factor construction, all nonlinear coefficient work,
line searches, failed attempts, exact endpoint closures and startup if history
seeding is requested. Dense factor and solve counts omit all other arithmetic;
coefficient-element counters include attempted evaluations. Timers cover each
run from its initial closure through the final accepted closure, including all
solver setup and failed work. Shared basis construction, process/MPI startup,
and the independently timed reference are outside that timer. CPU costs do
not predict GPU speedup.

The spatial problem remains periodic pseudo-one-dimensional full-energy Euler,
with strong DGSEM/Rusanov transport and continuous lumped LGL sigma. Source
construction uses broken conservative gradients and the shared production IGR
algebra. It does not reconstruct a distributional source across jumps or
establish shock convergence. Temporal methods at one physical alpha target the
same semidiscrete equations; changing alpha changes the regularization.

## First screening results

The first screen uses eight elements, p2, periodic compression, `dt=.004`,
`t_final=.04`, current closure tolerance `1e-10`, and collocation tolerance
`1e-11`. The FP64 RK4 reference uses `dt=.0000625` on the identical spatial
problem and alpha. The fronts initially lie at element faces. The alpha
exponents 0, 2 and 4 give `ell/h=1`, `1/3` and `1/9`, respectively.

| Alpha exponent | Joint factor | Completed | Joint iterations | Relative state error | Coefficient-element evaluations | CPU time |
|---|---|---:|---:|---:|---:|---:|
| 0 | full oracle | yes | 112 | 4.05e-8 | 5,624 | .01605 s |
| 0 | four 2-element blocks | no, t=0 | 40 | — | 4,248 | .01048 s |
| 0 | two 4-element blocks | no, t=0 | 40 | — | 4,248 | .01096 s |
| 2 | full oracle | yes | 109 | 3.30e-8 | 5,576 | .01577 s |
| 2 | four 2-element blocks | yes | 255 | 3.30e-8 | 7,912 | .02007 s |
| 4 | four 2-element blocks | yes | 130 | 3.79e-8 | 5,912 | .01495 s |

RK4 completes every corresponding case with 328 coefficient-element
 evaluations, approximately `.00093–.00100 s`, and errors
`2.32e-7`, `1.97e-7`, `2.08e-7` for exponents 0, 2, 4. The smaller same-step
Gauss error is real, but substantially too small to compensate for its work.
These are short CPU mechanism measurements, not stable application timing.

Shrinking alpha does make the disjoint joint blocks converge more easily.
Increasing a block from two elements to four barely changes this particular
front-at-interface test: the transmission across the remaining front-bearing
boundaries still matters. The estimate `sech(h/ell)` describes transmission
across an entire element; it is not a bound for a source within `O(ell)` of a
face. No uniform improvement in shock representation follows from these solver
results.

Each factor build uses 224 full residual evaluations for Jacobian columns.
Even assigning **zero cost** to all Jacobian construction and factorization
would leave 2,040, 4,328 and 2,328 coefficient-element evaluations in the
exponent-0 full, exponent-2 local and exponent-4 local cases, versus RK4's 328.
Therefore an analytic or colored Jacobian alone does not resolve this loss.

Joint cubic history reduces the exponent-2 full solve to 40 Newton iterations
on its seven non-startup steps, but it still needs 4,496 coefficient-element
visits and `.01251 s`, versus RK4's 328 visits and `.00096 s`. Its error is
`7.73e-8`, including the charged three RK4 startup steps. Omitting its entire
factor build would still leave 912 coefficient-element visits and about three
times RK4's measured CPU time. History does not rescue this variant.

At `t_final=.128` and `dt=.008`, the exponent-2 local solver completes with
`7.31e-7` error, 20,680 coefficient-element visits and `.05015 s`. RK4 takes
`.00156 s`, 520 visits and has `3.64e-6` error. At `dt=.016`, several retained
chord factors fail before the horizon; this is a failure of the nonlinear
solver configuration, not a proof that the underlying Gauss method is
unstable. Exponent-4 local blocks do complete that larger step with `1.16e-5`
error and `.02367 s`, versus RK4's `6.44e-5` and `.00078 s`. Greater alpha
locality is visible, but the complete cost remains unfavorable.

## Same-physics block placement and practical tolerances

Rotating the two-element blocks by one cell puts each compression front inside
a block without changing the DG mesh, initial condition, alpha or equations.
At exponent 2 this reduces the strict nonlinear iteration count from **255 to
172**; at exponent 4 it reduces **130 to 109**. Exponent 0 still misses the
strict closure target after 40 iterations, though the shifted residual is
smaller (`9.09e-9` versus `1.33e-7`). This is evidence for organizing local joint
work around a front, rather than assuming that arbitrarily placed element
blocks are equally effective.

The practical-tolerance screen uses exponent 2, `dt=.008`, `t_final=.128`,
closure tolerance `1e-3`, and retained factors throughout the run:

| Block placement | State equation tolerance | Completed | Joint iterations | Relative state error | Coefficient visits | Timed CPU run |
|---|---:|---:|---:|---:|---:|---:|
| Front at block cut | 1e-8 | no, t=.072 | 216 | — | 7,280 | .01795 s |
| Front inside block | 1e-8 | yes | 158 | 7.32e-7 | 6,504 | .01613 s |
| Front at block cut | 1e-6 | yes | 292 | 3.84e-6 | 8,648 | .02102 s |
| Front inside block | 1e-6 | yes | **98** | **2.22e-6** | **5,544** | **.01371 s** |
| RK4 | — | yes | — | 3.64e-6 | 520 | .00154 s |

Both the last joint row and RK4 meet a common `4e-6` state-error budget, while
the joint method costs **8.88 times** as much in this CPU implementation. Even
removing all Jacobian setup leaves 1,960 coefficient-element evaluations,
versus RK4's 520. Four local factors have dimension 56, compared with the full
oracle's dimension 224. Their entries occupy 100,352 bytes; the stored unfactored
matrices double this amount. The local factorization arithmetic is
approximately 468,309 FLOPs per build before endpoint factors; the full oracle
costs 7,492,949. These are useful dense operations, without padding in this CPU
implementation. They do not include coefficient construction, current residuals
or other work.

The failed `1e-8` unshifted row already meets its closure target: its last
closure residual is `8.92e-5`, while its Euler collocation residual is
`2.70e-8`. That example fails because of the coupled state iteration, even after
the elliptic criterion passes. Reducing alpha does not remove all joint
coupling difficulties.

Refreshing the full factor every step restores completion at `dt=.016` for
exponent 2, with error `1.03e-5`, but increases complete time to `.07365 s`.
Retained-factor failures must therefore not be identified as a Gauss stability
limit. The iteration/factor lifetime is part of the numerical solver design.

## Accuracy checks and decision

A strict same-alpha matched-quality comparison is also unfavorable: the
shifted joint solve with `dt=.004` has error `3.30e-8` and costs `.01665 s`;
RK4 with `dt=.002` has **smaller error `1.20e-8`** and costs `.00196 s` at the
same `.04` horizon. Joint execution is **8.48 times slower**.

The smooth temporal-order control uses exponent 2, shifted blocks, horizon
`.032`, and stricter nonlinear tolerances. Joint errors for steps
`.008/.004/.002` are `7.3626e-9/4.6128e-10/2.8852e-11`, giving observed orders
**3.996 and 3.999**. The compression refinement is less regular: errors
`3.3048e-8/2.5722e-9/4.9986e-10` give orders **3.68 and 2.36**. The source of
that reduced observed order was not isolated, and smooth fourth order is not
being generalized to the shocked application. Halving the reference step
changes the reported `.004` compression error by about `2.2e-13`. A direct
comparison of the two RK4 reference fields gives relative state L2 difference
`1.93e-12` and sigma nodal difference `3.98e-11`, well below the candidate state
errors in that refinement. This validation-only run overlapped compilation;
its timings are not performance evidence. All 48 rows
in the final screen have conservative integral drift below `9e-16`, including
states retained after failed attempts.

At p1 and p3, separate fixed-physical-alpha controls distinguish changing
regularization from changing approximation order. On p3, exponent-2 scaling
reduces strict iterations from 583 at fixed `alpha=1/576` to 443 at
`alpha=h²/16`. At p1 the scaled alpha is larger than that fixed value, and the
iteration count increases from 193 to 251. These are solver observations on
each corresponding semidiscrete problem, not evidence of matched spatial
accuracy or a universal benefit from changing alpha.

The present finite-difference/chord-Newton CPU implementation should not be
advanced into the application as a performance candidate. Its useful surviving
results are the same-equation sensitivity to front/block alignment, the better
local convergence under reduced alpha, and the verified joint formulation and
current-equation gates. They provide requirements for a substantially different
GPU implementation. The measurements do **not** rule out batched mixed-precision
factors, fused on-chip nonlinear work, analytic or colored Jacobians, or a
joint tensor/ordinary-CUDA method. Those cost balances are unmeasured. A future
candidate must demonstrate that its local matrix work and larger accurate step
compensate for repeated coefficient and interface evaluations; fewer standalone
elliptic factorizations alone is insufficient.

## Reproduction and identities

The authoritative final screen contains 25 configurations and 48 method rows.
The initial screen contains another 25 configurations and 50 rows. Complete
commands and values are included in
[`igr_hp_breakthrough_results.json`](igr_hp_breakthrough_results.json).
The final binary SHA256 is
`759f8ea3301bbd19fbba83370d5af7131b6dd24dfcd5ce2671349d3b8ff2b83c`, built from
prototype source SHA256
`6615a1dd95997c17aa337949ed8c187e71a132de4640b489ade14520a25a115a`.
The initial binary SHA256 is
`cd229bac97d7a2fdb7b87f9ff0acf0b7b943c46ab13bb35a305468519448d231`.
The initial binary omitted final failed-attempt defects; the final screen
corrects that omission. After the final screen, independent review moved the
residual-call increment to function entry so future admission-failing trial
calls are counted. Auditing every final row found no omitted calls in those
measurements, so their work counts remain valid.

Build and focused validation use the canonical driver:

```bash
source /home/jarias9/spack/share/spack/setup-env.sh
./build.sh --gpu --cuda-arch 120 --hdg-runtime --no-clean --no-deps \
  --test-regex '^dg_igr_screened_spacetime_cpu$' -j 16
```

With `PROBE` pointing to `tests/dg_igr_screened_spacetime_probe` in the build
output, the main practical comparison is:

```bash
"$PROBE" --method rk4-joint --elements 8 --order 2 --halo 0 \
  --case compression --alpha-h-power 2 --dt .008 --final-time .128 \
  --reference-dt .0000625 --joint-block-elements 2 --joint-block-offset 1 \
  --joint-rebuild 0 --closure-tolerance 1e-3 --state-tolerance 1e-6
```

Use `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` for these CPU
measurements. MPI initialization requires the authorized execution environment
outside the sandbox. The shared self-test checks conservation, current-density
closure correction, joint current-equation convergence with retained factors,
and wraparound block ownership. No GPU timing was performed for this method.
