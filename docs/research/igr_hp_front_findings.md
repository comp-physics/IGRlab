# hp-dependent IGR scale and front representation

This is a CPU approximation and identifiability screen using the existing full-energy Mach-2 stationary-profile construction. It contains no time advancement, no GPU performance result, and no adoption of a changed production closure. The exact profile follows the [full-energy traveling-wave derivation](https://arxiv.org/html/2607.12693v1); its existence and regularity do not prove dynamic attraction of these states.

## Why a p-dependent scale can help the solver while harming the shock

Write

\[
\alpha=C h^2/(p+1)^s,\qquad \ell=\sqrt\alpha.
\]

Then the three relevant dimensionless ratios have different behavior:

| Quantity | Scaling, up to constants |
|---|---|
| Screening length divided by element width | \(\ell/h=\sqrt C(p+1)^{-s/2}\) |
| Screening length divided by typical interior nodal spacing | \(\ell/\Delta x_{\rm interior}\sim\sqrt C(p+1)^{1-s/2}\) |
| Screening length divided by LGL endpoint spacing | \(\ell/\Delta x_{\rm endpoint}\sim\sqrt C(p+1)^{2-s/2}\) |

Thus `s=2` makes screening more element-local but keeps only a roughly constant number of interior nodes across the layer. It improves resolution of a face-aligned layer because LGL nodes cluster at endpoints. `s=4` approximately holds endpoint resolution fixed while making an interior layer *less* resolved as p increases. This is an asymptotic explanation of translation sensitivity, not an assertion that every source or polynomial has the same error.

A second estimate concerns high-order stiffness. The largest resolved derivative scale is approximately `p²/h`; reaction-scaled diffusion eigenvalues can therefore grow as `alpha p⁴/h² ~ C p^(4-s)`. Removing that growth by choosing `s=4` does not show that the underlying front is accurately represented. The solver and approximation incentives can point in opposite directions.

For the one-dimensional constant-density screened operator, transmission across a whole element has opposite-face/self-face ratio `sech(h/ell)`. At `s=2`, `C=1`, this is `sech(p+1)`. This calculation describes transmission between the *two different faces of one element*. It does not eliminate coupling between the two sides of a single face. A front at distance `d=O(ell)` from a face can still interact strongly with the neighboring element, even if `h/ell` is large. A local screened patch can follow that front across the face; strictly independent element solves cannot infer that transmission from the small opposite-face coefficient alone.

The tests separate `s=0`, which holds physical alpha fixed when h is fixed, from `s=2,4`, which change the regularization as p changes. Each representation error is measured against the corresponding high-resolution profile. These are not equivalent algebraic formulations and lower linear residuals are not used as shock-accuracy evidence.

## Probe and information supplied to each representation

`tests/research/igr_hp_front_probe.py` imports the existing profile and flux formulas from `scripts/perf/igr_shock_representation_probe.py`. It extends the same integrating-factor quadrature to the far fields, stopping within `exp(-28)` of each asymptotic density. The profile is rescaled and translated over seventeen unit elements on `[-8,9]`, with fronts at `0`, `0.37`, `0.5`, and `0.95`. Thus a face, two interior positions, and a near-face position are all represented.

Controls are componentwise best L2 polynomial projection and LGL nodal interpolation of rho, momentum, total energy and sigma. Their nonlinear source is then evaluated from the reconstructed broken velocity derivative. This is deliberately more informative than checking the density plot alone. The source moments use `2 alpha u_x²` against `1`, `(x-x_front)/ell`, and its square. Errors are normalized by the absolute reference moment, so cancellation of an odd moment cannot make the denominator vanish. Derivative ringing far from the front contributes to the moments; it is not discarded by restricting integration to the apparent layer. No distributional square of a DG jump is introduced.

The alternative represents the state using a translated known stationary profile. Its position is inferred either from noisy LGL density samples on the three-element patch `[-1,2]` or from that patch’s conserved density integral. The one-moment fit constrains density only; it does not independently preserve the energy means of a distorted input state. A complete Euler representation would need conservative residual coefficients or additional moment constraints. The position is searched within a coarse bracket `[-0.25,1.25]`; no subelement location is supplied. The latter is a different input: it assumes a conservative mean evolved by the numerical method, rather than claiming that an underresolved point quadrature can recover the exact mean. End states, alpha and the stationary Mach-2 shape are explicit prior information. Exact reconstruction of an exact translated member of that family is therefore a parameter-identifiability check, not general-purpose shock reconstruction.

A richer fit allows one positive width parameter while retaining the prescribed alpha in every source evaluation. This is an approximation of the supplied state; fitting the width does not replace the physical alpha. Every profile-coordinate state is reconstructed with the full-energy invariant formulas and retains positive density and pressure. It does not automatically satisfy the IGR closure for distorted data.

We test bounded deterministic density-sample errors of `0`, `1e-4`, and `1e-3` times the density jump. The conserved-mass test separately applies errors of the same numerical magnitudes times `jump*h`, rather than treating point and integral errors as identical observations. A smooth local density perturbation of amplitude 2% of the jump tests template mismatch. Its momentum, energy and sigma use the same flux-invariant relations, but it is a prescribed positive state, not an exact stationary IGR solution.

## A priori location conditioning

With known end states and a sufficiently large enclosing interval, the total density mass is

\[
M(x_f)=\rho_+ b-\rho_- a-(\rho_+-\rho_-)x_f+\ell c_{\rm profile}
\]

up to exponentially small tail terms. Therefore

\[
\delta x_f\simeq-\delta M/(\rho_+-\rho_-),\qquad
\frac{|\delta x_f|}{\ell}\simeq
\frac{|\delta M|}{(\rho_+-\rho_-)h}\frac{h}{\ell}.
\]

Conserved means can retain location information after point samples have saturated into the two far states. They do not make location accuracy free: for a fixed fractional width error, the allowable mean error shrinks like `(p+1)^(-s/2)`. This provides a useful accuracy budget for a front coordinate evolved through conservative fluxes.

This also supplies a conservative motion equation. With fixed end states, `dM/dt=f_mass(a)-f_mass(b)` implies

\[
\dot x_f=\frac{f_{\rm mass}(b)-f_{\rm mass}(a)+c_{\rm profile}\dot\ell}{\rho_+-\rho_-}.
\]

At fixed alpha this is the Rankine–Hugoniot motion law in the narrow-layer limit. If h or p adaptation changes ell, its shape-mass term must be included; simply narrowing a representation can change mass or front position. Varying end states introduce further chain-rule terms. These equations define a possible conservative parameter update, not a proof that the remaining Euler and instantaneous IGR equations are satisfied.

For point samples separated by `Delta x`, a layer much narrower than that spacing may move substantially without perceptibly changing the observations. In the step limit its position is only bracketed by adjacent samples. The unresolved interval divided by ell scales as `(p+1)^(s/2-1)` at interior nodes. Nonlinear profile fitting cannot recover information that was not observed.

## Measurements

The front/alpha screen has 1,520 records: the main `p=1,3,7,15`, `s=0,2,4` sweep; fixed-alpha `1/16` and `1/256` controls; even `p=2,4` controls; a fixed-alpha p31 control; and selected quadrature refinement. Total CPU analysis time across those six commands was 57.26 seconds. This includes common profile construction, nonlinear fitting, projection and diagnostics; it does not predict per-stage GPU cost.

For an interior front at `x_f=0.37h`, the density L2 error divided by `density_jump*sqrt(ell)` is:

| Alpha family | p3 | p7 | p15 | p31 |
|---|---:|---:|---:|---:|
| Fixed alpha `1/16` | 0.0780 | 0.0286 | 0.00954 | not tested |
| `h²/(p+1)²` | 0.0780 | 0.0816 | 0.0747 | not tested |
| Fixed alpha `1/256` | 0.4819 | 0.2182 | 0.0747 | 0.0267 |
| `h²/(p+1)^4` | 0.4819 | 0.8789 | 1.3907 | not tested |

These are best componentwise L2 projections; the source is subsequently computed from the reconstructed velocity. Holding physical alpha fixed permits density approximation to improve. Shrinking it like `(p+1)^-2` nearly cancels that gain at interior positions, while shrinking it like `(p+1)^-4` reverses the trend. A face-aligned `s=2` front improves substantially: the corresponding density errors decrease from 0.0248 to 0.0078 to 0.0028 at p3, p7 and p15. Near-face and even-order tests support the same location dependence. At p7, `s=4`, `x_f=0.37h`, even the best componentwise projection has negative pressure, approximately `-0.157`; the p2 counterpart also fails positivity. The profile reconstructions remain positive in every tested case.

Density convergence does not imply convergence of the differentiated nonlinear source. For `s=2`, a p15 front at the element center has only a 0.0755 normalized density error, but the total source integral is 40% high and the second source moment is over 100 times its reference value. The second-moment diagnostic counts derivative ringing away from the narrow layer. This is not a claim that Euler solution error is 100 times too large; it identifies a failure of the source reconstruction. Fixed-alpha p refinement improves density, but the polynomial derivative/source moments can remain nonmonotone even at p31. LGL interpolation often has smaller source errors than the best L2 projection despite a larger density error; neither control removes the `s=2,4` phase sensitivity.

For the exact template family, fitting a single position to local nodal samples is much more accurate than differentiating the polynomial. With `s=2`, p3/p7/p15, all four front positions and sample errors of `1e-3` times the density jump, the worst relative weak-source-moment error remains below 0.9%. This is the useful parameter-identifiability result. At `s=4`, the information loss between nodes eventually dominates: at p15 and the cell center, the same sample error gives a position error of about `5.36 ell`, with very inaccurate source moments. Freeing the width makes that ambiguous case worse rather than recovering missing information.

Conserved mass avoids this phase blindness. With the correct template and an integral error of `1e-4 * density_jump * h`, the p15 `s=2` source-moment error is about 0.52% across all four phases. With `s=4` it is about 8.36%. For an error ten times larger these become approximately 5.2% and 83.6%. These results follow the derived `delta M/ell` conditioning and do not assume that existing evolved cell means already meet those budgets.

Template bias is a separate obstacle. Adding a local smooth density distortion of amplitude 2% of the jump creates approximately 12% source-moment error even with the exact front location. Fitting both position and width partly repairs that distortion: for `s=2`, p3/p7/p15 and the `1e-3` sample perturbation, the tested errors range from about 2.6% to 13.3%. It does not pass a 1% robust source gate. Flux invariants and positivity remain satisfied, showing that those checks alone cannot certify the nonlinear closure source. More local shape freedom or a compatible residual representation is needed before treating a profile coordinate as a general shock approximation.

All high-resolution profile checks pass: field refinement changes at most `2.81e-9`, normalized weak closure moments at most `7.52e-11`, and the density primitive differs from independent quadrature by less than `9.88e-13`. Doubling quadrature for the selected difficult source cases changes the reported source error by less than `6.39e-11 * max(1, error)`. Self-tests check conservative moment inversion and point-fit recovery where transition samples exist; they deliberately do not assert recoverability from saturated samples.

## Reproduction and identities

The tested source was HEAD `c04fd008bd50bd88439da0d08a294514b9616856` plus this new probe. The first six commands used probe SHA256 `e84d818cf034b5eb93f21e5d4f714782ba91446f4aba6269e207bdbe228c906b`. The final source adds only the optional H1 control and its tests; its tested SHA256 is `f634a748545f9eb9cf9d65d157761c2b935d63204702637414850e6b65e827a7`. Imported profile-helper SHA256: `4bc6814ab317c77e656d1d7adce955e27657567d01f288c50e7cc54622b2d8c1`. NumPy and SciPy versions, full arguments, per-case work and file hashes are stored in the results and `/tmp/igr-hp-front-evidence-index.json`. These are test-only Python experiments and do not invoke an application build.

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_hp_front_probe.py --self-test
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_hp_front_probe.py --output /tmp/igr-hp-front-final.json
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_hp_front_probe.py --output /tmp/igr-hp-front-fixed-c00625.json --exponents 0 --c-alpha .0625 --orders 3 7 15 --noise 0 .001
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_hp_front_probe.py --output /tmp/igr-hp-front-fixed-c000390625.json --exponents 0 --c-alpha .00390625 --orders 3 7 15 --noise 0 .001
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_hp_front_probe.py --output /tmp/igr-hp-front-refine.json --orders 7 15 --exponents 2 4 --fronts .37 .5 --noise .001 --distortions 0 .02 --quadrature 48
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_hp_front_probe.py --output /tmp/igr-hp-front-even-orders.json --orders 2 4 --exponents 2 4 --fronts 0 .37 .5 .95 --noise .001 --distortions 0 .02
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_hp_front_probe.py --output /tmp/igr-hp-front-fixed-p31.json --orders 31 --exponents 0 --c-alpha .00390625 --fronts 0 .37 .5 .95 --noise .001 --distortions 0 .02 --quadrature 48
```

The additional H1 commands were:

```bash
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_hp_front_probe.py --output /tmp/igr-hp-front-h1.json --h1-controls --orders 3 7 15 --exponents 0 2 4 --fronts 0 .37 .5 .95 --noise .001 --distortions .02
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_hp_front_probe.py --output /tmp/igr-hp-front-h1-fixed.json --h1-controls --orders 3 7 15 --exponents 0 --c-alpha .0625 --fronts 0 .37 .5 .95 --noise .001 --distortions .02
```

The complete index therefore contains 1,856 records across eight commands, with 71.33 seconds of CPU analysis. The raw results have deliberately explicit names under `/tmp`; the [campaign results archive](igr_hp_breakthrough_results.json) preserves the selected evidence without adding production diagnostic infrastructure.

## Conservative H1 recovery from available traces

A separate control tests whether a local reconstruction can reduce derivative ringing without knowing the front location or its derivative. Let `c` be the coefficients of one available polynomial, `M` its exact element mass matrix, and `K` its gradient Gram matrix. Integration by parts gives

\[
(M+\alpha K)(\widetilde c-c)
=\alpha\left[\phi'_R(\widehat U_R-U_R)
-\phi'_L(\widehat U_L-U_L)\right].
\]

The shared trace is the average of the two neighboring polynomial traces. All inputs are available polynomial coefficients and face values. The two response columns can be computed once for each h,p,alpha combination and reused. The constant test function has zero derivative, so each conserved component’s cell mean is preserved exactly. This is a candidate gradient/state reconstruction, not an equivalent change to the evolving DG scheme.

There is an immediate limitation: if the two polynomial traces already match, the correction is identically zero. In particular, interpolation of a single continuous profile at shared LGL nodes has this property. Its derivative ringing cannot be cured by a reconstruction that only responds to value jumps. This makes the LGL control a useful falsification test, rather than assuming H1 recovery must help.

The measured control matches that prediction. Across both sweeps, every component’s cell mean changes by exactly zero in the stored FP64 coefficients, and the largest LGL coefficient correction is less than `3.54e-15`. The source behavior improves in some discontinuous L2 controls but does not pass a robust p-scaled gate. At `alpha=1`, p15, face-aligned front, the largest source-moment error decreases from 2.1% to about 0.25%. At `s=2`, p15, face-aligned front, total source error improves from -15.2% to -1.9%, but the second-moment error increases from 28.3% to 131.2%: the correction spreads source away from the narrow layer. For the p15 interior center the worst moment error decreases from 113.8 to 69.3 times the reference, still unacceptable. The H1 reconstruction improves that error metric in 22 of 36 main cases and 6 of 12 fixed-alpha cases, with substantial losses elsewhere. It is retained as a test-only negative/conditional control, not integrated into the advancing application.

These results rule out a simple value-jump correction as the missing ingredient. Matching conservative means, face values, positivity or total source separately does not ensure a correctly localized nonlinear source. In particular, the remaining LGL ringing is invisible to this reconstruction.

## Practical next gate

The useful candidate is a conservative front coordinate plus compatible weak nonlinear moments on a small patch that follows the front across faces. It should retain a current face correction even where the screening length is below the element width. A moving test must include imperfect position, profile mismatch, positivity, conservation, source consistency, and complete update cost. Simply choosing `alpha ~ h²/p²` with the existing polynomial source is not enough. A stationary template by itself is also insufficient if realistic shape variations leave a significant source bias.
