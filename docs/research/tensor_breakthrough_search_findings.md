# Screened closure, space-time coupling, and shock representations

This investigation tested three ways to change the numerical work behind DG+IGR: exploit screening in the current elliptic operator; exchange global stage solves for coupled local space-time work; and represent an IGR shock according to its actual regularity. The results support specific follow-up experiments, not a claim of a novel breakthrough or a tensor-core speedup.

The production target remains full-energy Euler with instantaneous IGR. Retained patch preconditioning changes the algebraic solver only. The reduced space-time experiment changes spatial and temporal discretization, and deliberately truncated stage coupling requires its own error assessment. Cusp enrichment is currently an approximation experiment, not an advancing solver. No changed continuum model was adopted.

The integrated code identity is `32292ede7183034b138b33a5c8f8a42bdae81a92`, tree `fd82584a62720cb26514f3e7bf7f3d53eef64d3e`. Six focused canonical tests passed, including CUDA patch action/current-operator correction, DGSEM parity, temporal tests, and stationary correction. The accompanying [results archive](tensor_breakthrough_search_results.json) preserves tested identities, commands, environment overrides, numerical results, and validation evidence. Earlier measurements are labeled by their actual binary/source identity rather than retrospectively assigned to the final commit.

## 1. Screening helps correction; it does not make closure element-local

For constant density, the closure reduces to a screened Poisson equation with length scale `sqrt(alpha)`. This suggests overlapping patches whose radius follows physical screening, followed by correction against the full current operator. Localized elliptic correctors have substantial prior art, including [reduced-basis localized decomposition for reaction–convection–diffusion](https://arxiv.org/abs/2211.15221). The implemented center-restricted principal-patch inverse is ordinary [restricted additive Schwarz](https://doi.org/10.1137/S106482759732678X), not a new numerical method by itself.

The frozen-operator probe uses the application's full-energy state conversion, IGR coefficient/source construction, L2 closure space, broken velocity gradient, and exact DG face operator. An independently checked FP64 dense solve provides the reference. Prescribed smooth compression, cell-constant loads, and high-mode loads separate actual closure behavior from Green-function locality. Host reference sampling and local factorizations are charged and identified as CPU probe work.

Closure FGMRES iteration counts at relative residual tolerance `1e-8` were:

| Prescribed smooth shock | Element block, radius 0 | Radius 1 | Radius 2 | Radius 4 |
|---|---:|---:|---:|---:|
| 16×1, p3, C=1 | 59 | 16 | 10 | 6 |
| 16×1, p5, C=1 | 87 | 16 | 10 | 6 |
| 64×1, p3, C=1 | 80 | 17 | 10 | 6 |
| 8×8, p3, C=1 | 149 | 45 | 15 | unmeasured |
| 16×1, p3, C=0.1 | 20 | 6 | 4 | 2 |
| 16×1, p3, C=10 | failed at 200 | 28 | 23 | 16 |

These results support patch correction across p3–p5 and a longer strip. They do not establish a speedup over a strong CUDA solver. The 2D p3 probe needed 11 iterations at radius 3. Point Jacobi often exhausted its 200-iteration cap and is not the application performance baseline.

At C=1, approximately 9.24% of the cell-constant response L2 norm remains beyond two face hops, and 1.26% beyond four. At C=10 those tails grow to 49.2% and 28.7%; at C=0.1 they fall to 0.0717% and 0.000120%. Sixteen undamped radius-two corrections at C=10 still had relative residual 1.80. Thus a small fixed number of local sweeps cannot simply replace a certified instantaneous solve.

Here `alpha=C h²`; the N=64 and C sweeps intentionally change physical alpha and the prescribed transition width. Their nearly unchanged patch iterations do not prove mesh-independent locality at fixed physical alpha. At fixed alpha, resolving the same physical patch requires more elements as h decreases.

The integrated CUDA implementation reconstructs face blocks with graph-colored operator probes, factors principal patches in FP64, and retains only center inverse rows. Its optional FP32 mode casts input values, products and accumulation to FP32, then returns FP64 output; this is ordinary CUDA, not tensor execution. FGMRES still applies the current FP64 operator. Independent nonsymmetric local-solve tests check layout, and changed-operator tests verify corrected residual and solution accuracy with retained initial factors. Setup accounting includes sampling, validation, factorization, conversion, upload and synchronization. `upload_seconds` and `conversion_upload_seconds` are aliases, not additive costs.

The next useful tensor experiment would batch several genuine temporal right-hand sides through each patch matrix. Single-vector row application alone offers no demonstrated tensor benefit. Even the two temporal right-hand sides below make a narrow matrix product; tensor occupancy and padding amortization remain unestablished. The decisive gate is complete accurate-horizon cost, including patch reconstruction/rebuilds and correction iterations, against calibrated ordinary CUDA.

### Application tests: evolving density changes the useful patch size and lifetime

Both application cases use the existing full-energy DGSEM transport, central plus LLF volume flux, Rusanov faces, Zhang–Shu positivity limiter, SSPRK104, L2/BrokenL2 closure, exact DG faces, and the fused ordinary FP64 residual. Every stage receives a current closure solve. `alpha=h_local²`; refreshing a preconditioner does not lag the physical closure. Complete process timings include startup, coefficient construction, operator formation, setup/rebuilds, conversions, all stages, final closure, conservation accounting and initial/final VTK output.

The curved-cylinder case uses the existing 1,322-element mesh at p1, with 5,288 closure DOFs, beyond the existing global inverse's 4,096-DOF cap. All candidate cylinder runs reach time 0.5 in 100 fixed steps of 0.005, with zero rejected attempts. This is a finite transient horizon, not a demonstrated steady bow-shock solution. The reference uses the same mesh, FP64 CG tolerance `1e-6` and time step 0.0025, taking 200 steps.

Screening results, each initially measured once, explain the selection:

| Closure solver | Complete seconds | Outcome |
|---|---:|---|
| Radius 2, initial factors retained, FGMRES `1e-4` | 52.43 | Loses; 40.6 iterations per closure |
| Radius 4, initial factors retained, FGMRES `1e-3` | 28.32 | Loses; 24.8 iterations per closure |
| Radius 4, FP64 refresh every 50 calls | 16.81 | 5.54 seconds spent rebuilding |
| Radius 4, FP32 refresh every 50 calls | 16.81 | No resolved FP32 gain |
| Radius 4, refresh every 100 / 200 calls | 15.36 / 15.70 | Less setup offsets more iterations |
| Radius 3, refresh every 100 calls | 13.47 | Best screened-patch calibration |
| CG, tolerance `1e-2`, Chebyshev degree 1 / 2 / 4 | 19.47 / 17.19 / 18.10 | Degree 2 is the strongest measured ordinary control |

The selected patch mode uses FP64 FGMRES tolerance `1e-3`. Its maximum conservative-component relative sample RMS difference from the reference is `2.67e-7`, versus `5.52e-7` for the selected CG control; sigma differences are `4.01e-7` and `9.47e-7`. These are unweighted comparisons at identical decoded VTK points, with geometry, connectivity, array schemas and physical settings checked. They are not physical-space L2 integrals or a spatially converged cylinder reference. All saved density and independently reconstructed pressure values are positive. The centerline extraction has only 11 binned points: its shared standoff proxy of 0.154316 cannot establish equal physical shock positions.

Three new repetitions of each selected cylinder mode, with order reversed in the middle pair, give median complete times **13.550 seconds for patches and 17.279 seconds for CG: 1.28× speedup, or 21.6% less elapsed time**. The respective ranges are 13.497–13.556 and 17.265–17.312 seconds. The first repeated patch run spends 6.608 seconds forming current operators, 0.896 seconds on preconditioner setup/rebuilds, and 2.428 seconds in Krylov solves. It performs 6,447 iterations over 1,001 closure calls, including the output solve. The first repeated CG run performs 24,027 iterations. Operator formation is now a larger target than the patch matrix-vector multiply; the gain is not attributable to tensor hardware.

For 5,288 useful closure unknowns, each selected patch application performs 489,376 scalar multiply-accumulate terms. A rebuild samples/validates through 39 current-operator applications and factors patches containing 11,549,280 matrix entries in total, with at most 112 DOFs per patch. Those overlapping factorizations repeat local algebra; they are fully charged. Only center inverse rows and indexing remain on device, totaling 4,430,828 bytes. There is no tensor-tile padding in this implementation. Final VTK files are byte-identical within each three-run method group; the maximum conservation/boundary-balance relative defect across the six repeats is `1.34e-13`.

The planar p2 case gives the opposite ranking. On the existing 32×1 mesh, all six repeated runs reach time 6 in 120 fixed steps of 0.05, with no rejections. The existing global FP64 inverse with CG tolerance `1e-2` has median complete time **3.450 seconds** (3.370–3.594), versus **4.231 seconds** (4.219–4.249) for retained radius-four patches with FGMRES tolerance `1e-4`. Against a same-mesh FP64 dense reference at step 0.025 and tolerance `1e-10`, physical-space polynomial quadrature gives relative density errors `1.00e-8` and `2.44e-8`, and sigma errors `4.26e-8` and `9.53e-8`. Transverse momentum is near zero and is assessed by absolute error, below `4e-14`, not its meaningless relative error. These very small reference differences include both time and linear-solve error; they do not certify spatial shock accuracy. Comparing patches only with the earlier tighter CG or dense baselines would overstate their value. Initial p1/p2/p3 screens also differ in spatial solution and are not a matched-accuracy p-ranking.

The retained factors remain an opt-in, single-rank CUDA research option for uniform scalar nodal L2 quadrilateral spaces; unsupported configurations reject explicitly. Radius is measured in face hops. Current setup is host factorization with CUDA row application, bounded by 512 DOFs per patch and a sampled-block memory cap. Increasing p can exceed that cap. No default production solver was changed.

## 2. Joint history enables fourth-order local space-time work, but has not won

Standard [ADER local space-time predictors](https://doi.org/10.1007/s10915-021-01429-8) exploit a local PDE. Instantaneous elliptic IGR prevents assuming that every element has an independent predictor. The reduced implementation instead exchanges elliptic boundary waveforms across overlapping three-element patches. It uses two Gauss time nodes, accepted-state patch factors, two genuine stage right-hand sides, and one shared conservative corrector flux per face per time node.

This is a periodic pseudo-1D full-energy Euler experiment with 12 p3 elements and a continuous, lumped LGL closure discretization, alpha=0.0025. It is not the application's 2D SIPG discretization. Both reaction and diffusion use current density in the closure residual. Accepted endpoints are globally reclosed; three RK4 startup steps supply the joint cubic history and are charged. The strong RK4 control reuses the accepted endpoint closure, costing `1+4*steps` global factorizations.

The order experiments distinguish materially different methods. Converging the constant-seed Gauss/Schwarz equations gives fourth-order smooth accuracy, but requires roughly 12–15 outer sweeps per step and substantial extra CPU work. One fixed outer update with a constant seed approaches first order; extrapolating sigma alone approaches second order. Joint cubic histories of state and sigma recover approximately fourth order. With one inner local predictor sweep, compression errors at final time 0.04 for dt=0.004, 0.002, 0.001, 0.0005 are `5.72e-5, 3.49e-6, 2.12e-7, 1.32e-8` in relative state L2, with final observed order 4.01.

Removing that inner sweep defines a separate explicit history-based Gauss flux-quadrature method; it does not target the collocation equations and reduces to AB4 for a linear autonomous ODE. Its compression errors are `2.93e-3, 2.19e-5, 9.05e-7, 5.91e-8`, with final observed order 3.94. This variant must not inherit implicit Gauss stability claims.

At dt=0.001, strong RK4 has state error `2.05e-8`. The inner-one method has about ten times that error and takes about 2.6 times the CPU solver time. The inner-zero method has 44 times the error and takes about 1.2 times the time. Both reduce global factorizations from 161 to 50 and estimated dense factor work from 2.50 million to 0.853 million FLOPs, but increase coefficient-element visits from 1,932 to 8,148 or 5,484. Retained patch matrices plus factors occupy 12,288 bytes. Solver times include initial closure, startup, patch setup and certification, but exclude shared basis/problem construction, MPI startup and reference computation; process manifests record the latter costs too. All these runs reach the requested short horizon; none is a measured performance winner.

Fixed outer work leaves maximum current stage-closure residuals around `8e-5` in that compression case, so these runs are not certified instantaneous stage solves. Endpoint reclosing and fourth-order convergence are insufficient to erase that distinction. Conservation drift is about `1e-15`; halving the reference time step changes the reported compression errors by only a few parts per million. A v1 failure caused by normalization against a zero volume source, despite nonzero boundary driving, was corrected in v2 and is not evidence of method instability.

The longer compression horizon of 0.2 rejects the inner-zero variant at dt=0.004: density/pressure admission fails after time 0.052. At dt=0.002 its current-density patch correction fails after time 0.102. At dt=0.008 it also fails the shorter 0.08 target after time 0.056. Strong RK4 reaches every corresponding target. Inner zero reaches 0.2 at dt=0.001, but its state error is `1.96e-6`, versus RK4's `5.03e-8`, and it takes 0.0362 s versus 0.0285 s. The failed attempts are charged and excluded from performance winners.

Inner one reaches 0.2 at dt=0.004, 0.002 and 0.001, with state errors `3.05e-5, 1.93e-6, 1.32e-7`, approximately fourth-order decay. At the same steps it takes about 2.7 times the RK4 CPU time and has 2.6–2.9 times its state error; stage closure remains uncertified. The reduced evidence justifies stopping development of these variants as local performance candidates. Whether reduced global communication outweighs extra coefficient and predictor work in the actual application remains unresolved. It does not justify implementing a full ADER solver first or claiming a tensor speedup. Raw results, failed horizons, work counts and exact commands are in `spacetime-analysis.json` and `spacetime-commands.json`, v2 entries, under `/tmp/dg-igr-breakthrough`.

## 3. A real IGR shock favors different state and pressure representations

Recent full-energy traveling-wave analysis establishes a continuous density profile with a singular derivative at its sonic crossing, under specific one-dimensional shock assumptions. For gamma=1.4, the density exponent is 6/7. This is **not an H1 incompatibility**: density remains locally H1. Since the pressure reconstruction has zero density derivative at the sonic state, our regularity inference gives `Sigma-Sigma_s ~ |x-x_s|^(12/7)` and `Sigma_x ~ |x-x_s|^(5/7)`. Sigma is locally C1 and H2 here. The paper does not establish dynamic emergence or stability from general initial data. [Primary shock-profile analysis](https://arxiv.org/abs/2607.12693)

The CPU representation probe constructs the actual ideal-gas Mach-2 stationary profile from its integrating factor on `[-2,2]*sqrt(alpha)`. It checks quadrature/grid refinement, positivity, full-energy flux invariants and weak closure moments. Independent density-space source integration checks x-space quadrature. Refinement changes reference fields by at most `6.86e-10`; weak closure moments are below `7e-11`. All 56 fits pass finite-value, sampled-positivity, rank and quadrature checks.

At 64 independent coefficients per conservative/sigma field:

| Representation | Relative density L2 | Relative Sigma L2 | Worst source-moment error |
|---|---:|---:|---:|
| Uniform p3 | 1.824e-4 | 1.278e-4 | 5.40% |
| Split polynomial, exact cusp | 9.399e-6 | 1.017e-6 | 1.30% |
| Graded hp, exact cusp | 2.420e-6 | 7.899e-6 | 0.362% |
| Four-power enrichment, exact cusp | 1.688e-6 | 5.707e-7 | 0.0349% |
| Enrichment, location error 0.02√alpha | 1.496e-4 | 4.578e-5 | 1.625% |
| Enrichment, location error 0.1√alpha | 4.366e-4 | 2.487e-4 | 10.60% |

Source moments integrate `2u_x²` against `1,x,x²`, normalized by the corresponding absolute reference moment. Fits report face jumps separately; no jump-square distribution is invented. Adaptive representations consume one additional location parameter. The uniform control already places the cusp on a face. Graded hp's smallest cell is 128 times smaller than uniform p3's, potentially erasing approximation gains through the explicit CFL restriction.

This is a capacity result, with strong location sensitivity, not a dynamic speedup. An enriched density derivative is infinite at the cusp; inserting it unchanged into pointwise LGL source evaluation is invalid. A credible advancing method needs compatible weak source moments, specialized quadrature, conservative fluxes and imperfect-location tests. The frozen discontinuous-state probe reinforces that question: an element-aligned jump gives projected broken-gradient source near `3e-31`, whereas a shifted nodal jump gives 2.81. Those are diagnostic discrete states, not two equivalent smooth continuum solutions.

A separate 12-case check translates continuous tanh fronts by 0.37h. At width 0.25h, the frozen reference sigma L2 norm changes by +52.5%, −16.1% and +0.6% at p1, p3 and p5. At width 0.125h, those changes are +90.2%, −21.3% and −13.7%. These are prescribed underresolved states, not exact stationary IGR profiles; a small finite-boundary influence also remains possible. The improvement at p5 for the wider front argues against declaring the continuum formulation intrinsically defective. It does support testing source construction and front representation together rather than merely replacing h with h/(p+1) in alpha.

Ordinary [hp/subcell shock capturing](https://doi.org/10.1007/s10915-022-01770-6), [shock tracking](https://doi.org/10.1002/fld.5293), and [DG IGR itself](https://arxiv.org/abs/2608.02223) are established. An IGR-specific compatible coupling may be worth developing; its novelty has not been established.

## Other hypotheses and the next decisive experiments

Three materially different directions remain unimplemented. A discretize-first logarithmic-barrier formulation could derive compatible gradients and energy work; the [original geometric IGR construction](https://arxiv.org/abs/2308.14127) is prior art. Its first gate is consistency with the full-energy instantaneous equations and translation-stable source moments for one resolved shock. Positive-part source clipping or changing alpha must be labeled a model/regularization change.

A hyperbolic pseudotime closure could reuse the DG engine until the current elliptic residual converges. [Euler–Poisson DG already demonstrates this approach](https://arxiv.org/abs/2008.10593). Its cheapest falsification is a frozen shock operator: count every pseudo-step to the same current residual and compare total work. Advancing finite relaxation in physical time instead introduces a model approximation requiring separate limits and authorization before production adoption.

Nonlinear elimination of coupled shock patches could leave a smaller interface problem, using [nonlinear Schwarz/substructuring](https://arxiv.org/abs/2103.16999). A two-to-four-patch full-energy test should reject it if local Newton work and global correction erase any larger accurate time step. None of these directions has been timed here.

The strongest new-method hypothesis suggested by these experiments combines a tracked cusp coordinate and separate state/sigma enrichments with weak source moments, inside a screened local space-time solve. Schematically, the state uses polynomial coefficients plus powers of the signed distance to the front; sigma uses its smoother pressure-specific powers. Front position becomes an evolved unknown, not an oracle. Local stage coefficients and front motion could supply genuine multiple right-hand sides or block Newton work for tensor hardware, followed by conservative interface and current-closure correction. This is a proposed combination, with no novelty or stability claim. Its next decisive question is whether an imperfectly located, moving shock can retain the measured source-moment advantage without violating conservation, positivity or the instantaneous closure. The current location-sensitivity results make that a harder and more useful gate than another dense contraction benchmark.

An end-to-end DG computation can combine tensor contractions with ordinary CUDA fluxes, coefficient construction, admissibility checks and communication. This campaign has not established an end-to-end tensor-core winner for shocked DG+IGR. Tensor throughput alone cannot remove the costs of stage dependency, nonlinear coefficient updates, reductions, packing or changing local factors. The two-stage probe creates genuine matrix right-hand sides but currently increases complete numerical work; the best new application result uses ordinary CUDA. H100/H200/B200 deployment remains unmeasured. The RTX result must not be extrapolated using sparse or FP8 headline throughput: relevant comparisons need dense throughput at the actual FP64, TF32 or FP16/BF16-with-FP32-accumulation precision, plus correction costs. In particular, these tests do not decide the value of FP64 tensor hardware on data-center GPUs.

## Reproduction and remaining scope

Build through `dg-igr-build` and the canonical `build.sh` route. This run used CUDA 13.3, the local RTX 5080, Release, one MPI rank and `OMP_NUM_THREADS=1`, app architecture 120 and MFEM 80-virtual, with MFEM pin `d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec`. Initialize Spack, then run the canonical focused validation:

```bash
source /home/jarias9/spack/share/spack/setup-env.sh
flock -n /tmp/dg-igr-gpu0.lock ./build.sh --gpu --cuda-arch 120 --hdg-runtime \
  --no-clean --no-deps -j 16 --test-regex \
  '^dg_igr_(igr_screened_patch_cuda|screened_spacetime_cpu|igr_stationary_correction_cuda|dgsem_igr_tests|dgsem_operator_cuda|igr_tests)$'
```

Use the build directory reported by the driver as `BUILD`; here it was `/home/jarias9/.cache/dg-igr-build/worktrees/wt-1f1e8ddf9e6432a2/cuda-sm120/app/build`. Run GPU/MPI executables outside the filesystem sandbox; keep CPU probes and compilation outside GPU timing windows. A representative application command is:

```bash
flock -n /tmp/dg-igr-gpu0.lock python3 scripts/perf/dgsem_precision_horizon.py \
  --executable "$BUILD/examples/supersonic_example_w_igr/supersonic_cylinder_w_igr" \
  --case supersonic-cylinder --order 1 --dt .005 --final-time .5 \
  --integrator ssprk104 --sigma-space L2 --volume-dissipation llf \
  --positivity-limiter zhangshu --diagnostic-scans off --conservation-balance \
  --fused-residual --cheb-order 2 --cheb-power-iterations 3 --timeout 90 \
  --linear-solver fgmres --linear-rtol 1e-3 --patch-radius 3 \
  --patch-rebuild-interval 100 --output-dir /tmp/igr-cylinder-patch
```

For the ordinary control, remove the two patch options, use `--linear-solver cg --linear-rtol 1e-2`, and choose a fresh output directory. For the cylinder reference use CG degree 4, tolerance `1e-6`, and `--dt .0025`. Exact commands for every application run are in the results archive. Representative reduced commands are:

```bash
"$BUILD/tests/dg_igr_screened_locality_probe" --nx 16 --ny 1 --order 5 --radius 4 --c-alpha 1 --state shock-smooth
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$BUILD/tests/dg_igr_screened_spacetime_probe" --method both --case compression --fixed-outer 1 --inner 1 --sigma-predictor history3 --state-predictor history3 --final-time .04 --dt .001 --reference-dt .00003125 --closure-tolerance 1e-12 --state-tolerance 1e-13
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$BUILD/tests/dg_igr_screened_spacetime_probe" --method both --case compression --fixed-outer 1 --inner 0 --sigma-predictor history3 --state-predictor history3 --final-time .2 --dt .004 --reference-dt .00003125 --closure-tolerance 1e-12 --state-tolerance 1e-13
OPENBLAS_NUM_THREADS=1 python3 scripts/perf/igr_shock_representation_probe.py --output /tmp/dg-igr-breakthrough/representation-final.json
```

The second space-time command reproduces the failed long-horizon candidate alongside completed RK4; inspect each JSON row's `complete` and `reached_time`, because process success alone does not imply candidate completion.

The representation script SHA256 is `4bc6814ab317c77e656d1d7adce955e27657567d01f288c50e7cc54622b2d8c1`; the authoritative space-time v2 binary SHA256 is `2cb0ccb78d1c52b5363a6c98f7f2ed05bd26afd05d999e5b9a893af7700077e7`. The tested cylinder binary is `800c561ac56e4ed9d9da72506f27d8e5e0b283d0344bdfde5a456fc2365bbb55`. Full raw VTK and logs remain under `/tmp/dg-igr-breakthrough`; the repository archive preserves bounded measurements and reproduction information. No remote resources or deployment GPUs were used.
