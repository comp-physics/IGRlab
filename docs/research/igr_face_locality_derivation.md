# Element-local interiors and coupled faces for instantaneous IGR

The elliptic interiors can be solved independently while the remaining coupling lives on element boundaries. This is an exact algebraic option for the current SIPG operator after a suitable basis change. It does not make the inverse finite-range, and simply weakening the face coupling changes the discrete closure. The test-only probe accompanying this note distinguishes those cases on frozen operators exported by the application.

## 1. Energy and what can be confined

Freeze an admissible full-energy Euler state and write the closure as

\[
c\sigma-\nabla\!\cdot(k\nabla\sigma)=f,
\qquad c=\rho^{-1}>0,\quad k=\alpha\rho^{-1}>0.
\]

The source remains the application's source; this derivation does not replace it with an invented jump-square term. With natural boundary conditions, the continuous energy is

\[
\mathcal E(v)=\tfrac12\int(c v^2+k|\nabla v|^2)-\int fv.
\]

Positive reaction removes the constant nullspace. For an approximate closure with residual functional \(r\), the exact energy error satisfies \(\|e\|_A=\|r\|_{A^{-1}}\). A Euclidean residual tolerance is only a surrogate for this norm. Uniform coefficient bounds and suitable SIPG penalty/trace bounds give discrete coercivity; positivity of density alone does not certify an arbitrarily chosen DG penalty. The probe checks symmetry and positive definiteness of each actual sampled operator.

For piecewise-constant coefficients, a disturbance crossing distance \(L\) has a characteristic decay \(\exp(-L/\sqrt\alpha)\). The coefficient contrast, boundaries, tangential mode and source norm affect any quantitative bound. A small screening length supports inexpensive correction over nearby elements. It does not turn an exact elliptic response into a compactly supported one.

## 2. An exact element-interior elimination for the existing SIPG matrix

The current closure basis is Gauss–Legendre nodal L2. Every GL nodal basis function generally has a nonzero boundary trace. Marking GL nodes as geometrically interior and deleting their interelement coupling is therefore invalid.

Instead change basis within each element to the same \(Q_p\) polynomial space expressed at Gauss–Lobatto nodes. Let \(T_K\) evaluate the LGL cardinal basis at the original GL nodes. This is an invertible change of coordinates, not a different quadrature or spatial method:

\[
\sigma_{\rm GL}=T\widetilde\sigma,\quad
\widetilde A=T^TAT,\quad \widetilde b=T^Tb.
\]

Split the transformed coefficients into zero-trace bubbles \(I\) and the remaining coefficients \(\Gamma\). Every SIPG face term contains a trace of at least one argument. Hence two zero-trace bubble functions on different elements do not couple, although the normal derivative of a bubble need not vanish. Consequently

\[
\begin{bmatrix}A_{II}&A_{I\Gamma}\\A_{\Gamma I}&A_{\Gamma\Gamma}\end{bmatrix}
\begin{bmatrix}\sigma_I\\\sigma_\Gamma\end{bmatrix}
=\begin{bmatrix}b_I\\b_\Gamma\end{bmatrix},
\qquad A_{II}=\operatorname{blockdiag}_K A_{II,K}.
\]

This locality must be checked on the actual operator. The prototype measures all transformed cross-element bubble blocks before using local factors. SIPG derivative terms can couple one element's bubble to a **neighbor's** trace. They must remain in \(A_{I\Gamma}\); ignoring them would change the matrix.

Eliminating bubbles gives

\[
S\sigma_\Gamma=g,\qquad
S=A_{\Gamma\Gamma}-A_{\Gamma I}A_{II}^{-1}A_{I\Gamma},\qquad
g=b_\Gamma-A_{\Gamma I}A_{II}^{-1}b_I.
\]

After the interface solve, each interior is recovered independently. For SPD \(A\),

\[
z^TSz=\min_y
\begin{bmatrix}y\\z\end{bmatrix}^T\widetilde A
\begin{bmatrix}y\\z\end{bmatrix}>0\quad(z\ne0).
\]

Thus the condensed system remains SPD and the original discrete solution is unchanged. Exact recovery leaves residual only in the interface equations. Its energy error is exactly \(r_\Gamma^TS^{-1}r_\Gamma\). This provides a natural accuracy budget for inexact interface work.

There are \((p-1)^d\) bubble unknowns per tensor element and \((p+1)^d-(p-1)^d\) retained coefficients. In 2D the retained count is \(4p\), versus \((p+1)^2\) original unknowns. Elimination removes 25% at p3, 44.4% at p5 and 56.25% at p7. At p1 it removes nothing. In 3D the retained count is \(6p^2+2\); local dense solves grow faster and may become more suitable for tensor hardware, but their construction and factorization must be charged.

This is condensation of the **existing discontinuous basis**: two elements retain distinct copies of their boundary coefficients. It is not a claim that all methods can be converted to a single-valued HDG trace merely by merging nodes. Eliminating bubbles can also fill trace blocks between elements two face hops apart, because they couple through the same eliminated interior. A smaller vector does not necessarily imply less matrix storage or better conditioning. Matrix-free application of the Schur complement is an alternative to storing that fill.

A second exact basis trades fewer eliminated unknowns for less interface fill. For p≥3, use four cubic endpoint value/derivative functions in each coordinate, supplemented by \(x^2(1-x)^2P_j(2x-1)\), \(0\le j\le p-4\). Tensor products of the latter vanish in both value and normal derivative on every face. These stronger bubbles have no SIPG coupling to **any** coefficient on another element. Their elimination therefore preserves the one-hop element graph. There are only \((p-3)^d\) such interior unknowns, compared with \((p-1)^d\) zero-trace LGL bubbles. This is an exact alternative representation with more retained arithmetic but potentially less communication and storage. The prototype measures both; the basis conditioning and smaller elimination fraction are costs, not details to hide.

Hybridized mixed and DG methods provide related formulations whose global unknowns are traces and whose element fields are recovered locally. Their bilinear forms and numerical fluxes must be matched explicitly before claiming equivalence to this application's SIPG matrix. [Cockburn, Gopalakrishnan and Lazarov, unified hybridization](https://doi.org/10.1137/070706616)

For a coupled implicit or space-time stage, the same principle can act on the **joint** Euler/closure interior. If \(R_E(U,\sigma)\) is the transport residual and \(G=(\partial_U A)\sigma-\partial_U b\), its local Newton block has the schematic form

\[
J_{II}=\begin{bmatrix}M/\Delta t+\partial_U R_E&\partial_\sigma R_E\\G&A\end{bmatrix}_{II}.
\]

Given the required face-time traces, local elimination leaves \(J_{\Gamma\Gamma}-J_{\Gamma I}J_{II}^{-1}J_{I\Gamma}\). This is a route to substantial local matrix work that retains the chosen coupled stage equations. It requires local Jacobian invertibility, accurate nonlinear iteration and correct interface residuals; its interface matrix is generally not SPD. The scalar frozen-closure probe below does **not** implement or validate this joint space-time method. It is an algebraic next step justified by the local dependence structure, not an ADER stability or performance claim.

## 3. Robin transmission: an a priori estimate of the remaining difficulty

There is an especially concrete calculation before any implementation. On a one-dimensional element of width h with constant c and k, set \(z=h/\sqrt\alpha\) and \(d=\sqrt\alpha/\rho\). The homogeneous equation's exact endpoint Dirichlet-to-Neumann matrix is

\[
D_K=d\begin{bmatrix}\coth z&-\operatorname{csch}z\\
-\operatorname{csch}z&\coth z\end{bmatrix}.
\]

A local particular solution supplies the source contribution. All volume work is local; the endpoint matrices are then coupled by continuity and flux balance. The ratio of opposite-face to self-face coupling is \(\operatorname{sech}z\). For \(\alpha=C h^2\), that ratio is approximately 0.0845 at C=0.1, 0.648 at C=1, and 0.952 at C=10. Thus screening over roughly one element does **not** make opposite faces weakly coupled. This predicts why a few-element patch can be much more effective than independent elements. Increasing p at fixed h and alpha does not change this low-mode transmission. Choosing \(\sqrt\alpha=h/(p+1)\) would reduce it to \(\operatorname{sech}(p+1)\), but also changes the regularization and does nothing by itself to resolve its sharper source/profile within the element.

On a flat interface with constant coefficients, take a tangential Fourier mode \(e^{i\xi\cdot y}\). A decaying homogeneous solution has normal decay

\[
\mu(\xi)=\sqrt{\alpha^{-1}+|\xi|^2},\qquad
d(\xi)=k\mu(\xi)=\frac{\sqrt\alpha}{\rho}\sqrt{1+\alpha|\xi|^2}.
\]

Here \(d\) is the Dirichlet-to-Neumann symbol: it maps interface values to outward conormal flux. An exact exterior DtN map would supply the missing exterior response. In more than one dimension that map is generally nonlocal *along the face*, even though it is smaller than a volume operator.

For a Robin transmission parameter \(\beta>0\), the magnitude of the reflected error mode in this model is

\[
q(\xi)=\left|\frac{\beta-d(\xi)}{\beta+d(\xi)}\right|.
\]

Two equal subdomain exchanges give the squared factor. An overlap of physical width \(\delta\) additionally attenuates a crossing by \(e^{-\mu\delta}\). If the relevant DtN eigenvalues lie in \([d_{\min},d_{\max}]\), the best scalar parameter for this idealized interval is

\[
\beta_* = \sqrt{d_{\min}d_{\max}},\qquad
q_* = \frac{\sqrt{d_{\max}}-\sqrt{d_{\min}}}
{\sqrt{d_{\max}}+\sqrt{d_{\min}}}.
\]

This predicts why matching only the zero tangential mode with \(\beta=\sqrt\alpha/\rho\) becomes less effective as p increases. One scalar cannot match the growing tangential spectrum. A small polynomial/rational approximation to the DtN map, or a dense face operator assembled from local elimination, is a mathematically motivated alternative. Discrete extremal frequencies can scale differently from the informal resolved-wave estimate p/h; actual face spectra must determine the parameters. Curved faces, variable density, finite subdomains and cross-points invalidate a direct uniform convergence claim from the flat two-subdomain formula. [Optimized Schwarz analysis](https://doi.org/10.1137/S0036142903425409), [cross-point limitations](https://doi.org/10.1137/110837218)

There is a subtle consistency trap. Imposing \(t_K+\beta\sigma_K=\beta\lambda\) with \(t_K=k\partial_{n_K}\sigma_K\), a common \(\lambda\), and only \(t_K+t_L=0\), yields

\[
\sigma_K-\sigma_L=-2t_K/\beta.
\]

That is finite interface resistance, not exact continuity. In an exact Robin Schwarz iteration, incoming data are exchanged as \(r_K=\beta\sigma_L-t_L\); imposing both sides' relations at convergence recovers both continuity and flux balance. A consistent HDG formulation instead derives its mixed local equations and numerical flux, for example \(\widehat q\cdot n=q\cdot n+\tau(\sigma-\lambda)\). It should not be inferred from the oversimplified common-target Robin construction.

## 4. What an intentionally confined discrete response changes

Let \(D\) be the element block diagonal of the exact Schur matrix and \(S=D+E\). The simple family

\[
S_\eta=D+\eta E=(1-\eta)D+\eta S,\qquad0\le\eta\le1,
\]

stays SPD. At \(\eta=0\), the retained trace blocks are independent; at \(\eta=1\), the original discrete closure is recovered. The exact bubble forcing and recovery still involve neighboring trace data. The prototype therefore describes \(\eta=0\) as independent *trace block solving*, not complete element independence.

For an attenuated solution \(\lambda_\eta\), its residual against the original interface equation is

\[
S\lambda_\eta-g=(1-\eta)E\lambda_\eta.
\]

This is a computable consistency defect, not a solver tolerance that disappears by solving \(S_\eta\) more accurately. Positivity of the altered quadratic form does not preserve the original IGR closure, the discrete constant response, or the coupled Euler shock selection. Such a change needs a mesh/p/alpha consistency analysis and full-energy evolution assessment before production adoption.

An alternative that retains the target equation uses a fixed number of local interface corrections \(\lambda^{j+1}=\lambda^j+\omega D^{-1}(g-S\lambda^j)\). If the generalized eigenvalues of \((S,D)\) lie in \([a,b]\), optimal scalar damping gives contraction \((b-a)/(b+a)\). Chebyshev corrections give the familiar bound \(2[(\sqrt\kappa-1)/(\sqrt\kappa+1)]^m\) in the appropriate preconditioned energy norm. Fixed m yields bounded graph reach but an inexact closure. The probe supplies exact spectral bounds as an oracle diagnostic; obtaining reliable bounds in an application has a cost.

The location of the local blocks matters as much as their size. SIPG's stiff penalty directly couples copies of the same physical trace value across a face. An element-block inverse leaves those coupled copies in different blocks. A tiny block spanning the matching copies instead solves that stiff jump mode locally. The prototype groups duplicate LGL trace nodes, with up to four coefficients meeting at a vertex on the tested Cartesian meshes and five on the cylinder mesh. This is still a preconditioner for the exact Schur operator. Its purpose is to change the interface spectrum, not to attenuate physical communication. A curved mesh needs topological matching; interpolated coordinates alone are insufficient when the geometry order exceeds the closure order.

A more defensible changed discretization scales the complete face form rather than selected Schur blocks. Let \(A_0=R+K_{\rm volume}\) contain the production reaction and within-element diffusion forms, and let \(A\) be the production SIPG matrix. Then

\[
A_\theta=(1-\theta)A_0+\theta A
=R+K_{\rm volume}+\theta K_{\rm faces},\quad0\le\theta\le1.
\]

Both endpoints are SPD because reaction is positive, so the convex family is SPD. For natural exterior conditions, \(A_\theta\mathbf1=R\mathbf1\): the original constant response survives. At theta=0, each element solves an independent reaction-diffusion problem with natural internal boundaries. At theta=1, the current DG closure is recovered. Intermediate theta changes consistency at faces; it is not exact instantaneous IGR merely because it preserves constants. Its original residual is \((1-\theta)(A-A_0)\sigma_\theta\). The probe keeps the source unchanged and measures that defect against the original solution.

## 5. Prototype and rejection criteria

`tests/research/igr_face_condensation_probe.py` reads the sampled production matrix, production closure RHS, physical element masses, element DOFs and reference nodes. It constructs no alternative Euler or source physics. It checks both LGL and value/derivative basis transformations, cross-element bubble locality, exact reconstructed residual, physical L2 error, and the Schur energy identity. It compares element-block CG on the full and condensed systems, fixed optimally damped interface sweeps, and the deliberately altered \(S_\eta\) family. It reports interface fill and dense probe construction work. All timings are host research costs, not GPU performance estimates.

The first rejection criteria are straightforward: no bubble dimension to remove at the target p; loss of original-operator accuracy; large Schur fill; or an interface spectrum that still requires many sequential exchanges. A promising result must survive matrix-free/tensor versus ordinary-CUDA accounting with current coefficients, local factor construction, packing, interface residuals, reconstruction and complete time advancement. Static condensation and Robin transmission are established methods. An IGR-specific face approximation tied to screening and controlled temporal closure error remains a research hypothesis, not an established novel method.

## 6. First measured results and interpretation

Frozen 4×4 prescribed tanh compression operators use the production full-energy source, positive density, exact SIPG faces, natural exterior boundaries and alpha=h². The p3/p5/p7 systems have 256/576/1,024 closure unknowns. The saved planar case instead uses an evolved application shock state; its strongly stretched 32×1 mesh and source spectrum differ from the square-grid probes. Every reported CG count below is stopped at the same original GL matrix relative residual, 1e-8, rather than a basis-dependent Schur residual.

| Frozen state | Full operator, element blocks | Condensed, element trace blocks | Condensed, matching face-node blocks |
|---|---:|---:|---:|
| Tanh p3 | 76 | 75 | 17 |
| Tanh p5 | 113 | 109 | 25 |
| Tanh p7 | 145 | 138 | 33 |
| Saved planar shock p3 | 59 | 61 | 68 |
| Tanh p3, C=0.1 | 49 | 49 | 14 |
| Tanh p3, C=10 | 88 | 85 | 19 |

Pure condensation barely improves the difficult interface spectrum. The tiny face-node blocks change the p3/p5/p7 generalized condition numbers from 402/882/1,544 to 26.3/46.6/70.5, with only 420/644/868 entries across their local factors. The improvement in these square-mesh iteration counts is not a measured GPU speedup. The saved planar case reverses the ranking even though the worst-case condition number improves: eigenvalue distribution and the actual RHS matter. Adding element-block and face-node inverses did not resolve that limitation; it required 37/50/60 iterations on the tanh cases and 62 on the saved planar case. That combination is not pursued further.

LGL condensation removes 25%, 44.4% and 56.25% of unknowns. Explicit Schur block storage changes to 1.160, 0.637 and 0.395 times original block storage. At p3 the extra two-hop fill outweighs the smaller block dimension. The transformed cross-element bubble coupling is below numerical roundoff relative to the matrix. Exact recovery gives original residuals from 2.4e-14 to 1.7e-12 across these LGL cases, independently checked against the original sampled matrix and physical mass norm.

The alternative Hermite value/derivative split preserves the one-hop graph, as derived, but its condition and iteration counts do not improve. At p7 its coordinate transformation condition is about 9.1e4, versus 7.32 for LGL; recovered FP64 residual reaches about 6.3e-11. This particular unscaled representation is not a promising implementation candidate. A different normalization would alter its coordinate condition but does not remove the observed lack of useful interface elimination.

Fixed optimally damped trace Jacobi remains inaccurate: on the p3 tanh case, 32 sweeps retain 79.2% physical closure L2 error. A combined condensation/damped-wave recurrence with exact spectral bounds needs approximately 256 sampled sweeps to pass 1e-8 in that case. These are screening results with oracle bounds, not an argument to replace the current certified solve with fixed work.

Applying Chebyshev correction with the better face-node blocks needs 49/67/83 stages for the p3/p5/p7 tanh cases, versus 17/25/33 face-block CG iterations. At C=0.1 it needs 17 stages; at C=10 it needs 155. The latter contrast with 19 CG iterations illustrates why an extremal spectral interval can be conservative for a particular RHS. The recurrence itself requires no dot products, but its oracle spectral setup and the prototype's every-step certification are explicitly extra work. Whether avoiding reductions outweighs the additional operator applications is an implementation question.

Naively scaling Schur off-element blocks fails even when SPD is retained. On the p3 tanh operator, eta=0.75 gives 92.9% physical closure L2 error; eta=0.99 gives 61.9%; eta=0.999 still gives 15.5%. The SIPG diagonal and cross-element terms contain large cancellations for an almost continuous response. Scaling only one part destroys that balance. This rejects this specific confinement family, not all variationally consistent face formulations.

The complete-face family \(A_\theta\) is much better behaved. At theta=0, independent Neumann element solves still give 74.0% closure L2 error on the tanh and saved planar cases, and 79.5% on the saved cylinder. At theta=0.5 the errors are 0.2485%, 0.3069% and 2.036%; at theta=0.75 they are 0.0623%, 0.0734% and 0.6298%. At theta=0.99 they remain 1.77e-5, 2.15e-5 and 1.77e-4 relative. The constant-response residual is below 1.3e-13 across the sampled cases. This family merits a consistency/accuracy-cost analysis, but substantial confinement has a measurable closure error; solving the modified matrix tightly does not remove it.

The actual curved-cylinder p1 state has 5,288 closure DOFs, so no bubble condensation occurs. A sparse exact GL-to-LGL transformation still enables physical **vertex** blocks: 1,414 blocks with up to five coefficients and 20,440 inverse entries. The preconditioner is \(P=T\,\operatorname{blockdiag}_v(\widetilde A_{vv}^{-1})T^T\), an SPD algebraic approximation to the original inverse. The topology, rather than interpolated geometry, determines the vertex grouping.

| Saved curved-cylinder p1, original residual 1e-8 | Point Jacobi CG | Element-block CG | Vertex-block CG |
|---|---:|---:|---:|
| Zero initial closure | 175 | 101 | 21 |
| Prior exact closure, prescribed 0.1% density perturbation | 104 | 60 | 13 |

The warm test starts with original relative residual 4.74e-4; it is a controlled frozen-state perturbation, not a recorded time-stage pair. Final physical closure L2 errors are around 1e-9, against an independently checked sparse FP64 reference with residual 2.1e-14. These counts justify an application implementation experiment. They do not by themselves establish accurate-horizon performance or tensor-core use. At this low order, the local blocks are particularly suitable for ordinary CUDA; larger useful local stage blocks would be needed to justify matrix hardware.

The build and frozen export use the repository's canonical `build.sh` route. Analyze already-exported, source-identified matrices with NumPy and SciPy:

```bash
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_face_condensation_probe.py \
  /tmp/igr-local-four/tanh-p3.json /tmp/igr-local-four/tanh-p5.json \
  /tmp/igr-local-four/tanh-p7.json /tmp/igr-local-four/planar-p3.json \
  /tmp/igr-local-four/tanh-p3-c01.json /tmp/igr-local-four/tanh-p3-c10.json \
  --output /tmp/igr-local-four/face-small-results.json
```

The output records export and script SHA256, configuration metadata, numerical checks, host probe cost, and all failed or inaccurate variants. Host dense eigensolves and global inverses are bounded reference work, not proposed production storage. This probe does not measure full time advancement or GPU performance; application measurements belong in the accompanying campaign report.

For the topology-aware sparse cylinder and complete-face-family screen, use the v2 exports:

```bash
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_face_condensation_probe.py \
  /tmp/igr-local-four/v2/tanh-p3.json /tmp/igr-local-four/v2/planar-p3.json \
  /tmp/igr-local-four/v2/cylinder-p1.json \
  --output /tmp/igr-local-four/face-v2-results.json
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_face_condensation_probe.py \
  /tmp/igr-local-four/v2/cylinder-p1-warm.json \
  --previous-export /tmp/igr-local-four/v2/cylinder-p1.json \
  --output /tmp/igr-local-four/face-v2-cylinder-warm.json
```
