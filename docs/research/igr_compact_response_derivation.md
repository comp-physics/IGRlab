# Finite-range IGR response: derivation and bounded research probe

This note separates three objects: the existing instantaneous IGR equations, a finite-range numerical approximation to their inverse, and a deliberately changed continuum response. No changed continuum response is adopted by the application. A symmetric positive response does not by itself establish the original geometric derivation, entropy behavior, shock selection, or time-dependent stability.

The test-only implementation is [igr_compact_response_probe.py](../../tests/research/igr_compact_response_probe.py). It consumes frozen production matrices, uses the existing physical mass matrices for error norms, and compares with a current-operator FP64 solve. It is a mathematical and model-effect screen, not a GPU performance benchmark.

## A compact continuum response with an explicit uniform error bound

Start with constant density in one dimension, so the scaled closure is

\[
(1-\ell^2\partial_{xx})\sigma=f,\qquad \ell=\sqrt\alpha.
\]

The inverse has Fourier multiplier \(\widehat G(k)=1/(1+\ell^2k^2)\) and kernel \(G(x)=e^{-|x|/\ell}/(2\ell)\). The response is not compactly supported. However, its inverse factorization suggests a specific compact approximation: truncate the *one-sided factor*, rather than abruptly truncating the final Green function.

Let

\[
g_L(x)=e^{-x/\ell}\mathbf1_{[0,L]}(x),\qquad
K_L=\frac{g_L*\widetilde g_L}{(\int g_L)^2},\quad
\widetilde g_L(x)=g_L(-x).
\]

For \(r=|x|\le L\), direct integration gives

\[
K_L(r)=\frac{e^{-r/\ell}-e^{-(2L-r)/\ell}}
 {2\ell(1-e^{-L/\ell})^2};\qquad K_L(r)=0\quad(r>L).
\]

This response is nonnegative, even, has unit integral, and is positive definite because its Fourier multiplier is a squared modulus. With \(q=e^{-L/\ell}\),

\[
\widehat K_L(k)=
\frac{1+q^2-2q\cos(kL)}{(1-q)^2(1+\ell^2k^2)}.
\]

Consequently,

\[
0\le\frac{\widehat K_L}{\widehat G}-1
=\frac{2q(1-\cos(kL))}{(1-q)^2}
\le\frac{4q}{(1-q)^2}=:\varepsilon_L.
\]

The bound is uniform in frequency: it applies to high-frequency loads as well as smooth ones. On the line, for sufficiently regular loads, \(\|K_L*f-G*f\|_{L^2}\le\varepsilon_L\|G*f\|_{L^2}\), and the same relative bound holds in the energy norm of \(1-\ell^2\partial_{xx}\). The current elliptic residual also satisfies \(\|(1-\ell^2\partial_{xx})(K_L*f)-f\|_{L^2}\le\varepsilon_L\|f\|_{L^2}\). These statements follow from the Fourier multiplier and Plancherel; they do not assume an evolving solution remains stable under the changed response.

The required radius is explicit:

\[
L/\ell\ge 2\operatorname{asinh}(\varepsilon^{-1/2}).
\]

| Uniform relative response budget | Radius \(L/\ell\) |
|---:|---:|
| 10% | 3.737 |
| 1% | 5.996 |
| 0.01% | 10.597 |
| 0.0001% | 15.202 |

Thus strict finite support and controlled instantaneous response error are mathematically compatible. They do not imply a very small support at a stringent tolerance. When \(\alpha=C h^2\), the number of physical cells in this radius scales with \(\sqrt C\), independently of global domain size. Confinement to one cell requires either weaker regularization, a much looser error budget, or an interface correction.

The compact kernel retains the screened kernel's \(k^{-2}\) high-frequency scaling, with a bounded oscillating multiplier. Its second moment is

\[
\int x^2K_L(x)\,dx=2\ell^2-
\frac{2e^{-L/\ell}L^2}{(1-e^{-L/\ell})^2}.
\]

It preserves the constant response exactly but does not exactly preserve the screening kernel's second moment. This matters when comparing widths or moments over a long tail.

This construction is a factorization-based research candidate, not a novelty claim. Autocorrelation and compact positive kernels are established mathematical tools. Directly applying this translation-invariant kernel to variable-density, multidimensional IGR would be unjustified. A product of coordinate-wise copies would introduce anisotropy and mixed derivative terms; it is not the isotropic screened inverse. Density scaling on only one side of a kernel would generally lose self-adjointness. Even for constant \(\alpha\) in one dimension, writing \(w=\rho^{-1/2}\) gives

\[
\rho^{-1}-\alpha\partial_x(\rho^{-1}\partial_x)
=w\left[1-\alpha\partial_{xx}+\alpha w''/w\right]w.
\]

Ignoring the induced potential \(\alpha w''/w\) is a substantive approximation near a shock. The constant-density error bound cannot simply be assigned to the actual variable-density closure.

## A finite-wave-horizon extension retains the bound for variable density

There is a stronger extension than reusing the one-dimensional kernel. Decompose the current weak operator as \(A=R+K\), where \(R\succ0\) is the reaction operator and \(K\succeq0\) is the diffusion operator with its consistent face and boundary terms. Define

\[
H=R^{-1/2}KR^{-1/2},\qquad
A^{-1}=R^{-1/2}(I+H)^{-1}R^{-1/2}.
\]

For dimensionless artificial-wave horizon \(T\), set \(q=e^{-T}\) and

\[
F_T(H)=\frac{(1+q^2)I-2q\cos(T\sqrt H)}{(1-q)^2(I+H)},
\qquad B_T=R^{-1/2}F_T(H)R^{-1/2}.
\]

Spectral calculus gives the same relative bound for every nonnegative eigenvalue of \(H\):

\[
1\le (1+\lambda)F_T(\lambda)
\le1+\varepsilon_T,\qquad
\varepsilon_T=4e^{-T}/(1-e^{-T})^2.
\]

Hence \(B_T\succ0\), \(\|B_Tb-A^{-1}b\|_A\le\varepsilon_T\|A^{-1}b\|_A\), and \(\|AB_Tb-b\|_{A^{-1}}\le\varepsilon_T\|b\|_{A^{-1}}\). The corresponding relative estimates also hold in the reaction norm \(\|\cdot\|_R\) for the response and \(\|\cdot\|_{R^{-1}}\) for the residual. These residual statements are not assertions about the unweighted Euclidean norm under variable coefficients. If \(K\mathbf1=0\), then \(B_T(R\mathbf1)=\mathbf1\), so the weak constant balance is exact. These are matrix statements valid in any dimension under the specified SPD/PSD assumptions.

The key localization identity is

\[
F_T(H)=\int_0^T
\frac{e^{-t}-e^{t-2T}}{(1-e^{-T})^2}\cos(t\sqrt H)\,dt.
\]

It follows by integrating the scalar cosine identity and applying spectral calculus. In the continuum, this amounts to evolving the homogeneous artificial wave equation

\[
a(x)v_{tt}-\nabla\cdot(\alpha(x)a(x)\nabla v)=0,
\quad a=1/\rho,\quad v(0)=b/a,\quad v_t(0)=0,
\]

and accumulating the weighted time integral. For positive bounded coefficients and local admissible boundary conditions, the wave equation has finite propagation. A local energy estimate bounds the propagation cone by \(\sqrt{\alpha_{\max}}T\); under the regularity needed to define the intrinsic distance, the sharper cone uses travel time \(\int ds/\sqrt\alpha\). Density affects amplitudes and reflections, while the principal wave speed is \(\sqrt\alpha\). Thus the continuum response is genuinely finite range even with variable density and in multiple dimensions. The general connection between cosine functional calculus and finite spatial propagation is established theory; see [finite propagation and spectral multipliers](https://link.springer.com/article/10.1007/s00209-017-1986-9).

This construction can be interpreted as a changed response with a prescribed error budget, or as an approximate inverse inside corrections to the original closure. It does not evolve IGR in physical time. A finite \(T\) by itself does not exactly reproduce instantaneous IGR.

There is an important computational obstruction: the exact wave propagator of a finite-dimensional DG matrix is generally dense at any nonzero time. Continuum finite propagation does not imply strict support for the exact semidiscrete cosine. An explicitly local wave integrator has bounded graph support, but its time-discretization and integration errors require a separate budget. Direct central differences require \(\Delta t\le2/\sqrt{\lambda_{\max}(H)}\), hence at least \(T\sqrt{\lambda_{\max}(H)}/2\) stages for stability alone. Accuracy requires more. The usual high-order DG inverse estimate can make this scale like \(Tp^2\sqrt C\) when \(\alpha=C h^2\).

The probe evaluates \(F_T\) exactly on the small frozen matrices as a spectral oracle and reports this stability-only stage estimate. It does not count a dense eigendecomposition as a viable application implementation. Local exact interior propagation, face Schur corrections, or batched local space-time solves are possible ways to attack the stage cost; their usefulness remains to be measured.

An independently checked discrete counterpart removes the exact-cosine support issue. For the reaction-scaled spectrum \(\lambda\in[1,\Lambda]\), let \(z=(\sqrt\Lambda-1)/(\sqrt\Lambda+1)\), \(a=(\Lambda+1)/2\), \(b=(\Lambda-1)/2\), and \(C=(a-\lambda)/b\). The response

\[
p_{m-1}(\lambda)=
\frac{1+z^{2m}-2z^m T_m(C)}{(1-z^m)^2\lambda}
\]

is exactly a degree-\(m-1\) polynomial: its numerator vanishes at \(\lambda=0\). Its relative response lies in \([1,1+4z^m/(1-z^m)^2]\), and \(p_{m-1}(1)=1\). This is the squared finite geometric-series, or Poisson-kernel, analogue of the wave window. With a block-local reaction inverse, it has strict \(m-1\)-layer discrete graph support. Its degree still scales approximately as \(\tfrac12\sqrt\Lambda\log(4/\varepsilon)\). It is not an escape from the condition-number cost. The parent investigation prototypes this alongside ordinary Chebyshev iteration; the filter algebra itself is not claimed novel.

## Why smooth compact kernels can fail a shock-frequency screen

As a second, independently useful control, consider the normalized one-dimensional Wendland kernel

\[
W_R(x)=\frac{3}{2R}(1-|x|/R)_+^4(1+4|x|/R).
\]

Its zeroth, second, and fourth moments are \(1,R^2/14,R^4/70\). Setting \(R=\sqrt{28}\ell\) matches the screened response's second moment, but its fourth moment becomes \(11.2\ell^4\), versus \(24\ell^4\) for the screened kernel. Therefore

\[
\widehat W_R(k)-\widehat G(k)
=-\frac8{15}(k\ell)^4+O((k\ell)^6).
\]

The high-frequency response decays as \(k^{-4}\). Matching the long-wave width therefore does not match sharp-source behavior. Direct quadrature gives a 28.6% transfer error at \(k\ell=1\). For Gaussian source widths \(0.25\ell,\ell,4\ell\), the response L2 errors are 26.2%, 11.0%, and 0.372%, respectively. These are constant-density response errors, not shock solution errors.

The positive-definite compact radial function is established prior art in dimensions up to three; its use here does not establish a new IGR model. See [Wendland's original construction](https://doi.org/10.1007/BF02123482). Positive covariance tapering is also established prior art: [Furrer, Genton, and Nychka](https://marcgenton.github.io/2006.FGN.JCGS.pdf).

## Frozen production matrices: two finite-range SPD constructions

Let the current assembled closure be \(A\sigma=b\), with \(A=A^T\succ0\). Let \(\mathbf1\) denote the constant nodal field and \(m=A\mathbf1\). Under the natural boundary treatment, \(m\) is the reaction load. The probe checks this against the exported reaction blocks.

### Tapering the exact inverse: an oracle, not a scalable setup

Set \(C=A^{-1}\), form \(T_{ij}=\phi(|x_i-x_j|/R)\), and define \(B=C\circ T\). A positive semidefinite taper with positive diagonal makes \(B\) positive definite by the Schur product theorem. Duplicate DG coordinates do not invalidate this result: \(T\) can be semidefinite while \(C\circ T\) remains definite. The response has strict physical range \(R\).

This construction screens the effect of enforcing finite range on the actual production closure, including variable density, order, and faces. It requires an exact inverse to construct and is therefore restricted to small matrices. Its construction time and storage are reported explicitly. It is not a scalable algorithm or a speedup candidate as implemented.

Tapering generally violates \(Bm=\mathbf1\). Row normalization can repair the constant response but generally destroys symmetry; the probe does not silently apply it. Positive definiteness is a quadratic-form property, not entrywise positivity, and does not imply positive sigma for arbitrary loads. In multiple dimensions the IGR source itself can be sign-indefinite.

### A locally constructed, moment-preserving symmetric patch response

For overlapping element patches \(P\), let \(R_P\) restrict to the patch, \(A_P=R_PA R_P^T\), and choose positive partition-of-unity weights \(\chi_P\) such that \(\sum_P R_P^T\chi_P=\mathbf1\). The probe uses inverse overlap counts. Begin with

\[
H_P=D_{\sqrt{\chi_P}} A_P^{-1} D_{\sqrt{\chi_P}}\succ0.
\]

Set \(y_P=R_Pm\), \(s_P=\chi_P\), and \(c_P=y_P^Ts_P\). If \(c_P>0\), define the local inverse-secant update

\[
\widetilde H_P=
\left(I-\frac{s_Py_P^T}{c_P}\right)H_P
\left(I-\frac{y_Ps_P^T}{c_P}\right)
+\frac{s_Ps_P^T}{c_P}.
\]

This is the inverse-BFGS algebra applied to a response constraint. It satisfies \(\widetilde H_Py_P=s_P\) and remains SPD. For any nonzero \(z\), the rank-one term is positive when \(s_P^Tz\ne0\); otherwise the first term reduces to \(z^TH_Pz>0\). Summing the embedded local maps gives

\[
B=\sum_P R_P^T\widetilde H_P R_P\succ0,
\qquad Bm=\mathbf1.
\]

The moment condition has a useful consequence for *every* load:

\[
m^T(Bb)=\mathbf1^Tb.
\]

Thus the weak constant balance is preserved exactly without a global normalization or a global rank-one correction. Patch radii of \(r\) face-neighbor layers yield a response supported within at most \(2r\) graph layers. At \(r=0\), each response is element independent; the artificial-boundary diagonal is retained in \(A_P\), and no claim of accurate shock coupling follows from this moment property.

Positive definiteness and one exact moment do not bound the remaining modes. The probe measures load-dependent physical/energy errors and, on small cases, the eigenvalues of \(L^TBL\), where \(A=LL^T\). Then

\[
\|I-BA\|_A=\max_j|1-\lambda_j(L^TBL)|.
\]

This separates a good response for one particular shock source from an operator approximation that is uniformly useful. The same spectrum predicts the optimal damped Richardson contraction \((\lambda_{\max}-\lambda_{\min})/(\lambda_{\max}+\lambda_{\min})\) if the map is used as a solver. A one-shot map changes the discrete closure; converging corrections against the current \(A\) recovers the existing target equations.

## Costs and limits for tensor execution

Symmetric patch maps produce dense local matrix operations with strict bounded support. They also compute redundant overlapping outputs and require gathering/scattering or reductions. The probe reports actual patch matrix products, union nonzeros, full stored local matrix bytes, and hypothetical 16-by-16 padding separately. Cholesky factorization FLOPs exclude the additional triangular solves used to materialize each inverse; the measured CPU setup includes them. No tensor throughput is inferred from the CPU diagnostic.

One current RHS per patch remains a matrix-vector workload. Multiple temporal RHS or local nonlinear corrections could improve tensor utilization only if their construction, accuracy, factor reuse, and additional work are favorable. Density changes can force reconstruction. The constant-response correction is inexpensive once the local inverse exists but does not remove the local factorization cost.

The decisive next screen for the finite-wave-horizon response is a numerically local evaluation whose total error respects the spectral bound without an excessive number of high-order stages, followed by a short fully coupled shocked run. The decisive screen for the patch maps is whether their one-shot error is acceptable at a useful radius; otherwise they remain preconditioners. Neither a small frozen residual nor an SPD map alone qualifies a changed continuum model for production.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_compact_response_probe.py \
  --self-test --output /tmp/igr-compact-continuum.json

# INPUT is a frozen production JSON exported by the locality probe.
OPENBLAS_NUM_THREADS=1 python3 tests/research/igr_compact_response_probe.py \
  --input "$INPUT" --patch-radii 0 1 2 --taper-radii 1 2 4 8 \
  --output /tmp/igr-compact-frozen.json
```

Each result records the exact script and input SHA-256 identities. The script limits the global inverse oracle to 1024 DOFs and patch dimension to 512 by default. Larger cases use a sparse FP64 reference solve and locally constructed patches; they skip global inverse tapering and dense spectral analysis. The input mass matrices provide the physical L2 norm. No application code, build route, or continuum target is changed by running this Python screen.

## Frozen-operator outcomes

The spectral oracle was tested on production broken-DG operators with current variable density. At \(T=6\), the rigorous relative energy bound is 0.9964%:

| Frozen case | Scalar DOFs | Measured energy error | Physical L2 error | Minimum central-difference stages for stability only |
|---|---:|---:|---:|---:|
| Tanh front, p3, C=1 | 256 | 0.5314% | 0.3381% | 200 |
| Tanh front, p5, C=1 | 576 | 0.5316% | 0.3389% | 438 |
| Tanh front, p7, C=1 | 1024 | 0.5315% | 0.3389% | 767 |
| Tanh front, p3, C=0.1 | 256 | 0.6810% | 0.6729% | 64 |
| Tanh front, p3, C=10 | 256 | 0.0965% | 0.0179% | 632 |
| Saved planar full-energy shock, p3 | 512 | 0.6241% | 0.6177% | 146 |

Constant-response and weak-balance errors were at most a few times \(10^{-12}\). The high-order generalized diffusion spectral maxima were 4437, 21232, and 65257 for p3, p5, and p7. The mathematically controlled response therefore has an expensive straightforward local wave evaluation. Dense oracle times are not application speedups. Individual source errors can be much smaller than the worst-case bound because the cosine phase and source spectrum interact; this is not a uniform improvement with order.

The simpler moment-preserving patch response failed decisively as a one-shot replacement. On the saved 5288-DOF curved-cylinder closure, radii 0, 1, 2, and 4 gave physical L2 errors of 79.5%, 27.8%, 17.3%, and 24.2%, despite weak-balance errors around \(10^{-15}\). Radius 4 required 29.82 million overlapping local products versus 2.53 million products in the assembled union, with maximum patch dimension 188. Its relative current closure residual was 6.49. Radius 2 had residual 14.7. Increasing overlap did not monotonically repair the response shape; this branch was stopped as a one-shot closure. These negative results do not contradict the bounded spectral filter, which constrains all eigenmodes rather than only one moment.
