# hp scaling and face-conditioned local correction

This research screen changes the scalar value passed to the existing production
closure to represent

\[
\alpha=C h_K^2/(p+1)^s,\qquad s\in\{0,2,4\}.
\]

It compares algorithms against an accurate solve of that **same current
operator and source**. Changing s changes the mesh-dependent regularization;
agreement with this reference is not evidence that the changed closure gives
an accurate evolving shock. The companion front and application experiments
must assess that separate question. All computations here use one CPU BLAS
thread, existing exported physics, and bounded dense numerical oracles.

## The useful local unit changes with the screening length

Let n=p+1 and \(\ell=\sqrt\alpha\). Then

\[
\ell/h_K=\sqrt C/n^{s/2},\qquad
\frac{\text{opposite-face DtN coupling}}{\text{self-face DtN coupling}}
=\operatorname{sech}(n^{s/2}/\sqrt C)
\]

for the constant-coefficient one-dimensional element problem. At s=2, C=1,
the second quantity is 0.0366, 0.00496 and 0.000671 for p=3,5,7. At s=4 it is
already approximately 2.25e-7 at p=3. These are exact elementary calculations
for that model, not universal bounds for curved, variable-density DG cells.

The distinction between *different faces of one element* and *the two sides
of one shared face* matters. Shortening ell suppresses transmission through
the element. It does not remove the need to couple the two boundary layers
meeting at a shared face. Both sides' local trace impedance scales with ell,
so their relative interaction can remain order one. After local interior
elimination, a good local unit can consequently be a paired face or vertex
block. Isolating each element is a much stronger approximation.

There are also two different resolution scales. Polynomial inverse estimates
give a high-mode diffusion/reaction ratio of order
\(\alpha n^4/h_K^2=C n^{4-s}\). Thus s=2 can shorten physical screening while
retaining increasing high-mode stiffness; s=4 can bound that stiffness. The
scale h/n is characteristic of bulk polynomial resolution, whereas h/n^2
occurs near element endpoints in clustered nodal representations. Resolving
a narrow layer at a face does not establish resolution of the same layer
moving through an element interior.

Tangential face modes suggest a further block choice. For a flat interface at
constant coefficients, the DtN symbol is proportional to
\(\ell\sqrt{1+\ell^2|\xi|^2}/\rho\). With the largest polynomial trace
frequency of order n^2/h, its highest/lowest impedance ratio is of order
\(\sqrt{1+C n^{4-s}}\): order n for s=2, and bounded for s=4. Pointwise
paired-face blocks capture the jump between sides but leave this tangential
structure to iteration. A whole-face block can capture it at additional local
matrix cost. Corners and heterogeneous coefficients still require testing;
this symbolic argument is not a uniform DG condition-number proof.

## Exact elimination and inexpensive face correction

The test changes GL coordinates to LGL traces and bubbles without changing the
space or quadrature. It eliminates the element-diagonal interior block and
solves the exact Schur system. Every iteration is judged using the original
GL current-operator residual, with the reconstructed sigma energy and physical
L2 errors also recorded. This avoids comparing basis-dependent stopping norms.

The completed screen found the following for the prescribed compression
on 4x4 quadrilaterals. Counts use relative current residual 1e-8.

| p | alpha exponent s | Face-node condition | CG iterations | Chebyshev, exact spectral interval | Chebyshev, local certified interval |
|---|---:|---:|---:|---:|---:|
| 3 | 0 | 26.34 | 17 | 49 | 104 |
| 5 | 0 | 46.64 | 25 | 67 | >256 |
| 7 | 0 | 70.48 | 33 | 83 | >256 |
| 3 | 2 | 2.92 | 13 | 15 | 27 |
| 5 | 2 | 3.62 | 14 | 17 | 28 |
| 7 | 2 | 4.57 | 14 | 19 | 35 |
| 3 | 4 | 3.07 | 10 | 15 | 27 |
| 5 | 4 | 2.85 | 9 | 14 | 25 |
| 7 | 4 | 2.71 | 9 | 13 | 23 |

Here the face metric groups polynomial trace coefficients sharing a physical
point. It does not identify or average their unknown values. This is the same
basic representation that previously worked in the p1 cylinder application,
now combined with p-scaled alpha and exact local interiors.

The improvement is not simply reduced dimension. At s=2, the element-trace
metric still has condition numbers 47.2/65.4/85.1 and needs 44/48/51 CG
iterations at p3/5/7. Correctly representing the shared-face interaction remains
valuable after shortening alpha.

On the saved planar shock state, the original s=0 ranking favored element
blocks for its particular source. Scaling alpha changed that ranking:

| p | s | Face-node CG | Element-trace CG | Face-node Chebyshev, oracle/local |
|---|---:|---:|---:|---:|
| 3 | 0 | 68 | 61 | 90/>256 |
| 5 | 0 | 105 | 69 | 132/>256 |
| 3 | 2 | 17 | 21 | 26/74 |
| 5 | 2 | 19 | 20 | 29/79 |
| 3 | 4 | 9 | 14 | 16/27 |
| 5 | 4 | 9 | 13 | 15/23 |

This source is the actual saved full-energy p2 polynomial evaluated at higher
closure orders; it is not a newly evolved p3/p5 flow. Its elements have strong
geometric anisotropy. Neither a condition-number trend nor a favorable source
proves a general solver ranking.

We also tested the tangential-block hypothesis directly. A whole-face block
contains all non-vertex trace coefficients on both sides of that physical
face, while separate blocks collect the vertex copies. The partition remains
disjoint and local. It improves some counts, at increased factor storage:

| Case | Paired-node CG / oracle Chebyshev / certified Chebyshev | Whole-face CG / oracle Chebyshev / certified Chebyshev | Node / face factor entries |
|---|---:|---:|---:|
| Compression p7, s2 | 14 / 19 / 35 | 13 / 18 / 26 | 868 / 4,228 |
| Compression p7, s4 | 9 / 13 / 23 | 8 / 11 / 19 | 868 / 4,228 |
| Planar p5, s0 | 105 / 132 / >256 | 69 / 89 / >256 | 1,012 / 3,292 |
| Planar p5, s2 | 19 / 29 / 79 | 14 / 25 / 45 | 1,012 / 3,292 |
| Planar p5, s4 | 9 / 15 / 23 | 8 / 13 / 21 | 1,012 / 3,292 |

The largest whole-face block is twelve unknowns at p7 and eight at p5.
This is a qualified improvement, not a demonstrated performance result.
At p3, s0, compression CG actually increased from 17 to 18 iterations despite
the improved worst-case condition number. Shared corners and within-element
trace interactions remain after whole-face factorization.

## How far local spectral certification gets

For a block metric B, whiten the Schur system to
\(H=B^{-1/2}SB^{-1/2}\). An exact eigensolve provides the deliberately charged
oracle interval. Two finite-neighborhood alternatives are tested:

1. Condense the reaction matrix R locally to obtain \(S_R\). If A-R is
   positive semidefinite, Schur monotonicity gives \(S\succeq S_R\). A local
   absolute-row bound on \(S_R^{-1/2}BS_R^{-1/2}\) gives a positive lower bound
   on H. An absolute-row bound gives its upper endpoint.
2. Refine that upper endpoint using \(\|H^k\|_\infty^{1/k}\), and obtain a
   two-sided interval around c from
   \(c\pm\|(H-cI)^k\|_\infty^{1/k}\), k=1,2,4,8. Symmetry makes these
   matrix-norm certificates valid. Matrix products enlarge the neighborhood.

The small production matrices separately audit A-R positivity and all claimed
intervals against independent eigenvalues. The random-SPD self-test checks the
same inequalities without using production structure. Exact spectral endpoints
and local matrix bounds are different setup costs; the output records both.

At s=0, the cheap local lower bound was extremely pessimistic: p3 face-node
condition 26.3 became a certified ratio of about 38,176. Eight powers reduced
this to 115, but already filled the 4x4 test domain. At p5/p7 it remained too
loose for 256 stages. This is a negative result for this particular cheap
certification route at the original alpha.

At s=2 and s=4, powers two to four usually gave useful bounds on the compression
case. The resulting 23–35 stages are plausibly schedulable with only endpoint
residual checks. They still exceed the 9–14 same-operator CG iterations, and
the matrix setup is not free. The anisotropic planar s=2 case needed 74–79
stages with local bounds, versus 17–19 CG iterations; that version is a weak
candidate. The s=4 case is much less sensitive to those bounds.

The prototype inspects every intermediate stage to find first passage. Those
are explicitly counted test-only checks. A proposed no-inner-reduction kernel
would select a fixed degree and retain a final current residual certificate;
no such GPU implementation or performance claim is made here.

## What independent elements still miss

Solving only each production volume-plus-reaction block changes the closure.
Measured errors relative to the exact same-alpha global solution are:

| Source | p | s | Sigma physical L2 error | Sigma current energy error |
|---|---:|---:|---:|---:|
| Compression | 3 | 2 | 19.4% | 133% |
| Compression | 5 | 2 | 13.1% | 112% |
| Compression | 7 | 2 | 9.80% | 97.9% |
| Compression | 3 | 4 | 6.02% | 15.4% |
| Compression | 5 | 4 | 1.26% | 3.34% |
| Compression | 7 | 4 | 0.679% | 1.77% |
| Saved planar shock | 3 | 4 | 12.7% | 36.8% |
| Saved planar shock | 5 | 4 | 8.21% | 24.1% |

The last two rows rule out a blanket claim that p scaling makes the closure
element independent. A small opposite-face DtN ratio does not control forcing
or trace layers immediately next to an interface. Energy error is especially
important because the flow uses closure derivatives and fluxes. All cases
require the corresponding advancing-flow check before an approximation is
accepted as useful IGR.

## Follow-up on evolved p-scaled shocked states

The first saved planar screen deliberately kept its old polynomial state while
varying alpha. A second screen removes that limitation: it exports the actual
p3 flow after evolution to t=6 with C_alpha=1/16 and volume LLF disabled, for
the originally face-aligned front and a front initially shifted by 0.4625.
Both use 32 elements, 512 closure unknowns, and the current production
operator and source. Input polynomial/mesh matching error is zero and sampled
operator action discrepancies are below 1.2e-15.

| Initial front phase | Node-face CG | Whole-face CG | Element-trace CG | Whole-face oracle / certified Chebyshev | Independent-element L2 / energy error |
|---|---:|---:|---:|---:|---:|
| Aligned | 17 | 15 | 21 | 24 / 47 | 29.6% / 186% |
| Shifted | 17 | 15 | 25 | 24 / 44 | 30.9% / 195% |

The whole-face condition numbers are 5.986 and 6.098. Its factors therefore
remain effective on the evolved, sharper sources. Independent-element current
residuals remain approximately 13.9 and 13.1 times the source norm. Removing
face coupling remains a poor approximation even though the physical screening
length is only one quarter of an element.

This is also a useful separation of solver and flow quality. The parent
application experiment found markedly different pressure behavior in the
aligned and shifted runs; the closure conditioning and iteration counts here
are almost unchanged. Accurate, inexpensive solution of the closure is not a
certificate that the coupled shock discretization is acceptable. See the
parent application accuracy evidence for that assessment.

## Cost and next experiment

The probe also constructs fixed degree-two/four/eight Chebyshev correction
matrices exactly, without dropping entries, and reports their setup products,
stored dense entries, and significant-entry count separately from recurrence
work. This is an algebraic representation screen. A precomposed matrix can
exchange repeated sparse passes for a larger local matrix, but its coefficients
depend on the current operator, and its support expands with degree. A dense
small-domain implementation is not evidence of scalable element-local work.

For compression p7, s2, two/four/eight stages need approximately
75,192/150,384/300,768 Schur matrix products if evaluated successively (before
metric and vector work). The explicit precomposed matrix stores 200,704
entries at every degree; respectively 39,268/175,780/200,700 exceed the
reported relative 1e-12 significance threshold. No entries are actually
dropped. Its dense coefficient construction requires at least
180/360/719 million products, plus the metric composition and vector algebra;
the measured cumulative host setup includes those costs. At degree eight it
is already essentially global across this small 4x4 domain. The planar p5,
s2 version stores 409,600 entries versus 220,864 successive Schur products at
eight stages. This stopped dense global precomposition as a directly useful
implementation. Local retained tiles or on-chip recurrence remain different
implementations to assess, especially when multiple right-hand sides reuse
their coefficients.

The most credible next combination is short-screened local interiors plus
paired face transmission, with either a small fixed correction degree and a
current residual gate or inexpensive CG. Changing the shock representation
must make that shorter screening accurately resolvable first. For tensor
hardware, batching larger local interior, face, or joint state/closure blocks
is an opportunity to measure, not a consequence of the iteration counts.

## Reproduction and identities

The complete screen has eighteen operators and four metrics per operator:
twelve compression combinations p1/3/5/7 with s0/2/4, and six saved planar
combinations p3/5 with s0/2/4. Every reference residual is checked below 1e-9;
the actual source, coefficient construction, and graph sampling are supplied
by the root production export experiment. This script builds no application.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 tests/research/igr_hp_face_probe.py --self-test
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 tests/research/igr_hp_face_probe.py \
  /tmp/igr-hp-breakthrough/tanh-p{1,3,5,7}-s{0,2,4}.json \
  /tmp/igr-hp-breakthrough/planar-p{3,5}-s{0,2,4}.json \
  --output /tmp/igr-hp-face-screen-final.json --compose
```

The random-SPD certificate self-test passed. All eighteen production spectral
audits and numerical-reference residual gates passed. Solver trials that did
not reach the current residual target remain marked unconverged. No GPU timing
was performed by this probe. CPU work paused during the root GPU measurements.
The script and all imported helpers are retained in the repository; each row
contains its exact input hash, source hash and numerical-oracle costs.

- Tested script SHA-256:
  `afd6b5625710f5729e12d09ab1ba2784037a7b3bb2e839f8a3b612dbb9728128`.
- Complete raw result SHA-256:
  `d30d5c6c6c67559b505b0b6a8032d38b5b083d3c91a16c3ec7f42a5c7b813554`.
- Raw output: `/tmp/igr-hp-face-screen-final.json`.
- Export commands: `/tmp/igr-hp-breakthrough/frozen-export-commands.json`.

The evolved-state follow-up used the unchanged tested script:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 tests/research/igr_hp_face_probe.py \
  /tmp/igr-hp-breakthrough/evolved-nollf-aligned.json \
  /tmp/igr-hp-breakthrough/evolved-nollf-shifted.json \
  --output /tmp/igr-hp-face-evolved.json --compose
```

- Follow-up result SHA-256:
  `5a62fe9a05bc204b71c7d4f9c1cb5908e41d493b6c082ed3f1ef723bd589ccf2`.
- Aligned operator SHA-256:
  `0bd543af6b1b14499a2f4c78c5b632c17863550157e5533f8d7ae9a24e354818`.
- Shifted operator SHA-256:
  `0f10f3f9953bbe08fc408dfb04f32a7e8178b8b36de3f81bb50271ab4009a10f`.

These hashes identify the tested screen independently of a later documentation
commit. The parent campaign's archived result should preserve this output
alongside its canonical build, source, configuration and application identities.
