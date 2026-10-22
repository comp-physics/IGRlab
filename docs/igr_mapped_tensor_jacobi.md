**Geometry-aware projected local tensor inverse — bounded prototype results**

The opt-in projected reference applies shared small tensor transforms on fixed mapped quadrilaterals.
Native qualification passed, and FP64 WMMA reduced the
measured large curved-mesh inverse application from 0.311760 ms to 0.205824 ms
(1.515×). These are local-inverse measurements. A short mapped Sedov comparison passes the
conserved-field budget; late-time accuracy and a matched whole-application speedup
have not been established.

Tested application source: `edba7971aacfa33b8886c94d67f38c6cbd9d6c23`. Job **3439089**, node **gh032**, used one GH200, canonical `build.sh`, CUDA SM90.
The native receipt binds the complete frozen source graph and application/fixture SHA256 identities.

Let $D_{0,e}$ be the exact initial volume block plus SIPG face self terms.
Normalize with the diagonal of the actual physical mass:

\[
m_{e,i}=\sum_q w_q\det J_e(q)\,\rho_e(q)^{-1}\phi_i(q)^2,
\quad S_e=\operatorname{diag}(m_{e,i}^{-1/2}),\quad H_{0,e}=S_{0,e}D_{0,e}S_{0,e}.
\]

Two squared-basis contractions evaluate this diagonal on the operator's actual
mass quadrature: GL7×GL7 for Q2 mapping, versus GL6×GL6 on the Q1 fixtures.
Initial normalized blocks are averaged within each physical-boundary mask in
local edge order, giving at most 16 classes. Partial traces select two symmetric 1D axes; their orthogonal eigenbases give $Q_c=Q_{y,c}\otimes Q_{x,c}$.
Their gauge-dependent eigensums need not be positive and are never inverted. For each element, setup computes $d_{e,k}=(Q_c^T H_{0,e}Q_c)_{kk}>0$ on the GPU. Runtime applies

\[
P_e=S_{\mathrm{current},e}Q_c\operatorname{diag}(d_{e,k}^{-1})Q_c^TS_{\mathrm{current},e}.
\]

This reference is SPD in exact arithmetic. It is the initial block inverse only
when $H_{0,e}$ is diagonal in the shared basis. General geometry introduces an
explicit initial-block approximation; heterogeneous evolving density adds the
current-mass surrogate approximation. A uniform density change scales this
projected inverse exactly, without making a general mapped block inverse exact.
Current $A$ and RHS retain their original FP64 physical geometry and coefficients.

Admission requires one native CUDA rank, conforming 2D scalar GL p5 L2 quads,
identity true/local layout, fixed Q1/Q2 mappings, fixed topology and alpha policy.
Positive finite signed Jacobians are checked at setup samples, including vertices,
mapping nodes, state nodes and mass points; positivity between samples is unproved.
Initial block symmetry and positive resolved projected diagonals are checked. There is no Cartesian alignment or separability requirement in projected mode.
`IGRLAB_IGR_LOCAL_TENSOR_REFERENCE=projected` selects it; `exact` remains the default
with its original Cartesian/separability admission. Shared dense execution modes
are rejected for projected references because the spectral diagonal varies by element.

SIMT FP64 uses four directional contractions. Grouped implementations process 16 distinct elements using the same shared axes: FP64 uses CUDA 8×8×4 WMMA APIs;
TF32 uses 16×16×8 WMMA with FP32 accumulation and rounded intermediate operands.
Both fuse gather, current FP64 mass factors and FP64 scatter. TF32 direct casts restrict range and can
perturb symmetry/SPD; FP64 admission does not certify its rounded arithmetic.
NCU job **3439260** independently confirms **1,573,344 DMMA instructions**,
zero HMMA, 12.11% tensor-pipe activity and 90.44% active-warps occupancy.
Its diagnostic duration is 216.992 microseconds with 333,429,248 DRAM bytes
(1.54 TB/s); benchmark comparisons use the separate unprofiled timings above.

All **11 native suites passed**, covering retained exact-mode guards/regressions,
projected Cartesian/rotated-orthogonal/sheared/curved Q2 cases, reordered elements,
GL7 mass versus assembled MFEM, shared classes with distinct element diagonals,
uniform/heterogeneous/returning density, symmetry/positive action, dense-mode
rejection, lifecycle and converged FP64 FGMRES integration. The independent oracle
assembles MFEM blocks, integrates full shape-squared mass and uses a separate cyclic
eigensolver and dense projected inverse. Maximum FP64 projected-action relative
L2 error across the small fixtures was **3.615e-14**; physical block assembly differed
by at most **2.918e-16**. These qualify the surrogate implementation, not its accuracy
as an inverse of every current block.

The large benchmark uses 512² final-resolution Q2 elements, p5, 9,437,184 scalar
DOFs, warp $F=(x+a\sin(2\pi x)\sin(2\pi y),y+a\sin(2\pi x)\sin(2\pi y))$,
$a=.035$, GL7 mass, GL6 diffusion/faces, alpha=6.357828776041666e-7 and penalty 144.
Signed residuals and positive heterogeneous density are synthetic, not evolved fields.
CUDA events time complete production `Mult`, including gather/scaling/scatter;
mass refresh and setup are separate. Each of three rotated/reversed rounds has
8 warmups and 64 timed calls; ranges below are ranges of round medians, not CIs.

| Execution | Mult median, ms | Round-median range, ms | Speedup | Correction relative L2 | Mass refresh median, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| SIMT FP64 | 0.311760 | 0.309968–0.313152 | 1.000× | baseline | 0.196272 |
| WMMA factor FP64 | 0.205824 | 0.205008–0.206224 | 1.515× | 0 observed | 0.196096 |
| WMMA factor TF32 | 0.195760 | 0.194896–0.195760 | 1.593× | 4.42225e-4 | 0.196016 |

Large-case errors compare with projected SIMT FP64; the small oracle is independent. TF32 relative max-norm difference was 7.47260e-4. Backend setup took 4.945–5.056 s,
including initial blocks/refresh and excluding fixture construction.
Nine shared bases occupy **5,184 bytes**; 36 FP64 reciprocals per element occupy
**75,497,472 bytes (72 MiB)**. Logical backend-owned storage is 292,558,224 bytes
(279.005 MiB) for SIMT and 293,672,676 bytes (280.068 MiB) for grouped modes.
Grouped execution has 16,389 groups and 80 padded slots. Dense EA is freed after setup;
no per-element dense inverse is retained. These are not process peak memory.

**SPD does not certify fixed-sweep stability.** The coarse 5×4 curved Q2 fixture
has initial local generalized eigenvalues 0.10631246–**3.02660929**; heterogeneous
density raises the maximum to 3.04518955. For a stationary update, an eigenmode
has factor $1-\omega\lambda$: at omega=.7 the initial maximum gives −1.11863,
outside the unit circle; its local threshold is omega<0.660805. These local-block
spectra are not measured full global spectra, but they rule out a general safety
claim for omega=.7. The large-case off-diagonal Frobenius fraction 0.0176645 also
does not bound the inverse error or establish contractivity of the coupled iteration.

The fixed mapped Sedov case uses 512² Q2 elements, ordinary DG p5, L2/BrokenL2,
SSPRK3, CFL .3, C_alpha=1/6 with max-h, scans/limiters/modal damping off and no restart.
The actual mesh resolves h_max=.001953125 and logged alpha=6.3578288e-7.
Both arms use the same FP64 WMMA projected inverse and original current FP64 A/RHS.
Control **3439162** runs FGMRES to RHS-relative 1e-6 with factor-one original-A
certification; candidate **3439241** uses 50 initial / 2 subsequent sweeps, omega=.7.
Both reached **t=.0025 in 162 steps**, with 487 sigma calls. All 487 control solves
were certified; the maximum logged residual was 9.9990288e-7. The candidate is
explicitly approximate, without a residual certificate.

| Observation | Converged control | Fixed-sweep candidate |
| --- | ---: | ---: |
| Application wall, including setup and output | 432.541 s | 278.415 s |
| Native reported loop, including final output | 315.024 s | 166.814 s |
| Iterations / sweeps | 32,665 | 1,022 |
| Current operator applications | 34,121 | 1,022 |
| Sampled total GPU memory | 35,151 MiB | 29,099 MiB |

The arms ran on different GPU allocations after an external audit-only stop,
so these are raw costs, **not a paired speedup measurement**. Both physical
applications finished successfully. Their generic packed FP64 Euler face path
was selected after the optional tensor-face representation rejected Q2 geometry.
Mass and volume also use general geometry paths. Host assembly/solve scopes
can absorb deferred GPU work; they are not an exclusive GPU cost breakdown.

Complete saved Q5 fields were compared with GL8 quadrature and actual fitted Q2
Jacobians; cell geometry and connectivity match exactly. The largest conserved
relative L2 difference is **2.8753e-8** (momentum), and sampled normalized Linf is
**6.7181e-8**. Density L2 is 6.9603e-10 and energy L2 is 2.3791e-9. Both arms have
positive sampled density, pressure and q_sigma at all 16,777,216 GL8 points.
These are engineering comparisons with the converged run, not discretization-error
estimates or positivity proofs between samples.

**Sigma differs by 23.0209% in L2** (31.3431% sampled normalized Linf). The reference
sigma amplitude is only 1.006e-6, while pressure is at least1.0000028. This early
agreement does not establish the two-sweep method's later accuracy or stability.
No full t=.1 comparison has run, and no TF32 physical comparison is claimed.

The prototype establishes that shared small transforms plus per-element diagonal
weights work on this mapped mesh, with actual FP64 tensor-core execution and no
retained dense inverse per element. It does not establish a universally stable
fixed-sweep solver. Further validation should use an efficient converged reference
with a coarse correction before paying for a late-time mapped comparison: the
local-only control averaged 67.1 FGMRES iterations. Existing generic element/Q1
two-level preconditioning is source-compatible but not yet measured on this mesh.

The original metadata writer emits bare `nan` for two unavailable fixed-sweep
residual norms. Raw output is preserved; a separate postprocessor treats only
those audited diagnostic entries as null. This is unrelated to the physical field
finiteness checks or the disabled runtime scans.

Frozen runtime arguments/environment and numeric observations are saved in
[case settings](benchmarks/dg_mapped_tensor_jacobi_case.json) and
[measurements](benchmarks/dg_mapped_tensor_jacobi_measurements.json).
Case environment keys are suffixes: the runner sets both IGRLAB_ and legacy
DG_IGR_ aliases, with PROFILE_NVTX handled separately. `method.euler_faces` in
that unchanged pre-run case records the intended optional representation; the
measurements record the actual generic packed path.

Native qualification uses the canonical `build.sh` targets and CTest suites listed
in the measurements. The fixture can generate the exact map with
`--write-mapped-mesh PATH --mesh-n 512`; benchmark with
`--benchmark --reference projected --mesh-n 512 --backend all --warmup 8 --repetitions 64 --rounds 3`.
Do not substitute a differently warped/refined mesh or a historical Cartesian VTK.
