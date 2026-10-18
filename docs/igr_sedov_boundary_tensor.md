# Sedov with a Cartesian physical-boundary tensor inverse

The stored `examples/sedov_taylor_blast/sedov_taylor.msh` is exactly Cartesian:
1,024 identical affine squares and 1,089 vertices on [0,1]^2, with binary-exact
spacing 1/32. It has 1,984 interior edges and 128 boundary edges. Tags 1–4 are
bottom, right, top and left; all four Euler boundaries are slip walls. The
scalar IGR closure uses natural homogeneous flux boundaries, with interior
SIPG terms and no physical-boundary diffusion penalty. The mesh is not periodic.

Four parallel uniform refinements and p5 give 512×512 elements, 9,437,184 scalar
IGR DOFs and 37,748,736 conserved-state DOFs. The initializer retains its
Gaussian deposition width of 0.1875 under refinement. This is the existing
smooth finite-width Sedov case, with gamma=1.4, background density/pressure 1,
deposited energy 1, and zero initial velocity.

## Implementation

Numerical source commit: `5f1f5e0`. The extension is restricted to spectral mode;
the existing periodic shared-patch and multigrid admissions remain intact.
The new path checks a connected, fully open Cartesian rectangle with uniform
positive axis-aligned affine geometry, p5 scalar L2 nodal maps and one CUDA rank.
GL and GLL nodal coordinates are supported by the modal transformation; this
application run uses ordinary DG with GL nodes.

GPU probes reduce into nine interior/edge/corner stencil classes, using storage
linear in element count instead of dense per-element matrices. The reference
includes both one-dimensional interior operators and four physical-boundary
diagonal corrections. All nine classes are checked against that separable
model, including corner additivity and the screened constant mode. Open-axis
matrices omit wrap couplings and include the actual boundary corrections;
simply deleting wrapped neighbors would leave incorrect diagonal blocks.

Class averaging is not an SPD proof. Symmetry, complete FP64 eigenpairs,
orthogonality and positive screening are checked before inversion. The four
large cuBLAS transforms retain explicit TF32 arithmetic. The evolving physical
operator and outer FGMRES residual remain FP64. The reference is an approximate
preconditioner once the density evolves, not an exact inverse of that equation.

Mixed-periodic rectangles, curved/nonuniform/skew geometry, arbitrary graph
connectivity, other orders and distributed transforms are not added. Unsupported
topology or a nonseparable reference fails admission. Rebuilding at a later
heterogeneous state can likewise fail the reference checks; general adaptive
reference reconstruction is not established by this run.

## Build and focused native check

DeltaAI job 3405250 used the canonical `build.sh --gpu --cuda-arch 90 --iterate`
route with targets `sedov_taylor_blast` and `dg_igr_tensor_multigrid_test`.
The focused `dg_igr_tensor_multigrid_cuda` check passed in 4.09 seconds. It retains
existing periodic coverage and adds one translated/reordered 5×7 natural
rectangle with unequal side lengths. An independently assembled FP64 SIPG+mass
oracle checks generic and corner loads, constants, and changed-current-A solves.
FP64 inverse residuals were at most 5.05e-13; the TF32-preconditioned changed-A
residuals were at most 1.89e-11. Independent source review found no blockers.

Tested binary SHA256:
`f1aa5f4958bb1bf68dbeaec33030d8ce9de02dbb66baf1308510b9f561162e6e`.
The s1 manifest binds 770 application/configuration/test files and the three
changed numerical/test paths. Subsequent header edits only clarify comments.

## Direct large run

Job 3405258 used one GH200, account `bdiy-dtai-gh`, partition `ghx4`, constraint
`projects`, eight CPUs and 96G requested host memory. It ran ordinary DG from
t=0 to t=0.1 with CFL=0.3, C_alpha=1 and max-length scaling: physical
alpha=3.814697265625e-6. Average/local/max policies coincide on this mesh.

The configuration uses L2/BrokenL2, every-stage IGR, exact SIPG faces,
FGMRES rtol=1e-6, atol=0, maxit=1000, and error-on-nonconvergence. Both positivity
limiting and modal damping are off. Optional admissibility and conservation
diagnostics are off; native solver and nonfinite checks remain. VTK output
cadence is 0.02, with pressure evaluated from the conservative field at output
points and no H1 diagnostics.

The preconditioner uses the new natural-boundary TF32 spectral reference.
Euler retains the previously measured FP32 shared tensor-line contractions;
state, nonlinear fluxes, faces and the current IGR equation remain FP64. No new
matched FP64 performance comparison or end-to-end speedup is claimed.

| Result | Measured value |
|---|---:|
| Final physical time | 0.1 |
| Accepted / rejected steps | 4,940 / 0 |
| Application wall time | 2,192.84 s (36m 33s) |
| Reported loop average before final output | 0.4211 s/step |
| Converged IGR solves | 14,821 / 14,821 |
| Mean / maximum Krylov iterations | 7.97 / 12 |
| Final current-A relative residual | 2.17e-10, against requested 1e-6 |
| Preconditioner applications / GEMMs | 118,127 / 472,508 |
| Sampled process GPU memory peak | 29,042 MiB (28.36 GiB) |
| GPU allocation time | 2,199 s (0.61083 GPU-hours) |

The setup logged `topology=natural-boundary-cartesian reference_classes=9`,
324 probes, 3072×3072 transform dimensions, and four GEMMs per application.
Its separable-reference defect was 2.14e-15, eigen residual 1.55e-15,
orthogonality error 5.39e-15, and no eigenvalues were clipped. Reference setup
cost 3.67 s. Native logs report no fallback. IGR linear solves consumed 673.78 s. The recorded assembly/form/setup/solve
subtotal was 785.10 s, about 37% of the reported evolution time; complete IGR
updates totaled 846.81 s.

This is a successful direct application run, not a continuum-error or mesh
convergence study. The linear residual certificate covers the IGR equation;
it does not measure the Euler FP32 discretization error. The six native VTK
frames are at t=0,0.02,...,0.1. The final preview evaluates the Q5 conserved
polynomials at all 512² element centers, then derives pressure and speed.
Sampled density ranges from 0.4494 to 1.2874 and pressure from 1.0003 to 1.9120.
These center samples do not establish positivity everywhere. The case's
printed `Solution error` measures change from its initial condition, not error
against an analytic Sedov similarity solution.

## Reproduction and artifacts

The full scripts, source manifests, build receipt and complete native VTK
series remain on DeltaAI:
`/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/sedov-boundary-20261018`.
The local evidence bundle is:
`/home/jarias9/.cache/igrlab-experiments/sedov-boundary-20261018`.

`build-s1.slurm` contains the canonical build and focused-check command.
`sedov-s1.slurm` launches the single direct application run.
`run-sedov-s1.py` records all arguments, environment settings, binary/mesh/driver
hashes and application PID. Memory samples are filtered to that PID.
`result-s1.json` contains the final native summaries. The local `final-vtk`
directory contains only the final native snapshot and a one-frame
`sedov-final.pvd`; the original six-frame index is retained separately.
`sedov-t0p1.png`, `.json` and `.npz` contain the preview and center samples.
Build/check allocation time was 72 s, bringing build plus run usage to
0.63083 GPU-hours. No further simulation campaign was run.
