# IGRlab

IGRlab develops Euler and information-geometric regularization methods with
MFEM, retaining DG, DGSEM, and HDG implementations on CPU and GPU.

| Backend | CPU / ordinary GPU | Specialized tensor implementation |
|---|---|---|
| DG | Shared application and MFEM device kernels | TF32 WMMA in the shared IGR screened-patch solve |
| DGSEM | Reference and packed CPU/CUDA operators | Explicit FP64 WMMA fused CUDA residual |
| HDG | Generic CPU and ordinary CUDA engines | FP64 WMMA volume/condensation, with optional arithmetic variants |

Support depends on dimension, element geometry, polynomial order, numerical
policy, and device architecture. Existing admission checks enforce these
limits. The HDG generic application currently admits conforming two-dimensional
quadrilateral meshes with orders p1-p3. See [A100 support](docs/a100.md) for the
SM80 build requirements and the scope of recorded hardware validation.

The C++ application namespace is `igrlab`; `dg_igr` remains a source alias.
`IGRlab::core` names the existing library target. Canonical configuration and
runtime environment names use `IGRLAB_*`; legacy `DG_IGR_*` names, existing
executable/CTest names, and checkpoint formats remain supported during this
migration. Canonical runtime environment values take precedence, including an explicitly
empty value. [Consolidation records](docs/consolidation/README.md) describe the
recovery archive, retained features, validation, and proposed final cleanup.

## Building

MFEM is pinned as the `tpl/mfem` submodule and built locally so this project can
consume its installed CMake target. Spack supplies MFEM's dependencies.

Initialize the submodule and load Spack's shell support once:

```bash
git submodule update --init
source /path/to/spack/share/spack/setup-env.sh
```

Then use the unified build driver:

```bash
./build.sh --cpu-only -j 16
./build.sh --gpu --cuda-arch 120 -j 16
./build.sh --a100 --hdg-runtime -j 16
```

Application and locally built MFEM outputs live outside the checkout. By default they are scoped under
`${XDG_CACHE_HOME:-$HOME/.cache}/igrlab-build` by canonical worktree and
profile; choose another external location with `--output-root DIR` or
`IGRLAB_OUTPUT_ROOT`. A selected output root must be outside every detected Git
checkout.

```bash
./build.sh --cpu-only --output-root /scratch/$USER/dg-igr -j 16
./build.sh --gpu --cuda-arch 120 --print-plan
```

`--print-plan` reports the output root, deterministic worktree/profile scope,
and resolved app/MFEM paths without creating output-root content or acquiring
the output lock. `build_mfem_tpl.sh` is an internal helper; invoke `build.sh`
for all builds.

Spack environments live under `spack-envs/`. The CUDA profiles are
`cuda-sm120` (Blackwell, CUDA 13 and HYPRE `develop`) and `cuda-sm80`
(A100, CUDA 12.9.1, HYPRE 2.32.0 and Umpire 2024.07.0). The A100 manifest
uses static HYPRE with CUDA/Umpire, as supported by older site Spack recipes. It is a portable template: configure site compilers and
CUDA/MPI externals in a site environment selected with `IGRLAB_SPACK_ENV`,
then concretize/install through the driver. For a shared x86 CPU build and
A100 GPU allocation, constrain the site environment to a CPU target supported
by both nodes (for example `target=x86_64_v3`), rather than the build node
architecture.

GPU architecture selection has three deliberately separate roles. For the
checked-in Blackwell profile, `--cuda-arch 120` selects Spack dependencies and
isolated build directories with `cuda_arch=120`; MFEM is compiled with
`MFEM_CUDA_ARCHS=80-virtual` PTX to avoid CUDA 13's expensive native `sm_120`
compilation; and this project's CUDA sources are compiled natively with
`APP_CUDA_ARCHS=120`. The driver enforces that split before dependency or cache
actions and rejects the ambiguous legacy `CUDA_ARCHS` variable. These values
normally do not need to be set explicitly.

`--a100` is equivalent to `--gpu --cuda-arch 80`: dependencies, MFEM, and
application CUDA sources all use native architecture 80, with separate
`cuda-sm80` output directories. It rejects an explicit CPU mode or any
conflicting `--cuda-arch` option. `--hdg-runtime` supports native 80 and 120,
and tensor/native runtime paths require a device matching the configured
architecture. The SM80 route still requires compilation and device validation on the target
system; a successful `--print-plan` only validates the build policy. Existing
SM120 disassembly qualification checks remain specific to that profile.

The application and MFEM builds both use `-j 16` in the example above. Use
`--mfem-jobs N` only when MFEM needs different concurrency, such as:

```bash
./build.sh --gpu --cuda-arch 120 -j 16 --mfem-jobs 8
```

`./build.sh --gpu --cuda-arch 120 --print-plan` validates and prints the
resolved architecture and job policy without invoking Spack, CMake, cache
cleanup, or a build.

Both commands install dependencies, rebuild MFEM, build and install the app,
and run CTest. Useful controls include `--no-deps`, `--no-clean`, `--no-tests`,
and `--smoke`. Run `./build.sh --help` for the full interface.

Standalone numerical campaign probes are excluded from ordinary builds.
Enable them explicitly with `IGRLAB_BUILD_RESEARCH_TESTS=ON ./build.sh ...`.
Direct operator, conservation, damping, source-oracle, and tensor-parity checks
remain available in ordinary correctness-test builds. Existing caches can be
selected with `--output-root`; the default cache root is now `igrlab-build`.

## Modal damping (M16)

Add these options to a DG Euler/IGR example to enable the original M16
conservative modal damping:

```bash
--modal-damping-rate 80000 --igr-sigma-cadence every-stage
```

The default rate is zero (disabled). The rate is in inverse model time. At
each Runge–Kutta stage the solver transforms each conservative component to
Legendre coefficients and adds the source
`dU_k/dt = -rate * max((39*k - 29*p)/(10*p), 0)^4 * U_k`.
For degree `p=39`, modes 0–29 are untouched, mode 30 decays at rate 8,
and mode 39 at rate 80000. Other degrees use the same fractional cutoff.
When IGR is enabled, its stress must be recomputed every stage.

This implementation supports affine, nodal L2 DG segments, quadrilaterals,
and hexahedra with uniform polynomial order. In two or three dimensions,
the weight uses the largest modal index across the reference directions;
this is a tensor extension of the original one-dimensional treatment.
Physical cell averages are preserved. Curved and non-affine elements and
HDG are rejected when damping is enabled.

The timestep is capped at `1/rate` (0.0000125 for M16). This guards the
isolated damping term; the coupled Euler/IGR system can require a smaller
step. The handoff's original fine-step M16 comparison used
`dt = 1.953125e-7`. Damping is a mitigation whose effect on gradients and
pressure should be measured in each case; it does not guarantee positivity
or removal of ringing. Run metadata records the selected rate and profile.
