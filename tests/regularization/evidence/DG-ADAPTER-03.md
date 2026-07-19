# DG-ADAPTER-03 evidence

## Scope and identity

- Task: corrected frozen-scalar IGR BrokenDG CPU adapter.
- Required base/HEAD/local `dev`/`origin/dev` before edits:
  `28cdaddde8d07dae404a0b3e69344b4e598bee55`.
- Required base tree:
  `833fa24b662fa4795fa61637a002b37ba658e074`.
- Worktree:
  `/home/jarias9/.codex/worktrees/1b64/dg_igr_miniapp`.
- Branch: `feature/dg-frozen-scalar-igr-adapter-corrected`.
- Worktree lock reason: `DG-ADAPTER-03 persistent task`.
- MFEM marker:
  `d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec`.
- Rejected historical candidate
  `7d6e5fed8ecad453dc73245abc5f80ea372091c8` was inspected only as
  read-only design evidence. It was not merged or cherry-picked.

The final commit and tree are reported outside this manifest to avoid
self-reference. No build output is stored in the repository.

## Corrected operation requirements

| Operation | Required evaluated fields |
| --- | --- |
| AuxiliaryCoefficient | state |
| AuxiliarySource | state, conservative-state gradient |
| VolumeCorrection | state, scalar sigma |
| NormalCorrection | state, scalar sigma, normal |
| CharacteristicBounds | state, scalar sigma, normal, per-point scalar base characteristic speed |
| WallLaw | disabled; hook absent |
| Diagnostics | disabled; hook absent |

Coefficient and source builders bind no current sigma storage, and their
validation contexts carry zero auxiliary producer authority. The adapter uses
a private launch copy solely to express the generic contract's scalar
coefficient/source output extent; it does not bind, read, or require sigma.
Characteristic bounds call `point.BaseCharacteristicSpeed()` and then
`frozen_scalar_igr_algebra::FluxCharacteristicSpeed`. The adapter parameter
object contains no base-speed constant. Default adapter selection is
`Disabled`.

## Changed-content SHA-256

These hashes cover every non-manifest changed path:

```text
82a6db69774f2f9c99f84b8e46c2c0b792117a8083aa82bc1de606c48860fa9d  include/physics/regularization/adapters/FROZEN_SCALAR_IGR_BROKEN_DG_CPU_INTEGRATION_REQUEST.md
785274168956cbe7949ce247feb14de8faed0891b8ca4e28902498db40a0a2f1  include/physics/regularization/adapters/frozen_scalar_igr_broken_dg_cpu.hpp
772ba6577239d4660e823e1842ecdd8bd4f112e7e0a65f8bee19e4755fd306fc  include/physics/regularization/models/frozen_scalar_igr_broken_dg.hpp
612d3bdbad755b97aaf42ca75c12f65e2b454202ed40fed59ee62a7b9866183d  src/physics/regularization/adapters/frozen_scalar_igr_broken_dg_cpu.cpp
ab9d31a5f31c29c05fb8bbb6bbcd84ae0f8f07e67618618bb7b5a52b64f8cb41  tests/regularization/CMakeLists.txt
79dd30d29934e27fbcf562aac64681d2e1c4507d75a26ea5a2237baa666c1179  tests/regularization/test_frozen_scalar_igr_broken_dg_cpu_adapter.cpp
```

## Toolchain and sanctioned dependency evidence

- CMake: 3.31.11.
- Compiler: GCC 13.3.0 through the existing Open MPI `mpicxx` wrapper.
- MFEM package:
  `/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/install-cpu/lib/cmake/mfem`.
- No MPI launch, CUDA/GPU path, profiler, long simulation, or
  `/tmp/dg_igr-sm120.lock` acquisition occurred.

Sanctioned read-only header hashes:

```text
1371afec748ece81106bc49c5bee841892640b6cb01a8815bfaf1d78f105e1c0  /home/jarias9/projects/dg_igr_miniapp/tpl/mfem/fem/picojson.h
1081222fade1e9dd2e900193f575ef109cdac764eeae8400888f0e3c02567627  /home/jarias9/projects/dg_igr_miniapp/tpl/mfem/general/tinyxml2.h
```

## Exact-final post-normalization binding

The six non-manifest hashes above are the final LF/EOF-normalized source
bytes. After normalization, each already-fresh accepted build tree was rebuilt
serially and verbosely. The build output showed recompilation of
`frozen_scalar_igr_broken_dg_cpu.cpp` and
`test_frozen_scalar_igr_broken_dg_cpu_adapter.cpp`; no acceptance result below
is inferred from the earlier pre-normalization binaries.

Exact-final commands and exits:

```text
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake --build /tmp/dg_igr-adapter03-release-019f7a43-final --parallel 1 --verbose
rc=0
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/ctest --test-dir /tmp/dg_igr-adapter03-release-019f7a43-final --output-on-failure
rc=0, 2/2 passed
/tmp/dg_igr-adapter03-release-019f7a43-final/dg_igr_frozen_scalar_igr_broken_dg_cpu_adapter_test
rc=0, all reported parity errors zero

/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake --build /tmp/dg_igr-adapter03-asan-019f7a43-final --parallel 1 --verbose
rc=0
env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 /home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/ctest --test-dir /tmp/dg_igr-adapter03-asan-019f7a43-final --output-on-failure
rc=0, 2/2 passed

/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake --build /tmp/dg_igr-adapter03-project-019f7a43-final --parallel 1 --verbose --target dg_igr_frozen_scalar_igr_broken_dg_cpu_adapter_test dg_igr_regularization_contract_test dg_igr_frozen_scalar_igr_algebra_test dg_igr_unit_tests
rc=0
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/ctest --test-dir /tmp/dg_igr-adapter03-project-019f7a43-final --output-on-failure -R '^(dg_igr_frozen_scalar_igr_broken_dg_cpu_adapter|dg_igr_regularization_contract_cpu|dg_igr_frozen_scalar_igr_algebra_tests|dg_igr_igr_tests|dg_igr_dgsem_igr_tests|dg_igr_time_defaults_tests)$'
rc=0, 6/6 passed
```

## Fresh strict C++17 Release

Accepted directory:
`/tmp/dg_igr-adapter03-release-019f7a43-final`.

Configure, build, and test commands:

```sh
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake \
  -S tests/regularization \
  -B /tmp/dg_igr-adapter03-release-019f7a43-final \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/home/jarias9/spack/opt/spack/linux-x86_64_v4/openmpi-5.0.10-belap7umgzjinzggojo3kmnw5le2fmez/bin/mpicxx \
  -DMFEM_DIR=/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/install-cpu/lib/cmake/mfem \
  -DCMAKE_CXX_FLAGS='-Wall -Wextra -Wpedantic -Werror'
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake \
  --build /tmp/dg_igr-adapter03-release-019f7a43-final --parallel 2
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/ctest \
  --test-dir /tmp/dg_igr-adapter03-release-019f7a43-final --output-on-failure
```

All exits were 0. The generated adapter target flags were exactly
`-Wall -Wextra -Wpedantic -Werror -O3 -DNDEBUG -std=c++17`.
CTest passed 2/2 (accepted regularization contract and adapter).
CMake cache SHA-256:
`09390803a30946bd43b1e60fac685423f1b27bd9fce761e6982f336d735ad4d4`.

## Fresh strict ASAN/UBSAN

Accepted directory:
`/tmp/dg_igr-adapter03-asan-019f7a43-final`.

Configure flags:

```text
-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -Wall -Wextra -Wpedantic -Werror
```

The configure command otherwise used the same compiler and MFEM package as the
Release gate. The build was confirmed serially with:

```sh
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake \
  --build /tmp/dg_igr-adapter03-asan-019f7a43-final --parallel 1
env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  /home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/ctest \
  --test-dir /tmp/dg_igr-adapter03-asan-019f7a43-final --output-on-failure
```

All confirmed exits were 0; CTest passed 2/2 with leak detection and
halt-on-error enabled. CMake cache SHA-256:
`e049cbf0eb9681aa6257ddda8a4383e6bbb53883f6a70d3975da6350e3ee1b02`.

## Fresh bounded project CPU gate

Accepted directory:
`/tmp/dg_igr-adapter03-project-019f7a43-final`.

The first configure invocation used Release, the compiler/MFEM package above,
the exact flags
`-I/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/fem -I/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/general`,
and:

```text
-DZLIB_INCLUDE_DIR=/home/jarias9/spack/opt/spack/linux-x86_64_v4/zlib-ng-2.3.3-i64qsvxt7m2fzhmzcov3g7ser3h2z2gx/include
-DZLIB_LIBRARY=/home/jarias9/spack/opt/spack/linux-x86_64_v4/zlib-ng-2.3.3-i64qsvxt7m2fzhmzcov3g7ser3h2z2gx/lib/libz.so
```

Bounded targets:

```text
dg_igr_frozen_scalar_igr_broken_dg_cpu_adapter_test
dg_igr_regularization_contract_test
dg_igr_frozen_scalar_igr_algebra_test
dg_igr_unit_tests
```

The build exit was 0. Focused CTest regex:

```text
^(dg_igr_frozen_scalar_igr_broken_dg_cpu_adapter|dg_igr_regularization_contract_cpu|dg_igr_frozen_scalar_igr_algebra_tests|dg_igr_igr_tests|dg_igr_dgsem_igr_tests|dg_igr_time_defaults_tests)$
```

CTest passed 6/6. CMake cache SHA-256:
`bb16f2c5c16bea4ac95d872dadc6cddfc7b65b35d483a6066f06d602ea9664f6`.

## Numerical and negative results

Direct adapter harness output:

```text
PARITY dimensions=1,2,3 max_coefficient_error=0 max_source_error=0 max_tensor_error=0 max_normal_error=0 max_characteristic_error=0 max_two_state_error=0
STATIC_BINDINGS operations=5 host_selections=1 successful_launches=18
NEGATIVE_GATES selection=14 capabilities=2 parameters=5 required_fields=15 identity_provenance=16 producer_authorities=15 outputs=6 aliases=2 overflow=1
```

The bounds cases use at least three distinct per-point base speeds and compare
each result exactly to the authoritative algebra. Operation point counts and
point-set identities differ. Required producer generations are nonzero,
independent, and unequal; stage 0 launches. The scalar sigma descriptor is
`ColdRecomputed` with zero restart and output identities.

Every negative checks the intended `ValidationCode`, unchanged launch count,
and unchanged writable sentinel. The matrix covers default/explicit disabled,
unsupported axes/backend, invalid alpha/floors, missing/shape/size/stale
state/gradient/sigma/normal/base speed, scalar extents, site/count/dimension/
point-set/stage/state generations, producer authority, missing/wrong outputs,
two exact aliases, overflow, wall law, and diagnostics.

## Preserved failed/diagnostic attempts

- The built-in patch helper could add new files but crashed on updates because
  its filesystem guard could not find `bubblewrap`. Authorized updates were
  applied with auditable unified patches instead.
- `/tmp/dg_igr-adapter03-release-019f7a43-a`: first configure failed because
  the initial owned CMake registration escaped three CMake variables
  literally. The directory is preserved and excluded.
- `/tmp/dg_igr-adapter03-release-019f7a43-b`: diagnostic build first failed
  on an unused test helper, then on one stale builder call. After correction,
  the same diagnostic directory built and passed 2/2; it is excluded from the
  fresh acceptance claim.
- Two sanitizer build polling calls returned before a process exit while the
  compiler remained live. The process was allowed to finish; the serial build
  confirmation and sanitizer CTest exits above are the accepted evidence.
- Fallback patch hunks created `.orig` backups. Each exact generated backup
  was removed before the final commit audit; no deliverable/evidence content
  was removed.
- Two system-patch attempts to add exact-final provenance had malformed hunk
  counts and exited 2 without changing the manifest. The corrected unified
  diff was applied next.
- The first exact-final post-normalization rebuild invocation used bare
  `cmake`; that escalated login shell did not expose CMake on `PATH`, so it
  exited 127 before checking or changing a build. The cached absolute CMake
  executable was then used for every accepted exact-final command above.

## Non-goals

No shared configuration, solver/time-loop registration, defaults, boundary
science, output/restart wiring, generic contract edit, frozen-scalar algebra
edit, DG/DGSEM/HDG operator edit, production activation, merge, push, rebase,
or PR is included.
