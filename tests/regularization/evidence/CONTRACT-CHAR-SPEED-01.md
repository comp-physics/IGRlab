# CONTRACT-CHAR-SPEED-01 evidence

## Scope and repository identity

- Task: bounded evaluated per-point base-characteristic-speed contract lane.
- Frozen base: `99290bf3bdc7a959ffeb43128623540cac4f6324`.
- `origin/dev` at base verification:
  `99290bf3bdc7a959ffeb43128623540cac4f6324`.
- `HEAD` during configure/build/test evidence:
  `99290bf3bdc7a959ffeb43128623540cac4f6324` (pre-commit).
- Branch: `contract/evaluated-base-characteristic-speed`.
- Worktree:
  `/home/jarias9/.codex/worktrees/30b8/dg_igr_miniapp`.
- Worktree lock reason: `CONTRACT-CHAR-SPEED-01_persistent_bounded_task`.
- Pre-manifest owned-content diff SHA-256:
  `a66191aa0b1d0ea0eee8e552202d2334f24be16bd1a6c8ac8b955ed1d3292e3b`.
- Final commit and Git tree IDs are intentionally reported after the one allowed
  commit; embedding either in this committed file would be self-referential.

No generated build output is stored in the repository. All build trees below
are task-local paths under `/tmp`.

## Changed content and SHA-256

The hashes below cover every changed contract/test/document path before this
self-referential manifest was added.

```text
9805f65e4edd6753a8739b2a06df168c5177608b773433a53836556986c01104  include/physics/regularization/FROZEN_SCALAR_IGR_ADAPTER.md
10170fed9297097a8d067be359f8a4f3db127329ce4ab1891f483fe44c5e5b0c  include/physics/regularization/evaluated_field.hpp
94f640622fe48e17fa0454407c8c2b885e45e963a16c141a348b0376006d0c34  include/physics/regularization/validation.hpp
8a763f1f8ddff6cde9169fb78fe136399506b06387b107785cb3086b6a231634  include/physics/regularization/traits.hpp
dc3472a72f9c40f4f1c10e859c4be1aa94f011751a3713e6a1291b894a3203e3  include/physics/regularization/execution.hpp
1ad5ae1493769c393930e59486bff076c1f3203621264ee8aa0f5621ea9955c5  src/physics/regularization/validation.cpp
5ad52890f7e1c14104d6bfeaff916829875b585d69d74eb8ead7a567a22f88b4  tests/regularization/test_regularization_contract.cpp
```

## Toolchain and backend

- CMake: 3.31.11.
- Compiler wrapper:
  `/home/jarias9/spack/opt/spack/linux-x86_64_v4/openmpi-5.0.10-belap7umgzjinzggojo3kmnw5le2fmez/bin/mpicxx`.
- Compiler: GCC 13.3.0.
- MFEM: 4.9, Git
  `tags/v4.9-0-gd9d6526cc1749980a2ba1da16e2c1ca1e07d82ec`.
- MFEM backend: MPI enabled; CUDA and HIP disabled.
- Runtime device in the contract harness: `mfem::Device("cpu")`.
- The accepted BrokenDG probe combination is statically CPU-only
  (`ProbeDevice::value == false`), so the conditional MFEM-device-mode
  exercise is not supported by this frozen harness and was not claimed.

## Fresh strict C++17 Release gate

Fresh directory:
`/tmp/dg_igr-contract-char-speed-release-019f7a01`.

Configure:

```sh
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake \
  -S tests/regularization \
  -B /tmp/dg_igr-contract-char-speed-release-019f7a01 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/home/jarias9/spack/opt/spack/linux-x86_64_v4/openmpi-5.0.10-belap7umgzjinzggojo3kmnw5le2fmez/bin/mpicxx \
  -DMFEM_DIR=/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/install-cpu/lib/cmake/mfem \
  -DCMAKE_CXX_FLAGS='-Wall -Wextra -Wpedantic -Werror'
```

Result: PASS, first configure invocation. Generated target flags were
`-Wall -Wextra -Wpedantic -Werror -O3 -DNDEBUG -std=c++17`.

Build and test:

```sh
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake \
  --build /tmp/dg_igr-contract-char-speed-release-019f7a01 --parallel 2
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/ctest \
  --test-dir /tmp/dg_igr-contract-char-speed-release-019f7a01 \
  --output-on-failure
```

Result: PASS, warning-clean build; CTest 1/1 passed.

## Fresh strict C++17 ASAN/UBSAN gate

Fresh directory:
`/tmp/dg_igr-contract-char-speed-asan-019f7a01`.

Configure:

```sh
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake \
  -S tests/regularization \
  -B /tmp/dg_igr-contract-char-speed-asan-019f7a01 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=/home/jarias9/spack/opt/spack/linux-x86_64_v4/openmpi-5.0.10-belap7umgzjinzggojo3kmnw5le2fmez/bin/mpicxx \
  -DMFEM_DIR=/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/install-cpu/lib/cmake/mfem \
  -DCMAKE_CXX_FLAGS='-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -Wall -Wextra -Wpedantic -Werror'
```

Result: PASS, first configure invocation. Generated target flags include
`-fsanitize=address,undefined -std=c++17`.

Build and test:

```sh
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake \
  --build /tmp/dg_igr-contract-char-speed-asan-019f7a01 --parallel 2
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake \
  --build /tmp/dg_igr-contract-char-speed-asan-019f7a01 --parallel 1 --verbose
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/ctest \
  --test-dir /tmp/dg_igr-contract-char-speed-asan-019f7a01 \
  --output-on-failure
```

Result: PASS. The serial verbose confirmation reported the target 100% built;
CTest 1/1 passed with sanitizer halt-on-error and leak detection enabled.

## Project configure hygiene

Excluded attempt:
`/tmp/dg_igr-contract-char-speed-project-release-019f7a01`.

- Its initial fresh configure failed because ZLIB was not auto-discovered.
- The same directory was then reconfigured successfully with explicit local
  ZLIB paths.
- Its bounded build was interrupted after the coordinator clarified that a
  reconfigured CMake tree must not count as a fresh acceptance gate.
- The directory was preserved and is excluded from accepted evidence.

Accepted fresh directory:
`/tmp/dg_igr-contract-char-speed-project-fresh-019f7a01-a`.

The first configure invocation included all read-only dependency paths:

```sh
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake \
  -S . \
  -B /tmp/dg_igr-contract-char-speed-project-fresh-019f7a01-a \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/home/jarias9/spack/opt/spack/linux-x86_64_v4/openmpi-5.0.10-belap7umgzjinzggojo3kmnw5le2fmez/bin/mpicxx \
  -DMFEM_DIR=/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/install-cpu/lib/cmake/mfem \
  -DZLIB_INCLUDE_DIR=/home/jarias9/spack/opt/spack/linux-x86_64_v4/zlib-ng-2.3.3-i64qsvxt7m2fzhmzcov3g7ser3h2z2gx/include \
  -DZLIB_LIBRARY=/home/jarias9/spack/opt/spack/linux-x86_64_v4/zlib-ng-2.3.3-i64qsvxt7m2fzhmzcov3g7ser3h2z2gx/lib/libz.so \
  -DCMAKE_CXX_FLAGS='-I/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/fem -I/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/general'
```

Result: PASS, first configure invocation.

Bounded build and tests:

```sh
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake \
  --build /tmp/dg_igr-contract-char-speed-project-fresh-019f7a01-a \
  --target dg_igr_regularization_contract_test \
           dg_igr_frozen_scalar_igr_algebra_test dg_igr_unit_tests \
  --parallel 2
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/ctest \
  --test-dir /tmp/dg_igr-contract-char-speed-project-fresh-019f7a01-a \
  --output-on-failure \
  -R '^(dg_igr_regularization_contract_cpu|dg_igr_frozen_scalar_igr_algebra_tests|dg_igr_igr_tests|dg_igr_dgsem_igr_tests|dg_igr_time_defaults_tests)$'
```

Result: PASS. The three bounded targets built successfully; CTest 5/5 passed:

- `dg_igr_regularization_contract_cpu`;
- `dg_igr_frozen_scalar_igr_algebra_tests`;
- `dg_igr_igr_tests`;
- `dg_igr_dgsem_igr_tests`; and
- `dg_igr_time_defaults_tests`.

## Audit result

Before adding this manifest:

- `git diff --check`: PASS;
- hard-coded `index < 5` bounds in the owned validation implementation: none;
- source/test/document changes: exactly the seven hashed authorized paths above;
- generated repository build output: none.

> Remediation status: all evidence above is the preserved pre-remediation
> packet for rejected commit a4abd809c8b9c31a721792e6b60d6e826c7cc614.
> It is retained for audit history and is not the acceptance packet. The
> post-remediation evidence below supersedes it.

## Post-remediation evidence (superseding acceptance packet)

### Rejected packet identity and remediation scope

- Rejected commit: a4abd809c8b9c31a721792e6b60d6e826c7cc614.
- Rejected commit parent: 99290bf3bdc7a959ffeb43128623540cac4f6324.
- Rejected commit tree: 6cc49e8f19ce2efcf744cd9dbd32039a0487714b.
- Rejected manifest SHA-256:
  f70415a67b3f82d19fa6b9869b0c8850737ba07862a725e6cbbb71e8b4ca1801.
- The original Release, sanitizer, failed/reconfigured project, and accepted
  fresh project directories recorded above are preserved as pre-remediation
  evidence and were not reused.
- Remediation test-integrity change is confined to
  tests/regularization/test_regularization_contract.cpp: both BrokenDG
  normal-authority negatives now start from
  CharacteristicBoundsValidationContext(), the zero-authority case changes
  only expected normal authority, and the stale case changes only the bound
  normal producer generation. A valid-authorities control proves that normal
  generation 31 and base-speed generation 37 launch and write output before
  the two reason-coded negatives are exercised.
- Post-remediation test-integrity delta from the rejected commit, excluding
  this self-referential manifest, as a binary Git diff SHA-256:
  f067bd38d172c47ff87f2217764ed26311f23da543c1f9fc1da9dd71e6bd5514.
- Post-remediation owned-content diff from the frozen parent, excluding this
  self-referential manifest, SHA-256:
  4426c166096b70f91139cab2021bf828cacd632740bf19858ffab0ebd730420e.

### Post-remediation changed-content SHA-256

These hashes cover all seven non-manifest paths in the amended commit:

    9805f65e4edd6753a8739b2a06df168c5177608b773433a53836556986c01104  include/physics/regularization/FROZEN_SCALAR_IGR_ADAPTER.md
    10170fed9297097a8d067be359f8a4f3db127329ce4ab1891f483fe44c5e5b0c  include/physics/regularization/evaluated_field.hpp
    dc3472a72f9c40f4f1c10e859c4be1aa94f011751a3713e6a1291b894a3203e3  include/physics/regularization/execution.hpp
    8a763f1f8ddff6cde9169fb78fe136399506b06387b107785cb3086b6a231634  include/physics/regularization/traits.hpp
    94f640622fe48e17fa0454407c8c2b885e45e963a16c141a348b0376006d0c34  include/physics/regularization/validation.hpp
    1ad5ae1493769c393930e59486bff076c1f3203621264ee8aa0f5621ea9955c5  src/physics/regularization/validation.cpp
    39827158ba97a34b8676f1c4fcd667cd0e549f46e51fe3f3667f1ec61b94de0b  tests/regularization/test_regularization_contract.cpp

The amended commit, tree, and final manifest SHA-256 remain intentionally
reported outside this file to avoid self-reference.

### Fresh post-remediation strict C++17 Release gate

Fresh directory:
/tmp/dg_igr-contract-char-speed-remediation-release-019f7a01.

    /home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake \
      -S tests/regularization \
      -B /tmp/dg_igr-contract-char-speed-remediation-release-019f7a01 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=/home/jarias9/spack/opt/spack/linux-x86_64_v4/openmpi-5.0.10-belap7umgzjinzggojo3kmnw5le2fmez/bin/mpicxx \
      -DMFEM_DIR=/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/install-cpu/lib/cmake/mfem \
      -DCMAKE_CXX_FLAGS='-Wall -Wextra -Wpedantic -Werror'
    /home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake \
      --build /tmp/dg_igr-contract-char-speed-remediation-release-019f7a01 \
      --parallel 2
    /home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/ctest \
      --test-dir /tmp/dg_igr-contract-char-speed-remediation-release-019f7a01 \
      --output-on-failure

Result: PASS on the directory's first configure invocation; warning-clean
strict C++17 build; CTest 1/1 passed.

### Fresh post-remediation ASAN/UBSAN gate

Fresh directory:
/tmp/dg_igr-contract-char-speed-remediation-asan-019f7a01.

    /home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake \
      -S tests/regularization \
      -B /tmp/dg_igr-contract-char-speed-remediation-asan-019f7a01 \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_COMPILER=/home/jarias9/spack/opt/spack/linux-x86_64_v4/openmpi-5.0.10-belap7umgzjinzggojo3kmnw5le2fmez/bin/mpicxx \
      -DMFEM_DIR=/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/install-cpu/lib/cmake/mfem \
      -DCMAKE_CXX_FLAGS='-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -Wall -Wextra -Wpedantic -Werror'
    /home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake \
      --build /tmp/dg_igr-contract-char-speed-remediation-asan-019f7a01 \
      --parallel 2
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    /home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/ctest \
      --test-dir /tmp/dg_igr-contract-char-speed-remediation-asan-019f7a01 \
      --output-on-failure

Result: PASS on the directory's first configure invocation; CTest 1/1 passed
with leak detection and ASAN/UBSAN halt-on-error enabled.

### Fresh post-remediation bounded project gate

Fresh directory:
/tmp/dg_igr-contract-char-speed-remediation-project-fresh-019f7a01-a.
The first and only configure invocation supplied the sanctioned read-only MFEM
and explicit ZLIB paths from the outset:

    /home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake \
      -S . \
      -B /tmp/dg_igr-contract-char-speed-remediation-project-fresh-019f7a01-a \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=/home/jarias9/spack/opt/spack/linux-x86_64_v4/openmpi-5.0.10-belap7umgzjinzggojo3kmnw5le2fmez/bin/mpicxx \
      -DMFEM_DIR=/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/install-cpu/lib/cmake/mfem \
      -DZLIB_INCLUDE_DIR=/home/jarias9/spack/opt/spack/linux-x86_64_v4/zlib-ng-2.3.3-i64qsvxt7m2fzhmzcov3g7ser3h2z2gx/include \
      -DZLIB_LIBRARY=/home/jarias9/spack/opt/spack/linux-x86_64_v4/zlib-ng-2.3.3-i64qsvxt7m2fzhmzcov3g7ser3h2z2gx/lib/libz.so \
      -DCMAKE_CXX_FLAGS='-I/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/fem -I/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/general'
    /home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake \
      --build /tmp/dg_igr-contract-char-speed-remediation-project-fresh-019f7a01-a \
      --target dg_igr_regularization_contract_test \
               dg_igr_frozen_scalar_igr_algebra_test dg_igr_unit_tests \
      --parallel 2
    /home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/ctest \
      --test-dir /tmp/dg_igr-contract-char-speed-remediation-project-fresh-019f7a01-a \
      --output-on-failure \
      -R '^(dg_igr_regularization_contract_cpu|dg_igr_frozen_scalar_igr_algebra_tests|dg_igr_igr_tests|dg_igr_dgsem_igr_tests|dg_igr_time_defaults_tests)$'

Result: PASS on the directory's first configure invocation; the three bounded
targets built; CTest 5/5 passed. The excluded ZLIB-discovery failure directory
recorded above remains preserved and excluded.

### Post-remediation audit

- CharacteristicBounds negatives were audited: the sixth-field matrix and
  every BrokenDG CharacteristicBounds reason-coded case now carry nonzero
  authority for all other required fields.
- The two normal-producer cases each have exactly their intended invalid
  condition, and their launcher-count/output-sentinel assertions remain in the
  shared failure helper.
- git diff --check: PASS before amendment.
- Changed paths remain within the eight authorized contract/test/document
  paths, including this evidence manifest; no generated build output is in the
  repository.
