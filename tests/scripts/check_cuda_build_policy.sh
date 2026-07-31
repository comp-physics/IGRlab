#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [repository-root]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
BUILD_DRIVER="${REPO_ROOT}/build.sh"
MFEM_DRIVER="${REPO_ROOT}/build_mfem_tpl.sh"

fail() {
  echo "cuda-build-policy: FAIL: $*" >&2
  exit 1
}

assert_contains() {
  local text="$1"
  local expected="$2"
  [[ "${text}" == *"${expected}"* ]] || {
    fail "expected output to contain '${expected}', got: ${text}"
  }
}

assert_not_exists() {
  local path="$1"
  [[ ! -e "${path}" ]] || fail "unexpected action marker exists: ${path}"
}

assert_adjacent_args() {
  local path="$1"
  local first="$2"
  local second="$3"
  local previous=""
  local current
  while IFS= read -r current; do
    if [[ "${previous}" == "${first}" && "${current}" == "${second}" ]]; then
      return 0
    fi
    previous="${current}"
  done <"${path}"
  fail "helper arguments do not contain adjacent '${first} ${second}'"
}

assert_reuse_transaction_order() {
  local path="$1"
  local install_line revalidate_line view_line
  install_line="$(grep -nFx '  spack -e "${SPACK_ENV}" install' \
    "${path}" | cut -d: -f1 || true)"
  revalidate_line="$(grep -nFx \
    '    revalidate_reused_mfem_after_dependencies' \
    "${path}" | cut -d: -f1 || true)"
  view_line="$(grep -nFx 'SPACK_VIEW="${SPACK_ENV}/.spack-env/view"' \
    "${path}" | cut -d: -f1 || true)"
  [[ "${install_line}" =~ ^[0-9]+$ &&
     "${revalidate_line}" =~ ^[0-9]+$ &&
     "${view_line}" =~ ^[0-9]+$ &&
     ${install_line} -lt ${revalidate_line} &&
     ${revalidate_line} -lt ${view_line} ]] || {
    fail "reuse revalidation must occur once after dependency install and before SPACK_VIEW"
  }
}

[[ -f "${BUILD_DRIVER}" ]] || fail "missing build driver: ${BUILD_DRIVER}"
[[ -f "${MFEM_DRIVER}" ]] || fail "missing MFEM driver: ${MFEM_DRIVER}"

if [[ -n "${DG_IGR_TEST_REUSE_MFEM_ROOT:-}" ]]; then
  mfem_source="${DG_IGR_TEST_REUSE_MFEM_ROOT}"
else
  mfem_source="${REPO_ROOT}/tpl/mfem"
fi
[[ -e "${mfem_source}/.git" ]] || {
  fail "MFEM source is not initialized: ${mfem_source}"
}
mfem_source="$(realpath -e -- "${mfem_source}")"
mfem_source_top="$(git -C "${mfem_source}" rev-parse --show-toplevel)"
mfem_source_top="$(realpath -e -- "${mfem_source_top}")"
[[ "${mfem_source_top}" == "${mfem_source}" ]] || {
  fail "MFEM source is not the checkout top level: ${mfem_source}"
}
real_mfem_gitlink="$(git -C "${REPO_ROOT}" rev-parse HEAD:tpl/mfem)"
real_mfem_head="$(git -C "${mfem_source}" rev-parse HEAD)"
[[ "${real_mfem_head}" == "${real_mfem_gitlink}" ]] || {
  fail "MFEM source/gitlink mismatch: source=${real_mfem_head} gitlink=${real_mfem_gitlink}"
}
real_mfem_status="$(GIT_OPTIONAL_LOCKS=0 git -C "${mfem_source}" \
  status --porcelain=v1 --untracked-files=no)"
[[ -z "${real_mfem_status}" ]] || {
  fail "MFEM source has tracked changes: ${real_mfem_status}"
}

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/dg_igr_cuda_policy.XXXXXX")"
cleanup() {
  rm -rf -- "${tmp_root}"
}
trap cleanup EXIT

fixture_root="${tmp_root}/fixture"
fixture_mfem="${fixture_root}/tpl/mfem"
fake_bin="${tmp_root}/fake-bin"
mkdir -p "${fixture_root}/spack-envs/cuda-sm120/.spack-env/view" \
         "${fixture_root}/tpl" "${fake_bin}"
git -c advice.detachedHead=false clone --quiet --shared --no-checkout \
  "${mfem_source}" "${fixture_mfem}"
git -c advice.detachedHead=false -C "${fixture_mfem}" checkout --quiet \
  --detach "${real_mfem_gitlink}"
git -C "${fixture_root}" init --quiet
git -C "${fixture_root}" -c advice.addEmbeddedRepo=false add -- tpl/mfem
git -C "${fixture_root}" \
  -c user.name=dg-igr-fixture \
  -c user.email=dg-igr-fixture@example.invalid \
  -c commit.gpgSign=false \
  -c core.hooksPath=/dev/null \
  commit --quiet -m "fixture: record exact MFEM gitlink"

fixture_mfem_gitlink="$(git -C "${fixture_root}" rev-parse HEAD:tpl/mfem)"
fixture_mfem_head="$(git -C "${fixture_mfem}" rev-parse HEAD)"
fixture_mfem_status="$(GIT_OPTIONAL_LOCKS=0 git -C "${fixture_mfem}" \
  status --porcelain=v1 --untracked-files=no)"
fixture_mfem_entry="$(git -C "${fixture_root}" ls-tree HEAD -- tpl/mfem)"
[[ "${fixture_mfem_gitlink}" == "${real_mfem_gitlink}" ]] || {
  fail "fixture gitlink mismatch: fixture=${fixture_mfem_gitlink} real=${real_mfem_gitlink}"
}
[[ "${fixture_mfem_head}" == "${real_mfem_gitlink}" ]] || {
  fail "fixture MFEM HEAD mismatch: fixture=${fixture_mfem_head} real=${real_mfem_gitlink}"
}
[[ -z "${fixture_mfem_status}" ]] || {
  fail "fixture MFEM source has tracked changes: ${fixture_mfem_status}"
}
[[ "${fixture_mfem_entry}" == \
   "160000 commit ${real_mfem_gitlink}"$'\t'"tpl/mfem" ]] || {
  fail "fixture tpl/mfem is not the exact gitlink: ${fixture_mfem_entry}"
}

cp "${BUILD_DRIVER}" "${fixture_root}/build.sh"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  '{' \
  '  printf "MFEM_SRC=%s\\n" "${MFEM_SRC:?}"' \
  '  printf "MFEM_BUILD=%s\\n" "${MFEM_BUILD:?}"' \
  '  printf "MFEM_INSTALL=%s\\n" "${MFEM_INSTALL:?}"' \
  '  printf "%s\\n" "$@"' \
  '} >"${FAKE_HELPER_LOG:?}"' \
  'exit 77' >"${fixture_root}/build_mfem_tpl.sh"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'printf "spack-called\\n" >>"${FAKE_SPACK_LOG:?}"' \
  'exit 78' >"${fake_bin}/spack"
chmod +x "${fixture_root}/build_mfem_tpl.sh" "${fake_bin}/spack"

clean_env=(
  env
  -u CUDA_ARCHS
  -u MFEM_CUDA_ARCHS
  -u APP_CUDA_ARCHS
  -u SPACK_CUDA_ARCH
  -u DG_IGR_SPACK_ENV
)

# The synthetic exact-source composition exercises normal-mode source
# validation even when the application worktree's submodule is uninitialized.
# The canonical sm120 plan keeps dependencies and project sources native while
# compiling only MFEM through compatibility PTX.
plan="$("${clean_env[@]}" bash "${fixture_root}/build.sh" \
  --gpu --cuda-arch 120 -j 16 --print-plan)"
assert_contains "${plan}" "profile=cuda-sm120"
assert_contains "${plan}" "dependency_cuda_arch=120"
assert_contains "${plan}" "mfem_cuda_archs=80-virtual"
assert_contains "${plan}" "app_cuda_archs=120"
assert_contains "${plan}" "app_jobs=16"
assert_contains "${plan}" "mfem_jobs=16"

plan="$("${clean_env[@]}" bash "${fixture_root}/build.sh" \
  --gpu --cuda-arch 120 -j 16 --mfem-jobs 7 --print-plan)"
assert_contains "${plan}" "app_jobs=16"
assert_contains "${plan}" "mfem_jobs=7"

# CUDA-specific variables remain inert on the CPU path.
plan="$(CUDA_ARCHS=120 MFEM_CUDA_ARCHS=120 APP_CUDA_ARCHS=80-virtual \
  bash "${fixture_root}/build.sh" --cpu-only -j 9 --print-plan)"
assert_contains "${plan}" "mode=cpu profile=cpu dependency_cuda_arch=none"
assert_contains "${plan}" "mfem_cuda_archs=none app_cuda_archs=none"
assert_contains "${plan}" "app_jobs=9 mfem_jobs=9"

# Unsafe architecture selections must fail before dependency, helper, or cache
# actions even without --print-plan.
set +e
reject_output="$(env -u CUDA_ARCHS -u APP_CUDA_ARCHS -u SPACK_CUDA_ARCH \
  PATH="${fake_bin}:/usr/bin:/bin" \
  FAKE_HELPER_LOG="${tmp_root}/unsafe-mfem-helper.log" \
  FAKE_SPACK_LOG="${tmp_root}/unsafe-mfem-spack.log" \
  MFEM_CUDA_ARCHS=120 \
  bash "${fixture_root}/build.sh" --gpu --cuda-arch 120 \
    --clean --no-tests 2>&1)"
reject_rc=$?
set -e
[[ ${reject_rc} -eq 2 ]] || fail "unsafe MFEM override exited ${reject_rc}, expected 2"
assert_contains "${reject_output}" "requires MFEM_CUDA_ARCHS=80-virtual"
assert_not_exists "${tmp_root}/unsafe-mfem-helper.log"
assert_not_exists "${tmp_root}/unsafe-mfem-spack.log"

set +e
reject_output="$(env -u CUDA_ARCHS -u MFEM_CUDA_ARCHS -u SPACK_CUDA_ARCH \
  PATH="${fake_bin}:/usr/bin:/bin" \
  FAKE_HELPER_LOG="${tmp_root}/unsafe-app-helper.log" \
  FAKE_SPACK_LOG="${tmp_root}/unsafe-app-spack.log" \
  APP_CUDA_ARCHS=80-virtual \
  bash "${fixture_root}/build.sh" --gpu --cuda-arch 120 \
    --clean --no-tests 2>&1)"
reject_rc=$?
set -e
[[ ${reject_rc} -eq 2 ]] || fail "unsafe app override exited ${reject_rc}, expected 2"
assert_contains "${reject_output}" "requires APP_CUDA_ARCHS=120"
assert_not_exists "${tmp_root}/unsafe-app-helper.log"
assert_not_exists "${tmp_root}/unsafe-app-spack.log"

set +e
reject_output="$(env -u MFEM_CUDA_ARCHS -u APP_CUDA_ARCHS -u SPACK_CUDA_ARCH \
  PATH="${fake_bin}:/usr/bin:/bin" \
  FAKE_HELPER_LOG="${tmp_root}/legacy-helper.log" \
  FAKE_SPACK_LOG="${tmp_root}/legacy-spack.log" \
  CUDA_ARCHS=120 \
  bash "${fixture_root}/build.sh" --gpu --cuda-arch 120 \
    --clean --no-tests 2>&1)"
reject_rc=$?
set -e
[[ ${reject_rc} -eq 2 ]] || fail "legacy CUDA_ARCHS exited ${reject_rc}, expected 2"
assert_contains "${reject_output}" "CUDA_ARCHS is ambiguous"
assert_not_exists "${tmp_root}/legacy-helper.log"
assert_not_exists "${tmp_root}/legacy-spack.log"

# The MFEM entry point enforces the same policy independently and before its
# clean operation. Sentinels prove an invalid direct call cannot remove caches.
mfem_build="${tmp_root}/mfem-build"
mfem_install="${tmp_root}/mfem-install"
mkdir -p "${mfem_build}" "${mfem_install}"
touch "${mfem_build}/sentinel" "${mfem_install}/sentinel"
set +e
reject_output="$(env -u CUDA_ARCHS -u SPACK_CUDA_ARCH \
  PATH="${fake_bin}:/usr/bin:/bin" \
  FAKE_SPACK_LOG="${tmp_root}/direct-helper-spack.log" \
  MFEM_BUILD="${mfem_build}" MFEM_INSTALL="${mfem_install}" \
  bash "${MFEM_DRIVER}" --gpu --cuda-arch 120 --cuda-codegen 120 \
    --clean 2>&1)"
reject_rc=$?
set -e
[[ ${reject_rc} -eq 2 ]] || fail "unsafe direct helper exited ${reject_rc}, expected 2"
assert_contains "${reject_output}" "requires --cuda-codegen 80-virtual"
[[ -f "${mfem_build}/sentinel" ]] || fail "unsafe helper removed MFEM build sentinel"
[[ -f "${mfem_install}/sentinel" ]] || fail "unsafe helper removed MFEM install sentinel"
assert_not_exists "${tmp_root}/direct-helper-spack.log"

set +e
reject_output="$(env -u SPACK_CUDA_ARCH \
  PATH="${fake_bin}:/usr/bin:/bin" \
  FAKE_SPACK_LOG="${tmp_root}/direct-legacy-spack.log" \
  CUDA_ARCHS=80-virtual \
  bash "${MFEM_DRIVER}" --gpu --cuda-arch 120 --cuda-codegen 80-virtual \
    --clean 2>&1)"
reject_rc=$?
set -e
[[ ${reject_rc} -eq 2 ]] || fail "direct legacy CUDA_ARCHS exited ${reject_rc}, expected 2"
assert_contains "${reject_output}" "CUDA_ARCHS is ambiguous"
assert_not_exists "${tmp_root}/direct-legacy-spack.log"

mfem_plan="$("${clean_env[@]}" bash "${MFEM_DRIVER}" \
  --gpu --cuda-arch 120 --cuda-codegen 80-virtual -j 7 --print-plan)"
assert_contains "${mfem_plan}" "profile=cuda-sm120"
assert_contains "${mfem_plan}" "dependency_cuda_arch=120"
assert_contains "${mfem_plan}" "cuda_codegen_archs=80-virtual jobs=7"

# A fake helper captures the canonical driver's explicit forwarding without
# allowing the test to reach CMake or any build command.
set +e
env -u CUDA_ARCHS -u MFEM_CUDA_ARCHS -u APP_CUDA_ARCHS -u SPACK_CUDA_ARCH \
  PATH="${fake_bin}:/usr/bin:/bin" \
  FAKE_HELPER_LOG="${tmp_root}/forwarded-args.log" \
  FAKE_SPACK_LOG="${tmp_root}/forwarded-spack.log" \
  DG_IGR_SPACK_ENV="${fixture_root}/spack-envs/cuda-sm120" \
  bash "${fixture_root}/build.sh" --gpu --cuda-arch 120 \
  -j 16 --mfem-jobs 7 --no-clean --no-deps --no-tests \
  >"${tmp_root}/forwarding-output.log" 2>&1
forward_rc=$?
set -e
[[ ${forward_rc} -eq 77 ]] || fail "fake-helper forwarding exited ${forward_rc}, expected 77"
forwarded_log="$(<"${tmp_root}/forwarded-args.log")"
assert_contains "${forwarded_log}" "MFEM_SRC=${fixture_mfem}"
assert_contains "${forwarded_log}" \
  "MFEM_BUILD=${fixture_mfem}/build-gpu-sm120"
assert_contains "${forwarded_log}" \
  "MFEM_INSTALL=${fixture_mfem}/install-gpu-sm120"
assert_adjacent_args "${tmp_root}/forwarded-args.log" "--jobs" "7"
assert_adjacent_args "${tmp_root}/forwarded-args.log" "--cuda-arch" "120"
assert_adjacent_args "${tmp_root}/forwarded-args.log" "--cuda-codegen" "80-virtual"
assert_not_exists "${tmp_root}/forwarded-spack.log"

# The reuse tuple is revalidated only after a real dependency install and
# before any view export or build consumer can observe the transition.
assert_reuse_transaction_order "${BUILD_DRIVER}"

# Keep the application-level fallback and gate native to the selected profile.
if grep -Fq 'set(_dg_igr_default_cuda_arch "80-virtual")' "${REPO_ROOT}/CMakeLists.txt"; then
  fail "application CMake still defaults sm120 to 80-virtual"
fi
if grep -Fq '_dg_igr_cuda_arch_compat' "${REPO_ROOT}/CMakeLists.txt"; then
  fail "application CMake still admits the MFEM compatibility architecture"
fi
if grep -Fq '_dg_igr_arch STREQUAL "${_dg_igr_expected_arch}-virtual"' \
  "${REPO_ROOT}/CMakeLists.txt"; then
  fail "application CMake still admits virtual-only profile code generation"
fi
if ! grep -Fq '_dg_igr_arch STREQUAL "${_dg_igr_expected_arch}-real"' \
  "${REPO_ROOT}/CMakeLists.txt"; then
  fail "application CMake no longer admits native real code generation"
fi

echo "cuda-build-policy: PASS"
