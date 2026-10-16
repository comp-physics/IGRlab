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

plan_field() {
  local plan="$1" key="$2"
  sed -n "s/.*${key}=\\([^ ]*\\).*/\\1/p" <<<"${plan}" | tail -n 1
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

assert_stamped_reuse_scope_policy() {
  local driver="$1" init_body runtime_body

  init_body="$(awk '
    /^initialize_scoped_output_layout\(\)/ { capture = 1 }
    capture { print }
    capture && /^}/ { exit }
  ' "${driver}")"
  runtime_body="$(awk '
    /^if \[\[ "\$\{MFEM_REUSE_STATE\}" == "validated-read-only" \]\]; then/ { capture = 1 }
    capture { print }
    capture && /^CMAKE_BIN=/ { exit }
  ' "${driver}")"
  [[ "${init_body}" == *'application build scope'* &&
     "${init_body}" == *'if [[ "${MFEM_REUSE_STATE}" != "validated-read-only" && -z "${MFEM_OUTPUT_ROOT_REQUEST}" ]]; then'* &&
     "${init_body}" == *'MFEM build scope'* ]] || {
    fail "stamped reuse scope policy no longer separates app and MFEM paths"
  }
  [[ "${runtime_body}" == *'if [[ "${MFEM_REUSE_STATE}" == "validated-read-only" ]]; then'* &&
     "${runtime_body}" == *'else'* &&
     "${runtime_body}" == *'rm -rf "${MFEM_BUILD}" "${MFEM_INSTALL}"'* ]] || {
    fail "stamped reuse runtime route no longer keeps legacy MFEM cleanup out of the reuse branch"
  }
}

assert_reviewed_admission_route() {
  local driver="$1"
  local route="$2"
  local record_reference="$3"
  local sha_reference="$4"
  local route_case

  route_case="$(awk -v route="${route}" '
    $0 == "    " route ")" { capture = 1 }
    capture { print }
    capture && $0 == "      ;;" { exit }
  ' "${driver}")"
  [[ -n "${route_case}" ]] || {
    fail "build driver does not define reviewed admission route ${route}"
  }
  assert_contains "${route_case}" "${record_reference}"
  assert_contains "${route_case}" "${sha_reference}"
}

assert_reviewed_admission_record() {
  local record="$1"
  local expected_sha="$2"
  local route_mode="$3"
  local route_profile="$4"
  local anchor_key="$5"
  local anchor_sha="$6"
  local label="$7"
  local record_sha altered_record altered_sha

  [[ -f "${record}" && ! -L "${record}" ]] || {
    fail "${label} reviewed MFEM admission record is missing or symlinked: ${record}"
  }
  record_sha="$(sha256sum "${record}" | cut -d' ' -f1)"
  [[ "${record_sha}" == "${expected_sha}" ]] || {
    fail "${label} reviewed MFEM admission record hash mismatch: ${record_sha}"
  }
  grep -Fq "\"mode\": \"${route_mode}\"" "${record}" || {
    fail "${label} reviewed MFEM admission mode is not ${route_mode}"
  }
  grep -Fq "\"profile\": \"${route_profile}\"" "${record}" || {
    fail "${label} reviewed MFEM admission profile is not ${route_profile}"
  }
  grep -Fq \
    '"claim": "current tuple observed, independently reviewed, prospectively admitted for reuse; raw W8A ledgers are corroboration, not retroactive immutable proof"' \
    "${record}" || fail "${label} reviewed MFEM admission claim changed"
  grep -Fq "\"${anchor_key}\": \"${anchor_sha}\"" "${record}" || {
    fail "${label} reviewed MFEM admission lost its DEV-3 anchor"
  }

  altered_record="${tmp_root}/altered-${label}-reviewed-mfem-admission.json"
  sed 's/"version": "1"/"version": "0"/' \
    "${record}" >"${altered_record}"
  altered_sha="$(sha256sum "${altered_record}" | cut -d' ' -f1)"
  [[ "${altered_sha}" != "${record_sha}" ]] || {
    fail "${label} reviewed MFEM admission mismatch fixture retained the admitted hash"
  }
}

assert_reuse_identity_policy() {
  local driver="$1"
  local cpu_record="$2"
  local gpu_record="$3"
  local expected
  local cpu_sha gpu_sha

  cpu_sha="d4e0f68d795d30ecb3e403f797ecf4ccd4ec8ed79ba5746a997f68bd6eae98ec"
  gpu_sha="00559ec1cdd0ad5728a0ff92a165fdbf7d757da6f9c672153d8a665cc9e489a9"

  for expected in \
      'cmake_install.cmake' \
      'CMakeFiles/cmake.check_cache' \
      'MFEMTargets.cmake' \
      'MFEMTargets-${targets_config_suffix}.cmake' \
      'CMakeCXXCompiler.cmake' \
      'require_exact_mfem_directory' \
      'require_exact_mfem_file' \
      'write_local_mfem_identity_stamp' \
      'verify_reviewed_mfem_admission' \
      'mfem_reviewed_admission_record_matches' \
      'MFEM_REUSE_POST_MANIFEST_SHA256' \
      'MFEM_REUSE_PRE_MANIFEST_SHA256' \
      'MFEM_REUSE_POST_LOCK_SHA256' \
      'MFEM_REUSE_PRE_LOCK_SHA256' \
      'export DG_IGR_TEST_REUSE_PROFILE="${PROFILE}"' \
      'export DG_IGR_TEST_REUSE_CALIPER="${CALIPER}"' \
      'export DG_IGR_TEST_REUSE_CUDA_ARCH="${DEPENDENCY_CUDA_ARCH}"'; do
    grep -Fq "${expected}" "${driver}" || {
      fail "reuse identity policy is missing '${expected}'"
    }
  done
  if grep -Fq 'make_reuse_fixture' "${driver}" ||
     grep -Fq 'DG_IGR_REUSE_FIXTURE_CACHE' "${driver}" ||
     grep -Fq 'cp "${source_build}/CMakeCache.txt"' "${driver}"; then
    fail "reuse fixtures still reconstruct an accepted MFEM tuple"
  fi

  grep -Fq \
    'MFEM_CPU_REVIEWED_ADMISSION_RECORD="${SCRIPT_DIR}/cmake/mfem-reuse-identities/cpu-w8a-d9d6526-973e5abc.json"' \
    "${driver}" || fail "build driver does not pin the CPU admission record"
  grep -Fq \
    "MFEM_CPU_REVIEWED_ADMISSION_SHA256=\"${cpu_sha}\"" \
    "${driver}" || fail "build driver does not pin the CPU admission hash"
  grep -Fq \
    'MFEM_GPU_SM120_REVIEWED_ADMISSION_RECORD="${SCRIPT_DIR}/cmake/mfem-reuse-identities/gpu-w8a-d9d6526-345d64a2.json"' \
    "${driver}" || fail "build driver does not pin the GPU admission record"
  grep -Fq \
    "MFEM_GPU_SM120_REVIEWED_ADMISSION_SHA256=\"${gpu_sha}\"" \
    "${driver}" || fail "build driver does not pin the GPU admission hash"
  [[ "${cpu_sha}" != "${gpu_sha}" ]] || {
    fail "CPU and GPU reviewed MFEM admissions share one hash"
  }

  assert_reviewed_admission_route "${driver}" cpu:cpu \
    '${MFEM_CPU_REVIEWED_ADMISSION_RECORD}' \
    '${MFEM_CPU_REVIEWED_ADMISSION_SHA256}'
  assert_reviewed_admission_route "${driver}" gpu:cuda-sm120 \
    '${MFEM_GPU_SM120_REVIEWED_ADMISSION_RECORD}' \
    '${MFEM_GPU_SM120_REVIEWED_ADMISSION_SHA256}'
  grep -Fq 'assert_reviewed_admission_route_rejected gpu cpu' \
    "${driver}" || fail "reviewed admission routes lost the gpu/cpu negative anchor"
  grep -Fq 'assert_reviewed_admission_route_rejected cpu cuda-sm120' \
    "${driver}" || fail "reviewed admission routes lost the cpu/gpu negative anchor"
  grep -Fq 'assert_reviewed_admission_route_rejected gpu cuda-sm90' \
    "${driver}" || fail "reviewed admission routes lost the unsupported-profile anchor"

  assert_reviewed_admission_record "${cpu_record}" \
    "${cpu_sha}" cpu cpu sealed_dev3_cpu_manifest_sha256 \
    008605ffe7d0192689a6dcdcebfcee7fcac4ead307ff6c6403ee36731ce2d59d cpu
  assert_reviewed_admission_record "${gpu_record}" \
    "${gpu_sha}" gpu cuda-sm120 sealed_dev3_gpu_manifest_sha256 \
    ccb280318581f926b2bb65d98581868a29075ae4ba4f1ddf15eb24cf8939c059 gpu-sm120
}

[[ -f "${BUILD_DRIVER}" ]] || fail "missing build driver: ${BUILD_DRIVER}"
[[ -f "${MFEM_DRIVER}" ]] || fail "missing MFEM driver: ${MFEM_DRIVER}"

if [[ -n "${IGRLAB_TEST_MFEM_SOURCE_ROOT:-}" ]]; then
  mfem_source="${IGRLAB_TEST_MFEM_SOURCE_ROOT}"
elif [[ -n "${DG_IGR_TEST_REUSE_MFEM_ROOT:-}" ]]; then
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
         "${fixture_root}/spack-envs/cpu/.spack-env/view" \
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
mkdir -p "${fixture_root}/scripts/build"
cp "${REPO_ROOT}/scripts/build/prebuilt_validation.py" \
  "${REPO_ROOT}/scripts/build/mfem_cache.sh" "${fixture_root}/scripts/build/"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  '{' \
  '  printf "MFEM_SRC=%s\\n" "${MFEM_SRC:?}"' \
  '  printf "MFEM_BUILD=%s\\n" "${MFEM_BUILD:?}"' \
  '  printf "MFEM_INSTALL=%s\\n" "${MFEM_INSTALL:?}"' \
  '  printf "IGRLAB_TEST_MFEM_SOURCE_ROOT=%s\\n" "${IGRLAB_TEST_MFEM_SOURCE_ROOT:-}"' \
  '  for name in DG_IGR_TEST_REUSE_MFEM_ROOT DG_IGR_TEST_REUSE_SPACK_ENV DG_IGR_TEST_REUSE_MODE DG_IGR_TEST_REUSE_PROFILE DG_IGR_TEST_REUSE_CALIPER DG_IGR_TEST_REUSE_CUDA_ARCH; do' \
  '    printf "%s=%s\\n" "$name" "${!name-}"' \
  '  done' \
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
  -u DG_IGR_OUTPUT_ROOT
  -u IGRLAB_SPACK_ENV
  -u IGRLAB_OUTPUT_ROOT
  -u IGRLAB_ACCELERATOR_KERNEL_SOURCES
  -u IGRLAB_RUN_REUSE_FIXTURES
  -u IGRLAB_BUILD_RESEARCH_TESTS
  -u DG_IGR_BUILD_RESEARCH_TESTS
  -u IGRLAB_CUDA_ARCH
  -u IGRLAB_MFEM_CUDA_ARCHS
  -u IGRLAB_APP_CUDA_ARCHS
  -u IGRLAB_VALIDATION_MODE
)

# The synthetic exact-source composition exercises normal-mode source
# validation even when the application worktree's submodule is uninitialized.
# The canonical sm120 plan keeps dependencies and project sources native while
# compiling only MFEM through compatibility PTX.
default_cache_root="${tmp_root}/default-cache"
default_output_root="${default_cache_root}/igrlab-build"
plan="$("${clean_env[@]}" XDG_CACHE_HOME="${default_cache_root}" \
  bash "${fixture_root}/build.sh" --gpu --cuda-arch 120 -j 16 --print-plan)"
assert_contains "${plan}" "profile=cuda-sm120"
assert_contains "${plan}" "dependency_cuda_arch=120"
assert_contains "${plan}" "mfem_cuda_archs=80-virtual"
assert_contains "${plan}" "app_cuda_archs=120"
assert_contains "${plan}" "app_jobs=16"
assert_contains "${plan}" "mfem_jobs=16"
assert_contains "${plan}" "output_root=${default_output_root}"
assert_not_exists "${default_output_root}"

# Iteration preserves the application build scope and skips dependency setup.
iteration_plan="$("${clean_env[@]}" XDG_CACHE_HOME="${default_cache_root}" \
  bash "${fixture_root}/build.sh" --gpu --cuda-arch 120 --iterate --print-plan)"
assert_contains "${iteration_plan}" "route=app-only dependency_work=none artifact_hashes=none run_tests=0"
[[ "$(plan_field "${iteration_plan}" scope)" == "$(plan_field "${plan}" scope)" ]] || \
  fail "iteration changes the persistent build scope"
ambient_plan="$("${clean_env[@]}" IGRLAB_VALIDATION_MODE=iteration \
  bash "${fixture_root}/build.sh" --cpu-only --print-plan)"
assert_contains "${ambient_plan}" 'clean=1 install_deps=1'
for filter in regex label no-tests; do
  case "${filter}" in
    regex) iteration_args=(--tests-only --test-regex '^focused$') ;;
    label) iteration_args=(--tests-only --test-label '^special$') ;;
    no-tests) iteration_args=(--no-tests) ;;
  esac
  filtered_plan="$("${clean_env[@]}" bash "${fixture_root}/build.sh" \
    --cpu-only --iterate "${iteration_args[@]}" --print-plan)"
  [[ "${filtered_plan}" != *'test_label=^numerical-smoke$'* ]] || \
    fail "iteration overrides an explicit filter or adds tests to a build-only plan"
done
for clean_order in first last overwritten; do
  case "${clean_order}" in
    first) iteration_args=(--clean --iterate) ;;
    last) iteration_args=(--iterate --clean) ;;
    overwritten) iteration_args=(--clean --no-clean --iterate) ;;
  esac
  if conflict_plan="$("${clean_env[@]}" bash "${fixture_root}/build.sh" \
    --cpu-only "${iteration_args[@]}" --print-plan 2>&1)"; then
    fail "iteration accepts explicit --clean (${clean_order})"
  fi
  assert_contains "${conflict_plan}" '--iterate conflicts with --clean'
done
assert_not_exists "${default_output_root}"

# Canonical build names preserve legacy scripts and select explicit overrides.
canonical_output_root="${tmp_root}/canonical-output"
legacy_output_root="${tmp_root}/legacy-output"
canonical_plan="$("${clean_env[@]}" IGRLAB_OUTPUT_ROOT="${canonical_output_root}" \
  DG_IGR_OUTPUT_ROOT="${legacy_output_root}" IGRLAB_CUDA_ARCH=80 \
  bash "${fixture_root}/build.sh" --gpu --print-plan)"
assert_contains "${canonical_plan}" "profile=cuda-sm80"
assert_contains "${canonical_plan}" "output_root=${canonical_output_root}"
assert_not_exists "${canonical_output_root}"
assert_not_exists "${legacy_output_root}"
cli_plan="$("${clean_env[@]}" IGRLAB_OUTPUT_ROOT="${canonical_output_root}" \
  IGRLAB_CUDA_ARCH=80 bash "${fixture_root}/build.sh" --gpu --cuda-arch 120 \
  --output-root "${legacy_output_root}" --print-plan)"
assert_contains "${cli_plan}" "profile=cuda-sm120"
assert_contains "${cli_plan}" "output_root=${legacy_output_root}"
research_plan="$("${clean_env[@]}" IGRLAB_BUILD_RESEARCH_TESTS=ON \
  bash "${fixture_root}/build.sh" --cpu-only --print-plan)"
assert_contains "${research_plan}" "research_tests=ON"
research_off_plan="$("${clean_env[@]}" IGRLAB_BUILD_RESEARCH_TESTS=ON \
  bash "${fixture_root}/build.sh" --cpu-only --no-research-tests --print-plan)"
assert_contains "${research_off_plan}" "research_tests=OFF"

# A100's friendly selector and numeric interface resolve the same native tuple.
a100_output_root="${tmp_root}/a100-output"
a100_plan="$("${clean_env[@]}" bash "${fixture_root}/build.sh" \
  --a100 --hdg-runtime --output-root "${a100_output_root}" --print-plan)"
numeric_a100_plan="$("${clean_env[@]}" bash "${fixture_root}/build.sh" \
  --gpu --cuda-arch 80 --hdg-runtime --output-root "${a100_output_root}" --print-plan)"
[[ "${a100_plan}" == "${numeric_a100_plan}" ]] || fail "A100 alias changes the native SM80 plan"
assert_contains "${a100_plan}" "mode=gpu profile=cuda-sm80 dependency_cuda_arch=80 mfem_cuda_archs=80 app_cuda_archs=80"
assert_contains "${a100_plan}" "hdg_runtime=ON"
assert_contains "${a100_plan}" "env=${fixture_root}/spack-envs/cuda-sm80"
assert_not_exists "${a100_output_root}"
# Native SM90 uses the supplied environment; this does not imply a site adapter.
h100_output_root="${tmp_root}/h100-output"
h100_spack_env="${tmp_root}/h100-spack-env"
h100_plan="$("${clean_env[@]}" IGRLAB_SPACK_ENV="${h100_spack_env}" \
  bash "${fixture_root}/build.sh" --gpu --cuda-arch 90 --hdg-runtime \
  --output-root "${h100_output_root}" --print-plan)"
assert_contains "${h100_plan}" "mode=gpu profile=cuda-sm90 dependency_cuda_arch=90 mfem_cuda_archs=90 app_cuda_archs=90"
assert_contains "${h100_plan}" "hdg_runtime=ON"
assert_contains "${h100_plan}" "env=${h100_spack_env}"
assert_not_exists "${h100_output_root}"
assert_not_exists "${h100_spack_env}"
# A command-line architecture selection overrides the environment default.
plan="$("${clean_env[@]}" SPACK_CUDA_ARCH=120 bash "${fixture_root}/build.sh" \
  --a100 --gpu --cuda-arch 80 --print-plan)"
assert_contains "${plan}" "profile=cuda-sm80"
v100_output_root="${tmp_root}/v100-output"
v100_plan="$("${clean_env[@]}" bash "${fixture_root}/build.sh" \
  --gpu --cuda-arch 70 --output-root "${v100_output_root}" --print-plan)"
assert_contains "${v100_plan}" "mode=gpu profile=cuda-sm70 dependency_cuda_arch=70 mfem_cuda_archs=70 app_cuda_archs=70"
assert_contains "${v100_plan}" "hdg_runtime=OFF"
assert_not_exists "${v100_output_root}"
for conflict in cpu-first cpu-last arch-first arch-last overwritten-arch hdg-unsupported hdg-v100; do
  case "${conflict}" in
    cpu-first) conflict_args=(--cpu-only --a100) ;;
    cpu-last) conflict_args=(--a100 --cpu) ;;
    arch-first) conflict_args=(--cuda-arch 120 --a100) ;;
    arch-last) conflict_args=(--a100 --cuda-arch 120) ;;
    overwritten-arch) conflict_args=(--cuda-arch 120 --cuda-arch 80 --a100) ;;
    hdg-unsupported) conflict_args=(--gpu --cuda-arch 100 --hdg-runtime) ;;
    hdg-v100) conflict_args=(--gpu --cuda-arch 70 --hdg-runtime) ;;
  esac
  set +e
  reject_output="$("${clean_env[@]}" PATH="${fake_bin}:/usr/bin:/bin" \
    FAKE_HELPER_LOG="${tmp_root}/a100-${conflict}-helper.log" \
    FAKE_SPACK_LOG="${tmp_root}/a100-${conflict}-spack.log" \
    bash "${fixture_root}/build.sh" "${conflict_args[@]}" 2>&1)"
  reject_rc=$?
  set -e
  [[ ${reject_rc} -eq 2 ]] || fail "A100 conflict ${conflict} exited ${reject_rc}, expected 2"
  if [[ "${conflict}" == hdg-* ]]; then
    assert_contains "${reject_output}" "--hdg-runtime requires native CUDA architecture 80, 90, or 120"
  else
    assert_contains "${reject_output}" "--a100 conflicts"
  fi
  assert_not_exists "${tmp_root}/a100-${conflict}-helper.log"
  assert_not_exists "${tmp_root}/a100-${conflict}-spack.log"
done

explicit_output_root="${tmp_root}/explicit-output"
plan="$("${clean_env[@]}" bash "${fixture_root}/build.sh" \
  --gpu --cuda-arch 120 --output-root "${explicit_output_root}" --print-plan)"
scope="$(plan_field "${plan}" scope)"
[[ "${scope}" == "${explicit_output_root}/worktrees/"*/cuda-sm120 ]] || {
  fail "explicit output scope is not worktree/profile scoped: ${scope}"
}
assert_contains "${plan}" "app_build=${scope}/app/build"
assert_contains "${plan}" "app_install=${scope}/app/install"
assert_contains "${plan}" "mfem_build=${scope}/mfem/build"
assert_contains "${plan}" "mfem_install=${scope}/mfem/install"
assert_not_exists "${explicit_output_root}"

env_output_root="${tmp_root}/env-output"
cli_output_root="${tmp_root}/cli-output"
plan="$("${clean_env[@]}" DG_IGR_OUTPUT_ROOT="${env_output_root}" \
  bash "${fixture_root}/build.sh" --gpu --cuda-arch 120 \
    --output-root "${cli_output_root}" --print-plan)"
assert_contains "${plan}" "output_root=${cli_output_root}"
assert_not_exists "${cli_output_root}"
plan="$("${clean_env[@]}" DG_IGR_OUTPUT_ROOT="${env_output_root}" \
  bash "${fixture_root}/build.sh" --gpu --cuda-arch 120 --print-plan)"
assert_contains "${plan}" "output_root=${env_output_root}"
assert_not_exists "${env_output_root}"

isolation_output_root="${tmp_root}/profile-isolation"
gpu_scope="$(plan_field "$("${clean_env[@]}" bash "${fixture_root}/build.sh" \
  --gpu --cuda-arch 120 --output-root "${isolation_output_root}" --print-plan)" scope)"
cpu_scope="$(plan_field "$("${clean_env[@]}" bash "${fixture_root}/build.sh" \
  --cpu-only --output-root "${isolation_output_root}" --print-plan)" scope)"
[[ "${gpu_scope}" != "${cpu_scope}" && "${gpu_scope}" == */cuda-sm120 &&
   "${cpu_scope}" == */cpu ]] || {
  fail "worktree/profile scopes are not isolated: gpu=${gpu_scope} cpu=${cpu_scope}"
}
assert_not_exists "${isolation_output_root}"

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
reject_output="$(PATH="${fake_bin}:/usr/bin:/bin" \
  FAKE_HELPER_LOG="${tmp_root}/empty-reuse-helper.log" \
  FAKE_SPACK_LOG="${tmp_root}/empty-reuse-spack.log" \
  bash "${fixture_root}/build.sh" --reuse-mfem-root '' \
    --no-deps --no-tests --print-plan 2>&1)"
reject_rc=$?
set -e
[[ ${reject_rc} -eq 2 ]] || fail "empty reuse root exited ${reject_rc}, expected 2"
assert_contains "${reject_output}" \
  "--reuse-mfem-root requires a non-empty value"
assert_not_exists "${tmp_root}/empty-reuse-helper.log"
assert_not_exists "${tmp_root}/empty-reuse-spack.log"

# An output root within a checkout is rejected before any helper or Spack path.
set +e
reject_output="$(PATH="${fake_bin}:/usr/bin:/bin" \
  FAKE_HELPER_LOG="${tmp_root}/checkout-output-helper.log" \
  FAKE_SPACK_LOG="${tmp_root}/checkout-output-spack.log" \
  bash "${fixture_root}/build.sh" --gpu --cuda-arch 120 \
    --output-root "${fixture_root}/forbidden-output" \
    --no-deps --no-tests --print-plan 2>&1)"
reject_rc=$?
set -e
[[ ${reject_rc} -eq 3 ]] || fail "checkout output root exited ${reject_rc}, expected 3"
assert_contains "${reject_output}" "output path must be outside detected Git checkouts"
assert_not_exists "${tmp_root}/checkout-output-helper.log"
assert_not_exists "${tmp_root}/checkout-output-spack.log"

set +e
reject_output="$(PATH="${fake_bin}:/usr/bin:/bin" \
  FAKE_HELPER_LOG="${tmp_root}/env-checkout-output-helper.log" \
  FAKE_SPACK_LOG="${tmp_root}/env-checkout-output-spack.log" \
  DG_IGR_OUTPUT_ROOT="${fixture_root}/forbidden-env-output" \
  bash "${fixture_root}/build.sh" --gpu --cuda-arch 120 \
    --no-deps --no-tests --print-plan 2>&1)"
reject_rc=$?
set -e
[[ ${reject_rc} -eq 3 ]] || fail "environment checkout output root exited ${reject_rc}, expected 3"
assert_contains "${reject_output}" "output path must be outside detected Git checkouts"
assert_not_exists "${tmp_root}/env-checkout-output-helper.log"
assert_not_exists "${tmp_root}/env-checkout-output-spack.log"

set +e
reject_output="$(PATH="${fake_bin}:/usr/bin:/bin" \
  FAKE_HELPER_LOG="${tmp_root}/relative-output-helper.log" \
  FAKE_SPACK_LOG="${tmp_root}/relative-output-spack.log" \
  bash "${fixture_root}/build.sh" --gpu --cuda-arch 120 \
    --output-root relative-output --no-deps --no-tests --print-plan 2>&1)"
reject_rc=$?
set -e
[[ ${reject_rc} -eq 3 ]] || fail "relative output root exited ${reject_rc}, expected 3"
assert_contains "${reject_output}" "output path must be absolute"
assert_not_exists "${tmp_root}/relative-output-helper.log"
assert_not_exists "${tmp_root}/relative-output-spack.log"

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

# The private helper rejects clean even when its advisory caller marker is set.
mfem_build="${tmp_root}/mfem-build"
mfem_install="${tmp_root}/mfem-install"
mkdir -p "${mfem_build}" "${mfem_install}"
touch "${mfem_build}/sentinel" "${mfem_install}/sentinel"
helper_caller="$(cd -- "$(dirname -- "${MFEM_DRIVER}")" && pwd)/build.sh"
set +e
reject_output="$(PATH="${fake_bin}:/usr/bin:/bin" \
  DG_IGR_BUILD_HELPER_CALLER="${helper_caller}" \
  DG_IGR_SPACK_ENV="${tmp_root}/spoofed-helper-spack" \
  MFEM_SRC="${fixture_mfem}" MFEM_BUILD="${mfem_build}" \
  MFEM_INSTALL="${mfem_install}" \
  bash "${MFEM_DRIVER}" --gpu --cuda-arch 120 \
    --cuda-codegen 80-virtual --clean 2>&1)"
reject_rc=$?
set -e
[[ ${reject_rc} -eq 2 ]] || fail "spoofed helper clean exited ${reject_rc}, expected 2"
assert_contains "${reject_output}" "private helper and rejects --clean"
[[ -f "${mfem_build}/sentinel" ]] || fail "spoofed helper removed MFEM build sentinel"
[[ -f "${mfem_install}/sentinel" ]] || fail "spoofed helper removed MFEM install sentinel"

contained_build="${fixture_mfem}/direct-build"
contained_install="${fixture_mfem}/direct-install"
mkdir -p "${contained_build}" "${contained_install}"
touch "${contained_build}/sentinel" "${contained_install}/sentinel"
set +e
reject_output="$(PATH="${fake_bin}:/usr/bin:/bin" \
  DG_IGR_BUILD_HELPER_CALLER="${helper_caller}" \
  DG_IGR_SPACK_ENV="${tmp_root}/contained-helper-spack" \
  MFEM_SRC="${fixture_mfem}" MFEM_BUILD="${contained_build}" \
  MFEM_INSTALL="${contained_install}" \
  FAKE_SPACK_LOG="${tmp_root}/contained-helper-spack.log" \
  bash "${MFEM_DRIVER}" --cpu-only --no-clean 2>&1)"
reject_rc=$?
set -e
[[ ${reject_rc} -eq 2 ]] || fail "source-contained helper exited ${reject_rc}, expected 2"
assert_contains "${reject_output}" "rejects MFEM_BUILD inside MFEM_SRC"
[[ -f "${contained_build}/sentinel" && -f "${contained_install}/sentinel" ]] || \
  fail "source-contained helper altered a sentinel"
assert_not_exists "${tmp_root}/contained-helper-spack.log"

# Scoped component validation rejects intermediate app/MFEM build symlinks
# before Spack, the helper, or cleanup can reach their sentinel targets.
app_link_output_root="${tmp_root}/app-link-output"
app_link_plan="$("${clean_env[@]}" bash "${fixture_root}/build.sh" \
  --gpu --cuda-arch 120 --output-root "${app_link_output_root}" --print-plan)"
app_link_scope="$(plan_field "${app_link_plan}" scope)"
app_link_target="${tmp_root}/app-link-target"
mkdir -p "${app_link_scope}/app" "${app_link_target}"
printf 'app-link-sentinel\n' >"${app_link_target}/sentinel"
ln -s "${app_link_target}" "${app_link_scope}/app/build"
set +e
reject_output="$(PATH="${fake_bin}:/usr/bin:/bin" \
  FAKE_HELPER_LOG="${tmp_root}/app-link-helper.log" \
  FAKE_SPACK_LOG="${tmp_root}/app-link-spack.log" \
  bash "${fixture_root}/build.sh" --gpu --cuda-arch 120 \
    --output-root "${app_link_output_root}" --clean --no-deps --no-tests 2>&1)"
reject_rc=$?
set -e
[[ ${reject_rc} -eq 3 ]] || fail "app symlink scope exited ${reject_rc}, expected 3"
assert_contains "${reject_output}" "application build scope has a symlink component"
[[ "$(<"${app_link_target}/sentinel")" == "app-link-sentinel" ]] || \
  fail "app symlink target sentinel changed"
assert_not_exists "${tmp_root}/app-link-helper.log"
assert_not_exists "${tmp_root}/app-link-spack.log"

mfem_link_output_root="${tmp_root}/mfem-link-output"
mfem_link_plan="$("${clean_env[@]}" bash "${fixture_root}/build.sh" \
  --gpu --cuda-arch 120 --output-root "${mfem_link_output_root}" --print-plan)"
mfem_link_scope="$(plan_field "${mfem_link_plan}" scope)"
mfem_link_target="${tmp_root}/mfem-link-target"
mkdir -p "${mfem_link_scope}/mfem" "${mfem_link_target}"
printf 'mfem-link-sentinel\n' >"${mfem_link_target}/sentinel"
ln -s "${mfem_link_target}" "${mfem_link_scope}/mfem/build"
set +e
reject_output="$(PATH="${fake_bin}:/usr/bin:/bin" \
  FAKE_HELPER_LOG="${tmp_root}/mfem-link-helper.log" \
  FAKE_SPACK_LOG="${tmp_root}/mfem-link-spack.log" \
  bash "${fixture_root}/build.sh" --gpu --cuda-arch 120 \
    --output-root "${mfem_link_output_root}" --clean --no-deps --no-tests 2>&1)"
reject_rc=$?
set -e
[[ ${reject_rc} -eq 3 ]] || fail "MFEM symlink scope exited ${reject_rc}, expected 3"
assert_contains "${reject_output}" "MFEM build scope has a symlink component"
[[ "$(<"${mfem_link_target}/sentinel")" == "mfem-link-sentinel" ]] || \
  fail "MFEM symlink target sentinel changed"
assert_not_exists "${tmp_root}/mfem-link-helper.log"
assert_not_exists "${tmp_root}/mfem-link-spack.log"

# A fake helper captures the canonical driver's explicit forwarding without
# allowing the test to reach CMake or any build command.
forward_output_root="${tmp_root}/forward-output"
forward_spack_env="${tmp_root}/forward-spack-env"
mkdir -p "${forward_spack_env}/.spack-env/view"
cp "${REPO_ROOT}/spack-envs/cuda-sm120/spack.yaml" \
  "${forward_spack_env}/spack.yaml"
set +e
env -u CUDA_ARCHS -u MFEM_CUDA_ARCHS -u APP_CUDA_ARCHS -u SPACK_CUDA_ARCH \
  PATH="${fake_bin}:/usr/bin:/bin" \
  FAKE_HELPER_LOG="${tmp_root}/forwarded-args.log" \
  FAKE_SPACK_LOG="${tmp_root}/forwarded-spack.log" \
  DG_IGR_SPACK_ENV="${forward_spack_env}" \
  bash "${fixture_root}/build.sh" --gpu --cuda-arch 120 \
  --output-root "${forward_output_root}" \
  -j 16 --mfem-jobs 7 --no-clean --no-deps --no-tests \
  >"${tmp_root}/forwarding-output.log" 2>&1
forward_rc=$?
set -e
[[ ${forward_rc} -eq 77 ]] || fail "fake-helper forwarding exited ${forward_rc}, expected 77"
forwarded_log="$(<"${tmp_root}/forwarded-args.log")"
forward_scope="$(plan_field "$(<"${tmp_root}/forwarding-output.log")" scope)"
[[ "${forward_scope}" == "${forward_output_root}/worktrees/"*/cuda-sm120 ]] || {
  fail "forwarded output scope is not worktree/profile scoped: ${forward_scope}"
}
assert_contains "${forwarded_log}" "MFEM_SRC=${fixture_mfem}"
assert_contains "${forwarded_log}" \
  "MFEM_BUILD=${forward_scope}/mfem/build"
assert_contains "${forwarded_log}" \
  "MFEM_INSTALL=${forward_scope}/mfem/install"
assert_adjacent_args "${tmp_root}/forwarded-args.log" "--jobs" "7"
assert_adjacent_args "${tmp_root}/forwarded-args.log" "--cuda-arch" "120"
assert_adjacent_args "${tmp_root}/forwarded-args.log" "--cuda-codegen" "80-virtual"
assert_not_exists "${tmp_root}/forwarded-spack.log"
assert_not_exists "${fixture_root}/build-deps"

# Fresh external MFEM sources must reach the policy fixture without admitting
# the unrelated full build/install reuse fixtures.
external_app="${tmp_root}/external-app"
external_mfem="${tmp_root}/external-mfem"
git clone --quiet --shared "${fixture_root}" "${external_app}"
git -c advice.detachedHead=false clone --quiet --shared --no-checkout \
  "${mfem_source}" "${external_mfem}"
git -c advice.detachedHead=false -C "${external_mfem}" checkout --quiet \
  --detach "${real_mfem_gitlink}"
git -C "${external_mfem}" remote set-url origin \
  https://github.com/mfem/mfem.git
cp "${fixture_root}/build.sh" "${fixture_root}/build_mfem_tpl.sh" \
  "${external_app}/"
mkdir -p "${external_app}/scripts/build"
cp "${fixture_root}/scripts/build/prebuilt_validation.py" \
  "${fixture_root}/scripts/build/mfem_cache.sh" "${external_app}/scripts/build/"
mkdir -p "${external_app}/spack-envs/cpu/.spack-env/view"
cp "${REPO_ROOT}/spack-envs/cpu/spack.yaml" \
  "${external_app}/spack-envs/cpu/spack.yaml"
[[ ! -e "${external_app}/tpl/mfem/.git" ]] || \
  fail "external-source fixture initialized its local submodule"
set +e
env -u IGRLAB_TEST_MFEM_SOURCE_ROOT \
  -u DG_IGR_RUN_REUSE_FIXTURES \
  -u DG_IGR_TEST_REUSE_MFEM_ROOT -u DG_IGR_TEST_REUSE_SPACK_ENV \
  -u DG_IGR_TEST_REUSE_MODE -u DG_IGR_TEST_REUSE_PROFILE \
  -u DG_IGR_TEST_REUSE_CALIPER -u DG_IGR_TEST_REUSE_CUDA_ARCH \
  -u IGRLAB_SPACK_ENV -u DG_IGR_SPACK_ENV \
  PATH="${fake_bin}:/usr/bin:/bin" \
  FAKE_HELPER_LOG="${tmp_root}/external-source.log" \
  FAKE_SPACK_LOG="${tmp_root}/external-spack.log" \
  bash "${external_app}/build.sh" --cpu-only \
  --reuse-mfem-root "${external_mfem}" \
  --output-root "${tmp_root}/external-output" \
  --no-clean --no-deps --no-tests \
  >"${tmp_root}/external-source-output.log" 2>&1
external_rc=$?
set -e
[[ ${external_rc} -eq 77 ]] || \
  fail "external source forwarding exited ${external_rc}, expected 77"
external_log="$(<"${tmp_root}/external-source.log")"
assert_contains "${external_log}" "MFEM_SRC=${external_mfem}"
assert_contains "${external_log}" \
  "IGRLAB_TEST_MFEM_SOURCE_ROOT=${external_mfem}"
for name in DG_IGR_TEST_REUSE_MFEM_ROOT DG_IGR_TEST_REUSE_SPACK_ENV \
  DG_IGR_TEST_REUSE_MODE DG_IGR_TEST_REUSE_PROFILE \
  DG_IGR_TEST_REUSE_CALIPER DG_IGR_TEST_REUSE_CUDA_ARCH; do
  grep -Fxq "${name}=" "${tmp_root}/external-source.log" || \
    fail "external source forwarding enabled full reuse metadata: ${name}"
done
assert_not_exists "${tmp_root}/external-spack.log"

# Distinct profiles may race to create a fresh shared output root; each owns a
# separate scope and reaches the fake helper without a spurious mkdir failure.
concurrent_output_root="${tmp_root}/concurrent-output"
set +e
PATH="${fake_bin}:/usr/bin:/bin" \
  FAKE_HELPER_LOG="${tmp_root}/concurrent-gpu-helper.log" \
  FAKE_SPACK_LOG="${tmp_root}/concurrent-gpu-spack.log" \
  bash "${fixture_root}/build.sh" --gpu --cuda-arch 120 \
    --output-root "${concurrent_output_root}" --no-clean --no-deps --no-tests \
    >"${tmp_root}/concurrent-gpu.out" 2>&1 &
concurrent_gpu_pid=$!
PATH="${fake_bin}:/usr/bin:/bin" \
  FAKE_HELPER_LOG="${tmp_root}/concurrent-cpu-helper.log" \
  FAKE_SPACK_LOG="${tmp_root}/concurrent-cpu-spack.log" \
  bash "${fixture_root}/build.sh" --cpu-only \
    --output-root "${concurrent_output_root}" --no-clean --no-deps --no-tests \
    >"${tmp_root}/concurrent-cpu.out" 2>&1 &
concurrent_cpu_pid=$!
wait "${concurrent_gpu_pid}"
concurrent_gpu_rc=$?
wait "${concurrent_cpu_pid}"
concurrent_cpu_rc=$?
set -e
[[ ${concurrent_gpu_rc} -eq 77 && ${concurrent_cpu_rc} -eq 77 ]] || {
  fail "concurrent output initialization failed: gpu=${concurrent_gpu_rc} cpu=${concurrent_cpu_rc}"
}
assert_not_exists "${tmp_root}/concurrent-gpu-spack.log"
assert_not_exists "${tmp_root}/concurrent-cpu-spack.log"

# A held per-worktree/profile lock fails closed before either Spack or helper.
lock_output_root="${tmp_root}/lock-output"
lock_plan="$("${clean_env[@]}" bash "${fixture_root}/build.sh" \
  --gpu --cuda-arch 120 --output-root "${lock_output_root}" --print-plan)"
lock_worktree_id="$(plan_field "${lock_plan}" worktree_id)"
lock_path="${lock_output_root}/locks/${lock_worktree_id}-cuda-sm120.lock"
mkdir -p "$(dirname -- "${lock_path}")"
flock -n "${lock_path}" sleep 5 &
lock_holder=$!
set +e
reject_output="$(PATH="${fake_bin}:/usr/bin:/bin" \
  FAKE_HELPER_LOG="${tmp_root}/locked-helper.log" \
  FAKE_SPACK_LOG="${tmp_root}/locked-spack.log" \
  DG_IGR_SPACK_ENV="${forward_spack_env}" \
  bash "${fixture_root}/build.sh" --gpu --cuda-arch 120 \
    --output-root "${lock_output_root}" --no-deps --no-tests 2>&1)"
reject_rc=$?
set -e
wait "${lock_holder}"
[[ ${reject_rc} -eq 3 ]] || fail "locked output exited ${reject_rc}, expected 3"
assert_contains "${reject_output}" "output lock is already held"
assert_not_exists "${tmp_root}/locked-helper.log"
assert_not_exists "${tmp_root}/locked-spack.log"

# SM70/SM80/SM90 environments own release pins; both dependency lookup sites must
# accept that release while leaving the reviewed SM120 develop query unchanged.
for driver_and_variable in "${BUILD_DRIVER}:hypre_spec" "${MFEM_DRIVER}:HYPRE_SPEC"; do
  query_driver="${driver_and_variable%:*}"
  query_variable="${driver_and_variable##*:}"
  query_selection="$(awk -v variable="${query_variable}" '
    $0 ~ "^[ ]*" variable "=\"hypre@develop" { capture = 1 }
    capture { print }
    capture && /^[ ]*fi$/ { exit }
  ' "${query_driver}")"
  [[ -n "${query_selection}" ]] || fail "missing HYPRE query in ${query_driver}"
  for query_arch in 70 80 90 120; do
    selected_query="$(SPACK_CUDA_ARCH="${query_arch}" bash -c \
      "${query_selection}"$'\n''printf "%s" "${!1}"' -- "${query_variable}")"
    if [[ "${query_arch}" == 70 || "${query_arch}" == 80 || "${query_arch}" == 90 ]]; then
      [[ "${selected_query}" == "hypre+mpi+cuda cuda_arch=${query_arch}" ]] || \
        fail "SM${query_arch} HYPRE query does not admit its release pin: ${selected_query}"
    else
      [[ "${selected_query}" == "hypre@develop+mpi+cuda cuda_arch=120" ]] || \
        fail "SM120 HYPRE query changed: ${selected_query}"
    fi
  done
done

# CPU and GPU configure share the full FindHYPRE refresh. Check the production
# argument array without invoking CMake; changing HYPRE_DIR must also replace
# its singular discovery paths and rerun version/backend probes.
(
  base_arguments="$(awk '
    /^cmake_args=\(/ { capture = 1 }
    capture { print }
    capture && /^\)$/ { exit }
  ' "${MFEM_DRIVER}")"
  [[ -n "${base_arguments}" ]] || fail "missing shared MFEM configure arguments"
  CMAKE_BUILD_TYPE=Release
  MFEM_INSTALL="${tmp_root}/mfem-install"
  MPI_PREFIX="${tmp_root}/mpi"
  METIS_PREFIX="${tmp_root}/metis"
  for hypre_revision in old current; do
    HYPRE_PREFIX="${tmp_root}/${hypre_revision}-hypre"
    HYPRE_HEADER="${HYPRE_PREFIX}/include/HYPRE.h"
    HYPRE_LIBRARY="${HYPRE_PREFIX}/lib/libHYPRE.so"
    eval "${base_arguments}"
    for expected_arg in \
        -UHYPRE_FOUND -UHYPRE_INCLUDE_DIR -UHYPRE_INCLUDE_DIRS \
        -UHYPRE_LIBRARY -UHYPRE_LIBRARIES -UHYPRE_VERSION '-UHYPRE_VERSION_*' \
        -UHYPRE_USING_CUDA -UHYPRE_USING_HIP \
        -UTPL_HYPRE_INCLUDE_DIRS -UTPL_HYPRE_LIBRARIES; do
      [[ " ${cmake_args[*]} " == *" ${expected_arg} "* ]] || \
        fail "shared HYPRE refresh omits ${expected_arg}"
    done
    for actual_arg in "${cmake_args[@]}"; do
      case "${actual_arg}" in
        -UHYPRE_*|-UTPL_HYPRE_*) ;;
        -U*) fail "shared HYPRE refresh clears an unrelated cache entry: ${actual_arg}" ;;
      esac
    done
    assert_contains " ${cmake_args[*]} " " -DHYPRE_DIR=${HYPRE_PREFIX} "
    assert_contains " ${cmake_args[*]} " " -DHYPRE_INCLUDE_DIR:PATH=${HYPRE_PREFIX}/include "
    assert_contains " ${cmake_args[*]} " " -DHYPRE_LIBRARY:FILEPATH=${HYPRE_LIBRARY} "
  done
)

# Exercise the SM70/SM80/SM90 static-HYPRE linkage selector without invoking
# a compiler. SM120 must not query OpenBLAS or gain host LAPACK dependencies.
(
  eval "$(awk '
    /^configure_cuda_hypre_linalg\(\)/ { capture = 1 }
    capture { print }
    capture && /^}/ { exit }
  ' "${MFEM_DRIVER}")"
  openblas_fixture="${tmp_root}/openblas"
  mkdir -p "${openblas_fixture}/lib64"
  openblas_hash=0123456789abcdef0123456789abcdef
  SPACK_ENV="${tmp_root}/linalg-env"
  spack_location() { [[ "$1" == openblas ]] || return 1; printf '%s' "${openblas_fixture}"; }
  spack() {
    [[ "$1" == -e && "$2" == "${SPACK_ENV}" && "$3" == find && "$4" == --format ]] || return 1
    case "$5:$6" in
      '{hash}:openblas') printf '%s' "${openblas_hash}" ;;
      "{^blas.hash}|{^lapack.hash}:${HYPRE_SPEC}")
        printf '%s|%s' "${openblas_hash}" "${openblas_hash}" ;;
      *) return 1 ;;
    esac
  }
  for SPACK_CUDA_ARCH in 70 80 90; do
    HYPRE_SPEC="hypre+mpi+cuda cuda_arch=${SPACK_CUDA_ARCH}"
    touch "${openblas_fixture}/lib64/libopenblas.so"
    cmake_args=()
    configure_cuda_hypre_linalg
    for expected_arg in \
        '-DHYPRE_REQUIRED_PACKAGES:STRING=BLAS;LAPACK;UMPIRE' \
        -DUMPIRE_TARGET_NAMES:STRING=umpire -DUMPIRE_TARGET_FORCE:BOOL=OFF; do
      [[ " ${cmake_args[*]} " == *" ${expected_arg} "* ]] || \
        fail "SM${SPACK_CUDA_ARCH} static-HYPRE dependency arguments omit ${expected_arg}"
    done
    for actual_arg in "${cmake_args[@]}"; do
      case "${actual_arg}" in
        -U*) fail "SM${SPACK_CUDA_ARCH} linkage selector duplicates the shared HYPRE cache refresh: ${actual_arg}" ;;
      esac
    done
    assert_contains "${cmake_args[*]}" '-DBLA_VENDOR:STRING=OpenBLAS'
    assert_contains "${cmake_args[*]}" "-DBLAS_openblas_LIBRARY:FILEPATH=${openblas_fixture}/lib64/libopenblas.so"
    assert_contains "${cmake_args[*]}" "-DLAPACK_openblas_LIBRARY:FILEPATH=${openblas_fixture}/lib64/libopenblas.so"
    [[ "${cmake_args[*]}" != *MFEM_USE_LAPACK* ]] || fail "SM${SPACK_CUDA_ARCH} changed MFEM's own LAPACK policy"
    mv "${openblas_fixture}/lib64/libopenblas.so" "${tmp_root}/outside-openblas-${SPACK_CUDA_ARCH}.so"
    ln -s "${tmp_root}/outside-openblas-${SPACK_CUDA_ARCH}.so" "${openblas_fixture}/lib64/libopenblas.so"
    set +e
    output="$(configure_cuda_hypre_linalg 2>&1)"
    status=$?
    set -e
    [[ "${status}" == 3 ]] || fail "SM${SPACK_CUDA_ARCH} redirected OpenBLAS library was admitted"
    assert_contains "${output}" "SM${SPACK_CUDA_ARCH} requires a shared OpenBLAS library inside its concrete prefix"
    rm "${openblas_fixture}/lib64/libopenblas.so"
  done
  SPACK_CUDA_ARCH=120
  cmake_args=()
  spack_location() { fail "SM120 queried an OpenBLAS prefix"; }
  spack() { fail "SM120 queried OpenBLAS metadata"; }
  configure_cuda_hypre_linalg
  [[ ${#cmake_args[@]} == 0 ]] || fail "SM120 gained SM70/SM80/SM90-only linalg arguments"
)

# The reuse tuple is revalidated only after a real dependency install and
# before any view export or build consumer can observe the transition.
assert_reuse_transaction_order "${BUILD_DRIVER}"
assert_stamped_reuse_scope_policy "${BUILD_DRIVER}"
assert_reuse_identity_policy "${BUILD_DRIVER}" \
  "${REPO_ROOT}/cmake/mfem-reuse-identities/cpu-w8a-d9d6526-973e5abc.json" \
  "${REPO_ROOT}/cmake/mfem-reuse-identities/gpu-w8a-d9d6526-345d64a2.json"

reuse_fixture_cmake="${REPO_ROOT}/tests/hdg/runtime/app/test_cli_process.cmake"
grep -Fq 'set(reuse_mode_args --cpu-only --caliper)' \
  "${reuse_fixture_cmake}" || fail "Caliper reuse fixtures are not propagated"
grep -Fq '"$ENV{DG_IGR_TEST_REUSE_CUDA_ARCH}"' \
  "${reuse_fixture_cmake}" || fail "selected reuse CUDA architecture is not propagated"
if grep -Fq 'set(reuse_mode_args --gpu --cuda-arch 120)' \
    "${reuse_fixture_cmake}"; then
  fail "reuse fixture still hardcodes CUDA architecture 120"
fi

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

bash "${REPO_ROOT}/tests/scripts/check_spack_external_artifacts.sh" "${REPO_ROOT}"

echo "cuda-build-policy: PASS"
