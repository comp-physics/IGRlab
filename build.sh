#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="cpu"
JOBS="${NPROC:-4}"
MFEM_JOBS=""
CLEAN=1
INSTALL_DEPS=1
RUN_TESTS=1
RUN_SMOKE=0
CALIPER=0
HDG_EULER_EXPERIMENT="OFF"
HDG_EULER_DEVICE_EXPERIMENT="OFF"
HDG_RUNTIME="OFF"
HDG_AUXILIARY_EXPERIMENT="OFF"
SPACK_CUDA_ARCH="${SPACK_CUDA_ARCH:-120}"
CUDA_ARCH_WAS_SET=0
PRINT_PLAN=0
TEMP_ROOT=""
TEMP_ROOT_SOURCE=""
MFEM_REUSE_REQUEST=""
MFEM_REUSE_STATE="disabled"
MFEM_REUSE_ABI=""
MFEM_TUPLE_STATE="unresolved"
MFEM_SOURCE_ROOT=""
MFEM_BUILD=""
MFEM_INSTALL=""
MFEM_BUILD_NAME=""
MFEM_INSTALL_NAME=""
MFEM_SOURCE_PIN=""
MFEM_PICOJSON_BLOB=""
MFEM_TINYXML2_BLOB=""
SPACK_MANIFEST_SHA256=""
SPACK_LOCK_SHA256=""

usage() {
  cat <<'EOF'
Usage: ./build.sh [--cpu-only | --gpu] [options]

Modes:
  --cpu-only        Build dependencies, MFEM, and the app without CUDA (default).
  --gpu             Build the CUDA dependency profile, MFEM, and the app.

Options:
  --caliper         Build the CPU-only Caliper observability profile.
  --hdg-euler-experiment
                    Enable the CPU-only experimental HDG Euler path.
  --hdg-euler-device-experiment
                    Enable the experimental HDG Euler CPU-fallback/device path.
  --hdg-runtime     Enable the persistent HDG runtime component seam.
  --hdg-auxiliary-experiment
                    Enable the CPU-only experimental HDG auxiliary seam.
  -j, --jobs N      Parallel build jobs (default: NPROC or 4).
  --mfem-jobs N     Parallel MFEM build jobs (default: --jobs value).
  --cuda-arch N     Concrete GPU architecture profile (default: 120).
  --reuse-mfem-root DIR
                    Reuse a validated, read-only MFEM source/build/install
                    tuple rooted at DIR instead of invoking build_mfem_tpl.sh.
  --print-plan      Validate and print the resolved build policy, then exit.
  --clean           Rebuild the selected mode from clean caches (default).
  --no-clean        Reuse existing build and install directories.
  --no-deps         Do not run `spack install`.
  --no-tests        Do not run CTest.
  --smoke           Run a short two-rank Sedov smoke test after tests.
  -h, --help        Show this help.

Environment overrides:
  DG_IGR_SPACK_ENV, SPACK_CUDA_ARCH, MFEM_CUDA_ARCHS, APP_CUDA_ARCHS,
  CMAKE_BUILD_TYPE, DG_IGR_ACCELERATOR_KERNEL_SOURCES, TMPDIR

CUDA_ARCHS is intentionally rejected in GPU mode. Use MFEM_CUDA_ARCHS and
APP_CUDA_ARCHS so MFEM compatibility code generation cannot be confused with
native application code generation.
EOF
}

while (($#)); do
  case "$1" in
    --cpu|--cpu-only) MODE="cpu" ;;
    --gpu) MODE="gpu" ;;
    --caliper) CALIPER=1 ;;
    --hdg-euler-experiment) HDG_EULER_EXPERIMENT="ON" ;;
    --hdg-euler-device-experiment) HDG_EULER_DEVICE_EXPERIMENT="ON" ;;
    --hdg-runtime) HDG_RUNTIME="ON" ;;
    --hdg-auxiliary-experiment) HDG_AUXILIARY_EXPERIMENT="ON" ;;
    --cuda-arch)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
      SPACK_CUDA_ARCH="$2"
      CUDA_ARCH_WAS_SET=1
      shift
      ;;
    -j|--jobs)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
      JOBS="$2"
      shift
      ;;
    --mfem-jobs)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
      MFEM_JOBS="$2"
      shift
      ;;
    --reuse-mfem-root)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
      MFEM_REUSE_REQUEST="$2"
      shift
      ;;
    --print-plan) PRINT_PLAN=1 ;;
    --clean) CLEAN=1 ;;
    --no-clean) CLEAN=0 ;;
    --no-deps) INSTALL_DEPS=0 ;;
    --no-tests) RUN_TESTS=0 ;;
    --smoke) RUN_SMOKE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

cache_value() {
  local cache="$1"
  local key="$2"
  sed -n "s/^${key}:[^=]*=//p" "${cache}" | tail -n 1
}

config_value() {
  local config="$1"
  local key="$2"
  sed -n \
    -e "s|^set(${key} \"\(.*\)\")$|\1|p" \
    -e "s|^set(${key} \([^)]*\))$|\1|p" \
    "${config}" | tail -n 1
}

canonical_existing_path() {
  local path="$1"
  [[ -n "${path}" && -e "${path}" ]] || return 1
  realpath -e -- "${path}"
}

preflight_temp_root() {
  local requested probe_dir regular_probe fifo_probe probe_error cleanup_error

  command -v realpath >/dev/null 2>&1 || {
    echo "ERROR: temporary-directory preflight requires realpath." >&2
    exit 3
  }
  command -v mktemp >/dev/null 2>&1 || {
    echo "ERROR: temporary-directory preflight requires mktemp." >&2
    exit 3
  }
  command -v mkfifo >/dev/null 2>&1 || {
    echo "ERROR: temporary-directory preflight requires mkfifo." >&2
    exit 3
  }

  if [[ -n "${TMPDIR:-}" ]]; then
    requested="${TMPDIR}"
    TEMP_ROOT_SOURCE="environment:TMPDIR"
  else
    requested="/tmp"
    if [[ -n "${TMPDIR+x}" ]]; then
      TEMP_ROOT_SOURCE="default:/tmp:TMPDIR-empty"
    else
      TEMP_ROOT_SOURCE="default:/tmp:TMPDIR-unset"
    fi
  fi
  [[ -d "${requested}" ]] || {
    echo "ERROR: selected temporary root is not a directory: ${requested}" >&2
    exit 3
  }
  TEMP_ROOT="$(realpath -e -- "${requested}")" || {
    echo "ERROR: unable to canonicalize selected temporary root: ${requested}" >&2
    exit 3
  }
  [[ -d "${TEMP_ROOT}" ]] || {
    echo "ERROR: canonical temporary root is not a directory: ${TEMP_ROOT}" >&2
    exit 3
  }

  export TMPDIR="${TEMP_ROOT}"
  export TEMP="${TEMP_ROOT}"
  export TMP="${TEMP_ROOT}"

  probe_dir="$(mktemp -d -- \
    "${TEMP_ROOT}/dg-igr-build-temp-preflight.XXXXXXXX")" || {
    echo "ERROR: selected temporary root cannot create a unique directory: ${TEMP_ROOT}" >&2
    exit 3
  }
  regular_probe="${probe_dir}/regular-file"
  fifo_probe="${probe_dir}/fifo"
  probe_error=""
  cleanup_error=0

  if ! (umask 077; : >"${regular_probe}"); then
    probe_error="cannot create a regular file"
  elif [[ ! -f "${regular_probe}" ]]; then
    probe_error="regular-file probe did not create a regular file"
  elif ! mkfifo -m 600 -- "${fifo_probe}"; then
    probe_error="cannot create a FIFO"
  elif [[ ! -p "${fifo_probe}" ]]; then
    probe_error="FIFO probe did not create a FIFO"
  fi

  rm -f -- "${regular_probe}" "${fifo_probe}" || cleanup_error=1
  rmdir -- "${probe_dir}" || cleanup_error=1
  if [[ -n "${probe_error}" ]]; then
    echo "ERROR: selected temporary root ${probe_error}: ${TEMP_ROOT}" >&2
    exit 3
  fi
  if ((cleanup_error)); then
    echo "ERROR: unable to remove temporary preflight directory: ${probe_dir}" >&2
    exit 3
  fi
}

validate_mfem_source_checkout() {
  local requested="$1"
  local source_kind="$2"
  local root root_top root_head expected_pin tracked_status
  local header header_blob header_type actual_blob

  [[ -d "${requested}" ]] || {
    echo "ERROR: ${source_kind} MFEM source is not initialized: ${requested}" >&2
    exit 3
  }
  [[ -e "${requested}/.git" ]] || {
    echo "ERROR: ${source_kind} MFEM source has no checkout metadata and is not initialized: ${requested}" >&2
    exit 3
  }
  root="$(realpath -e -- "${requested}")" || {
    echo "ERROR: unable to canonicalize ${source_kind} MFEM source: ${requested}" >&2
    exit 3
  }
  expected_pin="$(git -C "${SCRIPT_DIR}" rev-parse HEAD:tpl/mfem 2>/dev/null)" || {
    echo "ERROR: current application HEAD has no readable tpl/mfem gitlink." >&2
    exit 3
  }
  root_top="$(git -C "${root}" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "ERROR: ${source_kind} MFEM source is not a Git checkout: ${root}" >&2
    exit 3
  }
  root_top="$(realpath -e -- "${root_top}")"
  [[ "${root_top}" == "${root}" ]] || {
    echo "ERROR: ${source_kind} MFEM source is not the checkout top level: root=${root} top=${root_top}" >&2
    exit 3
  }
  root_head="$(git -C "${root}" rev-parse HEAD 2>/dev/null)" || {
    echo "ERROR: ${source_kind} MFEM source has no readable HEAD: ${root}" >&2
    exit 3
  }
  [[ "${root_head}" == "${expected_pin}" ]] || {
    echo "ERROR: ${source_kind} MFEM pin mismatch: expected=${expected_pin} actual=${root_head}" >&2
    exit 3
  }

  for header in fem/picojson.h general/tinyxml2.h; do
    header_blob="$(git -C "${root}" rev-parse \
      "${expected_pin}:${header}" 2>/dev/null)" || {
      echo "ERROR: MFEM source header is not tracked at the pinned commit: ${header}" >&2
      exit 3
    }
    header_type="$(git -C "${root}" cat-file -t "${header_blob}" 2>/dev/null)" || {
      echo "ERROR: MFEM source header object is unreadable: path=${header} blob=${header_blob}" >&2
      exit 3
    }
    [[ "${header_type}" == "blob" ]] || {
      echo "ERROR: MFEM source header is not a blob: path=${header} type=${header_type}" >&2
      exit 3
    }
    [[ -f "${root}/${header}" ]] || {
      echo "ERROR: MFEM source header is missing: ${root}/${header}" >&2
      exit 3
    }
    actual_blob="$(git -C "${root}" hash-object --no-filters -- "${header}")" || {
      echo "ERROR: unable to hash MFEM source header: ${root}/${header}" >&2
      exit 3
    }
    [[ "${actual_blob}" == "${header_blob}" ]] || {
      echo "ERROR: MFEM source header identity mismatch: path=${root}/${header} expected_blob=${header_blob} actual_blob=${actual_blob}" >&2
      exit 3
    }
    case "${header}" in
      fem/picojson.h) MFEM_PICOJSON_BLOB="${header_blob}" ;;
      general/tinyxml2.h) MFEM_TINYXML2_BLOB="${header_blob}" ;;
    esac
  done

  tracked_status="$(GIT_OPTIONAL_LOCKS=0 git -C "${root}" status \
    --porcelain=v1 --untracked-files=no)" || {
    echo "ERROR: unable to inspect ${source_kind} MFEM tracked state: ${root}" >&2
    exit 3
  }
  [[ -z "${tracked_status}" ]] || {
    echo "ERROR: ${source_kind} MFEM tracked source/index is dirty: ${root}" >&2
    printf '%s\n' "${tracked_status}" >&2
    exit 3
  }

  MFEM_SOURCE_ROOT="${root}"
  MFEM_SOURCE_PIN="${expected_pin}"
}

validate_local_mfem_artifacts() {
  local cache config archive cache_home cache_prefix
  local canonical_home canonical_prefix

  validate_mfem_source_checkout "${MFEM_SOURCE_ROOT}" "local"
  cache="${MFEM_BUILD}/CMakeCache.txt"
  config="${MFEM_INSTALL}/lib/cmake/mfem/MFEMConfig.cmake"
  archive="${MFEM_INSTALL}/lib/libmfem.a"
  [[ -f "${cache}" ]] || {
    echo "ERROR: local MFEM CMake cache is missing after helper success: ${cache}" >&2
    exit 3
  }
  [[ -f "${config}" ]] || {
    echo "ERROR: local MFEM package config is missing after helper success: ${config}" >&2
    exit 3
  }
  [[ -f "${archive}" ]] || {
    echo "ERROR: local MFEM archive is missing after helper success: ${archive}" >&2
    exit 3
  }
  cache_home="$(cache_value "${cache}" CMAKE_HOME_DIRECTORY)"
  cache_prefix="$(cache_value "${cache}" CMAKE_INSTALL_PREFIX)"
  canonical_home="$(canonical_existing_path "${cache_home}")" || {
    echo "ERROR: local MFEM cache source identity is missing or moved: ${cache_home:-<missing>}" >&2
    exit 3
  }
  canonical_prefix="$(canonical_existing_path "${cache_prefix}")" || {
    echo "ERROR: local MFEM cache install identity is missing or moved: ${cache_prefix:-<missing>}" >&2
    exit 3
  }
  [[ "${canonical_home}" == "${MFEM_SOURCE_ROOT}" &&
     "${canonical_prefix}" == "${MFEM_INSTALL}" ]] || {
    echo "ERROR: local MFEM cache does not match the selected source/install tuple." >&2
    echo "       source expected=${MFEM_SOURCE_ROOT} actual=${canonical_home}" >&2
    echo "       install expected=${MFEM_INSTALL} actual=${canonical_prefix}" >&2
    exit 3
  }
  MFEM_TUPLE_STATE="local-source-build-install-validated"
  echo "[mfem] tuple validated: source=${MFEM_SOURCE_ROOT} build=${MFEM_BUILD} install=${MFEM_INSTALL}"
}

spack_identity() {
  local package="$1"
  local prefix_variable="$2"
  local hash_variable="$3"
  local compiler_variable="$4"
  local prefix records hash compiler line_count
  prefix="$(SPACK_COLOR=never spack -e "${SPACK_ENV}" location -i \
      "${package}" 2>/dev/null)" || {
    echo "ERROR: active Spack environment has no unique installed ${package}." >&2
    exit 3
  }
  records="$(SPACK_COLOR=never spack -e "${SPACK_ENV}" find \
      --format '{hash}|{compiler.name}@{compiler.version}' \
      "${package}" 2>/dev/null)" || {
    echo "ERROR: active Spack environment has no concrete ${package} identity." >&2
    exit 3
  }
  line_count="$(printf '%s\n' "${records}" | sed '/^$/d' | wc -l)"
  [[ "${line_count}" == "1" ]] || {
    echo "ERROR: active Spack environment has ambiguous ${package} identities." >&2
    exit 3
  }
  hash="${records%%|*}"
  compiler="${records#*|}"
  [[ "${hash}" =~ ^[a-z0-9]{32}$ && -n "${compiler}" ]] || {
    echo "ERROR: active Spack ${package} identity is malformed." >&2
    exit 3
  }
  prefix="$(canonical_existing_path "${prefix}")" || {
    echo "ERROR: active Spack ${package} prefix is missing or moved." >&2
    exit 3
  }
  printf -v "${prefix_variable}" '%s' "${prefix}"
  printf -v "${hash_variable}" '%s' "${hash}"
  printf -v "${compiler_variable}" '%s' "${compiler}"
}

spack_dependency_hash() {
  local package="$1"
  local dependency="$2"
  local records line_count
  records="$(SPACK_COLOR=never spack -e "${SPACK_ENV}" find \
      --format "{^${dependency}.hash}" "${package}" 2>/dev/null)" || {
    echo "ERROR: active Spack ${package} has no ${dependency} dependency identity." >&2
    exit 3
  }
  line_count="$(printf '%s\n' "${records}" | sed '/^$/d' | wc -l)"
  [[ "${line_count}" == "1" && "${records}" =~ ^[a-z0-9]{32}$ ]] || {
    echo "ERROR: active Spack ${package}->${dependency} identity is missing or ambiguous." >&2
    exit 3
  }
  printf '%s' "${records}"
}

assert_reuse_fixture_rejected() {
  local expected="$1"
  shift
  local output status
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  [[ ${status} -eq 3 && "${output}" == *"${expected}"* ]] || {
    echo "ERROR: reuse fixture expected '${expected}' with exit 3; exit=${status} output=${output}" >&2
    exit 3
  }
}

assert_pre_spack_fixture_rejected() {
  local expected="$1"
  local sentinel="$2"
  shift 2
  local output status
  [[ ! -e "${sentinel}" ]] || {
    echo "ERROR: pre-Spack fixture sentinel already exists: ${sentinel}" >&2
    exit 3
  }
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  [[ ${status} -eq 3 && "${output}" == *"${expected}"* &&
     ! -e "${sentinel}" ]] || {
    echo "ERROR: pre-Spack fixture expected '${expected}' with exit 3 and no Spack call; exit=${status} output=${output}" >&2
    exit 3
  }
}

assert_reuse_fixture_accepted() {
  local expected_source="$1"
  local expected_build="$2"
  local expected_install="$3"
  local expected_temp_source="$4"
  shift 4
  local output status
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  [[ ${status} -eq 0 &&
     "${output}" == *"mfem_source=${expected_source}"* &&
     "${output}" == *"mfem_build=${expected_build}"* &&
     "${output}" == *"mfem_install=${expected_install}"* &&
     "${output}" == *"temp_source=${expected_temp_source}"* ]] || {
    echo "ERROR: reuse fixture expected an exact-source plan; exit=${status} output=${output}" >&2
    exit 3
  }
}

make_reuse_fixture() {
  local label="$1"
  local source_root="$2"
  local fixture="${DG_IGR_REUSE_FIXTURE_ROOT}/${label}"
  local source_build source_install
  git -c advice.detachedHead=false clone --quiet --shared \
    "${source_root}" "${fixture}"
  git -c advice.detachedHead=false -C "${fixture}" checkout --quiet \
    --detach "$(git -C "${source_root}" rev-parse HEAD)"
  source_build="${source_root}/${MFEM_BUILD_NAME}"
  source_install="${source_root}/${MFEM_INSTALL_NAME}"
  mkdir -p "${fixture}/${MFEM_BUILD_NAME}" \
           "${fixture}/${MFEM_INSTALL_NAME}/lib/cmake/mfem" \
           "${fixture}/${MFEM_INSTALL_NAME}/lib"
  cp "${source_build}/CMakeCache.txt" \
     "${fixture}/${MFEM_BUILD_NAME}/CMakeCache.txt"
  cp "${source_install}/lib/cmake/mfem/MFEMConfig.cmake" \
     "${fixture}/${MFEM_INSTALL_NAME}/lib/cmake/mfem/MFEMConfig.cmake"
  ln -s "${source_install}/lib/libmfem.a" \
        "${fixture}/${MFEM_INSTALL_NAME}/lib/libmfem.a"
  sed -i -e "s|${source_root}|${fixture}|g" \
    "${fixture}/${MFEM_BUILD_NAME}/CMakeCache.txt"
  DG_IGR_REUSE_FIXTURE="${fixture}"
  DG_IGR_REUSE_FIXTURE_CACHE="${fixture}/${MFEM_BUILD_NAME}/CMakeCache.txt"
}

run_reuse_compatibility_fixtures() {
  local source_root="${MFEM_SOURCE_ROOT}"
  local fake_spack_dir bad_temp_root sentinel
  local -a mode_args=(--cpu-only)
  if [[ "${MODE}" == "gpu" ]]; then
    mode_args=(--gpu --cuda-arch "${SPACK_CUDA_ARCH}")
  elif ((CALIPER)); then
    mode_args=(--cpu-only --caliper)
  fi
  DG_IGR_REUSE_FIXTURE_ROOT="$(mktemp -d \
    "${TMPDIR:-/tmp}/dg-igr-reuse-fixtures.XXXXXX")"
  export DG_IGR_REUSE_FIXTURE_ROOT
  trap 'rm -rf -- "${DG_IGR_REUSE_FIXTURE_ROOT}"' EXIT

  fake_spack_dir="${DG_IGR_REUSE_FIXTURE_ROOT}/fake-spack-bin"
  mkdir -p "${fake_spack_dir}"
  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    ': >"${DG_IGR_FAKE_SPACK_SENTINEL:?}"' \
    'exit 99' >"${fake_spack_dir}/spack"
  chmod +x "${fake_spack_dir}/spack"

  bad_temp_root="${DG_IGR_REUSE_FIXTURE_ROOT}/not-a-directory"
  : >"${bad_temp_root}"
  sentinel="${DG_IGR_REUSE_FIXTURE_ROOT}/temp-spack-called"
  assert_pre_spack_fixture_rejected \
    "selected temporary root is not a directory" "${sentinel}" \
    env TMPDIR="${bad_temp_root}" PATH="${fake_spack_dir}:${PATH}" \
    DG_IGR_FAKE_SPACK_SENTINEL="${sentinel}" \
    DG_IGR_RUN_REUSE_FIXTURES=0 DG_IGR_SPACK_ENV="${SPACK_ENV}" \
    bash "${SCRIPT_DIR}/build.sh" "${mode_args[@]}" \
    --reuse-mfem-root "${source_root}" --no-deps --no-tests --print-plan

  sentinel="${DG_IGR_REUSE_FIXTURE_ROOT}/source-spack-called"
  assert_pre_spack_fixture_rejected \
    "MFEM source has no checkout metadata" "${sentinel}" \
    env PATH="${fake_spack_dir}:${PATH}" \
    DG_IGR_FAKE_SPACK_SENTINEL="${sentinel}" \
    DG_IGR_RUN_REUSE_FIXTURES=0 DG_IGR_SPACK_ENV="${SPACK_ENV}" \
    bash "${SCRIPT_DIR}/build.sh" "${mode_args[@]}" \
    --reuse-mfem-root "${source_root}/fem" \
    --no-deps --no-tests --print-plan

  make_reuse_fixture exact-source "${source_root}"
  assert_reuse_fixture_accepted \
    "${DG_IGR_REUSE_FIXTURE}" \
    "${DG_IGR_REUSE_FIXTURE}/${MFEM_BUILD_NAME}" \
    "${DG_IGR_REUSE_FIXTURE}/${MFEM_INSTALL_NAME}" \
    "default:/tmp:TMPDIR-empty" \
    env TMPDIR= DG_IGR_RUN_REUSE_FIXTURES=0 DG_IGR_SPACK_ENV="${SPACK_ENV}" \
    bash "${SCRIPT_DIR}/build.sh" "${mode_args[@]}" \
    --reuse-mfem-root "${DG_IGR_REUSE_FIXTURE}" \
    --no-deps --no-tests --print-plan

  make_reuse_fixture missing-header "${source_root}"
  rm -f -- "${DG_IGR_REUSE_FIXTURE}/fem/picojson.h"
  sentinel="${DG_IGR_REUSE_FIXTURE_ROOT}/missing-header-spack-called"
  assert_pre_spack_fixture_rejected \
    "MFEM source header is missing" "${sentinel}" \
    env PATH="${fake_spack_dir}:${PATH}" \
    DG_IGR_FAKE_SPACK_SENTINEL="${sentinel}" \
    DG_IGR_RUN_REUSE_FIXTURES=0 DG_IGR_SPACK_ENV="${SPACK_ENV}" \
    bash "${SCRIPT_DIR}/build.sh" "${mode_args[@]}" \
    --reuse-mfem-root "${DG_IGR_REUSE_FIXTURE}" \
    --no-deps --no-tests --print-plan

  make_reuse_fixture modified-header "${source_root}"
  printf '\n// reuse-fixture modification\n' >> \
    "${DG_IGR_REUSE_FIXTURE}/general/tinyxml2.h"
  sentinel="${DG_IGR_REUSE_FIXTURE_ROOT}/modified-header-spack-called"
  assert_pre_spack_fixture_rejected \
    "MFEM source header identity mismatch" "${sentinel}" \
    env PATH="${fake_spack_dir}:${PATH}" \
    DG_IGR_FAKE_SPACK_SENTINEL="${sentinel}" \
    DG_IGR_RUN_REUSE_FIXTURES=0 DG_IGR_SPACK_ENV="${SPACK_ENV}" \
    bash "${SCRIPT_DIR}/build.sh" "${mode_args[@]}" \
    --reuse-mfem-root "${DG_IGR_REUSE_FIXTURE}" \
    --no-deps --no-tests --print-plan

  make_reuse_fixture compiler "${source_root}"
  sed -i \
    's|^CMAKE_CXX_COMPILER:[^=]*=.*$|CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++|' \
    "${DG_IGR_REUSE_FIXTURE_CACHE}"
  assert_reuse_fixture_rejected "C++/MPI wrapper mismatch" \
    env DG_IGR_RUN_REUSE_FIXTURES=0 DG_IGR_SPACK_ENV="${SPACK_ENV}" \
    bash "${SCRIPT_DIR}/build.sh" "${mode_args[@]}" \
    --reuse-mfem-root "${DG_IGR_REUSE_FIXTURE}" \
    --no-deps --no-tests --print-plan

  make_reuse_fixture hypre "${source_root}"
  sed -i 's|^HYPRE_DIR:[^=]*=.*$|HYPRE_DIR:PATH=/usr|' \
    "${DG_IGR_REUSE_FIXTURE_CACHE}"
  assert_reuse_fixture_rejected "HYPRE root mismatch" \
    env DG_IGR_RUN_REUSE_FIXTURES=0 DG_IGR_SPACK_ENV="${SPACK_ENV}" \
    bash "${SCRIPT_DIR}/build.sh" "${mode_args[@]}" \
    --reuse-mfem-root "${DG_IGR_REUSE_FIXTURE}" \
    --no-deps --no-tests --print-plan

  make_reuse_fixture raja "${source_root}"
  sed -i 's|^MFEM_USE_RAJA:[^=]*=.*$|MFEM_USE_RAJA:BOOL=ON|' \
    "${DG_IGR_REUSE_FIXTURE_CACHE}"
  assert_reuse_fixture_rejected "RAJA disabled" \
    env DG_IGR_RUN_REUSE_FIXTURES=0 DG_IGR_SPACK_ENV="${SPACK_ENV}" \
    bash "${SCRIPT_DIR}/build.sh" "${mode_args[@]}" \
    --reuse-mfem-root "${DG_IGR_REUSE_FIXTURE}" \
    --no-deps --no-tests --print-plan

  if [[ "${MODE}" == "gpu" ]]; then
    make_reuse_fixture umpire "${source_root}"
    sed -i 's|^UMPIRE_DIR:[^=]*=.*$|UMPIRE_DIR:PATH=/usr|' \
      "${DG_IGR_REUSE_FIXTURE_CACHE}"
    assert_reuse_fixture_rejected "Umpire root mismatch" \
      env DG_IGR_RUN_REUSE_FIXTURES=0 DG_IGR_SPACK_ENV="${SPACK_ENV}" \
      bash "${SCRIPT_DIR}/build.sh" "${mode_args[@]}" \
      --reuse-mfem-root "${DG_IGR_REUSE_FIXTURE}" \
      --no-deps --no-tests --print-plan

    make_reuse_fixture cuda "${source_root}"
    sed -i 's|^CUDAToolkit_ROOT:[^=]*=.*$|CUDAToolkit_ROOT:PATH=/usr|' \
      "${DG_IGR_REUSE_FIXTURE_CACHE}"
    assert_reuse_fixture_rejected "CUDA toolkit/compiler mismatch" \
      env DG_IGR_RUN_REUSE_FIXTURES=0 DG_IGR_SPACK_ENV="${SPACK_ENV}" \
      bash "${SCRIPT_DIR}/build.sh" "${mode_args[@]}" \
      --reuse-mfem-root "${DG_IGR_REUSE_FIXTURE}" \
      --no-deps --no-tests --print-plan
  fi

  rm -rf -- "${DG_IGR_REUSE_FIXTURE_ROOT}"
  trap - EXIT
  unset DG_IGR_REUSE_FIXTURE_ROOT DG_IGR_REUSE_FIXTURE \
        DG_IGR_REUSE_FIXTURE_CACHE
  echo "[reuse-fixtures] exact-source, header, and compatibility cases passed"
}

is_cmake_true() {
  local value
  value="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
  [[ "${value}" == "1" || "${value}" == "ON" ||
     "${value}" == "TRUE" || "${value}" == "YES" ]]
}

is_cmake_false() {
  local value
  value="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
  [[ "${value}" == "0" || "${value}" == "OFF" ||
     "${value}" == "FALSE" || "${value}" == "NO" ]]
}

validate_reused_mfem() {
  local requested="$1"
  local root build install cache config archive
  local cache_home cache_prefix canonical_home canonical_prefix
  local mpi_enabled cuda_enabled raja_enabled umpire_enabled configured_archs
  local cache_cxx cache_mpi_cxx config_cxx expected_mpicxx expected_mpicc
  local c_command cxx_command compiler_name compiler_version
  local mpi_prefix mpi_hash mpi_compiler hypre_prefix hypre_hash hypre_compiler
  local hypre_mpi_hash raja_state umpire_state
  local cuda_prefix cuda_hash cuda_compiler umpire_prefix umpire_hash
  local umpire_compiler hypre_cuda_hash hypre_umpire_hash umpire_cuda_hash
  local cache_hypre cache_umpire cache_cuda cache_cuda_compiler
  local spack_view view_mpicc view_mpicxx view_hypre_header
  local view_hypre_library view_library_candidate view_nvcc view_umpire_config
  local cuda_metadata_records cuda_metadata_count cuda_metadata
  local metadata_cuda metadata_cuda_root cuda_host_compiler
  local cuda_host_compiler_id cuda_host_compiler_version

  validate_mfem_source_checkout "${requested}" "reused"
  root="${MFEM_SOURCE_ROOT}"

  command -v spack >/dev/null 2>&1 || {
    echo "ERROR: --reuse-mfem-root requires Spack metadata; source setup-env.sh first." >&2
    exit 3
  }
  [[ -d "${SPACK_ENV}" && -f "${SPACK_ENV}/spack.yaml" &&
     -f "${SPACK_ENV}/spack.lock" ]] || {
    echo "ERROR: selected Spack environment is missing its manifest or lock: ${SPACK_ENV}" >&2
    exit 3
  }
  command -v sha256sum >/dev/null 2>&1 || {
    echo "ERROR: --reuse-mfem-root requires sha256sum for environment identity." >&2
    exit 3
  }
  SPACK_MANIFEST_SHA256="$(sha256sum "${SPACK_ENV}/spack.yaml" | \
    cut -d' ' -f1)"
  SPACK_LOCK_SHA256="$(sha256sum "${SPACK_ENV}/spack.lock" | \
    cut -d' ' -f1)"
  [[ "${SPACK_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ &&
     "${SPACK_LOCK_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "ERROR: selected Spack manifest/lock hashes are malformed." >&2
    exit 3
  }
  [[ -d "${SPACK_ENV}/.spack-env/view" ]] || {
    echo "ERROR: selected Spack environment view is unavailable: ${SPACK_ENV}/.spack-env/view" >&2
    exit 3
  }
  spack_view="$(realpath -e -- "${SPACK_ENV}/.spack-env/view")" || {
    echo "ERROR: selected Spack environment view is missing or moved." >&2
    exit 3
  }

  build="${root}/${MFEM_BUILD_NAME}"
  install="${root}/${MFEM_INSTALL_NAME}"
  cache="${build}/CMakeCache.txt"
  config="${install}/lib/cmake/mfem/MFEMConfig.cmake"
  archive="${install}/lib/libmfem.a"
  [[ -f "${cache}" ]] || {
    echo "ERROR: reused MFEM CMake cache is missing: ${cache}" >&2
    exit 3
  }
  [[ -f "${config}" ]] || {
    echo "ERROR: reused MFEM package config is missing: ${config}" >&2
    exit 3
  }
  [[ -f "${archive}" ]] || {
    echo "ERROR: reused MFEM archive is missing: ${archive}" >&2
    exit 3
  }

  cache_home="$(cache_value "${cache}" CMAKE_HOME_DIRECTORY)"
  cache_prefix="$(cache_value "${cache}" CMAKE_INSTALL_PREFIX)"
  [[ -n "${cache_home}" && -d "${cache_home}" ]] || {
    echo "ERROR: reused MFEM cache has a missing or moved home directory: ${cache_home:-<missing>}" >&2
    exit 3
  }
  [[ -n "${cache_prefix}" && -d "${cache_prefix}" ]] || {
    echo "ERROR: reused MFEM cache has a missing or moved install prefix: ${cache_prefix:-<missing>}" >&2
    exit 3
  }
  canonical_home="$(realpath -e -- "${cache_home}")"
  canonical_prefix="$(realpath -e -- "${cache_prefix}")"
  [[ "${canonical_home}" == "${root}" ]] || {
    echo "ERROR: reused MFEM cache home mismatch: expected=${root} actual=${canonical_home}" >&2
    exit 3
  }
  [[ "${canonical_prefix}" == "${install}" ]] || {
    echo "ERROR: reused MFEM install-prefix mismatch: expected=${install} actual=${canonical_prefix}" >&2
    exit 3
  }

  mpi_enabled="$(cache_value "${cache}" MFEM_USE_MPI)"
  cuda_enabled="$(cache_value "${cache}" MFEM_USE_CUDA)"
  raja_enabled="$(cache_value "${cache}" MFEM_USE_RAJA)"
  umpire_enabled="$(cache_value "${cache}" MFEM_USE_UMPIRE)"
  is_cmake_true "${mpi_enabled}" || {
    echo "ERROR: reused MFEM cache does not enable MPI: ${mpi_enabled:-<missing>}" >&2
    exit 3
  }
  grep -Eqi '^set\(MFEM_USE_MPI (1|ON|TRUE|YES)\)$' "${config}" || {
    echo "ERROR: reused MFEM package config does not report MPI enabled: ${config}" >&2
    exit 3
  }
  is_cmake_false "${raja_enabled}" || {
    echo "ERROR: reused MFEM cache must report RAJA disabled for this profile: ${raja_enabled:-<missing>}" >&2
    exit 3
  }
  grep -Eqi '^set\(MFEM_USE_RAJA (0|OFF|FALSE|NO)\)$' "${config}" || {
    echo "ERROR: reused MFEM package config does not report RAJA disabled: ${config}" >&2
    exit 3
  }

  spack_identity openmpi mpi_prefix mpi_hash mpi_compiler
  spack_identity hypre hypre_prefix hypre_hash hypre_compiler
  hypre_mpi_hash="$(spack_dependency_hash hypre mpi)"
  [[ "${hypre_mpi_hash}" == "${mpi_hash}" ]] || {
    echo "ERROR: active Spack HYPRE/MPI concretization mismatch: hypre-mpi=${hypre_mpi_hash} mpi=${mpi_hash}" >&2
    exit 3
  }
  [[ "${hypre_compiler}" == "${mpi_compiler}" ]] || {
    echo "ERROR: active Spack HYPRE/MPI compiler mismatch: hypre=${hypre_compiler} mpi=${mpi_compiler}" >&2
    exit 3
  }
  cache_hypre="$(canonical_existing_path "$(cache_value "${cache}" HYPRE_DIR)")" || {
    echo "ERROR: reused MFEM HYPRE cache root is missing or moved." >&2
    exit 3
  }
  [[ "${cache_hypre}" == "${hypre_prefix}" ]] || {
    echo "ERROR: reused MFEM HYPRE root mismatch: expected=${hypre_prefix} actual=${cache_hypre}" >&2
    exit 3
  }

  expected_mpicc="$(canonical_existing_path "${mpi_prefix}/bin/mpicc")" || {
    echo "ERROR: active Spack MPI C wrapper is unavailable." >&2
    exit 3
  }
  expected_mpicxx="$(canonical_existing_path "${mpi_prefix}/bin/mpicxx")" || {
    echo "ERROR: active Spack MPI C++ wrapper is unavailable." >&2
    exit 3
  }
  view_mpicc="$(canonical_existing_path "${spack_view}/bin/mpicc")" || {
    echo "ERROR: selected Spack view has no usable MPI C wrapper." >&2
    exit 3
  }
  view_mpicxx="$(canonical_existing_path "${spack_view}/bin/mpicxx")" || {
    echo "ERROR: selected Spack view has no usable MPI C++ wrapper." >&2
    exit 3
  }
  [[ "${view_mpicc}" == "${expected_mpicc}" &&
     "${view_mpicxx}" == "${expected_mpicxx}" ]] || {
    echo "ERROR: selected Spack view MPI wrappers do not match its concrete OpenMPI prefix." >&2
    exit 3
  }
  view_hypre_header="$(canonical_existing_path "${spack_view}/include/HYPRE.h")" || {
    echo "ERROR: selected Spack view has no usable HYPRE header." >&2
    exit 3
  }
  view_hypre_library=""
  for view_library_candidate in \
      "${spack_view}/lib/libHYPRE.so" \
      "${spack_view}/lib64/libHYPRE.so"; do
    if [[ -e "${view_library_candidate}" ]]; then
      view_hypre_library="$(canonical_existing_path "${view_library_candidate}")"
      break
    fi
  done
  [[ -n "${view_hypre_library}" ]] || {
    echo "ERROR: selected Spack view has no usable HYPRE library." >&2
    exit 3
  }
  case "${view_hypre_header}" in
    "${hypre_prefix}"/*) ;;
    *)
      echo "ERROR: selected Spack view HYPRE header does not match its concrete HYPRE prefix." >&2
      exit 3
      ;;
  esac
  case "${view_hypre_library}" in
    "${hypre_prefix}"/*) ;;
    *)
      echo "ERROR: selected Spack view HYPRE library does not match its concrete HYPRE prefix." >&2
      exit 3
      ;;
  esac
  cache_cxx="$(canonical_existing_path "$(cache_value "${cache}" CMAKE_CXX_COMPILER)")" || {
    echo "ERROR: reused MFEM C++ compiler identity is missing or moved." >&2
    exit 3
  }
  cache_mpi_cxx="$(canonical_existing_path "$(cache_value "${cache}" MPI_CXX_COMPILER)")" || {
    echo "ERROR: reused MFEM MPI C++ compiler identity is missing or moved." >&2
    exit 3
  }
  config_cxx="$(canonical_existing_path "$(config_value "${config}" MFEM_CXX_COMPILER)")" || {
    echo "ERROR: reused MFEM package C++ compiler identity is missing or moved." >&2
    exit 3
  }
  [[ "${cache_cxx}" == "${expected_mpicxx}" &&
     "${cache_mpi_cxx}" == "${expected_mpicxx}" &&
     "${config_cxx}" == "${expected_mpicxx}" ]] || {
    echo "ERROR: reused MFEM C++/MPI wrapper mismatch with active Spack MPI." >&2
    exit 3
  }
  c_command="$("${mpi_prefix}/bin/mpicc" --showme:command 2>/dev/null)" || {
    echo "ERROR: active Spack MPI C compiler identity is unavailable." >&2
    exit 3
  }
  cxx_command="$("${mpi_prefix}/bin/mpicxx" --showme:command 2>/dev/null)" || {
    echo "ERROR: active Spack MPI C++ compiler identity is unavailable." >&2
    exit 3
  }
  [[ "${mpi_compiler}" =~ ^([^@]+)@(.+)$ ]] || {
    echo "ERROR: active Spack compiler identity is malformed: ${mpi_compiler}" >&2
    exit 3
  }
  compiler_name="${BASH_REMATCH[1]}"
  compiler_version="${BASH_REMATCH[2]}"
  [[ "${compiler_name}" == "gcc" && -x "${c_command}" &&
     -x "${cxx_command}" &&
     "$("${c_command}" -dumpfullversion)" == "${compiler_version}" &&
     "$("${cxx_command}" -dumpfullversion)" == "${compiler_version}" ]] || {
    echo "ERROR: active Spack MPI C/C++ compiler commands do not match ${mpi_compiler}." >&2
    exit 3
  }
  raja_state="off"
  if [[ "${MODE}" == "gpu" ]]; then
    is_cmake_true "${cuda_enabled}" || {
      echo "ERROR: reused GPU MFEM cache does not enable CUDA: ${cuda_enabled:-<missing>}" >&2
      exit 3
    }
    is_cmake_true "${umpire_enabled}" || {
      echo "ERROR: reused GPU MFEM cache does not enable Umpire: ${umpire_enabled:-<missing>}" >&2
      exit 3
    }
    configured_archs="$(cache_value "${cache}" CMAKE_CUDA_ARCHITECTURES)"
    [[ "${configured_archs}" == "${MFEM_CUDA_ARCHS}" ]] || {
      echo "ERROR: reused GPU MFEM architecture mismatch: expected=${MFEM_CUDA_ARCHS} actual=${configured_archs:-<missing>}" >&2
      exit 3
    }
    grep -Eqi '^set\(MFEM_USE_CUDA (1|ON|TRUE|YES)\)$' "${config}" || {
      echo "ERROR: reused GPU MFEM package config does not report CUDA enabled: ${config}" >&2
      exit 3
    }
    grep -Eqi '^set\(MFEM_USE_UMPIRE (1|ON|TRUE|YES)\)$' "${config}" || {
      echo "ERROR: reused GPU MFEM package config does not report Umpire enabled: ${config}" >&2
      exit 3
    }
    spack_identity cuda cuda_prefix cuda_hash cuda_compiler
    spack_identity umpire umpire_prefix umpire_hash umpire_compiler
    cache_cuda="$(canonical_existing_path "$(cache_value "${cache}" CUDAToolkit_ROOT)")" || {
      echo "ERROR: reused MFEM CUDA toolkit root is missing or moved." >&2
      exit 3
    }
    cache_cuda_compiler="$(canonical_existing_path "$(cache_value "${cache}" CMAKE_CUDA_COMPILER)")" || {
      echo "ERROR: reused MFEM CUDA compiler identity is missing or moved." >&2
      exit 3
    }
    cache_umpire="$(canonical_existing_path "$(cache_value "${cache}" UMPIRE_DIR)")" || {
      echo "ERROR: reused MFEM Umpire root is missing or moved." >&2
      exit 3
    }
    [[ "${cache_cuda}" == "${cuda_prefix}" &&
       "${cache_cuda_compiler}" == "$(canonical_existing_path "${cuda_prefix}/bin/nvcc")" ]] || {
      echo "ERROR: reused MFEM CUDA toolkit/compiler mismatch with active Spack CUDA." >&2
      exit 3
    }
    [[ "${cache_umpire}" == "${umpire_prefix}" ]] || {
      echo "ERROR: reused MFEM Umpire root mismatch: expected=${umpire_prefix} actual=${cache_umpire}" >&2
      exit 3
    }
    view_nvcc="$(canonical_existing_path "${spack_view}/bin/nvcc")" || {
      echo "ERROR: selected GPU Spack view has no usable CUDA compiler." >&2
      exit 3
    }
    view_umpire_config="$(canonical_existing_path \
      "${spack_view}/lib/cmake/umpire/umpire-config.cmake")" || {
      echo "ERROR: selected GPU Spack view has no usable Umpire package config." >&2
      exit 3
    }
    [[ "${view_nvcc}" == "$(canonical_existing_path "${cuda_prefix}/bin/nvcc")" ]] || {
      echo "ERROR: selected GPU Spack view CUDA compiler does not match its concrete CUDA prefix." >&2
      exit 3
    }
    case "${view_umpire_config}" in
      "${umpire_prefix}"/*) ;;
      *)
        echo "ERROR: selected GPU Spack view Umpire config does not match its concrete Umpire prefix." >&2
        exit 3
        ;;
    esac
    cuda_metadata_records="$(find "${build}/CMakeFiles" -mindepth 2 \
      -maxdepth 2 -type f -name CMakeCUDACompiler.cmake -print)"
    cuda_metadata_count="$(printf '%s\n' "${cuda_metadata_records}" | \
      sed '/^$/d' | wc -l)"
    [[ "${cuda_metadata_count}" == "1" ]] || {
      echo "ERROR: reused MFEM CUDA host-compiler metadata is missing or ambiguous." >&2
      exit 3
    }
    cuda_metadata="$(canonical_existing_path "${cuda_metadata_records}")" || {
      echo "ERROR: reused MFEM CUDA compiler metadata is missing or moved." >&2
      exit 3
    }
    metadata_cuda="$(canonical_existing_path \
      "$(config_value "${cuda_metadata}" CMAKE_CUDA_COMPILER)")" || {
      echo "ERROR: reused MFEM CUDA metadata has no usable CUDA compiler." >&2
      exit 3
    }
    metadata_cuda_root="$(canonical_existing_path \
      "$(config_value "${cuda_metadata}" CMAKE_CUDA_COMPILER_TOOLKIT_ROOT)")" || {
      echo "ERROR: reused MFEM CUDA metadata has no usable toolkit root." >&2
      exit 3
    }
    cuda_host_compiler="$(canonical_existing_path \
      "$(config_value "${cuda_metadata}" CMAKE_CUDA_HOST_COMPILER)")" || {
      echo "ERROR: reused MFEM CUDA metadata has no usable host compiler." >&2
      exit 3
    }
    cuda_host_compiler_id="$(config_value \
      "${cuda_metadata}" CMAKE_CUDA_HOST_COMPILER_ID)"
    cuda_host_compiler_version="$(config_value \
      "${cuda_metadata}" CMAKE_CUDA_HOST_COMPILER_VERSION)"
    [[ "${metadata_cuda}" == "${cache_cuda_compiler}" &&
       "${metadata_cuda_root}" == "${cuda_prefix}" &&
       "${cuda_host_compiler}" == "${expected_mpicxx}" &&
       "${cuda_host_compiler_id}" == "GNU" &&
       "${cuda_host_compiler_version}" == "${compiler_version}" ]] || {
      echo "ERROR: reused MFEM CUDA host/compiler metadata does not match the active Spack compiler/toolkit." >&2
      exit 3
    }
    [[ "${umpire_compiler}" == "${mpi_compiler}" ]] || {
      echo "ERROR: active Spack Umpire compiler mismatch: umpire=${umpire_compiler} mpi=${mpi_compiler}" >&2
      exit 3
    }
    hypre_cuda_hash="$(spack_dependency_hash hypre cuda)"
    hypre_umpire_hash="$(spack_dependency_hash hypre umpire)"
    umpire_cuda_hash="$(spack_dependency_hash umpire cuda)"
    [[ "${hypre_cuda_hash}" == "${cuda_hash}" &&
       "${hypre_umpire_hash}" == "${umpire_hash}" &&
       "${umpire_cuda_hash}" == "${cuda_hash}" ]] || {
      echo "ERROR: active Spack HYPRE/Umpire/CUDA dependency identities disagree." >&2
      exit 3
    }
    umpire_state="${umpire_hash}"
    MFEM_REUSE_ABI="compiler=${mpi_compiler};mpi=${mpi_hash};hypre=${hypre_hash};raja=${raja_state};umpire=${umpire_state};cuda=${cuda_hash};arch=${configured_archs}"
  else
    is_cmake_false "${cuda_enabled}" || {
      echo "ERROR: reused CPU MFEM cache does not report CUDA disabled: ${cuda_enabled:-<missing>}" >&2
      exit 3
    }
    is_cmake_false "${umpire_enabled}" || {
      echo "ERROR: reused CPU MFEM cache does not report Umpire disabled: ${umpire_enabled:-<missing>}" >&2
      exit 3
    }
    grep -Eqi '^set\(MFEM_USE_CUDA (0|OFF|FALSE|NO)\)$' "${config}" || {
      echo "ERROR: reused CPU MFEM package config does not report CUDA disabled: ${config}" >&2
      exit 3
    }
    grep -Eqi '^set\(MFEM_USE_UMPIRE (0|OFF|FALSE|NO)\)$' "${config}" || {
      echo "ERROR: reused CPU MFEM package config does not report Umpire disabled: ${config}" >&2
      exit 3
    }
    umpire_state="off"
    MFEM_REUSE_ABI="compiler=${mpi_compiler};mpi=${mpi_hash};hypre=${hypre_hash};raja=${raja_state};umpire=${umpire_state};cuda=off;arch=none"
  fi

  MFEM_SOURCE_ROOT="${root}"
  MFEM_BUILD="${build}"
  MFEM_INSTALL="${install}"
  MFEM_REUSE_STATE="validated-read-only"
  MFEM_TUPLE_STATE="reuse-source-build-install-validated"
}

if [[ -z "${MFEM_JOBS}" ]]; then
  MFEM_JOBS="${JOBS}"
fi

if [[ "${MODE}" == "gpu" && "${HDG_EULER_EXPERIMENT}" == "ON" ]]; then
  echo "ERROR: --hdg-euler-experiment is only valid with --cpu-only." >&2
  exit 2
fi
if [[ "${MODE}" != "cpu" && "${HDG_AUXILIARY_EXPERIMENT}" == "ON" ]]; then
  echo "ERROR: --hdg-auxiliary-experiment is only valid with --cpu-only." >&2
  exit 2
fi
[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: jobs must be a positive integer (got '${JOBS}')" >&2
  exit 2
}
[[ "${MFEM_JOBS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: MFEM jobs must be a positive integer (got '${MFEM_JOBS}')" >&2
  exit 2
}
[[ "${SPACK_CUDA_ARCH}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: --cuda-arch must be a concrete numeric architecture." >&2
  exit 2
}
if [[ "${MODE}" == "cpu" && "${CUDA_ARCH_WAS_SET}" == "1" ]]; then
  echo "ERROR: --cuda-arch is only valid with --gpu." >&2
  exit 2
fi
if [[ "${MODE}" == "gpu" && "${CALIPER}" == "1" ]]; then
  echo "ERROR: --caliper is only valid with --cpu-only." >&2
  exit 2
fi
if [[ "${MODE}" == "gpu" && -n "${CUDA_ARCHS+x}" ]]; then
  echo "ERROR: CUDA_ARCHS is ambiguous and is rejected in GPU mode." >&2
  echo "       Use MFEM_CUDA_ARCHS for MFEM and APP_CUDA_ARCHS for project sources." >&2
  exit 2
fi

preflight_temp_root

CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
if [[ "${MODE}" == "gpu" ]]; then
  PROFILE="cuda-sm${SPACK_CUDA_ARCH}"
  DEPENDENCY_CUDA_ARCH="${SPACK_CUDA_ARCH}"
  SPACK_ENV="${DG_IGR_SPACK_ENV:-${SCRIPT_DIR}/spack-envs/${PROFILE}}"
  MFEM_BUILD_NAME="build-gpu-sm${SPACK_CUDA_ARCH}"
  MFEM_INSTALL_NAME="install-gpu-sm${SPACK_CUDA_ARCH}"
  APP_BUILD="${SCRIPT_DIR}/build-gpu-sm${SPACK_CUDA_ARCH}"
  APP_INSTALL="${SCRIPT_DIR}/install-gpu-sm${SPACK_CUDA_ARCH}"
  EXPECTED_MFEM_CUDA_ARCHS="${SPACK_CUDA_ARCH}"
  if [[ "${SPACK_CUDA_ARCH}" == "120" ]]; then
    EXPECTED_MFEM_CUDA_ARCHS="80-virtual"
  fi
  MFEM_CUDA_ARCHS="${MFEM_CUDA_ARCHS:-${EXPECTED_MFEM_CUDA_ARCHS}}"
  APP_CUDA_ARCHS="${APP_CUDA_ARCHS:-${SPACK_CUDA_ARCH}}"

  if [[ "${MFEM_CUDA_ARCHS}" != "${EXPECTED_MFEM_CUDA_ARCHS}" ]]; then
    echo "ERROR: CUDA profile ${SPACK_CUDA_ARCH} requires MFEM_CUDA_ARCHS=${EXPECTED_MFEM_CUDA_ARCHS}." >&2
    if [[ "${SPACK_CUDA_ARCH}" == "120" ]]; then
      echo "       Native MFEM sm_120 code generation is blocked because it can exhaust memory." >&2
    fi
    exit 2
  fi
  if [[ "${APP_CUDA_ARCHS}" != "${SPACK_CUDA_ARCH}" ]]; then
    echo "ERROR: CUDA profile ${SPACK_CUDA_ARCH} requires APP_CUDA_ARCHS=${SPACK_CUDA_ARCH}." >&2
    echo "       Application sources use the native profile architecture." >&2
    exit 2
  fi
else
  DEPENDENCY_CUDA_ARCH="none"
  if ((CALIPER)); then
    PROFILE="cpu-caliper"
    SPACK_ENV="${DG_IGR_SPACK_ENV:-${SCRIPT_DIR}/spack-envs/cpu-caliper}"
    MFEM_BUILD_NAME="build-cpu-caliper"
    MFEM_INSTALL_NAME="install-cpu-caliper"
    APP_BUILD="${SCRIPT_DIR}/build-cpu-caliper"
    APP_INSTALL="${SCRIPT_DIR}/install-cpu-caliper"
  else
    PROFILE="cpu"
    SPACK_ENV="${DG_IGR_SPACK_ENV:-${SCRIPT_DIR}/spack-envs/cpu}"
    MFEM_BUILD_NAME="build-cpu"
    MFEM_INSTALL_NAME="install-cpu"
    APP_BUILD="${SCRIPT_DIR}/build-cpu"
    APP_INSTALL="${SCRIPT_DIR}/install-cpu"
  fi
  MFEM_CUDA_ARCHS="none"
  APP_CUDA_ARCHS="none"
fi

if [[ -n "${MFEM_REUSE_REQUEST}" ]]; then
  validate_reused_mfem "${MFEM_REUSE_REQUEST}"
else
  validate_mfem_source_checkout "${SCRIPT_DIR}/tpl/mfem" "local"
  MFEM_BUILD="${MFEM_SOURCE_ROOT}/${MFEM_BUILD_NAME}"
  MFEM_INSTALL="${MFEM_SOURCE_ROOT}/${MFEM_INSTALL_NAME}"
  MFEM_TUPLE_STATE="local-source-validated-artifacts-planned"
fi
if [[ "${DG_IGR_RUN_REUSE_FIXTURES:-0}" == "1" ]]; then
  [[ "${MFEM_REUSE_STATE}" == "validated-read-only" ]] || {
    echo "ERROR: reuse compatibility fixtures require a validated reuse root." >&2
    exit 3
  }
  run_reuse_compatibility_fixtures
fi

echo "[build-plan] mode=${MODE} profile=${PROFILE} dependency_cuda_arch=${DEPENDENCY_CUDA_ARCH} mfem_cuda_archs=${MFEM_CUDA_ARCHS} app_cuda_archs=${APP_CUDA_ARCHS} app_jobs=${JOBS} mfem_jobs=${MFEM_JOBS}"
echo "[build-plan] env=${SPACK_ENV} app_build=${APP_BUILD} app_install=${APP_INSTALL}"
echo "[build-plan] temp_root=${TEMP_ROOT} temp_source=${TEMP_ROOT_SOURCE} temp_env=TMPDIR,TEMP,TMP"
echo "[build-plan] hdg_runtime=${HDG_RUNTIME} hdg_auxiliary_experiment=${HDG_AUXILIARY_EXPERIMENT} hdg_euler_experiment=${HDG_EULER_EXPERIMENT} hdg_euler_device_experiment=${HDG_EULER_DEVICE_EXPERIMENT}"
echo "[build-plan] mfem_source=${MFEM_SOURCE_ROOT} mfem_build=${MFEM_BUILD} mfem_install=${MFEM_INSTALL} mfem_tuple_state=${MFEM_TUPLE_STATE} mfem_reuse_state=${MFEM_REUSE_STATE}"
echo "[build-plan] mfem_source_pin=${MFEM_SOURCE_PIN} fem_picojson_blob=${MFEM_PICOJSON_BLOB} general_tinyxml2_blob=${MFEM_TINYXML2_BLOB}"
if [[ "${MFEM_REUSE_STATE}" == "validated-read-only" ]]; then
  echo "[build-plan] mfem_reuse_abi=${MFEM_REUSE_ABI}"
  echo "[build-plan] spack_manifest_sha256=${SPACK_MANIFEST_SHA256} spack_lock_sha256=${SPACK_LOCK_SHA256}"
fi

if ((PRINT_PLAN)); then
  exit 0
fi

command -v spack >/dev/null 2>&1 || {
  echo "ERROR: spack is not available; source Spack's setup-env.sh first." >&2
  exit 2
}

if ((INSTALL_DEPS)); then
  echo "[build] installing ${MODE} dependencies from ${SPACK_ENV}"
  if ((CALIPER)); then
    spack -e "${SPACK_ENV}" concretize
  else
    spack -e "${SPACK_ENV}" concretize -f
  fi
  spack -e "${SPACK_ENV}" install
fi

SPACK_VIEW="${SPACK_ENV}/.spack-env/view"
[[ -d "${SPACK_VIEW}" ]] || {
  echo "ERROR: Spack environment view not found at ${SPACK_VIEW}" >&2
  echo "       Run without --no-deps once to install the selected profile." >&2
  exit 3
}
export PATH="${SPACK_VIEW}/bin:${PATH}"
export LD_LIBRARY_PATH="${SPACK_VIEW}/lib:${SPACK_VIEW}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mfem_args=("--${MODE}" --jobs "${MFEM_JOBS}")
if ((CALIPER)); then
  mfem_args+=(--caliper)
fi
if [[ "${MODE}" == "gpu" ]]; then
  mfem_args+=(
    --cuda-arch "${SPACK_CUDA_ARCH}"
    --cuda-codegen "${MFEM_CUDA_ARCHS}"
  )
fi
if ((CLEAN)); then
  mfem_args+=(--clean)
else
  mfem_args+=(--no-clean)
fi
if [[ "${MFEM_REUSE_STATE}" == "validated-read-only" ]]; then
  echo "[mfem] reuse validated: source=${MFEM_SOURCE_ROOT} build=${MFEM_BUILD} install=${MFEM_INSTALL}"
  export DG_IGR_TEST_REUSE_MFEM_ROOT="${MFEM_SOURCE_ROOT}"
  export DG_IGR_TEST_REUSE_SPACK_ENV="${SPACK_ENV}"
  export DG_IGR_TEST_REUSE_MODE="${MODE}"
else
  DG_IGR_SPACK_ENV="${SPACK_ENV}" CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    MFEM_SRC="${MFEM_SOURCE_ROOT}" MFEM_BUILD="${MFEM_BUILD}" \
    MFEM_INSTALL="${MFEM_INSTALL}" \
    "${SCRIPT_DIR}/build_mfem_tpl.sh" "${mfem_args[@]}"
  validate_local_mfem_artifacts
fi

CMAKE_BIN="$(spack -e "${SPACK_ENV}" location -i cmake)/bin/cmake"
MPI_PREFIX="$(spack -e "${SPACK_ENV}" location -i openmpi)"
if ((CLEAN)); then
  rm -rf "${APP_BUILD}" "${APP_INSTALL}"
fi

echo "[build] configuring app (${MODE}, hdg_runtime=${HDG_RUNTIME}, hdg_auxiliary_experiment=${HDG_AUXILIARY_EXPERIMENT}, hdg_euler_experiment=${HDG_EULER_EXPERIMENT}, hdg_euler_device_experiment=${HDG_EULER_DEVICE_EXPERIMENT})"
app_cmake_args=(
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  -DCMAKE_CXX_COMPILER="${MPI_PREFIX}/bin/mpicxx"
  -DCMAKE_INSTALL_PREFIX="${APP_INSTALL}"
  -DMFEM_DIR="${MFEM_INSTALL}/lib/cmake/mfem"
  -DDG_IGR_MFEM_SOURCE_ROOT:PATH="${MFEM_SOURCE_ROOT}"
  -DZLIB_ROOT="${SPACK_VIEW}"
  -DDG_IGR_ENABLE_HDG_EULER_EXPERIMENT="${HDG_EULER_EXPERIMENT}"
  -DDG_IGR_ENABLE_HDG_EULER_DEVICE_EXPERIMENT="${HDG_EULER_DEVICE_EXPERIMENT}"
  -DDG_IGR_ENABLE_HDG_RUNTIME="${HDG_RUNTIME}"
  -DDG_IGR_ENABLE_HDG_AUXILIARY_EXPERIMENT="${HDG_AUXILIARY_EXPERIMENT}"
)
if ((CALIPER)); then
  app_cmake_args+=(
    -DCMAKE_PREFIX_PATH="${SPACK_VIEW}"
    -DDG_IGR_ENABLE_OBSERVABILITY=ON
    -DDG_IGR_OBSERVABILITY_USE_NVTX=OFF
    -DDG_IGR_OBSERVABILITY_USE_CALIPER=ON
  )
fi
if [[ "${MODE}" == "gpu" ]]; then
  CUDA_PREFIX="$(spack -e "${SPACK_ENV}" location -i cuda)"
  UMPIRE_PREFIX="$(spack -e "${SPACK_ENV}" location -i umpire)"
  app_cmake_args+=(
    -DDG_IGR_REQUIRE_MFEM_CUDA=ON
    -DDG_IGR_EXPECT_CUDA_ARCH="${SPACK_CUDA_ARCH}"
    -DDG_IGR_ACCELERATOR_KERNEL_SOURCES="${DG_IGR_ACCELERATOR_KERNEL_SOURCES:-}"
    -DCMAKE_CUDA_COMPILER="${CUDA_PREFIX}/bin/nvcc"
    -DCMAKE_CUDA_ARCHITECTURES="${APP_CUDA_ARCHS}"
    -DCUDAToolkit_ROOT="${CUDA_PREFIX}"
    -Dumpire_DIR="${UMPIRE_PREFIX}/lib/cmake/umpire"
  )
fi
configure_args=()
if ((CLEAN)); then
  configure_args+=(--fresh)
fi
"${CMAKE_BIN}" "${configure_args[@]}" -S "${SCRIPT_DIR}" -B "${APP_BUILD}" \
  "${app_cmake_args[@]}"
if [[ "${MODE}" == "gpu" ]]; then
  app_cache="${APP_BUILD}/CMakeCache.txt"
  [[ -f "${app_cache}" ]] || {
    echo "ERROR: application CMake cache was not created at ${app_cache}" >&2
    exit 3
  }
  configured_app_cuda_archs="$(
    sed -n 's/^CMAKE_CUDA_ARCHITECTURES:[^=]*=//p' "${app_cache}" | tail -n 1
  )"
  if [[ "${configured_app_cuda_archs}" != "${APP_CUDA_ARCHS}" ]]; then
    echo "ERROR: application CUDA architecture cache mismatch." >&2
    echo "       requested=${APP_CUDA_ARCHS} configured=${configured_app_cuda_archs:-<missing>}" >&2
    exit 3
  fi
fi
"${CMAKE_BIN}" --build "${APP_BUILD}" -j"${JOBS}"
"${CMAKE_BIN}" --install "${APP_BUILD}"

if ((RUN_TESTS)); then
  CTEST_BIN="$(dirname -- "${CMAKE_BIN}")/ctest"
  "${CTEST_BIN}" --test-dir "${APP_BUILD}" --output-on-failure
fi

if ((RUN_SMOKE)); then
  smoke_exe="${APP_BUILD}/examples/sedov_taylor_blast/sedov_taylor_blast"
  smoke_args=(--t-final 0.001 -rs 0 -rp 1 -o 1 -c 0.25
              -vs 10 -ca 5.0 -is H1 -ig H1 -dt_save 0.1)
  if [[ "${MODE}" == "gpu" ]]; then
    if ! "${smoke_exe}" --help 2>&1 | grep -q -- '--device'; then
      echo "ERROR: this branch does not expose the CUDA runtime device option." >&2
      echo "       Rebase the GPU runtime support before using --gpu --smoke." >&2
      exit 4
    fi
    smoke_args=(--device cuda "${smoke_args[@]}")
  fi
  "${MPI_PREFIX}/bin/mpirun" -np 2 \
    "${smoke_exe}" "${smoke_args[@]}"
fi

echo "[build] complete: profile=${PROFILE} hdg_runtime=${HDG_RUNTIME} hdg_auxiliary_experiment=${HDG_AUXILIARY_EXPERIMENT} hdg_euler_experiment=${HDG_EULER_EXPERIMENT} hdg_euler_device_experiment=${HDG_EULER_DEVICE_EXPERIMENT} build=${APP_BUILD} install=${APP_INSTALL}"
