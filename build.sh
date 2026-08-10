#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# OpenMPI consults HOME even for serial CTest children.  The supervised
# controller intentionally supplies a closed environment, so recover only the
# effective user's passwd home when HOME is absent; an ambiguous or unusable
# passwd entry is a configuration error rather than a guessed scratch path.
if [[ -z "${HOME:-}" ]]; then
  command -v getent >/dev/null 2>&1 || {
    echo "ERROR: HOME is unset and getent is unavailable for passwd-home recovery." >&2
    exit 2
  }
  passwd_record="$(getent passwd "${EUID}")" || {
    echo "ERROR: HOME is unset and the effective user's passwd entry is unavailable." >&2
    exit 2
  }
  [[ -n "${passwd_record}" && "${passwd_record}" != *$'\n'* ]] || {
    echo "ERROR: HOME is unset and the effective user's passwd entry is ambiguous." >&2
    exit 2
  }
  IFS=: read -r _ _ _ _ _ passwd_home _ <<< "${passwd_record}"
  [[ -n "${passwd_home}" && "${passwd_home}" == /* && -d "${passwd_home}" ]] || {
    echo "ERROR: HOME is unset and the effective user's passwd home is unusable." >&2
    exit 2
  }
  export HOME="${passwd_home}"
fi

MODE="cpu"
JOBS="${NPROC:-4}"
MFEM_JOBS=""
CLEAN=1
INSTALL_DEPS=1
RUN_TESTS=1
TEST_REGEX=""
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
MFEM_REUSE_STAMP=""
MFEM_REUSE_STATE="disabled"
MFEM_FRESH_FROM_SEED=0
MFEM_FRESH_SEED_ROOT=""
MFEM_FRESH_SEED_STATUS=""
MFEM_FRESH_TARGET_ROOT=""
MFEM_CANONICAL_ORIGIN="https://github.com/mfem/mfem.git"
MFEM_REUSE_ABI=""
MFEM_REUSE_TRANSACTION_STATE="disabled"
MFEM_REUSE_PRE_SOURCE_ROOT=""
MFEM_REUSE_PRE_BUILD=""
MFEM_REUSE_PRE_INSTALL=""
MFEM_REUSE_PRE_MANIFEST_SHA256=""
MFEM_REUSE_PRE_LOCK_SHA256=""
MFEM_REUSE_PRE_ABI=""
MFEM_REUSE_PRE_IDENTITY_SHA256=""
MFEM_REUSE_PRE_TRUST_MODE=""
MFEM_REUSE_PRE_CACHE_SHA256=""
MFEM_REUSE_PRE_CONFIG_SHA256=""
MFEM_REUSE_PRE_ARCHIVE_SHA256=""
MFEM_REUSE_PRE_TARGETS_SHA256=""
MFEM_REUSE_PRE_TARGETS_CONFIG_SHA256=""
MFEM_REUSE_PRE_COMPILER_METADATA_SHA256=""
MFEM_REUSE_PRE_INSTALL_SCRIPT_SHA256=""
MFEM_REUSE_PRE_CHECK_CACHE_SHA256=""
MFEM_REUSE_POST_MANIFEST_SHA256=""
MFEM_REUSE_POST_LOCK_SHA256=""
MFEM_REUSE_POST_ABI=""
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
MFEM_IDENTITY_STAMP_NAME="dg-igr-mfem-identity-v1.txt"
MFEM_IDENTITY_STAMP=""
MFEM_IDENTITY_STAMP_SHA256=""
MFEM_IDENTITY_TRUST_MODE=""
MFEM_REVIEWED_ADMISSION_RECORD=""
MFEM_REVIEWED_ADMISSION_SHA256=""
MFEM_CPU_REVIEWED_ADMISSION_RECORD="${SCRIPT_DIR}/cmake/mfem-reuse-identities/cpu-w8a-d9d6526-973e5abc.json"
MFEM_CPU_REVIEWED_ADMISSION_SHA256="d4e0f68d795d30ecb3e403f797ecf4ccd4ec8ed79ba5746a997f68bd6eae98ec"
MFEM_GPU_SM120_REVIEWED_ADMISSION_RECORD="${SCRIPT_DIR}/cmake/mfem-reuse-identities/gpu-w8a-d9d6526-345d64a2.json"
MFEM_GPU_SM120_REVIEWED_ADMISSION_SHA256="00559ec1cdd0ad5728a0ff92a165fdbf7d757da6f9c672153d8a665cc9e489a9"
MFEM_IDENTITY_CACHE_SHA256=""
MFEM_IDENTITY_CONFIG_SHA256=""
MFEM_IDENTITY_ARCHIVE_SHA256=""
MFEM_IDENTITY_TARGETS_SHA256=""
MFEM_IDENTITY_TARGETS_CONFIG_PATH=""
MFEM_IDENTITY_TARGETS_CONFIG_SHA256=""
MFEM_IDENTITY_COMPILER_METADATA_PATH=""
MFEM_IDENTITY_COMPILER_METADATA_SHA256=""
MFEM_IDENTITY_INSTALL_SCRIPT_SHA256=""
MFEM_IDENTITY_CHECK_CACHE_SHA256=""
MFEM_IDENTITY_BUILD_TYPE=""
MFEM_IDENTITY_OPENMP_STATE=""
MFEM_IDENTITY_METIS_STATE=""
MFEM_IDENTITY_CALIPER_STATE=""
MFEM_IDENTITY_COMPILER=""
MFEM_IDENTITY_MPI_SPEC=""
MFEM_IDENTITY_MPI_HASH=""
MFEM_IDENTITY_MPI_PREFIX=""
MFEM_IDENTITY_HYPRE_SPEC=""
MFEM_IDENTITY_HYPRE_HASH=""
MFEM_IDENTITY_HYPRE_PREFIX=""
MFEM_IDENTITY_METIS_SPEC=""
MFEM_IDENTITY_METIS_HASH=""
MFEM_IDENTITY_METIS_PREFIX=""
MFEM_IDENTITY_RAJA_STATE=""
MFEM_IDENTITY_UMPIRE_SPEC=""
MFEM_IDENTITY_UMPIRE_HASH=""
MFEM_IDENTITY_UMPIRE_PREFIX=""
MFEM_IDENTITY_CALIPER_SPEC=""
MFEM_IDENTITY_CALIPER_HASH=""
MFEM_IDENTITY_CUDA_SPEC=""
MFEM_IDENTITY_CUDA_HASH=""
MFEM_IDENTITY_CUDA_PREFIX=""
MFEM_IDENTITY_CUDA_COMPILER=""
MFEM_IDENTITY_CUDA_COMPILER_METADATA_PATH=""
MFEM_IDENTITY_CUDA_COMPILER_METADATA_SHA256=""
MFEM_IDENTITY_INSTALL_MANIFEST_PATH=""
MFEM_IDENTITY_INSTALL_MANIFEST_SHA256=""

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
                    Reuse a strictly stamped MFEM tuple rooted at DIR.  An
                    unstamped pinned clean checkout is a read-only seed for a
                    target-local fresh MFEM build, never an in-place reuse.
  --print-plan      Validate and print the resolved build policy, then exit.
  --clean           Rebuild the selected mode from clean caches (default).
  --no-clean        Reuse existing build and install directories.
  --no-deps         Do not run `spack install`.
  --no-tests        Do not run CTest.
  --test-regex REGEX
                    Require REGEX to select one or more CTest tests, then run
                    only that selection.  Requires tests to remain enabled.
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
      [[ -n "$2" ]] || {
        echo "ERROR: --reuse-mfem-root requires a non-empty value" >&2
        exit 2
      }
      MFEM_REUSE_REQUEST="$2"
      shift
      ;;
    --print-plan) PRINT_PLAN=1 ;;
    --clean) CLEAN=1 ;;
    --no-clean) CLEAN=0 ;;
    --no-deps) INSTALL_DEPS=0 ;;
    --no-tests) RUN_TESTS=0 ;;
    --test-regex)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
      [[ -n "$2" && "$2" != *$'\n'* && "$2" != *$'\r'* ]] || {
        echo "ERROR: --test-regex requires one non-empty single-line CTest regex." >&2
        exit 2
      }
      TEST_REGEX="$2"
      shift
      ;;
    --smoke) RUN_SMOKE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ -n "${TEST_REGEX}" && "${RUN_TESTS}" != "1" ]]; then
  echo "ERROR: --test-regex requires CTest; do not combine it with --no-tests." >&2
  exit 2
fi

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

require_exact_mfem_directory() {
  local path="$1"
  local description="$2"
  local canonical

  [[ -d "${path}" && ! -L "${path}" ]] || {
    echo "ERROR: ${description} is missing or is not a non-symlink directory: ${path}" >&2
    exit 3
  }
  canonical="$(realpath -e -- "${path}")" || {
    echo "ERROR: unable to canonicalize ${description}: ${path}" >&2
    exit 3
  }
  [[ "${canonical}" == "${path}" ]] || {
    echo "ERROR: ${description} is not at its exact canonical path: expected=${path} actual=${canonical}" >&2
    exit 3
  }
}

require_exact_mfem_file() {
  local path="$1"
  local description="$2"
  local missing_context="${3:-}"
  local canonical

  [[ -f "${path}" ]] || {
    echo "ERROR: ${description} is missing${missing_context}: ${path}" >&2
    exit 3
  }
  [[ ! -L "${path}" ]] || {
    echo "ERROR: ${description} is not a regular non-symlink file: ${path}" >&2
    exit 3
  }
  canonical="$(realpath -e -- "${path}")" || {
    echo "ERROR: unable to canonicalize ${description}: ${path}" >&2
    exit 3
  }
  [[ "${canonical}" == "${path}" ]] || {
    echo "ERROR: ${description} is not at its exact canonical path: expected=${path} actual=${canonical}" >&2
    exit 3
  }
}

render_mfem_identity_payload() {
  printf '%s\n' \
    "schema=dg-igr-mfem-identity" \
    "version=1" \
    "mfem_pin=${MFEM_SOURCE_PIN}" \
    "source_root=${MFEM_SOURCE_ROOT}" \
    "build_root=${MFEM_BUILD}" \
    "install_root=${MFEM_INSTALL}" \
    "mode=${MODE}" \
    "profile=${PROFILE}" \
    "build_type=${MFEM_IDENTITY_BUILD_TYPE}" \
    "mfem_use_openmp=${MFEM_IDENTITY_OPENMP_STATE}" \
    "mfem_use_metis=${MFEM_IDENTITY_METIS_STATE}" \
    "mfem_use_caliper=${MFEM_IDENTITY_CALIPER_STATE}" \
    "compiler=${MFEM_IDENTITY_COMPILER}" \
    "mpi_spec=${MFEM_IDENTITY_MPI_SPEC}" \
    "mpi_hash=${MFEM_IDENTITY_MPI_HASH}" \
    "mpi_prefix=${MFEM_IDENTITY_MPI_PREFIX}" \
    "hypre_spec=${MFEM_IDENTITY_HYPRE_SPEC}" \
    "hypre_hash=${MFEM_IDENTITY_HYPRE_HASH}" \
    "hypre_prefix=${MFEM_IDENTITY_HYPRE_PREFIX}" \
    "metis_spec=${MFEM_IDENTITY_METIS_SPEC}" \
    "metis_hash=${MFEM_IDENTITY_METIS_HASH}" \
    "metis_prefix=${MFEM_IDENTITY_METIS_PREFIX}" \
    "raja_state=${MFEM_IDENTITY_RAJA_STATE}" \
    "umpire_spec=${MFEM_IDENTITY_UMPIRE_SPEC}" \
    "umpire_hash=${MFEM_IDENTITY_UMPIRE_HASH}" \
    "caliper_spec=${MFEM_IDENTITY_CALIPER_SPEC}" \
    "caliper_hash=${MFEM_IDENTITY_CALIPER_HASH}" \
    "cuda_spec=${MFEM_IDENTITY_CUDA_SPEC}" \
    "cuda_hash=${MFEM_IDENTITY_CUDA_HASH}" \
    "cuda_compiler=${MFEM_IDENTITY_CUDA_COMPILER}" \
    "dependency_cuda_arch=${DEPENDENCY_CUDA_ARCH}" \
    "mfem_cuda_archs=${MFEM_CUDA_ARCHS}" \
    "cmake_cache_sha256=${MFEM_IDENTITY_CACHE_SHA256}" \
    "mfem_config_sha256=${MFEM_IDENTITY_CONFIG_SHA256}" \
    "mfem_targets_sha256=${MFEM_IDENTITY_TARGETS_SHA256}" \
    "mfem_targets_config_path=${MFEM_IDENTITY_TARGETS_CONFIG_PATH}" \
    "mfem_targets_config_sha256=${MFEM_IDENTITY_TARGETS_CONFIG_SHA256}" \
    "mfem_archive_sha256=${MFEM_IDENTITY_ARCHIVE_SHA256}" \
    "compiler_metadata_path=${MFEM_IDENTITY_COMPILER_METADATA_PATH}" \
    "compiler_metadata_sha256=${MFEM_IDENTITY_COMPILER_METADATA_SHA256}" \
    "cmake_install_sha256=${MFEM_IDENTITY_INSTALL_SCRIPT_SHA256}" \
    "cmake_check_cache_sha256=${MFEM_IDENTITY_CHECK_CACHE_SHA256}" \
    "reuse_abi=${MFEM_REUSE_ABI}"
}

mfem_identity_stamp_matches() {
  local stamp="$1"
  local expected_payload="$2"
  local actual_payload

  [[ -f "${stamp}" && ! -L "${stamp}" ]] || return 1
  actual_payload="$(<"${stamp}")" || return 1
  [[ "${actual_payload}" == "${expected_payload}" ]]
}

mfem_admission_record_matches() {
  local record="$1"
  local expected_sha="$2"
  local actual_sha

  [[ -f "${record}" && ! -L "${record}" ]] || return 1
  actual_sha="$(sha256sum "${record}" | cut -d' ' -f1)" || return 1
  [[ "${actual_sha}" == "${expected_sha}" ]]
}

mfem_reviewed_admission_record_matches() {
  mfem_admission_record_matches "$1" \
    "${MFEM_REVIEWED_ADMISSION_SHA256}"
}

reviewed_admission_route() {
  local route_mode="$1"
  local route_profile="$2"

  case "${route_mode}:${route_profile}" in
    cpu:cpu)
      printf '%s|%s\n' \
        "${MFEM_CPU_REVIEWED_ADMISSION_RECORD}" \
        "${MFEM_CPU_REVIEWED_ADMISSION_SHA256}"
      ;;
    gpu:cuda-sm120)
      printf '%s|%s\n' \
        "${MFEM_GPU_SM120_REVIEWED_ADMISSION_RECORD}" \
        "${MFEM_GPU_SM120_REVIEWED_ADMISSION_SHA256}"
      ;;
    *) return 1 ;;
  esac
}

select_reviewed_admission_route() {
  local route

  MFEM_REVIEWED_ADMISSION_RECORD=""
  MFEM_REVIEWED_ADMISSION_SHA256=""
  route="$(reviewed_admission_route "${MODE}" "${PROFILE}")" || return 1
  IFS='|' read -r MFEM_REVIEWED_ADMISSION_RECORD \
    MFEM_REVIEWED_ADMISSION_SHA256 <<<"${route}"
  [[ -n "${MFEM_REVIEWED_ADMISSION_RECORD}" &&
     "${MFEM_REVIEWED_ADMISSION_SHA256}" =~ ^[0-9a-f]{64}$ ]]
}

hash_mfem_identity_artifacts() {
  local cache="$1"
  local config="$2"
  local archive="$3"
  local targets="$4"
  local targets_config="$5"
  local compiler_metadata="$6"
  local install_script="$7"
  local check_cache="$8"

  MFEM_IDENTITY_CACHE_SHA256="$(sha256sum "${cache}" | cut -d' ' -f1)"
  MFEM_IDENTITY_CONFIG_SHA256="$(sha256sum "${config}" | cut -d' ' -f1)"
  MFEM_IDENTITY_ARCHIVE_SHA256="$(sha256sum "${archive}" | cut -d' ' -f1)"
  MFEM_IDENTITY_TARGETS_SHA256="$(sha256sum "${targets}" | cut -d' ' -f1)"
  MFEM_IDENTITY_TARGETS_CONFIG_PATH="${targets_config}"
  MFEM_IDENTITY_TARGETS_CONFIG_SHA256="$(sha256sum \
    "${targets_config}" | cut -d' ' -f1)"
  MFEM_IDENTITY_COMPILER_METADATA_PATH="${compiler_metadata}"
  MFEM_IDENTITY_COMPILER_METADATA_SHA256="$(sha256sum \
    "${compiler_metadata}" | cut -d' ' -f1)"
  MFEM_IDENTITY_INSTALL_SCRIPT_SHA256="$(sha256sum \
    "${install_script}" | cut -d' ' -f1)"
  MFEM_IDENTITY_CHECK_CACHE_SHA256="$(sha256sum \
    "${check_cache}" | cut -d' ' -f1)"
  [[ "${MFEM_IDENTITY_CACHE_SHA256}" =~ ^[0-9a-f]{64}$ &&
     "${MFEM_IDENTITY_CONFIG_SHA256}" =~ ^[0-9a-f]{64}$ &&
     "${MFEM_IDENTITY_ARCHIVE_SHA256}" =~ ^[0-9a-f]{64}$ &&
     "${MFEM_IDENTITY_TARGETS_SHA256}" =~ ^[0-9a-f]{64}$ &&
     "${MFEM_IDENTITY_TARGETS_CONFIG_SHA256}" =~ ^[0-9a-f]{64}$ &&
     "${MFEM_IDENTITY_COMPILER_METADATA_SHA256}" =~ ^[0-9a-f]{64}$ &&
     "${MFEM_IDENTITY_INSTALL_SCRIPT_SHA256}" =~ ^[0-9a-f]{64}$ &&
     "${MFEM_IDENTITY_CHECK_CACHE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "ERROR: MFEM artifact identity hashes are malformed." >&2
    exit 3
  }
}

verify_mfem_identity_stamp() {
  local expected_payload actual_stamp_sha drift=0

  if [[ "${MFEM_REUSE_TRANSACTION_STATE}" == \
        "pre-dependency-validated" ]]; then
    actual_stamp_sha="$(sha256sum "${MFEM_IDENTITY_STAMP}" | cut -d' ' -f1)"
    if [[ "${actual_stamp_sha}" != "${MFEM_REUSE_PRE_IDENTITY_SHA256}" ]]; then
      echo "ERROR: generated MFEM identity stamp changed across dependency installation." >&2
      echo "       pre=${MFEM_REUSE_PRE_IDENTITY_SHA256}" >&2
      echo "       post=${actual_stamp_sha:-<missing>}" >&2
      drift=1
    fi
    local label pre_hash post_hash
    while IFS='|' read -r label pre_hash post_hash; do
      if [[ "${pre_hash}" != "${post_hash}" ]]; then
        echo "ERROR: MFEM ${label} changed across dependency installation." >&2
        echo "       pre=${pre_hash}" >&2
        echo "       post=${post_hash}" >&2
        drift=1
      fi
    done <<EOF
CMake cache|${MFEM_REUSE_PRE_CACHE_SHA256}|${MFEM_IDENTITY_CACHE_SHA256}
package config|${MFEM_REUSE_PRE_CONFIG_SHA256}|${MFEM_IDENTITY_CONFIG_SHA256}
archive|${MFEM_REUSE_PRE_ARCHIVE_SHA256}|${MFEM_IDENTITY_ARCHIVE_SHA256}
targets|${MFEM_REUSE_PRE_TARGETS_SHA256}|${MFEM_IDENTITY_TARGETS_SHA256}
configuration targets|${MFEM_REUSE_PRE_TARGETS_CONFIG_SHA256}|${MFEM_IDENTITY_TARGETS_CONFIG_SHA256}
compiler metadata|${MFEM_REUSE_PRE_COMPILER_METADATA_SHA256}|${MFEM_IDENTITY_COMPILER_METADATA_SHA256}
CMake install provenance|${MFEM_REUSE_PRE_INSTALL_SCRIPT_SHA256}|${MFEM_IDENTITY_INSTALL_SCRIPT_SHA256}
CMake cache provenance|${MFEM_REUSE_PRE_CHECK_CACHE_SHA256}|${MFEM_IDENTITY_CHECK_CACHE_SHA256}
EOF
    if ((drift)); then
      exit 3
    fi
    MFEM_IDENTITY_STAMP_SHA256="${actual_stamp_sha}"
    MFEM_IDENTITY_TRUST_MODE="generated-stamp-v1"
    return
  fi

  expected_payload="$(render_mfem_identity_payload)"
  mfem_identity_stamp_matches "${MFEM_IDENTITY_STAMP}" \
    "${expected_payload}" || {
    echo "ERROR: reused MFEM identity stamp does not match the exact source, artifacts, features, or Spack ABI: ${MFEM_IDENTITY_STAMP}" >&2
    exit 3
  }
  MFEM_IDENTITY_STAMP_SHA256="$(sha256sum "${MFEM_IDENTITY_STAMP}" | \
    cut -d' ' -f1)"
  [[ "${MFEM_IDENTITY_STAMP_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "ERROR: reused MFEM identity stamp hash is malformed." >&2
    exit 3
  }
  MFEM_IDENTITY_TRUST_MODE="generated-stamp-v1"
}

write_local_mfem_identity_stamp() {
  local stamp_tmp

  stamp_tmp="$(mktemp -- \
    "${MFEM_BUILD}/.${MFEM_IDENTITY_STAMP_NAME}.XXXXXXXX")" || {
    echo "ERROR: unable to create the local MFEM identity stamp transaction." >&2
    exit 3
  }
  if ! render_mfem_identity_payload >"${stamp_tmp}"; then
    rm -f -- "${stamp_tmp}"
    echo "ERROR: unable to render the local MFEM identity stamp." >&2
    exit 3
  fi
  if ! chmod 0444 "${stamp_tmp}" ||
     ! mv -f -- "${stamp_tmp}" "${MFEM_IDENTITY_STAMP}"; then
    rm -f -- "${stamp_tmp}"
    echo "ERROR: unable to atomically publish the local MFEM identity stamp: ${MFEM_IDENTITY_STAMP}" >&2
    exit 3
  fi
  require_exact_mfem_file \
    "${MFEM_IDENTITY_STAMP}" "local MFEM identity stamp after helper success"
  verify_mfem_identity_stamp
  echo "[mfem] identity stamp published: path=${MFEM_IDENTITY_STAMP} sha256=${MFEM_IDENTITY_STAMP_SHA256}"
}

reviewed_admission_value() {
  local key="$1"
  local values count

  values="$(sed -nE \
    "s/^[[:space:]]*\"${key}\":[[:space:]]*\"([^\"]*)\",?[[:space:]]*$/\\1/p" \
    "${MFEM_REVIEWED_ADMISSION_RECORD}")"
  count="$(printf '%s\n' "${values}" | sed '/^$/d' | wc -l)"
  [[ "${count}" == "1" && -n "${values}" ]] || {
    echo "ERROR: reviewed MFEM admission field is missing or ambiguous: ${key}" >&2
    exit 3
  }
  printf '%s' "${values}"
}

reviewed_admission_expect() {
  local key="$1"
  local actual="$2"
  local expected

  expected="$(reviewed_admission_value "${key}")"
  [[ "${actual}" == "${expected}" ]] || {
    echo "ERROR: reviewed MFEM admission mismatch for ${key}." >&2
    echo "       expected=${expected}" >&2
    echo "       actual=${actual:-<missing>}" >&2
    exit 3
  }
}

verify_reviewed_mfem_admission() {
  local record_sha source_tree build_rel install_rel
  local claim post_dependency_revalidation=0

  [[ -n "${MFEM_REVIEWED_ADMISSION_RECORD}" &&
     "${MFEM_REVIEWED_ADMISSION_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "ERROR: no prospective MFEM reuse admission is pinned for mode=${MODE} profile=${PROFILE}." >&2
    return 1
  }
  require_exact_mfem_file \
    "${MFEM_REVIEWED_ADMISSION_RECORD}" "reviewed MFEM admission record"
  record_sha="$(sha256sum "${MFEM_REVIEWED_ADMISSION_RECORD}" | cut -d' ' -f1)"
  mfem_reviewed_admission_record_matches \
    "${MFEM_REVIEWED_ADMISSION_RECORD}" || {
    echo "ERROR: reviewed MFEM admission record hash mismatch: expected=${MFEM_REVIEWED_ADMISSION_SHA256} actual=${record_sha:-<missing>}" >&2
    exit 3
  }
  [[ "${MFEM_SOURCE_ROOT}" == \
     "$(reviewed_admission_value source_root)" ]] || return 1

  source_tree="$(git -C "${MFEM_SOURCE_ROOT}" rev-parse HEAD^{tree} \
    2>/dev/null)" || {
    echo "ERROR: unable to resolve reviewed MFEM source tree." >&2
    exit 3
  }
  build_rel="${MFEM_BUILD#"${MFEM_SOURCE_ROOT}/"}"
  install_rel="${MFEM_INSTALL#"${MFEM_SOURCE_ROOT}/"}"
  claim="current tuple observed, independently reviewed, prospectively admitted for reuse; raw W8A ledgers are corroboration, not retroactive immutable proof"
  if [[ "${MFEM_REUSE_TRANSACTION_STATE}" == \
        "pre-dependency-validated" ]]; then
    post_dependency_revalidation=1
  fi

  reviewed_admission_expect schema \
    "dg-igr-reviewed-mfem-reuse-admission"
  reviewed_admission_expect version "1"
  reviewed_admission_expect trust_mode "prospective-reviewed-admission"
  reviewed_admission_expect claim "${claim}"
  reviewed_admission_expect source_root "${MFEM_SOURCE_ROOT}"
  reviewed_admission_expect build_root "${MFEM_BUILD}"
  reviewed_admission_expect install_root "${MFEM_INSTALL}"
  reviewed_admission_expect source_commit "${MFEM_SOURCE_PIN}"
  reviewed_admission_expect source_tree "${source_tree}"
  reviewed_admission_expect mode "${MODE}"
  reviewed_admission_expect profile "${PROFILE}"
  reviewed_admission_expect build_type "${MFEM_IDENTITY_BUILD_TYPE}"
  reviewed_admission_expect mfem_use_openmp \
    "${MFEM_IDENTITY_OPENMP_STATE}"
  reviewed_admission_expect mfem_use_metis "${MFEM_IDENTITY_METIS_STATE}"
  reviewed_admission_expect mfem_use_caliper \
    "${MFEM_IDENTITY_CALIPER_STATE}"
  reviewed_admission_expect mfem_use_raja "OFF"
  if [[ "${MFEM_IDENTITY_UMPIRE_HASH}" == "off" ]]; then
    reviewed_admission_expect mfem_use_umpire "OFF"
  else
    reviewed_admission_expect mfem_use_umpire "ON"
  fi
  if [[ "${MFEM_IDENTITY_CUDA_HASH}" == "off" ]]; then
    reviewed_admission_expect mfem_use_cuda "OFF"
  else
    reviewed_admission_expect mfem_use_cuda "ON"
  fi
  reviewed_admission_expect dependency_cuda_arch "${DEPENDENCY_CUDA_ARCH}"
  reviewed_admission_expect mfem_cuda_archs "${MFEM_CUDA_ARCHS}"
  if ((post_dependency_revalidation == 0)); then
    reviewed_admission_expect compiler "${MFEM_IDENTITY_COMPILER}"
    reviewed_admission_expect mpi_spec "${MFEM_IDENTITY_MPI_SPEC}"
    reviewed_admission_expect mpi_hash "${MFEM_IDENTITY_MPI_HASH}"
    reviewed_admission_expect mpi_prefix "${MFEM_IDENTITY_MPI_PREFIX}"
    reviewed_admission_expect hypre_spec "${MFEM_IDENTITY_HYPRE_SPEC}"
    reviewed_admission_expect hypre_hash "${MFEM_IDENTITY_HYPRE_HASH}"
    reviewed_admission_expect hypre_prefix "${MFEM_IDENTITY_HYPRE_PREFIX}"
    reviewed_admission_expect metis_spec "${MFEM_IDENTITY_METIS_SPEC}"
    reviewed_admission_expect metis_hash "${MFEM_IDENTITY_METIS_HASH}"
    reviewed_admission_expect metis_prefix "${MFEM_IDENTITY_METIS_PREFIX}"
    if [[ "${MODE}" == "gpu" ]]; then
      reviewed_admission_expect umpire_spec "${MFEM_IDENTITY_UMPIRE_SPEC}"
      reviewed_admission_expect umpire_hash "${MFEM_IDENTITY_UMPIRE_HASH}"
      reviewed_admission_expect umpire_prefix "${MFEM_IDENTITY_UMPIRE_PREFIX}"
      reviewed_admission_expect cuda_spec "${MFEM_IDENTITY_CUDA_SPEC}"
      reviewed_admission_expect cuda_hash "${MFEM_IDENTITY_CUDA_HASH}"
      reviewed_admission_expect cuda_prefix "${MFEM_IDENTITY_CUDA_PREFIX}"
    fi
    reviewed_admission_expect spack_manifest_sha256 \
      "${SPACK_MANIFEST_SHA256}"
    reviewed_admission_expect spack_lock_sha256 "${SPACK_LOCK_SHA256}"
  fi
  reviewed_admission_expect cmake_cache_path \
    "${build_rel}/CMakeCache.txt"
  reviewed_admission_expect cmake_cache_sha256 \
    "${MFEM_IDENTITY_CACHE_SHA256}"
  reviewed_admission_expect cmake_install_path \
    "${build_rel}/cmake_install.cmake"
  reviewed_admission_expect cmake_install_sha256 \
    "${MFEM_IDENTITY_INSTALL_SCRIPT_SHA256}"
  reviewed_admission_expect cmake_check_cache_path \
    "${build_rel}/CMakeFiles/cmake.check_cache"
  reviewed_admission_expect cmake_check_cache_sha256 \
    "${MFEM_IDENTITY_CHECK_CACHE_SHA256}"
  reviewed_admission_expect compiler_metadata_path \
    "${MFEM_IDENTITY_COMPILER_METADATA_PATH#"${MFEM_SOURCE_ROOT}/"}"
  reviewed_admission_expect compiler_metadata_sha256 \
    "${MFEM_IDENTITY_COMPILER_METADATA_SHA256}"
  if [[ "${MODE}" == "gpu" ]]; then
    reviewed_admission_expect cuda_compiler_metadata_path \
      "${MFEM_IDENTITY_CUDA_COMPILER_METADATA_PATH#"${MFEM_SOURCE_ROOT}/"}"
    reviewed_admission_expect cuda_compiler_metadata_sha256 \
      "${MFEM_IDENTITY_CUDA_COMPILER_METADATA_SHA256}"
    reviewed_admission_expect install_manifest_path \
      "${MFEM_IDENTITY_INSTALL_MANIFEST_PATH#"${MFEM_SOURCE_ROOT}/"}"
    reviewed_admission_expect install_manifest_sha256 \
      "${MFEM_IDENTITY_INSTALL_MANIFEST_SHA256}"
  fi
  reviewed_admission_expect mfem_config_path \
    "${install_rel}/lib/cmake/mfem/MFEMConfig.cmake"
  reviewed_admission_expect mfem_config_sha256 \
    "${MFEM_IDENTITY_CONFIG_SHA256}"
  reviewed_admission_expect mfem_targets_path \
    "${install_rel}/lib/cmake/mfem/MFEMTargets.cmake"
  reviewed_admission_expect mfem_targets_sha256 \
    "${MFEM_IDENTITY_TARGETS_SHA256}"
  reviewed_admission_expect mfem_targets_release_path \
    "${MFEM_IDENTITY_TARGETS_CONFIG_PATH#"${MFEM_SOURCE_ROOT}/"}"
  reviewed_admission_expect mfem_targets_release_sha256 \
    "${MFEM_IDENTITY_TARGETS_CONFIG_SHA256}"
  reviewed_admission_expect mfem_archive_path \
    "${install_rel}/lib/libmfem.a"
  reviewed_admission_expect mfem_archive_sha256 \
    "${MFEM_IDENTITY_ARCHIVE_SHA256}"
  reviewed_admission_expect journal_commit \
    "85f48b3bdc3b4fd29263c83bd2768be0764814dd"
  reviewed_admission_expect journal_tree \
    "caf56d5e06ccd0a23b9483d12d0fea3d1f06e3de"
  reviewed_admission_expect journal_receipt_path \
    "evidence/receipts/2026-07-30-w8a-functional-admission.md"
  reviewed_admission_expect journal_receipt_blob \
    "7f76c37336323ff448c8cc2a6340f3b8379ee872"
  reviewed_admission_expect journal_receipt_sha256 \
    "4c00ae91974bb5a111b24f313a13a0ddf19ca91d06cf443c08096340202b150e"
  if [[ "${MODE}" == "gpu" ]]; then
    reviewed_admission_expect sealed_dev3_gpu_manifest_sha256 \
      "ccb280318581f926b2bb65d98581868a29075ae4ba4f1ddf15eb24cf8939c059"
  else
    reviewed_admission_expect sealed_dev3_cpu_manifest_sha256 \
      "008605ffe7d0192689a6dcdcebfcee7fcac4ead307ff6c6403ee36731ce2d59d"
  fi
  reviewed_admission_expect shared_spack_authority_ledger_sha256 \
    "5ef52a45844fe2414ef28bd4067ad0de7c7bc860efc64d40dfc5a2b440736ea0"

  MFEM_IDENTITY_TRUST_MODE="reviewed-admission-v1"
  MFEM_IDENTITY_STAMP_SHA256="${record_sha}"
  echo "[mfem] reviewed reuse admission: claim=${claim} record=${MFEM_REVIEWED_ADMISSION_RECORD} sha256=${record_sha}"
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

capture_mfem_seed_status() {
  local root="$1"
  printf 'head=%s\n' "$(git -C "${root}" rev-parse HEAD)"
  printf 'tree=%s\n' "$(git -C "${root}" rev-parse HEAD^{tree})"
  GIT_OPTIONAL_LOCKS=0 git -C "${root}" status \
    --porcelain=v1 --untracked-files=normal
}

validate_mfem_origin() {
  local root="$1" source_kind="$2" origin
  origin="$(git -C "${root}" remote get-url origin 2>/dev/null)" || {
    echo "ERROR: ${source_kind} MFEM source has no origin remote: ${root}" >&2
    exit 3
  }
  [[ "${origin}" == "${MFEM_CANONICAL_ORIGIN}" ]] || {
    echo "ERROR: ${source_kind} MFEM origin mismatch: expected=${MFEM_CANONICAL_ORIGIN} actual=${origin}" >&2
    exit 3
  }
}

require_target_local_mfem_parent() {
  local create_if_absent="${1:-0}"
  local parent="${SCRIPT_DIR}/build-deps" canonical
  if [[ ! -e "${parent}" && ! -L "${parent}" ]]; then
    [[ "${create_if_absent}" == "1" ]] || {
      echo "ERROR: target-local MFEM parent is absent: ${parent}" >&2
      exit 3
    }
    mkdir -p -- "${parent}"
  fi
  [[ -d "${parent}" && ! -L "${parent}" ]] || {
    echo "ERROR: target-local MFEM parent is not a regular directory: ${parent}" >&2
    exit 3
  }
  canonical="$(realpath -e -- "${parent}")" || {
    echo "ERROR: unable to canonicalize target-local MFEM parent: ${parent}" >&2
    exit 3
  }
  [[ "${canonical}" == "${parent}" ]] || {
    echo "ERROR: target-local MFEM parent is not at its exact path: expected=${parent} actual=${canonical}" >&2
    exit 3
  }
}

plan_target_local_fresh_mfem() {
  local requested="$1" pinprefix target target_canonical
  validate_mfem_source_checkout "${requested}" "fresh MFEM seed"
  MFEM_FRESH_SEED_ROOT="${MFEM_SOURCE_ROOT}"
  validate_mfem_origin "${MFEM_FRESH_SEED_ROOT}" "fresh MFEM seed"
  MFEM_FRESH_SEED_STATUS="$(capture_mfem_seed_status "${MFEM_FRESH_SEED_ROOT}")" || {
    echo "ERROR: unable to record fresh MFEM seed status: ${MFEM_FRESH_SEED_ROOT}" >&2
    exit 3
  }
  pinprefix="${MFEM_SOURCE_PIN:0:12}"
  target="${SCRIPT_DIR}/build-deps/mfem-${PROFILE}-${pinprefix}"
  MFEM_FRESH_TARGET_ROOT="${target}"
  if [[ -e "${target}" || -L "${target}" ]]; then
    require_target_local_mfem_parent 0
    [[ -d "${target}" && ! -L "${target}" ]] || {
      echo "ERROR: existing target-local MFEM root is not a regular directory: ${target}" >&2
      exit 3
    }
    target_canonical="$(realpath -e -- "${target}")" || {
      echo "ERROR: unable to canonicalize existing target-local MFEM root: ${target}" >&2
      exit 3
    }
    [[ "${target_canonical}" == "${target}" ]] || {
      echo "ERROR: existing target-local MFEM root is not at its exact path: expected=${target} actual=${target_canonical}" >&2
      exit 3
    }
    validate_mfem_source_checkout "${target}" "existing target-local MFEM"
    validate_mfem_origin "${MFEM_SOURCE_ROOT}" "existing target-local MFEM"
    MFEM_BUILD="${target}/${MFEM_BUILD_NAME}"
    MFEM_INSTALL="${target}/${MFEM_INSTALL_NAME}"
    MFEM_REUSE_STATE="target-local-fresh-from-seed"
    MFEM_TUPLE_STATE="target-local-fresh-from-seed-retry"
    return
  fi
  MFEM_SOURCE_ROOT="${target}"
  MFEM_BUILD="${target}/${MFEM_BUILD_NAME}"
  MFEM_INSTALL="${target}/${MFEM_INSTALL_NAME}"
  MFEM_REUSE_STATE="target-local-fresh-from-seed"
  MFEM_TUPLE_STATE="target-local-fresh-from-seed-planned"
}

materialize_target_local_fresh_mfem() {
  local seed="${MFEM_FRESH_SEED_ROOT}" target="${MFEM_FRESH_TARGET_ROOT}"
  local before after staging attempt=0
  [[ "${MFEM_FRESH_FROM_SEED}" == "1" && -n "${seed}" && -n "${target}" ]] || return 0
  before="$(capture_mfem_seed_status "${seed}")" || {
    echo "ERROR: unable to re-read fresh MFEM seed status before clone: ${seed}" >&2
    exit 3
  }
  [[ "${before}" == "${MFEM_FRESH_SEED_STATUS}" ]] || {
    echo "ERROR: fresh MFEM seed status changed before materialization: ${seed}" >&2
    exit 3
  }
  if [[ -e "${target}" || -L "${target}" ]]; then
    local target_canonical
    require_target_local_mfem_parent 0
    [[ -d "${target}" && ! -L "${target}" ]] || {
      echo "ERROR: existing target-local MFEM root is not a regular directory: ${target}" >&2
      exit 3
    }
    target_canonical="$(realpath -e -- "${target}")" || {
      echo "ERROR: unable to canonicalize existing target-local MFEM root: ${target}" >&2
      exit 3
    }
    [[ "${target_canonical}" == "${target}" ]] || {
      echo "ERROR: existing target-local MFEM root is not at its exact path: expected=${target} actual=${target_canonical}" >&2
      exit 3
    }
    validate_mfem_source_checkout "${target}" "existing target-local MFEM"
    validate_mfem_origin "${MFEM_SOURCE_ROOT}" "existing target-local MFEM"
    after="$(capture_mfem_seed_status "${seed}")" || {
      echo "ERROR: unable to re-read fresh MFEM seed status after retry validation: ${seed}" >&2
      exit 3
    }
    [[ "${after}" == "${MFEM_FRESH_SEED_STATUS}" ]] || {
      echo "ERROR: fresh MFEM seed identity/status changed during retry validation: ${seed}" >&2
      exit 3
    }
    MFEM_SOURCE_ROOT="${target}"
    MFEM_BUILD="${target}/${MFEM_BUILD_NAME}"
    MFEM_INSTALL="${target}/${MFEM_INSTALL_NAME}"
    MFEM_TUPLE_STATE="target-local-fresh-from-seed-retry"
    return
  fi
  require_target_local_mfem_parent 1
  while ((attempt < 16)); do
    staging="${target}.staging.${BASHPID}.${RANDOM}"
    [[ ! -e "${staging}" && ! -L "${staging}" ]] && break
    ((++attempt))
  done
  ((attempt < 16)) || { echo "ERROR: unable to allocate unique MFEM clone sibling." >&2; exit 3; }
  git clone --no-local --no-hardlinks --no-checkout -- "${seed}" "${staging}"
  git -C "${staging}" remote set-url origin "${MFEM_CANONICAL_ORIGIN}"
  git -C "${staging}" checkout --detach "${MFEM_SOURCE_PIN}"
  validate_mfem_source_checkout "${staging}" "target-local fresh MFEM"
  validate_mfem_origin "${MFEM_SOURCE_ROOT}" "target-local fresh MFEM"
  after="$(capture_mfem_seed_status "${seed}")" || {
    echo "ERROR: unable to re-read fresh MFEM seed status after clone: ${seed}" >&2
    exit 3
  }
  [[ "${after}" == "${MFEM_FRESH_SEED_STATUS}" ]] || {
    echo "ERROR: fresh MFEM seed identity/status changed during materialization: ${seed}" >&2
    exit 3
  }
  [[ ! -e "${target}" && ! -L "${target}" ]] || {
    echo "ERROR: target-local fresh MFEM root appeared before atomic publish: ${target}" >&2
    exit 3
  }
  mv -T -- "${staging}" "${target}"
  MFEM_SOURCE_ROOT="${target}"
  MFEM_BUILD="${target}/${MFEM_BUILD_NAME}"
  MFEM_INSTALL="${target}/${MFEM_INSTALL_NAME}"
  MFEM_TUPLE_STATE="target-local-fresh-from-seed-materialized"
}

validate_local_mfem_artifacts() {
  local cache install_script check_cache config archive
  local cache_home cache_prefix
  local canonical_home canonical_prefix

  validate_mfem_source_checkout "${MFEM_SOURCE_ROOT}" "local"
  cache="${MFEM_BUILD}/CMakeCache.txt"
  install_script="${MFEM_BUILD}/cmake_install.cmake"
  check_cache="${MFEM_BUILD}/CMakeFiles/cmake.check_cache"
  config="${MFEM_INSTALL}/lib/cmake/mfem/MFEMConfig.cmake"
  archive="${MFEM_INSTALL}/lib/libmfem.a"
  require_exact_mfem_directory \
    "${MFEM_BUILD}" "local MFEM build directory after helper success"
  require_exact_mfem_file \
    "${cache}" "local MFEM CMake cache" " after helper success"
  require_exact_mfem_file \
    "${install_script}" "local MFEM CMake install provenance" \
    " after helper success"
  require_exact_mfem_file \
    "${check_cache}" "local MFEM CMake cache provenance" \
    " after helper success"
  require_exact_mfem_directory \
    "${MFEM_INSTALL}" "local MFEM install directory after helper success"
  require_exact_mfem_file \
    "${config}" "local MFEM package config" " after helper success"
  require_exact_mfem_file \
    "${archive}" "local MFEM archive" " after helper success"
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
  validate_reused_mfem "${MFEM_SOURCE_ROOT}" "create" "local"
  write_local_mfem_identity_stamp
  MFEM_REUSE_STATE="disabled"
  MFEM_REUSE_ABI=""
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
     "${output}" == *"temp_source=${expected_temp_source}"* &&
     "${output}" == *"mfem_identity_trust_mode="* &&
     "${output}" == *"mfem_reuse_transaction_state=pre-dependency-validated"* &&
     "${output}" != *"mfem_reuse_post_"* ]] || {
    echo "ERROR: reuse fixture expected an exact-source plan; exit=${status} output=${output}" >&2
    exit 3
  }
}

assert_identity_stamp_payload_rejected() {
  local label="$1"
  local stamp="$2"
  local expected_payload="$3"

  if mfem_identity_stamp_matches "${stamp}" "${expected_payload}"; then
    echo "ERROR: MFEM identity stamp fixture unexpectedly accepted ${label}." >&2
    exit 3
  fi
}

assert_reviewed_admission_route() {
  local route_mode="$1"
  local route_profile="$2"
  local expected_record="$3"
  local expected_sha="$4"
  local actual

  actual="$(reviewed_admission_route \
    "${route_mode}" "${route_profile}")" || {
    echo "ERROR: reviewed MFEM admission route fixture rejected mode=${route_mode} profile=${route_profile}." >&2
    exit 3
  }
  [[ "${actual}" == "${expected_record}|${expected_sha}" ]] || {
    echo "ERROR: reviewed MFEM admission route fixture mismatch for mode=${route_mode} profile=${route_profile}." >&2
    echo "       expected=${expected_record}|${expected_sha}" >&2
    echo "       actual=${actual}" >&2
    exit 3
  }
}

assert_reviewed_admission_route_rejected() {
  local route_mode="$1"
  local route_profile="$2"
  local actual

  if actual="$(reviewed_admission_route \
      "${route_mode}" "${route_profile}")"; then
    echo "ERROR: reviewed MFEM admission route fixture accepted wrong mode/profile=${route_mode}/${route_profile}: ${actual}" >&2
    exit 3
  fi
}

run_reviewed_admission_route_fixtures() {
  local zero_hash

  zero_hash="0000000000000000000000000000000000000000000000000000000000000000"
  assert_reviewed_admission_route cpu cpu \
    "${MFEM_CPU_REVIEWED_ADMISSION_RECORD}" \
    "${MFEM_CPU_REVIEWED_ADMISSION_SHA256}"
  assert_reviewed_admission_route gpu cuda-sm120 \
    "${MFEM_GPU_SM120_REVIEWED_ADMISSION_RECORD}" \
    "${MFEM_GPU_SM120_REVIEWED_ADMISSION_SHA256}"
  assert_reviewed_admission_route_rejected gpu cpu
  assert_reviewed_admission_route_rejected cpu cuda-sm120
  assert_reviewed_admission_route_rejected gpu cuda-sm90

  mfem_admission_record_matches \
    "${MFEM_CPU_REVIEWED_ADMISSION_RECORD}" \
    "${MFEM_CPU_REVIEWED_ADMISSION_SHA256}" || {
    echo "ERROR: pinned CPU reviewed MFEM admission fixture failed." >&2
    exit 3
  }
  mfem_admission_record_matches \
    "${MFEM_GPU_SM120_REVIEWED_ADMISSION_RECORD}" \
    "${MFEM_GPU_SM120_REVIEWED_ADMISSION_SHA256}" || {
    echo "ERROR: pinned GPU reviewed MFEM admission fixture failed." >&2
    exit 3
  }
  if mfem_admission_record_matches \
      "${MFEM_GPU_SM120_REVIEWED_ADMISSION_RECORD}" "${zero_hash}"; then
    echo "ERROR: wrong-hash reviewed MFEM admission fixture was accepted." >&2
    exit 3
  fi
  if mfem_admission_record_matches \
      "${DG_IGR_REUSE_FIXTURE_ROOT}/missing-reviewed-admission" \
      "${MFEM_GPU_SM120_REVIEWED_ADMISSION_SHA256}"; then
    echo "ERROR: missing reviewed MFEM admission fixture was accepted." >&2
    exit 3
  fi
  echo "[reuse-fixtures] CPU/GPU admission routing and negative cases passed"
}

run_identity_stamp_negative_fixtures() {
  local expected_payload mismatched_stamp hash_stamp hash_payload
  local mismatched_admission zero_hash

  expected_payload="$(render_mfem_identity_payload)"
  run_reviewed_admission_route_fixtures
  assert_identity_stamp_payload_rejected missing-stamp \
    "${DG_IGR_REUSE_FIXTURE_ROOT}/missing-identity-stamp" \
    "${expected_payload}"

  mismatched_stamp="${DG_IGR_REUSE_FIXTURE_ROOT}/mismatched-identity-stamp"
  printf '%s\n' \
    'schema=dg-igr-mfem-identity' \
    'version=0' >"${mismatched_stamp}"
  assert_identity_stamp_payload_rejected mismatched-stamp \
    "${mismatched_stamp}" "${expected_payload}"

  zero_hash="0000000000000000000000000000000000000000000000000000000000000000"
  hash_payload="${expected_payload/cmake_cache_sha256=${MFEM_IDENTITY_CACHE_SHA256}/cmake_cache_sha256=${zero_hash}}"
  [[ "${hash_payload}" != "${expected_payload}" ]] || {
    echo "ERROR: MFEM identity hash fixture could not alter the expected payload." >&2
    exit 3
  }
  hash_stamp="${DG_IGR_REUSE_FIXTURE_ROOT}/mismatched-hash-identity-stamp"
  printf '%s\n' "${hash_payload}" >"${hash_stamp}"
  assert_identity_stamp_payload_rejected mismatched-artifact-hash \
    "${hash_stamp}" "${expected_payload}"

  if mfem_reviewed_admission_record_matches \
      "${DG_IGR_REUSE_FIXTURE_ROOT}/missing-reviewed-admission"; then
    echo "ERROR: missing reviewed MFEM admission fixture was accepted." >&2
    exit 3
  fi
  mismatched_admission="${DG_IGR_REUSE_FIXTURE_ROOT}/mismatched-reviewed-admission.json"
  printf '%s\n' '{"schema":"dg-igr-reviewed-mfem-reuse-admission","version":"0"}' \
    >"${mismatched_admission}"
  if mfem_reviewed_admission_record_matches "${mismatched_admission}"; then
    echo "ERROR: mismatched reviewed MFEM admission fixture was accepted." >&2
    exit 3
  fi
}

make_mfem_source_fixture() {
  local label="$1"
  local source_root="$2"
  local fixture="${DG_IGR_REUSE_FIXTURE_ROOT}/${label}"
  git -c advice.detachedHead=false clone --quiet --shared \
    "${source_root}" "${fixture}"
  git -c advice.detachedHead=false -C "${fixture}" checkout --quiet \
    --detach "$(git -C "${source_root}" rev-parse HEAD)"
  DG_IGR_REUSE_FIXTURE="${fixture}"
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

  run_identity_stamp_negative_fixtures

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

  assert_reuse_fixture_accepted \
    "${source_root}" \
    "${source_root}/${MFEM_BUILD_NAME}" \
    "${source_root}/${MFEM_INSTALL_NAME}" \
    "default:/tmp:TMPDIR-empty" \
    env TMPDIR= DG_IGR_RUN_REUSE_FIXTURES=0 DG_IGR_SPACK_ENV="${SPACK_ENV}" \
    bash "${SCRIPT_DIR}/build.sh" "${mode_args[@]}" \
    --reuse-mfem-root "${source_root}" \
    --no-deps --no-tests --print-plan

  make_mfem_source_fixture missing-header "${source_root}"
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

  make_mfem_source_fixture modified-header "${source_root}"
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

  make_mfem_source_fixture symlinked-cache "${source_root}"
  mkdir -p "${DG_IGR_REUSE_FIXTURE}/${MFEM_BUILD_NAME}"
  ln -s "${source_root}/${MFEM_BUILD_NAME}/CMakeCache.txt" \
    "${DG_IGR_REUSE_FIXTURE}/${MFEM_BUILD_NAME}/CMakeCache.txt"
  sentinel="${DG_IGR_REUSE_FIXTURE_ROOT}/symlinked-cache-spack-called"
  assert_pre_spack_fixture_rejected \
    "reused MFEM CMake cache is not a regular non-symlink file" \
    "${sentinel}" \
    env PATH="${fake_spack_dir}:${PATH}" \
    DG_IGR_FAKE_SPACK_SENTINEL="${sentinel}" \
    DG_IGR_RUN_REUSE_FIXTURES=0 DG_IGR_SPACK_ENV="${SPACK_ENV}" \
    bash "${SCRIPT_DIR}/build.sh" "${mode_args[@]}" \
    --reuse-mfem-root "${DG_IGR_REUSE_FIXTURE}" \
    --no-deps --no-tests --print-plan

  rm -rf -- "${DG_IGR_REUSE_FIXTURE_ROOT}"
  trap - EXIT
  unset DG_IGR_REUSE_FIXTURE_ROOT DG_IGR_REUSE_FIXTURE
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
  local stamp_policy="${2:-require}"
  local source_kind="${3:-reused}"
  local root build install cache install_script check_cache install_manifest
  local config archive
  local targets targets_config targets_config_suffix compiler_metadata_records
  local compiler_metadata_count compiler_metadata
  local cache_home cache_prefix canonical_home canonical_prefix
  local cache_build_type mpi_enabled openmp_enabled metis_enabled
  local openmp_config_enabled metis_config_enabled
  local cuda_enabled raja_enabled umpire_enabled configured_archs
  local caliper_enabled caliper_config_enabled cache_cmake_prefix_path
  local cache_cxx cache_mpi_cxx config_cxx expected_mpicxx expected_mpicc
  local c_command cxx_command compiler_name compiler_version
  local mpi_spec hypre_spec metis_spec caliper_spec umpire_spec cuda_spec
  local mpi_prefix mpi_hash mpi_compiler hypre_prefix hypre_hash hypre_compiler
  local hypre_mpi_hash raja_state umpire_state caliper_state
  local metis_prefix metis_hash metis_compiler cache_metis
  local caliper_prefix caliper_hash caliper_compiler caliper_mpi_hash
  local cuda_prefix cuda_hash cuda_compiler umpire_prefix umpire_hash
  local umpire_compiler hypre_cuda_hash hypre_umpire_hash umpire_cuda_hash
  local cache_hypre cache_umpire cache_cuda cache_cuda_compiler
  local spack_view view_mpicc view_mpicxx view_hypre_header
  local view_hypre_library view_library_candidate view_nvcc view_umpire_config
  local view_caliper_config view_metis_header view_metis_library
  local cuda_metadata_records cuda_metadata_count cuda_metadata
  local metadata_cuda metadata_cuda_root cuda_host_compiler
  local cuda_host_compiler_id cuda_host_compiler_version

  [[ "${stamp_policy}" == "require" || "${stamp_policy}" == "create" ]] || {
    echo "ERROR: internal MFEM identity stamp policy is invalid: ${stamp_policy}" >&2
    exit 3
  }
  validate_mfem_source_checkout "${requested}" "${source_kind}"
  root="${MFEM_SOURCE_ROOT}"
  build="${root}/${MFEM_BUILD_NAME}"
  install="${root}/${MFEM_INSTALL_NAME}"
  cache="${build}/CMakeCache.txt"
  install_script="${build}/cmake_install.cmake"
  check_cache="${build}/CMakeFiles/cmake.check_cache"
  MFEM_IDENTITY_STAMP="${build}/${MFEM_IDENTITY_STAMP_NAME}"
  config="${install}/lib/cmake/mfem/MFEMConfig.cmake"
  targets="${install}/lib/cmake/mfem/MFEMTargets.cmake"
  targets_config_suffix="$(printf '%s' "${CMAKE_BUILD_TYPE}" | \
    tr '[:upper:]' '[:lower:]')"
  [[ "${targets_config_suffix}" =~ ^[a-z0-9_+-]+$ ]] || {
    echo "ERROR: requested MFEM build type cannot name installed target provenance: ${CMAKE_BUILD_TYPE}" >&2
    exit 3
  }
  targets_config="${install}/lib/cmake/mfem/MFEMTargets-${targets_config_suffix}.cmake"
  archive="${install}/lib/libmfem.a"
  require_exact_mfem_directory \
    "${build}" "reused MFEM build directory"
  require_exact_mfem_file "${cache}" "reused MFEM CMake cache"
  require_exact_mfem_file \
    "${install_script}" "reused MFEM CMake install provenance"
  require_exact_mfem_file \
    "${check_cache}" "reused MFEM CMake cache provenance"
  compiler_metadata_records="$(find "${build}/CMakeFiles" -mindepth 2 \
    -maxdepth 2 -type f -name CMakeCXXCompiler.cmake -print)"
  compiler_metadata_count="$(printf '%s\n' "${compiler_metadata_records}" | \
    sed '/^$/d' | wc -l)"
  [[ "${compiler_metadata_count}" == "1" ]] || {
    echo "ERROR: reused MFEM C++ compiler metadata is missing or ambiguous." >&2
    exit 3
  }
  compiler_metadata="${compiler_metadata_records}"
  require_exact_mfem_file \
    "${compiler_metadata}" "reused MFEM C++ compiler metadata"
  if [[ "${stamp_policy}" == "require" &&
        ( -e "${MFEM_IDENTITY_STAMP}" || -L "${MFEM_IDENTITY_STAMP}" ) ]]; then
    require_exact_mfem_file \
      "${MFEM_IDENTITY_STAMP}" "reused MFEM identity stamp"
  fi
  require_exact_mfem_directory \
    "${install}" "reused MFEM install directory"
  require_exact_mfem_file "${config}" "reused MFEM package config"
  require_exact_mfem_file "${targets}" "reused MFEM targets provenance"
  require_exact_mfem_file \
    "${targets_config}" "reused MFEM configuration-targets provenance"
  require_exact_mfem_file "${archive}" "reused MFEM archive"

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

  cache_build_type="$(cache_value "${cache}" CMAKE_BUILD_TYPE)"
  mpi_enabled="$(cache_value "${cache}" MFEM_USE_MPI)"
  openmp_enabled="$(cache_value "${cache}" MFEM_USE_OPENMP)"
  metis_enabled="$(cache_value "${cache}" MFEM_USE_METIS)"
  openmp_config_enabled="$(config_value "${config}" MFEM_USE_OPENMP)"
  metis_config_enabled="$(config_value "${config}" MFEM_USE_METIS)"
  cuda_enabled="$(cache_value "${cache}" MFEM_USE_CUDA)"
  raja_enabled="$(cache_value "${cache}" MFEM_USE_RAJA)"
  umpire_enabled="$(cache_value "${cache}" MFEM_USE_UMPIRE)"
  caliper_enabled="$(cache_value "${cache}" MFEM_USE_CALIPER)"
  caliper_config_enabled="$(config_value "${config}" MFEM_USE_CALIPER)"
  [[ "${cache_build_type}" == "${CMAKE_BUILD_TYPE}" ]] || {
    echo "ERROR: reused MFEM build type mismatch: expected=${CMAKE_BUILD_TYPE} actual=${cache_build_type:-<missing>}" >&2
    exit 3
  }
  is_cmake_true "${mpi_enabled}" || {
    echo "ERROR: reused MFEM cache does not enable MPI: ${mpi_enabled:-<missing>}" >&2
    exit 3
  }
  grep -Eqi '^set\(MFEM_USE_MPI (1|ON|TRUE|YES)\)$' "${config}" || {
    echo "ERROR: reused MFEM package config does not report MPI enabled: ${config}" >&2
    exit 3
  }
  is_cmake_false "${openmp_enabled}" || {
    echo "ERROR: reused MFEM cache does not explicitly disable OpenMP: ${openmp_enabled:-<missing>}" >&2
    exit 3
  }
  is_cmake_false "${openmp_config_enabled}" || {
    echo "ERROR: reused MFEM package config does not explicitly disable OpenMP: ${openmp_config_enabled:-<missing>}" >&2
    exit 3
  }
  is_cmake_true "${metis_enabled}" || {
    echo "ERROR: reused MFEM cache does not enable METIS: ${metis_enabled:-<missing>}" >&2
    exit 3
  }
  is_cmake_true "${metis_config_enabled}" || {
    echo "ERROR: reused MFEM package config does not report METIS enabled: ${metis_config_enabled:-<missing>}" >&2
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
  if ((CALIPER)); then
    is_cmake_true "${caliper_enabled}" || {
      echo "ERROR: reused Caliper MFEM cache does not enable Caliper: ${caliper_enabled:-<missing>}" >&2
      exit 3
    }
    is_cmake_true "${caliper_config_enabled}" || {
      echo "ERROR: reused Caliper MFEM package config does not report Caliper enabled: ${caliper_config_enabled:-<missing>}" >&2
      exit 3
    }
    cache_cmake_prefix_path="$(canonical_existing_path \
      "$(cache_value "${cache}" CMAKE_PREFIX_PATH)")" || {
      echo "ERROR: reused Caliper MFEM CMAKE_PREFIX_PATH is missing or moved." >&2
      exit 3
    }
    [[ "${cache_cmake_prefix_path}" == "${spack_view}" ]] || {
      echo "ERROR: reused Caliper MFEM CMAKE_PREFIX_PATH does not match the selected Spack view: expected=${spack_view} actual=${cache_cmake_prefix_path}" >&2
      exit 3
    }
  else
    is_cmake_false "${caliper_enabled}" || {
      echo "ERROR: reused non-Caliper MFEM cache does not explicitly disable Caliper: ${caliper_enabled:-<missing>}" >&2
      exit 3
    }
    is_cmake_false "${caliper_config_enabled}" || {
      echo "ERROR: reused non-Caliper MFEM package config does not explicitly disable Caliper: ${caliper_config_enabled:-<missing>}" >&2
      exit 3
    }
  fi

  metis_spec="metis"
  caliper_spec="off"
  umpire_spec="off"
  cuda_spec="off"
  if [[ "${MODE}" == "gpu" ]]; then
    mpi_spec="openmpi+cuda"
    hypre_spec="hypre@develop+mpi+cuda cuda_arch=${SPACK_CUDA_ARCH}"
  else
    mpi_spec="openmpi~cuda"
    hypre_spec="hypre+mpi~cuda"
  fi
  spack_identity "${mpi_spec}" mpi_prefix mpi_hash mpi_compiler
  spack_identity "${hypre_spec}" hypre_prefix hypre_hash hypre_compiler
  spack_identity "${metis_spec}" metis_prefix metis_hash metis_compiler
  hypre_mpi_hash="$(spack_dependency_hash "${hypre_spec}" mpi)"
  [[ "${hypre_mpi_hash}" == "${mpi_hash}" ]] || {
    echo "ERROR: active Spack HYPRE/MPI concretization mismatch: hypre-mpi=${hypre_mpi_hash} mpi=${mpi_hash}" >&2
    exit 3
  }
  [[ "${hypre_compiler}" == "${mpi_compiler}" ]] || {
    echo "ERROR: active Spack HYPRE/MPI compiler mismatch: hypre=${hypre_compiler} mpi=${mpi_compiler}" >&2
    exit 3
  }
  [[ "${metis_compiler}" == "${mpi_compiler}" ]] || {
    echo "ERROR: active Spack METIS/MPI compiler mismatch: metis=${metis_compiler} mpi=${mpi_compiler}" >&2
    exit 3
  }
  caliper_state="off"
  caliper_hash="off"
  if ((CALIPER)); then
    caliper_spec="caliper+mpi~cuda"
    spack_identity "${caliper_spec}" \
      caliper_prefix caliper_hash caliper_compiler
    [[ "${caliper_compiler}" == "${mpi_compiler}" ]] || {
      echo "ERROR: active Spack Caliper/MPI compiler mismatch: caliper=${caliper_compiler} mpi=${mpi_compiler}" >&2
      exit 3
    }
    caliper_mpi_hash="$(spack_dependency_hash "${caliper_spec}" mpi)"
    [[ "${caliper_mpi_hash}" == "${mpi_hash}" ]] || {
      echo "ERROR: active Spack Caliper/MPI concretization mismatch: caliper-mpi=${caliper_mpi_hash} mpi=${mpi_hash}" >&2
      exit 3
    }
    view_caliper_config="$(canonical_existing_path \
      "${spack_view}/share/cmake/caliper/caliper-config.cmake")" || {
      echo "ERROR: selected Caliper Spack view has no usable package config." >&2
      exit 3
    }
    case "${view_caliper_config}" in
      "${caliper_prefix}"/*) ;;
      *)
        echo "ERROR: selected Spack view Caliper config does not match its concrete Caliper prefix." >&2
        exit 3
        ;;
    esac
    caliper_state="${caliper_hash}"
  fi
  cache_hypre="$(canonical_existing_path "$(cache_value "${cache}" HYPRE_DIR)")" || {
    echo "ERROR: reused MFEM HYPRE cache root is missing or moved." >&2
    exit 3
  }
  [[ "${cache_hypre}" == "${hypre_prefix}" ]] || {
    echo "ERROR: reused MFEM HYPRE root mismatch: expected=${hypre_prefix} actual=${cache_hypre}" >&2
    exit 3
  }
  cache_metis="$(canonical_existing_path "$(cache_value "${cache}" METIS_DIR)")" || {
    echo "ERROR: reused MFEM METIS cache root is missing or moved." >&2
    exit 3
  }
  [[ "${cache_metis}" == "${metis_prefix}" ]] || {
    echo "ERROR: reused MFEM METIS root mismatch: expected=${metis_prefix} actual=${cache_metis}" >&2
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
  view_metis_header="$(canonical_existing_path \
    "${spack_view}/include/metis.h")" || {
    echo "ERROR: selected Spack view has no usable METIS header." >&2
    exit 3
  }
  view_metis_library=""
  for view_library_candidate in \
      "${spack_view}/lib/libmetis.so" \
      "${spack_view}/lib64/libmetis.so" \
      "${spack_view}/lib/libmetis.a" \
      "${spack_view}/lib64/libmetis.a"; do
    if [[ -e "${view_library_candidate}" ]]; then
      view_metis_library="$(canonical_existing_path \
        "${view_library_candidate}")"
      break
    fi
  done
  [[ -n "${view_metis_library}" ]] || {
    echo "ERROR: selected Spack view has no usable METIS library." >&2
    exit 3
  }
  case "${view_metis_header}" in
    "${metis_prefix}"/*) ;;
    *)
      echo "ERROR: selected Spack view METIS header does not match its concrete METIS prefix." >&2
      exit 3
      ;;
  esac
  case "${view_metis_library}" in
    "${metis_prefix}"/*) ;;
    *)
      echo "ERROR: selected Spack view METIS library does not match its concrete METIS prefix." >&2
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
  umpire_state="off"
  umpire_hash="off"
  cuda_hash="off"
  configured_archs="none"
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
    cuda_spec="cuda"
    umpire_spec="umpire"
    spack_identity "${cuda_spec}" cuda_prefix cuda_hash cuda_compiler
    spack_identity "${umpire_spec}" \
      umpire_prefix umpire_hash umpire_compiler
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
    install_manifest="${build}/install_manifest.txt"
    require_exact_mfem_file \
      "${install_manifest}" "reused MFEM install manifest"
    MFEM_IDENTITY_CUDA_COMPILER_METADATA_PATH="${cuda_metadata}"
    MFEM_IDENTITY_CUDA_COMPILER_METADATA_SHA256="$(sha256sum \
      "${cuda_metadata}" | cut -d' ' -f1)"
    MFEM_IDENTITY_INSTALL_MANIFEST_PATH="${install_manifest}"
    MFEM_IDENTITY_INSTALL_MANIFEST_SHA256="$(sha256sum \
      "${install_manifest}" | cut -d' ' -f1)"
    [[ "${MFEM_IDENTITY_CUDA_COMPILER_METADATA_SHA256}" =~ ^[0-9a-f]{64}$ &&
       "${MFEM_IDENTITY_INSTALL_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
      echo "ERROR: reused MFEM GPU provenance hashes are malformed." >&2
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
    hypre_cuda_hash="$(spack_dependency_hash "${hypre_spec}" cuda)"
    hypre_umpire_hash="$(spack_dependency_hash "${hypre_spec}" umpire)"
    umpire_cuda_hash="$(spack_dependency_hash "${umpire_spec}" cuda)"
    [[ "${hypre_cuda_hash}" == "${cuda_hash}" &&
       "${hypre_umpire_hash}" == "${umpire_hash}" &&
       "${umpire_cuda_hash}" == "${cuda_hash}" ]] || {
      echo "ERROR: active Spack HYPRE/Umpire/CUDA dependency identities disagree." >&2
      exit 3
    }
    umpire_state="${umpire_hash}"
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
  fi

  MFEM_SOURCE_ROOT="${root}"
  MFEM_BUILD="${build}"
  MFEM_INSTALL="${install}"
  MFEM_REUSE_ABI="build_type=${cache_build_type};openmp=off;metis=${metis_hash};compiler=${mpi_compiler};mpi_spec=${mpi_spec};mpi=${mpi_hash};hypre_spec=${hypre_spec};hypre=${hypre_hash};raja=${raja_state};umpire_spec=${umpire_spec};umpire=${umpire_state};caliper_spec=${caliper_spec};caliper=${caliper_state};cuda_spec=${cuda_spec};cuda=${cuda_hash};dependency_arch=${DEPENDENCY_CUDA_ARCH};mfem_arch=${configured_archs}"
  MFEM_IDENTITY_BUILD_TYPE="${cache_build_type}"
  MFEM_IDENTITY_OPENMP_STATE="OFF"
  MFEM_IDENTITY_METIS_STATE="ON"
  if ((CALIPER)); then
    MFEM_IDENTITY_CALIPER_STATE="ON"
  else
    MFEM_IDENTITY_CALIPER_STATE="OFF"
  fi
  MFEM_IDENTITY_COMPILER="${mpi_compiler}"
  MFEM_IDENTITY_MPI_SPEC="${mpi_spec}"
  MFEM_IDENTITY_MPI_HASH="${mpi_hash}"
  MFEM_IDENTITY_MPI_PREFIX="${mpi_prefix}"
  MFEM_IDENTITY_HYPRE_SPEC="${hypre_spec}"
  MFEM_IDENTITY_HYPRE_HASH="${hypre_hash}"
  MFEM_IDENTITY_HYPRE_PREFIX="${hypre_prefix}"
  MFEM_IDENTITY_METIS_SPEC="${metis_spec}"
  MFEM_IDENTITY_METIS_HASH="${metis_hash}"
  MFEM_IDENTITY_METIS_PREFIX="${metis_prefix}"
  MFEM_IDENTITY_RAJA_STATE="${raja_state}"
  MFEM_IDENTITY_UMPIRE_SPEC="${umpire_spec}"
  MFEM_IDENTITY_UMPIRE_HASH="${umpire_hash}"
  if [[ "${MODE}" == "gpu" ]]; then
    MFEM_IDENTITY_UMPIRE_PREFIX="${umpire_prefix}"
  else
    MFEM_IDENTITY_UMPIRE_PREFIX="off"
  fi
  MFEM_IDENTITY_CALIPER_SPEC="${caliper_spec}"
  MFEM_IDENTITY_CALIPER_HASH="${caliper_hash}"
  MFEM_IDENTITY_CUDA_SPEC="${cuda_spec}"
  MFEM_IDENTITY_CUDA_HASH="${cuda_hash}"
  if [[ "${MODE}" == "gpu" ]]; then
    MFEM_IDENTITY_CUDA_PREFIX="${cuda_prefix}"
    MFEM_IDENTITY_CUDA_COMPILER="${cuda_compiler}"
  else
    MFEM_IDENTITY_CUDA_PREFIX="off"
    MFEM_IDENTITY_CUDA_COMPILER="off"
  fi
  hash_mfem_identity_artifacts \
    "${cache}" "${config}" "${archive}" "${targets}" \
    "${targets_config}" "${compiler_metadata}" "${install_script}" \
    "${check_cache}"
  if [[ "${stamp_policy}" == "require" ]]; then
    if [[ -e "${MFEM_IDENTITY_STAMP}" || -L "${MFEM_IDENTITY_STAMP}" ]]; then
      verify_mfem_identity_stamp
    elif ! verify_reviewed_mfem_admission; then
      echo "ERROR: reused MFEM identity stamp is missing and no exact reviewed admission matches: ${MFEM_IDENTITY_STAMP}" >&2
      exit 3
    fi
  fi
  MFEM_REUSE_STATE="validated-read-only"
  MFEM_TUPLE_STATE="reuse-source-build-install-validated"
}

freeze_reuse_pre_dependency_identity() {
  [[ "${MFEM_REUSE_STATE}" == "validated-read-only" &&
     "${MFEM_TUPLE_STATE}" == "reuse-source-build-install-validated" &&
     "${SPACK_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ &&
     "${SPACK_LOCK_SHA256}" =~ ^[0-9a-f]{64}$ &&
     "${MFEM_IDENTITY_STAMP_SHA256}" =~ ^[0-9a-f]{64}$ &&
     "${MFEM_IDENTITY_CACHE_SHA256}" =~ ^[0-9a-f]{64}$ &&
     "${MFEM_IDENTITY_CONFIG_SHA256}" =~ ^[0-9a-f]{64}$ &&
     "${MFEM_IDENTITY_ARCHIVE_SHA256}" =~ ^[0-9a-f]{64}$ &&
     -n "${MFEM_REUSE_ABI}" ]] || {
    echo "ERROR: cannot freeze an incomplete pre-dependency MFEM reuse identity." >&2
    exit 3
  }
  MFEM_REUSE_PRE_SOURCE_ROOT="${MFEM_SOURCE_ROOT}"
  MFEM_REUSE_PRE_BUILD="${MFEM_BUILD}"
  MFEM_REUSE_PRE_INSTALL="${MFEM_INSTALL}"
  MFEM_REUSE_PRE_MANIFEST_SHA256="${SPACK_MANIFEST_SHA256}"
  MFEM_REUSE_PRE_LOCK_SHA256="${SPACK_LOCK_SHA256}"
  MFEM_REUSE_PRE_ABI="${MFEM_REUSE_ABI}"
  MFEM_REUSE_PRE_IDENTITY_SHA256="${MFEM_IDENTITY_STAMP_SHA256}"
  MFEM_REUSE_PRE_TRUST_MODE="${MFEM_IDENTITY_TRUST_MODE}"
  MFEM_REUSE_PRE_CACHE_SHA256="${MFEM_IDENTITY_CACHE_SHA256}"
  MFEM_REUSE_PRE_CONFIG_SHA256="${MFEM_IDENTITY_CONFIG_SHA256}"
  MFEM_REUSE_PRE_ARCHIVE_SHA256="${MFEM_IDENTITY_ARCHIVE_SHA256}"
  MFEM_REUSE_PRE_TARGETS_SHA256="${MFEM_IDENTITY_TARGETS_SHA256}"
  MFEM_REUSE_PRE_TARGETS_CONFIG_SHA256="${MFEM_IDENTITY_TARGETS_CONFIG_SHA256}"
  MFEM_REUSE_PRE_COMPILER_METADATA_SHA256="${MFEM_IDENTITY_COMPILER_METADATA_SHA256}"
  MFEM_REUSE_PRE_INSTALL_SCRIPT_SHA256="${MFEM_IDENTITY_INSTALL_SCRIPT_SHA256}"
  MFEM_REUSE_PRE_CHECK_CACHE_SHA256="${MFEM_IDENTITY_CHECK_CACHE_SHA256}"
  MFEM_REUSE_TRANSACTION_STATE="pre-dependency-validated"
}

revalidate_reused_mfem_after_dependencies() {
  local frozen_source="${MFEM_REUSE_PRE_SOURCE_ROOT}"
  local frozen_build="${MFEM_REUSE_PRE_BUILD}"
  local frozen_install="${MFEM_REUSE_PRE_INSTALL}"
  local drift=0

  [[ "${MFEM_REUSE_TRANSACTION_STATE}" == \
     "pre-dependency-validated" && -n "${frozen_source}" ]] || {
    echo "ERROR: MFEM reuse post-dependency validation has no frozen pre-dependency identity." >&2
    exit 3
  }
  validate_reused_mfem "${frozen_source}"
  MFEM_REUSE_POST_MANIFEST_SHA256="${SPACK_MANIFEST_SHA256}"
  MFEM_REUSE_POST_LOCK_SHA256="${SPACK_LOCK_SHA256}"
  MFEM_REUSE_POST_ABI="${MFEM_REUSE_ABI}"
  if [[ "${MFEM_SOURCE_ROOT}" != "${frozen_source}" ||
        "${MFEM_BUILD}" != "${frozen_build}" ||
        "${MFEM_INSTALL}" != "${frozen_install}" ]]; then
    echo "ERROR: MFEM reuse source/build/install tuple changed across dependency installation." >&2
    echo "       pre=${frozen_source}|${frozen_build}|${frozen_install}" >&2
    echo "       post=${MFEM_SOURCE_ROOT}|${MFEM_BUILD}|${MFEM_INSTALL}" >&2
    drift=1
  fi
  if [[ "${MFEM_REUSE_POST_ABI}" != "${MFEM_REUSE_PRE_ABI}" ]]; then
    echo "ERROR: MFEM reuse ABI changed across dependency installation." >&2
    echo "       pre=${MFEM_REUSE_PRE_ABI}" >&2
    echo "       post=${MFEM_REUSE_POST_ABI}" >&2
    drift=1
  fi
  if [[ "${MFEM_IDENTITY_TRUST_MODE}" != "${MFEM_REUSE_PRE_TRUST_MODE}" ||
        "${MFEM_IDENTITY_STAMP_SHA256}" != \
        "${MFEM_REUSE_PRE_IDENTITY_SHA256}" ]]; then
    echo "ERROR: MFEM reuse trust anchor changed across dependency installation." >&2
    echo "       pre=${MFEM_REUSE_PRE_TRUST_MODE}|${MFEM_REUSE_PRE_IDENTITY_SHA256}" >&2
    echo "       post=${MFEM_IDENTITY_TRUST_MODE}|${MFEM_IDENTITY_STAMP_SHA256}" >&2
    drift=1
  fi
  if [[ "${MFEM_REUSE_POST_MANIFEST_SHA256}" != \
        "${MFEM_REUSE_PRE_MANIFEST_SHA256}" ]]; then
    echo "ERROR: Spack manifest changed across MFEM reuse dependency installation." >&2
    echo "       pre=${MFEM_REUSE_PRE_MANIFEST_SHA256}" >&2
    echo "       post=${MFEM_REUSE_POST_MANIFEST_SHA256}" >&2
    drift=1
  fi
  if [[ "${MFEM_REUSE_POST_LOCK_SHA256}" != \
        "${MFEM_REUSE_PRE_LOCK_SHA256}" ]]; then
    echo "ERROR: Spack lock changed across MFEM reuse dependency installation." >&2
    echo "       pre=${MFEM_REUSE_PRE_LOCK_SHA256}" >&2
    echo "       post=${MFEM_REUSE_POST_LOCK_SHA256}" >&2
    drift=1
  fi
  if ((drift)); then
    exit 3
  fi
  MFEM_REUSE_TRANSACTION_STATE="post-dependency-validated"
  echo "[reuse-transaction] mfem_reuse_transaction_state=${MFEM_REUSE_TRANSACTION_STATE} phase=post-dependency state=validated mfem_source=${MFEM_SOURCE_ROOT} mfem_build=${MFEM_BUILD} mfem_install=${MFEM_INSTALL}"
  echo "[reuse-transaction] mfem_reuse_post_abi=${MFEM_REUSE_POST_ABI}"
  echo "[reuse-transaction] spack_post_manifest_sha256=${MFEM_REUSE_POST_MANIFEST_SHA256} spack_post_lock_sha256=${MFEM_REUSE_POST_LOCK_SHA256}"
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
  MFEM_REUSE_STAMP="${MFEM_REUSE_REQUEST}/${MFEM_BUILD_NAME}/${MFEM_IDENTITY_STAMP_NAME}"
  if [[ -e "${MFEM_REUSE_STAMP}" || -L "${MFEM_REUSE_STAMP}" ]]; then
    select_reviewed_admission_route || true
    validate_reused_mfem "${MFEM_REUSE_REQUEST}"
    freeze_reuse_pre_dependency_identity
  else
    MFEM_FRESH_FROM_SEED=1
    plan_target_local_fresh_mfem "${MFEM_REUSE_REQUEST}"
  fi
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
if [[ "${MFEM_FRESH_FROM_SEED}" == "1" ]]; then
  echo "[build-plan] mfem_route=target-local-fresh-from-seed seed=${MFEM_FRESH_SEED_ROOT} target=${MFEM_FRESH_TARGET_ROOT}"
fi
echo "[build-plan] mfem_source_pin=${MFEM_SOURCE_PIN} fem_picojson_blob=${MFEM_PICOJSON_BLOB} general_tinyxml2_blob=${MFEM_TINYXML2_BLOB}"
if [[ "${MFEM_REUSE_STATE}" == "validated-read-only" ]]; then
  echo "[build-plan] mfem_reuse_abi=${MFEM_REUSE_ABI}"
  echo "[build-plan] mfem_identity_trust_mode=${MFEM_IDENTITY_TRUST_MODE} mfem_identity_sha256=${MFEM_IDENTITY_STAMP_SHA256}"
  echo "[build-plan] spack_manifest_sha256=${SPACK_MANIFEST_SHA256} spack_lock_sha256=${SPACK_LOCK_SHA256}"
  echo "[build-plan] mfem_reuse_transaction_state=${MFEM_REUSE_TRANSACTION_STATE}"
  echo "[build-plan] mfem_reuse_pre_abi=${MFEM_REUSE_PRE_ABI}"
  echo "[build-plan] spack_pre_manifest_sha256=${MFEM_REUSE_PRE_MANIFEST_SHA256} spack_pre_lock_sha256=${MFEM_REUSE_PRE_LOCK_SHA256}"
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
  if [[ "${MFEM_REUSE_STATE}" == "validated-read-only" ]]; then
    revalidate_reused_mfem_after_dependencies
  fi
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
  export DG_IGR_TEST_REUSE_PROFILE="${PROFILE}"
  export DG_IGR_TEST_REUSE_CALIPER="${CALIPER}"
  export DG_IGR_TEST_REUSE_CUDA_ARCH="${DEPENDENCY_CUDA_ARCH}"
else
  materialize_target_local_fresh_mfem
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
  if [[ -n "${TEST_REGEX}" ]]; then
    selected_tests="$("${CTEST_BIN}" --test-dir "${APP_BUILD}" -N -R "${TEST_REGEX}")" || {
      echo "ERROR: --test-regex is not accepted by CTest: ${TEST_REGEX}" >&2
      exit 2
    }
    if ! grep -Eq '^  Test #[0-9]+:' <<< "${selected_tests}"; then
      echo "ERROR: --test-regex selected no tests: ${TEST_REGEX}" >&2
      exit 2
    fi
    "${CTEST_BIN}" --test-dir "${APP_BUILD}" --output-on-failure \
      -R "${TEST_REGEX}"
  else
    "${CTEST_BIN}" --test-dir "${APP_BUILD}" --output-on-failure
  fi
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
