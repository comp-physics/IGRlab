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
MFEM_REUSE_REQUEST=""
MFEM_REUSE_ROOT=""
MFEM_REUSE_BUILD=""
MFEM_REUSE_STATE="disabled"

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
                    Reuse a validated, read-only MFEM build/install rooted at
                    DIR instead of invoking build_mfem_tpl.sh.
  --print-plan      Validate and print the resolved build policy, then exit.
  --clean           Rebuild the selected mode from clean caches (default).
  --no-clean        Reuse existing build and install directories.
  --no-deps         Do not run `spack install`.
  --no-tests        Do not run CTest.
  --smoke           Run a short two-rank Sedov smoke test after tests.
  -h, --help        Show this help.

Environment overrides:
  DG_IGR_SPACK_ENV, SPACK_CUDA_ARCH, MFEM_CUDA_ARCHS, APP_CUDA_ARCHS,
  CMAKE_BUILD_TYPE, DG_IGR_ACCELERATOR_KERNEL_SOURCES

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
  local root expected_pin root_head root_top tracked_status
  local build_name install_name build install cache config archive
  local cache_home cache_prefix canonical_home canonical_prefix
  local mpi_enabled cuda_enabled umpire_enabled configured_archs

  command -v realpath >/dev/null 2>&1 || {
    echo "ERROR: --reuse-mfem-root requires realpath." >&2
    exit 3
  }
  [[ -d "${requested}" ]] || {
    echo "ERROR: reused MFEM root does not exist or is not a directory: ${requested}" >&2
    exit 3
  }
  root="$(realpath -e -- "${requested}")" || {
    echo "ERROR: unable to canonicalize reused MFEM root: ${requested}" >&2
    exit 3
  }
  expected_pin="$(git -C "${SCRIPT_DIR}" rev-parse HEAD:tpl/mfem 2>/dev/null)" || {
    echo "ERROR: current application HEAD has no readable tpl/mfem gitlink." >&2
    exit 3
  }
  root_top="$(git -C "${root}" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "ERROR: reused MFEM root is not a Git checkout: ${root}" >&2
    exit 3
  }
  root_top="$(realpath -e -- "${root_top}")"
  [[ "${root_top}" == "${root}" ]] || {
    echo "ERROR: reused MFEM root is not the checkout top level: root=${root} top=${root_top}" >&2
    exit 3
  }
  root_head="$(git -C "${root}" rev-parse HEAD 2>/dev/null)" || {
    echo "ERROR: reused MFEM root has no readable HEAD: ${root}" >&2
    exit 3
  }
  [[ "${root_head}" == "${expected_pin}" ]] || {
    echo "ERROR: reused MFEM pin mismatch: expected=${expected_pin} actual=${root_head}" >&2
    exit 3
  }
  tracked_status="$(GIT_OPTIONAL_LOCKS=0 git -C "${root}" status --porcelain=v1 --untracked-files=no)" || {
    echo "ERROR: unable to inspect reused MFEM tracked state: ${root}" >&2
    exit 3
  }
  [[ -z "${tracked_status}" ]] || {
    echo "ERROR: reused MFEM tracked source/index is dirty: ${root}" >&2
    printf '%s\n' "${tracked_status}" >&2
    exit 3
  }

  if [[ "${MODE}" == "gpu" ]]; then
    build_name="build-gpu-sm${SPACK_CUDA_ARCH}"
    install_name="install-gpu-sm${SPACK_CUDA_ARCH}"
  elif ((CALIPER)); then
    build_name="build-cpu-caliper"
    install_name="install-cpu-caliper"
  else
    build_name="build-cpu"
    install_name="install-cpu"
  fi
  build="${root}/${build_name}"
  install="${root}/${install_name}"
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
  umpire_enabled="$(cache_value "${cache}" MFEM_USE_UMPIRE)"
  is_cmake_true "${mpi_enabled}" || {
    echo "ERROR: reused MFEM cache does not enable MPI: ${mpi_enabled:-<missing>}" >&2
    exit 3
  }
  grep -Eqi '^set\(MFEM_USE_MPI (1|ON|TRUE|YES)\)$' "${config}" || {
    echo "ERROR: reused MFEM package config does not report MPI enabled: ${config}" >&2
    exit 3
  }
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

  MFEM_REUSE_ROOT="${root}"
  MFEM_REUSE_BUILD="${build}"
  MFEM_REUSE_STATE="validated-read-only"
  MFEM_INSTALL="${install}"
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

CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
if [[ "${MODE}" == "gpu" ]]; then
  PROFILE="cuda-sm${SPACK_CUDA_ARCH}"
  DEPENDENCY_CUDA_ARCH="${SPACK_CUDA_ARCH}"
  SPACK_ENV="${DG_IGR_SPACK_ENV:-${SCRIPT_DIR}/spack-envs/${PROFILE}}"
  MFEM_INSTALL="${SCRIPT_DIR}/tpl/mfem/install-gpu-sm${SPACK_CUDA_ARCH}"
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
    MFEM_INSTALL="${SCRIPT_DIR}/tpl/mfem/install-cpu-caliper"
    APP_BUILD="${SCRIPT_DIR}/build-cpu-caliper"
    APP_INSTALL="${SCRIPT_DIR}/install-cpu-caliper"
  else
    PROFILE="cpu"
    SPACK_ENV="${DG_IGR_SPACK_ENV:-${SCRIPT_DIR}/spack-envs/cpu}"
    MFEM_INSTALL="${SCRIPT_DIR}/tpl/mfem/install-cpu"
    APP_BUILD="${SCRIPT_DIR}/build-cpu"
    APP_INSTALL="${SCRIPT_DIR}/install-cpu"
  fi
  MFEM_CUDA_ARCHS="none"
  APP_CUDA_ARCHS="none"
fi

if [[ -n "${MFEM_REUSE_REQUEST}" ]]; then
  validate_reused_mfem "${MFEM_REUSE_REQUEST}"
fi

echo "[build-plan] mode=${MODE} profile=${PROFILE} dependency_cuda_arch=${DEPENDENCY_CUDA_ARCH} mfem_cuda_archs=${MFEM_CUDA_ARCHS} app_cuda_archs=${APP_CUDA_ARCHS} app_jobs=${JOBS} mfem_jobs=${MFEM_JOBS}"
echo "[build-plan] env=${SPACK_ENV} app_build=${APP_BUILD} app_install=${APP_INSTALL}"
echo "[build-plan] hdg_runtime=${HDG_RUNTIME} hdg_auxiliary_experiment=${HDG_AUXILIARY_EXPERIMENT} hdg_euler_experiment=${HDG_EULER_EXPERIMENT} hdg_euler_device_experiment=${HDG_EULER_DEVICE_EXPERIMENT}"
echo "[build-plan] mfem_reuse_root=${MFEM_REUSE_ROOT:-none} mfem_reuse_build=${MFEM_REUSE_BUILD:-none} mfem_install=${MFEM_INSTALL} mfem_reuse_state=${MFEM_REUSE_STATE}"

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
  echo "[mfem] reuse validated: root=${MFEM_REUSE_ROOT} build=${MFEM_REUSE_BUILD} install=${MFEM_INSTALL}"
else
  DG_IGR_SPACK_ENV="${SPACK_ENV}" CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    "${SCRIPT_DIR}/build_mfem_tpl.sh" "${mfem_args[@]}"
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
