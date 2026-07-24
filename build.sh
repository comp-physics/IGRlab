#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="cpu"
JOBS="${NPROC:-4}"
CLEAN=1
INSTALL_DEPS=1
RUN_TESTS=1
RUN_SMOKE=0
CALIPER=0
HDG_EULER_EXPERIMENT="OFF"
SPACK_CUDA_ARCH="${SPACK_CUDA_ARCH:-120}"
CUDA_ARCH_WAS_SET=0

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
  -j, --jobs N      Parallel build jobs (default: NPROC or 4).
  --cuda-arch N     Concrete GPU architecture profile (default: 120).
  --clean           Rebuild the selected mode from clean caches (default).
  --no-clean        Reuse existing build and install directories.
  --no-deps         Do not run `spack install`.
  --no-tests        Do not run CTest.
  --smoke           Run a short two-rank Sedov smoke test after tests.
  -h, --help        Show this help.

Environment overrides:
  DG_IGR_SPACK_ENV, CUDA_ARCHS, SPACK_CUDA_ARCH, CMAKE_BUILD_TYPE,
  DG_IGR_ACCELERATOR_KERNEL_SOURCES
EOF
}

while (($#)); do
  case "$1" in
    --cpu|--cpu-only) MODE="cpu" ;;
    --gpu) MODE="gpu" ;;
    --caliper) CALIPER=1 ;;
    --hdg-euler-experiment) HDG_EULER_EXPERIMENT="ON" ;;
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

if [[ "${MODE}" == "gpu" && "${HDG_EULER_EXPERIMENT}" == "ON" ]]; then
  echo "ERROR: --hdg-euler-experiment is only valid with --cpu-only." >&2
  exit 2
fi
[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: jobs must be a positive integer (got '${JOBS}')" >&2
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
command -v spack >/dev/null 2>&1 || {
  echo "ERROR: spack is not available; source Spack's setup-env.sh first." >&2
  exit 2
}

CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
if [[ "${MODE}" == "gpu" ]]; then
  PROFILE="cuda-sm${SPACK_CUDA_ARCH}"
  SPACK_ENV="${DG_IGR_SPACK_ENV:-${SCRIPT_DIR}/spack-envs/${PROFILE}}"
  MFEM_INSTALL="${SCRIPT_DIR}/tpl/mfem/install-gpu-sm${SPACK_CUDA_ARCH}"
  APP_BUILD="${SCRIPT_DIR}/build-gpu-sm${SPACK_CUDA_ARCH}"
  APP_INSTALL="${SCRIPT_DIR}/install-gpu-sm${SPACK_CUDA_ARCH}"
  CUDA_ARCHS="${CUDA_ARCHS:-80-virtual}"
else
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
fi

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

mfem_args=("--${MODE}" --jobs "${JOBS}")
if ((CALIPER)); then
  mfem_args+=(--caliper)
fi
if [[ "${MODE}" == "gpu" ]]; then
  mfem_args+=(--cuda-arch "${SPACK_CUDA_ARCH}")
fi
if ((CLEAN)); then
  mfem_args+=(--clean)
else
  mfem_args+=(--no-clean)
fi
DG_IGR_SPACK_ENV="${SPACK_ENV}" CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
  "${SCRIPT_DIR}/build_mfem_tpl.sh" "${mfem_args[@]}"

CMAKE_BIN="$(spack -e "${SPACK_ENV}" location -i cmake)/bin/cmake"
MPI_PREFIX="$(spack -e "${SPACK_ENV}" location -i openmpi)"
if ((CLEAN)); then
  rm -rf "${APP_BUILD}" "${APP_INSTALL}"
fi

echo "[build] configuring app (${MODE}, hdg_euler_experiment=${HDG_EULER_EXPERIMENT})"
app_cmake_args=(
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  -DCMAKE_CXX_COMPILER="${MPI_PREFIX}/bin/mpicxx"
  -DCMAKE_INSTALL_PREFIX="${APP_INSTALL}"
  -DMFEM_DIR="${MFEM_INSTALL}/lib/cmake/mfem"
  -DZLIB_ROOT="${SPACK_VIEW}"
  -DDG_IGR_ENABLE_HDG_EULER_EXPERIMENT="${HDG_EULER_EXPERIMENT}"
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
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}"
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

echo "[build] complete: profile=${PROFILE} hdg_euler_experiment=${HDG_EULER_EXPERIMENT} build=${APP_BUILD} install=${APP_INSTALL}"
