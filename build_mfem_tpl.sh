#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="cpu"
JOBS="${NPROC:-4}"
CLEAN=0
CALIPER=0
SPACK_CUDA_ARCH="${SPACK_CUDA_ARCH:-120}"
CUDA_ARCH_WAS_SET=0

usage() {
  cat <<'EOF'
Usage: ./build_mfem_tpl.sh [--cpu-only | --gpu] [options]

Options:
  --cpu-only        Build MPI-enabled MFEM without CUDA (default).
  --gpu             Build MPI/CUDA-enabled MFEM.
  --caliper         Build CPU-only MFEM with Caliper support.
  -j, --jobs N      Parallel build jobs (default: NPROC or 4).
  --cuda-arch N     Concrete GPU architecture profile (default: 120).
  --clean           Remove the selected MFEM build and install directories.
  --no-clean        Reconfigure and continue the existing build.
  -h, --help        Show this help.

Environment overrides:
  DG_IGR_SPACK_ENV, MFEM_SRC, MFEM_BUILD, MFEM_INSTALL, CMAKE_BUILD_TYPE,
  CUDA_ARCHS, SPACK_CUDA_ARCH
EOF
}

while (($#)); do
  case "$1" in
    --cpu|--cpu-only) MODE="cpu" ;;
    --gpu) MODE="gpu" ;;
    --caliper) CALIPER=1 ;;
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
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

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

REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"
MFEM_SRC="${MFEM_SRC:-${REPO_DIR}/tpl/mfem}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

if [[ "${MODE}" == "gpu" ]]; then
  SPACK_ENV="${DG_IGR_SPACK_ENV:-${REPO_DIR}/spack-envs/cuda-sm${SPACK_CUDA_ARCH}}"
  MFEM_BUILD="${MFEM_BUILD:-${MFEM_SRC}/build-gpu-sm${SPACK_CUDA_ARCH}}"
  MFEM_INSTALL="${MFEM_INSTALL:-${MFEM_SRC}/install-gpu-sm${SPACK_CUDA_ARCH}}"
  CUDA_ARCHS="${CUDA_ARCHS:-80-virtual}"
  HYPRE_SPEC="hypre@develop+mpi+cuda cuda_arch=${SPACK_CUDA_ARCH}"
  MPI_SPEC="openmpi+cuda"
else
  if ((CALIPER)); then
    SPACK_ENV="${DG_IGR_SPACK_ENV:-${REPO_DIR}/spack-envs/cpu-caliper}"
    MFEM_BUILD="${MFEM_BUILD:-${MFEM_SRC}/build-cpu-caliper}"
    MFEM_INSTALL="${MFEM_INSTALL:-${MFEM_SRC}/install-cpu-caliper}"
  else
    SPACK_ENV="${DG_IGR_SPACK_ENV:-${REPO_DIR}/spack-envs/cpu}"
    MFEM_BUILD="${MFEM_BUILD:-${MFEM_SRC}/build-cpu}"
    MFEM_INSTALL="${MFEM_INSTALL:-${MFEM_SRC}/install-cpu}"
  fi
  HYPRE_SPEC="hypre+mpi~cuda"
  MPI_SPEC="openmpi~cuda"
fi

[[ -f "${SPACK_ENV}/spack.yaml" ]] || {
  echo "ERROR: Spack environment not found at ${SPACK_ENV}" >&2
  exit 2
}
SPACK_VIEW="${SPACK_ENV}/.spack-env/view"
[[ -d "${SPACK_VIEW}" ]] || {
  echo "ERROR: Spack environment view not found at ${SPACK_VIEW}" >&2
  exit 3
}
export PATH="${SPACK_VIEW}/bin:${PATH}"
export LD_LIBRARY_PATH="${SPACK_VIEW}/lib:${SPACK_VIEW}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
[[ -f "${MFEM_SRC}/CMakeLists.txt" ]] || {
  echo "ERROR: MFEM submodule is missing; run git submodule update --init." >&2
  exit 2
}

spack_location() {
  spack -e "${SPACK_ENV}" location -i "$1"
}

CMAKE_PREFIX="$(spack_location cmake)"
MPI_PREFIX="$(spack_location "${MPI_SPEC}")"
HYPRE_PREFIX="$(spack_location "${HYPRE_SPEC}")"
METIS_PREFIX="$(spack_location metis)"
CMAKE_BIN="${CMAKE_PREFIX}/bin/cmake"

if ((CLEAN)); then
  rm -rf "${MFEM_BUILD}" "${MFEM_INSTALL}"
fi

cmake_args=(
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  -DCMAKE_INSTALL_PREFIX="${MFEM_INSTALL}"
  -DCMAKE_CXX_COMPILER="${MPI_PREFIX}/bin/mpicxx"
  -DMFEM_USE_MPI=YES
  -DMFEM_USE_OPENMP=OFF
  -DMFEM_USE_METIS=ON
  -DMFEM_ENABLE_BENCHMARKS=OFF
  -DMFEM_ENABLE_EXAMPLES=OFF
  -DMFEM_ENABLE_MINIAPPS=OFF
  -DMFEM_ENABLE_TESTING=OFF
  -DHYPRE_DIR="${HYPRE_PREFIX}"
  -DMETIS_DIR="${METIS_PREFIX}"
)

if [[ "${MODE}" == "gpu" ]]; then
  CUDA_PREFIX="$(spack_location cuda)"
  UMPIRE_PREFIX="$(spack_location umpire)"
  cmake_args+=(
    -DMFEM_USE_CUDA=ON
    -DMFEM_USE_UMPIRE=ON
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}"
    -DCMAKE_CUDA_COMPILER="${CUDA_PREFIX}/bin/nvcc"
    -DCUDAToolkit_ROOT="${CUDA_PREFIX}"
    -DUMPIRE_DIR="${UMPIRE_PREFIX}"
  )
else
  cmake_args+=(
    -DMFEM_USE_CUDA=OFF
    -DMFEM_USE_UMPIRE=OFF
  )
  if ((CALIPER)); then
    cmake_args+=(
      -DCMAKE_PREFIX_PATH="${SPACK_VIEW}"
      -DMFEM_USE_CALIPER=ON
    )
  fi
fi

echo "[mfem] mode=${MODE} env=${SPACK_ENV} build=${MFEM_BUILD} install=${MFEM_INSTALL} jobs=${JOBS}"
configure_args=()
if ((CLEAN)); then
  configure_args+=(--fresh)
fi
"${CMAKE_BIN}" "${configure_args[@]}" \
  -S "${MFEM_SRC}" -B "${MFEM_BUILD}" "${cmake_args[@]}"
"${CMAKE_BIN}" --build "${MFEM_BUILD}" -j"${JOBS}"
"${CMAKE_BIN}" --install "${MFEM_BUILD}"

MFEM_CONFIG="${MFEM_INSTALL}/lib/cmake/mfem/MFEMConfig.cmake"
[[ -f "${MFEM_CONFIG}" ]] || {
  echo "ERROR: MFEMConfig.cmake was not installed at ${MFEM_CONFIG}" >&2
  exit 3
}

if [[ "${MODE}" == "gpu" ]]; then
  grep -Fq 'set(MFEM_USE_CUDA ON)' "${MFEM_CONFIG}" || {
    echo "ERROR: installed MFEM does not report CUDA support." >&2
    exit 3
  }
else
  grep -Fq 'set(MFEM_USE_CUDA OFF)' "${MFEM_CONFIG}" || {
    echo "ERROR: installed MFEM unexpectedly reports CUDA support." >&2
    exit 3
  }
  if ((CALIPER)); then
    grep -Fq 'set(MFEM_USE_CALIPER ON)' "${MFEM_CONFIG}" || {
      echo "ERROR: installed MFEM does not report Caliper support." >&2
      exit 3
    }
  fi
fi

echo "[mfem] complete: ${MFEM_CONFIG}"
