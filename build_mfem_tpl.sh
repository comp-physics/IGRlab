#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ "${DG_IGR_BUILD_HELPER_CALLER:-}" == "${SCRIPT_DIR}/build.sh" ]] || {
  echo "ERROR: build_mfem_tpl.sh is a private helper; invoke build.sh instead." >&2
  exit 2
}
MODE="cpu"
JOBS="${NPROC:-4}"
CLEAN=0
CALIPER=0
SPACK_CUDA_ARCH="${SPACK_CUDA_ARCH:-120}"
CUDA_ARCH_WAS_SET=0
CUDA_CODEGEN=""
CUDA_CODEGEN_WAS_SET=0
PRINT_PLAN=0

usage() {
  cat <<'EOF'
Usage: ./build_mfem_tpl.sh [--cpu-only | --gpu] [options]

Options:
  --cpu-only        Build MPI-enabled MFEM without CUDA (default).
  --gpu             Build MPI/CUDA-enabled MFEM.
  --caliper         Build CPU-only MFEM with Caliper support.
  -j, --jobs N      Parallel build jobs (default: NPROC or 4).
  --cuda-arch N     Concrete GPU architecture profile (default: 120).
  --cuda-codegen A  MFEM CUDA code-generation architecture.
  --print-plan      Validate and print the resolved MFEM policy, then exit.
  --clean           Rejected: build.sh owns scoped MFEM cleanup.
  --no-clean        Required internal helper mode from build.sh.
  -h, --help        Show this help.

Environment overrides:
  DG_IGR_SPACK_ENV, MFEM_SRC, MFEM_BUILD, MFEM_INSTALL, CMAKE_BUILD_TYPE,
  SPACK_CUDA_ARCH

CUDA_ARCHS is intentionally rejected in GPU mode. The canonical build driver
passes MFEM's code-generation architecture explicitly with --cuda-codegen.
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
    --cuda-codegen)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
      CUDA_CODEGEN="$2"
      CUDA_CODEGEN_WAS_SET=1
      shift
      ;;
    -j|--jobs)
      [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
      JOBS="$2"
      shift
      ;;
    --print-plan) PRINT_PLAN=1 ;;
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
if [[ "${MODE}" == "cpu" && "${CUDA_CODEGEN_WAS_SET}" == "1" ]]; then
  echo "ERROR: --cuda-codegen is only valid with --gpu." >&2
  exit 2
fi
if [[ "${MODE}" == "gpu" && "${CALIPER}" == "1" ]]; then
  echo "ERROR: --caliper is only valid with --cpu-only." >&2
  exit 2
fi
if [[ "${MODE}" == "gpu" && -n "${CUDA_ARCHS+x}" ]]; then
  echo "ERROR: CUDA_ARCHS is ambiguous and is rejected in GPU mode." >&2
  echo "       Pass MFEM code generation explicitly with --cuda-codegen." >&2
  exit 2
fi
if ((CLEAN)); then
  echo "ERROR: build_mfem_tpl.sh is a private helper and rejects --clean; build.sh owns scoped cleanup." >&2
  exit 2
fi

MFEM_SRC="${MFEM_SRC:?private helper requires MFEM_SRC from build.sh}"
MFEM_BUILD="${MFEM_BUILD:?private helper requires MFEM_BUILD from build.sh}"
MFEM_INSTALL="${MFEM_INSTALL:?private helper requires MFEM_INSTALL from build.sh}"
SPACK_ENV="${DG_IGR_SPACK_ENV:?private helper requires DG_IGR_SPACK_ENV from build.sh}"
MFEM_SRC_CANON="$(realpath -m -- "${MFEM_SRC}")" || {
  echo "ERROR: private MFEM helper cannot canonicalize MFEM_SRC." >&2
  exit 2
}
MFEM_BUILD_CANON="$(realpath -m -- "${MFEM_BUILD}")" || {
  echo "ERROR: private MFEM helper cannot canonicalize MFEM_BUILD." >&2
  exit 2
}
MFEM_INSTALL_CANON="$(realpath -m -- "${MFEM_INSTALL}")" || {
  echo "ERROR: private MFEM helper cannot canonicalize MFEM_INSTALL." >&2
  exit 2
}
case "${MFEM_BUILD_CANON}" in
  "${MFEM_SRC_CANON}"|"${MFEM_SRC_CANON}"/*)
    echo "ERROR: private MFEM helper rejects MFEM_BUILD inside MFEM_SRC." >&2
    exit 2
    ;;
esac
case "${MFEM_INSTALL_CANON}" in
  "${MFEM_SRC_CANON}"|"${MFEM_SRC_CANON}"/*)
    echo "ERROR: private MFEM helper rejects MFEM_INSTALL inside MFEM_SRC." >&2
    exit 2
    ;;
esac
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

if [[ "${MODE}" == "gpu" ]]; then
  PROFILE="cuda-sm${SPACK_CUDA_ARCH}"
  DEPENDENCY_CUDA_ARCH="${SPACK_CUDA_ARCH}"
  EXPECTED_CUDA_CODEGEN="${SPACK_CUDA_ARCH}"
  if [[ "${SPACK_CUDA_ARCH}" == "120" ]]; then
    EXPECTED_CUDA_CODEGEN="80-virtual"
  fi
  if [[ -z "${CUDA_CODEGEN}" ]]; then
    CUDA_CODEGEN="${EXPECTED_CUDA_CODEGEN}"
  fi
  if [[ "${CUDA_CODEGEN}" != "${EXPECTED_CUDA_CODEGEN}" ]]; then
    echo "ERROR: CUDA profile ${SPACK_CUDA_ARCH} requires --cuda-codegen ${EXPECTED_CUDA_CODEGEN} for MFEM." >&2
    if [[ "${SPACK_CUDA_ARCH}" == "120" ]]; then
      echo "       Native MFEM sm_120 code generation is blocked because it can exhaust memory." >&2
    fi
    exit 2
  fi
  HYPRE_SPEC="hypre@develop+mpi+cuda cuda_arch=${SPACK_CUDA_ARCH}"
  if [[ "${SPACK_CUDA_ARCH}" == "80" ]]; then
    # The SM80 manifest pins a compatible release; identity still binds its DAG hash.
    HYPRE_SPEC="hypre+mpi+cuda cuda_arch=${SPACK_CUDA_ARCH}"
  fi
  MPI_SPEC="openmpi+cuda"
else
  if ((CALIPER)); then
    PROFILE="cpu-caliper"
  else
    PROFILE="cpu"
  fi
  HYPRE_SPEC="hypre+mpi~cuda"
  MPI_SPEC="openmpi~cuda"
  DEPENDENCY_CUDA_ARCH="none"
  CUDA_CODEGEN="none"
fi

echo "[mfem-plan] mode=${MODE} profile=${PROFILE} dependency_cuda_arch=${DEPENDENCY_CUDA_ARCH} cuda_codegen_archs=${CUDA_CODEGEN} jobs=${JOBS}"
echo "[mfem-plan] env=${SPACK_ENV} source=${MFEM_SRC} build=${MFEM_BUILD} install=${MFEM_INSTALL}"

if ((PRINT_PLAN)); then
  exit 0
fi

command -v spack >/dev/null 2>&1 || {
  echo "ERROR: spack is not available; source Spack's setup-env.sh first." >&2
  exit 2
}

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

configure_sm80_hypre_linalg() {
  [[ "${SPACK_CUDA_ARCH}" == "80" ]] || return 0
  local prefix hash dependencies library="" candidate
  prefix="$(spack_location openblas)"
  prefix="$(realpath -e -- "${prefix}")" || exit 3
  hash="$(SPACK_COLOR=never spack -e "${SPACK_ENV}" find --format '{hash}' openblas)"
  dependencies="$(SPACK_COLOR=never spack -e "${SPACK_ENV}" find \
    --format '{^blas.hash}|{^lapack.hash}' "${HYPRE_SPEC}")"
  [[ "${hash}" =~ ^[a-z0-9]{32}$ && "${dependencies}" == "${hash}|${hash}" ]] || {
    echo "ERROR: SM80 HYPRE BLAS/LAPACK must match the selected Spack OpenBLAS identity." >&2
    exit 3
  }
  for candidate in "${prefix}/lib/libopenblas.so" "${prefix}/lib64/libopenblas.so"; do
    if [[ -f "${candidate}" ]]; then
      library="$(realpath -e -- "${candidate}")" || exit 3
      break
    fi
  done
  case "${library}" in
    "${prefix}"/*) ;;
    *)
      echo "ERROR: SM80 requires a shared OpenBLAS library inside its concrete prefix: ${prefix}" >&2
      exit 3
      ;;
  esac
  # FindBLAS/FindLAPACK reset their output *_LIBRARIES variables. Pin their
  # actual find_library cache entries while retaining their symbol checks.
  # MFEM must export Umpire's concrete imported archive after static HYPRE;
  # the bare umpire target can otherwise be emitted before HYPRE by consumers.
  # Clear only HYPRE's discovery results so an iterative configure rebuilds
  # that dependency list instead of taking FindHYPRE's cached early return.
  cmake_args+=(
    -UHYPRE_FOUND -UHYPRE_LIBRARIES -UHYPRE_INCLUDE_DIRS
    '-DHYPRE_REQUIRED_PACKAGES:STRING=BLAS;LAPACK;UMPIRE'
    -DUMPIRE_TARGET_NAMES:STRING=umpire
    -DUMPIRE_TARGET_FORCE:BOOL=OFF
    -DBLA_VENDOR:STRING=OpenBLAS
    -DBLAS_openblas_LIBRARY:FILEPATH="${library}"
    -DLAPACK_openblas_LIBRARY:FILEPATH="${library}"
  )
}

CMAKE_PREFIX="$(spack_location cmake)"
MPI_PREFIX="$(spack_location "${MPI_SPEC}")"
HYPRE_PREFIX="$(spack_location "${HYPRE_SPEC}")"
METIS_PREFIX="$(spack_location metis)"
CMAKE_BIN="${CMAKE_PREFIX}/bin/cmake"

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
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_CODEGEN}"
    -DCMAKE_CUDA_COMPILER="${CUDA_PREFIX}/bin/nvcc"
    -DCUDAToolkit_ROOT="${CUDA_PREFIX}"
    -DUMPIRE_DIR="${UMPIRE_PREFIX}"
  )
  configure_sm80_hypre_linalg
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
if [[ "${MODE}" == "gpu" ]]; then
  mfem_cache="${MFEM_BUILD}/CMakeCache.txt"
  [[ -f "${mfem_cache}" ]] || {
    echo "ERROR: MFEM CMake cache was not created at ${mfem_cache}" >&2
    exit 3
  }
  configured_mfem_cuda_archs="$(
    sed -n 's/^CMAKE_CUDA_ARCHITECTURES:[^=]*=//p' "${mfem_cache}" | tail -n 1
  )"
  if [[ "${configured_mfem_cuda_archs}" != "${CUDA_CODEGEN}" ]]; then
    echo "ERROR: MFEM CUDA architecture cache mismatch." >&2
    echo "       requested=${CUDA_CODEGEN} configured=${configured_mfem_cuda_archs:-<missing>}" >&2
    exit 3
  fi
fi
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
