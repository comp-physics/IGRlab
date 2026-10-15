#!/usr/bin/env bash
set -euo pipefail

# Exercise the real MFEM artifact validator without a compiler, Spack install,
# or GPU. Only source-checkout admission is replaced; its separate source/Git
# fixtures remain in check_cuda_build_policy.sh.
TEST_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(cd -- "${TEST_SCRIPT_DIR}/../.." && pwd)}"
# Extracted production functions resolve repository helpers through SCRIPT_DIR.
SCRIPT_DIR="$(cd -- "${REPO_ROOT}" && pwd)"
fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/dg_igr_external_artifacts.XXXXXX")"
trap 'rm -rf -- "${fixture_root}"' EXIT

fail() { echo "spack-external-artifacts: FAIL: $*" >&2; exit 1; }
for function_name in cache_value config_value canonical_existing_path \
    require_exact_mfem_directory require_exact_mfem_file \
    hash_mfem_identity_artifacts spack_identity spack_artifact_root umpire_package_config \
    spack_dependency_hash is_cmake_true is_cmake_false validate_reused_mfem; do
  body="$(awk -v signature="${function_name}() {" '
    $0 == signature { capture = 1 }
    capture { print }
    capture && /^}/ { exit }
  ' "${REPO_ROOT}/build.sh")"
  [[ -n "${body}" ]] || fail "missing production function ${function_name}"
  eval "${body}"
done
validate_mfem_source_checkout() { MFEM_SOURCE_ROOT="$1"; }

declare -A package_hash package_prefix artifacts
packages=(openmpi hypre metis cuda umpire caliper)
artifacts[openmpi]='bin/mpicc bin/mpicxx'
artifacts[hypre]='include/HYPRE.h include/HYPRE_config.h lib/libHYPRE.so'
artifacts[metis]='include/metis.h lib/libmetis.a'
artifacts[cuda]='bin/nvcc'
artifacts[umpire]='lib/cmake/umpire/umpire-config.cmake'
artifacts[caliper]='share/cmake/caliper/caliper-config.cmake'
for index in "${!packages[@]}"; do
  package="${packages[index]}"
  printf -v "package_hash[${package}]" '%032d' "$((index + 1))"
  package_prefix[${package}]="${fixture_root}/prefix/${package}"
done

# The fake command implements only the exact identity queries made by the
# production validator, including hash-qualified external-status lookup.
spack() {
  [[ "$1" == -e && "$2" == "${SPACK_ENV}" ]] || return 99
  shift 2
  local action="$1" format="" spec package candidate status
  shift
  if [[ "${action}" == location ]]; then
    [[ "$1" == -i ]] || return 99
    spec="$2"
  else
    [[ "${action}" == find && "$1" == --format ]] || return 99
    format="$2"; spec="$3"
  fi
  package="${spec%%[+~@ ]*}"
  if [[ "${spec}" == /* ]]; then
    package=""
    for candidate in "${packages[@]}"; do
      [[ "${spec}" != "/${package_hash[${candidate}]}" ]] || package="${candidate}"
    done
  fi
  [[ -n "${package}" && -n "${package_hash[${package}]:-}" ]] || return 99
  case "${action}:${format}" in
    location:) printf '%s\n' "${package_prefix[${package}]}" ;;
    'find:{hash}|{compiler.name}@{compiler.version}')
      printf '%s|gcc@13.3.0\n' "${package_hash[${package}]}" ;;
    'find:{hash}|{external}')
      [[ "${spec}" == "/${package_hash[${package}]}" ]] || return 99
      status=False
      [[ " ${external_packages} " != *" ${package} "* ]] || status=True
      case "${bad_status}" in
        missing) return 0 ;;
        malformed) status=unknown ;;
        wrong-hash) printf '%032d|True\n' 99; return 0 ;;
        duplicate) printf '%s|%s\n' "${package_hash[${package}]}" "${status}" ;;
        failure) return 1 ;;
      esac
      printf '%s|%s\n' "${package_hash[${package}]}" "${status}" ;;
    'find:{^mpi.hash}') printf '%s\n' "${package_hash[openmpi]}" ;;
    'find:{^cuda.hash}') printf '%s\n' "${package_hash[cuda]}" ;;
    'find:{^umpire.hash}') printf '%s\n' "${package_hash[umpire]}" ;;
    *) return 99 ;;
  esac
}

prepare_fixture() {
  external_packages="$1"; MODE="${2:-gpu}"; CALIPER="${3:-0}"
  bad_status=""
  SPACK_ENV="${fixture_root}/env"
  MFEM_BUILD="${fixture_root}/build"
  MFEM_INSTALL="${fixture_root}/install"
  MFEM_IDENTITY_STAMP_NAME=dg-igr-mfem-identity-v1.txt
  CMAKE_BUILD_TYPE=Release
  SPACK_CUDA_ARCH=80; MFEM_CUDA_ARCHS=80; DEPENDENCY_CUDA_ARCH=80
  local package artifact view="${SPACK_ENV}/.spack-env/view" cuda_on=ON caliper_on=OFF
  local hypre_library="" hypre_includes="${package_prefix[hypre]}/include"
  local fixture_cmake_version="${4:-3.31.11}" fixture_cmake_minor
  fixture_cmake_minor="${fixture_cmake_version#*.}"
  [[ "${MODE}" == gpu ]] || cuda_on=OFF
  ((CALIPER == 0)) || caliper_on=ON
  rm -rf -- "${fixture_root}/prefix" "${SPACK_ENV}" "${MFEM_BUILD}" "${MFEM_INSTALL}"
  mkdir -p "${view}" "${fixture_root}/source" \
    "${MFEM_BUILD}/CMakeFiles/${fixture_cmake_version}" "${MFEM_INSTALL}/lib/cmake/mfem" \
    "${MFEM_INSTALL}/include"
  touch "${SPACK_ENV}/spack.yaml" "${SPACK_ENV}/spack.lock" \
    "${MFEM_BUILD}/cmake_install.cmake" "${MFEM_BUILD}/CMakeFiles/cmake.check_cache" \
    "${MFEM_BUILD}/install_manifest.txt" "${MFEM_BUILD}/CMakeFiles/${fixture_cmake_version}/CMakeCXXCompiler.cmake" \
    "${MFEM_INSTALL}/lib/libmfem.a" "${MFEM_INSTALL}/lib/cmake/mfem/MFEMTargets.cmake" \
    "${MFEM_INSTALL}/lib/cmake/mfem/MFEMTargets-release.cmake"
  for package in "${packages[@]}"; do
    for artifact in ${artifacts[${package}]}; do
      mkdir -p "$(dirname -- "${package_prefix[${package}]}/${artifact}")"
      touch "${package_prefix[${package}]}/${artifact}"
      if [[ " ${external_packages} " != *" ${package} "* ]]; then
        mkdir -p "$(dirname -- "${view}/${artifact}")"
        ln -s "${package_prefix[${package}]}/${artifact}" "${view}/${artifact}"
      fi
    done
  done
  for artifact in ${artifacts[hypre]}; do
    [[ "${artifact}" != */libHYPRE.* ]] || hypre_library="${package_prefix[hypre]}/${artifact}"
  done
  [[ -n "${hypre_library}" ]] || fail "fixture lacks a HYPRE library"
  if [[ "${MODE}" == gpu ]]; then
    mkdir -p "${package_prefix[umpire]}/include" "${package_prefix[cuda]}/include"
    hypre_includes+=";${package_prefix[umpire]}/include;${package_prefix[cuda]}/include"
  fi
  printf '#!/bin/sh\nprintf "13.3.0\\n"\n' >"${fixture_root}/compiler"
  chmod +x "${fixture_root}/compiler"
  for artifact in mpicc mpicxx; do
    printf '#!/bin/sh\nprintf "%%s\\n" "%s"\n' "${fixture_root}/compiler" \
      >"${package_prefix[openmpi]}/bin/${artifact}"
    chmod +x "${package_prefix[openmpi]}/bin/${artifact}"
  done
  cat >"${MFEM_BUILD}/CMakeCache.txt" <<CACHE
CMAKE_HOME_DIRECTORY:PATH=${fixture_root}/source
CMAKE_INSTALL_PREFIX:PATH=${MFEM_INSTALL}
CMAKE_BUILD_TYPE:STRING=Release
CMAKE_CACHE_MAJOR_VERSION:INTERNAL=${fixture_cmake_version%%.*}
CMAKE_CACHE_MINOR_VERSION:INTERNAL=${fixture_cmake_minor%%.*}
CMAKE_CACHE_PATCH_VERSION:INTERNAL=${fixture_cmake_version##*.}
MFEM_USE_MPI:BOOL=ON
MFEM_USE_OPENMP:BOOL=OFF
MFEM_USE_METIS:BOOL=ON
MFEM_USE_RAJA:BOOL=OFF
MFEM_USE_CALIPER:BOOL=${caliper_on}
MFEM_USE_CUDA:BOOL=${cuda_on}
MFEM_USE_UMPIRE:BOOL=${cuda_on}
CMAKE_PREFIX_PATH:PATH=${view}
HYPRE_DIR:PATH=${package_prefix[hypre]}
HYPRE_INCLUDE_DIR:PATH=${package_prefix[hypre]}/include
HYPRE_INCLUDE_DIRS:STRING=${hypre_includes}
HYPRE_LIBRARY:FILEPATH=${hypre_library}
HYPRE_LIBRARIES:STRING=${hypre_library}
METIS_DIR:PATH=${package_prefix[metis]}
CMAKE_CXX_COMPILER:FILEPATH=${package_prefix[openmpi]}/bin/mpicxx
MPI_CXX_COMPILER:FILEPATH=${package_prefix[openmpi]}/bin/mpicxx
CMAKE_CUDA_ARCHITECTURES:STRING=80
CUDAToolkit_ROOT:PATH=${package_prefix[cuda]}
CMAKE_CUDA_COMPILER:FILEPATH=${package_prefix[cuda]}/bin/nvcc
UMPIRE_DIR:PATH=${package_prefix[umpire]}
CACHE
  cat >"${MFEM_INSTALL}/lib/cmake/mfem/MFEMConfig.cmake" <<CONFIG
set(MFEM_USE_MPI ON)
set(MFEM_USE_OPENMP OFF)
set(MFEM_USE_METIS ON)
set(MFEM_USE_RAJA OFF)
set(MFEM_USE_CALIPER ${caliper_on})
set(MFEM_USE_CUDA ${cuda_on})
set(MFEM_USE_UMPIRE ${cuda_on})
set(MFEM_CXX_COMPILER "${package_prefix[openmpi]}/bin/mpicxx")
set(MFEM_INCLUDE_DIRS "\${PACKAGE_PREFIX_DIR}/include;${hypre_includes}")
CONFIG
  cat >"${MFEM_INSTALL}/lib/cmake/mfem/MFEMTargets.cmake" <<TARGETS
set_target_properties(mfem PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "\${_IMPORT_PREFIX}/include;${hypre_includes}"
  INTERFACE_LINK_LIBRARIES "${hypre_library};MPI::MPI_CXX"
)
TARGETS
  cat >"${MFEM_BUILD}/CMakeFiles/${fixture_cmake_version}/CMakeCXXCompiler.cmake" <<CONFIG
set(CMAKE_CXX_COMPILER "${package_prefix[openmpi]}/bin/mpicxx")
set(CMAKE_CXX_COMPILER_ID "GNU")
set(CMAKE_CXX_COMPILER_VERSION "13.3.0")
CONFIG
  cat >"${MFEM_BUILD}/CMakeFiles/${fixture_cmake_version}/CMakeCUDACompiler.cmake" <<CONFIG
set(CMAKE_CUDA_COMPILER "${package_prefix[cuda]}/bin/nvcc")
set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "${package_prefix[cuda]}")
set(CMAKE_CUDA_HOST_COMPILER "${package_prefix[openmpi]}/bin/mpicxx")
set(CMAKE_CUDA_HOST_COMPILER_ID "GNU")
set(CMAKE_CUDA_HOST_COMPILER_VERSION "13.3.0")
CONFIG
}

check_validation() {
  local expected="$1" label="$2" output status
  set +e
  output="$(set -e; validate_reused_mfem "${fixture_root}/source" create local 2>&1)"
  status=$?
  set -e
  [[ "${status}" == "${expected}" ]] || \
    fail "${label}: expected ${expected}, got ${status}: ${output}"
}

prepare_fixture ''
check_validation 0 installed-view
prepare_fixture 'openmpi metis cuda'
check_validation 0 mixed-site-externals
prepare_fixture "${packages[*]}"
check_validation 0 all-external
prepare_fixture "${packages[*]}" cpu 1
check_validation 0 external-caliper

# CMake <3.31 did not emit CUDA host ID/version. The fallback must bind the
# same wrapper to adjacent C++ metadata from the cache's exact CMake version.
prepare_legacy_cuda_metadata() {
  prepare_fixture "${1:-}" gpu 0 3.30.2
  legacy_cuda_metadata="${MFEM_BUILD}/CMakeFiles/3.30.2/CMakeCUDACompiler.cmake"
  legacy_cxx_metadata="${MFEM_BUILD}/CMakeFiles/3.30.2/CMakeCXXCompiler.cmake"
  sed -i -E '/^set\(CMAKE_CUDA_HOST_COMPILER_(ID|VERSION) /d' "${legacy_cuda_metadata}"
}
for legacy_externals in '' "${packages[*]}"; do
  prepare_legacy_cuda_metadata "${legacy_externals}"
  check_validation 0 "legacy-cuda-host-${legacy_externals:-installed}"
done
for missing_field in ID VERSION; do
  prepare_fixture '' gpu 0 3.30.2
  sed -i "/^set(CMAKE_CUDA_HOST_COMPILER_${missing_field} /d" \
    "${MFEM_BUILD}/CMakeFiles/3.30.2/CMakeCUDACompiler.cmake"
  check_validation 3 "partial-cuda-host-${missing_field}"
done
for invalid_metadata in path id version missing-id missing-version nonadjacent wrong-cmake-version; do
  prepare_legacy_cuda_metadata
  case "${invalid_metadata}" in
    path) printf 'set(CMAKE_CXX_COMPILER "%s")\n' "${fixture_root}/compiler" >>"${legacy_cxx_metadata}" ;;
    id) printf 'set(CMAKE_CXX_COMPILER_ID "Clang")\n' >>"${legacy_cxx_metadata}" ;;
    version) printf 'set(CMAKE_CXX_COMPILER_VERSION "12.3.0")\n' >>"${legacy_cxx_metadata}" ;;
    missing-id) sed -i '/^set(CMAKE_CXX_COMPILER_ID /d' "${legacy_cxx_metadata}" ;;
    missing-version) sed -i '/^set(CMAKE_CXX_COMPILER_VERSION /d' "${legacy_cxx_metadata}" ;;
    nonadjacent)
      mkdir "${MFEM_BUILD}/CMakeFiles/other"
      mv "${legacy_cxx_metadata}" "${MFEM_BUILD}/CMakeFiles/other/CMakeCXXCompiler.cmake"
      ;;
    wrong-cmake-version) printf 'CMAKE_CACHE_PATCH_VERSION:INTERNAL=1\n' >>"${MFEM_BUILD}/CMakeCache.txt" ;;
  esac
  check_validation 3 "legacy-cuda-host-${invalid_metadata}"
done
prepare_fixture ''
sed -i -E '/^set\(CMAKE_CUDA_HOST_COMPILER_(ID|VERSION) /d' \
  "${MFEM_BUILD}/CMakeFiles/3.31.11/CMakeCUDACompiler.cmake"
check_validation 3 modern-cuda-host-metadata-missing

# CUDA/Umpire HYPRE may be static. Both supported library directories retain
# the same concrete-prefix containment check for installed and external specs.
for library_dir in lib lib64; do
  artifacts[hypre]="include/HYPRE.h include/HYPRE_config.h ${library_dir}/libHYPRE.a"
  for hypre_external in '' hypre; do
    prepare_fixture "${hypre_external}"
    check_validation 0 "static-hypre-${library_dir}-${hypre_external:-installed}"
    rm -- "${package_prefix[hypre]}/${library_dir}/libHYPRE.a"
    ln -s "${package_prefix[metis]}/lib/libmetis.a" \
      "${package_prefix[hypre]}/${library_dir}/libHYPRE.a"
    check_validation 3 "redirected-static-hypre-${library_dir}-${hypre_external:-installed}"
  done
done
artifacts[hypre]='include/HYPRE.h include/HYPRE_config.h lib/libHYPRE.so'

# Umpire may install its config under lib or lib64. Both the validator and
# application configuration use this resolver and require exact-prefix binding.
for library_dir in lib lib64; do
  artifacts[umpire]="${library_dir}/cmake/umpire/umpire-config.cmake"
  for umpire_external in '' umpire; do
    prepare_fixture "${umpire_external}"
    check_validation 0 "umpire-${library_dir}-${umpire_external:-installed}"
    resolved_config="$(umpire_package_config \
      "${package_prefix[umpire]}" "${package_prefix[umpire]}")"
    [[ "${resolved_config}" == "${package_prefix[umpire]}/${artifacts[umpire]}" ]] || \
      fail "application Umpire config directory differs from the validated layout"
    rm -- "${package_prefix[umpire]}/${artifacts[umpire]}"
    ln -s "${package_prefix[caliper]}/${artifacts[caliper]}" \
      "${package_prefix[umpire]}/${artifacts[umpire]}"
    check_validation 3 "redirected-umpire-${library_dir}-${umpire_external:-installed}"
    if umpire_package_config "${package_prefix[umpire]}" \
        "${package_prefix[umpire]}" >/dev/null 2>&1; then
      fail "application resolver admitted a redirected Umpire config"
    fi
  done
done
artifacts[umpire]='lib/cmake/umpire/umpire-config.cmake'

# Nonexternal dependencies still require their view entries, even when their
# concrete prefix contains a usable artifact. No missing-entry fallback.
for package in openmpi hypre metis cuda umpire; do
  prepare_fixture ''
  read -r artifact _ <<<"${artifacts[${package}]}"
  rm -- "${SPACK_ENV}/.spack-env/view/${artifact}"
  check_validation 3 "missing-installed-view-${package}"
done
for package in openmpi hypre metis cuda umpire; do
  prepare_fixture "${packages[*]}"
  read -r artifact _ <<<"${artifacts[${package}]}"
  rm -- "${package_prefix[${package}]}/${artifact}"
  check_validation 3 "missing-external-artifact-${package}"
done

for bad in missing malformed wrong-hash duplicate failure; do
  prepare_fixture "${packages[*]}"
  bad_status="${bad}"
  check_validation 3 "external-status-${bad}"
done

# Canonical prefix containment and compiler/cache identity remain enforced.
# A correct HYPRE_DIR cannot admit stale singular/plural discovery values or
# stale installed interfaces left over from an earlier dependency graph.
for binding in HYPRE_INCLUDE_DIR HYPRE_INCLUDE_DIRS HYPRE_LIBRARY HYPRE_LIBRARIES; do
  prepare_fixture "${packages[*]}"
  mkdir -p "${fixture_root}/stale-hypre/include" "${fixture_root}/stale-hypre/lib"
  touch "${fixture_root}/stale-hypre/include/HYPRE.h" \
    "${fixture_root}/stale-hypre/include/HYPRE_config.h" \
    "${fixture_root}/stale-hypre/lib/libHYPRE.so"
  if [[ "${binding}" == *INCLUDE* ]]; then
    stale_value="${fixture_root}/stale-hypre/include"
  else
    stale_value="${fixture_root}/stale-hypre/lib/libHYPRE.so"
  fi
  printf '%s:STRING=%s\n' "${binding}" "${stale_value}" >>"${MFEM_BUILD}/CMakeCache.txt"
  check_validation 3 "stale-hypre-discovery-${binding}"
done
for export_file in MFEMConfig.cmake MFEMTargets.cmake; do
  prepare_fixture "${packages[*]}"
  mkdir -p "${fixture_root}/stale-hypre/include" "${fixture_root}/stale-hypre/lib"
  touch "${fixture_root}/stale-hypre/include/HYPRE.h" \
    "${fixture_root}/stale-hypre/include/HYPRE_config.h" \
    "${fixture_root}/stale-hypre/lib/libHYPRE.so"
  sed -i "s|${package_prefix[hypre]}|${fixture_root}/stale-hypre|g" \
    "${MFEM_INSTALL}/lib/cmake/mfem/${export_file}"
  check_validation 3 "stale-installed-hypre-${export_file}"
done
prepare_fixture "${packages[*]}"
rm -- "${package_prefix[metis]}/include/metis.h"
ln -s "${fixture_root}/compiler" "${package_prefix[metis]}/include/metis.h"
check_validation 3 redirected-external-header
prepare_fixture ''
rm -- "${SPACK_ENV}/.spack-env/view/lib/libHYPRE.so"
ln -s "${package_prefix[metis]}/lib/libmetis.a" "${SPACK_ENV}/.spack-env/view/lib/libHYPRE.so"
check_validation 3 redirected-installed-library
prepare_fixture "${packages[*]}"
printf 'CMAKE_CXX_COMPILER:FILEPATH=%s\n' "${fixture_root}/compiler" >>"${MFEM_BUILD}/CMakeCache.txt"
check_validation 3 changed-cached-compiler
prepare_fixture "${packages[*]}"
mv -- "${MFEM_BUILD}/CMakeCache.txt" "${fixture_root}/cache-copy"
ln -s "${fixture_root}/cache-copy" "${MFEM_BUILD}/CMakeCache.txt"
check_validation 3 symlinked-mfem-cache

echo 'spack-external-artifacts: PASS'
