#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 UNIQUE_RUN_DIRECTORY" >&2
  exit 64
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_dir="$1"
build_dir="${run_dir}/build-cpu"
output_dir="${run_dir}/output-cpu"
log_dir="${run_dir}/logs-cpu"
mfem_dir="/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/install-cpu/lib/cmake/mfem"
cmake_bin="/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake"
mpi_bin="/home/jarias9/spack/opt/spack/linux-x86_64_v4/openmpi-5.0.10-belap7umgzjinzggojo3kmnw5le2fmez/bin"

mkdir -p "${build_dir}" "${output_dir}" "${log_dir}"

"${cmake_bin}" -S "${script_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${mpi_bin}/mpicxx" \
  -DMPI_CXX_COMPILER="${mpi_bin}/mpicxx" \
  -DMFEM_DIR="${mfem_dir}" \
  2>&1 | tee "${log_dir}/configure.log"
"${cmake_bin}" --build "${build_dir}" --parallel 2 \
  2>&1 | tee "${log_dir}/build.log"

"${build_dir}/hdg_oracle" --mode serial "${output_dir}" \
  2>&1 | tee "${log_dir}/serial.log"

lock_file="/tmp/dg_igr-sm120.lock"
exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "HDG_ORACLE_MPI_LEASE BUSY ${lock_file}" | tee "${log_dir}/mpi_lease.log"
  exit 75
fi
echo "HDG_ORACLE_MPI_LEASE ACQUIRED ${lock_file}" | tee "${log_dir}/mpi_lease.log"

"${mpi_bin}/mpiexec" -n 1 "${build_dir}/hdg_oracle" --mode mpi "${output_dir}" \
  2>&1 | tee "${log_dir}/mpi_1.log"
"${mpi_bin}/mpiexec" -n 2 "${build_dir}/hdg_oracle" --mode mpi "${output_dir}" \
  2>&1 | tee "${log_dir}/mpi_2.log"

echo "HDG_ORACLE_MPI_LEASE RELEASED ${lock_file}" | tee -a "${log_dir}/mpi_lease.log"
