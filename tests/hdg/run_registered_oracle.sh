#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 serial|mpi RANKS EXECUTABLE RUN_ROOT [MPIEXEC NUMPROC_FLAG]" >&2
  exit 64
fi

mode="$1"
ranks="$2"
executable="$3"
run_root="$4"

mkdir -p "${run_root}"
run_dir="$(mktemp -d "${run_root}/${mode}-${ranks}.XXXXXX")"
output_dir="${run_dir}/output"

if [[ "${mode}" == "serial" ]]; then
  if [[ "${ranks}" != "1" || $# -ne 4 ]]; then
    echo "serial mode requires exactly one rank and four arguments" >&2
    exit 64
  fi
  exec "${executable}" --mode serial "${output_dir}"
fi

if [[ "${mode}" != "mpi" || $# -ne 6 ]]; then
  echo "mpi mode requires MPIEXEC and NUMPROC_FLAG" >&2
  exit 64
fi

mpiexec="$5"
numproc_flag="$6"
exec 9>/tmp/dg_igr-sm120.lock
if ! flock -n 9; then
  echo "HDG_ORACLE_MPI_LEASE BUSY /tmp/dg_igr-sm120.lock"
  exit 75
fi
echo "HDG_ORACLE_MPI_LEASE ACQUIRED /tmp/dg_igr-sm120.lock"
"${mpiexec}" "${numproc_flag}" "${ranks}" \
  "${executable}" --mode mpi "${output_dir}"
echo "HDG_ORACLE_MPI_LEASE RELEASED /tmp/dg_igr-sm120.lock"
