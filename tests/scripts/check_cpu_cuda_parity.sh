#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <executable> <dg|dgsem> [extra args...]" >&2
  exit 2
fi

exe="$1"
backend="$2"
shift 2

if [[ ! -x "${exe}" ]]; then
  echo "parity-check: executable not found or not executable: ${exe}" >&2
  exit 2
fi

if [[ "${backend}" != "dg" && "${backend}" != "dgsem" ]]; then
  echo "parity-check: backend must be 'dg' or 'dgsem', got '${backend}'" >&2
  exit 2
fi

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/dg_igr_parity.XXXXXX")"

preserve_failure_logs() {
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    local log_root="${DG_IGR_PARITY_LOG_ROOT:-${PWD}/parity-logs}"
    local stamp destination
    stamp="$(date +%Y%m%d-%H%M%S)-${backend}-$$"
    destination="${log_root}/${stamp}"
    mkdir -p "${destination}"
    cp -a "${tmpdir}/." "${destination}/"
    {
      echo "backend=${backend}"
      echo "executable=${exe}"
      echo "exit_code=${rc}"
      echo "pwd=${PWD}"
      echo "PATH=${PATH}"
      echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    } >"${destination}/environment.txt"
    env | grep -E '^(SPACK|PATH|LD_LIBRARY_PATH|CUDA|OMPI|PMIX|DG_IGR)' \
      | sort >"${destination}/runtime-environment.txt" || true
    ldd "${exe}" >"${destination}/ldd.txt" 2>&1 || true
    echo "parity-check: retained logs in ${destination}" >&2
    echo "parity-check: resolved runtime libraries:" >&2
    grep -Ei 'mpi|hypre|umpire|cuda|mfem|not found' \
      "${destination}/ldd.txt" >&2 || true
  fi
  rm -rf "${tmpdir}"
  exit "${rc}"
}
trap preserve_failure_logs EXIT

cpu_log="${tmpdir}/cpu.log"
cuda_log="${tmpdir}/cuda.log"

common_args=(
  -tf 0.02
  -o 3
  -rs 0
  -rp 0
  -is H1
  -ig H1
  -iprint 0
  -no-vtk
  -no-glvis
  --dg-backend "${backend}"
)
if [[ $# -gt 0 ]]; then
  common_args+=("$@")
fi

native_mass_inv="${DG_IGR_DG_NATIVE_MASS_INV:-0}"
packed_invmass_device="${DG_IGR_DG_PACKED_INVMASS_DEVICE:-1}"

extract_solution_error() {
  local log_file="$1"
  awk '/Solution error:/ {v=$3} END {if (v == "") exit 1; print v}' "${log_file}"
}

extract_sigma_min() {
  local log_file="$1"
  awk '/Sigma min\/max:/ {v=$3} END {if (v == "") exit 1; print v}' "${log_file}"
}

extract_sigma_max() {
  local log_file="$1"
  awk '/Sigma min\/max:/ {v=$5} END {if (v == "") exit 1; print v}' "${log_file}"
}

extract_total_igr_pct() {
  local log_file="$1"
  awk '
    /\[igr-linear\] event=run-summary/ {
      for (i = 1; i <= NF; ++i) {
        if ($i ~ /^total_igr_pct=/) {
          split($i, a, "=");
          v = a[2];
        }
      }
    }
    END { if (v == "") v = "0.0"; print v }
  ' "${log_file}"
}

require_line() {
  local pattern="$1"
  local log_file="$2"
  if ! grep -Eq "${pattern}" "${log_file}"; then
    echo "parity-check: missing pattern '${pattern}' in ${log_file}" >&2
    exit 1
  fi
}

echo "parity-check: running CPU backend (${backend})"
DG_IGR_DG_NATIVE_MASS_INV="${native_mass_inv}" \
DG_IGR_DG_PACKED_INVMASS_DEVICE="${packed_invmass_device}" \
"${exe}" --device cpu "${common_args[@]}" >"${cpu_log}" 2>&1

echo "parity-check: running CUDA backend (${backend})"
set +e
DG_IGR_DG_NATIVE_MASS_INV="${native_mass_inv}" \
DG_IGR_DG_PACKED_INVMASS_DEVICE="${packed_invmass_device}" \
"${exe}" --device cuda "${common_args[@]}" >"${cuda_log}" 2>&1
cuda_rc=$?
set -e

if [[ ${cuda_rc} -ne 0 ]]; then
  if grep -Eq "cudaGetDeviceCount|CUDA error|No CUDA|no CUDA|without CUDA support" "${cuda_log}"; then
    if [[ "${DG_IGR_REQUIRE_CUDA_PARITY:-0}" == "1" ]]; then
      echo "parity-check: CUDA run failed and DG_IGR_REQUIRE_CUDA_PARITY=1." >&2
      tail -n 60 "${cuda_log}" >&2 || true
      exit 1
    fi
    echo "SKIP: CUDA backend unavailable in current runtime."
    exit 0
  fi
  echo "parity-check: CUDA run failed with unexpected error." >&2
  tail -n 80 "${cuda_log}" >&2 || true
  exit ${cuda_rc}
fi

require_line "\\[device\\] requested=cpu" "${cpu_log}"
require_line "done," "${cpu_log}"
require_line "\\[device\\] requested=cuda" "${cuda_log}"
require_line "Device configuration: cuda,cpu" "${cuda_log}"
require_line "done," "${cuda_log}"
require_line "Solution error:" "${cpu_log}"
require_line "Solution error:" "${cuda_log}"
require_line "Sigma min/max:" "${cpu_log}"
require_line "Sigma min/max:" "${cuda_log}"

cpu_solution="$(extract_solution_error "${cpu_log}")"
cuda_solution="$(extract_solution_error "${cuda_log}")"
cpu_sigma_min="$(extract_sigma_min "${cpu_log}")"
cuda_sigma_min="$(extract_sigma_min "${cuda_log}")"
cpu_sigma_max="$(extract_sigma_max "${cpu_log}")"
cuda_sigma_max="$(extract_sigma_max "${cuda_log}")"
cpu_igr_pct="$(extract_total_igr_pct "${cpu_log}")"
cuda_igr_pct="$(extract_total_igr_pct "${cuda_log}")"

python3 - "${cpu_solution}" "${cuda_solution}" "${cpu_sigma_min}" "${cuda_sigma_min}" "${cpu_sigma_max}" "${cuda_sigma_max}" <<'PY'
import os
import sys

cpu_solution = float(sys.argv[1])
cuda_solution = float(sys.argv[2])
cpu_sigma_min = float(sys.argv[3])
cuda_sigma_min = float(sys.argv[4])
cpu_sigma_max = float(sys.argv[5])
cuda_sigma_max = float(sys.argv[6])

sol_rtol = float(os.environ.get("DG_IGR_PARITY_SOL_RTL", "1.0e-5"))
sigma_rtol = float(os.environ.get("DG_IGR_PARITY_SIGMA_RTL", "1.0e-5"))
sigma_atol = float(os.environ.get("DG_IGR_PARITY_SIGMA_ATOL", "1.0e-6"))


def close(a: float, b: float, rtol: float, atol: float) -> bool:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= max(atol, rtol * scale)


if not close(cpu_solution, cuda_solution, sol_rtol, 1.0e-10):
    print(
        f"parity-check: Solution error mismatch cpu={cpu_solution:.12e} "
        f"cuda={cuda_solution:.12e} (rtol={sol_rtol})",
        file=sys.stderr,
    )
    sys.exit(1)

if not close(cpu_sigma_min, cuda_sigma_min, sigma_rtol, sigma_atol):
    print(
        f"parity-check: sigma_min mismatch cpu={cpu_sigma_min:.12e} "
        f"cuda={cuda_sigma_min:.12e} (rtol={sigma_rtol}, atol={sigma_atol})",
        file=sys.stderr,
    )
    sys.exit(1)

if not close(cpu_sigma_max, cuda_sigma_max, sigma_rtol, sigma_atol):
    print(
        f"parity-check: sigma_max mismatch cpu={cpu_sigma_max:.12e} "
        f"cuda={cuda_sigma_max:.12e} (rtol={sigma_rtol}, atol={sigma_atol})",
        file=sys.stderr,
    )
    sys.exit(1)
PY

echo "parity-check: PASS backend=${backend} solution(cpu=${cpu_solution},cuda=${cuda_solution}) sigma_min(cpu=${cpu_sigma_min},cuda=${cuda_sigma_min}) sigma_max(cpu=${cpu_sigma_max},cuda=${cuda_sigma_max}) igr_pct(cpu=${cpu_igr_pct},cuda=${cuda_igr_pct})"
