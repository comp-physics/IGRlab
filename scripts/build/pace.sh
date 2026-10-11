#!/usr/bin/env bash
# PACE environment and Slurm front end; all builds use the canonical driver.
set -euo pipefail
pace_script_dir="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
pace_source="$(cd -- "${pace_script_dir}/../.." && pwd -P)"
pace_config="${pace_source}/cmake/sites/pace.json"

pace_fail() { echo "ERROR: PACE: $*" >&2; exit 2; }

pace_prepare_site_directory() {
  local target="$1" component
  for component in "$pace_dependency_root/site" "$pace_dependency_root/site/pace" "$target"; do
    [[ ! -L $component && (! -e $component || -d $component) ]] || \
      pace_fail "site directory is symlinked or not a directory: $component"
  done
  mkdir -p -- "$target"
}

pace_prepare_dependency_lock() {
  pace_prepare_site_directory "$pace_dependency_root/site/pace/locks"
  local lock_path="$pace_dependency_root/site/pace/locks/$pace_profile.lock"
  [[ ! -L $lock_path && (! -e $lock_path || -f $lock_path) ]] || \
    pace_fail "dependency lock is symlinked or not a regular file: $lock_path"
  exec {pace_environment_lock}>>"$lock_path"
  flock "$pace_environment_lock"
  export IGRLAB_PACE_ENV_LOCK_FD="$pace_environment_lock"
}

# sbatch executes the tracked helper, with the original argv kept as JSON.
if [[ ${1:-} == --worker ]]; then
  [[ $# == 3 && ($2 == build || $2 == validation) ]] || pace_fail "invalid worker arguments"
  pace_stage=$2
  pace_request=$3
  # Slurm runs a spool copy, so BASH_SOURCE cannot locate the checkout here.
  pace_source=$(python3 - "$pace_request" <<'PY'
import json, sys
from pathlib import Path
source = Path(json.load(open(sys.argv[1]))['source'])
if not source.is_absolute() or source != source.resolve(strict=True):
    raise SystemExit('invalid checkout path in PACE build request')
print(source)
PY
  )
  [[ -f $pace_source/build.sh && -f $pace_source/scripts/build/pace.sh ]] || pace_fail "requested checkout no longer exists"
  pace_worker_tasks=1
  [[ $pace_stage == build ]] || pace_worker_tasks=4
  [[ ${SLURM_JOB_ACCOUNT:-} == gts-sbryngelson3 && ${SLURM_JOB_QOS:-} == embers &&
     ${SLURM_NTASKS:-} == "$pace_worker_tasks" ]] || pace_fail "worker requires the free $pace_worker_tasks-task embers allocation"
  mapfile -d '' -t pace_original < <(python3 - "$pace_request" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for arg in r['argv']:
    sys.stdout.buffer.write(arg.encode() + b'\0')
PY
  )
  ((${#pace_original[@]})) || pace_fail "empty build request"
  pace_worker_args=()
  while ((${#pace_original[@]})); do
    pace_arg=${pace_original[0]}; pace_original=("${pace_original[@]:1}")
    case "$pace_arg" in
      --no-tests) [[ $pace_stage == build ]] || pace_worker_args+=("$pace_arg") ;;
      --no-clean|--clean|--no-deps)
        [[ $pace_stage == validation ]] || pace_worker_args+=("$pace_arg") ;;
      --smoke) [[ $pace_stage == build ]] || pace_worker_args+=("$pace_arg") ;;
      --test-regex)
        ((${#pace_original[@]})) || pace_fail "missing test regex in request"
        [[ $pace_stage == build ]] || pace_worker_args+=("$pace_arg" "${pace_original[0]}")
        pace_original=("${pace_original[@]:1}") ;;
      *) pace_worker_args+=("$pace_arg") ;;
    esac
  done
  if [[ $pace_stage == build ]]; then
    pace_worker_args+=(--no-tests)
  else
    pace_worker_args+=(--no-clean --no-deps)
  fi
  exec "${pace_source}/build.sh" "${pace_worker_args[@]}"
fi

pace_args=()
pace_submit=0 pace_plan=0 pace_tests=1 pace_smoke=0 pace_mode=cpu pace_arch="" pace_clean_set=0 pace_help=0
pace_root="" pace_dependency_root="" pace_spack_root=${IGRLAB_PACE_SPACK_ROOT:-}
pace_jobs=${IGRLAB_BUILD_JOBS:-4} pace_mfem_jobs=""
while (($#)); do
  case "$1" in
    --site) [[ $# -ge 2 && $2 == pace ]] || pace_fail "supported site is pace"; shift ;;
    --submit) pace_submit=1 ;;
    --site-root) [[ $# -ge 2 ]] || pace_fail "--site-root requires a path"; pace_dependency_root=$2; shift ;;
    --spack-root) [[ $# -ge 2 ]] || pace_fail "--spack-root requires a path"; pace_spack_root=$2; shift ;;
    --output-root) [[ $# -ge 2 ]] || pace_fail "--output-root requires a path"; pace_root=$2; pace_args+=("$1" "$2"); shift ;;
    --a100|--gpu) pace_mode=gpu; pace_args+=("$1") ;;
    --cpu|--cpu-only) pace_mode=cpu; pace_args+=("$1") ;;
    --cuda-arch) [[ $# -ge 2 ]] || pace_fail "--cuda-arch requires a value"; pace_arch=$2; pace_args+=("$1" "$2"); shift ;;
    -j|--jobs) [[ $# -ge 2 ]] || pace_fail "$1 requires a value"; pace_jobs=$2; pace_args+=("$1" "$2"); shift ;;
    --mfem-jobs) [[ $# -ge 2 ]] || pace_fail "$1 requires a value"; pace_mfem_jobs=$2; pace_args+=("$1" "$2"); shift ;;
    --print-plan) pace_plan=1; pace_args+=("$1") ;;
    --no-tests) pace_tests=0; pace_args+=("$1") ;;
    --smoke) pace_smoke=1; pace_args+=("$1") ;;
    --clean|--no-clean) pace_clean_set=1; pace_args+=("$1") ;;
    --caliper) pace_fail "the validated site profiles are cpu and cuda-sm80; Caliper has no PACE profile yet" ;;
    --help|-h) pace_help=1; pace_args+=("$1") ;;
    *) pace_args+=("$1") ;;
  esac
  shift
done
[[ $pace_help == 0 ]] || exec "${pace_source}/build.sh" --help
[[ -n $pace_root && $pace_root == /* ]] || pace_fail "provide an absolute --output-root on scratch"
pace_root=$(realpath -m -- "$pace_root")
[[ -z $pace_dependency_root || $pace_dependency_root == /* ]] || pace_fail "--site-root must be absolute"
pace_dependency_root=$(realpath -m -- "${pace_dependency_root:-$pace_root}")
[[ $pace_clean_set == 1 ]] || pace_args+=(--no-clean)
[[ $pace_jobs =~ ^[1-9][0-9]*$ && ${pace_mfem_jobs:-$pace_jobs} =~ ^[1-9][0-9]*$ ]] || pace_fail "build jobs must be positive integers"
pace_cpu_limit=$(( ${SLURM_CPUS_PER_TASK:-8} * ${SLURM_NTASKS:-1} ))
((pace_jobs <= pace_cpu_limit && ${pace_mfem_jobs:-$pace_jobs} <= pace_cpu_limit)) || \
  pace_fail "build jobs exceed the $pace_cpu_limit allocated CPUs"
export IGRLAB_BUILD_JOBS="$pace_jobs"
pace_profile=cpu
if [[ $pace_mode == gpu ]]; then
  [[ -z $pace_arch || $pace_arch == 80 ]] || pace_fail "the PACE GPU profile requires native A100 architecture 80"
  pace_profile=cuda-sm80
  pace_args+=(--a100)
fi
export IGRLAB_BUILD_SITE=pace IGRLAB_PACE_DEPENDENCY_ROOT="$pace_dependency_root"
export IGRLAB_SPACK_ENV="$pace_dependency_root/site/pace/spack-envs/$pace_profile"
export IGRLAB_CUDA_ARCH=80 IGRLAB_MFEM_CUDA_ARCHS=80 IGRLAB_APP_CUDA_ARCHS=80
if [[ $pace_plan == 1 ]]; then
  printf '[site-plan] site=pace profile=%s target=x86_64_v3 account=gts-sbryngelson3 qos=embers submit=%s\n' "$pace_profile" "$pace_submit"
  printf '[site-plan] dependency_root=%s spack_env=%s\n' "$pace_dependency_root" "$IGRLAB_SPACK_ENV"
  exec "${pace_source}/build.sh" "${pace_args[@]}"
fi
[[ $(hostname -f) == *.pace.gatech.edu || -d /storage/pace-apps ]] || pace_fail "this profile requires PACE; use --print-plan for an offline plan"
if [[ -z $pace_spack_root ]]; then
  command -v spack >/dev/null 2>&1 || pace_fail "initialize Spack once or provide --spack-root /path/to/spack/core"
  pace_spack_root=$(spack location -r)
fi
pace_spack_root=$(realpath -e -- "$pace_spack_root")
[[ -f $pace_spack_root/share/spack/setup-env.sh ]] || pace_fail "Spack shell setup is missing"
pace_recipe_root=${IGRLAB_PACE_RECIPE_ROOT:-$(dirname -- "$pace_spack_root")/packages/repos/spack_repo/builtin}
[[ -d $pace_recipe_root ]] || pace_fail "set IGRLAB_PACE_RECIPE_ROOT to the Spack builtin recipe repository"
pace_recipe_root=$(realpath -e -- "$pace_recipe_root")

if [[ $pace_submit == 1 ]]; then
  [[ -z ${SLURM_JOB_ID:-} ]] || pace_fail "--submit is for the login node; omit it inside an allocation"
  # Reject source/flag errors before occupying a queued compute allocation.
  "${pace_source}/build.sh" "${pace_args[@]}" --print-plan
  if [[ $pace_dependency_root != "$pace_root" ]]; then
    "${pace_source}/build.sh" "${pace_args[@]}" --output-root "$pace_dependency_root" --print-plan
  fi
  pace_prepare_site_directory "$pace_dependency_root/site/pace/requests"
  pace_prepare_site_directory "$pace_dependency_root/site/pace/logs"
  pace_request=$(mktemp "$pace_dependency_root/site/pace/requests/build.XXXXXXXX.json")
  python3 - "$pace_request" "$pace_source" --site pace --site-root "$pace_dependency_root" --spack-root "$pace_spack_root" "${pace_args[@]}" <<'PY'
import json, sys
with open(sys.argv[1], 'w') as f:
    json.dump({'source': sys.argv[2], 'argv': sys.argv[3:]}, f, indent=2)
    f.write('\n')
PY
  pace_job=$(sbatch --parsable --account=gts-sbryngelson3 --qos=embers --partition=cpu-small \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=08:00:00 --no-requeue \
    --job-name=IGRlab-build --output="$pace_dependency_root/site/pace/logs/build-%j.out" \
    --error="$pace_dependency_root/site/pace/logs/build-%j.err" "$0" --worker build "$pace_request")
  pace_job=${pace_job%%;*}
  printf 'PACE build submitted: %s (embers, cpu-small)\n' "$pace_job"
  if [[ $pace_tests == 1 || $pace_smoke == 1 ]]; then
    pace_partition=cpu-small pace_resources=()
    if [[ $pace_profile == cuda-sm80 ]]; then
      pace_partition=gpu-a100; pace_resources+=(--gres=gpu:a100:1)
    fi
    pace_validation=$(sbatch --parsable --dependency="afterok:$pace_job" --account=gts-sbryngelson3 \
      --qos=embers --partition="$pace_partition" "${pace_resources[@]}" --nodes=1 --ntasks=4 \
      --cpus-per-task=2 --mem=48G --time=02:00:00 --no-requeue --job-name=IGRlab-check \
      --output="$pace_dependency_root/site/pace/logs/check-%j.out" \
      --error="$pace_dependency_root/site/pace/logs/check-%j.err" "$0" --worker validation "$pace_request")
    printf 'PACE validation submitted: %s (after build, embers, %s)\n' "${pace_validation%%;*}" "$pace_partition"
  fi
  printf 'Logs: %s/site/pace/logs; request: %s\n' "$pace_dependency_root" "$pace_request"
  exit 0
fi
[[ -n ${SLURM_JOB_ID:-} && ${SLURM_JOB_ACCOUNT:-} == gts-sbryngelson3 &&
   ${SLURM_JOB_QOS:-} == embers ]] || pace_fail "use --submit or a free embers compute allocation"
[[ $(hostname -f) != *login* ]] || pace_fail "compilation requires a compute node"
if [[ $pace_profile == cuda-sm80 && ($pace_tests == 1 || $pace_smoke == 1) ]]; then
  [[ ${SLURM_JOB_PARTITION:-} == gpu-a100 && -n ${SLURM_JOB_GPUS:-} ]] || \
    pace_fail "CUDA validation requires an allocated A100; use --submit or --no-tests for compilation"
  export DG_IGR_TEST_DEVICE=cuda
fi
type module >/dev/null 2>&1 || pace_fail "PACE environment modules are unavailable"
module purge
mapfile -t pace_modules < <(python3 - "$pace_config" "$pace_profile" <<'PY'
import json, sys
print('\n'.join(json.load(open(sys.argv[1]))['profiles'][sys.argv[2]]['modules']))
PY
)
module load "${pace_modules[@]}"
mapfile -t pace_prefixes < <(python3 - "$pace_config" "$pace_profile" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
print(c['profiles'][sys.argv[2]]['mpi_prefix'])
print(c['cuda_prefix'])
PY
)
pace_mpi=${pace_prefixes[0]} pace_cuda=${pace_prefixes[1]}
export PATH="$pace_mpi/bin:$PATH" LD_LIBRARY_PATH="$pace_mpi/lib:${LD_LIBRARY_PATH:-}"
[[ $(readlink -f "$(command -v mpicc)") == $(readlink -f "$pace_mpi/bin/mpicc") ]] || pace_fail "selected MPI wrapper mismatch"
if [[ $pace_profile == cuda-sm80 ]]; then
  [[ $(readlink -f "$(command -v nvcc)") == $(readlink -f "$pace_cuda/bin/nvcc") ]] || pace_fail "selected CUDA compiler mismatch"
fi
unset OMPI_MCA_plm OMPI_MCA_ras OMPI_MCA_pml OMPI_MCA_btl OMPI_MCA_ess OMPI_MCA_ess_singleton_isolated
unset SPACK_ENV SPACK_ROOT CUDA_ARCHS MFEM_CUDA_ARCHS APP_CUDA_ARCHS SPACK_CUDA_ARCH
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
export IGRLAB_CUDA_ARCH=80 IGRLAB_MFEM_CUDA_ARCHS=80 IGRLAB_APP_CUDA_ARCHS=80
export SPACK_USER_CONFIG_PATH="$pace_dependency_root/site/pace/spack-user"
export SPACK_SYSTEM_CONFIG_PATH="$pace_dependency_root/site/pace/spack-system"
export SPACK_USER_CACHE_PATH="$pace_dependency_root/spack-cache"
unset SPACK_DISABLE_LOCAL_CONFIG
source "$pace_spack_root/share/spack/setup-env.sh"
[[ $(spack --version) == 1.3.* ]] || pace_fail "this site profile requires the tested Spack 1.3 tool/recipe schema"
export IGRLAB_PACE_RECIPE_ROOT="$pace_recipe_root"
# Validate source and external output paths before preparing any site files.
"${pace_source}/build.sh" "${pace_args[@]}" --print-plan
if [[ $pace_dependency_root != "$pace_root" ]]; then
  "${pace_source}/build.sh" "${pace_args[@]}" --output-root "$pace_dependency_root" --print-plan
fi
# Worktrees can share a dependency root. The driver releases this profile lock
# after preparing dependencies, so independent compilation can run concurrently.
pace_prepare_dependency_lock
python3 "$pace_script_dir/pace_manifest.py" --root "$pace_dependency_root" \
  --profile "$pace_profile" --recipe-root "$pace_recipe_root"
export TMPDIR="$pace_dependency_root/tmp" OMP_NUM_THREADS=1 CTEST_PARALLEL_LEVEL=1
mkdir -p "$TMPDIR" "$SPACK_SYSTEM_CONFIG_PATH"
printf '[site] PACE profile=%s target=x86_64_v3 MPI=%s account=%s qos=%s\n' \
  "$pace_profile" "$pace_mpi" "$SLURM_JOB_ACCOUNT" "$SLURM_JOB_QOS"
exec "${pace_source}/build.sh" "${pace_args[@]}"
