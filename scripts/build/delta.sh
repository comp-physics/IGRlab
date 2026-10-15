#!/usr/bin/env bash
# Delta environment and Slurm front end; all builds use the canonical driver.
set -euo pipefail
delta_script_dir="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
delta_source="$(cd -- "${delta_script_dir}/../.." && pwd -P)"
delta_config="${delta_source}/cmake/sites/delta.json"

delta_fail() { echo "ERROR: Delta: $*" >&2; exit 2; }

delta_validate_wall_time() {
  local value="$1" option="$2"
  [[ $value =~ ^([0-9]{2,}):([0-5][0-9]):([0-5][0-9])$ ]] || \
    delta_fail "$option requires HH:MM:SS with minutes/seconds below 60"
  [[ $value =~ [1-9] ]] || delta_fail "$option must be positive"
}

delta_admit_compute() {
  local expected_account="$1" expected_partition="$2" compilation="$3"
  python3 -B - "$expected_account" "$expected_partition" "$compilation" <<'PYCODE'
import os, pwd, re, socket, subprocess, sys
account, partition, compilation = sys.argv[1:]
job = os.environ.get('SLURM_JOB_ID', '')
if not re.fullmatch(r'[1-9][0-9]*', job) or 'login' in socket.getfqdn():
    raise SystemExit('ERROR: Delta: a real compute allocation is required')
result = subprocess.run(['scontrol', 'show', 'job', '-o', job], text=True,
                        capture_output=True, timeout=30)
fields = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)', result.stdout))
identity = f'{pwd.getpwuid(os.getuid()).pw_name}({os.getuid()})'
if result.returncode or fields.get('JobId') != job or fields.get('JobState') != 'RUNNING' or \
        fields.get('UserId') != identity or fields.get('Account') != account or \
        fields.get('Partition') != partition or any(os.environ.get(key) != fields.get(field)
        for key, field in [('SLURM_JOB_ACCOUNT', 'Account'), ('SLURM_JOB_PARTITION', 'Partition')]):
    raise SystemExit('ERROR: Delta: scheduler identity does not match the owned running allocation')
if compilation == 'yes' and any(re.search(r'(?:^|,)gres/gpu(?:[^=]*)=[1-9]', fields.get(key, ''))
                                 for key in ('AllocTRES', 'ReqTRES')):
    raise SystemExit('ERROR: Delta: compilation requires CPU compute without GPUs')
PYCODE
}

delta_prepare_site_directory() {
  local target="$1" component
  for component in "$delta_dependency_root/site" "$delta_dependency_root/site/delta" "$target"; do
    [[ ! -L $component && (! -e $component || -d $component) ]] || \
      delta_fail "site directory is symlinked or not a directory: $component"
  done
  mkdir -p -- "$target"
}

delta_prepare_dependency_lock() {
  delta_prepare_site_directory "$delta_dependency_root/site/delta/locks"
  local lock_path="$delta_dependency_root/site/delta/locks/$delta_profile.lock"
  [[ ! -L $lock_path && (! -e $lock_path || -f $lock_path) ]] || \
    delta_fail "dependency lock is symlinked or not a regular file: $lock_path"
  exec {delta_environment_lock}>>"$lock_path"
  flock "$delta_environment_lock"
  export IGRLAB_DELTA_ENV_LOCK_FD="$delta_environment_lock"
}

delta_admit_prebuilt_site() {
  python3 -B - "$delta_script_dir" "$delta_dependency_root" "$delta_profile" "$delta_recipe_root" "$delta_discovered" <<'PYCODE'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import delta_manifest
root, profile, recipes = Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
config = delta_manifest.resolved_config(json.loads(sys.argv[5]), profile)
expected = {
    root / 'site/delta/spack-envs' / profile / 'spack.yaml': delta_manifest.manifest(root, profile, recipes, config),
    root / 'site/delta/spack-envs' / profile / 'site-packages.json': config,
    root / 'site/delta/spack-user/repos.yaml': {'repos': {'builtin': str(recipes)}},
}
for path, document in expected.items():
    if not path.is_file() or path.is_symlink() or path != path.resolve(strict=True) or path.read_text() != json.dumps(document, indent=2) + '\n':
        raise SystemExit(f'ERROR: Delta: missing or changed retained configuration for --tests-only: {path}')
PYCODE
}

# sbatch executes the tracked helper, with the original argv kept as JSON.
if [[ ${1:-} == --worker ]]; then
  [[ $# == 3 && ($2 == build || $2 == validation) ]] || delta_fail "invalid worker arguments"
  delta_stage=$2
  delta_request=$3
  # Slurm runs a spool copy, so BASH_SOURCE cannot locate the checkout here.
  delta_source=$(python3 - "$delta_request" <<'PY'
import json, sys
from pathlib import Path
source = Path(json.load(open(sys.argv[1]))['source'])
if not source.is_absolute() or source != source.resolve(strict=True):
    raise SystemExit('invalid checkout path in Delta build request')
print(source)
PY
  )
  [[ -f $delta_source/build.sh && -f $delta_source/scripts/build/delta.sh ]] || delta_fail "requested checkout no longer exists"
  delta_worker_tasks=1
  [[ $delta_stage == build ]] || delta_worker_tasks=4
  [[ ${SLURM_NTASKS:-} == "$delta_worker_tasks" ]] || \
    delta_fail "worker requires the requested $delta_worker_tasks-task allocation"
  mapfile -d '' -t delta_original < <(python3 - "$delta_request" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for arg in r['argv']:
    sys.stdout.buffer.write(arg.encode() + b'\0')
PY
  )
  ((${#delta_original[@]})) || delta_fail "empty build request"
  delta_worker_args=()
  while ((${#delta_original[@]})); do
    delta_arg=${delta_original[0]}; delta_original=("${delta_original[@]:1}")
    case "$delta_arg" in
      --no-tests) [[ $delta_stage == build ]] || delta_worker_args+=("$delta_arg") ;;
      --tests-only) [[ $delta_stage == validation ]] || delta_worker_args+=("$delta_arg") ;;
      --no-clean|--clean|--no-deps)
        [[ $delta_stage == validation ]] || delta_worker_args+=("$delta_arg") ;;
      --smoke) [[ $delta_stage == build ]] || delta_worker_args+=("$delta_arg") ;;
      --test-regex|--test-label)
        ((${#delta_original[@]})) || delta_fail "missing test filter in request"
        [[ $delta_stage == build ]] || delta_worker_args+=("$delta_arg" "${delta_original[0]}")
        delta_original=("${delta_original[@]:1}") ;;
      *) delta_worker_args+=("$delta_arg") ;;
    esac
  done
  if [[ $delta_stage == build ]]; then
    delta_worker_args+=(--no-tests)
  else
    delta_worker_args+=(--tests-only --no-clean --no-deps)
  fi
  exec "${delta_source}/build.sh" "${delta_worker_args[@]}"
fi

delta_args=()
delta_submit=0 delta_plan=0 delta_tests_only=0 delta_tests=1 delta_smoke=0 delta_mode=cpu delta_arch="" delta_clean_set=0 delta_help=0
delta_a100=0 delta_cpu_set=0 delta_arch_set=0 delta_a100_arch_conflict=0
delta_iterate=0 delta_explicit_clean=0 delta_no_deps=0 delta_test_filter=0 delta_target=""
delta_root="" delta_dependency_root="" delta_spack_root=${IGRLAB_DELTA_SPACK_ROOT:-}
delta_jobs=${IGRLAB_BUILD_JOBS:-4} delta_mfem_jobs=""
delta_build_time=01:00:00 delta_validation_time=01:00:00
while (($#)); do
  case "$1" in
    --site) [[ $# -ge 2 && $2 == delta ]] || delta_fail "supported site is delta"; shift ;;
    --submit) delta_submit=1 ;;
    --build-time) [[ $# -ge 2 ]] || delta_fail "--build-time requires HH:MM:SS"; delta_build_time=$2; shift ;;
    --validation-time) [[ $# -ge 2 ]] || delta_fail "--validation-time requires HH:MM:SS"; delta_validation_time=$2; shift ;;
    --site-root) [[ $# -ge 2 ]] || delta_fail "--site-root requires a path"; delta_dependency_root=$2; shift ;;
    --spack-root) [[ $# -ge 2 ]] || delta_fail "--spack-root requires a path"; delta_spack_root=$2; shift ;;
    --output-root) [[ $# -ge 2 ]] || delta_fail "--output-root requires a path"; delta_root=$2; delta_args+=("$1" "$2"); shift ;;
    --a100) delta_mode=gpu; delta_a100=1; delta_args+=("$1") ;;
    --gpu) delta_mode=gpu; delta_args+=("$1") ;;
    --cpu|--cpu-only) delta_mode=cpu; delta_cpu_set=1; delta_args+=("$1") ;;
    --cuda-arch)
      [[ $# -ge 2 ]] || delta_fail "--cuda-arch requires a value"
      delta_arch=$2; delta_arch_set=1
      [[ $delta_arch == 80 ]] || delta_a100_arch_conflict=1
      delta_args+=("$1" "$2"); shift ;;
    -j|--jobs) [[ $# -ge 2 ]] || delta_fail "$1 requires a value"; delta_jobs=$2; delta_args+=("$1" "$2"); shift ;;
    --mfem-jobs) [[ $# -ge 2 ]] || delta_fail "$1 requires a value"; delta_mfem_jobs=$2; delta_args+=("$1" "$2"); shift ;;
    --print-plan) delta_plan=1; delta_args+=("$1") ;;
    --no-tests) delta_tests=0; delta_args+=("$1") ;;
    --tests-only) delta_tests_only=1; delta_args+=("$1") ;;
    --smoke) delta_smoke=1; delta_args+=("$1") ;;
    --clean) delta_explicit_clean=1; delta_clean_set=1; delta_args+=("$1") ;;
    --no-clean) delta_clean_set=1; delta_args+=("$1") ;;
    --no-deps) delta_no_deps=1; delta_args+=("$1") ;;
    --iterate) delta_iterate=1; delta_args+=("$1") ;;
    --target)
      [[ $# -ge 2 && -n $2 ]] || delta_fail "--target requires a name"
      delta_target=$2; delta_args+=("$1" "$2"); shift ;;
    --test-regex|--test-label)
      [[ $# -ge 2 && -n $2 ]] || delta_fail "$1 requires a regex"
      delta_test_filter=1; delta_args+=("$1" "$2"); shift ;;
    --caliper) delta_fail "the Delta site profiles are cpu and cuda-sm80; Caliper has no Delta profile yet" ;;
    --help|-h) delta_help=1; delta_args+=("$1") ;;
    *) delta_args+=("$1") ;;
  esac
  shift
done
[[ $delta_iterate == 0 || $delta_explicit_clean == 0 ]] || delta_fail "--iterate conflicts with --clean"
[[ -z $delta_target || $delta_iterate == 1 ]] || delta_fail "--target requires --iterate"
if [[ $delta_iterate == 1 ]]; then
  [[ $delta_smoke == 0 ]] || delta_fail "--iterate does not run --smoke; select tests with --tests-only and a filter"
  [[ $delta_no_deps == 1 ]] || delta_args+=(--no-deps)
  if [[ $delta_tests_only == 1 ]]; then
    [[ $delta_tests == 1 ]] || delta_fail "--tests-only conflicts with --no-tests"
    [[ $delta_test_filter == 1 ]] || delta_fail "--iterate --tests-only requires --test-regex or --test-label"
    [[ -z $delta_target ]] || delta_fail "--target conflicts with --tests-only"
  else
    [[ $delta_test_filter == 0 ]] || delta_fail "iteration test filters require --tests-only"
    [[ $delta_tests == 0 ]] || delta_args+=(--no-tests)
    delta_tests=0
  fi
fi
if [[ $delta_help == 1 ]]; then
  printf 'Delta site:\n  --site delta --site-root DIR --spack-root DIR --submit\n  --a100 / --gpu            Native A100 architecture 80 (gpuA100x4).\n  --build-time HH:MM:SS       CPU build limit (default 01:00:00).\n  --validation-time HH:MM:SS  Separate validation limit (default 01:00:00).\n  Both limits may span multiple hours, subject to Delta scheduler policy.\n  --no-tests submits only a CPU build; --tests-only submits only validation.\n\n'
  exec "${delta_source}/build.sh" --help
fi
delta_validate_wall_time "$delta_build_time" --build-time
delta_validate_wall_time "$delta_validation_time" --validation-time
[[ $delta_tests_only == 0 || $delta_tests == 1 ]] || delta_fail "--tests-only conflicts with --no-tests"
[[ -n $delta_root && $delta_root == /* ]] || delta_fail "provide an absolute --output-root on scratch"
delta_root=$(realpath -m -- "$delta_root")
[[ -z $delta_dependency_root || $delta_dependency_root == /* ]] || delta_fail "--site-root must be absolute"
delta_dependency_root=$(realpath -m -- "${delta_dependency_root:-$delta_root}")
[[ $delta_clean_set == 1 ]] || delta_args+=(--no-clean)
[[ $delta_jobs =~ ^[1-9][0-9]*$ && ${delta_mfem_jobs:-$delta_jobs} =~ ^[1-9][0-9]*$ ]] || delta_fail "build jobs must be positive integers"
delta_cpu_limit=$(( ${SLURM_CPUS_PER_TASK:-8} * ${SLURM_NTASKS:-1} ))
((delta_jobs <= delta_cpu_limit && ${delta_mfem_jobs:-$delta_jobs} <= delta_cpu_limit)) || \
  delta_fail "build jobs exceed the $delta_cpu_limit allocated CPUs"
export IGRLAB_BUILD_JOBS="$delta_jobs"
delta_profile=cpu
delta_native_arch=80 delta_gpu_partition="" delta_gpu_gres="" delta_gpu_model=""
if [[ $delta_a100 == 1 ]]; then
  [[ $delta_cpu_set == 0 && $delta_a100_arch_conflict == 0 ]] || \
    delta_fail "--a100 conflicts with --cpu-only/--cpu or --cuda-arch other than 80"
fi
if [[ $delta_mode == gpu ]]; then
  if [[ $delta_arch_set == 1 ]]; then
    [[ $delta_arch == 80 ]] || delta_fail "Delta GPU builds require native A100 architecture 80"
    delta_native_arch=$delta_arch
  fi
  delta_profile=cuda-sm${delta_native_arch}
  delta_gpu_partition=gpuA100x4 delta_gpu_gres=1 delta_gpu_model=A100
  [[ $delta_a100 == 1 ]] || delta_args+=(--a100)
elif [[ $delta_arch_set == 1 ]]; then
  delta_fail "--cuda-arch is only valid with --gpu"
fi
export IGRLAB_BUILD_SITE=delta IGRLAB_DELTA_DEPENDENCY_ROOT="$delta_dependency_root"
export IGRLAB_SITE_MPI_SPEC='openmpi+cuda'
export IGRLAB_SPACK_ENV="$delta_dependency_root/site/delta/spack-envs/$delta_profile"
export IGRLAB_CUDA_ARCH="$delta_native_arch" IGRLAB_MFEM_CUDA_ARCHS="$delta_native_arch" IGRLAB_APP_CUDA_ARCHS="$delta_native_arch"
if [[ $delta_plan == 1 ]]; then
  printf '[site-plan] site=delta profile=%s build_account=bdiy-delta-cpu build_partition=cpu submit=%s\n' "$delta_profile" "$delta_submit"
  printf '[site-plan] dependency_root=%s spack_env=%s\n' "$delta_dependency_root" "$IGRLAB_SPACK_ENV"
  printf '[site-plan] build_time=%s validation_time=%s\n' "$delta_build_time" "$delta_validation_time"
  [[ $delta_mode != gpu ]] || printf '[site-plan] validation_account=bdiy-delta-gpu validation_partition=%s validation_gpus_per_node=%s\n' "$delta_gpu_partition" "$delta_gpu_gres"
  exec "${delta_source}/build.sh" "${delta_args[@]}"
fi
[[ $(hostname -f) == *.delta.ncsa.illinois.edu ]] || delta_fail "this profile requires Delta; use --print-plan for an offline plan"
delta_spack_args=()
if [[ $delta_iterate == 0 ]]; then
  if [[ -z $delta_spack_root ]]; then
    command -v spack >/dev/null 2>&1 || delta_fail "initialize Spack once or provide --spack-root /path/to/spack/core"
    delta_spack_root=$(spack location -r)
  fi
  delta_spack_root=$(realpath -e -- "$delta_spack_root")
  [[ -f $delta_spack_root/share/spack/setup-env.sh ]] || delta_fail "Spack shell setup is missing"
  delta_spack_args=(--spack-root "$delta_spack_root")
fi
if [[ $delta_submit == 1 ]]; then
  [[ -z ${SLURM_JOB_ID:-} ]] || delta_fail "--submit is for the login node; omit it inside an allocation"
  # Reject source/flag errors before occupying a queued compute allocation.
  "${delta_source}/build.sh" "${delta_args[@]}" --print-plan
  if [[ $delta_iterate == 0 && $delta_dependency_root != "$delta_root" ]]; then
    "${delta_source}/build.sh" "${delta_args[@]}" --output-root "$delta_dependency_root" --print-plan
  fi
  delta_prepare_site_directory "$delta_dependency_root/site/delta/requests"
  delta_prepare_site_directory "$delta_dependency_root/site/delta/logs"
  delta_request=$(mktemp "$delta_dependency_root/site/delta/requests/build.XXXXXXXX.json")
  python3 - "$delta_request" "$delta_source" --site delta --site-root "$delta_dependency_root" "${delta_spack_args[@]}" \
    --build-time "$delta_build_time" --validation-time "$delta_validation_time" "${delta_args[@]}" <<'PY'
import json, sys
with open(sys.argv[1], 'w') as f:
    json.dump({'source': sys.argv[2], 'argv': sys.argv[3:]}, f, indent=2)
    f.write('\n')
PY
  chmod 0444 "$delta_request"
  delta_validation_dependency=()
  if [[ $delta_tests_only == 0 ]]; then
  delta_job=$(sbatch --parsable --account=bdiy-delta-cpu --partition=cpu \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time="$delta_build_time" --no-requeue \
    --job-name=IGRlab-build --output="$delta_dependency_root/site/delta/logs/build-%j.out" \
    --error="$delta_dependency_root/site/delta/logs/build-%j.err" "$0" --worker build "$delta_request")
  delta_job=${delta_job%%;*}
  [[ $delta_job =~ ^[1-9][0-9]*$ ]] || delta_fail "scheduler returned an invalid build job ID"
  printf 'Delta build submitted: %s (bdiy-delta-cpu, cpu)\n' "$delta_job"
    delta_validation_dependency=(--dependency="afterok:$delta_job")
  fi
  if [[ $delta_tests == 1 || $delta_smoke == 1 ]]; then
    delta_partition=cpu delta_account=bdiy-delta-cpu delta_resources=()
    if [[ $delta_mode == gpu ]]; then
      delta_partition=$delta_gpu_partition; delta_account=bdiy-delta-gpu
      delta_resources+=(--gpus-per-node="$delta_gpu_gres")
    fi
    delta_validation=$(sbatch --parsable "${delta_validation_dependency[@]}" --kill-on-invalid-dep=yes --account="$delta_account" \
      --partition="$delta_partition" "${delta_resources[@]}" --nodes=1 --ntasks=4 \
      --cpus-per-task=2 --mem=48G --time="$delta_validation_time" --no-requeue --job-name=IGRlab-check \
      --output="$delta_dependency_root/site/delta/logs/check-%j.out" \
      --error="$delta_dependency_root/site/delta/logs/check-%j.err" "$0" --worker validation "$delta_request")
    delta_validation=${delta_validation%%;*}
    [[ $delta_validation =~ ^[1-9][0-9]*$ ]] || delta_fail "scheduler returned an invalid validation job ID"
    delta_validation_kind='prebuilt validation'
    [[ $delta_iterate == 0 ]] || delta_validation_kind='selected iteration tests'
    printf 'Delta validation submitted: %s (%s, %s, %s)\n' "$delta_validation" "$delta_validation_kind" "$delta_account" "$delta_partition"
  fi
  printf 'Logs: %s/site/delta/logs; request: %s\n' "$delta_dependency_root" "$delta_request"
  exit 0
fi
if [[ $delta_tests_only == 0 ]]; then
  [[ $delta_tests == 0 && $delta_smoke == 0 ]] || \
    delta_fail "compute builds require --no-tests; submit or run --tests-only separately"
  delta_admit_compute bdiy-delta-cpu cpu yes
elif [[ $delta_mode == gpu ]]; then
  delta_admit_compute bdiy-delta-gpu "$delta_gpu_partition" no
else
  delta_admit_compute bdiy-delta-cpu cpu no
fi
if [[ $delta_mode == gpu && $delta_tests_only == 1 ]]; then
  # Interactive srun steps can expose only the step's assigned GPU IDs.
  [[ ${SLURM_JOB_PARTITION:-} == "$delta_gpu_partition" && -n ${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}} ]] || \
    delta_fail "CUDA validation requires an allocated $delta_gpu_model; use --submit or --no-tests for compilation"
  export DG_IGR_TEST_DEVICE=cuda
fi
type module >/dev/null 2>&1 || delta_fail "Delta environment modules are unavailable"
# App iteration enters the cached CMake build directly. No dependency discovery,
# Spack activation, manifest preparation, or qualification receipts are involved.
if [[ $delta_iterate == 1 ]]; then
  module purge
  mapfile -t delta_modules < <(python3 - "$delta_config" "$delta_profile" <<'PYCODE'
import json, sys
print('\n'.join(json.load(open(sys.argv[1]))['profiles'][sys.argv[2]]['modules']))
PYCODE
  )
  module load "${delta_modules[@]}"
  export OMPI_CC=gcc OMPI_CXX=g++ OMPI_FC=gfortran
  unset OMPI_MCA_plm OMPI_MCA_ras OMPI_MCA_pml OMPI_MCA_btl OMPI_MCA_ess OMPI_MCA_ess_singleton_isolated
  unset SPACK_ENV SPACK_ROOT CUDA_ARCHS MFEM_CUDA_ARCHS APP_CUDA_ARCHS SPACK_CUDA_ARCH
  unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CTEST_PARALLEL_LEVEL=1
  printf '[site] Delta app iteration profile=%s account=%s\n' "$delta_profile" "$SLURM_JOB_ACCOUNT"
  exec "${delta_source}/build.sh" "${delta_args[@]}"
fi
# Retained environments own their recipes; fresh roots query the selected tool.
delta_recipe_args=(--site-root "$delta_dependency_root" --recipe-profile "$delta_profile")
if [[ -n ${IGRLAB_DELTA_RECIPE_ROOT:-} ]]; then
  delta_recipe_args+=(--recipe-override "$IGRLAB_DELTA_RECIPE_ROOT")
fi
delta_recipe_root=$(SPACK_ROOT="$delta_spack_root" python3 "$delta_script_dir/delta_discover.py" \
  --spack "$delta_spack_root/bin/spack" --recipe-root "${delta_recipe_args[@]}")
[[ -d $delta_recipe_root ]] || delta_fail "Spack's builtin recipe repository is unavailable; configure it with spack repo add"
delta_recipe_root=$(realpath -e -- "$delta_recipe_root")
module purge
mapfile -t delta_modules < <(python3 - "$delta_config" "$delta_profile" <<'PY'
import json, sys
print('\n'.join(json.load(open(sys.argv[1]))['profiles'][sys.argv[2]]['modules']))
PY
)
module load "${delta_modules[@]}"
# Use Delta's system HPC-X MPI with the selected GNU toolchain for both
# profiles. CPU HYPRE/MFEM still disable CUDA; the MPI external is CUDA-aware.
export OMPI_CC=gcc OMPI_CXX=g++ OMPI_FC=gfortran
export IGRLAB_SITE_MPI_SPEC='openmpi+cuda'
# Discover the loaded site executables directly; no shared Spack store is
# queried or rewritten, and no package prefixes are embedded here.
delta_anchor=$(type -P gcc)
[[ -x $delta_anchor ]] || delta_fail "compiler module did not expose GCC"
delta_discovery_args=(--profile "$delta_profile" --spack "$delta_spack_root/bin/spack"
  --anchor "$delta_anchor" --cmake "$(type -P cmake)" --mpi "$(type -P mpicxx)"
  --cuda "$(type -P nvcc)")
delta_discovered=$(python3 "$delta_script_dir/delta_discover.py" "${delta_discovery_args[@]}")
mapfile -t delta_prefixes < <(python3 -c 'import json, sys; c=json.load(sys.stdin); print(c["profiles"][sys.argv[1]]["mpi_prefix"]); print(c.get("cuda_prefix", "")); print("\n".join(c["compiler"]["executables"][key] for key in ("c", "cxx", "fortran")))' \
  "$delta_profile" <<<"$delta_discovered")
delta_mpi=${delta_prefixes[0]} delta_cuda=${delta_prefixes[1]}
# The system wrappers accept bare GNU names for discovery. Canonical MFEM
# admission requires executable paths, so use the same discovered and verified
# compiler files for builds and validation without changing the retained graph.
export OMPI_CC="${delta_prefixes[2]}" OMPI_CXX="${delta_prefixes[3]}" OMPI_FC="${delta_prefixes[4]}"
export PATH="$delta_mpi/bin:$PATH" LD_LIBRARY_PATH="$delta_mpi/lib:$delta_mpi/lib64:${LD_LIBRARY_PATH:-}"
[[ $(readlink -f "$(command -v mpicc)") == $(readlink -f "$delta_mpi/bin/mpicc") ]] || delta_fail "selected MPI wrapper mismatch"
if [[ $delta_mode == gpu ]]; then
  [[ $(readlink -f "$(command -v nvcc)") == $(readlink -f "$delta_cuda/bin/nvcc") ]] || delta_fail "selected CUDA compiler mismatch"
fi
unset OMPI_MCA_plm OMPI_MCA_ras OMPI_MCA_pml OMPI_MCA_btl OMPI_MCA_ess OMPI_MCA_ess_singleton_isolated
unset SPACK_ENV SPACK_ROOT CUDA_ARCHS MFEM_CUDA_ARCHS APP_CUDA_ARCHS SPACK_CUDA_ARCH
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
export IGRLAB_CUDA_ARCH="$delta_native_arch" IGRLAB_MFEM_CUDA_ARCHS="$delta_native_arch" IGRLAB_APP_CUDA_ARCHS="$delta_native_arch"
export SPACK_USER_CONFIG_PATH="$delta_dependency_root/site/delta/spack-user"
export SPACK_SYSTEM_CONFIG_PATH="$delta_dependency_root/site/delta/spack-system"
export SPACK_USER_CACHE_PATH="$delta_dependency_root/spack-cache"
unset SPACK_DISABLE_LOCAL_CONFIG
source "$delta_spack_root/share/spack/setup-env.sh"
[[ $(spack --version) == 1.2.* ]] || delta_fail "this site profile requires the selected Spack 1.2 tool/recipe schema"
export IGRLAB_DELTA_RECIPE_ROOT="$delta_recipe_root"
# Validate source and external output paths before preparing any site files.
"${delta_source}/build.sh" "${delta_args[@]}" --print-plan
if [[ $delta_dependency_root != "$delta_root" ]]; then
  "${delta_source}/build.sh" "${delta_args[@]}" --output-root "$delta_dependency_root" --print-plan
fi
# Worktrees can share a dependency root. The driver releases this profile lock
# after preparing dependencies, so independent compilation can run concurrently.
if [[ $delta_tests_only == 1 ]]; then
  # Run-only validation neither creates nor rewrites dependency configuration.

  delta_existing_lock="$delta_dependency_root/site/delta/locks/$delta_profile.lock"
  [[ -f $delta_existing_lock && ! -L $delta_existing_lock ]] || delta_fail "missing retained dependency lock for --tests-only"
  exec {delta_environment_lock}<"$delta_existing_lock"
  flock -s "$delta_environment_lock"
  export IGRLAB_DELTA_ENV_LOCK_FD="$delta_environment_lock"
  delta_admit_prebuilt_site
else
  delta_prepare_dependency_lock
  python3 "$delta_script_dir/delta_manifest.py" --root "$delta_dependency_root" \
    --profile "$delta_profile" --recipe-root "$delta_recipe_root" --site-config - <<<"$delta_discovered"
fi
delta_temp_base=${IGRLAB_DELTA_TMPDIR:-${SLURM_TMPDIR:-/tmp}}
[[ $delta_temp_base == /* && -d $delta_temp_base && -w $delta_temp_base ]] || \
  delta_fail "temporary root must be an existing writable absolute directory"
# Python copy2 preserves extended attributes. Git's read-only pack files on
# Lustre carry ACL/stripe attributes that cannot be copied back to Lustre.
# Clone temporary Git sources on node-local storage; retain the dependency
# stage, cache, installation and concrete graph under the selected site root.
export TMPDIR
TMPDIR=$(mktemp -d "$delta_temp_base/igrlab-delta-${SLURM_JOB_ID}-${delta_profile}.XXXXXXXX")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CTEST_PARALLEL_LEVEL=1
if [[ $delta_tests_only == 1 ]]; then
  [[ -d $TMPDIR && -d $SPACK_SYSTEM_CONFIG_PATH ]] || delta_fail "missing retained runtime directories for --tests-only"
else
  mkdir -p "$TMPDIR" "$SPACK_SYSTEM_CONFIG_PATH"
fi
printf '[site] Delta profile=%s MPI=%s account=%s partition=%s\n' \
  "$delta_profile" "$delta_mpi" "$SLURM_JOB_ACCOUNT" "$SLURM_JOB_PARTITION"
if [[ $delta_tests_only == 1 && $delta_tests == 1 ]]; then
  delta_worktree="wt-$(printf '%s' "$delta_source" | sha256sum | cut -c1-16)"
  delta_prebuilt="$delta_root/worktrees/$delta_worktree/$delta_profile/app/build/igrlab-prebuilt-validation-v1.json"
  exec python3 -B "$delta_script_dir/test_result_receipt.py" \
    --prebuilt-receipt "$delta_prebuilt" \
    --result-root "$delta_dependency_root/site/delta/validation-results" \
    -- "$delta_source/build.sh" "${delta_args[@]}"
fi
exec "${delta_source}/build.sh" "${delta_args[@]}"
