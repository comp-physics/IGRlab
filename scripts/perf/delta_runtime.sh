#!/usr/bin/env bash
# Run a frozen study in its actual Delta allocation; no build or installation.
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: delta_runtime.sh PLAN STAGE EXECUTOR' >&2; exit 2; }
study_plan=$1 study_stage=$2 study_executor=$3
[[ -n ${SLURM_JOB_ID:-} && ${SLURM_NTASKS:-} == 1 && $(hostname -f) == *.delta.ncsa.illinois.edu &&
   $(hostname -f) != *login* ]] || { echo 'Delta studies require the actual owned compute allocation' >&2; exit 2; }
for study_mpi_variable in PMIX_NAMESPACE PMIX_RANK PMI_RANK PMI_SIZE; do
  [[ -z ${!study_mpi_variable:-} ]] || { echo 'start the Delta shell with srun --mpi=none' >&2; exit 2; }
done
python3 -B "$study_executor/gpu_study.py" verify --run-dir "$(dirname -- "$study_plan")"
mapfile -t study_paths < <(python3 - "$study_plan" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
if p['site'] != 'delta':
    raise SystemExit('Delta runtime refuses another site plan')
print(p['site_root']); print(p['spack_root']); print(p['sources']['candidate']['source_snapshot'])
print(p.get('execution_profile', 'cuda-sm80')); print(p['builds']['candidate']['spack_env'])
PY
)
[[ ${#study_paths[@]} == 5 ]] || { echo 'incomplete Delta runtime plan' >&2; exit 2; }
study_site=${study_paths[0]} study_spack=${study_paths[1]} study_source=${study_paths[2]}
study_profile=${study_paths[3]} study_env=${study_paths[4]}
case "$study_profile" in
  cuda-sm80)
    [[ ${SLURM_JOB_ACCOUNT:-} == bdiy-delta-gpu && ${SLURM_JOB_PARTITION:-} == gpuA100x4 &&
       -n ${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}} ]] || { echo 'Delta CUDA worker requires bdiy-delta-gpu/gpuA100x4' >&2; exit 2; }
    ;;
  cpu)
    [[ ${SLURM_JOB_ACCOUNT:-} == bdiy-delta-cpu && ${SLURM_JOB_PARTITION:-} == cpu &&
       ${SLURM_CPUS_PER_TASK:-} == 8 && $study_stage == timings ]] || { echo 'Delta CPU worker requires its owned timing allocation' >&2; exit 2; }
    ;;
  *) echo 'unsupported Delta execution profile' >&2; exit 2;;
esac
type module >/dev/null 2>&1 || { echo 'Delta modules are unavailable' >&2; exit 2; }
module purge
mapfile -t study_modules < <(python3 - "$study_source/cmake/sites/delta.json" "$study_profile" <<'PY'
import json, sys
print('\n'.join(json.load(open(sys.argv[1]))['profiles'][sys.argv[2]]['modules']))
PY
)
module load "${study_modules[@]}"
while IFS= read -r study_variable; do
  case "$study_variable" in IGRLAB_*|DG_IGR_*) unset "$study_variable";; esac
done < <(compgen -e)
unset SPACK_ENV SPACK_ROOT SPACK_DISABLE_LOCAL_CONFIG CUDA_ARCHS MFEM_CUDA_ARCHS APP_CUDA_ARCHS
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
unset OMPI_MCA_plm OMPI_MCA_ras OMPI_MCA_pml OMPI_MCA_btl OMPI_MCA_ess OMPI_MCA_ess_singleton_isolated
export OMPI_CC=gcc OMPI_CXX=g++ OMPI_FC=gfortran
export SPACK_USER_CONFIG_PATH="$study_site/site/delta/spack-user"
export SPACK_SYSTEM_CONFIG_PATH="$study_site/site/delta/spack-system"
export SPACK_USER_CACHE_PATH="$study_site/spack-cache"
source "$study_spack/share/spack/setup-env.sh"
[[ -f $study_env/spack.lock ]] || { echo 'retained Delta dependency lock is missing' >&2; exit 2; }
mapfile -t study_compilers < <(python3 - "$study_env/site-packages.json" <<'PY'
import json, os, sys
from pathlib import Path
tools=json.load(open(sys.argv[1]))['compiler']['executables']
for name in ('c', 'cxx', 'fortran'):
    path=Path(tools[name])
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit('retained Delta compiler executable is unavailable')
    print(path)
PY
)
[[ ${#study_compilers[@]} == 3 ]] || { echo 'incomplete retained Delta compiler policy' >&2; exit 2; }
export OMPI_CC="${study_compilers[0]}" OMPI_CXX="${study_compilers[1]}" OMPI_FC="${study_compilers[2]}"
study_packages=(openmpi hypre openblas metis)
[[ $study_profile == cpu ]] || study_packages=(openmpi cuda hypre umpire openblas metis)
study_mpi_prefix=
for study_package in "${study_packages[@]}"; do
  study_prefix=$(spack -e "$study_env" location -i "$study_package")
  [[ -d $study_prefix ]] || { echo "missing selected package: $study_package" >&2; exit 2; }
  [[ $study_package != openmpi ]] || study_mpi_prefix=$study_prefix
  [[ ! -d $study_prefix/bin ]] || export PATH="$study_prefix/bin:$PATH"
  for study_lib in lib lib64; do
    [[ ! -d $study_prefix/$study_lib ]] || export LD_LIBRARY_PATH="$study_prefix/$study_lib:${LD_LIBRARY_PATH:-}"
  done
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Open MPI 4 otherwise interprets an inherited srun step as direct PMI launch.
# Keep the real allocation/GPU identity; the worker still performs full live
# admission before applications. Only its rank-one launch selectors are removed.
if [[ $study_profile == cuda-sm80 ]]; then
  # Keep the genuine step selector for counter-exclusive controller admission;
  # Open MPI must not see it as a direct PMI launch selector.
  study_step=${SLURM_STEP_ID:-${SLURM_STEPID:-}}
  if [[ -n $study_step ]]; then
    [[ -z ${DELTA_OWNED_STEP_ID:-} || $DELTA_OWNED_STEP_ID == "$study_step" ]] || {
      echo 'conflicting actual Delta step selectors' >&2; exit 2;
    }
    export DELTA_OWNED_STEP_ID="$study_step"
  fi
  unset SLURM_STEP_ID SLURM_STEPID
  # HPC-X's non-exec orted shell wrapper retains the singleton report pipe.
  # Select its own native daemon while keeping the admitted MPI libraries/setup.
  [[ -x $study_mpi_prefix/bin/.bin/orted && -x $study_mpi_prefix/bin/.bin/ompi_info ]] || {
    echo 'selected HPC-X native daemon/tool directory is unavailable' >&2; exit 2;
  }
  export OPAL_BINDIR="$study_mpi_prefix/bin/.bin"
fi
exec python3 -B "$study_executor/gpu_study.py" worker --run-dir "$(dirname -- "$study_plan")" --stage "$study_stage"
