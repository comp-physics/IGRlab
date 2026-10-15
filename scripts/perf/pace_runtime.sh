#!/usr/bin/env bash
# Submitted by gpu_study.py; executor is explicit because Slurm spools this file.
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: pace_runtime.sh PLAN STAGE EXECUTOR' >&2; exit 2; }
study_plan=$1 study_stage=$2 study_executor=$3
[[ -n ${SLURM_JOB_ID:-} && ${SLURM_JOB_ACCOUNT:-} == gts-sbryngelson3 &&
   ${SLURM_JOB_QOS:-} == embers && ${SLURM_NTASKS:-} == 1 ]] && {
  [[ ${SLURM_JOB_PARTITION:-} == cpu-small ]] ||
  [[ ${SLURM_JOB_PARTITION:-} == gpu-a100 && -n ${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}} ]]
} || {
  echo 'study requires one rank in a free PACE embers CPU or A100 allocation' >&2; exit 2;
}
for study_mpi_variable in PMIX_NAMESPACE PMIX_RANK PMI_RANK PMI_SIZE; do
  [[ -z ${!study_mpi_variable:-} ]] || {
    echo "GPU study direct launches cannot inherit $study_mpi_variable; start the interactive shell with srun --mpi=none --pty bash -l" >&2
    exit 2
  }
done
[[ $(hostname -f) != *login* ]] || { echo 'GPU studies require a compute node' >&2; exit 2; }
python3 "$study_executor/gpu_study.py" verify --run-dir "$(dirname -- "$study_plan")"
mapfile -t study_paths < <(python3 - "$study_plan" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
print(p['site_root']); print(p['spack_root']); print(p['sources']['candidate']['source_snapshot'])
print(p.get('execution_profile', 'cuda-sm80')); print(p['builds']['candidate']['spack_env'])
PY
)
study_site=${study_paths[0]} study_spack=${study_paths[1]} study_source=${study_paths[2]}
study_profile=${study_paths[3]} study_env=${study_paths[4]}
case "$study_profile" in
  cpu)
    [[ ${SLURM_JOB_PARTITION:-} == cpu-small && ${SLURM_CPUS_PER_TASK:-} == 8 && $study_stage == timings ]] || {
      echo 'CPU-only study requires cpu-small with eight CPUs and the timing stage' >&2; exit 2;
    }
    ;;
  cuda-sm80)
    [[ ${SLURM_JOB_PARTITION:-} == gpu-a100 && -n ${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}} ]] || {
      echo 'GPU study requires the free A100 allocation' >&2; exit 2;
    }
    ;;
  *) echo 'unsupported frozen execution profile' >&2; exit 2;;
esac
type module >/dev/null 2>&1 || { echo 'PACE modules are unavailable' >&2; exit 2; }
module purge
mapfile -t study_modules < <(python3 - "$study_source/cmake/sites/pace.json" "$study_profile" <<'PY'
import json, sys
print('\n'.join(json.load(open(sys.argv[1]))['profiles'][sys.argv[2]]['modules']))
PY
)
module load "${study_modules[@]}"
while IFS= read -r study_variable; do
  case "$study_variable" in IGRLAB_*|DG_IGR_*) unset "$study_variable";; esac
done < <(compgen -e)
unset SPACK_ENV SPACK_ROOT SPACK_DISABLE_LOCAL_CONFIG
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS CUDA_ARCHS MFEM_CUDA_ARCHS APP_CUDA_ARCHS
unset OMPI_MCA_plm OMPI_MCA_ras OMPI_MCA_pml OMPI_MCA_btl OMPI_MCA_ess OMPI_MCA_ess_singleton_isolated
export SPACK_USER_CONFIG_PATH="$study_site/site/pace/spack-user"
export SPACK_SYSTEM_CONFIG_PATH="$study_site/site/pace/spack-system"
export SPACK_USER_CACHE_PATH="$study_site/spack-cache"
source "$study_spack/share/spack/setup-env.sh"
[[ -f $study_env/spack.lock ]] || { echo 'retained selected Spack environment is missing' >&2; exit 2; }
# Query the concretized environment; installation prefixes are never embedded.
study_packages=(openmpi hypre openblas metis)
[[ $study_profile == cpu ]] || study_packages=(openmpi cuda hypre umpire openblas metis)
for study_package in "${study_packages[@]}"; do
  study_prefix=$(spack -e "$study_env" location -i "$study_package")
  [[ -d $study_prefix ]] || { echo "missing selected package: $study_package" >&2; exit 2; }
  [[ ! -d $study_prefix/bin ]] || export PATH="$study_prefix/bin:$PATH"
  for study_lib in lib lib64; do
    [[ ! -d $study_prefix/$study_lib ]] || export LD_LIBRARY_PATH="$study_prefix/$study_lib:${LD_LIBRARY_PATH:-}"
  done
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$study_site/tmp"
exec python3 "$study_executor/gpu_study.py" worker --run-dir "$(dirname -- "$study_plan")" --stage "$study_stage"
