#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <new-output-directory> <unique-build-tag>" >&2
  exit 64
fi

repo=$(git rev-parse --show-toplevel)
cd "$repo"
output=$1
build_tag=$2
if [[ ! $build_tag =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid build tag: $build_tag" >&2
  exit 64
fi
if [[ -e "$output" ]]; then
  echo "refusing to reuse evidence directory: $output" >&2
  exit 73
fi

cpu_build="build-dgsem-production-device-operator-${build_tag}-cpu"
asan_build="build-dgsem-production-device-operator-${build_tag}-asan"
gpu_build="build-dgsem-production-device-operator-${build_tag}-gpu"
for build_dir in "$cpu_build" "$asan_build" "$gpu_build"; do
  if [[ -e "$build_dir" ]]; then
    echo "refusing to reuse build directory: $build_dir" >&2
    exit 73
  fi
done

mkdir -p "$output/cpu" "$output/cpu-mpi" "$output/asan" "$output/gpu"
output=$(cd "$output" && pwd)

base=e3d9ad5a418e3ecce8cc4732fc121e11aca17c2d
expected_dev=99290bf3bdc7a959ffeb43128623540cac4f6324
expected_branch=feature/dgsem-production-device-operator
expected_lock="DGSEM-OP-GPU-01 active; coordinator-owned until disposition"
expected_mfem=-d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec
lock=/tmp/dg_igr-sm120.lock
cmake_bin=/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/cmake
ctest_bin=/home/jarias9/spack/opt/spack/linux-x86_64_v4/cmake-3.31.11-pmpktklqoi7vhujc6hxwper32bbmw3jt/bin/ctest
cpu_mpicxx=/home/jarias9/spack/opt/spack/linux-x86_64_v4/openmpi-5.0.10-hwkz2biq5sl7zswilcmx3ipeno7vgdn5/bin/mpicxx
gpu_mpicxx=/home/jarias9/spack/opt/spack/linux-x86_64_v4/openmpi-5.0.10-5udmoissk2z7velbzl2jzsw2d7oq2xia/bin/mpicxx
nvcc=/home/jarias9/spack/opt/spack/linux-x86_64_v4/cuda-13.3.0-a3bneobuiznprbss2rgxubwpdg7yg53v/bin/nvcc
mfem_cpu=/home/jarias9/.codex/worktrees/29ed/dg_igr_miniapp/install-dgsem-ref-29ed-mfem-cpu/lib/cmake/mfem
mfem_gpu=/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/install-gpu-sm120/lib/cmake/mfem
zlib_cpu=/home/jarias9/spack/opt/spack/linux-x86_64_v4/zlib-ng-2.3.3-i64qsvxt7m2fzhmzcov3g7ser3h2z2gx
zlib_gpu=/home/jarias9/projects/dg_igr_miniapp/spack-envs/cuda-sm120/.spack-env/view
umpire_gpu=/home/jarias9/spack/opt/spack/linux-x86_64_v4/umpire-2025.12.0-utquvk3eocmtnh2uzqjzdhtgqce4jpxg/lib/cmake/umpire
cuda_root=/home/jarias9/spack/opt/spack/linux-x86_64_v4/cuda-13.3.0-a3bneobuiznprbss2rgxubwpdg7yg53v
mfem_includes="-I/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/fem -I/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/general"

gate_failures=0
lease_contention=0
printf 'label\tlease\tstarted_utc\tfinished_utc\tobserved_exit\texpectation\tgate\tcommand\n' > "$output/commands.tsv"

record_command()
{
  local label=$1 lease=$2 start=$3 finish=$4 rc=$5 expectation=$6 gate=$7 command=$8
  printf '%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n' \
    "$label" "$lease" "$start" "$finish" "$rc" "$expectation" "$gate" "$command" \
    >> "$output/commands.tsv"
}

run_logged()
{
  local label=$1
  shift
  local start finish rc command gate=pass
  printf -v command '%q ' "$@"
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  "$@" > "$output/$label.log" 2>&1
  rc=$?
  finish=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [[ $rc -ne 0 ]]; then
    gate=fail
    gate_failures=$((gate_failures + 1))
  fi
  record_command "$label" none "$start" "$finish" "$rc" zero "$gate" "$command"
}

run_leased()
{
  local label=$1
  shift
  local start finish rc command gate=pass state=acquired
  printf -v command '%q ' "$@"
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  flock -n -E 75 "$lock" "$@" > "$output/$label.log" 2>&1
  rc=$?
  finish=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [[ $rc -eq 75 ]]; then
    state=contended
    gate=contended
    lease_contention=$((lease_contention + 1))
    gate_failures=$((gate_failures + 1))
  elif [[ $rc -ne 0 ]]; then
    gate=fail
    gate_failures=$((gate_failures + 1))
  fi
  record_command "$label" "$state" "$start" "$finish" "$rc" zero "$gate" "$command"
}

run_expected_3d_guard()
{
  local probe=$1 fragment=$2
  local label="legacy-3d-${probe}" start finish rc command gate=pass
  command="env DG_IGR_DGSEM_LEGACY_3D_PROBE=$probe $cpu_build/tests/dg_igr_unit_tests --dgsem-operator"
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  env DG_IGR_DGSEM_LEGACY_3D_PROBE="$probe" \
    "$cpu_build/tests/dg_igr_unit_tests" --dgsem-operator \
    > "$output/$label.log" 2>&1
  rc=$?
  finish=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [[ $rc -eq 0 ]] ||
     ! rg -Fq "$fragment" "$output/$label.log" ||
     rg -Fq "unexpectedly passed its dimension guard" "$output/$label.log"; then
    gate=fail
    gate_failures=$((gate_failures + 1))
  fi
  record_command "$label" none "$start" "$finish" "$rc" \
    "nonzero+exact-guard" "$gate" "$command"
}

branch=$(git branch --show-current)
head=$(git rev-parse HEAD)
parent=$(git rev-parse HEAD^)
tree=$(git rev-parse HEAD^{tree})
dev=$(git rev-parse dev)
origin_dev=$(git rev-parse origin/dev)
commit_count=$(git rev-list --count "$base"..HEAD)
worktree_lock=$(git worktree list --porcelain |
  awk -v p="worktree $repo" '$0 == p {found=1} found && /^locked / {sub(/^locked /,""); print; exit}')
mfem_marker=$(git submodule status tpl/mfem | awk '{print $1}')
clean_status=$(git status --porcelain)

{
  echo "task=DGSEM-OP-GPU-01-remediation"
  echo "repo=$repo"
  echo "branch=$branch"
  echo "head=$head"
  echo "parent=$parent"
  echo "tree=$tree"
  echo "base=$base"
  echo "dev=$dev"
  echo "origin_dev=$origin_dev"
  echo "commit_count_from_base=$commit_count"
  echo "worktree_lock=$worktree_lock"
  echo "mfem_marker=$mfem_marker"
  echo "cpu_build=$cpu_build"
  echo "asan_build=$asan_build"
  echo "gpu_build=$gpu_build"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "status_begin"
  git status --short
  echo "status_end"
} > "$output/identity.log"

identity_ok=true
[[ $repo == /home/jarias9/.codex/worktrees/dgsem-production-device-operator/dg_igr_miniapp ]] || identity_ok=false
[[ $branch == "$expected_branch" ]] || identity_ok=false
[[ $parent == "$base" && $commit_count -eq 1 ]] || identity_ok=false
[[ $dev == "$expected_dev" && $origin_dev == "$expected_dev" ]] || identity_ok=false
[[ $worktree_lock == "$expected_lock" ]] || identity_ok=false
[[ $mfem_marker == "$expected_mfem" ]] || identity_ok=false
[[ -z $clean_status ]] || identity_ok=false
if ! $identity_ok; then
  echo "identity gate failed" >> "$output/identity.log"
  exit 74
fi

allowed_paths=(
  docs/dgsem/DGSEM_OP_GPU_01_COORDINATOR_REQUEST.md
  include/numerics/dgsem/dgsem_device.hpp
  include/numerics/dgsem/dgsem_operator.hpp
  include/numerics/dgsem/dgsem_production.hpp
  include/physics/igr/igr_flux.hpp
  scripts/dgsem/run_dgsem_op_gpu_01_worker_evidence.sh
  src/numerics/dgsem/dgsem_device.cpp
  src/numerics/dgsem/dgsem_operator.cpp
  src/numerics/dgsem/dgsem_production.cpp
  tests/test_dgsem_operator.cpp
  tests/test_dgsem_production_operator.hpp
)
printf '%s\n' "${allowed_paths[@]}" | sort > "$output/allowed-paths.txt"
git diff --name-only "$base"..HEAD | sort > "$output/changed-paths.txt"
run_logged exact-scope diff -u "$output/allowed-paths.txt" "$output/changed-paths.txt"
run_logged diff-check git diff --check "$base"..HEAD
run_logged clean-index git diff --exit-code
run_logged clean-index-cached git diff --cached --exit-code

while IFS= read -r path; do
  [[ -n $path ]] && sha256sum "$path"
done < "$output/changed-paths.txt" > "$output/source-sha256s.txt"

run_logged host-uname uname -a
run_logged host-lscpu lscpu
run_logged mpi-version /usr/bin/mpiexec --version
run_logged cpu-compiler-version "$cpu_mpicxx" --version
run_logged cmake-version "$cmake_bin" --version

run_logged cpu-configure "$cmake_bin" -S "$repo" -B "$cpu_build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$cpu_mpicxx" \
  -DCMAKE_CXX_FLAGS="$mfem_includes" \
  -DMFEM_DIR="$mfem_cpu" \
  -DZLIB_ROOT="$zlib_cpu" \
  -DDG_IGR_BUILD_TESTS=ON
run_logged asan-configure "$cmake_bin" -S "$repo" -B "$asan_build" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER="$cpu_mpicxx" \
  -DCMAKE_CXX_FLAGS="$mfem_includes -fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DMFEM_DIR="$mfem_cpu" \
  -DZLIB_ROOT="$zlib_cpu" \
  -DDG_IGR_BUILD_TESTS=ON
run_leased gpu-configure "$cmake_bin" -S "$repo" -B "$gpu_build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$gpu_mpicxx" \
  -DCMAKE_CUDA_COMPILER="$nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$gpu_mpicxx" \
  -DCMAKE_CUDA_ARCHITECTURES=80-virtual \
  -DCUDAToolkit_ROOT="$cuda_root" \
  -DCMAKE_CXX_FLAGS="$mfem_includes" \
  -DMFEM_DIR="$mfem_gpu" \
  -DZLIB_ROOT="$zlib_gpu" \
  -Dumpire_DIR="$umpire_gpu" \
  -DDG_IGR_BUILD_TESTS=ON \
  -DDG_IGR_REQUIRE_MFEM_CUDA=ON \
  -DDG_IGR_EXPECT_CUDA_ARCH=120

run_leased cpu-full-build "$cmake_bin" --build "$cpu_build" -j2
run_leased asan-focused-build "$cmake_bin" --build "$asan_build" \
  --target dg_igr_unit_tests -j2
run_leased gpu-focused-build "$cmake_bin" --build "$gpu_build" \
  --target dg_igr_unit_tests -j2

run_logged cpu-ctest-list "$ctest_bin" --test-dir "$cpu_build" -N
run_logged cpu-ctest-count rg -Fx "Total Tests: 46" \
  "$output/cpu-ctest-list.log"
run_logged cpu-matrix-serial "$ctest_bin" --test-dir "$cpu_build" \
  --output-on-failure -E '(mpi|parity_(ref|check)|dgsem_perf_cpu_benchmark)'
run_leased cpu-matrix-mpi timeout 1200s "$ctest_bin" --test-dir "$cpu_build" \
  --output-on-failure -R '(mpi|parity_(ref|check))' \
  -E '^dg_igr_hdg_oracle_mpi_'
# Registered HDG MPI wrappers acquire the shared lease themselves.
run_logged cpu-matrix-hdg-mpi-self-leased "$ctest_bin" \
  --test-dir "$cpu_build" --output-on-failure \
  -R '^dg_igr_hdg_oracle_mpi_[12]_cpu$'

run_logged cpu-focused env DG_IGR_TEST_QUIET=1 \
  DG_IGR_DGSEM_DEVICE_OUTPUT_DIR="$output/cpu" \
  DG_IGR_DGSEM_PRODUCTION_OUTPUT_DIR="$output/cpu" \
  "$cpu_build/tests/dg_igr_unit_tests" --dgsem-operator
run_leased cpu-focused-mpi timeout 300s env DG_IGR_TEST_QUIET=1 \
  DG_IGR_DGSEM_DEVICE_OUTPUT_DIR="$output/cpu-mpi" \
  DG_IGR_DGSEM_PRODUCTION_OUTPUT_DIR="$output/cpu-mpi" \
  /usr/bin/mpiexec -n 2 \
  "$cpu_build/tests/dg_igr_unit_tests" --dgsem-operator
run_leased cpu-split-communicator timeout 120s env \
  DG_IGR_DGSEM_SPLIT_COMM_PROBE=1 \
  /usr/bin/mpiexec -n 4 \
  "$cpu_build/tests/dg_igr_unit_tests" --dgsem-operator
for rank in 0 1 2 3; do
  if [[ $rank -lt 2 ]]; then
    split_reason=invalid-density
  else
    split_reason=output-size-mismatch
  fi
  run_logged "cpu-split-map-rank-${rank}" rg -Fx \
    "[DGSEM_SPLIT_MAP] world_rank=$rank group=$((rank / 2)) local_rank=$((rank % 2)) reason=$split_reason" \
    "$output/cpu-split-communicator.log"
done
run_logged cpu-split-cfl-group-0 rg -x \
  '\[DGSEM_SPLIT_CFL\] group=0 expected=[0-9.e+-]+ actual=[0-9.e+-]+' \
  "$output/cpu-split-communicator.log"
run_logged cpu-split-cfl-group-1 rg -x \
  '\[DGSEM_SPLIT_CFL\] group=1 expected=[0-9.e+-]+ actual=[0-9.e+-]+' \
  "$output/cpu-split-communicator.log"

run_expected_3d_guard surface \
  "DGSEM legacy surface oracle supports only 1D/2D."
run_expected_3d_guard volume \
  "DGSEM legacy volume oracle supports only 1D/2D."
run_expected_3d_guard inverse-mass \
  "DGSEM legacy inverse-mass oracle supports only 1D/2D."
run_expected_3d_guard residual \
  "DGSEM legacy residual oracle supports only 1D/2D."

run_logged asan-focused env \
  ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  DG_IGR_TEST_QUIET=1 \
  DG_IGR_DGSEM_DEVICE_OUTPUT_DIR="$output/asan" \
  DG_IGR_DGSEM_PRODUCTION_OUTPUT_DIR="$output/asan" \
  "$asan_build/tests/dg_igr_unit_tests" --dgsem-operator

run_logged cxx17 "$cpu_mpicxx" -std=c++17 -pedantic-errors -fsyntax-only \
  -I"$repo/include" \
  -I/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/fem \
  -I/home/jarias9/projects/dg_igr_miniapp/tpl/mfem/general \
  -isystem /home/jarias9/.codex/worktrees/29ed/dg_igr_miniapp/install-dgsem-ref-29ed-mfem-cpu/include \
  -isystem /home/jarias9/spack/opt/spack/linux-x86_64_v4/hypre-3.1.0-z4khky5xdjwk6hwyyvwarvh72civxjdl/include \
  -isystem /home/jarias9/spack/opt/spack/linux-x86_64_v4/metis-5.1.0-4av6w4ta7uhh223bq4gkhzapiywrvoyf/include \
  -isystem /home/jarias9/spack/opt/spack/linux-x86_64_v4/zlib-ng-2.3.3-i64qsvxt7m2fzhmzcov3g7ser3h2z2gx/include \
  src/numerics/dgsem/dgsem_device.cpp \
  src/numerics/dgsem/dgsem_production.cpp \
  src/numerics/dgsem/dgsem_operator.cpp

run_leased gpu-identity nvidia-smi \
  --query-gpu=index,name,uuid,pci.bus_id,driver_version,compute_cap,memory.total \
  --format=csv,noheader,nounits
run_leased gpu-focused timeout 300s env CUDA_VISIBLE_DEVICES=0 \
  DG_IGR_DGSEM_DEVICE=cuda DG_IGR_TEST_QUIET=1 \
  DG_IGR_DGSEM_DEVICE_OUTPUT_DIR="$output/gpu" \
  DG_IGR_DGSEM_PRODUCTION_OUTPUT_DIR="$output/gpu" \
  "$gpu_build/tests/dg_igr_unit_tests" --dgsem-operator

run_logged post-evidence-clean git diff --exit-code
run_logged post-evidence-clean-cached git diff --cached --exit-code

{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "head=$head"
  echo "tree=$tree"
  echo "parent=$parent"
  echo "lease_contention_count=$lease_contention"
  echo "gate_failure_count=$gate_failures"
  echo "validation_control_cases=$(sed -n 's/.*\"validation_control_cases\": \([0-9][0-9]*\).*/\1/p' "$output/cpu/dgsem_production_operator.json")"
  echo "direct_validation_rejection_cases=$(sed -n 's/.*\"direct_validation_rejection_cases\": \([0-9][0-9]*\).*/\1/p' "$output/cpu/dgsem_production_operator.json")"
  echo "assemble_no_write_rejection_cases=$(sed -n 's/.*\"assemble_no_write_rejection_cases\": \([0-9][0-9]*\).*/\1/p' "$output/cpu/dgsem_production_operator.json")"
  echo "collective_rejection_cases=$(sed -n 's/.*\"collective_rejection_cases\": \([0-9][0-9]*\).*/\1/p' "$output/cpu-mpi/dgsem_production_operator.json")"
} > "$output/result.log"

git show --no-ext-diff --format=fuller --stat HEAD > "$output/final-commit.log"
git status --short > "$output/final-status.log"
find "$output" -type f \
  ! -name sha256sums.txt ! -name manifest.sha256 \
  -print0 | sort -z | xargs -0 sha256sum > "$output/sha256sums.txt"
sha256sum "$output/sha256sums.txt" > "$output/manifest.sha256"

echo "evidence=$output"
cat "$output/result.log"
cat "$output/manifest.sha256"
if [[ $gate_failures -ne 0 ]]; then
  exit 1
fi
