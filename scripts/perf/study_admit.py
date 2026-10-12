#!/usr/bin/env python3
"""Compute-side admission of canonical CPU artifacts for a mixed GPU study.

Uses the canonical prebuilt receipt verifier and dependency checker. It does
not install dependencies, build, submit jobs, or execute an application.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import subprocess
from types import SimpleNamespace

import study_build


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'missing regular artifact: {path}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(source, name):
    spec = importlib.util.spec_from_file_location('study_cpu_' + name,
                                                 Path(source) / 'scripts/build' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def run(argv, env=None):
    return subprocess.run(argv, env={**os.environ, **env} if env is not None else None,
                          text=True, capture_output=True, timeout=60)


def completed_jobs(plan, receipt):
    """Bind successful owned accounting rows to the real public submission."""
    require(receipt.get('success') is True and receipt.get('returncode') == 0,
            'a successful real canonical submission receipt is required')
    if plan.get('origin') != 'existing-canonical-request':
        require(receipt.get('source_fingerprint') == plan['source']['source_fingerprint'] and
                receipt.get('configuration_key') == plan['configuration_key'],
                'canonical CPU submission source/configuration identity differs from plan')
    stdout = receipt.get('stdout', '')
    ids = {}
    for stage, key, title in (('build', 'build_job_id', 'build'),
                              ('validation', 'validation_job_id', 'validation')):
        job = str(receipt.get(key, ''))
        found = re.findall(r'^PACE ' + title + r' submitted: ([0-9]+) \(', stdout, re.M)
        require(re.fullmatch(r'[1-9][0-9]*', job) and found == [job],
                f'canonical {stage} job ID does not match submission stdout')
        ids[stage] = job
    require(len(set(ids.values())) == 2, 'build and validation job IDs must differ')
    matches = re.findall(r'^Logs: .*; request: (.+)$', stdout, re.M)
    require(len(matches) == 1, 'canonical submission must identify one public request')
    request_path = Path(matches[0])
    require(request_path.is_absolute(), 'canonical request path must be absolute')
    request_sha256 = sha(request_path)
    if plan.get('origin') == 'existing-canonical-request':
        require(str(request_path) == plan['request_path'] and sha(request_path) == plan['request_sha256'],
                'canonical CPU request changed or differs from planned request')
    request = json.loads(request_path.read_text())
    source = plan['source']['source_snapshot']
    require(request.get('source') == source, 'canonical request source differs from frozen source')
    observed = study_build.existing_cpu_plan(plan['source'], request_path)
    for key in ('app_build', 'app_install', 'mfem_source', 'mfem_build', 'mfem_install',
                'spack_env', 'spack_root', 'correctness_regex'):
        require(observed[key] == plan[key], f'canonical CPU request differs from plan: {key}')
    command = ['sacct', '-n', '-P', '-j', ','.join(ids.values()),
               '--format=JobID,State,ExitCode,User%64,Account%64,QOS%64,JobName%64,SubmitLine%4096']
    result = run(command)
    require(result.returncode == 0, 'canonical CPU accounting query failed')
    rows = {}
    for line in result.stdout.splitlines():
        row = line.split('|', 7)
        if len(row) == 8 and row[0] in ids.values():
            require(row[0] not in rows, 'ambiguous canonical CPU accounting record')
            rows[row[0]] = row
    user = pwd.getpwuid(os.getuid()).pw_name
    for stage, job in ids.items():
        row = rows.get(job, [])
        require(len(row) == 8 and row[1:7] ==
                ['COMPLETED', '0:0', user, 'gts-sbryngelson3', 'embers',
                 'IGRlab-build' if stage == 'build' else 'IGRlab-check'],
                f'canonical CPU {stage} job did not complete successfully in the owned free queue')
        tokens = shlex.split(row[7])
        worker = [str(Path(source) / 'scripts/build/pace.sh'), '--worker', stage, str(request_path)]
        require(sum(tokens[index:index + 4] == worker for index in range(len(tokens))) == 1,
                f'canonical CPU {stage} accounting belongs to a different request')
    require(sha(request_path) == request_sha256, 'canonical CPU request changed during admission')
    return {'job_ids': ids, 'accounting': rows, 'request': str(request_path),
            'request_sha256': request_sha256}


def admit_build(plan, binary_names, receipt):
    """Admit a genuine CPU-only HDG build; failures never publish a receipt."""
    require(plan.get('profile') == 'cpu', 'CPU admission requires an explicit cpu build plan')
    study_build.verify_source(plan['source'])
    jobs = completed_jobs(plan, receipt)
    source = Path(plan['source']['source_snapshot'])
    checker = module(source, 'check_dependency_bindings')
    verifier = module(source, 'prebuilt_validation')
    locations = {}
    env = study_build._environment(plan['env'])
    # Explicit -e selects the CPU concrete graph even inside a CUDA allocation.
    for package in ('hypre', 'openmpi', 'cmake', 'openblas', 'metis'):
        result = run([str(Path(plan['spack_root']) / 'bin/spack'), '-e', plan['spack_env'],
                      'location', '-i', package], env=env)
        path = Path(result.stdout.strip())
        require(result.returncode == 0 and path.is_absolute() and path.is_dir(),
                f'CPU Spack did not resolve one installed {package} prefix')
        locations[package] = str(path.resolve(strict=True))
    saved_path = Path(plan['app_build']) / verifier.RECEIPT
    saved = json.loads(saved_path.read_text())
    expected_policy = {'mode': 'cpu', 'profile': 'cpu', 'build_type': 'Release', 'site': 'pace',
                       'caliper': '0', 'nvtx_profile': '0', 'hdg_runtime': 'ON',
                       'hdg_auxiliary': 'OFF', 'hdg_euler': 'OFF', 'hdg_euler_device': 'OFF',
                       'research_tests': 'OFF', 'dependency_cuda_arch': 'none',
                       'mfem_cuda_archs': 'none', 'app_cuda_archs': 'none', 'kernel_sources': '',
                       'mfem_seed': '', 'mfem_output_root': plan['mfem_cache']['output_root']}
    require(saved.get('policy') == expected_policy, 'canonical receipt is not the planned CPU-only HDG configuration')
    args = SimpleNamespace(source=source, app_build=Path(plan['app_build']),
                           app_install=Path(plan['app_install']), spack_env=Path(plan['spack_env']),
                           cmake=Path(locations['cmake']) / 'bin/cmake',
                           mpi=Path(locations['openmpi']), hypre=Path(locations['hypre']),
                           expected_source=None, policy=[f'{key}={value}' for key, value in expected_policy.items()])
    for key in ('mfem_source', 'mfem_build', 'mfem_install'):
        require(saved.get('paths', {}).get(key) == plan[key], f'canonical CPU receipt differs from plan: {key}')
    verified = verifier.verify(args)
    mfem_cache = checker.cache_values(Path(plan['mfem_build']) / 'CMakeCache.txt')
    require(mfem_cache.get('MFEM_USE_CUDA') == 'OFF' and mfem_cache.get('MFEM_USE_HIP') in (None, 'OFF') and
            mfem_cache.get('CMAKE_BUILD_TYPE') == 'Release', 'MFEM artifact is not a Release CPU build')
    hypre, mpi = Path(locations['hypre']), Path(locations['openmpi'])
    bindings = {'cache': checker.check_cache(mfem_cache, hypre, mpi),
                'installed': checker.check_installed(Path(plan['mfem_config']), Path(plan['mfem_targets']), hypre),
                'loader': {}}
    artifacts = {path: identity['sha256'] for path, identity in verified['artifacts']['files'].items()}
    artifacts[str(saved_path)] = sha(saved_path)
    artifacts[jobs['request']] = jobs['request_sha256']
    binaries = {}
    require(binary_names and len(set(binary_names)) == len(binary_names), 'CPU binary names must be nonempty and unique')
    for name in binary_names:
        require(isinstance(name, str) and Path(name).name == name and name not in ('.', '..'), 'invalid CPU binary name')
        matches = [path for path in Path(plan['app_build']).rglob(name)
                   if path.is_file() and not path.is_symlink() and os.access(path, os.X_OK) and
                   'CMakeFiles' not in path.parts]
        require(len(matches) == 1, f'expected exactly one CPU executable: {name}')
        binary = matches[0]
        require(str(binary) in artifacts and binary.read_bytes()[:4] == b'\x7fELF',
                f'CPU executable is absent from canonical artifact receipt: {name}')
        loaded = run(['ldd', str(binary)], env=runtime_environment(locations, {}))
        require(not re.search(r'lib(?:cuda|cudart|cublas|cusparse|curand|nvrtc)', loaded.stdout + loaded.stderr),
                'CPU artifact unexpectedly links CUDA runtime libraries')
        bindings['loader'][name] = checker.check_loader_output(loaded.stdout + loaded.stderr, hypre, mpi, loaded.returncode)
        for library in re.findall(r'=>\s+(/\S+)\s+\(0x', loaded.stdout):
            resolved = Path(library).resolve(strict=True)
            artifacts[str(resolved)] = sha(resolved)
        binaries[name] = str(binary)
    launcher = (mpi / 'bin/mpiexec').resolve(strict=True)
    require(launcher.is_relative_to(mpi) and launcher.is_file() and os.access(launcher, os.X_OK),
            'CPU MPI launcher escapes selected prefix or is not executable')
    artifacts[str(launcher)] = sha(launcher)
    study_build.verify_source(plan['source'])
    return {'schema_version': 1, 'profile': 'cpu', 'source_fingerprint': plan['source']['source_fingerprint'],
            'configuration_key': plan['configuration_key'], 'binaries': binaries,
            'artifact_sha256': artifacts, 'artifact_links': verified['artifacts']['links'],
            'canonical_jobs': jobs, 'spack_locations': locations, 'dependency_bindings': bindings,
            'runtime': {'launcher': str(launcher), 'ranks': 1,
                        'environment': runtime_environment(locations, {})}}


def runtime_environment(locations, overrides):
    # Return only explicit per-child overrides, never a receipt of unrelated
    # inherited variables (which can include credentials).
    env = dict(overrides)
    bins = [str(Path(locations[name]) / 'bin') for name in ('openmpi', 'hypre', 'openblas', 'metis')]
    libs = [str(Path(locations[name]) / directory) for name in ('openmpi', 'hypre', 'openblas', 'metis')
            for directory in ('lib', 'lib64')]
    inherited_path = env.get('PATH', os.environ.get('PATH', ''))
    inherited_libs = env.get('LD_LIBRARY_PATH', os.environ.get('LD_LIBRARY_PATH', ''))
    env['PATH'] = os.pathsep.join(bins + ([inherited_path] if inherited_path else []))
    env['LD_LIBRARY_PATH'] = os.pathsep.join(libs + ([inherited_libs] if inherited_libs else []))
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def runtime_command(admission, argv, env):
    """Select CPU MPI per child; the surrounding allocation may remain CUDA."""
    require(admission.get('profile') == 'cpu' and argv and
            argv[0] in admission['binaries'].values(), 'CPU command does not use an admitted CPU binary')
    launcher = admission['runtime']['launcher']
    # The coordinator verifies the complete merged admission before and after
    # each stage. Per child, check only the selected executable and launcher;
    # hashing every test executable/library here would change cache conditions.
    for path in (argv[0], launcher):
        require(sha(path) == admission['artifact_sha256'][path],
                f'CPU runtime artifact changed after admission: {path}')
    for path, expected in admission['artifact_links'].items():
        require(Path(path).is_symlink() and os.readlink(path) == expected['link'] and
                str(Path(path).resolve(strict=True)) == expected['target'], f'CPU runtime link changed: {path}')
    return [launcher, '-n', '1', *argv], runtime_environment(admission['spack_locations'], env)
