#!/usr/bin/env python3
"""Versioned, paired GPU studies using canonical builds and compute-only profiling.

This coordinator deliberately has no application-side instrumentation or replay.
Raw measurements and unsuccessful attempts are retained in their own directories.
"""
import argparse
import contextlib
import csv
import fcntl
import getpass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import uuid

# Frozen source readers must not gain interpreter-generated files at runtime.
sys.dont_write_bytecode = True
import study_build
import study_profiles
import study_workloads

SCHEMA = 1
MEMBERS = ('baseline', 'candidate')
EXECUTOR_FILES = ('gpu_study.py', 'study_build.py', 'study_profiles.py',
                  'study_workloads.py', 'pace_runtime.sh')
SAFE_ENV = ('CUDA_VISIBLE_DEVICES', 'SLURM_JOB_ID', 'SLURM_JOB_ACCOUNT',
            'SLURM_JOB_QOS', 'SLURM_JOB_PARTITION', 'SLURM_JOB_GPUS',
            'SLURM_STEP_GPUS', 'SLURM_NTASKS', 'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(data)
        return
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x') as stream:
        stream.write(data)
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def absolute(value):
    path = Path(value)
    if not path.is_absolute() or any(ord(c) < 32 for c in str(path)):
        raise ValueError('paths must be absolute and contain no control characters')
    # A run must not be redirected into another tree by a symlink.
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError(f'symlinked path is not a study directory: {ancestor}')
    return path


@contextlib.contextmanager
def locked(root):
    lock = root / '.study.lock'
    if lock.is_symlink():
        raise ValueError('study lock must be a regular file')
    with lock.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another command is updating this study') from exc
        yield


class Runner:
    """Each subprocess has an exclusive receipt and a bounded process group."""
    def __init__(self, output, max_seconds, command_timeout=300):
        self.output = Path(output)
        self.deadline = time.monotonic() + max_seconds
        self.command_timeout = command_timeout

    def run(self, label, argv, *, cwd=None, env=None, timeout=None):
        relative = Path(label)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('command label must be confined to this attempt')
        directory = self.output / relative
        directory.mkdir(parents=True, exist_ok=False)
        remaining = self.deadline - time.monotonic()
        if remaining < 1:
            raise RuntimeError('study attempt budget exhausted')
        cwd = Path(cwd) if cwd else directory
        cwd.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy() if env is None else env
        receipt = {'argv': [str(x) for x in argv], 'cwd': str(cwd),
                   'environment': {k: environment[k] for k in SAFE_ENV if k in environment},
                   'status': 'running', 'started_unix_seconds': time.time()}
        write_json(directory / 'command.json', receipt, exclusive=True)
        started = time.monotonic()
        with (directory / 'stdout.log').open('x') as stdout, (directory / 'stderr.log').open('x') as stderr:
            process = subprocess.Popen(receipt['argv'], cwd=cwd, env=environment,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                code = process.wait(timeout=min(remaining, timeout or self.command_timeout))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
        receipt.update(returncode=code, process_wall_seconds=time.monotonic() - started,
                       status='complete' if code == 0 else 'failed',
                       stdout_sha256=sha(directory / 'stdout.log'),
                       stderr_sha256=sha(directory / 'stderr.log'))
        write_json(directory / 'result.json', receipt, exclusive=True)
        print(f'{label}: exit={code}', flush=True)
        if code:
            raise RuntimeError(f'{label} failed (exit={code}); logs: {directory}')
        return directory


def load_plan(root):
    plan = read_json(root / 'plan.json')
    expected = plan.pop('plan_sha256')
    if plan.get('schema_version') != SCHEMA or digest(plan) != expected:
        raise RuntimeError('study plan identity mismatch')
    plan['plan_sha256'] = expected
    if Path(plan['run_dir']) != root:
        raise RuntimeError('study was moved; absolute source/build identities no longer match')
    return plan


def verify(root, *, binaries=False):
    plan = load_plan(root)
    for name, expected in plan['executor_sha256'].items():
        path = root / 'executor' / name
        if path.is_symlink() or sha(path) != expected:
            raise RuntimeError(f'frozen executor changed: {path}')
    for identity in plan['sources'].values():
        study_build.verify_source(identity)
    if sha(root / 'suite.json') != plan['suite_sha256']:
        raise RuntimeError('frozen workload suite changed')
    if binaries:
        admission = read_json(root / 'admission.json')
        if admission['plan_sha256'] != plan['plan_sha256']:
            raise RuntimeError('binary admission belongs to a different study')
        for path, expected in admission['artifact_sha256'].items():
            if sha(path) != expected:
                raise RuntimeError(f'admitted build/dependency artifact changed: {path}')
    if (root / 'state.json').exists():
        for attempt in read_json(root / 'state.json')['attempts']:
            if attempt['status'] not in ('timings_complete', 'analysis_complete'):
                continue
            for relative, expected in attempt.get('artifact_sha256', {}).items():
                path = root / relative
                if not path.resolve().is_relative_to(root) or sha(path) != expected:
                    raise RuntimeError(f'completed measurement artifact changed: {path}')
    return plan


def selected_cases(suite):
    return [case for case in suite['cases'] if case.get('enabled', True)]


def cached_source(identity, cache):
    """Publish a standalone immutable snapshot at a content-addressed path."""
    cache.mkdir(parents=True, exist_ok=True)
    original = Path(identity['source_snapshot'])
    destination = cache / 'sources' / identity['source_fingerprint']
    reused = dict(identity, source_snapshot=str(destination))
    with locked(cache):
        if destination.exists():
            study_build.verify_source(reused)
            # Only the private snapshot just created for this run is discarded.
            shutil.rmtree(original)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            original.rename(destination)
            study_build.verify_source(reused)
    return reused


def create(args):
    root = absolute(args.run_dir)
    source = absolute(args.source).resolve()
    site_root = absolute(args.site_root)
    cache = absolute(args.cache_root) if args.cache_root else site_root / 'study-cache'
    spack_root = args.spack_root
    if not spack_root:
        spack_root = subprocess.check_output(['spack', 'location', '-r'], text=True, timeout=30).strip()
    spack_root = str(absolute(spack_root).resolve())
    suite = study_workloads.load_suite(args.suite)
    if args.cases:
        ids = set(args.cases.split(','))
        if not ids <= {case['id'] for case in suite['cases']}:
            raise ValueError('unknown case in --cases')
        for case in suite['cases']:
            case['enabled'] = case['id'] in ids
    if not selected_cases(suite):
        raise ValueError('select at least one workload')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'executor').mkdir()
    for name in EXECUTOR_FILES:
        shutil.copy2(Path(__file__).parent / name, root / 'executor' / name)
    write_json(root / 'suite.json', suite, exclusive=True)
    sources = {}
    builds = {}
    for member in MEMBERS:
        frozen = study_build.freeze(source, getattr(args, member), root / 'sources' / member)
        sources[member] = cached_source(frozen, cache)
        builds[member] = study_build.build_plan(sources[member], cache / 'builds',
                                              site_root, Path(spack_root), suite['correctness_regex'])
    # Same site environment policy is necessary for matched compiler/dependency comparisons.
    configs = [sha(Path(s['source_snapshot']) / 'cmake/sites/pace.json') for s in sources.values()]
    if configs[0] != configs[1]:
        raise RuntimeError('baseline and candidate must use the same PACE dependency policy')
    plan = {'schema_version': SCHEMA, 'site': 'pace', 'run_dir': str(root),
            'site_root': str(site_root), 'spack_root': spack_root, 'cache_root': str(cache),
            'sources': sources, 'builds': builds, 'suite_sha256': sha(root / 'suite.json'),
            'executor_sha256': {name: sha(root / 'executor' / name) for name in EXECUTOR_FILES},
            'created_unix_seconds': time.time()}
    plan['plan_sha256'] = digest(plan)
    write_json(root / 'plan.json', plan, exclusive=True)
    write_json(root / 'state.json', {'schema_version': SCHEMA, 'builds': {}, 'build_attempts': [], 'jobs': [],
                                   'attempts': []}, exclusive=True)
    return root, plan


def submit_stage(root, stage, dependencies=()):
    load_plan(root)
    (root / 'logs').mkdir(exist_ok=True)
    command = ['sbatch', '--parsable', '--account=gts-sbryngelson3', '--qos=embers',
               '--partition=gpu-a100', '--gres=gpu:a100:1', '--nodes=1', '--ntasks=1',
               '--cpus-per-task=8', '--mem=48G', '--time=01:00:00', '--no-requeue',
               '--kill-on-invalid-dep=yes',
               '--job-name=IGRlab-' + stage, '--output=' + str(root / 'logs' / (stage + '-%j.out')),
               '--error=' + str(root / 'logs' / (stage + '-%j.err'))]
    if dependencies:
        command += ['--dependency=afterok:' + ':'.join(str(x) for x in dependencies)]
    command += [str(root / 'executor' / 'pace_runtime.sh'), str(root / 'plan.json'),
                stage, str(root / 'executor')]
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    job_id = result.stdout.strip().split(';')[0]
    state = read_json(root / 'state.json')
    record = {'stage': stage, 'argv': command, 'returncode': result.returncode,
              'stdout': result.stdout, 'stderr': result.stderr, 'submitted_unix_seconds': time.time()}
    if result.returncode == 0 and re.fullmatch(r'[0-9]+', job_id):
        record['job_id'] = job_id
    state['jobs'].append(record)
    write_json(root / 'state.json', state)
    if 'job_id' not in record:
        raise RuntimeError(f'{stage} submission failed: {result.stderr or result.stdout}')
    print(f'PACE {stage} submitted: {job_id} (embers, gpu-a100)')
    return job_id


def submit(args):
    if os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('submit the study from the PACE login node')
    root, plan = create(args)
    if args.prepare_only:
        print(f'Prepared immutable study: {root}\nSubmit with: ./perf.sh resume --stage builds --run-dir {root}')
        return
    with locked(root):
        submit_pipeline(root, plan)
    print(f'Study: {root}\nInspect with: ./perf.sh status --run-dir {root}')


def job_state(job_id):
    result = subprocess.run(['sacct', '-n', '-P', '-j', str(job_id), '--format=JobID,State,ExitCode'],
                            text=True, capture_output=True, timeout=30)
    rows = [line.split('|') for line in result.stdout.splitlines() if line.split('|')[0] == str(job_id)]
    if result.returncode or len(rows) != 1 or len(rows[0]) < 3:
        raise RuntimeError(f'could not admit scheduler state for existing job {job_id}')
    return rows[0][1], rows[0][2]


def recover_orphan(root, job_id, expected_name, predecessors, expected_command):
    """Cancel only a recorded pending job proven to have a failed dependency."""
    result = subprocess.run(['scontrol', 'show', 'job', '-o', str(job_id)],
                            text=True, capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f'cannot inspect recorded pending job {job_id}')
    info = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)', result.stdout))
    if info.get('JobState') != 'PENDING' or info.get('Reason') != 'DependencyNeverSatisfied':
        return False
    valid = (info.get('JobId') == str(job_id) and info.get('JobName') == expected_name and
             info.get('Account') == 'gts-sbryngelson3' and info.get('QOS') == 'embers' and
             info.get('UserId', '').split('(')[0] == getpass.getuser() and
             info.get('Command') == str(expected_command))
    dependency_text = info.get('Dependency', '')
    dependency_parts = dependency_text.split(',')
    dependency_atom = r'[0-9]+(?:\((?:failed|unfulfilled|fulfilled|invalid)\))?'
    expression_valid = bool(dependency_text) and all(re.fullmatch(
        r'afterok:' + dependency_atom + r'(?::' + dependency_atom + r')*', part)
        for part in dependency_parts)
    dependencies = set(re.findall(r':([0-9]+)', dependency_text)) if expression_valid else set()
    if not valid or not expression_valid or not dependencies or not dependencies <= set(map(str, predecessors)):
        raise RuntimeError('orphan recovery withheld: recorded scheduler identity/dependency changed')
    terminal = {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
                'BOOT_FAIL', 'DEADLINE', 'PREEMPTED'}
    states = {job: job_state(job) for job in dependencies}
    if not any(status.split()[0] in terminal for status, code in states.values()):
        raise RuntimeError('orphan recovery withheld: no recorded failed predecessor')
    cancellation = subprocess.run(['scancel', str(job_id)], capture_output=True, text=True, timeout=30)
    record = {'job_id': str(job_id), 'action': 'cancel_proven_pending_orphan',
              'scheduler_identity': info, 'predecessor_states': states,
              'returncode': cancellation.returncode, 'stdout': cancellation.stdout,
              'stderr': cancellation.stderr, 'unix_seconds': time.time()}
    state = read_json(root / 'state.json')
    state.setdefault('recovery_actions', []).append(record)
    write_json(root / 'state.json', state)
    if cancellation.returncode:
        raise RuntimeError('could not release proven orphan; recovery receipt retained')
    return True


def recorded_orphans(root, plan):
    state = read_json(root / 'state.json')
    validations = [str(attempt['validation_job_id']) for attempt in state.get('build_attempts', [])
                   if attempt.get('validation_job_id')]
    timings_ids = [job['job_id'] for job in state['jobs'] if job['stage'] == 'timings' and job.get('job_id')]
    for job in state['jobs']:
        if not job.get('job_id') or job_state(job['job_id'])[0] != 'PENDING':
            continue
        recover_orphan(root, job['job_id'], 'IGRlab-' + job['stage'],
                       validations if job['stage'] == 'timings' else timings_ids,
                       root / 'executor/pace_runtime.sh')


def submit_pipeline(root, plan):
    """Build retries use the same verified cache; public submission receipts stay."""
    reused = {}
    for member in MEMBERS:
        state = read_json(root / 'state.json')
        key = plan['builds'][member]['app_build']
        if key in reused:
            state['builds'][member] = reused[key]
            state.setdefault('build_attempts', []).append({'member': member, **reused[key]})
            write_json(root / 'state.json', state)
            continue
        existing = state['builds'].get(member)
        if existing:
            job = existing.get('validation_job_id') or existing.get('build_job_id')
            if job:
                status, exit_code = job_state(job)
                if status in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'):
                    if existing.get('validation_job_id'):
                        if status != 'PENDING' or not recover_orphan(
                                root, job, 'IGRlab-check', [existing['build_job_id']],
                                Path(plan['sources'][member]['source_snapshot']) / 'scripts/build/pace.sh'):
                            reused[plan['builds'][member]['app_build']] = existing
                            continue
                    else:
                        raise RuntimeError(f'partial build submission {job} is active; retry after it finishes')
                if existing.get('validation_job_id') and (status, exit_code) == ('COMPLETED', '0:0'):
                    reused[plan['builds'][member]['app_build']] = existing
                    continue
        result = reused.get(key) or study_build.submit_build(plan['builds'][member])
        state = read_json(root / 'state.json')
        state['builds'][member] = result
        state.setdefault('build_attempts', []).append({'member': member, **result})
        write_json(root / 'state.json', state)
        if not result.get('success') or not result.get('validation_job_id'):
            raise RuntimeError('canonical build submission incomplete; receipts retained; resume builds')
        reused[key] = result
    state = read_json(root / 'state.json')
    if completed(state, 'timings'):
        raise RuntimeError('timings already complete; resume profiles for missing diagnostics')
    recorded_orphans(root, plan)
    for job in state['jobs']:
        if 'job_id' in job and job['stage'] in ('timings', 'profiles'):
            status, _ = job_state(job['job_id'])
            if status in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'):
                raise RuntimeError(f"study stage job {job['job_id']} is active; inspect status")
    dependencies = [state['builds'][m]['validation_job_id'] for m in MEMBERS]
    timing_job = submit_stage(root, 'timings', dependencies)
    submit_stage(root, 'profiles', [timing_job])


def allocation():
    required = {'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3', 'SLURM_JOB_QOS': 'embers',
                'SLURM_JOB_PARTITION': 'gpu-a100', 'SLURM_NTASKS': '1'}
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS') or any(
            os.environ.get(key) != value for key, value in required.items()):
        raise RuntimeError('GPU study requires the free, single-rank PACE A100 allocation')
    if 'login' in socket.getfqdn():
        raise RuntimeError('timing and native trace processing require a compute node')
    return study_profiles.gpu_identity()


def checked_builds(root):
    state = read_json(root / 'state.json')
    ids = [str(state['builds'][member]['validation_job_id']) for member in MEMBERS]
    result = subprocess.run(['sacct', '-n', '-P', '-j', ','.join(ids),
                             '--format=JobID,State,ExitCode'],
                            text=True, capture_output=True, timeout=30)
    states = {row[0]: row[1:] for line in result.stdout.splitlines()
              if len(row := line.split('|')) >= 3 and row[0] in ids}
    if result.returncode or any(states.get(job, [])[:2] != ['COMPLETED', '0:0'] for job in ids):
        raise RuntimeError('both canonical correctness jobs must complete successfully before measurement')


def admit(root, plan, suite):
    artifacts = {}
    binaries = {}
    locations = {}
    for package in ('hypre', 'openmpi', 'cuda'):
        command = [str(Path(plan['spack_root']) / 'bin/spack'), '-e',
                   plan['builds']['candidate']['spack_env'], 'location', '-i', package]
        locations[package] = subprocess.check_output(command, text=True, timeout=30).strip()
        if not Path(locations[package]).is_absolute():
            raise RuntimeError('Spack returned an invalid installation prefix')
    checker_path = Path(plan['sources']['candidate']['source_snapshot']) / 'scripts/build/check_dependency_bindings.py'
    spec = importlib.util.spec_from_file_location('study_dependency_bindings', checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    binding_records = {}
    for member in MEMBERS:
        build = plan['builds'][member]
        app_build = Path(build['app_build'])
        cache = app_build / 'CMakeCache.txt'
        cache_values = dict(re.findall(r'^([^/#][^:]*):[^=]*=(.*)$', cache.read_text(), re.M))
        for key, expected in {'CMAKE_BUILD_TYPE': 'Release', 'CMAKE_CUDA_ARCHITECTURES': '80',
                              'DG_IGR_OBSERVABILITY_USE_NVTX': 'ON'}.items():
            if cache_values.get(key) != expected:
                raise RuntimeError(f'build is not Release/native SM80: {cache}')
        paths = [cache, Path(build['mfem_build']) / 'dg-igr-mfem-identity-v1.txt']
        paths += [Path(build['spack_env']) / name for name in ('spack.yaml', 'spack.lock')]
        # Retain the actual MFEM configuration and library artifacts too.
        paths += [Path(build['mfem_build']) / 'CMakeCache.txt']
        paths += list(Path(build['mfem_install']).rglob('MFEMConfig.cmake'))
        paths += list(Path(build['mfem_install']).rglob('MFEMTargets*.cmake'))
        libraries = [p.resolve() for p in Path(build['mfem_install']).rglob('libmfem*') if p.is_file()]
        if not libraries:
            raise RuntimeError('installed MFEM library is missing')
        paths += libraries
        configs = list(Path(build['mfem_install']).rglob('MFEMConfig.cmake'))
        targets = list(Path(build['mfem_install']).rglob('MFEMTargets.cmake'))
        if len(configs) != 1 or len(targets) != 1:
            raise RuntimeError('ambiguous installed MFEM configuration')
        hypre = Path(locations['hypre']).resolve(strict=True)
        mpi = Path(locations['openmpi']).resolve(strict=True)
        mfem_cache = Path(build['mfem_build']) / 'CMakeCache.txt'
        binding_records[member] = {
            'cache': checker.check_cache(checker.cache_values(mfem_cache), hypre, mpi),
            'installed': checker.check_installed(configs[0], targets[0], hypre), 'loader': {}}
        binaries[member] = {}
        for name in sorted({case['binary'] for case in selected_cases(suite)}):
            matches = [p for p in app_build.rglob(name) if p.is_file() and not p.is_symlink()
                       and os.access(p, os.X_OK)]
            if len(matches) != 1:
                raise RuntimeError(f'expected one built executable {name} in {app_build}')
            binaries[member][name] = str(matches[0])
            paths.append(matches[0])
            loaded = subprocess.run(['ldd', str(matches[0])], capture_output=True, text=True, timeout=30)
            binding_records[member]['loader'][name] = checker.check_loader_output(
                loaded.stdout + loaded.stderr, hypre, mpi, loaded.returncode)
            for library in re.findall(r'=>\s+(/\S+)\s+\(0x', loaded.stdout):
                paths.append(Path(library).resolve(strict=True))
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f'missing regular build identity: {path}')
            artifacts[str(path)] = sha(path)
    value = {'schema_version': SCHEMA, 'plan_sha256': plan['plan_sha256'],
             'binaries': binaries, 'artifact_sha256': artifacts,
             'spack_locations': locations, 'dependency_bindings': binding_records}
    path = root / 'admission.json'
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError('study build identity changed after admission')
    else:
        write_json(path, value, exclusive=True)
    return value


def observe(runner, root, plan, admission, case, member, label, gpu, *, fields=False):
    workdir = runner.output / label
    binary = Path(admission['binaries'][member][case['binary']])
    source = Path(plan['sources'][member]['source_snapshot'])
    command = study_workloads.command(case, binary, source, workdir, fields=fields)
    environment = os.environ.copy()
    environment.update(study_workloads.environment(case, workdir, fields=fields))
    directory = runner.run(label, command, cwd=workdir, env=environment)
    observation = study_workloads.parse(case, directory / 'stdout.log', directory / 'stderr.log', gpu)
    check_observation_identity(observation, plan, member)
    if case['adapter'] == 'hdg' and (observation['host_export'] != fields or
            observation['profile']['enabled'] or observation['profile']['measured_step'] != -1):
        raise RuntimeError('timing/export invocation mode does not match benchmark observation')
    observation['receipt'] = str(directory / 'result.json')
    write_json(directory / 'observation.json', observation, exclusive=True)
    return observation, directory


def check_observation_identity(observation, plan, member):
    if 'build_identity' not in observation:
        return
    identity = observation['build_identity']
    for key in ('source_revision', 'source_tree', 'source_status'):
        if identity.get(key) != plan['sources'][member][key]:
            raise RuntimeError(f'benchmark executable source identity mismatch: {member}/{key}')
    if Path(identity.get('build_directory', '')) != Path(plan['builds'][member]['app_build']):
        raise RuntimeError('benchmark executable reports a different build directory')
    if Path(identity.get('cmake_cache', '')) != Path(plan['builds'][member]['app_build']) / 'CMakeCache.txt':
        raise RuntimeError('benchmark executable reports a different CMake cache')


def sample_statistics(baseline, candidate):
    if len(baseline) != len(candidate) or len(baseline) < 3 or any(
            not math.isfinite(x) or x <= 0 for x in baseline + candidate):
        raise RuntimeError('need at least three independent positive process pairs')
    changes = [100 * (1 - c / b) for b, c in zip(baseline, candidate)]
    return {'baseline_seconds': baseline, 'candidate_seconds': candidate,
            'baseline_median_seconds': statistics.median(baseline),
            'candidate_median_seconds': statistics.median(candidate),
            'paired_percent_less_time': changes,
            'paired_median_percent_less_time': statistics.median(changes),
            'paired_range_percent_less_time': [min(changes), max(changes)],
            'baseline_relative_stdev': statistics.stdev(baseline) / statistics.mean(baseline),
            'candidate_relative_stdev': statistics.stdev(candidate) / statistics.mean(candidate),
            'statistical_scope': 'independent alternating process pairs; no significance threshold'}


def hardware_snapshot(runner, label, gpu):
    fields = ('uuid', 'driver_version', 'pstate', 'clocks.sm', 'clocks.mem',
              'power.draw', 'power.limit', 'temperature.gpu', 'ecc.mode.current')
    try:
        directory = runner.run('hardware/' + label, ['nvidia-smi', '--query-gpu=' + ','.join(fields),
                                                    '--format=csv,noheader'], timeout=15)
        rows = list(csv.reader((directory / 'stdout.log').read_text().splitlines()))
        if len(rows) != 1 or len(rows[0]) != len(fields) or rows[0][0].strip() != gpu['uuid']:
            raise RuntimeError('snapshot device differs from admitted allocation')
        return {'status': 'available', 'unix_seconds': time.time(),
                'values_with_native_units': dict(zip(fields, (x.strip() for x in rows[0]))),
                'sampling_scope': 'one endpoint sample; no average or in-solver clock claim'}
    except (RuntimeError, OSError) as exc:
        return {'status': 'unavailable', 'reason': str(exc)}


def timings(root, plan, suite, admission, runner, gpu):
    result = {'schema_version': SCHEMA, 'status': 'timings_complete', 'gpu': gpu, 'cases': {}}
    for case in selected_cases(suite):
        identifier = case['id']
        reference = None
        values = {m: [] for m in MEMBERS}
        observations = []
        for index in range(suite['warmups'] + suite['repetitions']):
            is_warmup = index < suite['warmups']
            pair = index - suite['warmups']
            phases = MEMBERS if index % 2 == 0 else MEMBERS[::-1]
            for member in phases:
                label = f'{identifier}/warmup-{index}-{member}' if is_warmup else f'{identifier}/pair-{pair}-{member}'
                observation, _ = observe(runner, root, plan, admission, case, member, label, gpu)
                if reference is None:
                    reference = observation['signature']
                if observation['signature'] != reference:
                    raise RuntimeError(f'numerical policy/work changed in {identifier}')
                observations.append({'member': member, 'pair': pair, 'warmup': is_warmup, **observation})
                if not is_warmup:
                    values[member].append(observation['seconds'])
        exported = {}
        directories = {}
        for member in MEMBERS:
            exported[member], directories[member] = observe(
                runner, root, plan, admission, case, member, f'{identifier}/fields-{member}', gpu,
                fields=case['adapter'] != 'dgsem')
            if exported[member]['signature'] != reference:
                raise RuntimeError(f'exported run changed work in {identifier}')
        if case['adapter'] == 'dg':
            field_gate = study_workloads.compare_fields(case, directories['baseline'], directories['candidate'],
                                                        Path(plan['sources']['candidate']['source_snapshot']))
        elif case['adapter'] == 'hdg':
            if exported['baseline']['checksums'] != exported['candidate']['checksums']:
                raise RuntimeError(f'exported accepted-step states differ in {identifier}')
            field_gate = {'status': 'passed', 'criterion': 'within-engine exported state checksums equal'}
        else:
            field_gate = {'status': 'observations_only', 'criterion': 'kernel-only long-double errors retained'}
        result['cases'][identifier] = {'adapter': case['adapter'], 'scope': observations[0]['scope'],
                                      'statistics': sample_statistics(values['baseline'], values['candidate']),
                                      'observations': observations, 'fields': field_gate,
                                      'exported_observations': exported}
    return result


def profiles(root, plan, suite, admission, runner, gpu):
    value = {'schema_version': SCHEMA, 'status': 'analysis_complete', 'gpu': gpu,
             'scope': 'fresh matched profile cohort; profiled durations are diagnostic', 'cases': {}}
    timing_attempt = completed(read_json(root / 'state.json'), 'timings')
    timing_reference = read_json(timing_attempt['summary'])
    for case in selected_cases(suite):
        captures = {}
        for member in MEMBERS:
            directory = runner.output / case['id'] / member
            directory.mkdir(parents=True)
            binary = Path(admission['binaries'][member][case['binary']])
            source = Path(plan['sources'][member]['source_snapshot'])
            argv = study_workloads.command(case, binary, source, directory, profile=True)
            captures[member] = study_profiles.capture(case, argv, runner, directory, gpu)
            if captures[member]['status'] != 'complete':
                value['status'] = 'profiles_partial'
            else:
                # Native capture exit zero alone is insufficient for application admission.
                logs_to_validate = captures[member].get('application_logs', [])
                if not logs_to_validate:
                    raise RuntimeError('complete profiler capture has no application validation logs')
                for logs in logs_to_validate:
                    observation = study_workloads.parse(case, Path(logs['stdout']), Path(logs['stderr']), gpu)
                    check_observation_identity(observation, plan, member)
                    if case['adapter'] == 'hdg' and (observation['host_export'] or
                            not observation['profile']['enabled'] or observation['profile']['measured_step'] != 0):
                        raise RuntimeError('HDG capture did not retain selected measured step zero')
                    reference = timing_reference['cases'][case['id']]['observations'][0]['signature']
                    if observation['signature'] != reference:
                        raise RuntimeError('profiled numerical policy/work differs from unprofiled timing')
        value['cases'][case['id']] = captures
        write_json(runner.output / 'partial.json', value)
    return value


def completed(state, stage):
    return next((a for a in reversed(state['attempts']) if a['stage'] == stage and
                 a['status'] in ('timings_complete', 'analysis_complete')), None)


def worker(args):
    root = absolute(args.run_dir)
    with locked(root):
        plan = verify(root)
        suite = study_workloads.load_suite(root / 'suite.json')
        state = read_json(root / 'state.json')
        if completed(state, args.stage):
            raise RuntimeError('stage already complete; create a new study for new measurements')
        if args.stage == 'profiles' and not completed(state, 'timings'):
            raise RuntimeError('complete and admit unprofiled timings before profiling')
        gpu = allocation()
        checked_builds(root)
        admission = admit(root, plan, suite)
        verify(root, binaries=True)
        attempt_id = f"{args.stage}-{os.environ['SLURM_JOB_ID']}-{uuid.uuid4().hex[:8]}"
        output = root / 'attempts' / attempt_id
        output.mkdir(parents=True, exist_ok=False)
        attempt = {'id': attempt_id, 'stage': args.stage, 'status': 'running',
                   'job_id': os.environ['SLURM_JOB_ID'], 'gpu': gpu,
                   'plan_sha256': plan['plan_sha256'], 'directory': str(output)}
        state['attempts'].append(attempt)
        write_json(root / 'state.json', state)
        try:
            runner = Runner(output, suite.get(args.stage + '_budget_seconds', 3000),
                            suite.get('command_timeout_seconds', 300))
            attempt['hardware_before'] = hardware_snapshot(runner, 'before', gpu)
            function = timings if args.stage == 'timings' else profiles
            summary = function(root, plan, suite, admission, runner, gpu)
            attempt['hardware_after'] = hardware_snapshot(runner, 'after', gpu)
            summary['hardware_snapshots'] = {'before': attempt['hardware_before'],
                                             'after': attempt['hardware_after']}
            verify(root, binaries=True)
            write_json(output / 'summary.json', summary, exclusive=True)
            attempt.update(status=summary['status'], summary=str(output / 'summary.json'),
                           summary_sha256=sha(output / 'summary.json'),
                           artifact_sha256={str(p.relative_to(root)): sha(p) for p in output.rglob('*')
                                            if p.is_file() and not p.is_symlink() and
                                            (args.stage == 'timings' or p.suffix == '.json')})
            if summary['status'] == 'profiles_partial':
                raise RuntimeError('profiler evidence is incomplete; timings retained; resume profiles')
        except Exception as exc:
            attempt.update(status='failed' if args.stage == 'timings' else 'profiles_partial', error=str(exc))
            raise
        finally:
            attempt['finished_unix_seconds'] = time.time()
            write_json(root / 'state.json', state)
            report(root)


def resume(args):
    root = absolute(args.run_dir)
    with locked(root):
        plan = verify(root, binaries=(root / 'admission.json').exists())
        if args.stage == 'builds':
            submit_pipeline(root, plan)
            return
        state = read_json(root / 'state.json')
        if completed(state, args.stage):
            raise RuntimeError('stage already complete; completed measurements are immutable')
        if args.stage == 'profiles' and not completed(state, 'timings'):
            raise RuntimeError('profiling requires completed timings')
        checked_builds(root)
        recorded_orphans(root, plan)
        # Do not launch a duplicate when a queued/running stage still owns the study.
        for job in state['jobs']:
            if job['stage'] == args.stage and 'job_id' in job:
                result = subprocess.run(['squeue', '-h', '-j', job['job_id'], '-o', '%T'],
                                        text=True, capture_output=True, timeout=30)
                if result.returncode != 0:
                    raise RuntimeError('could not check existing Slurm jobs; resume withheld')
                if result.stdout.strip():
                    raise RuntimeError(f"stage has active Slurm job {job['job_id']}")
        job = submit_stage(root, args.stage)
        if args.stage == 'timings':
            submit_stage(root, 'profiles', [job])


def report(root):
    plan = load_plan(root)
    state = read_json(root / 'state.json')
    result = {'schema_version': SCHEMA, 'plan_sha256': plan['plan_sha256'],
              'sources': {m: {k: v for k, v in plan['sources'][m].items() if k in
                             ('source_revision', 'source_tree', 'source_status', 'source_fingerprint')}
                          for m in MEMBERS}, 'attempts': state['attempts'], 'timings': None, 'profiles': None}
    for stage in ('timings', 'profiles'):
        attempt = completed(state, stage)
        if attempt:
            if sha(attempt['summary']) != attempt['summary_sha256']:
                raise RuntimeError('completed stage summary changed')
            result[stage] = read_json(attempt['summary'])
        elif stage == 'profiles':
            latest = next((a for a in reversed(state['attempts']) if a['stage'] == stage), None)
            if latest:
                partial = Path(latest['directory']) / 'partial.json'
                if partial.exists():
                    result['profile_partial'] = read_json(partial)
    lines = ['# PACE A100 GPU study', '', f"Plan: `{plan['plan_sha256']}`", '',
             'Timing authority: independent unprofiled process pairs. Positive percentages mean less time.', '']
    if result['timings']:
        lines += ['| Case | Baseline median (s) | Candidate median (s) | Paired median less time | Paired range |',
                  '|---|---:|---:|---:|---:|']
        for name, case in result['timings']['cases'].items():
            stats = case['statistics']
            low, high = stats['paired_range_percent_less_time']
            lines.append(f"| {name} | {stats['baseline_median_seconds']:.6g} | {stats['candidate_median_seconds']:.6g} | {stats['paired_median_percent_less_time']:.3f}% | {low:.3f}% to {high:.3f}% |")
    else:
        lines.append('Timings have not completed.')
    lines += ['', 'Profile state: ' + ('analysis_complete' if result['profiles'] else
               (state['attempts'][-1]['status'] if state['attempts'] else 'not_started')),
              '', 'Native reports and trace import remain on PACE compute nodes. GPU API, kernel and copy durations overlap and are reported separately.',
              'Occupancy and counters diagnose the selected kernels; they do not establish a full-solver speedup.', '']
    write_json(root / 'report.json', result)
    (root / 'report.md').write_text('\n'.join(lines))
    return result


def collect(args):
    root = absolute(args.run_dir)
    with locked(root):
        verify(root, binaries=(root / 'admission.json').exists())
        report(root)
        destination = absolute(args.destination)
        destination.mkdir(parents=True, exist_ok=False)
        # Positive allowlist: no report extensions, SQLite, full CSV, VTK, or executables.
        for name in ('plan.json', 'suite.json', 'state.json', 'admission.json', 'report.json', 'report.md'):
            path = root / name
            if path.exists():
                if path.stat().st_size > 8 * 1024 * 1024:
                    raise RuntimeError(f'compact record exceeds size limit: {path}')
                shutil.copy2(path, destination / name)
        snapshots = root / 'scheduler-observations'
        if snapshots.exists():
            records = sorted((read_json(p) for p in snapshots.glob('*.json')),
                             key=lambda record: record['observed_unix_seconds'])
            write_json(destination / 'scheduler.json', records[-32:], exclusive=True)
        hashes = {p.name: sha(p) for p in destination.iterdir()}
        write_json(destination / 'collection.json', {'schema_version': SCHEMA,
                   'source_run_dir': str(root), 'artifact_sha256': hashes,
                   'native_reports_included': False}, exclusive=True)
    print(destination)


def status(root):
    verify(root)
    state = read_json(root / 'state.json')
    jobs = [str(build[key]) for build in state.get('build_attempts', list(state['builds'].values()))
            for key in ('build_job_id', 'validation_job_id') if build.get(key)]
    jobs += [v['job_id'] for v in state['jobs'] if 'job_id' in v]
    jobs = list(dict.fromkeys(jobs))
    if jobs:
        result = subprocess.run(['sacct', '-n', '-P', '-j', ','.join(jobs),
                                 '--format=JobID,State,ExitCode,Elapsed,NodeList'],
                                text=True, capture_output=True, timeout=30)
        print(result.stdout, end='')
        if result.returncode:
            raise RuntimeError(result.stderr)
        observations = root / 'scheduler-observations'
        observations.mkdir(exist_ok=True)
        write_json(observations / (uuid.uuid4().hex + '.json'),
                   {'observed_unix_seconds': time.time(), 'jobs': result.stdout.splitlines()}, exclusive=True)
    print(json.dumps(state['attempts'], indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    submit_parser = sub.add_parser('submit', help='freeze inputs, build/validate, time, then profile on PACE')
    submit_parser.add_argument('--site', choices=['pace'], default='pace')
    submit_parser.add_argument('--source', default=str(Path(__file__).resolve().parents[2]))
    submit_parser.add_argument('--baseline', required=True)
    submit_parser.add_argument('--candidate', default='HEAD', help='Git ref or literal working-tree')
    submit_parser.add_argument('--run-dir', required=True, help='new absolute scratch directory')
    submit_parser.add_argument('--site-root', required=True, help='shared canonical PACE dependency root')
    submit_parser.add_argument('--cache-root', help='retained source/configuration cache; default SITE_ROOT/study-cache')
    submit_parser.add_argument('--spack-root', help='optional tool root; otherwise query spack location -r')
    submit_parser.add_argument('--suite', default=str(Path(__file__).parent / 'suites/pace-a100.json'))
    submit_parser.add_argument('--cases', help='comma-separated case ids; explicit selection enables optional cases')
    submit_parser.add_argument('--prepare-only', action='store_true', help='freeze and inspect plans without submitting jobs')
    for name in ('worker', 'resume'):
        p = sub.add_parser(name, help='run or resubmit an incomplete stage without replacing completed measurements')
        p.add_argument('--run-dir', required=True)
        p.add_argument('--stage', choices=['timings', 'profiles'] + (['builds'] if name == 'resume' else []), required=True)
    for name in ('status', 'report', 'verify'):
        p = sub.add_parser(name)
        p.add_argument('--run-dir', required=True)
    p = sub.add_parser('collect', help='copy only compact summaries/provenance to an export directory')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--destination', required=True)
    args = parser.parse_args(argv)
    if args.command == 'submit':
        submit(args)
    elif args.command == 'worker':
        worker(args)
    elif args.command == 'resume':
        resume(args)
    elif args.command == 'collect':
        collect(args)
    else:
        root = absolute(args.run_dir)
        if args.command == 'verify':
            verify(root)
        elif args.command == 'status':
            status(root)
        else:
            with locked(root):
                report(root)
            print(root / 'report.md')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f'GPU study: {exc}', file=sys.stderr)
        sys.exit(1)
