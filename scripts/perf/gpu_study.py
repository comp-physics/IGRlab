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
import study_admit
import study_profiles
import study_workloads

SCHEMA = 1
MEMBERS = ('baseline', 'candidate')
EXECUTOR_FILES = ('gpu_study.py', 'study_build.py', 'study_profiles.py',
                  'study_workloads.py', 'pace_runtime.sh', 'study_admit.py')
LEGACY_EXECUTOR_FILES = tuple(name for name in EXECUTOR_FILES if name != 'study_admit.py')
CPU_CORRECTNESS_REGEX = ('^(hdg_generic_euler_cpu_rank1|'
                        'dg_igr_hdg_main_backend_cpu_igr_p(1|3)_rank1_cfl|'
                        'dg_igr_hdg_main_backend_igr_vtk_cpu_rank1|'
                        'dg_igr_dg_operator_tests|dg_igr_dgsem_operator_tests)$')
SAFE_ENV = ('CUDA_VISIBLE_DEVICES', 'SLURM_JOB_ID', 'SLURM_JOB_ACCOUNT',
            'SLURM_JOB_QOS', 'SLURM_JOB_PARTITION', 'SLURM_JOB_GPUS',
            'SLURM_STEP_GPUS', 'SLURM_NTASKS', 'SLURM_JOB_END_TIME', 'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'IGRLAB_PROFILE_NVTX', 'DG_IGR_PROFILE_NVTX',
            'DG_IGR_DGSEM_FUSED_RESIDUAL', 'DG_IGR_DGSEM_FUSED_DMMA',
            'IGRLAB_DGSEM_FUSED_RESIDUAL', 'IGRLAB_DGSEM_FUSED_DMMA') + tuple(
                prefix + name for prefix in ('IGRLAB_', 'DG_IGR_')
                for name in study_workloads.IGR_CONTROLS)
CLEANUP_RESERVE_SECONDS = 120


def wall_time(value, *, maximum_hours=1, minimum_minutes=5):
    """Admit canonical minute-aligned limits before preparing/submitting work."""
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]{2}:[0-9]{2}:00', value):
        raise ValueError('wall time must be minute-aligned HH:MM:SS')
    hours, minutes, _ = map(int, value.split(':'))
    if minutes >= 60 or not minimum_minutes * 60 <= 3600 * hours + 60 * minutes <= maximum_hours * 3600:
        raise ValueError(f'wall time must be between 00:{minimum_minutes:02d}:00 and {maximum_hours:02d}:00:00')
    return value


def stage_wall_time(plan, stage):
    if stage not in ('timings', 'profiles'):
        raise ValueError('unknown GPU study stage')
    if 'wall_time_limits' not in plan:
        return '01:00:00'  # Retained plans predate explicit limits.
    limits = plan['wall_time_limits']
    if not isinstance(limits, dict) or set(limits) != {'timings', 'profiles'}:
        raise ValueError('plan wall_time_limits must specify timings and profiles')
    return wall_time(limits[stage])


def worker_budget(plan, suite, stage, entered):
    """Bound commands by suite and allocation time, preserving cleanup time."""
    hours, minutes, seconds = map(int, stage_wall_time(plan, stage).split(':'))
    elapsed = time.monotonic() - entered
    requested = suite.get(stage + '_budget_seconds', 3000)
    if not math.isfinite(elapsed) or elapsed < 0 or type(requested) not in (int, float) or not 0 < requested <= 3000:
        raise ValueError('invalid worker elapsed time or suite budget')
    remaining = hours * 3600 + minutes * 60 + seconds - elapsed - CLEANUP_RESERVE_SECONDS
    if 'SLURM_JOB_END_TIME' in os.environ:
        raw = os.environ['SLURM_JOB_END_TIME']
        if not re.fullmatch(r'[0-9]+', raw):
            raise ValueError('SLURM_JOB_END_TIME must be an integer UNIX timestamp')
        try:
            end = float(int(raw))
        except (ValueError, OverflowError) as exc:
            raise ValueError('SLURM_JOB_END_TIME must be finite') from exc
        now = time.time()
        if not math.isfinite(end) or end <= now:
            raise RuntimeError('SLURM_JOB_END_TIME is not a finite future timestamp')
        remaining = min(remaining, end - now - CLEANUP_RESERVE_SECONDS)
    budget = min(requested, remaining)
    if budget < 1:
        raise RuntimeError('worker budget exhausted after reserving cleanup time')
    return budget


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
    for path, expected in plan.get('retry_origin', {}).get('build_artifact_sha256', {}).items():
        artifact = absolute(path)
        if not artifact.is_file() or sha(artifact) != expected:
            raise RuntimeError(f'retried completed build artifact changed: {artifact}')
    for path, expected in plan.get('timing_reuse', {}).get('artifact_sha256', {}).items():
        artifact = absolute(path)
        if not artifact.is_file() or sha(artifact) != expected:
            raise RuntimeError(f'reused timing artifact changed: {artifact}')
    if binaries:
        admission = read_json(root / 'admission.json')
        if admission['plan_sha256'] != plan['plan_sha256']:
            raise RuntimeError('binary admission belongs to a different study')
        for path, expected in admission['artifact_sha256'].items():
            if sha(path) != expected:
                raise RuntimeError(f'admitted build/dependency artifact changed: {path}')
        for cpu in admission.get('cpu_artifacts', {}).values():
            for path, expected in cpu.get('artifact_links', {}).items():
                if (not Path(path).is_symlink() or os.readlink(path) != expected['link'] or
                        str(Path(path).resolve(strict=True)) != expected['target']):
                    raise RuntimeError(f'admitted CPU artifact link changed: {path}')
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


def needs_cpu_build(suite):
    return any(case['adapter'] == 'dg' and case.get('parameters', {}).get('discretization') == 'hdg'
               and case['parameters'].get('device') == 'cpu' for case in selected_cases(suite))


def build_entries(plan):
    """CUDA plans stay unchanged; CPU artifacts are optional and independently admitted."""
    return [(group, member, plan[group][member]) for group in ('builds', 'cpu_builds')
            if group in plan for member in MEMBERS]


def validate_cuda_profile_cases(suite, *, cpu_available=False):
    """A CUDA facade cannot stand in for an enabled CPU HDG workload."""
    for case in selected_cases(suite):
        parameters = case.get('parameters', {})
        if (case['adapter'] == 'dg' and parameters.get('discretization') == 'hdg' and
                parameters.get('device') == 'cpu' and not cpu_available):
            raise ValueError('CPU HDG requires a separately admitted CPU-profile executable; '
                             'this study uses CUDA-profile builds')


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
    limits = {stage: wall_time(getattr(args, stage + '_time', '01:00:00'))
              for stage in ('timings', 'profiles')}
    validation_time = getattr(args, 'validation_time', None)
    if validation_time is not None:
        validation_time = wall_time(validation_time, maximum_hours=2, minimum_minutes=1)
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
    cpu_required = needs_cpu_build(suite)
    validate_cuda_profile_cases(suite, cpu_available=cpu_required)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'executor').mkdir()
    for name in EXECUTOR_FILES:
        shutil.copy2(Path(__file__).parent / name, root / 'executor' / name)
    write_json(root / 'suite.json', suite, exclusive=True)
    sources = {}
    builds = {}
    cpu_builds = {}
    for member in MEMBERS:
        frozen = study_build.freeze(source, getattr(args, member), root / 'sources' / member)
        sources[member] = cached_source(frozen, cache)
        builds[member] = study_build.build_plan(sources[member], cache / 'builds',
                                              site_root, Path(spack_root), suite['correctness_regex'],
                                              validation_time=validation_time)
        if cpu_required:
            cpu_builds[member] = study_build.build_plan(sources[member], cache / 'builds',
                                                      site_root, Path(spack_root), CPU_CORRECTNESS_REGEX,
                                                      validation_time=validation_time, profile='cpu')
    # Same site environment policy is necessary for matched compiler/dependency comparisons.
    configs = [sha(Path(s['source_snapshot']) / 'cmake/sites/pace.json') for s in sources.values()]
    if configs[0] != configs[1]:
        raise RuntimeError('baseline and candidate must use the same PACE dependency policy')
    plan = {'schema_version': SCHEMA, 'site': 'pace', 'run_dir': str(root),
            'site_root': str(site_root), 'spack_root': spack_root, 'cache_root': str(cache),
            'sources': sources, 'builds': builds, 'suite_sha256': sha(root / 'suite.json'),
            'wall_time_limits': limits,
            'executor_sha256': {name: sha(root / 'executor' / name) for name in EXECUTOR_FILES},
            'created_unix_seconds': time.time()}
    if cpu_builds:
        plan['cpu_builds'] = cpu_builds
    plan['plan_sha256'] = digest(plan)
    write_json(root / 'plan.json', plan, exclusive=True)
    write_json(root / 'state.json', {'schema_version': SCHEMA, 'builds': {}, 'build_attempts': [], 'jobs': [],
                                   'attempts': []}, exclusive=True)
    return root, plan


def submit_stage(root, stage, dependencies=()):
    plan = load_plan(root)
    if stage == 'timings' and plan.get('timing_reuse'):
        raise RuntimeError('profiles-only retry cannot submit new timings')
    limit = stage_wall_time(plan, stage)
    (root / 'logs').mkdir(exist_ok=True)
    command = ['sbatch', '--parsable', '--account=gts-sbryngelson3', '--qos=embers',
               '--partition=gpu-a100', '--gres=gpu:a100:1', '--nodes=1', '--ntasks=1',
               '--cpus-per-task=8', '--mem=48G', '--time=' + limit, '--no-requeue',
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
    def retain(group, member, result):
        state = read_json(root / 'state.json')
        state.setdefault(group, {})[member] = result
        state.setdefault('build_attempts', []).append({'member': member, 'group': group, **result})
        write_json(root / 'state.json', state)

    def admit_existing(group, member, existing):
        result = study_build.ensure_validation_time(plan[group][member], existing)
        retain(group, member, result)
        if not result.get('success') or not result.get('validation_job_id'):
            raise RuntimeError('requested correctness limit not admitted; receipts retained; resume builds')
        return result

    for group, member, build in build_entries(plan):
        state = read_json(root / 'state.json')
        key = build['app_build']
        if key in reused:
            state.setdefault(group, {})[member] = reused[key]
            state.setdefault('build_attempts', []).append({'member': member, 'group': group, **reused[key]})
            write_json(root / 'state.json', state)
            continue
        existing = state.get(group, {}).get(member)
        if existing:
            job = existing.get('validation_job_id') or existing.get('build_job_id')
            if job:
                status, exit_code = job_state(job)
                if status in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'):
                    if existing.get('validation_job_id'):
                        if status != 'PENDING' or not recover_orphan(
                                root, job, 'IGRlab-check', [existing['build_job_id']],
                                Path(plan['sources'][member]['source_snapshot']) / 'scripts/build/pace.sh'):
                            reused[key] = admit_existing(group, member, existing)
                            continue
                    else:
                        raise RuntimeError(f'partial build submission {job} is active; retry after it finishes')
                if existing.get('validation_job_id') and (status, exit_code) == ('COMPLETED', '0:0'):
                    reused[key] = admit_existing(group, member, existing)
                    continue
        result = reused.get(key) or study_build.submit_build(build)
        retain(group, member, result)
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
    dependencies = list(dict.fromkeys(state[group][member]['validation_job_id']
                                     for group, member, _ in build_entries(plan)))
    timing_job = submit_stage(root, 'timings', dependencies)
    submit_stage(root, 'profiles', [timing_job])


def allocation():
    required = {'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3', 'SLURM_JOB_QOS': 'embers',
                'SLURM_JOB_PARTITION': 'gpu-a100', 'SLURM_NTASKS': '1'}
    if not os.environ.get('SLURM_JOB_ID') or not (
            os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS')) or any(
            os.environ.get(key) != value for key, value in required.items()):
        raise RuntimeError('GPU study requires the free, single-rank PACE A100 allocation')
    if 'login' in socket.getfqdn():
        raise RuntimeError('timing and native trace processing require a compute node')
    return study_profiles.gpu_identity()


def checked_builds(root):
    state = read_json(root / 'state.json')
    ids = [str(state['builds'][member]['validation_job_id']) for member in MEMBERS]
    plan = load_plan(root)
    if 'cpu_builds' in plan:
        if set(state.get('cpu_builds', {})) != set(MEMBERS):
            raise RuntimeError('both canonical CPU correctness jobs must be recorded before measurement')
        ids += [str(state['cpu_builds'][member]['validation_job_id']) for member in MEMBERS]
    result = subprocess.run(['sacct', '-n', '-P', '-j', ','.join(ids),
                             '--format=JobID,State,ExitCode'],
                            text=True, capture_output=True, timeout=30)
    states = {row[0]: row[1:] for line in result.stdout.splitlines()
              if len(row := line.split('|')) >= 3 and row[0] in ids}
    if result.returncode or any(states.get(job, [])[:2] != ['COMPLETED', '0:0'] for job in ids):
        raise RuntimeError('all canonical correctness jobs must complete successfully before measurement')


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
        # CMake caches contain blank lines and comments. Parse one line at a
        # time with the canonical reader; a multiline negated character class
        # can consume newlines/comments into the next key.
        cache_values = checker.cache_values(cache)
        for key, expected in {'CMAKE_BUILD_TYPE': 'Release', 'CMAKE_CUDA_ARCHITECTURES': '80',
                              'DG_IGR_OBSERVABILITY_USE_NVTX': 'ON'}.items():
            if cache_values.get(key) != expected:
                raise RuntimeError(f'build configuration mismatch: {key}={cache_values.get(key)!r}, '
                                   f'expected {expected!r}: {cache}')
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
    if 'cpu_builds' in plan:
        state = read_json(root / 'state.json')
        names = sorted({case['binary'] for case in selected_cases(suite)
                        if case.get('parameters', {}).get('device') == 'cpu'})
        value['cpu_artifacts'] = {}
        for member in MEMBERS:
            cpu = study_admit.admit_build(plan['cpu_builds'][member], names,
                                          state['cpu_builds'][member])
            for artifact, expected in cpu['artifact_sha256'].items():
                if artifact in artifacts and artifacts[artifact] != expected:
                    raise RuntimeError('CPU and CUDA artifact receipts disagree')
                artifacts[artifact] = expected
            value['cpu_artifacts'][member] = cpu
    path = root / 'admission.json'
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError('study build identity changed after admission')
    else:
        write_json(path, value, exclusive=True)
    return value


def observe(runner, root, plan, admission, case, member, label, gpu, *, fields=False):
    workdir = runner.output / label
    cpu = (admission.get('cpu_artifacts', {}).get(member)
           if case.get('parameters', {}).get('device') == 'cpu' else None)
    if (case.get('parameters', {}).get('device') == 'cpu' and
            case.get('parameters', {}).get('discretization') == 'hdg' and cpu is None):
        raise RuntimeError('CPU HDG requires its independently admitted CPU artifact')
    binary = Path(cpu['binaries'][case['binary']] if cpu else
                  admission['binaries'][member][case['binary']])
    source = Path(plan['sources'][member]['source_snapshot'])
    command = study_workloads.command(case, binary, source, workdir, fields=fields)
    environment = os.environ.copy()
    environment.update(study_workloads.environment(case, workdir, fields=fields))
    if cpu:
        command, overrides = study_admit.runtime_command(cpu, command, environment)
        environment.update(overrides)
    directory = runner.run(label, command, cwd=workdir, env=environment)
    timing_metadata = {}
    if case['adapter'] == 'dg' and case['parameters'].get('discretization') == 'hdg':
        timing_metadata['wall_seconds'] = read_json(directory / 'result.json')['process_wall_seconds']
    observation = study_workloads.parse(case, directory / 'stdout.log', directory / 'stderr.log', gpu,
                                       **timing_metadata)
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
    cpu_info = Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').is_file() else ''
    cpu_models = sorted(set(re.findall(r'^model name\s*:\s*(.+)$', cpu_info, re.M)))
    result['host'] = {'hostname': socket.getfqdn(), 'cpu_models': cpu_models,
                      'allowed_logical_cpus': sorted(os.sched_getaffinity(0)),
                      'thread_environment': {k: os.environ.get(k) for k in
                          ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                      'scope': 'host CPUs of this A100 allocation; process affinity inherited'}
    field_directories = {}
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
        if case['adapter'] == 'dg':
            field_directories[identifier] = directories
    # Compare backends only within the same discretization and numerical workload.
    # The source-pair comparison above remains independent of these CPU references.
    groups = {}
    for case in selected_cases(suite):
        if case['adapter'] != 'dg' or 'device' not in case['parameters']:
            continue
        policy = {k: v for k, v in case['parameters'].items()
                  if k not in ('device', 'engine', 'dgsem_residual')}
        groups.setdefault(digest(policy), []).append(case)
    comparisons = []
    for cases in groups.values():
        references = [c for c in cases if c['parameters']['device'] == 'cpu']
        if len(references) > 1:
            raise RuntimeError('ambiguous CPU reference for backend comparison')
        if not references:
            continue
        reference_case = references[0]
        for case in cases:
            if case is reference_case:
                continue
            for member in MEMBERS:
                reference_signature = result['cases'][reference_case['id']]['exported_observations'][member]['signature']
                signature = result['cases'][case['id']]['exported_observations'][member]['signature']
                if (reference_signature['work'] != signature['work'] or
                        reference_signature.get('final_time') != signature.get('final_time')):
                    raise RuntimeError(f"accepted work/horizon differs across CPU and {case['id']}")
                gate = study_workloads.compare_fields(case,
                    field_directories[reference_case['id']][member], field_directories[case['id']][member],
                    Path(plan['sources'][member]['source_snapshot']))
                comparisons.append({'reference_case': reference_case['id'], 'case': case['id'],
                                    'member': member, 'accepted_work_equal': True, 'physical_fields': gate,
                                    'scope': 'same accepted horizon; internal HDG solver iteration counts may differ'})
    if groups:
        result['backend_comparisons'] = comparisons
    return result


def admit_reused_timing_gpu(plan, gpu):
    expected = plan.get('timing_reuse', {}).get('gpu')
    if expected and any(gpu.get(key) != expected.get(key)
                        for key in ('uuid', 'name', 'compute_capability')):
        raise RuntimeError('profiles-only retry requires the original timing GPU UUID/model')


def profiles(root, plan, suite, admission, runner, gpu):
    admit_reused_timing_gpu(plan, gpu)
    value = {'schema_version': SCHEMA, 'status': 'analysis_complete', 'gpu': gpu,
             'scope': 'fresh matched profile cohort; profiled durations are diagnostic', 'cases': {}}
    timing_attempt = completed(read_json(root / 'state.json'), 'timings')
    timing_reference = read_json(timing_attempt['summary'])
    def save_partial():
        write_json(runner.output / 'partial.json', {**value, 'status': 'profiles_partial'})

    def validate_application(case, member, record):
        # Native capture exit zero alone is insufficient for application admission.
        logs_to_validate = record.get('application_logs', [])
        if not logs_to_validate:
            raise RuntimeError('complete profiler capture has no application validation logs')
        for logs in logs_to_validate:
            timing_metadata = {}
            if case['adapter'] == 'dg' and case['parameters'].get('discretization') == 'hdg':
                timing_metadata['wall_seconds'] = read_json(logs['receipt'])['process_wall_seconds']
            observation = study_workloads.parse(case, Path(logs['stdout']), Path(logs['stderr']), gpu,
                                               **timing_metadata)
            check_observation_identity(observation, plan, member)
            if case['adapter'] == 'hdg' and (observation['host_export'] or
                    not observation['profile']['enabled'] or observation['profile']['measured_step'] != 0):
                raise RuntimeError('HDG capture did not retain selected measured step zero')
            reference = timing_reference['cases'][case['id']]['observations'][0]['signature']
            if observation['signature'] != reference:
                raise RuntimeError('profiled numerical policy/work differs from unprofiled timing')
        return len(logs_to_validate)

    for case in selected_cases(suite):
        captures = {}
        if case['adapter'] == 'dg' and case['parameters'].get('device') == 'cpu':
            value['cases'][case['id']] = {member: {
                'status': 'not_applicable', 'reason': 'CPU execution; CUDA occupancy and counters do not apply',
                'scope': 'CPU timings and physical fields retained; CPU hardware counters not collected'}
                for member in MEMBERS}
            save_partial()
            continue
        value['cases'][case['id']] = captures
        for member in MEMBERS:
            directory = runner.output / case['id'] / member
            directory.mkdir(parents=True)
            binary = Path(admission['binaries'][member][case['binary']])
            source = Path(plan['sources'][member]['source_snapshot'])
            argv = study_workloads.command(case, binary, source, directory, profile=True)
            environment = os.environ.copy()
            environment.update(study_workloads.environment(case, directory, profile=True))
            def checkpoint(record):
                record['application_validation'] = {'status': 'pending',
                    'reason': 'capture logs must still match admitted unprofiled numerical policy/work'}
                captures[member] = record
                save_partial()

            try:
                captures[member] = study_profiles.capture(case, argv, runner, directory, gpu,
                    environment=environment, checkpoint=checkpoint)
            except Exception as error:
                captures.setdefault(member, {'status': 'failed', 'application_validation': {'status': 'pending'}})
                captures[member]['error'] = str(error)
                save_partial()
                raise
            if captures[member]['status'] != 'complete':
                value['status'] = 'profiles_partial'
                captures[member]['application_validation'] = {'status': 'pending',
                    'reason': 'incomplete profiler capture; application admission not established'}
            else:
                try:
                    log_count = validate_application(case, member, captures[member])
                except Exception as error:
                    captures[member]['status'] = 'partial'
                    captures[member]['application_validation'] = {'status': 'failed', 'error': str(error)}
                    save_partial()
                    raise
                captures[member]['application_validation'] = {'status': 'complete',
                    'validated_log_count': log_count, 'scope': 'same numerical policy/work as unprofiled timing'}
            save_partial()
    return value


def completed(state, stage):
    return next((a for a in reversed(state['attempts']) if a['stage'] == stage and
                 a['status'] in ('timings_complete', 'analysis_complete')), None)


def worker(args):
    entered = time.monotonic()
    root = absolute(args.run_dir)
    with locked(root):
        plan = verify(root)
        suite = study_workloads.load_suite(root / 'suite.json')
        state = read_json(root / 'state.json')
        if completed(state, args.stage):
            raise RuntimeError('stage already complete; create a new study for new measurements')
        if args.stage == 'profiles' and not completed(state, 'timings'):
            raise RuntimeError('complete and admit unprofiled timings before profiling')
        worker_budget(plan, suite, args.stage, entered)  # Fail before queries/captures if time is exhausted.
        gpu = allocation()
        if args.stage == 'timings' and plan.get('timing_reuse'):
            raise RuntimeError('profiles-only retry cannot run new timings')
        admit_reused_timing_gpu(plan, gpu)
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
            budget = worker_budget(plan, suite, args.stage, entered)
            attempt.update(worker_budget_seconds=budget, cleanup_reserve_seconds=CLEANUP_RESERVE_SECONDS,
                           requested_wall_time=stage_wall_time(plan, args.stage))
            runner = Runner(output, budget,
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


def retry_build_artifacts(plan, suite, old_admission=None):
    """Freeze completed build bytes; compute admission still proves live bindings."""
    paths = set((old_admission or {}).get('artifact_sha256', {}))
    for group, _, build in build_entries(plan):
        app, mfem, install, env = map(Path, (build['app_build'], build['mfem_build'],
                                           build['mfem_install'], build['spack_env']))
        paths.update(map(str, (app / 'CMakeCache.txt', mfem / 'CMakeCache.txt',
                              mfem / 'dg-igr-mfem-identity-v1.txt',
                              env / 'spack.yaml', env / 'spack.lock')))
        for name in ('MFEMConfig.cmake', 'MFEMTargets.cmake'):
            matches = list(install.rglob(name))
            if len(matches) != 1:
                raise RuntimeError(f'missing or ambiguous completed MFEM configuration: {name}')
            paths.add(str(matches[0]))
        libraries = [p for p in install.rglob('libmfem*') if p.is_file()]
        if not libraries:
            raise RuntimeError('completed MFEM library is missing')
        paths.update(map(str, libraries))
        paths.update(str(p) for p in install.rglob('MFEMTargets-*.cmake'))
        cases = [case for case in selected_cases(suite) if group == 'builds' or
                 case.get('parameters', {}).get('device') == 'cpu']
        for name in {case['binary'] for case in cases}:
            matches = [p for p in app.rglob(name) if p.is_file() and os.access(p, os.X_OK)]
            if len(matches) != 1:
                raise RuntimeError(f'missing or ambiguous completed executable: {name}')
            paths.add(str(matches[0]))
    result = {}
    for value in sorted(paths):
        path = absolute(value)
        if not path.is_file():
            raise RuntimeError(f'missing regular completed build artifact: {path}')
        result[value] = sha(path)
    return result


def retained_timing(plan, state, old):
    if any(attempt.get('status') == 'running' for attempt in state['attempts']):
        raise RuntimeError('profiles-only retry withheld: original worker is running')
    attempt = completed(state, 'timings')
    if not attempt or not attempt.get('summary_sha256') or not attempt.get('artifact_sha256'):
        raise RuntimeError('profiles-only retry requires completed, fingerprinted timings')
    summary = absolute(attempt['summary'])
    if not summary.resolve().is_relative_to(old) or sha(summary) != attempt['summary_sha256']:
        raise RuntimeError('completed timing summary changed')
    timing = read_json(summary)
    original_suite = study_workloads.load_suite(old / 'suite.json')
    if set(timing.get('cases', {})) != {case['id'] for case in selected_cases(original_suite)}:
        raise RuntimeError('completed timing cases differ from the exact original suite')
    gpu = timing.get('gpu', {})
    if (timing.get('status') != 'timings_complete' or not isinstance(gpu.get('uuid'), str) or
            not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', gpu['uuid']) or
            'A100' not in gpu.get('name', '') or gpu.get('compute_capability') != 80):
        raise RuntimeError('completed timings lack the original A100 identity')
    if attempt.get('gpu') and any(attempt['gpu'].get(key) != gpu.get(key)
                                  for key in ('uuid', 'name', 'compute_capability')):
        raise RuntimeError('timing attempt and summary GPU identities differ')
    artifacts = {}
    if len(attempt['artifact_sha256']) > 10000:
        raise RuntimeError('timing artifact manifest exceeds bound')
    for relative, expected in attempt['artifact_sha256'].items():
        path = absolute(old / relative)
        if not path.resolve().is_relative_to(old) or not path.is_file() or sha(path) != expected:
            raise RuntimeError('completed timing artifact changed or escaped')
        artifacts[str(path)] = expected
    if artifacts.get(str(summary)) != attempt['summary_sha256']:
        raise RuntimeError('timing artifact manifest does not bind the summary')
    artifacts.update(plan.get('timing_reuse', {}).get('artifact_sha256', {}))
    return {'run_dir': str(old), 'plan_sha256': plan['plan_sha256'],
            'suite_sha256': plan.get('timing_reuse', {}).get('suite_sha256', plan['suite_sha256']),
            'attempt': attempt,
            'summary': str(summary), 'summary_sha256': attempt['summary_sha256'],
            'artifact_sha256': artifacts, 'gpu': gpu,
            'scope': 'original unprofiled timings reused byte-for-byte; no new timing observations'}


def admit_interactive_timing_job(job, state, retained):
    """A live user-owned shell may retain finished measurements, never a batch worker."""
    if (job != str(retained['attempt'].get('job_id')) or
            any(str(record.get('job_id')) == job for record in state.get('jobs', []))):
        raise RuntimeError('profiles-only retry withheld: recorded batch stage is active')
    query = subprocess.run(['scontrol', 'show', 'job', '-o', job], text=True,
                           capture_output=True, timeout=30)
    info = dict(re.findall(r'(\w+)=([^\s]+)', query.stdout))
    user = re.fullmatch(r'([^()]+)\(([0-9]+)\)', info.get('UserId', ''))
    gpu_tres = re.findall(r'(?:^|,)gres/gpu=([0-9]+)(?:,|$)', info.get('AllocTRES', ''))
    if (query.returncode or not user or user[1] != getpass.getuser() or int(user[2]) != os.getuid() or
            info.get('JobId') != job or info.get('JobState') != 'RUNNING' or
            info.get('BatchFlag') != '0' or info.get('Account') != 'gts-sbryngelson3' or
            info.get('QOS') != 'embers' or info.get('Partition') != 'gpu-a100' or
            info.get('NumTasks') != '1' or gpu_tres != ['1']):
        raise RuntimeError('profiles-only retry withheld: active job is not the owned free one-A100 interactive allocation')
    return info


def profile_only_overrides(original, replacement):
    """Profile selection may change; every other frozen suite field must match."""
    old = study_workloads.load_suite(original)
    new = study_workloads.load_suite(replacement)
    def without_profiles(suite):
        return {**suite, 'cases': [{key: value for key, value in case.items() if key != 'profile'}
                                   for case in suite['cases']]}
    if digest(without_profiles(old)) != digest(without_profiles(new)):
        raise ValueError('profiles-only --suite may change only existing case.profile subtrees')
    return {before['id']: {'original': before['profile'], 'replacement': after['profile']}
            for before, after in zip(old['cases'], new['cases'])
            if digest(before['profile']) != digest(after['profile'])}


def retry(args):
    """A fresh executor/cohort may reuse only inactive, verified completed builds."""
    if os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('prepare or submit a retry from the PACE login node')
    old, root = absolute(args.from_run).resolve(strict=True), absolute(args.run_dir).resolve()
    with locked(old):
        plan = verify(old, binaries=(old / 'admission.json').exists())
        state = read_json(old / 'state.json')
        if set(plan['executor_sha256']) not in (set(EXECUTOR_FILES), set(LEGACY_EXECUTOR_FILES)):
            raise RuntimeError('old run lacks the complete frozen executor identity')
        old_suite = study_workloads.load_suite(old / 'suite.json')
        suite_path = getattr(args, 'suite', None)
        cases = getattr(args, 'cases', None)
        profiles_only = bool(getattr(args, 'profiles_only', False))
        if profiles_only and cases is not None:
            raise ValueError('profiles-only retry forbids --cases; timing workload selection must match')
        retained = retained_timing(plan, state, old) if profiles_only else None
        supplied_profile_suite_sha256 = (sha(suite_path)
                                        if profiles_only and suite_path is not None else None)
        suite = study_workloads.load_suite(suite_path) if suite_path is not None else old_suite
        profile_overrides = (profile_only_overrides(old / 'suite.json', suite_path)
                             if profiles_only and suite_path is not None else {})
        if supplied_profile_suite_sha256 is not None and sha(suite_path) != supplied_profile_suite_sha256:
            raise ValueError('supplied profile suite changed during validation')
        if suite['correctness_regex'] != old_suite['correctness_regex']:
            raise ValueError('retry cannot change correctness_regex; completed validation must match')
        if cases is not None:
            names = cases.split(',')
            ids = set(names)
            if not names or any(not name for name in names) or len(ids) != len(names) or not ids <= {
                    case['id'] for case in suite['cases']}:
                raise ValueError('retry --cases requires unique known case IDs')
            for case in suite['cases']:
                case['enabled'] = case['id'] in ids
        if not selected_cases(suite):
            raise ValueError('retry requires at least one enabled workload')
        validate_cuda_profile_cases(suite, cpu_available='cpu_builds' in plan)
        protected = [old, Path(plan['cache_root']), Path(__file__).resolve().parents[2]]
        protected += [Path(s['source_snapshot']) for s in plan['sources'].values()]
        protected += [Path(b[key]) for _, _, b in build_entries(plan) for key in
                      ('app_build', 'app_install', 'mfem_build', 'mfem_install', 'spack_env')]
        if any(root.resolve().is_relative_to(p.resolve()) for p in protected):
            raise ValueError('retry destination is inside an old run, source, or protected cache')
        if root.exists():
            raise ValueError('retry run directory must be new')
        for group, member, build in build_entries(plan):
            receipt = state.get(group, {}).get(member, {})
            if (receipt.get('success') is not True or receipt.get('returncode') != 0 or
                    receipt.get('source_fingerprint') != plan['sources'][member]['source_fingerprint'] or
                    receipt.get('configuration_key') != build['configuration_key'] or
                    receipt.get('argv') != build['argv'] or not receipt.get('build_job_id')):
                raise RuntimeError('retry requires matching successful canonical build receipts')
        ids = set()
        def collect_ids(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ('job_id', 'build_job_id', 'validation_job_id',
                               'build_job_ids', 'validation_job_ids') and item is not None:
                        for job in item if isinstance(item, list) else [item]:
                            if not isinstance(job, str) or not re.fullmatch(r'[0-9]+', job):
                                raise RuntimeError('invalid recorded scheduler job identity')
                            ids.add(job)
                    else:
                        collect_ids(item)
            elif isinstance(value, list):
                for item in value: collect_ids(item)
        collect_ids(state)
        if not ids or len(ids) > 128:
            raise RuntimeError('missing or excessive recorded scheduler jobs')
        # Completed IDs can disappear from squeue while remaining in sacct.
        queue = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%T'],
                               text=True, capture_output=True, timeout=30)
        if queue.returncode:
            raise RuntimeError('retry withheld: could not prove every recorded job inactive')
        active = set()
        for row in queue.stdout.splitlines():
            match = re.fullmatch(r'([0-9]+)(?:[_+][^|]+)?\|[^|]+', row.strip())
            if not match:
                raise RuntimeError('retry withheld: could not prove every recorded job inactive')
            if match[1] in ids:
                active.add(match[1])
        if active and not profiles_only:
            raise RuntimeError('retry withheld: could not prove every recorded job inactive')
        outcomes = {job: job_state(job) for job in sorted(ids)}
        terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
                    'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED'}
        interactive = {}
        for job, (status, _) in outcomes.items():
            if status.split()[0] not in terminal:
                if not profiles_only or status != 'RUNNING' or job not in active:
                    raise RuntimeError('retry withheld: recorded scheduler state is not terminal')
                interactive[job] = admit_interactive_timing_job(job, state, retained)
            elif job in active:
                raise RuntimeError('retry withheld: scheduler active/accounting states disagree')
        checked_builds(old)
        if any(outcomes[state[group][member]['build_job_id']] != ('COMPLETED', '0:0')
               for group, member, _ in build_entries(plan)):
            raise RuntimeError('retry requires completed successful canonical builds')
        artifacts = retry_build_artifacts(plan, suite, read_json(old / 'admission.json')
                                         if (old / 'admission.json').exists() else None)
        limits = {stage: wall_time(getattr(args, stage + '_time', None) or stage_wall_time(plan, stage))
                  for stage in ('timings', 'profiles')}
        new_plan = {k: v for k, v in plan.items() if k != 'plan_sha256'}
        new_plan.pop('timing_reuse', None)
        new_plan.update(run_dir=str(root), created_unix_seconds=time.time(), wall_time_limits=limits,
                        retry_origin={'run_dir': str(old), 'plan_sha256': plan['plan_sha256'],
                                      'state_sha256': sha(old / 'state.json'),
                                      'executor_sha256': plan['executor_sha256'],
                                      'suite_sha256': plan['suite_sha256'], 'scheduler_outcomes': outcomes,
                                      'build_artifact_sha256': artifacts,
                                      'binding_scope': 'fresh compute admission required'})
        if profiles_only:
            new_plan['timing_reuse'] = {**retained, 'interactive_scheduler_admission': interactive}
        root.mkdir(parents=True, exist_ok=False)
        (root / 'executor').mkdir()
        for name in EXECUTOR_FILES:
            shutil.copy2(Path(__file__).parent / name, root / 'executor' / name)
        if profiles_only or suite_path is None and cases is None:
            shutil.copy2(suite_path if suite_path is not None else old / 'suite.json', root / 'suite.json')
        else:
            write_json(root / 'suite.json', suite, exclusive=True)
        new_plan['suite_sha256'] = sha(root / 'suite.json')
        if (supplied_profile_suite_sha256 is not None and
                new_plan['suite_sha256'] != supplied_profile_suite_sha256):
            raise ValueError('supplied profile suite changed before freezing')
        if profiles_only:
            new_plan['timing_reuse'].update(profile_suite_sha256=new_plan['suite_sha256'],
                                            profile_overrides=profile_overrides)
        new_plan['executor_sha256'] = {name: sha(root / 'executor' / name) for name in EXECUTOR_FILES}
        new_plan['plan_sha256'] = digest(new_plan)
        write_json(root / 'plan.json', new_plan, exclusive=True)
        attempts = []
        if profiles_only:
            directory = root / 'attempts' / 'timings-reused'
            directory.mkdir(parents=True)
            shutil.copy2(retained['summary'], directory / 'summary.json')
            attempts = [{**retained['attempt'], 'id': 'timings-reused', 'directory': str(directory),
                         'summary': str(directory / 'summary.json'), 'plan_sha256': new_plan['plan_sha256'],
                         'artifact_sha256': {'attempts/timings-reused/summary.json': retained['summary_sha256']},
                         'timings_reused': True, 'reused_from': {'run_dir': str(old),
                             'summary': retained['summary'], 'summary_sha256': retained['summary_sha256']}}]
        new_state = {'schema_version': SCHEMA, 'builds': state['builds'],
                     'build_attempts': [{'member': member, 'group': group, **state[group][member]}
                                        for group, member, _ in build_entries(plan)],
                     'jobs': [], 'attempts': attempts}
        if 'cpu_builds' in plan:
            new_state['cpu_builds'] = state['cpu_builds']
        write_json(root / 'state.json', new_state, exclusive=True)
    with locked(root):
        verify(root)
        if not args.prepare_only:
            if profiles_only:
                submit_stage(root, 'profiles')
            else:
                timing = submit_stage(root, 'timings')
                submit_stage(root, 'profiles', [timing])
    scope = 'Original timings reused; fresh profiles only' if profiles_only else 'Fresh timing/profile samples'
    print(f'Prepared retry: {root}\n{scope}; completed builds reused from {old}')


def comparison_description(plan):
    fingerprints = [plan['sources'][member].get('source_fingerprint') for member in MEMBERS]
    if all(isinstance(value, str) and value for value in fingerprints):
        if fingerprints[0] == fingerprints[1]:
            return ('identical-source-control',
                    'identical-source workflow/repeatability control; paired variation does not measure an optimization change')
        return 'baseline-candidate', 'baseline/candidate source comparison'
    return 'identity-unavailable', 'source fingerprints unavailable in the retained plan'


def control_repeatability(timing):
    """Describe the observed control pairs without inventing a noise threshold."""
    result = {'status': 'timings_pending' if timing is None else 'timings_complete',
              'scope': 'observed variation within this allocation; not a confidence interval or a portable acceptance threshold',
              'cases': {}}
    if timing is None:
        return result
    for name, case in timing['cases'].items():
        # Recompute from independent process observations, never accepted steps
        # or a profiler duration. Retained summaries already contain these arrays.
        values = case['statistics']
        stats = sample_statistics(values['baseline_seconds'], values['candidate_seconds'])
        changes = stats['paired_percent_less_time']
        result['cases'][name] = {
            'independent_pairs': len(changes),
            'paired_median_offset_percent': stats['paired_median_percent_less_time'],
            'paired_range_percent': stats['paired_range_percent_less_time'],
            'median_absolute_paired_offset_percent': statistics.median(map(abs, changes)),
            'maximum_absolute_paired_offset_percent': max(map(abs, changes)),
            'baseline_coefficient_of_variation_percent': 100 * stats['baseline_relative_stdev'],
            'candidate_coefficient_of_variation_percent': 100 * stats['candidate_relative_stdev']}
    return result


def report(root):
    plan = load_plan(root)
    state = read_json(root / 'state.json')
    comparison_kind, comparison_text = comparison_description(plan)
    result = {'schema_version': SCHEMA, 'plan_sha256': plan['plan_sha256'],
              'comparison_kind': comparison_kind,
              'sources': {m: {k: v for k, v in plan['sources'][m].items() if k in
                             ('source_revision', 'source_tree', 'source_status', 'source_fingerprint')}
                          for m in MEMBERS}, 'attempts': state['attempts'], 'timings': None, 'profiles': None}
    result['validation_execution'] = {
        member: ('run-only' if plan.get('builds', {}).get(member, {}).get('run_only_validation') is True
                 else 'legacy-driver' if plan.get('builds', {}).get(member, {}).get('run_only_validation') is False
                 else 'not-recorded') for member in MEMBERS}
    if 'cpu_builds' in plan:
        result['validation_execution'].update({
            'cpu-' + member: ('run-only-prebuilt' if plan['cpu_builds'][member].get('run_only_validation')
                              else 'not-recorded') for member in MEMBERS})
    result['timing_provenance'] = plan.get('timing_reuse', {'scope': 'fresh timing observations in this run'})
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
    suite = read_json(root / 'suite.json')
    if sha(root / 'suite.json') != plan['suite_sha256']:
        raise RuntimeError('frozen workload suite changed')
    timing_cases = (result['timings'] or {}).get('cases', {})
    profile_cases = (result['profiles'] or result.get('profile_partial') or {}).get('cases', {})
    coverage = []
    for case in selected_cases(suite):
        name, adapter, parameters = case['id'], case.get('adapter'), case.get('parameters', {})
        discretization = parameters.get('discretization', 'hdg' if adapter == 'hdg' else
                                        'dgsem-volume-probe' if adapter == 'dgsem' else
                                        'dg' if adapter == 'dg' else 'not-recorded')
        device = parameters.get('device', 'cuda' if adapter in ('dg', 'hdg', 'dgsem') else 'not-recorded')
        declared_scope = ('whole-process wall time including startup/setup/shutdown'
                          if adapter == 'dg' and discretization == 'hdg' else
                          'full-solver elapsed time' if adapter == 'dg' else
                          'accepted-step median per process' if adapter == 'hdg' else
                          'central-volume per-call GPU duration' if adapter == 'dgsem' else 'not-recorded')
        timing = timing_cases.get(name, {})
        captured = profile_cases.get(name, {})
        member_profiles = {member: captured.get(member, {}).get('status', 'pending') for member in MEMBERS}
        if device == 'cpu':
            member_profiles = dict.fromkeys(MEMBERS, 'not_applicable')
        profile_status = ('not_applicable' if device == 'cpu' else
                          'complete' if all(value == 'complete' for value in member_profiles.values()) else
                          'partial' if captured else 'pending')
        coverage.append({'case_id': name, 'adapter': adapter, 'discretization': discretization,
                         'device': device, 'engine': parameters.get('engine', 'not-recorded'),
                         'dgsem_residual': parameters.get('dgsem_residual', 'not-recorded'),
                         'method': parameters.get('method', 'not-recorded'),
                         'timing_scope': timing.get('scope', declared_scope),
                         'timing_scope_source': 'observation' if 'scope' in timing else 'adapter-contract',
                         'timing_status': 'measured' if name in timing_cases else 'pending',
                         'profile_status': profile_status, 'member_profile_status': member_profiles})
    result['coverage'] = {'enabled_case_count': len(coverage), 'cases': coverage,
                          'scope': 'enabled frozen-suite cases only; this does not imply all-backend coverage'}
    if comparison_kind == 'identical-source-control':
        result['repeatability'] = control_repeatability(result['timings'])
    lines = ['# PACE A100 GPU study', '', f"Plan: `{plan['plan_sha256']}`", '',
             'Comparison: ' + comparison_text + '.', '',
             'Timing authority: independent unprofiled process pairs. Positive percentages mean less time.', '']
    if plan.get('timing_reuse'):
        lines += ['Timings reused byte-for-byte from `' + plan['timing_reuse']['run_dir'] +
                  '`; this retry collects fresh profiles and no new timing measurements.', '']
    lines += ['Validation execution: ' + ', '.join(
        f'{member}={mode}' for member, mode in result['validation_execution'].items()) + '.', '']
    if 'legacy-driver' in result['validation_execution'].values():
        lines += ['A historical source uses its original validation driver, which may configure/build in the validation allocation.', '']
    lines += [f"Enabled workload coverage: {len(coverage)} cases; selections do not imply all-backend coverage.", '',
              '| Case | Discretization | Device | Engine / DGSEM residual | Timing scope | Timing | Profiles |',
              '|---|---|---|---|---|---|---|']
    for case in coverage:
        lines.append(f"| {case['case_id']} | {case['discretization']} | {case['device']} | "
                     f"{case['engine']} / {case['dgsem_residual']} | {case['timing_scope']} | "
                     f"{case['timing_status']} | {case['profile_status']} |")
    lines += ['', 'Compare timing values only within the same numerical workload and timing scope; '
              'HDG whole-process wall time is distinct from DG/DGSEM solver elapsed time and accepted-step or volume-kernel timings.', '']
    if any(case['device'] == 'cpu' for case in coverage):
        lines += ['CPU cases use one thread on the host CPUs of this A100 allocation. '
                  'CUDA occupancy does not apply; CPU hardware counters are not collected.', '']
    if result['timings']:
        lines += ['| Case | Baseline median (s) | Candidate median (s) | Paired median less time | Paired range |',
                  '|---|---:|---:|---:|---:|']
        for name, case in result['timings']['cases'].items():
            stats = case['statistics']
            low, high = stats['paired_range_percent_less_time']
            lines.append(f"| {name} | {stats['baseline_median_seconds']:.6g} | {stats['candidate_median_seconds']:.6g} | {stats['paired_median_percent_less_time']:.3f}% | {low:.3f}% to {high:.3f}% |")
    else:
        lines.append('Timings have not completed.')
    if comparison_kind == 'identical-source-control' and result['timings']:
        lines += ['', 'Observed control repeatability (same source, same allocation):', '',
                  '| Case | Independent pairs | Median absolute paired offset | Maximum absolute paired offset | Baseline / candidate CV |',
                  '|---|---:|---:|---:|---:|']
        for name, stats in result['repeatability']['cases'].items():
            lines.append(f"| {name} | {stats['independent_pairs']} | "
                         f"{stats['median_absolute_paired_offset_percent']:.3f}% | "
                         f"{stats['maximum_absolute_paired_offset_percent']:.3f}% | "
                         f"{stats['baseline_coefficient_of_variation_percent']:.3f}% / "
                         f"{stats['candidate_coefficient_of_variation_percent']:.3f}% |")
        lines += ['', 'These are descriptive observations, not a confidence interval or a portable acceptance threshold. '
                  'Small changes need more independent pairs and a fresh allocation; always remeasure the baseline alongside the candidate.']
    latest_profile = next((attempt for attempt in reversed(state['attempts'])
                           if attempt['stage'] == 'profiles'), None)
    lines += ['', 'Profile state: ' + ('analysis_complete' if result['profiles'] else
               (latest_profile['status'] if latest_profile else 'not_started')),
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
    plan = verify(root)
    print('Comparison: ' + comparison_description(plan)[1] + '.')
    if 'cpu_builds' in plan:
        print('Build profiles: CUDA SM80 and separately validated CPU artifacts; CPU children use CPU MPI.')
    if plan.get('timing_reuse'):
        print('Timings reused from ' + plan['timing_reuse']['run_dir'] + '; fresh profiles only.')
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
    submit_parser.add_argument('--timings-time', type=wall_time, default='01:00:00',
                               help='timing allocation HH:MM:SS, 5 minutes to 1 hour (default: 01:00:00)')
    submit_parser.add_argument('--profiles-time', type=wall_time, default='01:00:00',
                               help='profiling allocation HH:MM:SS, 5 minutes to 1 hour (default: 01:00:00)')
    submit_parser.add_argument('--validation-time', type=lambda value: wall_time(value, maximum_hours=2, minimum_minutes=1),
                               help='optional canonical correctness allocation HH:MM:SS, 1 minute to 2 hours')
    submit_parser.add_argument('--prepare-only', action='store_true', help='freeze and inspect plans without submitting jobs')
    p = sub.add_parser('retry', help='freeze a new measurement executor using verified completed builds/correctness')
    p.add_argument('--from-run', required=True)
    p.add_argument('--run-dir', required=True, help='new absolute scratch directory')
    p.add_argument('--suite', help='validated replacement suite; correctness_regex must match the old run')
    p.add_argument('--cases', help='comma-separated case IDs from the selected suite; enables optional cases')
    p.add_argument('--profiles-only', action='store_true',
                   help='reuse completed timings on the same GPU; --suite may change only case.profile; no --cases')
    p.add_argument('--timings-time', type=wall_time, help='override inherited timing allocation limit')
    p.add_argument('--profiles-time', type=wall_time, help='override inherited profiling allocation limit')
    p.add_argument('--prepare-only', action='store_true', help='prepare without submitting measurement jobs')
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
    elif args.command == 'retry':
        retry(args)
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
