#!/usr/bin/env python3
"""Private compute-only dispatcher for distinct, approved Delta profile studies.

This does not reserve, cancel or replace an allocation. Four independent frozen
workers may run on separate physical GPUs; their elapsed times are diagnostic.
A fixed intent is pinned by SHA256, and an interrupted launch is never replayed.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import re
import signal
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import gpu_study
import study_profiles
import study_site

CLEANUP = 120
CAPACITY = 4
STOP = 0
HEX = re.compile(r'[0-9a-f]{64}')
ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return gpu_study.sha(path)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    """Exclusive, fully written publication; readers never see partial JSON."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def confined(value, root):
    path = Path(value)
    require(path.is_absolute() and path.resolve(strict=True) == path and
            path.is_relative_to(root), 'path must be real and confined to the approved root')
    return path


def intent(path, expected):
    path = Path(path)
    require(HEX.fullmatch(expected) and path.is_file() and not path.is_symlink() and
            sha(path) == expected, 'approved intent hash changed')
    value = read(path)
    require(value.get('schema_version') == 1 and value.get('approved') is True and
            value.get('controller_sha256') == sha(__file__), 'unapproved controller/intent')
    root = Path(value['site_root'])
    require(root.is_absolute() and root.resolve(strict=True) == root, 'invalid site root')
    tasks = value.get('tasks')
    require(isinstance(tasks, list) and bool(tasks), 'no approved profile tasks')
    identifiers, runs = set(), set()
    for task in tasks:
        identifier = task.get('id', '')
        require(isinstance(identifier, str) and ID.fullmatch(identifier), 'invalid task ID')
        run = confined(task['run_dir'], root / 'studies')
        require(identifier not in identifiers and run not in runs, 'duplicate task or study writer')
        identifiers.add(identifier)
        runs.add(run)
        require(isinstance(task.get('plan_sha256'), str) and HEX.fullmatch(task['plan_sha256']),
                'task needs an exact plan digest')
        require(set(task.get('executor_sha256', {})) == {'gpu_study.py', 'delta_runtime.sh'} and
                all(isinstance(digest, str) and HEX.fullmatch(digest)
                    for digest in task['executor_sha256'].values()), 'task needs frozen executor pins')
        if 'origin_admission_sha256' in task:
            require(isinstance(task['origin_admission_sha256'], str) and
                    HEX.fullmatch(task['origin_admission_sha256']), 'invalid original admission pin')
            confined(task['origin_run_dir'], root / 'studies')
    return value


def check_task(task):
    """Small readiness check; the frozen worker retains every full content gate."""
    run = Path(task['run_dir'])
    plan = gpu_study.load_plan(run)
    require(plan['plan_sha256'] == task['plan_sha256'] and plan.get('site') == 'delta' and
            not gpu_study.cpu_only(plan) and bool(plan.get('profile_continuation')),
            'pool task needs its admitted, explicit cross-GPU profile continuation')
    for name, expected in task['executor_sha256'].items():
        file = run / 'executor' / name
        require(not file.is_symlink() and sha(file) == expected and
                plan['executor_sha256'].get(name) == expected, 'frozen task executor changed')
    require(task['executor_sha256']['gpu_study.py'] == sha(gpu_study.__file__),
            'task executor lacks this reviewed pool admission')
    if (run / 'admission.json').exists():
        admission = read(run / 'admission.json')
        require(admission['plan_sha256'] == task['plan_sha256'], 'task has no matching binary admission')
    else:
        # Public prepare-only retries intentionally defer fresh admission until
        # the canonical in-step worker; bind the actual admitted timing origin.
        site_root = next((p.parent for p in run.parents if p.name == 'studies'), None)
        require(site_root is not None and HEX.fullmatch(task.get('origin_admission_sha256', '')),
                'prepared continuation needs its original admission pin')
        origin = confined(task['origin_run_dir'], site_root / 'studies')
        original = gpu_study.load_plan(origin)
        require(all(plan.get(k) == original.get(k) for k in ('site', 'site_root', 'spack_root', 'sources')) and
                original.get('site') == 'delta', 'original admission belongs to different frozen sources/site')
        for source in original['sources'].values():
            confined(source['source_snapshot'], site_root)
        for section in ('retry_origin', 'timing_reuse', 'profile_continuation'):
            require(plan.get(section, {}).get('run_dir') == str(origin) and
                    plan[section]['plan_sha256'] == original['plan_sha256'],
                    'continuation original plan/path binding changed')
        prior = origin / 'admission.json'
        require(not prior.is_symlink() and sha(prior) == task['origin_admission_sha256'] and
                read(prior)['plan_sha256'] == original['plan_sha256'], 'original binary admission seal changed')
    state = read(run / 'state.json')
    timing = gpu_study.completed(state, 'timings')
    require(timing and sha(timing['summary']) == timing['summary_sha256'], 'timings not complete/sealed')
    if gpu_study.completed(state, 'profiles'):
        return 'already_complete'
    require(not any(a['stage'] == 'profiles' for a in state['attempts']),
            'previous profile attempt needs a reviewed missing-only continuation, never replay')
    for recorded in state.get('jobs', []):
        if recorded.get('stage') == 'profiles' and recorded.get('job_id'):
            reply = subprocess.run(['squeue', '-h', '-j', str(recorded['job_id']), '-o', '%T'],
                                   text=True, capture_output=True, timeout=30)
            require(reply.returncode == 0 and not reply.stdout.strip(), 'profile scheduler owner remains active')
    return 'ready'


def scheduler(kind, selector):
    env = {**os.environ, 'TZ': 'UTC', 'SLURM_TIME_FORMAT': 'standard'}
    argv = ['scontrol', 'show', kind, '-o', selector]
    reply = subprocess.run(argv, text=True, capture_output=True, timeout=30, env=env)
    require(reply.returncode == 0, 'actual scheduler query failed')
    fields = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)', reply.stdout))
    return fields, argv


def parent_allocation():
    job, host = os.environ.get('SLURM_JOB_ID', ''), socket.getfqdn()
    require(re.fullmatch(r'[1-9][0-9]*', job) and 'login' not in host and
            host.endswith('.delta.ncsa.illinois.edu') and
            (os.environ.get('SLURM_STEP_ID') or os.environ.get('SLURM_STEPID')) == 'batch',
            'profile coordinator requires its real compute-resident batch step')
    require(not any(os.environ.get(k) for k in ('PMIX_NAMESPACE', 'PMIX_RANK', 'PMI_RANK', 'PMI_SIZE')),
            'coordinator requires MPI-none')
    fields, argv = scheduler('job', job)
    expected = {'JobId': job, 'JobState': 'RUNNING', 'NumNodes': '1', 'NumTasks': '1',
                'UserId': f'{pwd.getpwuid(os.getuid()).pw_name}({os.getuid()})',
                **study_site.scheduler_fields('delta')}
    require(all(fields.get(k) == v for k, v in expected.items()) and
            gpu_study._delta_profile_pool_parent(fields), 'unapproved four-GPU parent resources/custody')
    require(fields.get('NodeList', '').split('.')[0] == host.split('.')[0], 'coordinator outside owned node')
    node, node_argv = scheduler('node', fields['NodeList'])
    configured = dict(item.split('=', 1) for item in node.get('CfgTRES', '').split(',') if '=' in item)
    memory = re.fullmatch(r'([1-9][0-9]*)([MG])', configured.get('mem', ''))
    require(configured.get('cpu') == '64' and configured.get('gres/gpu') == '4' and memory and
            int(memory[1]) * (1024 if memory[2] == 'G' else 1) >= 192 * 1024,
            'actual node cannot fit four reserved 48GiB roles')
    end = datetime.fromisoformat(fields['EndTime']).replace(tzinfo=timezone.utc).timestamp()
    require(math.isfinite(end) and time.time() < end - CLEANUP, 'allocation reached cleanup')
    return {'job_id': job, 'hostname': host, 'end_epoch': end, 'scheduler': fields,
            'node_scheduler': node, 'query_argv': [argv, node_argv],
            'scope': 'whole-node batch coordinator; four distinct physical-GPU profile roles'}


def idle_gpu(gpu=None):
    argv = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits']
    if gpu is not None:
        require(study_profiles.UUID_RE.fullmatch(gpu['uuid']), 'idle query needs a physical GPU UUID')
        argv.append('--id=' + gpu['uuid'])
    reply = subprocess.run(argv, text=True, capture_output=True, timeout=15)
    require(reply.returncode == 0 and not reply.stdout.strip(), 'role GPU is occupied or cannot be inspected')
    return {'query_argv': argv, 'stdout_sha256': hashlib.sha256(reply.stdout.encode()).hexdigest(),
            'gpu_uuid': gpu['uuid'] if gpu is not None else None}


def rendezvous(records, allocation, allowed):
    """Bind physical identity, not remapped CUDA ordinal strings."""
    require(1 <= len(records) <= CAPACITY, 'invalid ready wave')
    uuids, steps, used = set(), set(), set()
    for value in records:
        live = value['allocation']
        cpus = value['cpu_affinity']
        uuid = live['gpu']['uuid']
        require(live['job_id'] == allocation['job_id'] and
                live['hostname'].split('.')[0] == allocation['hostname'].split('.')[0] and
                live['end_epoch'] == allocation['end_epoch'] and live.get('step') and
                live['step']['step_id'] not in steps and
                study_profiles.UUID_RE.fullmatch(uuid) and uuid not in uuids,
                'role allocation/physical GPU is ambiguous or overlaps')
        require(isinstance(cpus, list) and len(cpus) == 8 and len(set(cpus)) == 8 and
                all(type(c) is int and 0 <= c < (os.cpu_count() or 1) for c in cpus) and
                not used.intersection(cpus),
                'role CPU affinities overlap or differ from eight actual allowed CPUs')
        uuids.add(uuid)
        steps.add(live['step']['step_id'])
        used.update(cpus)
    available = set(allowed) - used
    require(available, 'no disjoint CPU remains for the coordinator')
    return min(available)


def step_argv(task, allocation, source, intent_path, intent_sha, session):
    return ['srun', '--jobid=' + allocation['job_id'], '--exclusive', '--exact', '--nodes=1',
            '--ntasks=1', '--cpus-per-task=8', '--mem=48G', '--gpus-per-task=1',
            '--gpu-bind=single:1', '--cpu-bind=cores', '--mpi=none',
            '--kill-on-bad-exit=0', sys.executable, '-B', str(source), '--intent', str(intent_path),
            '--intent-sha256', intent_sha, '--role', task['id'], '--session', str(session)]


def stop_process(process):
    def group_alive():
        process.poll()  # Reap the leader before checking for remaining descendants.
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False
    # A reaped leader alone does not prove its private group is empty.
    for number in (signal.SIGTERM, signal.SIGKILL):
        if not group_alive():
            return
        try:
            os.killpg(process.pid, number)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 5
        while group_alive() and time.monotonic() < deadline:
            time.sleep(.05)
    require(not group_alive(), 'owned process group remains after bounded cleanup')


def parent_idle(allocation):
    """Actual four-device identity and empty GPU proof before a retained handoff."""
    again = parent_allocation()
    require(all(again[key] == allocation[key] for key in ('job_id', 'hostname', 'end_epoch')),
            'parent custody changed before handoff')
    argv = ['nvidia-smi', '--query-gpu=uuid,name,compute_cap', '--format=csv,noheader,nounits']
    reply = subprocess.run(argv, text=True, capture_output=True, timeout=15)
    rows = list(gpu_study.csv.reader(reply.stdout.splitlines()))
    require(reply.returncode == 0 and len(rows) == 4 and all(len(row) == 3 for row in rows),
            'retained parent must expose its four actual GPUs')
    devices = [tuple(field.strip() for field in row) for row in rows]
    require(len({uuid for uuid, _, _ in devices}) == 4 and all(
            study_profiles.UUID_RE.fullmatch(uuid) and 'A100' in name and capability == '8.0'
            for uuid, name, capability in devices), 'retained parent device identities are ambiguous')
    idle = idle_gpu()
    return {'gpu_uuids': [row[0] for row in devices], 'idle': idle,
            'query_argv': argv, 'scheduler': again['scheduler']}


def event(root, allocation, kind, family, message):
    # Existing bridge schema only. Actual parent closure belongs to its outer owner.
    job = allocation['job_id']
    record = {'event_id': f'{job}:{kind}:{family}', 'job_id': job, 'kind': kind,
              'family': family, 'utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
              'message': message[:1024], 'node': allocation['hostname'],
              'end_epoch': allocation['end_epoch'],
              'remaining_seconds': max(0, allocation['end_epoch'] - time.time())}
    path = root / ('hpc-events-' + job + '.jsonl')
    try:
        with path.open('a+') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0)
            ids = {json.loads(line)['event_id'] for line in stream if line.strip()}
            if record['event_id'] not in ids:
                stream.seek(0, os.SEEK_END)
                stream.write(json.dumps(record, sort_keys=True) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
    except (OSError, ValueError, KeyError):
        # Callback failure must never abandon a retained allocation or a live step.
        pass


def outcome(task, allocation, before, returncode):
    """A process exit is not scientific/profile analysis success."""
    state = read(Path(task['run_dir']) / 'state.json')
    new = [a for a in state['attempts'] if a['id'] not in before and a['stage'] == 'profiles']
    require(len(new) == 1 and new[0]['job_id'] == allocation['job_id'] and
            new[0]['gpu']['uuid'] == allocation['gpu']['uuid'], 'worker outcome lacks exact invocation binding')
    attempt = new[0]
    result = {'task_id': task['id'], 'returncode': returncode, 'attempt': attempt,
              'native_status': 'partial_or_unresolved', 'analysis_status': 'incomplete',
              'scope': 'concurrent cross-study profiling is diagnostic; no timing/speed authority'}
    if returncode == 0 and attempt['status'] == 'analysis_complete':
        require(sha(attempt['summary']) == attempt['summary_sha256'], 'analysis summary seal changed')
        result.update(native_status='complete_and_admitted', analysis_status='complete')
    return result


def role(args, value):
    task = next(t for t in value['tasks'] if t['id'] == args.role)
    session = confined(args.session, Path(value['site_root']) / 'records' / 'delta-profile-pools')
    directory = session / task['id']
    live = gpu_study.delta_allocation(profile_pool=True)
    require(os.environ.get('NV_COMPUTE_PROFILER_DISABLE_CONCURRENT_PROFILING') is None,
            'NCU global serialization override must be unset')
    check_task(task)
    record = {'task_id': task['id'], 'intent_sha256': args.intent_sha256,
              'allocation': live, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'idle_gpu': idle_gpu(live['gpu'])}
    write(directory / 'ready.json', record)
    deadline = live['end_epoch'] - CLEANUP
    gate = directory / 'start.gate'
    while not gate.exists() and not STOP and time.time() < deadline:
        time.sleep(.2)
    require(not STOP and time.time() < deadline and gate.is_file(), 'role start gate was not admitted')
    require(read(gate) == {'task_id': task['id'], 'intent_sha256': args.intent_sha256,
                           'job_id': live['job_id'], 'ready_sha256': sha(directory / 'ready.json')},
            'role start gate identity changed')
    again = gpu_study.delta_allocation(profile_pool=True)
    require(again['gpu']['uuid'] == live['gpu']['uuid'] and
            sorted(os.sched_getaffinity(0)) == record['cpu_affinity'], 'role custody changed before capture')
    idle_gpu(live['gpu'])
    check_task(task)
    before = {a['id'] for a in read(Path(task['run_dir']) / 'state.json')['attempts']}
    run = Path(task['run_dir'])
    argv = ['bash', str(run / 'executor' / 'delta_runtime.sh'), str(run / 'plan.json'),
            'profiles', str(run / 'executor')]
    process = None
    started = time.time()
    try:
        with (directory / 'worker.log').open('x') as stream:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None and not STOP and time.time() < deadline:
                time.sleep(.2)
            stop_process(process)
            result = outcome(task, live, before, process.returncode)
            final = gpu_study.delta_allocation(profile_pool=True)
            require(final['gpu']['uuid'] == live['gpu']['uuid'] and
                    final['step']['step_id'] == live['step']['step_id'], 'role custody changed after capture')
            result['final_gpu_idle'] = idle_gpu(live['gpu'])
    except Exception as error:
        if process is not None:
            try:
                stop_process(process)
            except Exception as cleanup_error:
                error = RuntimeError(str(error) + '; cleanup: ' + str(cleanup_error))
        result = {'task_id': task['id'], 'native_status': 'partial_or_unresolved',
                  'analysis_status': 'incomplete', 'unsafe_role': True, 'error': str(error)}
    result.update(argv=argv, started_epoch=started, finished_epoch=time.time(), allocation=live)
    write(directory / 'outcome.json', result)
    return 0 if result['analysis_status'] == 'complete' else 1


def wave(tasks, allocation, session, args, allowed):
    running, receipts = [], []
    failure = None
    try:
        for task in tasks:
            directory = session / task['id']
            directory.mkdir()
            argv = step_argv(task, allocation, Path(__file__).resolve(), args.intent,
                             args.intent_sha256, session)
            write(directory / 'launch.json', {'argv': argv, 'started_epoch': time.time()})
            stream = (directory / 'srun.log').open('x')
            try:
                process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            finally:
                stream.close()
            running.append((task, directory, process))
        deadline = allocation['end_epoch'] - CLEANUP
        while not STOP and time.time() < deadline:
            if all((directory / 'ready.json').is_file() for _, directory, _ in running):
                break
            require(not any(p.poll() is not None for _, _, p in running), 'role failed before rendezvous')
            time.sleep(.2)
        require(not STOP and time.time() < deadline, 'cleanup reached before rendezvous')
        records = [read(directory / 'ready.json') for _, directory, _ in running]
        for (task, _, _), record in zip(running, records):
            require(record['task_id'] == task['id'] and record['intent_sha256'] == args.intent_sha256,
                    'ready record belongs to a different approved task')
        coordinator_cpu = rendezvous(records, allocation, allowed)
        os.sched_setaffinity(0, {coordinator_cpu})
        for task, directory, _ in running:
            write(directory / 'start.gate', {'task_id': task['id'], 'intent_sha256': args.intent_sha256,
                                             'job_id': allocation['job_id'],
                                             'ready_sha256': sha(directory / 'ready.json')})
        while any(p.poll() is None for _, _, p in running) and not STOP and time.time() < deadline:
            time.sleep(.2)
    except Exception as error:
        failure = str(error)
    finally:
        for task, directory, process in running:
            cleanup_error = None
            try:
                stop_process(process)
            except Exception as error:
                cleanup_error = str(error)
            try:
                result = read(directory / 'outcome.json')
                require(isinstance(result, dict) and result.get('task_id') == task['id'],
                        'worker outcome belongs to a different task')
                if result.get('analysis_status') == 'complete':
                    ready = read(directory / 'ready.json')['allocation']
                    require(result.get('native_status') == 'complete_and_admitted' and
                            result.get('final_gpu_idle', {}).get('gpu_uuid') == ready['gpu']['uuid'],
                            'complete worker outcome lacks its final idle-device proof')
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                result = {'task_id': task['id'], 'native_status': 'partial_or_unresolved',
                          'analysis_status': 'incomplete', 'unsafe_role': True,
                          'error': 'step ended without a bound worker outcome: ' + str(error)}
            result['srun_returncode'] = process.returncode
            if cleanup_error or result.get('unsafe_role'):
                failure = failure or cleanup_error or result.get('error') or 'unsafe role custody'
                result.update(analysis_status='incomplete', unsafe_role=True)
                if cleanup_error:
                    result['cleanup_error'] = cleanup_error
            elif process.returncode != 0:
                result['analysis_status'] = 'incomplete'
            receipts.append(result)
    if failure is not None:
        receipts.append({'analysis_status': 'incomplete', 'error': failure, 'wave_failure': True,
                         'scope': 'wave stopped; exact task directories retain partials'})
    return receipts


def retain(allocation):
    while not STOP and time.time() < allocation['end_epoch'] - CLEANUP:
        time.sleep(min(1, max(0, allocation['end_epoch'] - CLEANUP - time.time())))


def coordinate(args):
    allocation = parent_allocation()
    root = session = None
    # Any later preflight/transport error still retains this genuine owned node.
    try:
        value = intent(args.intent, args.intent_sha256)
        root = Path(value['site_root'])
        pool_root = root / 'records' / 'delta-profile-pools'
        pool_root.mkdir(parents=True, exist_ok=True)
        require(pool_root.resolve() == pool_root, 'pool records must not traverse symlinks')
        session = pool_root / (allocation['job_id'] + '-' + args.intent_sha256)
        require(not session.exists(), 'existing pool intent needs reconciliation; never relaunch unknown steps')
        session.mkdir()
        write(session / 'intent.json', value)
        write(session / 'allocation.json', allocation)
        results = []
        allowed = set(os.sched_getaffinity(0))
        with ExitStack() as locks:
            # Claims supplement, rather than replace, each canonical worker lock.
            ready = []
            for task in value['tasks']:
                stream = locks.enter_context((Path(task['run_dir']) / '.profile-pool-claim').open('a'))
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with gpu_study.locked(Path(task['run_dir'])):
                    status = check_task(task)
                if status == 'already_complete':
                    results.append({'task_id': task['id'], 'analysis_status': 'already_complete',
                                    'scope': 'no capture repeated'})
                else:
                    ready.append(task)
            event(root, allocation, 'profiles_started', 'pool-' + args.intent_sha256[:16],
                  'Approved distinct-study profile pool admitted; physical bindings precede every start gate.')
            for index in range(0, len(ready), CAPACITY):
                if STOP or time.time() >= allocation['end_epoch'] - CLEANUP:
                    results.extend({'task_id': t['id'], 'analysis_status': 'not_started',
                                    'scope': 'cleanup; no application launched'} for t in ready[index:])
                    break
                wave_results = wave(ready[index:index + CAPACITY], allocation, session, args, allowed)
                results.extend(wave_results)
                if any(r.get('wave_failure') for r in wave_results):
                    results.extend({'task_id': t['id'], 'analysis_status': 'not_started',
                                    'scope': 'earlier wave custody failure; no blind continuation'}
                                   for t in ready[index + CAPACITY:])
                    break
        # Claims are released only after every owned role's cleanup was attempted.
        success = all(r['analysis_status'] in ('complete', 'already_complete') for r in results)
        write(session / 'outcome.json', {'allocation': allocation, 'tasks': results,
            'scope': 'parallel diagnostic profiles; serial unprofiled cohorts remain speed authority'})
        event(root, allocation, 'profiles_complete' if success else 'failure',
              'pool-' + args.intent_sha256[:16],
              'Profile pool analysis complete.' if success else 'Profile pool incomplete; preserve partials and diagnose before continuation.')
        safe_roles = not any(r.get('unsafe_role') or r.get('wave_failure') for r in results)
        if safe_roles and not STOP and time.time() < allocation['end_epoch'] - CLEANUP:
            handoff = parent_idle(allocation)
            write(session / 'retained-handoff.json', handoff)
            event(root, allocation, 'deferred', 'allocation_retained',
                  'Profile roles are terminal and all four owned GPUs idle; allocation retained for the next approved action.')
        return 0 if success else 1
    except Exception as error:
        if root is not None:
            event(root, allocation, 'failure', 'pool-' + args.intent_sha256[:16],
                  'Profile pool preflight failed; allocation retained for inspection: ' + str(error))
        if session is not None and session.is_dir() and not (session / 'failure.json').exists():
            write(session / 'failure.json', {'error': str(error), 'allocation': allocation,
                                           'scope': 'no unknown task is replayed'})
        raise
    finally:
        retain(allocation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intent', required=True, type=Path)
    parser.add_argument('--intent-sha256', required=True)
    parser.add_argument('--role')
    parser.add_argument('--session')
    args = parser.parse_args()
    require(bool(args.role) == bool(args.session), 'role and session must be supplied together')
    def stopping(number, _frame):
        global STOP
        STOP = number
    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    try:
        return role(args, intent(args.intent, args.intent_sha256)) if args.role else coordinate(args)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
