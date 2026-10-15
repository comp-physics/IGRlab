#!/usr/bin/env python3
"""Explicit missing-only continuation of immutable imported CPU timing cohorts.

Preparation reads/seals records and never runs applications. Compute execution
uses the coordinator's existing CPU admission, process runner and field gates.
"""
from copy import deepcopy
import getpass
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def protocol(g, suite):
    result = []
    for case in g.selected_cases(suite):
        for index in range(suite['warmups'] + suite['repetitions']):
            warmup, pair = index < suite['warmups'], index - suite['warmups']
            for member in g.MEMBERS if index % 2 == 0 else g.MEMBERS[::-1]:
                suffix = f'warmup-{index}-{member}' if warmup else f'pair-{pair}-{member}'
                result.append({'label': case['id'] + '/' + suffix, 'case_id': case['id'],
                               'member': member, 'warmup': warmup, 'pair': pair, 'fields': False})
        for member in g.MEMBERS:
            result.append({'label': case['id'] + '/fields-' + member, 'case_id': case['id'],
                           'member': member, 'warmup': False, 'pair': None, 'fields': True})
    return result


def host():
    info = Path('/proc/cpuinfo')
    models = sorted(set(re.findall(r'^model name\s*:\s*(.+)$', info.read_text(), re.M))) if info.is_file() else []
    return {'hostname': socket.getfqdn(), 'cpu_models': models,
            'allowed_logical_cpus': sorted(os.sched_getaffinity(0)),
            'thread_environment': {key: os.environ.get(key) for key in
                                   ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
            'scope': 'current free cpu-small allocation; one application/MPI rank and one BLAS/OpenMP thread'}


def record(g, plan, case, item, directory, admission, allocation, host_metadata=None):
    """Admit a successful command and reparse its original output, without apps."""
    directory = g.absolute(directory)
    result_path = directory / 'result.json'
    result, command, saved = (g.read_json(directory / name) for name in
                               ('result.json', 'command.json', 'observation.json'))
    require(result.get('status') == 'complete' and type(result.get('returncode')) is int and
            result['returncode'] == 0, 'retained process did not complete successfully')
    require(command.get('status') == 'running' and all(command.get(key) == result.get(key) for key in
            ('argv', 'cwd', 'environment', 'started_unix_seconds')), 'command/result controls differ')
    require(result.get('cwd') == str(directory), 'retained process working directory differs')
    for name in ('stdout', 'stderr'):
        require(g.sha(directory / (name + '.log')) == result.get(name + '_sha256'),
                'retained ' + name + ' bytes differ from process receipt')
    for key in ('started_unix_seconds', 'process_wall_seconds'):
        require(type(result.get(key)) in (int, float) and math.isfinite(result[key]) and result[key] > 0,
                'invalid retained process time')
    cpu = admission['cpu_artifacts'][item['member']]
    require(cpu.get('profile') == 'cpu', 'retained command lacks a genuine CPU artifact')
    source = Path(plan['sources'][item['member']]['source_snapshot'])
    binary = Path(cpu['binaries'][case['binary']])
    expected = [cpu['runtime']['launcher'], '-n', '1', *g.study_workloads.command(
        case, binary, source, directory, fields=item['fields'])]
    require(result.get('argv') == expected, 'retained scientific command/binary/MPI route differs')
    environment = result.get('environment', {})
    require(type(environment) is dict and set(environment) <= set(g.SAFE_ENV),
            'retained environment contains unowned controls')
    controls = g.study_workloads.environment(case, directory, fields=item['fields'])
    controls.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    recorded = {key: value for key, value in controls.items() if key in g.SAFE_ENV}
    require(all(environment.get(key) == value for key, value in recorded.items()),
            'retained recorded scientific/thread controls differ')
    require(all(environment.get(key) == value for key, value in {
        'SLURM_JOB_ID': allocation['job_id'], 'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3',
        'SLURM_JOB_QOS': 'embers', 'SLURM_JOB_PARTITION': 'cpu-small', 'SLURM_NTASKS': '1'}.items()),
        'retained command allocation differs from its original CPU attempt')
    metadata = {'wall_seconds': result['process_wall_seconds']} if case['parameters'].get('discretization') == 'hdg' else {}
    observation = g.study_workloads.parse(case, directory / 'stdout.log', directory / 'stderr.log', None, **metadata)
    g.check_observation_identity(observation, plan, item['member'])
    require(saved.get('receipt') == str(result_path) and
            {key: value for key, value in saved.items() if key != 'receipt'} == observation,
            'retained observation differs from reparsed native output')
    require(type(observation.get('seconds')) in (int, float) and math.isfinite(observation['seconds']) and
            observation['seconds'] > 0, 'retained timing is invalid')
    return {'protocol': item, 'directory': str(directory), 'observation': saved,
            'result_sha256': g.sha(result_path), 'process_wall_seconds': result['process_wall_seconds'],
            'origin': {'job_id': allocation['job_id'], 'hostname': allocation['hostname'],
                       'cpu_allocation': allocation, 'host': host_metadata,
                       'hardware_availability': 'recorded' if host_metadata else
                           'CPU model/affinity unavailable in original failed attempt'},
            'unrecorded_controls': {key: value for key, value in controls.items() if key not in g.SAFE_ENV}}


def prepare(g, args):
    if os.environ.get('SLURM_JOB_ID'):
        require(args.prepare_only, 'use --prepare-only inside the owned CPU allocation; never submit another job there')
        require(not any(os.environ.get(key) for key in ('PMIX_NAMESPACE', 'PMIX_RANK', 'PMI_RANK', 'PMI_SIZE')),
                'CPU continuation preparation cannot inherit an MPI/PMIx step context')
        g.cpu_allocation()  # Same live ownership/resource/end guard as the timing worker.
    require(getattr(args, 'reuse_validated_artifacts', False) is True,
            'continue-timings requires explicit --reuse-validated-artifacts')
    old, root = g.absolute(args.from_run).resolve(strict=True), g.absolute(args.run_dir).resolve()
    require(not root.exists(), 'continuation requires an exclusive new run directory')
    with g.locked(old):
        plan = g.verify(old)
        require(g.cpu_only(plan) and plan.get('import_origin') and
                plan['import_origin'].get('reuse_validated_artifacts') is True,
                'only prevalidated imported CPU timing studies can continue')
        require(not plan.get('timing_reuse') and not plan.get('profile_continuation'),
                'GPU profile retries cannot become CPU timing continuations')
        require(set(plan['executor_sha256']) in (set(g.PRE_CPU_CONTINUATION_EXECUTOR_FILES),
                                               set(g.EXECUTOR_FILES)),
                'unsupported CPU executor generation')
        suite = g.study_workloads.load_suite(old / 'suite.json')
        g.validate_cpu_cases(suite)
        require(plan['executor_sha256'].get('study_workloads.py') == g.sha(Path(g.__file__).with_name('study_workloads.py')),
                'retained numerical adapter differs; preserve its gates before continuing')
        state = g.read_json(old / 'state.json')
        require(not g.completed(state, 'timings'), 'completed timing cohorts are immutable')
        attempts = [a for a in state['attempts'] if a.get('stage') == 'timings']
        require(attempts and all(a.get('status') == 'failed' and a.get('finished_unix_seconds') for a in attempts),
                'original CPU worker is not terminal/inactive')
        ids = sorted({str(a['job_id']) for a in attempts} |
                     {str(job['job_id']) for job in state['jobs'] if job.get('job_id')})
        require(ids and all(re.fullmatch(r'[0-9]+', value) for value in ids), 'invalid original worker job IDs')
        active = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%T'],
                                text=True, capture_output=True, timeout=30)
        require(active.returncode == 0, 'original CPU study live queue is unavailable')
        for row in active.stdout.splitlines():
            match = re.fullmatch(r'([0-9]+)(?:[_+][^|]+)?\|[^|]+', row.strip())
            require(match and match[1] not in ids, 'original CPU study has an active/unknown queued worker')
        g.checked_builds(old)  # Sealed canonical records; explicit mode does not query historical accounting.
        admission = g.read_json(old / 'admission.json')
        require(admission.get('plan_sha256') == plan['plan_sha256'] and
                set(admission.get('cpu_artifacts', {})) == set(g.MEMBERS), 'original CPU admission is missing')
        paths = {old / name for name in ('plan.json', 'state.json', 'suite.json', 'admission.json')}
        paths.update(old / 'executor' / name for name in plan['executor_sha256'])
        inherited = deepcopy(plan.get('cpu_timing_continuation', {}).get('retained', {}))
        seals = dict(plan.get('cpu_timing_continuation', {}).get('artifact_sha256', {}))
        items = protocol(g, suite)
        wanted = {item['label']: item for item in items}
        require(set(inherited) <= set(wanted), 'inherited continuation has unexpected labels')
        cases = {case['id']: case for case in g.selected_cases(suite)}
        retained = inherited
        for label, value in inherited.items():
            checked = record(g, plan, cases[wanted[label]['case_id']], wanted[label], Path(value['directory']),
                             admission, value['origin']['cpu_allocation'], value['origin'].get('host'))
            require(checked == value, 'inherited CPU observation/provenance changed')
        for attempt in attempts:
            directory = g.absolute(attempt['directory'])
            require(directory.is_relative_to(old / 'attempts') and directory.is_dir(), 'invalid original attempt path')
            require(attempt.get('gpu') is None and attempt.get('cpu_allocation', {}).get('job_id') == attempt['job_id'],
                    'original attempt lacks its actual CPU allocation')
            paths.update(path for path in directory.rglob('*') if path.is_file())
            results = {str(path.parent.relative_to(directory)): path for path in directory.rglob('result.json')}
            require(set(results) <= set(wanted), 'original attempt contains unexpected command labels')
            for label, path in results.items():
                result = g.read_json(path)
                if result.get('status') != 'complete' or result.get('returncode') != 0:
                    continue  # Failure is sealed below, never promoted to a pass.
                require(label not in retained, 'duplicate successful protocol label; no sample pooling')
                item = wanted[label]
                retained[label] = record(g, plan, cases[item['case_id']], item, path.parent, admission,
                                         attempt['cpu_allocation'], attempt.get('cpu_host'))
        require(retained and len(retained) < len(items), 'continuation needs admitted successes and missing protocol labels')
        require(all(value['protocol'] == wanted[label] for label, value in retained.items()), 'retained protocol changed')
        require(set(retained) == {item['label'] for item in items[:len(retained)]},
                'retained successes are not the original chronological prefix')
        for case in cases.values():
            signatures = [value['observation']['signature'] for value in retained.values()
                          if value['protocol']['case_id'] == case['id']]
            require(not signatures or all(value == signatures[0] for value in signatures),
                    'retained numerical policy/work differs')
        for path in paths:
            require(not path.is_symlink() and path.is_file(), 'original evidence must be regular immutable files')
            seals[str(path)] = g.sha(path)
        protected = [old, Path(g.__file__).resolve().parents[2]]
        protected += [Path(value['source_snapshot']) for value in plan['sources'].values()]
        protected += [Path(build[key]) for build in plan['builds'].values() for key in
                      ('app_build', 'app_install', 'mfem_source', 'mfem_build', 'mfem_install', 'spack_env')]
        require(not any(root.is_relative_to(path.resolve()) for path in protected), 'new continuation overlaps immutable inputs')
        new = {key: deepcopy(value) for key, value in plan.items() if key != 'plan_sha256'}
        new.update(run_dir=str(root), created_unix_seconds=time.time(),
                   cpu_timing_continuation={'run_dir': str(old), 'plan_sha256': plan['plan_sha256'],
                       'state_sha256': g.sha(old / 'state.json'), 'suite_sha256': plan['suite_sha256'],
                       'artifact_sha256': seals, 'retained': retained,
                       'missing_labels': [item['label'] for item in items if item['label'] not in retained],
                       'scope': 'missing-only CPU process continuation; original failures immutable; allocation cohorts distinct'})
        new['wall_time_limits']['timings'] = g.wall_time(getattr(args, 'timings_time', None) or g.stage_wall_time(plan, 'timings'))
        root.mkdir(parents=True, exist_ok=False)
        g.freeze_executor(root)
        shutil.copy2(old / 'suite.json', root / 'suite.json')
        require(g.sha(root / 'suite.json') == plan['suite_sha256'], 'suite bytes changed during preparation')
        new['executor_sha256'] = {name: g.sha(root / 'executor' / name) for name in g.EXECUTOR_FILES}
        new['plan_sha256'] = g.digest(new)
        g.write_json(root / 'plan.json', new, exclusive=True)
        g.write_json(root / 'state.json', {'schema_version': g.SCHEMA, 'builds': deepcopy(state['builds']),
            'build_attempts': deepcopy(state.get('build_attempts', [])), 'jobs': [], 'attempts': []}, exclusive=True)
        g.verify(root)
        print(root / 'plan.json')
        if not args.prepare_only:
            with g.locked(root):
                g.submit_stage(root, 'timings')
        return root, new


def execute(g, root, plan, suite, admission, runner):
    context = plan['cpu_timing_continuation']
    items = protocol(g, suite)
    cases = {case['id']: case for case in g.selected_cases(suite)}
    records = deepcopy(context['retained'])
    attempt = next(a for a in g.read_json(root / 'state.json')['attempts'] if a['directory'] == str(runner.output))
    current_host = host()
    require(attempt.get('gpu') is None and attempt.get('cpu_allocation'), 'continuation requires live CPU admission')
    # Retained records are reparsed again on compute, with the freshly admitted
    # same-path CPU binary/MPI route. No retained application is executed.
    for item in items:
        if item['label'] in records:
            old = records[item['label']]
            checked = record(g, plan, cases[item['case_id']], item, Path(old['directory']), admission,
                             old['origin']['cpu_allocation'], old['origin'].get('host'))
            require(checked == old, 'retained observation/provenance changed before execution')

    def checkpoint(error=None):
        value = {'schema_version': g.SCHEMA, 'status': 'timings_partial', 'gpu': None,
                 'host': current_host, 'cpu_timing_continuation': context,
                 'records': records, 'missing_labels': [x['label'] for x in items if x['label'] not in records]}
        if error:
            value['error'] = str(error)
        g.write_json(runner.output / 'partial.json', value)

    checkpoint()
    try:
        for item in items:
            if item['label'] in records:
                continue
            case = cases[item['case_id']]
            _, directory = g.observe(runner, root, plan, admission, case, item['member'], item['label'], None,
                                     fields=item['fields'])
            completed = record(g, plan, case, item, directory, admission, attempt['cpu_allocation'], current_host)
            signature = completed['observation']['signature']
            require(all(value['observation']['signature'] == signature for value in records.values()
                        if value['protocol']['case_id'] == item['case_id']), 'continued numerical policy/work differs')
            records[item['label']] = completed
            checkpoint()
        result = {'schema_version': g.SCHEMA, 'status': 'timings_complete', 'gpu': None, 'host': current_host,
                  'cpu_timing_continuation': context, 'cases': {}}
        for case in g.selected_cases(suite):
            selected = [records[item['label']] for item in items if item['case_id'] == case['id']]
            signatures = [x['observation']['signature'] for x in selected]
            require(all(value == signatures[0] for value in signatures), 'continued policy/work/export differs')
            measured = [x for x in selected if not x['protocol']['fields']]
            observations = [{'member': x['protocol']['member'], 'pair': x['protocol']['pair'],
                             'warmup': x['protocol']['warmup'], **x['observation'],
                             'process_wall_seconds': x['process_wall_seconds'],
                             'process_provenance': x['origin']} for x in measured]
            exported = {x['protocol']['member']: x['observation'] for x in selected if x['protocol']['fields']}
            directories = {x['protocol']['member']: Path(x['directory']) for x in selected if x['protocol']['fields']}
            require(set(exported) == set(g.MEMBERS), 'both complete field exports are required')
            gate = g.study_workloads.compare_fields(case, directories['baseline'], directories['candidate'],
                                                   Path(plan['sources']['candidate']['source_snapshot']))
            values = {member: [x['observation']['seconds'] for x in measured if
                       not x['protocol']['warmup'] and x['protocol']['member'] == member] for member in g.MEMBERS}
            statistics = g.sample_statistics(values['baseline'], values['candidate'])
            wall_values = {member: [x['process_wall_seconds'] for x in measured if
                           not x['protocol']['warmup'] and x['protocol']['member'] == member] for member in g.MEMBERS}
            wall_statistics = g.sample_statistics(wall_values['baseline'], wall_values['candidate'])
            pairs = []
            for pair in range(suite['repetitions']):
                origins = {x['protocol']['member']: x['origin'] for x in measured if
                           not x['protocol']['warmup'] and x['protocol']['pair'] == pair}
                same = all(origins['baseline'][key] == origins['candidate'][key] for key in ('job_id', 'hostname'))
                pairs.append({'pair': pair, 'members': origins, 'same_allocation': same})
            statistics['statistical_scope'] = ('independent alternating pairs across explicitly recorded allocations; '
                'not a homogeneous single-host cohort; split-allocation pairs are not matched hardware')
            wall_statistics['statistical_scope'] = statistics['statistical_scope']
            wall_statistics['timer_scope'] = 'whole process wall including MPI/startup and lazy initialization; no prewarming'
            result['cases'][case['id']] = {'adapter': case['adapter'], 'scope': observations[0]['scope'],
                'statistics': statistics, 'observations': observations, 'exported_observations': exported,
                'whole_process_wall_statistics': wall_statistics,
                'fields': gate, 'allocation_pairs': pairs,
                'warmup_scope': 'original successful warmups retained; no repeated warmup in the continuation allocation'}
        return result
    except Exception as error:
        checkpoint(error)
        raise
