#!/usr/bin/env python3
"""Admit existing canonical builds and sealed tests, without building or relocating.

Preparation reads metadata and the source's public --print-plan. The measurement
worker additionally runs the existing canonical prebuilt/binding admission.
An allocation's later exit state is retained separately from its earlier tests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pwd
import re
import shlex
import subprocess

import study_build
import study_site

ORIGIN = 'existing-canonical-test-result'
MEMBERS = ('baseline', 'candidate')


def require(value, message):
    if not value:
        raise ValueError(message)


def exact(path):
    path = Path(path)
    require(path.is_absolute() and path.exists() and path == path.resolve(strict=True),
            f'missing or noncanonical import path: {path}')
    study_build._no_symlink_ancestors(path)
    return path


def sha(path):
    path = exact(path)
    require(path.is_file(), f'missing regular import artifact: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def result_module():
    # A frozen operator carries the exact receipt semantics used at preparation.
    path = Path(__file__).with_name('canonical_test_result_receipt.py')
    if not path.exists():
        path = Path(__file__).parents[1] / 'build/test_result_receipt.py'
    spec = importlib.util.spec_from_file_location('study_canonical_result', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def run(argv):
    return subprocess.run(argv, capture_output=True, text=True, timeout=60,
                          env={**os.environ, 'TZ': 'UTC', 'SLURM_TIME_FORMAT': 'standard'})


def options(argv):
    require(isinstance(argv, list) and all(isinstance(item, str) and item for item in argv),
            'invalid canonical request argv')
    values, switches = {}, set()
    value_flags = {'--site', '--site-root', '--spack-root', '--output-root',
                   '--mfem-output-root', '--jobs', '-j', '--mfem-jobs',
                   '--build-time', '--validation-time', '--test-regex', '--test-label'}
    allowed = {'--a100', '--cpu', '--cpu-only', '--hdg-runtime', '--nvtx-profile',
               '--no-research-tests', '--no-clean', '--no-deps', '--no-tests'}
    index = 0
    while index < len(argv):
        flag = argv[index]
        # The public PACE wrapper historically appends --a100 to an already
        # selected A100 request. This identical target switch is idempotent.
        require(flag not in values and (flag not in switches or flag == '--a100'),
                f'duplicate canonical option: {flag}')
        if flag in value_flags:
            require(index + 1 < len(argv), f'missing canonical option value: {flag}')
            values[flag] = argv[index + 1]
            index += 2
        else:
            require(flag in allowed, f'unsupported existing canonical option: {flag}')
            switches.add(flag)
            index += 1
    return values, switches


def existing_plan(record, correctness_regex, *, reuse_validated_artifacts=False):
    require(type(reuse_validated_artifacts) is bool, 'artifact reuse mode must be an explicit boolean')
    require(isinstance(record, dict) and set(record) == {'build_request', 'build_job_id', 'test_result'},
            'import member requires build_request, build_job_id and test_result only')
    require(re.fullmatch(r'[1-9][0-9]*', str(record['build_job_id'])), 'invalid real scheduler job ID')
    request_path, result_path = exact(record['build_request']), exact(record['test_result'])
    request_sha, result_sha = sha(request_path), sha(result_path)
    request = json.loads(request_path.read_text())
    require(set(request) == {'source', 'argv'}, 'unsupported public canonical request')
    source = exact(request['source'])
    identity = study_build._identity(source)
    study_build.verify_source(identity)
    values, switches = options(request['argv'])
    site = values.get('--site')
    study_site.policy(site)
    require('--hdg-runtime' in switches and '--no-research-tests' in switches,
            'import requires canonical site HDG/runtime builds')
    cpu = bool(switches & {'--cpu', '--cpu-only'})
    require((cpu and len(switches & {'--cpu', '--cpu-only'}) == 1 and '--a100' not in switches and
             '--nvtx-profile' not in switches) or
            (not cpu and {'--a100', '--nvtx-profile'} <= switches),
            'import requires an explicit CPU build or native A100 NVTX build')
    profile = 'cpu' if cpu else 'cuda-sm80'
    roots = {flag: exact(values.get(flag, '')) for flag in
             ('--output-root', '--mfem-output-root', '--site-root', '--spack-root')}
    environment = {**study_build.BUILD_ENV,
                   **{name: os.environ.get(name, '') for name in study_build.BUILD_FLAG_ENV}}
    command = [str(source / 'build.sh'), *request['argv'], '--print-plan']
    response = study_build._run(command, env=study_build._environment(environment))
    stdout = response.stdout.decode(errors='replace')
    wt = 'wt-' + hashlib.sha256(os.fsencode(source)).hexdigest()[:16]
    scope = roots['--output-root'] / 'worktrees' / wt / profile
    mfem = roots['--mfem-output-root']
    env_path = roots['--site-root'] / 'site' / site / 'spack-envs' / profile
    expected = (f'[build-plan] output_root={roots["--output-root"]} worktree_id={wt} scope={scope}',
                f'[build-plan] env={env_path} app_build={scope}/app/build app_install={scope}/app/install',
                f'[build-plan] mfem_source={mfem}/source mfem_build={mfem}/build mfem_install={mfem}/install ')
    require(all(line in stdout for line in expected), 'recorded paths differ from the frozen canonical plan')
    configuration = {'source_fingerprint': identity['source_fingerprint'], 'profile': profile,
                     'request_sha256': request_sha}
    plan = {'schema_version': 1, 'origin': ORIGIN, 'profile': profile, 'source': identity,
            'reuse_validated_artifacts': reuse_validated_artifacts,
            'configuration': configuration, 'configuration_key': study_build._json_hash(configuration),
            'request_path': str(request_path), 'request_sha256': request_sha,
            'request_argv': request['argv'], 'argv': command[:-1], 'env': environment,
            'correctness_regex': correctness_regex, 'run_only_validation': True,
            'output_root': str(roots['--output-root']), 'site_root': str(roots['--site-root']),
            'spack_root': str(roots['--spack-root']), 'spack_env': str(env_path),
            'app_build': str(scope / 'app/build'), 'app_install': str(scope / 'app/install'),
            'mfem_source': str(mfem / 'source'), 'mfem_build': str(mfem / 'build'),
            'mfem_install': str(mfem / 'install'),
            'mfem_identity': str(mfem / 'build/dg-igr-mfem-identity-v1.txt'),
            'mfem_config': str(mfem / 'install/lib/cmake/mfem/MFEMConfig.cmake'),
            'mfem_targets': str(mfem / 'install/lib/cmake/mfem/MFEMTargets.cmake'),
            'mfem_cache': {'mode': 'shared-canonical', 'output_root': str(mfem), 'lock': str(mfem / '.lock')},
            'plan_stdout': stdout, 'plan_stderr': response.stderr.decode(errors='replace'),
            'test_result': str(result_path), 'test_result_sha256': result_sha,
            'test_result_runner_sha256': sha(result_module()[1]),
            'build_job_id': str(record['build_job_id'])}
    if site != 'pace':
        plan['site'] = site
    validation = validate_result(plan)
    plan['import_artifact_sha256'] = validation['artifact_sha256']
    require(sha(request_path) == request_sha and sha(result_path) == result_sha,
            'public request or test result changed during import planning')
    return plan


def validate_result(plan, *, full=False):
    """Use the existing canonical receipt semantics, never infer a pass from Slurm."""
    require(sha(plan['request_path']) == plan['request_sha256'] and
            sha(plan['test_result']) == plan['test_result_sha256'], 'import request or test result changed')
    result_path = exact(plan['test_result'])
    result = json.loads(result_path.read_text())
    module, runner_path = result_module()
    require(result.get('schema') == module.SCHEMA and result.get('status') == 'passed' and
            result.get('success') is True and result.get('returncode') == 0 and
            result.get('identity_verified_before_and_after') is True,
            'import requires a final successful actual canonical test-result receipt')
    require(result.get('runner_sha256') == plan.get('test_result_runner_sha256', sha(runner_path)),
            'unsupported test-result runner identity')
    saved_path = exact(result['prebuilt_receipt'])
    require(saved_path == Path(plan['app_build']) / 'igrlab-prebuilt-validation-v1.json' and
            sha(saved_path) == result['prebuilt_receipt_sha256'], 'prebuilt receipt differs from actual test result')
    saved = json.loads(saved_path.read_text())
    require(saved.get('schema') == 'igrlab-prebuilt-validation-v1', 'unsupported canonical prebuilt receipt')
    for key in ('source', 'app_build', 'app_install', 'mfem_source', 'mfem_build', 'mfem_install', 'spack_env'):
        expected = plan['source']['source_snapshot'] if key == 'source' else plan[key]
        require(saved['paths'].get(key) == expected, f'prebuilt import path differs: {key}')
    require(result['source'] == saved['source'] and result['policy'] == saved['policy'] and
            saved['policy']['profile'] == plan['profile'] and saved['policy']['site'] == study_site.name(plan),
            'test result source/policy differs from the canonical build')
    require(sha(Path(plan['source']['source_snapshot']) / 'scripts/build/prebuilt_validation.py') ==
            result.get('prebuilt_checker_sha256'), 'canonical prebuilt checker changed')
    checker, _ = module.checker(Path(plan['source']['source_snapshot']))
    require(checker.source_identity(Path(plan['source']['source_snapshot'])) == saved['source'],
            'current source differs from the source actually built and tested')
    require(module.command_selection(result['argv'], saved) == result['selection'],
            'actual canonical test command/selection differs from recorded result')
    # The actual child must repeat exactly the build configuration and roots.
    test_values, test_switches = options([arg for arg in result['argv'][1:]
                                        if arg != '--tests-only'])
    build_values, build_switches = options(plan['request_argv'])
    # pace.sh prepares the site runtime and strips its own three root/profile
    # options from the real canonical child. The saved paths/policy and public
    # request bind those roots; an outer wrapper may instead retain the flags.
    for key in ('--site', '--site-root', '--spack-root'):
        require(key not in test_values or test_values[key] == build_values.get(key),
                f'actual test command changed {key}')
    for key in ('--output-root', '--mfem-output-root'):
        require(test_values.get(key) == build_values.get(key), f'actual test command changed {key}')
    policy_switches = {'--a100', '--cpu', '--cpu-only', '--hdg-runtime', '--nvtx-profile', '--no-research-tests'}
    require(test_switches & policy_switches == build_switches & policy_switches,
            'actual test command changed canonical build features')
    selected = result['selected_ctest']
    require(selected.get('returncode') == 0, 'actual selected CTest graph query failed')
    tests = selected['graph'].get('tests', [])
    full_tests = {test['name']: test for test in saved['artifacts']['test_graph']['tests']}
    semantics = lambda test: {key: value for key, value in test.items() if key != 'backtrace'}
    names = [test.get('name') for test in tests]
    require(names and len(names) == len(set(names)) and all(name in full_tests for name in names) and
            all(semantics(test) == semantics(full_tests[test['name']]) for test in tests),
            'actual selected test commands differ from the canonical graph')
    require(not any(prop.get('name') == 'DISABLED' and prop.get('value') not in
                    (False, 0, '', 'FALSE', 'OFF', '0') for test in tests for prop in test.get('properties', [])),
            'disabled tests are not executed acceptance')
    required = {name for name in full_tests if re.search(plan['correctness_regex'], name)}
    require(required and required <= set(names), 'passed test set does not cover the suite correctness regex')
    artifacts = {str(result_path): plan['test_result_sha256'], plan['request_path']: plan['request_sha256'],
                 str(saved_path): result['prebuilt_receipt_sha256']}
    require(set(result.get('logs', {})) == {'stdout.log', 'stderr.log'}, 'actual test logs are missing')
    for name, log in result['logs'].items():
        path = exact(log['path'])
        require(path == result_path.parent / name and sha(path) == log['sha256'], 'actual test log changed')
        artifacts[str(path)] = log['sha256']
    outcomes = module.test_outcomes((result_path.parent / 'stdout.log').read_text(), selected['graph'])
    require(outcomes == result.get('test_outcomes'), 'actual Passed outcomes differ from sealed test result')
    before, after = result['allocation_before'], result['allocation_after']
    require(before['uid'] == after['uid'] == os.getuid() and before['hostname'] == after['hostname'] and
            'login' not in before['hostname'] and before['environment'] == after['environment'] and
            before.get('gpu') == after.get('gpu'), 'actual test allocation/device identity changed')
    for snapshot in (before, after):
        fields = snapshot['scheduler']['fields']
        site = study_site.name(plan)
        site_match = (fields.get('Account') == 'gts-sbryngelson3' and fields.get('QOS') == 'embers'
                      if site == 'pace' else study_site.scheduler_matches(fields, site, plan['profile']))
        require(snapshot.get('authority') == study_site.policy(study_site.name(plan))['authority'] and
                snapshot['scheduler']['returncode'] == 0 and fields.get('JobState') == 'RUNNING' and
                site_match and
                fields.get('UserId') == f'{pwd.getpwuid(os.getuid()).pw_name}({os.getuid()})' and
                fields.get('JobId') == snapshot['environment'].get('SLURM_JOB_ID'),
                'test result is not from an owned live free compute allocation')
    if plan['profile'] == 'cuda-sm80':
        require(before['scheduler']['fields'].get('Partition') ==
                study_site.policy(study_site.name(plan))['gpu_partition'] and
                'A100' in before.get('gpu', {}).get('name', '') and
                before.get('gpu', {}).get('compute_capability') == '8.0' and
                re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',
                             before.get('gpu', {}).get('uuid', '')), 'test result lacks native A100 identity')
    if full:
        module.admission(checker, saved, saved_path, result['prebuilt_receipt_sha256'])
        for path, identity in saved['artifacts']['files'].items():
            artifacts[path] = identity['sha256']
    for path, expected in plan.get('import_artifact_sha256', {}).items():
        require(sha(path) == expected, f'imported result artifact changed: {path}')
    return {'result': result, 'prebuilt': saved, 'artifact_sha256': artifacts,
            'artifact_links': saved['artifacts'].get('links', {}) if full else {}}


def timestamp(value):
    require(isinstance(value, str) and value not in ('Unknown', 'None', ''), 'missing scheduler timestamp')
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()


def accounting(job, site='pace', profile='cuda-sm80'):
    require(re.fullmatch(r'[1-9][0-9]*', str(job)), 'invalid real scheduler job ID')
    command = ['sacct', '-n', '-P', '-j', str(job), '--format=JobID,State%64,ExitCode,User%64,Account%64,QOS%64,JobName%64,Start,End,NodeList%1024,SubmitLine%4096']
    result = run(command)
    require(result.returncode == 0, 'import accounting query failed')
    rows = [line.split('|', 10) for line in result.stdout.splitlines() if line.split('|', 1)[0] == str(job)]
    require(len(rows) == 1 and len(rows[0]) == 11, 'missing or ambiguous imported job accounting')
    row = rows[0]
    require(row[3] == pwd.getpwuid(os.getuid()).pw_name and
            study_site.accounting_matches(row, site, profile),
            'imported job is not the current user in the selected site account/QOS')
    return {'argv': command, 'returncode': result.returncode, 'row': row}


def completed_records(plan, receipt, *, full=False):
    reuse = plan.get('reuse_validated_artifacts', False)
    require(type(reuse) is bool and type(receipt.get('reuse_validated_artifacts', False)) is bool and
            receipt.get('reuse_validated_artifacts', False) == reuse,
            'imported artifact reuse mode differs from its receipt')
    require(plan.get('origin') == ORIGIN and receipt.get('origin') == ORIGIN and receipt.get('success') is True and
            receipt.get('build_job_id') == plan['build_job_id'] and
            receipt.get('test_result') == plan['test_result'] and 'validation_job_id' not in receipt,
            'import must retain real build and test-result records without a validation job substitute')
    study_build.verify_source(plan['source'])
    validation = validate_result(plan, full=full)
    if reuse:
        # This explicit mode uses the actual canonical build/test seals. It
        # makes no claim about the later state of their historical Slurm jobs.
        result = validation['result']
        start, finish = result['started_unix_seconds'], result['finished_unix_seconds']
        job = result['allocation_before']['scheduler']['fields']['JobId']
        require(all(re.fullmatch(r'[1-9][0-9]*', str(value)) for value in (plan['build_job_id'], job)) and
                type(start) in (int, float) and type(finish) in (int, float) and
                math.isfinite(start) and math.isfinite(finish) and start <= finish,
                'invalid retained build/test allocation identity or sealed interval')
        authority = {'query_performed': False, 'query_status': 'not_queried',
                     'mode': 'reuse_validated_artifacts',
                     'authority': 'canonical prebuilt receipt and sealed actual Passed test outcomes',
                     'prebuilt_receipt': result['prebuilt_receipt'],
                     'prebuilt_receipt_sha256': result['prebuilt_receipt_sha256'],
                     'test_result': plan['test_result'], 'test_result_sha256': plan['test_result_sha256']}
        validation.update(job_ids={'build': plan['build_job_id']}, request=plan['request_path'],
                          request_sha256=plan['request_sha256'],
                          build_accounting={**authority, 'historical_job_id': plan['build_job_id']},
                          test_allocation_accounting={**authority, 'historical_job_id': job},
                          test_execution_scope='sealed Passed outcomes; historical accounting not queried by explicit opt-in; no historical job completion claim')
        return validation
    site = study_site.name(plan)
    build = (accounting(plan['build_job_id']) if site == 'pace' else
             accounting(plan['build_job_id'], site, 'cpu'))
    row = build['row']
    require(row[1:3] == ['COMPLETED', '0:0'] and row[6] == 'IGRlab-build',
            'existing canonical build did not complete successfully')
    worker = [str(Path(plan['source']['source_snapshot']) / 'scripts/build' / (site + '.sh')),
              '--worker', 'build', plan['request_path']]
    tokens = shlex.split(row[10])
    require(sum(tokens[index:index + 4] == worker for index in range(len(tokens))) == 1,
            'build accounting belongs to a different public request')
    result = validation['result']
    snapshot = result['allocation_before']
    fields = snapshot['scheduler']['fields']
    job = fields['JobId']
    observed = accounting(job) if site == 'pace' else accounting(job, site, plan['profile'])
    row = observed['row']
    start, finish = result['started_unix_seconds'], result['finished_unix_seconds']
    require(type(start) in (int, float) and type(finish) in (int, float) and
            math.isfinite(start) and math.isfinite(finish) and start <= finish and
            timestamp(row[7]) <= start and row[9] == fields.get('NodeList'),
            'sealed tests do not fall within their actual allocation/node identity')
    hosts = run(['scontrol', 'show', 'hostnames', row[9]])
    require(hosts.returncode == 0 and snapshot['hostname'].split('.')[0] in hosts.stdout.splitlines(),
            'sealed tests ran outside the recorded compute nodes')
    if row[1] == 'RUNNING':
        current = run(['scontrol', 'show', 'job', '-o', job])
        require(current.returncode == 0, 'live imported allocation query failed')
        actual = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)', current.stdout))
        # Time strings from the earlier shell can use a different client TZ.
        # UTC accounting above binds the allocation start and test interval.
        for key in ('JobId', 'UserId', 'Account', 'QOS', 'NodeList', 'Command', 'BatchFlag'):
            require(fields.get(key) and actual.get(key) == fields[key], f'live test allocation changed: {key}')
        require(actual.get('JobState') == 'RUNNING' and actual.get('BatchFlag') == '0' and
                (not os.environ.get('SLURM_JOB_ID') or os.environ['SLURM_JOB_ID'] == job),
                'only the same owned running interactive allocation may reuse its sealed tests')
    else:
        # Standard sacct End omits fractional seconds. It denotes that second's
        # interval; the sealed result must finish strictly before the next one.
        require(row[1].split()[0] in {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'} and
                finish < timestamp(row[8]) + 1, 'validation allocation has no final accounting covering its sealed tests')
    command = fields.get('Command', '')
    require(command and command not in ('(null)', 'Unknown') and
            any(token == command or Path(token).name == Path(command).name
                for token in shlex.split(row[10])),
            'test allocation submit identity differs from the sealed allocation command')
    require(timestamp(build['row'][8]) <= start, 'tests precede completion of the imported canonical build')
    validation.update(job_ids={'build': plan['build_job_id']}, request=plan['request_path'],
                      request_sha256=plan['request_sha256'], build_accounting=build,
                      test_allocation_accounting=observed,
                      test_execution_scope='sealed Passed outcomes; allocation exit remains separately reported')
    return validation


def selected_binaries(plan, names):
    """Pin only selected executable bytes during metadata-only preparation."""
    saved = validate_result(plan)['prebuilt']
    artifacts = {}
    for name in names:
        matches = [Path(path) for path in saved['artifacts']['files']
                   if Path(path).name == name and Path(path).is_relative_to(Path(plan['app_build']))]
        require(len(matches) == 1, f'selected executable missing or ambiguous in canonical receipt: {name}')
        binary = exact(matches[0])
        with binary.open('rb') as stream:
            magic = stream.read(4)
        require(os.access(binary, os.X_OK) and magic == b'\x7fELF' and
                sha(binary) == saved['artifacts']['files'][str(binary)]['sha256'],
                f'selected canonical executable changed: {binary}')
        artifacts[str(binary)] = saved['artifacts']['files'][str(binary)]['sha256']
    return artifacts


def receipt(plan):
    return {'origin': ORIGIN, 'success': True, 'returncode': 0,
            'reuse_validated_artifacts': plan.get('reuse_validated_artifacts', False),
            'build_job_id': plan['build_job_id'], 'test_result': plan['test_result'],
            'test_result_sha256': plan['test_result_sha256'],
            'argv': plan['argv'],
            'source_fingerprint': plan['source']['source_fingerprint'],
            'configuration_key': plan['configuration_key']}


def inspect_manifest(path, correctness_regex, cpu_regex, *, reuse_validated_artifacts=False,
                     cpu_only=False):
    require(type(cpu_only) is bool, 'cpu_only must be an explicit boolean')
    path = exact(path)
    expected_sha = sha(path)
    manifest = json.loads(path.read_text())
    require(isinstance(manifest, dict) and set(manifest) in
            ({'schema_version', 'builds'}, {'schema_version', 'builds', 'cpu_builds'}) and
            type(manifest.get('schema_version')) is int and manifest['schema_version'] == 1,
            'unsupported artifact import manifest')
    require(not cpu_only or set(manifest) == {'schema_version', 'builds'},
            'CPU-only import requires one CPU builds cohort without cpu_builds')
    plans, receipts, records = {}, {}, {}
    for group in ('builds', 'cpu_builds'):
        if group not in manifest:
            continue
        require(set(manifest[group]) == set(MEMBERS), 'import requires exactly baseline and candidate members')
        plans[group], receipts[group] = {}, {}
        for member in MEMBERS:
            plan = existing_plan(manifest[group][member], correctness_regex if group == 'builds' else cpu_regex,
                                 reuse_validated_artifacts=reuse_validated_artifacts)
            require(plan['profile'] == ('cpu' if cpu_only or group == 'cpu_builds' else 'cuda-sm80'),
                    'import artifact profile differs from its CUDA/CPU group')
            plans[group][member] = plan
            receipts[group][member] = receipt(plan)
            records[group + '/' + member] = completed_records(plan, receipts[group][member])
    for member in MEMBERS:
        if 'cpu_builds' in plans:
            require(plans['cpu_builds'][member]['source'] == plans['builds'][member]['source'],
                    'CPU and CUDA member sources must be the exact same standalone snapshot')
    require(len({plan['site_root'] for group in plans.values() for plan in group.values()}) == 1 and
            len({study_site.name(plan) for group in plans.values() for plan in group.values()}) == 1 and
            len({plan['spack_root'] for group in plans.values() for plan in group.values()}) == 1,
            'imported cohorts require the same site and buildtool roots')
    require(sha(path) == expected_sha, 'artifact import manifest changed during preparation')
    return plans, receipts, records, {str(path): expected_sha}
