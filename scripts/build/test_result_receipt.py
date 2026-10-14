#!/usr/bin/env python3
"""Seal actual canonical tests-only outcomes; a prebuilt receipt is not a test pass."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import socket
import subprocess
import sys
import tempfile
import time


SCHEMA = 'igrlab-canonical-test-result-v1'
SLURM_KEYS = ('SLURM_JOB_ID', 'SLURM_JOB_ACCOUNT', 'SLURM_JOB_QOS',
              'SLURM_JOB_PARTITION', 'SLURM_NTASKS', 'SLURM_JOB_GPUS',
              'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path == path.resolve(strict=True),
            f'missing or symlinked receipt artifact: {path}')
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def exact_components(path):
    path = Path(path)
    require(path.is_absolute(), f'path must be absolute: {path}')
    for component in (path, *path.parents):
        require(not component.is_symlink(), f'symlinked receipt path: {component}')
    require(path == path.resolve(), f'noncanonical receipt path: {path}')
    return path


def write_json(path, value):
    # Each attempt has a new private directory. Never replace a final outcome.
    fd, temporary = tempfile.mkstemp(prefix='.receipt-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write('\n')
        os.chmod(temporary, 0o444)
        os.link(temporary, path)  # Atomic publication, fails if path exists.
    finally:
        os.unlink(temporary)


def option(command, name):
    positions = [index for index, value in enumerate(command) if value == name]
    require(len(positions) <= 1, f'duplicate test command option: {name}')
    if not positions:
        return None
    index = positions[0]
    require(index + 1 < len(command) and command[index + 1], f'missing value for {name}')
    return command[index + 1]


def command_selection(command, saved):
    source = Path(saved['paths']['source'])
    require(command and command[0] == str(source / 'build.sh'),
            'test command must invoke the exact prebuilt source canonical build.sh')
    require(command.count('--tests-only') == 1 and not set(command[1:]) &
            {'--submit', '--print-plan', '--help', '-h', '--no-tests', '--clean'},
            'receipt command must execute canonical --tests-only without build/plan overrides')
    output = option(command, '--output-root')
    require(output is not None, 'test command requires explicit --output-root')
    output = exact_components(Path(output))
    worktree = 'wt-' + hashlib.sha256(os.fsencode(source)).hexdigest()[:16]
    profile = saved['policy']['profile']
    require(str(output / 'worktrees' / worktree / profile / 'app/build') == saved['paths']['app_build'],
            'test command output scope differs from the prebuilt receipt')
    return {'regex': option(command, '--test-regex'), 'label': option(command, '--test-label')}


def checker(source):
    path = source / 'scripts/build/prebuilt_validation.py'
    spec = importlib.util.spec_from_file_location('canonical_prebuilt_test_result', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def admission(module, saved, receipt, expected_sha):
    require(sha(receipt) == expected_sha, 'prebuilt receipt changed during validation')
    args = argparse.Namespace(**{name: Path(value) for name, value in saved['paths'].items()},
                              expected_source=None,
                              policy=[key + '=' + value for key, value in saved['policy'].items()])
    require(module.verify(args) == saved, 'prebuilt admission differs from the retained receipt')
    require(sha(receipt) == expected_sha, 'prebuilt receipt changed during admission')


def selected_graph(saved, selection):
    command = [str(Path(saved['paths']['cmake']).with_name('ctest')),
               '--test-dir', saved['paths']['app_build'], '--show-only=json-v1']
    for key, flag in (('regex', '-R'), ('label', '-L')):
        if selection[key] is not None:
            command += [flag, selection[key]]
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    require(result.returncode == 0, f'CTest graph query failed: {result.stderr}')
    graph = json.loads(result.stdout)
    tests = graph.get('tests')
    require(isinstance(tests, list) and tests, 'test regex/label selected no tests')
    full = {test['name']: test for test in saved['artifacts']['test_graph']['tests']}
    names = [test.get('name') for test in tests]
    require(len(set(names)) == len(names) and all(name in full for name in names),
            'selected tests are absent from the admitted CTest graph')
    # CTest renumbers its backtraceGraph when filtering. Its integer backtrace
    # references are presentation metadata; compare the actual test semantics.
    semantics = lambda test: {key: value for key, value in test.items() if key != 'backtrace'}
    require(all(semantics(test) == semantics(full[test['name']]) for test in tests),
            'selected test commands/configuration differ from the admitted graph')
    require(not any(prop.get('name') == 'DISABLED' and prop.get('value') not in
                    (False, 0, '', 'FALSE', 'OFF', '0')
                    for test in tests for prop in test.get('properties', [])),
            'selected CTest graph includes a disabled test')
    return {'argv': command, 'returncode': result.returncode, 'graph': graph}


def test_outcomes(stdout, graph):
    """Require one actual CTest Passed progress row for every selected test.

    The frozen driver predates JUnit export. Missing/changed output formatting
    therefore fails closed; zero-exit skip/disabled outcomes are not test passes.
    """
    expected = {test['name'] for test in graph['tests']}
    observed = {}
    positions = set()
    pattern = re.compile(r'^\s*([0-9]+)/([0-9]+)\s+Test\s+#\s*([0-9]+):\s*'
                         r'(.*?)\s+\.{2,}\s+(.*?)\s+([0-9.eE+-]+)\s+sec\s*$', re.M)
    for match in pattern.finditer(stdout):
        position, count, test_id, name, status, seconds = match.groups()
        require(name in expected and name not in observed and int(count) == len(expected) and
                1 <= int(position) <= len(expected) and int(position) not in positions,
                'ambiguous or unexpected canonical CTest outcome rows')
        positions.add(int(position))
        observed[name] = {'name': name, 'test_number': int(test_id), 'status': status,
                          'seconds': seconds}
    require(set(observed) == expected and all(row['status'] == 'Passed' for row in observed.values()),
            'not every selected CTest test has an actual Passed outcome; skipped/disabled/missing tests reject')
    return [observed[test['name']] for test in graph['tests']]


def delta_counter_step(fields, job, host):
    """Record a real single-GPU step within Delta's counter reservation."""
    def resources(value, cpus, gpus):
        entries = value.split(',')
        tres = dict(item.split('=', 1) for item in entries if '=' in item)
        return (len(entries) == len(tres) and tres.get('cpu') == cpus and
                tres.get('mem') in ('48G', '49152M') and tres.get('gres/gpu') == gpus and
                all(key in ('gres/gpu:a100', 'gres/gpu:nvidia_a100') and count == gpus
                    for key, count in tres.items() if key.startswith('gres/gpu:')))

    require(host.endswith('.delta.ncsa.illinois.edu') and
            fields.get('Partition') == 'gpuA100x4' and fields.get('NumNodes') == '1' and
            fields.get('NumTasks') == '1' and fields.get('NumCPUs') == '64' and
            resources(fields.get('AllocTRES', ''), '64', '4') and
            resources(fields.get('ReqTRES', ''), '8', '1') and
            set(re.split('[,&]', fields.get('Features', ''))) >= {'perf', 'nvperf'},
            'Delta counter parent must reserve four GPUs/64 CPUs for the requested one-GPU/eight-CPU step')
    selectors = {name: os.environ.get(name, '') for name in
                 ('DELTA_OWNED_STEP_ID', 'SLURM_STEP_ID', 'SLURM_STEPID')}
    original = selectors['SLURM_STEP_ID'] or selectors['SLURM_STEPID']
    selector = selectors['DELTA_OWNED_STEP_ID'] or original
    require(re.fullmatch(r'[0-9]+', selector) and
            all(not value or value == selector for value in selectors.values()),
            'Delta counter validation requires its preserved actual numeric step ID')
    command = ['scontrol', 'show', 'step', '-o', job + '.' + selector]
    query = subprocess.run(command, text=True, capture_output=True, timeout=30)
    step = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)', query.stdout))
    expected = {'StepId': job + '.' + selector, 'UserId': str(os.getuid()), 'State': 'RUNNING',
                'Partition': 'gpuA100x4', 'Nodes': '1', 'CPUs': '8', 'Tasks': '1'}
    require(query.returncode == 0 and all(step.get(key) == value for key, value in expected.items()) and
            step.get('NodeList', '').split('.')[0] == host.split('.')[0] and
            resources(step.get('TRES', ''), '8', '1'),
            'actual Delta counter step is not the owned running eight-CPU single-GPU step')
    node_command = ['scontrol', 'show', 'hostnames', fields.get('NodeList', '')]
    nodes = subprocess.run(node_command, text=True, capture_output=True, timeout=30)
    require(nodes.returncode == 0 and
            [node.split('.')[0] for node in nodes.stdout.split()] == [host.split('.')[0]],
            'Delta validation is outside its actual counter reservation node')
    return {'step_id': step['StepId'], 'selector_environment': selectors,
            'scheduler': {'argv': command, 'returncode': query.returncode,
                          'stdout': query.stdout, 'fields': step},
            'node_query': {'argv': node_command, 'returncode': nodes.returncode,
                           'stdout': nodes.stdout},
            'scope': 'single-GPU execution step inside actual four-GPU counter-exclusive parent allocation'}


def allocation(saved):
    """Record actual live scheduler/device identity, never a completion claim."""
    environment = {name: os.environ.get(name, '') for name in SLURM_KEYS}
    value = {'hostname': socket.getfqdn(), 'uid': os.getuid(), 'environment': environment}
    site = saved['policy'].get('site')
    if site not in ('pace', 'delta'):
        value['authority'] = 'non-PACE; no scheduler or GPU admission'
        return value
    job = environment['SLURM_JOB_ID']
    require(re.fullmatch(r'[1-9][0-9]*', job) and 'login' not in value['hostname'],
            f'{site} test receipt requires a live compute allocation')
    command = ['scontrol', 'show', 'job', '-o', job]
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    require(result.returncode == 0, f'allocation identity query failed: {result.stderr}')
    fields = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)', result.stdout))
    profile = saved['policy']['profile']
    account = 'gts-sbryngelson3' if site == 'pace' else (
        'bdiy-delta-gpu' if profile.startswith('cuda-sm') else 'bdiy-delta-cpu')
    require(fields.get('JobId') == job and fields.get('JobState') == 'RUNNING' and
            fields.get('Account') == account and (site != 'pace' or fields.get('QOS') == 'embers') and
            fields.get('UserId') == f'{pwd.getpwuid(os.getuid()).pw_name}({os.getuid()})' and
            fields.get('BatchFlag') in ('0', '1'),
            f'test allocation is not the current owned running {site} job')
    require(all(environment[key] == fields[field] for key, field in (
                ('SLURM_JOB_ACCOUNT', 'Account'), ('SLURM_JOB_PARTITION', 'Partition'))) and
            (site != 'pace' or environment['SLURM_JOB_QOS'] == fields.get('QOS')),
            'Slurm test environment differs from live allocation identity')
    value.update(authority=f'live {"PACE" if site == "pace" else "Delta"} allocation', scheduler={'argv': command,
                 'returncode': result.returncode, 'stdout': result.stdout, 'fields': fields})
    if site == 'delta' and profile == 'cpu':
        require(fields.get('Partition') == 'cpu', 'Delta CPU validation requires the CPU partition')
    if profile.startswith('cuda-sm'):
        model, capability, partition = {'cuda-sm80': ('A100', '8.0', 'gpu-a100'),
                                        'cuda-sm70': ('V100', '7.0', 'gpu-v100')}[profile]
        if site == 'delta':
            require(profile == 'cuda-sm80', 'Delta site currently supports native A100 validation')
            partition = 'gpuA100x4'
            if 'gres/gpu=1' not in fields.get('AllocTRES', '').split(','):
                value['step'] = delta_counter_step(fields, job, value['hostname'])
                require(re.fullmatch(r'(?:[0-9]+|GPU-[0-9a-fA-F-]+)', environment['SLURM_STEP_GPUS']),
                        'Delta counter validation requires one assigned execution-step GPU')
        visible = environment['CUDA_VISIBLE_DEVICES']
        require(fields.get('Partition') == partition and
                ('gres/gpu=1' in fields.get('AllocTRES', '').split(',') or 'step' in value) and
                (environment['SLURM_JOB_GPUS'] or environment['SLURM_STEP_GPUS']) and
                re.fullmatch(r'(?:[0-9]+|GPU-[0-9a-fA-F-]+)', visible),
                'CUDA validation requires one assigned GPU in the selected partition')
        command = ['nvidia-smi', '--query-gpu=uuid,name,compute_cap', '--format=csv,noheader']
        result = subprocess.run(command, text=True, capture_output=True, timeout=30)
        require(result.returncode == 0, f'GPU identity query failed: {result.stderr}')
        rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
        require(len(rows) == 1 and len(rows[0]) == 3, 'validation must expose exactly one GPU')
        uuid, name, compute = [cell.strip() for cell in rows[0]]
        require(re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', uuid) and
                model in name and compute == capability and
                (not visible.startswith('GPU-') or uuid.lower().startswith(visible.lower())),
                'validation GPU identity/capability differs from selected native profile')
        value['gpu'] = {'uuid': uuid, 'name': name, 'compute_capability': compute,
                        'argv': command, 'returncode': result.returncode, 'stdout': result.stdout}
    return value


def run(receipt, result_root, command):
    """Wrap the real canonical command; missing/failing final receipts never pass."""
    receipt = exact_components(receipt)
    saved_sha = sha(receipt)
    saved = json.loads(receipt.read_text())
    require(saved.get('schema') == 'igrlab-prebuilt-validation-v1', 'unsupported prebuilt receipt')
    require(receipt == Path(saved['paths']['app_build']) / 'igrlab-prebuilt-validation-v1.json',
            'prebuilt receipt is outside its recorded application build')
    selection = command_selection(command, saved)
    module, checker_path = checker(Path(saved['paths']['source']))
    result_root = exact_components(result_root)
    require(all(not result_root.is_relative_to(Path(saved['paths'][name])) for name in
                ('source', 'app_build', 'app_install', 'mfem_source', 'mfem_build',
                 'mfem_install', 'spack_env', 'mpi', 'hypre')),
            'test results must remain outside admitted source/build/dependency artifacts')
    result_root.mkdir(parents=True, exist_ok=True)
    exact_components(result_root)
    attempt = Path(tempfile.mkdtemp(prefix=os.environ.get('SLURM_JOB_ID', 'host') + '-', dir=result_root))
    started = {'schema': SCHEMA, 'status': 'started', 'argv': command, 'selection': selection,
               'test_result_scope': 'selected CTest outcomes; no branch coverage or job completion claim',
               'prebuilt_receipt': str(receipt), 'prebuilt_receipt_sha256': saved_sha,
               'source': saved['source'], 'policy': saved['policy'],
               'runner_sha256': sha(Path(__file__).resolve()),
               'prebuilt_checker_sha256': sha(checker_path), 'started_unix_seconds': time.time()}
    write_json(attempt / 'started.json', started)
    print(f'[validation] test result attempt: {attempt}', flush=True)
    result = {**started, 'status': 'failed', 'success': False, 'returncode': None}
    code = 3
    try:
        admission(module, saved, receipt, saved_sha)
        result['selected_ctest'] = selected_graph(saved, selection)
        result['allocation_before'] = allocation(saved)
        with (attempt / 'stdout.log').open('x') as stdout, (attempt / 'stderr.log').open('x') as stderr:
            completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        result['returncode'] = completed.returncode
        code = completed.returncode if completed.returncode >= 0 else 128 - completed.returncode
        # Test logs are mutable in CTest's working directory. Our exact copies
        # and the immutable receipt bind the genuine command outcome instead.
        for name in ('stdout.log', 'stderr.log'):
            path = attempt / name
            path.chmod(0o444)
            result.setdefault('logs', {})[name] = {'path': str(path), 'sha256': sha(path)}
            with path.open() as stream:
                destination = sys.stdout if name == 'stdout.log' else sys.stderr
                for block in iter(lambda: stream.read(65536), ''):
                    destination.write(block)
        admission(module, saved, receipt, saved_sha)
        require(selected_graph(saved, selection) == result['selected_ctest'],
                'selected CTest graph changed during execution')
        if completed.returncode == 0:
            result['test_outcomes'] = test_outcomes((attempt / 'stdout.log').read_text(),
                                                   result['selected_ctest']['graph'])
        result['allocation_after'] = allocation(saved)
        before, after = result['allocation_before'], result['allocation_after']
        require(before['uid'] == after['uid'] and before['hostname'] == after['hostname'] and
                before['environment'] == after['environment'] and
                before.get('gpu') == after.get('gpu') and
                before.get('step', {}).get('step_id') == after.get('step', {}).get('step_id'),
                'validation allocation/device identity changed')
        result['identity_verified_before_and_after'] = True
        result['success'] = completed.returncode == 0
        result['status'] = 'passed' if result['success'] else 'failed'
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        result['error'] = str(error)
        code = code or 3
        print(f'ERROR: test result receipt: {error}', file=sys.stderr)
    result['finished_unix_seconds'] = time.time()
    write_json(attempt / 'result.json', result)
    print(f'[validation] test result receipt: {attempt / "result.json"}; status={result["status"]}', flush=True)
    return code, attempt / 'result.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prebuilt-receipt', type=Path, required=True)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        return run(args.prebuilt_receipt, args.result_root, command)[0]
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f'ERROR: test result receipt: {error}', file=sys.stderr)
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
