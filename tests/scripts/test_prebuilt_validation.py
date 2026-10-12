#!/usr/bin/env python3
"""Host-only run-only admission contracts; no application is compiled or run."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prebuilt', SOURCE / 'scripts/build/prebuilt_validation.py')
prebuilt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prebuilt)
result_spec = importlib.util.spec_from_file_location('test_results', SOURCE / 'scripts/build/test_result_receipt.py')
test_results = importlib.util.module_from_spec(result_spec)
result_spec.loader.exec_module(test_results)


class CTestGraphSelection(unittest.TestCase):
    def test_real_ctest_filtered_backtraces_do_not_change_command_identity(self):
        cmake = os.environ.get('IGRLAB_TEST_CMAKE') or shutil.which('cmake')
        if not cmake:
            self.skipTest('supply IGRLAB_TEST_CMAKE for the metadata-only CTest graph fixture')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, build = root / 'source', root / 'build'
            source.mkdir()
            (source / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.15)
project(CTestMetadata NONE)
enable_testing()
add_test(NAME first COMMAND "${CMAKE_COMMAND}" -E echo first)
add_test(NAME second COMMAND "${CMAKE_COMMAND}" -E echo second)
add_test(NAME third COMMAND "${CMAKE_COMMAND}" -E echo third)
set_tests_properties(third PROPERTIES LABELS "acceptance")
''')
            subprocess.run([cmake, '-S', str(source), '-B', str(build)],
                           check=True, capture_output=True, text=True, timeout=30)
            ctest = str(Path(cmake).resolve().with_name('ctest'))
            graph = json.loads(subprocess.check_output([ctest, '--test-dir', str(build),
                               '--show-only=json-v1'], text=True, timeout=30))
            saved = {'paths': {'cmake': str(Path(cmake).resolve()), 'app_build': str(build)},
                     'artifacts': {'test_graph': graph}}
            selected = test_results.selected_graph(saved, {'regex': '^third$', 'label': 'acceptance'})
            self.assertEqual([test['name'] for test in selected['graph']['tests']], ['third'])
            full = next(test for test in graph['tests'] if test['name'] == 'third')
            self.assertNotEqual(full['backtrace'], selected['graph']['tests'][0]['backtrace'])
            full['command'][-1] = 'wrong-argument'
            with self.assertRaisesRegex(ValueError, 'commands/configuration differ'):
                test_results.selected_graph(saved, {'regex': '^third$', 'label': 'acceptance'})


class PrebuiltValidation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / 'repo'
        (self.repo / 'scripts/build').mkdir(parents=True)
        for name in ('build.sh', 'scripts/build/prebuilt_validation.py'):
            shutil.copy2(SOURCE / name, self.repo / name)
        self.put(self.repo / 'scripts/build/check_dependency_bindings.py', 'print("fixture binding admission")\n')
        self.put(self.repo / '.gitignore', '__pycache__/\nignored/\n')
        self.put(self.repo / 'solver.cpp', 'int unchanged_source;\n')
        self.init_git(self.repo)
        self.mfem = self.root / 'mfem'
        self.mfem.mkdir()
        self.put(self.mfem / 'mfem.hpp', '// fixture MFEM source\n')
        self.init_git(self.mfem)
        self.output = self.root / 'outputs'
        worktree = 'wt-' + prebuilt.digest(os.fsencode(self.repo))[:16]
        scope = self.output / 'worktrees' / worktree / 'cpu'
        self.build, self.install = scope / 'app/build', scope / 'app/install'
        self.mbuild, self.minstall = scope / 'mfem/build', scope / 'mfem/install'
        self.env_path = self.root / 'spack-env'
        self.cmake = self.root / 'cmake/bin/cmake'
        self.mpi, self.hypre = self.root / 'mpi', self.root / 'hypre'
        for path in (self.build, self.install, self.mbuild, self.minstall, self.env_path,
                     self.mpi, self.hypre, self.env_path / '.spack-env/view'):
            path.mkdir(parents=True)
        self.put(self.cmake, '#!/bin/sh\necho forbidden-cmake >> "$CALLS"\nexit 99\n', executable=True)
        self.ctest = self.cmake.with_name('ctest')
        self.put(self.ctest, '''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
if '--show-only=json-v1' in args:
    tests = [] if any('missing' in a for a in args) else [{'name': 'numerical', 'command': ['retained-fixture']}]
    if os.environ.get('FIXTURE_DISABLED') and tests:
        tests[0]['properties'] = [{'name': 'DISABLED', 'value': True}]
    print(json.dumps({'kind': 'ctestInfo', 'tests': tests}))
elif '-N' in args:
    if not any('missing' in a for a in args):
        print('  Test #1: numerical')
    print('Total Tests: 1')
else:
    with open(os.environ['CALLS'], 'a') as f:
        f.write(json.dumps(args) + '\\n')
    if os.environ.get('FIXTURE_MUTATE_PATH'):
        os.chmod(os.environ['FIXTURE_MUTATE_PATH'], os.stat(os.environ['FIXTURE_MUTATE_PATH']).st_mode | 0o200)
        with open(os.environ['FIXTURE_MUTATE_PATH'], 'ab') as f: f.write(b'changed during test')
    status = os.environ.get('FIXTURE_CTEST_STATUS', 'Passed')
    if not os.environ.get('FIXTURE_NO_OUTCOME'):
        print('  1/1 Test  #1: numerical .................. ' + status + '    0.01 sec')
    raise SystemExit(int(os.environ.get('FIXTURE_CTEST_RC', '0')))
''', executable=True)
        for root in (self.build, self.mbuild):
            for name in ('CMakeCache.txt', 'cmake_install.cmake', 'CMakeFiles/cmake.check_cache'):
                self.put(root / name, 'fixture configuration\n')
        self.put(self.build / 'CTestTestfile.cmake', '# fixture CTest graph\n')
        self.put(self.mbuild / 'dg-igr-mfem-identity-v1.txt', 'fixture stamped tuple\n')
        for name in ('lib/libmfem.a', 'lib/cmake/mfem/MFEMConfig.cmake',
                     'lib/cmake/mfem/MFEMTargets.cmake'):
            self.put(self.minstall / name, 'fixture MFEM artifact\n')
        for name in ('spack.yaml', 'spack.lock'):
            self.put(self.env_path / name, 'fixture locked dependencies\n')
        header = b'\x7fELF\x02\x01' + b'\0' * 10 + b'\x02\x00'
        self.exe = self.build / 'tests/unit_tests_main'
        self.put(self.exe, header, executable=True)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.put(self.bin / 'spack', '''#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
if args[:1] == ['-e'] and args[2:4] == ['location', '-i']:
    print(os.environ['FIXTURE_' + args[4].upper()])
else:
    with open(os.environ['CALLS'], 'a') as f: f.write('forbidden-spack ' + repr(args))
    raise SystemExit(99)
''', executable=True)
        self.environment = {k: v for k, v in os.environ.items()
                            if not k.startswith(('IGRLAB_', 'DG_IGR_', 'SPACK_', 'SLURM_'))}
        self.environment.update(PATH=str(self.bin) + ':' + self.environment['PATH'],
                                IGRLAB_SPACK_ENV=str(self.env_path),
                                FIXTURE_CMAKE=str(self.cmake.parent.parent),
                                FIXTURE_OPENMPI=str(self.mpi), FIXTURE_HYPRE=str(self.hypre),
                                CALLS=str(self.root / 'calls'), PYTHONDONTWRITEBYTECODE='1')
        self.args = argparse.Namespace(source=self.repo, app_build=self.build, app_install=self.install,
                                       mfem_source=self.mfem, mfem_build=self.mbuild,
                                       mfem_install=self.minstall, spack_env=self.env_path,
                                       cmake=self.cmake, mpi=self.mpi, hypre=self.hypre,
                                       expected_source=None, policy=self.policy())
        prebuilt.write(self.args)

    def tearDown(self):
        self.temp.cleanup()

    def init_git(self, root):
        for args in (['init', '-q'], ['add', '.'], ['-c', 'user.name=Fixture',
                     '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture']):
            subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)

    def put(self, path, data, executable=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        if executable:
            path.chmod(0o755)

    def policy(self):
        return ['mode=cpu', 'profile=cpu', 'build_type=Release', 'site=', 'caliper=0',
                'nvtx_profile=0', 'hdg_runtime=OFF', 'hdg_auxiliary=OFF', 'hdg_euler=OFF',
                'hdg_euler_device=OFF', 'research_tests=OFF', 'dependency_cuda_arch=none',
                'mfem_cuda_archs=none', 'app_cuda_archs=none', 'kernel_sources=',
                'mfem_seed=', 'mfem_output_root=']

    def run_driver(self, *extra, environment=None):
        return subprocess.run(['bash', str(self.repo / 'build.sh'), '--cpu-only', '--tests-only',
                               '--output-root', str(self.output), *extra],
                              env=self.environment | (environment or {}), text=True,
                              capture_output=True, timeout=20)

    def calls(self):
        path = self.root / 'calls'
        return path.read_text().splitlines() if path.exists() else []

    def run_receipted(self, *extra, environment=None, result_root=None):
        command = [str(self.repo / 'build.sh'), '--cpu-only', '--tests-only',
                   '--output-root', str(self.output), *extra]
        return subprocess.run(['python3', '-B', str(SOURCE / 'scripts/build/test_result_receipt.py'),
                               '--prebuilt-receipt', str(self.build / prebuilt.RECEIPT),
                               '--result-root', str(result_root or self.root / 'results'), '--', *command],
                              env=self.environment | (environment or {}), text=True,
                              capture_output=True, timeout=30)

    def outcome(self):
        paths = list((self.root / 'results').glob('*/result.json'))
        self.assertEqual(len(paths), 1)
        return paths[0], json.loads(paths[0].read_text())

    def test_real_result_binds_selected_tests_and_prebuilt_identity_without_build(self):
        result = self.run_receipted('--test-regex', '^numerical$', '--test-label', 'acceptance')
        self.assertEqual(result.returncode, 0, result.stderr)
        path, outcome = self.outcome()
        self.assertEqual(outcome['status'], 'passed')
        self.assertIs(outcome['success'], True)
        self.assertEqual(outcome['returncode'], 0)
        self.assertEqual(outcome['selection'], {'regex': '^numerical$', 'label': 'acceptance'})
        self.assertEqual(outcome['selected_ctest']['graph']['tests'][0]['name'], 'numerical')
        self.assertEqual(outcome['test_outcomes'][0]['status'], 'Passed')
        self.assertEqual(outcome['source'], prebuilt.source_identity(self.repo))
        self.assertEqual(outcome['prebuilt_receipt_sha256'], test_results.sha(self.build / prebuilt.RECEIPT))
        self.assertIs(outcome['identity_verified_before_and_after'], True)
        self.assertEqual(outcome['allocation_before']['authority'], 'non-PACE; no scheduler or GPU admission')
        self.assertEqual(path.stat().st_mode & 0o777, 0o444)
        for log in outcome['logs'].values():
            self.assertEqual(test_results.sha(Path(log['path'])), log['sha256'])
        self.assertEqual(len(self.calls()), 1)
        self.assertNotIn('forbidden-cmake', '\n'.join(self.calls()))
        self.assertNotIn('forbidden-spack', '\n'.join(self.calls()))

    def test_real_failed_tests_retain_actual_exit_and_never_pass(self):
        result = self.run_receipted(environment={'FIXTURE_CTEST_RC': '8'})
        self.assertEqual(result.returncode, 8, result.stderr)
        _, outcome = self.outcome()
        self.assertEqual(outcome['returncode'], 8)
        self.assertEqual(outcome['status'], 'failed')
        self.assertIs(outcome['success'], False)
        self.assertIs(outcome['identity_verified_before_and_after'], True)
        self.assertEqual(len(self.calls()), 1)

    def test_real_result_rejects_changed_artifact_before_any_test(self):
        self.exe.write_bytes(self.exe.read_bytes() + b'changed')
        result = self.run_receipted()
        self.assertEqual(result.returncode, 3)
        _, outcome = self.outcome()
        self.assertEqual(outcome['status'], 'failed')
        self.assertIsNone(outcome['returncode'])
        self.assertIn('changed', outcome['error'])
        self.assertEqual(self.calls(), [])

    def test_successful_child_with_changed_binary_or_receipt_never_passes(self):
        for artifact in (self.exe, self.build / prebuilt.RECEIPT):
            with self.subTest(artifact=artifact):
                original = artifact.read_bytes()
                result = self.run_receipted(environment={'FIXTURE_MUTATE_PATH': str(artifact)})
                self.assertEqual(result.returncode, 3, result.stderr)
                _, outcome = self.outcome()
                self.assertEqual(outcome['returncode'], 0)
                self.assertEqual(outcome['status'], 'failed')
                self.assertIs(outcome['success'], False)
                self.assertIn('changed', outcome['error'])
                artifact.write_bytes(original)
                shutil.rmtree(self.root / 'results')
                (self.root / 'calls').unlink()

    def test_receipt_empty_selection_and_noop_overrides_reject_without_tests(self):
        result = self.run_receipted('--test-regex', 'missing')
        self.assertEqual(result.returncode, 3)
        _, outcome = self.outcome()
        self.assertIs(outcome['success'], False)
        self.assertIn('selected no tests', outcome['error'])
        self.assertEqual(self.calls(), [])
        shutil.rmtree(self.root / 'results')
        for option in ('--print-plan', '--submit', '--help', '--no-tests', '--clean'):
            with self.subTest(option=option):
                result = self.run_receipted(option)
                self.assertEqual(result.returncode, 3)
                self.assertFalse((self.root / 'results').exists())
                self.assertEqual(self.calls(), [])

    def test_receipt_requires_exact_command_scope_and_separate_regular_result_root(self):
        for args in (('--output-root', str(self.root / 'other')),
                     ('--test-regex', 'numerical', '--test-regex', 'numerical')):
            result = self.run_receipted(*args)
            self.assertEqual(result.returncode, 3)
            self.assertFalse((self.root / 'results').exists())
        result = self.run_receipted(result_root=self.build / 'result')
        self.assertEqual(result.returncode, 3)
        self.assertFalse((self.build / 'result').exists())
        (self.root / 'results').symlink_to(self.root / 'external')
        result = self.run_receipted()
        self.assertEqual(result.returncode, 3)
        self.assertFalse((self.root / 'external').exists())
        self.assertEqual(self.calls(), [])

    def test_successive_attempts_preserve_old_exact_outcome(self):
        self.assertEqual(self.run_receipted().returncode, 0)
        path, _ = self.outcome()
        original = path.read_bytes()
        self.assertEqual(self.run_receipted().returncode, 0)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(len(list((self.root / 'results').glob('*/result.json'))), 2)

    def test_zero_exit_skipped_or_missing_test_outcomes_never_pass(self):
        for environment in ({'FIXTURE_CTEST_STATUS': '***Skipped'},
                            {'FIXTURE_CTEST_STATUS': '***Not Run (Disabled)'},
                            {'FIXTURE_NO_OUTCOME': '1'}):
            with self.subTest(environment=environment):
                result = self.run_receipted(environment=environment)
                self.assertEqual(result.returncode, 3, result.stderr)
                _, outcome = self.outcome()
                self.assertEqual(outcome['returncode'], 0)
                self.assertIs(outcome['success'], False)
                self.assertIn('actual Passed', outcome['error'])
                shutil.rmtree(self.root / 'results')
                (self.root / 'calls').unlink()

    def test_disabled_selected_graph_rejects_before_child(self):
        # Pin the disabled graph as part of the genuine prebuilt metadata.
        with mock.patch.dict(os.environ, self.environment | {'FIXTURE_DISABLED': '1'}, clear=True):
            prebuilt.write(self.args)
        result = self.run_receipted(environment={'FIXTURE_DISABLED': '1'})
        self.assertEqual(result.returncode, 3, result.stderr)
        _, outcome = self.outcome()
        self.assertIsNone(outcome['returncode'])
        self.assertIn('disabled test', outcome['error'])
        self.assertEqual(self.calls(), [])

    def test_interruption_leaves_started_attempt_without_passed_final_receipt(self):
        real_run = subprocess.run
        command = [str(self.repo / 'build.sh'), '--cpu-only', '--tests-only',
                   '--output-root', str(self.output)]
        def interrupt(argv, **kwargs):
            if argv == command:
                raise KeyboardInterrupt()
            return real_run(argv, **kwargs)
        with mock.patch.dict(os.environ, self.environment, clear=True), \
                mock.patch.object(test_results.subprocess, 'run', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                test_results.run(self.build / prebuilt.RECEIPT, self.root / 'results', command)
        self.assertEqual(list((self.root / 'results').glob('*/result.json')), [])
        started = list((self.root / 'results').glob('*/started.json'))
        self.assertEqual(len(started), 1)
        self.assertEqual(json.loads(started[0].read_text())['status'], 'started')
        self.assertEqual(self.calls(), [])

    def allocation_fixture(self, *, scheduler_changes=None, gpu_rows=None, environment=None):
        saved = {'policy': {'site': 'pace', 'profile': 'cuda-sm80'}}
        env = {name: '' for name in test_results.SLURM_KEYS}
        env.update(SLURM_JOB_ID='123', SLURM_JOB_ACCOUNT='gts-sbryngelson3',
                   SLURM_JOB_QOS='embers', SLURM_JOB_PARTITION='gpu-a100',
                   SLURM_NTASKS='1', SLURM_STEP_GPUS='5', CUDA_VISIBLE_DEVICES='0')
        env.update(environment or {})
        fields = {'JobId': '123', 'JobState': 'RUNNING', 'Account': 'gts-sbryngelson3',
                  'QOS': 'embers', 'Partition': 'gpu-a100', 'BatchFlag': '0',
                  'UserId': f'{pwd.getpwuid(os.getuid()).pw_name}({os.getuid()})',
                  'AllocTRES': 'cpu=8,mem=48G,gres/gpu=1,gres/gpu:a100=1'}
        fields.update(scheduler_changes or {})
        gpu = gpu_rows or 'GPU-12345678-abcd-abcd-abcd-123456789abc, NVIDIA A100-SXM4-80GB, 8.0\n'
        def query(argv, **kwargs):
            if argv[0] == 'scontrol':
                self.assertEqual(argv, ['scontrol', 'show', 'job', '-o', '123'])
                return subprocess.CompletedProcess(argv, 0, ' '.join(key + '=' + value for key, value in fields.items()), '')
            self.assertEqual(argv, ['nvidia-smi', '--query-gpu=uuid,name,compute_cap', '--format=csv,noheader'])
            self.assertFalse(any(arg.startswith('--id') for arg in argv))
            return subprocess.CompletedProcess(argv, 0, gpu, '')
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(test_results.socket, 'getfqdn', return_value='atl1-compute.pace.gatech.edu'), \
                mock.patch.object(test_results.subprocess, 'run', side_effect=query):
            return test_results.allocation(saved)

    def test_real_live_interactive_allocation_does_not_need_job_completion(self):
        value = self.allocation_fixture()
        self.assertEqual(value['scheduler']['fields']['JobState'], 'RUNNING')
        self.assertEqual(value['scheduler']['fields']['BatchFlag'], '0')
        self.assertEqual(value['gpu']['uuid'], 'GPU-12345678-abcd-abcd-abcd-123456789abc')
        self.assertEqual(value['environment']['SLURM_STEP_GPUS'], '5')
        self.assertEqual(value['environment']['CUDA_VISIBLE_DEVICES'], '0')

    def test_live_allocation_rejects_unowned_paid_terminal_or_unassigned_identity(self):
        for changes in ({'UserId': 'unrelated(999)'}, {'Account': 'unrelated'},
                        {'QOS': 'inferno'}, {'JobState': 'COMPLETED'}, {'JobId': '321'},
                        {'AllocTRES': 'cpu=8,gres/gpu=2'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.allocation_fixture(scheduler_changes=changes)
        for environment in ({'CUDA_VISIBLE_DEVICES': '0,1'}, {'SLURM_STEP_GPUS': ''},
                            {'CUDA_VISIBLE_DEVICES': 'GPU-99999999'}, {'SLURM_JOB_QOS': 'inferno'}):
            with self.subTest(environment=environment), self.assertRaises(ValueError):
                self.allocation_fixture(environment=environment)

    def test_live_gpu_identity_rejects_multiple_devices_wrong_board_and_wrong_capability(self):
        valid = 'GPU-12345678-abcd-abcd-abcd-123456789abc, NVIDIA A100-SXM4-80GB, 8.0\n'
        for rows in (valid * 2, valid.replace('A100', 'V100'), valid.replace('8.0', '7.0')):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.allocation_fixture(gpu_rows=rows)

    def test_outcome_reader_requires_exact_all_selected_passes(self):
        graph = {'tests': [{'name': 'first'}, {'name': 'second'}]}
        text = (' 1/2 Test #10: first ........ Passed 0.01 sec\n'
                ' 2/2 Test #11: second ....... Passed 0.02 sec\n')
        self.assertEqual([row['name'] for row in test_results.test_outcomes(text, graph)], ['first', 'second'])
        for bad in (text.replace('Passed', '***Skipped', 1), text.splitlines()[0],
                    text + text.splitlines()[0], text.replace('second', 'unexpected'),
                    text.replace('2/2', '1/2')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                test_results.test_outcomes(bad, graph)

    def test_canonical_tests_only_never_configures_builds_or_installs(self):
        result = self.run_driver('--test-regex', '^numerical$', '--test-label', 'acceptance')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('no configure/build/install', result.stdout)
        self.assertEqual(len(self.calls()), 1)
        argv = json.loads(self.calls()[0])
        self.assertEqual(argv[-4:], ['-R', '^numerical$', '-L', 'acceptance'])

    def test_test_failure_propagates_without_build(self):
        result = self.run_driver(environment={'FIXTURE_CTEST_RC': '8'})
        self.assertEqual(result.returncode, 8, result.stderr)
        self.assertEqual(len(self.calls()), 1)

    def test_empty_regex_and_label_intersection_never_runs_a_test(self):
        for option in ('--test-regex', '--test-label'):
            with self.subTest(option=option):
                result = self.run_driver(option, 'missing')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('selected no tests', result.stderr)
                self.assertEqual(self.calls(), [])

    def test_missing_receipt_rejected_before_test_or_build(self):
        (self.build / prebuilt.RECEIPT).unlink()
        result = self.run_driver()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_stale_source_binary_configuration_and_dependency_rejected(self):
        for path in (self.repo / 'solver.cpp', self.exe, self.build / 'CMakeCache.txt',
                     self.env_path / 'spack.lock', self.minstall / 'lib/libmfem.a',
                     self.mbuild / 'dg-igr-mfem-identity-v1.txt'):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                result = self.run_driver()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.calls(), [])
                path.write_bytes(original)

    def test_changed_requested_policy_rejected(self):
        result = self.run_driver('--hdg-runtime')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_ignored_outputs_and_test_results_do_not_invalidate(self):
        self.put(self.repo / 'ignored/output', 'ignored output')
        self.put(self.repo / '__pycache__/fake.pyc', 'ignored bytecode')
        self.put(self.build / 'Testing/Temporary/LastTest.log', 'mutable CTest result')
        prebuilt.verify(self.args)

    def test_nonignored_untracked_source_is_pinned(self):
        self.put(self.repo / 'new.cpp', 'new source')
        with self.assertRaisesRegex(ValueError, 'changed'):
            prebuilt.verify(self.args)

    def test_external_mfem_allows_empty_uninitialized_gitlink_but_pins_it(self):
        pin = subprocess.check_output(['git', '-C', str(self.mfem), 'rev-parse', 'HEAD']).decode().strip()
        subprocess.run(['git', '-C', str(self.repo), 'update-index', '--add', '--cacheinfo',
                        '160000,' + pin + ',tpl/mfem'], check=True)
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=Fixture',
                        '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'gitlink'], check=True)
        empty = self.repo / 'tpl/mfem'
        empty.mkdir(parents=True)
        prebuilt.write(self.args)
        prebuilt.verify(self.args)
        self.put(empty / 'unrecorded.hpp', 'untracked submodule bytes')
        with self.assertRaisesRegex(ValueError, 'unrecorded files'):
            prebuilt.verify(self.args)

    def test_test_graph_command_changes_are_rejected(self):
        original = self.ctest.read_text()
        self.ctest.write_text(original.replace('retained-fixture', 'different-fixture'))
        with self.assertRaisesRegex(ValueError, 'changed'):
            prebuilt.verify(self.args)

    def test_objects_and_compiler_id_are_not_application_executables(self):
        self.exe.unlink()
        self.put(self.build / 'only.o', b'\x7fELF\x02\x01' + b'\0' * 10 + b'\x01\x00', executable=True)
        self.put(self.build / 'CMakeFiles/CompilerId/probe',
                 b'\x7fELF\x02\x01' + b'\0' * 10 + b'\x02\x00', executable=True)
        with self.assertRaisesRegex(ValueError, 'no ELF executable'):
            prebuilt.write(self.args)

    def test_runtime_library_symlink_retarget_is_rejected(self):
        for name in ('liba.so.1', 'liba.so.2'):
            self.put(self.mpi / 'lib' / name, 'same library bytes')
        link = self.mpi / 'lib/liba.so'
        link.symlink_to('liba.so.1')
        prebuilt.write(self.args)
        link.unlink()
        link.symlink_to('liba.so.2')
        with self.assertRaisesRegex(ValueError, 'changed'):
            prebuilt.verify(self.args)

    def source_mesh_fixture(self, alias=False):
        relative = Path('examples/dgsem_sedov_taylor_blast/sedov_taylor.msh')
        mesh = self.repo / ('examples/sedov_taylor_blast/sedov_taylor.msh' if alias else relative)
        self.put(mesh, '$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
        source_mesh = self.repo / relative
        if alias:
            source_mesh.parent.mkdir(parents=True, exist_ok=True)
            source_mesh.symlink_to('../sedov_taylor_blast/sedov_taylor.msh')
        subprocess.run(['git', '-C', str(self.repo), 'add', '--', 'examples'], check=True)
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=Fixture',
                        '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'tracked meshes'], check=True)
        link = self.build / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(mesh)
        return mesh, source_mesh, link

    def test_tracked_source_mesh_links_and_source_aliases_are_pinned(self):
        for alias in (False, True):
            with self.subTest(alias=alias):
                mesh, source_mesh, link = self.source_mesh_fixture(alias)
                receipt = prebuilt.write(self.args)
                self.assertEqual(receipt['artifacts']['links'][str(link)],
                                 {'link': str(mesh), 'target': str(mesh)})
                self.assertEqual(receipt['artifacts']['files'][str(mesh)]['sha256'],
                                 prebuilt.digest(mesh.read_bytes()))
                prebuilt.verify(self.args)
                result = self.run_driver('--test-regex', '^numerical$')
                self.assertEqual(result.returncode, 0, result.stderr)
                link.unlink()
                source_mesh.unlink()

    def test_source_mesh_bytes_and_literal_link_mapping_cannot_drift(self):
        mesh, _, link = self.source_mesh_fixture()
        prebuilt.write(self.args)
        original = mesh.read_bytes()
        mesh.write_bytes(original + b'changed mesh')
        with self.assertRaisesRegex(ValueError, 'changed'):
            prebuilt.verify(self.args)
        mesh.write_bytes(original)
        link.unlink()
        link.symlink_to(os.path.relpath(mesh, link.parent))
        with self.assertRaisesRegex(ValueError, 'changed'):
            prebuilt.verify(self.args)

    def test_wrong_tracked_mesh_target_is_rejected_before_publication(self):
        mesh, _, link = self.source_mesh_fixture()
        other = self.repo / 'examples/other/other.msh'
        self.put(other, mesh.read_bytes())
        subprocess.run(['git', '-C', str(self.repo), 'add', '--', str(other)], check=True)
        link.unlink()
        link.symlink_to(other)
        with self.assertRaisesRegex(ValueError, 'retargeted source mesh'):
            prebuilt.write(self.args)

    def test_unknown_untracked_and_outside_source_mesh_links_reject(self):
        mesh, _, link = self.source_mesh_fixture()
        for target in (self.repo / 'examples/unknown/unknown.msh', self.root / 'outside.msh'):
            self.put(target, mesh.read_bytes())
            link.unlink()
            link.symlink_to(target)
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'source mesh'):
                prebuilt.write(self.args)
        link.unlink()
        link.symlink_to(mesh)
        subprocess.run(['git', '-C', str(self.repo), 'rm', '--cached', '--', str(mesh)], check=True, capture_output=True)
        with self.assertRaisesRegex(ValueError, 'source mesh'):
            prebuilt.write(self.args)

    def test_runtime_executable_library_and_disguised_mesh_escapes_reject(self):
        for name in ('external-executable', 'libexternal.so'):
            external = self.root / name
            self.put(external, 'external runtime bytes', executable=True)
            link = self.build / name
            link.symlink_to(external)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'escapes its root'):
                prebuilt.write(self.args)
            link.unlink()
        mesh, _, _ = self.source_mesh_fixture()
        mesh.write_bytes(b'\x7fELF' + b'fake executable in mesh path')
        with self.assertRaisesRegex(ValueError, 'runtime artifact'):
            prebuilt.write(self.args)

    def test_source_drift_during_build_prevents_receipt_publication(self):
        self.args.expected_source = json.dumps(prebuilt.source_identity(self.repo))
        (self.repo / 'solver.cpp').write_text('changed during build')
        with self.assertRaisesRegex(ValueError, 'during the canonical build'):
            prebuilt.write(self.args)

    def test_clean_and_disabled_tests_are_rejected(self):
        for option in ('--clean', '--no-tests'):
            result = self.run_driver(option)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(self.calls(), [])


if __name__ == '__main__':
    unittest.main()
