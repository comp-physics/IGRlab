#!/usr/bin/env python3
"""Exercise build.sh's app-only route using fake CMake/CTest, never compilers."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[2]


class AppIteration(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / 'output'
        self.worktree = 'wt-' + hashlib.sha256(str(SOURCE).encode()).hexdigest()[:16]
        self.build = self.output / 'worktrees' / self.worktree / 'cuda-sm80/app/build'
        self.build.mkdir(parents=True)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.log = self.root / 'calls.jsonl'
        fake_tool = '''#!/usr/bin/env python3
import json, os, pathlib, sys
with open(os.environ['ITERATION_TEST_LOG'], 'a') as stream:
    stream.write(json.dumps({'tool': pathlib.Path(sys.argv[0]).name,
                            'argv': sys.argv[1:]}) + '\\n')
raise SystemExit(int(os.environ.get('ITERATION_TEST_EXIT', '0')))
'''
        for tool in ('cmake', 'ctest', 'spack'):
            path = self.bin / tool
            path.write_text(fake_tool)
            path.chmod(0o755)
        self.cache = {
            'CMAKE_HOME_DIRECTORY': str(SOURCE),
            'CMAKE_CACHEFILE_DIR': str(self.build),
            'CMAKE_CUDA_ARCHITECTURES': '80',
            'CMAKE_COMMAND': str(self.bin / 'cmake'),
        }
        self.write_cache()
        self.receipt = self.build / 'igrlab-prebuilt-validation-v1.json'
        self.receipt.write_text('old qualification\n')
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(('IGRLAB_', 'DG_IGR_', 'SPACK_', 'SLURM_'))
                    and key not in ('CUDA_ARCHS', 'MFEM_CUDA_ARCHS', 'APP_CUDA_ARCHS')}
        self.env.update(PATH=str(self.bin) + os.pathsep + self.env['PATH'],
                        ITERATION_TEST_LOG=str(self.log))

    def tearDown(self):
        self.temporary.cleanup()

    def write_cache(self):
        (self.build / 'CMakeCache.txt').write_text(''.join(
            f'{key}:STRING={value}\n' for key, value in self.cache.items()))

    def run_driver(self, *args, iterate=True):
        return subprocess.run(
            ['bash', str(SOURCE / 'build.sh'), '--a100', '--output-root',
             str(self.output), '--jobs', '2', *(['--iterate'] if iterate else []), *args],
            env=self.env, capture_output=True, text=True, timeout=10)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_only_selected_application_targets_are_built(self):
        result = self.run_driver('--target', 'dgsem_smooth_advection_periodic',
                                 '--target', 'dg_igr_exact_face_test')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), [{'tool': 'cmake', 'argv': [
            '--build', str(self.build), '-j', '2', '--target',
            'dgsem_smooth_advection_periodic', 'dg_igr_exact_face_test']}])
        self.assertFalse(self.receipt.exists())

    def test_default_build_has_no_tests_install_or_dependency_work(self):
        result = self.run_driver()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), [{'tool': 'cmake', 'argv': [
            '--build', str(self.build), '-j', '2']}])
        self.assertIn('dependency_work=none artifact_hashes=none run_tests=0', result.stdout)
        self.assertFalse((self.build.parent.parent / 'mfem').exists())

    def test_failed_build_invalidates_old_receipt_and_propagates_failure(self):
        self.env['ITERATION_TEST_EXIT'] = '17'
        result = self.run_driver('--target', 'dgsem_smooth_advection_periodic')
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertFalse(self.receipt.exists())
        self.assertEqual(len(self.calls()), 1)

    def test_explicit_tests_run_once_without_build_or_receipt_wrapper(self):
        result = self.run_driver('--tests-only', '--test-regex', '^selected$', '--test-label', 'cuda')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), [{'tool': 'ctest', 'argv': [
            '--test-dir', str(self.build), '--output-on-failure', '--no-tests=error',
            '-R', '^selected$', '-L', 'cuda']}])
        self.assertEqual(self.receipt.read_text(), 'old qualification\n')

    def test_test_failures_propagate(self):
        self.env['ITERATION_TEST_EXIT'] = '8'
        result = self.run_driver('--tests-only', '--test-regex', '^selected$')
        self.assertEqual(result.returncode, 8, result.stderr)
        self.assertEqual([call['tool'] for call in self.calls()], ['ctest'])

    def test_missing_or_incompatible_cache_fails_without_dependency_fallback(self):
        cache_path = self.build / 'CMakeCache.txt'
        original = self.cache.copy()
        mutations = [None, ('CMAKE_HOME_DIRECTORY', '/different/checkout'),
                     ('CMAKE_CACHEFILE_DIR', '/moved/build'),
                     ('CMAKE_CUDA_ARCHITECTURES', '120'), ('CMAKE_COMMAND', '/unavailable/cmake')]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.cache = original.copy()
                if mutation is None:
                    cache_path.unlink()
                else:
                    self.cache[mutation[0]] = mutation[1]
                    self.write_cache()
                result = self.run_driver()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.calls(), [])
                self.assertTrue(self.receipt.exists())
                self.cache = original.copy()
                self.write_cache()

    def test_incompatible_flags_and_non_application_targets_are_rejected(self):
        for args in [('--clean',), ('--smoke',), ('--tests-only',),
                     ('--test-regex', 'x'), ('--tests-only', '--test-regex', 'x', '--target', 'app'),
                     *[('--target', target) for target in ('clean', 'install', 'install/strip',
                                                          'package', 'test', '--help', '../app')]]:
            with self.subTest(args=args):
                self.assertNotEqual(self.run_driver(*args).returncode, 0)
        self.assertNotEqual(self.run_driver('--target', 'app', iterate=False).returncode, 0)
        self.assertEqual(self.calls(), [])
        self.assertTrue(self.receipt.exists())

    def test_print_plan_does_not_create_outputs_or_touch_receipt(self):
        before = {path: path.stat().st_mtime_ns for path in self.root.rglob('*')}
        result = self.run_driver('--target', 'app', '--print-plan')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, {path: path.stat().st_mtime_ns for path in self.root.rglob('*')})
        self.assertEqual(self.calls(), [])

    def test_held_output_lock_prevents_build_and_receipt_invalidation(self):
        lock = self.output / 'locks' / f'{self.worktree}-cuda-sm80.lock'
        lock.parent.mkdir()
        with lock.open('w') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_driver()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('output lock is already held', result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertTrue(self.receipt.exists())


if __name__ == '__main__':
    unittest.main()
