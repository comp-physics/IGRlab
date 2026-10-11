#!/usr/bin/env python3
"""Host-only run-only admission contracts; no application is compiled or run."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prebuilt', SOURCE / 'scripts/build/prebuilt_validation.py')
prebuilt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prebuilt)


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
    print(json.dumps({'kind': 'ctestInfo', 'tests': tests}))
elif '-N' in args:
    if not any('missing' in a for a in args):
        print('  Test #1: numerical')
    print('Total Tests: 1')
else:
    with open(os.environ['CALLS'], 'a') as f:
        f.write(json.dumps(args) + '\\n')
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
