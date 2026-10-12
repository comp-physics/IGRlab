#!/usr/bin/env python3
"""CPU admission fixtures: no application, MPI, Spack, or scheduler execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import study_admit as study
import study_import


class StudyAdmitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / 'source'
        self.plan = {'profile': 'cpu', 'source': {'source_snapshot': str(source), 'source_fingerprint': 'pinned'},
                     'configuration_key': 'cpu-key', 'correctness_regex': '^cpu-igr$', 'env': {},
                     'app_build': str(self.root / 'app/build'), 'app_install': str(self.root / 'app/install'),
                     'spack_root': str(self.root / 'spack-tool'), 'spack_env': str(self.root / 'env/cpu'),
                     'mfem_source': str(self.root / 'mfem/source'), 'mfem_build': str(self.root / 'mfem/build'),
                     'mfem_install': str(self.root / 'mfem/install'),
                     'mfem_cache': {'output_root': str(self.root / 'mfem')}}
        self.plan.update(mfem_config=self.plan['mfem_install'] + '/lib/cmake/mfem/MFEMConfig.cmake',
                         mfem_targets=self.plan['mfem_install'] + '/lib/cmake/mfem/MFEMTargets.cmake')
        self.request = self.root / 'request.json'
        self.request.write_text(json.dumps({'source': str(source), 'argv': ['--cpu-only']}))
        self.receipt = {'success': True, 'returncode': 0, 'build_job_id': '123', 'validation_job_id': '456',
                        'source_fingerprint': 'pinned', 'configuration_key': 'cpu-key',
                        'stdout': f'PACE build submitted: 123 (embers, cpu-small)\n'
                                  f'PACE validation submitted: 456 (prebuilt validation, embers, cpu-small)\n'
                                  f'Logs: /site/logs; request: {self.request}\n'}
        self.locations = {name: str(self.root / 'installed' / name)
                          for name in ('hypre', 'openmpi', 'cmake', 'openblas', 'metis')}
        for prefix in self.locations.values():
            Path(prefix).mkdir(parents=True)
        self.binary = Path(self.plan['app_build']) / 'examples/app'
        self.write(self.binary, b'\x7fELFfake CPU binary', executable=True)
        mpi, hypre = Path(self.locations['openmpi']), Path(self.locations['hypre'])
        for relative in ('bin/mpicxx', 'bin/mpiexec', 'lib/libmpi.so'):
            self.write(mpi / relative, b'fixture', executable=True)
        for relative in ('include/HYPRE.h', 'include/HYPRE_config.h', 'lib/libHYPRE.so'):
            self.write(hypre / relative, b'fixture')
        cache = f'''CMAKE_BUILD_TYPE:STRING=Release
MFEM_USE_CUDA:BOOL=OFF
MFEM_USE_HIP:BOOL=OFF
HYPRE_DIR:PATH={hypre}
HYPRE_INCLUDE_DIR:PATH={hypre}/include
HYPRE_INCLUDE_DIRS:STRING={hypre}/include
HYPRE_LIBRARY:FILEPATH={hypre}/lib/libHYPRE.so
HYPRE_LIBRARIES:STRING={hypre}/lib/libHYPRE.so
CMAKE_CXX_COMPILER:FILEPATH={mpi}/bin/mpicxx
MPI_CXX_COMPILER:FILEPATH={mpi}/bin/mpicxx
'''
        self.write(Path(self.plan['mfem_build']) / 'CMakeCache.txt', cache.encode())
        self.write(Path(self.plan['mfem_config']), f'set(MFEM_INCLUDE_DIRS "{hypre}/include")'.encode())
        self.write(Path(self.plan['mfem_targets']), f'INTERFACE_INCLUDE_DIRECTORIES "{hypre}/include"\n'
                   f'INTERFACE_LINK_LIBRARIES "{hypre}/lib/libHYPRE.so"'.encode())
        self.policy = {'mode': 'cpu', 'profile': 'cpu', 'build_type': 'Release', 'site': 'pace',
                       'caliper': '0', 'nvtx_profile': '0', 'hdg_runtime': 'ON',
                       'hdg_auxiliary': 'OFF', 'hdg_euler': 'OFF', 'hdg_euler_device': 'OFF',
                       'research_tests': 'OFF', 'dependency_cuda_arch': 'none',
                       'mfem_cuda_archs': 'none', 'app_cuda_archs': 'none', 'kernel_sources': '',
                       'mfem_seed': '', 'mfem_output_root': self.plan['mfem_cache']['output_root']}
        self.saved = {'policy': self.policy, 'paths': {name: self.plan[name]
                       for name in ('mfem_source', 'mfem_build', 'mfem_install')},
                      'artifacts': {'files': {str(self.binary): {'sha256': study.sha(self.binary)}}, 'links': {}}}
        self.receipt_path = Path(self.plan['app_build']) / 'igrlab-prebuilt-validation-v1.json'
        self.write(self.receipt_path, json.dumps(self.saved).encode())
        checker_path = Path(__file__).resolve().parents[1] / 'build/check_dependency_bindings.py'
        spec = importlib.util.spec_from_file_location('real_checker', checker_path)
        self.checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.checker)
        self.verifier = SimpleNamespace(RECEIPT=self.receipt_path.name)
        self.verifier.verify = lambda args: self.saved
        self.calls = []
        self.ldd = (f'libHYPRE.so => {hypre}/lib/libHYPRE.so (0x1)\n'
                    f'libmpi.so => {mpi}/lib/libmpi.so (0x2)\n')
        self.accounting = self.accounting_text()

    def write(self, path, contents, executable=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        if executable:
            path.chmod(0o755)

    def accounting_text(self):
        source = self.plan['source']['source_snapshot']
        user = pwd.getpwuid(os.getuid()).pw_name
        return '\n'.join(f'{job}|COMPLETED|0:0|{user}|gts-sbryngelson3|embers|{name}|'
                         f'sbatch {source}/scripts/build/pace.sh --worker {stage} {self.request}'
                         for job, stage, name in [('123', 'build', 'IGRlab-build'),
                                                 ('456', 'validation', 'IGRlab-check')])

    def fake_run(self, argv, env=None):
        self.calls.append((argv, env))
        if argv[0] == 'sacct':
            output = self.accounting
        elif argv[0] == 'ldd':
            output = self.ldd
        else:
            self.assertEqual(argv[-3:-1], ['location', '-i'])
            self.assertEqual(argv[2], self.plan['spack_env'])
            output = self.locations[argv[-1]] + '\n'
        return subprocess.CompletedProcess(argv, 0, output, '')

    def admission(self):
        with patch.object(study.study_build, 'verify_source'), \
             patch.object(study.study_build, 'existing_cpu_plan', return_value=self.plan), \
             patch.object(study, 'module', side_effect=lambda source, name: self.checker
                          if name == 'check_dependency_bindings' else self.verifier), \
             patch.object(study, 'run', side_effect=self.fake_run):
            return study.admit_build(self.plan, ['app'], self.receipt)

    def test_cpu_artifacts_use_dynamic_cpu_prefixes_and_selected_rank_one_launcher(self):
        admission = self.admission()
        argv, env = study.runtime_command(admission, [str(self.binary), '--device', 'cpu'], {'PHYSICS': 'same'})
        self.assertEqual(argv, [self.locations['openmpi'] + '/bin/mpiexec', '-n', '1',
                                str(self.binary), '--device', 'cpu'])
        self.assertTrue(env['LD_LIBRARY_PATH'].startswith(self.locations['openmpi'] + '/lib:'))
        self.assertEqual(env['PHYSICS'], 'same')
        self.assertEqual(env['OMP_NUM_THREADS'], '1')
        self.assertNotIn('cuda', admission['spack_locations'])
        self.assertIn(str(self.binary), admission['artifact_sha256'])
        self.assertIn('123', admission['canonical_jobs']['accounting'])

    def test_unrelated_or_failed_jobs_reject_before_spack_or_artifact_admission(self):
        original = self.accounting
        for old, new in [('COMPLETED|0:0', 'FAILED|8:0'), ('embers', 'paid'),
                         ('IGRlab-check', 'unrelated-check'), ('--worker validation', '--worker build'),
                         (str(self.request), '/unrelated/request.json')]:
            with self.subTest(new=new):
                self.calls.clear()
                self.accounting = original.replace(old, new)
                with self.assertRaises(ValueError):
                    self.admission()
                self.assertEqual([call[0][0] for call in self.calls], ['sacct'])

    def test_substituted_job_id_and_source_request_reject(self):
        self.receipt['validation_job_id'] = '999'
        with self.assertRaisesRegex(ValueError, 'ID does not match'):
            self.admission()
        self.receipt['validation_job_id'] = '456'
        self.request.write_text(json.dumps({'source': '/different/source', 'argv': ['--cpu-only']}))
        with self.assertRaisesRegex(ValueError, 'request source differs'):
            self.admission()

    def test_submission_source_configuration_and_exact_worker_request_tokens_reject(self):
        for key in ('source_fingerprint', 'configuration_key'):
            original = self.receipt[key]
            self.receipt[key] = 'unrelated'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'identity differs'):
                self.admission()
            self.receipt[key] = original
        self.accounting = self.accounting.replace(str(self.request), str(self.request) + '.other')
        with self.assertRaisesRegex(ValueError, 'different request'):
            self.admission()

    def test_cuda_profile_receipt_or_mfem_rejects(self):
        self.plan['profile'] = 'cuda-sm80'
        with self.assertRaisesRegex(ValueError, 'explicit cpu'):
            self.admission()
        self.plan['profile'] = 'cpu'
        self.saved['policy']['mode'] = 'gpu'
        self.receipt_path.write_text(json.dumps(self.saved))
        with self.assertRaisesRegex(ValueError, 'CPU-only'):
            self.admission()
        self.saved['policy']['mode'] = 'cpu'
        self.receipt_path.write_text(json.dumps(self.saved))
        cache = Path(self.plan['mfem_build']) / 'CMakeCache.txt'
        cache.write_text(cache.read_text().replace('MFEM_USE_CUDA:BOOL=OFF', 'MFEM_USE_CUDA:BOOL=ON'))
        with self.assertRaisesRegex(ValueError, 'Release CPU build'):
            self.admission()

    def test_canonical_verifier_failure_is_not_replaced_by_performance_receipt(self):
        def reject(args):
            raise ValueError('canonical artifacts changed')
        self.verifier.verify = reject
        with self.assertRaisesRegex(ValueError, 'canonical artifacts changed'):
            self.admission()

    def test_imported_reuse_audit_is_preserved_with_full_cpu_artifact_checks(self):
        self.plan.update(origin=study_import.ORIGIN, reuse_validated_artifacts=True,
                         test_result='/retained/passed/result.json', test_result_sha256='sealed-result')
        authority = {'query_performed': False, 'query_status': 'not_queried',
                     'authority': 'canonical prebuilt receipt and sealed actual Passed test outcomes'}
        observed = {'job_ids': {'build': '123'}, 'request': str(self.request),
                    'request_sha256': study.sha(self.request),
                    'build_accounting': {**authority, 'historical_job_id': '123'},
                    'test_allocation_accounting': {**authority, 'historical_job_id': '789'},
                    'test_execution_scope': 'sealed outcomes; historical accounting not queried'}
        with patch.object(study_import, 'completed_records', return_value=observed) as retained:
            admission = self.admission()
            retained.assert_called_once_with(self.plan, self.receipt)
            audit = admission['canonical_jobs']
            self.assertEqual(audit['build_accounting'], observed['build_accounting'])
            self.assertEqual(audit['test_allocation_accounting'], observed['test_allocation_accounting'])
            self.assertNotIn('accounting', audit)
            self.assertNotIn('sacct', [argv[0] for argv, _ in self.calls])
            self.assertIn('loader', admission['dependency_bindings'])
            self.assertEqual(admission['artifact_sha256'][str(self.binary)], study.sha(self.binary))
            self.verifier.verify = lambda args: (_ for _ in ()).throw(ValueError('canonical artifacts changed'))
            with self.assertRaisesRegex(ValueError, 'canonical artifacts changed'):
                self.admission()

    def test_stale_loader_or_cuda_link_rejects(self):
        self.ldd = self.ldd.replace(self.locations['openmpi'], str(self.root / 'other-mpi'))
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.admission()
        self.ldd = 'libcudart.so => /cuda/lib/libcudart.so (0x1)\n'
        with self.assertRaisesRegex(ValueError, 'CUDA runtime'):
            self.admission()

    def test_ambiguous_or_unreceipted_binary_rejects(self):
        self.write(Path(self.plan['app_build']) / 'other/app', b'\x7fELFduplicate', executable=True)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self.admission()
        (Path(self.plan['app_build']) / 'other/app').unlink()
        self.saved['artifacts']['files'] = {}
        with self.assertRaisesRegex(ValueError, 'absent from canonical'):
            self.admission()

    def test_changed_runtime_binary_and_unadmitted_command_reject(self):
        admission = self.admission()
        with self.assertRaisesRegex(ValueError, 'admitted CPU binary'):
            study.runtime_command(admission, ['/cuda/app'], {})
        self.binary.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed after admission'):
            study.runtime_command(admission, [str(self.binary)], {})

    def test_each_cpu_child_hashes_only_selected_binary_and_launcher(self):
        admission = self.admission()
        with patch.object(study, 'sha', wraps=study.sha) as fingerprint:
            study.runtime_command(admission, [str(self.binary)], {})
        self.assertEqual([call.args[0] for call in fingerprint.call_args_list],
                         [str(self.binary), admission['runtime']['launcher']])

    def test_runtime_environment_returns_only_explicit_overrides_not_inherited_secrets(self):
        with patch.dict(os.environ, {'SECRET_FIXTURE': 'never-record', 'LD_LIBRARY_PATH': '/gpu/runtime'}):
            env = study.runtime_environment(self.locations, {'PHYSICS': 'keep'})
        self.assertNotIn('SECRET_FIXTURE', env)
        self.assertTrue(env['LD_LIBRARY_PATH'].endswith('/gpu/runtime'))


if __name__ == '__main__':
    unittest.main()
