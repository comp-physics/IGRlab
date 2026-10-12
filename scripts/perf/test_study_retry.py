#!/usr/bin/env python3
"""Host-only retry fixtures: no build, application, profiler or real scheduler."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import gpu_study as study


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.old, self.new = self.base / 'old', self.base / 'new'
        self.old.mkdir()
        (self.old / '.study.lock').touch()
        (self.old / 'executor').mkdir()
        for name in study.EXECUTOR_FILES:
            (self.old / 'executor' / name).write_text('historical frozen executor\n')
        suite = json.loads((Path(study.__file__).parent / 'suites/pace-a100.json').read_text())
        for case in suite['cases']:
            case['enabled'] = case['id'] == 'dg-p1'
        study.write_json(self.old / 'suite.json', suite)
        self.cache = self.base / 'cache'
        self.source = self.cache / 'source'
        self.source.mkdir(parents=True)
        (self.source / 'code.cpp').write_text('immutable application source\n')
        identity = {'source_snapshot': str(self.source),
                    'source_fingerprint': study.sha(self.source / 'code.cpp')}
        self.app, self.mfem, self.install, self.env = [self.cache / name for name in
                                                     ('app', 'mfem', 'install', 'environment')]
        for directory in (self.app, self.mfem, self.install, self.env):
            directory.mkdir()
        for path in (self.app / 'CMakeCache.txt', self.mfem / 'CMakeCache.txt',
                     self.mfem / 'dg-igr-mfem-identity-v1.txt', self.env / 'spack.yaml',
                     self.env / 'spack.lock', self.install / 'MFEMConfig.cmake',
                     self.install / 'MFEMTargets.cmake', self.install / 'libmfem.a'):
            path.write_text('completed build metadata\n')
        self.binary = self.app / study.study_workloads.BINARY_NAMES['dg']
        self.binary.write_text('saved executable bytes, never executed\n')
        self.binary.chmod(0o755)
        build = {'app_build': str(self.app), 'app_install': str(self.install),
                 'mfem_build': str(self.mfem), 'mfem_install': str(self.install),
                 'spack_env': str(self.env), 'configuration_key': 'config', 'argv': ['canonical-build']}
        self.plan = {'schema_version': 1, 'site': 'pace', 'run_dir': str(self.old),
                     'site_root': str(self.base / 'site'), 'spack_root': str(self.base / 'spack'),
                     'cache_root': str(self.cache), 'sources': {m: identity for m in study.MEMBERS},
                     'builds': {m: build for m in study.MEMBERS},
                     'suite_sha256': study.sha(self.old / 'suite.json'),
                     'executor_sha256': {n: study.sha(self.old / 'executor' / n) for n in study.EXECUTOR_FILES}}
        self.save_plan()
        receipt = {'success': True, 'returncode': 0, 'build_job_id': '11', 'validation_job_id': '12',
                   'source_fingerprint': identity['source_fingerprint'],
                   'configuration_key': 'config', 'argv': ['canonical-build'],
                   'stdout': 'canonical submission receipt', 'stderr': ''}
        attempt = self.old / 'attempts' / 'old-measurement'
        attempt.mkdir(parents=True)
        study.write_json(attempt / 'summary.json', {'old_samples': [1, 2, 3]})
        self.state = {'schema_version': 1, 'builds': {m: receipt for m in study.MEMBERS},
                      'build_attempts': [{'member': 'baseline', **receipt}],
                      'jobs': [{'stage': 'timings', 'job_id': '33'}, {'stage': 'profiles', 'job_id': '34'}],
                      'attempts': [{'stage': 'timings', 'status': 'timings_complete', 'job_id': '33',
                                    'directory': str(attempt), 'summary': str(attempt / 'summary.json'),
                                    'artifact_sha256': {str((attempt / 'summary.json').relative_to(self.old)):
                                                        study.sha(attempt / 'summary.json')}}]}
        self.save_state()
        self.outcomes = {'11': ('COMPLETED', '0:0'), '12': ('COMPLETED', '0:0'),
                         '33': ('FAILED', '1:0'), '34': ('CANCELLED', '0:0')}
        self.queue_stdout, self.queue_code, self.profile_code = '', 0, 0
        self.calls = []
        def verify_source(identity):
            if study.sha(Path(identity['source_snapshot']) / 'code.cpp') != identity['source_fingerprint']:
                raise ValueError('frozen source changed')
        self.source_patch = patch.object(study.study_build, 'verify_source', side_effect=verify_source)
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.run_patch = patch.object(study.subprocess, 'run', side_effect=self.scheduler)
        self.run_patch.start()
        self.addCleanup(self.run_patch.stop)

    def tearDown(self):
        self.temp.cleanup()

    def save_plan(self):
        self.plan.pop('plan_sha256', None)
        self.plan['plan_sha256'] = study.digest(self.plan)
        study.write_json(self.old / 'plan.json', self.plan)

    def save_state(self):
        study.write_json(self.old / 'state.json', self.state)

    def scheduler(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[0] == 'squeue':
            return subprocess.CompletedProcess(argv, self.queue_code, self.queue_stdout, 'scheduler unavailable')
        if argv[0] == 'sacct':
            ids = argv[argv.index('-j') + 1].split(',')
            stdout = ''.join(f'{job}|{self.outcomes[job][0]}|{self.outcomes[job][1]}|\n' for job in ids)
            return subprocess.CompletedProcess(argv, 0, stdout, '')
        if argv[0] == 'sbatch':
            profiles = '--job-name=IGRlab-profiles' in argv
            return subprocess.CompletedProcess(argv, self.profile_code if profiles else 0,
                                               '102\n' if profiles else '101\n', 'profile submit error' if profiles else '')
        self.fail(f'unexpected external command: {argv}')

    def retry(self, *, prepare=True, **changes):
        args = argparse.Namespace(from_run=str(self.old), run_dir=str(self.new),
                                  prepare_only=prepare, timings_time=None, profiles_time=None)
        for key, value in changes.items(): setattr(args, key, value)
        with contextlib.redirect_stdout(io.StringIO()): study.retry(args)
        return study.load_plan(self.new)

    def snapshot(self):
        return {str(p.relative_to(self.old)): p.read_bytes() for p in self.old.rglob('*') if p.is_file()}

    def test_prepare_reuses_exact_sources_builds_suite_and_receipts_without_measurement_mix(self):
        before = self.snapshot()
        plan = self.retry()
        self.assertEqual(before, self.snapshot())
        self.assertEqual(plan['sources'], self.plan['sources'])
        self.assertEqual(plan['builds'], self.plan['builds'])
        self.assertEqual((self.new / 'suite.json').read_bytes(), (self.old / 'suite.json').read_bytes())
        self.assertEqual(plan['retry_origin']['plan_sha256'], self.plan['plan_sha256'])
        self.assertEqual(plan['retry_origin']['suite_sha256'], self.plan['suite_sha256'])
        self.assertEqual(plan['suite_sha256'], self.plan['suite_sha256'])
        self.assertEqual(plan['retry_origin']['binding_scope'], 'fresh compute admission required')
        self.assertNotEqual(plan['plan_sha256'], self.plan['plan_sha256'])
        self.assertNotEqual(plan['executor_sha256'], self.plan['executor_sha256'])
        state = study.read_json(self.new / 'state.json')
        self.assertEqual(state['builds'], self.state['builds'])
        self.assertEqual((state['jobs'], state['attempts']), ([], []))
        self.assertFalse((self.new / 'admission.json').exists())
        self.assertFalse((self.new / 'attempts').exists())
        self.assertTrue(all(command[0] in ('squeue', 'sacct') for command in self.calls))

    def test_selected_optional_cases_reuse_builds_and_pin_new_benchmark_without_compilation(self):
        benchmark = self.app / study.study_workloads.BINARY_NAMES['dgsem']
        benchmark.write_text('previously unmeasured built benchmark\n')
        benchmark.chmod(0o755)
        before = self.snapshot()
        with patch.object(study.study_build, 'submit_build', side_effect=AssertionError('must not build')):
            plan = self.retry(cases='dgsem-n4,dgsem-n8')
        self.assertEqual(before, self.snapshot())
        self.assertEqual(plan['builds'], self.plan['builds'])
        self.assertEqual(plan['sources'], self.plan['sources'])
        suite = study.study_workloads.load_suite(self.new / 'suite.json')
        self.assertEqual([case['id'] for case in study.selected_cases(suite)], ['dgsem-n4', 'dgsem-n8'])
        self.assertEqual(plan['suite_sha256'], study.sha(self.new / 'suite.json'))
        self.assertNotEqual(plan['suite_sha256'], self.plan['suite_sha256'])
        self.assertEqual(plan['retry_origin']['suite_sha256'], self.plan['suite_sha256'])
        self.assertEqual(plan['retry_origin']['build_artifact_sha256'][str(benchmark)], study.sha(benchmark))
        study.verify(self.new)
        benchmark.write_text('changed after retry freeze\n')
        with self.assertRaisesRegex(RuntimeError, 'completed build artifact changed'):
            study.verify(self.new)

    def test_replacement_suite_and_selection_are_frozen_with_same_correctness(self):
        suite = study.read_json(self.old / 'suite.json')
        suite['name'] = 'new-measurement-parameters'
        suite['repetitions'] = 5
        suite['cases'][0]['parameters']['final_time'] = 0.004
        path = self.base / 'new-suite.json'
        study.write_json(path, suite)
        before = self.snapshot()
        plan = self.retry(suite=str(path), cases='dg-p1')
        self.assertEqual(before, self.snapshot())
        frozen = study.study_workloads.load_suite(self.new / 'suite.json')
        self.assertEqual(frozen['repetitions'], 5)
        self.assertEqual(frozen['cases'][0]['parameters']['final_time'], 0.004)
        self.assertEqual(frozen['correctness_regex'], suite['correctness_regex'])
        self.assertEqual(plan['builds'], self.plan['builds'])
        path.write_text('changed outside the frozen run')
        study.verify(self.new)

    def test_cpu_hdg_selection_rejects_before_scheduler_or_destination(self):
        suite = study.read_json(Path(study.__file__).parent / 'suites/pace-full-backends.json')
        path = self.base / 'backend-suite.json'
        study.write_json(path, suite)
        with self.assertRaisesRegex(ValueError, 'separately admitted CPU-profile'):
            self.retry(suite=str(path), cases='hdg-cpu-p3')
        self.assertFalse(self.new.exists())
        self.assertEqual(self.calls, [])

    def test_changed_correctness_malformed_suite_and_unknown_selection_reject_before_creation(self):
        suite = study.read_json(self.old / 'suite.json')
        suite['correctness_regex'] = '^new-test$'
        path = self.base / 'new-suite.json'
        study.write_json(path, suite)
        with self.assertRaisesRegex(ValueError, 'cannot change correctness_regex'):
            self.retry(suite=str(path))
        self.assertFalse(self.new.exists())
        self.assertEqual(self.calls, [])
        path.write_text('{}')
        with self.assertRaises(ValueError):
            self.retry(suite=str(path))
        for cases in ('', 'unknown', 'dg-p1,', 'dg-p1,dg-p1'):
            with self.assertRaisesRegex(ValueError, 'unique known case IDs'):
                self.retry(cases=cases)
        self.assertFalse(self.new.exists())
        self.assertEqual(self.calls, [])

    def test_newly_selected_missing_ambiguous_and_symlinked_executable_reject_without_build(self):
        benchmark = self.app / study.study_workloads.BINARY_NAMES['dgsem']
        with self.assertRaisesRegex(RuntimeError, 'missing or ambiguous completed executable'):
            self.retry(cases='dgsem-n4')
        benchmark.write_text('benchmark\n')
        benchmark.chmod(0o755)
        duplicate = self.app / 'duplicate' / benchmark.name
        duplicate.parent.mkdir()
        duplicate.write_text('duplicate benchmark\n')
        duplicate.chmod(0o755)
        with self.assertRaisesRegex(RuntimeError, 'missing or ambiguous completed executable'):
            self.retry(cases='dgsem-n4')
        duplicate.unlink()
        benchmark.unlink()
        benchmark.symlink_to(self.binary)
        with self.assertRaisesRegex(ValueError, 'symlinked path'):
            self.retry(cases='dgsem-n4')
        self.assertFalse(self.new.exists())
        self.assertTrue(all(command[0] in ('squeue', 'sacct') for command in self.calls))

    def test_limits_inherit_or_explicitly_override_frozen_plan(self):
        self.plan['wall_time_limits'] = {'timings': '00:20:00', 'profiles': '00:40:00'}
        self.save_plan()
        self.assertEqual(self.retry(profiles_time='00:30:00')['wall_time_limits'],
                         {'timings': '00:20:00', 'profiles': '00:30:00'})

    def test_new_build_manifest_rejects_later_artifact_change(self):
        self.retry()
        self.binary.write_text('changed since retry freeze')
        with self.assertRaisesRegex(RuntimeError, 'completed build artifact changed'): study.verify(self.new)

    def test_old_admission_extra_artifacts_are_verified_and_preserved_in_origin_manifest(self):
        extra = self.cache / 'dependency.so'
        extra.write_text('admitted dependency bytes')
        study.write_json(self.old / 'admission.json', {'plan_sha256': self.plan['plan_sha256'],
                                                     'artifact_sha256': {str(extra): study.sha(extra)}})
        self.assertIn(str(extra), self.retry()['retry_origin']['build_artifact_sha256'])

    def test_changed_source_executor_suite_and_old_admission_fail_before_destination(self):
        paths = (self.source / 'code.cpp', self.old / 'executor' / 'pace_runtime.sh', self.old / 'suite.json')
        for path in paths:
            original = path.read_bytes()
            path.write_bytes(original + b'changed')
            with self.assertRaises((ValueError, RuntimeError)): self.retry()
            self.assertFalse(self.new.exists())
            path.write_bytes(original)
        study.write_json(self.old / 'admission.json', {'plan_sha256': self.plan['plan_sha256'],
                                                     'artifact_sha256': {str(self.binary): '0' * 64}})
        with self.assertRaisesRegex(RuntimeError, 'admitted build'): self.retry()
        self.assertFalse(self.new.exists())

    def test_active_unknown_and_scheduler_error_fail_closed(self):
        for stdout, code in (('34|PENDING\n', 0), ('', 1)):
            self.queue_stdout, self.queue_code = stdout, code
            with self.assertRaisesRegex(RuntimeError, 'inactive'): self.retry()
            self.assertFalse(self.new.exists())
        self.queue_stdout, self.queue_code = '', 0
        self.outcomes['34'] = ('RUNNING', '0:0')
        with self.assertRaisesRegex(RuntimeError, 'not terminal'): self.retry()
        self.assertFalse(self.new.exists())

    def test_unrelated_active_jobs_do_not_block_inactive_accounting_only_builds(self):
        self.queue_stdout = '999|RUNNING\n'
        self.retry()
        command = next(c for c in self.calls if c[0] == 'squeue')
        self.assertIn('-u', command)
        self.assertNotIn('-j', command)

    def test_failed_build_or_correctness_and_mismatched_receipt_reject(self):
        for job in ('11', '12'):
            previous = self.outcomes[job]
            self.outcomes[job] = ('FAILED', '1:0')
            with self.assertRaisesRegex(RuntimeError, 'successful|successfully'): self.retry()
            self.assertFalse(self.new.exists())
            self.outcomes[job] = previous
        self.state['builds']['baseline']['configuration_key'] = 'wrong'
        self.save_state()
        with self.assertRaisesRegex(RuntimeError, 'matching successful'): self.retry()

    def test_missing_ambiguous_or_symlinked_build_artifacts_reject(self):
        library = self.install / 'libmfem.a'
        library.unlink()
        with self.assertRaisesRegex(RuntimeError, 'MFEM library'): self.retry()
        library.write_text('restored')
        nested = self.app / 'duplicate'
        nested.mkdir()
        duplicate = nested / self.binary.name
        duplicate.write_bytes(self.binary.read_bytes())
        duplicate.chmod(0o755)
        with self.assertRaisesRegex(RuntimeError, 'ambiguous completed executable'): self.retry()
        duplicate.unlink()
        library.unlink()
        library.symlink_to(self.mfem / 'CMakeCache.txt')
        with self.assertRaisesRegex(ValueError, 'symlinked path'): self.retry()
        self.assertFalse(self.new.exists())

    def test_destination_inside_old_source_cache_and_existing_directory_reject(self):
        for root in (self.old / 'child', self.source / 'child', self.cache / 'child', self.old):
            with self.assertRaises(ValueError): self.retry(run_dir=str(root))
        self.new.mkdir()
        with self.assertRaisesRegex(ValueError, 'must be new'): self.retry()
        self.assertFalse(any(command[0] == 'sbatch' for command in self.calls))

    def test_submit_only_fresh_timing_then_dependent_profiles_and_keep_partial_receipts(self):
        self.profile_code = 1
        with self.assertRaisesRegex(RuntimeError, 'profiles submission failed'): self.retry(prepare=False)
        state = study.read_json(self.new / 'state.json')
        self.assertEqual([job['stage'] for job in state['jobs']], ['timings', 'profiles'])
        self.assertEqual(state['jobs'][0]['job_id'], '101')
        self.assertNotIn('job_id', state['jobs'][1])
        self.assertIn('--dependency=afterok:101', state['jobs'][1]['argv'])
        self.assertEqual(state['attempts'], [])
        with self.assertRaises(ValueError): self.retry(prepare=False)
        self.assertEqual(sum(c[0] == 'sbatch' for c in self.calls), 2)

    def test_retry_cli_overrides_and_invalid_limits(self):
        argv = ['retry', '--from-run', str(self.old), '--run-dir', str(self.new), '--prepare-only']
        with patch.object(study, 'retry') as retry:
            study.main(argv + ['--timings-time', '00:15:00', '--profiles-time', '00:30:00',
                               '--suite', 'suite.json', '--cases', 'dgsem-n4,dgsem-n8'])
            args = retry.call_args.args[0]
            self.assertEqual((args.timings_time, args.profiles_time, args.prepare_only),
                             ('00:15:00', '00:30:00', True))
            self.assertEqual((args.suite, args.cases), ('suite.json', 'dgsem-n4,dgsem-n8'))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                study.main(argv + ['--timings-time', '00:04:00'])


if __name__ == '__main__':
    unittest.main()
