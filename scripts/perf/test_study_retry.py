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
from types import SimpleNamespace
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
        if argv[0] == 'scontrol':
            return subprocess.CompletedProcess(argv, 0, self.interactive_info, '')
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

    def test_legacy_cuda_executor_set_remains_eligible_for_fresh_retry(self):
        for name in set(study.EXECUTOR_FILES) - set(study.LEGACY_EXECUTOR_FILES):
            self.plan['executor_sha256'].pop(name)
            (self.old / 'executor' / name).unlink()
        self.save_plan()
        plan = self.retry()
        self.assertEqual(set(plan['executor_sha256']), set(study.EXECUTOR_FILES))
        self.assertEqual(plan['retry_origin']['executor_sha256'], self.plan['executor_sha256'])
        self.assertNotIn('cpu_builds', plan)

    def test_pre_import_cpu_capable_executor_remains_eligible_for_retry(self):
        for name in set(study.EXECUTOR_FILES) - set(study.PRE_IMPORT_EXECUTOR_FILES):
            self.plan['executor_sha256'].pop(name)
            (self.old / 'executor' / name).unlink()
        self.save_plan()
        plan = self.retry()
        self.assertEqual(set(plan['executor_sha256']), set(study.EXECUTOR_FILES))
        self.assertEqual(plan['retry_origin']['executor_sha256'], self.plan['executor_sha256'])

    def test_imported_canonical_test_receipts_retry_without_validation_job_substitution(self):
        record = self.old / 'actual-test-result.json'
        record.write_text('sealed original canonical outcome')
        self.plan['import_origin'] = {'artifact_sha256': {str(record): study.sha(record)}}
        for member in study.MEMBERS:
            self.plan['builds'][member].update(origin=study.study_import.ORIGIN,
                                              build_job_id='11', test_result=str(record))
            receipt = self.state['builds'][member]
            receipt.pop('validation_job_id', None)
            receipt.update(origin=study.study_import.ORIGIN, test_result=str(record))
        self.save_plan()
        self.save_state()
        with patch.object(study.study_import, 'completed_records') as imported:
            plan = self.retry()
        self.assertEqual(imported.call_count, 2)
        self.assertEqual(plan['import_origin'], self.plan['import_origin'])
        self.assertNotIn('validation_job_id', study.read_json(self.new / 'state.json')['builds']['baseline'])

    def test_retry_retains_cpu_build_receipts_and_new_cpu_binary_hash_without_recompiling(self):
        cpu_app = self.cache / 'cpu-app'
        cpu_app.mkdir()
        (cpu_app / 'CMakeCache.txt').write_text('completed CPU build metadata\n')
        cpu_binary = cpu_app / self.binary.name
        cpu_binary.write_text('CPU executable bytes, never executed\n')
        cpu_binary.chmod(0o755)
        cpu_build = {**self.plan['builds']['baseline'], 'profile': 'cpu', 'app_build': str(cpu_app),
                     'configuration_key': 'cpu-key', 'argv': ['canonical', '--cpu-only']}
        cpu_receipt = {**self.state['builds']['baseline'], 'configuration_key': 'cpu-key',
                       'argv': cpu_build['argv'], 'build_job_id': '21', 'validation_job_id': '22',
                       'build_job_ids': ['21'], 'validation_job_ids': ['22']}
        self.plan['cpu_builds'] = {member: dict(cpu_build) for member in study.MEMBERS}
        self.state['cpu_builds'] = {member: dict(cpu_receipt) for member in study.MEMBERS}
        self.outcomes.update({'21': ('COMPLETED', '0:0'), '22': ('COMPLETED', '0:0')})
        suite = study.read_json(self.old / 'suite.json')
        # A CPU DG case can share a separately validated CPU artifact, even if
        # this fixture does not itself provide a native CPU admission receipt.
        case = next(case for case in suite['cases'] if case['enabled'])
        case['parameters']['device'] = 'cpu'
        study.write_json(self.old / 'suite.json', suite)
        self.plan['suite_sha256'] = study.sha(self.old / 'suite.json')
        self.save_plan()
        self.save_state()
        plan = self.retry()
        state = study.read_json(self.new / 'state.json')
        self.assertEqual(plan['cpu_builds'], self.plan['cpu_builds'])
        self.assertEqual(state['cpu_builds'], self.state['cpu_builds'])
        self.assertEqual(plan['retry_origin']['build_artifact_sha256'][str(cpu_binary)], study.sha(cpu_binary))
        self.assertTrue(all(command[0] in ('squeue', 'sacct') for command in self.calls))
        self.assertFalse((self.new / 'admission.json').exists())

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
            study.main(argv + ['--profiles-only'])
            self.assertTrue(retry.call_args.args[0].profiles_only)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                study.main(argv + ['--timings-time', '00:04:00'])

    def prepare_profile_timing(self):
        gpu = {'uuid': 'GPU-00112233-4455-6677-8899-aabbccddeeff',
               'name': 'NVIDIA A100-SXM4-80GB', 'compute_capability': 80}
        attempt = self.state['attempts'][0]
        summary = Path(attempt['summary'])
        study.write_json(summary, {'status': 'timings_complete', 'gpu': gpu,
            'cases': {case['id']: {'statistics': study.sample_statistics([1, 2, 3], [1, 2, 3]),
                                  'observations': []} for case in study.selected_cases(
                                      study.study_workloads.load_suite(self.old / 'suite.json'))}})
        other = summary.parent / 'field.vtu'
        other.write_text('existing field bytes, not a trace\n')
        attempt.update(summary_sha256=study.sha(summary), gpu=gpu,
            artifact_sha256={str(path.relative_to(self.old)): study.sha(path) for path in (summary, other)})
        self.save_state()
        return gpu, summary, other

    def prepare_unstarted_profiles(self):
        gpu, summary, _ = self.prepare_profile_timing()
        timing = study.read_json(summary)
        for item in timing['cases'].values():
            item['observations'] = [{'signature': {'work': 10}}]
        study.write_json(summary, timing)
        attempt = self.state['attempts'][0]
        attempt['summary_sha256'] = study.sha(summary)
        attempt['artifact_sha256'][str(summary.relative_to(self.old))] = study.sha(summary)
        self.save_state()
        case = study.selected_cases(study.study_workloads.load_suite(self.old / 'suite.json'))[0]
        return gpu, case, summary

    def prepare_mixed_unstarted_profiles(self):
        suite = study.read_json(self.old / 'suite.json')
        cases = [case for case in suite['cases'] if case['adapter'] == 'dg' and
                 case.get('parameters', {}).get('device') != 'cpu'][:2]
        self.assertEqual(len(cases), 2)
        names = {case['id'] for case in cases}
        for case in suite['cases']:
            case['enabled'] = case['id'] in names
        study.write_json(self.old / 'suite.json', suite)
        self.plan['suite_sha256'] = study.sha(self.old / 'suite.json')
        self.save_plan()
        gpu, _, summary = self.prepare_unstarted_profiles()
        frozen = study.study_workloads.load_suite(self.old / 'suite.json')
        return gpu, frozen, study.selected_cases(frozen), summary

    def test_profile_subset_preserves_full_suite_timing_bytes_and_unselected_coverage(self):
        gpu, suite, cases, summary = self.prepare_mixed_unstarted_profiles()
        before = self.snapshot()
        plan = self.retry(profiles_only=True, continue_profiles=True, profile_cases=cases[1]['id'])
        self.assertEqual(before, self.snapshot())
        self.assertEqual(plan['profile_case_ids'], [cases[1]['id']])
        self.assertEqual((self.new / 'suite.json').read_bytes(), (self.old / 'suite.json').read_bytes())
        shadow = study.completed(study.read_json(self.new / 'state.json'), 'timings')
        self.assertEqual(Path(shadow['summary']).read_bytes(), summary.read_bytes())
        result, calls, _, output = self.run_continuation(plan, cases[1], gpu, suite=suite)
        self.assertEqual(set(result['cases']), {cases[1]['id']})
        self.assertEqual(result['profile_case_ids'], [cases[1]['id']])
        self.assertEqual(len(calls), 2)
        self.assertIn('selected profile_case_ids only', result['scope'])
        study.write_json(output / 'summary.json', result)
        state = study.read_json(self.new / 'state.json')
        state['attempts'].append({'id': 'profiles-fixture', 'stage': 'profiles', 'status': 'analysis_complete',
            'summary': str(output / 'summary.json'), 'summary_sha256': study.sha(output / 'summary.json'),
            'artifact_sha256': {'attempts/profiles-new/summary.json': study.sha(output / 'summary.json')}})
        study.write_json(self.new / 'state.json', state)
        report = study.report(self.new)
        coverage = {row['case_id']: row for row in report['coverage']['cases']}
        self.assertEqual(coverage[cases[0]['id']]['profile_status'], 'not_requested')
        self.assertEqual(coverage[cases[1]['id']]['profile_status'], 'complete')
        self.assertTrue(all(row['timing_status'] == 'measured' for row in coverage.values()))
        self.assertFalse(any(command[0] == 'sbatch' for command in self.calls))

    def test_default_profile_selection_retains_full_enabled_suite(self):
        gpu, suite, cases, _ = self.prepare_mixed_unstarted_profiles()
        plan = self.retry(profiles_only=True, continue_profiles=True)
        self.assertNotIn('profile_case_ids', plan)
        result, calls, _, _ = self.run_continuation(plan, cases[0], gpu, suite=suite)
        self.assertEqual(set(result['cases']), {case['id'] for case in cases})
        self.assertEqual(len(calls), 4)

    def test_profile_subset_invalid_ids_modes_and_cpu_selection_reject_before_preparation(self):
        _, _, cases, _ = self.prepare_mixed_unstarted_profiles()
        before = self.snapshot()
        for names in ('', 'unknown', cases[0]['id'] + ',', cases[0]['id'] + ',' + cases[0]['id']):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, 'unique enabled GPU'):
                self.retry(profiles_only=True, continue_profiles=True, profile_cases=names)
            self.assertFalse(self.new.exists())
        for options in ({}, {'profiles_only': True}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'requires --profiles-only'):
                self.retry(profile_cases=cases[0]['id'], **options)
        with self.assertRaisesRegex(ValueError, 'enabled GPU'):
            study.selected_profile_cases({'profile_case_ids': ['cpu']}, {'cases': [
                {'id': 'cpu', 'enabled': True, 'parameters': {'device': 'cpu'}}]})
        self.assertEqual(before, self.snapshot())

    def test_selected_profile_keeps_original_numerical_gate(self):
        gpu, suite, cases, _ = self.prepare_mixed_unstarted_profiles()
        plan = self.retry(profiles_only=True, continue_profiles=True, profile_cases=cases[1]['id'])
        result, _, _, _ = self.run_continuation(plan, cases[1], gpu, suite=suite, wrong_signature=True)
        self.assertEqual(set(result['cases']), {cases[1]['id']})
        self.assertEqual(result['cases'][cases[1]['id']]['baseline']['application_validation']['status'], 'failed')

    def test_missing_only_continuation_locks_subset_and_reuses_successful_tools(self):
        gpu, suite, cases, _ = self.prepare_mixed_unstarted_profiles()
        first = self.retry(profiles_only=True, continue_profiles=True, profile_cases=cases[1]['id'])
        _, _, _, output = self.run_continuation(first, cases[1], gpu, suite=suite, fail_ncu=True)
        state = study.read_json(self.new / 'state.json')
        state['attempts'].append({'id': 'profiles-new', 'stage': 'profiles', 'status': 'profiles_partial',
                                 'job_id': '35', 'directory': str(output), 'gpu': gpu})
        study.write_json(self.new / 'state.json', state)
        self.old, self.new = self.new, self.base / 'next'
        self.plan, self.state = first, state
        self.outcomes['35'] = ('TIMEOUT', '0:0')
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'preserve its original case subset'):
            self.retry(profiles_only=True, continue_profiles=True, profile_cases=cases[0]['id'])
        self.assertEqual(before, self.snapshot())
        plan = self.retry(profiles_only=True, continue_profiles=True)
        self.assertEqual(plan['profile_case_ids'], [cases[1]['id']])
        self.assertEqual(set(plan['profile_continuation']['captures']), {cases[1]['id']})
        result, calls, _, _ = self.run_continuation(plan, cases[1], gpu, suite=suite)
        self.assertEqual(calls, [('baseline', ['ncu']), ('candidate', ['nsys', 'ncu'])])
        self.assertEqual(result['status'], 'analysis_complete')

    def test_existing_full_profile_continuation_cannot_shrink_selection(self):
        self.prepare_continuation(names=('nsys',))
        suite = study.study_workloads.load_suite(self.old / 'suite.json')
        case = study.selected_cases(suite)[0]
        # Historical default is its complete enabled GPU selection, not any new subset.
        other = {**case, 'id': 'other-profile'}
        suite['cases'].append(other)
        with self.assertRaisesRegex(ValueError, 'preserve its original case subset'):
            study.retry_profile_selection(self.plan, self.state, suite, case['id'])

    def test_new_timing_cohort_does_not_inherit_a_profile_only_subset(self):
        _, _, cases, _ = self.prepare_mixed_unstarted_profiles()
        self.plan['profile_case_ids'] = [cases[1]['id']]
        self.save_plan()
        plan = self.retry()
        self.assertNotIn('profile_case_ids', plan)
        self.assertEqual((self.new / 'suite.json').read_bytes(), (self.old / 'suite.json').read_bytes())

    def test_public_prepare_only_shape_binds_original_admission_for_pool(self):
        import delta_profile_pool as pool
        original = self.old
        self.old = self.base / 'studies' / 'old'
        self.old.parent.mkdir()
        original.rename(self.old)
        self.new = self.base / 'studies' / 'profiles-new'
        attempt = self.state['attempts'][0]
        attempt['directory'] = str(self.old / 'attempts/old-measurement')
        attempt['summary'] = str(Path(attempt['directory']) / 'summary.json')
        self.plan.update(run_dir=str(self.old), site='delta', site_root=str(self.base / 'dependencies'),
                         spack_root=str(self.base / 'spack'))
        self.save_plan()
        self.save_state()
        _, case, _ = self.prepare_unstarted_profiles()
        study.write_json(self.old / 'admission.json', {'plan_sha256': self.plan['plan_sha256'], 'artifact_sha256': {}})
        plan = self.retry(profiles_only=True, continue_profiles=True, profile_cases=case['id'])
        self.assertFalse((self.new / 'admission.json').exists())
        task = {'id': 'prepared', 'run_dir': str(self.new), 'plan_sha256': plan['plan_sha256'],
                'executor_sha256': {name: plan['executor_sha256'][name]
                                   for name in ('gpu_study.py', 'delta_runtime.sh')},
                'origin_run_dir': str(self.old), 'origin_admission_sha256': study.sha(self.old / 'admission.json')}
        self.assertEqual(pool.check_task(task), 'ready')
        with self.assertRaisesRegex(RuntimeError, 'admission seal changed'):
            pool.check_task({**task, 'origin_admission_sha256': '0' * 64})
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            pool.check_task({**task, 'origin_run_dir': str(self.new)})
        self.assertFalse((self.new / 'admission.json').exists())

    def test_zero_attempt_continuation_preserves_timings_and_collects_only_new_diagnostics(self):
        gpu, case, summary = self.prepare_unstarted_profiles()
        before = self.snapshot()
        plan = self.retry(profiles_only=True, continue_profiles=True, prepare=False)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(plan['profile_continuation']['origin_kind'], 'no-profile-attempt')
        self.assertEqual(plan['profile_continuation']['captures'], {})
        self.assertEqual(Path(plan['timing_reuse']['summary']).read_bytes(), summary.read_bytes())
        self.assertFalse((self.new / 'admission.json').exists())
        self.assertEqual([j['stage'] for j in study.read_json(self.new / 'state.json')['jobs']], ['profiles'])
        different = {**gpu, 'uuid': 'GPU-11223344-5566-7788-99aa-bbccddeeff00'}
        result, calls, _, _ = self.run_continuation(plan, case, different)
        self.assertEqual(calls, [('baseline', ['nsys', 'ncu']), ('candidate', ['nsys', 'ncu'])])
        self.assertEqual(result['status'], 'analysis_complete')
        self.assertIn('no matched profiler timing', result['scope'])
        for member in study.MEMBERS:
            self.assertEqual(result['cases'][case['id']][member]['gpu_uuid'], different['uuid'])
        self.assertEqual(plan['timing_reuse']['gpu'], gpu)

    def test_zero_attempt_continuation_rejects_unrecorded_profile_tree(self):
        self.prepare_unstarted_profiles()
        orphan = self.old / 'attempts/profiles-unrecorded'
        orphan.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'unrecorded profile directory'):
            self.retry(profiles_only=True, continue_profiles=True)
        self.assertFalse(self.new.exists())

    def test_zero_attempt_continuation_rejects_active_recorded_jobs(self):
        self.prepare_unstarted_profiles()
        self.queue_stdout = '34|RUNNING\n'
        with self.assertRaisesRegex(RuntimeError, 'inactive'):
            self.retry(profiles_only=True, continue_profiles=True)
        self.assertFalse(self.new.exists())

    def test_owned_cpu_compute_can_prepare_only_profile_continuation_without_submission(self):
        self.prepare_unstarted_profiles()
        before = self.snapshot()
        proof = {'profile': 'cpu', 'job_id': '55', 'hostname': 'owned-compute', 'uid': os.getuid(),
                 'scope': 'live owned free CPU allocation'}
        with patch.dict(os.environ, {'SLURM_JOB_ID': '55'}), \
             patch.object(study, 'cpu_allocation', return_value=proof) as allocation:
            plan = self.retry(profiles_only=True, continue_profiles=True)
        allocation.assert_called_once_with()
        self.assertEqual(plan['retry_origin']['preparation_allocation'], proof)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(study.read_json(self.new / 'state.json')['jobs'], [])
        self.assertFalse(any(command[0] == 'sbatch' for command in self.calls))

    def test_compute_retry_rejects_other_modes_before_any_evidence_read(self):
        for options in ({}, {'profiles_only': True}, {'continue_profiles': True},
                        {'profiles_only': True, 'continue_profiles': True, 'prepare': False}):
            with self.subTest(options=options), patch.dict(os.environ, {'SLURM_JOB_ID': '55'}), \
                 patch.object(study, 'verify') as verify, patch.object(study, 'cpu_allocation') as allocation:
                with self.assertRaisesRegex(RuntimeError, 'compute retry permits only'):
                    self.retry(**options)
                verify.assert_not_called()
                allocation.assert_not_called()
            self.assertFalse(self.new.exists())

    def test_compute_prepare_requires_actual_owned_allocation_before_evidence_read(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '55'}), \
             patch.object(study, 'cpu_allocation', side_effect=RuntimeError('live scheduler resources differ')), \
             patch.object(study, 'verify') as verify:
            with self.assertRaisesRegex(RuntimeError, 'live scheduler resources differ'):
                self.retry(profiles_only=True, continue_profiles=True)
            verify.assert_not_called()
        self.assertFalse(self.new.exists())

    def imported_profile_fixture(self, *, running=False):
        if not any(attempt.get('stage') == 'profiles' for attempt in self.state['attempts']):
            _, _, checkpoint = self.prepare_continuation(names=('nsys',))
        else:
            checkpoint = Path(self.state['attempts'][-1]['directory']) / 'partial.json'
        self.plan['import_origin'] = {'artifact_sha256': {}, 'scope': 'saved actual imports'}
        for member in study.MEMBERS:
            self.plan['builds'][member] = {**self.plan['builds'][member],
                'origin': study.study_import.ORIGIN, 'test_result': '/actual/sealed/result.json',
                'reuse_validated_artifacts': False}
            receipt = {**self.state['builds'][member], 'origin': study.study_import.ORIGIN,
                       'test_result': '/actual/sealed/result.json', 'reuse_validated_artifacts': False}
            receipt.pop('validation_job_id', None)
            self.state['builds'][member] = receipt
        if running:
            self.state['attempts'][-1]['status'] = 'running'
        self.save_plan()
        self.save_state()
        def prove(build, receipt, *, full=False):
            self.assertTrue(full)
            self.assertTrue(build['reuse_validated_artifacts'])
            self.assertTrue(receipt['reuse_validated_artifacts'])
            self.assertEqual(build['test_result'], '/actual/sealed/result.json')
            self.assertEqual(receipt['test_result'], build['test_result'])
            return {'build_accounting': {'query_performed': False},
                    'test_allocation_accounting': {'query_performed': False},
                    'test_execution_scope': 'actual Passed seals revalidated'}
        return checkpoint, prove

    def test_opt_in_profile_reuse_queries_only_measurement_jobs_and_keeps_original_bytes(self):
        _, prove = self.imported_profile_fixture()
        before = self.snapshot()
        with patch.object(study.study_import, 'completed_records', side_effect=prove) as admission:
            plan = self.retry(profiles_only=True, continue_profiles=True, reuse_validated_artifacts=True)
        self.assertEqual(admission.call_count, 2)
        self.assertEqual(before, self.snapshot())
        queried = {job for argv in self.calls if argv[0] == 'sacct'
                   for job in argv[argv.index('-j') + 1].split(',')}
        self.assertEqual(queried, {'33', '34'})
        self.assertTrue(plan['import_origin']['reuse_validated_artifacts'])
        self.assertTrue(all(build['reuse_validated_artifacts'] for build in plan['builds'].values()))
        self.assertTrue(all(receipt['reuse_validated_artifacts'] for receipt in
                            study.read_json(self.new / 'state.json')['builds'].values()))
        self.assertEqual(set(plan['profile_continuation']['captures']['dg-p1']['baseline']), {'nsys'})
        self.assertFalse(any(argv[0] == 'sbatch' for argv in self.calls))

    def test_terminal_profile_reconciliation_is_explicit_and_read_only(self):
        checkpoint, prove = self.imported_profile_fixture(running=True)
        before = self.snapshot()
        with patch.object(study.study_import, 'completed_records', side_effect=prove):
            with self.assertRaisesRegex(RuntimeError, 'explicit terminal profile reconciliation'):
                self.retry(profiles_only=True, continue_profiles=True, reuse_validated_artifacts=True)
            plan = self.retry(profiles_only=True, continue_profiles=True, reuse_validated_artifacts=True,
                              reconcile_terminal_profile_attempt=True)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(study.read_json(self.old / 'state.json')['attempts'][-1]['status'], 'running')
        self.assertEqual(plan['profile_continuation']['checkpoint_sha256'], study.sha(checkpoint))
        proof = plan['retry_origin']['reconciled_terminal_profile_attempts']
        self.assertEqual(proof[0]['job_id'], '34')
        self.assertEqual(proof[0]['original_status'], 'running')
        self.assertEqual(proof[0]['scheduler_outcome'], ['CANCELLED', '0:0'])
        self.assertFalse(any(a.get('stage') == 'profiles' for a in study.read_json(self.new / 'state.json')['attempts']))

    def test_terminal_reconciliation_rejects_active_unknown_and_running_timing_attempts(self):
        for failure in ('active', 'unknown', 'timings'):
            with self.subTest(failure=failure):
                _, prove = self.imported_profile_fixture(running=True)
                self.queue_stdout = '34|RUNNING\n' if failure == 'active' else ''
                self.outcomes['34'] = ('UNKNOWN', '') if failure == 'unknown' else ('CANCELLED', '0:0')
                if failure == 'timings':
                    self.state['attempts'][0]['status'] = 'running'
                    self.save_state()
                before = self.snapshot()
                with patch.object(study.study_import, 'completed_records', side_effect=prove):
                    with self.assertRaises(RuntimeError):
                        self.retry(profiles_only=True, continue_profiles=True, reuse_validated_artifacts=True,
                                   reconcile_terminal_profile_attempt=True)
                self.assertEqual(before, self.snapshot())
                self.assertFalse(self.new.exists())

    def test_artifact_reuse_requires_actual_imports_and_does_not_swallow_receipt_failure(self):
        self.prepare_continuation(names=('nsys',))
        with self.assertRaisesRegex(RuntimeError, 'genuine imported canonical Passed'):
            self.retry(profiles_only=True, continue_profiles=True, reuse_validated_artifacts=True)
        _, _ = self.imported_profile_fixture()
        with patch.object(study.study_import, 'completed_records', side_effect=RuntimeError('sealed artifact changed')):
            with self.assertRaisesRegex(RuntimeError, 'sealed artifact changed'):
                self.retry(profiles_only=True, continue_profiles=True, reuse_validated_artifacts=True)
        self.assertFalse(self.new.exists())

    def test_reuse_and_reconciliation_flags_reject_other_modes_before_reads(self):
        for options in ({'reuse_validated_artifacts': True},
                        {'profiles_only': True, 'reuse_validated_artifacts': True},
                        {'profiles_only': True, 'continue_profiles': True, 'prepare': False,
                         'reuse_validated_artifacts': True},
                        {'profiles_only': True, 'continue_profiles': True,
                         'reconcile_terminal_profile_attempt': True}):
            with self.subTest(options=options), patch.object(study, 'verify') as verify:
                with self.assertRaises(ValueError):
                    self.retry(**options)
                verify.assert_not_called()

    def test_owned_a100_can_prepare_missing_profile_continuation_without_submitting(self):
        _, prove = self.imported_profile_fixture()
        before = self.snapshot()
        proof = {'profile': 'cuda-sm80', 'job_id': '55', 'hostname': 'owned-compute',
                 'gpu': {'uuid': 'GPU-fixture'}, 'scope': 'live owned A100 metadata preparation'}
        with patch.dict(os.environ, {'SLURM_JOB_ID': '55', 'SLURM_JOB_PARTITION': 'gpu-a100'}), \
             patch.object(study, 'gpu_preparation_allocation', return_value=proof) as allocation, \
             patch.object(study.study_import, 'completed_records', side_effect=prove):
            plan = self.retry(profiles_only=True, continue_profiles=True, reuse_validated_artifacts=True)
        allocation.assert_called_once_with()
        self.assertEqual(plan['retry_origin']['preparation_allocation'], proof)
        self.assertEqual(before, self.snapshot())
        self.assertFalse(any(argv[0] == 'sbatch' for argv in self.calls))

    def test_gpu_preparation_proves_live_scheduler_resources_node_gpu_and_mpi_none(self):
        fields = (f'JobId=55 JobState=RUNNING BatchFlag=0 UserId={study.getpass.getuser()}({os.getuid()}) '
                  'Account=gts-sbryngelson3 QOS=embers Partition=gpu-a100 NumNodes=1 NumCPUs=8 NumTasks=1 '
                  'AllocTRES=cpu=8,mem=48G,gres/gpu=1,gres/gpu:a100=1 NodeList=owned-compute '
                  'TimeLimit=01:00:00 EndTime=2030-01-01T01:00:00')
        gpu = {'uuid': 'GPU-fixture', 'name': 'NVIDIA A100', 'compute_capability': 80}
        def query(argv, **kwargs):
            self.assertEqual(argv[0], 'scontrol')
            self.assertEqual(kwargs['env']['TZ'], 'UTC')
            return subprocess.CompletedProcess(argv, 0, 'owned-compute\n' if 'hostnames' in argv else fields, '')
        with patch.dict(os.environ, {'SLURM_JOB_ID': '55'}), \
             patch.object(study, 'allocation', return_value=gpu), \
             patch.object(study.socket, 'getfqdn', return_value='owned-compute.example'), \
             patch.object(study.subprocess, 'run', side_effect=query) as queried:
            admitted = study.gpu_preparation_allocation()
            self.assertEqual(admitted['gpu'], gpu)
            self.assertEqual(admitted['scheduler']['JobId'], '55')
            self.assertEqual(queried.call_count, 2)
            for original, wrong in [('mem=48G', 'mem=96G'), ('QOS=embers', 'QOS=paid'),
                                    ('BatchFlag=0', 'BatchFlag=1'), ('gres/gpu=1', 'gres/gpu=2'),
                                    ('NumCPUs=8', 'NumCPUs=16')]:
                valid = fields
                fields = fields.replace(original, wrong)
                with self.subTest(wrong=wrong), self.assertRaisesRegex(RuntimeError, 'identity/resources differ'):
                    study.gpu_preparation_allocation()
                fields = valid
            with patch.dict(os.environ, {'PMIX_NAMESPACE': 'inherited'}):
                with self.assertRaisesRegex(RuntimeError, 'MPI-none'):
                    study.gpu_preparation_allocation()
            with patch.object(study.socket, 'getfqdn', return_value='different-node.example'):
                with self.assertRaisesRegex(RuntimeError, 'actual allocated node'):
                    study.gpu_preparation_allocation()

    def test_zero_attempt_continuation_seals_old_state_before_new_capture(self):
        self.prepare_unstarted_profiles()
        plan = self.retry(profiles_only=True, continue_profiles=True)
        self.assertIn(str(self.old / 'state.json'), plan['profile_continuation']['artifact_sha256'])
        self.state['unexpected_change'] = True
        self.save_state()
        with self.assertRaisesRegex(RuntimeError, 'retained profile metadata changed'):
            study.verify(self.new)

    def test_zero_attempt_timeout_override_changes_only_new_profile_recipes(self):
        raw = study.read_json(self.old / 'suite.json')
        for case in raw['cases']:
            case['profile']['timeout'] = 240
        study.write_json(self.old / 'suite.json', raw)
        self.plan['suite_sha256'] = study.sha(self.old / 'suite.json')
        self.save_plan()
        self.prepare_unstarted_profiles()
        before = self.snapshot()
        plan = self.retry(profiles_only=True, continue_profiles=True, profile_timeout_seconds=3600,
                          profiles_budget_seconds=3600)
        self.assertEqual(before, self.snapshot())
        original = study.read_json(self.old / 'suite.json')
        fresh = study.read_json(self.new / 'suite.json')
        for case in fresh['cases']:
            old = next(item for item in original['cases'] if item['id'] == case['id'])
            if case.get('enabled', True) and case.get('parameters', {}).get('device') != 'cpu':
                self.assertEqual(case['profile']['timeout'], 3600)
                restored = {**case, 'profile': dict(old['profile'])}
                self.assertEqual(restored, old)
            else:
                self.assertEqual(case, old)
        self.assertEqual(plan['timing_reuse']['suite_sha256'], study.sha(self.old / 'suite.json'))
        self.assertNotEqual(plan['suite_sha256'], self.plan['suite_sha256'])
        self.assertEqual(fresh['profiles_budget_seconds'], 3600)
        self.assertEqual({**fresh, 'profiles_budget_seconds': original['profiles_budget_seconds'],
                          'cases': original['cases']}, original)
        self.assertEqual(plan['timing_reuse']['profiles_budget_override'],
                         {'original': original['profiles_budget_seconds'], 'replacement': 3600})

    def test_zero_attempt_budget_only_override_preserves_recipes_and_timing_policy(self):
        self.prepare_unstarted_profiles()
        before = self.snapshot()
        self.retry(profiles_only=True, continue_profiles=True, profiles_budget_seconds=7200)
        self.assertEqual(before, self.snapshot())
        old = study.read_json(self.old / 'suite.json')
        new = study.read_json(self.new / 'suite.json')
        self.assertEqual(new['profiles_budget_seconds'], 7200)
        self.assertEqual({**new, 'profiles_budget_seconds': old['profiles_budget_seconds']}, old)

    def test_budget_override_rejects_nonfinite_and_retained_tool_changes(self):
        self.prepare_unstarted_profiles()
        for budget in (0, -1, True, float('inf'), float('nan'), 10 ** 1000):
            with self.subTest(budget=type(budget).__name__), self.assertRaisesRegex(ValueError, 'positive finite'):
                self.retry(profiles_only=True, continue_profiles=True, profiles_budget_seconds=budget)
            self.assertFalse(self.new.exists())
        self.prepare_continuation()
        with self.assertRaisesRegex(ValueError, 'zero-attempt'):
            self.retry(profiles_only=True, continue_profiles=True, profiles_budget_seconds=3600)

    def test_timeout_override_rejects_nonfinite_and_retained_tool_changes(self):
        self.prepare_unstarted_profiles()
        for timeout in (0, -1, True, float('inf'), float('nan')):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, 'positive finite'):
                self.retry(profiles_only=True, continue_profiles=True, profile_timeout_seconds=timeout)
            self.assertFalse(self.new.exists())
        self.prepare_continuation()
        with self.assertRaisesRegex(ValueError, 'zero-attempt'):
            self.retry(profiles_only=True, continue_profiles=True, profile_timeout_seconds=3600)

    def test_first_difference_is_exact_bounded_and_contains_no_values(self):
        before = {'cpu_artifacts': {'baseline': {'runtime': {'PATH': 'SECRET-old'}}}}
        after = {'cpu_artifacts': {'baseline': {'runtime': {'PATH': 'SECRET-new'}}}}
        difference = study.first_difference(before, after)
        self.assertEqual(difference['path'], '/cpu_artifacts/baseline/runtime/PATH')
        self.assertEqual(difference['expected_sha256'], study.digest('SECRET-old'))
        self.assertNotIn('SECRET', json.dumps(difference))
        self.assertIsNone(study.first_difference(before, before))
        self.assertEqual(study.first_difference({'a': [1]}, {'a': [1, 2]})['path'], '/a/1')
        self.assertEqual(study.first_difference({'a': 1}, {'a': True})['observed_type'], 'bool')
        self.assertLessEqual(len(study.first_difference({'x' * 2000: 1}, {})['path']), 512)

    def test_admission_difference_does_not_overwrite_existing_artifact_evidence(self):
        path = self.base / 'diagnostic-admission.json'
        value = {'artifact_sha256': {'library': 'original'}, 'configuration_key': 'release'}
        study.write_json(path, value)
        before = path.read_bytes()
        changed = {**value, 'artifact_sha256': {'library': 'changed'}}
        with self.assertRaisesRegex(RuntimeError, '/artifact_sha256/library'):
            study.publish_admission(path, changed)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(study.publish_admission(path, changed, publish=False), changed)
        self.assertEqual(path.read_bytes(), before)

    def test_hour_defaults_remain_bounded_by_actual_allocation(self):
        with patch.object(study.time, 'monotonic', return_value=100), \
                patch.object(study.time, 'time', return_value=1000):
            self.assertEqual(study.worker_budget({}, {}, 'profiles', 100), 3480)
            self.assertEqual(study.worker_budget({}, {'profiles_budget_seconds': 7200}, 'profiles', 100,
                                                 end_epoch=1400), 280)
            for invalid in (True, 0, -1, float('inf'), float('nan')):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    study.worker_budget({}, {'profiles_budget_seconds': invalid}, 'profiles', 100)
        runner = study.Runner(self.base / 'default-runner', 5000)
        self.assertEqual(runner.command_timeout, 3600)
        self.assertEqual(study.Runner(self.base / 'long-runner', 8000, 7200).command_timeout, 7200)

    def test_frozen_diagnostic_recovers_legacy_value_without_writing(self):
        frozen = self.old / 'executor/gpu_study.py'
        frozen.write_text('''from types import SimpleNamespace
study_workloads=SimpleNamespace(load_suite=lambda path: {})
def admit(root,plan,suite):
    value={"cpu_artifacts":{},"ambient_fixture":"secret-current-shell"}
    raise RuntimeError("study build identity changed after admission")
''')
        before = self.snapshot()
        module_before = study.sys.modules['study_workloads']
        observed = study.frozen_admission_value(self.old, self.plan)
        self.assertEqual(observed['ambient_fixture'], 'secret-current-shell')
        self.assertEqual(before, self.snapshot())
        self.assertIs(study.sys.modules['study_workloads'], module_before)

    def test_frozen_diagnostic_refuses_earlier_failure_and_any_write(self):
        frozen = self.old / 'executor/gpu_study.py'
        for statement, message in [('raise RuntimeError("canonical binding changed")', 'canonical binding'),
                                   ('write_json(root/"admission.json",{})', 'refused a study write')]:
            frozen.write_text('from types import SimpleNamespace\n'
                'study_workloads=SimpleNamespace(load_suite=lambda path: {})\n'
                'def admit(root,plan,suite):\n    ' + statement + '\n')
            before = self.snapshot()
            with self.subTest(statement=statement), self.assertRaisesRegex(RuntimeError, message):
                study.frozen_admission_value(self.old, self.plan)
            self.assertEqual(before, self.snapshot())

    def test_owned_cpu_diagnostic_reports_actual_legacy_field_without_execution(self):
        owner = study.pwd.getpwuid(os.getuid()).pw_name + '(' + str(os.getuid()) + ')'
        controller = ('JobId=55 UserId=' + owner + ' JobState=RUNNING Account=gts-sbryngelson3 '
                      'QOS=embers Partition=cpu-small NumNodes=1 NumTasks=1 NodeList=cpu-fixture')
        expected = {'plan_sha256': self.plan['plan_sha256'], 'artifact_sha256': {}, 'runtime': {'PATH': 'secret-old'}}
        observed = {**expected, 'runtime': {'PATH': 'secret-new'}}
        study.write_json(self.old / 'admission.json', expected)
        before = self.snapshot()
        commands = []
        def metadata(argv, **kwargs):
            commands.append(argv)
            return subprocess.CompletedProcess(argv, 0, controller if 'job' in argv else 'cpu-fixture\n', '')
        with patch.dict(os.environ, {'SLURM_JOB_ID': '55'}), \
             patch.object(study.socket, 'getfqdn', return_value='cpu-fixture.pace'), \
             patch.object(study.socket, 'gethostname', return_value='cpu-fixture'), \
             patch.object(study.subprocess, 'run', side_effect=metadata), \
             patch.object(study, 'verify', return_value=self.plan), \
             patch.object(study, 'checked_builds'), \
             patch.object(study, 'frozen_admission_value', return_value=observed), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            study.diagnose_admission(argparse.Namespace(run_dir=str(self.old)))
        record = json.loads(output.getvalue())
        self.assertEqual(record['legacy_exact_difference']['path'], '/runtime/PATH')
        self.assertFalse(record['application_execution'])
        self.assertFalse(record['study_writes'])
        self.assertNotIn('secret-', output.getvalue())
        self.assertEqual(before, self.snapshot())
        self.assertEqual([command[0] for command in commands], ['scontrol', 'scontrol'])

    def test_diagnostic_requires_owned_compute_and_an_existing_admission(self):
        with self.assertRaisesRegex(RuntimeError, 'owned free compute'):
            study.diagnose_admission(argparse.Namespace(run_dir=str(self.old)))
        with patch.dict(os.environ, {'SLURM_JOB_ID': '55'}), \
             patch.object(study.socket, 'getfqdn', return_value='compute'), \
             patch.object(study.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0,
                 'JobId=55 UserId=unrelated(0) JobState=RUNNING Account=paid', '')), \
             patch.object(study, 'verify') as verify:
            with self.assertRaisesRegex(RuntimeError, 'live owned'):
                study.diagnose_admission(argparse.Namespace(run_dir=str(self.old)))
            verify.assert_not_called()

    def test_profiles_only_preserves_timing_bytes_and_submits_only_profiles(self):
        gpu, summary, other = self.prepare_profile_timing()
        before = self.snapshot()
        plan = self.retry(profiles_only=True, prepare=False)
        self.assertEqual(before, self.snapshot())
        state = study.read_json(self.new / 'state.json')
        attempt = study.completed(state, 'timings')
        self.assertTrue(attempt['timings_reused'])
        self.assertEqual(Path(attempt['summary']).read_bytes(), summary.read_bytes())
        self.assertEqual(plan['timing_reuse']['gpu'], gpu)
        self.assertEqual(plan['timing_reuse']['artifact_sha256'][str(other)], study.sha(other))
        self.assertEqual([job['stage'] for job in state['jobs']], ['profiles'])
        self.assertFalse(any('--dependency' in value for value in state['jobs'][0]['argv']))
        with self.assertRaisesRegex(RuntimeError, 'cannot submit new timings'):
            study.submit_stage(self.new, 'timings')
        report = study.report(self.new)
        self.assertEqual(report['timing_provenance']['summary_sha256'], study.sha(summary))
        self.assertIn('no new timing measurements', (self.new / 'report.md').read_text())
        other.write_text('modified retained field')
        with self.assertRaisesRegex(RuntimeError, 'reused timing artifact changed'):
            study.verify(self.new)

    def test_profiles_only_requires_exact_suite_and_completed_fingerprinted_timing(self):
        with self.assertRaisesRegex(ValueError, 'forbids --cases'):
            self.retry(profiles_only=True, cases='dg-p1')
        with self.assertRaisesRegex(RuntimeError, 'fingerprinted timings'):
            self.retry(profiles_only=True)
        gpu, summary, _ = self.prepare_profile_timing()
        summary.write_text('changed completed summary')
        with self.assertRaisesRegex(RuntimeError, 'completed measurement artifact changed'):
            self.retry(profiles_only=True)
        self.assertFalse(self.new.exists())

    def test_profiles_only_suite_changes_only_capture_selection_and_retains_exact_timing(self):
        _, summary, _ = self.prepare_profile_timing()
        suite = study.read_json(self.old / 'suite.json')
        original_profile = dict(suite['cases'][0]['profile'])
        suite['cases'][0]['profile'].update(kernel_regex='ClosureKernel', launch_count=2)
        replacement = self.base / 'closure-suite.json'
        replacement.write_text(json.dumps(suite, separators=(',', ':')))
        before = self.snapshot()
        plan = self.retry(profiles_only=True, suite=str(replacement))
        self.assertEqual(before, self.snapshot())
        self.assertEqual((self.new / 'suite.json').read_bytes(), replacement.read_bytes())
        reuse = plan['timing_reuse']
        self.assertEqual(reuse['suite_sha256'], study.sha(self.old / 'suite.json'))
        self.assertEqual(reuse['profile_suite_sha256'], plan['suite_sha256'])
        self.assertEqual(reuse['profile_overrides'], {suite['cases'][0]['id']: {
            'original': original_profile, 'replacement': suite['cases'][0]['profile']}})
        attempt = study.completed(study.read_json(self.new / 'state.json'), 'timings')
        self.assertEqual(Path(attempt['summary']).read_bytes(), summary.read_bytes())
        self.assertNotEqual(plan['suite_sha256'], plan['retry_origin']['suite_sha256'])
        self.assertEqual(self.calls[0][0], 'squeue')
        self.assertFalse(any(call[0] == 'sbatch' for call in self.calls))

    def test_profiles_only_sparse_tracked_suite_matches_explicit_frozen_defaults(self):
        tracked = Path(study.__file__).parent / 'suites/pace-full-gpu-igr.json'
        sparse = study.read_json(tracked)
        frozen = study.study_workloads.load_suite(tracked)
        self.assertNotIn('binary', sparse['cases'][0])
        self.assertIn('binary', frozen['cases'][0])
        self.assertNotIn('atol', sparse['cases'][0]['parameters'])
        study.write_json(self.old / 'suite.json', frozen)
        self.plan['suite_sha256'] = study.sha(self.old / 'suite.json')
        self.save_plan()
        _, summary, _ = self.prepare_profile_timing()
        sparse['cases'][0]['profile']['kernel_regex'] = 'MeasuredClosureKernel'
        replacement = self.base / 'sparse-closure-suite.json'
        study.write_json(replacement, sparse)
        plan = self.retry(profiles_only=True, suite=str(replacement))
        self.assertEqual((self.new / 'suite.json').read_bytes(), replacement.read_bytes())
        self.assertEqual(set(plan['timing_reuse']['profile_overrides']), {sparse['cases'][0]['id']})
        shadow = study.completed(study.read_json(self.new / 'state.json'), 'timings')
        self.assertEqual(Path(shadow['summary']).read_bytes(), summary.read_bytes())

    def test_profiles_only_canonical_defaults_reject_tolerance_and_changed_default_drift(self):
        tracked = Path(study.__file__).parent / 'suites/pace-full-gpu-igr.json'
        sparse = study.read_json(tracked)
        frozen_path = self.base / 'explicit-frozen-suite.json'
        study.write_json(frozen_path, study.study_workloads.load_suite(tracked))
        replacement = self.base / 'sparse-suite.json'
        for field, value in (('atol', 1e-4), ('rtol', 1e-3), ('dt', 2e-5), ('engine', 'tensor')):
            changed = json.loads(json.dumps(sparse))
            changed['cases'][0]['parameters'][field] = value
            study.write_json(replacement, changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                study.profile_only_overrides(frozen_path, replacement)
        study.write_json(replacement, sparse)
        with patch.object(study.study_workloads, 'ATOL', 1e-4):
            with self.assertRaisesRegex(ValueError, 'case.profile'):
                study.profile_only_overrides(frozen_path, replacement)

    def test_profiles_only_suite_rejects_physics_selection_environment_and_limit_drift(self):
        self.prepare_profile_timing()
        original = study.read_json(self.old / 'suite.json')
        replacement = self.base / 'different-suite.json'
        mutations = [
            lambda suite: suite['cases'][0]['parameters'].update(dt=2e-5),
            lambda suite: suite['cases'][0].update(environment={'IGRLAB_IGR_DENSE_INVERSE': '1'}),
            lambda suite: suite['cases'][0].update(enabled=False),
            lambda suite: suite['cases'][1].update(enabled=True),
            lambda suite: suite['cases'][0].update(optional=True),
            lambda suite: suite['cases'][0].update(id='renamed-case'),
            lambda suite: suite['cases'].append({**suite['cases'][0], 'id': 'added-case'}),
            lambda suite: suite['cases'].pop(),
            lambda suite: suite['cases'].reverse(),
            lambda suite: suite.update(repetitions=4),
            lambda suite: suite.update(warmups=2),
            lambda suite: suite.update(command_timeout_seconds=42),
            lambda suite: suite.update(profiles_budget_seconds=42),
            lambda suite: suite.update(correctness_regex='^another-test$'),
            lambda suite: suite.update(name='another-name'),
        ]
        for mutate in mutations:
            value = json.loads(json.dumps(original))
            mutate(value)
            replacement.write_text(json.dumps(value))
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.retry(profiles_only=True, suite=str(replacement))
            self.assertFalse(self.new.exists())
        self.assertEqual(self.calls, [])

    def test_profiles_only_suite_mutation_between_admission_and_copy_rejects_before_plan(self):
        self.prepare_profile_timing()
        replacement = self.base / 'mutable-suite.json'
        value = study.read_json(self.old / 'suite.json')
        value['cases'][0]['profile']['kernel_regex'] = 'ClosureKernel'
        study.write_json(replacement, value)
        before = self.snapshot()
        copy = study.shutil.copy2
        def replace_then_copy(source, destination):
            if Path(source) == replacement:
                value['cases'][0]['parameters']['dt'] = 2e-5
                study.write_json(replacement, value)
            return copy(source, destination)
        with patch.object(study.shutil, 'copy2', side_effect=replace_then_copy):
            with self.assertRaisesRegex(ValueError, 'changed before freezing'):
                self.retry(profiles_only=True, suite=str(replacement), prepare=False)
        self.assertEqual(before, self.snapshot())
        self.assertFalse((self.new / 'plan.json').exists())
        self.assertFalse(any(call[0] == 'sbatch' for call in self.calls))

    def test_chained_profile_overrides_retain_original_timing_suite_hash(self):
        self.prepare_profile_timing()
        replacement = self.base / 'closure-suite.json'
        value = study.read_json(self.old / 'suite.json')
        value['cases'][0]['profile']['kernel_regex'] = 'ClosureKernel'
        study.write_json(replacement, value)
        intermediate = self.retry(profiles_only=True, suite=str(replacement))
        final = self.base / 'third'
        value['cases'][0]['profile']['kernel_regex'] = 'AnotherClosureKernel'
        study.write_json(replacement, value)
        before = self.snapshot()
        args = argparse.Namespace(from_run=str(self.new), run_dir=str(final), profiles_only=True,
            suite=str(replacement), cases=None, prepare_only=True, timings_time=None, profiles_time=None)
        with contextlib.redirect_stdout(io.StringIO()):
            study.retry(args)
        plan = study.load_plan(final)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(plan['timing_reuse']['suite_sha256'], self.plan['suite_sha256'])
        self.assertEqual(plan['retry_origin']['suite_sha256'], intermediate['suite_sha256'])
        self.assertEqual(plan['timing_reuse']['profile_suite_sha256'], study.sha(replacement))
        study.verify(final)

    def test_profiles_only_rejects_missing_case_and_mismatched_timing_gpu(self):
        gpu, summary, _ = self.prepare_profile_timing()
        for mutation in ('case', 'gpu'):
            value = study.read_json(summary)
            if mutation == 'case': value['cases'] = {}
            else: value['gpu'] = {**gpu, 'uuid': 'GPU-10112233-4455-6677-8899-aabbccddeeff'}
            study.write_json(summary, value)
            attempt = self.state['attempts'][0]
            attempt['summary_sha256'] = study.sha(summary)
            attempt['artifact_sha256'][str(summary.relative_to(self.old))] = study.sha(summary)
            self.save_state()
            with self.assertRaisesRegex(RuntimeError, 'cases differ|GPU identities differ'):
                self.retry(profiles_only=True)
            self.prepare_profile_timing()
        self.assertFalse(self.new.exists())

    def test_profiles_only_admits_only_original_owned_interactive_allocation(self):
        self.prepare_profile_timing()
        self.state['jobs'] = []
        self.save_state()
        self.queue_stdout = '33|RUNNING\n'
        self.outcomes['33'] = ('RUNNING', '0:0')
        self.interactive_info = (f'JobId=33 UserId={study.getpass.getuser()}({os.getuid()}) '
            'JobState=RUNNING BatchFlag=0 Account=gts-sbryngelson3 QOS=embers Partition=gpu-a100 '
            'NumTasks=1 AllocTRES=cpu=8,mem=48G,node=1,gres/gpu=1')
        before = self.snapshot()
        plan = self.retry(profiles_only=True)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(plan['timing_reuse']['interactive_scheduler_admission']['33']['BatchFlag'], '0')
        self.assertEqual(study.read_json(self.new / 'state.json')['jobs'], [])

    def test_profiles_only_rejects_running_workers_active_batch_and_wrong_allocation(self):
        self.prepare_profile_timing()
        self.state['attempts'].append({'stage': 'profiles', 'status': 'running', 'job_id': '33'})
        self.save_state()
        with self.assertRaisesRegex(RuntimeError, 'worker is running'):
            self.retry(profiles_only=True)
        self.state['attempts'].pop()
        self.save_state()
        self.queue_stdout = '33|RUNNING\n'
        self.outcomes['33'] = ('RUNNING', '0:0')
        with self.assertRaisesRegex(RuntimeError, 'batch stage is active'):
            self.retry(profiles_only=True)
        self.state['jobs'] = []
        self.save_state()
        valid = (f'JobId=33 UserId={study.getpass.getuser()}({os.getuid()}) '
            'JobState=RUNNING BatchFlag=0 Account=gts-sbryngelson3 QOS=embers Partition=gpu-a100 '
            'NumTasks=1 AllocTRES=cpu=8,gres/gpu=1')
        for previous, wrong in (('BatchFlag=0', 'BatchFlag=1'), ('QOS=embers', 'QOS=paid'),
                                ('Partition=gpu-a100', 'Partition=gpu-v100'),
                                ('gres/gpu=1', 'gres/gpu=2'),
                                (f'UserId={study.getpass.getuser()}', 'UserId=unrelated')):
            self.interactive_info = valid.replace(previous, wrong)
            with self.subTest(wrong=wrong), self.assertRaisesRegex(RuntimeError, 'owned free one-A100'):
                self.retry(profiles_only=True)
        self.assertFalse(self.new.exists())

    def profile_record(self, case, directory, gpu, job, names=('nsys', 'ncu')):
        profile = case['profile']
        record = {'status': 'complete', 'adapter': case['adapter'], 'gpu_uuid': gpu['uuid'],
            'selected_kernel_regex': profile['kernel_regex'], 'selection': profile.get('selection', 'launches'),
            'launch_skip': profile['launch_skip'], 'launch_count': profile['launch_count'],
            'required_kernel_groups': profile.get('kernel_groups', {}), 'captures': {}, 'tools': {},
            'application_logs': []}
        def artifact(path):
            return {'path': str(path), 'sha256': study.sha(path), 'bytes': path.stat().st_size}
        for name in names:
            log = directory / name
            log.mkdir(parents=True)
            (log / 'stdout.log').write_text('admitted numerical output fixture\n')
            (log / 'stderr.log').write_text('')
            native = directory / ('capture-' + name + ('.nsys-rep' if name == 'nsys' else '.ncu-repz'))
            native.write_text('synthetic native bytes; no profiler execution\n')
            identity = {'executable': {'path': '/selected/' + name, 'sha256': 'a' * 64, 'bytes': 1}}
            for key in ('version_stdout', 'help_stdout'):
                path = log / (key + '.log')
                path.write_text('synthetic tool identity\n')
                identity[key] = artifact(path)
            app = study.study_workloads.command(case, self.binary, self.source, directory, profile=True)
            if name == 'nsys':
                prefix = ['/selected/nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=none',
                          '--output', str(directory / 'capture-nsys')]
            else:
                prefix = ['/selected/ncu', '--config-file', 'off', '--target-processes', 'application-only',
                          '--profile-from-start', 'on', '--kernel-name-base', 'demangled']
                if profile.get('selection') == 'first-per-kernel':
                    prefix += ['--kernel-id', '::regex:' + study.study_profiles._kernel_id_regex(profile['kernel_regex']) + ':1']
                else:
                    prefix += ['--kernel-name', 'regex:' + profile['kernel_regex'], '--launch-skip',
                               str(profile['launch_skip']), '--launch-count', str(profile['launch_count'])]
                prefix += ['--rename-kernels', 'off', '--metrics', 'gpu__time_duration.avg',
                           '--section', 'LaunchStats', '--export', str(directory / 'capture-ncu')]
            env = {'SLURM_JOB_ID': job, 'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3', 'SLURM_JOB_QOS': 'embers',
                   'SLURM_JOB_PARTITION': 'gpu-a100', 'SLURM_NTASKS': '1', 'CUDA_VISIBLE_DEVICES': '0'}
            env.update({key: value for key, value in study.study_workloads.environment(case, directory, profile=True).items()
                        if key in study.SAFE_ENV})
            command = {'argv': prefix + app, 'cwd': str(log), 'environment': env, 'status': 'running'}
            study.write_json(log / 'command.json', command)
            study.write_json(log / 'result.json', {**command, 'status': 'complete', 'returncode': 0,
                'stdout_sha256': study.sha(log / 'stdout.log'), 'stderr_sha256': study.sha(log / 'stderr.log'),
                'process_wall_seconds': 3.})
            logs = {'stdout': str(log / 'stdout.log'), 'stderr': str(log / 'stderr.log'),
                    'receipt': str(log / 'result.json')}
            capture = {'status': 'complete', 'native': artifact(native),
                       'application_stdout': logs['stdout'], 'application_stderr': logs['stderr'],
                       'application_stdout_sha256': study.sha(log / 'stdout.log')}
            if name == 'ncu':
                capture.update(queried_metrics=['gpu__time_duration.avg'], sections=['LaunchStats'])
            record['captures'][name] = capture
            record['tools'][name] = identity
            record['application_logs'].append(logs)
        return record

    def prepare_continuation(self, names=('nsys', 'ncu')):
        gpu, summary, _ = self.prepare_profile_timing()
        timing = study.read_json(summary)
        for item in timing['cases'].values():
            item['observations'] = [{'signature': {'work': 10}}]
        study.write_json(summary, timing)
        attempt = self.state['attempts'][0]
        attempt['summary_sha256'] = study.sha(summary)
        attempt['artifact_sha256'][str(summary.relative_to(self.old))] = study.sha(summary)
        for name, source in [('gpu_study.py', study.__file__), ('study_workloads.py', study.study_workloads.__file__)]:
            (self.old / 'executor' / name).write_bytes(Path(source).read_bytes())
            self.plan['executor_sha256'][name] = study.sha(self.old / 'executor' / name)
        self.save_plan()
        case = study.selected_cases(study.study_workloads.load_suite(self.old / 'suite.json'))[0]
        directory = self.old / 'attempts' / 'profiles-old'
        baseline = self.profile_record(case, directory / case['id'] / 'baseline', gpu, '34', names)
        checkpoint = directory / 'partial.json'
        study.write_json(checkpoint, {'schema_version': 1, 'status': 'profiles_partial', 'gpu': gpu,
            'cases': {case['id']: {'baseline': baseline, 'candidate': {'status': 'failed', 'captures': {}}}}})
        self.state['attempts'].append({'id': 'profiles-old', 'stage': 'profiles', 'status': 'profiles_partial',
            'job_id': '34', 'directory': str(directory), 'gpu': gpu})
        self.save_state()
        return gpu, case, checkpoint

    def run_continuation(self, plan, case, gpu, *, fail_ncu=False, wrong_signature=False, suite=None):
        output = self.new / 'attempts' / 'profiles-new'
        output.mkdir()
        runner = SimpleNamespace(output=output)
        admission = {'binaries': {member: {case['binary']: str(self.binary)} for member in study.MEMBERS}}
        calls = []
        def capture(case, argv, runner, directory, expected_gpu, **kwargs):
            names = kwargs['requested_tools']
            calls.append((directory.name, names))
            record = self.profile_record(case, directory, expected_gpu, '35',
                                         tuple(name for name in names if not fail_ncu or name != 'ncu'))
            for name, item in record['captures'].items():
                item.update(gpu_uuid=expected_gpu['uuid'], application_logs=[
                    log for log in record['application_logs'] if Path(log['stdout']).parent.name == name])
            if fail_ncu:
                record['captures']['ncu'] = {'status': 'failed', 'gpu_uuid': expected_gpu['uuid']}
                record['status'] = 'partial'
                kwargs['checkpoint'](record)
                raise RuntimeError('synthetic NCU timeout')
            return record
        observed = {'signature': {'work': 11 if wrong_signature else 10}}
        with patch.object(study.study_profiles, 'capture', side_effect=capture), \
                patch.object(study.study_workloads, 'parse', return_value=observed) as parse:
            if fail_ncu or wrong_signature:
                with self.assertRaises(RuntimeError):
                    study.profiles(self.new, plan, suite or {'cases': [case]}, admission, runner, gpu)
                result = study.read_json(output / 'partial.json') if (output / 'partial.json').exists() else None
            else:
                result = study.profiles(self.new, plan, suite or {'cases': [case]}, admission, runner, gpu)
        return result, calls, parse.call_args_list, output

    def test_continue_profiles_freezes_successful_baseline_without_timing_or_original_mutation(self):
        gpu, case, checkpoint = self.prepare_continuation()
        before = self.snapshot()
        plan = self.retry(profiles_only=True, continue_profiles=True, prepare=False)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(plan['profile_continuation']['checkpoint_sha256'], study.sha(checkpoint))
        self.assertEqual(set(plan['profile_continuation']['captures'][case['id']]['baseline']), {'nsys', 'ncu'})
        self.assertNotIn('candidate', plan['profile_continuation']['captures'][case['id']])
        state = study.read_json(self.new / 'state.json')
        self.assertEqual([job['stage'] for job in state['jobs']], ['profiles'])
        different = {**gpu, 'uuid': 'GPU-11112233-4455-6677-8899-aabbccddeeff'}
        study.admit_reused_timing_gpu(plan, different)
        result, calls, parses, _ = self.run_continuation(plan, case, different)
        self.assertEqual(calls, [('candidate', ['nsys', 'ncu'])])
        self.assertEqual(result['status'], 'analysis_complete')
        self.assertTrue(all(not row['matched_gpu_identity'] for row in result['tool_cohorts'][case['id']].values()))
        self.assertEqual([item.args[3]['uuid'] for item in parses], [gpu['uuid']] * 2 + [different['uuid']] * 2)
        for name in ('nsys', 'ncu'):
            self.assertEqual(result['cases'][case['id']]['baseline']['captures'][name]['gpu_uuid'], gpu['uuid'])
        study.report(self.new)
        self.assertIn('do not claim matched profiler timing', (self.new / 'report.md').read_text())

    def test_continue_profiles_only_nsys_success_requires_baseline_ncu_and_both_candidate_tools(self):
        gpu, case, _ = self.prepare_continuation(names=('nsys',))
        plan = self.retry(profiles_only=True, continue_profiles=True)
        result, calls, _, _ = self.run_continuation(plan, case, gpu)
        self.assertEqual(calls, [('baseline', ['ncu']), ('candidate', ['nsys', 'ncu'])])
        self.assertEqual(set(result['cases'][case['id']]['baseline']['captures']), {'nsys', 'ncu'})

    def test_continue_profiles_pre_checkpoint_failure_still_preserves_retained_tool(self):
        gpu, case, _ = self.prepare_continuation(names=('nsys',))
        plan = self.retry(profiles_only=True, continue_profiles=True)
        output = self.new / 'attempts' / 'early-failure'
        output.mkdir()
        admission = {'binaries': {member: {case['binary']: str(self.binary)} for member in study.MEMBERS}}
        with patch.object(study.study_workloads, 'parse', return_value={'signature': {'work': 10}}), \
                patch.object(study.study_profiles, 'capture', side_effect=RuntimeError('tool discovery failure')) as capture:
            with self.assertRaisesRegex(RuntimeError, 'tool discovery'):
                study.profiles(self.new, plan, {'cases': [case]}, admission, SimpleNamespace(output=output), gpu)
        self.assertEqual(capture.call_args.kwargs['requested_tools'], ['ncu'])
        partial = study.read_json(output / 'partial.json')['cases'][case['id']]['baseline']
        self.assertEqual(set(partial['captures']), {'nsys'})
        self.assertEqual(partial['captures']['nsys']['status'], 'complete')
        self.assertEqual(partial['status'], 'partial')

    def test_continue_profiles_requires_exact_mode_and_suite(self):
        self.prepare_continuation()
        for kwargs in ({'continue_profiles': True}, {'profiles_only': True, 'continue_profiles': True, 'suite': 'x'},
                       {'profiles_only': True, 'continue_profiles': True, 'cases': 'dg-p1'}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'requires --profiles-only'):
                self.retry(**kwargs)
        self.assertFalse(self.new.exists())

    def test_continue_profiles_wrong_retained_application_fails_before_fresh_capture(self):
        gpu, case, _ = self.prepare_continuation()
        plan = self.retry(profiles_only=True, continue_profiles=True)
        _, calls, _, _ = self.run_continuation(plan, case, gpu, wrong_signature=True)
        self.assertEqual(calls, [])

    def test_continue_profiles_checks_native_bytes_only_on_compute_before_capture(self):
        gpu, case, _ = self.prepare_continuation()
        plan = self.retry(profiles_only=True, continue_profiles=True)
        native = Path(plan['profile_continuation']['captures'][case['id']]['baseline']['nsys']['capture']['native']['path'])
        native.write_text('changed native bytes')
        study.verify(self.new)  # Login-side verify does not read native reports.
        runner = SimpleNamespace(output=self.new / 'attempts')
        with patch.object(study.study_profiles, 'capture') as capture, self.assertRaisesRegex(RuntimeError, 'native profile artifact changed'):
            study.profiles(self.new, plan, {'cases': [case]}, {}, runner, gpu)
        capture.assert_not_called()

    def test_continue_profiles_receipt_command_control_and_gpu_drift_reject(self):
        _, case, checkpoint = self.prepare_continuation()
        original = study.read_json(checkpoint)
        logs = original['cases'][case['id']]['baseline']['application_logs'][0]
        result_path, command_path = Path(logs['receipt']), Path(logs['receipt']).with_name('command.json')
        result, command = study.read_json(result_path), study.read_json(command_path)
        for field in ('argv', 'environment'):
            changed_result, changed_command = json.loads(json.dumps(result)), json.loads(json.dumps(command))
            if field == 'argv':
                changed_result[field][-1] = 'wrong-workload'
                changed_command[field] = changed_result[field]
            else:
                changed_result[field]['IGRLAB_PROFILE_NVTX'] = '0'
                changed_command[field] = changed_result[field]
            study.write_json(result_path, changed_result)
            study.write_json(command_path, changed_command)
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'argv differs|environment differs'):
                self.retry(profiles_only=True, continue_profiles=True)
        study.write_json(result_path, result)
        study.write_json(command_path, command)
        original['cases'][case['id']]['baseline']['gpu_uuid'] = 'GPU-11112233-4455-6677-8899-aabbccddeeff'
        study.write_json(checkpoint, original)
        with self.assertRaisesRegex(RuntimeError, 'GPU identities differ'):
            self.retry(profiles_only=True, continue_profiles=True)

    def test_continue_profiles_empty_identity_changed_log_and_active_jobs_reject(self):
        _, case, checkpoint = self.prepare_continuation()
        original = study.read_json(checkpoint)
        altered = json.loads(json.dumps(original))
        altered['cases'][case['id']]['baseline']['tools']['nsys'] = {}
        study.write_json(checkpoint, altered)
        with self.assertRaisesRegex(RuntimeError, 'profiler identity'):
            self.retry(profiles_only=True, continue_profiles=True)
        study.write_json(checkpoint, original)
        log = Path(original['cases'][case['id']]['baseline']['application_logs'][0]['stdout'])
        content = log.read_bytes()
        log.write_text('changed stdout')
        with self.assertRaisesRegex(RuntimeError, 'matching logs'):
            self.retry(profiles_only=True, continue_profiles=True)
        log.write_bytes(content)
        self.queue_stdout, self.outcomes['34'] = '34|RUNNING\n', ('RUNNING', '0:0')
        with self.assertRaisesRegex(RuntimeError, 'inactive'):
            self.retry(profiles_only=True, continue_profiles=True)
        self.assertFalse(self.new.exists())

    def test_continue_profiles_missing_receipt_and_changed_profiler_help_reject_before_capture(self):
        gpu, case, checkpoint = self.prepare_continuation()
        record = study.read_json(checkpoint)['cases'][case['id']]['baseline']
        receipt = Path(record['application_logs'][0]['receipt'])
        content = receipt.read_bytes()
        receipt.unlink()
        with self.assertRaisesRegex(RuntimeError, 'artifact is missing'):
            self.retry(profiles_only=True, continue_profiles=True)
        receipt.write_bytes(content)
        plan = self.retry(profiles_only=True, continue_profiles=True)
        help_file = Path(record['tools']['nsys']['help_stdout']['path'])
        help_file.write_text('changed help provenance')
        runner = SimpleNamespace(output=self.new / 'attempts')
        with patch.object(study.study_profiles, 'capture') as capture, self.assertRaisesRegex(RuntimeError, 'native profile artifact changed'):
            study.profiles(self.new, plan, {'cases': [case]}, {}, runner, gpu)
        capture.assert_not_called()

    def test_continue_profiles_no_success_or_no_missing_tools_reject(self):
        _, case, checkpoint = self.prepare_continuation()
        original = study.read_json(checkpoint)
        empty = json.loads(json.dumps(original))
        empty['cases'][case['id']]['baseline']['captures'] = {}
        study.write_json(checkpoint, empty)
        with self.assertRaisesRegex(RuntimeError, 'both retained completed tools and missing'):
            self.retry(profiles_only=True, continue_profiles=True)
        original['cases'][case['id']]['candidate'] = self.profile_record(case,
            checkpoint.parent / case['id'] / 'candidate', original['gpu'], '34')
        study.write_json(checkpoint, original)
        with self.assertRaisesRegex(RuntimeError, 'both retained completed tools and missing'):
            self.retry(profiles_only=True, continue_profiles=True)

    def test_chained_continuation_retains_original_gpu_and_only_missing_candidate_ncu(self):
        gpu, case, _ = self.prepare_continuation()
        plan = self.retry(profiles_only=True, continue_profiles=True)
        different = {**gpu, 'uuid': 'GPU-11112233-4455-6677-8899-aabbccddeeff'}
        partial, calls, _, output = self.run_continuation(plan, case, different, fail_ncu=True)
        self.assertEqual(calls, [('candidate', ['nsys', 'ncu'])])
        self.assertEqual(partial['cases'][case['id']]['candidate']['status'], 'partial')
        state = study.read_json(self.new / 'state.json')
        state['attempts'].append({'id': 'profiles-new', 'stage': 'profiles', 'status': 'profiles_partial',
            'directory': str(output), 'job_id': '35', 'gpu': different})
        study.write_json(self.new / 'state.json', state)
        self.outcomes['35'] = ('FAILED', '1:0')
        self.old, self.new = self.new, self.base / 'third'
        next_plan = self.retry(profiles_only=True, continue_profiles=True)
        captures = next_plan['profile_continuation']['captures'][case['id']]
        self.assertEqual(captures['baseline']['nsys']['gpu']['uuid'], gpu['uuid'])
        self.assertEqual(captures['candidate']['nsys']['gpu']['uuid'], different['uuid'])
        newest = {**gpu, 'uuid': 'GPU-22112233-4455-6677-8899-aabbccddeeff'}
        result, calls, _, _ = self.run_continuation(next_plan, case, newest)
        self.assertEqual(calls, [('candidate', ['ncu'])])
        self.assertEqual(result['status'], 'analysis_complete')
        self.assertIsNone(result['cases'][case['id']]['candidate']['gpu_uuid'])
        self.assertFalse(result['tool_cohorts'][case['id']]['nsys']['matched_gpu_identity'])
        self.assertFalse(result['tool_cohorts'][case['id']]['ncu']['matched_gpu_identity'])


if __name__ == '__main__':
    unittest.main()
