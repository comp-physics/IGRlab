#!/usr/bin/env python3
"""Host-only study protocol tests; no application builds or GPU commands."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import gpu_study as study


class StudyProtocol(unittest.TestCase):
    def setUp(self):
        # Source snapshot contracts have independent real-Git fixtures.
        source_check = patch.object(study.study_build, 'verify_source')
        source_check.start()
        self.addCleanup(source_check.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'study'
        self.root.mkdir()
        (self.root / 'executor').mkdir()
        for name in study.EXECUTOR_FILES:
            (self.root / 'executor' / name).write_text('frozen executor\n')
        suite = {'schema_version': 1, 'warmups': 1, 'repetitions': 3,
                 'cases': [{'id': 'fixture', 'binary': 'fixture', 'adapter': 'hdg', 'parameters': {}}]}
        study.write_json(self.root / 'suite.json', suite)
        self.plan = {'schema_version': 1, 'run_dir': str(self.root),
                     'sources': {m: {} for m in study.MEMBERS}, 'builds': {},
                     'suite_sha256': study.sha(self.root / 'suite.json'),
                     'executor_sha256': {name: study.sha(self.root / 'executor' / name)
                                        for name in study.EXECUTOR_FILES}}
        self.plan['plan_sha256'] = study.digest(self.plan)
        study.write_json(self.root / 'plan.json', self.plan)
        self.state = {'schema_version': 1, 'builds': {}, 'jobs': [], 'attempts': []}
        study.write_json(self.root / 'state.json', self.state)

    def tearDown(self):
        self.temp.cleanup()

    def test_command_receipts_retain_nvtx_policy_without_unrelated_environment(self):
        runner = study.Runner(self.root / 'nvtx-receipts', 10)
        for enabled in ('0', '1'):
            env = {**os.environ, 'IGRLAB_PROFILE_NVTX': enabled,
                   'DG_IGR_PROFILE_NVTX': enabled, 'UNRELATED_SECRET': 'omit-me'}
            directory = runner.run('nvtx-' + enabled, [sys.executable, '-c', 'pass'], env=env)
            receipt = study.read_json(directory / 'command.json')
            self.assertEqual(receipt['environment']['IGRLAB_PROFILE_NVTX'], enabled)
            self.assertEqual(receipt['environment']['DG_IGR_PROFILE_NVTX'], enabled)
            self.assertNotIn('UNRELATED_SECRET', receipt['environment'])

    def save_plan(self):
        self.plan.pop('plan_sha256', None)
        self.plan['plan_sha256'] = study.digest(self.plan)
        study.write_json(self.root / 'plan.json', self.plan)

    def test_submit_cli_wall_times_defaults_overrides_and_invalid_limits(self):
        argv = ['submit', '--baseline', 'HEAD', '--run-dir', str(self.root), '--site-root', str(self.root)]
        with patch.object(study, 'submit') as submit:
            study.main(argv)
            args = submit.call_args.args[0]
            self.assertEqual((args.timings_time, args.profiles_time, args.validation_time),
                             ('01:00:00', '01:00:00', None))
            study.main(argv + ['--timings-time', '00:15:00', '--profiles-time', '00:30:00',
                              '--validation-time', '02:00:00'])
            args = submit.call_args.args[0]
            self.assertEqual((args.timings_time, args.profiles_time, args.validation_time),
                             ('00:15:00', '00:30:00', '02:00:00'))
            study.main(argv + ['--validation-time', '00:01:00'])
            self.assertEqual(submit.call_args.args[0].validation_time, '00:01:00')
        for flag, value in (('--timings-time', '15'), ('--timings-time', '00:04:00'),
                            ('--profiles-time', '01:01:00'), ('--profiles-time', '00:15:01'),
                            ('--profiles-time', '00:60:00'), ('--validation-time', '02:01:00'),
                            ('--validation-time', '00:00:00')):
            with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()), \
                    patch.object(study, 'submit') as submit:
                with self.assertRaises(SystemExit): study.main(argv + [flag, value])
                submit.assert_not_called()

    def test_create_freezes_stage_limits_and_passes_validation_to_canonical_plan(self):
        source = Path(self.temp.name) / 'source'
        (source / 'cmake/sites').mkdir(parents=True)
        (source / 'cmake/sites/pace.json').write_text('{"site":"pace"}')
        destination = Path(self.temp.name) / 'new-study'
        args = argparse.Namespace(run_dir=str(destination), source=str(source), site_root=str(self.root),
                                  cache_root=None, spack_root=str(self.root / 'spack'),
                                  suite=str(Path(study.__file__).parent / 'suites/pace-a100.json'),
                                  cases='dg-p1', baseline='HEAD', candidate='HEAD',
                                  timings_time='00:15:00', profiles_time='00:30:00', validation_time='00:15:00')
        with patch.object(study.study_build, 'freeze', return_value={'source_snapshot': str(source)}), \
                patch.object(study, 'cached_source', side_effect=lambda identity, cache: identity), \
                patch.object(study.study_build, 'build_plan', return_value={'app_build': '/fixture'}) as build:
            root, plan = study.create(args)
        self.assertEqual(plan['wall_time_limits'], {'timings': '00:15:00', 'profiles': '00:30:00'})
        self.assertEqual(study.load_plan(root)['wall_time_limits'], plan['wall_time_limits'])
        self.assertEqual(build.call_count, 2)
        self.assertTrue(all(call.kwargs == {'validation_time': '00:15:00'} for call in build.call_args_list))
        args.run_dir = str(Path(self.temp.name) / 'invalid-study')
        args.timings_time = '03:00:00'
        with patch.object(study.study_build, 'freeze') as freeze:
            with self.assertRaises(ValueError): study.create(args)
            freeze.assert_not_called()
        self.assertFalse(Path(args.run_dir).exists())

    def test_worker_budget_uses_suite_setup_elapsed_and_actual_scheduler_end(self):
        plan = {'wall_time_limits': {'timings': '00:15:00', 'profiles': '00:30:00'}}
        with patch.dict(os.environ, {}, clear=True), patch.object(study.time, 'monotonic', return_value=160):
            self.assertEqual(study.worker_budget({}, {}, 'timings', 100), 3000)
            self.assertEqual(study.worker_budget(plan, {}, 'timings', 100), 720)
            self.assertEqual(study.worker_budget(plan, {}, 'profiles', 100), 1620)
            self.assertEqual(study.worker_budget(plan, {'timings_budget_seconds': 500}, 'timings', 100), 500)
            with self.assertRaisesRegex(RuntimeError, 'cleanup'):
                study.worker_budget(plan, {}, 'timings', -620)
        with patch.dict(os.environ, {'SLURM_JOB_END_TIME': '1100'}, clear=True), \
                patch.object(study.time, 'monotonic', return_value=160), patch.object(study.time, 'time', return_value=500):
            self.assertEqual(study.worker_budget(plan, {}, 'timings', 100), 480)
        for value in ('', '1100.0', '-1100', 'nan', 'inf', '9' * 400, '500', '600'):
            with self.subTest(value=value), patch.dict(os.environ, {'SLURM_JOB_END_TIME': value}, clear=True), \
                    patch.object(study.time, 'monotonic', return_value=160), patch.object(study.time, 'time', return_value=500):
                with self.assertRaises((ValueError, RuntimeError)):
                    study.worker_budget(plan, {}, 'timings', 100)

    def test_worker_exhaustion_rejects_before_gpu_queries(self):
        with patch.object(study, 'verify', return_value=self.plan), \
                patch.object(study.study_workloads, 'load_suite', return_value={}), \
                patch.dict(os.environ, {'SLURM_JOB_END_TIME': '500'}, clear=True), \
                patch.object(study.time, 'time', return_value=500), patch.object(study, 'allocation') as allocation:
            with self.assertRaisesRegex(RuntimeError, 'future timestamp'):
                study.worker(argparse.Namespace(run_dir=str(self.root), stage='timings'))
            allocation.assert_not_called()
        self.assertEqual(study.read_json(self.root / 'state.json')['attempts'], [])

    def test_worker_runner_budget_subtracts_admission_setup_time(self):
        self.plan['wall_time_limits'] = {'timings': '00:15:00', 'profiles': '00:30:00'}
        self.save_plan()
        clock = [100]
        def admit(*args):
            clock[0] = 160
            return {}
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}, clear=True), \
                patch.object(study.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(study, 'verify', return_value=self.plan), \
                patch.object(study.study_workloads, 'load_suite', return_value={}), \
                patch.object(study, 'allocation', return_value={}), patch.object(study, 'checked_builds'), \
                patch.object(study, 'admit', side_effect=admit), patch.object(study, 'Runner') as runner, \
                patch.object(study, 'hardware_snapshot', return_value={}), patch.object(study, 'report'), \
                patch.object(study, 'timings', return_value={'status': 'timings_complete', 'cases': {}}):
            study.worker(argparse.Namespace(run_dir=str(self.root), stage='timings'))
        self.assertEqual(runner.call_args.args[1:], (720, 300))
        attempt = study.read_json(self.root / 'state.json')['attempts'][-1]
        self.assertEqual((attempt['worker_budget_seconds'], attempt['cleanup_reserve_seconds'], attempt['requested_wall_time']),
                         (720, 120, '00:15:00'))

    def finish_timings(self):
        output = self.root / 'attempts' / 'timings-fixture'
        output.mkdir(parents=True)
        summary = {'schema_version': 1, 'status': 'timings_complete', 'cases': {}}
        study.write_json(output / 'summary.json', summary)
        (output / 'stdout.log').write_text('accepted fixed work\n')
        attempt = {'id': 'timings-fixture', 'stage': 'timings', 'status': 'timings_complete',
                   'directory': str(output), 'summary': str(output / 'summary.json'),
                   'summary_sha256': study.sha(output / 'summary.json'),
                   'artifact_sha256': {str(p.relative_to(self.root)): study.sha(p)
                                       for p in output.iterdir()}}
        self.state['attempts'].append(attempt)
        study.write_json(self.root / 'state.json', self.state)
        return output

    def test_frozen_executor_suite_and_plan_are_checked(self):
        study.verify(self.root)
        (self.root / 'executor' / 'pace_runtime.sh').write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'executor changed'):
            study.verify(self.root)
        self.plan['site'] = 'changed'
        study.write_json(self.root / 'plan.json', self.plan)
        with self.assertRaisesRegex(RuntimeError, 'plan identity'):
            study.load_plan(self.root)

    def test_modified_timing_log_prevents_resume(self):
        output = self.finish_timings()
        study.verify(self.root)
        (output / 'stdout.log').write_text('modified measurement')
        with self.assertRaisesRegex(RuntimeError, 'completed measurement artifact'):
            study.verify(self.root)

    def test_identical_source_report_and_status_label_control_without_claiming_optimization(self):
        self.plan['sources'] = {member: {'source_fingerprint': 'a' * 64, 'source_status': 'dirty'}
                                for member in study.MEMBERS}
        self.plan.pop('plan_sha256')
        self.plan['plan_sha256'] = study.digest(self.plan)
        study.write_json(self.root / 'plan.json', self.plan)
        self.finish_timings()
        report = study.report(self.root)
        self.assertEqual(report['comparison_kind'], 'identical-source-control')
        text = (self.root / 'report.md').read_text()
        self.assertIn('identical-source workflow/repeatability control', text)
        self.assertIn('paired variation does not measure an optimization change', text)
        self.assertIn('Profile state: not_started', text)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            study.status(self.root)
        self.assertIn('identical-source workflow/repeatability control', output.getvalue())

    def test_control_report_waits_for_timings_and_preserves_independent_pair_scope(self):
        self.plan['sources'] = {member: {'source_fingerprint': 'a' * 64}
                                for member in study.MEMBERS}
        self.save_plan()
        pending = study.report(self.root)
        self.assertEqual(pending['repeatability']['status'], 'timings_pending')
        self.assertEqual(pending['repeatability']['cases'], {})
        output = self.finish_timings()
        summary = study.read_json(output / 'summary.json')
        summary['cases']['fixture'] = {'statistics': study.sample_statistics(
            [10., 10., 10.], [9., 10., 11.]),
            'observations': [{'accepted_steps': 5}] * 6}
        study.write_json(output / 'summary.json', summary)
        self.state['attempts'][-1]['summary_sha256'] = study.sha(output / 'summary.json')
        study.write_json(self.root / 'state.json', self.state)
        report = study.report(self.root)
        values = report['repeatability']['cases']['fixture']
        self.assertEqual(values['independent_pairs'], 3)
        self.assertEqual(values['paired_median_offset_percent'], 0.)
        self.assertAlmostEqual(values['median_absolute_paired_offset_percent'], 10.)
        self.assertAlmostEqual(values['maximum_absolute_paired_offset_percent'], 10.)
        self.assertEqual(values['baseline_coefficient_of_variation_percent'], 0.)
        self.assertEqual(values['candidate_coefficient_of_variation_percent'], 10.)
        self.assertIn('not a confidence interval', report['repeatability']['scope'])
        self.assertIn('Observed control repeatability', (self.root / 'report.md').read_text())
        # Derived reporting does not rewrite the admitted observations.
        self.assertEqual(study.read_json(output / 'summary.json'), summary)

    def test_control_repeatability_rejects_insufficient_or_nonfinite_samples(self):
        for baseline, candidate in (([1.], [1.]), ([1., 2., 3.], [1., 2.]),
                                    ([1., 2., 3.], [1., float('nan'), 3.]),
                                    ([1., 2., 3.], [1., float('inf'), 3.]),
                                    ([1., 0., 3.], [1., 2., 3.])):
            with self.subTest(baseline=baseline, candidate=candidate):
                timing = {'cases': {'fixture': {'statistics': {
                    'baseline_seconds': baseline, 'candidate_seconds': candidate}}}}
                with self.assertRaisesRegex(RuntimeError, 'independent positive process pairs'):
                    study.control_repeatability(timing)

    def test_report_distinguishes_run_only_legacy_and_unrecorded_validation(self):
        self.assertEqual(study.report(self.root)['validation_execution'],
                         dict.fromkeys(study.MEMBERS, 'not-recorded'))
        self.plan['builds'] = {'baseline': {'run_only_validation': False},
                               'candidate': {'run_only_validation': True}}
        self.save_plan()
        result = study.report(self.root)
        self.assertEqual(result['validation_execution'],
                         {'baseline': 'legacy-driver', 'candidate': 'run-only'})
        self.assertIn('may configure/build in the validation allocation',
                      (self.root / 'report.md').read_text())

    def test_dirty_source_comparison_uses_content_fingerprints_and_profile_attempt_state(self):
        self.plan['sources'] = {member: {'source_fingerprint': fingerprint,
                                        'source_revision': 'same-head', 'source_status': 'dirty'}
                                for member, fingerprint in zip(study.MEMBERS, ('a' * 64, 'b' * 64))}
        self.plan.pop('plan_sha256')
        self.plan['plan_sha256'] = study.digest(self.plan)
        study.write_json(self.root / 'plan.json', self.plan)
        self.finish_timings()
        self.state['attempts'].insert(0, {'stage': 'profiles', 'status': 'profiles_partial',
                                         'id': 'retained-profile', 'directory': str(self.root)})
        study.write_json(self.root / 'state.json', self.state)
        report = study.report(self.root)
        self.assertEqual(report['comparison_kind'], 'baseline-candidate')
        self.assertNotIn('repeatability', report)
        text = (self.root / 'report.md').read_text()
        self.assertIn('Comparison: baseline/candidate source comparison.', text)
        self.assertIn('Profile state: profiles_partial', text)
        self.assertNotIn('identical-source workflow', text)

    def test_retained_plan_without_fingerprints_reports_unknown_identity(self):
        report = study.report(self.root)
        self.assertEqual(report['comparison_kind'], 'identity-unavailable')
        self.assertIn('source fingerprints unavailable', (self.root / 'report.md').read_text())

    def test_report_lists_only_enabled_cases_and_individual_backend_timing_profile_scope(self):
        suite = {'cases': [
            {'id': 'cpu', 'adapter': 'dg', 'parameters': {'discretization': 'dgsem', 'device': 'cpu'}},
            {'id': 'tensor', 'adapter': 'dg', 'parameters': {'discretization': 'dgsem', 'device': 'cuda',
                                                          'dgsem_residual': 'fused-dmma'}},
            {'id': 'hdg', 'adapter': 'dg', 'parameters': {'discretization': 'hdg', 'device': 'cuda',
                                                       'engine': 'ordinary'}},
            {'id': 'disabled-probe', 'adapter': 'dgsem', 'enabled': False, 'parameters': {'n': 4}}]}
        study.write_json(self.root / 'suite.json', suite)
        self.plan['suite_sha256'] = study.sha(self.root / 'suite.json')
        self.save_plan()
        output = self.finish_timings()
        statistics = study.sample_statistics([1., 1., 1.], [1., 1., 1.])
        summary = {'cases': {'cpu': {'statistics': statistics, 'scope': 'CPU solver elapsed time'},
                             'hdg': {'statistics': statistics}}}
        study.write_json(output / 'summary.json', summary)
        self.state['attempts'][-1]['summary_sha256'] = study.sha(output / 'summary.json')
        partial = self.root / 'profile-attempt'
        partial.mkdir()
        study.write_json(partial / 'partial.json', {'cases': {
            'tensor': {'baseline': {'status': 'complete'}, 'candidate': {'status': 'partial'}}}})
        self.state['attempts'].append({'stage': 'profiles', 'status': 'profiles_partial',
                                       'directory': str(partial)})
        study.write_json(self.root / 'state.json', self.state)
        before = (output / 'summary.json').read_bytes()
        with patch.object(study.study_workloads, 'load_suite', side_effect=AssertionError('report must read retained cases')):
            result = study.report(self.root)
        coverage = result['coverage']
        self.assertEqual(coverage['enabled_case_count'], 3)
        cases = {case['case_id']: case for case in coverage['cases']}
        self.assertEqual(set(cases), {'cpu', 'tensor', 'hdg'})
        self.assertEqual((cases['cpu']['timing_status'], cases['cpu']['profile_status']),
                         ('measured', 'not_applicable'))
        self.assertEqual(cases['cpu']['member_profile_status'], dict.fromkeys(study.MEMBERS, 'not_applicable'))
        self.assertEqual(cases['cpu']['timing_scope'], 'CPU solver elapsed time')
        self.assertEqual(cases['cpu']['timing_scope_source'], 'observation')
        self.assertEqual((cases['tensor']['timing_status'], cases['tensor']['profile_status']), ('pending', 'partial'))
        self.assertEqual(cases['tensor']['dgsem_residual'], 'fused-dmma')
        self.assertEqual(cases['hdg']['timing_scope'], 'whole-process wall time including startup/setup/shutdown')
        self.assertEqual(cases['hdg']['timing_scope_source'], 'adapter-contract')
        self.assertEqual(before, (output / 'summary.json').read_bytes())
        text = (self.root / 'report.md').read_text()
        self.assertIn('3 cases; selections do not imply all-backend coverage', text)
        self.assertNotIn('disabled-probe', text)
        self.assertIn('CPU hardware counters are not collected', text)

    def test_report_keeps_historical_parameter_free_cases_and_pending_profiles_visible(self):
        suite = {'cases': [{'id': 'historical', 'adapter': 'hdg'}]}
        study.write_json(self.root / 'suite.json', suite)
        self.plan['suite_sha256'] = study.sha(self.root / 'suite.json')
        self.save_plan()
        case = study.report(self.root)['coverage']['cases'][0]
        self.assertEqual(case['case_id'], 'historical')
        self.assertEqual((case['timing_status'], case['profile_status']), ('pending', 'pending'))
        self.assertEqual(case['engine'], 'not-recorded')
        self.assertEqual(case['timing_scope'], 'accepted-step median per process')
        study.write_json(self.root / 'suite.json', {'cases': []})
        with self.assertRaisesRegex(RuntimeError, 'frozen workload suite changed'):
            study.report(self.root)

    def test_profile_resume_preserves_timings_and_rejects_duplicate_job(self):
        output = self.finish_timings()
        before = (output / 'summary.json').read_bytes()
        self.state['jobs'].append({'stage': 'profiles', 'job_id': '123'})
        self.state['attempts'].append({'stage': 'profiles', 'status': 'profiles_partial',
                                       'id': 'failed-attempt', 'error': 'native import failed'})
        study.write_json(self.root / 'state.json', self.state)
        args = argparse.Namespace(run_dir=str(self.root), stage='profiles')
        with patch.object(study, 'checked_builds'), patch.object(study, 'recorded_orphans'), patch.object(study.subprocess, 'run',
                    return_value=subprocess.CompletedProcess([], 0, 'RUNNING\n', '')):
            with self.assertRaisesRegex(RuntimeError, 'active Slurm job'):
                study.resume(args)
        with patch.object(study, 'checked_builds'), patch.object(study, 'recorded_orphans'), patch.object(study.subprocess, 'run',
                    return_value=subprocess.CompletedProcess([], 0, '', '')), \
                    patch.object(study, 'submit_stage') as submit:
            study.resume(args)
            submit.assert_called_once_with(self.root, 'profiles')
        self.assertEqual((output / 'summary.json').read_bytes(), before)
        self.assertEqual(study.read_json(self.root / 'state.json')['attempts'][-1]['error'], 'native import failed')

    def test_complete_stage_cannot_be_replaced(self):
        self.finish_timings()
        with self.assertRaisesRegex(RuntimeError, 'already complete'):
            study.resume(argparse.Namespace(run_dir=str(self.root), stage='timings'))

    def test_collection_excludes_native_reports_and_csv(self):
        self.finish_timings()
        (self.root / 'capture.nsys-rep').write_text('large native data')
        (self.root / 'capture.ncu-repz').write_text('native')
        (self.root / 'capture.sqlite').write_text('database')
        (self.root / 'raw.csv').write_text('full export')
        destination = Path(self.temp.name) / 'compact'
        study.collect(argparse.Namespace(run_dir=str(self.root), destination=str(destination)))
        self.assertTrue((destination / 'report.json').exists())
        self.assertFalse(any(p.suffix in ('.nsys-rep', '.ncu-repz', '.sqlite', '.csv') for p in destination.iterdir()))
        self.assertFalse(study.read_json(destination / 'collection.json')['native_reports_included'])

    def test_stage_submission_free_queue_dependency_and_frozen_executor(self):
        with patch.object(study.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '456;cluster\n', '')) as run:
            self.assertEqual(study.submit_stage(self.root, 'profiles', ['123']), '456')
        command = run.call_args.args[0]
        for arg in ('--qos=embers', '--account=gts-sbryngelson3', '--partition=gpu-a100',
                    '--ntasks=1', '--dependency=afterok:123', '--gres=gpu:a100:1', '--time=01:00:00'):
            self.assertIn(arg, command)
        self.assertEqual(command[-1], str(self.root / 'executor'))

    def test_stage_submission_uses_frozen_limits_and_rejects_invalid_plan(self):
        self.plan['wall_time_limits'] = {'timings': '00:15:00', 'profiles': '00:30:00'}
        self.save_plan()
        with patch.object(study.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '456\n', '')) as run:
            study.submit_stage(self.root, 'timings', ['123'])
            self.assertIn('--time=00:15:00', run.call_args.args[0])
            study.submit_stage(self.root, 'profiles', ['456'])
            self.assertIn('--time=00:30:00', run.call_args.args[0])
        self.plan['wall_time_limits']['profiles'] = '02:00:00'
        self.save_plan()
        with patch.object(study.subprocess, 'run') as run:
            with self.assertRaises(ValueError): study.submit_stage(self.root, 'profiles')
            run.assert_not_called()

    def test_failed_validation_limit_retry_reuses_jobs_and_retains_error_receipt(self):
        self.plan['builds'] = {member: {'app_build': '/shared/identical-build', 'validation_time': '00:15:00'}
                              for member in study.MEMBERS}
        self.plan['sources'] = {member: {'source_snapshot': str(self.root)} for member in study.MEMBERS}
        existing = {'success': False, 'build_job_id': '100', 'validation_job_id': '101',
                    'validation_time_error': 'scheduler update failed'}
        self.state['builds']['baseline'] = existing
        study.write_json(self.root / 'state.json', self.state)
        with patch.object(study, 'job_state', return_value=('PENDING', '0:0')), \
                patch.object(study, 'recover_orphan', return_value=False), \
                patch.object(study.study_build, 'ensure_validation_time', return_value=existing, create=True) as ensure, \
                patch.object(study.study_build, 'submit_build') as build, patch.object(study, 'submit_stage') as stages:
            with self.assertRaisesRegex(RuntimeError, 'correctness limit not admitted'):
                study.submit_pipeline(self.root, self.plan)
            ensure.assert_called_once_with(self.plan['builds']['baseline'], existing)
            build.assert_not_called()
            stages.assert_not_called()
        state = study.read_json(self.root / 'state.json')
        self.assertEqual(state['build_attempts'][-1]['validation_job_id'], '101')
        self.assertEqual(state['build_attempts'][-1]['validation_time_error'], 'scheduler update failed')
        fixed = {**existing, 'success': True, 'validation_time_error': None}
        with patch.object(study, 'job_state', return_value=('PENDING', '0:0')), \
                patch.object(study, 'recover_orphan', return_value=False), patch.object(study, 'recorded_orphans'), \
                patch.object(study.study_build, 'ensure_validation_time', return_value=fixed, create=True), \
                patch.object(study.study_build, 'submit_build') as build, \
                patch.object(study, 'submit_stage', side_effect=['102', '103']) as stages:
            study.submit_pipeline(self.root, self.plan)
            build.assert_not_called()
            self.assertEqual(stages.call_count, 2)
        state = study.read_json(self.root / 'state.json')
        self.assertEqual(state['builds']['baseline'], state['builds']['candidate'])
        self.assertTrue(state['builds']['baseline']['success'])

    def test_pipeline_deduplicates_identical_build_and_preserves_failed_submission(self):
        self.plan['builds'] = {m: {'app_build': '/shared/identical-build'} for m in study.MEMBERS}
        built = {'success': True, 'build_job_id': '100', 'validation_job_id': '101'}
        with patch.object(study.study_build, 'submit_build', return_value=built) as build, \
                patch.object(study, 'submit_stage', side_effect=['102', '103']) as stages:
            study.submit_pipeline(self.root, self.plan)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(stages.call_args_list[0].args, (self.root, 'timings', ['101']))
        state = study.read_json(self.root / 'state.json')
        self.assertEqual(state['builds']['baseline'], state['builds']['candidate'])
        state['builds'] = {}
        study.write_json(self.root / 'state.json', state)
        partial = {'success': False, 'build_job_id': '200', 'validation_job_id': None,
                   'stderr': 'validation submission failed'}
        with patch.object(study.study_build, 'submit_build', return_value=partial), \
                patch.object(study, 'submit_stage') as stages:
            with self.assertRaisesRegex(RuntimeError, 'submission incomplete'):
                study.submit_pipeline(self.root, self.plan)
            stages.assert_not_called()
        self.assertEqual(study.read_json(self.root / 'state.json')['build_attempts'][-1]['build_job_id'], '200')

    def test_orphan_recovery_requires_recorded_failed_dependency_and_scheduler_identity(self):
        control = (f'JobId=202 JobName=IGRlab-profiles UserId={study.getpass.getuser()}(1000) '
                   f'Account=gts-sbryngelson3 QOS=embers JobState=PENDING '
                   f'Reason=DependencyNeverSatisfied Dependency=afterok:201(failed) '
                   f'Command={self.root}/executor/pace_runtime.sh')
        with patch.object(study.subprocess, 'run', side_effect=[
                subprocess.CompletedProcess([], 0, control, ''),
                subprocess.CompletedProcess([], 0, '', '')]) as run, \
                patch.object(study, 'job_state', return_value=('FAILED', '1:0')):
            self.assertTrue(study.recover_orphan(self.root, '202', 'IGRlab-profiles', ['201'],
                                                self.root / 'executor/pace_runtime.sh'))
        self.assertEqual(run.call_args_list[-1].args[0], ['scancel', '202'])
        self.assertEqual(study.read_json(self.root / 'state.json')['recovery_actions'][-1]['job_id'], '202')
        with patch.object(study.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0,
                          control.replace('Account=gts-sbryngelson3', 'Account=unrelated'), '')) as run:
            with self.assertRaisesRegex(RuntimeError, 'identity/dependency changed'):
                study.recover_orphan(self.root, '202', 'IGRlab-profiles', ['201'],
                                     self.root / 'executor/pace_runtime.sh')
            self.assertEqual(run.call_count, 1)
        for dependency in ('afterok:201(failed):999(unfulfilled)',
                           'afterok:201(failed),afterany:999(unfulfilled)'):
            with self.subTest(dependency=dependency), patch.object(study.subprocess, 'run',
                    return_value=subprocess.CompletedProcess([], 0,
                        control.replace('afterok:201(failed)', dependency), '')) as run:
                with self.assertRaisesRegex(RuntimeError, 'identity/dependency changed'):
                    study.recover_orphan(self.root, '202', 'IGRlab-profiles', ['201'],
                                         self.root / 'executor/pace_runtime.sh')
                self.assertEqual(run.call_count, 1)
        with patch.object(study.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0,
                          control.replace('JobState=PENDING', 'JobState=RUNNING'), '')) as run:
            self.assertFalse(study.recover_orphan(self.root, '202', 'IGRlab-profiles', ['201'],
                                                 self.root / 'executor/pace_runtime.sh'))
            self.assertEqual(run.call_count, 1)

    def test_actual_paired_process_protocol_separates_export_and_warmup(self):
        script = self.root / 'fixture.py'
        script.write_text('''import json, sys
member, exported = sys.argv[1:]
print(json.dumps({'seconds': 10.0 if member == 'baseline' else 9.0,
 'signature': {'work': 6}, 'checksums': ['same-final-state'], 'scope': 'host fixture',
 'host_export': exported == 'yes',
 'profile': {'enabled': False, 'measured_step': -1}}))
''')
        admission = {'binaries': {m: {'fixture': m} for m in study.MEMBERS}}
        self.plan['sources'] = {m: {'source_snapshot': str(self.root)} for m in study.MEMBERS}
        suite = study.read_json(self.root / 'suite.json')
        runner = study.Runner(self.root / 'protocol', 20)
        def command(case, binary, source, workdir, *, profile=False, fields=False):
            return [sys.executable, str(script), str(binary), 'yes' if fields else 'no']
        def parse(case, stdout, stderr, gpu):
            return json.loads(stdout.read_text())
        with patch.object(study.study_workloads, 'command', side_effect=command), \
                patch.object(study.study_workloads, 'environment', return_value={}), \
                patch.object(study.study_workloads, 'parse', side_effect=parse):
            result = study.timings(self.root, self.plan, suite, admission, runner, {})
        case = result['cases']['fixture']
        self.assertEqual(case['statistics']['baseline_seconds'], [10., 10., 10.])
        self.assertEqual(case['statistics']['candidate_seconds'], [9., 9., 9.])
        self.assertEqual([row['member'] for row in case['observations']],
                         ['baseline', 'candidate', 'candidate', 'baseline',
                          'baseline', 'candidate', 'candidate', 'baseline'])
        self.assertEqual(sum(row['warmup'] for row in case['observations']), 2)
        self.assertTrue(all(row['host_export'] for row in case['exported_observations'].values()))
        self.assertFalse(any(row['host_export'] for row in case['observations']))

    def test_worker_rejects_paid_queue_before_gpu_query(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '1', 'SLURM_JOB_GPUS': '3',
                                    'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3', 'SLURM_JOB_QOS': 'inferno',
                                    'SLURM_JOB_PARTITION': 'gpu-a100', 'SLURM_NTASKS': '1'}, clear=True), \
                patch.object(study.study_profiles, 'gpu_identity') as gpu:
            with self.assertRaisesRegex(RuntimeError, 'free, single-rank'):
                study.allocation()
            gpu.assert_not_called()

    def test_allocation_accepts_step_gpu_metadata_without_mapping_visible_index(self):
        environment = {'SLURM_JOB_ID': '1', 'SLURM_STEP_GPUS': '5',
                       'CUDA_VISIBLE_DEVICES': '0', 'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3',
                       'SLURM_JOB_QOS': 'embers', 'SLURM_JOB_PARTITION': 'gpu-a100',
                       'SLURM_NTASKS': '1'}
        identity = {'uuid': 'allocation-exposed-gpu', 'compute_capability': 80}
        for job_gpus in (None, '', '5'):
            values = dict(environment)
            if job_gpus is not None:
                values['SLURM_JOB_GPUS'] = job_gpus
            with self.subTest(job_gpus=job_gpus), patch.dict(os.environ, values, clear=True), \
                    patch.object(study.socket, 'getfqdn', return_value='compute'), \
                    patch.object(study.study_profiles, 'gpu_identity', return_value=identity) as gpu:
                self.assertEqual(study.allocation(), identity)
                gpu.assert_called_once_with()
        uuid = 'GPU-12345678-1234-1234-1234-123456789abc'
        with patch.dict(os.environ, environment, clear=True), \
                patch.object(study.socket, 'getfqdn', return_value='compute'), \
                patch.object(study.study_profiles.subprocess, 'run', return_value=
                    subprocess.CompletedProcess([], 0, f'{uuid}, A100-SXM4-80GB, 8.0\n', '')) as query:
            actual = study.allocation()
            self.assertEqual(actual['uuid'], uuid)
            self.assertEqual(actual['allocation_environment'],
                             {'CUDA_VISIBLE_DEVICES': '0', 'SLURM_STEP_GPUS': '5'})
            self.assertFalse(any(value.startswith('--id') for value in query.call_args.args[0]))
        for changes in ({'SLURM_STEP_GPUS': ''}, {'SLURM_NTASKS': '2'},
                        {'SLURM_JOB_QOS': 'inferno'}, {'SLURM_JOB_PARTITION': 'cpu-small'}):
            with self.subTest(changes=changes), patch.dict(os.environ, {**environment, **changes}, clear=True), \
                    patch.object(study.study_profiles, 'gpu_identity') as gpu:
                with self.assertRaisesRegex(RuntimeError, 'free, single-rank'):
                    study.allocation()
                gpu.assert_not_called()

    def test_runtime_shell_accepts_step_gpu_fallback_before_frozen_verification(self):
        runtime = Path(study.__file__).with_name('pace_runtime.sh')
        environment = {'PATH': os.defpath, 'SLURM_JOB_ID': '1', 'SLURM_STEP_GPUS': '5',
                       'CUDA_VISIBLE_DEVICES': '0', 'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3',
                       'SLURM_JOB_QOS': 'embers', 'SLURM_JOB_PARTITION': 'gpu-a100',
                       'SLURM_NTASKS': '1'}
        # Intercept the first metadata verifier; no modules, executors, or GPU commands run.
        command = ['bash', '-c', 'hostname() { printf "compute\\n"; }; '
                   'python3() { exit 77; }; source "$1" plan timings executor',
                   'fixture', str(runtime)]
        for changes, expected in (({}, 77), ({'SLURM_JOB_GPUS': ''}, 77),
                                  ({'SLURM_JOB_GPUS': '5', 'SLURM_STEP_GPUS': ''}, 77),
                                  ({'SLURM_STEP_GPUS': ''}, 2), ({'SLURM_NTASKS': '2'}, 2),
                                  ({'SLURM_JOB_QOS': 'inferno'}, 2)):
            with self.subTest(changes=changes):
                result = subprocess.run(command, env={**environment, **changes}, text=True,
                                        capture_output=True, timeout=5)
                self.assertEqual(result.returncode, expected, result.stderr)

    def test_runtime_rejects_inherited_mpi_client_context_before_verification(self):
        runtime = Path(study.__file__).with_name('pace_runtime.sh')
        environment = {'PATH': os.defpath, 'SLURM_JOB_ID': '1', 'SLURM_STEP_GPUS': '5',
                       'CUDA_VISIBLE_DEVICES': '0', 'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3',
                       'SLURM_JOB_QOS': 'embers', 'SLURM_JOB_PARTITION': 'gpu-a100',
                       'SLURM_NTASKS': '1'}
        command = ['bash', '-c', 'hostname() { printf "compute\\n"; }; '
                   'python3() { printf "verifier reached\\n"; exit 77; }; '
                   'source "$1" plan timings executor', 'fixture', str(runtime)]
        for variable, value in (('PMIX_NAMESPACE', 'inherited-client'), ('PMIX_RANK', '0'),
                                ('PMI_RANK', '0'), ('PMI_SIZE', '1')):
            with self.subTest(variable=variable):
                result = subprocess.run(command, env={**environment, variable: value},
                                        text=True, capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 2)
                self.assertIn(variable, result.stderr)
                self.assertIn('srun --mpi=none --pty bash -l', result.stderr)
                self.assertNotIn('verifier reached', result.stdout)
                self.assertNotIn(value, result.stderr)
        result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 77)
        self.assertIn('verifier reached', result.stdout)

    def test_correctness_jobs_are_required(self):
        self.state['builds'] = {m: {'validation_job_id': str(i + 1)} for i, m in enumerate(study.MEMBERS)}
        study.write_json(self.root / 'state.json', self.state)
        with patch.object(study.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '1|COMPLETED|0:0\n2|FAILED|1:0\n', '')):
            with self.assertRaisesRegex(RuntimeError, 'correctness jobs'):
                study.checked_builds(self.root)

    def test_admission_reads_commented_cmake_cache_by_line_and_reports_actual_mismatch(self):
        build = self.root / 'app-build'
        build.mkdir()
        cache = build / 'CMakeCache.txt'
        text = ('# This is the CMakeCache file.\n\n'
                '//Choose the type of build, options are: Debug Release\n'
                'CMAKE_BUILD_TYPE:STRING=Release\n\n'
                '//No help, variable specified on the command line.\n'
                'CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=80\n\n'
                '//Compile optional NVTX observability\n'
                'DG_IGR_OBSERVABILITY_USE_NVTX:BOOL=ON\n')
        cache.write_text(text)
        plan = {'spack_root': '/spack',
                'sources': {'candidate': {'source_snapshot': str(Path(study.__file__).resolve().parents[2])}},
                'builds': {member: {'app_build': str(build), 'spack_env': '/environment',
                                     'mfem_build': str(self.root / 'mfem-build'),
                                     'mfem_install': str(self.root / 'mfem-install')}
                           for member in study.MEMBERS}}
        with patch.object(study.subprocess, 'check_output', return_value='/selected/package\n'):
            # Passing the real cache parser reaches the next independent gate;
            # this fixture deliberately has no installed MFEM or binaries.
            with self.assertRaisesRegex(RuntimeError, 'installed MFEM library is missing'):
                study.admit(self.root, plan, {})
            for key, previous, changed in (('CMAKE_BUILD_TYPE', 'Release', 'Debug'),
                                             ('CMAKE_CUDA_ARCHITECTURES', '80', '120'),
                                             ('DG_IGR_OBSERVABILITY_USE_NVTX', 'ON', 'OFF')):
                with self.subTest(key=key):
                    cache.write_text(text.replace('=' + previous + '\n', '=' + changed + '\n'))
                    with self.assertRaisesRegex(RuntimeError, key + '=' + repr(changed)):
                        study.admit(self.root, plan, {})
            cache.write_text(text.replace('CMAKE_BUILD_TYPE:STRING=Release\n', ''))
            with self.assertRaisesRegex(RuntimeError, 'CMAKE_BUILD_TYPE=None'):
                study.admit(self.root, plan, {})
        self.assertFalse((self.root / 'admission.json').exists())

    def test_independent_pairs_do_not_count_accepted_steps_as_repetitions(self):
        stats = study.sample_statistics([10., 12., 11.], [9., 10., 10.])
        self.assertEqual(len(stats['paired_percent_less_time']), 3)
        self.assertAlmostEqual(stats['paired_median_percent_less_time'], 10.)
        with self.assertRaisesRegex(RuntimeError, 'three independent'):
            study.sample_statistics([10.], [9.])

    def test_lock_and_symlink_guards(self):
        link = Path(self.temp.name) / 'link'
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlinked'):
            study.absolute(str(link / 'run'))
        with study.locked(self.root):
            with self.assertRaisesRegex(RuntimeError, 'another command'):
                with study.locked(self.root):
                    self.fail('duplicate writer acquired lock')

    def test_runner_exact_argv_timeout_receipt_and_exclusive_attempt(self):
        runner = study.Runner(self.root, 15, command_timeout=0.05)
        args = 'literal $HOME `touch should-not-exist`\nnext line'
        path = runner.run('success', [sys.executable, '-c', 'import sys; print(sys.argv[1])', args], timeout=5)
        self.assertEqual((path / 'stdout.log').read_text(), args + '\n')
        with self.assertRaises(FileExistsError):
            runner.run('success', ['true'])
        with self.assertRaisesRegex(RuntimeError, 'exit=124'):
            runner.run('timeout', [sys.executable, '-c', 'import time; time.sleep(20)'])
        self.assertEqual(study.read_json(self.root / 'timeout/result.json')['returncode'], 124)


if __name__ == '__main__':
    unittest.main()
