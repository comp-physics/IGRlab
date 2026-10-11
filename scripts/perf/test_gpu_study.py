#!/usr/bin/env python3
"""Host-only study protocol tests; no application builds or GPU commands."""
import argparse
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
                    '--ntasks=1', '--dependency=afterok:123', '--gres=gpu:a100:1'):
            self.assertIn(arg, command)
        self.assertEqual(command[-1], str(self.root / 'executor'))

    def test_pipeline_deduplicates_identical_build_and_preserves_failed_submission(self):
        self.plan['builds'] = {m: {'app_build': '/shared/identical-build'} for m in study.MEMBERS}
        built = {'success': True, 'build_job_id': '100', 'validation_job_id': '101'}
        with patch.object(study.study_build, 'submit_build', return_value=built) as build, \
                patch.object(study, 'submit_stage', side_effect=['102', '103']) as stages:
            study.submit_pipeline(self.root, self.plan)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(stages.call_args_list[0].args, (self.root, 'timings', ['101', '101']))
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

    def test_correctness_jobs_are_required(self):
        self.state['builds'] = {m: {'validation_job_id': str(i + 1)} for i, m in enumerate(study.MEMBERS)}
        study.write_json(self.root / 'state.json', self.state)
        with patch.object(study.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '1|COMPLETED|0:0\n2|FAILED|1:0\n', '')):
            with self.assertRaisesRegex(RuntimeError, 'correctness jobs'):
                study.checked_builds(self.root)

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
