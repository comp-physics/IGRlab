#!/usr/bin/env python3
"""Mixed CPU/CUDA orchestration fixtures; no builds, jobs, or applications."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gpu_study as study


class CpuStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'run'
        self.root.mkdir()
        self.source = Path(self.temp.name) / 'source'
        (self.source / 'cmake/sites').mkdir(parents=True)
        (self.source / 'cmake/sites/pace.json').write_text('{"site":"pace"}')
        self.plan = {'sources': {member: {'source_snapshot': str(self.source)} for member in study.MEMBERS},
                     'builds': {member: {'app_build': '/cache/cuda', 'configuration_key': 'cuda'}
                                for member in study.MEMBERS},
                     'cpu_builds': {member: {'app_build': '/cache/cpu', 'configuration_key': 'cpu',
                                             'profile': 'cpu'} for member in study.MEMBERS}}
        self.state = {'builds': {}, 'build_attempts': [], 'jobs': [], 'attempts': []}
        study.write_json(self.root / 'state.json', self.state)

    def test_create_adds_cpu_plans_only_for_enabled_cpu_hdg_and_freezes_helper(self):
        suite = copy.deepcopy(study.study_workloads.load_suite(
            Path(study.__file__).parent / 'suites/pace-full-backends.json'))
        for case in suite['cases']:
            case['enabled'] = case['id'] == 'hdg-cpu-p3'
        path = Path(self.temp.name) / 'suite.json'
        path.write_text(json.dumps(suite))
        destination = Path(self.temp.name) / 'new'
        args = argparse.Namespace(run_dir=str(destination), source=str(self.source), site_root=str(self.root),
                                  cache_root=None, spack_root=str(self.root / 'spack'), suite=str(path),
                                  cases=None, baseline='HEAD', candidate='HEAD')
        def planned(identity, output, site, spack, regex, **kwargs):
            return {'profile': kwargs.get('profile', 'cuda-sm80'), 'app_build': '/not-created',
                    'correctness_regex': regex}
        with patch.object(study.study_build, 'freeze', return_value={'source_snapshot': str(self.source)}), \
                patch.object(study, 'cached_source', side_effect=lambda identity, cache: identity), \
                patch.object(study.study_build, 'build_plan', side_effect=planned) as build:
            root, plan = study.create(args)
        self.assertEqual(build.call_count, 4)
        self.assertEqual(set(plan['cpu_builds']), set(study.MEMBERS))
        for member in study.MEMBERS:
            self.assertEqual(plan['cpu_builds'][member]['profile'], 'cpu')
            self.assertEqual(plan['cpu_builds'][member]['correctness_regex'], study.CPU_CORRECTNESS_REGEX)
        self.assertIn('study_admit.py', plan['executor_sha256'])
        self.assertTrue((root / 'executor/study_admit.py').is_file())
        suite['cases'][0]['enabled'] = True
        for case in suite['cases']:
            if case['id'] == 'hdg-cpu-p3': case['enabled'] = False
        self.assertFalse(study.needs_cpu_build(suite))

    def test_pipeline_deduplicates_per_profile_and_depends_on_both_correctness_jobs(self):
        def submit(plan):
            cpu = plan.get('profile') == 'cpu'
            return {'success': True, 'returncode': 0, 'build_job_id': '200' if cpu else '100',
                    'validation_job_id': '201' if cpu else '101'}
        with patch.object(study.study_build, 'submit_build', side_effect=submit) as build, \
                patch.object(study, 'recorded_orphans'), patch.object(study, 'submit_stage', return_value='300') as stage:
            study.submit_pipeline(self.root, self.plan)
        self.assertEqual(build.call_count, 2)
        self.assertEqual(stage.call_args_list[0].args, (self.root, 'timings', ['101', '201']))
        self.assertEqual(stage.call_args_list[1].args, (self.root, 'profiles', ['300']))
        state = study.read_json(self.root / 'state.json')
        self.assertEqual(state['cpu_builds']['baseline'], state['cpu_builds']['candidate'])
        self.assertEqual(len(state['build_attempts']), 4)

    def test_partial_cpu_submission_keeps_job_ids_and_never_submits_timings(self):
        def submit(plan):
            return {'success': plan.get('profile') != 'cpu', 'returncode': 1 if plan.get('profile') == 'cpu' else 0,
                    'build_job_id': '200' if plan.get('profile') == 'cpu' else '100',
                    'validation_job_id': None if plan.get('profile') == 'cpu' else '101'}
        with patch.object(study.study_build, 'submit_build', side_effect=submit), \
                patch.object(study, 'submit_stage') as stage:
            with self.assertRaisesRegex(RuntimeError, 'submission incomplete'):
                study.submit_pipeline(self.root, self.plan)
            stage.assert_not_called()
        state = study.read_json(self.root / 'state.json')
        self.assertEqual(state['cpu_builds']['baseline']['build_job_id'], '200')
        self.assertFalse(state['cpu_builds']['baseline']['success'])

    def test_failed_or_missing_cpu_correctness_prevents_measurement(self):
        self.state['builds'] = {m: {'validation_job_id': '101'} for m in study.MEMBERS}
        self.state['cpu_builds'] = {m: {'validation_job_id': '201'} for m in study.MEMBERS}
        study.write_json(self.root / 'state.json', self.state)
        with patch.object(study, 'load_plan', return_value=self.plan), \
                patch.object(study.subprocess, 'run', return_value=subprocess.CompletedProcess(
                    [], 0, '101|COMPLETED|0:0\n201|FAILED|8:0\n', '')):
            with self.assertRaisesRegex(RuntimeError, 'correctness jobs'):
                study.checked_builds(self.root)
        del self.state['cpu_builds']['candidate']
        study.write_json(self.root / 'state.json', self.state)
        with patch.object(study, 'load_plan', return_value=self.plan), patch.object(study.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'CPU correctness jobs'):
                study.checked_builds(self.root)
            run.assert_not_called()

    def test_cpu_dispatch_requires_cpu_artifact_and_preserves_application_arguments(self):
        case = {'adapter': 'dg', 'binary': 'app', 'parameters': {'device': 'cpu', 'discretization': 'hdg'}}
        runner = SimpleNamespace(output=self.root, run=lambda *args, **kwargs: self.root)
        admission = {'binaries': {'baseline': {'app': '/cuda/app'}}}
        with patch.object(study.study_workloads, 'command') as command:
            with self.assertRaisesRegex(RuntimeError, 'independently admitted CPU artifact'):
                study.observe(runner, self.root, self.plan, admission, case, 'baseline', 'sample', {})
            command.assert_not_called()
        cpu = {'binaries': {'app': '/cpu/app'}}
        admission['cpu_artifacts'] = {'baseline': cpu}
        runner.run = lambda *args, **kwargs: self.root
        observed = {}
        def run(label, argv, **kwargs):
            observed.update(argv=argv, env=kwargs['env'])
            return self.root
        runner.run = run
        with patch.object(study.study_workloads, 'command', return_value=['/cpu/app', '--device', 'cpu', '-dt', '1e-5']), \
                patch.object(study.study_workloads, 'environment', return_value={'PHYSICS': 'same'}), \
                patch.object(study.study_workloads, 'parse', return_value={'seconds': 2.0}), \
                patch.object(study.study_admit, 'runtime_command', side_effect=lambda admitted, argv, env:
                             (['/cpu-mpi/bin/mpiexec', '-n', '1', *argv], {'LD_LIBRARY_PATH': '/cpu/libs'})), \
                patch.object(study, 'read_json', return_value={'process_wall_seconds': 2.0}), \
                patch.object(study, 'write_json'):
            study.observe(runner, self.root, self.plan, admission, case, 'baseline', 'sample', {})
        self.assertEqual(observed['argv'], ['/cpu-mpi/bin/mpiexec', '-n', '1', '/cpu/app', '--device', 'cpu', '-dt', '1e-5'])
        self.assertEqual(observed['env']['PHYSICS'], 'same')
        self.assertEqual(observed['env']['LD_LIBRARY_PATH'], '/cpu/libs')

    def test_legacy_retry_selection_cannot_invent_cpu_hdg_artifacts(self):
        suite = {'cases': [{'adapter': 'dg', 'parameters': {'device': 'cpu', 'discretization': 'hdg'}}]}
        with self.assertRaisesRegex(ValueError, 'separately admitted CPU-profile'):
            study.validate_cuda_profile_cases(suite)
        study.validate_cuda_profile_cases(suite, cpu_available=True)


if __name__ == '__main__':
    unittest.main()
