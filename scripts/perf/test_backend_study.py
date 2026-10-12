#!/usr/bin/env python3
"""Host-only backend protocol checks; no application or profiler execution."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gpu_study as study


class BackendStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runner = SimpleNamespace(output=self.root / 'attempt')
        self.runner.output.mkdir()
        self.plan = {'sources': {m: {'source_snapshot': str(self.root)} for m in study.MEMBERS}}

    def cases(self):
        result = []
        for name, device, residual in [('cpu', 'cpu', 'unfused'), ('simt', 'cuda', 'fused-simt'),
                                       ('tensor', 'cuda', 'fused-dmma')]:
            result.append({'id': name, 'adapter': 'dg', 'binary': 'app', 'parameters': {
                'device': device, 'engine': 'ordinary', 'discretization': 'dgsem',
                'dgsem_residual': residual, 'order': 3, 'mesh': 'same.mesh',
                'dt': 1e-5, 'final_time': 1e-4, 'atol': 5e-9, 'rtol': 5e-8}})
        return result

    def test_reused_timing_gpu_must_match_before_any_profile_capture(self):
        expected = {'uuid': 'GPU-00112233-4455-6677-8899-aabbccddeeff',
                    'name': 'NVIDIA A100-SXM4-80GB', 'compute_capability': 80}
        self.plan['timing_reuse'] = {'gpu': expected}
        study.admit_reused_timing_gpu(self.plan, dict(expected))
        for key, wrong in (('uuid', 'GPU-10112233-4455-6677-8899-aabbccddeeff'),
                           ('name', 'NVIDIA A100-PCIE-40GB'), ('compute_capability', 70)):
            with self.subTest(key=key), patch.object(study.study_profiles, 'capture') as capture:
                with self.assertRaisesRegex(RuntimeError, 'original timing GPU UUID/model'):
                    study.profiles(self.root, self.plan, {'cases': []}, {}, self.runner,
                                   {**expected, key: wrong})
                capture.assert_not_called()

    def test_runner_receipt_retains_pinned_igr_controls_including_empty_overrides(self):
        import sys
        environment = {'IGRLAB_IGR_DENSE_INVERSE': '', 'IGRLAB_IGR_CHEB_BASE_OPERATOR': '0',
                       'DG_IGR_IGR_CHEB_ORDER': '3'}
        runner = study.Runner(self.runner.output, 10)
        directory = runner.run('igr-policy-receipt', [sys.executable, '-c', 'pass'], env=environment)
        receipt = study.read_json(directory / 'result.json')
        self.assertEqual(receipt['environment'], environment)

    def test_cpu_case_never_calls_cuda_profiler_and_is_explicitly_not_applicable(self):
        summary = self.root / 'timings.json'
        study.write_json(summary, {'cases': {}})
        study.write_json(self.root / 'state.json', {'attempts': [
            {'stage': 'timings', 'status': 'timings_complete', 'summary': str(summary)}]})
        with patch.object(study.study_profiles, 'capture') as capture:
            result = study.profiles(self.root, self.plan, {'cases': [self.cases()[0]]}, {}, self.runner, {})
        capture.assert_not_called()
        self.assertEqual(result['status'], 'analysis_complete')
        for entry in result['cases']['cpu'].values():
            self.assertEqual(entry['status'], 'not_applicable')
            self.assertIn('CPU hardware counters not collected', entry['scope'])

    def test_hdg_process_timing_comes_from_runner_receipt(self):
        case = {'id': 'hdg-cpu', 'adapter': 'dg', 'binary': 'app',
                'parameters': {'discretization': 'hdg', 'device': 'cpu'}}
        def run(label, command, **kwargs):
            directory = self.runner.output / label
            directory.mkdir()
            (directory / 'stdout.log').write_text('application output has no elapsed field')
            (directory / 'stderr.log').write_text('')
            study.write_json(directory / 'result.json', {'returncode': 0, 'process_wall_seconds': 7.25})
            return directory
        self.runner.run = run
        admission = {'binaries': {'baseline': {'app': '/saved/cuda-app'}},
                     'cpu_artifacts': {'baseline': {'binaries': {'app': '/saved/cpu-app'}}}}
        with patch.object(study.study_workloads, 'parse', return_value={'seconds': 7.25}) as parse, \
                patch.object(study.study_admit, 'runtime_command', side_effect=lambda cpu, argv, env:
                             (['/selected/cpu/mpiexec', '-n', '1', *argv], env)) as launch:
            observation, _ = study.observe(self.runner, self.root, self.plan, admission, case,
                                           'baseline', 'hdg', {})
        self.assertEqual(parse.call_args.kwargs, {'wall_seconds': 7.25})
        self.assertEqual(launch.call_args.args[1][0], '/saved/cpu-app')
        self.assertEqual(observation['seconds'], 7.25)

    def test_profiled_hdg_receipt_is_only_admission_metadata(self):
        case = {'id': 'hdg-cuda', 'adapter': 'dg', 'binary': 'app',
                'parameters': {'discretization': 'hdg', 'device': 'cuda'}}
        summary = self.root / 'timings.json'
        study.write_json(summary, {'cases': {'hdg-cuda': {'observations': [{'signature': {'work': 10}}]}}})
        study.write_json(self.root / 'state.json', {'attempts': [
            {'stage': 'timings', 'status': 'timings_complete', 'summary': str(summary)}]})
        admission = {'binaries': {m: {'app': '/saved/app'} for m in study.MEMBERS}}
        def capture(case, argv, runner, directory, gpu, *, environment):
            receipt = directory / 'result.json'
            study.write_json(receipt, {'returncode': 0, 'process_wall_seconds': 99.})
            return {'status': 'complete', 'application_logs': [{
                'stdout': str(directory / 'stdout.log'), 'stderr': str(directory / 'stderr.log'),
                'receipt': str(receipt)}]}
        with patch.object(study.study_profiles, 'capture', side_effect=capture), \
                patch.object(study.study_workloads, 'parse', return_value={'signature': {'work': 10}, 'seconds': 99.}) as parse:
            result = study.profiles(self.root, self.plan, {'cases': [case]}, admission, self.runner, {})
        self.assertEqual([c.kwargs for c in parse.call_args_list], [{'wall_seconds': 99.}] * 2)
        self.assertNotIn('statistics', result['cases']['hdg-cuda']['baseline'])

    def run_timings(self, cases, failure=False):
        def observe(runner, root, plan, admission, case, member, label, gpu, *, fields=False):
            return {'signature': {'work': case.get('fixture_work', 10)}, 'seconds': 1., 'scope': 'fixture'}, self.root / label
        def fields(case, left, right, source):
            if failure and left.name == 'fields-baseline' and left.parent.name == 'cpu' and right.parent.name == 'tensor':
                raise ValueError('physical tolerance failed')
            return {'passed': True}
        with patch.object(study, 'observe', side_effect=observe), \
                patch.object(study.study_workloads, 'compare_fields', side_effect=fields) as compare:
            result = study.timings(self.root, self.plan, {'cases': cases, 'warmups': 1, 'repetitions': 3},
                                   {}, self.runner, {})
        return result, compare

    def test_all_gpu_backends_compare_to_cpu_for_each_source_without_cross_discretization_mix(self):
        cases = self.cases()
        other = {**cases[-1], 'id': 'other-dg', 'parameters': {
            **cases[-1]['parameters'], 'discretization': 'dg', 'dgsem_residual': 'unfused'}}
        result, _ = self.run_timings([*cases, other])
        actual = {(r['reference_case'], r['case'], r['member']) for r in result['backend_comparisons']}
        self.assertEqual(actual, {('cpu', c, m) for c in ('simt', 'tensor') for m in study.MEMBERS})

    def test_cross_backend_physical_mismatch_rejects_timings_completion(self):
        with self.assertRaisesRegex(ValueError, 'physical tolerance failed'):
            self.run_timings(self.cases(), failure=True)

    def test_equal_fields_do_not_admit_different_accepted_work(self):
        cases = self.cases()
        cases[-1]['fixture_work'] = 11
        with self.assertRaisesRegex(RuntimeError, 'accepted work/horizon differs'):
            self.run_timings(cases)

    def test_ambiguous_cpu_reference_is_not_silently_omitted(self):
        cases = self.cases()
        cases.append({**cases[0], 'id': 'cpu-duplicate'})
        with self.assertRaisesRegex(RuntimeError, 'ambiguous CPU reference'):
            self.run_timings(cases)

    def test_cpu_active_igr_references_compare_sigma_with_independent_iteration_counts(self):
        suite = study.study_workloads.load_suite(
            Path(__file__).parent / 'suites/pace-igr-cpu-parity.json')
        cases = study.selected_cases(suite)

        def observe(runner, root, plan, admission, case, member, label, gpu, *, fields=False):
            cpu = case['parameters']['device'] == 'cpu'
            signature = {'parameters': case['parameters'],
                         'work': {'steps': 10, 'accepted_attempts': 10, 'rejected_attempts': 0,
                                  'rhs_evaluations': 30}, 'final_time': 1e-4,
                         'igr': {'policy': 'same-fixed-cg-chebyshev', 'production_calls': 30,
                                 'production_converged_calls': 30}}
            return {'signature': signature, 'seconds': 1., 'scope': 'fixture',
                    'igr_evidence': {'linear_counts': {'total_iters': 60 if cpu else 61}}}, self.root / label

        def fields(case, left, right, source):
            self.assertEqual(case['parameters']['regularization'], 'igr')
            self.assertEqual((case['parameters']['rtol'], case['parameters']['atol']), (5e-8, 5e-9))
            return {'passed': True, 'nontrivial_sigma': True,
                    'arrays': {name: {'passed': True} for name in (*study.study_workloads.FIELDS, 'sigma')}}

        with patch.object(study, 'observe', side_effect=observe), \
                patch.object(study.study_workloads, 'compare_fields', side_effect=fields) as compare:
            result = study.timings(self.root, self.plan, suite, {}, self.runner, {})
        expected = {(f'{d}-igr-cpu-p3-n6', f'{d}-igr-cuda-p3-n6', m)
                    for d in ('dg', 'dgsem') for m in study.MEMBERS}
        actual = {(r['reference_case'], r['case'], r['member']) for r in result['backend_comparisons']}
        self.assertEqual(actual, expected)
        self.assertEqual(compare.call_count, 8)  # Four source pairs and four CPU/CUDA comparisons.
        for comparison in result['backend_comparisons']:
            self.assertTrue(comparison['accepted_work_equal'])
            self.assertTrue(comparison['physical_fields']['nontrivial_sigma'])
        for case in cases:
            observation = result['cases'][case['id']]['exported_observations']['baseline']
            self.assertEqual(observation['igr_evidence']['linear_counts']['total_iters'],
                             60 if case['parameters']['device'] == 'cpu' else 61)


if __name__ == '__main__':
    unittest.main()
