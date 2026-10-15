#!/usr/bin/env python3
"""Missing-only CPU continuation fixtures; no applications or scheduler calls."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gpu_study as g
import study_cpu_continuation as continuation


class CpuContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.old = self.base / 'old'
        self.old.mkdir()
        with g.locked(self.old):
            pass  # Existing worker lock is coordination metadata, not a timing artifact.
        self.new = self.base / 'new'
        suite = g.study_workloads.load_suite(Path(g.__file__).parent / 'suites/pace-igr-cpu-parity.json')
        self.suite = {**suite, 'cases': [deepcopy(suite['cases'][0])]}
        self.case = self.suite['cases'][0]
        self.id = self.case['id']
        g.write_json(self.old / 'suite.json', self.suite)
        self.source = self.base / 'source'
        self.source.mkdir()
        self.plan = {'schema_version': 1, 'run_dir': str(self.old), 'execution_profile': 'cpu',
            'wall_time_limits': {'timings': '01:00:00', 'profiles': '01:00:00'},
            'suite_sha256': g.sha(self.old / 'suite.json'), 'sources': {}, 'builds': {},
            'import_origin': {'reuse_validated_artifacts': True, 'artifact_sha256': {}}}
        for member in g.MEMBERS:
            self.plan['sources'][member] = {'source_snapshot': str(self.source)}
            self.plan['builds'][member] = {key: str(self.base / 'build' / key) for key in
                ('app_build', 'app_install', 'mfem_source', 'mfem_build', 'mfem_install', 'spack_env')}
            self.plan['builds'][member]['profile'] = 'cpu'
        g.freeze_executor(self.old)
        self.plan['executor_sha256'] = {name: g.sha(self.old / 'executor' / name)
                                       for name in g.PRE_CPU_CONTINUATION_EXECUTOR_FILES}
        self.plan['plan_sha256'] = g.digest(self.plan)
        g.write_json(self.old / 'plan.json', self.plan)
        self.admission = {'plan_sha256': self.plan['plan_sha256'], 'cpu_artifacts': {
            member: {'profile': 'cpu', 'runtime': {'launcher': '/cpu-mpi/bin/orterun'},
                     'binaries': {self.case['binary']: '/cpu/app'}} for member in g.MEMBERS}}
        g.write_json(self.old / 'admission.json', self.admission)
        self.allocation = {'job_id': '111', 'hostname': 'old-node', 'end_epoch': 2000}
        self.output = self.old / 'attempts/timings-111-original'
        self.output.mkdir(parents=True)
        self.attempt = {'id': 'timings-111-original', 'stage': 'timings', 'status': 'failed',
            'finished_unix_seconds': 1000, 'job_id': '111', 'gpu': None,
            'directory': str(self.output), 'cpu_allocation': self.allocation}
        self.state = {'schema_version': 1, 'builds': {'baseline': {'genuine': True},
            'candidate': {'genuine': True}}, 'build_attempts': [], 'jobs': [], 'attempts': [self.attempt]}
        self.items = continuation.protocol(g, self.suite)
        self.env = {'DG_IGR_PROFILE_NVTX': '0', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1', 'SLURM_JOB_ID': '111', 'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3',
            'SLURM_JOB_QOS': 'embers', 'SLURM_JOB_PARTITION': 'cpu-small', 'SLURM_NTASKS': '1'}
        self.commands = []
        self.patches = [patch.object(g.study_build, 'verify_source'),
            patch.object(g, 'checked_builds'),
            patch.object(g.study_workloads, 'command', side_effect=self.command),
            patch.object(g.study_workloads, 'environment', return_value={
                'DG_IGR_PROFILE_NVTX': '0', 'DG_IGR_DGSEM_PRODUCTION_STRICT': '1'}),
            patch.object(g.study_workloads, 'parse', side_effect=self.parse),
            patch.dict(os.environ, {}, clear=True)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        for item in self.items[:6]:
            self.process(self.output, item)
        self.process(self.output, self.items[6], returncode=124)
        g.write_json(self.old / 'state.json', self.state)

    @staticmethod
    def command(case, binary, source, directory, *, fields=False, **kwargs):
        return [str(binary), '--device', 'cpu', '--order', str(case['parameters']['order']),
                '--fields', str(bool(fields)).lower()]

    @staticmethod
    def parse(case, stdout, stderr, gpu, **kwargs):
        if gpu is not None:
            raise AssertionError('CPU parsing must not invent a GPU')
        return json.loads(stdout.read_text())

    def process(self, output, item, *, allocation=None, returncode=0, signature=None):
        allocation = allocation or self.allocation
        directory = output / item['label']
        directory.mkdir(parents=True)
        observation = {'seconds': 10.0 if item['member'] == 'baseline' else 9.0,
            'signature': signature or {'work': {'accepted': 10, 'rejected': 0}, 'policy': 'strict'},
            'scope': 'application solver elapsed', 'gpu_uuid': None, 'checksums': None,
            'igr_evidence': {'calls': 31, 'converged_calls': 31, 'total_iters': 18352}}
        (directory / 'stdout.log').write_text(json.dumps(observation))
        (directory / 'stderr.log').write_text('')
        environment = {**self.env, 'SLURM_JOB_ID': allocation['job_id']}
        result = {'argv': ['/cpu-mpi/bin/orterun', '-n', '1', *self.command(
            self.case, Path('/cpu/app'), self.source, directory, fields=item['fields'])],
            'cwd': str(directory), 'environment': environment, 'started_unix_seconds': 900.0,
            'status': 'complete' if returncode == 0 else 'failed', 'returncode': returncode,
            'process_wall_seconds': 12.5, 'stdout_sha256': g.sha(directory / 'stdout.log'),
            'stderr_sha256': g.sha(directory / 'stderr.log')}
        command = {key: result[key] for key in ('argv', 'cwd', 'environment', 'started_unix_seconds')}
        g.write_json(directory / 'command.json', {**command, 'status': 'running'})
        g.write_json(directory / 'result.json', result)
        if returncode == 0:
            g.write_json(directory / 'observation.json', {**observation, 'receipt': str(directory / 'result.json')})
        return directory

    def prepare(self, *, queue='', rc=0, old=None, new=None):
        args = argparse.Namespace(from_run=str(old or self.old), run_dir=str(new or self.new),
                                  prepare_only=True, reuse_validated_artifacts=True, timings_time=None)
        with patch.object(continuation.subprocess, 'run', return_value=subprocess.CompletedProcess(
                [], rc, queue, '')) as query:
            result = continuation.prepare(g, args)
        self.assertEqual(query.call_args.args[0][:3], ['squeue', '-h', '-u'])
        return result

    def test_protocol_preserves_alternation_and_only_four_missing_labels(self):
        self.assertEqual([item['label'].split('/')[-1] for item in self.items], [
            'warmup-0-baseline', 'warmup-0-candidate', 'pair-0-candidate', 'pair-0-baseline',
            'pair-1-baseline', 'pair-1-candidate', 'pair-2-candidate', 'pair-2-baseline',
            'fields-baseline', 'fields-candidate'])

    def test_record_reparses_exact_controls_with_explicit_unrecorded_keys(self):
        value = continuation.record(g, self.plan, self.case, self.items[0],
                                    self.output / self.items[0]['label'], self.admission, self.allocation)
        self.assertIsNone(value['origin']['host'])
        self.assertIn('unavailable', value['origin']['hardware_availability'])
        self.assertEqual(value['unrecorded_controls'], {'DG_IGR_DGSEM_PRODUCTION_STRICT': '1'})
        self.assertEqual(value['observation']['receipt'], str(self.output / self.items[0]['label'] / 'result.json'))

    def test_record_rejects_log_command_environment_and_observation_drift(self):
        directory = self.output / self.items[0]['label']
        for filename, change in (
            ('stdout.log', lambda value: value + ' '),
            ('command.json', lambda value: {**value, 'argv': ['wrong']}),
            ('result.json', lambda value: {**value, 'environment': {**value['environment'], 'OMP_NUM_THREADS': '2'}}),
            ('observation.json', lambda value: {**value, 'signature': {'different': True}})):
            with self.subTest(filename=filename):
                path = directory / filename
                before = path.read_bytes()
                if filename.endswith('.json'):
                    g.write_json(path, change(g.read_json(path)))
                else:
                    path.write_text(change(path.read_text()))
                with self.assertRaises(ValueError):
                    continuation.record(g, self.plan, self.case, self.items[0], directory,
                                        self.admission, self.allocation)
                path.write_bytes(before)

    def test_prepare_seals_failure_and_retains_six_without_mutating_old(self):
        before = {str(path): g.sha(path) for path in self.old.rglob('*') if path.is_file()}
        root, plan = self.prepare(queue='222|RUNNING\n')
        context = plan['cpu_timing_continuation']
        self.assertEqual(len(context['retained']), 6)
        self.assertEqual(context['missing_labels'], [item['label'] for item in self.items[6:]])
        failed = str(self.output / self.items[6]['label'] / 'result.json')
        self.assertIn(failed, context['artifact_sha256'])
        self.assertEqual(g.read_json(failed)['returncode'], 124)
        self.assertEqual(before, {str(path): g.sha(path) for path in self.old.rglob('*') if path.is_file()})
        self.assertEqual(g.read_json(root / 'state.json')['attempts'], [])
        self.assertEqual(g.read_json(root / 'state.json')['jobs'], [])
        self.assertIn('study_cpu_continuation.py', plan['executor_sha256'])
        self.assertEqual(g.sha(root / 'suite.json'), self.plan['suite_sha256'])
        self.assertEqual(g.verify(root)['plan_sha256'], plan['plan_sha256'])

    def test_prepare_rejects_active_malformed_or_unavailable_live_queue(self):
        for queue, rc in [('111|RUNNING\n', 0), ('not-a-queue-row', 0), ('', 1)]:
            with self.subTest(queue=queue, rc=rc), self.assertRaises(ValueError):
                self.prepare(queue=queue, rc=rc)
            self.assertFalse(self.new.exists())

    def test_compute_preparation_requires_actual_cpu_admission_and_never_submits(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '222'}), \
                patch.object(g, 'cpu_allocation', return_value={'job_id': '222', 'profile': 'cpu'}) as admitted, \
                patch.object(g, 'submit_stage') as submit:
            root, _ = self.prepare(queue='222|RUNNING\n')
        admitted.assert_called_once_with()
        submit.assert_not_called()
        self.assertEqual(g.read_json(root / 'state.json')['attempts'], [])

    def test_compute_preparation_rejects_gpu_or_mpi_context_before_evidence_reads(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '222', 'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3',
                'SLURM_JOB_QOS': 'embers', 'SLURM_NTASKS': '1', 'SLURM_CPUS_PER_TASK': '8',
                'SLURM_JOB_PARTITION': 'gpu-a100'}), \
                patch.object(g, 'verify') as verify:
            with self.assertRaisesRegex(RuntimeError, 'cpu-small'):
                self.prepare()
            verify.assert_not_called()
        for key in ('PMIX_NAMESPACE', 'PMIX_RANK', 'PMI_RANK', 'PMI_SIZE'):
            with self.subTest(key=key), patch.dict(os.environ, {'SLURM_JOB_ID': '222', key: '0'}), \
                    patch.object(g, 'cpu_allocation') as admitted, patch.object(g, 'verify') as verify:
                with self.assertRaisesRegex(ValueError, 'MPI/PMIx'):
                    self.prepare()
                admitted.assert_not_called()
                verify.assert_not_called()
                self.assertFalse(self.new.exists())

    def test_compute_preparation_cannot_submit_nested_jobs(self):
        args = argparse.Namespace(from_run=str(self.old), run_dir=str(self.new),
                                  prepare_only=False, reuse_validated_artifacts=True)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '222'}), \
                patch.object(g, 'cpu_allocation') as admitted, patch.object(g, 'submit_stage') as submit:
            with self.assertRaisesRegex(ValueError, 'prepare-only'):
                continuation.prepare(g, args)
        admitted.assert_not_called()
        submit.assert_not_called()
        self.assertFalse(self.new.exists())

    def test_prepare_rejects_valid_but_changed_work_before_publication(self):
        directory = self.output / self.items[5]['label']
        payload = g.read_json(directory / 'observation.json')
        payload['signature'] = {'work': {'accepted': 9, 'rejected': 0}, 'policy': 'strict'}
        g.write_json(directory / 'stdout.log', {key: value for key, value in payload.items() if key != 'receipt'})
        g.write_json(directory / 'observation.json', payload)
        receipt = g.read_json(directory / 'result.json')
        receipt['stdout_sha256'] = g.sha(directory / 'stdout.log')
        g.write_json(directory / 'result.json', receipt)
        with self.assertRaisesRegex(ValueError, 'policy/work differs'):
            self.prepare()
        self.assertFalse(self.new.exists())

    def test_sealed_original_failure_cannot_change_after_preparation(self):
        root, _ = self.prepare()
        path = self.output / self.items[6]['label'] / 'result.json'
        g.write_json(path, {**g.read_json(path), 'returncode': 0})
        with self.assertRaisesRegex(RuntimeError, 'retained CPU timing evidence changed'):
            g.verify(root)

    def execution(self, fail_label=None, changed_label=None, field_failure=False):
        root, plan = self.prepare()
        output = root / 'attempts/timings-222-new'
        output.mkdir(parents=True)
        allocation = {'job_id': '222', 'hostname': 'new-node', 'end_epoch': 3000}
        state = g.read_json(root / 'state.json')
        state['attempts'].append({'id': 'timings-222-new', 'stage': 'timings', 'status': 'running',
            'directory': str(output), 'job_id': '222', 'gpu': None, 'cpu_allocation': allocation,
            'cpu_host': {'hostname': 'new-node'}})
        g.write_json(root / 'state.json', state)
        g.write_json(root / 'admission.json', {**self.admission, 'plan_sha256': plan['plan_sha256']})
        observed = []
        def observe(runner, root, plan, admission, case, member, label, gpu, *, fields=False):
            observed.append(label)
            if label == fail_label:
                item = next(item for item in self.items if item['label'] == label)
                self.process(output, item, allocation=allocation, returncode=124)
                raise RuntimeError('bounded process timeout')
            item = next(item for item in self.items if item['label'] == label)
            signature = {'changed': True} if label == changed_label else None
            directory = self.process(output, item, allocation=allocation, signature=signature)
            return g.read_json(directory / 'observation.json'), directory
        gate = RuntimeError('field mismatch') if field_failure else {'status': 'passed', 'max_abs': 0.0}
        with patch.object(g, 'observe', side_effect=observe), \
                patch.object(g.study_workloads, 'compare_fields', side_effect=gate if field_failure else None,
                             return_value=gate), \
                patch.object(continuation, 'host', return_value={'hostname': 'new-node'}):
            result = continuation.execute(g, root, plan, self.suite, self.admission, SimpleNamespace(output=output))
        return root, plan, output, observed, result

    def test_execute_runs_only_missing_pair_then_both_fields(self):
        root, plan, output, observed, result = self.execution()
        self.assertEqual(observed, [item['label'] for item in self.items[6:]])
        case = result['cases'][self.id]
        self.assertEqual(result['status'], 'timings_complete')
        self.assertIsNone(result['gpu'])
        self.assertEqual(len(case['observations']), 8)
        self.assertEqual(case['statistics']['baseline_seconds'], [10.0] * 3)
        self.assertEqual(case['statistics']['candidate_seconds'], [9.0] * 3)
        self.assertEqual(case['whole_process_wall_statistics']['baseline_seconds'], [12.5] * 3)
        self.assertEqual(case['whole_process_wall_statistics']['candidate_seconds'], [12.5] * 3)
        self.assertIn('startup', case['whole_process_wall_statistics']['timer_scope'])
        self.assertTrue(all(value['process_wall_seconds'] == 12.5 for value in case['observations']))
        self.assertEqual([pair['members']['baseline']['job_id'] for pair in case['allocation_pairs']],
                         ['111', '111', '222'])
        self.assertIn('not a homogeneous', case['statistics']['statistical_scope'])
        self.assertEqual(case['fields']['status'], 'passed')
        self.assertEqual(set(case['exported_observations']), set(g.MEMBERS))
        self.assertEqual(g.read_json(self.output / self.items[6]['label'] / 'result.json')['returncode'], 124)

    def test_timeout_preserves_new_success_without_promoting_old_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'timeout'):
            self.execution(fail_label=self.items[7]['label'])
        output = self.new / 'attempts/timings-222-new'
        partial = g.read_json(output / 'partial.json')
        self.assertEqual(len(partial['records']), 7)
        self.assertEqual(partial['missing_labels'], [item['label'] for item in self.items[7:]])
        self.assertIn('timeout', partial['error'])
        self.assertFalse((output / 'summary.json').exists())

    def test_changed_new_work_is_not_admitted_to_partial(self):
        with self.assertRaisesRegex(ValueError, 'policy/work differs'):
            self.execution(changed_label=self.items[6]['label'])
        partial = g.read_json(self.new / 'attempts/timings-222-new/partial.json')
        self.assertEqual(len(partial['records']), 6)

    def test_second_continuation_keeps_first_continuation_success(self):
        with self.assertRaisesRegex(RuntimeError, 'timeout'):
            self.execution(fail_label=self.items[7]['label'])
        state = g.read_json(self.new / 'state.json')
        state['attempts'][0].update(status='failed', finished_unix_seconds=2500)
        g.write_json(self.new / 'state.json', state)
        destination = self.base / 'third'
        root, plan = self.prepare(old=self.new, new=destination)
        context = plan['cpu_timing_continuation']
        self.assertEqual(len(context['retained']), 7)
        self.assertEqual(context['missing_labels'], [item['label'] for item in self.items[7:]])
        retained = context['retained'][self.items[6]['label']]
        self.assertEqual(retained['origin']['job_id'], '222')
        self.assertEqual(retained['origin']['hostname'], 'new-node')
        self.assertEqual(retained['origin']['host'], {'hostname': 'new-node'})
        self.assertIn(str(self.output / self.items[6]['label'] / 'result.json'), context['artifact_sha256'])
        self.assertIn(str(self.new / 'attempts/timings-222-new' / self.items[7]['label'] / 'result.json'),
                      context['artifact_sha256'])
        self.assertEqual(g.read_json(root / 'state.json')['attempts'], [])

    def test_field_failure_cannot_complete_cohort(self):
        with self.assertRaisesRegex(RuntimeError, 'field mismatch'):
            self.execution(field_failure=True)
        partial = g.read_json(self.new / 'attempts/timings-222-new/partial.json')
        self.assertEqual(partial['status'], 'timings_partial')
        self.assertFalse((self.new / 'attempts/timings-222-new/summary.json').exists())

    def test_existing_attempt_cannot_resume_or_rerun_successes_in_place(self):
        root, plan = self.prepare()
        state = g.read_json(root / 'state.json')
        state['attempts'] = [{'stage': 'timings', 'status': 'failed'}]
        g.write_json(root / 'state.json', state)
        args = argparse.Namespace(run_dir=str(root), stage='timings')
        with patch.object(g, 'verify', return_value=plan), patch.object(g, 'worker_budget') as budget:
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                g.worker(args)
            budget.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'cannot resume in place'):
                g.resume(args)

    def test_prepared_fresh_continuation_can_submit_only_its_timing_job(self):
        root, _ = self.prepare()
        with patch.object(g, 'recorded_orphans'), patch.object(g, 'submit_stage', return_value='222') as submit:
            g.resume(argparse.Namespace(run_dir=str(root), stage='timings'))
        submit.assert_called_once_with(root, 'timings')
        self.assertEqual(g.read_json(root / 'state.json')['attempts'], [])

    def test_cli_is_explicit_and_forwards_no_new_scientific_parameters(self):
        with patch.object(continuation, 'prepare') as prepared:
            g.main(['continue-timings', '--from-run', '/old', '--run-dir', '/new',
                    '--prepare-only', '--reuse-validated-artifacts'])
        args = prepared.call_args.args[1]
        self.assertTrue(args.prepare_only)
        self.assertTrue(args.reuse_validated_artifacts)
        self.assertEqual(args.from_run, '/old')


if __name__ == '__main__':
    unittest.main()
