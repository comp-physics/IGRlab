#!/usr/bin/env python3
"""Delta study contracts with intercepted tools; no native applications/jobs."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import subprocess
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gpu_study as study
import study_import
import study_site
import test_study_import as import_fixtures
import test_study_build as build_fixtures


class DeltaImport(unittest.TestCase):
    def setUp(self):
        # Reuse the genuine-semantics fixture, never a fabricated native result.
        self.fixture = import_fixtures.StudyImportTests('runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        f.values['--site'] = 'delta'
        f.write(f.source / 'cmake/sites/delta.json', '{}')
        request = json.loads(f.request.read_text())
        request['argv'] = ['delta' if arg == 'pace' else arg for arg in request['argv']]
        f.write(f.request, json.dumps(request))
        f.saved['policy']['site'] = 'delta'
        f.saved['paths']['spack_env'] = f.saved['paths']['spack_env'].replace('/site/pace/', '/site/delta/')
        f.write(f.prebuilt, json.dumps(f.saved))
        f.result['policy'] = deepcopy(f.saved['policy'])
        f.result['prebuilt_receipt_sha256'] = study_import.sha(f.prebuilt)
        f.result['argv'] = ['delta' if arg == 'pace' else arg for arg in f.result['argv']]
        for snapshot in (f.result['allocation_before'], f.result['allocation_after']):
            snapshot['authority'] = 'live Delta allocation'
            snapshot['hostname'] = 'compute-1.delta.ncsa.illinois.edu'
            snapshot['scheduler']['fields'].update(Account='bdiy-delta-gpu', QOS='normal', Partition='gpuA100x4')
        f.publish_result()
        original_query = f.plan_query
        def delta_query(argv, **kwargs):
            result = original_query(argv, **kwargs)
            return subprocess.CompletedProcess(result.args, result.returncode,
                result.stdout.replace(b'/site/pace/', b'/site/delta/'), result.stderr)
        self.addCleanup(patch.stopall)
        patch.object(study_import.study_build, '_run', side_effect=delta_query).start()

    def plan(self):
        return self.fixture.plan(reuse_validated_artifacts=True)

    def test_delta_request_receipt_and_source_policy_bind_without_historical_queries(self):
        with patch.object(study_import, 'run', side_effect=AssertionError('historical query')):
            plan = self.plan()
            self.assertEqual(plan['site'], 'delta')
            self.assertIn('/site/delta/spack-envs/cuda-sm80', plan['spack_env'])
            observed = study_import.completed_records(plan, study_import.receipt(plan), full=True)
        self.assertEqual(observed['result']['policy']['site'], 'delta')
        self.assertEqual(observed['artifact_sha256'][str(self.fixture.binary)], study_import.sha(self.fixture.binary))

    def test_delta_receipt_refuses_wrong_site_account_partition_authority_and_missing_pass(self):
        original = deepcopy(self.fixture.result)
        for changed in ('account', 'partition', 'authority', 'outcome', 'policy'):
            with self.subTest(changed=changed):
                self.fixture.result = deepcopy(original)
                r = self.fixture.result
                if changed == 'account':
                    r['allocation_before']['scheduler']['fields']['Account'] = 'gts-sbryngelson3'
                elif changed == 'partition':
                    r['allocation_before']['scheduler']['fields']['Partition'] = 'gpu-a100'
                elif changed == 'authority':
                    r['allocation_before']['authority'] = 'live PACE allocation'
                elif changed == 'outcome':
                    r['test_outcomes'] = r['test_outcomes'][:-1]
                else:
                    r['policy']['site'] = 'pace'
                self.fixture.publish_result()
                with self.assertRaises(ValueError):
                    self.plan()

    def test_delta_import_records_site_and_frozen_runtime_and_owned_preparation(self):
        f = self.fixture
        suite = {'name': 'delta fixture', 'correctness_regex': '^accept-', 'cases': [
            {'id': 'case', 'binary': f.binary.name, 'adapter': 'dg', 'parameters': {'device': 'cuda'}}]}
        args = SimpleNamespace(prepare_only=True, suite='/suite', artifacts='/manifest',
                               run_dir=str(f.root / 'delta-run'), cpu_only=False,
                               reuse_validated_artifacts=True, timings_time='02:00:00', profiles_time='01:00:00')
        plan = self.plan()
        plans = {'builds': {'baseline': plan, 'candidate': deepcopy(plan)}}
        receipts = {'builds': {name: study_import.receipt(value) for name, value in plans['builds'].items()}}
        with patch.object(study.study_workloads, 'load_suite', return_value=suite), \
             patch.object(study.study_import, 'inspect_manifest', return_value=(plans, receipts, {}, {})), \
             patch.object(study, 'validate_cuda_profile_cases'), \
             patch.object(study, 'delta_allocation', return_value={'job_id': '900', 'end_epoch': time.time() + 7200}), \
             patch.object(study.study_import, 'selected_binaries', return_value={}), \
             patch.dict(os.environ, {}, clear=True):
            root, result = study.import_existing(args)
        self.assertEqual(result['site'], 'delta')
        self.assertEqual(result['wall_time_limits']['timings'], '02:00:00')
        self.assertEqual(result['preparation_allocation']['job_id'], '900')
        self.assertIn('delta_runtime.sh', result['executor_sha256'])
        self.assertIn('study_site.py', result['executor_sha256'])
        self.assertEqual((root / 'executor/delta_runtime.sh').read_bytes(),
                         Path(study.__file__).with_name('delta_runtime.sh').read_bytes())


class DeltaAllocation(unittest.TestCase):
    def setUp(self):
        self.end = time.time() + 7200
        self.environment = {**study_site.environment('delta'), 'SLURM_JOB_ID': '900',
                            'SLURM_JOB_GPUS': '0', 'CUDA_VISIBLE_DEVICES': '0'}
        self.fields = {'JobId': '900', 'JobState': 'RUNNING', 'BatchFlag': '0',
            'UserId': f'{pwd.getpwuid(os.getuid()).pw_name}({os.getuid()})',
            **study_site.scheduler_fields('delta'), 'QOS': 'normal', 'NumNodes': '1',
            'NumTasks': '1', 'NumCPUs': '8', 'NodeList': 'gpua022',
            'AllocTRES': 'cpu=8,mem=48G,node=1,gres/gpu=1,gres/gpu:nvidia_a100=1',
            'EndTime': datetime.fromtimestamp(self.end, timezone.utc).isoformat().removesuffix('+00:00')}
        self.gpu = {'uuid': 'GPU-11111111-2222-3333-4444-555555555555', 'name': 'NVIDIA A100',
                    'compute_capability': 80}

    def admit(self, *, fields=None, env=None, node='gpua022'):
        fields = self.fields if fields is None else fields
        answer = ' '.join(key + '=' + value for key, value in fields.items())
        with patch.dict(os.environ, self.environment if env is None else env, clear=True), \
             patch.object(study.socket, 'getfqdn', return_value='gpua022.delta.ncsa.illinois.edu'), \
             patch.object(study.subprocess, 'run', side_effect=[
                 subprocess.CompletedProcess([], 0, answer, ''),
                 subprocess.CompletedProcess([], 0, node + '\n', '')]), \
             patch.object(study.study_profiles, 'gpu_identity', return_value=self.gpu):
            return study.delta_allocation(interactive=True)

    def test_real_delta_identity_and_end_are_retained_without_pace_qos(self):
        result = self.admit()
        self.assertEqual(result['gpu'], self.gpu)
        self.assertEqual(result['scheduler']['Account'], 'bdiy-delta-gpu')
        self.assertAlmostEqual(result['end_epoch'], self.end, places=5)

    def test_wrong_owner_account_partition_gpu_count_node_or_cleanup_reject(self):
        changes = [('UserId', 'somebody(123)'), ('Account', 'bdiy-delta-cpu'),
                   ('Partition', 'gpuA100x8'), ('JobState', 'PENDING'), ('BatchFlag', '1'),
                   ('AllocTRES', 'cpu=8,mem=48G,node=1,gres/gpu=2'),
                   ('EndTime', datetime.fromtimestamp(time.time() + 30, timezone.utc).isoformat().removesuffix('+00:00'))]
        for key, value in changes:
            with self.subTest(key=key):
                with self.assertRaises(RuntimeError):
                    self.admit(fields={**self.fields, key: value})
        with self.assertRaises(RuntimeError):
            self.admit(node='gpua099')
        with self.assertRaises(RuntimeError):
            self.admit(env={**self.environment, 'PMIX_NAMESPACE': 'not-mpi-none'})

    def test_delta_stage_submission_uses_actual_account_runtime_and_one_gpu(self):
        with patch.object(study, 'load_plan', return_value={'site': 'delta', 'run_dir': '/unused'}), \
             patch.object(study, 'stage_wall_time', return_value='02:00:00'), \
             patch.object(study, 'read_json', return_value={'jobs': []}), \
             patch.object(study, 'write_json'), \
             patch.object(Path, 'mkdir'), \
             patch.object(study.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '900\n', '')) as run:
            for stage in ('timings', 'profiles'):
                with self.subTest(stage=stage):
                    self.assertEqual(study.submit_stage(Path('/unused'), stage), '900')
                    argv = run.call_args.args[0]
                    self.assertIn('--account=bdiy-delta-gpu', argv)
                    self.assertIn('--partition=gpuA100x4', argv)
                    self.assertIn('--gpus-per-node=1', argv)
                    self.assertIn('/unused/executor/delta_runtime.sh', argv)
                    self.assertEqual('--constraint=perf,nvperf' in argv, stage == 'profiles')
                    self.assertFalse(any(arg.startswith('--qos=') or arg.startswith('--exclusive') for arg in argv))

    def test_worker_budget_respects_actual_end_and_cleanup(self):
        plan = {'site': 'delta', 'wall_time_limits': {'timings': '02:00:00', 'profiles': '02:00:00'}}
        with patch.object(study.time, 'monotonic', return_value=100), patch.object(study.time, 'time', return_value=1000), \
             patch.dict(os.environ, {}, clear=True):
            self.assertEqual(study.worker_budget(plan, {'timings_budget_seconds': 7200}, 'timings', 100,
                                                 end_epoch=2000), 880)
        self.assertEqual(study.stage_wall_time(plan, 'timings'), '02:00:00')
        with self.assertRaises(ValueError):
            study.stage_wall_time({**plan, 'site': 'pace'}, 'timings')


class DeltaBuildPlan(unittest.TestCase):
    def test_public_delta_build_plan_and_submission_keep_site_policy(self):
        fixture = build_fixtures.StudyBuildTests('runTest')
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.write('build.sh', build_fixtures.DRIVER.replace('PACE', 'Delta').replace('/site/pace/', '/site/delta/'), executable=True)
        fixture.write('cmake/sites/delta.json', '{"site":"delta"}\n')
        fixture.write('scripts/build/delta.sh', '# Delta fixture\n')
        fixture.git(fixture.source, 'add', '.')
        fixture.git(fixture.source, 'commit', '-qm', 'Delta fixture')
        identity = fixture.freeze()
        plan = study.study_build.build_plan(identity, fixture.root / 'outputs', fixture.root / 'site',
            fixture.root / 'spack', '^accept-', site='delta')
        self.assertEqual(plan['site'], 'delta')
        self.assertEqual(plan['argv'][plan['argv'].index('--site') + 1], 'delta')
        self.assertIn('/site/delta/spack-envs/cuda-sm80', plan['spack_env'])
        self.assertIn('cmake/sites/delta.json', plan['dependency_configuration']['policy_sha256'])
        self.assertNotIn('cmake/sites/pace.json', plan['dependency_configuration']['policy_sha256'])
        with patch.dict(os.environ, {}, clear=True):
            receipt = study.study_build.submit_build(plan)
        self.assertTrue(receipt['success'])
        self.assertEqual(receipt['build_job_id'], '987123')
        self.assertEqual(receipt['validation_job_id'], '567890')


if __name__ == '__main__':
    unittest.main()
