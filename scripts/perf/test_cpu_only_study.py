#!/usr/bin/env python3
"""CPU-only study contracts; intercepted scheduler/runtime and metadata only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import gpu_study as study


class CpuOnlyStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'run'
        self.root.mkdir()
        self.case = {'id': 'cpu', 'adapter': 'dg', 'binary': 'app',
                     'parameters': {'device': 'cpu', 'discretization': 'dg', 'order': 3}}
        self.suite = {'correctness_regex': '^passed$', 'warmups': 1, 'repetitions': 3,
                      'cases': [self.case]}
        self.plan = {'execution_profile': 'cpu', 'plan_sha256': 'plan',
                     'builds': {member: {'profile': 'cpu'} for member in study.MEMBERS},
                     'sources': {member: {'source_snapshot': str(self.root)} for member in study.MEMBERS}}
        study.write_json(self.root / 'state.json', {'schema_version': 1, 'builds': {},
                         'jobs': [], 'build_attempts': [], 'attempts': []})
        self.env = {'SLURM_JOB_ID': '123', 'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3',
                    'SLURM_JOB_QOS': 'embers', 'SLURM_JOB_PARTITION': 'cpu-small',
                    'SLURM_NTASKS': '1', 'SLURM_CPUS_PER_TASK': '8'}
        self.fields = {'JobId': '123', 'JobState': 'RUNNING',
                       'UserId': f'{study.getpass.getuser()}({os.getuid()})',
                       'Account': 'gts-sbryngelson3', 'QOS': 'embers', 'Partition': 'cpu-small',
                       'NumNodes': '1', 'NumCPUs': '8', 'NumTasks': '1',
                       'AllocTRES': 'cpu=8,mem=48G,node=1,billing=8', 'NodeList': 'compute-1',
                       'EndTime': datetime.fromtimestamp(2000, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S'),
                       'TimeLimit': '01:00:00'}

    def scheduler(self, argv, **kwargs):
        if argv[:4] == ['scontrol', 'show', 'job', '-o']:
            self.assertEqual(kwargs['env']['TZ'], 'UTC')
            self.assertEqual(kwargs['env']['SLURM_TIME_FORMAT'], 'standard')
            return subprocess.CompletedProcess(argv, 0, ' '.join(f'{k}={v}' for k, v in self.fields.items()), '')
        self.assertEqual(argv, ['scontrol', 'show', 'hostnames', 'compute-1'])
        return subprocess.CompletedProcess(argv, 0, 'compute-1\n', '')

    def test_cli_explicit_cpu_only_keeps_gpu_default(self):
        argv = ['import', '--artifacts', '/artifacts', '--suite', '/suite', '--run-dir', '/new']
        with patch.object(study, 'import_existing') as imported:
            study.main(argv)
            self.assertFalse(imported.call_args.args[0].cpu_only)
            study.main(argv + ['--cpu-only', '--reuse-validated-artifacts'])
            self.assertTrue(imported.call_args.args[0].cpu_only)
            self.assertTrue(imported.call_args.args[0].reuse_validated_artifacts)

    def test_cpu_import_freezes_mode_and_submits_only_timing(self):
        source = Path(self.temp.name) / 'source'
        (source / 'cmake/sites').mkdir(parents=True)
        (source / 'cmake/sites/pace.json').write_text('{}')
        build = {'source': {'source_snapshot': str(source)}, 'profile': 'cpu',
                 'site_root': str(self.root), 'spack_root': str(self.root / 'spack'),
                 'import_artifact_sha256': {}, 'test_result': str(self.root / 'tests/result.json')}
        for name in ('app_build', 'app_install', 'mfem_source', 'mfem_build', 'mfem_install', 'spack_env'):
            build[name] = str(self.root / name)
        plans = {'builds': {member: dict(build) for member in study.MEMBERS}}
        receipts = {'builds': {member: {'genuine': True} for member in study.MEMBERS}}
        args = argparse.Namespace(suite='/suite', artifacts='/artifacts',
                                  run_dir=str(Path(self.temp.name) / 'imported'), prepare_only=False,
                                  cpu_only=True, reuse_validated_artifacts=True)
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(study.study_workloads, 'load_suite', return_value=self.suite), \
                patch.object(study.study_import, 'inspect_manifest', return_value=(plans, receipts, {}, {})) as inspected, \
                patch.object(study.study_import, 'selected_binaries', return_value={}), \
                patch.object(study, 'submit_stage', return_value='456') as submit:
            root, plan = study.import_existing(args)
        self.assertEqual(inspected.call_args.kwargs, {'reuse_validated_artifacts': True, 'cpu_only': True})
        self.assertEqual(plan['execution_profile'], 'cpu')
        self.assertEqual(plan['builds'], plans['builds'])
        self.assertNotIn('cpu_builds', plan)
        submit.assert_called_once_with(root, 'timings')
        self.assertEqual(study.load_plan(root)['execution_profile'], 'cpu')

    def test_gpu_workload_rejected_before_import_or_publication(self):
        for case in ({**self.case, 'parameters': {'device': 'cuda'}}, {**self.case, 'adapter': 'hdg'}):
            args = argparse.Namespace(suite='/suite', run_dir='/unused', prepare_only=True, cpu_only=True)
            with patch.object(study.study_workloads, 'load_suite', return_value={'cases': [case]}), \
                    patch.object(study.study_import, 'inspect_manifest') as inspect:
                with self.assertRaisesRegex(ValueError, 'CPU full-application'):
                    study.import_existing(args)
                inspect.assert_not_called()

    def test_submission_requests_cpu_resources_and_never_gpu(self):
        with patch.object(study, 'load_plan', return_value=self.plan), \
                patch.object(study.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '456\n', '')) as submit:
            self.assertEqual(study.submit_stage(self.root, 'timings'), '456')
            with self.assertRaisesRegex(RuntimeError, 'timing jobs only'):
                study.submit_stage(self.root, 'profiles')
        self.assertEqual(submit.call_count, 1)
        argv = submit.call_args.args[0]
        for flag in ('--partition=cpu-small', '--ntasks=1', '--cpus-per-task=8', '--mem=48G', '--time=01:00:00'):
            self.assertIn(flag, argv)
        self.assertFalse(any(flag.startswith('--gres') for flag in argv))
        self.assertEqual(study.read_json(self.root / 'state.json')['jobs'][0]['job_id'], '456')

    def test_live_cpu_allocation_is_owned_bounded_and_has_no_gpu_query(self):
        with patch.dict(os.environ, self.env, clear=True), patch.object(study.socket, 'getfqdn', return_value='compute-1.pace'), \
                patch.object(study.time, 'time', return_value=1000), \
                patch.object(study.subprocess, 'run', side_effect=self.scheduler), \
                patch.object(study.study_profiles, 'gpu_identity') as gpu:
            allocated = study.cpu_allocation()
        gpu.assert_not_called()
        self.assertEqual(allocated['profile'], 'cpu')
        self.assertEqual(allocated['end_epoch'], 2000)
        self.assertNotIn('gpu', allocated)

    def test_live_cpu_scheduler_identity_resources_and_deadline_fail_closed(self):
        for key, value in (('JobState', 'PENDING'), ('UserId', 'other(1)'), ('Account', 'other'),
                           ('Partition', 'gpu-a100'), ('NumTasks', '2'), ('NumCPUs', '16'),
                           ('AllocTRES', 'cpu=8,mem=48G,node=1,gres/gpu=1'),
                           ('AllocTRES', 'cpu=8,mem=32G,node=1'), ('EndTime', 'Unknown'),
                           ('TimeLimit', '02:00:00')):
            old = self.fields[key]
            self.fields[key] = value
            with self.subTest(key=key, value=value), patch.dict(os.environ, self.env, clear=True), \
                    patch.object(study.socket, 'getfqdn', return_value='compute-1.pace'), \
                    patch.object(study.time, 'time', return_value=1000), \
                    patch.object(study.subprocess, 'run', side_effect=self.scheduler):
                with self.assertRaises(RuntimeError): study.cpu_allocation()
            self.fields[key] = old
        with patch.dict(os.environ, self.env, clear=True), \
                patch.object(study.socket, 'getfqdn', return_value='compute-1.pace'), \
                patch.object(study.time, 'time', return_value=1880), \
                patch.object(study.subprocess, 'run', side_effect=self.scheduler):
            with self.assertRaisesRegex(RuntimeError, 'exhausted'): study.cpu_allocation()

    def test_cpu_admission_reuses_real_cpu_helper_without_cuda_queries(self):
        state = study.read_json(self.root / 'state.json')
        state['builds'] = dict.fromkeys(study.MEMBERS, {'genuine': True})
        study.write_json(self.root / 'state.json', state)
        def cpu(build, names, receipt):
            self.assertEqual(names, ['app'])
            return {'binaries': {'app': '/cpu/app'}, 'artifact_sha256': {'/cpu/app': 'hash'},
                    'spack_locations': {'openmpi': '/mpi4'}, 'dependency_bindings': {'mode': 'cpu'}}
        with patch.object(study.study_admit, 'admit_build', side_effect=cpu) as admitted, \
                patch.object(study.subprocess, 'check_output') as query:
            result = study.admit(self.root, self.plan, self.suite)
        query.assert_not_called()
        self.assertEqual(admitted.call_count, 2)
        self.assertEqual(result['execution_profile'], 'cpu')
        self.assertEqual(set(result['cpu_artifacts']), set(study.MEMBERS))
        self.assertNotIn('cuda', result['spack_locations'])

    def test_cpu_worker_uses_null_gpu_and_actual_end_cleanup_budget(self):
        allocation = {'profile': 'cpu', 'end_epoch': 1400, 'job_id': '123'}
        with patch.dict(os.environ, self.env, clear=True), \
                patch.object(study.time, 'time', return_value=1000), \
                patch.object(study, 'verify', return_value=self.plan), \
                patch.object(study.study_workloads, 'load_suite', return_value=self.suite), \
                patch.object(study, 'cpu_allocation', return_value=allocation), \
                patch.object(study, 'allocation') as gpu, patch.object(study, 'checked_builds'), \
                patch.object(study, 'admit', return_value={}), patch.object(study, 'Runner') as runner, \
                patch.object(study, 'timings', return_value={'status': 'timings_complete', 'gpu': None, 'cases': {}}) as timings, \
                patch.object(study, 'report'):
            study.worker(argparse.Namespace(run_dir=str(self.root), stage='timings'))
        gpu.assert_not_called()
        self.assertEqual(timings.call_args.args[-1], None)
        self.assertEqual(runner.call_args.args[1], 280)
        runner.return_value.run.assert_not_called()
        attempt = study.read_json(self.root / 'state.json')['attempts'][0]
        self.assertIsNone(attempt['gpu'])
        self.assertEqual(attempt['cpu_allocation'], allocation)
        self.assertEqual(attempt['hardware_before']['status'], 'not_applicable')
        self.assertEqual(attempt['cleanup_reserve_seconds'], 120)

    def test_cpu_profile_worker_and_resume_reject_before_allocation_or_submission(self):
        args = argparse.Namespace(run_dir=str(self.root), stage='profiles')
        with patch.object(study, 'verify', return_value=self.plan), \
                patch.object(study.study_workloads, 'load_suite', return_value=self.suite), \
                patch.object(study, 'cpu_allocation') as allocate, patch.object(study, 'submit_stage') as submit:
            with self.assertRaisesRegex(RuntimeError, 'no GPU profiling'): study.worker(args)
            with self.assertRaisesRegex(RuntimeError, 'resume timings only'): study.resume(args)
        allocate.assert_not_called()
        submit.assert_not_called()

    def test_runtime_selects_cpu_modules_and_only_cpu_graph_packages(self):
        site, source, tool = self.root / 'site', self.root / 'source', self.root / 'spack'
        graph = site / 'site/pace/spack-envs/cpu'
        graph.mkdir(parents=True)
        (graph / 'spack.lock').write_text('{}')
        (source / 'cmake/sites').mkdir(parents=True)
        (source / 'cmake/sites/pace.json').write_text(json.dumps({'profiles': {
            'cpu': {'modules': ['gcc/12', 'openmpi/4']}, 'cuda-sm80': {'modules': ['cuda/12', 'openmpi/5']}}}))
        (tool / 'share/spack').mkdir(parents=True)
        (tool / 'share/spack/setup-env.sh').write_text(
            'spack() { printf "package:%s\\n" "$5" >&2; printf "%s\\n" "$FIXTURE_PREFIX"; }\n')
        prefix = self.root / 'installed'
        prefix.mkdir()
        self.plan.update(site_root=str(site), spack_root=str(tool))
        self.plan['sources']['candidate']['source_snapshot'] = str(source)
        self.plan['builds']['candidate']['spack_env'] = str(graph)
        plan = self.root / 'plan.json'
        study.write_json(plan, self.plan)
        script = ('hostname() { printf "compute-1\\n"; }; '
                  'module() { printf "module:%s\\n" "$*" >&2; }; '
                  'python3() { if [[ $1 == */gpu_study.py ]]; then '
                  '[[ $2 == verify ]] && return 0; [[ $2 == worker ]] && exit 77; fi; '
                  'command python3 "$@"; }; source "$1" "$2" timings "$3"')
        result = subprocess.run(['bash', '-c', script, 'fixture',
                                 str(Path(study.__file__).with_name('pace_runtime.sh')), str(plan), str(self.root / 'executor')],
                                env={**self.env, 'PATH': os.defpath, 'FIXTURE_PREFIX': str(prefix)},
                                text=True, capture_output=True, timeout=10)
        # exec bypasses the shell function; intercept only metadata/module/package
        # setup and the missing frozen worker then exits before any application.
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('module:load gcc/12 openmpi/4', result.stderr)
        self.assertEqual([line for line in result.stderr.splitlines() if line.startswith('package:')],
                         ['package:openmpi', 'package:hypre', 'package:openblas', 'package:metis'])
        self.assertNotIn('module:load cuda', result.stderr)


if __name__ == '__main__':
    unittest.main()
