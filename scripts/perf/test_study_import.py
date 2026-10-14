#!/usr/bin/env python3
"""Existing-artifact import contracts; no scheduler, compiler or application runs."""
from copy import deepcopy
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gpu_study
import study_admit
import study_import as study


class StudyImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.write(self.source / 'build.sh', 'canonical-driver')
        self.write(self.source / 'production.cpp', 'original-production')
        self.write(self.source / 'cmake/sites/pace.json', '{}')
        self.write(self.source / 'scripts/build/prebuilt_validation.py', 'pinned-checker')
        self.identity = {'source_snapshot': str(self.source), 'source_fingerprint': 'source-pinned',
                         'source_revision': 'frozen-revision', 'source_status': 'clean'}
        self.values = {'--site': 'pace', '--site-root': str(self.root / 'site'),
                       '--spack-root': str(self.root / 'tool'), '--output-root': str(self.root / 'output'),
                       '--mfem-output-root': str(self.root / 'mfem')}
        for value in self.values.values():
            if value != 'pace':
                Path(value).mkdir()
        self.request = self.root / 'build.request.json'
        argv = [argument for pair in self.values.items() for argument in pair]
        argv += ['--a100', '--hdg-runtime', '--nvtx-profile', '--no-research-tests', '--no-clean', '--no-tests', '--a100']
        self.write(self.request, json.dumps({'source': str(self.source), 'argv': argv}))
        self.wt = 'wt-' + hashlib.sha256(os.fsencode(self.source)).hexdigest()[:16]
        self.scope = Path(self.values['--output-root']) / 'worktrees' / self.wt / 'cuda-sm80'
        self.app = self.scope / 'app/build'
        self.app.mkdir(parents=True)
        self.prebuilt = self.app / 'igrlab-prebuilt-validation-v1.json'
        paths = {'source': str(self.source), 'app_build': str(self.app), 'app_install': str(self.scope / 'app/install'),
                 'spack_env': self.values['--site-root'] + '/site/pace/spack-envs/cuda-sm80',
                 **{key: self.values['--mfem-output-root'] + '/' + key.removeprefix('mfem_') for key in
                    ('mfem_source', 'mfem_build', 'mfem_install')}}
        self.binary = self.app / 'dgsem_smooth_advection_periodic'
        self.write(self.binary, '\x7fELFexact-artifact')
        self.binary.chmod(0o755)
        self.tests = [{'name': name, 'command': ['/actual/' + name], 'properties': [], 'backtrace': index + 2}
                      for index, name in enumerate(('accept-a', 'accept-b', 'other'))]
        self.saved = {'schema': 'igrlab-prebuilt-validation-v1', 'paths': paths,
                      'source': {'revision': 'frozen-revision', 'fingerprint': 'canonical-source'},
                      'policy': {'site': 'pace', 'profile': 'cuda-sm80'},
                      'artifacts': {'test_graph': {'tests': self.tests},
                                    'files': {str(self.binary): {'sha256': study.sha(self.binary)}}, 'links': {}}}
        self.write(self.prebuilt, json.dumps(self.saved))
        self.result_path = self.root / 'real-test-attempt/result.json'
        self.result_path.parent.mkdir()
        self.stdout = self.result_path.with_name('stdout.log')
        self.stderr = self.result_path.with_name('stderr.log')
        self.write(self.stdout, '1/2 Test #4: accept-a ........ Passed 0.25 sec\n'
                                '2/2 Test #9: accept-b ........ Passed 0.30 sec\n')
        self.write(self.stderr, '')
        _, runner_path = study.result_module()
        self.write(self.source / 'scripts/build/test_result_receipt.py', runner_path.read_text())
        self.user = pwd.getpwuid(os.getuid()).pw_name
        self.fields = {'JobId': '202', 'JobState': 'RUNNING', 'UserId': f'{self.user}({os.getuid()})',
                       'Account': 'gts-sbryngelson3', 'QOS': 'embers', 'Partition': 'gpu-a100',
                       'NodeList': 'compute-1', 'Command': '/bin/bash', 'BatchFlag': '0',
                       'StartTime': self.date(1700000000), 'SubmitTime': self.date(1699999990)}
        snapshot = {'uid': os.getuid(), 'hostname': 'compute-1.pace', 'authority': 'live PACE allocation',
                    'environment': {'SLURM_JOB_ID': '202'},
                    'scheduler': {'returncode': 0, 'fields': self.fields},
                    'gpu': {'name': 'NVIDIA A100', 'compute_capability': '8.0',
                            'uuid': 'GPU-11111111-2222-3333-4444-555555555555'}}
        selected = deepcopy(self.tests[:2])
        for test in selected:
            test['backtrace'] = 1  # Real CTest renumbers its filtered graph.
        command = [str(self.source / 'build.sh'), *argv[:-1]]
        command.remove('--no-tests')
        command += ['--tests-only', '--no-deps', '--test-regex', '^accept-']
        module, _ = study.result_module()
        self.result = {'schema': module.SCHEMA, 'status': 'passed', 'success': True, 'returncode': 0,
                       'identity_verified_before_and_after': True,
                       'argv': command, 'selection': {'regex': '^accept-', 'label': None},
                       'source': self.saved['source'], 'policy': self.saved['policy'],
                       'runner_sha256': study.sha(runner_path),
                       'prebuilt_checker_sha256': study.sha(self.source / 'scripts/build/prebuilt_validation.py'),
                       'prebuilt_receipt': str(self.prebuilt), 'prebuilt_receipt_sha256': study.sha(self.prebuilt),
                       'selected_ctest': {'returncode': 0, 'graph': {'tests': selected}},
                       'test_outcomes': module.test_outcomes(self.stdout.read_text(), {'tests': selected}),
                       'allocation_before': snapshot, 'allocation_after': deepcopy(snapshot),
                       'started_unix_seconds': 1700000100, 'finished_unix_seconds': 1700000200,
                       'logs': {path.name: {'path': str(path), 'sha256': study.sha(path)}
                                for path in (self.stdout, self.stderr)}}
        self.publish_result()
        self.record = {'build_request': str(self.request), 'build_job_id': '101', 'test_result': str(self.result_path)}
        self.build_state = 'COMPLETED'
        self.test_state = 'TIMEOUT'
        self.test_end = 1700000300
        self.calls = []
        self.patches = [patch.object(study.study_build, '_identity', return_value=self.identity),
                        patch.object(study.study_build, 'verify_source', side_effect=self.verify_source),
                        patch.object(study.study_build, '_run', side_effect=self.plan_query),
                        patch.object(study, 'run', side_effect=self.scheduler)]
        self.receipt_module, self.runner_path = study.result_module()
        self.receipt_module.checker = lambda source: (
            SimpleNamespace(source_identity=self.canonical_source, verify=lambda args: self.saved),
            source / 'scripts/build/prebuilt_validation.py')
        self.patches.append(patch.object(study, 'result_module', return_value=(self.receipt_module, self.runner_path)))
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def publish_result(self):
        self.write(self.result_path, json.dumps(self.result))

    @staticmethod
    def date(epoch):
        return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec='seconds').removesuffix('+00:00')

    def verify_source(self, identity):
        self.assertEqual(identity, self.identity)
        study.require((self.source / 'build.sh').read_text() == 'canonical-driver', 'source changed')

    def canonical_source(self, source):
        study.require((source / 'production.cpp').read_text() == 'original-production',
                      'current source differs from the source actually built and tested')
        return self.saved['source']

    def plan_query(self, argv, **kwargs):
        self.calls.append(argv)
        self.assertEqual(argv[0], str(self.source / 'build.sh'))
        self.assertEqual(argv[-1], '--print-plan')
        profile = 'cpu' if '--cpu-only' in argv else 'cuda-sm80'
        scope = Path(self.values['--output-root']) / 'worktrees' / self.wt / profile
        env = self.values['--site-root'] + '/site/pace/spack-envs/' + profile
        text = (f'[build-plan] output_root={self.values["--output-root"]} worktree_id={self.wt} scope={scope}\n'
                f'[build-plan] env={env} app_build={scope}/app/build app_install={scope}/app/install\n'
                f'[build-plan] mfem_source={self.values["--mfem-output-root"]}/source '
                f'mfem_build={self.values["--mfem-output-root"]}/build '
                f'mfem_install={self.values["--mfem-output-root"]}/install lock=immutable\n')
        return subprocess.CompletedProcess(argv, 0, text.encode(), b'')

    def scheduler(self, argv):
        self.calls.append(argv)
        if argv[0] == 'sacct':
            job = argv[argv.index('-j') + 1]
            if job == '101':
                row = [job, self.build_state, '0:0', self.user, 'gts-sbryngelson3', 'embers', 'IGRlab-build',
                       self.date(1699999800), self.date(1699999900), 'cpu-1',
                       f'sbatch {self.source}/scripts/build/pace.sh --worker build {self.request}']
            else:
                row = [job, self.test_state, '0:0' if self.test_state == 'RUNNING' else '0:1', self.user,
                       'gts-sbryngelson3', 'embers', 'bash', self.date(1700000000),
                       self.date(self.test_end) if self.test_state != 'RUNNING' else 'Unknown', 'compute-1', 'srun --pty bash -l']
            output = '|'.join(row)
        elif argv == ['scontrol', 'show', 'hostnames', 'compute-1']:
            output = 'compute-1\n'
        else:
            self.assertEqual(argv, ['scontrol', 'show', 'job', '-o', '202'])
            output = ' '.join(key + '=' + value for key, value in self.fields.items())
        return subprocess.CompletedProcess(argv, 0, output, '')

    def plan(self, *, reuse_validated_artifacts=False):
        return study.existing_plan(self.record, '^accept-',
                                   reuse_validated_artifacts=reuse_validated_artifacts)

    def test_distinct_baseline_and_candidate_source_runners_import(self):
        identities = []
        for member in ('baseline', 'candidate'):
            with self.subTest(member=member):
                runner = self.source / 'scripts/build/test_result_receipt.py'
                self.write(runner, self.runner_path.read_text() + '\n# frozen ' + member + ' revision\n')
                self.result['runner_sha256'] = study.sha(runner)
                self.publish_result()
                plan = self.plan()
                self.assertEqual(plan['test_result_runner_sha256'], study.sha(runner))
                study.validate_result(plan, full=True)
                identities.append(plan['test_result_runner_sha256'])
        self.assertNotEqual(*identities)

    def test_forged_runner_and_changed_source_runner_rejected(self):
        runner = self.source / 'scripts/build/test_result_receipt.py'
        plan = self.plan()
        self.result['runner_sha256'] = '0' * 64
        self.publish_result()
        plan['test_result_sha256'] = study.sha(self.result_path)
        with self.assertRaisesRegex(ValueError, 'runner identity'):
            study.validate_result(plan)
        self.result['runner_sha256'] = study.sha(runner)
        self.publish_result()
        plan['test_result_sha256'] = study.sha(self.result_path)
        runner.write_text(runner.read_text() + '\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'runner identity'):
            study.validate_result(plan)

    def test_legacy_plan_without_explicit_runner_pin_still_imports(self):
        plan = self.plan()
        del plan['test_result_runner_sha256']
        study.validate_result(plan, full=True)

    def test_explicit_artifact_reuse_retains_seals_without_historical_queries(self):
        # Neither expired accounting nor the later state of the old allocation
        # is an authority in this mode; the sealed real outcomes remain so.
        with patch.object(study, 'run', side_effect=AssertionError('historical query')), \
                patch.object(study, 'accounting', side_effect=AssertionError('historical accounting')):
            plan = self.plan(reuse_validated_artifacts=True)
            result = study.completed_records(plan, study.receipt(plan), full=True)
        self.assertTrue(plan['reuse_validated_artifacts'])
        self.assertTrue(study.receipt(plan)['reuse_validated_artifacts'])
        self.assertEqual(result['artifact_sha256'][str(self.binary)], study.sha(self.binary))
        for key, job in (('build_accounting', '101'), ('test_allocation_accounting', '202')):
            audit = result[key]
            self.assertFalse(audit['query_performed'])
            self.assertEqual(audit['query_status'], 'not_queried')
            self.assertEqual(audit['historical_job_id'], job)
            self.assertEqual(audit['test_result_sha256'], study.sha(self.result_path))
            self.assertEqual(audit['prebuilt_receipt_sha256'], study.sha(self.prebuilt))
            self.assertNotIn('row', audit)
        self.assertEqual(result['job_ids'], {'build': '101'})
        self.assertIn('no historical job completion claim', result['test_execution_scope'])

    def test_default_and_legacy_imports_keep_strict_accounting(self):
        plan = self.plan()
        receipt = study.receipt(plan)
        del plan['reuse_validated_artifacts']
        del receipt['reuse_validated_artifacts']
        self.build_state = 'FAILED'
        with self.assertRaisesRegex(ValueError, 'build did not complete'):
            study.completed_records(plan, receipt)
        self.assertEqual(self.calls[-1][0], 'sacct')

    def test_artifact_reuse_requires_exact_boolean_plan_and_receipt_mode(self):
        for mode in (1, 'yes', None):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'explicit boolean'):
                self.plan(reuse_validated_artifacts=mode)
        plan = self.plan(reuse_validated_artifacts=True)
        for mode in (False, 1, None):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'reuse mode differs'):
                study.completed_records(plan, {**study.receipt(plan), 'reuse_validated_artifacts': mode})

    def test_artifact_reuse_rejects_changed_seals_and_canonical_verifier_failure(self):
        plan = self.plan(reuse_validated_artifacts=True)
        with patch.object(study, 'run', side_effect=AssertionError('historical query')):
            for path in (self.request, self.result_path, self.stdout, self.prebuilt,
                         self.source / 'scripts/build/prebuilt_validation.py', self.source / 'production.cpp'):
                original = path.read_bytes()
                with self.subTest(path=path):
                    path.write_bytes(original + b' ')
                    with self.assertRaises(ValueError):
                        study.completed_records(plan, study.receipt(plan))
                    path.write_bytes(original)
            with patch.object(self.receipt_module, 'admission',
                              side_effect=ValueError('canonical artifact bytes changed')):
                with self.assertRaisesRegex(ValueError, 'canonical artifact bytes changed'):
                    study.completed_records(plan, study.receipt(plan), full=True)

    def test_artifact_reuse_still_requires_passed_coverage_and_native_free_allocation(self):
        original = deepcopy(self.result)
        for change in ({'success': False}, {'returncode': 8}, {'test_outcomes': []},
                       {'identity_verified_before_and_after': False}):
            with self.subTest(change=change):
                self.result = {**deepcopy(original), **change}
                self.publish_result()
                with self.assertRaises(ValueError):
                    self.plan(reuse_validated_artifacts=True)
        for field, value in (('QOS', 'paid'), ('Account', 'unrelated')):
            with self.subTest(field=field):
                self.result = deepcopy(original)
                self.result['allocation_before']['scheduler']['fields'][field] = value
                self.result['allocation_after']['scheduler']['fields'][field] = value
                self.publish_result()
                with self.assertRaises(ValueError):
                    self.plan(reuse_validated_artifacts=True)
        self.result = deepcopy(original)
        self.result['allocation_before']['gpu']['compute_capability'] = '7.0'
        self.result['allocation_after']['gpu']['compute_capability'] = '7.0'
        self.publish_result()
        with self.assertRaises(ValueError):
            self.plan(reuse_validated_artifacts=True)
        self.result = deepcopy(original)
        self.result['selected_ctest']['graph']['tests'] = self.tests[:1]
        self.publish_result()
        with self.assertRaisesRegex(ValueError, 'cover the suite'):
            self.plan(reuse_validated_artifacts=True)

    def test_manifest_applies_explicit_reuse_to_each_imported_member(self):
        manifest = self.root / 'reuse-import.json'
        self.write(manifest, json.dumps({'schema_version': 1, 'builds': dict.fromkeys(study.MEMBERS, self.record)}))
        with patch.object(study, 'run', side_effect=AssertionError('historical query')):
            plans, receipts, records, _ = study.inspect_manifest(
                manifest, '^accept-', '^cpu-', reuse_validated_artifacts=True)
        for member in study.MEMBERS:
            self.assertTrue(plans['builds'][member]['reuse_validated_artifacts'])
            self.assertTrue(receipts['builds'][member]['reuse_validated_artifacts'])
            self.assertEqual(records['builds/' + member]['build_accounting']['query_status'], 'not_queried')

    def test_actual_build_only_duplicate_a100_and_filtered_backtrace_import(self):
        plan = self.plan()
        record = study.completed_records(plan, study.receipt(plan))
        self.assertNotIn('validation_job_id', study.receipt(plan))
        self.assertEqual(record['test_allocation_accounting']['row'][1:3], ['TIMEOUT', '0:1'])
        self.assertEqual(plan['app_build'], str(self.app))
        self.assertTrue(all(argv[0] in ('sacct', 'scontrol') or argv[-1] == '--print-plan' for argv in self.calls))

    def test_real_live_interactive_same_job_allowed_other_job_rejected(self):
        self.test_state = 'RUNNING'
        plan = self.plan()
        with patch.dict(os.environ, {'SLURM_JOB_ID': '202'}):
            study.completed_records(plan, study.receipt(plan))
        with patch.dict(os.environ, {'SLURM_JOB_ID': '303'}):
            with self.assertRaisesRegex(ValueError, 'same owned running'):
                study.completed_records(plan, study.receipt(plan))

    def test_live_batch_job_rejected(self):
        self.test_state = 'RUNNING'
        self.fields['BatchFlag'] = '1'
        self.result['allocation_after']['scheduler']['fields']['BatchFlag'] = '1'
        self.publish_result()
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, 'interactive'):
            study.completed_records(plan, study.receipt(plan))

    def test_failed_build_accounting_rejected(self):
        self.build_state = 'FAILED'
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, 'build did not complete'):
            study.completed_records(plan, study.receipt(plan))

    def test_failed_skipped_missing_and_partial_outcomes_rejected(self):
        original = deepcopy(self.result)
        for change in ({'success': False}, {'status': 'failed'}, {'returncode': 8},
                       {'test_outcomes': []}, {'identity_verified_before_and_after': False}):
            with self.subTest(change=change):
                self.result = {**original, **change}
                self.publish_result()
                with self.assertRaises(ValueError):
                    self.plan()

    def test_missing_required_test_and_changed_test_command_rejected(self):
        self.result['selected_ctest']['graph']['tests'][1]['command'] = ['/different/solver']
        self.publish_result()
        with self.assertRaisesRegex(ValueError, 'commands differ'):
            self.plan()
        self.result['selected_ctest']['graph']['tests'][1]['command'] = self.tests[1]['command']
        self.result['selected_ctest']['graph']['tests'] = self.result['selected_ctest']['graph']['tests'][:1]
        self.publish_result()
        with self.assertRaisesRegex(ValueError, 'cover the suite'):
            self.plan()

    def test_changed_logs_receipt_prebuilt_source_checker_rejected(self):
        plan = self.plan()
        for path in (self.stdout, self.result_path, self.prebuilt,
                     self.source / 'scripts/build/prebuilt_validation.py'):
            original = path.read_bytes()
            with self.subTest(path=path):
                path.write_bytes(original + b' ')
                with self.assertRaises(ValueError):
                    study.validate_result(plan)
                path.write_bytes(original)
        (self.source / 'build.sh').write_text('changed-source')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            study.completed_records(plan, study.receipt(plan))

    def test_missing_result_and_symlink_request_rejected(self):
        self.result_path.unlink()
        with self.assertRaises(ValueError):
            self.plan()
        self.publish_result()
        link = self.request.with_name('alias.json')
        link.symlink_to(self.request)
        with self.assertRaises(ValueError):
            study.existing_plan({**self.record, 'build_request': str(link)}, '^accept-')

    def test_build_request_and_test_features_cannot_conflict(self):
        self.result['argv'].remove('--nvtx-profile')
        self.publish_result()
        with self.assertRaisesRegex(ValueError, 'build features'):
            self.plan()

    def test_arbitrary_validation_job_substitution_rejected(self):
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, 'validation job substitute'):
            study.completed_records(plan, {**study.receipt(plan), 'validation_job_id': '999'})

    def test_scheduler_time_and_node_proof_rejected(self):
        for change in ({'started_unix_seconds': 1699999000}, {'finished_unix_seconds': 1700000400}):
            original = deepcopy(self.result)
            self.result.update(change)
            self.publish_result()
            plan = self.plan()
            with self.assertRaises(ValueError):
                study.completed_records(plan, study.receipt(plan))
            self.result = original
        self.result['allocation_before']['hostname'] = 'unallocated-node.pace'
        self.result['allocation_after']['hostname'] = 'unallocated-node.pace'
        self.publish_result()
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, 'outside the recorded compute'):
            study.completed_records(plan, study.receipt(plan))

    def test_sacct_end_second_interval_accepts_fraction_but_not_next_second(self):
        # Observed native receipt: finish 05:25:53.7302136Z, sacct End05:25:53.
        self.test_state = 'COMPLETED'
        self.test_end = 1791782753
        self.result['started_unix_seconds'] = 1791782667.1940846
        for finish in (float(self.test_end), 1791782753.7302136,
                       math.nextafter(float(self.test_end + 1), float(self.test_end))):
            with self.subTest(accepted_finish=finish):
                self.result['finished_unix_seconds'] = finish
                self.publish_result()
                plan = self.plan()
                record = study.completed_records(plan, study.receipt(plan))
                self.assertEqual(record['test_allocation_accounting']['row'][8], '2026-10-12T05:25:53')
        for finish in (float(self.test_end + 1), self.test_end + 1.25):
            with self.subTest(rejected_finish=finish):
                self.result['finished_unix_seconds'] = finish
                self.publish_result()
                plan = self.plan()
                with self.assertRaisesRegex(ValueError, 'final accounting covering'):
                    study.completed_records(plan, study.receipt(plan))

    def test_full_worker_admission_reuses_canonical_verifier(self):
        plan = self.plan()
        actual, _ = study.result_module()
        checker = SimpleNamespace(verify=lambda args: self.saved, source_identity=self.canonical_source)
        module = SimpleNamespace(**vars(actual))
        module.checker = lambda source: (checker, source / 'scripts/build/prebuilt_validation.py')
        with patch.object(study, 'result_module', return_value=(module, Path(actual.__file__))):
            verified = study.completed_records(plan, study.receipt(plan), full=True)
        self.assertEqual(verified['artifact_sha256'][str(self.binary)], study.sha(self.binary))

    def test_public_import_freezes_current_operator_no_build_submission(self):
        manifest = self.root / 'import.json'
        self.write(manifest, json.dumps({'schema_version': 1,
                                       'builds': dict.fromkeys(study.MEMBERS, self.record)}))
        suite = Path(__file__).parent / 'suites/pace-full-gpu-solvers.json'
        # Its genuine selected correctness regex is admitted only by a covering
        # recorded graph. This test changes that selection to our shim graph.
        value = json.loads(suite.read_text())
        value['correctness_regex'] = '^accept-'
        suite = self.root / 'suite.json'
        self.write(suite, json.dumps(value))
        args = SimpleNamespace(artifacts=str(manifest), suite=str(suite), run_dir=str(self.root / 'new-study'),
                               prepare_only=True, timings_time='00:15:00', profiles_time='00:30:00')
        with patch.object(gpu_study, 'submit_stage') as submit:
            root, plan = gpu_study.import_existing(args)
        submit.assert_not_called()
        state = gpu_study.read_json(root / 'state.json')
        self.assertEqual(state['attempts'], [])
        self.assertEqual(plan['sources']['baseline'], self.identity)
        self.assertIn('study_import.py', plan['executor_sha256'])
        self.assertIn('canonical_test_result_receipt.py', plan['executor_sha256'])
        gpu_study.verify(root)
        gpu_study.checked_builds(root)
        self.stdout.write_text('changed actual test log')
        with self.assertRaises(RuntimeError):
            gpu_study.verify(root)

    def test_real_site_wrapper_strips_only_its_own_flags(self):
        for flag in ('--site', '--site-root', '--spack-root'):
            index = self.result['argv'].index(flag)
            del self.result['argv'][index:index + 2]
        self.publish_result()
        study.completed_records(self.plan(), study.receipt(self.plan()))
        index = self.result['argv'].index('--output-root')
        self.result['argv'][index + 1] = self.values['--site-root']
        self.publish_result()
        with self.assertRaises(ValueError):
            self.plan()

    def test_inside_allocation_import_cannot_submit_more_gpu_jobs(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '202'}), \
                patch.object(gpu_study, 'submit_stage') as submit:
            with self.assertRaisesRegex(RuntimeError, 'prepare-only'):
                gpu_study.import_existing(SimpleNamespace(prepare_only=False))
        submit.assert_not_called()

    def test_historical_executor_sets_remain_explicit(self):
        self.assertEqual(len(gpu_study.LEGACY_EXECUTOR_FILES), 5)
        self.assertEqual(len(gpu_study.PRE_IMPORT_EXECUTOR_FILES), 6)
        self.assertEqual(len(gpu_study.PRE_CPU_CONTINUATION_EXECUTOR_FILES), 8)
        self.assertEqual(len(gpu_study.PRE_DELTA_EXECUTOR_FILES), 9)
        self.assertEqual(gpu_study.PRE_DELTA_EXECUTOR_FILES,
                         gpu_study.PRE_CPU_CONTINUATION_EXECUTOR_FILES + ('study_cpu_continuation.py',))
        self.assertEqual(gpu_study.EXECUTOR_FILES,
                         gpu_study.PRE_DELTA_EXECUTOR_FILES + ('study_site.py', 'delta_runtime.sh'))

    def test_current_changed_production_source_cannot_be_bound_to_old_build(self):
        (self.source / 'production.cpp').write_text('new-unbuilt-source')
        with self.assertRaisesRegex(ValueError, 'actually built and tested'):
            self.plan()

    def test_new_selected_binary_missing_or_drift_rejected_before_publication(self):
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, 'missing or ambiguous'):
            study.selected_binaries(plan, ['previously-unbuilt-probe'])
        self.binary.write_text('\x7fELFchanged-binary')
        with self.assertRaisesRegex(ValueError, 'executable changed'):
            study.selected_binaries(plan, [self.binary.name])

    def cpu_request(self):
        request = json.loads(self.request.read_text())
        request['argv'] = [arg for arg in request['argv'] if arg not in ('--a100', '--nvtx-profile')]
        request['argv'].append('--cpu-only')
        self.write(self.request, json.dumps(request))
        scope = self.scope.with_name('cpu')
        prebuilt = scope / 'app/build/igrlab-prebuilt-validation-v1.json'
        saved = deepcopy(self.saved)
        saved['paths'].update(app_build=str(prebuilt.parent), app_install=str(scope / 'app/install'),
                              spack_env=self.values['--site-root'] + '/site/pace/spack-envs/cpu')
        saved['policy']['profile'] = 'cpu'
        self.write(prebuilt, json.dumps(saved))
        self.result['policy'] = saved['policy']
        self.result['prebuilt_receipt'] = str(prebuilt)
        self.result['prebuilt_receipt_sha256'] = study.sha(prebuilt)
        self.result['argv'] = [arg for arg in self.result['argv'] if arg not in ('--a100', '--nvtx-profile')]
        self.result['argv'].append('--cpu-only')
        self.publish_result()

    def test_real_cpu_request_is_admitted_without_cuda_or_fake_validation_job(self):
        self.cpu_request()
        plan = self.plan()
        receipt = study.receipt(plan)
        admitted = study_admit.completed_jobs(plan, receipt)
        self.assertEqual(plan['profile'], 'cpu')
        self.assertNotIn('validation_job_id', receipt)
        self.assertEqual(admitted['test_result'], str(self.result_path))
        self.assertEqual(admitted['job_ids'], {'build': '101'})
        with patch.object(study, 'run', side_effect=AssertionError('historical query')):
            reuse_plan = self.plan(reuse_validated_artifacts=True)
            reused = study_admit.completed_jobs(reuse_plan, study.receipt(reuse_plan))
        self.assertEqual(reused['test_allocation_accounting']['query_status'], 'not_queried')
        self.assertEqual(reused['build_accounting']['historical_job_id'], '101')

    def test_explicit_cpu_only_manifest_uses_cpu_seals_and_selected_regex(self):
        self.cpu_request()
        manifest = self.root / 'cpu-only.json'
        self.write(manifest, json.dumps({'schema_version': 1,
                                       'builds': dict.fromkeys(study.MEMBERS, self.record)}))
        with patch.object(study, 'run', side_effect=AssertionError('historical query')):
            plans, receipts, records, hashes = study.inspect_manifest(
                manifest, '^accept-', '^unrelated-', cpu_only=True,
                reuse_validated_artifacts=True)
        self.assertEqual(set(plans), {'builds'})
        self.assertEqual(hashes, {str(manifest): study.sha(manifest)})
        for member in study.MEMBERS:
            self.assertEqual(plans['builds'][member]['profile'], 'cpu')
            self.assertEqual(plans['builds'][member]['correctness_regex'], '^accept-')
            self.assertNotIn('validation_job_id', receipts['builds'][member])
            self.assertEqual(records['builds/' + member]['build_accounting']['query_status'], 'not_queried')
        with self.assertRaisesRegex(ValueError, 'profile differs'):
            study.inspect_manifest(manifest, '^accept-', '^unrelated-')
        with self.assertRaisesRegex(ValueError, 'cover the suite'):
            study.inspect_manifest(manifest, '^other$', '^accept-', cpu_only=True,
                                   reuse_validated_artifacts=True)

    def test_cpu_only_rejects_cuda_extra_cohort_and_nonboolean_mode(self):
        manifest = self.root / 'cpu-only.json'
        value = {'schema_version': 1, 'builds': dict.fromkeys(study.MEMBERS, self.record)}
        self.write(manifest, json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'profile differs'):
            study.inspect_manifest(manifest, '^accept-', '^accept-', cpu_only=True)
        value['cpu_builds'] = value['builds']
        self.write(manifest, json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'one CPU builds cohort'):
            study.inspect_manifest(manifest, '^accept-', '^accept-', cpu_only=True)
        for mode in (1, 'true', None):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'explicit boolean'):
                study.inspect_manifest(manifest, '^accept-', '^accept-', cpu_only=mode)

    def test_cpu_group_cannot_import_cuda_facade_records(self):
        manifest = self.root / 'mixed.json'
        self.write(manifest, json.dumps({'schema_version': 1,
                       'builds': dict.fromkeys(study.MEMBERS, self.record),
                       'cpu_builds': dict.fromkeys(study.MEMBERS, self.record)}))
        with self.assertRaisesRegex(ValueError, 'profile differs'):
            study.inspect_manifest(manifest, '^accept-', '^accept-')

    def test_default_import_submits_stages_under_worker_exclusion_lock(self):
        manifest = self.root / 'import.json'
        self.write(manifest, json.dumps({'schema_version': 1, 'builds': dict.fromkeys(study.MEMBERS, self.record)}))
        value = json.loads((Path(__file__).parent / 'suites/pace-full-gpu-solvers.json').read_text())
        value['correctness_regex'] = '^accept-'
        suite = self.root / 'suite.json'
        self.write(suite, json.dumps(value))
        args = SimpleNamespace(artifacts=str(manifest), suite=str(suite), run_dir=str(self.root / 'new-study'),
                               prepare_only=False, timings_time='00:15:00', profiles_time='00:30:00')
        held, calls = [], []

        @contextlib.contextmanager
        def lock(root):
            held.append(root)
            yield
            held.pop()

        def submit(root, stage, dependencies=()):
            self.assertEqual(held, [root])
            calls.append((stage, dependencies))
            return '123'

        with patch.dict(os.environ, {}, clear=True), patch.object(gpu_study, 'locked', side_effect=lock), \
                patch.object(gpu_study, 'submit_stage', side_effect=submit):
            gpu_study.import_existing(args)
        self.assertEqual(calls, [('timings', ()), ('profiles', ['123'])])


if __name__ == '__main__':
    unittest.main()
