#!/usr/bin/env python3
"""Compiler-free contracts; all Slurm/GPU/process replies are synthetic fixtures."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import pwd
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE / 'scripts/perf'))
import delta_profile_pool as pool
import gpu_study


def parent_fields():
    return {'JobId': '42', 'JobState': 'RUNNING', 'NumNodes': '1', 'NumTasks': '1',
            'NumCPUs': '64', 'BatchFlag': '1', 'OverSubscribe': 'NO',
            'UserId': f'{pwd.getpwuid(os.getuid()).pw_name}({os.getuid()})',
            'Account': 'bdiy-delta-gpu', 'Partition': 'gpuA100x4', 'NodeList': 'gpua024',
            'Features': 'perf&nvperf', 'ReqTRES': 'cpu=33,mem=192G,node=1,gres/gpu=4',
            'AllocTRES': 'cpu=64,mem=192G,node=1,gres/gpu=4,gres/gpu:a100=4',
            'EndTime': '2099-10-14T12:00:00'}


def allocation(index=0):
    return {'job_id': '42', 'hostname': 'gpua024.delta.ncsa.illinois.edu',
            'end_epoch': time.time() + 3600, 'scheduler': parent_fields(),
            'gpu': {'uuid': 'GPU-' + f'{index:08x}' + '-1234-1234-1234-123456789abc',
                    'name': 'NVIDIA A100-SXM4-40GB', 'compute_capability': 80},
            'step': {'step_id': '42.' + str(index)}}


class PoolFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.session = self.root / 'records/delta-profile-pools/session'
        self.session.mkdir(parents=True)
        self.tasks = []
        self.plans = {}
        for index in range(4):
            run = self.root / 'studies' / str(index)
            executor = run / 'executor'
            executor.mkdir(parents=True)
            (executor / 'gpu_study.py').write_bytes(Path(gpu_study.__file__).read_bytes())
            (executor / 'delta_runtime.sh').write_text('# fixture frozen runtime\n')
            digest = str(index) * 64
            task = {'id': 'study-' + str(index), 'run_dir': str(run), 'plan_sha256': digest,
                    'executor_sha256': {name: pool.sha(executor / name)
                                        for name in ('gpu_study.py', 'delta_runtime.sh')}}
            summary = run / 'timing-summary.json'
            summary.write_text('{}\n')
            (run / 'admission.json').write_text(json.dumps({'plan_sha256': digest}))
            (run / 'state.json').write_text(json.dumps({'attempts': [
                {'id': 'timings-1', 'stage': 'timings', 'status': 'timings_complete',
                 'summary': str(summary), 'summary_sha256': pool.sha(summary)}], 'jobs': []}))
            self.plans[run] = {'plan_sha256': digest, 'site': 'delta',
                               'execution_profile': 'cuda-sm80', 'profile_continuation': {'captures': {}},
                               'executor_sha256': task['executor_sha256']}
            self.tasks.append(task)
        self.value = {'schema_version': 1, 'approved': True, 'site_root': str(self.root),
                      'controller_sha256': pool.sha(pool.__file__), 'tasks': self.tasks}
        self.input = self.root / 'intent.json'
        self.input.write_text(json.dumps(self.value))
        self.digest = pool.sha(self.input)
        self.args = SimpleNamespace(intent=self.input, intent_sha256=self.digest,
                                    role=None, session=None)
        pool.STOP = 0

    def tearDown(self):
        self.temp.cleanup()

    def load_plan(self, path):
        return self.plans[Path(path)]

    def change_intent(self, value):
        self.input.write_text(json.dumps(value))
        return pool.sha(self.input)

    def test_pinned_intent_distinct_runs_and_controller(self):
        self.assertEqual(pool.intent(self.input, self.digest)['tasks'], self.tasks)
        self.input.write_text(self.input.read_text() + ' ')
        with self.assertRaisesRegex(RuntimeError, 'hash changed'):
            pool.intent(self.input, self.digest)
        for field, bad in [('approved', False), ('controller_sha256', '0' * 64)]:
            value = copy.deepcopy(self.value)
            value[field] = bad
            with self.assertRaises(RuntimeError):
                pool.intent(self.input, self.change_intent(value))

    def test_duplicate_study_alias_id_and_foreign_root_rejected(self):
        for key, bad in [('run_dir', self.tasks[0]['run_dir']), ('id', self.tasks[0]['id']),
                         ('run_dir', str(self.root / 'records')), ('id', '../escape')]:
            value = copy.deepcopy(self.value)
            value['tasks'][1][key] = bad
            with self.assertRaises(RuntimeError):
                pool.intent(self.input, self.change_intent(value))
        alias = self.root / 'studies/alias'
        alias.symlink_to(Path(self.tasks[0]['run_dir']), target_is_directory=True)
        value['tasks'][1]['run_dir'] = str(alias)
        with self.assertRaises(RuntimeError):
            pool.intent(self.input, self.change_intent(value))

    def test_new_parent_is_strict_and_never_ordinary_or_legacy_admission(self):
        fields = parent_fields()
        self.assertTrue(gpu_study._delta_profile_pool_parent(fields))
        for key, bad in [('BatchFlag', '0'), ('NumCPUs', '32'), ('OverSubscribe', 'YES'),
                         ('Features', 'nvperf'), ('ReqTRES', 'cpu=32,mem=192G,gres/gpu=4'),
                         ('ReqTRES', 'cpu=33,mem=48G,gres/gpu=4'),
                         ('AllocTRES', 'cpu=64,mem=192G,gres/gpu=1'),
                         ('AllocTRES', 'cpu=64,cpu=64,mem=192G,gres/gpu=4'),
                         ('AllocTRES', 'cpu=64,mem=192G,gres/gpu=4,gres/gpu:a100=1')]:
            with self.subTest(key=key, bad=bad):
                changed = {**fields, key: bad}
                self.assertFalse(gpu_study._delta_profile_pool_parent(changed))
        with self.assertRaises(RuntimeError):
            gpu_study.delta_gpu_resources(fields, '42', 'gpua024', {})
        ordinary = {'NumCPUs': '8', 'AllocTRES': 'cpu=8,mem=48G,gres/gpu=1'}
        self.assertIsNone(gpu_study.delta_gpu_resources(ordinary, '42', 'gpua024', {}))

    def test_actual_step_required_inside_pool_and_legacy_parent_unchanged(self):
        step = f'StepId=42.7 UserId={os.getuid()} State=RUNNING Partition=gpuA100x4 Nodes=1 CPUs=8 Tasks=1 NodeList=gpua024 TRES=cpu=8,mem=48G,gres/gpu=1'
        with mock.patch.dict(os.environ, {'DELTA_OWNED_STEP_ID': '7'}, clear=True), \
             mock.patch.object(gpu_study.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=step)) as query:
            result = gpu_study.delta_gpu_resources(parent_fields(), '42', 'gpua024', {}, profile_pool=True)
            self.assertEqual(result['step_id'], '42.7')
            self.assertEqual(query.call_args.args[0], ['scontrol', 'show', 'step', '-o', '42.7'])
            legacy = {**parent_fields(), 'BatchFlag': '0', 'ReqTRES': 'cpu=8,mem=48G,gres/gpu=1',
                      'AllocTRES': 'cpu=64,mem=48G,gres/gpu=4'}
            self.assertEqual(gpu_study.delta_gpu_resources(legacy, '42', 'gpua024', {})['step_id'], '42.7')
            query.return_value.stdout = step.replace('CPUs=8', 'CPUs=7')
            with self.assertRaises(RuntimeError):
                gpu_study.delta_gpu_resources(parent_fields(), '42', 'gpua024', {}, profile_pool=True)
        with mock.patch.dict(os.environ, {'DELTA_OWNED_STEP_ID': '7', 'SLURM_STEP_ID': '8'}, clear=True):
            with self.assertRaises(RuntimeError):
                gpu_study.delta_gpu_resources(parent_fields(), '42', 'gpua024', {}, profile_pool=True)

    def test_worker_enables_new_admission_only_for_profiles(self):
        run = Path(self.tasks[0]['run_dir'])
        original_state = pool.read(run / 'state.json')
        for stage in ('timings', 'profiles'):
            (run / 'state.json').write_text(json.dumps(original_state if stage == 'profiles' else
                                                      {'attempts': [], 'jobs': []}))
            with mock.patch.object(gpu_study, 'verify', return_value=self.plans[run]), \
                 mock.patch.object(gpu_study.study_workloads, 'load_suite', return_value={'cases': []}), \
                 mock.patch.object(gpu_study, 'worker_budget'), \
                 mock.patch.object(gpu_study, 'delta_allocation', side_effect=RuntimeError('fixture stop')) as admit:
                with self.assertRaisesRegex(RuntimeError, 'fixture stop'):
                    gpu_study.worker(SimpleNamespace(run_dir=str(run), stage=stage))
                admit.assert_called_once_with(profile_pool=(stage == 'profiles'))

    def test_real_parent_and_node_capacity_precede_children(self):
        fields = parent_fields()
        node = {'CfgTRES': 'cpu=64,mem=250000M,gres/gpu=4'}
        with mock.patch.dict(os.environ, {'SLURM_JOB_ID': '42', 'SLURM_STEP_ID': 'batch'}, clear=True), \
             mock.patch.object(pool.socket, 'getfqdn', return_value='gpua024.delta.ncsa.illinois.edu'), \
             mock.patch.object(pool, 'scheduler', side_effect=[(fields, ['job']), (node, ['node'])]):
            self.assertEqual(pool.parent_allocation()['job_id'], '42')
        node['CfgTRES'] = 'cpu=64,mem=128G,gres/gpu=4'
        with mock.patch.dict(os.environ, {'SLURM_JOB_ID': '42', 'SLURM_STEP_ID': 'batch'}, clear=True), \
             mock.patch.object(pool.socket, 'getfqdn', return_value='gpua024.delta.ncsa.illinois.edu'), \
             mock.patch.object(pool, 'scheduler', side_effect=[(fields, []), (node, [])]):
            with self.assertRaisesRegex(RuntimeError, 'cannot fit'):
                pool.parent_allocation()
        with mock.patch.dict(os.environ, {'SLURM_JOB_ID': '42', 'SLURM_STEP_ID': '0'}, clear=True):
            with self.assertRaises(RuntimeError):
                pool.parent_allocation()

    def records(self, live):
        records = []
        for index in range(4):
            role = allocation(index)
            role['end_epoch'] = live['end_epoch']
            records.append({'allocation': role, 'cpu_affinity': list(range(index * 8, index * 8 + 8))})
        return records

    def test_rendezvous_uses_uuid_and_affinity_not_visible_ordinal(self):
        live = allocation()
        records = self.records(live)
        for item in records:
            item['allocation']['gpu']['allocation_environment'] = {'CUDA_VISIBLE_DEVICES': '0'}
        with mock.patch.object(pool.os, 'cpu_count', return_value=64):
            self.assertEqual(pool.rendezvous(records, live, set(range(33))), 32)
            # A batch coordinator can have a narrower affinity than its children.
            self.assertEqual(pool.rendezvous(records, live, {40}), 40)
            for mutate in (lambda r: r[1]['allocation']['gpu'].update(uuid=r[0]['allocation']['gpu']['uuid']),
                           lambda r: r[1].update(cpu_affinity=r[0]['cpu_affinity']),
                           lambda r: r[1]['allocation'].update(job_id='other'),
                           lambda r: r[1]['allocation']['step'].update(step_id='42.0')):
                changed = copy.deepcopy(records)
                mutate(changed)
                with self.assertRaises(RuntimeError):
                    pool.rendezvous(changed, live, set(range(64)))
            with self.assertRaisesRegex(RuntimeError, 'no disjoint'):
                pool.rendezvous(records, live, set(range(32)))

    def test_step_command_reserves_real_disjoint_roles_only_no_parent_submission(self):
        argv = pool.step_argv(self.tasks[0], allocation(), Path(pool.__file__), self.input, self.digest, self.session)
        for flag in ('--jobid=42', '--exclusive', '--exact', '--ntasks=1', '--cpus-per-task=8',
                     '--mem=48G', '--gpus-per-task=1', '--mpi=none', '--cpu-bind=cores'):
            self.assertIn(flag, argv)
        self.assertEqual(argv[0], 'srun')
        self.assertNotIn('sbatch', argv)
        self.assertNotIn('scancel', argv)

    def test_readiness_rejects_replay_mutation_and_preserves_completed(self):
        task = self.tasks[0]
        run = Path(task['run_dir'])
        with mock.patch.object(gpu_study, 'load_plan', side_effect=self.load_plan):
            self.assertEqual(pool.check_task(task), 'ready')
            state = pool.read(run / 'state.json')
            state['attempts'].append({'id': 'p', 'stage': 'profiles', 'status': 'profiles_partial'})
            (run / 'state.json').write_text(json.dumps(state))
            with self.assertRaisesRegex(RuntimeError, 'never replay'):
                pool.check_task(task)
            state['attempts'][-1]['status'] = 'analysis_complete'
            (run / 'state.json').write_text(json.dumps(state))
            self.assertEqual(pool.check_task(task), 'already_complete')
            (run / 'executor/delta_runtime.sh').write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'executor changed'):
                pool.check_task(task)

    def test_outcome_requires_bound_attempt_and_seal_not_just_exit_zero(self):
        task = self.tasks[0]
        run = Path(task['run_dir'])
        state = pool.read(run / 'state.json')
        summary = run / 'profile-summary.json'
        summary.write_text('{"status":"analysis_complete"}')
        live = allocation()
        new = {'id': 'profiles-42-native', 'stage': 'profiles', 'job_id': '42', 'gpu': live['gpu'],
               'status': 'analysis_complete', 'summary': str(summary), 'summary_sha256': pool.sha(summary)}
        state['attempts'].append(new)
        (run / 'state.json').write_text(json.dumps(state))
        result = pool.outcome(task, live, {'timings-1'}, 0)
        self.assertEqual(result['analysis_status'], 'complete')
        self.assertEqual(pool.outcome(task, live, {'timings-1'}, 1)['analysis_status'], 'incomplete')
        new['job_id'] = '43'
        (run / 'state.json').write_text(json.dumps(state))
        with self.assertRaisesRegex(RuntimeError, 'binding'):
            pool.outcome(task, live, {'timings-1'}, 0)
        new['job_id'] = '42'
        (run / 'state.json').write_text(json.dumps(state))
        summary.write_text('tampered')
        with self.assertRaisesRegex(RuntimeError, 'seal changed'):
            pool.outcome(task, live, {'timings-1'}, 0)

    def test_atomic_gate_cannot_overwrite_original(self):
        gate = self.session / 'gate'
        pool.write(gate, {'original': True})
        with self.assertRaises(FileExistsError):
            pool.write(gate, {'original': False})
        self.assertEqual(pool.read(gate), {'original': True})
        self.assertFalse(list(self.session.glob('*.tmp-*')))

    def test_callback_is_once_and_failure_does_not_break_retention(self):
        live = allocation()
        pool.event(self.root, live, 'profiles_started', 'pool-x', 'first')
        pool.event(self.root, live, 'profiles_started', 'pool-x', 'same semantic signal')
        self.assertEqual(len((self.root / 'hpc-events-42.jsonl').read_text().splitlines()), 1)
        with mock.patch.object(Path, 'open', side_effect=OSError('transport disconnected')):
            pool.event(self.root, live, 'failure', 'pool-x', 'must not raise')
        with mock.patch.object(pool, 'parent_allocation', return_value=live), \
             mock.patch.object(pool, 'retain') as retain:
            self.args.intent_sha256 = '0' * 64
            with self.assertRaises(RuntimeError):
                pool.coordinate(self.args)
            retain.assert_called_once_with(live)

    def test_unknown_session_never_launches_again_and_still_retains(self):
        live = allocation()
        session = self.root / 'records/delta-profile-pools' / ('42-' + self.digest)
        session.mkdir()
        with mock.patch.object(pool, 'parent_allocation', return_value=live), \
             mock.patch.object(pool, 'retain') as retain, \
             mock.patch.object(pool.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'never relaunch'):
                pool.coordinate(self.args)
            launch.assert_not_called()
            retain.assert_called_once()

    def test_fixed_pool_drains_distinct_studies_without_repeating_complete(self):
        live = allocation()
        with mock.patch.object(pool, 'parent_allocation', return_value=live), \
             mock.patch.object(pool, 'check_task', side_effect=['already_complete', 'ready', 'ready', 'ready']), \
             mock.patch.object(pool, 'wave', return_value=[{'task_id': t['id'], 'analysis_status': 'complete'}
                                                          for t in self.tasks[1:]]) as wave, \
             mock.patch.object(pool, 'retain') as retain, \
             mock.patch.object(pool, 'parent_idle', return_value={'actual_idle_fixture': True}):
            self.assertEqual(pool.coordinate(self.args), 0)
            self.assertEqual(wave.call_args.args[0], self.tasks[1:])
            retain.assert_called_once_with(live)

    def test_all_four_gates_are_bound_before_work_and_one_child_failure_is_honest(self):
        live = allocation()
        entries = self.records(live)
        for task, entry in zip(self.tasks, entries):
            entry.update(task_id=task['id'], intent_sha256=self.digest)
        class FakeProcess:
            def __init__(self, argv, **_kwargs):
                identifier = argv[argv.index('--role') + 1]
                self.index = int(identifier.split('-')[1])
                self.directory = self_outer.session / identifier
                self.returncode = None
                pool.write(self.directory / 'ready.json', entries[self.index])
            def poll(self):
                if (self.directory / 'start.gate').exists():
                    # No role has an application until the coordinator admitted the entire wave.
                    assert all((self_outer.session / t['id'] / 'ready.json').exists() for t in self_outer.tasks)
                    if not (self.directory / 'outcome.json').exists():
                        pool.write(self.directory / 'outcome.json', {'task_id': self_outer.tasks[self.index]['id'],
                            'analysis_status': 'complete' if self.index != 2 else 'incomplete',
                            'native_status': 'complete_and_admitted',
                            'final_gpu_idle': {'gpu_uuid': entries[self.index]['allocation']['gpu']['uuid']}})
                    self.returncode = 0 if self.index != 2 else 1
                return self.returncode
        self_outer = self
        with mock.patch.object(pool.subprocess, 'Popen', side_effect=FakeProcess), \
             mock.patch.object(pool.os, 'cpu_count', return_value=64), \
             mock.patch.object(pool, 'stop_process', side_effect=lambda p: self.assertIsNotNone(p.poll())), \
             mock.patch.object(pool.os, 'sched_setaffinity') as affinity:
            receipts = pool.wave(self.tasks, live, self.session, self.args, set(range(64)))
        affinity.assert_called_once_with(0, {32})
        self.assertEqual([r['analysis_status'] for r in receipts], ['complete', 'complete', 'incomplete', 'complete'])
        for task in self.tasks:
            directory = self.session / task['id']
            self.assertEqual(pool.read(directory / 'start.gate')['ready_sha256'], pool.sha(directory / 'ready.json'))

    def test_bad_physical_rendezvous_starts_nothing_and_reaps_all_owned_steps(self):
        live = allocation()
        entries = self.records(live)
        entries[1]['allocation']['gpu'] = entries[0]['allocation']['gpu']
        owners = []
        class WaitingProcess:
            def __init__(self, argv, **_kwargs):
                identifier = argv[argv.index('--role') + 1]
                index = int(identifier.split('-')[1])
                record = {**entries[index], 'task_id': identifier, 'intent_sha256': self_outer.digest}
                pool.write(self_outer.session / identifier / 'ready.json', record)
                self.returncode = None
                owners.append(self)
            def poll(self):
                return self.returncode
        self_outer = self
        with mock.patch.object(pool.subprocess, 'Popen', side_effect=WaitingProcess), \
             mock.patch.object(pool, 'stop_process', side_effect=lambda p: setattr(p, 'returncode', -15)) as stop, \
             mock.patch.object(pool.os, 'cpu_count', return_value=64), \
             mock.patch.object(pool.os, 'sched_setaffinity') as affinity:
            results = pool.wave(self.tasks, live, self.session, self.args, set(range(64)))
        self.assertEqual(stop.call_count, 4)
        self.assertTrue(all(p.returncode == -15 for p in owners))
        affinity.assert_not_called()
        self.assertFalse(list(self.session.glob('*/start.gate')))
        self.assertTrue(results[-1]['wave_failure'])
        self.assertTrue(all(r['analysis_status'] == 'incomplete' for r in results))

    def test_more_than_four_distinct_ready_studies_drain_in_bounded_waves(self):
        fifth = copy.deepcopy(self.tasks[0])
        fifth.update(id='study-4', run_dir=str(self.root / 'studies/4'))
        Path(fifth['run_dir']).mkdir()
        self.value['tasks'].append(fifth)
        self.args.intent_sha256 = self.change_intent(self.value)
        def results(tasks, *_args):
            return [{'task_id': t['id'], 'analysis_status': 'complete'} for t in tasks]
        with mock.patch.object(pool, 'parent_allocation', return_value=allocation()), \
             mock.patch.object(pool, 'check_task', return_value='ready'), \
             mock.patch.object(pool, 'wave', side_effect=results) as wave, \
             mock.patch.object(pool, 'retain'), \
             mock.patch.object(pool, 'parent_idle', return_value={'actual_idle_fixture': True}):
            self.assertEqual(pool.coordinate(self.args), 0)
        self.assertEqual([len(call.args[0]) for call in wave.call_args_list], [4, 1])
        self.assertEqual([t['id'] for call in wave.call_args_list for t in call.args[0]],
                         ['study-0', 'study-1', 'study-2', 'study-3', 'study-4'])

    def test_exited_leader_still_cleans_its_owned_group(self):
        process = SimpleNamespace(pid=987654321, poll=mock.Mock(return_value=0))
        alive = [True]
        def killpg(pid, number):
            self.assertEqual(pid, process.pid)
            if not alive[0]:
                raise ProcessLookupError()
            if number == pool.signal.SIGTERM:
                alive[0] = False
        with mock.patch.object(pool.os, 'killpg', side_effect=killpg) as kill:
            pool.stop_process(process)
        self.assertIn(mock.call(process.pid, pool.signal.SIGTERM), kill.call_args_list)

    def test_unreapable_role_does_not_skip_cleanup_of_remaining_steps(self):
        live = allocation()
        entries = self.records(live)
        owners = []
        class WaitingProcess:
            def __init__(self, argv, **_kwargs):
                identifier = argv[argv.index('--role') + 1]
                index = int(identifier.split('-')[1])
                record = {**entries[index], 'task_id': identifier, 'intent_sha256': self_outer.digest}
                pool.write(self_outer.session / identifier / 'ready.json', record)
                self.returncode = 0
                owners.append(self)
            def poll(self):
                return self.returncode
        self_outer = self
        def cleanup(process):
            if process is owners[0]:
                raise RuntimeError('unreapable private group')
        with mock.patch.object(pool.subprocess, 'Popen', side_effect=WaitingProcess), \
             mock.patch.object(pool, 'stop_process', side_effect=cleanup) as stop, \
             mock.patch.object(pool.os, 'cpu_count', return_value=64), \
             mock.patch.object(pool.os, 'sched_setaffinity'):
            results = pool.wave(self.tasks, live, self.session, self.args, set(range(64)))
        self.assertEqual(stop.call_count, 4)
        self.assertTrue(results[-1]['wave_failure'])
        self.assertEqual(results[0]['cleanup_error'], 'unreapable private group')

    def test_parent_idle_requires_four_native_identities_and_no_compute_process(self):
        live = allocation()
        identities = '\n'.join(allocation(i)['gpu']['uuid'] + ', NVIDIA A100-SXM4-40GB, 8.0' for i in range(4))
        with mock.patch.object(pool, 'parent_allocation', return_value=live), \
             mock.patch.object(pool.subprocess, 'run', side_effect=[
                 SimpleNamespace(returncode=0, stdout=identities), SimpleNamespace(returncode=0, stdout='')]):
            self.assertEqual(len(pool.parent_idle(live)['gpu_uuids']), 4)
        with mock.patch.object(pool, 'parent_allocation', return_value=live), \
             mock.patch.object(pool.subprocess, 'run', side_effect=[
                 SimpleNamespace(returncode=0, stdout=identities),
                 SimpleNamespace(returncode=0, stdout=allocation()['gpu']['uuid'] + ', 1234')]):
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                pool.parent_idle(live)

    def test_role_idle_filters_physical_uuid_while_other_gpu_work_is_allowed(self):
        own, other = allocation(0)['gpu'], allocation(1)['gpu']
        def query(argv, **_kwargs):
            # NVML ignores CUDA_VISIBLE_DEVICES: only its explicit --id filter scopes this query.
            stdout = '' if '--id=' + own['uuid'] in argv else other['uuid'] + ', 2468'
            return SimpleNamespace(returncode=0, stdout=stdout)
        with mock.patch.object(pool.subprocess, 'run', side_effect=query) as call:
            self.assertEqual(pool.idle_gpu(own)['gpu_uuid'], own['uuid'])
            self.assertIn('--id=' + own['uuid'], call.call_args.args[0])
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                pool.idle_gpu()  # Whole-parent handoff must still see the other role's app.
        with mock.patch.object(pool.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout=own['uuid'] + ', 1234')):
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                pool.idle_gpu(own)

    def test_unsafe_role_has_failure_only_no_attachable_handoff(self):
        live = allocation()
        with mock.patch.object(pool, 'parent_allocation', return_value=live), \
             mock.patch.object(pool, 'check_task', return_value='ready'), \
             mock.patch.object(pool, 'wave', return_value=[{'analysis_status': 'incomplete', 'unsafe_role': True}]), \
             mock.patch.object(pool, 'parent_idle') as idle, \
             mock.patch.object(pool, 'retain'):
            self.assertEqual(pool.coordinate(self.args), 1)
        idle.assert_not_called()
        events = [json.loads(line) for line in (self.root / 'hpc-events-42.jsonl').read_text().splitlines()]
        self.assertTrue(any(e['kind'] == 'failure' for e in events))
        self.assertFalse(any(e['family'] == 'allocation_retained' for e in events))

    def test_role_rejects_bad_gate_and_global_ncu_override_before_application(self):
        task = self.tasks[0]
        (self.session / task['id']).mkdir()
        self.args.role, self.args.session = task['id'], str(self.session)
        live = allocation()
        with mock.patch.object(gpu_study, 'delta_allocation', return_value=live), \
             mock.patch.object(pool, 'check_task', return_value='ready'), \
             mock.patch.object(pool, 'idle_gpu', return_value={}), \
             mock.patch.object(pool.os, 'sched_getaffinity', return_value=set(range(8))), \
             mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(pool.subprocess, 'Popen') as launch, \
             mock.patch.object(pool.time, 'sleep', side_effect=lambda _: pool.write(
                 self.session / task['id'] / 'start.gate', {'wrong': 'binding'})):
            with self.assertRaisesRegex(RuntimeError, 'gate identity'):
                pool.role(self.args, self.value)
            launch.assert_not_called()
        with mock.patch.object(gpu_study, 'delta_allocation', return_value=live), \
             mock.patch.dict(os.environ, {'NV_COMPUTE_PROFILER_DISABLE_CONCURRENT_PROFILING': '0'}, clear=True), \
             mock.patch.object(pool.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'override'):
                pool.role(self.args, self.value)
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
