#!/usr/bin/env python3
"""Host-only fixtures for workload output, configuration and field admission."""
import array
import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest

import study_workloads as workloads

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / 'scripts/perf/suites/pace-a100.json'
BACKEND_SUITE = ROOT / 'scripts/perf/suites/pace-full-backends.json'
GPU_SOLVER_SUITE = ROOT / 'scripts/perf/suites/pace-full-gpu-solvers.json'
GPU = {'uuid': 'GPU-00112233-4455-6677-8899-aabbccddeeff', 'name': 'NVIDIA A100-SXM4-80GB',
       'compute_capability': 80}


def hdg_record(case):
    p = workloads.parameters(case)
    samples = []
    for index in range(p['measured_steps'] + 1):
        sample = {key: 0 for key in workloads.COUNTS}
        sample.update({'phase': 'warmup' if index == 0 else 'measured', 'measured_index': index - 1,
                       'step': index + 1, 'generation': index + 1, 'time': (index + 1) * p['dt'],
                       'trace_solves': 2, 'duration_seconds': 0.01,
                       'tensor_core': {'fallback_count': 0, 'condensation_launches': 2 if p['engine'] == 'tensor' else 0},
                       'gates': {key: True for key in workloads.GATES},
                       'committed_interior_hash': '1234567890abcdef' * 4, 'committed_trace_hash': 'fedcba0987654321' * 4,
                       'exported_interior_checksum_fnv1a64': '0123456789abcdef',
                       'final_residual': 1e-12, 'true_relative_residual': 1e-12,
                       'maximum_conservation_error': 0, 'maximum_conservation_relative_error': 0})
        samples.append(sample)
    policy = {key: 'fp64' for key in ('external_state', 'local_lu', 'local_triangular_solve',
                                     'tensor_operands', 'tensor_accumulation', 'acceptance_gates')}
    policy.update({key: False for key in ('mixed_precision_enabled', 'iterative_refinement_enabled',
                                        'residual_replacement_enabled', 'precision_correction_enabled')})
    return {'schema': 2, 'benchmark': 'dg_igr_hdg_cuda_accepted_step', 'success': True,
            'gpu_name': GPU['name'], 'gpu_uuid': GPU['uuid'], 'gpu_compute_capability': 80, 'rank_count': 1,
            'mode': p['mode'], 'cuda_engine': p['engine'], 'polynomial_order': p['order'], 'mesh_n': p['mesh_n'],
            'domain_periods': p['domain_periods'], 'dt': p['dt'], 'measured_steps': p['measured_steps'],
            'time_integrator': 'backward_euler', 'time_step_policy': 'fixed', 'precision': 'fp64',
            'requested_trace_policy': p['trace_policy'], 'resolved_trace_policy': 'production-resolved',
            'trace_solver': 'solver', 'trace_preconditioner': 'preconditioner', 'precision_policy': policy,
            'numerical_policy': {key: 1e-12 for key in ('nonlinear_relative_tolerance',
                'newton_correction_linear_relative_tolerance', 'final_coherent_linear_relative_tolerance',
                'true_residual_tightening_linear_relative_tolerance', 'true_residual_relative_tolerance',
                'conservation_relative_tolerance')},
            'warmup_steps': 1, 'samples': samples, 'raw_durations_seconds': [0.01] * p['measured_steps'],
            'median_seconds': 0.01, 'host_export': True, 'profiling_observability': False, 'profile_measured_step': -1,
            'timing_scope': 'existing-accepted-step-scope', 'build_identity': {'source_revision': '1' * 40,
                'source_tree': '2' * 40, 'source_status': 'clean', 'build_directory': '/build', 'cmake_cache': '/build/CMakeCache.txt'}}


def vtk_array(name, values, vtk_type='Float64', components=1):
    typecode = {'Float64': 'd', 'Int32': 'i', 'UInt8': 'B'}[vtk_type]
    payload = array.array(typecode, values).tobytes()
    encoded = base64.b64encode(len(payload).to_bytes(4, 'little')).decode() + base64.b64encode(payload).decode()
    return f'<DataArray Name="{name}" type="{vtk_type}" NumberOfComponents="{components}" format="binary">{encoded}</DataArray>'


def write_fields(directory, *, density=1.0, x=0.0, horizon=0.002, compressor=False):
    directory = directory / 'vtk'
    directory.mkdir(parents=True)
    fields = ''.join(vtk_array(name, [density if name == 'density' else 1.0]) for name in workloads.FIELDS)
    geometry = '<Points>' + vtk_array('points', [x, 0.0, 0.0], components=3) + '</Points><Cells>' + \
        vtk_array('connectivity', [0], 'Int32') + vtk_array('offsets', [1], 'Int32') + vtk_array('types', [1], 'UInt8') + '</Cells>'
    attr = ' compressor="vtkZLibDataCompressor"' if compressor else ''
    (directory / 'data.vtu').write_text(f'<VTKFile type="UnstructuredGrid" byte_order="LittleEndian" header_type="UInt32"{attr}><UnstructuredGrid><Piece NumberOfPoints="1" NumberOfCells="1"><PointData>{fields}</PointData>{geometry}</Piece></UnstructuredGrid></VTKFile>')
    (directory / 'series.pvd').write_text(f'<VTKFile type="Collection"><Collection><DataSet timestep="{horizon}" file="data.vtu"/></Collection></VTKFile>')


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)
        self.suite = workloads.load_suite(SUITE)
        self.dg = self.suite['cases'][0]
        self.hdg = next(case for case in self.suite['cases'] if case['adapter'] == 'hdg')
        self.volume = next(case for case in self.suite['cases'] if case['adapter'] == 'dgsem')
        self.stdout, self.stderr = self.work / 'stdout.log', self.work / 'stderr.log'
        self.stderr.write_text('')

    def tearDown(self):
        self.temp.cleanup()

    def parse_hdg(self, record):
        self.stdout.write_text(json.dumps(record) + '\n')
        return workloads.parse(self.hdg, self.stdout, self.stderr, GPU)

    def test_default_suite_is_lean_and_optional_probes_explicit(self):
        self.assertEqual(sum(case['enabled'] for case in self.suite['cases']), 6)
        self.assertEqual(sum(case['optional'] for case in self.suite['cases']), 2)
        self.assertEqual(self.suite['repetitions'], 3)
        for case in self.suite['cases']:
            if case['adapter'] == 'hdg':
                self.assertEqual(case['profile']['selection'], 'first-per-kernel')
                self.assertEqual(len(case['profile']['kernel_groups']), 5)
                self.assertLessEqual(case['profile']['launch_count'], 32)

    def test_kernel_group_selection_is_bounded_and_validated(self):
        for mutation in ('selection', 'skip', 'count', 'name', 'regex', 'too_many', 'newline'):
            suite = json.loads(SUITE.read_text())
            profile = suite['cases'][2]['profile']
            if mutation == 'selection': profile['selection'] = 'everything'
            elif mutation == 'skip': profile['launch_skip'] = 1
            elif mutation == 'count': profile['launch_count'] = 33
            elif mutation == 'name': profile['kernel_groups']['../other'] = 'kernel'
            elif mutation == 'regex': profile['kernel_groups']['volume'] = '['
            elif mutation == 'too_many': profile['kernel_groups'] = {f'group{i}': 'kernel' for i in range(13)}
            else: profile['kernel_groups']['volume'] = 'kernel\nother'
            path = self.work / 'invalid-suite.json'
            path.write_text(json.dumps(suite))
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, workloads.re.error)):
                workloads.load_suite(path)

    def test_suite_physical_budgets_and_work_count_limits(self):
        suite = json.loads(SUITE.read_text())
        suite.update(repetitions=31, warmups=5, timings_budget_seconds=3000,
                     profiles_budget_seconds=0.25, command_timeout_seconds=600.0)
        suite['cases'] += [{**copy.deepcopy(suite['cases'][0]), 'id': f'extra-{i}'} for i in range(8)]
        path = self.work / 'suite.json'
        path.write_text(json.dumps(suite))
        admitted = workloads.load_suite(path)
        self.assertEqual(len(admitted['cases']), 16)
        self.assertEqual(admitted['profiles_budget_seconds'], 0.25)
        invalid = [('repetitions', 32), ('repetitions', 3.0), ('warmups', 6), ('warmups', True),
                   ('timings_budget_seconds', 3001), ('profiles_budget_seconds', 3000.1),
                   ('command_timeout_seconds', 600.01), ('timings_budget_seconds', 0),
                   ('profiles_budget_seconds', -1), ('command_timeout_seconds', True),
                   ('timings_budget_seconds', '30'), ('profiles_budget_seconds', float('inf')),
                   ('command_timeout_seconds', float('nan')), ('unknown_budget', 100)]
        for key, value in invalid:
            changed = copy.deepcopy(suite)
            changed[key] = value
            path.write_text(json.dumps(changed))
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                workloads.load_suite(path)
        suite['cases'].append({**copy.deepcopy(suite['cases'][0]), 'id': 'extra-final'})
        path.write_text(json.dumps(suite))
        with self.assertRaisesRegex(ValueError, '1..16 cases'):
            workloads.load_suite(path)

    def test_invalid_suite_ids_repetitions_and_profile_overrides(self):
        for mutation in ('duplicate', 'escape', 'repetitions', 'sections', 'bool_count'):
            suite = json.loads(SUITE.read_text())
            if mutation == 'duplicate': suite['cases'][1]['id'] = suite['cases'][0]['id']
            elif mutation == 'escape': suite['cases'][0]['id'] = '../other'
            elif mutation == 'repetitions': suite['repetitions'] = 2
            elif mutation == 'sections': suite['cases'][0]['profile']['sections'] = ['ignored']
            else: suite['cases'][0]['profile']['launch_count'] = True
            path = self.work / 'suite.json'
            path.write_text(json.dumps(suite))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                workloads.load_suite(path)

    def test_argv_keeps_field_and_profile_calls_separate(self):
        argv = workloads.command(self.hdg, Path('/binary'), ROOT, self.work, profile=True)
        self.assertEqual(argv[-2:], ['--profile-measured-step', '0'])
        self.assertIn('--no-export', argv)
        self.assertNotIn('mpiexec', argv)
        dg = workloads.command(self.dg, Path('/binary'), ROOT, self.work, fields=True)
        self.assertIn('-vtk', dg)
        self.assertEqual(workloads.environment(self.dg, self.work, fields=True),
                         {'DG_IGR_VTK_PREFIX_PATH': str(self.work / 'vtk')})
        with self.assertRaises(ValueError):
            workloads.command(self.hdg, Path('/binary'), ROOT, self.work, profile=True, fields=True)

    def test_hdg_complete_finite_record_and_configurable_measured_steps(self):
        self.hdg['parameters']['measured_steps'] = 3
        value = self.parse_hdg(hdg_record(self.hdg))
        self.assertEqual(value['seconds'], 0.01)
        self.assertEqual(len(value['checksums']), 4)
        self.assertEqual(value['gpu_uuid'], GPU['uuid'])
        self.assertEqual(value['profile']['expected_condensation_launches'], 2)

    def test_hdg_trace_configuration_is_part_of_pair_signature(self):
        record = hdg_record(self.hdg)
        record['trace_ilu_type'] = 30
        original = self.parse_hdg(record)['signature']
        record['trace_ilu_type'] = 0
        self.assertNotEqual(original, self.parse_hdg(record)['signature'])
        record['trace_sigma_jacobi_damping'] = float('nan')
        with self.assertRaises(ValueError): self.parse_hdg(record)

    def test_hdg_sha256_and_fnv_checksum_widths_are_distinct(self):
        record = hdg_record(self.hdg)
        value = self.parse_hdg(record)
        self.assertEqual(len(value['checksums'][0]['committed_interior_hash']), 64)
        self.assertEqual(len(value['checksums'][0]['committed_trace_hash']), 64)
        self.assertEqual(len(value['checksums'][0]['exported_interior_checksum_fnv1a64']), 16)
        for key, invalid in (('committed_interior_hash', 'a' * 16),
                             ('committed_trace_hash', 'b' * 16),
                             ('exported_interior_checksum_fnv1a64', 'c' * 64),
                             ('committed_interior_hash', 'g' * 64)):
            changed = copy.deepcopy(record)
            changed['samples'][0][key] = invalid
            with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                self.parse_hdg(changed)

    def test_hdg_failure_gate_device_policy_and_duration_rejected(self):
        for kind in ('gate', 'device', 'policy', 'retry', 'duration', 'steps', 'median', 'fallback'):
            record = hdg_record(self.hdg)
            if kind == 'gate': record['samples'][0]['gates']['conservation'] = False
            elif kind == 'device': record['gpu_uuid'] = 'GPU-other'
            elif kind == 'policy': record['precision_policy']['mixed_precision_enabled'] = True
            elif kind == 'retry': record['samples'][1]['retry_index'] = 1
            elif kind == 'duration': record['raw_durations_seconds'][0] = float('nan')
            elif kind == 'steps': record['samples'].pop()
            elif kind == 'median': record['median_seconds'] = 2.0
            else: record['samples'][1]['tensor_core']['fallback_count'] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.parse_hdg(record)

    def test_hdg_duplicate_summary_and_cleanup_failure_rejected(self):
        record = json.dumps(hdg_record(self.hdg)) + '\n'
        self.stdout.write_text(record * 2)
        with self.assertRaises(ValueError): workloads.parse(self.hdg, self.stdout, self.stderr, GPU)
        self.stdout.write_text(record)
        self.stderr.write_text('{"schema":2,"success":false,"error":"CUDA device reset failed"}\n')
        with self.assertRaises(ValueError): workloads.parse(self.hdg, self.stdout, self.stderr, GPU)

    def dg_log(self, steps=200, final=0.002, success=1):
        return ('done, 0.5s.\n[explicit-time] event=run-summary '
                f'success={success} termination_reason=target-reached final_time={final} steps={steps} '
                f'accepted_attempts={steps} rejected_attempts=0 rhs_evaluations=800 '
                'cfl_control_enabled=0 dt_min_reached=0\n')

    def test_dg_fixed_horizon_and_short_or_failed_run(self):
        self.stdout.write_text(self.dg_log())
        self.assertEqual(workloads.parse(self.dg, self.stdout, self.stderr, GPU)['seconds'], 0.5)
        for log in (self.dg_log(steps=20), self.dg_log(final=0.001), self.dg_log(success=0)):
            self.stdout.write_text(log)
            with self.assertRaises(ValueError): workloads.parse(self.dg, self.stdout, self.stderr, GPU)

    def test_legacy_dg_command_and_environment_preserved(self):
        self.assertEqual(workloads.command(self.dg, Path('/binary'), ROOT, self.work),
            ['/binary', '-m', str(ROOT / workloads.DG_MESH), '--dg-backend', 'dg', '-o', '1',
             '-rs', '0', '-rp', '0', '-tf', '0.002', '-dt', '1e-05', '-c', '0', '-no-glvis',
             '-vs', '1000000', '-nan-scan', '--device', 'cuda', '-no-save-state', '-no-vtk'])
        self.assertEqual(workloads.environment(self.dg, self.work), {})

    def test_serial_refine_is_bounded_and_repeated_for_every_application_call(self):
        for refinement in (0, 1, 2):
            case = copy.deepcopy(self.dg)
            case['parameters']['serial_refine'] = refinement
            for options in ({}, {'profile': True}, {'fields': True}):
                argv = workloads.command(case, Path('/binary'), ROOT, self.work, **options)
                with self.subTest(refinement=refinement, options=options):
                    self.assertEqual(argv[argv.index('-rs') + 1], str(refinement))
                    self.assertEqual(argv[argv.index('-rp') + 1], '0')
                    self.assertEqual(argv[argv.index('-tf') + 1], '0.002')
                    self.assertEqual(argv[argv.index('-dt') + 1], '1e-05')
                    self.assertIn('-nan-scan', argv)
        for refinement in (-1, 3, True, False, 1.0, '1', float('nan')):
            case = copy.deepcopy(self.dg)
            case['parameters']['serial_refine'] = refinement
            with self.subTest(refinement=refinement), self.assertRaises(ValueError):
                workloads.command(case, Path('/binary'), ROOT, self.work)

    def test_refinement_changes_work_signature_and_mesh_still_has_field_gate(self):
        self.stdout.write_text(self.dg_log())
        original = workloads.parse(self.dg, self.stdout, self.stderr, GPU)['signature']
        refined = copy.deepcopy(self.dg)
        refined['parameters']['serial_refine'] = 1
        self.assertNotEqual(original, workloads.parse(refined, self.stdout, self.stderr, GPU)['signature'])
        left, right = self.work / 'left', self.work / 'right'
        write_fields(left)
        write_fields(right, x=0.25)
        with self.assertRaisesRegex(ValueError, 'geometry/topology differs'):
            workloads.compare_fields(refined, left, right, ROOT)

    def test_gpu_solver_companion_has_six_full_application_cases_and_larger_meshes(self):
        suite = workloads.load_suite(GPU_SOLVER_SUITE)
        small = workloads.load_suite(BACKEND_SUITE)
        self.assertEqual(suite['correctness_regex'], small['correctness_regex'])
        self.assertEqual((suite['repetitions'], suite['warmups']), (3, 1))
        self.assertEqual(len(suite['cases']), 6)
        self.assertEqual({case['id'] for case in suite['cases']},
                         {'dg-cuda-p3', 'dgsem-cuda-p3', 'dgsem-fused-simt-p3',
                          'dgsem-fused-dmma-p3', 'hdg-cuda-p3', 'hdg-tensor-p3'})
        for case in suite['cases']:
            p = case['parameters']
            with self.subTest(case=case['id']):
                self.assertEqual(case['adapter'], 'dg')
                self.assertTrue(case['enabled'])
                self.assertEqual((p['order'], p['device'], p['dt']), (3, 'cuda', 1e-5))
                lines = [line.strip() for line in (ROOT / p['mesh']).read_text().splitlines()]
                cells = int(lines[lines.index('elements') + 1]) * 4 ** p['serial_refine']
                self.assertEqual(cells, 256 if p['discretization'] == 'hdg' else 1024)
                self.assertEqual(p['final_time'], 5e-5 if p['discretization'] == 'hdg' else 0.002)
                self.assertEqual(case['profile']['selection'], 'first-per-kernel')
                self.assertEqual(case['profile']['timeout'], 240)
                self.assertTrue(case['profile']['kernel_groups'])
                argv = workloads.command(case, Path('/binary'), ROOT, self.work, profile=True)
                self.assertNotIn('--profile-measured-step', argv)
                self.assertEqual(argv[argv.index('-rs') + 1], str(p['serial_refine']))

    def test_full_backend_suite_pins_small_same_mesh_horizon(self):
        suite = workloads.load_suite(BACKEND_SUITE)
        self.assertEqual(len(suite['cases']), 9)
        self.assertEqual(suite['correctness_regex'], self.suite['correctness_regex'])
        shared = {(case['parameters']['order'], case['parameters']['mesh'],
                   case['parameters']['dt'], case['parameters']['final_time']) for case in suite['cases']}
        self.assertEqual(len(shared), 1)
        self.assertEqual(sum(c['parameters']['device'] == 'cpu' for c in suite['cases']), 3)
        for case in suite['cases']:
            p = case['parameters']
            self.assertTrue((ROOT / p['mesh']).is_file())
            argv = workloads.command(case, Path('/binary'), ROOT, self.work)
            self.assertEqual(argv[argv.index('--device') + 1], p['device'])
            if p['discretization'] == 'hdg':
                self.assertNotIn('--dg-backend', argv)
                self.assertIn('backward-euler', argv)
            if p['device'] == 'cuda':
                self.assertEqual(case['profile']['selection'], 'first-per-kernel')
                self.assertTrue(case['profile']['kernel_groups'])

    def test_application_unsupported_backend_combinations_rejected(self):
        for change in ({'engine': 'tensor'}, {'discretization': 'hdg', 'device': 'cpu', 'engine': 'tensor'},
                       {'dgsem_residual': 'fused-dmma'},
                       {'discretization': 'dgsem', 'device': 'cpu', 'dgsem_residual': 'fused-simt'},
                       {'discretization': 'dgsem', 'order': 5}, {'discretization': 'hdg', 'order': 4}):
            case = copy.deepcopy(self.dg)
            case['parameters'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError): workloads.parameters(case)

    def test_cpu_dg_admission_does_not_require_gpu_identity(self):
        case = copy.deepcopy(self.dg)
        case['parameters']['device'] = 'cpu'
        self.stdout.write_text('[device] requested=cpu\n' + self.dg_log())
        result = workloads.parse(case, self.stdout, self.stderr, None)
        self.assertIn('CPU', result['scope'])
        self.assertEqual(workloads.environment(case, self.work)['OMP_NUM_THREADS'], '1')
        self.stdout.write_text('[device] requested=cuda\n' + self.dg_log())
        with self.assertRaisesRegex(ValueError, 'device'): workloads.parse(case, self.stdout, self.stderr, None)

    def test_full_dgsem_native_selection_and_fused_precision(self):
        case = copy.deepcopy(self.dg)
        case['parameters'].update(discretization='dgsem', dgsem_residual='fused-dmma')
        ordinary = ('[device] requested=cuda\n'
                    '[dgsem-production] selected=mfem-native-packed strict=1 backend=cuda order=1 dim=2 volume_dissipation=none\n')
        fused = '[dgsem-fused-residual] enabled=1 precision=fp64 contraction=dmma8x8x4 order=1\n'
        self.stdout.write_text(ordinary + fused + self.dg_log())
        self.assertIn('fused-dmma', workloads.parse(case, self.stdout, self.stderr, GPU)['scope'])
        env = workloads.environment(case, self.work)
        for prefix in ('IGRLAB_', 'DG_IGR_'):
            self.assertEqual(env[prefix + 'DGSEM_FUSED_DMMA'], '1')
            self.assertEqual(env[prefix + 'DGSEM_SUBCELL_BLEND'], '0')
            self.assertEqual(env[prefix + 'DGSEM_VOLUME_QUADRATURE_MIN_ORDER'], '0')
        for invalid in (ordinary.replace('strict=1', 'strict=0') + fused,
                        ordinary.replace('backend=cuda', 'backend=cpu') + fused,
                        ordinary + fused.replace('dmma8x8x4', 'simt'), ordinary + fused * 2, ordinary):
            self.stdout.write_text(invalid + self.dg_log())
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                workloads.parse(case, self.stdout, self.stderr, GPU)
        case['parameters'].update(device='cpu', dgsem_residual='unfused')
        self.stdout.write_text(ordinary.replace('cuda', 'cpu') + self.dg_log())
        self.assertIn('CPU DGSEM', workloads.parse(case, self.stdout, self.stderr, None)['scope'])
        self.stdout.write_text(ordinary.replace('cuda', 'cpu') + fused + self.dg_log())
        with self.assertRaisesRegex(ValueError, 'unexpected fused'):
            workloads.parse(case, self.stdout, self.stderr, None)

    def hdg_application_log(self, device='cuda', engine='tensor', *, step=200, final=0.002, cycle=0):
        selected = '[device] requested=cuda hdg_cuda_engine=' + engine + '\n' if device == 'cuda' else ''
        engine_fields = (' cuda_engine=' + engine + ' fallback_count=0 requested_trace_policy=engine-default'
                         ' resolved_trace_policy=engine-default trace_solver=native trace_preconditioner=ilu0'
                         ' trace_ilu_type=0') if device == 'cuda' else ''
        return (selected + f'[hdg-main] status=Success step={step} time={final} output_cycle={cycle}' +
                engine_fields + ' time_treatment=backward-euler regularization=none positivity_limiter=none'
                ' minimum_accepted_dt=1e-05 maximum_accepted_dt=1e-05\n')

    def test_generic_hdg_application_wall_horizon_engine_and_export_signature(self):
        case = copy.deepcopy(self.dg)
        case['parameters'].update(discretization='hdg', engine='tensor')
        self.stdout.write_text(self.hdg_application_log())
        result = workloads.parse(case, self.stdout, self.stderr, GPU, wall_seconds=0.75)
        self.assertEqual(result['seconds'], 0.75)
        self.assertIn('startup', result['scope'])
        self.assertIn('no benchmark per-step', result['gate_scope'])
        self.stdout.write_text(self.hdg_application_log(cycle=1))
        self.assertEqual(result['signature'], workloads.parse(case, self.stdout, self.stderr, GPU,
                                                             wall_seconds=1.25)['signature'])
        for invalid in (self.hdg_application_log(step=199), self.hdg_application_log(final=0.001),
                        self.hdg_application_log(engine='ordinary'), self.hdg_application_log() * 2,
                        self.hdg_application_log().replace('fallback_count=0', 'fallback_count=1'),
                        self.hdg_application_log().replace('maximum_accepted_dt=1e-05', 'maximum_accepted_dt=2e-05'),
                        self.hdg_application_log().replace('regularization=none', 'regularization=igr'),
                        self.hdg_application_log() + '[hdg-main] status=Rejected reason=test\n'):
            self.stdout.write_text(invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                workloads.parse(case, self.stdout, self.stderr, GPU, wall_seconds=0.75)
        self.stdout.write_text(self.hdg_application_log())
        for seconds in (None, 0, float('nan'), -1):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                workloads.parse(case, self.stdout, self.stderr, GPU, wall_seconds=seconds)
        case['parameters'].update(device='cpu', engine='ordinary')
        self.stdout.write_text(self.hdg_application_log(device='cpu'))
        self.assertIn('CPU HDG', workloads.parse(case, self.stdout, self.stderr, None,
                                                wall_seconds=0.5)['scope'])
        self.stdout.write_text(self.hdg_application_log())
        with self.assertRaisesRegex(ValueError, 'CPU HDG'):
            workloads.parse(case, self.stdout, self.stderr, None, wall_seconds=0.5)

    def test_dgsem_complete_methods_errors_are_observations(self):
        rows = [{'device': GPU['name'], 'compute_major': 8, 'compute_minor': 0}]
        p = self.volume['parameters']
        for method in workloads.METHODS:
            rows.append({'probe': 'dgsem-central-volume', 'method': method, 'n1d': p['n'],
                'elements': p['elements'], 'iterations': p['iterations'], 'gpu_seconds': 0.01,
                'setup_seconds': 0, 'warmup_wall_seconds': 0, 'pack_seconds': 0, 'multiply_seconds': 0,
                'unpack_seconds': 0, 'relative_l2_error': 0.1, 'maximum_absolute_error': 1.0,
                'relative_component_sum_error': 0.2, 'scope': 'central-volume-only-prescribed-sigma',
                'reference': 'long-double-same-operator-on-FP64-flux'})
        self.stdout.write_text('\n'.join(json.dumps(row) for row in rows))
        result = workloads.parse(self.volume, self.stdout, self.stderr, GPU)
        self.assertEqual(len(result['errors']), 14)
        self.assertFalse(result['error_acceptance_thresholds_added'])
        rows.pop()
        self.stdout.write_text('\n'.join(json.dumps(row) for row in rows))
        with self.assertRaises(ValueError): workloads.parse(self.volume, self.stdout, self.stderr, GPU)

    def test_dg_five_fields_and_ordered_geometry(self):
        a, b = self.work / 'a', self.work / 'b'
        write_fields(a)
        write_fields(b)
        result = workloads.compare_fields(self.dg, a, b, ROOT)
        self.assertTrue(result['passed'])
        self.assertEqual(set(result['arrays']), set(workloads.FIELDS))
        write_fields(self.work / 'c', x=1.0)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            workloads.compare_fields(self.dg, a, self.work / 'c', ROOT)

    def test_dg_field_nonpositive_tolerance_horizon_and_encoding_fail_closed(self):
        a = self.work / 'a'
        write_fields(a)
        for kind in ('nonpositive', 'tolerance', 'horizon', 'compressed'):
            b = self.work / kind
            write_fields(b, density=0 if kind == 'nonpositive' else 2 if kind == 'tolerance' else 1,
                         horizon=0.001 if kind == 'horizon' else 0.002, compressor=kind == 'compressed')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                workloads.compare_fields(self.dg, a, b, ROOT)


if __name__ == '__main__':
    unittest.main()
