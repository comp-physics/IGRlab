#!/usr/bin/env python3
"""Workload contracts for bounded studies of existing rank-one binaries.

This module builds argv and validates existing application output. Scheduling,
dependency builds, timing pairs, profiler execution and stage recovery belong
to the study coordinator. DGSEM probes report kernel-only error observations;
they do not establish application accuracy or time to solution.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys
import xml.etree.ElementTree as ET

BINARY_NAMES = {
    'dg': 'dgsem_smooth_advection_periodic',
    'hdg': 'dg_igr_hdg_runtime_cuda_accepted_step_benchmark',
    'dgsem': 'dg_igr_dgsem_tensor_volume_benchmark',
}
FIELDS = ('density', 'momentum', 'energy', 'pressure', 'velocity')
GATES = ('candidate_ready', 'finite', 'admissible', 'recovery', 'true_residual',
         'conservation', 'telemetry', 'products_coupled_and_ordered', 'success')
COUNTS = ('step', 'generation', 'time', 'attempt_sequence', 'retry_index',
          'nonlinear_updates', 'trace_solves', 'linear_iterations',
          'linear_solve_attempts', 'complete_assemblies', 'complete_condensations', 'complete_recoveries')
METHODS = ('existing-fp64', 'cooperative-fp64', 'cooperative-fp32',
           'cooperative-matrix-fp64', 'fused-lines-fp64-wmma', 'cooperative-matrix-fp32',
           'fused-lines-tf32', 'fused-lines-tf32x3', 'fused-lines-fp16', 'fused-lines-fp16x3',
           'factorized-lines-tf32', 'factorized-lines-tf32x3', 'dense2d-tf32', 'dense2d-tf32x3')
DG_MESH = 'examples/dgsem_smooth_advection_periodic/smooth_advection_periodic.msh'
RTOL, ATOL = 5e-8, 5e-9


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and
            (value > 0 if positive else value >= 0), f'invalid {name}')
    return value


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f'invalid {name}')
    return value


def finite_tree(value, name):
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f'{name}.{key}')
    elif isinstance(value, list):
        for item in value:
            finite_tree(item, name)
    elif type(value) in (int, float):
        require(math.isfinite(value), f'nonfinite {name}')


def parameters(case):
    adapter, supplied = case['adapter'], case['parameters']
    defaults = {
        'dg': {'order': 1, 'mesh': DG_MESH, 'final_time': 0.002, 'dt': 1e-5,
               'rtol': RTOL, 'atol': ATOL, 'discretization': 'dg', 'device': 'cuda',
               'engine': 'ordinary', 'dgsem_residual': 'unfused'},
        'hdg': {'mode': 'igr', 'engine': 'ordinary', 'order': 3, 'mesh_n': 16,
                'domain_periods': 1, 'dt': 1e-5, 'measured_steps': 5, 'trace_policy': 'engine-default'},
        'dgsem': {'n': 4, 'elements': 1024, 'iterations': 100, 'method': 'existing-fp64'},
    }
    require(adapter in defaults and isinstance(supplied, dict), 'unknown adapter or invalid parameters')
    require(set(supplied) <= set(defaults[adapter]), f'unknown {adapter} parameter')
    result = {**defaults[adapter], **supplied}
    if adapter in ('dg', 'hdg'):
        require(integer(result['order'], 'order', 1) <= (8 if adapter == 'dg' else 3), 'unsupported order')
        number(result['dt'], 'dt', positive=True)
    if adapter == 'dg':
        require(result['discretization'] in ('dg', 'dgsem', 'hdg') and
                result['device'] in ('cpu', 'cuda') and result['engine'] in ('ordinary', 'tensor'),
                'unsupported application discretization/device/engine')
        require(result['engine'] == 'ordinary' or
                (result['discretization'] == 'hdg' and result['device'] == 'cuda'),
                'tensor engine requires CUDA HDG; DGSEM uses its explicit residual selection')
        require(result['dgsem_residual'] in ('unfused', 'fused-simt', 'fused-dmma') and
                (result['dgsem_residual'] == 'unfused' or
                 (result['discretization'] == 'dgsem' and result['device'] == 'cuda')),
                'fused residual requires CUDA DGSEM')
        require(result['discretization'] != 'dgsem' or result['order'] <= 4,
                'application study uses qualified DGSEM orders p1..p4')
        require(result['discretization'] != 'hdg' or result['order'] <= 3,
                'application study uses supported HDG orders p1..p3')
        number(result['final_time'], 'final_time', positive=True)
        require(result['dt'] <= result['final_time'], 'dt exceeds DG horizon')
        number(result['rtol'], 'rtol')
        number(result['atol'], 'atol')
        mesh = Path(result['mesh']) if isinstance(result['mesh'], str) else Path('/')
        require(isinstance(result['mesh'], str) and result['mesh'] and not mesh.is_absolute() and
                '..' not in mesh.parts, 'DG mesh must be a confined source-relative path')
    elif adapter == 'hdg':
        require(result['mode'] in ('conventional', 'igr') and result['engine'] in ('ordinary', 'tensor'),
                'unsupported HDG mode/engine')
        require(isinstance(result['trace_policy'], str) and result['trace_policy'] == 'engine-default',
                'this study adapter requires the production engine-default trace policy')
        integer(result['mesh_n'], 'mesh_n', 1)
        periods = integer(result['domain_periods'], 'domain_periods', 1)
        require(periods <= result['mesh_n'] and result['mesh_n'] % periods == 0, 'invalid domain replication')
        require(integer(result['measured_steps'], 'measured_steps', 1) <= 64, 'unbounded measured steps')
    else:
        require(result['n'] in (2, 4, 8, 16) and type(result['n']) is int, 'unsupported complete-method DGSEM point')
        require(integer(result['elements'], 'elements', 1) <= 32768 and
                integer(result['iterations'], 'iterations', 1) <= 10000, 'DGSEM bounded probe limit')
        require(result['method'] in METHODS, 'unknown DGSEM timing method')
    return result


def load_suite(path):
    suite = json.loads(Path(path).read_text())
    require(isinstance(suite, dict) and type(suite.get('schema_version')) is int and
            suite['schema_version'] == 1, 'suite requires schema_version=1')
    require(set(suite) <= {'schema_version', 'name', 'repetitions', 'warmups', 'correctness_regex', 'cases',
                           'timings_budget_seconds', 'profiles_budget_seconds', 'command_timeout_seconds'},
            'unknown suite field')
    if 'name' in suite:
        require(isinstance(suite['name'], str) and suite['name'] and '\n' not in suite['name'] and
                '\r' not in suite['name'], 'invalid suite name')
    require(integer(suite.get('repetitions'), 'repetitions', 3) <= 31, 'suite repetitions must be 3..31')
    require(integer(suite.get('warmups'), 'warmups', 1) <= 5, 'suite warmups must be 1..5')
    for field, maximum in (('timings_budget_seconds', 3000), ('profiles_budget_seconds', 3000),
                           ('command_timeout_seconds', 600)):
        if field in suite:
            require(number(suite[field], field, positive=True) <= maximum,
                    f'{field} exceeds {maximum} seconds')
    regex = suite.get('correctness_regex')
    require(isinstance(regex, str) and regex and '\n' not in regex and '\r' not in regex,
            'suite requires a single-line correctness_regex')
    re.compile(regex)
    cases = suite.get('cases')
    require(isinstance(cases, list) and 1 <= len(cases) <= 16, 'suite requires 1..16 cases')
    seen = set()
    for case in cases:
        require(isinstance(case, dict), 'invalid suite case')
        identity = case.get('id')
        require(isinstance(identity, str) and re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', identity) and
                identity not in seen, 'case IDs must be unique safe slugs')
        seen.add(identity)
        require(case.get('adapter') in BINARY_NAMES and isinstance(case.get('parameters'), dict),
                'case requires supported adapter and parameters')
        case['parameters'] = parameters(case)
        expected_binary = BINARY_NAMES[case['adapter']]
        require(case.get('binary', expected_binary) == expected_binary, 'adapter binary contract differs')
        case['binary'] = expected_binary
        for field in ('enabled', 'optional'):
            require(type(case.get(field, field == 'enabled')) is bool, f'invalid {field}')
        case.setdefault('enabled', True)
        case.setdefault('optional', False)
        profile = case.get('profile')
        require(isinstance(profile, dict) and
                {'kernel_regex', 'launch_skip', 'launch_count'} <= set(profile), 'case profile contract missing')
        require(set(profile) <= {'kernel_regex', 'launch_skip', 'launch_count', 'ncu', 'nsys',
                                'timeout', 'max_metric_rows', 'selection', 'kernel_groups'},
                'unknown profile parameter; native provider owns section/metric selections')
        require(isinstance(profile['kernel_regex'], str) and profile['kernel_regex'] and
                '\n' not in profile['kernel_regex'] and '\r' not in profile['kernel_regex'],
                'invalid profile kernel_regex')
        re.compile(profile['kernel_regex'])
        require(integer(profile['launch_skip'], 'launch_skip') <= 10000, 'unbounded profile launch_skip')
        require(integer(profile['launch_count'], 'launch_count', 1) <= 32, 'unbounded profile launch_count')
        selection = profile.get('selection', 'launches')
        require(selection in ('launches', 'first-per-kernel'), 'unknown profile selection')
        require(selection != 'first-per-kernel' or profile['launch_skip'] == 0,
                'first-per-kernel selection requires zero launch_skip')
        groups = profile.get('kernel_groups', {})
        require(isinstance(groups, dict) and len(groups) <= 12, 'invalid/unbounded profile kernel_groups')
        for name, regex in groups.items():
            require(isinstance(name, str) and re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', name) and
                    isinstance(regex, str) and 0 < len(regex) <= 2048 and '\n' not in regex and '\r' not in regex,
                    'invalid profile kernel group')
            re.compile(regex)
        if 'timeout' in profile:
            require(number(profile['timeout'], 'profile timeout', positive=True) <= 240, 'unbounded profile timeout')
        if 'max_metric_rows' in profile:
            require(integer(profile['max_metric_rows'], 'max_metric_rows', 1) <= 120, 'unbounded profile metric rows')
        for field in ('ncu', 'nsys'):
            if field in profile:
                require(isinstance(profile[field], str) and profile[field], 'invalid profiler executable')
    require(any(case['enabled'] for case in cases), 'suite has no enabled cases')
    return suite


def command(case, binary: Path, source: Path, workdir: Path, *, profile=False, fields=False):
    p, adapter = parameters(case), case['adapter']
    require(not (profile and fields), 'profile and field export are separate calls')
    if adapter == 'dg':
        selection = (['--discretization', 'hdg', '--hdg-cuda-engine', p['engine'],
                      '--time-integrator', 'backward-euler', '--regularization', 'none',
                      '--positivity-limiter', 'none'] if p['discretization'] == 'hdg' else
                     ['--dg-backend', p['discretization']])
        if p['discretization'] == 'dgsem':
            selection += ['--discretization', 'dgsem', '--regularization', 'none',
                          '--dgsem-stab', 'none', '--dgsem-volume-flux', 'central',
                          '--dgsem-volume-dissipation', 'none', '--dgsem-surface-flux', 'rusanov']
        argv = [str(binary), '-m', str(source / p['mesh']), *selection, '-o', str(p['order']),
                '-rs', '0', '-rp', '0', '-tf', str(p['final_time']), '-dt', str(p['dt']), '-c', '0',
                '-no-glvis', '-vs', '1000000', '-nan-scan', '--device', p['device'], '-no-save-state',
                '-vtk' if fields else '-no-vtk']
        if fields:
            argv += ['-dt_save', str(p['final_time']), '-no-vtk-h1', '-no-vtk-l2', '-no-vtk-rank', '-no-vtk-zs']
        return argv
    if adapter == 'hdg':
        argv = [str(binary), '--mode', p['mode'], '--engine', p['engine'], '--trace-policy', p['trace_policy'],
                '--order', str(p['order']), '--mesh-n', str(p['mesh_n']), '--domain-periods', str(p['domain_periods']),
                '--dt', str(p['dt']), '--measured-steps', str(p['measured_steps']), '--export' if fields else '--no-export']
        return argv + (['--profile-measured-step', '0'] if profile else [])
    require(not fields, 'DGSEM probe has no physical field export')
    return [str(binary), '--n', str(p['n']), '--elements', str(p['elements']), '--iterations', str(p['iterations'])]


def environment(case, workdir, fields=False):
    p = parameters(case)
    result = {'DG_IGR_VTK_PREFIX_PATH': str(Path(workdir) / 'vtk')} if case['adapter'] == 'dg' and fields else {}
    if case['adapter'] == 'dg' and p['device'] == 'cpu':
        result['OMP_NUM_THREADS'] = '1'
    if case['adapter'] == 'dg' and p['discretization'] == 'dgsem':
        controls = {'DGSEM_PRODUCTION_STRICT': '1', 'DGSEM_HIGH_ORDER_RESEARCH': '0',
                    'DGSEM_VOLUME_QUADRATURE_MIN_ORDER': '0', 'DGSEM_SUBCELL_BLEND': '0',
                    'DGSEM_FUSED_RESIDUAL': '0' if p['dgsem_residual'] == 'unfused' else '1',
                    'DGSEM_FUSED_DMMA': '1' if p['dgsem_residual'] == 'fused-dmma' else '0'}
        # Canonical names take precedence, including inherited values. Pin both
        # aliases so the same frozen case works with historical application code.
        result.update({prefix + name: value for name, value in controls.items()
                       for prefix in ('IGRLAB_', 'DG_IGR_')})
    return result


def expected_device(expected_gpu):
    require(isinstance(expected_gpu, dict), 'missing allocated GPU identity')
    name = expected_gpu.get('name', expected_gpu.get('gpu_name'))
    capability = expected_gpu.get('compute_capability', expected_gpu.get('gpu_compute_capability',
                                  expected_gpu.get('compute_cap')))
    require(isinstance(name, str) and 'A100' in name and capability in (80, '80', '8.0', 8.0),
            'workload requires the allocated native A100 identity')
    return expected_gpu.get('uuid', expected_gpu.get('gpu_uuid'))


def json_rows(text):
    rows = []
    for line in text.splitlines():
        if line.startswith('{'):
            value = json.loads(line)
            require(isinstance(value, dict), 'unexpected JSON output record')
            rows.append(value)
    return rows


def parse_hdg_application(p, text, wall_seconds):
    """Admit the existing generic application summary, not benchmark schema 2."""
    require('[hdg-main] status=Rejected' not in text, 'HDG application rejected the workload')
    rows = [dict(re.findall(r'(\w+)=([^ ]+)', line)) for line in text.splitlines()
            if line.startswith('[hdg-main] status=Success ')]
    require(len(rows) == 1, 'expected one successful HDG application summary')
    row = rows[0]
    steps = integer(int(row['step']), 'HDG application steps', 1)
    nominal = math.ceil(p['final_time'] / p['dt'] - 1e-10)
    final = number(float(row['time']), 'HDG application final time')
    require(steps == nominal and abs(final - p['final_time']) <= max(1e-14, p['final_time'] * 1e-12),
            'HDG application accepted horizon differs')
    require(row.get('time_treatment') == 'backward-euler' and row.get('regularization') == 'none' and
            row.get('positivity_limiter') == 'none' and int(row.get('rejected_attempts', '0')) == 0,
            'HDG application numerical policy differs')
    for name in ('minimum_accepted_dt', 'maximum_accepted_dt'):
        dt = number(float(row[name]), name, positive=True)
        require(abs(dt - p['dt']) <= max(1e-15, p['dt'] * 1e-12), 'HDG accepted timestep differs')
    if p['device'] == 'cuda':
        require(re.findall(r'^\[device\] requested=([^\s]+)', text, re.M) == ['cuda'] and
                row.get('cuda_engine') == p['engine'] and row.get('fallback_count') == '0',
                'HDG application CUDA engine/fallback differs')
        require(all(row.get(key) for key in ('requested_trace_policy', 'resolved_trace_policy',
                                            'trace_solver', 'trace_preconditioner')),
                'HDG application trace policy missing')
    else:
        require('cuda_engine' not in row and not re.search(r'^\[device\] requested=cuda\b', text, re.M),
                'CPU HDG application selected CUDA')
    seconds = number(wall_seconds, 'HDG application runner wall seconds', positive=True)
    policy = {key: value for key, value in row.items() if key.startswith('trace_') or key in
              ('requested_trace_policy', 'resolved_trace_policy', 'time_treatment', 'regularization',
               'positivity_limiter', 'zhang_shu_points', 'cuda_engine', 'fallback_count',
               'minimum_accepted_dt', 'maximum_accepted_dt')}
    return {'seconds': seconds, 'signature': {'adapter': 'dg', 'parameters': p,
                'work': {'steps': steps, 'final_time': final}, 'application_policy': policy},
            'checksums': None, 'gpu_uuid': None, 'precision': {'state': 'fp64'},
            'errors': {'horizon_error': abs(final - p['final_time'])},
            'gpu_identity_scope': ('CPU execution; host hardware recorded by coordinator' if p['device'] == 'cpu'
                                   else 'allocation checked by coordinator; application log has no GPU UUID'),
            'scope': f"rank-one {p['device'].upper()} HDG {p['engine']} full application process wall time, "
                     'including startup/setup/shutdown; production success and fixed accepted horizon',
            'gate_scope': 'existing application success contract; no benchmark per-step gate records'}


def parse(case, stdout: Path, stderr: Path, expected_gpu: dict, *, wall_seconds=None):
    p, adapter = parameters(case), case['adapter']
    cpu_application = adapter == 'dg' and p['device'] == 'cpu'
    uuid = None if cpu_application else expected_device(expected_gpu)
    text = Path(stdout).read_text(errors='replace') + '\n' + Path(stderr).read_text(errors='replace')
    if adapter == 'dg':
        if p['discretization'] == 'hdg':
            return parse_hdg_application(p, text, wall_seconds)
        if p['device'] == 'cpu' or p['discretization'] == 'dgsem':
            require(re.findall(r'^\[device\] requested=([^\s]+)', text, re.M) == [p['device']],
                    'application selected device differs')
        if p['discretization'] == 'dgsem':
            native = [dict(re.findall(r'(\w+)=([^ ]+)', line)) for line in text.splitlines()
                      if line.startswith('[dgsem-production] ')]
            require(len(native) == 1 and native[0].get('selected') == 'mfem-native-packed' and
                    native[0].get('strict') == '1' and native[0].get('backend') == p['device'] and
                    native[0].get('order') == str(p['order']) and native[0].get('dim') == '2' and
                    native[0].get('volume_dissipation') == 'none', 'DGSEM production selection differs')
            fused = [dict(re.findall(r'(\w+)=([^ ]+)', line)) for line in text.splitlines()
                     if line.startswith('[dgsem-fused-residual] ')]
            if p['dgsem_residual'] == 'unfused':
                require(not fused, 'unexpected fused DGSEM residual')
            else:
                contraction = 'dmma8x8x4' if p['dgsem_residual'] == 'fused-dmma' else 'simt'
                require(len(fused) == 1 and fused[0].get('enabled') == '1' and
                        fused[0].get('precision') == 'fp64' and fused[0].get('contraction') == contraction and
                        fused[0].get('order') == str(p['order']), 'DGSEM fused arithmetic selection differs')
        summaries = [line for line in text.splitlines() if line.startswith('[explicit-time] event=run-summary ')]
        require(len(summaries) == 1, 'missing/duplicate DG run-summary')
        row = dict(re.findall(r'(\w+)=([^ ]+)', summaries[0]))
        require(row.get('success') == '1' and row.get('termination_reason') == 'target-reached' and
                row.get('cfl_control_enabled') == '0' and row.get('dt_min_reached') == '0',
                'DG fixed-horizon application did not succeed')
        final = number(float(row['final_time']), 'DG final_time')
        require(abs(final - p['final_time']) <= max(1e-14, p['final_time'] * 1e-12), 'DG horizon differs')
        counts = {key: integer(int(row[key]), key) for key in
                  ('steps', 'accepted_attempts', 'rejected_attempts', 'rhs_evaluations')}
        nominal = math.ceil(p['final_time'] / p['dt'] - 1e-10)
        require(nominal <= counts['steps'] <= nominal + 1 and counts['accepted_attempts'] == counts['steps'] and
                counts['rejected_attempts'] == 0 and counts['rhs_evaluations'] > 0, 'DG accepted work differs')
        elapsed = re.findall(r'\bdone,\s+([0-9.eE+\-]+)s\.', text)
        require(len(elapsed) == 1, 'missing/duplicate DG solver elapsed time')
        seconds = number(float(elapsed[0]), 'DG solver seconds', positive=True)
        scope = f"rank-one {p['device'].upper()} {p['discretization'].upper()} {p['dgsem_residual']} full solver, fixed horizon, NaN scans enabled"
        return {'seconds': seconds, 'signature': {'adapter': adapter, 'parameters': p, 'work': counts,
                    'final_time': final, 'cfl_control_enabled': False}, 'checksums': None, 'gpu_uuid': None,
                'gpu_identity_scope': ('CPU execution; host hardware recorded by coordinator' if cpu_application else
                                       'allocation checked by coordinator; application log has no GPU UUID'),
                'precision': {'state': 'fp64'}, 'errors': {'horizon_error': abs(final - p['final_time'])},
                'scope': ('rank-one CUDA ordinary DG full solver, fixed horizon, NaN scans enabled'
                          if p['discretization'] == 'dg' and p['device'] == 'cuda' else scope)}
    rows = json_rows(text)
    if adapter == 'hdg':
        require(not any(row.get('schema') == 2 and row.get('success') is False for row in rows),
                'HDG emitted a failed numerical/cleanup record')
        values = [row for row in rows if row.get('benchmark') == 'dg_igr_hdg_cuda_accepted_step']
        require(len(values) == 1 and values[0].get('schema') == 2 and values[0].get('success') is True,
                'expected one successful schema-2 accepted-step summary')
        v = values[0]
        require(uuid and v['gpu_uuid'] == uuid and v['gpu_compute_capability'] == 80 and
                'A100' in v['gpu_name'] and v['rank_count'] == 1, 'HDG device/rank identity differs')
        for key, wanted in (('mode', p['mode']), ('cuda_engine', p['engine']), ('polynomial_order', p['order']),
                            ('mesh_n', p['mesh_n']), ('domain_periods', p['domain_periods']), ('dt', p['dt']),
                            ('measured_steps', p['measured_steps']), ('time_integrator', 'backward_euler'),
                            ('time_step_policy', 'fixed'), ('precision', 'fp64')):
            require(v[key] == wanted, f'HDG configuration differs: {key}')
        require(v['requested_trace_policy'] == p['trace_policy'] and
                all(isinstance(v[key], str) and v[key] for key in
                    ('resolved_trace_policy', 'trace_solver', 'trace_preconditioner')), 'HDG trace policy missing/different')
        require(isinstance(v['numerical_policy'], dict) and v['numerical_policy'] and
                isinstance(v['precision_policy'], dict), 'HDG numerical/precision policy missing')
        finite_tree(v['numerical_policy'], 'numerical_policy')
        for key in ('nonlinear_relative_tolerance', 'newton_correction_linear_relative_tolerance',
                    'final_coherent_linear_relative_tolerance', 'true_residual_tightening_linear_relative_tolerance',
                    'true_residual_relative_tolerance', 'conservation_relative_tolerance'):
            number(v['numerical_policy'][key], key, positive=True)
        policy = v['precision_policy']
        require(all(policy.get(key) == 'fp64' for key in ('external_state', 'local_lu', 'local_triangular_solve',
                    'tensor_operands', 'tensor_accumulation', 'acceptance_gates')) and
                all(policy.get(key) is False for key in ('mixed_precision_enabled', 'iterative_refinement_enabled',
                    'residual_replacement_enabled', 'precision_correction_enabled')), 'HDG FP64 precision contract differs')
        require(v['warmup_steps'] == 1, 'HDG internal warmup contract differs')
        samples = v['samples']
        measured = [s for s in samples if s['phase'] == 'measured']
        require(len(samples) == 1 + p['measured_steps'] and samples[0]['phase'] == 'warmup' and
                samples[0]['measured_index'] == -1 and len(measured) == p['measured_steps'] and
                [s['measured_index'] for s in measured] == list(range(p['measured_steps'])), 'HDG accepted horizon incomplete')
        signatures, checksums = [], []
        for sample in samples:
            require(all(sample['gates'].get(gate) is True for gate in GATES), 'HDG numerical acceptance gate failed')
            require(sample['retry_index'] == 0 and sample['tensor_core']['fallback_count'] == 0, 'HDG retry/fallback changed work')
            for key in ('duration_seconds', 'final_residual', 'true_relative_residual',
                        'maximum_conservation_error', 'maximum_conservation_relative_error'):
                number(sample[key], key, positive=key == 'duration_seconds')
            for key in COUNTS:
                number(sample[key], key) if key == 'time' else integer(sample[key], key)
            signatures.append({key: sample[key] for key in COUNTS} | {'gates': sample['gates']})
            item = {}
            for key in ('committed_interior_hash', 'committed_trace_hash', 'exported_interior_checksum_fnv1a64'):
                width = 16 if key == 'exported_interior_checksum_fnv1a64' else 64
                require(isinstance(sample[key], str) and re.fullmatch(r'[0-9a-fA-F]{' + str(width) + '}', sample[key]),
                        f'invalid {key}: expected {width} hex characters')
                if key != 'exported_interior_checksum_fnv1a64' or v['host_export']:
                    item[key] = sample[key]
            checksums.append(item)
        durations = v['raw_durations_seconds']
        require(len(durations) == p['measured_steps'] and durations == [s['duration_seconds'] for s in measured],
                'HDG timing samples differ from accepted records')
        for duration in durations:
            number(duration, 'accepted-step duration', positive=True)
        seconds = statistics.median(durations)
        require(v['median_seconds'] == seconds, 'HDG reported median differs from raw accepted durations')
        require(type(v['host_export']) is bool and type(v['profiling_observability']) is bool and
                v['profile_measured_step'] == (0 if v['profiling_observability'] else -1), 'HDG export/profile contract differs')
        identity = v['build_identity']
        require(isinstance(identity, dict) and all(isinstance(identity.get(key), str) and identity[key] for key in
                    ('source_revision', 'source_tree', 'source_status', 'build_directory', 'cmake_cache')) and
                identity['source_status'] in ('clean', 'dirty'), 'HDG embedded build identity missing')
        require(isinstance(v['timing_scope'], str) and v['timing_scope'], 'HDG timing scope missing')
        selected = measured[0]
        trace_policy = {key: value for key, value in v.items() if key.startswith('trace_') or
                        key in ('requested_trace_policy', 'resolved_trace_policy')}
        finite_tree(trace_policy, 'trace_policy')
        return {'seconds': seconds, 'signature': {'adapter': adapter, 'parameters': p,
                    'numerical_policy': v['numerical_policy'], 'precision_policy': policy,
                    'trace_policy': trace_policy,
                    'samples': signatures}, 'checksums': checksums, 'gpu_uuid': v['gpu_uuid'],
                'precision': policy, 'errors': {key: max(s[key] for s in samples) for key in
                    ('final_residual', 'true_relative_residual', 'maximum_conservation_error', 'maximum_conservation_relative_error')},
                'scope': v['timing_scope'], 'build_identity': v['build_identity'],
                'profile': {'enabled': v['profiling_observability'], 'measured_step': v['profile_measured_step'],
                    'expected_condensation_launches': selected['trace_solves'] if p['engine'] == 'ordinary' else
                        selected['tensor_core']['condensation_launches']},
                'host_export': v['host_export']}
    devices = [row for row in rows if 'device' in row]
    probes = [row for row in rows if row.get('probe') == 'dgsem-central-volume']
    require(len(devices) == 1 and 'A100' in devices[0]['device'] and
            devices[0]['compute_major'] == 8 and devices[0]['compute_minor'] == 0, 'DGSEM probe device differs')
    require(len(probes) == len(METHODS) and {row['method'] for row in probes} == set(METHODS), 'DGSEM method coverage incomplete')
    errors, method_seconds = {}, {}
    for row in probes:
        require(row['n1d'] == p['n'] and row['elements'] == p['elements'] and row['iterations'] == p['iterations'] and
                row['scope'] == 'central-volume-only-prescribed-sigma' and
                row['reference'] == 'long-double-same-operator-on-FP64-flux', 'DGSEM probe workload/reference differs')
        for key in ('gpu_seconds', 'setup_seconds', 'warmup_wall_seconds', 'pack_seconds', 'multiply_seconds',
                    'unpack_seconds', 'relative_l2_error', 'maximum_absolute_error', 'relative_component_sum_error'):
            number(row[key], key, positive=key == 'gpu_seconds')
        method_seconds[row['method']] = row['gpu_seconds']
        errors[row['method']] = {key: row[key] for key in
            ('relative_l2_error', 'maximum_absolute_error', 'relative_component_sum_error')}
    return {'seconds': method_seconds[p['method']], 'signature': {'adapter': adapter, 'parameters': p,
                'methods': list(METHODS), 'reference': 'long-double-same-operator-on-FP64-flux'},
            'checksums': None, 'gpu_uuid': None, 'precision': {'methods': list(METHODS), 'input_flux': 'fp64'},
            'errors': errors, 'method_seconds': method_seconds,
            'scope': 'central-volume-only-prescribed-sigma; selected method per-call GPU duration; no time-to-solution claim',
            'error_acceptance_thresholds_added': False}


def compare_fields(case, baseline_dir, candidate_dir, source):
    """DG field gate uses the existing reader with explicit encoding checks."""
    p = parameters(case)
    require(case['adapter'] == 'dg', 'physical VTK comparison is supported by the DG adapter only')
    require(sys.byteorder == 'little', 'existing VTK geometry reader requires little-endian host')
    reader_path = Path(source) / 'tests/scripts/compare_vtk_pvd_fields.py'
    spec = importlib.util.spec_from_file_location('study_vtk_reader', reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    frames, pieces, arrays = [], [], []
    for directory in (baseline_dir, candidate_dir):
        directory = Path(directory).resolve()
        pvds = list((directory / 'vtk').rglob('*.pvd'))
        require(len(pvds) == 1, 'expected one PVD per separate DG field run')
        series = reader.parse_pvd(pvds[0])
        require(series and math.isfinite(series[-1]['timestep']) and
                abs(series[-1]['timestep'] - p['final_time']) <= max(1e-14, p['final_time'] * 1e-12),
                'saved DG fields do not reach the requested horizon')
        frame = series[-1]
        dataset = Path(frame['file']).resolve()
        require(directory in dataset.parents, 'DG field dataset escaped its run directory')
        paths = list(reader.iter_piece_paths(dataset))
        require(len(paths) == 1 and directory in paths[0].parents, 'expected one confined rank-one VTU piece')
        tree = ET.parse(paths[0]).getroot()
        require(tree.get('compressor') is None and tree.get('byte_order', 'LittleEndian') == 'LittleEndian' and
                tree.get('header_type', 'UInt32') == 'UInt32', 'unsupported VTU encoding')
        found = tree.findall('./UnstructuredGrid/Piece')
        require(len(found) == 1, 'expected one VTU piece')
        piece = found[0]
        # The existing reader decodes inline UInt32 binary arrays. Validate its
        # byte-count contract before asking it to decode; no compressed fallback.
        for node in piece.findall('.//DataArray'):
            require(node.get('format') == 'binary' and node.text, 'unsupported/empty VTK data array')
            compact = ''.join(node.text.split())
            header, payload = base64.b64decode(compact[:8], validate=True), base64.b64decode(compact[8:], validate=True)
            require(len(header) == 4 and int.from_bytes(header, 'little') == len(payload), 'VTK payload length differs')
        field_nodes = piece.findall('./PointData/DataArray')
        for name in FIELDS:
            require(sum(node.get('Name') == name for node in field_nodes) == 1, f'missing/duplicate physical field: {name}')
        frames.append(frame)
        pieces.append(piece)
        arrays.append(reader.load_point_arrays(dataset, FIELDS))
    for location in ('./Points/DataArray', './Cells/DataArray'):
        left, right = [piece.findall(location) for piece in pieces]
        require(left and len(left) == len(right), 'geometry arrays missing/different')
        for a, b in zip(left, right):
            require(a.attrib == b.attrib, 'ordered geometry layout differs')
            av, bv = [reader.read_vtk_binary_data(node.text, node.get('type')) for node in (a, b)]
            require(av and len(av) == len(bv) and list(av) == list(bv) and
                    all(math.isfinite(float(x)) for x in av), 'ordered geometry/topology differs or is nonfinite')
    result = {'passed': True, 'rtol': p['rtol'], 'atol': p['atol'], 'final_time': p['final_time'], 'arrays': {},
              'scope': 'same ordered mesh, five finite physical fields, positive density/pressure',
              'reader_sha256': hashlib.sha256(reader_path.read_bytes()).hexdigest()}
    for name in FIELDS:
        a, b = [values[name] for values in arrays]
        require(a and len(a) == len(b) and all(math.isfinite(float(x)) for x in a) and
                all(math.isfinite(float(x)) for x in b), f'empty/mismatched/nonfinite physical field: {name}')
        if name in ('density', 'pressure'):
            require(min(a) > 0 and min(b) > 0, f'nonpositive physical field: {name}')
        require(all(abs(y - x) <= p['atol'] + p['rtol'] * abs(x) for x, y in zip(a, b)), f'physical tolerance failed: {name}')
        result['arrays'][name] = {'passed': True, **reader.diff_stats(a, b)}
    return result
