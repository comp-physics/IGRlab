#!/usr/bin/env python3
"""Bounded native profiler diagnostics for the GPU study workflow.

The caller owns allocation admission, workload validation, and primary timing.
Call this provider on a compute allocation; native reports and complete CSV stay
there. Returned durations describe profiler captures/replay, never speedups.
Historical profile_harness qualification is intentionally independent.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess


class ProfileError(RuntimeError):
    """A requested capture/import failed; its runner logs remain available."""


SECTIONS = ('LaunchStats', 'Occupancy', 'SpeedOfLight', 'ComputeWorkloadAnalysis',
            'MemoryWorkloadAnalysis', 'SchedulerStats', 'WarpStateStats')
OPTIONAL_SECTIONS = ('SpeedOfLight_RooflineChart',)
DURATION = 'gpu__time_duration.avg'
DMMA = 'sm__inst_executed_pipe_tensor_op_dmma.sum'
ACHIEVED = 'sm__warps_active.avg.pct_of_peak_sustained_active'
THEORETICAL = ('sm__maximum_warps_per_active_cycle_pct', 'launch__theoretical_occupancy')
RESOURCES = ('launch__block_size', 'launch__grid_size', 'launch__registers_per_thread',
             'launch__shared_mem_per_block_static', 'launch__shared_mem_per_block_dynamic',
             'launch__waves_per_multiprocessor')
LIMITS = tuple('launch__occupancy_limit_' + name for name in
               ('registers', 'shared_mem', 'blocks', 'warps'))
CORE_COVERAGE = (*RESOURCES, 'achieved_occupancy', 'theoretical_occupancy',
                 'compute_throughput', 'memory_throughput')
MEMORY_DETAIL = ('lts__t_sector_hit_rate', 'l1tex__t_sector_hit_rate',
                 'l1tex__average_t_sectors_per_request_', 'l1tex__data_bank_conflicts_')
UUID_RE = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')
TIME_FACTORS = {'s': 1., 'second': 1., 'seconds': 1., 'ms': 1e-3, 'msecond': 1e-3,
                'us': 1e-6, 'usecond': 1e-6, 'ns': 1e-9, 'nsecond': 1e-9}


def require(condition, message):
    if not condition:
        raise ProfileError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def artifact(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file() and path.stat().st_size > 0,
            f'missing/nonregular native artifact: {path}')
    return {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def gpu_identity() -> dict:
    """Query allocation-exposed NVML, never map Slurm global IDs to NVML IDs."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(bool(re.fullmatch(r'(?:[0-9]+|GPU-[0-9a-fA-F]+(?:-[0-9a-fA-F]+)*)', visible)),
            'requires exactly one valid CUDA_VISIBLE_DEVICES token')
    command = ['nvidia-smi', '--query-gpu=uuid,name,compute_cap',
               '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProfileError(f'bounded GPU identity query failed: {error}') from error
    require(result.returncode == 0, f'GPU identity query exited {result.returncode}')
    rows = list(csv.reader(io.StringIO(result.stdout)))
    require(len(rows) == 1 and len(rows[0]) == 3, 'requires exactly one NVML-exposed GPU')
    uuid, name, capability = (value.strip() for value in rows[0])
    require(UUID_RE.fullmatch(uuid) is not None and 'A100' in name and capability == '8.0',
            'requires a native A100 with compute capability 8.0')
    if visible.startswith('GPU-'):
        require(uuid.lower().startswith(visible.lower()), 'visible GPU UUID does not match NVML')
    return {'uuid': uuid, 'name': name, 'compute_capability': 80,
            'identity_command': command,
            'identity_stdout_sha256': hashlib.sha256(result.stdout.encode()).hexdigest(),
            'allocation_environment': {key: os.environ[key] for key in
                ('CUDA_VISIBLE_DEVICES', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS') if key in os.environ}}


def seconds(value, unit):
    require(unit in TIME_FACTORS, f'unrecognized profiler duration unit: {unit!r}')
    return value * TIME_FACTORS[unit]


def metric_category(name):
    if name == DURATION:
        return 'duration'
    if name == ACHIEVED or name in THEORETICAL:
        return 'occupancy'
    if name in (*RESOURCES, *LIMITS) or name.startswith(('launch__block_dim_', 'launch__grid_dim_',
                                                      'launch__shared_mem_per_block_')):
        return 'launch_resources'
    if name == DMMA:
        return 'tensor_instruction'
    if name.startswith(('sm__', 'smsp__', 'gpc__', 'dram__')) and (
            'cycles_elapsed' in name or 'clock' in name):
        return 'clock_cycle_input'
    if name.startswith('derived__local_spilling') or ('register_spilling' in name and
            name.startswith(('sass__', 'sm__', 'smsp__', 'derived__'))):
        return 'direct_spill'
    if 'mem_local' in name or 'inst_executed_local_' in name or name.startswith('memory_l2_theoretical_sectors_local'):
        return 'local_memory'
    if 'branch' in name and name.startswith(('sm__', 'smsp__', 'sass__', 'derived__')):
        return 'branch'
    if ('issue_stalled' in name or 'warp_cycles_per_issue' in name) and name.startswith(('sm__', 'smsp__')):
        return 'warp_stall'
    if name.startswith(('smsp__warps_eligible', 'smsp__warps_active', 'smsp__warps_issued',
                        'smsp__issue_active', 'smsp__inst_issued', 'smsp__issue_slot_utilization')):
        return 'scheduler'
    if name.startswith(MEMORY_DETAIL):
        return 'memory'
    if re.search(r'__sass_thread_inst_executed_op_[dfh](?:add|mul|fma)_pred_on', name) or (
            name.startswith('derived__') and any(key in name for key in ('flop', 'arithmetic_intensity'))):
        return 'roofline_input'
    if name.startswith(('dram__', 'lts__', 'l1tex__')) and (
            '.pct_of_peak_sustained_' in name or 'bytes' in name):
        return 'memory'
    if name.startswith(('sm__', 'smsp__')) and '.pct_of_peak_sustained_' in name:
        return 'compute'
    return None


def _number(raw):
    text = raw.strip()
    if text.lower() in {'', 'n/a', 'na', 'not available', 'unavailable', 'not supported'}:
        return None
    try:
        value = float(text.replace(',', ''))
    except ValueError as error:
        raise ProfileError(f'invalid numerical profiler metric: {raw!r}') from error
    require(math.isfinite(value) and value >= 0, 'nonfinite/negative profiler metric')
    return value


def _ncu_observations(text):
    """Accept long CSV or wide CSV with an explicit aligned units row."""
    rows = list(csv.reader(io.StringIO(text)))
    header, units, format_name = None, None, None
    observations = []
    for row in rows:
        if {'ID', 'Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value'} <= set(row):
            header, units, format_name = row, None, 'long'
            continue
        if {'ID', 'Kernel Name', DURATION} <= set(row):
            header, units, format_name = row, None, 'wide'
            continue
        if header is None or len(row) != len(header):
            continue
        item = dict(zip(header, row))
        if format_name == 'wide' and units is None:
            require(not item['ID'] and not item['Kernel Name'] and item[DURATION] in TIME_FACTORS,
                    'wide NCU CSV requires an explicit duration units row')
            units = item
            continue
        if not item['ID'] or not item['Kernel Name']:
            continue
        require(re.fullmatch(r'[0-9]+', item['ID']) is not None, 'invalid native launch ID')
        if format_name == 'long':
            observations.append((item['ID'], item['Kernel Name'], item['Metric Name'],
                                 item['Metric Unit'], item['Metric Value']))
        else:
            for name in header:
                if '__' in name:
                    observations.append((item['ID'], item['Kernel Name'], name, units[name], item[name]))
    require(format_name is not None and observations, 'no native NCU metric observations')
    return format_name, observations


def parse_ncu(text, kernel_regex, maximum_launches=3, metric_limit=120):
    """Normalize compact counters, preserving units, zeros and unavailable data."""
    require(type(maximum_launches) is int and 1 <= maximum_launches <= 32 and
            type(metric_limit) is int and 1 <= metric_limit <= 120, 'invalid compact NCU bounds')
    pattern = re.compile(kernel_regex)
    format_name, observations = _ncu_observations(text)
    seen, metrics = {}, {}
    for launch, kernel, name, unit, raw in observations:
        require(pattern.search(kernel) is not None,
                f'native NCU kernel outside requested filter: launch={launch}, kernel={kernel[:512]!r}')
        key = launch, kernel, name
        kind = metric_category(name)
        # Unknown/textual metrics stay only in the native CSV, but duplicate
        # conflicts are still rejected instead of silently taking the last row.
        value = _number(raw) if kind is not None else raw.strip()
        observation = unit, value
        require(key not in seen or seen[key] == observation, 'conflicting duplicate NCU launch/metric')
        seen[key] = observation
        if kind is not None:
            metrics[key] = {'metric': name, 'unit': unit, 'value': value, 'category': kind,
                            'availability': 'available' if value is not None else 'unavailable'}
    launches = sorted({key[:2] for key in seen})
    require(0 < len(launches) <= maximum_launches, 'missing/unbounded NCU launch coverage')
    result = []
    for launch, kernel in launches:
        values = {key[2]: value for key, value in metrics.items() if key[:2] == (launch, kernel)}
        duration = values.get(DURATION)
        require(duration is not None and duration['value'] is not None and duration['value'] > 0,
                'missing positive native launch duration')
        for name in ('launch__block_size', 'launch__grid_size'):
            if name in values and values[name]['value'] is not None:
                require(values[name]['value'] > 0, 'nonpositive native launch size')
        for name in (ACHIEVED, *THEORETICAL):
            if name in values and values[name]['value'] is not None:
                require(values[name]['unit'] == '%' and values[name]['value'] <= 100,
                        'invalid occupancy percent/unit')
        if DMMA in values and values[DMMA]['value'] is not None:
            require(values[DMMA]['unit'] == 'inst', 'unexpected DMMA metric unit')
        available = {name for name, value in values.items() if value['value'] is not None}
        has = lambda kind: any(value['category'] == kind and value['value'] is not None for value in values.values())
        coverage = {name: name in available for name in (*RESOURCES, *LIMITS)}
        coverage.update(achieved_occupancy=ACHIEVED in available,
            theoretical_occupancy=bool(set(THEORETICAL) & available),
            compute_throughput=any(name.startswith(('sm__throughput.', 'smsp__throughput.')) and
                                  '.pct_of_peak_sustained_' in name for name in available),
            memory_throughput=any(name.startswith(('dram__throughput.', 'lts__throughput.', 'l1tex__throughput.')) and
                                 '.pct_of_peak_sustained_' in name for name in available),
            tensor_dmma_instructions=DMMA in available, issue_activity=any(name.startswith('smsp__issue_active.') for name in available),
            eligible_warps=any(name.startswith('smsp__warps_eligible.') for name in available),
            aggregate_warp_stalls=has('warp_stall'), branch_counters=has('branch'),
            branch_divergence=any('branch' in name and ('diverg' in name or 'uniform' in name) for name in available),
            local_memory_traffic=has('local_memory'), direct_register_spill_counters=has('direct_spill'),
            cache_hit_rate=any(name.startswith(MEMORY_DETAIL[:2]) for name in available),
            coalescing=any(name.startswith(MEMORY_DETAIL[2]) for name in available),
            shared_bank_conflicts=any(name.startswith(MEMORY_DETAIL[3]) for name in available),
            clock_cycle_inputs=has('clock_cycle_input'), roofline_inputs=has('roofline_input'))
        fp64_operations = {op: any(re.search(r'__sass_thread_inst_executed_op_d' + op +
                               r'_pred_on\.sum$', name) for name in available) for op in ('add', 'mul', 'fma')}
        coverage.update({'fp64_scalar_' + op + '_counter': present for op, present in fp64_operations.items()})
        coverage['dram_byte_counter_inputs'] = ('dram__bytes.sum' in available or
                                               {'dram__bytes_read.sum', 'dram__bytes_write.sum'} <= available)
        coverage['fp64_scalar_roofline_inputs_complete'] = (all(fp64_operations.values()) and
                                                           coverage['dram_byte_counter_inputs'])
        def priority(value):
            name = value['metric']
            if name == DURATION or name == DMMA or name in (*RESOURCES, *LIMITS, ACHIEVED, *THEORETICAL) or '__throughput.' in name:
                return 0, name
            if value['category'] in ('scheduler', 'warp_stall', 'branch', 'local_memory',
                                      'direct_spill', 'roofline_input', 'clock_cycle_input') or name.startswith(MEMORY_DETAIL):
                return 1, name
            return 2, name
        compact = sorted(values.values(), key=priority)
        result.append({'launch_id': launch, 'kernel': kernel,
            'replay_duration_seconds': seconds(duration['value'], duration['unit']),
            'metric_rows': compact[:metric_limit], 'omitted_metric_count': max(0, len(compact) - metric_limit),
            'metric_limit': metric_limit, 'diagnostic_availability': coverage,
            'missing_diagnostics': [name for name, present in coverage.items() if not present],
            'availability_scope': 'all imported native rows; compact rows may omit breakdowns'})
    return {'format': format_name, 'observed_launches': len(result), 'launches': result,
            'scope': 'NCU replay diagnostics; unavailable counters are not zero',
            'limitations': ['Local-memory traffic does not establish register spilling',
                           'Branch instruction counts do not establish branch divergence',
                           'Scalar roofline counters exclude tensor-operation FLOPs',
                           'Aggregate stalls are not elapsed durations or additive solver costs']}


def parse_nsys(text, row_limit=30):
    """Keep API, GPU kernel and GPU memory durations as separate sums."""
    require(type(row_limit) is int and 1 <= row_limit <= 50, 'invalid Nsys row bound')
    groups = {name: [] for name in ('cuda_api', 'gpu_kernel', 'gpu_memory')}
    header, kind, count_key, time_key, unit = None, None, None, None, None
    for row in csv.reader(io.StringIO(text)):
        if not row:
            header = None
            continue
        if any(value.startswith('Total Time (') for value in row):
            keys = [value for value in row if value.startswith('Total Time (')]
            require(len(keys) == 1, 'ambiguous Nsys duration column')
            time_key = keys[0]
            unit = time_key[len('Total Time ('):-1]
            seconds(1., unit)
            if 'Num Calls' in row and 'Name' in row:
                kind, count_key = 'cuda_api', 'Num Calls'
            elif 'Instances' in row and 'Name' in row:
                kind, count_key = 'gpu_kernel', 'Instances'
            elif 'Count' in row and 'Operation' in row:
                kind, count_key = 'gpu_memory', 'Count'
            else:
                raise ProfileError('unrecognized Nsys summary table')
            header = row
            continue
        if header is None or len(header) != len(row):
            continue
        item = dict(zip(header, row))
        count = item[count_key].replace(',', '')
        require(re.fullmatch(r'[0-9]+', count) is not None and int(count) > 0,
                'invalid Nsys instance count')
        value = _number(item[time_key])
        require(value is not None, 'missing Nsys total duration')
        name = item.get('Name', item.get('Operation', ''))
        require(bool(name), 'missing Nsys operation name')
        groups[kind].append({'name': name, 'instances': int(count), 'duration_seconds': seconds(value, unit)})
    require(groups['cuda_api'] and groups['gpu_kernel'], 'missing Nsys API/kernel summary coverage')
    result = {}
    for name, rows in groups.items():
        rows.sort(key=lambda row: (-row['duration_seconds'], row['name']))
        result[name] = {'availability': 'available' if rows else 'unavailable',
            'reason': None if rows else 'native summary emitted no memory operations',
            'duration_seconds': sum(row['duration_seconds'] for row in rows) if rows else None,
            'instances': sum(row['instances'] for row in rows) if rows else None,
            'rows': rows[:row_limit], 'omitted_rows': max(0, len(rows) - row_limit)}
    result['scope'] = 'separate overlapping API, kernel and copy sums; not additive elapsed time'
    return result


def native_ncu(prefix):
    candidates = [Path(str(prefix) + suffix) for suffix in ('.ncu-rep', '.ncu-repz')]
    present = [path for path in candidates if path.exists() or path.is_symlink()]
    require(len(present) == 1, 'expected exactly one .ncu-rep or .ncu-repz report')
    artifact(present[0])
    return present[0]


def _tool(name):
    path = shutil.which(str(name))
    return Path(path).resolve() if path else None


def _available_names(text, names):
    return [name for name in names if re.search(r'(?<![A-Za-z0-9_])' + re.escape(name) + r'(?![A-Za-z0-9_])', text)]


def capture(case: dict, argv: list[str], runner, directory: Path, expected_gpu: dict) -> dict:
    """Capture a bounded selection; caller validates returned application logs."""
    adapter = case.get('adapter')
    profile = case.get('profile', {})
    require(adapter in {'hdg', 'dg', 'dgsem'} and isinstance(profile, dict), 'unsupported study profiler adapter')
    require(isinstance(argv, list) and argv and all(isinstance(value, str) and value for value in argv), 'invalid capture argv')
    regex = profile.get('kernel_regex')
    require(isinstance(regex, str) and 0 < len(regex) <= 2048, 'bounded kernel_regex is required')
    re.compile(regex)
    skip = profile.get('launch_skip', profile.get('skip', 0))
    count = profile.get('launch_count', profile.get('count', 1))
    limit = profile.get('max_metric_rows', 120)
    timeout = profile.get('timeout', 240)
    require(type(skip) is int and 0 <= skip <= 10000 and type(count) is int and 1 <= count <= 32 and
            type(limit) is int and 1 <= limit <= 120 and type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= 240,
            'invalid bounded profiler selection')
    selection = profile.get('selection', 'launches')
    groups = profile.get('kernel_groups', {})
    require(selection in {'launches', 'first-per-kernel'} and isinstance(groups, dict) and
            len(groups) <= 12 and all(isinstance(name, str) and name and isinstance(value, str) and
                                      0 < len(value) <= 2048 for name, value in groups.items()),
            'invalid bounded kernel group selection')
    for value in groups.values():
        re.compile(value)
    require(selection != 'first-per-kernel' or skip == 0, 'first-per-kernel selection requires zero launch skip')
    for preferred, alias in (('launch_skip', 'skip'), ('launch_count', 'count')):
        require(preferred not in profile or alias not in profile or profile[preferred] == profile[alias],
                'conflicting profiler selection aliases')
    if adapter == 'hdg':
        indices = [index for index, value in enumerate(argv) if value == '--profile-measured-step']
        require(len(indices) == 1 and indices[0] + 1 < len(argv) and argv[indices[0] + 1] == '0',
                'HDG capture requires explicit selected measured step zero')
    root = Path(runner.output).resolve()
    directory = Path(directory).absolute()
    require(directory.resolve() == directory and root in directory.parents, 'capture directory must be confined below runner output')
    directory.mkdir(parents=True, exist_ok=True)
    base = directory.relative_to(root).as_posix()
    def run(label, command, command_timeout=timeout):
        try:
            result = Path(runner.run(base + '/' + label, command, cwd=None, env=None, timeout=command_timeout))
            require(result.resolve() == root / (base + '/' + label), 'runner returned an unexpected command directory')
            receipt = json.loads((result / 'result.json').read_text())
            require(receipt.get('returncode') == 0, f'profiler command failed; logs retained at {result}')
            return result
        except (OSError, ValueError, RuntimeError) as error:
            raise ProfileError(f'{base}/{label}: {error}; command logs retained') from error
    observed = gpu_identity()
    require(observed['uuid'] == expected_gpu.get('uuid') and observed['compute_capability'] == 80,
            'capture GPU differs from admitted allocation')
    result = {'status': 'complete', 'adapter': adapter, 'gpu_uuid': observed['uuid'],
              'scope': 'selected measured step zero' if adapter == 'hdg' else 'whole application capture with bounded NCU selection',
              'primary_timing': False, 'selected_kernel_regex': regex, 'launch_skip': skip,
              'launch_count': count, 'selection': selection, 'required_kernel_groups': groups,
              'tools': {}, 'captures': {}, 'application_logs': []}
    tools = {name: _tool(profile.get(name, name)) for name in ('nsys', 'ncu')}
    for name, tool in tools.items():
        if tool is None:
            result['captures'][name] = {'status': 'unavailable', 'reason': 'profiler executable unavailable'}
            continue
        # NCU otherwise composes sections from config.ncu-cfg in cwd/home with
        # explicit CLI sections, which can silently expand this bounded study.
        tool_command = [str(tool), '--config-file', 'off'] if name == 'ncu' else [str(tool)]
        version = run('tools/' + name + '-version', [*tool_command, '--version'], 20)
        help_log = run('tools/' + name + '-help', [*tool_command, '--help'], 20)
        result['tools'][name] = {'executable': artifact(tool), 'version_stdout': artifact(version / 'stdout.log'),
                                'help_stdout': artifact(help_log / 'stdout.log')}
        if name == 'ncu':
            result['tools'][name]['config_file'] = 'off (explicit)'
    if tools['nsys']:
        tool = tools['nsys']
        prefix = directory / 'capture-nsys'
        require(not Path(str(prefix) + '.nsys-rep').exists(), 'native Nsys report already exists')
        command = [str(tool), 'profile', '--trace=cuda,nvtx,osrt', '--sample=none']
        if adapter == 'hdg':
            command += ['--capture-range=cudaProfilerApi', '--capture-range-end=stop']
        log = run('nsys', [*command, '--output', str(prefix), *argv])
        native = artifact(Path(str(prefix) + '.nsys-rep'))
        stats = run('nsys-stats', [str(tool), 'stats', '--report',
            'cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum', '--format', 'csv', native['path']], 120)
        result['captures']['nsys'] = {'status': 'complete', 'native': native,
            'application_stdout': str(log / 'stdout.log'), 'application_stderr': str(log / 'stderr.log'),
            'application_stdout_sha256': sha256(log / 'stdout.log'),
            'statistics': parse_nsys((stats / 'stdout.log').read_text()),
            'statistics_csv': artifact(stats / 'stdout.log')}
        result['application_logs'].append({'stdout': str(log / 'stdout.log'), 'stderr': str(log / 'stderr.log')})
    if tools['ncu']:
        tool = tools['ncu']
        tool_command = [str(tool), '--config-file', 'off']
        listed = run('tools/ncu-sections', [*tool_command, '--list-sections'], 30)
        section_text = (listed / 'stdout.log').read_text()
        selected = _available_names(section_text, (*SECTIONS, *OPTIONAL_SECTIONS))
        missing = [name for name in SECTIONS if name not in selected]
        metrics_log = run('tools/ncu-metrics', [*tool_command, '--query-metrics', '--query-metrics-mode', 'all', '--devices', '0'], 30)
        queried = _available_names((metrics_log / 'stdout.log').read_text(), (DURATION, DMMA))
        if missing or DURATION not in queried:
            result['captures']['ncu'] = {'status': 'unavailable', 'reason': 'required NCU diagnostics unavailable',
                'missing_sections': missing, 'duration_metric_available': DURATION in queried}
        else:
            prefix = directory / 'capture-ncu'
            require(not any(Path(str(prefix) + suffix).exists() for suffix in ('.ncu-rep', '.ncu-repz')),
                    'native NCU report already exists')
            command = [*tool_command, '--target-processes', 'application-only', '--profile-from-start',
                'off' if adapter == 'hdg' else 'on', '--kernel-name-base', 'demangled']
            if selection == 'first-per-kernel':
                # Express both constraints in the documented single selector;
                # an empty-name kernel-id must not replace the name filter.
                command += ['--kernel-id', '::regex:' + regex + ':1']
            else:
                command += ['--kernel-name', 'regex:' + regex,
                            '--launch-skip', str(skip), '--launch-count', str(count)]
            command += ['--rename-kernels', 'off', '--metrics', ','.join(queried)]
            command += [argument for section in selected for argument in ('--section', section)]
            log = run('ncu', [*command, '--export', str(prefix), *argv])
            native = artifact(native_ncu(prefix))
            imported = run('ncu-import', [*tool_command, '--import', native['path'], '--csv', '--page', 'raw',
                                        '--print-units', 'base', '--rename-kernels', 'off'], 120)
            normalized = parse_ncu((imported / 'stdout.log').read_text(), regex, count, limit)
            if selection == 'first-per-kernel':
                require(len({row['kernel'] for row in normalized['launches']}) == normalized['observed_launches'],
                        'more than one representative launch per unique kernel name')
            observed_groups = set()
            for row in normalized['launches']:
                matched = [name for name, pattern in groups.items() if re.search(pattern, row['kernel'])]
                require(len(matched) <= 1, 'ambiguous required kernel group match')
                row['kernel_group'] = matched[0] if matched else 'additional_selected_kernel'
                observed_groups.update(matched)
            require(set(groups) <= observed_groups, 'required NCU kernel group coverage missing')
            normalized['observed_kernel_groups'] = sorted(observed_groups)
            missing_core = [{'launch_id': row['launch_id'], 'kernel': row['kernel'],
                             'missing': [name for name in CORE_COVERAGE if not row['diagnostic_availability'][name]]}
                            for row in normalized['launches']]
            missing_core = [row for row in missing_core if row['missing']]
            count_complete = selection == 'first-per-kernel' or normalized['observed_launches'] == count
            result['captures']['ncu'] = {'status': 'partial' if missing_core or not count_complete else 'complete', 'native': native,
                'application_stdout': str(log / 'stdout.log'), 'application_stderr': str(log / 'stderr.log'),
                'application_stdout_sha256': sha256(log / 'stdout.log'), 'sections': selected,
                'unavailable_optional_sections': [name for name in OPTIONAL_SECTIONS if name not in selected],
                'queried_metrics': queried, 'tensor_dmma_metric_available': DMMA in queried,
                'missing_required_diagnostics': missing_core,
                'requested_launches': count if selection == 'launches' else None,
                'launch_coverage_complete': count_complete,
                'launch_selection_scope': 'exact requested matching launch count' if selection == 'launches' else
                    'one representative per unique kernel name, required families covered, bounded maximum count',
                'section_query': artifact(listed / 'stdout.log'), 'metric_query': artifact(metrics_log / 'stdout.log'),
                'metrics': normalized,
                'import_csv': artifact(imported / 'stdout.log'),
                'profiler_policies': {'config_file': 'off (explicit)',
                    'replay_mode': 'native default; unspecified by command',
                    'clock_control': 'native default; unspecified by command',
                    'cache_control': 'native default; unspecified by command',
                    'default_authority': result['tools']['ncu']['help_stdout'],
                    'observed_solver_clocks': False}}
            result['application_logs'].append({'stdout': str(log / 'stdout.log'), 'stderr': str(log / 'stderr.log')})
    for name, identity in result['tools'].items():
        require(sha256(identity['executable']['path']) == identity['executable']['sha256'],
                f'{name} executable changed during capture')
    states = [value['status'] for value in result['captures'].values()]
    result['status'] = 'complete' if all(state == 'complete' for state in states) else (
        'unavailable' if all(state == 'unavailable' for state in states) else 'partial')
    return result
