#!/usr/bin/env python3
"""Bounded native profiler diagnostics for the GPU study workflow.

The caller owns allocation admission, workload validation, and primary timing.
Call this provider on a compute allocation; native reports and complete CSV stay
there. Returned durations describe profiler captures/replay, never speedups.
Historical profile_harness qualification is intentionally independent.
"""
from __future__ import annotations

import csv
from copy import deepcopy
from decimal import Decimal, InvalidOperation
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


def artifact(path, *, allow_empty=False):
    path = Path(path)
    require(not path.is_symlink() and path.is_file() and (allow_empty or path.stat().st_size > 0),
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
            'tensor_dmma_instructions': values.get(DMMA),
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


NSYS_OPTIONAL_REPORTS = ('nvtx_sum', 'osrt_sum', 'nvtx_gpu_proj_sum', 'cuda_gpu_trace')
NSYS_CORE_REPORTS = ('cuda_api_sum', 'cuda_gpu_kern_sum', 'cuda_gpu_mem_time_sum')


def parse_nsys_host_summary(text, report, row_limit=30):
    """Parse one named report; OSRT and CUDA APIs have identical headers."""
    require(report in NSYS_OPTIONAL_REPORTS[:3] and type(row_limit) is int and
            1 <= row_limit <= 50, 'invalid optional Nsys summary selection')
    header, rows = None, []
    projection = report == 'nvtx_gpu_proj_sum'
    time_name = 'Total Proj Time' if projection else 'Total Time'
    count_name = 'Range Instances' if projection else ('Num Calls' if report == 'osrt_sum' else 'Instances')
    label = 'Name' if report == 'osrt_sum' else 'Range'
    duration_keys = {}
    for row in csv.reader(io.StringIO(text)):
        if any(value.startswith(time_name + ' (') for value in row):
            require(len(set(row)) == len(row) and count_name in row and label in row,
                    'unsupported optional Nsys summary header')
            duration_keys = {}
            for name in (time_name, 'Total Range Time') if projection else (time_name,):
                keys = [value for value in row if value.startswith(name + ' (') and value.endswith(')')]
                require(len(keys) == 1, 'ambiguous optional Nsys duration column')
                unit = keys[0][len(name) + 2:-1]
                seconds(1., unit)
                duration_keys[name] = keys[0], unit
            header = row
            continue
        if not row:
            header = None
        if header is None or len(row) != len(header):
            continue
        item = dict(zip(header, row))
        count = item[count_name].replace(',', '')
        require(re.fullmatch(r'[0-9]+', count) is not None and int(count) > 0 and bool(item[label]),
                'invalid optional Nsys range/call count')
        value = {'name': item[label], 'instances': int(count)}
        for name, (key, unit) in duration_keys.items():
            number = _number(item[key])
            require(number is not None, 'missing optional Nsys duration')
            value['range_duration_seconds' if name == 'Total Range Time' else 'duration_seconds'] = seconds(number, unit)
        if report != 'osrt_sum':
            require(bool(item.get('Style')), 'missing NVTX range style')
            value['style'] = item['Style']
        if projection:
            count = item.get('Total GPU Ops', '').replace(',', '')
            require(re.fullmatch(r'[0-9]+', count) is not None, 'missing projected GPU operation count')
            value['gpu_operations'] = int(count)
        rows.append(value)
        require(len(rows) <= 10000, 'unbounded optional Nsys summary rows')
    rows.sort(key=lambda value: (-value['duration_seconds'], value['name']))
    return {'availability': 'available' if rows else 'unavailable',
            'reason': None if rows else 'supported report emitted no matching trace data',
            'duration_seconds': sum(value['duration_seconds'] for value in rows) if rows else None,
            'instances': sum(value['instances'] for value in rows) if rows else None,
            'rows': rows[:row_limit], 'omitted_rows': max(0, len(rows) - row_limit),
            'scope': 'separate overlapping traced OS runtime calls' if report == 'osrt_sum' else
                ('GPU span projected from operations launched inside each same-thread NVTX range; nested ranges overlap'
                 if projection else 'host NVTX range durations; nested ranges overlap'),
            'report': report}


def parse_nsys_gpu_activity(lines, expected_events, maximum_events=None):
    """Stream sorted native CUDA intervals, returning no event-level records."""
    require(type(expected_events) is int and expected_events > 0,
            'invalid CUDA timestamp coverage bound')
    # The independent kernel/copy summaries bound the exact expected export.
    # Large realistic solves do not need an arbitrary event or file-size cap:
    # this parser retains only the current interval and ten largest gaps.
    if maximum_events is None:
        maximum_events = expected_events
    require(type(maximum_events) is int and maximum_events > 0,
            'invalid CUDA timestamp coverage bound')
    header, start_key, duration_key, factors = None, None, None, None
    first, segment_start, previous_start, merged_end = None, None, None, None
    active, gaps, largest, devices, count, zero = Decimal(0), 0, [], set(), 0, 0
    for row in csv.reader(lines):
        if any(value.startswith('Start (') for value in row):
            require(header is None and len(set(row)) == len(row) and {'Device', 'Name'} <= set(row),
                    'unsupported CUDA GPU timestamp header')
            keys = [[key for key in row if key.startswith(name + ' (') and key.endswith(')')]
                    for name in ('Start', 'Duration')]
            require(all(len(values) == 1 for values in keys), 'missing explicit CUDA timestamp units')
            start_key, duration_key = (values[0] for values in keys)
            units = (start_key[len('Start ('):-1], duration_key[len('Duration ('):-1])
            for unit in units:
                seconds(1., unit)
            factors = tuple(Decimal(str(TIME_FACTORS[unit])) for unit in units)
            header = row
            continue
        if header is None or not row:
            continue
        require(len(row) == len(header), 'malformed CUDA GPU timestamp row')
        item = dict(zip(header, row))
        require(bool(item['Device']) and bool(item['Name']), 'missing CUDA timestamp device/operation')
        try:
            start, duration = (Decimal(item[key].replace(',', '')) * factor
                               for key, factor in zip((start_key, duration_key), factors))
        except InvalidOperation as error:
            raise ProfileError('invalid CUDA GPU timestamp number') from error
        require(start.is_finite() and duration.is_finite() and start >= 0 and duration >= 0 and
                (previous_start is None or start >= previous_start), 'invalid/unsorted CUDA GPU timestamps')
        previous_start = start
        count += 1
        require(count <= maximum_events, 'CUDA timestamp event bound exceeded')
        devices.add(item['Device'])
        require(len(devices) == 1, 'CUDA timestamp export includes multiple GPU devices')
        if duration == 0:
            zero += 1
            continue
        end = start + duration
        if first is None:
            first, segment_start, merged_end = start, start, end
        elif start > merged_end:
            active += merged_end - segment_start
            gap = start - merged_end
            gaps += 1
            largest.append(float(gap))
            largest = sorted(largest, reverse=True)[:10]
            segment_start, merged_end = start, end
        else:
            merged_end = max(merged_end, end)
    require(header is not None and count == expected_events and first is not None,
            'CUDA timestamp coverage does not match kernel/copy summaries')
    active += merged_end - segment_start
    span = merged_end - first
    require(span > 0 and 0 < active <= span and math.isfinite(float(span)),
            'invalid CUDA interval-union duration')
    return {'availability': 'available', 'event_count': count, 'zero_duration_events': zero,
            'device': next(iter(devices)), 'active_union_seconds': float(active),
            'first_to_last_operation_seconds': float(span), 'internal_gap_seconds': float(span - active),
            'active_fraction_of_operation_span': float(active / span), 'internal_gap_count': gaps,
            'largest_internal_gaps_seconds': largest,
            'scope': 'union of traced CUDA kernel/memory intervals between first and last positive-duration operation',
            'limitations': ['Excludes startup before first GPU operation and shutdown after last GPU operation',
                           'No traced CUDA operation is not proof of hardware idle or its cause',
                           'Activity fraction is not SM utilization, occupancy, or whole-application GPU utilization',
                           'Profiler tracing overhead applies; primary unprofiled timings remain authoritative']}


def tensor_instruction_evidence(case, ncu):
    """Assess emitted DMMA on selected tensor launches, independently of status."""
    parameters = case.get('parameters', {})
    if case.get('adapter') == 'dg' and parameters.get('dgsem_residual') == 'fused-dmma':
        target = 'DGSEM fused FP64 residual'
        matching = lambda row: 'FusedEulerResidualKernel' in row['kernel']
    elif parameters.get('engine') == 'tensor' and (case.get('adapter') == 'hdg' or
            case.get('adapter') == 'dg' and parameters.get('discretization') == 'hdg'):
        target = 'HDG FP64 tensor volume contraction'
        matching = lambda row: row.get('kernel_group') == 'volume_contraction' or 'BorrowedEulerTensorVolumeContractionKernel' in row['kernel']
    else:
        return {'status': 'not_applicable', 'reason': 'case does not request full-solver FP64 tensor execution'}
    launches = [row for row in ncu.get('metrics', {}).get('launches', []) if matching(row)]
    counters = [row.get('tensor_dmma_instructions') for row in launches]
    available = [row['value'] for row in counters if row is not None and row['value'] is not None]
    return {'status': 'observed' if any(value > 0 for value in available) else
                     ('not_observed' if available else 'unavailable'),
            'target': target, 'metric': DMMA, 'unit': 'inst',
            'selected_tensor_launch_count': len(launches), 'available_counter_launch_count': len(available),
            'unavailable_counter_launch_count': len(launches) - len(available),
            'positive_counter_launch_count': sum(value > 0 for value in available),
            'selected_replay_dmma_instruction_sum': sum(available) if available else None,
            'scope': 'emitted native DMMA counter on selected NCU replay launches; not full-solver tensor throughput',
            'limitations': ['Unavailable counters are not zero',
                           'Zero selected counter values do not establish that the whole solver avoided tensor operations']}


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


def nsys_report_names(stdout, stderr, returncode):
    """Some native releases exit 1 for valid --help-reports output."""
    text = stdout + '\n' + stderr
    require(returncode in (0, 1) and
            re.search(r'^\s*The following built-in reports are available:\s*$', text, re.M),
            'invalid native Nsys report help status/listing')
    names = re.findall(r'^\s{2,}([a-z][a-z0-9_]*)(?:\[[^\r\n]*\])?\s+--\s+\S[^\r\n]*$', text, re.M)
    require(len(names) == len(set(names)) and set(NSYS_CORE_REPORTS) <= set(names),
            'native Nsys report help lacks explicit core report entries')
    return [name for name in NSYS_OPTIONAL_REPORTS if name in names]


def nsys_profile_help(stdout, stderr, returncode):
    text = stdout + '\n' + stderr
    require(returncode in (0, 1) and re.search(r'^\s*(?:usage:\s*)?nsys\s+profile\s+\[', text, re.M | re.I) and
            all(re.search(r'(?<![\w-])' + flag + r'(?![\w-])', text)
                for flag in ('--trace', '--sample', '--output', '--capture-range')),
            'invalid native Nsys profile help status/options')


def nsys_no_data(stdout, stderr, report, returncode):
    """Only an explicit native optional-report no-data skip is unavailable."""
    text = stdout + '\n' + stderr
    label = r'(?:NV Tools Extension \(NVTX\)|NVTX)' if report.startswith('nvtx_') else r'(?:OS Runtime|OSRT)'
    require(returncode == 1 and not re.search(r'\b(?:ERROR|FAILED|Traceback)\b', text, re.I) and
            re.search(r'^\s*SKIPPED:\s+[^\r\n]*does not contain\s+' + label + r'\s+data\.?\s*$',
                      text, re.M | re.I), 'optional native Nsys report failed without an explicit no-data skip')
    return {'availability': 'unavailable', 'reason': 'native report explicitly skipped: no matching trace data',
            'duration_seconds': None, 'instances': None, 'rows': [], 'omitted_rows': 0, 'report': report}


def _kernel_id_regex(expression):
    """Encode a regex without colons, which delimit NCU kernel-id fields.

    Capturing versus noncapturing groups does not change search membership
    unless group numbering is referenced; reject that conversion explicitly.
    The original expression remains the independent imported-name gate.
    """
    output, index, in_class, class_first = [], 0, False, -1
    converted, references = False, False
    while index < len(expression):
        character = expression[index]
        if character == '\\':
            following = expression[index + 1]
            references |= not in_class and following in '123456789'
            output.append(r'\x3a' if following == ':' else expression[index:index + 2])
            index += 2
            continue
        if not in_class:
            if expression.startswith('(?:', index):
                output.append('(')
                converted = True
                index += 3
                continue
            require(re.match(r'\(\?[aiLmsux-]+:', expression[index:]) is None,
                    'NCU kernel-id cannot encode scoped regex flags without changing matching')
            references |= expression.startswith('(?(', index)
            if character == '[':
                in_class = True
                class_first = index + 1 + (expression[index + 1:index + 2] == '^')
        elif character == ']' and index > class_first:
            in_class = False
        output.append(r'\x3a' if character == ':' else character)
        index += 1
    require(not (converted and references),
            'NCU kernel-id cannot convert noncapturing groups with numbered references or conditionals')
    encoded = ''.join(output)
    require(':' not in encoded, 'NCU kernel-id regex must not contain field delimiters')
    re.compile(encoded)
    return encoded


def capture(case: dict, argv: list[str], runner, directory: Path, expected_gpu: dict,
            *, environment=None, checkpoint=None, requested_tools=None) -> dict:
    """Capture a bounded selection; caller validates application logs.

    An optional checkpoint receives independent partial snapshots before NCU
    starts and if it fails, so completed Nsys analysis survives that failure.
    An explicit requested_tools selection runs only those native tools; omitted
    tools are absent from the result and are never counted as completed.
    """
    require(checkpoint is None or callable(checkpoint), 'capture checkpoint must be callable')
    if requested_tools is None:
        requested_tools = ('nsys', 'ncu')
    require(isinstance(requested_tools, (tuple, list)) and requested_tools and
            all(isinstance(name, str) and name in {'nsys', 'ncu'} for name in requested_tools) and
            len(set(requested_tools)) == len(requested_tools), 'invalid requested profiler tools')
    requested_tools = tuple(name for name in ('nsys', 'ncu') if name in requested_tools)
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
    selector_regex = _kernel_id_regex(regex) if selection == 'first-per-kernel' else None
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
    def run(label, command, command_timeout=timeout, *, allow_failure=False):
        try:
            expected = root / (base + '/' + label)
            try:
                result = Path(runner.run(base + '/' + label, command, cwd=None, env=environment, timeout=command_timeout))
            except RuntimeError:
                # The real study Runner raises after preserving nonzero exits.
                # Recover only this optional query's completed command receipt;
                # exhausted budgets, launch failures and timeouts still fail.
                require(allow_failure and expected.resolve() == expected,
                        'profiler query failed without an admissible optional receipt')
                receipt_path = expected / 'result.json'
                require(not receipt_path.is_symlink() and receipt_path.is_file(), 'missing optional query result receipt')
                receipt = json.loads(receipt_path.read_text())
                code = receipt.get('returncode')
                require(type(code) is int and 0 < code <= 255 and code != 124 and
                        receipt.get('status') == 'failed' and receipt.get('argv') == command and
                        receipt.get('cwd') == str(expected), 'invalid/timeout optional query result receipt')
                for name in ('stdout', 'stderr'):
                    observed_log = artifact(expected / (name + '.log'), allow_empty=True)
                    require(observed_log['sha256'] == receipt.get(name + '_sha256'),
                            'optional query output differs from retained result receipt')
                result = expected
            require(result.resolve() == expected, 'runner returned an unexpected command directory')
            receipt = json.loads((result / 'result.json').read_text())
            code = receipt.get('returncode')
            require(type(code) is int and (code == 0 or allow_failure and 0 < code <= 255 and code != 124),
                    f'profiler command failed; logs retained at {result}')
            return result
        except (OSError, ValueError, RuntimeError) as error:
            raise ProfileError(f'{base}/{label}: {error}; command logs retained') from error
    observed = gpu_identity()
    require(observed['uuid'] == expected_gpu.get('uuid') and observed['compute_capability'] == 80,
            'capture GPU differs from admitted allocation')
    result = {'status': 'partial', 'adapter': adapter, 'gpu_uuid': observed['uuid'],
              'requested_tools': list(requested_tools),
              'scope': 'selected measured step zero' if adapter == 'hdg' else 'whole application capture with bounded NCU selection',
              'primary_timing': False, 'selected_kernel_regex': regex, 'launch_skip': skip,
              'launch_count': count, 'selection': selection, 'required_kernel_groups': groups,
              'ncu_kernel_id_regex': selector_regex,
              'tools': {}, 'captures': {}, 'application_logs': []}
    tools = {name: _tool(profile.get(name, name)) if name in requested_tools else None
             for name in ('nsys', 'ncu')}
    for name, tool in tools.items():
        if name not in requested_tools:
            continue
        if tool is None:
            result['captures'][name] = {'status': 'unavailable', 'gpu_uuid': observed['uuid'],
                                        'reason': 'profiler executable unavailable'}
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
    def application_log(name, log):
        return {'tool': name, 'stdout': str(log / 'stdout.log'), 'stderr': str(log / 'stderr.log'),
                'receipt': str(log / 'result.json'),
                'stdout_sha256': artifact(log / 'stdout.log', allow_empty=True)['sha256'],
                'stderr_sha256': artifact(log / 'stderr.log', allow_empty=True)['sha256'],
                'receipt_sha256': artifact(log / 'result.json')['sha256']}

    def capture_nsys():
        tool = tools['nsys']
        capability_log = run('tools/nsys-environment', [str(tool), 'status', '--environment'], 20, allow_failure=True)
        capability_exit = json.loads((capability_log / 'result.json').read_text())['returncode']
        result['tools']['nsys']['environment_capability'] = {
            'availability': 'available' if capability_exit == 0 else 'unavailable', 'returncode': capability_exit,
            'stdout': artifact(capability_log / 'stdout.log', allow_empty=True),
            'stderr': artifact(capability_log / 'stderr.log', allow_empty=True),
            'scope': 'native environment capability query; CPU sampling remains disabled'}
        reports_log = run('tools/nsys-reports', [str(tool), 'stats', '--help-reports'], 30, allow_failure=True)
        reports_exit = json.loads((reports_log / 'result.json').read_text())['returncode']
        available_reports = nsys_report_names((reports_log / 'stdout.log').read_text(),
                                              (reports_log / 'stderr.log').read_text(), reports_exit)
        result['tools']['nsys'].update(report_query=artifact(reports_log / 'stdout.log', allow_empty=True),
            report_query_stderr=artifact(reports_log / 'stderr.log', allow_empty=True),
            report_query_returncode=reports_exit, available_optional_reports=available_reports)
        profile_help = run('tools/nsys-profile-help', [str(tool), 'profile', '--help'], 20, allow_failure=True)
        profile_exit = json.loads((profile_help / 'result.json').read_text())['returncode']
        nsys_profile_help((profile_help / 'stdout.log').read_text(), (profile_help / 'stderr.log').read_text(), profile_exit)
        result['tools']['nsys'].update(profile_help_stdout=artifact(profile_help / 'stdout.log', allow_empty=True),
            profile_help_stderr=artifact(profile_help / 'stderr.log', allow_empty=True), profile_help_returncode=profile_exit)
        prefix = directory / 'capture-nsys'
        require(not Path(str(prefix) + '.nsys-rep').exists(), 'native Nsys report already exists')
        command = [str(tool), 'profile', '--trace=cuda,nvtx,osrt', '--sample=none']
        if adapter == 'hdg':
            command += ['--capture-range=cudaProfilerApi', '--capture-range-end=stop']
        log = run('nsys', [*command, '--output', str(prefix), *argv])
        native = artifact(Path(str(prefix) + '.nsys-rep'))
        # Native report/export timestamps can disagree even immediately after
        # the first query. Never allow an older SQLite cache to substitute for
        # this capture; every bounded query regenerates its export explicitly.
        def stats_command(report):
            return [str(tool), 'stats', '--report', report, '--format', 'csv',
                    '--force-export=true', native['path']]
        stats = run('nsys-stats', stats_command(','.join(NSYS_CORE_REPORTS)), 120)
        statistics = parse_nsys((stats / 'stdout.log').read_text())
        host_summaries = {}
        for report in NSYS_OPTIONAL_REPORTS[:3]:
            if report not in available_reports:
                host_summaries[report] = {'availability': 'unavailable', 'reason': 'not listed by native Nsys report help'}
                continue
            exported = run('nsys-' + report, stats_command(report), 120, allow_failure=True)
            export_exit = json.loads((exported / 'result.json').read_text())['returncode']
            stdout, stderr = [(exported / name).read_text() for name in ('stdout.log', 'stderr.log')]
            host_summaries[report] = (parse_nsys_host_summary(stdout, report) if export_exit == 0 else
                                      nsys_no_data(stdout, stderr, report, export_exit))
            host_summaries[report]['statistics_csv'] = artifact(exported / 'stdout.log', allow_empty=True)
            host_summaries[report]['statistics_stderr'] = artifact(exported / 'stderr.log', allow_empty=True)
            host_summaries[report]['returncode'] = export_exit
        expected_events = statistics['gpu_kernel']['instances'] + (statistics['gpu_memory']['instances'] or 0)
        activity = {'availability': 'unavailable', 'reason': 'not listed by native Nsys report help'}
        if 'cuda_gpu_trace' in available_reports:
            exported = run('nsys-cuda_gpu_trace', stats_command('cuda_gpu_trace'), 120)
            trace = exported / 'stdout.log'
            with trace.open(newline='') as stream:
                activity = parse_nsys_gpu_activity(stream, expected_events)
            activity['statistics_csv'] = artifact(trace)
        logs = application_log('nsys', log)
        result['captures']['nsys'] = {'status': 'complete', 'gpu_uuid': observed['uuid'], 'native': native,
            'application_logs': [logs],
            'application_stdout': str(log / 'stdout.log'), 'application_stderr': str(log / 'stderr.log'),
            'application_stdout_sha256': sha256(log / 'stdout.log'),
            'statistics': statistics, 'host_summaries': host_summaries, 'gpu_activity': activity,
            'statistics_csv': artifact(stats / 'stdout.log'),
            'profiler_policies': {'trace': 'cuda,nvtx,osrt', 'cpu_sampling': 'none (explicit)',
                'statistics_export': '--force-export=true for every native-report stats query (explicit)',
                'osrt_threshold': 'native default; unspecified by command',
                'default_authority': result['tools']['nsys']['profile_help_stdout'],
                'sampled_cpu_stack_attribution_available': False},
            'limitations': ['OSRT covers traced library calls above the native threshold, not all host computation',
                'CPU sampling is disabled; uninstrumented CPU work has no sampled stack attribution',
                'NVTX and projected ranges may overlap and do not define startup/solve phases unless source annotates them',
                'CUDA API, GPU kernel, memory, OSRT and NVTX durations overlap; never sum them into solver wall time']}
        result['application_logs'].append(logs)
    def save_checkpoint():
        for name, identity in result['tools'].items():
            require(sha256(identity['executable']['path']) == identity['executable']['sha256'],
                    f'{name} executable changed during capture')
        result['tensor_instruction_evidence'] = tensor_instruction_evidence(case, result['captures'].get('ncu', {}))
        if checkpoint is not None:
            checkpoint(deepcopy(result))

    if tools['ncu']:
        result['captures']['ncu'] = {'status': 'pending', 'gpu_uuid': observed['uuid']}
    if tools['nsys']:
        try:
            capture_nsys()
        except Exception as error:
            result['captures']['nsys'] = {'status': 'failed', 'gpu_uuid': observed['uuid'], 'error': str(error)}
            save_checkpoint()
            raise
    save_checkpoint()

    def capture_ncu():
        tool = tools['ncu']
        tool_command = [str(tool), '--config-file', 'off']
        listed = run('tools/ncu-sections', [*tool_command, '--list-sections'], 30)
        section_text = (listed / 'stdout.log').read_text()
        selected = _available_names(section_text, (*SECTIONS, *OPTIONAL_SECTIONS))
        missing = [name for name in SECTIONS if name not in selected]
        metrics_log = run('tools/ncu-metrics', [*tool_command, '--query-metrics', '--query-metrics-mode', 'all', '--devices', '0'], 30)
        queried = _available_names((metrics_log / 'stdout.log').read_text(), (DURATION, DMMA))
        if missing or DURATION not in queried:
            result['captures']['ncu'] = {'status': 'unavailable', 'gpu_uuid': observed['uuid'],
                'reason': 'required NCU diagnostics unavailable',
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
                command += ['--kernel-id', '::regex:' + selector_regex + ':1']
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
            logs = application_log('ncu', log)
            result['captures']['ncu'] = {'status': 'partial' if missing_core or not count_complete else 'complete',
                'gpu_uuid': observed['uuid'], 'native': native, 'application_logs': [logs],
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
            result['application_logs'].append(logs)
    if tools['ncu']:
        try:
            capture_ncu()
        except Exception as error:
            result['captures']['ncu'] = {'status': 'failed', 'gpu_uuid': observed['uuid'], 'error': str(error)}
            save_checkpoint()
            raise
        save_checkpoint()
    for name, identity in result['tools'].items():
        require(sha256(identity['executable']['path']) == identity['executable']['sha256'],
                f'{name} executable changed during capture')
    result['tensor_instruction_evidence'] = tensor_instruction_evidence(case, result['captures'].get('ncu', {}))
    states = [value['status'] for value in result['captures'].values()]
    result['status'] = 'complete' if all(state == 'complete' for state in states) else (
        'unavailable' if all(state == 'unavailable' for state in states) else 'partial')
    return result
