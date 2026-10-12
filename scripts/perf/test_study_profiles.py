#!/usr/bin/env python3
"""Host-only intercepted fixtures; no GPU, application, or profiler execution."""
import csv
import io
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import study_profiles as profiles


UUID = 'GPU-12345678-1234-1234-1234-123456789abc'
GPU = {'uuid': UUID, 'name': 'NVIDIA A100-SXM4-80GB', 'compute_capability': 80}
KERNEL = 'void PackedInverseMassKernel()'


def csv_text(rows):
    stream = io.StringIO()
    csv.writer(stream).writerows(rows)
    return stream.getvalue()


def long_csv(extra=(), duration_unit='us'):
    return csv_text([['ID', 'Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value'],
                    ['0', KERNEL, profiles.DURATION, duration_unit, '2'], *extra])


def wide_csv():
    return csv_text([
        ['ID', 'Process ID', 'Kernel Name', 'Context', profiles.DURATION,
         profiles.DMMA, 'launch__waves_per_multiprocessor', 'launch__block_size', profiles.ACHIEVED],
        ['', '', '', '', 'ns', 'inst', '', '', '%'],
        ['0', '123', KERNEL, '1', '2000', '0', '0.00', '128', '25'],
        ['1', '123', KERNEL, '1', '3000', '0', '0.00', '128', 'N/A']])


def wide_complete():
    rows = list(csv.reader(io.StringIO(wide_csv())))
    extras = [('launch__grid_size', '', '16'), ('launch__registers_per_thread', '', '32'),
              ('launch__shared_mem_per_block_static', 'byte', '0'),
              ('launch__shared_mem_per_block_dynamic', 'byte', '0'),
              ('launch__theoretical_occupancy', '%', '50'),
              ('sm__throughput.avg.pct_of_peak_sustained_elapsed', '%', '10'),
              ('dram__throughput.avg.pct_of_peak_sustained_elapsed', '%', '10')]
    rows[0] += [name for name, unit, value in extras]
    rows[1] += [unit for name, unit, value in extras]
    for row in rows[2:]:
        row[-1] = '25'
        row += [value for name, unit, value in extras]
    return csv_text(rows)


def nsys_csv():
    return csv_text([
        ['Time (%)', 'Total Time (ns)', 'Num Calls', 'Name'], ['100', '5000', '2', 'cudaStreamSynchronize'], [],
        ['Time (%)', 'Total Time (us)', 'Instances', 'Name'], ['100', '3', '1', KERNEL], [],
        ['Time (%)', 'Total Time (ms)', 'Count', 'Operation'], ['100', '0.002', '1', '[CUDA memcpy Device-to-Host]']])


def host_summary(report):
    if report == 'osrt_sum':
        return csv_text([['Time (%)', 'Total Time (us)', 'Num Calls', 'Name'],
                         ['100', '8', '2', 'poll']])
    if report == 'nvtx_gpu_proj_sum':
        return csv_text([['Range', 'Style', 'Total Proj Time (ns)', 'Total Range Time (ns)',
                          'Range Instances', 'Total GPU Ops'],
                         ['DG/Mult', 'PushPop', '4000', '6000', '1', '2']])
    return csv_text([['Time (%)', 'Total Time (ms)', 'Instances', 'Style', 'Range'],
                     ['100', '0.006', '1', 'PushPop', 'DG/Mult']])


def gpu_trace(rows=None):
    return csv_text([['Start (ns)', 'Duration (us)', 'Device', 'Ctx', 'Strm', 'Name'],
                     *(rows or [['1000', '3', 'GPU 0', '1', '1', KERNEL],
                                ['3000', '2', 'GPU 0', '1', '2', '[CUDA memcpy DtoH]']])])


def large_gpu_trace(count):
    yield 'Start (ns),Duration (ns),Device,Ctx,Strm,Name\n'
    for index in range(count):
        name = KERNEL if index + 1 < count else '[CUDA memcpy DtoH]'
        yield f'{index * 2000},1000,GPU 0,1,1,{name}\n'


def reports_help(optional=True):
    names = (*profiles.NSYS_CORE_REPORTS, *(profiles.NSYS_OPTIONAL_REPORTS if optional else ()))
    return '\nThe following built-in reports are available:\n\n' + '\n'.join(
        '  ' + name + ('[:nvtx-name][:base|:mangled]' if name.endswith('kern_sum') else '') +
        ' -- Native report description' for name in names) + '\n'


PROFILE_HELP = 'usage: nsys profile [<args>] [application ...]\n  --trace --sample --output --capture-range\n'


class FakeRunner:
    def __init__(self, output, *, missing_sections=False, missing_reports=False, fail=None):
        self.output = output
        self.commands = []
        self.environments = {}
        self.missing_sections = missing_sections
        self.missing_reports = missing_reports
        self.fail = fail

    def run(self, label, argv, *, cwd=None, env=None, timeout=None):
        self.commands.append((label, argv, timeout))
        self.environments[label] = env
        directory = self.output / label
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'stderr.log').write_text('')
        (directory / 'result.json').write_text(json.dumps({'returncode': 1 if self.fail == label else 0}))
        text = 'native tool version/help fixture'
        if label.endswith('ncu-sections'):
            text = ' '.join(profiles.SECTIONS[:-1] if self.missing_sections else profiles.SECTIONS)
        elif label.endswith('ncu-metrics'):
            text = profiles.DURATION + '\n' + profiles.DMMA
        elif label.endswith('nsys-stats'):
            text = nsys_csv()
        elif label.endswith('nsys-reports'):
            text = reports_help(not self.missing_reports)
        elif label.endswith('nsys-profile-help'):
            text = PROFILE_HELP
        elif any(label.endswith('nsys-' + report) for report in profiles.NSYS_OPTIONAL_REPORTS[:3]):
            text = host_summary(label.rsplit('nsys-', 1)[1])
        elif label.endswith('nsys-cuda_gpu_trace'):
            text = gpu_trace()
        elif label.endswith('ncu-import'):
            text = wide_complete()
        elif label.endswith('/nsys'):
            prefix = argv[argv.index('--output') + 1]
            Path(prefix + '.nsys-rep').write_bytes(b'native fixture')
            text = '{"application":"fixture"}'
        elif label.endswith('/ncu'):
            prefix = argv[argv.index('--export') + 1]
            Path(prefix + '.ncu-repz').write_bytes(b'compressed native fixture')
            text = '{"application":"fixture"}'
        (directory / 'stdout.log').write_text(text)
        return directory


class IdentityTests(unittest.TestCase):
    def query(self, visible='0', output=None):
        environment = {'CUDA_VISIBLE_DEVICES': visible, 'SLURM_JOB_GPUS': '1', 'SLURM_STEP_GPUS': '1'}
        result = subprocess.CompletedProcess([], 0, output or f'{UUID}, NVIDIA A100-SXM4-80GB, 8.0\n', '')
        with patch.dict(os.environ, environment, clear=True), patch.object(profiles.subprocess, 'run', return_value=result) as run:
            value = profiles.gpu_identity()
            command = run.call_args.args[0]
            self.assertFalse(any(argument.startswith('--id') for argument in command))
            self.assertEqual(run.call_args.kwargs['timeout'], 15)
            return value

    def test_slurm_numeric_identifier_is_not_nvml_index(self):
        identity = self.query()
        self.assertEqual(identity['uuid'], UUID)
        self.assertEqual(identity['allocation_environment']['SLURM_JOB_GPUS'], '1')

    def test_visible_uuid_prefix_matches(self):
        self.assertEqual(self.query('GPU-12345678')['compute_capability'], 80)

    def test_uuid_mismatch_and_multiple_nvml_reject(self):
        with self.assertRaises(profiles.ProfileError):
            self.query('GPU-87654321')
        with self.assertRaises(profiles.ProfileError):
            self.query(output=f'{UUID}, A100, 8.0\n{UUID}, A100, 8.0\n')

    def test_invalid_visibility_rejects_before_query(self):
        for value in ('', '0,1', '-1', 'GPU-', ' 0', 'MIG-1234'):
            with self.subTest(value=value), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': value}, clear=True), \
                    patch.object(profiles.subprocess, 'run') as run:
                with self.assertRaises(profiles.ProfileError):
                    profiles.gpu_identity()
                run.assert_not_called()

    def test_invalid_board_or_capability_reject(self):
        for output in (f'{UUID}, RTX 5080, 12.0\n', 'not-a-uuid, A100, 8.0\n'):
            with self.subTest(output=output), self.assertRaises(profiles.ProfileError):
                self.query(output=output)


class MetricTests(unittest.TestCase):
    def test_wide_units_and_zero_observations(self):
        value = profiles.parse_ncu(wide_csv(), 'PackedInverseMassKernel', 2)
        self.assertEqual(value['format'], 'wide')
        self.assertEqual(value['observed_launches'], 2)
        self.assertAlmostEqual(value['launches'][0]['replay_duration_seconds'], 2e-6)
        self.assertTrue(value['launches'][0]['diagnostic_availability']['launch__waves_per_multiprocessor'])
        self.assertFalse(value['launches'][1]['diagnostic_availability']['achieved_occupancy'])

    def test_wide_requires_explicit_units(self):
        rows = list(csv.reader(io.StringIO(wide_csv())))
        with self.assertRaises(profiles.ProfileError):
            profiles.parse_ncu(csv_text([rows[0], *rows[2:]]), 'PackedInverseMassKernel', 2)

    def test_wide_textual_launch_metadata_is_not_a_numerical_metric(self):
        rows = list(csv.reader(io.StringIO(wide_csv())))
        rows[0].append('launch__execution_model')
        rows[1].append('')
        for row in rows[2:]:
            row.append('SIMT')
        self.assertEqual(profiles.parse_ncu(csv_text(rows), 'PackedInverseMassKernel', 2)['observed_launches'], 2)

    def test_long_units_are_not_assumed(self):
        for unit, expected in [('ns', 2e-9), ('usecond', 2e-6), ('msecond', 0.002), ('s', 2)]:
            with self.subTest(unit=unit):
                value = profiles.parse_ncu(long_csv(duration_unit=unit), 'PackedInverseMassKernel')
                self.assertAlmostEqual(value['launches'][0]['replay_duration_seconds'], expected)
        with self.assertRaises(profiles.ProfileError):
            profiles.parse_ncu(long_csv(duration_unit=''), 'PackedInverseMassKernel')

    def test_identical_duplicates_accept_conflicting_reject(self):
        duplicate = ['0', KERNEL, profiles.DURATION, 'us', '2.0']
        self.assertEqual(profiles.parse_ncu(long_csv([duplicate]), 'PackedInverseMassKernel')['observed_launches'], 1)
        duplicate[-1] = '3'
        with self.assertRaises(profiles.ProfileError):
            profiles.parse_ncu(long_csv([duplicate]), 'PackedInverseMassKernel')

    def test_unavailable_optional_counters_are_not_zero(self):
        row = ['0', KERNEL, 'smsp__warps_eligible.avg', 'warp', 'N/A']
        launch = profiles.parse_ncu(long_csv([row]), 'PackedInverseMassKernel')['launches'][0]
        metric = next(row for row in launch['metric_rows'] if row['metric'] == 'smsp__warps_eligible.avg')
        self.assertIsNone(metric['value'])
        self.assertEqual(metric['availability'], 'unavailable')
        self.assertFalse(launch['diagnostic_availability']['eligible_warps'])

    def test_zero_memory_bytes_do_not_establish_throughput(self):
        rows = [['0', KERNEL, 'dram__bytes.sum', 'byte', '0'],
                ['0', KERNEL, 'lts__t_sector_hit_rate.pct', '%', '0'],
                ['0', KERNEL, 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum', 'conflict', '0']]
        launch = profiles.parse_ncu(long_csv(rows), 'PackedInverseMassKernel')['launches'][0]
        self.assertFalse(launch['diagnostic_availability']['memory_throughput'])
        self.assertTrue(launch['diagnostic_availability']['cache_hit_rate'])
        self.assertTrue(launch['diagnostic_availability']['shared_bank_conflicts'])

    def test_negative_nan_foreign_kernel_and_missing_duration_reject(self):
        for text in (long_csv().replace(',2\r', ',-1\r'), long_csv().replace(',2\r', ',nan\r'),
                     long_csv().replace(KERNEL, 'foreign'), long_csv().replace(',2\r', ',N/A\r')):
            with self.subTest(text=text), self.assertRaises(profiles.ProfileError):
                profiles.parse_ncu(text, 'PackedInverseMassKernel')

    def test_nsys_sums_keep_units_and_overlap_separate(self):
        value = profiles.parse_nsys(nsys_csv())
        self.assertAlmostEqual(value['cuda_api']['duration_seconds'], 5e-6)
        self.assertAlmostEqual(value['gpu_kernel']['duration_seconds'], 3e-6)
        self.assertAlmostEqual(value['gpu_memory']['duration_seconds'], 2e-6)
        self.assertNotIn('total_seconds', value)

    def test_nsys_host_reports_disambiguate_identical_api_headers(self):
        osrt = profiles.parse_nsys_host_summary(host_summary('osrt_sum'), 'osrt_sum')
        nvtx = profiles.parse_nsys_host_summary(host_summary('nvtx_sum'), 'nvtx_sum')
        projected = profiles.parse_nsys_host_summary(host_summary('nvtx_gpu_proj_sum'), 'nvtx_gpu_proj_sum')
        self.assertEqual(osrt['rows'][0]['name'], 'poll')
        self.assertAlmostEqual(osrt['duration_seconds'], 8e-6)
        self.assertAlmostEqual(nvtx['duration_seconds'], 6e-6)
        self.assertAlmostEqual(projected['duration_seconds'], 4e-6)
        self.assertAlmostEqual(projected['rows'][0]['range_duration_seconds'], 6e-6)
        self.assertEqual(projected['rows'][0]['gpu_operations'], 2)
        self.assertEqual(profiles.parse_nsys_host_summary('SKIPPED: no NVTX data', 'nvtx_sum')['availability'], 'unavailable')
        for text in (host_summary('nvtx_sum').replace('(ms)', '(ticks)'),
                     host_summary('osrt_sum').replace(',2,', ',0,')):
            with self.assertRaises(profiles.ProfileError):
                profiles.parse_nsys_host_summary(text, 'nvtx_sum' if 'Instances' in text else 'osrt_sum')

    def test_cuda_activity_union_avoids_overlapping_sums(self):
        value = profiles.parse_nsys_gpu_activity(io.StringIO(gpu_trace()), 2)
        self.assertAlmostEqual(value['active_union_seconds'], 4e-6)
        self.assertAlmostEqual(value['first_to_last_operation_seconds'], 4e-6)
        self.assertEqual(value['internal_gap_seconds'], 0)
        self.assertEqual(value['internal_gap_count'], 0)
        self.assertEqual(value['active_fraction_of_operation_span'], 1)
        self.assertNotIn('total_seconds', value)

    def test_cuda_activity_gaps_and_large_timestamp_precision(self):
        base = 9000000000000000000
        rows = [[str(base), '0', 'GPU 0', '1', '1', 'zero'],
                [str(base + 1000), '1', 'GPU 0', '1', '1', 'a'],
                [str(base + 3000), '2', 'GPU 0', '1', '2', 'b'],
                [str(base + 8000), '1', 'GPU 0', '1', '1', 'c']]
        value = profiles.parse_nsys_gpu_activity(io.StringIO(gpu_trace(rows)), 4)
        self.assertAlmostEqual(value['active_union_seconds'], 4e-6)
        self.assertAlmostEqual(value['first_to_last_operation_seconds'], 8e-6)
        self.assertAlmostEqual(value['internal_gap_seconds'], 4e-6)
        self.assertEqual(value['internal_gap_count'], 2)
        self.assertEqual(value['zero_duration_events'], 1)
        self.assertEqual(value['largest_internal_gaps_seconds'], [3e-6, 1e-6])

    def test_cuda_activity_bad_coverage_units_device_and_bounds_reject(self):
        examples = [(gpu_trace(), 3, 4), (gpu_trace(), 2, 1),
                    (gpu_trace().replace('Start (ns)', 'Start'), 2, 4),
                    (gpu_trace().replace('Duration (us)', 'Duration (tick)'), 2, 4),
                    (gpu_trace().replace('3000', '500'), 2, 4),
                    (gpu_trace().replace('3000,2,GPU 0', '3000,2,GPU 1'), 2, 4),
                    (gpu_trace().replace('1000,3', '1000,nan'), 2, 4)]
        for text, count, bound in examples:
            with self.subTest(text=text), self.assertRaises(profiles.ProfileError):
                profiles.parse_nsys_gpu_activity(io.StringIO(text), count, bound)

    def test_large_cuda_activity_streams_with_summary_derived_coverage(self):
        count = 250001
        value = profiles.parse_nsys_gpu_activity(large_gpu_trace(count), count)
        self.assertEqual(value['event_count'], count)
        self.assertAlmostEqual(value['active_union_seconds'], count * 1e-6)
        self.assertEqual(value['internal_gap_count'], count - 1)
        self.assertEqual(len(value['largest_internal_gaps_seconds']), 10)
        self.assertNotIn('events', value)
        with self.assertRaisesRegex(profiles.ProfileError, 'event bound exceeded'):
            profiles.parse_nsys_gpu_activity(large_gpu_trace(3), 2)

    def test_tensor_instruction_evidence_observed_zero_and_unavailable(self):
        for case, kernel in (({'adapter': 'dg', 'parameters': {'dgsem_residual': 'fused-dmma'}},
                              'void FusedEulerResidualKernel<double, true>()'),
                             ({'adapter': 'hdg', 'parameters': {'engine': 'tensor'}},
                              'void BorrowedEulerTensorVolumeContractionKernel()'),
                             ({'adapter': 'dg', 'parameters': {'discretization': 'hdg', 'engine': 'tensor'}},
                              'void BorrowedEulerTensorVolumeContractionKernel()')):
            for number, status in (('12', 'observed'), ('0', 'not_observed'), ('N/A', 'unavailable')):
                row = ['0', kernel, profiles.DMMA, 'inst', number]
                text = csv_text([['ID', 'Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value'],
                                 ['0', kernel, profiles.DURATION, 'us', '2'], row])
                parsed = profiles.parse_ncu(text, 'Kernel', metric_limit=1)
                evidence = profiles.tensor_instruction_evidence(case, {'metrics': parsed})
                self.assertEqual(evidence['status'], status)
                self.assertEqual(evidence['selected_tensor_launch_count'], 1)
                self.assertEqual(evidence['positive_counter_launch_count'], int(number == '12'))
                self.assertNotIn('tensor_throughput', evidence)
        evidence = profiles.tensor_instruction_evidence(case, {'status': 'unavailable'})
        self.assertEqual(evidence['status'], 'unavailable')
        self.assertIsNone(evidence['selected_replay_dmma_instruction_sum'])
        self.assertEqual(profiles.tensor_instruction_evidence({'adapter': 'dgsem'}, {})['status'], 'not_applicable')

    def test_native_report_repz_and_ambiguous_encodings(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / 'capture'
            Path(str(prefix) + '.ncu-repz').write_bytes(b'report')
            self.assertEqual(profiles.native_ncu(prefix).suffix, '.ncu-repz')
            Path(str(prefix) + '.ncu-rep').write_bytes(b'report')
            with self.assertRaises(profiles.ProfileError):
                profiles.native_ncu(prefix)


class CaptureTests(unittest.TestCase):
    def test_completed_nsys_checkpoint_precedes_ncu_and_is_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary)
            original = runner.run
            snapshots = []

            def check_before_ncu(label, argv, **kwargs):
                if label.endswith('ncu-sections'):
                    self.assertEqual(len(snapshots), 1)
                    self.assertEqual(snapshots[0]['status'], 'partial')
                    self.assertEqual(snapshots[0]['captures']['nsys']['status'], 'complete')
                    self.assertEqual(snapshots[0]['captures']['ncu']['status'], 'pending')
                    self.assertEqual(len(snapshots[0]['application_logs']), 1)
                return original(label, argv, **kwargs)

            case = {'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2}}
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU), \
                    patch.object(runner, 'run', side_effect=check_before_ncu):
                value = profiles.capture(case, ['app'], runner, directory, GPU, checkpoint=snapshots.append)
            self.assertEqual(value['status'], 'complete')
            self.assertEqual(snapshots[0]['captures']['ncu']['status'], 'pending')
            self.assertEqual(len(snapshots[0]['application_logs']), 1)
            native = snapshots[0]['captures']['nsys']['native']
            self.assertEqual(native['sha256'], profiles.sha256(native['path']))

    def test_real_runner_ncu_timeout_preserves_finished_nsys_and_still_fails(self):
        from gpu_study import Runner
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary)
            tool.write_text('#!' + sys.executable + '\nimport time\ntime.sleep(2)\n')
            real_runner, original = Runner(runner.output, 30), runner.run
            snapshots = []

            def real_ncu(label, argv, **kwargs):
                if label.endswith('/ncu'):
                    return real_runner.run(label, argv, **kwargs)
                return original(label, argv, **kwargs)

            case = {'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel',
                                               'launch_count': 2, 'timeout': 0.05}}
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU), \
                    patch.object(runner, 'run', side_effect=real_ncu), self.assertRaises(profiles.ProfileError):
                profiles.capture(case, ['app'], runner, directory, GPU, checkpoint=snapshots.append)
            self.assertEqual([row['captures']['ncu']['status'] for row in snapshots], ['pending', 'failed'])
            self.assertEqual(snapshots[-1]['status'], 'partial')
            self.assertEqual(snapshots[-1]['captures']['nsys']['status'], 'complete')
            self.assertEqual(len(snapshots[-1]['application_logs']), 1)
            receipt = json.loads((directory / 'ncu/result.json').read_text())
            self.assertEqual((receipt['status'], receipt['returncode']), ('failed', 124))
            self.assertFalse(any(label.endswith('ncu-import') for label, argv, timeout in runner.commands))

    def test_invalid_checkpoint_rejects_before_any_query(self):
        with patch.object(profiles, 'gpu_identity') as identity, self.assertRaises(profiles.ProfileError):
            profiles.capture({}, ['app'], None, Path('/not-used'), GPU, checkpoint=True)
        identity.assert_not_called()

    def test_all_native_stats_queries_force_matching_export_before_optional_and_trace_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary)
            original = runner.run
            def stale_guard(label, argv, **kwargs):
                path = original(label, argv, **kwargs)
                if argv[1:2] == ['stats'] and '--report' in argv and '--force-export=true' not in argv:
                    (path / 'stdout.log').write_text('')
                    (path / 'stderr.log').write_text('WARNING: Existing SQLite export found: capture-nsys.sqlite\n'
                        ' File is older than input file: capture-nsys.nsys-rep\n'
                        ' Use --force-export=true to update export file.\n usage: nsys stats ...\n')
                    (path / 'result.json').write_text(json.dumps({'returncode': 1, 'status': 'failed',
                        'argv': argv, 'cwd': str(path), 'stdout_sha256': profiles.sha256(path / 'stdout.log'),
                        'stderr_sha256': profiles.sha256(path / 'stderr.log')}))
                    raise RuntimeError('stale native SQLite export')
                return path
            case = {'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2}}
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU), \
                    patch.object(runner, 'run', side_effect=stale_guard):
                value = profiles.capture(case, ['app'], runner, directory, GPU)
            self.assertEqual(value['status'], 'complete')
            queries = [argv for label, argv, timeout in runner.commands if argv[1:2] == ['stats'] and '--report' in argv]
            self.assertEqual(len(queries), 5)
            self.assertEqual({argv[argv.index('--report') + 1] for argv in queries},
                             {','.join(profiles.NSYS_CORE_REPORTS), *profiles.NSYS_OPTIONAL_REPORTS})
            for argv in queries:
                self.assertEqual(argv.count('--force-export=true'), 1)
                self.assertEqual(argv[-1], value['captures']['nsys']['native']['path'])
            self.assertIn('--force-export=true', value['captures']['nsys']['profiler_policies']['statistics_export'])
            self.assertFalse(any('--force-export=true' in argv for label, argv, timeout in runner.commands
                                 if '--help-reports' in argv))

    def test_stale_export_warning_is_not_an_optional_no_data_skip(self):
        warning = ('WARNING: Existing SQLite export found: capture-nsys.sqlite\n'
                   ' File is older than input file: capture-nsys.nsys-rep\n'
                   ' Use --force-export=true to update export file.\n usage: nsys stats ...\n')
        with self.assertRaises(profiles.ProfileError):
            profiles.nsys_no_data('', warning, 'nvtx_sum', 1)

    def test_nsys_native_help_exit_one_requires_explicit_listing_and_options(self):
        for code in (0, 1):
            for stdout, stderr in ((reports_help(), ''), ('', reports_help()),
                                   (reports_help(False), '\n  nvtx_sum -- Native report description\n')):
                with self.subTest(code=code, stdout=bool(stdout)):
                    names = profiles.nsys_report_names(stdout, stderr, code)
                    self.assertIn('nvtx_sum', names)
            profiles.nsys_profile_help(PROFILE_HELP, '', code)
            profiles.nsys_profile_help('', PROFILE_HELP, code)
        for text, code in ((reports_help(), 2), (reports_help(), 124),
                           ('ERROR: ' + ' '.join((*profiles.NSYS_CORE_REPORTS, *profiles.NSYS_OPTIONAL_REPORTS)), 1),
                           (reports_help().replace('  cuda_api_sum --', '  cuda_api_trace --'), 1),
                           (reports_help() + '  cuda_api_sum -- duplicate\n', 1), ('', 1)):
            with self.subTest(text=text, code=code), self.assertRaises(profiles.ProfileError):
                profiles.nsys_report_names(text, '', code)
        for text, code in ((PROFILE_HELP, 2), (PROFILE_HELP, 124),
                           (PROFILE_HELP.replace('--trace', '--trace-other'), 1),
                           (PROFILE_HELP.replace('usage: nsys profile [', 'ERROR nsys profile '), 1)):
            with self.subTest(text=text, code=code), self.assertRaises(profiles.ProfileError):
                profiles.nsys_profile_help('', text, code)

    def test_help_exit_one_preserves_both_streams_and_real_runner_receipts(self):
        from gpu_study import Runner
        for stream in ('stdout', 'stderr'):
            with self.subTest(stream=stream), tempfile.TemporaryDirectory() as temporary:
                tool, runner, directory = self.setup_capture(temporary)
                real_runner, original = Runner(runner.output, 30), runner.run
                def real_query(label, argv, **kwargs):
                    if label.endswith(('nsys-reports', 'nsys-profile-help')):
                        return real_runner.run(label, argv, **kwargs)
                    return original(label, argv, **kwargs)
                tool.write_text('#!' + sys.executable + '\nimport sys\n' +
                    'text = ' + repr(reports_help()) + ' if "--help-reports" in sys.argv else ' + repr(PROFILE_HELP) + '\n' +
                    'print(text, file=sys.' + stream + ')\nsys.exit(1)\n')
                case = {'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2}}
                with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU), \
                        patch.object(runner, 'run', side_effect=real_query):
                    value = profiles.capture(case, ['app'], runner, directory, GPU)
                self.assertEqual(value['status'], 'complete')
                native = value['tools']['nsys']
                self.assertEqual(native['report_query_returncode'], 1)
                self.assertEqual(native['profile_help_returncode'], 1)
                for key in ('report_query', 'report_query_stderr', 'profile_help_stdout', 'profile_help_stderr'):
                    self.assertEqual(native[key]['sha256'], profiles.sha256(native[key]['path']))
                self.assertGreater(native['report_query' if stream == 'stdout' else 'report_query_stderr']['bytes'], 0)

    def test_malformed_or_timeout_help_does_not_reach_capture(self):
        for code, text in ((1, 'no report help available'), (2, reports_help()), (124, reports_help())):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                tool, runner, directory = self.setup_capture(temporary)
                original = runner.run
                def bad_query(label, argv, **kwargs):
                    path = original(label, argv, **kwargs)
                    if label.endswith('nsys-reports'):
                        (path / 'stdout.log').write_text(text)
                        (path / 'result.json').write_text(json.dumps({'returncode': code}))
                    return path
                case = {'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2}}
                with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU), \
                        patch.object(runner, 'run', side_effect=bad_query), self.assertRaises(profiles.ProfileError):
                    profiles.capture(case, ['app'], runner, directory, GPU)
                self.assertFalse(any(label.endswith('/nsys') for label, argv, timeout in runner.commands))

    def test_optional_native_no_data_skip_is_explicit_and_nonzero_errors_reject(self):
        for report, description in (('nvtx_sum', 'NV Tools Extension (NVTX)'),
                                    ('nvtx_gpu_proj_sum', 'NVTX'), ('osrt_sum', 'OS Runtime')):
            text = f'SKIPPED: report.sqlite does not contain {description} data.\n'
            row = profiles.nsys_no_data('', text, report, 1)
            self.assertEqual(row['availability'], 'unavailable')
            self.assertIsNone(row['duration_seconds'])
            for invalid, code in ((text, 2), (text, 124), (text + 'ERROR: cannot read trace\n', 1),
                                  ('SKIPPED: report failed\n', 1), (text.replace(description, 'other'), 1)):
                with self.subTest(report=report, code=code), self.assertRaises(profiles.ProfileError):
                    profiles.nsys_no_data(invalid, '', report, code)

    def test_empty_nvtx_exit_one_retains_core_and_skip_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary)
            original = runner.run
            def absent_nvtx(label, argv, **kwargs):
                path = original(label, argv, **kwargs)
                if label.endswith(('nsys-nvtx_sum', 'nsys-nvtx_gpu_proj_sum')):
                    (path / 'stdout.log').write_text('')
                    (path / 'stderr.log').write_text('SKIPPED: trace.sqlite does not contain NV Tools Extension (NVTX) data.\n')
                    (path / 'result.json').write_text(json.dumps({'returncode': 1, 'status': 'failed',
                        'argv': argv, 'cwd': str(path), 'stdout_sha256': profiles.sha256(path / 'stdout.log'),
                        'stderr_sha256': profiles.sha256(path / 'stderr.log')}))
                    raise RuntimeError('native optional report exited1')
                return path
            case = {'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2}}
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU), \
                    patch.object(runner, 'run', side_effect=absent_nvtx):
                value = profiles.capture(case, ['app'], runner, directory, GPU)
            self.assertEqual(value['status'], 'complete')
            for report in ('nvtx_sum', 'nvtx_gpu_proj_sum'):
                row = value['captures']['nsys']['host_summaries'][report]
                self.assertEqual(row['availability'], 'unavailable')
                self.assertEqual(row['returncode'], 1)
                self.assertEqual(row['statistics_csv']['bytes'], 0)
                self.assertGreater(row['statistics_stderr']['bytes'], 0)

    def test_host_reports_and_timestamp_processing_keep_compact_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary)
            case = {'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2}}
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU):
                value = profiles.capture(case, ['app'], runner, directory, GPU)
            nsys = value['captures']['nsys']
            self.assertEqual(value['status'], 'complete')
            self.assertEqual(set(nsys['host_summaries']), set(profiles.NSYS_OPTIONAL_REPORTS[:3]))
            self.assertEqual(nsys['gpu_activity']['event_count'], 2)
            self.assertFalse(nsys['profiler_policies']['sampled_cpu_stack_attribution_available'])
            self.assertEqual(value['tools']['nsys']['environment_capability']['availability'], 'available')
            self.assertNotIn('events', nsys['gpu_activity'])
            self.assertLess(len(json.dumps(nsys)), 12000)
            commands = {label: argv for label, argv, timeout in runner.commands}
            for report in profiles.NSYS_OPTIONAL_REPORTS:
                self.assertEqual(commands['profiles/nsys-' + report][3], report)
            self.assertIn('--sample=none', commands['profiles/nsys'])

    def test_capability_query_failure_preserves_receipt_without_enabling_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary, fail='profiles/tools/nsys-environment')
            case = {'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2}}
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU):
                value = profiles.capture(case, ['app'], runner, directory, GPU)
            self.assertEqual(value['status'], 'complete')
            capability = value['tools']['nsys']['environment_capability']
            self.assertEqual(capability['availability'], 'unavailable')
            self.assertEqual(capability['returncode'], 1)
            self.assertTrue(Path(capability['stderr']['path']).is_file())
            nsys = next(argv for label, argv, timeout in runner.commands if label == 'profiles/nsys')
            self.assertIn('--sample=none', nsys)

    def test_real_runner_optional_query_nonzero_recovers_exact_receipt(self):
        from gpu_study import Runner
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary)
            real_runner = Runner(runner.output, 30)
            original = runner.run
            def real_query(label, argv, **kwargs):
                if label.endswith('nsys-environment'):
                    # Substitute only the fake profiler, retaining the exact
                    # provider argv in the receipt produced by the real Runner.
                    return real_runner.run(label, argv, **kwargs)
                return original(label, argv, **kwargs)
            case = {'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2}}
            # The tool identity is frozen before capability queries, so write
            # the intercepted executable before entering capture.
            tool.write_text('#!' + sys.executable + '\nimport sys\nsys.exit(2)\n')
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU), \
                    patch.object(runner, 'run', side_effect=real_query):
                value = profiles.capture(case, ['app'], runner, directory, GPU)
            self.assertEqual(value['status'], 'complete')
            self.assertEqual(value['tools']['nsys']['environment_capability']['returncode'], 2)

    def test_optional_query_does_not_swallow_missing_receipt_or_timeout(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as temporary:
                tool, runner, directory = self.setup_capture(temporary)
                original = runner.run
                def broken_query(label, argv, **kwargs):
                    if not label.endswith('nsys-environment'):
                        return original(label, argv, **kwargs)
                    if timeout:
                        path = original(label, argv, **kwargs)
                        (path / 'result.json').write_text(json.dumps({'status': 'failed', 'returncode': 124,
                            'argv': argv, 'cwd': str(path)}))
                    raise RuntimeError('bounded query failed')
                case = {'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2}}
                with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU), \
                        patch.object(runner, 'run', side_effect=broken_query), self.assertRaises(profiles.ProfileError):
                    profiles.capture(case, ['app'], runner, directory, GPU)

    def test_optional_reports_unavailable_do_not_invent_activity_or_fail_core(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary, missing_reports=True)
            case = {'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2}}
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU):
                value = profiles.capture(case, ['app'], runner, directory, GPU)
            self.assertEqual(value['status'], 'complete')
            nsys = value['captures']['nsys']
            self.assertEqual(nsys['gpu_activity']['availability'], 'unavailable')
            self.assertNotIn('active_union_seconds', nsys['gpu_activity'])
            self.assertTrue(all(row['availability'] == 'unavailable' for row in nsys['host_summaries'].values()))
            self.assertFalse(any(label.endswith('nsys-cuda_gpu_trace') for label, argv, timeout in runner.commands))

    def test_empty_nvtx_exports_keep_zero_byte_receipts_and_no_range_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary)
            original = runner.run
            def empty_nvtx(label, argv, **kwargs):
                result = original(label, argv, **kwargs)
                if label.endswith(('nsys-nvtx_sum', 'nsys-nvtx_gpu_proj_sum')):
                    (result / 'stdout.log').write_text('')
                return result
            case = {'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2}}
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU), \
                    patch.object(runner, 'run', side_effect=empty_nvtx):
                value = profiles.capture(case, ['app'], runner, directory, GPU)
            self.assertEqual(value['status'], 'complete')
            for report in ('nvtx_sum', 'nvtx_gpu_proj_sum'):
                rows = value['captures']['nsys']['host_summaries'][report]
                self.assertEqual(rows['availability'], 'unavailable')
                self.assertIsNone(rows['duration_seconds'])
                self.assertEqual(rows['statistics_csv']['bytes'], 0)

    def test_large_timestamp_export_is_streamed_on_compute(self):
        count = 250001
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary)
            original = runner.run
            def large_trace(label, argv, **kwargs):
                result = original(label, argv, **kwargs)
                if label.endswith('nsys-stats'):
                    text = (result / 'stdout.log').read_text()
                    text = text.replace(f'3,1,{KERNEL}', f'3,{count - 1},{KERNEL}')
                    (result / 'stdout.log').write_text(text)
                elif label.endswith('nsys-cuda_gpu_trace'):
                    with (result / 'stdout.log').open('w') as stream:
                        stream.writelines(large_gpu_trace(count))
                return result
            case = {'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2}}
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU), \
                    patch.object(runner, 'run', side_effect=large_trace):
                value = profiles.capture(case, ['app'], runner, directory, GPU)
            self.assertEqual(value['status'], 'complete')
            activity = value['captures']['nsys']['gpu_activity']
            self.assertEqual(activity['event_count'], count)
            self.assertNotIn('events', activity)
            self.assertTrue(any(label.endswith('nsys-cuda_gpu_trace') and timeout == 120
                                for label, argv, timeout in runner.commands))

    def test_tensor_dispatch_environment_reaches_both_capture_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary)
            case = {'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2}}
            environment = {'DG_IGR_DGSEM_FUSED_RESIDUAL': '1', 'DG_IGR_DGSEM_FUSED_DMMA': '1'}
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU):
                value = profiles.capture(case, ['app'], runner, directory, GPU, environment=environment)
            self.assertEqual(value['status'], 'complete')
            for tool_name in ('nsys', 'ncu'):
                self.assertEqual(runner.environments['profiles/' + tool_name], environment)
            for logs in value['application_logs']:
                self.assertEqual(json.loads(Path(logs['receipt']).read_text())['returncode'], 0)

    def setup_capture(self, temporary, **runner_options):
        root = Path(temporary)
        tool = root / 'native-tool'
        tool.write_text('native binary fixture')
        tool.chmod(0o700)
        output = root / 'output'
        output.mkdir()
        return tool, FakeRunner(output, **runner_options), output / 'profiles'

    def test_bounded_hdg_selected_step_imports_repz(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary)
            case = {'adapter': 'hdg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2}}
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU):
                value = profiles.capture(case, ['app', '--profile-measured-step', '0'], runner, directory, GPU)
            self.assertEqual(value['status'], 'complete')
            self.assertEqual(value['captures']['ncu']['native']['path'], str(directory / 'capture-ncu.ncu-repz'))
            ncu = next(argv for label, argv, timeout in runner.commands if label == 'profiles/ncu')
            self.assertEqual(ncu[ncu.index('--profile-from-start') + 1], 'off')
            self.assertEqual(ncu[ncu.index('--launch-count') + 1], '2')
            self.assertEqual(ncu[ncu.index('--kernel-name') + 1], 'regex:PackedInverseMassKernel')
            self.assertNotIn('--kernel-id', ncu)
            self.assertIn(profiles.DMMA, ncu[ncu.index('--metrics') + 1])
            self.assertTrue(all(timeout <= 240 for label, argv, timeout in runner.commands))
            self.assertEqual(len(value['application_logs']), 2)

    def test_unique_kernel_selection_has_no_shared_launch_cap(self):
        class UniqueRunner(FakeRunner):
            def run(self, label, argv, **kwargs):
                directory = super().run(label, argv, **kwargs)
                if label.endswith('ncu-import'):
                    rows = list(csv.reader(io.StringIO(wide_complete())))
                    rows[-1][2] = 'void BorrowedEulerFaceKernel()'
                    (directory / 'stdout.log').write_text(csv_text(rows))
                return directory
        with tempfile.TemporaryDirectory() as temporary:
            tool, old_runner, directory = self.setup_capture(temporary)
            runner = UniqueRunner(old_runner.output)
            case = {'adapter': 'hdg', 'profile': {'kernel_regex': '(?:PackedInverseMassKernel|BorrowedEulerFaceKernel)',
                'launch_count': 2, 'selection': 'first-per-kernel',
                'kernel_groups': {'mass': 'PackedInverseMassKernel', 'face': 'BorrowedEulerFaceKernel'}}}
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU):
                value = profiles.capture(case, ['app', '--profile-measured-step', '0'], runner, directory, GPU)
            ncu = next(argv for label, argv, timeout in runner.commands if label == 'profiles/ncu')
            self.assertEqual(ncu[ncu.index('--kernel-id') + 1],
                             '::regex:(PackedInverseMassKernel|BorrowedEulerFaceKernel):1')
            self.assertNotIn('--kernel-name', ncu)
            self.assertNotIn('--launch-count', ncu)
            self.assertNotIn('--launch-skip', ncu)
            self.assertEqual(value['captures']['ncu']['metrics']['observed_kernel_groups'], ['face', 'mass'])

    def test_single_selector_preserves_namespaces_and_regex_without_weaker_import_filter(self):
        class NamespacedRunner(FakeRunner):
            def run(self, label, argv, **kwargs):
                directory = super().run(label, argv, **kwargs)
                if label.endswith('ncu-import'):
                    rows = list(csv.reader(io.StringIO(wide_complete())))
                    rows[-2][2] = 'void BorrowedCondenseKernel(DeviceView, MeshShape)'
                    rows[-1][2] = 'void cusparse::spsm_v2_kernel<int, double>(int, double*)'
                    (directory / 'stdout.log').write_text(csv_text(rows))
                return directory
        regex = '.*(?:BorrowedCondenseKernel|cusparse::spsm_v2_kernel).*'
        with tempfile.TemporaryDirectory() as temporary:
            tool, previous, directory = self.setup_capture(temporary)
            runner = NamespacedRunner(previous.output)
            case = {'adapter': 'hdg', 'profile': {'kernel_regex': regex, 'launch_count': 2,
                'selection': 'first-per-kernel', 'kernel_groups': {
                    'condensation': 'BorrowedCondenseKernel', 'triangular': 'cusparse::spsm_v2_kernel'}}}
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU):
                result = profiles.capture(case, ['app', '--profile-measured-step', '0'], runner, directory, GPU)
            ncu = next(argv for label, argv, timeout in runner.commands if label == 'profiles/ncu')
            encoded = r'.*(BorrowedCondenseKernel|cusparse\x3a\x3aspsm_v2_kernel).*'
            self.assertEqual(ncu[ncu.index('--kernel-id') + 1], '::regex:' + encoded + ':1')
            self.assertNotIn('--kernel-name', ncu)
            self.assertEqual(result['selected_kernel_regex'], regex)
            self.assertEqual(result['ncu_kernel_id_regex'], encoded)
            self.assertEqual(result['captures']['ncu']['metrics']['observed_kernel_groups'],
                             ['condensation', 'triangular'])
        text = long_csv([['1', 'void UnrequestedZeroKernel()', profiles.DURATION, 'us', '1']])
        with self.assertRaisesRegex(profiles.ProfileError, r'launch=1, kernel=.*UnrequestedZeroKernel'):
            profiles.parse_ncu(text, 'PackedInverseMassKernel')

    def test_colon_free_selector_preserves_search_membership_and_escaped_literals(self):
        patterns = [r'.*(?:foo|ns::bar).*', r'[:](?:foo|bar)', r'foo\:bar',
                    r'\(\?:literal', r'[(?:]+foo', r'[]:(]+foo', r'(foo)::\1']
        names = ['foo', 'bar', 'ns::bar', 'void ns::bar(int)', 'void baz()',
                 ':foo', 'foo:bar', '(?:literal', '(:foo', ']foo', 'foo::foo']
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                encoded = profiles._kernel_id_regex(pattern)
                self.assertNotIn(':', encoded)
                self.assertEqual([bool(re.search(pattern, name)) for name in names],
                                 [bool(re.search(encoded, name)) for name in names])

    def test_colon_free_selector_rejects_matching_changing_group_conversions(self):
        for pattern in (r'(?:foo)(bar)\1', r'(foo)(?:bar)(?(1)yes|no)', r'(?i:foo)'):
            with self.subTest(pattern=pattern), self.assertRaises(profiles.ProfileError):
                profiles._kernel_id_regex(pattern)

    def test_all_ncu_commands_disable_ambient_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary)
            case = {'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2}}
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU):
                value = profiles.capture(case, ['app'], runner, directory, GPU)
            commands = [argv for label, argv, timeout in runner.commands if 'ncu' in label]
            self.assertEqual(len(commands), 6)
            for command in commands:
                self.assertEqual(command[1:3], ['--config-file', 'off'])
            self.assertEqual(value['tools']['ncu']['config_file'], 'off (explicit)')
            self.assertEqual(value['captures']['ncu']['profiler_policies']['config_file'], 'off (explicit)')

    def test_missing_declared_kernel_group_rejects(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary)
            case = {'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2,
                                               'kernel_groups': {'face': 'BorrowedEulerFaceKernel'}}}
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU), \
                    self.assertRaises(profiles.ProfileError):
                profiles.capture(case, ['app'], runner, directory, GPU)

    def test_missing_core_diagnostics_leave_capture_partial(self):
        class SparseRunner(FakeRunner):
            def run(self, label, argv, **kwargs):
                directory = super().run(label, argv, **kwargs)
                if label.endswith('ncu-import'):
                    (directory / 'stdout.log').write_text(wide_csv())
                return directory
        with tempfile.TemporaryDirectory() as temporary:
            tool, old_runner, directory = self.setup_capture(temporary)
            runner = SparseRunner(old_runner.output)
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU):
                value = profiles.capture({'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2}},
                                         ['app'], runner, directory, GPU)
            self.assertEqual(value['status'], 'partial')
            self.assertEqual(value['captures']['ncu']['status'], 'partial')
            self.assertTrue(value['captures']['ncu']['missing_required_diagnostics'])
            self.assertTrue((directory / 'capture-ncu.ncu-repz').is_file())

    def test_fewer_than_requested_launches_remain_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary)
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU):
                value = profiles.capture({'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 3}},
                                         ['app'], runner, directory, GPU)
            self.assertEqual(value['status'], 'partial')
            self.assertFalse(value['captures']['ncu']['launch_coverage_complete'])
            self.assertEqual(value['captures']['ncu']['requested_launches'], 3)
            self.assertEqual(value['captures']['ncu']['metrics']['observed_launches'], 2)
            self.assertFalse(value['captures']['ncu']['missing_required_diagnostics'])

    def test_dg_profiles_from_start_and_missing_sections_are_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary, missing_sections=True)
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU):
                value = profiles.capture({'adapter': 'dg', 'profile': {'kernel_regex': 'PackedInverseMassKernel'}},
                                         ['app'], runner, directory, GPU)
            self.assertEqual(value['status'], 'partial')
            self.assertEqual(value['captures']['ncu']['status'], 'unavailable')
            self.assertFalse(any(label == 'profiles/ncu' for label, argv, timeout in runner.commands))

    def test_dg_and_dgsem_capture_from_start_without_hdg_range(self):
        for adapter in ('dg', 'dgsem'):
            with self.subTest(adapter=adapter), tempfile.TemporaryDirectory() as temporary:
                tool, runner, directory = self.setup_capture(temporary)
                case = {'adapter': adapter, 'profile': {'kernel_regex': 'PackedInverseMassKernel', 'launch_count': 2}}
                with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU):
                    value = profiles.capture(case, ['app'], runner, directory, GPU)
                self.assertEqual(value['status'], 'complete')
                ncu = next(argv for label, argv, timeout in runner.commands if label == 'profiles/ncu')
                self.assertEqual(ncu[ncu.index('--profile-from-start') + 1], 'on')
                nsys = next(argv for label, argv, timeout in runner.commands if label == 'profiles/nsys')
                self.assertFalse(any(argument.startswith('--capture-range') for argument in nsys))

    def test_missing_profilers_are_structured_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary)
            with patch.object(profiles, '_tool', return_value=None), patch.object(profiles, 'gpu_identity', return_value=GPU):
                value = profiles.capture({'adapter': 'dg', 'profile': {'kernel_regex': 'Mass'}}, ['app'], runner, directory, GPU)
            self.assertEqual(value['status'], 'unavailable')
            self.assertFalse(runner.commands)

    def test_failed_command_retains_logs_and_raises(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary, fail='profiles/nsys')
            with patch.object(profiles, '_tool', return_value=tool), patch.object(profiles, 'gpu_identity', return_value=GPU), \
                    self.assertRaises(profiles.ProfileError):
                profiles.capture({'adapter': 'dg', 'profile': {'kernel_regex': 'Mass'}}, ['app'], runner, directory, GPU)
            self.assertTrue((directory / 'nsys/result.json').is_file())

    def test_wrong_gpu_and_missing_hdg_step_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool, runner, directory = self.setup_capture(temporary)
            case = {'adapter': 'hdg', 'profile': {'kernel_regex': 'Mass'}}
            with self.assertRaises(profiles.ProfileError):
                profiles.capture(case, ['app'], runner, directory, GPU)
            with patch.object(profiles, 'gpu_identity', return_value={**GPU, 'uuid': 'other'}), self.assertRaises(profiles.ProfileError):
                profiles.capture(case, ['app', '--profile-measured-step', '0'], runner, directory, GPU)


if __name__ == '__main__':
    unittest.main()
