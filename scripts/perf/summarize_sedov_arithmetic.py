#!/usr/bin/env python3
"""Summarize completed Sedov arithmetic ablations; failures receive no speed credit.

Reads small existing JSON records only. Does not run a solver or profiler.
  python scripts/perf/summarize_sedov_arithmetic.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / 'tests/research/results/sedov_arithmetic_20260927'
DEFAULT_OUTPUT = REPO / 'docs/research/sedov_arithmetic_20260927'
QUALITY_FIELDS = ('density', 'pressure', 'velocity_vector', 'sigma', 'div_velocity')
SCOPE = ('Only selected volume contractions use reduced precision. Coefficient construction, '
         'local condensation, faces, trace solve, state storage and time integration remain FP64. '
         'The controls independently select FP64 WMMA or ordinary CUDA contractions/condensation. '
         'These are not end-to-end FP32 measurements.')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method_label(run):
    volume = {'fp64-wmma': 'FP64 WMMA', 'simt-fp64': 'FP64 CUDA',
              'simt-fp32': 'FP32 CUDA Jacobian', 'tf32': 'TF32 Jacobian',
              'tf32x3': 'TF32 ×3 Jacobian'}[run['mode']]
    if run['mode'] not in ('fp64-wmma', 'simt-fp64'):
        volume += '\nFP64 residual' if run.get('residual') == 'fp64' else '\nmatched residual'
    elif (run['mode'] == 'fp64-wmma') != (run.get('condensation') == 'fp64-wmma'):
        volume += '\n' + run.get('condensation', '?') + ' condensation'
    return volume


def key(run):
    return tuple(run.get(k) for k in ('binary_sha256', 'mode', 'residual', 'condensation',
                                      'horizon', 'rtol', 'atol', 'conservation_rtol', 'nonlinear_rtol'))


def aggregate(records):
    grouped = defaultdict(list)
    excluded = []
    for run in records:
        if (run.get('status') != 'complete' or not run.get('accepted') or run.get('profiled')
                or not math.isclose(run.get('horizon', -1), .1, abs_tol=1e-12)
                or not run.get('analysis_data')):
            excluded.append(run)
            continue
        seconds = run.get('monotonic_raw_seconds')
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            excluded.append(run)
            continue
        grouped[key(run)].append(run)
    groups = []
    for index, runs in enumerate(grouped.values()):
        r = runs[0]
        times = [v['monotonic_raw_seconds'] for v in runs]
        quality = defaultdict(list)
        for v in runs:
            for ref in v['analysis_data']['references']:
                quality[ref['name']].append(ref)
        references = []
        for name, refs in quality.items():
            fields = {}
            for field in QUALITY_FIELDS:
                values = [v['differences'][field] for v in refs if field in v['differences']]
                fields[field] = {metric: max(v[metric] for v in values if v.get(metric) is not None)
                                 for metric in ('l2', 'relative_l2', 'max_abs_at_quadrature')
                                 if any(v.get(metric) is not None for v in values)}
            references.append({'name': name, 'run_json': refs[0]['run_json'], 'count': len(refs),
                               'max_over_runs': fields})
        groups.append({'id': f'group{index+1}', 'label': method_label(r),
            'mode': r['mode'], 'residual': r.get('residual'), 'condensation': r.get('condensation'),
            'binary_sha256': r['binary_sha256'], 'source_build_manifest': r.get('build_manifest'),
            'horizon': r['horizon'], 'time_rtol': r['rtol'], 'time_atol': r['atol'],
            'conservation_rtol': r.get('conservation_rtol'), 'nonlinear_rtol': r.get('nonlinear_rtol'),
            'names': [v['name'] for v in runs], 'n': len(runs),
            'raw_seconds': {'values': times, 'median': statistics.median(times),
                            'min': min(times), 'max': max(times)},
            'steps': sorted({int(v['summaries']['[hdg-main]:setup']['step']) for v in runs}),
            'rejected_attempts': sorted({int(v['summaries']['[hdg-main]:setup']['rejected_attempts']) for v in runs}),
            'mass_relative_change_max_abs': max(abs(v['analysis_data']['integrals']['mass_relative_change']) for v in runs),
            'energy_relative_change_max_abs': max(abs(v['analysis_data']['integrals']['energy_relative_change']) for v in runs),
            'perf_to_raw_ratios': [v['perf_counter_seconds']/v['monotonic_raw_seconds'] for v in runs],
            'references': references})
    order = {'fp64-wmma': 0, 'simt-fp64': 1, 'simt-fp32': 2, 'tf32': 3, 'tf32x3': 4}
    groups.sort(key=lambda g: (g['conservation_rtol'] or 0, g['nonlinear_rtol'] or 0, order[g['mode']], g['binary_sha256']))
    return groups, excluded


def make_figure(groups, excluded, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(10.1, max(3.7, 1.0*len(groups)+2.0)))
    colors = {'fp64-wmma': '#ed7d31', 'simt-fp64': '#414141', 'simt-fp32': '#557a95',
              'tf32': '#9e587b', 'tf32x3': '#7a6c9d'}
    for i, g in enumerate(groups):
        t = g['raw_seconds']
        ax.barh(i, t['median'], color=colors[g['mode']], height=.58)
        if g['n'] >= 2:
            ax.errorbar(t['median'], i, xerr=np.array([[t['median']-t['min']], [t['max']-t['median']]]),
                        color='black', capsize=5, marker='o', markersize=3, linewidth=1.5)
        ax.text(t['max']+1.2, i, f"{t['median']:.2f} s  (n={g['n']})", va='center', fontsize=11)
        if i and (g['conservation_rtol'],g['nonlinear_rtol']) != (groups[i-1]['conservation_rtol'],groups[i-1]['nonlinear_rtol']):
            ax.axhline(i-.5, color='#aaaaaa', linestyle=':', linewidth=1)
    labels = [g['label'] + f"\nconservation tol. {g['conservation_rtol']:.0e}" for g in groups]
    ax.set_yticks(range(len(groups)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(g['raw_seconds']['max'] for g in groups)*1.36)
    ax.set_xlabel('Complete process time (raw monotonic seconds)')
    ax.set_title('Sedov: same trace solver, changed dense arithmetic', loc='left', pad=24)
    ax.text(0, 1.035, '32 × 32 elements · p = 3 · full-energy Euler + IGR · physical horizon t = 0.1',
            transform=ax.transAxes, fontsize=10.5)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color='#dddddd', linewidth=.5)
    ax.set_axisbelow(True)
    failed = sum(r.get('status') == 'failed' for r in excluded)
    footer = 'Median; whiskers show min–max only with ≥2 runs. Single trials have no uncertainty estimate.\n'
    footer += 'Reduced precision affects volume Jacobian only where labeled; these are not full FP32 solvers.'
    if failed:
        footer += f'  {failed} failed attempts excluded.'
    fig.text(.015, .015, footer, fontsize=9, va='bottom')
    fig.tight_layout(rect=(0, .105, 1, 1))
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(output/f'end_to_end_timings.{ext}', dpi=240, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def markdown(summary):
    lines = ['# Sedov arithmetic experiment: measured tables', '', summary['scope'], '',
             'Timing: complete fresh processes, including setup, rejected work and saved output. '
             'Groups with different tolerances or binaries are separate. Failed/incomplete runs receive no performance ranking.', '',
             '| Arithmetic | Conservation tolerance | Nonlinear tolerance | Runs | Median raw s | Min–max s | Steps | Rejected |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in summary['groups']:
        t = g['raw_seconds']
        span = f"{t['min']:.4f}–{t['max']:.4f}" if g['n'] >= 2 else 'single trial'
        lines.append(f"| {g['label'].replace(chr(10), '; ')} | {g['conservation_rtol']:.0e} | {g['nonlinear_rtol']:.0e} | {g['n']} | {t['median']:.4f} | {span} | {g['steps']} | {g['rejected_attempts']} |")
    lines += ['', '## Endpoint differences from numerical references', '',
              'Entries are the largest absolute L2 differences among repeats in each group. '
              'Velocity is the vector norm; div(u) is the cell-local derivative of reconstructed conservative fields, without face lift. '
              'Reference differences are not continuum errors.', '',
              '| Arithmetic / conservation tolerance | Reference | Density | Pressure | Velocity | IGR stress | div(u) |',
              '|---|---|---:|---:|---:|---:|---:|']
    for g in summary['groups']:
        for ref in g['references']:
            errors = ' | '.join(f"{ref['max_over_runs'][field]['l2']:.3e}" for field in QUALITY_FIELDS)
            lines.append(f"| {g['label'].replace(chr(10), '; ')} / {g['conservation_rtol']:.0e} | {ref['name']} | {errors} |")
    lines += ['', '## Conservation diagnostics', '',
              '| Arithmetic / conservation tolerance | Max absolute relative mass change | Max absolute relative energy change |',
              '|---|---:|---:|']
    for g in summary['groups']:
        lines.append(f"| {g['label'].replace(chr(10), '; ')} / {g['conservation_rtol']:.0e} | {g['mass_relative_change_max_abs']:.3e} | {g['energy_relative_change_max_abs']:.3e} |")
    lines += ['', '## Excluded full-horizon attempts', '', '| Run | Status | Reason |', '|---|---|---|']
    for r in summary['excluded']:
        reason = r.get('validation_error') or 'Incomplete or not eligible for unprofiled endpoint comparison'
        lines.append(f"| {r['name']} | {r['status']} | {reason.replace('|', '/')} |")
    lines += ['', 'Exact commands, build identities and field-reference hashes are in each input run record and summary.json.', '']
    return '\n'.join(lines)



def profile_summaries(input_root, output):
    labels = {
        'prof-coefficient': 'FP64 coefficient construction',
        'prof-wmma64-volume': 'FP64 WMMA volume',
        'prof-simt64-volume': 'FP64 CUDA volume',
        'prof-simt32-r64-volume': 'FP32 CUDA Jacobian + FP64 residual',
        'prof-tf32x3-r64-volume': 'TF32 ×3 Jacobian + FP64 residual',
        'prof-wmma64-condensation': 'FP64 WMMA condensation',
        'prof-simt64-condensation': 'FP64 CUDA condensation'}
    metrics = {'time': 'gpu__time_duration.sum', 'dmma': 'sm__inst_executed_pipe_fp64_op_dmma.sum',
               'hmma': 'sm__inst_executed_pipe_tensor_subpipe_hmma_op_hmma.sum'}
    entries = []
    for name, label in labels.items():
        path = input_root/name/'metrics.csv'
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open()))
        units = [r for r in rows if not r.get('ID', '').strip()]
        captures = [r for r in rows if r.get('ID', '').strip()]
        if len(units) != 1 or len(captures) != 3 or len({r['ID'] for r in captures}) != 3:
            raise ValueError(f'{path}: expected one units row and three distinct captures')
        kernels = {r['Kernel Name'] for r in captures}
        if len(kernels) != 1:
            raise ValueError(f'{path}: captures do not share one kernel')
        if units[0][metrics['time']] != 'ns' or any(units[0][metrics[k]] != 'inst' for k in ('dmma', 'hmma')):
            raise ValueError(f'{path}: unexpected timing/count units')
        values = {key: [float(r[column].replace(',', '')) for r in captures] for key, column in metrics.items()}
        if any(not math.isfinite(v) or v < 0 for series in values.values() for v in series) or min(values['time']) <= 0:
            raise ValueError(f'{path}: nonfinite/nonpositive timing or invalid counts')
        if any(v != int(v) for k in ('dmma', 'hmma') for v in values[k]):
            raise ValueError(f'{path}: noninteger instruction count')
        record_path = path.parent/'run.json'
        record = json.loads(record_path.read_text())
        if not record.get('accepted') or not record.get('profiled'):
            raise ValueError(f'{record_path}: expected completed profiled run')
        duration = [v*1e-6 for v in values['time']]
        entries.append({'name': name, 'label': label, 'captures': len(captures),
            'kernel': next(iter(kernels)), 'binary_sha256': record['binary_sha256'],
            'source_manifest': record.get('build_manifest'), 'executed_command': record['executed_command'],
            'physical_horizon': record['horizon'], 'mode': record['mode'], 'residual': record['residual'],
            'condensation': record['condensation'], 'duration_ms': {'values': duration,
                'median': statistics.median(duration), 'min': min(duration), 'max': max(duration)},
            'dynamic_instructions': {k: {'values': [int(v) for v in values[k]],
                'median': statistics.median(values[k]), 'min': min(values[k]), 'max': max(values[k])}
                for k in ('dmma', 'hmma')},
            'replay_passes': [int(r['profiler__replayer_passes']) for r in captures],
            'sources': {str(q.resolve()): sha(q) for q in (path, record_path, *path.parent.glob('*.ncu-rep'))}})
    limitation = ('Three early filtered launches from separate profiler processes to t=0.005. '
                  'Profiler replay/cache behavior applies; these are not independent timing trials, '
                  'confidence intervals, complete-phase totals or fractions of the full trajectory. '
                  'Dynamic instruction counts establish execution, not an isolated hardware speedup.')
    profile = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'scope': limitation,
               'metrics': metrics, 'profiles': entries, 'script_sha256': sha(Path(__file__))}
    by_name = {r['name']: r for r in entries}
    if 'prof-coefficient' in by_name:
        coeff = by_name['prof-coefficient']['duration_ms']['median']
        profile['separate_median_sums_coefficient_plus_volume_ms'] = {
            name: coeff + item['duration_ms']['median'] for name, item in by_name.items() if name.endswith('-volume')}
        profile['median_sum_note'] = 'Sum of independently measured medians; not a directly measured combined launch duration.'
    (output/'kernel_profiles.json').write_text(json.dumps(profile, indent=2, allow_nan=False)+'\n')
    lines = ['# Kernel profile samples', '', limitation, '',
             '| Kernel path | Median ms | Observed min–max ms | DMMA instructions / launch | HMMA instructions / launch |',
             '|---|---:|---:|---:|---:|']
    for entry in entries:
        t, c = entry['duration_ms'], entry['dynamic_instructions']
        lines.append(f"| {entry['label']} | {t['median']:.6f} | {t['min']:.6f}–{t['max']:.6f} | {c['dmma']['median']:,.0f} | {c['hmma']['median']:,.0f} |")
    lines += ['', 'The ordinary CUDA captures have zero DMMA and zero HMMA counts. '
              'Mixed volume measurements include their FP64 residual recomputation.', '']
    (output/'kernel_profiles.md').write_text('\n'.join(lines))
    if entries:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10.0, 5.8))
        for i, entry in enumerate(entries):
            t = entry['duration_ms']
            color = '#ed7d31' if 'wmma' in entry['name'] else ('#7a6c9d' if 'tf32' in entry['name'] else '#557a95')
            ax.barh(i, t['median'], color=color, height=.58)
            ax.text(t['median']+.12, i, f"{t['median']:.3f} ms", va='center')
        ax.set_yticks(range(len(entries)), [v['label'] for v in entries])
        ax.invert_yaxis()
        ax.set_xlim(0, max(v['duration_ms']['max'] for v in entries)*1.2)
        ax.set_xlabel('Median kernel duration (ms)')
        ax.set_title('Sedov: early kernel samples', loc='left', pad=23)
        ax.text(0, 1.035, 'Three filtered launches per kernel · independent profiler processes',
                transform=ax.transAxes, fontsize=10)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#dddddd', linewidth=.5); ax.set_axisbelow(True)
        fig.text(.015, .02, 'Includes FP64 residual correction in mixed kernels. Counts and observed ranges are in kernel_profiles.md.\n'
                 'These samples do not determine full-trajectory phase costs or timing confidence intervals.', fontsize=9)
        fig.tight_layout(rect=(0, .1, 1, 1))
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(output/f'kernel_profiles.{ext}', dpi=240, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    return profile


def supplemental_controls(input_root, output):
    reader_path = REPO/'tests/scripts/compare_vtk_pvd_fields.py'
    spec = importlib.util.spec_from_file_location('supplemental_sedov_reader', reader_path)
    reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)
    def endpoint_pieces(record):
        frame = max(record['frames'], key=lambda f: f['time'])
        return sorted(sha(Path(piece)) for piece in reader.iter_piece_paths(Path(frame['file'])))
    controls = []
    for name, baseline in (('control8-wmma64','full1-wmma64'), ('control8-simt64','full1-simt64')):
        path = input_root/name/'run.json'
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        if not r.get('accepted'):
            continue
        base_path = input_root/baseline/'run.json'; b = json.loads(base_path.read_text())
        analysis_path = path.parent/'analysis.json'; a = json.loads(analysis_path.read_text())
        pieces, base_pieces = endpoint_pieces(r), endpoint_pieces(b)
        controls.append({'name': name, 'raw_seconds': r['monotonic_raw_seconds'],
            'conservation_rtol': r['conservation_rtol'], 'nonlinear_rtol': r['nonlinear_rtol'],
            'steps': int(r['summaries']['[hdg-main]:setup']['step']),
            'rejected_attempts': int(r['summaries']['[hdg-main]:setup']['rejected_attempts']),
            'comparison_to_strict_same_arithmetic': {'name': baseline,
                'endpoint_piece_files_byte_identical': pieces == base_pieces,
                'control_hashes': pieces, 'baseline_hashes': base_pieces},
            'references': a['references'], 'binary_sha256': r['binary_sha256'],
            'sources': {str(q.resolve()): sha(q) for q in (path,base_path,analysis_path)}})
    report = {'scope': 'Supplemental one-trial controls at the mixed-candidate conservation gate; not added to strict-gate timing groups.',
              'controls': controls}
    (output/'supplemental_controls.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    records, sources = [], {}
    for path in sorted(args.input.glob('full*/run.json')):
        if not re.fullmatch(r'full(?:[123]|8[b-c]?)-.+', path.parent.name):
            continue
        run = json.loads(path.read_text())
        run['record_path'] = str(path.resolve())
        sources[str(path.resolve())] = sha(path)
        analysis = path.parent/'analysis.json'
        if analysis.is_file():
            run['analysis_data'] = json.loads(analysis.read_text())
            sources[str(analysis.resolve())] = sha(analysis)
        records.append(run)
    groups, excluded = aggregate(records)
    summary = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'scope': SCOPE,
               'runner_filter': 'full1-*, full2-*, full3-*, full8-*, full8b-*, full8c-*; physical horizon .1 only',
               'script_sha256': sha(Path(__file__)), 'sources': sources, 'groups': groups,
               'excluded': [{k: r.get(k) for k in ('name', 'status', 'mode', 'residual', 'condensation',
                    'horizon', 'conservation_rtol', 'nonlinear_rtol', 'validation_error', 'failure_events', 'record_path')}
                            for r in excluded]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (args.output/'MEASURED_TABLES.md').write_text(markdown(summary))
    if groups:
        make_figure(groups, excluded, args.output)
    profiles = profile_summaries(args.input, args.output)
    supplemental_controls(args.input, args.output)
    print(json.dumps({'output': str(args.output), 'profiles': len(profiles['profiles']), 'groups': len(groups),
                      'completed_runs': sum(g['n'] for g in groups), 'excluded_runs': len(excluded)}))


if __name__ == '__main__':
    main()
