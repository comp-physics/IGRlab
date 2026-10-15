#!/usr/bin/env python3
"""Small site policy selectors; scientific and native-artifact gates stay shared."""
SITES = {
    'pace': {'title': 'PACE', 'cpu_account': 'gts-sbryngelson3',
             'gpu_account': 'gts-sbryngelson3', 'cpu_partition': 'cpu-small',
             'gpu_partition': 'gpu-a100', 'qos': 'embers', 'gpu_gres': 'gpu:a100:1',
             'gpu_tres_types': ('gres/gpu:a100',), 'authority': 'live PACE allocation'},
    'delta': {'title': 'Delta', 'cpu_account': 'bdiy-delta-cpu',
              'gpu_account': 'bdiy-delta-gpu', 'cpu_partition': 'cpu',
              'gpu_partition': 'gpuA100x4', 'qos': None, 'gpu_gres': '1',
              'profiles_constraint': 'perf,nvperf',
              'gpu_tres_types': ('gres/gpu:nvidia_a100', 'gres/gpu:a100'),
              'authority': 'live Delta allocation'},
}


def policy(site='pace'):
    if site not in SITES:
        raise ValueError('unsupported study site: ' + str(site))
    return SITES[site]


def name(plan):
    site = plan.get('site', 'pace')
    policy(site)
    return site


def scheduler_fields(site='pace', profile='cuda-sm80'):
    selected = policy(site)
    if profile not in ('cpu', 'cuda-sm80'):
        raise ValueError('unsupported study profile')
    lane = 'cpu' if profile == 'cpu' else 'gpu'
    fields = {'Account': selected[lane + '_account'], 'Partition': selected[lane + '_partition']}
    if selected['qos'] is not None:
        fields['QOS'] = selected['qos']
    return fields


def environment(site='pace', profile='cuda-sm80'):
    selected = scheduler_fields(site, profile)
    return {'SLURM_JOB_' + key.upper(): value for key, value in selected.items()} | {'SLURM_NTASKS': '1'}


def scheduler_matches(fields, site='pace', profile='cuda-sm80'):
    return all(fields.get(key) == value for key, value in scheduler_fields(site, profile).items())


def accounting_matches(row, site='pace', profile='cuda-sm80'):
    """sacct row uses account/QOS columns4/5; partition is bound elsewhere."""
    selected = policy(site)
    lane = 'cpu' if profile == 'cpu' else 'gpu'
    return len(row) > 5 and row[4] == selected[lane + '_account'] and (
        selected['qos'] is None or row[5] == selected['qos'])


def runtime(site='pace'):
    policy(site)
    return site + '_runtime.sh'


def max_wall_hours(site='pace'):
    policy(site)
    return 1 if site == 'pace' else None
