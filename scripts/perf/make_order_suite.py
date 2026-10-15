#!/usr/bin/env python3
"""Write a bounded, site-independent current-architecture order suite to stdout.

This prepares existing application cases only; it never builds or submits jobs.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path


ORDERS = (1, 3, 5, 7)
BACKENDS = ('dg', 'dgsem', 'hdg-ordinary', 'hdg-tensor')
HDG_TEST_ORDERS = (1, 2, 3, 4, 5, 7, 10)
TARGET_CONSERVED_DOFS = 262144


def make_suite(orders=ORDERS, backends=BACKENDS, solver_policy='qualified-fp64'):
    if (not orders or any(type(order) is not int or not 1 <= order <= 2**31-1 for order in orders) or
            len(set(orders)) != len(orders) or
            not backends or len(set(backends)) != len(backends) or not set(backends) <= set(BACKENDS)):
        raise ValueError('choose unique positive native integer orders and supported backends')
    if len(orders) * len(backends) > 16:
        raise ValueError('select at most 16 backend/order cases per suite')
    if any(backend.startswith('hdg-') for backend in backends) and not set(orders) <= set(HDG_TEST_ORDERS):
        raise ValueError('HDG qualification suites require registered test orders: 1,2,3,4,5,7,10')
    if solver_policy not in ('reference', 'qualified-fp64'):
        raise ValueError('unknown solver policy')
    templates = json.loads((Path(__file__).parent / 'suites/delta-p5-dof-scaling.json').read_text())
    cases = []
    for backend in backends:
        method, engine = backend.split('-') if backend.startswith('hdg-') else (backend, 'ordinary')
        template = next(case for case in templates['cases'] if
                        case['parameters']['discretization'] == method and
                        case['parameters']['engine'] == engine)
        for order in orders:
            case = deepcopy(template)
            refine = min(range(3), key=lambda value:
                         abs(4 * (32 * 2**value)**2 * (order+1)**2 - TARGET_CONSERVED_DOFS))
            n = 32 * 2**refine
            case.update(id=f'{backend}-igr-cuda-p{order}-n{n}', enabled=True, optional=False,
                        mesh_n=n, elements=n*n, tier='multiorder', timeout_seconds=180,
                        expected_conserved_dofs=4*n*n*(order+1)**2)
            case['parameters'].update(order=order, serial_refine=refine, final_time=1e-4,
                                      solver_policy=solver_policy, timing_scope='process-wall')
            case['profile']['timeout'] = 180
            cases.append(case)
    order_regex = '(' + '|'.join(map(str, orders)) + ')'
    checks = []
    if 'dg' in backends:
        checks.append('dg_igr_dg_operator_tests')
    if 'dgsem' in backends:
        checks.append('dg_igr_dgsem_operator_cuda')
    if any(backend.startswith('hdg-') for backend in backends):
        checks.append('dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_p' + order_regex + '_rank1')
        for order in orders:
            for engine in ('ordinary', 'tensor'):
                if 'hdg-' + engine not in backends:
                    continue
                name = ('p' + str(order) + '_' + engine if order > 3 else
                        ('tensor_' if engine == 'tensor' else '') + 'p' + str(order))
                checks.append('dg_igr_hdg_runtime_cuda_generic_euler_extension_' + name + '_rank1')
    return {'schema_version': 1, 'name': 'current-architecture-orders',
            'repetitions': 3, 'warmups': 1, 'correctness_regex': '^(' + '|'.join(checks) + ')$',
            'command_timeout_seconds': 180, 'timings_budget_seconds': 3480,
            'profiles_budget_seconds': 3480, 'cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', nargs='+', type=int, default=ORDERS,
                        help='positive native orders (default: 1 3 5 7); HDG requires registered qualification tests')
    parser.add_argument('--backends', nargs='+', choices=BACKENDS, default=BACKENDS)
    parser.add_argument('--solver-policy', choices=('reference', 'qualified-fp64'), default='qualified-fp64')
    args = parser.parse_args()
    try:
        suite = make_suite(args.orders, args.backends, args.solver_policy)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(suite, indent=2))


if __name__ == '__main__':
    main()
