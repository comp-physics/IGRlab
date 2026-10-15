#!/usr/bin/env python3
"""Internal build.sh app-only route: use configured CMake, never prepare dependencies."""
import argparse
import os
from pathlib import Path
import subprocess


def cache_values(path):
    values = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith(('#', '//')) and '=' in line and ':' in line.split('=', 1)[0]:
            key, value = line.split('=', 1)
            values[key.split(':', 1)[0]] = value
    return values


def run(args):
    cache = args.app_build / 'CMakeCache.txt'
    if not cache.is_file():
        raise ValueError('no configured application build; run the normal build once first')
    values = cache_values(cache)
    if values.get('CMAKE_HOME_DIRECTORY') != str(args.source):
        raise ValueError('existing application build belongs to a different checkout')
    if values.get('CMAKE_CACHEFILE_DIR') != str(args.app_build):
        raise ValueError('application build directory was moved')
    architecture = values.get('CMAKE_CUDA_ARCHITECTURES', '')
    if args.profile.startswith('cuda-sm'):
        if architecture != args.profile.removeprefix('cuda-sm'):
            raise ValueError('existing application CUDA architecture differs from the requested profile')
    elif architecture:
        raise ValueError('CPU iteration cannot use an existing CUDA application build')
    cmake = Path(values.get('CMAKE_COMMAND', ''))
    if not cmake.is_absolute() or not os.access(cmake, os.X_OK):
        raise ValueError('configured CMake executable is unavailable')

    # Restore paths from the configured build, without querying Spack or checking
    # dependency contents. The caller's site wrapper loads the compiler modules.
    env = os.environ.copy()
    view = args.spack_env / '.spack-env/view'
    if view.is_dir():
        env['PATH'] = str(view / 'bin') + os.pathsep + env.get('PATH', '')
        env['LD_LIBRARY_PATH'] = ':'.join((str(view / 'lib'), str(view / 'lib64'),
                                         env.get('LD_LIBRARY_PATH', '')))
    mpi_exe = Path(values.get('MPIEXEC_EXECUTABLE', ''))
    if mpi_exe.is_absolute():
        mpi = mpi_exe.parent.parent
        env['LD_LIBRARY_PATH'] = ':'.join((str(mpi / 'lib'), str(mpi / 'lib64'),
                                         env.get('LD_LIBRARY_PATH', '')))
        native = mpi / 'bin/.bin'
        if (native / 'orted').is_file() and (native / 'ompi_info').is_file():
            env['OPAL_BINDIR'] = str(native)

    if args.tests_only:
        if not (args.test_regex or args.test_label) or args.target:
            raise ValueError('iteration tests require an explicit filter and no build targets')
        ctest = str(cmake.with_name('ctest'))
        command = [ctest, '--test-dir', str(args.app_build), '--output-on-failure', '--no-tests=error']
        if args.test_regex:
            command += ['-R', args.test_regex]
        if args.test_label:
            command += ['-L', args.test_label]
    else:
        # A partial or failed rebuild must never retain a whole-build qualification.
        (args.app_build / 'igrlab-prebuilt-validation-v1.json').unlink(missing_ok=True)
        command = [str(cmake), '--build', str(args.app_build), '-j', str(args.jobs)]
        if args.target:
            command += ['--target', *args.target]
    print('[iterate] existing application configuration; dependencies untouched; no qualification receipt', flush=True)
    return subprocess.run(command, env=env).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'app-build', 'spack-env'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--jobs', type=int, required=True)
    parser.add_argument('--target', action='append', default=[])
    parser.add_argument('--tests-only', action='store_true')
    parser.add_argument('--test-regex')
    parser.add_argument('--test-label')
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
