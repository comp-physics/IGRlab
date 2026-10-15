#!/usr/bin/env python3
"""Check the dependency paths CMake and the loader actually selected.

The caller supplies concrete prefixes. This checker neither installs packages
nor executes application code; optional executable checks invoke ldd only.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path


HYPRE_HEADERS = ('HYPRE.h', 'HYPRE_config.h')
HYPRE_LIBRARY = re.compile(r'^libHYPRE\.(?:a|so(?:\..*)?|dylib)$')
MPI_LIBRARY = re.compile(r'^lib(?:mpi[^/]*|mpich[^/]*|open-pal|open-rte)\.so(?:\.|$)')


class BindingError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise BindingError(message)


def existing(value, label):
    require(bool(str(value)), f'{label} is empty')
    path = Path(value)
    require(path.is_absolute(), f'{label} is not an absolute path: {value}')
    return path.resolve(strict=True)


def inside(value, prefix, label):
    path = existing(value, label)
    require(path.is_relative_to(prefix),
            f'{label} escapes selected prefix {prefix}: {value} -> {path}')
    require(path.is_file(), f'{label} is not a file: {path}')
    return str(path)


def cache_values(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        match = re.match(r'([^/#][^:]*):[^=]*=(.*)$', line)
        if match:
            result[match[1]] = match[2]
    return result


def include_bindings(entries, hypre, label, singular=False):
    """Allow transitive GPU includes, checking every dir providing HYPRE headers."""
    found, transitive = [], []
    for entry in entries:
        directory = existing(entry, label)
        require(directory.is_dir(), f'{label} is not a directory: {directory}')
        supplies_hypre = any((directory / name).exists() or
                             (directory / name).is_symlink() for name in HYPRE_HEADERS)
        if supplies_hypre or singular:
            found.append({name: inside(directory / name, hypre, f'{label}/{name}')
                          for name in HYPRE_HEADERS})
        else:
            transitive.append(str(directory))
    require(found, f'{label} contains no selected HYPRE headers')
    return {'hypre_headers': found, 'transitive_include_dirs': transitive}


def link_entry(entry):
    # Installed targets may protect an absolute static archive with LINK_ONLY.
    while entry.startswith('$<LINK_ONLY:') and entry.endswith('>'):
        entry = entry[len('$<LINK_ONLY:'):-1]
    return entry


def library_bindings(entries, hypre, label, required=True):
    found = []
    for item in entries:
        entry = link_entry(item)
        if HYPRE_LIBRARY.fullmatch(Path(entry).name):
            found.append(inside(entry, hypre, label))
        elif entry in ('HYPRE', '-lHYPRE') or entry.startswith('HYPRE::'):
            raise BindingError(f'{label} has an unresolved HYPRE library: {entry}')
    require(not required or found, f'{label} contains no resolved HYPRE library')
    return found


def check_cache(cache, hypre, mpi):
    require(cache.get('HYPRE_DIR'), 'missing HYPRE_DIR in MFEM cache')
    require(existing(cache['HYPRE_DIR'], 'HYPRE_DIR') == hypre,
            'MFEM cached HYPRE_DIR differs from selected concrete prefix')
    result = {}
    for name in ('HYPRE_INCLUDE_DIR', 'HYPRE_INCLUDE_DIRS'):
        require(cache.get(name), f'missing {name} in MFEM cache')
        result[name] = include_bindings(cache[name].split(';'), hypre, name,
                                       singular=(name == 'HYPRE_INCLUDE_DIR'))
    for name in ('HYPRE_LIBRARY', 'HYPRE_LIBRARIES'):
        require(cache.get(name), f'missing {name} in MFEM cache')
        result[name] = library_bindings(cache[name].split(';'), hypre, name)
    expected = existing(mpi / 'bin/mpicxx', 'selected MPI C++ wrapper')
    for name in ('CMAKE_CXX_COMPILER', 'MPI_CXX_COMPILER'):
        require(cache.get(name), f'missing {name} in MFEM cache')
        require(existing(cache[name], name) == expected,
                f'MFEM cached {name} differs from selected MPI C++ wrapper')
    result['mpi_cxx_compiler'] = str(expected)
    return result


def cmake_lists(text, name, assignment=False):
    # Read generated literal properties, without running CMake or evaluating code.
    text = '\n'.join(line for line in text.splitlines()
                     if not line.lstrip().startswith('#'))
    prefix = r'\bset\s*\(\s*' if assignment else r'\b'
    pattern = prefix + re.escape(name) + r'\s+"((?:[^"\\]|\\.)*)"'
    values = re.findall(pattern, text, flags=re.S)
    require(values, f'generated CMake file has no quoted {name} property')
    return [entry.replace('\\"', '"') for value in values for entry in value.split(';')]


def installed_entries(path, entries):
    # Both files live under <prefix>/{lib,lib64}/cmake/mfem.
    prefix = Path(path).resolve(strict=True).parents[3]
    result = []
    for entry in entries:
        for variable in ('${PACKAGE_PREFIX_DIR}', '${_IMPORT_PREFIX}'):
            entry = entry.replace(variable, str(prefix))
        result.append(entry)
    return result


def check_installed(config, targets, hypre):
    result = {}
    if config:
        entries = cmake_lists(Path(config).read_text(), 'MFEM_INCLUDE_DIRS', assignment=True)
        result['config_includes'] = include_bindings(
            installed_entries(config, entries), hypre, 'installed MFEM_INCLUDE_DIRS')
    if targets:
        text = Path(targets).read_text()
        includes = installed_entries(targets, cmake_lists(text, 'INTERFACE_INCLUDE_DIRECTORIES'))
        result['target_includes'] = include_bindings(
            includes, hypre, 'installed MFEM target includes')
        libraries = installed_entries(targets, cmake_lists(text, 'INTERFACE_LINK_LIBRARIES'))
        result['target_hypre_libraries'] = library_bindings(
            libraries, hypre, 'installed MFEM target link libraries')
    return result


def check_loader_output(output, hypre, mpi, returncode=0):
    require(output.strip(), 'ldd produced no dependency information')
    require('not found' not in output, 'loader reports an unresolved dependency')
    static = ('not a dynamic executable' in output or 'statically linked' in output)
    require(returncode == 0 or (returncode == 1 and static),
            f'ldd failed with status {returncode}: {output.strip()}')
    result = {'hypre': [], 'mpi': [], 'static_executable': static}
    for line in output.splitlines():
        match = re.match(r'\s*(\S+)\s+=>\s+(.+?)\s+\(0x[0-9a-fA-F]+\)\s*$', line)
        if not match:
            continue
        name, value = match.groups()
        if name.startswith('libHYPRE.so'):
            result['hypre'].append(inside(value, hypre, 'dynamic HYPRE library'))
        elif MPI_LIBRARY.match(name):
            result['mpi'].append(inside(value, mpi, 'dynamic MPI library'))
    # An unused or statically linked HYPRE/MPI library legitimately has no entry.
    return result


def app_executables(root):
    root = existing(root, 'application build directory')
    require(root.is_dir(), 'application build path is not a directory')
    result = []
    for path in sorted(root.rglob('*')):
        if ('CMakeFiles' in path.relative_to(root).parts or '.so' in path.name or
                not path.is_file() or not path.stat().st_mode & 0o111):
            continue
        with path.open('rb') as stream:
            if stream.read(4) == b'\x7fELF':
                result.append(path)
    require(result, 'no compiled ELF applications found in application build directory')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hypre-prefix', type=Path, required=True)
    parser.add_argument('--mpi-prefix', type=Path, required=True)
    parser.add_argument('--mfem-cache', type=Path)
    parser.add_argument('--mfem-config', type=Path)
    parser.add_argument('--mfem-targets', type=Path)
    parser.add_argument('--app-build', type=Path)
    parser.add_argument('--executable', type=Path, action='append', default=[])
    args = parser.parse_args()
    require(any((args.mfem_cache, args.mfem_config, args.mfem_targets,
                 args.app_build, args.executable)), 'no bindings requested for validation')
    hypre = existing(args.hypre_prefix, 'selected HYPRE prefix')
    mpi = existing(args.mpi_prefix, 'selected MPI prefix')
    require(hypre.is_dir() and mpi.is_dir(), 'selected dependency prefix is not a directory')
    result = {'status': 'DEPENDENCY_BINDINGS_VALIDATED',
              'hypre_prefix': str(hypre), 'mpi_prefix': str(mpi)}
    if args.mfem_cache:
        result['cache'] = check_cache(cache_values(args.mfem_cache), hypre, mpi)
    if args.mfem_config or args.mfem_targets:
        result['installed'] = check_installed(args.mfem_config, args.mfem_targets, hypre)
    executables = list(args.executable)
    if args.app_build:
        executables += app_executables(args.app_build)
    if executables:
        result['executables'] = {}
        for path in sorted(set(executables)):
            binary = existing(path, 'compiled application')
            require(binary.is_file() and binary.stat().st_mode & 0o111,
                    f'compiled application is not an executable file: {binary}')
            with binary.open('rb') as stream:
                require(stream.read(4) == b'\x7fELF',
                        f'compiled application is not ELF: {binary}')
            linkage = subprocess.run(['ldd', str(binary)], capture_output=True,
                                     text=True, timeout=30)
            result['executables'][str(binary)] = check_loader_output(
                linkage.stdout + linkage.stderr, hypre, mpi, linkage.returncode)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except (BindingError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f'ERROR: dependency bindings: {error}')
