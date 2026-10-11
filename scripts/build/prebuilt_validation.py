#!/usr/bin/env python3
"""Receipts for canonical run-only validation; never builds or runs a solver."""
import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
import subprocess
import tempfile


RECEIPT = 'igrlab-prebuilt-validation-v1.json'
SCHEMA = 'igrlab-prebuilt-validation-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def git(root, *args):
    env = {k: v for k, v in os.environ.items()
           if k not in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE')}
    env['GIT_OPTIONAL_LOCKS'] = '0'
    return subprocess.run(['git', '-C', str(root), *args], env=env,
                          check=True, capture_output=True, timeout=60).stdout


def exact_directory(path):
    path = Path(path)
    require(path.is_absolute() and path.is_dir() and path == path.resolve(strict=True),
            f'missing, moved, or symlinked directory: {path}')
    return path


def file_identity(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and
            path == path.resolve(strict=True), f'missing, moved, or symlinked artifact: {path}')
    return {'sha256': digest(path.read_bytes()), 'mode': path.stat().st_mode & 0o777}


def source_identity(root):
    """Exact tracked and nonignored bytes, including initialized submodules."""
    root = exact_directory(root)
    require(Path(os.fsdecode(git(root, 'rev-parse', '--show-toplevel')).strip()) == root,
            f'source is not a Git checkout root: {root}')
    head = git(root, 'rev-parse', 'HEAD').decode().strip()
    tree = git(root, 'ls-tree', '-r', '-z', 'HEAD').split(b'\0')
    tracked, submodules = {}, {}
    for record in filter(None, tree):
        metadata, name = record.split(b'\t', 1)
        mode, kind, blob = metadata.split()
        name = os.fsdecode(name)
        if kind == b'commit':
            submodule = root / name
            require(not submodule.is_symlink(), f'symlinked submodule: {submodule}')
            initialized = (submodule / '.git').exists()
            require(initialized or not submodule.exists() or
                    (submodule.is_dir() and not any(submodule.iterdir())),
                    f'uninitialized submodule contains unrecorded files: {submodule}')
            submodules[name] = {'gitlink': blob.decode(),
                                'source': source_identity(submodule) if initialized else None}
        else:
            tracked[name] = mode.decode()
    names = set(tracked)
    names.update(os.fsdecode(p) for p in git(root, 'ls-files', '-z', '--cached',
                                            '--others', '--exclude-standard').split(b'\0') if p)
    files = {}
    for name in sorted(names - set(submodules)):
        path = root / name
        require(path.is_relative_to(root) and '..' not in Path(name).parts,
                f'invalid source path: {name}')
        if path.is_symlink():
            require(tracked.get(name) == '120000' and path.resolve(strict=True).is_relative_to(root),
                    f'untracked or escaping source symlink: {path}')
            files[name] = {'link': os.readlink(path)}
        elif path.exists():
            require(path.is_file() and path.resolve(strict=True).is_relative_to(root),
                    f'invalid source file: {path}')
            files[name] = {'sha256': digest(path.read_bytes()),
                           'executable': bool(path.stat().st_mode & 0o111)}
        else:
            files[name] = {'missing': True}
    try:
        release_tag = git(root, 'rev-parse', '--verify', 'refs/tags/v4.9').decode().strip()
    except subprocess.CalledProcessError:
        release_tag = None
    content = {'revision': head, 'files': files, 'submodules': submodules, 'release_tag': release_tag}
    return {'revision': head, 'fingerprint': digest(json.dumps(content, sort_keys=True).encode())}


def artifacts(app_build, app_install, mfem_build, mfem_install, spack_env, cmake, mpi, hypre):
    """Hash runtime bytes and configuration, excluding mutable test-result files."""
    roots = [exact_directory(p) for p in (app_build, app_install, mfem_build, mfem_install, spack_env)]
    app_build, app_install, mfem_build, mfem_install, spack_env = roots
    required = [app_build / 'CMakeCache.txt', app_build / 'CTestTestfile.cmake',
                mfem_build / 'CMakeCache.txt', mfem_build / 'dg-igr-mfem-identity-v1.txt',
                mfem_install / 'lib/libmfem.a',
                mfem_install / 'lib/cmake/mfem/MFEMConfig.cmake',
                mfem_install / 'lib/cmake/mfem/MFEMTargets.cmake',
                spack_env / 'spack.yaml', spack_env / 'spack.lock',
                Path(cmake), Path(cmake).with_name('ctest')]
    selected = set(required)
    executables, links = [], {}
    for root in (app_build, app_install, mfem_install):
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in ('Testing', '_deps', 'CMakeFiles')]
            for name in files:
                path = Path(directory) / name
                if path.is_symlink():
                    resolved = path.resolve(strict=True)
                    require(resolved.is_relative_to(root), f'artifact symlink escapes its root: {path}')
                    require(resolved.is_file(), f'artifact symlink does not resolve to a file: {path}')
                    links[str(path)] = {'link': os.readlink(path), 'target': str(resolved)}
                    selected.add(resolved)
                    continue
                if name.endswith(('.a', '.so', '.dylib')) or '.so.' in name:
                    selected.add(path)
                elif root in (app_build, app_install) and path.stat().st_mode & 0o111:
                    with path.open('rb') as stream:
                        header = stream.read(18)
                    # Executables/shared objects, never relocatable .o/compiler probes.
                    if len(header) == 18 and header[:4] == b'\x7fELF' and header[5] in (1, 2):
                        elf_type = struct.unpack('<H' if header[5] == 1 else '>H', header[16:18])[0]
                        if elf_type in (2, 3):
                            selected.add(path)
                            executables.append(path)
    require(executables, 'prebuilt application contains no ELF executable')
    for root in (app_build, mfem_build):
        for name in ('cmake_install.cmake', 'CMakeFiles/cmake.check_cache'):
            selected.add(root / name)
        for path in (root / 'CMakeFiles').glob('*/*Compiler.cmake'):
            selected.add(path)
    for path in (mfem_install / 'lib/cmake/mfem').glob('MFEMTargets-*.cmake'):
        selected.add(path)
    # Runtime package bytes, not just their containing prefixes, remain pinned.
    for prefix in (exact_directory(mpi), exact_directory(hypre)):
        for library_dir in ('lib', 'lib64'):
            for path in (prefix / library_dir).glob('*'):
                if path.is_file() and ('.so' in path.name or path.suffix in ('.a', '.dylib')):
                    if path.is_symlink():
                        links[str(path)] = {'link': os.readlink(path), 'target': str(path.resolve(strict=True))}
                    selected.add(path.resolve(strict=True))
        for name in ('mpiexec', 'mpirun', 'mpicxx'):
            path = prefix / 'bin' / name
            if path.exists():
                if path.is_symlink():
                    links[str(path)] = {'link': os.readlink(path), 'target': str(path.resolve(strict=True))}
                selected.add(path.resolve(strict=True))
        for name in ('HYPRE.h', 'HYPRE_config.h'):
            path = prefix / 'include' / name
            if path.exists():
                selected.add(path.resolve(strict=True))
    graph_command = [str(Path(cmake).with_name('ctest')), '--test-dir', str(app_build), '--show-only=json-v1']
    graph = json.loads(subprocess.run(graph_command, check=True, capture_output=True,
                                     text=True, timeout=60).stdout)
    require(isinstance(graph.get('tests'), list) and graph['tests'], 'prebuilt CTest graph contains no tests')
    return {'files': {str(path): file_identity(path) for path in sorted(selected)},
            'links': links, 'test_graph': graph}



def collect(args, mfem_paths=None):
    source = source_identity(args.source)
    if args.expected_source:
        require(source == json.loads(args.expected_source), 'source changed during the canonical build')
    paths = {'source': str(exact_directory(args.source)),
             'app_build': str(exact_directory(args.app_build)),
             'app_install': str(exact_directory(args.app_install)),
             'spack_env': str(exact_directory(args.spack_env)),
             'cmake': str(Path(args.cmake).resolve(strict=True)),
             'mpi': str(exact_directory(args.mpi)), 'hypre': str(exact_directory(args.hypre))}
    for name in ('mfem_source', 'mfem_build', 'mfem_install'):
        value = mfem_paths[name] if mfem_paths is not None else getattr(args, name)
        paths[name] = str(exact_directory(value))
    policy = dict(item.split('=', 1) for item in args.policy)
    require(len(policy) == len(args.policy), 'duplicate validation policy key')
    return {'schema': SCHEMA, 'paths': paths, 'source': source, 'policy': policy,
            'mfem_source': source_identity(paths['mfem_source']),
            'artifacts': artifacts(paths['app_build'], paths['app_install'], paths['mfem_build'],
                                   paths['mfem_install'], paths['spack_env'], paths['cmake'],
                                   paths['mpi'], paths['hypre'])}


def write(args):
    receipt = collect(args)
    destination = Path(args.app_build) / RECEIPT
    require(not destination.is_symlink(), f'validation receipt is symlinked: {destination}')
    fd, temporary = tempfile.mkstemp(prefix='.' + RECEIPT, dir=args.app_build)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write('\n')
        os.chmod(temporary, 0o444)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return receipt


def verify(args):
    destination = Path(args.app_build) / RECEIPT
    file_identity(destination)
    saved = json.loads(destination.read_text())
    require(saved.get('schema') == SCHEMA and isinstance(saved.get('paths'), dict),
            'missing or unsupported prebuilt validation receipt; perform a canonical build first')
    actual = collect(args, saved['paths'])
    require(saved == actual, 'prebuilt source, configuration, dependencies, or artifacts changed; rebuild canonically')
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('source', 'write', 'verify'))
    parser.add_argument('--source', type=Path, required=True)
    for name in ('app-build', 'app-install', 'mfem-source', 'mfem-build', 'mfem-install',
                 'spack-env', 'cmake', 'mpi', 'hypre'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--expected-source')
    parser.add_argument('--policy', action='append', default=[])
    args = parser.parse_args()
    try:
        if args.action == 'source':
            result = source_identity(args.source)
        else:
            required = ('app_build', 'app_install', 'spack_env', 'cmake', 'mpi', 'hypre')
            if args.action == 'write':
                required += ('mfem_source', 'mfem_build', 'mfem_install')
            require(all(getattr(args, name) is not None for name in required), 'missing receipt path argument')
            result = write(args) if args.action == 'write' else verify(args)
        if args.action == 'verify':
            result = result['paths']
        elif args.action == 'write':
            result = {'receipt': str(Path(args.app_build) / RECEIPT),
                      'source_fingerprint': result['source']['fingerprint']}
        print(json.dumps(result, sort_keys=True))
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        parser.exit(3, f'ERROR: prebuilt validation: {error}\n')


if __name__ == '__main__':
    main()
