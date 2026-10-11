#!/usr/bin/env python3
"""Immutable inputs and canonical build plans for a bounded PACE study.

This module never invokes CMake, CTest, module commands, or a compiler. Build
submission goes through the frozen checkout's public build.sh entry point.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
from typing import Any


MFEM_PATH = 'tpl/mfem'
MFEM_ORIGIN = 'https://github.com/mfem/mfem.git'
REQUIRED_FILES = ('build.sh', 'cmake/sites/pace.json', 'scripts/build/pace.sh')
BUILD_ENV = {'IGRLAB_BUILD_TYPE': 'Release', 'OMP_NUM_THREADS': '1',
             'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


class StudyBuildError(ValueError):
    pass


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise StudyBuildError(message)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(',', ':')).encode()).hexdigest()


def _environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('IGRLAB_', 'DG_IGR_', 'GIT_'))
           and k not in ('CMAKE_BUILD_TYPE', 'MFEM_CUDA_ARCHS', 'APP_CUDA_ARCHS',
                        'CUDA_ARCHS', 'SPACK_CUDA_ARCH', 'NPROC')}
    env.update(extra or {})
    return env


def _run(argv: list[str], *, input: bytes | None = None,
         env: dict[str, str] | None = None, check: bool = True,
         timeout: int = 300) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(argv, input=input, capture_output=True,
                                env=env or _environment(), timeout=timeout)
    except subprocess.TimeoutExpired as error:
        if not check:
            return subprocess.CompletedProcess(argv, 124, error.stdout or b'',
                                               (error.stderr or b'') + b'\ncommand timed out')
        raise StudyBuildError(f'command could not complete: {argv[0]}: {error}') from error
    except OSError as error:
        if not check:
            return subprocess.CompletedProcess(argv, 127, b'', str(error).encode())
        raise StudyBuildError(f'command could not complete: {argv[0]}: {error}') from error
    _require(not check or result.returncode == 0,
             f'command failed ({result.returncode}): {argv[0]}: '
             f'{result.stderr.decode(errors="replace").strip()[:2000]}')
    return result


def _git(root: Path, *args: str, input: bytes | None = None) -> bytes:
    return _run(['git', '-C', str(root), *args], input=input).stdout


def _text(root: Path, *args: str) -> str:
    return os.fsdecode(_git(root, *args)).strip()


def _safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    _require(value and not path.is_absolute() and '..' not in path.parts and
             '.git' not in path.parts, f'unsafe source path: {value!r}')
    return Path(value)


def _no_symlink_ancestors(path: Path) -> None:
    for parent in (path, *path.parents):
        _require(not parent.is_symlink(), f'symlinked source/output component: {parent}')


def _checkout_root(source: Path) -> Path:
    source = source.absolute()
    _no_symlink_ancestors(source)
    source = source.resolve(strict=True)
    _require(Path(_text(source, 'rev-parse', '--show-toplevel')) == source,
             f'source must be a checkout top level: {source}')
    return source


def _tree(root: Path, revision: str) -> dict[str, tuple[str, str]]:
    entries = {}
    for raw in _git(root, 'ls-tree', '-r', '-z', revision).split(b'\0'):
        if not raw:
            continue
        metadata, name = raw.split(b'\t', 1)
        mode, kind, oid = metadata.decode().split()
        value = os.fsdecode(name)
        _safe_relative(value)
        _require(kind in ('blob', 'commit'), f'unsupported Git tree entry: {value}')
        entries[value] = (mode, oid)
    return entries


def _record(root: Path, relative: str, expected: tuple[str, str] | None) -> dict[str, Any]:
    path = root / _safe_relative(relative)
    _no_symlink_ancestors(path.parent)
    if not path.exists() and not path.is_symlink():
        return {'kind': 'missing'}
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        _require(expected is not None and expected[0] == '120000',
                 f'new/untracked symlink is unsupported: {relative}')
        target = os.readlink(path)
        committed = os.fsdecode(_git(root, 'cat-file', 'blob', expected[1]))
        _require(target == committed, f'modified tracked symlink is unsupported: {relative}')
        _require(not Path(target).is_absolute() and
                 (path.parent / target).resolve().is_relative_to(root),
                 f'tracked symlink escapes snapshot: {relative}')
        return {'kind': 'symlink', 'target': target,
                'sha256': hashlib.sha256(os.fsencode(target)).hexdigest()}
    _require(stat.S_ISREG(mode), f'nonregular source file is unsupported: {relative}')
    return {'kind': 'file', 'executable': bool(mode & 0o111),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _inventory(root: Path, *, allow_mfem: bool = True) -> dict[str, Any]:
    revision = _text(root, 'rev-parse', 'HEAD')
    entries = _tree(root, 'HEAD')
    tracked = {}
    for name, entry in entries.items():
        if entry[0] == '160000':
            _require(allow_mfem and name == MFEM_PATH,
                     f'unsupported nested submodule: {name}')
            tracked[name] = {'kind': 'submodule', 'revision': entry[1]}
        else:
            tracked[name] = _record(root, name, entry)
    untracked = {}
    for raw in _git(root, 'ls-files', '--others', '--exclude-standard', '-z').split(b'\0'):
        if raw:
            name = os.fsdecode(raw)
            untracked[name] = _record(root, name, None)
    # Files added only to the index belong to literal working-tree content too.
    # Classify these relative to the resolved HEAD tree, consistently before
    # and after applying the complete HEAD diff to the detached snapshot.
    for raw in _git(root, 'ls-files', '-z').split(b'\0'):
        if raw:
            name = os.fsdecode(raw)
            if name not in entries:
                untracked[name] = _record(root, name, None)
    return {'revision': revision, 'tree': _text(root, 'rev-parse', 'HEAD^{tree}'),
            'tracked': tracked, 'untracked': untracked}


def _standalone(root: Path) -> None:
    value = Path(_text(root, 'rev-parse', '--git-path', 'objects/info/alternates'))
    path = value if value.is_absolute() else root / value
    _require(not path.exists() and not path.is_symlink(),
             f'snapshot has borrowed Git objects: {root}')


def _origin(root: Path) -> str | None:
    names = _text(root, 'remote').splitlines()
    return _text(root, 'remote', 'get-url', 'origin') if 'origin' in names else None


def _clone(seed: Path, dest: Path, revision: str, origin: str | None) -> None:
    _run(['git', 'clone', '--no-hardlinks', '--no-checkout', '--', str(seed), str(dest)])
    _git(dest, 'checkout', '--detach', revision)
    if origin:
        _git(dest, 'remote', 'set-url', 'origin', origin)
    else:
        _git(dest, 'remote', 'remove', 'origin')
    _standalone(dest)


def _mfem_seed(source: Path, revision: str, *, working: bool) -> tuple[Path, str]:
    entries = _tree(source, revision)
    _require(MFEM_PATH in entries and entries[MFEM_PATH][0] == '160000',
             'source must track the pinned tpl/mfem submodule')
    pin = entries[MFEM_PATH][1]
    seed = _checkout_root(source / MFEM_PATH)
    _require(not _git(seed, 'status', '--porcelain', '--untracked-files=all'),
             'MFEM seed/submodule is dirty; freeze requires a clean local seed')
    _require(_origin(seed) == MFEM_ORIGIN, 'MFEM seed origin is not the upstream MFEM repository')
    _require(_text(seed, 'cat-file', '-t', 'refs/tags/v4.9') == 'tag' and
             _text(seed, 'rev-parse', 'refs/tags/v4.9^{commit}') == pin,
             'MFEM gitlink requires the available annotated upstream v4.9 tag')
    if working:
        _require(_text(seed, 'rev-parse', 'HEAD') == pin,
                 'working-tree MFEM submodule revision differs from its gitlink')
        patch = _git(source, 'diff', '--binary', 'HEAD', '--', MFEM_PATH)
        _require(not patch, 'working-tree submodule changes are unsupported')
    return seed, pin


def freeze(source: Path, ref: str, dest: Path) -> dict[str, Any]:
    """Clone one immutable source, preserving literal working-tree bytes."""
    source = _checkout_root(source)
    _require(bool(ref), 'a source ref is required')
    working = ref == 'working-tree'
    revision = _text(source, 'rev-parse', '--verify', '--end-of-options',
                     ('HEAD' if working else ref) + '^{commit}')
    _require(re.fullmatch(r'[0-9a-f]{40,64}', revision), 'malformed source revision')
    entries = _tree(source, revision)
    for name in REQUIRED_FILES:
        _require(name in entries and entries[name][0] in ('100644', '100755'),
                 f'unsupported historical source: missing tracked canonical {name}; no build-driver overlay is applied')
    _require(entries['build.sh'][0] == '100755', 'canonical build.sh must be tracked executable')
    seed, mfem_pin = _mfem_seed(source, revision, working=working)
    original = _inventory(source) if working else None
    patch = _git(source, 'diff', '--binary', '--no-ext-diff', '--no-textconv', 'HEAD') if working else b''
    dest = dest.absolute()
    _no_symlink_ancestors(dest)
    dest = dest.resolve()
    _require(not dest.exists(), f'snapshot destination already exists: {dest}')
    _require(not dest.is_relative_to(source), 'snapshot destination must be outside its source checkout')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.mkdir()  # Exclusive ownership; only this newly created directory is removed on failure.
    try:
        _clone(source, dest, revision, _origin(source))
        if working and patch:
            _git(dest, 'apply', '--binary', '-', input=patch)
        if working:
            # Copy literal bytes too: Git clean filters/index flags can otherwise
            # hide differences that a binary diff alone does not preserve.
            for name, record in {**original['tracked'], **original['untracked']}.items():
                target = dest / _safe_relative(name)
                if record['kind'] == 'file':
                    _no_symlink_ancestors(target.parent)
                    _require(not target.is_symlink(), f'file would follow a snapshot symlink: {name}')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / name, target)
                elif record['kind'] == 'missing' and target.exists():
                    _require(target.is_file() and not target.is_symlink(), f'unsafe deleted file: {name}')
                    target.unlink()
        _clone(seed, dest / MFEM_PATH, mfem_pin, MFEM_ORIGIN)
        if working:
            _require(_inventory(source) == original, 'original working-tree bytes changed while freezing')
            _require(_inventory(dest) == original, 'snapshot does not preserve exact working-tree contents')
        identity = _identity(dest)
        identity.update(source_ref=ref, original_source=str(source))
        verify_source(identity)
        return identity
    except Exception:
        shutil.rmtree(dest)
        raise


def _identity(snapshot: Path) -> dict[str, Any]:
    inventory = _inventory(snapshot)
    mfem = snapshot / MFEM_PATH
    mfem_inventory = _inventory(mfem, allow_mfem=False)
    _require(not _git(mfem, 'status', '--porcelain', '--untracked-files=all'),
             'snapshot MFEM submodule is dirty')
    mfem_pin = inventory['tracked'][MFEM_PATH]['revision']
    _require(mfem_inventory['revision'] == mfem_pin and
             _text(mfem, 'cat-file', '-t', 'refs/tags/v4.9') == 'tag' and
             _text(mfem, 'rev-parse', 'refs/tags/v4.9^{commit}') == mfem_pin,
             'snapshot MFEM revision/release tag changed')
    _standalone(snapshot)
    _standalone(mfem)
    tag_object = _text(mfem, 'rev-parse', 'refs/tags/v4.9')
    content = {'source': inventory, 'mfem': mfem_inventory,
               'mfem_tag_object': tag_object}
    return {'schema_version': 1, 'source_snapshot': str(snapshot),
            'source_revision': inventory['revision'], 'source_tree': inventory['tree'],
            'source_status': 'dirty' if _git(snapshot, 'status', '--porcelain', '--untracked-files=all') else 'clean',
            'tracked_contents_sha256': _json_hash({'source': inventory['tracked'], 'mfem': mfem_inventory['tracked']}),
            'contents_sha256': _json_hash(content), 'source_fingerprint': _json_hash(content),
            'source_origin': _origin(snapshot), 'mfem_revision': mfem_pin,
            'mfem_origin': _origin(mfem), 'mfem_tag': 'v4.9',
            'mfem_tag_object': tag_object}


def verify_source(identity: dict[str, Any]) -> None:
    """Fail before build/runtime if any frozen source bytes or Git identity drift."""
    snapshot = _checkout_root(Path(identity['source_snapshot']))
    actual = _identity(snapshot)
    for key, value in actual.items():
        _require(identity.get(key) == value, f'frozen source identity changed: {key}')
    _require(actual['mfem_origin'] == MFEM_ORIGIN, 'snapshot MFEM origin changed')


def _validation_seconds(value: str) -> int:
    _require(isinstance(value, str) and re.fullmatch(r'[0-9]{2}:[0-5][0-9]:00', value),
             'validation_time requires minute-aligned HH:MM:00')
    hours, minutes, _ = map(int, value.split(':'))
    seconds = 3600 * hours + 60 * minutes
    _require(0 < seconds <= 7200, 'validation_time must be positive and no greater than 02:00:00')
    return seconds


def build_plan(identity: dict[str, Any], output_root: Path, site_root: Path,
               spack_root: Path, correctness_regex: str, *,
               validation_time: str | None = None) -> dict[str, Any]:
    """Inspect only the canonical --print-plan; never build on the login node."""
    verify_source(identity)
    _require(bool(correctness_regex), 'a nonempty focused correctness regex is required')
    roots = [Path(p).absolute() for p in (output_root, site_root, spack_root)]
    for root in roots:
        _no_symlink_ancestors(root)
    roots = [p.resolve() for p in roots]
    requested_output, site_root, spack_root = roots
    snapshot = Path(identity['source_snapshot'])
    validation_mode, capability = None, None
    if validation_time is not None:
        _validation_seconds(validation_time)
        help_result = _run([str(snapshot / 'build.sh'), '--site', 'pace', '--help'], env=_environment(BUILD_ENV))
        help_text = help_result.stdout.decode(errors='replace')
        validation_mode = 'canonical' if re.search(r'(?m)^\s*--validation-time(?:\s|$)', help_text) else 'scheduler-update'
        capability = {'argv': help_result.args, 'returncode': help_result.returncode,
                      'stdout_sha256': hashlib.sha256(help_result.stdout).hexdigest()}
    flags = ['--site', 'pace', '--a100', '--hdg-runtime', '--nvtx-profile',
             '--no-clean', '--no-research-tests', '--submit', '-j', '4',
             '--mfem-jobs', '4', '--test-regex', correctness_regex]
    config = {'source_fingerprint': identity['source_fingerprint'], 'flags': flags,
              'site_config_sha256': hashlib.sha256((snapshot / 'cmake/sites/pace.json').read_bytes()).hexdigest(),
              'environment': BUILD_ENV}
    config_key = _json_hash(config)
    output_root = requested_output / ('config-' + config_key[:20])
    argv = [str(snapshot / 'build.sh'), *flags, '--output-root', str(output_root),
            '--site-root', str(site_root), '--spack-root', str(spack_root)]
    # Scheduling limits do not change compiled artifacts or the cache key.
    if validation_mode == 'canonical':
        argv += ['--validation-time', validation_time]
    result = _run([*argv, '--print-plan'], env=_environment(BUILD_ENV))
    stdout, stderr = result.stdout.decode(errors='replace'), result.stderr.decode(errors='replace')
    wt = 'wt-' + hashlib.sha256(os.fsencode(snapshot)).hexdigest()[:16]
    scope = output_root / 'worktrees' / wt / 'cuda-sm80'
    app_build, app_install = scope / 'app/build', scope / 'app/install'
    mfem_build, mfem_install = scope / 'mfem/build', scope / 'mfem/install'
    env_path = site_root / 'site/pace/spack-envs/cuda-sm80'
    # Match complete known records instead of splitting paths/regexes on spaces.
    _require(f'[build-plan] output_root={output_root} worktree_id={wt} scope={scope}' in stdout and
             f'[build-plan] env={env_path} app_build={app_build} app_install={app_install}' in stdout,
             'canonical build plan did not admit the expected source/configuration paths')
    verify_source(identity)
    return {'schema_version': 1, 'source': identity, 'configuration_key': config_key,
            'configuration': config, 'requested_output_root': str(requested_output),
            'output_root': str(output_root), 'site_root': str(site_root), 'spack_root': str(spack_root),
            'argv': argv, 'env': dict(BUILD_ENV), 'spack_env': str(env_path),
            'app_build': str(app_build), 'app_install': str(app_install),
            'mfem_build': str(mfem_build), 'mfem_install': str(mfem_install),
            'mfem_identity': str(mfem_build / 'dg-igr-mfem-identity-v1.txt'),
            'mfem_config': str(mfem_install / 'lib/cmake/mfem/MFEMConfig.cmake'),
            'mfem_targets': str(mfem_install / 'lib/cmake/mfem/MFEMTargets.cmake'),
            'validation_time': validation_time, 'validation_time_mode': validation_mode,
            'validation_time_capability': capability,
            'plan_stdout': stdout, 'plan_stderr': stderr}


def ensure_validation_time(plan: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    """Admit a recorded validation job and shorten only a pending owned job."""
    if plan.get('validation_time') is None:
        return dict(receipt)
    result = dict(receipt)
    canonical_success = result.get('canonical_submission_success', result.get('success', False))
    result['canonical_submission_success'] = canonical_success
    metadata = {'requested': plan['validation_time'], 'mode': plan.get('validation_time_mode'),
                'commands': [], 'status': 'withheld'}
    result['validation_time'] = metadata

    def command(argv):
        completed = _run(argv, check=False, timeout=30)
        observation = {'argv': argv, 'returncode': completed.returncode,
                       'stdout': completed.stdout.decode(errors='replace'),
                       'stderr': completed.stderr.decode(errors='replace')}
        metadata['commands'].append(observation)
        _require(completed.returncode == 0, f'validation limit command failed ({completed.returncode}): {observation["stderr"]}')
        return observation['stdout']

    def inspect(job):
        text = command(['scontrol', 'show', 'job', '-o', job])
        values = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)', text))
        executable = re.search(r'(?:^|\s)Command=(.*?)(?=\s+[A-Z][A-Za-z0-9_]*=|$)', text)
        uid = re.search(r'\(([0-9]+)\)$', values.get('UserId', ''))
        _require(values.get('JobId') == job and values.get('JobName') == 'IGRlab-check' and
                 values.get('Account') == 'gts-sbryngelson3' and values.get('QOS') == 'embers' and
                 uid is not None and int(uid[1]) == os.getuid() and executable is not None and
                 executable[1] == str(Path(plan['source']['source_snapshot']) / 'scripts/build/pace.sh'),
                 'validation limit withheld: scheduler job identity changed')
        return values

    def limit_seconds(value):
        limit = re.fullmatch(r'(?:(\d+)-)?(\d+):([0-5][0-9]):([0-5][0-9])', value or '')
        _require(limit is not None, 'unknown validation job time limit')
        seconds = int(limit[1] or 0) * 86400 + int(limit[2]) * 3600 + int(limit[3]) * 60 + int(limit[4])
        _require(seconds > 0, 'validation job time limit must be positive')
        return seconds

    try:
        wanted = _validation_seconds(plan['validation_time'])
        _require(metadata['mode'] in ('canonical', 'scheduler-update'), 'invalid validation limit mode')
        _require(canonical_success, 'canonical build submission was incomplete')
        for key, expected in (('source_fingerprint', plan['source']['source_fingerprint']),
                              ('configuration_key', plan['configuration_key'])):
            _require(result.get(key) == expected, f'validation receipt identity changed: {key}')
        verify_source(plan['source'])
        job = result.get('validation_job_id')
        _require(isinstance(job, str) and re.fullmatch(r'[0-9]+', job), 'missing/invalid validation job ID')
        values = inspect(job)
        metadata['observed_state'] = values.get('JobState')
        metadata['actual'] = values.get('TimeLimit')
        if values.get('JobState') == 'COMPLETED':
            metadata['status'] = 'completed-noop'
        else:
            actual = limit_seconds(values.get('TimeLimit'))
            _require(values.get('JobState') in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'),
                     'validation job is not pending, active, or completed')
            if actual <= wanted:
                metadata['status'] = 'within-limit-noop'
            else:
                _require(values.get('JobState') == 'PENDING', 'running validation job exceeds requested limit; update withheld')
                metadata['update_attempted'] = True
                command(['scontrol', 'update', 'JobId=' + job, 'TimeLimit=' + plan['validation_time']])
                readback = inspect(job)
                metadata['actual'] = readback.get('TimeLimit')
                metadata['readback_state'] = readback.get('JobState')
                _require(limit_seconds(metadata['actual']) <= wanted, 'validation limit readback exceeds requested limit')
                metadata['status'] = 'updated'
        result['success'] = True
        result.pop('error', None)
    except (StudyBuildError, OSError, KeyError, TypeError, ValueError) as error:
        result['success'] = False
        result['error'] = str(error)
        metadata['status'] = 'failed'
        metadata['error'] = str(error)
    return result


def submit_build(plan: dict[str, Any]) -> dict[str, Any]:
    """Submit only through build.sh and retain the public driver's receipts."""
    verify_source(plan['source'])
    expected = build_plan(plan['source'], Path(plan['requested_output_root']),
                          Path(plan['site_root']), Path(plan['spack_root']),
                          plan['configuration']['flags'][plan['configuration']['flags'].index('--test-regex') + 1],
                          validation_time=plan.get('validation_time'))
    for key in ('argv', 'env', 'configuration_key', 'app_build', 'spack_env', 'validation_time', 'validation_time_mode'):
        _require(plan.get(key) == expected[key], f'build submission plan changed: {key}')
    result = _run(plan['argv'], env=_environment(plan['env']), check=False)
    stdout, stderr = result.stdout.decode(errors='replace'), result.stderr.decode(errors='replace')
    build = re.findall(r'^PACE build submitted: ([0-9]+) \(', stdout, re.M)
    validation = re.findall(r'^PACE validation submitted: ([0-9]+) \(', stdout, re.M)
    receipt = {'schema_version': 1, 'success': result.returncode == 0 and len(build) == len(validation) == 1,
            'returncode': result.returncode,
            'build_job_id': build[0] if len(build) == 1 else None,
            'validation_job_id': validation[0] if len(validation) == 1 else None,
            'build_job_ids': build, 'validation_job_ids': validation,
            'stdout': stdout, 'stderr': stderr, 'argv': plan['argv'],
            'source_fingerprint': plan['source']['source_fingerprint'],
            'configuration_key': plan['configuration_key']}
    try:
        _require(len(build) <= 1 and len(validation) <= 1, 'public submission returned ambiguous job IDs')
        verify_source(plan['source'])
    except (StudyBuildError, OSError, KeyError, TypeError, ValueError) as error:
        receipt.update(success=False, error=str(error))
    return ensure_validation_time(plan, receipt)
