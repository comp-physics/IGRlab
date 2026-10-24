#!/usr/bin/env python3
"""Run one command in private DeltaAI job scratch; retain only selected files."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

GIB = 1024 ** 3
TMP_ROOT, PROJECTS_ROOT = Path('/tmp'), Path('/projects')
OUTPUT_FLAGS = {'--output', '--output-root', '--run-dir', '--out-dir'}
XML_SUFFIXES = {'.xml', '.vtu', '.pvtu', '.pvd', '.vtp', '.vti', '.vtr', '.vts'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def contained(path, root):
    return path.resolve().is_relative_to(root.resolve())


def project_id(text, parent):
    lines = [line.split(maxsplit=2) for line in text.splitlines() if line.strip()]
    require(len(lines) == 1 and len(lines[0]) == 3 and lines[0][0].isdecimal() and
            lines[0][1] == 'P' and lines[0][2] == str(parent), 'Missing or ambiguous inherited Lustre project ID')
    return lines[0][0]


def check_quota(text, cap, needed_files):
    rows, lines = [], text.splitlines()
    for index, line in enumerate(lines):
        fields = line.split()
        if fields and fields[0] == '/projects':
            if len(fields) == 1 and index + 1 < len(lines):
                fields += lines[index + 1].split()
            rows.append(fields)
    require(len(rows) == 1 and len(rows[0]) == 9, 'Ambiguous Lustre project quota')
    values = [rows[0][i].rstrip('*') for i in (1, 2, 3, 5, 6, 7)]
    require(all(value.isdecimal() for value in values), 'Non-numeric Lustre project quota')
    used, soft, hard, files, file_soft, file_hard = map(int, values)
    byte_limits = [value * 1024 for value in (soft, hard) if value > 0]
    file_limits = [value for value in (file_soft, file_hard) if value > 0]
    require(not byte_limits or used * 1024 + cap <= min(byte_limits), 'Project quota cannot fit retention cap')
    require(not file_limits or files + needed_files < min(file_limits), 'Project inode quota cannot fit retained files')
    return dict(used_bytes=used * 1024, byte_limit=min(byte_limits) if byte_limits else None,
                used_inodes=files, inode_limit=min(file_limits) if file_limits else None,
                required_headroom_bytes=cap, required_headroom_inodes=needed_files)


def project_quota(parent, cap, keep):
    try:
        def query(argv):
            return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=10)
        pid = project_id(query(['lfs', 'project', '-d', str(parent)]), parent)
        # Include logs, receipt, staging directory, and conservatively count every selected parent directory.
        needed = 4 + sum(1 + len(Path(name).parents) for name in set(keep))
        result = check_quota(query(['lfs', 'quota', '-p', pid, '/projects']), cap, needed)
        return dict(result, project_id=pid)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError('Lustre project quota query failed: ' + str(error)) from error


def check_outputs(argv, run, scratch):
    for index, argument in enumerate(argv):
        flag, equals, value = argument.partition('=')
        if flag in OUTPUT_FLAGS:
            if not equals:
                require(index + 1 < len(argv), 'Missing output path: ' + flag)
                value = argv[index + 1]
            require(bool(value), 'Empty output path: ' + flag)
            path = Path(value)
            require(contained(path if path.is_absolute() else run / path, scratch),
                    'Output path must stay in job scratch: ' + value)


def check_xml(path):
    if path.suffix.lower() not in XML_SUFFIXES:
        return
    with path.open('rb') as stream:
        head = stream.read(16384)
        stream.seek(max(0, path.stat().st_size - 4096))
        tail = stream.read().rstrip()
    root = re.search(rb'<([A-Za-z_][\w:.-]*)\b', head)
    require(root is not None and tail.endswith(b'</' + root[1] + b'>'),
            'Incomplete XML footer: ' + str(path))
    if path.suffix.lower() != '.xml':
        require(root[1] == b'VTKFile', 'Expected VTK XML: ' + str(path))


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def retain(scratch, run, destination, keep, cap, record):
    """Publish checked copies atomically; invalid selections retain logs only."""
    logs = [(scratch / name, Path(name)) for name in ('stdout.log', 'stderr.log')]
    chosen = list(logs)
    try:
        for name in dict.fromkeys(keep):
            relative = Path(name)
            require(not relative.is_absolute() and '..' not in relative.parts and
                    not any(c in name for c in '*?[]'), 'Keep paths must be exact relative files')
            source = run / relative
            require(contained(source, run) and not source.is_symlink() and source.is_file(),
                    'Missing, unsafe or non-file selection: ' + name)
            check_xml(source)
            chosen.append((source, Path('selected') / relative))
        require(sum(source.stat().st_size for source, _ in chosen) <= cap,
                'Selected files exceed retention cap')
    except (OSError, ValueError) as error:
        record['storage_error'] = str(error)
        chosen = logs
    while True:
        if record.get('storage_error'):
            record['wrapper_returncode'] = 2
        record['files'] = [dict(path=str(target), bytes=source.stat().st_size,
                                sha256=digest(source)) for source, target in chosen]
        payload = (json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n').encode()
        if sum(item['bytes'] for item in record['files']) + len(payload) <= cap:
            break
        require(bool(chosen), 'Receipt alone exceeds retention cap')
        record['storage_error'] = 'Files plus receipt exceed retention cap; files omitted'
        chosen = logs if chosen != logs else []
    require(not os.path.lexists(destination), 'Results directory must be new')
    staging = Path(tempfile.mkdtemp(prefix='.' + destination.name + '-', dir=destination.parent))
    try:
        for (source, target), item in zip(chosen, record['files']):
            copied = staging / target
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, copied)
            require(copied.stat().st_size == item['bytes'] and digest(copied) == item['sha256'],
                    'Retained copy failed size/hash verification: ' + str(target))
        (staging / 'run.json').write_bytes(payload)
        require(not os.path.lexists(destination), 'Results directory appeared during staging')
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def stop_child(child):
    if child is None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    except ProcessLookupError:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)  # Include any surviving descendants.
    except ProcessLookupError:
        pass
    child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--keep', action='append', default=[], help='Exact file relative to {run}; repeatable')
    parser.add_argument('--min-free-gib', type=float, default=32)
    parser.add_argument('--max-retained-gib', type=float, default=4)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    require(command, 'Supply a command after --')
    require(all(math.isfinite(x) and x > 0 for x in (args.min_free_gib, args.max_retained_gib)),
            'Storage limits must be finite and positive')
    job = os.environ.get('SLURM_JOB_ID', '')
    require(job.isdecimal() and os.environ.get('SLURM_JOB_NUM_NODES', os.environ.get('SLURM_NNODES')) == '1',
            'Run inside a one-node Slurm allocation')
    tmp_root = TMP_ROOT.resolve(strict=True)
    base = Path(os.environ.get('SLURM_TMPDIR') or tmp_root).resolve(strict=True)
    require(base.is_dir() and contained(base, tmp_root),
            'SLURM_TMPDIR must resolve under /tmp; no persistent fallback')
    require(shutil.disk_usage(base).free >= args.min_free_gib * GIB, 'Insufficient job-local free space')
    destination = args.results_dir.parent.resolve(strict=True) / args.results_dir.name
    require(not contained(destination, tmp_root), 'Retained results must be outside /tmp')
    require(not os.path.lexists(destination), 'Results directory must be new')
    cap = int(args.max_retained_gib * GIB)
    quota = project_quota(destination.parent, cap, args.keep) if contained(destination.parent, PROJECTS_ROOT) else None
    scratch = Path(tempfile.mkdtemp(prefix='igrlab-' + job + '-', dir=base))
    identity = scratch.stat()
    run, temporary = scratch / 'run', scratch / 'tmp'
    child, received, status = None, [], 0
    record = dict(schema='deltaai-job-local-v1', job=job, node=os.environ.get('SLURMD_NODENAME'),
                  scratch=str(scratch), run=str(run), started_unix=time.time(), retention_cap_bytes=cap,
                  project_quota_preflight=quota)
    def interrupted(signum, frame):
        received.append(signum)
        raise InterruptedError('Received signal ' + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, interrupted)
    try:
        run.mkdir(); temporary.mkdir()
        argv = [x.replace('{run}', str(run)).replace('{scratch}', str(scratch)) for x in command]
        record['argv'] = argv
        check_outputs(argv, run, scratch)
        env = dict(os.environ, TMPDIR=str(temporary), TEMP=str(temporary), TMP=str(temporary),
                   IGRLAB_JOB_OUTPUT_DIR=str(run), IGRLAB_VTK_PREFIX_PATH=str(run / 'vtk'),
                   DG_IGR_VTK_PREFIX_PATH=str(run / 'vtk'), IGRLAB_OUTPUT_ROOT=str(scratch / 'builds'),
                   DG_IGR_OUTPUT_ROOT=str(scratch / 'builds'))
        with (scratch / 'stdout.log').open('wb') as stdout, (scratch / 'stderr.log').open('wb') as stderr:
            child = subprocess.Popen(argv, cwd=run, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            status = child.wait()
    except (OSError, ValueError) as error:
        record['error'] = str(error)
        status = 128 + received[-1] if received else 2
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(sig, signal.SIG_IGN)
        try:
            stop_child(child)
            for name in ('stdout.log', 'stderr.log'):
                (scratch / name).touch(exist_ok=True)
            record.update(child_returncode=child.returncode if child else None,
                          wrapper_returncode=status if status >= 0 else 128 - status, finished_unix=time.time())
            retain(scratch, run, destination, args.keep, record['retention_cap_bytes'], record)
            if record.get('storage_error'):
                status = 2
        finally:
            actual = scratch.lstat()
            require((actual.st_dev, actual.st_ino) == (identity.st_dev, identity.st_ino) and not scratch.is_symlink(),
                    'Refusing cleanup of replaced scratch directory')
            shutil.rmtree(scratch)
    return status if status >= 0 else 128 - status


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print('deltaai_job_local: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
