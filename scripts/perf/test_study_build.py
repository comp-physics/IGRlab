#!/usr/bin/env python3
"""Host fixtures use tiny local Git repos and a fake public build driver."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import study_build as study


DRIVER = '''#!/usr/bin/env python3
import hashlib,json,os,pathlib,sys
args=sys.argv[1:]
if '--help' in args:
 print('--validation-time HH:MM:SS')
 print('--mfem-output-root DIR')
 print('--tests-only')
 raise SystemExit(0)
value=lambda flag: args[args.index(flag)+1]
source=pathlib.Path(__file__).resolve().parent
output=pathlib.Path(value('--output-root'))
site=pathlib.Path(value('--site-root'))
wt='wt-'+hashlib.sha256(os.fsencode(source)).hexdigest()[:16]
profile='cpu' if '--cpu-only' in args or '--cpu' in args else 'cuda-sm80'
scope=output/'worktrees'/wt/profile
if '--print-plan' in args:
 print(f'[build-plan] output_root={output} worktree_id={wt} scope={scope}')
 print(f'[build-plan] env={site}/site/pace/spack-envs/{profile} app_build={scope}/app/build app_install={scope}/app/install')
 if '--mfem-output-root' in args:
  mfem=pathlib.Path(value('--mfem-output-root'))
  print(f'[build-plan] mfem_source={mfem}/source mfem_build={mfem}/build mfem_install={mfem}/install mfem_tuple_state=shared-planned')
else:
 if os.environ.get('SLURM_JOB_ID'):
  print('inside-allocation submission rejected',file=sys.stderr)
  raise SystemExit(2)
 if os.environ.get('FAKE_SHARED_CACHE_REJECT'):
  print('canonical MFEM cache identity rejected',file=sys.stderr)
  raise SystemExit(3)
 print('PACE build submitted: 987123 (embers, cpu-small)')
 if os.environ.get('FAKE_PARTIAL'):
  print('validation submission failed',file=sys.stderr)
  raise SystemExit(1)
 print('PACE validation submitted: 567890 (after build, embers, gpu-a100)')
 print('literal regex='+value('--test-regex'))
'''


class StudyBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.seed = self.root / 'mfem-seed'
        self.seed.mkdir()
        self.git(self.seed, 'init', '-q')
        self.configure(self.seed)
        (self.seed / 'mfem.hpp').write_text('tiny mfem\n')
        self.git(self.seed, 'add', '.')
        self.git(self.seed, 'commit', '-qm', 'mfem fixture')
        self.git(self.seed, 'tag', '-a', 'v4.9', '-m', 'annotated release')
        self.git(self.seed, 'remote', 'add', 'origin', study.MFEM_ORIGIN)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.git(self.source, 'init', '-q')
        self.configure(self.source)
        self.write('build.sh', DRIVER, executable=True)
        self.write('cmake/sites/pace.json', '{"site":"pace"}\n')
        self.write('scripts/build/pace.sh', '# tracked fixture\n')
        self.write('bytes.bin', b'\x00\xfforiginal\n')
        self.write('delete-me', 'keep\n')
        self.write('.gitignore', 'ignored*\n')
        self.write('mesh-real.msh', 'mesh\n')
        (self.source / 'mesh-link.msh').symlink_to('mesh-real.msh')
        self.git(self.source, '-c', 'protocol.file.allow=always', 'submodule', 'add', '-q', str(self.seed), study.MFEM_PATH)
        self.git(self.source / study.MFEM_PATH, 'remote', 'set-url', 'origin', study.MFEM_ORIGIN)
        self.git(self.source, 'config', '-f', '.gitmodules', 'submodule.tpl/mfem.url', study.MFEM_ORIGIN)
        self.git(self.source, 'add', '.')
        self.git(self.source, 'commit', '-qm', 'source fixture')
        self.dest = self.root / 'snapshot'

    def tearDown(self):
        self.temp.cleanup()

    def configure(self, root):
        self.git(root, 'config', 'user.name', 'Fixture')
        self.git(root, 'config', 'user.email', 'fixture@example.invalid')

    def git(self, root, *args):
        result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, check=True)
        return result.stdout.decode().strip()

    def write(self, name, data, executable=False):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        if executable:
            path.chmod(0o755)

    def freeze(self, ref='HEAD'):
        return study.freeze(self.source, ref, self.dest)

    def test_clean_standalone_detached_snapshot_with_annotated_mfem_tag(self):
        identity = self.freeze()
        self.assertEqual(identity['source_status'], 'clean')
        study.verify_source(identity)
        for root in (self.dest, self.dest / study.MFEM_PATH):
            self.assertFalse((root / '.git/objects/info/alternates').exists())
            self.assertEqual(self.git(root, 'rev-parse', '--abbrev-ref', 'HEAD'), 'HEAD')
        self.assertEqual(self.git(self.dest / study.MFEM_PATH, 'cat-file', '-t', 'v4.9'), 'tag')
        self.assertEqual(self.git(self.dest / study.MFEM_PATH, 'remote', 'get-url', 'origin'), study.MFEM_ORIGIN)
        original = self.source / '.git/objects'
        for path in original.rglob('*'):
            if path.is_file() and (copied := self.dest / '.git/objects' / path.relative_to(original)).exists():
                self.assertNotEqual(path.stat().st_ino, copied.stat().st_ino)

    def test_working_tree_preserves_binary_deleted_staged_and_literal_untracked_bytes(self):
        dirty = b'\xff\x00candidate\n'
        self.write('bytes.bin', dirty)
        (self.source / 'delete-me').unlink()
        self.write('added-to-index', b'\x01staged\x00')
        self.git(self.source, 'add', 'added-to-index')
        literal = 'literal $HOME `echo unsafe`\n space.bin'
        self.write(literal, b'\x00\x80literal')
        self.write('ignored-output', 'ignored bytes')
        before = self.git(self.source, 'status', '--porcelain')
        identity = self.freeze('working-tree')
        self.assertEqual((self.dest / 'bytes.bin').read_bytes(), dirty)
        self.assertEqual((self.dest / literal).read_bytes(), b'\x00\x80literal')
        self.assertEqual((self.dest / 'added-to-index').read_bytes(), b'\x01staged\x00')
        self.assertFalse((self.dest / 'delete-me').exists())
        self.assertFalse((self.dest / 'ignored-output').exists())
        self.assertEqual(identity['source_status'], 'dirty')
        self.assertEqual(self.git(self.source, 'status', '--porcelain'), before)
        study.verify_source(identity)

    def test_standalone_snapshot_can_publish_at_stable_content_cache_path(self):
        self.write('bytes.bin', b'dirty candidate\x00')
        identity = self.freeze('working-tree')
        before = self.plan(identity)
        published = self.root / 'cache/sources' / identity['source_fingerprint']
        published.parent.mkdir(parents=True)
        self.dest.rename(published)
        reused = dict(identity, source_snapshot=str(published))
        study.verify_source(reused)
        after = self.plan(reused)
        self.assertEqual(before['configuration_key'], after['configuration_key'])
        self.assertNotEqual(before['app_build'], after['app_build'])
        self.assertEqual(after['app_build'], self.plan(reused)['app_build'])
        self.assertEqual(os.readlink(published / 'mesh-link.msh'), 'mesh-real.msh')
        self.assertEqual((published / study.MFEM_PATH / 'mfem.hpp').read_text(), 'tiny mfem\n')
        self.assertFalse((published / '.git/objects/info/alternates').exists())
        self.assertFalse((published / study.MFEM_PATH / '.git/objects/info/alternates').exists())

    def test_literal_working_bytes_change_content_fingerprint(self):
        first = self.freeze('working-tree')
        self.write('bytes.bin', b'new binary\x00')
        second = study.freeze(self.source, 'working-tree', self.root / 'snapshot2')
        self.assertEqual(first['source_revision'], second['source_revision'])
        self.assertNotEqual(first['source_fingerprint'], second['source_fingerprint'])

    def test_ordinary_ref_excludes_dirty_source_without_changing_it(self):
        self.write('bytes.bin', b'dirty')
        identity = self.freeze('HEAD')
        self.assertEqual((self.dest / 'bytes.bin').read_bytes(), b'\x00\xfforiginal\n')
        self.assertEqual((self.source / 'bytes.bin').read_bytes(), b'dirty')
        self.assertEqual(identity['source_status'], 'clean')

    def test_tracked_confined_symlink_is_preserved_exactly(self):
        identity = self.freeze('working-tree')
        self.assertEqual(os.readlink(self.dest / 'mesh-link.msh'), 'mesh-real.msh')
        study.verify_source(identity)

    def test_untracked_and_modified_symlinks_reject(self):
        (self.source / 'untracked-link').symlink_to('bytes.bin')
        with self.assertRaisesRegex(study.StudyBuildError, 'symlink'):
            self.freeze('working-tree')
        (self.source / 'untracked-link').unlink()
        (self.source / 'mesh-link.msh').unlink()
        (self.source / 'mesh-link.msh').symlink_to('delete-me')
        with self.assertRaisesRegex(study.StudyBuildError, 'modified tracked symlink'):
            self.freeze('working-tree')

    def test_tracked_escape_symlink_rejects(self):
        (self.source / 'mesh-link.msh').unlink()
        (self.source / 'mesh-link.msh').symlink_to('../mfem-seed/mfem.hpp')
        self.git(self.source, 'add', 'mesh-link.msh')
        self.git(self.source, 'commit', '-qm', 'escaping link')
        with self.assertRaisesRegex(study.StudyBuildError, 'escapes snapshot'):
            self.freeze()

    def test_dirty_mfem_rejects_and_original_is_preserved(self):
        path = self.source / study.MFEM_PATH / 'mfem.hpp'
        path.write_text('dirty mfem')
        with self.assertRaisesRegex(study.StudyBuildError, 'dirty'):
            self.freeze('working-tree')
        self.assertEqual(path.read_text(), 'dirty mfem')

    def test_missing_annotated_tag_rejects(self):
        self.git(self.source / study.MFEM_PATH, 'tag', '-d', 'v4.9')
        with self.assertRaises(study.StudyBuildError):
            self.freeze()

    def test_annotated_tag_metadata_drift_rejects(self):
        identity = self.freeze()
        mfem = self.dest / study.MFEM_PATH
        self.configure(mfem)
        self.git(mfem, 'tag', '-fa', 'v4.9', '-m', 'changed annotated metadata')
        self.assertEqual(self.git(mfem, 'rev-parse', 'v4.9^{commit}'), identity['mfem_revision'])
        with self.assertRaisesRegex(study.StudyBuildError, 'identity changed'):
            study.verify_source(identity)

    def test_unsupported_historical_driver_rejects_before_creating_snapshot(self):
        self.git(self.source, 'rm', 'cmake/sites/pace.json')
        self.git(self.source, 'commit', '-qm', 'old unsupported source')
        with self.assertRaisesRegex(study.StudyBuildError, 'unsupported historical source'):
            self.freeze()
        self.assertFalse(self.dest.exists())

    def test_source_and_mfem_drift_reject_before_planning(self):
        identity = self.freeze()
        (self.dest / 'bytes.bin').write_bytes(b'altered')
        with self.assertRaisesRegex(study.StudyBuildError, 'identity changed'):
            study.verify_source(identity)
        (self.dest / 'bytes.bin').write_bytes(b'\x00\xfforiginal\n')
        (self.dest / study.MFEM_PATH / 'mfem.hpp').write_text('altered')
        with self.assertRaisesRegex(study.StudyBuildError, 'dirty'):
            study.verify_source(identity)

    def test_existing_or_symlinked_destination_rejects(self):
        self.dest.mkdir()
        (self.dest / 'unrelated').write_text('preserved')
        with self.assertRaisesRegex(study.StudyBuildError, 'already exists'):
            self.freeze()
        self.assertEqual((self.dest / 'unrelated').read_text(), 'preserved')
        link = self.root / 'parent-link'
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(study.StudyBuildError, 'symlinked'):
            study.freeze(self.source, 'HEAD', link / 'snapshot3')

    def plan(self, identity, regex='^(dg_operator|sigma)$', *, validation_time=None, profile='cuda-sm80'):
        return study.build_plan(identity, self.root / 'output with $HOME `literal`',
                                self.root / 'shared-site', self.root / 'spack-tool', regex,
                                validation_time=validation_time, profile=profile)

    def test_cpu_plan_uses_independent_cpu_cache_and_public_cpu_only_route(self):
        identity = self.freeze()
        default = self.plan(identity)
        cuda = self.plan(identity, profile='cuda-sm80')
        cpu = self.plan(identity, profile='cpu')
        self.assertEqual(default, cuda)
        self.assertEqual(cpu['profile'], 'cpu')
        self.assertIn('--cpu-only', cpu['argv'])
        self.assertNotIn('--a100', cpu['argv'])
        self.assertNotIn('--nvtx-profile', cpu['argv'])
        self.assertNotEqual(cpu['dependency_configuration_key'], cuda['dependency_configuration_key'])
        self.assertNotEqual(cpu['app_build'], cuda['app_build'])
        self.assertTrue(cpu['spack_env'].endswith('/cpu'))
        self.assertEqual(cpu['dependency_configuration']['mfem_cuda_archs'], 'none')
        self.assertFalse(Path(cpu['mfem_build']).exists())
        self.assertTrue(study.submit_build(cpu)['success'])
        with self.assertRaisesRegex(study.StudyBuildError, 'support cpu'):
            self.plan(identity, profile='cuda-sm70')

    def cpu_request(self, identity):
        request = self.root / 'literal request $HOME.json'
        argv = ['--site', 'pace', '--cpu-only', '--hdg-runtime', '--no-clean',
                '--no-research-tests', '--no-deps', '--output-root', str(self.root / 'already-built'),
                '--site-root', str(self.root / 'site'), '--spack-root', str(self.root / 'spack'),
                '--mfem-output-root', str(self.root / 'already-built-mfem'), '--jobs', '8',
                '--test-regex', '^(cpu-igr|cpu-fields)$']
        request.write_text(json.dumps({'source': identity['source_snapshot'], 'argv': argv}))
        return request

    def test_existing_cpu_request_uses_exact_public_paths_without_submission(self):
        identity = self.freeze()
        request = self.cpu_request(identity)
        before = request.read_bytes()
        plan = study.existing_cpu_plan(identity, request)
        self.assertEqual(plan['request_sha256'], study.hashlib.sha256(before).hexdigest())
        self.assertTrue(plan['app_build'].startswith(str(self.root / 'already-built')))
        self.assertEqual(plan['mfem_build'], str(self.root / 'already-built-mfem/build'))
        self.assertEqual(plan['correctness_regex'], '^(cpu-igr|cpu-fields)$')
        self.assertEqual(request.read_bytes(), before)
        self.assertFalse(Path(plan['app_build']).exists())
        with self.assertRaisesRegex(study.StudyBuildError, 'cannot be resubmitted'):
            study.submit_build(plan)

    def test_existing_cpu_request_rejects_cuda_source_and_option_drift(self):
        identity = self.freeze()
        request = self.cpu_request(identity)
        original = json.loads(request.read_text())
        variants = [{**original, 'source': str(self.source)},
                    {**original, 'argv': original['argv'] + ['--a100']},
                    {**original, 'argv': original['argv'] + ['--no-tests']},
                    {**original, 'argv': original['argv'] + ['--nvtx-profile']},
                    {**original, 'argv': original['argv'] + ['--test-regex', 'different']}]
        for value in variants:
            with self.subTest(value=value):
                request.write_text(json.dumps(value))
                with self.assertRaises(study.StudyBuildError):
                    study.existing_cpu_plan(identity, request)

    def test_existing_cpu_request_change_during_print_plan_rejects(self):
        identity = self.freeze()
        request = self.cpu_request(identity)
        original = study._run
        def mutate(argv, **kwargs):
            result = original(argv, **kwargs)
            if '--print-plan' in argv:
                request.write_text(request.read_text() + '\n')
            return result
        with patch.object(study, '_run', side_effect=mutate):
            with self.assertRaisesRegex(study.StudyBuildError, 'request changed'):
                study.existing_cpu_plan(identity, request)

    def legacy_plan(self):
        self.write('build.sh', DRIVER.replace("print('--validation-time HH:MM:SS')", "print('legacy site helper')"), executable=True)
        self.git(self.source, 'add', 'build.sh')
        self.git(self.source, 'commit', '-qm', 'legacy fixture')
        return self.plan(self.freeze(), validation_time='00:15:00')

    def receipt(self, plan):
        return {'success': True, 'build_job_id': '987123', 'validation_job_id': '567890',
                'stdout': 'canonical queued receipt', 'stderr': '',
                'source_fingerprint': plan['source']['source_fingerprint'],
                'configuration_key': plan['configuration_key']}

    def control(self, plan, state='PENDING', limit='02:00:00'):
        return (f'JobId=567890 JobName=IGRlab-check UserId=fixture({os.getuid()}) '
                f'Account=gts-sbryngelson3 QOS=embers JobState={state} TimeLimit={limit} '
                f'Command={plan["source"]["source_snapshot"]}/scripts/build/pace.sh WorkDir=/fixture')

    def scheduler(self, responses, calls):
        original = study._run
        iterator = iter(responses)
        def run(argv, **kwargs):
            if argv[0] != 'scontrol':
                return original(argv, **kwargs)
            calls.append(argv)
            response = next(iterator)
            code, stdout, stderr = response if isinstance(response, tuple) else (0, response, '')
            return subprocess.CompletedProcess(argv, code, stdout.encode(), stderr.encode())
        return run

    def test_plan_is_source_configuration_scoped_and_has_no_build_side_effect(self):
        identity = self.freeze()
        regex = '^(one|two) $HOME `literal`\nlast$'
        plan = self.plan(identity, regex)
        self.assertEqual(plan['argv'][plan['argv'].index('--test-regex') + 1], regex)
        self.assertIn('--submit', plan['argv'])
        self.assertIn('--no-clean', plan['argv'])
        self.assertFalse(Path(plan['output_root']).exists())
        self.assertEqual(plan['spack_env'], str(self.root / 'shared-site/site/pace/spack-envs/cuda-sm80'))
        self.assertEqual(plan['configuration_key'], self.plan(identity, regex)['configuration_key'])
        other = self.plan(identity, '^another$')
        self.assertEqual(plan['output_root'], other['output_root'])
        self.assertEqual(plan['app_build'], other['app_build'])
        self.assertEqual(plan['configuration_key'], other['configuration_key'])
        self.assertNotEqual(plan['argv'], other['argv'])
        self.assertEqual(other['correctness_regex'], '^another$')
        self.assertNotIn('--test-regex', plan['configuration']['flags'])
        self.assertEqual(plan['dependency_configuration_key'], self.plan(identity, '^another$')['dependency_configuration_key'])
        self.assertEqual(plan['mfem_build'], self.plan(identity, '^another$')['mfem_build'])
        self.assertEqual(plan['mfem_cache']['mode'], 'shared-canonical')
        self.assertTrue(plan['run_only_validation'])
        self.assertFalse(Path(plan['mfem_cache']['output_root']).exists())

    def test_application_only_change_and_validation_selection_reuse_dependency_tuple(self):
        first = self.plan(self.freeze())
        self.write('bytes.bin', b'application change\x00')
        second = self.plan(study.freeze(self.source, 'working-tree', self.root / 'snapshot2'), '^different$')
        self.assertNotEqual(first['configuration_key'], second['configuration_key'])
        self.assertNotEqual(first['app_build'], second['app_build'])
        for key in ('dependency_configuration_key', 'mfem_source', 'mfem_build', 'mfem_install', 'mfem_identity'):
            self.assertEqual(first[key], second[key], key)
        self.assertEqual(first['mfem_cache']['lock'], second['mfem_cache']['lock'])

    def test_dependency_key_changes_with_mfem_source_and_release_identity(self):
        first = self.plan(self.freeze())
        mfem = self.source / study.MFEM_PATH
        self.configure(mfem)
        (mfem / 'mfem.hpp').write_text('next clean MFEM source\n')
        self.git(mfem, 'add', '.')
        self.git(mfem, 'commit', '-qm', 'new source')
        self.git(mfem, 'tag', '-d', 'v4.9')
        self.git(mfem, 'tag', '-a', 'v4.9', '-m', 'new annotated fixture release')
        self.git(self.source, 'add', study.MFEM_PATH)
        self.git(self.source, 'commit', '-qm', 'new MFEM pin')
        second = self.plan(study.freeze(self.source, 'HEAD', self.root / 'snapshot2'))
        self.assertNotEqual(first['dependency_configuration_key'], second['dependency_configuration_key'])
        self.assertNotEqual(first['mfem_build'], second['mfem_build'])

    def test_site_compiler_configuration_and_canonical_policy_change_dependency_key(self):
        first = self.plan(self.freeze())
        for number, name, data in ((2, 'cmake/sites/pace.json', '{"compiler":{"version":"next"}}\n'),
                                    (3, 'build_mfem_tpl.sh', '# changed canonical MFEM policy\n'),
                                    (4, 'scripts/build/mfem_cache.sh', '# changed cache admission\n')):
            with self.subTest(name=name):
                self.write(name, data)
                frozen = study.freeze(self.source, 'working-tree', self.root / f'snapshot{number}')
                changed = self.plan(frozen)
                self.assertNotEqual(first['dependency_configuration_key'], changed['dependency_configuration_key'])
                self.assertNotEqual(first['mfem_build'], changed['mfem_build'])

    def test_spack_tool_root_bytes_and_site_graph_namespace_change_dependency_key(self):
        identity = self.freeze()
        spack = self.root / 'spack-tool'
        (spack / 'bin').mkdir(parents=True)
        (spack / 'bin/spack').write_text('selected Spack tool\n')
        first = self.plan(identity)
        (spack / 'bin/spack').write_text('updated selected Spack tool\n')
        second = self.plan(identity)
        third = study.build_plan(identity, self.root / 'output with $HOME `literal`',
                                 self.root / 'another-site', spack, '^(dg_operator|sigma)$')
        fourth = study.build_plan(identity, self.root / 'output with $HOME `literal`',
                                  self.root / 'shared-site', self.root / 'another-spack', '^(dg_operator|sigma)$')
        self.assertEqual(len({plan['dependency_configuration_key'] for plan in (first, second, third, fourth)}), 4)

    def test_first_site_lock_creation_does_not_change_planned_dependency_identity(self):
        identity = self.freeze()
        first = self.plan(identity)
        environment = Path(first['spack_env'])
        environment.mkdir(parents=True)
        (environment / 'spack.lock').write_text('first build creates the concrete graph')
        second = self.plan(identity)
        self.assertEqual(first['dependency_configuration_key'], second['dependency_configuration_key'])
        self.assertEqual(first['mfem_build'], second['mfem_build'])

    def test_effective_compile_flags_are_frozen_and_change_app_and_mfem_keys(self):
        identity = self.freeze()
        with patch.dict(os.environ, {name: '' for name in study.BUILD_FLAG_ENV}):
            first = self.plan(identity)
            self.assertTrue(all(first['env'][name] == '' for name in study.BUILD_FLAG_ENV))
            with patch.dict(os.environ, {'CXXFLAGS': '-fno-omit-frame-pointer', 'CUDAFLAGS': '-lineinfo'}):
                changed = self.plan(identity)
                self.assertNotEqual(first['dependency_configuration_key'], changed['dependency_configuration_key'])
                self.assertNotEqual(first['configuration_key'], changed['configuration_key'])
                self.assertEqual(changed['env']['CUDAFLAGS'], '-lineinfo')
                with self.assertRaisesRegex(study.StudyBuildError, 'plan changed'):
                    study.submit_build(first)

    def test_historical_driver_uses_its_original_mfem_scope_without_overlay(self):
        old = DRIVER.replace(" print('--mfem-output-root DIR')\n", '').replace(" print('--tests-only')\n", '')
        self.write('build.sh', old, executable=True)
        self.git(self.source, 'add', 'build.sh')
        self.git(self.source, 'commit', '-qm', 'historical cache capability')
        plan = self.plan(self.freeze())
        self.assertNotIn('--mfem-output-root', plan['argv'])
        self.assertEqual(plan['mfem_cache']['mode'], 'historical-source-scoped')
        self.assertFalse(plan['run_only_validation'])
        self.assertEqual(Path(plan['mfem_build']).parents[1], Path(plan['app_build']).parents[1])

    def test_corrupt_or_incomplete_shared_cache_canonical_rejection_has_no_local_fallback(self):
        plan = self.plan(self.freeze())
        with patch.dict(os.environ, {'FAKE_SHARED_CACHE_REJECT': '1'}):
            result = study.submit_build(plan)
        self.assertFalse(result['success'])
        self.assertEqual(result['returncode'], 3)
        self.assertIsNone(result['build_job_id'])
        self.assertIn('canonical MFEM cache identity rejected', result['stderr'])
        self.assertEqual(result['argv'], plan['argv'])

    def test_shared_dependency_plan_fields_are_verified_before_submission(self):
        plan = self.plan(self.freeze())
        plan['mfem_build'] = str(self.root / 'unrelated-build')
        with self.assertRaisesRegex(study.StudyBuildError, 'plan changed: mfem_build'):
            study.submit_build(plan)

    def test_public_submission_preserves_ids_stdout_and_literal_argv(self):
        plan = self.plan(self.freeze(), '^literal $HOME `never run`$')
        result = study.submit_build(plan)
        self.assertTrue(result['success'])
        self.assertEqual((result['build_job_id'], result['validation_job_id']), ('987123', '567890'))
        self.assertIn('literal $HOME `never run`', result['stdout'])

    def test_partial_public_submission_retains_existing_job_and_failure(self):
        plan = self.plan(self.freeze())
        with patch.dict(os.environ, {'FAKE_PARTIAL':'1'}):
            result = study.submit_build(plan)
        self.assertFalse(result['success'])
        self.assertEqual(result['build_job_id'], '987123')
        self.assertIsNone(result['validation_job_id'])
        self.assertIn('validation submission failed', result['stderr'])

    def test_public_allocation_guard_is_preserved(self):
        plan = self.plan(self.freeze())
        with patch.dict(os.environ, {'SLURM_JOB_ID':'123'}):
            result = study.submit_build(plan)
        self.assertFalse(result['success'])
        self.assertIn('inside-allocation', result['stderr'])

    def test_plan_tampering_rejects_before_submission(self):
        plan = self.plan(self.freeze())
        plan['argv'].append('--clean')
        with self.assertRaisesRegex(study.StudyBuildError, 'plan changed'):
            study.submit_build(plan)

    def test_validation_schedule_does_not_change_build_cache_and_capability_is_explicit(self):
        identity = self.freeze()
        ordinary = self.plan(identity)
        short = self.plan(identity, validation_time='00:15:00')
        shorter = self.plan(identity, validation_time='00:05:00')
        self.assertEqual(ordinary['configuration_key'], short['configuration_key'])
        self.assertEqual(short['app_build'], shorter['app_build'])
        self.assertEqual(ordinary['app_build'], short['app_build'])
        self.assertEqual(short['validation_time_mode'], 'canonical')
        self.assertEqual(short['argv'][-2:], ['--validation-time', '00:15:00'])

    def test_legacy_pending_validation_is_shortened_and_observed_readback_retained(self):
        plan = self.legacy_plan()
        self.assertEqual(plan['validation_time_mode'], 'scheduler-update')
        self.assertNotIn('--validation-time', plan['argv'])
        calls = []
        with patch.object(study, '_run', side_effect=self.scheduler(
                [self.control(plan), '', self.control(plan, limit='00:15:00')], calls)):
            result = study.submit_build(plan)
        self.assertTrue(result['success'])
        self.assertEqual(result['validation_job_id'], '567890')
        self.assertEqual(result['validation_time']['status'], 'updated')
        self.assertEqual(result['validation_time']['actual'], '00:15:00')
        self.assertEqual(calls[1], ['scontrol', 'update', 'JobId=567890', 'TimeLimit=00:15:00'])
        self.assertEqual(len(result['validation_time']['commands']), 3)

    def test_native_limit_and_completed_validation_are_guarded_noops(self):
        plan = self.plan(self.freeze(), validation_time='00:15:00')
        for state, limit, wanted in (('PENDING', '00:15:00', 'within-limit-noop'),
                                      ('RUNNING', '00:10:00', 'within-limit-noop'),
                                      ('COMPLETED', '02:00:00', 'completed-noop')):
            calls = []
            with self.subTest(state=state), patch.object(study, '_run', side_effect=self.scheduler(
                    [self.control(plan, state, limit)], calls)):
                result = study.ensure_validation_time(plan, self.receipt(plan))
            self.assertTrue(result['success'])
            self.assertEqual(result['validation_time']['status'], wanted)
            self.assertEqual(len(calls), 1)

    def test_unrelated_or_running_incompatible_validation_rejects_without_mutation(self):
        plan = self.legacy_plan()
        valid = self.control(plan)
        wrong = [valid.replace(f'fixture({os.getuid()})', f'fixture({os.getuid() + 1})'),
                 valid.replace('Account=gts-sbryngelson3', 'Account=unrelated'),
                 valid.replace('QOS=embers', 'QOS=inferno'),
                 valid.replace('JobName=IGRlab-check', 'JobName=unrelated'),
                 valid.replace('Command=', 'Command=/unrelated'),
                 valid.replace('JobId=567890', 'JobId=999'),
                 self.control(plan, limit='00:00:00'),
                 self.control(plan, 'RUNNING')]
        for text in wrong:
            calls = []
            with self.subTest(text=text), patch.object(study, '_run', side_effect=self.scheduler([text], calls)):
                result = study.ensure_validation_time(plan, self.receipt(plan))
            self.assertFalse(result['success'])
            self.assertEqual(result['validation_job_id'], '567890')
            self.assertEqual(result['stdout'], 'canonical queued receipt')
            self.assertIn('error', result)
            self.assertEqual(len(calls), 1)

    def test_failed_limit_update_preserves_ids_and_retries_existing_receipt(self):
        plan = self.legacy_plan()
        calls = []
        with patch.object(study, '_run', side_effect=self.scheduler(
                [self.control(plan), (1, '', 'update rejected')], calls)):
            failed = study.ensure_validation_time(plan, self.receipt(plan))
        self.assertFalse(failed['success'])
        self.assertTrue(failed['canonical_submission_success'])
        self.assertEqual(failed['validation_time']['commands'][-1]['returncode'], 1)
        calls = []
        with patch.object(study, '_run', side_effect=self.scheduler(
                [self.control(plan, limit='00:15:00')], calls)):
            retried = study.ensure_validation_time(plan, failed)
        self.assertTrue(retried['success'])
        self.assertNotIn('error', retried)
        self.assertEqual(retried['build_job_id'], '987123')
        self.assertEqual(retried['validation_job_id'], '567890')
        self.assertEqual(len(calls), 1)

    def test_successful_update_with_wrong_readback_is_not_admitted(self):
        plan = self.legacy_plan()
        calls = []
        with patch.object(study, '_run', side_effect=self.scheduler(
                [self.control(plan), '', self.control(plan)], calls)):
            result = study.ensure_validation_time(plan, self.receipt(plan))
        self.assertFalse(result['success'])
        self.assertEqual(result['validation_time']['actual'], '02:00:00')
        self.assertIn('readback exceeds', result['error'])
        self.assertEqual(result['validation_job_id'], '567890')

    def test_invalid_validation_limits_and_mode_tampering_reject_before_submission(self):
        identity = self.freeze()
        for value in ('00:00:00', '00:15:01', '00:60:00', '2:00:00', '02:01:00', False):
            with self.subTest(value=value), self.assertRaises(study.StudyBuildError):
                self.plan(identity, validation_time=value)
        plan = self.plan(identity, validation_time='00:15:00')
        plan['validation_time_mode'] = 'scheduler-update'
        with self.assertRaisesRegex(study.StudyBuildError, 'plan changed'):
            study.submit_build(plan)

    def test_no_requested_limit_leaves_legacy_receipt_unchanged(self):
        plan = self.plan(self.freeze())
        receipt = self.receipt(plan)
        with patch.object(study, '_run') as run:
            self.assertEqual(study.ensure_validation_time(plan, receipt), receipt)
            run.assert_not_called()

    def test_submission_timeout_preserves_job_ids_from_partial_stdout(self):
        plan = self.plan(self.freeze())
        original = study.subprocess.run
        def run(argv, **kwargs):
            if argv == plan['argv']:
                raise subprocess.TimeoutExpired(argv, 300,
                    output=b'PACE build submitted: 123 (embers)\nPACE validation submitted: 456 (embers)\n')
            return original(argv, **kwargs)
        with patch.object(study.subprocess, 'run', side_effect=run):
            result = study.submit_build(plan)
        self.assertFalse(result['success'])
        self.assertEqual(result['returncode'], 124)
        self.assertEqual((result['build_job_id'], result['validation_job_id']), ('123', '456'))
        self.assertIn('timed out', result['stderr'])


if __name__ == '__main__':
    unittest.main()
