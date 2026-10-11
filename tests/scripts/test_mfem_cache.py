#!/usr/bin/env python3
"""Host-only shared MFEM source/lock fixtures; no application builds."""
from pathlib import Path
import subprocess
import tempfile
import time
import unittest


HELPER = Path(__file__).resolve().parents[2] / 'scripts/build/mfem_cache.sh'


class MfemCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.seed = self.root / 'seed'
        self.seed.mkdir()
        self.git(self.seed, 'init', '-q')
        self.git(self.seed, 'config', 'user.name', 'Fixture')
        self.git(self.seed, 'config', 'user.email', 'fixture@example.invalid')
        (self.seed / 'mfem.hpp').write_text('tiny MFEM source\n')
        self.git(self.seed, 'add', '.')
        self.git(self.seed, 'commit', '-qm', 'source')
        self.git(self.seed, 'tag', '-a', 'v4.9', '-m', 'release')
        self.git(self.seed, 'remote', 'add', 'origin', 'https://github.com/mfem/mfem.git')
        self.pin = self.git(self.seed, 'rev-parse', 'HEAD')
        self.cache = self.root / 'cache with spaces'

    def tearDown(self):
        self.temp.cleanup()

    def git(self, directory, *args):
        return subprocess.check_output(['git', '-C', str(directory), *args],
                                       stderr=subprocess.PIPE, text=True).strip()

    def command(self, body, cache=None):
        return ['bash', '-c', 'set -euo pipefail\nsource "$1"\n'
                'mfem_cache_plan "$2" "$3" "$4"\n' + body,
                'fixture', str(HELPER), str(cache or self.cache), str(self.seed), self.pin]

    def run_helper(self, body, cache=None):
        return subprocess.run(self.command(body, cache), capture_output=True, text=True, timeout=15)

    def fill(self):
        result = self.run_helper('mfem_cache_acquire\nmfem_cache_prepare_source\nmfem_cache_release')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_plan_is_pure_and_outputs_stable_tuple(self):
        result = self.run_helper('printf "%s\\n" "$MFEM_CACHE_SOURCE" "$MFEM_CACHE_BUILD" "$MFEM_CACHE_INSTALL"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [str(self.cache / name) for name in ('source', 'build', 'install')])
        self.assertFalse(self.cache.exists())

    def test_cold_fill_then_hit_retains_clean_standalone_source(self):
        self.fill()
        source = self.cache / 'source'
        before = (source / 'mfem.hpp').stat()
        self.fill()
        self.assertEqual((source / 'mfem.hpp').stat().st_mtime_ns, before.st_mtime_ns)
        self.assertEqual(self.git(source, 'rev-parse', 'HEAD'), self.pin)
        self.assertEqual(self.git(source, 'status', '--porcelain'), '')
        self.assertFalse((source / '.git/objects/info/alternates').exists())
        oid = self.git(source, 'rev-parse', 'HEAD:mfem.hpp')
        self.assertNotEqual((source / '.git/objects' / oid[:2] / oid[2:]).stat().st_ino,
                            (self.seed / '.git/objects' / oid[:2] / oid[2:]).stat().st_ino)
        self.assertFalse((self.cache / 'build').exists())
        self.assertFalse((self.cache / 'install').exists())

    def test_dirty_seed_and_dirty_cached_source_reject(self):
        self.fill()
        (self.seed / 'untracked').write_text('preserve')
        result = self.run_helper('mfem_cache_acquire\nmfem_cache_prepare_source')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('clean selected checkout', result.stderr)
        (self.seed / 'untracked').unlink()
        source = self.cache / 'source'
        (source / 'mfem.hpp').write_text('modified')
        result = self.run_helper('mfem_cache_acquire\nmfem_cache_prepare_source')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('incomplete, dirty', result.stderr)
        self.assertEqual((source / 'mfem.hpp').read_text(), 'modified')

    def test_index_flags_cannot_hide_dirty_seed_or_source(self):
        self.fill()
        for checkout in (self.seed, self.cache / 'source'):
            with self.subTest(checkout=checkout):
                self.git(checkout, 'update-index', '--assume-unchanged', 'mfem.hpp')
                (checkout / 'mfem.hpp').write_text('hidden changed source')
                self.assertEqual(self.git(checkout, 'status', '--porcelain'), '')
                result = self.run_helper('mfem_cache_acquire\nmfem_cache_prepare_source')
                self.assertNotEqual(result.returncode, 0)
                self.git(checkout, 'update-index', '--no-assume-unchanged', 'mfem.hpp')
                self.git(checkout, 'checkout', '--', 'mfem.hpp')

    def test_incomplete_source_is_preserved_and_rejected(self):
        source = self.cache / 'source'
        source.mkdir(parents=True)
        (source / 'unrelated').write_text('preserve')
        result = self.run_helper('mfem_cache_acquire\nmfem_cache_prepare_source')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((source / 'unrelated').read_text(), 'preserve')

    def test_different_origin_tag_or_borrowed_objects_reject(self):
        self.fill()
        source = self.cache / 'source'
        for mode in ('origin', 'tag', 'alternates'):
            with self.subTest(mode=mode):
                if mode == 'origin':
                    self.git(source, 'remote', 'set-url', 'origin', 'https://example.invalid/mfem.git')
                elif mode == 'tag':
                    self.git(source, 'tag', '-f', 'v4.9', self.pin)
                else:
                    (source / '.git/objects/info/alternates').write_text(str(self.seed / '.git/objects'))
                result = self.run_helper('mfem_cache_acquire\nmfem_cache_prepare_source')
                self.assertNotEqual(result.returncode, 0, result.stderr)
                if mode == 'origin':
                    self.git(source, 'remote', 'set-url', 'origin', self.git(self.seed, 'remote', 'get-url', 'origin'))
                elif mode == 'tag':
                    self.git(source, 'update-ref', 'refs/tags/v4.9', self.git(self.seed, 'rev-parse', 'refs/tags/v4.9'))
                else:
                    (source / '.git/objects/info/alternates').unlink()

    def test_symlinked_paths_and_lock_reject(self):
        self.cache.mkdir()
        (self.cache / '.lock').symlink_to(self.root / 'unrelated')
        self.assertNotEqual(self.run_helper('true').returncode, 0)
        (self.cache / '.lock').unlink()
        (self.cache / 'source').symlink_to(self.seed, target_is_directory=True)
        self.assertNotEqual(self.run_helper('true').returncode, 0)
        link = self.root / 'link'
        link.symlink_to(self.root, target_is_directory=True)
        self.assertNotEqual(self.run_helper('true', link / 'cache2').returncode, 0)
        self.assertNotEqual(self.run_helper('true', self.seed / 'build-cache').returncode, 0)

    def test_prepare_requires_lock(self):
        result = self.run_helper('mfem_cache_prepare_source')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires the writer lock', result.stderr)
        self.assertFalse(self.cache.exists())

    def test_matching_partial_build_intent_is_retained_without_rewrite(self):
        body = 'mfem_cache_acquire\nmfem_cache_admit_intent $\'pin=selected\\ncompiler=selected\\n\'\nmfem_cache_prepare_source\nmfem_cache_release'
        result = self.run_helper(body)
        self.assertEqual(result.returncode, 0, result.stderr)
        intent = self.cache / 'intent-v1.txt'
        original = (intent.read_bytes(), intent.stat().st_mtime_ns)
        (self.cache / 'build').mkdir()
        (self.cache / 'build/partial-object').write_text('incomplete but admitted')
        result = self.run_helper(body)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((intent.read_bytes(), intent.stat().st_mtime_ns), original)
        self.assertEqual((self.cache / 'build/partial-object').read_text(), 'incomplete but admitted')

    def test_changed_symlinked_or_unlocked_intent_rejects_without_cleanup(self):
        self.assertNotEqual(self.run_helper('mfem_cache_admit_intent selected').returncode, 0)
        result = self.run_helper('mfem_cache_acquire\nmfem_cache_admit_intent selected\nmfem_cache_release')
        self.assertEqual(result.returncode, 0, result.stderr)
        intent = self.cache / 'intent-v1.txt'
        original = intent.read_bytes()
        result = self.run_helper('mfem_cache_acquire\nmfem_cache_admit_intent changed')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('intent differs', result.stderr)
        self.assertEqual(intent.read_bytes(), original)
        intent.unlink()
        target = self.root / 'unrelated-intent'
        target.write_text('preserve')
        intent.symlink_to(target)
        result = self.run_helper('mfem_cache_acquire\nmfem_cache_admit_intent selected')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(target.read_text(), 'preserve')

    def test_foreign_partial_artifacts_without_intent_reject_but_empty_dirs_admit(self):
        for name in ('build', 'install'):
            with self.subTest(name=name):
                directory = self.cache / name
                directory.mkdir(parents=True)
                artifact = directory / 'foreign-partial'
                artifact.write_text('preserve')
                result = self.run_helper('mfem_cache_acquire\nmfem_cache_admit_intent selected')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('no admitted build intent', result.stderr)
                self.assertEqual(artifact.read_text(), 'preserve')
                self.assertFalse((self.cache / 'intent-v1.txt').exists())
                artifact.unlink()
        result = self.run_helper('mfem_cache_acquire\nmfem_cache_admit_intent selected\nmfem_cache_release')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_concurrent_misses_serialize_and_second_reuses_published_source(self):
        first = subprocess.Popen(self.command('mfem_cache_acquire\nmfem_cache_prepare_source\n'
                    'printf "ready\\n"\nread -r release\nmfem_cache_release'),
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        second = None
        try:
            self.assertEqual(first.stdout.readline().strip(), 'ready')
            before = (self.cache / 'source/mfem.hpp').stat().st_mtime_ns
            second = subprocess.Popen(self.command('mfem_cache_acquire\nmfem_cache_prepare_source\n'
                        'printf "second\\n"\nmfem_cache_release'),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            time.sleep(0.1)
            self.assertIsNone(second.poll(), 'second writer did not wait for shared lock')
            first_output, first_error = first.communicate('release\n', timeout=10)
            self.assertEqual(first.returncode, 0, first_error)
            second_output, second_error = second.communicate(timeout=10)
            self.assertEqual(second.returncode, 0, second_error)
            self.assertEqual(second_output.strip(), 'second')
            self.assertEqual((self.cache / 'source/mfem.hpp').stat().st_mtime_ns, before)
            self.assertEqual(list(self.cache.glob('.source.*')), [])
        finally:
            for process in (first, second):
                if process is not None and process.poll() is None:
                    process.kill()
                    process.communicate()


if __name__ == '__main__':
    unittest.main()
