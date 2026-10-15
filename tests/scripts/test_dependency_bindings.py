#!/usr/bin/env python3
"""Test dependency admission with host fixtures, without building or running apps."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[2]
CHECKER = SOURCE / 'scripts/build/check_dependency_bindings.py'
spec = importlib.util.spec_from_file_location('dependency_bindings', CHECKER)
bindings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bindings)


class DependencyBindings(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.hypre = self.root / 'active HYPRE'
        self.stale = self.root / 'stale HYPRE'
        self.mpi = self.root / 'active MPI'
        self.stale_mpi = self.root / 'stale MPI'
        for prefix in (self.hypre, self.stale):
            (prefix / 'include').mkdir(parents=True)
            (prefix / 'lib').mkdir()
            for name in bindings.HYPRE_HEADERS:
                (prefix / 'include' / name).write_text('fixture\n')
            for name in ('libHYPRE.so', 'libHYPRE.a'):
                (prefix / 'lib' / name).write_text('fixture\n')
        for prefix in (self.mpi, self.stale_mpi):
            (prefix / 'bin').mkdir(parents=True)
            (prefix / 'lib').mkdir()
            (prefix / 'bin/mpicxx').write_text('fixture\n')
            (prefix / 'lib/libmpi.so').write_text('fixture\n')
        self.cache = {
            'HYPRE_DIR': str(self.hypre),
            'HYPRE_INCLUDE_DIR': str(self.hypre / 'include'),
            'HYPRE_INCLUDE_DIRS': str(self.hypre / 'include'),
            'HYPRE_LIBRARY': str(self.hypre / 'lib/libHYPRE.so'),
            'HYPRE_LIBRARIES': str(self.hypre / 'lib/libHYPRE.so'),
            'CMAKE_CXX_COMPILER': str(self.mpi / 'bin/mpicxx'),
            'MPI_CXX_COMPILER': str(self.mpi / 'bin/mpicxx'),
        }
        self.install = self.root / 'MFEM install'
        (self.install / 'include').mkdir(parents=True)
        cmake = self.install / 'lib/cmake/mfem'
        cmake.mkdir(parents=True)
        self.config, self.targets = cmake / 'MFEMConfig.cmake', cmake / 'MFEMTargets.cmake'
        self.write_exports()

    def tearDown(self):
        self.temporary.cleanup()

    def write_exports(self, header_prefix=None, library_prefix=None, static=False):
        header_prefix = header_prefix or self.hypre
        library_prefix = library_prefix or self.hypre
        library = library_prefix / ('lib/libHYPRE.a' if static else 'lib/libHYPRE.so')
        if static:
            library = f'$<LINK_ONLY:{library}>'
        self.config.write_text(
            f'set(MFEM_INCLUDE_DIRS "${{PACKAGE_PREFIX_DIR}}/include;{header_prefix}/include")\n')
        self.targets.write_text(
            'set_target_properties(mfem PROPERTIES\n'
            f' INTERFACE_INCLUDE_DIRECTORIES "${{_IMPORT_PREFIX}}/include;{header_prefix}/include"\n'
            f' INTERFACE_LINK_LIBRARIES "{library};MPI::MPI_CXX;CUDA::cublas"\n)\n')

    def loader_output(self, hypre=None, mpi=None):
        hypre, mpi = hypre or self.hypre, mpi or self.mpi
        return (f'libHYPRE.so => {hypre}/lib/libHYPRE.so (0x1)\n'
                f'libmpi.so.40 => {mpi}/lib/libmpi.so (0x2)\n'
                'libc.so.6 => /system/lib/libc.so.6 (0x3)\n')

    def assert_cache_rejected(self, cache):
        with self.assertRaises((bindings.BindingError, OSError)):
            bindings.check_cache(cache, self.hypre, self.mpi)

    def test_current_cpu_cache_and_non_site_prefixes(self):
        result = bindings.check_cache(self.cache, self.hypre, self.mpi)
        self.assertEqual(result['HYPRE_LIBRARY'], [str(self.hypre / 'lib/libHYPRE.so')])

    def test_gpu_transitive_include_dirs_and_library_targets_are_valid(self):
        gpu = dict(self.cache)
        for name in ('Umpire', 'CAMP', 'fmt', 'CUDA'):
            directory = self.root / name / 'include'
            directory.mkdir(parents=True)
            gpu['HYPRE_INCLUDE_DIRS'] += ';' + str(directory)
        gpu['HYPRE_LIBRARIES'] += ';camp;fmt::fmt-header-only;CUDA::cusparse;dl'
        result = bindings.check_cache(gpu, self.hypre, self.mpi)
        self.assertEqual(len(result['HYPRE_INCLUDE_DIRS']['transitive_include_dirs']), 4)

    def test_merged_view_include_checks_actual_header_targets(self):
        view = self.root / 'view/include'
        view.mkdir(parents=True)
        for name in bindings.HYPRE_HEADERS:
            (view / name).symlink_to(self.hypre / 'include' / name)
        cache = dict(self.cache)
        cache['HYPRE_INCLUDE_DIRS'] += ';' + str(view)
        bindings.check_cache(cache, self.hypre, self.mpi)
        (view / 'HYPRE.h').unlink()
        (view / 'HYPRE.h').symlink_to(self.stale / 'include/HYPRE.h')
        self.assert_cache_rejected(cache)

    def test_correct_hypre_dir_does_not_admit_stale_discovery_values(self):
        for key in ('HYPRE_INCLUDE_DIR', 'HYPRE_INCLUDE_DIRS',
                    'HYPRE_LIBRARY', 'HYPRE_LIBRARIES'):
            with self.subTest(key=key):
                cache = dict(self.cache)
                suffix = 'include' if 'INCLUDE' in key else 'lib/libHYPRE.so'
                cache[key] = str(self.stale / suffix)
                self.assert_cache_rejected(cache)

    def test_appended_stale_hypre_headers_are_not_transitive_includes(self):
        cache = dict(self.cache)
        cache['HYPRE_INCLUDE_DIRS'] += ';' + str(self.stale / 'include')
        self.assert_cache_rejected(cache)

    def test_header_and_library_symlinks_cannot_escape(self):
        for name in ('include/HYPRE.h', 'lib/libHYPRE.so'):
            with self.subTest(name=name):
                path = self.hypre / name
                path.unlink()
                path.symlink_to(self.stale / name)
                self.assert_cache_rejected(self.cache)
                path.unlink()
                path.write_text('fixture\n')

    def test_missing_hypre_configuration_header_rejected(self):
        (self.hypre / 'include/HYPRE_config.h').unlink()
        self.assert_cache_rejected(self.cache)

    def test_static_hypre_archive_supported(self):
        cache = dict(self.cache)
        cache['HYPRE_LIBRARY'] = cache['HYPRE_LIBRARIES'] = str(self.hypre / 'lib/libHYPRE.a')
        bindings.check_cache(cache, self.hypre, self.mpi)
        self.write_exports(static=True)
        result = bindings.check_installed(self.config, self.targets, self.hypre)
        self.assertEqual(result['target_hypre_libraries'], [str(self.hypre / 'lib/libHYPRE.a')])

    def test_cached_mpi_wrapper_must_match_concrete_prefix(self):
        cache = dict(self.cache)
        cache['MPI_CXX_COMPILER'] = str(self.stale_mpi / 'bin/mpicxx')
        self.assert_cache_rejected(cache)

    def test_installed_config_and_target_interfaces_are_checked(self):
        bindings.check_installed(self.config, self.targets, self.hypre)
        self.write_exports(header_prefix=self.stale)
        with self.assertRaises(bindings.BindingError):
            bindings.check_installed(self.config, self.targets, self.hypre)

    def test_stale_installed_link_interface_rejected_with_valid_cache(self):
        bindings.check_cache(self.cache, self.hypre, self.mpi)
        self.write_exports(library_prefix=self.stale)
        with self.assertRaises(bindings.BindingError):
            bindings.check_installed(self.config, self.targets, self.hypre)

    def test_missing_export_property_rejected(self):
        self.targets.write_text('# no imported properties\n')
        with self.assertRaises(bindings.BindingError):
            bindings.check_installed(self.config, self.targets, self.hypre)

    def test_loader_paths_are_checked_even_when_cache_is_current(self):
        result = bindings.check_loader_output(self.loader_output(), self.hypre, self.mpi)
        self.assertEqual(result['hypre'], [str(self.hypre / 'lib/libHYPRE.so')])
        for output in (self.loader_output(hypre=self.stale),
                       self.loader_output(mpi=self.stale_mpi)):
            with self.subTest(output=output), self.assertRaises(bindings.BindingError):
                bindings.check_loader_output(output, self.hypre, self.mpi)

    def test_unresolved_failed_and_empty_loader_output_rejected(self):
        for output, code in (('libHYPRE.so => not found', 0), ('loader failed', 1), ('', 0)):
            with self.subTest(output=output), self.assertRaises(bindings.BindingError):
                bindings.check_loader_output(output, self.hypre, self.mpi, code)

    def test_static_or_unused_dynamic_dependencies_are_reported(self):
        result = bindings.check_loader_output('not a dynamic executable', self.hypre, self.mpi, 1)
        self.assertTrue(result['static_executable'])
        result = bindings.check_loader_output('libc.so.6 => /system/lib/libc.so.6 (0x3)',
                                              self.hypre, self.mpi)
        self.assertEqual(result['hypre'], [])

    def test_application_discovery_excludes_cmake_probes_libraries_and_scripts(self):
        build = self.root / 'app'
        for name, contents in (('example', b'\x7fELFfixture'),
                               ('CMakeFiles/id/probe', b'\x7fELFfixture'),
                               ('libhelper.so', b'\x7fELFfixture'),
                               ('script', b'#!/bin/sh\n')):
            path = build / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
            path.chmod(0o755)
        self.assertEqual(bindings.app_executables(build), [build / 'example'])

    def test_cli_cache_and_exports_without_running_an_application(self):
        cache_file = self.root / 'CMakeCache.txt'
        cache_file.write_text('\n'.join(f'{key}:STRING={value}' for key, value in self.cache.items()))
        command = [sys.executable, '-B', str(CHECKER), '--hypre-prefix', str(self.hypre),
                   '--mpi-prefix', str(self.mpi), '--mfem-cache', str(cache_file),
                   '--mfem-config', str(self.config), '--mfem-targets', str(self.targets)]
        result = subprocess.run(command, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'DEPENDENCY_BINDINGS_VALIDATED')
        self.cache['HYPRE_LIBRARY'] = str(self.stale / 'lib/libHYPRE.so')
        cache_file.write_text('\n'.join(f'{key}:STRING={value}' for key, value in self.cache.items()))
        result = subprocess.run(command, text=True, capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('escapes selected prefix', result.stderr)


if __name__ == '__main__':
    unittest.main()
