#!/usr/bin/env python3
"""Verify generated test commands without compiling or running executables."""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[2]
PARSER = argparse.ArgumentParser()
PARSER.add_argument('--cmake', default=shutil.which('cmake'))
ARGS, UNITTEST_ARGS = PARSER.parse_known_args()


class MPITestLauncher(unittest.TestCase):
    def setUp(self):
        if not ARGS.cmake:
            self.skipTest('CMake is unavailable')
        version = subprocess.run([ARGS.cmake, '--version'], text=True,
                                 capture_output=True, check=True).stdout.split()[2]
        if tuple(map(int, version.split('.')[:2])) < (3, 29):
            self.skipTest('TEST_LAUNCHER needs CMake 3.29 or newer')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'source'
        self.build = self.root / 'build'
        self.repo.mkdir()
        self.launcher = self.root / 'selected MPI/bin/mpiexec'
        self.launcher.parent.mkdir(parents=True)
        self.launcher.touch()
        self.launcher.chmod(0o755)
        (self.repo / 'dummy.cpp').write_text('int main() { return 0; }\n')
        child = self.repo / 'nested'
        child.mkdir()
        (child / 'CMakeLists.txt').write_text('''add_executable(nested_mpi ../dummy.cpp)
set_property(TARGET nested_mpi PROPERTY LINKER_LANGUAGE CXX)
target_link_libraries(nested_mpi PRIVATE component)
add_test(NAME nested COMMAND nested_mpi --child)
''')
        self.env = os.environ.copy()
        self.env.pop('CMAKE_TEST_LAUNCHER', None)
        serial_args = re.search(r'^set\(_igrlab_serial_mpi_args ""\)\n.*?^endif\(\)',
                                (SOURCE / 'tests/CMakeLists.txt').read_text(), re.M | re.S).group(0)
        # No compiler is enabled. These rules let CMake generate regular
        # executable target metadata; no build command is ever invoked.
        self.content = '''cmake_minimum_required(VERSION 3.29)
project(launcher_fixture LANGUAGES NONE)
enable_testing()
set(CMAKE_CXX_LINK_EXECUTABLE "unused-link-rule")
set(MFEM_USE_MPI ON)
set(MPIEXEC_EXECUTABLE "@LAUNCHER@")
set(MPIEXEC_NUMPROC_FLAG -n)
set(MPIEXEC_PREFLAGS --fixture-pref)
include("@HELPER@")
@SERIAL_ARGS@
add_library(mfem INTERFACE)
add_library(MPI::MPI_CXX INTERFACE IMPORTED)
add_library(component INTERFACE)
target_link_libraries(component INTERFACE "$<LINK_ONLY:mfem>")
foreach(target dg_igr_unit_tests dg_igr_modal_damping_test dg_igr_explicit_ssprk_test pure)
  add_executable(${target} dummy.cpp)
  set_property(TARGET ${target} PROPERTY LINKER_LANGUAGE CXX)
endforeach()
target_link_libraries(dg_igr_unit_tests PRIVATE mfem)
target_link_libraries(dg_igr_modal_damping_test PRIVATE MPI::MPI_CXX)
target_link_libraries(dg_igr_explicit_ssprk_test PRIVATE component)
add_test(NAME unit COMMAND dg_igr_unit_tests --dg-operator)
add_test(NAME modal COMMAND dg_igr_modal_damping_test cpu)
add_test(NAME ssprk COMMAND dg_igr_explicit_ssprk_test cpu)
add_test(NAME pure COMMAND pure)
add_test(NAME explicit_two COMMAND "${MPIEXEC_EXECUTABLE}" -n 2 "$<TARGET_FILE:dg_igr_unit_tests>" --mpi)
add_test(NAME explicit_four COMMAND "${MPIEXEC_EXECUTABLE}" -n 4 "$<TARGET_FILE:dg_igr_unit_tests>" --mpi)
add_test(NAME shell COMMAND /bin/sh -c "echo shell")
add_test(NAME python COMMAND /usr/bin/env python3 -c "print('python')")
add_test(NAME target_file_script COMMAND /bin/sh "$<TARGET_FILE:dg_igr_unit_tests>")
add_test(NAME environment COMMAND dg_igr_unit_tests --dgsem-operator)
set_tests_properties(environment PROPERTIES ENVIRONMENT "DG_IGR_TEST_DEVICE=cuda;DG_IGR_DGSEM_DEVICE=cuda")
add_test(NAME explicit_environment_two COMMAND "${CMAKE_COMMAND}" -E env FIXTURE=two "${MPIEXEC_EXECUTABLE}" -n 2 "$<TARGET_FILE:dg_igr_unit_tests>" --mpi)
add_test(NAME vtk_script COMMAND /usr/bin/env python3 /fixture/check_vtk_restart.py ${_igrlab_serial_mpi_args} --exe "$<TARGET_FILE:dg_igr_unit_tests>" --device cpu)
add_subdirectory(nested)
@EXTRA@
igrlab_configure_mpi_test_launchers()
'''.replace('@LAUNCHER@', self.launcher.as_posix()).replace(
            '@HELPER@', (SOURCE / 'cmake/IGRlabMPITestLauncher.cmake').as_posix()).replace(
            '@SERIAL_ARGS@', serial_args)

    def configure(self, enabled=True, extra=''):
        (self.repo / 'CMakeLists.txt').write_text(self.content.replace('@EXTRA@', extra))
        return subprocess.run([ARGS.cmake, '-S', str(self.repo), '-B', str(self.build),
                               '-DIGRLAB_MPI_TEST_LAUNCHER:BOOL=' + ('ON' if enabled else 'OFF')],
                              env=self.env, text=True, capture_output=True, timeout=15)

    def commands(self):
        # Existing placeholder files are enough for CTest's show-only mode.
        for target in ('dg_igr_unit_tests', 'dg_igr_modal_damping_test',
                       'dg_igr_explicit_ssprk_test', 'pure'):
            (self.build / target).touch()
        (self.build / 'nested/nested_mpi').touch()
        ctest = Path(ARGS.cmake).resolve().with_name('ctest')
        result = subprocess.run([str(ctest), '--test-dir', str(self.build),
                                 '--show-only=json-v1'], env=self.env, text=True,
                                capture_output=True, check=True, timeout=10)
        self.generated_tests = {test['name']: test for test in json.loads(result.stdout)['tests']}
        return {name: test['command'] for name, test in self.generated_tests.items()}

    def test_mpi_direct_transitive_and_pure_commands(self):
        result = self.configure()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands()
        for name, target, argument in (('unit', 'dg_igr_unit_tests', '--dg-operator'),
                                       ('modal', 'dg_igr_modal_damping_test', 'cpu'),
                                       ('ssprk', 'dg_igr_explicit_ssprk_test', 'cpu')):
            self.assertEqual(commands[name], [str(self.launcher), '-n', '1', '--fixture-pref',
                                               str(self.build / target), argument])
        self.assertEqual(commands['pure'], [str(self.build / 'pure')])
        self.assertEqual(commands['nested'], [str(self.launcher), '-n', '1', '--fixture-pref',
                                              str(self.build / 'nested/nested_mpi'), '--child'])

    def test_explicit_multi_rank_and_script_commands_are_unchanged(self):
        result = self.configure()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands()
        for name, rank in (('explicit_two', '2'), ('explicit_four', '4')):
            self.assertEqual(commands[name], [str(self.launcher), '-n', rank,
                                               str(self.build / 'dg_igr_unit_tests'), '--mpi'])
        self.assertEqual(commands['shell'], ['/bin/sh', '-c', 'echo shell'])
        self.assertEqual(commands['python'], ['/usr/bin/env', 'python3', '-c', "print('python')"])
        self.assertEqual(commands['target_file_script'], ['/bin/sh', str(self.build / 'dg_igr_unit_tests')])
        self.assertEqual(commands['explicit_environment_two'], [str(Path(ARGS.cmake).resolve()), '-E',
                          'env', 'FIXTURE=two', str(self.launcher), '-n', '2',
                          str(self.build / 'dg_igr_unit_tests'), '--mpi'])

    def test_environment_property_survives_rank_one_launcher(self):
        result = self.configure()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands()
        self.assertEqual(commands['environment'], [str(self.launcher), '-n', '1', '--fixture-pref',
                                                  str(self.build / 'dg_igr_unit_tests'), '--dgsem-operator'])
        properties = {p['name']: p['value'] for p in self.generated_tests['environment']['properties']}
        self.assertEqual(properties['ENVIRONMENT'], ['DG_IGR_TEST_DEVICE=cuda', 'DG_IGR_DGSEM_DEVICE=cuda'])

    def test_repeated_configuration_is_idempotent(self):
        result = self.configure(extra='igrlab_configure_mpi_test_launchers()')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands()['unit'].count(str(self.launcher)), 1)

    def test_non_pace_default_keeps_direct_commands(self):
        result = self.configure(enabled=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands()['unit'], [str(self.build / 'dg_igr_unit_tests'), '--dg-operator'])

    def test_pace_only_script_arguments_in_generated_command(self):
        prefix = ['/usr/bin/env', 'python3', '/fixture/check_vtk_restart.py']
        suffix = ['--exe', str(self.build / 'dg_igr_unit_tests'), '--device', 'cpu']
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                result = self.configure(enabled=enabled)
                self.assertEqual(result.returncode, 0, result.stderr)
                mpi_args = ['--mpi-single-rank', '--mpi-ranks', '1', '--mpiexec',
                            str(self.launcher), '--mpi-numproc-flag=-n'] if enabled else []
                self.assertEqual(self.commands()['vtk_script'], prefix + mpi_args + suffix)

    def test_missing_launcher_rejects_before_generation(self):
        self.launcher.unlink()
        result = self.configure()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('selected MPIEXEC_EXECUTABLE', result.stderr)

    def test_mpi_disabled_rejects(self):
        result = self.configure(extra='set(MFEM_USE_MPI OFF)')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('MPI-enabled MFEM', result.stderr)

    def test_missing_numproc_flag_rejects(self):
        result = self.configure(extra='set(MPIEXEC_NUMPROC_FLAG "")')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('selected MPIEXEC_EXECUTABLE', result.stderr)

    def test_conflicting_custom_launcher_rejects(self):
        result = self.configure(extra='set_property(TARGET dg_igr_unit_tests PROPERTY TEST_LAUNCHER custom)')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('conflicts with TEST_LAUNCHER', result.stderr)

    def test_unsupported_postflags_reject_instead_of_being_ignored(self):
        result = self.configure(extra='set(MPIEXEC_POSTFLAGS --post)')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('empty MPIEXEC_POSTFLAGS', result.stderr)


class ScriptSingleRankLauncher(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def capture(self, script, device, ranks, single_rank):
        filename = SOURCE / 'tests/scripts' / (script + '.py')
        spec = importlib.util.spec_from_file_location('launcher_fixture_' + script, filename)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, 'path', [str(filename.parent), *sys.path]):
            spec.loader.exec_module(module)
        argv = [str(filename), '--exe', str(self.root / 'application'), '--device', device,
                '--out-dir', str(self.root / script), '--mpi-ranks', str(ranks),
                '--mpiexec', '/selected MPI/bin/mpiexec', '--mpi-numproc-flag=-n']
        if script == 'check_dgsem_vtk_restart':
            argv.extend(['--order', '3', '--extrapolated-llf'])
        elif script == 'check_vtk_restart':
            argv.extend(['--order', '3', '--time-integrator', 'ssprk104', '--zhang-shu-vtk-points'])
        else:
            argv.extend(['--igr-stage-rtol-scale', '7'])
        if single_rank:
            argv.append('--mpi-single-rank')

        class Captured(Exception):
            pass

        captured = []

        def intercept(command, *args, **kwargs):
            command = list(command)
            # DGSEM's unique attempt directory changes only this mesh path.
            if '-m' in command:
                mesh_index = command.index('-m') + 1
                self.assertTrue(Path(command[mesh_index]).is_file())
                command[mesh_index] = '<generated fixture mesh>'
            captured.append(command)
            raise Captured()

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, 'argv', argv))
            stack.enter_context(mock.patch.object(module.subprocess, 'run',
                                side_effect=AssertionError('a fixture tried to launch a process')))
            if script == 'check_dgsem_vtk_restart':
                stack.enter_context(mock.patch.object(module.Lease, 'acquire', return_value=True))
                stack.enter_context(mock.patch.object(module.Lease, 'release'))
                stack.enter_context(mock.patch.object(module.Runner, 'run', side_effect=intercept))
            else:
                function = 'run' if script == 'check_vtk_restart' else 'run_case'
                stack.enter_context(mock.patch.object(module, function, side_effect=intercept))
            with self.assertRaises(Captured):
                module.main()
        self.assertEqual(len(captured), 1)
        return captured[0]

    def test_three_script_families_preserve_default_and_physics_arguments(self):
        for script in ('check_vtk_restart', 'check_sigma_vtk_lifecycle', 'check_dgsem_vtk_restart'):
            for device in ('cpu', 'cuda'):
                with self.subTest(script=script, device=device):
                    default = self.capture(script, device, 1, False)
                    launched = self.capture(script, device, 1, True)
                    self.assertEqual(default[0], str(self.root / 'application'))
                    self.assertEqual(launched[:3], ['/selected MPI/bin/mpiexec', '-n', '1'])
                    self.assertEqual(launched[3:], default)

    def test_existing_multi_rank_launches_are_not_nested(self):
        for script in ('check_vtk_restart', 'check_sigma_vtk_lifecycle', 'check_dgsem_vtk_restart'):
            for device in ('cpu', 'cuda'):
                with self.subTest(script=script, device=device):
                    before = self.capture(script, device, 2, False)
                    opted_in = self.capture(script, device, 2, True)
                    self.assertEqual(opted_in, before)
                    self.assertEqual(before[:3], ['/selected MPI/bin/mpiexec', '-n', '2'])
                    self.assertEqual(before.count('/selected MPI/bin/mpiexec'), 1)


if __name__ == '__main__':
    unittest.main(argv=[__file__, *UNITTEST_ARGS])
