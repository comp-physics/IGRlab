#!/usr/bin/env python3
"""Exercise submission/worker argument handling without Slurm or a GPU."""
import json
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2]


class PaceSubmission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        helper_dir = self.repo / 'scripts/build'
        helper_dir.mkdir(parents=True)
        self.helper = helper_dir / 'pace.sh'
        shutil.copy2(SOURCE / 'scripts/build/pace.sh', self.helper)
        self.fake_bin = self.root / 'bin'
        self.fake_bin.mkdir()
        self.write_executable(self.fake_bin / 'hostname', '#!/bin/sh\necho login-phoenix.pace.gatech.edu\n')
        self.write_executable(self.repo / 'build.sh', '''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['BUILD_CALLS'], 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
''')
        self.write_executable(self.fake_bin / 'sbatch', '''#!/usr/bin/env python3
import json, os, sys
p = os.environ['SBATCH_CALLS']
try:
    count = len(open(p).readlines())
except FileNotFoundError:
    count = 0
with open(p, 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
print(9000 + count)
''')
        self.spack = self.root / 'tool/core'
        (self.spack / 'share/spack').mkdir(parents=True)
        (self.spack / 'share/spack/setup-env.sh').touch()
        (self.root / 'tool/packages/repos/spack_repo/builtin').mkdir(parents=True)
        self.output = self.root / 'outputs'
        self.env = os.environ.copy()
        for key in list(self.env):
            if key.startswith(('SLURM_', 'IGRLAB_', 'DG_IGR_')):
                del self.env[key]
        self.env.update(PATH=str(self.fake_bin) + ':' + self.env['PATH'],
                        BUILD_CALLS=str(self.root / 'build.calls'),
                        SBATCH_CALLS=str(self.root / 'sbatch.calls'))

    def tearDown(self):
        self.temp.cleanup()

    def write_executable(self, path, content):
        path.write_text(content)
        path.chmod(0o755)

    def run_helper(self, *args, env=None):
        return subprocess.run(['bash', str(self.helper), *args], env=env or self.env,
                              text=True, capture_output=True, timeout=10)

    def submit(self, *args):
        return self.run_helper('--site', 'pace', '--submit', '--output-root',
                               str(self.output), '--spack-root', str(self.spack), *args)

    def calls(self, name):
        path = self.root / (name + '.calls')
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_gpu_compiles_on_cpu_then_validates_on_a100_free_queue(self):
        result = self.submit('--a100', '--hdg-runtime')
        self.assertEqual(result.returncode, 0, result.stderr)
        build, check = self.calls('sbatch')
        for call in (build, check):
            self.assertIn('--account=gts-sbryngelson3', call)
            self.assertIn('--qos=embers', call)
        self.assertIn('--ntasks=1', build)
        self.assertIn('--ntasks=4', check)
        self.assertIn('--cpus-per-task=2', check)
        self.assertIn('--partition=cpu-small', build)
        self.assertIn('--time=08:00:00', build)
        self.assertIn('--time=02:00:00', check)
        self.assertFalse(any(arg.startswith('--gres=') for arg in build))
        self.assertIn('--partition=gpu-a100', check)
        self.assertIn('--gres=gpu:a100:1', check)
        self.assertIn('--dependency=afterok:9000', check)
        self.assertIn('--kill-on-invalid-dep=yes', check)

    def test_cpu_validation_and_compile_only(self):
        result = self.submit('--cpu-only')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--partition=cpu-small', self.calls('sbatch')[1])
        (self.root / 'sbatch.calls').unlink()
        result = self.submit('--a100', '--no-tests')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls('sbatch')), 1)

    def test_offline_plan_does_not_submit_or_create_output(self):
        result = self.submit('--a100', '--print-plan')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.calls('sbatch'), [])

    def test_focused_wall_times_are_site_options_and_retained_in_worker_request(self):
        result = self.submit('--a100', '--build-time', '00:30:00',
                             '--validation-time', '00:15:00', '--test-regex', '^focused$')
        self.assertEqual(result.returncode, 0, result.stderr)
        build, check = self.calls('sbatch')
        self.assertIn('--time=00:30:00', build)
        self.assertIn('--time=00:15:00', check)
        # The site wrapper owns these options; its generic offline plan does not.
        for args in self.calls('build'):
            self.assertNotIn('--build-time', args)
            self.assertNotIn('--validation-time', args)
        request = json.loads(Path(build[-1]).read_text())
        for option, wanted in (('--build-time', '00:30:00'), ('--validation-time', '00:15:00')):
            self.assertEqual(request['argv'][request['argv'].index(option) + 1], wanted)
        for call, tasks in ((build, '1'), (check, '4')):
            result = self.run_helper(*call[-3:], env=self.env | {
                'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3', 'SLURM_JOB_QOS': 'embers', 'SLURM_NTASKS': tasks})
            self.assertEqual(result.returncode, 0, result.stderr)
            worker = self.calls('build')[-1]
            self.assertEqual(worker[worker.index('--validation-time') + 1], '00:15:00')

    def test_wall_time_plan_and_help_are_explicit_without_submitting(self):
        result = self.submit('--build-time', '00:00:01', '--validation-time', '00:15:00', '--print-plan')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('build_time=00:00:01 validation_time=00:15:00', result.stdout)
        self.assertEqual(self.calls('sbatch'), [])
        self.assertFalse(self.output.exists())
        result = self.run_helper('--site', 'pace', '--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--build-time HH:MM:SS', result.stdout)
        self.assertIn('--validation-time HH:MM:SS', result.stdout)

    def test_invalid_wall_times_fail_before_plan_or_submission(self):
        for option, value in (('--build-time', '00:00:00'), ('--validation-time', '00:60:00'),
                              ('--validation-time', '00:00:60'), ('--build-time', '8:00:00'),
                              ('--build-time', '08:00:01'), ('--validation-time', '02:00:01'),
                              ('--build-time', '-1:00:00'), ('--validation-time', '00:15:00;false')):
            with self.subTest(option=option, value=value):
                result = self.submit(option, value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(option, result.stderr)
                self.assertEqual(self.calls('build'), [])
                self.assertEqual(self.calls('sbatch'), [])
                self.assertFalse(self.output.exists())
        result = self.submit('--validation-time')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires HH:MM:SS', result.stderr)

    def test_login_node_refuses_compilation_before_discovery(self):
        for allocation in ({}, {'SLURM_JOB_ID': '9000', 'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3',
                                'SLURM_JOB_QOS': 'embers', 'SLURM_NTASKS': '1'}):
            result = self.run_helper('--site', 'pace', '--cpu-only', '--no-tests',
                                     '--output-root', str(self.output), '--spack-root', str(self.spack),
                                     env=self.env | allocation)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('compute', result.stderr)
            self.assertEqual(self.calls('build'), [])
            self.assertFalse(self.output.exists())

    def test_other_gpu_architecture_rejected_before_submission(self):
        result = self.submit('--gpu', '--cuda-arch', '120')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls('sbatch'), [])

    def test_interactive_step_gpu_assignment_is_admitted_without_job_gpu_variable(self):
        self.write_executable(self.fake_bin / 'hostname', '#!/bin/sh\necho atl1-compute.pace.gatech.edu\n')
        allocated = self.env | {'SLURM_JOB_ID': '13963258',
                               'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3', 'SLURM_JOB_QOS': 'embers',
                               'SLURM_JOB_PARTITION': 'gpu-a100', 'SLURM_STEP_GPUS': '5',
                               'CUDA_VISIBLE_DEVICES': '0'}
        args = ('--site', 'pace', '--a100', '--tests-only', '--output-root', str(self.output),
                '--spack-root', str(self.spack))
        for assignment in ({}, {'SLURM_JOB_GPUS': ''}, {'SLURM_JOB_GPUS': '5', 'SLURM_STEP_GPUS': ''}):
            with self.subTest(assignment=assignment):
                result = self.run_helper(*args, env=allocated | assignment)
                # Reach the next independent gate, without mocking site modules
                # or launching a build/application.
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('environment modules are unavailable', result.stderr)
                self.assertNotIn('requires an allocated A100', result.stderr)
        for invalid, message in (({'SLURM_STEP_GPUS': ''}, 'requires an allocated A100'),
                                 ({'SLURM_JOB_PARTITION': 'cpu-small'}, 'requires an allocated A100'),
                                 ({'SLURM_JOB_ACCOUNT': 'unrelated'}, 'free embers'),
                                 ({'SLURM_JOB_QOS': 'inferno'}, 'free embers')):
            with self.subTest(invalid=invalid):
                result = self.run_helper(*args, env=allocated | invalid)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
        self.assertEqual(self.calls('sbatch'), [])
        self.assertEqual(self.calls('build'), [])

    def test_relative_dependency_root_rejected(self):
        result = self.submit('--site-root', 'relative')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls('sbatch'), [])

    def test_submission_rejects_symlinked_site_directory(self):
        elsewhere = self.root / 'elsewhere'
        elsewhere.mkdir()
        self.output.mkdir()
        (self.output / 'site').symlink_to(elsewhere, target_is_directory=True)
        result = self.submit('--a100', '--no-tests')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(elsewhere.iterdir()), [])
        self.assertEqual(self.calls('sbatch'), [])

    def prepare_lock(self):
        # Exercise the real path guard/open logic without needing PACE modules.
        text = self.helper.read_text()
        functions = '\n'.join(re.search(r'^' + name + r'\(\) \{\n.*?^\}', text,
                                         flags=re.M | re.S).group(0)
                              for name in ('pace_prepare_site_directory', 'pace_prepare_dependency_lock'))
        script = ('pace_fail() { echo "$*" >&2; exit 2; };\n' + functions +
                  '\npace_dependency_root=$1; pace_profile=cpu; pace_prepare_dependency_lock\n')
        return subprocess.run(['bash', '-c', script, 'fixture', str(self.output)],
                              text=True, capture_output=True, timeout=10)

    def test_dependency_lock_preserves_contents_and_rejects_symlink(self):
        locks = self.output / 'site/pace/locks'
        locks.mkdir(parents=True)
        lock = locks / 'cpu.lock'
        lock.write_text('preserved metadata')
        result = self.prepare_lock()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(lock.read_text(), 'preserved metadata')
        lock.unlink()
        victim = self.root / 'unrelated'
        victim.write_text('unrelated content')
        lock.symlink_to(victim)
        result = self.prepare_lock()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(victim.read_text(), 'unrelated content')

    def test_worker_preserves_literal_arguments_and_separates_tests(self):
        regex = '^(one|two) $HOME `literal`\nthird$'
        result = self.submit('--a100', '--test-regex', regex, '--smoke', '--clean', '--no-deps')
        self.assertEqual(result.returncode, 0, result.stderr)
        build, check = self.calls('sbatch')
        allocated = self.env | {'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3',
                                'SLURM_JOB_QOS': 'embers', 'SLURM_NTASKS': '1'}
        for call in (build, check):
            allocated['SLURM_NTASKS'] = '1' if call[-2] == 'build' else '4'
            result = self.run_helper(*call[-3:], env=allocated)
            self.assertEqual(result.returncode, 0, result.stderr)
        build_args, check_args = self.calls('build')[-2:]
        self.assertIn('--no-tests', build_args)
        self.assertIn('--clean', build_args)
        self.assertIn('--no-deps', build_args)
        self.assertNotIn('--test-regex', build_args)
        self.assertNotIn('--smoke', build_args)
        self.assertIn(regex, check_args)
        self.assertIn('--smoke', check_args)
        self.assertIn('--no-clean', check_args)
        self.assertIn('--no-deps', check_args)
        self.assertIn('--tests-only', check_args)
        self.assertNotIn('--clean', check_args)

    def test_tests_only_submission_requests_validation_without_a_build_job(self):
        result = self.submit('--a100', '--tests-only', '--test-label', '^numerical-acceptance$')
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls('sbatch')
        self.assertEqual(len(calls), 1)
        self.assertIn('--job-name=IGRlab-check', calls[0])
        self.assertIn('--gres=gpu:a100:1', calls[0])
        self.assertFalse(any(a.startswith('--dependency=') for a in calls[0]))
        result = self.run_helper(*calls[0][-3:], env=self.env | {
            'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3', 'SLURM_JOB_QOS': 'embers', 'SLURM_NTASKS': '4'})
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.calls('build')[-1]
        self.assertIn('--tests-only', argv)
        self.assertEqual(argv[argv.index('--test-label') + 1], '^numerical-acceptance$')

    def test_worker_rejects_paid_queue(self):
        request = self.root / 'request.json'
        request.write_text(json.dumps({'source': str(self.repo.resolve()), 'argv': ['--cpu-only']}))
        allocated = self.env | {'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3',
                                'SLURM_JOB_QOS': 'inferno', 'SLURM_NTASKS': '1'}
        result = self.run_helper('--worker', 'build', str(request), env=allocated)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('free', result.stderr)
        self.assertEqual(self.calls('build'), [])

    def prebuilt_site_fixture(self):
        spec = importlib.util.spec_from_file_location('pace_manifest_fixture', SOURCE / 'scripts/build/pace_manifest.py')
        manifest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(manifest)
        config = manifest.load_config()
        for name in ('compiler', 'cmake', 'metis', 'openblas'):
            config[name]['prefix'] = '/fixture-site/' + name
        config['profiles']['cpu']['mpi_prefix'] = '/fixture-site/openmpi'
        recipes = self.root / 'retained-recipes'
        return manifest, config, recipes

    def admit_prebuilt_site(self, config, recipes):
        text = self.helper.read_text()
        function = re.search(r'^pace_admit_prebuilt_site\(\) \{.*?^\}(?=\n\n# sbatch)', text, re.M | re.S).group(0)
        command = function + '\npace_script_dir="$1"; pace_dependency_root="$2"; pace_profile=cpu; pace_recipe_root="$3"; pace_discovered="$4"; pace_admit_prebuilt_site'
        return subprocess.run(['bash', '-eu', '-c', command, 'fixture',
                               str(SOURCE / 'scripts/build'), str(self.output), str(recipes), json.dumps(config)],
                              env=self.env | {'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True, timeout=10)

    def test_run_only_site_admission_is_read_only_and_missing_state_rejects(self):
        manifest, config, recipes = self.prebuilt_site_fixture()
        result = self.admit_prebuilt_site(config, recipes)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('retained configuration', result.stderr)
        self.assertFalse(self.output.exists())
        paths = manifest.write_environment(self.output, 'cpu', recipes, config)
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        result = self.admit_prebuilt_site(config, recipes)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths})

    def test_run_only_site_drift_is_rejected_without_repair(self):
        manifest, config, recipes = self.prebuilt_site_fixture()
        paths = manifest.write_environment(self.output, 'cpu', recipes, config)
        paths[0].write_text('preserve conflicting manifest')
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        result = self.admit_prebuilt_site(config, recipes)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths})

    def test_slurm_spool_copy_uses_original_checkout(self):
        result = self.submit('--a100', '--no-tests')
        self.assertEqual(result.returncode, 0, result.stderr)
        call = self.calls('sbatch')[0]
        spool = self.root / 'var/spool/slurmd/job9000/slurm_script'
        spool.parent.mkdir(parents=True)
        shutil.copy2(self.helper, spool)
        allocated = self.env | {'SLURM_JOB_ACCOUNT': 'gts-sbryngelson3',
                                'SLURM_JOB_QOS': 'embers', 'SLURM_NTASKS': '1'}
        result = subprocess.run(['bash', str(spool), *call[-3:]], env=allocated,
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--no-tests', self.calls('build')[-1])

    def test_missing_mfem_release_description_fails_before_build(self):
        # Hide only Git's release description; leave the real pinned source and
        # every other identity check intact. No checkout/tag mutation is needed.
        real_git = shutil.which('git')
        if real_git is None or not (SOURCE / 'tpl/mfem/.git').exists():
            self.skipTest('initialized MFEM Git checkout required')
        self.write_executable(self.fake_bin / 'git', f'''#!/usr/bin/env python3
import os, sys
if 'describe' in sys.argv:
    print('heads/commit-only-bundle-0-gd9d6526cc1749980a2ba1da16e2c1ca1e07d82ec')
else:
    os.execv({real_git!r}, [{real_git!r}] + sys.argv[1:])
''')
        result = subprocess.run(['bash', str(SOURCE / 'build.sh'), '--cpu-only',
                                 '--hdg-runtime', '--print-plan', '--output-root',
                                 str(self.output)], env=self.env, text=True,
                                capture_output=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('MFEM v4.9 release tag before building dependencies', result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
