#!/usr/bin/env python3
"""Compiler-free Delta policy, submission, and scheduler-admission contracts.

All scheduler/tool replies below are intercepted fixtures, not qualification
evidence. Only public --print-plan calls may use the actual build driver.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[2]
WRAPPER = SOURCE / "scripts/build/delta.sh"


def closed_environment():
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("SLURM_", "IGRLAB_", "DG_IGR_", "SPACK_", "OMPI_", "PMIX_")):
            del env[key]
    return env


class DeltaSubmission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        helper_dir = self.repo / "scripts/build"
        helper_dir.mkdir(parents=True)
        self.helper = helper_dir / "delta.sh"
        shutil.copy2(WRAPPER, self.helper)
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        self.executable(self.fake_bin / "hostname", "#!/bin/sh\necho login.delta.ncsa.illinois.edu\n")
        self.executable(self.repo / "build.sh", '''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['BUILD_CALLS'], 'a') as f:
    f.write(json.dumps({'argv': sys.argv[1:], 'env': {
        k: os.environ.get(k) for k in ('IGRLAB_BUILD_SITE', 'IGRLAB_CUDA_ARCH',
        'IGRLAB_MFEM_CUDA_ARCHS', 'IGRLAB_APP_CUDA_ARCHS', 'IGRLAB_SPACK_ENV')}}) + '\\n')
''')
        self.executable(self.fake_bin / "sbatch", '''#!/usr/bin/env python3
import json, os, sys
p = os.environ['SBATCH_CALLS']
try:
    count = len(open(p).readlines())
except FileNotFoundError:
    count = 0
with open(p, 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
print(os.environ.get('SBATCH_REPLY', str(7000 + count)))
''')
        self.spack = self.root / "spack"
        (self.spack / "share/spack").mkdir(parents=True)
        (self.spack / "share/spack/setup-env.sh").touch()
        self.output = self.root / "outputs"
        self.env = closed_environment()
        self.env.update(PATH=str(self.fake_bin) + ":" + self.env["PATH"],
                        BUILD_CALLS=str(self.root / "build.calls"),
                        SBATCH_CALLS=str(self.root / "sbatch.calls"))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def executable(path, text):
        path.write_text(text)
        path.chmod(0o755)

    def run_wrapper(self, *args, env=None):
        return subprocess.run(["bash", str(self.helper), *args],
                              env=self.env if env is None else env,
                              capture_output=True, text=True, timeout=10)

    def submit(self, *args, env=None):
        return self.run_wrapper("--site", "delta", "--submit", "--output-root",
                                str(self.output), "--spack-root", str(self.spack),
                                *args, env=env)

    def calls(self, name):
        p = self.root / (name + ".calls")
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def test_gpu_build_uses_cpu_account_then_separate_afterok_gpu_validation(self):
        result = self.submit("--a100", "--hdg-runtime", "--nvtx-profile")
        self.assertEqual(result.returncode, 0, result.stderr)
        build, validation = self.calls("sbatch")
        self.assertIn("--account=bdiy-delta-cpu", build)
        self.assertIn("--partition=cpu", build)
        self.assertIn("--ntasks=1", build)
        self.assertIn("--cpus-per-task=8", build)
        self.assertFalse(any(x.startswith(("--gres", "--gpus")) for x in build))
        self.assertIn("--account=bdiy-delta-gpu", validation)
        self.assertIn("--partition=gpuA100x4", validation)
        self.assertIn("--gpus-per-node=1", validation)
        self.assertIn("--ntasks=4", validation)
        self.assertIn("--cpus-per-task=2", validation)
        self.assertIn("--dependency=afterok:7000", validation)
        self.assertIn("--kill-on-invalid-dep=yes", validation)
        for call in (build, validation):
            self.assertIn("--mem=48G", call)
            self.assertIn("--time=01:00:00", call)
            self.assertFalse(any(x.startswith("--qos") for x in call))
        request = json.loads(Path(build[-1]).read_text())
        self.assertEqual(request["source"], str(self.repo))
        self.assertIn("--site", request["argv"])
        self.assertIn("delta", request["argv"])

    def test_cpu_build_and_validation_remain_cpu_and_build_only_has_no_followup(self):
        result = self.submit("--cpu-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        for call in self.calls("sbatch"):
            self.assertIn("--account=bdiy-delta-cpu", call)
            self.assertIn("--partition=cpu", call)
            self.assertFalse(any(x.startswith(("--gres", "--gpus")) for x in call))
        (self.root / "sbatch.calls").unlink()
        result = self.submit("--a100", "--no-tests")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls("sbatch")), 1)

    def test_tests_only_has_no_build_or_artificial_dependency(self):
        result = self.submit("--a100", "--tests-only", "--test-regex", "^selected$")
        self.assertEqual(result.returncode, 0, result.stderr)
        (call,) = self.calls("sbatch")
        self.assertIn("--partition=gpuA100x4", call)
        self.assertFalse(any(x.startswith("--dependency") for x in call))
        self.assertEqual(call[-2], "validation")

    def test_workers_split_compilation_from_tests_and_keep_filters_only_for_validation(self):
        result = self.submit("--a100", "--test-regex", "^selected$",
                             "--test-label", "numerical-smoke", "--no-clean")
        self.assertEqual(result.returncode, 0, result.stderr)
        build, validation = self.calls("sbatch")
        for call, tasks in ((build, "1"), (validation, "4")):
            result = self.run_wrapper(*call[-3:], env=self.env | {"SLURM_NTASKS": tasks})
            self.assertEqual(result.returncode, 0, result.stderr)
            args = self.calls("build")[-1]["argv"]
            if tasks == "1":
                self.assertEqual(args.count("--no-tests"), 1)
                self.assertNotIn("--tests-only", args)
                self.assertNotIn("--test-regex", args)
                self.assertNotIn("--test-label", args)
            else:
                self.assertEqual(args.count("--tests-only"), 1)
                self.assertNotIn("--no-tests", args)
                self.assertIn("--no-deps", args)
                self.assertEqual(args[args.index("--test-regex") + 1], "^selected$")
                self.assertEqual(args[args.index("--test-label") + 1], "numerical-smoke")

    def test_existing_tests_only_worker_does_not_duplicate_mode(self):
        result = self.submit("--cpu-only", "--tests-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        (call,) = self.calls("sbatch")
        result = self.run_wrapper(*call[-3:], env=self.env | {"SLURM_NTASKS": "4"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls("build")[-1]["argv"].count("--tests-only"), 1)

    def test_offline_plan_creates_no_outputs_or_scheduler_calls_and_pins_native_arch(self):
        result = self.submit("--gpu", "--cuda-arch", "80", "--print-plan")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("site=delta profile=cuda-sm80", result.stdout)
        self.assertIn("validation_account=bdiy-delta-gpu", result.stdout)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.calls("sbatch"), [])
        call = self.calls("build")[-1]
        self.assertIn("--print-plan", call["argv"])
        for key in ("IGRLAB_CUDA_ARCH", "IGRLAB_MFEM_CUDA_ARCHS", "IGRLAB_APP_CUDA_ARCHS"):
            self.assertEqual(call["env"][key], "80")
        self.assertEqual(call["env"]["IGRLAB_BUILD_SITE"], "delta")
        self.assertEqual(call["env"]["IGRLAB_SPACK_ENV"], str(self.output / "site/delta/spack-envs/cuda-sm80"))

    def test_multihour_wall_limits_are_retained_without_arbitrary_cap(self):
        result = self.submit("--a100", "--build-time", "03:00:00", "--validation-time", "02:15:00")
        self.assertEqual(result.returncode, 0, result.stderr)
        build, validation = self.calls("sbatch")
        self.assertIn("--time=03:00:00", build)
        self.assertIn("--time=02:15:00", validation)
        request = json.loads(Path(build[-1]).read_text())["argv"]
        self.assertEqual(request[request.index("--validation-time") + 1], "02:15:00")
        self.assertNotIn("--validation-time", self.calls("build")[0]["argv"])

    def test_invalid_modes_and_wall_times_fail_before_plan_or_submission(self):
        cases = [("--a100", "--cpu-only"), ("--cpu-only", "--a100"),
                 ("--gpu", "--cuda-arch", "70"), ("--gpu", "--cuda-arch", "90"),
                 ("--cpu-only", "--cuda-arch", "80"), ("--tests-only", "--no-tests"),
                 ("--build-time", "00:00:00"), ("--validation-time", "00:60:00"),
                 ("--build-time", "1:00:00"), ("--build-time", "01:00:00;false")]
        for args in cases:
            with self.subTest(args=args):
                result = self.submit(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.calls("build"), [])
                self.assertEqual(self.calls("sbatch"), [])
                self.assertFalse(self.output.exists())

    def test_nested_submission_and_invalid_scheduler_ack_stop_without_followup(self):
        result = self.submit("--a100", env=self.env | {"SLURM_JOB_ID": "7010"})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls("sbatch"), [])
        result = self.submit("--a100", env=self.env | {"SBATCH_REPLY": "not-a-job"})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.calls("sbatch")), 1)

    def test_direct_login_build_rejects_before_discovery_or_build(self):
        result = self.run_wrapper("--site", "delta", "--a100", "--no-tests",
                                  "--output-root", str(self.output), "--spack-root", str(self.spack))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("real compute allocation", result.stderr)
        self.assertEqual(self.calls("build"), [])
        self.assertFalse(self.output.exists())


class DeltaComputeAdmission(unittest.TestCase):
    """Execute the wrapper's actual small guard with intercepted scheduler I/O."""
    def run_guard(self, *, job_changes=None, environment_changes=None, hostname="cn001.delta.ncsa.illinois.edu"):
        source = WRAPPER.read_text()
        match = re.search(r"delta_admit_compute\(\).*?<<'PYCODE'\n(.*?)\nPYCODE", source, re.S)
        self.assertIsNotNone(match, "wrapper compute guard must remain discoverable")
        job = {"JobId": "7001", "JobState": "RUNNING",
               "UserId": f"{pwd.getpwuid(os.getuid()).pw_name}({os.getuid()})",
               "Account": "bdiy-delta-cpu", "Partition": "cpu",
               "AllocTRES": "cpu=8,mem=48G,node=1", "ReqTRES": "cpu=8,mem=48G,node=1"}
        job.update(job_changes or {})
        env = {"SLURM_JOB_ID": "7001", "SLURM_JOB_ACCOUNT": "bdiy-delta-cpu", "SLURM_JOB_PARTITION": "cpu"}
        env.update(environment_changes or {})
        reply = SimpleNamespace(returncode=0, stdout=" ".join(f"{k}={v}" for k, v in job.items()))
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(sys, "argv", ["guard", "bdiy-delta-cpu", "cpu", "yes"]), \
             mock.patch("socket.getfqdn", return_value=hostname), \
             mock.patch("subprocess.run", return_value=reply) as query:
            exec(compile(match.group(1), str(WRAPPER) + ":compute_guard", "exec"), {})
            query.assert_called_once()
            self.assertEqual(query.call_args.args[0], ["scontrol", "show", "job", "-o", "7001"])

    def test_owned_cpu_compute_is_admitted_without_module_or_application_execution(self):
        self.run_guard()

    def test_gpu_tres_on_cpu_partition_cannot_compile(self):
        for key in ("AllocTRES", "ReqTRES"):
            with self.subTest(key=key), self.assertRaisesRegex(SystemExit, "without GPUs"):
                self.run_guard(job_changes={key: "cpu=8,mem=48G,node=1,gres/gpu=1"})

    def test_wrong_owner_state_account_partition_and_login_reject(self):
        cases = [{"UserId": "someoneelse(999999)"}, {"JobState": "PENDING"},
                 {"JobId": "7010"}, {"Account": "unselected"}, {"Partition": "gpuA100x4"}]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(SystemExit):
                self.run_guard(job_changes=change)
        with self.assertRaisesRegex(SystemExit, "real compute allocation"):
            self.run_guard(hostname="login.delta.ncsa.illinois.edu")
        with self.assertRaises(SystemExit):
            self.run_guard(environment_changes={"SLURM_JOB_ACCOUNT": "unselected"})


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / f"scripts/build/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeltaDiscovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.helper = load_helper("delta_discover")
        self.config = json.loads((SOURCE / "cmake/sites/delta.json").read_text())
        self.tools = {}
        for family, names in (("gcc", ("gcc", "g++", "gfortran")),
                              ("cmake", ("cmake",)),
                              ("mpi", ("mpicc", "mpicxx", "mpifort", "ompi_info")),
                              ("cuda", ("nvcc",))):
            directory = self.root / family / "bin"
            directory.mkdir(parents=True)
            for name in names:
                path = directory / name
                path.write_text("#!/bin/sh\nexit 99 # must never execute in these fixtures\n")
                path.chmod(0o755)
                self.tools[name] = path
        (self.root / "cuda/include").mkdir()
        (self.root / "cuda/include/cuda.h").touch()
        (self.root / "cuda/lib64").mkdir()
        (self.root / "cuda/version.json").write_text(json.dumps({
            "cuda": {"version": self.config["cuda_version"]}}))
        blas_metadata = self.root / "openblas/.spack/spec.json"
        blas_metadata.parent.mkdir(parents=True)
        blas_metadata.write_text(json.dumps({"spec": {"nodes": [{
            "name": "openblas", "version": self.config["openblas"]["version"],
            "arch": {"target": self.config["openblas"]["target"]},
            "parameters": copy.deepcopy(self.config["openblas"]["parameters"]),
            "compiler": {"name": "gcc", "version": "observed-system-build"}}]}}))
        self.queries = []
        self.reply_changes = {}

    def tearDown(self):
        self.temp.cleanup()

    def fake_query(self, command, **kwargs):
        self.queries.append(command)
        name = Path(command[0]).name
        if name in ("gcc", "g++", "gfortran"):
            answer = "14.2.1"
        elif name == "cmake":
            answer = "cmake version 3.31.8"
        elif name == "nvcc":
            answer = "Cuda compilation tools, release 12.9, V12.9.86"
        elif name == "ompi_info" and "--version" in command:
            answer = "Open MPI v4.1.9a1"
        elif name == "ompi_info" and "--path" in command:
            answer = "path:prefix:" + str(self.root / "mpi")
        elif name == "ompi_info":
            answer = "options:mpi_built_with_cuda_support:value:true\nbuild:compiler:unknown-system-build"
        elif name in ("mpicc", "mpicxx", "mpifort"):
            answer = {"mpicc": "gcc", "mpicxx": "g++", "mpifort": "gfortran"}[name]
        else:
            self.fail("unexpected discovery operation: " + repr(command))
        answer = self.reply_changes.get((name, tuple(command[1:])), answer)
        return SimpleNamespace(returncode=0, stdout=answer, stderr="")

    def discover(self, profile):
        return self.helper.discover_config(
            profile, anchor=self.tools["gcc"], cmake=self.tools["cmake"],
            cuda=self.tools["nvcc"], mpi=self.tools["mpicxx"],
            environ={"PATH": str(self.root / "gcc/bin"),
                     "OPENBLAS_HOME": str(self.root / "openblas")}, run=self.fake_query)

    def test_external_system_mpi_cuda_capability_is_preserved_for_both_profiles(self):
        for profile in ("cpu", "cuda-sm80"):
            with self.subTest(profile=profile):
                config = self.discover(profile)
                selected = config["profiles"][profile]
                self.assertEqual(selected["mpi_version"], "4.1.9a1")
                self.assertEqual(selected["mpi_parameters"], {"cuda": True})
                self.assertEqual(selected["mpi_prefix"], str(self.root / "mpi"))
                self.assertEqual(selected["mpi_wrapper_commands"], {
                    "mpicc": "gcc", "mpicxx": "g++", "mpifort": "gfortran"})
                self.assertEqual(config["cuda_prefix"], str(self.root / "cuda"))
                self.assertEqual(config["openblas"]["prefix"], str(self.root / "openblas"))
                self.assertEqual(config["openblas"]["binary_build_provenance"]["compiler"]["version"],
                                 "observed-system-build")
                self.assertEqual(len(config["cuda_version_file_sha256"]), 64)
                self.assertIn("unknown-system-build", selected["mpi_build_provenance"][0])
                self.assertEqual(len(selected["mpi_metadata_sha256"]), 64)
        self.assertFalse(any(Path(c[0]).name == "spack" for c in self.queries))

    def test_wrong_versions_wrappers_and_unavailable_cuda_capability_reject(self):
        cases = [("gcc", ("-dumpfullversion", "-dumpversion"), "12.2.0"),
                 ("cmake", ("--version",), "cmake version 3.31.9"),
                 ("ompi_info", ("--version",), "Open MPI v5.0.9"),
                 ("ompi_info", ("--parsable", "--all"), "options:mpi_built_with_cuda_support:value:false"),
                 ("mpicxx", ("--showme:command",), "nvcc"),
                 ("mpicc", ("--showme:command",), "gcc -fextra"),
                 ("nvcc", ("--version",), "release 12.9, V12.9.99")]
        for name, args, response in cases:
            with self.subTest(name=name, args=args):
                self.reply_changes = {(name, args): response}
                with self.assertRaises(self.helper.DiscoveryError):
                    self.discover("cpu")

    def test_system_mpi_symlink_retains_wrapper_invocation_name(self):
        target = self.root / "mpi/bin/opal-wrapper"
        target.write_text(self.tools["mpicxx"].read_text())
        target.chmod(0o755)
        self.tools["mpicxx"].unlink()
        self.tools["mpicxx"].symlink_to(target.name)
        config = self.discover("cpu")
        self.assertEqual(config["profiles"]["cpu"]["mpi_wrapper"], str(self.tools["mpicxx"]))
        manifest = load_helper("delta_manifest")
        self.assertEqual(manifest.resolved_config(config, "cpu"), config)

    def test_recipe_authority_is_reused_without_spack_query_or_file_rewrite(self):
        recipes = self.root / "recipes"
        (recipes / "packages").mkdir(parents=True)
        (recipes / "repo.yaml").write_text("repo: {namespace: builtin}\n")
        site = self.root / "output/site/delta"
        retained = site / "spack-envs/cpu/spack.yaml"
        retained.parent.mkdir(parents=True)
        retained.write_text(json.dumps({"spack": {"repos": {"builtin": str(recipes)}}}))
        before = (retained.read_bytes(), retained.stat().st_mtime_ns)
        with mock.patch.object(self.helper, "query", side_effect=AssertionError("Spack query on retained authority")):
            chosen = self.helper.discover_recipe_root(None, self.root / "output", "cuda-sm80")
        self.assertEqual(chosen, str(recipes))
        self.assertEqual(before, (retained.read_bytes(), retained.stat().st_mtime_ns))
        changed = self.root / "different-recipes"
        (changed / "packages").mkdir(parents=True)
        (changed / "repo.yaml").touch()
        with self.assertRaisesRegex(self.helper.DiscoveryError, "changes retained recipes"):
            self.helper.discover_recipe_root(None, self.root / "output", "cpu", override=changed)
        self.assertEqual(before, (retained.read_bytes(), retained.stat().st_mtime_ns))


def observed_config(profile):
    """Synthetic observations using the checked-in version/module policy."""
    helper = load_helper("delta_manifest")
    config = helper.load_config()
    config["compiler"].update(prefix="/observed/gcc", executables={
        "c": "/observed/gcc/bin/gcc", "cxx": "/observed/gcc/bin/g++",
        "fortran": "/observed/gcc/bin/gfortran"}, sha256={k: "a" * 64 for k in ("c", "cxx", "fortran")})
    config["cmake"].update(prefix="/observed/cmake", executable="/observed/cmake/bin/cmake", sha256="b" * 64)
    config["profiles"][profile].update(mpi_prefix="/observed/hpcx", mpi_wrapper="/observed/hpcx/bin/mpicxx",
        mpi_wrapper_commands={"mpicc": "gcc", "mpicxx": "g++", "mpifort": "gfortran"},
        mpi_metadata_sha256="c" * 64, mpi_build_provenance=["build:compiler:observed-system-metadata"])
    config.update(cuda_prefix="/observed/cuda", cuda_executable="/observed/cuda/bin/nvcc", cuda_sha256="d" * 64)
    config["cuda_version_file_sha256"] = "e" * 64
    config["openblas"].update(prefix="/observed/openblas", metadata_sha256="f" * 64,
        binary_build_provenance={"compiler": {"name": "gcc", "version": "observed-system-build"}})
    return config


def concrete_lock(profile, config):
    nodes, ids = {}, {}

    def add(name, version, parameters=None, external=None, dependencies=(), root=True):
        identity = str(len(nodes) + 1).zfill(32)
        ids[name] = identity
        nodes[identity] = {"name": name, "version": version,
            "arch": {"target": config["target"]}, "parameters": parameters or {},
            "dependencies": [{"name": d, "hash": ids[d]} for d in dependencies],
            "fixture_root": root}
        if external is not None:
            nodes[identity]["external"] = {"path": external}

    add("gcc", config["compiler"]["version"], external=config["compiler"]["prefix"], root=False)
    add("cmake", config["cmake"]["version"], external=config["cmake"]["prefix"])
    nodes[ids["cmake"]]["arch"]["target"] = config["cmake"]["target"]
    add("cuda", config["cuda_version"], external=config["cuda_prefix"])
    add("openmpi", config["profiles"][profile]["mpi_version"], {"cuda": True},
        config["profiles"][profile]["mpi_prefix"], dependencies=("cuda",))
    add("metis", config["metis_version"], {"shared": True, "int64": False, "real64": False}, dependencies=("gcc",))
    add("openblas", config["openblas"]["version"], copy.deepcopy(config["openblas"]["parameters"]),
        external=config["openblas"]["prefix"])
    nodes[ids["openblas"]]["arch"]["target"] = config["openblas"]["target"]
    add("zlib", config["zlib_version"], dependencies=("gcc",))
    gpu = profile == "cuda-sm80"
    params = {"precision": "double", "complex": False, "shared": True, "mpi": True,
              "cuda": gpu, "umpire": gpu, "cxxstd": "17", "openmp": False}
    deps = ["gcc", "openmpi", "openblas"]
    if gpu:
        add("umpire", config["umpire_version"], {"cuda_arch": ["80"], "cuda": True,
            "shared": True, "c": True, "fortran": False, "examples": False}, dependencies=("gcc", "cuda"))
        params.update(cuda_arch=["80"], **{"gpu-aware-mpi": True, "unified-memory": False})
        deps += ["cuda", "umpire"]
    add("hypre", config["hypre_version"], params, dependencies=deps)
    roots = [{"hash": identity} for identity, node in nodes.items() if node.pop("fixture_root")]
    return {"concrete_specs": nodes, "roots": roots}


def named_node(lock, name):
    return next(node for node in lock["concrete_specs"].values() if node["name"] == name)


class DeltaManifest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / "outputs"
        self.recipes = self.root / "recipes"
        self.helper = load_helper("delta_manifest")

    def tearDown(self):
        self.temp.cleanup()

    def test_policy_is_portable_and_system_mpi_external_for_both_profiles(self):
        config = self.helper.load_config()
        text = json.dumps(config)
        for path in ("/home/", "/scratch/", "/storage/", "jarias9"):
            self.assertNotIn(path, text)
        for profile in self.helper.PROFILES:
            observed = observed_config(profile)
            rendered = self.helper.manifest(self.output, profile, self.recipes, observed)["spack"]
            mpi = rendered["packages"]["openmpi"]
            self.assertIs(mpi["buildable"], False)
            self.assertEqual(mpi["externals"][0]["prefix"], "/observed/hpcx")
            self.assertIn("openmpi@4.1.9a1 +cuda", mpi["externals"][0]["spec"])
            self.assertEqual(rendered["config"]["install_tree"]["root"], str(self.output / "spack-store" / profile))
            hypre = next(spec for spec in rendered["specs"] if spec.startswith("hypre@"))
            self.assertIn("~cuda ~umpire" if profile == "cpu" else "cuda_arch=80", hypre)
            self.assertFalse(self.output.exists())

    def test_concrete_cpu_graph_accepts_cuda_mpi_without_cuda_hypre(self):
        for profile in self.helper.PROFILES:
            config = observed_config(profile)
            lock = concrete_lock(profile, config)
            result = self.helper.check_lock(lock, profile, config)
            self.assertTrue(result["mpi_external_cuda"])
        cpu = observed_config("cpu")
        for name, key, value in (("hypre", "cuda", True), ("hypre", "umpire", True),
                                 ("openmpi", "cuda", False), ("openblas", "threads", "pthreads")):
            with self.subTest(name=name, key=key):
                lock = concrete_lock("cpu", cpu)
                named_node(lock, name)["parameters"][key] = value
                with self.assertRaises(self.helper.DeltaError):
                    self.helper.check_lock(lock, "cpu", cpu)

    def test_gpu_arch_precision_and_dependency_identity_drift_reject(self):
        config = observed_config("cuda-sm80")
        for name, key, values in (("hypre", "cuda_arch", (["70"], ["80", "90"])),
                                   ("umpire", "cuda_arch", (["120"], [])),
                                   ("hypre", "precision", ("single", "mixed"))):
            for value in values:
                with self.subTest(name=name, key=key, value=value):
                    lock = concrete_lock("cuda-sm80", config)
                    named_node(lock, name)["parameters"][key] = value
                    with self.assertRaises(self.helper.DeltaError):
                        self.helper.check_lock(lock, "cuda-sm80", config)
        lock = concrete_lock("cuda-sm80", config)
        named_node(lock, "hypre")["dependencies"] = []
        with self.assertRaises(self.helper.DeltaError):
            self.helper.check_lock(lock, "cuda-sm80", config)

    def test_retained_manifest_is_immutable_and_policy_drift_creates_nothing(self):
        config = observed_config("cpu")
        paths = self.helper.write_environment(self.output, "cpu", self.recipes, config)
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        self.helper.write_environment(self.output, "cpu", self.recipes, config)
        self.helper.write_environment(self.output, "cuda-sm80", self.recipes, observed_config("cuda-sm80"))
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths})
        relocated = copy.deepcopy(config)
        relocated["profiles"]["cpu"]["mpi_prefix"] = "/changed/system-mpi"
        with self.assertRaisesRegex(self.helper.DeltaError, "retained Delta configuration differs"):
            self.helper.write_environment(self.output, "cpu", self.recipes, relocated)
        changed = copy.deepcopy(config)
        changed["profiles"]["cpu"]["mpi_parameters"]["cuda"] = False
        with self.assertRaisesRegex(self.helper.DeltaError, "checked-in Delta policy"):
            self.helper.write_environment(self.root / "never-created", "cpu", self.recipes, changed)
        self.assertFalse((self.root / "never-created").exists())
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths})

    def test_public_canonical_cpu_and_native_sm80_plans_create_nothing(self):
        for mode, profile in (("--cpu-only", "cpu"), ("--a100", "cuda-sm80")):
            with self.subTest(profile=profile):
                output = self.root / profile
                result = subprocess.run(["bash", str(SOURCE / "build.sh"), "--site", "delta",
                    mode, "--no-tests", "--hdg-runtime", "--output-root", str(output),
                    "--print-plan", "-j", "2"], text=True, capture_output=True,
                    env=closed_environment(), timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("[site-plan] site=delta profile=" + profile, result.stdout)
                self.assertIn("[build-plan] mode=", result.stdout)
                self.assertIn("dependency_cuda_arch=80 mfem_cuda_archs=80 app_cuda_archs=80" if profile != "cpu"
                              else "mfem_cuda_archs=none", result.stdout)
                self.assertFalse(output.exists())


    def test_canonical_check_lock_hook_reads_retained_sibling_config(self):
        config = observed_config("cpu")
        paths = self.helper.write_environment(self.output, "cpu", self.recipes, config)
        lock_path = paths[0].parent / "spack.lock"
        lock_path.write_text(json.dumps(concrete_lock("cpu", config)))
        result = subprocess.run([sys.executable, "-B", str(SOURCE / "scripts/build/delta_manifest.py"),
            "--profile", "cpu", "--check-lock", str(lock_path)],
            text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["profile"], "cpu")
        paths[1].unlink()
        result = subprocess.run([sys.executable, "-B", str(SOURCE / "scripts/build/delta_manifest.py"),
            "--profile", "cpu", "--check-lock", str(lock_path)],
            text=True, capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)


class DeltaTestReceipt(unittest.TestCase):
    def allocation(self, profile, *, changes=None, gpu_row=None):
        helper = load_helper("test_result_receipt")
        gpu = profile != "cpu"
        account = "bdiy-delta-gpu" if gpu else "bdiy-delta-cpu"
        partition = "gpuA100x4" if gpu else "cpu"
        fields = {"JobId": "7001", "JobState": "RUNNING", "Account": account,
            "Partition": partition, "BatchFlag": "1",
            "UserId": f"{pwd.getpwuid(os.getuid()).pw_name}({os.getuid()})",
            "AllocTRES": "cpu=8,mem=48G,node=1" + (",gres/gpu=1" if gpu else "")}
        fields.update(changes or {})
        env = {"SLURM_JOB_ID": "7001", "SLURM_JOB_ACCOUNT": fields["Account"],
            "SLURM_JOB_PARTITION": fields["Partition"], "CUDA_VISIBLE_DEVICES": "0",
            "SLURM_STEP_GPUS": "2" if gpu else ""}
        replies = [SimpleNamespace(returncode=0, stdout=" ".join(f"{k}={v}" for k, v in fields.items()), stderr="")]
        if gpu:
            replies.append(SimpleNamespace(returncode=0, stdout=gpu_row or
                "GPU-12345678-abcd-abcd-abcd-123456789abc, NVIDIA A100-SXM4-40GB, 8.0\n", stderr=""))
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("socket.getfqdn", return_value="cn001.delta.ncsa.illinois.edu"), \
             mock.patch.object(helper.subprocess, "run", side_effect=replies) as query:
            result = helper.allocation({"policy": {"site": "delta", "profile": profile}})
            self.assertEqual(query.call_count, 2 if gpu else 1)
        return helper, result

    def test_delta_receipts_admit_selected_cpu_and_a100_without_pace_qos(self):
        _, cpu = self.allocation("cpu")
        self.assertEqual(cpu["authority"], "live Delta allocation")
        self.assertNotIn("gpu", cpu)
        _, gpu = self.allocation("cuda-sm80")
        self.assertEqual(gpu["gpu"]["compute_capability"], "8.0")
        for profile, changes in (("cpu", {"Account": "bdiy-delta-gpu"}),
                                  ("cpu", {"Partition": "gpuA100x4"}),
                                  ("cuda-sm80", {"Account": "bdiy-delta-cpu"}),
                                  ("cuda-sm80", {"Partition": "cpu"}),
                                  ("cuda-sm80", {"AllocTRES": "cpu=8,gres/gpu=2"})):
            with self.subTest(profile=profile, changes=changes), self.assertRaises(ValueError):
                self.allocation(profile, changes=changes)
        with self.assertRaises(ValueError):
            self.allocation("cuda-sm80", gpu_row="GPU-12345678-abcd-abcd-abcd-123456789abc, V100, 7.0\n")

    def test_zero_returncode_is_not_enough_without_each_actual_passed_outcome(self):
        helper = load_helper("test_result_receipt")
        graph = {"tests": [{"name": "first"}, {"name": "second"}]}
        for stdout in ("", "1/2 Test #1: first ... Passed 0.1 sec\n",
                        "1/2 Test #1: first ... Passed 0.1 sec\n2/2 Test #2: second ... ***Skipped 0.1 sec\n"):
            with self.subTest(stdout=stdout), self.assertRaises(ValueError):
                helper.test_outcomes(stdout, graph)


if __name__ == "__main__":
    unittest.main()
