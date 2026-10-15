#!/usr/bin/env python3
"""Check PACE rendering and incompatible dependency rejection without Spack/builds."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2]
SCRIPT = SOURCE / "scripts/build/pace_manifest.py"
SPEC = importlib.util.spec_from_file_location("pace_manifest", SCRIPT)
PACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACE)
CONFIG = PACE.load_config()


def discovered_fixture(profile):
    config = PACE.profile_config(CONFIG, profile)
    for name in ("compiler", "cmake", "metis", "openblas"):
        config[name]["prefix"] = f"/site-fixture/{name}"
    mpi_profile = "cuda-sm80" if profile == "cuda-sm70" else profile
    config["profiles"][profile]["mpi_prefix"] = f"/site-fixture/{mpi_profile}/openmpi"
    if profile != "cpu":
        config["cuda_prefix"] = "/site-fixture/cuda"
    return config


def concrete_fixture(profile):
    """Minimal concrete lock with the real site's declared external identities."""
    config = discovered_fixture(profile)
    selected = config["profiles"][profile]
    gpu = profile != "cpu"
    cuda_arch = profile.removeprefix("cuda-sm") if gpu else None
    nodes, identities = {}, {}

    def add(name, version, parameters=None, prefix=None, rooted=True):
        identity = f"{len(nodes) + 1:032d}"
        identities[name] = identity
        node = {"name": name, "version": version, "arch": {"target": config["target"]},
                "parameters": parameters or {}, "dependencies": []}
        if prefix is not None:
            node["external"] = {"path": prefix, "module": None, "extra_attributes": {}}
        nodes[identity] = node
        return node

    add("gcc", config["compiler"]["version"], prefix=config["compiler"]["prefix"], rooted=False)
    for name in ("cmake", "metis"):
        add(name, config[name]["version"], parameters=copy.deepcopy(config[name].get("parameters", {})), prefix=config[name]["prefix"])
    add("openmpi", selected["mpi_version"], copy.deepcopy(selected["mpi_parameters"]), selected["mpi_prefix"])
    add("openblas", config["openblas"]["version"], {"threads": "none", "shared": True, "ilp64": False}, config["openblas"]["prefix"])
    hypre = add("hypre", config["hypre_version"], {"precision": "double", "complex": False,
        "shared": True, "mpi": True, "cuda": gpu, "umpire": gpu, "openmp": False, "cxxstd": "17"})
    dependencies = ["gcc", "openmpi", "openblas"]
    if gpu:
        add("cuda", config["cuda_version"], prefix=config["cuda_prefix"])
        umpire = add("umpire", config["umpire_version"], {"shared": True, "c": True, "cuda": True,
                     "cuda_arch": [cuda_arch], "fortran": False, "examples": False})
        umpire["dependencies"] = [{"name": "cuda", "hash": identities["cuda"]}]
        hypre["parameters"].update(cuda_arch=[cuda_arch], **{"gpu-aware-mpi": True, "unified-memory": False})
        dependencies.extend(("cuda", "umpire"))
    hypre["dependencies"] = [{"name": name, "hash": identities[name]} for name in dependencies]
    roots = [{"hash": identity, "spec": node["name"]} for identity, node in nodes.items() if node["name"] != "gcc"]
    return {"roots": roots, "concrete_specs": nodes}


def node(lock, name):
    return next(value for value in lock["concrete_specs"].values() if value["name"] == name)


class PaceManifest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "outputs"
        self.recipes = self.base / "recipes"

    def tearDown(self):
        self.temp.cleanup()

    def cli(self, *args):
        profile = args[args.index("--profile") + 1]
        config_path = self.base / "discovered.json"
        config_path.write_text(json.dumps(discovered_fixture(profile)))
        args = (*args, "--site-config", config_path)
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], text=True,
                              capture_output=True, timeout=10)

    def test_dry_run_is_portable_and_creates_nothing(self):
        for profile in PACE.PROFILES:
            result = self.cli("--root", self.root, "--profile", profile,
                              "--recipe-root", self.recipes, "--dry-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            rendered = json.loads(result.stdout)["spack"]
            self.assertFalse(self.root.exists())
            self.assertFalse(self.recipes.exists())
            self.assertEqual(rendered["config"]["install_tree"]["root"], str(self.root / "spack-store" / profile))
            self.assertEqual(rendered["repos"], {"builtin": str(self.recipes)})
            self.assertEqual(rendered["concretizer"]["targets"], {"granularity": "generic", "host_compatible": True})
            self.assertEqual(rendered["upstreams:"], {})
            self.assertTrue(rendered["view"])
            for spec in rendered["specs"]:
                self.assertIn("target=x86_64_v3", spec)
                if "%gcc" in spec:
                    self.assertLess(spec.index("target=x86_64_v3"), spec.index("%gcc"))
            self.assertIn("target=x86_64_v3", rendered["packages"]["hypre"]["require"])
            self.assertIn("precision=double", rendered["packages"]["hypre"]["require"])

    def test_site_config_has_no_user_or_campaign_paths(self):
        encoded = json.dumps(CONFIG)
        self.assertNotIn("jarias9", encoded)
        self.assertNotIn("20260927", encoded)
        self.assertNotIn("/scratch", encoded)
        self.assertNotIn("/home", encoded)
        self.assertNotIn("/storage", encoded)
        self.assertNotIn("prefix", encoded)
        self.assertEqual(CONFIG["profiles"]["cpu"]["mpi_version"], "4.1.5")
        self.assertEqual(CONFIG["profiles"]["cuda-sm80"]["mpi_version"], "5.0.9")

    def test_discovered_paths_can_move_without_source_edits(self):
        config = discovered_fixture("cpu")
        for name in ("compiler", "cmake", "metis", "openblas"):
            config[name]["prefix"] = f"/relocated/site/{name}"
        config["profiles"]["cpu"]["mpi_prefix"] = "/relocated/site/openmpi"
        rendered = PACE.manifest(self.root, "cpu", self.recipes, config)["spack"]
        self.assertEqual(rendered["packages"]["gcc"]["externals"][0]["prefix"], "/relocated/site/compiler")
        self.assertEqual(rendered["packages"]["openmpi"]["externals"][0]["prefix"], "/relocated/site/openmpi")
        paths = PACE.write_environment(self.root, "cpu", self.recipes, config)
        self.assertEqual(json.loads(paths[2].read_text()), config)
        with self.assertRaisesRegex(PACE.PaceError, "choose a new --site-root"):
            PACE.write_environment(self.root, "cpu", self.recipes, discovered_fixture("cpu"))

    def test_discovery_cannot_change_policy_or_omit_prefixes(self):
        with self.assertRaisesRegex(PACE.PaceError, "must be discovered"):
            PACE.manifest(self.root, "cpu", self.recipes)
        config = discovered_fixture("cpu")
        config["hypre_version"] = "0.0.0"
        with self.assertRaisesRegex(PACE.PaceError, "constraints"):
            PACE.manifest(self.root, "cpu", self.recipes, config)

    def test_identical_graph_is_reused_without_file_rewrites(self):
        paths = PACE.write_environment(self.root, "cpu", self.recipes, discovered_fixture("cpu"))
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
        self.assertEqual(paths[0], self.root / "site/pace/spack-envs/cpu/spack.yaml")
        self.assertEqual(paths[1], self.root / "site/pace/spack-user/repos.yaml")
        PACE.write_environment(self.root, "cpu", self.recipes, discovered_fixture("cpu"))
        self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths})
        PACE.write_environment(self.root, "cuda-sm80", self.recipes, discovered_fixture("cuda-sm80"))
        self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths})

    def test_v100_profile_uses_native_separate_graph_and_preserves_legacy_documents(self):
        retained = []
        for profile in ("cpu", "cuda-sm80"):
            config = discovered_fixture(profile)
            self.assertNotIn("cuda-sm70", config["profiles"])
            retained.extend(PACE.write_environment(self.root, profile, self.recipes, config))
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in retained}
        paths = PACE.write_environment(self.root, "cuda-sm70", self.recipes, discovered_fixture("cuda-sm70"))
        rendered = json.loads(paths[0].read_text())["spack"]
        self.assertEqual(rendered["config"]["install_tree"]["root"], str(self.root / "spack-store/cuda-sm70"))
        for name in ("hypre", "umpire"):
            spec = next(spec for spec in rendered["specs"] if spec.startswith(name + "@"))
            self.assertIn("cuda_arch=70", spec)
            self.assertNotIn("cuda_arch=80", spec)
        for profile in ("cpu", "cuda-sm80"):
            # Full policy discovery and an already-retained two-profile receipt
            # must normalize to exactly the same existing documents.
            config = discovered_fixture(profile)
            config["profiles"]["cuda-sm70"] = copy.deepcopy(CONFIG["profiles"]["cuda-sm70"])
            PACE.write_environment(self.root, profile, self.recipes, config)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in retained})

    def test_v100_requires_native_architecture_in_every_gpu_dependency(self):
        for name in ("hypre", "umpire"):
            for value in (["80"], ["70", "80"], ["120"], "none"):
                with self.subTest(name=name, value=value):
                    lock = concrete_fixture("cuda-sm70")
                    node(lock, name)["parameters"]["cuda_arch"] = value
                    with self.assertRaisesRegex(PACE.PaceError, "cuda_arch"):
                        PACE.check_lock(lock, "cuda-sm70", discovered_fixture("cuda-sm70"))
        for profile, other in (("cuda-sm70", "cuda-sm80"), ("cuda-sm80", "cuda-sm70")):
            with self.assertRaisesRegex(PACE.PaceError, "cuda_arch"):
                PACE.check_lock(concrete_fixture(other), profile, discovered_fixture(profile))

    def test_v100_retained_policy_cannot_change_modules_or_mpi_variants(self):
        for key, value in (("modules", ["unrelated/module"]), ("mpi_version", "4.1.5")):
            config = discovered_fixture("cuda-sm70")
            config["profiles"]["cuda-sm70"][key] = value
            with self.assertRaisesRegex(PACE.PaceError, "constraints"):
                PACE.manifest(self.root, "cuda-sm70", self.recipes, config)

    def test_changed_manifest_is_preserved_and_new_root_is_requested(self):
        path = PACE.write_environment(self.root, "cpu", self.recipes, discovered_fixture("cpu"))[0]
        changed = path.read_text().replace("precision=double", "precision=single")
        path.write_text(changed)
        result = self.cli("--root", self.root, "--profile", "cpu", "--recipe-root", self.recipes)
        self.assertEqual(result.returncode, 2)
        self.assertIn("choose a new --site-root", result.stderr)
        self.assertEqual(path.read_text(), changed)

    def test_changed_recipe_scope_rejects_before_creating_manifest(self):
        path = self.root / "site/pace/spack-user/repos.yaml"
        path.parent.mkdir(parents=True)
        path.write_text("preserve this external scope\n")
        with self.assertRaisesRegex(PACE.PaceError, "choose a new --site-root"):
            PACE.write_environment(self.root, "cpu", self.recipes, discovered_fixture("cpu"))
        self.assertFalse((self.root / "site/pace/spack-envs").exists())
        self.assertEqual(path.read_text(), "preserve this external scope\n")

    def test_symlinked_configuration_is_not_followed(self):
        path = self.root / "site/pace/spack-user/repos.yaml"
        path.parent.mkdir(parents=True)
        target = self.base / "elsewhere"
        target.write_text("preserve\n")
        path.symlink_to(target)
        with self.assertRaisesRegex(PACE.PaceError, "symlinked"):
            PACE.write_environment(self.root, "cpu", self.recipes, discovered_fixture("cpu"))
        self.assertEqual(target.read_text(), "preserve\n")

    def test_relative_paths_reject_without_outputs(self):
        for root, recipes in (("relative", self.recipes), (self.root, "relative")):
            result = self.cli("--root", root, "--profile", "cpu", "--recipe-root", recipes, "--dry-run")
            self.assertEqual(result.returncode, 2)
            self.assertIn("normalized absolute path", result.stderr)
        self.assertFalse(self.root.exists())

    def test_supported_concrete_profiles_and_target_dictionary(self):
        for profile in PACE.PROFILES:
            lock = concrete_fixture(profile)
            for value in lock["concrete_specs"].values():
                value["arch"]["target"] = {"name": "x86_64_v3", "vendor": "generic"}
            result = PACE.check_lock(lock, profile, discovered_fixture(profile))
            self.assertEqual(result["profile"], profile)
            self.assertEqual(result["precision"], "double")

    def test_host_specific_target_in_any_dependency_rejects(self):
        for name in ("hypre", "gcc", "cmake"):
            for target in ("cascadelake", "zen3", "x86_64_v4", "x86_64"):
                with self.subTest(name=name, target=target):
                    lock = concrete_fixture("cpu")
                    node(lock, name)["arch"]["target"] = target
                    with self.assertRaisesRegex(PACE.PaceError, "all PACE nodes"):
                        PACE.check_lock(lock, "cpu", discovered_fixture("cpu"))

    def test_hypre_precision_and_configuration_mismatches_reject(self):
        for profile in PACE.PROFILES:
            for key, value in (("precision", "single"), ("precision", "mixed"), ("complex", True),
                               ("shared", False), ("mpi", False), ("cxxstd", "14"), ("openmp", True)):
                with self.subTest(profile=profile, key=key, value=value):
                    lock = concrete_fixture(profile)
                    node(lock, "hypre")["parameters"][key] = value
                    with self.assertRaises(PACE.PaceError):
                        PACE.check_lock(lock, profile, discovered_fixture(profile))

    def test_metis_module_abi_mismatches_reject(self):
        for key, value in (("int64", True), ("real64", True), ("shared", False)):
            lock = concrete_fixture("cpu")
            node(lock, "metis")["parameters"][key] = value
            with self.assertRaises(PACE.PaceError):
                PACE.check_lock(lock, "cpu", discovered_fixture("cpu"))

    def test_versions_prefixes_and_mpi_profile_mismatches_reject(self):
        for profile in PACE.PROFILES:
            for name in ("gcc", "hypre", "openmpi", "cmake", "metis", "openblas"):
                with self.subTest(profile=profile, version=name):
                    lock = concrete_fixture(profile)
                    node(lock, name)["version"] = "0.0.0"
                    with self.assertRaises(PACE.PaceError): PACE.check_lock(lock, profile, discovered_fixture(profile))
            for name in ("gcc", "openmpi", "cmake", "metis", "openblas"):
                with self.subTest(profile=profile, prefix=name):
                    lock = concrete_fixture(profile)
                    node(lock, name)["external"]["path"] = "/unrelated/prefix"
                    with self.assertRaises(PACE.PaceError): PACE.check_lock(lock, profile, discovered_fixture(profile))
            lock = concrete_fixture(profile)
            node(lock, "openmpi")["parameters"]["cuda"] = profile == "cpu"
            with self.assertRaises(PACE.PaceError): PACE.check_lock(lock, profile, discovered_fixture(profile))

    def test_hypre_cannot_bind_another_mpi_or_blas_identity(self):
        for name in ("openmpi", "openblas", "gcc"):
            lock = concrete_fixture("cpu")
            node(lock, "hypre")["dependencies"] = [item for item in node(lock, "hypre")["dependencies"] if item["name"] != name]
            with self.assertRaisesRegex(PACE.PaceError, "DAG identity"):
                PACE.check_lock(lock, "cpu", discovered_fixture("cpu"))

    def test_gpu_version_native_architecture_and_runtime_contract_reject(self):
        for name, key, value in (("cuda", "version", "13.0.1"), ("umpire", "version", "2024.07.0")):
            lock = concrete_fixture("cuda-sm80")
            node(lock, name)[key] = value
            with self.assertRaises(PACE.PaceError): PACE.check_lock(lock, "cuda-sm80", discovered_fixture("cuda-sm80"))
        for name, key, value in (("hypre", "cuda_arch", ["120"]), ("hypre", "cuda_arch", ["80", "120"]),
                                 ("umpire", "cuda_arch", ["120"]), ("hypre", "gpu-aware-mpi", False),
                                 ("hypre", "unified-memory", True), ("umpire", "cuda", False)):
            lock = concrete_fixture("cuda-sm80")
            node(lock, name)["parameters"][key] = value
            with self.assertRaises(PACE.PaceError): PACE.check_lock(lock, "cuda-sm80", discovered_fixture("cuda-sm80"))

    def test_missing_or_duplicate_roots_and_dangling_edges_reject(self):
        lock = concrete_fixture("cpu")
        lock["roots"] = [item for item in lock["roots"] if item["spec"] != "hypre"]
        with self.assertRaisesRegex(PACE.PaceError, "concrete root"): PACE.check_lock(lock, "cpu", discovered_fixture("cpu"))
        lock = concrete_fixture("cpu")
        lock["concrete_specs"]["duplicate"] = copy.deepcopy(node(lock, "hypre"))
        with self.assertRaisesRegex(PACE.PaceError, "exactly one"): PACE.check_lock(lock, "cpu", discovered_fixture("cpu"))
        lock = concrete_fixture("cpu")
        node(lock, "hypre")["dependencies"][0]["hash"] = "missing"
        with self.assertRaisesRegex(PACE.PaceError, "outside the concrete graph"): PACE.check_lock(lock, "cpu", discovered_fixture("cpu"))

    def test_cpu_cannot_admit_a_cuda_graph(self):
        with self.assertRaises(PACE.PaceError): PACE.check_lock(concrete_fixture("cuda-sm80"), "cpu", discovered_fixture("cpu"))

    def test_check_lock_mode_needs_no_render_paths_and_never_writes(self):
        path = self.base / "spack.lock"
        path.write_text(json.dumps(concrete_fixture("cpu")))
        before = path.read_bytes(), path.stat().st_mtime_ns
        result = self.cli("--profile", "cpu", "--check-lock", path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["profile"], "cpu")
        self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
        self.assertFalse(self.root.exists())

    def test_check_lock_uses_saved_discovery_without_new_queries(self):
        path = self.base / "spack.lock"
        path.write_text(json.dumps(concrete_fixture("cpu")))
        receipt = path.parent / "site-packages.json"
        receipt.write_text(json.dumps(discovered_fixture("cpu")))
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (path, receipt)}
        result = subprocess.run([sys.executable, str(SCRIPT), "--profile", "cpu", "--check-lock", str(path)],
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (path, receipt)})
        receipt.unlink()
        result = subprocess.run([sys.executable, str(SCRIPT), "--profile", "cpu", "--check-lock", str(path)],
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("site-packages.json", result.stderr)

    def test_malformed_lock_is_an_actionable_failure(self):
        path = self.base / "spack.lock"
        for content in ("not json", "{}", '{"concrete_specs": []}', '{"concrete_specs": {"bad": {}}}'):
            path.write_text(content)
            result = self.cli("--profile", "cpu", "--check-lock", path)
            self.assertEqual(result.returncode, 2)
            self.assertIn("ERROR: PACE manifest", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
