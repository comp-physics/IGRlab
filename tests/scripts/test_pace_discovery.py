#!/usr/bin/env python3
"""Exercise site discovery admission with host metadata fixtures, without Spack."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2]
SCRIPT = SOURCE / "scripts/build/pace_discover.py"
SPEC = importlib.util.spec_from_file_location("pace_discover", SCRIPT)
PACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACE)


class PaceDiscovery(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.tool = self.root / "tool/bin/spack"
        self.tool.parent.mkdir(parents=True)
        self.tool.write_text("fixture\n")
        self.tool.chmod(0o755)
        self.store = self.root / "shared store"
        self.index = self.store / ".spack-db/index.json"
        self.index.parent.mkdir(parents=True)
        self.index.write_text(json.dumps({"database": {"version": "8", "installs": {}}}))
        self.config = json.loads(PACE.CONFIG_PATH.read_text())
        self.records, self.prefixes, self.locations = {}, {}, {}
        versions = {name: self.config[name]["version"] for name in ("compiler", "cmake", "metis", "openblas")}
        versions["gcc"] = versions.pop("compiler")
        versions["cuda"] = self.config["cuda_version"]
        for index, (name, version) in enumerate(versions.items()):
            self.add(name, version, chr(ord("a") + index) * 32)
        self.add("openmpi-cpu", self.config["profiles"]["cpu"]["mpi_version"], "f" * 32, name="openmpi")
        self.add("openmpi-gpu", self.config["profiles"]["cuda-sm80"]["mpi_version"], "g" * 32, name="openmpi")
        self.records["gcc"][0]["parameters"] = {"languages": ["c", "c++", "fortran"]}
        self.records["metis"][0]["parameters"] = copy.deepcopy(self.config["metis"]["parameters"])
        self.records["openblas"][0]["parameters"] = {"threads": "none", "shared": True, "ilp64": False}
        for profile, key in (("cpu", "openmpi-cpu"), ("cuda-sm80", "openmpi-gpu")):
            self.records[key][0]["parameters"] = copy.deepcopy(self.config["profiles"][profile]["mpi_parameters"])
            for variant in ("fabrics", "schedulers"):
                self.records[key][0]["parameters"][variant] = [self.records[key][0]["parameters"][variant]]
        self.anchor = self.prefixes["gcc"] / "bin/gcc"
        self.anchor.parent.mkdir()
        self.anchor.write_text("compiler fixture\n")
        self.recipe = self.root / "configured recipes"
        (self.recipe / "packages").mkdir(parents=True)
        (self.recipe / "repo.yaml").write_text("repo:\n  namespace: builtin\n")
        self.environment = {key: "/unrelated/private/scope" for key in PACE.ISOLATED_VARIABLES}
        self.environment.update(METIS_ROOT=str(self.prefixes["metis"]), PATH="/fixture/path")
        self.calls = []
        self.version = "1.3.0.dev0\n"
        self.recipe_output = str(self.recipe) + "\n"
        self.scoped_recipe_output = None
        self.site_root = self.root / "dependency root"

    def tearDown(self):
        self.temporary.cleanup()

    def add(self, key, version, identity, name=None):
        name = name or key
        prefix = self.store / (key + " prefix")
        prefix.mkdir(exist_ok=True)
        self.prefixes[key] = prefix
        self.locations[identity] = prefix
        record = {"name": name, "version": version, "hash": identity,
                  "arch": {"target": self.config["target"]}, "parameters": {}, "dependencies": []}
        if name not in ("gcc", "cuda"):
            record["annotations"] = {"compiler": "gcc@=" + self.config["compiler"]["version"]}
        self.records[key] = [record]
        return record

    def query(self, spack, arguments, environment):
        self.calls.append((str(spack), list(arguments), dict(environment)))
        if arguments == ["--version"]:
            return self.version
        if arguments[-3:] == ["location", "--repo", "builtin"]:
            if self.scoped_recipe_output is not None:
                return self.scoped_recipe_output
            if arguments[:1] == ["-e"]:
                document = json.loads((Path(arguments[1]) / "spack.yaml").read_text())
                return document["spack"]["repos"]["builtin"] + "\n"
            repos = Path(environment.get("SPACK_USER_CONFIG_PATH", "")) / "repos.yaml"
            if repos.is_file():
                return json.loads(repos.read_text())["repos"]["builtin"] + "\n"
            return self.recipe_output
        self.assertEqual(arguments[:2], ["-c", f"config:install_tree:root:{self.store}"])
        if arguments[2:4] == ["find", "--json"]:
            specification = arguments[4]
            if specification.startswith("openmpi"):
                key = "openmpi-cpu" if self.config["profiles"]["cpu"]["mpi_version"] in specification else "openmpi-gpu"
            else:
                key = specification.split("@", 1)[0]
            return json.dumps(self.records[key])
        self.assertEqual(arguments[2:4], ["location", "-i"])
        return str(self.locations[arguments[4][1:]]) + "\n"

    def discover(self, profile="cpu"):
        with patch.object(PACE, "query", self.query):
            return PACE.discover_config(profile, self.tool, self.anchor, self.config, self.environment)

    def ambiguous_metis(self):
        original = self.records["metis"][0]
        for key, identity, real64 in (("metis-other", "h" * 32, False), ("metis-real64", "i" * 32, True)):
            record = self.add(key, self.config["metis"]["version"], identity, name="metis")
            record["parameters"] = copy.deepcopy(original["parameters"])
            record["parameters"]["real64"] = real64
            self.records["metis"].append(record)

    def test_cpu_gpu_only_populate_selected_policy_prefixes(self):
        for profile in PACE.PROFILES:
            with self.subTest(profile=profile):
                result = self.discover(profile)
                stripped = copy.deepcopy(result)
                for key in ("compiler", "cmake", "metis", "openblas"):
                    expected = self.prefixes["gcc" if key == "compiler" else key]
                    self.assertEqual(stripped[key].pop("prefix"), str(expected))
                mpi = "openmpi-cpu" if profile == "cpu" else "openmpi-gpu"
                self.assertEqual(stripped["profiles"][profile].pop("mpi_prefix"), str(self.prefixes[mpi]))
                if profile != "cpu":
                    self.assertEqual(stripped.pop("cuda_prefix"), str(self.prefixes["cuda"]))
                expected = copy.deepcopy(self.config)
                if profile in ("cpu", "cuda-sm80"):
                    expected["profiles"].pop("cuda-sm70")
                self.assertEqual(stripped, expected)
                self.assertEqual(self.index.read_text(), json.dumps({"database": {"version": "8", "installs": {}}}))

    def test_loaded_metis_module_resolves_three_actual_dag_shapes(self):
        self.ambiguous_metis()
        result = self.discover()
        self.assertEqual(result["metis"]["prefix"], str(self.prefixes["metis"]))
        selected = [call[1][-1] for call in self.calls if call[1][2:4] == ["location", "-i"]]
        self.assertIn("/" + "h" * 32, selected)
        self.assertIn("/" + "i" * 32, selected)
        self.assertFalse(any("--first" in call[1] for call in self.calls))

    def test_ambiguous_metis_without_loaded_module_rejects(self):
        self.ambiguous_metis()
        self.environment.pop("METIS_ROOT")
        with self.assertRaisesRegex(PACE.DiscoveryError, "ambiguous installed metis"):
            self.discover()

    def test_module_aliases_must_agree_and_belong_to_store(self):
        self.environment["METISROOT"] = str(self.prefixes["cmake"])
        with self.assertRaisesRegex(PACE.DiscoveryError, "conflict"):
            self.discover()
        self.environment.pop("METISROOT")
        self.environment["METIS_ROOT"] = str(self.root)
        with self.assertRaisesRegex(PACE.DiscoveryError, "escapes"):
            self.discover()

    def test_two_dags_at_same_module_prefix_remain_ambiguous(self):
        self.ambiguous_metis()
        self.locations["h" * 32] = self.prefixes["metis"]
        with self.assertRaisesRegex(PACE.DiscoveryError, "exactly one"):
            self.discover()

    def test_missing_and_non_metis_ambiguous_packages_do_not_fallback(self):
        self.records["cmake"] = []
        with self.assertRaisesRegex(PACE.DiscoveryError, "no installed cmake"):
            self.discover()
        self.add("cmake", self.config["cmake"]["version"], "j" * 32)
        second = copy.deepcopy(self.records["cmake"][0])
        second["hash"] = "k" * 32
        self.records["cmake"].append(second)
        with self.assertRaisesRegex(PACE.DiscoveryError, "ambiguous installed cmake"):
            self.discover()

    def test_wrong_name_version_target_or_identity_rejects_before_location(self):
        original = copy.deepcopy(self.records["cmake"])
        for key, value in (("name", "other"), ("version", "0"), ("arch", {"target": "cascadelake"}), ("hash", "short")):
            with self.subTest(key=key):
                self.records["cmake"] = copy.deepcopy(original)
                self.records["cmake"][0][key] = value
                with self.assertRaises(PACE.DiscoveryError):
                    self.discover()

    def test_required_precision_abi_and_mpi_variants_reject(self):
        cases = (("metis", "int64", True, "cpu"), ("metis", "real64", True, "cpu"),
                 ("openblas", "threads", "openmp", "cpu"), ("openmpi-gpu", "cuda", False, "cuda-sm80"),
                 ("openmpi-gpu", "cuda", False, "cuda-sm70"))
        for key, variant, value, profile in cases:
            with self.subTest(key=key, variant=variant):
                saved = self.records[key][0]["parameters"][variant]
                self.records[key][0]["parameters"][variant] = value
                with self.assertRaisesRegex(PACE.DiscoveryError, variant):
                    self.discover(profile)
                self.records[key][0]["parameters"][variant] = saved

    def test_legacy_compiler_annotation_and_modern_gcc_edge_are_validated(self):
        cmake = self.records["cmake"][0]
        cmake["annotations"]["compiler"] = "gcc@=99.0"
        with self.assertRaisesRegex(PACE.DiscoveryError, "compiler annotation"):
            self.discover()
        cmake.pop("annotations")
        cmake["dependencies"] = [{"name": "gcc", "hash": self.records["gcc"][0]["hash"]}]
        self.discover()
        cmake["dependencies"][0]["hash"] = "z" * 32
        with self.assertRaisesRegex(PACE.DiscoveryError, "compiler dependency"):
            self.discover()

    def test_compiler_metadata_cannot_be_omitted(self):
        self.records["cmake"][0].pop("annotations")
        with self.assertRaisesRegex(PACE.DiscoveryError, "does not identify"):
            self.discover()

    def test_external_find_and_location_prefixes_must_match(self):
        record = self.records["openmpi-cpu"][0]
        record["external"] = {"path": str(self.prefixes["cmake"])}
        with self.assertRaisesRegex(PACE.DiscoveryError, "prefix mismatch"):
            self.discover()

    def test_packages_cannot_escape_shared_store(self):
        self.locations[self.records["cmake"][0]["hash"]] = self.root
        with self.assertRaisesRegex(PACE.DiscoveryError, "escapes"):
            self.discover()

    def test_anchor_must_match_discovered_gcc(self):
        self.locations[self.records["gcc"][0]["hash"]] = self.prefixes["cmake"]
        with self.assertRaisesRegex(PACE.DiscoveryError, "exactly one"):
            self.discover()

    def test_compiler_module_anchor_can_select_among_same_version_gcc_dags(self):
        second = self.add("gcc-other", self.config["compiler"]["version"], "j" * 32, name="gcc")
        second["parameters"] = copy.deepcopy(self.records["gcc"][0]["parameters"])
        self.records["gcc"].append(second)
        self.assertEqual(self.discover()["compiler"]["prefix"], str(self.prefixes["gcc"]))

    def test_symlinked_module_prefixes_are_normalized(self):
        alias = self.root / "site-link"
        alias.symlink_to(self.store, target_is_directory=True)
        self.environment["METIS_ROOT"] = str(alias / self.prefixes["metis"].name)
        self.locations[self.records["metis"][0]["hash"]] = alias / self.prefixes["metis"].name
        self.assertEqual(self.discover()["metis"]["prefix"], str(self.prefixes["metis"]))

    def test_incompatible_reader_and_database_do_not_query_shared_store(self):
        self.version = "0.24.0.dev0\n"
        with self.assertRaisesRegex(PACE.DiscoveryError, "compatible"):
            self.discover()
        self.assertEqual([call[1] for call in self.calls], [["--version"]])
        self.version = "1.3.0.dev0\n"
        self.calls.clear()
        self.index.write_text('{"database":{"version":"7"}}')
        with self.assertRaisesRegex(PACE.DiscoveryError, "possible reindex"):
            self.discover()
        self.assertEqual([call[1] for call in self.calls], [["--version"]])

    def test_missing_or_nested_shared_store_is_rejected(self):
        self.index.unlink()
        with self.assertRaisesRegex(PACE.DiscoveryError, "exactly one"):
            self.discover()
        self.index.write_text('{"database":{"version":"8"}}')
        outer = self.root / ".spack-db/index.json"
        outer.parent.mkdir()
        outer.write_text('{"database":{"version":"8"}}')
        with self.assertRaisesRegex(PACE.DiscoveryError, "exactly one"):
            self.discover()

    def test_site_queries_clear_private_scopes_and_bind_supplied_tool(self):
        self.discover()
        for executable, arguments, environment in self.calls:
            self.assertEqual(executable, str(self.tool))
            for key in PACE.ISOLATED_VARIABLES:
                if key == "SPACK_ROOT":
                    self.assertEqual(environment[key], str(self.tool.parent.parent))
                else:
                    self.assertNotIn(key, environment)
            self.assertEqual(environment["PATH"], "/fixture/path")
            self.assertNotIn("install", arguments)
            self.assertNotIn("reindex", arguments)

    def test_recipe_public_query_keeps_configured_scopes(self):
        with patch.object(PACE, "query", self.query):
            result = PACE.discover_recipe_root(self.tool, self.environment)
        self.assertEqual(result, str(self.recipe))
        self.assertEqual(self.calls[-1][1], ["location", "--repo", "builtin"])
        for key in PACE.ISOLATED_VARIABLES:
            expected = str(self.tool.parent.parent) if key == "SPACK_ROOT" else self.environment[key]
            self.assertEqual(self.calls[-1][2][key], expected)

    def test_recipe_path_must_identify_one_complete_repository(self):
        self.recipe_output += str(self.root) + "\n"
        with patch.object(PACE, "query", self.query), self.assertRaisesRegex(PACE.DiscoveryError, "one path"):
            PACE.discover_recipe_root(self.tool, self.environment)
        self.recipe_output = str(self.root) + "\n"
        with patch.object(PACE, "query", self.query), self.assertRaisesRegex(PACE.DiscoveryError, "lacks repo.yaml"):
            PACE.discover_recipe_root(self.tool, self.environment)

    def retain(self, profile="cpu", shared=True, recipe=None):
        recipe = str(recipe or self.recipe)
        environment = self.site_root / "site/pace/spack-envs" / profile
        environment.mkdir(parents=True, exist_ok=True)
        (environment / "spack.yaml").write_text(json.dumps({"spack": {"repos": {"builtin": recipe}}}))
        if shared:
            metadata = self.site_root / "site/pace/spack-user/repos.yaml"
            metadata.parent.mkdir(parents=True, exist_ok=True)
            metadata.write_text(json.dumps({"repos": {"builtin": recipe}}))
        return environment

    def discover_recipe(self, profile="cpu", override=None):
        with patch.object(PACE, "query", self.query):
            return PACE.discover_recipe_root(self.tool, self.environment,
                                            self.site_root, profile, override)

    def test_retained_profile_wins_over_ambient_recipe_drift(self):
        self.recipe_output = str(self.root) + "\n"
        for profile in PACE.PROFILES:
            with self.subTest(profile=profile):
                selected = self.retain(profile)
                self.assertEqual(self.discover_recipe(profile), str(self.recipe))
                self.assertEqual(self.calls[-1][1], ["-e", str(selected), "location", "--repo", "builtin"])

    def test_retained_queries_clear_ambient_scopes_and_bind_private_scopes(self):
        self.retain()
        self.discover_recipe()
        environment = self.calls[-1][2]
        self.assertNotIn("SPACK_ENV", environment)
        self.assertNotIn("SPACK_DISABLE_LOCAL_CONFIG", environment)
        self.assertEqual(environment["SPACK_ROOT"], str(self.tool.parent.parent))
        self.assertEqual(environment["SPACK_USER_CONFIG_PATH"], str(self.site_root / "site/pace/spack-user"))
        self.assertEqual(environment["SPACK_SYSTEM_CONFIG_PATH"], str(self.site_root / "site/pace/spack-system"))
        self.assertEqual(environment["SPACK_USER_CACHE_PATH"], str(self.site_root / "spack-cache"))
        self.assertEqual(environment["PATH"], self.environment["PATH"])

    def test_new_second_profile_uses_retained_shared_user_scope(self):
        self.retain("cpu")
        self.recipe_output = str(self.root) + "\n"
        self.assertEqual(self.discover_recipe("cuda-sm80"), str(self.recipe))
        self.assertEqual(self.calls[-1][1], ["location", "--repo", "builtin"])
        self.assertFalse((self.site_root / "site/pace/spack-envs/cuda-sm80").exists())

    def test_v100_uses_same_exact_cuda_mpi_stack_without_installing_or_touching_a100(self):
        selected = self.retain("cuda-sm80")
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in selected.iterdir()}
        discovered = self.discover("cuda-sm70")
        self.assertEqual(discovered["profiles"]["cuda-sm70"]["mpi_prefix"], str(self.prefixes["openmpi-gpu"]))
        self.assertEqual(discovered["cuda_prefix"], str(self.prefixes["cuda"]))
        self.assertEqual(self.discover_recipe("cuda-sm70"), str(self.recipe))
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in selected.iterdir()})
        self.assertFalse((self.site_root / "site/pace/spack-envs/cuda-sm70").exists())
        self.assertFalse(any("install" in arguments for _, arguments, _ in self.calls))

    def test_other_profile_environment_supplies_authority_without_shared_scope(self):
        selected = self.retain("cuda-sm80", shared=False)
        self.assertEqual(self.discover_recipe(), str(self.recipe))
        self.assertEqual(self.calls[-1][1][:2], ["-e", str(selected)])

    def test_retained_shared_scope_can_supply_authority_without_profile(self):
        selected = self.retain()
        (selected / "spack.yaml").unlink()
        selected.rmdir()
        self.assertEqual(self.discover_recipe("cuda-sm80"), str(self.recipe))
        self.assertEqual(self.calls[-1][1], ["location", "--repo", "builtin"])

    def test_initial_root_keeps_current_public_discovery(self):
        self.assertEqual(self.discover_recipe(), str(self.recipe))
        self.assertEqual(self.calls[-1][1], ["location", "--repo", "builtin"])
        self.assertEqual(self.calls[-1][2]["SPACK_ENV"], self.environment["SPACK_ENV"])
        self.assertFalse(self.site_root.exists())

    def test_explicit_override_admits_initial_or_matching_retained_authority(self):
        self.assertEqual(self.discover_recipe(override=str(self.recipe)), str(self.recipe))
        self.assertEqual(self.calls, [])
        self.retain()
        self.assertEqual(self.discover_recipe(override=str(self.recipe)), str(self.recipe))
        self.assertIn("-e", self.calls[-1][1])

    def test_explicit_override_conflict_rejects_before_spack_queries(self):
        self.retain()
        other = self.root / "other recipes"
        (other / "packages").mkdir(parents=True)
        (other / "repo.yaml").write_text("repo:\n  namespace: builtin\n")
        with self.assertRaisesRegex(PACE.DiscoveryError, "explicit.*conflicts"):
            self.discover_recipe(override=str(other))
        self.assertEqual(self.calls, [])

    def test_retained_profiles_and_shared_scope_must_agree(self):
        self.retain()
        other = self.root / "other recipes"
        (other / "packages").mkdir(parents=True)
        (other / "repo.yaml").write_text("repo:\n  namespace: builtin\n")
        self.retain("cuda-sm80", shared=False, recipe=other)
        with self.assertRaisesRegex(PACE.DiscoveryError, "profile/shared.*conflict"):
            self.discover_recipe()
        self.assertEqual(self.calls, [])

    def test_missing_malformed_and_incomplete_retained_metadata_do_not_fallback(self):
        selected = self.retain(shared=False)
        metadata = selected / "spack.yaml"
        for content in (None, "not JSON", '{"spack":{}}', '{"spack":{"repos":{"builtin":null}}}'):
            with self.subTest(content=content):
                if content is None:
                    metadata.unlink()
                else:
                    metadata.write_text(content)
                with self.assertRaises(PACE.DiscoveryError):
                    self.discover_recipe()
                self.assertEqual(self.calls, [])
        metadata.write_text(json.dumps({"spack": {"repos": {"builtin": str(self.recipe)}}}))
        user_scope = self.site_root / "site/pace/spack-user"
        user_scope.mkdir()
        with self.assertRaisesRegex(PACE.DiscoveryError, "metadata is missing"):
            self.discover_recipe()

    def test_missing_retained_repository_and_query_mismatch_do_not_fallback(self):
        self.retain()
        (self.recipe / "repo.yaml").unlink()
        with self.assertRaisesRegex(PACE.DiscoveryError, "lacks repo.yaml"):
            self.discover_recipe()
        self.assertEqual(self.calls, [])
        (self.recipe / "repo.yaml").write_text("repo:\n  namespace: builtin\n")
        self.scoped_recipe_output = str(self.store) + "\n"
        (self.store / "packages").mkdir()
        (self.store / "repo.yaml").write_text("repo:\n  namespace: builtin\n")
        with self.assertRaisesRegex(PACE.DiscoveryError, "refusing ambient fallback"):
            self.discover_recipe()

    def test_subprocess_failure_and_timeout_are_bounded_actionable_errors(self):
        failed = subprocess.CompletedProcess([], 1, "", "No package matches")
        with patch.object(PACE.subprocess, "run", return_value=failed), self.assertRaisesRegex(PACE.DiscoveryError, "No package matches"):
            PACE.query(self.tool, ["find", "--json", "missing@0"], self.environment)
        with patch.object(PACE.subprocess, "run", side_effect=subprocess.TimeoutExpired([], 30)), self.assertRaisesRegex(PACE.DiscoveryError, "timed out"):
            PACE.query(self.tool, ["find", "--json", "missing@0"], self.environment)

    def test_cli_requires_anchor_before_running_spack(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--spack", str(self.tool), "--profile", "cpu"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--profile requires --anchor", result.stderr)

    def test_cli_recipe_scope_arguments_are_paired_and_recipe_only(self):
        for arguments in (("--recipe-root", "--site-root", str(self.site_root)),
                          ("--recipe-root", "--recipe-profile", "cpu"),
                          ("--profile", "cpu", "--anchor", str(self.anchor), "--recipe-override", str(self.recipe))):
            with self.subTest(arguments=arguments):
                result = subprocess.run([sys.executable, str(SCRIPT), "--spack", str(self.tool), *arguments],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn("recipe", result.stderr)
                self.assertNotIn("Spack metadata query", result.stderr)


if __name__ == "__main__":
    unittest.main()
