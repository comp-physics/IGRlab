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
        if arguments == ["location", "--repo", "builtin"]:
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
                if profile == "cuda-sm80":
                    self.assertEqual(stripped.pop("cuda_prefix"), str(self.prefixes["cuda"]))
                self.assertEqual(stripped, self.config)
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
                 ("openblas", "threads", "openmp", "cpu"), ("openmpi-gpu", "cuda", False, "cuda-sm80"))
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


if __name__ == "__main__":
    unittest.main()
