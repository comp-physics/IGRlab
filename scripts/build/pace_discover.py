#!/usr/bin/env python3
"""Discover installed PACE externals through read-only, exact Spack queries.

The loaded GCC module identifies the shared store. The selected Spack1.3
reader queries that store with a command-line override, before the build's
private configuration scopes are enabled. No install tree is edited.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys

CONFIG_PATH = Path(__file__).resolve().parents[2] / "cmake/sites/pace.json"
PROFILES = ("cpu", "cuda-sm80")
ISOLATED_VARIABLES = ("SPACK_ENV", "SPACK_USER_CONFIG_PATH", "SPACK_SYSTEM_CONFIG_PATH",
                      "SPACK_USER_CACHE_PATH", "SPACK_DISABLE_LOCAL_CONFIG", "SPACK_ROOT")


class DiscoveryError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise DiscoveryError(message)


def existing_path(value, label, directory=False):
    require(bool(value), f"{label} is empty")
    path = Path(value)
    require(path.is_absolute(), f"{label} must be absolute: {value}")
    path = path.resolve(strict=True)
    require(path.is_dir() if directory else path.is_file(), f"{label} has the wrong file type: {path}")
    return path


def reader_environment(spack, environ=None, isolated=True):
    environment = dict(os.environ if environ is None else environ)
    if isolated:
        for variable in ISOLATED_VARIABLES:
            environment.pop(variable, None)
    # An inherited SPACK_ROOT must never change which supplied reader runs.
    environment["SPACK_ROOT"] = str(spack.parent.parent)
    return environment


def query(spack, arguments, environment):
    command = [str(spack), *arguments]
    try:
        result = subprocess.run(command, env=environment, capture_output=True,
                                text=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        raise DiscoveryError(f"Spack metadata query timed out: {' '.join(arguments)}") from error
    require(result.returncode == 0,
            f"Spack metadata query failed ({' '.join(arguments)}): {result.stderr.strip()[:1000]}")
    return result.stdout


def admitted_reader(spack, environ=None, isolated=True):
    spack = existing_path(spack, "--spack")
    require(spack.name == "spack" and spack.parent.name == "bin" and os.access(spack, os.X_OK),
            "--spack must identify an executable bin/spack in the selected tool installation")
    environment = reader_environment(spack, environ, isolated)
    version = query(spack, ["--version"], environment).strip()
    require(re.fullmatch(r"1\.3\.\S+", version),
            f"PACE discovery requires the compatible Spack1.3 reader; found {version!r}")
    return spack, environment


def discover_store(anchor):
    anchor = existing_path(anchor, "--anchor")
    require(anchor.name == "gcc" and anchor.parent.name == "bin",
            "--anchor must be the gcc executable exposed by the loaded PACE compiler module")
    stores = [parent for parent in anchor.parents if (parent / ".spack-db/index.json").is_file()]
    require(len(stores) == 1, "loaded GCC anchor must have exactly one shared Spack database ancestor")
    store = stores[0]
    # A newer reader automatically upgrades an older local DB during a query.
    # Refuse that path before invoking Spack against this shared install tree.
    database = json.loads((store / ".spack-db/index.json").read_text())
    require(database.get("database", {}).get("version") == "8",
            "shared store must use database version 8 for read-only Spack 1.3 discovery; refusing a possible reindex")
    return store, anchor.parent.parent


def parameter_matches(observed, expected):
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(observed, list):
        return observed == ([expected] if not isinstance(expected, list) else expected)
    return observed == expected


def validate_record(record, name, version, target, parameters=None, gcc_version=None, gcc_hash=None):
    require(isinstance(record, dict), f"{name} metadata record is not an object")
    require(record.get("name") == name and str(record.get("version")) == version,
            f"{name} metadata name/version differs from the exact policy selector")
    observed_target = record.get("arch", {}).get("target")
    if isinstance(observed_target, dict):
        observed_target = observed_target.get("name")
    require(observed_target == target, f"{name} target={observed_target!r}; required {target}")
    require(re.fullmatch(r"[a-z2-7]{32}", record.get("hash", "")) is not None,
            f"{name} metadata has no complete concrete DAG identity")
    for key, expected in (parameters or {}).items():
        require(parameter_matches(record.get("parameters", {}).get(key), expected),
                f"{name} {key} differs from the required value {expected!r}")
    if gcc_version is not None:
        compiler = record.get("compiler")
        annotation = record.get("annotations", {}).get("compiler")
        edges = [edge for edge in record.get("dependencies", []) if edge.get("name") == "gcc"]
        require(compiler is not None or annotation is not None or bool(edges),
                f"{name} metadata does not identify its GCC compiler")
        if compiler is not None:
            require(isinstance(compiler, dict) and compiler.get("name") == "gcc" and
                    str(compiler.get("version")) == gcc_version, f"{name} compiler differs from GCC{gcc_version}")
        if annotation is not None:
            require(annotation in (f"gcc@{gcc_version}", f"gcc@={gcc_version}"),
                    f"{name} compiler annotation differs from GCC{gcc_version}")
        if edges:
            require(len(edges) == 1 and edges[0].get("hash") == gcc_hash,
                    f"{name} compiler dependency differs from the selected GCC DAG")


def module_metis_prefix(environment, store):
    prefixes = [existing_path(environment[key], key, directory=True)
                for key in ("METIS_ROOT", "METISROOT") if environment.get(key)]
    require(not prefixes or len(set(prefixes)) == 1, "loaded METIS_ROOT/METISROOT conflict")
    if prefixes:
        require(prefixes[0].is_relative_to(store), "loaded METIS module prefix escapes the shared store")
    return prefixes[0] if prefixes else None


def installed_package(spack, environment, store, specification, name, version, target,
                      parameters=None, gcc_version=None, gcc_hash=None, module_prefix=None):
    override = ["-c", f"config:install_tree:root:{store}"]
    records = json.loads(query(spack, [*override, "find", "--json", specification], environment))
    require(isinstance(records, list) and records, f"no installed {name} matches {specification}")
    # Validate identity fields for every returned candidate before interpreting
    # paths. Variant/compiler admission follows the explicit module selection.
    for record in records:
        validate_record(record, name, version, target)
    identities = [record["hash"] for record in records]
    require(len(set(identities)) == len(identities), f"duplicate {name} DAG identities in Spack metadata")
    require(len(records) == 1 or module_prefix is not None,
            f"ambiguous installed {name}: {len(records)} matches; load the configured module to identify its prefix")
    candidates = []
    for record in records:
        location = query(spack, [*override, "location", "-i", f'/{record["hash"]}'], environment)
        require(len(location.strip().splitlines()) == 1, f"{name} location returned multiple paths")
        prefix = existing_path(location.strip(), f"{name} installed prefix", directory=True)
        require(prefix.is_relative_to(store), f"{name} installed prefix escapes the shared store")
        recorded_prefix = record.get("prefix") or (record.get("external") or {}).get("path")
        if recorded_prefix:
            require(existing_path(recorded_prefix, f"{name} recorded prefix", directory=True) == prefix,
                    f"{name} find/location prefix mismatch")
        if module_prefix is None or prefix == module_prefix:
            candidates.append((record, prefix))
    require(len(candidates) == 1, f"loaded {name} module prefix must select exactly one installed DAG")
    record, prefix = candidates[0]
    validate_record(record, name, version, target, parameters, gcc_version, gcc_hash)
    return str(prefix), record["hash"]


def discover_config(profile, spack, anchor, config=None, environ=None):
    require(profile in PROFILES, f"unsupported PACE profile: {profile}")
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text()) if config is None else config)
    spack, environment = admitted_reader(spack, environ)
    store, compiler_anchor = discover_store(anchor)
    target, gcc = config["target"], config["compiler"]["version"]
    prefix, gcc_hash = installed_package(spack, environment, store, f"gcc@={gcc} target={target}",
        "gcc", gcc, target, {"languages": ["c", "c++", "fortran"]}, module_prefix=compiler_anchor)
    require(Path(prefix) == compiler_anchor, "selected GCC prefix differs from the loaded compiler anchor")
    config["compiler"]["prefix"] = prefix
    common = f"target={target} %gcc@={gcc}"
    for name in ("cmake", "metis", "openblas"):
        version = config[name]["version"]
        variants = "threads=none +shared ~ilp64 " if name == "openblas" else ""
        parameters = {"threads": "none", "shared": True, "ilp64": False} if name == "openblas" else None
        module_prefix = module_metis_prefix(environment, store) if name == "metis" else None
        if name == "metis":
            parameters = config[name]["parameters"]
        config[name]["prefix"], _ = installed_package(spack, environment, store,
            f"{name}@={version} {variants}{common}", name, version, target,
            parameters, gcc, gcc_hash, module_prefix)
    selected = config["profiles"][profile]
    mpi_spec = selected["mpi_spec"].replace(f'openmpi@{selected["mpi_version"]} ',
        f'openmpi@={selected["mpi_version"]} ', 1).replace(f"%gcc@{gcc}", f"%gcc@={gcc}")
    selected["mpi_prefix"], _ = installed_package(spack, environment, store, mpi_spec,
        "openmpi", selected["mpi_version"], target, selected["mpi_parameters"], gcc, gcc_hash)
    if profile == "cuda-sm80":
        version = config["cuda_version"]
        config["cuda_prefix"], _ = installed_package(spack, environment, store,
            f"cuda@={version} target={target}", "cuda", version, target)
    return config


def recipe_path(value, label):
    require(isinstance(value, str) and bool(value), f"{label} must identify the builtin repository")
    try:
        root = existing_path(value, label, directory=True)
    except OSError as error:
        raise DiscoveryError(f"{label} is unavailable: {value}") from error
    require((root / "repo.yaml").is_file() and (root / "packages").is_dir(),
            f"{label} lacks repo.yaml or packages: {root}")
    return root


def retained_recipe_scope(site_root, profile):
    """Inspect only the JSON/YAML-subset metadata written by pace_manifest.py."""
    require(profile in PROFILES, "retained recipe discovery requires a supported --recipe-profile")
    root = Path(site_root)
    require(root.is_absolute(), "--site-root must be absolute")
    site = root.resolve() / "site/pace"
    authorities, environments = [], {}
    for selected in PROFILES:
        environment = site / "spack-envs" / selected
        if environment.exists() or environment.is_symlink():
            environments[selected] = environment
            authorities.append((environment / "spack.yaml", "spack"))
    user_scope = site / "spack-user"
    if user_scope.exists() or user_scope.is_symlink():
        authorities.append((user_scope / "repos.yaml", None))
    repositories = []
    for metadata, section in authorities:
        require(metadata.is_file() and not metadata.is_symlink(),
                f"retained repository metadata is missing or not a regular file: {metadata}")
        try:
            document = json.loads(metadata.read_text())
            config = document[section] if section else document
            value = config["repos"]["builtin"]
        except (ValueError, KeyError, TypeError) as error:
            raise DiscoveryError(f"malformed retained repository metadata: {metadata}") from error
        repositories.append(recipe_path(value, f"retained builtin repository in {metadata}"))
    require(len(set(repositories)) <= 1,
            "retained profile/shared builtin repositories conflict; choose a new --site-root")
    environment = environments.get(profile)
    if environment is None and not user_scope.exists() and environments:
        environment = next(iter(environments.values()))
    return (repositories[0] if repositories else None), environment, site


def discover_recipe_root(spack, environ=None, site_root=None, profile=None, override=None):
    retained, selected, site = retained_recipe_scope(site_root, profile) if site_root else (None, None, None)
    if override:
        explicit = recipe_path(override, "explicit builtin recipe root")
        require(retained is None or explicit == retained,
                "explicit builtin recipe root conflicts with retained repository metadata; choose a new --site-root")
        if retained is None:
            return str(explicit)
    # A retained graph owns its repository scope; inherited scopes must not
    # select ambient recipes. Fresh roots still query the tool's current scopes.
    spack, environment = admitted_reader(spack, environ, isolated=retained is not None)
    arguments = ["location", "--repo", "builtin"]
    if retained is not None:
        environment.update(SPACK_USER_CONFIG_PATH=str(site / "spack-user"),
                           SPACK_SYSTEM_CONFIG_PATH=str(site / "spack-system"),
                           SPACK_USER_CACHE_PATH=str(site.parent.parent / "spack-cache"))
        if selected is not None:
            arguments = ["-e", str(selected), *arguments]
    location = query(spack, arguments, environment)
    require(len(location.strip().splitlines()) == 1, "builtin repository query must return one path")
    root = recipe_path(location.strip(), "builtin recipe root")
    require(retained is None or root == retained,
            "Spack builtin repository query differs from retained repository metadata; refusing ambient fallback")
    return str(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spack", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--profile", choices=PROFILES)
    mode.add_argument("--recipe-root", action="store_true")
    parser.add_argument("--anchor", help="absolute gcc executable from the loaded PACE module")
    parser.add_argument("--site-root", help="dependency root whose recipe authority must be retained")
    parser.add_argument("--recipe-profile", choices=PROFILES)
    parser.add_argument("--recipe-override", help="explicit builtin repository, subject to retained conflicts")
    args = parser.parse_args()
    if args.profile and not args.anchor:
        parser.error("--profile requires --anchor from the loaded GCC module")
    if args.recipe_root and args.anchor:
        parser.error("--anchor belongs to --profile discovery")
    if (args.site_root or args.recipe_profile or args.recipe_override) and not args.recipe_root:
        parser.error("recipe scope arguments belong to --recipe-root discovery")
    if bool(args.site_root) != bool(args.recipe_profile):
        parser.error("--site-root and --recipe-profile must be supplied together")
    try:
        if args.recipe_root:
            print(discover_recipe_root(args.spack, site_root=args.site_root,
                                      profile=args.recipe_profile, override=args.recipe_override))
        else:
            print(json.dumps(discover_config(args.profile, args.spack, args.anchor), indent=2))
    except (DiscoveryError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: PACE discovery: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
