#!/usr/bin/env python3
"""Discover loaded Delta compiler/CMake/CUDA tools without querying Spack stores.

The selected system OpenMPI is an external; numerical dependencies are built
in the private site environment. The
site module paths are measured from exposed executables, never reconstructed
from a shared Spack database or installation naming convention.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import subprocess
import sys

CONFIG_PATH = Path(__file__).resolve().parents[2] / "cmake/sites/delta.json"
PROFILES = ("cpu", "cuda-sm80")


class DiscoveryError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise DiscoveryError(message)


def executable(value, name):
    value = value or shutil.which(name)
    require(value, f"loaded Delta modules do not expose {name}")
    path = Path(value)
    require(path.is_absolute(), f"{name} executable must be absolute")
    # Preserve the invocation name of legitimate module wrapper symlinks.
    path = Path(os.path.normpath(path))
    require(path.is_file() and os.access(path, os.X_OK), f"unavailable {name}: {path}")
    return path


def query(command, run=subprocess.run, environment=None):
    try:
        result = run([str(arg) for arg in command], capture_output=True, text=True,
                     stdin=subprocess.DEVNULL, timeout=30, env=environment)
    except subprocess.TimeoutExpired as error:
        raise DiscoveryError(f"tool query timed out: {command[0]}") from error
    require(result.returncode == 0,
            f"tool query failed: {command[0]}: {result.stderr.strip()[:500]}")
    return result.stdout.strip()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def installed_metadata(prefix, name):
    """Read only the selected installation's own metadata, never a store DB."""
    path = prefix / ".spack/spec.json"
    document = json.loads(path.read_text())
    content = document.get("spec", document)
    nodes = content.get("nodes", [content])
    matches = [node for node in nodes if node.get("name") == name]
    require(len(matches) == 1, f"{path} must identify exactly one {name}")
    return matches[0], file_hash(path)


def discover_config(profile, spack=None, anchor=None, config=None, environ=None,
                    cmake=None, cuda=None, mpi=None, openblas_prefix=None, run=subprocess.run):
    require(profile in PROFILES, f"unsupported Delta profile: {profile}")
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text()) if config is None else config)
    environment = dict(os.environ if environ is None else environ)
    cc = executable(anchor, "gcc")
    compiler_tools = {"c": cc}
    for key, name in (("cxx", "g++"), ("fortran", "gfortran")):
        # Require the same module installation, not an ambient compiler.
        compiler_tools[key] = executable(str(cc.parent / name), name)
    for key, path in compiler_tools.items():
        version = query([path, "-dumpfullversion", "-dumpversion"], run, environment)
        require(version == config["compiler"]["version"],
                f"Delta {key} version={version!r}; required {config['compiler']['version']}")
    config["compiler"].update(prefix=str(cc.parent.parent.resolve()),
        executables={key: str(path) for key, path in compiler_tools.items()},
        sha256={key: file_hash(path) for key, path in compiler_tools.items()})
    cmake_tool = executable(cmake, "cmake")
    version = query([cmake_tool, "--version"], run, environment)
    match = re.search(r"^cmake version ([0-9.]+)", version)
    require(match and match[1] == config["cmake"]["version"],
            f"loaded CMake must be {config['cmake']['version']}")
    config["cmake"].update(prefix=str(cmake_tool.resolve().parent.parent),
                           executable=str(cmake_tool), sha256=file_hash(cmake_tool))
    blas_prefix = openblas_prefix or environment.get("OPENBLAS_HOME")
    require(blas_prefix, "openblas/0.3.29 must expose OPENBLAS_HOME or an explicit prefix")
    blas_prefix = Path(blas_prefix)
    require(blas_prefix.is_absolute(), "OpenBLAS prefix must be absolute")
    blas_prefix = blas_prefix.resolve(strict=True)
    node, metadata_hash = installed_metadata(blas_prefix, "openblas")
    policy = config["openblas"]
    target = node.get("arch", {}).get("target")
    target = target.get("name") if isinstance(target, dict) else target
    require(str(node.get("version")) == policy["version"] and target == policy["target"],
            "exposed OpenBLAS version/target differs from the Delta policy")
    for key, expected in policy["parameters"].items():
        actual = node.get("parameters", {}).get(key)
        if isinstance(actual, list) and not isinstance(expected, bool):
            actual = actual[0] if len(actual) == 1 else actual
        require(actual == expected and (not isinstance(expected, bool) or actual is expected),
                f"exposed OpenBLAS {key} differs from the Delta policy")
    policy.update(prefix=str(blas_prefix), metadata_sha256=metadata_hash,
                  binary_build_provenance={key: node[key] for key in ("arch", "compiler", "annotations") if key in node})
    mpi_wrapper = executable(mpi, "mpicxx")
    ompi = executable(str(mpi_wrapper.parent / "ompi_info"), "ompi_info")
    version = query([ompi, "--version"], run, environment)
    selected = config["profiles"][profile]
    match = re.search(r"Open MPI v?([0-9][0-9A-Za-z.]+)", version)
    require(match and match[1] == selected["mpi_version"],
            f"system OpenMPI must be {selected['mpi_version']}")
    mpi_prefix = query([ompi, "--parsable", "--path", "prefix"], run, environment)
    match = re.search(r"^path:prefix:(.+)$", mpi_prefix, re.MULTILINE)
    require(match, "ompi_info must identify its real installation prefix")
    prefix = Path(match[1]).resolve(strict=True)
    require(prefix.is_dir() and (prefix / "bin/mpicxx").resolve() == mpi_wrapper.resolve(),
            "MPI wrapper differs from ompi_info installation")
    commands = {}
    for name, key in (("mpicc", "c"), ("mpicxx", "cxx"), ("mpifort", "fortran")):
        wrapper = executable(str(prefix / "bin" / name), name)
        command = query([wrapper, "--showme:command"], run, environment)
        argv = shlex.split(command)
        require(len(argv) == 1, f"{name} must select one unambiguous compiler executable")
        actual = shutil.which(argv[0], path=environment.get("PATH"))
        require(actual and Path(actual).resolve() == compiler_tools[key].resolve(),
                f"{name} must select the admitted GCC compiler; set OMPI_CC/CXX/FC")
        commands[name] = command
    metadata = query([ompi, "--parsable", "--all"], run, environment)
    require(re.search(r"mpi_built_with_cuda_support:value:(true|yes|1)$", metadata,
                      re.MULTILINE | re.IGNORECASE), "system OpenMPI does not report CUDA support")
    selected.update(mpi_prefix=str(prefix), mpi_wrapper=str(mpi_wrapper),
                    mpi_wrapper_commands=commands, mpi_metadata_sha256=hashlib.sha256(metadata.encode()).hexdigest(),
                    mpi_build_provenance=[line for line in metadata.splitlines()
                        if re.search(r"(?:compiler|build:|config:)", line, re.IGNORECASE)])
    # CPU HYPRE/MFEM remain CPU-only even though the selected external MPI
    # itself has CUDA capability and may carry a CUDA dependency edge.
    if selected["mpi_parameters"]["cuda"] or profile == "cuda-sm80":
        nvcc = executable(cuda, "nvcc")
        version = query([nvcc, "--version"], run, environment)
        match = re.search(r"\bV([0-9]+\.[0-9]+\.[0-9]+)\b", version)
        require(match and match[1] == config["cuda_compiler_version"],
                f"loaded nvcc must be {config['cuda_compiler_version']}; load cuda/12.9 last")
        prefix = nvcc.resolve().parent.parent
        version_file = prefix / "version.json"
        version_document = json.loads(version_file.read_text())
        sdk_version = version_document.get("cuda", {}).get("version")
        require(sdk_version == config["cuda_version"],
                "CUDA SDK version.json differs from the retained Delta toolkit policy")
        require((prefix / "include/cuda.h").is_file(), "nvcc prefix lacks CUDA headers")
        require((prefix / "lib64").is_dir() or (prefix / "targets/x86_64-linux/lib").is_dir(),
                "nvcc prefix lacks CUDA libraries")
        config.update(cuda_prefix=str(prefix), cuda_executable=str(nvcc),
                      cuda_sha256=file_hash(nvcc), cuda_version_file_sha256=file_hash(version_file))
    return config


def recipe_path(value):
    path = Path(value)
    require(path.is_absolute(), "builtin recipe path must be absolute")
    path = path.resolve(strict=True)
    require((path / "repo.yaml").is_file() and (path / "packages").is_dir(),
            f"builtin recipe root lacks repo.yaml/packages: {path}")
    return path


def discover_recipe_root(spack, site_root, profile, override=None, environ=None,
                         run=subprocess.run):
    require(profile in PROFILES, "recipe discovery requires a supported profile")
    site_root = Path(site_root)
    require(site_root.is_absolute(), "site root must be absolute")
    site = site_root.resolve() / "site/delta"
    retained = []
    metadata = [site / "spack-user/repos.yaml"]
    metadata += [site / "spack-envs" / name / "spack.yaml" for name in PROFILES]
    for path in metadata:
        if path.exists() or path.is_symlink():
            require(path.is_file() and not path.is_symlink(), f"invalid retained repository: {path}")
            document = json.loads(path.read_text())
            retained.append(recipe_path(document.get("spack", document)["repos"]["builtin"]))
    require(len(set(retained)) <= 1, "retained Delta recipe authorities conflict; use a fresh site root")
    if override:
        selected = recipe_path(override)
    elif retained:
        selected = retained[0]
    else:
        tool = executable(spack, "spack")
        require(tool.parent.name == "bin", "supplied Spack must be a bin/spack executable")
        environment = dict(os.environ if environ is None else environ)
        for name in ("SPACK_ENV", "SPACK_ROOT", "SPACK_DISABLE_LOCAL_CONFIG"):
            environment.pop(name, None)
        environment.update(SPACK_USER_CONFIG_PATH=str(site / "spack-user"),
                           SPACK_SYSTEM_CONFIG_PATH=str(site / "spack-system"),
                           SPACK_USER_CACHE_PATH=str(site_root / "spack-cache"))
        # This reads repository configuration only, never find/location -i or a DB.
        location = query([tool, "location", "--repo", "builtin"], run, environment)
        require(len(location.splitlines()) == 1, "builtin repository query must return one path")
        selected = recipe_path(location)
    require(not retained or selected == retained[0],
            "builtin override changes retained recipes; use a fresh site root")
    return str(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--profile", choices=PROFILES)
    mode.add_argument("--recipe-root", action="store_true")
    parser.add_argument("--spack", help="private build Spack; no shared DB queries")
    parser.add_argument("--anchor", help="gcc executable exposed by gcc-native/14")
    parser.add_argument("--cmake")
    parser.add_argument("--cuda")
    parser.add_argument("--mpi", help="system OpenMPI C++ wrapper")
    parser.add_argument("--openblas-prefix", help="selected module's OpenBLAS installation")
    parser.add_argument("--site-root")
    parser.add_argument("--recipe-profile", choices=PROFILES)
    parser.add_argument("--recipe-override")
    args = parser.parse_args()
    try:
        if args.recipe_root:
            require(args.site_root and args.recipe_profile, "recipe discovery requires site root/profile")
            print(discover_recipe_root(args.spack, args.site_root, args.recipe_profile,
                                      args.recipe_override))
        else:
            print(json.dumps(discover_config(args.profile, args.spack, args.anchor,
                                            cmake=args.cmake, cuda=args.cuda, mpi=args.mpi,
                                            openblas_prefix=args.openblas_prefix), indent=2))
    except (DiscoveryError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: Delta discovery: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
