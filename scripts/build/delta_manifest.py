#!/usr/bin/env python3
"""Render/check private Delta dependency environments without building anything."""
import argparse
import copy
import json
from pathlib import Path
import re
import sys

CONFIG_PATH = Path(__file__).resolve().parents[2] / "cmake/sites/delta.json"
PROFILES = ("cpu", "cuda-sm80")


class DeltaError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise DeltaError(message)


def load_config():
    return json.loads(CONFIG_PATH.read_text())


def absolute_path(value, label):
    require(isinstance(value, (str, Path)), f"{label} must identify a path")
    path = Path(value)
    require(path.is_absolute() and path == path.resolve(),
            f"{label} must be a normalized absolute path: {path}")
    return path


def invocation_path(value, label):
    # Module tools can be wrapper symlinks (notably HPCX's env.sh). Preserve
    # their invocation names; executable contents are bound by discovery.
    require(isinstance(value, (str, Path)), f"{label} must identify a path")
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts,
            f"{label} must be a normalized absolute invocation path")
    return path


def resolved_config(config, profile):
    require(profile in PROFILES, f"unsupported Delta profile: {profile}")
    config = copy.deepcopy(config)
    candidate = copy.deepcopy(config)
    compiler = candidate["compiler"]
    absolute_path(compiler.pop("prefix"), "compiler prefix")
    paths, hashes = compiler.pop("executables"), compiler.pop("sha256")
    require(set(paths) == set(hashes) == {"c", "cxx", "fortran"}, "compiler tools are incomplete")
    for name, path in paths.items():
        invocation_path(path, f"compiler {name}")
        require(re.fullmatch(r"[0-9a-f]{64}", hashes[name]), f"compiler {name} hash is malformed")
    cmake = candidate["cmake"]
    absolute_path(cmake.pop("prefix"), "CMake prefix")
    invocation_path(cmake.pop("executable"), "CMake executable")
    require(re.fullmatch(r"[0-9a-f]{64}", cmake.pop("sha256")), "CMake hash is malformed")
    blas = candidate["openblas"]
    absolute_path(blas.pop("prefix"), "OpenBLAS prefix")
    require(re.fullmatch(r"[0-9a-f]{64}", blas.pop("metadata_sha256")),
            "OpenBLAS installation metadata hash is malformed")
    require(isinstance(blas.pop("binary_build_provenance"), dict),
            "OpenBLAS binary provenance must be observed installation metadata")
    selected = candidate["profiles"][profile]
    absolute_path(selected.pop("mpi_prefix"), "MPI prefix")
    invocation_path(selected.pop("mpi_wrapper"), "MPI wrapper")
    commands = selected.pop("mpi_wrapper_commands")
    require(isinstance(commands, dict) and set(commands) == {"mpicc", "mpicxx", "mpifort"},
            "MPI compiler wrapper observations are incomplete")
    require(re.fullmatch(r"[0-9a-f]{64}", selected.pop("mpi_metadata_sha256")),
            "MPI metadata hash is malformed")
    provenance = selected.pop("mpi_build_provenance")
    require(isinstance(provenance, list) and all(isinstance(item, str) for item in provenance),
            "MPI build provenance must be observed text")
    absolute_path(candidate.pop("cuda_prefix"), "CUDA prefix")
    invocation_path(candidate.pop("cuda_executable"), "CUDA executable")
    require(re.fullmatch(r"[0-9a-f]{64}", candidate.pop("cuda_sha256")), "CUDA hash is malformed")
    require(re.fullmatch(r"[0-9a-f]{64}", candidate.pop("cuda_version_file_sha256")),
            "CUDA toolkit version file hash is malformed")
    require(candidate == load_config(),
            "discovered configuration changes the checked-in Delta policy")
    return config


def manifest(root, profile, recipe_root, config=None):
    config = resolved_config(load_config() if config is None else config, profile)
    root, recipe_root = absolute_path(root, "root"), absolute_path(recipe_root, "recipes")
    selected, gcc, target = config["profiles"][profile], config["compiler"], config["target"]
    common = f'target={target} %gcc@{gcc["version"]}'
    mpi_spec = f'openmpi@{selected["mpi_version"]} +cuda {common}'
    packages = {
        "all": {"providers": {"mpi": ["openmpi"], "blas": ["openblas"],
                 "lapack": ["openblas"], "zlib-api": ["zlib"]}, "require": [f"target={target}"]},
        "gcc": {"buildable": False, "externals": [{
            "spec": f'gcc@{gcc["version"]} languages:=c,c++,fortran target={target}',
            "prefix": gcc["prefix"], "extra_attributes": {"compilers": gcc["executables"]}}]},
        "cmake": {"buildable": False, "externals": [{
            "spec": f'cmake@{config["cmake"]["version"]} target={config["cmake"]["target"]}',
            "prefix": config["cmake"]["prefix"]}]},
        "openblas": {"buildable": False,
            "require": [f'target={config["openblas"]["target"]}'], "externals": [{
            "spec": f'openblas@{config["openblas"]["version"]} +shared ~ilp64 threads=none target={config["openblas"]["target"]}',
            "prefix": config["openblas"]["prefix"]}]},
        "openmpi": {"buildable": False, "externals": [{
            "spec": mpi_spec, "prefix": selected["mpi_prefix"]}]},
        "cuda": {"buildable": False, "externals": [{
            "spec": f'cuda@{config["cuda_version"]} target={target}', "prefix": config["cuda_prefix"]}]},
        "hypre": {"buildable": True, "externals": [], "require": [f"target={target}",
            "precision=double", f'^openmpi@{selected["mpi_version"]}']},
    }
    packages["cmake"]["require"] = [f'target={config["cmake"]["target"]}']
    for name in ("metis", "umpire", "camp", "zlib"):
        packages[name] = {"buildable": True, "externals": []}
    specs = [packages[name]["externals"][0]["spec"] for name in ("cmake", "openmpi", "cuda", "openblas")]
    specs.extend([f'metis@{config["metis_version"]} ~int64 ~real64 +shared {common}',
                  f'zlib@{config["zlib_version"]} {common}'])
    hypre = f'hypre@{config["hypre_version"]} precision=double ~complex +shared +mpi ~openmp cxxstd=17 '
    if profile == "cuda-sm80":
        specs.extend([hypre + f'+cuda +umpire +gpu-aware-mpi ~unified-memory cuda_arch=80 {common}',
            f'umpire@{config["umpire_version"]} +shared +c +cuda ~fortran ~examples cuda_arch=80 {common}'])
    else:
        specs.append(hypre + f'~cuda ~umpire {common}')
    return {"spack": {
        "specs": specs, "packages": packages, "repos": {"builtin": str(recipe_root)}, "view": True,
        "concretizer": {"reuse": False, "unify": True,
            "targets": {"granularity": "generic", "host_compatible": True},
            "splice": {"automatic": False, "explicit": []}},
        "config": {"install_tree": {"root": str(root / "spack-store" / profile)},
            "build_stage": [str(root / "spack-stage" / profile)],
            "source_cache": str(root / "spack-source-cache"),
            "misc_cache": str(root / "spack-cache/misc"), "build_jobs": 4},
        "bootstrap": {"root": str(root / "spack-bootstrap"), "enable": True},
        "upstreams:": {},
    }}


def write_environment(root, profile, recipe_root, config=None):
    root, recipe_root = absolute_path(root, "root"), absolute_path(recipe_root, "recipes")
    config = resolved_config(load_config() if config is None else config, profile)
    documents = {
        root / "site/delta/spack-envs" / profile / "spack.yaml": manifest(root, profile, recipe_root, config),
        root / "site/delta/spack-envs" / profile / "site-packages.json": config,
        root / "site/delta/spack-user/repos.yaml": {"repos": {"builtin": str(recipe_root)}},
    }
    rendered = {path: json.dumps(value, indent=2) + "\n" for path, value in documents.items()}
    for path, value in rendered.items():
        require(not any(item.is_symlink() for item in (path, *path.parents)),
                f"refusing symlinked Delta metadata: {path}")
        require(not path.exists() or (path.is_file() and path.read_text() == value),
                f"retained Delta configuration differs: {path}; use a fresh site root")
    for path, value in rendered.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("x") as stream:
                    stream.write(value)
            except FileExistsError:
                require(not path.is_symlink() and path.read_text() == value,
                        f"Delta configuration changed concurrently: {path}")
    return tuple(rendered)


def check_lock(lock, profile, config=None):
    config = resolved_config(load_config() if config is None else config, profile)
    selected, gpu = config["profiles"][profile], profile == "cuda-sm80"
    nodes, roots = lock.get("concrete_specs"), lock.get("roots")
    require(isinstance(nodes, dict) and nodes and isinstance(roots, list) and roots,
            "lock requires a nonempty concrete graph and roots")
    rooted = {item["hash"] for item in roots}
    require(rooted <= nodes.keys(), "lock root is missing from concrete graph")
    for node in nodes.values():
        target = node["arch"]["target"]
        target = target.get("name") if isinstance(target, dict) else target
        expected_target = (config[node["name"]]["target"]
                           if node["name"] in ("cmake", "openblas") else config["target"])
        require(target == expected_target, f'{node["name"]} changes the Delta target')
        require(all(item["hash"] in nodes for item in node.get("dependencies", [])),
                f'{node["name"]} references a dependency outside the graph')

    def one(name, version, root=True):
        matches = [(identity, node) for identity, node in nodes.items() if node["name"] == name]
        require(len(matches) == 1, f"exactly one concrete {name} is required")
        identity, node = matches[0]
        require(str(node["version"]) == version and (not root or identity in rooted),
                f"{name} version/root differs from Delta policy")
        return identity, node

    def params(node, expected):
        for key, value in expected.items():
            actual = node.get("parameters", {}).get(key)
            if isinstance(actual, (list, tuple)) and not isinstance(value, bool):
                actual = actual[0] if len(actual) == 1 else actual
            require(actual == value and (not isinstance(value, bool) or actual is value),
                    f'{node["name"]} {key} differs from Delta policy')

    def edge(node, identity):
        require(any(item["hash"] == identity for item in node.get("dependencies", [])),
                f'{node["name"]} does not bind the selected dependency identity')

    gcc_id, gcc = one("gcc", config["compiler"]["version"], False)
    mpi_id, mpi = one("openmpi", selected["mpi_version"])
    _, cmake = one("cmake", config["cmake"]["version"])
    cuda_id, cuda = one("cuda", config["cuda_version"])
    for node, prefix in ((gcc, config["compiler"]["prefix"]), (mpi, selected["mpi_prefix"]),
                         (cmake, config["cmake"]["prefix"]), (cuda, config["cuda_prefix"])):
        require(node.get("external", {}).get("path") == prefix,
                f'{node["name"]} differs from its observed external installation')
    params(mpi, {"cuda": True})
    blas_id, blas = one("openblas", config["openblas"]["version"])
    require(blas.get("external", {}).get("path") == config["openblas"]["prefix"],
            "OpenBLAS differs from the selected NCSA installation")
    _, metis = one("metis", config["metis_version"])
    _, zlib = one("zlib", config["zlib_version"])
    hypre_id, hypre = one("hypre", config["hypre_version"])
    for node in (metis, zlib, hypre):
        require(not node.get("external"), f'{node["name"]} must be built in the private Delta graph')
        edge(node, gcc_id)
    params(blas, {"threads": "none", "shared": True, "ilp64": False})
    params(metis, {"int64": False, "real64": False, "shared": True})
    params(hypre, {"precision": "double", "complex": False, "shared": True, "mpi": True,
                   "cuda": gpu, "umpire": gpu, "cxxstd": "17", "openmp": False})
    edge(hypre, mpi_id)
    edge(hypre, blas_id)
    if gpu:
        umpire_id, umpire = one("umpire", config["umpire_version"])
        require(not umpire.get("external"), "Umpire must be built in the private graph")
        params(hypre, {"cuda_arch": "80", "gpu-aware-mpi": True, "unified-memory": False})
        params(umpire, {"cuda_arch": "80", "cuda": True, "shared": True,
                        "c": True, "fortran": False, "examples": False})
        edge(hypre, cuda_id)
        edge(hypre, umpire_id)
        edge(umpire, cuda_id)
        edge(umpire, gcc_id)
    else:
        require(not any(node["name"] == "umpire" for node in nodes.values()),
                "CPU HYPRE graph must not enable Umpire")
    return {"profile": profile, "target": config["target"], "hypre_hash": hypre_id,
            "mpi_hash": mpi_id, "mpi_external_cuda": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--recipe-root", type=Path)
    parser.add_argument("--site-config")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--check-lock", type=Path)
    args = parser.parse_args()
    try:
        site_config = args.site_config
        if site_config is None and args.check_lock:
            site_config = args.check_lock.parent / "site-packages.json"
        require(site_config is not None, "rendering requires observed --site-config")
        config = json.load(sys.stdin) if site_config == "-" else json.loads(Path(site_config).read_text())
        if args.check_lock:
            print(json.dumps(check_lock(json.loads(args.check_lock.read_text()), args.profile, config), sort_keys=True))
        else:
            require(args.root is not None and args.recipe_root is not None, "rendering requires root/recipe root")
            if args.dry_run:
                print(json.dumps(manifest(args.root, args.profile, args.recipe_root, config), indent=2))
            else:
                for path in write_environment(args.root, args.profile, args.recipe_root, config):
                    print(path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: Delta manifest: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
