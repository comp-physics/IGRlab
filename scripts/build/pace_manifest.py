#!/usr/bin/env python3
"""Render a stable PACE Spack environment or inspect its concrete lock, without building."""
import argparse
import copy
import json
from pathlib import Path
import sys

CONFIG_PATH = Path(__file__).resolve().parents[2] / "cmake/sites/pace.json"
PROFILES = ("cpu", "cuda-sm80", "cuda-sm70")


class PaceError(ValueError):
    """An actionable site configuration or dependency mismatch."""


def load_config():
    return json.loads(CONFIG_PATH.read_text())


def profile_config(config, profile):
    """Keep retained CPU/SM80 documents unchanged when adding V100 policy."""
    if profile not in PROFILES:
        raise PaceError(f"unsupported PACE profile: {profile}")
    config = copy.deepcopy(config)
    if profile in ("cpu", "cuda-sm80"):
        config["profiles"].pop("cuda-sm70", None)
    return config


def resolved_config(config, profile):
    """Accept discovered paths only; the repository still owns constraints."""
    policy = profile_config(load_config(), profile)
    config = profile_config(config, profile)
    candidate = copy.deepcopy(config)
    fields = [(candidate[name], "prefix") for name in ("compiler", "cmake", "metis", "openblas")]
    fields.append((candidate["profiles"][profile], "mpi_prefix"))
    if profile != "cpu":
        fields.append((candidate, "cuda_prefix"))
    for owner, key in fields:
        if not owner.get(key):
            raise PaceError("site prefixes must be discovered with pace_discover.py inside a compute allocation")
        absolute_path(owner.pop(key), f"discovered {key}")
    if candidate != policy:
        raise PaceError("discovered site configuration changes the checked-in PACE constraints")
    return config


def absolute_path(path, label):
    path = Path(path)
    if not path.is_absolute() or path != path.resolve():
        raise PaceError(f"{label} must be a normalized absolute path: {path}")
    return path


def external(spec, prefix, **extra):
    return {"buildable": False, "externals": [{"spec": spec, "prefix": prefix, **extra}]}


def manifest(root, profile, recipe_root, config=None):
    """Pure rendering: paths may refer to another host and need not exist."""
    config = resolved_config(load_config() if config is None else config, profile)
    root = absolute_path(root, "--root")
    recipe_root = absolute_path(recipe_root, "--recipe-root")
    if profile not in PROFILES:
        raise PaceError(f"unsupported PACE profile: {profile}")
    gpu = profile != "cpu"
    cuda_arch = profile.removeprefix("cuda-sm") if gpu else None
    selected = config["profiles"][profile]
    target = config["target"]
    gcc = config["compiler"]
    # Constraints after %gcc apply to that compiler dependency in Spack1.3.
    # Put the package target first, including on every external root spec.
    common = f'target={target} %gcc@{gcc["version"]}'
    packages = {
        "all": {"providers": {"mpi": ["openmpi"], "blas": ["openblas"],
                              "lapack": ["openblas"], "zlib-api": ["zlib"]},
                "require": [f"target={target}"]},
        "gcc": external(f'gcc@{gcc["version"]} languages:=c,c++,fortran target={target}', gcc["prefix"],
                        extra_attributes={"compilers": {"c": f'{gcc["prefix"]}/bin/gcc',
                            "cxx": f'{gcc["prefix"]}/bin/g++', "fortran": f'{gcc["prefix"]}/bin/gfortran'}}),
        "openmpi": external(selected["mpi_spec"], selected["mpi_prefix"]),
        # A package-specific require list replaces the all fallback. Keep its
        # target explicit so a host-specific descendant cannot enter HYPRE.
        "hypre": {"buildable": True, "externals": [], "require": [f"target={target}",
                    "precision=double", f'^openmpi@{selected["mpi_version"]}']},
        "umpire": {"buildable": True, "externals": []},
        "camp": {"buildable": True, "externals": []},
        "zlib": {"buildable": True, "externals": []},
    }
    for name in ("cmake", "metis", "openblas"):
        package = config[name]
        variants = "threads=none +shared ~ilp64 " if name == "openblas" else ""
        if name == "metis":
            variants = "~int64 ~real64 +shared "
        packages[name] = external(f'{name}@{package["version"]} {variants}{common}', package["prefix"])
    specs = [packages[name]["externals"][0]["spec"] for name in ("cmake", "openmpi", "metis", "openblas")]
    specs.append(f'zlib@{config["zlib_version"]} {common}')
    hypre = f'hypre@{config["hypre_version"]} precision=double ~complex +shared +mpi cxxstd=17 '
    if gpu:
        packages["cuda"] = external(f'cuda@{config["cuda_version"]}', config["cuda_prefix"])
        specs.extend([f'cuda@{config["cuda_version"]} target={target}',
            hypre + f'+cuda +umpire +gpu-aware-mpi ~unified-memory cuda_arch={cuda_arch} {common}',
            f'umpire@{config["umpire_version"]} +shared +c +cuda ~fortran ~examples cuda_arch={cuda_arch} {common}'])
    else:
        specs.append(hypre + f'~cuda ~umpire {common}')
    return {"spack": {
        "specs": specs, "packages": packages,
        "concretizer": {"reuse": False, "unify": True,
                        "targets": {"granularity": "generic", "host_compatible": True},
                        "splice": {"automatic": False, "explicit": []}},
        "repos": {"builtin": str(recipe_root)}, "view": True,
        "config": {"install_tree": {"root": str(root / "spack-store" / profile)},
                   "build_stage": [str(root / "spack-stage" / profile)],
                   "source_cache": str(root / "spack-source-cache"),
                   "misc_cache": str(root / "spack-cache/misc"), "build_jobs": 4},
        "bootstrap": {"root": str(root / "spack-bootstrap"), "enable": True},
        # Spack's YAML loader interprets this trailing colon as a scope override.
        "upstreams:": {},
    }}


def write_environment(root, profile, recipe_root, config=None):
    root = absolute_path(root, "--root")
    recipe_root = absolute_path(recipe_root, "--recipe-root")
    config = resolved_config(load_config() if config is None else config, profile)
    documents = {
        root / "site/pace/spack-envs" / profile / "spack.yaml": manifest(root, profile, recipe_root, config),
        root / "site/pace/spack-user/repos.yaml": {"repos": {"builtin": str(recipe_root)}},
        root / "site/pace/spack-envs" / profile / "site-packages.json": config,
    }
    rendered = {path: json.dumps(document, indent=2) + "\n" for path, document in documents.items()}
    # Check every destination before creating anything. Existing differing
    # graphs are preserved; identical files retain their timestamps/contents.
    for path, content in rendered.items():
        for component in (path, *path.parents):
            if component.is_symlink():
                raise PaceError(f"refusing symlinked site configuration: {component}; choose a new --site-root")
        if path.exists() and (not path.is_file() or path.read_text() != content):
            raise PaceError(f"existing PACE configuration differs: {path}; choose a new --site-root "
                            "(or --output-root) to preserve this dependency graph")
    for path, content in rendered.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also prevents an accidental concurrent overwrite.
            try:
                with path.open("x") as stream:
                    stream.write(content)
            except FileExistsError:
                if path.is_symlink() or path.read_text() != content:
                    raise PaceError(f"PACE configuration changed concurrently: {path}; choose a new --site-root")
    return tuple(rendered)


def check_lock(lock, profile, config=None):
    """Inspect concrete records only; never concretize, install, rewrite or execute."""
    config = resolved_config(load_config() if config is None else config, profile)
    if profile not in PROFILES:
        raise PaceError(f"unsupported PACE profile: {profile}")
    selected = config["profiles"][profile]
    gpu = profile != "cpu"
    cuda_arch = profile.removeprefix("cuda-sm") if gpu else None
    nodes = lock.get("concrete_specs", {})
    if not isinstance(nodes, dict) or not nodes:
        raise PaceError("lock must contain a nonempty concrete dependency graph")
    roots = lock.get("roots", [])
    if not isinstance(roots, list) or not roots:
        raise PaceError("lock must identify its concrete root specs")
    root_hashes = {item["hash"] for item in roots}
    for identity, node in nodes.items():
        target = node["arch"]["target"]
        target = target.get("name") if isinstance(target, dict) else target
        if target != config["target"]:
            raise PaceError(f'{node["name"]} target={target}; all PACE nodes must use {config["target"]}')
        for dependency in node.get("dependencies", []):
            if dependency["hash"] not in nodes:
                raise PaceError(f'{node["name"]} has a dependency outside the concrete graph')
    if not root_hashes.issubset(nodes):
        raise PaceError("lock root references a missing concrete spec")

    def one(name, version, rooted=False):
        matches = [(identity, node) for identity, node in nodes.items() if node["name"] == name]
        if len(matches) != 1:
            raise PaceError(f"exactly one concrete {name} is required")
        identity, node = matches[0]
        if str(node["version"]) != version:
            raise PaceError(f'{name}@{node["version"]}; this PACE profile requires {version}')
        if rooted and identity not in root_hashes:
            raise PaceError(f"{name} must be a concrete root of this PACE environment")
        return identity, node

    def parameters(node, expected):
        actual = node["parameters"]
        for key, value in expected.items():
            observed = actual.get(key)
            if isinstance(value, bool):
                matches = observed is value
            elif isinstance(observed, (list, tuple)):
                matches = list(observed) == [value]
            else:
                matches = observed == value
            if not matches:
                raise PaceError(f'{node["name"]} {key}={observed!r}; required {value!r}')

    def prefix(node, expected):
        if node.get("external", {}).get("path") != expected:
            raise PaceError(f'{node["name"]} must use the configured site external prefix {expected}')

    def edge(node, identity, name):
        if not any(item["hash"] == identity for item in node.get("dependencies", [])):
            raise PaceError(f'{node["name"]} must depend on the selected {name} DAG identity')

    hypre_id, hypre = one("hypre", config["hypre_version"], rooted=True)
    mpi_id, mpi = one("openmpi", selected["mpi_version"], rooted=True)
    gcc_id, gcc = one("gcc", config["compiler"]["version"])
    blas_id, blas = one("openblas", config["openblas"]["version"], rooted=True)
    for name in ("cmake", "metis"):
        _, node = one(name, config[name]["version"], rooted=True)
        prefix(node, config[name]["prefix"])
        if name == "metis":
            parameters(node, config["metis"]["parameters"])
    prefix(gcc, config["compiler"]["prefix"])
    prefix(mpi, selected["mpi_prefix"])
    prefix(blas, config["openblas"]["prefix"])
    parameters(mpi, selected["mpi_parameters"])
    parameters(blas, {"threads": "none", "shared": True, "ilp64": False})
    parameters(hypre, {"precision": "double", "complex": False, "shared": True, "mpi": True,
                       "cuda": gpu, "umpire": gpu, "cxxstd": "17", "openmp": False})
    if hypre.get("external"):
        raise PaceError("HYPRE must belong to this profile's built dependency graph")
    edge(hypre, mpi_id, "MPI")
    edge(hypre, gcc_id, "GCC")
    edge(hypre, blas_id, "OpenBLAS")
    if gpu:
        cuda_id, cuda = one("cuda", config["cuda_version"], rooted=True)
        umpire_id, umpire = one("umpire", config["umpire_version"], rooted=True)
        prefix(cuda, config["cuda_prefix"])
        parameters(hypre, {"cuda_arch": cuda_arch, "gpu-aware-mpi": True, "unified-memory": False})
        parameters(umpire, {"cuda": True, "cuda_arch": cuda_arch, "shared": True,
                            "c": True, "fortran": False, "examples": False})
        if umpire.get("external"):
            raise PaceError("Umpire must belong to this profile's built dependency graph")
        edge(hypre, cuda_id, "CUDA")
        edge(hypre, umpire_id, "Umpire")
        edge(umpire, cuda_id, "CUDA")
    elif any(node["name"] in ("cuda", "umpire") for node in nodes.values()):
        raise PaceError("CPU PACE graph must exclude CUDA/Umpire")
    return {"profile": profile, "target": config["target"], "hypre_hash": hypre_id,
            "mpi_hash": mpi_id, "hypre_version": config["hypre_version"], "precision": "double"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--recipe-root", type=Path)
    parser.add_argument("--site-config", help="discovered site JSON, or - for stdin; defaults to lock's sibling site-packages.json")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="print the manifest without creating files")
    modes.add_argument("--check-lock", type=Path, help="inspect an existing lock without modifying it")
    args = parser.parse_args(argv)
    try:
        site_config = args.site_config
        if site_config is None and args.check_lock:
            site_config = args.check_lock.parent / "site-packages.json"
        if site_config is None:
            raise PaceError("provide --site-config from pace_discover.py; installation prefixes are not hard-coded")
        config = json.load(sys.stdin) if site_config == "-" else json.loads(Path(site_config).read_text())
        if args.check_lock:
            result = check_lock(json.loads(args.check_lock.read_text()), args.profile, config)
            print(json.dumps(result, sort_keys=True))
        else:
            if args.root is None or args.recipe_root is None:
                parser.error("rendering requires --root ABS and --recipe-root ABS")
            if args.dry_run:
                print(json.dumps(manifest(args.root, args.profile, args.recipe_root, config), indent=2))
            else:
                for path in write_environment(args.root, args.profile, args.recipe_root, config):
                    print(path)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f"ERROR: PACE manifest: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
