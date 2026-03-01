#!/usr/bin/env python3
import argparse
import hashlib
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_list_of_str(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    out: List[str] = []
    for item in value:
        out.append(str(item))
    return out


def load_config(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("config root must be a JSON object")
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError("unsupported schema_version (expected 1)")
    if "cases" not in payload or not isinstance(payload["cases"], list):
        raise ValueError("config must include 'cases' as a list")
    return payload


def build_env(default_env: Dict[str, str], case_env: Dict[str, str], cli_env: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(default_env)
    env.update(case_env)
    env.update(cli_env)
    return env


def parse_env_overrides(values: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"invalid --env '{raw}', expected KEY=VALUE")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid --env '{raw}', empty KEY")
        out[key] = value
    return out


def sweep_combinations(sweep: Dict[str, Any]) -> List[List[Tuple[str, Any]]]:
    if not sweep:
        return [[]]
    keys = list(sweep.keys())
    values: List[List[Any]] = []
    for k in keys:
        v = sweep[k]
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(f"sweep entry '{k}' must be a non-empty list")
        values.append(v)
    combos: List[List[Tuple[str, Any]]] = []
    for product in itertools.product(*values):
        combos.append(list(zip(keys, product)))
    return combos


def append_option_tokens(dst: List[str], option: str, value: Any) -> None:
    # null/False means "skip option for this run"
    if value is None or value is False:
        return
    # True means "flag present with no value"
    if value is True:
        dst.append(option)
        return
    # List value lets users pass a multi-token value payload.
    if isinstance(value, list):
        dst.append(option)
        for item in value:
            dst.append(str(item))
        return
    dst.append(option)
    dst.append(str(value))


def _sanitize_token(value: str) -> str:
    token = value.strip()
    token = token.replace(".", "p")
    token = token.replace("+", "")
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", token)
    token = token.strip("_")
    return token or "x"


def _option_token(option: str) -> str:
    base = option.strip()
    base = re.sub(r"^-+", "", base)
    return _sanitize_token(base or "opt")


def _value_token(value: Any) -> str:
    if isinstance(value, list):
        joined = "x".join(_sanitize_token(str(v)) for v in value)
        return joined or "x"
    return _sanitize_token(str(value))


def build_run_id(
    case_name: str,
    idx: int,
    combo: List[Tuple[str, Any]],
    max_len: int = 180,
) -> str:
    parts: List[str] = []
    for option, value in combo:
        parts.append(f"{_option_token(option)}_{_value_token(value)}")

    base = _sanitize_token(case_name)
    if parts:
        candidate = f"{base}__{'__'.join(parts)}__{idx:04d}"
    else:
        candidate = f"{base}_{idx:04d}"

    if len(candidate) <= max_len:
        return candidate

    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:10]
    keep = max(8, max_len - 13)
    trimmed = candidate[:keep].rstrip("_")
    return f"{trimmed}__{digest}"


def run_single(
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    started = time.time()
    started_utc = utc_now()
    cmd_str = shlex.join(command)

    if dry_run:
        return {
            "command": command,
            "command_str": cmd_str,
            "cwd": str(cwd),
            "started_utc": started_utc,
            "finished_utc": started_utc,
            "duration_s": 0.0,
            "returncode": 0,
            "log_path": str(log_path),
            "dry_run": True,
        }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )

    ended = time.time()
    return {
        "command": command,
        "command_str": cmd_str,
        "cwd": str(cwd),
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "duration_s": ended - started,
        "returncode": proc.returncode,
        "log_path": str(log_path),
        "dry_run": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run parametric sweeps for multiple executables sequentially."
    )
    parser.add_argument("--config", required=True, help="Path to JSON runner config.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for logs + summary JSON. Defaults to scripting/runner/out/<timestamp>.",
    )
    parser.add_argument(
        "--only-case",
        action="append",
        default=[],
        help="Case name filter (repeatable).",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra env override KEY=VALUE (repeatable).",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running after failures.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print/record generated commands without executing them.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    payload = load_config(config_path)

    only_set = set(args.only_case)
    cli_env = parse_env_overrides(args.env)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = (Path("scripting/runner/out") / timestamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    defaults = payload.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be an object")

    default_launcher = ensure_list_of_str(defaults.get("launcher", []), "defaults.launcher")
    default_base_args = ensure_list_of_str(defaults.get("base_args", []), "defaults.base_args")
    default_workdir = Path(str(defaults.get("workdir", "."))).resolve()
    default_env = {str(k): str(v) for k, v in dict(defaults.get("env", {})).items()}

    keep_going = args.keep_going or bool(defaults.get("keep_going", False))

    all_runs: List[Dict[str, Any]] = []
    total_failed = 0
    total_planned = 0

    for case in payload["cases"]:
        if not isinstance(case, dict):
            raise ValueError("each case entry must be an object")
        name = str(case.get("name", "")).strip()
        if not name:
            raise ValueError("each case must define a non-empty name")
        if only_set and name not in only_set:
            continue

        exe = str(case.get("executable", "")).strip()
        if not exe:
            raise ValueError(f"case '{name}' missing executable")

        launcher = default_launcher + ensure_list_of_str(case.get("launcher", []), f"case[{name}].launcher")
        base_args = default_base_args + ensure_list_of_str(case.get("base_args", []), f"case[{name}].base_args")
        workdir = Path(str(case.get("workdir", default_workdir))).resolve()
        case_env = {str(k): str(v) for k, v in dict(case.get("env", {})).items()}
        run_env = build_env(default_env, case_env, cli_env)

        sweep = case.get("sweep", {})
        if sweep is None:
            sweep = {}
        if not isinstance(sweep, dict):
            raise ValueError(f"case '{name}': sweep must be an object")
        combinations = sweep_combinations(sweep)
        total_planned += len(combinations)

        for idx, combo in enumerate(combinations, start=1):
            combo_tokens: List[str] = []
            combo_desc: Dict[str, Any] = {}
            for option, value in combo:
                combo_desc[option] = value
                append_option_tokens(combo_tokens, option, value)

            command = launcher + [exe] + base_args + combo_tokens
            run_id = build_run_id(name, idx, combo)
            log_path = out_dir / "logs" / f"{run_id}.log"

            print(f"[runner] case={name} run={idx}/{len(combinations)}")
            print(f"[runner] cmd={shlex.join(command)}")

            result = run_single(
                command=command,
                cwd=workdir,
                env=run_env,
                log_path=log_path,
                dry_run=args.dry_run,
            )
            result["case_name"] = name
            result["run_id"] = run_id
            result["sweep_values"] = combo_desc
            all_runs.append(result)

            if result["returncode"] != 0:
                total_failed += 1
                print(f"[runner] FAIL case={name} run={idx} rc={result['returncode']}")
                if not keep_going and not args.dry_run:
                    summary = {
                        "schema_version": 1,
                        "generated_utc": utc_now(),
                        "config_path": str(config_path),
                        "out_dir": str(out_dir),
                        "total_planned_runs": total_planned,
                        "total_executed_runs": len(all_runs),
                        "total_failed_runs": total_failed,
                        "dry_run": args.dry_run,
                        "runs": all_runs,
                    }
                    summary_path = out_dir / "summary.json"
                    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
                    print(f"[runner] wrote {summary_path}")
                    return 2

    summary = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "total_planned_runs": total_planned,
        "total_executed_runs": len(all_runs),
        "total_failed_runs": total_failed,
        "dry_run": args.dry_run,
        "runs": all_runs,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[runner] wrote {summary_path}")

    if total_failed > 0 and not args.dry_run:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"parametric-runner: ERROR: {exc}", file=sys.stderr)
        raise
