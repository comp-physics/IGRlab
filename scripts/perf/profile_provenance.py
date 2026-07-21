#!/usr/bin/env python3
"""Write an explicit, tool-agnostic profiling provenance sidecar.

This utility records supplied capture inputs and Git identity only.  It does
not invoke a profiler, query a GPU, inspect build trees, or infer workload
defaults.  An omitted optional value is represented as unavailable, never as
an empty successful value.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def available(value: str) -> dict[str, str]:
    return {"status": "available", "value": value}


def unavailable(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def supplied(value: str, name: str) -> dict[str, str]:
    return available(value) if value else unavailable(f"{name} was not supplied")


def git_value(root: Path, *args: str) -> dict[str, str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        return unavailable(f"git could not run: {exc}")
    value = result.stdout.strip()
    if result.returncode == 0 and value:
        return available(value)
    detail = result.stderr.strip() or f"git exited {result.returncode}"
    return unavailable(detail)


def git_status(root: Path) -> dict[str, str]:
    """Record a clean worktree explicitly instead of treating it as missing."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--short"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        return unavailable(f"git could not run: {exc}")
    if result.returncode == 0:
        return available(result.stdout.strip() or "clean")
    return unavailable(result.stderr.strip() or f"git exited {result.returncode}")


def collect_source(root: Path) -> dict[str, dict[str, str]]:
    if not root.is_dir():
        missing = unavailable(f"source root does not exist: {root}")
        return {"root": missing, "revision": missing, "tree": missing, "status": missing}
    return {
        "root": available(str(root.resolve())),
        "revision": git_value(root, "rev-parse", "HEAD"),
        "tree": git_value(root, "rev-parse", "HEAD^{tree}"),
        "status": git_status(root),
    }


def parse_labels(values: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"label must be KEY=VALUE: {value!r}")
        key, item = value.split("=", 1)
        if not key or not item:
            raise ValueError(f"label must have non-empty KEY and VALUE: {value!r}")
        if key in labels:
            raise ValueError(f"duplicate label key: {key!r}")
        labels[key] = item
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tool", required=True, help="Profiler tool name (for example nsys).")
    parser.add_argument("--tool-executable", default="")
    parser.add_argument("--tool-version", default="")
    parser.add_argument("--output-stem", default="")
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--environment", action="append", default=[], metavar="KEY")
    parser.add_argument("--label", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--command", nargs=argparse.REMAINDER, required=True)
    argv = sys.argv[1:]
    command_index = argv.index("--command") if "--command" in argv else -1
    if command_index >= 0 and len(argv) > command_index + 1 and argv[command_index + 1] == "--":
        del argv[command_index + 1]
    args = parser.parse_args(argv)

    if not args.tool.strip():
        parser.error("--tool must be non-empty")
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command or any(not item for item in command):
        parser.error("--command must contain a non-empty command after --")

    environment: dict[str, dict[str, str]] = {}
    for key in args.environment:
        if not key or "=" in key:
            parser.error(f"--environment expects a variable name: {key!r}")
        environment[key] = (
            available(os.environ[key])
            if key in os.environ
            else unavailable(f"environment variable {key} is not set")
        )

    try:
        labels = parse_labels(args.label)
    except ValueError as exc:
        parser.error(str(exc))

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tool": {
            "name": args.tool,
            "executable": supplied(args.tool_executable, "tool executable"),
            "version": supplied(args.tool_version, "tool version"),
        },
        "capture": {
            "command": command,
            "output_stem": supplied(args.output_stem, "output stem"),
        },
        "source": collect_source(args.source_root),
        "environment": environment,
        "labels": labels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"profile-provenance: wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"profile-provenance: ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
