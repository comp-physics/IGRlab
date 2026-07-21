#!/usr/bin/env python3
"""Validate a profile-provenance-v1 sidecar without contacting profiling tools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def fact_is_valid(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("status") not in {"available", "unavailable"}:
        return False
    if value["status"] == "available":
        return isinstance(value.get("value"), str) and bool(value["value"])
    return isinstance(value.get("reason"), str) and bool(value["reason"])


def lookup(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--require", action="append", default=[], metavar="PATH")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    try:
        payload = json.loads(args.metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"could not read metadata: {exc}")
        payload = {}

    if not isinstance(payload, dict):
        errors.append("top-level metadata is not an object")
        payload = {}
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(payload.get("created_utc"), str) or not payload["created_utc"]:
        errors.append("created_utc is missing or empty")
    tool = payload.get("tool")
    if not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not tool["name"]:
        errors.append("tool.name is missing or empty")
    capture = payload.get("capture")
    if not isinstance(capture, dict) or not isinstance(capture.get("command"), list) or not capture["command"]:
        errors.append("capture.command is missing or empty")
    elif any(not isinstance(item, str) or not item for item in capture["command"]):
        errors.append("capture.command contains a non-string or empty item")

    fact_paths = [
        "tool.executable", "tool.version", "capture.output_stem",
        "source.root", "source.revision", "source.tree", "source.status",
    ]
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        errors.append("environment is not an object")
    else:
        fact_paths.extend(f"environment.{key}" for key in environment)
    for path in fact_paths:
        if not fact_is_valid(lookup(payload, path)):
            errors.append(f"{path} is not a valid availability fact")

    labels = payload.get("labels")
    if not isinstance(labels, dict) or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in labels.items()):
        errors.append("labels must be an object with non-empty string keys and values")

    for path in args.require:
        value = lookup(payload, path)
        if not fact_is_valid(value):
            errors.append(f"required {path} is absent or malformed")
        elif value["status"] != "available":
            errors.append(f"required {path} is unavailable: {value['reason']}")

    result = {"schema_version": SCHEMA_VERSION, "metadata": str(args.metadata), "status": "pass" if not errors else "fail", "errors": errors}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"profile-provenance-check: status={result['status']} errors={len(errors)}")
    for error in errors:
        print(f"profile-provenance-check: ERROR: {error}", file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
