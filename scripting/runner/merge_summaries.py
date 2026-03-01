#!/usr/bin/env python3
"""Merge multiple runner summary.json files into one combined summary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_summary(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"summary root must be object: {path}")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"summary missing 'runs' list: {path}")
    return payload


def run_key(run: Dict[str, Any], mode: str) -> str:
    if mode == "run-id":
        return str(run.get("run_id", ""))
    if mode == "run-id-log":
        return f"{run.get('run_id', '')}|{run.get('log_path', '')}"
    raise ValueError(f"unsupported dedupe mode: {mode}")


def merge_summaries(
    summaries: List[Path],
    dedupe: str,
) -> Dict[str, Any]:
    merged_runs: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}
    dropped_dupes = 0
    total_planned = 0

    for src in summaries:
        payload = load_summary(src)
        total_planned += int(payload.get("total_planned_runs", 0))

        for raw in payload.get("runs", []):
            if not isinstance(raw, dict):
                continue
            run = dict(raw)
            run["summary_source"] = str(src)
            if dedupe == "none":
                merged_runs.append(run)
                continue

            key = run_key(run, dedupe)
            if key in seen:
                # Keep latest entry in CLI order (last wins).
                merged_runs[seen[key]] = run
                dropped_dupes += 1
            else:
                seen[key] = len(merged_runs)
                merged_runs.append(run)

    total_failed = sum(1 for r in merged_runs if int(r.get("returncode", 1)) != 0)
    any_dry_run = any(bool(r.get("dry_run", False)) for r in merged_runs)

    return {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "merged_from": [str(p) for p in summaries],
        "source_count": len(summaries),
        "total_planned_runs": total_planned,
        "total_executed_runs": len(merged_runs),
        "total_failed_runs": total_failed,
        "dry_run": any_dry_run,
        "dedupe_mode": dedupe,
        "dropped_duplicate_runs": dropped_dupes,
        "runs": merged_runs,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Merge multiple scripting/runner summary.json files."
    )
    ap.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Path to input summary.json (repeatable, merged in given order).",
    )
    ap.add_argument(
        "--dedupe",
        choices=["none", "run-id", "run-id-log"],
        default="run-id",
        help="Duplicate handling mode (default: run-id).",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output merged summary path.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    summaries = [Path(s).expanduser().resolve() for s in args.summary]
    out_path = Path(args.out).expanduser().resolve()

    merged = merge_summaries(summaries, dedupe=str(args.dedupe))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(
        "[merge-summaries] wrote"
        f" path={out_path}"
        f" runs={merged['total_executed_runs']}"
        f" failed={merged['total_failed_runs']}"
        f" dropped_dupes={merged['dropped_duplicate_runs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

