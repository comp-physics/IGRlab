#!/usr/bin/env python3
"""Resolve runner outputs (summary.json + logs) into render-ready dataset manifest."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _is_numeric_token(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def parse_option_map(tokens: List[str]) -> Dict[str, Any]:
    """Parse command token list into last-seen option map.

    Handles flags with values including negative numeric values.
    """
    out: Dict[str, Any] = {}
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if not tok.startswith("-"):
            i += 1
            continue

        next_is_value = False
        if i + 1 < n:
            nxt = tokens[i + 1]
            if not nxt.startswith("-") or _is_numeric_token(nxt):
                next_is_value = True

        if next_is_value:
            out[tok] = tokens[i + 1]
            i += 2
        else:
            out[tok] = True
            i += 1
    return out


def option_get(opts: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in opts:
            return opts[key]
    return default


def normalize_ca_token(value: Any) -> str:
    if value is None:
        return "1"
    try:
        numeric = float(value)
        return f"{numeric:.15g}"
    except Exception:
        return str(value)


def parse_dataset_name_from_log(log_path: Path) -> Optional[str]:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    matches = re.findall(r"VTK dataset:\s*(\S+)", text)
    return matches[-1] if matches else None


def infer_dataset_name(run: Dict[str, Any], cli_opts: Dict[str, Any]) -> str:
    case_name = str(run.get("case_name", "")).strip()
    if not case_name:
        raise ValueError("run missing case_name")

    order = option_get(cli_opts, "-o", "--order", default=3)
    rs = option_get(cli_opts, "-rs", "--serial-refine", default=0)
    rp = option_get(cli_opts, "-rp", "--parallel-refine", default=0)
    igr_space = option_get(cli_opts, "-is", "--igr-space", default="H1")
    igr_grad = option_get(cli_opts, "-ig", "--igr-grad", default="H1")
    c_alpha = normalize_ca_token(option_get(cli_opts, "-ca", "--c-alpha", default=1.0))

    try:
        total_ref = int(rs) + int(rp)
    except Exception:
        total_ref = 0

    return (
        f"{case_name}_p_{order}_ref_{total_ref}_"
        f"is_{igr_space}_ig_{igr_grad}_ca_{c_alpha}"
    )


@dataclass
class DiscoveryConfig:
    out_root: Path
    include_cases: List[str]
    exclude_cases: List[str]
    include_run_ids: List[str]
    require_success: bool = True
    skip_dry_run: bool = True
    require_dataset: bool = True


def _find_case_dataset_dirs(case_root: Path) -> List[Path]:
    if not case_root.exists():
        return []
    out: List[Path] = []
    for child in case_root.iterdir():
        if not child.is_dir():
            continue
        if any(child.glob("*.pvd")):
            out.append(child)
    out.sort()
    return out


def _resolve_dataset_dir(
    case_root: Path,
    requested_dataset: str,
    run: Dict[str, Any],
    cli_opts: Dict[str, Any],
) -> Optional[Path]:
    exact = case_root / requested_dataset
    if exact.exists() and exact.is_dir():
        return exact

    dirs = _find_case_dataset_dirs(case_root)
    if not dirs:
        return None

    prefix = f"{run['case_name']}_p_{option_get(cli_opts, '-o', '--order', default=3)}"
    prefix_matches = [d for d in dirs if d.name.startswith(prefix)]

    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(dirs) == 1:
        return dirs[0]

    requested_matches = [d for d in dirs if d.name == requested_dataset]
    if len(requested_matches) == 1:
        return requested_matches[0]

    return None


def _resolve_pvd(dataset_dir: Path, dataset_name: str) -> Optional[Path]:
    exact = dataset_dir / f"{dataset_name}.pvd"
    if exact.exists():
        return exact
    pvds = sorted(dataset_dir.glob("*.pvd"))
    if len(pvds) == 1:
        return pvds[0]
    if pvds:
        # Prefer exact prefix match if present.
        for p in pvds:
            if p.stem == dataset_name:
                return p
        return pvds[0]
    return None


def discover_from_summary(summary_path: Path, cfg: DiscoveryConfig) -> Dict[str, Any]:
    summary = load_json(summary_path)
    runs = summary.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("summary missing 'runs' list")

    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    include_case_set = set(cfg.include_cases)
    exclude_case_set = set(cfg.exclude_cases)
    include_run_set = set(cfg.include_run_ids)

    for run in runs:
        if not isinstance(run, dict):
            continue

        case_name = str(run.get("case_name", "")).strip()
        run_id = str(run.get("run_id", "")).strip()
        if include_case_set and case_name not in include_case_set:
            continue
        if case_name in exclude_case_set:
            continue
        if include_run_set and run_id not in include_run_set:
            continue
        if cfg.require_success and int(run.get("returncode", 1)) != 0:
            skipped.append({"run_id": run_id, "reason": "returncode-nonzero"})
            continue
        if cfg.skip_dry_run and bool(run.get("dry_run", False)):
            skipped.append({"run_id": run_id, "reason": "dry-run"})
            continue

        command = [str(x) for x in run.get("command", [])]
        cli_opts = parse_option_map(command)

        log_path = Path(str(run.get("log_path", ""))).expanduser().resolve()
        dataset_name = parse_dataset_name_from_log(log_path)
        source = "log"
        if not dataset_name:
            dataset_name = infer_dataset_name(run, cli_opts)
            source = "inferred"

        case_root = cfg.out_root / case_name
        dataset_dir = _resolve_dataset_dir(case_root, dataset_name, run, cli_opts)
        if dataset_dir is None:
            reason = "dataset-dir-not-found"
            if cfg.require_dataset:
                skipped.append(
                    {
                        "run_id": run_id,
                        "reason": reason,
                        "dataset_name": dataset_name,
                        "case_root": str(case_root),
                    }
                )
                continue
            pvd_path = None
        else:
            pvd_path = _resolve_pvd(dataset_dir, dataset_name)

        if cfg.require_dataset and (dataset_dir is None or pvd_path is None):
            skipped.append(
                {
                    "run_id": run_id,
                    "reason": "pvd-not-found",
                    "dataset_dir": str(dataset_dir) if dataset_dir else None,
                    "dataset_name": dataset_name,
                }
            )
            continue

        run_metadata = None
        if dataset_dir is not None:
            metadata_candidate = dataset_dir / "run_metadata.json"
            if metadata_candidate.exists():
                run_metadata = str(metadata_candidate)

        records.append(
            {
                "run_id": run_id,
                "case_name": case_name,
                "dataset_name": dataset_name,
                "dataset_name_source": source,
                "dataset_dir": str(dataset_dir) if dataset_dir else None,
                "pvd_path": str(pvd_path) if pvd_path else None,
                "run_metadata_path": run_metadata,
                "log_path": str(log_path),
                "cwd": run.get("cwd"),
                "sweep_values": run.get("sweep_values", {}),
                "command": command,
                "command_str": run.get("command_str", ""),
                "started_utc": run.get("started_utc"),
                "finished_utc": run.get("finished_utc"),
                "duration_s": run.get("duration_s"),
                "returncode": run.get("returncode"),
            }
        )

    manifest = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "summary_path": str(summary_path),
        "out_root": str(cfg.out_root),
        "record_count": len(records),
        "skipped_count": len(skipped),
        "records": records,
        "skipped": skipped,
    }
    return manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build render manifest from runner summary JSON.")
    ap.add_argument("--summary", required=True, help="Path to scripting/runner summary.json")
    ap.add_argument("--out-root", default="out", help="Root directory where VTK case outputs live.")
    ap.add_argument("--out-manifest", required=True, help="Output render manifest JSON path.")
    ap.add_argument("--only-case", action="append", default=[], help="Case name filter (repeatable)")
    ap.add_argument("--exclude-case", action="append", default=[], help="Case exclusion filter")
    ap.add_argument("--only-run-id", action="append", default=[], help="Run-id filter (repeatable)")
    ap.add_argument("--allow-failed", action="store_true", help="Include runs with non-zero returncode")
    ap.add_argument("--include-dry-run", action="store_true", help="Include dry-run entries")
    ap.add_argument(
        "--allow-missing-datasets",
        action="store_true",
        help="Keep entries even when dataset directory/pvd is not found",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary).expanduser().resolve()
    out_manifest = Path(args.out_manifest).expanduser().resolve()
    cfg = DiscoveryConfig(
        out_root=Path(args.out_root).expanduser().resolve(),
        include_cases=list(args.only_case),
        exclude_cases=list(args.exclude_case),
        include_run_ids=list(args.only_run_id),
        require_success=not args.allow_failed,
        skip_dry_run=not args.include_dry_run,
        require_dataset=not args.allow_missing_datasets,
    )
    manifest = discover_from_summary(summary_path, cfg)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        "[render-discover] wrote"
        f" path={out_manifest} records={manifest['record_count']} skipped={manifest['skipped_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
