#!/usr/bin/env python3
"""End-to-end render pipeline orchestrator for scripting/runner outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from render_discover import DiscoveryConfig, discover_from_summary


def utc_stamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_seconds(s: float) -> str:
    if s < 60.0:
        return f"{s:.1f}s"
    m = int(s // 60.0)
    sec = s - 60.0 * m
    return f"{m}m{sec:04.1f}s"


def progress_bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + ("?" * width) + "]"
    frac = max(0.0, min(1.0, float(done) / float(total)))
    fill = int(round(frac * width))
    return "[" + ("#" * fill) + ("-" * (width - fill)) + "]"


class Progress:
    def __init__(self, label: str, total: int, min_interval_s: float = 1.0):
        self.label = label
        self.total = max(0, int(total))
        self.done = 0
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.start = time.monotonic()
        self.last_print = 0.0
        self._print()

    def step(self, inc: int = 1, suffix: str = "") -> None:
        self.done += int(inc)
        now = time.monotonic()
        if self.done >= self.total or (now - self.last_print) >= self.min_interval_s:
            self._print(now=now, suffix=suffix)

    def _print(self, now: float | None = None, suffix: str = "") -> None:
        if now is None:
            now = time.monotonic()
        elapsed = max(1e-12, now - self.start)
        if self.total > 0:
            frac = max(0.0, min(1.0, float(self.done) / float(self.total)))
            rate = float(self.done) / elapsed if self.done > 0 else 0.0
            remain = max(0, self.total - self.done)
            eta = float(remain) / rate if rate > 1e-12 else float("inf")
            eta_txt = "?" if eta == float("inf") else fmt_seconds(eta)
            extra = f" {suffix}" if suffix else ""
            print(
                f"[render-pipeline] {self.label} {progress_bar(self.done, self.total)} "
                f"{100.0 * frac:5.1f}% ({self.done}/{self.total}) "
                f"elapsed={fmt_seconds(elapsed)} eta={eta_txt}{extra}",
                flush=True,
            )
        else:
            print(
                f"[render-pipeline] {self.label} {progress_bar(0, 0)} done={self.done} "
                f"elapsed={fmt_seconds(elapsed)}",
                flush=True,
            )
        self.last_print = now


def load_config(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("config root must be JSON object")
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError("unsupported schema_version (expected 1)")
    if "summary_path" not in payload:
        raise ValueError("config requires 'summary_path'")
    return payload


def run_command(cmd: List[str]) -> None:
    print("[render-pipeline] cmd:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def run_command_in_dir(cmd: List[str], cwd: Path) -> None:
    print("[render-pipeline] cmd:", " ".join(cmd), f"(cwd={cwd})", flush=True)
    subprocess.check_call(cmd, cwd=str(cwd))


def run_render_command_with_glx_recovery(
    cmd: List[str], ranges_path: Path, frames_path: Path
) -> None:
    """Run render command, tolerating known ParaView GLX teardown failures.

    Some environments generate valid screenshots/manifests but still return a
    non-zero exit code with GLXBadContext on process teardown. Recover only when
    artifacts are present and non-empty.
    """
    print("[render-pipeline] cmd:", " ".join(cmd))
    start_ts = datetime.now(tz=timezone.utc).timestamp()
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    lines: List[str] = []
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            lines.append(line)
        proc.wait()
        returncode = int(proc.returncode)
    output = "".join(lines)

    if returncode == 0:
        return

    known_glx = (
        "GLXBadContext" in output
        or "X Error of failed request" in output
        or "X_GLXMakeCurrent" in output
    )
    known_teardown_segv = (
        returncode in (-11, 139)
        or "terminated with signal 11" in output
        or "Segmentation fault" in output
    )

    artifacts_ok = False
    if ranges_path.exists() and frames_path.exists():
        try:
            frames_payload = load_json(frames_path)
            frame_count = int(frames_payload.get("frame_count", 0))
            ranges_mtime = ranges_path.stat().st_mtime
            frames_mtime = frames_path.stat().st_mtime
            artifacts_fresh = (
                ranges_mtime >= start_ts - 1.0 and frames_mtime >= start_ts - 1.0
            )
            artifacts_ok = frame_count > 0 and artifacts_fresh
        except Exception:
            artifacts_ok = False

    if (known_glx or known_teardown_segv) and artifacts_ok:
        print(
            "[render-pipeline] warning: non-zero ParaView exit on teardown;"
            " render artifacts are complete, continuing."
        )
        return

    raise subprocess.CalledProcessError(returncode, cmd, output=output)


def as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def normalize_latex_length(value: Any, default: str) -> str:
    text = str(value if value is not None else default)
    # JSON "\textwidth" can become a literal tab if not double-escaped.
    if "\t" in text and "extwidth" in text:
        text = text.replace("\t", r"\t")
    return text


def sanitize_name(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return out.strip("_") or "job"


def prepare_output_dir(output_dir: Path, clean_output_dir: bool) -> None:
    if clean_output_dir and output_dir.exists():
        resolved = output_dir.resolve()
        cwd = Path.cwd().resolve()
        home = Path.home().resolve()
        if resolved in (Path("/"), cwd, home):
            raise RuntimeError(
                f"refusing to clean unsafe output_dir={resolved}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def normalize_render_jobs(
    render_defaults: Dict[str, Any],
    render_jobs_cfg: Any,
    render_enabled: bool,
) -> List[Dict[str, Any]]:
    if not render_enabled:
        return []

    jobs: List[Dict[str, Any]] = []
    if render_jobs_cfg is None:
        merged = dict(render_defaults)
        merged.setdefault("name", "all")
        jobs.append(merged)
        return expand_scale_mode_jobs(jobs)

    if not isinstance(render_jobs_cfg, list):
        raise ValueError("render_jobs must be a list when provided")

    for idx, raw in enumerate(render_jobs_cfg, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"render_jobs[{idx}] must be an object")
        merged = dict(render_defaults)
        merged.update(raw)
        merged.setdefault("name", f"job_{idx:02d}")
        jobs.append(merged)

    jobs = expand_scale_mode_jobs(jobs)
    return jobs


def expand_scale_mode_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for job in jobs:
        base_name = sanitize_name(str(job.get("name", "job")))

        scale_modes_cfg = job.get("scale_modes")
        if scale_modes_cfg is None:
            modes = [str(job.get("scale_mode", "per_dataset"))]
        else:
            if not isinstance(scale_modes_cfg, list) or len(scale_modes_cfg) == 0:
                raise ValueError(
                    f"render job '{base_name}' has invalid scale_modes; expected non-empty list"
                )
            seen: set[str] = set()
            modes = []
            for x in scale_modes_cfg:
                mode = str(x)
                if mode in seen:
                    continue
                seen.add(mode)
                modes.append(mode)

        multi = len(modes) > 1
        if multi:
            print(
                f"[render-pipeline] warning: job '{base_name}' configured with "
                f"{len(modes)} scale modes; runtime and image count scale roughly with mode count."
            )
        for mode in modes:
            item = dict(job)
            item["scale_mode"] = mode
            item["_base_name"] = base_name
            item["_scale_mode_tag"] = sanitize_name(mode)
            if multi:
                item["name"] = f"{base_name}__sm_{sanitize_name(mode)}"
            else:
                item["name"] = base_name
            expanded.append(item)

    return expanded


def record_selected(record: Dict[str, Any], job: Dict[str, Any]) -> bool:
    case_name = str(record.get("case_name", ""))
    run_id = str(record.get("run_id", ""))

    include_cases = set(str(x) for x in job.get("include_cases", []) or [])
    exclude_cases = set(str(x) for x in job.get("exclude_cases", []) or [])
    include_run_ids = set(str(x) for x in job.get("include_run_ids", []) or [])
    exclude_run_ids = set(str(x) for x in job.get("exclude_run_ids", []) or [])

    if include_cases and case_name not in include_cases:
        return False
    if case_name in exclude_cases:
        return False
    if include_run_ids and run_id not in include_run_ids:
        return False
    if run_id in exclude_run_ids:
        return False
    return True


def filter_records(records: List[Dict[str, Any]], job: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [r for r in records if record_selected(r, job)]


def build_manifest_like(base_manifest: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base_manifest)
    out["records"] = records
    out["record_count"] = len(records)
    return out


def load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def render_job(
    output_dir: Path,
    manifest: Dict[str, Any],
    job: Dict[str, Any],
    render_script: Path,
) -> Dict[str, Any]:
    job_name = sanitize_name(str(job.get("name", "job")))
    scale_mode = str(job.get("scale_mode", "per_dataset"))
    scale_tag = sanitize_name(str(job.get("_scale_mode_tag", scale_mode)))

    job_dir = output_dir / "jobs" / f"sm_{scale_tag}" / job_name
    job_manifest_path = job_dir / "render_manifest.json"

    records = list(manifest.get("records", []))
    selected_records = filter_records(records, job)
    allow_empty = as_bool(job.get("allow_empty"), False)
    if not selected_records and not allow_empty:
        raise RuntimeError(f"render job '{job_name}' matched zero records")

    job_manifest = build_manifest_like(manifest, selected_records)
    write_json(job_manifest_path, job_manifest)

    images_dir = output_dir / "images_raw" / f"sm_{scale_tag}" / job_name
    ranges_path = job_dir / "render_ranges.json"
    frames_path = job_dir / "frames_raw.json"

    if not selected_records:
        write_json(ranges_path, {
            "schema_version": 1,
            "generated_utc": utc_now(),
            "scale_mode": job.get("scale_mode", "per_dataset"),
            "fixed_ranges": {},
            "resolved_ranges": {}
        })
        write_json(frames_path, {
            "schema_version": 1,
            "generated_utc": utc_now(),
            "render_outdir": str(images_dir),
            "manifest_path": str(job_manifest_path),
            "frame_count": 0,
            "frames": []
        })
    else:
        pvpython = str(job.get("pvpython", "pvpython"))
        fields = [str(x) for x in job.get("fields", [])]
        if not fields:
            raise ValueError(f"render job '{job_name}' requires non-empty fields")

        pvpython_args = [str(x) for x in job.get("pvpython_args", []) or []]
        cmd = [
            pvpython,
            *pvpython_args,
            str(render_script),
            "--manifest",
            str(job_manifest_path),
            "--outdir",
            str(images_dir),
            "--fields",
            *fields,
            "--w",
            str(int(job.get("width", 2400))),
            "--h",
            str(int(job.get("height", 1600))),
            "--scale-mode",
            scale_mode,
            "--write-ranges",
            str(ranges_path),
            "--write-frames",
            str(frames_path),
        ]

        times = [str(x) for x in job.get("times", []) or []]
        if times:
            cmd.extend(["--times", *times])
        if as_bool(job.get("use_time_values"), False):
            cmd.append("--use-time-values")
        if as_bool(job.get("transparent_bg"), False):
            cmd.append("--transparent-bg")
        if not as_bool(job.get("show_scalar_bar"), True):
            cmd.append("--hide-scalar-bar")
        if not as_bool(job.get("show_title"), True):
            cmd.append("--hide-title")
        color_preset = job.get("color_preset")
        if color_preset:
            cmd.extend(["--color-preset", str(color_preset)])
        field_colormaps = job.get("field_colormaps", {}) or {}
        if field_colormaps:
            if not isinstance(field_colormaps, dict):
                raise ValueError(
                    f"render job '{job_name}' field_colormaps must be an object"
                )
            for field_name, preset_name in field_colormaps.items():
                cmd.extend(
                    [
                        "--field-colormap",
                        f"{field_name}:{preset_name}",
                    ]
                )
        invert_fields = job.get("invert_field_colormaps", []) or []
        if invert_fields:
            if not isinstance(invert_fields, list):
                raise ValueError(
                    f"render job '{job_name}' invert_field_colormaps must be a list"
                )
            for field_name in invert_fields:
                cmd.extend(
                    [
                        "--invert-field-colormap",
                        str(field_name),
                    ]
                )
        filename_metadata_level = job.get("filename_metadata_level")
        if filename_metadata_level is None:
            # Backward compatibility for prior boolean toggle.
            if "embed_scale_in_filename" in job:
                filename_metadata_level = (
                    "legacy"
                    if as_bool(job.get("embed_scale_in_filename"), True)
                    else "minimal"
                )
            else:
                filename_metadata_level = "rich"
        cmd.extend(
            [
                "--filename-metadata-level",
                str(filename_metadata_level),
            ]
        )

        fixed_ranges = job.get("fixed_ranges", {}) or {}
        if fixed_ranges:
            for field, pair in fixed_ranges.items():
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError(
                        f"render job '{job_name}' fixed_ranges must map field -> [min,max]"
                    )
                cmd.extend(["--fixed-range", f"{field}:{pair[0]}:{pair[1]}"])

        run_render_command_with_glx_recovery(cmd, ranges_path, frames_path)

    return {
        "name": job_name,
        "record_count": len(selected_records),
        "manifest_path": str(job_manifest_path),
        "images_dir": str(images_dir),
        "ranges_path": str(ranges_path),
        "frames_path": str(frames_path),
        "scale_mode": scale_mode,
        "scale_mode_tag": scale_tag,
        "times": [str(x) for x in job.get("times", []) or []],
        "fields": [str(x) for x in job.get("fields", []) or []],
    }


def combine_job_frames(output_dir: Path, jobs: List[Dict[str, Any]]) -> Path:
    combined_frames: List[Dict[str, Any]] = []
    for job in jobs:
        frames_payload = load_json(Path(job["frames_path"]))
        frames = frames_payload.get("frames", [])
        if not isinstance(frames, list):
            continue
        scale_tag = sanitize_name(str(job.get("scale_mode_tag", job.get("scale_mode", "unknown"))))
        rel_prefix = f"sm_{scale_tag}/{job['name']}"
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            item = dict(frame)
            item["job_name"] = job["name"]
            item["scale_mode_tag"] = scale_tag
            rel = str(item.get("relative_path", ""))
            item["relative_path"] = f"{rel_prefix}/{rel}" if rel else rel_prefix
            combined_frames.append(item)

    combined_path = output_dir / "frames_raw.json"
    payload = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "frame_count": len(combined_frames),
        "jobs": jobs,
        "frames": combined_frames,
    }
    write_json(combined_path, payload)
    return combined_path


def combine_job_ranges(output_dir: Path, jobs: List[Dict[str, Any]]) -> Path:
    entries: List[Dict[str, Any]] = []
    for job in jobs:
        ranges_payload = load_json(Path(job["ranges_path"]))
        entries.append(
            {
                "job_name": job["name"],
                "ranges_path": job["ranges_path"],
                "scale_mode": ranges_payload.get("scale_mode"),
                "fixed_ranges": ranges_payload.get("fixed_ranges", {}),
                "resolved_ranges": ranges_payload.get("resolved_ranges", {}),
            }
        )

    combined_path = output_dir / "render_ranges.json"
    payload = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "job_count": len(entries),
        "jobs": entries,
    }
    write_json(combined_path, payload)
    return combined_path


def export_image_metadata_tables(frames_manifest: Path, output_dir: Path) -> Dict[str, str]:
    payload = load_json(frames_manifest)
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError(f"frames manifest has invalid 'frames': {frames_manifest}")

    csv_path = output_dir / "image_metadata.csv"
    jsonl_path = output_dir / "image_metadata.jsonl"
    fieldnames = [
        "job_name",
        "case_name",
        "run_id",
        "dataset_name",
        "field",
        "time_token",
        "time_value",
        "scale_mode",
        "field_association",
        "range_min",
        "range_max",
        "relative_path",
        "output_path",
        "pvd_path",
        "sweep_values_json",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f_csv, jsonl_path.open(
        "w", encoding="utf-8"
    ) as f_jsonl:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        progress = Progress(label="metadata-export", total=len(frames))
        for frame in frames:
            if not isinstance(frame, dict):
                progress.step()
                continue
            sweep_values = frame.get("sweep_values", {})
            row = {
                "job_name": frame.get("job_name", ""),
                "case_name": frame.get("case_name", ""),
                "run_id": frame.get("run_id", ""),
                "dataset_name": frame.get("dataset_name", ""),
                "field": frame.get("field", ""),
                "time_token": frame.get("time_token", ""),
                "time_value": frame.get("time_value", ""),
                "scale_mode": frame.get("scale_mode", ""),
                "field_association": frame.get("field_association", ""),
                "range_min": frame.get("range_min", frame.get("range", {}).get("min", "")),
                "range_max": frame.get("range_max", frame.get("range", {}).get("max", "")),
                "relative_path": frame.get("relative_path", ""),
                "output_path": frame.get("output_path", ""),
                "pvd_path": frame.get("pvd_path", ""),
                "sweep_values_json": json.dumps(sweep_values, sort_keys=True),
            }
            writer.writerow(row)
            f_jsonl.write(json.dumps(frame, sort_keys=True) + "\n")
            progress.step()

    return {
        "image_metadata_csv": str(csv_path),
        "image_metadata_jsonl": str(jsonl_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run discover -> render -> post -> tikz pipeline.")
    ap.add_argument("--config", required=True, help="Render pipeline JSON config")
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)

    stamp = utc_stamp()
    output_dir = Path(cfg.get("output_dir") or Path("scripting/renders/out") / stamp).expanduser().resolve()
    clean_output_dir = as_bool(cfg.get("clean_output_dir"), False)
    prepare_output_dir(output_dir, clean_output_dir)

    summary_path = Path(str(cfg["summary_path"])).expanduser().resolve()

    discover_cfg_raw = cfg.get("discovery", {}) or {}
    discover_cfg = DiscoveryConfig(
        out_root=Path(str(discover_cfg_raw.get("out_root", "out"))).expanduser().resolve(),
        include_cases=[str(x) for x in discover_cfg_raw.get("include_cases", [])],
        exclude_cases=[str(x) for x in discover_cfg_raw.get("exclude_cases", [])],
        include_run_ids=[str(x) for x in discover_cfg_raw.get("include_run_ids", [])],
        require_success=as_bool(discover_cfg_raw.get("require_success"), True),
        skip_dry_run=as_bool(discover_cfg_raw.get("skip_dry_run"), True),
        require_dataset=as_bool(discover_cfg_raw.get("require_dataset"), True),
    )

    plan = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "config_path": str(config_path),
        "summary_path": str(summary_path),
        "output_dir": str(output_dir),
        "config": cfg,
    }
    plan_path = output_dir / "render_plan.json"
    write_json(plan_path, plan)

    manifest = discover_from_summary(summary_path, discover_cfg)
    manifest_path = output_dir / "render_manifest.json"
    write_json(manifest_path, manifest)

    artifacts: Dict[str, Any] = {
        "render_plan": str(plan_path),
        "render_manifest": str(manifest_path),
        "render_record_count": manifest.get("record_count", 0),
        "render_skipped_count": manifest.get("skipped_count", 0),
    }

    render_defaults = dict(cfg.get("render", {}) or {})
    render_enabled = as_bool(render_defaults.get("enabled"), True)
    jobs_cfg = cfg.get("render_jobs")
    jobs = normalize_render_jobs(render_defaults, jobs_cfg, render_enabled)

    frames_path = output_dir / "frames_raw.json"
    ranges_path = output_dir / "render_ranges.json"

    if jobs:
        if manifest.get("record_count", 0) == 0:
            raise RuntimeError("discovery produced zero records; nothing to render")

        render_script = Path(__file__).with_name("render_paraview.py").resolve()
        executed_jobs: List[Dict[str, Any]] = []
        jobs_progress = Progress(label="render-jobs", total=len(jobs))
        for job in jobs:
            executed_jobs.append(render_job(output_dir, manifest, job, render_script))
            jobs_progress.step(suffix=f"last={executed_jobs[-1]['name']}")

        frames_path = combine_job_frames(output_dir, executed_jobs)
        ranges_path = combine_job_ranges(output_dir, executed_jobs)

        artifacts.update(
            {
                "render_frames": str(frames_path),
                "render_ranges": str(ranges_path),
                "render_jobs": executed_jobs,
            }
        )

    post_cfg = cfg.get("postprocess", {}) or {}
    post_enabled = as_bool(post_cfg.get("enabled"), False)
    post_frames_path = output_dir / "frames_post.json"
    post_images_dir = output_dir / str(post_cfg.get("out_subdir", "images_post"))

    active_frames_path = frames_path
    source_frames_manifest = post_cfg.get("source_frames_manifest")
    if source_frames_manifest is not None:
        active_frames_path = Path(str(source_frames_manifest)).expanduser().resolve()
        artifacts["post_source_frames_manifest"] = str(active_frames_path)

    if post_enabled:
        if not active_frames_path.exists():
            raise RuntimeError("postprocess enabled but frames manifest does not exist")

        post_script = Path(__file__).with_name("render_post.py").resolve()
        cmd = [
            sys.executable,
            str(post_script),
            "--frames",
            str(active_frames_path),
            "--outdir",
            str(post_images_dir),
            "--write-frames",
            str(post_frames_path),
        ]
        crop = post_cfg.get("crop")
        if crop:
            cmd.extend(["--crop", str(crop)])
        crop_gravity = post_cfg.get("crop_gravity")
        if crop_gravity:
            cmd.extend(["--crop-gravity", str(crop_gravity)])
        trim_bg_color = post_cfg.get("trim_background_color")
        if trim_bg_color:
            cmd.extend(["--trim-bg-color", str(trim_bg_color)])
            trim_bg_fuzz = post_cfg.get("trim_background_fuzz")
            if trim_bg_fuzz is not None:
                cmd.extend(["--trim-bg-fuzz", str(trim_bg_fuzz)])
            trim_bg_border = post_cfg.get("trim_background_border")
            if trim_bg_border is not None:
                cmd.extend(["--trim-bg-border", str(int(trim_bg_border))])
        if as_bool(post_cfg.get("trim"), False):
            cmd.append("--trim")

        run_command(cmd)
        active_frames_path = post_frames_path
        artifacts.update(
            {
                "post_images_dir": str(post_images_dir),
                "post_frames": str(post_frames_path),
            }
        )

    tikz_cfg = cfg.get("tikz", {}) or {}
    tikz_enabled = as_bool(tikz_cfg.get("enabled"), False)

    if tikz_enabled:
        if not active_frames_path.exists():
            raise RuntimeError("tikz enabled but frames manifest does not exist")

        tikz_script = Path(__file__).with_name("render_tikz.py").resolve()
        tikz_out = output_dir / str(tikz_cfg.get("out_file", "figures.tex"))
        tikz_image_width = normalize_latex_length(
            tikz_cfg.get("image_width"), r"0.3\textwidth"
        )
        cmd = [
            sys.executable,
            str(tikz_script),
            "--frames",
            str(active_frames_path),
            "--out",
            str(tikz_out),
            "--group-by",
            str(tikz_cfg.get("group_by", "field")),
            "--image-width",
            tikz_image_width,
            "--max-cols",
            str(int(tikz_cfg.get("max_cols", 3))),
            "--max-rows",
            str(int(tikz_cfg.get("max_rows", 3))),
            "--label-mode",
            str(tikz_cfg.get("label_mode", "short")),
            "--max-sweep-keys",
            str(int(tikz_cfg.get("max_sweep_keys", 2))),
        ]
        for field in tikz_cfg.get("fields", []) or []:
            cmd.extend(["--field", str(field)])
        for token in tikz_cfg.get("time_tokens", []) or []:
            cmd.extend(["--time-token", str(token)])

        run_command(cmd)
        artifacts["tikz_out"] = str(tikz_out)

    latex_cfg = cfg.get("latex", {}) or {}
    latex_enabled = as_bool(latex_cfg.get("enabled"), False)
    if latex_enabled:
        if not tikz_enabled:
            raise RuntimeError("latex enabled but tikz stage is disabled")
        if "tikz_out" not in artifacts:
            raise RuntimeError("latex enabled but tikz output is missing")

        tikz_path = Path(str(artifacts["tikz_out"])).resolve()
        if not tikz_path.exists():
            raise RuntimeError(f"latex enabled but tikz file not found: {tikz_path}")

        wrapper_name = str(latex_cfg.get("wrapper_file", "figures_wrapper.tex"))
        wrapper_path = output_dir / wrapper_name
        rel_tikz = tikz_path.relative_to(output_dir).as_posix()
        preamble_lines = latex_cfg.get("preamble_lines", [])
        if preamble_lines is None:
            preamble_lines = []
        if not isinstance(preamble_lines, list):
            raise ValueError("latex.preamble_lines must be a list")

        wrapper_lines = [
            r"\documentclass{article}",
            r"\usepackage[margin=0.5in]{geometry}",
            r"\usepackage{graphicx}",
            r"\usepackage{tikz}",
            r"\usetikzlibrary{matrix}",
            r"\setlength{\parindent}{0pt}",
            r"\raggedbottom",
        ]
        for line in preamble_lines:
            wrapper_lines.append(str(line))
        wrapper_lines.extend(
            [
                r"\pagestyle{empty}",
                r"\begin{document}",
                rf"\input{{{rel_tikz}}}",
                r"\end{document}",
            ]
        )
        wrapper_path.write_text("\n".join(wrapper_lines) + "\n")

        engine = str(latex_cfg.get("engine", "pdflatex"))
        runs = int(latex_cfg.get("runs", 1))
        if runs < 1:
            runs = 1
        latex_log = output_dir / "latex_compile.log"
        latex_progress = Progress(label="latex-runs", total=runs)
        with latex_log.open("w", encoding="utf-8") as lf:
            for _ in range(runs):
                cmd = [
                    engine,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    wrapper_name,
                ]
                print(
                    "[render-pipeline] cmd:",
                    " ".join(cmd),
                    f"(cwd={output_dir})",
                )
                proc = subprocess.run(
                    cmd,
                    cwd=str(output_dir),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"latex compile failed (see {latex_log})"
                    )
                latex_progress.step()

        pdf_path = output_dir / (Path(wrapper_name).stem + ".pdf")
        if not pdf_path.exists():
            raise RuntimeError(f"expected latex PDF not found: {pdf_path}")

        artifacts["latex_wrapper"] = str(wrapper_path)
        artifacts["latex_log"] = str(latex_log)
        artifacts["latex_pdf"] = str(pdf_path)

    artifacts.update(export_image_metadata_tables(active_frames_path, output_dir))

    summary = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "artifacts": artifacts,
    }
    summary_path_out = output_dir / "pipeline_summary.json"
    write_json(summary_path_out, summary)

    print(f"[render-pipeline] wrote {summary_path_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
