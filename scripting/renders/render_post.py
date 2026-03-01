#!/usr/bin/env python3
"""Postprocess rendered images (crop/trim/copy) using frame manifests."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


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

    def step(self, inc: int = 1) -> None:
        self.done += int(inc)
        now = time.monotonic()
        if self.done >= self.total or (now - self.last_print) >= self.min_interval_s:
            self._print(now=now)

    def _print(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.monotonic()
        elapsed = max(1e-12, now - self.start)
        if self.total > 0:
            frac = max(0.0, min(1.0, float(self.done) / float(self.total)))
            rate = float(self.done) / elapsed if self.done > 0 else 0.0
            remain = max(0, self.total - self.done)
            eta = float(remain) / rate if rate > 1e-12 else float("inf")
            eta_txt = "?" if eta == float("inf") else fmt_seconds(eta)
            print(
                f"[render-post] {self.label} {progress_bar(self.done, self.total)} "
                f"{100.0 * frac:5.1f}% ({self.done}/{self.total}) "
                f"elapsed={fmt_seconds(elapsed)} eta={eta_txt}",
                flush=True,
            )
        else:
            print(
                f"[render-post] {self.label} {progress_bar(0, 0)} done={self.done} "
                f"elapsed={fmt_seconds(elapsed)}",
                flush=True,
            )
        self.last_print = now


def load_frames(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("frames payload must be JSON object")
    if not isinstance(payload.get("frames"), list):
        raise ValueError("frames payload missing 'frames' list")
    return payload


def resolve_imagemagick_cmd() -> List[str]:
    magick = shutil.which("magick")
    if magick:
        return [magick]
    convert = shutil.which("convert")
    if convert:
        return [convert]
    raise RuntimeError("ImageMagick not found (expected 'magick' or 'convert' on PATH)")


def magick_query(cmd_prefix: List[str], src: Path, fmt: str) -> str:
    out = subprocess.check_output(
        cmd_prefix + [str(src), "-format", fmt, "info:"],
        text=True,
    )
    return out.strip()


def detect_corner_background_color(cmd_prefix: List[str], src: Path) -> str:
    wh = magick_query(cmd_prefix, src, "%w %h").split()
    if len(wh) != 2:
        return "black"
    w = max(1, int(wh[0]))
    h = max(1, int(wh[1]))
    coords = [
        (0, 0),
        (w - 1, 0),
        (0, h - 1),
        (w - 1, h - 1),
    ]
    pixels: List[str] = []
    for x, y in coords:
        pix = magick_query(cmd_prefix, src, f"%[pixel:p{{{x},{y}}}]")
        pixels.append(pix)
    if not pixels:
        return "black"
    return Counter(pixels).most_common(1)[0][0]


def run_magick(
    src: Path,
    dst: Path,
    crop: Optional[str],
    trim: bool,
    crop_gravity: Optional[str],
    trim_bg_color: Optional[str],
    trim_bg_fuzz: Optional[str],
    trim_bg_border: int,
) -> None:
    cmd_prefix = resolve_imagemagick_cmd()
    cmd = cmd_prefix + [str(src)]
    if crop:
        if crop_gravity:
            cmd.extend(["-gravity", crop_gravity])
        cmd.extend(["-crop", crop])
        # Normalize canvas/page geometry after crop for consistent downstream layout.
        cmd.append("+repage")
    if trim_bg_color:
        bg_color = str(trim_bg_color)
        if bg_color.lower() == "auto":
            bg_color = detect_corner_background_color(cmd_prefix, src)
        cmd.extend(["-alpha", "off"])
        if trim_bg_border > 0:
            cmd.extend(
                [
                    "-bordercolor",
                    bg_color,
                    "-border",
                    f"{trim_bg_border}x{trim_bg_border}",
                ]
            )
        if trim_bg_fuzz:
            cmd.extend(["-fuzz", str(trim_bg_fuzz)])
        cmd.extend(["-trim", "+repage"])
    if trim:
        cmd.extend(["-trim", "+repage"])
    cmd.append(str(dst))
    subprocess.check_call(cmd)


def process_frames(
    payload: Dict[str, Any],
    outdir: Path,
    crop: Optional[str],
    trim: bool,
    crop_gravity: Optional[str],
    trim_bg_color: Optional[str],
    trim_bg_fuzz: Optional[str],
    trim_bg_border: int,
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    frames: List[Dict[str, Any]] = list(payload.get("frames", []))

    if crop or trim or trim_bg_color:
        _ = resolve_imagemagick_cmd()

    processed: List[Dict[str, Any]] = []
    progress = Progress(label="postprocess", total=len(frames))
    for frame in frames:
        src = Path(str(frame["output_path"])).expanduser().resolve()
        rel = Path(str(frame["relative_path"]))
        dst = (outdir / rel).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)

        if crop or trim or trim_bg_color:
            run_magick(
                src=src,
                dst=dst,
                crop=crop,
                trim=trim,
                crop_gravity=crop_gravity,
                trim_bg_color=trim_bg_color,
                trim_bg_fuzz=trim_bg_fuzz,
                trim_bg_border=int(max(0, trim_bg_border)),
            )
        else:
            shutil.copy2(src, dst)

        item = dict(frame)
        item["source_path"] = str(src)
        item["output_path"] = str(dst)
        item["relative_path"] = str(rel)
        item["postprocess"] = {
            "crop": crop,
            "trim": trim,
            "crop_gravity": crop_gravity,
            "trim_background_color": trim_bg_color,
            "trim_background_fuzz": trim_bg_fuzz,
            "trim_background_border": int(max(0, trim_bg_border)),
        }
        processed.append(item)
        progress.step()

    return {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "source_frames_manifest": payload.get("manifest_path", None),
        "render_outdir": str(outdir),
        "frame_count": len(processed),
        "frames": processed,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Postprocess rendered image frames.")
    ap.add_argument("--frames", required=True, help="Path to frames manifest JSON")
    ap.add_argument("--outdir", required=True, help="Processed image root directory")
    ap.add_argument("--crop", default=None, help="ImageMagick crop geometry, e.g. 3000x2000+600+0")
    ap.add_argument(
        "--crop-gravity",
        default=None,
        help="Optional ImageMagick gravity for crop (e.g. Center, NorthWest).",
    )
    ap.add_argument(
        "--trim-bg-color",
        default=None,
        help="Background color to trim away after optional crop (e.g. black, white).",
    )
    ap.add_argument(
        "--trim-bg-fuzz",
        default="2%",
        help="ImageMagick fuzz for background trim color matching (default: 2%%).",
    )
    ap.add_argument(
        "--trim-bg-border",
        type=int,
        default=1,
        help="Temporary border in pixels before background trim (default: 1).",
    )
    ap.add_argument("--trim", action="store_true", help="Apply ImageMagick trim + repage")
    ap.add_argument("--write-frames", required=True, help="Output processed frames manifest JSON")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    frames_path = Path(args.frames).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    out_frames = Path(args.write_frames).expanduser().resolve()

    payload = load_frames(frames_path)
    processed = process_frames(
        payload=payload,
        outdir=outdir,
        crop=args.crop,
        trim=bool(args.trim),
        crop_gravity=(
            str(args.crop_gravity).strip() if args.crop_gravity is not None else None
        ),
        trim_bg_color=(
            str(args.trim_bg_color).strip() if args.trim_bg_color is not None else None
        ),
        trim_bg_fuzz=(
            str(args.trim_bg_fuzz).strip() if args.trim_bg_fuzz is not None else None
        ),
        trim_bg_border=int(args.trim_bg_border),
    )

    out_frames.parent.mkdir(parents=True, exist_ok=True)
    out_frames.write_text(json.dumps(processed, indent=2, sort_keys=True) + "\n")
    print(f"[render-post] wrote path={out_frames} frames={processed['frame_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
