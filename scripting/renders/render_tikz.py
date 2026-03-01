#!/usr/bin/env python3
"""Generate TikZ wrappers for rendered frame manifests."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


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

    def _print(self, now: float | None = None) -> None:
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
                f"[render-tikz] {self.label} {progress_bar(self.done, self.total)} "
                f"{100.0 * frac:5.1f}% ({self.done}/{self.total}) "
                f"elapsed={fmt_seconds(elapsed)} eta={eta_txt}",
                flush=True,
            )
        else:
            print(
                f"[render-tikz] {self.label} {progress_bar(0, 0)} done={self.done} "
                f"elapsed={fmt_seconds(elapsed)}",
                flush=True,
            )
        self.last_print = now


def load_frames(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("frames payload must be JSON object")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("frames payload missing 'frames' list")
    return payload


def filter_frames(
    frames: Sequence[Dict[str, Any]],
    fields: List[str],
    time_tokens: List[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    field_set = set(fields)
    time_set = set(time_tokens)
    for frame in frames:
        if field_set and str(frame.get("field")) not in field_set:
            continue
        if time_set and str(frame.get("time_token")) not in time_set:
            continue
        out.append(frame)
    return out


def latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def frame_label(frame: Dict[str, Any], label_mode: str, max_sweep_keys: int) -> str:
    case = str(frame.get("case_name", ""))
    run = str(frame.get("run_id", ""))
    token = str(frame.get("time_token", ""))
    if label_mode == "none":
        return ""
    if label_mode == "run":
        return run
    if label_mode == "short":
        return f"{run} / t={token}"
    sweep = frame.get("sweep_values", {})
    if isinstance(sweep, dict) and sweep:
        items = [f"{k}={v}" for k, v in sorted(sweep.items())]
        if max_sweep_keys >= 0:
            items = items[:max_sweep_keys]
        return f"{case} / {run} / {'; '.join(items)}"
    return f"{case} / {run}"


def chunk_items(
    items: Sequence[Dict[str, Any]], max_cols: int, max_rows: int
) -> List[List[Dict[str, Any]]]:
    page_size = max(1, max_cols) * max(1, max_rows)
    chunks: List[List[Dict[str, Any]]] = []
    for i in range(0, len(items), page_size):
        chunks.append(list(items[i : i + page_size]))
    return chunks


def write_group_matrix(
    lines: List[str],
    group_title: str,
    items: List[Dict[str, Any]],
    out_tex: Path,
    image_width: str,
    max_cols: int,
    label_mode: str,
    max_sweep_keys: int,
) -> None:
    if not items:
        lines.append("% (no frames in this group)")
        return

    lines.append(r"\begin{figure}[p]")
    lines.append(r"\centering")
    lines.append(rf"\textbf{{{latex_escape(group_title)}}}\\[2mm]")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\matrix[matrix of nodes, row sep=3mm, column sep=3mm]{")

    row: List[str] = []
    for idx, item in enumerate(items, start=1):
        img = Path(str(item["output_path"])).resolve()
        try:
            rel = img.relative_to(out_tex.parent.resolve())
            rel_tex = rel.as_posix()
        except ValueError:
            rel_tex = Path(
                os.path.relpath(str(img), str(out_tex.parent.resolve()))
            ).as_posix()
        rel_tex = rel_tex.replace("_", r"\_")
        label = latex_escape(frame_label(item, label_mode, max_sweep_keys))
        if label:
            cell = (
                r"\node[align=center,inner sep=0pt]{"
                + rf"\includegraphics[width={image_width}]{{{rel_tex}}}"
                + r"\\"
                + rf"\scriptsize {label}"
                + r"};"
            )
        else:
            cell = (
                r"\node[align=center,inner sep=0pt]{"
                + rf"\includegraphics[width={image_width}]{{{rel_tex}}}"
                + r"};"
            )
        row.append(cell)

        if len(row) == max_cols or idx == len(items):
            lines.append("  " + " & ".join(row) + r"\\")
            row = []

    lines.append("};")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{figure}")
    lines.append(r"\clearpage")


def generate_tikz(
    payload: Dict[str, Any],
    out_tex: Path,
    group_by: str,
    image_width: str,
    max_cols: int,
    max_rows: int,
    fields: List[str],
    time_tokens: List[str],
    label_mode: str,
    max_sweep_keys: int,
) -> Dict[str, Any]:
    frames = filter_frames(list(payload.get("frames", [])), fields, time_tokens)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        if group_by == "case":
            key = str(frame.get("case_name", "unknown"))
        elif group_by == "job":
            key = str(frame.get("job_name", "unknown"))
        elif group_by == "job_field":
            key = f"{frame.get('job_name', 'unknown')} / {frame.get('field', 'unknown')}"
        elif group_by == "case_field":
            key = f"{frame.get('case_name', 'unknown')} / {frame.get('field', 'unknown')}"
        elif group_by == "time":
            key = str(frame.get("time_token", "unknown"))
        else:
            key = str(frame.get("field", "unknown"))
        grouped[key].append(frame)

    for group_items in grouped.values():
        group_items.sort(
            key=lambda f: (
                str(f.get("case_name", "")),
                str(f.get("run_id", "")),
                str(f.get("field", "")),
                float(f.get("time_value", 0.0)),
            )
        )

    lines: List[str] = []
    lines.append("% Auto-generated by scripting/renders/render_tikz.py")
    lines.append(f"% generated_utc: {utc_now()}")
    lines.append(
        "% Requires: \\usepackage{tikz}, \\usetikzlibrary{matrix}, and \\usepackage{graphicx}"
    )
    lines.append("")

    for key in sorted(grouped.keys()):
        items = grouped[key]
        pages = chunk_items(items, max_cols=max_cols, max_rows=max_rows)
        page_progress = Progress(
            label=f"group={key}",
            total=len(pages),
            min_interval_s=0.5,
        )
        for idx, page_items in enumerate(pages, start=1):
            page_title = key
            if len(pages) > 1:
                page_title = f"{key} ({idx}/{len(pages)})"
            lines.append(f"% group: {key} page {idx}/{len(pages)}")
            write_group_matrix(
                lines,
                page_title,
                page_items,
                out_tex,
                image_width,
                max_cols,
                label_mode,
                max_sweep_keys,
            )
            lines.append("")
            page_progress.step()

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n")

    return {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "out_tex": str(out_tex),
        "group_by": group_by,
        "max_rows": max_rows,
        "group_count": len(grouped),
        "frame_count": len(frames),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate TikZ figure wrapper from frame manifest.")
    ap.add_argument("--frames", required=True, help="Frames manifest JSON")
    ap.add_argument("--out", required=True, help="Output .tex path")
    ap.add_argument(
        "--group-by",
        choices=["field", "case", "time", "job", "job_field", "case_field"],
        default="field",
        help="How to split into figure groups",
    )
    ap.add_argument("--image-width", default=r"0.3\textwidth")
    ap.add_argument("--max-cols", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=3)
    ap.add_argument(
        "--label-mode",
        choices=["none", "run", "short", "full"],
        default="short",
    )
    ap.add_argument("--max-sweep-keys", type=int, default=2)
    ap.add_argument("--field", action="append", default=[], help="Field filter (repeatable)")
    ap.add_argument("--time-token", action="append", default=[], help="Time-token filter (repeatable)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    frames_path = Path(args.frames).expanduser().resolve()
    out_tex = Path(args.out).expanduser().resolve()

    payload = load_frames(frames_path)
    summary = generate_tikz(
        payload=payload,
        out_tex=out_tex,
        group_by=args.group_by,
        image_width=args.image_width,
        max_cols=max(1, int(args.max_cols)),
        max_rows=max(1, int(args.max_rows)),
        fields=list(args.field),
        time_tokens=list(args.time_token),
        label_mode=str(args.label_mode),
        max_sweep_keys=int(args.max_sweep_keys),
    )
    print(
        f"[render-tikz] wrote path={summary['out_tex']}"
        f" groups={summary['group_count']} frames={summary['frame_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
