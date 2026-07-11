#!/usr/bin/env python3
"""Compare point-data arrays between two VTK PVD time series.

This intentionally stays small and dependency-free. It is used by the L2
CPU-gold drift workflow to compare saved MFEM VTK/PVD fields across multiple
frames.
"""

from __future__ import annotations

import argparse
import array
import base64
import json
import math
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Frame = Dict[str, object]
NumericArray = array.array

DEFAULT_ARRAYS = [
    "density",
    "pressure",
    "sigma",
    "energy",
    "momentum",
    "velocity",
    "zhang_shu_active",
    "zhang_shu_theta_p",
    "zhang_shu_theta_rho",
]


def typecode_for_vtk_type(vtk_type: str) -> str:
    if vtk_type == "Float64":
        return "d"
    if vtk_type == "Float32":
        return "f"
    if vtk_type in ("Int32", "UInt32"):
        return "i"
    if vtk_type in ("Int64", "UInt64"):
        return "q"
    if vtk_type == "UInt8":
        return "B"
    raise ValueError(f"unsupported VTK DataArray type: {vtk_type}")


def parse_pvd(path: Path) -> List[Frame]:
    root = ET.parse(path).getroot()
    frames: List[Frame] = []
    for dataset in root.findall(".//DataSet"):
        file_name = dataset.get("file")
        if not file_name:
            continue
        timestep_raw = dataset.get("timestep", "")
        try:
            timestep = float(timestep_raw)
        except ValueError:
            timestep = math.nan
        frames.append(
            {
                "timestep": timestep,
                "timestep_raw": timestep_raw,
                "file": str((path.parent / file_name).resolve()),
            }
        )
    return frames


def read_vtk_binary_data(text: Optional[str], vtk_type: str) -> NumericArray:
    out = array.array(typecode_for_vtk_type(vtk_type))
    if not text:
        return out
    compact = "".join(text.split())
    if not compact:
        return out

    # MFEM's XML VTK binary writer emits a 32-bit little-endian byte-count
    # header as a padded base64 block, immediately followed by the payload.
    if len(compact) >= 8:
        header = base64.b64decode(compact[:8])
        payload_size = int.from_bytes(header[:4], byteorder="little", signed=False)
        payload = base64.b64decode(compact[8:])
        if payload_size <= len(payload):
            payload = payload[:payload_size]
        out.frombytes(payload)
        if sys.byteorder != "little":
            out.byteswap()
        return out

    return out


def iter_piece_paths(dataset_file: Path) -> Iterable[Path]:
    root = ET.parse(dataset_file).getroot()
    if root.get("type") == "PUnstructuredGrid":
        for piece in root.findall(".//Piece"):
            source = piece.get("Source")
            if source:
                yield (dataset_file.parent / source).resolve()
    else:
        yield dataset_file.resolve()


def load_point_arrays(dataset_file: Path,
                      requested_arrays: Optional[Sequence[str]]) -> Dict[str, NumericArray]:
    names = set(requested_arrays) if requested_arrays else None
    merged: Dict[str, NumericArray] = {}
    for piece_path in iter_piece_paths(dataset_file):
        root = ET.parse(piece_path).getroot()
        for data_array in root.findall(".//PointData/DataArray"):
            name = data_array.get("Name")
            if not name or (names is not None and name not in names):
                continue
            vtk_type = data_array.get("type", "")
            values = read_vtk_binary_data(data_array.text, vtk_type)
            if name in merged:
                merged[name].extend(values)
            else:
                merged[name] = values
    return merged


def array_stats(values: NumericArray) -> Dict[str, float]:
    count = len(values)
    if count == 0:
        return {
            "count": 0,
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "rms": math.nan,
        }
    total = math.fsum(float(v) for v in values)
    total_sq = math.fsum(float(v) * float(v) for v in values)
    return {
        "count": count,
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": total / count,
        "rms": math.sqrt(total_sq / count),
    }


def diff_stats(a: NumericArray, b: NumericArray) -> Dict[str, float]:
    if len(a) != len(b):
        raise ValueError(f"array length mismatch: {len(a)} != {len(b)}")
    count = len(a)
    if count == 0:
        return {
            "count": 0,
            "l2": math.nan,
            "rms": math.nan,
            "rel_rms": math.nan,
            "max_abs": math.nan,
            "mean_abs": math.nan,
        }
    max_abs = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    ref_sq = 0.0
    for av, bv in zip(a, b):
        af = float(av)
        bf = float(bv)
        diff = bf - af
        adiff = abs(diff)
        max_abs = max(max_abs, adiff)
        sum_abs += adiff
        sum_sq += diff * diff
        ref_sq += af * af
    l2 = math.sqrt(sum_sq)
    rms = math.sqrt(sum_sq / count)
    ref_rms = math.sqrt(ref_sq / count)
    rel_rms = rms / ref_rms if ref_rms > 0.0 else (0.0 if rms == 0.0 else math.inf)
    return {
        "count": count,
        "l2": l2,
        "rms": rms,
        "rel_rms": rel_rms,
        "max_abs": max_abs,
        "mean_abs": sum_abs / count,
    }


def as_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def common_timesteps(a: Sequence[Frame], b: Sequence[Frame], tol: float) -> List[Tuple[Frame, Frame]]:
    pairs: List[Tuple[Frame, Frame]] = []
    used_b = set()
    for frame_a in a:
        time_a = float(frame_a["timestep"])
        best_idx = None
        best_diff = float("inf")
        for idx, frame_b in enumerate(b):
            if idx in used_b:
                continue
            diff = abs(float(frame_b["timestep"]) - time_a)
            if diff < best_diff:
                best_idx = idx
                best_diff = diff
        if best_idx is not None and best_diff <= tol:
            used_b.add(best_idx)
            pairs.append((frame_a, b[best_idx]))
    return pairs


def compare_frames(frame_a: Frame,
                   frame_b: Frame,
                   arrays: Optional[Sequence[str]]) -> Dict[str, object]:
    arrays_a = load_point_arrays(Path(str(frame_a["file"])), arrays)
    arrays_b = load_point_arrays(Path(str(frame_b["file"])), arrays)
    names = sorted(set(arrays_a).intersection(arrays_b))
    result_arrays: Dict[str, object] = {}
    for name in names:
        a_values = arrays_a[name]
        b_values = arrays_b[name]
        result_arrays[name] = {
            "a": array_stats(a_values),
            "b": array_stats(b_values),
            "diff": diff_stats(a_values, b_values),
        }
    return {
        "timestep": frame_a["timestep"],
        "a_timestep": frame_a["timestep"],
        "b_timestep": frame_b["timestep"],
        "time_diff": abs(
            as_float(frame_b.get("timestep")) - as_float(frame_a.get("timestep"))
        ),
        "a_file": frame_a["file"],
        "b_file": frame_b["file"],
        "arrays": result_arrays,
    }


def update_worst_frame(entry: Dict[str, object],
                       metric: str,
                       value: object,
                       frame_idx: int,
                       frame: Dict[str, object]) -> None:
    numeric_value = as_float(value)
    if not math.isfinite(numeric_value):
        return
    current_value = as_float(entry.get(metric), -math.inf)
    current_frame = int(as_float(entry.get(f"{metric}_frame_index"), -1.0))
    if current_frame >= 0 and numeric_value <= current_value:
        return
    entry[metric] = numeric_value
    entry[f"{metric}_frame_index"] = frame_idx
    entry[f"{metric}_timestep"] = as_float(frame.get("timestep"))
    entry[f"{metric}_a_timestep"] = as_float(frame.get("a_timestep"))
    entry[f"{metric}_b_timestep"] = as_float(frame.get("b_timestep"))
    entry[f"{metric}_a_file"] = str(frame.get("a_file", ""))
    entry[f"{metric}_b_file"] = str(frame.get("b_file", ""))


def aggregate_frame_diffs(frames: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_name: Dict[str, Dict[str, object]] = {}
    for frame_idx, frame in enumerate(frames):
        arrays = frame.get("arrays", {})
        if not isinstance(arrays, dict):
            continue
        for name, payload in arrays.items():
            if not isinstance(payload, dict):
                continue
            diff = payload.get("diff", {})
            if not isinstance(diff, dict):
                continue
            entry = by_name.setdefault(
                name,
                {
                    "frames": 0,
                    "max_abs": 0.0,
                    "max_rms": 0.0,
                    "max_rel_rms": 0.0,
                    "final_rms": 0.0,
                    "final_rel_rms": 0.0,
                    "final_max_abs": 0.0,
                    "max_abs_frame_index": -1,
                    "max_abs_timestep": math.nan,
                    "max_abs_a_timestep": math.nan,
                    "max_abs_b_timestep": math.nan,
                    "max_abs_a_file": "",
                    "max_abs_b_file": "",
                    "max_rms_frame_index": -1,
                    "max_rms_timestep": math.nan,
                    "max_rms_a_timestep": math.nan,
                    "max_rms_b_timestep": math.nan,
                    "max_rms_a_file": "",
                    "max_rms_b_file": "",
                    "max_rel_rms_frame_index": -1,
                    "max_rel_rms_timestep": math.nan,
                    "max_rel_rms_a_timestep": math.nan,
                    "max_rel_rms_b_timestep": math.nan,
                    "max_rel_rms_a_file": "",
                    "max_rel_rms_b_file": "",
                },
            )
            entry["frames"] += 1
            update_worst_frame(entry, "max_abs", diff.get("max_abs"), frame_idx, frame)
            update_worst_frame(entry, "max_rms", diff.get("rms"), frame_idx, frame)
            rel = diff.get("rel_rms", math.nan)
            update_worst_frame(entry, "max_rel_rms", rel, frame_idx, frame)
            entry["final_rms"] = float(diff.get("rms", 0.0))
            entry["final_max_abs"] = float(diff.get("max_abs", 0.0))
            if isinstance(rel, (int, float)) and math.isfinite(float(rel)):
                entry["final_rel_rms"] = float(rel)
    return by_name


def worst_aggregate_metrics(aggregate: Dict[str, object]) -> Dict[str, object]:
    worst: Dict[str, object] = {}
    for metric in ("max_abs", "max_rms", "max_rel_rms"):
        best_name = ""
        best_payload: Optional[Dict[str, object]] = None
        best_value = -math.inf
        for name, payload in aggregate.items():
            if not isinstance(payload, dict):
                continue
            value = as_float(payload.get(metric), -math.inf)
            if math.isfinite(value) and value > best_value:
                best_name = name
                best_payload = payload
                best_value = value
        if best_payload is None:
            continue
        worst[metric] = {
            "array": best_name,
            "value": best_value,
            "frame_index": int(as_float(best_payload.get(f"{metric}_frame_index"), -1.0)),
            "timestep": as_float(best_payload.get(f"{metric}_timestep")),
            "a_timestep": as_float(best_payload.get(f"{metric}_a_timestep")),
            "b_timestep": as_float(best_payload.get(f"{metric}_b_timestep")),
            "a_file": str(best_payload.get(f"{metric}_a_file", "")),
            "b_file": str(best_payload.get(f"{metric}_b_file", "")),
        }
    return worst


def write_markdown(path: Path, payload: Dict[str, object]) -> None:
    lines: List[str] = [
        "# VTK PVD Field Comparison",
        "",
        f"- Generated (UTC): {datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- A: `{payload['a_pvd']}`",
        f"- B: `{payload['b_pvd']}`",
        f"- Matched frames: {payload['matched_frames']}",
        "",
        "| array | frames | max_abs | max_abs_time | max_rms | max_rel_rms | max_rel_rms_time | final_max_abs | final_rms | final_rel_rms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    aggregate = payload.get("aggregate", {})
    if isinstance(aggregate, dict):
        for name in sorted(aggregate):
            row = aggregate[name]
            if not isinstance(row, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        str(row.get("frames", "")),
                        f"{float(row.get('max_abs', math.nan)):.6g}",
                        f"{float(row.get('max_abs_timestep', math.nan)):.6g}",
                        f"{float(row.get('max_rms', math.nan)):.6g}",
                        f"{float(row.get('max_rel_rms', math.nan)):.6g}",
                        f"{float(row.get('max_rel_rms_timestep', math.nan)):.6g}",
                        f"{float(row.get('final_max_abs', math.nan)):.6g}",
                        f"{float(row.get('final_rms', math.nan)):.6g}",
                        f"{float(row.get('final_rel_rms', math.nan)):.6g}",
                    ]
                )
                + " |"
            )
    worst = payload.get("worst", {})
    if isinstance(worst, dict) and worst:
        lines.extend(
            [
                "",
                "## Worst Metrics",
                "",
                "| metric | array | frame | timestep | value |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for metric in ("max_abs", "max_rms", "max_rel_rms"):
            row = worst.get(metric, {})
            if not isinstance(row, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        metric,
                        str(row.get("array", "")),
                        str(row.get("frame_index", "")),
                        f"{float(row.get('timestep', math.nan)):.6g}",
                        f"{float(row.get('value', math.nan)):.6g}",
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare point-data arrays between two VTK PVD time series."
    )
    parser.add_argument("--a-pvd", required=True)
    parser.add_argument("--b-pvd", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--arrays", nargs="+", default=DEFAULT_ARRAYS)
    parser.add_argument("--time-tol", type=float, default=1.0e-12)
    args = parser.parse_args()

    a_pvd = Path(args.a_pvd)
    b_pvd = Path(args.b_pvd)
    frames_a = parse_pvd(a_pvd)
    frames_b = parse_pvd(b_pvd)
    pairs = common_timesteps(frames_a, frames_b, args.time_tol)
    if not pairs:
        raise RuntimeError("no matching PVD timesteps")

    frame_results = [compare_frames(a, b, args.arrays) for a, b in pairs]
    aggregate = aggregate_frame_diffs(frame_results)
    payload: Dict[str, object] = {
        "schema_version": 1,
        "benchmark": "vtk_pvd_field_comparison",
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "a_pvd": str(a_pvd),
        "b_pvd": str(b_pvd),
        "a_frames": len(frames_a),
        "b_frames": len(frames_b),
        "matched_frames": len(pairs),
        "arrays": args.arrays,
        "frames": frame_results,
        "aggregate": aggregate,
        "worst": worst_aggregate_metrics(aggregate),
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    write_markdown(output_md, payload)
    print(f"vtk-pvd-compare: wrote {output_json}")
    print(f"vtk-pvd-compare: wrote {output_md}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vtk-pvd-compare: ERROR: {exc}", file=sys.stderr)
        raise
