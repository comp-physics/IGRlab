#!/usr/bin/env pvpython
"""Unified ParaView renderer for runner manifests."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from paraview.simple import (  # type: ignore
    Calculator,
    ColorBy,
    GetActiveViewOrCreate,
    GetAnimationScene,
    GetColorTransferFunction,
    GetOpacityTransferFunction,
    GetScalarBar,
    GetTimeKeeper,
    Gradient,
    Hide,
    OpenDataFile,
    Render,
    ResetCamera,
    SaveScreenshot,
    Show,
    Text,
    UpdatePipeline,
)


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
            eta_txt = "?" if not math.isfinite(eta) else fmt_seconds(eta)
            print(
                f"[render-paraview] {self.label} "
                f"{progress_bar(self.done, self.total)} "
                f"{100.0 * frac:5.1f}% ({self.done}/{self.total}) "
                f"elapsed={fmt_seconds(elapsed)} eta={eta_txt}",
                flush=True,
            )
        else:
            print(
                f"[render-paraview] {self.label} "
                f"{progress_bar(0, 0)} done={self.done} "
                f"elapsed={fmt_seconds(elapsed)}",
                flush=True,
            )
        self.last_print = now


def sanitize_token(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_")


def fmt_num(x: float) -> str:
    ax = abs(x)
    if ax != 0.0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.3e}".replace("+", "")
    return f"{x:.6g}"


def compute_time_int_pad_width(times: List[float]) -> int:
    finite = [abs(t) for t in times if math.isfinite(t)]
    if not finite:
        return 1
    max_t = max(finite)
    if max_t < 1.0:
        return 1
    return int(math.floor(math.log10(max_t))) + 1


def token_fraction_digits(tok: str) -> int:
    s = tok.strip()
    if not s:
        return 0
    try:
        d = Decimal(s)
    except InvalidOperation:
        return 0
    exp = d.as_tuple().exponent
    return int(-exp) if exp < 0 else 0


def compute_time_frac_digits(pairs: List[Tuple[str, float]]) -> int:
    if not pairs:
        return 1
    max_digits = 0
    for tok, _t in pairs:
        max_digits = max(max_digits, token_fraction_digits(tok))
    return max(1, max_digits)


def sortable_time_token(t: float, int_pad: int, frac_digits: int) -> str:
    if not math.isfinite(t):
        return sanitize_token(str(t))
    abs_t = abs(t)
    abs_txt = f"{abs_t:.{max(1, frac_digits)}f}"
    whole, frac = abs_txt.split(".", 1)
    out = f"{whole.zfill(max(1, int_pad))}.{frac}"
    if t >= 0.0:
        return out
    return "neg_" + out


def scale_mode_code(mode: str) -> str:
    mapping = {
        "per_frame": "pf",
        "per_dataset": "pd",
        "per_case": "pc",
        "global": "g",
        "fixed": "fx",
    }
    return mapping.get(mode, sanitize_token(mode))


def assoc_code(assoc: str) -> str:
    mapping = {
        "points": "p",
        "cells": "c",
    }
    return mapping.get(assoc.lower(), sanitize_token(assoc.lower()))


def parse_field_colormaps(values: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in values:
        if ":" not in raw:
            raise ValueError(
                f"invalid --field-colormap '{raw}', expected field:preset"
            )
        field, preset = raw.split(":", 1)
        field = field.strip()
        preset = preset.strip()
        if not field or not preset:
            raise ValueError(
                f"invalid --field-colormap '{raw}', expected field:preset"
            )
        out[field] = preset
        out[field.lower()] = preset
    return out


def parse_field_list(values: List[str]) -> set[str]:
    out: set[str] = set()
    for raw in values:
        field = str(raw).strip()
        if not field:
            continue
        out.add(field)
        out.add(field.lower())
    return out


def _find_array_info(proxy, array_name: str):
    di = proxy.GetDataInformation()
    pdi = di.GetPointDataInformation()
    cdi = di.GetCellDataInformation()

    ai = pdi.GetArrayInformation(array_name)
    if ai is not None:
        return "POINTS", ai

    ai = cdi.GetArrayInformation(array_name)
    if ai is not None:
        return "CELLS", ai

    return None, None


def get_array_assoc_ncomp(proxy, array_name: str) -> Tuple[str, int]:
    assoc, ai = _find_array_info(proxy, array_name)
    if ai is None or assoc is None:
        raise RuntimeError(f"Array '{array_name}' not found in point/cell data")
    return assoc, int(ai.GetNumberOfComponents())


def get_array_range(
    proxy, array_name: str, use_magnitude: bool = False
) -> Tuple[Tuple[float, float], str]:
    assoc, ai = _find_array_info(proxy, array_name)
    if ai is None or assoc is None:
        raise RuntimeError(f"Array '{array_name}' not found in point/cell data")

    comp = 0
    if use_magnitude and int(ai.GetNumberOfComponents()) > 1:
        comp = -1

    r = ai.GetComponentRange(comp)
    return (float(r[0]), float(r[1])), assoc


def color_by_array(display, assoc: str, array_name: str, use_magnitude: bool) -> None:
    if use_magnitude:
        ColorBy(display, (assoc, array_name, "Magnitude"))
    else:
        ColorBy(display, (assoc, array_name))


def build_schlieren_spec(src) -> Dict[str, Any]:
    assoc, _ = get_array_assoc_ncomp(src, "pressure")
    grad_name = "grad_pressure"
    grad = Gradient(Input=src)
    grad.ScalarArray = [assoc, "pressure"]
    grad.ResultArrayName = grad_name
    return {
        "name": "schlieren",
        # Render schlieren as |grad(pressure)| by using magnitude directly.
        "proxy": grad,
        "array_name": grad_name,
        "use_magnitude": True,
    }


def build_vorticity_spec(src) -> Dict[str, Any]:
    assoc, ncomp = get_array_assoc_ncomp(src, "velocity")
    if ncomp < 2:
        raise RuntimeError(
            f"velocity must be vector-valued for vorticity (ncomp={ncomp})"
        )

    vel_input = src
    vel_array_name = "velocity"
    if ncomp == 2:
        # ParaView's vorticity computation expects a 3-component vector.
        # Promote 2D velocity to 3D with zero z-component.
        vel3 = Calculator(Input=src)
        vel3.ResultArrayName = "velocity_3d"
        vel3.Function = "velocity_X*iHat + velocity_Y*jHat + 0*kHat"
        vel_input = vel3
        vel_array_name = "velocity_3d"

    grad = Gradient(Input=vel_input)
    grad.ScalarArray = [assoc, vel_array_name]
    grad.ResultArrayName = "grad_velocity"
    try:
        grad.ComputeVorticity = 1
    except Exception:
        pass

    return {
        "name": "vorticity",
        "proxy": grad,
        "array_name": "Vorticity",
        "use_magnitude": True,
    }


def build_field_specs(src, fields: List[str]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for raw in fields:
        field = str(raw).strip()
        lower = field.lower()

        if lower == "schlieren":
            specs.append(build_schlieren_spec(src))
            continue
        if lower == "vorticity":
            specs.append(build_vorticity_spec(src))
            continue

        _assoc, ncomp = get_array_assoc_ncomp(src, field)
        specs.append(
            {
                "name": field,
                "proxy": src,
                "array_name": field,
                # For vector arrays (e.g., momentum, velocity), render magnitude.
                "use_magnitude": ncomp > 1,
            }
        )
    return specs


def set_view_minus_z(view) -> None:
    ResetCamera(view)
    fp = list(view.CameraFocalPoint)
    pos = list(view.CameraPosition)
    dist = math.sqrt(
        (pos[0] - fp[0]) ** 2 + (pos[1] - fp[1]) ** 2 + (pos[2] - fp[2]) ** 2
    )
    if dist < 1e-12:
        dist = 1.0
    view.CameraPosition = [fp[0], fp[1], fp[2] - dist]
    view.CameraFocalPoint = fp
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = 1


def resolve_times(
    timekeeper, time_specs: List[str], use_time_values: bool
) -> List[Tuple[str, float]]:
    available = list(timekeeper.TimestepValues)
    if use_time_values:
        pairs: List[Tuple[str, float]] = []
        seen: set[str] = set()
        for ts in time_specs:
            token = sanitize_token(ts)
            if token in seen:
                continue
            seen.add(token)
            pairs.append((token, float(ts)))
        return pairs

    if not time_specs:
        time_specs = [str(i) for i in range(len(available))]

    pairs: List[Tuple[str, float]] = []
    seen_idx: set[int] = set()
    for ts in time_specs:
        raw = float(ts)
        rounded = round(raw)
        if abs(raw - rounded) > 1e-12:
            raise ValueError(
                "non-integer time token provided while using index mode: "
                f"'{ts}'. Use --use-time-values for physical times."
            )
        idx = int(rounded)
        idx = max(0, min(idx, len(available) - 1))
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        pairs.append((str(idx), float(available[idx])))
    return pairs


def parse_fixed_ranges(values: List[str]) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for raw in values:
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(f"invalid --fixed-range '{raw}', expected field:min:max")
        field, lo_s, hi_s = parts
        lo = float(lo_s)
        hi = float(hi_s)
        if hi < lo:
            raise ValueError(f"invalid --fixed-range '{raw}', max < min")
        out[field] = (lo, hi)
    return out


def load_manifest(path: str) -> Dict[str, Any]:
    data = json.loads(open(path, "r", encoding="utf-8").read())
    if not isinstance(data, dict):
        raise ValueError("manifest root must be JSON object")
    return data


def scope_for(
    mode: str, rec: Dict[str, Any], field: str, ts_token: Optional[str] = None
) -> str:
    if mode == "per_frame":
        assert ts_token is not None
        return f"run:{rec['run_id']}|field:{field}|time:{ts_token}"
    if mode == "per_dataset":
        return f"run:{rec['run_id']}|field:{field}"
    if mode == "per_case":
        return f"case:{rec['case_name']}|field:{field}"
    if mode == "global":
        return f"global|field:{field}"
    raise ValueError(f"unsupported mode {mode}")


def scan_ranges(
    records: List[Dict[str, Any]],
    fields: List[str],
    time_specs: List[str],
    use_time_values: bool,
    scale_mode: str,
) -> Dict[str, Tuple[float, float]]:
    ranges: Dict[str, List[float]] = {}

    for rec_idx, rec in enumerate(records, start=1):
        pvd_path = rec.get("pvd_path")
        if not pvd_path:
            continue

        src = OpenDataFile(pvd_path)
        src.UpdatePipeline()
        specs = build_field_specs(src, fields)

        scene = GetAnimationScene()
        scene.UpdateAnimationUsingDataTimeSteps()
        timekeeper = GetTimeKeeper()
        pairs = resolve_times(timekeeper, list(time_specs), use_time_values)
        rec_total = len(pairs) * len(specs)
        rec_name = sanitize_token(str(rec.get("case_name", f"record_{rec_idx}")))
        progress = Progress(
            label=f"scan record={rec_idx}/{len(records)} case={rec_name}",
            total=rec_total,
        )

        for ts_token, t_val in pairs:
            scene.TimeKeeper.Time = t_val
            UpdatePipeline(time=t_val, proxy=src)
            for spec in specs:
                UpdatePipeline(time=t_val, proxy=spec["proxy"])
                field_name = str(spec["name"])
                (lo, hi), _assoc = get_array_range(
                    spec["proxy"],
                    str(spec["array_name"]),
                    bool(spec["use_magnitude"]),
                )
                key = scope_for(
                    scale_mode,
                    rec,
                    field_name,
                    ts_token if scale_mode == "per_frame" else None,
                )
                cur = ranges.get(key)
                if cur is None:
                    ranges[key] = [lo, hi]
                else:
                    if lo < cur[0]:
                        cur[0] = lo
                    if hi > cur[1]:
                        cur[1] = hi
                progress.step()

    out: Dict[str, Tuple[float, float]] = {}
    for key, pair in ranges.items():
        lo, hi = float(pair[0]), float(pair[1])
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise RuntimeError(f"non-finite range for {key}")
        out[key] = (lo, hi)
    return out


def configure_scalar_bar(field: str, view, lut):
    sb = GetScalarBar(lut, view)
    sb.Title = field
    sb.ComponentTitle = ""
    sb.Orientation = "Vertical"
    sb.LabelColor = [0.0, 0.0, 0.0]
    sb.TitleColor = [0.0, 0.0, 0.0]
    try:
        sb.WindowLocation = "Any Location"
        sb.Position = [0.90, 0.12]
        sb.ScalarBarLength = 0.76
    except Exception:
        pass
    return sb


def hide_all_scalar_bars(scalarbars: Dict[str, Any]) -> None:
    for sb in scalarbars.values():
        try:
            sb.Visibility = 0
        except Exception:
            pass


def run_render(
    records: List[Dict[str, Any]],
    fields: List[str],
    outdir: str,
    width: int,
    height: int,
    time_specs: List[str],
    use_time_values: bool,
    transparent_bg: bool,
    scale_mode: str,
    fixed_ranges: Dict[str, Tuple[float, float]],
    ranges: Dict[str, Tuple[float, float]],
    show_scalar_bar: bool,
    show_title: bool,
    filename_metadata_level: str,
    color_preset: Optional[str],
    field_colormaps: Dict[str, str],
    invert_field_colormaps: set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    os.makedirs(outdir, exist_ok=True)

    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [width, height]
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.UseColorPaletteForBackground = 0
    view.Background = [1.0, 1.0, 1.0]

    title = None
    title_disp = None
    if show_title:
        title = Text()
        title_disp = Show(title, view)
        title_disp.Position = [0.24, 0.92]
        title_disp.FontSize = 16
        try:
            title_disp.Color = [0.0, 0.0, 0.0]
        except Exception:
            pass

    frames: List[Dict[str, Any]] = []
    scalarbars: Dict[str, Any] = {}

    for rec in records:
        pvd_path = rec.get("pvd_path")
        if not pvd_path:
            continue

        src = OpenDataFile(pvd_path)
        src.UpdatePipeline()
        specs = build_field_specs(src, fields)

        display_cache: Dict[Any, Any] = {}

        def ensure_display(proxy):
            disp = display_cache.get(proxy)
            if disp is None:
                disp = Show(proxy, view)
                disp.Representation = "Surface"
                display_cache[proxy] = disp
            return disp

        def hide_all_displays() -> None:
            for disp in display_cache.values():
                try:
                    disp.Visibility = 0
                except Exception:
                    pass

        scene = GetAnimationScene()
        scene.UpdateAnimationUsingDataTimeSteps()
        timekeeper = GetTimeKeeper()
        pairs = resolve_times(timekeeper, list(time_specs), use_time_values)

        case_name_raw = str(rec.get("case_name", "case"))
        run_id_raw = str(rec.get("run_id", "run"))
        dataset_raw = str(rec.get("dataset_name", "dataset"))

        case_name = sanitize_token(case_name_raw)
        run_id = sanitize_token(run_id_raw)
        dataset = sanitize_token(dataset_raw)
        time_int_pad = compute_time_int_pad_width([t for _tok, t in pairs])
        time_frac_digits = compute_time_frac_digits(pairs)
        rec_total = len(pairs) * len(specs)
        progress = Progress(
            label=f"render case={case_name} run={run_id}",
            total=rec_total,
        )

        if re.fullmatch(rf"{re.escape(case_name_raw)}_[0-9]{{4}}", run_id_raw):
            run_folder = dataset
        else:
            run_folder = run_id

        for ts_idx, (ts_token, t_val) in enumerate(pairs):
            scene.TimeKeeper.Time = t_val
            UpdatePipeline(time=t_val, proxy=src)
            for spec in specs:
                UpdatePipeline(time=t_val, proxy=spec["proxy"])

            for spec in specs:
                field_name = str(spec["name"])
                array_name = str(spec["array_name"])
                use_magnitude = bool(spec["use_magnitude"])
                field_folder = sanitize_token(field_name)

                (_local_lo, _local_hi), assoc = get_array_range(
                    spec["proxy"], array_name, use_magnitude
                )

                if scale_mode == "fixed":
                    if field_name not in fixed_ranges:
                        raise RuntimeError(f"field '{field_name}' missing fixed range")
                    lo, hi = fixed_ranges[field_name]
                elif scale_mode == "per_frame":
                    lo, hi = _local_lo, _local_hi
                else:
                    key = scope_for(
                        scale_mode,
                        rec,
                        field_name,
                        ts_token if scale_mode == "per_frame" else None,
                    )
                    if key not in ranges:
                        raise RuntimeError(f"missing range key '{key}'")
                    lo, hi = ranges[key]

                hide_all_displays()
                display = ensure_display(spec["proxy"])
                display.Visibility = 1
                # Keep camera reset after a visible display exists; this avoids
                # first-frame corner framing when no actor is visible yet.
                set_view_minus_z(view)

                color_by_array(display, assoc, array_name, use_magnitude)

                lut = GetColorTransferFunction(array_name)
                preset_name = field_colormaps.get(field_name) or field_colormaps.get(
                    field_name.lower()
                )
                if not preset_name:
                    preset_name = color_preset
                if preset_name:
                    try:
                        lut.ApplyPreset(str(preset_name), True)
                    except Exception as exc:
                        raise RuntimeError(
                            f"failed to apply colormap preset '{preset_name}' for field '{field_name}': {exc}"
                        )
                if (
                    field_name in invert_field_colormaps
                    or field_name.lower() in invert_field_colormaps
                ):
                    try:
                        lut.InvertTransferFunction()
                    except Exception as exc:
                        raise RuntimeError(
                            f"failed to invert colormap for field '{field_name}': {exc}"
                        )

                pwf = GetOpacityTransferFunction(array_name)
                lut.RescaleTransferFunction(lo, hi)
                pwf.RescaleTransferFunction(lo, hi)

                if show_scalar_bar:
                    sb = configure_scalar_bar(field_name, view, lut)
                    scalarbars[field_name] = sb
                    hide_all_scalar_bars(scalarbars)
                    sb.Visibility = 1
                    display.SetScalarBarVisibility(view, True)
                else:
                    hide_all_scalar_bars(scalarbars)
                    display.SetScalarBarVisibility(view, False)

                if title is not None:
                    title.Text = (
                        f"{field_name}  t={fmt_num(t_val)}  {scale_mode}=[{fmt_num(lo)} , {fmt_num(hi)}]"
                    )

                Render(view)

                t_sort_tok = sanitize_token(
                    sortable_time_token(t_val, time_int_pad, time_frac_digits)
                )
                fname_root = f"{sanitize_token(field_name)}_{dataset}_t_{t_sort_tok}"
                if filename_metadata_level in ("legacy", "rich"):
                    lo_tok = sanitize_token(fmt_num(lo))
                    hi_tok = sanitize_token(fmt_num(hi))
                    fname_root += f"_scale_{lo_tok}_to_{hi_tok}"
                if filename_metadata_level == "rich":
                    t_tok = sanitize_token(fmt_num(t_val))
                    ts_tok = sanitize_token(ts_token)
                    mode_tok = scale_mode_code(scale_mode)
                    assoc_tok = assoc_code(str(assoc))
                    if t_tok != ts_tok:
                        fname_root += f"_tv_{t_tok}"
                    if ts_tok != t_sort_tok:
                        fname_root += f"_ts_{ts_tok}"
                    fname_root += f"_m_{mode_tok}_a_{assoc_tok}"
                fname = f"{fname_root}.png"
                rel = os.path.join(case_name, run_folder, field_folder, fname)
                outpath = os.path.join(outdir, rel)
                os.makedirs(os.path.dirname(outpath), exist_ok=True)

                SaveScreenshot(
                    outpath,
                    view,
                    ImageResolution=[width, height],
                    TransparentBackground=1 if transparent_bg else 0,
                )

                frames.append(
                    {
                        "case_name": rec.get("case_name"),
                        "run_id": rec.get("run_id"),
                        "run_folder": run_folder,
                        "dataset_name": rec.get("dataset_name"),
                        "field": field_name,
                        "field_source_array": array_name,
                        "field_use_magnitude": use_magnitude,
                        "field_association": str(assoc).lower(),
                        "time_token": ts_token,
                        "time_index": ts_idx,
                        "time_value": t_val,
                        "scale_mode": scale_mode,
                        "range_min": lo,
                        "range_max": hi,
                        "range": {"min": lo, "max": hi},
                        "relative_path": rel,
                        "output_path": os.path.abspath(outpath),
                        "pvd_path": pvd_path,
                        "sweep_values": rec.get("sweep_values", {}),
                    }
                )
                progress.step()

        hide_all_displays()
        for proxy in display_cache.keys():
            try:
                Hide(proxy, view)
            except Exception:
                pass

    if title is not None:
        Hide(title, view)

    return (
        frames,
        {
            "schema_version": 1,
            "generated_utc": utc_now(),
            "scale_mode": scale_mode,
            "fixed_ranges": {k: [v[0], v[1]] for k, v in fixed_ranges.items()},
            "resolved_ranges": {k: [v[0], v[1]] for k, v in ranges.items()},
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Render fields from runner manifest with ParaView."
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--w", type=int, default=2400)
    ap.add_argument("--h", type=int, default=1600)
    ap.add_argument("--times", nargs="*", default=[])
    ap.add_argument("--use-time-values", action="store_true")
    ap.add_argument("--transparent-bg", action="store_true")
    ap.add_argument(
        "--scale-mode",
        choices=["per_frame", "per_dataset", "per_case", "global", "fixed"],
        default="per_dataset",
    )
    ap.add_argument(
        "--fixed-range",
        action="append",
        default=[],
        help="field:min:max (repeatable; required for --scale-mode fixed)",
    )
    ap.add_argument("--show-scalar-bar", dest="show_scalar_bar", action="store_true")
    ap.add_argument("--hide-scalar-bar", dest="show_scalar_bar", action="store_false")
    ap.set_defaults(show_scalar_bar=True)
    ap.add_argument("--show-title", dest="show_title", action="store_true")
    ap.add_argument("--hide-title", dest="show_title", action="store_false")
    ap.set_defaults(show_title=True)
    ap.add_argument(
        "--filename-metadata-level",
        choices=["minimal", "legacy", "rich"],
        default="rich",
    )
    ap.add_argument(
        "--color-preset",
        default=None,
        help="ParaView colormap preset applied to all fields unless overridden.",
    )
    ap.add_argument(
        "--field-colormap",
        action="append",
        default=[],
        help="Field-specific colormap override: field:preset (repeatable).",
    )
    ap.add_argument(
        "--invert-field-colormap",
        action="append",
        default=[],
        help="Invert colormap for a specific field (repeatable).",
    )
    ap.add_argument("--write-ranges", required=True)
    ap.add_argument("--write-frames", required=True)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    records = [r for r in manifest.get("records", []) if r.get("pvd_path")]
    if not records:
        raise RuntimeError("manifest contains no renderable records")

    fixed_ranges = parse_fixed_ranges(args.fixed_range)
    if args.scale_mode == "fixed" and not fixed_ranges:
        raise RuntimeError("--scale-mode fixed requires at least one --fixed-range")

    if args.scale_mode in ("fixed", "per_frame"):
        ranges: Dict[str, Tuple[float, float]] = {}
    else:
        ranges = scan_ranges(
            records=records,
            fields=list(args.fields),
            time_specs=list(args.times),
            use_time_values=bool(args.use_time_values),
            scale_mode=args.scale_mode,
        )

    field_colormaps = parse_field_colormaps(list(args.field_colormap))
    invert_field_colormaps = parse_field_list(list(args.invert_field_colormap))

    frames, ranges_payload = run_render(
        records=records,
        fields=list(args.fields),
        outdir=args.outdir,
        width=args.w,
        height=args.h,
        time_specs=list(args.times),
        use_time_values=bool(args.use_time_values),
        transparent_bg=bool(args.transparent_bg),
        scale_mode=args.scale_mode,
        fixed_ranges=fixed_ranges,
        ranges=ranges,
        show_scalar_bar=bool(args.show_scalar_bar),
        show_title=bool(args.show_title),
        filename_metadata_level=str(args.filename_metadata_level),
        color_preset=(
            str(args.color_preset).strip() if args.color_preset is not None else None
        ),
        field_colormaps=field_colormaps,
        invert_field_colormaps=invert_field_colormaps,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.write_ranges)), exist_ok=True)
    with open(args.write_ranges, "w", encoding="utf-8") as f:
        json.dump(ranges_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    frames_payload = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "render_outdir": os.path.abspath(args.outdir),
        "manifest_path": os.path.abspath(args.manifest),
        "frame_count": len(frames),
        "frames": frames,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.write_frames)), exist_ok=True)
    with open(args.write_frames, "w", encoding="utf-8") as f:
        json.dump(frames_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(
        f"[render-paraview] wrote frames={len(frames)}"
        f" ranges={args.write_ranges} frames_manifest={args.write_frames}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
