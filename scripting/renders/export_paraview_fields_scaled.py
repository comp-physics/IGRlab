#!/usr/bin/env pvpython
import os
import re
import argparse
import math

from paraview.simple import (
    OpenDataFile, GetActiveViewOrCreate, Show, Hide, Render,
    GetAnimationScene, GetTimeKeeper, UpdatePipeline,
    ColorBy, GetColorTransferFunction, GetOpacityTransferFunction,
    GetScalarBar, SaveScreenshot, ResetCamera,
    Text
)

def sanitize_basename(s: str) -> str:
    s = os.path.basename(s)
    s = re.sub(r"\.[^.]+$", "", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_")

def fmt_num(x: float) -> str:
    ax = abs(x)
    if ax != 0.0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.3e}".replace("+", "")
    return f"{x:.6g}"

def get_array_range(src, array_name: str):
    """
    Return ((min,max), association) where association is "POINTS" or "CELLS".
    Uses component 0 range. If sigma is vector/tensor, you may want magnitude instead.
    """
    di = src.GetDataInformation()
    pdi = di.GetPointDataInformation()
    cdi = di.GetCellDataInformation()

    ai = pdi.GetArrayInformation(array_name)
    if ai is not None:
        r = ai.GetComponentRange(0)
        return (r[0], r[1]), "POINTS"

    ai = cdi.GetArrayInformation(array_name)
    if ai is not None:
        r = ai.GetComponentRange(0)
        return (r[0], r[1]), "CELLS"

    raise RuntimeError(f"Array '{array_name}' not found in point or cell data.")

def set_view_minus_z_like_button(view):
    """
    Mimic clicking the -Z axis button:
      - ResetCamera (fits data)
      - rotate view direction to look along -Z -> +Z while keeping fitted distance
    """
    ResetCamera(view)

    fp = list(view.CameraFocalPoint)
    pos = list(view.CameraPosition)

    dist = math.sqrt((pos[0]-fp[0])**2 + (pos[1]-fp[1])**2 + (pos[2]-fp[2])**2)
    if dist < 1e-12:
        dist = 1.0

    view.CameraPosition = [fp[0], fp[1], fp[2] - dist]
    view.CameraFocalPoint = fp
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = 1

def configure_scalar_bar(field: str, view, lut):
    """
    Default placement (don’t set WindowLocation/Position), but readable on white.
    """
    sb = GetScalarBar(lut, view)
    sb.Title = field
    sb.ComponentTitle = ""
    sb.Orientation = "Vertical"
    sb.LabelColor = [0.0, 0.0, 0.0]
    sb.TitleColor = [0.0, 0.0, 0.0]
    return sb

def ensure_single_scalar_bar(scalarbars: dict, current_sb):
    for sb in scalarbars.values():
        try:
            sb.Visibility = 0
        except Exception:
            pass
    try:
        current_sb.Visibility = 1
    except Exception:
        pass

def resolve_times_for_file(timekeeper, time_specs, use_time_values):
    """
    Return a list of (t_spec, t_value) pairs for this file.
    - If --times omitted: use all available timesteps indices.
    - If indices: clamp to available indices.
    - If time values: use provided float values directly.
    """
    available_times = list(timekeeper.TimestepValues)

    pairs = []
    if use_time_values:
        for ts in time_specs:
            t = float(ts)
            pairs.append((ts, t))
        return pairs

    # indices
    if not time_specs:
        time_specs = [str(i) for i in range(len(available_times))]
    for ts in time_specs:
        idx = int(float(ts))
        idx = max(0, min(idx, len(available_times) - 1))
        t = available_times[idx]
        pairs.append((str(idx), t))
    return pairs

def scan_global_ranges(inputs, fields, time_specs, use_time_values):
    """
    Pass 1: compute global min/max per field across:
      - all input files
      - all requested timesteps (or all timesteps if none specified)
    """
    global_ranges = {f: [float("inf"), float("-inf")] for f in fields}

    for input_path in inputs:
        src = OpenDataFile(input_path)
        src.UpdatePipeline()

        scene = GetAnimationScene()
        scene.UpdateAnimationUsingDataTimeSteps()
        timekeeper = GetTimeKeeper()

        pairs = resolve_times_for_file(timekeeper, list(time_specs), use_time_values)

        for (ts_token, t) in pairs:
            scene.TimeKeeper.Time = t
            UpdatePipeline(time=t, proxy=src)

            for field in fields:
                (lo, hi), _assoc = get_array_range(src, field)
                if lo < global_ranges[field][0]:
                    global_ranges[field][0] = lo
                if hi > global_ranges[field][1]:
                    global_ranges[field][1] = hi

        Hide(src)
        # (ParaView GC is a bit magical; hiding is usually enough for long runs)

    # Convert lists -> tuples
    out = {}
    for f, (lo, hi) in global_ranges.items():
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise RuntimeError(f"Global range for '{f}' could not be computed.")
        out[f] = (lo, hi)
    return out

def render_all(inputs, fields, time_specs, use_time_values, outdir, w, h, transparent_bg, global_ranges):
    """
    Pass 2: render all images using the global range per field.
    """
    os.makedirs(outdir, exist_ok=True)

    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [w, h]
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.UseColorPaletteForBackground = 0
    view.Background = [1.0, 1.0, 1.0]

    # Title overlay (black text)
    title = Text()
    title_disp = Show(title, view)
    title_disp.Position = [0.30, 0.92]
    title_disp.FontSize = 18
    try:
        title_disp.Color = [0.0, 0.0, 0.0]
    except Exception:
        pass

    scalarbars = {}

    for input_path in inputs:
        case_tag = sanitize_basename(input_path)
        case_outdir = os.path.join(outdir, case_tag)
        os.makedirs(case_outdir, exist_ok=True)

        src = OpenDataFile(input_path)
        src.UpdatePipeline()

        display = Show(src, view)
        display.Representation = "Surface"

        scene = GetAnimationScene()
        scene.UpdateAnimationUsingDataTimeSteps()
        timekeeper = GetTimeKeeper()
        pairs = resolve_times_for_file(timekeeper, list(time_specs), use_time_values)

        for (ts_token, t) in pairs:
            scene.TimeKeeper.Time = t
            UpdatePipeline(time=t, proxy=src)

            set_view_minus_z_like_button(view)

            for field in fields:
                # association for this file
                (_lo_local, _hi_local), assoc = get_array_range(src, field)
                glo, ghi = global_ranges[field]

                if assoc == "POINTS":
                    ColorBy(display, ("POINTS", field))
                else:
                    ColorBy(display, ("CELLS", field))

                lut = GetColorTransferFunction(field)
                pwf = GetOpacityTransferFunction(field)

                # Force GLOBAL scaling
                lut.RescaleTransferFunction(glo, ghi)
                pwf.RescaleTransferFunction(glo, ghi)

                sb = configure_scalar_bar(field, view, lut)
                scalarbars[field] = sb
                ensure_single_scalar_bar(scalarbars, sb)
                display.SetScalarBarVisibility(view, True)

                title.Text = (
                    f"{field}   t={fmt_num(t)}   "
                    f"GLOBAL scale=[{fmt_num(glo)} , {fmt_num(ghi)}]"
                )

                Render(view)

                fname = (
                    f"{sanitize_basename(field)}_{case_tag}"
                    f"_ts_{sanitize_basename(ts_token)}"
                    f"_scale_{fmt_num(glo)}_to_{fmt_num(ghi)}.png"
                )
                outpath = os.path.join(case_outdir, fname)

                SaveScreenshot(
                    outpath,
                    view,
                    ImageResolution=[w, h],
                    TransparentBackground=1 if transparent_bg else 0,
                )
                print(outpath)

        Hide(src, view)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--outdir", default="renders_scaled")
    ap.add_argument("--w", type=int, default=2400)
    ap.add_argument("--h", type=int, default=1600)
    ap.add_argument("--times", nargs="*", default=[],
                    help="Indices (default) or time values if --use-time-values. Omit => all timesteps per file.")
    ap.add_argument("--use-time-values", action="store_true")
    ap.add_argument("--transparent-bg", action="store_true")
    ap.add_argument("--write-range-summary", action="store_true",
                    help="Write global ranges to outdir/_global_ranges.txt")
    args = ap.parse_args()

    # Pass 1: scan global ranges
    global_ranges = scan_global_ranges(args.inputs, args.fields, args.times, args.use_time_values)

    os.makedirs(args.outdir, exist_ok=True)
    if args.write_range_summary:
        summary_path = os.path.join(args.outdir, "_global_ranges.txt")
        with open(summary_path, "w") as f:
            for field in args.fields:
                lo, hi = global_ranges[field]
                f.write(f"{field}: {lo} {hi}\n")
        print(f"Wrote {summary_path}")

    # Pass 2: render with global ranges
    render_all(
        inputs=args.inputs,
        fields=args.fields,
        time_specs=args.times,
        use_time_values=args.use_time_values,
        outdir=args.outdir,
        w=args.w,
        h=args.h,
        transparent_bg=args.transparent_bg,
        global_ranges=global_ranges,
    )

if __name__ == "__main__":
    main()
