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
      - rotate view direction to look along -Z -> +Z while keeping distance
    """
    ResetCamera(view)

    fp = list(view.CameraFocalPoint)
    pos = list(view.CameraPosition)

    dist = math.sqrt((pos[0]-fp[0])**2 + (pos[1]-fp[1])**2 + (pos[2]-fp[2])**2)
    if dist < 1e-12:
        dist = 1.0

    # camera on negative Z side, looking at focal point
    view.CameraPosition = [fp[0], fp[1], fp[2] - dist]
    view.CameraFocalPoint = fp
    view.CameraViewUp = [0.0, 1.0, 0.0]

    # Optional but usually desired for “2D plot-like” look:
    view.CameraParallelProjection = 1

def ensure_single_scalar_bar(view, scalarbars: dict, current_sb):
    """
    Hide all scalar bars we have created, then show only current_sb.
    This avoids duplicates without ever touching the main dataset representation.
    """
    for sb in scalarbars.values():
        try:
            sb.Visibility = 0
        except Exception:
            pass
    try:
        current_sb.Visibility = 1
    except Exception:
        pass

def configure_scalar_bar(field: str, view, lut):
    """
    Use ParaView default placement (don’t set WindowLocation/Position),
    but make text visible on white background.
    """
    sb = GetScalarBar(lut, view)
    sb.Title = field
    sb.ComponentTitle = ""
    sb.Orientation = "Vertical"

    # Make labels readable on white background
    sb.LabelColor = [0.0, 0.0, 0.0]
    sb.TitleColor = [0.0, 0.0, 0.0]

    return sb

def export_one_case(input_path, fields, outdir, width, height, time_specs, use_time_values, transparent_bg):
    case_tag = sanitize_basename(input_path)
    case_outdir = os.path.join(outdir, case_tag)
    os.makedirs(case_outdir, exist_ok=True)

    src = OpenDataFile(input_path)
    src.UpdatePipeline()

    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [width, height]
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.UseColorPaletteForBackground = 0
    view.Background = [1.0, 1.0, 1.0]

    display = Show(src, view)
    display.Representation = "Surface"

    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    timekeeper = GetTimeKeeper()
    available_times = list(timekeeper.TimestepValues)

    # Title annotation
    title = Text()
    title_disp = Show(title, view)
    # Avoid enum churn; just position it
    title_disp.Position = [0.30, 0.92]
    title_disp.FontSize = 18
    try:
        title_disp.Color = [0.0, 0.0, 0.0]
    except Exception:
        pass

    # Track scalar bars we create so we can hide/show deterministically
    scalarbars = {}

    def set_time(t_spec: str) -> float:
        if use_time_values:
            t = float(t_spec)
            scene.TimeKeeper.Time = t
            UpdatePipeline(time=t, proxy=src)
            return t
        else:
            idx = int(float(t_spec))
            idx = max(0, min(idx, len(available_times) - 1))
            t = available_times[idx]
            scene.TimeKeeper.Time = t
            UpdatePipeline(time=t, proxy=src)
            return t

    if not time_specs:
        time_specs = [str(i) for i in range(len(available_times))]

    for t_spec in time_specs:
        t = set_time(t_spec)

        # Important: fit + orient every timestep (like clicking -Z)
        set_view_minus_z_like_button(view)

        ts_token = t_spec if not use_time_values else fmt_num(t)

        for field in fields:
            (lo, hi), assoc = get_array_range(src, field)

            if assoc == "POINTS":
                ColorBy(display, ("POINTS", field))
            else:
                ColorBy(display, ("CELLS", field))

            lut = GetColorTransferFunction(field)
            pwf = GetOpacityTransferFunction(field)

            lut.RescaleTransferFunction(lo, hi)
            pwf.RescaleTransferFunction(lo, hi)

            # Create/configure scalar bar for this LUT and ensure only one shows
            sb = configure_scalar_bar(field, view, lut)
            scalarbars[field] = sb
            ensure_single_scalar_bar(view, scalarbars, sb)
            display.SetScalarBarVisibility(view, True)

            title.Text = f"{field}   t={fmt_num(t)}   scale=[{fmt_num(lo)} , {fmt_num(hi)}]"

            Render(view)

            # Filename: FIELDNAME_INPUTNAME_ts_TIMESTEP_scale_LOW_to_HIGH.png
            fname = (
                f"{sanitize_basename(field)}_{case_tag}"
                f"_ts_{sanitize_basename(ts_token)}"
                f"_scale_{fmt_num(lo)}_to_{fmt_num(hi)}.png"
            )
            outpath = os.path.join(case_outdir, fname)

            SaveScreenshot(
                outpath,
                view,
                ImageResolution=[width, height],
                TransparentBackground=1 if transparent_bg else 0,
            )
            print(outpath)

    Hide(title, view)
    Hide(src, view)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--outdir", default="renders")
    ap.add_argument("--w", type=int, default=2400)
    ap.add_argument("--h", type=int, default=1600)
    ap.add_argument("--times", nargs="*", default=[])
    ap.add_argument("--use-time-values", action="store_true")
    ap.add_argument("--transparent-bg", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for p in args.inputs:
        export_one_case(
            input_path=p,
            fields=args.fields,
            outdir=args.outdir,
            width=args.w,
            height=args.h,
            time_specs=args.times,
            use_time_values=args.use_time_values,
            transparent_bg=args.transparent_bg,
        )

if __name__ == "__main__":
    main()
