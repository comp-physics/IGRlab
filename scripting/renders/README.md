# Render Pipeline (Runner-Integrated)

This directory contains the manifest-driven rendering workflow for
`dg_igr_miniapp`. It consumes `summary.json` from `scripting/runner`, resolves
VTK datasets, renders with ParaView, optionally crops/trims images, optionally
generates TikZ wrappers, and can optionally compile a PDF via LaTeX.

## Scripts

- `render_pipeline.py`: orchestrates discover -> render -> postprocess -> tikz -> latex.
- `render_discover.py`: resolves `.pvd` datasets from runner summary + logs.
- `render_paraview.py`: unified `pvpython` renderer with configurable scale modes.
- `render_post.py`: optional crop/trim/copy stage for generated images.
- `render_tikz.py`: generate TikZ wrappers from frame manifests.

## Quick Start

1. Run sweeps with `scripting/runner/parametric_runner.py` and ensure VTK output is enabled.
2. Copy/edit `scripting/renders/example_render_config.json`.
3. Run:

```bash
python3 scripting/renders/render_pipeline.py \
  --config scripting/renders/example_render_config.json
```

Trim-only rerun (no ParaView render pass):

```bash
python3 scripting/renders/render_pipeline.py \
  --config scripting/renders/example_trim_only_config.json
```

Artifacts are written under `output_dir`:

- `render_plan.json`
- `render_manifest.json`
- `render_ranges.json`
- `frames_raw.json`
- `frames_post.json` (if postprocess enabled)
- `image_metadata.csv` (flat table: one row per image)
- `image_metadata.jsonl` (full per-image metadata records)
- `figures.tex` (if tikz enabled)
- `figures_wrapper.pdf` (if latex enabled)
- `pipeline_summary.json`

Rendered image folder structure:

- `images_raw/sm_<scale_mode>/<job_name>/<case>/<run>/<field>/<image>.png`
- `images_cropped/...` mirrors the same relative structure when postprocess is enabled.

To keep a stable output location and overwrite each run, set:

- `output_dir: "out/render/latest"`
- `clean_output_dir: true`

## Mixed-Case Summaries (Important)

If one `summary.json` contains unrelated problems, define separate `render_jobs`
so each problem gets independent:

- record selection (`include_cases`, `include_run_ids`),
- field list,
- time sampling,
- scaling mode/ranges.

This prevents cross-problem scale pollution.

For headless environments, set:

- `render.pvpython_args: ["--force-offscreen-rendering"]`

The pipeline also tolerates a known ParaView teardown issue (`GLXBadContext`)
only when render artifacts were generated successfully; otherwise it still
fails hard.

For clean central-image extraction with the same constants used in legacy
rendering, use high-resolution rendering and fixed crop geometry:

- render size: `4200x2300`
- postprocess crop: `3000x2000+600+0`
- optionally disable overlays:
  - `show_scalar_bar: false`
  - `show_title: false`

To embed scale bounds directly in PNG filenames for collaborator handoff:

- `filename_metadata_level: "legacy"` (old style: scale in filename)
- `filename_metadata_level: "rich"` (recommended: scale + compact metadata tags)

`rich` filename tags:

- `_m_<code>` scale-mode code:
  - `pf` = per_frame
  - `pd` = per_dataset
  - `pc` = per_case
  - `g` = global
  - `fx` = fixed
- `_a_<code>` field association:
  - `p` = points
  - `c` = cells
- optional `_tv_<time>` appears only when physical time differs from `ts`
  (typically index-mode runs).

Field rendering notes:

- Vector fields (e.g. `momentum`, `velocity`) are rendered as magnitude.
- Derived fields are available directly in `fields`:
  - `schlieren` = `|grad(pressure)|`
  - `vorticity` = `|curl(velocity)|`

For TikZ/PDF layout that avoids oversized, unbreakable pages:

- set `tikz.group_by: "job_field"` (or `"case_field"`),
- keep `tikz.max_cols` moderate (e.g. `3`),
- set `tikz.max_rows` to cap per-page matrix height (e.g. `2` or `3`).

## Scale Modes

Configured per render job via `scale_mode`:

- `per_frame`: each frame uses its own min/max.
- `per_dataset`: one range per `(run_id, field)`.
- `per_case`: one range per `(case_name, field)` within that job.
- `global`: one range per field within that job.
- `fixed`: explicit user ranges (`fixed_ranges`).

To render the same job under multiple scale policies in one pipeline run, use
`scale_modes` instead of `scale_mode`, e.g.:

- `"scale_modes": ["per_case", "global"]`

The pipeline will create a separate top-level scale directory for each mode
(`sm_per_case`, `sm_global`, etc.).

Performance note:

- each scale mode is a separate full render job, so runtime and image count
  scale roughly linearly with `len(scale_modes)`.
- prefer one mode per run when you are throughput-bound.
- duplicate entries in `scale_modes` are automatically deduplicated.

Renderer progress:

- `render_paraview.py` now prints per-record scan/render progress bars with
  elapsed time and ETA.
- `per_frame` mode skips the pre-scan range pass and uses per-frame ranges
  directly during rendering.
- `render_pipeline.py` now prints stage progress for render-job dispatch,
  metadata export, and LaTeX runs.
- `render_post.py` now prints per-frame postprocess progress with ETA.

Colormap controls:

- `color_preset`: default ParaView preset for all fields.
- `field_colormaps`: per-field override map.
  - Example: `"field_colormaps": {"schlieren": "Inferno (matplotlib)"}`.
- `invert_field_colormaps`: per-field colormap inversion list.
  - Example: `"invert_field_colormaps": ["schlieren"]` for dark-high schlieren.

## Time Selection Semantics

- `use_time_values: false` means `times` are interpreted as timestep indices (`0`, `1`, `2`, ...).
- `use_time_values: true` means `times` are interpreted as physical times (`0.1`, `0.15`, ...).
- Fractional `times` in index mode now fail fast with an explicit error to prevent silent clamping/collapse.

## Notes

- Discovery prefers `VTK dataset: ...` in run logs and falls back to inferred
  naming from CLI options (`-o`, `-rs`, `-rp`, `-is`, `-ig`, `-ca`).
- For rendering, runner configs must not use `-no-vtk`.
- Postprocess crop/trim requires ImageMagick (`magick` or `convert` in PATH).
- Cropping applies `+repage` after `-crop` for stable image geometry metadata.
- Optional `postprocess.crop_gravity` can force deterministic crop anchoring
  (`NorthWest`, `Center`, etc.).
- To remove black (or other solid) background and keep only colored plot area:
  - `postprocess.trim_background_color` (e.g. `"black"`, `"white"`, or `"auto"`),
  - `postprocess.trim_background_fuzz` (e.g. `"4%"`),
  - `postprocess.trim_background_border` (default `1`).
- `trim_background_fuzz` controls color tolerance around the chosen background.
  - `0%` means exact color-only trim (strict).
  - higher values trim near-background shades too.
- `trim_background_border` adds a temporary border of the background color before
  trim so edge-touching content still trims correctly.
- `postprocess.source_frames_manifest` lets you rerun only postprocess (trim/crop)
  from an existing frames manifest without rerendering.
- TikZ snippets require `\\usetikzlibrary{matrix}` in the LaTeX document.
- For cleaner figure layout, use `tikz.label_mode: "none"` or `"short"`.
