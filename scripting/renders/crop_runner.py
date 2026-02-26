import subprocess
import shlex
from pathlib import Path

INROOT = Path("renders_scaled")
OUTROOT = Path("renders_scaled_cropped")

CROP = "3000x2000+600+0"

GLOB = "**/*.png"
# Examples:
# GLOB = "**/density_*.png"
# GLOB = "**/*ts_55*.png"

PYTHON = "python"      # or "python3"
SCRIPT = "crop_renders.py"

cmd = [
    PYTHON, SCRIPT,
    "--inroot", str(INROOT),
    "--outroot", str(OUTROOT),
    "--crop", CROP,
    "--glob", GLOB,
]

print("Running:\n", " ".join(shlex.quote(c) for c in cmd))
subprocess.check_call(cmd)
