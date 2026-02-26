import argparse
import subprocess
from pathlib import Path

def crop_one(src: Path, dst: Path, crop: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["magick", str(src), "-crop", crop, "-trim", "+repage", str(dst)]
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inroot", required=True)
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--crop", required=True, help='e.g. 3000x2000+600+0')
    ap.add_argument("--glob", default="**/*.png")
    args = ap.parse_args()

    inroot = Path(args.inroot).expanduser().resolve()
    outroot = Path(args.outroot).expanduser().resolve()

    files = sorted(inroot.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.glob} under {inroot}")

    for src in files:
        rel = src.relative_to(inroot)
        dst = outroot / rel
        crop_one(src, dst, args.crop)
        print(dst)

if __name__ == "__main__":
    main()
