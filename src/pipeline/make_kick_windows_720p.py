import os, csv, shutil, subprocess
from pathlib import Path

# Defaults (override via env vars if you want)
KICK_CSV = Path(os.getenv("KICK_CSV", "data/meta/kick_times.csv"))
IN_DIR   = Path(os.getenv("IN_DIR",   "data/clips/penalties_720p"))
OUT_DIR  = Path(os.getenv("OUT_DIR",  "data/clips/kick_windows_720p_v2"))
OUT_CSV  = Path(os.getenv("OUT_CSV",  "data/meta/kick_windows_720p.csv"))

PRE  = float(os.getenv("KICK_PRE",  "1.5"))
POST = float(os.getenv("KICK_POST", "2.5"))

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

ffmpeg = os.getenv("FFMPEG_EXE") or shutil.which("ffmpeg")
if not ffmpeg:
    raise SystemExit("ffmpeg not found. Set $env:FFMPEG_EXE to full path of ffmpeg.exe")

def run(cmd):
    subprocess.run(cmd, check=True)

def pick(row, keys):
    for k in keys:
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
    return ""

def resolve_src(row):
    # If you ever have absolute paths in CSV
    cp = pick(row, ["src_file", "clip_path", "path", "full_path"])
    if cp:
        p = Path(cp)
        return p

    # Typical case: CSV stores filename in column "clip_name" or (your case!) "c"
    name = pick(row, ["clip_name", "c", "file", "filename", "name"])
    if name:
        return IN_DIR / name

    return None

def get_kick_time(row):
    v = pick(row, ["kick_time_s", "kick_s", "kick_time", "t"])
    return float(v) if v else None

def get_kick_frame(row):
    v = pick(row, ["kick_frame", "frame"])
    if not v:
        return ""
    return str(int(float(v)))

with KICK_CSV.open(newline="", encoding="utf-8") as f, OUT_CSV.open("w", newline="", encoding="utf-8") as g:
    r = csv.DictReader(f)
    w = csv.DictWriter(g, fieldnames=["src_file","window_file","start_s","dur_s","kick_in_window_s","kick_frame"])
    w.writeheader()

    for row in r:
        src = resolve_src(row)
        kick_t = get_kick_time(row)
        kick_fr = get_kick_frame(row)

        if src is None or kick_t is None:
            print("[SKIP bad row]", row)
            continue

        if src.is_dir():
            print("[SKIP src is a folder, not a file]", src)
            continue

        if not src.exists():
            print("[SKIP missing]", src)
            continue

        start = max(0.0, kick_t - PRE)
        dur   = PRE + POST
        kick_in_window = kick_t - start

        dst = OUT_DIR / (src.stem + "_KICK.mp4")

        # NOTE: -ss after -i is more accurate for your “kick moment” precision
        cmd = [
    ffmpeg, "-y",
    "-hide_banner", "-loglevel", "error",
    "-i", str(src),
    "-ss", f"{start:.3f}",
    "-t",  f"{dur:.3f}",
    "-an",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "23",
    str(dst)
]

        run(cmd)

        w.writerow({
            "src_file": str(src).replace("\\","/"),
            "window_file": str(dst).replace("\\","/"),
            "start_s": f"{start:.3f}",
            "dur_s": f"{dur:.3f}",
            "kick_in_window_s": f"{kick_in_window:.3f}",
            "kick_frame": kick_fr,
        })
        print("[OK]", dst.name)

print("Saved:", OUT_CSV)
