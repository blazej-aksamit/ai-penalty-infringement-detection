import csv
import os
from pathlib import Path

import cv2

# ---------------------------------
# Defaults (can override with env vars)
WINDOWS_DIR = Path(os.getenv("WINDOWS_DIR", "data/clips/kick_windows_720p_v2"))
OUT_CSV = Path(os.getenv("OUT_CSV", "data/meta/keeper_violation_labels.csv"))
# ---------------------------------


def list_clips():
    return sorted(WINDOWS_DIR.glob("*.mp4"))


def load_done():
    done = {}
    if OUT_CSV.exists() and OUT_CSV.stat().st_size > 0:
        with OUT_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clip_name = str(row.get("clip_name", "")).strip()
                if clip_name:
                    done[clip_name] = row
    return done


def append_row(row):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not OUT_CSV.exists()) or OUT_CSV.stat().st_size == 0

    with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "clip_name",
            "window_file",
            "violation",
            "uncertain",
            "fps",
            "total_frames",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    clips = list_clips()
    if not clips:
        raise SystemExit(f"No clips found in {WINDOWS_DIR}")

    done = load_done()

    print(f"Clips found: {len(clips)} | Already labeled: {len(done)}")
    print("Controls:")
    print("  [space] play/pause")
    print("  a / d   step -/+1 frame")
    print("  s / w   step -/+10 frames")
    print("  r       restart clip")
    print("  0       label NO violation")
    print("  1       label VIOLATION")
    print("  u       label UNCERTAIN")
    print("  n       skip clip (no save)")
    print("  q       quit")

    cv2.namedWindow("Label keeper violation", cv2.WINDOW_NORMAL)

    for clip in clips:
        if clip.name in done:
            continue

        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            print(f"[SKIP] cannot open {clip.name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        pos = 0
        paused = True

        def read_at(frame_idx):
            nonlocal pos
            if total > 0:
                frame_idx = max(0, min(frame_idx, total - 1))
            else:
                frame_idx = max(0, frame_idx)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if ok:
                pos = frame_idx
            return ok, frame

        ok, frame = read_at(0)
        if not ok:
            print(f"[SKIP] cannot read frames {clip.name}")
            cap.release()
            continue

        while True:
            show = frame.copy()
            t = pos / fps if fps else 0.0

            overlay_lines = [
                f"{clip.name}",
                f"frame {pos}/{max(total - 1, 0)} | t={t:.3f}s | fps={fps:.2f}",
                "[space]=play/pause  a/d=-/+1  s/w=-/+10  r=restart",
                "0=no violation   1=violation   u=uncertain   n=skip   q=quit",
            ]

            y = 30
            for line in overlay_lines:
                cv2.putText(show, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(show, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                y += 28

            cv2.imshow("Label keeper violation", show)

            if not paused:
                ok, frame = read_at(pos + 1)
                if not ok:
                    paused = True

            key = cv2.waitKey(20 if not paused else 0) & 0xFF

            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("Stopped. Saved:", OUT_CSV)
                return

            if key == ord(" "):
                paused = not paused

            elif key == ord("a"):
                paused = True
                ok, frame = read_at(pos - 1)

            elif key == ord("d"):
                paused = True
                ok, frame = read_at(pos + 1)

            elif key == ord("s"):
                paused = True
                ok, frame = read_at(pos - 10)

            elif key == ord("w"):
                paused = True
                ok, frame = read_at(pos + 10)

            elif key == ord("r"):
                paused = True
                ok, frame = read_at(0)

            elif key == ord("n"):
                print(f"[SKIP] {clip.name}")
                break

            elif key == ord("0"):
                append_row(
                    {
                        "clip_name": clip.name,
                        "window_file": str(clip).replace("\\", "/"),
                        "violation": "0",
                        "uncertain": "0",
                        "fps": f"{fps:.6f}",
                        "total_frames": str(total),
                        "notes": "",
                    }
                )
                print(f"[OK] {clip.name} -> NO violation")
                break

            elif key == ord("1"):
                append_row(
                    {
                        "clip_name": clip.name,
                        "window_file": str(clip).replace("\\", "/"),
                        "violation": "1",
                        "uncertain": "0",
                        "fps": f"{fps:.6f}",
                        "total_frames": str(total),
                        "notes": "",
                    }
                )
                print(f"[OK] {clip.name} -> VIOLATION")
                break

            elif key == ord("u"):
                append_row(
                    {
                        "clip_name": clip.name,
                        "window_file": str(clip).replace("\\", "/"),
                        "violation": "",
                        "uncertain": "1",
                        "fps": f"{fps:.6f}",
                        "total_frames": str(total),
                        "notes": "",
                    }
                )
                print(f"[OK] {clip.name} -> UNCERTAIN")
                break

        cap.release()

    cv2.destroyAllWindows()
    print("Done. Saved:", OUT_CSV)


if __name__ == "__main__":
    main()