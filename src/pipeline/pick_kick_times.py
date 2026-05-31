from pathlib import Path
import csv
import os
import cv2

CLIPS_DIR = Path(os.getenv("CLIPS_DIR", "data/clips/penalties_720p"))
OUT_CSV = Path(os.getenv("OUT_CSV", "data/meta/kick_times.csv"))
CLIP_GLOB = os.getenv("CLIP_GLOB", "*.mp4")

def list_clips():
    return sorted(CLIPS_DIR.glob(CLIP_GLOB))

def load_done():
    done = {}
    if OUT_CSV.exists():
        with OUT_CSV.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                done[row["clip_name"]] = row
    return done

def append_row(row):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = OUT_CSV.exists()
    with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip_name", "kick_time_s", "kick_frame"])
        if not exists:
            w.writeheader()
        w.writerow(row)

def main():
    clips = list_clips()
    if not clips:
        raise SystemExit(f"No clips found in {CLIPS_DIR} matching {CLIP_GLOB}")

    done = load_done()
    print(f"Clips dir: {CLIPS_DIR}")
    print(f"Clip filter: {CLIP_GLOB}")
    print(f"Output CSV: {OUT_CSV}")
    print(f"Clips found: {len(clips)} | Already labeled: {len(done)}")
    print("Controls: [space]=pause/play  a/d=step -/+1 frame  s/w=step -/+10 frames")
    print("          k=mark kick frame   n=skip clip          q=quit")

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
            frame_idx = max(0, min(frame_idx, max(0, total - 1)))
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
            t = pos / fps
            text = f"{clip.name} | frame {pos}/{max(total-1,0)} | t={t:.3f}s | fps={fps:.2f}"
            cv2.putText(show, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Pick kick frame", show)

            if not paused:
                ok, frame = read_at(pos + 1)
                if not ok:
                    paused = True

            key = cv2.waitKey(20 if not paused else 0) & 0xFF

            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

            if key == ord(' '):
                paused = not paused

            elif key == ord('a'):
                paused = True
                ok, frame = read_at(pos - 1)

            elif key == ord('d'):
                paused = True
                ok, frame = read_at(pos + 1)

            elif key == ord('s'):
                paused = True
                ok, frame = read_at(pos - 10)

            elif key == ord('w'):
                paused = True
                ok, frame = read_at(pos + 10)

            elif key == ord('n'):
                print(f"[SKIP] {clip.name}")
                break

            elif key == ord('k'):
                t = pos / fps
                append_row({"clip_name": clip.name, "kick_time_s": f"{t:.6f}", "kick_frame": str(pos)})
                print(f"[OK] {clip.name} kick at t={t:.3f}s (frame {pos})")
                break

        cap.release()

    cv2.destroyAllWindows()
    print("Done. Saved:", OUT_CSV)

if __name__ == "__main__":
    main()
