import argparse
import csv
from pathlib import Path
import cv2

def draw_overlay(img, lines, x=10, y=28, dy=24):
    for i, s in enumerate(lines):
        cv2.putText(img, s, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, s, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

def to_path(p: str) -> Path:
    return Path(p.replace("/", "\\")).resolve()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    index_path = Path(args.index)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] reading index: {index_path.resolve()}")
    with index_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("[ERROR] index CSV has 0 rows")

    needed = {"src_file","window_file","start_s","kick_in_window_s"}
    missing = needed - set(rows[0].keys())
    if missing:
        raise SystemExit(f"[ERROR] index missing columns: {sorted(missing)}. Have: {list(rows[0].keys())}")

    done = set()
    if args.resume and out_path.exists() and out_path.stat().st_size > 0:
        with out_path.open("r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add(r["src_file"])

    write_header = (not out_path.exists()) or out_path.stat().st_size == 0
    f_out = out_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        f_out,
        fieldnames=["src_file","window_file","start_s","kick_time_s","kick_time_in_window_s","kick_frame","fps"]
    )
    if write_header:
        writer.writeheader()
        f_out.flush()

    cv2.namedWindow("Review kick moment", cv2.WINDOW_NORMAL)

    print(f"[INFO] loaded {len(rows)} rows, resume_done={len(done)}")
    for i, r in enumerate(rows, start=1):
        if r["src_file"] in done:
            continue

        src_file = r["src_file"]
        window_file = to_path(r["window_file"])
        start_s = float(r["start_s"])
        guess_in_window = float(r["kick_in_window_s"])

        if not window_file.exists():
            print(f"[SKIP {i}] missing window_file: {window_file}")
            continue

        cap = cv2.VideoCapture(str(window_file))
        if not cap.isOpened():
            print(f"[SKIP {i}] cannot open video: {window_file}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        def clamp_frame(fi):
            if total_frames > 0:
                return max(0, min(total_frames - 1, fi))
            return max(0, fi)

        frame_idx = clamp_frame(int(round(guess_in_window * fps)))

        def goto_frame(fi):
            nonlocal frame_idx
            frame_idx = clamp_frame(fi)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, fr = cap.read()
            return ok, fr

        ok, frame = goto_frame(frame_idx)
        if not ok:
            cap.release()
            continue

        paused = True
        print(f"\n[REVIEW {i}] {window_file.name}")
        print("Controls: SPACE play/pause | A/D -/+1 frame | J/L -/+1s | K save | Q quit")

        while True:
            t_win = frame_idx / fps
            t_abs = start_s + t_win

            overlay = [
                f"window: {window_file.name}",
                f"src: {Path(src_file).name}",
                f"frame: {frame_idx}/{max(total_frames-1,0)}  fps={fps:.3f}",
                f"time_in_window={t_win:.3f}s  abs_time={t_abs:.3f}s",
                "SPACE play/pause | A/D -/+1 frame | J/L -/+1s | K save | Q quit",
            ]
            show = frame.copy()
            draw_overlay(show, overlay)
            cv2.imshow("Review kick moment", show)

            key = cv2.waitKey(0 if paused else max(1, int(1000 / fps))) & 0xFF

            if key in (ord("q"), 27):
                cap.release()
                f_out.close()
                cv2.destroyAllWindows()
                return

            if key == ord(" "):
                paused = not paused
                if not paused:
                    continue

            if not paused:
                ok, frame = cap.read()
                if not ok:
                    paused = True
                    ok, frame = goto_frame(frame_idx)
                else:
                    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                continue

            if key == ord("a"):
                ok, frame = goto_frame(frame_idx - 1)
            elif key == ord("d"):
                ok, frame = goto_frame(frame_idx + 1)
            elif key == ord("j"):
                ok, frame = goto_frame(frame_idx - int(round(fps)))
            elif key == ord("l"):
                ok, frame = goto_frame(frame_idx + int(round(fps)))
            elif key == ord("k"):
                kick_time_in_window_s = frame_idx / fps
                kick_time_s = start_s + kick_time_in_window_s
                kick_frame = int(round(kick_time_s * fps))

                writer.writerow({
                    "src_file": src_file,
                    "window_file": str(window_file).replace("\\","/"),
                    "start_s": f"{start_s:.3f}",
                    "kick_time_s": f"{kick_time_s:.6f}",
                    "kick_time_in_window_s": f"{kick_time_in_window_s:.6f}",
                    "kick_frame": str(kick_frame),
                    "fps": f"{fps:.6f}",
                })
                f_out.flush()
                print(f"[OK] saved abs={kick_time_s:.3f}s (win={kick_time_in_window_s:.3f}s frame={frame_idx})")
                break

        cap.release()

    f_out.close()
    cv2.destroyAllWindows()
    print(f"\nSaved corrected kick times to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
