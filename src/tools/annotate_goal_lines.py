"""
Goal-line manual annotation tool.

Usage:
    python scripts/tools/annotate_goal_lines.py \
        --batch-dirs runs/evaluation/batch_final_orig/test runs/evaluation/batch_final_ext/test_ext \
        --out data/meta/manual_goal_lines.json

Controls:
    Left-click    - place point 1 or point 2
    R             - reset current clip (redo)
    ENTER / SPACE - confirm line and advance to next clip
    ESC           - quit and save progress
    S             - skip current clip (mark as no_line)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

WINDOW = "Goal-line annotator  |  Click 2 pts -> ENTER confirm | R redo | S skip | ESC quit"


def find_frame(clip_dir: Path):
    frames_dir = clip_dir / "frames"
    if frames_dir.exists():
        imgs = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
        if imgs:
            return imgs[0]
    return None


def collect_clips(batch_dirs):
    clips = []
    for bd in batch_dirs:
        bd = Path(bd)
        for clip_dir in sorted(bd.iterdir()):
            if not clip_dir.is_dir():
                continue
            frame = find_frame(clip_dir)
            if frame is None:
                print(f"  [SKIP - no frame] {clip_dir.name}")
                continue
            clips.append({"name": clip_dir.name, "frame": frame, "clip_dir": clip_dir})
    return clips


class Annotator:
    def __init__(self, img_path: Path):
        self.img_orig = cv2.imread(str(img_path))
        if self.img_orig is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        self.h, self.w = self.img_orig.shape[:2]
        self.pts = []
        self.confirmed = False
        self.skipped = False

    def _draw(self):
        vis = self.img_orig.copy()

        # draw progress dots / line
        for i, (px, py) in enumerate(self.pts):
            cv2.circle(vis, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(vis, str(i + 1), (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if len(self.pts) == 2:
            cv2.line(vis, self.pts[0], self.pts[1], (0, 255, 255), 2)
            # extend across full image width
            x1, y1 = self.pts[0]
            x2, y2 = self.pts[1]
            if abs(x2 - x1) > 1:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                lx1, ly1 = 0, int(round(b))
                lx2, ly2 = self.w - 1, int(round(m * (self.w - 1) + b))
                cv2.line(vis, (lx1, ly1), (lx2, ly2), (255, 200, 0), 1)

        status = f"Points: {len(self.pts)}/2"
        if len(self.pts) == 2:
            status += "  ->  ENTER to confirm | R to redo"
        else:
            status += "  ->  click to place point"
        cv2.putText(vis, status, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        return vis

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pts) < 2:
                self.pts.append((x, y))
            cv2.imshow(WINDOW, self._draw())

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, min(1280, self.w), min(720, self.h))
        cv2.setMouseCallback(WINDOW, self.mouse_cb)
        cv2.imshow(WINDOW, self._draw())

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('r') or key == ord('R'):
                self.pts = []
                cv2.imshow(WINDOW, self._draw())
            elif key in (13, 32):  # ENTER or SPACE
                if len(self.pts) == 2:
                    self.confirmed = True
                    break
            elif key == ord('s') or key == ord('S'):
                self.skipped = True
                break
            elif key == 27:  # ESC
                break

        return self.confirmed, self.skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dirs", nargs="+", required=True)
    ap.add_argument("--out", default="data/meta/manual_goal_lines.json")
    ap.add_argument("--resume", action="store_true",
                    help="Skip clips already present in output file")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing annotations if resuming
    annotations = {}
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            annotations = json.load(f)
        print(f"Resuming — {len(annotations)} clips already annotated.")

    clips = collect_clips(args.batch_dirs)
    print(f"Found {len(clips)} clips to annotate.")

    quit_requested = False
    for i, clip in enumerate(clips):
        name = clip["name"]

        if args.resume and name in annotations:
            print(f"  [{i+1}/{len(clips)}] SKIP (already done): {name}")
            continue

        print(f"\n[{i+1}/{len(clips)}] {name}")
        print(f"  Frame: {clip['frame']}")

        ann = Annotator(clip["frame"])
        confirmed, skipped = ann.run()

        if not confirmed and not skipped:
            # ESC pressed — save and quit
            quit_requested = True
            print("  ESC — saving and quitting.")
            break

        if skipped:
            annotations[name] = None
            print(f"  Skipped (no line).")
        else:
            x1, y1 = ann.pts[0]
            x2, y2 = ann.pts[1]
            annotations[name] = [x1, y1, x2, y2]
            print(f"  Line: ({x1},{y1}) -> ({x2},{y2})")

        # Save after every clip so no progress is lost
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2)

    cv2.destroyAllWindows()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    done = sum(1 for v in annotations.values() if v is not None)
    skipped_count = sum(1 for v in annotations.values() if v is None)
    print(f"\nDone. Annotated: {done}  Skipped: {skipped_count}  Saved to: {out_path}")

    if quit_requested:
        remaining = len(clips) - len(annotations)
        print(f"  {remaining} clips remaining — re-run with --resume to continue.")


if __name__ == "__main__":
    main()
