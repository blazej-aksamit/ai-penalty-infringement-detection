"""
Creates a demo video for thesis:
  - plays the clip normally up to kick frame
  - freezes on kick frame: first shows line-violation overlay, then encroachment overlay
  - continues playing after kick
  - exports as MP4

Usage:
    python scripts/tools/make_demo_video.py \
        --clip data/clips/kick_windows_720p_v2/...KICK.mp4 \
        --overlay-line   runs/evaluation/.../hybrid/final_overlay.jpg \
        --overlay-enc    runs/encroachment_49/.../encroachment_overlay.jpg \
        --kick-frame 36 \
        --out runs/demo/ludogorets_demo.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def fade(img_a, img_b, t):
    """Blend img_a -> img_b linearly, t in [0,1]."""
    return cv2.addWeighted(img_b, t, img_a, 1.0 - t, 0)


def make_demo_video(
    clip_path: Path,
    overlay_line: Path,
    overlay_enc: Path | None,
    kick_frame: int,
    out_path: Path,
    freeze_line_s: float = 2.5,
    freeze_enc_s: float  = 2.5,
    fade_frames: int     = 10,
    play_before: int     = 35,
    play_after: int      = 20,
    output_fps: float    = 25.0,
):
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {clip_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ov_line = cv2.imread(str(overlay_line))
    if ov_line is None:
        raise RuntimeError(f"Cannot read overlay: {overlay_line}")
    ov_line = cv2.resize(ov_line, (w, h))

    ov_enc = None
    if overlay_enc and overlay_enc.exists():
        ov_enc = cv2.imread(str(overlay_enc))
        if ov_enc is not None:
            ov_enc = cv2.resize(ov_enc, (w, h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, output_fps, (w, h))

    freeze_line_f = int(round(freeze_line_s * output_fps))
    freeze_enc_f  = int(round(freeze_enc_s  * output_fps)) if ov_enc is not None else 0

    start_frame = max(0, kick_frame - play_before)
    end_frame   = min(total - 1, kick_frame + play_after)

    print(f"Clip      : {clip_path.name}")
    print(f"Resolution: {w}x{h}  src_fps={src_fps:.1f}")
    print(f"Kick frame: {kick_frame}  playing [{start_frame}, {end_frame}]")
    print(f"Freeze    : line={freeze_line_s}s  enc={freeze_enc_s}s")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for fi in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if fi == kick_frame:
            # Phase 1: fade in line overlay, hold
            for k in range(freeze_line_f):
                alpha = min(1.0, k / fade_frames)
                writer.write(fade(frame, ov_line, alpha))

            # Phase 2: crossfade to encroachment overlay, hold
            if ov_enc is not None:
                for k in range(freeze_enc_f):
                    if k < fade_frames:
                        alpha = k / fade_frames
                        writer.write(fade(ov_line, ov_enc, alpha))
                    else:
                        writer.write(ov_enc)
        else:
            writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip",          required=True)
    ap.add_argument("--overlay-line",  required=True)
    ap.add_argument("--overlay-enc",   default=None)
    ap.add_argument("--kick-frame",    type=int, required=True)
    ap.add_argument("--out",           required=True)
    ap.add_argument("--freeze-line",   type=float, default=2.5)
    ap.add_argument("--freeze-enc",    type=float, default=2.5)
    ap.add_argument("--play-before",   type=int,   default=35)
    ap.add_argument("--play-after",    type=int,   default=20)
    ap.add_argument("--fps",           type=float, default=25.0)
    args = ap.parse_args()

    make_demo_video(
        clip_path    = Path(args.clip),
        overlay_line = Path(args.overlay_line),
        overlay_enc  = Path(args.overlay_enc) if args.overlay_enc else None,
        kick_frame   = args.kick_frame,
        out_path     = Path(args.out),
        freeze_line_s= args.freeze_line,
        freeze_enc_s = args.freeze_enc,
        play_before  = args.play_before,
        play_after   = args.play_after,
        output_fps   = args.fps,
    )


if __name__ == "__main__":
    main()
