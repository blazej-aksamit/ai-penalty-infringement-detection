"""
Ball Motion-Based Kick Frame Detector

Detects the kick moment in penalty videos by tracking ball motion changes.
Uses YOLO for ball detection and velocity analysis to find the kick frame.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics import YOLO


def load_yolo_model(model_path: str):
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ultralytics is not installed in the active Python environment. "
            "Install it or run this script from the environment used for YOLO training."
        ) from exc

    return YOLO(model_path)


def _select_ball_detection(boxes, min_confidence: float, prev_xy: Optional[Tuple[float, float]], max_tracking_jump_px: float):
    if len(boxes) == 0:
        return None

    candidates = []
    confidences = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()

    for idx, conf in enumerate(confidences):
        conf = float(conf)
        if conf < min_confidence:
            continue

        x1, y1, x2, y2 = xyxy[idx]
        x_center = float((x1 + x2) / 2.0)
        y_center = float((y1 + y2) / 2.0)

        if prev_xy is None:
            score = conf
        else:
            dist = float(np.hypot(x_center - prev_xy[0], y_center - prev_xy[1]))
            if dist > max_tracking_jump_px:
                continue
            score = conf - 0.0025 * dist

        candidates.append((score, x_center, y_center, conf))

    if not candidates:
        return None

    _, x_center, y_center, conf = max(candidates, key=lambda item: item[0])
    return x_center, y_center, conf


def _smooth_series(values: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if len(values) <= 1 or kernel_size <= 1:
        return values.copy()

    kernel = np.ones(kernel_size, dtype=float) / float(kernel_size)
    pad = kernel_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(values)]


def _estimate_kick_frame_from_velocities(
    velocities: List[Tuple[int, float]],
    onset_factor: float,
    min_sustained_velocity: float,
    fallback_to_peak: bool,
) -> Dict[str, Any]:
    if len(velocities) < 2:
        return {
            "kick_frame": None,
            "confidence": 0.0,
            "method": "failed",
            "reason": "insufficient_velocity_samples",
            "threshold": None,
            "peak_velocity": None,
            "baseline_velocity": None,
            "smoothed_velocities": [],
        }

    velocity_frames = [frame_idx for frame_idx, _ in velocities]
    velocity_values = np.array([value for _, value in velocities], dtype=float)
    smoothed = _smooth_series(velocity_values, kernel_size=3)

    baseline_window = max(2, min(5, len(smoothed)))
    baseline_velocity = float(np.median(smoothed[:baseline_window]))
    velocity_std = float(np.std(smoothed))
    peak_velocity = float(np.max(smoothed))

    threshold = max(
        float(min_sustained_velocity),
        baseline_velocity * float(onset_factor),
        baseline_velocity + 2.0 * velocity_std,
    )

    onset_idx = None
    for idx, value in enumerate(smoothed):
        if float(value) < threshold:
            continue

        lookahead = smoothed[idx : min(len(smoothed), idx + 3)]
        if len(lookahead) >= 2 and float(np.mean(lookahead)) >= 0.8 * threshold:
            onset_idx = idx
            break

    if onset_idx is not None:
        kick_frame = max(0, int(velocity_frames[onset_idx] - 1))
        method = "motion_onset"
    elif fallback_to_peak:
        peak_idx = int(np.argmax(smoothed))
        kick_frame = max(0, int(velocity_frames[peak_idx] - 1))
        method = "velocity_peak_fallback"
    else:
        kick_frame = None
        method = "failed"

    if kick_frame is None:
        confidence = 0.0
        reason = "no_reliable_motion_onset"
    else:
        denom = velocity_std if velocity_std > 1e-6 else max(1.0, baseline_velocity)
        prominence = (peak_velocity - baseline_velocity) / denom
        confidence = float(np.clip(prominence / 3.0, 0.0, 1.0))
        reason = "ok"

    return {
        "kick_frame": kick_frame,
        "confidence": confidence,
        "method": method,
        "reason": reason,
        "threshold": float(threshold),
        "peak_velocity": peak_velocity,
        "baseline_velocity": baseline_velocity,
        "smoothed_velocities": [
            {"frame_idx": int(frame_idx), "velocity": float(value), "smoothed_velocity": float(smoothed[idx])}
            for idx, (frame_idx, value) in enumerate(velocities)
        ],
    }


def detect_kick_frame_ball_motion_details(
    video_path: str,
    yolo_model,
    window_start: int = 0,
    window_end: Optional[int] = None,
    ball_class_id: int = 1,
    min_confidence: float = 0.3,
    velocity_prominence_threshold: float = 2.5,
    max_tracking_jump_px: float = 180.0,
    min_sustained_velocity: float = 2.0,
    fallback_to_peak: bool = True,
) -> Dict[str, Any]:
    """
    Detect kick frame by analyzing ball motion changes and returning diagnostics.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "kick_frame": None,
            "confidence": 0.0,
            "reason": "video_open_failed",
            "method": "failed",
            "ball_trajectory": [],
            "velocities": [],
            "smoothed_velocities": [],
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if window_end is None:
        window_end = total_frames
    window_start = max(0, int(window_start))
    window_end = min(total_frames, int(window_end))

    print(f"Analyzing video: {Path(video_path).name}")
    print(f"  Total frames: {total_frames}, FPS: {fps:.2f}")
    print(f"  Analysis window: {window_start} - {window_end}")

    ball_trajectory = []
    frame_idx = 0
    prev_xy = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= window_end:
            break

        if frame_idx >= window_start:
            results = yolo_model(frame, classes=[ball_class_id], verbose=False)
            selected = _select_ball_detection(
                boxes=results[0].boxes,
                min_confidence=min_confidence,
                prev_xy=prev_xy,
                max_tracking_jump_px=max_tracking_jump_px,
            )

            if selected is not None:
                x_center, y_center, conf = selected
                ball_trajectory.append((frame_idx, x_center, y_center, conf))
                prev_xy = (x_center, y_center)

        frame_idx += 1

    cap.release()

    print(f"  Ball detected in {len(ball_trajectory)}/{max(1, window_end - window_start)} frames")

    if len(ball_trajectory) < 3:
        print("  Error: Insufficient ball detections for velocity analysis")
        return {
            "kick_frame": None,
            "confidence": 0.0,
            "reason": "insufficient_ball_detections",
            "method": "failed",
            "fps": fps,
            "total_frames": total_frames,
            "window_start": window_start,
            "window_end": window_end,
            "ball_trajectory": ball_trajectory,
            "velocities": [],
            "smoothed_velocities": [],
        }

    velocities = []
    for i in range(1, len(ball_trajectory)):
        prev_frame, prev_x, prev_y, _ = ball_trajectory[i - 1]
        curr_frame, curr_x, curr_y, _ = ball_trajectory[i]

        dx = curr_x - prev_x
        dy = curr_y - prev_y
        dt = curr_frame - prev_frame

        if dt > 0:
            velocities.append((curr_frame, float(np.hypot(dx, dy) / dt)))

    if len(velocities) < 2:
        print("  Error: Insufficient velocity samples")
        return {
            "kick_frame": None,
            "confidence": 0.0,
            "reason": "insufficient_velocity_samples",
            "method": "failed",
            "fps": fps,
            "total_frames": total_frames,
            "window_start": window_start,
            "window_end": window_end,
            "ball_trajectory": ball_trajectory,
            "velocities": [{"frame_idx": int(f), "velocity": float(v)} for f, v in velocities],
            "smoothed_velocities": [],
        }

    estimate = _estimate_kick_frame_from_velocities(
        velocities=velocities,
        onset_factor=velocity_prominence_threshold,
        min_sustained_velocity=min_sustained_velocity,
        fallback_to_peak=fallback_to_peak,
    )

    kick_frame = estimate["kick_frame"]
    confidence = estimate["confidence"]

    if kick_frame is not None:
        print(f"  Detected kick frame: {kick_frame}")
        print(f"  Method: {estimate['method']}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Baseline velocity: {estimate['baseline_velocity']:.2f} px/frame")
        print(f"  Peak velocity: {estimate['peak_velocity']:.2f} px/frame")
        print(f"  Threshold: {estimate['threshold']:.2f} px/frame")

    return {
        "kick_frame": kick_frame,
        "confidence": confidence,
        "reason": estimate["reason"],
        "method": estimate["method"],
        "fps": fps,
        "total_frames": total_frames,
        "window_start": window_start,
        "window_end": window_end,
        "threshold": estimate["threshold"],
        "peak_velocity": estimate["peak_velocity"],
        "baseline_velocity": estimate["baseline_velocity"],
        "ball_trajectory": ball_trajectory,
        "velocities": [{"frame_idx": int(f), "velocity": float(v)} for f, v in velocities],
        "smoothed_velocities": estimate["smoothed_velocities"],
        "max_tracking_jump_px": float(max_tracking_jump_px),
        "min_sustained_velocity": float(min_sustained_velocity),
        "velocity_prominence_threshold": float(velocity_prominence_threshold),
    }


def detect_kick_frame_ball_motion(
    video_path: str,
    yolo_model,
    window_start: int = 0,
    window_end: Optional[int] = None,
    ball_class_id: int = 1,
    min_confidence: float = 0.3,
    velocity_prominence_threshold: float = 2.5,
    max_tracking_jump_px: float = 180.0,
    min_sustained_velocity: float = 2.0,
    fallback_to_peak: bool = True,
) -> Tuple[Optional[int], float, List[Tuple[int, float, float, float]]]:
    """
    Backward-compatible wrapper that returns the legacy tuple.
    """

    details = detect_kick_frame_ball_motion_details(
        video_path=video_path,
        yolo_model=yolo_model,
        window_start=window_start,
        window_end=window_end,
        ball_class_id=ball_class_id,
        min_confidence=min_confidence,
        velocity_prominence_threshold=velocity_prominence_threshold,
        max_tracking_jump_px=max_tracking_jump_px,
        min_sustained_velocity=min_sustained_velocity,
        fallback_to_peak=fallback_to_peak,
    )
    return details["kick_frame"], details["confidence"], details["ball_trajectory"]


def visualize_kick_detection(
    video_path: str,
    kick_frame: int,
    ball_trajectory: List[Tuple[int, float, float, float]],
    output_path: str,
    context_frames: int = 30,
) -> None:
    """
    Create visualization overlay showing detected kick frame and ball trajectory.
    """

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    trajectory_dict = {frame: (x, y, conf) for frame, x, y, conf in ball_trajectory}

    start_frame = max(0, kick_frame - context_frames)
    end_frame = kick_frame + context_frames

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    while cap.isOpened() and frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in trajectory_dict:
            x, y, conf = trajectory_dict[frame_idx]
            cv2.circle(frame, (int(x), int(y)), 12, (0, 255, 0), 3)
            cv2.putText(
                frame,
                f"{conf:.2f}",
                (int(x) + 15, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if frame_idx == kick_frame:
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 15)
            cv2.putText(
                frame,
                "KICK FRAME DETECTED",
                (50, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 0, 255),
                4,
            )

        frames_from_kick = frame_idx - kick_frame
        sign = "+" if frames_from_kick > 0 else ""
        cv2.putText(
            frame,
            f"Frame: {frame_idx} ({sign}{frames_from_kick})",
            (50, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print(f"Visualization saved to: {output_path}")


def batch_detect_kicks(
    video_dir: str,
    yolo_model_path: str,
    output_csv: str,
    video_extension: str = "*.mp4",
) -> Dict[str, Any]:
    """
    Batch process multiple penalty videos to detect kick frames.
    """

    import pandas as pd
    from tqdm import tqdm

    video_dir = Path(video_dir)
    yolo_model = load_yolo_model(yolo_model_path)

    video_paths = sorted(video_dir.glob(video_extension))
    print(f"Found {len(video_paths)} videos in {video_dir}")

    results = []
    for video_path in tqdm(video_paths, desc="Processing videos"):
        details = detect_kick_frame_ball_motion_details(str(video_path), yolo_model)
        results.append(
            {
                "video_name": video_path.name,
                "detected_kick_frame": details["kick_frame"],
                "kick_detection_confidence": details["confidence"],
                "kick_detection_method": details["method"],
                "kick_detection_reason": details["reason"],
                "num_ball_detections": len(details["ball_trajectory"]),
                "detection_success": details["kick_frame"] is not None,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print(f"\nResults saved to: {output_csv}")
    print(f"Success rate: {df['detection_success'].sum()}/{len(df)} ({df['detection_success'].mean():.1%})")

    return {"rows": results}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect kick frames in penalty videos")
    parser.add_argument("--video", type=str, help="Path to single video file")
    parser.add_argument("--video-dir", type=str, help="Directory with multiple videos")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model weights")
    parser.add_argument("--output", type=str, default="kick_detections.csv", help="Output CSV path")
    parser.add_argument("--visualize", action="store_true", help="Create visualization videos")

    args = parser.parse_args()

    if args.video:
        model = load_yolo_model(args.model)
        details = detect_kick_frame_ball_motion_details(args.video, model)

        if args.visualize and details["kick_frame"] is not None:
            output_vis = Path(args.video).parent / f"{Path(args.video).stem}_kick_detection.mp4"
            visualize_kick_detection(args.video, details["kick_frame"], details["ball_trajectory"], str(output_vis))

    elif args.video_dir:
        batch_detect_kicks(args.video_dir, args.model, args.output)

    else:
        print("Error: Specify either --video or --video-dir")
