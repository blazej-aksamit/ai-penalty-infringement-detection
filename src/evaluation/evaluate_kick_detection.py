from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


REPO_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(REPO_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_SCRIPTS_DIR))

from kick_detection.ball_motion_detector import (  # noqa: E402
    detect_kick_frame_ball_motion_details,
    load_yolo_model,
)


def safe_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_video_fps(video_path: Path) -> float:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is not installed in the active Python environment. "
            "Run this script from the environment used for video-based ML work."
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS reported for video: {video_path}")
    return fps


def build_markdown_report(
    summary: Dict[str, object],
    reason_counts: Dict[str, int],
    method_counts: Dict[str, int],
) -> str:
    lines: List[str] = [
        "# Kick Detection Evaluation",
        "",
        f"- clips attempted: `{summary['clips_attempted']}`",
        f"- clips found on disk: `{summary['clips_found']}`",
        f"- successful detections: `{summary['success_count']}`",
        f"- detection success rate: `{summary['success_rate']:.3f}`",
        f"- exact frame accuracy: `{summary['exact_accuracy']:.3f}`",
        f"- within +/-1 frames: `{summary['within_1']:.3f}`",
        f"- within +/-2 frames: `{summary['within_2']:.3f}`",
        f"- within +/-3 frames: `{summary['within_3']:.3f}`",
        f"- within +/-5 frames: `{summary['within_5']:.3f}`",
        "",
    ]

    if summary["success_count"] > 0:
        lines.extend(
            [
                "## Error Summary (successful detections only)",
                "",
                f"- mean absolute error: `{summary['mae_frames']:.3f}` frames",
                f"- median absolute error: `{summary['median_ae_frames']:.3f}` frames",
                f"- mean signed error: `{summary['mean_signed_error']:.3f}` frames",
                f"- early predictions: `{summary['early_count']}`",
                f"- late predictions: `{summary['late_count']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Detection Methods",
            "",
        ]
    )
    for key, count in sorted(method_counts.items()):
        lines.append(f"- `{key}`: `{count}`")

    lines.extend(
        [
            "",
            "## Failure / Diagnostic Reasons",
            "",
        ]
    )
    for key, count in sorted(reason_counts.items()):
        lines.append(f"- `{key}`: `{count}`")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Evaluate automatic kick-frame detection against kick_times.csv.")
    parser.add_argument("--clips-dir", default="data/clips/penalties_720p")
    parser.add_argument("--kick-times-csv", default="data/meta/kick_times.csv")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window-start-s", type=float, default=4.0)
    parser.add_argument("--window-end-s", type=float, default=12.0)
    parser.add_argument("--ball-class-id", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--velocity-prominence-threshold", type=float, default=2.5)
    parser.add_argument("--max-tracking-jump-px", type=float, default=180.0)
    parser.add_argument("--min-sustained-velocity", type=float, default=2.0)
    parser.add_argument("--frame-adjust", type=int, default=0, help="Additive frame adjustment applied after automatic kick detection")
    parser.add_argument("--disable-peak-fallback", action="store_true")
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--clip-substring", default=None)
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    kick_times_csv = Path(args.kick_times_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kick_df = pd.read_csv(kick_times_csv).copy()
    if args.clip_substring:
        kick_df = kick_df.loc[kick_df["clip_name"].astype(str).str.contains(args.clip_substring, regex=False)].copy()
    if args.max_clips is not None:
        kick_df = kick_df.head(int(args.max_clips)).copy()

    if kick_df.empty:
        raise ValueError("No clips selected for evaluation.")

    model = load_yolo_model(args.model_path)

    rows = []
    for row in kick_df.itertuples(index=False):
        clip_name = str(row.clip_name)
        gt_frame = int(row.kick_frame)
        gt_time_s = safe_float(row.kick_time_s)
        video_path = clips_dir / clip_name

        if not video_path.exists():
            rows.append(
                {
                    "clip_name": clip_name,
                    "video_path": str(video_path).replace("\\", "/"),
                    "video_found": False,
                    "fps": None,
                    "gt_kick_frame": gt_frame,
                    "gt_kick_time_s": gt_time_s,
                    "pred_kick_frame": None,
                    "pred_confidence": 0.0,
                    "success": False,
                    "method": "failed",
                    "reason": "video_missing",
                }
            )
            continue

        try:
            fps = get_video_fps(video_path)
            window_start = int(round(float(args.window_start_s) * fps))
            window_end = int(round(float(args.window_end_s) * fps))
            details = detect_kick_frame_ball_motion_details(
                video_path=str(video_path),
                yolo_model=model,
                window_start=window_start,
                window_end=window_end,
                ball_class_id=int(args.ball_class_id),
                min_confidence=float(args.min_confidence),
                velocity_prominence_threshold=float(args.velocity_prominence_threshold),
                max_tracking_jump_px=float(args.max_tracking_jump_px),
                min_sustained_velocity=float(args.min_sustained_velocity),
                fallback_to_peak=not args.disable_peak_fallback,
            )
            pred_frame = details.get("kick_frame")
            raw_pred_frame = pred_frame
            if pred_frame is not None:
                pred_frame = max(0, int(pred_frame) + int(args.frame_adjust))
            success = pred_frame is not None
            signed_error = int(pred_frame) - gt_frame if success else None
            abs_error = abs(signed_error) if signed_error is not None else None

            rows.append(
                {
                    "clip_name": clip_name,
                    "video_path": str(video_path).replace("\\", "/"),
                    "video_found": True,
                    "fps": fps,
                    "gt_kick_frame": gt_frame,
                    "gt_kick_time_s": gt_time_s,
                    "pred_kick_frame": pred_frame,
                    "raw_pred_kick_frame": raw_pred_frame,
                    "pred_confidence": safe_float(details.get("confidence")) or 0.0,
                    "success": success,
                    "method": str(details.get("method", "")),
                    "reason": str(details.get("reason", "")),
                    "signed_error_frames": signed_error,
                    "abs_error_frames": abs_error,
                    "window_start_frame": window_start,
                    "window_end_frame": window_end,
                    "ball_detection_count": len(details.get("ball_trajectory", [])),
                    "velocity_sample_count": len(details.get("velocities", [])),
                    "peak_velocity": safe_float(details.get("peak_velocity")),
                    "baseline_velocity": safe_float(details.get("baseline_velocity")),
                    "threshold": safe_float(details.get("threshold")),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "clip_name": clip_name,
                    "video_path": str(video_path).replace("\\", "/"),
                    "video_found": True,
                    "fps": None,
                    "gt_kick_frame": gt_frame,
                    "gt_kick_time_s": gt_time_s,
                    "pred_kick_frame": None,
                    "pred_confidence": 0.0,
                    "success": False,
                    "method": "failed",
                    "reason": f"exception:{type(exc).__name__}",
                    "exception_message": str(exc),
                }
            )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "per_clip_results.csv", index=False)

    success_df = result_df.loc[result_df["success"] == True].copy()
    found_df = result_df.loc[result_df["video_found"] == True].copy()

    def rate_within(threshold: int) -> float:
        if len(found_df) == 0:
            return 0.0
        return float(((result_df["success"] == True) & (result_df["abs_error_frames"] <= threshold)).sum() / len(found_df))

    summary = {
        "clips_attempted": int(len(result_df)),
        "clips_found": int(len(found_df)),
        "success_count": int(len(success_df)),
        "success_rate": float(len(success_df) / len(found_df)) if len(found_df) else 0.0,
        "exact_accuracy": rate_within(0),
        "within_1": rate_within(1),
        "within_2": rate_within(2),
        "within_3": rate_within(3),
        "within_5": rate_within(5),
        "mae_frames": float(success_df["abs_error_frames"].mean()) if len(success_df) else None,
        "median_ae_frames": float(success_df["abs_error_frames"].median()) if len(success_df) else None,
        "mean_signed_error": float(success_df["signed_error_frames"].mean()) if len(success_df) else None,
        "early_count": int((success_df["signed_error_frames"] < 0).sum()) if len(success_df) else 0,
        "late_count": int((success_df["signed_error_frames"] > 0).sum()) if len(success_df) else 0,
        "config": {
            "clips_dir": str(clips_dir).replace("\\", "/"),
            "kick_times_csv": str(kick_times_csv).replace("\\", "/"),
            "model_path": str(Path(args.model_path)).replace("\\", "/"),
            "window_start_s": float(args.window_start_s),
            "window_end_s": float(args.window_end_s),
            "ball_class_id": int(args.ball_class_id),
            "min_confidence": float(args.min_confidence),
            "velocity_prominence_threshold": float(args.velocity_prominence_threshold),
            "max_tracking_jump_px": float(args.max_tracking_jump_px),
            "min_sustained_velocity": float(args.min_sustained_velocity),
            "fallback_to_peak": not args.disable_peak_fallback,
            "frame_adjust": int(args.frame_adjust),
        },
    }

    reason_counts = Counter(result_df["reason"].fillna("missing_reason").astype(str).tolist())
    method_counts = Counter(result_df["method"].fillna("missing_method").astype(str).tolist())

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_md = build_markdown_report(summary=summary, reason_counts=dict(reason_counts), method_counts=dict(method_counts))
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved per-clip results to: {out_dir / 'per_clip_results.csv'}")
    print(f"Saved report to: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
