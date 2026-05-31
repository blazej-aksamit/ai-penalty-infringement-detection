from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


POSE_INDEX = {
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


def landmark_row(name, landmark, image_shape):
    h, w = image_shape[:2]
    return {
        "name": name,
        "x_px": float(landmark.x * w),
        "y_px": float(landmark.y * h),
        "visibility": float(getattr(landmark, "visibility", 0.0)),
        "presence": float(getattr(landmark, "presence", 0.0)),
    }


def collect_foot_rows(landmarks, image_shape):
    rows = {}
    for name, idx in POSE_INDEX.items():
        rows[name] = landmark_row(name, landmarks[idx], image_shape)
    return rows


def best_contact_point(foot_rows):
    candidates = [
        foot_rows["left_heel"],
        foot_rows["right_heel"],
        foot_rows["left_foot_index"],
        foot_rows["right_foot_index"],
    ]
    visible = [row for row in candidates if row["visibility"] >= 0.25]
    if not visible:
        return None
    return max(visible, key=lambda row: (row["y_px"], row["visibility"]))


def draw_point(image, row, color, label):
    x = int(round(row["x_px"]))
    y = int(round(row["y_px"]))
    cv2.circle(image, (x, y), 10, color, -1)
    cv2.circle(image, (x, y), 14, (255, 255, 255), 2)
    cv2.putText(
        image,
        label,
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def resolve_manifest(source_dir: Path, explicit_manifest: str | None):
    if explicit_manifest:
        manifest_path = Path(explicit_manifest)
        return manifest_path if manifest_path.exists() else None

    candidate = source_dir.parent / f"{source_dir.name}_manifest.csv"
    if candidate.exists():
        return candidate
    return None


def load_manifest_by_image_name(manifest_path: Path | None):
    if manifest_path is None or not manifest_path.exists():
        return {}
    df = pd.read_csv(manifest_path)
    if "image_name" not in df.columns:
        return {}
    return {str(row["image_name"]): row.to_dict() for _, row in df.iterrows()}


def draw_full_frame_overlay(
    full_image,
    manifest_row,
    best_contact,
    crop_rows,
):
    vis = full_image.copy()
    crop_x1 = float(manifest_row["crop_x1"])
    crop_y1 = float(manifest_row["crop_y1"])
    crop_x2 = float(manifest_row["crop_x2"])
    crop_y2 = float(manifest_row["crop_y2"])
    orig_w = float(manifest_row["original_crop_width"])
    orig_h = float(manifest_row["original_crop_height"])
    saved_w = float(manifest_row["saved_crop_width"])
    saved_h = float(manifest_row["saved_crop_height"])

    cv2.rectangle(
        vis,
        (int(round(crop_x1)), int(round(crop_y1))),
        (int(round(crop_x2)), int(round(crop_y2))),
        (255, 255, 0),
        2,
    )

    for name, color in [
        ("left_ankle", (255, 0, 255)),
        ("right_ankle", (255, 0, 255)),
        ("left_heel", (0, 255, 255)),
        ("right_heel", (0, 255, 255)),
        ("left_foot_index", (0, 165, 255)),
        ("right_foot_index", (0, 165, 255)),
    ]:
        if name not in crop_rows:
            continue
        row = crop_rows[name]
        x_full = crop_x1 + (float(row["x_px"]) / saved_w) * orig_w
        y_full = crop_y1 + (float(row["y_px"]) / saved_h) * orig_h
        cv2.circle(vis, (int(round(x_full)), int(round(y_full))), 6, color, -1)

    if best_contact is not None:
        x_full = crop_x1 + (float(best_contact["x_px"]) / saved_w) * orig_w
        y_full = crop_y1 + (float(best_contact["y_px"]) / saved_h) * orig_h
        cv2.circle(vis, (int(round(x_full)), int(round(y_full))), 10, (0, 0, 255), -1)
        cv2.circle(vis, (int(round(x_full)), int(round(y_full))), 16, (255, 255, 255), 2)
        cv2.putText(
            vis,
            f"best={best_contact['name']}",
            (int(round(x_full)) + 12, int(round(y_full)) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    return vis


def main():
    parser = argparse.ArgumentParser(description="Run a MediaPipe pose pilot on goalkeeper crops.")
    parser.add_argument("--source", required=True, help="Directory with crop images")
    parser.add_argument("--out-dir", required=True, help="Directory for MediaPipe pilot outputs")
    parser.add_argument("--model-asset-path", required=True, help="Path to MediaPipe pose_landmarker.task model")
    parser.add_argument("--manifest-csv", default=None, help="Optional crop manifest with crop coordinates")
    parser.add_argument("--full-image-dir", default="data/line_logic_blazej_evaluation/images", help="Optional directory with full-size source frames")
    parser.add_argument("--num-poses", type=int, default=1)
    parser.add_argument("--min-pose-detection-confidence", type=float, default=0.3)
    parser.add_argument("--min-pose-presence-confidence", type=float, default=0.3)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.3)
    args = parser.parse_args()

    source_dir = Path(args.source)
    out_dir = Path(args.out_dir)
    overlay_dir = out_dir / "overlays"
    zoom_dir = out_dir / "overlays_zoom"
    full_frame_dir = out_dir / "full_frame_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    zoom_dir.mkdir(parents=True, exist_ok=True)
    full_frame_dir.mkdir(parents=True, exist_ok=True)
    manifest_map = load_manifest_by_image_name(resolve_manifest(source_dir, args.manifest_csv))
    full_image_dir = Path(args.full_image_dir)

    image_paths = sorted([p for p in source_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    base_options = mp_python.BaseOptions(model_asset_path=str(Path(args.model_asset_path)))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=args.num_poses,
        min_pose_detection_confidence=args.min_pose_detection_confidence,
        min_pose_presence_confidence=args.min_pose_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    rows = []
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            row = {
                "image_name": image_path.name,
                "landmarks_detected": bool(result.pose_landmarks),
                "best_contact_name": None,
                "best_contact_x_px": None,
                "best_contact_y_px": None,
                "best_contact_visibility": None,
            }

            overlay = image.copy()
            foot_rows = {}
            best = None
            if result.pose_landmarks:
                foot_rows = collect_foot_rows(result.pose_landmarks[0], image.shape)
                best = best_contact_point(foot_rows)
                if best is not None:
                    row["best_contact_name"] = best["name"]
                    row["best_contact_x_px"] = best["x_px"]
                    row["best_contact_y_px"] = best["y_px"]
                    row["best_contact_visibility"] = best["visibility"]

                for name, color in [
                    ("left_ankle", (255, 0, 255)),
                    ("right_ankle", (255, 0, 255)),
                    ("left_heel", (0, 255, 255)),
                    ("right_heel", (0, 255, 255)),
                    ("left_foot_index", (0, 165, 255)),
                    ("right_foot_index", (0, 165, 255)),
                ]:
                    draw_point(overlay, foot_rows[name], color, name)

                if best is not None:
                    cv2.circle(
                        overlay,
                        (int(round(best["x_px"])), int(round(best["y_px"]))),
                        7,
                        (0, 0, 255),
                        -1,
                    )

            overlay_path = overlay_dir / image_path.name
            cv2.imwrite(str(overlay_path), overlay)
            zoom_overlay = cv2.resize(
                overlay,
                (overlay.shape[1] * 2, overlay.shape[0] * 2),
                interpolation=cv2.INTER_CUBIC,
            )
            cv2.imwrite(str(zoom_dir / image_path.name), zoom_overlay)

            full_overlay_path = None
            manifest_row = manifest_map.get(image_path.name)
            if manifest_row is not None:
                full_image_path = full_image_dir / image_path.name
                if full_image_path.exists():
                    full_image = cv2.imread(str(full_image_path))
                    if full_image is not None:
                        full_overlay = draw_full_frame_overlay(
                            full_image,
                            manifest_row,
                            best,
                            foot_rows,
                        )
                        full_overlay_path = full_frame_dir / image_path.name
                        cv2.imwrite(str(full_overlay_path), full_overlay)

            for key, values in foot_rows.items():
                row[f"{key}_x_px"] = values["x_px"]
                row[f"{key}_y_px"] = values["y_px"]
                row[f"{key}_visibility"] = values["visibility"]

            row["overlay_path"] = str(overlay_path).replace("\\", "/")
            row["zoom_overlay_path"] = str((zoom_dir / image_path.name)).replace("\\", "/")
            row["full_frame_overlay_path"] = None if full_overlay_path is None else str(full_overlay_path).replace("\\", "/")
            rows.append(row)

    csv_path = out_dir / "mediapipe_pose_pilot.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "source": str(source_dir).replace("\\", "/"),
        "out_dir": str(out_dir).replace("\\", "/"),
        "model_asset_path": str(Path(args.model_asset_path)).replace("\\", "/"),
        "image_count": len(image_paths),
        "landmarks_detected": int(sum(1 for row in rows if row["landmarks_detected"])),
        "best_contact_detected": int(sum(1 for row in rows if row["best_contact_name"])),
        "csv_path": str(csv_path).replace("\\", "/"),
        "overlay_dir": str(overlay_dir).replace("\\", "/"),
        "zoom_overlay_dir": str(zoom_dir).replace("\\", "/"),
        "full_frame_overlay_dir": str(full_frame_dir).replace("\\", "/"),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved overlays to: {overlay_dir}")


if __name__ == "__main__":
    main()
