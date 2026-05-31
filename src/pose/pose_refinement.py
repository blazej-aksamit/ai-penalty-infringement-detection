from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


COCO_LEFT_KNEE = 13
COCO_RIGHT_KNEE = 14
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16

_POSE_MODEL_CACHE: Dict[str, Any] = {}


def load_pose_model(model_path: str):
    model_key = str(model_path)
    if model_key in _POSE_MODEL_CACHE:
        return _POSE_MODEL_CACHE[model_key]

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ultralytics is not installed in the active Python environment. "
            "Run pose-assisted pipeline steps from the YOLO environment."
        ) from exc

    model = YOLO(model_path)
    _POSE_MODEL_CACHE[model_key] = model
    return model


def _expand_box(
    box: Dict[str, float],
    img_w: int,
    img_h: int,
    pad_ratio_x: float,
    pad_ratio_y_top: float,
    pad_ratio_y_bottom: float,
):
    x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
    bw = x2 - x1
    bh = y2 - y1

    pad_x = bw * pad_ratio_x
    pad_y_top = bh * pad_ratio_y_top
    pad_y_bottom = bh * pad_ratio_y_bottom

    nx1 = max(0, int(round(x1 - pad_x)))
    ny1 = max(0, int(round(y1 - pad_y_top)))
    nx2 = min(img_w - 1, int(round(x2 + pad_x)))
    ny2 = min(img_h - 1, int(round(y2 + pad_y_bottom)))
    return nx1, ny1, nx2, ny2


def _resize_for_pose(crop: np.ndarray, min_short_side: int, min_long_side: int):
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return crop, 1.0

    short_side = min(h, w)
    long_side = max(h, w)

    scale_candidates = [1.0]
    if min_short_side > 0 and short_side < min_short_side:
        scale_candidates.append(float(min_short_side) / float(short_side))
    if min_long_side > 0 and long_side < min_long_side:
        scale_candidates.append(float(min_long_side) / float(long_side))

    scale = max(scale_candidates)
    if scale <= 1.0:
        return crop, 1.0

    new_w = max(1, int(math.ceil(w * scale)))
    new_h = max(1, int(math.ceil(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized, scale


def _pick_detection(result) -> Optional[int]:
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None or boxes.conf is None or keypoints.data is None:
        return None

    if len(boxes) == 0 or len(keypoints.data) == 0:
        return None

    confs = boxes.conf.cpu().numpy()
    if len(confs) == 0:
        return None
    return int(np.argmax(confs))


def _safe_xyxy(result) -> Optional[np.ndarray]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return None
    return boxes.xyxy.cpu().numpy()


def _bbox_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _center_distance(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    return float(np.hypot(acx - bcx, acy - bcy))


def _pick_detection_by_target_box(
    result,
    *,
    target_box_full: Tuple[float, float, float, float],
    scale: float,
    offset_x: float,
    offset_y: float,
) -> Optional[int]:
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    xyxy = _safe_xyxy(result)
    if boxes is None or keypoints is None or boxes.conf is None or keypoints.data is None or xyxy is None:
        return None

    confs = boxes.conf.cpu().numpy()
    if len(confs) == 0:
        return None

    best_idx = None
    best_score = -1e18
    for idx, det_box in enumerate(xyxy):
        x1, y1, x2, y2 = det_box.tolist()
        mapped_box = (
            float(offset_x + x1 / scale),
            float(offset_y + y1 / scale),
            float(offset_x + x2 / scale),
            float(offset_y + y2 / scale),
        )
        iou = _bbox_iou(mapped_box, target_box_full)
        dist = _center_distance(mapped_box, target_box_full)
        conf = float(confs[idx])
        score = 6.0 * iou + conf - 0.002 * dist
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _extract_keypoints(result, det_idx: int):
    data = result.keypoints.data[det_idx].cpu().numpy()
    # shape: (17, 3) -> x, y, conf
    return data


def _run_pose_predict(model, image: np.ndarray, *, pose_conf: float, pose_imgsz: int):
    results = model.predict(
        source=image,
        conf=pose_conf,
        imgsz=pose_imgsz,
        verbose=False,
        save=False,
    )
    if not results:
        return None
    return results[0]


def _point_dict(
    name: str,
    kp,
    scale: float,
    crop_x1: int,
    crop_y1: int,
    gk_y2: float,
    *,
    ground_proxy_x: Optional[float] = None,
):
    x_up, y_up, conf = float(kp[0]), float(kp[1]), float(kp[2])
    x_crop = x_up / scale
    y_crop = y_up / scale
    x_full = crop_x1 + x_crop
    y_full = crop_y1 + y_crop
    return {
        "name": name,
        "x_full": x_full,
        "y_full": y_full,
        "ground_proxy_x": x_full if ground_proxy_x is None else float(ground_proxy_x),
        "ground_proxy_y": gk_y2,
        "confidence": conf,
    }


def _estimate_ground_proxy_x(
    ankle_kp,
    knee_kp,
    *,
    scale: float,
    offset_x: int,
    leg_extension_factor: float,
):
    ankle_x_full = float(offset_x + float(ankle_kp[0]) / scale)
    if knee_kp is None:
        return ankle_x_full

    knee_x_full = float(offset_x + float(knee_kp[0]) / scale)
    return ankle_x_full + leg_extension_factor * (ankle_x_full - knee_x_full)


def _collect_pose_points(
    keypoints: np.ndarray,
    *,
    scale: float,
    offset_x: int,
    offset_y: int,
    gk_y2: float,
    min_keypoint_conf: float,
    leg_extension_factor: float,
):
    pose_points: Dict[str, Dict[str, float]] = {}
    ankle_to_knee = {
        "pose_left_ankle": COCO_LEFT_KNEE,
        "pose_right_ankle": COCO_RIGHT_KNEE,
    }
    for name, kp_idx in [
        ("pose_left_ankle", COCO_LEFT_ANKLE),
        ("pose_right_ankle", COCO_RIGHT_ANKLE),
        ("pose_left_knee", COCO_LEFT_KNEE),
        ("pose_right_knee", COCO_RIGHT_KNEE),
    ]:
        kp = keypoints[kp_idx]
        if float(kp[2]) < float(min_keypoint_conf):
            continue
        ground_proxy_x = None
        if name in ankle_to_knee:
            knee_idx = ankle_to_knee[name]
            knee_kp = keypoints[knee_idx]
            if float(knee_kp[2]) >= float(min_keypoint_conf):
                ground_proxy_x = _estimate_ground_proxy_x(
                    kp,
                    knee_kp,
                    scale=scale,
                    offset_x=offset_x,
                    leg_extension_factor=leg_extension_factor,
                )
        pose_points[name] = _point_dict(
            name,
            kp,
            scale,
            offset_x,
            offset_y,
            gk_y2,
            ground_proxy_x=ground_proxy_x,
        )
    return pose_points


def _select_pose_points(pose_points: Dict[str, Dict[str, float]]):
    selected_pose_points = []
    for name in ["pose_left_ankle", "pose_right_ankle"]:
        if name in pose_points:
            selected_pose_points.append(name)
    if not selected_pose_points:
        for name in ["pose_left_knee", "pose_right_knee"]:
            if name in pose_points:
                selected_pose_points.append(name)
    return selected_pose_points


def _pick_support_ankle(pose_points: Dict[str, Dict[str, float]], selected_pose_points):
    ankle_names = [name for name in selected_pose_points if "ankle" in str(name)]
    if not ankle_names:
        return None

    return max(
        ankle_names,
        key=lambda name: (
            float(pose_points[name]["y_full"]),
            float(pose_points[name]["confidence"]),
        ),
    )


def run_pose_refinement(
    frame: np.ndarray,
    gk_box: Optional[Dict[str, float]],
    *,
    pose_model_path: Optional[str],
    out_dir: Optional[Path] = None,
    pose_conf: float = 0.25,
    pose_imgsz: int = 640,
    pad_ratio_x: float = 0.5,
    pad_ratio_y_top: float = 0.35,
    pad_ratio_y_bottom: float = 0.45,
    min_short_side: int = 256,
    min_long_side: int = 320,
    min_keypoint_conf: float = 0.2,
    leg_extension_factor: float = 0.35,
    full_frame_fallback: bool = True,
    full_frame_pose_conf: float = 0.1,
    full_frame_pose_imgsz: int = 960,
) -> Dict[str, Any]:
    result_payload: Dict[str, Any] = {
        "available": False,
        "reason": "pose_not_requested" if not pose_model_path else "pose_not_run",
        "pose_points": {},
        "pose_point_count": 0,
        "selected_pose_points": [],
    }

    if gk_box is None:
        result_payload["reason"] = "no_goalkeeper"
        return result_payload
    if not pose_model_path:
        return result_payload

    img_h, img_w = frame.shape[:2]
    crop_x1, crop_y1, crop_x2, crop_y2 = _expand_box(
        gk_box,
        img_w=img_w,
        img_h=img_h,
        pad_ratio_x=pad_ratio_x,
        pad_ratio_y_top=pad_ratio_y_top,
        pad_ratio_y_bottom=pad_ratio_y_bottom,
    )
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    if crop.size == 0:
        result_payload["reason"] = "empty_crop"
        return result_payload

    resized_crop, scale = _resize_for_pose(crop, min_short_side=min_short_side, min_long_side=min_long_side)
    pose_model = load_pose_model(str(pose_model_path))
    gk_y2 = float(gk_box["y2"])
    target_box_full = (
        float(gk_box["x1"]),
        float(gk_box["y1"]),
        float(gk_box["x2"]),
        float(gk_box["y2"]),
    )

    result = _run_pose_predict(
        pose_model,
        resized_crop,
        pose_conf=pose_conf,
        pose_imgsz=pose_imgsz,
    )
    source_mode = "crop"
    det_idx = None if result is None else _pick_detection_by_target_box(
        result,
        target_box_full=target_box_full,
        scale=scale,
        offset_x=crop_x1,
        offset_y=crop_y1,
    )

    pose_points: Dict[str, Dict[str, float]] = {}
    selected_pose_points = []
    if result is not None and det_idx is not None:
        keypoints = _extract_keypoints(result, det_idx)
        if keypoints.shape[0] > COCO_RIGHT_ANKLE:
            pose_points = _collect_pose_points(
                keypoints,
                scale=scale,
                offset_x=crop_x1,
                offset_y=crop_y1,
                gk_y2=gk_y2,
                min_keypoint_conf=min_keypoint_conf,
                leg_extension_factor=leg_extension_factor,
            )
            selected_pose_points = _select_pose_points(pose_points)

    if (not selected_pose_points) and full_frame_fallback:
        full_result = _run_pose_predict(
            pose_model,
            frame,
            pose_conf=full_frame_pose_conf,
            pose_imgsz=max(full_frame_pose_imgsz, pose_imgsz),
        )
        full_det_idx = None if full_result is None else _pick_detection_by_target_box(
            full_result,
            target_box_full=target_box_full,
            scale=1.0,
            offset_x=0.0,
            offset_y=0.0,
        )
        if full_result is not None and full_det_idx is not None:
            keypoints = _extract_keypoints(full_result, full_det_idx)
            if keypoints.shape[0] > COCO_RIGHT_ANKLE:
                pose_points = _collect_pose_points(
                    keypoints,
                    scale=1.0,
                    offset_x=0,
                    offset_y=0,
                    gk_y2=gk_y2,
                    min_keypoint_conf=min_keypoint_conf,
                    leg_extension_factor=leg_extension_factor,
                )
                selected_pose_points = _select_pose_points(pose_points)
                if selected_pose_points:
                    result = full_result
                    source_mode = "full_frame"
                    scale = 1.0
                    crop_x1 = 0
                    crop_y1 = 0
                    crop = frame.copy()

    result_payload.update(
        {
            "available": len(selected_pose_points) > 0,
            "reason": "ok" if len(selected_pose_points) > 0 else ("no_person_detected" if det_idx is None and not pose_points else "no_reliable_lower_body_keypoints"),
            "pose_points": pose_points,
            "pose_point_count": len(pose_points),
            "selected_pose_points": selected_pose_points,
            "decision_pose_point": _pick_support_ankle(pose_points, selected_pose_points),
            "crop_box": {
                "x1": crop_x1,
                "y1": crop_y1,
                "x2": crop_x2,
                "y2": crop_y2,
            },
            "scale": scale,
            "source_mode": source_mode,
        }
    )

    if out_dir is not None:
        pose_dir = Path(out_dir)
        pose_dir.mkdir(parents=True, exist_ok=True)
        crop_path = pose_dir / "pose_crop.jpg"
        overlay_path = pose_dir / "pose_overlay.jpg"
        json_path = pose_dir / "pose_result.json"

        cv2.imwrite(str(crop_path), crop)
        plotted = result.plot()
        cv2.imwrite(str(overlay_path), plotted)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)

        result_payload["crop_path"] = str(crop_path).replace("\\", "/")
        result_payload["overlay_path"] = str(overlay_path).replace("\\", "/")
        result_payload["json_path"] = str(json_path).replace("\\", "/")

    return result_payload


def get_pose_guided_ground_points(gk_box: Dict[str, float], pose_result: Optional[Dict[str, Any]]):
    points = {
        "left_bottom": (float(gk_box["x1"]), float(gk_box["y2"])),
        "center_bottom": ((float(gk_box["x1"]) + float(gk_box["x2"])) / 2.0, float(gk_box["y2"])),
        "right_bottom": (float(gk_box["x2"]), float(gk_box["y2"])),
    }

    if not pose_result or not pose_result.get("available"):
        return points

    decision_pose_point = pose_result.get("decision_pose_point")
    if not decision_pose_point:
        return points

    point = pose_result.get("pose_points", {}).get(decision_pose_point)
    if not point:
        return points

    points[decision_pose_point] = (
        float(point["ground_proxy_x"]),
        float(point["ground_proxy_y"]),
    )

    return points
