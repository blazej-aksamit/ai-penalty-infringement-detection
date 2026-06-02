import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.kick_detection.ball_motion_detector import (
    detect_kick_frame_ball_motion_details,
    load_yolo_model,
)
from src.pipeline.run_full_penalty_pipeline import extract_frame


def _load_ultralytics_model(model_path: Path):
    from ultralytics import YOLO

    return YOLO(str(model_path))


def predict_boxes(
    model,
    image_bgr,
    conf: float = 0.2,
    classes: Optional[List[int]] = None,
    imgsz: int = 640,
) -> List[Dict[str, float]]:
    results = model.predict(
        source=image_bgr,
        conf=conf,
        classes=classes,
        imgsz=imgsz,
        verbose=False,
    )
    boxes: List[Dict[str, float]] = []
    if not results:
        return boxes

    result = results[0]
    if result.boxes is None:
        return boxes

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clses = result.boxes.cls.cpu().numpy()
    for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clses):
        boxes.append(
            {
                "cls": int(cls_id),
                "conf": float(score),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )
    return boxes


def pick_goalkeeper_box(boxes: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    candidates = [b for b in boxes if int(b["cls"]) == 0]
    if not candidates:
        return None
    return sorted(candidates, key=lambda b: b["conf"], reverse=True)[0]


def pick_ball_box(boxes: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    candidates = [b for b in boxes if int(b["cls"]) == 1]
    if not candidates:
        return None
    return sorted(candidates, key=lambda b: b["conf"], reverse=True)[0]


def box_center(box: Dict[str, float]) -> Tuple[float, float]:
    return ((box["x1"] + box["x2"]) / 2.0, (box["y1"] + box["y2"]) / 2.0)


def bottom_points(box: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    return {
        "left_bottom": (box["x1"], box["y2"]),
        "center_bottom": ((box["x1"] + box["x2"]) / 2.0, box["y2"]),
        "right_bottom": (box["x2"], box["y2"]),
    }


def iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    xa1, ya1, xa2, ya2 = a["x1"], a["y1"], a["x2"], a["y2"]
    xb1, yb1, xb2, yb2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1 = max(xa1, xb1)
    iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2)
    iy2 = min(ya2, yb2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def deduplicate_person_boxes(
    boxes: List[Dict[str, float]],
    iou_thresh: float = 0.6,
) -> List[Dict[str, float]]:
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: float(b.get("conf", 0.0)), reverse=True)
    kept: List[Dict[str, float]] = []
    for box in ordered:
        if any(iou(box, existing) >= iou_thresh for existing in kept):
            continue
        kept.append(box)
    return kept


def signed_line_value(point: Tuple[float, float], line: Tuple[int, int, int, int]) -> float:
    px, py = point
    x1, y1, x2, y2 = line
    return (y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1


def point_line_distance(point: Tuple[float, float], line: Tuple[int, int, int, int]) -> float:
    px, py = point
    x1, y1, x2, y2 = line
    den = float(np.hypot(y2 - y1, x2 - x1))
    if den == 0:
        return 1e9
    return abs(signed_line_value(point, line)) / den


def point_segment_distance(point: Tuple[float, float], line: Tuple[int, int, int, int]) -> float:
    px, py = point
    x1, y1, x2, y2 = map(float, line)
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-6:
        return float(np.hypot(px - x1, py - y1))
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return float(np.hypot(px - proj_x, py - proj_y))


def clamp_int(v: float, lo: int, hi: int) -> int:
    return max(lo, min(int(round(v)), hi))


def estimate_pitch_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (28, 35, 35), (95, 255, 255))
    green_mask = cv2.medianBlur(green_mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return green_mask


def estimate_whiteline_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, 155), (180, 75, 255))


def sample_mask_ratio(mask: np.ndarray, center: Tuple[float, float], half_size: int = 8) -> float:
    h, w = mask.shape[:2]
    cx = clamp_int(center[0], 0, w - 1)
    cy = clamp_int(center[1], 0, h - 1)
    x1 = max(0, cx - half_size)
    x2 = min(w, cx + half_size + 1)
    y1 = max(0, cy - half_size)
    y2 = min(h, cy + half_size + 1)
    patch = mask[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(np.count_nonzero(patch)) / float(patch.size)


def distance_point_to_box(point: Tuple[float, float], box: Dict[str, float]) -> float:
    px, py = point
    dx = max(box["x1"] - px, 0.0, px - box["x2"])
    dy = max(box["y1"] - py, 0.0, py - box["y2"])
    return float(np.hypot(dx, dy))


def line_y_at_x(line: Tuple[int, int, int, int], x: float) -> Optional[float]:
    x1, y1, x2, y2 = map(float, line)
    dx = x2 - x1
    if abs(dx) < 1e-6:
        return None
    t = (x - x1) / dx
    return y1 + t * (y2 - y1)


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def fit_player_alignment_line(
    *,
    gk_center: Tuple[float, float],
    anchor_center: Tuple[float, float],
    person_boxes: List[Dict[str, object]],
    goalkeeper_idx: Optional[int],
    kicker_idx: Optional[int],
    pitch_mask: np.ndarray,
    whiteline_mask: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    field_direction = 1.0 if anchor_center[0] >= gk_center[0] else -1.0
    x_lo = min(gk_center[0], anchor_center[0]) - 320.0
    x_hi = max(gk_center[0], anchor_center[0]) + 320.0
    y_lo = min(gk_center[1], anchor_center[1]) - 60.0
    y_hi = max(gk_center[1], anchor_center[1]) + 280.0

    points: List[Tuple[float, float]] = []
    for idx, person in enumerate(person_boxes):
        if idx in {goalkeeper_idx, kicker_idx}:
            continue
        if not person.get("on_pitch", False):
            continue
        if not person.get("likely_player", False):
            continue
        pt = bottom_points(person)["center_bottom"]
        if not (x_lo <= pt[0] <= x_hi and y_lo <= pt[1] <= y_hi):
            continue
        progress = (pt[0] - gk_center[0]) * field_direction
        if progress < 20.0:
            continue
        ball_foot_distance = float(person.get("ball_foot_distance") or 1e9)
        ball_box_distance = float(person.get("ball_box_distance") or 1e9)
        near_anchor_zone = (
            abs(pt[0] - anchor_center[0]) <= 380.0
            and abs(pt[1] - anchor_center[1]) <= 260.0
        )
        near_ball_zone = ball_foot_distance <= 340.0 or ball_box_distance <= 200.0
        if not (near_anchor_zone or near_ball_zone):
            continue
        points.append((float(pt[0]), float(pt[1])))

    if len(points) < 2:
        return None

    if len(points) == 2:
        (px1, py1), (px2, py2) = points
        if abs(px2 - px1) < 1e-6:
            return None
        slope = float((py2 - py1) / (px2 - px1))
        x0 = float((px1 + px2) / 2.0)
        y0 = float((py1 + py2) / 2.0)
        angle = abs(float(np.degrees(np.arctan2(py2 - py1, px2 - px1))))
    else:
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        try:
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        except cv2.error:
            return None
        if abs(vx) < 1e-6:
            return None
        slope = float(vy / vx)
        angle = abs(float(np.degrees(np.arctan2(vy, vx))))

    if angle > 32.0 and angle < 148.0:
        return None

    residuals = [abs(py - (y0 + slope * (px - x0))) for px, py in points]
    if mean(residuals) > 70.0:
        return None

    xs = [pt[0] for pt in points]
    line_x1 = max(0.0, min(xs) - 70.0)
    line_x2 = max(line_x1 + 40.0, min(float(pitch_mask.shape[1] - 1), max(xs) + 70.0))
    line_y1 = y0 + slope * (line_x1 - x0)
    line_y2 = y0 + slope * (line_x2 - x0)

    h, w = pitch_mask.shape[:2]
    line = (
        clamp_int(line_x1, 0, w - 1),
        clamp_int(line_y1, 0, h - 1),
        clamp_int(line_x2, 0, w - 1),
        clamp_int(line_y2, 0, h - 1),
    )

    gk_dist = point_line_distance(gk_center, line)
    anchor_dist = point_line_distance(anchor_center, line)
    if gk_dist < 30.0 or gk_dist > 620.0:
        return None
    if anchor_dist < 5.0 or anchor_dist > 380.0:
        return None

    return line


def find_relaxed_penalty_line_candidate(
    *,
    all_lines: List[Tuple[int, int, int, int]],
    gk_center: Tuple[float, float],
    anchor_center: Tuple[float, float],
    active_player_bottoms: List[Tuple[float, float]],
    kicker_bottom: Optional[Tuple[float, float]],
    pitch_mask: np.ndarray,
    whiteline_mask: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    if not all_lines:
        return None

    h, w = pitch_mask.shape[:2]
    field_direction = 1.0 if anchor_center[0] >= gk_center[0] else -1.0
    field_span_x = max(40.0, abs(anchor_center[0] - gk_center[0]))
    relaxed_candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []

    for line in all_lines:
        x1, y1, x2, y2 = line
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 70.0:
            continue

        angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        if angle > 55.0 and angle < 125.0:
            continue

        xmid = (x1 + x2) / 2.0
        ymid = (y1 + y2) / 2.0
        if not (w * 0.05 <= xmid <= w * 0.98):
            continue
        if not (h * 0.22 <= ymid <= h * 0.94):
            continue

        x_progress = (xmid - gk_center[0]) * field_direction
        if x_progress <= -30.0:
            continue
        if x_progress > field_span_x + 420.0:
            continue

        gk_side = signed_line_value(gk_center, line)
        anchor_side = signed_line_value(anchor_center, line)
        if gk_side == 0 or anchor_side == 0:
            continue
        if gk_side * anchor_side <= 0:
            continue

        gk_dist = point_line_distance(gk_center, line)
        anchor_dist = point_line_distance(anchor_center, line)
        if gk_dist < 25.0 or gk_dist > 700.0:
            continue
        if anchor_dist < 0.0 or anchor_dist > 460.0:
            continue

        support = line_support_stats(line, pitch_mask, whiteline_mask)
        if support["white_avg"] < 0.006:
            continue
        if support["pitch_avg"] < 0.12:
            continue
        if min(support["pitch_pos_avg"], support["pitch_neg_avg"]) < 0.015:
            continue

        player_hits = 0
        player_distances: List[float] = []
        for px, py in active_player_bottoms:
            if px < min(x1, x2) - 120 or px > max(x1, x2) + 120:
                continue
            ly = line_y_at_x(line, px)
            if ly is None:
                continue
            vertical_gap = abs(py - ly)
            player_distances.append(vertical_gap)
            if vertical_gap <= 155.0:
                player_hits += 1

        kicker_gap = 999.0
        if kicker_bottom is not None:
            ly = line_y_at_x(line, kicker_bottom[0])
            if ly is not None:
                kicker_gap = abs(kicker_bottom[1] - ly)

        if player_hits == 0 and not player_distances and kicker_gap > 230.0:
            continue
        if player_hits == 0 and player_distances and mean(player_distances) > 220.0 and kicker_gap > 210.0:
            continue

        score = length
        score += max(0.0, 280.0 - abs(gk_dist - 170.0)) * 0.55
        score += max(0.0, 220.0 - abs(anchor_dist - 80.0)) * 0.45
        score += support["white_avg"] * 180.0
        score += min(support["pitch_pos_avg"], support["pitch_neg_avg"]) * 140.0
        score += min(player_hits, 4) * 55.0
        if player_distances:
            score += max(0.0, 170.0 - mean(player_distances)) * 1.0
        if kicker_gap < 999.0:
            score += max(0.0, 220.0 - kicker_gap) * 0.20
        relaxed_candidates.append((score, line))

    if not relaxed_candidates:
        return None
    relaxed_candidates.sort(key=lambda item: item[0], reverse=True)
    return relaxed_candidates[0][1]


def infer_goalkeeper_from_people(
    *,
    image_shape: Tuple[int, int],
    person_boxes: List[Dict[str, object]],
    ball_box: Optional[Dict[str, float]],
) -> Tuple[Optional[int], Optional[Dict[str, float]]]:
    h, w = image_shape[:2]
    if not person_boxes:
        return None, None

    active = [(idx, person) for idx, person in enumerate(person_boxes) if person.get("on_pitch", False)]
    if not active:
        return None, None

    centers = [(idx, box_center(person)) for idx, person in active]
    ys = [cy for _, (_, cy) in centers]
    median_y = float(np.median(ys)) if ys else (h / 2.0)

    ball_center = box_center(ball_box) if ball_box is not None else None
    scored: List[Tuple[float, int]] = []
    for idx, person in active:
        cx, cy = box_center(person)
        height = float(person["y2"] - person["y1"])
        width = float(person["x2"] - person["x1"])
        if height < 45.0 or width < 18.0:
            continue
        edge_proximity = max(cx, w - cx)
        score = edge_proximity * 0.35
        score -= abs(cy - median_y) * 0.12
        score -= max(0.0, cy - median_y) * 0.18
        score += min(height, 220.0) * 0.08
        if ball_center is not None:
            bx, by = ball_center
            score += min(abs(cx - bx), 900.0) * 0.06
            score -= max(0.0, cy - by) * 0.10
            if cy <= by + 80.0:
                score += 28.0
        scored.append((score, idx))

    if not scored:
        return None, None
    scored.sort(reverse=True)
    best_idx = scored[0][1]
    person = person_boxes[best_idx]
    gk_box = {
        "cls": 0,
        "conf": float(person.get("conf", 0.0)),
        "x1": float(person["x1"]),
        "y1": float(person["y1"]),
        "x2": float(person["x2"]),
        "y2": float(person["y2"]),
    }
    return best_idx, gk_box


def compute_motion_map(curr_bgr: np.ndarray, prev_bgr: Optional[np.ndarray]) -> np.ndarray:
    if prev_bgr is None or prev_bgr.shape != curr_bgr.shape:
        return np.zeros(curr_bgr.shape[:2], dtype=np.uint8)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(curr_gray, prev_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    return diff


def motion_score_for_box(motion_map: np.ndarray, box: Dict[str, float]) -> float:
    h, w = motion_map.shape[:2]
    x1 = clamp_int(box["x1"], 0, w - 1)
    x2 = clamp_int(box["x2"], 0, w - 1)
    y1 = clamp_int(box["y1"], 0, h - 1)
    y2 = clamp_int(box["y2"], 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = motion_map[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch))


def line_support_stats(
    line: Tuple[int, int, int, int],
    pitch_mask: np.ndarray,
    whiteline_mask: np.ndarray,
    offset_px: float = 14.0,
    samples: int = 25,
) -> Dict[str, float]:
    h, w = pitch_mask.shape[:2]
    x1, y1, x2, y2 = map(float, line)
    dx = x2 - x1
    dy = y2 - y1
    seg_len = float(np.hypot(dx, dy))
    if seg_len <= 1e-6:
        return {
            "white_avg": 0.0,
            "pitch_avg": 0.0,
            "pitch_pos_avg": 0.0,
            "pitch_neg_avg": 0.0,
        }

    nx = -dy / seg_len
    ny = dx / seg_len
    white_vals: List[float] = []
    pitch_vals: List[float] = []
    pitch_pos_vals: List[float] = []
    pitch_neg_vals: List[float] = []

    for t in np.linspace(0.05, 0.95, samples):
        px = x1 + dx * t
        py = y1 + dy * t
        white_vals.append(sample_mask_ratio(whiteline_mask, (px, py), half_size=4))
        pitch_vals.append(sample_mask_ratio(pitch_mask, (px, py), half_size=5))
        pitch_pos_vals.append(sample_mask_ratio(pitch_mask, (px + nx * offset_px, py + ny * offset_px), half_size=5))
        pitch_neg_vals.append(sample_mask_ratio(pitch_mask, (px - nx * offset_px, py - ny * offset_px), half_size=5))

    return {
        "white_avg": mean(white_vals),
        "pitch_avg": mean(pitch_vals),
        "pitch_pos_avg": mean(pitch_pos_vals),
        "pitch_neg_avg": mean(pitch_neg_vals),
    }


def extract_jersey_hsv(image_bgr: np.ndarray, box: Dict[str, float]) -> Optional[Tuple[float, float, float]]:
    h, w = image_bgr.shape[:2]
    x1 = clamp_int(box["x1"], 0, w - 1)
    x2 = clamp_int(box["x2"], 0, w - 1)
    y1 = clamp_int(box["y1"], 0, h - 1)
    y2 = clamp_int(box["y2"], 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1
    crop_x1 = clamp_int(x1 + bw * 0.2, 0, w - 1)
    crop_x2 = clamp_int(x2 - bw * 0.2, 0, w - 1)
    crop_y1 = clamp_int(y1 + bh * 0.15, 0, h - 1)
    crop_y2 = clamp_int(y1 + bh * 0.55, 0, h - 1)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    patch = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    if patch.size == 0:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    flat = hsv.reshape(-1, 3)
    sat_mask = flat[:, 1] >= 40
    val_mask = flat[:, 2] >= 40
    valid = flat[sat_mask & val_mask]
    if valid.size == 0:
        valid = flat
    if valid.size == 0:
        return None
    med = np.median(valid, axis=0)
    return float(med[0]), float(med[1]), float(med[2])


def hsv_distance(a: Optional[Tuple[float, float, float]], b: Optional[Tuple[float, float, float]]) -> float:
    if a is None or b is None:
        return 1e9
    hue_diff = abs(a[0] - b[0])
    hue_diff = min(hue_diff, 180.0 - hue_diff)
    sat_diff = abs(a[1] - b[1]) / 4.0
    val_diff = abs(a[2] - b[2]) / 6.0
    return float(hue_diff + sat_diff + val_diff)


def is_probably_on_pitch(
    box: Dict[str, float],
    pitch_mask: np.ndarray,
    whiteline_mask: np.ndarray,
) -> Tuple[bool, Dict[str, float]]:
    pts = bottom_points(box)
    center_bottom = pts["center_bottom"]
    pitch_ratio = sample_mask_ratio(pitch_mask, center_bottom, half_size=9)
    line_ratio = sample_mask_ratio(whiteline_mask, center_bottom, half_size=9)
    below_ratio = sample_mask_ratio(pitch_mask, (center_bottom[0], center_bottom[1] + 8.0), half_size=7)
    below_line_ratio = sample_mask_ratio(whiteline_mask, (center_bottom[0], center_bottom[1] + 8.0), half_size=7)
    left_ratio = sample_mask_ratio(pitch_mask, pts["left_bottom"], half_size=6)
    right_ratio = sample_mask_ratio(pitch_mask, pts["right_bottom"], half_size=6)

    support_pitch = max(left_ratio, right_ratio, below_ratio)
    # White field lines should count only when they are supported by nearby grass.
    on_pitch = (
        pitch_ratio >= 0.28
        or below_ratio >= 0.25
        or (line_ratio >= 0.10 and support_pitch >= 0.18)
        or (below_line_ratio >= 0.10 and support_pitch >= 0.16)
        or max(left_ratio, right_ratio) >= 0.24
    )
    return on_pitch, {
        "pitch_ratio": float(pitch_ratio),
        "line_ratio": float(line_ratio),
        "below_pitch_ratio": float(below_ratio),
        "below_line_ratio": float(below_line_ratio),
        "left_pitch_ratio": float(left_ratio),
        "right_pitch_ratio": float(right_ratio),
        "support_pitch_ratio": float(support_pitch),
    }


def _penalty_spot_from_trajectory(kick_details: Optional[Dict[str, object]]) -> Optional[Tuple[float, float]]:
    """Return the ball position from the earliest trajectory entry (ball at rest on penalty spot).

    When the ball is in flight at the analyzed frame its current bbox position is far
    from the penalty spot, causing wrong kicker identification.  The trajectory recorded
    during auto-kick detection starts before the ball moves, so the first few entries give
    a reliable penalty-spot estimate.
    """
    if kick_details is None:
        return None
    traj = kick_details.get("ball_trajectory")
    if not traj:
        return None
    # Take median of first 3 entries to smooth any outliers
    early = traj[:3]
    xs = [float(entry[1]) for entry in early]
    ys = [float(entry[2]) for entry in early]
    return (float(np.median(xs)), float(np.median(ys)))


def pick_kicker_idx(
    person_boxes: List[Dict[str, object]],
    ball_box: Optional[Dict[str, float]],
    goalkeeper_idx: Optional[int],
    gk_box: Optional[Dict[str, float]] = None,
    kick_details: Optional[Dict[str, object]] = None,
) -> Optional[int]:
    scored: List[Tuple[float, int]] = []
    if ball_box is not None:
        # Prefer early-trajectory position (ball at penalty spot) over current ball bbox.
        # At the kick frame the ball is already in flight; its current position is far from
        # the penalty spot and causes the kicker to be confused with a nearby encroacher.
        penalty_spot = _penalty_spot_from_trajectory(kick_details)
        ball_center = penalty_spot if penalty_spot is not None else box_center(ball_box)
        ball_x, ball_y = ball_center
        candidate_pool = [
            (idx, person) for idx, person in enumerate(person_boxes)
            if idx != goalkeeper_idx and person.get("on_pitch", True)
        ]
        likely_pool = [(idx, person) for idx, person in candidate_pool if person.get("likely_player", False)]
        active_pool = likely_pool if likely_pool else candidate_pool

        # When we have a reliable penalty-spot estimate, apply a hard distance filter:
        # the kicker must be within 250px of the spot (handles zoomed-out broadcast views).
        # Tiny bounding boxes (height < 50px) are unlikely to be the kicker - they're
        # distant players or partial detections.
        if penalty_spot is not None:
            close_pool = [
                (idx, person) for idx, person in active_pool
                if distance_point_to_box(penalty_spot, person) <= 250.0
                and float(person["y2"] - person["y1"]) >= 50.0
            ]
            if close_pool:
                active_pool = close_pool

        for idx, person in active_pool:
            cx, cy = box_center(person)
            points = bottom_points(person)
            center_bottom = points["center_bottom"]
            foot_dist = float(np.hypot(center_bottom[0] - ball_x, center_bottom[1] - ball_y))
            box_dist = distance_point_to_box(ball_center, person)
            horiz_gap = abs(cx - ball_x)
            vertical_to_feet = abs(center_bottom[1] - ball_y)
            motion = float(person.get("motion_score", 0.0))
            height = float(person["y2"] - person["y1"])
            likely_bonus = 18.0 if person.get("likely_player", False) else 0.0
            ball_within_body_x = person["x1"] - 8.0 <= ball_x <= person["x2"] + 8.0
            ball_near_feet = person["y1"] + height * 0.45 <= ball_y <= person["y2"] + 22.0

            if foot_dist > max(135.0, height * 1.05) and box_dist > 35.0:
                continue

            # Prefer the player whose lower body is actually closest to the ball,
            # but trust the trajectory-derived penalty spot more strongly when available.
            if penalty_spot is not None:
                score = motion * 0.90
                score -= foot_dist * 0.48
                score -= box_dist * 0.16
                score -= horiz_gap * 0.10
                score -= vertical_to_feet * 0.08
                score += min(height, 220.0) * 0.03
            else:
                score = motion * 1.35
                score -= foot_dist * 0.42
                score -= box_dist * 0.12
                score -= horiz_gap * 0.08
                score -= vertical_to_feet * 0.06
                score += min(height, 220.0) * 0.02
            score += likely_bonus
            if ball_within_body_x:
                score += 22.0
            if ball_near_feet:
                score += 28.0
            scored.append((score, idx))
    else:
        if gk_box is None:
            return None
        gk_center = box_center(gk_box)
        active: List[Tuple[int, Dict[str, object]]] = []
        for idx, person in enumerate(person_boxes):
            if idx == goalkeeper_idx:
                continue
            if not person.get("on_pitch", False):
                continue
            if not person.get("likely_player", False):
                continue
            active.append((idx, person))
        if not active:
            return None

        centers = [box_center(person) for _, person in active]
        center_xs = [c[0] for c in centers]
        center_ys = [c[1] for c in centers]
        median_x = float(np.median(center_xs))
        median_y = float(np.median(center_ys))
        goal_direction = 1.0 if gk_center[0] >= median_x else -1.0
        progresses = [((cx - median_x) * goal_direction) for cx, _ in centers]
        progress_threshold = float(np.percentile(progresses, 60)) if progresses else 0.0

        for idx, person in active:
            cx, cy = box_center(person)
            motion = float(person.get("motion_score", 0.0))
            height = float(person["y2"] - person["y1"])
            progress = (cx - median_x) * goal_direction
            if progress < progress_threshold - 10.0:
                continue
            if abs(cy - median_y) > 140.0:
                continue
            if height < 45.0:
                continue
            teammate_dists = [
                float(np.hypot(cx - ox, cy - oy))
                for (other_idx, _), (ox, oy) in zip(active, centers)
                if other_idx != idx
            ]
            isolation = min(teammate_dists) if teammate_dists else 0.0
            score = motion * 1.8
            score += progress * 0.30
            score += min(isolation, 180.0) * 0.18
            score -= abs(cy - median_y) * 0.06
            score -= abs(cx - median_x) * 0.03
            score += min(height, 220.0) * 0.03
            scored.append((score, idx))

    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def detect_penalty_area_front_line(
    image_bgr,
    gk_box: Optional[Dict[str, float]],
    ball_box: Optional[Dict[str, float]],
    person_boxes: List[Dict[str, object]],
    goalkeeper_idx: Optional[int],
    kicker_idx: Optional[int],
    pitch_mask: np.ndarray,
    whiteline_mask: np.ndarray,
) -> Tuple[Optional[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    h, w = image_bgr.shape[:2]
    line_seed = cv2.bitwise_and(whiteline_mask, pitch_mask)
    kernel = np.ones((3, 3), np.uint8)
    line_seed = cv2.morphologyEx(line_seed, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.Canny(line_seed, 40, 120)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=90,
        maxLineGap=18,
    )
    if lines is None:
        # Some broadcast clips have a visible front-box line but weak white-line
        # segmentation; fall back to broader grayscale edges on-pitch only.
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        broad_edges = cv2.Canny(gray, 70, 180)
        broad_edges = cv2.bitwise_and(broad_edges, pitch_mask)
        broad_edges = cv2.morphologyEx(broad_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        lines = cv2.HoughLinesP(
            broad_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=28,
            minLineLength=70,
            maxLineGap=26,
        )
    if lines is None or gk_box is None:
        return None, []

    gk_center = box_center(gk_box)
    anchor_center: Optional[Tuple[float, float]] = box_center(ball_box) if ball_box is not None else None

    active_player_bottoms: List[Tuple[float, float]] = []
    for idx, person in enumerate(person_boxes):
        if idx == kicker_idx:
            continue
        if not person.get("on_pitch", False):
            continue
        if not person.get("likely_player", False):
            continue
        active_player_bottoms.append(bottom_points(person)["center_bottom"])

    if kicker_idx is not None and 0 <= kicker_idx < len(person_boxes):
        kicker_bottom = bottom_points(person_boxes[kicker_idx])["center_bottom"]
        if anchor_center is None:
            anchor_center = box_center(person_boxes[kicker_idx])
    else:
        kicker_bottom = None

    if anchor_center is None:
        return None, []
    field_direction = 1.0 if anchor_center[0] >= gk_center[0] else -1.0
    field_span_x = max(40.0, abs(anchor_center[0] - gk_center[0]))

    candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []
    all_lines: List[Tuple[int, int, int, int]] = []
    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, raw)
        line = (x1, y1, x2, y2)
        all_lines.append(line)
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 90:
            continue

        angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        if angle > 42 and angle < 138:
            continue

        xmid = (x1 + x2) / 2.0
        ymid = (y1 + y2) / 2.0
        if not (w * 0.12 <= xmid <= w * 0.95):
            continue
        if not (h * 0.32 <= ymid <= h * 0.90):
            continue
        x_progress = (xmid - gk_center[0]) * field_direction
        if x_progress <= -10.0:
            continue
        if x_progress > field_span_x + 280.0:
            continue

        gk_side = signed_line_value(gk_center, line)
        anchor_side = signed_line_value(anchor_center, line)
        if gk_side == 0 or anchor_side == 0:
            continue
        if gk_side * anchor_side <= 0:
            continue

        gk_dist = point_line_distance(gk_center, line)
        anchor_dist = point_line_distance(anchor_center, line)
        if gk_dist < 40 or gk_dist > 450:
            continue
        if anchor_dist < 5 or anchor_dist > 360:
            continue

        support = line_support_stats(line, pitch_mask, whiteline_mask)
        if support["white_avg"] < 0.018:
            continue
        if support["pitch_avg"] < 0.26:
            continue
        if min(support["pitch_pos_avg"], support["pitch_neg_avg"]) < 0.10:
            continue

        player_hits = 0
        player_distances: List[float] = []
        for px, py in active_player_bottoms:
            if px < min(x1, x2) - 90 or px > max(x1, x2) + 90:
                continue
            ly = line_y_at_x(line, px)
            if ly is None:
                continue
            vertical_gap = abs(py - ly)
            player_distances.append(vertical_gap)
            if vertical_gap <= 120.0:
                player_hits += 1

        kicker_gap = 999.0
        if kicker_bottom is not None:
            ly = line_y_at_x(line, kicker_bottom[0])
            if ly is not None:
                kicker_gap = abs(kicker_bottom[1] - ly)

        if player_hits == 0 and not player_distances:
            continue
        if player_hits == 0 and mean(player_distances) > 165.0:
            continue

        score = length
        score += max(0.0, 220.0 - abs(gk_dist - 160.0)) * 0.8
        score += max(0.0, 160.0 - abs(anchor_dist - 70.0)) * 0.6
        score += max(0.0, 120.0 - abs(xmid - ((anchor_center[0] + gk_center[0]) / 2.0))) * 0.5
        score += support["white_avg"] * 260.0
        score += min(support["pitch_pos_avg"], support["pitch_neg_avg"]) * 220.0
        score += min(player_hits, 4) * 65.0
        if player_distances:
            score += max(0.0, 140.0 - mean(player_distances)) * 1.4
        if kicker_gap < 999.0:
            score += max(0.0, 180.0 - kicker_gap) * 0.25
        candidates.append((score, line))

    if not candidates:
        relaxed_line = find_relaxed_penalty_line_candidate(
            all_lines=all_lines,
            gk_center=gk_center,
            anchor_center=anchor_center,
            active_player_bottoms=active_player_bottoms,
            kicker_bottom=kicker_bottom,
            pitch_mask=pitch_mask,
            whiteline_mask=whiteline_mask,
        )
        if relaxed_line is not None:
            return relaxed_line, all_lines

        fallback_line = fit_player_alignment_line(
            gk_center=gk_center,
            anchor_center=anchor_center,
            person_boxes=person_boxes,
            goalkeeper_idx=goalkeeper_idx,
            kicker_idx=kicker_idx,
            pitch_mask=pitch_mask,
            whiteline_mask=whiteline_mask,
        )
        if fallback_line is not None:
            return fallback_line, all_lines
        return None, all_lines
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], all_lines


def detect_kick_frame(video_path: Path, kick_model_path: Path, frame_adjust: int, start_s: float, end_s: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS for {video_path}")

    model = load_yolo_model(str(kick_model_path))
    details = detect_kick_frame_ball_motion_details(
        video_path=str(video_path),
        yolo_model=model,
        window_start=max(0, int(round(start_s * fps))),
        window_end=min(total_frames, int(round(end_s * fps))),
        min_confidence=0.3,
        velocity_prominence_threshold=2.5,
        max_tracking_jump_px=180.0,
        min_sustained_velocity=2.0,
        fallback_to_peak=True,
    )
    if details.get("kick_frame") is None:
        raise RuntimeError(f"Automatic kick detection failed: {details.get('reason')}")
    raw_frame = int(details["kick_frame"])
    details["raw_kick_frame"] = raw_frame
    details["kick_frame"] = max(0, raw_frame + int(frame_adjust))
    details["frame_adjust"] = int(frame_adjust)
    return details


def read_frame_bgr(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_idx < 0 or frame_idx >= frame_count:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def draw_overlay(
    image_bgr,
    line: Optional[Tuple[int, int, int, int]],
    people: List[Dict[str, object]],
    kicker_idx: Optional[int],
    goalkeeper_idx: Optional[int],
    candidates: List[int],
    title: str,
):
    vis = image_bgr.copy()
    if line is not None:
        x1, y1, x2, y2 = line
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)

    for idx, person in enumerate(people):
        if not person.get("display", True) and idx not in {goalkeeper_idx, kicker_idx}:
            continue
        x1, y1, x2, y2 = map(int, [person["x1"], person["y1"], person["x2"], person["y2"]])
        color = (0, 255, 255)
        label = "person"
        if idx == goalkeeper_idx:
            color = (0, 255, 0)
            label = "goalkeeper"
        elif idx == kicker_idx:
            color = (255, 255, 0)
            label = "kicker"
        elif idx in candidates:
            color = (0, 0, 255)
            label = "encroach?"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        for px, py in bottom_points(person).values():
            cv2.circle(vis, (int(round(px)), int(round(py))), 4, color, -1)
        cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    panel = vis.copy()
    cv2.rectangle(panel, (12, 12), (760, 92), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.42, vis, 0.58, 0, vis)
    cv2.putText(vis, title, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(
        vis,
        "Heuristic probe: detected persons near the penalty-area front line are marked as possible encroachment candidates.",
        (24, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (230, 230, 230),
        1,
    )
    return vis


def classify_encroachment_result(
    kick_details: Optional[Dict[str, object]],
    gk_box: Optional[Dict[str, float]],
    ball_box: Optional[Dict[str, float]],
    kicker_idx: Optional[int],
    penalty_line: Optional[Tuple[int, int, int, int]],
    encroachment_candidates: List[int],
) -> Tuple[str, str]:
    if gk_box is None:
        return "uncertain", "no_goalkeeper"
    if penalty_line is None:
        if ball_box is None:
            return "uncertain", "no_ball_and_no_penalty_area_line"
        return "uncertain", "no_penalty_area_line"
    if kicker_idx is None and ball_box is None:
        return "uncertain", "no_kicker"

    if kick_details is not None:
        conf = float(kick_details.get("confidence") or 0.0)
        method = str(kick_details.get("method") or "")
        if conf < 0.10:
            return "uncertain", "low_kick_confidence"
        if method == "velocity_peak_fallback" and conf < 0.25:
            return "uncertain", "kick_peak_fallback_low_conf"

    if encroachment_candidates:
        return "encroachment", "player_inside_penalty_area"
    return "no_encroachment", "no_inside_players_detected"


def frame_selection_score(payload: Dict[str, object], target_frame_idx: int) -> float:
    decision = str(payload.get("decision") or "")
    reason = str(payload.get("decision_reason") or "")
    score = 0.0

    if decision != "uncertain":
        score += 1000.0
    if payload.get("has_goalkeeper_box"):
        score += 120.0
    if payload.get("has_ball_box"):
        score += 160.0
    if payload.get("penalty_area_front_line") is not None:
        score += 260.0

    line_candidate_count = int(payload.get("line_candidate_count") or 0)
    score += min(line_candidate_count, 60) * 2.5

    candidate_count = int(payload.get("encroachment_candidate_count") or 0)
    score += min(candidate_count, 6) * 35.0
    score += min(int(payload.get("line_zone_player_count") or 0), 8) * 18.0

    if reason == "player_inside_penalty_area":
        score += 60.0
    elif reason == "no_inside_players_detected":
        score += 45.0
    elif reason == "no_penalty_area_line":
        score -= 120.0
    elif reason == "no_ball_and_no_penalty_area_line":
        score -= 220.0

    score -= abs(int(payload.get("frame_idx") or target_frame_idx) - int(target_frame_idx)) * 30.0
    return score


def analyze_frame(
    *,
    video_path: Path,
    frame_idx: int,
    kick_source: str,
    kick_details: Optional[Dict[str, object]],
    frames_dir: Path,
    kick_model,
    player_model,
    player_conf: float = 0.15,
    player_imgsz: int = 1280,
) -> Tuple[Dict[str, object], np.ndarray]:
    frame_path = frames_dir / f"{video_path.stem}__frame_{frame_idx:06d}.jpg"
    frame_info = extract_frame(video_path, frame_idx, frame_path)
    image_bgr = cv2.imread(str(frame_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not read extracted frame: {frame_path}")
    prev_image_bgr = read_frame_bgr(video_path, max(0, frame_idx - 1))

    pitch_mask = estimate_pitch_mask(image_bgr)
    whiteline_mask = estimate_whiteline_mask(image_bgr)
    motion_map = compute_motion_map(image_bgr, prev_image_bgr)

    kick_boxes = predict_boxes(kick_model, image_bgr, conf=0.05, imgsz=960)
    gk_box = pick_goalkeeper_box(kick_boxes)
    ball_box = pick_ball_box(kick_boxes)
    person_boxes = predict_boxes(player_model, image_bgr, conf=player_conf, classes=[0], imgsz=player_imgsz)
    person_boxes = deduplicate_person_boxes(person_boxes, iou_thresh=0.62)

    goalkeeper_idx = None
    if gk_box is not None:
        overlaps = [(idx, iou(person, gk_box)) for idx, person in enumerate(person_boxes)]
        overlaps = [item for item in overlaps if item[1] > 0.05]
        if overlaps:
            goalkeeper_idx = sorted(overlaps, key=lambda item: item[1], reverse=True)[0][0]

    for idx, person in enumerate(person_boxes):
        on_pitch, pitch_debug = is_probably_on_pitch(person, pitch_mask, whiteline_mask)
        person["on_pitch"] = on_pitch
        person["pitch_debug"] = pitch_debug
        person["motion_score"] = motion_score_for_box(motion_map, person)
        person["jersey_hsv"] = extract_jersey_hsv(image_bgr, person)
        person["likely_player"] = bool(on_pitch)
        person["display"] = on_pitch
        if ball_box is not None:
            ball_center = box_center(ball_box)
            person["ball_box_distance"] = float(distance_point_to_box(ball_center, person))
            person["ball_foot_distance"] = float(
                np.hypot(
                    bottom_points(person)["center_bottom"][0] - ball_center[0],
                    bottom_points(person)["center_bottom"][1] - ball_center[1],
                )
            )
        else:
            person["ball_box_distance"] = None
            person["ball_foot_distance"] = None
        if idx == goalkeeper_idx:
            person["on_pitch"] = True
            person["likely_player"] = True
            person["display"] = True

    kicker_idx = None
    on_pitch_indices = [
        idx for idx, person in enumerate(person_boxes)
        if person.get("on_pitch", False) and idx not in {goalkeeper_idx}
    ]
    for idx in on_pitch_indices:
        if idx == kicker_idx:
            continue
        jersey = person_boxes[idx].get("jersey_hsv")
        color_neighbors = 0
        for jdx in on_pitch_indices:
            if jdx == idx:
                continue
            other = person_boxes[jdx].get("jersey_hsv")
            if hsv_distance(jersey, other) <= 34.0:
                color_neighbors += 1
        close_to_ball = (
            ball_box is not None
            and (
                float(person_boxes[idx].get("ball_foot_distance") or 1e9) <= 120.0
                or float(person_boxes[idx].get("ball_box_distance") or 1e9) <= 28.0
            )
        )
        if color_neighbors == 0 and not close_to_ball:
            person_boxes[idx]["likely_player"] = False
            person_boxes[idx]["display"] = False

    if gk_box is None:
        inferred_idx, inferred_gk_box = infer_goalkeeper_from_people(
            image_shape=image_bgr.shape[:2],
            person_boxes=person_boxes,
            ball_box=ball_box,
        )
        if inferred_gk_box is not None:
            goalkeeper_idx = inferred_idx
            gk_box = inferred_gk_box
            person_boxes[goalkeeper_idx]["on_pitch"] = True
            person_boxes[goalkeeper_idx]["likely_player"] = True
            person_boxes[goalkeeper_idx]["display"] = True

    kicker_idx = pick_kicker_idx(
        person_boxes,
        ball_box,
        goalkeeper_idx,
        gk_box=gk_box,
        kick_details=kick_details,
    )
    if kicker_idx is not None:
        person_boxes[kicker_idx]["on_pitch"] = True
        person_boxes[kicker_idx]["likely_player"] = True
        person_boxes[kicker_idx]["display"] = True

    penalty_line, line_candidates = detect_penalty_area_front_line(
        image_bgr,
        gk_box,
        ball_box,
        person_boxes,
        goalkeeper_idx,
        kicker_idx,
        pitch_mask,
        whiteline_mask,
    )

    encroachment_candidates: List[int] = []
    candidate_debug: List[Dict[str, object]] = []
    line_zone_player_count = 0
    if penalty_line is not None and gk_box is not None and (kicker_idx is not None or ball_box is not None):
        gk_center = box_center(gk_box)
        anchor_center = box_center(ball_box) if ball_box is not None else box_center(person_boxes[kicker_idx])
        gk_sign = signed_line_value(gk_center, penalty_line)
        anchor_sign = signed_line_value(anchor_center, penalty_line)
        inside_sign = 1.0 if (gk_sign + anchor_sign) >= 0 else -1.0
        line_x_min = min(penalty_line[0], penalty_line[2])
        line_x_max = max(penalty_line[0], penalty_line[2])
        gk_x = gk_center[0]
        field_direction = 1.0 if anchor_center[0] >= gk_x else -1.0

        for idx, person in enumerate(person_boxes):
            if idx in {goalkeeper_idx, kicker_idx}:
                continue
            if not person.get("on_pitch", True):
                continue
            if not person.get("likely_player", True):
                continue
            person_points = bottom_points(person)
            center_bottom = person_points["center_bottom"]
            point_values = {
                name: signed_line_value(pt, penalty_line) * inside_sign
                for name, pt in person_points.items()
            }
            values = list(point_values.values())
            seg_dist = point_segment_distance(center_bottom, penalty_line)
            if field_direction > 0:
                x_ok = (gk_x - 40.0) <= center_bottom[0] <= (line_x_max + 120.0)
                display_zone = (
                    (gk_x - 60.0) <= center_bottom[0] <= (line_x_max + 220.0)
                    and seg_dist <= 260.0
                )
            else:
                x_ok = (line_x_min - 120.0) <= center_bottom[0] <= (gk_x + 40.0)
                display_zone = (
                    (line_x_min - 220.0) <= center_bottom[0] <= (gk_x + 60.0)
                    and seg_dist <= 260.0
                )
            near_line = seg_dist <= 125.0
            if x_ok and near_line:
                line_zone_player_count += 1
            center_value = float(point_values["center_bottom"])
            positive_count = sum(v > 6.0 for v in values)
            strong_inside = center_value > 12.0 or (center_value > 7.0 and positive_count >= 2)
            deep_inside = max(values) > 20.0 and positive_count >= 2
            near_ball = False
            if ball_box is not None:
                ball_center = box_center(ball_box)
                near_ball = float(np.hypot(center_bottom[0] - ball_center[0], center_bottom[1] - ball_center[1])) <= 58.0
            is_candidate = x_ok and near_line and (strong_inside or deep_inside) and not near_ball
            person["display"] = bool(person.get("display", False) and display_zone)
            candidate_debug.append(
                {
                    "idx": idx,
                    "inside_values": [float(v) for v in values],
                    "center_value": center_value,
                    "positive_count": int(positive_count),
                    "segment_distance_px": seg_dist,
                    "x_ok": x_ok,
                    "near_line": near_line,
                    "near_ball": near_ball,
                    "display_zone": display_zone,
                    "center_bottom": [float(center_bottom[0]), float(center_bottom[1])],
                }
            )
            if is_candidate:
                encroachment_candidates.append(idx)

    decision, decision_reason = classify_encroachment_result(
        kick_details=kick_details,
        gk_box=gk_box,
        ball_box=ball_box,
        kicker_idx=kicker_idx,
        penalty_line=penalty_line,
        encroachment_candidates=encroachment_candidates,
    )

    payload = {
        "video_path": str(video_path).replace("\\", "/"),
        "frame_idx": frame_idx,
        "timestamp_s": frame_info.get("timestamp_s"),
        "kick_source": kick_source,
        "kick_details": kick_details,
        "decision": decision,
        "decision_reason": decision_reason,
        "has_goalkeeper_box": gk_box is not None,
        "has_ball_box": ball_box is not None,
        "player_count": len(person_boxes),
        "goalkeeper_idx": goalkeeper_idx,
        "kicker_idx": kicker_idx,
        "penalty_area_front_line": penalty_line,
        "line_candidate_count": len(line_candidates),
        "line_zone_player_count": line_zone_player_count,
        "encroachment_candidate_indices": encroachment_candidates,
        "encroachment_candidate_count": len(encroachment_candidates),
        "candidate_debug": candidate_debug,
        "frame_path": str(frame_path).replace("\\", "/"),
    }
    if gk_box is not None:
        payload["goalkeeper_box"] = gk_box
    if ball_box is not None:
        payload["ball_box"] = ball_box
    payload["people"] = person_boxes
    return payload, image_bgr


def main():
    parser = argparse.ArgumentParser(description="Prototype player encroachment detection on the kick frame.")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--kick-model-path", default="models/train4_best.pt")
    parser.add_argument("--player-model-path", default="yolo26n.pt")
    parser.add_argument("--frame-idx", type=int, default=None)
    parser.add_argument("--auto-kick", action="store_true")
    parser.add_argument("--kick-window-start-s", type=float, default=0.5)
    parser.add_argument("--kick-window-end-s", type=float, default=2.5)
    parser.add_argument("--kick-frame-adjust", type=int, default=-1)
    parser.add_argument("--temporal-search-radius", type=int, default=4)
    parser.add_argument("--player-conf", type=float, default=0.15)
    parser.add_argument("--player-imgsz", type=int, default=1280)
    parser.add_argument("--out-root", default="runs/encroachment")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    out_dir = Path(args.out_root) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    kick_details = None
    frame_idx = args.frame_idx
    kick_source = "manual"
    if args.auto_kick:
        kick_details = detect_kick_frame(
            video_path=video_path,
            kick_model_path=Path(args.kick_model_path),
            frame_adjust=args.kick_frame_adjust,
            start_s=args.kick_window_start_s,
            end_s=args.kick_window_end_s,
        )
        frame_idx = int(kick_details["kick_frame"])
        kick_source = "auto_ball_motion"
    if frame_idx is None:
        raise RuntimeError("Provide --frame-idx or use --auto-kick")

    kick_model = _load_ultralytics_model(Path(args.kick_model_path))
    player_model = _load_ultralytics_model(Path(args.player_model_path))
    original_frame_idx = int(frame_idx)
    payload, image_bgr = analyze_frame(
        video_path=video_path,
        frame_idx=frame_idx,
        kick_source=kick_source,
        kick_details=kick_details,
        frames_dir=frames_dir,
        kick_model=kick_model,
        player_model=player_model,
        player_conf=args.player_conf,
        player_imgsz=args.player_imgsz,
    )

    retry_reasons = {
        "no_ball_and_no_penalty_area_line",
        "no_penalty_area_line",
        "no_kicker",
    }
    # Also trigger temporal search when no_encroachment but few players were visible in the
    # penalty-area zone ??this catches cases where the chosen frame is slightly too early
    # (e.g. Genoa-Juventus: frame 36 shows an empty penalty arc area, frame 40 shows players).
    _no_enc_sparse = (
        payload.get("decision") == "no_encroachment"
        and int(payload.get("line_zone_player_count") or 0) == 0
        and payload.get("penalty_area_front_line") is not None
    )
    temporal_candidates: List[Dict[str, object]] = []
    if (
        int(args.temporal_search_radius) > 0
        and (
            (
                payload.get("decision") == "uncertain"
                and str(payload.get("decision_reason") or "") in retry_reasons
            )
            or (
                payload.get("decision") == "encroachment"
                and int(payload.get("encroachment_candidate_count") or 0) <= 2
                and int(payload.get("line_zone_player_count") or 0) <= 3
            )
            or _no_enc_sparse
        )
    ):
        best_payload = payload
        best_image = image_bgr
        best_score = frame_selection_score(payload, original_frame_idx)
        temporal_candidates.append(
            {
                "frame_idx": int(payload["frame_idx"]),
                "decision": payload.get("decision"),
                "decision_reason": payload.get("decision_reason"),
                "score": best_score,
            }
        )
        for offset in range(1, int(args.temporal_search_radius) + 1):
            for candidate_frame_idx in (original_frame_idx - offset, original_frame_idx + offset):
                if candidate_frame_idx < 0:
                    continue
                try:
                    candidate_payload, candidate_image = analyze_frame(
                        video_path=video_path,
                        frame_idx=candidate_frame_idx,
                        kick_source=kick_source,
                    kick_details=kick_details,
                    frames_dir=frames_dir,
                    kick_model=kick_model,
                    player_model=player_model,
                    player_conf=args.player_conf,
                    player_imgsz=args.player_imgsz,
                )
                except Exception:
                    continue
                candidate_score = frame_selection_score(candidate_payload, original_frame_idx)
                temporal_candidates.append(
                    {
                        "frame_idx": int(candidate_payload["frame_idx"]),
                        "decision": candidate_payload.get("decision"),
                        "decision_reason": candidate_payload.get("decision_reason"),
                        "score": candidate_score,
                    }
                )
                if candidate_score > best_score:
                    best_payload = candidate_payload
                    best_image = candidate_image
                    best_score = candidate_score
        payload = best_payload
        image_bgr = best_image

    frame_idx = int(payload["frame_idx"])
    decision = str(payload.get("decision") or "uncertain")
    decision_reason = str(payload.get("decision_reason") or "unknown")
    gk_box = payload.get("goalkeeper_box")
    ball_box = payload.get("ball_box")
    person_boxes = payload.get("people", [])
    goalkeeper_idx = payload.get("goalkeeper_idx")
    kicker_idx = payload.get("kicker_idx")
    penalty_line = payload.get("penalty_area_front_line")
    encroachment_candidates = payload.get("encroachment_candidate_indices", [])
    payload["selected_from_frame_idx"] = original_frame_idx
    payload["temporal_refined"] = bool(frame_idx != original_frame_idx)
    payload["temporal_candidates"] = temporal_candidates

    overlay_path = out_dir / "encroachment_overlay.jpg"
    title = (
        f"{decision} | candidates={len(encroachment_candidates)} | frame={frame_idx} | kick={kick_source}"
    )
    overlay = draw_overlay(
        image_bgr,
        penalty_line,
        person_boxes,
        kicker_idx,
        goalkeeper_idx,
        encroachment_candidates,
        title,
    )
    cv2.imwrite(str(overlay_path), overlay)
    payload["overlay_path"] = str(overlay_path).replace("\\", "/")

    json_path = out_dir / "encroachment_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Encroachment probe finished.")
    print(f"Frame: {frame_idx}")
    print(f"Kick source: {kick_source}")
    print(f"Players detected: {len(person_boxes)}")
    print(f"Encroachment candidates: {len(encroachment_candidates)}")
    print(f"Saved overlay: {overlay_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
