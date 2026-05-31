import sys
from pathlib import Path
import argparse
import json
import subprocess
import shutil
import cv2
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.line_logic.uncertainty_policy import apply_uncertainty_policy
from src.pose.pose_refinement import (
    get_pose_guided_ground_points,
    run_pose_refinement,
)


def extract_frame(video_path: Path, frame_idx: int, out_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if frame_idx < 0 or frame_idx >= frame_count:
        cap.release()
        raise ValueError(f"frame_idx {frame_idx} out of range [0, {frame_count - 1}]")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)

    return {
        "frame_count": frame_count,
        "fps": fps,
        "frame_idx": frame_idx,
        "timestamp_s": frame_idx / fps if fps > 0 else None,
        "frame_path": str(out_path).replace("\\", "/"),
    }


def run_yolo_detect(image_path: Path, model_path: Path, project_dir: Path, conf: float):
    cli_path = shutil.which("yolo")
    if cli_path:
        save_root = project_dir.parent.resolve()
        save_dir = save_root / project_dir.name
        cmd = [
            cli_path,
            "task=detect",
            "mode=predict",
            f"model={model_path}",
            f"source={image_path}",
            "save=True",
            "save_txt=True",
            f"conf={conf}",
            f"project={save_root}",
            f"name={project_dir.name}",
            "exist_ok=True",
        ]
        print("Running YOLO detect via CLI...")
        subprocess.run(cmd, check=True)
        return save_dir

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not find the `yolo` CLI and `ultralytics` is not installed in the active Python environment."
        ) from exc

    print("Running YOLO detect via Python API...")
    model = YOLO(str(model_path))
    save_root = project_dir.parent.resolve()
    results = model.predict(
        source=str(image_path),
        conf=conf,
        save=True,
        save_txt=True,
        project=str(save_root),
        name=project_dir.name,
        exist_ok=True,
        verbose=False,
    )
    if results:
        return Path(results[0].save_dir)
    return project_dir


def auto_detect_kick_frame(video_path: Path, args, out_dir: Path):
    from src.kick_detection.ball_motion_detector import (
        detect_kick_frame_ball_motion_details,
        load_yolo_model,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for automatic kick detection: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS while reading {video_path}")

    window_start = max(0, int(round(args.kick_window_start_s * fps)))
    window_end = min(total_frames, int(round(args.kick_window_end_s * fps)))
    if window_end <= window_start:
        raise ValueError("Automatic kick-detection window is empty. Adjust --kick-window-start-s / --kick-window-end-s.")

    kick_model_path = Path(args.kick_model_path or args.model_path)
    yolo_model = load_yolo_model(str(kick_model_path))
    details = detect_kick_frame_ball_motion_details(
        video_path=str(video_path),
        yolo_model=yolo_model,
        window_start=window_start,
        window_end=window_end,
        min_confidence=args.kick_min_confidence,
        velocity_prominence_threshold=args.kick_onset_factor,
        max_tracking_jump_px=args.kick_max_tracking_jump_px,
        min_sustained_velocity=args.kick_min_sustained_velocity,
        fallback_to_peak=not args.kick_disable_peak_fallback,
    )

    details["video_path"] = str(video_path).replace("\\", "/")
    details["model_path"] = str(kick_model_path).replace("\\", "/")
    details["window_start_s"] = window_start / fps
    details["window_end_s"] = window_end / fps
    details["kick_frame_adjust"] = int(args.kick_frame_adjust)

    if details.get("kick_frame") is not None:
        raw_kick_frame = int(details["kick_frame"])
        adjusted_kick_frame = max(0, raw_kick_frame + int(args.kick_frame_adjust))
        details["raw_kick_frame"] = raw_kick_frame
        details["kick_frame"] = adjusted_kick_frame

    kick_json_path = out_dir / "kick_detection.json"
    with open(kick_json_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)

    return details, kick_json_path


def load_yolo_boxes(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists():
        return boxes

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        xc = float(parts[1]) * img_w
        yc = float(parts[2]) * img_h
        w = float(parts[3]) * img_w
        h = float(parts[4]) * img_h
        conf = float(parts[5]) if len(parts) >= 6 else 1.0

        x1 = int(round(xc - w / 2))
        y1 = int(round(yc - h / 2))
        x2 = int(round(xc + w / 2))
        y2 = int(round(yc + h / 2))

        boxes.append(
            {
                "cls": cls_id,
                "conf": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return boxes


def pick_goalkeeper(boxes, class_goalkeeper=0, conf_min=0.25):
    gk_boxes = [b for b in boxes if b["cls"] == class_goalkeeper and b["conf"] >= conf_min]
    if not gk_boxes:
        return None
    return sorted(gk_boxes, key=lambda b: b["conf"], reverse=True)[0]


def point_to_line_distance(px, py, line):
    x1, y1, x2, y2 = line
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    if den == 0:
        return 1e9
    return num / den


def line_y_at_x(line, x):
    x1, y1, x2, y2 = line
    if x2 == x1:
        return (y1 + y2) / 2.0
    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def get_bbox_foot_proxies(gk_box):
    x1, y1, x2, y2 = gk_box["x1"], gk_box["y1"], gk_box["x2"], gk_box["y2"]
    return {
        "left_bottom": (float(x1), float(y2)),
        "center_bottom": ((x1 + x2) / 2.0, float(y2)),
        "right_bottom": (float(x2), float(y2)),
    }


def split_pose_and_bbox_points(foot_points):
    pose_points = {}
    bbox_points = {}
    for name, point in (foot_points or {}).items():
        if str(name).startswith("pose_"):
            pose_points[name] = point
        else:
            bbox_points[name] = point
    return pose_points, bbox_points


def detect_goal_line_candidates(img, gk_box=None):
    LINE_MIN_LENGTH = 80
    GK_LINE_MAX_DIST_PX = 60.0
    GK_LINE_MID_Y_MAX_ABOVE = 70
    GK_LINE_X_MARGIN = 120

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=cv2.cv2.PI / 180 if hasattr(cv2, "cv2") else 3.141592653589793 / 180,
        threshold=60,
        minLineLength=LINE_MIN_LENGTH,
        maxLineGap=20,
    )

    if lines is None:
        return [], edges

    candidates = []

    gk_xmid = None
    gk_ybot = None
    if gk_box is not None:
        gk_xmid = (gk_box["x1"] + gk_box["x2"]) / 2.0
        gk_ybot = float(gk_box["y2"])

    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, raw)
        dx = x2 - x1
        dy = y2 - y1
        length = float((dx * dx + dy * dy) ** 0.5)
        if length < LINE_MIN_LENGTH:
            continue

        angle = abs(float(pd.np.degrees(pd.np.arctan2(dy, dx)))) if False else abs(
            __import__("math").degrees(__import__("math").atan2(dy, dx))
        )
        if 25 < angle < 155:
            continue

        xmid = (x1 + x2) / 2.0
        ymid = (y1 + y2) / 2.0
        line = (x1, y1, x2, y2)

        if not (h * 0.35 <= ymid <= h * 0.95):
            continue
        if not (w * 0.05 <= xmid <= w * 0.95):
            continue

        base_score = length

        if gk_xmid is not None and gk_ybot is not None:
            dist_to_gk_bottom = point_to_line_distance(gk_xmid, gk_ybot, line)
            if dist_to_gk_bottom > GK_LINE_MAX_DIST_PX:
                continue

            if ymid < (gk_ybot - GK_LINE_MID_Y_MAX_ABOVE):
                continue

            minx, maxx = min(x1, x2), max(x1, x2)
            if not (minx - GK_LINE_X_MARGIN <= gk_xmid <= maxx + GK_LINE_X_MARGIN):
                continue

            base_score += 40
            base_score += max(0, 80 - abs(xmid - gk_xmid)) * 1.0
            base_score += max(0, 60 - abs(ymid - gk_ybot)) * 0.8
            base_score += max(0, 60 - dist_to_gk_bottom) * 1.5

        candidates.append({"line": line, "base_score": base_score})

    candidates = sorted(candidates, key=lambda x: x["base_score"], reverse=True)
    return candidates[:20], edges


def _choose_best_line_and_point_for_points(candidates, gk_box, pts, point_source):
    LINE_ABSURD_DIST_PX = 120.0

    if gk_box is None:
        return None

    gk_ybot = float(gk_box["y2"])

    best = None
    best_score = -1e18

    for cand in candidates:
        line = cand["line"]
        base_score = cand["base_score"]

        per_point_dists = {}
        per_point_local_y_err = {}

        for name, (px, py) in pts.items():
            dist = point_to_line_distance(px, py, line)
            line_y = line_y_at_x(line, px)
            local_y_err = abs(line_y - gk_ybot)

            per_point_dists[name] = dist
            per_point_local_y_err[name] = local_y_err

        point_name = min(
            pts.keys(),
            key=lambda n: (per_point_local_y_err[n], per_point_dists[n])
        )

        min_dist = per_point_dists[point_name]
        local_y_err = per_point_local_y_err[point_name]

        if local_y_err > 35:
            continue

        if min_dist > LINE_ABSURD_DIST_PX:
            continue

        joint_score = base_score
        joint_score += max(0, 40 - local_y_err) * 4.0
        joint_score += max(0, 40 - min_dist) * 3.0

        if best is None or joint_score > best_score:
            best_score = joint_score
            best = {
                "line": line,
                "point_name": point_name,
                "point_source": point_source,
                "min_dist": min_dist,
                "local_y_err": local_y_err,
                "all_dists": per_point_dists,
                "joint_score": joint_score,
            }

    return best


def choose_best_line_and_point(candidates, gk_box, foot_points=None):
    if gk_box is None:
        return None

    all_points = foot_points if foot_points is not None else get_bbox_foot_proxies(gk_box)
    pose_points, bbox_points = split_pose_and_bbox_points(all_points)

    if pose_points:
        pose_choice = _choose_best_line_and_point_for_points(
            candidates,
            gk_box,
            pose_points,
            point_source="pose",
        )
        if pose_choice is not None:
            return pose_choice

    return _choose_best_line_and_point_for_points(
        candidates,
        gk_box,
        bbox_points,
        point_source="bbox",
    )


def classify_hybrid(gk_box, best_choice, line_dist_thresh_px=10.0):
    if gk_box is None:
        return {
            "decision": "uncertain",
            "reason": "no_goalkeeper",
            "min_dist": None,
            "point_name": None,
            "all_dists": {},
            "local_y_err": None,
        }

    if best_choice is None:
        return {
            "decision": "uncertain",
            "reason": "no_line",
            "min_dist": None,
            "point_name": None,
            "all_dists": {},
            "local_y_err": None,
        }

    min_dist = best_choice["min_dist"]
    decision = "on_line" if min_dist <= line_dist_thresh_px else "off_line"

    return {
        "decision": decision,
        "reason": "joint_line_point_selection",
        "min_dist": min_dist,
        "point_name": best_choice["point_name"],
        "point_source": best_choice.get("point_source"),
        "all_dists": best_choice["all_dists"],
        "proxy_spread_px": (
            max(best_choice["all_dists"].values()) - min(best_choice["all_dists"].values())
            if best_choice.get("all_dists")
            else None
        ),
        "local_y_err": best_choice["local_y_err"],
    }


def _extend_line_to_image(line, img_w, img_h):
    x1, y1, x2, y2 = map(float, line)
    if abs(x2 - x1) < 1e-6:
        x = int(round(x1))
        return (x, 0, x, img_h - 1)

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    points = []
    for x in (0.0, float(img_w - 1)):
        y = m * x + b
        if 0.0 <= y <= float(img_h - 1):
            points.append((int(round(x)), int(round(y))))
    if abs(m) > 1e-6:
        for y in (0.0, float(img_h - 1)):
            x = (y - b) / m
            if 0.0 <= x <= float(img_w - 1):
                points.append((int(round(x)), int(round(y))))

    unique_points = []
    for point in points:
        if point not in unique_points:
            unique_points.append(point)
    if len(unique_points) < 2:
        return tuple(map(int, line))
    return (*unique_points[0], *unique_points[-1])


def draw_result(img, gk_box, line, result, foot_points=None, frame_idx=None, kick_source=None):
    vis = img.copy()
    h, w = img.shape[:2]

    if line is not None:
        x1, y1, x2, y2 = _extend_line_to_image(line, w, h)
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if gk_box is not None:
        cv2.rectangle(
            vis,
            (gk_box["x1"], gk_box["y1"]),
            (gk_box["x2"], gk_box["y2"]),
            (0, 255, 0),
            2,
        )

        pts = foot_points if foot_points is not None else get_bbox_foot_proxies(gk_box)
        for name, (px, py) in pts.items():
            color = (0, 255, 255)
            radius = 4
            if str(name).startswith("pose_"):
                color = (255, 0, 255)
            if result["point_name"] == name:
                color = (0, 165, 255) if result.get("point_source") != "pose" else (255, 0, 255)
                radius = 6
            cv2.circle(vis, (int(round(px)), int(round(py))), radius, color, -1)

    panel = vis.copy()
    cv2.rectangle(panel, (14, 14), (540, 118), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.42, vis, 0.58, 0, vis)

    decision = str(result["decision"]).upper()
    decision_color = {
        "ON_LINE": (0, 220, 0),
        "OFF_LINE": (0, 0, 255),
        "UNCERTAIN": (0, 200, 255),
    }.get(decision, (255, 255, 255))

    cv2.putText(vis, decision, (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, decision_color, 3)

    metrics = []
    if result["min_dist"] is not None:
        metrics.append(f"min_dist={result['min_dist']:.1f}px")
    if result["local_y_err"] is not None:
        metrics.append(f"local_y_err={result['local_y_err']:.1f}px")
    if result.get("proxy_spread_px") is not None:
        metrics.append(f"proxy_spread={result['proxy_spread_px']:.1f}px")
    if result.get("point_name"):
        metrics.append(f"best_point={result['point_name']}")
    if result.get("point_source"):
        metrics.append(f"source={result['point_source']}")
    if frame_idx is not None:
        metrics.append(f"frame={frame_idx}")
    if kick_source:
        metrics.append(f"kick={kick_source}")

    cv2.putText(
        vis,
        " | ".join(metrics[:3]),
        (28, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        vis,
        f"reason={result['reason']}" + ("" if len(metrics) <= 3 else " | " + " | ".join(metrics[3:])),
        (28, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (220, 220, 220),
        2,
    )
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True, help="Path to full penalty video")
    parser.add_argument("--frame-idx", type=int, help="Manual frame index to test pipeline")
    parser.add_argument("--auto-kick", action="store_true", help="Automatically detect the kick frame using ball motion analysis")
    parser.add_argument("--kick-model-path", default=None, help="Optional YOLO weights for kick detection (defaults to --model-path)")
    parser.add_argument("--kick-window-start-s", type=float, default=4.0, help="Start of the automatic kick-detection window in seconds")
    parser.add_argument("--kick-window-end-s", type=float, default=12.0, help="End of the automatic kick-detection window in seconds")
    parser.add_argument("--kick-min-confidence", type=float, default=0.25, help="Minimum ball confidence for kick detection")
    parser.add_argument("--kick-onset-factor", type=float, default=2.5, help="Velocity onset factor above baseline for kick detection")
    parser.add_argument("--kick-min-sustained-velocity", type=float, default=2.0, help="Minimum sustained ball velocity in pixels/frame")
    parser.add_argument("--kick-max-tracking-jump-px", type=float, default=180.0, help="Maximum allowed frame-to-frame ball jump during tracking")
    parser.add_argument("--kick-disable-peak-fallback", action="store_true", help="Disable fallback to the peak-velocity frame when no clear motion onset is found")
    parser.add_argument("--kick-frame-adjust", type=int, default=0, help="Additive frame adjustment applied after automatic kick detection (e.g. -1 or -2)")
    parser.add_argument("--model-path", default="runs/detect/train4/weights/best.pt")
    parser.add_argument("--pose-model-path", default=None, help="Optional YOLO pose model used to refine the goalkeeper foot point")
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--pose-imgsz", type=int, default=640)
    parser.add_argument("--pose-min-keypoint-conf", type=float, default=0.35)
    parser.add_argument("--pose-leg-extension-factor", type=float, default=0.35)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--line-dist-thresh", type=float, default=10.0)
    parser.add_argument("--apply-uncertain-policy", action="store_true", help="Convert borderline line decisions into an explicit `uncertain` output")
    parser.add_argument("--uncertainty-margin-px", type=float, default=2.0, help="Distance band around the line threshold that becomes uncertain")
    parser.add_argument("--uncertainty-local-y-err-px", type=float, default=8.0, help="Local vertical geometry threshold used by the uncertainty policy")
    parser.add_argument("--uncertainty-bbox-proxy-spread-px", type=float, default=17.0, help="High spread between bbox-foot proxy distances that triggers a conservative uncertain output")
    parser.add_argument("--out-root", default="runs/pipeline")
    args = parser.parse_args()

    if args.frame_idx is None and not args.auto_kick:
        parser.error("Provide either --frame-idx or enable --auto-kick.")

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_name = video_path.stem
    out_dir = Path(args.out_root) / video_name
    frames_dir = out_dir / "frames"
    detect_dir = out_dir / "detect"
    hybrid_dir = out_dir / "hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    hybrid_dir.mkdir(parents=True, exist_ok=True)

    kick_details = None
    kick_json_path = None
    kick_source = "manual"
    frame_idx = args.frame_idx

    if args.auto_kick:
        kick_details, kick_json_path = auto_detect_kick_frame(video_path, args, out_dir)
        if kick_details["kick_frame"] is not None:
            frame_idx = int(kick_details["kick_frame"])
            kick_source = "auto_ball_motion"
        elif args.frame_idx is not None:
            kick_source = "manual_fallback_after_auto_failure"
        else:
            raise RuntimeError(
                "Automatic kick detection failed and no manual --frame-idx was provided. "
                f"Reason: {kick_details.get('reason')}"
            )

    if frame_idx is None:
        raise RuntimeError("No frame index available after kick-detection resolution.")

    frame_path = frames_dir / f"{video_name}__frame_{frame_idx:06d}.jpg"
    info = extract_frame(video_path, frame_idx, frame_path)
    info["kick_source"] = kick_source
    if kick_details is not None:
        info["kick_detection_confidence"] = kick_details.get("confidence")
        info["kick_detection_method"] = kick_details.get("method")
        info["kick_detection_reason"] = kick_details.get("reason")
        info["kick_detection_json"] = str(kick_json_path).replace("\\", "/") if kick_json_path else None

    video_info_path = out_dir / "video_info.json"
    with open(video_info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    detect_output_dir = run_yolo_detect(
        image_path=frame_path,
        model_path=Path(args.model_path),
        project_dir=detect_dir,
        conf=args.conf,
    )

    label_path = detect_output_dir / "labels" / f"{frame_path.stem}.txt"
    img = cv2.imread(str(frame_path))
    if img is None:
        raise RuntimeError(f"Could not read extracted frame image: {frame_path}")

    h, w = img.shape[:2]
    boxes = load_yolo_boxes(label_path, w, h)
    gk_box = pick_goalkeeper(boxes)
    pose_result = run_pose_refinement(
        img,
        gk_box,
        pose_model_path=args.pose_model_path,
        out_dir=out_dir / "pose",
        pose_conf=args.pose_conf,
        pose_imgsz=args.pose_imgsz,
        min_keypoint_conf=args.pose_min_keypoint_conf,
        leg_extension_factor=args.pose_leg_extension_factor,
    )
    foot_points = get_pose_guided_ground_points(gk_box, pose_result) if gk_box is not None else None

    candidates, edges = detect_goal_line_candidates(img, gk_box)
    best_choice = choose_best_line_and_point(candidates, gk_box, foot_points=foot_points)
    result = classify_hybrid(gk_box, best_choice, line_dist_thresh_px=args.line_dist_thresh)
    if args.apply_uncertain_policy:
        result = apply_uncertainty_policy(
            result,
            line_dist_thresh_px=args.line_dist_thresh,
            uncertainty_margin_px=args.uncertainty_margin_px,
            local_y_err_thresh_px=args.uncertainty_local_y_err_px,
            bbox_proxy_spread_thresh_px=args.uncertainty_bbox_proxy_spread_px,
        )

    line = None if best_choice is None else best_choice["line"]
    vis = draw_result(
        img,
        gk_box,
        line,
        result,
        foot_points=foot_points,
        frame_idx=frame_idx,
        kick_source=kick_source,
    )

    overlay_path = hybrid_dir / "final_overlay.jpg"
    cv2.imwrite(str(overlay_path), vis)

    result_dict = {
        "video_path": str(video_path).replace("\\", "/"),
        "frame_idx": frame_idx,
        "timestamp_s": info["timestamp_s"],
        "kick_source": kick_source,
        "kick_detection_confidence": None if kick_details is None else kick_details.get("confidence"),
        "kick_detection_method": None if kick_details is None else kick_details.get("method"),
        "kick_detection_reason": None if kick_details is None else kick_details.get("reason"),
        "decision": result["decision"],
        "reason": result["reason"],
        "raw_decision": result.get("raw_decision"),
        "raw_reason": result.get("raw_reason"),
        "policy_decision": result.get("policy_decision"),
        "policy_reason": result.get("policy_reason"),
        "policy_flags": result.get("policy_flags"),
        "min_dist_px": result["min_dist"],
        "local_y_err_px": result["local_y_err"],
        "has_goalkeeper": gk_box is not None,
        "has_line": line is not None,
        "best_point": result["point_name"],
        "best_point_source": result.get("point_source"),
        "proxy_spread_px": result.get("proxy_spread_px"),
        "all_dists": result.get("all_dists"),
        "pose_model_path": args.pose_model_path,
        "pose_available": pose_result.get("available"),
        "pose_reason": pose_result.get("reason"),
        "pose_point_count": pose_result.get("pose_point_count"),
        "pose_selected_points": pose_result.get("selected_pose_points"),
        "pose_decision_point": pose_result.get("decision_pose_point"),
        "pose_source_mode": pose_result.get("source_mode"),
        "pose_overlay_path": pose_result.get("overlay_path"),
        "pose_json_path": pose_result.get("json_path"),
        "frame_path": str(frame_path).replace("\\", "/"),
        "overlay_path": str(overlay_path).replace("\\", "/"),
        "label_path": str(label_path).replace("\\", "/"),
        "kick_detection_json": None if kick_json_path is None else str(kick_json_path).replace("\\", "/"),
    }

    result_json_path = out_dir / "final_result.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    result_csv_path = out_dir / "final_result.csv"
    pd.DataFrame([result_dict]).to_csv(result_csv_path, index=False)

    print("\nPipeline finished.")
    print(f"Decision: {result['decision']}")
    print(f"Reason: {result['reason']}")
    print(f"Frame: {frame_idx}")
    print(f"Timestamp (s): {info['timestamp_s']}")
    print(f"Kick source: {kick_source}")
    print(f"Saved result JSON: {result_json_path}")
    print(f"Saved result CSV:  {result_csv_path}")
    print(f"Saved overlay:     {overlay_path}")


if __name__ == "__main__":
    main()
