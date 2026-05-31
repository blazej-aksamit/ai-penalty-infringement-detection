from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd


CLASS_GOALKEEPER = 0
GK_CONF_MIN = 0.25

LINE_MIN_LENGTH = 80
LINE_DIST_THRESH_PX = 10.0
LINE_ABSURD_DIST_PX = 120.0

GK_LINE_MAX_DIST_PX = 60.0
GK_LINE_MID_Y_MAX_ABOVE = 70
GK_LINE_X_MARGIN = 120


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


def pick_goalkeeper(boxes):
    gk_boxes = [b for b in boxes if b["cls"] == CLASS_GOALKEEPER and b["conf"] >= GK_CONF_MIN]
    if not gk_boxes:
        return None
    return sorted(gk_boxes, key=lambda b: b["conf"], reverse=True)[0]


def point_to_line_distance(px, py, line):
    x1, y1, x2, y2 = line
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = np.hypot(y2 - y1, x2 - x1)
    if den == 0:
        return 1e9
    return num / den


def line_y_at_x(line, x):
    x1, y1, x2, y2 = line

    if x2 == x1:
        return (y1 + y2) / 2.0

    t = (x - x1) / (x2 - x1)
    y = y1 + t * (y2 - y1)
    return float(y)


def get_bbox_foot_proxies(gk_box):
    x1, y1, x2, y2 = gk_box["x1"], gk_box["y1"], gk_box["x2"], gk_box["y2"]
    return {
        "left_bottom": (float(x1), float(y2)),
        "center_bottom": ((x1 + x2) / 2.0, float(y2)),
        "right_bottom": (float(x2), float(y2)),
    }


def detect_goal_line_candidates(img, gk_box=None):
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
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
        length = float(np.hypot(dx, dy))
        if length < LINE_MIN_LENGTH:
            continue

        angle = abs(np.degrees(np.arctan2(dy, dx)))
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

        candidates.append(
            {
                "line": line,
                "base_score": base_score,
            }
        )

    candidates = sorted(candidates, key=lambda x: x["base_score"], reverse=True)
    return candidates[:20], edges


def choose_best_line_and_point(candidates, gk_box):
    if gk_box is None:
        return None

    pts = get_bbox_foot_proxies(gk_box)
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

        # Joint score: prefer line-point pair that is locally good under keeper
        joint_score = base_score
        joint_score += max(0, 40 - local_y_err) * 4.0
        joint_score += max(0, 40 - min_dist) * 3.0

        if best is None or joint_score > best_score:
            best_score = joint_score
            best = {
                "line": line,
                "point_name": point_name,
                "min_dist": min_dist,
                "local_y_err": local_y_err,
                "all_dists": per_point_dists,
                "joint_score": joint_score,
            }

    return best


def classify_hybrid(gk_box, best_choice):
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

    if min_dist <= LINE_DIST_THRESH_PX:
        decision = "on_line"
    else:
        decision = "off_line"

    return {
        "decision": decision,
        "reason": "joint_line_point_selection",
        "min_dist": min_dist,
        "point_name": best_choice["point_name"],
        "all_dists": best_choice["all_dists"],
        "local_y_err": best_choice["local_y_err"],
    }


def draw_result(img, gk_box, line, result):
    vis = img.copy()

    if line is not None:
        x1, y1, x2, y2 = line
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if gk_box is not None:
        cv2.rectangle(
            vis,
            (gk_box["x1"], gk_box["y1"]),
            (gk_box["x2"], gk_box["y2"]),
            (0, 255, 0),
            2,
        )

        pts = get_bbox_foot_proxies(gk_box)
        for name, (px, py) in pts.items():
            color = (0, 255, 255)
            radius = 4
            if result["point_name"] == name:
                color = (0, 165, 255)
                radius = 6
            cv2.circle(vis, (int(round(px)), int(round(py))), radius, color, -1)

    txt = result["decision"]
    if result["min_dist"] is not None:
        txt += f" | min_dist={result['min_dist']:.1f}px"
    if result["local_y_err"] is not None:
        txt += f" | local_y_err={result['local_y_err']:.1f}"
    txt += f" | via={result['reason']}"

    cv2.putText(vis, txt, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="data/line_logic_blazej_evaluation/images")
    parser.add_argument("--detect-labels-dir", default="runs/detect/predict3/labels")
    parser.add_argument("--out-dir", default="runs/hybrid_line_logic_blazej_joint")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    detect_labels_dir = Path(args.detect_labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    rows = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read: {img_path}")
            continue

        h, w = img.shape[:2]
        label_path = detect_labels_dir / f"{img_path.stem}.txt"

        boxes = load_yolo_boxes(label_path, w, h)
        gk_box = pick_goalkeeper(boxes)

        candidates, edges = detect_goal_line_candidates(img, gk_box)
        best_choice = choose_best_line_and_point(candidates, gk_box)
        result = classify_hybrid(gk_box, best_choice)

        line = None if best_choice is None else best_choice["line"]
        vis = draw_result(img, gk_box, line, result)
        cv2.imwrite(str(out_dir / img_path.name), vis)

        rows.append(
            {
                "image_name": img_path.name,
                "decision": result["decision"],
                "reason": result["reason"],
                "min_dist_px": "" if result["min_dist"] is None else float(result["min_dist"]),
                "local_y_err_px": "" if result["local_y_err"] is None else float(result["local_y_err"]),
                "has_goalkeeper": gk_box is not None,
                "has_line": line is not None,
                "best_point": result["point_name"],
                "left_bottom_dist": result["all_dists"].get("left_bottom", ""),
                "center_bottom_dist": result["all_dists"].get("center_bottom", ""),
                "right_bottom_dist": result["all_dists"].get("right_bottom", ""),
            }
        )

        print(
            f"{img_path.name}: {result['decision']} | "
            f"reason={result['reason']} | "
            f"min_dist={result['min_dist']} | "
            f"local_y_err={result['local_y_err']}"
        )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "hybrid_line_decision_results.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nSaved overlays to: {out_dir}")
    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()