from pathlib import Path
import cv2
import numpy as np

# ---- paths ----
IMAGE_DIR = Path("data/line_logic_dev/images")
PREDICT_DIR = Path("runs/detect/predict")
LABELS_DIR = PREDICT_DIR / "labels"
OUT_DIR = Path("runs/line_logic_dev")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# class ids from your YOLO dataset
CLASS_GOALKEEPER = 0
CLASS_BALL = 1

# simple thresholds
GK_CONF_MIN = 0.25
LINE_MIN_LENGTH = 80
DIST_THRESH_PX = 8  # tolerance around line


def load_yolo_boxes(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists():
        return boxes

    for line in label_path.read_text().strip().splitlines():
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
    # highest confidence goalkeeper
    return sorted(gk_boxes, key=lambda b: b["conf"], reverse=True)[0]


def detect_goal_line(img):
    """
    Very simple v1 line detector:
    - grayscale
    - blur
    - canny
    - Hough lines
    - choose the longest near-horizontal line in lower-middle image
    """
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
        maxLineGap=15,
    )

    if lines is None:
        return None, edges

    candidates = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, line)
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < LINE_MIN_LENGTH:
            continue

        angle = abs(np.degrees(np.arctan2(dy, dx)))
        # allow mild slope due to perspective, but avoid near-vertical lines
        if angle > 25 and angle < 155:
            continue

        y_mid = (y1 + y2) / 2
        x_mid = (x1 + x2) / 2

        # prioritize lines in the lower-middle region
        score = length
        if h * 0.35 <= y_mid <= h * 0.95:
            score += 50
        if w * 0.15 <= x_mid <= w * 0.85:
            score += 20

        candidates.append((score, (x1, y1, x2, y2)))

    if not candidates:
        return None, edges

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1], edges


def point_to_line_distance(px, py, line):
    x1, y1, x2, y2 = line
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = np.hypot(y2 - y1, x2 - x1)
    if den == 0:
        return 1e9
    return num / den


def classify_position(gk_box, line):
    if gk_box is None or line is None:
        return "uncertain", None

    # goalkeeper bottom center as first proxy for feet
    px = (gk_box["x1"] + gk_box["x2"]) / 2
    py = gk_box["y2"]

    dist = point_to_line_distance(px, py, line)

    if dist <= DIST_THRESH_PX:
        return "on_line", dist
    else:
        return "off_line", dist


def draw_result(img, gk_box, line, label, dist):
    vis = img.copy()

    if gk_box is not None:
        cv2.rectangle(vis, (gk_box["x1"], gk_box["y1"]), (gk_box["x2"], gk_box["y2"]), (0, 255, 0), 2)
        px = int(round((gk_box["x1"] + gk_box["x2"]) / 2))
        py = int(round(gk_box["y2"]))
        cv2.circle(vis, (px, py), 5, (0, 255, 255), -1)

    if line is not None:
        x1, y1, x2, y2 = line
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    text = f"{label}"
    if dist is not None:
        text += f" | dist={dist:.1f}px"

    cv2.putText(vis, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return vis


def main():
    image_paths = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")))
    if not image_paths:
        print("No images found in", IMAGE_DIR)
        return

    results = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print("Could not read:", img_path)
            continue

        h, w = img.shape[:2]
        label_path = LABELS_DIR / f"{img_path.stem}.txt"

        boxes = load_yolo_boxes(label_path, w, h)
        gk_box = pick_goalkeeper(boxes)
        line, edges = detect_goal_line(img)
        decision, dist = classify_position(gk_box, line)

        vis = draw_result(img, gk_box, line, decision, dist)
        out_img = OUT_DIR / img_path.name
        cv2.imwrite(str(out_img), vis)

        results.append(
            {
                "image_name": img_path.name,
                "decision": decision,
                "distance_px": "" if dist is None else float(dist),
                "has_goalkeeper": gk_box is not None,
                "has_line": line is not None,
            }
        )

        print(f"{img_path.name}: {decision} | line={'yes' if line is not None else 'no'} | gk={'yes' if gk_box is not None else 'no'}")

    import pandas as pd
    pd.DataFrame(results).to_csv(OUT_DIR / "line_decision_results.csv", index=False)
    print(f"\nSaved overlays to: {OUT_DIR}")
    print(f"Saved CSV to: {OUT_DIR / 'line_decision_results.csv'}")


if __name__ == "__main__":
    main()