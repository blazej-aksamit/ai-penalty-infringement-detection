from pathlib import Path
import argparse
import cv2
import numpy as np


def line_length(line):
    x1, y1, x2, y2 = line
    return float(np.hypot(x2 - x1, y2 - y1))


def line_angle_deg(line):
    x1, y1, x2, y2 = line
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def midpoint(line):
    x1, y1, x2, y2 = line
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def merge_similar_lines(lines, orientation="vertical", pos_thresh=20, angle_thresh=10):
    if not lines:
        return []

    merged = []
    used = [False] * len(lines)

    for i, a in enumerate(lines):
        if used[i]:
            continue

        group = [a]
        used[i] = True

        for j in range(i + 1, len(lines)):
            if used[j]:
                continue

            b = lines[j]
            ang_a = line_angle_deg(a)
            ang_b = line_angle_deg(b)

            if abs(ang_a - ang_b) > angle_thresh:
                continue

            if orientation == "vertical":
                xa = midpoint(a)[0]
                xb = midpoint(b)[0]
                if abs(xa - xb) > pos_thresh:
                    continue
            else:
                ya = midpoint(a)[1]
                yb = midpoint(b)[1]
                if abs(ya - yb) > pos_thresh:
                    continue

            group.append(b)
            used[j] = True

        pts = []
        for g in group:
            pts.extend([(g[0], g[1]), (g[2], g[3])])

        if orientation == "vertical":
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x = int(round(np.median(xs)))
            y1 = int(min(ys))
            y2 = int(max(ys))
            merged.append((x, y1, x, y2))
        else:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            y = int(round(np.median(ys)))
            x1 = int(min(xs))
            x2 = int(max(xs))
            merged.append((x1, y, x2, y))

    return merged


def detect_goalposts_only(img):
    """Wykrywa TYLKO słupki (vertical lines), ignoruje resztę"""
    h, w = img.shape[:2]

    # White isolation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 170], dtype=np.uint8)
    upper_white = np.array([180, 70, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mask_lab = cv2.inRange(l, 180, 255)

    mask = cv2.bitwise_and(mask_hsv, mask_lab)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(mask, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=30,
        maxLineGap=12,
    )

    if lines is None:
        return mask, edges, []

    raw_lines = [tuple(map(int, l[0])) for l in lines]

    vertical = []

    for ln in raw_lines:
        length = line_length(ln)
        if length < 30:
            continue

        angle = abs(line_angle_deg(ln))
        xm, ym = midpoint(ln)

        if ym < h * 0.12 or ym > h * 0.95:
            continue

        # ONLY vertical posts
        if 70 <= angle <= 110:
            if h * 0.20 <= ym <= h * 0.90:
                vertical.append(ln)

    vertical = merge_similar_lines(vertical, orientation="vertical", pos_thresh=18, angle_thresh=12)

    return mask, edges, vertical


def estimate_goalframe_from_posts(vertical_lines, img_shape):
    """
    Tworzy goalframe używając TYLKO słupków + FIFA geometry
    Ignoruje horizontal lines całkowicie
    """
    h, w = img_shape[:2]

    if len(vertical_lines) < 2:
        return None

    best = None
    best_score = -1

    for i in range(len(vertical_lines)):
        for j in range(i + 1, len(vertical_lines)):
            lpost = vertical_lines[i]
            rpost = vertical_lines[j]

            lx = midpoint(lpost)[0]
            rx = midpoint(rpost)[0]
            if lx > rx:
                lpost, rpost = rpost, lpost
                lx, rx = rx, lx

            goal_width_px = rx - lx
            if goal_width_px < 35 or goal_width_px > w * 0.7:
                continue

            # FIFA: 732cm width, 244cm height → ratio 3:1
            goal_height_px = goal_width_px * (244 / 732)

            # Bottom = dolna część słupków
            left_bottom_y = max(lpost[1], lpost[3])
            right_bottom_y = max(rpost[1], rpost[3])
            bottom_y = int((left_bottom_y + right_bottom_y) / 2)

            # Top = bottom - estimated height
            top_y = int(bottom_y - goal_height_px)

            # 4 corners (ESTIMATED, nie wykryte!)
            corners = [
                [int(lx), bottom_y],   # left bottom
                [int(rx), bottom_y],   # right bottom
                [int(lx), top_y],      # left top
                [int(rx), top_y]       # right top
            ]

            score = goal_width_px

            # Bonus za plausible location
            xmid = (lx + rx) / 2.0
            if w * 0.15 <= xmid <= w * 0.85:
                score += 100

            if score > best_score:
                best_score = score
                best = {
                    "left_post": lpost,
                    "right_post": rpost,
                    "corners": corners,  # ESTIMATED corners
                    "score": score,
                    "goal_width_px": goal_width_px,
                    "goal_height_px": goal_height_px,
                }

    return best


def draw_simple_probe(img, mask, edges, vertical, best_frame):
    vis = img.copy()

    # Vertical candidates (blue)
    for ln in vertical:
        x1, y1, x2, y2 = ln
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if best_frame is not None:
        lp = best_frame["left_post"]
        rp = best_frame["right_post"]
        corners = best_frame["corners"]

        # Słupki (GRUBE niebieskie)
        cv2.line(vis, (lp[0], lp[1]), (lp[2], lp[3]), (255, 0, 0), 5)
        cv2.line(vis, (rp[0], rp[1]), (rp[2], rp[3]), (255, 0, 0), 5)

        # ESTIMATED GOALFRAME (zielony prostokąt)
        cv2.line(vis, tuple(corners[0]), tuple(corners[1]), (0, 255, 0), 3)  # bottom
        cv2.line(vis, tuple(corners[2]), tuple(corners[3]), (0, 255, 0), 3)  # top
        cv2.line(vis, tuple(corners[0]), tuple(corners[2]), (0, 255, 0), 3)  # left
        cv2.line(vis, tuple(corners[1]), tuple(corners[3]), (0, 255, 0), 3)  # right

        # Corners (czerwone kółka)
        for corner in corners:
            cv2.circle(vis, tuple(corner), 8, (0, 0, 255), -1)

        cv2.putText(
            vis,
            f"ESTIMATED GOALFRAME | width={best_frame['goal_width_px']:.0f}px height={best_frame['goal_height_px']:.0f}px",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            vis,
            "NO GOALPOSTS DETECTED",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="data/line_logic_blazej_evaluation/images")
    parser.add_argument("--out-dir", default="runs/goalframe_simple")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import os
    all_files = os.listdir(image_dir)
    image_paths = sorted([
        image_dir / f for f in all_files 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    rows = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read: {img_path}")
            continue

        mask, edges, vertical = detect_goalposts_only(img)
        best_frame = estimate_goalframe_from_posts(vertical, img.shape)

        vis = draw_simple_probe(img, mask, edges, vertical, best_frame)
        cv2.imwrite(str(out_dir / img_path.name), vis)

        row = {
            "image_name": img_path.name,
            "num_vertical": len(vertical),
            "has_goalframe": best_frame is not None,
            "goal_width_px": "" if best_frame is None else float(best_frame["goal_width_px"]),
            "goal_height_px": "" if best_frame is None else float(best_frame["goal_height_px"]),
        }
        rows.append(row)

        print(
            f"{img_path.name}: "
            f"goalframe={'YES' if best_frame is not None else 'NO'} | "
            f"vertical={len(vertical)}"
        )

    csv_path = out_dir / "goalframe_simple_results.csv"
    import pandas as pd
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nSaved overlays to: {out_dir}")
    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()