from pathlib import Path
import cv2

IMAGE_DIR = Path("data/line_logic_dev/images")
LABELS_DIR = Path("runs/detect/predict/labels")
OUT_DIR = Path("data/pose_dev/crops")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_GOALKEEPER = 0
GK_CONF_MIN = 0.25
PAD_RATIO_X = 0.20
PAD_RATIO_Y_TOP = 0.15
PAD_RATIO_Y_BOTTOM = 0.20


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


def pad_box(box, img_w, img_h):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(round(bw * PAD_RATIO_X))
    pad_y_top = int(round(bh * PAD_RATIO_Y_TOP))
    pad_y_bottom = int(round(bh * PAD_RATIO_Y_BOTTOM))

    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y_top)
    nx2 = min(img_w - 1, x2 + pad_x)
    ny2 = min(img_h - 1, y2 + pad_y_bottom)

    return nx1, ny1, nx2, ny2


def main():
    image_paths = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")))
    if not image_paths:
        print(f"No images found in {IMAGE_DIR}")
        return

    saved = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read: {img_path}")
            continue

        h, w = img.shape[:2]
        label_path = LABELS_DIR / f"{img_path.stem}.txt"

        boxes = load_yolo_boxes(label_path, w, h)
        gk = pick_goalkeeper(boxes)

        if gk is None:
            print(f"No goalkeeper found: {img_path.name}")
            continue

        x1, y1, x2, y2 = pad_box(gk, w, h)
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            print(f"Empty crop: {img_path.name}")
            continue

        out_path = OUT_DIR / img_path.name
        cv2.imwrite(str(out_path), crop)
        saved += 1

        print(f"Saved crop: {out_path.name} | conf={gk['conf']:.3f} | box=({x1},{y1},{x2},{y2})")

    print(f"\nDone. Saved {saved} crops to {OUT_DIR}")


if __name__ == "__main__":
    main()