from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image


CLASS_GOALKEEPER = 0


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


def pick_goalkeeper(boxes, conf_min: float):
    gk_boxes = [b for b in boxes if b["cls"] == CLASS_GOALKEEPER and b["conf"] >= conf_min]
    if not gk_boxes:
        return None
    return sorted(gk_boxes, key=lambda b: b["conf"], reverse=True)[0]


def pad_box(box, img_w: int, img_h: int, pad_ratio_x: float, pad_ratio_y_top: float, pad_ratio_y_bottom: float):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(round(bw * pad_ratio_x))
    pad_y_top = int(round(bh * pad_ratio_y_top))
    pad_y_bottom = int(round(bh * pad_ratio_y_bottom))

    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y_top)
    nx2 = min(img_w - 1, x2 + pad_x)
    ny2 = min(img_h - 1, y2 + pad_y_bottom)

    return nx1, ny1, nx2, ny2


def upscale_crop(crop: Image.Image, min_short_side: int, min_long_side: int):
    width, height = crop.size
    if width <= 0 or height <= 0:
        return crop

    short_side = min(width, height)
    long_side = max(width, height)

    scale_factors = [1.0]
    if min_short_side > 0 and short_side < min_short_side:
        scale_factors.append(float(min_short_side) / float(short_side))
    if min_long_side > 0 and long_side < min_long_side:
        scale_factors.append(float(min_long_side) / float(long_side))

    scale = max(scale_factors)
    if scale <= 1.0:
        return crop

    new_width = max(1, int(math.ceil(width * scale)))
    new_height = max(1, int(math.ceil(height * scale)))
    return crop.resize((new_width, new_height), Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser(description="Prepare goalkeeper crops for a pose-estimation pilot.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--gk-conf-min", type=float, default=0.25)
    parser.add_argument("--pad-ratio-x", type=float, default=0.20)
    parser.add_argument("--pad-ratio-y-top", type=float, default=0.15)
    parser.add_argument("--pad-ratio-y-bottom", type=float, default=0.20)
    parser.add_argument("--min-short-side", type=int, default=0, help="Upscale crops so their shorter side is at least this many pixels")
    parser.add_argument("--min-long-side", type=int, default=0, help="Upscale crops so their longer side is at least this many pixels")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    manifest_csv = Path(args.manifest_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    if not image_paths:
        raise SystemExit(f"No images found in {image_dir}")

    rows = []
    saved = 0
    skipped = 0

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            rows.append({"image_name": img_path.name, "status": "image_read_failed"})
            skipped += 1
            continue

        w, h = img.size
        label_path = labels_dir / f"{img_path.stem}.txt"

        boxes = load_yolo_boxes(label_path, w, h)
        gk = pick_goalkeeper(boxes, conf_min=args.gk_conf_min)
        if gk is None:
            rows.append(
                {
                    "image_name": img_path.name,
                    "label_path": str(label_path).replace("\\", "/"),
                    "status": "no_goalkeeper",
                }
            )
            skipped += 1
            continue

        x1, y1, x2, y2 = pad_box(
            gk,
            w,
            h,
            pad_ratio_x=args.pad_ratio_x,
            pad_ratio_y_top=args.pad_ratio_y_top,
            pad_ratio_y_bottom=args.pad_ratio_y_bottom,
        )
        crop = img.crop((x1, y1, x2, y2))
        if crop.size[0] <= 0 or crop.size[1] <= 0:
            rows.append(
                {
                    "image_name": img_path.name,
                    "label_path": str(label_path).replace("\\", "/"),
                    "status": "empty_crop",
                }
            )
            skipped += 1
            continue

        original_width, original_height = crop.size
        crop = upscale_crop(
            crop,
            min_short_side=args.min_short_side,
            min_long_side=args.min_long_side,
        )

        out_path = out_dir / img_path.name
        crop.save(out_path)
        saved += 1

        rows.append(
            {
                "image_name": img_path.name,
                "label_path": str(label_path).replace("\\", "/"),
                "crop_path": str(out_path).replace("\\", "/"),
                "status": "saved",
                "gk_conf": gk["conf"],
                "crop_x1": x1,
                "crop_y1": y1,
                "crop_x2": x2,
                "crop_y2": y2,
                "crop_width": x2 - x1,
                "crop_height": y2 - y1,
                "original_crop_width": original_width,
                "original_crop_height": original_height,
                "saved_crop_width": crop.size[0],
                "saved_crop_height": crop.size[1],
            }
        )

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {saved} crops to {out_dir}")
    print(f"Skipped {skipped} images")
    print(f"Manifest: {manifest_csv}")


if __name__ == "__main__":
    main()
