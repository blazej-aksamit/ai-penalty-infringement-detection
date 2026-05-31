from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_label_stats(label_path: Path) -> Dict[str, int]:
    if not label_path.exists():
        return {
            "label_exists": 0,
            "box_count": 0,
            "goalkeeper_boxes": 0,
            "ball_boxes": 0,
        }

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return {
            "label_exists": 1,
            "box_count": 0,
            "goalkeeper_boxes": 0,
            "ball_boxes": 0,
        }

    box_count = 0
    goalkeeper_boxes = 0
    ball_boxes = 0
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            continue
        box_count += 1
        if cls_id == 0:
            goalkeeper_boxes += 1
        elif cls_id == 1:
            ball_boxes += 1

    return {
        "label_exists": 1,
        "box_count": box_count,
        "goalkeeper_boxes": goalkeeper_boxes,
        "ball_boxes": ball_boxes,
    }


def main():
    parser = argparse.ArgumentParser(description="Build a canonical YOLO disk index without modifying original data.")
    parser.add_argument("--dataset-root", default="data/yolo_gk_ball")
    parser.add_argument("--metadata-csv", default="data/yolo_gk_ball/meta/frames_metadata.csv")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    metadata_csv = Path(args.metadata_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_csv(metadata_csv).copy() if metadata_csv.exists() else pd.DataFrame()
    metadata_lookup = {}
    if not metadata_df.empty and "image_name" in metadata_df.columns:
        metadata_lookup = metadata_df.set_index("image_name").to_dict("index")

    rows: List[Dict[str, object]] = []
    for split_name in ["train", "val", "test"]:
        image_dir = dataset_root / "images" / split_name
        label_dir = dataset_root / "labels" / split_name

        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            label_stats = parse_label_stats(label_path)
            md_row = metadata_lookup.get(image_path.name, {})

            rows.append(
                {
                    "image_name": image_path.name,
                    "disk_split": split_name,
                    "image_path": str(image_path).replace("\\", "/"),
                    "label_path": str(label_path).replace("\\", "/"),
                    **label_stats,
                    "metadata_split": md_row.get("split"),
                    "metadata_image_path": md_row.get("image_path"),
                    "clip_name": md_row.get("clip_name"),
                    "clip_path": md_row.get("clip_path"),
                    "match_id": md_row.get("match_id"),
                    "violation": md_row.get("violation"),
                    "frame_idx": md_row.get("frame_idx"),
                    "frame_count": md_row.get("frame_count"),
                    "metadata_status": (
                        "missing_metadata"
                        if not md_row
                        else ("match" if md_row.get("split") == split_name else "split_mismatch")
                    ),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved canonical YOLO disk index to: {out_csv}")
    print(df[['disk_split', 'label_exists', 'metadata_status']].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
