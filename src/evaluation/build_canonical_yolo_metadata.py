from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


FRAME_SUFFIX_RE = re.compile(r"__f(\d+)$")


def derive_clip_name(image_name: str) -> str:
    stem = Path(image_name).stem
    if "__f" in stem:
        return stem.split("__f", 1)[0] + ".mp4"
    return stem + ".mp4"


def derive_match_id(clip_name: str) -> str:
    clip_stem = Path(clip_name).stem
    for token in ["_H1_", "_H2_"]:
        if token in clip_stem:
            return clip_stem.split(token, 1)[0]
    return clip_stem


def derive_frame_idx(image_name: str) -> Optional[int]:
    stem = Path(image_name).stem
    match = FRAME_SUFFIX_RE.search(stem)
    if not match:
        return None
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser(description="Build canonical YOLO metadata from disk state without modifying original files.")
    parser.add_argument("--canonical-index-csv", required=True)
    parser.add_argument("--violation-labels-csv", default="data/meta/keeper_violation_labels_final.csv")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--missing-labels-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    canonical_index_csv = Path(args.canonical_index_csv)
    violation_labels_csv = Path(args.violation_labels_csv)
    out_csv = Path(args.out_csv)
    missing_labels_csv = Path(args.missing_labels_csv)
    summary_json = Path(args.summary_json)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    missing_labels_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(canonical_index_csv).copy()
    labels = pd.read_csv(violation_labels_csv).copy()

    labels_lookup: Dict[str, Dict[str, object]] = {}
    if "clip_name" in labels.columns:
        labels_lookup = labels.set_index("clip_name").to_dict("index")

    rows = []
    for row in idx.to_dict("records"):
        image_name = str(row["image_name"])
        clip_name = row.get("clip_name")
        if pd.isna(clip_name) or not str(clip_name).strip():
            clip_name = derive_clip_name(image_name)
        clip_name = str(clip_name)

        label_row = labels_lookup.get(clip_name, {})
        clip_path = row.get("clip_path")
        if pd.isna(clip_path) or not str(clip_path).strip():
            clip_path = label_row.get("window_file") or f"data/clips/kick_windows_720p_v2/{clip_name}"

        violation = row.get("violation")
        if pd.isna(violation):
            violation = label_row.get("violation")

        frame_count = row.get("frame_count")
        if pd.isna(frame_count):
            frame_count = label_row.get("total_frames")

        canonical_row = {
            "image_name": image_name,
            "image_path": row["image_path"],
            "clip_name": clip_name,
            "clip_path": str(clip_path).replace("\\", "/"),
            "match_id": row.get("match_id") if pd.notna(row.get("match_id")) else derive_match_id(clip_name),
            "split": row["disk_split"],
            "violation": violation,
            "frame_idx": row.get("frame_idx") if pd.notna(row.get("frame_idx")) else derive_frame_idx(image_name),
            "frame_count": frame_count,
            "label_path": row["label_path"],
            "label_exists": int(row["label_exists"]),
            "box_count": int(row["box_count"]),
            "goalkeeper_boxes": int(row["goalkeeper_boxes"]),
            "ball_boxes": int(row["ball_boxes"]),
            "metadata_status": row["metadata_status"],
            "original_metadata_split": row.get("metadata_split"),
        }
        rows.append(canonical_row)

    canonical_df = pd.DataFrame(rows).sort_values(["split", "clip_name", "frame_idx", "image_name"]).reset_index(drop=True)
    canonical_df.to_csv(out_csv, index=False)

    missing_labels_df = canonical_df.loc[canonical_df["label_exists"] == 0].copy()
    missing_labels_df.to_csv(missing_labels_csv, index=False)

    summary = {
        "canonical_index_csv": str(canonical_index_csv).replace("\\", "/"),
        "violation_labels_csv": str(violation_labels_csv).replace("\\", "/"),
        "output_csv": str(out_csv).replace("\\", "/"),
        "missing_labels_csv": str(missing_labels_csv).replace("\\", "/"),
        "rows_total": int(len(canonical_df)),
        "rows_with_labels": int((canonical_df["label_exists"] == 1).sum()),
        "rows_missing_labels": int((canonical_df["label_exists"] == 0).sum()),
        "metadata_status_counts": canonical_df["metadata_status"].value_counts(dropna=False).to_dict(),
        "split_counts": canonical_df["split"].value_counts(dropna=False).to_dict(),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved canonical YOLO metadata to: {out_csv}")
    print(f"Saved missing-label manifest to: {missing_labels_csv}")
    print(pd.DataFrame([summary]).T.to_string(header=False))


if __name__ == "__main__":
    main()
