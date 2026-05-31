from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def clip_key_from_image_name(image_name: str) -> str:
    stem = Path(image_name).stem
    if "__f" in stem:
        return stem.split("__f", 1)[0]
    return stem


def build_pose_lookup(directory: Path, source_name: str) -> pd.DataFrame:
    rows = []
    for path in sorted(list(directory.glob("*.jpg")) + list(directory.glob("*.png"))):
        rows.append(
            {
                "image_name": path.name,
                "clip_key": clip_key_from_image_name(path.name),
                f"{source_name}_path": str(path).replace("\\", "/"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["image_name", "clip_key", f"{source_name}_path"])
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Build a review sheet for the pose-estimation pilot.")
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--pose-crops-manifest", required=True)
    parser.add_argument("--pose-run-a", default="runs/pose/predict")
    parser.add_argument("--pose-run-b", default="runs/pose/predict2")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    comparison_csv = Path(args.comparison_csv)
    pose_crops_manifest = Path(args.pose_crops_manifest)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    comp = pd.read_csv(comparison_csv).copy()
    crops = pd.read_csv(pose_crops_manifest).copy()

    comp["clip_key"] = comp["image_name"].map(clip_key_from_image_name)
    crops["clip_key"] = crops["image_name"].map(clip_key_from_image_name)

    pose_a_same = build_pose_lookup(Path(args.pose_run_a), "pose_run_a_same_image")
    pose_b_same = build_pose_lookup(Path(args.pose_run_b), "pose_run_b_same_image")
    pose_a_clip = build_pose_lookup(Path(args.pose_run_a), "pose_run_a_same_clip")
    pose_b_clip = build_pose_lookup(Path(args.pose_run_b), "pose_run_b_same_clip")

    pose_a_same = pose_a_same.rename(columns={"image_name": "pose_run_a_same_image_name"})
    pose_b_same = pose_b_same.rename(columns={"image_name": "pose_run_b_same_image_name"})
    pose_a_clip = pose_a_clip.rename(columns={"image_name": "pose_run_a_same_clip_image_name"})
    pose_b_clip = pose_b_clip.rename(columns={"image_name": "pose_run_b_same_clip_image_name"})

    merged = comp.merge(
        crops[["image_name", "crop_path", "gk_conf", "crop_width", "crop_height", "clip_key"]],
        on=["image_name", "clip_key"],
        how="left",
    )
    merged = merged.merge(
        pose_a_same[["pose_run_a_same_image_name", "clip_key", "pose_run_a_same_image_path"]],
        left_on=["image_name", "clip_key"],
        right_on=["pose_run_a_same_image_name", "clip_key"],
        how="left",
    )
    merged = merged.merge(
        pose_b_same[["pose_run_b_same_image_name", "clip_key", "pose_run_b_same_image_path"]],
        left_on=["image_name", "clip_key"],
        right_on=["pose_run_b_same_image_name", "clip_key"],
        how="left",
    )
    merged = merged.merge(
        pose_a_clip[["pose_run_a_same_clip_image_name", "clip_key", "pose_run_a_same_clip_path"]],
        on="clip_key",
        how="left",
    )
    merged = merged.merge(
        pose_b_clip[["pose_run_b_same_clip_image_name", "clip_key", "pose_run_b_same_clip_path"]],
        on="clip_key",
        how="left",
    )

    def priority(row) -> str:
        comments = str(row.get("comments", "")).strip().lower()
        status = str(row.get("status", "")).strip().lower()
        if status == "mismatch":
            return "high_mismatch"
        if "occluded" in comments:
            return "medium_occlusion"
        return "normal"

    merged["pose_pilot_priority"] = merged.apply(priority, axis=1)
    merged["pose_helpful"] = ""
    merged["pose_visible_left"] = ""
    merged["pose_visible_right"] = ""
    merged["pose_chosen_keypoint"] = ""
    merged["pose_notes"] = ""

    cols_front = [
        "pose_pilot_priority",
        "image_name",
        "clip_key",
        "friend_label",
        "system_label",
        "status",
        "comments",
        "best_point",
        "min_dist_px",
        "crop_path",
        "pose_run_a_same_image_path",
        "pose_run_b_same_image_path",
        "pose_run_a_same_clip_path",
        "pose_run_b_same_clip_path",
        "pose_helpful",
        "pose_visible_left",
        "pose_visible_right",
        "pose_chosen_keypoint",
        "pose_notes",
    ]
    remaining = [c for c in merged.columns if c not in cols_front]
    merged = merged[cols_front + remaining]
    merged = merged.sort_values(["pose_pilot_priority", "image_name"], ascending=[True, True]).reset_index(drop=True)

    merged.to_csv(out_csv, index=False)
    print(f"Saved pose pilot review sheet: {out_csv}")
    print(merged["pose_pilot_priority"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
