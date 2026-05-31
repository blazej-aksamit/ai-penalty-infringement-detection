from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def list_labels(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])


def parse_label_file(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {"boxes": 0, "class_counts": Counter(), "empty": True}

    class_counts = Counter()
    boxes = 0
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            continue
        class_counts[cls_id] += 1
        boxes += 1

    return {"boxes": boxes, "class_counts": class_counts, "empty": boxes == 0}


def summarize_split(
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    metadata_df: pd.DataFrame | None,
) -> Dict[str, object]:
    image_paths = list_images(images_dir)
    label_paths = list_labels(labels_dir)

    image_stems = {p.stem for p in image_paths}
    label_stems = {p.stem for p in label_paths}

    missing_labels = sorted(image_stems - label_stems)
    extra_labels = sorted(label_stems - image_stems)

    total_boxes = 0
    empty_label_files = 0
    class_counts = Counter()

    for path in label_paths:
        parsed = parse_label_file(path)
        total_boxes += int(parsed["boxes"])
        empty_label_files += int(parsed["empty"])
        class_counts.update(parsed["class_counts"])

    split_summary: Dict[str, object] = {
        "split": split_name,
        "image_count": len(image_paths),
        "label_count": len(label_paths),
        "missing_label_count": len(missing_labels),
        "extra_label_count": len(extra_labels),
        "empty_label_count": empty_label_files,
        "total_boxes": total_boxes,
        "class_counts": dict(sorted(class_counts.items())),
        "sample_missing_labels": missing_labels[:10],
        "sample_extra_labels": extra_labels[:10],
    }

    if metadata_df is not None and not metadata_df.empty:
        md_split = metadata_df.loc[metadata_df["split"] == split_name].copy()
        md_image_names = set(md_split["image_name"].astype(str).tolist()) if "image_name" in md_split.columns else set()
        disk_image_names = {p.name for p in image_paths}
        split_summary["metadata_rows"] = int(len(md_split))
        split_summary["metadata_clip_count"] = int(md_split["clip_name"].nunique()) if "clip_name" in md_split.columns else None
        split_summary["metadata_match_count"] = int(md_split["match_id"].nunique()) if "match_id" in md_split.columns else None
        split_summary["metadata_missing_image_count"] = len(disk_image_names - md_image_names)
        split_summary["metadata_extra_image_count"] = len(md_image_names - disk_image_names)
        split_summary["sample_metadata_missing_images"] = sorted(list(disk_image_names - md_image_names))[:10]
        split_summary["sample_metadata_extra_images"] = sorted(list(md_image_names - disk_image_names))[:10]

    return split_summary


def markdown_dataset_report(
    dataset_root: Path,
    class_names: Dict[int, str],
    split_summaries: List[Dict[str, object]],
    metadata_summary: Dict[str, object],
    train_run_summary: Dict[str, object] | None,
) -> str:
    lines = [
        "# YOLO Dataset Audit",
        "",
        f"- Dataset root: `{str(dataset_root).replace(chr(92), '/')}`",
        "",
        "## Split Summary",
        "",
        "| split | images | labels | missing labels | extra labels | empty labels | boxes | clips | matches | metadata missing | metadata extra |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in split_summaries:
        lines.append(
            f"| {row['split']} | {row['image_count']} | {row['label_count']} | {row['missing_label_count']} | "
            f"{row['extra_label_count']} | {row['empty_label_count']} | {row['total_boxes']} | "
            f"{row.get('metadata_clip_count', '')} | {row.get('metadata_match_count', '')} | "
            f"{row.get('metadata_missing_image_count', '')} | {row.get('metadata_extra_image_count', '')} |"
        )

    lines.extend(
        [
            "",
            "## Class Names",
            "",
        ]
    )
    for cls_id, cls_name in sorted(class_names.items()):
        lines.append(f"- `{cls_id}`: `{cls_name}`")

    lines.extend(
        [
            "",
            "## Annotation Counts By Split",
            "",
        ]
    )
    for row in split_summaries:
        lines.append(f"### {row['split']}")
        if row["class_counts"]:
            for cls_id, count in sorted(row["class_counts"].items()):
                cls_name = class_names.get(int(cls_id), f"class_{cls_id}")
                lines.append(f"- `{cls_name}`: `{count}`")
        else:
            lines.append("- no parsed boxes")
        if row["sample_missing_labels"]:
            lines.append(f"- sample missing labels: `{', '.join(row['sample_missing_labels'])}`")
        if row["sample_extra_labels"]:
            lines.append(f"- sample extra labels: `{', '.join(row['sample_extra_labels'])}`")
        if row.get("sample_metadata_missing_images"):
            lines.append(f"- sample images missing from metadata: `{', '.join(row['sample_metadata_missing_images'])}`")
        if row.get("sample_metadata_extra_images"):
            lines.append(f"- sample metadata-only images: `{', '.join(row['sample_metadata_extra_images'])}`")
        lines.append("")

    lines.extend(
        [
            "## Metadata Summary",
            "",
            f"- rows in `frames_metadata.csv`: `{metadata_summary['rows']}`",
            f"- unique clips: `{metadata_summary['unique_clips']}`",
            f"- unique matches: `{metadata_summary['unique_matches']}`",
        ]
    )

    if train_run_summary is not None:
        lines.extend(
            [
                "",
                "## Best YOLO Run",
                "",
                f"- run directory: `{train_run_summary['run_dir']}`",
                f"- model source: `{train_run_summary['model']}`",
                f"- epochs: `{train_run_summary['epochs']}`",
                f"- precision(B): `{train_run_summary['precision']:.3f}`",
                f"- recall(B): `{train_run_summary['recall']:.3f}`",
                f"- mAP50(B): `{train_run_summary['map50']:.3f}`",
                f"- mAP50-95(B): `{train_run_summary['map5095']:.3f}`",
            ]
        )

    return "\n".join(lines) + "\n"


def load_class_names(data_yaml: Path) -> Dict[int, str]:
    class_names: Dict[int, str] = {}
    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    in_names = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if not in_names:
            continue
        if not stripped:
            continue
        if ":" not in stripped:
            continue
        key_text, value_text = stripped.split(":", 1)
        key_text = key_text.strip()
        value_text = value_text.strip()
        if not key_text.isdigit():
            continue
        class_names[int(key_text)] = value_text
    return class_names


def load_train_run_summary(run_dir: Path) -> Dict[str, object] | None:
    results_csv = run_dir / "results.csv"
    args_yaml = run_dir / "args.yaml"
    if not results_csv.exists():
        return None

    results_df = pd.read_csv(results_csv)
    best_row = results_df.iloc[-1].to_dict()

    model_path = ""
    epochs = None
    if args_yaml.exists():
        args_text = args_yaml.read_text(encoding="utf-8").splitlines()
        for line in args_text:
            if line.startswith("model:"):
                model_path = line.split(":", 1)[1].strip()
            elif line.startswith("epochs:"):
                try:
                    epochs = int(float(line.split(":", 1)[1].strip()))
                except ValueError:
                    epochs = None

    return {
        "run_dir": str(run_dir).replace("\\", "/"),
        "model": model_path,
        "epochs": epochs,
        "precision": float(best_row["metrics/precision(B)"]),
        "recall": float(best_row["metrics/recall(B)"]),
        "map50": float(best_row["metrics/mAP50(B)"]),
        "map5095": float(best_row["metrics/mAP50-95(B)"]),
    }


def build_disk_vs_metadata_diff(dataset_root: Path, metadata_df: pd.DataFrame | None) -> pd.DataFrame:
    disk_rows = []
    for split_name in ["train", "val", "test"]:
        for path in list_images(dataset_root / "images" / split_name):
            disk_rows.append({"image_name": path.name, "disk_split": split_name})

    disk_df = pd.DataFrame(disk_rows)
    if metadata_df is None or metadata_df.empty:
        disk_df["metadata_split"] = None
        disk_df["status"] = "missing_metadata"
        return disk_df

    md_df = metadata_df.copy()
    md_df = md_df[["image_name", "split"]].rename(columns={"split": "metadata_split"})
    merged = disk_df.merge(md_df, on="image_name", how="outer")

    def classify(row) -> str:
        disk_split = row.get("disk_split")
        metadata_split = row.get("metadata_split")
        if pd.isna(disk_split):
            return "metadata_only"
        if pd.isna(metadata_split):
            return "missing_metadata"
        if str(disk_split) != str(metadata_split):
            return "split_mismatch"
        return "match"

    merged["status"] = merged.apply(classify, axis=1)
    return merged.sort_values(["status", "image_name"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Audit YOLO dataset splits and labels.")
    parser.add_argument("--dataset-root", default="data/yolo_gk_ball")
    parser.add_argument("--metadata-csv", default="data/yolo_gk_ball/meta/frames_metadata.csv")
    parser.add_argument("--train-run-dir", default="runs/detect/train4")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    metadata_csv = Path(args.metadata_csv)
    train_run_dir = Path(args.train_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml: {data_yaml}")

    class_names = load_class_names(data_yaml)
    metadata_df = pd.read_csv(metadata_csv) if metadata_csv.exists() else None

    split_summaries = []
    for split_name in ["train", "val", "test"]:
        split_summaries.append(
            summarize_split(
                split_name=split_name,
                images_dir=dataset_root / "images" / split_name,
                labels_dir=dataset_root / "labels" / split_name,
                metadata_df=metadata_df,
            )
        )

    metadata_summary = {
        "rows": int(len(metadata_df)) if metadata_df is not None else 0,
        "unique_clips": int(metadata_df["clip_name"].nunique()) if metadata_df is not None and "clip_name" in metadata_df.columns else 0,
        "unique_matches": int(metadata_df["match_id"].nunique()) if metadata_df is not None and "match_id" in metadata_df.columns else 0,
    }

    split_df = pd.DataFrame(
        [
            {
                "split": row["split"],
                "image_count": row["image_count"],
                "label_count": row["label_count"],
                "missing_label_count": row["missing_label_count"],
                "extra_label_count": row["extra_label_count"],
                "empty_label_count": row["empty_label_count"],
                "total_boxes": row["total_boxes"],
                "metadata_rows": row.get("metadata_rows"),
                "metadata_clip_count": row.get("metadata_clip_count"),
                "metadata_match_count": row.get("metadata_match_count"),
                "metadata_missing_image_count": row.get("metadata_missing_image_count"),
                "metadata_extra_image_count": row.get("metadata_extra_image_count"),
            }
            for row in split_summaries
        ]
    )
    split_df.to_csv(out_dir / "split_summary.csv", index=False)

    disk_vs_metadata_df = build_disk_vs_metadata_diff(dataset_root, metadata_df)
    disk_vs_metadata_df.to_csv(out_dir / "disk_vs_metadata_diff.csv", index=False)

    annotation_rows = []
    for row in split_summaries:
        for cls_id, count in sorted(row["class_counts"].items()):
            annotation_rows.append(
                {
                    "split": row["split"],
                    "class_id": cls_id,
                    "class_name": class_names.get(int(cls_id), f"class_{cls_id}"),
                    "box_count": count,
                }
            )
    pd.DataFrame(annotation_rows).to_csv(out_dir / "annotation_counts.csv", index=False)

    train_run_summary = load_train_run_summary(train_run_dir)

    payload = {
        "dataset_root": str(dataset_root).replace("\\", "/"),
        "class_names": class_names,
        "metadata_summary": metadata_summary,
        "split_summaries": split_summaries,
        "train_run_summary": train_run_summary,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    report_md = markdown_dataset_report(
        dataset_root=dataset_root,
        class_names=class_names,
        split_summaries=split_summaries,
        metadata_summary=metadata_summary,
        train_run_summary=train_run_summary,
    )
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    print(f"Saved YOLO dataset audit to: {out_dir}")
    print(split_df.to_string(index=False))


if __name__ == "__main__":
    main()
