from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_class_names(data_yaml: Path) -> List[str]:
    names: List[str] = []
    in_names = False
    mapping = {}
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if not in_names or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key.isdigit():
            mapping[int(key)] = value
    for key in sorted(mapping):
        names.append(mapping[key])
    return names


def main():
    parser = argparse.ArgumentParser(description="Prepare a clean YOLO evaluation subset from aligned image-label pairs.")
    parser.add_argument("--dataset-root", default="data/yolo_gk_ball")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    split = args.split
    out_root = Path(args.out_root)

    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    out_images_dir = out_root / "images" / "val"
    out_labels_dir = out_root / "labels" / "val"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    label_paths = sorted([p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])

    image_map = {p.stem: p for p in image_paths}
    label_map = {p.stem: p for p in label_paths}

    matched_stems = sorted(image_map.keys() & label_map.keys())
    missing_labels = sorted(image_map.keys() - label_map.keys())
    extra_labels = sorted(label_map.keys() - image_map.keys())

    manifest_rows = []
    for stem in matched_stems:
        src_img = image_map[stem]
        src_lab = label_map[stem]
        dst_img = out_images_dir / src_img.name
        dst_lab = out_labels_dir / src_lab.name
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lab, dst_lab)
        manifest_rows.append(
            {
                "stem": stem,
                "image_name": src_img.name,
                "label_name": src_lab.name,
                "src_image_path": str(src_img).replace("\\", "/"),
                "src_label_path": str(src_lab).replace("\\", "/"),
                "dst_image_path": str(dst_img).replace("\\", "/"),
                "dst_label_path": str(dst_lab).replace("\\", "/"),
            }
        )

    names = load_class_names(dataset_root / "data.yaml")
    yaml_lines = [
        f"path: {str(out_root).replace(chr(92), '/')}",
        "train: images/val",
        "val: images/val",
        "test: images/val",
        "",
        "names:",
    ]
    for idx, name in enumerate(names):
        yaml_lines.append(f"  {idx}: {name}")
    (out_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    pd.DataFrame(manifest_rows).to_csv(out_root / "matched_pairs_manifest.csv", index=False)
    pd.DataFrame({"missing_label_stem": missing_labels}).to_csv(out_root / "missing_labels.csv", index=False)
    pd.DataFrame({"extra_label_stem": extra_labels}).to_csv(out_root / "extra_labels.csv", index=False)

    summary = {
        "dataset_root": str(dataset_root).replace("\\", "/"),
        "split": split,
        "matched_pairs": len(matched_stems),
        "missing_labels": len(missing_labels),
        "extra_labels": len(extra_labels),
        "out_root": str(out_root).replace("\\", "/"),
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Prepared YOLO eval subset at: {out_root}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
