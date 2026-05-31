from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run YOLO detection evaluation and save a small summary.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/evaluation")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ultralytics is not installed in the active Python environment. "
            "Run this script from the environment used for YOLO work."
        ) from exc

    project_dir = Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model_path)
    metrics = model.val(
        data=args.data_yaml,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        plots=True,
        save_json=False,
    )

    summary = {
        "model_path": str(Path(args.model_path)).replace("\\", "/"),
        "data_yaml": str(Path(args.data_yaml)).replace("\\", "/"),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(project_dir).replace("\\", "/"),
        "name": args.name,
        "save_dir": str(Path(metrics.save_dir)).replace("\\", "/"),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map5095": float(metrics.box.map),
    }

    out_json = Path(metrics.save_dir) / "evaluation_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved summary to: {out_json}")


if __name__ == "__main__":
    main()
