from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main():
    parser = argparse.ArgumentParser(description="Run YOLO pose inference on a directory of goalkeeper crops.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--project", default="runs/pose")
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ultralytics is not installed in the active Python environment. "
            "Run this script from the environment used for YOLO work."
        ) from exc

    model_path = resolve_repo_path(args.model_path)
    source_path = resolve_repo_path(args.source)
    project_dir = resolve_repo_path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        imgsz=args.imgsz,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        save=True,
        save_txt=True,
        save_conf=True,
        verbose=False,
        device=args.device,
    )

    save_dir = Path(results[0].save_dir) if results else project_dir / args.name
    summary = {
        "model_path": str(model_path).replace("\\", "/"),
        "source": str(source_path).replace("\\", "/"),
        "project": str(project_dir).replace("\\", "/"),
        "name": args.name,
        "save_dir": str(save_dir).replace("\\", "/"),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "image_count": len(results),
    }

    out_json = save_dir / "pose_inference_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved summary to: {out_json}")


if __name__ == "__main__":
    main()
