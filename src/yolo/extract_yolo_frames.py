from pathlib import Path
import cv2
import pandas as pd

LABELS_CSV = Path("data/meta/keeper_violation_labels_final.csv")
SPLITS_CSV = Path("data/meta/splits_violation.csv")
OUT_ROOT = Path("data/yolo_gk_ball")

FRAMES_RELATIVE = [-20, -16, -12, -8]  # around clip midpoint


def safe_stem(name: str) -> str:
    return Path(name).stem.replace(" ", "_")


def main():
    labels = pd.read_csv(LABELS_CSV)
    splits = pd.read_csv(SPLITS_CSV)

    df = labels.merge(
        splits[["clip_name", "match_id", "split"]],
        on="clip_name",
        how="inner",
    )

    out_rows = []

    for _, row in df.iterrows():
        clip_name = row["clip_name"]
        split = row["split"]
        violation = int(row["violation"])
        match_id = row["match_id"]

        if "window_file" in row and pd.notna(row["window_file"]):
            clip_path = Path(row["window_file"])
        else:
            clip_path = Path("data/clips/kick_windows_720p_v2") / clip_name

        if not clip_path.exists():
            print(f"Missing clip: {clip_path}")
            continue

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print(f"Could not open: {clip_path}")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        midpoint = frame_count // 2

        for rel in FRAMES_RELATIVE:
            frame_idx = max(0, min(frame_count - 1, midpoint + rel))

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"Could not read frame {frame_idx} from {clip_path.name}")
                continue

            out_name = f"{safe_stem(clip_name)}__f{frame_idx:04d}.jpg"
            out_path = OUT_ROOT / "images" / split / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(out_path), frame)

            out_rows.append(
                {
                    "image_name": out_name,
                    "image_path": str(out_path).replace("\\", "/"),
                    "clip_name": clip_name,
                    "clip_path": str(clip_path).replace("\\", "/"),
                    "match_id": match_id,
                    "split": split,
                    "violation": violation,
                    "frame_idx": frame_idx,
                    "frame_count": frame_count,
                }
            )

        cap.release()

    meta_dir = OUT_ROOT / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    meta_df = pd.DataFrame(out_rows)
    meta_path = meta_dir / "frames_metadata.csv"
    meta_df.to_csv(meta_path, index=False)

    print(f"Saved metadata: {meta_path}")
    print(meta_df["split"].value_counts())
    print(f"Total images: {len(meta_df)}")


if __name__ == "__main__":
    main()