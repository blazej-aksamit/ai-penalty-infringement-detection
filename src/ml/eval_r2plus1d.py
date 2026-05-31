import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r2plus1d_18


def find_clip_path(row: pd.Series) -> Path:
    if "window_file" in row and pd.notna(row["window_file"]):
        p = Path(str(row["window_file"]))
        if p.exists():
            return p
    if "clip_path" in row and pd.notna(row["clip_path"]):
        p = Path(str(row["clip_path"]))
        if p.exists():
            return p
    if "clip_name" in row and pd.notna(row["clip_name"]):
        p = Path("data/clips/kick_windows_720p_v2") / str(row["clip_name"])
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not resolve clip path for row: {row.to_dict()}")


def load_dataframe(labels_csv: str, splits_csv: str) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    splits = pd.read_csv(splits_csv)

    df = labels.merge(
        splits[["clip_name", "split"]],
        on="clip_name",
        how="inner",
    )

    df = df[df["violation"].astype(str).isin(["0", "1"])].copy()
    df["violation"] = df["violation"].astype(int)

    clip_paths = []
    keep_idx = []
    for i, row in df.iterrows():
        try:
            clip_paths.append(str(find_clip_path(row)))
            keep_idx.append(i)
        except FileNotFoundError:
            pass

    df = df.loc[keep_idx].copy()
    df["clip_path"] = clip_paths
    df = df.reset_index(drop=True)
    return df


def read_video_frames(video_path: str, num_frames: int = 16, size: int = 112) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"No frames in video: {video_path}")

    idxs = np.linspace(0, max(frame_count - 1, 0), num_frames).astype(int)
    frames = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Could not decode any frames from: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    arr = np.stack(frames).astype(np.float32) / 255.0
    mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
    std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (3, 0, 1, 2))
    return torch.tensor(arr, dtype=torch.float32)


class PenaltyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, num_frames: int = 16, size: int = 112):
        self.df = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = read_video_frames(row["clip_path"], self.num_frames, self.size)
        y = torch.tensor(int(row["violation"]), dtype=torch.long)
        return x, y, row["clip_name"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--outdir", default="runs/violation_r2plus1d")
    args = parser.parse_args()

    df = load_dataframe(args.labels, args.splits)
    df = df[df["split"] == args.split].copy()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = {}
    num_frames = int(ckpt_args.get("num_frames", 16))
    size = int(ckpt_args.get("size", 112))

    ds = PenaltyDataset(df, num_frames=num_frames, size=size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = r2plus1d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    rows = []

    with torch.no_grad():
        for xb, yb, names in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()

            for i in range(len(names)):
                rows.append(
                    {
                        "clip_name": names[i],
                        "y_true": int(yb[i].item()),
                        "y_pred": int(preds[i]),
                        "prob_0": float(probs[i, 0].cpu().item()),
                        "prob_1": float(probs[i, 1].cpu().item()),
                    }
                )

    pred_df = pd.DataFrame(rows)

    tp = int(((pred_df["y_pred"] == 1) & (pred_df["y_true"] == 1)).sum())
    tn = int(((pred_df["y_pred"] == 0) & (pred_df["y_true"] == 0)).sum())
    fp = int(((pred_df["y_pred"] == 1) & (pred_df["y_true"] == 0)).sum())
    fn = int(((pred_df["y_pred"] == 0) & (pred_df["y_true"] == 1)).sum())

    acc = float((pred_df["y_pred"] == pred_df["y_true"]).mean()) if len(pred_df) else 0.0
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = (
        2 * precision_pos * recall_pos / (precision_pos + recall_pos)
        if (precision_pos + recall_pos) > 0
        else 0.0
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pred_path = outdir / f"predictions_{args.split}.csv"
    mis_path = outdir / f"misclassified_{args.split}.csv"

    pred_df.to_csv(pred_path, index=False)
    pred_df[pred_df["y_true"] != pred_df["y_pred"]].to_csv(mis_path, index=False)

    print(f"split={args.split}")
    print(f"n={len(pred_df)}")
    print(f"acc={acc:.4f}")
    print(f"tp={tp} tn={tn} fp={fp} fn={fn}")
    print(f"precision_pos={precision_pos:.4f}")
    print(f"recall_pos={recall_pos:.4f}")
    print(f"f1_pos={f1_pos:.4f}")
    print(f"Saved: {pred_path}")
    print(f"Saved: {mis_path}")


if __name__ == "__main__":
    main()