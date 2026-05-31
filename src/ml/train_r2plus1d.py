import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


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

    if "clip_name" not in labels.columns:
        raise ValueError("labels CSV must contain clip_name")
    if "violation" not in labels.columns:
        raise ValueError("labels CSV must contain violation")
    if "clip_name" not in splits.columns or "split" not in splits.columns:
        raise ValueError("splits CSV must contain clip_name and split")

    df = labels.merge(
        splits[["clip_name", "split"]],
        on="clip_name",
        how="inner",
    )

    df = df[df["violation"].astype(str).isin(["0", "1"])].copy()
    df["violation"] = df["violation"].astype(int)

    clip_paths: List[str] = []
    keep_idx: List[int] = []

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


def read_video_frames(
    video_path: str,
    num_frames: int = 16,
    size: int = 112,
) -> torch.Tensor:
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
    arr = np.transpose(arr, (3, 0, 1, 2))  # C, T, H, W
    return torch.tensor(arr, dtype=torch.float32)


class PenaltyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, num_frames: int = 16, size: int = 112):
        self.df = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.size = size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        x = read_video_frames(
            row["clip_path"],
            num_frames=self.num_frames,
            size=self.size,
        )
        y = torch.tensor(int(row["violation"]), dtype=torch.long)
        return x, y


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    n = 0

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = ce(logits, yb)

            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)

            preds = torch.argmax(logits, dim=1)
            preds_all.extend(preds.cpu().numpy().tolist())
            targets_all.extend(yb.cpu().numpy().tolist())

    preds_all = np.array(preds_all)
    targets_all = np.array(targets_all)

    tp = int(((preds_all == 1) & (targets_all == 1)).sum())
    tn = int(((preds_all == 0) & (targets_all == 0)).sum())
    fp = int(((preds_all == 1) & (targets_all == 0)).sum())
    fn = int(((preds_all == 0) & (targets_all == 1)).sum())

    acc = float((preds_all == targets_all).mean()) if len(targets_all) else 0.0
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = (
        2 * precision_pos * recall_pos / (precision_pos + recall_pos)
        if (precision_pos + recall_pos) > 0
        else 0.0
    )

    return {
        "loss": total_loss / max(n, 1),
        "acc": acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--outdir", default="runs/violation_r2plus1d")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=112)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(args.labels, args.splits)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"Loaded dataset: total={len(df)} train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    c0 = int((train_df["violation"] == 0).sum())
    c1 = int((train_df["violation"] == 1).sum())
    print(f"Train class counts: c0={c0} c1={c1}")

    y_train = train_df["violation"].to_numpy()
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Class weights: {class_weights.tolist()}")

    train_ds = PenaltyDataset(train_df, num_frames=args.num_frames, size=args.size)
    val_ds = PenaltyDataset(val_df, num_frames=args.num_frames, size=args.size)
    test_ds = PenaltyDataset(test_df, num_frames=args.num_frames, size=args.size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weights = R2Plus1D_18_Weights.DEFAULT
    model = r2plus1d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    best_path = outdir / "best.pt"
    last_path = outdir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            seen += xb.size(0)

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"step {step}/{len(train_loader)} | "
                f"loss={loss.item():.4f}"
            )

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"tp={val_metrics['tp']} tn={val_metrics['tn']} "
            f"fp={val_metrics['fp']} fn={val_metrics['fn']} "
            f"precision_pos={val_metrics['precision_pos']:.4f} "
            f"recall_pos={val_metrics['recall_pos']:.4f} "
            f"f1_pos={val_metrics['f1_pos']:.4f}"
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "args": vars(args),
        }
        torch.save(state, last_path)

        if val_metrics["f1_pos"] > best_f1:
            best_f1 = val_metrics["f1_pos"]
            torch.save(state, best_path)
            print(f"Saved new best checkpoint: {best_path}")

    print("\nFinal test evaluation using best checkpoint...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print(test_metrics)


if __name__ == "__main__":
    main()