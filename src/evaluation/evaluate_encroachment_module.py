from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def gt_label_from_row(row: pd.Series) -> str:
    uncertain = int(row.get("uncertain", 0))
    encroachment = row.get("encroachment", "")
    if uncertain == 1:
        return "uncertain"
    return "encroachment" if str(encroachment).strip() == "1" else "no_encroachment"


def main():
    parser = argparse.ArgumentParser(description="Evaluate encroachment module against manual labels.")
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--labels-csv", default="data/meta/encroachment_labels.csv")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    results_csv = resolve_repo_path(args.results_csv)
    labels_csv = resolve_repo_path(args.labels_csv)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(results_csv).copy()
    labels_df = pd.read_csv(labels_csv).copy()

    results_df["frame_idx_gt"] = results_df["frame_idx_gt"].astype(int)
    labels_df["frame_idx_gt"] = labels_df["frame_idx_gt"].astype(int)
    labels_df["truth_label"] = labels_df.apply(gt_label_from_row, axis=1)

    merged = results_df.merge(
        labels_df[["clip_name", "frame_idx_gt", "truth_label", "encroachment", "uncertain"]],
        on=["clip_name", "frame_idx_gt"],
        how="inner",
    )

    merged["pred_label"] = merged["decision"].astype(str)
    merged["exact_match"] = merged["pred_label"] == merged["truth_label"]
    merged["certain_pred"] = merged["pred_label"] != "uncertain"

    certain_df = merged.loc[merged["certain_pred"]].copy()
    tp = int(((certain_df["pred_label"] == "encroachment") & (certain_df["truth_label"] == "encroachment")).sum())
    fp = int(((certain_df["pred_label"] == "encroachment") & (certain_df["truth_label"] != "encroachment")).sum())
    tn = int(((certain_df["pred_label"] == "no_encroachment") & (certain_df["truth_label"] == "no_encroachment")).sum())
    fn = int(((certain_df["pred_label"] == "no_encroachment") & (certain_df["truth_label"] == "encroachment")).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    summary = {
        "labeled_samples": int(len(merged)),
        "exact_match_rate_all": float(merged["exact_match"].mean()) if len(merged) else 0.0,
        "coverage_non_uncertain": float(merged["certain_pred"].mean()) if len(merged) else 0.0,
        "selective_accuracy_certain_only": float(certain_df["exact_match"].mean()) if len(certain_df) else 0.0,
        "truth_counts": merged["truth_label"].value_counts(dropna=False).to_dict(),
        "pred_counts": merged["pred_label"].value_counts(dropna=False).to_dict(),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision_encroachment": precision,
        "recall_encroachment": recall,
        "f1_encroachment": f1,
    }

    merged.to_csv(out_dir / "encroachment_eval_joined.csv", index=False)
    with open(out_dir / "encroachment_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation join to: {out_dir / 'encroachment_eval_joined.csv'}")


if __name__ == "__main__":
    main()
