from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def normalize_binary(value) -> Optional[int]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "positive", "violation", "off_line"}:
        return 1
    if text in {"0", "false", "no", "negative", "valid", "on_line"}:
        return 0
    return None


def summarize_group(df: pd.DataFrame, positive_col: str, split_name: str) -> Dict[str, float]:
    positive_count = int(df[positive_col].sum())
    total = int(len(df))
    negative_count = total - positive_count
    majority_accuracy = max(positive_count, negative_count) / total if total else 0.0
    return {
        "split": split_name,
        "total": total,
        "negative_count": negative_count,
        "positive_count": positive_count,
        "negative_rate": negative_count / total if total else 0.0,
        "positive_rate": positive_count / total if total else 0.0,
        "majority_class_baseline_accuracy": majority_accuracy,
    }


def build_markdown(summary_df: pd.DataFrame, positive_name: str, negative_name: str) -> str:
    lines = [
        "# Dataset Class Balance",
        "",
        f"- Positive class: `{positive_name}`",
        f"- Negative class: `{negative_name}`",
        "",
        "| split | total | " + negative_name + " | " + positive_name + " | " + negative_name + " rate | " + positive_name + " rate | majority baseline accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['split']} | {int(row['total'])} | {int(row['negative_count'])} | {int(row['positive_count'])} | "
            f"{row['negative_rate']:.1%} | {row['positive_rate']:.1%} | {row['majority_class_baseline_accuracy']:.3f} |"
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize class balance overall and by split.")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--splits-csv", required=True)
    parser.add_argument("--key-col", default="clip_name")
    parser.add_argument("--label-col", default="violation")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--positive-name", default="violation")
    parser.add_argument("--negative-name", default="valid")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    labels_csv = Path(args.labels_csv)
    splits_csv = Path(args.splits_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(labels_csv).copy()
    splits_df = pd.read_csv(splits_csv).copy()

    if args.key_col not in labels_df.columns:
        raise ValueError(f"Missing key column in labels CSV: {args.key_col}")
    if args.key_col not in splits_df.columns:
        raise ValueError(f"Missing key column in splits CSV: {args.key_col}")
    if args.label_col not in labels_df.columns:
        raise ValueError(f"Missing label column in labels CSV: {args.label_col}")
    if args.split_col not in splits_df.columns:
        raise ValueError(f"Missing split column in splits CSV: {args.split_col}")

    merged = labels_df.merge(
        splits_df[[args.key_col, args.split_col]],
        on=args.key_col,
        how="left",
    )
    merged["_binary_label"] = merged[args.label_col].map(normalize_binary)
    merged_clean = merged.loc[merged["_binary_label"].isin([0, 1])].copy()
    if merged_clean.empty:
        raise ValueError("No binary labels could be parsed from the provided label column.")

    summaries = [summarize_group(merged_clean, "_binary_label", "overall")]
    for split_name, split_df in merged_clean.groupby(args.split_col, dropna=False):
        split_label = "missing_split" if pd.isna(split_name) else str(split_name)
        summaries.append(summarize_group(split_df, "_binary_label", split_label))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "class_balance.csv", index=False)

    summary_payload = {
        "labels_csv": str(labels_csv).replace("\\", "/"),
        "splits_csv": str(splits_csv).replace("\\", "/"),
        "key_col": args.key_col,
        "label_col": args.label_col,
        "split_col": args.split_col,
        "positive_name": args.positive_name,
        "negative_name": args.negative_name,
        "rows_total": int(len(merged)),
        "rows_used": int(len(merged_clean)),
        "summary": summaries,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    report_md = build_markdown(summary_df, args.positive_name, args.negative_name)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    print(f"Saved class balance summary to: {out_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
