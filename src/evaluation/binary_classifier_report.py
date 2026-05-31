from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def normalize_label(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def compute_binary_metrics(truth: List[str], pred: List[str], positive_label: str) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(truth, pred) if t == positive_label and p == positive_label)
    tn = sum(1 for t, p in zip(truth, pred) if t != positive_label and p != positive_label)
    fp = sum(1 for t, p in zip(truth, pred) if t != positive_label and p == positive_label)
    fn = sum(1 for t, p in zip(truth, pred) if t == positive_label and p != positive_label)

    total = len(truth)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    return {
        "support": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_pos": precision,
        "recall_pos": recall,
        "specificity": specificity,
        "npv": npv,
        "f1_pos": f1,
    }


def build_report_markdown(
    metrics: Dict[str, float],
    priors: Dict[str, float],
    baseline_rows: List[Dict[str, float]],
    truth_col: str,
    pred_col: str,
    positive_label: str,
    negative_label: str,
    excluded_count: int,
) -> str:
    lines = [
        "# Binary Decision Evaluation",
        "",
        "## Setup",
        "",
        f"- Ground-truth column: `{truth_col}`",
        f"- Prediction column: `{pred_col}`",
        f"- Positive class: `{positive_label}`",
        f"- Negative class: `{negative_label}`",
        f"- Evaluated rows: `{metrics['support']}`",
        f"- Excluded rows: `{excluded_count}`",
        "",
        "## Class Priors",
        "",
        f"- `{negative_label}`: `{int(priors['negative_count'])}` ({priors['negative_rate']:.1%})",
        f"- `{positive_label}`: `{int(priors['positive_count'])}` ({priors['positive_rate']:.1%})",
        "",
        "## Model Metrics",
        "",
        f"- Accuracy: `{metrics['accuracy']:.3f}`",
        f"- Balanced accuracy: `{metrics['balanced_accuracy']:.3f}`",
        f"- Precision ({positive_label}): `{metrics['precision_pos']:.3f}`",
        f"- Recall ({positive_label}): `{metrics['recall_pos']:.3f}`",
        f"- F1 ({positive_label}): `{metrics['f1_pos']:.3f}`",
        f"- Specificity: `{metrics['specificity']:.3f}`",
        "",
        "## Confusion Matrix",
        "",
        f"- TP: `{int(metrics['tp'])}`",
        f"- FP: `{int(metrics['fp'])}`",
        f"- TN: `{int(metrics['tn'])}`",
        f"- FN: `{int(metrics['fn'])}`",
        "",
        "## No-Skill Baselines",
        "",
        "| baseline | accuracy | balanced_accuracy | precision_pos | recall_pos | f1_pos |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in baseline_rows:
        lines.append(
            f"| {row['baseline']} | {row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | "
            f"{row['precision_pos']:.3f} | {row['recall_pos']:.3f} | {row['f1_pos']:.3f} |"
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a binary classifier against ground truth.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--truth-col", required=True)
    parser.add_argument("--pred-col", required=True)
    parser.add_argument("--positive-label", required=True)
    parser.add_argument("--negative-label", default=None)
    parser.add_argument("--drop-labels", default="uncertain")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--random-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv).copy()
    if args.truth_col not in df.columns:
        raise ValueError(f"Missing truth column: {args.truth_col}")
    if args.pred_col not in df.columns:
        raise ValueError(f"Missing prediction column: {args.pred_col}")

    df["_truth"] = df[args.truth_col].map(normalize_label)
    df["_pred"] = df[args.pred_col].map(normalize_label)

    drop_labels = {normalize_label(x) for x in args.drop_labels.split(",") if normalize_label(x)}
    valid_mask = (
        (df["_truth"] != "")
        & (df["_pred"] != "")
        & (~df["_truth"].isin(drop_labels))
        & (~df["_pred"].isin(drop_labels))
    )

    excluded = df.loc[~valid_mask].copy()
    filtered = df.loc[valid_mask].copy()
    if filtered.empty:
        raise ValueError("No valid rows left after filtering.")

    positive_label = normalize_label(args.positive_label)
    truth_labels = sorted(filtered["_truth"].unique().tolist())
    if positive_label not in truth_labels:
        raise ValueError(
            f"Positive label '{positive_label}' not present in filtered ground truth labels: {truth_labels}"
        )

    if args.negative_label:
        negative_label = normalize_label(args.negative_label)
    else:
        remaining = [label for label in truth_labels if label != positive_label]
        if len(remaining) != 1:
            raise ValueError(
                "Could not infer negative label. Provide --negative-label explicitly. "
                f"Filtered truth labels: {truth_labels}"
            )
        negative_label = remaining[0]

    truth = filtered["_truth"].tolist()
    pred = filtered["_pred"].tolist()

    positive_count = sum(1 for x in truth if x == positive_label)
    negative_count = len(truth) - positive_count
    priors = {
        "positive_label": positive_label,
        "negative_label": negative_label,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": positive_count / len(truth),
        "negative_rate": negative_count / len(truth),
    }

    metrics = compute_binary_metrics(truth, pred, positive_label)
    metrics["positive_label"] = positive_label
    metrics["negative_label"] = negative_label

    baseline_predictions = {
        "always_positive": [positive_label] * len(truth),
        "always_negative": [negative_label] * len(truth),
        "majority_class": ([positive_label] if positive_count >= negative_count else [negative_label]) * len(truth),
    }

    baseline_rows = []
    for baseline_name, baseline_pred in baseline_predictions.items():
        row = {"baseline": baseline_name}
        row.update(compute_binary_metrics(truth, baseline_pred, positive_label))
        baseline_rows.append(row)

    rng = np.random.default_rng(args.seed)
    random_rows = []
    for _ in range(args.random_repeats):
        draw = rng.random(len(truth)) < priors["positive_rate"]
        random_pred = [positive_label if flag else negative_label for flag in draw]
        random_rows.append(compute_binary_metrics(truth, random_pred, positive_label))

    random_mean = {"baseline": f"random_prior_mc_{args.random_repeats}"}
    for key in random_rows[0].keys():
        random_mean[key] = float(np.mean([row[key] for row in random_rows]))
    baseline_rows.append(random_mean)

    confusion_df = pd.DataFrame(
        [
            {"truth": negative_label, f"pred_{negative_label}": metrics["tn"], f"pred_{positive_label}": metrics["fp"]},
            {"truth": positive_label, f"pred_{negative_label}": metrics["fn"], f"pred_{positive_label}": metrics["tp"]},
        ]
    )

    filtered.to_csv(out_dir / "evaluated_rows.csv", index=False)
    excluded.to_csv(out_dir / "excluded_rows.csv", index=False)
    confusion_df.to_csv(out_dir / "confusion_matrix.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(out_dir / "baseline_metrics.csv", index=False)

    summary = {
        "input_csv": str(input_csv).replace("\\", "/"),
        "truth_col": args.truth_col,
        "pred_col": args.pred_col,
        "priors": priors,
        "metrics": metrics,
        "excluded_rows": int(len(excluded)),
        "baseline_metrics": baseline_rows,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_md = build_report_markdown(
        metrics=metrics,
        priors=priors,
        baseline_rows=baseline_rows,
        truth_col=args.truth_col,
        pred_col=args.pred_col,
        positive_label=positive_label,
        negative_label=negative_label,
        excluded_count=int(len(excluded)),
    )
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    print(f"Saved binary evaluation report to: {out_dir}")
    print(f"Evaluated rows: {metrics['support']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 ({positive_label}): {metrics['f1_pos']:.3f}")


if __name__ == "__main__":
    main()
