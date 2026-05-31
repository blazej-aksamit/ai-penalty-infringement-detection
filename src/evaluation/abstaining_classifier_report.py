from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def main():
    parser = argparse.ArgumentParser(description="Report metrics for a classifier that can abstain via an `uncertain` label.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--truth-col", required=True)
    parser.add_argument("--pred-col", required=True)
    parser.add_argument("--positive-label", required=True)
    parser.add_argument("--negative-label", required=True)
    parser.add_argument("--uncertain-label", default="uncertain")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv).copy()

    truth = df[args.truth_col].astype(str)
    pred = df[args.pred_col].astype(str)

    uncertain_mask = pred == args.uncertain_label
    certain_mask = ~uncertain_mask
    certain_df = df.loc[certain_mask].copy()

    total = int(len(df))
    certain = int(certain_mask.sum())
    uncertain = int(uncertain_mask.sum())

    tp = int(((truth == args.positive_label) & (pred == args.positive_label)).sum())
    fp = int(((truth == args.negative_label) & (pred == args.positive_label)).sum())
    tn = int(((truth == args.negative_label) & (pred == args.negative_label)).sum())
    fn = int(((truth == args.positive_label) & (pred == args.negative_label)).sum())

    resolved_correct = tp + tn
    selective_accuracy = safe_div(resolved_correct, certain)
    coverage = safe_div(certain, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    lower_bound_accuracy = safe_div(resolved_correct, total)

    uncertain_positive = int(((truth == args.positive_label) & uncertain_mask).sum())
    uncertain_negative = int(((truth == args.negative_label) & uncertain_mask).sum())

    summary = {
        "input_csv": str(input_csv).replace("\\", "/"),
        "positive_label": args.positive_label,
        "negative_label": args.negative_label,
        "uncertain_label": args.uncertain_label,
        "total_samples": total,
        "certain_predictions": certain,
        "uncertain_predictions": uncertain,
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "lower_bound_accuracy": lower_bound_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "uncertain_positive": uncertain_positive,
        "uncertain_negative": uncertain_negative,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    confusion_df = pd.DataFrame(
        [
            {"truth": args.positive_label, "pred": args.positive_label, "count": tp},
            {"truth": args.positive_label, "pred": args.negative_label, "count": fn},
            {"truth": args.positive_label, "pred": args.uncertain_label, "count": uncertain_positive},
            {"truth": args.negative_label, "pred": args.positive_label, "count": fp},
            {"truth": args.negative_label, "pred": args.negative_label, "count": tn},
            {"truth": args.negative_label, "pred": args.uncertain_label, "count": uncertain_negative},
        ]
    )
    confusion_df.to_csv(out_dir / "confusion_with_uncertain.csv", index=False)

    report_lines = [
        "# Abstaining Classifier Report",
        "",
        f"- total samples: `{total}`",
        f"- certain predictions: `{certain}`",
        f"- uncertain predictions: `{uncertain}`",
        f"- coverage: `{coverage:.3f}`",
        f"- selective accuracy: `{selective_accuracy:.3f}`",
        f"- lower-bound accuracy over all samples: `{lower_bound_accuracy:.3f}`",
        f"- precision for `{args.positive_label}`: `{precision:.3f}`",
        f"- recall for `{args.positive_label}`: `{recall:.3f}`",
        f"- F1 for `{args.positive_label}`: `{f1:.3f}`",
        "",
        "## Certain-only confusion counts",
        "",
        f"- TP: `{tp}`",
        f"- FP: `{fp}`",
        f"- TN: `{tn}`",
        f"- FN: `{fn}`",
        "",
        "## Abstentions by truth class",
        "",
        f"- uncertain on `{args.positive_label}`: `{uncertain_positive}`",
        f"- uncertain on `{args.negative_label}`: `{uncertain_negative}`",
    ]
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved report to: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
