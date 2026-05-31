from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


TRUE_TOKENS = {"1", "true", "yes", "y"}
FALSE_TOKENS = {"0", "false", "no", "n"}


def parse_optional_bool(value):
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in TRUE_TOKENS:
        return True
    if text in FALSE_TOKENS:
        return False
    return None


def normalize_comment(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def main():
    parser = argparse.ArgumentParser(description="Apply a configurable `uncertain` policy to line-decision results.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--decision-col", default="system_label")
    parser.add_argument("--line-thresh-px", type=float, default=10.0)
    parser.add_argument("--uncertainty-margin-px", type=float, default=2.0)
    parser.add_argument("--local-y-err-thresh-px", type=float, default=8.0)
    parser.add_argument("--min-dist-col", default="min_dist_px")
    parser.add_argument("--local-y-col", default="local_y_err_px")
    parser.add_argument("--has-goalkeeper-col", default="has_goalkeeper")
    parser.add_argument("--has-line-col", default="has_line")
    parser.add_argument("--left-ankle-col", default="left_ankle_visible")
    parser.add_argument("--right-ankle-col", default="right_ankle_visible")
    parser.add_argument("--comments-col", default="comments")
    parser.add_argument("--reason-col", default="reason")
    parser.add_argument("--output-decision-col", default="policy_decision")
    parser.add_argument("--output-reason-col", default="policy_uncertain_reason")
    args = parser.parse_args()

    in_csv = Path(args.input_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv).copy()

    policy_decisions: List[str] = []
    policy_reasons: List[str] = []
    policy_flags: List[str] = []

    line_thresh = float(args.line_thresh_px)
    margin = float(args.uncertainty_margin_px)
    local_y_thresh = float(args.local_y_err_thresh_px)

    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        decision = str(row_dict.get(args.decision_col, "")).strip()
        min_dist = row_dict.get(args.min_dist_col)
        local_y_err = row_dict.get(args.local_y_col)
        has_goalkeeper = parse_optional_bool(row_dict.get(args.has_goalkeeper_col))
        has_line = parse_optional_bool(row_dict.get(args.has_line_col))
        left_ankle = parse_optional_bool(row_dict.get(args.left_ankle_col))
        right_ankle = parse_optional_bool(row_dict.get(args.right_ankle_col))
        comments = normalize_comment(row_dict.get(args.comments_col))
        base_reason = str(row_dict.get(args.reason_col, "")).strip()

        reason = ""
        flags: List[str] = []
        final_decision = decision

        if decision == "uncertain":
            final_decision = "uncertain"
            reason = base_reason or "already_uncertain"
            flags.append("already_uncertain")
        elif has_goalkeeper is False:
            final_decision = "uncertain"
            reason = "no_goalkeeper"
            flags.append("no_goalkeeper")
        elif has_line is False:
            final_decision = "uncertain"
            reason = "no_line"
            flags.append("no_line")
        elif pd.isna(min_dist):
            final_decision = "uncertain"
            reason = "missing_min_dist"
            flags.append("missing_min_dist")
        elif pd.isna(local_y_err):
            final_decision = "uncertain"
            reason = "missing_local_y_err"
            flags.append("missing_local_y_err")
        else:
            min_dist = float(min_dist)
            local_y_err = float(local_y_err)

            near_boundary = abs(min_dist - line_thresh) <= margin
            high_local_y = local_y_err >= local_y_thresh
            ankle_occlusion = (left_ankle is False) or (right_ankle is False)
            comment_occlusion = "occluded" in comments

            if near_boundary:
                flags.append("near_boundary")
            if high_local_y:
                flags.append("high_local_y_err")
            if ankle_occlusion:
                flags.append("ankle_occlusion")
            if comment_occlusion:
                flags.append("comment_occlusion")

            if near_boundary and high_local_y:
                final_decision = "uncertain"
                reason = "boundary_plus_local_y"
            elif near_boundary and ankle_occlusion:
                final_decision = "uncertain"
                reason = "boundary_plus_ankle_occlusion"
            elif near_boundary and comment_occlusion:
                final_decision = "uncertain"
                reason = "boundary_plus_comment_occlusion"
            elif near_boundary:
                final_decision = "uncertain"
                reason = "boundary_margin"
            else:
                reason = ""

        policy_decisions.append(final_decision)
        policy_reasons.append(reason)
        policy_flags.append("|".join(flags))

    df[args.output_decision_col] = policy_decisions
    df[args.output_reason_col] = policy_reasons
    df["policy_flags"] = policy_flags
    df["policy_uncertain"] = df[args.output_decision_col].astype(str) == "uncertain"
    df.to_csv(out_csv, index=False)

    print(f"Saved uncertainty-policy output to: {out_csv}")
    print(df[args.output_decision_col].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
