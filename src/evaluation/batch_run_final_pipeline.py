from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def truth_label_from_row(row: pd.Series) -> str:
    uncertain = int(row.get("uncertain", 0))
    violation = int(row.get("violation", 0))
    if uncertain == 1:
        return "uncertain"
    return "violation" if violation == 1 else "valid"


def pred_label_from_decision(decision: str) -> str:
    decision = str(decision).strip()
    if decision == "off_line":
        return "violation"
    if decision == "on_line":
        return "valid"
    return "uncertain"


def main():
    parser = argparse.ArgumentParser(description="Batch-run the final pipeline over labeled kick-window clips.")
    parser.add_argument("--labels-csv", default="data/meta/keeper_violation_labels_final.csv")
    parser.add_argument("--splits-csv", default="data/meta/splits_violation.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--pipeline-script", default="scripts/pipeline/run_full_penalty_pipeline.py")
    parser.add_argument("--model-path", default="runs/detect/train4/weights/best.pt")
    parser.add_argument("--pose-model-path", default=None)
    parser.add_argument("--pose-leg-extension-factor", type=float, default=0.35)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--kick-time-in-window-s", type=float, default=1.5)
    parser.add_argument("--use-auto-kick", action="store_true")
    parser.add_argument("--kick-frame-adjust", type=int, default=0)
    parser.add_argument("--apply-uncertain-policy", action="store_true")
    parser.add_argument("--uncertainty-margin-px", type=float, default=2.0)
    parser.add_argument("--uncertainty-local-y-err-px", type=float, default=8.0)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--clip-substring", default=None)
    args = parser.parse_args()

    labels_df = pd.read_csv(resolve_repo_path(args.labels_csv)).copy()
    splits_df = pd.read_csv(resolve_repo_path(args.splits_csv)).copy()
    merged = labels_df.merge(
        splits_df[["clip_name", "split"]],
        on="clip_name",
        how="inner",
    )
    merged = merged.loc[merged["split"] == args.split].copy()

    if args.clip_substring:
        merged = merged.loc[merged["clip_name"].astype(str).str.contains(args.clip_substring, regex=False)].copy()
    if args.max_clips is not None:
        merged = merged.head(int(args.max_clips)).copy()

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_script = resolve_repo_path(args.pipeline_script)
    model_path = resolve_repo_path(args.model_path)

    rows: List[Dict[str, object]] = []
    for row in merged.itertuples(index=False):
        clip_name = str(row.clip_name)
        window_file = resolve_repo_path(str(row.window_file))
        clip_out_root = out_dir / args.split
        clip_out_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(pipeline_script),
            "--video-path",
            str(window_file),
            "--model-path",
            str(model_path),
            "--out-root",
            str(clip_out_root),
        ]

        if args.pose_model_path:
            cmd.extend(
                [
                    "--pose-model-path",
                    str(resolve_repo_path(args.pose_model_path)),
                    "--pose-leg-extension-factor",
                    str(args.pose_leg_extension_factor),
                ]
            )

        if args.use_auto_kick:
            fps = float(row.fps)
            frame_idx = int(round(float(args.kick_time_in_window_s) * fps))
            cmd.extend(
                [
                    "--auto-kick",
                    "--frame-idx",
                    str(frame_idx),
                    "--kick-window-start-s",
                    "0.5",
                    "--kick-window-end-s",
                    "2.5",
                    "--kick-frame-adjust",
                    str(args.kick_frame_adjust),
                ]
            )
        else:
            fps = float(row.fps)
            frame_idx = int(round(float(args.kick_time_in_window_s) * fps))
            cmd.extend(["--frame-idx", str(frame_idx)])

        if args.apply_uncertain_policy:
            cmd.extend(
                [
                    "--apply-uncertain-policy",
                    "--uncertainty-margin-px",
                    str(args.uncertainty_margin_px),
                    "--uncertainty-local-y-err-px",
                    str(args.uncertainty_local_y_err_px),
                ]
            )

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(REPO_ROOT),
            )
            clip_result_dir = clip_out_root / Path(clip_name).stem
            result_json_path = clip_result_dir / "final_result.json"
            result_payload = json.loads(result_json_path.read_text(encoding="utf-8"))
            pred_label = pred_label_from_decision(str(result_payload.get("decision", "")))
            truth_label = truth_label_from_row(pd.Series(row._asdict()))

            rows.append(
                {
                    "clip_name": clip_name,
                    "window_file": str(window_file).replace("\\", "/"),
                    "split": args.split,
                    "truth_label": truth_label,
                    "violation": int(row.violation),
                    "uncertain_truth": int(row.uncertain),
                    "pipeline_ok": True,
                    "decision": result_payload.get("decision"),
                    "reason": result_payload.get("reason"),
                    "pred_label": pred_label,
                    "raw_decision": result_payload.get("raw_decision"),
                    "raw_reason": result_payload.get("raw_reason"),
                    "policy_decision": result_payload.get("policy_decision"),
                    "policy_reason": result_payload.get("policy_reason"),
                    "policy_flags": result_payload.get("policy_flags"),
                    "min_dist_px": result_payload.get("min_dist_px"),
                    "local_y_err_px": result_payload.get("local_y_err_px"),
                    "has_goalkeeper": result_payload.get("has_goalkeeper"),
                    "has_line": result_payload.get("has_line"),
                    "best_point": result_payload.get("best_point"),
                    "best_point_source": result_payload.get("best_point_source"),
                    "pose_available": result_payload.get("pose_available"),
                    "pose_reason": result_payload.get("pose_reason"),
                    "pose_point_count": result_payload.get("pose_point_count"),
                    "pose_selected_points": result_payload.get("pose_selected_points"),
                    "pose_decision_point": result_payload.get("pose_decision_point"),
                    "pose_source_mode": result_payload.get("pose_source_mode"),
                    "frame_idx": result_payload.get("frame_idx"),
                    "kick_source": result_payload.get("kick_source"),
                    "result_json": str(result_json_path).replace("\\", "/"),
                    "stdout_tail": "\n".join((completed.stdout or "").splitlines()[-8:]),
                }
            )
        except subprocess.CalledProcessError as exc:
            rows.append(
                {
                    "clip_name": clip_name,
                    "window_file": str(window_file).replace("\\", "/"),
                    "split": args.split,
                    "truth_label": truth_label_from_row(pd.Series(row._asdict())),
                    "violation": int(row.violation),
                    "uncertain_truth": int(row.uncertain),
                    "pipeline_ok": False,
                    "decision": None,
                    "reason": None,
                    "pred_label": None,
                    "stderr_tail": "\n".join((exc.stderr or "").splitlines()[-12:]),
                    "stdout_tail": "\n".join((exc.stdout or "").splitlines()[-12:]),
                }
            )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / f"{args.split}_pipeline_batch_results.csv", index=False)

    ok_df = result_df.loc[result_df["pipeline_ok"] == True].copy()
    summary = {
        "split": args.split,
        "clips_attempted": int(len(result_df)),
        "pipeline_ok": int(ok_df.shape[0]),
        "pipeline_failed": int((result_df["pipeline_ok"] == False).sum()),
        "decision_counts": ok_df["decision"].value_counts(dropna=False).to_dict() if len(ok_df) else {},
        "pred_label_counts": ok_df["pred_label"].value_counts(dropna=False).to_dict() if len(ok_df) else {},
        "exact_match_rate": float((ok_df["pred_label"] == ok_df["truth_label"]).mean()) if len(ok_df) else 0.0,
        "coverage_non_uncertain": float(ok_df["pred_label"].ne("uncertain").mean()) if len(ok_df) else 0.0,
    }
    with open(out_dir / f"{args.split}_pipeline_batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved batch CSV to: {out_dir / f'{args.split}_pipeline_batch_results.csv'}")


if __name__ == "__main__":
    main()
