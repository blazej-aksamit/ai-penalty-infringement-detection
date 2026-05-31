from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def window_clip_to_full_clip(window_clip_name: str) -> str:
    path = Path(window_clip_name)
    stem = path.stem
    if stem.endswith("_KICK"):
        stem = stem[:-5]
    return f"{stem}.mp4"


def load_split_window_names(splits_csv: Path, split: Optional[str]) -> List[str]:
    splits_df = pd.read_csv(splits_csv)
    if split:
        splits_df = splits_df.loc[splits_df["split"].astype(str) == split].copy()
    return list(splits_df["clip_name"].astype(str).tolist())


def map_goalkeeper_truth(violation: Optional[float], uncertain: Optional[float]) -> Optional[str]:
    if pd.isna(violation) and pd.isna(uncertain):
        return None
    if int(uncertain or 0) == 1:
        return "uncertain"
    if int(violation or 0) == 1:
        return "violation"
    return "valid"


def map_goalkeeper_pred(decision: Optional[str]) -> Optional[str]:
    if decision == "off_line":
        return "violation"
    if decision == "on_line":
        return "valid"
    if decision == "uncertain":
        return "uncertain"
    return None


def main():
    parser = argparse.ArgumentParser(description="Batch-run combined goalkeeper-line + encroachment checks on GT kick frames.")
    parser.add_argument("--kick-times-csv", default="data/meta/kick_times.csv")
    parser.add_argument("--splits-csv", default="data/meta/splits_violation.csv")
    parser.add_argument("--keeper-labels-csv", default="data/meta/keeper_violation_labels_final.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--clips-dir", default="data/clips/penalties_720p")
    parser.add_argument("--combined-script", default="scripts/pipeline/run_combined_penalty_officiating_pipeline.py")
    parser.add_argument("--model-path", default="runs/detect/runs/detect/train_yolo26n_gk_ball/weights/best.pt")
    parser.add_argument("--player-model-path", default="yolo26n.pt")
    parser.add_argument("--pose-model-path", default=None)
    parser.add_argument("--apply-uncertain-policy", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--clip-substring", default=None)
    args = parser.parse_args()

    kick_times_csv = resolve_repo_path(args.kick_times_csv)
    splits_csv = resolve_repo_path(args.splits_csv)
    keeper_labels_csv = resolve_repo_path(args.keeper_labels_csv)
    clips_dir = resolve_repo_path(args.clips_dir)
    combined_script = resolve_repo_path(args.combined_script)
    model_path = resolve_repo_path(args.model_path)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_window_names = load_split_window_names(splits_csv, args.split)
    split_full_to_window = {window_clip_to_full_clip(name): name for name in split_window_names}

    kick_df = pd.read_csv(kick_times_csv).copy()
    kick_df = kick_df.loc[kick_df["clip_name"].astype(str).isin(split_full_to_window.keys())].copy()

    keeper_truth_lookup: Dict[str, Dict[str, object]] = {}
    if keeper_labels_csv.exists():
        keeper_df = pd.read_csv(keeper_labels_csv).copy()
        for row in keeper_df.itertuples(index=False):
            keeper_truth_lookup[str(row.clip_name)] = {
                "goalkeeper_truth_label": map_goalkeeper_truth(
                    getattr(row, "violation", None),
                    getattr(row, "uncertain", None),
                ),
                "goalkeeper_truth_violation": None if pd.isna(getattr(row, "violation", None)) else int(getattr(row, "violation")),
                "goalkeeper_truth_uncertain": None if pd.isna(getattr(row, "uncertain", None)) else int(getattr(row, "uncertain")),
            }

    if args.clip_substring:
        kick_df = kick_df.loc[kick_df["clip_name"].astype(str).str.contains(args.clip_substring, regex=False)].copy()
    if args.max_clips is not None:
        kick_df = kick_df.head(int(args.max_clips)).copy()

    rows: List[Dict[str, object]] = []
    for row in kick_df.itertuples(index=False):
        full_clip_name = str(row.clip_name)
        frame_idx = int(row.kick_frame)
        video_path = clips_dir / full_clip_name
        if not video_path.exists():
            rows.append(
                {
                    "clip_name": full_clip_name,
                    "window_clip_name": split_full_to_window.get(full_clip_name),
                    "frame_idx_gt": frame_idx,
                    "pipeline_ok": False,
                    "error": "video_not_found",
                }
            )
            continue

        cmd = [
            sys.executable,
            str(combined_script),
            "--video-path",
            str(video_path),
            "--frame-idx",
            str(frame_idx),
            "--model-path",
            str(model_path),
            "--player-model-path",
            str(args.player_model_path),
            "--out-root",
            str(out_dir),
        ]
        if args.pose_model_path:
            cmd.extend(["--pose-model-path", str(args.pose_model_path)])
        if args.apply_uncertain_policy:
            cmd.append("--apply-uncertain-policy")

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(REPO_ROOT),
            )
            clip_out_dir = out_dir / Path(full_clip_name).stem
            result_json_path = clip_out_dir / "combined_result.json"
            payload = json.loads(result_json_path.read_text(encoding="utf-8"))
            gk = payload.get("goalkeeper_line", {})
            enc = payload.get("encroachment", {})
            truth = keeper_truth_lookup.get(split_full_to_window.get(full_clip_name, ""), {})
            gk_pred_label = map_goalkeeper_pred(gk.get("decision"))
            gk_truth_label = truth.get("goalkeeper_truth_label")
            rows.append(
                {
                    "clip_name": full_clip_name,
                    "window_clip_name": split_full_to_window.get(full_clip_name),
                    "frame_idx_gt": frame_idx,
                    "kick_time_s_gt": float(row.kick_time_s),
                    "pipeline_ok": True,
                    "goalkeeper_decision": gk.get("decision"),
                    "goalkeeper_reason": gk.get("reason"),
                    "goalkeeper_pred_label": gk_pred_label,
                    "goalkeeper_truth_label": gk_truth_label,
                    "goalkeeper_exact_match": (
                        None if gk_truth_label is None or gk_pred_label is None else gk_truth_label == gk_pred_label
                    ),
                    "goalkeeper_min_dist_px": gk.get("min_dist_px"),
                    "goalkeeper_local_y_err_px": gk.get("local_y_err_px"),
                    "encroachment_decision": enc.get("decision"),
                    "encroachment_reason": enc.get("decision_reason"),
                    "encroachment_candidate_count": enc.get("encroachment_candidate_count"),
                    "has_goalkeeper_box": enc.get("has_goalkeeper_box"),
                    "has_ball_box": enc.get("has_ball_box"),
                    "combined_result_json": str(result_json_path).replace("\\", "/"),
                    "combined_overlay_path": payload.get("combined_overlay_path"),
                    "stdout_tail": "\n".join((completed.stdout or "").splitlines()[-12:]),
                }
            )
        except subprocess.CalledProcessError as exc:
            rows.append(
                {
                    "clip_name": full_clip_name,
                    "window_clip_name": split_full_to_window.get(full_clip_name),
                    "frame_idx_gt": frame_idx,
                    "kick_time_s_gt": float(row.kick_time_s),
                    "pipeline_ok": False,
                    "error": "pipeline_failed",
                    "stderr_tail": "\n".join((exc.stderr or "").splitlines()[-12:]),
                    "stdout_tail": "\n".join((exc.stdout or "").splitlines()[-12:]),
                }
            )

    result_df = pd.DataFrame(rows)
    result_csv = out_dir / f"{args.split}_combined_officiating_results.csv"
    result_df.to_csv(result_csv, index=False)

    ok_df = result_df.loc[result_df["pipeline_ok"] == True].copy()
    summary = {
        "split": args.split,
        "clips_attempted": int(len(result_df)),
        "pipeline_ok": int(ok_df.shape[0]),
        "pipeline_failed": int((result_df["pipeline_ok"] == False).sum()),
        "goalkeeper_decision_counts": ok_df["goalkeeper_decision"].value_counts(dropna=False).to_dict() if len(ok_df) else {},
        "encroachment_decision_counts": ok_df["encroachment_decision"].value_counts(dropna=False).to_dict() if len(ok_df) else {},
    }
    exact_series = ok_df["goalkeeper_exact_match"].dropna() if "goalkeeper_exact_match" in ok_df else pd.Series(dtype=float)
    if len(exact_series):
        summary["goalkeeper_exact_match_rate"] = float(exact_series.mean())

    summary_json = out_dir / f"{args.split}_combined_officiating_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved combined batch CSV to: {result_csv}")


if __name__ == "__main__":
    main()
