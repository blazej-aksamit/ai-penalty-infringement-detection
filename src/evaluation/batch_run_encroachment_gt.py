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


def load_split_clip_names(splits_csv: Path, split: Optional[str]) -> List[str]:
    splits_df = pd.read_csv(splits_csv)
    if split:
        splits_df = splits_df.loc[splits_df["split"].astype(str) == split].copy()
    return list(splits_df["clip_name"].astype(str).tolist())


def main():
    parser = argparse.ArgumentParser(description="Batch-run encroachment module on GT kick frames.")
    parser.add_argument("--kick-times-csv", default="data/meta/kick_times.csv")
    parser.add_argument("--splits-csv", default="data/meta/splits_violation.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--clips-dir", default="data/clips/penalties_720p")
    parser.add_argument("--encroachment-script", default="scripts/pipeline/run_player_encroachment_probe.py")
    parser.add_argument("--kick-model-path", default="runs/detect/runs/detect/train_yolo26n_gk_ball/weights/best.pt")
    parser.add_argument("--player-model-path", default="yolo26n.pt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--clip-substring", default=None)
    args = parser.parse_args()

    kick_times_csv = resolve_repo_path(args.kick_times_csv)
    splits_csv = resolve_repo_path(args.splits_csv)
    clips_dir = resolve_repo_path(args.clips_dir)
    encroachment_script = resolve_repo_path(args.encroachment_script)
    kick_model_path = resolve_repo_path(args.kick_model_path)
    player_model_path = resolve_repo_path(args.player_model_path)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_window_names = load_split_clip_names(splits_csv, args.split)
    split_full_names = {window_clip_to_full_clip(name): name for name in split_window_names}

    kick_df = pd.read_csv(kick_times_csv).copy()
    kick_df = kick_df.loc[kick_df["clip_name"].astype(str).isin(split_full_names.keys())].copy()

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
                    "window_clip_name": split_full_names.get(full_clip_name),
                    "frame_idx_gt": frame_idx,
                    "pipeline_ok": False,
                    "error": "video_not_found",
                }
            )
            continue

        cmd = [
            sys.executable,
            str(encroachment_script),
            "--video-path",
            str(video_path),
            "--kick-model-path",
            str(kick_model_path),
            "--player-model-path",
            str(player_model_path),
            "--frame-idx",
            str(frame_idx),
            "--out-root",
            str(out_dir),
        ]

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(REPO_ROOT),
            )
            clip_out_dir = out_dir / Path(full_clip_name).stem
            result_json_path = clip_out_dir / "encroachment_result.json"
            result_payload = json.loads(result_json_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "clip_name": full_clip_name,
                    "window_clip_name": split_full_names.get(full_clip_name),
                    "frame_idx_gt": frame_idx,
                    "kick_time_s_gt": float(row.kick_time_s),
                    "pipeline_ok": True,
                    "decision": result_payload.get("decision"),
                    "decision_reason": result_payload.get("decision_reason"),
                    "encroachment_candidate_count": result_payload.get("encroachment_candidate_count"),
                    "goalkeeper_idx": result_payload.get("goalkeeper_idx"),
                    "kicker_idx": result_payload.get("kicker_idx"),
                    "has_goalkeeper_box": result_payload.get("has_goalkeeper_box"),
                    "has_ball_box": result_payload.get("has_ball_box"),
                    "line_candidate_count": result_payload.get("line_candidate_count"),
                    "result_json": str(result_json_path).replace("\\", "/"),
                    "overlay_path": result_payload.get("overlay_path"),
                    "stdout_tail": "\n".join((completed.stdout or "").splitlines()[-8:]),
                }
            )
        except subprocess.CalledProcessError as exc:
            rows.append(
                {
                    "clip_name": full_clip_name,
                    "window_clip_name": split_full_names.get(full_clip_name),
                    "frame_idx_gt": frame_idx,
                    "kick_time_s_gt": float(row.kick_time_s),
                    "pipeline_ok": False,
                    "error": "pipeline_failed",
                    "stderr_tail": "\n".join((exc.stderr or "").splitlines()[-12:]),
                    "stdout_tail": "\n".join((exc.stdout or "").splitlines()[-12:]),
                }
            )

    result_df = pd.DataFrame(rows)
    result_csv = out_dir / f"{args.split}_encroachment_gt_results.csv"
    result_df.to_csv(result_csv, index=False)

    ok_df = result_df.loc[result_df["pipeline_ok"] == True].copy()
    summary = {
        "split": args.split,
        "clips_attempted": int(len(result_df)),
        "pipeline_ok": int(ok_df.shape[0]),
        "pipeline_failed": int((result_df["pipeline_ok"] == False).sum()),
        "decision_counts": ok_df["decision"].value_counts(dropna=False).to_dict() if len(ok_df) else {},
        "candidate_count_mean": float(ok_df["encroachment_candidate_count"].fillna(0).mean()) if len(ok_df) else 0.0,
    }
    summary_json = out_dir / f"{args.split}_encroachment_gt_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved GT batch CSV to: {result_csv}")


if __name__ == "__main__":
    main()
