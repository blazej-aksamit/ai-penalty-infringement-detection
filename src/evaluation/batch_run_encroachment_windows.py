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


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run encroachment module on kick-window clips using a fixed in-window kick frame."
    )
    parser.add_argument("--labels-csv", default="data/meta/keeper_violation_labels_final.csv")
    parser.add_argument("--splits-csv", default="data/meta/splits_violation.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--encroachment-script", default="scripts/pipeline/run_player_encroachment_probe.py")
    parser.add_argument("--kick-model-path", default="runs/detect/train4/weights/best.pt")
    parser.add_argument("--player-model-path", default="models/yolov8n.pt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--kick-time-in-window-s", type=float, default=1.5)
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

    encroachment_script = resolve_repo_path(args.encroachment_script)
    kick_model_path = resolve_repo_path(args.kick_model_path)
    player_model_path = resolve_repo_path(args.player_model_path)

    rows: List[Dict[str, object]] = []
    for row in merged.itertuples(index=False):
        clip_name = str(row.clip_name)
        window_file = resolve_repo_path(str(row.window_file))
        fps = float(row.fps)
        frame_idx = int(round(float(args.kick_time_in_window_s) * fps))

        cmd = [
            sys.executable,
            str(encroachment_script),
            "--video-path",
            str(window_file),
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
            clip_out_dir = out_dir / Path(clip_name).stem
            result_json_path = clip_out_dir / "encroachment_result.json"
            result_payload = json.loads(result_json_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "clip_name": clip_name,
                    "window_file": str(window_file).replace("\\", "/"),
                    "split": args.split,
                    "frame_idx": frame_idx,
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
                    "clip_name": clip_name,
                    "window_file": str(window_file).replace("\\", "/"),
                    "split": args.split,
                    "frame_idx": frame_idx,
                    "pipeline_ok": False,
                    "error": "pipeline_failed",
                    "stderr_tail": "\n".join((exc.stderr or "").splitlines()[-12:]),
                    "stdout_tail": "\n".join((exc.stdout or "").splitlines()[-12:]),
                }
            )

    result_df = pd.DataFrame(rows)
    result_csv = out_dir / f"{args.split}_encroachment_window_results.csv"
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
    summary_json = out_dir / f"{args.split}_encroachment_window_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved window batch CSV to: {result_csv}")


if __name__ == "__main__":
    main()
