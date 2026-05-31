"""
Batch runner for 6 encroachment-labeled clips.
Uses encroachment_labels.csv (frame_idx_gt + encroachment GT).
Runs run_player_encroachment_probe.py via --auto-kick;
falls back to GT frame when auto-kick fails.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

CLIPS_DIR = REPO_ROOT / "data" / "clips" / "kick_windows_720p_v2"
ENCROACHMENT_CSV = REPO_ROOT / "data" / "meta" / "encroachment_labels.csv"
PROBE_SCRIPT = REPO_ROOT / "scripts" / "pipeline" / "run_player_encroachment_probe.py"
KICK_MODEL = REPO_ROOT / "runs" / "detect" / "train4" / "weights" / "best.pt"
PLAYER_MODEL = REPO_ROOT / "models" / "yolov8n.pt"  # standard COCO model, person class
OUT_DIR = REPO_ROOT / "runs" / "encroachment_v2"


def stem_to_kick_path(clip_name: str) -> Path:
    """Map 'X.mp4' -> 'X_KICK.mp4' in CLIPS_DIR."""
    stem = Path(clip_name).stem  # strip .mp4
    return CLIPS_DIR / f"{stem}_KICK.mp4"


FALLBACK_FRAME = 36  # ~1.44s into a 4s window clip (25fps); kick is typically around here


def run_probe(video_path: Path, frame_idx: int, out_root: Path, use_auto_kick: bool = True):
    """Run the encroachment probe; returns subprocess result."""
    base_cmd = [
        sys.executable,
        str(PROBE_SCRIPT),
        "--video-path", str(video_path),
        "--kick-model-path", str(KICK_MODEL),
        "--player-model-path", str(PLAYER_MODEL),
        "--out-root", str(out_root),
        "--temporal-search-radius", "4",
        # Cover almost the full 4s window clip (0.5s-3.5s = frames 12-87)
        "--kick-window-start-s", "0.5",
        "--kick-window-end-s", "3.5",
    ]

    if use_auto_kick:
        cmd = base_cmd + ["--auto-kick"]
    else:
        cmd = base_cmd + ["--frame-idx", str(frame_idx)]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=300,
    )
    return result


def load_result(video_path: Path, out_root: Path):
    result_json = out_root / video_path.stem / "encroachment_result.json"
    if result_json.exists():
        return json.loads(result_json.read_text(encoding="utf-8"))
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ENCROACHMENT_CSV)
    rows = []

    for _, row in df.iterrows():
        clip_name = str(row["clip_name"])
        frame_idx_gt = int(row["frame_idx_gt"])
        encroachment_gt = int(row["encroachment"])
        video_path = stem_to_kick_path(clip_name)

        print(f"\n{'='*60}")
        print(f"Clip: {clip_name}")
        print(f"GT frame: {frame_idx_gt}  |  GT encroachment: {encroachment_gt}")

        if not video_path.exists():
            print(f"  [SKIP] Video not found: {video_path}")
            rows.append({
                "clip_name": clip_name,
                "encroachment_gt": encroachment_gt,
                "pipeline_ok": False,
                "error": "video_not_found",
            })
            continue

        # Try auto-kick first; fall back to fixed frame when ball detection fails
        result = run_probe(video_path, frame_idx_gt, OUT_DIR, use_auto_kick=True)
        if result.returncode != 0:
            print(f"  Auto-kick failed (rc={result.returncode}), retrying with fallback frame {FALLBACK_FRAME}...")
            result = run_probe(video_path, FALLBACK_FRAME, OUT_DIR, use_auto_kick=False)

        if result.returncode != 0:
            print(f"  PIPELINE FAILED\n  stderr: {result.stderr[-400:]}")
            rows.append({
                "clip_name": clip_name,
                "encroachment_gt": encroachment_gt,
                "pipeline_ok": False,
                "error": "pipeline_failed",
                "stderr": result.stderr[-200:],
            })
            continue

        payload = load_result(video_path, OUT_DIR)
        if payload is None:
            rows.append({
                "clip_name": clip_name,
                "encroachment_gt": encroachment_gt,
                "pipeline_ok": False,
                "error": "result_json_missing",
            })
            continue

        decision = payload.get("decision", "unknown")
        decision_reason = payload.get("decision_reason", "")
        candidates = payload.get("encroachment_candidate_count", 0)
        frame_used = payload.get("frame_idx", -1)
        kick_source = payload.get("kick_source", "?")
        kicker_idx = payload.get("kicker_idx")
        line_zone = payload.get("line_zone_player_count", 0)

        # Correctness: GT encroachment=1 → expect "encroachment"; GT=0 → expect "no_encroachment"
        expected = "encroachment" if encroachment_gt == 1 else "no_encroachment"
        correct = decision == expected or (encroachment_gt == 0 and decision == "uncertain")

        print(f"  Frame used: {frame_used}  (GT: {frame_idx_gt})  kick_source: {kick_source}")
        print(f"  Decision: {decision} ({decision_reason})")
        print(f"  Candidates: {candidates}  line_zone_players: {line_zone}  kicker_idx: {kicker_idx}")
        print(f"  Expected: {expected}  ->  {'CORRECT' if correct else 'WRONG'}")

        rows.append({
            "clip_name": clip_name,
            "encroachment_gt": encroachment_gt,
            "frame_idx_gt": frame_idx_gt,
            "frame_used": frame_used,
            "kick_source": kick_source,
            "pipeline_ok": True,
            "decision": decision,
            "decision_reason": decision_reason,
            "expected": expected,
            "correct": correct,
            "candidates": candidates,
            "line_zone_players": line_zone,
            "kicker_idx": kicker_idx,
            "overlay_path": payload.get("overlay_path", ""),
        })

    result_df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "encroachment_batch_v2_results.csv"
    result_df.to_csv(csv_path, index=False)

    ok = result_df[result_df.get("pipeline_ok", False) == True] if "pipeline_ok" in result_df.columns else result_df
    correct_count = int(result_df.get("correct", pd.Series(dtype=bool)).sum()) if "correct" in result_df.columns else 0

    print(f"\n{'='*60}")
    print(f"BATCH SUMMARY")
    print(f"  Total clips:   {len(result_df)}")
    print(f"  Pipeline OK:   {int((result_df.get('pipeline_ok', False) == True).sum())}")
    print(f"  Correct:       {correct_count} / {len(result_df)}")
    if "decision" in result_df.columns:
        print(f"  Decisions:     {result_df['decision'].value_counts(dropna=False).to_dict()}")
    print(f"  Results CSV:   {csv_path}")


if __name__ == "__main__":
    main()
