"""
Batch encroachment evaluation on the full 49-clip test set.
Uses encroachment_labels_49.csv (no frame_idx_gt — uses auto-kick + fallback=36).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

ENCROACHMENT_CSV = REPO_ROOT / "data" / "meta" / "encroachment_labels_49.csv"
PROBE_SCRIPT     = REPO_ROOT / "scripts" / "pipeline" / "run_player_encroachment_probe.py"
KICK_MODEL       = REPO_ROOT / "runs" / "detect" / "train4" / "weights" / "best.pt"
PLAYER_MODEL     = REPO_ROOT / "models" / "yolov8n.pt"
OUT_DIR          = REPO_ROOT / "runs" / "encroachment_49"
FALLBACK_FRAME   = 36  # ~1.44s into a 4s window clip at 25fps


def run_probe(video_path: Path, out_root: Path, frame_idx: int | None = None):
    base = [
        sys.executable, str(PROBE_SCRIPT),
        "--video-path", str(video_path),
        "--kick-model-path", str(KICK_MODEL),
        "--player-model-path", str(PLAYER_MODEL),
        "--out-root", str(out_root),
        "--temporal-search-radius", "4",
        "--kick-window-start-s", "0.5",
        "--kick-window-end-s", "3.5",
    ]
    if frame_idx is None:
        cmd = base + ["--auto-kick"]
    else:
        cmd = base + ["--frame-idx", str(frame_idx)]

    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(REPO_ROOT), timeout=300)


def load_result(video_path: Path, out_root: Path):
    p = out_root / video_path.stem / "encroachment_result.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ENCROACHMENT_CSV)
    rows = []

    for _, row in df.iterrows():
        clip_name      = str(row["clip_name"])
        encroachment_gt = int(row["encroachment"])
        video_path     = REPO_ROOT / str(row["clips_dir"]) / clip_name

        print(f"\n{'='*60}")
        print(f"Clip: {clip_name}  |  GT encroachment: {encroachment_gt}")

        if not video_path.exists():
            print(f"  [SKIP] Not found: {video_path}")
            rows.append({"clip_name": clip_name, "encroachment_gt": encroachment_gt,
                         "pipeline_ok": False, "error": "video_not_found"})
            continue

        # auto-kick first, fallback to fixed frame
        res = run_probe(video_path, OUT_DIR)
        if res.returncode != 0:
            print(f"  Auto-kick failed (rc={res.returncode}), retrying frame {FALLBACK_FRAME}...")
            res = run_probe(video_path, OUT_DIR, frame_idx=FALLBACK_FRAME)

        if res.returncode != 0:
            print(f"  PIPELINE FAILED\n  {res.stderr[-300:]}")
            rows.append({"clip_name": clip_name, "encroachment_gt": encroachment_gt,
                         "pipeline_ok": False, "error": "pipeline_failed",
                         "stderr": res.stderr[-200:]})
            continue

        payload = load_result(video_path, OUT_DIR)
        if payload is None:
            rows.append({"clip_name": clip_name, "encroachment_gt": encroachment_gt,
                         "pipeline_ok": False, "error": "result_json_missing"})
            continue

        decision       = payload.get("decision", "unknown")
        decision_reason = payload.get("decision_reason", "")
        frame_used     = payload.get("frame_idx", -1)
        kick_source    = payload.get("kick_source", "?")
        candidates     = payload.get("encroachment_candidate_count", 0)
        line_zone      = payload.get("line_zone_player_count", 0)
        kicker_idx     = payload.get("kicker_idx")

        expected = "encroachment" if encroachment_gt == 1 else "no_encroachment"
        correct  = (decision == expected) or (encroachment_gt == 0 and decision == "uncertain")

        print(f"  Frame: {frame_used}  kick_source: {kick_source}")
        print(f"  Decision: {decision} ({decision_reason})")
        print(f"  Candidates: {candidates}  line_zone: {line_zone}  kicker_idx: {kicker_idx}")
        print(f"  Expected: {expected}  ->  {'CORRECT' if correct else 'WRONG'}")

        rows.append({
            "clip_name": clip_name,
            "encroachment_gt": encroachment_gt,
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
        })

    result_df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "encroachment_49_results.csv"
    result_df.to_csv(csv_path, index=False)

    n_ok      = int((result_df.get("pipeline_ok", False) == True).sum())
    n_correct = int(result_df.get("correct", pd.Series(dtype=bool)).sum()) if "correct" in result_df else 0
    n_total   = len(result_df)

    # Per-class breakdown
    tp = int(((result_df["encroachment_gt"] == 1) & (result_df.get("correct", False) == True)).sum()) if "correct" in result_df.columns else 0
    tn = int(((result_df["encroachment_gt"] == 0) & (result_df.get("correct", False) == True)).sum()) if "correct" in result_df.columns else 0

    print(f"\n{'='*60}")
    print(f"BATCH SUMMARY — 49-clip encroachment test")
    print(f"  Total clips    : {n_total}")
    print(f"  Pipeline OK    : {n_ok}")
    print(f"  Correct        : {n_correct} / {n_total}  ({n_correct/n_total:.1%})")
    print(f"  TP (enc=1 ok)  : {tp} / {(result_df['encroachment_gt']==1).sum()}")
    print(f"  TN (enc=0 ok)  : {tn} / {(result_df['encroachment_gt']==0).sum()}")
    if "decision" in result_df.columns:
        print(f"  Decisions      : {result_df['decision'].value_counts(dropna=False).to_dict()}")
    print(f"  Results CSV    : {csv_path}")


if __name__ == "__main__":
    main()
